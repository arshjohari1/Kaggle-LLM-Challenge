"""Fold-local training and offline output contracts; no torch dependency here."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ntl import generative as g


@pytest.mark.parametrize("folder", ["competitions/name-that-llm", "name-that-llm",
                                    "datasets/user/custom-data", "renamed-input/nested/files"])
def test_data_discovery_supports_kaggle_mount_layouts(tmp_path, folder):
    data = tmp_path / folder
    data.mkdir(parents=True)
    (data / "train.csv").touch()
    (data / "test.csv").touch()
    assert g.discover_data(tmp_path) == data


@pytest.mark.parametrize("filename", [None, "train.csv"])
def test_missing_data_reports_visible_files_and_attachment_steps(tmp_path, filename):
    if filename:
        (tmp_path / filename).touch()
    with pytest.raises(RuntimeError, match="Add Input → Competitions") as error:
        g.discover_data(tmp_path)
    assert (filename or "none") in str(error.value)


def test_data_discovery_prefers_official_input_and_rejects_ambiguous_fallback(tmp_path):
    for folder in ("copy-a", "copy-b"):
        data = tmp_path / folder
        data.mkdir()
        (data / "train.csv").touch()
        (data / "test.csv").touch()
    with pytest.raises(RuntimeError, match="Multiple"):
        g.discover_data(tmp_path)
    official = tmp_path / "competitions/name-that-llm"
    official.mkdir(parents=True)
    (official / "train.csv").touch()
    (official / "test.csv").touch()
    assert g.discover_data(tmp_path) == official


def test_submission_preserves_ids_and_rejects_invalid_labels(tmp_path):
    frame = pd.DataFrame({"id": ["0001", "0020"], "text": ["a", "b"]})
    path = g.write_submission(frame, ["model-a", "model-b"], ["model-a", "model-b"], tmp_path)
    assert pd.read_parquet(path).id.tolist() == ["0001", "0020"]
    with pytest.raises(ValueError, match="invalid"):
        g.write_submission(frame, ["wrong", "wrong"], ["model-a"], tmp_path)
    with pytest.raises(ValueError, match="Duplicate"):
        g.check_frame(pd.concat([frame, frame]), ["id", "text"])


def test_original_folds_reject_changed_dataset(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        g.official_folds(pd.DataFrame(), path)


def test_model_discovery_rejects_wrong_variant_and_ambiguity(tmp_path):
    for name, layers in [("1.7b-base", 28), ("1.7b-instruct", 28), ("8b-base", 32)]:
        target = tmp_path / name
        target.mkdir()
        (target / "config.json").write_text(json.dumps({"model_type": "qwen3", "hidden_size": 2048, "num_hidden_layers": layers}))
    assert g.discover_model(tmp_path) == tmp_path / "1.7b-base"
    duplicate = tmp_path / "duplicate-1.7b-base"
    duplicate.mkdir()
    (duplicate / "config.json").write_text((tmp_path / "1.7b-base/config.json").read_text())
    with pytest.raises(RuntimeError, match="exactly one"):
        g.discover_model(tmp_path)


def test_heads_fit_scaling_only_on_training_prompts(monkeypatch, tmp_path):
    rng = np.random.default_rng(42)
    # Five prompts, each containing all 18 authors, with deliberately shifted means.
    groups = np.repeat(np.arange(5), 18)
    frame = pd.DataFrame({"id": [str(i) for i in range(90)], "prompt_id": groups.astype(str),
                          "model": np.tile([f"author-{i:02d}" for i in range(18)], 5), "text": ["short text"] * 90})
    features = rng.normal(size=(90, 4, 6)).astype(np.float32) + groups[:, None, None] * 100
    fitted_means = []
    original = g.StandardScaler.fit

    def observe(self, x, y=None, **kwargs):
        fitted_means.append(x.mean(axis=0))
        return original(self, x, y, **kwargs)

    monkeypatch.setattr(g.StandardScaler, "fit", observe)
    best, scores = g.evaluate_heads(features, frame, groups, tmp_path)
    assert len(scores) == 8 and len(fitted_means) == 40
    for candidate in range(8):
        for fold in range(5):
            np.testing.assert_allclose(fitted_means[candidate * 5 + fold], features[groups != fold, candidate // 2].mean(axis=0))
    assert best["candidate"] in {row["candidate"] for row in scores}
    assert len(pd.read_csv(tmp_path / "comparison.csv")) == 8
    assert len(pd.read_csv(tmp_path / best["candidate"] / "oof.csv")) == 90
    assert np.load(tmp_path / best["candidate"] / "decision_scores.npy").shape == (90, 18)


def test_generated_script_and_notebook_embed_tested_source():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/ntl/generative.py").read_text()
    script = (root / "notebooks/kaggle_qwen_frozen.py").read_text()
    notebook = json.loads((root / "notebooks/kaggle_qwen_frozen.ipynb").read_text())
    assert source in script
    assert script == "".join(notebook["cells"][1]["source"])
    compile(script, "kaggle_qwen_frozen.py", "exec")
    assert "from ntl" not in script and "pip install" not in script
