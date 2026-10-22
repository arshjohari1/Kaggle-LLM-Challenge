import json
import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from ntl.cli import predict_frame, validate_submission
from ntl.data import LABELS, groups
from ntl.ensemble import fit_grouped
from ntl.models import CANDIDATES, build_model


@pytest.mark.parametrize("candidate", CANDIDATES)
def test_model_roundtrip_and_fold_local_vocabulary(candidate, passages, tmp_path):
    train = passages[passages.prompt_id != "9"]
    heldout = passages[passages.prompt_id == "9"].copy()
    heldout["text"] = heldout.text + " zzzuniqueheldoutcanaryzzz "
    model = build_model(candidate)
    with threadpool_limits(limits=1):
        fit_grouped(model, train.text, train.model, groups(train))
        predictions = model.predict(heldout.text)
    assert set(predictions) <= set(LABELS)
    features = model.named_steps["features"] if hasattr(model, "named_steps") else None
    vectorizers = ([v for _, v in features.transformer_list if hasattr(v, "vocabulary_")]
                   if hasattr(features, "transformer_list") else
                   ([features] if hasattr(features, "vocabulary_") else []))
    for vectorizer in vectorizers:
        assert not any("uniqueheldoutcanary" in term for term in vectorizer.vocabulary_)
        assert "zzz" not in vectorizer.vocabulary_
    if candidate == "style_svm":
        style = dict(features.transformer_list)["style"]
        expected_mean = style.named_steps["extract"].transform(train.text).mean(axis=0)
        np.testing.assert_allclose(style.named_steps["scale"].mean_, expected_mean)
    if candidate == "style_rbf":
        expected_mean = features.named_steps["extract"].transform(train.text).mean(axis=0)
        np.testing.assert_allclose(features.named_steps["scale"].mean_, expected_mean)
    if candidate == "stacked":
        assert (model.inner_fold_ >= 0).all()
        for p in train.prompt_id.unique():
            assert len(set(model.inner_fold_[train.prompt_id.to_numpy() == p])) == 1
        for base in model.models_:
            parts = base.named_steps["features"]
            if hasattr(parts, "transformer_list"):
                for _, part in parts.transformer_list:
                    if hasattr(part, "vocabulary_"):
                        assert "zzz" not in part.vocabulary_
    file = tmp_path / "model.joblib"
    joblib.dump({"pipeline": model, "labels": LABELS}, file)
    loaded = joblib.load(file)
    output = predict_frame(loaded, heldout, batch_size=7)
    assert output.model.tolist() == list(predictions)
    output.to_parquet(tmp_path / "submission.parquet", index=False)
    validate_submission(pd.read_parquet(tmp_path / "submission.parquet"), heldout)


def test_submission_rejects_missing_duplicate_invalid(passages):
    test = passages[["id", "text"]].iloc[:3]
    good = pd.DataFrame({"id": test.id, "model": [LABELS[0]] * 3})
    validate_submission(good, test)
    for bad in [good.iloc[:2], good.assign(id=[test.id.iloc[0]] * 3), good.assign(model="invalid")]:
        with pytest.raises(ValueError):
            validate_submission(bad, test)


def test_cli_end_to_end(passages, tmp_path):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    passages.to_csv(train, index=False)
    passages[["id", "text"]].iloc[:8].to_csv(test, index=False)
    def run(*args):
        return subprocess.run([sys.executable, "-m", "ntl.cli", *map(str, args)],
                              check=True, capture_output=True, text=True,
                              env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    run("prepare", "--train", train, "--out", tmp_path / "prepared")
    run("evaluate", "--train", train, "--folds", tmp_path / "prepared/folds.csv",
        "--out", tmp_path / "eval", "--candidates", "word_lr", "--threads", 1)
    metrics = json.loads((tmp_path / "eval/word_lr/metrics.json").read_text())
    assert len(metrics["folds"]) == 5
    assert metrics["oof_balanced_accuracy"] > 0.9  # Deliberately easy synthetic author markers.
    oof = pd.read_csv(tmp_path / "eval/word_lr/oof.csv")
    assert len(oof) == len(passages) and not oof.prediction.isna().any()
    run("train", "--train", train, "--candidate", "word_lr", "--out", tmp_path / "artifact")
    run("predict", "--model", tmp_path / "artifact/model.joblib", "--test", test,
        "--out", tmp_path / "submission.parquet")
    run("benchmark", "--model", tmp_path / "artifact/model.joblib", "--test", test,
        "--rows", 4000, "--out", tmp_path / "benchmark.json")
    assert json.loads((tmp_path / "benchmark.json").read_text())["rows"] == 4000
    (tmp_path / "artifact/model.joblib").write_bytes(b"corrupt model must be rejected before unpickling")
    with pytest.raises(subprocess.CalledProcessError) as caught:
        run("predict", "--model", tmp_path / "artifact/model.joblib", "--test", test,
            "--out", tmp_path / "tampered.parquet")
    assert "checksum" in caught.value.stderr
    # Same IDs with altered data cannot silently reuse an old split version.
    passages.loc[0, "text"] += " changed"
    passages.to_csv(train, index=False)
    with pytest.raises(subprocess.CalledProcessError):
        run("evaluate", "--train", train, "--folds", tmp_path / "prepared/folds.csv",
            "--out", tmp_path / "stale")


def test_numeric_submission_dtype(passages, tmp_path):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    passages.to_csv(train, index=False)
    passages.iloc[:4][["id", "text"]].assign(id=[0, 1, 2, 3]).to_csv(test, index=False)
    def run(*args):
        return subprocess.run([sys.executable, "-m", "ntl.cli", *map(str, args)],
                              check=True, capture_output=True, text=True)
    run("train", "--train", train, "--candidate", "word_lr", "--out", tmp_path / "model")
    run("predict", "--model", tmp_path / "model/model.joblib", "--test", test,
        "--out", tmp_path / "numeric.parquet", "--id-dtype", "int64")
    out = pd.read_parquet(tmp_path / "numeric.parquet")
    assert out.id.dtype == np.dtype("int64")
    df = pd.read_csv(test, dtype={"id": str})
    df.loc[0, "id"] = "00001"
    df.to_csv(test, index=False)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        run("predict", "--model", tmp_path / "model/model.joblib", "--test", test,
            "--out", tmp_path / "invalid-conversion.parquet", "--id-dtype", "int64")
    assert "not lossless" in caught.value.stderr
