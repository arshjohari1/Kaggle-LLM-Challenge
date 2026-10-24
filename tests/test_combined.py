import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ntl import combined
from ntl.combined import ordered_scores, rank_blend


def test_rank_blend_is_scale_independent_and_breaks_ties_by_cpu():
    cpu = np.array([[9., 4., 1.], [1., 1., 1.]])
    qwen = np.array([[1., 4., 9.], [0., 1., 3.]])
    assert rank_blend(cpu, qwen).tolist() == [0, 2]
    np.testing.assert_array_equal(rank_blend(cpu, qwen), rank_blend(cpu * 100 + 27, qwen * 3 - 9))
    with pytest.raises(ValueError):
        rank_blend(cpu, qwen[:, :2])
    with pytest.raises(ValueError):
        rank_blend(cpu, qwen * np.nan)


def test_class_order_alignment():
    class Head:
        classes_ = np.array(['b', 'a', 'c'])
        def decision_function(self, values):
            return np.array([[2, 5, 1]])
    np.testing.assert_array_equal(ordered_scores(Head(), None, ['a', 'b', 'c']), [[5, 2, 1]])
    with pytest.raises(ValueError):
        ordered_scores(Head(), None, ['a', 'c'])


def test_standalone_combined_matches_tested_rule():
    root = Path(__file__).resolve().parents[1]
    script = (root / 'notebooks/kaggle_combined.py').read_text()
    notebook = json.loads((root / 'notebooks/kaggle_combined.ipynb').read_text())
    assert script == ''.join(notebook['cells'][1]['source'])
    assert 'from ntl.' not in script
    namespace = {}
    exec(compile(script.rsplit('\nrun_combined()', 1)[0], 'combined', 'exec'), namespace)
    rng = np.random.default_rng(42)
    cpu, qwen = rng.normal(size=(100, 18)), rng.normal(size=(100, 18))
    np.testing.assert_array_equal(namespace['rank_blend'](cpu, qwen), rank_blend(cpu, qwen))


def test_complete_orchestration_with_synthetic_features(monkeypatch, tmp_path):
    data = tmp_path / 'name-that-llm'
    data.mkdir()
    groups = np.repeat(np.arange(5), 18)
    train = pd.DataFrame({'id': [str(i) for i in range(90)], 'prompt_id': groups.astype(str),
                          'model': np.tile([f'author-{i:02d}' for i in range(18)], 5),
                          'text': [f'Passage by writer{i % 18} explores topic{i // 18}. A sentence, with details!' for i in range(90)]})
    test = pd.DataFrame({'id': [f'{i:08d}' for i in range(4000)], 'text': ['A new paragraph with details.'] * 4000})
    train.to_csv(data / 'train.csv', index=False)
    test.to_csv(data / 'test.csv', index=False)
    monkeypatch.setattr(combined, 'official_folds', lambda *args: groups)
    monkeypatch.setattr(combined, 'discover_model', lambda *args: data)
    monkeypatch.setattr(combined, 'load_frozen', lambda *args, **kwargs: (None, None, {'purpose': 'synthetic test'}))
    monkeypatch.setattr(combined, 'extract_frozen', lambda frame, *args: (
        np.random.default_rng(42).normal(size=(len(frame), 4, 18)).astype(np.float32), {'rows': len(frame)}))
    result = combined.run_combined(tmp_path, tmp_path / 'working', allow_cpu=True)
    saved = pd.read_parquet(tmp_path / 'working/submission.parquet')
    assert saved.id.tolist() == test.id.tolist()
    assert len(saved) == 4000 and saved.model.isin(train.model).all()
    assert len(result['validation']['folds']) == 5
    assert result['historical_qwen_correct_on_cpu_errors'] == 417
    assert (tmp_path / 'working/combined-v1/summary.json').is_file()
