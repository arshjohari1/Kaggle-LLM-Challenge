import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import hashlib
import sys
from types import SimpleNamespace

from ntl.deberta import best_epoch, check_deadline, ensure_sentencepiece, inner_prompt_split, prepare_deberta


def test_inner_epoch_selection_excludes_outer_prompts():
    frame = pd.DataFrame({'prompt_id': np.repeat([f'p{i}' for i in range(15)], 3),
                          'model': np.tile(['a', 'b', 'c'], 15)})
    outer_validation = np.arange(9)
    outer_training = np.arange(9, len(frame))
    fit, selection = inner_prompt_split(frame, outer_training, 42)
    assert set(fit) | set(selection) == set(outer_training)
    assert not set(fit) & set(selection)
    assert not set(outer_validation) & (set(fit) | set(selection))
    assert not set(frame.prompt_id.iloc[fit]) & set(frame.prompt_id.iloc[selection])
    assert not set(frame.prompt_id.iloc[outer_validation]) & set(frame.prompt_id.iloc[selection])
    assert np.array_equal(selection, inner_prompt_split(frame, outer_training, 42)[1])


def test_epoch_selection_prefers_earlier_tie_and_budget_stops():
    assert best_epoch([{'epoch': 1, 'balanced_accuracy': .4}, {'epoch': 2, 'balanced_accuracy': .5},
                       {'epoch': 3, 'balanced_accuracy': .5}]) == 2
    with pytest.raises(TimeoutError):
        check_deadline(0)


def test_deberta_standalone_embeds_tested_source():
    root = Path(__file__).resolve().parents[1]
    code = (root / 'notebooks/kaggle_deberta.py').read_text()
    nb = json.loads((root / 'notebooks/kaggle_deberta.ipynb').read_text())
    assert code == ''.join(nb['cells'][1]['source'])
    assert 'from ntl.' not in code
    source = (root / 'src/ntl/deberta.py').read_text()
    for function in ('inner_prompt_split', 'train_author', 'predict_author'):
        import ast
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == function)
        assert ast.get_source_segment(source, node) in code
    compile(code, 'deberta', 'exec')


def test_official_download_hashes_and_corrupt_cache_rejection(monkeypatch, tmp_path):
    resources = {'config.json': json.dumps({'model_type': 'deberta-v2', 'hidden_size': 1024,
                                           'num_hidden_layers': 24}).encode(),
                 'pytorch_model.bin': b'fake weight bytes for checksum protocol only', 'spm.model': b'tokenizer bytes'}
    entries = []
    for name, data in resources.items():
        entries.append(SimpleNamespace(rfilename=name,
            lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()) if name.endswith('.bin') else None,
            blob_id=hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()))
    class API:
        def model_info(self, repo, files_metadata):
            assert repo == 'microsoft/deberta-v3-large' and files_metadata
            return SimpleNamespace(sha='a' * 40, card_data=SimpleNamespace(license='mit'), siblings=entries)
    def snapshot(repo, revision, local_dir, allow_patterns):
        assert revision == 'a' * 40
        for name in allow_patterns:
            (Path(local_dir) / name).write_bytes(resources[name])
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(HfApi=API, snapshot_download=snapshot))
    path, receipt = prepare_deberta(tmp_path)
    assert receipt['revision'] == 'a' * 40 and receipt['weight_file'] == 'pytorch_model.bin'
    assert prepare_deberta(tmp_path)[1] == receipt
    (path / 'pytorch_model.bin').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='checksum mismatch'):
        prepare_deberta(tmp_path)


def test_offline_tokenizer_wheel_is_verified_and_installed_without_network(monkeypatch, tmp_path):
    import importlib.util
    import subprocess
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: None)
    name = 'sentencepiece-0.2.1-test.whl'
    wheel = tmp_path / name
    wheel.write_bytes(b'wheel integrity protocol fixture; not executed')
    receipt = {'filename': name, 'sha256': hashlib.sha256(wheel.read_bytes()).hexdigest()}
    (tmp_path / 'sentencepiece-wheel.json').write_text(json.dumps(receipt))
    calls = []
    monkeypatch.setattr(subprocess, 'run', lambda args, check: calls.append(args))
    ensure_sentencepiece(tmp_path)
    assert len(calls) == 1 and '--no-index' in calls[0] and '--no-deps' in calls[0]
    wheel.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        ensure_sentencepiece(tmp_path)
    assert len(calls) == 1
