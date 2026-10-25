import json
from pathlib import Path

import numpy as np
import pytest

from ntl.qwen8 import QWEN8_SHARDS, discover_qwen8, gpu_memory_plan, pilot_projection, validate_device_map, validate_shard_index


def test_model_identity_and_shard_index(tmp_path):
    path = tmp_path / '8b-base'
    path.mkdir()
    config = {'model_type': 'qwen3', 'hidden_size': 4096, 'num_hidden_layers': 36}
    (path / 'config.json').write_text(json.dumps(config))
    assert discover_qwen8(tmp_path) == path
    index = {'weight_map': {f'layer{i}': name for i, name in enumerate(QWEN8_SHARDS)}}
    (path / 'model.safetensors.index.json').write_text(json.dumps(index))
    validate_shard_index(path)
    index['weight_map']['rogue'] = '../untrusted.safetensors'
    (path / 'model.safetensors.index.json').write_text(json.dumps(index))
    with pytest.raises(ValueError):
        validate_shard_index(path)
    config['num_hidden_layers'] = 32
    (path / 'config.json').write_text(json.dumps(config))
    with pytest.raises(RuntimeError):
        discover_qwen8(tmp_path)


def test_two_gpu_memory_reserve_and_offload_rejection():
    plan = gpu_memory_plan([15 * 2**30, 14 * 2**30])
    assert plan == {0: 12 * 2**30, 1: 11 * 2**30}
    for available in ([16 * 2**30], [15 * 2**30, 10 * 2**30]):
        with pytest.raises(RuntimeError):
            gpu_memory_plan(available)
    validate_device_map({'embed': 0, 'layer': 1})
    for placement in ({'embed': 0, 'layer': 'cpu'}, {'layer': 'disk'}, {}):
        with pytest.raises(RuntimeError):
            validate_device_map(placement)


def test_pilot_projection_preserves_hidden_test_budget():
    assert pilot_projection(60, 16, 16, 2699) == 60 + 2699 + 4000 + 1800
    assert pilot_projection(60, 1, 16, 2699) < 7200
    assert pilot_projection(60, 32, 16, 2699) > 7200


def test_qwen8_standalone_contains_tested_runtime():
    root = Path(__file__).resolve().parents[1]
    code = (root / 'notebooks/kaggle_qwen8.py').read_text()
    notebook = json.loads((root / 'notebooks/kaggle_qwen8.ipynb').read_text())
    assert code == ''.join(notebook['cells'][1]['source']) and 'from ntl.' not in code
    namespace = {}
    exec(compile(code.rsplit('\n_ = run_qwen8()', 1)[0], 'qwen8', 'exec'), namespace)
    assert namespace['QWEN8_SHARDS'] == QWEN8_SHARDS
    assert namespace['gpu_memory_plan']([15 * 2**30] * 2) == {0: 12 * 2**30, 1: 12 * 2**30}
    assert namespace['run_qwen8'].__globals__['run_combined'] is namespace['run_combined']
