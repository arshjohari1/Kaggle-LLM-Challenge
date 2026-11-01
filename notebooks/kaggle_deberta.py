import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

TRAIN_SHA256 = "40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
FOLD_SHA256 = "d019f60a2cfaab61dcbb73f64a8c126b0871ff037c23142ef45b81c4cdab6804"
FOLD_PATTERN = [4,4,0,1,4,4,1,3,2,4,0,4,2,3,2,2,4,2,2,0,4,3,1,4,2,3,3,3,2,1,0,1,0,2,1,3,3,2,3,1,3,1,0,4,4,4,4,4,4,2,2,2,0,3,0,4,1,4,2,4,0,1,0,4,2,3,3,1,4,3,0,0,4,1,1,4,0,0,4,3,4,1,0,3,1,2,3,0,0,1,3,0,3,0,0,2,1,0,1,3,3,4,4,2,2,2,1,1,2,0,1,1,2,1,2,3,4,3,3,4,2,2,0,1,2,0,1,2,4,1,1,0,0,2,4,0,0,3,3,0,3,3,1,3,0,1,1,2,2,3]

def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()

def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)

def official_folds(train, train_path):
    if sha256_file(train_path) != TRAIN_SHA256:
        raise ValueError("Training data changed; regenerate and review fixed folds before comparing scores.")
    prompts = sorted(train.prompt_id.unique())
    if len(prompts) != len(FOLD_PATTERN):
        raise ValueError("Unexpected prompt count")
    folds = train.prompt_id.map(dict(zip(prompts, FOLD_PATTERN))).to_numpy(dtype=int)
    serialized = pd.DataFrame({"id": train.id, "fold": folds}).to_csv(index=False, lineterminator="\n")
    if hashlib.sha256(serialized.encode()).hexdigest() != FOLD_SHA256:
        raise ValueError("Fold reconstruction differs from the original baseline")
    # Enforce exact duplicate isolation, including duplicates linked across prompts.
    if pd.DataFrame({"text": train.text, "fold": folds}).groupby("text").fold.nunique().max() != 1:
        raise ValueError("Duplicate text crosses validation folds")
    return folds

def check_frame(frame, columns):
    if frame.empty or not set(columns) <= set(frame.columns):
        raise ValueError("Empty input or missing required columns")
    if frame.id.duplicated().any():
        raise ValueError("Duplicate passage IDs")
    for column in columns:
        if not frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"Invalid or blank {column}")

def discover_data(input_root):
    """Prefer official mounts; find an unambiguous train/test pair elsewhere."""
    root = Path(input_root)
    for path in (root / "competitions/name-that-llm", root / "name-that-llm"):
        if (path / "train.csv").is_file() and (path / "test.csv").is_file():
            return path
    matches = sorted({path.parent for path in root.rglob("train.csv")
                      if (path.parent / "test.csv").is_file()})
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError("Multiple train.csv/test.csv pairs found. Remove duplicate data inputs so the intended test file is unambiguous: "
                           + ", ".join(str(path) for path in matches))
    visible = sorted(str(path.relative_to(root)) for path in root.rglob("*.csv"))[:20]
    raise RuntimeError("Attach the Name That LLM competition data using Add Input → Competitions → Name That LLM → Add. "
                       "The notebook needs train.csv and test.csv together. Visible CSV files: "
                       + (", ".join(visible) if visible else "none"))

"""Prompt-grouped DeBERTa fine-tuning experiment. Torch imports are lazy."""
import gc
import hashlib
import importlib.metadata
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score



DEBERTA_REPO = 'microsoft/deberta-v3-large'
DEBERTA_EXPERIMENT = 'deberta-v2'
SETTINGS = {'seed': 42, 'max_length': 512, 'batch_size': 2, 'accumulation': 8,
            'max_epochs': 3, 'patience': 1, 'encoder_lr': 1e-5, 'head_lr': 1e-4,
            'weight_decay': 0.01, 'warmup_fraction': 0.1, 'budget_seconds': 6 * 3600}


def ensure_sentencepiece(input_root='/kaggle/input'):
    import importlib.util
    import subprocess
    import sys

    if importlib.util.find_spec('sentencepiece') is not None:
        return
    receipts = list(Path(input_root).rglob('sentencepiece-wheel.json'))
    if len(receipts) != 1:
        raise RuntimeError('SentencePiece is missing. With Internet On, run %pip install sentencepiece==0.2.1 in a separate cell and restart. For offline training attach the output of kaggle_deberta_prepare.py including its wheel receipt.')
    receipt = json.loads(receipts[0].read_text())
    name = receipt['filename']
    if Path(name).name != name or not name.startswith('sentencepiece-0.2.1-') or not name.endswith('.whl'):
        raise ValueError('Unexpected prepared tokenizer wheel')
    wheel = receipts[0].parent / name
    if sha256_file(wheel) != receipt['sha256']:
        raise ValueError('Prepared tokenizer wheel failed checksum verification')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', str(wheel)], check=True)
    importlib.invalidate_caches()


def inner_prompt_split(frame, outer_training_indices, seed):
    """Select 20% of training prompts; no outer validation rows enter selection."""
    indices = np.asarray(outer_training_indices, dtype=int)
    prompts = np.array(sorted(frame.prompt_id.iloc[indices].unique()))
    if len(prompts) < 5:
        raise ValueError('Need at least five training prompts for epoch selection')
    selected = set(np.random.default_rng(seed).permutation(prompts)[:max(1, math.ceil(len(prompts) * .2))])
    mask = frame.prompt_id.iloc[indices].isin(selected).to_numpy()
    fitting, selection = indices[~mask], indices[mask]
    if set(frame.model.iloc[fitting]) != set(frame.model.iloc[selection]):
        raise ValueError('Inner partitions must contain all author classes')
    return fitting, selection


def best_epoch(history):
    if not history:
        raise ValueError('No epoch selection results')
    return int(max(history, key=lambda row: (row['balanced_accuracy'], -row['epoch']))['epoch'])


def representative_pilot_indices(targets, seed, rows=128):
    targets = np.asarray(targets)
    if rows < 1 or len(targets) == 0:
        raise ValueError('Pilot sample must contain rows')
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(targets), min(rows, len(targets)), replace=False).tolist()
    present = set(targets[chosen])
    for label in sorted(set(targets) - present):
        chosen.append(int(rng.choice(np.flatnonzero(targets == label))))
    return np.asarray(chosen, dtype=int)


def projected_training_seconds(elapsed, measured_seconds, sampled_rows, workload_rows):
    if sampled_rows <= 0 or measured_seconds <= 0 or workload_rows < 0:
        raise ValueError('Invalid training pilot measurement')
    return elapsed + measured_seconds / sampled_rows * workload_rows * 1.5 + 20 * 60


def validate_recovery_epochs(epochs, folds, max_epochs):
    if epochs is None:
        return None
    if sorted(set(folds)) != list(range(5)) or len(epochs) != 5:
        raise ValueError('Recovery requires one epoch count for each original fold 0–4')
    if any(type(value) is not int or not 1 <= value <= max_epochs for value in epochs):
        raise ValueError('Invalid original inner-selected epoch counts')
    return list(epochs)


def bundle_deberta_reports(out, archive):
    import zipfile

    names = ('pilot.json', 'summary.json', 'manifest.json', 'tokenization.json',
             'oof.csv', 'oof_scores.npz', 'per_class.csv', 'confusion.csv')
    receipt = {name: {'sha256': sha256_file(out / name), 'bytes': (out / name).stat().st_size} for name in names}
    write_json(out / 'report_files.json', receipt)
    pending = archive.with_suffix('.zip.part')
    with zipfile.ZipFile(pending, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in (*names, 'report_files.json'):
            bundle.write(out / name, arcname=name)
    with zipfile.ZipFile(pending) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('Report archive failed its readback check')
        for name, info in receipt.items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != info['sha256']:
                raise ValueError('Report archive checksum mismatch')
    pending.replace(archive)
    return receipt


def check_deadline(deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError('Fine-tuning budget reached. Completed folds are retained; stop the GPU session and share progress.')


def prepare_deberta(destination):
    """Official free checkpoint, resolved to one revision, hashes checked before use."""
    from huggingface_hub import HfApi, snapshot_download

    destination = Path(destination)
    receipt = destination / 'verified.json'
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if record.get('repo') != DEBERTA_REPO or record.get('license') not in ('mit', 'apache-2.0'):
            raise ValueError('Unexpected pretrained source/license in cached receipt')
        for name, digest in record['file_sha256'].items():
            if Path(name).name != name:
                raise ValueError('Unexpected path in pretrained receipt')
            if sha256_file(destination / name) != digest:
                raise ValueError('Cached pretrained resource checksum mismatch: ' + name)
        return destination, record
    print('Resolving official Microsoft checkpoint and license; Internet is needed for this first download.', flush=True)
    info = HfApi().model_info(DEBERTA_REPO, files_metadata=True)
    license_name = getattr(info.card_data, 'license', None)
    if license_name not in ('mit', 'apache-2.0'):
        raise ValueError('Review the official model license before use: ' + str(license_name))
    siblings = {f.rfilename: f for f in info.siblings}
    weight = 'model.safetensors' if 'model.safetensors' in siblings else 'pytorch_model.bin'
    if weight not in siblings:
        raise ValueError('Expected a single official checkpoint file')
    names = [name for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json',
                               'special_tokens_map.json', 'spm.model', weight) if name in siblings]
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(DEBERTA_REPO, revision=info.sha, local_dir=destination, allow_patterns=names)
    hashes = {}
    for name in names:
        data_path = destination / name
        hashes[name] = sha256_file(data_path)
        meta = siblings[name]
        lfs = meta.lfs
        expected = (lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)) if lfs else None
        if expected:
            if hashes[name] != expected:
                raise ValueError('Official resource SHA-256 mismatch: ' + name)
        elif meta.blob_id:
            data = data_path.read_bytes()
            if hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() != meta.blob_id:
                raise ValueError('Official resource Git blob mismatch: ' + name)
        else:
            raise ValueError('Missing authoritative checksum: ' + name)
    config = json.loads((destination / 'config.json').read_text())
    if config.get('model_type') != 'deberta-v2' or config.get('hidden_size') != 1024 or config.get('num_hidden_layers') != 24:
        raise ValueError('Official checkpoint differs from the expected DeBERTa-v3-large architecture')
    record = {'repo': DEBERTA_REPO, 'revision': info.sha, 'license': license_name,
              'file_sha256': hashes, 'weight_file': weight, 'source': 'Official Hugging Face repository; no third-party fine-tuned data'}
    write_json(receipt, record)
    return destination, record


class OriginalSentencePiece:
    """Single-passage adapter to the original model, independent of HF fast regexes."""
    def __init__(self, model_dir):
        import sentencepiece

        root = Path(model_dir)
        settings = json.loads((root / 'tokenizer_config.json').read_text()) if (root / 'tokenizer_config.json').exists() else {}
        if settings.get('do_lower_case', False) or settings.get('split_by_punct', False):
            raise ValueError('Unexpected normalization settings; review the original tokenizer.')
        self.spm = sentencepiece.SentencePieceProcessor(model_file=str(root / 'spm.model'))
        self.special = {}
        for token in ('[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'):
            index = self.spm.piece_to_id(token)
            if self.spm.id_to_piece(index) == token:
                self.special[token] = index
        for key, value in settings.get('added_tokens_decoder', {}).items():
            token = value['content'] if isinstance(value, dict) else value
            self.special[token] = int(key)
        if '[MASK]' not in self.special:
            self.special['[MASK]'] = self.spm.get_piece_size()
        if not {'[PAD]', '[UNK]', '[CLS]', '[SEP]'} <= set(self.special):
            raise ValueError('Original DeBERTa SentencePiece special IDs are missing')
        self.pad_token_id = self.special['[PAD]']
        self.special_pattern = re.compile('(' + '|'.join(re.escape(s) for s in sorted(self.special, key=len, reverse=True)) + ')')

    def __call__(self, texts, padding=False, truncation=False, max_length=None):
        if padding:
            raise ValueError('Use dynamic batch padding via pad()')
        result = []
        for text in texts:
            ids = []
            for part in self.special_pattern.split(text):
                if part in self.special:
                    ids.append(self.special[part])
                elif part:
                    ids.extend(self.spm.piece_to_id(piece) for piece in self.spm.encode(part, out_type=str))
            if truncation:
                if max_length is None or max_length < 2:
                    raise ValueError('Invalid SentencePiece truncation limit')
                ids = ids[:max_length - 2]
            result.append([self.special['[CLS]']] + ids + [self.special['[SEP]']])
        return {'input_ids': result, 'attention_mask': [[1] * len(ids) for ids in result]}

    def pad(self, rows, padding=True, return_tensors='pt'):
        import torch

        if not padding or return_tensors != 'pt':
            raise ValueError('Expected padded PyTorch batches')
        width = max(len(row['input_ids']) for row in rows)
        return {'input_ids': torch.tensor([row['input_ids'] + [self.pad_token_id] * (width - len(row['input_ids'])) for row in rows], dtype=torch.long),
                'attention_mask': torch.tensor([row['attention_mask'] + [0] * (width - len(row['input_ids'])) for row in rows], dtype=torch.long)}


def assert_training_precision(model):
    import torch

    wrong = [(name, str(p.dtype)) for name, p in model.named_parameters() if p.requires_grad and p.dtype != torch.float32]
    if wrong:
        raise ValueError('AMP training requires FP32 trainable parameters; found: ' + str(wrong[:5]))


def build_author_model(model_dir, classes, seed, device):
    import torch
    from transformers import AutoModel

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # weights_only=True requires modern PyTorch and prevents arbitrary pickle loading.
    version = tuple(int(part) for part in importlib.metadata.version('transformers').split('.')[:2])
    precision = {'dtype': torch.float32} if version >= (4, 56) else {'torch_dtype': torch.float32}
    encoder, loading = AutoModel.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False,
                                                weights_only=True, output_loading_info=True, **precision)
    if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
        raise ValueError('Backbone checkpoint is incomplete or incompatible: ' + str(loading))
    encoder.gradient_checkpointing_enable()
    head = torch.nn.Sequential(torch.nn.Dropout(.1), torch.nn.Linear(encoder.config.hidden_size, classes))
    model = torch.nn.ModuleDict({'encoder': encoder, 'head': head}).to(device=device, dtype=torch.float32)
    assert_training_precision(model)
    return model


def author_logits(model, batch):
    hidden = model['encoder'](**batch).last_hidden_state.float()
    mask = batch['attention_mask'].unsqueeze(-1)
    mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return model['head'](mean)


def batches(tokenized, indices, tokenizer, batch_size, device):
    for start in range(0, len(indices), batch_size):
        group = indices[start:start + batch_size]
        raw = [{'input_ids': tokenized['input_ids'][int(i)], 'attention_mask': tokenized['attention_mask'][int(i)]} for i in group]
        batch = tokenizer.pad(raw, padding=True, return_tensors='pt')
        yield group, {key: value.to(device) for key, value in batch.items()}


def predict_author(model, tokenized, indices, tokenizer, config, device, deadline):
    import torch

    model.eval()
    results = []
    with torch.inference_mode():
        for group, batch in batches(tokenized, indices, tokenizer, config['batch_size'], device):
            check_deadline(deadline)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device.startswith('cuda')):
                scores = author_logits(model, batch)
            values = scores.float().cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite classification scores')
            results.append(values)
    return np.concatenate(results)


def train_author(model, tokenized, targets, indices, tokenizer, config, epochs, seed, device,
                 deadline, selection=None, progress=None):
    import torch

    assert_training_precision(model)
    indices = np.asarray(indices, dtype=int)
    counts = np.bincount(targets[indices], minlength=len(set(targets)))
    if (counts == 0).any():
        raise ValueError('Training partition missing a class')
    weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW([{'params': model['encoder'].parameters(), 'lr': config['encoder_lr']},
                                  {'params': model['head'].parameters(), 'lr': config['head_lr']}],
                                 weight_decay=config['weight_decay'])
    steps_per_epoch = math.ceil(math.ceil(len(indices) / config['batch_size']) / config['accumulation'])
    total_steps = max(1, epochs * steps_per_epoch)
    warmup = max(1, int(total_steps * config['warmup_fraction']))
    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0., (total_steps - step) / max(1, total_steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024., enabled=device.startswith('cuda'))
    history = []
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(indices)
        optimizer.zero_grad(set_to_none=True)
        epoch_loss, seen, optimizer_steps, skipped_steps = 0., 0, 0, 0
        batch_count = math.ceil(len(order) / config['batch_size'])
        for step, (group, batch) in enumerate(batches(tokenized, order, tokenizer, config['batch_size'], device)):
            check_deadline(deadline)
            # Correctly normalize the last partial accumulation window too.
            window_start = (step // config['accumulation']) * config['accumulation']
            divisor = min(config['accumulation'], batch_count - window_start)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device.startswith('cuda')):
                logits = author_logits(model, batch)
                loss = loss_fn(logits, torch.as_tensor(targets[group], dtype=torch.long, device=device))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            scaler.scale(loss / divisor).backward()
            epoch_loss += float(loss.detach()) * len(group)
            seen += len(group)
            if (step + 1) % config['accumulation'] == 0 or step + 1 == batch_count:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                    optimizer_steps += 1
                else:
                    skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
        if optimizer_steps == 0:
            raise RuntimeError('All optimizer steps were skipped by FP16 overflow protection; stop and share this error.')
        row = {'epoch': epoch, 'loss': epoch_loss / seen, 'optimizer_steps': optimizer_steps, 'skipped_steps': skipped_steps}
        if selection is not None:
            scores = predict_author(model, tokenized, selection, tokenizer, config, device, deadline)
            row['balanced_accuracy'] = float(balanced_accuracy_score(targets[selection], scores.argmax(axis=1)))
        history.append(row)
        print('Training epoch:', row, flush=True)
        if progress:
            write_json(progress, {'history': history})
        if selection is not None and epoch - best_epoch(history) >= config['patience']:
            break
    return history


def measure_training_pilot(model, tokenized, targets, lengths, tokenizer, config, device,
                           deadline, workload_rows, started):
    import torch

    memory_indices = np.array(sorted(range(len(targets)), key=lambda i: lengths[i], reverse=True)[:18])
    memory_indices = np.unique(np.concatenate([memory_indices,
        [int(np.flatnonzero(targets == k)[0]) for k in sorted(set(targets))]]))
    timing_indices = representative_pilot_indices(targets, config['seed'])
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    print('MEMORY PILOT: longest passages; production batch size and accumulation.', flush=True)
    before = time.perf_counter()
    train_author(model, tokenized, targets, memory_indices, tokenizer, config, 1, config['seed'], device,
                 min(deadline, before + 10 * 60))
    memory_seconds = time.perf_counter() - before
    print('TIMING PILOT: representative sample; same settings, warmed model.', len(timing_indices), 'rows', flush=True)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    before = time.perf_counter()
    train_author(model, tokenized, targets, timing_indices, tokenizer, config, 1, config['seed'], device,
                 min(deadline, before + 10 * 60))
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    seconds = time.perf_counter() - before
    projected = projected_training_seconds(time.perf_counter() - started, seconds, len(timing_indices), workload_rows)
    report = {'pilot_version': 2, 'seconds_per_training_row': seconds / len(timing_indices),
              'memory_pilot_seconds': memory_seconds, 'memory_pilot_rows': len(memory_indices),
              'timing_pilot_seconds': seconds, 'timing_pilot_rows': len(timing_indices),
              'batch_size': config['batch_size'], 'accumulation': config['accumulation'],
              'timing_mean_tokens': float(np.mean(np.asarray(lengths)[timing_indices])),
              'training_mean_tokens': float(np.mean(lengths)), 'training_rows_bound': workload_rows,
              'projected_total_seconds': projected, 'budget_seconds': config['budget_seconds'],
              'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30 if device.startswith('cuda') else None,
              'passed': projected <= config['budget_seconds'],
              'note': 'Longest-text memory check is separate from representative production-setting throughput. Includes 50% overhead plus 20-minute reserve; disposable weights discarded.'}
    return report


def run_deberta(input_root='/kaggle/input', working='/kaggle/working', recovery_epochs=None):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('Enable a free GPU. This fine-tuning experiment is not intended for CPU.')
    if tuple(int(part) for part in torch.__version__.split('.')[:2]) < (2, 6):
        raise RuntimeError('PyTorch >=2.6 is required for safe official checkpoint loading.')
    config = dict(SETTINGS)
    device = 'cuda:0'
    started = time.perf_counter()
    deadline = started + config['budget_seconds']
    output_name = 'deberta-recovery-v1' if recovery_epochs is not None else DEBERTA_EXPERIMENT
    out = Path(working) / output_name
    out.mkdir(parents=True, exist_ok=True)
    data = discover_data(input_root)
    train = pd.read_csv(data / 'train.csv', dtype=str, keep_default_na=False)
    check_frame(train, ['id', 'prompt_id', 'text', 'model'])
    folds = official_folds(train, data / 'train.csv')
    recovery_epochs = validate_recovery_epochs(recovery_epochs, folds, config['max_epochs'])
    labels = np.array(sorted(train.model.unique()))
    targets = pd.Categorical(train.model, categories=labels).codes.astype(np.int64)
    attached = [p.parent for p in Path(input_root).rglob('verified.json')
                if (p.parent / 'config.json').is_file()]
    if len(attached) > 1:
        raise ValueError('Attach only one prepared DeBERTa checkpoint')
    old_cache = Path(working) / 'deberta-v1/pretrained'
    cache = old_cache if (old_cache / 'verified.json').exists() else Path(working) / 'deberta-pretrained'
    model_dir, pretrained = prepare_deberta(attached[0] if attached else cache)
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'transformers', 'huggingface-hub', 'tokenizers', 'sentencepiece',
                 'numpy', 'pandas', 'scikit-learn', 'pyarrow')}
    signature = {'config': config, 'pretrained': pretrained, 'versions': versions,
                 'procedure': {'version': 2, 'pooling': 'masked_mean', 'head_dropout': .1,
                               'tokenizer': 'original_sentencepiece_single_text_v1', 'trainable_dtype': 'float32',
                               'inner_prompt_fraction': .2, 'class_weights': 'inverse_training_frequency', 'amp_initial_scale': 1024},
                 'train_sha256': TRAIN_SHA256, 'fold_sha256': FOLD_SHA256, 'labels': labels.tolist()}
    if recovery_epochs is not None:
        signature['recovery'] = {'epochs': recovery_epochs, 'source': 'Original user-reported training-side inner selections',
                                 'original_pretrained_and_package_manifest_available': False,
                                 'note': 'Replays original outer fits; exact original numeric reproduction cannot be guaranteed without its lost manifest.'}
        print('RECOVERY MODE: replaying original inner-selected epoch counts:', recovery_epochs, flush=True)
    manifest = out / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError('Existing experiment settings differ. Use a new output directory.')
    write_json(manifest, signature)
    tokenizer = OriginalSentencePiece(model_dir)
    print('Training parameters: float32; compute: CUDA autocast float16; tokenizer: original SentencePiece.', flush=True)
    full = tokenizer(train.text.tolist(), padding=False, truncation=False)
    lengths = [len(ids) for ids in full['input_ids']]
    tokenized = tokenizer(train.text.tolist(), padding=False, truncation=True, max_length=config['max_length'])
    write_json(out / 'tokenization.json', {'max_tokens': max(lengths), 'truncated_rows': sum(n > config['max_length'] for n in lengths),
                                          'max_length': config['max_length']})
    print('GPU:', torch.cuda.get_device_name(0), 'Token maximum:', max(lengths), 'truncated:', sum(n > config['max_length'] for n in lengths), flush=True)
    # Pilot weights are discarded; all evaluation fits restart from pretrained weights.
    pilot = build_author_model(model_dir, len(labels), config['seed'], device)
    # Includes five inner-selection runs + five fresh outer refits, worst 3 epochs.
    training_rows_bound = 0
    for fold in sorted(set(folds)):
        outer_training = np.flatnonzero(folds != fold)
        if (out / f'fold-{fold}' / 'complete.json').exists():
            continue  # Completed fold checksums are verified before scores are reused below.
        if recovery_epochs is not None:
            training_rows_bound += len(outer_training) * recovery_epochs[int(fold)]
        else:
            fitting, _ = inner_prompt_split(train, outer_training, config['seed'] + int(fold))
            training_rows_bound += (len(fitting) + len(outer_training)) * config['max_epochs']
    pilot_report = measure_training_pilot(pilot, tokenized, targets, lengths, tokenizer, config,
                                          device, deadline, training_rows_bound, started)
    write_json(out / 'pilot.json', pilot_report)
    del pilot
    gc.collect(); torch.cuda.empty_cache()
    print('PILOT RESULT:', json.dumps(pilot_report), flush=True)
    if not pilot_report['passed']:
        print('PILOT STOP: projected run exceeds the six-hour budget. No full-fold training started. Stop Session and share pilot.json.', flush=True)
        return {'status': 'stopped_at_budget_guard', 'pilot': pilot_report}
    oof_scores = np.full((len(train), len(labels)), np.nan, dtype=np.float32)
    fold_reports = []
    for fold in sorted(set(folds)):
        folder = out / f'fold-{fold}'
        folder.mkdir(exist_ok=True)
        complete = folder / 'complete.json'
        tr, va = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if complete.exists():
            cached = json.loads(complete.read_text())
            if cached['metrics']['fold'] != int(fold):
                raise ValueError('Completed checkpoint belongs to a different fold')
            for name, digest in cached['file_sha256'].items():
                if sha256_file(folder / name) != digest:
                    raise ValueError('Completed fold checksum mismatch')
            restored = np.load(folder / 'scores.npy', allow_pickle=False)
            if restored.shape != (len(va), len(labels)) or not np.isfinite(restored).all():
                raise ValueError('Invalid completed-fold score dimensions')
            oof_scores[va] = restored
            fold_reports.append(cached['metrics'])
            continue
        if recovery_epochs is not None:
            epochs, history = recovery_epochs[int(fold)], []
        else:
            fitting, selection = inner_prompt_split(train, tr, config['seed'] + int(fold))
            model = build_author_model(model_dir, len(labels), config['seed'] + int(fold), device)
            history = train_author(model, tokenized, targets, fitting, tokenizer, config, config['max_epochs'],
                                   config['seed'] + int(fold), device, deadline, selection, folder / 'inner_progress.json')
            epochs = best_epoch(history)
            del model
            gc.collect(); torch.cuda.empty_cache()
        model = build_author_model(model_dir, len(labels), config['seed'] + int(fold), device)
        outer_history = train_author(model, tokenized, targets, tr, tokenizer, config, epochs,
                                     config['seed'] + int(fold), device, deadline, progress=folder / 'outer_progress.json')
        scores = predict_author(model, tokenized, va, tokenizer, config, device, deadline)
        oof_scores[va] = scores
        metric = {'fold': int(fold), 'selected_epochs': epochs, 'inner_history': history,
                  'epoch_selection_origin': 'replayed_original_inner_choice' if recovery_epochs is not None else 'new_inner_selection',
                  'outer_history': outer_history, 'balanced_accuracy': float(balanced_accuracy_score(targets[va], scores.argmax(axis=1))),
                  'training_rows': len(tr), 'validation_rows': len(va)}
        pending = folder / 'model.pt.part'
        torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, pending)
        pending.replace(folder / 'model.pt')
        np.save(folder / 'scores.npy', scores)
        write_json(complete, {'metrics': metric, 'file_sha256': {name: sha256_file(folder / name) for name in ('model.pt', 'scores.npy')}})
        fold_reports.append(metric)
        print('OUTER FOLD RESULT:', metric['fold'], metric['balanced_accuracy'], 'epochs', epochs, flush=True)
        del model
        gc.collect(); torch.cuda.empty_cache()
    if not np.isfinite(oof_scores).all():
        raise ValueError('Incomplete or invalid OOF scores')
    predictions = labels[oof_scores.argmax(axis=1)]
    score = float(balanced_accuracy_score(train.model, predictions))
    pd.DataFrame({'id': train.id, 'fold': folds, 'truth': train.model, 'prediction': predictions}).to_csv(out / 'oof.csv', index=False)
    np.savez_compressed(out / 'oof_scores.npz', scores=oof_scores, labels=labels, ids=train.id.to_numpy(dtype=str))
    pd.DataFrame({'model': labels, 'recall': recall_score(train.model, predictions, labels=labels, average=None)}).to_csv(out / 'per_class.csv', index=False)
    pd.DataFrame(confusion_matrix(train.model, predictions, labels=labels), index=labels, columns=labels).to_csv(out / 'confusion.csv')
    report = {'balanced_accuracy': score, 'folds': fold_reports, 'manifest': signature,
              'elapsed_minutes': (time.perf_counter() - started) / 60,
              'note': 'Original inner-selected epochs replayed; fresh outer refits regenerate OOF reports. No new epoch search or exact reproduction guarantee.' if recovery_epochs is not None else
                      'Inner prompt groups select epochs; fresh outer refits predict untouched outer prompts. No final full-data fit or competition submission yet.'}
    write_json(out / 'summary.json', report)
    archive = Path(working) / (output_name + '-reports.zip')
    bundle_deberta_reports(out, archive)
    print('REPORT BUNDLE:', archive, flush=True)
    print('RESULT TO SHARE:', json.dumps({'balanced_accuracy': score, 'elapsed_minutes': report['elapsed_minutes'],
                                        'selected_epochs': [row['selected_epochs'] for row in fold_reports]}), flush=True)
    print('SUCCESS: DeBERTa five-fold experiment complete. Save reports before stopping the GPU session.', flush=True)
    return report

ensure_sentencepiece()

_ = run_deberta()
