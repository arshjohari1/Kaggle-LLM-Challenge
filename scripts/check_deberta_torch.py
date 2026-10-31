"""Exercise genuine DeBERTa architecture training without claiming pretrained accuracy."""
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import DebertaV2Config, DebertaV2Model, PreTrainedTokenizerFast

from ntl.deberta import SETTINGS, OriginalSentencePiece, assert_training_precision, author_logits, build_author_model, predict_author, train_author
from ntl.generative import write_json


torch.set_num_threads(2)
torch.manual_seed(42)
started = time.perf_counter()
vocab = {'[PAD]': 0, '[UNK]': 1, 'bright': 2, 'short': 3, 'sentences': 4, 'long': 5,
         'words': 6, 'with': 7, 'punctuation': 8, 'and': 9, 'style': 10}
raw = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
raw.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='[PAD]', unk_token='[UNK]')
texts = ['bright short sentences', 'long words with punctuation and style'] * 6
targets = np.tile([0, 1], 6).astype(np.int64)
encoded = tokenizer(texts, truncation=True, max_length=32)
config = {**SETTINGS, 'batch_size': 2, 'accumulation': 2, 'encoder_lr': .001, 'head_lr': .01}
architecture = DebertaV2Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
                               num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=64,
                               relative_attention=True, pos_att_type=['p2c', 'c2p'], pad_token_id=0)
with tempfile.TemporaryDirectory(prefix='deberta-check-') as folder:
    path = Path(folder)
    DebertaV2Model(architecture).half().save_pretrained(path, safe_serialization=True)
    model = build_author_model(path, 2, 42, 'cpu')
    assert all(p.dtype == torch.float32 for p in model.parameters())
    # Reproduce the user's exact GradScaler error on CPU with half parameters.
    bad = torch.nn.Linear(2, 2).half()
    optimizer = torch.optim.SGD(bad.parameters(), lr=.01)
    amp = torch.amp.GradScaler('cpu', init_scale=8)
    amp.scale(bad(torch.ones(2, 2).half()).float().sum()).backward()
    try:
        amp.unscale_(optimizer)
    except ValueError as error:
        assert 'unscale FP16 gradients' in str(error)
    else:
        raise AssertionError('FP16 regression reproduction did not fail as expected')
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    amp = torch.amp.GradScaler('cpu', init_scale=8)
    batch = tokenizer(texts[:2], padding=True, return_tensors='pt')
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss = torch.nn.functional.cross_entropy(author_logits(model, batch).float(), torch.tensor([0, 1]))
    amp.scale(loss).backward()
    amp.unscale_(optimizer)
    assert all(p.grad is None or p.grad.dtype == torch.float32 for p in model.parameters())
    amp.step(optimizer); amp.update(); optimizer.zero_grad(set_to_none=True)
    encoder_before = {k: v.detach().clone() for k, v in model['encoder'].state_dict().items()}
    head_before = model['head'][1].weight.detach().clone()
    history = train_author(model, encoded, targets, np.arange(8), tokenizer, config, 3, 42, 'cpu',
                           time.perf_counter() + 120, np.arange(8, 12))
    assert any(not torch.equal(v, encoder_before[k]) for k, v in model['encoder'].state_dict().items())
    assert not torch.equal(head_before, model['head'][1].weight)
    assert model['encoder'].is_gradient_checkpointing
    scores = predict_author(model, encoded, np.arange(12), tokenizer, config, 'cpu', time.perf_counter() + 120)
    assert scores.shape == (12, 2) and np.isfinite(scores).all()
    torch.save(model.state_dict(), path / 'checkpoint.pt')
    restored = build_author_model(path, 2, 99, 'cpu')
    restored.load_state_dict(torch.load(path / 'checkpoint.pt', weights_only=True, map_location='cpu'))
    again = predict_author(restored, encoded, np.arange(12), tokenizer, config, 'cpu', time.perf_counter() + 120)
    np.testing.assert_allclose(scores, again, atol=1e-6)
    try:
        train_author(model, encoded, targets, np.arange(8), tokenizer, config, 1, 42, 'cpu', 0)
    except TimeoutError:
        pass
    else:
        raise AssertionError('Expired training budget accepted')
out = Path('reports/deberta-v2')
out.mkdir(parents=True, exist_ok=True)
write_json(out / 'local_training_check.json', {
    'purpose': 'Random tiny 2-layer DeBERTa-v2 architecture functional test, not pretrained-large evaluation',
    'half_checkpoint_loaded_as_fp32': True, 'original_fp16_unscale_error_reproduced_on_cpu': True,
    'fp32_parameters_cpu_autocast_and_grad_scaler_step_passed': True,
    'encoder_and_head_weights_updated': True, 'gradient_checkpointing_enabled': True,
    'finite_predictions': True, 'weights_only_checkpoint_roundtrip': True, 'deadline_stops_training': True,
    'epochs_executed': len(history), 'seconds': time.perf_counter() - started,
    'torch_version': torch.__version__, 'official_pretrained_weights_tested': False,
    'real_gpu_memory_speed_accuracy': 'unrun', 'network_blocker': 'Hugging Face and Kaggle proxy returned 403 in this cloud environment; official download verified at Kaggle runtime.'})
print('PASS: real DeBERTa architecture backward updates, checkpointing, saved-weight reload and budget guard.')

# Compare the new original-SPM adapter to the actual legacy HF slow implementation.
import sentencepiece as sp
from transformers import DebertaV2Tokenizer
with tempfile.TemporaryDirectory(prefix='deberta-spm-check-') as directory:
    directory = Path(directory)
    corpus = directory / 'text.txt'
    corpus.write_text(('Hello WORLD! Unicode café — punctuation.\nMultiple   spaces and headings:\nMathematics 123.45 with contractions: isn\'t.\n' * 20))
    sp.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(directory / 'spm'), vocab_size=64,
                                 hard_vocab_limit=False, pad_id=0, bos_id=1, eos_id=2, unk_id=3,
                                 pad_piece='[PAD]', bos_piece='[CLS]', eos_piece='[SEP]', unk_piece='[UNK]',
                                 minloglevel=2)
    legacy = DebertaV2Tokenizer(vocab_file=str(directory / 'spm.model'))
    adapter = OriginalSentencePiece(directory)
    examples = ['Hello WORLD!', 'Unicode café — punctuation.', 'Multiple   spaces\nand\ttabs.',
                "Mathematics 123.45: isn't.", 'Hello [MASK] world', '[CLS] Hello [SEP]']
    for truncation in (False, True):
        options = {'truncation': truncation, 'max_length': 12} if truncation else {}
        assert adapter(examples, **options)['input_ids'] == legacy(examples, **options)['input_ids']
    encoded = adapter(examples)
    padded = adapter.pad([{'input_ids': ids, 'attention_mask': mask} for ids, mask in zip(encoded['input_ids'], encoded['attention_mask'])])
    official = legacy(examples, padding=True, return_tensors='pt')
    assert torch.equal(padded['input_ids'], official['input_ids'])
    assert torch.equal(padded['attention_mask'], official['attention_mask'])
    report_path = out / 'local_training_check.json'
    import json
    report = json.loads(report_path.read_text())
    report['original_sentencepiece_matches_legacy_hf_on_unicode_whitespace_specials_padding_truncation'] = True
    write_json(report_path, report)
print('PASS: original SentencePiece parity against legacy Hugging Face DeBERTa tokenizer.')
