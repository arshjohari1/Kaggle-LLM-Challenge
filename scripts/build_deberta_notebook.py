"""Build the fine-tuning notebook without a custom package dependency."""
import ast
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    gen = (root / 'src/ntl/generative.py').read_text()
    source = (root / 'src/ntl/deberta.py').read_text()
    definitions = {node.name: ast.get_source_segment(gen, node) for node in ast.parse(gen).body
                   if isinstance(node, ast.FunctionDef)}
    tree = ast.parse(gen)
    constants = '\n'.join(ast.get_source_segment(gen, node) for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id in ('TRAIN_SHA256', 'FOLD_SHA256', 'FOLD_PATTERN') for t in node.targets))
    imports = '''import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
'''
    data = '\n\n'.join(definitions[name] for name in ('sha256_file', 'write_json', 'official_folds', 'check_frame', 'discover_data'))
    source = '\n'.join(line for line in source.splitlines() if not line.startswith('from ntl.'))
    # Fail early with instructions rather than silently installing network packages.
    preflight = 'ensure_sentencepiece()\n'
    code = imports + '\n' + constants + '\n\n' + data + '\n\n' + source + '\n\n' + preflight + '\n_ = run_deberta()\n'
    intro = '''# DeBERTa-v3-large author-classification fine-tuning

TRAINING EXPERIMENT ONLY — keep the 0.417 competition notebook unchanged.
Attach Name That LLM competition data. Enable a free GPU and Internet On for
the first official Microsoft checkpoint download; it is revision/hash verified.
No Qwen input is required. Uses GPU 0 only, mixed precision and checkpointing.
This is not an offline competition submission notebook. A later deployment uses
attached weights/packages with Internet Off; there is no submission.parquet yet.
Pilot uses longest passages, trains a disposable model, then discards its weights.
Five fixed outer prompt folds each use inner training prompts to select 1–3 epochs,
then a fresh outer-training refit predicts untouched held-out prompts.
Default compute budget 6 hours; pilot may stop before full training. Stop Session
explicitly after saving results. Partial completed folds resume in the same session.
Share deberta-v2/{pilot.json,summary.json,manifest.json,tokenization.json,oof.csv,
oof_scores.npz,per_class.csv,confusion.csv}; do not upload large model checkpoints.
No full GPU training or pretrained-weight smoke test has been run in the CPU
development environment; local tests cover a tiny randomly initialized encoder.
'''
    (root / 'notebooks/kaggle_deberta.py').write_text(code)
    nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'cells': [
            {'cell_type': 'markdown', 'id': 'instructions', 'metadata': {}, 'source': intro.splitlines(keepends=True)},
            {'cell_type': 'code', 'id': 'deberta', 'metadata': {}, 'execution_count': None,
             'outputs': [], 'source': code.splitlines(keepends=True)}]}
    (root / 'notebooks/kaggle_deberta.ipynb').write_text(json.dumps(nb, indent=1) + '\n')
    epochs = json.loads((root / 'reports/deberta-v2/user_completion_result.json').read_text())['selected_epochs']
    recovery = code.rsplit('\n_ = run_deberta()', 1)[0] + f'\n_ = run_deberta(recovery_epochs={epochs!r})\n'
    (root / 'notebooks/kaggle_deberta_recovery.py').write_text(recovery)
    recovery_intro = '''# DeBERTa report recovery — run as a SAVED BACKGROUND VERSION

Replays original inner-selected epochs [3,2,3,3,3], skipping repeated selection.
No new hyperparameter search; original outer prompt folds/learning settings stay.
Original model/package manifest was lost, so exact 32.08% reproduction is not guaranteed.
Attach competition data, select free GPU and Internet On, then use
Save Version → Save & Run All DIRECTLY (not an interactive Run All first).
Save a GPU-enabled training version; this is not a competition submission.
Output: deberta-recovery-v1-reports.zip contains all eight small reports and checksums.
Open the completed saved version's Output to download it. Per-fold checkpoints
are large and excluded from the report ZIP. Keep the 0.417 submission unchanged.
Pilot and six-hour budget guard remain. No recovery GPU run has been executed here.
'''
    recovery_nb = {**nb, 'cells': [
        {'cell_type': 'markdown', 'id': 'recovery-instructions', 'metadata': {}, 'source': recovery_intro.splitlines(keepends=True)},
        {'cell_type': 'code', 'id': 'recovery', 'metadata': {}, 'execution_count': None,
         'outputs': [], 'source': recovery.splitlines(keepends=True)}]}
    (root / 'notebooks/kaggle_deberta_recovery.ipynb').write_text(json.dumps(recovery_nb, indent=1) + '\n')
    preparation = imports + '\n' + constants + '\n\n' + data + '\n\n' + source
    preparation += '''

path, receipt = prepare_deberta('/kaggle/working/deberta-pretrained')
import subprocess, sys
wheels = path / 'wheels'
wheels.mkdir(exist_ok=True)
subprocess.run([sys.executable, '-m', 'pip', 'download', '--only-binary=:all:', '--no-deps',
                'sentencepiece==0.2.1', '-d', str(wheels)], check=True)
wheel = next(wheels.glob('sentencepiece-0.2.1-*.whl'))
write_json(wheels / 'sentencepiece-wheel.json', {'filename': wheel.name, 'sha256': sha256_file(wheel),
                                               'source': 'PyPI sentencepiece==0.2.1 via verified TLS'})
print('SUCCESS: verified official checkpoint and offline tokenizer wheel prepared at', path)
print('Revision:', receipt['revision'], 'License:', receipt['license'])
'''
    (root / 'notebooks/kaggle_deberta_prepare.py').write_text(
        '# CPU preparation only: Internet On, no competition data required, no GPU.\n' + preparation)


if __name__ == '__main__':
    main()
