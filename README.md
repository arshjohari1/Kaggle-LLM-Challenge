# Name That LLM

A CPU-first, reproducible starting point for the Kaggle authorship competition.
Given a passage, the pipeline predicts one of 18 model labels. It starts with
TF-IDF and linear classifiers before adding expensive neural models.

**Current status:** official data received and real prompt-grouped experiments
completed. Labels, metric, string IDs, inference path, Parquet output, offline
execution, and nine-hour limits are confirmed against current organizer pages.
See [current evidence](docs/status.md) and [aggregate experiment reports](reports/).
Local cross-validation results are not leaderboard scores. No Kaggle submission
has been made and no spending is authorized.

## Install

Python 3.12 on Linux is the tested configuration. In the cloud workspace:

```bash
cd /workspace/Kaggle-LLM-Challenge
python -m venv /workspace/venvs/name-that-llm
source /workspace/venvs/name-that-llm/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pytest -q
```

For another machine, replace the virtual environment path with your preferred
location. `requirements.lock` pins the tested environment, including development
tools. Data, model artifacts, wheels, and experiment outputs are gitignored.
Do not commit competition passages, credentials, or third-party model weights.

## Run the real baseline

The user has handled participation requirements and supplied `train.csv`,
`test.csv`, and `sample_submission.parquet`, retained privately in
`data/official/`. On a new machine, obtain your authorized copies of these files.
Preserve originals; their SHA-256 identifies each experiment's data version.
Do not use public extra corpora until their licenses, model identities, and
competition eligibility have been audited.

```bash
ntl prepare --train data/official/train.csv --out runs/official-v1
ntl evaluate --train data/official/train.csv \
  --folds runs/official-v1/folds.csv --out runs/baselines-v1
```

The five candidates are word TF-IDF + logistic regression, character TF-IDF +
logistic regression, combined TF-IDF + logistic regression, combined TF-IDF +
linear SVM, and combined TF-IDF/style features + linear SVM. Run only the first
candidate with `--candidates word_lr` for the simplest baseline.

Additional candidates include `wide_svm`, `punct_svm`, `multiview_svm`, nonlinear
style models, an optimized RBF text kernel, a prompt-residual kernel, and `stacked`.
The stacker constructs its meta-training predictions using three inner grouped
folds inside every outer training partition; the combiner never trains on outer
validation labels. The linear grid encodes C and style weight in candidate names,
for example `multi_c015_s005` (C=0.15, weight=0.05). The optional
`--cache /workspace/cache/ntl-linear-features` reuses fold-local feature fits;
the cache is not required for inference. Keep thread limits modest.

`prepare` validates all 18 labels, audits class counts and lengths, and saves
fixed five-fold prompt-grouped splits. Prompts linked by duplicate passages are
kept together as well. `evaluate` refuses changed data with stale fold files.
Representations and feature scaling are fitted separately inside each fold.
Punctuation and case are retained. Missing text, duplicate IDs, invalid labels,
and folds missing a class cause explicit failures.

Each candidate saves its out-of-fold predictions and decision scores, full
settings, fold scores, pooled balanced accuracy, per-class recall, confusion
matrix, confused pairs, word-count slices, wall time, and process peak memory.
The manifest records seeds, package versions, Git revision/dirty status, and data
and fold hashes. Decision scores are **not calibrated probabilities**.

Balanced accuracy is the average recall across classes. Random guessing has an
expected score of 1/18, about 5.56%. The primary comparison is pooled out-of-fold
balanced accuracy; mean and standard deviation of fold scores are also saved.
Length slices explicitly report how many classes are represented.

Repeatedly choosing methods using these folds can overfit validation. Treat the
best cross-validation score as a model-selection result. Reserve fresh prompt
groups or use nested grouped validation for a stronger final estimate, especially
before tuning ensemble weights or calibration. No weights are tuned on the same
held-out predictions and presented as an unbiased score.

## Train and prepare offline inference

Choose a candidate after inspecting actual validation results. Example commands
below use `union_svm`; this is an example, **not a measured winner**.

```bash
ntl train --train data/official/train.csv --candidate union_svm --out artifacts/union-svm-v1
ntl predict --model artifacts/union-svm-v1/model.joblib \
  --test data/official/test.csv --out artifacts/union-svm-v1/submission.parquet
ntl benchmark --model artifacts/union-svm-v1/model.joblib \
  --test data/official/test.csv --rows 4000 --out artifacts/union-svm-v1/benchmark.json
python scripts/build_kaggle_bundle.py --artifact artifacts/union-svm-v1 \
  --out artifacts/kaggle-bundle-v1
```

Commands refuse to overwrite run directories and prediction files. Inference is
batched and validates ID coverage, uniqueness, nonnull predictions, and labels,
including a Parquet readback. IDs stay strings, as confirmed by the official
sample. For other tasks needing integers, `--id-dtype int64` requires lossless
conversion.
The 4,000-row benchmark repeats available test samples; it measures engineering
cost only and must be repeated with real representative lengths and final model.
Peak memory is the process high-water mark, not an incremental model allocation.

Attach the contents of `artifacts/kaggle-bundle-v1/` to a Kaggle notebook as an
input dataset, then import [the notebook](notebooks/kaggle_inference.ipynb).
The bundle includes the model, package wheel, dependency wheels, and checksums.
Its notebook installs from attached wheels with `--no-index` and needs no online
inference. Build wheels using the **actual Kaggle Python version and platform**;
the notebook checks the Python minor version and fails clearly on a mismatch.
Only load trusted joblib artifacts: deserialization can execute Python code.

Default paths and output schema are confirmed by the organizer's Evaluation
page and starter notebook. Package compatibility still needs checking on the
actual Kaggle runtime. Turn Internet off and run the complete notebook there,
then inspect its saved output.
This repository does not create Kaggle versions or submit anything automatically.

## Development check without competition access

```bash
python scripts/make_synthetic.py --out data/synthetic-v1
ntl prepare --train data/synthetic-v1/train.csv --out runs/synthetic-v1
ntl evaluate --train data/synthetic-v1/train.csv --folds runs/synthetic-v1/folds.csv \
  --out runs/synthetic-baselines-v1
```

Synthetic passages contain intentionally obvious author markers. High scores
on them prove plumbing, **not real authorship detection**.

See [experiment plan](docs/experiments.md), [current evidence](docs/status.md),
and [agent instructions](AGENTS.md). All implementation is original; public
resource links are context, not reproduced results or copied implementations.

## Frozen encoder experiment

An optional, separate CPU environment contains `torch==2.8.0+cpu`,
`transformers==4.57.6`, and `safetensors==0.7.0`. It uses the public MIT-licensed
Kaggle copy `happenmass/bge-small-en-v1.5/transformers/bge/2`. The local encoder is
at `/workspace/models/bge-small-en-v1.5`, with checksums and provenance. No
third-party weights are committed. Only safetensors and static configuration
are loaded, with remote code and network loading disabled. Exact hashes of the
Kaggle copy identify this experiment; an upstream Hugging Face revision is not
claimed.

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  /workspace/venvs/ntl-embeddings/bin/python scripts/embedding_experiment.py \
  --encoder /workspace/models/bge-small-en-v1.5 \
  --train data/official/train.csv --folds runs/official-v1/folds.csv \
  --out runs/bge-small-v2
```

Extraction caches CLS and masked-mean embeddings. Downstream scaling and
classifiers fit inside each fold. Encoder weights are not fine-tuned.
