# Current evidence

## Implemented and verified in this cloud machine

- Python 3.12 environment with pinned dependencies; package integrity checks pass.
- 13 pytest cases pass, covering reproducible grouped splits, duplicate-linked
  prompt isolation, schema failures, ID preservation, fold-local vocabularies and
  style scaling, all five classifiers, serialization, submission rejection,
  complete CLI execution, changed-data rejection, corrupt-artifact rejection,
  and lossless integer submission IDs.
- Five candidates each completed all five validation folds on a 270-row,
  15-prompt **synthetic fixture** with 18 labels and deliberately obvious author
  markers. All fits completed without convergence warnings. This validates the
  pipeline and reporting; it does not measure competition performance.
- A synthetic style/SVM artifact was trained, saved, checksum/version-checked,
  reloaded, and used to write and read back valid Parquet predictions.
- A 4,000-row inference check on repeated synthetic passages completed in about
  1.48 seconds with process peak RSS about 202 MiB. This is a tiny-vocabulary
  integration check. A real fitted vocabulary and representative passage lengths
  must be benchmarked separately; these numbers do not predict Kaggle runtime.
- The exact notebook code cells were executed locally using only the attached
  wheels (`pip --no-index`). They installed the package in a separate target,
  loaded the trusted fixture artifact in a new Python process, and produced an
  18-row Parquet identical to the ordinary CLI result. The actual Kaggle runtime
  has **not** been tested.

Fixture inputs live in ignored `data/synthetic-v1/`; experiment reports are in
ignored `runs/synthetic-baselines-v1/`. The model and wheel bundle are in ignored
`artifacts/synthetic-style-v1/` and `artifacts/synthetic-kaggle-bundle-v1/`.
These files stay local and are not part of the Git push. Early fixture manifests
have a null Git revision because they ran before this repository's first commit.
Official experiments should be run from a recorded commit.

## Blocked or unrun

- Official competition pages and data API: proxy-level network denial.
- Real data audit, real cross-validation scores, error analysis, final model:
  waiting for authorized competition files.
- Eligibility, parental consent, prize requirements, current organizer replies,
  dates, metric, labels, offline paths and output dtype: official verification
  still required.
- Kaggle runtime, free accelerator quota, offline hidden-test inference and real
  submission: no account runtime connected and no submission attempted.
- Frozen transformer embeddings, neural fine-tuning, and ensembles: deliberately
  pending the real baseline, rules verification and free compute checks.

## What the user can provide

1. Download the official train/test CSVs and sample submission after accepting
   competition rules, then upload the files to this chat. Do not share tokens.
2. Supply the current rules/organizer clarifications or make Kaggle pages
   reachable through the environment network settings. Review the saved Kaggle
   domain additions and save/publish the environment configuration as prompted.
3. Review age/consent and prize eligibility with a parent or guardian; contact
   Kaggle/the organizer if the terms are unclear. Check free GPU quota when we
   reach transformer experiments.

No spending, paid API generation, Kaggle submission, or acceptance of terms on
the user's behalf has occurred. No third-party training data or code was copied.
