# Current evidence

The user supplied the competition ZIP and confirmed participation requirements
are handled. Originals are retained privately in `data/official/`. The data have
2,699 passages, 150 prompts, and 18 official labels. Every class has 150 rows
except claude-fable-5 (149). There are no blank passages, duplicate IDs,
normalized duplicate passages, or exact train/test text overlaps. The visible
test has three rows. The sample Parquet has string id/model columns. Word counts
range from 36 to 274 (median 101).

## Completed CPU experiments

Fixed five-fold StratifiedGroupKFold splits use seed 42 and prompt_id groups.
Every split contains all classes; each prompt stays wholly in one fold.
Duplicate-connected prompt isolation is implemented even though no duplicates
occur. Learned vectorizers, scaling, projections and classifiers fit only within
each training partition. Every comparison uses the same data and fold hashes.

See [full comparison](../reports/cpu-v1/comparison.csv),
[class recalls](../reports/cpu-v1/per_class_recall.csv), and
[confusion matrix](../reports/cpu-v1/confusion.csv).

| Method | Pooled out-of-fold balanced accuracy |
| --- | ---: |
| Word TF-IDF + logistic regression | 28.11% |
| Character TF-IDF + logistic regression | 31.41% |
| Word + character logistic regression | 31.04% |
| Word + character linear SVM | 30.63% |
| Word + character + original style features, primal SVM | 36.67% |
| Wider character/word/style SVM | 36.45% |
| Boundary-character/punctuation SVM | 36.12% |
| Four text views + style, linear SVM | 37.12% |
| Tuned four-view SVM, C=1.5/style weight=0.2 | **37.56%** |
| Nonlinear text RBF SVM | 31.90% |
| Style-only RBF SVM | 27.23% |
| Style-only histogram gradient boosting | 26.64% |
| Cosine text kernel control | 34.86% |
| Training-prompt residual kernel | 35.56% |
| Nested three-family stack | 34.78% |

Additional completed linear-grid results are in the CSV. Small differences among
the leading settings are not established improvements. Selected fold scores are
36.50%, 39.26%, 37.96%, 35.74%, and 38.33%. The conditional 95% prompt-bootstrap
interval is approximately 35.81–39.33%. The paired improvement over word-only
logistic regression is about 7.71–11.15 percentage points. Intervals resample 150
prompt groups 2,000 times with seed 2026; they do not include selection optimism,
training randomness or hidden-test distribution shift. **These are local model
selection results, not leaderboard scores or guarantees.**

The stacker learns its combiner from three inner grouped OOF sets inside each
outer training fold. It is slower and worse than the direct linear model and
was not selected. Prompt residual directions use training prompts only; test
prompt IDs are unnecessary. That method improved its control but not the leader.

The original unscaled-style dual solver and direct sparse-libsvm RBF attempt
were stopped for excessive cost before producing complete results. The primal
solver and explicit Gram-kernel replacements completed all five folds. Heavy
style-weight grid settings were pruned for cost; partial fits do not count as
completed evaluations and are not used for selection.

## Frozen encoder experiment

The public MIT-licensed Kaggle BGE-small copy was downloaded over verified TLS.
Its model card, version, archive and individual hashes are retained locally.
Only static tokenizer/configuration and safetensors files are loaded; remote
code is disabled. An original upstream Hugging Face revision is not claimed.
No extra labelled corpus is used.

The separate `/workspace/venvs/ntl-embeddings` environment uses CPU-only PyTorch
2.8.0, Transformers 4.57.6 and safetensors 0.7.0. Frozen extraction uses no
retrieval prompt prefix, max length 512, batch size 16, and CLS/masked-mean pools.
Downstream classifier/scaler fits use the same outer folds. Extraction took 339
seconds, with no truncation (maximum 381 tokens). Eight pooling/classifier
settings scored 11.67–15.15%. Six further training-prompt residual settings
scored 9.52–16.22%. These small retrieval embeddings were not selected. This does
not establish how larger generative encoders would perform. See
[all 14 embedding results](../reports/frozen-embeddings-v1/comparison.csv).

## Validation and retained outputs

33 tests pass, including prompt/duplicate isolation, held-out vocabulary canaries,
fold-local style scaling, all model round trips, inner stack folds, kernel
numerical equivalence and positive semidefiniteness, corrupt-artifact rejection,
stale-fold rejection, and the full CLI/submission path. Thread limits prevent
OpenMP oversubscription. Synthetic fixture scores are never presented as
competition performance.

Raw OOF predictions, settings, margins, manifests and fold files stay in ignored
`runs/`. Only class-level aggregates and hashes are committed under `reports/`.
The selected `multi_c150_s020` model was refitted on all 2,699 rows in 33.75
seconds from clean source commit `9fdd233`, and saved in
`artifacts/cpu-best-v1/model.joblib` (about 34 MB). The training manifest pins
versions, data checksum and model checksum. Three-row predictions match the
official schema and test IDs. The default string-ID path preserves the supplied
IDs including their leading zeroes.

The final artifact processed 4,000 deterministic samples of real training
passages in 12.56 seconds including load/validation, with process peak RSS about
394 MiB. This is a timing check, not evaluation on training passages. The exact
offline notebook code cells installed the attached wheels with no package index,
predicted 4,000 rows, wrote/read Parquet, and matched the CLI predictions and
file checksum exactly. Notebook inference took 14.12 seconds and the entire
offline install/inference took 20.39 seconds. See [validation evidence](../reports/cpu-v1/offline_validation.json).

The ready local bundle is `artifacts/cpu-kaggle-bundle-v1.zip` (weights plus
offline wheels), with notebook `notebooks/kaggle_inference.ipynb`. See
[Kaggle run instructions](kaggle.md). Actual Kaggle execution and free accelerator
quotas remain unverified; no GPU is exposed here. No leaderboard score is known.

No spending, paid API calls, billable compute, Kaggle submission, acceptance of
terms on the user's behalf, or publication of competition passages has occurred.
