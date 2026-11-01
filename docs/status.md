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

60 tests pass, including prompt/duplicate isolation, held-out vocabulary canaries,
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
[Kaggle run instructions](kaggle.md). The user subsequently ran the standalone
CPU notebook twice, saved version 1, and submitted it successfully. Their
screenshot reports **0.372 public balanced accuracy**. This is a real Kaggle
score, separate from the local 0.37560 selection result. No GPU is exposed here.

No spending, paid API calls, billable compute, automatic Kaggle submission,
acceptance of terms on the user's behalf, or publication of competition passages
has occurred.

## Frozen generative transformer result

The user authorized this stage and reports **00:00 / 30:00 GPU hours** on Kaggle
(zero used of thirty). The earlier screenshot of Benchmarks AI dollar credits
was a separate quota. No Kaggle GPU has been launched from this workspace.

The new main body is the Apache-2.0 official **Qwen3-1.7B-Base** model, frozen
with four feature pools and two SVM heads. The original five-fold assignment is
embedded as fold numbers by sorted prompt rank and verified against both the
original training-file and fold-file SHA-256 values. All scaler/head fits remain
fold-local. Extraction caches resume batches and verify checksums; downstream
reports include OOF margins, per-class recall, confusion matrices and length
slices. Full-data refitting produces a dynamic offline Parquet submission.

The actual official weights passed the CPU functional checks documented in
[the integration report](../reports/qwen-frozen-v1/cpu_integration.json).
The test suite also verifies held-out prompts cannot influence feature scaling
and that the generated self-contained notebook embeds exactly the tested source.
The user subsequently supplied a successful Kaggle run's final log:
`middle_mean_linear` achieved **0.3511881680338056 grouped balanced accuracy**,
with **16.524883034566667 minutes** elapsed on **Tesla T4**. The visible
three-row submission passed the format checks. This is 2.44 percentage points
below the selected CPU model's grouped score, and is not a leaderboard score.
See [the reported result](../reports/qwen-frozen-v1/kaggle_user_result.json).

All six requested files were subsequently received and retained privately in
`runs/qwen-kaggle-v1-received-20261023/`, with a verified separate ZIP in
`artifacts/qwen-kaggle-v1-reports.zip`. Input file hashes and non-row-level
reports are preserved in Git. Uploaded file IDs are retained only in the private
receipt manifest, for possible re-download. Raw IDs/predictions and margins are
not committed. Future raw-file access depends on retaining the workspace or its
snapshot; Git alone cannot restore private outputs. The entire six-file upload
is 318,167 bytes, with a roughly 199 KiB ZIP backup including its manifest.

`scripts/analyze_qwen_reports.py` verified data/fold hashes, all 2,699 row IDs and
fold assignments, labels, eight candidate summaries, finite margins, and
reconstructed class recalls/confusion matrices. It reproduced both CPU and Qwen
scores. Qwen was correct on **417 CPU errors**; CPU was correct on **483 Qwen
errors**, and both were correct on 531 rows. Fixed equal-weight per-passage rank
averaging of their decision scores (CPU scores break exact ties) measured
**0.4093263733532189**, a **3.37 percentage-point** gain over CPU, improving all
five folds and all three length slices. No ensemble weights, thresholds or
calibration were fitted on held-out labels. Already-selected base models and
reused folds still create selection optimism. The conditional paired-prompt
bootstrap interval for the gain is about **1.71–5.07 percentage points** and
excludes refitting uncertainty and selection effects.

Preserve the CPU model and its 0.372 public submission. The next candidate is the
implemented CPU/Qwen rank-blend inference notebook before further GPU training;
its user-run GPU execution reproduces the score and its reported public score is 0.417. See
[paired findings](../reports/qwen-frozen-v1/analysis.md). The historical local
CPU integration/fold-verification reports describe their original checks;
their unrun GPU fields do not describe the subsequent user-run experiment.

## Combined submission implemented

`notebooks/kaggle_combined.py` and `.ipynb` embed the tested CPU features, frozen
Qwen extraction and `src/ntl/combined.py`. The historical 417 complementary
answers and exact 0.4093263733532189 score are preserved and checked by the
integration script. The inference rule matches the diagnostic, with explicit
class ordering and CPU-score tie breaking. The notebook repeats fixed five-fold
validation for the two fixed classifiers and blend, then fits both on all labels,
benchmarks actual full-data heads at 4,000-row scale, and writes dynamic Parquet
predictions. It needs competition data, the same official Qwen input, free GPU,
and Internet Off. See [exact Kaggle steps](combined.md).

The 49-test suite includes a complete synthetic 4,000-row execution of the
training/validation/output workflow. The separate real-weight CPU check uses
Qwen on 18 actual passages, a full-data CPU fit, and a 4,000-row prediction and
Parquet round trip. That small Qwen head does not establish full-head or GPU
runtime; see `reports/combined-v1/local_validation.json`. Modern Transformers
use `dtype=`; older versions retain the supported legacy keyword. No additional
GPU run, fine-tuning, purchase or competition submission was performed here.

## Combined Kaggle execution verified

The user ran the combined notebook successfully and supplied all six requested
reports. Independent reconstruction from the 2,699 aligned OOF rows and both
score matrices reproduces CPU **0.37560278399204566**, Qwen
**0.3511881680338056**, and combined **0.4093263733532189**, including **417**
Qwen-correct CPU errors. Both classifiers are beaten by the blend in all five
folds. Original data/fold hashes, labels, class recalls, confusion matrix and
all combined predictions match. The run took **15.85 minutes**; actual full-head
combined prediction on 4,000 cached feature rows took **27.88 seconds**, excluding
transformer extraction. The visible three-row submission passed notebook checks;
the Parquet itself was not uploaded or independently inspected here.

Six original reports and upload IDs are retained privately under
`runs/combined-kaggle-v1-received/`, with checksum-verified archive
`artifacts/combined-kaggle-v1-reports.zip`. Safe aggregates and hash receipts are
committed under `reports/combined-v1/`. Retaining the workspace/snapshot preserves
raw outputs; Git alone does not. See `kaggle_verification.json` there. Recommend
preserving both submitted versions as references. The user subsequently
submitted the combined notebook and reports **0.417 public balanced accuracy**,
up from **0.372** for CPU: **4.5 percentage points**. The combined private score
and exact notebook version number are unknown; the public score was not fetched
independently. See `reports/combined-v1/public_result.json`. No automatic submission
was performed. The extra dictionary printed after SUCCESS is the final function's
returned report displayed by Jupyter, not an error or another training run.

## Recommended next experiment

Preserve the 0.417 submission unchanged. Next evaluate frozen Qwen3-8B-Base with
the same prompt folds, token limit and classifier recipe before fine-tuning.
Verify the public model license/version, free GPU quota, two-GPU placement and
memory/runtime in a small extraction pilot first. Compare both standalone
performance and complementarity with existing models; the 417 earlier Qwen-only
correct predictions show why standalone accuracy is insufficient. A larger
model has no measured improvement here yet. Parameter-efficient fine-tuning of
the more promising backbone follows if resources and validation support it.

## Qwen8 pilot prepared; DeBERTa queued

The user authorized the frozen Qwen3-8B experiment and added DeBERTa testing.
`src/ntl/qwen8.py` and `notebooks/kaggle_qwen8.py/.ipynb` pin the official 8B
upstream revision/shards/config/index, split the model over two GPUs, reject
CPU/disk offload, and run a 16-longest-passage pilot. The conservative projection
includes 4,000 hidden rows and a classifier reserve, with a default two-hour
experiment budget. Extraction checks elapsed time between batches and preserves
completed feature checkpoints. A Python stop does not terminate the GPU session.

Shared pooling now handles hidden states across GPUs and moves masks/indices to
the correct device; input tensors use the embedding device. Existing 1.7B
defaults are preserved. Core tests (53) pass. A random tiny 36-layer Qwen3 checks
layer selection, zero-budget stop and cache resume. The actual 1.7B integration
recheck still reproduces the historical blend score and 417 count. Real 8B
weights, dual-GPU dispatch, peak memory, speed and CV remain unrun in this CPU
workspace. See `reports/qwen8-v1/local_architecture_check.json`,
[Kaggle steps](qwen8.md) and [the DeBERTa queue](experiment-queue.md).

## Qwen8 Kaggle result verified

The user supplied all seven requested files and a SUCCESS log. Original hashes,
2,699 unique row IDs, truths, folds, all 18 classes, pinned model revision/shard
hashes, both finite score matrices, all rank-blend predictions and reconstructed
recalls/confusion matrices match. The original CPU scores are unchanged.
Qwen8 alone scored **0.3511633109619687**, CPU + Qwen8
**0.4070966940094457**, versus **0.4093263733532189** with Qwen1.7. The new
blend improves on CPU in every fold but is 0.22 points below the old blend
overall. Conditional paired-prompt bootstrap interval for that difference is
approximately **-1.52 to +0.96 percentage points**, so this is not an established
accuracy difference. Qwen8 fixes **423 CPU errors**, compared with the historical
417 for Qwen1.7; more unique fixes alone do not establish a better ensemble.

The pilot passed, full run took **16.94 minutes**, training extraction **430.85
seconds**, and full-head classifier prediction on 4,000 cached feature rows
**45.76 seconds**, excluding extraction. No training text was truncated. The
three-row Parquet passed notebook checks; it was not uploaded/inspected here.

A single equal-rank three-way diagnostic (CPU, Qwen1.7, Qwen8; CPU score on
ties) scored **0.41117325379070346** on reused OOF rows. This +0.18-point gain
has conditional interval roughly **-1.19 to +1.63 points**, excluding refitting
and model-selection uncertainty. It is exploratory, not implemented/tested for
full inference or submitted, and does not yet justify added deployment cost.
Preserve the actual **0.417** public submission; DeBERTa fine-tuning is the next
queued experiment. No Qwen8 leaderboard score is known.

Original seven files, source IDs and checksums are retained privately under
`runs/qwen8-kaggle-v1-received/`, with a verified separate ZIP at
`artifacts/qwen8-kaggle-v1-reports.zip`. Aggregates, versions, timing and checksums
are committed under `reports/qwen8-v1/kaggle_*` and `received_files.json`.
Raw outputs require retaining the workspace/snapshot; Git stores aggregates
only. See `reports/qwen8-v1/kaggle_verification.json` for the verified findings.

## DeBERTa fine-tuning experiment implemented

The user authorized building this next experiment. `src/ntl/deberta.py` and
`notebooks/kaggle_deberta.py/.ipynb` implement full encoder/head fine-tuning,
masked mean pooling and 18-class cross-entropy. The original outer prompt folds
are unchanged. Inner training prompts select the epoch count (maximum 3,
patience 1); a fresh pretrained outer refit then predicts outer-held prompts.
Class weights are learned from each training partition only. Settings, original
hashes, library versions and the first-download revision/weight/tokenizer hashes
are recorded. A six-hour budget, disposable training pilot, FP16 gradient
scaling, gradient checkpointing and completed-fold checksummed resume are added.
Missing pretrained encoder weights, nonfinite scores or all-skipped optimizer
steps stop the run rather than produce an apparent result.

This is a training experiment with an initial Internet-On official Microsoft
download, not the offline competition notebook. If required,
`notebooks/kaggle_deberta_prepare.py` prepares a verified checkpoint separately
without competition input or GPU. No full-data final fit or submission.parquet
is generated until results justify deploying a final model. Preserve the 0.417
submission and the historical 417 complementary Qwen predictions.

Core 58 tests pass, including inner prompt isolation, ties, official-download
checksum protocol and corrupt-cache rejection. The optional real-architecture
CPU test updates both encoder and head of a tiny random DeBERTa-v2 model,
checks gradient checkpointing, safe checkpoint round-trip prediction equality
and the deadline guard. It is not pretrained-large evaluation. Cloud requests
to Hugging Face and Kaggle currently fail with proxy 403 (including the
escalated read-only attempt); official pretrained/tokenizer and GPU checks
therefore await Kaggle's runtime. See `reports/deberta-v1/local_training_check.json`
and [exact steps](deberta.md). No paid compute or remote training has been launched.

## DeBERTa pilot precision failure fixed

The user's first pretrained-large pilot loaded and verified the public weights
but stopped at `GradScaler.unscale_` with FP16 trainable gradients. No completed
DeBERTa outer fold or accuracy result exists. The loader now explicitly requests
FP32, converts the whole encoder/head module to FP32 and guards the dtype before
optimizer setup, keeping FP16 autocast and scaling for computation. A CPU AMP
regression reproduces the original failure with half parameters and passes with
the corrected loader on a half-precision saved checkpoint.

The user's Transformers 5.16 tokenizer also emitted an unrelated Mistral regex
warning. New code uses an original-SentencePiece adapter that matches the legacy
HF DeBERTa slow tokenizer on Unicode, whitespace, special tokens, padding and
truncation. It does not apply a Mistral-specific regex alteration. SentencePiece
0.2.1 is now in the optional environment lockfile. The unused pretraining heads
and unauthenticated HF rate-limit warning require no model/authentication fix.

New outputs use `deberta-v2/`; verified old downloads are reused if present.
Actual CUDA pilot retry is still required; see `reports/deberta-v2/` for CPU
regression evidence and [rerun instructions](deberta.md). The 0.417 model stays
unchanged and no valid DeBERTa competition score is claimed.

## DeBERTa timing pilot calibrated after intentional budget stop

The user retry passed the precision/tokenizer fixes and completed 18 optimizer
steps with no skipped steps, using 8.11 GiB. The old conservative estimate was
29,625.89 seconds (8.23 hours), above the 21,600-second budget. This was an
intentional budget stop, not a user mistake or failed backward pass. No outer
validation training was started. See `reports/deberta-v2/budget_stop_user_report.json`.

The old pilot timed mainly long examples and used accumulation=1, unlike the
actual accumulation=8. Updated code separates the long-text memory test from a
reproducible 128-row representative throughput sample, and uses production batch/
accumulation settings in both. Runtime projection retains 50% overhead plus a
20-minute reserve and now counts exact inner-training row counts. The six-hour
guard, model, seeds and nested validation procedure are unchanged. A predicted
over-budget run prints PILOT STOP and returns normally, never printing SUCCESS.
Actual new GPU throughput remains unmeasured. The 60-test suite and tiny genuine
DeBERTa training check cover sampling/class coverage, production pilot settings,
budget arithmetic and both pilot stages. Keep the successful 0.417 submission.

## DeBERTa completion reported; raw files pending recovery

The user pasted a completed five-fold SUCCESS block: **0.32083768332090484**
grouped balanced accuracy in **146.89593964078333 minutes** (about 2h27m),
selected epochs **[3, 2, 3, 3, 3]**, and fold 4 score **0.2981481481481481**.
See `reports/deberta-v2/user_completion_result.json`. This is a user-reported
completed run, not independently reconstructed from predictions. Folds 0–3
scores and report files have not been received. The user returned after a long
absence, found a new session and no working files; temporary outputs may be lost.
Check saved-version outputs and preserve console scores before any rerun.

This configuration scores below standalone CPU (37.56%), frozen Qwen (~35.12%),
and the existing blend (40.93%) on the nominal same outer folds. Preserve the
0.417 public submission. A lower standalone score does not prove no ensemble
value; assessing complementarity needs the missing OOF predictions/scores.
Four folds selected the maximum three epochs, so inner learning curves would be
useful, but extra training is not proven to help. Do not blindly repeat the same
2h27m run just to recover reports. For future long runs use a saved background
Save & Run All version directly so output artifacts persist after completion.
