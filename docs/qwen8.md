# Frozen Qwen3-8B experiment

**Run completed:** the user supplied all seven requested reports. They reproduce
35.12% standalone Qwen8 and 40.71% CPU + Qwen8 grouped balanced accuracy, with
423 Qwen-correct CPU errors. The run took 16.94 minutes, no passages were
truncated, and the pilot passed. This does not clearly improve on the 40.93%
1.7B blend. The existing 0.417 public model remains the reference; no Qwen8
leaderboard score is known. See
[`kaggle_verification.json`](../reports/qwen8-v1/kaggle_verification.json).

Keep the successful CPU + Qwen1.7 notebook and **0.417 public submission**.
The earlier Qwen model correctly identified **417 CPU errors**; this motivates
testing complementary features. It does not guarantee 8B will improve on it.
DeBERTa-v3-large supervised fine-tuning remains a queued experiment; no DeBERTa
training, paid compute or API data generation is part of this run.

## Run now on Kaggle

1. Create a **new** private competition notebook named **Name That LLM Qwen8 v1**.
2. Open [the new Python file](https://github.com/arshjohari1/Kaggle-LLM-Challenge/blob/main/notebooks/kaggle_qwen8.py),
   choose **Copy raw file**, and paste it into one code cell. Remove other code.
3. **Add Input → Competitions → Name That LLM → Add**. Confirm train.csv/test.csv.
4. **Add Input → Models**: attach the official
   [qwen-lm/qwen-3 → Transformers → 8b-base → version 1](https://www.kaggle.com/models/qwen-lm/qwen-3/transformers/8b-base/1).
   This experiment uses **8b-base**, not 8b instruct, 1.7b-base or a quantized copy.
   The model input is about 16 GB; mounting it may take time. No manual weight
   download/upload is needed. Attach only this backbone and the competition data.
5. Set **Accelerator: GPU T4 ×2**, **Internet: Off**, then **Run All**.
   Both GPUs need at least 12 GiB free; a clean session is recommended. No pip
   installs or secret keys are needed. The runtime needs Qwen3-capable Transformers
   and Accelerate; missing dependencies produce a specific error for offline repair.
6. Expect five weight-shard verification lines, then **PILOT RESULT**. The first
   16 longest training passages test finite features and memory. A conservative
   estimate covers training + 4,000 hidden passages and reserves 30 minutes for
   classifier work. The default experiment budget is **two hours**, not an
   expected duration. Pilot extraction itself is capped at 10 minutes.
7. If the pilot passes, the notebook continues to the full experiment. If it
   fails, save `qwen8-v1/pilot.json` if present and copy the error, then **Stop
   Session**. A Python stop does not release the GPU allocation automatically.
   Do not repeatedly restart a failed memory or budget check without diagnosis.
8. On success, send **RESULT TO SHARE** and these files from
   `/kaggle/working/qwen8-v1/`:

   ```text
   pilot.json
   summary.json
   validation.json
   oof.csv
   oof_scores.npz
   per_class.csv
   confusion.csv
   ```

   Download them before stopping the interactive GPU session. The final line is
   `SUCCESS: /kaggle/working/submission.parquet with 3 rows. Combined model run complete.`
   Three is the visible sample size; hidden inference handles arbitrary row counts.
   Review the experiment results before saving another full run or submitting.

## What changes and what stays fixed

The pinned **Qwen3-8B-Base** backbone has 36 layers and hidden size 4096. Features
are pooled at layers **18, 27, 36**, plus the final real token. Only the middle
mean is used by the fixed classifier, matching the relative layer choice in the
1.7B experiment. Layer fractions, raw text, token limit 512, padding exclusion,
batch size 2, classifier/scaler recipe, CPU pipeline, rank rule and original
five prompt folds are unchanged. No fine-tuning, label-fitted blending or
pooling/hyperparameter sweep is performed. The run evaluates standalone CPU,
standalone Qwen8, and CPU + Qwen8 on identical held-out prompts, then refits all
labels and writes a dynamic offline submission. It does not run a three-model
ensemble with the old Qwen1.7 backbone yet.

The model is sharded across two GPUs using Accelerate, with a 3 GiB reserve per
GPU. CPU/disk offload is rejected. Pooling moves masks and indices to each hidden
state's device before returning the small vectors to the input device. T4 uses
FP16; native BF16-capable GPUs use BF16. The old Kaggle run reported BF16 on T4,
so this numerical precision detail also differs and is recorded in provenance.
No quantization is used. Feature caching is model/settings/data-specific, resumes
flushed batches, and observes the extraction deadline. CV/classifier operations
are not forcibly interrupted by that deadline; the runtime estimate includes a
reserve for them. It is a guard, not a guarantee of total wall time or GPU quota.

## Provenance and local checks

Apache-2.0 official model, upstream revision
`49e3418fbbbca6ecbdf9608b4d22e5a407081db4`. Five shard SHA-256 values and config/index
Git blob hashes are pinned from the official Hugging Face API. Every shard and
configuration/index is verified before loading. Actual tokenizer hashes, model
device placement, GPU names, dtype and package versions are recorded.

This workspace has no GPU. **Full 8B loading and execution remain unrun here;
the later user-supplied Kaggle reports verify the external run as noted above.** Unit tests cover memory
planning, CPU/disk-offload rejection, model identity, shard-index paths, hidden
test runtime projection and generated script composition. A random tiny 36-layer
Qwen3 check exercises pooling, extraction deadlines and cache reuse on CPU; it
is not pretrained 8B evaluation. The existing actual 1.7B weights also recheck
the shared extraction and combination path. Any local 4,000-row check involving
a small head remains functional validation, not full 8B runtime measurement.

Generation: `python scripts/build_generative_notebook.py` and
`python scripts/build_qwen8_notebook.py`. Core tests: `python -m pytest -q`.
Optional Torch checks: `python scripts/check_qwen8_torch.py` and
`python scripts/check_combined.py` using the optional embeddings environment.
