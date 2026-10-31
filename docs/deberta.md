# DeBERTa-v3-large fine-tuning experiment

**Precision/tokenizer update (v2):** the initial user pilot failed at gradient
unscaling because the newer loader used half-precision parameters. Training now
explicitly loads/converts all trainable parameters to FP32 and checks them before
the optimizer is built; autocast still handles mixed-precision computation.
The tokenizer uses the original `spm.model` directly, avoiding the newer fast
backend's Mistral regex warning. No Mistral-specific flag is applied.

Replace the entire Kaggle code cell with the latest GitHub Python file and
restart the kernel/session before Run All to release the old model from memory.
Keep Internet On and the same competition input/GPU. New reports go to
`deberta-v2/`, avoiding a stale v1 manifest. The verified download at
`deberta-v1/pretrained/` is reused if it survived the session restart; otherwise
the official checkpoint is downloaded and verified again. No token is needed for
the public model. Unused `lm_predictions`/`mask_predictions` head messages are
expected because the encoder receives a new 18-author head.

This is a new **training experiment**, not a competition submission notebook.
Keep the successful **0.417** CPU/Qwen submission unchanged. No DeBERTa accuracy,
full pretrained-model backward pass, GPU memory or runtime has been measured yet.

## Model and training

DeBERTa is a bidirectional encoder transformer. Its attention represents content
and relative positions separately ("disentangled attention"). The v3 family was
pretrained using replaced-token detection: learning to distinguish original
tokens from replacements. This is background about pretraining, not a loss we
implement here. We reuse those pretrained language weights for author detection.
DeBERTa is itself a transformer; it is not a category separate from Qwen.

The selected large encoder has 24 layers and hidden size 1024, roughly 0.4B
parameters including its vocabulary embeddings. It outputs contextual token
vectors. We exclude padding, average token vectors, apply dropout, and use a
new linear layer with 18 outputs. Cross-entropy compares these scores to known
author labels. Backpropagation updates both the new head and all encoder weights.
The original SentencePiece model stays fixed and capitalization/punctuation are retained.
There is no text generation, paid API data, LoRA, frozen SVM or Qwen in this run.

Compared with TF-IDF, the encoder can learn contextual features. Compared with
our frozen Qwen experiments, those features adapt to the author task instead of
leaving all language weights unchanged. Its smaller size makes supervised
training more practical than fully updating Qwen8. Better accuracy is still a
hypothesis; overfitting the small data or prompt topics is a serious risk.

The fixed first configuration uses encoder LR 1e-5, head LR 1e-4, AdamW weight
decay .01, clipping 1.0, 10% warmup, dropout .1, batch size 2 with 8-step gradient
accumulation, token limit 512 and at most three epochs. Gradient checkpointing
and FP16 autocast/gradient scaling reduce computation memory while all trainable parameters and gradients remain FP32. It uses GPU 0 only. The
second T4 is not used; parallel fold training is deferred until fit/cost is known.

## Honest validation

Original data and five prompt-fold hashes must match. For each outer fold:

1. Set the held-out outer prompts aside.
2. Select 20% of the remaining training prompts for inner epoch selection.
3. Fine-tune on the other training prompts, evaluating inner balanced accuracy
   after each epoch. Select the earliest best epoch; stop after one epoch without
   improvement, up to three epochs.
4. Discard those weights. Start again from the same pretrained source and train
   on all outer-training rows for the selected epoch count.
5. Predict the untouched outer fold exactly once for the final OOF report.

Every training partition computes its own class weights. Tokenization is fixed,
so its cached encoding can be reused. No ensemble is learned or evaluated in
this notebook yet. We will compare OOF errors with the preserved CPU/Qwen results.
All compared configurations reuse folds, so future repeated selection can still
overfit validation. Inner selection prevents outer checkpoint-selection leakage;
it does not remove all model-selection optimism.

## Kaggle instructions

1. Create a **new private notebook** named **Name That LLM DeBERTa v1**.
2. Copy [this Python file](https://github.com/arshjohari1/Kaggle-LLM-Challenge/blob/main/notebooks/kaggle_deberta.py)
   using **Copy raw file**, and paste into one code cell. Alternatively import
   `notebooks/kaggle_deberta.ipynb`.
3. **Add Input → Competitions → Name That LLM → Add**. No Qwen model input is
   required. This experiment downloads Microsoft's original
   [microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large)
   instead of requiring a community conversion or Keras-format checkpoint.
4. Select a **free GPU** (T4 ×2 is suitable, with GPU 0 used), and **Internet On**
   for the first model download. This is allowed only as a training experiment;
   it is not the final offline scoring notebook. If Kaggle does not permit
   Internet with this competition input, use the preparation notebook described
   below to create an attached checkpoint first, then run fine-tuning offline.
5. If the preflight says SentencePiece is missing, run this in a separate cell:

   ```python
   %pip install sentencepiece==0.2.1
   ```

   Restart the session, then Run All. This is a free tokenizer dependency. The
   experiment does not install online packages automatically or weaken TLS/checksum
   checks. If you attach the prepared offline wheel described below, it can install
   that local checksum-verified wheel without Internet.
6. **Run All**. Official Hugging Face metadata resolves a fixed repository
   revision and checks its license before download. LFS SHA-256/Git blob hashes
   verify every selected resource. Model and tokenizer are then loaded locally
   with remote code disabled; pickle weights, if needed, require safe weights-only
   loading and modern PyTorch. The receipt records exact hashes/revision/versions.
7. Wait for **PILOT RESULT**. Longest examples plus coverage of all 18 classes
   exercise actual backpropagation. Those pilot weights are discarded. The
   estimate includes five inner searches, five outer refits and evaluation
   overhead. If it exceeds the **six-hour experiment budget**, training stops
   before the full folds. Copy the report/error, save outputs, and **Stop Session**.
   The budget is a training guard, not a promised duration or a hard GPU-allocation
   limit. Checks occur between batches; Python does not release the GPU session
   automatically.
8. If it passes, each epoch prints loss, inner accuracy where applicable, actual
   optimizer updates and skipped FP16 steps. Each completed outer fold prints
   **OUTER FOLD RESULT** and writes a checkpoint/score checksum receipt. Completed
   folds can resume in the same session with matching settings. An interrupted
   incomplete fold starts again from pretrained weights; no partial fold score
   is presented as full validation.
9. Finish at **RESULT TO SHARE** and
   **SUCCESS: DeBERTa five-fold experiment complete.** Download and send:

   ```text
   /kaggle/working/deberta-v2/
       pilot.json
       summary.json
       manifest.json
       tokenization.json
       oof.csv
       oof_scores.npz
       per_class.csv
       confusion.csv
   ```

   Save these before stopping the interactive GPU session. If it fails, send
   the last error, pilot.json if present, and any fold progress reports instead.
   Do not submit this training notebook; it creates **no submission.parquet**.

The `fold-N/model.pt` files are large optional CV checkpoints (about 1.7 GB each
at FP32); do not upload them to this chat. Reports suffice for error analysis.
Session files are temporary unless retained separately. After assessing results,
we can train one final model on all labels for an epoch count derived from these
training-side selections, then prepare a separate offline inference notebook.
This avoids paying GPU time for a final fit before knowing whether it helps.

## Offline checkpoint preparation if needed

Run `notebooks/kaggle_deberta_prepare.py` in a separate Kaggle notebook with
**Internet On**, **Accelerator None**, and **no competition input**. It downloads
and verifies the official checkpoint without training. Save its output as a
notebook version or a private model dataset, and attach that prepared output to
the training notebook. The training code finds exactly one `verified.json` next
to config.json and verifies its files before use. Preparation also downloads a
matching SentencePiece wheel with a hash receipt; attach that output as well.
Missing SentencePiece can then be installed locally without Internet. The
preparation/training Python versions must match. A Keras preset is not compatible
with this PyTorch experiment.

## Local validation and limits

The cloud proxy currently denies Hugging Face and Kaggle requests (403), so
official pretrained weights/tokenization and GPU execution could not be tested
here. No claim of real pretrained fine-tuning readiness is made before Kaggle's
pilot. The notebook checks the actual source/license/hashes at runtime; its
first successful receipt pins the precise revision for repeatable runs.

Tests verify inner prompt isolation, early-epoch ties, deadline checks and exact
standalone source embedding. `scripts/check_deberta_torch.py` uses a tiny random
DeBERTa-v2 encoder to perform backward updates to encoder and head, gradient
checkpointing, finite predictions, safe checkpoint reload equivalence and budget
stopping. A half-checkpoint regression reproduces the FP16 unscale failure and
verifies FP32 loading and a successful CPU autocast/GradScaler step. The original
SentencePiece adapter matches legacy HF DeBERTa token IDs/masks across Unicode,
specials, whitespace, padding and truncation. These are CPU checks; the actual
CUDA fix must still pass Kaggle's pilot. Synthetic scores are not competition performance. See
`reports/deberta-v2/local_training_check.json`. No paid service is used.
