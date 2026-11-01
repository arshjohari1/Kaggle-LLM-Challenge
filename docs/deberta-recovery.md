# Regenerate missing DeBERTa reports with a saved run

The original experiment completed at **32.0838% grouped balanced accuracy** in
146.90 minutes, but the user confirms **zero saved versions** and missing
interactive-session files. All five aggregate fold scores and the original
inner-selected epochs **[3, 2, 3, 3, 3]** are preserved in
`reports/deberta-v2/user_completion_result.json`. Original passage-level
predictions and the exact checkpoint/package manifest are unavailable.

This recovery mode retrains only the five outer models for those original epoch
counts. It skips repeating the inner epoch-selection fits. These choices came
from the original training-side prompt groups; no epoch is chosen from outer
accuracy. Original data/fold hashes, seeds, pooling, learning rates, optimizer,
class weights and precision procedure remain unchanged. There is no new tuning
or guaranteed improvement. Model revision/packages and numerical variation may
prevent exact reproduction of 32.0838%; the new provenance is fully recorded.

## Start a saved background run directly

1. Keep the 0.417 competition submission unchanged. Create a **new private**
   notebook named **DeBERTa report recovery**.
2. Copy [kaggle_deberta_recovery.py](https://github.com/arshjohari1/Kaggle-LLM-Challenge/blob/main/notebooks/kaggle_deberta_recovery.py)
   using **Copy raw file**, and paste the entire file into one code cell.
   Alternatively import `notebooks/kaggle_deberta_recovery.ipynb`.
3. Attach **Name That LLM competition data**. Select a **free GPU** (T4 ×2 is
   suitable; this code uses GPU 0) and **Internet On**. No Qwen input is needed.
   The checkpoint is downloaded/verified from the official Microsoft source.
   The existing SentencePiece preflight and optional offline-preparation route
   are unchanged; see [the main guide](deberta.md) if dependencies are missing.
4. **Do not perform an interactive Run All first.** Click **Save Version →
   Save & Run All** directly, with GPU enabled in the saved-run accelerator
   setting. This is a training version, not a competition submission.
5. Follow the saved job's status/logs from the version or Active Events view.
   It prints **RECOVERY MODE**, then the memory/timing pilot and five outer-fold
   results. There are no new inner-selection histories. The six-hour guard
   remains; the pilot estimates only the remaining outer-refit workload.
6. You may leave the saved background job running without keeping the browser
   open. Its GPU is released when that job ends. A separate interactive draft
   GPU session, if started, should be stopped separately.
7. After the saved version finishes successfully, open **that completed
   version's Output** and download:

   ```text
   deberta-recovery-v1-reports.zip
   ```

   Upload that one ZIP here, together with **RESULT TO SHARE**. It contains the
   eight small reports plus a checksum receipt. It excludes the large model
   checkpoints. The reports also appear under `deberta-recovery-v1/`.
8. If the saved run fails, send its final error or **PILOT STOP** and pilot.json
   if available. Do not blindly resubmit or start a duplicate interactive run.

No `submission.parquet` is created. We will verify OOF predictions, analyze errors
and test complementarity before deciding whether DeBERTa belongs in a submission.
The five model.pt files are large optional assets; do not upload them to chat.

## Verification

The generator derives recovery epochs from the preserved original result file.
Tests reject malformed epoch counts, verify the exact generated recovery entry
point, and read back every report ZIP member against SHA-256 receipts. Existing
gradient, precision, tokenizer and training tests still apply. No new recovery
GPU execution has been performed in this CPU-only workspace. It should cost less
than the original full inner-plus-outer experiment, but runtime is unmeasured.
