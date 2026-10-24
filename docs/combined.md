# Combined CPU + frozen Qwen submission

Qwen correctly identified **417 passages missed by the CPU model** in our
original paired validation. The fixed rank blend measured **40.93% balanced
accuracy**, versus CPU's 37.56% and frozen Qwen's 35.12%. Those are model-selection
results on reused prompt folds; the blend has no leaderboard score yet.
The user's combined Kaggle run reproduced all three scores exactly in **15.85
minutes**, and all six supplied reports were verified against original data,
folds and the production ranking rule. This now supports a formal submission.

Each passage gets 18 decision scores from the CPU TF-IDF/style SVM and 18 from
the frozen Qwen middle-layer mean + linear SVM. We rank each set separately,
add their ranks with equal weights, and use CPU decision scores to break exact
ties. Class labels are explicitly aligned before combining. There is no fitted
blender, confidence threshold, calibration or weight search. The 417 passages
are evidence of complementarity; we cannot know which model is right at test
time. Both contribute to every prediction.

## What to do on Kaggle

1. Keep the successful CPU notebook and create a new notebook named
   **Name That LLM Combined v1** from the competition Code page.
2. Open [the new Python file](https://github.com/arshjohari1/Kaggle-LLM-Challenge/blob/main/notebooks/kaggle_combined.py),
   click **Copy raw file**, and paste it into one code cell. Use this file alone.
   Alternatively import `notebooks/kaggle_combined.ipynb`.
3. **Add Input → Competitions → Name That LLM → Add**. Confirm `train.csv` and
   `test.csv` appear. Also attach the same official model as before:
   [qwen-lm/qwen-3 → Transformers → 1.7b-base → version 1](https://www.kaggle.com/models/qwen-lm/qwen-3/transformers/1.7b-base/1).
4. Set **Accelerator: free GPU** (T4 or P100) and **Internet: Off**. No extra
   spending or paid API is needed. The loader uses modern `dtype=` where
   supported, with compatibility for older Transformers. Extraction precision,
   token limit and views match the earlier frozen experiment to keep comparisons
   meaningful. Cached features can be reused within the same session if present.
5. **Run All once**. It verifies the training data and original folds, extracts
   unchanged Qwen features, and evaluates the two fixed models and their blend
   on the five original prompt folds. Every CPU vectorizer/style scaler and
   Qwen classifier scaler fits on that fold's training partition only. It then
   refits both classifiers on all labeled rows, benchmarks their combined
   prediction at 4,000-row scale, and predicts the provided test rows.
6. Wait for **RESULT TO SHARE** and
   `SUCCESS: /kaggle/working/submission.parquet with 3 rows. Combined model run complete.`
   Three is correct for the visible test. Send the final block and download
   `combined-v1/summary.json` and `combined-v1/validation.json`. The new
   validation score may differ slightly with library versions or numerical
   precision. It is calculated from this run, rather than printing 40.93% as a
   fresh result. The first combined visible-data run took **15.85 minutes**;
   other sessions and hidden-test runs may take longer.
7. If successful and validation remains promising, **Save Version → Save & Run
   All**, wait for the completed version, then **Submit to Competition**. Fresh
   saved runs may repeat extraction and consume more GPU time. Kaggle's hidden
   rerun dynamically handles all test IDs and may take longer than the sample run.
8. Save the outputs you want before **Stop Session**. Stopping a cell or closing
   the tab may leave the interactive GPU allocated. Preserve your CPU submission
   as the reference with public score 0.372.

The output directory also holds OOF predictions, decision scores, per-class
recall, confusion matrices and both fitted classifiers. Keep these private.
No model training, account operation or competition submission is launched from
this cloud workspace. Full GPU execution requires the user's Kaggle session.

## Local evidence

The production rank function is checked against the original 2,699 OOF score
rows, reproducing exactly **0.4093263733532189** and verifying the **417** count.
Tests cover score ordering, incompatible classes, tied ranks, invalid scores,
the standalone script, and a complete synthetic 4,000-row train/validate/output
workflow. A separate CPU integration check uses the actual pretrained Qwen
weights on 18 real passages and a CPU classifier fitted on all 2,699 rows,
then writes and reads 4,000 predictions. That functional head is trained on only
18 passages; its timing excludes extraction and does not establish full-model
GPU or hidden-test runtime. See [the evidence](../reports/combined-v1/local_validation.json).
The subsequent user GPU run and uploaded reports are separately checked in
[`kaggle_verification.json`](../reports/combined-v1/kaggle_verification.json).
Jupyter may display the returned report dictionary after SUCCESS because
`run_combined()` is the cell's last expression. This is harmless and does not
repeat execution.

Reproduce with `python scripts/build_combined_notebook.py`, core `python -m
pytest -q`, and optional embeddings environment `python scripts/check_combined.py`.
The submitted notebook benchmarks the actual full-data heads at 4,000 rows.
