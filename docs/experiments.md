# Experiment plan

1. Obtain authorized official CSVs and confirm competition rules. Audit raw data,
   labels, lengths, class counts, duplicates and inconsistent labels. Inspect
   representative passages locally without publishing restricted text.
2. Create and version five prompt-grouped folds once. Exact normalized duplicate
   passages connect their prompt groups. Use the same fold file for all initial
   candidates. Random row splitting is not a main validation method.
3. Evaluate the five CPU candidates. Begin with `word_lr` to establish the first
   result. Inspect convergence warnings before trusting a run. Record pooled and
   fold balanced accuracy, individual recalls, confused pairs, length effects,
   run time, memory and versions. Compare word/character complementarity.
4. Use error analysis to propose a small number of changes. Preserve a fresh
   group holdout or use nested grouped evaluation before substantial tuning.
   Fold-to-fold variation matters more than a tiny headline gain.
5. If rules and free compute permit, test a licensed frozen encoder with simple
   balanced linear classifiers. Record exact model revision, tokenizer, pooling,
   normalization, truncation, and precision. Batch extraction and checkpoint
   embeddings by passage ID and text/model checksum. Fit projections, centering,
   selection, scaling, calibration and classifiers within each training fold.
   Do not use training prompt_id at hidden-test inference.
6. Investigate topic reduction and provider hierarchy only against the direct
   18-class baseline, using unchanged folds. Record failed hypotheses as well as
   gains. Do not assume shared architecture or provider proves authorship style.
7. Consider transformer fine-tuning only after measuring free GPU capacity,
   training cost, and simpler results. Use checkpoints and early stopping inside
   the training partition, with a separate inner prompt-grouped split.
8. Ensemble only when held-out errors are complementary. Learn weights and
   calibration on inner grouped validation, evaluate once on outer groups. Raw
   SVM margins are not directly comparable calibrated probabilities.
9. Select the pipeline and refit on all eligible training data. Save manifest,
   code version, checksum, packages, and selected configuration. Validate an
   offline wheel/model bundle and benchmark about 4,000 representative passages.
10. Confirm official paths/schema and run the notebook with Internet disabled on
    Kaggle's actual runtime. Inspect the Parquet and runtime log before the user
    submits. Use public leaderboard feedback sparingly; retain private validation.

Do not scrape extra passages, call paid APIs, rent compute, or upload restricted
data automatically. Synthetic fixtures are software tests and never extra model
training data for competition claims.
