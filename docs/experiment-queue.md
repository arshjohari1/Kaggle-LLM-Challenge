# Experiment queue

1. **Completed: frozen Qwen3-8B-Base + fixed SVM, and its CPU rank blend.**
   Pilot passed; user run took 16.94 minutes. Standalone 35.12%, CPU blend 40.71%,
   versus 40.93% with 1.7B. All seven reports verified and retained privately.
   No clear improvement; preserve the existing 0.417 public submission. An
   exploratory three-model fixed rank blend scored 41.12% on reused OOF rows;
   its +0.18-point gain is inconclusive and does not yet justify deployment.
2. **Completed, console result only: DeBERTa-v3-large author classification fine-tuning.**
   User reports **32.08% grouped balanced accuracy**, 146.90 minutes, selected
   epochs [3,2,3,3,3]. Original files are missing from the new session; inspect
   saved versions before rerunning. No independent OOF verification or ensemble
   complementarity measurement yet, and no DeBERTa leaderboard submission.
   Pretrained encoder + masked mean + 18-class head, updating encoder and head.
   Inner training prompt groups select 1–3 epochs, then fresh outer refits
   evaluate the original five held-out folds. One-GPU FP16/checkpointing pilot,
   six-hour guard and completed-fold resume are implemented. Official Microsoft
   source/license/revision/hashes are verified at first Kaggle download. The
   cloud proxy blocks pretrained downloads; tiny architecture training tests
   pass, but pretrained/GPU execution remains unrun. See [instructions](deberta.md).
   The first user pilot later loaded the checkpoint but failed on FP16 gradient
   unscaling. The v2 precision/original-SentencePiece update is implemented and
   CPU-regression-tested; the subsequent revised timing pilot allowed the
   completed run reported above. Keep the historical sequence separate from
   current completion status.
3. **Conditional: another family or LoRA fine-tuning of a promising generative
   backbone.** Decide from measured accuracy, complementary errors, memory,
   runtime and remaining free quota. Preserve the 0.417 submission. No spending
   is authorized; the public catalog shortlist is not verified GPU feasibility.

The historical 417 correct Qwen answers on CPU errors and exact 40.93% grouped
blend score remain comparison evidence, not a promised score for another model.
