# Experiment queue

1. **Completed: frozen Qwen3-8B-Base + fixed SVM, and its CPU rank blend.**
   Pilot passed; user run took 16.94 minutes. Standalone 35.12%, CPU blend 40.71%,
   versus 40.93% with 1.7B. All seven reports verified and retained privately.
   No clear improvement; preserve the existing 0.417 public submission. An
   exploratory three-model fixed rank blend scored 41.12% on reused OOF rows;
   its +0.18-point gain is inconclusive and does not yet justify deployment.
2. **Active, implemented: DeBERTa-v3-large author classification fine-tuning.**
   Pretrained encoder + masked mean + 18-class head, updating encoder and head.
   Inner training prompt groups select 1–3 epochs, then fresh outer refits
   evaluate the original five held-out folds. One-GPU FP16/checkpointing pilot,
   six-hour guard and completed-fold resume are implemented. Official Microsoft
   source/license/revision/hashes are verified at first Kaggle download. The
   cloud proxy blocks pretrained downloads; tiny architecture training tests
   pass, but pretrained/GPU execution remains unrun. See [instructions](deberta.md).
3. **Conditional: another family or LoRA fine-tuning of a promising generative
   backbone.** Decide from measured accuracy, complementary errors, memory,
   runtime and remaining free quota. Preserve the 0.417 submission. No spending
   is authorized; the public catalog shortlist is not verified GPU feasibility.

The historical 417 correct Qwen answers on CPU errors and exact 40.93% grouped
blend score remain comparison evidence, not a promised score for another model.
