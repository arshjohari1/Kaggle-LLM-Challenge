# Experiment queue

1. **Completed: frozen Qwen3-8B-Base + fixed SVM, and its CPU rank blend.**
   Pilot passed; user run took 16.94 minutes. Standalone 35.12%, CPU blend 40.71%,
   versus 40.93% with 1.7B. All seven reports verified and retained privately.
   No clear improvement; preserve the existing 0.417 public submission. An
   exploratory three-model fixed rank blend scored 41.12% on reused OOF rows;
   its +0.18-point gain is inconclusive and does not yet justify deployment.
2. **Next queued: DeBERTa-v3-large author classification fine-tuning.** Add an 18-class
   head and evaluate within the same outer prompt folds. Early stopping and
   checkpoint selection use inner training prompt groups; outer validation is
   untouched. Frozen DeBERTa alone should not decide whether this supervised
   experiment is worthwhile. Audit exact weights, license, stack and compute
   before building. No DeBERTa GPU training has been run.
3. **Conditional: another family or LoRA fine-tuning of a promising generative
   backbone.** Decide from measured accuracy, complementary errors, memory,
   runtime and remaining free quota. Preserve the 0.417 submission. No spending
   is authorized; the public catalog shortlist is not verified GPU feasibility.

The historical 417 correct Qwen answers on CPU errors and exact 40.93% grouped
blend score remain comparison evidence, not a promised score for another model.
