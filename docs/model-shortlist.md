# Candidate screening and compute constraints

Checked October 24, 2026 against Kaggle's public model listing API. Exact
variants, versions, frameworks, catalog fineTunable flags, license names and
download sizes are in `reports/model-shortlist-v1/verified_candidates.json`.
Download size is not peak GPU memory or a parameter count. No listed candidate
has been benchmarked by this project except the existing Qwen3-1.7B control.

## Proposed shortlist

First compare Qwen3-8B-Base to the existing frozen Qwen3-1.7B-Base using the same
prompt groups and classifier recipe. Additional distinct candidates are
Qwen3.5-4B-Base (or 9B-Base), Gemma 4 12B, and DeBERTa-v3-large. Gemma 3 4B
pretrained and Llama 3.1 8B are alternative family controls. Qwen3-14B-Base is
a resource-dependent scale test; Qwen3-32B and Gemma 4 31B are more expensive
quantized stretch experiments, not first priorities. The Keras Qwen3.5 and
DeBERTa listings need a compatible extraction/training stack; they are not
drop-in replacements for the existing PyTorch script. Licenses/access and
runtime compatibility must be checked for the exact artifacts before use.

Screen roughly 4–6 backbones total including the existing 1.7B control, rather
than searching the entire catalog. A suggested five-model screen is 1.7B,
Qwen3-8B-Base, Qwen3.5-4B-Base, Gemma 4 12B (Gemma 3 4B as memory fallback),
and DeBERTa-v3-large. Fine-tune one or two promising candidates, not all five.
Frozen ranking does not necessarily predict fine-tuned ranking: a small
supervised encoder deserves a short grouped fine-tuning pilot even if its
frozen result is modest. Prompt groups for early stopping/checkpoint selection
must be inside each outer training partition. Include paired errors and gains
in the ensemble, not just standalone balanced accuracy. Avoid large repeated
parameter sweeps on reused folds; strengthen validation for close decisions.

## Why not pick the largest model?

The current accelerator is two T4 GPUs with approximately 16 GiB each. Two
devices do not automatically become one 32 GiB pool: placement/sharding and
inter-device communication matter. At 2 bytes per parameter, an 8B model needs
roughly 16 GB for weights alone; a 32B model roughly 64 GB. Activations and
fine-tuning state add memory. Quantization and LoRA/QLoRA can reduce memory,
but need compatible packages, hardware checks and measured quality/runtime.
Very large mixture-of-experts models retain all expert weights even when only
a small fraction are active for each token.

More reasoning/chat capability does not establish better author fingerprints.
Our frozen 1.7B model was worse standalone than TF-IDF but complementary on 417
CPU errors. New backbones should be selected by grouped balanced accuracy and
ensemble improvement. Only 2,699 labeled passages are available; a large model
can overfit after fine-tuning. The deployed pipeline must still run offline on
Kaggle within the competition's nine-hour limit. Paid training elsewhere does
not lift hidden-test inference restrictions.

## Paid compute information, not spending authorization

Kaggle's current [Notebook documentation](https://www.kaggle.com/docs/notebooks)
describes an **experimental, limited-time** Colab subscription linking promotion:
Colab Pro adds 15 Kaggle GPU hours/week and Pro+ adds 30. The documentation
says these use the same Kaggle hardware; more hours do not grant a larger GPU.
Account availability and current pricing require checking before purchase.
External rented GPUs could provide more memory for training, but final Kaggle
inference must still fit the competition environment. The budget remains $0;
no subscription, rental or API request has been purchased or scheduled.

The linked public listing pages establish availability and catalog metadata,
not account-specific access, measured task accuracy or verified fine-tuning fit.
Start with memory/runtime pilots and remaining free quota before spending.
