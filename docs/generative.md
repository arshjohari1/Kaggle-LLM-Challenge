# Frozen generative transformer: the next experiment

The existing CPU model scored **0.37560 in grouped local validation** and the
user's successful Kaggle submission scored **0.372 on the public leaderboard**.
These are separate measurements. The new transformer has no measured competition
score yet. Start with its frozen version; consider fine-tuning after evaluating it.

## What is changing

The new main body is **Qwen3-1.7B-Base**, a pretrained generative transformer with
1.7 billion parameters. It reads each passage and supplies numerical features.
Its weights stay unchanged, and it does not generate responses. An SVM learns to
map the features to the 18 author labels. This uses the supplied training data;
it does not add generated passages or make API calls.

We start at 1.7B because ordinary 16 GiB Kaggle GPUs can hold its unquantized
weights and inference batches. This is much larger than our earlier small BGE
encoder. An 8B model would require more memory or additional quantization setup;
we can evaluate that separately if the first results and remaining quota justify it.

The notebook compares masked average features from layers 14, 21 and 28, plus the
final real token from layer 28. Each feature view is evaluated with a linear and
an RBF SVM: eight configurations total. Padding tokens are excluded. Raw text,
case and punctuation are preserved; no chat template or instruction is added.
The token limit is 512 and truncation counts are printed. Each scaler, followed
by per-row L2 normalization and the SVM, is fitted on training folds only.

All responses to a prompt stay together in the original five folds. The training
CSV and reconstructed fold CSV must match the original SHA-256 values. No random
row split or validation-label fitting is used. Frozen representations can be
extracted once because their parameters are never learned from these passages.
The best of eight results is a model-selection result, with some selection
optimism; it is not a guaranteed leaderboard score.

## Exact Kaggle steps

1. Keep your successful CPU notebook. Create a **new notebook** from the
   [competition Code page](https://www.kaggle.com/competitions/name-that-llm/code),
   named **Name That LLM Frozen Qwen v1**. Keep it private while experimenting.
2. Open the [copyable Python file on GitHub](https://github.com/arshjohari1/Kaggle-LLM-Challenge/blob/main/notebooks/kaggle_qwen_frozen.py).
   Click **Copy raw file**, the same overlapping-squares icon used before.
   Paste the entire file into one code cell and remove unused starter cells.
   Alternatively import the matching `.ipynb` from `notebooks/`.
3. Under **Inputs**, confirm **Name That LLM competition data** is attached.
4. Choose **Add Input → Models**, search for **Qwen 3** by **qwen-lm**, and
   attach **Transformers → 1.7b-base → version 1**.
   [Exact model page](https://www.kaggle.com/models/qwen-lm/qwen-3/transformers/1.7b-base/1).
   Choose the **base** variant, not an instruct, embedding or speech variant.
   Attach only one copy. You do not need to download/upload its weights yourself.
5. In notebook/session **Settings**, select an available **free GPU**
   (**T4 ×2** or **P100** is suitable), and set **Internet: Off**. The code
   uses the first GPU; it does not require two GPUs. The reported
   **00:00 / 30:00** means zero hours used out of thirty. Benchmarks AI dollar
   credits are a different resource. No paid resource is authorized.
6. Click **Run All once**. It prints GPU name/free memory, verifies weights,
   extracts training features, evaluates eight heads, refits the best head on
   all labeled passages, and writes `submission.parquet` for the supplied test
   rows. It also times the classifier on 4,000 representative feature rows;
   that timing is not an accuracy test. Extraction progress includes an estimate
   of remaining time after actual batches. Full GPU runtime has not been measured
   here, so do not treat a fixed minute estimate as established.
7. Copy the final **RESULT TO SHARE** block into chat, together with the eight
   scores if possible. Full results are under
   `/kaggle/working/qwen-frozen-v1/summary.json` and `comparison.csv`; per-class
   recall, confusion matrices, length slices and OOF decisions are also saved.
   Those outputs contain competition IDs/labels and should stay private.
8. Stop the interactive session after saving the results you need. GPU quota
   can be consumed while the session is running, including idle time. A fresh
   **Save & Run All** can repeat extraction and use more quota. Same-session
   reruns resume completed batches; fresh sessions may not retain working files.

Review validation first. If the new approach is promising, create a completed
**Save Version → Save & Run All** version, then **Submit to Competition** as
before. It will dynamically process hidden test rows, rather than submit the
three visible rows as fixed predictions. Use submissions to confirm promising
changes, not to choose among many weak local experiments.

## What to send if something fails

Copy the final error and the GPU/package version lines. The notebook deliberately
stops if no GPU is enabled, the wrong weights are attached, data/folds changed,
or extraction produces invalid features. Qwen3 needs Transformers 4.51 or newer.
There are no network downloads or pip installs in this submission; an old Kaggle
runtime would require preparing compatible offline dependencies separately.

## After the frozen result

If frozen features help, try blending with the CPU model using honest grouped
validation, then consider parameter-efficient fine-tuning (such as LoRA).
Fine-tuning updates some transformer weights so that they distinguish authors;
training a large network from scratch is a lower priority with only 2,699 rows.

Compare frozen and fine-tuned versions using the same backbone, prompt folds,
data and token limit. Keep checkpoint selection and early stopping inside each
outer fold's training partition, using inner prompt groups. If the classifier
head changes too, part of the difference may come from that change; a frozen
control with the same head isolates the effect of updating transformer weights.
Submit the promising frozen and fine-tuned models separately only after local
evaluation. Preserve the CPU submission if neither improves on it.

## Sources and local verification

- [Official Qwen Hugging Face model](https://huggingface.co/Qwen/Qwen3-1.7B-Base),
  license **Apache-2.0**, upstream weight revision
  `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.
- Official Kaggle copy `qwen-lm/qwen-3/transformers/1.7b-base/1`, verified via
  public Kaggle model metadata on October 23, 2026. Archive SHA-256:
  `56c57b14bf9b9cc17cdb05c0366ba97fab5e566f71d3bfbf7ddfb256367f225b`.
- Weight SHA-256, matching upstream:
  `6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841`.
  Kaggle's README/tokenizer configuration differ from that upstream revision;
  actual resource hashes are recorded separately. No chat template is used.
- [CPU integration evidence](../reports/qwen-frozen-v1/cpu_integration.json)
  exercises the actual weights, masked pooling, normalized finite features,
  unchanged parameters, completed/partial cache recovery, changed-text isolation
  and corrupt-cache rejection. Synthetic examples are not competition results.
  GPU execution and full transformer grouped CV remain unrun in this cloud machine.

For local reproduction with the optional environment:

```bash
bash scripts/setup_embeddings.sh
/workspace/venvs/ntl-embeddings/bin/python scripts/download_generative.py \
  --out /workspace/models/Qwen3-1.7B-Base-kaggle-v1
/workspace/venvs/ntl-embeddings/bin/python scripts/check_generative_torch.py \
  --model /workspace/models/Qwen3-1.7B-Base-kaggle-v1 \
  --report runs/qwen-smoke-repeat.json
/workspace/venvs/name-that-llm/bin/python scripts/build_generative_notebook.py
/workspace/venvs/name-that-llm/bin/python -m pytest -q
```
