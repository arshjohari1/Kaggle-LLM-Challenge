# Run the prepared CPU solution on Kaggle

The local grouped CV result is 37.56% balanced accuracy; actual leaderboard
performance is unknown. The full model, 4,000-row runtime check and offline
installation have been tested locally. No Kaggle notebook version or submission
has been created automatically.

1. Retain/publish the prepared cloud environment snapshot if you want the local
   data, fitted models, caches and Python environments in future tasks. Git
   contains code and aggregate reports; trained artifacts are intentionally
   excluded.
2. Download `artifacts/cpu-kaggle-bundle-v1.zip` from this workspace. It contains
   `model.joblib`, `training_manifest.json`, `ntl_bundle.json` and `wheels/`.
   Create a **private** Kaggle dataset from its extracted contents, preserving
   that directory structure. No training CSV is required by inference.
3. Create a notebook for the Name That LLM competition and import
   `notebooks/kaggle_inference.ipynb` from this repository. Attach the competition
   input and the private bundle dataset. Attach exactly one bundle manifest.
4. Use CPU and turn Internet off. Run all cells. The notebook verifies every
   attached file checksum and Python minor compatibility, installs only the
   local wheels into an isolated target, then runs inference in a fresh process.
   The tested wheels are Python 3.12/Linux x86-64. If the runtime differs,
   rebuild the wheels using that runtime before proceeding.
5. Check `/kaggle/working/submission.parquet`: in the visible test it should
   contain three rows, the original string IDs and valid model names. The code
   automatically handles the larger hidden test during Kaggle's rerun; it does
   not use a fixed three-row prediction file. Local 4,000-row prediction used
   about 13 seconds and 394 MiB; actual platform timing may differ.
6. When you choose to submit, save/run a notebook version and use Kaggle's
   competition submission flow. Kaggle will rerun it with hidden data offline.
   Keep your local validation reports and record the notebook/version and
   leaderboard result separately. Daily limit is five and final selections two.

## Next research step

The current environment has no GPU. This round compared 19 CPU configurations
and 14 frozen small-encoder configurations using identical prompt groups.
The nested stack, nonlinear kernels and small BGE embeddings did not beat the
selected linear model. More complexity is not currently justified by these
results. The next distinct resource is the user's available **free Kaggle GPU
quota**, for evaluating a larger, suitably licensed generative encoder and then
possibly fine-tuning. No GPU job or paid API request has been launched. Purchases
and billable compute remain unauthorized. Check the account's actual free
accelerator quota before launching that stage.
