# Run the prepared CPU solution on Kaggle

The local grouped CV result is 37.56% balanced accuracy; actual leaderboard
performance is unknown. The full model, 4,000-row runtime check and offline
installation have been tested locally. No Kaggle notebook version or submission
has been created automatically.

## Simple route: no model downloads or custom dataset

The standalone version trains the same selected SVM directly on the competition
training CSV, then predicts the test CSV. This uses Kaggle's preinstalled
scientific Python packages, so no model ZIP, custom dataset, pip installation,
GPU or Internet access is required during execution. Package versions are
recorded in `training_info.json`; retraining avoids serialized-model version
compatibility requirements.

1. Open the [pasteable Python file](https://github.com/arshjohari1/Kaggle-LLM-Challenge/blob/main/notebooks/kaggle_train_and_submit.py).
   In GitHub's file toolbar, click **Copy raw file** (the overlapping-squares
   icon). Copy the full file, without line numbers.
2. Open the competition's [Code tab](https://www.kaggle.com/competitions/name-that-llm/code)
   and click **New Notebook**. Paste the copied Python into a code cell. Remove
   any unused starter code cells so only the submission code runs.
3. Confirm Name That LLM competition data is attached under Inputs. Choose
   **Accelerator: None** and **Internet: Off**.
4. Run All. It trains on all 2,699 labelled passages and creates
   `/kaggle/working/submission.parquet`. Look for **SUCCESS** and **Submission
   rows: 3** with the current visible data. CPU training takes tens of seconds
   locally; actual Kaggle time depends on the machine.
5. Click **Save Version > Save & Run All**, wait for success, then choose
   **Submit to Competition** for that saved version. Check **My Submissions**
   for the public leaderboard score.

Alternatively, open the [standalone notebook](https://github.com/arshjohari1/Kaggle-LLM-Challenge/blob/main/notebooks/kaggle_train_and_submit.ipynb),
use GitHub's **Download raw file** button, then **File > Import Notebook** in
Kaggle. It contains the same code split into cells. No extra model assets are
needed. Training the final model on all training passages is intentional; the
37.56% validation score was measured separately using held-out prompt groups.
This notebook does not calculate or claim an in-sample accuracy score.

## Alternative: use the pre-trained model bundle

1. Retain/publish the prepared cloud environment snapshot if you want the local
   data, fitted models, caches and Python environments in future tasks. Git
   contains code and aggregate reports; trained artifacts are intentionally
   excluded.
2. Download `artifacts/cpu-kaggle-bundle-v1.zip` (about 151 MiB) and
   `notebooks/kaggle_inference.ipynb` from this workspace. The ZIP contains
   `model.joblib`, `training_manifest.json`, `ntl_bundle.json` and `wheels/`.
   Keep its name unchanged. No training CSV is required by inference.
3. Sign in to Kaggle and open https://www.kaggle.com/datasets/new. Upload the
   ZIP, name the dataset **Name That LLM CPU Model v1**, select **Private**, and
   create it. If Kaggle extracts the archive, keep the `wheels/` folder and model
   files together. The notebook handles either extracted files or the intact ZIP.
4. Open https://www.kaggle.com/competitions/name-that-llm/code and choose
   **New Notebook**. In the editor choose **File > Import Notebook**, upload the
   supplied `.ipynb`, and name it **Name That LLM CPU Baseline v1**.
   Use the input sidebar's **Add Input** (sometimes called **Add Data**) to attach
   your private model dataset. Confirm the competition input is also attached;
   if absent, find Name That LLM under the competition inputs and add it.
   Attach exactly one copy of the model bundle.
5. Under notebook/session settings choose **Accelerator: None** and **Internet:
   Off**. Run all cells. The notebook verifies every attached file checksum and
   Python minor compatibility, installs only local wheels into a temporary
   target outside the working outputs, then runs inference in a fresh process.
   The tested wheels are Python 3.12/Linux x86-64. If the runtime differs,
   rebuild the wheels using that runtime before proceeding.
6. Check `/kaggle/working/submission.parquet`: in the visible test it should
   contain three rows, the original string IDs and valid model names. The code
   automatically handles the larger hidden test during Kaggle's rerun; it does
   not use a fixed three-row prediction file. Local 4,000-row prediction used
   about 13 seconds and 394 MiB; actual platform timing may differ.
   The final cells should print **SUCCESS: validated submission saved to
   /kaggle/working/submission.parquet**, then **Submission rows: 3** and the
   predictions. Re-running all cells is supported; a previous submission file
   is replaced only after the new one validates.
7. Click **Save Version**, select **Save & Run All** rather than a quick save,
   and save the version. Wait for it to finish successfully. Open the completed
   version's viewer/output and check that `submission.parquet` is present.
8. Choose **Submit to Competition** for that completed notebook version and
   confirm Name That LLM. If starting from the competition's **Submit Predictions**
   screen, choose your saved notebook/version there. Submit the notebook version
   so Kaggle reruns the code on hidden data; the visible three-row file is only
   a format check. Wait for the submission status to complete, then check the
   public score under the competition's **My Submissions** tab.

Kaggle's interface labels can vary slightly. Its public score need not equal the
37.56% local CV score. The private leaderboard score is determined separately.
Record notebook/version and score together. Daily limit is five submissions and
final selections two. This process needs no GPU and does not train the model.

## Common first-run messages

- **Add your private model-bundle dataset:** attach the dataset using Add Input;
  upload the provided model ZIP, not just the `.ipynb`.
- **Found more than one manifest:** remove duplicate model-bundle inputs.
- **Competition test.csv missing:** attach the competition input as well as the
  private model dataset.
- **Bundle Python 3.12 != runtime ...:** report the full message; wheels need to
  be rebuilt for that runtime. Do not turn Internet on to work around it.
- A red cell or failed saved version: inspect its final error message before
  submitting. A successful ordinary run is followed by Save & Run All to create
  the version Kaggle scores.

Official reference: https://www.kaggle.com/docs/competitions#notebooks-only-FAQ

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
