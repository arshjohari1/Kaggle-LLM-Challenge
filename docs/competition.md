# Official verification and access checklist

Status as of October 21, 2026 (America/Los_Angeles): **unverified**. Requests to
Kaggle overview, rules, starter notebook, and the data-list API were denied by the
cloud egress proxy with `403 Forbidden` before reaching Kaggle. This is a network
policy blocker, not evidence of an invalid competition or missing account access.

Network additions were saved to the cloud environment configuration draft for
`www.kaggle.com`, `kaggle.com`, and `storage.googleapis.com`. Draft saving does
not apply runtime networking. Review and save in environment settings, then
publish the environment as directed by the platform. Additional download hosts
should be added only if an actual authorized request identifies them.

There are no configured Kaggle bindings or credentials in the current machine.
Public-page verification needs network access first. Downloading competition
data may additionally require signing in and accepting rules in the user's
account. Uploading the original CSV files here is sufficient for local training;
no Kaggle credentials are required for this route. Do not put secrets in chat.

## Sources to check

- [Overview](https://www.kaggle.com/competitions/name-that-llm/overview)
- [Data](https://www.kaggle.com/competitions/name-that-llm/data)
- [Rules](https://www.kaggle.com/competitions/name-that-llm/rules)
- [Discussions and current organizer replies](https://www.kaggle.com/competitions/name-that-llm/discussion)
- [Organizer starter](https://www.kaggle.com/code/martynaplomecka/starter-notebook-for-name-that-llm-competition)
- [Kaggle terms](https://www.kaggle.com/terms)

Before any real submission, record the official source, retrieval date, and exact
requirement for each item below. The user's earlier research is a starting point,
not official verification by this project.

| Item | Supplied description, pending confirmation |
| --- | --- |
| Task/metric | 18-class authorship; balanced accuracy |
| Training | 2,699 rows, 150 prompt groups; id/prompt_id/text/model |
| Test | Small visible sample; about 4,000 hidden passages, unseen prompts |
| Labels | Provisional list in `src/ntl/data.py`; verify spelling and all versions |
| Test location | `/kaggle/input/competitions/name-that-llm/test.csv` |
| Output | `/kaggle/working/submission.parquet`, id and model; verify ID dtype |
| Inference | Offline code competition; reported 9-hour CPU/GPU limit |
| Deadlines | Entry/team January 6, 2027; final January 13, 2027, each 23:59 UTC (3:59 PM PST) |
| Team/submissions | Reported team size 5, five daily submissions, two final selections |
| Prizes | Reported $50,000 total, $5,000 for each of places 1–10 |
| External resources | Public data/models reportedly allowed; verify licensing, access/sharing deadlines and paid generation policy |

## Age, guardian consent, and prizes

The user is 16. **Eligibility, consent, and prize eligibility are not confirmed.**
The user and a parent/guardian should review the current Kaggle terms and this
competition's rules together, including age/residence conditions, who accepts the
rules, guardian authorization, identity verification, and any separate prize or
tax paperwork. If the rules do not answer these points, request a written answer
from Kaggle or the organizer before entering or relying on prize eligibility.
Do not infer that being able to make an account or upload predictions establishes
eligibility. Do not provide birth dates, IDs, or private legal documents here.

## Free compute

This cloud machine exposes approximately 32 GiB RAM, five CPU cores, 30 GiB free
disk at initial inspection, and no NVIDIA device or `nvidia-smi`. No Kaggle account
is connected, so GPU availability and remaining free quotas are unknown. Check
the notebook accelerator quota in your Kaggle account before scheduling neural
experiments. No paid compute or paid data generation is authorized.

## Public resources

User-supplied resources, not yet reproduced:

- [TF-IDF/SVM](https://www.kaggle.com/code/dariushafshar/name-that-llm-cv-0-37-tf-idf-svm-baseline/comments)
- [Frozen embeddings](https://www.kaggle.com/code/qasminumber/name-that-llm-v5)
- [CPU/prompt effects](https://www.kaggle.com/code/patimakornsrinamen/name-that-llm-the-prompt-hides-the-author-cpu?scriptVersionId=360010467)
- [Punctuation/external data](https://www.kaggle.com/code/mizuno0522/model-fingerprints-punctuation-external-data)
- [Data overview](https://www.kaggle.com/code/olegzholobov/name-that-llm-wide-data-overview)
- [API passages](https://www.kaggle.com/datasets/seyitkaangunes/ntl-extra-api-passages)
- [OPERA repository](https://github.com/Ragu-123/namethatllm)

The OPERA README was reachable through GitHub. Its headline score and listed fold
scores do not directly agree; no claim has been accepted as a reproduced result.
No external code, model, or corpus has been incorporated. Check licenses and
attribution requirements before reuse.
