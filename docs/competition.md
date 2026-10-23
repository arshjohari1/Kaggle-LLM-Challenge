# Competition requirements and provenance

The user confirmed participation requirements have been checked. No further
eligibility or guardian-consent action is requested by this project.

Public Kaggle access now works. Organizer content was retrieved through the
website's public APIs: `competitions.CompetitionService/GetCompetition` with
competitionName=name-that-llm, and `competitions.PageService/ListPages` with
competitionId 170484. The legacy competition-list endpoint returned 401, but
organizer pages and public notebooks were available without credentials. No
private account data were accessed.

## Confirmed requirements

- [Data](https://www.kaggle.com/competitions/name-that-llm/data): 2,699 training
  passages, 150 prompts, unseen test prompts, about 4,000 hidden passages. Three
  visible test rows are confirmed by the supplied files and Data page; the
  Evaluation page's older single-row sentence is inconsistent.
- [Evaluation](https://www.kaggle.com/competitions/name-that-llm/overview/evaluation):
  balanced accuracy, all 18 exact labels in `src/ntl/data.py`, and differing
  public/private class mixes. IDs and predictions are strings.
- Read `/kaggle/input/competitions/name-that-llm/test.csv`, write
  `/kaggle/working/submission.parquet`. Every ID appears once, with no nulls or
  invalid labels. The sample's large_string fields and ordinary Parquet string
  fields both represent the required logical string values.
- Code Requirements: notebook submission, Internet disabled, CPU/GPU runtime
  at most nine hours; freely/publicly available external data and pretrained
  models permitted.
- Competition-specific rules section 6: external resources must be publicly and
  equally accessible at no cost, or meet the organizer's reasonableness standard.
  This project uses supplied labels, an openly downloadable MIT-licensed
  encoder, and the Apache-2.0 Qwen model described below. Paid generated data
  are not used.
- Rules/metadata: five daily submissions, two final selections, team size five,
  $50,000 total with ten $5,000 awards. No account/team changes were made by the
  assistant; the user subsequently submitted the CPU notebook themselves.

## Timeline discrepancy

The published Timeline page says **January 13, 2027, 23:59 UTC** (3:59 PM PST),
while GetCompetition metadata returns `2027-01-14T23:59:00.817Z`. Entry/team-merger
metadata both say January 6, 2027, 23:59 UTC (3:59 PM PST). Plan for January 13
until the organizer clarifies; do not assume an extension from the backend field.
Sources were checked October 21, 2026 Pacific time.

## Public methods reviewed

- [Organizer starter, v4](https://www.kaggle.com/code/martynaplomecka/starter-notebook-for-name-that-llm-competition):
  word/character TF-IDF, logistic regression, grouped balanced accuracy, input
  path and Parquet output.
- [Dariush Afshar, v2](https://www.kaggle.com/code/dariushafshar/name-that-llm-cv-0-37-tf-idf-svm-baseline):
  short character n-grams, whitespace words and style features. Fetched source
  does not compute its advertised CV result; this project's scores are independent.
- [Mizuno0522, v2](https://www.kaggle.com/code/mizuno0522/model-fingerprints-punctuation-external-data):
  punctuation/boundary-character hypotheses and external-formatting concerns.
  No notebook code or external corpus was copied.
- [OPERA](https://github.com/Ragu-123/namethatllm): README headline and fold
  scores do not directly agree. Its code is not used and its claims are unverified.

Implementation is original. Public notebooks motivate hypotheses; `reports/`
contains this repository's own measurements.

## Encoder and networking

The public copy is
[happenmass/bge-small-en-v1.5/transformers/bge/2](https://www.kaggle.com/models/happenmass/bge-small-en-v1.5/transformers/bge/2).
Kaggle metadata and packaged model card specify MIT. Archive SHA-256:
`952f6b1903d5a00a38fcbe676fb3c6a454c15a3629a92b9ccd3860a9d97cfbb5`.
Individual hashes, metadata and original card are retained in `/workspace/models/`.
The Kaggle artifact is pinned; an upstream Hugging Face commit is not asserted.
Safetensors weights are used rather than pickle weights.

Kaggle and Hugging Face metadata are reachable. Large Hugging Face weights now
redirect to `us.aws.cdn.hf.co`, which the current proxy denies; that optional
domain was added to the saved environment draft without claiming it was applied.
The official Kaggle mirror downloaded successfully over verified TLS, with
archive and upstream-weight checksums verified. Environment publication and
actual Kaggle account quotas are separate from local validation. There are no
paid services or online inference calls in the solution.

## Generative model provenance (October 23, 2026)

Official [Qwen3-1.7B-Base](https://www.kaggle.com/models/qwen-lm/qwen-3/transformers/1.7b-base/1)
is published by qwen-lm under Apache-2.0. Public Kaggle model metadata, official
Hugging Face metadata and the upstream LICENSE were checked. The Kaggle weight
file matches the upstream weight SHA-256. The packaged tokenizer configuration
and README differ from the pinned upstream copy, so individual resource hashes
are recorded rather than claiming an identical full-repository revision.
See [hashes and reproducible instructions](generative.md#sources-and-local-verification).
No third-party notebook code or additional labeled corpus is used. Safetensors
and local-only AutoModel/AutoTokenizer with trust_remote_code=False are required.
