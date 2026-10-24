# Project instructions

- The user authorized project creation, development, experiments, and pushing to
  `main`. Do not submit to Kaggle, accept rules on their behalf, purchase services,
  generate paid API data, or use billable compute without explicit authorization.
  The budget is $0. An optional reserve is not spending permission.
- Explain major modeling choices plainly. The user has introductory Python/ML
  experience. Do implementation and debugging independently when possible.
- Each cloud task is already isolated. Use the existing checkout; do not create
  Git worktrees unless explicitly requested.
- Use `/workspace/venvs/name-that-llm` in this cloud environment. Run
  `python -m pytest -q` after implementation changes. Keep data and outputs out of
  Git and preserve existing runs rather than overwriting them.
- Group validation by prompt_id, connect duplicate passages across prompt groups,
  and fit all learned preprocessing inside training folds. Use the same fixed
  folds and balanced accuracy for comparisons. Never call synthetic results
  competition performance. Do not report unexecuted or public claims as measured.
- The user has checked participation requirements; do not ask again. Rules for
  external data, runtime, and free Kaggle quotas still need confirmation before
  using external resources or scheduling remote training. See `docs/competition.md`.
- Do not request secrets in chat, print environment values, or commit credentials.
  Do not commit competition data or model artifacts.
- Keep seeds, data/fold hashes, versions, scores, class recall, timing, and
  provenance for real experiments. Audit licenses before using external resources.
- Prefer small reproducible CPU baselines; only expand to embeddings, fine-tuning,
  or ensembles when access, compute, rules, and measured results support them.
- Preserve the measured complementarity: frozen Qwen correctly identified 417
  passages missed by the CPU model in the original OOF reports. The fixed equal
  rank blend scored 0.4093263733532189 on those reused folds. These historical
  facts motivate the blend; they are not test-time routing rules or a promised
  leaderboard score. See reports/qwen-frozen-v1/analysis.md and docs/combined.md.
