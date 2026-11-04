# AGENTS.md

Guide for coding agents working in this repo (team G5, IIC3633 Sistemas Recomendadores,
PUC Chile, 2026-2). Humans: setup and reproduction are in [`README.md`](README.md).

## The project

We recommend jobs to candidates (**c2j**) on the JTH dataset (Romero, Millet & Behar,
RecSys '26). The goal is to beat the paper's best c2j model, logistic regression, under the
paper's own protocol, and to measure what each of our components adds:

1. **Contextual filter on whether a job is still open.** v1 is a hard cutoff by recent
   activity. v2 is a soft weight by an estimated probability that the job is still open.
2. **Ensemble** of logistic regression and CF + recency. They are complementary: on warm
   candidates CF + recency is slightly better (MRR 0.244 vs 0.237), on new candidates it
   collapses (0.033 vs 0.317).
3. **Stage 2, a job-centric temporal model.** Start with a dynamic job profile (the average
   attributes of the job's recent candidates), then a sequential version. The best one joins
   the ensemble.

Deliverables (Canvas, 20:00, −0.5 per late hour):
- H2 Midterm: 2026-11-06, 3–4 pages.
- H4 Poster: 2026-12-03, in person, attendance mandatory.
- H3 Paper: 2026-12-08, ≤ 8 pages including references, ACM/ICML/NeurIPS format, with code
  and README. Every result and figure must be backed by code here.

Teaching-team feedback on the proposal is in `docs/feedback_H1_propuesta.txt`. Read it
before planning work. Every deliverable must include the link to this repo.

## Commands

```bash
uv sync                                    # env (Python 3.12, pinned)
uv run python scripts/download_data.py     # data -> data/raw/
bash scripts/run_upstream.sh               # official split + baselines (~50 min)
WITH_SLOW=1 bash scripts/run_upstream.sh   # + logistic regression c2j (~17 min)
uv run python scripts/make_table.py        # results/tabla_baselines.tex
uv run python scripts/estadisticas.py      # results/tabla_datos.tex + Sec. 2 numbers
uv run python scripts/figuras.py           # figures/fig_*.png
uv run pytest                              # must stay green
```

Always run Python through `uv run`. Add dependencies with `uv add` and keep them few, well
known and actively maintained.

## Protocol rules

These keep our numbers comparable with the paper. Do not break them without the team agreeing.

- **Reported numbers use the official split and the official evaluator.** The split is in
  `data/upstream_split/`, the evaluator in `external/JTH/code/utils/evaluator.py`. Our own
  evaluator (`src/jth/evaluation.py`) is a cross-check only.
- **The task.** A query is a (candidate, day t) pair with a process in test. There are
  8,341 queries. The relevant item is that process's job, whatever stage it reached. The
  model ranks its top 1000 jobs, and jobs created after t are dropped before scoring.
  - Main metric: MRR.
  - Secondary metric: NDCG@10.
  - Always split warm (the candidate was seen before) vs cold.
- **No future information.** At day t a model may use only interactions before t and jobs
  created by t.
- **Never tune on test.** This covers thresholds, ensemble weights and hyperparameters. How
  the validation slice is built is still undecided, so ask.
- **Say whose number it is.** Paper-reported numbers are in `results/paper_reported.csv`.
  Our re-run of the official code is in `results/upstream/results.csv`. Example: logistic
  regression is 0.283 MRR in the paper and 0.286 in our re-run.
- **No sensitive attributes as features:** `llm_sex`, `llm_nationality`, `llm_age_bucket`.
  The data is CC BY-NC 4.0, and re-identification is forbidden.
- **Never commit** data, PDFs, prediction files, figures or generated `.tex` tables (see
  `.gitignore`).
- **Use synthetic data** in tests and examples, never real rows.

## Prediction files and the official evaluator

- **Format.** A prediction file is a gzip pickle of a dict. Each key is the JSON string
  `["<query_id>", "YYYY-MM-DD"]` and each value is a ranked list of up to 1000 target ids.
  Ids may be str or int; the evaluator normalises them.
- **Naming and scoring.** Name files `<model>_<c2j|j2c>.pkl.gz` and put them in
  `results/upstream/`. The loop at the end of `run_upstream.sh` scores every file there into
  `results/upstream/results.csv`.
- **Re-scoring.** The evaluator keys rows by the prediction path and skips paths it already
  has. Pass paths relative to the repo root. To re-score a file, delete its row first.
- **Skipped runs.** `run_upstream.sh` skips prediction files that already exist. Delete a
  file to regenerate it.

## Known quirks of the official code

- **Logistic regression** (`feature_classifier_baseline.py`):
  - It is trained once on train and never refreshed during the test stream.
  - Its negatives are sampled from all jobs, not only open ones.
  - Some numeric LLM fields (years of experience, estimated salary) are parsed as booleans.
  - It writes `features.csv` to the working directory.
- **The splitter** (`binary_splitter.py`) cuts at a row index, so one day can fall on both
  sides of the split.
- **One job per query.** When a candidate starts two processes on the same day, the
  evaluator keeps only one as the relevant job.
- **Our replication** matches the paper within ±0.005 MRR, except CF + recency (0.114 vs
  0.119).

## Running long jobs

- Use `python -u` (or `PYTHONUNBUFFERED=1`) for background runs. Otherwise the log stays
  empty until the process exits.
- Never wait on `pgrep -f <pattern>` when the pattern is also in your own command line. It
  matches itself and loops forever, which once cost us a night. Wait on the output file or
  the PID instead.

## Report outputs

- **Generated, never committed, never hand-edited.** Figures and LaTeX tables are
  git-ignored. Regenerate them after a fresh clone, or whenever their inputs change:

  | Output | Command | Needs |
  |---|---|---|
  | `results/tabla_baselines.tex` | `uv run python scripts/make_table.py` | `results/upstream/results.csv` (tracked) |
  | `results/tabla_datos.tex` | `uv run python scripts/estadisticas.py` | `data/raw/` + `data/upstream_split/` |
  | `figures/fig_*.png`, `results/tabla_cobertura.tex` | `uv run python scripts/figuras.py` | `data/raw/` + `data/upstream_split/` |

  `data/upstream_split/` is written by the first step of `run_upstream.sh`. To change an
  output, edit its script and rerun it.
- **Figures** are simple matplotlib with Spanish titles and axis labels. Thousands use a dot
  and decimals use a comma.
- **Tables** use booktabs and float as `[htbp]`.
- **References.** `references/recsys.bib` is what the report cites (unsrt, order of
  appearance). `references/candidatas.bib` holds verified references not cited yet. Check
  any new reference against Crossref or the publisher, and never invent bibliographic
  details.
- **Report text.** If asked to draft or edit it, follow `docs/report_style.md`.

## How to work with this team

- **You are a copilot.** Build code, run experiments, produce figures and tables. Don't
  write report text unless asked.
- **Check in before big builds or long runs.** Give a short status and plan, then wait for
  a go.
- **Reuse or wrap the official code** before reimplementing anything.
- **Don't add what nobody agreed on.** That includes extra statistical tests, baselines,
  splits or analyses. Propose them instead.
- **Ask for missing inputs** (a paper, a decision) instead of guessing.
- **Keep every number traceable** from the report to a script and an output file.
- **Don't commit or push unless asked.** No secrets in code. Before the final submission,
  check dependencies with Dependabot on the GitHub repo or `pip-audit`.
