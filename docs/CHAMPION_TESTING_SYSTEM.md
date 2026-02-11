# Champion Testing System (Single Source of Truth)

This document defines the canonical process for training, calibration, backtesting, validation, and champion promotion for:

- **Pregame**
- **Halftime**
- **Q3**

The source-of-truth orchestrator is:

- `src/pipelines/champion_e2e.py`
- Config: `config/champion_testing_v1.json`

---

## 1) Definitive model pool

All states must evaluate the same candidate family unless explicitly disabled with justification:

1. ridge
2. elasticnet
3. random_forest
4. gradient_boosting
5. xgboost
6. lightgbm
7. catboost
8. neural_network
9. quantile_gbm
10. huber
11. extra_trees
12. stacked_ensemble

No champion can be selected if required models are missing from the canonical leaderboard.

---

## 2) Definitive test design

Mandatory checks for each state:

- `fold_strategy = nested_walk_forward`
- `outer_test_size = 200`
- `outer_step_size = 200`
- `inner_folds = 5`
- `min_train_size = 500`
- `seed = 42`

Design principles:

- No random shuffle split for model selection.
- All tuning must occur in the inner temporal loop only.
- Final promotion must use only out-of-sample outer-fold metrics.

---

## 3) Data contracts and checks & balances

For each state, before model execution:

1. Dataset must exist.
2. Required columns must exist.
3. Dataset must have >0 rows.
4. Stage artifacts must exist after each stage.
5. Leaderboard must include `model` and all required models.

If any check fails:

- Run status is failed.
- Champion promotion is blocked.

---

## 4) Canonical pipeline stages

Each state has explicit stages in `config/champion_testing_v1.json`:

- `build` (dataset construction)
- `train` (temporal nested backtest + training)

The system logs every stage command and stdout/stderr into:

- `reports/champion_runs/<RUN_ID>/<state>_<stage>.log`

And writes machine-verifiable reports:

- `reports/champion_runs/<RUN_ID>/run_report.json`
- `reports/champion_runs/<RUN_ID>/champion_candidates.json`

Latest pointers are mirrored to:

- `reports/champion_runs/latest/run_report.json`
- `reports/champion_runs/latest/champion_candidates.json`

---

## 5) Promotion guardrails

Promotion to runtime champions is allowed only when **all checks pass**.

Run:

```bash
python src/pipelines/champion_e2e.py --config config/champion_testing_v1.json --promote
```

When successful, champions are written to:

- `data/processed/champion_models.json`

If a run is invalid, promotion does not occur.

---

## 6) Operational commands

Dry run (CI / command integrity):

```bash
python src/pipelines/champion_e2e.py --dry-run --skip-checks
```

Full run with checks:

```bash
python src/pipelines/champion_e2e.py --config config/champion_testing_v1.json
```

Promote on fully valid run:

```bash
python src/pipelines/champion_e2e.py --config config/champion_testing_v1.json --promote
```

---

## 7) Required leaderboard schema (per state)

At minimum:

- `model`
- `mae_total`
- `mae_margin`
- `rmse_total`
- `ece_win`
- `stability_std_mae_total`

Recommended additions:

- calibration coverage at 50/60/70/80/90/95
- fold-level significance stats
- latency and inference cost

---

## 8) Why this is the single source of truth

This system centralizes:

- model pool definition,
- data checks,
- stage command execution,
- artifact verification,
- leaderboard completeness checks,
- champion promotion gating.

No external ad-hoc report should override `run_report.json` / `champion_candidates.json` from this pipeline.

---

## 9) Execution readiness status (what was missing + now added)

To make this executable end-to-end after commit, the following piping is now in place:

- Fold metrics → leaderboard aggregation script:
  - `src/pipelines/build_champion_leaderboard.py`
- Config-wired leaderboard stage for each state in:
  - `config/champion_testing_v1.json`
- Refresh policy and recurring readiness evaluation:
  - `config/champion_refresh_policy_v1.json`
  - `src/pipelines/champion_refresh_cycle.py`
- Single operational entrypoint that chains refresh-readiness + champion testing:
  - `src/pipelines/run_champion_ops_cycle.py`

This means the platform now has:

1. policy,
2. orchestration,
3. validation gates,
4. refresh/retrain decisioning,
5. promotion wiring.

---

## 10) One-command operations cycle

Dry-run ops cycle:

```bash
python src/pipelines/run_champion_ops_cycle.py --dry-run --skip-checks
```

Full ops cycle (no promotion):

```bash
python src/pipelines/run_champion_ops_cycle.py
```

Full ops cycle with gated promotion:

```bash
python src/pipelines/run_champion_ops_cycle.py --promote
```

---

## 11) Remaining platform prerequisites (outside code)

For production execution, confirm these are configured in your environment:

- Python deps installed (pandas + parquet reader + model libs).
- Scheduler entries (cron/Airflow/GitHub Actions) for:
  - refresh readiness,
  - canonical champion testing,
  - optional promotion job.
- Runtime access to latest datasets and model artifact storage.
- Alerting hook on failed `run_report.json` status.

Code and docs are now in place; these are environment/ops tasks.
