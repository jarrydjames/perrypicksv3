# Robust Tuning Playbook (Vibe Platform)

This guide is the operational runbook for progressing safely from baseline random search to stronger Optuna-based tuning for XGBoost/CatBoost in nested walk-forward training.

It is designed for non-interactive vibe coding platforms where runs should fail fast on broken artifacts/logs and only promote champions after explicit gates pass.

---

## 1) Prerequisites

Minimum runtime for champion and tuning jobs:

- Python 3.10+
- `numpy`, `pandas`, `pyarrow`, `scikit-learn`
- Optional (recommended for robust tuning): `optuna`
- Optional candidates used by tuning runs: `xgboost`, `catboost`

Install example:

```bash
pip install -r requirements-dev.txt
pip install optuna
```

---

## 2) Guardrails now enforced in pipeline

`src/pipelines/champion_e2e.py` now enforces these checks:

1. **Artifact integrity**: required artifacts must exist, be non-empty, and be fresh for that stage.
2. **Resolved artifact paths**: `reports/champion_runs/latest/...` aliases are resolved before validation.
3. **Log scanning**: stage logs are scanned for common hard-failure signatures (`Traceback`, import/runtime errors, `FAILED`, etc.).
4. **Exit behavior**: the pipeline exits non-zero if global `ok` is false.

Operational implication: do **not** treat the run as successful from file presence alone. Always gate on `run_report.json` + process exit code.

---

## 3) Tuning strategy ladder (recommended progression)

Use this exact progression to reduce risk and isolate issues quickly.

### Stage A — Wiring smoke (no training)

Purpose: verify orchestration wiring and output locations.

```bash
scripts/vibe_run_champion_pipeline.sh --dry-run --skip-checks
```

Pass criteria:
- command exits `0`
- `reports/champion_runs/latest/run_report.json` exists
- no path/command-template issues

### Stage B — Baseline random search

Purpose: produce a reliable baseline before adding optimizer complexity.

```bash
python src/modeling/nested_walkforward_backtest.py \
  --data data/processed/halftime_with_temporal_features_total.parquet \
  --out reports/champion_runs/latest/halftime_fold_metrics.csv \
  --include-xgb --include-cat \
  --inner-folds 3 --trials 15 --seed 42 --tuner random
```

Pass criteria:
- fold metrics CSV created and non-empty
- no traceback in logs
- repeatability acceptable across two runs with same seed

### Stage C — Optuna canary (limited budget)

Purpose: validate optimizer behavior with bounded compute.

```bash
python src/modeling/nested_walkforward_backtest.py \
  --data data/processed/halftime_with_temporal_features_total.parquet \
  --out reports/champion_runs/latest/halftime_fold_metrics_optuna_canary.csv \
  --include-xgb --include-cat \
  --inner-folds 3 --trials 20 --seed 42 \
  --tuner optuna --optuna-timeout-s 600
```

Pass criteria:
- run completes without fallback/import errors
- model params in output differ from baseline random draws
- objective metrics are stable/improved vs Stage B median

### Stage D — Full Optuna tune

Purpose: run production-grade Bayesian tuning.

```bash
python src/modeling/nested_walkforward_backtest.py \
  --data data/processed/halftime_with_temporal_features_total.parquet \
  --out reports/champion_runs/latest/halftime_fold_metrics_optuna_full.csv \
  --include-xgb --include-cat \
  --inner-folds 5 --trials 50 --seed 42 \
  --tuner optuna --optuna-timeout-s 1800
```

Pass criteria:
- no stage-level guardrail failures
- improved or equivalent error/calibration with lower variance
- no evidence of unstable hyperparameter swings across folds

### Stage E — Canonical champion cycle

Purpose: roll tuned artifacts through full readiness + champion gating.

```bash
scripts/vibe_run_champion_pipeline.sh
```

Promotion (only after manual review of latest report):

```bash
scripts/vibe_run_champion_pipeline.sh --promote
```

---

## 4) What to inspect after each stage

Primary artifacts:

- `reports/champion_runs/latest/run_report.json`
- `reports/champion_runs/latest/champion_candidates.json`
- state fold metrics CSVs under `reports/champion_runs/latest/`
- per-stage logs under timestamped run directories in `reports/champion_runs/<RUN_ID>/`

Key checks:

1. `run_report.json` top-level `ok == true`
2. every stage has `return_code == 0`
3. every stage has `artifact_check.ok == true`
4. every stage has `log_check.ok == true`
5. leaderboard checks contain required columns/models

---

## 5) Recommended tuning defaults by environment

### CI / shared runner

- `--tuner random`
- `--inner-folds 3`
- `--trials 10-20`

### Nightly training

- `--tuner optuna`
- `--inner-folds 3-5`
- `--trials 30-60`
- `--optuna-timeout-s 1200-2400`

### Release-candidate refresh

- `--tuner optuna`
- fixed seed and data snapshot
- run twice to confirm stability
- promote only when both runs pass and metrics are consistent

---

## 6) Failure handling / rollback

If any stage fails:

1. Stop; do not promote.
2. Inspect failing state log in `reports/champion_runs/<RUN_ID>/`.
3. Re-run with reduced scope:
   - one state only (manual command)
   - lower folds/trials
   - switch to `--tuner random` for diagnosis
4. Restore last known-good champion map if needed from source control/artifact store.

---

## 7) Next-step hardening (recommended backlog)

To make tuning even more robust in future iterations:

- add Optuna pruners (`MedianPruner` / `SuccessiveHalvingPruner`)
- persist Optuna studies to SQLite for resumable tuning
- add per-fold early-stop/report JSON for long runs
- add explicit drift gate between current champion and challenger distributions
- add CI contract test: stale artifacts must force non-zero exit
