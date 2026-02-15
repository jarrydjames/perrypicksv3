# Halftime Backtest Alignment Guide

This document describes the implementation changes that align `scripts/halftime_backtest_espn.py` with the champion tuning pipeline behavior.

## What was added

1. **Fail-fast feature gate before training**
   - The script now builds inference rows and runs feature-health validation *before* model fitting.
   - If critical feature integrity checks fail, the run exits early unless `--allow-feature-issues` is passed.

2. **Robust top-k parameter selection**
   - `--param-selection topk` no longer blindly takes a single fold row.
   - It now builds a robust parameter set from top-k folds:
     - numeric hyperparameters: median
     - categorical/discrete hyperparameters: mode
   - Metrics now record `selected_param_folds` for traceability.

3. **Calibration parity with nested walk-forward**
   - Tail-split sigma scaling (`sigma_k_total`, `sigma_k_margin`) is applied during one-day backtest.
   - Margin sigma calibration is used for `pred_win_prob` and Brier calculations.

4. **Environment check utility**
   - Added `scripts/check_halftime_backtest_env.py` to validate required runtime dependencies.

## Commands

```bash
# Verify environment dependencies first
python scripts/check_halftime_backtest_env.py

# Run with robust top-k params and strict feature gates
python scripts/halftime_backtest_espn.py --date 2026-02-11 --param-selection topk --param-topk 5

# Only for diagnostics, bypass feature gate failure
python scripts/halftime_backtest_espn.py --date 2026-02-11 --allow-feature-issues
```

## Output artifacts

- `reports/backtest/feature_health_<date>.json`
- `reports/backtest/halftime_backtest_<date>.csv`
- `reports/backtest/halftime_backtest_<date>_detailed.csv`
- `reports/backtest/metrics_<date>.json` (includes calibration and parameter-selection metadata)
