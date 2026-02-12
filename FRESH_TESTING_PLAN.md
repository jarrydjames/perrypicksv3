# 🔄 FRESH CHAMPION TESTING PLAN
**Date:** February 12, 2026
**Objective:** Execute fresh, robust champion testing using latest methodology

## 🚨 ACKNOWLEDGMENT

**All previous testing was flawed and produced unreliable results.**

### Previous Issues:
1. ❌ Q3 used wrong targets (q3_total/q3_margin instead of remaining_total/remaining_margin)
2. ❌ LightGBM wasn't properly tuned (only evaluated, not optimized)
3. ❌ No robust calibration methodology applied
4. ❌ Inconsistent model evaluation across game states
5. ❌ No proper artifact validation or log scanning

### What's Fixed:
1. ✅ Correct targets for all game states
2. ✅ Full tuning support for XGBoost, CatBoost, LightGBM
3. ✅ Robust pipeline with artifact checks and log scanning
4. ✅ Optuna-based Bayesian optimization
5. ✅ Proper nested walk-forward validation
6. ✅ Comprehensive calibration methodology

---

## 📊 TESTING METHODOLOGY (From ROBUST_TUNING_PLAYBOOK.md)

### Stage Progression:

**Stage A** - Wiring smoke (dry-run validation)
**Stage B** - Baseline random search
**Stage C** - Optuna canary (limited budget)
**Stage D** - Full Optuna tune (production-grade)
**Stage E** - Canonical champion cycle

**For this fresh run, we'll jump to Stage D (Full Optuna tune) with production-grade settings.**

---

## 🎯 TESTING PLAN

### Game State Order:
1. **HALFTIME** (first)
2. **PREGAME** (second)
3. **Q3** (third)

### Per-State Testing Protocol:

#### For each game state:
1. **Build dataset** (if needed)
2. **Train models** with nested walk-forward validation
3. **Tune hyperparameters** with Optuna
4. **Generate fold metrics** CSV
5. **Build champion leaderboard**
6. **Validate artifacts** and logs

---

## 📋 DETAILED EXECUTION PLAN

### PHASE 1: HALFTIME TESTING

**Dataset:** `data/processed/halftime_with_temporal_features_total.parquet`
**Targets:** `h2_total`, `h2_margin`
**Models:** 8 total
- 5 baseline: Ridge, RandomForest, GBT, ElasticNet, MLP
- 3 tuned: XGBoost, CatBoost, LightGBM (with Optuna)

**Configuration:**
```bash
--data data/processed/halftime_with_temporal_features_total.parquet
--out reports/champion_runs/latest/halftime_fold_metrics.csv
--include-xgb --include-cat --include-lgbm
--target-total h2_total --target-margin h2_margin
--tuner optuna
--optuna-timeout-s 1800  # 30 minutes per model
--inner-folds 5
--trials 50
--seed 42
--train-min 500
--test-size 200
--step-size 200
```

**Expected Output:**
- `reports/champion_runs/latest/halftime_fold_metrics.csv`
- `reports/champion_runs/latest/halftime_leaderboard.csv`
- Calibration plots and metrics

**Estimated Duration:** ~6-8 hours (depending on dataset size)

---

### PHASE 2: PREGAME TESTING

**Dataset:** `data/processed/pregame_team_v2.parquet`
**Targets:** `total`, `margin`
**Models:** Same 8 models

**Configuration:**
```bash
--data data/processed/pregame_team_v2.parquet
--out reports/champion_runs/latest/pregame_fold_metrics.csv
--include-xgb --include-cat --include-lgbm
--target-total total --target-margin margin
--tuner optuna
--optuna-timeout-s 1800
--inner-folds 5
--trials 50
--seed 42
--train-min 500
--test-size 200
--step-size 200
```

**Expected Output:**
- `reports/champion_runs/latest/pregame_fold_metrics.csv`
- `reports/champion_runs/latest/pregame_leaderboard.csv`

**Estimated Duration:** ~5-7 hours

---

### PHASE 3: Q3 TESTING

**Dataset:** `data/processed/q3_team_v2.parquet`
**Targets:** `remaining_total`, `remaining_margin` ← **CORRECTED!**
**Models:** Same 8 models

**Configuration:**
```bash
--data data/processed/q3_team_v2.parquet
--out reports/champion_runs/latest/q3_fold_metrics.csv
--include-xgb --include-cat --include-lgbm
--target-total remaining_total --target-margin remaining_margin
--tuner optuna
--optuna-timeout-s 1800
--inner-folds 5
--trials 50
--seed 42
--train-min 500
--test-size 200
--step-size 200
```

**Expected Output:**
- `reports/champion_runs/latest/q3_fold_metrics.csv`
- `reports/champion_runs/latest/q3_leaderboard.csv`

**Estimated Duration:** ~5-7 hours

---

## 🤖 MODELS TO TEST (8 TOTAL)

### Baseline Models (No Tuning):
1. **Ridge** - L2 regularized linear regression
2. **RandomForest** - Ensemble of decision trees
3. **GBT** - Gradient Boosting Trees (sklearn)
4. **ElasticNet** - L1+L2 regularized linear regression
5. **MLP** - Neural network (2 hidden layers: 64, 32)

### Tunable Models (Optuna Optimization):
6. **XGBoost** - Extreme Gradient Boosting
   - Hyperparameter space: n_estimators (300-1800), learning_rate (0.015-0.12), max_depth (3-8), etc.
   - Tuning: 50 trials, 30-minute timeout

7. **CatBoost** - Category Boosting (Yandex)
   - Hyperparameter space: iterations (600-3500), learning_rate (0.015-0.12), depth (4-10), etc.
   - Tuning: 50 trials, 30-minute timeout

8. **LightGBM** - Light Gradient Boosting (Microsoft)
   - Hyperparameter space: n_estimators (300-1800), learning_rate (0.015-0.12), max_depth (3-8), etc.
   - Tuning: 50 trials, 30-minute timeout

---

## 📊 EVALUATION METRICS

### Primary Metrics:
- **MAE_total** - Mean Absolute Error for total points
- **MAE_margin** - Mean Absolute Error for margin
- **RMSE_total** - Root Mean Squared Error for total points
- **RMSE_margin** - Root Mean Squared Error for margin

### Calibration Metrics:
- **PI80_coverage** - 80% Prediction Interval coverage (target: ~0.80)
- **Brier_win** - Brier score for win probability (lower is better)
- **ECE_win** - Expected Calibration Error for win probability

### Stability Metrics:
- **Stability** - Standard deviation of MAE across folds (lower is better)

---

## 🔒 VALIDATION GATES

### Per-Stage Checks:
1. ✅ **Return code = 0** (no errors)
2. ✅ **Artifact exists** (required CSV files present)
3. ✅ **Artifact non-empty** (size > 0 bytes)
4. ✅ **Artifact fresh** (created during this run)
5. ✅ **Log clean** (no Traceback, ImportError, RuntimeError, etc.)
6. ✅ **Leaderboard valid** (contains all required columns and models)

### Run-Level Checks:
1. ✅ **run_report.json exists and ok=true**
2. ✅ **All stages passed** (no failures)
3. ✅ **Champion candidates identified**
4. ✅ **Metrics within expected ranges**

---

## 📁 OUTPUT FILES

### Per-State Outputs:
```
reports/champion_runs/latest/
├── {state}_fold_metrics.csv         # All model metrics across folds
├── {state}_leaderboard.csv          # Champion leaderboard
└── {state}_champion_candidates.json # Top model candidates
```

### Run-Level Outputs:
```
reports/champion_runs/latest/
├── run_report.json                  # Overall run status
├── champion_candidates.json         # All champion candidates
└── preflight.json                   # Preflight check results
```

### Timestamped Logs:
```
reports/champion_runs/<TIMESTAMP>/
├── {state}_build.log
├── {state}_train.log
├── {state}_leaderboard.log
└── run_report.json
```

---

## ⏱️ TIMELINE ESTIMATE

| Phase | State | Duration | Cumulative |
|-------|-------|----------|------------|
| 1 | Halftime | 6-8 hours | 6-8 hours |
| 2 | Pregame | 5-7 hours | 11-15 hours |
| 3 | Q3 | 5-7 hours | 16-22 hours |

**Total Estimated Duration:** 16-22 hours (~1 day)

---

## 🚀 EXECUTION STEPS

### Step 1: Clean Up Old Results
```bash
rm -rf reports/champion_runs/latest/*
```

### Step 2: Run Halftime Testing
```bash
python src/modeling/nested_walkforward_backtest.py \
  --data data/processed/halftime_with_temporal_features_total.parquet \
  --out reports/champion_runs/latest/halftime_fold_metrics.csv \
  --include-xgb --include-cat --include-lgbm \
  --target-total h2_total --target-margin h2_margin \
  --tuner optuna --optuna-timeout-s 1800 \
  --inner-folds 5 --trials 50 --seed 42
```

### Step 3: Generate Halftime Leaderboard
```bash
python src/pipelines/build_champion_leaderboard.py \
  --input reports/champion_runs/latest/halftime_fold_metrics.csv \
  --output reports/champion_runs/latest/halftime_leaderboard.csv \
  --state halftime
```

### Step 4: Run Pregame Testing
```bash
python src/modeling/nested_walkforward_backtest.py \
  --data data/processed/pregame_team_v2.parquet \
  --out reports/champion_runs/latest/pregame_fold_metrics.csv \
  --include-xgb --include-cat --include-lgbm \
  --target-total total --target-margin margin \
  --tuner optuna --optuna-timeout-s 1800 \
  --inner-folds 5 --trials 50 --seed 42
```

### Step 5: Generate Pregame Leaderboard
```bash
python src/pipelines/build_champion_leaderboard.py \
  --input reports/champion_runs/latest/pregame_fold_metrics.csv \
  --output reports/champion_runs/latest/pregame_leaderboard.csv \
  --state pregame
```

### Step 6: Run Q3 Testing
```bash
python src/modeling/nested_walkforward_backtest.py \
  --data data/processed/q3_team_v2.parquet \
  --out reports/champion_runs/latest/q3_fold_metrics.csv \
  --include-xgb --include-cat --include-lgbm \
  --target-total remaining_total --target-margin remaining_margin \
  --tuner optuna --optuna-timeout-s 1800 \
  --inner-folds 5 --trials 50 --seed 42
```

### Step 7: Generate Q3 Leaderboard
```bash
python src/pipelines/build_champion_leaderboard.py \
  --input reports/champion_runs/latest/q3_fold_metrics.csv \
  --output reports/champion_runs/latest/q3_leaderboard.csv \
  --state q3
```

### Step 8: Validate and Report
```bash
# Check all artifacts exist
ls -lh reports/champion_runs/latest/

# Verify leaderboards
head -20 reports/champion_runs/latest/*_leaderboard.csv
```

---

## ✅ SUCCESS CRITERIA

### For Each Game State:
1. ✅ All 8 models evaluated successfully
2. ✅ Fold metrics CSV contains all models × all folds
3. ✅ Leaderboard generated with champion rankings
4. ✅ No errors in logs (Traceback, ImportError, etc.)
5. ✅ Artifacts are fresh and non-empty
6. ✅ Calibration metrics within acceptable ranges

### Overall:
1. ✅ All 3 game states tested successfully
2. ✅ Champions identified for each state
3. ✅ Consistent methodology applied across all states
4. ✅ Results are reproducible (same seed)
5. ✅ Ready for production deployment

---

## 🎓 CHAMPION SELECTION CRITERIA

### Primary Selection:
- **Lowest MAE_total** (primary metric)
- **Stable across folds** (low std deviation)
- **Good calibration** (PI80 coverage near 0.80)

### Secondary Selection:
- **Brier score** (win probability calibration)
- **RMSE** (error penalization)
- **Consistency** across game states

---

## 📝 NOTES

1. **No promotion** until all validation gates pass
2. **Manual review** of champion candidates before deployment
3. **Reproducibility** ensured via fixed seed (42)
4. **Timeout safety** prevents runaway tuning (30 min per model)
5. **Incremental execution** allows monitoring and intervention

---

**Last Updated:** February 12, 2026
**Status:** Ready for execution