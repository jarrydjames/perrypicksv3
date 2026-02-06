# PerryPicks V3 — Start-to-Finish Guided Plan (Vibe-Coding Ready)

## Goal:
Provide a comprehensive, executable plan for a vibe-coding platform to fully train, calibrate, evaluate and select models across **all game states** and **all intended metrics** (game total, margin, winner, team totals), with standardized outputs and tests.

---

## 0) Preconditions & Conventions

### 0.1 Folder & Artifact Conventions
- **Datasets:** `data/processed/`
- **Models:** `models_v3/{pregame,halftime,q3}/`
- **Reports:** `data/processed/*_readout.txt` and summary JSONs
- **Contracts:** `data/contracts/*.json`

### 0.2 Model Families
Every state must be trained with **Ridge**, **Random Forest**, and **GBT**. Optional: XGBoost/CatBoost for research, never required for production.

### 0.3 Required Metrics (All States)
For each model and for each state:
- **Game Total:** MAE, RMSE, R²
- **Margin:** MAE, RMSE, R²
- **Winner:** Accuracy, ROI (if odds available)
- **Team Totals:** MAE, RMSE (home and away totals)
- **Calibration:** 80% interval coverage + width

---

## 1) Data Preparation (All States)

### 1.1 Pregame Dataset
**Build dataset:**
```bash
python src/build_dataset_pregame.py
```

**Output:**
- `data/processed/pregame_team_v2.parquet`

**Required columns (contract):**
- `game_id`, `home_tri`, `away_tri`, `total`, `margin`
- plus temporal features (rest days, recent form, head-to-head, sos_diff)

**Validation:**
```bash
python scripts/validate_dataset.py data/processed/pregame_team_v2.parquet data/contracts/pregame_team_v2.json
```

---

### 1.2 Halftime Dataset
**Build dataset:**
```bash
python src/build_dataset_v2.py
```

**Output:**
- `data/processed/halftime_team_v2.parquet`

**Validation:**
```bash
python scripts/validate_dataset.py data/processed/halftime_team_v2.parquet data/contracts/halftime_team_v2.json
```

---

### 1.3 Q3 Dataset
**Build dataset:**
```bash
python src/build_dataset_q3.py
```

**Output:**
- `data/processed/q3_team_v2.parquet`

**Validation:**
```bash
python scripts/validate_dataset.py data/processed/q3_team_v2.parquet data/contracts/q3_team_v2.json
```

---

## 2) Training (All States, All Models)

### 2.1 Pregame Training
```bash
python src/train_pregame_model.py
```
**Expected outputs:**
- `models_v3/pregame/ridge_twohead.joblib`
- `models_v3/pregame/randomforest_twohead.joblib`
- `models_v3/pregame/gbt_twohead.joblib`

---

### 2.2 Halftime Training
```bash
python src/train_halftime_model.py
```
**Expected outputs:**
- `models_v3/halftime/ridge_twohead.joblib`
- `models_v3/halftime/randomforest_twohead.joblib`
- `models_v3/halftime/gbt_twohead.joblib`

---

### 2.3 Q3 Training
```bash
python src/train_q3_model.py
```
**Expected outputs:**
- `models_v3/q3/ridge_twohead.joblib`
- `models_v3/q3/randomforest_twohead.joblib`
- `models_v3/q3/gbt_twohead.joblib`

---

## 3) Calibration (All States)

### 3.1 Pregame Calibration
```bash
python src/calibrate_intervals_pregame.py
```
**Output:** `models_v3/pregame/pregame_intervals.joblib`

### 3.2 Halftime Calibration
```bash
python src/calibrate_intervals_halftime.py
```
**Output:** `models_v3/halftime/halftime_intervals.joblib`

### 3.3 Q3 Calibration
```bash
python src/calibrate_intervals_q3.py
```
**Output:** `models_v3/q3/q3_intervals.joblib`

---

## 4) Backtesting & Metric Generation (All States)

### 4.1 Pregame Backtest
```bash
python src/backtest_pregame_with_accuracy.py
```
**Outputs:**
- `data/processed/pregame_backtest_results_with_accuracy.parquet`
- `data/processed/pregame_readout.txt`

### 4.2 Halftime Backtest
```bash
python src/backtest_models_full.py
```
**Outputs:**
- `data/processed/halftime_backtest_results.parquet`
- `data/processed/halftime_model_summary.json`

### 4.3 Q3 Backtest
```bash
python src/backtest_v2.py
```
**Outputs:**
- `data/processed/q3_backtest_results.parquet`
- `data/processed/q3_model_summary.json`

---

## 5) Champion Selection (All States)

**Goal:** pick one champion model per state, per metric.

### Required output file:
```
data/processed/champion_models.json
```

**Schema example:**
```json
{
  "pregame": {
    "total": "ridge_twohead.joblib",
    "margin": "ridge_twohead.joblib",
    "winner": "ridge_twohead.joblib",
    "team_total": "ridge_twohead.joblib"
  },
  "halftime": {
    "total": "gbt_twohead.joblib",
    "margin": "gbt_twohead.joblib",
    "winner": "gbt_twohead.joblib",
    "team_total": "gbt_twohead.joblib"
  },
  "q3": {
    "total": "gbt_twohead.joblib",
    "margin": "gbt_twohead.joblib",
    "winner": "gbt_twohead.joblib",
    "team_total": "gbt_twohead.joblib"
  }
}
```

---

## 6) Runtime Integration

### 6.1 Update Prediction Layer
Ensure prediction runtime loads champion models per state and metric.

### 6.2 Validate Output Consistency
All modes must return:
- `total`, `margin`, `winner`, `team_totals`, `bands80`, `model_used`

---

## 7) Testing Checklist (All States)

### 7.1 Unit Tests
- Contracts validation
- Registry tracking
- Temporal features

### 7.2 Integration Tests
- Build dataset → Train → Calibrate → Backtest (smoke run)

### 7.3 Regression Tests
- Validate new models outperform prior champions on same backtest windows

---

## 8) Automation (Optional but Recommended)
- **Scheduler:** `scripts/automation/scheduler.py`
- **Discord Posting:** `scripts/automation/discord_poster.py`
- **Grading:** `scripts/automation/bet_grader.py`

---

## 9) Completion Criteria

The plan is **complete** when:
- ✅ All three states have trained Ridge/RF/GBT models
- ✅ Calibration files exist for all three states
- ✅ Backtest metrics include total, margin, winner, team totals
- ✅ Champion selection file exists
- ✅ Prediction runtime uses champion models

---

## 10) Execution Order (Single Pipeline Run)

1. Build datasets (pregame, halftime, q3)
2. Train models (all states)
3. Calibrate intervals
4. Backtest + generate readouts
5. Champion selection
6. Runtime integration
7. Regression tests

---

**This plan is intentionally explicit so a vibe-coding platform can execute it step-by-step.**
EOF