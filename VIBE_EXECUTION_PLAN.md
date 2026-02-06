# PerryPicks V3 — Start-to-Finish Guided Plan (Vibe-Coding Ready)

## Goal:
Provide a comprehensive, executable plan for a vibe-coding platform to fully train, calibrate, evaluate, and select models across **all game states** and **all intended metrics** (game total, margin, winner, team totals), with standardized outputs and tests.

---

## 0) Preconditions & Conventions

### 0.1 Folder & Artifact Conventions
- **Datasets:** `data/processed/`
- **Models:** `models_v3/{pregame,halftime,q3}/`
- **Reports:** `data/processed/*_readout.txt` and summary JSONs
- **Contracts:** `data/contracts/*.json`

### 0.2 Model Families
Every state must be trained with **Ridge**, **Random Forest**, and **GBT**.  
Optional: XGBoost/CatBoost for research, never required for production.

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
