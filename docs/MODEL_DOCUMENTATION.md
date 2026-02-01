# Perry Picks v3 - Model Documentation

**Generated:** 2025-01-31  
**Commit:** e905c68

## Overview

This document describes all production models for Perry Picks v3.
All models use Ridge regression (L2 regularized linear regression) with alpha=2.0.

## Champion Models

### Halftime (3 seasons)
- **Dataset:** `data/processed/halftime_training_3_seasons.parquet` (3,520 games)
- **Models:** 
  - `models/team_2h_total.joblib` (second half total prediction)
  - `models/team_2h_margin.joblib` (second half margin prediction)
- **CV Results:** `data/processed/halftime_cv_results_3_seasons.parquet`

### Pregame (3 seasons)
- **Dataset:** `data/processed/pregame_team_v2.parquet` (3,520 games)
- **Models:**
  - `models_v3/pregame/ridge_total.joblib` (final total prediction)
  - `models_v3/pregame/ridge_margin.joblib` (final margin prediction)
- **CV Results:** `data/processed/pregame_cv_results.parquet`

### Q3 (2 seasons)
- **Dataset:** `data/processed/q3_team_v2.parquet` (2,000 games)
- **Models:**
  - `models_v3/q3/ridge_total.joblib` (final total prediction)
  - `models_v3/q3/ridge_margin.joblib` (final margin prediction)
- **CV Results:** `data/processed/q3_cv_results.parquet`

## Model Performance

### Halftime Model
- **Games:** 3,520 (3 seasons: 23-24, 24-25, 25-26)
- **Folds:** 14 walk-forward temporal CV
- **Champion:** RIDGE
- **MAE (h2_total):** 11.23
- **RMSE (h2_total):** 14.48
- **Calibrated SD (total):** 13.90
- **Calibrated SD (margin):** 95.68
- **Features:** 12 (h1_home, h1_away, h1_total, h1_margin, h1_events, h1_n_2pt, h1_n_3pt, h1_n_turnover, h1_n_rebound, h1_n_foul, h1_n_timeout, h1_n_sub)

### Pregame Model
- **Games:** 3,520 (3 seasons: 23-24, 24-25, 25-26)
- **Folds:** 11 walk-forward temporal CV
- **Champion:** RIDGE
- **MAE (total):** 3.51
- **RMSE (total):** 4.39
- **R² (total):** 0.949
- **Calibrated SD (total):** 4.00
- **Calibrated SD (margin):** 3.91
- **Features:** 14 (team stats: home_efg, home_ftr, home_tpar, home_tor, home_orbp, away_efg, away_ftr, away_tpar, away_tor, away_orbp, home_fga, home_fgm, away_fga, away_fgm)

### Q3 Model
- **Games:** 2,000 (2 seasons: 23-24, 24-25)
- **Folds:** 6 walk-forward temporal CV
- **Champion:** RIDGE
- **MAE (total):** 6.55
- **RMSE (total):** 9.28
- **R² (total):** 0.770
- **Calibrated SD (total):** 6.59
- **Calibrated SD (margin):** 5.09
- **Features:** 22 (q3_home, q3_away, q3_total, q3_margin, q3_events, q3_n_2pt, q3_n_3pt, q3_n_turnover, q3_n_rebound, q3_n_foul, q3_n_timeout, q3_n_sub, plus team stats)

## Usage

### Loading Models

```python
import joblib

# Load halftime model (PRODUCTION)
model_total = joblib.load("models/team_2h_total.joblib")
model_margin = joblib.load("models/team_2h_margin.joblib")

# Extract model components
model_obj = model_total["model"]  # Ridge sklearn model
features = model_total["features"]  # List of feature names
sd = model_total["sd"]  # Calibrated standard deviation for 80% CI
model_name = model_total["model_name"]  # "RIDGE"

# Load pregame model (RESEARCH)
pregame_total = joblib.load("models_v3/pregame/ridge_total.joblib")
pregame_margin = joblib.load("models_v3/pregame/ridge_margin.joblib")

# Load Q3 model (RESEARCH)
q3_total = joblib.load("models_v3/q3/ridge_total.joblib")
q3_margin = joblib.load("models_v3/q3/ridge_margin.joblib")
```

### Making Predictions

```python
import pandas as pd

# Prepare features in order
X = pd.DataFrame([[...]])[features]  # Feature values matching model's feature list

# Make prediction
prediction = model_obj.predict(X.values)

# 80% confidence interval
lower = prediction - (sd * 1.2816)
upper = prediction + (sd * 1.2816)
```

## Production Status

| Model | Status | Model Path | Target | Games | MAE | SD |
|-------|--------|-----------|---------|--------|-----|----|
| **Halftime Total** | ACTIVE | models/team_2h_total.joblib | h2_total | 3,520 | 11.23 | 13.90 |
| **Halftime Margin** | ACTIVE | models/team_2h_margin.joblib | h2_margin | 3,520 | — | 95.68 |
| **Pregame Total** | RESEARCH | models_v3/pregame/ridge_total.joblib | total | 3,520 | 3.51 | 4.00 |
| **Pregame Margin** | RESEARCH | models_v3/pregame/ridge_margin.joblib | margin | 3,520 | — | 3.91 |
| **Q3 Total** | RESEARCH | models_v3/q3/ridge_total.joblib | total | 2,000 | 6.55 | 6.59 |
| **Q3 Margin** | RESEARCH | models_v3/q3/ridge_margin.joblib | margin | 2,000 | — | 5.09 |

## Training Methodology

All models were trained using the following methodology:

1. **Dataset Collection:** Combined multiple seasons of NBA game data
2. **Feature Engineering:** Extracted relevant features for each game state
3. **Walk-Forward Temporal Cross-Validation:**
   - Train on historical data, test on future games
   - 6-14 folds depending on dataset size
   - Test size: 200 games per fold
4. **Model Comparison:** Compared Ridge, Random Forest, and Gradient Boosting Trees
5. **Statistical Significance:** Diebold-Mariano test for forecast accuracy
6. **Champion Selection:** Selected Ridge based on lowest MAE and statistical significance
7. **Final Training:** Trained Ridge on full dataset
8. **Confidence Interval Calibration:** Used 80% quantile of residuals

## Diebold-Mariano Test Results

| Model | Dataset | DM vs RF (p) | DM vs GBT (p) | Significance |
|-------|----------|----------------|------------------|-------------|
| Halftime | 3,520 games | 0.355 | 0.278 | LOW (lowest MAE) |
| Pregame | 3,520 games | 1.11e-05 | 2.11e-03 | HIGH (p < 0.05) |
| Q3 | 2,000 games | 0.081 | 0.308 | LOW (lowest MAE) |

## Summary

| Model | Games | MAE | RMSE | R² | Significance |
|-------|--------|------|------|----|-------------|
| Halftime | 3,520 | 11.23 | 14.48 | — | LOW |
| Pregame | 3,520 | 3.51 | 4.39 | 0.949 | HIGH |
| Q3 | 2,000 | 6.55 | 9.28 | 0.770 | LOW |

**Conclusion:** Ridge regression is the champion model for all three game states, with Pregame showing the highest statistical significance. All 6 models (3 models × 2 targets each) have been created and calibrated with 80% confidence intervals.
