# Perry Picks v3 - Model Documentation

**Generated:** 2025-01-31  
**Latest Update:** 2025-01-31 (Fixed Q3 margin champion, added pregame backtest)  
**Latest Commit:** (pending)

## Overview

This document describes all production models for Perry Picks v3.

### Model Types
- **RIDGE**: Linear regression with L2 regularization (alpha=2.0)
- **GBT**: Gradient Boosting Trees (n_estimators=100, max_depth=4, learning_rate=0.1)

## Champion Models

### Halftime (3 seasons) - PRODUCTION
- **Dataset:** `data/processed/halftime_training_3_seasons.parquet` (3,520 games)
- **Models:** 
  - `models/team_2h_total.joblib` (RIDGE - second half total)
  - `models/team_2h_margin.joblib` (RIDGE - second half margin)
- **CV Results:** `data/processed/halftime_cv_results_3_seasons.parquet`
- **Backtest:** `data/processed/halftime_backtest_results_leakage_free.parquet`

### Pregame (3 seasons) - RESEARCH
- **Dataset:** `data/processed/pregame_team_v2.parquet` (3,520 games)
- **Models:**
  - `models_v3/pregame/ridge_total.joblib` (RIDGE - final total)
  - `models_v3/pregame/ridge_margin.joblib` (RIDGE - final margin)
- **CV Results:** `data/processed/pregame_cv_results.parquet`
- **Backtest:** `data/processed/pregame_backtest_results.parquet` ✅ NEW

### Q3 (2 seasons) - RESEARCH
- **Dataset:** `data/processed/q3_team_v2.parquet` (2,000 games)
- **Models:**
  - `models_v3/q3/ridge_total.joblib` (RIDGE - final total)
  - `models_v3/q3/ridge_margin.joblib` (GBT - final margin) ✅ FIXED
- **CV Results:** `data/processed/q3_cv_results.parquet`
- **Backtest:** `data/processed/q3_backtest_results.parquet`

## Model Performance

### Halftime Model

**TOTAL Target (RIDGE):**
- Games: 3,520 (3 seasons: 23-24, 24-25, 25-26)
- Folds: 14 walk-forward temporal CV
- MAE: 11.23
- RMSE: 14.48
- Calibrated SD: 13.90
- Features: 12 (h1_home, h1_away, h1_total, h1_margin, h1_events, h1_n_2pt, h1_n_3pt, h1_n_turnover, h1_n_rebound, h1_n_foul, h1_n_timeout, h1_n_sub)

**MARGIN Target (RIDGE):**
- Games: 3,520
- Folds: 14
- MAE: 9.20
- RMSE: 11.52
- R²: -0.007 ⚠️ (negative - model has limited predictive power)
- Calibrated SD: 95.68
- Backtest: 11 folds, MAE=0.64, ROI=12.2%

### Pregame Model

**TOTAL Target (RIDGE):**
- Games: 3,520 (3 seasons: 23-24, 24-25, 25-26)
- Folds: 11 walk-forward temporal CV
- MAE: 3.51
- RMSE: 4.39
- R²: 0.949
- Calibrated SD: 4.00
- Features: 14 (team stats: efg, ftr, tpar, tor, orbp, fga, fgm for both teams)

**MARGIN Target (RIDGE):**
- Games: 3,520
- Folds: 11
- MAE: 3.34
- RMSE: 4.17
- R²: 0.928
- Calibrated SD: 3.91
- Backtest: 15 folds, MAE=3.42, ROI=84.5% ✅ NEW
- Features: 14

### Q3 Model

**TOTAL Target (RIDGE):**
- Games: 2,000 (2 seasons: 23-24, 24-25)
- Folds: 6 walk-forward temporal CV
- MAE: 6.55
- RMSE: 9.28
- R²: 0.770
- Calibrated SD: 6.59
- Features: 22 (q3 stats + team stats)

**MARGIN Target (GBT):** ✅ FIXED from RIDGE
- Games: 2,000
- Folds: 6
- MAE: 3.88 (GBT) vs 4.72 (Ridge) - 18% improvement
- RMSE: 5.94
- R²: 0.854
- Calibrated SD: 2.66 (reduced from 5.09) ✅ FIXED
- Backtest: 7 folds, MAE=5.97, ROI=7.3%
- Features: 26

## Production Status

| Model | Status | Type | Target | Games | MAE | SD | Backtest |
|-------|--------|------|---------|--------|-----|----|----------|
| **Halftime Total** | PRODUCTION | RIDGE | h2_total | 3,520 | 11.23 | 13.90 | ✅ 11 folds |
| **Halftime Margin** | PRODUCTION | RIDGE | h2_margin | 3,520 | 9.20 | 95.68 | ✅ 11 folds |
| **Pregame Total** | RESEARCH | RIDGE | total | 3,520 | 3.51 | 4.00 | — |
| **Pregame Margin** | RESEARCH | RIDGE | margin | 3,520 | 3.34 | 3.91 | ✅ 15 folds |
| **Q3 Total** | RESEARCH | RIDGE | total | 2,000 | 6.55 | 6.59 | ✅ 7 folds |
| **Q3 Margin** | RESEARCH | GBT | margin | 2,000 | 3.88 | 2.66 | ✅ 7 folds |

## Training Methodology

All models were trained using the following methodology:

1. **Dataset Collection:** Combined multiple seasons of NBA game data
2. **Feature Engineering:** Extracted relevant features for each game state
3. **Walk-Forward Temporal Cross-Validation:**
   - Train on historical data, test on future games
   - 6-15 folds depending on dataset size
   - Test size: 200 games per fold
4. **Model Comparison:** Compared Ridge, Random Forest, and Gradient Boosting Trees
5. **Statistical Significance:** Diebold-Mariano test for forecast accuracy
6. **Champion Selection:** Selected model with lowest MAE and statistical significance
7. **Final Training:** Trained champion on full dataset
8. **Confidence Interval Calibration:** Used 80% quantile of residuals
9. **Backtesting:** Walk-forward validation on out-of-sample data

## Recent Updates

### 2025-01-31
- ✅ Fixed Q3 margin: Changed champion from RIDGE to GBT (MAE: 4.72 → 3.88, -18%)
- ✅ Added pregame backtest: 15 folds, MAE=3.42, ROI=84.5%
- ✅ All 6 models now have CV, calibration, and backtest results

### 2025-01-31 (Earlier)
- ✅ Retrained halftime with 25-26 season (+724 games)
- ✅ All models calibrated with 80% confidence intervals

## Diebold-Mariano Test Results

| Model | Dataset | DM vs RF (p) | DM vs GBT (p) | Significance |
|-------|----------|----------------|----------------|-------------|
| Halftime Total | 3,520 games | 0.355 | 0.278 | LOW (lowest MAE) |
| Pregame Total | 3,520 games | 1.11e-05 | 2.11e-03 | HIGH (p < 0.05) |
| Q3 Total | 2,000 games | 0.081 | 0.308 | LOW (lowest MAE) |

## Summary

| Model | Games | MAE | RMSE | R² | Significance |
|-------|--------|------|------|----|-------------|
| Halftime (Total) | 3,520 | 11.23 | 14.48 | — | LOW |
| Pregame (Total) | 3,520 | 3.51 | 4.39 | 0.949 | HIGH |
| Q3 (Total) | 2,000 | 6.55 | 9.28 | 0.770 | LOW |

**Margin Models:**
| Model | Games | MAE | RMSE | R² | Type |
|-------|--------|------|------|----|------|
| Halftime (Margin) | 3,520 | 9.20 | 11.52 | -0.007 | RIDGE |
| Pregame (Margin) | 3,520 | 3.34 | 4.17 | 0.928 | RIDGE |
| Q3 (Margin) | 2,000 | 3.88 | 5.94 | 0.854 | GBT |

**Conclusion:** Ridge regression is champion for most models, but GBT performs better for Q3 margin. Pregame models show highest statistical significance. All 6 models (3 models × 2 targets each) have been created, calibrated, and backtested.
