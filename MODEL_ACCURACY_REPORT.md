# PerryPicks v3 - Champion Model Accuracy Report

**Report Date:** February 9, 2025
**Purpose:** Identify % accuracy of each champion model for each game state and each metric
**Disclaimer:** This report documents findings WITHOUT making any changes to the system.

---

## Executive Summary

This report compiles accuracy metrics for all champion models across three game states:
1. **Pregame** - Before games start
2. **Halftime** - At halftime (end of Q2)
3. **Q3** - After end of Q3

Multiple sources document different metrics. The most reliable and comprehensive source is **VIBE_EXECUTION_REPORT.md** which performed rigorous backtesting with champion model selection.

---

## Source Documents Reviewed

| Document | Date | Focus | Reliability |
|----------|------|-------|------------|
| **VIBE_EXECUTION_REPORT.md** | 2026-02-05 | Champion model selection, comprehensive backtesting | HIGH - Rigorous statistical testing |
| **README_MODELS.md** | Current | Model usage guide, champion models | HIGH - Production documentation |
| **UPDATE_SUMMARY.md** | 2026-01-31 | Out-of-sample test on 33 games | HIGH - Real-world performance |
| **ML_IMPROVEMENTS_REPORT.md** | 2026-02-01 | Feature engineering and model improvements | HIGH - Testing results |
| **BASELINE_COMPARISON_REPORT.md** | 2026-01-31 | Baseline performance comparison | MEDIUM - Multiple evaluation methods |
| **pregame_predictions_vs_actual_LAST_4_DAYS_COMPLETE.csv** | 2026-01-30 | 33 games of actual predictions | HIGH - Real data |

---

## Pregame Model Accuracy

### Champion Model: **Ridge Regression (twohead)**

**Source:** VIBE_EXECUTION_REPORT.md (2026-02-05)

#### Test Metrics

| Metric | Value | Context |
|--------|-------|---------|
| **Total MAE** | **3.508** points | Root Mean Absolute Error on test set |
| **Total RMSE** | **4.389** points | Root Mean Squared Error |
| **Total R²** | **0.9493** | 94.93% variance explained |
| **Margin MAE** | **3.343** points | Absolute error in point spread |
| **Margin RMSE** | **4.173** points | RMS error in point spread |
| **Margin R²** | **0.9279** | 92.79% variance explained |
| **Total Games Tested** | **3,520** | Cross-validation on full dataset |

#### Statistical Significance

| Comparison | Diebold-Mariano Test | P-value | Significance |
|-----------|-------------------|----------|-------------|
| Ridge vs Random Forest | DM=-6.114 | 1.11e-05 | **HIGH** - Ridge is statistically superior |
| Ridge vs GBT | DM=-3.815 | 2.11e-03 | **HIGH** - Ridge is statistically superior |

**Conclusion:** Ridge regression is statistically superior to both Random Forest and Gradient Boosting Trees.

---

### Real-World Performance (Out-of-Sample)

**Source:** UPDATE_SUMMARY.md + pregame_predictions_vs_actual_LAST_4_DAYS_COMPLETE.csv

| Metric | Value | Details |
|--------|-------|---------|
| **Games Tested** | **33** | Jan 27-30, 2026 (4 days) |
| **Winner Accuracy** | **90.9%** (30/33 correct) | Excellent performance |
| **Total MAE** | **3.37** points | Close to backtest (3.508) |
| **Margin MAE** | **3.80** points | Close to backtest (3.343) |
| **Within 3 pts** | **51.5%** (17/33) | Accuracy within 3 points |
| **Within 5 pts** | **69.7%** (23/33) | Accuracy within 5 points |
| **Within 10 pts** | **97.0%** (32/33) | Nearly perfect |

**Analysis:** The pregame model generalizes excellently to out-of-sample data:
- Winner accuracy of 90.9% is very strong
- Total MAE of 3.37 is within expectations
- 97.0% of predictions within 10 points

---

## Halftime Model Accuracy

### Champion Model: **Ridge Regression (twohead)**

**Source:** VIBE_EXECUTION_REPORT.md (2026-02-05)

#### Test Metrics

| Metric | Value | Context |
|--------|-------|---------|
| **Total MAE** | **1.183** points | Second half total prediction error |
| **Total RMSE** | **3.273** points | RMS error |
| **Total R²** | **0.6000** | 60.0% variance explained |
| **Margin MAE** | **0.638** points | Second half margin error |
| **Margin RMSE** | **1.224** points | RMS error |
| **Margin R²** | **0.5500** | 55.0% variance explained |
| **Total Folds Tested** | **11** | 11-fold cross-validation |
| **Games Tested** | **2,200** | Backtest on historical games |

#### Additional Metrics

| Metric | Value |
|--------|-------|
| **Average ROI** | **12.24%** | Potential return on investment |

**Conclusion:** The halftime model achieves excellent accuracy:
- Total MAE of 1.183 means predictions are within ~1.2 points on average
- Margin MAE of 0.638 is extremely precise
- 12.24% ROI indicates strong predictive power

---

## Q3 Model Accuracy

### Champion Model: **Ridge Regression (twohead)**

**Source:** VIBE_EXECUTION_REPORT.md (2026-02-05)

#### Test Metrics

| Metric | Value | Context |
|--------|-------|---------|
| **Total MAE** | **6.549** points | Final game prediction from Q3 |
| **Total RMSE** | **9.275** points | RMS error |
| **Total R²** | **0.7699** | 76.99% variance explained |
| **Margin MAE** | **4.717** points | Final margin prediction from Q3 |
| **Margin RMSE** | **5.940** points | RMS error |
| **Margin R²** | **0.8541** | 85.41% variance explained |
| **Total Folds Tested** | **6** | 6-fold cross-validation |
| **Games Tested** | **2,000** | Backtest on historical games |

#### Statistical Significance

| Comparison | Diebold-Mariano Test | P-value | Significance |
|-----------|-------------------|----------|-------------|
| Ridge vs Random Forest | DM=-3.174 | 8.09e-02 | LOW - Not statistically significant |
| Ridge vs GBT | DM=-1.317 | 3.08e-01 | LOW - Not statistically significant |

**Conclusion:** Ridge regression was selected based on lowest MAE, though differences are not statistically significant.

---

## Comparative Summary: All Champion Models

### MAE Comparison (Lower is Better)

| Game State | Champion Model | Total MAE | Margin MAE | Total R² | Margin R² | Games Tested |
|-----------|---------------|-----------|------------|----------|-----------|-------------|
| **Pregame** | Ridge (twohead) | **3.508** | **3.343** | 0.9493 | 0.9279 | 3,520 |
| **Halftime** | Ridge (twohead) | **1.183** | **0.638** | 0.6000 | 0.5500 | 2,200 |
| **Q3** | Ridge (twohead) | **6.549** | **4.717** | 0.7699 | 0.8541 | 2,000 |

**Observations:**
1. **Halftime model is most accurate** (lowest MAE for both total and margin)
2. **Pregame model is second most accurate** with excellent R² scores
3. **Q3 model is least accurate** but still good (76.99% variance explained)

### Winner Accuracy (Real-World)

| Game State | Champion Model | Winner Accuracy | Sample Size | Source |
|-----------|---------------|----------------|-------------|--------|
| **Pregame** | Ridge (twohead) | **90.9%** (30/33) | 33 games | Out-of-sample test |

**Note:** Halftime and Q3 winner accuracy not documented in reviewed sources.

---

## Accuracy by Metric

### Total Points Prediction Accuracy

| Game State | MAE (points) | RMSE (points) | R² | Interpretation |
|-----------|-------------|---------------|----|----------------|
| Pregame | 3.508 | 4.389 | 0.9493 | Excellent - 94.93% variance explained |
| Halftime | 1.183 | 3.273 | 0.6000 | Good - 60.0% variance explained (2nd half only) |
| Q3 | 6.549 | 9.275 | 0.7699 | Good - 76.99% variance explained |

**Ranking:** Halftime (best) → Pregame (2nd) → Q3 (3rd)

### Margin Prediction Accuracy

| Game State | MAE (points) | RMSE (points) | R² | Interpretation |
|-----------|-------------|---------------|----|----------------|
| Pregame | 3.343 | 4.173 | 0.9279 | Excellent - 92.79% variance explained |
| Halftime | 0.638 | 1.224 | 0.5500 | Excellent - 55.0% variance explained (2nd half only) |
| Q3 | 4.717 | 5.940 | 0.8541 | Good - 85.41% variance explained |

**Ranking:** Halftime (best) → Pregame (2nd) → Q3 (3rd)

---

## Performance Percentages

### Pregame Model Performance Breakdown (33 OOS Games)

| Metric | Percentage | Value |
|--------|-------------|-------|
| **Winner Accuracy** | **90.9%** | 30/33 correct |
| **Total within 3 pts** | **51.5%** | 17/33 games |
| **Total within 5 pts** | **69.7%** | 23/33 games |
| **Total within 10 pts** | **97.0%** | 32/33 games |
| **Variance Explained (Total)** | **94.9%** | R² = 0.9493 |
| **Variance Explained (Margin)** | **92.8%** | R² = 0.9279 |

### Halftime Model Performance Breakdown

| Metric | Percentage | Value |
|--------|-------------|-------|
| **Variance Explained (Total)** | **60.0%** | R² = 0.6000 (2nd half only) |
| **Variance Explained (Margin)** | **55.0%** | R² = 0.5500 (2nd half only) |
| **Average ROI** | **12.24%** | Potential betting returns |

### Q3 Model Performance Breakdown

| Metric | Percentage | Value |
|--------|-------------|-------|
| **Variance Explained (Total)** | **76.99%** | R² = 0.7699 |
| **Variance Explained (Margin)** | **85.41%** | R² = 0.8541 |

---

## Model Selection Rationale

### Pregame Champion: Ridge Regression

**Reasons for Selection:**
1. **Lowest Total MAE:** 3.508 vs 5.477 (RF) vs 4.323 (GBT)
2. **Lowest Margin MAE:** 3.343 vs 4.919 (RF) vs 3.778 (GBT)
3. **Highest R²:** 0.9493 total, 0.9279 margin
4. **Statistically Superior:** Diebold-Mariano test shows significant improvement over RF and GBT

**Statistical Significance:** HIGH
- P-value vs Random Forest: 1.11e-05 (highly significant)
- P-value vs GBT: 2.11e-03 (significant)

### Halftime Champion: Ridge Regression

**Reasons for Selection:**
1. **Lowest Total MAE:** 1.183 (only model tested)
2. **Lowest Margin MAE:** 0.638 (only model tested)
3. **High ROI:** 12.24% average return
4. **Simplicity:** Ridge regression is simple and interpretable

**Statistical Significance:** N/A
- Only one model type tested (Ridge)
- Performance is excellent regardless

### Q3 Champion: Ridge Regression

**Reasons for Selection:**
1. **Lowest Total MAE:** 6.549 vs 7.522 (RF) vs 6.895 (GBT)
2. **Competitive Margin MAE:** 4.717 vs 4.968 (RF) vs 3.875 (GBT)
3. **Best Overall Performance:** Selected based on combined total and margin performance

**Statistical Significance:** LOW
- P-value vs Random Forest: 8.09e-02 (not significant)
- P-value vs GBT: 3.08e-01 (not significant)
- Differences are not statistically significant
- Selected based on lowest MAE despite no statistical significance

---

## Historical Context: Previous Models

### Alternative Documented Champions (From README_MODELS.md)

| Game State | Alternative Champion | Total MAE | Margin MAE | R² (Total) | R² (Margin) | Status |
|-----------|---------------------|-----------|------------|-------------|--------------|--------|
| Pregame | Neural Network | 9.58 | 2.95 | 0.579 | 0.673 | Different source |
| Halftime | XGBoost | 7.92 | 6.03 | 0.551 | 0.536 | Different source |
| Q3 | Neural Network | 8.34 | 6.58 | 0.538 | 0.685 | Different source |

**Note:** These metrics differ from VIBE_EXECUTION_REPORT.md. The discrepancy may be due to:
- Different training datasets
- Different evaluation methods
- Different time periods
- Different feature sets

**Recommendation:** Use VIBE_EXECUTION_REPORT.md as the authoritative source for champion model selection and performance.

---

## Additional Performance Metrics (From ML_IMPROVEMENTS_REPORT.md)

### Enhanced Features Performance

| Feature Set | Total MAE | Margin MAE | Improvement |
|-------------|-----------|------------|-------------|
| Baseline (34 features) | 15.92 | 12.00 | - |
| Enhanced (46 features) | 15.62 | 11.21 | -0.30 / -0.79 |

### Individual Models vs Ensembles

| Approach | Total MAE | Margin MAE |
|----------|-----------|------------|
| Best Individual (RF) | 15.62 | 11.24 |
| Best Ensemble (Best 2) | 15.65 | 11.19 |
| Best Individual (Linear) | 15.72 | 11.21 |
| Best Ensemble (Simple Avg) | 15.65 | 11.14 |

**Finding:** Ensembles provide minimal improvement over best individual models (0.0-0.07 MAE)

---

## Champion Model Configuration

### Current Champion Models (From VIBE_EXECUTION_REPORT.md)

```json
{
  "pregame": {
    "total": "ridge_twohead.joblib",
    "margin": "ridge_twohead.joblib",
    "winner": "ridge_twohead.joblib",
    "team_total": "ridge_twohead.joblib",
    "best_mae": 3.508
  },
  "halftime": {
    "total": "ridge_twohead.joblib",
    "margin": "ridge_twohead.joblib",
    "winner": "ridge_twohead.joblib",
    "team_total": "ridge_twohead.joblib",
    "best_mae": 0.638
  },
  "q3": {
    "total": "ridge_twohead.joblib",
    "margin": "ridge_twohead.joblib",
    "winner": "ridge_twohead.joblib",
    "team_total": "ridge_twohead.joblib",
    "best_mae": 6.549
  }
}
```

---

## Key Findings Summary

### 🏆 Best Performing Model
**Halftime Model (Ridge Regression)**
- Total MAE: 1.183 points
- Margin MAE: 0.638 points
- Most accurate of all three models

### 📊 Pregame Model Performance
- Winner accuracy: 90.9% (30/33 games)
- Total MAE: 3.508 points
- Margin MAE: 3.343 points
- Excellent out-of-sample generalization

### 🎯 Q3 Model Performance
- Total MAE: 6.549 points
- Margin MAE: 4.717 points
- Good but least accurate of three models
- Selected despite no statistical significance

### 📈 Overall Trends
1. **Halftime > Pregame > Q3** in accuracy ranking
2. **All models use Ridge Regression** as champion
3. **High R² scores** indicate good fit to data
4. **Out-of-sample performance matches backtest** for pregame model

---

## Metrics Not Documented

The following metrics were NOT found in reviewed sources:

### Halftime Model
- ❌ Winner accuracy (real-world or backtest)
- ❌ Spread betting hit rate
- ❌ Over/under betting hit rate

### Q3 Model
- ❌ Winner accuracy (real-world or backtest)
- ❌ Spread betting hit rate
- ❌ Over/under betting hit rate

---

## Data Gaps Identified

1. **Missing real-world performance for Halftime and Q3**
   - Only pregame model has out-of-sample test results
   - Need to test Halftime and Q3 on completed games

2. **Missing betting-specific metrics**
   - No documentation of spread hit rate
   - No documentation of over/under hit rate
   - No documentation of betting line accuracy

3. **Conflicting sources**
   - VIBE_EXECUTION_REPORT.md shows Ridge as champion for all models
   - README_MODELS.md shows different champions (NN, XGBoost)
   - Need clarification on which is authoritative

---

## Conclusions

### Primary Findings

1. **All three game states use Ridge Regression as the champion model**
2. **Halftime model is most accurate** (MAE: 1.183 total, 0.638 margin)
3. **Pregame model achieves 90.9% winner accuracy** in real-world testing
4. **Q3 model is least accurate** but still performs well (R²: 0.7699-0.8541)

### Model Accuracy Rankings (by Total MAE)

1. **Halftime:** 1.183 points (⭐ Best)
2. **Pregame:** 3.508 points (⭐ Good)
3. **Q3:** 6.549 points (⭐ Fair)

### Model Accuracy Rankings (by Margin MAE)

1. **Halftime:** 0.638 points (⭐ Best)
2. **Pregame:** 3.343 points (⭐ Good)
3. **Q3:** 4.717 points (⭐ Fair)

### Model Accuracy Rankings (by R²)

**Total Points:**
1. **Pregame:** 0.9493 (⭐ Excellent)
2. **Q3:** 0.7699 (⭐ Good)
3. **Halftime:** 0.6000 (⭐ Fair - but only 2nd half)

**Margin:**
1. **Pregame:** 0.9279 (⭐ Excellent)
2. **Q3:** 0.8541 (⭐ Good)
3. **Halftime:** 0.5500 (⭐ Fair - but only 2nd half)

---

## Recommendations for Further Analysis

1. **Run out-of-sample tests for Halftime and Q3 models**
   - Match pregame model's 33-game test
   - Document real-world winner accuracy
   - Document betting-specific hit rates

2. **Clarify champion model selection**
   - Resolve discrepancy between VIBE_EXECUTION_REPORT.md and README_MODELS.md
   - Determine which models are currently in production

3. **Add betting performance metrics**
   - Track spread hit rate
   - Track over/under hit rate
   - Track ROI for different bet types

4. **Implement confidence calibration**
   - Validate confidence scores
   - Ensure 70% confidence = 70% win rate

---

**Report End**

**Document Status:** Complete ✅
**Date:** February 9, 2025
**Prepared by:** Perry 🐶 (code-puppy-0c2adb)
**Purpose:** Document accuracy metrics without making changes
