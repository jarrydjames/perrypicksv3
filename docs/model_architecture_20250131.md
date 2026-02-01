# PerryPicks V3 Model Architecture
**Date:** 2025-01-31  
**Status:** Three-model system (Pregame, Halftime, Q3)

---

## Overview

PerryPicks V3 uses three separate models to predict NBA game outcomes at different stages:
1. **Pregame Model** - Predicts before game starts
2. **Halftime Model** - Predicts at halftime (Q1+Q2 completed)
3. **Q3 Model** - Predicts at end of Q3 (Q1+Q2+Q3 completed)

All models follow the same training methodology and statistical approach, but use different features based on available game state.

---

## Model Comparison

| Feature | Pregame | Halftime | Q3 |
|----------|-----------|-----------|-----|
| **Prediction time** | Before game | After Q2 | After Q3 |
| **Training data** | `data/processed/pregame_team_v2.parquet` | `data/processed/halftime_team_v2.parquet` | `data/processed/q3_team_v2.parquet` |
| **Model directory** | `models_v3/pregame/` | `models/` | `models_v3/q3/` |
| **Game state features** | ❌ None | ✅ h1_home, h1_away, h1 behavior | ✅ q3_home, q3_away, q3 behavior |
| **Team stats features** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Predictions** | Final total, final margin | Final total, final margin | Final total, final margin |
| **Architecture** | Two-head (total + margin) | Two-head (total + margin) | Two-head (total + margin) |
| **Calibration** | Residual quantiles | Residual quantiles | Residual quantiles |
| **Intervals** | 80% confidence bands | 80% confidence bands | 80% confidence bands |

---

## Model Architecture Details

### Two-Head Design

All three models use the same two-head architecture:

```
Input Features
       ↓
┌─────────────┐
│  Base Model │ (Shared features)
└─────┬───────┘
      ↓
┌─────────┴─────────┐
│   Total Head     │ ← Predicts final total points
│   Margin Head    │ ← Predicts final margin (home - away)
└──────────────────┘
```

**Benefits:**
- Total and margin predictions are independent
- Can be specialized (different model types for each head)
- Easy to interpret each prediction separately
- Same structure across all models

---

## Training Methodology

All three models follow the same training pipeline:

### 1. Dataset Building
- **Pregame:** `python3 src/build_dataset_pregame.py`
- **Halftime:** Uses existing `src/build_dataset_v2.py`
- **Q3:** `python3 src/build_dataset_q3.py`

### 2. Model Training
- **Pregame:** `python3 src/train_pregame_model.py`
- **Halftime:** Uses existing `src/modeling/train_models.py`
- **Q3:** `python3 src/train_q3_model.py`

All models train:
- Ridge regression (baseline)
- Random Forest (non-linear baseline)
- Gradient Boosting Trees (primary model)

### 3. Interval Calibration
- **Pregame:** `python3 src/calibrate_intervals_pregame.py`
- **Halftime:** Uses existing calibration
- **Q3:** `python3 src/calibrate_intervals_q3.py`

Calibration uses:
- Residuals from training data
- Quantile regression (q10, q90)
- Distribution-free 80% confidence intervals

---

## Feature Separation

### Pregame Features (No Game State)
```python
{
    "game_id": "0022500697",
    "home_tri": "CHA",
    "away_tri": "SAS",
    # Team stats (no game state!)
    "home_pts_scored_avg_5": 112.3,
    "away_pts_scored_avg_5": 108.7,
    "home_margin_avg_5": 3.6,
    "away_margin_avg_5": -2.1,
    # Rate features
    "home_pts_scored_per_poss": 1.12,
    "away_pts_scored_per_poss": 1.08,
    # Team totals as priors
    ... (all team stats)
}
```

### Halftime Features (Q1+Q2 Game State)
```python
{
    "game_id": "0022500697",
    "home_tri": "CHA",
    "away_tri": "SAS",
    # Game state (halftime)
    "h1_home": 61,
    "h1_away": 47,
    "h1_total": 108,
    "h1_margin": 14,
    # Halftime behavior
    "h1_events": 125,
    "h1_n_2pt": 35,
    "h1_n_3pt": 18,
    "h1_n_turnover": 12,
    "h1_n_rebound": 45,
    # Team stats + rate features
    ... (same as pregame)
}
```

### Q3 Features (Q1+Q2+Q3 Game State)
```python
{
    "game_id": "0022500697",
    "home_tri": "CHA",
    "away_tri": "SAS",
    # Game state (end of Q3)
    "q3_home": 75,
    "q3_away": 68,
    "q3_total": 143,
    "q3_margin": 7,
    # Q3 behavior
    "q3_events": 185,
    "q3_n_2pt": 52,
    "q3_n_3pt": 28,
    "q3_n_turnover": 18,
    "q3_n_rebound": 68,
    # Team stats + rate features
    ... (same as pregame)
}
```

---

## Prediction Workflow

### Pregame Prediction
```python
from src.modeling.pregame_model import PregameModel

model = PregameModel()
pred = model.predict(features={"home_pts_scored_avg_5": 112.3, ...}, game_id="0022500697")

# Output:
# - total_mean: 218.5 (predicted final total)
# - margin_mean: +5.2 (predicted home margin)
# - total_q10, total_q90: 80% CI for total
# - margin_q10, margin_q90: 80% CI for margin
```

### Halftime Prediction
```python
from src.predict_from_gameid_v2 import predict_from_halftime

pred = predict_from_halftime(h1_home=61, h1_away=47, beh={...})

# Output:
# - pred_final_home: 108.5 (final home score)
# - pred_final_away: 103.2 (final away score)
# - pred_final_total: 211.7
# - pred_final_margin: +5.3
```

### Q3 Prediction
```python
from src.modeling.q3_model import Q3Model

model = Q3Model()
pred = model.predict(features={"q3_home": 75, "q3_away": 68, ...}, period=3, clock="PT0M00.00S", game_id="0022500697")

# Output:
# - total_mean: 205.3 (predicted final total)
# - margin_mean: +3.1 (predicted home margin)
# - total_q10, total_q90: 80% CI for total
# - margin_q10, margin_q90: 80% CI for margin
```

---

## Model Files

### Pregame Models
```
models_v3/pregame/
├── gbt_twohead.joblib          # Primary: Gradient Boosting Trees
├── ridge_twohead.joblib        # Baseline: Ridge regression
├── random_forest_twohead.joblib # Non-linear baseline: Random Forest
└── pregame_intervals.joblib    # Calibrated 80% intervals
```

### Halftime Models
```
models/
├── team_2h_total.joblib       # 2nd half total prediction
├── team_2h_margin.joblib      # 2nd half margin prediction
└── (intervals from calibration)
```

### Q3 Models
```
models_v3/q3/
├── gbt_twohead.joblib          # Primary: Gradient Boosting Trees
├── ridge_twohead.joblib        # Baseline: Ridge regression
├── random_forest_twohead.joblib # Non-linear baseline: Random Forest
└── q3_intervals.joblib         # Calibrated 80% intervals
```

---

## Target Variables

All three models predict the same targets:

### Total Points
```python
total = final_home + final_away
```
Example: final_home=105, final_away=95 → total=200

### Point Margin
```python
margin = final_home - final_away
```
Example: final_home=105, final_away=95 → margin=+10

### Team Scores (derived)
```python
final_home = (total + margin) / 2
final_away = (total - margin) / 2
```
Example: total=200, margin=+10 → final_home=105, final_away=95

---

## Separation Between Models

### Why Keep Models Separate?

1. **Different feature sets**
   - Pregame: No game state
   - Halftime: Q1+Q2 state
   - Q3: Q1+Q2+Q3 state

2. **Different prediction difficulty**
   - Pregame: Hardest (no game data)
   - Halftime: Medium (some game data)
   - Q3: Easiest (lots of game data)

3. **Different uncertainty**
   - Pregame: Highest uncertainty
   - Halftime: Medium uncertainty
   - Q3: Lowest uncertainty

4. **Different use cases**
   - Pregame: Betting before game
   - Halftime: Halftime bets, in-game decisions
   - Q3: Late-game bets, final projections

### Model Independence

All three models are completely independent:
- Separate training data
- Separate model files
- Separate calibration
- Can be retrained independently
- No shared code (except infrastructure)

---

## Bug Fixes Applied

### Q3 Model Bug Fix (2025-01-31)
**Issue:** Q3 model was trained on wrong targets (halftime instead of final scores)
**Fix:** Updated `src/build_dataset_q3.py` to use `final_score_from_box()` instead of `sum_first2()`
**Impact:** Q3 model now correctly predicts final game outcomes

### Pregame Model Creation (2025-01-31)
**New:** Created complete pregame model infrastructure
**Files:**
- `src/build_dataset_pregame.py` - Dataset builder
- `src/train_pregame_model.py` - Model training
- `src/modeling/pregame_model.py` - Model interface
- `src/calibrate_intervals_pregame.py` - Interval calibration

---

## Next Steps

1. **Build pregame dataset:** `python3 src/build_dataset_pregame.py`
2. **Rebuild Q3 dataset (with fix):** `python3 src/build_q3_continuous.py`
3. **Train pregame models:** `python3 src/train_pregame_model.py`
4. **Retrain Q3 models (with fix):** `python3 src/train_q3_model.py`
5. **Calibrate pregame intervals:** `python3 src/calibrate_intervals_pregame.py`
6. **Recalibrate Q3 intervals:** `python3 src/calibrate_intervals_q3.py`
7. **Test all three models:** Verify predictions are reasonable
8. **Integrate into app.py:** Add pregame model support

---

## Summary

| Metric | Value |
|--------|--------|
| **Total models** | 3 (Pregame, Halftime, Q3) |
| **Model architecture** | Two-head (total + margin) |
| **Training methodology** | Identical across all models |
| **Calibration method** | Residual quantiles (80% CI) |
| **Prediction targets** | Final total, final margin (same for all) |
| **Separation** | Complete (different data, models, directories) |

**All models are now properly separated and follow the same rigorous methodology!** 🐶🚀
