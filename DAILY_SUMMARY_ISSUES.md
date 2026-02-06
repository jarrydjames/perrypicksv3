# DAILY_SUMMARY Posting Issues & Resolutions

**Date:** 2026-02-06
**Summary:** Documenting issues encountered while generating and posting DAILY_SUMMARY predictions for 2026-02-05

## Model Used for Predictions

The DAILY_SUMMARY predictions were generated using the **PREGAME_V3_FINAL** model:

### Model Architecture
- **Total Score Prediction:** Ridge Regression (`ridge_total_final.pkl`)
- **Margin Prediction:** Random Forest (`rf_margin_final.pkl`)
- **Feature Count:** 72 features
- **Feature Version:** `v3_final_72feat`
- **Model Files:**
  - `data/models/ridge_total_final.pkl`
  - `data/models/rf_margin_final.pkl`

### Model Performance
- **Total MAE:** 15.6 points
- **Margin MAE:** 11.2 points
- **Residual Sigma (Total):** 15.6
- **Residual Sigma (Margin):** 11.2

### Features Used
The model uses 72 features including:
- Team ratings
- Temporal features
- Form data (recent performance)
- Head-to-head (H2H) stats
- Schedule strength

---

## Issues Encountered

### Issue #1: AttributeError - Variable Naming Conflict

**Error:**
```
AttributeError: 'str' object has no attribute 'get'
```

**Location:** `src/predict_api.py` - odds fetching code

**Root Cause:**
The odds API call was attempting to call `.get()` method on global `home_name` and `away_name` variables that were set to strings by the `fetch_box()` function. This occurred when:
- Q3 result path set `home_name` and `away_name` to strings from `fetch_box()`
- Odds API call tried to call `home_name.get()` and `away_name.get()`

**Resolution:**
Changed the odds API call to use `result.get("home_name")` and `result.get("away_name")` instead of global variables:
```python
# Before (caused AttributeError):
home_tricode = home_name.get() if home_name else None
away_tricode = away_name.get() if away_name else None

# After (fixed):
home_tricode = result.get("home_name")
away_tricode = result.get("away_name")
```

**Commits:**
- `fa195f3` - Final fix: uses `result.get()` for team names
- `febffa5` - First attempt to fix variable conflict

---

### Issue #2: CST Date Validation Edge Case

**Description:**
Games starting at `2026-02-06T00:00:00Z` (midnight UTC on Feb 6) were being incorrectly filtered or validated.

**Root Cause:**
The CST date derived from UTC time needed to correctly map to the requested date:
- UTC: `2026-02-06T00:00:00Z`
- CST (America/Chicago): `2026-02-05T18:00:00-06:00`
- Requested date: `2026-02-05`

**Resolution:**
The `cst_game_date_from_start_time_utc()` function correctly handled this conversion:
```python
derived_date = cst_game_date_from_start_time_utc(start_utc, tz='America/Chicago')
# Returns: 