# Fix: Use Trained Q3 Models Instead of Stubs
**Date:** 2025-01-31
**Priority:** HIGH
**Status:** COMPLETED ✅

---

## Problem

**User Concern:**
"App is using model that we put all work into developing and training. This should not have been adjusted."

**Context:**
- User trained sophisticated Q3 models in `models_v3/q3/`
- App was using stub models (placeholders created to fix crash)
- User wanted their trained work to be utilized

---

## Root Cause Analysis

**Why App Used Stubs:**

1. **Default Path:** `eval_at_q3: bool = False`
   - App defaulted to halftime model path
   - Halftime model expects: `models/team_2h_*.joblib`
   - These files didn't exist

2. **Q3 Models Exist:**
   - Trained models in `models_v3/q3/`
   - `ridge_twohead.joblib` (margin/spread)
   - `gbt_twohead.joblib` (game total)
   - `q3_intervals.joblib` (calibration)

3. **FileNotFoundError Triggered:**
   - Code looked for `models/team_2h_*.joblib`
   - Files didn't exist → crash
   - Fixed with stub models → user upset

**Model Mismatch:**
```
Code Expected:        models/team_2h_*.joblib      (halftime, don't exist)
Trained Models:       models_v3/q3/*_twohead.joblib  (Q3, exist)
Documentation Says:    models_v2/*_twohead.joblib      (v2 path, doesn't exist)
```

---

## Solution

**Fix 1: Change Default to Q3 Models**

Changed from:
```python
def predict_from_game_id(
    game_input: str,
    *,
    eval_at_q3: bool = False,  # ← Defaulting to non-existent path!
) -> Dict[str, Any]:
```

To:
```python
def predict_from_game_id(
    game_input: str,
    *,
    eval_at_q3: bool = True,  # ← Use trained Q3 models
) -> Dict[str, Any]:
```

**Fix 2: Remove Stub Models**

```bash
# Deleted placeholder stubs
rm models/team_2h_total.joblib
rm models/team_2h_margin.joblib

# Removed empty directory
rmdir models/
```

---

## Impact

**Immediate:**
- ✅ App now uses trained Q3 models
- ✅ Predictions from actual trained work
- ✅ No more placeholder/stub models
- ✅ User's training work is utilized

**Models Now Used:**
- **Margin/Spread:** `models_v3/q3/ridge_twohead.joblib`
  - Calibration-first probabilities for betting decisions
  - Most stable + best calibrated (coverage close to target)
- **Game Total:** `models_v3/q3/gbt_twohead.joblib`
  - Small model artifact and strong overall performance
  - Drives derived team totals
- **Intervals:** `models_v3/q3/q3_intervals.joblib`
  - 80% confidence bands for all predictions

---

## Model Architecture

**Q3 Model Features:**
- Game-clock aware (evaluates at any game state)
- Supports halftime, end-of-Q3, or during play
- Same structure as halftime model for compatibility
- 27 halftime features (h1 scores, efficiency, possessions, etc.)

**Two-Head Architecture:**
- `ridge_twohead`: Predicts margin + home_win_prob simultaneously
- `gbt_twohead`: Predicts total + variance simultaneously
- Multi-task learning for better calibration

---

## Files Changed

**Modified:**
- `src/predict_from_gameid_v3_runtime.py`
  - Changed default: `eval_at_q3: bool = False` → `True`

**Deleted:**
- `models/team_2h_total.joblib` (stub)
- `models/team_2h_margin.joblib` (stub)
- `models/` directory (now removed)

---

## Summary

Issue: App using stubs instead of trained models  
User Request: Utilize trained Q3 models from models_v3/q3/  
Solution: Changed default to eval_at_q3=True, removed stub models  
Status: COMPLETED AND PUSHED  
Files Changed: 1 modified, 2 deleted, 1 directory removed  
Models Now Used: ridge_twohead, gbt_twohead, q3_intervals  
Ready for: Streamlit Cloud deployment