# CRITICAL BUG: Q3 Model Trained on Wrong Targets
**Date:** 2025-01-31  
**Severity:** **CRITICAL - Models are predicting garbage**  
**Status:** **IDENTIFIED - Needs Fix**  

---

## Executive Summary

**The Q3 model is predicting garbage because it's trained on the WRONG TARGETS.**

### The Bug
The Q3 model training dataset (`src/build_dataset_q3.py`) has a copy-paste error that trains the model to predict **HALFTIME scores** instead of **FINAL game scores**.

### Impact
- **Q3 predictions are mathematically impossible** (team scores don't sum to total)
- **Predictions are based on wrong training data** (halftime targets instead of final game targets)
- **All Q3 model outputs are invalid** until this bug is fixed
- **The model is useless in its current state**

---

## Root Cause Analysis

### Location of Bug
**File:** `src/build_dataset_q3.py`, lines 106-110

```python
# Add final labels (game outcomes - same as halftime)
final_home = sum_first2(home.get("periods"))  # ← BUG: Only sums Q1+Q2!
final_away = sum_first2(away.get("periods"))  # ← BUG: Only sums Q1+Q2!

row["total"] = final_home + final_away  # ← WRONG: Halftime total, not final!
row["margin"] = final_home - final_away  # ← WRONG: Halftime margin, not final!
```

### What sum_first2 Does
```python
def sum_first2(periods):
    """Sum scores from first 2 quarters (halftime)."""
    s = 0
    for p in (periods or []):
        if int(p.get("period", 0)) in (1,2):  # ← Only periods 1 and 2
            for key in ("score","points","pts"):
                if key in p and p[key] is not None:
                    s += int(p[key]); break
    return s
```

### What Should Happen

**Q3 model should predict FINAL GAME SCORES:**
- Input: Features from end of Q3 (q3_home, q3_away, Q3 behavior stats, team stats)
- Output: Final game total, final game margin

**But it's actually trained to predict HALFTIME SCORES:**
- Input: Features from end of Q3 (correct)
- Output: Halftime total, halftime margin (WRONG!)

### Why This Is Disastrous

1. **Model Input:** "After Q3, here's the game state..."
2. **Model Training:** "Predict what the score was at halftime"
3. **Model Output:** "Halftime total: 108, Halftime margin: +5"
4. **Code Interpretation:** "Final total: 108, Final margin: +5" (WRONG!)
5. **Result:** Garbage predictions that don't make sense

---

## Mathematical Proof of Wrong Predictions

### Example: Spurs @ Hornets (Game 0022500697)

**Actual Game State:**
- Halftime: Hornets 61, Spurs 47
- Q3: Hornets 75, Spurs 68
- Final: Hornets 105, Spurs 95 (example)

**What Q3 Model Is Trained To Do:**
- Input: q3_home=75, q3_away=68, [other features]
- **Training target:** total=108 (61+47), margin=+14 (61-47)
- **Model learns:** "Given end-of-Q3 state, predict halftime score"

**What Code Expects:**
- Input: q3_home=75, q3_away=68, [other features]
- **Expected output:** total=200 (105+95), margin=+10 (105-95)
- **Model output:** total=108, margin=+14 (HALFTIME, not final!)

**Result:**
- Model says: "Final total: 108"
- But at Q3, score is already 75+68=143
- **Impossible! Final total (108) < current Q3 total (143)**
- User sees: "Garbage predictions"

---

## Why the Math Fix Didn't Work

I previously fixed the calculation in `predict_from_gameid_v3_runtime.py`:

```python
pred_final_home = q3_home + (pred.total_mean - (q3_home + q3_away)) / 2
pred_final_away = q3_away + (pred.total_mean - (q3_home + q3_away)) / 2
```

This is mathematically correct **IF** `pred.total_mean` is the actual final total.

**But the bug makes `pred.total_mean` = HALFTIME total, not final total!**

So:
- q3_home=75, q3_away=68 (current Q3 score)
- pred.total_mean=108 (model predicts halftime total)
- pred_2h_total = 108 - (75+68) = -35 (IMPOSSIBLE - negative 2nd half points!)
- pred_final_home = 75 + (-35)/2 = 57.5 (less than current score!)
- pred_final_away = 68 + (-35)/2 = 50.5 (less than current score!)

**Mathematically impossible predictions!**

---

## The Fix

### Step 1: Fix build_dataset_q3.py

**File:** `src/build_dataset_q3.py`

**Current (WRONG):**
```python
# Add final labels (game outcomes - same as halftime)
final_home = sum_first2(home.get("periods"))  # ← BUG
final_away = sum_first2(away.get("periods"))  # ← BUG

row["total"] = final_home + final_away
row["margin"] = final_home - final_away
```

**Fixed (CORRECT):**
```python
# Add final labels (game outcomes - same as halftime)
from src.build_dataset_team_v2 import final_score_from_box

fin = final_score_from_box(game)
if fin is None:
    raise ValueError(f"Missing final score for game {gid}")
final_home, final_away = fin

row["total"] = final_home + final_away
row["margin"] = final_home - final_away
```

### Step 2: Rebuild Q3 Dataset

```bash
python3 src/build_q3_continuous.py
```

This will:
- Rebuild `data/processed/q3_team_v2.parquet` with correct targets
- Ensure all games have valid final scores

### Step 3: Retrain Q3 Models

```bash
python3 src/train_q3_model.py
```

This will:
- Retrain all Q3 models (GBT, Ridge, Random Forest)
- Train on correct targets (final total/margin)
- Save updated models to `models_v3/q3/`

### Step 4: Recalibrate Intervals

```bash
python3 src/calibrate_intervals_q3.py
```

This will:
- Recompute residual quantiles from new predictions
- Generate correct 80% confidence intervals
- Save to `models_v3/q3/q3_intervals.joblib`

---

## Verification Plan

After fixes, verify with game ID 0022500697:

1. **Model Predictions Should Be Reasonable:**
   - Final total should be > current Q3 total
   - Team scores should be > current Q3 scores
   - Team scores should sum to total
   - Values should be realistic (100-130 range per team)

2. **Mathematical Coherence:**
   ```
   pred_final_home + pred_final_away = total_mean
   pred_final_home - pred_final_away = margin_mean
   pred_final_home = q3_home + predicted_2h_home
   pred_final_away = q3_away + predicted_2h_away
   ```

3. **UI Display:**
   - Predictions should display correctly
   - No mathematically impossible values
   - Intervals should be reasonable width

---

## Summary of Issues

| Issue | Type | Severity | Status |
|-------|------|----------|--------|
| Q3 model trained on wrong targets (halftime instead of final) | Data Pipeline | CRITICAL | ❌ IDENTIFIED, needs fix |
| Math fix didn't work because model output is wrong | Code | HIGH | ❌ Dependent on above |
| Q3 predictions are mathematically impossible | Model | CRITICAL | ❌ Dependent on above |
| All Q3 outputs are invalid | Model | CRITICAL | ❌ Dependent on above |

---

## Root Cause Category

**Type:** Copy-paste error in data pipeline  
**Origin:** `src/build_dataset_q3.py` line 107-108  
**Impact:** Complete model failure for Q3 predictions  
**Risk Level:** CRITICAL - entire Q3 model is unusable  

---

## Recommendations

1. **IMMEDIATE:** Fix `src/build_dataset_q3.py` to use final scores
2. **IMMEDIATE:** Rebuild Q3 training dataset with correct targets
3. **IMMEDIATE:** Retrain all Q3 models on correct data
4. **IMMEDIATE:** Recalibrate Q3 intervals
5. **SHORT-TERM:** Add data validation to prevent this bug
6. **SHORT-TERM:** Add unit tests for dataset building
7. **MEDIUM-TERM:** Add sanity checks in training pipeline
8. **MEDIUM-TERM:** Document all training targets clearly

---

## Next Steps

1. Fix the bug in `src/build_dataset_q3.py`
2. Rebuild the dataset
3. Retrain models
4. Test with game ID 0022500697
5. Verify predictions are reasonable
6. Push fixes to git
7. Deploy to Streamlit Cloud

---

## Related Files

- `src/build_dataset_q3.py` - **BUG HERE**
- `src/train_q3_model.py` - Trains Q3 models
- `src/calibrate_intervals_q3.py` - Calibrates intervals
- `src/modeling/q3_model.py` - Q3 model interface
- `src/predict_from_gameid_v3_runtime.py` - Uses Q3 predictions
- `data/processed/q3_team_v2.parquet` - Training data (WRONG TARGETS)
- `models_v3/q3/gbt_twohead.joblib` - Model trained on wrong data
- `models_v3/q3/q3_intervals.joblib` - Intervals from wrong predictions

---

**This is a CRITICAL bug that makes the entire Q3 model useless. Fixing it is essential for accurate predictions.** 🐶🚨
