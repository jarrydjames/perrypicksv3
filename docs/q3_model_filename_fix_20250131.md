# Q3 Model Filename Mismatch Fix
**Date:** 2025-01-31
**Priority:** HIGH
**Status:** COMPLETED ✅

---

## Problem

**Error Message:**
```
Prediction failed: FileNotFoundError(2, No such file or directory)
```

**Context:**
- App was switched to use trained Q3 models (eval_at_q3=True)
- Q3 models exist in models_v3/q3/
- But app couldn't load them

---

## Root Cause Analysis

**Q3Model Looking for Wrong Filenames:**

In src/modeling/q3_model.py (lines 95-96):
```python
total_path = self.models_dir / "q3_total_twohead.joblib"  # ❌ Doesn't exist!
margin_path = self.models_dir / "q3_margin_twohead.joblib"  # ❌ Doesn't exist!
```

**Actual Filenames in Repository:**
```
models_v3/q3/gbt_twohead.joblib        ✅ EXISTS (1306 KB)
models_v3/q3/ridge_twohead.joblib      ✅ EXISTS (843 KB)
models_v3/q3/random_forest_twohead.joblib ✅ EXISTS (4086 KB)
models_v3/q3/q3_intervals.joblib      ✅ EXISTS (126 bytes)
```

**The Mismatch:**
| Looking For | Actual Filename | Status |
|-------------|-----------------|--------|
| q3_total_twohead.joblib | gbt_twohead.joblib | ❌ Not found |
| q3_margin_twohead.joblib | ridge_twohead.joblib | ❌ Not found |

---

## Solution

**Fix: Update Filenames in Q3Model.load_models()**

Changed from:
```python
total_path = self.models_dir / "q3_total_twohead.joblib"
margin_path = self.models_dir / "q3_margin_twohead.joblib"
```

To:
```python
total_path = self.models_dir / "gbt_twohead.joblib"  # Fixed filename
margin_path = self.models_dir / "ridge_twohead.joblib"  # Fixed filename
```

**Why These Filenames:**
- **gbt_twohead.joblib**: Gradient Boosted Two-Head model for totals
  - Small artifact, strong overall performance
  - Drives derived team totals
- **ridge_twohead.joblib**: Ridge Two-Head model for margin
  - Calibration-first probabilities for betting decisions
  - Most stable + best calibrated (coverage close to target)

---

## Impact

**Immediate:**
- ✅ Q3 models now load successfully
- ✅ App uses trained Q3 models
- ✅ No more FileNotFoundError
- ✅ All model paths match actual files

**Model Stack Now Active:**
| Component | Model File | Status |
|-----------|-------------|--------|
| **Margin/Spread** | models_v3/q3/ridge_twohead.joblib | ✅ LOADING |
| **Game Total** | models_v3/q3/gbt_twohead.joblib | ✅ LOADING |
| **Intervals** | models_v3/q3/q3_intervals.joblib | ✅ LOADING |

---

## Files Changed

**Modified:**
- `src/modeling/q3_model.py`
  - Line 95: Changed `q3_total_twohead.joblib` → `gbt_twohead.joblib`
  - Line 96: Changed `q3_margin_twohead.joblib` → `ridge_twohead.joblib`

---

## Summary

Issue: FileNotFoundError loading Q3 models  
Root Cause: Q3Model looking for wrong model filenames  
Solution: Updated filenames to match actual trained models  
Status: COMPLETED AND PUSHED  
Files Changed: 1 file (2 lines)  
Models Now Loading: gbt_twohead + ridge_twohead  
Ready for: Streamlit Cloud deployment