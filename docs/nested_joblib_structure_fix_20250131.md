# Nested Joblib Structure Fix - Model Extraction
**Date:** 2025-01-31
**Priority:** HIGH
**Status:** COMPLETED ✅

---

## Problem

**Error Message:**
```
Prediction failed: AttributeError('NoneType' object has no attribute 'predict')
```

**Context:**
- Error persisted even after adding None checks for quantile models
- Main models were still returning None
- Models loaded from joblib but .predict() failed

---

## Root Cause Analysis

**Joblib Model Structure (Hidden Complexity):**

The trained model files have a **nested structure** that wasn't documented:

```python
# gbt_twohead.joblib and ridge_twohead.joblib structure:
{
  "model_name": "gbt" or "ridge",
  "model_version": "1",
  "feature_version": "v3_q3",
  "features": [...],
  "joint": {"model_total": ..., "model_margin": ..., "residual_cov": [...]},
  "total": {
    "model": <HistGradientBoostingRegressor or Ridge>,
    "residual_sigma": <float>,
    "q10_model": <Pipeline>,
    "q90_model": <Pipeline>
  },
  "margin": {
    "model": <HistGradientBoostingRegressor or Ridge>,
    "residual_sigma": <float>,
    "q10_model": <Pipeline>,
    "q90_model": <Pipeline>
  }
}
```

**The Mismatch:**

**Wrong Access (returns None):**
```python
total_head.model = self.total_model.get("model")  # ❌ None!
margin_head.model = self.margin_model.get("model")  # ❌ None!
```

**Correct Access (nested):**
```python
total_head.model = self.total_model["total"]["model"]  # ✅ Works!
margin_head.model = self.margin_model["margin"]["model"]  # ✅ Works!
```

**Why It Failed:**
1. `self.total_model` loads `gbt_twohead.joblib` as a dict
2. `.get("model")` looks for top-level key "model" → **None** (doesn't exist at top level!)
3. The actual model is nested under `["total"]["model"]`
4. Same for `self.margin_model` (ridge_twohead.joblib) - model is nested under `["margin"]["model"]`

---

## Solution

**Fix: Extract Models from Nested Structure**

**Changed From (Wrong):**
```python
total_head = TrainedHead(
    features=list(feature_names),
    model=self.total_model.get("model"),  # ❌ Returns None!
    residual_sigma=self.total_model.get("residual_sigma", 2.0),  # ❌ Wrong!
)
margin_head = TrainedHead(
    features=list(feature_names),
    model=self.margin_model.get("model"),  # ❌ Returns None!
    residual_sigma=self.margin_model.get("residual_sigma", 2.0),  # ❌ Wrong!
)

total_q10_model = self.total_model.get("q10_model")  # ❌ None!
```

**Changed To (Correct):**
```python
# Use gbt_twohead.joblib["total"]["model"] for total predictions
# And ridge_twohead.joblib["margin"]["model"] for margin predictions
total_head = TrainedHead(
    features=list(feature_names),
    model=self.total_model.get("total", {}).get("model"),  # ✅ Nested!
    residual_sigma=self.total_model.get("total", {}).get("residual_sigma", 2.0),  # ✅ Nested!
)
margin_head = TrainedHead(
    features=list(feature_names),
    model=self.margin_model.get("margin", {}).get("model"),  # ✅ Nested!
    residual_sigma=self.margin_model.get("margin", {}).get("residual_sigma", 2.0),  # ✅ Nested!
)

# Extract quantile models from nested structure
total_q10_model = self.total_model.get("total", {}).get("q10_model")  # ✅ Nested!
total_q90_model = self.total_model.get("total", {}).get("q90_model")  # ✅ Nested!
margin_q10_model = self.margin_model.get("margin", {}).get("q10_model")  # ✅ Nested!
margin_q90_model = self.margin_model.get("margin", {}).get("q90_model")  # ✅ Nested!
```

**Safe Access Pattern:**
- Use `.get("key", {}).get("nested_key")` for safe nested access
- Returns None if any intermediate key is missing
- Prevents KeyError exceptions

---

## Impact

**Immediate:**
- ✅ Main models (GBT + Ridge) extract correctly from joblib files
- ✅ Quantile models extract correctly from nested structure
- ✅ All model components load successfully
- ✅ No more NoneType AttributeError

**Model Stack Now Active:**
| Component | Source | Status |
|-----------|--------|--------|
| **Total Model** | `models_v3/q3/gbt_twohead.joblib["total"]["model"]` | ✅ LOADING |
| **Total Sigma** | `models_v3/q3/gbt_twohead.joblib["total"]["residual_sigma"]` | ✅ LOADING |
| **Total Q10** | `models_v3/q3/gbt_twohead.joblib["total"]["q10_model"]` | ✅ LOADING |
| **Total Q90** | `models_v3/q3/gbt_twohead.joblib["total"]["q90_model"]` | ✅ LOADING |
| **Margin Model** | `models_v3/q3/ridge_twohead.joblib["margin"]["model"]` | ✅ LOADING |
| **Margin Sigma** | `models_v3/q3/ridge_twohead.joblib["margin"]["residual_sigma"]` | ✅ LOADING |
| **Margin Q10** | `models_v3/q3/ridge_twohead.joblib["margin"]["q10_model"]` | ✅ LOADING |
| **Margin Q90** | `models_v3/q3/ridge_twohead.joblib["margin"]["q90_model"]` | ✅ LOADING |

---

## Files Changed

**Modified:**
- `src/modeling/q3_model.py`
  - Lines 110-127: Fixed nested structure access for main models
  - Lines 131-134: Fixed nested structure access for quantile models
  - Line 195: Fixed feature_version extraction

---

## Summary

Issue: NoneType AttributeError when calling .predict() on models  
Root Cause: Joblib files have nested structure {total: {model: ...}, margin: {model: ...}} - code was accessing wrong level  
Solution: Updated all model extraction to use nested structure with safe .get() pattern  
Status: COMPLETED AND PUSHED  
Files Changed: 1 file (12 lines)  
All Model Components: Loading correctly from nested structure  
Ready for: Streamlit Cloud deployment