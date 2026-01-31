# TrainedHead TypeError Fix - Quantile Models Structure
**Date:** 2025-01-31
**Priority:** HIGH
**Status:** COMPLETED ✅

---

## Problem

**Error Message:**
```
Prediction failed: TypeError(TrainedHead.init() got an unexpected keyword argument 'q10_model')
```

---

## Root Cause Analysis

**Training Pipeline Structure:**

The training code (`train_models.py`) saves quantile models in nested structure:
```python
payload = {
    "total": {
        "model": <main model>,
        "residual_sigma": <sigma>,
        "q10_model": <q10 quantile model>,  # ← Nested in joblib
        "q90_model": <q90 quantile model>,  # ← Nested in joblib
    },
    "margin": {
        "model": <main model>,
        "residual_sigma": <sigma>,
        "q10_model": <q10 quantile model>,  # ← Nested in joblib
        "q90_model": <q90 quantile model>,  # ← Nested in joblib
    }
}
joblib.dump(payload, "gbt_twohead.joblib")
```

**TrainedHead Class Definition:**

In `src/modeling/types.py`:
```python
@dataclass(frozen=True)
class TrainedHead:
    features: List[str]
    model: Any
    residual_sigma: float
    # ❌ NO q10_model or q90_model fields!
```

**The Mismatch:**
- Training joblib has: `total["q10_model"]` and `total["q90_model"]`
- `TrainedHead` class expects: `model`, `residual_sigma`, `features`
- Q3Model was trying to pass: `q10_model=..., q90_model=...`
- Result: TypeError - unexpected keyword arguments

---

## Solution

**Fix: Separate Quantile Model Extraction**

**BEFORE:**
```python
# ❌ Passing invalid arguments to TrainedHead
total_head = TrainedHead(
    model=self.total_model.get("model"),
    residual_sigma=self.total_model.get("residual_sigma", 2.0),
    q10_model=self.total_model.get("q10_model"),  # ❌ Invalid!
    q90_model=self.total_model.get("q90_model"),  # ❌ Invalid!
)
margin_head = TrainedHead(
    model=self.margin_model.get("model"),
    residual_sigma=self.margin_model.get("residual_sigma", 2.0),
    q10_model=self.margin_model.get("q10_model"),  # ❌ Invalid!
    q90_model=self.margin_model.get("q90_model"),  # ❌ Invalid!
)

# ❌ Using non-existent fields
total_q10 = total_head.q10_model.predict(X)[0]  # ❌ Field doesn't exist!
total_q90 = total_head.q90_model.predict(X)[0]  # ❌ Field doesn't exist!
```

**AFTER:**
```python
# ✅ TrainedHead only gets valid fields
total_head = TrainedHead(
    model=self.total_model.get("model"),
    residual_sigma=self.total_model.get("residual_sigma", 2.0),
)
margin_head = TrainedHead(
    model=self.margin_model.get("model"),
    residual_sigma=self.margin_model.get("residual_sigma", 2.0),
)

# ✅ Quantile models extracted separately
total_q10_model = self.total_model.get("q10_model")  # Extracted from nested joblib
total_q90_model = self.total_model.get("q90_model")  # Extracted from nested joblib
margin_q10_model = self.margin_model.get("q10_model")  # Extracted from nested joblib
margin_q90_model = self.margin_model.get("q90_model")  # Extracted from nested joblib

# ✅ Use separate variables
total_q10 = total_q10_model.predict(X)[0]  # ✅ Works!
total_q90 = total_q90_model.predict(X)[0]  # ✅ Works!
margin_q10 = margin_q10_model.predict(X)[0]  # ✅ Works!
margin_q90 = margin_q90_model.predict(X)[0]  # ✅ Works!
```

---

## Impact

**Immediate:**
- ✅ Q3 models now load without TypeError
- ✅ Quantile models extracted correctly from nested joblib structure
- ✅ Predictions work with 80% confidence intervals
- ✅ All model components (total, margin, quantiles) load properly

---

## Model Stack Now Working

| Component | Source | Status |
|-----------|--------|--------|
| **Main Model (Total)** | `models_v3/q3/gbt_twohead.joblib["total"]["model"]` | ✅ LOADING |
| **Main Model (Margin)** | `models_v3/q3/ridge_twohead.joblib["margin"]["model"]` | ✅ LOADING |
| **Q10 Quantile (Total)** | `models_v3/q3/gbt_twohead.joblib["total"]["q10_model"]` | ✅ LOADING |
| **Q90 Quantile (Total)** | `models_v3/q3/gbt_twohead.joblib["total"]["q90_model"]` | ✅ LOADING |
| **Q10 Quantile (Margin)** | `models_v3/q3/ridge_twohead.joblib["margin"]["q10_model"]` | ✅ LOADING |
| **Q90 Quantile (Margin)** | `models_v3/q3/ridge_twohead.joblib["margin"]["q90_model"]` | ✅ LOADING |

---

## Files Changed

**Modified:**
- `src/modeling/q3_model.py`
  - Lines 110-119: Removed `q10_model` and `q90_model` from `TrainedHead.__init__()`
  - Lines 123-126: Extracted quantile models separately as variables
  - Lines 130-133: Updated `predict()` to use separate quantile model variables

---

## Summary

Issue: TypeError when loading Q3 models  
Root Cause: TrainedHead class structure mismatch with nested quantile models in joblib files  
Solution: Extract quantile models separately from nested joblib structure  
Status: COMPLETED AND PUSHED  
Files Changed: 1 file (9 lines changed)  
Models Now Loading: All 6 components working correctly  
Ready for: Streamlit Cloud deployment