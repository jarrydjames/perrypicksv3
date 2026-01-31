# AttributeError Fix - Stub Model Structure
**Date:** 2025-01-31
**Priority:** HIGH
**Status:** COMPLETED ✅

---

## Problem

**Error Message:**
```
AttributeError: Can't get attribute 'predict_total' on <module 'main' from '/mount/src/perrypicksv3/app.py'
```

**Symptoms:**
1. App crashed immediately on load
2. User could not see app interface
3. AttributeError during model unpickling

---

## Root Cause Analysis

**What Code Expects:**

From src/predict_from_gameid.py:98:
```
features_total, m_total = load_model("models/team_2h_total.joblib")
```

The load_model() function expects a dict with 'features' and 'model' keys.

**What Was Wrong with Stub:**

First stub attempt created structure that broke unpickling. joblib.dump() doesn't understand custom 'predict' key. During unpickling, joblib tried to find 'predict_total' attribute and failed.

---

## Solution

Fix: Use Standard Dict Structure

Correct joblib.dump() format:
```
model_total = {"features": features, "model": m_total}
joblib.dump(model_total, "models/team_2h_total.joblib")
```

---

## Files Modified

Recreated:
- models/team_2h_total.joblib (recreated with correct structure)
- models/team_2h_margin.joblib (recreated with correct structure)

---

## Impact

Immediate:
- App loads without AttributeError
- Model unpickles successfully
- Predictions can be made (though conservative)
- User sees app interface

Prediction Behavior:
- Very conservative: No change from 1H to 2H
- Total prediction: 2x halftime total
- Margin prediction: 0 (no advantage to either team)
- These are NOT accurate predictions, just placeholders

---

## Summary

Issue: AttributeError during model unpickling  
Root Cause: Stub model used custom 'predict' key that joblib couldn't handle  
Solution: Recreated stub models with standard dict structure  
Status: COMPLETED AND PUSHED  
Files Changed: 2 stub model files (recreated)  
Ready for: Streamlit Cloud deployment