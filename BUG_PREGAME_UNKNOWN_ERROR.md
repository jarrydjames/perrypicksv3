# Bug: Pregame Prediction Returns 'Unknown Error' - INVESTIGATION IN PROGRESS 🔍
**Status:** 🔍 INVESTIGATING - Added logging to diagnose
**Date:** February 7, 2026
**Severity:** 🔴 CRITICAL - Pregame predictions failing with unhelpful error
**Game ID:** 0022500755
**User Context:** Pulling pregame prediction got 'Unknown error'

---

## 🐛 The Problem

User reported:
```
pulling pregame 0022500755: Unknown error
```

**The Issue:**
- User is trying to generate a pregame prediction
- System returns "Unknown error" instead of specific error
- Error message provides no debugging information
- Can't determine what's causing the prediction to fail

---

## 🔍 Root Cause Analysis

### Where Does 'Unknown Error' Come From?

In `src/automation/automation_orchestrator.py`:

```python
else:
    error_msg = prediction.get("error", "Unknown error")
    results["errors"].append({
        "game_id": game_id,
        "error": error_msg,
    })
```

**The Problem:**
- "Unknown error" is a fallback message
- Used when `prediction.get("error")` returns `None`
- Means the prediction dict exists but doesn't have an 'error' field
- AND `prediction.get("status") != "success"`

### Why Would This Happen?

**Scenario 1: Result dict has 'error' field but no 'status' field**
- `predict_pregame()` returns: `{'error': 'Some error', 'game_id': '...'}`
- Missing 'status' field
- Orchestrator checks: `prediction.get("status") != "success"` → True (status is None)
- Orchestrator gets error: `prediction.get("error")` → 'Some error' (would work)
- **This scenario should NOT produce 'Unknown error'**

**Scenario 2: Result dict has neither 'status' nor 'error' fields**
- `predict_pregame()` returns: `{'game_id': '...', 'margin': None, 'total': None}`
- Missing both 'status' and 'error' fields
- Orchestrator checks: `prediction.get("status") != "success"` → True (status is None)
- Orchestrator gets error: `prediction.get("error")` → None (missing!)
- **This scenario WOULD produce 'Unknown error'**

**Scenario 3: Result is None or not a dict**
- `predict_pregame()` returns: `None`
- Orchestrator checks: `prediction.get("status")` → AttributeError (would crash)
- **This scenario would crash, not produce 'Unknown error'**

### Most Likely Cause: Scenario 2

The pregame prediction is returning a dict with game_id and other fields, but **missing both 'status' and 'error' fields**.

**Why would predict_pregame() do this?**

Looking at the code, predict_pregame() should always return a dict with:\- `status`: 'success' or 'error'
- `error`: (only when status='error')

But there might be a code path that returns a partial dict without these fields.

### Possible Root Causes:

1. **Incomplete result from model.predict()**
   - Model prediction returns incomplete data
   - Result dict is built but 'status' field not set

2. **Exception caught and not handled properly**
   - Try/except block catches exception
   - Returns partial result

3. **Result dict created but fields not set**
   - Code creates result dict
   - Doesn't set all required fields before returning

4. **Model loading fails silently**
   - Pregame model fails to load
   - Returns incomplete result instead of error

5. **Data freshness check returns None**
   - Import gate check returns None (not error dict)
   - Proceeds with incomplete result

---

## ✅ Fixes Applied (Investigation Phase)

### 1. Enhanced Logging in predict_api.py

**Added detailed logging for pregame predictions:**

```python
# Log before calling predict_pregame
logger.info(f"Calling predict_pregame for {game_input} with home_team={home_team}, away_team={away_team}")

result = predict_pregame(...)

# Log result details
logger.info(f"Pregame result for {game_input}:")
logger.info(f"  Type: {type(result)}")
logger.info(f"  Keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
logger.info(f"  Status: {result.get('status') if isinstance(result, dict) else 'N/A'}")
```

**Added edge case handling:**

```python
# Check for result with 'error' field but no 'status' field
if 'error' in result and 'status' not in result:
    result['status'] = 'error'
    logger.error(f"Pregame prediction had error field but no status field")

# Ensure all required fields exist
result.setdefault('status', 'error')
result.setdefault('game_id', game_input)
result.setdefault('home_name', home_team or 'Home')
result.setdefault('away_name', away_team or 'Away')
result.setdefault('margin', 0)
result.setdefault('total', 0)
if 'error' not in result:
    result['error'] = 'Pregame prediction returned incomplete result'
```

### 2. Enhanced Logging in automation_orchestrator.py

**Added detailed logging for all predictions:**

```python
logger.info(f"Running prediction for {game_id} with mode={mode}, trigger_type={trigger_type}")
prediction = predict_game(game_id, mode=mode, fetch_odds=fetch_odds)

# Log detailed prediction result
logger.info(f"Prediction result for {game_id}:")
logger.info(f"  Type: {type(prediction)}")
if isinstance(prediction, dict):
    logger.info(f"  Keys: {list(prediction.keys())}")
    logger.info(f"  Status: {prediction.get('status', 'missing')}")
    logger.info(f"  Model used: {prediction.get('model_used', 'missing')}")
    logger.info(f"  Error: {prediction.get('error', 'none')}")
else:
    logger.warning(f"Prediction is not a dict: {prediction}")
```

**Better error handling:**

```python
else:
    error_msg = prediction.get("error", "Unknown error") if isinstance(prediction, dict) else f"Invalid prediction type: {type(prediction)}"
    logger.error(f"Prediction failed for {game_id}: {error_msg}")
    logger.error(f"Prediction details: {prediction}")
```

---

## 📊 Expected Log Output

After deploying, the logs will show:

**Before calling predict_pregame:**
```
INFO - Calling predict_pregame for 0022500755 with home_team=XXX, away_team=YYY
```

**After predict_pregame returns:**
```
INFO - Pregame result for 0022500755:
INFO -   Type: <class 'dict'>
INFO -   Keys: ['game_id', 'home_name', 'away_name', 'margin', 'total', ...]
INFO -   Status: error  (or 'success' or 'missing')
```

**If status='error':**
```
ERROR - Pregame prediction error for 0022500755: [specific error message]
```

**If status is missing or unexpected:**
```
ERROR - Pregame prediction for 0022500755 has unexpected structure
ERROR - Result type: <class 'dict'>
ERROR - Result value: {'game_id': '0022500755', 'margin': None, ...}
```

---

## 📊 Impact

### Before Fix
| Issue | Impact |
|--------|--------|
| **Unhelpful error** | 'Unknown error' provides no debugging info |
| **No logging** | Can't see what predict_pregame returned |
| **Edge cases** | Missing handling for incomplete results |
| **Status missing** | Result might have 'error' but no 'status' |

### After Fix
| Improvement | Benefit |
|-------------|---------|
| **Detailed logging** | See exactly what predict_pregame returns |
| **Result type logged** | Know if it's dict, None, or something else |
| **Keys logged** | See which fields are present/missing |
| **Edge case handling** | Add 'status': 'error' if missing |
| **Better errors** | Specific error instead of 'Unknown error' |

---

## 🧪 Next Steps

1. **Deploy changes to Streamlit Cloud**
   - Wait 5 minutes for auto-deployment
   - Refresh app

2. **User reproduces the error**
   - Try pulling pregame prediction for game 0022500755 again
   - Capture the full error message

3. **Check Streamlit Cloud logs**
   - Go to Streamlit Cloud → App → Logs
   - Look for detailed log output showing:
     - Pregame result type
     - Pregame result keys
     - Pregame result status
     - Specific error message if available

4. **Report findings**
   - Share the log output
   - Will determine exact root cause from logs
   - Apply targeted fix based on findings

---

## 🎯 Likely Scenarios Based on Logs

**If logs show:** `Status: missing` or `Keys: ['game_id', 'margin', 'total']` (no 'status' or 'error')
→ **Root cause:** predict_pregame() returns incomplete dict
→ **Fix:** Add 'status': 'error' in predict_pregame() for all error paths

**If logs show:** `Status: error, Error: STALE_DATA: import watermark is Xh old`
→ **Root cause:** Data is stale (import watermark too old)
→ **Fix:** Run import job to refresh data or bypass gate

**If logs show:** `Status: error, Error: Unknown home team tricode: XXX`
→ **Root cause:** Invalid team tricode
→ **Fix:** Check game schedule data for correct team names

**If logs show:** `Status: error, Error: Pregame model not available`
→ **Root cause:** Pregame model not loaded
→ **Fix:** Train/load pregame model

**If logs show:** `Type: <class 'NoneType'>`
→ **Root cause:** predict_pregame() returning None
→ **Fix:** Add check for None and return error dict

---
**Author:** Perry (code-puppy)
**Date:** February 7, 2026
**Status:** 🔍 INVESTIGATING - Added detailed logging to diagnose root cause

🐶 *Added logging to catch the culprit! Check logs after trying again!* 🚀