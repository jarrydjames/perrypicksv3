# Fix for KeyError: '_derived' in UI Code
**Date:** 2025-01-31  
**Status:** FIXED ✅  
**Commit:** fcdb4ba

---

## Error

```
KeyError: '_derived'

Location: app.py, line 919

Traceback:
919 │ │ per = pred["_derived"].get(period)
920 │ │ mmss = pred["_derived"].get(clock_mmss)
924 │ │ elif pred["_derived"]["min_remaining"] is not None:
```

---

## Root Cause

**The Problem:**
UI code was accessing `pred["_derived"]` directly without checking if the key exists.

**Why This Happened:**
1. `predict_game()` creates a prediction with standard fields (home_name, away_name, bands80, normal, etc.)
2. **BUT** `predict_game()` does NOT include `_derived` key in its return value
3. `_derived` is only added later by `run_prediction()` (line 711): `pred["_derived"] = {...}`
4. UI code (lines 919-924) tries to access `pred["_derived"]` directly
5. If prediction was just created by `predict_game()` and hasn't gone through `run_prediction()` yet → **KeyError!**

**The Issue:**
```python
# Line 919 - Direct access:
per = pred["_derived"].get("period")  # ← KeyError if _derived doesn't exist!

# Line 920 - Direct access:
mmss = pred["_derived"].get("clock_mmss")  # ← KeyError!

# Line 924 - Direct access:
elif pred["_derived"]["min_remaining"] is not None:  # ← KeyError!
```

---

## Solution

**Use safe `.get()` access with fallback instead of direct `[]` access:**

```python
# Before (lines 919-924):
per = pred["_derived"].get("period")
mmss = pred["_derived"].get("clock_mmss")
if per and mmss:
    st.markdown(...)
elif pred["_derived"]["min_remaining"] is not None:
    st.markdown(...)

# After (lines 918-926):
# Safely access _derived (may not exist if prediction just created)
derived = pred.get("_derived", {})
per = derived.get("period")
mmss = derived.get("clock_mmss")
if per and mmss:
    st.markdown(...)
elif derived.get("min_remaining") is not None:
    st.markdown(...)
```

**The Fix:**
- `derived = pred.get("_derived", {})` - Safely get _derived with empty dict fallback
- Access period, clock_mmss, min_remaining through `derived` variable
- No KeyError when `_derived` doesn't exist yet
- Works for both new predictions (no _derived) and existing predictions (has _derived)

---

## Impact

✅ **No more KeyError** - Safe access pattern handles missing _derived  
✅ **Predictions display correctly** - Works for both new and existing predictions  
✅ **Robust UI** - Handles cases where _derived doesn't exist yet  
✅ **Better error handling** - Graceful fallback instead of crash  

---

## Testing Checklist

- [x] Changed pred["_derived"] to pred.get("_derived", {})
- [x] Updated all _derived accesses in UI code
- [x] App compiles without errors
- [x] Changes committed to git
- [x] Changes pushed to remote

---

## Commit

**Hash:** fcdb4ba  
**Message:** fix: use safe .get() access for _derived to prevent KeyError

---

**Streamlit Cloud will auto-deploy commit `fcdb4ba`. The app should now work!** 🚀