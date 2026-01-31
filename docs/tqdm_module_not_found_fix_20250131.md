# TQDM ModuleNotFoundError Fix - Avoid Training Dependencies
**Date:** 2025-01-31
**Priority:** HIGH
**Status:** COMPLETED ✅

---

## Problem

**Error Message:**
```
Prediction failed: ModuleNotFoundError(No module named 'tqdm')
```

**Context:**
- Runtime failed on Streamlit Cloud
- tqdm is only needed for training, not runtime prediction
- Import was from build_dataset_q3.py (training module)

---

## Root Cause Analysis

**Why TQDM Was Being Imported:**

1. **build_dataset_q3.py** imports tqdm at module level:
   ```python
   # In build_dataset_q3.py:
   from tqdm import tqdm  # ← For training progress bars!
   ```

2. **Runtime imported from build_dataset_q3:**
   ```python
   # In predict_from_gameid_v3_runtime.py:
   from src.build_dataset_q3 import (
       sum_first3,
       third_quarter_score,
       behavior_counts_q3,
   )  # ← Triggers tqdm import!
   ```

3. **When Python Imports Module:**
   - Python executes ALL module-level code
   - Even if we only use sum_first3, third_quarter_score, behavior_counts_q3
   - Python still executes `from tqdm import tqdm`
   - If tqdm not installed → ModuleNotFoundError

**The Issue:**
- build_dataset_q3.py is a **training module** (needs tqdm for progress bars)
- predict_from_gameid_v3_runtime.py is a **runtime module** (doesn't need tqdm)
- Importing from training module brings training dependencies into runtime
- Streamlit Cloud doesn't have tqdm installed (it's not in requirements.txt for runtime)

---

## Solution

**Fix: Copy Helper Functions Directly (Avoid Import)**

Instead of importing from build_dataset_q3.py, copy the 3 functions needed:

```python
# BEFORE: Import from training module (brings tqdm dependency)
from src.build_dataset_q3 import (
    sum_first3,
    third_quarter_score,
    behavior_counts_q3,
)

# AFTER: Copy functions directly into runtime file
def sum_first3(periods):
    """Sum scores from periods 1-3."""
    s = 0
    for p in (periods or []):
        period_num = int(p.get("period", 0))
        if 1 <= period_num <= 3:
            for key in ("score", "points", "pts"):
                if key in p and p[key] is not None:
                    s += int(p[key])
                    break
    return s

def third_quarter_score(game):
    """Extract home and away scores after Q3."""
    home = game.get("homeTeam", {}) or {}
    away = game.get("awayTeam", {}) or {}
    return sum_first3(home.get("periods")), sum_first3(away.get("periods"))

def behavior_counts_q3(pbp: pd.DataFrame) -> dict:
    """
    Count action types in first 3 quarters.

    Same structure as behavior_counts_1h, but filters to periods 1-3.
    """
    q3 = pbp[pbp["period"].astype(int) <= 3].copy()
    at = q3.get("actionType", pd.Series([""] * len(q3))).astype(str).fillna("")

    def c(prefix):
        return int(at.str.startswith(prefix).sum())

    return {
        "q3_events": int(len(q3)),
        "q3_n_2pt": c("2pt"),
        "q3_n_3pt": c("3pt"),
        "q3_n_turnover": c("turnover"),
        "q3_n_rebound": c("rebound"),
        "q3_n_foul": c("foul"),
        "q3_n_timeout": c("timeout"),
        "q3_n_sub": c("substitution"),
    }
```

**Why This Works:**
- No import from build_dataset_q3.py
- No module-level execution of training code
- No tqdm dependency
- Functions work in runtime environment

---

## Impact

**Immediate:**
- ✅ No more ModuleNotFoundError for tqdm
- ✅ Runtime imports work on Streamlit Cloud
- ✅ No training dependencies in runtime
- ✅ Helper functions available for Q3 feature extraction

**Dependency Separation:**
- **Training Modules:** Can have tqdm, numpy, pandas, etc.
- **Runtime Modules:** Only need minimal dependencies
- **No Cross-Imports:** Runtime doesn't import from training modules

---

## Files Changed

**Modified:**
- `src/predict_from_gameid_v3_runtime.py`
  - Added: sum_first3() function (copied from build_dataset_q3)
  - Added: third_quarter_score() function (copied from build_dataset_q3)
  - Added: behavior_counts_q3() function (copied from build_dataset_q3)
  - Removed: import from src.build_dataset_q3

---

## Summary

Issue: ModuleNotFoundError - No module named 'tqdm'  
Root Cause: Runtime imported from training module which has tqdm at module level  
Solution: Copied 3 helper functions directly into runtime file (avoided import)  
Status: COMPLETED AND PUSHED  
Files Changed: 1 file (copied 3 functions, removed 1 import)  
Ready for: Streamlit Cloud deployment