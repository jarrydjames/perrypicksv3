# Bug: fetch_nba_odds_snapshot() Called with Positional Arguments - FIXED ✅
**Status:** ✅ FIXED
**Date:** February 7, 2026
**Severity:** 🔴 CRITICAL - Predictions failing

---

## 🐛 The Problem

User reported error:
```
0022500747: Prediction failed: fetch_nba_odds_snapshot() takes 0 positional arguments but 2 were given
```

This error occurred when trying to generate predictions. The prediction was failing immediately when trying to fetch odds.

---

## 🔍 Root Cause Analysis

### The Function Signature

In `src/odds/odds_api.py` line 83:

```python
def fetch_nba_odds_snapshot(
    *,  # ← Keyword-only marker!
    home_name: str,
    away_name: str,
    regions: str = "us",
    markets: str = "h2h,spreads,totals,team_totals",
    odds_format: str = "american",
    date_format: str = "iso",
    preferred_book: Optional[str] = None,
    timeout_s: int = 10,
) -> OddsAPIMarketSnapshot:
```

The `*,` makes **all parameters keyword-only**. This means you must call it like:
```python
fetch_nba_odds_snapshot(home_name="Lakers", away_name="Celtics")
```

**NOT:**
```python
fetch_nba_odds_snapshot("Lakers", "Celtics")  # ← WRONG! Positional arguments
```

---

### The Bad Calls

There were **2 places** where the function was called incorrectly:

#### 1. src/predict_pregame.py (line 686)

**Before:**
```python
odds_snapshot = fetch_nba_odds_snapshot(home_team, away_team)
#                                                ^^^^^^^^^  ^^^^^^^^ Positional!
```

**After:**
```python
odds_snapshot = fetch_nba_odds_snapshot(
    home_name=home_team,
    away_name=away_team
)
```

#### 2. src/odds/persistent_cache.py (line 142)

**Before:**
```python
snapshot = fetch_nba_odds_snapshot(home, away)
#                                    ^^^^  ^^^^ Positional!
```

**After:**
```python
snapshot = fetch_nba_odds_snapshot(
    home_name=home,
    away_name=away
)
```

---

### The Correct Call (Unchanged)

`src/odds/streamlit_cache.py` already had it correct:

```python
return fetch_nba_odds_snapshot(
    home_name=home,
    away_name=away,
    preferred_book=book,
    markets=_markets(want_tt),
)
```

This is why some parts of the app worked while others failed.

---

## ✅ The Fixes

### Fix #1: src/predict_pregame.py

**Before:**
```python
odds_snapshot = fetch_nba_odds_snapshot(home_team, away_team)
```

**After:**
```python
odds_snapshot = fetch_nba_odds_snapshot(
    home_name=home_team,
    away_name=away_team
)
```

### Fix #2: src/odds/persistent_cache.py

**Before:**
```python
snapshot = fetch_nba_odds_snapshot(home, away)
```

**After:**
```python
snapshot = fetch_nba_odds_snapshot(
    home_name=home,
    away_name=away
)
```

---

## 📊 Before vs After

| Aspect | Before | After |
|---------|--------|-------|
| **Call type** | Positional arguments | Keyword arguments |
| **Predictions work?** | ❌ No (TypeError) | ✅ Yes! |
| **Odds fetched?** | ❌ No | ✅ Yes! |
| **Error message** | "takes 0 positional arguments but 2 were given" | No error! |

---

## 🎯 What User Should See Now

**After this fix:**

When generating predictions:
```
✅ Prediction generated for game 0022500747!

Predicted: Celtics 105 - Lakers 98
Winner: Celtics by 7
Total: 203
```

No more TypeErrors! Predictions should work correctly.

---

## ✅ Summary

**Root Cause:**
- ❌ `fetch_nba_odds_snapshot()` uses `*,` to enforce keyword-only parameters
- ❌ Two places were calling it with positional arguments
- ❌ This caused TypeError when generating predictions

**Fixed:**
- ✅ Changed both bad calls to use keyword arguments
- ✅ All calls now consistent: `fetch_nba_odds_snapshot(home_name=X, away_name=Y)`
- ✅ Predictions now work without errors

**Files Modified:**
- `src/predict_pregame.py`
- `src/odds/persistent_cache.py`

**Commit:**
- `24f0fb1` - Fix: fetch_nba_odds_snapshot() being called with positional arguments

---
**Author:** Perry (code-puppy)
**Date:** February 7, 2026
**Status:** ✅ FIXED - Predictions now work!

🐶 *Always use keyword arguments when the function signature has *, !* 🚀