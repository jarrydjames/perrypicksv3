# Bug: OddsAPIMarketSnapshot Object Has No Attribute home_moneyline - FIXED ✅
**Status:** ✅ FIXED
**Date:** February 7, 2026
**Severity:** 🔴 CRITICAL - Predictions failing

---

## 🐛 The Problem

User reported error:
```
0022500752: Prediction failed: OddsAPIMarketSnapshot object has no attribute 'home_moneyline'
```

Predictions were failing when trying to access odds data from the snapshot object.

---

## 🔍 Root Cause Analysis

### The OddsAPIMarketSnapshot Class

The `OddsAPIMarketSnapshot` class (in `src/odds/odds_api.py`) has these attributes:

```python
class OddsAPIMarketSnapshot:
    # Main markets (full game)
    total_points: Optional[float]          # ← total_points, not total_line
    total_over_odds: Optional[int]
    total_under_odds: Optional[int]

    spread_home: Optional[float]           # ← spread_home, not spread_home_line
    spread_home_odds: Optional[int]
    spread_away_odds: Optional[int]

    moneyline_home: Optional[int]         # ← moneyline_home, not home_moneyline
    moneyline_away: Optional[int]         # ← moneyline_away, not away_moneyline

    # Team totals (if supported)
    team_total_home: Optional[float]
    team_total_home_over_odds: Optional[int]
    team_total_home_under_odds: Optional[int]

    team_total_away: Optional[float]
    team_total_away_over_odds: Optional[int]
    team_total_away_under_odds: Optional[int]

    bookmaker: Optional[str] = None
    last_update: Optional[str] = None
```

---

### The Wrong Code

There were **2 files** using incorrect attribute names:

#### 1. src/predict_pregame.py (lines 693-701)

**Before (WRONG):**
```python
result.update({
    "odds_home_ml": odds_snapshot.home_moneyline,      # ← WRONG!
    "odds_away_ml": odds_snapshot.away_moneyline,      # ← WRONG!
    "odds_total_line": odds_snapshot.total_line,         # ← WRONG!
    "odds_total_over": odds_snapshot.total_over_odds,
    "odds_total_under": odds_snapshot.total_under_odds,
    "odds_spread_home_line": odds_snapshot.spread_home_line,  # ← WRONG!
    "odds_spread_home": odds_snapshot.spread_home_odds,
    "odds_spread_away": odds_snapshot.spread_away_odds,
})
```

**After (CORRECT):**
```python
result.update({
    "odds_home_ml": odds_snapshot.moneyline_home,       # ← CORRECT!
    "odds_away_ml": odds_snapshot.moneyline_away,       # ← CORRECT!
    "odds_total_line": odds_snapshot.total_points,        # ← CORRECT!
    "odds_total_over": odds_snapshot.total_over_odds,
    "odds_total_under": odds_snapshot.total_under_odds,
    "odds_spread_home_line": odds_snapshot.spread_home,  # ← CORRECT!
    "odds_spread_home": odds_snapshot.spread_home_odds,
    "odds_spread_away": odds_snapshot.spread_away_odds,
})
```

#### 2. src/predict_from_gameid_v3_runtime.py (lines 306-314)

Same issue, same fix applied.

---

## ✅ The Fixes

### Attribute Name Mappings

| Wrong Name | Correct Name |
|------------|--------------|
| `home_moneyline` | `moneyline_home` |
| `away_moneyline` | `moneyline_away` |
| `total_line` | `total_points` |
| `spread_home_line` | `spread_home` |

### Files Fixed
1. `src/predict_pregame.py` (lines 693-701)
2. `src/predict_from_gameid_v3_runtime.py` (lines 306-314)

---

## 📊 Before vs After

| Aspect | Before | After |
|---------|--------|-------|
| **home_moneyline used?** | ❌ Yes (wrong) | ✅ No (fixed to moneyline_home) |
| **away_moneyline used?** | ❌ Yes (wrong) | ✅ No (fixed to moneyline_away) |
| **total_line used?** | ❌ Yes (wrong) | ✅ No (fixed to total_points) |
| **spread_home_line used?** | ❌ Yes (wrong) | ✅ No (fixed to spread_home) |
| **Predictions work?** | ❌ No (AttributeError) | ✅ Yes! |

---

## 🎯 What User Should See Now

**After this fix:**

Predictions should work without errors:
```
✅ Prediction generated for game 0022500752!

Predicted: Celtics 105 - Lakers 98
Winner: Celtics by 7
Total: 203

Odds:
- Moneyline: Home -150, Away +130
- Total: 203.5 (Over -110, Under -110)
- Spread: Home -2.5 (Home -110, Away -110)
```

No more AttributeErrors!

---

## ✅ Summary

**Root Cause:**
- ❌ Code was using wrong attribute names to access `OddsAPIMarketSnapshot` data
- ❌ Attribute names didn't match the actual class definition
- ❌ This caused AttributeError when generating predictions

**Fixed:**
- ✅ Corrected all 4 wrong attribute names to match class definition
- ✅ Fixed in 2 files where the issue occurred
- ✅ Predictions now work without AttributeError

**Files Modified:**
- `src/predict_pregame.py`
- `src/predict_from_gameid_v3_runtime.py`

**Commit:**
- `34bd0c6` - Fix: OddsAPIMarketSnapshot attribute names were wrong

---
**Author:** Perry (code-puppy)
**Date:** February 7, 2026
**Status:** ✅ FIXED - Predictions now work with correct odds attributes!

🐶 *Always check the class definition for correct attribute names!* 🚀
