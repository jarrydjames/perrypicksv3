# OddsAPIError Fix - Team Name Key Mismatch
**Date:** 2025-01-31
**Priority:** HIGH
**Status:** COMPLETED ✅

---

## Problem

**Error Message:**
```
OddsAPIError: No odds match found for AWAY @ HOME
Sample available games: San Antonio Spurs @ Charlotte Hornets, Atlanta Hawks @ Indiana Pacers, ...
```

**Symptoms:**
1. App loaded successfully but couldn't fetch odds
2. Error showed placeholder team names ("AWAY @ HOME")
3. Actual team names from NBA data weren't being used for odds lookup

---

## Root Cause Analysis

**What Code Was Doing:**

In src/predict_from_gameid_v3_runtime.py (lines 106-107):
```python
home_tri = result.get("home_tri", "HOME")
away_tri = result.get("away_tri", "AWAY")
```

**What predict_from_gameid_v2_ci Returns:**

```python
out = {
    "game_id": gid,
    "home_name": home_name,  # ← Uses 'home_name', not 'home_tri'
    "away_name": away_name,  # ← Uses 'away_name', not 'away_tri'
    ...
}
```

**The Problem:**
- Code was looking for 'home_tri' and 'away_tri' keys
- But predict_from_gameid_v2_ci returns 'home_name' and 'away_name' keys
- When keys not found, fell back to "HOME" and "AWAY" placeholders
- Placeholders don't match any actual team names in Odds API

**Additional Issue:**
- Q3 model path (eval_at_q3=True) didn't return team names at all
- This meant Q3 predictions would also fail with placeholder names

---

## Solution

**Fix 1: Use Correct Keys for Halftime Path**

Changed from:
```python
home_tri = result.get("home_tri", "HOME")
away_tri = result.get("away_tri", "AWAY")
```

To:
```python
# Note: predict_from_gameid_v2_ci returns "home_name" and "away_name", not "home_tri"/"away_tri"
home_tri = result.get("home_name", "HOME")
away_tri = result.get("away_name", "AWAY")
```

**Fix 2: Add Team Name Extraction for Q3 Path**

Q3 model path now fetches team names from game data:
```python
from src.predict_from_gameid_v2 import fetch_box
from src.predict_from_gameid_v2_ci import _safe_team_name

game = fetch_box(game_id)
home_team = game.get("homeTeam") or {}
away_team = game.get("awayTeam") or {}
home_name = _safe_team_name(home_team, "Home")
away_name = _safe_team_name(away_team, "Away")

result = {
    "game_id": game_id,
    "model_used": "Q3",
    "home_name": home_name,  # ← Added team name
    "away_name": away_name,  # ← Added team name
    ...
}
```

---

## Impact

**Immediate:**
- ✅ Odds fetching uses actual team names from NBA data
- ✅ Team names match Odds API format correctly
- ✅ No more placeholder "AWAY @ HOME" errors
- ✅ Odds successfully fetched and displayed

**Both Models Work:**
- ✅ Halftime model: Uses team names from predict_from_gameid_v2_ci
- ✅ Q3 model: Now extracts team names from game data

---

## Files Modified

**Changed:**
- src/predict_from_gameid_v3_runtime.py

**Changes:**
- Fixed key names: 'home_tri'/'away_tri' → 'home_name'/'away_name'
- Added team name extraction for Q3 model path
- Both prediction paths now return team names

---

## Summary

Issue: OddsAPIError with placeholder team names  
Root Cause: Wrong dictionary keys for team names  
Solution: Use correct keys ('home_name'/'away_name') and add extraction for Q3  
Status: COMPLETED AND PUSHED  
Files Changed: 1 file  
Ready for: Streamlit Cloud deployment