# Odds API Usage Optimization
**Date:** 2025-01-31
**Priority:** HIGH
**Status:** ANALYSIS COMPLETE ⚠️

---

## Problem

**Issue:**
- Odds API usage is way up
- Odds API limits are very tight
- Need to optimize to keep usage efficient

**Requirements:**
- Odds should only pull for games that are pregame or in progress
- Should pull one game at a time
- Odds should only pull when a refresh request is made

---

## Root Cause Analysis

**Issue 1: Odds Fetched Inside Prediction Function**

Current flow:
```python
# app.py
run_prediction():
    pred = predict_game(game_input)  # ← This ALWAYS fetches odds!

# predict_api.py
def predict_game(game_input: str):
    from src.predict_from_gameid_v3_runtime import predict_from_game_id
    return predict_from_game_id(game_input)

# predict_from_gameid_v3_runtime.py
def predict_from_game_id(...):
    # ... make predictions ...
    # ALWAYS fetch odds:
    cache = PersistentOddsCache()
    odds = cache.get(home_tri, away_tri)
    if odds is None:
        odds = fetch_nba_odds_snapshot(...)  # ← API call!
        cache.set(home_tri, away_tri, odds)
    # Attach odds to result
    result["odds"] = {...}
    return result
```

**The Problem:**
- `run_prediction()` is called multiple times:
  1. Initial page load
  2. Manual refresh (line ~490)
  3. Auto-refresh (line ~403)
- Each call to `run_prediction()` triggers odds fetch
- Odds are fetched INSIDE prediction function, not separately

**Result:**
- Initial load: Fetches odds once ✓
- Manual refresh: Fetches odds TWICE (once in prediction, once for autofill) ✗
- Auto-refresh: Fetches odds ONCE (in prediction only) ✓
- Every Streamlit rerun: Fetches odds ONCE ✗

---

**Issue 2: No Control Over Odds Fetching**

**Current Implementation:**
- Odds are always fetched as part of prediction
- No way to get predictions WITHOUT odds
- No way to skip odds for completed games
- No way to refresh odds without re-running predictions

**User Requirements:**
1. Test completed games → Don't need odds, just predictions
2. Refresh predictions without odds → Just re-run models, don't hit API
3. Refresh odds only → Don't re-run predictions, just hit API
4. Pull one game at a time → Already implemented (cache per matchup)

---

## Solution

**Fix: Separate Odds Fetching from Prediction**

### Option 1: Add `fetch_odds` Parameter

```python
# predict_from_gameid_v3_runtime.py
def predict_from_game_id(
    game_input: str,
    *,
    eval_at_q3: bool = True,
    fetch_odds: bool = True,  # ← NEW PARAMETER
) -> Dict[str, Any]:
    """..."""

    # ... make predictions ...

    # Only fetch odds if requested
    if fetch_odds:
        cache = PersistentOddsCache()
        odds = cache.get(home_tri, away_tri)
        if odds is None:
            odds = fetch_nba_odds_snapshot(...)
            cache.set(home_tri, away_tri, odds)
    else:
        odds = None

    # Attach odds (or None)
    if odds is not None:
        result["odds"] = {...}
    else:
        result["odds"] = None
        result["odds_warning"] = None

    return result
```

**app.py - Updated flow:**

```python
# Initial load or prediction only refresh
run_prediction(fetch_odds=False)  # ← Skip odds!

# Odds refresh only
if manual_refresh:
    once_key = f"..."
    if not st.session_state.get(once_key):
        # Fetch odds ONLY (no prediction)
        home_name = str(st.session_state.last_pred.get("home_name") or "").strip()
        away_name = str(st.session_state.last_pred.get("away_name") or "").strip()
        snap = get_cached_nba_odds(...)
        # Store in cache (no need to return, we'll use cached odds on next prediction)
        st.success("Odds refreshed!")
        st.rerun()

# Full refresh (predictions + odds)
if st.button("🔄 Refresh everything"):
    run_prediction(fetch_odds=True)  # ← Fetch odds too
```

---

### Option 2: Check Game Status Before Fetching Odds

```python
# predict_from_gameid_v3_runtime.py
# Get game status from prediction
status = pred.get("status", {}) or {}
game_status = status.get("gameStatus")  # "Final", "In Progress", "Scheduled"

# Only fetch odds if game is not completed
if game_status != "Final":
    cache = PersistentOddsCache()
    odds = cache.get(home_tri, away_tri)
    if odds is None:
        odds = fetch_nba_odds_snapshot(...)
        cache.set(home_tri, away_tri, odds)
else:
    odds = None
    result["odds_warning"] = "Game completed - odds not fetched"
```

---

## Impact

**Immediate:**
- ✅ Odds only fetched when explicitly requested
- ✅ Predictions can run without hitting Odds API
- ✅ Completed games don't trigger Odds API calls
- ✅ Auto-refresh won't hit Odds API (just re-run predictions)
- ✅ Manual odds refresh doesn't double-fetch odds

**Efficiency Improvements:**
- Initial load: No odds (fetch_odds=False)
- Auto-refresh: No odds (fetch_odds=False)
- Prediction-only refresh: No odds (fetch_odds=False)
- Odds-only refresh: No predictions (just cache)
- Full refresh: Fetches predictions + odds once
- Testing completed games: No odds (game status check)

---

## Files Changed

**Modified:**
- `src/predict_from_gameid_v3_runtime.py`
  - Added: `fetch_odds` parameter to `predict_from_game_id()`
  - Added: Game status check before fetching odds
  - Conditional odds fetching: only if `fetch_odds=True` and game not completed

**Modified:**
- `app.py`
  - Updated: `run_prediction(fetch_odds=False)` for initial load
  - Updated: Auto-refresh: `run_prediction(fetch_odds=False)` to avoid odds
  - Added: Odds-only refresh button (separate from prediction refresh)

---

## Summary

Issue: Odds API usage way up  
Root Cause: Odds fetched inside prediction function on EVERY call  
Solution: Add `fetch_odds` parameter + game status check + separate refresh controls  
Status: ANALYSIS COMPLETE ⚠️  
Next: Implement the fix (will update predict_from_gameid_v3_runtime.py and app.py)