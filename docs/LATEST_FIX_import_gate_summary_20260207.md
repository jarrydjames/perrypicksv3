# Latest Fix: Pre-Prediction Import Gate & Data Quality Controls
**Date:** 2026-02-07  
**Commit:** 490ea79 (PR #19)  
**Status:** ✅ Applied & Merged

---

## Executive Summary

The latest fix addresses the **silent default fallback** issue by implementing a **pre-prediction import gate**. Before this fix, predictions would silently fall back to default values when data was stale or missing, resulting in identical predictions across all games.

**Now:** Predictions are **explicitly blocked** when data imports are not recent, with clear error messages.

---

## Problem Solved

### Before (Silent Fallback):

```python
# System would silently use defaults when data unavailable
def predict_game(game_input, mode="pregame"):
    if mode == "pregame":
        result = predict_pregame(game_input, ...)
        # If data is stale → silent fallback to defaults
        # If data missing → silent fallback to defaults
        # Result: All predictions nearly identical (90.2 @ 91.3)
```

**Issues:**
- No validation of data freshness before prediction
- No checks for placeholder games (UNK teams)
- Silent fallback to default league averages
- No visibility into data quality issues
- Users receive meaningless predictions

### After (Explicit Blocking):

```python
# System now blocks predictions with explicit errors
def predict_game(game_input, mode="pregame"):
    if mode == "pregame":
        # NEW: Run import gate FIRST
        gate_error = _pregame_import_gate(
            game_id=game_input,
            home_team=home_team,
            away_team=away_team,
        )
        if gate_error is not None:
            return gate_error  # ← BLOCK with explicit error
        
        # Only proceed if gate passed
        result = predict_pregame(game_input, ...)
```

**Result:**
- ✅ Predictions blocked when data is stale
- ✅ Explicit error messages (STALE_DATA, PLACEHOLDER_GAME)
- ✅ No silent fallback to defaults
- ✅ Clear visibility into data quality issues
- ✅ Users know why prediction failed

---

## Three-Layer Defense System

### Layer 1: Import-Time Validation 🔒

**File:** `scripts/automation/game_scanner.py`

**Validations:**
1. Game ID format (must be 10 digits)
2. Home team tricode (not UNK, HOME, empty)
3. Away team tricode (not UNK, AWAY, empty)

**Actions:**
- Valid games → Process downstream
- Invalid games → Quarantine to audit file

**Quarantine File:** `data/diagnostics/quarantined_games_20260207.json`

```json
[
  {
    "game_id": "0022500742",
    "away": "UNK",
    "home": "UNK", 
    "status_text": "Scheduled",
    "reason": "INVALID_HOME_TEAM"
  }
]
```

### Layer 2: Import Watermarking 📝

**File:** `src/data/import_health.py`

**Watermark File:** `data/diagnostics/import_watermark.json`

```json
{
  "updated_at_utc": "2026-02-07T10:30:45.123456+00:00",
  "source": "cdn_nba_schedule_boxscore",
  "game_date": "2026-02-07",
  "valid_games": 12,
  "quarantined_games": 1,
  "latest_game_time_utc": "2026-02-07T23:59:59+00:00"
}
```

**Tracks:**
- Import timestamp (when was data last refreshed?)
- Data source (which API endpoint?)
- Game date (which day?)
- Valid games count (how many passed validation?)
- Quarantined games count (how many were rejected?)
- Latest game time (most recent game in data)

**Purpose:** Observable import state for freshness validation

### Layer 3: Prediction-Time Gating 🚦

**File:** `src/predict_api.py`

**Gate Function:** `_pregame_import_gate()`

**Checks:**

1. **Placeholder Team Check:**
```python
def _is_placeholder_team(tricode: Optional[str]) -> bool:
    t = str(tricode or "").strip().upper()
    return t in {"", "UNK", "HOME", "AWAY"}

if _is_placeholder_team(home_team) or _is_placeholder_team(away_team):
    return {
        "status": "error",
        "error": "PLACEHOLDER_GAME: invalid team tricode(s)",
        "model_used": "IMPORT_GATE",
    }
```

2. **Watermark Existence Check:**
```python
watermark = read_import_watermark()
if not watermark:
    return {
        "status": "error",
        "error": "STALE_DATA: import watermark not found; run game scanner/import job first",
        "game_id": game_id,
        "model_used": "IMPORT_GATE",
    }
```

3. **Watermark Freshness Check:**
```python
updated_at = watermark.get("updated_at_utc")
age_hours = (now_utc() - parse_timestamp(updated_at)).total_seconds() / 3600.0

max_hours = float(os.getenv("PREGAME_IMPORT_MAX_AGE_HOURS", "36"))

if age_hours > max_hours:
    return {
        "status": "error",
        "error": f"STALE_DATA: watermark is {age_hours:.1f}h old (max {max_hours:.1f}h)",
        "game_id": game_id,
        "model_used": "IMPORT_GATE",
        "data_freshness": {"watermark_age_hours": age_hours, "max_hours": max_hours},
    }
```

**Configuration:**
```bash
# Maximum age of import watermark (in hours)
# Predictions blocked if watermark is older than this
export PREGAME_IMPORT_MAX_AGE_HOURS=36
```

**Recommended Settings:**
- **Development:** 72 hours (permissive)
- **Production:** 12 hours (stricter)
- **Game Day:** 6 hours (very strict)

---

## Conclusion

The pre-prediction import gate ensures that **only real game data drives predictions** by:

1. ✅ Validating games at import time (quarantine invalid entries)
2. ✅ Recording import state (watermarking)
3. ✅ Blocking predictions when data is stale (import gate)
4. ✅ Returning explicit errors (no silent fallback)

**Result:** True predictions based on real 2025-26 season data, or clear error messages explaining why prediction was blocked!

**Status:** ✅ IMPLEMENTED & MERGED
**PR:** #19

---
**Document Generated:** 2026-02-07  
**Commit:** 490ea79  
**Status:** Active & Enforcing data quality controls
