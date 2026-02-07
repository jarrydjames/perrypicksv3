# Implementation Summary: Fixing NBA API Data Fetching
**Date:** 2026-02-07  
**Status:** Fixes Identified & Ready to Implement

---

## Executive Summary

Investigation revealed **THREE ROOT CAUSES** for identical predictions:

1. **Incorrect TEAM_ID mappings** - Utah Jazz & Washington Wizards have wrong IDs
2. **Staleness policy forcing historical** - Ignoring NBA API when historical is stale
3. **Column name mapping** - TM_TOV_PCT → TOV_PCT mismatch

**All three issues must be fixed together for predictions to work correctly.**

---

## Root Cause #1: Incorrect TEAM_ID Mappings

### Problem

Two teams have incorrect TEAM_ID values, causing NBA API to return no data:

| Tricode | Team Name | TEAM_ID in Code | TEAM_ID in NBA API | Status |
|----------|------------|----------------|----------------------|--------|
| UTA | Utah Jazz | 1610612764 | 1610612762 | ❌ INCORRECT |
| WAS | Washington Wizards | 1610612767 | 1610612764 | ❌ INCORRECT |

### Impact

- When predicting games involving Utah or Washington
- NBA API returns no data (wrong TEAM_ID)
- System falls back to historical data
- Predictions use stale averages instead of real stats

### Evidence

```python
# Code tries to fetch with wrong ID
fetch_team_stats(1610612767, ['2025-26'])  # WAS wrong ID
# NBA API returns: 0 rows
# System falls back to historical data
```

### Fix

**File:** `src/predict_pregame.py`

**Line:** 32

**Current:**
```python
TEAM_IDS = {
    ...
    'TOR': 1610612762, 'UTA': 1610612764, 'WAS': 1610612767,
}
```

**After:**
```python
TEAM_IDS = {
    ...
    'TOR': 1610612761, 'UTA': 1610612762, 'WAS': 1610612764,
}
```

**Note:** Toronto's TEAM_ID was also incorrect (1610612762 is actually Utah Jazz's ID).

---

## Root Cause #2: Staleness Policy Forcing Historical

### Problem

When historical data is >3 days old, system sets `force_historical_stats=True`, which causes ALL team stats to use historical data instead of NBA API.

### Flow

**Current (WRONG):**
```
1. Fetch NBA API stats → ✅ Returns real 2025-26 data
2. Check historical staleness → ⚠️ 5-day gap > 3-day threshold
3. Set force_historical_stats=True → ❌ Ignores NBA API data
4. Extract features → ❌ Uses stale historical data
5. Predict → ❌ All similar (historical averages)
```

**Correct Flow:**
```
1. Fetch NBA API stats → ✅ Returns real 2025-26 data
2. Check historical staleness → ⚠️ 5-day gap (warn user)
3. Use NBA API for team ratings → ✅ Fresh current-season data
4. Use historical for schedule/form/H2H → ✅ No NBA API alternative
5. Predict → ✅ Differentiated based on real stats
```

### Impact

- NBA API has fresh 2025-26 season data available
- But system ignores it when historical is stale
- All predictions use historical averages (90.3 @ 91.3)

### Example

```python
# NBA API returns fresh data
home_stats = {
    'OFF_RATING': 116.4,  # DET
    'DEF_RATING': 108.4,
    'PACE': 100.64,
}

away_stats = {
    'OFF_RATING': 109.6,  # WAS (when TEAM_ID fixed)
    'DEF_RATING': 119.9,
    'PACE': 101.03,
}

# But force_historical_stats=True ignores this and uses historical:
features['home_off_rating'] = 111.98  # Historical average
features['away_off_rating'] = 110.0   # Default
```

### Fix

**File:** `src/predict_pregame.py`

**Function:** `predict_from_game_id()`

**Remove these lines:**
```python
# If historical data is stale, prefer historical-derived ratings instead of stale API season table.
force_historical_stats = bool(freshness.get("force_historical_stats"))
```

**Pass `force_historical_stats=False` to `extract_core_features()`:**
```python
# Extract features with historical data
features = extract_core_features(
    home_stats,
    away_stats,
    home_id,
    away_id,
    resolved_game_datetime,
    force_home_historical=False,  # ← Always use NBA API if available
    force_away_historical=False,  # ← Always use NBA API if available
)
```

**Alternative approach:** Remove force_historical parameters entirely:
```python
def extract_core_features(
    home_stats: Optional[pd.Series],
    away_stats: Optional[pd.Series],
    home_team_id: int,
    away_team_id: int,
    game_date: datetime,
    # Remove force_home_historical and force_away_historical parameters
):
    # Use NBA API if available
    if home_stats is not None:
        features['home_off_rating'] = home_stats.get('OFF_RATING', 110.0)
        ...
```

---

## Root Cause #3: Column Name Mapping

### Problem

NBA API Advanced measure type returns `TM_TOV_PCT` but code expects `TOV_PCT`.

### Evidence

**NBA API Advanced columns:**
```
...
TM_TOV_PCT
OREB_PCT
...
```

**Code expects:**
```python
features['home_tov_rate'] = home_stats.get('TOV_PCT', 0.15)  # ← Won't find it!
```

**Result:** `TOV_PCT` field gets default value (0.15) instead of NBA API value.

### Fix

**File:** `src/predict_pregame.py`

**Function:** `extract_core_features()`

**Add mapping:**
```python
def extract_core_features(...):
    features = {}
    
    # Map NBA API column names to expected feature names
    def map_api_columns(stats_row: pd.Series) -> Dict[str, Any]:
        mapping = {
            'OFF_RATING': 'off_rating',
            'DEF_RATING': 'def_rating',
            'PACE': 'pace',
            'EFG_PCT': 'efg',
            'TM_TOV_PCT': 'tov_rate',  # ← MAP TM_TOV_PCT to tov_rate
            'OREB_PCT': 'orb_rate',
        }
        result = {}
        for api_col, feat_name in mapping.items():
            if api_col in stats_row.index:
                result[feat_name] = stats_row[api_col]
        return result
    
    # Use mapped columns
    if home_stats is not None:
        home_mapped = map_api_columns(home_stats)
        features['home_off_rating'] = home_mapped.get('off_rating', 110.0)
        features['home_tov_rate'] = home_mapped.get('tov_rate', 0.15)
        ...
```

---

## Implementation Plan

### Step 1: Fix TEAM_IDS Mapping

**File:** `src/predict_pregame.py`  
**Lines:** 30-32

**Action:** Update incorrect TEAM_IDs:
- TOR: 1610612762 → 1610612761
- UTA: 1610612764 → 1610612762
- WAS: 1610612767 → 1610612764

### Step 2: Remove Staleness Override

**File:** `src/predict_pregame.py`  
**Function:** `predict_from_game_id()`

**Action:** Remove `force_historical_stats` logic and always use NBA API data when available

**Approach A (Recommended):** Remove force parameters entirely
- Always use NBA API if stats available
- Remove staleness override
- Keep staleness warnings for user visibility

**Approach B (Alternative):** Smart override
- Only use historical if NBA API fails (not available)
- Don't use historical just because historical is stale
- Still warn user about staleness

### Step 3: Fix Column Mapping

**File:** `src/predict_pregame.py`  
**Function:** `extract_core_features()`

**Action:** Map NBA API column names correctly
- TM_TOV_PCT → TOV_PCT
- Handle any other column name mismatches

### Step 4: Test

**Actions:**
1. Run `debug_data_fetch.py` to verify NBA API returns data for all teams
2. Test prediction for a game with correct TEAM_IDs
3. Verify data_source shows "2025-26" not "HISTORICAL"
4. Run full daily summary to verify predictions vary by team

---

## Expected Results After Fixes

| Aspect | Before | After |
|---------|---------|--------|
| Utah Jazz NBA API | ❌ No data (wrong ID) | ✅ Real stats (correct ID) |
| Washington NBA API | ❌ No data (wrong ID) | ✅ Real stats (correct ID) |
| Staleness Override | ❌ Forces historical | ✅ Uses NBA API when available |
| TOV_PCT Field | ❌ Always default (0.15) | ✅ NBA API value |
| Predictions | ⚠️ All 90.3 @ 91.3 | ✅ Vary by team |
| Data Source | ⚠️ HISTORICAL/HISTORICAL | ✅ 2025-26/2025-26 |

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/predict_pregame.py` | Fix TEAM_IDS, remove staleness override, fix column mapping |
| `docs/NBA_API_DATA_ANALYSIS_20260207.md` | Already created (analysis) |
| `docs/TEAM_ID_MAPPING_FIX_20260207.md` | Already created (TEAM_ID fix details) |

---

**Status:** Ready to implement  
**Confidence:** HIGH - Root causes identified, fixes clear  
**Estimated Time:** 30 minutes
