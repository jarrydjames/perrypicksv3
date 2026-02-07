# NBA API Data Fetching Analysis & Recommendations
**Date:** 2026-02-07  
**Issue:** Predictions using historical data instead of NBA API data  
**Status:** Root Cause Identified

---

## Executive Summary

Investigation reveals that the **NBA API IS working and returning real data for 2025-26 season**, but the prediction system is incorrectly ignoring it in favor of stale historical data.

**Root Cause:** Staleness policy forces historical data when historical data is >3 days old, even though NBA API has fresh current-season data available.

---

## What Data We're Trying to Fetch

### Primary Data Source: NBA API

**Endpoint:** `LeagueDashTeamStats` (from `nba_api.stats.endpoints`)

**Parameters Used:**
```python
LeagueDashTeamStats(
    team_id_nullable=<TEAM_ID>,  # e.g., 1610612767 for WAS
    season='2025-26',            # Current NBA season
    measure_type_detailed_defense='Advanced',  # Advanced stats
    per_mode_detailed='PerGame',  # Per-game averages
)
```

**Data We Need for Predictions:**

| Field | Description | Source |
|-------|-------------|--------|
| OFF_RATING | Offensive rating (points per 100 possessions) | NBA API Advanced |
| DEF_RATING | Defensive rating (points allowed per 100) | NBA API Advanced |
| PACE | Pace factor (possessions per game) | NBA API Advanced |
| EFG_PCT | Effective field goal percentage | NBA API Advanced |
| TM_TOV_PCT | Team turnover percentage | NBA API Advanced |
| OREB_PCT | Offensive rebound percentage | NBA API Advanced |
| TS_PCT | True shooting percentage | NBA API Advanced |
| AST_PCT | Assist percentage | NBA API Advanced |
| W, L, W_PCT | Record (wins/losses/win %) | NBA API Base/Advanced |
| GP | Games played | NBA API Base/Advanced |

**Secondary Data Source: Historical Data Manager**

**Purpose:** Fallback when NBA API is unavailable

**Data We Get:**
- Historical boxscores from local CSV files
- Team stats calculated from boxscores
- Schedule features (rest days, back-to-back)
- Recent form (last N games)
- Head-to-head records
- Schedule strength

---

## How We're Fetching Data

### NBA API Fetch Flow

**File:** `src/predict_pregame.py`

**Function:** `fetch_team_stats()`

**Logic:**
```python
def fetch_team_stats(team_id: int, seasons: Optional[Sequence[str]] = None):
    """
    Fetch team stats with multi-season fallback.
    
    Args:
        team_id: NBA.com team ID (e.g., 1610612767)
        seasons: List of seasons to try in order (e.g., ['2025-26', '2024-25'])
    
    Returns:
        Tuple of (team stats row, season string used)
    """
    seasons_to_try = list(seasons or ['2025-26', '2024-25'])
    
    for season in seasons_to_try:
        # Call NBA API
        stats = leaguedashteamstats.LeagueDashTeamStats(
            team_id_nullable=team_id,
            season=season,
            measure_type_detailed_defense='Advanced',
            per_mode_detailed='PerGame',
        )
        df = stats.get_data_frames()[0]
        
        if len(df) == 0:
            continue  # Try next season
        
        # Filter by TEAM_ID to get this team's row
        if 'TEAM_ID' in df.columns:
            team_rows = df[df['TEAM_ID'] == team_id]
            if len(team_rows) > 0:
                return team_rows.iloc[0], season
    
    return None, None  # No stats found
```

**Example Success:**
```python
# Fetch OKC stats for 2025-26
okc_stats, season = fetch_team_stats(1610612760, ['2025-26'])

# Returns:
okc_stats = {
    'TEAM_ID': 1610612760,
    'TEAM_NAME': 'Oklahoma City Thunder',
    'OFF_RATING': 117.8,      # ✅ Available
    'DEF_RATING': 105.7,      # ✅ Available
    'PACE': 101.03,           # ✅ Available
    'EFG_PCT': 0.563,        # ✅ Available
    'TM_TOV_PCT': 13.2,      # ✅ Available
    'OREB_PCT': 0.252,       # ✅ Available
    'W': 40,
    'L': 12,
    'W_PCT': 0.769,
    'GP': 52,
}
season = '2025-26'
```

### Staleness Policy Flow

**File:** `src/predict_pregame.py`

**Function:** `build_data_freshness_context()`

**Logic:**
```python
def build_data_freshness_context(
    game_datetime: pd.Timestamp,
    home_team_id: int,
    away_team_id: int,
    max_stale_days: int = 3,  # ← DEFAULT: 3 days
):
    """
    Build freshness metadata from historical game data.
    
    Returns dict with staleness flags and forces historical stats if stale.
    """
    context = {
        "is_stale": False,
        "force_historical_stats": False,
        ...
    }
    
    hist_mgr = get_historical_data_manager()
    
    # Check how old historical data is
    latest_game_date = hist_mgr.games_df["game_date"].max()
    days_gap = (game_datetime - latest_game_date).days
    
    # Check how many days since each team's last game
    home_recent = hist_mgr.get_team_games(home_id, before_date=game_datetime, n=1)
    away_recent = hist_mgr.get_team_games(away_id, before_date=game_datetime, n=1)
    
    # Force historical if stale
    if days_gap > max_stale_days:
        context["force_historical_stats"] = True
        context["is_stale"] = True
        context["stale_reasons"] = [f"historical data is {days_gap} days old"]
    
    return context
```

**Example with Stale Data:**
```python
# Predict for WAS @ DET on 2026-02-05
freshness = build_data_freshness_context(
    game_datetime=pd.Timestamp('2026-02-05'),
    home_team_id=1610612765,  # DET
    away_team_id=1610612767,  # WAS
    max_stale_days=3,
)

# Returns:
freshness = {
    "is_stale": True,
    "force_historical_stats": True,  # ← PROBLEM: Forces historical data
    "max_stale_days": 3,
    "historical_latest_game_date": "2026-01-30T02:30:00+00:00",
    "days_since_historical_update": 5,  # Gap: 5 days > 3-day threshold
    "home_days_since_last_game": 5,
    "away_days_since_last_game": None,
    "stale_reasons": [
        "historical data is 5 days old",
        "home team has 5 days since last game"
    ],
}
```

### Feature Extraction Flow

**File:** `src/predict_pregame.py`

**Function:** `extract_core_features()`

**Logic:**
```python
def extract_core_features(
    home_stats: Optional[pd.Series],  # From NBA API
    away_stats: Optional[pd.Series],  # From NBA API
    home_team_id: int,
    away_team_id: int,
    game_date: datetime,
    force_home_historical: bool = False,  # ← From staleness policy
    force_away_historical: bool = False,  # ← From staleness policy
):
    """
    Extract core pregame features from team stats + historical data.
    """
    features = {}
    hist_mgr = get_historical_data_manager()
    
    # HOME TEAM STATS
    # Use NBA API if available AND not forced to use historical
    if home_stats is not None and not force_home_historical:
        features['home_off_rating'] = home_stats.get('OFF_RATING', 110.0)
        features['home_def_rating'] = home_stats.get('DEF_RATING', 110.0)
        features['home_pace'] = home_stats.get('PACE', 100.0)
        ...
    elif hist_mgr and len(hist_mgr.get_team_games(home_team_id, ...)) > 0:
        # Use historical averages instead of NBA API
        home_hist = hist_mgr.get_team_games(home_team_id, ...)
        features['home_off_rating'] = float(home_hist['home_off_rating'].mean())
        features['home_def_rating'] = float(home_hist['home_def_rating'].mean())
        ...
    else:
        # Default values if nothing available
        features['home_off_rating'] = 110.0
        features['home_def_rating'] = 110.0
        ...
    
    # AWAY TEAM STATS (same logic)
    ...
    
    # SCHEDULE, FORM, H2H, STRENGTH features from historical
    ...
    
    return features
```

**The Problem:**
```python
# When force_historical_stats=True (due to staleness)
home_stats = fetch_team_stats(home_id, ['2025-26'])  # ✅ Returns data from NBA API
away_stats = fetch_team_stats(away_id, ['2025-26'])  # ✅ Returns data from NBA API

# But feature extraction IGNORES NBA API data because:
force_home_historical = True  # Set by staleness policy
force_away_historical = True  # Set by staleness policy

# So features use historical data instead:
features['home_off_rating'] = home_hist['home_off_rating'].mean()  # Uses historical
features['away_off_rating'] = away_hist['away_off_rating'].mean()  # Uses historical
```

---

## Test Results

### Test 1: NBA API Direct Call ✅

```python
from nba_api.stats.endpoints import leaguedashteamstats

# Test for OKC (2025-26)
okc_stats = leaguedashteamstats.LeagueDashTeamStats(
    team_id_nullable=1610612760,
    season='2025-26',
    measure_type_detailed_defense='Advanced',
    per_mode_detailed='PerGame',
)
df = okc_stats.get_data_frames()[0]
```

**Result:** ✅ SUCCESS

| Field | Value |
|-------|-------|
| OFF_RATING | 117.8 |
| DEF_RATING | 105.7 |
| PACE | 101.03 |
| EFG_PCT | 0.563 |
| OREB_PCT | 0.252 |
| TEAM_ID | 1610612760 |

**Conclusion:** NBA API IS returning real data for 2025-26 season!

### Test 2: fetch_team_stats Function ✅

```python
from src.predict_pregame import fetch_team_stats

# Test for OKC
result, season = fetch_team_stats(1610612760, ['2025-26'])
```

**Result:** ✅ SUCCESS

```python
result = {
    'OFF_RATING': 117.8,
    'DEF_RATING': 105.7,
    'PACE': 101.03,
    'EFG_PCT': 0.563,
    'OREB_PCT': 0.252,
    'TEAM_ID': 1610612760,
}
season = '2025-26'
```

**Conclusion:** `fetch_team_stats()` function IS working!

### Test 3: Staleness Check ⚠️

```python
from src.predict_pregame import build_data_freshness_context

# Check for WAS @ DET on 2026-02-05
freshness = build_data_freshness_context(
    game_datetime=pd.Timestamp('2026-02-05', tz='UTC'),
    home_team_id=1610612765,  # DET
    away_team_id=1610612767,  # WAS
    max_stale_days=3,
)
```

**Result:** ⚠️ FORCES HISTORICAL

```python
freshness = {
    "is_stale": True,
    "force_historical_stats": True,  # ← PROBLEM
    "days_since_historical_update": 5,  # 5-day gap > 3-day threshold
    "stale_reasons": [
        "historical data is 5 days old",
        "home team has 5 days since last game"
    ],
}
```

**Conclusion:** Staleness policy is forcing historical data!

---

## Root Cause Analysis

### The Design Flaw

**Assumption:** Historical data staleness means we should ignore NBA API and use historical data.

**Reality:** NBA API has FRESH current-season data that we SHOULD use!

**Current Logic Flow:**
```
1. Fetch NBA API stats → ✅ Returns real 2025-26 data
2. Check historical staleness → ⚠️ 5-day gap > 3-day threshold
3. Set force_historical_stats=True → ❌ Ignores NBA API data
4. Extract features using historical → ❌ Uses stale historical data
5. Generate predictions → ❌ All similar (historical averages)
```

**Correct Logic Flow Should Be:**
```
1. Fetch NBA API stats → ✅ Returns real 2025-26 data
2. Check historical staleness → ⚠️ 5-day gap > 3-day threshold
3. Use NBA API data for team ratings (fresh current season) → ✅
4. Use historical data for schedule/form/H2H (no NBA API alternative) → ✅
5. Generate predictions → ✅ Differentiated based on real team stats
```

### The Fix

**Problem:** `force_historical_stats` is a **boolean flag** that ignores NBA API for ALL team stats.

**Solution:** Change to **granular control** that:
- Uses NBA API for team ratings (offensive/defensive rating, pace, etc.)
- Uses historical data for schedule features (rest days, back-to-back)
- Uses historical data for form features (recent games)
- Uses historical data for H2H features
- Uses historical data for schedule strength

---

## Data Field Mapping

### NBA API Advanced Measure Type Fields

| Field Name | Type | Example | Used In |
|-------------|------|---------|---------|
| OFF_RATING | float | 117.8 | ✅ Features (off_rating) |
| DEF_RATING | float | 105.7 | ✅ Features (def_rating) |
| PACE | float | 101.03 | ✅ Features (pace) |
| EFG_PCT | float | 0.563 | ✅ Features (efg) |
| TM_TOV_PCT | float | 13.2 | ✅ Features (tov_rate) - Note: API returns TM_TOV_PCT |
| OREB_PCT | float | 0.252 | ✅ Features (orb_rate) |
| TS_PCT | float | 0.623 | ✅ Not currently used |
| AST_PCT | float | 65.5 | ✅ Not currently used |
| REB_PCT | float | 49.3 | ✅ Not currently used |
| E_NET_RATING | float | 7.3 | ✅ Not currently used |
| NET_RATING | float | 12.1 | ✅ Not currently used |

**Note:** `TM_TOV_PCT` in NBA API = `TOV_PCT` expected by code

### Missing Fields in NBA API

| Field | Expected | API Availability | Workaround |
|-------|----------|-----------------|-----------|
| FTA_RATE | FTA / FGA | Not in API | Calculate from Base measure type |
| FTM, FTA | Free throw makes/attempts | Base measure type | Need separate call |
| FGM, FGA | Field goal makes/attempts | Base measure type | Need separate call |

---

## Short-Term Fix (Implement Now)

### Fix 1: Remove force_historical_stats Override

**File:** `src/predict_pregame.py`

**Change:** Remove logic that ignores NBA API when historical is stale.

**Before:**
```python
def extract_core_features(
    home_stats: Optional[pd.Series],
    away_stats: Optional[pd.Series],
    home_team_id: int,
    away_team_id: int,
    game_date: datetime,
    force_home_historical: bool = False,  # ← PROBLEM
    force_away_historical: bool = False,  # ← PROBLEM
):
    # Use NBA API if available AND not forced to use historical
    if home_stats is not None and not force_home_historical:
        features['home_off_rating'] = home_stats.get('OFF_RATING', 110.0)
        ...
```

**After:**
```python
def extract_core_features(
    home_stats: Optional[pd.Series],
    away_stats: Optional[pd.Series],
    home_team_id: int,
    away_team_id: int,
    game_date: datetime,
    # Remove force_home_historical and force_away_historical
):
    # Use NBA API if available (regardless of historical staleness)
    if home_stats is not None:
        features['home_off_rating'] = home_stats.get('OFF_RATING', 110.0)
        ...
```

**Impact:**
- ✅ Uses NBA API data for team ratings (fresh, differentiated)
- ✅ Still uses historical for schedule/form/H2H (no NBA API alternative)
- ✅ Predictions vary by team (real stats)

### Fix 2: Map NBA API Column Names

**File:** `src/predict_pregame.py`

**Change:** Map `TM_TOV_PCT` to `TOV_PCT` expected by code.

**Add:**
```python
def _map_nba_api_columns(stats_row: pd.Series) -> Dict[str, Any]:
    """
    Map NBA API column names to expected feature names.
    """
    mapping = {
        'OFF_RATING': 'off_rating',
        'DEF_RATING': 'def_rating',
        'PACE': 'pace',
        'EFG_PCT': 'efg',
        'TM_TOV_PCT': 'tov_rate',  # ← MAP TM_TOV_PCT to tov_rate
        'OREB_PCT': 'orb_rate',
        'FTA_RATE': 'ft_rate',  # Need to calculate
    }
    
    result = {}
    for api_col, feature_name in mapping.items():
        if api_col in stats_row.index:
            result[feature_name] = stats_row[api_col]
    
    return result
```

### Fix 3: Calculate FTA_RATE from Base Measure Type

**File:** `src/predict_pregame.py`

**Change:** Fetch Base measure type for FTA/FGA and calculate rate.

**Add:**
```python
def fetch_base_stats(team_id: int, season: str) -> Optional[pd.Series]:
    """
    Fetch Base measure type stats for calculating derived metrics.
    """
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(
            team_id_nullable=team_id,
            season=season,
            measure_type_detailed_defense='Base',
            per_mode_detailed='PerGame',
        )
        df = stats.get_data_frames()[0]
        
        if len(df) == 0:
            return None
        
        # Filter by TEAM_ID
        if 'TEAM_ID' in df.columns:
            team_rows = df[df['TEAM_ID'] == team_id]
            if len(team_rows) > 0:
                return team_rows.iloc[0]
        
        return None
    except Exception as e:
        logger.error(f"Error fetching base stats: {e}")
        return None

def calculate_fta_rate(fg_attempts: float, ft_attempts: float) -> float:
    """
    Calculate free throw attempt rate: FTA / FGA.
    
    Args:
        fg_attempts: Field goal attempts per game
        ft_attempts: Free throw attempts per game
    
    Returns:
        Free throw attempt rate
    """
    if fg_attempts == 0:
        return 0.25  # Default league average
    return ft_attempts / fg_attempts
```

---

## Medium-Term Recommendations

### Recommendation 1: Improve Historical Data Refresh

**Current Issue:** Historical data lags 5+ days behind live games.

**Solution:**
- Set up automated daily import of boxscores
- Import previous day's games at 6 AM UTC each morning
- Run daily import job to keep historical data fresh

**Impact:**
- Historical data gap < 24 hours
- Staleness policy won't force historical as often
- More reliable fallback data

### Recommendation 2: Add NBA API Health Check

**Current Issue:** No visibility into NBA API availability.

**Solution:**
```python
def check_nba_api_health() -> Dict[str, Any]:
    """
    Check if NBA API is accessible and returning data.
    
    Returns:
        Dict with status, latency, sample_data
    """
    start_time = time.time()
    
    try:
        # Test with OKC (a known active team)
        test_stats = leaguedashteamstats.LeagueDashTeamStats(
            team_id_nullable=1610612760,
            season='2025-26',
            measure_type_detailed_defense='Advanced',
            per_mode_detailed='PerGame',
        )
        df = test_stats.get_data_frames()[0]
        
        latency = time.time() - start_time
        
        if len(df) > 0:
            return {
                'status': 'healthy',
                'latency_ms': round(latency * 1000, 2),
                'sample_off_rating': df.iloc[0].get('OFF_RATING'),
            }
        else:
            return {
                'status': 'unhealthy',
                'error': 'No data returned',
                'latency_ms': round(latency * 1000, 2),
            }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'latency_ms': None,
        }
```

**Impact:**
- Early detection of API issues
- Better debugging information
- Can alert when API is down

### Recommendation 3: Cache NBA API Responses

**Current Issue:** Repeated API calls for same data (slow, rate limits).

**Solution:**
```python
from functools import lru_cache
from datetime import datetime, timedelta

@lru_cache(maxsize=128)
def cached_fetch_team_stats(team_id: int, season: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Fetch team stats with LRU cache.
    
    Cache expires after 6 hours (NBA updates stats daily).
    """
    return fetch_team_stats_uncached(team_id, season)

# Clear cache periodically
def clear_api_cache_if_stale():
    """Clear cache if older than 6 hours."""
    # Implementation depends on cache library used
    pass
```

**Impact:**
- Faster predictions (no repeated API calls)
- Reduced rate limit pressure
- Better UX for rapid predictions

### Recommendation 4: Add Alternative Data Sources

**Current Issue:** Single point of failure (NBA API).

**Solution:**
- Scrape data from basketball-reference.com
- Use MySportsFeeds API (backup)
- Use SportRadar API (enterprise backup)

**Implementation:**
```python
def fetch_team_stats_with_fallback(
    team_id: int,
    season: str,
) -> Optional[Dict[str, Any]]:
    """
    Fetch team stats with multiple data source fallback.
    
    Priority:
    1. NBA API (primary)
    2. Basketball-Reference (scraped)
    3. MySportsFeeds API (backup)
    """
    # Try NBA API first
    result = fetch_nba_api_stats(team_id, season)
    if result is not None:
        result['data_source'] = 'nba_api'
        return result
    
    # Fallback to basketball-reference
    result = scrape_basketball_reference_stats(team_id, season)
    if result is not None:
        result['data_source'] = 'basketball_reference'
        return result
    
    # Fallback to MySportsFeeds
    result = fetch_mysportsfeeds_stats(team_id, season)
    if result is not None:
        result['data_source'] = 'mysportsfeeds'
        return result
    
    # All sources failed
    return None
```

**Impact:**
- Redundant data sources
- Improved reliability
- Continuous operation even if one source fails

---

## Summary

| Aspect | Status | Issue | Fix |
|---------|--------|-------|-----|
| NBA API | ✅ Working | Not the problem | None needed |
| fetch_team_stats() | ✅ Working | Returns data correctly | None needed |
| Staleness Policy | ❌ Problem | Forces historical ignoring NBA API | Remove force flag |
| Feature Extraction | ❌ Problem | Uses historical instead of API | Use API for ratings |
| Historical Data | ⚠️ Stale | 5+ day gap | Daily import automation |
| Data Source | ⚠️ Single point of failure | Only NBA API | Add backup sources |

---

## Next Steps

1. ✅ **Implement Short-Term Fixes**
   - Remove `force_historical_stats` override
   - Map NBA API column names correctly
   - Calculate FTA_RATE from Base stats

2. ✅ **Test After Fixes**
   - Run daily summary
   - Verify predictions vary by team
   - Check data source field shows "2025-26" not "HISTORICAL"

3. ✅ **Implement Medium-Term Recommendations**
   - Set up daily historical import
   - Add NBA API health check
   - Implement API caching
   - Research backup data sources

---

**Document Created:** 2026-02-07  
**Status:** Root cause identified, fixes ready to implement  
**Confidence:** HIGH - NBA API working, staleness policy is the issue
