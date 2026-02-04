# Data Requirements Analysis - NBA API Migration

**Date:** 2025-02-04
**Purpose:** Analyze data requirements before refactoring NBA data layer

---

## Executive Summary

The system uses **THREE different NBA APIs** for different purposes:

1. **Schedule API** (CDN) - Game scheduling (WORKING ✅)
2. **Boxscore API** (CDN) - Game state and detailed stats (WORKING ✅)
3. **Team Stats API** (stats.nba.com) - Advanced team statistics (USED BY PREGAME MODEL)

**Key Finding:** The Team Stats API is ONLY used by pregame model and is a DIFFERENT API from game state fetching.

---

## Data Requirements by Prediction Type

### 1. Pregame Predictions

**Data Source:** `nba_api.stats.endpoints.leaguedashteamstats.LeagueDashTeamStats`

**Purpose:** Fetch advanced team statistics (ratings, eFG%, pace, etc.)

**API Endpoint:**
```python
stats = leaguedashteamstats.LeagueDashTeamStats(
    team_id_nullable=team_id,
    season=season,
    measure_type_detailed_defense='Advanced',
    per_mode_detailed='PerGame',
)
```

**Data Retrieved:**
- OFF_RATING, DEF_RATING (offensive/defensive ratings)
- PACE (tempo)
- EFG_PCT (effective field goal percentage)
- FTA_RATE (free throw rate)
- TOV_PCT (turnover percentage)
- OREB_PCT (offensive rebound percentage)
- W, GP (wins, games played)

**Features Extracted (72 total):**
- Basic team ratings (18): off/def rating, pace, efg, tov/orb/ft rate, win pct
- Schedule features (8): rest days, back-to-back
- Recent form features (11): recent points/allowed/margin/wins (last 10 games)
- Four factors / Net rating (20): net rating, TS proxy, four factor weighted
- Head-to-head features (13): H2H wins, total games, win pct, recent H2H
- Schedule strength (2): opponent strength

**API Status:**
- ⚠️ Uses `stats.nba.com` (NOT CDN-based)
- ✅ Works for fetching team-level advanced stats
- ⚠️ May need retry logic for rate limiting

**Migration Plan:**
- KEEP this API (it's the only source for advanced team stats)
- ADD retry logic and proper headers
- ADD caching (team stats don't change often)

---

### 2. Halftime Predictions

**Data Source:** CDN Boxscore API

**Purpose:** Fetch game state and detailed stats at halftime

**API Endpoint:**
```python
BOX_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"
```

**Data Retrieved:**
- Game status (gameStatusText: "Halftime", "Q3", "Final")
- Period and clock (period, gameClock)
- Team scores (score field)
- Team statistics (statistics object):
  - fieldGoalsAttempted, fieldGoalsMade
  - fieldGoalsEffectiveAdjusted (eFG%)
  - threePointersAttempted, threePointersMade
  - freeThrowsAttempted, freeThrowsMade
  - reboundsOffensive, reboundsDefensive, reboundsTotal
  - turnoversTotal
  - points
- Play-by-play data (for behavior counts)

**Features Extracted (12 total):**
- h1_home, h1_away (first half scores)
- h1_total, h1_margin (half total and margin)
- h1_events (total events)
- h1_n_2pt, h1_n_3pt (shot counts)
- h1_n_turnover, h1_n_rebound (action counts)
- h1_n_foul, h1_n_timeout, h1_n_sub (other actions)

**API Status:**
- ✅ CDN-based (no timeouts)
- ✅ Works from Streamlit Cloud
- ✅ Already implemented in `src/data/game_data.py`
- ✅ Fallback to schedule API on 403

**Migration Plan:**
- KEEP current implementation (it's already correct)
- MOVE to `core/data_sources.py` as single source of truth

---

### 3. Q3 Predictions

**Data Source:** CDN Boxscore API

**Purpose:** Fetch game state and detailed stats at Q3

**API Endpoint:**
```python
BOX_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"
```

**Data Retrieved:**
- Same as halftime (game status, scores, stats)
- Plus additional Q3-specific features

**Features Extracted (22-26 total):**
- Same 12 H1 features
- Plus Q3-specific features (team stats in Q3)

**API Status:**
- ✅ CDN-based (no timeouts)
- ✅ Works from Streamlit Cloud
- ✅ Same API as halftime

**Migration Plan:**
- KEEP current implementation
- MOVE to `core/data_sources.py`

---

## Automation Requirements

### Current Implementation

**File:** `core/data_sources.py`

**Game State Fetching:**
```python
# BROKEN - times out frequently
boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(
    game_id=game_id,
    timeout=NBA_API_TIMEOUT  # 30 seconds
)
data = boxscore.get_dict()
```

**Issues:**
- ❌ Hits `stats.nba.com` which times out
- ❌ Can't find games 0022500724 and 0022500725
- ❌ Can't update game status to "Completed"
- ❌ All 8 completed games stuck as "In Progress"

### Required Data for Automation

The automation ONLY needs game state:

1. **Game Status:** Scheduled / In Progress / Halftime / Final
2. **Period:** 0-4 (Q1-Q4)
3. **Clock:** Minutes and seconds remaining
4. **Scores:** Home and away team scores
5. **Teams:** Home and away tricodes

**API Status:**
- ✅ CDN boxscore API provides ALL of this data
- ✅ Already working in `src/data/game_data.py` and `src/data/scoreboard.py`
- ❌ NOT used by automation (uses broken stats.nba.com instead)

### Migration Plan for Automation

**Replace:**
```python
# core/data_sources.py - fetch_game_state()

# BEFORE (BROKEN):
boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(...)

# AFTER (WORKING):
# Use CDN boxscore API like src/data/game_data.py does
url = f"https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
response = requests.get(url, headers=NBA_HEADERS, timeout=30)
data = response.json()
```

**Benefits:**
- ✅ No more timeouts
- ✅ Can find all games
- ✅ Game state updates correctly
- ✅ Single implementation (DRY)

---

## API Inventory

### APIs to KEEP (They're needed):

1. **Schedule API** (CDN)
   - URL: `https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json`
   - Purpose: Fetch game schedule
   - Status: ✅ Working

2. **Boxscore API** (CDN)
   - URL: `https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json`
   - Purpose: Fetch game state and detailed stats
   - Status: ✅ Working (in src/data/)

3. **Team Stats API** (stats.nba.com)
   - Module: `nba_api.stats.endpoints.leaguedashteamstats`
   - Purpose: Fetch advanced team statistics for pregame model
   - Status: ⚠️ Needs retry/caching

### APIs to REPLACE:

1. **BoxScoreTraditionalV2** (stats.nba.com)
   - Module: `nba_api.stats.endpoints.boxscoretraditionalv2`
   - Used by: `core/data_sources.py` fetch_game_state()
   - Problem: Times out frequently
   - Replacement: CDN boxscore API

---

## Risk Assessment

### Risks of Migration

| Risk | Likelihood | Impact | Mitigation |
|-------|-----------|---------|------------|
| Team Stats API rate limiting | Medium | Medium | Add retry logic + caching |
| CDN boxscore missing some data | Low | Low | Test with multiple games |
| Breaking pregame predictions | Low | High | Test thoroughly before deploying |

### Benefits of Migration

| Benefit | Impact |
|---------|---------|
| Eliminate timeouts | Critical |
| All games found | Critical |
| Game state updates correctly | Critical |
| Single source of truth | High |
| Easier to maintain | High |
| Better performance | Medium |

---

## Recommended Approach

### Phase 1: Core Data Layer (No Live Games Needed)
1. Update `core/data_sources.py` with CDN-based fetch_game_state()
2. Add caching layer for game state
3. Add retry logic for Team Stats API
4. Test with historical games (2026-02-03 completed games)

### Phase 2: Refactor Automation
1. Update automation to use core/data_sources.py
2. Remove duplicate NBA API code
3. Test with completed games

### Phase 3: Testing & Validation
1. Test all three prediction types (pregame, halftime, Q3)
2. Verify all features are extracted correctly
3. Test with multiple historical games

### Phase 4: Deploy
1. Deploy to production
2. Monitor with live games
3. Validate all triggers fire correctly

---

## Conclusion

**Finding:** The system has 3 NBA APIs:
1. ✅ Schedule API (CDN) - Keep as-is
2. ✅ Boxscore API (CDN) - Already works, move to core
3. ⚠️ Team Stats API (stats.nba.com) - Keep, add retry/caching

**Critical Issue:** Automation uses BoxScoreTraditionalV2 (stats.nba.com) which times out
**Solution:** Replace with CDN boxscore API (already working in src/data/)

**Impact:** This will NOT impede ability to make predictions. In fact, it will improve prediction reliability by ensuring game state is always available.

**Recommendation:** Proceed with migration as planned. All necessary data will still be available.

