# Issues Summary: Rate Limiting, Automation Triggers, and Preview Table

**Date:** 2026-02-08  
**Status:** Partially Resolved

---

## Issue 1: NBA.com CDN Rate Limiting (403 Forbidden) 🚨

### Problem
All halftime and Q3 predictions failed with:
```
unknown - 0022500761: Unknown error
unknown - 0022500762: Unknown error
...
```

### Root Cause
NBA.com CDN endpoints are returning `403 Forbidden` errors for all requests:
- ❌ `https://cdn.nba.com/static/json/liveData/boxscore/boxscore_*.json`
- ❌ `https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_*.json`
- ❌ `https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_*.json` (NOW BLOCKED)

### Investigation
Tested different request cadences:
- **0s delay:** All 403 errors ❌
- **10s delay:** All 403 errors ❌
- **No amount of delay helps** - this is a hard IP block, not rate limiting

### Working vs. Blocked Endpoints

| Endpoint | Status | Used By |
|----------|--------|-----------|
| Main NBA.com API (schedule) | ✅ Working | Pregame predictions |
| CDN boxscore endpoint | ❌ Blocked (403) | Halftime, Q3 predictions |
| CDN playbyplay endpoint | ❌ Blocked (403) | Halftime, Q3 predictions |
| CDN scoreboard endpoint | ✅ Was working, now blocked | Fallback for halftime data |

### Impact on Predictions

| Prediction Type | Requires CDN | Status |
|----------------|--------------|--------|
| Pregame | ❌ No (uses main API) | ✅ **WORKING** |
| Halftime | ✅ Yes (boxscore + playbyplay) | ❌ **NOT WORKING** |
| Q3 | ✅ Yes (boxscore + playbyplay) | ❌ **NOT WORKING** |

### Fixes Applied

#### 1. Improved Error Messages ✅
**Before:**
```
unknown - 0022500761: Unknown error
```

**After:**
```
0022500761: NBA.com API returned 403 Forbidden - rate limiting or access issue
```

**File:** `src/predict_api.py`

#### 2. Added Fallback to Scoreboard Endpoint ✅
Added `fetch_box_from_scoreboard()` function that:
- Tries scoreboard endpoints when boxscore returns 403
- Falls back gracefully with better error messages

**File:** `src/predict_from_gameid_v2.py`

#### 3. Enhanced Retry Logic ✅
- Increased max retries from 3 to 5
- Increased backoff: 2s → 4s → 8s → 16s (was 1s → 2s → 4s)
- Added disk-based caching (5 min TTL) to avoid redundant requests

**File:** `src/predict_from_gameid_v2.py`

### Current Status
**CDN is completely blocked.** All retry strategies have failed. This is likely:
- **Hard IP block** (rate limiting that doesn't reset)
- **User-agent detection** (NBA.com detected automated requests)
- **Geographic blocking** (NBA.com blocking requests from this region)

### Recommended Solutions

#### Option 1: Wait for Block to Lift ⏰
- **Pros:** No code changes needed
- **Cons:** Could last indefinitely (days/weeks)
- **Likelihood:** Low success rate

#### Option 2: Focus on Pregame Only 🎯
- **Pros:** Fully functional now, provides value
- **Cons:** Missing in-game updates (halftime/Q3)
- **Recommendation:** **DO THIS NOW** - use what works

#### Option 3: Use a Proxy/VPN 🌐
- **Pros:** Might work immediately
- **Cons:** Complex setup, ongoing cost/maintenance, potential TOS issues
- **Effort:** High

#### Option 4: Use Third-Party Sports API 💰
- **Examples:** Sportradar, Stats Perform, etc.
- **Pros:** Reliable, no rate limiting
- **Cons:** Expensive ($$$), complex integration
- **Recommendation:** Long-term solution, not quick fix

#### Option 5: Implement Simplified Halftime Predictor 🔧
- **Approach:** Use main API (not CDN) for basic game data
- **Pros:** Works when CDN blocked, lower complexity
- **Cons:** Less accurate (missing play-by-play behavior data)
- **Effort:** Medium
- **Feasibility:** **HIGH** - main API works, just need to adapt model

---

## Issue 2: Full Day Automation Triggers Immediately ⚠️

### Problem
Full day automation is triggering halftime and Q3 predictions **immediately** when set up, instead of waiting for games to reach those states.

### Expected Behavior
According to docstring in `run_full_day_automation()`:
```
3. Halftime triggers for each game (game-time aware, auto-posts at halftime)
4. Q3 triggers for each game (game-time aware, auto-posts at Q3)
```

### Actual Behavior
- Calls `run_prediction()` immediately for all games
- Tries to generate predictions right away
- Predictions fail because:
  1. Games aren't at halftime/Q3 yet
  2. CDN is blocked (Issue #1)

### Root Cause Analysis

The codebase has trigger infrastructure built but **not integrated**:

**Components That Exist:**
- ✅ `TriggerEngine` - evaluates game state and fires triggers
- ✅ `GameStateMonitor` - monitors games in real-time
- ✅ `AutoQueueProcessor` - processes posts

**Current Implementation:**
- ❌ `run_full_day_automation()` calls `run_prediction()` directly
- ❌ Doesn't use TriggerEngine or GameStateMonitor
- ❌ Generates predictions immediately regardless of game state

### Code Comparison

**Expected Flow:**
```
run_full_day_automation()
  └─> TriggerEngine.register_games([game_ids])
      └─> GameStateMonitor.start() [background process]
          └─> (waits for game to reach halftime)
              └─> TriggerEngine.evaluate_game(game_id, game_state)
                  └─> predict_game(mode='halftime') [generates prediction]
                      └─> SocialMediaManager.queue() [queues post]
                          └─> AutoQueueProcessor.process() [posts at right time]
```

**Actual Flow:**
```
run_full_day_automation()
  └─> run_prediction(game_id, trigger_type='halftime') [IMMEDIATE]
      └─> predict_game(mode='halftime')
          └─> fetch_box() [fails - CDN blocked]
              └─> Error: "NBA.com API returned 403 Forbidden"
```

### Recommended Fix

#### Option 1: Integrate TriggerEngine (Proper Fix) 🔧
**Steps:**
1. Initialize TriggerEngine and GameStateMonitor in `run_full_day_automation()`
2. Register all game IDs with trigger engine
3. Start background monitoring process
4. Let trigger engine fire predictions when game reaches right state

**Pros:** Correct implementation, real-time triggers
**Cons:** Significant development work
**Effort:** High (2-4 hours)

#### Option 2: Skip Halftime/Q3 in Full Day (Temporary) 🚫
**Steps:**
1. Only run pregame predictions in full day automation
2. Document that halftime/Q3 require manual setup

**Pros:** Quick fix, eliminates errors
**Cons:** Loses in-game functionality
**Effort:** Low (5 minutes)

**Recommendation:** **DO THIS FIRST**, then implement Option 1

---

## Issue 3: Preview Table Missing Team Scores ✅ FIXED

### Problem
Full day preview table only showed:
- Game total
- Margin (home - away)

**Missing:**
- Individual team scores (predicted)

### Fix Applied

Updated `_generate_discord_full_slate()` in `src/automation/post_generator.py`:

**Added:**
- New `🏀 Score` column showing predicted scores for both teams
- Calculate: `home_score = (total + margin) / 2`
- Calculate: `away_score = (total - margin) / 2`
- Format as: `away_score - home_score` (e.g., `110.3-105.2`)

**Before:**
```
| # | Away → Home | 🏆 Winner | 📈 Prob | 🎯 Total | ➕ Margin |
```

**After:**
```
| # | Away → Home | 🏆 Winner | 📈 Prob | 🏀 Score | 🎯 Total | ➕ Margin |
```

### Example Output

| # | Away → Home | 🏆 Winner | 📈 Prob | 🏀 Score | 🎯 Total | ➕ Margin |
|---|-------------|------------|---------|----------|-----------|-----------|
| 1 | NYK → BOS | **NYK** | 55.0% | 110.3-105.2 | 215.5 | -5.2 |
| 2 | GSW → LAL | **LAL** | 58.0% | 113.8-116.6 | 230.3 | +2.8 |
| 3 | TOR → MIA | **TOR** | 51.0% | 110.0-108.7 | 218.7 | -1.3 |

**Status:** ✅ **FIXED AND DEPLOYED**

**Files Modified:**
- `src/automation/post_generator.py`

---

## Summary & Recommendations

### What's Working ✅
1. **Pregame predictions** - Fully functional
2. **Error messages** - Now show helpful information
3. **Preview table** - Includes team scores
4. **Caching** - Reduces redundant API calls

### What's Not Working ❌
1. **Halftime predictions** - CDN blocked, automation triggers wrong
2. **Q3 predictions** - CDN blocked, automation triggers wrong
3. **Full day automation** - Triggers immediately instead of waiting

### Immediate Actions (Recommended Priority)

#### Priority 1: Focus on Pregame Only 🎯
**Action:** Only run pregame predictions until CDN block lifts
- Full day automation: Skip halftime/Q3
- Individual predictions: Only use pregame mode
- Users still get valuable pregame insights

**Effort:** 5 minutes
**Impact:** Eliminates all "Unknown error" messages

#### Priority 2: Fix Automation Trigger Logic 🔧
**Action:** Integrate TriggerEngine into `run_full_day_automation()`
- Register games with trigger engine
- Start background monitoring
- Let predictions fire at correct game states

**Effort:** 2-4 hours
**Impact:** Halftime/Q3 will work once CDN is unblocked

#### Priority 3: Solve CDN Blocking 🌐
**Options:**
1. **Wait** - Block might lift in 1-24 hours (if rate limiting)
2. **Proxy** - Use different IP address
3. **Third-party API** - Long-term reliable solution (Sportradar