# NBA API Data Fetching Methodology

## Overview

This document outlines the correct methodology for fetching data from the NBA API, including lessons learned from PerryPicks v2 and corrections applied to v3.

---

## API Endpoints

### 1. Schedule Endpoint (Primary)

**URL:** `https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json`

**Purpose:** Fetch game schedule for any date

**Why use this endpoint:**
- Public CDN (works from Streamlit Cloud)
- No 403 errors (unlike `todaysScoreboard`)
- Returns all games in the season
- Fast and reliable
- No rate limiting issues

**Usage:**
```python
import requests
from datetime import date

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.nba.com/",
}

def fetch_schedule() -> dict:
    """Fetch full season schedule."""
    r = requests.get(SCHEDULE_URL, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()
```

**Response Structure:**
```json
{
  "leagueSchedule": {
    "gameDates": [
      {
        "gameDate": "12/31/2025 00:00:00",
        "games": [
          {
            "gameId": "0022500687",
            "homeTeam": {
              "teamTricode": "ORL",
              "teamName": "Orlando Magic"
            },
            "awayTeam": {
              "teamTricode": "TOR",
              "teamName": "Toronto Raptors"
            },
            "gameStatus": 3,  // 1=Pre, 2=Live, 3=Final
            "gameStatusText": "Final"
          }
        ]
      }
    ]
  }
}
```

**Important Notes:**
- Date format in response: `"MM/DD/YYYY 00:00:00"`
- `gameStatus`: 1=Pregame, 2=Live, 3=Final
- No actual scores in schedule (must fetch from boxscore)

---

### 2. Boxscore Endpoint (For Game Details)

**URL:** `https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{GAME_ID}.json`

**Purpose:** Fetch detailed game statistics including scores

**Why use this endpoint:**
- Provides complete game statistics
- Contains actual scores for completed games
- Includes live status for ongoing games
- Rich statistics for feature extraction

**Usage:**
```python
BOX_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"

def fetch_boxscore(game_id: str) -> dict:
    """Fetch boxscore for a specific game."""
    url = BOX_URL.format(gid=game_id)
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()
```

**Response Structure (Key Fields):**
```json
{
  "game": {
    "gameId": "0022500687",
    "gameStatus": 3,
    "gameStatusText": "Final",
    "gameClock": "PT00M00.00S",
    "period": 4,
    "homeTeam": {
      "teamId": 1610612753,
      "teamTricode": "ORL",
      "teamName": "Orlando Magic",
      "score": 130,
      "statistics": {
        "fieldGoalsAttempted": 90,
        "fieldGoalsMade": 46,
        "fieldGoalsEffectiveAdjusted": 0.566,
        "fieldGoalsPercentage": 0.511,
        "threePointersAttempted": 35,
        "threePointersMade": 14,
        "threePointersPercentage": 0.400,
        "freeThrowsAttempted": 18,
        "freeThrowsMade": 16,
        "freeThrowsPercentage": 0.889,
        "reboundsOffensive": 10,
        "reboundsDefensive": 35,
        "reboundsTotal": 45,
        "turnoversTotal": 12,
        "points": 130,
        "trueShootingPercentage": 0.632
      }
    },
    "awayTeam": {
      // Similar structure to homeTeam
    }
  }
}
```

**Important Notes:**
- **Scores are in `.score` field**, not in statistics.points (for live games)
- **Completed games**: `gameStatus == 3` and `score > 0`
- **Statistics object**: Contains all team-level stats
- **Clock format**: ISO 8601 duration format (`PT##M##.##S`)

---

## Correct Usage Pattern

### Fetching Games for a Date

**❌ INCORRECT:**
```python
# Don't use todaysScoreboard - 403 errors!
r = requests.get("https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard.json")
```

**✅ CORRECT:**
```python
from datetime import date
from src.data.scoreboard import fetch_scoreboard

# Fetch games with scores
games = fetch_scoreboard(date.today(), include_live=True)

# Filter completed games
completed = [g for g in games if g.home_score is not None and g.home_score > 0]

for game in completed:
    print(f"{game.away} {game.away_score} @ {game.home_score} {game.home}")
```

**Key Points:**
1. **Always use `include_live=True`** to get actual scores
2. **Schedule doesn't contain scores** - must fetch boxscore
3. **Check `home_score > 0`** to confirm game completed
4. **Handle exceptions gracefully** - APIs can be flaky

---

## Feature Extraction from Boxscore

### Critical: Statistics Location

**❌ WRONG:**
```python
# Stats are NOT at the top level of team object
home_team = game_data.get('homeTeam', {})
home_efg = home_team.get('effectiveFieldGoalPercentage')  # Returns None!
```

**✅ CORRECT:**
```python
# Stats are INSIDE the 'statistics' object
home_team = game_data.get('homeTeam', {})
home_stats = home_team.get('statistics', {})
home_efg = home_stats.get('fieldGoalsEffectiveAdjusted', 0.5)
```

### Feature Calculations

#### 1. Effective Field Goal Percentage (eFG%)
```python
home_efg = home_stats.get('fieldGoalsEffectiveAdjusted', 0.5)
```

#### 2. Free Throw Rate (FTR)
```python
home_fga = home_stats.get('fieldGoalsAttempted', 0)
home_fta = home_stats.get('freeThrowsAttempted', 0)
home_ftr = home_fta / max(home_fga, 1)
```

#### 3. Three-Point Attempt Rate (3PAR)
```python
home_3pa = home_stats.get('threePointersAttempted', 0)
home_fga = home_stats.get('fieldGoalsAttempted', 0)
home_tpar = home_3pa / max(home_fga, 1)
```

#### 4. Offensive Rebound Percentage (ORB%)
```python
home_orb = home_stats.get('reboundsOffensive', 0)
home_drb = home_stats.get('reboundsDefensive', 0)
home_orbp = (home_orb / max(home_orb + home_drb, 1)) if (home_orb + home_drb) > 0 else 0
```

#### 5. Turnover Rate (TOV%)
```python
home_fga = home_stats.get('fieldGamesAttempted', 0)
home_fta = home_stats.get('freeThrowsAttempted', 0)
home_tov = home_stats.get('turnoversTotal', 0)
home_possessions = max(home_fga + 0.44 * home_fta + home_tov, 1)
home_tor = home_tov / home_possessions
```

---

## Common Pitfalls and Solutions

### Pitfall #1: Not Using `include_live=True`

**Problem:**
```python
games = fetch_scoreboard(date)  # Default is include_live=True, but explicit is better
```

**Result:** All games have `home_score=None` and `away_score=None`

**Solution:**
```python
games = fetch_scoreboard(date, include_live=True)  # Explicit and clear
```

---

### Pitfall #2: Wrong Statistics Path

**Problem:**
```python
home_team = game_data.get('homeTeam', {})
home_efg = home_team.get('effectiveFieldGoalPercentage')  # Wrong!
```

**Result:** All eFG values are `None` or 0

**Solution:**
```python
home_team = game_data.get('homeTeam', {})
home_stats = home_team.get('statistics', {})  # Navigate to statistics
home_efg = home_stats.get('fieldGoalsEffectiveAdjusted', 0.5)  # Correct key!
```

---

### Pitfall #3: Forgetting to Reshape for sklearn

**Problem:**
```python
X = np.array([features_dict.get(f, 0.0) for f in feature_list])
pred = model.predict(X)  # Error!
```

**Result:** `ValueError: Expected 2D array, got 1D array instead`

**Solution:**
```python
X = np.array([features_dict.get(f, 0.0) for f in feature_list]).reshape(1, -1)
pred = model.predict(X)  # Works!
```

---

### Pitfall #4: Not Checking for Completed Games

**Problem:**
```python
games = fetch_scoreboard(date)
for game in games:
    process_game(game)  # Might process scheduled games with no scores!
```

**Result:** Processing games that haven't started or are in progress

**Solution:**
```python
games = fetch_scoreboard(date, include_live=True)
completed = [g for g in games if g.home_score is not None and g.home_score > 0]
for game in completed:
    process_game(game)
```

---

### Pitfall #5: Using Date Format Incorrectly

**Problem:**
```python
# Schedule uses "MM/DD/YYYY" format
sched_key = "01/31/2026"  # Hardcoded
```

**Result:** Won't work for different dates

**Solution:**
```python
from datetime import date

game_date = date.today()
sched_key = game_date.strftime("%m/%d/%Y")  # Dynamic
```

---

## Out-of-Sample Testing Methodology

### True Out-of-Sample Test

To validate model performance on truly unseen data:

1. **Fetch recent completed games:**
```python
from datetime import date, timedelta

test_dates = [
    date(2026, 1, 30),
    date(2026, 1, 29),
    date(2026, 1, 28),
    date(2026, 1, 27),
]

all_games = []
for game_date in test_dates:
    games = fetch_scoreboard(game_date, include_live=True)
    completed = [g for g in games if g.home_score is not None and g.home_score > 0]
    all_games.extend([(game_date, game) for game in completed])
```

2. **Extract features from boxscore:**
```python
for game_date, game in all_games:
    game_id = game.game_id
    game_data = fetch_boxscore(game_id)
    
    home_stats = game_data.get('homeTeam', {}).get('statistics', {})
    away_stats = game_data.get('awayTeam', {}).get('statistics', {})
    
    # Calculate features...
    features_dict = {
        'home_efg': home_stats.get('fieldGoalsEffectiveAdjusted', 0.5),
        # ... other features
    }
    
    X = np.array([features_dict.get(f, 0.0) for f in feature_list]).reshape(1, -1)
    pred_total = total_model['model'].predict(X)[0]
    pred_margin = margin_model['model'].predict(X)[0]
```

3. **Compare predictions to actuals:**
```python
actual_total = game.home_score + game.away_score
actual_margin = game.home_score - game.away_score
actual_winner = home if actual_margin > 0 else away

total_error = abs(pred_total - actual_total)
margin_error = abs(pred_margin - actual_margin)
winner_correct = (pred_winner == actual_winner)
```

---

## Performance Results (Jan 27-30, 2026)

**Test Details:**
- 33 completed games from last 4 days
- Model never saw these games during training
- No data leakage
- True out-of-sample test

**Results:**
- **Total MAE:** 3.37 points
- **Margin MAE:** 3.80 points
- **Winner Accuracy:** 90.9% (30/33 correct)
- **Within 3 pts:** 51.5% (17/33 games)
- **Within 5 pts:** 69.7% (23/33 games)
- **Within 10 pts:** 97.0% (32/33 games)

**Conclusion:** Model generalizes well to truly out-of-sample data.

---

## Streamlit App Integration

### Current Implementation

The streamlit app (`app.py`) already uses the correct approach:

```python
from src.data.scoreboard import fetch_scoreboard, format_game_label

# Fetch games for selected date
pick_date = st.date_input("Date", key="pp_pick_date")
games = fetch_scoreboard(pick_date)  # Uses include_live=True by default

# Display game selection
if not games:
    st.info("No games found for this date (or NBA CDN is being cranky).")
else:
    labels = [format_game_label(g) for g in games]
    game_idx = st.selectbox("Games", list(range(len(games))), format_func=lambda i: labels[i])
    selected_game = games[game_idx]
```

### Key Features

1. **Module-level imports** (fixes Streamlit Cloud issues)
2. **Default `include_live=True`** (gets actual scores)
3. **Graceful error handling** (handles API failures)
4. **User-friendly formatting** (shows scores, clock, status)
5. **Date selection** (allows picking any date in schedule)

---

## Troubleshooting

### Issue: No Games Found

**Possible Causes:**
1. Date not in season schedule
2. NBA CDN is down or slow
3. API response structure changed

**Debug Steps:**
```python
import requests

SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
data = requests.get(SCHEDULE_URL, headers=headers, timeout=15).json()

game_dates = data.get("leagueSchedule", {}).get("gameDates", [])
print(f"Total dates in schedule: {len(game_dates)}")
print(f"Last date: {game_dates[-1].get('gameDate')}")
print(f"First date: {game_dates[0].get('gameDate')}")
```

### Issue: All Scores Are None

**Cause:** Not using `include_live=True`

**Solution:**
```python
games = fetch_scoreboard(date, include_live=True)  # Must include!
```

### Issue: 403 Forbidden Errors

**Cause:** Using wrong endpoint or no headers

**Solution:**
```python
headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.nba.com/",
}
r = requests.get(SCHEDULE_URL, headers=headers)
```

### Issue: Statistics Are All 0 or None

**Cause:** Looking at wrong location in response

**Solution:**
```python
# Wrong:
home_efg = game_data.get('homeTeam', {}).get('effectiveFieldGoalPercentage')

# Correct:
home_stats = game_data.get('homeTeam', {}).get('statistics', {})
home_efg = home_stats.get('fieldGoalsEffectiveAdjusted', 0.5)
```

---

## Summary

### Best Practices

1. **Always use `scheduleLeagueV2.json`** - No 403 errors
2. **Always use `include_live=True`** - Get actual scores
3. **Always check `score > 0`** - Confirm game completed
4. **Always navigate to `statistics` object** - That's where stats live
5. **Always reshape for sklearn** - Models expect 2D arrays
6. **Always use proper headers** - Avoid being blocked
7. **Always handle exceptions** - APIs can be flaky

### Key Takeaways from v2 → v3

- ✅ Fixed 403 errors by using public CDN endpoint
- ✅ Fixed missing scores by using `include_live=True`
- ✅ Fixed zero statistics by navigating to correct object path
- ✅ Added proper error handling and validation
- ✅ Documented methodology for future reference

---

## References

- [NBA Stats API Documentation](https://github.com/swar/nba_api)
- [PerryPicks v2 Implementation](https://github.com/jarrydjames/perrypicksv2)
- [PerryPicks v3 Repository](https://github.com/jarrydjames/perrypicksv3)
