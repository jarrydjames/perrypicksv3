# Season 2025-26 Data Pull Guide

## Critical: Season Configuration

The 2025-26 season is **IN PROGRESS** as of Feb 1, 2026. All data fetching operations MUST use `season='2025-26'`.

## Season Format

The NBA API uses `YYYY-YY` format for season:
- 2025-26 season: Use `season='2025-26'`
- This covers all regular season games from October 2025 through April 2026

## Data Fetching Examples

### 1. Team Game Logs

```python
from nba_api.stats.endpoints import teamgamelog

# CORRECT - Season 2025-26
gamelog = teamgamelog.TeamGameLog(
    team_id=1610612760,  # OKC
    season='2025-26'     # ← USE THIS FORMAT
)

# INCORRECT - Will use wrong season or fail
# gamelog = teamgamelog.TeamGameLog(
#     team_id=1610612760,
#     season='2025-2026'  # Wrong format
# )
```

### 2. Team Stats (Advanced)

```python
from nba_api.stats.endpoints import leaguedashteamstats

# CORRECT - Season 2025-26
stats = leaguedashteamstats.LeagueDashTeamStats(
    team_id_nullable=1610612760,
    season='2025-26',                    # ← Season format
    measure_type_detailed_defense='Advanced',
    per_mode_detailed='PerGame'
)

# Available columns in Advanced mode:
# - PACE, OFF_RATING, DEF_RATING, EFG_PCT, FTA_RATE, OREB_PCT, TOV_PCT, W, L, GP
```

### 3. League Schedule

```python
from nba_api.stats.endpoints import leaguegamefinder

# CORRECT - Season 2025-26
gamefinder = leaguegamefinder.LeagueGameFinder(
    league_id_nullable='00',
    season_nullable='2025-26',           # ← Season format
    season_type_nullable='Regular Season'
)
```

### 4. Box Score Data

```python
from nba_api.stats.endpoints import boxscoretraditionalv2

# Box scores don't require season parameter - they use game_id
boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(
    game_id='0022500711'
)
```

## Common Column Names

### Advanced Stats Columns
```python
# Team Stats (Advanced mode)
columns = [
    'TEAM_ID', 'TEAM_NAME', 'GP', 'W', 'L', 'W_PCT', 'MIN',
    'E_OFF_RATING', 'OFF_RATING',      # Use OFF_RATING
    'E_DEF_RATING', 'DEF_RATING',      # Use DEF_RATING
    'NET_RATING', 'AST_PCT', 'AST_TO',
    'AST_RATIO', 'OREB_PCT', 'DREB_PCT',
    'REB_PCT', 'TM_TOV_PCT', 'EFG_PCT',
    'TS_PCT', 'PACE', 'PIE', 'POSS'
]
```

### Base Stats Columns
```python
# Team Stats (Base mode)
columns = [
    'TEAM_ID', 'TEAM_NAME', 'GP', 'W', 'L', 'W_PCT', 'MIN',
    'FGM', 'FGA', 'FG_PCT',
    'FG3M', 'FG3A', 'FG3_PCT',
    'FTM', 'FTA', 'FT_PCT',
    'OREB', 'DREB', 'REB',
    'AST', 'TOV', 'STL', 'BLK',
    'BLKA', 'PF', 'PFD',
    'PTS', 'PLUS_MINUS'
]
```

### Game Log Columns
```python
# TeamGameLog columns
columns = [
    'Team_ID', 'Game_ID', 'GAME_DATE', 'MATCHUP', 'WL',
    'MIN', 'FGM', 'FGA', 'FG_PCT',
    'FG3M', 'FG3A', 'FG3_PCT',
    'FTM', 'FTA', 'FT_PCT',
    'OREB', 'DREB', 'REB',
    'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS'
]
```

## Matchup Parsing

The NBA API returns matchups in two formats:

### Format 1: Away @ Home
```
"CHI @ MIA"
```
- Away team: CHI
- Home team: MIA

### Format 2: Home vs. Away
```
"MIA vs. CHI"
```
- Home team: MIA
- Away team: CHI

### Parse Function
```python
def parse_matchup(matchup):
    """
    Parse matchup string to extract home and away teams.
    
    Formats:
    - "CHI @ MIA" → Away=CHI, Home=MIA
    - "MIA vs. CHI" → Home=MIA, Away=CHI
    """
    matchup = matchup.strip()
    
    # Check for " @ " format (Away @ Home)
    if ' @ ' in matchup:
        parts = matchup.split(' @ ')
        if len(parts) == 2:
            away = parts[0].strip()
            home = parts[1].strip()
            return home, away
    
    # Check for " vs. " format (Home vs Away)
    if ' vs. ' in matchup:
        parts = matchup.split(' vs. ')
        if len(parts) == 2:
            home = parts[0].strip()
            away = parts[1].strip()
            return home, away
    
    # Check for " vs " format (no dot)
    if ' vs ' in matchup:
        parts = matchup.split(' vs ')
        if len(parts) == 2:
            home = parts[0].strip()
            away = parts[1].strip()
            return home, away
    
    logger.warning(f"Could not parse matchup: {matchup}")
    return 'Unknown', 'Unknown'
```

## Team ID Mapping

```python
TEAM_IDS = {
    'ATL': 1610612737, 'BOS': 1610612738, 'CLE': 1610612739, 'NOP': 1610612740,
    'CHI': 1610612741, 'DAL': 1610612742, 'DEN': 1610612743, 'GSW': 1610612744,
    'HOU': 1610612745, 'LAC': 1610612746, 'LAL': 1610612747, 'MIA': 1610612748,
    'MIL': 1610612749, 'MIN': 1610612750, 'BKN': 1610612751, 'NYK': 1610612752,
    'ORL': 1610612753, 'IND': 1610612754, 'PHI': 1610612755, 'PHX': 1610612756,
    'POR': 1610612757, 'SAC': 1610612758, 'SAS': 1610612759, 'OKC': 1610612760,
    'TOR': 1610612761, 'WAS': 1610612762, 'MEM': 1610612763, 'UTA': 1610612764,
    'DET': 1610612765, 'CHA': 1610612766
}
```

## Best Practices

### 1. Always Use SEASON Constant
```python
# Define at top of file
SEASON = '2025-26'

# Use everywhere
gamelog = TeamGameLog(team_id=OKC_ID, season=SEASON)
stats = LeagueDashTeamStats(team_id_nullable=OKC_ID, season=SEASON)
```

### 2. Check for Empty Results
```python
df = stats.get_data_frames()[0]

if len(df) == 0:
    logger.warning(f"No data found for {team_name}")
    return None
```

### 3. Handle Missing Columns
```python
# Use .get() with defaults for safety
pace = team_stats.get('PACE', 100.0)
off_rating = team_stats.get('OFF_RATING', 110.0)
win_pct = team_stats.get('W', 0) / team_stats.get('GP', 1)
```

### 4. Deduplicate Game Lists
```python
# LeagueGameFinder returns both home and away versions
df = df.drop_duplicates(subset=['GAME_ID'], keep='first')
```

## Complete Example: Pregame Prediction

```python
"""
Complete example of fetching 2025-26 data and making a prediction
"""
import logging
import pandas as pd
from nba_api.stats.endpoints import (
    leaguedashteamstats,
    teamgamelog,
    leaguegamefinder
)

# SEASON CONSTANT - ALWAYS USE THIS
SEASON = '2025-26'

def fetch_team_stats(team_id, team_name):
    """Fetch current season stats for a team."""
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(
            team_id_nullable=team_id,
            season=SEASON,  # ← Season 2025-26
            measure_type_detailed_defense='Advanced',
            per_mode_detailed='PerGame'
        )
        df = stats.get_data_frames()[0]
        
        if len(df) == 0:
            return None
        
        return df.iloc[0]
    except Exception as e:
        logging.error(f"Error fetching {team_name}: {e}")
        return None

def fetch_recent_games(team_id, team_name, n=10):
    """Fetch recent games for a team."""
    try:
        gamelog = teamgamelog.TeamGameLog(
            team_id=team_id,
            season=SEASON  # ← Season 2025-26
        )
        df = gamelog.get_data_frames()[0]
        
        if len(df) == 0:
            return None
        
        return df.head(n)
    except Exception as e:
        logging.error(f"Error fetching games for {team_name}: {e}")
        return None

def get_todays_games():
    """Get today's games from 2025-26 season."""
    try:
        gamefinder = leaguegamefinder.LeagueGameFinder(
            league_id_nullable='00',
            season_nullable=SEASON,  # ← Season 2025-26
            season_type_nullable='Regular Season'
        )
        df = gamefinder.get_data_frames()[0]
        
        # Filter for today's date
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        today = pd.Timestamp('2026-02-01').date()
        todays_games = df[df['GAME_DATE'].dt.date == today]
        
        # Deduplicate
        todays_games = todays_games.drop_duplicates(
            subset=['GAME_ID'], keep='first'
        )
        
        return todays_games
    except Exception as e:
        logging.error(f"Error fetching games: {e}")
        return pd.DataFrame()
```

## Troubleshooting

### Problem: No data returned
**Solution:** Check season format
```python
# Wrong
season='2025-2026'

# Correct
season='2025-26'
```

### Problem: Column not found
**Solution:** Use correct measure type
```python
# PACE, OFF_RATING, DEF_RATING are in Advanced mode only
measure_type_detailed_defense='Advanced'
```

### Problem: Duplicate games
**Solution:** Deduplicate by GAME_ID
```python
df = df.drop_duplicates(subset=['GAME_ID'], keep='first')
```

## Season Timeline (2025-26)

- **Start:** October 2025
- **Current:** February 1, 2026 (Season in progress)
- **End:** April 2026
- **Playoffs:** April-June 2026

## Quick Reference

| Task | Endpoint | Season Format |
|------|----------|---------------|
| Team Game Log | `TeamGameLog` | `season='2025-26'` |
| Team Stats | `LeagueDashTeamStats` | `season='2025-26'` |
| League Schedule | `LeagueGameFinder` | `season_nullable='2025-26'` |
| Box Score | `BoxScoreTraditionalV2` | Uses game_id only |

---

**Remember:** Always use `season='2025-26'` for the 2025-26 NBA season data!
