# 🏀 Prediction Checklist - BEFORE MAKING PREDICTIONS

**Created:** Feb 1, 2026  
**Purpose:** Prevent common mistakes when making NBA game predictions

---

## ✅ PRE-PREDICTION CHECKLIST

### 1. Season Configuration 🔴 CRITICAL

- [ ] **SEASON Constant Defined:** `SEASON = '2025-26'` at TOP of file
  - ❌ WRONG: `SEASON = '2025-2026'` 
  - ❌ WRONG: `season=2025.26`
  - ✅ CORRECT: `SEASON = '2025-26'`

- [ ] **Use SEASON Constant in ALL API calls:**
  ```python
  # ✅ CORRECT - Use constant everywhere
  gamelog = TeamGameLog(team_id=OKC_ID, season=SEASON)
  stats = LeagueDashTeamStats(team_id_nullable=OKC_ID, season=SEASON)
  gamefinder = LeagueGameFinder(season_nullable=SEASON)
  ```

- [ ] **Season Format:** Always use `YYYY-YY` format (hyphen, not dash/period)

---

### 2. Team ID Validation

- [ ] **Team IDs Defined:** All 30 teams in TEAM_IDS dictionary
- [ ] **Team IDs Match:** Verify team abbreviation maps to correct ID
  ```python
  TEAM_IDS = {
      'OKC': 1610612760,
      'DEN': 1610612743,
      # ... all 30 teams
  }
  ```
- [ ] **No Unknown Teams:** All team abbreviations must be in TEAM_IDS

---

### 3. Matchup Parsing

- [ ] **Handle Both Formats:**
  - Format 1: `"CHI @ MIA"` → Away=CHI, Home=MIA
  - Format 2: `"PHX vs. LAC"` → Home=PHX, Away=LAC
  
- [ ] **Parse Function:**
  ```python
  def parse_matchup(matchup):
      if ' @ ' in matchup:
          away, home = matchup.split(' @ ')
          return home.strip(), away.strip()
      elif ' vs. ' in matchup:
          home, away = matchup.split(' vs. ')
          return home.strip(), away.strip()
      # Return 'Unknown', 'Unknown' if can't parse
  ```

---

### 4. Data Fetching

- [ ] **Team Stats - Advanced Mode Required:**
  ```python
  stats = LeagueDashTeamStats(
      team_id_nullable=team_id,
      season=SEASON,  # ← MUST use constant
      measure_type_detailed_defense='Advanced',  # ← For PACE, OFF_RATING, DEF_RATING
      per_mode_detailed='PerGame'
  )
  ```

- [ ] **Recent Games - Last 10:**
  ```python
  gamelog = TeamGameLog(
      team_id=team_id,
      season=SEASON  # ← MUST use constant
  )
  ```

- [ ] **Empty Result Handling:**
  ```python
  df = stats.get_data_frames()[0]
  if len(df) == 0:
      logger.warning(f"No data for {team_name}")
      return None
  ```

---

### 5. Column Name Validation

- [ ] **Advanced Stats Columns:**
  - ✅ `PACE` (NOT in Base mode)
  - ✅ `OFF_RATING`
  - ✅ `DEF_RATING`
  - ✅ `EFG_PCT`
  - ✅ `OREB_PCT`
  - ✅ `TOV_PCT`
  - ✅ `W`, `L`, `GP`

- [ ] **Base Stats Columns:**
  - ✅ `FGM`, `FGA`, `FG_PCT`
  - ✅ `FG3M`, `FG3A`, `FG3_PCT`
  - ✅ `FTM`, `FTA`, `FT_PCT`
  - ✅ `PTS`, `REB`, `AST`, `STL`, `BLK`, `TOV`

- [ ] **Game Log Columns:**
  - ✅ `PTS` (for recent games average)
  - ✅ `WL` (for win percentage)

---

### 6. Game List Management

- [ ] **Deduplicate Games:** LeagueGameFinder returns both home and away versions
  ```python
  games = games.drop_duplicates(subset=['GAME_ID'], keep='first')
  ```

- [ ] **Filter by Date:**
  ```python
  df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
  todays_games = df[df['GAME_DATE'].dt.date == target_date]
  ```

---

## 🚨 COMMON MISTAKES TO AVOID

### Mistake 1: Wrong Season Format
```python
# ❌ WRONG
season='2025-2026'  # Uses wrong season!
season=2025.26       # Syntax error!

# ✅ CORRECT  
season='2025-26'
```

### Mistake 2: Missing Advanced Mode
```python
# ❌ WRONG - PACE, OFF_RATING, DEF_RATING not available!
stats = LeagueDashTeamStats(
    team_id_nullable=team_id,
    season=SEASON,
    measure_type_detailed_defense='Base'  # Wrong!
)

# ✅ CORRECT
stats = LeagueDashTeamStats(
    team_id_nullable=team_id,
    season=SEASON,
    measure_type_detailed_defense='Advanced'  # For PACE, OFF_RATING, DEF_RATING
)
```

### Mistake 3: Variable Name Conflicts
```python
# ❌ WRONG - Using result dict keys vs global variables
home_name = result.get("home_name")  # Returns STRING "OKC"
away_name = result.get("away_name")  # Returns STRING "DEN"
# Later code tries: home_name.get() → AttributeError!

# ✅ CORRECT - Use result.get() consistently
home_team = result.get("home_name")  
away_team = result.get("away_name")
```

### Mistake 4: Duplicate Games
```python
# ❌ WRONG - LeagueGameFinder returns duplicates
games = df  # 14 entries for 7 games (2 per game)

# ✅ CORRECT - Deduplicate
games = df.drop_duplicates(subset=['GAME_ID'], keep='first')  # 7 entries
```

### Mistake 5: Not Handling Missing Data
```python
# ❌ WRONG - Assumes data exists
pace = team_stats['PACE']  # KeyError if missing!

# ✅ CORRECT - Use .get() with defaults
pace = team_stats.get('PACE', 100.0)  # 100.0 if missing
```

---

## 📋 PREDICTION WORKFLOW

### Step 1: Get Games
```python
games = get_todays_games()
if not games:
    logger.error("No games found!")
    return
```

### Step 2: For Each Game
```python
for game in games:
    game_id = game['game_id']
    home_team = game['home_team']
    away_team = game['away_team']
    home_id = game['home_id']
    away_id = game['away_id']
```

### Step 3: Fetch Data
```python
# Season stats (Advanced mode)
home_stats = fetch_team_stats(home_id, home_team)
away_stats = fetch_team_stats(away_id, away_team)

# Recent games (last 10)
home_recent = fetch_recent_games(home_id, home_team, n=10)
away_recent = fetch_recent_games(away_id, away_team, n=10)
```

### Step 4: Calculate Features
```python
features = calculate_features(home_stats, away_stats, home_recent, away_recent)
```

### Step 5: Make Prediction
```python
prediction = make_prediction(features)
```

### Step 6: Store Results
```python
predictions.append({
    'game_id': game_id,
    'matchup': f"{away_team} @ {home_team}",
    'predicted_total': prediction['total'],
    'predicted_home_score': prediction['home_score'],
    'predicted_away_score': prediction['away_score'],
    'predicted_margin': prediction['margin'],
    'predicted_winner': prediction['winner'],
    'confidence': prediction['confidence']
})
```

---

## 📊 POST-PREDICTION CHECKLIST

### Output Validation
- [ ] All games have predictions
- [ ] No "Unknown" teams in output
- [ ] No NaN values in predictions
- [ ] Confidence scores between 0.5 and 0.85
- [ ] Total scores reasonable (180-260 range)
- [ ] Margins reasonable (-20 to +20 range)

### File Output
- [ ] Saved to `data/predictions/todays_predictions_YYYY-MM-DD.csv`
- [ ] CSV has all expected columns
- [ ] CSV is readable (no encoding issues)

### Logging
- [ ] All API calls logged
- [ ] All warnings/errors logged
- [ ] Prediction summary printed to console

---

## 🔄 QUICK REFERENCE

### Season Timeline
- **2025-26 Season:** October 2025 - April 2026
- **Current Date:** Feb 1, 2026
- **Status:** In Progress

### Team Count
- **Total Teams:** 30
- **Valid Team IDs:** 30/30

### Prediction Format
```
Matchup: AWAY @ HOME
Total: XXX.X ± 15.6
Home Score: XXX.X ± 7.8
Away Score: XXX.X ± 7.8
Margin: X.X ± 11.2
Winner: Home/Away
Confidence: 0.XX
```

---

## 🎯 FINAL CHECK

Before running predictions, verify:

- [ ] SEASON constant = '2025-26'
- [ ] All 30 teams in TEAM_IDS
- [ ] Matchup parsing handles both formats
- [ ] Advanced mode for team stats
- [ ] Deduplication enabled
- [ ] Default values for missing data
- [ ] Logging configured

**Total Checkboxes: ___ / 30**

---

## 📝 RELATED DOCUMENTATION

- `SEASON_DATA_GUIDE.md` - Complete NBA API data fetching guide
- `FINAL_REPORT.md` - Full system documentation
- `README.md` - Project overview

---

**Remember:** Check ALL boxes before running predictions! 🐶
