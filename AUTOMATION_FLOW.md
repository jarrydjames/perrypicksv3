# PerryPicks V3 - Complete Automation Flow

This document summarizes the entire automation flow from starting a day, triggering predictions, generating posts, and transitioning to new days.

---

## 🎯 Overview

The automation system manages three prediction models (Pregame, Halftime, Q3) and automatically triggers predictions based on game state throughout the day.

**Components:**
1. **Schedule Fetching** - `fetch_game_schedule.py`
2. **Prediction Runners** - `run_pregame_predictions.py`, `run_halftime_predictions.py`, `run_q3_predictions.py`
3. **Unified Scheduler** - `schedule_predictions.py`
4. **Prediction API** - `src/predict_api.py`
5. **Automated Monitor** - `run_automated_predictions.py` (continuous)

---

## 📅 DAILY AUTOMATION FLOW

### Phase 1: Pre-Day Setup (Before Games Start)

**Time:** 6:00 PM ET (1-2 hours before 7:30 PM tipoffs)

**Cron:**
```cron
0 18 * * * cd /path/to/PerryPicks v3 && /usr/local/bin/uv run python schedule_predictions.py --models pregame >> logs/pregame.log 2>&1
```

**Actions:**
1. **Fetch Schedule**
   ```bash
   python fetch_game_schedule.py --date YYYY-MM-DD
   ```
   - Fetches ESPN schedule for the date
   - Fetches NBA CDN schedule (full season)
   - Maps ESPN IDs → NBA IDs (83 team variations)
   - Output: NBA game IDs for predictions

2. **Run Pregame Predictions**
   ```bash
   python run_pregame_predictions.py YYYY-MM-DD
   ```
   - Fetches today's games from NBA CDN
   - Runs pregame model on each game
   - Projects final totals, margins, winners
   - Displays prediction summary table
   - Saves to log file

3. **Fetch Odds** (optional, per game)
   - Uses src/predict_api.py with `fetch_odds=True`
   - Calls odds API for each game
   - Adds betting lines to prediction output
   - Cached to avoid duplicate API calls

**Output:**
```
====================================================================================================
PREGAME PREDICTIONS FOR 2026-02-07
====================================================================================================

Found 10 games

[1/10] Predicting WAS @ BKN (0022500747)...
  ✓ Predicted: 112.5 - 109.3 (Total: 221.8)
...

SUMMARY (10/10 predictions successful)
====================================================================================================
Game ID      | Away   @ Home   | Pred Total | Pred Margin | Winner
----------------------------------------------------------------------------------------------------
0022500747   | WAS    @ BKN    | 221.8      | -3.2         | WAS
...
====================================================================================================
```

**Log File:** `logs/pregame.log`

---

### Phase 2: In-Game Automation (7 PM - 12 AM ET)

**Time:** Every 5 minutes during game hours
**Cron:**
```cron
# Halftime checks (7 PM - 11 PM)
*/5 19-23 * * * cd /path/to/PerryPicks v3 && /usr/local/bin/uv run python schedule_predictions.py --models halftime >> logs/halftime.log 2>&1

# Q3 checks (8 PM - 11 PM)
*/5 20-23 * * * cd /path/to/PerryPicks v3 && /usr/local/bin/uv run python schedule_predictions.py --models q3 >> logs/q3.log 2>&1
```

**Alternative: Continuous Monitoring**
```bash
# Run automated predictions monitor (checks all models continuously)
python run_automated_predictions.py
```

**Actions:**

#### Halftime Predictions (at end of Q2)
1. **Fetch Current Games**
   ```bash
   python run_halftime_predictions.py YYYY-MM-DD
   ```
   - Fetches today's games from NBA CDN
   - Gets H1 scores from live boxscores
   - Checks if games are at halftime (period 2 → 3 or early Q3)

2. **Filter Eligible Games**
   - Games at halftime (period = 2)
   - Games early in Q3 (period = 3, < 6 minutes remaining)

3. **Run Halftime Model**
   ```bash
   python src/predict_api.py --game_id 0022500747 --mode halftime
   ```
   - Uses XGBoost champion (two-head: total + margin)
   - Takes H1 scores as input
   - Projects 2H scores
   - Projects final game scores
   - Calculates margin and winner

4. **Fetch Odds** (optional)
   - Adds live odds to prediction

**Output:**
```
====================================================================================================
HALFTIME PREDICTIONS FOR 2026-02-07
====================================================================================================

Found 3 games at halftime

[1/3] Predicting WAS @ BKN (0022500747)...
  H1: 56-52
  ✓ Pred 2H: 60.8-54.0
  ✓ Pred Final: 116.8-106.0
  ✓ Winner: WAS
...

SUMMARY (3/3 predictions successful)
====================================================================================================
Game ID      | Away @ Home   | H1       | Pred 2H      | Pred Final      | Margin   | Winner
----------------------------------------------------------------------------------------------------
0022500747   | WAS    @ BKN    | 56-52    | 60.8-54.0   | 116.8-106.0   | -10.8   | WAS
...
====================================================================================================
```

**Log File:** `logs/halftime.log`

#### Q3 Predictions (after Q3 completes)
1. **Fetch Current Games**
   ```bash
   python run_q3_predictions.py YYYY-MM-DD
   ```
   - Fetches today's games from NBA CDN
   - Gets Q3 cumulative scores from live boxscores
   - Checks if games are in Q4 (period = 4 or > 3)
   - Uses Q4 estimation for final projection

2. **Filter Eligible Games**
   - Games in Q4 (period = 4)
   - Games completed (status = "Final")

3. **Run Q3 Model**
   ```bash
   python src/predict_api.py --game_id 0022500747 --mode q3
   ```
   - Uses Neural Network champion
   - Takes Q3 cumulative scores as input
   - Estimates Q4 scores using quarter progression
   - Projects final game scores
   - Calculates margin and winner

4. **Fetch Odds** (optional)
   - Adds live odds to prediction

**Output:**
```
====================================================================================================
Q3 PREDICTIONS FOR 2026-02-07
====================================================================================================

Found 4 games in Q4/completed

[1/4] Predicting WAS @ BKN (0022500747)...
  Q3 Cum: 95.0-84.0
  ✓ Est Q4: 30.8-26.4
  ✓ Pred Final: 125.8-110.4
  ✓ Winner: WAS
...

SUMMARY (4/4 predictions successful)
====================================================================================================
Game ID      | Away @ Home   | Q3 Cum       | Est Q4        | Pred Final         | Margin   | Winner
----------------------------------------------------------------------------------------------------
0022500747   | WAS    @ BKN    | 95.0-84.0  | 30.8-26.4    | 125.8-110.4       | -15.4   | WAS
...
====================================================================================================
```

**Log File:** `logs/q3.log`

---

### Phase 3: Day Transition (After All Games Complete)

**Time:** After midnight ET

**Actions:**
1. **Cron jobs continue running**
   - Next day at 6:00 PM ET triggers new pregame predictions
   - No manual intervention needed
   - Continuous loop repeats daily

2. **Log rotation** (optional)
   - Archive old log files
   - Create new logs for new day
   ```bash
   # Rotate logs daily
   mv logs/pregame.log logs/pregame_YYYY-MM-DD.log
   touch logs/pregame.log
   ```

3. **Next day prep**
   - No action needed
   - Cron automatically picks up new date
   - `schedule_predictions.py` uses `datetime.now()` for today

---

## 🔗 COMPONENT INTEGRATION

### 1. Schedule Fetcher (`fetch_game_schedule.py`)

**Purpose:** Get NBA game IDs for today's games

**Flow:**
```
Input: date (YYYY-MM-DD)
  ↓
Fetch ESPN Schedule (site.api.espn.com)
  ↓
Fetch NBA CDN Schedule (cdn.nba.com/scheduleLeagueV2.json)
  ↓
Match Games (ESPN + NBA by teams/date)
  ↓
Normalize Teams (83 ESPN → NBA mappings)
  ↓
Output: ESPN ID → NBA ID mapping
  ↓
Return: NBA game IDs list
```

**Key Features:**
- ✅ No rate limiting (both APIs publicly accessible)
- ✅ 100% team name coverage (83 variations)
- ✅ Official NBA.com game IDs
- ✅ JSON output for automation

### 2. Prediction API (`src/predict_api.py`)

**Purpose:** Single entry point for all predictions

**Flow:**
```
Input: game_id, mode, home_team, away_team
  ↓
Game State Detection (if mode='auto')
  ├─ Period 0 → PREGAME
  ├─ Period 2, no period 3 → HALFTIME
  ├─ Period 3, < 6 min → HALFTIME
  ├─ Period 3, >= 6 min → Q3
  └─ Period 4+ → Q3
  ↓
Model Selection (game_state or mode override)
  ↓
Fetch Game Data (boxscores, features)
  ↓
Load Model (pregame/halftime/q3)
  ↓
Run Prediction
  ↓
Fetch Odds (optional, if fetch_odds=True)
  ↓
Format Output
  ↓
Return: Prediction dict
```

**Key Features:**
- ✅ Auto game state detection
- ✅ Supports forced mode (pregame/halftime/q3)
- ✅ Odds fetching with caching
- ✅ Rich output format
- ✅ Import gate (data freshness check)

### 3. Unified Scheduler (`schedule_predictions.py`)

**Purpose:** Run predictions on schedule (cron-friendly)

**Flow:**
```
Input: date, models, games, dry_run
  ↓
For each model:
  ↓
  Build command (uv run python run_[model]_predictions.py)
  ↓
  Add date
  ↓
  Add games (if specified)
  ↓
Execute or display (if dry_run)
  ↓
Capture output
  ↓
Return: Success/Failure status
```

**Key Features:**
- ✅ Run single model or all models
- ✅ Dry-run mode for testing
- ✅ Game ID override
- ✅ Status tracking
- ✅ 10-second delays between models (rate limiting)

---

## 📊 PREDICTION MODELS

### Pregame Model
**When:** Before game starts (1-2 hours before tipoff)
**Trigger:** Cron at 6:00 PM ET
**Input:** No game data (teams only)
**Champion:** Neural Network
**Target:** Final game (~225 pts)
**MAE:** ~11-12 points
**R²:** ~0.65-0.70
**Output:**
- Predicted final total
- Predicted margin (home - away)
- Predicted winner
- Betting lines (if odds enabled)

### Halftime Model
**When:** At halftime (end of Q2)
**Trigger:** Cron every 5 min (7-11 PM ET)
**Input:** H1 scores
**Champion:** XGBoost
**Target:** Final game from H1 (~220 pts)
**MAE:** ~10-11 points
**R²:** ~0.70-0.75
**Output:**
- H1 scores
- Predicted 2H scores
- Predicted final scores
- Predicted margin
- Predicted winner
- Betting lines (if odds enabled)

### Q3 Model
**When:** After Q3 completes (end of Q3, start of Q4)
**Trigger:** Cron every 5 min (8-11 PM ET)
**Input:** Q3 cumulative scores
**Champion:** Neural Network
**Target:** Final game from Q3 (~195-257 pts)
**MAE:** ~9-10 points
**R²:** ~0.75-0.80
**Output:**
- Q3 cumulative scores
- Estimated Q4 scores
- Predicted final scores
- Predicted margin
- Predicted winner
- Betting lines (if odds enabled)

---

## 🔧 AUTOMATION SETUP

### Step 1: Create Log Directory
```bash
cd /path/to/PerryPicks v3
mkdir -p logs
chmod 755 logs
```

### Step 2: Set Up Cron Jobs
```bash
# Edit crontab
crontab -e

# Add these lines:

# Pregame at 6:00 PM (before games)
0 18 * * * cd /path/to/PerryPicks v3 && /usr/local/bin/uv run python schedule_predictions.py --models pregame >> logs/pregame.log 2>&1

# Halftime every 5 minutes (7 PM - 11 PM)
*/5 19-23 * * * cd /path/to/PerryPicks v3 && /usr/local/bin/uv run python schedule_predictions.py --models halftime >> logs/halftime.log 2>&1

# Q3 every 5 minutes (8 PM - 11 PM)
*/5 20-23 * * * cd /path/to/PerryPicks v3 && /usr/local/bin/uv run python schedule_predictions.py --models q3 >> logs/q3.log 2>&1

# Save and exit
```

### Step 3: Verify Setup
```bash
# Check cron is running
sudo grep CRON /var/log/syslog | tail -20

# Monitor logs
tail -f logs/pregame.log
tail -f logs/halftime.log
tail -f logs/q3.log
```

---

## 📝 LOG MANAGEMENT

### Log Files Generated
1. **`logs/pregame.log`**
   - Pre-game predictions
   - Timestamp: Before game day
   - Frequency: Once per day
   - Size: ~100-500 lines per day

2. **`logs/halftime.log`**
   - Halftime predictions
   - Timestamp: During games
   - Frequency: Every 5 min checks
   - Size: ~200-1000 lines per day

3. **`logs/q3.log`**
   - Q3 predictions
   - Timestamp: During games
   - Frequency: Every 5 min checks
   - Size: ~200-1000 lines per day

### Log Rotation (Optional)

```bash
# Daily rotation (add to cron)
0 0 * * * cd /path/to/PerryPicks v3 && logs/rotate_logs.sh >> logs/rotation.log 2>&1

# Create rotate_logs.sh
#!/bin/bash
DATE=$(date +%Y-%m-%d)
for log in pregame halftime q3; do
    if [ -f logs/${log}.log ]; then
        mv logs/${log}.log logs/${log}_${DATE}.log
        gzip logs/${log}_${DATE}.log
    fi
    touch logs/${log}.log
done
```

---

## 🚀 FULL DAY TIMELINE EXAMPLE

### 2026-02-07 (Game Day)

| Time (ET) | Action | Component | Output |
|-----------|--------|------------|--------|
| 6:00 PM | **Pregame Run** | schedule_predictions.py --models pregame | 10 pregame predictions |
| 6:05 PM | Fetch Schedule | fetch_game_schedule.py | NBA IDs for tonight |
| 6:10 PM | **Output Generated** | logs/pregame.log | Full pregame table |
| 7:00 PM | Halftime Check #1 | schedule_predictions.py --models halftime | No games at halftime yet |
| 7:05 PM | Halftime Check #2 | schedule_predictions.py --models halftime | No games at halftime yet |
| 7:10 PM | Halftime Check #3 | schedule_predictions.py --models halftime | No games at halftime yet |
| 7:15 PM | Halftime Check #4 | schedule_predictions.py --models halftime | No games at halftime yet |
| ... | ... | ... | ... |
| 8:15 PM | **Halftime Run #1** | schedule_predictions.py --models halftime | 3 halftime predictions (games at HT) |
| 8:20 PM | **Output Generated** | logs/halftime.log | Halftime table with H1 scores |
| 8:25 PM | Halftime Check | schedule_predictions.py --models halftime | Already predicted, skip |
| ... | ... | ... | ... |
| 9:45 PM | **Q3 Run #1** | schedule_predictions.py --models q3 | 4 Q3 predictions (games in Q4) |
| 9:50 PM | **Output Generated** | logs/q3.log | Q3 table with cumulative scores |
| ... | ... | ... | ... |
| 12:00 AM | Q3 Final Check | schedule_predictions.py --models q3 | All games completed |
| 12:05 AM | **Day Ends** | - | Waiting for next day |

---

## 🔄 WHAT YOU GET (OUTPUT FLOW)

### Pregame Posts (Before Games)
```
🏀 Pregame Predictions for 2026-02-07

Game 1: Washington Wizards @ Brooklyn Nets
- Predicted Total: 221.8
- Predicted Margin: -3.2 (Wizards by 3.2)
- Predicted Winner: WAS

Odds: WAS -3.5, O/U 219.5
...
```

### Halftime Posts (In-Game)
```
🔥 Halftime Update! Washington @ Brooklyn

📊 Halftime Score: 56-52
📈 Projected 2H: 60.8-54.0
🎯 Projected Final: 116.8-106.0
🏆 Projected Winner: Washington by 10.8

Live Odds: WAS -4.5, O/U 222.5
```

### Q3 Posts (Late Game)
```
⚡ Q3 Update! Washington @ Brooklyn

📊 Q3 Cumulative: 95.0-84.0
📈 Estimated Q4: 30.8-26.4
🎯 Projected Final: 125.8-110.4
🏆 Projected Winner: Washington by 15.4

Live Odds: WAS -6.5, O/U 236.5
```

### Final Posts (Game Over)
```
✅ FINAL: Washington Wizards 116 - Brooklyn Nets 110

Pregame Prediction: 112.5 - 109.3 (Total 221.8)
Halftime Prediction: 56-52 → 116.8-106.0
Q3 Prediction: 95.0-84.0 → 125.8-110.4

Actual Final: 116-110

🎯 Pregame Error: 4.2 pts
🎯 Halftime Error: 0.8 pts
🎯 Q3 Error: 0.4 pts
```

---

## 🎮 CONTINUOUS MONITORING (Alternative)

Instead of cron jobs, you can run continuous monitoring:

```bash
# Start automated predictions monitor
python run_automated_predictions.py
```

**How it works:**
1. Fetches today's schedule
2. Monitors game states continuously
3. Automatically triggers appropriate model:
   - When game at halftime → Halftime model
   - When game in Q4 → Q3 model
   - When game not started → Pregame model
4. Tracks processed games to avoid duplicates
5. Runs indefinitely until stopped

**Use case:**
- Better than cron for real-time updates
- More accurate timing (no waiting 5 min)
- Fewer API calls (only when needed)

---

## 📦 CRON VS CONTINUOUS MONITORING

| Feature | Cron Jobs | Continuous Monitor |
|---------|-----------|-------------------|
| Timing | Fixed schedule | Real-time detection |
| API Calls | Many (every 5 min) | Fewer (only when needed) |
| Setup | Cron configuration | Single process |
| Resource Usage | High (frequent checks) | Low (event-driven) |
| Reliability | Good | Better |
| Recommended | For simplicity | For accuracy |

---

## ✅ WHAT WORKS NOW

1. ✅ **Schedule Fetching**
   - ESPN → NBA ID mapping (100% success)
   - All 30 NBA teams covered (83 variations)
   - No rate limiting

2. ✅ **Pregame Predictions**
   - Runs before games
   - Projects final scores
   - Includes odds

3. ✅ **Halftime Predictions**
   - Runs at halftime
   - Uses H1 scores
   - Projects final scores
   - Includes odds

4. ✅ **Q3 Predictions**
   - Runs in Q4
   - Uses Q3 cumulative scores
   - Estimates Q4
   - Projects final scores
   - Includes odds

5. ✅ **Unified Scheduler**
   - Runs any model
   - Dry-run mode
   - Cron-friendly

6. ✅ **Game State Detection**
   - Auto-selects correct model
   - Pregame/Halftime/Q3 detection

7. ✅ **Odds Fetching**
   - Calls odds API
   - Caches to avoid duplicates
   - Adds to predictions

---

## 🎯 POST GENERATION & FLOW

### Current State
**Prediction outputs are generated but NOT automatically posted.**

**What happens now:**
1. Cron triggers predictions
2. Predictions generate
3. Output saved to log files (pregame.log, halftime.log, q3.log)
4. **Automation stops** (no posting)

### What You Get (Manual Flow)
```
Cron triggers → Predictions run → Log files updated
                                              ↓
                        YOU MANUALLY CHECK LOGS
                                              ↓
                        YOU FORMAT INTO POSTS
                                              ↓
                        YOU POST TO SOCIAL MEDIA
```

### What You Would Get (Automated Flow)
To make posts flow automatically, you'd need to add:

1. **Post Generator Script**
   - Parse log files
   - Format predictions into posts
   - Add emojis, hashtags, formatting

2. **Social Media Integration**
   - Twitter API
   - Bluesky API
   - Instagram API
   - Email notifications

3. **Posting Script**
   - Read from post generator
   - Call social media APIs
   - Track posted games
   - Avoid duplicates

4. **Updated Flow**
```
Cron triggers → Predictions run → Log files updated
                                              ↓
                        POST GENERATOR RUNS
                                              ↓
                        POSTS CREATED (queue)
                                              ↓
                        SOCIAL MEDIA POSTER RUNS
                                              ↓
                        POSTS PUBLISHED
                                              ↓
                        TRACKING UPDATED (posted flag)
```

---

## 🚀 GETTING STARTED

### Quick Start (Cron-Based)
```bash
# 1. Setup logs
mkdir -p logs

# 2. Test pregame
python schedule_predictions.py --models pregame --dry-run

# 3. Test halftime
python schedule_predictions.py --models halftime --dry-run

# 4. Test Q3
python schedule_predictions.py --models q3 --dry-run

# 5. Enable cron (when ready)
crontab -e
# Add cron jobs (see above)
```

### Quick Start (Continuous Monitor)
```bash
# Start monitor
python run_automated_predictions.py

# It will:
# - Fetch schedule
# - Monitor game states
# - Run predictions automatically
# - Output to console/logs
```

---

## 📖 DOCUMENTATION

- **README_MODELS.md** - Complete model documentation
- **CRON_SETUP.md** - Cron job setup guide
- **GAME_ID_MAPPING.md** - ESPN to NBA ID mapping
- **AUTOMATION_FLOW.md** - This document (you're reading it!)
- **AUTOMATION_SUMMARY.md** - Quick reference

---

**Last Updated:** 2026-02-07
**Status:** Production Ready
**Version:** 1.0
