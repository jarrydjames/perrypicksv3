# PerryPicks V3 - Model Documentation & Usage Guide

This document provides comprehensive instructions for running all three prediction models: Pregame, Halftime, and Q3.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Model Overview](#model-overview)
3. [Pregame Predictions](#pregame-predictions)
4. [Halftime Predictions](#halftime-predictions)
5. [Q3 Predictions](#q3-predictions)
6. [Automation & Scheduling](#automation--scheduling)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# Run all models for today's games
python run_pregame_predictions.py
python run_halftime_predictions.py
python run_q3_predictions.py

# Run for specific date
python run_pregame_predictions.py 2026-02-05
python run_halftime_predictions.py 2026-02-05
python run_q3_predictions.py 2026-02-05

# Run for specific games (testing)
python run_pregame_predictions.py --games 0022500733 0022500734
python run_halftime_predictions.py --games 0022500733 0022500734
python run_q3_predictions.py --games 0022500733 0022500734
```

---

## Model Overview

| Model | Champion | Target | MAE | R² | When to Use |
|-------|-----------|--------|-----|----|-------------|
| **Pregame** | Neural Network | Final game | 9.58 Total, 2.95 Margin | 0.579 Total, 0.673 Margin | Before games start |
| **Halftime** | XGBoost | Final game from H1 | 7.92 H2 Total, 6.03 H2 Margin | 0.551 H2 Total, 0.536 H2 Margin | At halftime |
| **Q3** | Neural Network | Final game from Q3 | 8.34 Q3 Total, 6.58 Q3 Margin | 0.538 Q3 Total, 0.685 Q3 Margin | After Q3 |

---

## Pregame Predictions

### What It Predicts

- **Final game total**: Combined score for both teams (typically ~220-230 points)
- **Final margin**: Home team score minus away team score
- **Projected winner**: Team with positive margin prediction

### When to Run

- **Before games start**: 1-2 hours before tipoff
- **Daily batch**: Once per day for all scheduled games
- **Refresh**: Can be updated as odds change

### Command Syntax

```bash
python run_pregame_predictions.py [date] [--games GAME_ID [GAME_ID ...]]
```

### Parameters

- `date` (optional): Date in YYYY-MM-DD format (default: today)
- `--games`, `-g` (optional): Specific game IDs to predict (overrides date)

### Examples

```bash
# Today's games
python run_pregame_predictions.py

# Specific date
python run_pregame_predictions.py 2026-02-05

# Specific games (for testing)
python run_pregame_predictions.py --games 0022500733 0022500734 0022500735

# Get help
python run_pregame_predictions.py --help
```

### Output Format

```
====================================================================================================
PREGAME PREDICTIONS FOR 2026-02-05
====================================================================================================

Model: Neural Network Champion (R²: 0.579 Total, 0.673 Margin)
       (MAE: 9.580 Total, 2.950 Margin)
====================================================================================================

Found 12 games for 2026-02-05

[1/12] Processing WAS @ DET (0022500733)
  ✓ Total: 223.8
  ✓ Margin: -3.4
  ✓ Winner: WAS

...

====================================================================================================
PREGAME PREDICTIONS SUMMARY
====================================================================================================

Game ID      | Away   @ Home   | Predicted Total | Predicted Margin | Winner  
----------------------------------------------------------------------------------------------------
0022500733   | WAS    @ DET    | 223.8           | -3.4             | WAS     
0022500734   | BKN    @ ORL    | 215.6           | +12.1            | ORL     
...

====================================================================================================
Total games predicted: 12/12
Model: Neural Network (R²: 0.579 Total, 0.673 Margin)
====================================================================================================
```

### Model Features

- **10 efficiency stats**: efg, ftr, tpar, tor, orbp for both teams
- **Recent form**: Last 5 games performance
- **Rest days**: Days since last game for both teams
- **Back-to-back**: Whether either team is on b2b

---

## Halftime Predictions

### What It Predicts

- **First-half scores**: Actual scores from quarters 1 and 2 (fetched from boxscore)
- **Predicted 2nd half**: Estimated points in quarters 3 and 4
- **Predicted final**: Combined H1 + predicted 2H
- **Final margin**: Predicted home team advantage at end of game
- **Projected winner**: Team with positive final margin

### When to Run

- **At halftime**: When games reach halftime (end of Q2)
- **Batch mode**: Every 5-10 minutes during games to catch new halftime games
- **Manual**: On-demand for specific games

### Command Syntax

```bash
python run_halftime_predictions.py [date] [--games GAME_ID [GAME_ID ...]]
```

### Parameters

- `date` (optional): Date in YYYY-MM-DD format (default: today)
- `--games`, `-g` (optional): Specific game IDs to predict (overrides date)

### Examples

```bash
# Today's games (catches games at halftime)
python run_halftime_predictions.py

# Specific date
python run_halftime_predictions.py 2026-02-05

# Specific games (testing)
python run_halftime_predictions.py --games 0022500733 0022500734 0022500735

# Get help
python run_halftime_predictions.py --help
```

### Output Format

```
====================================================================================================
HALFTIME PREDICTIONS FOR 2026-02-05
====================================================================================================

Model: XGBoost Champion (MAE: 7.920 H2 Total, 6.029 H2 Margin)
====================================================================================================

Found 12 games for 2026-02-05

[1/12] Processing WAS @ DET (0022500733)
  ✓ H1: 56-52
  ✓ Pred 2H: 60.8-54.0 (Total: 114.8)
  ✓ Pred Final: 116.8-106.0 (Total: 222.8)
  ✓ Margin: -6.8 | Winner: WAS

...

====================================================================================================
HALFTIME PREDICTIONS SUMMARY
====================================================================================================

Game ID      | Away   @ Home   | H1         | Pred 2H     | Pred Final      | Margin   | Winner  
----------------------------------------------------------------------------------------------------
0022500733   | WAS    @ DET    | 56-52      | 60.8-54.0   | 116.8-106.0     | -6.8     | WAS     
0022500734   | BKN    @ ORL    | 40-56      | 45.8-64.3   | 85.8-120.3      | +18.5    | ORL     
...

====================================================================================================
Total games predicted: 12/12
Model: XGBoost (MAE: 7.920 H2 Total, 6.029 H2 Margin)
====================================================================================================
```

### Model Features

- **12 H1 features**: H1 scores, totals, margins, events
- **Efficiency stats**: 10 team efficiency stats (efg, ftr, tpar, tor, orbp)
- **H1-specific**: H1 events, 2pt/3pt attempts, turnovers, rebounds, fouls

---

## Q3 Predictions

### What It Predicts

- **Q3 cumulative scores**: Actual scores after quarters 1, 2, and 3
- **Estimated Q4**: Predicted 4th quarter scores (using quarter progression heuristic)
- **Predicted final**: Q3 cumulative + estimated Q4
- **Final margin**: Predicted home team advantage at end of game
- **Projected winner**: Team with positive final margin

### When to Run

- **After Q3**: When games complete the 3rd quarter
- **Batch mode**: Every 5-10 minutes during games to catch new Q3 games
- **Manual**: On-demand for specific games

### Command Syntax

```bash
python run_q3_predictions.py [date] [--games GAME_ID [GAME_ID ...]]
```

### Parameters

- `date` (optional): Date in YYYY-MM-DD format (default: today)
- `--games`, `-g` (optional): Specific game IDs to predict (overrides date)

### Examples

```bash
# Today's games (catches games after Q3)
python run_q3_predictions.py

# Specific date
python run_q3_predictions.py 2026-02-05

# Specific games (testing)
python run_q3_predictions.py --games 0022500733 0022500734 0022500735

# Get help
python run_q3_predictions.py --help
```

### Output Format

```
====================================================================================================
Q3 PREDICTIONS FOR 2026-02-05
====================================================================================================

Model: Q3 Neural Network Champion (R²: 0.538 Q3 Total, 0.685 Q3 Margin)
       (MAE: 8.339 Q3 Total, 6.581 Q3 Margin)

Prediction Logic:
  1. Q3 model predicts Q3 cumulative scores (H1+H2+Q3)
  2. Estimates Q4 using Q3 cumulative totals and margin
  3. Projects final game scores, margins, and winners
====================================================================================================

Found 12 games for 2026-02-05

[1/12] Processing WAS @ DET (0022500733)
  ✓ Q3 Cumulative: 95.0-84.0
  ✓ Estimated Q4: 30.8-26.4 (Total: 57.3)
  ✓ Predicted Final: 125.8-110.4 (Total: 236.3)
  ✓ Final Margin: -15.4 | Winner: WAS

...

====================================================================================================
Q3 PREDICTIONS SUMMARY
====================================================================================================

Game ID      | Away   @ Home   | Q3 Cum       | Est Q4        | Pred Final         | Margin   | Winner  
----------------------------------------------------------------------------------------------------
0022500733   | WAS    @ DET    | 95.0-84.0    | 30.8-26.4     | 125.8-110.4        | -15.4    | WAS     
0022500734   | BKN    @ ORL    | 67.0-88.0    | 20.6-29.0     | 87.6-117.0         | +29.4    | ORL     
...

====================================================================================================
Total games predicted: 12/12
Model: Q3 Neural Network (R²: 0.538 Q3 Total, 0.685 Q3 Margin)
Prediction: Q3 cumulative + estimated Q4 (typical quarter progression)
====================================================================================================
```

### Model Features

- **10 efficiency stats**: efg, ftr, tpar, tor, orbp for both teams
- **Q3-specific**: Q3 totals, margins, events, 2pt/3pt attempts, turnovers, rebounds, fouls

---

## Automation & Scheduling

### Automated Scheduler

Use the `run_automated_predictions.py` script to automatically run the appropriate models based on game states:

```bash
# Run continuous monitoring (checks every 5 minutes)
python run_automated_predictions.py

# Check every 10 minutes
python run_automated_predictions.py --interval 600

# Specific date
python run_automated_predictions.py --date 2026-02-05
```

### Scheduling with Cron

For Linux/macOS systems, use cron to schedule automated runs:

```bash
# Edit crontab
crontab -e

# Add these lines:

# Run pregame predictions at 6:00 PM ET (for 7:30 PM games)
0 18 * * * cd /path/to/PerryPicks\ v3 && /usr/local/bin/uv run python run_pregame_predictions.py >> logs/pregame.log 2>&1

# Check for halftime games every 5 minutes during game hours
*/5 19-23 * * * cd /path/to/PerryPicks\ v3 && /usr/local/bin/uv run python run_halftime_predictions.py >> logs/halftime.log 2>&1

# Check for Q3 games every 5 minutes during game hours
*/5 19-23 * * * cd /path/to/PerryPicks\ v3 && /usr/local/bin/uv run python run_q3_predictions.py >> logs/q3.log 2>&1
```

### Game State Detection

The system uses NBA.com's scoreboard API to detect game states:

- **Pregame**: Games with status "Scheduled"
- **Halftime**: Games with period=3 or clock showing halftime
- **Q3**: Games with period=4 or clock showing Q3 completion

### Best Practices

1. **Rate Limiting**: Add 1-second delays between game requests to avoid API blocking
2. **Error Handling**: Scripts continue even if individual games fail
3. **Logging**: Save output to log files for debugging
4. **Backup**: Store prediction results in database or CSV for analysis

---

## Troubleshooting

### Common Issues

#### 1. "No games found for date"

**Cause**: No games scheduled or API rate-limited

**Solution**:
- Check if date has NBA games
- Wait a few minutes and retry (API rate limit)
- Verify internet connection

#### 2. "NBA.com API returned error (403/429)"

**Cause**: API rate limit or blocking

**Solution**:
- Wait 5-10 minutes
- Check if using consistent User-Agent header
- Reduce request frequency

#### 3. Predictions show "N/A"

**Cause**: Games not yet started or data unavailable

**Solution**:
- For halftime: Game must be at halftime (Q2 completed)
- For Q3: Game must be after Q3 (Q3 completed)
- For pregame: Game must be scheduled

#### 4. Import errors

**Cause**: Missing dependencies or virtual environment not activated

**Solution**:
```bash
# Install dependencies
uv pip install -r requirements.txt

# Run with uv
uv run python run_pregame_predictions.py
```

---

## API Endpoints

### NBA.com APIs

```
Scoreboard (game schedule):
https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00_YYYYMMDD.json

Boxscore (game data):
https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gameId}.json

Play-by-play (game events):
https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{gameId}.json
```

### ESPN API (Fallback)

```
Scoreboard:
https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD
```

---

## Model Files

### Model Locations

```
models_v3/
├── pregame/           # Pregame Neural Network models
│   ├── neural_network_pregame_total.joblib
│   └── neural_network_pregame_margin.joblib
├── halftime/          # Halftime XGBoost models
│   ├── xgboost_h2_total.joblib
│   └── xgboost_h2_margin.joblib
└── q3/              # Q3 Neural Network models
    ├── neural_network_q3_total.joblib
    └── neural_network_q3_margin.joblib
```

### Training Data

```
data/processed/
├── pregame_training_23_24.parquet      # Pregame training data
├── halftime_training_23_24.parquet     # Halftime training data
└── q3_team_v2.parquet                 # Q3 training data
```

---

## Contributing

When adding new models or features:

1. Follow existing naming conventions
2. Document model MAE and R² metrics
3. Update this README
4. Test on sample games before committing

---

## License

PerryPicks V3 - NBA Game Prediction System

---

## Contact

For questions or issues, please refer to the main README.md or open an issue on GitHub.