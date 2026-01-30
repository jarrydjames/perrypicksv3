# PerryPicks v3 - Automation Guide

## Overview

This guide explains the automated daily refresh pipeline for PerryPicks v3, which keeps temporal features and models up-to-date with the latest NBA game data.

## What Is Automated?

| Component | Status | Description |
|-----------|---------|-------------|
| **Game Fetching** | ✅ Automated | Pulls last N days of games from NBA CDN |
| **Temporal Features** | ✅ Automated | Calculates rolling statistics with time-aware filtering |
| **Data Merging** | ✅ Automated | Combines temporal features with halftime stats |
| **Model Training** | ✅ Automated | Retrains models with latest data |
| **Production Deployment** | ✅ Automated | Copies trained models to production directory |

## Files

| File | Purpose |
|------|---------|
| `src/fetch_today_games.py` | Fetches last N days of games from NBA CDN |
| `src/build_temporal_features.py` | Builds time-aware rolling statistics |
| `src/merge_temporal_halftime.py` | Merges temporal + halftime features |
| `src/train_halftime_model.py` | Trains model with temporal features |
| `scripts/daily_refresh.sh` | Orchestration script (one-command refresh) |
| `models_v3/production/` | Production model directory (deployed here) |

## How It Works

### Time-Aware Rolling Features

For each game, the system calculates rolling statistics using **ONLY** games that happened **BEFORE** that game.

**Example:**
```
Game on January 29, 2026:
  → Rolling features based on games from Jan 1-27, 2026
  → NOT including Jan 29, 2026 game (current game)
  → NOT including games from Jan 30+ (future games)
  → NOT including games from 24-25 season (too old)
```

### Daily Refresh Pipeline

```bash
./scripts/daily_refresh.sh

[1/5] Fetching today's games...
  → Pulls games from last 2 days (Jan 27-29, 2026)

[2/5] Updating temporal features...
  → Calculates rolling statistics (last 5, last 10 games)
  → Uses games from Aug 2, 2025 - Jan 28, 2026 (if days-filter=180)

[3/5] Merging temporal features with halftime stats...
  → Combines rolling features with first-half team statistics

[4/5] Retraining halftime model...
  → Trains model on updated dataset

[5/5] Deploying updated model to production...
  → Copies models to models_v3/production/
```

## Usage

### Run Daily Refresh (Default)

```bash
# Default: 180-day filter (current season focus)
./scripts/daily_refresh.sh
```

### Run With Custom Settings

```bash
# Custom date
./scripts/daily_refresh.sh --date "2026-01-29"

# Custom days filter
./scripts/daily_refresh.sh --days-filter 90

# Dry run (no actual execution)
./scripts/daily_refresh.sh --dry-run
```

### Run Individual Steps

```bash
# 1. Fetch games
python3 src/fetch_today_games.py --days-before 2 --days-after 1

# 2. Build temporal features
python3 src/build_temporal_features.py \
  --data-dir data/raw/box \
  --output-dir data/processed \
  --days-filter 180

# 3. Merge with halftime stats
python3 src/merge_temporal_halftime.py

# 4. Retrain model
python3 src/train_halftime_model.py \
  --dataset data/processed/halftime_with_temporal_features.parquet

# 5. Deploy to production
mkdir -p models_v3/production
cp models_v3/halftime/* models_v3/production/
```

## Scheduling

### Crontab Setup

Edit crontab:
```bash
crontab -e
```

Add these lines:
```cron
# Daily refresh at 8 AM (after all games from previous day)
0 8 * * * /Users/jarrydhawley/Desktop/Predictor/PerryPicks v3/scripts/daily_refresh.sh >> /tmp/perrypicks_refresh.log 2>&1

# Weekly refresh on Sundays at 10 AM (backup)
0 10 * * 0 /Users/jarrydhawley/Desktop/Predictor/PerryPicks v3/scripts/daily_refresh.sh --days-filter 180 >> /tmp/perrypicks_refresh.log 2>&1
```

### Schedule Options

| Frequency | Days Filter | Reason |
|-----------|-------------|---------|
| **Daily** | 180 days | Keeps model current with latest games |
| **Weekly** | 180 days | Balances freshness vs compute |
| **Monthly** | 180 days | For lower-compute environments |
| **Season Start** | 365 days | At beginning of new season, use full prior season |

## Troubleshooting

### Q: Why do I see games from 23-24 season in rolling features?
**A:** Apply `--days-filter 180` to only use last 6 months of games:
```bash
./scripts/daily_refresh.sh --days-filter 180
```

### Q: How do I get the most recent rolling features?
**A:** Run the daily refresh:
```bash
./scripts/daily_refresh.sh
```

### Q: Can I use different rolling windows?
**A:** Yes, modify `src/build_temporal_features.py` to use different window sizes.

### Q: What if a team has less than 5 games?
**A:** The system uses `--min-games 5` by default. Teams with fewer games get default values (0).

### Q: My model isn't improving after refresh. Why?
**A:** Check:
1. Are new games being fetched? Check `data/raw/todays_games.json`
2. Are temporal features populated? Check `data/processed/halftime_with_temporal_features.parquet`
3. Is the model using temporal features? Check `src/modeling/feature_columns.py`

### Q: How do I rollback to a previous model?
**A:** Keep model versions in `models_v3/halftime/`:
```bash
# List models
ls -lh models_v3/halftime/

# Copy previous model to production
cp models_v3/halftime/gbt_model_v1.joblib models_v3/production/
```

## Monitoring

### Check Last Refresh

```bash
# Check when models were last updated
ls -lh models_v3/production/

# Check dataset date range
python3 << 'EOF'
import pandas as pd

df = pd.read_parquet('data/processed/halftime_with_temporal_features.parquet')
print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
print(f"Games: {len(df)}")
EOF
```

### Check Temporal Features

```bash
python3 << 'EOF'
import pandas as pd

df = pd.read_parquet('data/processed/halftime_with_temporal_features.parquet')

# Check temporal features are populated
temporal_cols = ['home_pts_scored_avg_5', 'home_current_streak_5', 'home_days_since_last']
for col in temporal_cols:
    pct_null = df[col].isna().sum() / len(df) * 100
    print(f'{col}: {pct_null:.1f}% null')
EOF
```

### Check Cron Logs

```bash
# View last refresh
tail -50 /tmp/perrypicks_refresh.log

# View all crontab jobs
crontab -l
```

## Validation

### Validate Time-Aware Rolling Features

To verify temporal features are time-aware:

```python
import pandas as pd

df = pd.read_parquet('data/processed/games_with_temporal_features.parquet')

# Check one game
example_game = df.iloc[0]
print(f"Game date: {example_game['game_date']}")
print(f"Rolling features based on: games before {example_game['game_date']}")

# Verify: rolling features should reflect team form BEFORE this game
# NOT including this game or any later games
```

### Validate Dates

For a game on **January 29, 2026**, verify:

| Feature | Expected Dates |
|----------|----------------|
| `home_pts_scored_avg_5` | Jan 1-27, 2026 (last 5 games BEFORE Jan 29) |
| `home_current_streak_5` | Jan 1-27, 2026 |
| `home_days_since_last` | Jan 27, 2026 (if last game was Jan 27) |
| `home_is_back_to_back` | Jan 28, 2026 (if they played yesterday) |

### Validate Days Filter

With `--days-filter 180`:
- **Cutoff:** Aug 2, 2025 (180 days before Jan 29, 2026)
- **Games used:** Aug 2, 2025 - Jan 28, 2026
- **Games excluded:** Before Aug 2, 2025 (24-25 season)

## Advanced Usage

### Skip Steps

```bash
# Skip fetching (use cached games)
./scripts/daily_refresh.sh --skip-fetch

# Skip building (use existing temporal features)
./scripts/daily_refresh.sh --skip-build

# Skip merging (use existing merged dataset)
./scripts/daily_refresh.sh --skip-merge

# Skip training (use existing models)
./scripts/daily_refresh.sh --skip-train

# Skip deployment (models stay in training directory)
./scripts/daily_refresh.sh --skip-deploy
```

### Custom Model Training

```bash
# Train with different hyperparameters
python3 src/train_halftime_model.py \
  --dataset data/processed/halftime_with_temporal_features.parquet \
  --model-type gbt \
  --n-estimators 200 \
  --max-depth 8 \
  --learning-rate 0.03

# Train multiple models
python3 src/train_halftime_model.py \
  --model-type all \
  --include-xgb \
  --include-cat
```

## Summary

### What Was Built

- ✅ `src/fetch_today_games.py` - Fetches last N days of games
- ✅ `src/build_temporal_features.py` - Builds time-aware rolling statistics
- ✅ `src/merge_temporal_halftime.py` - Merges temporal + halftime features
- ✅ `src/train_halftime_model.py` - Trains model with temporal features
- ✅ `scripts/daily_refresh.sh` - One-command refresh orchestration
- ✅ `models_v3/production/` - Production model directory

### How to Use

1. **Quick start:** Run daily refresh
   ```bash
   ./scripts/daily_refresh.sh
   ```

2. **Automate:** Set up cron job
   ```bash
   # Daily at 8 AM
   0 8 * * * /path/to/PerryPicks v3/scripts/daily_refresh.sh
   ```

3. **Monitor:** Check logs and model files
   ```bash
   tail -50 /tmp/perrypicks_refresh.log
   ls -lh models_v3/production/
   ```

### Benefits

- ✅ Rolling features always use current season data
- ✅ Model learns from recent team performance
- ✅ One command to refresh everything
- ✅ Automatic updates every morning
- ✅ Production model directory for easy rollback

---

**Date:** January 29, 2026  
**Status:** AUTOMATION COMPLETE  
**Author:** Perry (Code Puppy)
