# Temporal Features - Refresh Mechanism

## Overview

The temporal features (`data/processed/games_with_temporal_features.parquet`) provide team-level rolling statistics that are time-aware - they only use games that happened BEFORE each prediction point.

## Time-Aware Rolling Features

**How it works:**

For each game, the system:
1. Sorts all games chronologically by team
2. For the current game, only looks at games that happened BEFORE this game
3. Calculates rolling statistics (last 5, last 10 games) on only those previous games

**Example:**
```
Game on Jan 30, 2025:
  → Rolling features based on games from Dec 2024 - Jan 29, 2025
  → NOT including games from Jan 31, 2025 onwards
  → NOT including games from 23-24 season (too old)
```

## Refresh Mechanism for Production

To keep temporal features current for the current NBA season:

### Option 1: Rebuild Entire Dataset (Recommended)

Run the temporal features builder with a days filter:

```bash
# Rebuild with last 180 days (6 months) of games
python3 src/build_temporal_features.py \
  --data-dir data/raw/box \
  --output-dir data/processed \
  --days-filter 180

# Rebuild with last 90 days (3 months)
python3 src/build_temporal_features.py \
  --data-dir data/raw/box \
  --output-dir data/processed \
  --days-filter 90

# Rebuild with last 60 days (2 months)
python3 src/build_temporal_features.py \
  --data-dir data/raw/box \
  --output-dir data/processed \
  --days-filter 60
```

**Why use days filter?**
- Ensures rolling features use ONLY recent games
- Prevents old season data from affecting current season predictions
- Adapts to current team form and roster changes

### Option 2: Live Refresh (Advanced)

For production predictions, implement a pipeline that:

1. **Fetch latest game data**
   ```python
   # Get games from last N days
   fetch_nba_games(days=180)
   ```

2. **Rebuild rolling features**
   ```python
   python3 src/build_temporal_features.py \
     --data-dir /path/to/new/games \
     --output-dir data/processed \
     --days-filter 180
   ```

3. **Train model on updated data**
   ```python
   python3 src/train_halftime_model.py \
     --dataset data/processed/games_with_temporal_features.parquet
   ```

4. **Deploy updated model**
   ```bash
   # Replace production model
   cp models_v3/halftime/gbt_twohead.joblib /path/to/production/
   ```

## Recommended Refresh Schedule

| Frequency | Days Filter | Reason |
|-----------|-------------|---------|
| **Daily** | 180 days | Keeps model current with latest games |
| **Weekly** | 180 days | Balances freshness vs compute |
| **Monthly** | 180 days | For lower-compute environments |
| **Season Start** | 365 days | At beginning of new season, use full prior season |

## Features Generated

### For Each Team:

**Last 5 Games:**
- `pts_scored_avg_5`: Average points scored
- `pts_allowed_avg_5`: Average points allowed
- `margin_avg_5`: Average point margin
- `total_avg_5`: Average total points
- `wins_5`: Number of wins in last 5
- `current_streak_5`: Current win/loss streak

**Last 10 Games:**
- `pts_scored_avg_10`: Average points scored
- `pts_allowed_avg_10`: Average points allowed
- `margin_avg_10`: Average point margin
- `total_avg_10`: Average total points
- `wins_10`: Number of wins in last 10

**Rest/Fatigue:**
- `days_since_last`: Days since previous game
- `is_back_to_back`: 1 if playing back-to-back, else 0

## Current Season Focus

**Current season (24-25):**
- Today: January 30, 2025
- Season started: October 2024
- Games so far: ~300-400 per team
- **Last 5 games** = Last 5 games from 24-25 season (Oct-Jan)
- **Last 10 games** = Last 10 games from 24-25 season

**Previous season (23-24):**
- Season ended: June 2024
- **Not used** in rolling features if we apply days filter

## Example: Updating for Today's Predictions

```bash
# 1. Fetch latest games (last 180 days)
# (This should include games from Aug 2024 - Jan 2025)

# 2. Rebuild temporal features with days filter
python3 src/build_temporal_features.py \
  --data-dir data/raw/box \
  --output-dir data/processed \
  --days-filter 180

# 3. Merge with halftime features
python3 src/merge_temporal_halftime.py

# 4. Retrain model
python3 src/train_halftime_model.py

# 5. Deploy to production
cp models_v3/halftime/* /path/to/production/
```

## Files

- `src/build_temporal_features.py` - Build rolling features from raw data
- `data/processed/rolling_features.parquet` - Team-level rolling statistics
- `data/processed/games_with_temporal_features.parquet` - Game-level data with temporal features

## Validation

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

## Troubleshooting

**Q: Why do I see games from 23-24 season in rolling features?**
A: Apply `--days-filter 180` to only use last 6 months of games.

**Q: How do I get the most recent rolling features?**
A: Rebuild the dataset with current data and use `--days-filter 180`.

**Q: Can I use different rolling windows?**
A: Yes, modify `calculate_rolling_features()` to use different window sizes.

**Q: What if a team has less than 5 games?**
A: The system uses `--min-games 5` by default. Teams with fewer games get default values (0).

## Summary

✅ **Time-aware:** Only uses games before prediction point
✅ **Current-season focused:** Use `--days-filter 180` to exclude old seasons
✅ **Refreshable:** Rebuild with latest data daily/weekly/monthly
✅ **Production-ready:** Add to cron job or CI/CD pipeline for automated updates

---
**Date:** January 30, 2025
**Status:** Phase 2: COMPLETE
