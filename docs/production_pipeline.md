# Production Pipeline - Automated Daily Refresh

## Overview

Currently, the PerryPicks v3 system uses **manual** game selection through the Streamlit UI. There's no automated pipeline to:
1. Pull today's games automatically
2. Calculate temporal features with latest data
3. Retrain model with recent performance
4. Deploy updated model to production

## Current Flow (Manual)

```
User opens Streamlit app
  ↓
User selects date (e.g., "Jan 29, 2026")
  ↓
App calls: fetch_scoreboard(pick_date)
  ↓
App displays: Game predictions
  ↓
User manually views predictions
```

## Missing Automation

| Component | Current Status | Impact |
|-----------|----------------|----------|
| **Daily Game Pull** | ❌ Manual only | No recent games for temporal features |
| **Temporal Feature Refresh** | ❌ Static dataset | Rolling features don't update with latest games |
| **Model Retraining** | ❌ Manual only | Model doesn't learn from recent team performance |
| **Production Deployment** | ❌ Manual only | No auto-update mechanism |

## Recommended Production Pipeline

### Daily Refresh Script

Create `src/daily_refresh.sh`:

```bash
#!/bin/bash

# PerryPicks v3 - Daily Refresh Pipeline
# Run this daily via cron: 0 8 * * * /path/to/refresh.sh

DATE=$(date +"%Y-%m-%d")

echo "===== PERRY PICKS DAILY REFRESH: $DATE ====="

# Step 1: Fetch today's games (last 2 days)
echo "[1/5] Fetching today's games..."
python3 src/fetch_today_games.py --date $DATE --days-before 2

# Step 2: Update temporal features
# The temporal features builder is already time-aware
# It will only use games BEFORE each prediction date
echo "[2/5] Updating temporal features..."
python3 src/build_temporal_features.py \
  --data-dir data/raw/box \
  --output-dir data/processed \
  --days-filter 180

# Step 3: Merge with halftime dataset
echo "[3/5] Merging temporal features with halftime stats..."
python3 src/merge_temporal_halftime.py

# Step 4: Retrain model
echo "[4/5] Retraining halftime model..."
python3 src/train_halftime_model.py \
  --dataset data/processed/halftime_with_temporal_features.parquet

# Step 5: Deploy to production
echo "[5/5] Deploying updated model..."
cp models_v3/halftime/* models_v3/production/

echo "===== REFRESH COMPLETE ====="
```

### Step 1: Fetch Today's Games

Create `src/fetch_today_games.py`:

```python
"""
Fetch today's and recent games for temporal feature updates.
"""
import requests
import json
from datetime import datetime, timedelta

def fetch_schedule(days_before: int = 2) -> list:
    """Fetch NBA games from recent days."""
    today = datetime.now()
    date = today - timedelta(days=days_before)
    
    # Format: YYYYMMDD for NBA API
    date_str = date.strftime("%Y%m%d")
    
    # Fetch schedule
    url = f"https://cdn.nba.com/static/json/liveData/scoreboard_todays_league_v2_{date_str}.json"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        return data.get('scoreboard', {}).get('games', [])
    except Exception as e:
        print(f"Error fetching schedule: {e}")
        return []

def save_games_to_cache(games: list):
    """Save games to cache for temporal feature builder."""
    cache_path = 'data/raw/todays_games.json'
    
    with open(cache_path, 'w') as f:
        json.dump(games, f, indent=2)
    
    print(f"Saved {len(games)} games to {cache_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch today\'s NBA games')
    parser.add_argument('--date', type=str, default=None, help='Date in YYYY-MM-DD format')
    parser.add_argument('--days-before', type=int, default=2, help='Days before today to fetch')
    parser.add_argument('--cache-only', action='store_true', help='Only save to cache, don\'t fetch')
    
    args = parser.parse_args()
    
    # Get date
    if args.date:
        date = datetime.strptime(args.date, '%Y-%m-%d')
    else:
        date = datetime.now()
    
    print(f"Fetching games around {date.strftime('%Y-%m-%d')}...")
    
    games = fetch_schedule(days_before=args.days_before)
    print(f"Found {len(games)} games")
    
    save_games_to_cache(games)

if __name__ == '__main__':
    main()
```

### Step 3: Merge Temporal + Halftime

Create `src/merge_temporal_halftime.py`:

```python
"""
Merge temporal features with halftime statistics.
"""
import pandas as pd
import os

def main():
    # Load temporal features
    temporal = pd.read_parquet('data/processed/games_with_temporal_features.parquet')
    
    # Load halftime data
    halftime = pd.read_parquet('data/processed/halftime_training_23_24_leakage_free.parquet')
    
    # Join on game_id
    merged = halftime.merge(
        temporal[['game_id', 'game_date', 
                 'home_pts_scored_avg_5', 'home_pts_allowed_avg_5', 'home_margin_avg_5', 'home_current_streak_5', 
                 'home_days_since_last', 'home_is_back_to_back',
                 'away_pts_scored_avg_5', 'away_pts_allowed_avg_5', 'away_margin_avg_5', 'away_current_streak_5',
                 'away_days_since_last', 'away_is_back_to_back']],
        on='game_id',
        how='left'
    )
    
    # Save
    output = 'data/processed/halftime_with_temporal_features.parquet'
    merged.to_parquet(output, index=False)
    print(f"Saved merged dataset: {output}")
    print(f"Games: {len(merged)}")
    print(f"Features: {len(merged.columns)}")

if __name__ == '__main__':
    main()
```

### Step 4: Retrain Model

Update `src/train_halftime_model.py` to use temporal features:

```python
# Add temporal feature columns
TEMPORAL_FEATURES = [
    'home_pts_scored_avg_5', 'home_pts_allowed_avg_5', 'home_margin_avg_5', 'home_current_streak_5',
    'home_days_since_last', 'home_is_back_to_back',
    'away_pts_scored_avg_5', 'away_pts_allowed_avg_5', 'away_margin_avg_5', 'away_current_streak_5',
    'away_days_since_last', 'away_is_back_to_back'
]

# Merge with existing features
FEATURE_COLS = [existing_features] + TEMPORAL_FEATURES
```

## Cron Schedule

```cron
# Daily refresh at 8 AM (after all games from previous day)
0 8 * * * /path/to/PerryPicks_v3/scripts/daily_refresh.sh

# Weekly refresh on Sundays at 10 AM
0 10 * * 0 /path/to/PerryPicks_v3/scripts/daily_refresh.sh
```

## Flow Comparison

### Manual (Current)
```
User: I want predictions for Jan 29
  ↓
App: What games do you have cached?
  ↓
User: I'll type "Jan 29, 2026"
  ↓
App: Here are the predictions
  ↓
Temporal features: Based on data from Aug 2024 (last 5 games before Jan 29)
```

### Automated (Recommended)
```
8 AM Daily:
  ↓
Cron: Fetch last 2 days of games (Jan 27-29, 2026)
  ↓
Script: Save to cache (data/raw/todays_games.json)
  ↓
Script: Build temporal features (Jan 1-27 for Jan 29 predictions)
  ↓
Script: Merge with halftime stats
  ↓
Script: Retrain model with updated features
  ↓
Script: Deploy to production
  ↓
User opens app at 9 AM
  ↓
Temporal features: Based on data from Jan 1-27 (last 5 games before Jan 29)
```

## Implementation Priority

| Priority | Task | Effort | Impact |
|----------|-------|--------|--------|
| **P1** | Create fetch_today_games.py | Low | Enables automation |
| **P1** | Update train_halftime_model.py to use temporal features | Low | Model learns from recent form |
| **P1** | Create merge_temporal_halftime.py | Low | Combines features correctly |
| **P2** | Create daily_refresh.sh script | Low | One-command refresh |
| **P2** | Set up cron job | Low | Fully automated |
| **P3** | Add model versioning | Medium | Rollback capability |
| **P3** | Add A/B testing framework | Medium | Test changes safely |

## Files Needed

- `src/fetch_today_games.py` - Fetch today's games
- `src/merge_temporal_halftime.py` - Merge temporal + halftime
- `scripts/daily_refresh.sh` - Automated refresh script
- `data/raw/todays_games.json` - Game cache
- `models_v3/production/` - Production model directory

## Validation

To verify the pipeline works:

```bash
# Test manual refresh
python3 src/daily_refresh.sh

# Check temporal features
python3 << 'VALIDATE_EOF'
import pandas as pd

# Load latest merged dataset
df = pd.read_parquet('data/processed/halftime_with_temporal_features.parquet')

# Check date range
print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")

# Check temporal features are populated
print(f"\nTemporal features (sample):")
for col in ['home_pts_scored_avg_5', 'home_current_streak_5', 'home_days_since_last']:
    print(f"  {col}: {df[col].mean():.2f} (non-null: {df[col].notna().sum()/len(df)*100:.1f}%)")
VALIDATE_EOF
```

## Summary

**Current State:** Manual game selection, static temporal features

**Recommended:** Fully automated daily refresh pipeline

**Benefits:**
- ✅ Rolling features always use last 5 games from current season
- ✅ Model learns from recent team performance
- ✅ One-command refresh (daily_refresh.sh)
- ✅ Automated via cron job
- ✅ Production model directory for easy rollback

**Next Steps:**
1. Implement P1 tasks (fetch_today_games.py, merge script, update training)
2. Test manual refresh
3. Set up cron job
4. Monitor performance with refreshed data

---
**Date:** January 29, 2026
**Status:** PRODUCTION PIPELINE DESIGN
**Author:** Perry (Code Puppy)