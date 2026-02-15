from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from collections import defaultdict


def add_temporal_features_to_halftime(input_path: str, output_path: str):
    """
    Add temporal features to halftime dataset using game history.
    
    Args:
        input_path: Path to halftime_team_v2.parquet
        output_path: Path to save halftime_with_temporal_features_total.parquet
    """
    print("=" * 70)
    print("ADDING TEMPORAL FEATURES TO HALFTIME DATA")
    print("=" * 70)
    print()
    
    # Load halftime data
    print(f"Loading halftime data from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"  Loaded {len(df)} games")
    print(f"  Columns: {len(df.columns)}")
    print()
    
    # Check what we have
    if 'game_id' not in df.columns:
        raise ValueError("Missing game_id column!")
    
    # We need game dates - check if we have them
    date_columns = [c for c in df.columns if 'date' in c.lower()]
    print(f"Date columns found: {date_columns}")
    
    # Load game IDs file to get dates and teams
    import json
    with open("data/processed/game_ids_2025.json", "r") as f:
        schedule = json.load(f)
    
    # Create game date and team lookup
    game_dates = {}
    game_teams = {}
    for game in schedule:
        game_id = game.get("gameId")
        game_date_str = game.get("gameDate")
        if game_id and game_date_str:
            try:
                game_dates[game_id] = pd.to_datetime(game_date_str[:10])
            except Exception as e:
                print(f"  Warning: Could not parse date for game {game_id}: {e}")
        
        # Get teams (might be string or dict)
        home_team = game.get("homeTeam")
        away_team = game.get("awayTeam")
        
        # Handle both string and dict formats
        home_tri = home_team if isinstance(home_team, str) else home_team.get("teamTricode") if isinstance(home_team, dict) else None
        away_tri = away_team if isinstance(away_team, str) else away_team.get("teamTricode") if isinstance(away_team, dict) else None
        
        game_teams[game_id] = {
            'home_tri': home_tri,
            'away_tri': away_tri
        }
    
    print(f"  Loaded game dates for {len(game_dates)} games")
    print()
    
    # Add game_date to dataframe
    df['game_date'] = df['game_id'].map(game_dates)
    
    # Check for missing dates
    missing_dates = df['game_date'].isna().sum()
    if missing_dates > 0:
        print(f"  Warning: {missing_dates} games missing dates")
        # Fill with a default date for games without dates
        df['game_date'] = df['game_date'].fillna(pd.to_datetime('2025-01-01'))
    
    # Get team IDs from the halftime data or schedule
    # We'll need home_team_id and away_team_id
    # Check if we have team triCodes
    if 'home_tri' in df.columns and 'away_tri' in df.columns:
        print("Using home_tri and away_tri for team identification")
        # Create team ID lookup from schedule
        team_id_lookup = {}
        for game in schedule:
            game_id = game.get("gameId")
            home_team = game.get("homeTeam", {})
            away_team = game.get("awayTeam", {})
            home_tri = home_team.get("teamTricode")
            away_tri = away_team.get("teamTricode")
            home_id = home_team.get("teamId")
            away_id = away_team.get("teamId")
            
            if home_tri and home_id:
                team_id_lookup[home_tri] = home_id
            if away_tri and away_id:
                team_id_lookup[away_tri] = away_id
        
        df['home_team_id'] = df['home_tri'].map(team_id_lookup)
        df['away_team_id'] = df['away_tri'].map(team_id_lookup)
        print(f"  Mapped team IDs for {df['home_team_id'].notna().sum()} games")
    
    # If we still don't have team IDs, get them from the schedule
    if 'home_team_id' not in df.columns or df['home_team_id'].isna().any():
        print("  Getting team IDs from schedule...")
        # Create unique team IDs from triCodes
        all_teams = set()
        for game_id, teams in game_teams.items():
            if teams['home_tri']:
                all_teams.add(teams['home_tri'])
            if teams['away_tri']:
                all_teams.add(teams['away_tri'])
        
        team_to_id = {team: idx for idx, team in enumerate(sorted(all_teams))}
        
        # Map to dataframe
        df['home_tri'] = df['game_id'].map(lambda x: game_teams.get(x, {}).get('home_tri'))
        df['away_tri'] = df['game_id'].map(lambda x: game_teams.get(x, {}).get('away_tri'))
        df['home_team_id'] = df['home_tri'].map(team_to_id)
        df['away_team_id'] = df['away_tri'].map(team_to_id)
    
    print(f"  Home team IDs: {df['home_team_id'].notna().sum()} / {len(df)}")
    print(f"  Away team IDs: {df['away_team_id'].notna().sum()} / {len(df)}")
    print()
    
    # Build team game history for temporal features
    print("Building team game histories...")
    
    # Create game history for each team
    team_games = defaultdict(list)
    
    for idx, row in df.iterrows():
        game_date = row['game_date']
        home_id = row['home_team_id']
        away_id = row['away_team_id']
        
        # Get scores - we need h1_home, h1_away from halftime data
        h1_home = row.get('h1_home', 0)
        h1_away = row.get('h1_away', 0)
        
        # Store home game
        team_games[home_id].append({
            'game_id': row['game_id'],
            'game_date': game_date,
            'team_id': home_id,
            'is_home': True,
            'points_scored': h1_home,
            'points_allowed': h1_away,
            'margin': h1_home - h1_away,
            'won': h1_home > h1_away
        })
        
        # Store away game
        team_games[away_id].append({
            'game_id': row['game_id'],
            'game_date': game_date,
            'team_id': away_id,
            'is_home': False,
            'points_scored': h1_away,
            'points_allowed': h1_home,
            'margin': h1_away - h1_home,
            'won': h1_away > h1_home
        })
    
    print(f"  Built histories for {len(team_games)} teams")
    print()
    
    # Calculate rolling features for each team
    print("Calculating rolling features...")
    
    team_features = {}
    
    for team_id, games in team_games.items():
        # Sort by date
        games_df = pd.DataFrame(games).sort_values('game_date').reset_index(drop=True)
        
        # Calculate rolling features (5-game window)
        games_df['pts_scored_avg_5'] = games_df['points_scored'].rolling(5, min_periods=1).mean().shift(1)
        games_df['pts_allowed_avg_5'] = games_df['points_allowed'].rolling(5, min_periods=1).mean().shift(1)
        games_df['margin_avg_5'] = games_df['margin'].rolling(5, min_periods=1).mean().shift(1)
        games_df['wins_5'] = games_df['won'].rolling(5, min_periods=1).sum().shift(1)
        
        # Calculate streak
        def calculate_streak(won_list):
            if not won_list or len(won_list) == 0:
                return 0
            streak = 0
            for won in reversed(won_list):
                if won:
                    if streak >= 0:
                        streak += 1
                    else:
                        break
                else:
                    if streak <= 0:
                        streak -= 1
                    else:
                        break
            return streak
        
        games_df['current_streak_5'] = games_df['won'].rolling(5, min_periods=1).apply(
            lambda x: calculate_streak(x.tolist()) if len(x) > 0 else 0
        ).shift(1)
        
        # Days since last game
        games_df['days_since_last'] = games_df['game_date'].diff().dt.days
        games_df['is_back_to_back'] = (games_df['days_since_last'] == 1).astype(int)
        
        # Fill NaN values
        games_df = games_df.fillna({
            'pts_scored_avg_5': 0,
            'pts_allowed_avg_5': 0,
            'margin_avg_5': 0,
            'wins_5': 0,
            'current_streak_5': 0,
            'days_since_last': 7,  # Default to 7 days if no previous game
            'is_back_to_back': 0
        })
        
        # Store by game_id
        for idx, row in games_df.iterrows():
            team_features[(team_id, row['game_id'])] = {
                'pts_scored_avg_5': row['pts_scored_avg_5'],
                'pts_allowed_avg_5': row['pts_allowed_avg_5'],
                'margin_avg_5': row['margin_avg_5'],
                'wins_5': row['wins_5'],
                'current_streak_5': row['current_streak_5'],
                'days_since_last': row['days_since_last'],
                'is_back_to_back': row['is_back_to_back']
            }
    
    print(f"  Calculated features for {len(team_features)} team-games")
    print()
    
    # Add features to original dataframe
    print("Adding temporal features to dataset...")
    
    # Home team features
    home_features = []
    for idx, row in df.iterrows():
        key = (row['home_team_id'], row['game_id'])
        features = team_features.get(key, {})
        home_features.append(features)
    
    home_df = pd.DataFrame(home_features)
    home_df.columns = [f'home_{c}' for c in home_df.columns]
    
    # Away team features
    away_features = []
    for idx, row in df.iterrows():
        key = (row['away_team_id'], row['game_id'])
        features = team_features.get(key, {})
        away_features.append(features)
    
    away_df = pd.DataFrame(away_features)
    away_df.columns = [f'away_{c}' for c in away_df.columns]
    
    # Combine
    result_df = pd.concat([df, home_df, away_df], axis=1)
    
    print(f"  Final dataset: {len(result_df)} games, {len(result_df.columns)} features")
    print()
    
    # Save
    print(f"Saving to {output_path}...")
    result_df.to_parquet(output_path, index=False)
    print("  ✓ Saved")
    print()
    
    # Summary
    print("=" * 70)
    print("TEMPORAL FEATURES ADDED")
    print("=" * 70)
    print(f"\nTotal games: {len(result_df)}")
    print(f"Total features: {len(result_df.columns)}")
    print(f"\nNew temporal features:")
    print(f"  home_pts_scored_avg_5")
    print(f"  home_pts_allowed_avg_5")
    print(f"  home_margin_avg_5")
    print(f"  home_wins_5")
    print(f"  home_current_streak_5")
    print(f"  home_days_since_last")
    print(f"  home_is_back_to_back")
    print(f"  away_pts_scored_avg_5")
    print(f"  away_pts_allowed_avg_5")
    print(f"  away_margin_avg_5")
    print(f"  away_wins_5")
    print(f"  away_current_streak_5")
    print(f"  away_days_since_last")
    print(f"  away_is_back_to_back")
    print()
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add temporal features to halftime dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/halftime_team_v2.parquet",
        help="Input halftime dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/halftime_with_temporal_features_total.parquet",
        help="Output dataset with temporal features"
    )
    
    args = parser.parse_args()
    
    add_temporal_features_to_halftime(args.input, args.output)
