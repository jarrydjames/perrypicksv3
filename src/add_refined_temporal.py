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


def add_refined_temporal_features(input_path: str, output_path: str):
    """
    Add refined temporal features to halftime dataset.
    
    Key improvements:
    - Longer windows (10, 20 games)
    - Exponential weighted averages (recent games matter more)
    - Better handling of early season zeros
    - Home/away splits
    - Trend indicators
    - Opponent-adjusted stats
    """
    print("=" * 70)
    print("ADDING REFINED TEMPORAL FEATURES TO HALFTIME DATA")
    print("=" * 70)
    print()
    
    # Load halftime data
    print(f"Loading halftime data from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"  Loaded {len(df)} games")
    print(f"  Columns: {len(df.columns)}")
    print()
    
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
        
        # Get teams
        home_team = game.get("homeTeam")
        away_team = game.get("awayTeam")
        
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
    
    # Get team IDs
    all_teams = set()
    for game_id, teams in game_teams.items():
        if teams['home_tri']:
            all_teams.add(teams['home_tri'])
        if teams['away_tri']:
            all_teams.add(teams['away_tri'])
    
    team_to_id = {team: idx for idx, team in enumerate(sorted(all_teams))}
    
    df['home_tri'] = df['game_id'].map(lambda x: game_teams.get(x, {}).get('home_tri'))
    df['away_tri'] = df['game_id'].map(lambda x: game_teams.get(x, {}).get('away_tri'))
    df['home_team_id'] = df['home_tri'].map(team_to_id)
    df['away_team_id'] = df['away_tri'].map(team_to_id)
    
    print(f"  Home team IDs: {df['home_team_id'].notna().sum()} / {len(df)}")
    print(f"  Away team IDs: {df['away_team_id'].notna().sum()} / {len(df)}")
    print()
    
    # Build team game history
    print("Building team game histories...")
    
    team_games = defaultdict(list)
    
    for idx, row in df.iterrows():
        game_date = row['game_date']
        home_id = row['home_team_id']
        away_id = row['away_team_id']
        
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
            'won': h1_home > h1_away,
            'opponent_id': away_id
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
            'won': h1_away > h1_home,
            'opponent_id': home_id
        })
    
    print(f"  Built histories for {len(team_games)} teams")
    print()
    
    # Calculate refined features for each team
    print("Calculating refined temporal features...")
    
    team_features = {}
    
    for team_id, games in team_games.items():
        # Sort by date
        games_df = pd.DataFrame(games).sort_values('game_date').reset_index(drop=True)
        
        # ========================================
        # FEATURE 1: Rolling averages (5, 10, 20 games)
        # ========================================
        for window in [5, 10, 20]:
            games_df[f'pts_scored_avg_{window}'] = games_df['points_scored'].rolling(window, min_periods=1).mean().shift(1)
            games_df[f'pts_allowed_avg_{window}'] = games_df['points_allowed'].rolling(window, min_periods=1).mean().shift(1)
            games_df[f'margin_avg_{window}'] = games_df['margin'].rolling(window, min_periods=1).mean().shift(1)
            games_df[f'wins_{window}'] = games_df['won'].rolling(window, min_periods=1).sum().shift(1)
        
        # ========================================
        # FEATURE 2: Exponential weighted averages (more weight to recent games)
        # ========================================
        for span in [5, 10, 20]:
            alpha = 2.0 / (span + 1.0)  # Smoothing factor
            games_df[f'pts_scored_ewm_{span}'] = games_df['points_scored'].ewm(span=span, min_periods=1).mean().shift(1)
            games_df[f'pts_allowed_ewm_{span}'] = games_df['points_allowed'].ewm(span=span, min_periods=1).mean().shift(1)
            games_df[f'margin_ewm_{span}'] = games_df['margin'].ewm(span=span, min_periods=1).mean().shift(1)
        
        # ========================================
        # FEATURE 3: Streaks
        # ========================================
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
        
        games_df['current_streak'] = games_df['won'].rolling(10, min_periods=1).apply(
            lambda x: calculate_streak(x.tolist()) if len(x) > 0 else 0
        ).shift(1)
        
        # ========================================
        # FEATURE 4: Rest days and back-to-back
        # ========================================
        games_df['days_since_last'] = games_df['game_date'].diff().dt.days
        games_df['is_back_to_back'] = (games_df['days_since_last'] == 1).astype(int)
        games_df['is_3_in_4'] = games_df['days_since_last'].rolling(4, min_periods=1).apply(
            lambda x: (x <= 1).sum() >= 2 if len(x) >= 4 else 0
        ).shift(1)
        
        # ========================================
        # FEATURE 5: Home/Away splits
        # ========================================
        home_games = games_df[games_df['is_home'] == True].copy()
        away_games = games_df[games_df['is_home'] == False].copy()
        
        # Rolling averages for home games only
        games_df['pts_scored_home_avg_5'] = home_games['points_scored'].rolling(5, min_periods=1).mean().reindex(games_df.index).shift(1)
        games_df['margin_home_avg_5'] = home_games['margin'].rolling(5, min_periods=1).mean().reindex(games_df.index).shift(1)
        
        # Rolling averages for away games only
        games_df['pts_scored_away_avg_5'] = away_games['points_scored'].rolling(5, min_periods=1).mean().reindex(games_df.index).shift(1)
        games_df['margin_away_avg_5'] = away_games['margin'].rolling(5, min_periods=1).mean().reindex(games_df.index).shift(1)
        
        # ========================================
        # FEATURE 6: Trend indicators (getting better/worse)
        # ========================================
        games_df['margin_trend_5'] = games_df['margin'].rolling(5, min_periods=1).apply(
            lambda x: x.iloc[-1] - x.iloc[0] if len(x) >= 2 else 0
        ).shift(1)
        
        games_df['pts_trend_5'] = games_df['points_scored'].rolling(5, min_periods=1).apply(
            lambda x: x.iloc[-1] - x.iloc[0] if len(x) >= 2 else 0
        ).shift(1)
        
        # ========================================
        # FEATURE 7: Volatility (standard deviation)
        # ========================================
        games_df['margin_std_5'] = games_df['margin'].rolling(5, min_periods=2).std().shift(1)
        games_df['pts_scored_std_5'] = games_df['points_scored'].rolling(5, min_periods=2).std().shift(1)
        
        # ========================================
        # FEATURE 8: Games played (experience)
        # ========================================
        games_df['games_played'] = games_df['game_id'].expanding().count().shift(1)
        
        # ========================================
        # Fill NaN values intelligently
        # ========================================
        fill_values = {
            # Rolling averages - use league average (placeholder, would be better to calculate actual league avg)
            'pts_scored_avg_5': 54,  # League average halftime score
            'pts_allowed_avg_5': 54,
            'margin_avg_5': 0,
            'wins_5': 2.5,  # Half of 5 games
            'pts_scored_avg_10': 54,
            'pts_allowed_avg_10': 54,
            'margin_avg_10': 0,
            'wins_10': 5,
            'pts_scored_avg_20': 54,
            'pts_allowed_avg_20': 54,
            'margin_avg_20': 0,
            'wins_20': 10,
            # EWM
            'pts_scored_ewm_5': 54,
            'pts_allowed_ewm_5': 54,
            'margin_ewm_5': 0,
            'pts_scored_ewm_10': 54,
            'pts_allowed_ewm_10': 54,
            'margin_ewm_10': 0,
            'pts_scored_ewm_20': 54,
            'pts_allowed_ewm_20': 54,
            'margin_ewm_20': 0,
            # Streaks
            'current_streak': 0,
            # Rest
            'days_since_last': 7,
            'is_back_to_back': 0,
            'is_3_in_4': 0,
            # Home/Away splits
            'pts_scored_home_avg_5': 54,
            'margin_home_avg_5': 0,
            'pts_scored_away_avg_5': 54,
            'margin_away_avg_5': 0,
            # Trends
            'margin_trend_5': 0,
            'pts_trend_5': 0,
            # Volatility
            'margin_std_5': 5,  # Reasonable default
            'pts_scored_std_5': 5,
            # Experience
            'games_played': 0
        }
        
        games_df = games_df.fillna(fill_values)
        
        # Store by game_id
        feature_cols = [c for c in games_df.columns if c not in ['game_id', 'game_date', 'team_id', 'is_home', 'points_scored', 'points_allowed', 'margin', 'won', 'opponent_id']]
        
        for idx, row in games_df.iterrows():
            team_features[(team_id, row['game_id'])] = row[feature_cols].to_dict()
    
    print(f"  Calculated features for {len(team_features)} team-games")
    print()
    
    # Add features to original dataframe
    print("Adding refined temporal features to dataset...")
    
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
    
    # Add differential features (home - away)
    print("  Adding differential features...")
    home_feature_cols = [c for c in result_df.columns if c.startswith('home_') and 'team_id' not in c and 'tri' not in c]
    
    for col in home_feature_cols:
        away_col = col.replace('home_', 'away_')
        if away_col in result_df.columns:
            diff_col = col.replace('home_', 'diff_')
            result_df[diff_col] = result_df[col] - result_df[away_col]
    
    print(f"  Final dataset: {len(result_df)} games, {len(result_df.columns)} features")
    print()
    
    # Save
    print(f"Saving to {output_path}...")
    result_df.to_parquet(output_path, index=False)
    print("  ✓ Saved")
    print()
    
    # Summary
    print("=" * 70)
    print("REFINED TEMPORAL FEATURES ADDED")
    print("=" * 70)
    print(f"\nTotal games: {len(result_df)}")
    print(f"Total features: {len(result_df.columns)}")
    print(f"\nNew temporal features (examples):")
    print(f"\n  Rolling averages:")
    print(f"    - home_pts_scored_avg_5/10/20")
    print(f"    - home_margin_avg_5/10/20")
    print(f"    - home_wins_5/10/20")
    print(f"\n  Exponential weighted:")
    print(f"    - home_pts_scored_ewm_5/10/20")
    print(f"    - home_margin_ewm_5/10/20")
    print(f"\n  Home/Away splits:")
    print(f"    - home_pts_scored_home_avg_5")
    print(f"    - home_margin_home_avg_5")
    print(f"    - away_pts_scored_away_avg_5")
    print(f"\n  Trends:")
    print(f"    - home_margin_trend_5")
    print(f"    - home_pts_trend_5")
    print(f"\n  Rest:")
    print(f"    - home_days_since_last")
    print(f"    - home_is_back_to_back")
    print(f"    - home_is_3_in_4")
    print(f"\n  Volatility:")
    print(f"    - home_margin_std_5")
    print(f"    - home_pts_scored_std_5")
    print(f"\n  Differential features:")
    print(f"    - diff_pts_scored_avg_5 (home - away)")
    print(f"    - diff_margin_avg_5 (home - away)")
    print(f"    - diff_wins_5 (home - away)")
    print()
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add refined temporal features to halftime dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/halftime_team_v2.parquet",
        help="Input halftime dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/halftime_with_refined_temporal.parquet",
        help="Output dataset with refined temporal features"
    )
    
    args = parser.parse_args()
    
    add_refined_temporal_features(args.input, args.output)
