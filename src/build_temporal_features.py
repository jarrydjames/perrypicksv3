"""
Phase 2: Temporal Features Builder

Extract team-level temporal features from raw game data:
- Rolling averages (last 5 games, last 10 games)
- Momentum indicators (win/loss streaks)
- Rest/fatigue (days since last game, back-to-back flag)
- Travel/home-away context
"""
import json
import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import glob
import numpy as np


def load_all_games(data_dir: str) -> List[Dict[str, Any]]:
    """Load all game JSON files from data directory."""
    game_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    games = []
    
    for file_path in game_files:
        try:
            with open(file_path, 'r') as f:
                game_data = json.load(f)
                games.append(game_data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return games


def extract_game_summary(game: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key information from game data."""
    # Check if game object is valid
    if not game or not isinstance(game, dict):
        print(f"Warning: Invalid game object: {type(game)}")
        return {}
    
    # Parse game date (try multiple fields)
    game_id = game.get('gameId', 'unknown')
    game_date_str = None
    for date_field in ['gameTimeUTC', 'gameTimeLocal', 'gameTimeHome', 'gameTimeAway', 'gameEt']:
        if date_field in game:
            game_date_str = game[date_field]
            break
    
    if game_date_str is None:
        print(f"Warning: No date field found for game {game_id}")
        game_date = datetime(2023, 1, 1)
    else:
        try:
            # Parse datetime string (handles both Z and +00:00 formats)
            game_date = datetime.fromisoformat(game_date_str)
        except Exception as e:
            # Try removing timezone info as fallback
            try:
                date_without_tz = game_date_str.split('+')[0].split('Z')[0]
                game_date = datetime.fromisoformat(date_without_tz)
            except Exception as e2:
                print(f"Warning: Could not parse date {game_date_str} for game {game_id}: {e}, {e2}")
                game_date = datetime(2023, 1, 1)
    
    # Get team information
    home_team = game.get('homeTeam', {})
    away_team = game.get('awayTeam', {})
    
    if not home_team or not away_team:
        print(f"Warning: Missing team data for game {game_id}")
        return {}
    
    # Get team information
    home_team = game['homeTeam']
    away_team = game['awayTeam']
    periods = game.get('periods', [])
    
    # Get period scores
    period_scores = {p['period']: p.get('score', 0) for p in periods if 'period' in p}
    
    return {
        'game_id': game.get('gameId', 'unknown'),
        'game_date': game_date,
        'game_code': game.get('gameCode', ''),
        
        # Home team
        'home_team_id': home_team.get('teamId'),
        'home_team_name': home_team.get('teamName'),
        'home_team_city': home_team.get('teamCity'),
        'home_team_code': home_team.get('teamTricode'),
        'home_score': home_team.get('score'),
        
        # Away team
        'away_team_id': away_team.get('teamId'),
        'away_team_name': away_team.get('teamName'),
        'away_team_city': away_team.get('teamCity'),
        'away_team_code': away_team.get('teamTricode'),
        'away_score': away_team.get('score'),
        
        # Period scores
        'periods': periods,
        
        # Halftime scores (Q1+Q2)
        'home_h1': period_scores.get(1, 0) + period_scores.get(2, 0),
        'away_h1': period_scores.get(1, 0) + period_scores.get(2, 0),
        
        # Q3 scores (for Q3 model)
        'home_q3': period_scores.get(3, 0),
        'away_q3': period_scores.get(3, 0),
        
        # Final scores
        'home_final': home_team['score'],
        'away_final': away_team['score'],
        
        # Margin and total
        'margin': home_team['score'] - away_team['score'],
        'total': home_team['score'] + away_team['score']
    }


def calculate_rolling_features(games_df: pd.DataFrame, days_filter: int = None, min_games: int = 5) -> pd.DataFrame:
    """Calculate rolling statistics for each team.
    
    Args:
        days_filter: Only use games within N days before each game
        min_games: Minimum number of games required for rolling window
    """
    
    # Create separate DataFrames for home and away games
    home_games = games_df[['game_id', 'game_date', 'home_team_id', 'home_team_code', 'home_score', 'away_score', 'margin', 'total']].copy()
    home_games.columns = ['game_id', 'game_date', 'team_id', 'team_code', 'points_scored', 'points_allowed', 'margin', 'total']
    
    away_games = games_df[['game_id', 'game_date', 'away_team_id', 'away_team_code', 'away_score', 'home_score', 'margin', 'total']].copy()
    away_games.columns = ['game_id', 'game_date', 'team_id', 'team_code', 'points_scored', 'points_allowed', 'margin', 'total']
    
    # Reverse margin for away games (negative of home margin)
    away_games['margin'] = -away_games['margin']
    
    # Combine
    all_team_games = pd.concat([home_games, away_games]).sort_values(['team_id', 'game_date'])
    
    # Calculate rolling features for each team
    rolling_features = []
    
    for team_id in all_team_games['team_id'].unique():
        team_games = all_team_games[all_team_games['team_id'] == team_id].sort_values('game_date')
        
        for idx, row in enumerate(team_games.itertuples()):
            # Previous games (before current game)
            prev_games = team_games.iloc[:idx]
            
            # Apply days filter if specified
            if days_filter is not None and len(prev_games) > 0:
                # Only use games within N days before current game
                cutoff_date = row.game_date - timedelta(days=days_filter)
                prev_games = prev_games[prev_games['game_date'] >= cutoff_date]
            
            # Calculate features based on previous games
            if len(prev_games) >= 5:
                # Last 5 games
                last_5 = prev_games.tail(5)
                features_5 = {
                    'pts_scored_avg_5': last_5['points_scored'].mean(),
                    'pts_allowed_avg_5': last_5['points_allowed'].mean(),
                    'margin_avg_5': last_5['margin'].mean(),
                    'total_avg_5': last_5['total'].mean(),
                    'wins_5': (last_5['margin'] > 0).sum(),
                    'current_streak_5': calculate_streak(last_5['margin'].tolist())
                }
            else:
                features_5 = {
                    'pts_scored_avg_5': 0, 'pts_allowed_avg_5': 0, 'margin_avg_5': 0,
                    'total_avg_5': 0, 'wins_5': 0, 'current_streak_5': 0
                }
            
            if len(prev_games) >= 10:
                # Last 10 games
                last_10 = prev_games.tail(10)
                features_10 = {
                    'pts_scored_avg_10': last_10['points_scored'].mean(),
                    'pts_allowed_avg_10': last_10['points_allowed'].mean(),
                    'margin_avg_10': last_10['margin'].mean(),
                    'total_avg_10': last_10['total'].mean(),
                    'wins_10': (last_10['margin'] > 0).sum(),
                }
            else:
                features_10 = {
                    'pts_scored_avg_10': 0, 'pts_allowed_avg_10': 0, 'margin_avg_10': 0,
                    'total_avg_10': 0, 'wins_10': 0
                }
            
            # Rest days
            if len(prev_games) >= 1:
                last_game_date = prev_games.iloc[-1]['game_date']
                days_since_last = (row.game_date - last_game_date).days
            else:
                days_since_last = 999  # No previous game
            
            # Back-to-back flag
            is_back_to_back = (days_since_last == 1)
            
            # Build feature row
            feature_row = {
                'team_id': team_id,
                'game_id': row.game_id,
                'game_date': row.game_date,
                **features_5,
                **features_10,
                'days_since_last': days_since_last,
                'is_back_to_back': int(is_back_to_back)
            }
            
            rolling_features.append(feature_row)
    
    return pd.DataFrame(rolling_features)


def calculate_streak(margins: List[int]) -> int:
    """Calculate current streak (positive = win streak, negative = loss streak)."""
    if not margins:
        return 0
    
    current_streak = 0
    for margin in reversed(margins):
        if margin > 0:  # Win
            if current_streak >= 0:
                current_streak += 1
            else:
                break
        elif margin < 0:  # Loss
            if current_streak <= 0:
                current_streak -= 1
            else:
                break
        else:  # Tie
            break
    
    return current_streak


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Build temporal features from raw game data')
    parser.add_argument('--data-dir', type=str, default='data/raw/box', help='Path to raw game JSON files')
    parser.add_argument('--output-dir', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--days-filter', type=int, default=None, 
                       help='Only use games within N days before prediction (e.g., 180 for 6 months)')
    parser.add_argument('--min-games', type=int, default=5,
                       help='Minimum number of games required for rolling window (default: 5)')
    
    args = parser.parse_args()
    
    print('=' * 70)
    print('PHASE 2: TEMPORAL FEATURES BUILDER')
    print('=' * 70)
    
    # Load all games
    print(f"\nLoading games from {args.data_dir}...")
    games = load_all_games(args.data_dir)
    print(f"Loaded {len(games)} games")
    
    # Extract game summaries
    print("\nExtracting game summaries...")
    game_summaries = [extract_game_summary(game) for game in games]
    games_df = pd.DataFrame(game_summaries)
    
    print(f"Date range: {games_df['game_date'].min()} to {games_df['game_date'].max()}")
    print(f"Unique teams: {games_df['home_team_id'].nunique()}")
    
    # Calculate rolling features
    print("\nCalculating rolling features...")
    if args.days_filter:
        print(f"  Using days filter: {args.days_filter} days")
    rolling_df = calculate_rolling_features(games_df, days_filter=args.days_filter, min_games=args.min_games)
    
    print(f"Rolling features: {len(rolling_df)} records")
    print(f"\nRolling feature columns: {rolling_df.columns.tolist()}")
    
    # Save rolling features
    output_path = os.path.join(args.output_dir, 'rolling_features.parquet')
    rolling_df.to_parquet(output_path, index=False)
    print(f"\nSaved rolling features: {output_path}")
    
    # Merge with game summaries
    print("\nMerging rolling features with game data...")
    # Join rolling features for home and away teams
    home_rolling = rolling_df.rename(columns={c: f'home_{c}' for c in rolling_df.columns if c not in ['game_id', 'team_id', 'game_date']})
    away_rolling = rolling_df.rename(columns={c: f'away_{c}' for c in rolling_df.columns if c not in ['game_id', 'team_id', 'game_date']})
    
    # Merge
    final_df = games_df.merge(
        home_rolling, on=['game_id'], how='left'
    ).merge(
        away_rolling, on=['game_id'], how='left'
    )
    
    # Save final dataset
    final_output = os.path.join(args.output_dir, 'games_with_temporal_features.parquet')
    final_df.to_parquet(final_output, index=False)
    print(f"\nSaved final dataset with temporal features: {final_output}")
    
    print(f"\nFinal dataset: {len(final_df)} games")
    print(f"Final columns: {final_df.columns.tolist()}")
    
    # Summary statistics
    print('\n' + '=' * 70)
    print('TEMPORAL FEATURES SUMMARY')
    print('=' * 70)
    print(f"\nRolling Statistics (Home Team):")
    print(f"  Home pts_scored_avg_5: {final_df['home_pts_scored_avg_5'].mean():.2f} ± {final_df['home_pts_scored_avg_5'].std():.2f}")
    print(f"  Home margin_avg_5: {final_df['home_margin_avg_5'].mean():.2f} ± {final_df['home_margin_avg_5'].std():.2f}")
    print(f"  Home current_streak_5: {final_df['home_current_streak_5'].mean():.2f} ± {final_df['home_current_streak_5'].std():.2f}")
    print(f"  Home is_back_to_back: {final_df['home_is_back_to_back'].mean()*100:.1f}% of games")
    print(f"\nRolling Statistics (Away Team):")
    print(f"  Away pts_scored_avg_5: {final_df['away_pts_scored_avg_5'].mean():.2f} ± {final_df['away_pts_scored_avg_5'].std():.2f}")
    print(f"  Away margin_avg_5: {final_df['away_margin_avg_5'].mean():.2f} ± {final_df['away_margin_avg_5'].std():.2f}")
    print(f"  Away current_streak_5: {final_df['away_current_streak_5'].mean():.2f} ± {final_df['away_current_streak_5'].std():.2f}")
    print(f"  Away is_back_to_back: {final_df['away_is_back_to_back'].mean()*100:.1f}% of games")
    
    print('=' * 70)
    print('PHASE 2: COMPLETE')
    print('=' * 70)

if __name__ == '__main__':
    main()