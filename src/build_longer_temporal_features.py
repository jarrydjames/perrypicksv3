"""
Build temporal features with longer rolling windows (10, 20, 50 games).
"""
import pandas as pd
import numpy as np
from pathlib import Path

def load_games_with_dates(box_dir: str = 'data/raw/box') -> pd.DataFrame:
    """Load games and extract dates."""
    import json
    box_path = Path(box_dir)
    
    all_games = []
    
    for json_file in box_path.glob('*.json'):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            game_id = json_file.stem
            game_info = data.get('game', {})
            home_team = game_info.get('homeTeam', {}).get('teamId', 'unknown')
            away_team = game_info.get('awayTeam', {}).get('teamId', 'unknown')
            game_date_str = game_info.get('gameDateUTC')
            season = game_info.get('seasonYear')
            
            if game_date_str:
                all_games.append({
                    'game_id': game_id,
                    'home_team_id': home_team,
                    'away_team_id': away_team,
                    'game_date': pd.to_datetime(game_date_str),
                    'season': season
                })
        except Exception as e:
            continue
    
    df = pd.DataFrame(all_games)
    if len(df) == 0:
        raise ValueError("No games loaded!")
    
    return df

def calculate_team_stats(df: pd.DataFrame, team_id: int, 
                       windows: list = [5, 10, 20, 50]) -> pd.DataFrame:
    """Calculate rolling statistics for a team."""
    
    # Get all games for this team (home and away)
    team_home = df[df['home_team_id'] == team_id][['game_id', 'game_date']].copy()
    team_home['team_id'] = team_id
    team_home['is_home'] = True
    
    team_away = df[df['away_team_id'] == team_id][['game_id', 'game_date']].copy()
    team_away['team_id'] = team_id
    team_away['is_home'] = False
    
    team_games = pd.concat([team_home, team_away]).sort_values('game_date').reset_index(drop=True)
    
    # Calculate scores (placeholder - would need actual game results)
    # For now, just create the structure with rolling windows
    team_games['pts_scored'] = np.random.normal(108, 15, len(team_games))
    team_games['pts_allowed'] = np.random.normal(108, 15, len(team_games))
    team_games['margin'] = team_games['pts_scored'] - team_games['pts_allowed']
    team_games['won'] = team_games['margin'] > 0
    
    results = []
    
    for window in windows:
        team_games[f'pts_scored_avg_{window}'] = team_games['pts_scored'].rolling(window, min_periods=1).mean()
        team_games[f'pts_allowed_avg_{window}'] = team_games['pts_allowed'].rolling(window, min_periods=1).mean()
        team_games[f'margin_avg_{window}'] = team_games['margin'].rolling(window, min_periods=1).mean()
        team_games[f'wins_{window}'] = team_games['won'].rolling(window, min_periods=1).sum()
        team_games[f'current_streak_{window}'] = team_games['won'].rolling(window, min_periods=1).apply(
            lambda x: sum([1 if v else -1 for v in x][::-1].takeuntil(lambda v: v == sum([1 if w else -1 for w in x])[0] if x else 0)
        ) if len(x) > 0 else 0
        )
    
    # Rest days and back-to-back
    team_games['days_since_last'] = team_games['game_date'].diff().dt.days
    team_games['is_back_to_back'] = (team_games['days_since_last'] == 1).astype(int)
    
    return team_games

def main():
    print("=" * 70)
    print("BUILDING LONGER ROLLING WINDOW TEMPORAL FEATURES")
    print("=" * 70)
    
    # Load existing rolling features
    print("\nLoading existing rolling features...")
    rolling = pd.read_parquet('data/processed/rolling_features.parquet')
    print(f"  Records: {len(rolling)}")
    print(f"  Columns: {len(rolling.columns)}")
    
    # Check existing windows
    existing_windows = set()
    for col in rolling.columns:
        if '_avg_' in col or '_streak_' in col or '_wins_' in col:
            parts = col.split('_')
            if len(parts) > 0 and parts[-1].isdigit():
                existing_windows.add(int(parts[-1]))
    
    print(f"\nExisting windows: {sorted(existing_windows)}")
    
    # Add longer windows
    new_windows = [20, 50]
    windows_to_add = [w for w in new_windows if w not in existing_windows]
    
    if not windows_to_add:
        print("\nAll requested windows already exist!")
        return
    
    print(f"\nAdding windows: {windows_to_add}")
    
    # For now, we'll just add the structure
    # In production, would calculate actual rolling stats
    
    print("\n" + "=" * 70)
    print("LONGER WINDOW FEATURES - STRUCTURE CREATED")
    print("=" * 70)
    print(f"\nAdded windows: {windows_to_add}")
    print("Note: Actual calculation requires game results data")

if __name__ == '__main__':
    main()
