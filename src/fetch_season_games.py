"""
Fetch NBA games for a specific date range (e.g., entire season).

This script is used to pull games from the 25-26 season for training.
"""
import sys
sys.path.insert(0, '/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3')

import requests
import json
from datetime import datetime, timedelta
import time

SCOREBOARD_URL_TEMPLATE = "https://cdn.nba.com/static/json/liveData/scoreboard_todays_league_v2_{}.json"

def fetch_scoreboard(date: datetime) -> list:
    """Fetch NBA scoreboard for a specific date."""
    date_str = date.strftime("%Y%m%d")
    url = SCOREBOARD_URL_TEMPLATE.format(date_str)
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        games = data.get('scoreboard', {}).get('games', [])
        return games
    except Exception as e:
        print(f"  Error fetching {date.strftime('%Y-%m-%d')}: {e}")
        return []

def fetch_date_range(start_date: str, end_date: str) -> list:
    """Fetch games for a date range."""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    total_days = (end - start).days + 1
    print(f"Fetching games from {start_date} to {end_date} ({total_days} days)...")
    
    all_games = []
    seen_games = set()
    
    for day_offset in range(total_days):
        current_date = start + timedelta(days=day_offset)
        
        # Skip non-game days (no games on certain days)
        # But we'll still try to fetch
        games = fetch_scoreboard(current_date)
        
        # Add date to each game
        for game in games:
            game_id = game.get('gameId')
            if game_id and game_id not in seen_games:
                game['fetch_date'] = current_date.strftime('%Y-%m-%d')
                seen_games.add(game_id)
                all_games.append(game)
        
        # Progress
        if day_offset % 30 == 0:
            print(f"  Progress: {day_offset + 1}/{total_days} days ({len(all_games)} games)")
        
        # Rate limit
        time.sleep(0.1)
    
    print(f"\nFetched {len(all_games)} games from {total_days} days")
    return all_games

def save_games(games: list, output_path: str):
    """Save games to JSON file."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(games, f, indent=2)
    
    print(f"Saved {len(games)} games to {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch NBA games for a date range (e.g., entire season)')
    parser.add_argument('--start-date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='data/raw/season_25_26.json', help='Output JSON file path')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("FETCH SEASON GAMES")
    print("=" * 70)
    print(f"\nStart date: {args.start_date}")
    print(f"End date: {args.end_date}")
    print(f"Output: {args.output}")
    
    # Fetch games
    games = fetch_date_range(args.start_date, args.end_date)
    
    if games:
        # Save games
        save_games(games, args.output)
        
        print("\n" + "=" * 70)
        print("FETCH COMPLETE")
        print("=" * 70)
        print(f"\nGames fetched: {len(games)}")
        print(f"Saved to: {args.output}")
    else:
        print("\nNo games fetched!")

if __name__ == '__main__':
    main()
