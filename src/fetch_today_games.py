"""
Fetch today's NBA games for temporal feature updates.

This script pulls the last N days of games from the NBA CDN,
which is used to calculate rolling features that are always current-season focused.
"""
import sys
import os
sys.path.insert(0, '/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3')

import requests
import json
from datetime import datetime, timedelta

# API Endpoints
SCHEDULE_URL = "https://cdn.nba.com/static/json/liveData/scoreboard_todays_league_v2.json"
SCOREBOARD_URL_TEMPLATE = "https://cdn.nba.com/static/json/liveData/scoreboard_todays_league_v2_{}.json"

def fetch_scoreboard(date: datetime) -> dict:
    """Fetch NBA scoreboard for a specific date."""
    # Format: YYYYMMDD for NBA API
    date_str = date.strftime("%Y%m%d")
    url = SCOREBOARD_URL_TEMPLATE.format(date_str)
    
    print(f"Fetching scoreboard for {date.strftime('%Y-%m-%d')}...")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        scoreboard = data.get('scoreboard', {}).get('games', [])
        
        print(f"  Found {len(scoreboard)} games")
        return scoreboard
    except Exception as e:
        print(f"  Error fetching scoreboard: {e}")
        return []

def fetch_games_around_date(days_before: int = 2) -> list:
    """Fetch games from multiple days around a target date."""
    today = datetime.now()
    target_date = today - timedelta(days=days_before)
    
    print(f"\nFetching games for: {target_date.strftime('%Y-%m-%d')}")
    print(f"Searching {days_before * 2 + 1} days ({target_date - timedelta(days=days_before)} to {today + timedelta(days=1)})...")
    
    all_games = []
    
    # Fetch target date and surrounding days
    for days_offset in range(-days_before, days_before + 2):
        check_date = target_date + timedelta(days=days_offset)
        games = fetch_scoreboard(check_date)
        
        # Add date to each game
        for game in games:
            game['fetch_date'] = target_date.strftime('%Y-%m-%d')
        
        all_games.extend(games)
    
    # Deduplicate games (same game might appear in multiple days)
    seen = set()
    unique_games = []
    for game in all_games:
        game_id = game.get('gameId')
        if game_id and game_id not in seen:
            seen.add(game_id)
            unique_games.append(game)
    
    print(f"\nTotal unique games: {len(unique_games)}")
    
    return unique_games

def save_games_to_cache(games: list, cache_path: str = 'data/raw/todays_games.json'):
    """Save games to cache for temporal feature builder."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    with open(cache_path, 'w') as f:
        json.dump(games, f, indent=2)
    
    print(f"Saved {len(games)} games to {cache_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch today\'s NBA games for temporal features')
    parser.add_argument('--date', type=str, default=None, help='Date in YYYY-MM-DD format')
    parser.add_argument('--days-before', type=int, default=2, help='Days before today to focus on')
    parser.add_argument('--days-after', type=int, default=1, help='Days after today to include')
    parser.add_argument('--cache-only', action='store_true', help='Only save to cache, don\'t fetch')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Get date
    if args.date:
        target_date = datetime.strptime(args.date, '%Y-%m-%d')
    else:
        target_date = datetime.now()
    
    print("=" * 70)
    print("FETCH TODAY'S GAMES - FOR TEMPORAL FEATURE REFRESH")
    print("=" * 70)
    print(f"\nTarget date: {target_date.strftime('%Y-%m-%d')}")
    print(f"Days before: {args.days_before}, Days after: {args.days_after}")
    
    # Fetch games
    games = fetch_games_around_date(days_before=args.days_before)
    
    if games:
        # Save to cache
        save_games_to_cache(games)
        
        # If output specified, also save there
        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(games, f, indent=2)
            print(f"Also saved to: {args.output}")
    
    print("\n" + "=" * 70)
    print("✅ FETCH COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    main()
