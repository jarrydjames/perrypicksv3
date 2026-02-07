#!/usr/bin/env python3
"""Pregame Predictions Runner

Fetches games for a given date and runs pregame predictions on all games.
Displays projected final totals, margins, and predicted winners.

Usage:
    python run_pregame_predictions.py [date]
    python run_pregame_predictions.py 2026-02-05
    python run_pregame_predictions.py  # Uses today's date
    python run_pregame_predictions.py --games 0022500733 0022500734 --teams WAS:DET BKN:ORL
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.predict_api import predict_game


def fetch_games_for_date(date_str: str) -> list:
    """Fetch games for a specific date from NBA.com API."""
    # Try NBA.com CDN first
    date_formatted = date_str.replace('-', '')
    url = f"https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00_{date_formatted}.json"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            games = []
            if 'scoreboard' in data and 'games' in data['scoreboard']:
                for game in data['scoreboard']['games']:
                    game_id = game['gameId']
                    home_team = game['homeTeam']['teamTricode']
                    away_team = game['awayTeam']['teamTricode']
                    games.append({
                        'game_id': game_id,
                        'home_team': home_team,
                        'away_team': away_team
                    })
            return games
    except Exception as e:
        pass
    
    # Try ESPN API as fallback
    date_formatted = date_str.replace('-', '')
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_formatted}"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            games = []
            if 'events' in data:
                for game in data['events']:
                    game_id = game['id']
                    competitors = game.get('competitions', [{}])[0].get('competitors', [])
                    
                    if len(competitors) >= 2:
                        # Determine home/away
                        if competitors[0].get('homeAway') == 'home':
                            home_team = competitors[0].get('team', {}).get('abbreviation', 'HOME')
                            away_team = competitors[1].get('team', {}).get('abbreviation', 'AWAY')
                        else:
                            home_team = competitors[1].get('team', {}).get('abbreviation', 'HOME')
                            away_team = competitors[0].get('team', {}).get('abbreviation', 'AWAY')
                    else:
                        home_team = 'HOME'
                        away_team = 'AWAY'
                    
                    games.append({
                        'game_id': game_id,
                        'home_team': home_team,
                        'away_team': away_team
                    })
            return games
    except Exception as e:
        print(f"Warning: Could not fetch games for {date_str}: {e}")
    
    return []


def run_pregame_predictions(date_str: str, game_ids: list = None, teams: dict = None):
    """Run pregame predictions for all games on given date.
    
    Args:
        date_str: Date string in YYYY-MM-DD format
        game_ids: Optional list of specific game IDs
        teams: Optional dict mapping game_id to (away_team, home_team) tuple
    """
    print("=" * 100)
    print(f"PREGAME PREDICTIONS FOR {date_str}")
    print("=" * 100)
    print()
    
    # Either use provided game_ids or fetch from API
    games = []
    if game_ids:
        # Use provided game IDs
        for game_id in game_ids:
            # Check if team mapping provided
            if teams and game_id in teams:
                away_team, home_team = teams[game_id]
            else:
                away_team, home_team = 'AWAY', 'HOME'
            
            games.append({
                'game_id': game_id,
                'home_team': home_team,
                'away_team': away_team
            })
    else:
        games = fetch_games_for_date(date_str)
    
    if not games:
        print("No games found for this date.")
        print()
        print("=" * 100)
        return
    
    print(f"Found {len(games)} games")
    print()
    
    # Run predictions
    results = []
    for i, game in enumerate(games, 1):
        game_id = game['game_id']
        home_team = game['home_team']
        away_team = game['away_team']
        
        print(f"[{i}/{len(games)}] Predicting {away_team} @ {home_team} ({game_id})...")
        
        # Run prediction with pregame mode
        result = predict_game(
            game_input=game_id,
            mode='pregame',  # Force pregame model
            home_team=home_team,
            away_team=away_team,
            fetch_odds=False  # Skip odds to save time
        )
        
        if result.get('status') == 'success':
            results.append(result)
            print(f"  ✓ Predicted: {result.get('away_score', 0):.1f} - {result.get('home_score', 0):.1f} (Total: {result.get('total_score', 0):.1f})")
        else:
            print(f"  ✗ Error: {result.get('error', 'Unknown error')}")
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
        
        print()
    
    # Print summary
    if results:
        print("=" * 100)
        print(f"SUMMARY ({len(results)}/{len(games)} predictions successful)")
        print("=" * 100)
        print()
        print(f"{'Game ID':<12} | {'Away':<6} @ {'Home':<6} | {'Pred Total':<12} | {'Pred Margin':<12} | {'Winner':<6}")
        print("-" * 100)
        
        for result in results:
            game_id = result.get('game_id', 'N/A')[:12]
            away_team = result.get('away_team', 'AWAY')[:6]
            home_team = result.get('home_team', 'HOME')[:6]
            pred_total = result.get('total_score', 0)
            pred_margin = result.get('margin', 0)
            winner = result.get('predicted_winner', 'Unknown')[:6]
            
            # Format margin with + or - sign
            if pred_margin > 0:
                margin_str = f"+{pred_margin:.1f}"
            else:
                margin_str = f"{pred_margin:.1f}"
            
            print(f"{game_id:<12} | {away_team:<6} @ {home_team:<6} | {pred_total:<12.1f} | {margin_str:<12} | {winner:<6}")
        
        print()
        print("=" * 100)
    else:
        print("No successful predictions.")
        print()
        print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Run pregame predictions for NBA games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict games for today
  python run_pregame_predictions.py
  
  # Predict games for specific date
  python run_pregame_predictions.py 2026-02-05
  
  # Predict specific games
  python run_pregame_predictions.py --games 0022500733 0022500734 --teams WAS:DET BKN:ORL

Note: Pregame predictions project final game scores before tipoff.
      Best run 1-2 hours before game time.

Team format: AWAY:HOME
  Example: WAS:DET means Washington @ Detroit
"""
    )
    
    parser.add_argument(
        'date',
        nargs='?',
        default=None,
        help='Date in YYYY-MM-DD format (default: today)',
    )
    
    parser.add_argument(
        '--games', '-g',
        nargs='+',
        default=None,
        help='Specific game IDs to predict (overrides date)',
    )
    
    parser.add_argument(
        '--teams', '-t',
        nargs='+',
        default=None,
        help='Team mapping for games (format: AWAY:HOME)',
    )
    
    args = parser.parse_args()
    
    # Determine date
    if args.games:
        # Use specific games, date not needed
        date_str = datetime.now().strftime('%Y-%m-%d')
        game_ids = args.games
    elif args.date:
        date_str = args.date
        game_ids = None
    else:
        date_str = datetime.now().strftime('%Y-%m-%d')
        game_ids = None
    
    # Parse teams if provided
    teams = None
    if args.teams and args.games:
        teams = {}
        for i, team_str in enumerate(args.teams):
            if i < len(args.games):
                if ':' in team_str:
                    parts = team_str.split(':')
                    if len(parts) == 2:
                        away_team = parts[0].upper().strip()
                        home_team = parts[1].upper().strip()
                        teams[args.games[i]] = (away_team, home_team)
    
    # Run predictions
    run_pregame_predictions(date_str, game_ids, teams)


if __name__ == '__main__':
    main()
