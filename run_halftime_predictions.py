"""Halftime Predictions Runner

Fetches games for a given date and runs halftime predictions on all games.
Displays H1 scores, predicted 2H scores, predicted final scores,
margins, and projected winners.

Usage:
    python run_halftime_predictions.py [date]

    python run_halftime_predictions.py 2026-02-05
    python run_halftime_predictions.py  # Uses today's date
"""

import argparse
import sys
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
                    home_team = game['competitions'][0]['competitors'][1]['team']['abbreviation']
                    away_team = game['competitions'][0]['competitors'][0]['team']['abbreviation']
                    games.append({
                        'game_id': game_id,
                        'home_team': home_team,
                        'away_team': away_team
                    })
            return games
    except Exception as e:
        print(f"Warning: Could not fetch games for {date_str}: {e}")
    
    return []


def run_halftime_predictions(date_str: str, game_ids: list = None):
    """Run halftime predictions for all games on given date."""
    print("=" * 100)
    print(f"HALFTIME PREDICTIONS FOR {date_str}")
    print("=" * 100)
    print()
    
    # Either use provided game_ids or fetch from API
    games = []
    if game_ids:
        # Use provided game IDs
        for game_id in game_ids:
            # Try to extract team names from a mapping or use placeholders
            games.append({
                'game_id': game_id,
                'home_team': 'HOME',
                'away_team': 'AWAY'
            })
        print(f"Using {len(games)} provided game IDs")
    else:
        # Fetch games from API
        games = fetch_games_for_date(date_str)
        if not games:
            print(f"No games found for {date_str}")
            print("This could mean:")
            print("  1. No games scheduled for this date")
            print("  2. Games are not yet started/completed (future date)")
            print("  3. API is rate-limited (try again later)")
            return
        print(f"Found {len(games)} games for {date_str}")
    
    print()
    
    predictions = []
    
    for i, game in enumerate(games, 1):
        game_id = game['game_id']
        away_team = game['away_team']
        home_team = game['home_team']
        
        print(f"[{i}/{len(games)}] Processing {away_team} @ {home_team} ({game_id})")
        
        try:
            # Add delay between requests to avoid rate limiting
            if i > 1:
                import time
                time.sleep(1)
            
            result = predict_game(
                game_input=game_id,
                mode='halftime',
                fetch_odds=False
            )
            
            if result.get('status') in ['success', 'warning']:
                h1_home = result.get('home_score', 0)
                h1_away = result.get('away_score', 0)
                h1_total = h1_home + h1_away
                
                pred_final_total = result.get('total', 0)
                pred_final_margin = result.get('margin', 0)
                
                # Calculate predicted 2H scores
                pred_2h_total = pred_final_total - h1_total
                pred_2h_home = (pred_2h_total + pred_final_margin) / 2
                pred_2h_away = (pred_2h_total - pred_final_margin) / 2
                
                pred_final_home = h1_home + pred_2h_home
                pred_final_away = h1_away + pred_2h_away
                
                # Determine winner
                winner = home_team if pred_final_margin > 0 else away_team
                
                # Use actual team names from result if available
                if result.get('home_name') and result.get('away_name'):
                    home_team = result['home_name']
                    away_team = result['away_name']
                    winner = home_team if pred_final_margin > 0 else away_team
                
                predictions.append({
                    'game_id': game_id,
                    'away': away_team,
                    'home': home_team,
                    'h1_away': h1_away,
                    'h1_home': h1_home,
                    'h1_total': h1_total,
                    'pred_2h_away': pred_2h_away,
                    'pred_2h_home': pred_2h_home,
                    'pred_2h_total': pred_2h_total,
                    'pred_final_away': pred_final_away,
                    'pred_final_home': pred_final_home,
                    'pred_final_total': pred_final_total,
                    'pred_final_margin': pred_final_margin,
                    'winner': winner
                })
                
                print(f"  ✓ H1: {h1_away}-{h1_home}")
                print(f"  ✓ Pred 2H: {pred_2h_away:.1f}-{pred_2h_home:.1f} (Total: {pred_2h_total:.1f})")
                print(f"  ✓ Pred Final: {pred_final_away:.1f}-{pred_final_home:.1f} (Total: {pred_final_total:.1f})")
                print(f"  ✓ Margin: {pred_final_margin:+.1f} | Winner: {winner}")
            else:
                print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
                # Add failed game with null predictions
                predictions.append({
                    'game_id': game_id,
                    'away': away_team,
                    'home': home_team,
                    'h1_away': None,
                    'h1_home': None,
                    'h1_total': None,
                    'pred_2h_away': None,
                    'pred_2h_home': None,
                    'pred_2h_total': None,
                    'pred_final_away': None,
                    'pred_final_home': None,
                    'pred_final_total': None,
                    'pred_final_margin': None,
                    'winner': 'ERROR'
                })
        except Exception as e:
            print(f"  ✗ Error: {e}")
            # Add failed game with null predictions
            predictions.append({
                'game_id': game_id,
                'away': away_team,
                'home': home_team,
                'h1_away': None,
                'h1_home': None,
                'h1_total': None,
                'pred_2h_away': None,
                'pred_2h_home': None,
                'pred_2h_total': None,
                'pred_final_away': None,
                'pred_final_home': None,
                'pred_final_total': None,
                'pred_final_margin': None,
                'winner': 'ERROR'
            })
        
        print()
    
    # Print summary table
    print("=" * 100)
    print("HALFTIME PREDICTIONS SUMMARY")
    print("=" * 100)
    print()
    print(f"{'Game ID':<12} | {'Away':<6} @ {'Home':<6} | {'H1':<10} | {'Pred 2H':<11} | {'Pred Final':<15} | {'Margin':<8} | {'Winner':<8}")
    print("-" * 100)
    
    for pred in predictions:
        if pred['h1_away'] is not None:
            h1 = f"{pred['h1_away']}-{pred['h1_home']}"
            pred_2h = f"{pred['pred_2h_away']:.1f}-{pred['pred_2h_home']:.1f}"
            pred_final = f"{pred['pred_final_away']:.1f}-{pred['pred_final_home']:.1f}"
            margin = f"{pred['pred_final_margin']:+.1f}"
        else:
            h1 = "N/A"
            pred_2h = "N/A"
            pred_final = "N/A"
            margin = "N/A"
        
        print(f"{pred['game_id']:<12} | {pred['away']:<6} @ {pred['home']:<6} | {h1:<10} | {pred_2h:<11} | {pred_final:<15} | {margin:<8} | {pred['winner']:<8}")
    
    print()
    print("=" * 100)
    print(f"Total games predicted: {len([p for p in predictions if p['h1_away'] is not None])}/{len(games)}")
    print(f"Model: XGBoost (MAE: 7.920 H2 Total, 6.029 H2 Margin)")
    print("=" * 100)
    
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Run halftime predictions for NBA games on a given date",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predictions for today's games
  python run_halftime_predictions.py
  
  # Predictions for a specific date
  python run_halftime_predictions.py 2026-02-05
  
  # Predictions for specific game IDs (useful for testing)
  python run_halftime_predictions.py --games 0022500733 0022500734 0022500735
"""
    )
    
    parser.add_argument(
        'date',
        nargs='?',
        help='Date in YYYY-MM-DD format (default: today)',
    )
    
    parser.add_argument(
        '--games', '-g',
        nargs='+',
        help='Specific game IDs to predict (overrides date fetching)',
    )
    
    args = parser.parse_args()
    
    # Determine date
    if args.date:
        date_str = args.date
    else:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    # Run predictions
    if args.games:
        run_halftime_predictions(date_str, game_ids=args.games)
    else:
        run_halftime_predictions(date_str)



if __name__ == '__main__':
    main()
