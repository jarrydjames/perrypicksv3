"""Q3 Predictions Runner

Fetches games for a given date and runs Q3 predictions on all games.
Displays Q3 scores, predicted Q3 totals and margins, and game progress.

Q3 Model Information:
- Champion: Neural Network (R²: 0.538 Q3 Total, 0.685 Q3 Margin)
- MAE: 8.339 (Q3 Total), 6.581 (Q3 Margin)
- Features: 10 efficiency stats (efg, ftr, tpar, tor, orbp for both teams)
- Predicts: Q3 quarter totals and margins (NOT final game)

Usage:
    python run_q3_predictions.py [date]

    python run_q3_predictions.py 2026-02-05
    python run_q3_predictions.py  # Uses today's date
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
import requests

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


def run_q3_predictions(date_str: str, game_ids: list = None):
    """Run Q3 predictions for all games on given date."""
    print("=" * 100)
    print("Q3 PREDICTIONS FOR " + date_str)
    print("=" * 100)
    print()
    print("Model: Q3 Neural Network Champion")
    print("       (R²: 0.538 Q3 Total, 0.685 Q3 Margin)")
    print("       (MAE: 8.339 Q3 Total, 6.581 Q3 Margin)")
    print("=" * 100)
    print()
    
    # Either use provided game_ids or fetch from API
    games = []
    if game_ids:
        # Use provided game IDs
        for game_id in game_ids:
            games.append({
                'game_id': game_id,
                'home_team': 'HOME',
                'away_team': 'AWAY',
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
                mode='q3',
                fetch_odds=False
            )
            
            if result.get('status') in ['success', 'warning']:
                q3_home = result.get('home_score', 0)
                q3_away = result.get('away_score', 0)
                q3_actual_total = q3_home + q3_away
                q3_actual_margin = q3_home - q3_away
                
                q3_pred_total = result.get('total', 0)
                q3_pred_margin = result.get('margin', 0)
                
                # Note: Q3 model predicts Q3 quarter statistics
                # The 'total' and 'margin' fields are predicted Q3 values
                # The 'home_score' and 'away_score' are actual Q3 values
                
                # Q3 leader
                q3_leader = home_team if q3_actual_margin > 0 else away_team
                
                # Use actual team names from result if available
                if result.get('home_name') and result.get('away_name'):
                    home_team = result['home_name']
                    away_team = result['away_name']
                    q3_leader = home_team if q3_actual_margin > 0 else away_team
                
                predictions.append({
                    'game_id': game_id,
                    'away': away_team,
                    'home': home_team,
                    'q3_away': q3_away,
                    'q3_home': q3_home,
                    'q3_actual_total': q3_actual_total,
                    'q3_actual_margin': q3_actual_margin,
                    'q3_pred_total': q3_pred_total,
                    'q3_pred_margin': q3_pred_margin,
                    'q3_total_error': q3_pred_total - q3_actual_total,
                    'q3_margin_error': q3_pred_margin - q3_actual_margin,
                    'q3_leader': q3_leader,
                })
                
                print(f"  ✓ Q3 Actual: {q3_away:.1f}-{q3_home:.1f} (Total: {q3_actual_total:.1f})")
                print(f"  ✓ Q3 Pred: Total={q3_pred_total:.1f}, Margin={q3_pred_margin:+.1f}")
                print(f"  ✓ Q3 Leader: {q3_leader}")
            else:
                print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
                # Add failed game with null predictions
                predictions.append({
                    'game_id': game_id,
                    'away': away_team,
                    'home': home_team,
                    'q3_away': None,
                    'q3_home': None,
                    'q3_actual_total': None,
                    'q3_actual_margin': None,
                    'q3_pred_total': None,
                    'q3_pred_margin': None,
                    'q3_total_error': None,
                    'q3_margin_error': None,
                    'q3_leader': 'ERROR',
                })
        except Exception as e:
            print(f"  ✗ Error: {e}")
            # Add failed game with null predictions
            predictions.append({
                'game_id': game_id,
                'away': away_team,
                'home': home_team,
                'q3_away': None,
                'q3_home': None,
                'q3_actual_total': None,
                'q3_actual_margin': None,
                'q3_pred_total': None,
                'q3_pred_margin': None,
                'q3_total_error': None,
                'q3_margin_error': None,
                'q3_leader': 'ERROR',
            })
        
        print()
    
    # Print summary table
    print("=" * 100)
    print("Q3 PREDICTIONS SUMMARY")
    print("=" * 100)
    print()
    print(f"{'Game ID':<12} | {'Away':<6} @ {'Home':<6} | {'Q3 Actual':<15} | {'Q3 Pred':<18} | {'Error':<8} | {'Leader':<8}")
    print("-" * 100)
    
    for pred in predictions:
        if pred['q3_away'] is not None:
            q3_actual = f"{pred['q3_away']:.1f}-{pred['q3_home']:.1f}"
            q3_pred = f"Total: {pred['q3_pred_total']:.1f}, Margin: {pred['q3_pred_margin']:+.1f}"
            error = f"{pred['q3_total_error']:+.1f}"
        else:
            q3_actual = "N/A"
            q3_pred = "N/A"
            error = "N/A"
        
        print(f"{pred['game_id']:<12} | {pred['away']:<6} @ {pred['home']:<6} | {q3_actual:<15} | {q3_pred:<18} | {error:<8} | {pred['q3_leader']:<8}")
    
    print()
    print("=" * 100)
    print(f"Total games predicted: {len([p for p in predictions if p['q3_away'] is not None])}/{len(games)}")
    print("Model: Q3 Neural Network (R²: 0.538 Q3 Total, 0.685 Q3 Margin)")
    print("Note: Q3 model predicts Q3 quarter statistics, NOT final game outcomes")
    print("=" * 100)
    
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Run Q3 predictions for NBA games on a given date",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predictions for today's games
  python run_q3_predictions.py
  
  # Predictions for a specific date
  python run_q3_predictions.py 2026-02-05
  
  # Predictions for specific game IDs (useful for testing)
  python run_q3_predictions.py --games 0022500733 0022500734 0022500735

Note: Q3 model predicts Q3 quarter totals and margins (not final game).
      It's used for mid-game analysis and in-game betting.
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
        run_q3_predictions(date_str, game_ids=args.games)
    else:
        run_q3_predictions(date_str)


if __name__ == '__main__':
    main()
