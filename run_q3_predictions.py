"""Q3 Predictions Runner

Fetches games for a given date and runs Q3 predictions on all games.
Displays Q3 scores, predicted final scores, margins, and winners.

Q3 Model Information:
- Champion: Neural Network (R²: 0.538 Q3 Total, 0.685 Q3 Margin)
- MAE: 8.339 (Q3 Total), 6.581 (Q3 Margin)
- Features: 10 efficiency stats (efg, ftr, tpar, tor, orbp for both teams)
  plus Q3 statistics (q3_total, q3_margin, q3_events, etc.)

Prediction Logic:
- Q3 model predicts Q3 cumulative scores (H1+H2+Q3)
- Estimates Q4 using team efficiency and Q3 stats
- Projects final game scores and margins

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


def estimate_q4_from_q3_state(q3_cumulative_total: float, q3_margin: float) -> tuple:
    """
    Estimate Q4 scores based on Q3 cumulative totals and margins.
    
    Heuristic based on typical NBA quarter progression:
    - Q3 cumulative ≈ 170-180 points (H1+H2+Q3)
    - Final game ≈ 220-230 points
    - Q4 ≈ 45-55 points (final - Q3_cumulative)
    
    Adjusts Q4 distribution based on Q3 margin.
    
    Args:
        q3_cumulative_total: Combined score after Q3
        q3_margin: Home - Away margin after Q3
    
    Returns:
        (q4_home, q4_away) estimated Q4 scores
    """
    # Typical ratio: Final ≈ Q3_cumulative * 1.32
    # So Q4 ≈ Q3_cumulative * 0.32
    q4_estimate_total = q3_cumulative_total * 0.32
    
    # Base Q4 for each team (half of estimate)
    q4_home_base = q4_estimate_total / 2
    q4_away_base = q4_estimate_total / 2
    
    # Adjust based on Q3 margin (momentum carries forward slightly)
    # If home is up by 10 at Q3, give them +2 in Q4 estimate
    margin_adjustment = q3_margin * 0.2
    
    q4_home = q4_home_base + margin_adjustment
    q4_away = q4_away_base - margin_adjustment
    
    # Ensure reasonable bounds (typical NBA quarter: 20-35 per team)
    q4_home = max(20, min(35, q4_home))
    q4_away = max(20, min(35, q4_away))
    
    return q4_home, q4_away


def run_q3_predictions(date_str: str, game_ids: list = None):
    """Run Q3 predictions for all games on given date."""
    print("=" * 100)
    print("Q3 PREDICTIONS FOR " + date_str)
    print("=" * 100)
    print()
    print("Model: Q3 Neural Network Champion")
    print("       (R²: 0.538 Q3 Total, 0.685 Q3 Margin)")
    print("       (MAE: 8.339 Q3 Total, 6.581 Q3 Margin)")
    print("")
    print("Prediction Logic:")
    print("  1. Q3 model predicts Q3 cumulative scores (H1+H2+Q3)")
    print("  2. Estimates Q4 using Q3 cumulative totals and margin")
    print("  3. Projects final game scores, margins, and winners")
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
                # Q3 cumulative scores (H1 + H2 + Q3)
                q3_cumulative_home = result.get('home_score', 0)
                q3_cumulative_away = result.get('away_score', 0)
                q3_cumulative_total = q3_cumulative_home + q3_cumulative_away
                q3_cumulative_margin = q3_cumulative_home - q3_cumulative_away
                
                # Estimate Q4 using Q3 cumulative totals and margin
                q4_home, q4_away = estimate_q4_from_q3_state(
                    q3_cumulative_total, q3_cumulative_margin
                )
                q4_total = q4_home + q4_away
                
                # Project final scores
                pred_final_home = q3_cumulative_home + q4_home
                pred_final_away = q3_cumulative_away + q4_away
                pred_final_total = pred_final_home + pred_final_away
                pred_final_margin = pred_final_home - pred_final_away
                
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
                    'q3_cumulative_home': q3_cumulative_home,
                    'q3_cumulative_away': q3_cumulative_away,
                    'q3_cumulative_total': q3_cumulative_total,
                    'q3_cumulative_margin': q3_cumulative_margin,
                    'q4_home': q4_home,
                    'q4_away': q4_away,
                    'q4_total': q4_total,
                    'pred_final_home': pred_final_home,
                    'pred_final_away': pred_final_away,
                    'pred_final_total': pred_final_total,
                    'pred_final_margin': pred_final_margin,
                    'winner': winner,
                })
                
                print(f"  ✓ Q3 Cumulative: {q3_cumulative_away:.1f}-{q3_cumulative_home:.1f}")
                print(f"  ✓ Estimated Q4: {q4_away:.1f}-{q4_home:.1f} (Total: {q4_total:.1f})")
                print(f"  ✓ Predicted Final: {pred_final_away:.1f}-{pred_final_home:.1f} (Total: {pred_final_total:.1f})")
                print(f"  ✓ Final Margin: {pred_final_margin:+.1f} | Winner: {winner}")
            else:
                print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
                # Add failed game with null predictions
                predictions.append({
                    'game_id': game_id,
                    'away': away_team,
                    'home': home_team,
                    'q3_cumulative_home': None,
                    'q3_cumulative_away': None,
                    'q3_cumulative_total': None,
                    'q3_cumulative_margin': None,
                    'q4_home': None,
                    'q4_away': None,
                    'q4_total': None,
                    'pred_final_home': None,
                    'pred_final_away': None,
                    'pred_final_total': None,
                    'pred_final_margin': None,
                    'winner': 'ERROR',
                })
        except Exception as e:
            print(f"  ✗ Error: {e}")
            # Add failed game with null predictions
            predictions.append({
                'game_id': game_id,
                'away': away_team,
                'home': home_team,
                'q3_cumulative_home': None,
                'q3_cumulative_away': None,
                'q3_cumulative_total': None,
                'q3_cumulative_margin': None,
                'q4_home': None,
                'q4_away': None,
                'q4_total': None,
                'pred_final_home': None,
                'pred_final_away': None,
                'pred_final_total': None,
                'pred_final_margin': None,
                'winner': 'ERROR',
            })
        
        print()
    
    # Print summary table
    print("=" * 100)
    print("Q3 PREDICTIONS SUMMARY")
    print("=" * 100)
    print()
    print(f"{'Game ID':<12} | {'Away':<6} @ {'Home':<6} | {'Q3 Cum':<12} | {'Est Q4':<13} | {'Pred Final':<18} | {'Margin':<8} | {'Winner':<8}")
    print("-" * 100)
    
    for pred in predictions:
        if pred['q3_cumulative_away'] is not None:
            q3_cum = f"{pred['q3_cumulative_away']:.1f}-{pred['q3_cumulative_home']:.1f}"
            est_q4 = f"{pred['q4_away']:.1f}-{pred['q4_home']:.1f}"
            pred_final = f"{pred['pred_final_away']:.1f}-{pred['pred_final_home']:.1f}"
            margin = f"{pred['pred_final_margin']:+.1f}"
        else:
            q3_cum = "N/A"
            est_q4 = "N/A"
            pred_final = "N/A"
            margin = "N/A"
        
        print(f"{pred['game_id']:<12} | {pred['away']:<6} @ {pred['home']:<6} | {q3_cum:<12} | {est_q4:<13} | {pred_final:<18} | {margin:<8} | {pred['winner']:<8}")
    
    print()
    print("=" * 100)
    print(f"Total games predicted: {len([p for p in predictions if p['q3_cumulative_away'] is not None])}/{len(games)}")
    print("Model: Q3 Neural Network (R²: 0.538 Q3 Total, 0.685 Q3 Margin)")
    print("Prediction: Q3 cumulative + estimated Q4 (typical quarter progression)")
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

Prediction Logic:
  - Q3 model predicts cumulative scores after Q3 (H1+H2+Q3)
  - Estimates Q4 using typical quarter progression (final ≈ Q3 * 1.32)
  - Projects final game scores, margins, and winners
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
