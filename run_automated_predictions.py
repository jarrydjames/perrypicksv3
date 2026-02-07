#!/usr/bin/env python3
"""Automated Prediction Scheduler

This script continuously monitors NBA games and automatically runs the appropriate
prediction models based on game state:

- Pregame: Run before games start
- Halftime: Run when games reach halftime
- Q3: Run when games complete Q3

Usage:
    python run_automated_predictions.py
    python run_automated_predictions.py --interval 300
    python run_automated_predictions.py --date 2026-02-05
    python run_automated_predictions.py --once
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.predict_api import predict_game


def fetch_scoreboard(date_str: str) -> dict:
    """Fetch scoreboard from NBA.com API."""
    date_formatted = date_str.replace('-', '')
    url = f"https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00_{date_formatted}.json"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        pass
    
    return {}


def get_game_state(game: dict) -> str:
    """
    Determine the current state of a game.
    
    Returns:
        'pregame': Game not started
        'halftime': Game at halftime (between Q2 and Q3)
        'q3': Game after Q3 (between Q3 and Q4 or Q4 completed)
        'completed': Game finished
        'other': Other state (in play, etc.)
    """
    game_status = game.get('gameStatus', 0)
    game_status_text = game.get('gameStatusText', '')
    period = game.get('period', 0)
    clock = game.get('gameClock', '')
    
    # Pregame: Game not started
    if game_status == 1 or game_status_text == 'Scheduled':
        return 'pregame'
    
    # Completed
    if game_status == 3 or game_status_text == 'Final':
        return 'completed'
    
    # Halftime: Between Q2 and Q3
    if game_status_text == 'Halftime':
        return 'halftime'
    
    # Q3: After Q3 completed (period 4 or higher)
    if period >= 4:
        return 'q3'
    
    # In play or other state
    return 'other'


def run_pregame_for_game(game_id: str, away_team: str, home_team: str) -> dict:
    """Run pregame prediction for a single game."""
    try:
        result = predict_game(
            game_input=game_id,
            mode='pregame',
            fetch_odds=False
        )
        
        if result.get('status') in ['success', 'warning']:
            total = result.get('total', 0)
            margin = result.get('margin', 0)
            winner = home_team if margin > 0 else away_team
            
            return {
                'success': True,
                'total': total,
                'margin': margin,
                'winner': winner
            }
    except Exception as e:
        pass
    
    return {'success': False, 'error': 'Prediction failed'}


def run_halftime_for_game(game_id: str, away_team: str, home_team: str) -> dict:
    """Run halftime prediction for a single game."""
    try:
        result = predict_game(
            game_input=game_id,
            mode='halftime',
            fetch_odds=False
        )
        
        if result.get('status') in ['success', 'warning']:
            h1_home = result.get('home_score', 0)
            h1_away = result.get('away_score', 0)
            pred_final_total = result.get('total', 0)
            pred_final_margin = result.get('margin', 0)
            winner = home_team if pred_final_margin > 0 else away_team
            
            pred_2h_total = pred_final_total - (h1_home + h1_away)
            pred_2h_home = (pred_2h_total + pred_final_margin) / 2
            pred_2h_away = (pred_2h_total - pred_final_margin) / 2
            
            pred_final_home = h1_home + pred_2h_home
            pred_final_away = h1_away + pred_2h_away
            
            return {
                'success': True,
                'h1_home': h1_home,
                'h1_away': h1_away,
                'pred_2h_home': pred_2h_home,
                'pred_2h_away': pred_2h_away,
                'pred_final_home': pred_final_home,
                'pred_final_away': pred_final_away,
                'pred_final_total': pred_final_total,
                'pred_final_margin': pred_final_margin,
                'winner': winner
            }
    except Exception as e:
        pass
    
    return {'success': False, 'error': 'Prediction failed'}


def run_q3_for_game(game_id: str, away_team: str, home_team: str) -> dict:
    """Run Q3 prediction for a single game."""
    try:
        result = predict_game(
            game_input=game_id,
            mode='q3',
            fetch_odds=False
        )
        
        if result.get('status') in ['success', 'warning']:
            q3_cumulative_home = result.get('home_score', 0)
            q3_cumulative_away = result.get('away_score', 0)
            q3_cumulative_total = q3_cumulative_home + q3_cumulative_away
            q3_cumulative_margin = q3_cumulative_home - q3_cumulative_away
            
            # Estimate Q4 using quarter progression heuristic
            q4_estimate_total = q3_cumulative_total * 0.32
            margin_adjustment = q3_cumulative_margin * 0.2
            q4_home = max(20, min(35, (q4_estimate_total / 2) + margin_adjustment))
            q4_away = max(20, min(35, (q4_estimate_total / 2) - margin_adjustment))
            
            pred_final_home = q3_cumulative_home + q4_home
            pred_final_away = q3_cumulative_away + q4_away
            pred_final_total = pred_final_home + pred_final_away
            pred_final_margin = pred_final_home - pred_final_away
            winner = home_team if pred_final_margin > 0 else away_team
            
            return {
                'success': True,
                'q3_cumulative_home': q3_cumulative_home,
                'q3_cumulative_away': q3_cumulative_away,
                'q4_home': q4_home,
                'q4_away': q4_away,
                'pred_final_home': pred_final_home,
                'pred_final_away': pred_final_away,
                'pred_final_total': pred_final_total,
                'pred_final_margin': pred_final_margin,
                'winner': winner
            }
    except Exception as e:
        pass
    
    return {'success': False, 'error': 'Prediction failed'}


def process_games(date_str: str, tracked_games: dict) -> dict:
    """
    Process all games and run appropriate predictions.
    
    Args:
        date_str: Date string in YYYY-MM-DD format
        tracked_games: Dict tracking which games have been processed in each state
    
    Returns:
        Updated tracked_games dict
    """
    print("=" * 80)
    print(f"CHECKING GAMES FOR {date_str} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    scoreboard = fetch_scoreboard(date_str)
    
    if not scoreboard:
        print("No scoreboard data available")
        return tracked_games
    
    if 'scoreboard' not in scoreboard or 'games' not in scoreboard['scoreboard']:
        print("No games found")
        return tracked_games
    
    games = scoreboard['scoreboard']['games']
    
    if not games:
        print("No games scheduled")
        return tracked_games
    
    print(f"Found {len(games)} games")
    print()
    
    for game in games:
        game_id = game.get('gameId')
        home_team = game.get('homeTeam', {}).get('teamTricode', 'HOME')
        away_team = game.get('awayTeam', {}).get('teamTricode', 'AWAY')
        game_status_text = game.get('gameStatusText', '')
        period = game.get('period', 0)
        
        state = get_game_state(game)
        
        # Initialize game entry if not exists
        if game_id not in tracked_games:
            tracked_games[game_id] = {
                'pregame_run': False,
                'halftime_run': False,
                'q3_run': False,
                'game_id': game_id,
                'away_team': away_team,
                'home_team': home_team
            }
        
        print(f"{away_team} @ {home_team} ({game_id}): {game_status_text} (Period: {period}) - State: {state}")
        
        # Run pregame if not already run and game is pregame
        if state == 'pregame' and not tracked_games[game_id]['pregame_run']:
            print(f"  → Running PREGAME prediction...")
            result = run_pregame_for_game(game_id, away_team, home_team)
            if result['success']:
                print(f"    ✓ Total: {result['total']:.1f}, Margin: {result['margin']:+.1f}, Winner: {result['winner']}")
                tracked_games[game_id]['pregame_run'] = True
            else:
                print(f"    ✗ Failed: {result.get('error', 'Unknown error')}")
        
        # Run halftime if not already run and game is at halftime
        elif state == 'halftime' and not tracked_games[game_id]['halftime_run']:
            print(f"  → Running HALFTIME prediction...")
            result = run_halftime_for_game(game_id, away_team, home_team)
            if result['success']:
                print(f"    ✓ H1: {result['h1_away']}-{result['h1_home']}")
                print(f"    ✓ Pred Final: {result['pred_final_away']:.1f}-{result['pred_final_home']:.1f} (Total: {result['pred_final_total']:.1f})")
                print(f"    ✓ Margin: {result['pred_final_margin']:+.1f}, Winner: {result['winner']}")
                tracked_games[game_id]['halftime_run'] = True
            else:
                print(f"    ✗ Failed: {result.get('error', 'Unknown error')}")
        
        # Run Q3 if not already run and game is after Q3
        elif state == 'q3' and not tracked_games[game_id]['q3_run']:
            print(f"  → Running Q3 prediction...")
            result = run_q3_for_game(game_id, away_team, home_team)
            if result['success']:
                print(f"    ✓ Q3 Cum: {result['q3_cumulative_away']:.1f}-{result['q3_cumulative_home']:.1f}")
                print(f"    ✓ Est Q4: {result['q4_away']:.1f}-{result['q4_home']:.1f}")
                print(f"    ✓ Pred Final: {result['pred_final_away']:.1f}-{result['pred_final_home']:.1f} (Total: {result['pred_final_total']:.1f})")
                print(f"    ✓ Margin: {result['pred_final_margin']:+.1f}, Winner: {result['winner']}")
                tracked_games[game_id]['q3_run'] = True
            else:
                print(f"    ✗ Failed: {result.get('error', 'Unknown error')}")
        
        elif state == 'other':
            print(f"  → In progress, waiting for next checkpoint...")
        
        elif state == 'completed':
            print(f"  → Game completed")
        
        print()
    
    # Print summary
    pregame_count = sum(1 for g in tracked_games.values() if g['pregame_run'])
    halftime_count = sum(1 for g in tracked_games.values() if g['halftime_run'])
    q3_count = sum(1 for g in tracked_games.values() if g['q3_run'])
    
    print("=" * 80)
    print(f"SUMMARY: {pregame_count} pregame, {halftime_count} halftime, {q3_count} Q3 predictions run")
    print("=" * 80)
    print()
    
    return tracked_games


def main():
    parser = argparse.ArgumentParser(
        description="Automated prediction scheduler - runs models based on game state",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Continuous monitoring (checks every 5 minutes)
  python run_automated_predictions.py
  
  # Check every 10 minutes
  python run_automated_predictions.py --interval 600
  
  # Specific date
  python run_automated_predictions.py --date 2026-02-05
  
  # Run once and exit
  python run_automated_predictions.py --once

Game States:
  - Pregame: Before games start (run pregame model)
  - Halftime: Between Q2 and Q3 (run halftime model)
  - Q3: After Q3 completes (run Q3 model)
  - Completed: Game finished (no predictions)
"""
    )
    
    parser.add_argument(
        '--date', '-d',
        default=None,
        help='Date in YYYY-MM-DD format (default: today)',
    )
    
    parser.add_argument(
        '--interval', '-i',
        type=int,
        default=300,
        help='Check interval in seconds (default: 300 = 5 minutes)',
    )
    
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run once and exit (do not loop)',
    )
    
    args = parser.parse_args()
    
    # Determine date
    if args.date:
        date_str = args.date
    else:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    # Track which games have been processed
    tracked_games = {}
    
    print()
    print("=" * 80)
    print("AUTOMATED PREDICTION SCHEDULER")
    print("=" * 80)
    print(f"Date: {date_str}")
    print(f"Check interval: {args.interval} seconds")
    print(f"Mode: {'Single run' if args.once else 'Continuous monitoring'}")
    print("=" * 80)
    print()
    
    try:
        if args.once:
            # Run once
            tracked_games = process_games(date_str, tracked_games)
        else:
            # Continuous monitoring
            while True:
                tracked_games = process_games(date_str, tracked_games)
                
                # Check if all games are completed
                all_completed = all(
                    g['pregame_run'] and g['halftime_run'] and g['q3_run']
                    for g in tracked_games.values()
                )
                
                if all_completed and tracked_games:
                    print("All games processed. Exiting.")
                    break
                
                # Wait for next check
                print(f"Waiting {args.interval} seconds until next check...")
                print()
                time.sleep(args.interval)
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    
    except Exception as e:
        print(f"\n\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
