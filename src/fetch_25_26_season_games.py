"""
Fetch recent games from the 2025-26 NBA season using game IDs.

NBA Game ID format for 2025-26 season: 00225 + game_num (4-digit zero-padding)
  - Example: 0022500686 = Game 686
  - Game 1: 0022500001
"""
import sys
sys.path.insert(0, '/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3')

import requests
import json
import time
from pathlib import Path

BOX_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{}.json"
PBP_URL = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{}.json"

def fetch_game(game_id: str, include_pbp: bool = True) -> dict:
    """Fetch box score and play-by-play for a game."""
    try:
        # Fetch box score
        box_response = requests.get(BOX_URL.format(game_id), timeout=30)
        box_response.raise_for_status()
        box_data = box_response.json()
        
        # Fetch play-by-play (optional)
        pbp_data = None
        if include_pbp:
            try:
                pbp_response = requests.get(PBP_URL.format(game_id), timeout=30)
                pbp_response.raise_for_status()
                pbp_data = pbp_response.json()
            except:
                pbp_data = None
        
        return {
            'game_id': game_id,
            'box_score': box_data,
            'play_by_play': pbp_data,
            'success': True
        }
    except Exception as e:
        return {
            'game_id': game_id,
            'error': str(e),
            'success': False
        }

def fetch_season_games(
    start_game: int,     # e.g., 680
    end_game: int,       # e.g., 686
    include_pbp: bool = True
) -> tuple:
    """Fetch games from 2025-26 season."""
    
    # Format: 00225 + 0 + game_num (4-digit padding)
    prefix = "00225"
    
    games_fetched = 0
    games_failed = 0
    
    print("=" * 70)
    print(f"FETCHING 25-26 SEASON GAMES: {start_game} to {end_game}")
    print("=" * 70)
    
    results = []
    
    for game_num in range(start_game, end_game + 1):
        # Format game ID: 00225 + 0 + 4-digit game number
        game_id = f"{prefix}0{game_num:04d}"
        
        # Fetch game
        result = fetch_game(game_id, include_pbp=include_pbp)
        results.append(result)
        
        if result['success']:
            games_fetched += 1
        else:
            games_failed += 1
        
        # Progress every 2 games
        if game_num % 2 == 0:
            print(f"  {game_num:4d}: {game_id} ({'✅' if result['success'] else '❌'})")
        
        # Rate limit
        time.sleep(0.1)
    
    print(f"\nSummary: {games_fetched} games fetched, {games_failed} failed")
    return results, games_fetched, games_failed

def save_games(results: list, box_dir: str = 'data/raw/box', pbp_dir: str = 'data/raw/pbp'):
    """Save fetched games to disk."""
    box_path = Path(box_dir)
    box_path.mkdir(parents=True, exist_ok=True)
    
    pbp_path = Path(pbp_dir)
    pbp_path.mkdir(parents=True, exist_ok=True)
    
    saved = 0
    for result in results:
        if result['success']:
            game_id = result['game_id']
            
            # Save box score
            box_file = box_path / f"{game_id}.json"
            with open(box_file, 'w') as f:
                json.dump(result['box_score'], f, indent=2)
            saved += 1
            
            # Save play-by-play (if available)
            if result['play_by_play']:
                pbp_file = pbp_path / f"{game_id}.json"
                with open(pbp_file, 'w') as f:
                    json.dump(result['play_by_play'], f, indent=2)
    
    print(f"Saved {saved} box scores to {box_dir}")
    print(f"Saved {saved} play-by-play files to {pbp_dir}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch 25-26 season games')
    parser.add_argument('--start-game', type=int, default=680, 
                       help='Starting game number (default: 680)')
    parser.add_argument('--end-game', type=int, default=686,
                       help='Ending game number (default: 686, today)')
    parser.add_argument('--skip-pbp', action='store_true',
                       help='Skip play-by-play (faster)')
    
    args = parser.parse_args()
    
    # Fetch games
    results, fetched, failed = fetch_season_games(
        start_game=args.start_game,
        end_game=args.end_game,
        include_pbp=not args.skip_pbp
    )
    
    if fetched > 0:
        # Save games
        save_games(results)
        
        print("\n" + "=" * 70)
        print("FETCH COMPLETE")
        print("=" * 70)
        print(f"\nGames fetched: {fetched}")
        print(f"Games failed: {failed}")
        print(f"\nNext steps:")
        print("  1. Build temporal features: python3 src/build_temporal_features.py")
        print("  2. Merge with halftime: python3 src/merge_temporal_halftime.py")
        print("  3. Retrain model: python3 src/train_halftime_model.py")
        print("  4. Run backtests")
    else:
        print("\nNo games fetched! The game range may not exist yet.")

if __name__ == '__main__':
    main()
