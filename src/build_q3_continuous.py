"""
Continuous Q3 dataset builder - runs multiple 100-game batches until reaching 2000 games.
"""
import sys
import time
sys.path.insert(0, '/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3')

import json
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm

# Import Q3 builder functions
from src.build_dataset_q3 import extract_q3_row

TARGET_GAMES = 4000
BATCH_SIZE = 100

def run_batch():
    """Run one batch of 100 games."""
    # Load existing dataset
    existing_path = Path('data/processed/q3_team_v2.parquet')
    if existing_path.exists():
        existing_df = pd.read_parquet(existing_path)
        existing_games = set(existing_df['game_id'].tolist())
    else:
        existing_df = pd.DataFrame()
        existing_games = set()
    
    # Load all available game IDs
    with open('data/processed/game_ids_3_seasons.json', 'r') as f:
        games = json.load(f)
    
    game_ids = [g['gameId'] for g in games if 'gameId' in g]
    unprocessed = [gid for gid in game_ids if gid not in existing_games]
    
    # Stop if we've reached target
    if len(existing_games) >= TARGET_GAMES:
        print(f"\n{'='*60}")
        print(f"🎉 TARGET REACHED: {len(existing_games)} games!")
        print(f"{'='*60}")
        return False
    
    # Calculate batch size
    games_to_process = unprocessed[:BATCH_SIZE]
    
    if len(games_to_process) == 0:
        print(f"\n{'='*60}")
        print(f"✅ ALL AVAILABLE GAMES PROCESSED: {len(existing_games)} games")
        print(f"{'='*60}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Batch {len(existing_games)//BATCH_SIZE + 1}")
    print(f"Current: {len(existing_games)} games | Target: {TARGET_GAMES}")
    print(f"Processing: {len(games_to_process)} games")
    print(f"Progress: {len(existing_games) / len(game_ids) * 100:.1f}% of available")
    print(f"{'='*60}\n")
    
    rows = []
    errors = []
    
    for i, gid in enumerate(tqdm(games_to_process, desc='Building Q3 dataset')):
        # Fetch with retry logic (only delay on 403)
        url = f'https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json'
        max_retries = 2
        base_delay = 3.0
        
        game = None
        for attempt in range(max_retries):
            try:
                r = requests.get(url, timeout=25)
                r.raise_for_status()
                game = r.json()
                break
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 403:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"  Rate limited (403) on {gid}, retry in {delay:.1f}s...")
                        time.sleep(delay)
                    else:
                        print(f"  Max retries exceeded for {gid}, skipping")
                        break
                else:
                    raise
            except requests.exceptions.RequestException:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    break
        
        if game is None:
            errors.append({'game_id': gid, 'error': 'Failed to fetch after retries'})
            continue
        
        try:
            row = extract_q3_row(gid)
            if row:
                rows.append(row)
            else:
                errors.append({'game_id': gid, 'error': 'Failed to extract Q3 row'})
        except Exception as e:
            errors.append({'game_id': gid, 'error': str(e)})
    
    # Save final dataset
    if len(rows) > 0:
        combined_df = pd.concat([existing_df, pd.DataFrame(rows)], ignore_index=True)
        combined_df.to_parquet('data/processed/q3_team_v2.parquet', index=False)
        print(f"\nSaved dataset: data/processed/q3_team_v2.parquet ({len(combined_df)} total rows)")
        
        # Log errors
        if errors:
            with open('data/processed/q3_continuous.errors.jsonl', 'a') as f:
                for err in errors:
                    f.write(json.dumps(err) + '\n')
    else:
        print("\nNo new games processed in this batch")
    
    # Save stats
    stats_path = Path('data/processed/q3_build_stats.json')
    stats = {
        'total_games': len(combined_df),
        'target_games': TARGET_GAMES,
        'games_remaining': TARGET_GAMES - len(combined_df),
        'progress_pct': len(combined_df) / TARGET_GAMES * 100,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Status: {len(combined_df)}/{TARGET_GAMES} games ({stats['progress_pct']:.1f}%)")
    print(f"Games remaining: {stats['games_remaining']}")
    print(f"Estimated batches left: {stats['games_remaining'] // BATCH_SIZE + 1}")
    print(f"{'='*60}")
    
    return True

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"🏀 Q3 DATASET BUILDER - CONTINUOUS MODE")
    print(f"{'='*60}")
    print(f"Target: {TARGET_GAMES} games")
    print(f"Batch size: {BATCH_SIZE} games")
    print(f"Strategy: Process batches continuously")
    print(f"{'='*60}")
    
    batch_num = 1
    max_batches = 20  # Safety limit: 2000 games max
    
    while batch_num <= max_batches:
        keep_going = run_batch()
        if not keep_going:
            break
        batch_num += 1
    
    print(f"\n{'='*60}")
    print(f"✅ BUILD COMPLETE")
    print(f"{'='*60}")
