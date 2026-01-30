"""
Fast Q3 dataset builder - no random delays, only retry on rate limits.
Processes games as fast as possible, like the successful first attempt.
"""
import sys
import time
sys.path.insert(0, '/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3')

import json
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import requests

# Import Q3 builder functions
from src.build_dataset_q3 import extract_q3_row, sum_first2

# Load existing dataset
existing_path = Path('data/processed/q3_team_v2.parquet')
if existing_path.exists():
    existing_df = pd.read_parquet(existing_path)
    existing_games = set(existing_df['game_id'].tolist())
    print(f"Loaded existing dataset: {len(existing_df)} games")
else:
    existing_df = pd.DataFrame()
    existing_games = set()
    print("No existing dataset found, starting fresh")

# Load all available game IDs
with open('data/processed/game_ids_from_box_cache.json', 'r') as f:
    games = json.load(f)

game_ids = [g['gameId'] for g in games if 'gameId' in g]

# Calculate games to process
unprocessed = [gid for gid in game_ids if gid not in existing_games]

# Small batch to ensure completion
BATCH_SIZE = 100
games_to_process = unprocessed[:BATCH_SIZE]

print(f"\n=== Processing {len(games_to_process)} new games (FAST MODE) ===")
print("Strategy: No random delays, only retry on rate limits")
print("Goal: Process as fast as possible (like first successful attempt)")
print()

rows = []
errors = []

for i, gid in enumerate(tqdm(games_to_process, desc='Building Q3 dataset (FAST)')):
    # No preemptive delays - process immediately
    
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
    
    # Save checkpoint every 25 games
    if len(rows) > 0 and len(rows) % 25 == 0:
        checkpoint_df = pd.DataFrame(rows)
        checkpoint_df.to_parquet('data/processed/q3_fast_checkpoint.parquet', index=False)
        print(f"  Saved checkpoint: {len(rows)} games")

# Save final dataset
if len(rows) > 0:
    # Combine with existing dataset
    combined_df = pd.concat([existing_df, pd.DataFrame(rows)], ignore_index=True)
    combined_df.to_parquet('data/processed/q3_team_v2.parquet', index=False)
    print(f"\nSaved updated dataset: data/processed/q3_team_v2.parquet ({len(combined_df)} total rows)")
else:
    print("\nNo new games processed")

# Save error log
if errors:
    with open('data/processed/q3_fast.errors.jsonl', 'a') as f:
        for err in errors:
            f.write(json.dumps(err) + '\n')
    print(f"Errors logged: data/processed/q3_fast.errors.jsonl ({len(errors)} errors)")

print()
print("=== Summary ===")
print(f"Games processed: {len(rows)}")
print(f"Total games in dataset: {len(existing_df) + len(rows)}")
games_remaining = len(game_ids) - len(existing_df) - len(rows)
print(f"Games remaining: {games_remaining}")
progress_pct = (len(existing_df) + len(rows)) / len(game_ids) * 100
print(f"Progress: {progress_pct:.1f}%")
print()
print("This should be much faster than the slow incremental builder!")
