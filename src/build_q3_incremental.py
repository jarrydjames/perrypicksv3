"""
Small-batch Q3 dataset builder - processes 50 games per run with aggressive rate limiting.
"""
import sys
sys.path.insert(0, '/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3')

import json
import time
import random
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

# Small batch to avoid timeouts
BATCH_SIZE = 50
games_to_process = unprocessed[:BATCH_SIZE]

print(f"\n=== Processing {len(games_to_process)} new games (small batch) ===")
print("Aggressive rate limiting: 5-15s random delays")
print()

rows = []
errors = []

for i, gid in enumerate(tqdm(games_to_process, desc='Building Q3 dataset')):
    # Aggressive random delay (5-15 seconds) per request
    delay = random.uniform(5, 15)
    if i > 0:
        print(f"  Delaying {delay:.1f}s for rate limit recovery...")
        time.sleep(delay)
    
    # Fetch with retry
    url = f'https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json'
    max_retries = 2
    base_delay = 5.0
    
    game = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            game = r.json()
            break
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + 10
                    print(f"  Rate limited, retry in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    print(f"  Skipping {gid} (max retries)")
                    break
            else:
                raise
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + 10
                time.sleep(delay)
            else:
                break
    
    if game is None:
        errors.append({'game_id': gid, 'error': 'Failed to fetch'})
        continue
    
    try:
        row = extract_q3_row(gid)
        if row:
            rows.append(row)
        else:
            errors.append({'game_id': gid, 'error': 'Failed to extract'})
    except Exception as e:
        errors.append({'game_id': gid, 'error': str(e)})

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
    with open('data/processed/q3_incremental.errors.jsonl', 'a') as f:
        for err in errors:
            f.write(json.dumps(err) + '\n')
    print(f"Errors logged: {len(errors)}")

print()
print("=== Summary ===")
print(f"Games processed: {len(rows)}")
print(f"Total games in dataset: {len(combined_df)}")
print(f"Games remaining: {len(game_ids) - len(combined_df)}")
print(f"Progress: {len(combined_df) / len(game_ids) * 100:.1f}%")
print()
print("Run this script again to continue building dataset.")
