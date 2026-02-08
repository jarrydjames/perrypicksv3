#!/usr/bin/env python3
"""Test automation flow for pregame prediction"""
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

from src.automation.automation_ui import run_prediction

print("------------------------------------------------------------")
print("Testing automation flow for game 0022500747")
print("------------------------------------------------------------")
print()

# Run prediction (dry run to avoid posting)
print("Running pregame prediction (dry run)...")
result = run_prediction(
    game_id='0022500747',
    trigger_type='pregame',
    platforms=[],  # No platforms = don't post
    dry_run=True,
    fetch_odds=False,
)

print()
print("------------------------------------------------------------")
print("AUTOMATION RESULT:")
print("------------------------------------------------------------")
for key, value in result.items():
    if key == 'predictions':
        print(f"{key}: {len(value)} prediction(s)")
        for pred in value:
            print(f"  - game_id: {pred.get('game_id')}")
            print(f"    status: {pred.get('status')}")
            print(f"    error: {pred.get('error', 'none')}")
    elif key == 'errors':
        print(f"{key}: {len(value)} error(s)")
        for err in value:
            print(f"  - game_id: {err.get('game_id')}")
            print(f"    error: {err.get('error')}")
    elif key == 'posted':
        print(f"{key}: {len(value)} posted")
    else:
        print(f"{key}: {value}")
print()
print("------------------------------------------------------------")
