#!/usr/bin/env python3
"""Manual DAILY_SUMMARY runner - posts predictions to Discord for a given date."""

import json
import os
import sqlite3
import sys
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from core.timezone import now_utc
from core.discord_client import DiscordWebhookClient

# Get webhook URL from environment (now loaded from .env)
webhook_url = os.getenv('DISCORD_WEBHOOK_URL', '')

if not webhook_url:
    print('ERROR: DISCORD_WEBHOOK_URL environment variable not set!')
    sys.exit(1)

# Get games from DB for 2026-02-05
conn = sqlite3.connect('data/automation.db')
cursor = conn.cursor()

cursor.execute('''
    SELECT game_id, home_team, away_team, start_time_utc
    FROM games
    WHERE game_date = '2026-02-05'
    AND NOT game_id LIKE 'test_%'
    ORDER BY start_time_utc
''')

games = []
for row in cursor.fetchall():
    games.append({
        'game_id': row[0],
        'home_team': row[1],
        'away_team': row[2],
        'start_time_utc': row[3]
    })

conn.close()

print(f'Found {len(games)} games for 2026-02-05')
for g in games:
    print(f'  {g["away_team"]} @ {g["home_team"]} ({g["game_id"]})')

# Build minimal payload
payload = {
    'date': '2026-02-05',
    'games': games
}

# Generate predictions for all games
predictions = []

for i, game in enumerate(games):
    game_id = game['game_id']
    away_name = game['away_team']
    home_name = game['home_team']
    
    print(f'\n[{i+1}/{len(games)}] Processing {away_name} @ {home_name} ({game_id})')
    
    try:
        # Add delay between requests to avoid rate limiting
        if i > 0:
            import time
            time.sleep(2)  # Reduced from 5s to 2s
        
        # Get pregame prediction
        from src.predict_api import predict_game
        result = predict_game(
            game_input=game_id,
            mode='pregame',
            fetch_odds=False
        )
        
        if result.get('status') in ['success', 'warning']:
            # Calculate individual scores from total and margin
            total = result.get('total', 0)
            margin = result.get('margin', 0)
            pred_home = (total + margin) / 2
            pred_away = (total - margin) / 2
            
            # Determine winner
            if margin > 0:
                pred_winner = result.get('home_name', 'Home')
            else:
                pred_winner = result.get('away_name', 'Away')
            
            predictions.append({
                'game_id': game_id,
                'away_name': result.get('away_name', 'Away'),
                'home_name': result.get('home_name', 'Home'),
                'predicted_away_score': pred_away,
                'predicted_home_score': pred_home,
                'predicted_total': total,
                'predicted_margin': margin,
                'predicted_winner': pred_winner
            })
            
            print(f'  ✓ Predicted: {pred_away:.1f} @ {pred_home:.1f} (Total: {total:.1f}, Winner: {pred_winner})')
        else:
            print(f'  ✗ Failed: {result.get("error", "Unknown error")}')
            # Add placeholder prediction
            predictions.append({
                'game_id': game_id,
                'away_name': away_name,
                'home_name': home_name,
                'predicted_away_score': 0,
                'predicted_home_score': 0,
                'predicted_total': 0,
                'predicted_margin': 0,
                'predicted_winner': 'Unknown'
            })
            
    except Exception as e:
        print(f'  ✗ Error: {e}')
        import traceback
        traceback.print_exc()
        # Add placeholder prediction
        predictions.append({
            'game_id': game_id,
            'away_name': away_name,
            'home_name': home_name,
            'predicted_away_score': 0,
            'predicted_home_score': 0,
            'predicted_total': 0,
            'predicted_margin': 0,
            'predicted_winner': 'Unknown'
        })

print(f'\nGenerated {len(predictions)} predictions')

# Post to Discord
discord = DiscordWebhookClient(webhook_url=webhook_url)
message = discord.format_daily_summary_post(
    predictions=predictions,
    timestamp=now_utc(),
    date='2026-02-05'
)

print('\n=== POSTING TO DISCORD ===')

# Post to Discord
message_id = discord.post_message(message)

if message_id:
    print(f'✅ Successfully posted to Discord! Message ID: {message_id}')
else:
    print(f'⚠️ Posted to Discord (no message ID returned)')

# Store post record
from core.storage import DiscordPostStorage
DiscordPostStorage.store_post(
    game_id='DAILY_20260205',
    trigger_type='DAILY_SUMMARY',
    channel_id='main',
    message_id=message_id if message_id else 'webhook-204',
    payload={
        'message': message,
        'predictions': predictions,
        'date': '2026-02-05'
    },
    db_path='data/automation.db'
)

# Mark existing trigger as fired
from core.storage import TriggerStorage
conn = sqlite3.connect('data/automation.db')
cursor = conn.cursor()
cursor.execute('''
    SELECT id FROM triggers
    WHERE game_id = 'DAILY_20260205'
    AND trigger_type = 'DAILY_SUMMARY'
    AND status = 'scheduled'
''')
row = cursor.fetchone()
if row:
    TriggerStorage.mark_triggered(
        trigger_id=row[0],
        fired_at_utc=now_utc(),
        db_path='data/automation.db'
    )
    print('DAILY_SUMMARY trigger marked as fired')
conn.close()

print('\n✅ DAILY_SUMMARY processed and posted to Discord!')