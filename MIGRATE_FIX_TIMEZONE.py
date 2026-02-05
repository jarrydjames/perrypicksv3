"""
Database migration script to fix timezone bug in games and DAILY_SUMMARY triggers.

This script will:
1. Delete all DAILY_SUMMARY triggers for affected dates
2. Delete games with incorrect start_time_utc values
3. Re-fetch games with corrected logic
4. Re-create DAILY_SUMMARY triggers with correct game slates
"""

import sys
import logging
from pathlib import Path

import pendulum
import sqlite3

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.storage import GameStorage, TriggerStorage
from core.data_sources import NBADataSource
from core.timezone import CST

from worker.scheduler import TriggerScheduler

from worker.triggers import TriggerFirer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def migrate_database(db_path: Path, dry_run: bool = True):
    """
    Migrate database to fix timezone bug.
    
    Args:
        db_path: Path to database file
        dry_run: If True, only report what would be done without actually doing it
    """
    logger.info(f"Starting timezone migration (dry_run={dry_run})")
    logger.info(f"Database: {db_path}")
    
    # Affected dates based on bug diagnosis
    affected_dates = ['2026-02-03', '2026-02-04', '2026-02-05']
    
    # Step 1: Find affected games
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Identifying affected games")
    logger.info("="*70)
    
    affected_games = []
    for date in affected_dates:
        games = GameStorage.get_games_for_date(date, db_path=db_path)
        logger.info(f"\nGames for date {date}: {len(games)} total")
        for g in games:
            game_id = g['game_id']
            start_utc = g['start_time_utc']
            game_date = g['game_date']
            
            # Parse start_time_utc
            if isinstance(start_utc, str):
                dt_utc = pendulum.parse(start_utc)
            else:
                dt_utc = start_utc
            
            # Calculate what game_date should be
            correct_date = dt_utc.in_timezone(CST).format('YYYY-MM-DD')
            
            if correct_date != game_date:
                affected_games.append({
                    'game_id': game_id,
                    'date': date,
                    'start_utc': dt_utc.to_iso8601_string(),
                    'game_date_current': game_date,
                    'game_date_correct': correct_date,
                    'needs_fix': True
                })
                logger.warning(f"  {game_id}: start_utc={dt_utc.to_iso8601_string()}, current={game_date}, correct={correct_date} ❌")
            else:
                logger.info(f"  {game_id}: OK ({dt_utc.to_iso8601_string()} → {game_date})")
    
    logger.info(f"\nTotal affected games: {len(affected_games)}")
    
    # Step 2: Find affected DAILY_SUMMARY triggers
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Identifying affected DAILY_SUMMARY triggers")
    logger.info("="*70)
    
    affected_triggers = []
    for date in affected_dates:
        trigger_id = f"DAILY_{date.replace('-', '')}"
        trigger = TriggerStorage.get_trigger(trigger_id, 'DAILY_SUMMARY', db_path=db_path)
        
        if trigger:
            payload = trigger.get('payload_json')
            if isinstance(payload, str):
                import json
                payload = json.loads(payload)
            
            games_in_trigger = payload.get('games', []) if payload else []
            logger.info(f"\nTrigger {trigger_id}: {len(games_in_trigger)} games")
            
            for g in games_in_trigger:
                game_id = g.get('game_id')
                start_utc = g.get('start_time_utc')
                
                if start_utc:
                    if isinstance(start_utc, str):
                        dt_utc = pendulum.parse(start_utc)
                    else:
                        dt_utc = start_utc
                    
                    # Check if game has wrong start_time_utc (games starting on wrong UTC day)
                    correct_date = dt_utc.in_timezone(CST).format('YYYY-MM-DD')
                    if correct_date != date:
                        affected_triggers.append({
                            'trigger_id': trigger_id,
                            'game_id': game_id,
                            'start_utc': dt_utc.to_iso8601_string(),
                            'trigger_date': date,
                            'correct_game_date': correct_date
                        })
                        logger.warning(f"  Game {game_id}: start_utc={dt_utc.to_iso8601_string()}, belongs to {correct_date}, not {date} ❌")
        else:
            logger.info(f"Trigger {trigger_id}: Not found")
    
    logger.info(f"\nTotal affected triggers: {len(affected_triggers)}")
    
    # Step 3: Delete and recreate (if not dry run)
    if not dry_run:
        logger.info("\n" + "="*70)
        logger.info("STEP 3: Deleting affected data")
        logger.info("="*70)
        
        # Delete DAILY_SUMMARY triggers for affected dates
        for date in affected_dates:
            trigger_id = f"DAILY_{date.replace('-', '')}"
            logger.info(f"Deleting trigger {trigger_id}...")
            # Delete trigger
            conn = sqlite3.connect(db_path)
            conn.execute("DELETE FROM triggers WHERE game_id = ? AND trigger_type = 'DAILY_SUMMARY'", (trigger_id,))
            conn.commit()
            conn.close()
        
        # Delete affected games
        conn = sqlite3.connect(db_path)
        deleted = 0
        for game in affected_games:
            game_id = game['game_id']
            conn.execute("DELETE FROM games WHERE game_id = ?", (game_id,))
            deleted += 1
        conn.commit()
        conn.close()
        
        logger.info(f"Deleted {deleted} games")
        
        # Step 4: Re-fetch and re-schedule
        logger.info("\n" + "="*70)
        logger.info("STEP 4: Re-fetching games and re-creating triggers")
        logger.info("="*70)
        
        odds_api_key = 'dummy'  # Not needed for schedule fetch
        scheduler = TriggerScheduler(db_path)
        
        for date in affected_dates:
            logger.info(f"\nRe-fetching games for {date}...")
            # Fetch games with NEW corrected logic
            games = NBADataSource.fetch_games_for_date(date)
            logger.info(f"Fetched {len(games)} games from API")
            
            # Upsert games to database
            for g in games:
                GameStorage.upsert_game(
                    game_id=g['game_id'],
                    start_time_utc=g['start_time_utc'],
                    home_team=g['home_team'],
                    away_team=g['away_team'],
                    status=g['status'],
                    game_date=g['game_date'],
                    db_path=db_path
                )
                logger.info(f"  Upserted {g['game_id']}: {g['start_time_utc'].to_iso8601_string()} → {g['game_date']}")
            
            # Schedule triggers
            scheduled = scheduler.schedule_games_for_date(date)
            logger.info(f"Scheduled {scheduled} games for {date}")
        
        logger.info("\nMigration complete!")
    else:
        logger.info("\n" + "="*70)
        logger.info("DRY RUN - No changes made")
        logger.info("="*70)
        logger.info("\nTo apply fixes, run with --apply flag")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Migrate database to fix timezone bug')
    parser.add_argument('--apply', action='store_true', help='Actually apply fixes (default is dry run)')
    parser.add_argument('--db', type=str, default='data/automation.db', help='Database path')
    args = parser.parse_args()
    
    db_path = Path(args.db)
    migrate_database(db_path, dry_run=not args.apply)


if __name__ == '__main__':
    main()
