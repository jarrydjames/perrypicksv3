#!/usr/bin/env python3
"""
Migration script for league_day and local_day_cst columns.

This script:
1) Ensures schema migration ran (adds league_day and local_day_cst columns)
2) Backfills local_day_cst for ALL games using start_time_utc derivation
3) Fetches games from NBA API for specified league_day range
4) Upserts games with league_day set
5) Deletes old DAILY_SUMMARY triggers
6) Rebuilds DAILY_SUMMARY triggers with new minimal payload

Usage:
    python scripts/migrate_league_day.py --dry-run
    python scripts/migrate_league_day.py --apply --start 2026-02-01 --end 2026-02-28
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pendulum
from core.storage import init_database, GameStorage, TriggerStorage
from core.data_sources import NBADataSource
from core.timezone import cst_game_date_from_start_time_utc, CST

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def ensure_schema_migration(db_path: Path):
    """Ensure league_day and local_day_cst columns exist."""
    logger.info("Ensuring schema migration...")
    init_database(db_path)
    logger.info("Schema migration complete")


def backfill_local_day_cst(db_path: Path):
    """Backfill local_day_cst for all games from start_time_utc."""
    import sqlite3
    
    logger.info("Backfilling local_day_cst for all games...")
    
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT game_id, start_time_utc FROM games")
        games = cursor.fetchall()
        
        updated = 0
        for game in games:
            game_id = game['game_id']
            start_time_str = game['start_time_utc']
            
            try:
                dt_utc = pendulum.parse(start_time_str)
                local_day_cst = cst_game_date_from_start_time_utc(dt_utc, tz=CST)
                cursor.execute("UPDATE games SET local_day_cst = ? WHERE game_id = ?", (local_day_cst, game_id))
                updated += 1
            except Exception as e:
                logger.warning(f"Failed to backfill local_day_cst for {game_id}: {e}")
        
        conn.commit()
        logger.info(f"Backfilled local_day_cst for {updated}/{len(games)} games")


def migrate_league_days(start_date: str, end_date: str, db_path: Path, dry_run: bool):
    """Migrate games for a range of league days."""
    from worker.scheduler import TriggerScheduler
    
    logger.info(f"Migrating league days from {start_date} to {end_date} (dry_run={dry_run})")
    
    start_dt = pendulum.parse(start_date)
    end_dt = pendulum.parse(end_date)
    
    nba_source = NBADataSource()
    scheduler = TriggerScheduler(db_path)
    
    current_dt = start_dt
    while current_dt <= end_dt:
        league_day = current_dt.format('YYYY-MM-DD')
        logger.info(f"Processing league_day {league_day}...")
        
        try:
            games = nba_source.fetch_games_for_league_day(league_day)
            logger.info(f"Fetched {len(games)} games for {league_day}")
        except Exception as e:
            logger.error(f"Failed to fetch games for {league_day}: {e}")
            current_dt = current_dt.add(days=1)
            continue
        
        if dry_run:
            logger.info(f"DRY RUN: Would upsert {len(games)} games for {league_day}")
        else:
            for g in games:
                try:
                    GameStorage.upsert_game(
                        game_id=g['game_id'],
                        start_time_utc=g['start_time_utc'],
                        home_team=g['home_team'],
                        away_team=g['away_team'],
                        status=g.get('status', 'Scheduled'),
                        game_date=g.get('local_day_cst'),
                        league_day=g.get('league_day'),
                        db_path=db_path
                    )
                except Exception as e:
                    logger.warning(f"Failed upserting {g.get('game_id')}: {e}")
            
            summary_game_id = f"DAILY_{league_day.replace('-', '')}"
            deleted = TriggerStorage.delete_trigger(summary_game_id, 'DAILY_SUMMARY', db_path=db_path)
            if deleted:
                logger.info(f"Deleted old DAILY_SUMMARY trigger for {league_day}")
            
            count = scheduler.schedule_games_for_league_day(league_day)
            logger.info(f"Rebuilt DAILY_SUMMARY trigger for {league_day} ({count} games)")
        
        current_dt = current_dt.add(days=1)
    
    logger.info("Migration complete!")


def main():
    parser = argparse.ArgumentParser(description='Migrate league_day and local_day_cst columns')
    parser.add_argument('--dry-run', action='store_true', help='Dry run (no changes)')
    parser.add_argument('--apply', action='store_true', help='Apply changes')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--db-path', type=str, default=None, help='Database path')
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.apply:
        print("ERROR: Must specify --dry-run or --apply")
        sys.exit(1)
    
    db_path = Path(args.db_path) if args.db_path else Path('data/automation.db')
    
    ensure_schema_migration(db_path)
    backfill_local_day_cst(db_path)
    
    if args.start and args.end:
        migrate_league_days(args.start, args.end, db_path, args.dry_run)
    else:
        logger.info("No --start/--end specified, only running schema migration and local_day_cst backfill")


if __name__ == '__main__':
    main()
