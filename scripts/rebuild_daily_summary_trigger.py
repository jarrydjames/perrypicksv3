#!/usr/bin/env python3
"""
Rebuild DAILY_SUMMARY trigger for a given CST date.

Why:
- Your diagnostic shows the DB was fixed after the trigger payload was already created,
  leaving stale games in payload_json.
This script deletes the existing trigger and re-schedules it from the corrected DB rows.

Usage:
    .venv/bin/python scripts/rebuild_daily_summary_trigger.py --date 2026-02-04
"""

import argparse
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import pendulum

from worker.scheduler import TriggerScheduler
from core.storage import TriggerStorage
from core.timezone import now_utc, CST

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(
        description='Rebuild DAILY_SUMMARY trigger for a given CST date.'
    )
    ap.add_argument(
        '--date',
        required=True,
        help='CST date in YYYY-MM-DD format'
    )
    ap.add_argument(
        '--db',
        default='data/automation.db',
        help='Path to database (default: data/automation.db)'
    )
    ap.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    args = ap.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    date = args.date
    db_path = Path(args.db)

    # Validate date format
    try:
        pendulum.parse(date, strict=True)
    except Exception as e:
        logger.error(f"Invalid date format '{date}': {e}")
        sys.exit(1)

    logger.info(f"Rebuilding DAILY_SUMMARY trigger for date: {date}")
    logger.info(f"Database path: {db_path}")

    # Delete existing trigger
    summary_game_id = f"DAILY_{date.replace('-', '')}"
    deleted = TriggerStorage.delete_trigger(
        summary_game_id,
        TriggerScheduler.DAILY_SUMMARY,
        db_path=db_path
    )
    logger.info(f"Deleted {deleted} existing DAILY_SUMMARY trigger(s) for {date}")

    # Re-schedule from corrected DB rows
    scheduler = TriggerScheduler(db_path)
    scheduled = scheduler.schedule_games_for_date(date)
    logger.info(f"Re-scheduled triggers for {scheduled} game(s) on {date}")

    if scheduled == 0:
        logger.warning(f"No games were scheduled for date {date}.")
        logger.warning("This could mean:")
        logger.warning("  1. No games exist for this date in the database")
        logger.warning("  2. All games failed CST date validation (check game_date values)")
        logger.warning("Run: SELECT game_id, start_time_utc, game_date FROM games; to verify")

    logger.info("Done!")


if __name__ == "__main__":
    main()
