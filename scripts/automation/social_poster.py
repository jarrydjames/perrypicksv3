#!/usr/bin/env python3
"""Social Media Posting Automation for PerryPicks v3.

Usage:
    # Run in scheduler mode (continuous)
    python social_poster.py --schedule --poll-interval 15
    
    # Run one-off predictions for specific games
    python social_poster.py --games 0022500747 0022500748 --trigger-type pregame
    
    # Process pending posts from queue
    python social_poster.py --process-queue
    
    # Show queue statistics
    python social_poster.py --stats
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.env import load_environment
from src.automation import (
    AutomationOrchestrator,
    run_automation,
    run_one_off_predictions,
)
from core.timezone import now_utc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PerryPicks v3 - Social Media Posting Automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--schedule",
        action="store_true",
        help="Run in scheduler mode (continuous loop)",
    )
    mode_group.add_argument(
        "--games",
        nargs="+",
        help="Game IDs to predict (one-off mode)",
    )
    mode_group.add_argument(
        "--process-queue",
        action="store_true",
        help="Process pending posts from queue",
    )
    mode_group.add_argument(
        "--stats",
        action="store_true",
        help="Show queue statistics",
    )
    
    # Prediction options
    parser.add_argument(
        "--trigger-type",
        default="pregame",
        choices=["pregame", "halftime", "q3"],
        help="Prediction trigger type (default: pregame)",
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "pregame", "halftime", "q3"],
        help="Prediction mode (default: auto)",
    )
    
    # Scheduler options
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=15,
        help="Poll interval in minutes for scheduler mode (default: 15)",
    )
    
    # Platform options
    parser.add_argument(
        "--platforms",
        nargs="+",
        choices=["twitter", "bluesky", "discord"],
        help="Platforms to post to (default: all enabled)",
    )
    
    # General options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode (don't actually post)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load environment
    load_environment()
    
    try:
        # Execute requested mode
        if args.schedule:
            logger.info(f"Starting scheduler mode (poll interval: {args.poll_interval}min)")
            run_automation(
                dry_run=args.dry_run,
                platforms=args.platforms,
                poll_interval_minutes=args.poll_interval,
            )
        
        elif args.games:
            logger.info(f"Running one-off predictions for {len(args.games)} games")
            results = run_one_off_predictions(
                game_ids=args.games,
                trigger_type=args.trigger_type,
                mode=args.mode,
                dry_run=args.dry_run,
                platforms=args.platforms,
            )
            
            # Print summary
            print("\n" + "=" * 60)
            print("PREDICTION SUMMARY")
            print("=" * 60)
            print(f"Games processed: {len(args.games)}")
            print(f"Successful predictions: {len(results['predictions'])}")
            print(f"Posted to platforms: {len(results['posted'])}")
            print(f"Errors: {len(results['errors'])}")
            print("=" * 60)
            
            if results["errors"]:
                print("\nErrors:")
                for error in results["errors"]:
                    print(f"  - {error['game_id']}: {error['error']}")
        
        elif args.process_queue:
            logger.info("Processing pending posts from queue")
            orchestrator = AutomationOrchestrator(
                dry_run=args.dry_run,
                platforms=args.platforms,
            )
            
            results = orchestrator.process_post_queue(batch_size=10)
            
            # Print summary
            print("\n" + "=" * 60)
            print("QUEUE PROCESSING SUMMARY")
            print("=" * 60)
            print(f"Posts processed: {results['processed']}")
            print(f"Successful: {results['successful']}")
            print(f"Failed: {results['failed']}")
            print("=" * 60)
        
        elif args.stats:
            orchestrator = AutomationOrchestrator(
                dry_run=args.dry_run,
                platforms=args.platforms,
            )
            
            stats = orchestrator.get_stats()
            
            # Print stats
            print("\n" + "=" * 60)
            print("AUTOMATION STATISTICS")
            print("=" * 60)
            print(f"Processed predictions: {stats['processed_predictions']}")
            print(f"Enabled platforms: {', '.join(stats['enabled_platforms'])}")
            print("\nQueue stats:")
            queue_stats = stats['queue_stats']
            print(f"  Total: {queue_stats['total']}")
            print(f"  Pending: {queue_stats['pending']}")
            print(f"  Posted: {queue_stats['posted']}")
            print(f"  Failed: {queue_stats['failed']}")
            print("=" * 60)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
