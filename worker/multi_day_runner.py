"""
Multi-Day Runner Wrapper for PerryPicks v4.
Automatically transitions between days and schedules games.
"""

import logging
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import argparse
import os
import time
import pytz

from core.env import load_environment

# Load environment variables from .env file (if it exists)
load_environment(search_from=Path(__file__).resolve().parents[1])

from core.storage import init_database, GameStorage, TriggerStorage
from core.data_sources import NBADataSource
from worker.scheduler import TriggerScheduler

logger = logging.getLogger(__name__)


class MultiDayRunner:
    """Wraps AutomationRunner for multi-day operation."""
    
    # Day transition time: midnight CST = 5am UTC
    DAY_TRANSITION_UTC_HOUR = 5
    
    def __init__(
        self,
        db_path: Path,
        odds_api_key: str,
        poll_interval: int = 60,
        dry_run: bool = False
    ):
        self.db_path = db_path
        self.odds_api_key = odds_api_key
        self.poll_interval = poll_interval
        self.dry_run = dry_run
        self.running = False
        
        # Initialize components
        self.nba_source = NBADataSource()
        self.scheduler = TriggerScheduler(db_path)
        
        # Track current date in CST
        self.current_date_cst = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}; shutting down gracefully...")
        self.running = False
    
    def initialize(self) -> bool:
        """Initialize database and schedule today's games."""
        try:
            # Initialize database
            init_database(self.db_path)
            logger.info("Database initialized")
            
            # Schedule today's games
            today_date_cst = self._get_current_date_cst()
            self.current_date_cst = today_date_cst
            
            games_scheduled = self.scheduler.schedule_games_for_date(today_date_cst)
            
            logger.info(f"Initialized {games_scheduled} games for {today_date_cst}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False
    
    def _get_current_date_cst(self) -> str:
        """Get current date in CST timezone."""
        cst_tz = pytz.timezone('America/Chicago')
        now_cst = datetime.now(timezone.utc).astimezone(cst_tz)
        return now_cst.strftime('%Y-%m-%d')
    
    def _check_day_transition(self) -> bool:
        """
        Check if we need to transition to the next day.
        
        Returns:
            True if transition needed, False otherwise
        """
        # Get current date in CST
        current_date_cst = self._get_current_date_cst()
        
        # Check if date changed
        if current_date_cst != self.current_date_cst:
            logger.info(f"Day transition detected: {self.current_date_cst} -> {current_date_cst}")
            return True
        
        return False
    
    def _transition_to_next_day(self) -> bool:
        """
        Transition to the next day by scheduling new games.
        
        Returns:
            True if transition successful, False otherwise
        """
        try:
            # Get new date
            new_date_cst = self._get_current_date_cst()
            
            logger.info(f"Transitioning to {new_date_cst}")
            
            # Schedule games for new date
            games_scheduled = self.scheduler.schedule_games_for_date(new_date_cst)
            
            if games_scheduled > 0:
                logger.info(f"Scheduled {games_scheduled} games for {new_date_cst}")
                self.current_date_cst = new_date_cst
                return True
            else:
                logger.warning(f"No games found for {new_date_cst}")
                self.current_date_cst = new_date_cst
                return False
            
        except Exception as e:
            logger.error(f"Failed to transition to next day: {e}")
            return False
    
    def run_single_cycle(self) -> int:
        """
        Run a single automation cycle with day transition check.
        
        Returns:
            Number of new games scheduled (0 if none)
        """
        new_games = 0
        
        # Check for day transition
        if self._check_day_transition():
            if self._transition_to_next_day():
                # Get count of new games scheduled
                games_today = GameStorage.get_games_for_date(
                    self.current_date_cst,
                    db_path=self.db_path
                )
                new_games = len(games_today)
        
        return new_games
    
    def run(self):
        """Main loop - runs until stopped."""
        logger.info(f"Starting multi-day runner (poll_interval={self.poll_interval}s)")
        logger.info(f"Day transition time: {self.DAY_TRANSITION_UTC_HOUR}:00 UTC (midnight CST)")
        self.running = True
        
        while self.running:
            try:
                # Check for day transition
                new_games = self.run_single_cycle()
                
                if new_games > 0:
                    logger.info(f"Day transition complete - {new_games} games scheduled")
                
                # Sleep until next cycle
                time.sleep(self.poll_interval)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                # Continue running despite errors
        
        logger.info("Multi-day runner stopped")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='PerryPicks v4 Multi-Day Runner Wrapper',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--poll-interval',
        type=int,
        default=60,
        help='Poll interval in seconds (default: 60)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without posting to Discord'
    )
    
    parser.add_argument(
        '--db-path',
        type=str,
        default=None,
        help='Path to database (default: data/automation.db)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/multi_day_automation.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set database path
    if args.db_path:
        db_path = Path(args.db_path)
    else:
        db_path = Path(__file__).parent.parent / 'data' / 'automation.db'
    
    # Load environment variables
    odds_api_key = os.getenv('ODDS_API_KEY', '')
    discord_webhook_url = os.getenv('DISCORD_WEBHOOK_URL', '')
    
    if not odds_api_key:
        logger.error("ODDS_API_KEY environment variable not set")
        sys.exit(1)
    
    if not discord_webhook_url:
        logger.error("DISCORD_WEBHOOK_URL environment variable not set")
        sys.exit(1)
    
    # Create and initialize runner
    runner = MultiDayRunner(
        db_path=db_path,
        odds_api_key=odds_api_key,
        poll_interval=args.poll_interval,
        dry_run=args.dry_run
    )
    
    if not runner.initialize():
        logger.error("Failed to initialize runner")
        sys.exit(1)
    
    # Run
    logger.info("Starting continuous multi-day automation")
    runner.run()


if __name__ == '__main__':
    main()
