"""
Main runner for PerryPicks v4 Automation System.
Local event-driven automation: monitors games, fires triggers, posts to Discord.
"""

import logging
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import argparse
import os
import pytz

# Load environment variables from .env file (if it exists)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        logger = logging.getLogger(__name__)
        logger.debug(f"Loaded environment from {env_path}")
except ImportError:
    # python-dotenv not installed, fall back to os.getenv
    pass

from core.storage import init_database, GameStorage, TriggerStorage, PickStorage, TrackingStorage, DiscordPostStorage
from core.data_sources import CombinedDataSource
from core.discord_client import DiscordWebhookClient
from core.analysis import AnalysisEngine
from worker.scheduler import TriggerScheduler
from worker.triggers import TriggerFirer

logger = logging.getLogger(__name__)


class AutomationRunner:
    """Main automation runner - processes triggers and posts to Discord."""
    
    def __init__(
        self,
        db_path: Path,
        odds_api_key: str,
        discord_webhook_url: str,
        poll_interval: int = 60,
        dry_run: bool = False,
        date: str = 'today'
    ):
        self.db_path = db_path
        self.poll_interval = poll_interval
        self.dry_run = dry_run
        self.running = False
        self.date = date
        
        # Initialize components
        self.data_source = CombinedDataSource(odds_api_key)
        self.discord_client = DiscordWebhookClient(discord_webhook_url)
        self.analysis_engine = AnalysisEngine()
        self.scheduler = TriggerScheduler(db_path)
        self.trigger_firer = TriggerFirer(db_path, dry_run)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}; shutting down gracefully...")
        self.running = False
    
    def initialize(self) -> bool:
        """Initialize database and schedule triggers for today's games."""
        try:
            # Initialize database
            init_database(self.db_path)
            logger.info("Database initialized")
            
            # Schedule games for the specified date
            # IMPORTANT: Use CST (Eastern time) for 'today' since NBA games are
            # scheduled in EST/CST timezone. Using UTC here would cause issues
            # with games that cross the UTC date boundary (e.g., 9pm CST games).
            if self.date == 'today':
                cst_tz = pytz.timezone('America/Chicago')
                now_cst = datetime.now(timezone.utc).astimezone(cst_tz)
                schedule_date = now_cst.strftime('%Y-%m-%d')
            else:
                schedule_date = self.date
            
            games_scheduled = self.scheduler.schedule_games_for_date(schedule_date)
            
            logger.info(f"Initialized {games_scheduled} games for {schedule_date}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize: {e}")
            return False
    
    def run_once(self) -> int:
        """
        Run a single poll cycle.
        
        Returns:
            Number of triggers processed
        """
        now_utc = datetime.now(timezone.utc)
        window_start = now_utc - timedelta(minutes=2)
        window_end = now_utc + timedelta(seconds=30)
        
        total_processed = 0
        
        # 1. Process scheduled time-based triggers
        due_triggers = TriggerStorage.get_due_triggers(
            window_start, window_end, db_path=self.db_path
        )
        
        for trigger in due_triggers:
            processed = self._process_scheduled_trigger(trigger)
            if processed:
                total_processed += 1
        
        # 2. Poll active games for game-state triggers
        active_games = GameStorage.get_active_games(db_path=self.db_path)
        
        for game in active_games:
            processed = self._process_active_game(game)
            if processed:
                total_processed += 1
        
        # 3. Create periodic snapshots for tracking
        self._create_periodic_snapshots(active_games)
        
        return total_processed
    
    def _process_scheduled_trigger(self, trigger: dict) -> bool:
        """Process a scheduled time-based trigger."""
        game_id = trigger['game_id']
        trigger_type = trigger['trigger_type']
        
        try:
            logger.info(f"Processing scheduled trigger: {game_id} {trigger_type}")
            
            # Refresh game data
            data = self.data_source.refresh_game_data(
                game_id=game_id,
                reason=trigger_type,
                db_path=self.db_path
            )
            
            if not data['game_state']:
                logger.warning(f"Game {game_id} not found; skipping trigger")
                return False
            
            # Run analysis
            picks = self.analysis_engine.run_analysis(
                game_state=data['game_state'],
                odds=data['odds'],
                mode=trigger_type
            )
            
            if not picks:
                logger.warning(f"No picks generated for {game_id} {trigger_type}")
                return False
            
            # Store picks
            for pick in picks:
                PickStorage.store_pick(
                    game_id=game_id,
                    trigger_type=trigger_type,
                    bet_rank=pick['bet_rank'],
                    bet_type=pick['bet_type'],
                    side=pick['side'],
                    line=pick.get('line'),
                    odds=pick['odds'],
                    book=pick['book'],
                    probability=pick['probability'],
                    edge=pick['edge'],
                    rationale=pick.get('rationale'),
                    payload=pick,
                    db_path=self.db_path
                )
            
            # Post to Discord (if not dry run)
            if not self.dry_run:
                message = self.discord_client.format_bet_post(
                    trigger_type=trigger_type,
                    game_data=data['game_state'],
                    picks=picks,
                    timestamp=datetime.now(timezone.utc)
                )
                
                message_id = self.discord_client.post_message(message)
                
                if message_id:
                    DiscordPostStorage.store_post(
                        game_id=game_id,
                        trigger_type=trigger_type,
                        channel_id='main',  # Would be from webhook URL
                        message_id=message_id,
                        payload={
                            'message': message,
                            'picks': picks,
                            'game_state': data['game_state']
                        },
                        db_path=self.db_path
                    )
            
            # Mark trigger as fired
            TriggerStorage.mark_triggered(
                trigger_id=trigger['id'],
                fired_at_utc=datetime.now(timezone.utc),
                db_path=self.db_path
            )
            
            logger.info(f"Completed {trigger_type} trigger for {game_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing scheduled trigger {game_id} {trigger_type}: {e}")
            return False
    
    def _process_game_state_trigger(
        self,
        game_id: str,
        trigger_type: str,
        game_state: Dict[str, Any]
    ) -> bool:
        """
        Process a game-state trigger (halftime, Q3) that was just detected.
        
        Args:
            game_id: NBA game ID
            trigger_type: Type of trigger (HALFTIME, Q3)
            game_state: Current game state from NBA API
        
        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            logger.info(f"Processing game-state trigger: {game_id} {trigger_type}")
            
            # Refresh game data to get odds
            data = self.data_source.refresh_game_data(
                game_id=game_id,
                reason=trigger_type,
                db_path=self.db_path
            )
            
            if not data['game_state']:
                logger.warning(f"Game {game_id} not found; skipping trigger")
                return False
            
            # Run analysis
            picks = self.analysis_engine.run_analysis(
                game_state=data['game_state'],
                odds=data['odds'],
                mode=trigger_type
            )
            
            if not picks:
                logger.warning(f"No picks generated for {game_id} {trigger_type}")
                return False
            
            # Store picks
            for pick in picks:
                PickStorage.store_pick(
                    game_id=game_id,
                    trigger_type=trigger_type,
                    bet_rank=pick['bet_rank'],
                    bet_type=pick['bet_type'],
                    side=pick['side'],
                    line=pick.get('line'),
                    odds=pick['odds'],
                    book=pick['book'],
                    probability=pick['probability'],
                    edge=pick['edge'],
                    rationale=pick.get('rationale'),
                    payload=pick,
                    db_path=self.db_path
                )
            
            # Post to Discord (if not dry run)
            if not self.dry_run:
                message = self.discord_client.format_bet_post(
                    trigger_type=trigger_type,
                    game_data=data['game_state'],
                    picks=picks,
                    timestamp=datetime.now(timezone.utc)
                )
                
                message_id = self.discord_client.post_message(message)
                
                if message_id:
                    DiscordPostStorage.store_post(
                        game_id=game_id,
                        trigger_type=trigger_type,
                        channel_id='main',  # Would be from webhook URL
                        message_id=message_id,
                        payload={
                            'message': message,
                            'picks': picks,
                            'game_state': data['game_state']
                        },
                        db_path=self.db_path
                    )
            
            # Mark trigger as fired (update status from 'scheduled' to 'fired')
            # Get trigger ID for this game_id and trigger_type
            all_triggers = TriggerStorage.get_triggers_for_game(game_id, db_path=self.db_path)
            matching_trigger = [
                t for t in all_triggers 
                if t['trigger_type'] == trigger_type and t['status'] == 'scheduled'
            ]
            
            if matching_trigger:
                TriggerStorage.mark_triggered(
                    trigger_id=matching_trigger[0]['id'],
                fired_at_utc=datetime.now(timezone.utc),
                db_path=self.db_path
            )
            
            logger.info(f"Completed {trigger_type} trigger for {game_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing game-state trigger {game_id} {trigger_type}: {e}")
            return False
    
    def _process_active_game(self, game: dict) -> bool:
        """Process an active game for game-state triggers."""
        game_id = game['game_id']
        
        try:
            # Fetch latest game state
            game_state = self.data_source.nba.fetch_game_state(game_id)
            
            if not game_state:
                return False
            
            # Update game in database
            GameStorage.upsert_game(
                game_id=game_id,
                start_time_utc=datetime.fromisoformat(game['start_time_utc']) if isinstance(game['start_time_utc'], str) else game['start_time_utc'],
                home_team=game['home_team'],
                away_team=game['away_team'],
                status=game_state['status'],
                current_period=game_state.get('current_period'),
                game_clock=game_state.get('game_clock'),
                score_home=game_state.get('score_home', 0),
                score_away=game_state.get('score_away', 0),
                db_path=self.db_path
            )
            
            # Check for game-state triggers (halftime, Q3)
            triggers_fired = self.trigger_firer.process_game_state_triggers(
                game_id=game_id,
                game_state=game_state
            )
            
            if triggers_fired > 0:
                # Run analysis and post for each fired trigger
                # Get all triggers for this game
                all_triggers = TriggerStorage.get_triggers_for_game(game_id, db_path=self.db_path)
                
                # Filter for triggers created in last 2 minutes
                now_utc = datetime.now(timezone.utc)
                recent_cutoff = now_utc - timedelta(minutes=2)
                
                recent_triggers = [
                    t for t in all_triggers
                    if t['fired_at_utc'] and 
                    datetime.fromisoformat(t['fired_at_utc']) > recent_cutoff
                ]
                
                # Process each recent trigger
                for trigger in recent_triggers:
                    self._process_game_state_trigger(
                        game_id=game_id,
                        trigger_type=trigger['trigger_type'],
                        game_state=game_state
                    )
            
            return triggers_fired > 0
            
        except Exception as e:
            logger.error(f"Error processing active game {game_id}: {e}")
            return False
    
    def _create_periodic_snapshots(self, games: list):
        """Create periodic tracking snapshots for all games."""
        now_utc = datetime.now(timezone.utc)
        
        for game in games:
            game_id = game['game_id']
            
            try:
                # Refresh odds for periodic poll (with longer TTL)
                data = self.data_source.refresh_game_data(
                    game_id=game_id,
                    reason='PERIODIC',
                    db_path=self.db_path
                )
                
                # Create snapshot
                TrackingStorage.store_snapshot(
                    game_id=game_id,
                    timestamp_utc=now_utc,
                    poll_type='periodic',
                    quarter=game.get('current_period'),
                    game_clock=game.get('game_clock'),
                    score_home=game.get('score_home', 0),
                    score_away=game.get('score_away', 0),
                    db_path=self.db_path
                )
                
            except Exception as e:
                logger.error(f"Error creating periodic snapshot for {game_id}: {e}")
    
    def run(self):
        """Main loop - runs until stopped."""
        logger.info(f"Starting automation runner (poll_interval={self.poll_interval}s)")
        self.running = True
        
        while self.running:
            try:
                processed = self.run_once()
                
                if processed > 0:
                    logger.info(f"Processed {processed} triggers this cycle")
                else:
                    logger.debug("No triggers to process this cycle")
                
                # Sleep until next cycle
                # Use shorter intervals to check more frequently
                import time
                time.sleep(self.poll_interval)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                # Continue running despite errors
        
        logger.info("Automation runner stopped")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='PerryPicks v4 Automation Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--date',
        type=str,
        default='today',
        help='Date to process (YYYY-MM-DD or "today") (default: today)'
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
        '--once',
        action='store_true',
        help='Run a single poll cycle and exit'
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
            logging.FileHandler('logs/automation.log'),
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
    runner = AutomationRunner(
        db_path=db_path,
        odds_api_key=odds_api_key,
        discord_webhook_url=discord_webhook_url,
        poll_interval=args.poll_interval,
        dry_run=args.dry_run,
        date=args.date
    )
    
    if not runner.initialize():
        logger.error("Failed to initialize runner")
        sys.exit(1)
    
    # Run
    if args.once:
        logger.info("Running single poll cycle")
        runner.run_once()
    else:
        logger.info("Starting continuous automation")
        runner.run()


if __name__ == '__main__':
    main()
