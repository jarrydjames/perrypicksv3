"""
Unified Runner for PerryPicks v4.
Handles multi-day transitions AND trigger processing in a single process.
"""

import logging
import signal
import sys
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import argparse
import os
import time
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
    pass

from core.timezone import parse_iso_utc
from core.storage import (
    init_database, GameStorage, TriggerStorage, PickStorage, 
    TrackingStorage, DiscordPostStorage
)
from core.data_sources import CombinedDataSource
from core.discord_client import DiscordWebhookClient
from core.analysis import AnalysisEngine
from worker.scheduler import TriggerScheduler
from worker.triggers import TriggerFirer

logger = logging.getLogger(__name__)


class UnifiedRunner:
    """Unified runner for multi-day operation and trigger processing."""
    
    # Day transition time: midnight CST = 5am UTC
    DAY_TRANSITION_UTC_HOUR = 5
    
    # Trigger types
    DAILY_SUMMARY = 'DAILY_SUMMARY'
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
        
        # Track current date in CST
        self.current_date_cst = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}; shutting down gracefully...")
        self.running = False
    
    def _safe_payload_date(self, payload_json) -> str | None:
        """
        Return payload['date'] if present.
        payload_json may be a dict already or a JSON string depending on storage layer.
        """
        try:
            if payload_json is None:
                return None
            if isinstance(payload_json, dict):
                return payload_json.get('date')
            if isinstance(payload_json, str):
                data = json.loads(payload_json)
                return data.get('date')
        except Exception:
            return None
        return None
    
    def _should_process_trigger(self, trigger) -> bool:
        """
        Critical fix:
        DAILY_SUMMARY triggers must be processed ONLY when payload.date matches
        runner's current CST date. Otherwise, the runner may fire the wrong day's summary
        simply because it is 'due' by scheduled_time_utc.
        """
        try:
            if trigger.get('trigger_type', None) != self.DAILY_SUMMARY:
                return True
            
            payload_date = self._safe_payload_date(trigger.get('payload_json', None))
            if not payload_date:
                # No payload date; safest is to skip
                logger.warning(f"Skipping DAILY_SUMMARY trigger with missing payload date: {trigger.get('game_id', 'UNKNOWN')}")
                return False
            
            if payload_date != self.current_date_cst:
                logger.info(
                    f"Skipping DAILY_SUMMARY {trigger.get('game_id', 'UNKNOWN')} "
                    f"for payload.date={payload_date} (current CST date={self.current_date_cst})"
                )
                return False
            
            return True
        except Exception as e:
            logger.warning(f"Error deciding whether to process trigger; skipping. err={e}")
            return False
    
    def initialize(self) -> bool:
        """Initialize database and schedule today's games."""
        try:
            # Initialize database
            init_database(self.db_path)
            logger.info("Database initialized")
            
            # Schedule games for specified date
            if self.date == 'today':
                self.current_date_cst = self._get_current_date_cst()
                schedule_date = self.current_date_cst
            else:
                schedule_date = self.date
                self.current_date_cst = schedule_date
            
            games_scheduled = self.scheduler.schedule_games_for_date(schedule_date)
            
            logger.info(f"Initialized {games_scheduled} games for {schedule_date}")
            logger.info(f"Day transition time: {self.DAY_TRANSITION_UTC_HOUR}:00 UTC (midnight CST)")
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
        """Check if we need to transition to the next day."""
        current_date_cst = self._get_current_date_cst()
        if current_date_cst != self.current_date_cst:
            logger.info(f"Day transition detected: {self.current_date_cst} -> {current_date_cst}")
            return True
        return False
    
    def _transition_to_next_day(self) -> bool:
        """Transition to the next day by scheduling new games."""
        try:
            new_date_cst = self._get_current_date_cst()
            logger.info(f"Transitioning to {new_date_cst}")
            
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
    
    def run_once(self) -> dict:
        """
        Run a single cycle: check day transition + process triggers.
        
        Returns:
            Dictionary with stats
        """
        stats = {
            'day_transition': False,
            'new_games': 0,
            'triggers_processed': 0,
            'snapshots_created': 0
        }
        
        # 1. Check for day transition
        if self._check_day_transition():
            stats['day_transition'] = True
            if self._transition_to_next_day():
                games_today = GameStorage.get_games_for_date(
                    self.current_date_cst,
                    db_path=self.db_path
                )
                stats['new_games'] = len(games_today)
        
        # 2. Process triggers
        now_utc = datetime.now(timezone.utc)
        window_start = now_utc - timedelta(minutes=2)
        window_end = now_utc + timedelta(seconds=30)
        
        # Process scheduled time-based triggers
        due_triggers = TriggerStorage.get_due_triggers(
            window_start, window_end, db_path=self.db_path
        )
        
        for trigger in due_triggers:
            # Critical fix: Only process DAILY_SUMMARY if payload date matches current CST date
            if not self._should_process_trigger(trigger):
                continue
            
            processed = self._process_scheduled_trigger(trigger)
            if processed:
                stats['triggers_processed'] += 1
        
        # Poll active games for game-state triggers
        active_games = GameStorage.get_active_games(db_path=self.db_path)
        
        for game in active_games:
            processed = self._process_active_game(game)
            if processed:
                stats['triggers_processed'] += 1
        
        # Create periodic snapshots
        self._create_periodic_snapshots(active_games)
        stats['snapshots_created'] = len(active_games)
        
        return stats
    
    def _process_scheduled_trigger(self, trigger: dict) -> bool:
        """Process a scheduled time-based trigger."""
        game_id = trigger['game_id']
        trigger_type = trigger['trigger_type']
        
        try:
            # Handle DAILY_SUMMARY triggers specially
            if trigger_type == self.DAILY_SUMMARY:
                return self._process_daily_summary(trigger)
            
            logger.info(f"Processing scheduled trigger: {game_id} {trigger_type}")
            
            data = self.data_source.refresh_game_data(
                game_id=game_id,
                reason=trigger_type,
                db_path=self.db_path
            )
            
            if not data['game_state']:
                logger.warning(f"Game {game_id} not found; skipping trigger")
                return False
            
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
            
            # Post to Discord
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
                        channel_id='main',
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
    
    def _process_daily_summary(self, trigger: dict) -> bool:
        """
        Process DAILY_SUMMARY trigger - post predictions for all games today.
        
        Args:
            trigger: DAILY_SUMMARY trigger with game list in payload
        
        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            # Parse payload_json (it's stored as JSON string in DB)
            import json
            import time
            
            payload_json = trigger.get('payload_json', '{}')
            payload = json.loads(payload_json) if payload_json else {}
            games = payload.get('games', [])
            date = payload.get('date', '')
            
            if not games:
                logger.warning("No games in DAILY_SUMMARY payload")
                return False
            
            logger.info(f"Processing DAILY_SUMMARY for {date} ({len(games)} games)")
            
            # Generate predictions for all games
            predictions = []
            for game in games:
                game_id = game['game_id']
                
                try:
                    # Add delay between requests to avoid NBA API rate limiting
                    if predictions:  # Don't delay on first game
                        time.sleep(5)  # 5 second delay between games to avoid 403 errors
                    
                    # Get pregame prediction
                    from src.predict_api import predict_game
                    result = predict_game(
                        game_input=game_id,
                        mode='pregame',
                        fetch_odds=False  # Don't need odds for summary
                    )
                    
                    if result.get('status') == 'success':
                        # Calculate individual scores from total and margin
                        total = result.get('total', 0)
                        margin = result.get('margin', 0)
                        pred_home = (total - margin) / 2
                        pred_away = (total + margin) / 2
                        
                        # Determine winner
                        if margin < 0:
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
                        
                except Exception as e:
                    logger.error(f"Error generating prediction for {game_id}: {e}")
                    # Add placeholder prediction even if failed
                    predictions.append({
                        'game_id': game_id,
                        'away_name': game.get('away_team', 'Away'),
                        'home_name': game.get('home_team', 'Home'),
                        'predicted_away_score': 0,
                        'predicted_home_score': 0,
                        'predicted_total': 0,
                        'predicted_margin': 0,
                        'predicted_winner': 'Unknown'
                    })
            
            # Post to Discord
            if not self.dry_run:
                message = self.discord_client.format_daily_summary_post(
                    predictions=predictions,
                    timestamp=datetime.now(timezone.utc),
                    date=date
                )
                
                message_id = self.discord_client.post_message(message)
                
                # Store post record (even if message_id is None - some webhooks return 204 without ID)
                DiscordPostStorage.store_post(
                    game_id=trigger['game_id'],
                    trigger_type='DAILY_SUMMARY',
                    channel_id='main',
                    message_id=message_id if message_id else 'webhook-204',
                    payload={
                        'message': message,
                        'predictions': predictions,
                        'date': date
                    },
                    db_path=self.db_path
                )
            
            # Mark trigger as fired
            TriggerStorage.mark_triggered(
                trigger_id=trigger['id'],
                fired_at_utc=datetime.now(timezone.utc),
                db_path=self.db_path
            )
            
            logger.info(f"Completed DAILY_SUMMARY for {date}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing DAILY_SUMMARY: {e}")
            return False
    
    def _process_active_game(self, game: dict) -> bool:
        """Process an active game for game-state triggers."""
        game_id = game['game_id']
        
        try:
            game_state = self.data_source.nba.fetch_game_state(game_id)
            
            if not game_state:
                return False
            
            # Update game in database
            GameStorage.upsert_game(
                game_id=game_id,
                start_time_utc=parse_iso_utc(game['start_time_utc']) if isinstance(game['start_time_utc'], str) else game['start_time_utc'],
                home_team=game['home_team'],
                away_team=game['away_team'],
                status=game_state['status'],
                current_period=game_state.get('current_period'),
                game_clock=game_state.get('game_clock'),
                score_home=game_state.get('score_home', 0),
                score_away=game_state.get('score_away', 0),
                db_path=self.db_path
            )
            
            # Check for game-state triggers
            triggers_fired = self.trigger_firer.process_game_state_triggers(
                game_id=game_id,
                game_state=game_state
            )
            
            if triggers_fired > 0:
                all_triggers = TriggerStorage.get_triggers_for_game(game_id, db_path=self.db_path)
                now_utc = datetime.now(timezone.utc)
                recent_cutoff = now_utc - timedelta(minutes=2)
                
                # FIX: Use created_at_utc instead of fired_at_utc
                # Game-state triggers are created on-the-fly with fired_at_utc=NULL
                # created_at_utc is set immediately, allowing them to be picked up
                recent_triggers = [
                    t for t in all_triggers
                    if t['created_at_utc'] and 
                    parse_iso_utc(t['created_at_utc']) > recent_cutoff
                ]
                
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
    
    def _process_game_state_trigger(self, game_id: str, trigger_type: str, game_state: dict) -> bool:
        """Process a game-state trigger."""
        try:
            logger.info(f"Processing game-state trigger: {game_id} {trigger_type}")
            
            data = self.data_source.refresh_game_data(
                game_id=game_id,
                reason=trigger_type,
                db_path=self.db_path
            )
            
            if not data['game_state']:
                return False
            
            picks = self.analysis_engine.run_analysis(
                game_state=data['game_state'],
                odds=data['odds'],
                mode=trigger_type
            )
            
            if not picks:
                return False
            
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
                        channel_id='main',
                        message_id=message_id,
                        payload={
                            'message': message,
                            'picks': picks,
                            'game_state': data['game_state']
                        },
                        db_path=self.db_path
                    )
            
            # Mark trigger as fired
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
    
    def _create_periodic_snapshots(self, games: list):
        """Create periodic tracking snapshots."""
        now_utc = datetime.now(timezone.utc)
        
        for game in games:
            game_id = game['game_id']
            
            try:
                data = self.data_source.refresh_game_data(
                    game_id=game_id,
                    reason='PERIODIC',
                    db_path=self.db_path
                )
                
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
        """Main loop."""
        logger.info(f"Starting unified runner (poll_interval={self.poll_interval}s)")
        self.running = True
        
        while self.running:
            try:
                stats = self.run_once()
                
                if stats['day_transition']:
                    logger.info(f"Day transition - {stats['new_games']} new games scheduled")
                
                if stats['triggers_processed'] > 0:
                    logger.info(f"Processed {stats['triggers_processed']} triggers this cycle")
                
                time.sleep(self.poll_interval)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
        
        logger.info("Unified runner stopped")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='PerryPicks v4 Unified Runner',
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
            logging.FileHandler('logs/unified_automation.log'),
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
    runner = UnifiedRunner(
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
        stats = runner.run_once()
        logger.info(f"Stats: {stats}")
    else:
        logger.info("Starting continuous unified automation")
        runner.run()


if __name__ == '__main__':
    main()
