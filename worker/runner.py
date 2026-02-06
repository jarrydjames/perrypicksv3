"""
Main runner for PerryPicks v4 Automation System.
Local event-driven automation: monitors games, fires triggers, posts to Discord.
"""

import logging
import signal
import sys
import datetime  # Keep timedelta for time arithmetic
from pathlib import Path
from typing import Optional, Dict, Any
import argparse
import os

import pendulum
from core.timezone import parse_iso_utc

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
from worker.scheduler import TriggerScheduler, AutoScheduler
from worker.triggers import TriggerFirer
from core.timezone import now_utc, to_iso, parse_date_str
from core.validation import validate_schedule_date, validate_system_clock

logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


class AutomationRunner:
    """Main automation runner - processes triggers and posts to Discord."""
    
    def __init__(
        self,
        db_path: Path,
        odds_api_key: str,
        discord_webhook_url: str,
        date: str,  # MUST be explicit YYYY-MM-DD format
        poll_interval: int = 60,
        dry_run: bool = False,
        auto_schedule: bool = True,  # Enable automatic game scheduling
        auto_schedule_days: int = 3,  # Schedule N days ahead
    ):
        self.db_path = db_path
        self.poll_interval = poll_interval
        self.dry_run = dry_run
        self.running = False
        
        # Auto-scheduling settings
        self.auto_schedule = auto_schedule
        self.auto_schedule_days = auto_schedule_days
        
        # Validate date format (no more 'today' support!)
        if date == 'today':
            raise ValueError(
                "Relative dates not supported. Use explicit YYYY-MM-DD format. "
                "Example: --date 2025-02-03"
            )
        
        # Validate date is reasonable (not in past, not too far in future)
        try:
            validate_schedule_date(date)
        except ValueError as e:
            raise ValueError(f"Invalid date '{date}': {e}") from e
        
        self.date = date
        
        # Initialize components
        self.data_source = CombinedDataSource(odds_api_key)
        self.discord_client = DiscordWebhookClient(discord_webhook_url)
        self.analysis_engine = AnalysisEngine()
        self.scheduler = TriggerScheduler(db_path)
        self.auto_scheduler = AutoScheduler(db_path)  # Auto-schedule games
        self.trigger_firer = TriggerFirer(db_path, dry_run)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}; shutting down gracefully...")
        self.running = False
    
    def initialize(self) -> bool:
        """Initialize database and schedule triggers for games."""
        try:
            # Initialize database
            init_database(self.db_path)
            logger.info("Database initialized")
            
            # Validate system clock
            clock_result = validate_system_clock()
            if not clock_result['valid']:
                logger.warning(f"System clock issue detected: {clock_result['warning']}")
            else:
                logger.info(f"System clock validated: {clock_result['drift_seconds']:.1f}s drift")
            
            # Schedule games for the specified date
            # (Date was already validated in __init__)
            games_scheduled = self.scheduler.schedule_games_for_date(self.date)
            
            logger.info(f"Initialized {games_scheduled} games for {self.date}")
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
        now_utc_val = now_utc()
        window_start = now_utc_val - pendulum.duration(minutes=2)
        window_end = now_utc_val + pendulum.duration(seconds=30)
        
        total_processed = 0
        
        # 0. Auto-schedule upcoming games (if enabled)
        if self.auto_schedule:
            try:
                results = self.auto_scheduler.auto_schedule_upcoming_games(
                    days_ahead=self.auto_schedule_days
                )
                if results['games_scheduled'] > 0:
                    logger.info(
                        f"Auto-scheduled {results['games_scheduled']} new games "
                        f"for {len(results['dates_scheduled'])} date(s)"
                    )
            except Exception as e:
                logger.error(f"Auto-scheduling failed: {e}", exc_info=True)
        
        # 1. Process scheduled time-based triggers", 
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
            
            # Handle DAILY_SUMMARY trigger (special case)
            if trigger_type == 'DAILY_SUMMARY':
                return self._process_daily_summary(trigger)
            
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
            
            # Handle PRE_GAME trigger (special formatting with odds and top 3 bets)
            if trigger_type == 'PRE_GAME':
                return self._process_pre_game_trigger(trigger, data, picks)
            
            # Handle HALFTIME trigger (special formatting without bets)
            if trigger_type == 'HALFTIME':
                return self._process_halftime_trigger(game_id, data['game_state'])
            
            # Store picks for other trigger types (Q3)
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
                    timestamp=now_utc()
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
                fired_at_utc=now_utc(),
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
            
            # Handle HALFTIME trigger (special formatting without bets)
            if trigger_type == 'HALFTIME':
                return self._process_halftime_trigger(game_id, data['game_state'])
            
            # Store picks for other trigger types (Q3)
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
                    timestamp=now_utc()
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
                fired_at_utc=now_utc(),
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
                now_utc = now_utc()
                recent_cutoff = now_utc - pendulum.duration(minutes=2)
                
                # FIX: Use created_at_utc instead of fired_at_utc
                # Game-state triggers are created on-the-fly with fired_at_utc=NULL
                # created_at_utc is set immediately, allowing them to be picked up
                recent_triggers = [
                    t for t in all_triggers
                    if t['created_at_utc'] and 
                    parse_iso_utc(t['created_at_utc']) > recent_cutoff
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
        current_time = now_utc()
        
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
                    timestamp_utc=to_iso(current_time),
                    poll_type='periodic',
                    quarter=game.get('current_period'),
                    game_clock=game.get('game_clock'),
                    score_home=game.get('score_home', 0),
                    score_away=game.get('score_away', 0),
                    db_path=self.db_path
                )
                
            except Exception as e:
                logger.error(f"Error creating periodic snapshot for {game_id}: {e}")
    
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
                        import time
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
                    timestamp=now_utc(),
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
                fired_at_utc=now_utc(),
                db_path=self.db_path
            )
            
            logger.info(f"Completed DAILY_SUMMARY for {date}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing DAILY_SUMMARY: {e}")
            return False
    
    def _process_pre_game_trigger(
        self,
        trigger: dict,
        data: dict,
        picks: list
    ) -> bool:
        """
        Process PRE_GAME trigger - post prediction with odds and top 3 bets.
        
        Args:
            trigger: PRE_GAME trigger
            data: Game data with game_state and odds
            picks: Analysis picks
        
        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            game_id = trigger['game_id']
            game_state = data['game_state']
            odds = data['odds']
            
            # Store picks
            for pick in picks:
                PickStorage.store_pick(
                    game_id=game_id,
                    trigger_type='PRE_GAME',
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
                # Get prediction from pregame model
                from src.predict_api import predict_game
                prediction_result = predict_game(
                    game_input=game_id,
                    mode='pregame',
                    fetch_odds=False
                )
                
                if prediction_result.get('status') == 'success':
                    # Add prediction to game_state for formatting
                    total = prediction_result.get('total', 0)
                    margin = prediction_result.get('margin', 0)
                    pred_home = (total + margin) / 2
                    pred_away = (total - margin) / 2
                    
                    if margin > 0:
                        pred_winner = prediction_result.get('home_name', 'Home')
                    else:
                        pred_winner = prediction_result.get('away_name', 'Away')
                    
                    game_state['predicted_away_score'] = pred_away
                    game_state['predicted_home_score'] = pred_home
                    game_state['predicted_total'] = total
                    game_state['predicted_margin'] = margin
                    game_state['predicted_winner'] = pred_winner
                    
                    # Use team names instead of tricodes for display
                    game_state['away_name'] = prediction_result.get('away_name', game_state.get('away_team', 'Away'))
                    game_state['home_name'] = prediction_result.get('home_name', game_state.get('home_team', 'Home'))
                
                message = self.discord_client.format_bet_post(
                    trigger_type='PRE_GAME',
                    game_data=game_state,
                    picks=picks[:3],  # Top 3 bets with highest edge
                    timestamp=now_utc()
                )
                
                message_id = self.discord_client.post_message(message)
                
                if message_id:
                    DiscordPostStorage.store_post(
                        game_id=game_id,
                        trigger_type='PRE_GAME',
                        channel_id='main',
                        message_id=message_id,
                        payload={
                            'message': message,
                            'picks': picks[:3],
                            'game_state': game_state
                        },
                        db_path=self.db_path
                    )
            
            # Mark trigger as fired
            TriggerStorage.mark_triggered(
                trigger_id=trigger['id'],
                fired_at_utc=now_utc(),
                db_path=self.db_path
            )
            
            logger.info(f"Completed PRE_GAME trigger for {game_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing PRE_GAME trigger for {trigger['game_id']}: {e}")
            return False
    
    def _process_halftime_trigger(
        self,
        game_id: str,
        game_state: Dict[str, Any]
    ) -> bool:
        """
        Process HALFTIME trigger - post halftime prediction without bets.
        
        Args:
            game_id: NBA game ID
            game_state: Current game state from NBA API
        
        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            logger.info(f"Processing HALFTIME trigger: {game_id}")
            
            # Get halftime prediction
            from src.predict_api import predict_game
            prediction_result = predict_game(
                game_input=game_id,
                mode='halftime',
                fetch_odds=False  # Don't need odds for halftime
            )
            
            if prediction_result.get('status') != 'success':
                logger.warning(f"Halftime prediction failed for {game_id}")
                return False
            
            # Extract prediction data
            total = prediction_result.get('total', 0)
            margin = prediction_result.get('margin', 0)
            pred_home = (total + margin) / 2
            pred_away = (total - margin) / 2
            
            # Determine winner
            if margin > 0:
                pred_winner = prediction_result.get('home_name', 'Home')
            else:
                pred_winner = prediction_result.get('away_name', 'Away')
            
            prediction = {
                'predicted_away_score': pred_away,
                'predicted_home_score': pred_home,
                'predicted_total': total,
                'predicted_margin': margin,
                'predicted_winner': pred_winner
            }
            
            # Add team names to game_state
            game_state['away_name'] = prediction_result.get('away_name', game_state.get('away_team', 'Away'))
            game_state['home_name'] = prediction_result.get('home_name', game_state.get('home_team', 'Home'))
            
            # Post to Discord (if not dry run)
            if not self.dry_run:
                message = self.discord_client.format_halftime_post(
                    game_data=game_state,
                    prediction=prediction,
                    timestamp=now_utc()
                )
                
                message_id = self.discord_client.post_message(message)
                
                if message_id:
                    DiscordPostStorage.store_post(
                        game_id=game_id,
                        trigger_type='HALFTIME',
                        channel_id='main',
                        message_id=message_id,
                        payload={
                            'message': message,
                            'prediction': prediction,
                            'game_state': game_state
                        },
                        db_path=self.db_path
                    )
            
            # Mark HALFTIME trigger as fired (it was already stored in triggers table)
            all_triggers = TriggerStorage.get_triggers_for_game(game_id, db_path=self.db_path)
            halftime_trigger = [
                t for t in all_triggers 
                if t['trigger_type'] == 'HALFTIME' and t['status'] == 'scheduled'
            ]
            
            if halftime_trigger:
                TriggerStorage.mark_triggered(
                    trigger_id=halftime_trigger[0]['id'],
                    fired_at_utc=now_utc(),
                    db_path=self.db_path
                )
            
            logger.info(f"Completed HALFTIME trigger for {game_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing HALFTIME trigger for {game_id}: {e}")
            return False
    
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
    
    # Convert 'today' to actual date (use UTC date for consistency)
    if args.date == 'today':
        from core.timezone import now_utc
        date_str = now_utc().format('YYYY-MM-DD')
        logger.info(f"Converting 'today' to date: {date_str}")
    else:
        date_str = args.date
    
    # Create and initialize runner
    runner = AutomationRunner(
        db_path=db_path,
        odds_api_key=odds_api_key,
        discord_webhook_url=discord_webhook_url,
        poll_interval=args.poll_interval,
        dry_run=args.dry_run,
        date=date_str
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