
"""
Scheduler for PerryPicks v4 Automation System.
Computes and schedules triggers for games at T-3H, T-1H, T-10M.
"""

import logging
from datetime import timedelta  # Keep timedelta for time arithmetic
from typing import List, Dict, Any
from pathlib import Path

import pendulum

from core.storage import GameStorage, TriggerStorage
from core.data_sources import NBADataSource
from core.timezone import now_utc, to_iso, parse_date_str
from core.validation import validate_schedule_date
from core.data_sources import NBADataSource

logger = logging.getLogger(__name__)


class TriggerScheduler:
    """Schedules time-based triggers for games."""
    
    # Trigger types
    DAILY_SUMMARY = 'DAILY_SUMMARY'  # 3h before earliest game
    PRE_GAME = 'PRE_GAME'           # 1h before each game
    HALFTIME = 'HALFTIME'          # At halftime
    
    # Time offsets for triggers (using pendulum.duration)
    TRIGGER_OFFSETS = {
        PRE_GAME: pendulum.duration(hours=-1),
    }
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.nba_source = NBADataSource()
    
    def schedule_games_for_date(self, date: str) -> int:
        """
        Fetch games for a date and schedule all time-based triggers.
        
        Args:
            date: Date in YYYY-MM-DD format
        
        Returns:
            Number of games processed
        """
        # Fetch games from NBA API
        games = self.nba_source.fetch_games_for_date(date)
        
        if not games:
            logger.info(f"No games found for date {date}")
            return 0
        
        # Sort games by start time
        games_sorted = sorted(games, key=lambda g: g['start_time_utc'])
        
        # Schedule DAILY_SUMMARY trigger (3h before earliest game)
        if games_sorted:
            earliest_game = games_sorted[0]
            summary_time = earliest_game['start_time_utc'] + timedelta(hours=-3)
            
            # Store as a special game_id for daily summary
            summary_game_id = f"DAILY_{date.replace('-', '')}"
            
            if not TriggerStorage.check_trigger_exists(summary_game_id, self.DAILY_SUMMARY, db_path=self.db_path):
                # Convert datetime objects to ISO strings for JSON serialization
                games_serializable = []
                for game in games:
                    game_copy = game.copy()
                    # Convert datetime to ISO string
                    if 'start_time_utc' in game_copy and isinstance(game_copy['start_time_utc'], pendulum.DateTime):
                        game_copy['start_time_utc'] = to_iso(game_copy['start_time_utc'])
                    games_serializable.append(game_copy)
                
                TriggerStorage.schedule_trigger(
                    game_id=summary_game_id,
                    trigger_type=self.DAILY_SUMMARY,
                    scheduled_time_utc=summary_time,
                    payload={
                        'date': date,
                        'games': games_serializable
                    },
                    db_path=self.db_path
                )
                logger.info(f"Scheduled DAILY_SUMMARY for {date} at {summary_time} UTC")
        
        scheduled_count = 0
        for game in games:
            # Upsert game into database
            GameStorage.upsert_game(
                game_id=game['game_id'],
                start_time_utc=game['start_time_utc'],
                home_team=game['home_team'],
                away_team=game['away_team'],
                status=game['status'],
                game_date=game['game_date'],
                db_path=self.db_path
            )
            
            # Schedule triggers for this game
            scheduled = self._schedule_game_triggers(game)
            if scheduled:
                scheduled_count += 1
        
        logger.info(f"Scheduled triggers for {scheduled_count} games on {date}")
        return scheduled_count
    
    def _schedule_game_triggers(self, game: Dict[str, Any]) -> bool:
        """
        Schedule all time-based triggers for a single game.
        Only schedules if triggers don't already exist.
        
        Returns:
            True if any new triggers scheduled, False otherwise
        """
        game_id = game['game_id']
        start_time = game['start_time_utc']
        
        any_scheduled = False
        
        # Schedule PRE_GAME trigger (1h before game)
        scheduled_time = start_time + timedelta(hours=-1)
        
        if not TriggerStorage.check_trigger_exists(game_id, self.PRE_GAME, db_path=self.db_path):
            TriggerStorage.schedule_trigger(
                game_id=game_id,
                trigger_type=self.PRE_GAME,
                scheduled_time_utc=scheduled_time,
                payload={
                    'home_team': game['home_team'],
                    'away_team': game['away_team']
                },
                db_path=self.db_path
            )
            any_scheduled = True
        
        return any_scheduled
    
    def reschedule_if_needed(self, game: Dict[str, Any]) -> bool:
        """
        Reschedule triggers if game start time changed.
        
        Returns:
            True if rescheduled, False otherwise
        """
        game_id = game['game_id']
        new_start_time = game['start_time_utc']
        
        # Get existing game from DB
        existing_game = GameStorage.get_game(game_id, db_path=self.db_path)
        
        if not existing_game:
            logger.warning(f"Game {game_id} not in database; scheduling from scratch")
            return self._schedule_game_triggers(game)
        
        old_start_time = existing_game['start_time_utc']
        
        # Check if start time changed significantly (> 1 minute difference)
        if abs((new_start_time - old_start_time).total_seconds()) > 60:
            logger.info(f"Game {game_id} start time changed; rescheduling triggers")
            
            # Delete existing triggers and reschedule
            # For now, we'll let unique constraint handle this
            # In production, you'd delete old triggers first
            return self._schedule_game_triggers(game)
        
        return False


class GameStateTracker:
    """Tracks game states for in-progress games."""
    
    @staticmethod
    def is_game_in_progress(status: str) -> bool:
        """Check if game is currently in progress."""
        return status in ['In Progress', 'Halftime']
    
    @staticmethod
    def _parse_game_clock_minutes(game_clock: str) -> Optional[float]:
        """
        Parse NBA game_clock format to extract minutes remaining.
        
        Format: "PT10M30.000S" = 10 minutes 30 seconds remaining
        
        Returns:
            Minutes remaining as float, or None if game_clock is empty/invalid
        """
        if not game_clock or game_clock.strip() == '':
            return None
        
        try:
            # Remove "PT" prefix and "S" suffix
            # e.g., "PT10M30.000S" -> "10M30.000"
            clock_str = game_clock.replace("PT", "").replace("S", "")
            
            # Split by "M" to get minutes and seconds
            # e.g., "10M30.000" -> ["10", "30.000"]
            parts = clock_str.split("M")
            
            if len(parts) != 2:
                return None
            
            minutes = int(parts[0])
            seconds = float(parts[1])
            
            # Convert to minutes
            total_minutes = minutes + (seconds / 60.0)
            
            return total_minutes
        except Exception as e:
            logger.error(f"Error parsing game_clock '{game_clock}': {e}")
            return None
    
    @staticmethod
    def get_active_trigger_types(
        status: str, 
        period: int, 
        game_clock: Optional[str] = None
    ) -> List[str]:
        """
        Determine which trigger types are valid for current game state.
        
        Args:
            status: Game status from NBA API
            period: Current quarter (1-4, 0 for halftime)
            game_clock: Time remaining in current period (e.g., "PT10M30.000S")
        
        Returns:
            List of trigger types that should fire
        """
        if not GameStateTracker.is_game_in_progress(status):
            return []
        
        triggers = []
        
        # Halftime detection
        if status == 'Halftime':
            triggers.append('HALFTIME')
        
        # Q3 detection - fire when 5 minutes or less remain in Q3
        if period == 3 and status == 'In Progress':
            # Parse game clock to get minutes remaining
            minutes_remaining = GameStateTracker._parse_game_clock_minutes(game_clock)
            
            # Fire Q3 trigger when 5 minutes or less remain
            if minutes_remaining is not None and minutes_remaining <= 5:
                triggers.append('Q3')
        
        return triggers


class AutoScheduler:
    """Automatically schedules games for upcoming dates."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.trigger_scheduler = TriggerScheduler(db_path)
    
    def auto_schedule_upcoming_games(
        self,
        days_ahead: int = 3,
        today_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Automatically schedule games for the next N days.
        
        Checks which dates already have games in the database and only
        schedules games for dates that don't have any.
        
        Args:
            days_ahead: Number of days to look ahead (default: 3)
            today_date: Optional explicit date for 'today' (YYYY-MM-DD)
                       If None, calculates today's date
        
        Returns:
            Dict with scheduling results:
            - dates_checked: List of dates checked
            - dates_scheduled: List of dates that had games scheduled
            - games_scheduled: Total number of games scheduled
            - dates_skipped: List of dates skipped (already have games)
            - errors: List of errors encountered
        """
        results = {
            'dates_checked': [],
            'dates_scheduled': [],
            'games_scheduled': 0,
            'dates_skipped': [],
            'errors': []
        }
        
        # Determine today's date
        if today_date is None:
            now_cst = now_utc().in_timezone('America/Chicago')
            today = now_cst.format('YYYY-MM-DD')
        else:
            today = today_date
        
        # Check upcoming dates
        for days in range(days_ahead + 1):  # +1 to include today
            target_date = (parse_date_str(today).add(days=days)).format('YYYY-MM-DD')
            results['dates_checked'].append(target_date)
            
            # Check if date already has games
            if GameStorage.has_games_for_date(target_date, db_path=self.db_path):
                logger.debug(f"Date {target_date} already has games, skipping")
                results['dates_skipped'].append(target_date)
                continue
            
            # Try to schedule games for this date
            try:
                games_count = self.trigger_scheduler.schedule_games_for_date(target_date)
                
                if games_count > 0:
                    logger.info(f"✅ Auto-scheduled {games_count} games for {target_date}")
                    results['dates_scheduled'].append(target_date)
                    results['games_scheduled'] += games_count
                else:
                    logger.info(f"ℹ️  No games available for {target_date} yet")
                    # Don't mark as skipped - might have games later
            except ValueError as e:
                if 'past' in str(e).lower():
                    logger.debug(f"Date {target_date} is in the past, skipping")
                    results['dates_skipped'].append(target_date)
                else:
                    error_msg = f"Failed to schedule {target_date}: {e}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
            except Exception as e:
                error_msg = f"Unexpected error scheduling {target_date}: {e}"
                logger.error(error_msg, exc_info=True)
                results['errors'].append(error_msg)
        
        # Log summary
        if results['dates_scheduled']:
            logger.info(
                f"Auto-scheduling complete: scheduled {results['games_scheduled']} games "
                f"for {len(results['dates_scheduled'])} date(s): {results['dates_scheduled']}"
            )
        elif not results['dates_skipped']:
            # Only log if nothing was scheduled AND nothing was skipped (all dates failed)
            logger.warning(
                f"Auto-scheduling: no games scheduled for any of {results['dates_checked']}"
            )
        
        return results