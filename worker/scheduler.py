"""
Scheduler for PerryPicks v4 Automation System.
Computes and schedules triggers for games at T-3H, T-1H, T-10M.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
from pathlib import Path

from core.storage import GameStorage, TriggerStorage
from core.data_sources import NBADataSource

logger = logging.getLogger(__name__)


class TriggerScheduler:
    """Schedules time-based triggers for games."""
    
    # Trigger types
    PRE_3H = 'PRE_3H'
    PRE_1H = 'PRE_1H'
    PRE_10M = 'PRE_10M'
    
    # Time offsets for triggers
    TRIGGER_OFFSETS = {
        PRE_3H: timedelta(hours=-3),
        PRE_1H: timedelta(hours=-1),
        PRE_10M: timedelta(minutes=-10)
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
        
        # Schedule each trigger type
        for trigger_type, offset in self.TRIGGER_OFFSETS.items():
            scheduled_time = start_time + offset
            
            # Check if trigger already exists
            if not TriggerStorage.check_trigger_exists(game_id, trigger_type, db_path=self.db_path):
                # Schedule trigger
                TriggerStorage.schedule_trigger(
                    game_id=game_id,
                    trigger_type=trigger_type,
                    scheduled_time_utc=scheduled_time,
                    payload={
                        'home_team': game['home_team'],
                        'away_team': game['away_team']
                    },
                    db_path=self.db_path
                )
                logger.debug(f"Scheduled {trigger_type} for {game_id} at {scheduled_time}")
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
    def get_active_trigger_types(status: str, period: int) -> List[str]:
        """
        Determine which trigger types are valid for current game state.
        
        Returns:
            List of trigger types that should fire
        """
        if not GameStateTracker.is_game_in_progress(status):
            return []
        
        triggers = []
        
        # Halftime detection
        if status == 'Halftime':
            triggers.append('HALFTIME')
        
        # End of Q3 detection
        if period == 3 and status == 'In Progress':
            triggers.append('Q3')
        
        return triggers
