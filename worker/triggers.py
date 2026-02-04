"""
Game-state trigger detection for PerryPicks v4 Automation System.
Detects halftime and end of Q3 triggers from live game data.
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

import pendulum

from core.storage import GameStorage, TriggerStorage, TrackingStorage
from core.timezone import now_utc, to_iso
from worker.scheduler import GameStateTracker

logger = logging.getLogger(__name__)


class GameTriggerDetector:
    """Detects game-state triggers (halftime, Q3 end)."""
    
    HALFTIME = 'HALFTIME'
    Q3 = 'Q3'
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
    
    def detect_triggers(self, game_id: str, game_state: Dict[str, Any]) -> List[str]:
        """
        Detect which game-state triggers should fire for a game.
        
        Args:
            game_id: NBA game ID
            game_state: Current game state from NBA API
        
        Returns:
            List of trigger types that should fire (HALFTIME, Q3)
        """
        triggers_to_fire = []
        status = game_state.get('status', '')
        current_period = game_state.get('current_period', 0)
        game_clock = game_state.get('game_clock', '0:00')
        
        # Only check if game is in progress or at halftime
        if not GameStateTracker.is_game_in_progress(status):
            return []
        
        # Detect Halftime
        if self._is_halftime(status, game_state):
            if self._should_fire_trigger(game_id, self.HALFTIME):
                triggers_to_fire.append(self.HALFTIME)
        
        # Detect End of Q3
        if self._is_end_of_q3(status, current_period, game_clock, game_state):
            if self._should_fire_trigger(game_id, self.Q3):
                triggers_to_fire.append(self.Q3)
        
        return triggers_to_fire
    
    def _is_halftime(
        self,
        status: str,
        game_state: Dict[str, Any]
    ) -> bool:
        """
        Detect if game is at halftime.
        
        Rules:
        - Status is 'Halftime' OR
        - Status is 'In Progress' with period 2 and clock '12:00'
        """
        # Check for explicit halftime status
        if status == 'Halftime':
            return True
        
        # Check for period 2 with full clock
        current_period = game_state.get('current_period', 0)
        game_clock = game_state.get('game_clock', '0:00')
        
        if current_period == 2 and self._is_full_period_clock(game_clock):
            return True
        
        return False
    
    def _is_end_of_q3(
        self,
        status: str,
        current_period: int,
        game_clock: str,
        game_state: Dict[str, Any]
    ) -> bool:
        """
        Detect if game just ended Q3.
        
        Rules:
        - Period 3 AND clock is 0:00 OR
        - Transition from Q3 to Q4 (period 4)
        """
        # Check if we're in Q3 and clock is full
        if current_period == 3 and self._is_full_period_clock(game_clock):
            return True
        
        # Check if transitioned to Q4
        if current_period == 4:
            # Verify previous state was Q3
            last_snapshot = self._get_last_snapshot(game_state.get('game_id'))
            if last_snapshot:
                last_period = last_snapshot.get('quarter', 0)
                if last_period == 3:
                    return True
        
        return False
    
    def _is_full_period_clock(self, clock: str) -> bool:
        """Check if clock shows full period (e.g., '12:00', '0:00')."""
        try:
            # Parse clock (format 'MM:SS' or 'M:SS')
            parts = clock.split(':')
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes >= 12 and seconds == 0
        except:
            pass
        return False
    
    def _should_fire_trigger(self, game_id: str, trigger_type: str) -> bool:
        """
        Check if trigger should fire (not already fired).
        Uses DB dedupe to prevent duplicate triggers.
        
        IMPORTANT: Only check for FIRED triggers, not scheduled ones!
        This allows re-firing if a game reaches the state again after
        a missed scheduled trigger (e.g., wrong scheduling date).
        """
        # Check if trigger already FIRED (not just scheduled)
        if TriggerStorage.check_trigger_fired(game_id, trigger_type, db_path=self.db_path):
            logger.debug(f"Trigger {trigger_type} already fired for {game_id}")
            return False
        
        return True
    
    def _get_last_snapshot(self, game_id: str) -> Optional[Dict[str, Any]]:
        """Get most recent tracking snapshot for a game."""
        snapshots = TrackingStorage.get_timeseries(game_id, db_path=self.db_path)
        if snapshots:
            return snapshots[-1]  # Last snapshot
        return None


class TriggerFirer:
    """Fires triggers and executes associated actions."""
    
    def __init__(self, db_path: Path, dry_run: bool = False):
        self.db_path = db_path
        self.dry_run = dry_run
        self.detector = GameTriggerDetector(db_path)
    
    def process_game_state_triggers(self, game_id: str, game_state: Dict[str, Any]) -> int:
        """
        Process a game state and fire any detected triggers.
        
        Returns:
            Number of triggers fired
        """
        triggers_fired = 0
        
        # Detect triggers
        trigger_types = self.detector.detect_triggers(game_id, game_state)
        
        # Fire each detected trigger
        for trigger_type in trigger_types:
            if self._fire_trigger(game_id, trigger_type, game_state):
                triggers_fired += 1
        
        return triggers_fired
    
    def fire_trigger(self, game_id: str, trigger_type: str) -> bool:
        """
        Manually fire a trigger for a game.
        
        This is a public method used by the monitoring portal to manually
        trigger predictions (pre-game, halftime, Q3, etc.).
        
        Args:
            game_id: NBA game ID
            trigger_type: Type of trigger to fire (PRE_3H, HALFTIME, Q3, etc.)
        
        Returns:
            True if trigger was fired successfully, False otherwise
        """
        try:
            # Fetch current game state from database
            game_state = GameStorage.get_game(game_id, self.db_path)
            
            if not game_state:
                logger.error(f"Game {game_id} not found in database")
                return False
            
            # Fire the trigger
            success = self._fire_trigger(game_id, trigger_type, game_state)
            
            if success:
                logger.info(f"Manually fired {trigger_type} trigger for {game_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error manually firing {trigger_type} trigger for {game_id}: {e}")
            return False
    
    def _fire_trigger(
        self,
        game_id: str,
        trigger_type: str,
        game_state: Dict[str, Any]
    ) -> bool:
        """
        Fire a single trigger:
        1. Store trigger as fired
        2. Create tracking snapshot
        3. Return True (caller handles analysis and Discord)
        """
        try:
            now_utc = now_utc()
            
            # Store as fired trigger
            # Note: This is called when trigger is detected, not scheduled
            # So we need to mark it as fired retroactively
            
            # For game-state triggers, we create them on-the-fly
            trigger_id = self._create_fired_trigger(game_id, trigger_type, now_utc)
            
            if not trigger_id:
                return False
            
            # Create tracking snapshot
            TrackingStorage.store_snapshot(
                game_id=game_id,
                timestamp_utc=now_utc,
                poll_type='trigger',
                trigger_type=trigger_type,
                quarter=game_state.get('current_period'),
                game_clock=game_state.get('game_clock'),
                score_home=game_state.get('score_home'),
                score_away=game_state.get('score_away'),
                db_path=self.db_path
            )
            
            logger.info(f"Fired {trigger_type} trigger for {game_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error firing {trigger_type} trigger for {game_id}: {e}")
            return False
    
    def _create_fired_trigger(
        self,
        game_id: str,
        trigger_type: str,
        fired_at_utc: datetime
    ) -> Optional[int]:
        """Create a fired trigger entry (for game-state triggers)."""
        try:
            from core.storage import TriggerStorage
            return TriggerStorage.schedule_trigger(
                game_id=game_id,
                trigger_type=trigger_type,
                scheduled_time_utc=fired_at_utc,  # Scheduled time = fired time
                payload={'auto_detected': True},
                db_path=self.db_path
            )
        except Exception as e:
            logger.error(f"Error creating fired trigger: {e}")
            return None
