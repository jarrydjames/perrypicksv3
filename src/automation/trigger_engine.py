"""Trigger Engine for PerryPicks v3.

Evaluates game state against trigger rules (halftime, Q3-5min).
Fires predictions and auto-processes queue when conditions are met.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import json
from pathlib import Path

from src.automation.game_state_monitor import GameStateMonitor, GameState
from src.automation.auto_queue_processor import AutoQueueProcessor
from src.predict_api import predict_game
from core.storage import GameStorage

logger = logging.getLogger(__name__)


class TriggerType:
    """Trigger types."""
    HALFTIME = "halftime"
    Q3_5MIN = "q3_5min"


class TriggerEvent:
    """Represents a trigger event that was fired."""
    
    def __init__(
        self,
        game_id: str,
        trigger_type: str,
        fired_at: datetime,
        prediction: Optional[Dict[str, Any]] = None,
    ):
        self.game_id = game_id
        self.trigger_type = trigger_type
        self.fired_at = fired_at
        self.prediction = prediction
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "game_id": self.game_id,
            "trigger_type": self.trigger_type,
            "fired_at": self.fired_at.isoformat(),
            "prediction": self.prediction,
        }


class TriggerEngine:
    """Engine for evaluating and firing game state triggers.
    
    Monitors game states and fires predictions when:
    - Game reaches halftime (end of Q2)
    - Game reaches 5 minutes left in Q3
    
    Automatically processes queue after generating predictions.
    """
    
    def __init__(
        self,
        game_state_monitor: GameStateMonitor,
        queue_processor: AutoQueueProcessor,
        storage: Optional[GameStorage] = None,
    ):
        """Initialize trigger engine.
        
        Args:
            game_state_monitor: Game state monitor instance
            queue_processor: Auto queue processor instance
            storage: Game storage for tracking fired triggers
        """
        self.monitor = game_state_monitor
        self.processor = queue_processor
        self.storage = storage or GameStorage()
        
        # Track fired triggers to prevent duplicates
        self.fired_triggers: Set[str] = set()
        
        # Load previously fired triggers from storage
        self._load_fired_triggers()
        
        logger.info("Trigger Engine initialized")
    
    def _load_fired_triggers(self):
        """Load previously fired triggers from storage."""
        try:
            # TODO: Implement storage loading if needed
            # For now, start fresh
            logger.info("Loaded 0 previously fired triggers")
        except Exception as e:
            logger.warning(f"Error loading fired triggers: {e}")
    
    def _save_fired_trigger(self, trigger_key: str):
        """Save a fired trigger to storage."""
        try:
            # TODO: Implement storage saving if needed
            self.fired_triggers.add(trigger_key)
            logger.info(f"Saved fired trigger: {trigger_key}")
        except Exception as e:
            logger.warning(f"Error saving fired trigger: {e}")
    
    def _make_trigger_key(self, game_id: str, trigger_type: str) -> str:
        """Create unique key for a trigger."""
        return f"{game_id}_{trigger_type}"
    
    def _has_fired(self, game_id: str, trigger_type: str) -> bool:
        """Check if trigger has already fired."""
        key = self._make_trigger_key(game_id, trigger_type)
        return key in self.fired_triggers
    
    def _mark_fired(self, game_id: str, trigger_type: str):
        """Mark a trigger as fired."""
        key = self._make_trigger_key(game_id, trigger_type)
        self._save_fired_trigger(key)
    
    def evaluate_game(self, game_id: str, game_state: GameState) -> Optional[TriggerEvent]:
        """Evaluate triggers for a single game.
        
        Args:
            game_id: Game ID to evaluate
            game_state: Current game state
            
        Returns:
            TriggerEvent if triggered, None otherwise
        """
        # Check halftime trigger
        if game_state.status == "halftime":
            trigger_key = self._make_trigger_key(game_id, TriggerType.HALFTIME)
            
            if trigger_key not in self.fired_triggers:
                logger.info(f"HALFTIME TRIGGER: {game_id}")
                
                # Generate prediction
                prediction = self._generate_prediction(game_id, TriggerType.HALFTIME)
                
                if prediction and prediction.get("status") in ("success", "warning"):
                    # Mark as fired
                    self._mark_fired(game_id, TriggerType.HALFTIME)
                    
                    # Create trigger event
                    event = TriggerEvent(
                        game_id=game_id,
                        trigger_type=TriggerType.HALFTIME,
                        fired_at=datetime.now(),
                        prediction=prediction,
                    )
                    
                    return event
        
        # Check Q3-5min trigger
        if self.monitor.is_q3_five_minutes_left(game_id):
            trigger_key = self._make_trigger_key(game_id, TriggerType.Q3_5MIN)
            
            if trigger_key not in self.fired_triggers:
                logger.info(f"Q3-5MIN TRIGGER: {game_id}")
                
                # Generate prediction
                prediction = self._generate_prediction(game_id, TriggerType.Q3_5MIN)
                
                if prediction and prediction.get("status") in ("success", "warning"):
                    # Mark as fired
                    self._mark_fired(game_id, TriggerType.Q3_5MIN)
                    
                    # Create trigger event
                    event = TriggerEvent(
                        game_id=game_id,
                        trigger_type=TriggerType.Q3_5MIN,
                        fired_at=datetime.now(),
                        prediction=prediction,
                    )
                    
                    return event
        
        return None
    
    def _generate_prediction(
        self,
        game_id: str,
        trigger_type: str,
        fetch_odds: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Generate prediction for a trigger.
        
        Args:
            game_id: Game ID to predict for
            trigger_type: Type of trigger (halftime, q3)
            fetch_odds: Whether to fetch odds
            
        Returns:
            Prediction dictionary or None if failed
        """
        try:
            # Map trigger type to prediction mode
            mode = "halftime" if trigger_type == TriggerType.HALFTIME else "q3"
            
            logger.info(f"Generating {mode} prediction for {game_id}...")
            
            # Generate prediction
            # For in-progress games (halftime, q3), bypass the import gate
            # This allows predictions even if schedule has placeholder teams (UNK @ UNK)
            # The actual boxscore data will have real team names
            bypass_gate = mode in ('halftime', 'q3')
            if bypass_gate:
                logger.info(f"Bypassing import gate for {game_id} (mode={mode})")
            
            prediction = predict_game(
                game_id=game_id,
                mode=mode,
                fetch_odds=fetch_odds,
                bypass_import_gate=bypass_gate,
            )
            
            if prediction and prediction.get("status") in ("success", "warning"):
                logger.info(
                    f"Prediction generated for {game_id}: "
                    f"{prediction.get('status')}"
                )
            else:
                logger.warning(
                    f"Prediction failed for {game_id}: "
                    f"{prediction.get('error', 'Unknown') if prediction else 'No prediction'}"
                )
            
            return prediction
        
        except Exception as e:
            logger.error(f"Error generating prediction for {game_id}: {e}")
            return None
    
    def process_trigger_event(self, event: TriggerEvent, platforms: List[str]) -> bool:
        """Process a trigger event - queue prediction and process.
        
        Args:
            event: Trigger event to process
            platforms: Platforms to post to
            
        Returns:
            True if processed successfully
        """
        try:
            logger.info(
                f"Processing trigger event: {event.game_id} "
                f"({event.trigger_type})"
            )
            
            # Queue prediction for posting
            prediction = event.prediction
            if not prediction:
                logger.error(f"No prediction to queue for {event.game_id}")
                return False
            
            # Use queue processor to post
            result = self.processor.queue_and_post(
                prediction=prediction,
                trigger_type=event.trigger_type,
                platforms=platforms,
            )
            
            if result.get("success"):
                logger.info(
                    f"Successfully queued and posted: {event.game_id} "
                    f"({event.trigger_type})"
                )
                return True
            else:
                logger.error(
                    f"Failed to queue/post: {event.game_id} "
                    f"({event.trigger_type}): {result.get('error', 'Unknown')}"
                )
                return False
        
        except Exception as e:
            logger.error(f"Error processing trigger event: {e}")
            return False
    
    def evaluate_all(self, platforms: List[str]) -> List[TriggerEvent]:
        """Evaluate triggers for all active games.
        
        Args:
            platforms: Platforms to post to
            
        Returns:
            List of fired TriggerEvent objects
        """
        fired_events = []
        
        try:
            # Get all game states
            game_states = self.monitor.get_all_states()
            
            logger.info(f"Evaluating triggers for {len(game_states)} games")
            
            # Evaluate each game
            for game_id, game_state in game_states.items():
                # Skip finished games
                if game_state.status == "finished":
                    continue
                
                # Evaluate triggers
                event = self.evaluate_game(game_id, game_state)
                
                if event:
                    # Process the event
                    success = self.process_trigger_event(event, platforms)
                    
                    if success:
                        fired_events.append(event)
            
            logger.info(f"Fired {len(fired_events)} trigger(s)")
        
        except Exception as e:
            logger.error(f"Error evaluating all triggers: {e}")
        
        return fired_events
    
    def get_stats(self) -> Dict[str, Any]:
        """Get trigger engine statistics."""
        return {
            "fired_triggers_count": len(self.fired_triggers),
            "games_monitored": len(self.monitor.get_all_states()),
        }
