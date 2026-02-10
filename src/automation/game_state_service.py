"""Game State Service for PerryPicks v3.

Main background service that coordinates game monitoring,
trigger evaluation, and automatic queue processing.

This is the hands-off automation service that runs continuously.
"""

from __future__ import annotations
import logging
import signal
import sys
import time
from typing import List, Optional
from datetime import datetime

from src.automation.game_state_monitor import GameStateMonitor, GameState
from src.automation.trigger_engine import TriggerEngine, TriggerEvent
from src.automation.auto_queue_processor import AutoQueueProcessor
from src.automation.automation_orchestrator import AutomationOrchestrator
from src.automation.social_media_manager import SocialMediaManager
from src.automation import (
    PostQueue,
)
logger = logging.getLogger(__name__)


class GameStateService:
    """Main service for game-state-aware automation.
    
    Coordinates:
    - Game State Monitor (live game tracking)
    - Trigger Engine (evaluates halftime, Q3-5min)
    - Auto Queue Processor (posts automatically)
    
    Runs continuously as background service.
    """
    
    def __init__(
        self,
        poll_interval_seconds: int = 30,
        platforms: Optional[List[str]] = None,
        dry_run: bool = False,
    ):
        """Initialize game state service.
        
        Args:
            poll_interval_seconds: How often to poll games (default: 30s)
            platforms: Platforms to post to (None = all enabled)
            dry_run: If True, don't actually post
        """
        self.poll_interval = poll_interval_seconds
        self.platforms = platforms
        self.dry_run = dry_run
        
        # Initialize components
        self.orchestrator = AutomationOrchestrator(dry_run=dry_run)
        self.game_monitor = GameStateMonitor(
            poll_interval_seconds=poll_interval_seconds,
        )
        self.queue_processor = AutoQueueProcessor(
            social_manager=self.orchestrator.social_manager,
        )
        self.trigger_engine = TriggerEngine(
            game_state_monitor=self.game_monitor,
            queue_processor=self.queue_processor,
        )
        
        self.running = False
        self.stats = {
            "started_at": None,
            "games_monitored": 0,
            "triggers_fired": 0,
            "posts_processed": 0,
            "errors": 0,
        }
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(
            f"Game State Service initialized. "
            f"Poll interval: {poll_interval_seconds}s. "
            f"Dry run: {dry_run}"
        )
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
    
    def start(self):
        """Start the game state service.
        
        Runs continuously, monitoring games and firing triggers.
        """
        self.running = True
        self.stats["started_at"] = datetime.now()
        
        logger.info("="*60)
        logger.info("GAME STATE SERVICE STARTED")
        logger.info("="*60)
        logger.info(f"Poll Interval: {self.poll_interval}s")
        logger.info(f"Platforms: {self.platforms or 'All enabled'}")
        logger.info(f"Dry Run: {self.dry_run}")
        logger.info("="*60)
        
        while self.running:
            try:
                # 1. Update game states
                logger.info("[1/3] Updating game states...")
                updated_states = self.game_monitor.update_all_games()
                self.stats["games_monitored"] += len(updated_states)
                
                logger.info(f"Updated {len(updated_states)} game(s)")
                
                # 2. Evaluate triggers
                logger.info("[2/3] Evaluating triggers...")
                fired_events = self.trigger_engine.evaluate_all(
                    platforms=self.platforms,
                )
                
                self.stats["triggers_fired"] += len(fired_events)
                
                if fired_events:
                    logger.info("="*60)
                    logger.info(f"FIRED {len(fired_events)} TRIGGER(S)")
                    logger.info("="*60)
                    
                    for event in fired_events:
                        logger.info(
                            f"  - {event.game_id} ({event.trigger_type}) "
                            f"at {event.fired_at.strftime('%H:%M:%S')}"
                        )
                else:
                    logger.info("No triggers fired this cycle")
                
                # 3. Process any pending posts
                logger.info("[3/3] Processing pending posts...")
                process_result = self.queue_processor.process_pending(
                    max_posts=50,
                )
                
                if process_result.get("success"):
                    processed = process_result.get("processed", 0)
                    self.stats["posts_processed"] += processed
                    logger.info(f"Processed {processed} post(s)")
                
                # Log stats
                self._log_stats()
                
                # Wait for next cycle
                logger.info(f"Waiting {self.poll_interval}s until next cycle...")
                time.sleep(self.poll_interval)
            
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received, shutting down...")
                self.stop()
            
            except Exception as e:
                logger.error(f"Error in main service loop: {e}")
                self.stats["errors"] += 1
                
                # Wait before retrying
                logger.info("Waiting 60s before retrying...")
                time.sleep(60)
        
        logger.info("="*60)
        logger.info("GAME STATE SERVICE STOPPED")
        logger.info("="*60)
        self._log_final_stats()
    
    def stop(self):
        """Stop the game state service."""
        logger.info("Stopping Game State Service...")
        self.running = False
        
        # Stop monitor
        if self.game_monitor.running:
            self.game_monitor.stop()
    
    def _log_stats(self):
        """Log current statistics."""
        uptime = (datetime.now() - self.stats["started_at"]).total_seconds() if self.stats["started_at"] else 0
        
        logger.info("="*60)
        logger.info("SERVICE STATS")
        logger.info("="*60)
        logger.info(f"Uptime: {uptime:.0f}s ({uptime/60:.1f}min)")
        logger.info(f"Games Monitored: {self.stats['games_monitored']}")
        logger.info(f"Triggers Fired: {self.stats['triggers_fired']}")
        logger.info(f"Posts Processed: {self.stats['posts_processed']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info("="*60)
    
    def _log_final_stats(self):
        """Log final statistics on shutdown."""
        if self.stats["started_at"]:
            uptime = (datetime.now() - self.stats["started_at"]).total_seconds()
            logger.info("="*60)
            logger.info("FINAL STATS")
            logger.info("="*60)
            logger.info(f"Total Uptime: {uptime:.0f}s ({uptime/60:.1f}min)")
            logger.info(f"Total Games Monitored: {self.stats['games_monitored']}")
            logger.info(f"Total Triggers Fired: {self.stats['triggers_fired']}")
            logger.info(f"Total Posts Processed: {self.stats['posts_processed']}")
            logger.info(f"Total Errors: {self.stats['errors']}")
            logger.info("="*60)
    
    def get_status(self) -> dict:
        """Get current service status."""
        game_states = self.game_monitor.get_all_states()
        
        return {
            "running": self.running,
            "games_monitored": len(game_states),
            "games_list": [
                {**state.to_dict()}
                for state in game_states.values()
            ],
            "stats": self.stats.copy(),
        }


def main():
    """Main entry point for game state service.
    
    Run as: python -m src.automation.game_state_service
    
    Environment variables:
        GAME_STATE_POLL_INTERVAL: Poll interval in seconds (default: 30)
        GAME_STATE_PLATFORMS: Comma-separated platforms (default: all enabled)
        GAME_STATE_DRY_RUN: "true" to run without posting (default: false)
    """
    import os
    
    # Get configuration from environment
    poll_interval = int(os.getenv("GAME_STATE_POLL_INTERVAL", "30"))
    platforms_str = os.getenv("GAME_STATE_PLATFORMS", "")
    platforms = platforms_str.split(",") if platforms_str else None
    dry_run = os.getenv("GAME_STATE_DRY_RUN", "false").lower() == "true"
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    # Create and start service
    service = GameStateService(
        poll_interval_seconds=poll_interval,
        platforms=platforms,
        dry_run=dry_run,
    )
    
    try:
        service.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
