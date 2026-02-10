"""Automation Orchestrator for PerryPicks v3.

Coordinates predictions and posting automation end-to-end.
"""

from __future__ import annotations
import logging
import time
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from pathlib import Path
import signal
import os

import pendulum

import schedule

from src.automation.social_media_manager import SocialMediaManager
from src.automation.post_queue import PostQueue
from src.predict_api import predict_game
from core.env import load_environment
from core.storage import init_database, GameStorage
from core.timezone import now_utc, to_iso
from core.qol import canonical_pick_id

logger = logging.getLogger(__name__)

class AutomationOrchestrator:
    """Orchestrate end-to-end automation."""
    
    def __init__(
        self,
        storage_path: Path = None,
        dry_run: bool = False,
        platforms: Optional[List[str]] = None,
    ):
        """
        Initialize automation orchestrator.
        
        Args:
            storage_path: Path to storage directory
            dry_run: If True, don't actually post
            platforms: List of platforms to post (None = all)
        """
        self.storage_path = storage_path or Path("data")
        self.dry_run = dry_run
        self.platforms = platforms
        self.running = False
        
        # Initialize components
        self.social_manager = SocialMediaManager(dry_run=dry_run)
        
        # Initialize database
        db_path = self.storage_path / "automation.db"
        init_database(db_path)
        self.game_storage = GameStorage()
        
        # Track processed games (for deduplication)
        self.processed_predictions: Dict[str, Set[str]] = {}
        # game_id -> set of trigger_types processed
        
        # Setup signal handlers (only if in main thread)
        try:
            signal.signal(signal.SIGINT, self._handle_shutdown)
            signal.signal(signal.SIGTERM, self._handle_shutdown)
        except ValueError as e:
            # Can't set signal handlers if not in main thread
            # This happens when running in subprocess - that's OK
            logger.warning(f"Could not set signal handlers (not in main thread): {e}")
        
        logger.info(
            f"Automation Orchestrator initialized. "
            f"Dry run: {dry_run}, Platforms: {platforms or 'all enabled'}"
        )
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}; shutting down gracefully...")
        self.running = False
    
    def run_predictions(
        self,
        game_ids: List[str],
        trigger_type: str = "pregame",
        mode: str = "auto",
        fetch_odds: bool = True,
        allow_duplicates: bool = False,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        Run predictions for a list of games.
        
        Args:
            game_ids: List of game IDs to predict
            trigger_type: Trigger type (pregame, halftime, q3)
            mode: Prediction mode (auto, pregame, halftime, q3)
            fetch_odds: If True, fetch odds from API (default True). Set False for testing.
            progress_callback: Optional callback(progress, message) for UI updates
            
        Returns:
            Results dictionary
        """
        results = {
            "trigger_type": trigger_type,
            "game_ids": game_ids,
            "total_games": len(game_ids),
            "predictions": [],
            "posted": [],
            "errors": [],
            "skipped": 0,  # Already processed
        }
        
        for i, game_id in enumerate(game_ids, 1):
            try:
                # Update progress
                progress = i / len(game_ids)
                message = f"Processing {game_id} ({i}/{len(game_ids)})..."
                logger.info(message)
                if progress_callback:
                    progress_callback(progress, message)
                
                # Check if already processed (unless allow_duplicates is True)
                if not allow_duplicates and self._is_prediction_processed(game_id, trigger_type):
                    logger.info(f"Skipping already processed: {game_id} {trigger_type}")
                    results["skipped"] += 1
                    if progress_callback:
                        progress_callback(progress, f"Skipped {game_id} (already processed)")
                    continue
                
                # Run prediction
                if progress_callback:
                    progress_callback(progress, f"Predicting {game_id}...")
                logger.info(f"Running prediction for {game_id} with mode={mode}, trigger_type={trigger_type}")
                
                # For in-progress games (halftime, q3), bypass the import gate
                # This allows predictions even if schedule has placeholder teams (UNK @ UNK)
                # The actual boxscore data will have real team names
                bypass_gate = mode in ('halftime', 'q3')
                if bypass_gate:
                    logger.info(f"Bypassing import gate for {game_id} (mode={mode})")
                
                prediction = predict_game(game_id, mode=mode, fetch_odds=fetch_odds, bypass_import_gate=bypass_gate)
                
                # Add trigger_type to prediction result for post_generator
                if isinstance(prediction, dict):
                    prediction['trigger_type'] = trigger_type
                
                # Log detailed prediction result
                logger.info(f"Prediction result for {game_id}:")
                logger.info(f"  Type: {type(prediction)}")
                if isinstance(prediction, dict):
                    logger.info(f"  Keys: {list(prediction.keys())}")
                    logger.info(f"  Status: {prediction.get('status', 'missing')}")
                    logger.info(f"  Model used: {prediction.get('model_used', 'missing')}")
                    logger.info(f"  Trigger type: {prediction.get('trigger_type', 'missing')}")
                    logger.info(f"  Error: {prediction.get('error', 'none')}")
                else:
                    logger.warning(f"Prediction is not a dict: {prediction}")
                
                results["predictions"].append(prediction)
                
                # Post to social media
                if prediction and prediction.get("status") in ("success", "warning"):
                    if progress_callback:
                        progress_callback(progress, f"Posting {game_id} to social media...")
                    post_results = self.social_manager.post_prediction(
                        prediction,
                        trigger_type=trigger_type,
                        platforms=self.platforms,
                        allow_duplicates=allow_duplicates,
                    )
                    logger.info(f"Post results for {game_id}: {post_results}")
                    results["posted"].append(post_results)
                    
                    # Mark as processed
                    self._mark_prediction_processed(game_id, trigger_type)
                    
                    # Count successful posts
                    platforms_dict = post_results.get('platforms', {})
                    queued_count = sum(1 for p in platforms_dict.values() if p and p.get('status') == 'queued')
                    duplicate_count = sum(1 for p in platforms_dict.values() if p and p.get('status') == 'duplicate')
                    error_count = sum(1 for p in platforms_dict.values() if p and p.get('status') == 'error')
                    
                    if progress_callback:
                        msg = f"✓ Completed {game_id}"
                        if queued_count > 0:
                            msg += f" ({queued_count} queued"
                            if duplicate_count > 0:
                                msg += f", {duplicate_count} duplicate"
                            if error_count > 0:
                                msg += f", {error_count} error"
                            msg += ")"
                        progress_callback(progress, msg)
                else:
                    error_msg = prediction.get("error", "Unknown error") if isinstance(prediction, dict) else f"Invalid prediction type: {type(prediction)}"
                    results["errors"].append({
                        "game_id": game_id,
                        "error": error_msg,
                    })
                    logger.error(f"Prediction failed for {game_id}: {error_msg}")
                    logger.error(f"Prediction details: {prediction}")
                    if progress_callback:
                        progress_callback(progress, f"✗ Failed {game_id}: {error_msg}")
            
            except Exception as e:
                results["errors"].append({
                    "game_id": game_id,
                    "error": str(e),
                })
                logger.error(f"Error processing {game_id}: {e}")
                if progress_callback:
                    progress_callback(progress, f"✗ Error {game_id}: {str(e)}")
        
        return results
    
    def process_post_queue(self, batch_size: int = 10) -> Dict[str, Any]:
        """
        Process pending posts from queue.
        
        Args:
            batch_size: Max posts to process
            
        Returns:
            Processing results
        """
        return self.social_manager.process_queue(max_posts=batch_size)
    
    def run_schedule(
        self,
        poll_interval_minutes: int = 15,
        prediction_schedule: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Run scheduled automation loop.
        
        Args:
            poll_interval_minutes: Minutes between queue processing cycles
            prediction_schedule: List of prediction schedules
                [{"time": "14:00", "trigger_type": "pregame", "game_ids": [...]}, ...]
        """
        self.running = True
        logger.info("Starting automation scheduler...")
        
        # Schedule queue processing
        schedule.every(poll_interval_minutes).minutes.do(self._process_queue_cycle)
        
        # Schedule predictions (if provided)
        if prediction_schedule:
            for schedule_item in prediction_schedule:
                schedule_time = schedule_item["time"]  # HH:MM format
                trigger_type = schedule_item["trigger_type"]
                game_ids = schedule_item["game_ids"]
                mode = schedule_item.get("mode", "auto")
                
                schedule.every().day.at(schedule_time).do(
                    self.run_predictions,
                    game_ids=game_ids,
                    trigger_type=trigger_type,
                    mode=mode,
                    progress_callback=None,  # No progress callback in scheduler
                )
                
                logger.info(f"Scheduled prediction at {schedule_time}: {trigger_type}")
        
        # Main loop
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
        
        logger.info("Automation scheduler stopped")
    
    def _process_queue_cycle(self):
        """Process queue (called by scheduler)."""
        try:
            # Cleanup old posts
            self.social_manager.cleanup_old_posts(older_than_hours=48)
            
            # Process pending posts
            results = self.process_post_queue(batch_size=10)
            
            logger.info(
                f"Queue cycle: Processed={results['processed']}, "
                f"Success={results['successful']}, Failed={results['failed']}"
            )
        
        except Exception as e:
            logger.error(f"Error processing queue cycle: {e}")
    
    def _is_prediction_processed(
        self,
        game_id: str,
        trigger_type: str,
    ) -> bool:
        """Check if prediction was already processed."""
        if game_id not in self.processed_predictions:
            return False
        return trigger_type in self.processed_predictions[game_id]
    
    def _mark_prediction_processed(
        self,
        game_id: str,
        trigger_type: str,
    ):
        """Mark prediction as processed."""
        if game_id not in self.processed_predictions:
            self.processed_predictions[game_id] = set()
        self.processed_predictions[game_id].add(trigger_type)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestration statistics."""
        queue_stats = self.social_manager.get_queue_stats()
        
        return {
            "processed_predictions": sum(
                len(triggers) for triggers in self.processed_predictions.values()
            ),
            "queue_stats": queue_stats,
            "enabled_platforms": list(self.social_manager.enabled_platforms),
        }


def run_automation(
    dry_run: bool = False,
    platforms: Optional[List[str]] = None,
    poll_interval_minutes: int = 15,
) -> None:
    """
    Run automation orchestrator.
    
    Args:
        dry_run: If True, don't actually post
        platforms: List of platforms to post
        poll_interval_minutes: Minutes between queue processing cycles
    """
    orchestrator = AutomationOrchestrator(
        dry_run=dry_run,
        platforms=platforms,
    )
    
    # Run scheduler
    orchestrator.run_schedule(poll_interval_minutes=poll_interval_minutes)


def run_one_off_predictions(
    game_ids: List[str],
    trigger_type: str = "pregame",
    mode: str = "auto",
    dry_run: bool = False,
    platforms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run one-off predictions.
    
    Args:
        game_ids: List of game IDs to predict
        trigger_type: Trigger type
        mode: Prediction mode
        dry_run: If True, don't actually post
        platforms: List of platforms to post
        
    Returns:
        Prediction results
    """
    orchestrator = AutomationOrchestrator(
        dry_run=dry_run,
        platforms=platforms,
    )
    
    return orchestrator.run_predictions(
        game_ids=game_ids,
        trigger_type=trigger_type,
        mode=mode,
    )
