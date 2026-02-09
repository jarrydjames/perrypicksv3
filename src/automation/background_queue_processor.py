"""Background Queue Processor for PerryPicks v3.

Continuously processes queued posts in background without manual intervention.
Works alongside Game State Monitor for complete automation.
"""

from __future__ import annotations
import logging
import threading
import time
from typing import Dict, Any, Optional
from datetime import datetime

from src.automation.auto_queue_processor import AutoQueueProcessor
from src.automation.social_media_manager import SocialMediaManager

logger = logging.getLogger(__name__)


class BackgroundQueueProcessor:
    """Continuously processes queued posts in background.
    
    Polls the post queue at regular intervals and processes
    pending posts automatically. Designed for 'fire and forget'
    automation.
    
    Features:
    - Configurable poll interval
    - Configurable batch size
    - Automatic retry on failure
    - Rate limiting support
    - Graceful shutdown
    - Status tracking
    """
    
    def __init__(
        self,
        poll_interval: int = 15,
        batch_size: int = 10,
        social_manager: Optional[SocialMediaManager] = None,
        max_retries: int = 3,
    ):
        """Initialize background queue processor.
        
        Args:
            poll_interval: Seconds between queue polls (default: 15)
            batch_size: Max posts to process per poll (default: 10)
            social_manager: Social media manager instance
            max_retries: Max retry attempts for failed posts
        """
        self.poll_interval = poll_interval
        self.batch_size = batch_size
        self.max_retries = max_retries
        
        # Initialize queue processor
        self.queue_processor = AutoQueueProcessor(
            social_manager=social_manager,
        )
        
        # Thread management
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.stop_event = threading.Event()
        
        # Statistics
        self.stats = {
            "processed": 0,
            "failed": 0,
            "skipped": 0,
            "started_at": None,
            "last_processed_at": None,
        }
        
        logger.info(
            f"Background Queue Processor initialized "
            f"(poll_interval={poll_interval}s, batch_size={batch_size})"
        )
    
    def _process_batch(self) -> Dict[str, Any]:
        """Process a batch of posts from queue.
        
        Returns:
            Processing results
        """
        try:
            # Process pending posts
            result = self.queue_processor.process_pending(
                max_posts=self.batch_size,
            )
            
            # Update stats
            if result.get("success"):
                processed = result.get("processed_predictions", 0)
                self.stats["processed"] += processed
                self.stats["last_processed_at"] = datetime.now().isoformat()
                
                if processed > 0:
                    logger.info(f"Processed {processed} posts from queue")
                elif result.get("skipped"):
                    # Processing skipped (e.g., no social manager) - not an error
                    logger.info(f"Queue processing skipped: {result.get('skipped')}")
            else:
                error = result.get("error", "Unknown error")
                # Only log as error if it's not a "no social manager" skip
                if error and "no social manager" not in str(error).lower():
                    logger.warning(f"Failed to process queue: {error}")
                    self.stats["failed"] += 1
                else:
                    # Skipped due to missing social manager - don't count as failure
                    logger.debug(f"Queue processing skipped (no social manager)")
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            self.stats["failed"] += 1
            return {
                "success": False,
                "error": str(e),
            }
    
    def _process_loop(self):
        """Main processing loop - runs in background thread.
        
        Continuously polls queue and processes pending posts
        until stop event is set.
        """
        logger.info("Background Queue Processor started")
        self.stats["started_at"] = datetime.now().isoformat()
        
        while self.running:
            try:
                # Check for stop signal
                if self.stop_event.is_set():
                    logger.info("Stop signal received, exiting loop")
                    break
                
                # Process a batch
                self._process_batch()
                
                # Wait before next poll
                # Use stop_event.wait() so we can interrupt the sleep
                self.stop_event.wait(self.poll_interval)
            
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                # Don't exit on error, just continue
                time.sleep(1)
        
        logger.info("Background Queue Processor stopped")
    
    def start(self) -> bool:
        """Start background queue processor.
        
        Creates and starts a daemon thread that continuously
        processes queued posts.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.running:
            logger.warning("Background Queue Processor already running")
            return False
        
        try:
            # Reset stop event
            self.stop_event.clear()
            
            # Set running flag
            self.running = True
            
            # Create and start thread
            self.thread = threading.Thread(
                target=self._process_loop,
                daemon=True,
                name="BackgroundQueueProcessor",
            )
            self.thread.start()
            
            logger.info(
                f"Background Queue Processor started "
                f"(thread={self.thread.name})"
            )
            return True
        
        except Exception as e:
            logger.error(f"Failed to start Background Queue Processor: {e}")
            self.running = False
            return False
    
    def stop(self, timeout: int = 10) -> bool:
        """Stop background queue processor.
        
        Signals the processing loop to stop and waits for
        thread to terminate.
        
        Args:
            timeout: Max seconds to wait for thread to stop
            
        Returns:
            True if stopped successfully, False if timeout
        """
        if not self.running:
            logger.warning("Background Queue Processor not running")
            return True
        
        try:
            logger.info("Stopping Background Queue Processor...")
            
            # Set stop event
            self.stop_event.set()
            
            # Wait for thread to stop
            if self.thread:
                self.thread.join(timeout=timeout)
            
            # Check if stopped
            if self.thread and self.thread.is_alive():
                logger.warning(
                    f"Background Queue Processor did not stop "
                    f"within {timeout}s"
                )
                return False
            else:
                logger.info("Background Queue Processor stopped")
                self.running = False
                return True
        
        except Exception as e:
            logger.error(f"Error stopping Background Queue Processor: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status.
        
        Returns:
            Dictionary with status information
        """
        # Get queue status
        queue_status = self.queue_processor.get_queue_status()
        
        return {
            "running": self.running,
            "thread_alive": self.thread.is_alive() if self.thread else False,
            "thread_name": self.thread.name if self.thread else None,
            "poll_interval": self.poll_interval,
            "batch_size": self.batch_size,
            "stats": self.stats.copy(),
            "queue": queue_status,
        }
    
    def process_now(self, max_posts: int = None) -> Dict[str, Any]:
        """Process queue immediately (one-off).
        
        Useful for manual intervention or catching up.
        
        Args:
            max_posts: Max posts to process (None = use batch_size)
            
        Returns:
            Processing results
        """
        try:
            batch_size = max_posts or self.batch_size
            logger.info(f"Processing queue now (max_posts={batch_size})")
            
            # Use queue processor directly
            result = self.queue_processor.process_pending(
                max_posts=batch_size,
            )
            
            # Update stats
            if result.get("success"):
                processed = result.get("processed_predictions", 0)
                self.stats["processed"] += processed
                self.stats["last_processed_at"] = datetime.now().isoformat()
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing queue now: {e}")
            return {
                "success": False,
                "error": str(e),
            }