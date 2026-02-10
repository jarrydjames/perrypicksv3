"""Auto Queue Processor for PerryPicks v3.

Automatically processes queued posts to platforms without manual intervention.
Handles both immediate posting and scheduled posting.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional


from src.automation.post_queue import PostQueue, PostStatus
from src.automation.social_media_manager import SocialMediaManager

logger = logging.getLogger(__name__)


class AutoQueueProcessor:
    """Automatically process queued posts.
    
    Processes pending posts and sends them to platforms.
    Designed for automatic operation without manual intervention.
    """
    
    def __init__(
        self,
        social_manager: Optional[SocialMediaManager] = None,
    ):
        """Initialize auto queue processor.
        
        Args:
            social_manager: Social media manager instance
        """
        self.social_manager = social_manager
        self.queue = PostQueue() if not social_manager else social_manager.queue
        
        logger.info("Auto Queue Processor initialized")
    
    def queue_and_post(
        self,
        prediction: Dict[str, Any],
        trigger_type: str,
        platforms: Optional[List[str]] = None,
        max_retries: int = 3,
        allow_duplicates: bool = False,
    ) -> Dict[str, Any]:
        """Queue prediction and immediately post it.
        
        Args:
            prediction: Prediction dictionary
            trigger_type: Type of trigger (halftime, q3, etc.)
            platforms: Platforms to post to
            max_retries: Max retries for failed posts
            allow_duplicates: If True, bypass duplicate detection
            
        Returns:
            Result dictionary with status
        """
        result = {
            "success": False,
            "queued": [],
            "posted": [],
            "errors": [],
        }
        
        game_id = prediction.get("game_id", "unknown")
        
        try:
            # Determine target platforms
            if platforms is None and self.social_manager:
                platforms = list(self.social_manager.enabled_platforms)
            
            if not platforms:
                logger.warning(f"No platforms available for {game_id}")
                result["error"] = "No platforms available"
                return result
            
            if not self.social_manager:
                logger.warning("No social manager available - cannot generate content or post")
                result["error"] = "No social manager available"
                return result

            normalized_trigger_type = "q3" if trigger_type in ("q3", "q3_5min") else trigger_type

            post_results = self.social_manager.post_prediction(
                prediction=prediction,
                trigger_type=normalized_trigger_type,
                platforms=platforms,
                allow_duplicates=allow_duplicates,
            )

            platform_results = post_results.get("platforms", {})
            for platform, details in platform_results.items():
                status = (details or {}).get("status")
                if status == "queued":
                    result["queued"].append({
                        "platform": platform,
                        "post_id": details.get("post_id"),
                        "status": "queued",
                    })
                elif status == "duplicate":
                    result["queued"].append({
                        "platform": platform,
                        "status": "duplicate",
                    })
                else:
                    result["errors"].append({
                        "platform": platform,
                        "error": (details or {}).get("error", "Unknown post generation error"),
                    })

            queued_count = len([q for q in result["queued"] if q.get("status") == "queued"])
            if queued_count == 0 and result["errors"]:
                result["success"] = False
                return result

            process_result = self.social_manager.process_queue(
                max_posts=max(queued_count, 1),
            )

            result["posted"] = process_result.get("posts", [])
            successful = int(process_result.get("successful", 0))
            failed = int(process_result.get("failed", 0))
            result["success"] = successful > 0 and failed == 0

            if failed > 0:
                result["errors"].append({
                    "platform": "queue",
                    "error": f"{failed} queued post(s) failed during processing",
                })

            return result
        
        except Exception as e:
            logger.error(f"Error in queue_and_post for {game_id}: {e}")
            result["error"] = str(e)
            return result
    
    def process_pending(self, max_posts: int = 10) -> Dict[str, Any]:
        """Process pending posts from queue.
        
        Args:
            max_posts: Maximum posts to process
            
        Returns:
            Processing results
        """
        try:
            if self.social_manager:
                return self.social_manager.process_queue(max_posts=max_posts)
            else:
                # No social manager - skip processing but don't fail
                # This allows queue processor to continue and retry when manager becomes available
                logger.warning("No social manager available - skipping queue processing (will retry)")
                return {
                    "success": True,  # Return success to avoid continuous error logging
                    "error": None,
                    "processed_predictions": 0,
                    "skipped": "No social manager available",
                }
        
        except Exception as e:
            logger.error(f"Error processing pending posts: {e}")
            return {"success": False, "error": str(e)}
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status.
        
        Returns:
            Dictionary with queue statistics
        """
        try:
            all_posts = self.queue.get_all_posts()
            pending_posts = [p for p in all_posts if p.status in (PostStatus.PENDING, PostStatus.POSTING)]
            failed_posts = [p for p in all_posts if p.status == PostStatus.FAILED]
            posted_posts = [p for p in all_posts if p.status == PostStatus.POSTED]
            
            return {
                "total": len(all_posts),
                "pending": len(pending_posts),
                "failed": len(failed_posts),
                "posted": len(posted_posts),
            }
        
        except Exception as e:
            logger.error(f"Error getting queue status: {e}")
            return {
                "total": 0,
                "pending": 0,
                "failed": 0,
                "posted": 0,
                "error": str(e),
            }