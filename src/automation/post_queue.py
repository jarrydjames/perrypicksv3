"""Post Queue for PerryPicks v3 Automation.

Manages post queue with duplicate detection and persistence.
"""

from __future__ import annotations
import logging
import json
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import time

import pendulum

logger = logging.getLogger(__name__)

class PostStatus(Enum):
    """Post status enum."""
    PENDING = "pending"
    POSTING = "posting"
    POSTED = "posted"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class PostItem:
    """Queued post item."""
    post_id: str
    game_id: str
    platform: str
    content: str
    trigger_type: str
    created_at_utc: str
    status: PostStatus = PostStatus.PENDING
    posted_at_utc: Optional[str] = None
    message_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PostItem":
        """Create from dictionary."""
        data["status"] = PostStatus(data["status"])
        return cls(**data)

class PostQueue:
    """Post queue with duplicate detection and persistence."""
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        dedupe_window_hours: int = 24,
    ):
        """
        Initialize post queue.
        
        Args:
            storage_path: Path to queue storage file
            dedupe_window_hours: Hours to consider for deduplication
        """
        self.storage_path = storage_path or Path("data/post_queue.json")
        self.dedupe_window_hours = dedupe_window_hours
        self.queue: Dict[str, PostItem] = {}
        self.posted_history: Dict[str, Dict[str, str]] = {}  # game_id -> {trigger_type -> post_id}
        self._load_queue()
    
    def _generate_post_id(
        self,
        game_id: str,
        trigger_type: str,
        platform: str,
        content: str,
    ) -> str:
        """Generate unique post ID."""
        # Hash of game_id + trigger_type + platform + content
        content_hash = hashlib.md5(
            f"{game_id}{trigger_type}{platform}{content}".encode()
        ).hexdigest()[:8]
        
        # Add timestamp for uniqueness
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"{game_id}_{trigger_type}_{timestamp}_{content_hash}"
    
    def _is_duplicate(
        self,
        game_id: str,
        trigger_type: str,
        platform: str,
    ) -> bool:
        """
        Check if post is a duplicate.
        
        Args:
            game_id: Game ID
            trigger_type: Trigger type (pregame, halftime, q3)
            platform: Platform (twitter, bluesky, discord)
            
        Returns:
            True if duplicate, False otherwise
        """
        # Get current date
        current_date = pendulum.now().strftime("%Y%m%d")
        
        # Check posted history
        if game_id in self.posted_history:
            if trigger_type in self.posted_history[game_id]:
                # Check if within dedupe window
                post_time = pendulum.parse(self.posted_history[game_id][trigger_type]["posted_at"])
                age_hours = (pendulum.now() - post_time).total_hours()
                
                if age_hours < self.dedupe_window_hours:
                    # Check if post was from a different date
                    # If so, allow reposting (for pregame predictions on new days)
                    post_date = post_time.strftime("%Y%m%d")
                    if post_date != current_date:
                        logger.info(
                            f"Allowing repost (different date): {game_id} {trigger_type} {platform} "
                            f"(posted on {post_date}, now {current_date})"
                        )
                        return False
                    
                    # Same date, within dedupe window
                    # For pregame, allow multiple posts per day (user may want to regenerate)
                    # For live predictions (halftime/Q3), block duplicates to prevent spam
                    if trigger_type == "pregame":
                        logger.info(
                            f"Allowing pregame repost (same day): {game_id} {trigger_type} {platform} "
                            f"({age_hours:.1f}h ago)"
                        )
                        return False
                    
                    # Live in-game predictions - block as duplicate
                    logger.warning(
                        f"Duplicate post detected: {game_id} {trigger_type} {platform} "
                        f"({age_hours:.1f}h ago)"
                    )
                    return True
        
        return False
    
    def enqueue(
        self,
        game_id: str,
        platform: str,
        content: str,
        trigger_type: str = "pregame",
        max_retries: int = 3,
        allow_duplicates: bool = False,
    ) -> Optional[str]:
        """
        Enqueue a post.
        
        Args:
            game_id: Game ID
            platform: Platform (twitter, bluesky, discord)
            content: Post content
            trigger_type: Trigger type
            max_retries: Maximum retry attempts
            allow_duplicates: If True, bypass duplicate detection
            
        Returns:
            Post ID if enqueued, None if duplicate (unless allow_duplicates=True)
        """
        # Check duplicate (unless override enabled)
        if not allow_duplicates and self._is_duplicate(game_id, trigger_type, platform):
            logger.info(f"Skipping duplicate post: {game_id} {trigger_type} {platform}")
            return None
        
        if allow_duplicates:
            logger.info(f"Allowing duplicate post (override enabled): {game_id} {trigger_type} {platform}")
        
        # Generate post ID
        post_id = self._generate_post_id(game_id, trigger_type, platform, content)
        
        # Create post item
        post = PostItem(
            post_id=post_id,
            game_id=game_id,
            platform=platform,
            content=content,
            trigger_type=trigger_type,
            created_at_utc=pendulum.now().to_iso8601_string(),
            max_retries=max_retries,
        )
        
        # Add to queue
        self.queue[post_id] = post
        self._save_queue()
        
        logger.info(f"Post enqueued: {post_id} ({platform})")
        return post_id
    
    def get_all_posts(self) -> List[PostItem]:
        """
        Get all posts from queue.
        
        Returns:
            List of all post items
        """
        return list(self.queue.values())
    
    def clear_queue(self) -> int:
        """
        Clear all posts from queue.
        
        Returns:
            Number of posts cleared
        """
        count = len(self.queue)
        self.queue = {}
        self._save_queue()
        logger.info(f"Cleared {count} posts from queue")
        return count
    
    def get_pending_posts(self, platform: Optional[str] = None) -> List[PostItem]:
        """
        Get all pending posts.
        
        Args:
            platform: Filter by platform (optional)
            
        Returns:
            List of pending post items
        """
        pending = []
        
        for post in self.queue.values():
            if post.status == PostStatus.PENDING:
                if platform is None or post.platform == platform:
                    pending.append(post)
        
        # Sort by created_at
        pending.sort(key=lambda p: p.created_at_utc)
        return pending
    
    def mark_posting(self, post_id: str) -> bool:
        """Mark post as posting."""
        if post_id not in self.queue:
            return False
        
        self.queue[post_id].status = PostStatus.POSTING
        self._save_queue()
        return True
    
    def mark_posted(
        self,
        post_id: str,
        message_id: str,
    ) -> bool:
        """
        Mark post as posted.
        
        Args:
            post_id: Internal post ID
            message_id: Platform-specific message ID
            
        Returns:
            True if successful, False otherwise
        """
        if post_id not in self.queue:
            return False
        
        post = self.queue[post_id]
        post.status = PostStatus.POSTED
        post.posted_at_utc = pendulum.now().to_iso8601_string()
        post.message_id = message_id
        
        # Add to posted history
        game_id = post.game_id
        trigger_type = post.trigger_type
        platform = post.platform
        
        if game_id not in self.posted_history:
            self.posted_history[game_id] = {}
        
        self.posted_history[game_id][trigger_type] = {
            "platform": platform,
            "message_id": message_id,
            "posted_at": post.posted_at_utc,
        }
        
        self._save_queue()
        logger.info(f"Post marked as posted: {post_id}")
        return True
    
    def mark_failed(
        self,
        post_id: str,
        error: str,
    ) -> bool:
        """
        Mark post as failed.
        
        Args:
            post_id: Internal post ID
            error: Error message
            
        Returns:
            True if successful, False otherwise
        """
        if post_id not in self.queue:
            return False
        
        post = self.queue[post_id]
        post.status = PostStatus.FAILED
        post.error = error
        post.retry_count += 1
        
        # Check if should retry
        if post.retry_count < post.max_retries:
            post.status = PostStatus.RETRYING
            logger.warning(
                f"Post failed, will retry ({post.retry_count}/{post.max_retries}): {post_id}"
            )
        
        self._save_queue()
        return True
    
    def cleanup_old_posts(self, older_than_hours: int = 48) -> int:
        """
        Cleanup old posts from queue.
        
        Args:
            older_than_hours: Remove posts older than this
            
        Returns:
            Number of posts removed
        """
        cutoff = pendulum.now() - timedelta(hours=older_than_hours)
        removed = 0
        
        to_remove = []
        for post_id, post in self.queue.items():
            post_time = pendulum.parse(post.created_at_utc)
            
            # Remove if old and (posted or failed)
            if post_time < cutoff:
                if post.status in [PostStatus.POSTED, PostStatus.FAILED]:
                    to_remove.append(post_id)
        
        for post_id in to_remove:
            del self.queue[post_id]
            removed += 1
        
        if removed > 0:
            self._save_queue()
            logger.info(f"Cleaned up {removed} old posts")
        
        return removed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        stats = {
            "total": len(self.queue),
            "pending": 0,
            "posting": 0,
            "posted": 0,
            "failed": 0,
            "retrying": 0,
        }
        
        for post in self.queue.values():
            stats[post.status.value] += 1
        
        return stats
    
    def _save_queue(self):
        """Save queue to disk."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save queue
            queue_data = {post_id: post.to_dict() for post_id, post in self.queue.items()}
            
            with open(self.storage_path, "w") as f:
                json.dump({
                    "queue": queue_data,
                    "posted_history": self.posted_history,
                }, f, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to save queue: {e}")
    
    def _load_queue(self):
        """Load queue from disk."""
        try:
            if not self.storage_path.exists():
                return
            
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            
            # Load queue
            queue_data = data.get("queue", {})
            self.queue = {
                post_id: PostItem.from_dict(post_data)
                for post_id, post_data in queue_data.items()
            }
            
            # Load posted history
            self.posted_history = data.get("posted_history", {})
            
            logger.info(f"Loaded {len(self.queue)} posts from queue")
        
        except Exception as e:
            logger.error(f"Failed to load queue: {e}")
            self.queue = {}
            self.posted_history = {}
