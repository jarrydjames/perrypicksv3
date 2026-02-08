"""Social Media Manager for PerryPicks v3 Automation.

Orchestrates posting across Twitter/X, Bluesky, and Discord.
"""

from __future__ import annotations
import logging
import time
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
import os

from src.automation.twitter_client import TwitterClient
from src.automation.bluesky_client import BlueskyClient
from src.automation.post_generator import PostGenerator
from src.automation.post_queue import PostQueue, PostStatus
from core.discord_client import DiscordWebhookClient
from core.env import load_environment

logger = logging.getLogger(__name__)

class SocialMediaManager:
    """Manage posting across all social platforms."""
    
    def __init__(
        self,
        twitter_client: Optional[TwitterClient] = None,
        bluesky_client: Optional[BlueskyClient] = None,
        discord_client: Optional[DiscordWebhookClient] = None,
        post_queue: Optional[PostQueue] = None,
        dry_run: bool = False,
    ):
        """
        Initialize social media manager.
        
        Args:
            twitter_client: Twitter API client
            bluesky_client: Bluesky API client
            discord_client: Discord webhook client
            post_queue: Post queue manager
            dry_run: If True, don't actually post
        """
        self.dry_run = dry_run
        self.queue = post_queue or PostQueue()
        
        # Initialize post generator
        self.post_generator = PostGenerator(
            include_odds=True,
            include_confidence=True,
            use_emojis=True,
            hashtags=["#NBAPredictions", "#PerryPicks"],
        )
        
        # Initialize clients (or use provided)
        env = load_environment()
        
        self.twitter = twitter_client or TwitterClient(dry_run=dry_run)
        self.bluesky = bluesky_client or BlueskyClient(dry_run=dry_run)
        
        # Discord client
        discord_webhook = os.getenv("DISCORD_WEBHOOK_URL")
        if discord_client:
            self.discord = discord_client
        elif discord_webhook:
            self.discord = DiscordWebhookClient(discord_webhook)
        else:
            self.discord = None
            logger.warning("Discord webhook not provided")
        
        # Track enabled platforms
        self.enabled_platforms = self._get_enabled_platforms()
        
        logger.info(
            f"Social Media Manager initialized. "
            f"Enabled: {self.enabled_platforms}. "
            f"Dry run: {dry_run}"
        )
    
    def _get_enabled_platforms(self) -> Set[str]:
        """Get set of enabled platforms."""
        enabled = set()
        
        if self.twitter.is_enabled():
            enabled.add("twitter")
        
        if self.bluesky.is_enabled():
            enabled.add("bluesky")
        
        if self.discord is not None:
            enabled.add("discord")
        
        return enabled
    
    def post_prediction(
        self,
        prediction: Dict[str, Any],
        trigger_type: str = "pregame",
        platforms: Optional[List[str]] = None,
        allow_duplicates: bool = False,
    ) -> Dict[str, Any]:
        """
        Post prediction to all enabled platforms.
        
        Args:
            prediction: Prediction dictionary from predict_game()
            trigger_type: Trigger type (pregame, halftime, q3)
            platforms: List of platforms to post (None = all enabled)
            allow_duplicates: If True, bypass duplicate detection
            
        Returns:
            Results dictionary with platform-specific results
        """
        game_id = prediction.get("game_id", "unknown")
        results = {
            "game_id": game_id,
            "trigger_type": trigger_type,
            "platforms": {},
            "success": True,
        }
        
        # Determine which platforms to post to
        target_platforms = platforms or list(self.enabled_platforms)
        target_platforms = [p for p in target_platforms if p in self.enabled_platforms]
        
        if not target_platforms:
            logger.warning(f"No enabled platforms to post to for game {game_id}")
            results["success"] = False
            return results
        
        # Generate posts for each platform
        for platform in target_platforms:
            try:
                # Generate platform-specific content
                if trigger_type == "pregame":
                    content = self.post_generator.generate_pregame_post(
                        prediction,
                        platform=platform,
                    )
                elif trigger_type == "halftime":
                    content = self.post_generator.generate_halftime_post(
                        prediction,
                        platform=platform,
                    )
                elif trigger_type == "q3":
                    content = self.post_generator.generate_q3_post(
                        prediction,
                        platform=platform,
                    )
                else:
                    logger.warning(f"Unknown trigger type: {trigger_type}")
                    continue
                
                # Enqueue post
                post_id = self.queue.enqueue(
                    game_id=game_id,
                    platform=platform,
                    content=content,
                    trigger_type=trigger_type,
                    max_retries=3,
                    allow_duplicates=allow_duplicates,
                )
                
                if post_id:
                    results["platforms"][platform] = {
                        "post_id": post_id,
                        "status": "queued",
                        "content": content,
                    }
                else:
                    # Duplicate post
                    results["platforms"][platform] = {
                        "status": "duplicate",
                        "reason": "Duplicate post detected",
                    }
                
            except Exception as e:
                logger.error(f"Error generating post for {platform}: {e}")
                results["platforms"][platform] = {
                    "status": "error",
                    "error": str(e),
                }
        
        return results
    
    def process_queue(self, max_posts: int = 10) -> Dict[str, Any]:
        """
        Process pending posts from queue.
        
        Args:
            max_posts: Maximum posts to process in one batch
            
        Returns:
            Processing results
        """
        # Get pending posts
        pending = self.queue.get_pending_posts()
        
        # Limit batch size
        pending = pending[:max_posts]
        
        if not pending:
            return {
                "processed": 0,
                "successful": 0,
                "failed": 0,
                "message": "No pending posts",
            }
        
        results = {
            "processed": len(pending),
            "successful": 0,
            "failed": 0,
            "posts": [],
        }
        
        # Process each post
        for post in pending:
            post_id = post.post_id
            platform = post.platform
            content = post.content
            
            # Mark as posting
            self.queue.mark_posting(post_id)
            
            # Post to platform
            platform_result = self._post_to_platform(platform, content)
            
            if platform_result:
                # Check if it's an error result
                if "error" in platform_result:
                    # Posting failed with specific error
                    error_msg = platform_result["error"]
                    logger.error(f"Posting to {platform} failed: {error_msg}")
                    self.queue.mark_failed(post_id, error_msg)
                    results["failed"] += 1
                    results["posts"].append({
                        "post_id": post_id,
                        "platform": platform,
                        "status": "failed",
                        "error": error_msg,
                    })
                else:
                    # Success
                    self.queue.mark_posted(post_id, platform_result["id"])
                    results["successful"] += 1
                    results["posts"].append({
                        "post_id": post_id,
                        "platform": platform,
                        "status": "posted",
                        "message_id": platform_result["id"],
                    })
            else:
                # Failure (None returned)
                error_msg = "Unknown error - platform returned None"
                logger.error(f"Posting to {platform} failed: {error_msg}")
                self.queue.mark_failed(post_id, error_msg)
                results["failed"] += 1
                results["posts"].append({
                    "post_id": post_id,
                    "platform": platform,
                    "status": "failed",
                    "error": error_msg,
                })
            
            # Small delay between posts
            time.sleep(2)
        
        return results
    
    def _post_to_platform(
        self,
        platform: str,
        content: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Post to specific platform.
        
        Args:
            platform: Platform (twitter, bluesky, discord)
            content: Post content
            
        Returns:
            Platform result dict or None
        """
        try:
            if platform == "twitter":
                return self.twitter.post(content)
            
            elif platform == "bluesky":
                return self.bluesky.post(content)
            
            elif platform == "discord":
                if self.discord:
                    try:
                        self.discord.post_message(
                            content=content,
                            username="PerryPicks"
                        )
                        return {"id": "discord_post", "platform": "discord"}
                    except Exception as e:
                        logger.error(f"Error posting to Discord: {e}")
                        return {"error": str(e)}
                else:
                    error_msg = "Discord webhook URL not configured. Set DISCORD_WEBHOOK_URL environment variable."
                    logger.error(error_msg)
                    return {"error": error_msg}
            
            else:
                logger.warning(f"Unknown platform: {platform}")
                return None
        
        except Exception as e:
            logger.error(f"Error posting to {platform}: {e}")
            return None
    
    def cleanup_old_posts(self, older_than_hours: int = 48) -> int:
        """
        Cleanup old posts from queue.
        
        Args:
            older_than_hours: Remove posts older than this
            
        Returns:
            Number of posts removed
        """
        return self.queue.cleanup_old_posts(older_than_hours)
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return self.queue.get_stats()
    
    def is_platform_enabled(self, platform: str) -> bool:
        """Check if platform is enabled."""
        return platform in self.enabled_platforms
