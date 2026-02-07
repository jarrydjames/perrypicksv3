"""Bluesky API Client for PerryPicks v3 Automation.

Supports posting predictions with Bluesky Social API.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import time

try:
    import atproto
    from atproto import Client as BskyClient
    ATPROTO_AVAILABLE = True
except ImportError:
    ATPROTO_AVAILABLE = False

from core.env import load_environment

logger = logging.getLogger(__name__)

class BlueskyClient:
    """Bluesky Social API client for posting predictions."""
    
    def __init__(
        self,
        handle: Optional[str] = None,
        app_password: Optional[str] = None,
        dry_run: bool = False,
    ):
        """
        Initialize Bluesky client.
        
        Args:
            handle: Bluesky handle (e.g., 'perrypicks.bsky.social')
            app_password: Bluesky app password (from Settings → App Passwords)
            dry_run: If True, don't actually post (log only)
        """
        self.dry_run = dry_run
        self.client = None
        self.handle = None
        
        # Load from environment if not provided
        if not all([handle, app_password]):
            env = load_environment()
            handle = handle or os.getenv("BLUESKY_HANDLE")
            app_password = app_password or os.getenv("BLUESKY_APP_PASSWORD")
        
        if not all([handle, app_password]):
            logger.warning("Bluesky credentials not provided. Bluesky posting disabled.")
            self.enabled = False
            return
        
        self.enabled = True
        
        # Initialize atproto client
        if ATPROTO_AVAILABLE:
            try:
                self.client = BskyClient()
                self.handle = handle
                
                # Login with handle + app password
                profile = self.client.login(handle, app_password)
                logger.info(f"Bluesky client initialized: @{handle}")
            except Exception as e:
                logger.error(f"Failed to initialize Bluesky client: {e}")
                self.enabled = False
        else:
            logger.warning("atproto not installed. Install with: pip install atproto")
            self.enabled = False
    
    def post(
        self,
        text: str,
        reply_to_id: Optional[str] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Post a skeet (Bluesky post).
        
        Args:
            text: Post text (no hard limit, but 300 chars recommended)
            reply_to_id: Post URI or AT URI to reply to
            **kwargs: Additional API parameters
            
        Returns:
            Post data if successful, None otherwise
        """
        if not self.enabled:
            logger.warning("Bluesky client not enabled, skipping post")
            return None
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Bluesky post: {text[:100]}...")
            return {"dry_run": True, "text": text}
        
        try:
            # Build post object
            post_kwargs = {"text": text}
            
            # Add reply if specified
            if reply_to_id:
                post_kwargs["reply"] = {
                    "root": reply_to_id,
                    "parent": reply_to_id,
                }
            
            response = self.client.send_post(**post_kwargs)
            post_uri = response.uri
            post_cid = response.cid
            
            logger.info(f"Bluesky post successful: URI={post_uri}")
            
            return {
                "id": post_uri,
                "cid": post_cid,
                "text": text,
                "platform": "bluesky",
            }
        
        except Exception as e:
            logger.error(f"Bluesky API error: {e}")
            return None
    
    def post_thread(
        self,
        posts: List[str],
        **kwargs
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Post a thread of skeets.
        
        Args:
            posts: List of post texts
            **kwargs: Additional API parameters
            
        Returns:
            List of post data if successful
        """
        if not self.enabled:
            logger.warning("Bluesky client not enabled, skipping thread")
            return []
        
        results = []
        last_post_uri = None
        
        for i, text in enumerate(posts):
            if last_post_uri:
                result = self.post(text, reply_to_id=last_post_uri, **kwargs)
            else:
                result = self.post(text, **kwargs)
            
            if result:
                results.append(result)
                last_post_uri = result["id"]
            else:
                logger.error(f"Failed to post skeet {i+1}/{len(posts)}")
                break
            
            # Small delay between thread posts
            time.sleep(1)
        
        return results
    
    def delete(self, post_uri: str) -> bool:
        """
        Delete a post.
        
        Args:
            post_uri: URI of post to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            logger.warning("Bluesky client not enabled, skipping delete")
            return False
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Bluesky delete: URI={post_uri}")
            return True
        
        try:
            # Delete by URI (atproto SDK handles this)
            self.client.delete_post(post_uri)
            logger.info(f"Bluesky post deleted: URI={post_uri}")
            return True
        except Exception as e:
            logger.error(f"Bluesky API error deleting post: {e}")
            return False
    
    def is_enabled(self) -> bool:
        """Check if Bluesky client is enabled."""
        return self.enabled
