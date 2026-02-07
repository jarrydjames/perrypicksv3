"""Twitter/X API Client for PerryPicks v3 Automation.

Supports posting predictions with Twitter API v2.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import time

import json

try:
    import tweepy
    TWEETPY_AVAILABLE = True
except ImportError:
    TWEETPY_AVAILABLE = False

from core.env import load_environment

logger = logging.getLogger(__name__)

class TwitterClient:
    """Twitter/X API v2 client for posting predictions."""
    
    def __init__(
        self,
        consumer_key: Optional[str] = None,
        consumer_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        access_token_secret: Optional[str] = None,
        bearer_token: Optional[str] = None,
        dry_run: bool = False,
    ):
        """
        Initialize Twitter client.
        
        Args:
            consumer_key: Twitter API consumer key
            consumer_secret: Twitter API consumer secret
            access_token: Twitter access token
            access_token_secret: Twitter access token secret
            bearer_token: Twitter API bearer token (for OAuth 2.0)
            dry_run: If True, don't actually post (log only)
        """
        self.dry_run = dry_run
        self.client = None
        
        # Load from environment if not provided
        if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
            env = load_environment()
            consumer_key = consumer_key or os.getenv("TWITTER_CONSUMER_KEY")
            consumer_secret = consumer_secret or os.getenv("TWITTER_CONSUMER_SECRET")
            access_token = access_token or os.getenv("TWITTER_ACCESS_TOKEN")
            access_token_secret = access_token_secret or os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
            bearer_token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN")
        
        if not all([consumer_key, consumer_secret, access_token, access_token_secret]) and not bearer_token:
            logger.warning("Twitter credentials not provided. Twitter posting disabled.")
            self.enabled = False
            return
        
        self.enabled = True
        
        # Initialize tweepy client
        if TWEETPY_AVAILABLE:
            try:
                if bearer_token:
                    # OAuth 2.0 (Bearer Token) - for read/write with bot
                    self.client = tweepy.Client(
                        bearer_token=bearer_token,
                        wait_on_rate_limit=True,
                    )
                else:
                    # OAuth 1.0a (User Context)
                    self.client = tweepy.Client(
                        consumer_key=consumer_key,
                        consumer_secret=consumer_secret,
                        access_token=access_token,
                        access_token_secret=access_token_secret,
                        wait_on_rate_limit=True,
                    )
                logger.info("Twitter client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Twitter client: {e}")
                self.enabled = False
        else:
            logger.warning("tweepy not installed. Install with: pip install tweepy")
            self.enabled = False
    
    def post(
        self,
        text: str,
        media_ids: Optional[List[str]] = None,
        reply_to_id: Optional[str] = None,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Post a tweet.
        
        Args:
            text: Tweet text (max 280 characters)
            media_ids: List of media IDs to attach
            reply_to_id: Tweet ID to reply to
            **kwargs: Additional API parameters
            
        Returns:
            Tweet data if successful, None otherwise
        """
        if not self.enabled:
            logger.warning("Twitter client not enabled, skipping post")
            return None
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Twitter post: {text[:100]}...")
            return {"dry_run": True, "text": text}
        
        try:
            # Truncate if too long
            if len(text) > 280:
                text = text[:277] + "..."
                logger.warning(f"Tweet truncated to 280 characters")
            
            response = self.client.create_tweet(
                text=text,
                media_ids=media_ids,
                in_reply_to_tweet_id=reply_to_id,
                **kwargs
            )
            
            tweet_id = response.data["id"]
            logger.info(f"Tweet posted successfully: ID={tweet_id}")
            
            return {
                "id": str(tweet_id),
                "text": text,
                "platform": "twitter",
            }
        
        except tweepy.TweepyException as e:
            logger.error(f"Twitter API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error posting to Twitter: {e}")
            return None
    
    def post_thread(
        self,
        tweets: List[str],
        **kwargs
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Post a thread of tweets.
        
        Args:
            tweets: List of tweet texts (max 280 chars each)
            **kwargs: Additional API parameters
            
        Returns:
            List of tweet data if successful
        """
        if not self.enabled:
            logger.warning("Twitter client not enabled, skipping thread")
            return []
        
        results = []
        last_tweet_id = None
        
        for i, text in enumerate(tweets):
            if last_tweet_id:
                result = self.post(text, reply_to_id=last_tweet_id, **kwargs)
            else:
                result = self.post(text, **kwargs)
            
            if result:
                results.append(result)
                last_tweet_id = result["id"]
            else:
                logger.error(f"Failed to post tweet {i+1}/{len(tweets)}")
                break
            
            # Small delay between thread posts
            time.sleep(1)
        
        return results
    
    def delete(self, tweet_id: str) -> bool:
        """
        Delete a tweet.
        
        Args:
            tweet_id: ID of tweet to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            logger.warning("Twitter client not enabled, skipping delete")
            return False
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Twitter delete: ID={tweet_id}")
            return True
        
        try:
            self.client.delete_tweet(tweet_id)
            logger.info(f"Tweet deleted successfully: ID={tweet_id}")
            return True
        except tweepy.TweepyException as e:
            logger.error(f"Twitter API error deleting tweet: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting tweet: {e}")
            return False
    
    def is_enabled(self) -> bool:
        """Check if Twitter client is enabled."""
        return self.enabled
