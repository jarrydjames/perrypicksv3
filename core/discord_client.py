"""
Discord client for PerryPicks v4 Automation System.
Handles posting messages to Discord with proper formatting.
"""

import logging
import requests
from typing import Optional, Dict, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class DiscordWebhookClient:
    """Discord webhook client for posting messages."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def post_message(self, content: str, embed: Optional[Dict] = None) -> Optional[str]:
        """
        Post a message to Discord via webhook.
        
        Args:
            content: Message content (markdown supported)
            embed: Optional embed object for rich formatting
        
        Returns:
            message_id if successful, None otherwise
        """
        try:
            payload = {'content': content}
            if embed:
                payload['embeds'] = [embed]
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            # Try to extract message ID from response
            # Webhook responses don't always include message ID
            # Some webhooks return JSON with id, others return empty response
            message_id = None
            try:
                if response.content:
                    data = response.json()
                    message_id = data.get('id')
            except Exception:
                # Response is empty or not JSON - that's okay
                pass
            
            logger.info(f"Posted Discord message (id={message_id})")
            return message_id
            
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error posting to Discord: {e}")
            return None
        except Exception as e:
            logger.error(f"Error posting to Discord: {e}")
            return None
    
    def format_bet_post(
        self,
        trigger_type: str,
        game_data: Dict[str, Any],
        picks: list,
        timestamp: datetime
    ) -> str:
        """
        Format a bet post according to the template:
        
        Header: [TRIGGER] Away @ Home — {local time} — current score/state
        Body: Top 3 Bets
        Footer: Data timestamp + odds caching note
        """
        # Format trigger type
        trigger_display = {
            'PRE_3H': '⏰ T-3H',
            'PRE_1H': '⏰ T-1H',
            'PRE_10M': '⏰ T-10M',
            'HALFTIME': '🏀 HALFTIME',
            'Q3': '🏀 END Q3'
        }.get(trigger_type, trigger_type)
        
        # Format local time (America/Chicago)
        local_time = timestamp.astimezone(timezone('America/Chicago'))
        local_time_str = local_time.strftime('%I:%M %p %Z')
        
        # Build header
        header = f"**{trigger_display} {game_data['away_team']} @ {game_data['home_team']}** — {local_time_str}"
        
        # Add current score/state if game in progress
        if game_data.get('status') in ['In Progress', 'Halftime']:
            period = game_data.get('current_period', 0)
            clock = game_data.get('game_clock', '0:00')
            score_home = game_data.get('score_home', 0)
            score_away = game_data.get('score_away', 0)
            header += f" — Q{period} {clock} — {away_team} {score_away} @ {home_team} {score_home}"
        
        # Build body with top 3 bets
        body_lines = []
        body_lines.append("\n**Top Bets:**\n")
        
        for i, pick in enumerate(picks[:3], 1):
            bet_type = pick.get('bet_type', 'Unknown')
            side = pick.get('side', 'Unknown')
            line = pick.get('line')
            odds = pick.get('odds')
            probability = pick.get('probability', 0) * 100
            edge = pick.get('edge', 0) * 100
            book = pick.get('book', 'Unknown')
            rationale = pick.get('rationale', '')
            
            # Format bet line
            bet_line = f"{i}. **{side}"
            if line is not None:
                bet_line += f" {line}"
            bet_line += f"** ({bet_type})"
            
            # Add odds and edge
            bet_line += f" | Prob: {probability:.1f}% | Edge: {edge:.1f}% | Odds: {odds} ({book})"
            
            body_lines.append(bet_line)
            
            if rationale:
                body_lines.append(f"   → {rationale}")
            body_lines.append("")  # Empty line for spacing
        
        # Build footer
        footer = f"\n_📊 Data: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC_"
        footer += f"\n_⚠️ Odds cached; check freshness before placing bets_"
        
        # Combine all parts
        full_message = f"{header}\n{''.join(body_lines)}{footer}"
        
        return full_message


class DiscordBotClient:
    """
    Discord bot client for advanced features (editing, replying).
    Falls back to webhook if bot token not available.
    """
    
    def __init__(self, bot_token: Optional[str] = None):
        self.bot_token = bot_token
        if bot_token:
            self.base_url = f"https://discord.com/api/v10"
            self.headers = {'Authorization': f'Bot {bot_token}'}
            self.use_bot = True
        else:
            self.use_bot = False
            logger.info("No bot token provided; will use webhook-only mode")
    
    def edit_message(
        self,
        channel_id: str,
        message_id: str,
        content: str
    ) -> bool:
        """Edit an existing message (requires bot token)."""
        if not self.use_bot:
            logger.warning("Cannot edit message: no bot token")
            return False
        
        try:
            url = f"{self.base_url}/channels/{channel_id}/messages/{message_id}"
            response = requests.patch(url, json={'content': content}, headers=self.headers, timeout=10)
            response.raise_for_status()
            logger.info(f"Edited Discord message {message_id}")
            return True
        except Exception as e:
            logger.error(f"Error editing Discord message: {e}")
            return False
    
    def reply_to_message(
        self,
        channel_id: str,
        message_id: str,
        content: str
    ) -> Optional[str]:
        """Reply to an existing message (requires bot token)."""
        if not self.use_bot:
            logger.warning("Cannot reply to message: no bot token")
            return None
        
        try:
            url = f"{self.base_url}/channels/{channel_id}/messages"
            payload = {
                'content': content,
                'message_reference': {'message_id': message_id}
            }
            response = requests.post(url, json=payload, headers=self.headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Replied to Discord message {message_id}")
            return data.get('id')
        except Exception as e:
            logger.error(f"Error replying to Discord message: {e}")
            return None
