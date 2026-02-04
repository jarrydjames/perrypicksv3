"""
Discord client for PerryPicks v4 Automation System.
Handles posting messages to Discord with proper formatting.
"""

import logging
import requests
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import pytz

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
            'DAILY_SUMMARY': '📊 DAILY SUMMARY',
            'PRE_GAME': '⏰ PRE-GAME',
            'HALFTIME': '🏀 HALFTIME',
            'Q3': '🏀 END Q3'
        }.get(trigger_type, trigger_type)
        
        # Format local time (America/Chicago)
        local_time = timestamp.astimezone(pytz.timezone('America/Chicago'))
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
    
    def format_daily_summary_post(
        self,
        predictions: list,
        timestamp: datetime,
        date: str
    ) -> str:
        """
        Format a daily summary post with predictions for all games.
        
        Args:
            predictions: List of game predictions with:
                - game_id, away_name, home_name
                - predicted_away_score, predicted_home_score
                - predicted_winner, predicted_margin
                - predicted_total
            timestamp: Current timestamp
            date: Date string (YYYY-MM-DD)
        
        Returns:
            Formatted Discord message
        """
        # Format local time
        local_time = timestamp.astimezone(pytz.timezone('America/Chicago'))
        local_time_str = local_time.strftime('%I:%M %p %Z')
        
        # Build header
        header = f"**📊 DAILY SUMMARY — {date}**"
        header += f"\n_Posted at {local_time_str}_"
        
        # Build predictions list
        body_lines = []
        body_lines.append("\n**Today's Games:**\n")
        
        for pred in predictions:
            away_name = pred.get('away_name', 'Away')
            home_name = pred.get('home_name', 'Home')
            pred_away = pred.get('predicted_away_score', 0)
            pred_home = pred.get('predicted_home_score', 0)
            pred_total = pred.get('predicted_total', 0)
            pred_margin = pred.get('predicted_margin', 0)
            pred_winner = pred.get('predicted_winner', 'Unknown')
            
            # Format margin with sign
            if pred_margin < 0:
                margin_str = f"{home_name} by {abs(pred_margin):.1f}"
            else:
                margin_str = f"{away_name} by {pred_margin:.1f}"
            
            # Format game line
            game_line = f"🏀 **{away_name} @ {home_name}**"
            game_line += f"\n   Pred: {away_name} {pred_away:.1f} - {home_name} {pred_home:.1f}"
            game_line += f"\n   Winner: {pred_winner} ({margin_str})"
            game_line += f"\n   Total: {pred_total:.1f}\n"
            
            body_lines.append(game_line)
        
        # Build footer
        footer = f"\n_📊 Data: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC_"
        footer += f"\n_⚠️ Predictions may change closer to game time_"
        
        # Combine all parts
        full_message = f"{header}\n{''.join(body_lines)}{footer}"
        
        return full_message
    
    def format_halftime_post(
        self,
        game_data: Dict[str, Any],
        prediction: Dict[str, Any],
        timestamp: datetime
    ) -> str:
        """
        Format a halftime prediction post.
        
        Args:
            game_data: Game state with current scores
            prediction: Prediction results
            timestamp: Current timestamp
        
        Returns:
            Formatted Discord message
        """
        # Format local time
        local_time = timestamp.astimezone(pytz.timezone('America/Chicago'))
        local_time_str = local_time.strftime('%I:%M %p %Z')
        
        # Get team names
        away_name = game_data.get('away_name', game_data.get('away_team', 'Away'))
        home_name = game_data.get('home_name', game_data.get('home_team', 'Home'))
        
        # Get current scores
        current_away = game_data.get('score_away', 0)
        current_home = game_data.get('score_home', 0)
        
        # Get prediction
        pred_away = prediction.get('predicted_away_score', 0)
        pred_home = prediction.get('predicted_home_score', 0)
        pred_total = prediction.get('predicted_total', 0)
        pred_margin = prediction.get('predicted_margin', 0)
        
        # Determine winner
        if pred_margin < 0:
            pred_winner = home_name
            abs_margin = abs(pred_margin)
        else:
            pred_winner = away_name
            abs_margin = pred_margin
        
        # Build header
        header = f"**🏀 HALFTIME UPDATE — {away_name} @ {home_name}**"
        header += f"\n_Posted at {local_time_str}_"
        
        # Build body
        body_lines = []
        body_lines.append("\n**Halftime Score:**\n")
        body_lines.append(f"{away_name} {current_away} - {home_name} {current_home}\n")
        
        body_lines.append("**Final Score Prediction:**\n")
        body_lines.append(f"{away_name} {pred_away:.1f} - {home_name} {pred_home:.1f}")
        body_lines.append(f"Winner: {pred_winner} by {abs_margin:.1f}")
        body_lines.append(f"Total: {pred_total:.1f}\n")
        
        # Build footer
        footer = f"\n_📊 Data: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC_"
        
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
