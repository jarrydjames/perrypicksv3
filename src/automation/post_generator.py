"""Post Generator for PerryPicks v3 Automation.

Formats predictions into platform-optimized posts.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

import json

from src.automation.prediction_formatter import format_prediction

logger = logging.getLogger(__name__)

class PostGenerator:
    """Generate platform-optimized posts from predictions."""
    
    def __init__(
        self,
        include_odds: bool = True,
        include_confidence: bool = True,
        use_emojis: bool = True,
        hashtags: List[str] = None,
    ):
        """
        Initialize post generator.
        
        Args:
            include_odds: Include betting odds in posts
            include_confidence: Include win probability in posts
            use_emojis: Use emojis in posts
            hashtags: Default hashtags to include
        """
        self.include_odds = include_odds
        self.include_confidence = include_confidence
        self.use_emojis = use_emojis
        self.hashtags = hashtags or ["#NBAPredictions", "#PerryPicks"]
    
    def generate_pregame_post(
        self,
        prediction: Dict[str, Any],
        platform: str = "twitter"
    ) -> str:
        """
        Generate pregame prediction post.
        
        Args:
            prediction: Prediction dictionary from predict_game()
            platform: Platform (twitter, bluesky, discord)
            
        Returns:
            Formatted post text
        """
        if prediction.get("status") not in ("success", "warning"):
            return self._generate_error_post(prediction, platform)
        
        # Extract prediction data
        home_team = prediction.get("home_name", "Home")
        away_team = prediction.get("away_name", "Away")
        total = prediction.get("total", 0)
        margin = prediction.get("margin", 0)
        home_win_prob = prediction.get("home_win_prob", 0.5)
        model = prediction.get("model_used", "Unknown")
        
        # Calculate scores
        if isinstance(total, (int, float)) and isinstance(margin, (int, float)):
            home_score = (total + margin) / 2
            away_score = (total - margin) / 2
        else:
            home_score = None
            away_score = None
        
        # Determine winner
        if isinstance(margin, (int, float)):
            winner = home_team if margin > 0 else away_team
        else:
            winner = "N/A"
        
        # Build post based on platform
        if platform == "twitter":
            return self._generate_twitter_pregame(
                away_team, home_team,
                away_score, home_score,
                total, margin, winner,
                home_win_prob, model,
                prediction
            )
        elif platform == "bluesky":
            return self._generate_bluesky_pregame(
                away_team, home_team,
                away_score, home_score,
                total, margin, winner,
                home_win_prob, model,
                prediction
            )
        elif platform == "discord":
            return self._generate_discord_pregame(
                away_team, home_team,
                away_score, home_score,
                total, margin, winner,
                home_win_prob, model,
                prediction
            )
        else:
            return self._generate_generic_pregame(
                away_team, home_team,
                away_score, home_score,
                total, margin, winner,
                home_win_prob, model,
                prediction
            )
    
    def generate_halftime_post(
        self,
        prediction: Dict[str, Any],
        platform: str = "twitter"
    ) -> str:
        """
        Generate halftime prediction post.
        
        Args:
            prediction: Prediction dictionary
            platform: Platform (twitter, bluesky, discord)
            
        Returns:
            Formatted post text
        """
        if prediction.get("status") not in ("success", "warning"):
            return self._generate_error_post(prediction, platform)
        
        home_team = prediction.get("home_name", "Home")
        away_team = prediction.get("away_name", "Away")
        h1_home = prediction.get("h1_home", 0)
        h1_away = prediction.get("h1_away", 0)
        pred_2h_home = prediction.get("pred_2h_home", 0)
        pred_2h_away = prediction.get("pred_2h_away", 0)
        pred_final_home = prediction.get("pred_final_home", 0)
        pred_final_away = prediction.get("pred_final_away", 0)
        
        if platform == "twitter":
            emoji = "🔥" if self.use_emojis else "[2H]"
            post = (
                f"{emoji} HALFTIME UPDATE\n\n"
                f"{away_team} @ {home_team}\n\n"
                f"Halftime: {away_team} {h1_away} - {h1_home} {home_team}\n\n"
                f"Projected 2H: {away_team} {pred_2h_away:.1f} - {pred_2h_home:.1f} {home_team}\n\n"
                f"Projected Final: {away_team} {pred_final_away:.1f} - {pred_final_home:.1f} {home_team}\n\n"
            )
        else:
            post = (
                f"🔥 HALFTIME UPDATE: {away_team} @ {home_team}\n\n"
                f"Halftime: {away_team} {h1_away} - {h1_home} {home_team}\n\n"
                f"Projected Final: {away_team} {pred_final_away:.1f} - {pred_final_home:.1f} {home_team}\n"
            )
        
        return self._add_hashtags(post, platform)
    
    def generate_q3_post(
        self,
        prediction: Dict[str, Any],
        platform: str = "twitter"
    ) -> str:
        """
        Generate Q3-end prediction post.
        
        Args:
            prediction: Prediction dictionary
            platform: Platform (twitter, bluesky, discord)
            
        Returns:
            Formatted post text
        """
        if prediction.get("status") not in ("success", "warning"):
            return self._generate_error_post(prediction, platform)
        
        home_team = prediction.get("home_name", "Home")
        away_team = prediction.get("away_name", "Away")
        q3_cum_home = prediction.get("home_score", 0)  # Q3 cumulative = current score
        q3_cum_away = prediction.get("away_score", 0)  # Q3 cumulative = current score
        
        # Use quarter progression heuristic (NOT model predictions!)
        # Q3 model was trained incorrectly - it predicts impossible low finals
        # Use documented approach: Q4 ≈ Q3_cumulative × 0.32
        # See: README_MODELS.md and run_q3_predictions.py
        q3_cumulative_total = q3_cum_home + q3_cum_away
        q4_estimate_total = q3_cumulative_total * 0.32
        
        # Base Q4 for each team (half of estimate)
        q4_home_base = q4_estimate_total / 2
        q4_away_base = q4_estimate_total / 2
        
        # Adjust based on Q3 margin (momentum carries forward slightly)
        q3_margin = q3_cum_home - q3_cum_away
        margin_adjustment = q3_margin * 0.2
        
        q4_home = q4_home_base + margin_adjustment
        q4_away = q4_away_base - margin_adjustment
        
        # Ensure reasonable bounds (typical NBA quarter: 20-35 per team)
        q4_home = max(20, min(35, q4_home))
        q4_away = max(20, min(35, q4_away))
        
        # Project final scores (Q3 cumulative + estimated Q4)
        pred_final_home = q3_cum_home + q4_home
        pred_final_away = q3_cum_away + q4_away
        
        if platform == "twitter":
            emoji = "⚡" if self.use_emojis else "[Q3]"
            post = (
                f"{emoji} Q3 UPDATE\n\n"
                f"{away_team} @ {home_team}\n\n"
                f"Q3 Score: {away_team} {q3_cum_away:.1f} - {q3_cum_home:.1f} {home_team}\n\n"
                f"Projected Final: {away_team} {pred_final_away:.1f} - {pred_final_home:.1f} {home_team}\n\n"
            )
        else:
            post = (
                f"⚡ Q3 UPDATE: {away_team} @ {home_team}\n\n"
                f"Q3 Score: {away_team} {q3_cum_away:.1f} - {q3_cum_home:.1f} {home_team}\n\n"
                f"Projected Final: {away_team} {pred_final_away:.1f} - {pred_final_home:.1f} {home_team}\n\n"
            )
        
        return self._add_hashtags(post, platform)
    
    def _generate_twitter_pregame(
        self, away_team, home_team, away_score, home_score,
        total, margin, winner, home_win_prob, model, prediction
    ) -> str:
        """Generate Twitter-optimized pregame post."""
        emoji = "🏀" if self.use_emojis else ""
        
        # Basic game info
        post = (
            f"{emoji} PREGAME PREDICTION\n\n"
            f"{away_team} @ {home_team}\n\n"
        )
        
        # Add scores if available
        if away_score and home_score:
            post += f"Projected: {away_score:.1f} - {home_score:.1f}\n\n"
        
        # Add totals/margin
        if total:
            post += f"Total: {total:.1f} | Margin: {margin:+.1f}\n\n"
        
        # Add winner and confidence
        if self.include_confidence and home_win_prob:
            win_pct = home_win_prob * 100 if margin > 0 else (1 - home_win_prob) * 100
            post += f"Winner: {winner} ({win_pct:.0f}% confidence)\n\n"
        else:
            post += f"Winner: {winner}\n\n"
        
        # Add odds if available and requested
        if self.include_odds and prediction.get("odds"):
            odds = prediction["odds"]
            spread = odds.get("spread", "N/A")
            ou = odds.get("over_under", "N/A")
            post += f"Odds: Spread {spread} | O/U {ou}\n\n"
        
        return self._add_hashtags(post, "twitter")
    
    def _generate_bluesky_pregame(
        self, away_team, home_team, away_score, home_score,
        total, margin, winner, home_win_prob, model, prediction
    ) -> str:
        """Generate Bluesky-optimized pregame post."""
        post = (
            f"🏀 PREGAME: {away_team} @ {home_team}\n\n"
            f"Projected: {away_score:.1f} - {home_score:.1f}\n"
            f"Total: {total:.1f} | Margin: {margin:+.1f}\n\n"
            f"Winner: {winner}\n"
        )
        
        return self._add_hashtags(post, "bluesky")
    
    def _generate_discord_pregame(
        self, away_team, home_team, away_score, home_score,
        total, margin, winner, home_win_prob, model, prediction
    ) -> str:
        """Generate Discord-optimized pregame post."""
        return format_prediction(prediction.get("game_id", "unknown"), prediction)
    
    def _generate_generic_pregame(
        self, away_team, home_team, away_score, home_score,
        total, margin, winner, home_win_prob, model, prediction
    ) -> str:
        """Generate generic pregame post."""
        return self._generate_twitter_pregame(
            away_team, home_team, away_score, home_score,
            total, margin, winner, home_win_prob, model, prediction
        )
    
    def _generate_error_post(
        self,
        prediction: Dict[str, Any],
        platform: str = "twitter"
    ) -> str:
        """Generate error post."""
        game_id = prediction.get("game_id", "unknown")
        error = prediction.get("error", "Unknown error")
        
        if platform == "twitter":
            emoji = "⚠️" if self.use_emojis else "[ERROR]"
            return f"{emoji} Prediction failed for {game_id}: {error}"
        else:
            return f"Prediction failed for {game_id}: {error}"
    
    def _add_hashtags(self, post: str, platform: str) -> str:
        """Add platform-specific hashtags."""
        if platform == "twitter":
            # Twitter: #NBA + team-specific hashtags
            tags = self.hashtags + ["#NBA"]
            return f"{post} {' '.join(tags)}"
        elif platform == "bluesky":
            # Bluesky: fewer hashtags (2-3 is optimal)
            tags = self.hashtags[:2]
            return f"{post} {' '.join(tags)}"
        elif platform == "discord":
            # Discord: no hashtags needed
            return post
        else:
            return f"{post} {' '.join(self.hashtags)}"
    
    def generate_thread(
        self,
        prediction: Dict[str, Any],
        platform: str = "twitter"
    ) -> List[str]:
        """
        Generate thread for multi-part posts.
        
        Args:
            prediction: Prediction dictionary
            platform: Platform
            
        Returns:
            List of post texts
        """
        tweets = []
        
        # First tweet: Basic prediction
        tweets.append(self.generate_pregame_post(prediction, platform))
        
        # Second tweet: Detailed breakdown (if space needed)
        if prediction.get("status") in ("success", "warning"):
            breakdown = self._generate_breakdown_post(prediction, platform)
            if breakdown:
                tweets.append(breakdown)
        
        return tweets
    
    def _generate_breakdown_post(
        self,
        prediction: Dict[str, Any],
        platform: str
    ) -> str:
        """Generate detailed breakdown post."""
        model = prediction.get("model_used", "Unknown")
        home_team = prediction.get("home_name", "Home")
        away_team = prediction.get("away_name", "Away")
        
        if platform == "twitter":
            emoji = "📊" if self.use_emojis else ""
            return (
                f"{emoji} MODEL BREAKDOWN\n\n"
                f"Game: {away_team} @ {home_team}\n"
                f"Model: {model}\n\n"
                f"#PerryPicks #NBAPredictions"
            )
        else:
            return f"Model: {model}"