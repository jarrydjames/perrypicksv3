"""Post Generator for PerryPicks v3 Automation.

Formats predictions into platform-optimized posts.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.automation.prediction_formatter import format_prediction
from src.automation.post_generator_helpers import (
    _format_probability,
    _generate_best_bets,
)

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
        """
        self.include_odds = include_odds
        self.include_confidence = include_confidence
        self.use_emojis = use_emojis
        self.hashtags = hashtags or ['#NBAPredictions', '#PerryPicks']
    
    def generate_pregame_post(
        self,
        prediction: Dict[str, Any],
        platform: str = 'twitter'
    ) -> str:
        """
        Generate pregame prediction post.
        """
        if prediction.get('status') not in ('success', 'warning'):
            return self._generate_error_post(prediction, platform)
        
        home_team = prediction.get('home_name', 'Home')
        away_team = prediction.get('away_name', 'Away')
        total = prediction.get('total', 0)
        margin = prediction.get('margin', 0)
        home_win_prob = prediction.get('home_win_prob', 0.5)
        model = prediction.get('model_used', 'Unknown')
        
        if platform == 'discord':
            return self._generate_discord_pregame(
                away_team, home_team, total, margin,
                home_win_prob, model, prediction
            )
        else:
            return self._generate_twitter_pregame(
                away_team, home_team, total, margin,
                home_win_prob, model, prediction
            )
    
    def generate_halftime_post(
        self,
        prediction: Dict[str, Any],
        platform: str = 'twitter'
    ) -> str:
        """
        Generate halftime prediction post with best bets.
        """
        if prediction.get('status') not in ('success', 'warning'):
            return self._generate_error_post(prediction, platform)
        
        home_team = prediction.get('home_name', 'Home')
        away_team = prediction.get('away_name', 'Away')
        h1_home = prediction.get('h1_home', 0)
        h1_away = prediction.get('h1_away', 0)
        pred_final_home = prediction.get('pred_final_home', 0)
        pred_final_away = prediction.get('pred_final_away', 0)
        
        # Calculate final stats and generate bets
        final_total = pred_final_home + pred_final_away
        final_margin = pred_final_home - pred_final_away
        winner = home_team if final_margin > 0 else away_team
        bets = _generate_best_bets(prediction, 'halftime', max_bets=3)
        
        if platform == 'twitter':
            emoji = '🔥' if self.use_emojis else '[2H]'
            post = (
                f'{emoji} HALFTIME UPDATE\n\n'
                f'{away_team} @ {home_team}\n\n'
                f'Halftime: {away_team} {h1_away} - {h1_home} {home_team}\n\n'
                f'Projected Final: {away_team} {pred_final_away:.1f} - {pred_final_home:.1f} {home_team}\n\n'
                f'Winner: {winner} | Margin: {final_margin:+.1f} | Total: {final_total:.1f}\n\n'
            )
            post += self._format_bets_section(bets, platform)
        else:
            post = (
                f'🔥 HALFTIME UPDATE: {away_team} @ {home_team}\n\n'
                f'Halftime: {away_team} {h1_away} - {h1_home} {home_team}\n\n'
                f'Projected Final: {away_team} {pred_final_away:.1f} - {pred_final_home:.1f} {home_team}\n\n'
                f'Winner: {winner} | Margin: {final_margin:+.1f} | Total: {final_total:.1f}\n\n'
            )
            post += self._format_bets_section(bets, platform)
        
        return self._add_hashtags(post, platform)
    
    def generate_q3_post(
        self,
        prediction: Dict[str, Any],
        platform: str = 'twitter'
    ) -> str:
        """
        Generate Q3-end prediction post with best bets.
        """
        if prediction.get('status') not in ('success', 'warning'):
            return self._generate_error_post(prediction, platform)
        
        home_team = prediction.get('home_name', 'Home')
        away_team = prediction.get('away_name', 'Away')
        q3_cum_home = prediction.get('home_score', 0)
        q3_cum_away = prediction.get('away_score', 0)
        
        # Use quarter progression heuristic
        q3_cumulative_total = q3_cum_home + q3_cum_away
        q4_estimate_total = q3_cumulative_total * 0.32
        
        q4_home_base = q4_estimate_total / 2
        q4_away_base = q4_estimate_total / 2
        
        q3_margin = q3_cum_home - q3_cum_away
        margin_adjustment = q3_margin * 0.2
        
        q4_home = q4_home_base + margin_adjustment
        q4_away = q4_away_base - margin_adjustment
        
        q4_home = max(20, min(35, q4_home))
        q4_away = max(20, min(35, q4_away))
        
        pred_final_home = q3_cum_home + q4_home
        pred_final_away = q3_cum_away + q4_away
        
        # Calculate final stats and generate bets
        final_total = pred_final_home + pred_final_away
        final_margin = pred_final_home - pred_final_away
        winner = home_team if final_margin > 0 else away_team
        bets = _generate_best_bets(prediction, 'q3', max_bets=3)
        
        if platform == 'twitter':
            emoji = '⚡' if self.use_emojis else '[Q3]'
            post = (
                f'{emoji} Q3 UPDATE\n\n'
                f'{away_team} @ {home_team}\n\n'
                f'Q3 Score: {away_team} {q3_cum_away:.1f} - {q3_cum_home:.1f} {home_team}\n\n'
                f'Projected Final: {away_team} {pred_final_away:.1f} - {pred_final_home:.1f} {home_team}\n\n'
                f'Winner: {winner} | Margin: {final_margin:+.1f} | Total: {final_total:.1f}\n\n'
            )
            post += self._format_bets_section(bets, platform)
        else:
            post = (
                f'⚡ Q3 UPDATE: {away_team} @ {home_team}\n\n'
                f'Q3 Score: {away_team} {q3_cum_away:.1f} - {q3_cum_home:.1f} {home_team}\n\n'
                f'Projected Final: {away_team} {pred_final_away:.1f} - {pred_final_home:.1f} {home_team}\n\n'
                f'Winner: {winner} | Margin: {final_margin:+.1f} | Total: {final_total:.1f}\n\n'
            )
            post += self._format_bets_section(bets, platform)
        
        return self._add_hashtags(post, platform)
    
    def _format_bets_section(
        self, bets: List[Dict[str, Any]], platform: str
    ) -> str:
        """
        Format best bets section for post.
        """
        if not bets:
            return ''
        
        if platform == 'twitter':
            section = 'Top Bets (by edge):\n\n'
            for i, bet in enumerate(bets, 1):
                section += f"{bet['side']} @ {bet['odds']} (edge {bet['edge']*100:+.1f}%)\n"
                section += f"   P: {_format_probability(bet['probability'])}\n\n"
        else:
            section = '🎯 Best Bets (Top 3 by Edge):\n\n'
            for i, bet in enumerate(bets, 1):
                emoji = '🔥' if i == 1 else ('✅' if i == 2 else '💰')
                section += f"{emoji} {i}. {bet['type']} {bet['side']} @ {bet['odds']} (edge {bet['edge']*100:+.1f}%)\n"
                section += f"   P: {_format_probability(bet['probability'])}\n\n"
        
        return section
    
    def _generate_twitter_pregame(
        self, away_team, home_team, total, margin,
        home_win_prob, model, prediction
    ) -> str:
        """Generate Twitter-optimized pregame post."""
        emoji = '🏀' if self.use_emojis else ''
        
        post = (
            f'{emoji} PREGAME PREDICTION\n\n'
            f'{away_team} @ {home_team}\n\n'
        )
        
        if isinstance(total, (int, float)) and isinstance(margin, (int, float)):
            home_score = (total + margin) / 2
            away_score = (total - margin) / 2
            post += f'Projected: {away_score:.1f} - {home_score:.1f}\n\n'
        
        if total:
            post += f'Total: {total:.1f} | Margin: {margin:+.1f}\n\n'
        
        if self.include_confidence and home_win_prob:
            win_pct = home_win_prob * 100 if margin > 0 else (1 - home_win_prob) * 100
            winner = home_team if margin > 0 else away_team
            post += f'Winner: {winner} ({win_pct:.0f}% confidence)\n\n'
        elif isinstance(margin, (int, float)):
            winner = home_team if margin > 0 else away_team
            post += f'Winner: {winner}\n\n'
        
        if self.include_odds and prediction.get('odds'):
            odds = prediction['odds']
            spread = odds.get('spread', 'N/A')
            ou = odds.get('over_under', 'N/A')
            post += f'Odds: Spread {spread} | O/U {ou}\n\n'
        
        return self._add_hashtags(post, 'twitter')
    
    def _generate_discord_pregame(
        self, away_team, home_team, total, margin,
        home_win_prob, model, prediction
    ) -> str:
        """Generate Discord-optimized pregame post (Option 3 table)."""
        games = prediction.get('games', [prediction])
        
        if len(games) == 1:
            return self._generate_discord_single_game(
                away_team, home_team, total, margin,
                home_win_prob, model, prediction
            )
        else:
            return self._generate_discord_full_slate(games, model, prediction)
    
    def _generate_discord_single_game(
        self, away_team, home_team, total, margin,
        home_win_prob, model, prediction
    ) -> str:
        """Generate Discord single game post."""
        home_score = (total + margin) / 2
        away_score = (total - margin) / 2
        winner = home_team if margin > 0 else away_team
        win_pct = home_win_prob * 100 if margin > 0 else (1 - home_win_prob) * 100
        
        post = (
            f'📊 **{away_team} @ {home_team}**\n\n'
            f'📈 **Predicted Scores:**\n'
            f'{away_team} {away_score:.1f} - {home_score:.1f} {home_team}\n\n'
            f'🎯 **Projected Winner:** {winner}\n'
            f'🏆 **Win Probability:** {win_pct:.1f}%\n'
            f'📊 **Game Total:** {total:.1f}\n'
            f'📏 **Margin:** {margin:+.1f}\n\n'
            f'Model: {model} | Confidence: Medium\n\n'
        )
        
        return post
    
    def _generate_discord_full_slate(
        self, games: List[Dict[str, Any]],
        model: str, prediction
    ) -> str:
        """Generate Discord full slate post (Option 3 table)."""
        date_str = datetime.now().strftime('%Y-%m-%d')
        
        post = f'📊 **NBA PREGAME PREDICTIONS - {date_str}**\n\n'
        post += '| # | Away → Home | 🏆 Winner | 📈 Prob | 🏀 Score | 🎯 Total | ➕ Margin |\n'
        post += '|---|-------------|------------|---------|----------|-----------|-----------|\n'
        
        for i, game in enumerate(games, 1):
            away = game.get('away_name', 'AWAY')
            home = game.get('home_name', 'HOME')
            total = game.get('total', 0)
            margin = game.get('margin', 0)
            home_win_prob = game.get('home_win_prob', 0.5)
            
            # Calculate individual team scores from total and margin
            # margin = home_score - away_score
            # total = home_score + away_score
            # home_score = (total + margin) / 2
            # away_score = (total - margin) / 2
            home_score = (total + margin) / 2
            away_score = (total - margin) / 2
            
            winner = home if margin > 0 else away
            win_pct = home_win_prob * 100 if margin > 0 else (1 - home_win_prob) * 100
            
            post += (
                f'| {i} | {away} → {home} | **{winner}** | {win_pct:.1f}% | '
                f'{away_score:.1f}-{home_score:.1f} | {total:.1f} | {margin:+.1f} |\n'
            )
        
        post += f'\nModel: {model} | Games: {len(games)} | Confidence: High\n\n'
        
        return post
    
    def _generate_error_post(
        self, prediction: Dict[str, Any], platform: str = 'twitter'
    ) -> str:
        """Generate error post."""
        game_id = prediction.get('game_id', 'unknown')
        error = prediction.get('error', 'Unknown error')
        
        if platform == 'twitter':
            emoji = '⚠️' if self.use_emojis else '[ERROR]'
            return f'{emoji} Prediction failed for {game_id}: {error}'
        else:
            return f'Prediction failed for {game_id}: {error}'
    
    def _add_hashtags(self, post: str, platform: str) -> str:
        """Add platform-specific hashtags."""
        if platform == 'twitter':
            tags = self.hashtags + ['#NBA']
            return f'{post} {" ".join(tags)}'
        elif platform == 'bluesky':
            tags = self.hashtags[:2]
            return f'{post} {" ".join(tags)}'
        else:
            return post
