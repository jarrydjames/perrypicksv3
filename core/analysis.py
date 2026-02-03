"""
Analysis module for PerryPicks v4 Automation System.
Integrates real prediction models from the existing framework.
"""

import logging
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# Import prediction framework
from src.predict_api import predict_game
from src.domain.markets import MarketInputs, evaluate_markets
from src.betting import parse_american_odds, breakeven_prob_from_american, kelly_fraction


class AnalysisEngine:
    """
    Wrapper around existing prediction models.
    Integrates pregame, halftime, and Q3 models with betting analysis.
    """
    
    def __init__(self):
        self.models_loaded = True  # predict_game handles lazy loading
    
    def run_analysis(
        self,
        game_state: Dict[str, Any],
        odds: Dict[str, Any],
        mode: str
    ) -> List[Dict[str, Any]]:
        """
        Run analysis for a game.
        
        Args:
            game_state: Current game state from NBA API
                - game_id, home_team, away_team, status, current_period, game_clock, scores, etc.
            odds: Cached or fresh odds from Odds API
                - spread, total, moneyline, books
            mode: Analysis mode (trigger type)
                - 'PRE_3H', 'PRE_1H', 'PRE_10M': Pregame analysis
                - 'HALFTIME': Halftime model
                - 'Q3': End of Q3 model
        
        Returns:
            List of top 3 bets (ranked by edge) with:
            - bet_rank, bet_type, side, line, odds, book
            - probability, edge, rationale
        """
        try:
            game_id = game_state.get('game_id')
            home_team = game_state.get('home_team')
            away_team = game_state.get('away_team')
            
            logger.info(f"Running {mode} analysis for {home_team} vs {away_team} ({game_id})")
            
            # Step 1: Get prediction from model
            prediction = self._get_prediction(game_id, mode, game_state)
            
            if not prediction:
                logger.warning(f"No prediction returned for {game_id} in mode {mode}")
                return []
            
            # Step 2: Build market inputs from odds
            market_inputs = self._build_market_inputs(odds, home_team, away_team)
            
            # Step 3: Evaluate markets to get bet recommendations
            recommendations = self._evaluate_markets(prediction, market_inputs, home_team, away_team, mode)
            
            # Step 4: Transform to automation format and rank
            picks = self._transform_to_picks(recommendations, prediction, mode, home_team, away_team)
            
            return picks[:3]  # Return top 3
            
        except Exception as e:
            logger.error(f"Error running {mode} analysis for {game_state.get('game_id')}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _get_prediction(
        self,
        game_id: str,
        mode: str,
        game_state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Get prediction from appropriate model based on mode.
        
        Uses correct model for each trigger type:
        - PRE_3H, PRE_1H, PRE_10M: Uses pregame model (no live data needed)
        - HALFTIME: Uses halftime model (needs Q1+Q2 data)
        - Q3: Uses Q3 model (needs Q1+Q2+Q3 data)
        
        This approach calls the correct model directly, avoiding the auto-detection
        issues in predict_game which can cause KeyError('period') errors.
        """
        try:
            # Determine which model to use based on trigger type
            pregame_modes = ['PRE_3H', 'PRE_1H', 'PRE_10M']
            
            if mode in pregame_modes:
                # Use pregame model - NO LIVE DATA NEEDED
                logger.info(f"Using pregame model for {game_id} ({mode})")
                from src.predict_pregame import predict_from_game_id as predict_pregame
                
                home_team = game_state.get('home_team')
                away_team = game_state.get('away_team')
                
                if not home_team or not away_team:
                    logger.error(f"Missing team tricodes for pregame prediction: {game_id}")
                    return None
                
                prediction = predict_pregame(
                    game_id=game_id,
                    home_team=home_team,
                    away_team=away_team,
                    fetch_odds=False
                )
                
            elif mode == 'HALFTIME':
                # Use halftime model - needs Q1+Q2 live data
                logger.info(f"Using halftime model for {game_id}")
                from src.predict_from_gameid_v2_ci import predict_from_game_id as predict_halftime
                
                prediction = predict_halftime(game_id)
                
                # Halftime model returns different structure - normalize it
                if prediction and isinstance(prediction.get('status'), dict):
                    pred = prediction.get('pred', {})
                    prediction = {
                        'game_id': prediction.get('game_id'),
                        'home_name': prediction.get('home_name'),
                        'away_name': prediction.get('away_name'),
                        'margin': pred.get('pred_final_margin', 0),
                        'total': pred.get('pred_final_total', 0),
                        'margin_q10': pred.get('pred_final_margin', 0) - 10,
                        'margin_q90': pred.get('pred_final_margin', 0) + 10,
                        'total_q10': pred.get('pred_final_total', 0) - 15,
                        'total_q90': pred.get('pred_final_total', 0) + 15,
                        'home_win_prob': None,
                        'margin_sd': None,
                        'total_sd': None,
                        'model_used': 'HALFTIME_V2_CI',
                        'status': 'success',
                    }
                
            elif mode == 'Q3':
                # Use Q3 model - needs Q1+Q2+Q3 live data
                logger.info(f"Using Q3 model for {game_id}")
                from src.predict_from_gameid_v3_runtime import predict_from_game_id as predict_q3
                
                prediction = predict_q3(game_id, fetch_odds=False)
                
                # Q3 predictor doesn't set 'status' field - set it if we have valid prediction
                if prediction and 'margin' in prediction and 'total' in prediction:
                    prediction['status'] = 'success'
                
            else:
                logger.warning(f"Unknown mode: {mode}")
                return None
            
            # Validate prediction
            if not prediction:
                logger.warning(f"No prediction returned for {game_id}")
                return None
            
            if prediction.get('status') == 'error':
                error_msg = prediction.get('error', 'Unknown error')
                logger.warning(f"Prediction error for {game_id}: {error_msg}")
                return None
            
            # Ensure required fields are present
            required_fields = ['game_id', 'home_name', 'away_name', 'margin', 'total']
            missing_fields = [f for f in required_fields if f not in prediction]
            if missing_fields:
                logger.error(f"Prediction missing required fields: {missing_fields}")
                return None
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error getting prediction for {game_id} (mode={mode}): {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _build_market_inputs(
        self,
        odds: Dict[str, Any],
        home_team: str,
        away_team: str
    ) -> MarketInputs:
        """Build MarketInputs from odds dictionary."""
        # Extract spread odds
        spread_odds = odds.get('spread', {})
        spread_home = float(spread_odds.get('home_spread', 0.0))
        odds_spread_home = self._safe_parse_odds(spread_odds.get('home_odds', -110))
        odds_spread_away = self._safe_parse_odds(spread_odds.get('away_odds', -110))
        
        # Extract total odds
        total_odds = odds.get('total', {})
        total_line = float(total_odds.get('total', 230.0))
        odds_total_over = self._safe_parse_odds(total_odds.get('over_odds', -110))
        odds_total_under = self._safe_parse_odds(total_odds.get('under_odds', -110))
        
        # Extract moneyline odds
        moneyline_odds = odds.get('moneyline', {})
        odds_home_ml = self._safe_parse_odds(moneyline_odds.get('home_ml'))
        odds_away_ml = self._safe_parse_odds(moneyline_odds.get('away_ml'))
        
        # Team totals (optional)
        team_home_odds = odds.get('team_total_home', {})
        team_away_odds = odds.get('team_total_away', {})
        
        return MarketInputs(
            total_line=total_line,
            odds_over=odds_total_over,
            odds_under=odds_total_under,
            spread_home=spread_home,
            odds_home=odds_spread_home,
            odds_away=odds_spread_away,
            moneyline_home=odds_home_ml,
            moneyline_away=odds_away_ml,
            team_total_home=float(team_home_odds.get('line', 0.0)),
            team_total_away=float(team_away_odds.get('line', 0.0)),
            odds_team_over_home=self._safe_parse_odds(team_home_odds.get('over_odds')),
            odds_team_under_home=self._safe_parse_odds(team_home_odds.get('under_odds')),
            odds_team_over_away=self._safe_parse_odds(team_away_odds.get('over_odds')),
            odds_team_under_away=self._safe_parse_odds(team_away_odds.get('under_odds')),
            bankroll=1000.0,
            kelly_mult=0.5,
        )
    
    def _safe_parse_odds(self, odds: Any) -> Optional[int]:
        """Safely parse American odds."""
        if odds is None:
            return None
        try:
            return int(parse_american_odds(odds))
        except (ValueError, TypeError):
            return None
    
    def _evaluate_markets(
        self,
        prediction: Dict[str, Any],
        inputs: MarketInputs,
        home_team: str,
        away_team: str,
        mode: str
    ) -> List[Dict[str, Any]]:
        """Evaluate markets using prediction and odds."""
        try:
            # Extract prediction values
            final_total_mu = prediction.get('total')
            final_margin_mu = prediction.get('margin')
            
            # Compute team totals from total + margin
            if final_total_mu is not None and final_margin_mu is not None:
                final_home_mu = (final_total_mu + final_margin_mu) / 2.0
                final_away_mu = (final_total_mu - final_margin_mu) / 2.0
            else:
                final_home_mu = None
                final_away_mu = None
            
            # Extract standard deviations
            sd_total = prediction.get('total_sd', 10.0)
            sd_margin = prediction.get('margin_sd', 8.0)
            
            # Team totals have higher uncertainty
            sd_team = max(0.01, ((sd_total ** 2 + sd_margin ** 2) ** 0.5) / 2.0)
            
            # Extract 80% confidence bands
            bands = {
                'final_total': [prediction.get('total_q10', final_total_mu - 15), 
                              prediction.get('total_q90', final_total_mu + 15)],
                'final_margin': [prediction.get('margin_q10', final_margin_mu - 10), 
                               prediction.get('margin_q90', final_margin_mu + 10)],
            }
            
            # Evaluate markets
            recs = evaluate_markets(
                pred={'bands80': bands},
                home_name=home_team,
                away_name=away_team,
                final_total_mu=final_total_mu,
                final_margin_mu=final_margin_mu,
                final_home_mu=final_home_mu,
                final_away_mu=final_away_mu,
                sd_total=sd_total,
                sd_margin=sd_margin,
                sd_team=sd_team,
                inputs=inputs,
                policy=None,  # No policy filtering for automation
            )
            
            return recs
            
        except Exception as e:
            logger.error(f"Error evaluating markets: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _transform_to_picks(
        self,
        recommendations: List[Dict[str, Any]],
        prediction: Dict[str, Any],
        mode: str,
        home_team: str,
        away_team: str
    ) -> List[Dict[str, Any]]:
        """
        Transform market recommendations to automation pick format.
        
        Format:
        {
            'bet_rank': int,
            'bet_type': str,  # 'spread', 'total', 'moneyline'
            'side': str,  # team name or 'Over'/'Under'
            'line': float,
            'odds': int,
            'book': str,
            'probability': float,
            'edge': float,
            'rationale': str
        }
        """
        picks = []
        
        for i, rec in enumerate(recommendations, 1):
            # Skip negative edge bets
            if rec.get('edge', 0) <= 0:
                continue
            
            bet_type = rec.get('type', 'Unknown')
            side = rec.get('side', 'Unknown')
            
            # Normalize bet type
            if bet_type == 'Total':
                bet_type = 'total'
            elif bet_type == 'Spread':
                bet_type = 'spread'
            elif bet_type == 'Moneyline':
                bet_type = 'moneyline'
            elif bet_type == 'Team total':
                bet_type = 'team_total'
            
            # Extract line and odds
            line = rec.get('line')
            if line is not None:
                line = float(line)
            
            odds = rec.get('odds')
            if odds is not None:
                odds = int(odds)
            
            # Build rationale based on prediction
            rationale = self._build_rationale(bet_type, side, line, prediction, mode)
            
            picks.append({
                'bet_rank': i,
                'bet_type': bet_type,
                'side': side,
                'line': line,
                'odds': odds,
                'book': 'Various',  # Odds are aggregated from multiple books
                'probability': rec.get('p', 0.5),
                'edge': rec.get('edge', 0.0),
                'rationale': rationale
            })
        
        return picks
    
    def _build_rationale(
        self,
        bet_type: str,
        side: str,
        line: float,
        prediction: Dict[str, Any],
        mode: str
    ) -> str:
        """Build rationale text for a bet."""
        total_mu = prediction.get('total')
        margin_mu = prediction.get('margin')
        
        if bet_type == 'total':
            if 'Over' in str(side):
                diff = total_mu - line if total_mu is not None and line else 0
                if diff > 5:
                    return f"Model predicts {total_mu:.1f} points, {diff:.1f} above line"
                elif diff > 0:
                    return f"Model predicts {total_mu:.1f} points, slightly above line"
                else:
                    return f"Model projects total near {total_mu:.1f}"
            else:  # Under
                diff = line - total_mu if total_mu is not None and line else 0
                if diff > 5:
                    return f"Model predicts {total_mu:.1f} points, {diff:.1f} below line"
                elif diff > 0:
                    return f"Model predicts {total_mu:.1f} points, slightly below line"
                else:
                    return f"Model projects total near {total_mu:.1f}"
        
        elif bet_type == 'spread':
            home_team = prediction.get('home_name', 'Home')
            away_team = prediction.get('away_name', 'Away')
            
            if home_team in str(side) or 'home' in str(side).lower():
                diff = margin_mu - line if margin_mu is not None and line else 0
                if diff > 3:
                    return f"Model predicts {home_team} wins by {margin_mu:.1f}, covering the spread"
                elif diff > 0:
                    return f"Model predicts {home_team} win margin is {margin_mu:.1f}, close to spread"
                else:
                    return f"Model projects {home_team} win margin near {margin_mu:.1f}"
            else:  # Away team
                if margin_mu is not None and line is not None:
                    spread_val = -line  # Away spread is negative of home spread
                    diff = margin_mu - spread_val
                    if diff < -3:
                        return f"Model predicts {away_team} wins by {-margin_mu:.1f}, covering the spread"
                    elif diff < 0:
                        return f"Model predicts {away_team} win margin is {-margin_mu:.1f}, close to spread"
                    else:
                        return f"Model projects {away_team} win margin near {-margin_mu:.1f}"
                else:
                    return f"Model favors {away_team} in this matchup"
        
        elif bet_type == 'moneyline':
            home_team = prediction.get('home_name', 'Home')
            away_team = prediction.get('away_name', 'Away')
            win_prob = prediction.get('home_win_prob', 0.5)
            
            if home_team in str(side):
                return f"Model gives {home_team} {win_prob*100:.1f}% win probability"
            else:
                return f"Model gives {away_team} {(1-win_prob)*100:.1f}% win probability"
        
        elif bet_type == 'team_total':
            if total_mu is not None:
                return f"Game projected for {total_mu:.1f} total points"
            return f"Team total projection based on model"
        
        return f"Model recommendation based on {mode} analysis"


class BetGrader:
    """Grades completed bets against final scores."""
    
    @staticmethod
    def grade_spread(
        pick: Dict[str, Any],
        final_score_home: int,
        final_score_away: int
    ) -> str:
        """Grade a spread bet. Returns 'win', 'loss', or 'push'."""
        line = pick.get('line', 0)
        side = pick.get('side', '')
        home_team = pick.get('home_team', 'Home')
        
        margin = final_score_home - final_score_away
        
        # If picked home team
        if home_team in str(side) or 'home' in str(side).lower():
            if margin > line:
                return 'win'
            elif abs(margin - line) < 0.01:  # Floating point comparison
                return 'push'
            else:
                return 'loss'
        else:  # Picked away team
            if margin < -line:
                return 'win'
            elif abs(margin + line) < 0.01:
                return 'push'
            else:
                return 'loss'
    
    @staticmethod
    def grade_total(
        pick: Dict[str, Any],
        final_score_home: int,
        final_score_away: int
    ) -> str:
        """Grade a total bet. Returns 'win', 'loss', or 'push'."""
        line = pick.get('line', 0)
        side = pick.get('side', '')
        
        total = final_score_home + final_score_away
        
        if 'Over' in str(side):
            if total > line:
                return 'win'
            elif abs(total - line) < 0.01:
                return 'push'
            else:
                return 'loss'
        else:  # Under
            if total < line:
                return 'win'
            elif abs(total - line) < 0.01:
                return 'push'
            else:
                return 'loss'
    
    @staticmethod
    def grade_moneyline(
        pick: Dict[str, Any],
        final_score_home: int,
        final_score_away: int
    ) -> str:
        """Grade a moneyline bet. Returns 'win' or 'loss'."""
        side = pick.get('side', '')
        home_team = pick.get('home_team', 'Home')
        
        if home_team in str(side):
            return 'win' if final_score_home > final_score_away else 'loss'
        else:
            return 'win' if final_score_away > final_score_home else 'loss'
