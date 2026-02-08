"""Helper functions for post generator - bet recommendations and formatting."""

from typing import Dict, Any, List, Optional

from src.betting import (
    breakeven_prob_from_american,
    prob_over_under_from_mean_sd,
    prob_spread_cover_from_mean_sd,
    edge as calc_edge,
)


def _format_probability(p: float) -> str:
    """Format probability as percentage."""
    if p is None:
        return "N/A"
    return f"{p*100:.1f}%"


def _generate_best_bets(
    prediction: Dict[str, Any],
    prediction_type: str = "halftime",  # or "q3"
    max_bets: int = 3,
    min_edge: float = 0.06,
) -> List[Dict[str, Any]]:
    """Generate top bet recommendations from prediction.
    
    Calculates probabilities and edges for totals, spreads, and moneylines.
    Returns top N bets by edge (positive edge only).
    
    Args:
        prediction: Prediction dictionary with model outputs and odds
        prediction_type: "halftime" or "q3"
        max_bets: Maximum number of bets to return
        min_edge: Minimum edge percentage (default 6%)
    
    Returns:
        List of bet dictionaries with type, side, odds, edge, probability
    """
    bets = []
    
    # Get prediction stats
    total = prediction.get("total", 0)
    margin = prediction.get("margin", 0)
    total_sd = prediction.get("total_sd", 8.0)
    margin_sd = prediction.get("margin_sd", 6.0)
    home_team = prediction.get("home_name", "Home")
    away_team = prediction.get("away_name", "Away")
    
    # Get odds
    odds_total_line = prediction.get("odds_total_line")
    odds_total_over = prediction.get("odds_total_over")
    odds_total_under = prediction.get("odds_total_under")
    odds_spread_home_line = prediction.get("odds_spread_home_line")
    odds_spread_home_odds = prediction.get("odds_spread_home")
    odds_spread_away_odds = prediction.get("odds_spread_away")
    odds_home_ml = prediction.get("odds_home_ml")
    odds_away_ml = prediction.get("odds_away_ml")
    
    # Calculate probabilities
    if isinstance(total, (int, float)) and isinstance(margin, (int, float)):
        # Total over/under
        if odds_total_line is not None and odds_total_over is not None:
            p_over = prob_over_under_from_mean_sd(total, total_sd, float(odds_total_line))
            be_over = breakeven_prob_from_american(int(odds_total_over))
            edge_over = calc_edge(p_over, be_over)
            
            if edge_over > min_edge:
                bets.append({
                    "type": "Total",
                    "side": f"Over {float(odds_total_line):.1f}",
                    "line": float(odds_total_line),
                    "odds": int(odds_total_over),
                    "probability": p_over,
                    "edge": edge_over,
                })
        
        # Spread
        if odds_spread_home_line is not None and odds_spread_home_odds is not None:
            p_home_cover = prob_spread_cover_from_mean_sd(margin, margin_sd, float(odds_spread_home_line))
            be_home = breakeven_prob_from_american(int(odds_spread_home_odds))
            edge_home = calc_edge(p_home_cover, be_home)
            
            if edge_home > min_edge:
                bets.append({
                    "type": "Spread",
                    "side": f"{home_team} {float(odds_spread_home_line):+.1f}",
                    "line": float(odds_spread_home_line),
                    "odds": int(odds_spread_home_odds),
                    "probability": p_home_cover,
                    "edge": edge_home,
                })
        
        # Moneyline (if available)
        if odds_home_ml is not None and odds_away_ml is not None:
            p_home_win = 1 - prob_spread_cover_from_mean_sd(0, margin_sd, -margin)
            be_home_ml = breakeven_prob_from_american(int(odds_home_ml))
            be_away_ml = breakeven_prob_from_american(int(odds_away_ml))
            edge_home_ml = calc_edge(p_home_win, be_home_ml)
            edge_away_ml = calc_edge(1 - p_home_win, be_away_ml)
            
            if edge_home_ml > min_edge:
                bets.append({
                    "type": "Moneyline",
                    "side": f"{home_team} ML",
                    "line": None,
                    "odds": int(odds_home_ml),
                    "probability": p_home_win,
                    "edge": edge_home_ml,
                })
            
            if edge_away_ml > min_edge:
                bets.append({
                    "type": "Moneyline",
                    "side": f"{away_team} ML",
                    "line": None,
                    "odds": int(odds_away_ml),
                    "probability": 1 - p_home_win,
                    "edge": edge_away_ml,
                })
    
    # Sort by edge and keep top max_bets
    bets.sort(key=lambda b: b["edge"], reverse=True)
    return bets[:max_bets]
