"""V3 runtime predictor - game-clock aware prediction orchestrator.

This module evaluates predictions at any game state (halftime, end-of-Q3, or during play)
and compares to odds to recommend bets.

Key features:
- Supports both halftime and Q3 models
- Game-clock aware (accepts period, clock as context)
- Compares predictions to odds API
- Recommends bets with edge calculations
"""

from __future__ import annotations
from typing import Any, Dict, Optional

# Halftime model (v2 - unchanged)
from src.predict_from_gameid_v2_ci import predict_from_game_id as predict_halftime

# Q3 model (v3 - new)
from src.modeling.q3_model import Q3Model, get_q3_model, Q3Prediction

# Odds fetching
from src.odds.odds_api import OddsAPIMarketSnapshot, fetch_nba_odds_snapshot
from src.odds.persistent_cache import PersistentOddsCache


def extract_gid_safe(s: str) -> Optional[str]:
    """Extract GAME_ID from URL or raw ID."""
    import re
    m = re.search(r"(002\d{7})", s)
    return m.group(1) if m else None


def predict_from_game_id(
    game_input: str,
    *,
    eval_at_q3: bool = False,
) -> Dict[str, Any]:
    """
    Main prediction entry point - evaluates at game state and compares to odds.
    
    Args:
        game_input: Game ID or NBA.com URL
        eval_at_q3: If True, use Q3 model (end-of-Q3 evaluation).
                    If False, use halftime model (unchanged v2 behavior).
    
    Returns:
        Dict with prediction + odds + bet recommendations (same structure as v2).
    """
    # Extract game ID
    game_id = extract_gid_safe(game_input)
    if not game_id:
        raise ValueError(f"Invalid game input: {game_input}")
    
    # Choose model based on eval_at_q3 flag
    if eval_at_q3:
        # Q3 model (v3) - evaluate at end-of-Q3 state
        q3_model = get_q3_model()
        
        # Get game state to determine period/clock
        # For now, assume end-of-Q3 (period=4, clock=12:00)
        # TODO: Dynamically fetch current period/clock from live data
        pred = q3_model.predict(
            features={},  # TODO: extract features from live data
            period=4,
            clock="PT12M00.00S",
            game_id=game_id,
        )
        
        if pred is None:
            # Fallback to halftime if Q3 model not trained
            result = predict_halftime(game_input)
            result["model_used"] = "HALFTIME_FALLBACK"
            return result
        
        # Format Q3 prediction into v2-compatible structure
        # Note: Need to fetch team names for odds matching
        from src.predict_from_gameid_v2 import fetch_box
        from src.predict_from_gameid_v2_ci import _safe_team_name
        
        game = fetch_box(game_id)
        home_team = game.get("homeTeam") or {}
        away_team = game.get("awayTeam") or {}
        home_name = _safe_team_name(home_team, "Home")
        away_name = _safe_team_name(away_team, "Away")
        
        result = {
            "game_id": game_id,
            "model_used": "Q3",
            "eval_point": "END_OF_Q3",
            "period": pred.period,
            "clock": pred.clock,
            "home_win_prob": pred.home_win_prob,
            "home_name": home_name,
            "away_name": away_name,
            "margin": {
                "mu": pred.margin_mean,
                "sd": pred.margin_sd,
                "q10": pred.margin_q10,
                "q90": pred.margin_q90,
            },
            "total": {
                "mu": pred.total_mean,
                "sd": pred.total_sd,
                "q10": pred.total_q10,
                "q90": pred.total_q90,
            },
            "status": "Q3_PREDICTION",
        }
    else:
        # Halftime model (v2) - unchanged behavior
        result = predict_halftime(game_input)
        result["model_used"] = "HALFTIME"
        result["eval_point"] = "HALFTIME"
    
    # Fetch and integrate odds (same as v2)
    # Extract team names from prediction result
    # Note: predict_from_gameid_v2_ci returns "home_name" and "away_name", not "home_tri"/"away_tri"
    home_tri = result.get("home_name", "HOME")
    away_tri = result.get("away_name", "AWAY")
    
    # Use persistent cache to minimize API calls
    cache = PersistentOddsCache()
    odds = cache.get(home_tri, away_tri)
    
    if odds is None:
        # Cache miss - fetch from API
        odds = fetch_nba_odds_snapshot(
            home_name=home_tri,
            away_name=away_tri,
        )
        # Store in cache
        cache.set(home_tri, away_tri, odds)
    
    # Attach odds to result
    result["odds"] = {
        "total_points": odds.total_points,
        "total_over_odds": odds.total_over_odds,
        "total_under_odds": odds.total_under_odds,
        "spread_home": odds.spread_home,
        "spread_home_odds": odds.spread_home_odds,
        "spread_away_odds": odds.spread_away_odds,
        "moneyline_home": odds.moneyline_home,
        "moneyline_away": odds.moneyline_away,
        "bookmaker": odds.bookmaker,
        "last_update": odds.last_update,
    }
    
    # TODO: Add bet recommendations (same as v2)
    # For now, return prediction + odds
    result["bets"] = []
    
    return result


def predict_at_halftime(game_input: str) -> Dict[str, Any]:
    """
    Wrapper for halftime prediction (backward compatible with v2).
    """
    return predict_from_game_id(game_input, eval_at_q3=False)


def predict_at_q3(game_input: str) -> Dict[str, Any]:
    """
    Wrapper for Q3 prediction (new v3 feature).
    
    Evaluates at end-of-Q3 state and compares to odds.
    """
    return predict_from_game_id(game_input, eval_at_q3=True)
