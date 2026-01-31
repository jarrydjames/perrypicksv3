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

import pandas as pd

# Q3 model (v3 - new)
from src.modeling.q3_model import Q3Model, get_q3_model, Q3Prediction

# Feature extraction from game data (v2)
from src.predict_from_gameid_v2 import (
    first_half_score,
    behavior_counts_1h,
    team_totals_from_box_team,
    add_rate_features,
    fetch_pbp_df,
)



# Possession/PPP features
from src.features.pbp_possessions import game_possessions_first_half

# Odds fetching
from src.odds.odds_api import OddsAPIMarketSnapshot, OddsAPIError, fetch_nba_odds_snapshot
from src.odds.persistent_cache import PersistentOddsCache


# Q3 helper functions (copied from build_dataset_q3 to avoid tqdm dependency)
# Q3 helper functions (copied from build_dataset_q3 to avoid tqdm dependency)

# Q3 helper functions (copied from build_dataset_q3 to avoid tqdm dependency)
def sum_first3(periods):
    """Sum scores from periods 1-3."""
    s = 0
    for p in (periods or []):
        # Skip if p is not a dict (handles string periods, etc.)
        if not isinstance(p, dict):
            continue
        period_val = p.get("period", 0)
        # Handle various types: string ("1", "Q1"), int, float
        try:
            period_num = int(float(period_val))
        except (ValueError, TypeError):
            period_num = 0
        if 1 <= period_num <= 3:
            for key in ("score", "points", "pts"):
                if key in p and p[key] is not None:
                    try:
                        s += float(p[key])
                    except (ValueError, TypeError):
                        s += 0
                    break
    return s


def third_quarter_score(game):
    """Extract home and away scores after Q3."""
    home = game.get("homeTeam", {}) or {}
    away = game.get("awayTeam", {}) or {}
    return sum_first3(home.get("periods")), sum_first3(away.get("periods"))


def behavior_counts_q3(pbp) -> dict:
    """
    Count action types in first 3 quarters.
    
    Same structure as behavior_counts_1h, but filters to periods 1-3.
    """
    import pandas as pd
    q3 = pbp[pbp["period"].astype(int) <= 3].copy()
    at = q3.get("actionType", pd.Series([""] * len(q3))).astype(str).fillna("")
    
    def c(prefix):
        return int(at.str.startswith(prefix).sum())
    
    return {
        "q3_events": int(len(q3)),
        "q3_n_2pt": c("2pt"),
        "q3_n_3pt": c("3pt"),
        "q3_n_turnover": c("turnover"),
        "q3_n_rebound": c("rebound"),
        "q3_n_foul": c("foul"),
        "q3_n_timeout": c("timeout"),
        "q3_n_sub": c("substitution"),
    }


def extract_gid_safe(s: str) -> Optional[str]:
    """Extract GAME_ID from URL or raw ID."""
    import re
    m = re.search(r"(002\d{7})", s)
    return m.group(1) if m else None


def predict_from_game_id(
    game_input: str,
    *,
    eval_at_q3: bool = True,
    fetch_odds: bool = True,  # ← NEW: Control odds fetching
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
        from src.predict_from_gameid_v2 import fetch_box
        
        # Fetch game data for feature extraction
        game = fetch_box(game_id)
        
        # Fetch play-by-play data for Q3 feature extraction
        pbp = fetch_pbp_df(game_id)
        
        # Extract team info
        home = game.get("homeTeam", {}) or {}
        away = game.get("awayTeam", {}) or {}
        # Get tri-codes for feature extraction
        home_tri = home.get("teamTricode", "HOME")
        away_tri = away.get("teamTricode", "AWAY")
        # Get full names for odds API matching
        home_name = home.get("teamName", home.get("name", home_tri))
        away_name = away.get("teamName", away.get("name", away_tri))
        
        # Extract Q3 scores (from box score periods 1-3)
        q3_home, q3_away = third_quarter_score(game)
        
        # Extract possession/PPP features (from PBP)
        poss_features = game_possessions_first_half(pbp.to_dict("records"), home_tri=home_tri, away_tri=away_tri)
        
        # Extract team stats from box score
        ht = team_totals_from_box_team(home)
        at = team_totals_from_box_team(away)
        
        # Build features dict (all 35 features that model expects)
        features = {
            # Q3 scores
            "q3_home": q3_home,
            "q3_away": q3_away,
            "q3_total": q3_home + q3_away,
            "q3_margin": q3_home - q3_away,
        }
        
        # Add possession/PPP features (these also include rate features)
        features.update(poss_features)
        
        # Add H1 scores (for compatibility with training data)
        h1_home, h1_away = first_half_score(game)
        features["h1_home"] = h1_home
        features["h1_away"] = h1_away
        features["h1_total"] = h1_home + h1_away
        features["h1_margin"] = h1_home - h1_away
        
        # Add H1 behavior counts (for compatibility)
        beh_h1 = behavior_counts_1h(pbp)
        for k, v in beh_h1.items():
            features[k] = v
        
        # Add market line features (set to defaults if not available)
        features["market_total_line"] = 0.0
        features["market_home_spread_line"] = 0.0
        features["market_home_team_total_line"] = 0.0
        features["market_away_team_total_line"] = 0.0
        
        # Get Q3 model and predict
        q3_model = get_q3_model()
        
        # For now, assume end-of-Q3 (period=4, clock=12:00)
        # TODO: Dynamically fetch current period/clock from live data
        pred = q3_model.predict(
            features=features,  # All 35 features!
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
        from src.predict_from_gameid_v2_ci import _safe_team_name
        
        home_name = _safe_team_name(home, "Home")
        away_name = _safe_team_name(away, "Away")
        
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
    
    # Only fetch odds if requested (fetch_odds parameter)
    # Also skip odds for completed games
    if fetch_odds:
        # Get game status from prediction result to skip completed games
        # This prevents fetching odds for games that are already finished
        game_status = None
        try:
            # Try to get status from the result (set by Q3 or halftime model)
            # Both Q3 and halftime models set "status" in result
            status = result.get("status", {}) or {}
            game_status = status.get("gameStatus")
        except Exception:
            game_status = None
        
        # Only fetch odds if game is not completed
        if game_status != "Final":
            if odds is None:
                # Cache miss - fetch from API
                try:
                    odds = fetch_nba_odds_snapshot(
                        home_name=home_name,
                        away_name=away_name,
                    )
                    # Store in cache
                    cache.set(home_tri, away_tri, odds)
                except OddsAPIError as e:
                    # Odds not available (game completed, not yet scheduled, or API error)
                    # Log the error but continue with predictions
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Odds not available for {away_name} @ {home_name}: {e}")
                    odds = None
            else:
                # Cache hit but game is completed - invalidate and don't return odds
                odds = None
                result["odds_warning"] = "Game completed - odds not fetched"
    else:
        # fetch_odds=False - odds not requested, use None
        odds = None
    
    # Attach odds to result (or None if not available)
    if odds is not None:
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
        result["odds_warning"] = None
    else:
        result["odds"] = None
        result["odds_warning"] = f"Odds not available for {away_name} @ {home_name}. The game may have completed or odds are not yet posted. Predictions are still available."
    
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
