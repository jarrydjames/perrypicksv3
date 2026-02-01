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
import requests  # For HTTPError handling

# Halftime model (v2 - unchanged)
from src.predict_from_gameid_v2_ci import predict_from_game_id as predict_halftime

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

# Q3 helper functions
def sum_first3(periods):
    """Sum scores from periods 1-3."""
    s = 0
    for p in (periods or []):
        if not isinstance(p, dict):
            continue
        period_val = p.get("period", 0)
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
    """Count action types in first 3 quarters."""
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

def predict_from_game_id(game_input: str, fetch_odds: bool = True) -> Dict[str, Any]:
    """Predict game outcome from NBA.com game ID."""
    import joblib
    import logging
    logger = logging.getLogger(__name__)
    
    # Load halftime and Q3 models once
    q3_model = get_q3_model()
    
    # Determine if we should use Q3 model (has Q3 data)
    game = None
    pbp = None
    
    try:
        # Fetch game data from NBA.com
        import json
        from src.data.scoreboard import fetch_scoreboard
        
        # Try to get scoreboard data
        try:
            from src.data.game_data import fetch_game_by_id
            game = fetch_game_by_id(game_input)
            
            if not game:
                raise ValueError(f"Game not found: {game_input}")
            
            # Check if we have Q3 data (periods 1-3)
            home_periods = (game.get("homeTeam", {}) or {}).get("periods", [])
            away_periods = (game.get("awayTeam", {}) or {}).get("periods", [])
            
            has_q3_data = any(
                isinstance(p, dict) and 1 <= int(float(p.get("period", 0))) <= 3
                for p in (home_periods or []) + (away_periods or [])
            )
            
            if not has_q3_data:
                # Fall back to halftime model
                result = predict_halftime(game_input)
                result["model_used"] = "HALFTIME_NO_Q3_DATA"
                return result
            
            # Fetch PBP data for Q3 features
            try:
                pbp = fetch_pbp_df(game_input)
            except Exception as e:
                # If PBP fails, fall back to halftime model
                import logging
                logging.warning(f"PBP fetch failed for {game_id}: {e}")
                result = predict_halftime(game_input)
                result["model_used"] = "HALFTIME_PBP_ERROR"
                return result
            
            # Extract team info
            home = game.get("homeTeam", {}) or {}
            away = game.get("awayTeam", {}) or {}
            
            if not home or not away:
                raise ValueError(f"Invalid game data: Missing team information for game {game_input}")
            
            # Get tri-codes
            home_tri = home.get("teamTricode", "HOME")
            away_tri = away.get("teamTricode", "AWAY")
            
            # Get full names
            home_name = home.get("teamName", home_tri)
            away_name = away.get("teamName", away_tri)
            
            # Extract Q3 scores
            q3_home, q3_away = third_quarter_score(game)
            
            # Extract possession/PPP features
            poss_features = game_possessions_first_half(pbp.to_dict("records"), home_tri=home_tri, away_tri=away_tri)
            
            # Extract team stats from box score
            ht = team_totals_from_box_team(home)
            at = team_totals_from_box_team(away)
            
            # Build ONLY the features that models actually need
            # Don't add H1, H1 behavior, or market line features!
            q3_behavior = behavior_counts_q3(pbp)
            
            # Add team efficiency features (needed by both models)
            features = add_rate_features(ht, at, base_features={})
            
            # Add Q3-specific features
            features["q3_home"] = q3_home
            features["q3_away"] = q3_away
            features["q3_total"] = q3_home + q3_away
            features["q3_margin"] = q3_home - q3_away
            features["q3_events"] = q3_behavior["q3_events"]
            features["q3_n_2pt"] = q3_behavior["q3_n_2pt"]
            features["q3_n_3pt"] = q3_behavior["q3_n_3pt"]
            features["q3_n_turnover"] = q3_behavior["q3_n_turnover"]
            features["q3_n_rebound"] = q3_behavior["q3_n_rebound"]
            features["q3_n_foul"] = q3_behavior["q3_n_foul"]
            features["q3_n_timeout"] = q3_behavior["q3_n_timeout"]
            features["q3_n_sub"] = q3_behavior["q3_n_sub"]
            
            # Get Q3 model and predict
            q3_model = get_q3_model()
            
            # For now, assume end-of-Q3 (period=4, clock=12:00)
            pred = q3_model.predict(
                features=features,  # Only features that models need!
                period=4,
                clock="PT12M00.00S",
                game_id=game_input,
            )
            
            if pred is None:
                # Fallback to halftime if Q3 model not loaded
                result = predict_halftime(game_input)
                result["model_used"] = "HALFTIME_FALLBACK"
                return result
            
            # Build result dict (same structure as halftime)
            result = {
                "game_id": game_input,
                "home_name": home_name,
                "away_name": away_name,
                "period": 4,
                "clock": "PT12M00.00S",
                "home_score": q3_home,
                "away_score": q3_away,
                "margin": pred.margin_mean,
                "total": pred.total_mean,
                "margin_q10": pred.margin_q10,
                "margin_q90": pred.margin_q90,
                "total_q10": pred.total_q10,
                "total_q90": pred.total_q90,
                "home_win_prob": pred.home_win_prob,
                "margin_sd": pred.margin_sd,
                "total_sd": pred.total_sd,
                "model_used": "Q3",
                "model_name": pred.model_name,
                "feature_version": pred.feature_version,
            }
            
            # Fetch odds if requested
            if fetch_odds:
                try:
                    # Use persistent cache to avoid repeated API calls
                    cache = PersistentOddsCache()
                    odds_snapshot = cache.get_or_fetch(home_name, away_name)
                    
                    if odds_snapshot:
                        result.update({
                            "odds_home_ml": odds_snapshot.home_moneyline,
                            "odds_away_ml": odds_snapshot.away_moneyline,
                            "odds_total_line": odds_snapshot.total_line,
                            "odds_total_over": odds_snapshot.total_over_odds,
                            "odds_total_under": odds_snapshot.total_under_odds,
                            "odds_spread_home_line": odds_snapshot.spread_home_line,
                            "odds_spread_home": odds_snapshot.spread_home_odds,
                            "odds_spread_away": odds_snapshot.spread_away_odds,
                        })
                except OddsAPIError as e:
                    logger.warning(f"Odds API error: {e}")
                    result["odds_error"] = str(e)
            
            return result
            
        except requests.HTTPError as e:
            # If NBA.com API fails (403/429), fall back to halftime model
            logger.warning(f"NBA.com API failed for {game_id}: {e}")
            result = predict_halftime(game_input)
            result["model_used"] = "HALFTIME_API_ERROR"
            result["api_error"] = f"NBA.com API error: {e}"
            return result
        
    except Exception as e:
        # General error handling
        import traceback
        logger.error(f"Prediction failed: {repr(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
