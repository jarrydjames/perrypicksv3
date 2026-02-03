"""Pregame prediction module - predicts before game starts.

This module makes predictions using only pregame information
(team stats, form, etc.) - no game state needed.

Uses the trained PregameModel from src/modeling/pregame_model.py

Uses ONLY the features from data/processed/pregame_feature_list.txt
"""
from __future__ import annotations
import logging
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

# Import pregame model
from src.modeling.pregame_model import PregameModel, get_pregame_model

# Import nba_api for fetching team stats (no NBA.com CDN needed)
try:
    from nba_api.stats.endpoints import leaguedashteamstats, teamgamelog
except ImportError:
    logging.warning("nba_api not available, pregame predictions will have limited features")
    leaguedashteamstats = None
    teamgamelog = None

logger = logging.getLogger(__name__)

# Team ID mapping (tri-code to ID)
TEAM_IDS = {
    'ATL': 1610612737, 'BOS': 1610612738, 'BKN': 1610612751,
    'CHA': 1610612766, 'CHI': 1610612741, 'CLE': 1610612739,
    'DAL': 1610612742, 'DEN': 1610612743, 'DET': 1610612765,
    'GSW': 1610612744, 'HOU': 1610612745, 'IND': 1610612754,
    'LAC': 1610612746, 'LAL': 1610612747, 'MEM': 1610612763,
    'MIA': 1610612748, 'MIL': 1610612749, 'MIN': 1610612750,
    'NOP': 1610612752, 'NYK': 1610612753, 'OKC': 1610612760,
    'ORL': 1610612755, 'PHI': 1610612756, 'PHX': 1610612757,
    'POR': 1610612758, 'SAC': 1610612759, 'SAS': 1610612761,
    'TOR': 1610612762, 'UTA': 1610612764, 'WAS': 1610612767,
}

# Features from training data - ONLY these to avoid mismatch
# OLD pregame model expects 34 features (from pregame_feature_list.txt)
PREGAME_FEATURES = [
    'home_off_rating',
    'away_off_rating',
    'home_def_rating',
    'away_def_rating',
    'home_pace',
    'away_pace',
    'home_efg',
    'away_efg',
    'home_tov_rate',
    'away_tov_rate',
    'home_orb_rate',
    'away_orb_rate',
    'home_ft_rate',
    'away_ft_rate',
    'home_win_pct',
    'away_win_pct',
    'home_home_win_pct',
    'away_road_win_pct',
    'off_rating_diff',
    'def_rating_diff',
    'pace_diff',
    'efg_diff',
    'tov_rate_diff',
    'orb_rate_diff',
    'ft_rate_diff',
    'win_pct_diff',
    'home_off_vs_away_def',
    'away_off_vs_home_def',
    'home_court_advantage',
    'expected_pace',
    'expected_total',
    'expected_margin',
    'off_x_pace',
    'pace_diff_x_home_adv',
]

def get_team_id(tricode: str) -> Optional[int]:
    """Get team ID from tricode."""
    return TEAM_IDS.get(tricode.upper())

def fetch_team_stats(team_id: int, season: str = '2025-26') -> Optional[pd.Series]:
    """Fetch current season stats for a team (Advanced mode)."""
    if leaguedashteamstats is None:
        return None
    
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(
            team_id_nullable=team_id,
            season=season,
            measure_type_detailed_defense='Advanced',
            per_mode_detailed='PerGame',
        )
        df = stats.get_data_frames()[0]
        
        if len(df) == 0:
            logger.warning(f"No stats found for team_id {team_id}")
            return None
        
        return df.iloc[0]
    except Exception as e:
        logger.error(f"Error fetching stats for team_id {team_id}: {e}")
        return None

def fetch_recent_games(team_id: int, season: str = '2025-26', n: int = 10) -> Optional[pd.DataFrame]:
    """Fetch recent games for a team."""
    if teamgamelog is None:
        return None
    
    try:
        gamelog = teamgamelog.TeamGameLog(
            team_id=team_id,
            season=season,
        )
        df = gamelog.get_data_frames()[0]
        
        if len(df) == 0:
            logger.warning(f"No games found for team_id {team_id}")
            return None
        
        return df.head(n)
    except Exception as e:
        logger.error(f"Error fetching gamelog for team_id {team_id}: {e}")
        return None

def calculate_pregame_features(
    home_stats: Optional[pd.Series],
    away_stats: Optional[pd.Series],
) -> Dict[str, float]:
    """Calculate pregame features from team stats (NO recent games needed)."""
    features = {}
    
    # Home team stats
    if home_stats is not None:
        features['home_pace'] = home_stats.get('PACE', 100.0)
        features['home_off_rating'] = home_stats.get('OFF_RATING', 110.0)
        features['home_def_rating'] = home_stats.get('DEF_RATING', 110.0)
        features['home_efg'] = home_stats.get('EFG_PCT', 0.50)
        features['home_ft_rate'] = home_stats.get('FTA_RATE', 0.25)
        features['home_orb_rate'] = home_stats.get('OREB_PCT', 0.25)
        features['home_tov_rate'] = home_stats.get('TOV_PCT', 0.15)
        gp = home_stats.get('GP', 1.0)
        wins = home_stats.get('W', 0)
        features['home_win_pct'] = wins / gp if gp > 0 else 0.5
    
    # Away team stats
    if away_stats is not None:
        features['away_pace'] = away_stats.get('PACE', 100.0)
        features['away_off_rating'] = away_stats.get('OFF_RATING', 110.0)
        features['away_def_rating'] = away_stats.get('DEF_RATING', 110.0)
        features['away_efg'] = away_stats.get('EFG_PCT', 0.50)
        features['away_ft_rate'] = away_stats.get('FTA_RATE', 0.25)
        features['away_orb_rate'] = away_stats.get('OREB_PCT', 0.25)
        features['away_tov_rate'] = away_stats.get('TOV_PCT', 0.15)
        gp = away_stats.get('GP', 1.0)
        wins = away_stats.get('W', 0)
        features['away_win_pct'] = wins / gp if gp > 0 else 0.5
    
    # Differential features
    if 'home_pace' in features and 'away_pace' in features:
        features['pace_diff'] = features['home_pace'] - features['away_pace']
        features['expected_pace'] = (features['home_pace'] + features['away_pace']) / 2
    
    if 'home_off_rating' in features and 'away_def_rating' in features:
        features['off_rating_diff'] = features['home_off_rating'] - features['away_off_rating']
        features['def_rating_diff'] = features['home_def_rating'] - features['away_def_rating']
        features['home_off_vs_away_def'] = features['home_off_rating'] - features['away_def_rating']
        features['away_off_vs_home_def'] = features['away_off_rating'] - features['home_def_rating']
    
    if 'home_efg' in features and 'away_efg' in features:
        features['efg_diff'] = features['home_efg'] - features['away_efg']
    
    if 'home_tov_rate' in features and 'away_tov_rate' in features:
        features['tov_rate_diff'] = features['home_tov_rate'] - features['away_tov_rate']
    
    if 'home_orb_rate' in features and 'away_orb_rate' in features:
        features['orb_rate_diff'] = features['home_orb_rate'] - features['away_orb_rate']
    
    if 'home_ft_rate' in features and 'away_ft_rate' in features:
        features['ft_rate_diff'] = features['home_ft_rate'] - features['away_ft_rate']
    
    if 'home_win_pct' in features and 'away_win_pct' in features:
        features['win_pct_diff'] = features['home_win_pct'] - features['away_win_pct']
    
    # Expected values
    if 'home_off_rating' in features and 'away_off_rating' in features:
        avg_off = (features['home_off_rating'] + features['away_off_rating']) / 2
        if 'expected_pace' in features:
            features['expected_total'] = avg_off * features['expected_pace'] / 100
        else:
            features['expected_total'] = avg_off * 2.15  # Default pace
    
    if 'home_off_rating' in features and 'away_def_rating' in features:
        features['off_x_pace'] = features['home_off_rating'] * features.get('home_pace', 100) / 100
        features['home_court_advantage'] = 3.0  # 3 pt home court adv
        features['pace_diff_x_home_adv'] = features.get('pace_diff', 0) + 3.0  # Home court advantage
        features['expected_margin'] = (features['home_off_rating'] - features['away_def_rating']) - \
                                        (features['away_off_rating'] - features['home_def_rating']) / 2 + \
                                        3.0  # Home court advantage
    
    return features

def predict_from_game_id(
    game_id: str,
    home_team: str,
    away_team: str,
    fetch_odds: bool = False,
    season: str = '2025-26',
) -> Dict[str, Any]:
    """
    Predict game outcome before it starts using pregame model.
    
    Args:
        game_id: NBA.com game ID
        home_team: Home team tricode
        away_team: Away team tricode
        fetch_odds: Whether to fetch odds from API
        season: NBA season (default: 2025-26)
    
    Returns:
        Dict with prediction results (same format as halftime/Q3)
    """
    logger.info(f"Running pregame prediction for {away_team} @ {home_team} ({game_id})")
    
    # Get team IDs
    home_id = get_team_id(home_team)
    away_id = get_team_id(away_team)
    
    if home_id is None:
        logger.error(f"Unknown home team tricode: {home_team}")
        return {
            "status": "error",
            "error": f"Unknown home team tricode: {home_team}",
            "game_id": game_id,
            "model_used": "ERROR",
        }
    
    if away_id is None:
        logger.error(f"Unknown away team tricode: {away_team}")
        return {
            "status": "error",
            "error": f"Unknown away team tricode: {away_team}",
            "game_id": game_id,
            "model_used": "ERROR",
        }
    
    # Fetch team stats
    home_stats = fetch_team_stats(home_id, season)
    away_stats = fetch_team_stats(away_id, season)
    
    # Calculate features (NO recent games - use only team stats)
    features = calculate_pregame_features(home_stats, away_stats)
    
    # Keep ONLY the features in the feature list
    filtered_features = {k: v for k, v in features.items() if k in PREGAME_FEATURES}
    logger.info(f"Calculated {len(filtered_features)} pregame features (filtered from {len(features)} total)")
    
    # Load pregame model and predict
    model = get_pregame_model()
    
    if model is None:
        logger.error("Pregame model not available")
        return {
            "status": "error",
            "error": "Pregame model not available. Please train model first.",
            "game_id": game_id,
            "model_used": "ERROR",
        }
    
    # Make prediction
    try:
        pred = model.predict(features=filtered_features, game_id=game_id)
    except ValueError as e:
        logger.error(f"Prediction error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "game_id": game_id,
            "home_name": home_team,  # Required by predict_api
            "away_name": away_team,   # Required by predict_api
            "margin": None,               # Required by predict_api
            "total": None,                # Required by predict_api
            "model_used": "ERROR",
        }
    
    if pred is None:
        logger.error("Pregame prediction failed")
        return {
            "status": "error",
            "error": "Pregame prediction failed",
            "game_id": game_id,
            "model_used": "ERROR",
        }
    
    # Get team names (use tricodes if full names not available)
    home_name = home_team  # Could expand to full names later
    away_name = away_team
    
    # Build result dict (same format as halftime/Q3)
    result = {
        "game_id": game_id,
        "home_name": home_name,
        "away_name": away_name,
        "home_team_id": home_id,
        "away_team_id": away_id,
        "period": 0,  # Pregame
        "clock": None,
        "home_score": None,
        "away_score": None,
        "margin": pred.margin_mean,
        "total": pred.total_mean,
        "margin_q10": pred.margin_q10,
        "margin_q90": pred.margin_q90,
        "total_q10": pred.total_q10,
        "total_q90": pred.total_q90,
        "home_win_prob": pred.home_win_prob,
        "margin_sd": pred.margin_sd,
        "total_sd": pred.total_sd,
        "model_used": "PREGAME",
        "model_name": pred.model_name,
        "feature_version": pred.feature_version,
        "status": "success",
    }
    
    # Fetch odds if requested
    if fetch_odds:
        try:
            from src.odds.odds_api import fetch_nba_odds_snapshot, OddsAPIError
            odds_snapshot = fetch_nba_odds_snapshot(home_name, away_name)
            
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
        except Exception as e:
            logger.warning(f"Odds API error: {e}")
            result["odds_error"] = str(e)
    
    logger.info(f"Pregame prediction complete: total={pred.total_mean:.1f}, margin={pred.margin_mean:.1f}")
    
    return result
