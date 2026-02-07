"""Pregame prediction - uses FINAL models with 72 features

This is the CORRECT pregame prediction system you built:
- 72 features including temporal data, form data, H2H stats, schedule strength
- Trained on 3,390 games
- Best model: Ridge (Test MAE: 15.6 for total, 11.2 for margin)

For immediate implementation, we'll use simplified feature extraction
focused on core predictive features while we build full temporal features.
"""
from __future__ import annotations
import logging
import os
from typing import Any, Dict, Optional, Sequence, Tuple
from datetime import datetime
import numpy as np
import pandas as pd

from src.modeling.pregame_model import PregameModel, get_pregame_model
from src.data.historical_data import get_historical_data_manager, TRICODE_TO_TEAM_ID

# Import nba_api for fetching team stats
try:
    from nba_api.stats.endpoints import leaguedashteamstats
except ImportError:
    logging.warning("nba_api not available, pregame predictions will have limited features")
    leaguedashteamstats = None

logger = logging.getLogger(__name__)

# Team ID mapping
TEAM_IDS = {
    'ATL': 1610612737, 'BOS': 1610612738, 'BKN': 1610612751,
    'CHA': 1610612766, 'CHI': 1610612741, 'CLE': 1610612739,
    'DAL': 1610612742, 'DEN': 1610612743, 'DET': 1610612765,
    'GSW': 1610612744, 'HOU': 1610612745, 'IND': 1610612754,
    'LAC': 1610612746, 'LAL': 1610612747, 'MEM': 1610612763,
    'MIA': 1610612748, 'MIL': 1610612749, 'MIN': 1610612750,
    'NOP': 1610612740, 'NYK': 1610612752, 'OKC': 1610612760,
    'ORL': 1610612753, 'PHI': 1610612755, 'PHX': 1610612756,
    'POR': 1610612757, 'SAC': 1610612758, 'SAS': 1610612759,
    'TOR': 1610612761, 'UTA': 1610612762, 'WAS': 1610612764,
}

def get_team_id(tricode: str) -> Optional[int]:
    """Get team ID from tricode."""
    return TEAM_IDS.get(tricode.upper())


def infer_season_from_game_id(game_id: str) -> Optional[str]:
    """Infer NBA season string (e.g. 2025-26) from game_id prefix 002YYxxxxx."""
    gid = str(game_id)
    if len(gid) < 5 or not gid[3:5].isdigit():
        return None

    season_start_yy = int(gid[3:5])
    season_start = 2000 + season_start_yy
    season_end_yy = (season_start_yy + 1) % 100
    return f"{season_start}-{season_end_yy:02d}"


def infer_season_from_datetime(game_datetime: pd.Timestamp) -> str:
    """Infer NBA season string from game datetime (season starts in October)."""
    ts = pd.Timestamp(game_datetime)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    season_start = ts.year if ts.month >= 10 else ts.year - 1
    return f"{season_start}-{(season_start + 1) % 100:02d}"

def _previous_season(season: str) -> Optional[str]:
    """Return previous NBA season string from current season string."""
    try:
        start_year = int(season.split('-')[0])
        prev_start = start_year - 1
        return f"{prev_start}-{(prev_start + 1) % 100:02d}"
    except Exception:
        return None


def fetch_team_stats(
    team_id: int,
    seasons: Optional[Sequence[str]] = None,
) -> Tuple[Optional[pd.Series], Optional[str]]:
    """Fetch team stats with multi-season fallback.

    Returns:
        Tuple of (team stats row, season string used).
    """
    if leaguedashteamstats is None:
        return None, None

    seasons_to_try = list(seasons or ['2025-26', '2024-25'])
    for season in seasons_to_try:
        try:
            stats = leaguedashteamstats.LeagueDashTeamStats(
                team_id_nullable=team_id,
                season=season,
                measure_type_detailed_defense='Advanced',
                per_mode_detailed='PerGame',
            )
            df = stats.get_data_frames()[0]

            if len(df) == 0:
                logger.warning("No stats found for team_id %s in season %s", team_id, season)
                continue

            # Some API responses can return the full league table even when
            # `team_id_nullable` is passed. Always select by TEAM_ID when present
            # so each team gets its own feature row.
            if 'TEAM_ID' in df.columns:
                team_rows = df[df['TEAM_ID'] == team_id]
                if len(team_rows) > 0:
                    return team_rows.iloc[0], season

                logger.warning(
                    "TEAM_ID %s not found in fetched stats payload for season %s; trying next season",
                    team_id,
                    season,
                )
                continue

            if len(df) == 1:
                logger.warning(
                    "TEAM_ID column missing for team_id %s in season %s; using single-row response",
                    team_id,
                    season,
                )
                return df.iloc[0], season

            logger.warning(
                "TEAM_ID column missing and multiple rows returned for team_id %s in season %s; trying next season",
                team_id,
                season,
            )
            continue
        except Exception as e:
            logger.error("Error fetching stats for team_id %s in season %s: %s", team_id, season, e)

    return None, None


def are_features_all_defaults(features: Dict[str, float]) -> bool:
    """Detect when both teams are effectively using default placeholder values."""
    checks = [
        np.isclose(features.get('home_off_rating', -1.0), 110.0),
        np.isclose(features.get('away_off_rating', -1.0), 110.0),
        np.isclose(features.get('home_def_rating', -1.0), 110.0),
        np.isclose(features.get('away_def_rating', -1.0), 110.0),
        np.isclose(features.get('home_pace', -1.0), 100.0),
        np.isclose(features.get('away_pace', -1.0), 100.0),
        np.isclose(features.get('off_rating_diff', 999.0), 0.0),
        np.isclose(features.get('def_rating_diff', 999.0), 0.0),
        np.isclose(features.get('pace_diff', 999.0), 0.0),
    ]
    return sum(bool(c) for c in checks) >= 8

def _safe_days_between(game_datetime: pd.Timestamp, reference_datetime: Optional[pd.Timestamp]) -> Optional[int]:
    if reference_datetime is None:
        return None
    try:
        gdt = pd.Timestamp(game_datetime)
        rdt = pd.Timestamp(reference_datetime)
        if gdt.tzinfo is None:
            gdt = gdt.tz_localize("UTC")
        else:
            gdt = gdt.tz_convert("UTC")
        if rdt.tzinfo is None:
            rdt = rdt.tz_localize("UTC")
        else:
            rdt = rdt.tz_convert("UTC")
        return max(int((gdt - rdt).days), 0)
    except Exception:
        return None


def build_data_freshness_context(
    game_datetime: pd.Timestamp,
    home_team_id: int,
    away_team_id: int,
    max_stale_days: int = 3,
) -> Dict[str, Any]:
    """Build freshness metadata and stale flags from historical game data."""
    context: Dict[str, Any] = {
        "is_stale": False,
        "max_stale_days": max_stale_days,
        "historical_latest_game_date": None,
        "days_since_historical_update": None,
        "home_days_since_last_game": None,
        "away_days_since_last_game": None,
    }

    hist_mgr = get_historical_data_manager()
    if not hist_mgr:
        return context

    latest_game_date: Optional[pd.Timestamp] = None
    if getattr(hist_mgr, "games_df", None) is not None and len(hist_mgr.games_df) > 0:  # type: ignore[attr-defined]
        latest_game_date = pd.Timestamp(hist_mgr.games_df["game_date"].max())  # type: ignore[index]
        context["historical_latest_game_date"] = latest_game_date.isoformat()
        context["days_since_historical_update"] = _safe_days_between(game_datetime, latest_game_date)

    home_recent = hist_mgr.get_team_games(home_team_id, before_date=game_datetime, n=1)
    away_recent = hist_mgr.get_team_games(away_team_id, before_date=game_datetime, n=1)

    if len(home_recent) > 0:
        home_last = pd.Timestamp(home_recent.iloc[0]["game_date"])
        context["home_days_since_last_game"] = _safe_days_between(game_datetime, home_last)
    if len(away_recent) > 0:
        away_last = pd.Timestamp(away_recent.iloc[0]["game_date"])
        context["away_days_since_last_game"] = _safe_days_between(game_datetime, away_last)

    global_gap = context.get("days_since_historical_update")
    home_gap = context.get("home_days_since_last_game")
    away_gap = context.get("away_days_since_last_game")

    stale_reasons: list[str] = []
    if isinstance(global_gap, int) and global_gap > max_stale_days:
        stale_reasons.append(f"historical data is {global_gap} days old")
    if isinstance(home_gap, int) and home_gap > max_stale_days:
        stale_reasons.append(f"home team has {home_gap} days since last game")
    if isinstance(away_gap, int) and away_gap > max_stale_days:
        stale_reasons.append(f"away team has {away_gap} days since last game")

    context["is_stale"] = len(stale_reasons) > 0
    context["stale_reasons"] = stale_reasons
    context["staleness_policy"] = "warn_only"
    return context


def extract_core_features(
    home_stats: Optional[pd.Series],
    away_stats: Optional[pd.Series],
    home_team_id: int,
    away_team_id: int,
    game_date: datetime,
) -> Dict[str, float]:
    """
    Extract core pregame features from team stats + historical data.
    
    This provides ALL 72 features:
    - Basic team ratings (18 features): off/def rating, pace, efg, tov/orb/ft rate, win pct
    - Schedule features (8 features): rest days, back-to-back
    - Recent form features (11 features): recent points/allowed/margin/wins (last 10 games)
    - Four factors / Net rating (20 features): net rating, TS proxy, four factor weighted
    - Head-to-head features (13 features): H2H wins, total games, win pct, recent H2H
    - Schedule strength features (2 features): opponent strength
    """
    features = {}
    hist_mgr = get_historical_data_manager()
    
    # Helper function to map NBA API column names to feature names
    def map_api_columns(stats_row: Optional[pd.Series]) -> Dict[str, float]:
        """Map NBA API column names to expected feature names.
        
        NBA API returns TM_TOV_PCT but code expects TOV_PCT.
        This helper handles the column name mapping."""
        if stats_row is None:
            return {}
        
        mapping = {
            'OFF_RATING': 'off_rating',
            'DEF_RATING': 'def_rating',
            'PACE': 'pace',
            'EFG_PCT': 'efg',
            'TM_TOV_PCT': 'tov_rate',  # NBA API returns TM_TOV_PCT
            'OREB_PCT': 'orb_rate',
        }
        
        result = {}
        for api_col, feat_name in mapping.items():
            if api_col in stats_row.index:
                result[feat_name] = float(stats_row[api_col])
        return result
    
    # ===== BASIC TEAM RATINGS (18 features) =====
    # Map NBA API column names to feature names for both teams
    home_mapped = map_api_columns(home_stats)
    away_mapped = map_api_columns(away_stats)
    
    # Use current season stats if available, otherwise use historical averages
    if home_stats is not None:
        features['home_off_rating'] = home_mapped.get('off_rating', 110.0)
        features['home_def_rating'] = home_mapped.get('def_rating', 110.0)
        features['home_pace'] = home_mapped.get('pace', 100.0)
        features['home_efg'] = home_mapped.get('efg', 0.50)
        features['home_tov_rate'] = home_mapped.get('tov_rate', 0.15)
        features['home_orb_rate'] = home_mapped.get('orb_rate', 0.25)
        gp = home_stats.get('GP', 1.0)
        wins = home_stats.get('W', 0)
        features['home_win_pct'] = wins / gp if gp > 0 else 0.5
    elif hist_mgr and len(hist_mgr.get_team_games(home_team_id, before_date=game_date, n=20)) > 0:
        # Use historical averages
        home_hist = hist_mgr.get_team_games(home_team_id, before_date=game_date, n=20)
        features['home_off_rating'] = float(home_hist['home_off_rating'].mean()) if 'home_off_rating' in home_hist else 110.0
        features['home_def_rating'] = float(home_hist['home_def_rating'].mean()) if 'home_def_rating' in home_hist else 110.0
        features['home_pace'] = float(home_hist['home_pace'].mean()) if 'home_pace' in home_hist else 100.0
        features['home_efg'] = float(home_hist['home_efg'].mean()) if 'home_efg' in home_hist else 0.50
        features['home_tov_rate'] = float(home_hist['home_tov_rate'].mean()) if 'home_tov_rate' in home_hist else 0.15
        features['home_orb_rate'] = float(home_hist['home_orb_rate'].mean()) if 'home_orb_rate' in home_hist else 0.25
        features['home_win_pct'] = float(home_hist['home_win_pct'].mean()) if 'home_win_pct' in home_hist else 0.5
    else:
        # Default values if stats unavailable
        for feat in ['off_rating', 'def_rating', 'pace', 'efg', 'tov_rate', 'orb_rate', 'win_pct']:
            features[f'home_{feat}'] = 110.0 if feat in ['off_rating', 'def_rating'] else (100.0 if feat == 'pace' else 0.5 if feat == 'efg' else 0.25)
        features['home_win_pct'] = 0.5
    
    # Add ft_rate defaults (not available in NBA API Advanced measure type)
    if 'home_ft_rate' not in features:
        features['home_ft_rate'] = 0.25
    if 'away_ft_rate' not in features:
        features['away_ft_rate'] = 0.25
    
    # Away team stats (using mapped columns)
    if away_stats is not None:
        features['away_off_rating'] = away_mapped.get('off_rating', 110.0)
        features['away_def_rating'] = away_mapped.get('def_rating', 110.0)
        features['away_pace'] = away_mapped.get('pace', 100.0)
        features['away_efg'] = away_mapped.get('efg', 0.50)
        features['away_tov_rate'] = away_mapped.get('tov_rate', 0.15)
        features['away_orb_rate'] = away_mapped.get('orb_rate', 0.25)
        gp = away_stats.get('GP', 1.0)
        wins = away_stats.get('W', 0)
        features['away_win_pct'] = wins / gp if gp > 0 else 0.5
    elif hist_mgr and len(hist_mgr.get_team_games(away_team_id, before_date=game_date, n=20)) > 0:
        # Use historical averages
        away_hist = hist_mgr.get_team_games(away_team_id, before_date=game_date, n=20)
        features['away_off_rating'] = float(away_hist['away_off_rating'].mean()) if 'away_off_rating' in away_hist else 110.0
        features['away_def_rating'] = float(away_hist['away_def_rating'].mean()) if 'away_def_rating' in away_hist else 110.0
        features['away_pace'] = float(away_hist['away_pace'].mean()) if 'away_pace' in away_hist else 100.0
        features['away_efg'] = float(away_hist['away_efg'].mean()) if 'away_efg' in away_hist else 0.50
        features['away_tov_rate'] = float(away_hist['away_tov_rate'].mean()) if 'away_tov_rate' in away_hist else 0.15
        features['away_orb_rate'] = float(away_hist['away_orb_rate'].mean()) if 'away_orb_rate' in away_hist else 0.25
        features['away_win_pct'] = float(away_hist['away_win_pct'].mean()) if 'away_win_pct' in away_hist else 0.5
    else:
        for feat in ['off_rating', 'def_rating', 'pace', 'efg', 'ft_rate', 'tov_rate', 'orb_rate', 'win_pct']:
            features[f'away_{feat}'] = 110.0 if feat in ['off_rating', 'def_rating'] else (100.0 if feat == 'pace' else 0.5 if feat == 'efg' else 0.25)
        features['away_win_pct'] = 0.5
    
    # Add ft_rate defaults (not available in NBA API Advanced measure type)
    if 'home_ft_rate' not in features:
        features['home_ft_rate'] = 0.25
    if 'away_ft_rate' not in features:
        features['away_ft_rate'] = 0.25
    
    # ===== SCHEDULE FEATURES (8 features) =====
    # Use historical data for rest days and back-to-back
    if hist_mgr:
        schedule_features = hist_mgr.calculate_schedule_features(home_team_id, away_team_id, game_date)
        features.update(schedule_features)
    else:
        features['home_rest_days'] = 7.0
        features['away_rest_days'] = 7.0
        features['rest_days_diff'] = 0.0
        features['home_is_b2b'] = 0.0
        features['away_is_b2b'] = 0.0
        features['home_b2b_x_home'] = 0.0
        features['away_b2b_x_away'] = 0.0
        features['b2b_diff'] = 0.0
    
    # ===== RECENT FORM FEATURES (11 features) =====
    # Use historical data for recent form
    if hist_mgr:
        recent_features = hist_mgr.calculate_recent_form(home_team_id, away_team_id, game_date)
        features.update(recent_features)
    else:
        # Simplified proxies
        features['home_recent_points'] = features['home_off_rating'] * 1.15
        features['away_recent_points'] = features['away_off_rating'] * 1.15
        features['home_recent_allowed'] = features['home_def_rating'] * 1.15
        features['away_recent_allowed'] = features['away_def_rating'] * 1.15
        features['home_recent_margin'] = (features['home_off_rating'] - features['home_def_rating']) * 1.0
        features['away_recent_margin'] = (features['away_off_rating'] - features['away_def_rating']) * 1.0
        features['home_recent_wins'] = features['home_win_pct']
        features['away_recent_wins'] = features['away_win_pct']
        features['recent_points_diff'] = features['home_recent_points'] - features['away_recent_points']
        features['recent_allowed_diff'] = features['home_recent_allowed'] - features['away_recent_allowed']
        features['recent_margin_diff'] = features['home_recent_margin'] - features['away_recent_margin']
        features['recent_wins_diff'] = features['home_recent_wins'] - features['away_recent_wins']
    
    # Net rating features
    features['home_net_rating'] = features['home_off_rating'] - features['home_def_rating']
    features['away_net_rating'] = features['away_off_rating'] - features['away_def_rating']
    features['net_rating_diff'] = features['home_net_rating'] - features['away_net_rating']
    
    # TS proxy and assist ratio (simplified)
    features['home_ts_proxy'] = features['home_efg'] * features['home_ft_rate']
    features['away_ts_proxy'] = features['away_efg'] * features['away_ft_rate']
    features['ts_proxy_diff'] = features['home_ts_proxy'] - features['away_ts_proxy']
    features['home_assist_ratio_proxy'] = features['home_pace'] / 100.0
    features['away_assist_ratio_proxy'] = features['away_pace'] / 100.0
    features['assist_ratio_diff'] = features['home_assist_ratio_proxy'] - features['away_assist_ratio_proxy']
    
    # Four factor weighted (simplified)
    features['home_four_factor_weighted'] = (
        features['home_efg'] * 0.4 +
        features['home_orb_rate'] * 0.3 +
        features['home_tov_rate'] * -0.15 +
        features['home_ft_rate'] * 0.15
    )
    features['away_four_factor_weighted'] = (
        features['away_efg'] * 0.4 +
        features['away_orb_rate'] * 0.3 +
        features['away_tov_rate'] * -0.15 +
        features['away_ft_rate'] * 0.15
    )
    features['four_factor_weighted_diff'] = features['home_four_factor_weighted'] - features['away_four_factor_weighted']
    
    # Differentials (calculated earlier)
    if 'home_off_rating' in features and 'away_off_rating' in features:
        features['off_rating_diff'] = features['home_off_rating'] - features['away_off_rating']
        features['def_rating_diff'] = features['home_def_rating'] - features['away_def_rating']
        features['pace_diff'] = features['home_pace'] - features['away_pace']
        features['efg_diff'] = features['home_efg'] - features['away_efg']
        features['tov_rate_diff'] = features['home_tov_rate'] - features['away_tov_rate']
        features['orb_rate_diff'] = features['home_orb_rate'] - features['away_orb_rate']
        features['ft_rate_diff'] = features['home_ft_rate'] - features['away_ft_rate']
    
    # Home/Road splits (simplified)
    features['home_home_win_pct'] = features['home_win_pct'] * 1.03  # Home court advantage
    features['away_road_win_pct'] = features['away_win_pct'] * 0.97
    
    # Efficiency scores
    features['home_efficiency_score'] = features['home_net_rating']
    features['away_efficiency_score'] = features['away_net_rating']
    features['efficiency_diff'] = features['home_efficiency_score'] - features['away_efficiency_score']
    
    # ===== HEAD-TO-HEAD FEATURES (13 features) =====
    # Use historical data for H2H lookup
    if hist_mgr:
        h2h_features = hist_mgr.calculate_h2h_features(home_team_id, away_team_id, game_date)
        features.update(h2h_features)
    else:
        features['h2h_home_wins'] = 5.0
        features['h2h_away_wins'] = 5.0
        features['h2h_total_games'] = 10.0
        features['h2h_home_win_pct'] = 0.5
        features['h2h_recent_home_wins'] = 2.0
        features['h2h_recent_away_wins'] = 2.0
        features['h2h_recent_total'] = 5.0
        features['h2h_recent_home_win_pct'] = 0.5
        features['h2h_wins_diff'] = 0.0
        features['h2h_win_pct_diff'] = 0.0
        features['h2h_recent_wins_diff'] = 0.0
        features['h2h_recent_win_pct_diff'] = 0.0
    
    # ===== SCHEDULE STRENGTH FEATURES (2 features) =====
    # Use historical data for schedule strength
    if hist_mgr:
        ss_features = hist_mgr.calculate_schedule_strength(home_team_id, away_team_id, game_date)
        features.update(ss_features)
    else:
        features['home_schedule_strength'] = 0.0
        features['away_schedule_strength'] = 0.0
        features['schedule_strength_diff'] = 0.0
    
    return features


def predict_from_game_id(
    game_id: str,
    home_team: str,
    away_team: str,
    fetch_odds: bool = False,
    season: Optional[str] = None,
    game_datetime: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Predict game outcome before it starts using pregame model.
    
    Uses FINAL models trained on 72 features:
    - Basic team ratings (18 features)
    - Schedule features (8 features)  
    - Recent form features (11 features)
    - Four factors / Net rating (20 features)
    - Head-to-head features (13 features)
    - Schedule strength (2 features)
    
    Args:
        game_id: NBA.com game ID
        home_team: Home team tricode
        away_team: Away team tricode
        fetch_odds: Whether to fetch odds from API
        season: NBA season (if omitted, inferred from game_id / game datetime)
        game_datetime: Scheduled game time (if available) for temporal features
    
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
            "home_name": home_team,
            "away_name": away_team,
            "margin": None,
            "total": None,
            "model_used": "ERROR",
        }
    
    if away_id is None:
        logger.error(f"Unknown away team tricode: {away_team}")
        return {
            "status": "error",
            "error": f"Unknown away team tricode: {away_team}",
            "game_id": game_id,
            "home_name": home_team,
            "away_name": away_team,
            "margin": None,
            "total": None,
            "model_used": "ERROR",
        }
    
    # Resolve game datetime
    if game_datetime is not None:
        resolved_game_datetime = pd.Timestamp(game_datetime)
        if resolved_game_datetime.tzinfo is None:
            resolved_game_datetime = resolved_game_datetime.tz_localize('UTC')
        else:
            resolved_game_datetime = resolved_game_datetime.tz_convert('UTC')
    else:
        # Fallback when schedule date is not available
        resolved_game_datetime = pd.Timestamp.now('UTC') - pd.Timedelta(days=1)

    # Resolve season
    resolved_season = season or infer_season_from_game_id(game_id) or infer_season_from_datetime(resolved_game_datetime)

    # Build freshness context from historical data
    max_stale_days = int(os.getenv("PREGAME_MAX_STALE_DAYS", "3"))
    freshness = build_data_freshness_context(
        resolved_game_datetime,
        home_id,
        away_id,
        max_stale_days=max_stale_days,
    )

    # Fetch team stats with fallback season
    fallback_season = _previous_season(resolved_season)
    seasons_to_try = [resolved_season]
    if fallback_season and fallback_season != resolved_season:
        seasons_to_try.append(fallback_season)

    logger.info("Using seasons=%s for pregame game_id=%s", seasons_to_try, game_id)
    home_stats, home_stats_season = fetch_team_stats(home_id, seasons_to_try)
    away_stats, away_stats_season = fetch_team_stats(away_id, seasons_to_try)

    # Extract features with NBA API data (when available)
    # Note: We always use NBA API data if available, regardless of historical staleness.
    # Historical data is still used for schedule/form/H2H features which have no NBA API alternative.
    features = extract_core_features(
        home_stats,
        away_stats,
        home_id,
        away_id,
        resolved_game_datetime,
    )
    
    logger.info(f"Extracted {len(features)} features for prediction")
    
    # Load pregame model and predict
    model = get_pregame_model()
    
    if model is None:
        logger.error("Pregame model not available")
        return {
            "status": "error",
            "error": "Pregame model not available. Please train model first.",
            "game_id": game_id,
            "home_name": home_team,
            "away_name": away_team,
            "margin": None,
            "total": None,
            "model_used": "ERROR",
        }
    
    # Make prediction
    try:
        pred = model.predict(features=features, game_id=game_id)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "game_id": game_id,
            "home_name": home_team,
            "away_name": away_team,
            "margin": None,
            "total": None,
            "model_used": "ERROR",
        }
    
    if pred is None:
        logger.error("Pregame prediction failed")
        return {
            "status": "error",
            "error": "Pregame prediction failed",
            "game_id": game_id,
            "home_name": home_team,
            "away_name": away_team,
            "margin": None,
            "total": None,
            "model_used": "ERROR",
        }
    
    used_defaults = (home_stats is None or away_stats is None) and are_features_all_defaults(features)
    stale_data = bool(freshness.get("is_stale"))

    # Build result dict (same format as halftime/Q3)
    result = {
        "game_id": game_id,
        "home_name": home_team,
        "away_name": away_team,
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
        "model_used": "PREGAME_V3_FINAL",
        "model_name": pred.model_name,
        "feature_version": pred.feature_version,
        "status": "warning" if (used_defaults or stale_data) else "success",
        "data_source": {
            "home_stats_season": home_stats_season or "DEFAULTS",
            "away_stats_season": away_stats_season or "DEFAULTS",
            "requested_season": resolved_season,
            "fallback_season": fallback_season,
        },
        "data_freshness": freshness,
    }

    warning_parts: list[str] = []
    if stale_data:
        stale_reasons = freshness.get("stale_reasons") or []
        if stale_reasons:
            warning_parts.append("Stale data detected: " + "; ".join(stale_reasons) + ".")
        else:
            warning_parts.append("Stale data detected from historical freshness checks.")
    if used_defaults:
        warning_parts.append(
            "Using league averages as default values because team stats were unavailable "
            "from NBA API/historical sources."
        )

    if warning_parts:
        result["data_warning"] = " ".join(warning_parts)
    
    # Fetch odds if requested
    if fetch_odds:
        try:
            from src.odds.odds_api import fetch_nba_odds_snapshot, OddsAPIError
            odds_snapshot = fetch_nba_odds_snapshot(home_team, away_team)
            
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
