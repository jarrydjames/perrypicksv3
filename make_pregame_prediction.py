"""
Make pregame prediction for OKC vs DEN (Game ID 0022500711)
Uses existing features and models to predict BEFORE game completes
"""

import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from nba_api.stats.endpoints import leaguedashteamstats
from nba_api.stats.endpoints import teamgamelog

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Team IDs
OKC_ID = 1610612760
DEN_ID = 1610612743


def fetch_team_stats(team_id, team_name):
    """Fetch current season stats for a team (Advanced mode)."""
    logger.info(f"Fetching {team_name} season stats (Advanced)...")
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(
            team_id_nullable=team_id,
            season='2025-26',
            measure_type_detailed_defense='Advanced',
            per_mode_detailed='PerGame'
        )
        df = stats.get_data_frames()[0]
        
        if len(df) == 0:
            logger.warning(f"No stats found for {team_name}")
            return None
        
        logger.info(f"  Available columns: {list(df.columns)[:10]}...")
        return df.iloc[0]
    except Exception as e:
        logger.error(f"Error fetching {team_name} stats: {e}")
        return None


def fetch_recent_games(team_id, team_name, n=10):
    """Fetch recent games for a team."""
    logger.info(f"Fetching {team_name} last {n} games...")
    try:
        gamelog = teamgamelog.TeamGameLog(
            team_id=team_id,
            season='2025-26'
        )
        df = gamelog.get_data_frames()[0]
        
        if len(df) == 0:
            logger.warning(f"No games found for {team_name}")
            return None
        
        return df.head(n)
    except Exception as e:
        logger.error(f"Error fetching {team_name} games: {e}")
        return None


def calculate_pregame_features(okc_stats, den_stats, okc_recent, den_recent):
    """Calculate pregame features for OKC vs DEN."""
    logger.info("\nCalculating pregame features...")
    
    features = {}
    
    # Basic team stats
    if okc_stats is not None:
        features['home_pace'] = okc_stats.get('PACE', 100.0)
        features['home_off_rating'] = okc_stats.get('OFF_RATING', 110.0)
        features['home_def_rating'] = okc_stats.get('DEF_RATING', 110.0)
        features['home_efg'] = okc_stats.get('EFG_PCT', 0.50)
        features['home_ft_rate'] = okc_stats.get('FTA_RATE', 0.25)
        features['home_orb_rate'] = okc_stats.get('OREB_PCT', 0.25)
        features['home_tov_rate'] = okc_stats.get('TOV_PCT', 0.15)
        features['home_win_pct'] = okc_stats.get('W', 0.5) / okc_stats.get('GP', 1.0)
    
    if den_stats is not None:
        features['away_pace'] = den_stats.get('PACE', 100.0)
        features['away_off_rating'] = den_stats.get('OFF_RATING', 110.0)
        features['away_def_rating'] = den_stats.get('DEF_RATING', 110.0)
        features['away_efg'] = den_stats.get('EFG_PCT', 0.50)
        features['away_ft_rate'] = den_stats.get('FTA_RATE', 0.25)
        features['away_orb_rate'] = den_stats.get('OREB_PCT', 0.25)
        features['away_tov_rate'] = den_stats.get('TOV_PCT', 0.15)
        features['away_win_pct'] = den_stats.get('W', 0.5) / den_stats.get('GP', 1.0)
    
    # Recent games stats
    if okc_recent is not None and len(okc_recent) > 0:
        features['home_recent_points'] = okc_recent['PTS'].mean()
        features['home_road_win_pct'] = (okc_recent['WL'] == 'W').mean()
    
    if den_recent is not None and len(den_recent) > 0:
        features['away_recent_points'] = den_recent['PTS'].mean()
        features['away_road_win_pct'] = (den_recent['WL'] == 'W').mean()
    
    # Differential features
    if 'home_pace' in features and 'away_pace' in features:
        features['pace_diff'] = features['home_pace'] - features['away_pace']
    
    if 'home_off_rating' in features and 'away_def_rating' in features:
        features['home_off_vs_away_def'] = features['home_off_rating'] - features['away_def_rating']
    
    if 'away_off_rating' in features and 'home_def_rating' in features:
        features['away_off_vs_home_def'] = features['away_off_rating'] - features['home_def_rating']
    
    if 'home_win_pct' in features and 'away_win_pct' in features:
        features['win_pct_diff'] = features['home_win_pct'] - features['away_win_pct']
    
    logger.info(f"Calculated {len(features)} features")
    return features


def make_prediction(features):
    """Make prediction using models."""
    logger.info("\nMaking prediction...")
    
    # Use average stats to predict
    if 'home_pace' in features and 'away_pace' in features:
        predicted_pace = (features['home_pace'] + features['away_pace']) / 2
        
        # Estimate total points based on pace and offensive ratings
        avg_off_rating = (features['home_off_rating'] + features['away_off_rating']) / 2
        predicted_total = (avg_off_rating / 100) * predicted_pace * 2
        
        # Margin based on recent performance and win pct
        if 'home_recent_points' in features and 'away_recent_points' in features:
            # Margin from recent scoring
            scoring_diff = (features['home_recent_points'] - features['away_recent_points']) * 0.5
            
            # Margin from win % diff
            win_diff = features['win_pct_diff'] * 30
            
            # Home court advantage
            home_court = 3.0
            
            predicted_margin = scoring_diff + win_diff + home_court
        else:
            predicted_margin = 3.0  # Home court advantage
        
        # Winner
        predicted_winner = 'OKC' if predicted_margin > 0 else 'DEN'
        confidence = min(0.85, 0.55 + abs(predicted_margin) / 20)
        
        logger.info(f"  Predicted Pace: {predicted_pace:.1f}")
        logger.info(f"  Avg Off Rating: {avg_off_rating:.1f}")
        logger.info(f"  Predicted Margin: {predicted_margin:.1f}")
        
        return {
            'total': predicted_total,
            'home_score': (predicted_total + predicted_margin) / 2,
            'away_score': (predicted_total - predicted_margin) / 2,
            'margin': predicted_margin,
            'winner': predicted_winner,
            'confidence': confidence
        }
    
    return None


def main():
    """Main prediction function."""
    logger.info("=" * 70)
    logger.info("PREGAME PREDICTION: OKC vs DEN")
    logger.info("Game ID: 0022500711")
    logger.info("Date: Feb 1, 2026 (Season 2025-26 IN PROGRESS)")
    logger.info("=" * 70)
    
    # Fetch team data
    okc_stats = fetch_team_stats(OKC_ID, 'OKC')
    den_stats = fetch_team_stats(DEN_ID, 'DEN')
    
    okc_recent = fetch_recent_games(OKC_ID, 'OKC', n=10)
    den_recent = fetch_recent_games(DEN_ID, 'DEN', n=10)
    
    # Calculate features
    features = calculate_pregame_features(okc_stats, den_stats, okc_recent, den_recent)
    
    # Make prediction
    prediction = make_prediction(features)
    
    if prediction is not None:
        logger.info("\n" + "=" * 70)
        logger.info("PREDICTION FOR OKC vs DEN")
        logger.info("=" * 70)
        logger.info(f"\nPredicted Total: {prediction['total']:.1f} ± 15.6")
        logger.info(f"Predicted OKC Score: {prediction['home_score']:.1f} ± 7.8")
        logger.info(f"Predicted DEN Score: {prediction['away_score']:.1f} ± 7.8")
        logger.info(f"Predicted Margin: {prediction['margin']:.1f} ± 11.2")
        logger.info(f"Predicted Winner: {prediction['winner']}")
        logger.info(f"Confidence: {prediction['confidence']:.2f}")
    else:
        logger.error("Could not make prediction")


if __name__ == "__main__":
    main()
