"""
Predictor: Use Pre-Game Models to Predict NBA Games
Makes predictions using only data available before tipoff.
"""

import joblib
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TeamRatingsPredictor:
    """
    Predict game outcomes using team ratings.
    
    Uses pre-game team ratings to predict:
    - Total points (over/under)
    - Point spread (winner and margin)
    """
    
    def __init__(self):
        self.processed_dir = Path("data/processed")
        self.models_dir = Path("data/models")
        
        # Load models
        self.total_model = joblib.load(self.models_dir / "total_model_pregame.pkl")
        self.margin_model = joblib.load(self.models_dir / "margin_model_pregame.pkl")
        
        # Load feature list
        with open(self.processed_dir / "pregame_feature_list.txt") as f:
            self.feature_cols = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded models and {len(self.feature_cols)} features")
    
    def load_latest_ratings(self) -> pd.DataFrame:
        """Load the latest team ratings from historical data."""
        try:
            df_ratings = pd.read_parquet(self.processed_dir / "team_ratings.parquet")
            return df_ratings
        except FileNotFoundError:
            logger.error("Team ratings file not found. Run Phase 5 first.")
            return pd.DataFrame()
    
    def get_team_rating(self, df_ratings: pd.DataFrame, team_id: int) -> Dict:
        """Get latest rating for a team."""
        # Find the last game for this team
        team_games = df_ratings[
            (df_ratings['home_team_id'] == team_id) |
            (df_ratings['away_team_id'] == team_id)
        ]
        
        if len(team_games) == 0:
            # Return default league averages
            return {
                'off_rating': 110.0,
                'def_rating': 110.0,
                'pace': 100.0,
                'efg': 0.52,
                'tov_rate': 0.14,
                'orb_rate': 0.25,
                'ft_rate': 0.20,
                'win_pct': 0.5,
                'home_win_pct': 0.5,
                'road_win_pct': 0.5,
            }
        
        # Get the most recent game with this team
        latest_game = team_games.sort_values('game_date').iloc[-1]
        
        # Determine if they were home or away
        if latest_game['home_team_id'] == team_id:
            return {
                'off_rating': latest_game['home_off_rating'],
                'def_rating': latest_game['home_def_rating'],
                'pace': latest_game['home_pace'],
                'efg': latest_game['home_efg'],
                'tov_rate': latest_game['home_tov_rate'],
                'orb_rate': latest_game['home_orb_rate'],
                'ft_rate': latest_game['home_ft_rate'],
                'win_pct': latest_game['home_win_pct'],
                'home_win_pct': latest_game['home_home_win_pct'],
                'road_win_pct': latest_game['away_road_win_pct'],  # Note: this is away team's road win%
            }
        else:
            return {
                'off_rating': latest_game['away_off_rating'],
                'def_rating': latest_game['away_def_rating'],
                'pace': latest_game['away_pace'],
                'efg': latest_game['away_efg'],
                'tov_rate': latest_game['away_tov_rate'],
                'orb_rate': latest_game['away_orb_rate'],
                'ft_rate': latest_game['away_ft_rate'],
                'win_pct': latest_game['away_win_pct'],
                'home_win_pct': latest_game['away_win_pct'],
                'road_win_pct': latest_game['away_road_win_pct'],
            }
    
    def build_features_for_matchup(self, home_rating: Dict, away_rating: Dict) -> Dict:
        """Build feature vector for a matchup from team ratings."""
        # Team ratings
        features = {
            'home_off_rating': home_rating['off_rating'],
            'away_off_rating': away_rating['off_rating'],
            'home_def_rating': home_rating['def_rating'],
            'away_def_rating': away_rating['def_rating'],
            'home_pace': home_rating['pace'],
            'away_pace': away_rating['pace'],
            'home_efg': home_rating['efg'],
            'away_efg': away_rating['efg'],
            'home_tov_rate': home_rating['tov_rate'],
            'away_tov_rate': away_rating['tov_rate'],
            'home_orb_rate': home_rating['orb_rate'],
            'away_orb_rate': away_rating['orb_rate'],
            'home_ft_rate': home_rating['ft_rate'],
            'away_ft_rate': away_rating['ft_rate'],
            
            # Win percentages
            'home_win_pct': home_rating['win_pct'],
            'away_win_pct': away_rating['win_pct'],
            'home_home_win_pct': home_rating['home_win_pct'],
            'away_road_win_pct': away_rating['road_win_pct'],
        }
        
        # Calculate differentials
        features['off_rating_diff'] = features['home_off_rating'] - features['away_off_rating']
        features['def_rating_diff'] = features['home_def_rating'] - features['away_def_rating']
        features['pace_diff'] = features['home_pace'] - features['away_pace']
        features['efg_diff'] = features['home_efg'] - features['away_efg']
        features['tov_rate_diff'] = features['home_tov_rate'] - features['away_tov_rate']
        features['orb_rate_diff'] = features['home_orb_rate'] - features['away_orb_rate']
        features['ft_rate_diff'] = features['home_ft_rate'] - features['away_ft_rate']
        features['win_pct_diff'] = features['home_win_pct'] - features['away_win_pct']
        
        # Expected pace
        features['expected_pace'] = (features['home_pace'] + features['away_pace']) / 2
        
        # Home court advantage
        features['home_court_advantage'] = (
            features['home_home_win_pct'] - features['away_road_win_pct']
        )
        
        # Offensive vs Defensive matchups
        features['home_off_vs_away_def'] = features['home_off_rating'] - features['away_def_rating']
        features['away_off_vs_home_def'] = features['away_off_rating'] - features['home_def_rating']
        
        # Combined ratings
        features['combined_off_rating'] = (
            features['home_off_rating'] + features['away_off_rating']
        ) / 2
        features['combined_def_rating'] = (
            features['home_def_rating'] + features['away_def_rating']
        ) / 2
        
        # Expected total and margin
        features['expected_total'] = (
            (features['home_off_rating'] + features['away_off_rating']) / 100 *
            features['expected_pace']
        )
        features['expected_margin'] = (
            features['home_off_vs_away_def'] - features['away_off_vs_home_def']
        )
        
        # Interaction features
        features['off_x_pace'] = features['combined_off_rating'] * features['expected_pace'] / 100
        features['pace_diff_x_home_adv'] = features['pace_diff'] * features['home_court_advantage']
        
        return features
    
    def predict_game(self, home_team_id: int, away_team_id: int) -> Dict:
        """
        Predict a game outcome.
        
        Args:
            home_team_id: NBA team ID for home team
            away_team_id: NBA team ID for away team
            
        Returns:
            Dict with predictions:
                - total: predicted total points
                - margin: predicted home - away margin
                - home_score: predicted home score
                - away_score: predicted away score
                - winner: predicted winner ('home' or 'away')
                - confidence: confidence level (0-1)
        """
        # Load team ratings
        df_ratings = self.load_latest_ratings()
        
        if len(df_ratings) == 0:
            logger.error("No ratings data available")
            return {}
        
        # Get team ratings
        home_rating = self.get_team_rating(df_ratings, home_team_id)
        away_rating = self.get_team_rating(df_ratings, away_team_id)
        
        # Build features
        features = self.build_features_for_matchup(home_rating, away_rating)
        
        # Create feature vector in correct order
        X = np.array([features[col] for col in self.feature_cols]).reshape(1, -1)
        
        # Make predictions
        pred_total = self.total_model.predict(X)[0]
        pred_margin = self.margin_model.predict(X)[0]
        
        # Predict scores
        pred_home_score = (pred_total + pred_margin) / 2
        pred_away_score = (pred_total - pred_margin) / 2
        
        # Predict winner
        winner = 'home' if pred_margin > 0 else 'away'
        
        # Confidence based on margin magnitude
        confidence = min(0.95, abs(pred_margin) / 20 + 0.5)
        
        return {
            'total': round(pred_total, 1),
            'margin': round(pred_margin, 1),
            'home_score': round(pred_home_score, 1),
            'away_score': round(pred_away_score, 1),
            'winner': winner,
            'confidence': round(confidence, 2),
            'features': features,
        }
    
    def predict_games(self, matchups: List[Tuple[int, int]]) -> List[Dict]:
        """
        Predict multiple games.
        
        Args:
            matchups: List of (home_team_id, away_team_id) tuples
            
        Returns:
            List of prediction dicts
        """
        predictions = []
        
        for home_id, away_id in matchups:
            pred = self.predict_game(home_id, away_id)
            pred['home_team_id'] = home_id
            pred['away_team_id'] = away_id
            predictions.append(pred)
        
        return predictions


def main():
    """Test the predictor."""
    print("="*70)
    print("TESTING PREGAME PREDICTOR")
    print("="*70)
    
    predictor = TeamRatingsPredictor()
    
    # Test with a sample matchup
    # Using recent team IDs
    home_id = 1610612747  # Example: Lakers
    away_id = 1610612744  # Example: Warriors
    
    print(f"\nPredicting matchup: Home {home_id} vs Away {away_id}")
    
    prediction = predictor.predict_game(home_id, away_id)
    
    print("\nPrediction:")
    print(f"  Total: {prediction['total']}")
    print(f"  Margin: {prediction['margin']}")
    print(f"  Home Score: {prediction['home_score']}")
    print(f"  Away Score: {prediction['away_score']}")
    print(f"  Winner: {prediction['winner']}")
    print(f"  Confidence: {prediction['confidence']}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
