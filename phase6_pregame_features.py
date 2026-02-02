"""
Phase 6: Build Pre-Game Features from Team Ratings
Create matchup features from historical team ratings (what we know before tipoff).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PregameFeatureBuilder:
    """
    Build pre-game features from team ratings.
    
    Features created from what we know BEFORE the game:
    - Team offensive/defensive ratings
    - Team pace
    - Team efficiency metrics (4 factors)
    - Win percentages
    - Home court advantage
    - Matchup differentials
    - Interaction features
    """
    
    def __init__(self):
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_team_ratings(self) -> pd.DataFrame:
        """Load team ratings history."""
        logger.info("Loading team ratings...")
        df = pd.read_parquet(self.processed_dir / "team_ratings.parquet")
        logger.info(f"  Loaded {len(df)} games")
        return df
    
    def build_pregame_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build pre-game features from team ratings."""
        logger.info("Building pre-game features...")
        
        # Create interaction features
        df['off_rating_diff'] = df['home_off_rating'] - df['away_off_rating']
        df['def_rating_diff'] = df['home_def_rating'] - df['away_def_rating']
        df['pace_diff'] = df['home_pace'] - df['away_pace']
        df['efg_diff'] = df['home_efg'] - df['away_efg']
        df['tov_rate_diff'] = df['home_tov_rate'] - df['away_tov_rate']
        df['orb_rate_diff'] = df['home_orb_rate'] - df['away_orb_rate']
        df['ft_rate_diff'] = df['home_ft_rate'] - df['away_ft_rate']
        df['win_pct_diff'] = df['home_win_pct'] - df['away_win_pct']
        
        # Expected pace (combination of both teams' pace)
        df['expected_pace'] = (df['home_pace'] + df['away_pace']) / 2
        
        # Home court advantage factor
        df['home_court_advantage'] = (
            df['home_home_win_pct'] - df['away_road_win_pct']
        )
        
        # Offensive vs Defensive matchups
        df['home_off_vs_away_def'] = df['home_off_rating'] - df['away_def_rating']
        df['away_off_vs_home_def'] = df['away_off_rating'] - df['home_def_rating']
        
        # Combined offensive rating (home offense + away offense adjusted for defense)
        df['combined_off_rating'] = (
            df['home_off_rating'] + df['away_off_rating']
        ) / 2
        
        # Combined defensive rating
        df['combined_def_rating'] = (
            df['home_def_rating'] + df['away_def_rating']
        ) / 2
        
        # Expected total points (using offensive ratings and pace)
        # Higher pace + higher offensive rating = higher total
        df['expected_total'] = (
            (df['home_off_rating'] + df['away_off_rating']) / 100 *
            (df['home_pace'] + df['away_pace']) / 2
        )
        
        # Expected margin
        df['expected_margin'] = (
            df['home_off_vs_away_def'] - df['away_off_vs_home_def']
        )
        
        # Win probability based on rating differentials
        df['home_win_prob'] = 0.5 + (
            df['off_rating_diff'] * 0.01 +
            df['def_rating_diff'] * 0.01 +
            df['home_court_advantage'] * 0.05
        )
        df['home_win_prob'] = df['home_win_prob'].clip(0.05, 0.95)
        
        # Interaction features
        df['off_x_pace'] = df['combined_off_rating'] * df['expected_pace'] / 100
        df['pace_diff_x_home_adv'] = df['pace_diff'] * df['home_court_advantage']
        
        # Normalize features to prevent extreme values
        df['home_win_pct'] = df['home_win_pct'].clip(0, 1)
        df['away_win_pct'] = df['away_win_pct'].clip(0, 1)
        df['home_home_win_pct'] = df['home_home_win_pct'].clip(0, 1)
        df['away_road_win_pct'] = df['away_road_win_pct'].clip(0, 1)
        
        logger.info(f"  Created {len(df.columns)} pre-game features")
        
        return df
    
    def select_feature_columns(self, df: pd.DataFrame) -> list:
        """Select feature columns for model training."""
        feature_cols = [
            # Team ratings
            'home_off_rating', 'away_off_rating',
            'home_def_rating', 'away_def_rating',
            'home_pace', 'away_pace',
            'home_efg', 'away_efg',
            'home_tov_rate', 'away_tov_rate',
            'home_orb_rate', 'away_orb_rate',
            'home_ft_rate', 'away_ft_rate',
            
            # Win percentages
            'home_win_pct', 'away_win_pct',
            'home_home_win_pct', 'away_road_win_pct',
            
            # Matchup differentials
            'off_rating_diff', 'def_rating_diff', 'pace_diff',
            'efg_diff', 'tov_rate_diff', 'orb_rate_diff', 'ft_rate_diff',
            'win_pct_diff',
            
            # Matchup features
            'home_off_vs_away_def', 'away_off_vs_home_def',
            'home_court_advantage',
            'expected_pace', 'expected_total', 'expected_margin',
            
            # Interaction features
            'off_x_pace', 'pace_diff_x_home_adv',
        ]
        
        logger.info(f"  Selected {len(feature_cols)} features for modeling")
        return feature_cols
    
    def run(self):
        """Run complete pre-game feature building."""
        logger.info("="*70)
        logger.info("PHASE 6: BUILD PREGAME FEATURES")
        logger.info("="*70)
        
        # Step 1: Load team ratings
        df = self.load_team_ratings()
        
        if len(df) == 0:
            logger.error("No data loaded - stopping")
            return None
        
        # Step 2: Build pre-game features
        df_features = self.build_pregame_features(df)
        
        # Step 3: Select feature columns
        feature_cols = self.select_feature_columns(df_features)
        
        # Step 4: Save
        output_path = self.processed_dir / "pregame_features.parquet"
        df_features.to_parquet(output_path, index=False)
        logger.info(f"Saved pre-game features to {output_path}")
        
        # Save feature list
        feature_list_path = self.processed_dir / "pregame_feature_list.txt"
        with open(feature_list_path, 'w') as f:
            f.write('\n'.join(feature_cols))
        logger.info(f"Saved feature list to {feature_list_path}")
        
        # Display sample
        logger.info(f"\nDataset shape: {df_features.shape}")
        logger.info("\nSample features (first 3 games):")
        sample_cols = ['game_id', 'expected_total', 'expected_margin', 
                      'home_win_prob'] + feature_cols[:5]
        print(df_features[sample_cols].head().to_string(index=False))
        
        # Display feature statistics
        logger.info("\nFeature Statistics:")
        print(df_features[feature_cols].describe().loc[['mean', 'std', 'min', 'max']])
        
        logger.info("="*70)
        logger.info("PHASE 6 COMPLETE")
        logger.info("="*70)
        
        return df_features, feature_cols


def main():
    builder = PregameFeatureBuilder()
    return builder.run()


if __name__ == '__main__':
    exit(main())
