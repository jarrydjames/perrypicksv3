"""
Phase 14: Add Advanced Team Stats
Add net rating, TS%, assist ratio features.
"""

import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedStatsBuilder:
    """
    Add advanced team statistics to features.
    """

    def __init__(self, features_path: str, output_path: str):
        self.features_path = features_path
        self.output_path = output_path
        self.features_df = None
        self.augmented_df = None

    def load_data(self):
        """Load enhanced features."""
        logger.info(f"Loading enhanced features from {self.features_path}")
        self.features_df = pd.read_parquet(self.features_path)
        logger.info(f"Loaded {len(self.features_df)} games with {len(self.features_df.columns)} columns")
        return self

    def add_net_rating(self, df):
        """Add net rating (offensive - defensive)."""
        logger.info("Adding net rating features...")

        df = df.copy()

        # Net rating = offensive rating - defensive rating
        df['home_net_rating'] = df['home_off_rating'] - df['home_def_rating']
        df['away_net_rating'] = df['away_off_rating'] - df['away_def_rating']
        df['net_rating_diff'] = df['home_net_rating'] - df['away_net_rating']

        logger.info("  ✓ home_net_rating, away_net_rating, net_rating_diff")
        return df

    def add_true_shooting(self, df):
        """Add True Shooting % (TS%)."""
        logger.info("Adding True Shooting % features...")

        df = df.copy()

        # TS% = points / (2 * (FGA + 0.44 * FTA))
        # We don't have FGA and FTA directly, so we'll use available stats
        # Approximation: TS% ≈ (PTS / possessions) / (2 * (FGA / possessions + 0.44 * FTA / possessions))
        # Simplified: TS% ≈ eFG% * (PTS / (2 * FGA)) with FT correction

        # Using eFG% as proxy (we already have it)
        # Let's calculate a more accurate TS% from available data

        # TS% = PTS / (2 * (FGA + 0.44 * FTA))
        # We need FGA and FTA. From team ratings, we have eFG%, pace, etc.
        # Let's calculate TS% using the formula:
        # TS% = eFG% * (PTS / (2 * FGA + 0.44 * FTA)) / (PTS / (2 * FGA))
        # This is getting complex, let's use a simpler approach

        # TS% ≈ eFG% with FT correction
        # For now, let's create TS% features from eFG% and FT rate
        # TS% = eFG% * (1 + 0.5 * (FTA/FGA * 0.44 / (1 + FTA/FGA * 0.44)))
        # This is getting too complicated. Let's use a simpler proxy.

        # Use eFG% as a proxy for TS% (already in data)
        # But let's create interaction with FT rate

        df['home_ts_proxy'] = df['home_efg'] + 0.1 * df['home_ft_rate']
        df['away_ts_proxy'] = df['away_efg'] + 0.1 * df['away_ft_rate']
        df['ts_proxy_diff'] = df['home_ts_proxy'] - df['away_ts_proxy']

        logger.info("  ✓ home_ts_proxy, away_ts_proxy, ts_proxy_diff")
        return df

    def add_assist_ratio(self, df):
        """Add assist ratio features (calculated from existing data)."""
        logger.info("Adding assist ratio features...")

        df = df.copy()

        # Assist Ratio = AST / (FGA + 0.44 * FTA + AST + TOV)
        # We don't have AST and TOV directly in team ratings
        # Let's use tov_rate as a proxy for the denominator part

        # Create proxy for assist ratio
        # Higher TOV rate typically correlates with lower assist ratio
        # We'll create a composite metric

        df['home_assist_ratio_proxy'] = df['home_efg'] * (1 - df['home_tov_rate'])
        df['away_assist_ratio_proxy'] = df['away_efg'] * (1 - df['away_tov_rate'])
        df['assist_ratio_diff'] = df['home_assist_ratio_proxy'] - df['away_assist_ratio_proxy']

        logger.info("  ✓ home_assist_ratio_proxy, away_assist_ratio_proxy, assist_ratio_diff")
        return df

    def add_four_factor_composites(self, df):
        """Add composite four factor metrics."""
        logger.info("Adding four factor composites...")

        df = df.copy()

        # Four Factors:
        # 1. eFG% (already have)
        # 2. TOV% (already have)
        # 3. ORB% (already have)
        # 4. FT Rate (already have)

        # Create differential composites
        df['four_factor_diff'] = (
            (df['home_efg'] - df['away_efg']) +
            (df['away_tov_rate'] - df['home_tov_rate']) +  # Lower is better for opponent
            (df['home_orb_rate'] - df['away_orb_rate']) +
            (df['home_ft_rate'] - df['away_ft_rate'])
        )

        # Weighted four factor (typical weights: 0.4 eFG%, 0.25 TOV%, 0.2 ORB%, 0.15 FT)
        df['home_four_factor_weighted'] = (
            0.4 * df['home_efg'] +
            -0.25 * df['home_tov_rate'] +
            0.2 * df['home_orb_rate'] +
            0.15 * df['home_ft_rate']
        )
        df['away_four_factor_weighted'] = (
            0.4 * df['away_efg'] +
            -0.25 * df['away_tov_rate'] +
            0.2 * df['away_orb_rate'] +
            0.15 * df['away_ft_rate']
        )
        df['four_factor_weighted_diff'] = df['home_four_factor_weighted'] - df['away_four_factor_weighted']

        logger.info("  ✓ four_factor_diff, four_factor_weighted, weighted_diff")
        return df

    def add_efficiency_metrics(self, df):
        """Add additional efficiency metrics."""
        logger.info("Adding efficiency metrics...")

        df = df.copy()

        # Off/Def rating differential
        df['off_rating_diff'] = df['home_off_rating'] - df['away_off_rating']
        df['def_rating_diff'] = df['away_def_rating'] - df['home_def_rating']  # Lower is better

        # Pace differential
        df['pace_diff'] = df['home_pace'] - df['away_pace']

        # Overall efficiency score
        df['home_efficiency_score'] = df['home_off_rating'] * (1 / df['home_def_rating'])
        df['away_efficiency_score'] = df['away_off_rating'] * (1 / df['away_def_rating'])
        df['efficiency_diff'] = df['home_efficiency_score'] - df['away_efficiency_score']

        logger.info("  ✓ off_rating_diff, def_rating_diff, pace_diff, efficiency_score, efficiency_diff")
        return df

    def build_features(self):
        """Build all advanced stats features."""
        logger.info("=" * 70)
        logger.info("PHASE 14: Building Advanced Team Stats Features")
        logger.info("=" * 70)

        # Load data
        self.load_data()

        # Add features - chain them properly
        df = self.features_df.copy()
        df = self.add_net_rating(df)
        df = self.add_true_shooting(df)
        df = self.add_assist_ratio(df)
        df = self.add_four_factor_composites(df)
        df = self.add_efficiency_metrics(df)

        self.augmented_df = df
        return self

    def save_features(self):
        """Save augmented features."""
        logger.info(f"Saving augmented features to {self.output_path}")
        self.augmented_df.to_parquet(self.output_path, index=False)
        logger.info(f"✓ Saved {len(self.augmented_df)} games with {len(self.augmented_df.columns)} columns")

        # Save feature list
        feature_list_path = self.output_path.replace('.parquet', '_feature_list.txt')
        with open(feature_list_path, 'w') as f:
            for col in self.augmented_df.columns:
                f.write(f"{col}\n")
        logger.info(f"✓ Feature list saved to {feature_list_path}")

        return self

    def get_new_features(self):
        """Return list of newly added features."""
        return [
            'home_net_rating',
            'away_net_rating',
            'net_rating_diff',
            'home_ts_proxy',
            'away_ts_proxy',
            'ts_proxy_diff',
            'home_assist_ratio_proxy',
            'away_assist_ratio_proxy',
            'assist_ratio_diff',
            'four_factor_diff',
            'home_four_factor_weighted',
            'away_four_factor_weighted',
            'four_factor_weighted_diff',
            'off_rating_diff',
            'def_rating_diff',
            'pace_diff',
            'home_efficiency_score',
            'away_efficiency_score',
            'efficiency_diff',
        ]


def main():
    """Run Phase 14."""
    features_path = 'data/processed/enhanced_features.parquet'
    output_path = 'data/processed/advanced_stats_features.parquet'

    builder = AdvancedStatsBuilder(features_path, output_path)
    builder.build_features()
    builder.save_features()

    # Summary
    new_features = builder.get_new_features()
    logger.info("=" * 70)
    logger.info("PHASE 14: COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Added {len(new_features)} new advanced stats features:")
    for feat in new_features:
        logger.info(f"  ✓ {feat}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
