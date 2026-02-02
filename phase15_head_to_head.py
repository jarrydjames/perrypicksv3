"""
Phase 15: Add Head-to-Head History
Add historical matchup features.
"""

import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HeadToHeadBuilder:
    """
    Add head-to-head history features.
    """

    def __init__(self, features_path: str, output_path: str):
        self.features_path = features_path
        self.output_path = output_path
        self.features_df = None
        self.augmented_df = None

    def load_data(self):
        """Load features."""
        logger.info(f"Loading features from {self.features_path}")
        self.features_df = pd.read_parquet(self.features_path)
        logger.info(f"Loaded {len(self.features_df)} games")
        return self

    def calculate_h2h_features(self):
        """Calculate head-to-head history for each matchup."""
        logger.info("Calculating head-to-head history...")

        df = self.features_df.sort_values('game_date').copy()

        # Initialize H2H columns
        df['h2h_home_wins'] = 0.0
        df['h2h_away_wins'] = 0.0
        df['h2h_total_games'] = 0.0
        df['h2h_home_win_pct'] = 0.5
        df['h2h_recent_home_wins'] = 0.0
        df['h2h_recent_away_wins'] = 0.0
        df['h2h_recent_total'] = 0.0
        df['h2h_recent_home_win_pct'] = 0.5

        # Get all unique team pairs
        team_pairs = df.groupby(['home_team_id', 'away_team_id']).size().index.tolist()

        # For each team pair, calculate H2H history
        for home_id, away_id in team_pairs:
            # Get all games between these teams
            mask = ((df['home_team_id'] == home_id) & (df['away_team_id'] == away_id)) | \
                   ((df['home_team_id'] == away_id) & (df['away_team_id'] == home_id))
            h2h_games = df[mask].sort_values('game_date')

            # Calculate H2H history for each game
            for i, (game_idx, game) in enumerate(h2h_games.iterrows()):
                if game['home_team_id'] == home_id:
                    # Home team is the original home team
                    home_wins_before = h2h_games.iloc[:i]['margin'].apply(lambda x: 1 if x > 0 else 0).sum()
                    total_games_before = i
                else:
                    # Home team is the original away team
                    home_wins_before = h2h_games.iloc[:i]['margin'].apply(lambda x: 1 if x < 0 else 0).sum()
                    total_games_before = i

                away_wins_before = total_games_before - home_wins_before

                # Set H2H stats
                df.loc[game_idx, 'h2h_home_wins'] = home_wins_before
                df.loc[game_idx, 'h2h_away_wins'] = away_wins_before
                df.loc[game_idx, 'h2h_total_games'] = total_games_before

                if total_games_before > 0:
                    df.loc[game_idx, 'h2h_home_win_pct'] = home_wins_before / total_games_before

                    # Recent H2H (last 5 games)
                    recent_start = max(0, total_games_before - 5)
                    recent_home_wins = 0
                    recent_away_wins = 0

                    for j in range(recent_start, total_games_before):
                        if game['home_team_id'] == home_id:
                            # Original home team
                            margin = h2h_games.iloc[j]['margin']
                            if (h2h_games.iloc[j]['home_team_id'] == home_id and margin > 0) or \
                               (h2h_games.iloc[j]['away_team_id'] == home_id and margin < 0):
                                recent_home_wins += 1
                            else:
                                recent_away_wins += 1
                        else:
                            # Original away team
                            margin = h2h_games.iloc[j]['margin']
                            if (h2h_games.iloc[j]['home_team_id'] == home_id and margin > 0) or \
                               (h2h_games.iloc[j]['away_team_id'] == home_id and margin < 0):
                                recent_home_wins += 1
                            else:
                                recent_away_wins += 1

                    recent_total = recent_home_wins + recent_away_wins
                    df.loc[game_idx, 'h2h_recent_home_wins'] = recent_home_wins
                    df.loc[game_idx, 'h2h_recent_away_wins'] = recent_away_wins
                    df.loc[game_idx, 'h2h_recent_total'] = recent_total

                    if recent_total > 0:
                        df.loc[game_idx, 'h2h_recent_home_win_pct'] = recent_home_wins / recent_total

        logger.info("  ✓ H2H history calculated")
        return df

    def add_h2h_differentials(self, df):
        """Add H2H differential features."""
        logger.info("Adding H2H differential features...")

        df['h2h_wins_diff'] = df['h2h_home_wins'] - df['h2h_away_wins']
        df['h2h_win_pct_diff'] = df['h2h_home_win_pct'] - (1 - df['h2h_home_win_pct'])
        df['h2h_recent_wins_diff'] = df['h2h_recent_home_wins'] - df['h2h_recent_away_wins']
        df['h2h_recent_win_pct_diff'] = df['h2h_recent_home_win_pct'] - (1 - df['h2h_recent_home_win_pct'])

        logger.info("  ✓ H2H differentials added")
        return df

    def build_features(self):
        """Build all H2H features."""
        logger.info("=" * 70)
        logger.info("PHASE 15: Building Head-to-Head Features")
        logger.info("=" * 70)

        self.load_data()

        # Calculate H2H features
        df = self.calculate_h2h_features()
        df = self.add_h2h_differentials(df)

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
            'h2h_home_wins',
            'h2h_away_wins',
            'h2h_total_games',
            'h2h_home_win_pct',
            'h2h_recent_home_wins',
            'h2h_recent_away_wins',
            'h2h_recent_total',
            'h2h_recent_home_win_pct',
            'h2h_wins_diff',
            'h2h_win_pct_diff',
            'h2h_recent_wins_diff',
            'h2h_recent_win_pct_diff',
        ]


def main():
    """Run Phase 15."""
    features_path = 'data/processed/advanced_stats_features.parquet'
    output_path = 'data/processed/h2h_features.parquet'

    builder = HeadToHeadBuilder(features_path, output_path)
    builder.build_features()
    builder.save_features()

    # Summary
    new_features = builder.get_new_features()
    logger.info("=" * 70)
    logger.info("PHASE 15: COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Added {len(new_features)} new H2H features:")
    for feat in new_features:
        logger.info(f"  ✓ {feat}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
