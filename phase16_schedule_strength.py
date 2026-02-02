"""
Phase 16: Add Schedule Strength
Add strength of schedule features.
"""

import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ScheduleStrengthBuilder:
    """
    Add schedule strength features.
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

    def calculate_team_ratings_over_time(self):
        """Calculate rolling team ratings over time."""
        logger.info("Calculating rolling team ratings...")

        df = self.features_df.sort_values('game_date').copy()

        # For each team, calculate rolling efficiency rating
        for team_col in ['home_team_id', 'away_team_id']:
            team_name = team_col.replace('_team_id', '')
            logger.info(f"  Calculating for {team_name} team...")

            # Calculate efficiency rating for each game
            df[f'{team_name}_efficiency_rating'] = (
                df[f'{team_name}_off_rating'] / df[f'{team_name}_def_rating']
            )

        # Calculate rolling averages (last 10 games)
        for team_col in ['home', 'away']:
            team_name = team_col
            logger.info(f"  Calculating rolling stats for {team_name} team...")

            # Get games where this team played (either home or away)
            team_games = df[
                (df['home_team_id'] == team_col) | (df['away_team_id'] == team_col)
            ].copy()

            # Calculate rolling efficiency
            df[f'{team_name}_rolling_efficiency'] = 0.0

            # This is complex - for now, use a simpler approach
            # Use net_rating as a proxy for schedule strength calculation
            pass

        return df

    def calculate_schedule_strength(self, df):
        """Calculate strength of schedule for each game."""
        logger.info("Calculating schedule strength...")

        df = df.sort_values('game_date').copy()

        # For each game, calculate the average rating of opponents in the last 10 games
        # Higher values = tougher schedule (played against strong teams)

        df['home_schedule_strength'] = 0.0
        df['away_schedule_strength'] = 0.0
        df['schedule_strength_diff'] = 0.0

        # Get all teams
        all_teams = set(df['home_team_id'].unique()) | set(df['away_team_id'].unique())

        # For each team, calculate rolling schedule strength
        for team in all_teams:
            # Get all games for this team (home or away)
            team_games = df[
                (df['home_team_id'] == team) | (df['away_team_id'] == team)
            ].sort_values('game_date').copy()

            # Calculate schedule strength (opponent net rating average)
            for i, (game_idx, game) in enumerate(team_games.iterrows()):
                # Get last 10 opponents before this game
                recent_games = team_games.iloc[:i].tail(10)

                if len(recent_games) > 0:
                    # Calculate average opponent net rating
                    opponent_net_ratings = []

                    for _, recent_game in recent_games.iterrows():
                        if recent_game['home_team_id'] == team:
                            # Team was home, opponent is away
                            opponent_net_rating = recent_game['away_net_rating']
                        else:
                            # Team was away, opponent is home
                            opponent_net_rating = recent_game['home_net_rating']

                        opponent_net_ratings.append(opponent_net_rating)

                    if opponent_net_ratings:
                        avg_opponent_net = np.mean(opponent_net_ratings)
                    else:
                        avg_opponent_net = 0.0
                else:
                    avg_opponent_net = 0.0

                # Set schedule strength for this game
                if game['home_team_id'] == team:
                    df.loc[game_idx, 'home_schedule_strength'] = avg_opponent_net
                else:
                    df.loc[game_idx, 'away_schedule_strength'] = avg_opponent_net

        # Calculate differential
        df['schedule_strength_diff'] = df['home_schedule_strength'] - df['away_schedule_strength']

        logger.info("  ✓ Schedule strength calculated")
        return df

    def calculate_recent_form(self, df):
        """Calculate recent form (last 5 games margin)."""
        logger.info("Calculating recent form...")

        df = df.sort_values('game_date').copy()

        df['home_recent_margin'] = 0.0
        df['away_recent_margin'] = 0.0
        df['recent_margin_diff'] = 0.0

        # Get all teams
        all_teams = set(df['home_team_id'].unique()) | set(df['away_team_id'].unique())

        # For each team, calculate recent form
        for team in all_teams:
            # Get all games for this team
            team_games = df[
                (df['home_team_id'] == team) | (df['away_team_id'] == team)
            ].sort_values('game_date').copy()

            # Calculate average margin in last 5 games
            for i, (game_idx, game) in enumerate(team_games.iterrows()):
                # Get last 5 games before this one
                recent_games = team_games.iloc[:i].tail(5)

                if len(recent_games) > 0:
                    margins = []
                    for _, recent_game in recent_games.iterrows():
                        if recent_game['home_team_id'] == team:
                            # Team was home
                            margins.append(recent_game['margin'])
                        else:
                            # Team was away (margin is home - away, so negative for away win)
                            margins.append(-recent_game['margin'])

                    if margins:
                        avg_margin = np.mean(margins)
                    else:
                        avg_margin = 0.0
                else:
                    avg_margin = 0.0

                # Set recent form for this game
                if game['home_team_id'] == team:
                    df.loc[game_idx, 'home_recent_margin'] = avg_margin
                else:
                    df.loc[game_idx, 'away_recent_margin'] = avg_margin

        # Calculate differential
        df['recent_margin_diff'] = df['home_recent_margin'] - df['away_recent_margin']

        logger.info("  ✓ Recent form calculated")
        return df

    def build_features(self):
        """Build all schedule strength features."""
        logger.info("=" * 70)
        logger.info("PHASE 16: Building Schedule Strength Features")
        logger.info("=" * 70)

        self.load_data()

        # Calculate features - chain properly
        df = self.features_df.copy()
        df = self.calculate_schedule_strength(df)
        df = self.calculate_recent_form(df)

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
            'home_schedule_strength',
            'away_schedule_strength',
            'schedule_strength_diff',
            'home_recent_margin',
            'away_recent_margin',
            'recent_margin_diff',
        ]


def main():
    """Run Phase 16."""
    features_path = 'data/processed/h2h_features.parquet'
    output_path = 'data/processed/schedule_strength_features.parquet'

    builder = ScheduleStrengthBuilder(features_path, output_path)
    builder.build_features()
    builder.save_features()

    # Summary
    new_features = builder.get_new_features()
    logger.info("=" * 70)
    logger.info("PHASE 16: COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Added {len(new_features)} new schedule strength features:")
    for feat in new_features:
        logger.info(f"  ✓ {feat}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
