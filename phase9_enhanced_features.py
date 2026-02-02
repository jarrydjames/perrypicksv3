"""
Phase 9: Enhanced Feature Engineering
Add rest days, back-to-back games, and recent form features.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedFeaturesBuilder:
    """
    Build enhanced features including rest, back-to-back, and recent form.
    """

    def __init__(self, ratings_path: str, output_path: str):
        self.ratings_path = ratings_path
        self.output_path = output_path
        self.ratings_df = None
        self.features_df = None

    def load_data(self):
        """Load team ratings data."""
        logger.info(f"Loading team ratings from {self.ratings_path}")
        self.ratings_df = pd.read_parquet(self.ratings_path)
        logger.info(f"Loaded {len(self.ratings_df)} games")
        return self

    def calculate_rest_days(self):
        """
        Calculate days since last game for each team.
        
        Rest days is crucial because:
        - 1-2 days: Fatigue from recent game
        - 3-4 days: Optimal rest
        - 5+ days: Rust from too much time off
        """
        logger.info("Calculating rest days...")
        
        # Sort by game date
        df = self.ratings_df.sort_values('game_date').copy()
        df['game_date_only'] = pd.to_datetime(df['game_date']).dt.date
        
        # Initialize rest days to 7 (typical offseason/early season)
        df['home_rest_days'] = 7.0
        df['away_rest_days'] = 7.0
        
        # Get unique teams
        all_teams = set(df['home_team_id'].unique()) | set(df['away_team_id'].unique())
        
        # For each team, calculate rest days
        for team_id in all_teams:
            # Get all games for this team (both home and away)
            team_games = df[
                (df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)
            ].sort_values('game_date_only')
            
            # Calculate rest days
            for i in range(len(team_games)):
                game = team_games.iloc[i]
                game_idx = game.name
                game_date = game['game_date_only']
                
                # Previous game
                if i > 0:
                    prev_game_date = team_games.iloc[i-1]['game_date_only']
                    rest_days = (game_date - prev_game_date).days
                else:
                    rest_days = 7.0  # First game of season
                
                # Set rest days
                if game['home_team_id'] == team_id:
                    df.loc[game_idx, 'home_rest_days'] = rest_days
                else:
                    df.loc[game_idx, 'away_rest_days'] = rest_days
        
        # Rest days differential
        df['rest_days_diff'] = df['home_rest_days'] - df['away_rest_days']
        
        logger.info("✓ Rest days calculated")
        return df

    def calculate_back_to_back(self, df):
        """
        Identify back-to-back games.
        
        Back-to-backs cause 2-3 point drop in scoring.
        Second night of B2B: ~-2.5 points
        Travel + B2B: ~-3.5 points
        """
        logger.info("Calculating back-to-back flags...")
        
        # B2B = rest days == 1
        df['home_is_b2b'] = (df['home_rest_days'] == 1).astype(int)
        df['away_is_b2b'] = (df['away_rest_days'] == 1).astype(int)
        
        # B2B x Home/Away (interaction)
        df['home_b2b_x_home'] = df['home_is_b2b']
        df['away_b2b_x_away'] = df['away_is_b2b']
        
        # Overall B2B differential
        df['b2b_diff'] = df['home_is_b2b'] - df['away_is_b2b']
        
        # Log B2B stats
        home_b2b_pct = df['home_is_b2b'].sum() / len(df) * 100
        away_b2b_pct = df['away_is_b2b'].sum() / len(df) * 100
        logger.info(f"  Home B2B games: {df['home_is_b2b'].sum()} ({home_b2b_pct:.1f}%)")
        logger.info(f"  Away B2B games: {df['away_is_b2b'].sum()} ({away_b2b_pct:.1f}%)")
        
        return df

    def calculate_recent_form(self, df, n_games=5):
        """
        Calculate recent form metrics for last N games.
        
        Recent form is important because:
        - Teams on hot/cold streaks continue
        - Captures short-term rating changes
        - Momentum effect is documented in sports
        
        Metrics:
        - Average points scored/allowed in last N games
        - Average margin in last N games
        - Win percentage in last N games
        """
        logger.info(f"Calculating recent form (last {n_games} games)...")
        
        # Sort by game date
        df = df.sort_values('game_date').copy()
        
        # Initialize recent form columns
        df['home_recent_points'] = np.nan
        df['away_recent_points'] = np.nan
        df['home_recent_allowed'] = np.nan
        df['away_recent_allowed'] = np.nan
        df['home_recent_margin'] = np.nan
        df['away_recent_margin'] = np.nan
        df['home_recent_wins'] = np.nan
        df['away_recent_wins'] = np.nan
        
        # Get unique teams
        all_teams = set(df['home_team_id'].unique()) | set(df['away_team_id'].unique())
        
        # For each team, calculate recent form
        for team_id in all_teams:
            # Get all games for this team (both home and away)
            team_games = df[
                (df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)
            ].sort_values('game_date')
            
            # Track recent performance
            recent_points = []
            recent_allowed = []
            recent_margin = []
            recent_wins = []
            
            for i, (game_idx, game) in enumerate(team_games.iterrows()):
                # Get team's performance in this game
                if game['home_team_id'] == team_id:
                    points = game['home_score']
                    allowed = game['away_score']
                    is_win = game['home_score'] > game['away_score']
                else:
                    points = game['away_score']
                    allowed = game['home_score']
                    is_win = game['away_score'] > game['home_score']
                
                margin = points - allowed
                
                # Update recent stats
                recent_points.append(points)
                recent_allowed.append(allowed)
                recent_margin.append(margin)
                recent_wins.append(1 if is_win else 0)
                
                # Keep only last N games (excluding current)
                if len(recent_points) > n_games:
                    recent_points = recent_points[-n_games:]
                    recent_allowed = recent_allowed[-n_games:]
                    recent_margin = recent_margin[-n_games:]
                    recent_wins = recent_wins[-n_games:]
                
                # For current game, use stats from previous games only
                if i < n_games:
                    # Not enough games yet, use available games
                    if i == 0:
                        # First game - no history, use team average (0)
                        avg_points = 0
                        avg_allowed = 0
                        avg_margin = 0
                        avg_wins = 0.5
                    else:
                        # Use previous games (excluding current)
                        avg_points = np.mean(recent_points[:-1]) if len(recent_points) > 1 else 0
                        avg_allowed = np.mean(recent_allowed[:-1]) if len(recent_allowed) > 1 else 0
                        avg_margin = np.mean(recent_margin[:-1]) if len(recent_margin) > 1 else 0
                        avg_wins = np.mean(recent_wins[:-1]) if len(recent_wins) > 1 else 0.5
                else:
                    # Have enough history
                    avg_points = np.mean(recent_points[:-1])
                    avg_allowed = np.mean(recent_allowed[:-1])
                    avg_margin = np.mean(recent_margin[:-1])
                    avg_wins = np.mean(recent_wins[:-1])
                
                # Set recent form for this game
                if game['home_team_id'] == team_id:
                    df.loc[game_idx, 'home_recent_points'] = avg_points
                    df.loc[game_idx, 'home_recent_allowed'] = avg_allowed
                    df.loc[game_idx, 'home_recent_margin'] = avg_margin
                    df.loc[game_idx, 'home_recent_wins'] = avg_wins
                else:
                    df.loc[game_idx, 'away_recent_points'] = avg_points
                    df.loc[game_idx, 'away_recent_allowed'] = avg_allowed
                    df.loc[game_idx, 'away_recent_margin'] = avg_margin
                    df.loc[game_idx, 'away_recent_wins'] = avg_wins
        
        # Form differentials
        df['recent_points_diff'] = df['home_recent_points'] - df['away_recent_points']
        df['recent_allowed_diff'] = df['home_recent_allowed'] - df['away_recent_allowed']
        df['recent_margin_diff'] = df['home_recent_margin'] - df['away_recent_margin']
        df['recent_wins_diff'] = df['home_recent_wins'] - df['away_recent_wins']
        
        # Fill NaN values (early games) with 0 or 0.5
        df['home_recent_points'].fillna(0, inplace=True)
        df['away_recent_points'].fillna(0, inplace=True)
        df['home_recent_allowed'].fillna(0, inplace=True)
        df['away_recent_allowed'].fillna(0, inplace=True)
        df['home_recent_margin'].fillna(0, inplace=True)
        df['away_recent_margin'].fillna(0, inplace=True)
        df['home_recent_wins'].fillna(0.5, inplace=True)
        df['away_recent_wins'].fillna(0.5, inplace=True)
        
        # Recalculate differentials after filling
        df['recent_points_diff'] = df['home_recent_points'] - df['away_recent_points']
        df['recent_allowed_diff'] = df['home_recent_allowed'] - df['away_recent_allowed']
        df['recent_margin_diff'] = df['home_recent_margin'] - df['away_recent_margin']
        df['recent_wins_diff'] = df['home_recent_wins'] - df['away_recent_wins']
        
        logger.info("✓ Recent form calculated")
        return df

    def build_features(self):
        """Build all enhanced features."""
        logger.info("=" * 70)
        logger.info("PHASE 9: Building Enhanced Features")
        logger.info("=" * 70)
        
        # Load data
        self.load_data()
        
        # Calculate features
        df = self.calculate_rest_days()
        df = self.calculate_back_to_back(df)
        df = self.calculate_recent_form(df, n_games=5)
        
        # Clean up
        df = df.drop(columns=['game_date_only'], errors='ignore')
        
        self.features_df = df
        return self

    def save_features(self):
        """Save enhanced features to file."""
        logger.info(f"Saving enhanced features to {self.output_path}")
        self.features_df.to_parquet(self.output_path, index=False)
        logger.info(f"✓ Saved {len(self.features_df)} games with {len(self.features_df.columns)} columns")
        
        # Save feature list
        feature_list_path = self.output_path.replace('.parquet', '_feature_list.txt')
        with open(feature_list_path, 'w') as f:
            for col in self.features_df.columns:
                f.write(f"{col}\n")
        logger.info(f"✓ Feature list saved to {feature_list_path}")
        
        return self

    def get_new_features(self):
        """Return list of newly added features."""
        return [
            'home_rest_days',
            'away_rest_days',
            'rest_days_diff',
            'home_is_b2b',
            'away_is_b2b',
            'home_b2b_x_home',
            'away_b2b_x_away',
            'b2b_diff',
            'home_recent_points',
            'away_recent_points',
            'home_recent_allowed',
            'away_recent_allowed',
            'home_recent_margin',
            'away_recent_margin',
            'home_recent_wins',
            'away_recent_wins',
            'recent_points_diff',
            'recent_allowed_diff',
            'recent_margin_diff',
            'recent_wins_diff',
        ]


def main():
    """Run Phase 9."""
    # Paths
    ratings_path = 'data/processed/team_ratings.parquet'
    output_path = 'data/processed/enhanced_features.parquet'
    
    # Build enhanced features
    builder = EnhancedFeaturesBuilder(ratings_path, output_path)
    builder.build_features()
    
    # Save features
    builder.save_features()
    
    # Summary
    new_features = builder.get_new_features()
    logger.info("=" * 70)
    logger.info("PHASE 9: COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Added {len(new_features)} new features:")
    for feat in new_features:
        logger.info(f"  ✓ {feat}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
