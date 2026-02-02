"""Advanced features including head-to-head and historical performance.

This module extracts:
- Head-to-head record
- Recent form
- Season series record
- Matchup-specific trends
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import date, timedelta
from collections import defaultdict


class AdvancedFeatures:
    """Extract advanced matchup and historical features."""
    
    def __init__(self, games_df: pd.DataFrame):
        """
        Initialize advanced features.
        
        Args:
            games_df: DataFrame with columns ['date', 'home', 'away', 'game_id', 'total', 'margin', 'winner']
        """
        self.games_df = games_df.copy()
        self.games_df['date'] = pd.to_datetime(self.games_df['date'])
        self.games_df = self.games_df.sort_values('date')
        
        # Build matchup history
        self.matchup_history = defaultdict(list)
        for _, row in self.games_df.iterrows():
            matchup_key = (row['home'], row['away'])
            self.matchup_history[matchup_key].append(row)
    
    def get_h2h_record(self, team_a: str, team_b: str, before_date: date) -> Tuple[int, int]:
        """
        Get head-to-head record between two teams before a date.
        
        Args:
            team_a: First team
            team_b: Second team
            before_date: Date to look before
            
        Returns:
            Tuple of (wins_a, wins_b)
        """
        wins_a = 0
        wins_b = 0
        
        # Check both directions of matchup
        for matchup_key, games in self.matchup_history.items():
            if (matchup_key[0] == team_a and matchup_key[1] == team_b) or \
               (matchup_key[0] == team_b and matchup_key[1] == team_a):
                for game in games:
                    game_dt = pd.to_datetime(game['date']).date()
                    if game_dt < before_date:
                        if game['winner'] == team_a:
                            wins_a += 1
                        elif game['winner'] == team_b:
                            wins_b += 1
        
        return wins_a, wins_b
    
    def get_recent_form(self, team: str, before_date: date, last_n: int = 10) -> Dict[str, float]:
        """
        Get recent form of a team (last N games).
        
        Args:
            team: Team tri-code
            before_date: Date to look before
            last_n: Number of recent games to consider
            
        Returns:
            Dictionary with recent form metrics
        """
        form_games = []
        
        # Find all games involving this team before the date
        for matchup_key, games in self.matchup_history.items():
            if team in matchup_key:
                for game in games:
                    game_dt = pd.to_datetime(game['date']).date()
                    if game_dt < before_date:
                        form_games.append(game)
        
        # Sort by date and take last N
        form_games.sort(key=lambda x: pd.to_datetime(x['date']).date(), reverse=True)
        form_games = form_games[:last_n]
        
        if len(form_games) == 0:
            return {
                'wins': 0,
                'losses': 0,
                'win_pct': 0.5,
                'avg_points_for': 110,
                'avg_points_against': 110,
            }
        
        wins = 0
        losses = 0
        points_for = []
        points_against = []
        
        for game in form_games:
            if game['winner'] == team:
                wins += 1
            else:
                losses += 1
            
            # Extract points for/against
            if game['home'] == team:
                pts_for = game.get('home_pts', 110)
                pts_against = game.get('away_pts', 110)
            else:
                pts_for = game.get('away_pts', 110)
                pts_against = game.get('home_pts', 110)
            
            points_for.append(pts_for)
            points_against.append(pts_against)
        
        return {
            'wins': wins,
            'losses': losses,
            'win_pct': wins / max(wins + losses, 1),
            'avg_points_for': np.mean(points_for) if points_for else 110,
            'avg_points_against': np.mean(points_against) if points_against else 110,
            'point_diff': np.mean(points_for) - np.mean(points_against) if points_for else 0,
        }
    
    def get_season_series_record(self, home_team: str, away_team: str, season_start: date, before_date: date) -> Tuple[int, int]:
        """
        Get season series record between two teams.
        
        Args:
            home_team: Home team
            away_team: Away team
            season_start: Season start date
            before_date: Date to look before
            
        Returns:
            Tuple of (home_wins, away_wins)
        """
        home_wins = 0
        away_wins = 0
        
        # Check both directions of matchup
        for matchup_key, games in self.matchup_history.items():
            if (matchup_key[0] == home_team and matchup_key[1] == away_team) or \
               (matchup_key[0] == away_team and matchup_key[1] == home_team):
                for game in games:
                    game_dt = pd.to_datetime(game['date']).date()
                    if season_start <= game_dt < before_date:
                        if game['winner'] == home_team:
                            home_wins += 1
                        elif game['winner'] == away_team:
                            away_wins += 1
        
        return home_wins, away_wins
    
    def get_features_for_game(self, game_date: date, home_team: str, away_team: str, season_start: date) -> Dict[str, float]:
        """
        Get all advanced features for a game.
        
        Args:
            game_date: Date of the game
            home_team: Home team tri-code
            away_team: Away team tri-code
            season_start: Season start date
            
        Returns:
            Dictionary of advanced features
        """
        features = {}
        
        # Head-to-head
        h2h_home_wins, h2h_away_wins = self.get_h2h_record(home_team, away_team, game_date)
        features['h2h_home_wins'] = h2h_home_wins
        features['h2h_away_wins'] = h2h_away_wins
        h2h_total = max(h2h_home_wins + h2h_away_wins, 1)
        features['h2h_home_win_pct'] = h2h_home_wins / h2h_total
        
        # Home team recent form
        home_form = self.get_recent_form(home_team, game_date, last_n=10)
        features['home_recent_wins'] = home_form['wins']
        features['home_recent_win_pct'] = home_form['win_pct']
        features['home_recent_avg_pts_for'] = home_form['avg_points_for']
        features['home_recent_avg_pts_against'] = home_form['avg_points_against']
        features['home_recent_point_diff'] = home_form['point_diff']
        
        # Away team recent form
        away_form = self.get_recent_form(away_team, game_date, last_n=10)
        features['away_recent_wins'] = away_form['wins']
        features['away_recent_win_pct'] = away_form['win_pct']
        features['away_recent_avg_pts_for'] = away_form['avg_points_for']
        features['away_recent_avg_pts_against'] = away_form['avg_points_against']
        features['away_recent_point_diff'] = away_form['point_diff']
        
        # Season series
        season_home_wins, season_away_wins = self.get_season_series_record(home_team, away_team, season_start, game_date)
        features['season_series_home_wins'] = season_home_wins
        features['season_series_away_wins'] = season_away_wins
        season_total = max(season_home_wins + season_away_wins, 1)
        features['season_series_home_win_pct'] = season_home_wins / season_total
        
        # Comparative features
        features['recent_win_pct_diff'] = features['home_recent_win_pct'] - features['away_recent_win_pct']
        features['recent_point_diff_diff'] = features['home_recent_point_diff'] - features['away_recent_point_diff']
        
        return features


def load_games_from_backtest(backtest_file: str) -> pd.DataFrame:
    """
    Load games from a backtest file.
    
    Args:
        backtest_file: Path to CSV file with backtest results
        
    Returns:
        DataFrame with game information
    """
    df = pd.read_csv(backtest_file)
    
    # Add derived columns
    df['winner'] = df.apply(
        lambda row: row['home'] if row['actual_margin'] > 0 else row['away'],
        axis=1
    )
    
    return df


if __name__ == '__main__':
    # Test the module
    print("Testing AdvancedFeatures module...")
    print("\nNote: This module requires actual game data to test properly.")
    print("\nTo use:")
    print("  1. Load game data from backtest file or API")
    print("  2. Initialize AdvancedFeatures with games DataFrame")
    print("  3. Call get_features_for_game() for each game")
