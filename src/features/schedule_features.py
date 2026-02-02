"""Schedule-based features including rest days and travel.

This module extracts:
- Rest days before game
- Back-to-back games
- Travel distance (if venue data available)
- Schedule density
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import date, timedelta
from collections import defaultdict


class ScheduleFeatures:
    """Extract schedule-related features."""
    
    def __init__(self, games_df: pd.DataFrame):
        """
        Initialize schedule features.
        
        Args:
            games_df: DataFrame with columns ['date', 'home', 'away', 'game_id']
        """
        self.games_df = games_df.copy()
        self.games_df['date'] = pd.to_datetime(self.games_df['date'])
        self.games_df = self.games_df.sort_values('date')
        
        # Build team game history
        self.team_games = defaultdict(list)
        for _, row in self.games_df.iterrows():
            self.team_games[row['home']].append(row)
            self.team_games[row['away']].append(row)
    
    def get_rest_days(self, team: str, game_date: date) -> int:
        """
        Calculate rest days before a game.
        
        Args:
            team: Team tri-code
            game_date: Date of the game
            
        Returns:
            Number of rest days (0 if no previous game found)
        """
        team_matchups = self.team_games[team]
        
        # Find most recent game before this date
        prev_game = None
        for game in team_matchups:
            game_dt = pd.to_datetime(game['date']).date()
            if game_dt < game_date:
                if prev_game is None or game_dt > pd.to_datetime(prev_game['date']).date():
                    prev_game = game
        
        if prev_game is None:
            return 0  # No previous game
        
        prev_date = pd.to_datetime(prev_game['date']).date()
        rest_days = (game_date - prev_date).days
        
        return max(rest_days, 0)
    
    def is_back_to_back(self, team: str, game_date: date) -> bool:
        """
        Check if team is on a back-to-back.
        
        Args:
            team: Team tri-code
            game_date: Date of the game
            
        Returns:
            True if back-to-back, False otherwise
        """
        return self.get_rest_days(team, game_date) == 0
    
    def get_schedule_density(self, team: str, game_date: date, window: int = 7) -> int:
        """
        Calculate schedule density (games in last N days).
        
        Args:
            team: Team tri-code
            game_date: Date of the game
            window: Number of days to look back
            
        Returns:
            Number of games in the last N days
        """
        team_matchups = self.team_games[team]
        
        window_start = game_date - timedelta(days=window)
        
        count = 0
        for game in team_matchups:
            game_dt = pd.to_datetime(game['date']).date()
            if window_start < game_dt < game_date:
                count += 1
        
        return count
    
    def get_days_since_last_road_game(self, team: str, game_date: date) -> int:
        """
        Calculate days since last road game.
        
        Args:
            team: Team tri-code
            game_date: Date of the game
            
        Returns:
            Days since last road game (large number if not found)
        """
        team_matchups = self.team_games[team]
        
        # Find most recent road game before this date
        prev_road_game = None
        for game in team_matchups:
            game_dt = pd.to_datetime(game['date']).date()
            if game_dt < game_date and game['away'] == team:
                if prev_road_game is None or game_dt > pd.to_datetime(prev_road_game['date']).date():
                    prev_road_game = game
        
        if prev_road_game is None:
            return 999  # No previous road game
        
        prev_date = pd.to_datetime(prev_road_game['date']).date()
        days = (game_date - prev_date).days
        
        return max(days, 0)
    
    def get_days_since_last_home_game(self, team: str, game_date: date) -> int:
        """
        Calculate days since last home game.
        
        Args:
            team: Team tri-code
            game_date: Date of the game
            
        Returns:
            Days since last home game (large number if not found)
        """
        team_matchups = self.team_games[team]
        
        # Find most recent home game before this date
        prev_home_game = None
        for game in team_matchups:
            game_dt = pd.to_datetime(game['date']).date()
            if game_dt < game_date and game['home'] == team:
                if prev_home_game is None or game_dt > pd.to_datetime(prev_home_game['date']).date():
                    prev_home_game = game
        
        if prev_home_game is None:
            return 999  # No previous home game
        
        prev_date = pd.to_datetime(prev_home_game['date']).date()
        days = (game_date - prev_date).days
        
        return max(days, 0)
    
    def get_features_for_game(self, game_date: date, home_team: str, away_team: str) -> Dict[str, float]:
        """
        Get all schedule features for a game.
        
        Args:
            game_date: Date of the game
            home_team: Home team tri-code
            away_team: Away team tri-code
            
        Returns:
            Dictionary of schedule features
        """
        features = {}
        
        # Home team features
        features['home_rest_days'] = self.get_rest_days(home_team, game_date)
        features['home_b2b'] = 1 if self.is_back_to_back(home_team, game_date) else 0
        features['home_schedule_density_7d'] = self.get_schedule_density(home_team, game_date, 7)
        features['home_days_since_last_road'] = self.get_days_since_last_road_game(home_team, game_date)
        features['home_days_since_last_home'] = self.get_days_since_last_home_game(home_team, game_date)
        
        # Away team features
        features['away_rest_days'] = self.get_rest_days(away_team, game_date)
        features['away_b2b'] = 1 if self.is_back_to_back(away_team, game_date) else 0
        features['away_schedule_density_7d'] = self.get_schedule_density(away_team, game_date, 7)
        features['away_days_since_last_road'] = self.get_days_since_last_road_game(away_team, game_date)
        features['away_days_since_last_home'] = self.get_days_since_last_home(away_team, game_date)
        
        # Comparative features
        features['rest_advantage'] = features['home_rest_days'] - features['away_rest_days']
        
        return features


def load_schedule_from_scoreboard(scoreboard_func, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Load schedule data from scoreboard API.
    
    Args:
        scoreboard_func: Function to fetch scoreboard for a date
        start_date: Start date
        end_date: End date
        
    Returns:
        DataFrame with schedule information
    """
    games = []
    current_date = start_date
    
    while current_date <= end_date:
        try:
            day_games = scoreboard_func(current_date, include_live=False)
            for game in day_games:
                games.append({
                    'date': current_date,
                    'game_id': game.game_id,
                    'home': game.home,
                    'away': game.away,
                })
        except Exception as e:
            print(f"Error fetching scoreboard for {current_date}: {e}")
        
        current_date += timedelta(days=1)
    
    return pd.DataFrame(games)


if __name__ == '__main__':
    # Test the module
    print("Testing ScheduleFeatures module...")
    print("\nNote: This module requires actual game data to test properly.")
    print("\nTo use:")
    print("  1. Load schedule data from scoreboard API")
    print("  2. Initialize ScheduleFeatures with games DataFrame")
    print("  3. Call get_features_for_game() for each game")
