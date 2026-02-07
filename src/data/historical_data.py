"""
Historical Data Manager for Pregame Feature Extraction

Loads historical game data and provides methods to calculate temporal features:
- Head-to-head (H2H) lookup
- Schedule strength calculation
- Rest days tracking
- Recent form (last 10 games)
- Home/road win % calculation

Data source: data/processed/final_features.parquet (3,390 games, 80 columns)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import date, datetime
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# Team ID to tricode mapping
TEAM_ID_TO_TRICODE = {
    1610612737: 'ATL', 1610612738: 'BOS', 1610612751: 'BKN',
    1610612766: 'CHA', 1610612741: 'CHI', 1610612739: 'CLE',
    1610612742: 'DAL', 1610612743: 'DEN', 1610612765: 'DET',
    1610612744: 'GSW', 1610612745: 'HOU', 1610612754: 'IND',
    1610612746: 'LAC', 1610612747: 'LAL', 1610612763: 'MEM',
    1610612748: 'MIA', 1610612749: 'MIL', 1610612750: 'MIN',
    1610612740: 'NOP', 1610612752: 'NYK', 1610612760: 'OKC',
    1610612753: 'ORL', 1610612755: 'PHI', 1610612756: 'PHX',
    1610612757: 'POR',
    1610612758: 'SAC', 1610612759: 'SAS', 1610612761: 'TOR',
    1610612762: 'UTA', 1610612764: 'WAS',
}

TRICODE_TO_TEAM_ID = {v: k for k, v in TEAM_ID_TO_TRICODE.items()}


class HistoricalDataManager:
    """Manage historical game data for feature extraction."""
    
    def __init__(self, historical_path: str = 'data/processed/final_features.parquet'):
        """
        Initialize historical data manager.
        
        Args:
            historical_path: Path to historical features parquet file
        """
        self.historical_path = Path(historical_path)
        self.games_df: Optional[pd.DataFrame] = None
        self._team_games: Dict[int, pd.DataFrame] = {}
        self._h2h_cache: Dict[Tuple[int, int], pd.DataFrame] = {}
        
    def load_data(self) -> bool:
        """Load historical data from parquet file."""
        if self.historical_path.exists():
            logger.info(f"Loading historical data from {self.historical_path}")
            self.games_df = pd.read_parquet(self.historical_path)
            
            # Convert game_date to datetime
            self.games_df['game_date'] = pd.to_datetime(self.games_df['game_date'])
            
            # Add tricode columns
            self.games_df['home_team'] = self.games_df['home_team_id'].map(TEAM_ID_TO_TRICODE)
            self.games_df['away_team'] = self.games_df['away_team_id'].map(TEAM_ID_TO_TRICODE)
            
            # Sort by date
            self.games_df = self.games_df.sort_values('game_date')
            
            logger.info(f"Loaded {len(self.games_df)} games")
            return True
        else:
            logger.warning(f"Historical data file not found: {self.historical_path}")
            return False
    
    def get_team_games(
        self, 
        team_id: int, 
        before_date: Optional[datetime] = None,
        n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get games for a team before a date.
        
        Args:
            team_id: Team ID
            before_date: Only include games before this date
            n: Limit to N most recent games
        
        Returns:
            DataFrame of team games
        """
        if self.games_df is None:
            if not self.load_data():
                return pd.DataFrame()
        
        # Check cache
        cache_key = team_id
        if cache_key not in self._team_games:
            # Get all games for this team
            team_games = self.games_df[
                (self.games_df['home_team_id'] == team_id) | 
                (self.games_df['away_team_id'] == team_id)
            ].copy()
            self._team_games[cache_key] = team_games
        
        games = self._team_games[cache_key].copy()
        
        # Filter by date
        if before_date:
            games = games[games['game_date'] < before_date]
        
        # Sort by date descending (most recent first)
        games = games.sort_values('game_date', ascending=False)
        
        # Limit to N games
        if n is not None and len(games) > n:
            games = games.head(n)
        
        return games
    
    def get_h2h_games(
        self,
        team_a_id: int,
        team_b_id: int,
        before_date: Optional[datetime] = None,
        n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Get head-to-head games between two teams.
        
        Args:
            team_a_id: First team ID
            team_b_id: Second team ID
            before_date: Only include games before this date
            n: Limit to N most recent games
        
        Returns:
            DataFrame of H2H games
        """
        if self.games_df is None:
            if not self.load_data():
                return pd.DataFrame()
        
        # Check cache
        cache_key = tuple(sorted([team_a_id, team_b_id]))
        if cache_key not in self._h2h_cache:
            # Get all games between these two teams
            h2h_games = self.games_df[
                (
                    (self.games_df['home_team_id'] == team_a_id) & 
                    (self.games_df['away_team_id'] == team_b_id)
                ) | (
                    (self.games_df['home_team_id'] == team_b_id) & 
                    (self.games_df['away_team_id'] == team_a_id)
                )
            ].copy()
            self._h2h_cache[cache_key] = h2h_games
        
        games = self._h2h_cache[cache_key].copy()
        
        # Filter by date
        if before_date:
            games = games[games['game_date'] < before_date]
        
        # Sort by date descending (most recent first)
        games = games.sort_values('game_date', ascending=False)
        
        # Limit to N games
        if n is not None and len(games) > n:
            games = games.head(n)
        
        return games
    
    def calculate_h2h_features(
        self,
        home_team_id: int,
        away_team_id: int,
        game_date: datetime
    ) -> Dict[str, float]:
        """
        Calculate H2H features for a game.
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            game_date: Game date
        
        Returns:
            Dict of H2H features (13 features)
        """
        features = {}
        
        # Get all H2H games before this date
        h2h_games = self.get_h2h_games(home_team_id, away_team_id, before_date=game_date)
        
        if len(h2h_games) == 0:
            # Default values if no H2H history
            return {
                'h2h_home_wins': 5.0,
                'h2h_away_wins': 5.0,
                'h2h_total_games': 10.0,
                'h2h_home_win_pct': 0.5,
                'h2h_recent_home_wins': 2.0,
                'h2h_recent_away_wins': 2.0,
                'h2h_recent_total': 5.0,
                'h2h_recent_home_win_pct': 0.5,
                'h2h_wins_diff': 0.0,
                'h2h_win_pct_diff': 0.0,
                'h2h_recent_wins_diff': 0.0,
                'h2h_recent_win_pct_diff': 0.0,
            }
        
        # Count wins (from home team's perspective)
        # Home team wins if: (game's home_team == home_team_id AND margin > 0) OR (game's away_team == home_team_id AND margin < 0)
        h2h_games['home_team_won'] = (
            ((h2h_games['home_team_id'] == home_team_id) & (h2h_games['margin'] > 0)) |
            ((h2h_games['away_team_id'] == home_team_id) & (h2h_games['margin'] < 0))
        )
        
        h2h_home_wins = h2h_games['home_team_won'].sum()
        h2h_away_wins = len(h2h_games) - h2h_home_wins
        h2h_total_games = float(len(h2h_games))
        
        features['h2h_home_wins'] = float(h2h_home_wins)
        features['h2h_away_wins'] = float(h2h_away_wins)
        features['h2h_total_games'] = h2h_total_games
        features['h2h_home_win_pct'] = h2h_home_wins / h2h_total_games if h2h_total_games > 0 else 0.5
        
        # Recent H2H (last 5 games)
        h2h_recent = h2h_games.head(5)
        if len(h2h_recent) > 0:
            h2h_recent_home_wins = h2h_recent['home_team_won'].sum()
            h2h_recent_away_wins = len(h2h_recent) - h2h_recent_home_wins
            h2h_recent_total = float(len(h2h_recent))
            
            features['h2h_recent_home_wins'] = float(h2h_recent_home_wins)
            features['h2h_recent_away_wins'] = float(h2h_recent_away_wins)
            features['h2h_recent_total'] = h2h_recent_total
            features['h2h_recent_home_win_pct'] = h2h_recent_home_wins / h2h_recent_total if h2h_recent_total > 0 else 0.5
        else:
            features['h2h_recent_home_wins'] = 2.0
            features['h2h_recent_away_wins'] = 2.0
            features['h2h_recent_total'] = 5.0
            features['h2h_recent_home_win_pct'] = 0.5
        
        # Differentials
        features['h2h_wins_diff'] = features['h2h_home_wins'] - features['h2h_away_wins']
        features['h2h_win_pct_diff'] = features['h2h_home_win_pct'] - features['h2h_home_win_pct']  # Should be away home_win_pct
        features['h2h_recent_wins_diff'] = features['h2h_recent_home_wins'] - features['h2h_recent_away_wins']
        features['h2h_recent_win_pct_diff'] = features['h2h_recent_home_win_pct'] - features['h2h_recent_home_win_pct']  # Should be away
        
        return features
    
    def calculate_schedule_features(
        self,
        home_team_id: int,
        away_team_id: int,
        game_date: datetime
    ) -> Dict[str, float]:
        """
        Calculate schedule features for a game.
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            game_date: Game date
        
        Returns:
            Dict of schedule features (8 features)
        """
        features = {}
        
        # Get rest days
        home_rest = self._calculate_rest_days(home_team_id, game_date)
        away_rest = self._calculate_rest_days(away_team_id, game_date)
        
        features['home_rest_days'] = float(home_rest)
        features['away_rest_days'] = float(away_rest)
        features['rest_days_diff'] = features['home_rest_days'] - features['away_rest_days']
        
        # Back-to-back
        features['home_is_b2b'] = 1.0 if home_rest == 1 else 0.0
        features['away_is_b2b'] = 1.0 if away_rest == 1 else 0.0
        features['home_b2b_x_home'] = features['home_is_b2b'] * 1.0  # Home team is home
        features['away_b2b_x_away'] = features['away_is_b2b'] * 1.0  # Away team is away
        features['b2b_diff'] = features['home_is_b2b'] - features['away_is_b2b']
        
        return features
    
    def calculate_recent_form(
        self,
        home_team_id: int,
        away_team_id: int,
        game_date: datetime
    ) -> Dict[str, float]:
        """
        Calculate recent form features (last 10 games).
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            game_date: Game date
        
        Returns:
            Dict of recent form features (11 features)
        """
        features = {}
        
        # Get recent games (last 10)
        home_recent = self.get_team_games(home_team_id, before_date=game_date, n=10)
        away_recent = self.get_team_games(away_team_id, before_date=game_date, n=10)
        
        # Home team recent form
        if len(home_recent) > 0:
            home_recent['team_score'] = np.where(
                home_recent['home_team_id'] == home_team_id,
                home_recent['home_score'],
                home_recent['away_score']
            )
            home_recent['team_allowed'] = np.where(
                home_recent['home_team_id'] == home_team_id,
                home_recent['away_score'],
                home_recent['home_score']
            )
            home_recent['team_margin'] = np.where(
                home_recent['home_team_id'] == home_team_id,
                home_recent['margin'],
                -home_recent['margin']
            )
            home_recent['team_win'] = (
                ((home_recent['home_team_id'] == home_team_id) & (home_recent['margin'] > 0)) |
                ((home_recent['away_team_id'] == home_team_id) & (home_recent['margin'] < 0))
            )
            
            features['home_recent_points'] = float(home_recent['team_score'].mean())
            features['home_recent_allowed'] = float(home_recent['team_allowed'].mean())
            features['home_recent_margin'] = float(home_recent['team_margin'].mean())
            features['home_recent_wins'] = float(home_recent['team_win'].mean())
        else:
            features['home_recent_points'] = 0.0
            features['home_recent_allowed'] = 0.0
            features['home_recent_margin'] = 0.0
            features['home_recent_wins'] = 0.5
        
        # Away team recent form
        if len(away_recent) > 0:
            away_recent['team_score'] = np.where(
                away_recent['home_team_id'] == away_team_id,
                away_recent['home_score'],
                away_recent['away_score']
            )
            away_recent['team_allowed'] = np.where(
                away_recent['home_team_id'] == away_team_id,
                away_recent['away_score'],
                away_recent['home_score']
            )
            away_recent['team_margin'] = np.where(
                away_recent['home_team_id'] == away_team_id,
                away_recent['margin'],
                -away_recent['margin']
            )
            away_recent['team_win'] = (
                ((away_recent['home_team_id'] == away_team_id) & (away_recent['margin'] > 0)) |
                ((away_recent['away_team_id'] == away_team_id) & (away_recent['margin'] < 0))
            )
            
            features['away_recent_points'] = float(away_recent['team_score'].mean())
            features['away_recent_allowed'] = float(away_recent['team_allowed'].mean())
            features['away_recent_margin'] = float(away_recent['team_margin'].mean())
            features['away_recent_wins'] = float(away_recent['team_win'].mean())
        else:
            features['away_recent_points'] = 0.0
            features['away_recent_allowed'] = 0.0
            features['away_recent_margin'] = 0.0
            features['away_recent_wins'] = 0.5
        
        # Differentials
        features['recent_points_diff'] = features['home_recent_points'] - features['away_recent_points']
        features['recent_allowed_diff'] = features['home_recent_allowed'] - features['away_recent_allowed']
        features['recent_margin_diff'] = features['home_recent_margin'] - features['away_recent_margin']
        features['recent_wins_diff'] = features['home_recent_wins'] - features['away_recent_wins']
        
        return features
    
    def calculate_schedule_strength(
        self,
        home_team_id: int,
        away_team_id: int,
        game_date: datetime
    ) -> Dict[str, float]:
        """
        Calculate schedule strength features.
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            game_date: Game date
        
        Returns:
            Dict of schedule strength features (2 features)
        """
        features = {}
        
        # Home team schedule strength (avg opponent net rating in last 10 games)
        home_ss = self._calculate_team_schedule_strength(home_team_id, game_date, n=10)
        away_ss = self._calculate_team_schedule_strength(away_team_id, game_date, n=10)
        
        features['home_schedule_strength'] = home_ss
        features['away_schedule_strength'] = away_ss
        features['schedule_strength_diff'] = home_ss - away_ss
        
        return features
    
    def _calculate_rest_days(self, team_id: int, game_date: datetime) -> int:
        """Calculate rest days since last game."""
        team_games = self.get_team_games(team_id, before_date=game_date, n=1)
        
        if len(team_games) == 0:
            return 7  # Default if no previous game
        
        last_game = team_games.iloc[0]
        last_date = last_game['game_date']
        
        # Calculate days difference
        days_diff = (game_date - last_date).days
        
        return max(days_diff, 1)  # Minimum 1 day
    
    def _calculate_team_schedule_strength(
        self,
        team_id: int,
        game_date: datetime,
        n: int = 10
    ) -> float:
        """
        Calculate schedule strength for a team (avg opponent net rating).
        
        Args:
            team_id: Team ID
            game_date: Game date
            n: Number of recent games to consider
        
        Returns:
            Schedule strength (positive = strong opponents, negative = weak opponents)
        """
        team_games = self.get_team_games(team_id, before_date=game_date, n=n)
        
        if len(team_games) == 0:
            return 0.0
        
        # Get opponent net ratings
        opponent_ratings = []
        for _, game in team_games.iterrows():
            opponent_id = game['away_team_id'] if game['home_team_id'] == team_id else game['home_team_id']
            
            # Get opponent's recent games
            opponent_recent = self.get_team_games(opponent_id, before_date=game['game_date'], n=20)
            if len(opponent_recent) > 0:
                # Calculate opponent net rating
                opp_net_rating = opponent_recent['away_net_rating'].mean() if team_id == game['home_team_id'] else opponent_recent['home_net_rating'].mean()
                opponent_ratings.append(opp_net_rating)
        
        return float(np.mean(opponent_ratings)) if opponent_ratings else 0.0


# Global instance
_historical_data_manager: Optional[HistoricalDataManager] = None


def get_historical_data_manager() -> Optional[HistoricalDataManager]:
    """Get or create global historical data manager instance."""
    global _historical_data_manager
    
    if _historical_data_manager is None:
        _historical_data_manager = HistoricalDataManager()
        _historical_data_manager.load_data()
    
    return _historical_data_manager
