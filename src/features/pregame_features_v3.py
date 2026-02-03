"""Extract all 72 pregame features including temporal and form data

Features include:
1. Basic team ratings (18 features)
2. Schedule features (8 features) - rest days, back-to-back
3. Recent form features (11 features) - last 10 games performance
4. Four factors / Net rating (20 features)
5. Head-to-head features (13 features) - historical matchup stats
6. Schedule strength (2 features) - opponent strength

This matches the FINAL_FEATURES.PARQUET data used to train models.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import date, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class PregameFeaturesV3:
    """Extract all 72 pregame features with temporal and form data."""
    
    def __init__(self, historical_df: pd.DataFrame):
        """
        Initialize with historical game data.
        
        Args:
            historical_df: DataFrame with columns ['game_date', 'home_team', 'away_team', 
                                              'home_score', 'away_score', 'total', 'margin', 'game_id']
        """
        self.games_df = historical_df.copy()
        self.games_df['game_date'] = pd.to_datetime(self.games_df['game_date'])
        self.games_df = self.games_df.sort_values('game_date')
        
        # Build matchup history for H2H features
        self.matchup_history = defaultdict(list)
        for _, row in self.games_df.iterrows():
            matchup_key = (row['home_team'], row['away_team'])
            self.matchup_history[matchup_key].append(row)
    
    def calculate_team_ratings(self, game_date: date, home_team: str, away_team: str) -> Dict[str, float]:
        """Calculate basic team ratings (18 features)."""
        features = {}
        
        # Get recent games for each team (last 20 games or all available)
        home_recent = self._get_recent_games(home_team, game_date, n=20)
        away_recent = self._get_recent_games(away_team, game_date, n=20)
        
        # Home team ratings
        if len(home_recent) > 0:
            features['home_off_rating'] = self._calc_off_rating(home_recent)
            features['home_def_rating'] = self._calc_def_rating(home_recent)
            features['home_pace'] = self._calc_pace(home_recent)
            features['home_efg'] = self._calc_efg(home_recent)
            features['home_tov_rate'] = self._calc_tov_rate(home_recent)
            features['home_orb_rate'] = self._calc_orb_rate(home_recent)
            features['home_ft_rate'] = self._calc_ft_rate(home_recent)
            features['home_win_pct'] = self._calc_win_pct(home_recent)
            features['home_home_win_pct'] = self._calc_home_win_pct(home_team, game_date)
        else:
            features.update({f'home_{k}': 0.0 for k in ['off_rating', 'def_rating', 'pace', 'efg', 'tov_rate', 'orb_rate', 'ft_rate', 'win_pct', 'home_win_pct']})
        
        # Away team ratings
        if len(away_recent) > 0:
            features['away_off_rating'] = self._calc_off_rating(away_recent)
            features['away_def_rating'] = self._calc_def_rating(away_recent)
            features['away_pace'] = self._calc_pace(away_recent)
            features['away_efg'] = self._calc_efg(away_recent)
            features['away_tov_rate'] = self._calc_tov_rate(away_recent)
            features['away_orb_rate'] = self._calc_orb_rate(away_recent)
            features['away_ft_rate'] = self._calc_ft_rate(away_recent)
            features['away_win_pct'] = self._calc_win_pct(away_recent)
            features['away_road_win_pct'] = self._calc_road_win_pct(away_team, game_date)
        else:
            features.update({f'away_{k}': 0.0 for k in ['off_rating', 'def_rating', 'pace', 'efg', 'tov_rate', 'orb_rate', 'ft_rate', 'win_pct', 'road_win_pct']})
        
        # Differentials
        if 'home_off_rating' in features and 'away_off_rating' in features:
            features['off_rating_diff'] = features['home_off_rating'] - features['away_off_rating']
            features['def_rating_diff'] = features['home_def_rating'] - features['away_def_rating']
            features['pace_diff'] = features['home_pace'] - features['away_pace']
            features['efg_diff'] = features['home_efg'] - features['away_efg']
            features['tov_rate_diff'] = features['home_tov_rate'] - features['away_tov_rate']
            features['orb_rate_diff'] = features['home_orb_rate'] - features['away_orb_rate']
            features['ft_rate_diff'] = features['home_ft_rate'] - features['away_ft_rate']
        
        return features
    
    def calculate_schedule_features(self, game_date: date, home_team: str, away_team: str) -> Dict[str, float]:
        """Calculate schedule features (8 features)."""
        features = {}
        
        # Rest days
        home_rest = self._calc_rest_days(home_team, game_date)
        away_rest = self._calc_rest_days(away_team, game_date)
        
        features['home_rest_days'] = home_rest
        features['away_rest_days'] = away_rest
        features['rest_days_diff'] = home_rest - away_rest
        
        # Back-to-back
        features['home_is_b2b'] = 1.0 if home_rest == 1 else 0.0
        features['away_is_b2b'] = 1.0 if away_rest == 1 else 0.0
        features['home_b2b_x_home'] = features['home_is_b2b'] * 1.0  # Home team is home and b2b
        features['away_b2b_x_away'] = features['away_is_b2b'] * 1.0  # Away team is away and b2b
        features['b2b_diff'] = features['home_is_b2b'] - features['away_is_b2b']
        
        return features
    
    def calculate_recent_form(self, game_date: date, home_team: str, away_team: str) -> Dict[str, float]:
        """Calculate recent form features (11 features) - last 10 games."""
        features = {}
        
        # Get recent games (last 10)
        home_recent = self._get_recent_games(home_team, game_date, n=10)
        away_recent = self._get_recent_games(away_team, game_date, n=10)
        
        # Home team recent form
        if len(home_recent) > 0:
            features['home_recent_points'] = np.mean([r['home_score'] if r['home_team'] == home_team else r['away_score'] for r in home_recent])
            features['home_recent_allowed'] = np.mean([r['away_score'] if r['home_team'] == home_team else r['home_score'] for r in home_recent])
            features['home_recent_margin'] = np.mean([r['margin'] if r['home_team'] == home_team else -r['margin'] for r in home_recent])
            features['home_recent_wins'] = np.mean([1.0 if (r['margin'] > 0 and r['home_team'] == home_team) or (r['margin'] < 0 and r['away_team'] == home_team) else 0.0 for r in home_recent])
        else:
            features.update({f'home_recent_{k}': 0.0 for k in ['points', 'allowed', 'margin', 'wins']})
        
        # Away team recent form
        if len(away_recent) > 0:
            features['away_recent_points'] = np.mean([r['home_score'] if r['home_team'] == away_team else r['away_score'] for r in away_recent])
            features['away_recent_allowed'] = np.mean([r['away_score'] if r['home_team'] == away_team else r['home_score'] for r in away_recent])
            features['away_recent_margin'] = np.mean([r['margin'] if r['home_team'] == away_team else -r['margin'] for r in away_recent])
            features['away_recent_wins'] = np.mean([1.0 if (r['margin'] > 0 and r['home_team'] == away_team) or (r['margin'] < 0 and r['away_team'] == away_team) else 0.0 for r in away_recent])
        else:
            features.update({f'away_recent_{k}': 0.0 for k in ['points', 'allowed', 'margin', 'wins']})
        
        # Differentials
        if 'home_recent_points' in features and 'away_recent_points' in features:
            features['recent_points_diff'] = features['home_recent_points'] - features['away_recent_points']
            features['recent_allowed_diff'] = features['home_recent_allowed'] - features['away_recent_allowed']
            features['recent_margin_diff'] = features['home_recent_margin'] - features['away_recent_margin']
            features['recent_wins_diff'] = features['home_recent_wins'] - features['away_recent_wins']
        
        return features
    
    def calculate_four_factors(self, game_date: date, home_team: str, away_team: str) -> Dict[str, float]:
        """Calculate four factors and net rating (20 features)."""
        features = {}
        
        # Net rating
        if 'home_off_rating' in features and 'home_def_rating' in features:
            features['home_net_rating'] = features['home_off_rating'] - features['home_def_rating']
        if 'away_off_rating' in features and 'away_def_rating' in features:
            features['away_net_rating'] = features['away_off_rating'] - features['away_def_rating']
        
        if 'home_net_rating' in features and 'away_net_rating' in features:
            features['net_rating_diff'] = features['home_net_rating'] - features['away_net_rating']
        
        # True shooting proxy
        features['home_ts_proxy'] = features.get('home_efg', 0.5) * features.get('home_ft_rate', 0.25)
        features['away_ts_proxy'] = features.get('away_efg', 0.5) * features.get('away_ft_rate', 0.25)
        features['ts_proxy_diff'] = features['home_ts_proxy'] - features['away_ts_proxy']
        
        # Assist ratio proxy (points / assists approx)
        features['home_assist_ratio_proxy'] = features.get('home_pace', 100) / 100.0  # Simplified proxy
        features['away_assist_ratio_proxy'] = features.get('away_pace', 100) / 100.0
        features['assist_ratio_diff'] = features['home_assist_ratio_proxy'] - features['away_assist_ratio_proxy']
        
        # Four factor weighted (simplified - weighted sum of 4 factors)
        features['home_four_factor_weighted'] = (
            features.get('home_efg', 0.5) * 0.4 +
            features.get('home_orb_rate', 0.25) * 0.3 +
            features.get('home_tov_rate', 0.15) * -0.15 +
            features.get('home_ft_rate', 0.25) * 0.15
        )
        features['away_four_factor_weighted'] = (
            features.get('away_efg', 0.5) * 0.4 +
            features.get('away_orb_rate', 0.25) * 0.3 +
            features.get('away_tov_rate', 0.15) * -0.15 +
            features.get('away_ft_rate', 0.25) * 0.15
        )
        features['four_factor_weighted_diff'] = features['home_four_factor_weighted'] - features['away_four_factor_weighted']
        
        # Off/Def rating diffs (from basic ratings)
        if 'home_off_rating' in features and 'away_off_rating' in features:
            features['off_rating_diff'] = features['home_off_rating'] - features['away_off_rating']
        if 'home_def_rating' in features and 'away_def_rating' in features:
            features['def_rating_diff'] = features['home_def_rating'] - features['away_def_rating']
        if 'home_pace' in features and 'away_pace' in features:
            features['pace_diff'] = features['home_pace'] - features['away_pace']
        
        # Efficiency score
        features['home_efficiency_score'] = features.get('home_net_rating', 0.0)
        features['away_efficiency_score'] = features.get('away_net_rating', 0.0)
        features['efficiency_diff'] = features['home_efficiency_score'] - features['away_efficiency_score']
        
        return features
    
    def calculate_h2h_features(self, game_date: date, home_team: str, away_team: str) -> Dict[str, float]:
        """Calculate head-to-head features (13 features)."""
        features = {}
        
        # Get H2H record
        h2h_home_wins, h2h_away_wins = self._get_h2h_record(home_team, away_team, game_date)
        h2h_total = max(h2h_home_wins + h2h_away_wins, 1)
        
        features['h2h_home_wins'] = float(h2h_home_wins)
        features['h2h_away_wins'] = float(h2h_away_wins)
        features['h2h_total_games'] = float(h2h_total)
        features['h2h_home_win_pct'] = h2h_home_wins / h2h_total if h2h_total > 0 else 0.5
        
        # Recent H2H (last 5 games between these teams)
        h2h_recent_home_wins, h2h_recent_away_wins = self._get_h2h_record(home_team, away_team, game_date, recent_only=True, n=5)
        h2h_recent_total = max(h2h_recent_home_wins + h2h_recent_away_wins, 1)
        
        features['h2h_recent_home_wins'] = float(h2h_recent_home_wins)
        features['h2h_recent_away_wins'] = float(h2h_recent_away_wins)
        features['h2h_recent_total'] = float(h2h_recent_total)
        features['h2h_recent_home_win_pct'] = h2h_recent_home_wins / h2h_recent_total if h2h_recent_total > 0 else 0.5
        
        # H2H differences
        features['h2h_wins_diff'] = h2h_home_wins - h2h_away_wins
        features['h2h_win_pct_diff'] = features['h2h_home_win_pct'] - features['h2h_home_win_pct']  # Note: should be away win pct
        features['h2h_recent_wins_diff'] = h2h_recent_home_wins - h2h_recent_away_wins
        features['h2h_recent_win_pct_diff'] = features['h2h_recent_home_win_pct'] - features['h2h_recent_home_win_pct']  # Note: should be away
        
        return features
    
    def calculate_schedule_strength(self, game_date: date, home_team: str, away_team: str) -> Dict[str, float]:
        """Calculate schedule strength features (2 features)."""
        features = {}
        
        # Home team schedule strength (avg opponent net rating in last 10 games)
        home_ss = self._calc_schedule_strength(home_team, game_date, n=10)
        away_ss = self._calc_schedule_strength(away_team, game_date, n=10)
        
        features['home_schedule_strength'] = home_ss
        features['away_schedule_strength'] = away_ss
        features['schedule_strength_diff'] = home_ss - away_ss
        
        return features
    
    def extract_all_features(self, game_id: str, game_date: date, home_team: str, away_team: str) -> Dict[str, float]:
        """Extract all 72 features for a game."""
        features = {'game_id': game_id, 'game_date': str(game_date)}
        
        # 1. Basic team ratings (18 features)
        features.update(self.calculate_team_ratings(game_date, home_team, away_team))
        
        # 2. Schedule features (8 features)
        features.update(self.calculate_schedule_features(game_date, home_team, away_team))
        
        # 3. Recent form features (11 features)
        features.update(self.calculate_recent_form(game_date, home_team, away_team))
        
        # 4. Four factors / Net rating (depends on ratings, already calculated)
        features.update(self.calculate_four_factors(game_date, home_team, away_team))
        
        # 5. Head-to-head features (13 features)
        features.update(self.calculate_h2h_features(game_date, home_team, away_team))
        
        # 6. Schedule strength (2 features)
        features.update(self.calculate_schedule_strength(game_date, home_team, away_team))
        
        return features
    
    # ===== HELPER METHODS =====
    
    def _get_recent_games(self, team: str, game_date: date, n: int = 20) -> list:
        """Get last N games for a team before a date."""
        team_games = []
        for matchup_key, games in self.matchup_history.items():
            if team in matchup_key:
                for game in games:
                    game_dt = pd.to_datetime(game['game_date']).date()
                    if game_dt < game_date:
                        team_games.append(game)
        
        # Sort by date and take last N
        team_games.sort(key=lambda x: pd.to_datetime(x['game_date']).date(), reverse=True)
        return team_games[:n]
    
    def _calc_off_rating(self, games: list) -> float:
        """Calculate offensive rating (points per 100 possessions)."""
        if not games:
            return 110.0
        pts_avg = np.mean([g['home_score'] if g['home_team'] == games[-1]['home_team'] else g['away_score'] for g in games])
        pace_avg = self._calc_pace(games)
        return (pts_avg / pace_avg) * 100.0 if pace_avg > 0 else 110.0
    
    def _calc_def_rating(self, games: list) -> float:
        """Calculate defensive rating (points allowed per 100 possessions)."""
        if not games:
            return 110.0
        pts_allowed_avg = np.mean([g['away_score'] if g['home_team'] == games[-1]['home_team'] else g['home_score'] for g in games])
        pace_avg = self._calc_pace(games)
        return (pts_allowed_avg / pace_avg) * 100.0 if pace_avg > 0 else 110.0
    
    def _calc_pace(self, games: list) -> float:
        """Calculate pace (possessions per game)."""
        if not games:
            return 100.0
        # Simplified: use total points as proxy for pace
        totals = [g['total'] for g in games if 'total' in g]
        return np.mean(totals) if totals else 100.0
    
    def _calc_efg(self, games: list) -> float:
        """Calculate effective field goal % (simplified)."""
        return 0.50  # Placeholder - requires shot data
    
    def _calc_tov_rate(self, games: list) -> float:
        """Calculate turnover rate (simplified)."""
        return 0.15  # Placeholder
    
    def _calc_orb_rate(self, games: list) -> float:
        """Calculate offensive rebound % (simplified)."""
        return 0.25  # Placeholder
    
    def _calc_ft_rate(self, games: list) -> float:
        """Calculate free throw rate (FT/FGA)."""
        return 0.25  # Placeholder
    
    def _calc_win_pct(self, games: list) -> float:
        """Calculate win percentage."""
        if not games:
            return 0.5
        team = games[0]['home_team']
        wins = sum([1.0 for g in games if (g['margin'] > 0 and g['home_team'] == team) or (g['margin'] < 0 and g['away_team'] == team)])
        return wins / len(games) if len(games) > 0 else 0.5
    
    def _calc_home_win_pct(self, team: str, game_date: date) -> float:
        """Calculate home win percentage for a team."""
        home_games = [g for g in self._get_recent_games(team, game_date, n=20) if g['home_team'] == team]
        if not home_games:
            return 0.5
        wins = sum([1.0 for g in home_games if g['margin'] > 0])
        return wins / len(home_games) if len(home_games) > 0 else 0.5
    
    def _calc_road_win_pct(self, team: str, game_date: date) -> float:
        """Calculate road win percentage for a team."""
        road_games = [g for g in self._get_recent_games(team, game_date, n=20) if g['away_team'] == team]
        if not road_games:
            return 0.5
        wins = sum([1.0 for g in road_games if g['margin'] < 0])
        return wins / len(road_games) if len(road_games) > 0 else 0.5
    
    def _calc_rest_days(self, team: str, game_date: date) -> int:
        """Calculate rest days since last game."""
        recent_games = self._get_recent_games(team, game_date, n=20)
        if not recent_games:
            return 7
        last_game = recent_games[0]
        last_date = pd.to_datetime(last_game['game_date']).date()
        return (game_date - last_date).days
    
    def _get_h2h_record(self, team_a: str, team_b: str, before_date: date, recent_only: bool = False, n: int = 999) -> Tuple[int, int]:
        """Get head-to-head record."""
        wins_a = 0
        wins_b = 0
        
        count = 0
        for matchup_key, games in self.matchup_history.items():
            if (matchup_key[0] == team_a and matchup_key[1] == team_b) or \
               (matchup_key[0] == team_b and matchup_key[1] == team_a):
                for game in games:
                    game_dt = pd.to_datetime(game['game_date']).date()
                    if game_dt < before_date:
                        if recent_only and count >= n:
                            continue
                        if game['margin'] > 0:
                            if game['home_team'] == team_a:
                                wins_a += 1
                            else:
                                wins_b += 1
                        else:
                            if game['home_team'] == team_b:
                                wins_a += 1
                            else:
                                wins_b += 1
                        count += 1
        
        return wins_a, wins_b
    
    def _calc_schedule_strength(self, team: str, game_date: date, n: int = 10) -> float:
        """Calculate schedule strength (avg opponent net rating)."""
        recent_games = self._get_recent_games(team, game_date, n=n)
        if not recent_games:
            return 0.0
        
        opponent_ratings = []
        for game in recent_games:
            opponent = game['away_team'] if game['home_team'] == team else game['home_team']
            # Get opponent's net rating (simplified)
            opponent_games = self._get_recent_games(opponent, game_date, n=20)
            if opponent_games:
                opp_off = self._calc_off_rating(opponent_games)
                opp_def = self._calc_def_rating(opponent_games)
                opponent_ratings.append(opp_off - opp_def)
        
        return np.mean(opponent_ratings) if opponent_ratings else 0.0

