"""Team-level feature extraction using nba_api.

This module extracts team statistics including:
- Pace-adjusted metrics
- Offensive and defensive efficiency
- Four factors
- Advanced stats
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import date, timedelta
from nba_api.stats.endpoints import leaguedashteamstats, leaguestandingsv3


class TeamFeatures:
    """Extract and calculate team-level features."""
    
    def __init__(self, season: str):
        """
        Initialize team features for a season.
        
        Args:
            season: Season ID (e.g., '2025-26')
        """
        self.season = season
        self.team_stats = None
        self.standings = None
        self._load_data()
    
    def _load_data(self):
        """Load team stats and standings from nba_api."""
        try:
            # Load team stats
            stats_obj = leaguedashteamstats.LeagueDashTeamStats(
                season=self.season,
                per_mode_detailed='PerGame',
                measure_type_detailed_defense='Advanced',
            )
            self.team_stats = stats_obj.get_data_frames()[0]
            
            # Load standings
            standings_obj = leaguestandingsv3.LeagueStandingsV3(
                league_id='00',
                season=self.season,
                season_type='Regular Season',
            )
            self.standings = standings_obj.get_data_frames()[0]
            
        except Exception as e:
            print(f"Error loading team data: {e}")
            self.team_stats = pd.DataFrame()
            self.standings = pd.DataFrame()
    
    def calculate_efg(self, row: pd.Series) -> float:
        """Calculate effective field goal percentage."""
        fgm = row.get('FGM', 0)
        fg3m = row.get('FG3M', 0)
        fga = row.get('FGA', 1)
        
        if fga == 0:
            return 0.5
        
        efg = (fgm + 0.5 * fg3m) / fga
        return efg
    
    def calculate_ftr(self, row: pd.Series) -> float:
        """Calculate free throw rate (FTM / FGA)."""
        ftm = row.get('FTM', 0)
        fta = row.get('FTA', 0)
        fga = row.get('FGA', 1)
        
        if fga == 0:
            return 0.0
        
        ftr = fta / fga
        return ftr
    
    def calculate_tpar(self, row: pd.Series) -> float:
        """Calculate three-point attempt rate (3PA / FGA)."""
        fg3a = row.get('FG3A', 0)
        fga = row.get('FGA', 1)
        
        if fga == 0:
            return 0.0
        
        tpar = fg3a / fga
        return tpar
    
    def calculate_tor(self, row: pd.Series) -> float:
        """Calculate turnover rate (TOV / Possessions)."""
        tov = row.get('TOV', 0)
        fga = row.get('FGA', 0)
        fta = row.get('FTA', 0)
        oreb = row.get('OREB', 0)
        dreb = row.get('DREB', 0)
        
        possessions = max(fga + 0.44 * fta + tov - oreb + dreb, 1)
        tor = tov / possessions
        return tor
    
    def calculate_orbp(self, row: pd.Series) -> float:
        """Calculate offensive rebound percentage."""
        oreb = row.get('OREB', 0)
        dreb = row.get('DREB', 0)
        oreb_opp = oreb  # Approximation using own defensive rebounds
        dreb_opp = dreb
        
        total_reb = oreb + dreb + oreb_opp + dreb_opp
        if total_reb == 0:
            return 0.0
        
        orbp = oreb / total_reb
        return orbp
    
    def get_team_stats(self, tri_code: str) -> Optional[Dict[str, float]]:
        """
        Get calculated stats for a team.
        
        Args:
            tri_code: Team tri-code (e.g., 'LAL')
            
        Returns:
            Dictionary of team stats or None if not found
        """
        if self.team_stats is None or len(self.team_stats) == 0:
            return None
        
        team_row = self.team_stats[self.team_stats['TEAM_ABBREVIATION'] == tri_code]
        
        if len(team_row) == 0:
            return None
        
        team_row = team_row.iloc[0]
        
        stats = {
            'tri_code': tri_code,
            'efg': self.calculate_efg(team_row),
            'ftr': self.calculate_ftr(team_row),
            'tpar': self.calculate_tpar(team_row),
            'tor': self.calculate_tor(team_row),
            'orbp': self.calculate_orbp(team_row),
            'fga': team_row.get('FGA', 0),
            'fgm': team_row.get('FGM', 0),
            'fg3m': team_row.get('FG3M', 0),
            'fg3a': team_row.get('FG3A', 0),
            'fta': team_row.get('FTA', 0),
            'ftm': team_row.get('FTM', 0),
            'tov': team_row.get('TOV', 0),
            'orb': team_row.get('OREB', 0),
            'drb': team_row.get('DREB', 0),
            'reb': team_row.get('REB', 0),
            'ast': team_row.get('AST', 0),
            'stl': team_row.get('STL', 0),
            'blk': team_row.get('BLK', 0),
            'pts': team_row.get('PTS', 0),
            'pf': team_row.get('PF', 0),
        }
        
        # Add advanced stats if available
        if 'OFF_RATING' in team_row:
            stats['off_rating'] = team_row['OFF_RATING']
        if 'DEF_RATING' in team_row:
            stats['def_rating'] = team_row['DEF_RATING']
        if 'PACE' in team_row:
            stats['pace'] = team_row['PACE']
        if 'NET_RATING' in team_row:
            stats['net_rating'] = team_row['NET_RATING']
        if 'TS_PCT' in team_row:
            stats['ts_pct'] = team_row['TS_PCT']
        if 'eFG_PCT' in team_row:
            stats['efg_pct'] = team_row['eFG_PCT']
        if 'AST_PCT' in team_row:
            stats['ast_pct'] = team_row['AST_PCT']
        if 'REB_PCT' in team_row:
            stats['reb_pct'] = team_row['REB_PCT']
        
        return stats
    
    def get_defensive_strength(self, tri_code: str) -> Optional[float]:
        """
        Get defensive strength (defensive rating).
        
        Args:
            tri_code: Team tri-code
            
        Returns:
            Defensive rating or None if not found
        """
        stats = self.get_team_stats(tri_code)
        if stats is None:
            return None
        
        # Use def_rating if available, otherwise estimate from pts
        if 'def_rating' in stats:
            return stats['def_rating']
        
        # Estimate from points allowed (lower is better)
        pts = stats.get('pts', 110)  # League avg ~110
        defensive_strength = 110 - pts  # Positive = better defense
        
        return defensive_strength
    
    def get_offensive_rating(self, tri_code: str) -> Optional[float]:
        """
        Get offensive rating.
        
        Args:
            tri_code: Team tri-code
            
        Returns:
            Offensive rating or None if not found
        """
        stats = self.get_team_stats(tri_code)
        if stats is None:
            return None
        
        return stats.get('off_rating', 110)
    
    def get_pace(self, tri_code: str) -> Optional[float]:
        """
        Get team pace (possessions per game).
        
        Args:
            tri_code: Team tri-code
            
        Returns:
            Pace or None if not found
        """
        stats = self.get_team_stats(tri_code)
        if stats is None:
            return None
        
        return stats.get('pace', 100.0)  # League avg ~100


if __name__ == '__main__':
    # Test the module
    tf = TeamFeatures('2025-26')
    
    # Test for a few teams
    for team in ['LAL', 'BOS', 'GSW']:
        stats = tf.get_team_stats(team)
        if stats:
            print(f"\n{team}:")
            print(f"  eFG: {stats['efg']:.3f}")
            print(f"  FTR: {stats['ftr']:.3f}")
            print(f"  TPAR: {stats['tpar']:.3f}")
            print(f"  TOR: {stats['tor']:.3f}")
            print(f"  ORBP: {stats['orbp']:.3f}")
            print(f"  Off Rating: {stats.get('off_rating', 'N/A')}")
            print(f"  Def Rating: {stats.get('def_rating', 'N/A')}")
            print(f"  Pace: {stats.get('pace', 'N/A')}")
