"""
Phase 5: Build Team Rating System
Calculate offensive/defensive efficiency, pace, and 4 factors for each team over time.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TeamRatingsBuilder:
    """
    Build team ratings over time based on historical game data.
    
    Ratings calculated:
    - Offensive rating: points scored per 100 possessions
    - Defensive rating: points allowed per 100 possessions
    - Pace: average possessions per game
    - eFG%: effective field goal percentage
    - TOV%: turnover rate
    - ORB%: offensive rebound percentage
    - FT/FGA: free throw rate
    
    All ratings are tracked over time so we know what the rating was BEFORE each game.
    """
    
    def __init__(self):
        self.boxscore_dir = Path("data/raw/box")
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def normalize_boxscore(self, boxscore: Dict) -> Dict:
        """Normalize boxscore format."""
        if 'game' in boxscore:
            return boxscore['game']
        return boxscore
    
    def get_stat_value(self, stats: Dict, stat_name: str, default: float = 0) -> float:
        """Extract stat value from stats dict."""
        return stats.get(stat_name, default)
    
    def calculate_possessions(self, stats_dict: Dict) -> float:
        """Calculate possessions: FGA - ORB + TOV + 0.44 * FTA."""
        fga = self.get_stat_value(stats_dict, 'fieldGoalsAttempted', 0)
        orb = self.get_stat_value(stats_dict, 'reboundsOffensive', 0)
        tov = self.get_stat_value(stats_dict, 'turnovers', 0)
        fta = self.get_stat_value(stats_dict, 'freeThrowsAttempted', 0)
        return max(1, fga - orb + tov + 0.44 * fta)
    
    def calculate_efficiency_metrics(self, team: Dict, opponent_score: float) -> Dict:
        """Calculate offensive/defensive efficiency for a single game."""
        stats = team.get('statistics', {})
        score = team.get('score', 0)
        
        # Possessions
        poss = self.calculate_possessions(stats)
        
        # Offensive rating (points per 100 possessions)
        off_rating = (score / poss) * 100 if poss > 0 else 0
        
        # Defensive rating (opponent points per 100 possessions)
        def_rating = (opponent_score / poss) * 100 if poss > 0 else 0
        
        # Pace (possessions)
        pace = poss
        
        # eFG%
        fgm = self.get_stat_value(stats, 'fieldGoalsMade', 0)
        fga = self.get_stat_value(stats, 'fieldGoalsAttempted', 0)
        fg3m = self.get_stat_value(stats, 'threePointersMade', 0)
        efg = (fgm + 0.5 * fg3m) / fga if fga > 0 else 0
        
        # TOV%
        tov = self.get_stat_value(stats, 'turnovers', 0)
        tov_rate = tov / poss if poss > 0 else 0
        
        # ORB%
        orb = self.get_stat_value(stats, 'reboundsOffensive', 0)
        drb_opp = self.get_stat_value(stats, 'reboundsDefensive', 0)
        orb_rate = orb / (orb + drb_opp) if (orb + drb_opp) > 0 else 0
        
        # FT/FGA
        fta = self.get_stat_value(stats, 'freeThrowsAttempted', 0)
        ft_rate = fta / fga if fga > 0 else 0
        
        return {
            'off_rating': off_rating,
            'def_rating': def_rating,
            'pace': pace,
            'efg': efg,
            'tov_rate': tov_rate,
            'orb_rate': orb_rate,
            'ft_rate': ft_rate,
            'points': score,
            'opponent_points': opponent_score,
        }
    
    def load_all_boxscores(self) -> List[Tuple[Dict, datetime]]:
        """Load and sort all boxscores by date."""
        logger.info("Loading all boxscores...")
        
        box_files = sorted(list(self.boxscore_dir.glob("*.json")))
        games_data = []
        
        for box_file in box_files:
            try:
                with open(box_file) as f:
                    boxscore = json.load(f)
                
                boxscore = self.normalize_boxscore(boxscore)
                game_date_str = boxscore.get('gameTimeUTC', '')
                game_date = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))
                
                games_data.append((boxscore, game_date))
            except Exception as e:
                logger.debug(f"Error loading {box_file}: {e}")
        
        logger.info(f"  Loaded {len(games_data)} games")
        
        # Sort by date
        games_data.sort(key=lambda x: x[1])
        
        return games_data
    
    def build_team_ratings_history(self, games_data: List[Tuple[Dict, datetime]]) -> pd.DataFrame:
        """
        Build team ratings over time.
        
        For each game, we calculate what each team's ratings were BEFORE that game
        (based on all previous games).
        """
        logger.info("Building team ratings history...")
        
        # Track cumulative stats for each team
        team_stats = defaultdict(lambda: {
            'games_played': 0,
            'off_rating_total': 0,
            'def_rating_total': 0,
            'pace_total': 0,
            'efg_total': 0,
            'tov_rate_total': 0,
            'orb_rate_total': 0,
            'ft_rate_total': 0,
            'points_total': 0,
            'opponent_points_total': 0,
            'home_games': 0,
            'away_games': 0,
            'wins': 0,
        })
        
        ratings_history = []
        
        for boxscore, game_date in games_data:
            game_id = boxscore.get('gameId', '')
            
            home_team = boxscore.get('homeTeam', {})
            away_team = boxscore.get('awayTeam', {})
            
            home_id = home_team.get('teamId')
            away_id = away_team.get('teamId')
            
            home_score = home_team.get('score', 0)
            away_score = away_team.get('score', 0)
            
            # Get ratings BEFORE this game
            home_ratings = self._get_team_ratings(team_stats[home_id])
            away_ratings = self._get_team_ratings(team_stats[away_id])
            
            # Record ratings history for this game
            ratings_history.append({
                'game_id': game_id,
                'game_date': game_date,
                'home_team_id': home_id,
                'away_team_id': away_id,
                'home_score': home_score,
                'away_score': away_score,
                'total': home_score + away_score,
                'margin': home_score - away_score,
                'home_off_rating': home_ratings['off_rating'],
                'away_off_rating': away_ratings['off_rating'],
                'home_def_rating': home_ratings['def_rating'],
                'away_def_rating': away_ratings['def_rating'],
                'home_pace': home_ratings['pace'],
                'away_pace': away_ratings['pace'],
                'home_efg': home_ratings['efg'],
                'away_efg': away_ratings['efg'],
                'home_tov_rate': home_ratings['tov_rate'],
                'away_tov_rate': away_ratings['tov_rate'],
                'home_orb_rate': home_ratings['orb_rate'],
                'away_orb_rate': away_ratings['orb_rate'],
                'home_ft_rate': home_ratings['ft_rate'],
                'away_ft_rate': away_ratings['ft_rate'],
                'home_win_pct': home_ratings['win_pct'],
                'away_win_pct': away_ratings['win_pct'],
                'home_home_win_pct': home_ratings['home_win_pct'],
                'away_road_win_pct': away_ratings['road_win_pct'],
            })
            
            # Calculate this game's metrics and update team stats
            home_metrics = self.calculate_efficiency_metrics(home_team, away_score)
            away_metrics = self.calculate_efficiency_metrics(away_team, home_score)
            
            self._update_team_stats(team_stats[home_id], home_metrics, is_home=True, is_win=home_score > away_score)
            self._update_team_stats(team_stats[away_id], away_metrics, is_home=False, is_win=away_score > home_score)
        
        df = pd.DataFrame(ratings_history)
        logger.info(f"  Built ratings history for {len(df)} games")
        logger.info(f"  Columns: {list(df.columns)}")
        
        return df
    
    def _get_team_ratings(self, team_stats: Dict) -> Dict:
        """Get current team ratings (averages)."""
        games = team_stats['games_played']
        
        if games == 0:
            # Return league averages for new teams
            return {
                'off_rating': 110.0,  # ~league avg
                'def_rating': 110.0,
                'pace': 100.0,
                'efg': 0.52,
                'tov_rate': 0.14,
                'orb_rate': 0.25,
                'ft_rate': 0.20,
                'win_pct': 0.5,
                'home_win_pct': 0.5,
                'road_win_pct': 0.5,
            }
        
        return {
            'off_rating': team_stats['off_rating_total'] / games,
            'def_rating': team_stats['def_rating_total'] / games,
            'pace': team_stats['pace_total'] / games,
            'efg': team_stats['efg_total'] / games,
            'tov_rate': team_stats['tov_rate_total'] / games,
            'orb_rate': team_stats['orb_rate_total'] / games,
            'ft_rate': team_stats['ft_rate_total'] / games,
            'win_pct': team_stats['wins'] / games,
            'home_win_pct': team_stats['wins'] / team_stats['home_games'] if team_stats['home_games'] > 0 else 0.5,
            'road_win_pct': team_stats['wins'] / team_stats['away_games'] if team_stats['away_games'] > 0 else 0.5,
        }
    
    def _update_team_stats(self, team_stats: Dict, metrics: Dict, is_home: bool, is_win: bool):
        """Update cumulative team stats after a game."""
        team_stats['games_played'] += 1
        team_stats['off_rating_total'] += metrics['off_rating']
        team_stats['def_rating_total'] += metrics['def_rating']
        team_stats['pace_total'] += metrics['pace']
        team_stats['efg_total'] += metrics['efg']
        team_stats['tov_rate_total'] += metrics['tov_rate']
        team_stats['orb_rate_total'] += metrics['orb_rate']
        team_stats['ft_rate_total'] += metrics['ft_rate']
        team_stats['points_total'] += metrics['points']
        team_stats['opponent_points_total'] += metrics['opponent_points']
        
        if is_home:
            team_stats['home_games'] += 1
        else:
            team_stats['away_games'] += 1
        
        if is_win:
            team_stats['wins'] += 1
    
    def run(self):
        """Run complete team ratings building."""
        logger.info("="*70)
        logger.info("PHASE 5: BUILD TEAM RATING SYSTEM")
        logger.info("="*70)
        
        # Step 1: Load all boxscores
        games_data = self.load_all_boxscores()
        
        if len(games_data) == 0:
            logger.error("No games loaded - stopping")
            return None
        
        # Step 2: Build team ratings history
        df = self.build_team_ratings_history(games_data)
        
        # Step 3: Save
        output_path = self.processed_dir / "team_ratings.parquet"
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved team ratings to {output_path}")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Date range: {df['game_date'].min()} to {df['game_date'].max()}")
        
        # Display sample
        logger.info("\nSample data (first 5 games):")
        print(df.head().to_string(index=False))
        
        logger.info("="*70)
        logger.info("PHASE 5 COMPLETE")
        logger.info("="*70)
        
        return df


def main():
    builder = TeamRatingsBuilder()
    return builder.run()


if __name__ == '__main__':
    exit(main())
