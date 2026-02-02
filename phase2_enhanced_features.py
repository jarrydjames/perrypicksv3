"""
Phase 2: Build Enhanced Features
Add pace, defensive rating, recent form, and interaction features
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedFeatureBuilder:
    def __init__(self):
        self.boxscore_dir = Path("data/raw/box")
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def normalize_boxscore(self, boxscore: Dict) -> Dict:
        """Normalize boxscore format - handle both nested and direct formats."""
        # Check if boxscore is wrapped in 'game' key
        if 'game' in boxscore:
            return boxscore['game']
        return boxscore
    
    def get_stat_value(self, stats: Dict, stat_name: str, default: float = 0) -> float:
        """Extract stat value from stats dict by name."""
        return stats.get(stat_name, default)
    
    def calculate_possessions(self, stats_dict: Dict) -> float:
        """Possessions = FGA - ORB + TOV + 0.44 * FTA"""
        fga = self.get_stat_value(stats_dict, 'fieldGoalsMade')
        orb = self.get_stat_value(stats_dict, 'reboundsOffensive')
        tov = self.get_stat_value(stats_dict, 'turnovers')
        fta = self.get_stat_value(stats_dict, 'freeThrowsMade')
        return fga - orb + tov + 0.44 * fta if fga > 0 else 0
    
    def extract_game_features(self, boxscore: Dict) -> Dict:
        """Extract comprehensive features from a boxscore."""
        # Normalize format
        boxscore = self.normalize_boxscore(boxscore)
        
        home_team = boxscore.get('homeTeam', {})
        away_team = boxscore.get('awayTeam', {})
        
        home_stats = home_team.get('statistics', {})
        away_stats = away_team.get('statistics', {})
        
        # Get basic stats
        home_pts = home_team.get('score', 0)
        away_pts = away_team.get('score', 0)
        
        # Calculate possessions
        home_poss = self.calculate_possessions(home_stats)
        away_poss = self.calculate_possessions(away_stats)
        total_poss = home_poss + away_poss
        
        # Get detailed stats
        home_fga = self.get_stat_value(home_stats, 'fieldGoalsMade')
        home_fg3a = self.get_stat_value(home_stats, 'threePointersMade')
        home_fta = self.get_stat_value(home_stats, 'freeThrowsMade')
        home_tov = self.get_stat_value(home_stats, 'turnovers')
        home_oreb = self.get_stat_value(home_stats, 'reboundsOffensive')
        home_reb = self.get_stat_value(home_stats, 'reboundsTotal')
        home_fgm = self.get_stat_value(home_stats, 'fieldGoalsMade')
        
        away_fga = self.get_stat_value(away_stats, 'fieldGoalsMade')
        away_fg3a = self.get_stat_value(away_stats, 'threePointersMade')
        away_fta = self.get_stat_value(away_stats, 'freeThrowsMade')
        away_tov = self.get_stat_value(away_stats, 'turnovers')
        away_oreb = self.get_stat_value(away_stats, 'reboundsOffensive')
        away_reb = self.get_stat_value(away_stats, 'reboundsTotal')
        away_fgm = self.get_stat_value(away_stats, 'fieldGoalsMade')
        
        # Calculated metrics
        home_efg = home_fgm / home_fga if home_fga > 0 else 0
        away_efg = away_fgm / away_fga if away_fga > 0 else 0
        
        home_ftr = home_fta / home_fga if home_fga > 0 else 0
        away_ftr = away_fta / away_fga if away_fga > 0 else 0
        
        home_tpar = home_fg3a / home_fga if home_fga > 0 else 0
        away_tpar = away_fg3a / away_fga if away_fga > 0 else 0
        
        home_tor = home_tov / home_fga if home_fga > 0 else 0
        away_tor = away_tov / away_fga if away_fga > 0 else 0
        
        home_orbp = home_oreb / home_reb if home_reb > 0 else 0
        away_orbp = away_oreb / away_reb if away_reb > 0 else 0
        
        # Pace and rating features
        home_pace = home_poss
        away_pace = away_poss
        avg_pace = total_poss / 2
        
        home_off_rating = (home_pts / home_poss * 100) if home_poss > 0 else 0
        away_off_rating = (away_pts / away_poss * 100) if away_poss > 0 else 0
        
        home_def_rating = (away_pts / home_poss * 100) if home_poss > 0 else 0
        away_def_rating = (home_pts / away_poss * 100) if away_poss > 0 else 0
        
        # Interaction features
        pace_diff = home_pace - away_pace
        off_rating_diff = home_off_rating - away_off_rating
        def_rating_diff = home_def_rating - away_def_rating
        
        return {
            'game_id': boxscore.get('gameId', ''),
            'game_date': boxscore.get('gameTimeUTC', ''),
            
            # Targets
            'total': home_pts + away_pts,
            'margin': home_pts - away_pts,
            
            # Basic stats
            'home_pts': home_pts,
            'away_pts': away_pts,
            'home_efg': home_efg,
            'away_efg': away_efg,
            'home_ftr': home_ftr,
            'away_ftr': away_ftr,
            'home_tpar': home_tpar,
            'away_tpar': away_tpar,
            'home_tor': home_tor,
            'away_tor': away_tor,
            'home_orbp': home_orbp,
            'away_orbp': away_orbp,
            
            # NEW: Pace features
            'home_pace': home_pace,
            'away_pace': away_pace,
            'avg_pace': avg_pace,
            'pace_diff': pace_diff,
            
            # NEW: Offensive/Defensive ratings
            'home_off_rating': home_off_rating,
            'away_off_rating': away_off_rating,
            'home_def_rating': home_def_rating,
            'away_def_rating': away_def_rating,
            'off_rating_diff': off_rating_diff,
            'def_rating_diff': def_rating_diff,
            
            # Team IDs (for season average lookup)
            'home_team_id': home_team.get('teamId'),
            'away_team_id': away_team.get('teamId'),
        }
    
    def build_comprehensive_dataset(self) -> pd.DataFrame:
        """Build dataset with enhanced features from all boxscores."""
        logger.info("Building comprehensive dataset with enhanced features...")
        
        box_files = sorted(list(self.boxscore_dir.glob("*.json")))
        all_games = []
        
        for box_file in box_files:
            game_id = box_file.stem
            try:
                with open(box_file) as f:
                    boxscore = json.load(f)
                
                features = self.extract_game_features(boxscore)
                
                # Validate features
                if features['game_id'] and features['total'] > 0:
                    all_games.append(features)
            
            except Exception as e:
                logger.debug(f"  Error processing {game_id}: {e}")
        
        df = pd.DataFrame(all_games)
        logger.info(f"  Extracted {len(df)} games with {len(df.columns)} features")
        
        return df
    
    def add_recent_form_features(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Add recent form features (last N games).
        This calculates rolling averages for each team.
        """
        logger.info(f"Adding recent form features (last {window} games)...")
        
        df['game_date_dt'] = pd.to_datetime(df['game_date'])
        df = df.sort_values('game_date_dt')
        
        # For each team, calculate rolling stats
        team_stats = defaultdict(list)
        
        recent_form_data = []
        
        for idx, row in df.iterrows():
            game_id = row['game_id']
            home_id = row['home_team_id']
            away_id = row['away_team_id']
            game_date = row['game_date_dt']
            
            # Get recent stats for home team
            home_recent = [g for g in team_stats[home_id] if g['date'] < game_date][-window:]
            away_recent = [g for g in team_stats[away_id] if g['date'] < game_date][-window:]
            
            # Calculate recent averages
            if len(home_recent) >= 2:
                home_recent_pts = sum(g['pts'] for g in home_recent) / len(home_recent)
                home_recent_total = sum(g['total'] for g in home_recent) / len(home_recent)
                home_recent_wins = sum(1 for g in home_recent if g['won'])
            else:
                home_recent_pts = row['home_pts']
                home_recent_total = row['total']
                home_recent_wins = 0
            
            if len(away_recent) >= 2:
                away_recent_pts = sum(g['pts'] for g in away_recent) / len(away_recent)
                away_recent_total = sum(g['total'] for g in away_recent) / len(away_recent)
                away_recent_wins = sum(1 for g in away_recent if g['won'])
            else:
                away_recent_pts = row['away_pts']
                away_recent_total = row['total']
                away_recent_wins = 0
            
            # Add form features to row
            row_copy = row.copy()
            row_copy['home_recent_pts'] = home_recent_pts
            row_copy['away_recent_pts'] = away_recent_pts
            row_copy['home_recent_total'] = home_recent_total
            row_copy['away_recent_total'] = away_recent_total
            row_copy['home_recent_win_pct'] = home_recent_wins / len(home_recent) if home_recent else 0.5
            row_copy['away_recent_win_pct'] = away_recent_wins / len(away_recent) if away_recent else 0.5
            
            recent_form_data.append(row_copy)
            
            # Add current game to team stats
            team_stats[home_id].append({
                'date': game_date,
                'pts': row['home_pts'],
                'total': row['total'],
                'won': row['margin'] > 0
            })
            team_stats[away_id].append({
                'date': game_date,
                'pts': row['away_pts'],
                'total': row['total'],
                'won': row['margin'] < 0
            })
        
        df_form = pd.DataFrame(recent_form_data)
        logger.info(f"  Added {6} recent form features to {len(df_form)} games")
        
        return df_form
    
    def run(self):
        """Run complete feature building."""
        logger.info("="*70)
        logger.info("PHASE 2: BUILD ENHANCED FEATURES")
        logger.info("="*70)
        
        # Step 1: Extract basic enhanced features
        df = self.build_comprehensive_dataset()
        
        if len(df) == 0:
            logger.error("No games extracted - stopping")
            return None
        
        # Step 2: Add recent form features
        df_form = self.add_recent_form_features(df, window=5)
        
        # Step 3: Save
        output_path = self.processed_dir / "enhanced_features.parquet"
        df_form.to_parquet(output_path, index=False)
        logger.info(f"Saved enhanced dataset to {output_path}")
        logger.info(f"  Shape: {df_form.shape}")
        logger.info(f"  Features: {list(df_form.columns)}")
        
        logger.info("="*70)
        logger.info("PHASE 2 COMPLETE")
        logger.info("="*70)
        
        return df_form


def main():
    builder = EnhancedFeatureBuilder()
    return builder.run()


if __name__ == '__main__':
    exit(main())
