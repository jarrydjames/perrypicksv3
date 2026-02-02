"""
OPTION A PHASE 2: Build True Leakage-Free Pregame Dataset
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TruePregameBuilder:
    def __init__(self, season_avgs_dir: str = "data/season_averages", 
                 boxscore_dir: str = "data/raw/box"):
        self.season_avgs_dir = Path(season_avgs_dir)
        self.boxscore_dir = Path(boxscore_dir)
        
        self.season_avgs = self._load_all_season_avgs()
        logger.info(f"Initialized with {len(self.season_avgs)} seasons")
    
    def _load_all_season_avgs(self) -> Dict[str, Dict[int, Dict]]:
        seasons = {}
        for season_file in self.season_avgs_dir.glob("season_avgs_*.parquet"):
            season = season_file.stem.replace("season_avgs_", "")
            df = pd.read_parquet(season_file)
            team_lookup = {}
            for _, row in df.iterrows():
                team_lookup[int(row['TEAM_ID'])] = row.to_dict()
            seasons[season] = team_lookup
            logger.info(f"   Loaded {season}: {len(team_lookup)} teams")
        return seasons
    
    def _get_season_from_date(self, game_date_str: str) -> Optional[str]:
        """
        Determine season from game date.
        NBA season typically starts in October and ends in April/June.
        - Games from Oct 2023 - Jun 2024 = 2023-24 season
        - Games from Oct 2022 - Jun 2023 = 2022-23 season
        - Games from Oct 2024 - Jun 2025 = 2024-25 season
        """
        if not game_date_str:
            return None
        
        try:
            game_date = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))
        except Exception:
            return None
        
        year = game_date.year
        month = game_date.month
        
        # October or later in year X belongs to season X-X+1
        # Before October in year X belongs to season X-1-X
        if month >= 10:
            season = f"{year}-{str(year+1)[2:]}"
        else:
            season = f"{year-1}-{str(year)[2:]}"
        
        return season
    
    def _extract_game_info_from_boxscore(self, game_id: str) -> Optional[Dict]:
        boxscore_path = self.boxscore_dir / f"{game_id}.json"
        
        if not boxscore_path.exists():
            return None
        
        try:
            with open(boxscore_path) as f:
                data = json.load(f)
            
            home_team_id = data.get('homeTeam', {}).get('teamId')
            away_team_id = data.get('awayTeam', {}).get('teamId')
            
            # Skip if team IDs are missing
            if not home_team_id or not away_team_id:
                return None
            
            game_date = data.get('gameTimeUTC')
            
            home_score = data.get('homeTeam', {}).get('score')
            away_score = data.get('awayTeam', {}).get('score')
            
            return {
                'game_id': game_id,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'game_date': game_date,
                'home_score': home_score,
                'away_score': away_score,
                'total': home_score + away_score if (home_score and away_score) else None,
                'margin': home_score - away_score if (home_score and away_score) else None,
            }
        except Exception as e:
            logger.error(f"Failed to extract game info from {game_id}: {e}")
            return None
    
    def _build_pregame_features(self, game_info: Dict) -> Dict:
        game_id = game_info['game_id']
        game_date = game_info['game_date']
        
        # Determine season from game date
        season = self._get_season_from_date(game_date)
        
        if not season:
            logger.warning(f"Could not determine season for {game_id}")
            return None
        
        home_team_id = game_info['home_team_id']
        away_team_id = game_info['away_team_id']
        
        # Get season averages
        season_avgs = self.season_avgs.get(season, {})
        home_avg = season_avgs.get(int(home_team_id))
        away_avg = season_avgs.get(int(away_team_id))
        
        if home_avg is None or away_avg is None:
            logger.warning(f"Missing season averages for {game_id} (season: {season})")
            return None
        
        features = {
            'game_id': game_id,
            'season': season,
            'game_date': game_date,
            
            'total': game_info.get('total'),
            'margin': game_info.get('margin'),
            
            'home_efg': home_avg.get('FG_PCT'),
            'home_ftr': home_avg.get('FTA') / home_avg.get('FGA') if home_avg.get('FGA', 0) > 0 else None,
            'home_tpar': home_avg.get('FG3A') / home_avg.get('FGA') if home_avg.get('FGA', 0) > 0 else None,
            'home_tor': home_avg.get('TOV') / home_avg.get('FGA') if home_avg.get('FGA', 0) > 0 else None,
            'home_orbp': home_avg.get('OREB') / home_avg.get('REB') if home_avg.get('REB', 0) > 0 else None,
            
            'away_efg': away_avg.get('FG_PCT'),
            'away_ftr': away_avg.get('FTA') / away_avg.get('FGA') if away_avg.get('FGA', 0) > 0 else None,
            'away_tpar': away_avg.get('FG3A') / away_avg.get('FGA') if away_avg.get('FGA', 0) > 0 else None,
            'away_tor': away_avg.get('TOV') / away_avg.get('FGA') if away_avg.get('FGA', 0) > 0 else None,
            'away_orbp': away_avg.get('OREB') / away_avg.get('REB') if away_avg.get('REB', 0) > 0 else None,
            
            'home_pts': home_avg.get('PTS'),
            'home_ast': home_avg.get('AST'),
            'home_reb': home_avg.get('REB'),
            'away_pts': away_avg.get('PTS'),
            'away_ast': away_avg.get('AST'),
            'away_reb': away_avg.get('REB'),
        }
        
        return features
    
    def build_pregame_dataset(self, max_games: int = None, start_from: int = 0) -> pd.DataFrame:
        logger.info("="*70)
        logger.info("BUILDING TRUE LEAKAGE-FREE PREGAME DATASET")
        logger.info("="*70)
        
        boxscore_files = sorted(list(self.boxscore_dir.glob("*.json")))
        
        if start_from > 0:
            boxscore_files = boxscore_files[start_from:]
        if max_games is not None:
            boxscore_files = boxscore_files[:max_games]
        
        logger.info(f"Processing {len(boxscore_files)} games...")
        
        all_features = []
        missing_stats_count = 0
        missing_team_count = 0
        missing_season_count = 0
        
        for idx, boxscore_file in enumerate(boxscore_files):
            game_id = boxscore_file.stem
            
            if idx % 100 == 0:
                logger.info(f"Progress: {idx}/{len(boxscore_files)} games")
            
            game_info = self._extract_game_info_from_boxscore(game_id)
            
            if game_info is None:
                missing_team_count += 1
                continue
            
            features = self._build_pregame_features(game_info)
            
            if features is None:
                # Determine if missing season or missing stats
                season = self._get_season_from_date(game_info['game_date'])
                if season not in self.season_avgs:
                    missing_season_count += 1
                else:
                    missing_stats_count += 1
                continue
            
            all_features.append(features)
        
        df = pd.DataFrame(all_features)
        
        logger.info(f"Built dataset with {len(df)} games")
        logger.info(f"   Features: {len(df.columns)}")
        logger.info(f"   Skipped (no team IDs): {missing_team_count}")
        logger.info(f"   Skipped (no season data): {missing_season_count}")
        logger.info(f"   Skipped (no team avgs): {missing_stats_count}")
        if 'season' in df:
            logger.info(f"   Seasons: {', '.join(sorted(df['season'].unique()))}")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, output_path: str = "data/processed/pregame_leakage_free.parquet"):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved to {output_path}")
        
        logger.info(f"\nDataset Summary:")
        logger.info(f"   Games: {len(df)}")
        logger.info(f"   Features: {len(df.columns)}")
        if len(df) > 0 and 'game_date' in df:
            logger.info(f"   Date range: {df['game_date'].min()} to {df['game_date'].max()}")
        if len(df) > 0 and 'season' in df:
            logger.info(f"   Seasons: {', '.join(sorted(df['season'].unique()))}")
        
        null_counts = df.isnull().sum()
        high_nulls = null_counts[null_counts > len(df) * 0.1]
        if len(high_nulls) > 0:
            logger.warning(f"\nFeatures with >10% nulls:")
            for col, count in high_nulls.items():
                logger.warning(f"   {col}: {count} ({count/len(df)*100:.1f}%)")


def main():
    logger.info("="*70)
    logger.info("OPTION A PHASE 2: BUILD TRUE LEAKAGE-FREE PREGAME DATASET")
    logger.info("="*70)
    
    builder = TruePregameBuilder()
    
    df = builder.build_pregame_dataset(max_games=None)
    
    builder.save_dataset(df, "data/processed/pregame_leakage_free.parquet")
    
    logger.info("\n" + "="*70)
    logger.info("PHASE 2 COMPLETE - LEAKAGE-FREE PREGAME DATASET BUILT")
    logger.info("="*70)
    
    return 0


if __name__ == '__main__':
    exit(main())
