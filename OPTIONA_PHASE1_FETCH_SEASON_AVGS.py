"""
OPTION A PHASE 1: Fetch Historical Season Averages

This script fetches team season averages from the NBA API for multiple seasons.
Data is cached locally to avoid API rate limits.

Seasons to fetch:
- 2022-23
- 2023-24
- 2024-25
- 2025-26 (current season, partial)

Usage:
    python OPTIONA_PHASE1_FETCH_SEASON_AVGS.py
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    from nba_api.stats.endpoints import leaguedashteamstats, leaguestandingsv3
    from nba_api.stats.static import teams
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SeasonAveragesFetcher:
    """Fetch and cache team season averages from NBA API."""
    
    def __init__(self, cache_dir: str = "data/season_averages"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.team_id_map = self._build_team_id_map()
        logger.info(f"✅ Initialized with {len(self.team_id_map)} teams")
    
    def _build_team_id_map(self) -> Dict[str, int]:
        """Build team name to team_id mapping."""
        if not NBA_API_AVAILABLE:
            logger.warning("NBA API not available, using cached team map")
            return {}
        
        nba_teams = teams.get_teams()
        return {team['full_name']: team['id'] for team in nba_teams}
    
    def _get_cache_path(self, season: str) -> Path:
        """Get cache file path for a season."""
        return self.cache_dir / f"season_avgs_{season}.parquet"
    
    def _cache_exists(self, season: str) -> bool:
        """Check if cache exists for a season."""
        return self._get_cache_path(season).exists()
    
    def _load_from_cache(self, season: str) -> Optional[pd.DataFrame]:
        """Load season averages from cache."""
        cache_path = self._get_cache_path(season)
        if not cache_path.exists():
            return None
        
        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"✅ Loaded {len(df)} teams from cache: {season}")
            return df
        except Exception as e:
            logger.error(f"❌ Failed to load cache {season}: {e}")
            return None
    
    def _save_to_cache(self, df: pd.DataFrame, season: str) -> None:
        """Save season averages to cache."""
        cache_path = self._get_cache_path(season)
        df.to_parquet(cache_path, index=False)
        logger.info(f"✅ Saved {len(df)} teams to cache: {season}")
    
    def fetch_league_dash_team_stats(
        self,
        season: str,
        season_type: str = "Regular Season",
        measure_type: str = "Base",
        per_mode: str = "PerGame",
        plus_minus: str = "N",
        pace_adjust: str = "N",
        rank: str = "N",
        outcome: str = "",
        location: str = "",
        month: int = 0,
        season_segment: str = "",
        date_from: str = "",
        date_to: str = "",
        opponent_team_id: int = 0,
        vs_conference: str = "",
        vs_division: str = "",
        game_segment: str = "",
        period: int = 0,
        shot_clock_range: str = "",
        last_n_games: int = 0
    ) -> Optional[Dict]:
        """Fetch LeagueDashTeamStats from NBA API."""
        
        try:
            response = leaguedashteamstats.LeagueDashTeamStats(
                league_id_nullable='00',
                season=season,
                season_type_all_star=season_type,
                measure_type_detailed_defense=measure_type,
                per_mode_detailed=per_mode,
                plus_minus=plus_minus,
                pace_adjust=pace_adjust,
                rank=rank,
                outcome_nullable=outcome,
                location_nullable=location,
                month=str(month),
                season_segment_nullable=season_segment,
                date_from_nullable=date_from,
                date_to_nullable=date_to,
                opponent_team_id=opponent_team_id,
                vs_conference_nullable=vs_conference,
                vs_division_nullable=vs_division,
                game_segment_nullable=game_segment,
                period=str(period),
                shot_clock_range_nullable=shot_clock_range,
                last_n_games=str(last_n_games)
            )
            
            return response.get_dict()
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch LeagueDashTeamStats: {e}")
            return None
    
    def fetch_season_averages(
        self,
        season: str,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """Fetch season averages for all teams.
        
        Args:
            season: Season string (e.g., "2023-24")
            force_refresh: If True, ignore cache and refetch
        
        Returns:
            DataFrame with team season averages
        """
        
        # Check cache
        if not force_refresh and self._cache_exists(season):
            cached_df = self._load_from_cache(season)
            if cached_df is not None:
                return cached_df
        
        logger.info(f"🔄 Fetching season averages: {season}")
        
        # Fetch base stats (per game)
        base_data = self.fetch_league_dash_team_stats(
            season=season,
            season_type="Regular Season",
            measure_type="Base",
            per_mode="PerGame"
        )
        
        if base_data is None:
            logger.error(f"❌ Failed to fetch base stats for {season}")
            return pd.DataFrame()
        
        # Extract row sets
        row_sets = base_data.get('resultSets', [])
        if not row_sets:
            logger.error(f"❌ No result sets for {season}")
            return pd.DataFrame()
        
        # Get main stats
        df = pd.DataFrame(row_sets[0]['rowSet'], columns=row_sets[0]['headers'])
        
        logger.info(f"✅ Fetched {len(df)} teams for {season}")
        
        # Cache
        self._save_to_cache(df, season)
        
        return df
    
    def fetch_all_seasons(
        self,
        seasons: List[str],
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Fetch season averages for multiple seasons.
        
        Args:
            seasons: List of season strings (e.g., ["2022-23", "2023-24"])
            force_refresh: If True, ignore cache and refetch all
        
        Returns:
            Dict mapping season to DataFrame
        """
        
        results = {}
        
        for season in seasons:
            logger.info("="*70)
            logger.info(f"Processing season: {season}")
            logger.info("="*70)
            
            df = self.fetch_season_averages(season, force_refresh=force_refresh)
            
            if not df.empty:
                results[season] = df
            else:
                logger.warning(f"⚠️ No data for {season}")
            
            # Rate limit: sleep between seasons
            if season != seasons[-1]:
                logger.info("💤 Sleeping 2 seconds between seasons...")
                time.sleep(2)
        
        logger.info("="*70)
        logger.info(f"✅ Complete: {len(results)}/{len(seasons)} seasons")
        logger.info("="*70)
        
        return results
    
    def summarize_season_data(self, df: pd.DataFrame) -> Dict:
        """Summarize season data for verification."""
        
        summary = {
            'num_teams': len(df),
            'season': df.get('SEASON', ['N/A']).iloc[0] if 'SEASON' in df else 'N/A',
            'columns': list(df.columns),
            'sample_columns': [
                'TEAM_ID', 'TEAM_NAME', 'GP', 'W', 'L', 'W_PCT',
                'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
                'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB',
                'AST', 'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD',
                'PTS', 'PLUS_MINUS'
            ]
        }
        
        return summary


def main():
    """Main entry point."""
    
    logger.info("="*70)
    logger.info("OPTION A PHASE 1: FETCH HISTORICAL SEASON AVERAGES")
    logger.info("="*70)
    
    # Seasons to fetch
    seasons = [
        "2022-23",
        "2023-24",
        "2024-25",
        "2025-26"  # Current season (partial)
    ]
    
    # Create fetcher
    fetcher = SeasonAveragesFetcher()
    
    # Fetch all seasons
    results = fetcher.fetch_all_seasons(seasons, force_refresh=False)
    
    # Summarize results
    logger.info("\n" + "="*70)
    logger.info("SEASON DATA SUMMARY")
    logger.info("="*70)
    
    for season, df in results.items():
        summary = fetcher.summarize_season_data(df)
        logger.info(f"\n{season}:")
        logger.info(f"   Teams: {summary['num_teams']}")
        logger.info(f"   Columns: {len(summary['columns'])}")
        
        # Show sample of key columns
        sample_cols = [c for c in summary['sample_columns'] if c in df.columns]
        if sample_cols:
            logger.info(f"   Sample columns: {', '.join(sample_cols[:10])}")
            
            # Show sample data
            logger.info(f"\n   Sample data (first 3 teams):")
            for idx, row in df.head(3).iterrows():
                logger.info(f"      {row.get('TEAM_NAME', 'N/A')}: GP={row.get('GP', 'N/A')}, W={row.get('W', 'N/A')}, PTS={row.get('PTS', 'N/A')}")
    
    logger.info("\n" + "="*70)
    logger.info("✅ PHASE 1 COMPLETE - SEASON AVERAGES FETCHED")
    logger.info("="*70)
    
    return 0


if __name__ == '__main__':
    exit(main())
