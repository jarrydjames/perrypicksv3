"""
Phase 1: Fetch 2025-2026 Season Data
Fetch all games from start of season (Oct 2025) to today (Feb 1, 2026)
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SeasonDataFetcher:
    def __init__(self):
        self.boxscore_dir = Path("data/raw/box")
        self.boxscore_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_season_schedule(self, season: str, start_date: str, end_date: str):
        """
        Fetch games for a season using NBA.com scoreboard API.
        
        Args:
            season: Season identifier (e.g., "2025-26")
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        logger.info(f"Fetching schedule for {season} from {start_date} to {end_date}")
        
        # Generate date range
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        date_range = []
        current = start
        
        while current <= end:
            date_range.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        
        logger.info(f"  Date range: {len(date_range)} days")
        
        games = []
        
        for date in date_range:
            date_str = date.replace('-', '')  # YYYYMMDD
            
            # Try NBA.com scoreboard API
            url = f"https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00{date_str}.json"
            
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                games_data = data.get('scoreboard', {}).get('games', [])
                
                for game in games_data:
                    games.append({
                        'game_id': game.get('gameId'),
                        'game_date': date,
                        'season': season,
                        'home_team_id': game.get('homeTeam', {}).get('teamId'),
                        'away_team_id': game.get('awayTeam', {}).get('teamId'),
                        'home_team_name': game.get('homeTeam', {}).get('teamName'),
                        'away_team_name': game.get('awayTeam', {}).get('teamName'),
                    })
                
                if games_data:
                    logger.info(f"  {date}: {len(games_data)} games")
                time.sleep(0.1)  # Rate limiting
            
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 404:
                    pass  # No games on this day
                else:
                    logger.warning(f"  {date}: HTTP error {e.response.status_code}")
            except Exception as e:
                logger.warning(f"  {date}: Error - {e}")
        
        logger.info(f"  Total games found: {len(games)}")
        return games
    
    def fetch_boxscores(self, games: List[Dict]):
        """Fetch boxscores for all games."""
        logger.info(f"Fetching boxscores for {len(games)} games...")
        
        base_url = "https://cdn.nba.com/static/json/liveData/boxscore/"
        
        fetched = 0
        failed = 0
        
        for game in games:
            game_id = game['game_id']
            url = f"{base_url}{game_id}_full.json"
            
            output_path = self.boxscore_dir / f"{game_id}.json"
            
            # Skip if already exists
            if output_path.exists():
                fetched += 1
                continue
            
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                boxscore = response.json()
                
                with open(output_path, 'w') as f:
                    json.dump(boxscore, f, indent=2)
                
                fetched += 1
                
                if fetched % 10 == 0:
                    logger.info(f"  Progress: {fetched}/{len(games)}")
                
                time.sleep(0.1)  # Rate limiting
            
            except Exception as e:
                failed += 1
                logger.warning(f"  {game_id}: Failed - {e}")
        
        logger.info(f"  Fetched: {fetched}, Failed: {failed}")
        return fetched
    
    def calculate_possessions(self, team_stats: Dict) -> float:
        """
        Calculate possessions using NBA formula:
        Possessions = FGA - ORB + TOV + 0.44 * FTA
        """
        fga = team_stats.get('FGA', 0)
        orb = team_stats.get('OREB', 0)
        tov = team_stats.get('TOV', 0)
        fta = team_stats.get('FTA', 0)
        
        return fga - orb + tov + 0.44 * fta if fga > 0 else 0
    
    def calculate_offensive_rating(self, team_stats: Dict, possessions: float) -> float:
        """Offensive Rating = (Points / Possessions) * 100"""
        pts = team_stats.get('PTS', 0)
        return (pts / possessions * 100) if possessions > 0 else 0
    
    def calculate_defensive_rating(self, team_stats: Dict, opp_possessions: float) -> float:
        """Defensive Rating = (Opponent Points / Opponent Possessions) * 100"""
        # For boxscore, we need opponent's points from the other team
        return None  # Will calculate from full game stats
    
    def extract_game_stats(self, game_id: str, boxscore: Dict) -> Dict:
        """Extract detailed stats from boxscore."""
        home_team = boxscore.get('homeTeam', {})
        away_team = boxscore.get('awayTeam', {})
        
        home_stats = home_team.get('statistics', [])
        away_stats = away_team.get('statistics', [])
        
        # Calculate possessions
        home_poss = self.calculate_possessions(home_stats)
        away_poss = self.calculate_possessions(away_stats)
        
        # Get scores
        home_score = home_team.get('score', 0)
        away_score = away_team.get('score', 0)
        
        # Calculate defensive ratings (opponent offensive rating)
        home_dr = (away_score / home_poss * 100) if home_poss > 0 else 0
        away_dr = (home_score / away_poss * 100) if away_poss > 0 else 0
        
        return {
            'game_id': game_id,
            'home_possessions': home_poss,
            'away_possessions': away_poss,
            'home_offensive_rating': self.calculate_offensive_rating(home_stats, home_poss),
            'away_offensive_rating': self.calculate_offensive_rating(away_stats, away_poss),
            'home_defensive_rating': home_dr,
            'away_defensive_rating': away_dr,
            'pace': (home_poss + away_poss) / 2,  # Average possessions
        }
    
    def build_season_averages_with_new_features(self, season: str) -> pd.DataFrame:
        """
        Build season averages including new features:
        - Pace (possessions per game)
        - Offensive rating
        - Defensive rating
        """
        logger.info(f"Building season averages for {season} with new features...")
        
        # Load all boxscores for this season
        box_files = sorted(list(self.boxscore_dir.glob("*.json")))
        
        team_game_stats = defaultdict(list)
        
        for f in box_files:
            game_id = f.stem
            try:
                with open(f) as f:
                    boxscore = json.load(f)
                
                stats = self.extract_game_stats(game_id, boxscore)
                
                # Get team IDs
                home_team_id = boxscore.get('homeTeam', {}).get('teamId')
                away_team_id = boxscore.get('awayTeam', {}).get('teamId')
                
                if home_team_id:
                    stats['TEAM_ID'] = home_team_id
                    stats['TEAM_NAME'] = boxscore.get('homeTeam', {}).get('teamName')
                    team_game_stats[home_team_id].append(stats)
                
                if away_team_id:
                    stats_copy = stats.copy()
                    stats_copy['TEAM_ID'] = away_team_id
                    stats_copy['TEAM_NAME'] = boxscore.get('awayTeam', {}).get('teamName')
                    # Swap offensive/defensive ratings
                    stats_copy['home_offensive_rating'], stats_copy['away_offensive_rating'] = \
                        stats_copy['away_offensive_rating'], stats_copy['home_offensive_rating']
                    stats_copy['home_defensive_rating'], stats_copy['away_defensive_rating'] = \
                        stats_copy['away_defensive_rating'], stats_copy['home_defensive_rating']
                    team_game_stats[away_team_id].append(stats_copy)
            
            except Exception as e:
                logger.warning(f"  Error processing {game_id}: {e}")
        
        # Calculate averages
        season_data = []
        
        for team_id, games in team_game_stats.items():
            if len(games) < 5:
                continue
            
            df_team = pd.DataFrame(games)
            
            team_avg = {
                'TEAM_ID': team_id,
                'TEAM_NAME': df_team['TEAM_NAME'].iloc[0],
                'GP': len(games),
                'PACE': df_team['pace'].mean(),
                'OFF_RATING': df_team['home_offensive_rating'].mean(),
                'DEF_RATING': df_team['home_defensive_rating'].mean(),
            }
            
            # Traditional stats
            for col in ['PTS', 'FG_PCT', 'FG3A', 'FGA', 'FTA', 'TOV', 'OREB', 'REB']:
                if col in df_team.columns:
                    team_avg[col] = df_team[col].mean()
            
            season_data.append(team_avg)
        
        df_season = pd.DataFrame(season_data)
        logger.info(f"  Averages for {len(df_season)} teams")
        
        # Save
        output_path = Path("data/season_averages") / f"season_avgs_{season}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_season.to_parquet(output_path, index=False)
        logger.info(f"  Saved to {output_path}")
        
        return df_season
    
    def run(self):
        """Run complete data fetching."""
        logger.info("="*70)
        logger.info("PHASE 1: FETCH 2025-2026 SEASON DATA")
        logger.info("="*70)
        
        # 2025-2026 season: Oct 2025 to Feb 1, 2026
        season = "2025-26"
        start_date = "2025-10-01"
        end_date = "2026-02-01"
        
        # Step 1: Fetch schedule
        games = self.fetch_season_schedule(season, start_date, end_date)
        
        if not games:
            logger.error("No games found - stopping")
            return
        
        # Step 2: Fetch boxscores
        self.fetch_boxscores(games)
        
        # Step 3: Build season averages with new features
        self.build_season_averages_with_new_features(season)
        
        logger.info("="*70)
        logger.info("PHASE 1 COMPLETE")
        logger.info("="*70)


def main():
    fetcher = SeasonDataFetcher()
    fetcher.run()
    return 0


if __name__ == '__main__':
    exit(main())
