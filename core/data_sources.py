"""
Data sources module for PerryPicks v4 Automation System.
Handles NBA API and Odds API calls with caching and rate limiting.
"""

import logging
import requests
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.storage import OddsCacheStorage
from nba_api.stats.endpoints import boxscoretraditionalv2

logger = logging.getLogger(__name__)

# Constants
SEASON = '2025-26'
NBA_API_TIMEOUT = 30
ODDS_API_TIMEOUT = 30

# ScheduleLeagueV2 URL - reliable for scheduled games (includes future games)
SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"

class NBADataSource:
    """NBA API data source."""
    
    @staticmethod
    def _parse_nba_datetime(dt_input: Any) -> Optional[datetime]:
        """
        Parse NBA API datetime to timezone-aware UTC datetime.
        
        NBA API returns various formats:
        - pandas Timestamp (with or without timezone)
        - ISO 8601 strings (with or without timezone)
        - Naive datetime objects
        
        Always returns timezone-aware datetime in UTC.
        """
        if dt_input is None:
            return None
        
        try:
            # Handle pandas Timestamp
            if hasattr(dt_input, 'to_pydatetime'):
                dt = dt_input.to_pydatetime()
                # Ensure timezone-aware
                if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            
            # Handle string inputs
            elif isinstance(dt_input, str):
                dt_str = dt_input.strip()
                
                # Try ISO format first
                try:
                    dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                    # Ensure timezone-aware
                    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except ValueError:
                    pass
                
                # Try parsing other common formats
                for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
                    try:
                        dt = datetime.strptime(dt_str, fmt)
                        return dt.replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue
                
                return None
            
            # Handle datetime objects
            elif isinstance(dt_input, datetime):
                dt = dt_input
                if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to parse datetime {dt_input}: {e}")
            return None
    
    @staticmethod
    def fetch_games_for_date(date: str) -> List[Dict[str, Any]]:
        """
        Fetch all games for a specific date (YYYY-MM-DD).
        Uses scheduleLeagueV2.json which has all scheduled games
        (including future games that haven't started yet).
        
        Returns list of game dicts with game_id, start_time, teams, etc.
        """
        try:
            # Fetch schedule
            headers = {
                'User-Agent': 'Mozilla/5.0',
                'Accept': 'application/json,text/plain,*/*',
                'Referer': 'https://www.nba.com/',
            }
            response = requests.get(SCHEDULE_URL, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            league = data.get('leagueSchedule', {})
            game_dates = league.get('gameDates', [])
            
            # Find games for the specified date
            # scheduleLeagueV2 uses "MM/DD/YYYY" format for gameDate
            # date parameter is YYYY-MM-DD, so we need to match
            target_month = date[5:7].lstrip('0')
            target_day = date[8:10].lstrip('0')
            target_year = date[:4]
            
            games_list = None
            for gd in game_dates:
                gd_str = gd.get('gameDate', '')
                if not gd_str:
                    continue
                    
                # Check if this matches our target date
                # Format: "02/02/2026 T00:00:00"
                if f'{target_month}/{target_day}/{target_year}' in gd_str:
                    games_list = gd.get('games', [])
                    break
            
            if not games_list:
                logger.info(f'No games found for date {date}')
                return []
            
            games = []
            for g in games_list:
                game_id = g.get('gameId')
                if not game_id:
                    continue
                
                # Get start time
                # scheduleLeagueV2 has gameTimeUTC in format like "2026-02-03T03:00:00Z"
                time_str = g.get('gameTimeUTC', '')
                if time_str:
                    game_time_utc = NBADataSource._parse_nba_datetime(time_str)
                else:
                    # Default to date at 00:00 UTC if no time
                    game_time_utc = datetime.strptime(f'{date}T00:00:00', '%Y-%m-%dT%H:%M:%S')
                    game_time_utc = game_time_utc.replace(tzinfo=timezone.utc)
                
                if not game_time_utc:
                    logger.warning(f"Could not parse game time for {game_id}")
                    continue
                
                # Get team names
                home_team_obj = g.get('homeTeam', {})
                away_team_obj = g.get('awayTeam', {})
                home_team = home_team_obj.get('teamTricode', 'UNK')
                away_team = away_team_obj.get('teamTricode', 'UNK')
                
                games.append({
                    'game_id': game_id,
                    'game_date': date,
                    'start_time_utc': game_time_utc,
                    'home_team': home_team,
                    'away_team': away_team,
                    'status': 'Scheduled',
                    'current_period': None,
                    'game_clock': None,
                    'score_home': 0,
                    'score_away': 0
                })
            
            logger.info(f"Fetched {len(games)} games for date {date}")
            return games
            
        except Exception as e:
            logger.error(f"Error fetching games for date {date}: {e}")
            return []
    
    @staticmethod
    def fetch_game_state(game_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch current game state (score, period, clock, status).
        Returns None if game not found.
        """
        try:
            boxscore = boxscoretraditionalv2.BoxScoreTraditionalV2(
                game_id=game_id,
                timeout=NBA_API_TIMEOUT
            )
            data = boxscore.get_dict()
            
            if not data or 'GameHeader' not in data:
                logger.warning(f"No data found for game {game_id}")
                return None
            
            game_header = data['GameHeader'][0]
            
            # Get scores from boxscore
            home_score = game_header.get('HOME_SCORE', 0)
            away_score = game_header.get('VISITOR_SCORE', 0)
            
            return {
                'game_id': game_id,
                'status': game_header.get('GAME_STATUS_TEXT', 'Unknown'),
                'current_period': game_header.get('PERIOD', 0),
                'game_clock': game_header.get('GAME_CLOCK', '0:00'),
                'score_home': home_score,
                'score_away': away_score,
                'home_team': game_header.get('HOME_TEAM_ABBREVIATION', ''),
                'away_team': game_header.get('VISITOR_TEAM_ABBREVIATION', ''),
                'last_updated': datetime.now(timezone.utc)
            }
            
        except Exception as e:
            logger.error(f"Error fetching game state for {game_id}: {e}")
            return None


class OddsDataSource:
    """Odds API data source with caching."""
    
    # TTL values for different trigger types (in seconds)
    TTL_VALUES = {
        'PRE_3H': 3600,
        'PRE_1H': 1800,
        'PRE_10M': 300,
        'HALFTIME': 300,
        'Q3': 300,
        'PERIODIC': 600
    }
    
    def __init__(self, api_key: str, base_url: str = "https://api.the-odds-api.com/v4"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        
    def get_odds(
        self,
        game_id: str,
        reason: str,
        db_path: Path,
        freshness_seconds: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get odds for a game with caching.
        
        Args:
            game_id: NBA game ID
            reason: Why we need odds (trigger type, e.g., 'PRE_3H')
            db_path: Path to database for caching
            freshness_seconds: Override TTL if specified
        
        Returns:
            Odds dict or None if error
        """
        # Use default TTL for reason if not specified
        if freshness_seconds is None:
            freshness_seconds = self.TTL_VALUES.get(reason, 600)
        
        # Check cache first
        cached = OddsCacheStorage.get_cached_odds(game_id, reason, db_path=db_path)
        if cached:
            logger.info(f"Using cached odds for {game_id} ({reason})")
            return cached
        
        # Fetch fresh odds
        logger.info(f"Fetching fresh odds for {game_id} ({reason})")
        return self._fetch_and_cache_odds(game_id, reason, freshness_seconds, db_path)
    
    def _fetch_and_cache_odds(
        self,
        game_id: str,
        reason: str,
        ttl_seconds: int,
        db_path: Path
    ) -> Optional[Dict[str, Any]]:
        """Fetch odds from API and cache result."""
        try:
            # Get game details from DB to find teams
            from core.storage import GameStorage
            game = GameStorage.get_game(game_id, db_path=db_path)
            
            if not game:
                logger.error(f"Game {game_id} not found in database")
                return None
            
            home_team = game['home_team']
            away_team = game['away_team']
            
            # TODO: Implement odds API call
            # For now, returning simple structure
            logger.warning("Odds API integration not yet implemented - returning cached odds if available")
            
            # Check if we have cached odds from earlier
            cached = OddsCacheStorage.get_cached_odds(game_id, reason, db_path=db_path)
            return cached
            
        except Exception as e:
            logger.error(f"Error fetching odds for {game_id}: {e}")
            return None


class CombinedDataSource:
    """Combines NBA and Odds data sources."""
    
    def __init__(self, odds_api_key: str):
        self.nba = NBADataSource()
        self.odds = OddsDataSource(odds_api_key)
    
    def refresh_game_data(
        self,
        game_id: str,
        reason: str,
        db_path: Path
    ) -> Dict[str, Any]:
        """
        Refresh both NBA game state and odds for a game.
        
        Returns dict with:
        - game_state: NBA game state
        - odds: Cached or fresh odds
        """
        game_state = self.nba.fetch_game_state(game_id)
        odds = self.odds.get_odds(game_id, reason, db_path=db_path)
        
        return {
            'game_state': game_state,
            'odds': odds
        }