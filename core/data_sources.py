"""
Data sources module for PerryPicks v4 Automation System.
Handles NBA API and Odds API calls with caching and rate limiting.
"""

import logging
import requests
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from pathlib import Path

# Use existing data fetching utilities from project
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.storage import OddsCacheStorage
import nba_api_stats.endpoints as nba_endpoints

logger = logging.getLogger(__name__)

# Constants
SEASON = '2025-26'
NBA_API_TIMEOUT = 30
ODDS_API_TIMEOUT = 30


class NBADataSource:
    """NBA API data source."""
    
    @staticmethod
    def fetch_games_for_date(date: str) -> List[Dict[str, Any]]:
        """
        Fetch all games for a specific date (YYYY-MM-DD).
        Returns list of game dicts with game_id, start_time, teams, etc.
        """
        try:
            gamefinder = nba_endpoints.leaguegamefinder.LeagueGameFinder(
                league_id_nullable='00',
                season_nullable=SEASON,
                season_type_nullable='Regular Season',
                game_date_nullable=date
            )
            df = gamefinder.get_data_frames()[0]
            
            # Deduplicate games (LeagueGameFinder returns both home and away views)
            df = df.drop_duplicates(subset=['GAME_ID'], keep='first')
            
            # Convert to list of dicts
            games = []
            for _, row in df.iterrows():
                game_time = row['GAME_DATE']
                game_date = game_time.date() if hasattr(game_time, 'date') else game_time
                
                # Parse matchup to get home/away teams
                matchup = row['MATCHUP']
                if '@' in matchup:
                    away, home = matchup.split('@')
                    home_team = home.strip()
                    away_team = away.strip()
                elif 'vs.' in matchup:
                    home, away = matchup.split('vs.')
                    home_team = home.strip()
                    away_team = away.strip()
                else:
                    logger.warning(f"Could not parse matchup: {matchup}")
                    continue
                
                games.append({
                    'game_id': row['GAME_ID'],
                    'game_date': str(game_date),
                    'start_time_utc': game_time,
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
            boxscore = nba_endpoints.boxscoretraditionalv2.BoxScoreTraditionalV2(
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
        'PRE_3H': 3600,      # 1 hour
        'PRE_1H': 1800,      # 30 minutes
        'PRE_10M': 300,      # 5 minutes
        'HALFTIME': 300,      # 5 minutes
        'Q3': 300,           # 5 minutes
        'PERIODIC': 600       # 10 minutes for periodic polls
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
        """Fetch odds from API and cache the result."""
        try:
            # Get game details from DB to find teams
            from core.storage import GameStorage
            game = GameStorage.get_game(game_id, db_path=db_path)
            
            if not game:
                logger.error(f"Game {game_id} not found in database")
                return None
            
            home_team = game['home_team']
            away_team = game['away_team']
            
            # Build API request for odds
            # Note: We'll need to map NBA team abbreviations to bookmaker team names
            # For now, returning a simple structure
            
            url = f"{self.base_url}/sports/basketball_nba/odds"
            params = {
                'apiKey': self.api_key,
                'regions': 'us',  # US market
                'markets': 'h2h,spreads,totals',  # Moneyline, spreads, totals
                'oddsFormat': 'american',  # American odds
                'dateFormat': 'iso'
            }
            
            response = self.session.get(url, params=params, timeout=ODDS_API_TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse odds for our game
            odds_data = self._parse_odds_response(data, home_team, away_team, game_id)
            
            if odds_data:
                # Cache the result
                OddsCacheStorage.cache_odds(
                    game_id=game_id,
                    reason=reason,
                    payload=odds_data,
                    ttl_seconds=ttl_seconds,
                    endpoint=url,
                    db_path=db_path
                )
                return odds_data
            else:
                logger.warning(f"No odds found for game {game_id}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error fetching odds: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching odds for {game_id}: {e}")
            return None
    
    def _parse_odds_response(
        self,
        data: List[Dict],
        home_team: str,
        away_team: str,
        game_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Parse odds API response to extract relevant odds for our game.
        
        Returns dict with:
        - moneyline: home_ml, away_ml
        - spread: home_spread, away_spread, home_odds, away_odds, book
        - total: total, over_odds, under_odds, book
        """
        if not data:
            return None
        
        # Find our game in the response
        # This is simplified - in practice you'd need to match by team names properly
        for game in data:
            # Try to match teams
            if 'home_team' in game and 'away_team' in game:
                api_home = game['home_team']
                api_away = game['away_team']
                
                # Simple abbreviation match (may need refinement)
                if api_home.startswith(home_team) and api_away.startswith(away_team):
                    return self._extract_odds_from_game(game, game_id)
        
        return None
    
    def _extract_odds_from_game(self, game: Dict, game_id: str) -> Dict[str, Any]:
        """Extract odds from a single game entry."""
        result = {
            'game_id': game_id,
            'moneyline': None,
            'spread': None,
            'total': None,
            'books': []
        }
        
        bookmakers = game.get('bookmakers', [])
        for bookmaker in bookmakers:
            book_name = bookmaker.get('title', 'Unknown')
            markets = bookmaker.get('markets', [])
            
            for market in markets:
                market_key = market.get('key')
                outcomes = market.get('outcomes', [])
                
                if market_key == 'h2h':  # Moneyline
                    for outcome in outcomes:
                        if outcome['name'] == game['home_team']:
                            result['moneyline'] = result['moneyline'] or {}
                            result['moneyline']['home_ml'] = outcome['price']
                            result['moneyline']['home_team'] = outcome['name']
                        elif outcome['name'] == game['away_team']:
                            result['moneyline']['away_ml'] = outcome['price']
                            result['moneyline']['away_team'] = outcome['name']
                
                elif market_key == 'spreads':  # Point spread
                    for outcome in outcomes:
                        if outcome['name'] == game['home_team']:
                            result['spread'] = result['spread'] or {}
                            result['spread']['home_spread'] = outcome.get('point', 0)
                            result['spread']['home_odds'] = outcome['price']
                            result['spread']['home_team'] = outcome['name']
                        elif outcome['name'] == game['away_team']:
                            result['spread']['away_spread'] = outcome.get('point', 0)
                            result['spread']['away_odds'] = outcome['price']
                            result['spread']['away_team'] = outcome['name']
                            result['spread']['book'] = book_name
                
                elif market_key == 'totals':  # Over/under total
                    for outcome in outcomes:
                        if outcome['name'] == 'Over':
                            result['total'] = result['total'] or {}
                            result['total']['total'] = outcome.get('point', 0)
                            result['total']['over_odds'] = outcome['price']
                        elif outcome['name'] == 'Under':
                            result['total']['under_odds'] = outcome['price']
                            result['total']['book'] = book_name
        
        if bookmakers:
            result['books'] = [b.get('title') for b in bookmakers]
        
        return result


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
