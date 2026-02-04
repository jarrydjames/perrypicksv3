"""Game data fetching module.

Provides fetch_game_by_id() function to get game data by ID.
Uses the boxscore API (fetch_box) to get full game data.
"""

from typing import Any, Dict, Optional
import requests
import pandas as pd

CDN_BOX = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"

# User-Agent headers to avoid NBA.com blocking
NBA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
}

def fetch_json(url: str, max_retries: int = 3) -> dict:
    """Fetch JSON from NBA.com CDN with proper headers and retry logic.
    
    Args:
        url: URL to fetch
        max_retries: Number of retries on 403/429 errors
        
    Returns:
        JSON response as dict
        
    Raises:
        requests.HTTPError: If all retries fail
    """
    import time
    
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=25, headers=NBA_HEADERS)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            # Retry on rate limiting (429) or forbidden (403) errors
            if e.response.status_code in (403, 429) and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                import logging
                logging.warning(f"NBA.com API returned {e.response.status_code}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            # For other errors or final retry, re-raise
            raise


def fetch_game_by_id(game_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch game data by ID from boxscore API.
    
    This fetches the full game data with periods, team stats, etc.
    
    Args:
        game_id: NBA game ID
        
    Returns:
        Game dict if found, None if not found
    """
    try:
        data = fetch_json(CDN_BOX.format(gid=game_id))
        game = data.get("game")
        
        if not game:
            return None
            
        # Return the full game dict as-is
        # The runtime expects game['homeTeam'], game['awayTeam'] with periods
        return game
        
    except requests.HTTPError as e:
        import logging
        # Special handling for 403 errors
        if e.response.status_code == 403:
            # Game not started yet or boxscore not available
            # Try to get basic info from schedule
            logging.info(f"Boxscore not available for {game_id} (game not started), checking schedule")
            return _get_game_from_schedule(game_id)
        else:
            # Other HTTP errors - log and return None
            logging.warning(f"Failed to fetch game {game_id}: {e}")
            return None
    except Exception as e:
        import logging
        logging.warning(f"Failed to fetch game {game_id}: {e}")
        return None


def _get_game_from_schedule(game_id: str) -> Optional[Dict[str, Any]]:
    """
    Get basic game info from schedule API (fallback when boxscore unavailable).
    
    This is used for games that haven't started yet and don't have
    boxscore data available.
    
    Args:
        game_id: NBA game ID
        
    Returns:
        Minimal game dict with homeTeam, awayTeam info if found, None if not found
    """
    try:
        # Fetch schedule
        SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
        
        response = requests.get(SCHEDULE_URL, timeout=30, headers=NBA_HEADERS)
        response.raise_for_status()
        data = response.json()
        
        league = data.get('leagueSchedule', {})
        game_dates = league.get('gameDates', [])
        
        # Search for the game
        for gd in game_dates:
            games = gd.get('games', [])
            for game in games:
                if game.get('gameId') == game_id:
                    # Found the game - return minimal game dict
                    # Structure matches boxscore format
                    return {
                        'gameId': game_id,
                        'gameStatus': 1,  # Scheduled
                        'gameTimeUTC': game.get('gameTimeUTC', ''),
                        'homeTeam': {
                            'teamId': game.get('homeTeam', {}).get('teamId', ''),
                            'teamTricode': game.get('homeTeam', {}).get('teamTricode', ''),
                            'teamName': game.get('homeTeam', {}).get('teamName', ''),
                            'teamCity': game.get('homeTeam', {}).get('teamCity', ''),
                            'periods': []  # No period data before game starts
                        },
                        'awayTeam': {
                            'teamId': game.get('awayTeam', {}).get('teamId', ''),
                            'teamTricode': game.get('awayTeam', {}).get('teamTricode', ''),
                            'teamName': game.get('awayTeam', {}).get('teamName', ''),
                            'teamCity': game.get('awayTeam', {}).get('teamCity', ''),
                            'periods': []  # No period data before game starts
                        },
                        'gameClock': 'PT00M00.00S',
                        'period': 0
                    }
        
        # Game not found in schedule
        return None
        
    except Exception as e:
        import logging
        logging.warning(f"Failed to get game {game_id} from schedule: {e}")
        return None
