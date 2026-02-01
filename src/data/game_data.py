"""Game data fetching module.

Provides fetch_game_by_id() function to get game data by ID.
Uses the scoreboard API internally.
"""

from typing import Any, Dict, Optional
from datetime import date

from src.data.scoreboard import fetch_scoreboard, ScoreboardGame


def fetch_game_by_id(game_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch game data by ID from scoreboard.
    
    Args:
        game_id: NBA game ID
        
    Returns:
        Game dict if found, None if not found
    """
    # Fetch scoreboard for today
    games = fetch_scoreboard(date=date.today(), include_live=True)
    
    # Find game by ID
    for game in games:
        if game.game_id == game_id:
            # Convert ScoreboardGame to dict
            return {
                'gameId': game.game_id,
                'gameCode': game.game_code,
                'gameStatus': game.game_status,
                'homeTeam': game.home_team,
                'awayTeam': game.away_team,
                'homeScore': game.home_score,
                'awayScore': game.away_score,
                'period': game.period,
                'gameClock': game.game_clock,
                'gameTimeUTC': game.game_time_utc,
            }
    
    # Not found
    return None
