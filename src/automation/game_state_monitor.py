"""Live Game State Monitor for PerryPicks v3.

Monitors NBA games in real-time and tracks period/time for trigger detection.
Runs continuously as background service.
"""

from __future__ import annotations
import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.data.scoreboard import fetch_scoreboard
from src.data.game_data import fetch_game_by_id
from core.storage import GameStorage

logger = logging.getLogger(__name__)


class GameState:
    """Represents current state of a game."""
    
    def __init__(
        self,
        game_id: str,
        status: str,
        period: int = 0,
        time_remaining: str = "0:00",
        home_score: int = 0,
        away_score: int = 0,
        home_name: Optional[str] = None,
        away_name: Optional[str] = None,
    ):
        self.game_id = game_id
        self.status = status  # 'scheduled', 'live', 'halftime', 'finished', etc.
        self.period = period  # Current quarter (1, 2, 3, 4)
        self.time_remaining = time_remaining  # Time remaining in period (e.g., "5:32", "0:00")
        self.home_score = home_score
        self.away_score = away_score
        self.home_name = home_name  # Team full name
        self.away_name = away_name  # Team full name
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "game_id": self.game_id,
            "status": self.status,
            "period": self.period,
            "time_remaining": self.time_remaining,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "home_name": self.home_name,
            "away_name": self.away_name,
            "last_updated": self.last_updated.isoformat(),
        }


class GameStateMonitor:
    """Monitor live NBA games and track game state.
    
    Polls NBA API periodically to update game states.
    Detects trigger conditions (halftime, Q3-5min).
    """
    
    def __init__(
        self,
        poll_interval_seconds: int = 30,
        max_retries: int = 3,
    ):
        """Initialize game state monitor.
        
        Args:
            poll_interval_seconds: How often to poll NBA API (default: 30s)
            max_retries: Max retries for API calls
        """
        self.poll_interval = poll_interval_seconds
        self.max_retries = max_retries
        self.game_states: Dict[str, GameState] = {}
        self.running = False
        self.storage = GameStorage()
        
        logger.info(
            f"Game State Monitor initialized. "
            f"Poll interval: {poll_interval_seconds}s"
        )
    
    def update_game_state(self, game_id: str) -> Optional[GameState]:
        """Update game state for a single game.
        
        Args:
            game_id: Game ID to update
            
        Returns:
            Updated GameState or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                # Fetch game data to get period/time
                game_data = fetch_game_by_id(game_id)
                
                if not game_data:
                    logger.warning(f"No game data for {game_id}")
                    return None
                
                # Extract game state
                period = game_data.get("period", 0)
                
                # Parse game clock (format: "PT12M30.00S")
                game_clock = game_data.get("gameClock", "PT00M00.00S")
                time_remaining = self._parse_game_clock(game_clock)
                
                game_status = game_data.get("gameStatus", 1)
                
                # Calculate scores from periods
                home_score, away_score = self._calculate_scores(game_data)
                
                # Detect halftime BEFORE normalizing status
                # Halftime = after Q2 ends, before Q3 starts
                # Correct logic: Q2 has finished (time_remaining is "00:00" or 0:00)
                home_team_data = game_data.get("homeTeam", {})
                away_team_data = game_data.get("awayTeam", {})
                home_periods = len(home_team_data.get("periods", []))
                away_periods = len(away_team_data.get("periods", []))
                
                # DEBUG: Log period data
                if period == 2 and time_remaining_zero:
                    logger.info(
                        f"[PERIOD DEBUG] {game_id}: "
                        f"home_periods={home_periods}, away_periods={away_periods}, "
                        f"home_periods_data={home_team_data.get('periods', [])}, "
                        f"away_periods_data={away_team_data.get('periods', [])}, "
                        f"game_status={game_status}"
                    )
                
                # FIX: Remove game_status check - rely on period and time only
                # The API might have race conditions where game_status updates before we check
                # If period == 2 and time_remaining == 00:00, it's halftime
                # regardless of what game_status says
                is_halftime = (
                    home_periods >= 2 and      # At least 2 periods (Q1 and Q2 completed)
                    away_periods >= 2 and      # Both teams have at least 2 periods
                    period == 2 and            # Currently at period 2 (end of Q2)
                    time_remaining_zero         # Time remaining is 00:00 (Q2 finished)
                )
                )
                
                # DEBUG: Log halftime detection details
                if period == 2 and time_remaining_zero:
                    logger.info(
                        f"[HALFTIME DEBUG] {game_id}: "
                        f"home_periods={home_periods}, away_periods={away_periods}, "
                        f"period={period}, game_status={game_status}, "
                        f"time_remaining_zero={time_remaining_zero}, "
                        f"is_halftime={is_halftime}"
                    )
                
                # Normalize status
                if is_halftime:
                    status = "halftime"
                    logger.info(f"✅ HALFTIME STATUS SET for {game_id}")
                elif game_status >= 6:  # Final (gameStatus 6 = Final, per automation_ui.py)
                    status = "finished"
                elif period > 0:
                    status = "live"
                else:
                    status = "scheduled"
                
                # Extract team names
                home_team = game_data.get("homeTeam", {})
                away_team = game_data.get("awayTeam", {})
                home_name = home_team.get("teamName", home_team.get("fullName", "Home"))
                away_name = away_team.get("teamName", away_team.get("fullName", "Away"))
                
                # Log for debugging
                if is_halftime:
                    logger.info(
                        f"HALFTIME DETECTED: {game_id} "
                        f"(periods: {home_periods}/{away_periods}, period: {period}, "
                        f"gameStatus: {game_status}, time_remaining: {time_remaining})"
                    )
                elif period == 2 and game_status == 2 and not time_remaining_zero:
                    # Log when we're in Q2 but not yet at halftime
                    logger.debug(
                        f"Q2 IN PROGRESS: {game_id} "
                        f"(periods: {home_periods}/{away_periods}, "
                        f"time_remaining: {time_remaining}, NOT HALFTIME YET)"
                    )
                
                # Create game state
                game_state = GameState(
                    game_id=game_id,
                    status=status,
                    period=period,
                    time_remaining=time_remaining,
                    home_score=home_score,
                    away_score=away_score,
                    home_name=home_name,
                    away_name=away_name,
                )
                
                # Update cache
                self.game_states[game_id] = game_state
                
                logger.info(
                    f"Updated {game_id}: {status} Q{period} {time_remaining} "
                    f"({away_score}-{home_score})"
                )
                
                return game_state
            
            except Exception as e:
                logger.error(
                    f"Attempt {attempt + 1}/{self.max_retries}: "
                    f"Failed to update game state for {game_id}: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def _parse_game_clock(self, game_clock: str) -> str:
        """Parse NBA game clock format to MM:SS.
        
        Args:
            game_clock: Game clock string (e.g., "PT12M30.00S")
            
        Returns:
            Time remaining as MM:SS string
        """
        try:
            # Format: PT12M30.00S or PT00M00.00S
            parts = game_clock.replace("PT", "").replace("S", "").split("M")
            if len(parts) >= 2:
                minutes = int(parts[0])
                seconds = int(float(parts[1]))
                return f"{minutes:02d}:{seconds:02d}"
            return "0:00"
        except Exception as e:
            logger.warning(f"Error parsing game clock '{game_clock}': {e}")
            return "0:00"
    
    def _calculate_scores(self, game_data: dict) -> tuple[int, int]:
        """Calculate scores from period data.
        
        Args:
            game_data: Game data dict
            
        Returns:
            Tuple of (home_score, away_score)
        """
        home_score = 0
        away_score = 0
        
        try:
            home_team = game_data.get("homeTeam", {})
            away_team = game_data.get("awayTeam", {})
            
            home_periods = home_team.get("periods", [])
            away_periods = away_team.get("periods", [])
            
            # Sum scores from periods
            for period in home_periods:
                home_score += period.get("score", 0)
            
            for period in away_periods:
                away_score += period.get("score", 0)
        
        except Exception as e:
            logger.warning(f"Error calculating scores: {e}")
        
        return home_score, away_score
    
    def update_all_games(self) -> List[GameState]:
        """Update game state for all active games.
        
        Returns:
            List of updated GameState objects
        """
        updated_states = []
        
        try:
            # Get today's games
            from datetime import date
            today = date.today()
            games = fetch_scoreboard(today)
            
            if not games:
                logger.warning("No games found for today")
                return []
            
            # Update each game
            for game in games:
                # ScoreboardGame object
                game_id = getattr(game, 'game_id', None)
                if not game_id:
                    # Try other attributes
                    game_id = getattr(game, 'gameId', None)
                if not game_id:
                    continue
                
                # Skip finished games
                if self.game_states.get(str(game_id)):
                    if self.game_states[str(game_id)].status == "finished":
                        continue
                
                state = self.update_game_state(str(game_id))
                if state:
                    updated_states.append(state)
        
        except Exception as e:
            logger.error(f"Error updating all games: {e}")
        
        return updated_states
    
    def get_game_state(self, game_id: str) -> Optional[GameState]:
        """Get current game state for a game."""
        return self.game_states.get(game_id)
    
    def get_all_states(self) -> Dict[str, GameState]:
        """Get all current game states."""
        return self.game_states.copy()
    
    def stop_monitoring_game(self, game_id: str) -> bool:
        """Stop monitoring a specific game.
        
        Args:
            game_id: Game ID to stop monitoring
            
        Returns:
            True if game was being monitored and stopped, False otherwise
        """
        if game_id in self.game_states:
            game_state = self.game_states[game_id]
            logger.info(f"Stopping monitoring for {game_id}: {game_state.home_name} vs {game_state.away_name}")
            del self.game_states[game_id]
            return True
        else:
            logger.warning(f"Game {game_id} not being monitored, cannot stop")
            return False
    
    def start(self):
        """Start monitoring loop."""
        self.running = True
        logger.info("Game State Monitor started")
        
        while self.running:
            try:
                logger.info("Updating game states...")
                updated = self.update_all_games()
                logger.info(f"Updated {len(updated)} games")
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # Sleep until next poll
            time.sleep(self.poll_interval)
        
        logger.info("Game State Monitor stopped")
    
    def stop(self):
        """Stop monitoring loop."""
        logger.info("Stopping Game State Monitor...")
        self.running = False
    
    def is_halftime(self, game_id: str) -> bool:
        """Check if game is at halftime.
        
        Args:
            game_id: Game ID to check
            
        Returns:
            True if at halftime
        """
        state = self.get_game_state(game_id)
        if not state:
            return False
        
        return state.status == "halftime"
    
    def is_q3_trigger(self, game_id: str) -> bool:
        """Check if game has reached the Q3 trigger point."""
        state = self.get_game_state(game_id)
        if not state:
            return False

        return state.period >= 3 and state.status in ("live", "halftime")

    def is_q3_five_minutes_left(self, game_id: str) -> bool:
        """Backward-compatible alias for Q3 trigger detection."""
        return self.is_q3_trigger(game_id)