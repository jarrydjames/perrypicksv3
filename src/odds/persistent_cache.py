"""Persistent odds cache with SQLite backend."""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from src.odds.odds_api import OddsAPIMarketSnapshot
from src.storage.sqlite_store import SQLiteStore


class PersistentOddsCache:
    """SQLite-backed odds cache with TTL."""
    
    # Default TTL: 10 minutes
    TTL_SECONDS = int(os.getenv("ODDS_CACHE_TTL", "600"))
    
    def __init__(self, store: Optional[SQLiteStore] = None):
        self.store = store or SQLiteStore(db_path="data/perrypicks.sqlite")
    
    def _make_cache_key(self, home: str, away: str) -> str:
        """Create cache key from team names."""
        # Normalize names for consistency
        home_norm = home.lower().strip()
        away_norm = away.lower().strip()
        # Sort to avoid duplicate cache for same matchup in different order
        teams = sorted([home_norm, away_norm])
        return f"{teams[0]}_{teams[1]}"
    
    def get(self, home: str, away: str) -> Optional[OddsAPIMarketSnapshot]:
        """Get odds from cache if available and not expired."""
        cache_key = self._make_cache_key(home, away)
        
        with self.store._connect() as con:
            row = con.execute(
                "SELECT response_json, expires_at_utc FROM odds_cache WHERE cache_key=?",
                (cache_key,)
            ).fetchone()
        
        if not row:
            return None
        
        expires_at = row["expires_at_utc"]
        # Check if expired
        if expires_at < time.time():
            return None
        
        # Deserialize
        try:
            data = json.loads(row["response_json"])
            return OddsAPIMarketSnapshot(
                total_points=data.get("total_points"),
                total_over_odds=data.get("total_over_odds"),
                total_under_odds=data.get("total_under_odds"),
                spread_home=data.get("spread_home"),
                spread_home_odds=data.get("spread_home_odds"),
                spread_away_odds=data.get("spread_away_odds"),
                moneyline_home=data.get("moneyline_home"),
                moneyline_away=data.get("moneyline_away"),
                team_total_home=data.get("team_total_home"),
                team_total_home_over_odds=data.get("team_total_home_over_odds"),
                team_total_home_under_odds=data.get("team_total_home_under_odds"),
                team_total_away=data.get("team_total_away"),
                team_total_away_over_odds=data.get("team_total_away_over_odds"),
                team_total_away_under_odds=data.get("team_total_away_under_odds"),
                bookmaker=data.get("bookmaker"),
                last_update=data.get("last_update"),
            )
        except Exception:
            return None
    
    def set(self, home: str, away: str, snapshot: OddsAPIMarketSnapshot) -> None:
        """Store odds in cache with TTL."""
        cache_key = self._make_cache_key(home, away)
        
        # Serialize
        data = {
            "total_points": snapshot.total_points,
            "total_over_odds": snapshot.total_over_odds,
            "total_under_odds": snapshot.total_under_odds,
            "spread_home": snapshot.spread_home,
            "spread_home_odds": snapshot.spread_home_odds,
            "spread_away_odds": snapshot.spread_away_odds,
            "moneyline_home": snapshot.moneyline_home,
            "moneyline_away": snapshot.moneyline_away,
            "team_total_home": snapshot.team_total_home,
            "team_total_home_over_odds": snapshot.team_total_home_over_odds,
            "team_total_home_under_odds": snapshot.team_total_home_under_odds,
            "team_total_away": snapshot.team_total_away,
            "team_total_away_over_odds": snapshot.team_total_away_over_odds,
            "team_total_away_under_odds": snapshot.team_total_away_under_odds,
            "bookmaker": snapshot.bookmaker,
            "last_update": snapshot.last_update,
        }
        
        # Calculate expiration (current time + TTL)
        expires_at = time.time() + self.TTL_SECONDS
        
        with self.store._connect() as con:
            con.execute(
                """INSERT OR REPLACE INTO odds_cache 
                   (cache_key, home, away, response_json, created_ts_utc, expires_at_utc)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    cache_key,
                    home.lower().strip(),
                    away.lower().strip(),
                    json.dumps(data, separators=(",", ":")),
                    time.time(),  # created_ts_utc as epoch time
                    expires_at,
                ),
            )
    
    def is_expired(self, home: str, away: str) -> bool:
        """Check if cached entry is expired."""
        cache_key = self._make_cache_key(home, away)
        
        with self.store._connect() as con:
            row = con.execute(
                "SELECT expires_at_utc FROM odds_cache WHERE cache_key=?",
                (cache_key,)
            ).fetchone()
        
        if not row:
            return True
        
        return row["expires_at_utc"] < time.time()
