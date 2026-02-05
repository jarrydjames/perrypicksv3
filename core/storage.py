"""
Storage layer for PerryPicks v4 Automation System
Handles SQLite database operations with proper schema, migrations, and deduping.
"""

import sqlite3
import json
import logging
from datetime import timedelta  # Keep timedelta for time arithmetic
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager
import hashlib

import pendulum

from core.timezone import now_utc, to_iso, parse_iso_utc, cst_game_date_from_start_time_utc, CST

logger = logging.getLogger(__name__)

# Database path
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "automation.db"


@contextmanager
def get_db_connection(db_path: Path = DEFAULT_DB_PATH):
    """Context manager for database connections."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        conn.close()


def init_database(db_path: Path = DEFAULT_DB_PATH) -> None:
    """
    Initialize database schema with all required tables.
    Run this once on startup; schema uses IF NOT EXISTS for safety.
    """
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        
        # 1. Games table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                start_time_utc TIMESTAMP NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                status TEXT NOT NULL,
                last_seen_utc TIMESTAMP NOT NULL,
                current_period INTEGER,
                game_clock TEXT,
                score_home INTEGER DEFAULT 0,
                score_away INTEGER DEFAULT 0,
                game_date TEXT,  -- YYYY-MM-DD for easy querying
                UNIQUE(game_id)
            )
        """)
        
        # 2. Triggers table (with dedupe constraint)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS triggers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                trigger_type TEXT NOT NULL,
                scheduled_time_utc TIMESTAMP NOT NULL,
                fired_at_utc TIMESTAMP,
                status TEXT NOT NULL DEFAULT 'scheduled',
                created_at_utc TIMESTAMP NOT NULL DEFAULT (datetime('now')),
                payload_json TEXT,
                UNIQUE(game_id, trigger_type, scheduled_time_utc)
            )
        """)
        
        # Index for finding due triggers
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_triggers_due 
            ON triggers(scheduled_time_utc, status)
        """)
        
        # 3. Odds cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS odds_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cache_key TEXT NOT NULL UNIQUE,
                fetched_at_utc TIMESTAMP NOT NULL,
                ttl_seconds INTEGER NOT NULL,
                expires_at_utc TIMESTAMP NOT NULL,
                payload_json TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'odds_api',
                usage_reason TEXT NOT NULL,
                api_endpoint TEXT,
                game_id TEXT
            )
        """)
        
        # Index for finding stale cache
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_odds_cache_key 
            ON odds_cache(cache_key, expires_at_utc)
        """)
        
        # 4. Picks table (bet recommendations)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS picks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                trigger_type TEXT NOT NULL,
                created_at_utc TIMESTAMP NOT NULL DEFAULT (datetime('now')),
                bet_rank INTEGER NOT NULL,
                bet_type TEXT NOT NULL,
                side TEXT NOT NULL,
                line REAL,
                odds REAL NOT NULL,
                book TEXT NOT NULL,
                probability REAL NOT NULL,
                edge REAL NOT NULL,
                rationale TEXT,
                payload_json TEXT,
                graded_status TEXT,  -- 'pending', 'win', 'loss', 'push'
                graded_at_utc TIMESTAMP,
                UNIQUE(game_id, trigger_type, bet_rank, bet_type, side)
            )
        """)
        
        # 5. Tracking snapshots (time-series data)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tracking_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                timestamp_utc TIMESTAMP NOT NULL DEFAULT (datetime('now')),
                trigger_type TEXT,
                poll_type TEXT,  -- 'scheduled', 'halftime', 'q3', 'periodic'
                quarter INTEGER,
                game_clock TEXT,
                score_home INTEGER,
                score_away INTEGER,
                model_probability REAL,
                model_edge REAL,
                live_line REAL,
                live_odds REAL,
                payload_json TEXT
            )
        """)
        
        # Index for time-series queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tracking_game_time 
            ON tracking_snapshots(game_id, timestamp_utc)
        """)
        
        # 6. Discord posts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS discord_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                trigger_type TEXT NOT NULL,
                posted_at_utc TIMESTAMP NOT NULL DEFAULT (datetime('now')),
                channel_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                post_payload_json TEXT NOT NULL,
                UNIQUE(game_id, trigger_type, channel_id, message_id)
            )
        """)
        
        logger.info(f"Database schema initialized at {db_path}")


class GameStorage:
    """Games table operations."""
    
    @staticmethod
    def upsert_game(
        game_id: str,
        start_time_utc: datetime,
        home_team: str,
        away_team: str,
        status: str,
        current_period: Optional[int] = None,
        game_clock: Optional[str] = None,
        score_home: int = 0,
        score_away: int = 0,
        game_date: Optional[str] = None,
        db_path: Path = DEFAULT_DB_PATH
    ) -> None:
        """Insert or update a game."""
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            now_utc_val = now_utc()
            
            # AUTHORITATIVE: Always derive game_date from start_time_utc in CST
            # This prevents any upstream (API) date bucketing bugs from polluting DB.
            if start_time_utc:
                try:
                    # Normalize to pendulum DateTime UTC first
                    if isinstance(start_time_utc, pendulum.DateTime):
                        dt_utc = start_time_utc.in_timezone('UTC')
                    elif isinstance(start_time_utc, str):
                        dt_utc = parse_iso_utc(start_time_utc)
                    else:
                        # Legacy naive datetime -> assume UTC unless you store tz-aware
                        dt_utc = pendulum.instance(start_time_utc).in_timezone('UTC')
                    
                    derived_game_date = cst_game_date_from_start_time_utc(dt_utc, tz=CST)
                    
                    # If caller supplied game_date, keep it only if it matches derived value
                    if game_date and game_date != derived_game_date:
                        logger.warning(
                            f"Game {game_id}: overriding mismatched game_date "
                            f"(incoming={game_date}, derived={derived_game_date})"
                        )
                    game_date = derived_game_date
                except Exception as e:
                    logger.warning(f"Game {game_id}: failed to derive CST game_date from start_time_utc: {e}")
            
            # Convert datetime to ISO string for SQLite (using pendulum's to_iso8601_string)
            if isinstance(start_time_utc, pendulum.DateTime):
                start_time_str = to_iso(start_time_utc)
            elif isinstance(start_time_utc, str):
                # Already an ISO string (from DB or API)
                start_time_str = start_time_utc
            elif start_time_utc:
                # Handle legacy datetime objects
                start_time_str = start_time_utc.isoformat()
            else:
                start_time_str = None
            now_str = to_iso(now_utc_val)
            
            cursor.execute("""
                INSERT INTO games (
                    game_id, start_time_utc, home_team, away_team, status,
                    last_seen_utc, current_period, game_clock, score_home, score_away, game_date
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_id) DO UPDATE SET
                    status = excluded.status,
                    last_seen_utc = excluded.last_seen_utc,
                    current_period = excluded.current_period,
                    game_clock = excluded.game_clock,
                    score_home = excluded.score_home,
                    score_away = excluded.score_away
            """, (
                game_id, start_time_str, home_team, away_team, status,
                now_str, current_period, game_clock, score_home, score_away, game_date
            ))
    
    @staticmethod
    def get_game(game_id: str, db_path: Path = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
        """Get a single game by ID."""
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM games WHERE game_id = ?", (game_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    @staticmethod
    def get_games_for_date(
        date: str,  # YYYY-MM-DD
        status: Optional[str] = None,
        db_path: Path = DEFAULT_DB_PATH
    ) -> List[Dict[str, Any]]:
        """Get all games for a specific date."""
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            if status:
                cursor.execute(
                    "SELECT * FROM games WHERE game_date = ? AND status = ?",
                    (date, status)
                )
            else:
                cursor.execute(
                    "SELECT * FROM games WHERE game_date = ?",
                    (date,)
                )
            return [dict(row) for row in cursor.fetchall()]
    
    @staticmethod
    def get_active_games(db_path: Path = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
        """Get all games that are in-progress."""
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM games WHERE status IN ('In Progress', 'Halftime') ORDER BY start_time_utc"
            )
            return [dict(row) for row in cursor.fetchall()]
    
    @staticmethod
    def has_games_for_date(
        date: str,  # YYYY-MM-DD
        db_path: Path = DEFAULT_DB_PATH
    ) -> bool:
        """Check if any games exist for a specific date."""
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) as count FROM games WHERE game_date = ?",
                (date,)
            )
            row = cursor.fetchone()
            return row['count'] > 0 if row else False
    


class TriggerStorage:
    """Triggers table operations."""
    
    TRIGGER_TYPES = ['PRE_3H', 'PRE_1H', 'PRE_10M', 'HALFTIME', 'Q3']
    
    @staticmethod
    def schedule_trigger(
        game_id: str,
        trigger_type: str,
        scheduled_time_utc: datetime,
        payload: Optional[Dict] = None,
        db_path: Path = DEFAULT_DB_PATH
    ) -> bool:
        """
        Schedule a trigger. Returns True if scheduled, False if already exists.
        Uses unique constraint to prevent duplicates.
        """
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            try:
                # Convert datetime to ISO string for SQLite
                scheduled_time_str = scheduled_time_utc.isoformat() if scheduled_time_utc else None
                
                cursor.execute("""
                    INSERT INTO triggers (
                        game_id, trigger_type, scheduled_time_utc, payload_json
                    )
                    VALUES (?, ?, ?, ?)
                """, (
                    game_id, trigger_type, scheduled_time_str,
                    json.dumps(payload) if payload else None
                ))
                return True
            except sqlite3.IntegrityError:
                logger.debug(f"Trigger already exists: {game_id} {trigger_type} at {scheduled_time_utc}")
                return False
    
    @staticmethod
    def get_due_triggers(
        window_start: datetime,
        window_end: datetime,
        db_path: Path = DEFAULT_DB_PATH
    ) -> List[Dict[str, Any]]:
        """Get all triggers that are due within the time window and not yet fired."""
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            # Convert datetime to ISO string for SQLite
            window_start_str = window_start.isoformat() if window_start else None
            window_end_str = window_end.isoformat() if window_end else None
            
            cursor.execute("""
                SELECT * FROM triggers
                WHERE status = 'scheduled'
                  AND scheduled_time_utc >= ?
                  AND scheduled_time_utc <= ?
                ORDER BY scheduled_time_utc
            """, (window_start_str, window_end_str))
            return [dict(row) for row in cursor.fetchall()]
    
    @staticmethod
    def mark_triggered(
        trigger_id: int,
        fired_at_utc: datetime,
        payload: Optional[Dict] = None,
        db_path: Path = DEFAULT_DB_PATH
    ) -> None:
        """Mark a trigger as fired."""
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            # Convert datetime to ISO string for SQLite
            fired_at_str = fired_at_utc.isoformat() if fired_at_utc else None
            
            cursor.execute("""
                UPDATE triggers
                SET status = 'fired',
                    fired_at_utc = ?,
                    payload_json = COALESCE(?, payload_json)
                WHERE id = ?
            """, (fired_at_str, json.dumps(payload) if payload else None, trigger_id))
    
    @staticmethod
    def check_trigger_exists(
        game_id: str,
        trigger_type: str,
        db_path: Path = DEFAULT_DB_PATH
    ) -> bool:
        """Check if a trigger of a given type exists for a game."""
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM triggers WHERE game_id = ? AND trigger_type = ?",
                (game_id, trigger_type)
            )
            return cursor.fetchone()[0] > 0
    
    @staticmethod
    def delete_trigger(
        game_id: str,
        trigger_type: str,
        db_path: Path = DEFAULT_DB_PATH
    ) -> int:
        """
        Delete a trigger by (game_id, trigger_type).
        
        Returns number of rows deleted.
        
        Args:
            game_id: Game identifier (e.g., '0022500731' or 'DAILY_20260204')
            trigger_type: Type of trigger ('PRE_3H', 'PRE_1H', 'PRE_10M', 'HALFTIME', 'Q3', 'DAILY_SUMMARY')
            db_path: Path to database
        
        Example:
            >>> deleted = TriggerStorage.delete_trigger('DAILY_20260204', 'DAILY_SUMMARY')
            >>> print(f"Deleted {deleted} trigger(s)")
            Deleted 1 trigger(s)
        """
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM triggers WHERE game_id = ? AND trigger_type = ?",
                (game_id, trigger_type)
            )
            conn.commit()
            return cursor.rowcount
    
    @staticmethod
    def check_trigger_fired(
        game_id: str,
        trigger_type: str,
        db_path: Path = DEFAULT_DB_PATH
    ) -> bool:
        """
        Check if a trigger has already been fired.
        
        This is used to prevent duplicate game-state triggers (HALFTIME, Q3).
        Unlike check_trigger_exists, this only checks for FIRED triggers,
        allowing re-firing if a game reaches the state again after a
        missed scheduled trigger.
        """
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT COUNT(*) FROM triggers 
                   WHERE game_id = ? 
                   AND trigger_type = ? 
                   AND status = 'fired'""",
                (game_id, trigger_type)
            )
            return cursor.fetchone()[0] > 0
    
    @staticmethod
    def get_triggers_for_game(
        game_id: str,
        db_path: Path = DEFAULT_DB_PATH
    ) -> List[Dict[str, Any]]:
        """Get all triggers for a game."""
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM triggers WHERE game_id = ?", (game_id,))
            return [dict(row) for row in cursor.fetchall()]


class OddsCacheStorage:
    """Odds cache table operations."""
    
    @staticmethod
    def _generate_cache_key(
        game_id: str,
        reason: str,
        endpoint: Optional[str] = None
    ) -> str:
        """Generate a unique cache key."""
        key_parts = [game_id, reason]
        if endpoint:
            key_parts.append(endpoint)
        key_str = ":".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @staticmethod
    def get_cached_odds(
        game_id: str,
        reason: str,
        endpoint: Optional[str] = None,
        db_path: Path = DEFAULT_DB_PATH
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached odds if valid. Returns None if not found or expired.
        """
        cache_key = OddsCacheStorage._generate_cache_key(game_id, reason, endpoint)
        
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT payload_json, expires_at_utc
                FROM odds_cache
                WHERE cache_key = ? AND expires_at_utc > datetime('now')
                ORDER BY fetched_at_utc DESC
                LIMIT 1
            """, (cache_key,))
            
            row = cursor.fetchone()
            if row:
                logger.debug(f"Odds cache hit: {game_id} {reason} (expires {row['expires_at_utc']})")
                return json.loads(row['payload_json'])
            
            logger.debug(f"Odds cache miss: {game_id} {reason}")
            return None
    
    @staticmethod
    def cache_odds(
        game_id: str,
        reason: str,
        payload: Dict[str, Any],
        ttl_seconds: int,
        endpoint: Optional[str] = None,
        db_path: Path = DEFAULT_DB_PATH
    ) -> None:
        """
        Cache odds with TTL.
        """
        cache_key = OddsCacheStorage._generate_cache_key(game_id, reason, endpoint)
        now_utc_val = now_utc()
        expires_at_utc = now_utc_val + pendulum.duration(seconds=ttl_seconds)
        
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO odds_cache (
                    cache_key, fetched_at_utc, ttl_seconds, expires_at_utc,
                    payload_json, source, usage_reason, api_endpoint, game_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cache_key, now_utc, ttl_seconds, expires_at_utc,
                json.dumps(payload), 'odds_api', reason, endpoint, game_id
            ))
            logger.info(f"Cached odds for {game_id} {reason} (TTL: {ttl_seconds}s, expires: {expires_at_utc})")


class PickStorage:
    """Picks (bet recommendations) table operations."""
    
    @staticmethod
    def store_pick(
        game_id: str,
        trigger_type: str,
        bet_rank: int,
        bet_type: str,
        side: str,
        odds: float,
        book: str,
        probability: float,
        edge: float,
        line: Optional[float] = None,
        rationale: Optional[str] = None,
        payload: Optional[Dict] = None,
        db_path: Path = DEFAULT_DB_PATH
    ) -> int:
        """Store a pick. Returns pick ID."""
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO picks (
                    game_id, trigger_type, bet_rank, bet_type, side,
                    line, odds, book, probability, edge, rationale, payload_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game_id, trigger_type, bet_rank, bet_type, side,
                line, odds, book, probability, edge, rationale,
                json.dumps(payload) if payload else None
            ))
            return cursor.lastrowid
    
    @staticmethod
    def get_picks_for_game(
        game_id: str,
        trigger_type: Optional[str] = None,
        db_path: Path = DEFAULT_DB_PATH
    ) -> List[Dict[str, Any]]:
        """Get picks for a game."""
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            if trigger_type:
                cursor.execute(
                    "SELECT * FROM picks WHERE game_id = ? AND trigger_type = ? ORDER BY bet_rank",
                    (game_id, trigger_type)
                )
            else:
                cursor.execute(
                    "SELECT * FROM picks WHERE game_id = ? ORDER BY created_at_utc, bet_rank",
                    (game_id,)
                )
            return [dict(row) for row in cursor.fetchall()]
    
    @staticmethod
    def grade_pick(
        pick_id: int,
        graded_status: str,
        db_path: Path = DEFAULT_DB_PATH
    ) -> None:
        """Grade a pick (win/loss/push)."""
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE picks
                SET graded_status = ?, graded_at_utc = datetime('now')
                WHERE id = ?
            """, (graded_status, pick_id))


class TrackingStorage:
    """Tracking snapshots table operations (time-series data)."""
    
    @staticmethod
    def store_snapshot(
        game_id: str,
        timestamp_utc: datetime,
        poll_type: str,
        quarter: Optional[int] = None,
        game_clock: Optional[str] = None,
        score_home: Optional[int] = None,
        score_away: Optional[int] = None,
        model_probability: Optional[float] = None,
        model_edge: Optional[float] = None,
        live_line: Optional[float] = None,
        live_odds: Optional[float] = None,
        payload: Optional[Dict] = None,
        trigger_type: Optional[str] = None,
        db_path: Path = DEFAULT_DB_PATH
    ) -> int:
        """Store a tracking snapshot."""
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO tracking_snapshots (
                    game_id, timestamp_utc, poll_type, quarter, game_clock,
                    score_home, score_away, model_probability, model_edge,
                    live_line, live_odds, payload_json, trigger_type
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game_id, timestamp_utc, poll_type, quarter, game_clock,
                score_home, score_away, model_probability, model_edge,
                live_line, live_odds, json.dumps(payload) if payload else None, trigger_type
            ))
            return cursor.lastrowid
    
    @staticmethod
    def get_timeseries(
        game_id: str,
        db_path: Path = DEFAULT_DB_PATH
    ) -> List[Dict[str, Any]]:
        """
        Get time-series data for a game, ordered chronologically.
        Used for live tracking charts.
        """
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM tracking_snapshots
                WHERE game_id = ?
                ORDER BY timestamp_utc ASC
            """, (game_id,))
            return [dict(row) for row in cursor.fetchall()]


class DiscordPostStorage:
    """Discord posts table operations."""
    
    @staticmethod
    def store_post(
        game_id: str,
        trigger_type: str,
        channel_id: str,
        message_id: str,
        payload: Dict[str, Any],
        db_path: Path = DEFAULT_DB_PATH
    ) -> int:
        """Store a Discord post."""
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO discord_posts (
                    game_id, trigger_type, channel_id, message_id, post_payload_json
                )
                VALUES (?, ?, ?, ?, ?)
            """, (
                game_id, trigger_type, channel_id, message_id, json.dumps(payload)
            ))
            return cursor.lastrowid
    
    @staticmethod
    def get_post(
        game_id: str,
        trigger_type: str,
        channel_id: str,
        db_path: Path = DEFAULT_DB_PATH
    ) -> Optional[Dict[str, Any]]:
        """Get a Discord post for replying/editing."""
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM discord_posts
                WHERE game_id = ? AND trigger_type = ? AND channel_id = ?
                ORDER BY posted_at_utc DESC
                LIMIT 1
            """, (game_id, trigger_type, channel_id))
            row = cursor.fetchone()
            return dict(row) if row else None
