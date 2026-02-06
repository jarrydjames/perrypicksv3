"""
Data sources module for PerryPicks v4 Automation System.
Handles NBA API and Odds API calls with caching and rate limiting.
"""

import logging
import requests
import time
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from datetime import datetime, timedelta
import sys
sys.path.append(str(Path(__file__).parent.parent))

import pendulum

from core.storage import OddsCacheStorage
from core.timezone import parse_iso_utc, to_iso, parse_date_str, now_utc, cst_game_date_from_start_time_utc, CST
from core.validation import validate_future_datetime, validate_nba_schedule

logger = logging.getLogger(__name__)

# Constants
SEASON = '2025-26'
NBA_API_TIMEOUT = 30
ODDS_API_TIMEOUT = 30

# CDN-based NBA API endpoints (reliable, no timeouts)
SCHEDULE_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"
BOXSCORE_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{gid}.json"

# Long-term-stable schedule source with canonical per-game datetimes
# (Used widely for full season schedules; contains consistent game date/time fields)
# https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/{YEAR}/league/00_full_schedule.json
FULL_SCHEDULE_URL_TMPL = "https://data.nba.com/data/10s/v2015/json/mobile_teams/nba/{year}/league/00_full_schedule.json"

# Headers that work with CDN
NBA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.nba.com/",
    "Accept-Language": "en-US,en;q=0.9",
}


class NBADataSource:
    """NBA API data source with caching and retry logic.
    
    Uses CDN-based APIs for reliability:
    - Schedule API: Fetch game schedule
    - Boxscore API: Fetch game state and detailed stats
    
    Note: Pregame model uses separate Team Stats API (stats.nba.com)
    for advanced team statistics - this is kept as-is.
    """
    
    # Simple in-memory cache (class-level, shared across instances)
    # Format: {key: (value, timestamp)}
    _cache: Dict[str, Tuple[Any, float]] = {}

    # Full schedule index cache:
    # { "season_key": ( {"00225...": pendulum.DateTime(UTC), ...}, timestamp ) }
    _full_schedule_index: Dict[str, Tuple[Dict[str, pendulum.DateTime], float]] = {}
    FULL_SCHEDULE_TTL_SECONDS = 6 * 3600  # 6h (schedule changes rarely)

    # Cache TTL values (in seconds)
    CACHE_TTL = {
        'schedule': 3600,      # 1 hour - schedule doesn't change often
        'game_state': 30,      # 30 seconds - game state changes frequently
        'game_stats': 300,     # 5 minutes - team stats change slowly
    }
    
    @classmethod
    def _get_cached(cls, cache_type: str, key: str) -> Optional[Any]:
        """Get from cache if fresh."""
        full_key = f'{cache_type}:{key}'
        if full_key not in cls._cache:
            return None
        
        value, timestamp = cls._cache[full_key]
        ttl = cls.CACHE_TTL.get(cache_type, 600)
        
        if time.time() - timestamp > ttl:
            # Expired - remove from cache
            del cls._cache[full_key]
            return None
        
        return value
    
    @classmethod
    def _set_cached(cls, cache_type: str, key: str, value: Any):
        """Set cache value."""
        full_key = f'{cache_type}:{key}'
        cls._cache[full_key] = (value, time.time())
    
    @staticmethod
    def _parse_nba_datetime(dt_input: Any) -> Optional[pendulum.DateTime]:
        """
        Parse NBA API datetime to timezone-aware UTC datetime.
        
        NBA API returns various formats:
        - pandas Timestamp (with or without timezone)
        - ISO 8601 strings (with or without timezone)
        - Naive datetime objects
        
        Always returns timezone-aware pendulum.DateTime in UTC.
        """
        if dt_input is None:
            return None
        
        try:
            # Handle pandas Timestamp
            if hasattr(dt_input, 'to_pydatetime'):
                dt = dt_input.to_pydatetime()
                # Convert to pendulum DateTime
                if isinstance(dt, datetime):
                    return pendulum.instance(dt).in_timezone('UTC')
                return pendulum.parse(str(dt)).in_timezone('UTC')
            
            # Handle string inputs
            elif isinstance(dt_input, str):
                dt_str = dt_input.strip()
                
                # Try ISO format first (using pendulum)
                try:
                    dt = pendulum.parse(dt_str, strict=True)
                    return dt.in_timezone('UTC')
                except Exception:
                    pass
                
                # Try parsing other common formats
                for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
                    try:
                        # Use datetime.strptime for these, then convert to pendulum
                        import datetime as dt_module
                        dt = dt_module.datetime.strptime(dt_str, fmt)
                        return pendulum.instance(dt, tz='UTC')
                    except ValueError:
                        continue
                
                return None
            
            # Handle datetime objects (old datetime module)
            elif isinstance(dt_input, datetime):
                return pendulum.instance(dt_input, tz='UTC')
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to parse datetime {dt_input}: {e}")
            return None
    
    @classmethod
    def _parse_nba_schedule_time(
        cls,
        api_date_str: str,
        game_time_utc_placeholder: str
    ) -> Optional[pendulum.DateTime]:
        """
        Parse NBA schedule API time to timezone-aware UTC datetime.

        CRITICAL: The NBA schedule API uses a non-intuitive format:
        - api_date_str (e.g., "02/05/2026") is Eastern Time date of game
        - game_time_utc_placeholder (e.g., "1900-01-01T19:00:00Z") uses 1900-01-01 as
          a placeholder date, but TIME portion is the Eastern Time start time!

        So for a game at 7:00 PM ET on Feb 5, 2026:
        - api_date_str = "02/05/2026"
        - game_time_utc_placeholder = "1900-01-01T19:00:00Z"
        - Correct parsing:
          1. Parse ET date from api_date_str: Feb 5, 2026 at midnight ET
          2. Extract ET time from placeholder: 19:00
          3. Combine: Feb 5, 2026 19:00 ET
          4. Convert to UTC: Feb 6, 2026 00:00:00Z

        Args:
            api_date_str: Date string from API (MM/DD/YYYY format)
            game_time_utc_placeholder: Time string with placeholder date (1900-01-01THH:MM:SSZ)

        Returns:
            pendulum.DateTime in UTC, or None if parsing fails
        """
        import datetime as dt_module

        try:
            # Step 1: Parse API date as Eastern Time date
            # Format: "02/05/2026" → Feb 5, 2026 at midnight ET
            # Use strptime since pendulum doesn't parse MM/DD/YYYY format
            dt_naive = dt_module.datetime.strptime(api_date_str, '%m/%d/%Y')
            # IMPORTANT: Must create pendulum.DateTime with explicit ET timezone
            # pendulum.instance(dt_naive) defaults to UTC, which is wrong!
            game_date_et = pendulum.datetime(dt_naive.year, dt_naive.month, dt_naive.day,
                                           tz='America/New_York').start_of('day')

            # Step 2: Extract time portion from placeholder
            # Placeholder format: "1900-01-01T19:00:00Z"
            placeholder_dt = pendulum.parse(game_time_utc_placeholder)
            hour = placeholder_dt.hour
            minute = placeholder_dt.minute
            second = placeholder_dt.second

            # Step 3: Combine ET date with ET time
            game_time_et = game_date_et.set(hour=hour, minute=minute, second=second, microsecond=0)

            # Step 4: Convert ET to UTC
            game_time_utc = game_time_et.in_timezone('UTC')

            return game_time_utc

        except Exception as e:
            logger.error(f"Failed to parse NBA schedule time (api_date={api_date_str}, time={game_time_utc_placeholder}): {e}")
            return None

    @classmethod
    def _season_start_year_for_season(cls, season: str) -> int:
        """
        SEASON is like '2025-26' -> return 2025 for data.nba.com URL path.
        """
        try:
            return int(season.split('-')[0])
        except Exception:
            # fallback: current year
            return int(pendulum.now('UTC').format('YYYY'))

    @classmethod
    def _get_full_schedule_index(cls) -> Dict[str, pendulum.DateTime]:
        """
        Build (or reuse) an index: game_id -> start_time_utc from data.nba.com full schedule.
        This is our long-term-stable source of truth for game start times.
        """
        season_key = f"full_schedule:{SEASON}"
        now_ts = time.time()
        cached = cls._full_schedule_index.get(season_key)
        if cached:
            idx, ts = cached
            if now_ts - ts < cls.FULL_SCHEDULE_TTL_SECONDS and idx:
                return idx

        year = cls._season_start_year_for_season(SEASON)
        url = FULL_SCHEDULE_URL_TMPL.format(year=year)
        try:
            resp = requests.get(url, headers=NBA_HEADERS, timeout=NBA_API_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch full schedule from data.nba.com (url={url}): {e}")
            # Keep old cache if present
            if cached:
                return cached[0]
            return {}

        # The full schedule file structure varies slightly by era; handle common patterns.
        # Common: data['lscd'][...]['mscd']['g'] list of games
        idx: Dict[str, pendulum.DateTime] = {}
        try:
            lscd = data.get('lscd', []) or []
            for month in lscd:
                mscd = month.get('mscd', {}) if isinstance(month, dict) else {}
                games = mscd.get('g', []) or []
                for g in games:
                    gid = g.get('gid') or g.get('gameId') or g.get('game_id')
                    if not gid:
                        continue

                    # Prefer a UTC-ish canonical datetime if present:
                    # Known fields in the wild include:
                    # - 'utctm' + 'utcdate' (or 'gdtutc'/'utcdt')
                    # - 'gdte' + 'tm' with ET date/time (less ideal)
                    #
                    # We attempt multiple strategies.

                    # Strategy A: If an ISO UTC field exists, use it
                    iso_candidates = [
                        g.get('gameDateTimeUTC'),
                        g.get('gdtutc'),
                        g.get('utcDateTime'),
                        g.get('utcdt'),
                        g.get('startTimeUTC'),
                    ]
                    iso_candidates = [c for c in iso_candidates if isinstance(c, str) and c.strip()]
                    dt_utc: Optional[pendulum.DateTime] = None
                    for s in iso_candidates:
                        try:
                            dt_utc = parse_iso_utc(s)
                            break
                        except Exception:
                            continue

                    # Strategy B: Separate UTC date + UTC time fields
                    if dt_utc is None:
                        utc_date = g.get('utcdate') or g.get('gdtutc') or g.get('utcd') or g.get('dateUTC')
                        utc_time = g.get('utctm') or g.get('timeUTC')
                        if isinstance(utc_date, str) and isinstance(utc_time, str) and utc_date and utc_time:
                            # Many files use 'YYYYMMDD' and 'HHMM' or 'HH:MM'
                            try:
                                if '-' in utc_date:
                                    # YYYY-MM-DD
                                    d = pendulum.parse(utc_date).in_timezone('UTC').start_of('day')
                                else:
                                    # YYYYMMDD
                                    d = pendulum.from_format(utc_date, 'YYYYMMDD', tz='UTC').start_of('day')
                                # normalize time
                                if ':' in utc_time:
                                    hh, mm = utc_time.split(':')[:2]
                                else:
                                    hh, mm = utc_time[:2], utc_time[2:4]
                                dt_utc = d.set(hour=int(hh), minute=int(mm), second=0, microsecond=0)
                            except Exception:
                                dt_utc = None

                    # Strategy C: ET date/time fallback (convert to UTC)
                    if dt_utc is None:
                        # Common: 'gdte' is YYYY-MM-DD (ET), 'tm' is '7:30 pm' etc
                        gdte = g.get('gdte') or g.get('gameDate') or g.get('date')
                        tm = g.get('tm') or g.get('time') or g.get('gameTime')
                        if isinstance(gdte, str) and gdte and isinstance(tm, str) and tm:
                            try:
                                # Parse ET date and 12h time
                                d_et = pendulum.parse(gdte).in_timezone('America/New_York').start_of('day')
                                # normalize '7:30 pm' -> hour/min
                                t = tm.strip().lower().replace('p.m.', 'pm').replace('a.m.', 'am')
                                # pendulum can parse times but be defensive:
                                parsed_t = pendulum.parse(t)
                                dt_et = d_et.set(hour=parsed_t.hour, minute=parsed_t.minute, second=0, microsecond=0)
                                dt_utc = dt_et.in_timezone('UTC')
                            except Exception:
                                dt_utc = None

                    if dt_utc is not None:
                        idx[str(gid)] = dt_utc
        except Exception as e:
            logger.error(f"Failed parsing full schedule structure: {e}")

        cls._full_schedule_index[season_key] = (idx, now_ts)
        logger.info(f"Built full schedule index: {len(idx)} games")
        return idx

    @classmethod
    def _is_placeholder_schedule_time(cls, game_time_utc_placeholder: str) -> bool:
        """
        Detect known-bad placeholder times from scheduleLeagueV2.
        If it's missing, '1900-01-01', or the time is 00:00:00Z for many games, treat as unreliable.
        """
        if not isinstance(game_time_utc_placeholder, str):
            return True
        s = game_time_utc_placeholder.strip()
        if not s:
            return True
        if s.startswith("1900-01-01T00:00:00"):
            return True
        # Many bad slates show 00:00:00Z, 05:00:00Z, etc. which can be midnight-ish placeholders
        # We treat the '1900-01-01' carrier as unreliable regardless.
        if s.startswith("1900-01-01"):
            return True
        return False

    @classmethod
    def fetch_games_for_date(cls, date: str) -> List[Dict[str, Any]]:
        """
        Fetch all games for a specific date (YYYY-MM-DD).
        Uses scheduleLeagueV2.json which has all scheduled games
        (including future games that haven't started yet).

        IMPORTANT: This fetches games based on API's date semantics,
        which are based on Eastern Time. The returned games will have
        correct start_time_utc values derived from ET date/time.

        Returns list of game dicts with game_id, start_time, teams, etc.
        """
        # DEBUG: Log when fetch_games_for_date is called
        logger.warning(f"[FETCH_GAMES_FOR_DATE] called with date={date}")
        
        # Check cache first
        cached = cls._get_cached('schedule', date)
        if cached:
            logger.debug(f"Using cached schedule for date {date}")
            return cached
        
        try:
            # Fetch schedule
            response = requests.get(SCHEDULE_URL, headers=NBA_HEADERS, timeout=NBA_API_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            league = data.get('leagueSchedule', {})
            game_dates = league.get('gameDates', [])
            
            # Validate NBA schedule response
            validation_result = validate_nba_schedule(data, date)
            if not validation_result['valid']:
                logger.error(f"NBA schedule validation failed for {date}: {validation_result['issues']}")
                return []
            
            # Log warnings
            for warning in validation_result.get('warnings', []):
                logger.warning(warning)
            
            # Find games for specified date
            # scheduleLeagueV2 uses "MM/DD/YYYY" format for gameDate
            # date parameter is YYYY-MM-DD, so we need to match
            # IMPORTANT: Do NOT strip leading zeros - API uses zero-padded format (MM/DD)
            # This was bug: lstrip('0') was causing "02/02/2026" to become "2/2/2026"
            # which doesn't match API format "02/02/2026 00:00:00"
            target_month = date[5:7]  # Keep zero-padding (e.g., "02" not "2")
            target_day = date[8:10]    # Keep zero-padding (e.g., "02" not "2")
            target_year = date[:4]
            
            games_list = None
            api_date_str = None
            for gd in game_dates:
                gd_str = gd.get('gameDate', '')
                if not gd_str:
                    continue

                # Check if this matches our target date
                # Format: "02/02/2026 T00:00:00"
                if f'{target_month}/{target_day}/{target_year}' in gd_str:
                    games_list = gd.get('games', [])
                    api_date_str = gd_str.split()[0]  # Extract just the date part (MM/DD/YYYY)
                    break
            
            if not games_list:
                logger.info(f'No games found for date {date}')
                return []

            # Build/refresh full schedule index (authoritative start times)
            full_idx = cls._get_full_schedule_index()

            games = []
            for g in games_list:
                game_id = g.get('gameId')
                if not game_id:
                    continue

                # Get start time using corrected parsing logic
                time_str_utc = g.get('gameTimeUTC', '')

                # 1) Prefer authoritative per-game start_time_utc from full schedule, if available
                game_time_utc = full_idx.get(str(game_id))

                # 2) If not available, fall back to scheduleLeagueV2 parsing, but only if not placeholder
                if game_time_utc is None:
                    if time_str_utc and not cls._is_placeholder_schedule_time(time_str_utc):
                        game_time_utc = cls._parse_nba_schedule_time(api_date_str, time_str_utc)
                    else:
                        # Placeholder / missing -> we DO NOT guess wildly; skip this game for now.
                        # This prevents "Feb 5 games showing as Feb 4" due to midnight placeholders.
                        logger.warning(
                            f"Game {game_id}: scheduleLeagueV2 time missing/placeholder ({time_str_utc}); "
                            f"no authoritative full-schedule time found; skipping until real time is available."
                        )
                        continue

                if game_time_utc:
                    et_time = game_time_utc.in_timezone('America/New_York')
                    logger.debug(f"Game {game_id}: ET {et_time.format('YYYY-MM-DD HH:mm Z')} → UTC {game_time_utc.to_iso8601_string()}")

                if not game_time_utc:
                    logger.warning(f"Could not parse game time for {game_id}")
                    continue

                # IMPORTANT: game_date must be derived from start_time_utc converted to CST
                game_date_cst = cst_game_date_from_start_time_utc(game_time_utc, tz=CST)
                
                # Get team names
                home_team_obj = g.get('homeTeam', {})
                away_team_obj = g.get('awayTeam', {})
                home_team = home_team_obj.get('teamTricode', 'UNK')
                away_team = away_team_obj.get('teamTricode', 'UNK')
                
                games.append({
                    'game_id': game_id,
                    'game_date': game_date_cst,
                    'start_time_utc': game_time_utc,
                    'home_team': home_team,
                    'away_team': away_team,
                    'status': 'Scheduled',
                    'current_period': None,
                    'game_clock': None,
                    'score_home': 0,
                    'score_away': 0
                })
            
            # Validate game times are in the future
            valid_games = []
            for game in games:
                try:
                    validate_future_datetime(game['start_time_utc'], hours_ahead=48)
                    valid_games.append(game)
                except ValueError as e:
                    logger.warning(f"Skipping game {game['game_id']}: {e}")
            
            # Cache the result
            cls._set_cached('schedule', date, valid_games)
            
            logger.info(f"Fetched {len(games)} games for date {date}, {len(valid_games)} valid after time validation")
            return valid_games
            
        except Exception as e:
            logger.error(f"Error fetching games for date {date}: {e}")
            return []
    
    @classmethod
    def fetch_game_state(cls, game_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch current game state (score, period, clock, status).
        
        Uses CDN-based boxscore API (reliable, no timeouts).
        Includes caching and retry logic with fallback to schedule API.
        
        Returns:
            Dict with game_id, status, period, clock, scores, teams, last_updated
            None if game not found
        """
        # Check cache first
        cached = cls._get_cached('game_state', game_id)
        if cached:
            logger.debug(f"Using cached game state for {game_id}")
            return cached
        
        # Try CDN boxscore API with retry logic
        for attempt in range(3):
            try:
                url = BOXSCORE_URL.format(gid=game_id)
                response = requests.get(url, headers=NBA_HEADERS, timeout=NBA_API_TIMEOUT)
                response.raise_for_status()
                data = response.json()
                
                game = data.get('game')
                if not game:
                    logger.warning(f"No game data found in response for {game_id}")
                    return None
                
                # Extract game state
                home_team_obj = game.get('homeTeam', {})
                away_team_obj = game.get('awayTeam', {})
                
                # Parse period
                period = game.get('period', 0)
                try:
                    period = int(period) if period else 0
                except (ValueError, TypeError):
                    period = 0
                
                # Parse game clock
                game_clock = game.get('gameClock', 'PT00M00.00S')
                
                # Get status
                game_status = game.get('gameStatusText', 'Unknown')
                
                # Get scores
                home_score = home_team_obj.get('score', 0)
                away_score = away_team_obj.get('score', 0)
                
                result = {
                    'game_id': game_id,
                    'status': game_status,
                    'current_period': period,
                    'game_clock': game_clock,
                    'score_home': home_score,
                    'score_away': away_score,
                    'home_team': home_team_obj.get('teamTricode', ''),
                    'away_team': away_team_obj.get('teamTricode', ''),
                    'home_name': home_team_obj.get('teamName', ''),
                    'away_name': away_team_obj.get('teamName', ''),
                    'last_updated': now_utc()
                }
                
                # Cache the result
                cls._set_cached('game_state', game_id, result)
                
                logger.info(f"Fetched game state for {game_id}: {game_status} Q{period}")
                return result
                
            except requests.HTTPError as e:
                # Check for 403/429 errors (rate limiting or game not started)
                if e.response.status_code in (403, 429):
                    if attempt < 2:  # Retry once before falling back
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"Boxscore API returned {e.response.status_code} for {game_id}, retrying in {wait_time}s (attempt {attempt+1}/3)")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Fallback to schedule API
                        logger.warning(f"Boxscore API unavailable for {game_id} (status {e.response.status_code}), falling back to schedule API")
                        return cls._get_game_state_from_schedule(game_id)
                else:
                    # Other HTTP errors - re-raise
                    logger.error(f"HTTP error fetching game state for {game_id}: {e}")
                    raise
                    
            except requests.Timeout as e:
                if attempt < 2:
                    wait_time = 2 ** attempt
                    logger.warning(f"Timeout fetching game state for {game_id}, retrying in {wait_time}s (attempt {attempt+1}/3)")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"All retries timed out for {game_id}: {e}")
                    return None
            
            except Exception as e:
                logger.error(f"Unexpected error fetching game state for {game_id}: {e}")
                return None
        
        return None
    
    @classmethod
    def _get_game_state_from_schedule(cls, game_id: str) -> Optional[Dict[str, Any]]:
        """
        Get basic game info from schedule API (fallback when boxscore unavailable).
        
        This is used for games that haven't started yet or don't have
        boxscore data available.
        
        Args:
            game_id: NBA game ID
            
        Returns:
            Minimal game dict with game_id, teams, status, period, clock, scores (all zero)
        """
        try:
            # Fetch schedule (can be cached)
            response = requests.get(SCHEDULE_URL, headers=NBA_HEADERS, timeout=NBA_API_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            league = data.get('leagueSchedule', {})
            game_dates = league.get('gameDates', [])
            
            # Search for the game in schedule
            for gd in game_dates:
                games = gd.get('games', [])
                for game in games:
                    if game.get('gameId') == game_id:
                        # Found the game - return minimal game state
                        home_team_obj = game.get('homeTeam', {})
                        away_team_obj = game.get('awayTeam', {})
                        
                        result = {
                            'game_id': game_id,
                            'status': 'Scheduled',
                            'current_period': 0,
                            'game_clock': 'PT00M00.00S',
                            'score_home': 0,
                            'score_away': 0,
                            'home_team': home_team_obj.get('teamTricode', ''),
                            'away_team': away_team_obj.get('teamTricode', ''),
                            'home_name': home_team_obj.get('teamName', ''),
                            'away_name': away_team_obj.get('teamName', ''),
                            'last_updated': now_utc()
                        }
                        
                        # Cache the result
                        cls._set_cached('game_state', game_id, result)
                        
                        return result
            
            # Game not found in schedule
            logger.warning(f"Game {game_id} not found in schedule API")
            return None
            
        except Exception as e:
            logger.error(f"Error getting game {game_id} from schedule: {e}")
            return None
    
    @classmethod
    def fetch_game_stats(cls, game_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch detailed game statistics for a game.
        
        Returns full boxscore data including team stats, player stats, etc.
        This is used by halftime and Q3 prediction models.
        
        Args:
            game_id: NBA game ID
            
        Returns:
            Full game data dict (includes all statistics)
        """
        try:
            url = BOXSCORE_URL.format(gid=game_id)
            response = requests.get(url, headers=NBA_HEADERS, timeout=NBA_API_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            game = data.get('game')
            if not game:
                logger.warning(f"No game data found for {game_id}")
                return None
            
            logger.info(f"Fetched detailed stats for {game_id}")
            return game
            
        except requests.HTTPError as e:
            if e.response.status_code in (403, 429):
                # Fall back to schedule for basic info
                logger.warning(f"Boxscore unavailable for {game_id}, falling back to schedule")
                return cls._get_game_state_from_schedule(game_id)
            else:
                logger.error(f"HTTP error fetching stats for {game_id}: {e}")
                return None
        except Exception as e:
            logger.error(f"Error fetching game stats for {game_id}: {e}")
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
            
            # Get game details from DB to find teams
            from core.storage import GameStorage
            game = GameStorage.get_game(game_id, db_path=db_path)
            
            if not game:
                logger.error(f"Game {game_id} not found in database")
                return None
            
            # NOTE: Odds API integration coming soon. Using cached odds for now.
            logger.info("Using cached odds for game")
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
    
    def fetch_games_for_cst_date(self, cst_date: str) -> List[Dict[str, Any]]:
        """
        NEW (correct): fetch schedule data covering CST date window,
        then bucket locally by CST day and filter to requested CST date.
        This avoids API "date semantics" bugs (ET/UTC/league-day).
        """
        # NOTE: This method assumes NBADataSource.fetch_games_for_date now:
        # - resolves start_time_utc from data.nba.com full schedule whenever possible
        # - otherwise skips games whose times are placeholders
        # That is intentional: correctness > guessing.

        # CST window: [cst_date 00:00, next day 00:00)
        cst_start = pendulum.parse(cst_date).in_timezone(CST).start_of("day")
        cst_end = cst_start.add(days=1)
        utc_start = cst_start.in_timezone("UTC")
        utc_end = cst_end.in_timezone("UTC")
        
        # We don't trust the API date semantics, so we fetch a small surrounding set
        # and then filter by utc_start/utc_end.
        # Many NBA schedule endpoints are keyed by "calendar date" in ET or UTC;
        # fetching utc_start.date() and utc_end.date() plus one buffer day is safest.
        fetch_dates = sorted({
            utc_start.to_date_string(),
            utc_end.to_date_string(),
            utc_start.subtract(days=1).to_date_string(),
        })
        
        raw_games: List[Dict[str, Any]] = []
        for d in fetch_dates:
            try:
                raw_games.extend(self.nba.fetch_games_for_date(d))
            except Exception as e:
                logger.warning(f"NBA schedule fetch failed for {d}: {e}")
        
        # De-dupe by game_id
        by_id = {}
        for g in raw_games:
            gid = g.get("game_id") or g.get("id") or g.get("gameId")
            if gid:
                by_id[gid] = g
        
        normalized: List[Dict[str, Any]] = []
        for g in by_id.values():
            # Normalize start_time_utc to ISO string if needed
            st = g.get("start_time_utc") or g.get("game_time_utc") or g.get("startTimeUTC")
            if not st:
                continue
            try:
                dt_utc = parse_iso_utc(st) if isinstance(st, str) else st
            except Exception:
                continue
            
            # Filter to UTC window (this ensures correct slate for CST day)
            if dt_utc < utc_start or dt_utc >= utc_end:
                continue
            
            # Compute authoritative CST game date
            game_date_cst = cst_game_date_from_start_time_utc(dt_utc, tz=CST)
            
            # Build normalized record (keep existing fields you rely on)
            normalized.append({
                "game_id": g.get("game_id") or g.get("id") or g.get("gameId"),
                "start_time_utc": dt_utc.to_iso8601_string() if hasattr(dt_utc, "to_iso8601_string") else str(st),
                "home_team": g.get("home_team") or g.get("homeTeam") or g.get("teamTricode"),
                "away_team": g.get("away_team") or g.get("awayTeam"),
                "status": g.get("status") or "Scheduled",
                "game_date": game_date_cst,
            })
        
        # Final filter strictly to requested CST date
        final = [g for g in normalized if g.get("game_date") == cst_date]
        logger.info(f"Fetched {len(final)} games for CST date {cst_date} (UTC window {utc_start}..{utc_end})")
        return final
    
    def refresh_game_data(
        self,
        game_id: str,
        reason: str,
        db_path: Path,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Refresh both NBA game state and odds for a game.
        
        Returns dict with:
        - game_state: NBA game state
        - odds: Cached or fresh odds
        
        Args:
            force_refresh: If True, always fetch from API. If False, use cached data.
        """
        # Check database FIRST before fetching from API
        from core.storage import GameStorage
        existing_game = GameStorage.get_game(game_id, db_path=db_path)
        
        if existing_game and not force_refresh:
            # Game exists in database - use it instead of fetching from API
            logger.info(f"Using cached game state for {game_id} from database")
            
            # Convert database row to game_state format
            game_state = {
                'game_id': existing_game['game_id'],
                'start_time_utc': existing_game['start_time_utc'],
                'home_team': existing_game['home_team'],
                'away_team': existing_game['away_team'],
                'home_score': existing_game.get('score_home', 0),
                'away_score': existing_game.get('score_away', 0),
                'current_period': existing_game.get('current_period', 0),
                'game_clock': existing_game.get('game_clock', ''),
                'status': existing_game.get('status', 'Scheduled')
            }
        else:
            # Game doesn't exist or force refresh - fetch from API
            logger.info(f"Fetching game {game_id} from NBA API")
            game_state = self.nba.fetch_game_state(game_id)
        
        odds = self.odds.get_odds(game_id, reason, db_path=db_path)
        
        return {
            'game_state': game_state,
            'odds': odds
        }

    @classmethod
    def fetch_games_for_league_day(cls, league_day: str) -> List[Dict[str, Any]]:
        """
        Fetch games for a specific NBA league day (ET-based).
        
        league_day is the canonical NBA slate key that matches the scheduleLeagueV2
        gameDate format (MM/DD/YYYY in Eastern Time). This is the authoritative
        source for game scheduling and DAILY_SUMMARY triggers.
        
        Args:
            league_day: Date in YYYY-MM-DD format
        
        Returns:
            List of game dicts with game_id, start_time_utc, home_team, away_team,
            status, league_day, local_day_cst
        """
        try:
            # Convert YYYY-MM-DD to MM/DD/YYYY (zero-padded, no leading zero strip)
            year, month, day = league_day.split('-')
            api_date_str = f"{month}/{day}/{year}"
            
            # Fetch scheduleLeagueV2.json
            response = requests.get(SCHEDULE_URL, headers=NBA_HEADERS, timeout=NBA_API_TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract games for specified date
            schedule_data = data.get('league', {}).get('standard', {}).get('schedule', [])
            
            games = []
            for day_schedule in schedule_data:
                schedule_date_str = day_schedule.get('gameDate')  # MM/DD/YYYY in ET
                if schedule_date_str == api_date_str:
                    game_data = day_schedule.get('games', [])
                    for g in game_data:
                        game_id = g.get('gameId')
                        
                        # Parse ET time with _parse_nba_schedule_time
                        start_time_utc = cls._parse_nba_schedule_time(
                            g.get('gameTimeUTC'), api_date_str, g.get('gameTimeET', "19:00")
                        )
                        
                        # Mild sanity check: start_time_utc should be within 18h-400d
                        now = pendulum.now('UTC')
                        time_until_game = (start_time_utc - now).total_seconds()
                        
                        if time_until_game < -18*3600:  # More than 18 hours ago
                            continue
                        if time_until_game > 400*24*3600:  # More than 400 days ahead
                            continue
                        
                        # Compute local_day_cst from start_time_utc
                        local_day_cst = cst_game_date_from_start_time_utc(start_time_utc, tz=CST)
                        
                        games.append({
                            'game_id': game_id,
                            'start_time_utc': start_time_utc,
                            'home_team': g.get('homeTeam', {}).get('teamTricode', ''),
                            'away_team': g.get('awayTeam', {}).get('teamTricode', ''),
                            'status': 'Scheduled',
                            'league_day': league_day,  # ET-based canonical
                            'local_day_cst': local_day_cst  # CST-derived for display
                        })
                    
                    break  # Found the date
            
            logger.info(f"Fetched {len(games)} games for league_day {league_day}")
            return games
            
        except Exception as e:
            logger.error(f"Error fetching games for league_day {league_day}: {e}")
            return []
