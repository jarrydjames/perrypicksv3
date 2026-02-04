'''
Validation functions for PerryPicks v4 Automation System.

Provides date validation, schedule validation, and system clock
validation to prevent time/date issues.

Usage:
    from core.validation import (
        validate_future_datetime,
        validate_schedule_date,
        validate_system_clock,
        validate_nba_schedule
    )
    
    # Validate a future datetime
    validate_future_datetime(game_start_time)
    
    # Validate a schedule date
    dt = validate_schedule_date('2025-02-03')
    
    # Check system clock
    validate_system_clock()
'''

import logging
import requests
from typing import Dict, Any, Optional, List

import pendulum

from core.timezone import now_utc, to_iso

logger = logging.getLogger(__name__)

# Constants
DEFAULT_MAX_HOURS_AHEAD = 48  # Maximum hours ahead to accept as valid
DEFAULT_MIN_HOURS_AHEAD = 0.1  # Minimum minutes (6 minutes) to be considered "future"
DEFAULT_MAX_DAYS_AHEAD = 7  # Maximum days ahead for schedule dates

# External time API for system clock validation
WORLDTIME_API = 'http://worldtimeapi.org/api/timezone/UTC'


def validate_future_datetime(
    dt_utc: pendulum.DateTime,
    hours_ahead: Optional[int] = None,
    minutes_ahead: Optional[int] = None
) -> bool:
    '''
    Validate that a datetime is in the future and within reasonable bounds.
    
    This prevents scheduling games with invalid times (e.g., in the past
    or too far in the future).
    
    Args:
        dt_utc: UTC datetime to validate
        hours_ahead: Maximum hours ahead to consider valid (default: 48)
        minutes_ahead: Minimum minutes ahead to be considered "future" (default: 6)
    
    Returns:
        True if datetime is valid (in future and within bounds)
    
    Raises:
        ValueError: If datetime is in past or too far in future
    
    Example:
        >>> now = now_utc()
        >>> future = now + hours(5)
        >>> validate_future_datetime(future)
        True
        
        >>> past = now - hours(1)
        >>> validate_future_datetime(past)
        ValueError: Datetime is in the past!
    '''
    if hours_ahead is None:
        hours_ahead = DEFAULT_MAX_HOURS_AHEAD
    if minutes_ahead is None:
        minutes_ahead = DEFAULT_MIN_HOURS_AHEAD * 60  # Convert hours to minutes
    
    now = now_utc()
    delta = dt_utc - now
    seconds_delta = delta.total_seconds()
    minutes_delta = seconds_delta / 60
    hours_delta = seconds_delta / 3600
    
    # Check if datetime is in past
    if seconds_delta < 0:
        error_msg = (
            f"Datetime {to_iso(dt_utc)} is in the past! "
            f"({abs(hours_delta):.1f} hours ago). "
            f"Current time: {to_iso(now)}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Check if datetime is too close (within 6 minutes)
    if minutes_delta < minutes_ahead:
        logger.warning(
            f"Datetime {to_iso(dt_utc)} is very close ({minutes_delta:.1f} minutes away). "
            f"This might cause issues."
        )
        # Don't raise, just warn
    
    # Check if datetime is too far in future
    if hours_delta > hours_ahead:
        error_msg = (
            f"Datetime {to_iso(dt_utc)} is too far in future! "
            f"({hours_delta:.1f} hours ahead, max {hours_ahead} hours). "
            f"Current time: {to_iso(now)}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.debug(
        f"Datetime {to_iso(dt_utc)} validated: "
        f"{hours_delta:.1f} hours ahead"
    )
    return True


def validate_schedule_date(
    date_str: str,
    days_ahead: Optional[int] = None
) -> pendulum.DateTime:
    '''
    Validate schedule date is reasonable and parse it.
    
    This ensures dates are in YYYY-MM-DD format, are not in the past,
    and are not too far in the future.
    
    Args:
        date_str: Date string in YYYY-MM-DD format
        days_ahead: Maximum days ahead to allow (default: 7)
    
    Returns:
        Parsed datetime (midnight UTC)
    
    Raises:
        ValueError: If date format is invalid or date is out of range
    
    Example:
        >>> # Tomorrow
        >>> tomorrow = (now_utc() + days(1)).format('YYYY-MM-DD')
        >>> dt = validate_schedule_date(tomorrow)
        
        >>> # Invalid format
        >>> validate_schedule_date('02/03/2025')
        ValueError: Invalid date format '02/03/2025'. Use YYYY-MM-DD.
        
        >>> # Past date
        >>> yesterday = (now_utc() - days(1)).format('YYYY-MM-DD')
        >>> validate_schedule_date(yesterday)
        ValueError: Date is in the past!
    '''
    if days_ahead is None:
        days_ahead = DEFAULT_MAX_DAYS_AHEAD
    
    # Validate format
    try:
        dt = pendulum.parse(date_str, strict=True)
    except Exception as e:
        error_msg = f"Invalid date format '{date_str}'. Use YYYY-MM-DD."
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    
    # Ensure it's at midnight UTC
    dt = dt.start_of('day').in_timezone('UTC')
    
    now = now_utc()
    now_midnight = now.start_of('day').in_timezone('UTC')
    
    # Check if date is in past
    if dt < now_midnight:
        error_msg = (
            f"Date '{date_str}' is in the past! "
            f"Current date: {now.format('YYYY-MM-DD')}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Check if date is too far in future
    max_date = now_midnight + pendulum.duration(days=days_ahead)
    if dt > max_date:
        error_msg = (
            f"Date '{date_str}' is too far in the future! "
            f"(>{days_ahead} days from now). "
            f"Current date: {now.format('YYYY-MM-DD')}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Validated schedule date: {date_str}")
    return dt


def validate_system_clock(max_seconds_drift: int = 300) -> Dict[str, Any]:
    '''
    Check if system clock is reasonable by comparing to external time source.
    
    Args:
        max_seconds_drift: Maximum allowed seconds drift (default: 5 minutes)
    
    Returns:
        Dict with validation results:
        {
            'valid': bool,
            'local_time': pendulum.DateTime,
            'external_time': pendulum.DateTime,
            'drift_seconds': float,
            'warning': Optional[str]
        }
    
    Example:
        >>> result = validate_system_clock()
        >>> result['valid']
        True
        >>> result['drift_seconds']
        2.5
        
        >>> # If clock is off
        >>> result = validate_system_clock()
        >>> result['valid']
        False
        >>> result['warning']
        'System clock is off by 350 seconds!'
    '''
    result = {
        'valid': True,
        'local_time': now_utc(),
        'external_time': None,
        'drift_seconds': 0.0,
        'warning': None
    }
    
    try:
        # Fetch external time
        response = requests.get(WORLDTIME_API, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        external_time_str = data.get('datetime')
        external_time = pendulum.parse(external_time_str)
        external_time = external_time.in_timezone('UTC')
        
        result['external_time'] = external_time
        
        # Calculate drift
        drift = external_time - result['local_time']
        result['drift_seconds'] = abs(drift.total_seconds())
        
        # Check if drift is too large
        if result['drift_seconds'] > max_seconds_drift:
            result['valid'] = False
            warning = (
                f"System clock is off by {result['drift_seconds']:.0f} seconds! "
                f"Expected: {to_iso(external_time)}, "
                f"Got: {to_iso(result['local_time'])}"
            )
            logger.warning(warning)
            result['warning'] = warning
        else:
            logger.debug(
                f"System clock validated: {result['drift_seconds']:.1f}s drift"
            )
    
    except requests.RequestException as e:
        warning = f"Could not validate system clock (network error): {e}"
        logger.warning(warning)
        result['warning'] = warning
        # Don't fail, just warn
    except Exception as e:
        warning = f"Could not validate system clock: {e}"
        logger.warning(warning)
        result['warning'] = warning
        # Don't fail, just warn
    
    return result


def validate_nba_schedule(
    schedule_data: Dict[str, Any],
    requested_date: str
) -> Dict[str, Any]:
    '''
    Validate that NBA API response contains reasonable data.
    
    This checks for missing data, unknown teams, and validates
    that the response makes sense.
    
    Args:
        schedule_data: Full response from NBA scheduleLeagueV2 API
        requested_date: Date that was requested (YYYY-MM-DD)
    
    Returns:
        Dict with validation results:
        {
            'valid': bool,
            'total_games': int,
            'issues': List[str],
            'warnings': List[str]
        }
    
    Example:
        >>> response = requests.get(SCHEDULE_URL).json()
        >>> result = validate_nba_schedule(response, '2025-02-03')
        >>> result['valid']
        True
        >>> result['total_games']
        10
    '''
    result = {
        'valid': True,
        'total_games': 0,
        'issues': [],
        'warnings': []
    }
    
    # Parse schedule data
    league = schedule_data.get('leagueSchedule', {})
    game_dates = league.get('gameDates', [])
    
    if not game_dates:
        issue = "NBA schedule returned no game dates!"
        logger.error(issue)
        result['issues'].append(issue)
        result['valid'] = False
        return result
    
    # Count total games
    total_games = sum(len(gd.get('games', [])) for gd in game_dates)
    result['total_games'] = total_games
    
    if total_games == 0:
        warning = f"No games found for date {requested_date}"
        logger.warning(warning)
        result['warnings'].append(warning)
        return result  # Not an error, might just be no games
    
    # Validate each game
    for gd in game_dates:
        for game in gd.get('games', []):
            game_id = game.get('gameId')
            home_team = game.get('homeTeam', {})
            away_team = game.get('awayTeam', {})
            home_tricode = home_team.get('triCode', 'UNK')
            away_tricode = away_team.get('triCode', 'UNK')
            
            # Check for missing game ID
            if not game_id:
                issue = "Game missing game_id!"
                logger.error(issue)
                result['issues'].append(issue)
                result['valid'] = False
                continue
            
            # Check for unknown teams
            if home_tricode == 'UNK' or away_tricode == 'UNK':
                warning = (
                    f"Game {game_id} has unknown teams ({away_tricode} @ {home_tricode}) - "
                    f"schedule may not be finalized yet"
                )
                logger.warning(warning)
                result['warnings'].append(warning)
    
    # Log summary
    if result['valid']:
        logger.info(
            f"NBA schedule validated for {requested_date}: "
            f"{total_games} games, "
            f"{len(result['warnings'])} warnings, "
            f"{len(result['issues'])} issues"
        )
    else:
        logger.error(
            f"NBA schedule validation FAILED for {requested_date}: "
            f"{len(result['issues'])} issues"
        )
    
    return result


def validate_game_time_range(
    games: List[Dict[str, Any]],
    max_hours_span: int = 24
) -> bool:
    '''
    Validate that game times are within a reasonable range.
    
    This catches cases where all games have the same time or times
    are spread over an unrealistically long period.
    
    Args:
        games: List of game dicts with 'start_time_utc' (pendulum.DateTime)
        max_hours_span: Maximum hours span to consider valid (default: 24)
    
    Returns:
        True if times are valid
    
    Raises:
        ValueError: If times are invalid
    
    Example:
        >>> games = [
        ...     {'game_id': '001', 'start_time_utc': now_utc() + hours(7)},
        ...     {'game_id': '002', 'start_time_utc': now_utc() + hours(10)},
        ... ]
        >>> validate_game_time_range(games)
        True
    '''
    if len(games) < 2:
        return True  # Not enough games to check
    
    # Extract start times
    start_times = [g['start_time_utc'] for g in games if 'start_time_utc' in g]
    
    if len(start_times) < 2:
        return True  # Not enough valid times
    
    # Calculate time span
    min_time = min(start_times)
    max_time = max(start_times)
    time_span = (max_time - min_time).total_seconds() / 3600
    
    if time_span > max_hours_span:
        error_msg = (
            f"Game times span too long: {time_span:.1f} hours "
            f"(max {max_hours_span} hours). "
            f"First game: {to_iso(min_time)}, "
            f"Last game: {to_iso(max_time)}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Check if all games have the same time
    if time_span < 0.1:  # Less than 6 minutes between all games
        warning = (
            f"All games have nearly identical start times: "
            f"{to_iso(min_time)} to {to_iso(max_time)}"
        )
        logger.warning(warning)
        # Don't raise, just warn
    
    logger.debug(f"Game time range validated: {time_span:.1f} hours span")
    return True


# Export all public functions
__all__ = [
    'validate_future_datetime',
    'validate_schedule_date',
    'validate_system_clock',
    'validate_nba_schedule',
    'validate_game_time_range',
]