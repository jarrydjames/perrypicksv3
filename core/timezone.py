'''
Timezone utilities for PerryPicks v4 Automation System.

Provides centralized timezone handling using Pendulum for reliable
datetime operations. All internal operations use UTC, with
conversion to local timezone only for display.

Usage:
    from core.timezone import now_utc, to_local, format_local
    
    # Get current UTC time
    utc_time = now_utc()
    
    # Convert to local timezone for display
    local_time = to_local(utc_time, tz='America/Chicago')
    
    # Format as string
    formatted = format_local(utc_time, tz='America/Chicago')
'''

import logging
from typing import Optional

import pendulum

logger = logging.getLogger(__name__)

# Central timezone definitions
UTC = 'UTC'
CST = 'America/Chicago'
EST = 'America/New_York'
DEFAULT_DISPLAY_TZ = CST  # Use CST for display by default


def now_utc() -> pendulum.DateTime:
    '''
    Get current time in UTC.
    
    Returns:
        Current UTC time as pendulum.DateTime
    
    Example:
        >>> now = now_utc()
        >>> now.timezone_name
        'UTC'
    '''
    return pendulum.now(UTC)


def to_utc(dt: pendulum.DateTime) -> pendulum.DateTime:
    '''
    Convert any datetime to UTC.
    
    Args:
        dt: pendulum.DateTime in any timezone
    
    Returns:
        pendulum.DateTime in UTC
    
    Example:
        >>> local = pendulum.now('America/Chicago')
        >>> utc = to_utc(local)
        >>> utc.timezone_name
        'UTC'
    '''
    return dt.in_timezone(UTC)


def to_local(dt: pendulum.DateTime, tz: Optional[str] = None) -> pendulum.DateTime:
    '''
    Convert UTC datetime to local timezone (for display).
    
    Args:
        dt: UTC datetime to convert
        tz: Target timezone (defaults to DEFAULT_DISPLAY_TZ)
    
    Returns:
        pendulum.DateTime in local timezone
    
    Example:
        >>> utc = now_utc()
        >>> local = to_local(utc, tz='America/Chicago')
        >>> local.timezone_name
        'America/Chicago'
    '''
    if tz is None:
        tz = DEFAULT_DISPLAY_TZ
    return dt.in_timezone(tz)


def format_local(
    dt: pendulum.DateTime,
    tz: Optional[str] = None,
    fmt: Optional[str] = None
) -> str:
    '''
    Format datetime in local timezone for display.
    
    Args:
        dt: UTC datetime to format
        tz: Target timezone (defaults to DEFAULT_DISPLAY_TZ)
        fmt: Custom format string (defaults to 'YYYY-MM-DD HH:mm:ss Z')
    
    Returns:
        Formatted string in local timezone
    
    Example:
        >>> utc = now_utc()
        >>> format_local(utc, tz='America/Chicago')
        '2025-02-03 14:30:00 -06:00'
    '''
    if tz is None:
        tz = DEFAULT_DISPLAY_TZ
    if fmt is None:
        fmt = 'YYYY-MM-DD HH:mm:ss Z'
    
    local_dt = dt.in_timezone(tz)
    return local_dt.format(fmt)


def parse_iso_utc(iso_str) -> pendulum.DateTime:
    '''
    Parse ISO 8601 string or pendulum.DateTime as UTC datetime.
    
    Args:
        iso_str: ISO 8601 string (e.g., '2025-02-03T14:30:00Z') or pendulum.DateTime
    
    Returns:
        pendulum.DateTime in UTC
    
    Raises:
        ValueError: If string cannot be parsed
    
    Example:
        >>> dt = parse_iso_utc('2025-02-03T14:30:00Z')
        >>> dt.timezone_name
        'UTC'
        >>> dt2 = pendulum.now('UTC')
        >>> parse_iso_utc(dt2).timezone_name
        'UTC'
    '''
    # If already a DateTime, just ensure it's in UTC
    if isinstance(iso_str, pendulum.DateTime):
        return iso_str.in_timezone(UTC)
    
    # Otherwise parse from string
    try:
        dt = pendulum.parse(iso_str, strict=True)
        return dt.in_timezone(UTC)
    except Exception as e:
        logger.error(f"Failed to parse ISO datetime '{iso_str}': {e}")
        raise ValueError(f"Invalid ISO datetime '{iso_str}': {e}") from e


def to_iso(dt: pendulum.DateTime) -> str:
    '''
    Convert datetime to ISO 8601 string (always UTC, using 'Z' suffix).
    
    Args:
        dt: pendulum.DateTime in any timezone
    
    Returns:
        ISO 8601 string with 'Z' suffix (UTC)
    
    Example:
        >>> dt = pendulum.now('UTC')
        >>> to_iso(dt)
        '2025-02-03T14:30:00Z'
    '''
    return dt.in_timezone(UTC).to_iso8601_string()


def parse_date_str(date_str: str) -> pendulum.DateTime:
    '''
    Parse date string in YYYY-MM-DD format as midnight UTC.
    
    Args:
        date_str: Date string in YYYY-MM-DD format
    
    Returns:
        pendulum.DateTime representing midnight UTC on that date
    
    Raises:
        ValueError: If date_str is not in YYYY-MM-DD format
    
    Example:
        >>> dt = parse_date_str('2025-02-03')
        >>> dt.to_date_string()
        '2025-02-03'
        >>> dt.time()
        datetime.time(0, 0)
    '''
    try:
        dt = pendulum.parse(date_str, strict=True)
        # Ensure it's at midnight UTC
        dt = dt.start_of('day').in_timezone(UTC)
        return dt
    except Exception as e:
        logger.error(f"Failed to parse date string '{date_str}': {e}")
        raise ValueError(f"Invalid date format '{date_str}'. Use YYYY-MM-DD.") from e


def validate_timezone(tz: str) -> bool:
    '''
    Validate that a timezone string is valid.
    
    Args:
        tz: Timezone string (e.g., 'America/Chicago')
    
    Returns:
        True if timezone is valid
    
    Example:
        >>> validate_timezone('America/Chicago')
        True
        >>> validate_timezone('Invalid/Timezone')
        False
    '''
    try:
        pendulum.now(tz)
        return True
    except Exception:
        return False


def get_timezone_offset(tz: str = None) -> str:
    '''
    Get current timezone offset for a timezone.
    
    Args:
        tz: Timezone string (defaults to DEFAULT_DISPLAY_TZ)
    
    Returns:
        Offset string (e.g., '-06:00')
    
    Example:
        >>> get_timezone_offset('America/Chicago')
        '-06:00'
    '''
    if tz is None:
        tz = DEFAULT_DISPLAY_TZ
    return pendulum.now(tz).format('ZZ')


def is_dst_in_effect(tz: str = None) -> bool:
    '''
    Check if Daylight Saving Time is in effect for a timezone.
    
    Args:
        tz: Timezone string (defaults to DEFAULT_DISPLAY_TZ)
    
    Returns:
        True if DST is in effect
    
    Example:
        >>> # CST is UTC-6, CDT is UTC-5 (DST)
        >>> is_dst_in_effect('America/Chicago')
        True  # or False depending on time of year
    '''
    if tz is None:
        tz = DEFAULT_DISPLAY_TZ
    return pendulum.now(tz).is_dst()


def hours_until(dt: pendulum.DateTime) -> float:
    '''
    Calculate hours from now until a future datetime.
    
    Args:
        dt: Future datetime (UTC)
    
    Returns:
        Hours until dt (negative if dt is in past)
    
    Example:
        >>> future = now_utc() + hours(3)
        >>> hours_until(future)
        3.0
    '''
    now = now_utc()
    return (dt - now).total_seconds() / 3600


def seconds_until(dt: pendulum.DateTime) -> float:
    '''
    Calculate seconds from now until a future datetime.
    
    Args:
        dt: Future datetime (UTC)
    
    Returns:
        Seconds until dt (negative if dt is in past)
    
    Example:
        >>> future = now_utc() + seconds(300)
        >>> seconds_until(future)
        300.0
    '''
    now = now_utc()
    return (dt - now).total_seconds()


def is_future(dt: pendulum.DateTime) -> bool:
    '''
    Check if a datetime is in future.
    
    Args:
        dt: Datetime to check (UTC)
    
    Returns:
        True if dt is in future
    
    Example:
        >>> future = now_utc() + hours(1)
        >>> is_future(future)
        True
        >>> past = now_utc() - hours(1)
        >>> is_future(past)
        False
    '''
    return dt > now_utc()


def is_past(dt: pendulum.DateTime) -> bool:
    '''
    Check if a datetime is in past.
    
    Args:
        dt: Datetime to check (UTC)
    
    Returns:
        True if dt is in past
    
    Example:
        >>> past = now_utc() - hours(1)
        >>> is_past(past)
        True
        >>> future = now_utc() + hours(1)
        >>> is_future(future)
        False
    '''
    return dt < now_utc()


def cst_game_date_from_start_time_utc(
    start_time_utc,
    tz: str = CST
) -> str:
    '''
    Canonical game-day rule: game_date = calendar day in America/Chicago of UTC start time.
    
    Args:
        start_time_utc: UTC start time (ISO string or pendulum.DateTime)
        tz: Target timezone (defaults to CST)
    
    Returns:
        'YYYY-MM-DD'
    
    Example:
        >>> dt = pendulum.parse("2026-02-04T03:00:00Z")
        >>> cst_game_date_from_start_time_utc(dt, tz=CST)
        '2026-02-03'
    '''
    dt_utc = parse_iso_utc(start_time_utc)
    dt_local = dt_utc.in_timezone(tz)
    return dt_local.format('YYYY-MM-DD')


def today_cst_date_str(tz: str = CST) -> str:
    '''
    Canonical 'today' for user-facing schedules.
    
    Args:
        tz: Target timezone (defaults to CST)
    
    Returns:
        'YYYY-MM-DD'
    
    Example:
        >>> today_cst_date_str()
        '2025-02-03'
    '''
    return now_utc().in_timezone(tz).format('YYYY-MM-DD')

# Convenience imports for backward compatibility
class DateTime(pendulum.DateTime):
    '''
    Wrapper around pendulum.DateTime with convenience methods.
    
    Provides a drop-in replacement for datetime.datetime
    with better timezone handling.
    '''
    
    @classmethod
    def from_iso(cls, iso_str: str) -> 'DateTime':
        '''Create from ISO 8601 string.'''
        return cls(parse_iso_utc(iso_str))
    
    @classmethod
    def from_date_str(cls, date_str: str) -> 'DateTime':
        '''Create from YYYY-MM-DD date string.'''
        return cls(parse_date_str(date_str))
    
    def to_iso(self) -> str:
        '''Convert to ISO 8601 string (UTC).'''
        return to_iso(self)
    
    def to_local_str(self, tz: Optional[str] = None) -> str:
        '''Format as local timezone string.'''
        return format_local(self, tz)
    
    def hours_until(self) -> float:
        '''Hours from now until this datetime.'''
        return hours_until(self)
    
    def seconds_until(self) -> float:
        '''Seconds from now until this datetime.'''
        return seconds_until(self)
    
    def is_future(self) -> bool:
        '''Check if this datetime is in future.'''
        return is_future(self)
    
    def is_past(self) -> bool:
        '''Check if this datetime is in past.'''
        return is_past(self)


# Export all public functions and classes
__all__ = [
    # Timezone constants
    'UTC',
    'CST',
    'EST',
    'DEFAULT_DISPLAY_TZ',
    
    # Utility functions
    'now_utc',
    'to_utc',
    'to_local',
    'format_local',
    'parse_iso_utc',
    'to_iso',
    'parse_date_str',
    'validate_timezone',
    'get_timezone_offset',
    'is_dst_in_effect',
    'hours_until',
    'seconds_until',
    'is_future',
    'is_past',
    'cst_game_date_from_start_time_utc',
    'today_cst_date_str',
    
    # DateTime wrapper
    'DateTime',
]
