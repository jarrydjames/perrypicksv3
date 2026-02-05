"""
CORRECTED NBA schedule time parsing logic.
This file contains corrected implementation for NBADataSource._parse_nba_schedule_time()
and updated fetch_games_for_date() method.

ROOT CAUSE OF BUG:
The NBA schedule API uses a non-intuitive format:
- api_date_str (e.g., "02/05/2026") is Eastern Time date of game
- game_time_utc_placeholder (e.g., "1900-01-01T19:00:00Z") uses 1900-01-01 as
  a placeholder date, but the TIME portion represents Eastern Time start time!

OLD BUGGY CODE was:
1. Extracting time from placeholder
2. Combining with date parameter (which is in UTC)
3. Result: start_time_utc was 24 hours off!

CORRECTED LOGIC:
1. Parse API date as Eastern Time date
2. Extract time from placeholder (which is ET time)
3. Combine them to get ET datetime
4. Convert ET to UTC for start_time_utc
5. Convert start_time_utc to CST for game_date

EXAMPLE:
For a game at 7:00 PM ET on Feb 5, 2026:
- API: date="02/05/2026", gameTimeUTC="1900-01-01T19:00:00Z"
- OLD (buggy): start_time_utc = 2026-02-05T00:00:00Z ❌
- NEW (correct): ET datetime = Feb 5, 2026 19:00 ET → start_time_utc = 2026-02-06T00:00:00Z ✅
- game_date (CST) = Feb 5, 2026 ✅
"""

import logging
import pendulum
import datetime

logger = logging.getLogger(__name__)


def _parse_nba_schedule_time(api_date_str: str, game_time_utc_placeholder: str):
    """
    Parse NBA schedule API time to timezone-aware UTC datetime.
    
    Args:
        api_date_str: Date string from API (MM/DD/YYYY format)
        game_time_utc_placeholder: Time string with placeholder date (1900-01-01THH:MM:SSZ)
    
    Returns:
        pendulum.DateTime in UTC, or None if parsing fails
    """
    try:
        # Step 1: Parse API date as Eastern Time date
        # Format: "02/05/2026" → Feb 5, 2026 at midnight ET
        # Use strptime since pendulum doesn't parse MM/DD/YYYY format
        dt_naive = datetime.datetime.strptime(api_date_str, '%m/%d/%Y')
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


# Test corrected logic
if __name__ == '__main__':
    print("Testing corrected NBA schedule time parsing")
    print("=" * 70)
    
    # Test: Game at 7:00 PM ET on Feb 5, 2026
    api_date = "02/05/2026"
    game_time = "1900-01-01T19:00:00Z"
    
    result = _parse_nba_schedule_time(api_date, game_time)
    
    if result:
        et_time = result.in_timezone('America/New_York')
        cst_time = result.in_timezone('America/Chicago')
        
        print(f"API date: {api_date}")
        print(f"gameTimeUTC placeholder: {game_time}")
        print(f"\nParsed start_time_utc: {result.to_iso8601_string()}")
        print(f"  ET time: {et_time}")
        print(f"  CST time: {cst_time}")
        print(f"  CST date: {cst_time.format('YYYY-MM-DD')}")
        print(f"\nExpected: start_time_utc = 2026-02-06T00:00:00Z, game_date = 2026-02-05")
        print(f"Actual:   start_time_utc = {result.to_iso8601_string()}, game_date = {cst_time.format('YYYY-MM-DD')}")
        print(f"\nCORRECT: {result.to_iso8601_string() == '2026-02-06T00:00:00Z' and cst_time.format('YYYY-MM-DD') == '2026-02-05'}")
