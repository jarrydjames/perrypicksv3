# FIX SUMMARY: CST Game-Day Timezone Bug

## Overview

Fixed critical timezone bug where games starting late in CST evening were incorrectly 
assigned to the next day's `game_date`, causing wrong games to appear in daily summaries.

## Root Cause

- NBA API returns game times in UTC (e.g., `2026-02-04T03:00:00Z`)
- Old code set `game_date = date` (input parameter, not derived from UTC time)
- But `2026-02-04T03:00:00Z` = **Feb 3, 9:00 PM CST** (previous day!)
- Games were bucketed into wrong date, causing summary failures

## Example of the Bug

```python
# Game starting at 9:00 PM CST on Feb 3rd:
start_time_utc = "2026-02-04T03:00:00Z"  # Feb 4, 3:00 AM UTC

# OLD (WRONG):
game_date = "2026-02-04"  # Uses input date parameter
# Result: Game appears in Feb 4th summary (WRONG!)

# NEW (CORRECT):
game_date = cst_game_date_from_start_time_utc(start_time_utc)
# Result: game_date = "2026-02-03" (Derived from CST)
# Game appears in Feb 3rd summary (CORRECT!)
```

## Changes Made

### 1. Added Timezone Utility Function (`core/timezone.py`)

```python
def cst_game_date_from_start_time_utc(
    start_time_utc: Union[str, pendulum.DateTime],
    tz: str = CST
) -> str:
    