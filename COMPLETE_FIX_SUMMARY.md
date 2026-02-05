# NBA Schedule Timezone Bug - Complete Fix Summary

## Root Cause Diagnosis

### The Bug
Games that NBA.com lists as "Feb 5" (based on Eastern Time) are being classified as "Feb 4" in our system because:
1. Games at 7:00 PM ET on Feb 5 start at 2026-02-06T00:00:00Z (midnight Feb 6 UTC)
2. When converted to CST, this is 2026-02-05 18:00 (Feb 5, 6:00 PM CST) - CORRECT
3. BUT our current buggy code calculates start_time_utc as 2026-02-05T00:00:00Z (24 hours too early)
4. When converted to CST, this becomes 2026-02-04 18:00 (Feb 4, 6:00 PM CST) - WRONG!

### Why Current Code is Wrong
The NBA schedule API uses a non-intuitive format:
- `api_date_str` (e.g., "02/05/2026") = Eastern Time date of game
- `gameTimeUTC` (e.g., "1900-01-01T19:00:00Z") = placeholder date + Eastern Time start time

**Current buggy logic:**
1. Parse gameTimeUTC "1900-01-01T19:00:00Z" as UTC → gets 19:00 UTC
2. Combine with date parameter "2026-02-05" (in UTC from pendulum.parse())
3. Result: 2026-02-05T19:00:00Z ❌ (Wrong! Should be 2026-02-06T00:00:00Z)
**Corrected logic:**
1. Parse api_date_str "02/05/2026" as Eastern Time date
2. Parse gameTimeUTC "1900-01-01T19:00:00Z" → extract 19:00 (ET time)
3. Combine: Feb 5, 2026 19:00 ET
4. Convert to UTC: 2026-02-06T00:00:00Z ✅

## Database State Analysis

### Affected Games (Feb 4-5, 2026)
```
game_id     | start_time_utc         | game_date   | UTC Day    | CST Time
------------|----------------------|------------|------------|----------
0022500733  | 2026-02-05T00:00:00Z | 2026-02-04 | Feb 5 UTC   | 18:00 (Feb 4)
0022500734  | 2026-02-05T00:00:00Z | 2026-02-04 | Feb 5 UTC   | 18:00 (Feb 4)
0022500735  | 2026-02-05T00:30:00Z | 2026-02-04 | Feb 5 UTC   | 18:30 (Feb 4)
0022500736  | 2026-02-05T00:30:00Z | 2026-02-04 | Feb 5 UTC   | 18:30 (Feb 4)
0022500737  | 2026-02-05T01:00:00Z | 2026-02-04 | Feb 5 UTC   | 19:00 (Feb 4)
0022500738  | 2026-02-05T01:30:00Z | 2026-02-04 | Feb 5 UTC   | 19:30 (Feb 4)
0022500739  | 2026-02-05T03:00:00Z | 2026-02-04 | Feb 5 UTC   | 21:00 (Feb 4)
0022500740  | 2026-02-05T03:00:00Z | 2026-02-04 | Feb 5 UTC   | 21:00 (Feb 4)
```
**Issue:** Games 0022500733-0022500740 are Feb 5 games (per NBA.com) but are stored with game_date=2026-02-04 because:
- Current start_time_utc is WRONG (24 hours too early)
- This causes game_date to be Feb 4 instead of Feb 5

### DAILY_SUMMARY Triggers
- **DAILY_20260204** (Feb 4): Contains 8 games (should be different games)
- **DAILY_20260205** (Feb 5): Contains the same 8 games (0022500733-0022500740) but they're classified as Feb 4 in CST

Both triggers have the same WRONG games!

## The Fix

### 1. Correct `NBADataSource.fetch_games_for_date()`
Add new method `_parse_nba_schedule_time()` that:
- Parses API date as Eastern Time (not UTC)
- Extracts time from gameTimeUTC placeholder (which is ET time)
- Combines them to get ET datetime
- Converts ET to UTC for correct start_time_utc


### 2. Database Migration
Delete and re-fetch affected games:
- Delete games with start_time_utc in range [2026-02-04T00:00:00Z, 2026-02-06T01:00:00Z)
- Re-fetch from API with corrected logic
- Games will get correct start_time_utc and game_date
- Delete affected DAILY_SUMMARY triggers and re-schedule

### 3. Result After Fix
For game 0022500733 (WAS @ DET):
```
BEFORE:
  start_time_utc: 2026-02-05T00:00:00Z ❌
  game_date: 2026-02-04 ❌
  CST time: Feb 4, 18:00 ❌
  
AFTER:
  start_time_utc: 2026-02-06T00:00:00Z ✅
  game_date: 2026-02-05 ✅
  CST time: Feb 5, 18:00 ✅
```
This matches NBA.com: "Thursday, February 5, 2026" at "7:00 PM ET"

## Files to Modify

1. **core/data_sources.py**
   - Add `_parse_nba_schedule_time()` method
   - Update `fetch_games_for_date()` to use new parsing logic

2. **MIGRATE_FIX_TIMEZONE.py** (new file)
   - Script to delete and re-fetch affected games
   - Dry-run mode to preview changes
   - --apply flag to execute migration

## Verification Commands

### 1. Test corrected parsing logic
```bash
cd /Users/jarrydhawley/Desktop/Predictor/PerryPicks\ v3
.venv/bin/python FIX_DATA_SOURCES.py
```
**Expected output:**
```
API date: 02/05/2026
gameTimeUTC placeholder: 1900-01-01T19:00:00Z
Parsed start_time_utc: 2026-02-06T00:00:00Z
  ET time: 2026-02-05 19:00:00-05:00
  CST time: 2026-02-05 18:00:00-06:00
  CST date: 2026-02-05

Expected: start_time_utc = 2026-02-06T00:00:00Z, game_date = 2026-02-05
Actual:   start_time_utc = 2026-02-06T00:00:00Z, game_date = 2026-02-05
CORRECT: True ✅
```

### 2. Preview migration (dry run)
```bash
cd /Users/jarrydhawley/Desktop/Predictor/PerryPicks\ v3
.venv/bin/python MIGRATE_FIX_TIMEZONE.py
```
This will show what games and triggers will be affected without making changes.

### 3. Apply migration
```bash
cd /Users/jarrydhawley/Desktop/Predictor/PerryPicks\ v3
.venv/bin/python MIGRATE_FIX_TIMEZONE.py --apply
```
This will:
- Delete affected DAILY_SUMMARY triggers
- Delete affected games
- Re-fetch from API with corrected logic
- Re-create triggers with correct games

### 4. Verify database after migration
```bash
sqlite3 data/automation.db <<SQL
.headers on
.mode column
SELECT 
    game_id,
    home_team || ' @ ' || away_team as matchup,
    start_time_utc,
    game_date,
    strftime('%H:%M', datetime(start_time_utc, '-6 hours')) as cst_time,
    CASE 
        WHEN datetime(start_time_utc) < datetime('2026-02-05') THEN 'Feb 4 UTC or earlier'
        WHEN datetime(start_time_utc) < datetime('2026-02-06') THEN 'Feb 5 UTC'
        WHEN datetime(start_time_utc) < datetime('2026-02-07') THEN 'Feb 6 UTC'
        ELSE 'Feb 7 UTC or later'
    END as utc_day
FROM games
WHERE game_date IN ('2026-02-04', '2026-02-05')
  AND game_id NOT LIKE 'test_%'
ORDER BY game_date, start_time_utc;
SQL
```
**Expected after fix:**
```
game_id     | matchup        | start_time_utc         | game_date   | CST Time
------------|---------------|----------------------|------------|----------
0022500733  | WAS @ DET     | 2026-02-06T00:00:00Z | 2026-02-05 | 18:00 (Feb 5)
...
0022500741  | BOS @ MIA     | 2026-02-06T00:30:00Z | 2026-02-05 | 18:30 (Feb 5)
```

### 5. Check DAILY_SUMMARY triggers
```bash
sqlite3 data/automation.db <<SQL
.headers on
.mode column
SELECT 
    game_id,
    payload_date,
    json_array_length(payload_json->'$.games') as num_games
FROM triggers
WHERE trigger_type = 'DAILY_SUMMARY'
  AND game_id IN ('DAILY_20260204', 'DAILY_20260205')
ORDER BY scheduled_time_utc;
SQL
```
**Expected after fix:**
- DAILY_20260204: Contains correct Feb 4 games (7 games from API entry 02/04/2026)
- DAILY_20260205: Contains correct Feb 5 games (8 games from API entry 02/05/2026)

## Implementation Steps

1. **Apply code changes** to `core/data_sources.py`
   ```bash
   # Apply the BUGFIX_TIMEZONE_PATCH.diff
   git apply BUGFIX_TIMEZONE_PATCH.diff
   ```

2. **Preview migration** to see what will change
   ```bash
   .venv/bin/python MIGRATE_FIX_TIMEZONE.py
   ```

3. **Review the output** and confirm affected games/triggers look correct


4. **Apply migration** with --apply flag
   ```bash
   .venv/bin/python MIGRATE_FIX_TIMEZONE.py --apply
   ```

5. **Verify results** using the verification commands above


6. **Commit changes**
   ```bash
   git add -A
   git commit -m "fix: Correct NBA schedule API timezone parsing

   git push origin main
   ```

## API Date Semantics Summary

The NBA schedule API (scheduleLeagueV2.json) uses:
- **Date field** (`gameDate`): Eastern Time calendar date of games
- **Time field** (`gameTimeUTC`): Eastern Time start time (with placeholder date 1900-01-01)
- **Games are grouped by ET calendar day**, not by UTC start time

**Example:**
- A game at 7:00 PM ET on Feb 5:
  - Listed under "02/05/2026" in API
  - gameTimeUTC = "1900-01-01T19:00:00Z"
  - Correct start_time_utc = 2026-02-06T00:00:00Z (Feb 6 midnight UTC)
  - CST time = Feb 5, 18:00 (Feb 5 in CST, even though it's Feb 6 UTC)
- If using CST calendar day for classification:
  - Game falls on Feb 5 in CST → correct!
  - But only if start_time_utc is correct (not 24 hours off)
