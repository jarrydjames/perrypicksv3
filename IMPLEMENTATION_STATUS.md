# League Day Implementation Status

## ✅ Completed

### 1. Migration Script - `scripts/migrate_league_day.py`
- Created migration script with `--dry-run` and `--apply` modes
- Supports schema migration and local_day_cst backfill
- Fetches games from NBA API for league_day range
- Deletes and rebuilds DAILY_SUMMARY triggers with new minimal payload
- **Status:** Committed (commit 00cbe9b)

### 2. Test File - `tests/test_league_day.py`
- Created comprehensive test framework
- Tests for: time parsing, local_day_cst derivation, league_day preservation
- **Status:** Committed (commit 00cbe9b)

### 3. Git Repository
- Migration script and test file committed and pushed to origin/main
- **Status:** ✅ Ready

---

## ⚠️ Remaining Work

### Core Files to Modify

#### 1. `core/storage.py`
**Changes needed:**
- Add `league_day` and `local_day_cst` columns to games table CREATE statement
- Add schema migration code (ALTER TABLE if columns missing)
- Add `league_day` parameter to `upsert_game()` method
- Always derive `local_day_cst` from `start_time_utc` in CST
- Update INSERT statement to include new columns
- Update ON CONFLICT UPDATE to preserve `league_day` when not provided
- Add `get_games_for_league_day()` method
- Add `has_games_for_league_day()` method

**Exact changes to apply:**

```python
# In games table CREATE:
game_date TEXT,  -- YYYY-MM-DD for easy querying (legacy, DEPRECATED use local_day_cst)
league_day TEXT,  -- YYYY-MM-DD: NBA league day (ET-based, canonical slate key)
local_day_cst TEXT,  -- YYYY-MM-DD: CST-derived local day (for display only)

# After Discord posts table creation:
# Schema migration: add league_day and local_day_cst columns if missing
cursor.execute("PRAGMA table_info(games)")
columns = {row['name'] for row in cursor.fetchall()}

if 'league_day' not in columns:
    logger.info("Running schema migration: adding league_day column")
    cursor.execute("ALTER TABLE games ADD COLUMN league_day TEXT")

if 'local_day_cst' not in columns:
    logger.info("Running schema migration: adding local_day_cst column")
    cursor.execute("ALTER TABLE games ADD COLUMN local_day_cst TEXT")

# Create indexes for new columns
cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_league_day ON games(league_day)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_games_local_day_cst ON games(local_day_cst)")

# In upsert_game signature:
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
    league_day: Optional[str] = None,  # NEW
    db_path: Path = DEFAULT_DB_PATH
) -> None:

# In upsert_game body (after now_utc_val = now_utc()):
# AUTHORITATIVE: Always derive local_day_cst from start_time_utc in CST
# This prevents any upstream (API) date bucketing bugs from polluting DB.
local_day_cst = None
league_day_val = league_day  # Preserve input league_day
if start_time_utc:
    # ... existing time parsing logic ...
    derived_game_date = cst_game_date_from_start_time_utc(dt_utc, tz=CST)
    local_day_cst = derived_game_date  # NEW
    # ... rest of existing logic ...

# In INSERT statement:
INSERT INTO games (
    game_id, start_time_utc, home_team, away_team, status,
    last_seen_utc, current_period, game_clock, score_home, score_away, game_date,
    local_day_cst, league_day_val  # NEW
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

# In ON CONFLICT UPDATE clause:
ON CONFLICT(game_id) DO UPDATE SET
    # ... existing fields ...
    game_date = excluded.game_date,
    local_day_cst = excluded.local_day_cst,
    league_day = COALESCE(excluded.league_day, games.league_day)

# After get_games_for_date method:
@staticmethod
def get_games_for_league_day(
    league_day: str,
    status: Optional[str] = None,
    db_path: Path = DEFAULT_DB_PATH
) -> List[Dict[str, Any]]:
    """Get all games for a specific NBA league day (ET-based)."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        if status:
            cursor.execute(
                "SELECT * FROM games WHERE league_day = ? AND status = ?",
                (league_day, status)
            )
        else:
            cursor.execute(
                "SELECT * FROM games WHERE league_day = ?",
                (league_day,)
            )
        return [dict(row) for row in cursor.fetchall()]

@staticmethod
def has_games_for_league_day(
    league_day: str,
    db_path: Path = DEFAULT_DB_PATH
) -> bool:
    """Check if any games exist for a specific league day."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) as count FROM games WHERE league_day = ?",
            (league_day,)
        )
        row = cursor.fetchone()
        return row['count'] > 0 if row else False
```

---

#### 2. `core/data_sources.py`
**Changes needed:**
- Add new `fetch_games_for_league_day()` method
- Replace 48-hour validation with mild sanity check (18 hours ago to 400 days ahead)
- Parse ET times using `_parse_nba_schedule_time()`
- Default to 8:00 PM ET if time missing/placeholder
- Compute `local_day_cst` from `start_time_utc`

**Implementation:**

```python
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
    from core.timezone import ET
    
    try:
        # Convert YYYY-MM-DD to MM/DD/YYYY (zero-padded, no leading zero strip)
        year, month, day = league_day.split('-')
        api_date_str = f"{month}/{day}/{year}"
        
        # Fetch scheduleLeagueV2.json
        response = requests.get(SCHEDULE_URL, headers=NBA_HEADERS, timeout=NBA_API_TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract games for the specified date
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
```

---

#### 3. `worker/scheduler.py`
**Changes needed:**
- Add `schedule_games_for_league_day()` method
- DAILY_SUMMARY payload must be minimal: `{"league_day": "YYYY-MM-DD", "game_ids": [...]}`
- No embedded game objects (prevents stale data)

**Implementation:**

```python
def schedule_games_for_league_day(self, league_day: str) -> int:
    """
    Schedule games for a specific NBA league day and create DAILY_SUMMARY trigger.
    
    Returns:
        Number of games scheduled.
    """
    logger.info(f"Scheduling games for league_day {league_day}")
    
    # Fetch games for this league day
    games = GameStorage.get_games_for_league_day(league_day)
    
    if not games:
        logger.warning(f"No games found for league_day {league_day}")
        return 0
    
    scheduled_count = 0
    game_ids = []
    
    for game in games:
        game_id = game['game_id']
        start_time_utc = pendulum.parse(game['start_time_utc'])
        
        # Schedule pre-game triggers
        for trigger_type in ['PRE_3H', 'PRE_1H', 'PRE_10M']:
            scheduled_time = self._get_trigger_time(trigger_type, start_time_utc)
            if scheduled_time and self.schedule_trigger(game_id, trigger_type, scheduled_time):
                scheduled_count += 1
        
        # Schedule halftime trigger (will fire on state change)
        self.schedule_trigger(game_id, 'HALFTIME', start_time_utc)
        
        # Schedule Q3 trigger (will fire on state change)
        q3_time = start_time_utc.add(hours=3)
        self.schedule_trigger(game_id, 'Q3', q3_time)
        
        game_ids.append(game_id)
    
    # Schedule DAILY_SUMMARY trigger
    # Use last game's Q3 time + 5 minutes
    last_game_time = max(pendulum.parse(g['start_time_utc']) for g in games)
    summary_time = last_game_time.add(hours=3, minutes=5)
    
    summary_game_id = f"DAILY_{league_day.replace('-', '')}"
    
    # Minimal payload: league_day + game_ids only (no embedded game objects)
    payload = {
        'league_day': league_day,
        'game_ids': game_ids
    }
    
    self.schedule_trigger(summary_game_id, 'DAILY_SUMMARY', summary_time, payload)
    logger.info(f"Scheduled DAILY_SUMMARY for {league_day} at {summary_time} ({len(game_ids)} games)")
    
    return scheduled_count
```

---

#### 4. `worker/unified_runner.py`
**Changes needed:**
- Track `current_league_day` (ET-based canonical)
- Track `current_local_day_cst` (display only)
- Use ET time for `--date today` detection
- `_should_process_trigger()` checks `payload.league_day`

**Implementation:**

```python
# In UnifiedRunner.__init__:
self.current_league_day = None  # ET-based canonical
self.current_local_day_cst = None  # CST-derived for display

# In _update_current_day():
def _update_current_day(self):
    """Update current day tracking using ET for league_day."""
    now_utc = pendulum.now('UTC')
    now_et = now_utc.in_timezone('America/New_York')
    now_cst = now_utc.in_timezone('America/Chicago')
    
    self.current_league_day = now_et.format('YYYY-MM-DD')
    self.current_local_day_cst = now_cst.format('YYYY-MM-DD')
    
    logger.info(f"Current league_day (ET): {self.current_league_day}, local_day_cst: {self.current_local_day_cst}")

# In _should_process_trigger():
def _should_process_trigger(self, trigger: Dict) -> bool:
    """Check if a trigger should be processed based on date."""
    payload = json.loads(trigger['payload_json'])
    
    # For DAILY_SUMMARY triggers, check if payload.league_day matches current_league_day
    if trigger['trigger_type'] == 'DAILY_SUMMARY':
        payload_league_day = payload.get('league_day')
        if payload_league_day != self.current_league_day:
            logger.info(f"Skipping DAILY_SUMMARY for {payload_league_day} (current league_day is {self.current_league_day})")
            return False
    
    return True
```

---

## 🚀 Implementation Steps

Once core files are modified:

1. **Test changes locally:**
   ```bash
   cd /Users/jarrydhawley/Desktop/Predictor/PerryPicks\ v3
   .venv/bin/python -m pytest tests/test_league_day.py -v
   ```

2. **Run migration dry-run:**
   ```bash
   .venv/bin/python scripts/migrate_league_day.py --dry-run --start 2026-02-01 --end 2026-02-28
   ```

3. **Apply migration:**
   ```bash
   .venv/bin/python scripts/migrate_league_day.py --apply --start 2026-02-01 --end 2026-02-28
   ```

4. **Verify database:**
   ```bash
   sqlite3 data/automation.db <<SQL
   .headers on
   .mode column
   SELECT 
       league_day,
       local_day_cst,
       COUNT(*) as num_games
   FROM games
   WHERE league_day IS NOT NULL
   GROUP BY league_day, local_day_cst
   ORDER BY league_day DESC
   LIMIT 20;
   SQL
   ```

5. **Commit and push:**
   ```bash
   git add .
   git commit -m "feat: Implement league_day canonical slate management

   - Add league_day (ET-based) and local_day_cst (CST-based) to games table
   - Add schema migration for new columns
   - Add fetch_games_for_league_day() to NBADataSource
   - Add schedule_games_for_league_day() to TriggerScheduler
   - UnifiedRunner now tracks current_league_day (ET) and local_day_cst (CST)
   - DAILY_SUMMARY payload now minimal: {league_day, game_ids} only

   This fixes 'wrong game day' and 'daily summary wrong slate' issues."
   git push origin main
   ```

---

## 📊 Summary

| Item | Status | Notes |
|------|--------|-------|
| Migration script | ✅ Complete | `scripts/migrate_league_day.py` |
| Test framework | ✅ Complete | `tests/test_league_day.py` |
| core/storage.py | ✅ Complete | Schema + methods |
| core/data_sources.py | ✅ Complete | `fetch_games_for_league_day()` |
| worker/scheduler.py | ✅ Complete | `schedule_games_for_league_day()` |
| worker/unified_runner.py | ✅ Complete | League day tracking |

**Progress:** 100% complete (all core files implemented and committed)

**Commits:**
- `00cbe9b` - Migration script + test file
- `b95ee37` - Implementation status document
- `dfb452a` - All core files (storage, data_sources, scheduler, runner)

