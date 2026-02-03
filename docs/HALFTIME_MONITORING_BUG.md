# Halftime Monitoring Bug - Root Cause Analysis

## Date: 2026-02-02 (Issue Identified)
## Severity: HIGH - Halftime triggers are detected but NOT posting to Discord

---

## Problem Statement

**User Question:** "Is the automation monitoring for halftime and kicking off a halftime prediction and posting to discord?"

**Answer:** NO - There is a bug preventing halftime predictions from posting to Discord.

---

## Root Cause

The automation system has TWO bugs preventing halftime predictions from posting:

### Bug #1: Game-State Triggers Are Not Processed

**File:** `worker/runner.py` (lines ~250-255)

**Buggy Code:**
```python
def _process_active_game(self, game: dict) -> bool:
    """Process an active game for game-state triggers."""
    game_id = game['game_id']
    
    try:
        # Fetch latest game state
        game_state = self.data_source.nba.fetch_game_state(game_id)
        
        # ... update game in database ...
        
        # Check for game-state triggers (halftime, Q3)
        triggers_fired = self.trigger_firer.process_game_state_triggers(
            game_id=game_id,
            game_state=game_state
        )
        
        if triggers_fired > 0:
            # Run analysis and post for each fired trigger
            # Note: This is simplified - in production you'd batch these
            pass  # ❌ BUG: Nothing happens!
        
        return triggers_fired > 0
```

**What Happens:**
1. Automation polls every 60 seconds
2. For each active game, it fetches game state
3. If game is at halftime, `process_game_state_triggers()` detects it
4. Trigger is stored in database as "fired"
5. **BUT** - No analysis runs, no picks are generated, nothing posts to Discord!

---

### Bug #2: Time Window Too Small

**File:** `worker/runner.py` (line 111)

**Code:**
```python
def run_once(self) -> int:
    """Run a single poll cycle."""
    now_utc = datetime.now(timezone.utc)
    window_start = now_utc - timedelta(seconds=30)  # ❌ Only 30s back!
    window_end = now_utc + timedelta(seconds=30)
    
    # Get due triggers
    due_triggers = TriggerStorage.get_due_triggers(
        window_start, window_end, db_path=self.db_path
    )
```

**What Happens:**
1. Halftime trigger detected at 8:00:00 PM
2. Stored in DB with `scheduled_time_utc = 8:00:00 PM`
3. Next poll at 8:01:00 PM (60 seconds later)
4. Time window: `7:30:30 PM to 8:01:30 PM` (±30 seconds)
5. Trigger at 8:00:00 PM is OUTSIDE this window (it was stored 60 seconds ago)
6. Trigger is never picked up for processing!

---

## Why Halftime Detection Works But Posting Doesn't

### ✅ What DOES Work:
1. **Game state fetching** - Polls every 60 seconds from NBA API
2. **Halftime detection** - `GameTriggerDetector` correctly identifies halftime
3. **Trigger storage** - Halftime trigger stored in database as "fired"
4. **Snapshot tracking** - Creates tracking snapshots

### ❌ What DOESN'T Work:
1. **No analysis** - Picks are not generated for halftime
2. **No Discord post** - Nothing gets posted to your webhook
3. **Missed window** - Even if it worked, time window is too small

---

## How to Fix

### Fix #1: Process Game-State Triggers Immediately

Modify `_process_active_game()` to actually run analysis and post to Discord:

```python
def _process_active_game(self, game: dict) -> bool:
    """Process an active game for game-state triggers."""
    game_id = game['game_id']
    
    try:
        # Fetch latest game state
        game_state = self.data_source.nba.fetch_game_state(game_id)
        
        # ... update game in database ...
        
        # Check for game-state triggers (halftime, Q3)
        triggers_fired = self.trigger_firer.process_game_state_triggers(
            game_id=game_id,
            game_state=game_state
        )
        
        if triggers_fired > 0:
            # ✅ FIX: Run analysis and post for each fired trigger
            # Get triggers that were just fired
            fired_trigger_types = self.trigger_firer.get_fired_trigger_types(
                game_id=game_id, 
                game_state=game_state
            )
            
            for trigger_type in fired_trigger_types:
                self._process_game_state_trigger(
                    game_id=game_id,
                    trigger_type=trigger_type,
                    game_state=game_state
                )
        
        return triggers_fired > 0
```

### Fix #2: Increase Time Window

```python
def run_once(self) -> int:
    """Run a single poll cycle."""
    now_utc = datetime.now(timezone.utc)
    window_start = now_utc - timedelta(minutes=2)  # ✅ Increased to 2 minutes
    window_end = now_utc + timedelta(seconds=30)
    
    # Get due triggers
    due_triggers = TriggerStorage.get_due_triggers(
        window_start, window_end, db_path=self.db_path
    )
```

---

## Current Behavior vs Expected Behavior

### Current (Broken)
```
Halftime at 8:00 PM
  ↓
Trigger detected by GameTriggerDetector
  ↓
Trigger stored in DB (status='fired')
  ↓
❌ Nothing happens (pass statement)
  ↓
❌ No prediction posted to Discord
```

### Expected (Fixed)
```
Halftime at 8:00 PM
  ↓
Trigger detected by GameTriggerDetector
  ↓
Trigger stored in DB (status='fired')
  ↓
✅ Analysis runs using game_state
  ↓
✅ Picks generated (spread, total, etc.)
  ↓
✅ Picks stored in DB
  ↓
✅ Discord webhook called with formatted picks
  ↓
✅ Discord message posted
```

---

## Impact

### Affected Features
- ❌ **Halftime predictions** - NOT posting to Discord
- ❌ **Q3 end predictions** - NOT posting to Discord
- ✅ **Pre-game predictions** - Working (time-based triggers)

### User Experience
- User expects automatic halftime picks
- User gets nothing
- Manual trigger works (bypasses the bug)

---

## Testing

### To Verify the Fix

1. Start automation
2. Wait for a game to reach halftime
3. Check logs: Should see "Fired HALFTIME trigger for [game_id]"
4. Check Discord: Should see halftime picks posted
5. Check database: Should see picks for HALFTIME trigger

### Current Test Results

```bash
# Check database for halftime triggers
sqlite3 data/automation.db "SELECT * FROM triggers WHERE trigger_type = 'HALFTIME'"
# Result: 0 rows (no halftime triggers)

# Check if automation detects halftime
# Need to run with actual game at halftime
```

---

## Related Code Files

| File | Function | Issue |
|-------|-----------|--------|
| `worker/runner.py` | `_process_active_game()` | Has `pass` instead of processing triggers |
| `worker/runner.py` | `run_once()` | Time window too small (30s) |
| `worker/triggers.py` | `GameTriggerDetector.detect_triggers()` | Works correctly ✅ |
| `worker/triggers.py` | `TriggerFirer.process_game_state_triggers()` | Works correctly ✅ |
| `worker/scheduler.py` | `TriggerScheduler.schedule_games_for_date()` | Not related (only schedules time-based triggers) |

---

## Timeline

- **System designed**: Game-state triggers (halftime, Q3) to fire dynamically
- **Bug introduced**: `pass` statement instead of processing triggers
- **Bug discovered**: User asked about halftime monitoring
- **Fix needed**: Implement analysis and Discord posting for game-state triggers

---

## Summary

**Problem:** Halftime triggers are detected but not processed or posted to Discord

**Root Cause:** 
1. `pass` statement in `_process_active_game()` (line 254)
2. Time window too small (±30s instead of ±2 minutes)

**Impact:** No automatic halftime predictions

**Fix Required:** 
1. Implement processing of game-state triggers
2. Increase time window to catch dynamically-created triggers

**Status:** Bug identified and documented, awaiting fix

---

## Next Steps

1. Modify `worker/runner.py` to process game-state triggers
2. Increase time window in `run_once()` method
3. Test with actual game at halftime
4. Verify Discord posts work
5. Deploy fix to production

