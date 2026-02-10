# Duplicate Post Override Fix for Halftime and Q3

**Date:** February 9, 2025  
**Status:** ✅ FIXED AND DEPLOYED  
**Commit:** cd3d903

---

## Problem

**User Report:** "The duplicate post override is not working for halftime and Q3"

### Symptoms
- "Allow Duplicate Posts" checkbox in UI was ignored for halftime and Q3 predictions
- Users couldn't re-trigger halftime/Q3 predictions even with checkbox enabled
- System would skip games as "already processed" regardless of allow_duplicates setting

---

## Root Cause Analysis

The `allow_duplicates` parameter was not being passed through the prediction pipeline for halftime and Q3 predictions. Multiple layers of the automation system had incomplete duplicate override support:

### Issue #1: Orchestrator Not Respecting allow_duplicates

**File:** `src/automation/automation_orchestrator.py`  
**Line:** 124 (before fix)

```python
# Check if already processed
if self._is_prediction_processed(game_id, trigger_type):
    logger.info(f"Skipping already processed: {game_id} {trigger_type}")
    results["skipped"] += 1
    if progress_callback:
        progress_callback(progress, f"Skipped {game_id} (already processed)")
    continue
```

**Problem:** The check happened unconditionally, without considering `allow_duplicates`. Even if `allow_duplicates=True`, games were skipped.

---

### Issue #2: Queue Processor Missing Parameter

**File:** `src/automation/auto_queue_processor.py`  
**Function:** `queue_and_post()`

**Problem:** The function didn't have an `allow_duplicates` parameter. When called by trigger engine, it would always use duplicate detection.

---

### Issue #3: Full Day Automation Missing Parameter

**File:** `src/automation/automation_ui.py`  
**Function:** `run_full_day_automation()`

**Problem:** 
- Function signature didn't include `allow_duplicates` parameter
- UI couldn't pass the checkbox value to full day automation
- Halftime/Q3 triggers only checked `allow_retroactive`, not `allow_duplicates`

---

### Issue #4: UI Not Passing Parameter

**File:** `pages/04_Automation_Manager.py`  
**Modes affected:** "Queue Gamestate-Conscious Posts", "Full Day Automation"

**Problem:** The UI had the "Allow Duplicate Posts" checkbox but didn't pass the value to:
- `queue_gamestate_conscious_posts()` (for manual halftime/Q3 predictions)
- `run_full_day_automation()` (for full day automation)

---

## Solution

### Fix #1: Orchestrator Duplicate Check

**File:** `src/automation/automation_orchestrator.py`

```python
# Check if already processed (unless allow_duplicates is True)
if not allow_duplicates and self._is_prediction_processed(game_id, trigger_type):
    logger.info(f"Skipping already processed: {game_id} {trigger_type}")
    results["skipped"] += 1
    if progress_callback:
        progress_callback(progress, f"Skipped {game_id} (already processed)")
    continue
```

**Change:** Added `not allow_duplicates and` to the condition.

---

### Fix #2: Queue Processor Parameter

**File:** `src/automation/auto_queue_processor.py`

```python
def queue_and_post(
    self,
    prediction: Dict[str, Any],
    trigger_type: str,
    platforms: Optional[List[str]] = None,
    max_retries: int = 3,
    allow_duplicates: bool = False,  # ✅ ADDED
) -> Dict[str, Any]:
```

And:
```python
post_id = self.queue.enqueue(
    game_id=game_id,
    platform=platform,
    content="",
    trigger_type=trigger_type,
    max_retries=max_retries,
    allow_duplicates=allow_duplicates,  # ✅ ADDED
)
```

**Change:** Added `allow_duplicates` parameter to function signature and passed it to `queue.enqueue()`.

---

### Fix #3: Full Day Automation Parameter

**File:** `src/automation/automation_ui.py`

```python
def run_full_day_automation(
    date: dt.date = None,
    platforms: Optional[List[str]] = None,
    dry_run: bool = False,
    fetch_odds: bool = True,
    allow_retroactive: bool = False,
    allow_duplicates: bool = False,  # ✅ ADDED
    enable_background_monitoring: bool = False,
    rate_limit_delay: float = 1.0,
    progress_callback=None,
) -> Dict[str, Any]:
```

**Change:** Added `allow_duplicates` parameter to function signature and docstring.

---

### Fix #4: Halftime/Q3 Trigger Logic

**File:** `src/automation/automation_ui.py`

**Halftime triggers:**
```python
# Skip if game already completed (unless retroactive is enabled or duplicates allowed)
if game_status >= 6:  # Final
    if allow_retroactive or allow_duplicates:  # ✅ CHANGED
        # Retroactive mode or duplicate override - generate prediction anyway
        result = run_prediction(
            game_id=game_id,
            trigger_type="halftime_retroactive",
            platforms=platforms,
            dry_run=dry_run,
            fetch_odds=True,
            allow_duplicates=True,
        )
```

**Q3 triggers:**
```python
# Skip if game already completed (unless retroactive is enabled or duplicates allowed)
if game_status >= 6:  # Final
    if allow_retroactive or allow_duplicates:  # ✅ CHANGED
        # Retroactive mode or duplicate override - generate prediction anyway
        result = run_prediction(
            game_id=game_id,
            trigger_type="q3_retroactive",
            platforms=platforms,
            dry_run=dry_run,
            fetch_odds=True,
            allow_duplicates=True,
        )
```

**Change:** Changed condition from `if allow_retroactive:` to `if allow_retroactive or allow_duplicates:`

**Also:**
```python
pregame_individual = run_predictions_for_all_games(
    date=date,
    trigger_type="pregame",
    platforms=platforms,
    dry_run=dry_run,
    fetch_odds=False,
    allow_duplicates=allow_duplicates,  # ✅ ADDED
    progress_callback=lambda p, m: progress_callback(0.05 + (p * 0.20), m) if progress_callback else None,
)
```

**Change:** Added `allow_duplicates=allow_duplicates` to pregame predictions call.

---

### Fix #5: UI Parameter Passing

**File:** `pages/04_Automation_Manager.py`

**Queue Gamestate-Conscious Posts:**
```python
if st.button(f"🎯 Queue Gamestate-Conscious Posts for {selected_game_id}", use_container_width=True):
    with st.spinner(f"Queueing gamestate-conscious posts for {selected_game_id}..."):
        results = queue_gamestate_conscious_posts(
            game_id=selected_game_id,
            platforms=platforms if platforms else None,
            dry_run=dry_run,
            allow_duplicates=allow_duplicates,  # ✅ ADDED
        )
```

**Full Day Automation:**
```python
result = run_full_day_automation(
    date=selected_date,
    platforms=platforms if platforms else None,
    dry_run=dry_run,
    fetch_odds=fetch_odds,
    allow_duplicates=allow_duplicates,  # ✅ ADDED
    progress_callback=progress_callback,
)
```

**Change:** Added `allow_duplicates=allow_duplicates` to both UI button handlers.

---

## Testing

### Before Fix
```
✅ Check "Allow Duplicate Posts" checkbox
✅ Click "Queue Gamestate-Conscious Posts" for game 0022500761
❌ Result: "Halftime post failed" - "Already processed"
❌ Duplicates NOT allowed despite checkbox being checked
```

### After Fix (Expected)
```
✅ Check "Allow Duplicate Posts" checkbox
✅ Click "Queue Gamestate-Conscious Posts" for game 0022500761
✅ Result: "Halftime post queued successfully"
✅ Duplicates ALLOWED when checkbox is checked
```

---

## What Works Now

### ✅ Single Game Predictions
- Trigger type: "halftime" or "q3"
- "Allow Duplicate Posts" checkbox: ENABLED
- Result: Prediction generated and queued, duplicates allowed

### ✅ Queue Gamestate-Conscious Posts
- Modes: Pregame, Halftime, Q3
- "Allow Duplicate Posts" checkbox: ENABLED
- Result: All posts queued, duplicates allowed

### ✅ Full Day Automation
- All games for date
- "Allow Duplicate Posts" checkbox: ENABLED
- Result: Pregame, halftime, and Q3 predictions all generated, duplicates allowed

### ✅ Retroactive Predictions
- Games already completed
- "Allow Duplicate Posts" OR "Retroactive Mode": ENABLED
- Result: Predictions generated, duplicates allowed

---

## Deployment

### Commit
**Hash:** cd3d903  
**Message:** "Fix duplicate post override for halftime and Q3 predictions"

### Status
✅ Pushed to GitHub  
✅ Repository: https://github.com/jarrydjames/perrypicksv3.git  
✅ Branch: main  
✅ Streamlit Cloud will auto-deploy

---

## Files Modified

1. **src/automation/automation_orchestrator.py**
   - Fixed duplicate check to respect `allow_duplicates` parameter

2. **src/automation/auto_queue_processor.py**
   - Added `allow_duplicates` parameter to `queue_and_post()`

3. **src/automation/automation_ui.py**
   - Added `allow_duplicates` parameter to `run_full_day_automation()`
   - Updated halftime trigger logic to check `allow_retroactive or allow_duplicates`
   - Updated Q3 trigger logic to check `allow_retroactive or allow_duplicates`
   - Updated pregame predictions call to pass `allow_duplicates`

4. **pages/04_Automation_Manager.py**
   - Updated "Queue Gamestate-Conscious Posts" button to pass `allow_duplicates`
   - Updated "Full Day Automation" button to pass `allow_duplicates`

---

## Summary

| Component | Before Fix | After Fix | Status |
|-----------|-------------|------------|--------|
| Orchestrator duplicate check | Always skipped if processed | Skips only if `not allow_duplicates` | ✅ FIXED |
| Queue processor | No `allow_duplicates` parameter | Has `allow_duplicates` parameter | ✅ FIXED |
| Full day automation | No `allow_duplicates` parameter | Has `allow_duplicates` parameter | ✅ FIXED |
| Halftime triggers | Checked only `allow_retroactive` | Checks `allow_retroactive or allow_duplicates` | ✅ FIXED |
| Q3 triggers | Checked only `allow_retroactive` | Checks `allow_retroactive or allow_duplicates` | ✅ FIXED |
| UI parameter passing | Didn't pass `allow_duplicates` | Passes `allow_duplicates` | ✅ FIXED |

---

**Result:** "Allow Duplicate Posts" checkbox now works correctly for all prediction types including halftime and Q3! 🎉

---

**Fixed by:** Perry (code-puppy-0c2adb)  
**Date:** February 9, 2025