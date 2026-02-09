# Full Background Monitoring Implementation Summary

## Overview

Implemented full background monitoring with trigger engine to enable real-time monitoring of games for halftime and Q3 triggers. Predictions are automatically generated and queued when triggers fire.

## Implementation Details

### 1. Full Background Monitoring in `run_full_day_automation()`

Added `enable_background_monitoring` parameter that when set to `True`:

- Initializes `GameStateMonitor` with 30-second polling interval
- Initializes `AutoQueueProcessor` for queueing posts
- Initializes `TriggerEngine` with monitor, queue processor, and storage
- Starts monitoring in background daemon thread
- Stores monitor reference globally for stopping
- Updates session state with automation status

```python
results = run_full_day_automation(
    date=date.today(),
    platforms=['twitter'],
    dry_run=True,
    enable_background_monitoring=True,  # Start full monitoring!
    progress_callback=lambda p, m: print(f"[{p*100:.0f}%] {m}")
)
```

### 2. Helper Functions

#### `stop_automation()`

Stops running background automation:
- Sets session state to "stopping"
- Calls `monitor.stop()` to stop game state monitoring
- Sets `_automation_stop_event` for signal
- Waits up to 10 seconds for thread to stop
- Updates session state to "stopped"

**Note:** Thread may take up to `poll_interval` (30 seconds) to stop because it might be sleeping.

#### `get_automation_status()`

Returns current automation status:
```python
{
    "running": bool,
    "thread_alive": bool,
    "thread_name": str,
    "status": dict,  # From session state
}
```

#### `force_evaluate_triggers()`

Manually evaluate triggers for all monitored games:
- Updates game states for all games
- Checks if triggers fire
- Useful for testing or catching up on missed triggers

#### `render_automation_status()`

UI component for Streamlit app:
- Shows status indicators (running/stopped)
- Shows thread name and alive status
- Shows last update time
- Provides buttons:
  - ⏹️  Stop Automation (stops monitoring)
  - 🔄 Refresh Status (rerun to see latest status)
  - 🚀 Force Evaluate (manually evaluate triggers)

### 3. Thread Management

**Global Variables:**
- `_automation_thread` - The monitoring thread
- `_automation_stop_event` - Event for signaling stop
- `_automation_monitor` - Monitor reference for stopping

**Session State Keys:**
- `SESSION_STATE_AUTOMATION_RUNNING` - Boolean: is automation running?
- `SESSION_STATE_AUTOMATION_STATUS` - Dict with status details

### 4. Background Thread Flow

```
run_full_day_automation() called
    ↓
Create GameStateMonitor, TriggerEngine, AutoQueueProcessor
    ↓
Start background thread with monitor_loop()
    ↓
monitor_loop() runs:
    - Sets session state to "running"
    - Calls monitor.start() (blocking)
    - monitor.start() loops:
        * Check monitor.running flag
        * Update game states
        * Sleep for poll_interval (30s)
    - On exit: sets session state to not running
```

**Stopping:**
```
stop_automation() called
    ↓
Set session state to "stopping"
    ↓
Call monitor.stop() (sets monitor.running = False)
    ↓
Set stop event
    ↓
Wait up to 10s for thread to join
    ↓
Thread exits when monitor.running is False
    (may take up to 30s if sleeping)
```

## Usage Examples

### Basic Usage - Start Monitoring

```python
from src.automation.automation_ui import run_full_day_automation
from datetime import date

results = run_full_day_automation(
    date=date.today(),
    platforms=['twitter'],
    dry_run=True,
    enable_background_monitoring=True,
)

print(f"Monitoring status: {results['background_monitoring']['status']}")
print(f"Games monitored: {results['background_monitoring']['games_monitored']}")
```

### Check Status

```python
from src.automation.automation_ui import get_automation_status

status = get_automation_status()
if status['running']:
    print("✅ Automation is running")
    print(f"   Thread: {status['thread_name']}")
else:
    print("⏸️  Automation is stopped")
```

### Stop Monitoring

```python
from src.automation.automation_ui import stop_automation

result = stop_automation()
if result['success']:
    print(f"✅ {result['message']}")
    if not result['thread_stopped']:
        print("   (Thread may take up to 30s to stop if sleeping)")
```

### Streamlit UI Integration

```python
import streamlit as st
from src.automation.automation_ui import (
    render_automation_status,
    run_full_day_automation,
)

# Show status and controls
render_automation_status()

# Start button
if st.button("🚀 Start Full Automation"):
    results = run_full_day_automation(
        date=st.session_state.get('selected_date', date.today()),
        platforms=['twitter'],
        dry_run=st.session_state.get('dry_run', True),
        enable_background_monitoring=True,
    )
    st.success("Automation started!")
```

## Important Notes

### 1. Thread Behavior

- Background monitoring runs in a **daemon thread**
- Daemon threads are killed when the main Python process exits
- Thread cannot be stopped immediately - must wait for current sleep to finish
- This is normal and expected behavior

### 2. Stopping Time

When calling `stop_automation()`:
- The `monitor.running` flag is set to `False`
- The thread checks this flag after each sleep
- If the thread is sleeping for 30 seconds, it will stop **after** the sleep completes
- This means stopping can take up to 30 seconds
- The function waits 10 seconds and returns with status about whether thread stopped

### 3. Multiple Runs

If `run_full_day_automation()` is called with `enable_background_monitoring=True` while already running:
- The existing thread is joined (waits up to 5 seconds)
- A new monitor is created
- A new thread is started
- This ensures only one automation runs at a time

### 4. Session State

The automation uses Streamlit session state to track:
- Whether automation is running
- Current status details
- Last update timestamp

If session state is lost (e.g., browser refresh):
- The actual thread continues running (daemon thread)
- But the UI loses track of it
- Automation appears "stopped" even though it's still running

**Recommendation:** Use the UI controls to start/stop automation rather than relying on session state.

### 5. Production Considerations

For production use:
- Monitor thread is daemon - exits when app exits
- No persistent queue storage - queued posts lost on app restart
- Consider adding database-backed queue for persistence
- Consider adding monitoring of queue size
- Consider adding alerts for failed posts

## Testing

Run the test script to verify functionality:

```bash
cd "PerryPicks v3"
.venv/bin/python test_full_automation.py
```

Expected output:
- ✅ Background monitoring starts successfully
- ✅ Thread is alive and running
- ✅ Status tracking works
- ✅ Stop function works (may take up to 30s)
- ✅ Final status shows not running

## Files Modified

- `src/automation/automation_ui.py`:
  - Added `_automation_monitor` global variable
  - Added `enable_background_monitoring` parameter to `run_full_day_automation()`
  - Added full background monitoring logic with threading
  - Added `stop_automation()` function
  - Added `get_automation_status()` function
  - Added `force_evaluate_triggers()` function
  - Added `render_automation_status()` UI function
  - Added `format_timedelta()` helper function

## Next Steps

### Short-term Improvements

1. **Add monitoring interval configuration** - Allow customizing poll interval
2. **Add queue status display** - Show how many posts are queued
3. **Add trigger history** - Track which triggers fired and when
4. **Better error handling** - Graceful handling of API failures
5. **Add auto-restart** - Restart automation if it crashes

### Long-term Enhancements

1. **Persistent queue** - Database-backed queue for app restarts
2. **Webhook support** - Receive push notifications instead of polling
3. **Multi-monitoring** - Monitor multiple dates simultaneously
4. **Alert system** - Send alerts for errors or important events
5. **Analytics** - Track trigger performance and prediction accuracy