# GAME MONITORING NOT STARTING - SIGNAL HANDLER FIX

**Date:** February 10, 2026  
**Status:** ✅ FIXED AND DEPLOYED  
**Commit:** 09672e4  
**Severity:** 🔴 CRITICAL

---

## User Report

**Diagnostic Test Result:** 
```
instantiate: signal only works in main thread of main interpreter
```

**Symptoms:**
- Game monitoring won't toggle on
- Dashboard shows: `running: false, thread_alive: false, status: idle`
- Clicking "Toggle Game Monitoring" does nothing
- No error messages visible in UI

---

## Root Cause Analysis

### The Problem

Both `GameStateService` and `AutomationOrchestrator` were trying to set up **signal handlers** in their `__init__` methods:

```python
# GameStateService (line 78-79)
signal.signal(signal.SIGINT, self._signal_handler)
signal.signal(signal.SIGTERM, self._signal_handler)

# AutomationOrchestrator (line 64-65)
signal.signal(signal.SIGINT, self._handle_shutdown)
signal.signal(signal.SIGTERM, self._handle_shutdown)
```

**Why This Failed:**

1. **Streamlit Callbacks Run in Non-Main Thread**
   - When user clicks "Toggle Game Monitoring" button
   - Streamlit executes the callback in a **separate thread**
   - This is NOT the main interpreter thread

2. **signal.signal() Only Works in Main Thread**
   - Python's `signal.signal()` can only be called from the **main thread**
   - Calling it from a non-main thread raises an error
   - The error message: "signal only works in main thread of the main interpreter"

3. **Initialization Was Failing**
   - The signal handler setup was NOT wrapped in try-except (GameStateService)
   - AutomationOrchestrator only caught `ValueError`, not the actual error type
   - When signal setup failed, the entire `__init__` method failed
   - Service could never be created
   - Game monitoring toggle appeared to do nothing

---

## Solution

### Fix #1: GameStateService Signal Handling

**File:** `src/automation/game_state_service.py`

**Wrapped signal setup in comprehensive try-except:**

```python
# Setup signal handlers for graceful shutdown
# This will fail in Streamlit callbacks (non-main thread), which is OK
try:
    signal.signal(signal.SIGINT, self._signal_handler)
    signal.signal(signal.SIGTERM, self._signal_handler)
    logger.info("Signal handlers set up successfully")
except (ValueError, RuntimeError, AttributeError) as e:
    # Can't set signal handlers if not in main thread
    # This happens when running in Streamlit callbacks - that's OK
    # Signal handlers are optional, service will work without them
    logger.warning(f"Could not set signal handlers (not in main thread): {type(e).__name__}: {e}")
except Exception as e:
    # Catch any other signal-related errors
    # Signal handlers are optional, service will work without them
    logger.warning(f"Unexpected error setting up signal handlers: {type(e).__name__}: {e}")
    logger.debug(f"Signal handler setup error details:", exc_info=True)
```

**What Changed:**
- ✅ Wrapped signal setup in try-except
- ✅ Catches `ValueError`, `RuntimeError`, `AttributeError`
- ✅ Added catch-all `Exception` handler
- ✅ Logs warnings instead of failing initialization
- ✅ Made signal handlers **completely optional**

---

### Fix #2: AutomationOrchestrator Signal Handling

**File:** `src/automation/automation_orchestrator.py`

**Broadened existing exception handling:**

```python
# BEFORE (only caught ValueError):
except ValueError as e:
    logger.warning(f"Could not set signal handlers (not in main thread): {e}")

# AFTER (catches all signal errors):
except (ValueError, RuntimeError, AttributeError) as e:
    logger.warning(f"Could not set signal handlers (not in main thread): {type(e).__name__}: {e}")
except Exception as e:
    logger.warning(f"Unexpected error setting up signal handlers: {type(e).__name__}: {e}")
    logger.debug(f"Signal handler setup error details:", exc_info=True)
```

**What Changed:**
- ✅ Added `RuntimeError` to exception handling
- ✅ Added `AttributeError` to exception handling
- ✅ Added catch-all `Exception` handler
- ✅ Added debug logging with traceback
- ✅ Made signal handlers **completely optional**

---

## How It Works Now

### Before Fix
```
User clicks "Toggle Game Monitoring"
  ↓
Streamlit executes callback in non-main thread
  ↓
GameStateService.__init__() tries to set signal handlers
  ↓
signal.signal() raises error: "signal only works in main thread"
  ↓
__init__ fails with uncaught exception
  ↓
Service not created
  ↓
Game monitoring appears to do nothing ❌
```

### After Fix
```
User clicks "Toggle Game Monitoring"
  ↓
Streamlit executes callback in non-main thread
  ↓
GameStateService.__init__() tries to set signal handlers
  ↓
signal.signal() raises error: 