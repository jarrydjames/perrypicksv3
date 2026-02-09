# Dashboard Service Status Features

## Overview

Added real-time status indicators and controls for game monitoring and automated queue processing to the PerryPicks dashboard.

---

## What's New

### 1. Dashboard Status Indicators

**Location:** Dashboard tab (first tab, top section)

Shows live status of both services:

```
┌─────────────────────────────────────────────────────────────────┐
│ 🚦 Service Status                                               │
├─────────────────────────────┬───────────────────────────────────┤
│ 🎮 Game Monitoring         │ 📨 Queue Processing              │
│ 🟢 LIVE                    │ 🟢 LIVE                          │
│ Thread: Running            │ Thread: Running                  │
│ Last activity: 2m ago      │ Posts processed: 42              │
│                           │                                   │
│ [Go to Game State Settings]│ [Toggle Queue Processing]        │
└─────────────────────────────┴───────────────────────────────────┘
```

**Features:**
- 🟢 **LIVE** (green) - Service is running and active
- 🔴 **STOPPED** (yellow/red) - Service is not running
- Thread status - Shows if background thread is alive
- Last activity - Shows when service was last active
- Quick toggle button - Start/stop queue processing
- Quick link - Jump to Game State tab

---

### 2. Game State Tab - Complete Controls

**Location:** Game State tab (last tab)

#### Service Status Section

```
┌─────────────────────────────────────────────────────────────────┐
│ 🚦 Service Status                                               │
├─────────────────────────────┬───────────────────────────────────┤
│ 🎮 Game Monitoring         │ 📨 Queue Processing              │
│ 🟢 LIVE - Game State       │ 🟢 LIVE - Queue Processor is     │
│    Monitor is active       │    active                        │
│ Thread: GameStateMonitor   │ Thread: BackgroundQueueProcessor  │
│ Last activity: 1m ago      │ Posts processed: 42              │
│                           │ Last processed: 2m ago            │
└─────────────────────────────┴───────────────────────────────────┘
```

#### Automated Queue Processing Toggle

```
┌─────────────────────────────────────────────────────────────────┐
│ 🎛️ Automated Queue Processing                                   │
├─────────────────────────────┬───────────────────────────────────┤
│ ⚡ Queue Processor Control │ ⚙️ Configuration                  │
│                           │                                   │
│ [🤖 Enable Automated Queue│ Poll Interval: [15] seconds      │
│  Processing] ☑️           │                                   │
│                           │ Batch Size: [10] posts           │
│ ✅ Automated queue        │                                   │
│    processing is ENABLED   │ [Apply Configuration]            │
│                           │                                   │
│ Queue will be processed   │                                   │
│ every 15 seconds          │                                   │
│ automatically             │                                   │
└─────────────────────────────┴───────────────────────────────────┘
```

**Toggle Switch Behavior:**

When you toggle ON:
- ✅ Queue processor starts automatically
- ✅ Shows "ENABLED" status
- ✅ Queue processes every 15 seconds
- ✅ Posts go out automatically

When you toggle OFF:
- ⏸️  Queue processor stops automatically
- ✅ Shows "DISABLED" status
- ✅ Queue processing becomes manual
- ⚠️  You must process queue manually

#### Manual Controls

```
┌─────────────────────────────────────────────────────────────────┐
│ 🎛️ Manual Controls                                             │
├─────────────────────┬─────────────────────┬─────────────────────┤
│ [▶️ Start Queue     │ [⏹️ Stop Queue      │ [⚡ Process Queue    │
│  Processor]        │  Processor]        │  Now]               │
│                     │                     │                     │
│ Starts background   │ Stops background   │ Process all         │
│ queue processor     │ queue processor    │ pending posts       │
└─────────────────────┴─────────────────────┴─────────────────────┘
```

#### Detailed Status

```
┌─────────────────────────────────────────────────────────────────┐
│ 📊 Detailed Status                                              │
├─────────────────────────────┬───────────────────────────────────┤
│ Game Monitor Details       │ Queue Processor Details           │
│                           │                                   │
│ ✅ Running                 │ ✅ Running                        │
│ 🧵 AutomationThread       │ 🧵 BackgroundQueueProcessor      │
│ 📊 Monitoring              │ ⏱️ 15s / 10 posts                │
│                           │                                   │
│ Total: 50                  │ Total: 50                        │
│ Pending: 5                 │ Pending: 5                       │
│ Posted: 42                 │ Posted: 42                       │
│ Failed: 3                  │ Failed: 3                        │
│                           │                                   │
│ 📊 Processor Stats        │ 📊 Processor Stats               │
│                           │                                   │
│ Processed: 42              │ Processed: 42                    │
│ Failed: 0                  │ Failed: 3                        │
│ Last: 2m ago              │ Last: 2m ago                     │
│                           │                                   │
│ [⏹️  Stop Automation]     │ [⏹️  Stop Queue Processor]       │
│ [▶️ Start Automation]     │ [▶️ Start Queue Processor]       │
│ [🔄 Refresh]              │ [🔄 Refresh]                      │
│ [⚡ Process Now]          │ [⚡ Process Now]                  │
└─────────────────────────────┴───────────────────────────────────┘
```

---

## How It Works

### Status Indicators

#### Game Monitoring Status

| Status | Color | Meaning |
|--------|-------|---------|
| LIVE | 🟢 Green | Game State Monitor is running and monitoring games |
| STOPPED | 🔴 Red | Game State Monitor is not running |
| Thread: Running | ✅ | Background thread is active |
| Thread: Inactive | ⏸️  | Background thread is stopped |
| Last activity: X ago | ℹ️ | Time since last game state update |

#### Queue Processing Status

| Status | Color | Meaning |
|--------|-------|---------|
| LIVE | 🟢 Green | Queue Processor is running and processing queue |
| STOPPED | 🔴 Red | Queue Processor is not running |
| Thread: Running | ✅ | Background thread is active |
| Thread: Inactive | ⏸️  | Background thread is stopped |
| Posts processed: N | 📊 | Total posts processed by queue processor |

### Toggle Switch

**How it works:**

1. Toggle ON → Queue processor starts automatically
2. Toggle OFF → Queue processor stops automatically
3. State is tracked in session state
4. Automatic start/stop without manual intervention

**Code flow:**

```python
# Get current status
queue_status = get_queue_processor_status()
is_running = queue_status.get("running", False)

# Toggle switch
auto_queue = st.toggle(
    "🤖 Enable Automated Queue Processing",
    value=is_running,
)

# Check if state changed
if auto_queue != st.session_state["auto_queue_enabled_prev"]:
    # State changed!
    st.session_state["auto_queue_enabled_prev"] = auto_queue
    
    if auto_queue:
        # Enable - start queue processor
        result = start_queue_processor(
            poll_interval=15,
            batch_size=10,
        )
    else:
        # Disable - stop queue processor
        result = stop_queue_processor()
```

### Dashboard Quick Toggle

**Location:** Dashboard tab

**What it does:**
- If queue processing is OFF → Starts it
- If queue processing is ON → Stops it

**Use case:**
- Quick toggle without going to Game State tab
- One-click control for automated queue processing

---

## Usage Examples

### Example 1: Enable Automated Queue Processing

**Goal:** Turn on automated queue processing

**Steps:**

1. Go to **Game State** tab
2. Find **"🎛️ Automated Queue Processing"** section
3. Toggle **"🤖 Enable Automated Queue Processing"** to ON
4. See status: **"✅ Automated queue processing is ENABLED"**

**Result:**
- Queue processor starts automatically
- Queue processes every 15 seconds
- Posts go out automatically

### Example 2: Check Service Status from Dashboard

**Goal:** Check if services are running

**Steps:**

1. Go to **Dashboard** tab
2. Find **"🚦 Service Status"** section
3. See both statuses:
   - **Game Monitoring:** 🟢 LIVE / 🔴 STOPPED
   - **Queue Processing:** 🟢 LIVE / 🔴 STOPPED

**Result:**
- Instant visibility of service status
- No need to navigate to Game State tab

### Example 3: Quick Toggle Queue Processing

**Goal:** Start/stop queue processing quickly

**Steps:**

1. Go to **Dashboard** tab
2. Find **"📨 Queue Processing"** in Service Status
3. Click **"🔘 Toggle Queue Processing"** button

**Result:**
- If OFF → Starts queue processing
- If ON → Stops queue processing
- Status updates immediately

### Example 4: Manual Queue Processing

**Goal:** Process queue manually (one-time)

**Steps:**

1. Go to **Game State** tab
2. Find **"🎛️ Manual Controls"** section
3. Click **"⚡ Process Queue Now"** button

**Result:**
- Processes all pending posts immediately
- One-time operation (doesn't affect automated processing)

### Example 5: Configure Queue Processor

**Goal:** Change poll interval or batch size

**Steps:**

1. Go to **Game State** tab
2. Find **"⚙️ Configuration"** section
3. Adjust settings:
   - **Poll Interval:** Change from 15s to 30s
   - **Batch Size:** Change from 10 to 20
4. Click **"⚙️ Apply Configuration"** button

**Result:**
- New settings applied on next start
- If queue processor is running, restart it

---

## Architecture

### Components

#### 1. Status Functions

- `get_automation_status()` - Get game monitoring status
- `get_queue_processor_status()` - Get queue processor status

#### 2. Control Functions

- `start_queue_processor()` - Start background queue processor
- `stop_queue_processor()` - Stop background queue processor

#### 3. UI Components

- `render_automation_status()` - Render game monitoring status details
- `render_queue_processor_status()` - Render queue processor status details

#### 4. Dashboard UI

- Status indicators (Dashboard tab)
- Quick toggle button (Dashboard tab)
- Complete controls (Game State tab)
- Detailed status views (Game State tab)

### Data Flow

```
User Interface (Streamlit)
    ↓
Status Functions
    ├─ get_automation_status()
    └─ get_queue_processor_status()
        ↓
Background Services
    ├─ Game State Monitor (thread)
    └─ Queue Processor (thread)
        ↓
Status Display
    ├─ Status indicators (LIVE/STOPPED)
    ├─ Thread status (Running/Inactive)
    └─ Last activity times
        ↓
User Actions
    ├─ Toggle switch → start/stop queue processor
    ├─ Manual controls → start/stop/process now
    └─ Configuration → apply settings
```

---

## Technical Details

### Session State Tracking

Toggle state is tracked to prevent duplicate actions:

```python
if "auto_queue_enabled_prev" not in st.session_state:
    st.session_state["auto_queue_enabled_prev"] = is_running

if auto_queue != st.session_state["auto_queue_enabled_prev"]:
    # State changed - take action
    ...
```

### Status Refresh

Status is fetched on every rerun:

```python
# Get current status (always fresh)
automation_status = get_automation_status()
queue_status = get_queue_processor_status()
```

### Thread Status

Thread status indicates if background daemon thread is alive:

```python
automation_status.get("thread_alive")  # True/False
queue_status.get("thread_alive")  # True/False
```

### Last Activity

Shows when service was last active:

```python
# Game monitoring
status_data.get("last_update")  # ISO timestamp

# Queue processing
stats.get("last_processed_at")  # ISO timestamp
```

### Automatic Start/Stop

Toggle switch automatically starts/stops queue processor:

```python
if auto_queue:
    result = start_queue_processor(poll_interval=15, batch_size=10)
else:
    result = stop_queue_processor()
```

---

## Troubleshooting

### Issue: Status Shows STOPPED but Service is Running

**Symptom:** Dashboard shows STOPPED but you know service is running

**Cause:** Service started outside of Streamlit (e.g., terminal)

**Solution:** 
- Services started externally won't show in Streamlit status
- Use Game State tab controls for Streamlit-aware service control
- Or check terminal/service logs for external service status

### Issue: Toggle Switch Doesn't Start Queue Processor

**Symptom:** Toggle ON but queue processor doesn't start

**Cause:** Social manager not initialized

**Solution:**
1. Go to Settings tab
2. Initialize orchestrator
3. Ensure API credentials are configured
4. Retry toggle

### Issue: Toggle Switch Doesn't Stop Queue Processor

**Symptom:** Toggle OFF but queue processor keeps running

**Cause:** Thread is sleeping (up to poll_interval)

**Solution:**
- Wait up to poll_interval (15 seconds) for thread to stop
- Check logs for errors
- Use "Stop Queue Processor" button for manual stop

### Issue: Status Doesn't Update

**Symptom:** Status shows stale information

**Cause:** Streamlit hasn't rerun

**Solution:**
- Click refresh button
- Or click "Toggle Queue Processing" to force rerun
- Status updates on every rerun

### Issue: Thread Status Shows Inactive but Service Shows LIVE

**Symptom:** Status says LIVE but thread says Inactive

**Cause:** Service was started externally

**Solution:**
- Use Game State tab controls to start service within Streamlit
- Or rely on external service and ignore Streamlit status

---

## Summary

**What was added:**

✅ **Dashboard Status Indicators**
- Game monitoring status (LIVE/STOPPED)
- Queue processing status (LIVE/STOPPED)
- Thread status (Running/Inactive)
- Last activity times
- Quick toggle button

✅ **Game State Tab Controls**
- Automated queue processing toggle
- Configuration controls (poll interval, batch size)
- Manual control buttons (Start, Stop, Process Now)
- Detailed status views
- Complete metrics and statistics

✅ **Real-Time Status**
- Always-fresh status on every rerun
- Instant visibility of service state
- Activity time tracking

✅ **Easy Control**
- One-click toggle for automated queue processing
- Manual controls for precise control
- Quick access from dashboard

**How to use:**

1. **Check status:** Go to Dashboard tab, see Service Status
2. **Toggle queue processing:** Click toggle switch in Game State tab
3. **Quick toggle:** Click "Toggle Queue Processing" in Dashboard
4. **Manual control:** Use buttons in Game State tab
5. **Configure:** Adjust poll interval and batch size

---

**Files Modified:**
- `pages/04_Automation_Manager.py` - Added status indicators, toggle switch, and controls

**Functions Used:**
- `get_automation_status()` - Get game monitoring status
- `get_queue_processor_status()` - Get queue processor status
- `start_queue_processor()` - Start queue processor
- `stop_queue_processor()` - Stop queue processor
- `render_automation_status()` - Render game monitoring details
- `render_queue_processor_status()` - Render queue processor details

---

**Author:** Perry (code-puppy)  
**Created:** February 8, 2026  
**Version:** 1.0.0  

🐶 *Full visibility and control!*
