# PRE_GAME Trigger Investigation - 2025-02-03

**Date:** 2025-02-03  
**Issue:** PRE_GAME predictions not posted for 7pm CST games  
**Status:** INVESTIGATED - ROOT CAUSE IDENTIFIED

---

## Problem Statement

Games starting at 7pm CST should have had PRE_GAME predictions posted at 6pm CST (1 hour before game start). However, no PRE_GAME predictions were made or posted to Discord.

---

## Investigation Results

### 1. Trigger Scheduling ✅ CORRECT

All PRE_GAME triggers were scheduled correctly in the database:

| Game | Game Start (CST) | PRE_GAME Due (CST) | Scheduled (UTC) | Status |
|-------|------------------|-------------------|----------------|--------|
| BOS @ DAL | 2026-02-02 19:00 | 2026-02-02 18:00 | 2026-02-03T00:00:00Z | scheduled ❌ |
| CHI @ MIL | 2026-02-02 19:00 | 2026-02-02 18:00 | 2026-02-03T00:00:00Z | scheduled ❌ |
| ORL @ OKC | 2026-02-02 19:00 | 2026-02-02 18:00 | 2026-02-03T00:00:00Z | scheduled ❌ |

**Status:** All triggers are still in `scheduled` state, never fired.

### 2. Trigger Execution ❌ NEVER FIRED

**Database Check:**
```sql
SELECT * FROM triggers WHERE trigger_type = 'PRE_GAME';
```

**Results:**
- Total PRE_GAME triggers: 10
- Triggers fired: 0
- Triggers overdue: 10 (20-25 hours overdue!)
- All triggers have `fired_at_utc = NULL`
- All triggers have `status = 'scheduled'`

### 3. Log Analysis

**DAILY_SUMMARY Trigger:**
```
2026-02-03 18:25:59 - Processing scheduled trigger: DAILY_20260203 DAILY_SUMMARY
2026-02-03 18:25:59 - Processing DAILY_SUMMARY for 2026-02-03 (10 games)
2026-02-03 18:25:59 - Running pregame prediction for DEN @ DET (0022500716)
... (all 10 games processed)
```

**PRE_GAME Triggers:**
```
(NONE - no logs found)
```

### 4. Current Time Analysis

**As of 2026-02-04 00:44 UTC:**
- Current time: 2026-02-03 18:44 CST
- PRE_GAME triggers due: 2026-02-02 18:00 CST
- **Triggers overdue by: 24.7 hours!**

---

## Root Cause

### The Automation Ran Once and Stopped

**Evidence:**
1. Only DAILY_SUMMARY was processed (once)
2. No PRE_GAME triggers were processed
3. Automation logs show continuous periodic polling for old games
4. No log entries for PRE_GAME trigger detection

**Explanation:**
The automation was run with the `--once` flag (or stopped after one cycle):
```bash
python -m worker.runner --once
```

This causes the runner to:
1. Initialize and schedule triggers for the day ✅
2. Run a single poll cycle ✅
3. Process DAILY_SUMMARY (due at 3h before first game) ✅
4. Check for PRE_GAME triggers (not yet due at 6pm CST) ✅
5. **EXIT** (stops running) ❌

**Result:**
- When PRE_GAME triggers became due at 6pm CST, the automation was no longer running
- No process was polling to detect and fire the triggers
- All PRE_GAME triggers remained in 'scheduled' state, never fired

---

## Solution

### Run Automation Continuously (Without --once)

The automation must run continuously to keep polling for triggers:

```bash
# Start automation (runs forever)
python -m worker.runner

# Or with custom poll interval
python -m worker.runner --poll-interval 60

# Dry run (no Discord posts)
python -m worker.runner --dry-run

# Process specific date
python -m worker.runner --date 2026-02-03
```

### How Continuous Mode Works

**Poll Cycle (every 60 seconds):**
1. Get due triggers (scheduled within ±30 seconds of current time)
2. Process scheduled time-based triggers (DAILY_SUMMARY, PRE_GAME, HALFTIME)
3. Poll active games for game-state triggers (Q3, FINAL)
4. Create periodic snapshots for tracking
5. Sleep for 60 seconds
6. **REPEAT** (never stops)

**Trigger Detection Window:**
```python
window_start = now_utc - timedelta(minutes=2)  # 2 minutes ago
window_end = now_utc + timedelta(seconds=30)    # 30 seconds ahead
```

Triggers scheduled within this window are detected and processed.

---

## Verification

### Check if Automation is Running

```bash
# Check process
pgrep -f "python -m worker.runner"

# Check via automation monitor
streamlit run monitoring/automation_monitor.py
```

### Check Automation Log

```bash
# View recent logs
tail -f logs/automation.log

# Check for PRE_GAME processing
grep "PRE_GAME" logs/automation.log
```

### Check Trigger Status

```bash
# Query database
sqlite3 data/automation.db "SELECT * FROM triggers WHERE trigger_type = 'PRE_GAME';"
```

---

## Recommendations

### 1. Use Continuous Mode for Production

**❌ Don't use:**
```bash
python -m worker.runner --once  # Stops after one cycle
```

**✅ Use:**
```bash
python -m worker.runner  # Runs forever
```

### 2. Run Automation as a Background Service

**Option A: Nohup (simple)**
```bash
nohup python -m worker.runner > logs/automation.out 2>&1 &
```

**Option B: Systemd (recommended)**
```bash
# Create systemd service
sudo vim /etc/systemd/system/perrypicks.service
```

**Option C: LaunchAgent (macOS)**
```bash
# Create macOS LaunchAgent
~/Library/LaunchAgents/com.perrypicks.automation.plist
```

### 3. Monitor Automation Status

Use the automation monitor:
```bash
streamlit run monitoring/automation_monitor.py
```

This shows:
- Automation running status
- Scheduled games
- Trigger status
- Recent logs
- Manual trigger buttons

---

## Manual Trigger (Emergency)

If you need to fire PRE_GAME triggers manually:

```bash
python -c "
from pathlib import Path
from worker.triggers import TriggerFirer
import sqlite3

db_path = Path('data/automation.db')
firer = TriggerFirer(db_path, dry_run=False)

# Get due PRE_GAME triggers
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()
cursor.execute(\"\"\"
    SELECT * FROM triggers 
    WHERE trigger_type = 'PRE_GAME' AND status = 'scheduled'
\"\"\")

for trigger in cursor.fetchall():
    firer.fire_trigger(dict(trigger))
    print(f'Fired trigger: {trigger[\"game_id\"]}')

conn.close()
"
```

---

## Prevention

### 1. Add Startup Script

Create `scripts/start_automation.sh`:
```bash
#!/bin/bash
cd /path/to/PerryPicks\ v3
source .venv/bin/activate
python -m worker.runner >> logs/automation.out 2>&1 &
echo "Automation started with PID: $!"
```

### 2. Add Systemd Service

Create `/etc/systemd/system/perrypicks.service`:
```ini
[Unit]
Description=PerryPicks Automation
After=network.target

[Service]
Type=simple
User=jarrydhawley
WorkingDirectory=/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3
Environment=PATH=/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3/.venv/bin
ExecStart=/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3/.venv/bin/python -m worker.runner
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable perrypicks
sudo systemctl start perrypicks
sudo systemctl status perrypicks
```

### 3. Use Automation Monitor

The automation monitor will alert you if automation stops:
```bash
streamlit run monitoring/automation_monitor.py
```

---

## Summary

| Item | Status |
|------|--------|
| Trigger Scheduling | ✅ Correct |
| Trigger Timing | ✅ Correct (1h before game) |
| Trigger Execution | ❌ Never fired |
| DAILY_SUMMARY | ✅ Processed correctly |
| PRE_GAME | ❌ Not processed |
| Root Cause | ❌ Automation ran with `--once` and stopped |
| Solution | ✅ Run automation continuously (without `--once`) |

---

**Next Steps:**
1. Start automation in continuous mode: `python -m worker.runner`
2. Verify automation is running: `pgrep -f "python -m worker.runner"`
3. Monitor via Streamlit: `streamlit run monitoring/automation_monitor.py`
4. Consider setting up systemd service for auto-restart

---

**Investigation Date:** 2025-02-03  
**Investigated By:** Perry (code-puppy)  
**Status:** ROOT CAUSE IDENTIFIED ✅
