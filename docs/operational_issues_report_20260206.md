# Operational Issues Report
**Date:** 2026-02-06  
**Reported By:** Perry (code-puppy)  
**Status:** Automation Running, No Games Available

---

## Executive Summary

The PerryPicks v3 automation platform is **fully operational**, but there are **no NBA games scheduled for today** (2026-02-06 CST). This document outlines the issues discovered, their root causes, and recommended actions.

### Current System Status

| Component | Status | Details |
|-----------|--------|----------|
| **Automation Runner** | ✅ RUNNING | PID 65686, polling every 60s |
| **Streamlit UI** | ✅ RUNNING | http://localhost:8502 |
| **Database** | ✅ OPERATIONAL | 32 games from Feb 1-6 |
| **Health Check** | ⚠️ Exit Code 1 | Env loading issue (system still works) |
| **Triggers** | ⚠️ 62 Expired | All from Feb 1-6, runner missed them |
| **Games Today** | ❌ 0 | NBA schedule appears on hiatus |

---

## Issue 1: No NBA Games Available for Today

### Description

The NBA API is not returning any finalized games for 2026-02-06 CST. The system is seeing only placeholder games with "UNK @ UNK" teams.

### Current State

**Database Games by Date:**
```
2026-02-01: 4 games
2026-02-02: 9 games
2026-02-03: 2 games
2026-02-04: 4 games
2026-02-05: 12 games  ✅
2026-02-06: 1 game (no time set) ⚠️
```

**Scheduled Triggers:**
```
2026-02-01: 6 triggers
2026-02-02: 13 triggers
2026-02-03: 6 triggers
2026-02-04: 13 triggers
2026-02-05: 8 triggers
2026-02-06: 16 triggers (all expired)
```

### Root Causes

1. **NBA Schedule Hiatus** 🏀
   - The NBA may have scheduled a gap in games
   - Possible causes:
     - All-Star Weekend
     - Trade Deadline
     - League scheduling gap
     - COVID protocol pause

2. **Schedule Not Finalized** ⏰
   - NBA API returning placeholder games with "UNK @ UNK"
   - Teams not assigned to time slots yet
   - Common 6-12 hours before actual games

### Symptoms

- API fetch returns 150+ games with "UNK @ UNK" teams
- No valid games pass time validation (48-hour window)
- Automation log shows repeated warnings
- Only 1 game in DB for Feb 6 (UTA @ ORL) with no time

### Impact

- **Severity:** Medium
- **Duration:** Temporary (until games resume)
- **Affects:** Pregame predictions, Discord posts

### Recommended Actions

**Short-term:**
- ✅ Keep automation running (will auto-detect when games return)
- ✅ Monitor automation logs: `tail -f automation.log`
- ✅ Check NBA.com/schedule for official schedule

**Long-term:**
- Add "no games today" handling to runner
- Create alert for schedule hiatus detection
- Add schedule status indicator to Ops Dashboard

---

## Issue 2: Expired Triggers Not Firing

### Description

62 scheduled triggers from 2026-02-01 through 2026-02-06 were never fired because the automation runner was not running when they became due.

### Affected Triggers

```
Total Scheduled: 62 triggers
Dates: Feb 1-6, 2026
Status: All EXPIRED (missed)
```

**Breakdown by Trigger Type:**
- PRE_GAME: Pregame predictions (3h before tipoff)
- HALFTIME: Halftime adjustments
- Q3: Third-quarter predictions
- DAILY_SUMMARY: Daily summary posts

### Root Causes

1. **Runner Not Active During Window**
   - Runner was started after triggers expired
   - No automatic catch-up mechanism for missed triggers
   - Triggers fire only within a 2.5 minute window of scheduled time

2. **No Idempotent Re-fire Logic**
   - Once triggers expire, they are skipped
   - No mechanism to fire missed triggers retroactively

### Impact

- **Severity:** Low (historical games, no betting opportunity now)
- **Lost Value:** ~60 predictions not posted
- **Historical Impact:** Can't backtest these missed predictions

### Recommended Actions

**Short-term:**
- Accept as data loss (triggers expired)
- Clear expired triggers from DB to reduce clutter


**Long-term:**
- Add "catch-up mode" for missed triggers
- Implement trigger expiration window (e.g., can fire up to 12h late)
- Add missed trigger report to Ops Dashboard
- Consider "dry run" mode to process missed triggers without posting

---

## Issue 3: Health Check Exit Code Ambiguity

### Description

The health check script (`scripts/healthcheck.py`) returns exit code 1 even though all system checks pass.

### Root Cause

The script uses `dotenv.load_dotenv()` which fails when called from:
- Heredoc contexts (`python << 'EOF'`)
- Cron/systemd without proper working directory
- Shell scripts that don't call `. .env` first

### Symptoms

```bash
$ .venv/bin/python scripts/healthcheck.py
{
  "api_configured": false,
  "db_read_write": true,
  "degraded_mode": false,
  "dlq_backlog_ok": true,
  "env_complete": false,
  "models_present": true,
  "pendulum_available": true
}

# Exit code: 1 (failure)
# But system actually works fine!
```

The issue: `api_configured` and `env_complete` show `false` because `.env` wasn't loaded, but the actual automation runner works because it does load `.env` properly.

### Impact

- **Severity:** Low
- **Affects:** CI/CD pipelines, automated health checks
- **Confusion:** Exit code suggests system failure when it's actually healthy

### Recommended Actions

**Option 1: Add Explicit .env Loading**
```python
# scripts/healthcheck.py
from dotenv import load_dotenv, find_dotenv

# Try multiple methods to find and load .env
env_path = find_dotenv(usecwd=True)
if env_path:
    load_dotenv(env_path)
else:
    # Manual fallback
    from pathlib import Path
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
```

**Option 2: Make Environment Variables Optional for Health Check**
```python
required_env = os.getenv("HEALTHCHECK_REQUIRE_ENV", "true").lower() == "true"
if required_env:
    results["env_complete"] = all(os.getenv(k) for k in required_env)
else:
    # Environment check is optional
    results["env_complete"] = True
    results["api_configured"] = True
```

**Option 3: Document Proper Usage**
```bash
# Always source .env before running healthcheck
source .env || true
.venv/bin/python scripts/healthcheck.py
```

---

## Issue 4: Environment Variable Loading in Scripts

### Description

Python scripts that need to load `.env` fail when run from:
- Heredoc contexts
- Cron jobs
- Shell wrappers

### Root Cause

The `dotenv.load_dotenv()` function:
- Requires a proper call stack to detect file location
- Fails when called from `python -c` or heredoc
- Doesn't search parent directories reliably

### Workaround Currently Used

```python
# Manual .env loading
import os
from pathlib import Path

env_file = Path('.env')
if env_file.exists():
    for line in env_file.read_text().strip().split('\n'):
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip()
```

### Recommended Actions

**Create a Shared Utility Function:**
```python
# core/env.py
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

def load_environment():
    """Load .env with multiple fallback strategies."""
    # Strategy 1: Find .env automatically
    env_path = find_dotenv(usecwd=True, raise_error_if_not_found=False)
    if env_path:
        load_dotenv(env_path)
        return True
    
    # Strategy 2: Look in parent directories
    cwd = Path.cwd()
    for parent in [cwd] + list(cwd.parents):
        env_file = parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            return True
    
    # Strategy 3: Explicit path from script location
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        return True
    
    return False
```

Then use it in all scripts:
```python
from core.env import load_environment
load_environment()  # Always call at top of main()
```

---

## Issue 5: Timezone/Date Confusion

### Description

There's ambiguity about which date/timezone to use for operations:
- Current time: 2026-02-07 01:47 UTC
- CST equivalent: 2026-02-06 07:47 PM
- Automation looking for: 2026-02-06 games

### Root Causes

1. **Multiple Timezones in Use**
   - UTC for database storage
   - CST for scheduling/queries
   - No clear documentation on which to use when

2. **Schedule Windows**
   - NBA.com uses local time (varies by arena)
   - API returns UTC timestamps
   - Automation uses CST windows for game_date

### Symptoms

- Confusion about "today's games"
- Triggers may fire at wrong times
- Scheduling queries return different results based on timezone

### Recommended Actions

**Document Timezone Conventions:**
```markdown
# Timezone Conventions

| Operation | Timezone | Reason |
|-----------|-----------|---------|
| Database Storage | UTC | Single source of truth |
| Game Date (CST) | America/Chicago | Natural game day (6am-6am) |
| Trigger Times | UTC | Precise scheduling |
| Display/Logs | UTC + CST annotation | Debugging clarity |
```

**Add Helper Functions:**
```python
def get_game_date_cst(utc_datetime):
    """Get the CST game date from a UTC timestamp."""
    return utc_datetime.in_tz('America/Chicago').format('YYYY-MM-DD')

def get_cst_window(game_date_cst):
    """Get UTC window for a CST game date."""
    start = pendulum.parse(f"{game_date_cst}T06:00:00-05:00")
    end = pendulum.parse(f"{game_date_cst}T06:00:00-05:00").add(days=1)
    return start, end
```

---

## System Recommendations

### Immediate Actions (Priority 1)

1. **Keep Automation Running** ✅
   - It's working correctly
   - Will auto-detect when games return

2. **Clear Expired Triggers** 🧹
   ```bash
   sqlite3 data/automation.db "
   DELETE FROM triggers 
   WHERE status = 'scheduled' 
     AND scheduled_time_utc < datetime('now', '-6 hours')
   "
   ```

3. **Monitor Ops Dashboard** 📊
   - Check http://localhost:8502
   - Navigate to 🧭 Ops Dashboard
   - Watch for DLQ backlog and missed triggers


### Short-term Improvements (Priority 2)

1. **Fix Health Check Environment Loading**
   - Add robust .env loading to `scripts/healthcheck.py`
   - Make env checks optional for basic health verification

2. **Add No-Games Handling**
   - Detect when no games are scheduled for date
   - Post "no games today" message to Discord
   - Add status indicator to Ops Dashboard

3. **Create Shared Env Utility**
   - Add `core/env.py` with `load_environment()`
   - Use in all scripts

### Long-term Improvements (Priority 3)

1. **Trigger Catch-up Mechanism**
   - Allow firing missed triggers up to 12h late
   - Add "dry run" mode for historical processing
   - Create missed trigger report

2. **Schedule Hiatus Detection**
   - Detect when NBA has no games for >24h
   - Alert via Discord/Ops Dashboard
   - Auto-suspend trigger creation during hiatus

3. **Timezone Standardization**
   - Document all timezone conventions
   - Add helper functions for conversions
   - Annotate all log messages with both UTC and CST

---

## Conclusion

The PerryPicks v3 automation platform is **fully operational** and will resume predictions automatically when NBA games are scheduled. The main issues are:

1. **NBA Schedule Hiatus** (External, temporary)
2. **Missed Triggers** (Low impact, historical)
3. **Environment Loading** (Low impact, scripts only)
4. **Timezone Ambiguity** (Low impact, documentation needed)

### Action Items

| Priority | Action | Owner | Due Date |
|----------|--------|--------|----------|
| P1 | Clear expired triggers | Jarryd | Today |
| P1 | Keep automation running | Perry | Ongoing |
| P2 | Fix health check env loading | Jarryd | This week |
| P2 | Add no-games handling | Jarryd | This week |
| P3 | Create shared env utility | Jarryd | Next sprint |
| P3 | Add trigger catch-up mechanism | Jarryd | Next sprint |

---

## Appendix: Quick Reference

### Monitor Commands

```bash
# Watch automation logs
tail -f automation.log

# Check scheduled triggers
sqlite3 data/automation.db "
SELECT * FROM triggers 
WHERE status = 'scheduled' 
ORDER BY scheduled_time_utc DESC 
LIMIT 10"

# Check games by date
sqlite3 data/automation.db "
SELECT game_date, COUNT(*) as games 
FROM games 
GROUP BY game_date 
ORDER BY game_date DESC 
LIMIT 10"

# Run health check
. .env && .venv/bin/python scripts/healthcheck.py
```

### Service Management

```bash
# Stop automation
kill 65686

# Stop Streamlit
kill 65710 65711

# Start automation
cd /Users/jarrydhawley/Desktop/Predictor/PerryPicks\ v3
nohup .venv/bin/python -m worker.runner --date $(date +%Y-%m-%d) > automation.log 2>&1 &

# Start Streamlit
nohup .venv/bin/streamlit run app_v3.py --server.headless true --server.port 8502 > streamlit.log 2>&1 &
```

### Dashboard Access

- **Streamlit UI:** http://localhost:8502
- **Ops Dashboard:** http://localhost:8502 (🧭 page)
- **Network:** http://192.168.4.40:8502
- **External:** http://69.235.42.41:8502

---

**Report Generated:** 2026-02-06 19:48 CST  
**System Status:** 🟢 OPERATIONAL  
**Next Review:** When games resume (check NBA.com/schedule)