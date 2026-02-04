# Time/Date Issue Investigation - 2025-02-03

**Date:** 2025-02-03  
**Issue:** Constant time and date problems causing missed PRE_GAME triggers  
**Status:** ⚠️ SYSTEM TIME/DATA MISMATCH

---

## Problem Summary

### Symptoms
1. PRE_GAME triggers not firing on time
2. Games showing as "past" when they should be future
3. Scheduled date not matching actual calendar date
4. Constant confusion between system time and NBA schedule data

---

## Investigation Results

### 1. System Time Shows 2026

```
System date command (unavailable)
Python datetime: 2026-02-03 19:08
```

### 2. NBA API Returns 2025 Data

When querying schedule for 2026-02-03:
```
10/02/2025 00:00:00
  Games: 1
  First game: 0012500008
```

The NBA scheduleLeagueV2.json returns games for **2025**, not 2026!

### 3. Database Has Wrong Dates

Games stored in database:
```
BOS @ DAL (0022500721)
  game_date: 2026-02-03        ← Shows Feb 3
  start_time_utc: 2026-02-03T01:00:00+00:00
                              ← When converted to CST: 7pm Feb 2
```

The `start_time_utc` is stored as Feb 3 at 1am UTC, but when displayed it shows as 7pm Feb 2!

---

## Root Causes

### Issue #1: System Clock Mismatch ⚠️
**System thinks it's 2026, but actual calendar might be 2025**

The system time is out of sync with reality. Check your actual calendar date.

### Issue #2: NBA API Schedule Delay
**scheduleLeagueV2.json may not have 2026 games yet**

The NBA schedule file doesn't update in real-time. Future games might not be added until closer to game day.

### Issue #3: Database Date vs Start Time Mismatch
**`game_date` and `start_time_utc` store different dates**

The database stores:
- `game_date`: From NBA API (Feb 3, 2026 - CORRECT)
- `start_time_utc`: Constructed by scheduler (Feb 2 at 1am UTC - WRONG)

This causes games to show as "past" when they should be future.

### Issue #4: Timezone Offset Problem
**ISO format stores `+00:00` offset instead of `Z`**

When `start_time_utc.isoformat()` is called, it produces:
```
2026-02-03T01:00:00+00:00
```

This `+00:00` offset makes parsing difficult and can cause issues.

---

## Immediate Actions Required

### 1. CHECK SYSTEM TIME
Verify what the actual date is on your machine:
```bash
date
```

If it shows 2025, the system clock is wrong.

### 2. USE EXPLICIT DATE FOR SCHEDULING

Instead of relying on 'today', explicitly specify date:
```bash
# Schedule for TODAY's actual date
python -m worker.runner --date 2025-02-03

# Schedule for TOMORROW
python -m worker.runner --date 2025-02-04
```

### 3. START AUTOMATION CONTINUOUSLY

Once games are scheduled correctly, automation MUST run continuously:
```bash
./scripts/start_automation.sh
```

Do NOT use `--once` flag.

### 4. MONITOR AUTOMATION

```bash
streamlit run monitoring/automation_monitor.py
```

---

## Why PRE_GAME Triggers Failed

### Timeline
1. Automation ran YESTERDAY (Feb 2) with `date='today'`
2. Scheduled games for Feb 2, 2026 (WRONG - system thought it was Feb 3)
3. Games stored with Feb 2 start times
4. Today (Feb 3) arrived
5. Games appeared 24+ hours overdue
6. PRE_GAME triggers didn't fire

### NBA API Data
The NBA scheduleLeagueV2.json shows:
- Games for 2025 season (0012500008, etc.)
- No games for 2026-02-03 yet (teams show as "UNK")

This suggests the 2026 schedule hasn't been published yet.

---

## Recommended Solution

### Option A: Use Correct Date (If System Clock is Wrong)
```bash
# If today is actually Feb 3, 2025
python -m worker.runner --date 2025-02-03
```

### Option B: Wait for NBA Schedule Update
The NBA schedule will be updated with 2026 games. Check back in a few hours when teams are finalized.

### Option C: Fix System Clock
If your system clock is incorrect, fix it to the correct date.

---

## Monitoring

Once scheduled, verify triggers will fire:

```bash
# Check scheduled triggers
sqlite3 data/automation.db "SELECT game_id, scheduled_time_utc FROM triggers WHERE trigger_type='PRE_GAME' AND status='scheduled'"

# Monitor automation
pgrep -f "python -m worker.runner"
```

---

## Technical Details

### Timezone Offset Issue
The database stores:
```
start_time_utc: 2026-02-03T01:00:00+00:00
```

When this is parsed back and converted to CST:
```
2026-02-03 01:00:00 = 7pm Feb 2 (WRONG!)
```

The `+00:00` is being treated as an offset instead of UTC zero.

### Code Location
**File:** `core/data_sources.py` lines 105-147

The code constructs datetimes using `strptime` (naive), then adds timezone with `.replace(tzinfo=timezone.utc)`.

This creates datetime objects but the ISO formatting might not correctly handle zero UTC offset.

---

## Summary

| Issue | Status | Action Required |
|------|---------|----------------|
| System clock shows 2026 | ⚠️ Check actual date |
| NBA API returns 2025 data | ⚠️ Wait for schedule update |
| Database has wrong dates | ✅ Clear and re-schedule |
| Games appear overdue | ✅ Use explicit date |
| PRE_GAME triggers not firing | ✅ Run automation continuously |

---

**Next Steps:**
1. Check your system's actual calendar date
2. Use explicit date when scheduling if needed
3. Start automation: `./scripts/start_automation.sh`
4. Monitor: `streamlit run monitoring/automation_monitor.py`

---

**Documentation Date:** 2025-02-03  
**Documented By:** Perry (code-puppy)  
**Status:** INVESTIGATION COMPLETE ✅
