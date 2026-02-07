# PerryPicks V3 - Cron Job Setup Guide

This guide explains how to set up automated prediction runs using cron jobs (Linux/macOS) or Task Scheduler (Windows).

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Cron Job Examples](#cron-job-examples)
4. [Windows Task Scheduler](#windows-task-scheduler)
5. [Monitoring & Logs](#monitoring--logs)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before setting up cron jobs, ensure:

1. **Python environment is configured**
   ```bash
   # Navigate to project directory
   cd /path/to/PerryPicks v3
   
   # Verify uv is installed
   uv --version
   
   # Test script runs manually
   uv run python schedule_predictions.py --dry-run
   ```

2. **Log directory exists**
   ```bash
   mkdir -p logs
     chmod 755 logs
   ```

3. **Full path to uv is known**
   ```bash
   which uv
   # Output: /usr/local/bin/uv
   ```

---

## Quick Start

### For Linux/macOS (Cron)

```bash
# 1. Open crontab
crontab -e

# 2. Add cron jobs (see examples below)

# 3. Save and exit

# 4. Verify cron is running
sudo service cron status  # Linux
brew services list           # macOS
```

### For Windows (Task Scheduler)

See [Windows Task Scheduler](#windows-task-scheduler) section below.

---

## Cron Job Examples

### Daily Pregame Predictions

Run pregame predictions at 6:00 PM ET (for 7:30 PM games):

```cron
# 0 18 * * * = At 6:00 PM every day
0 18 * * * cd /path/to/PerryPicks v3 && /usr/local/bin/uv run python schedule_predictions.py --models pregame >> logs/pregame.log 2>&1
```

### Continuous Halftime Monitoring

Check for halftime games every 5 minutes from 7:00 PM to 11:00 PM:

```cron
# */5 19-23 * * * = Every 5 minutes from 7 PM to 11 PM
*/5 19-23 * * * cd /path/to/PerryPicks v3 && /usr/local/bin/uv run python schedule_predictions.py --models halftime >> logs/halftime.log 2>&1
```

### Continuous Q3 Monitoring

Check for Q3 games every 5 minutes from 7:30 PM to 11:00 PM:

```cron
# */5 20-23 * * * = Every 5 minutes from 8 PM to 11 PM
*/5 20-23 * * * cd /path/to/PerryPicks v3 && /usr/local/bin/uv run python schedule_predictions.py --models q3 >> logs/q3.log 2>&1
```

### Full Daily Schedule

Complete automation for all three models:

```cron
# ==================================================
# PERRY PICKS V3 - AUTOMATED PREDICTIONS
# ==================================================

# Pregame predictions at 6:00 PM (before games)
0 18 * * * cd /Users/jarrydhawley/Desktop/Predictor/PerryPicks\ v3 && /usr/local/bin/uv run python schedule_predictions.py --models pregame >> logs/pregame.log 2>&1

# Halftime checks every 5 minutes (7 PM - 11 PM)
*/5 19-23 * * * cd /Users/jarrydhawley/Desktop/Predictor/PerryPicks\ v3 && /usr/local/bin/uv run python schedule_predictions.py --models halftime >> logs/halftime.log 2>&1

# Q3 checks every 5 minutes (8 PM - 11 PM)
*/5 20-23 * * * cd /Users/jarrydhawley/Desktop/Predictor/PerryPicks\ v3 && /usr/local/bin/uv run python schedule_predictions.py --models q3 >> logs/q3.log 2>&1

# ==================================================
```

### Advanced: Multiple Time Slots

Run pregame predictions at different times for different game start times:

```cron
# Early games (12:00 PM ET) - run at 11:30 AM
30 11 * * * cd /path/to/PerryPicks v3 && /usr/local/bin/uv run python schedule_predictions.py --models pregame >> logs/pregame_early.log 2>&1

# Prime time games (7:00 PM ET) - run at 6:30 PM
30 17 * * * cd /path/to/PerryPicks v3 && /usr/local/bin/uv run python schedule_predictions.py --models pregame >> logs/pregame_prime.log 2>&1
# Late games (9:30 PM ET) - run at 9:00 PM
0 21 * * * cd /path/to/PerryPicks v3 && /usr/local/bin/uv run python schedule_predictions.py --models pregame >> logs/pregame_late.log 2>&1
```

### Weekend Schedule

Run on weekends only:

```cron
# Saturday (6) and Sunday (0)
0 18 * * 6,0 cd /path/to/PerryPicks v3 && /usr/local/bin/uv run python schedule_predictions.py --models pregame >> logs/pregame.log 2>&1
```

---

## Windows Task Scheduler

### Setup Instructions

1. **Open Task Scheduler**
   - Press `Win + R`
   - Type `taskschd.msc`
   - Press Enter

2. **Create New Task**
   - Click "Create Task" in right panel
   - Name: "PerryPicks Pregame"
   - Description: "Run NBA pregame predictions daily"

3. **Triggers Tab**
   - Click "New..."
   - Begin the task: "On a schedule"
   - Daily at 6:00 PM
   - Click OK

4. **Actions Tab**
   - Click "New..."
   - Action: "Start a program"
   - Program/script: `C:\path\to\uv.exe`
   - Add arguments: `run python schedule_predictions.py --models pregame`
   - Start in: `C:\path\to\PerryPicks v3`
   - Click OK

5. **Conditions Tab (Optional)**
   - Uncheck "Start the task only if the computer is on AC power"
   - Check "Wake the computer to run this task" if needed

6. **Settings Tab**
   - Check "Allow task to be run on demand"
   - Uncheck "Stop the task if it runs longer than..."
   - Click OK

### Multiple Tasks

Create separate tasks for:
- "PerryPicks Pregame" (6:00 PM daily)
- "PerryPicks Halftime" (7:00 PM - 11:00 PM every 5 minutes)
- "PerryPicks Q3" (8:00 PM - 11:00 PM every 5 minutes)

---

## Monitoring & Logs

### Log Files

```bash
# View pregame logs
tail -f logs/pregame.log

# View halftime logs
tail -f logs/halftime.log

# View Q3 logs
tail -f logs/q3.log

# View all logs
cat logs/*.log | tail -100
```

### Log Rotation

To prevent log files from growing too large, add log rotation:

```bash
# Install logrotate (Linux)
sudo apt-get install logrotate  # Ubuntu/Debian
sudo yum install logrotate  # CentOS/RHEL

# Create logrotate config file
sudo nano /etc/logrotate.d/perrypicks
```

`/etc/logrotate.d/perrypicks`:
```
/path/to/PerryPicks v3/logs/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
```

### Email Notifications

Add email notifications to cron:

```cron
MAILTO="your-email@example.com"

0 18 * * * cd /path/to/PerryPicks v3 && /usr/local/bin/uv run python schedule_predictions.py --models pregame >> logs/pregame.log 2>&1
```

---

## Troubleshooting

### Cron Job Not Running

**Check cron service:**
```bash
# Linux
sudo service cron status
sudo systemctl status cron

# macOS
brew services list
launchctl list
```

**Check cron logs:**
```bash
# Linux (Ubuntu/Debian)
sudo grep CRON /var/log/syslog

# Linux (CentOS/RHEL)
sudo grep CRON /var/log/cron

# macOS
log show --predicate 'process == "cron"' --last 1h
```

### Permission Denied

**Make scripts executable:**
```bash
chmod +x schedule_predictions.py
chmod +x run_pregame_predictions.py
chmod +x run_halftime_predictions.py
chmod +x run_q3_predictions.py
```

### Path Issues

**Use full paths in cron:**
```bash
# Instead of:
python schedule_predictions.py

# Use:
/usr/local/bin/uv run python /full/path/to/schedule_predictions.py
```

### Environment Variables

**If cron jobs fail due to missing environment variables:**

```bash
# Create a wrapper script
cat > /usr/local/bin/perrypicks-cron.sh << 'EOF'
#!/bin/bash
export PATH="/usr/local/bin:/usr/bin:/bin"
cd /path/to/PerryPicks v3
/usr/local/bin/uv run python schedule_predictions.py "$@" >> logs/schedule.log 2>&1
EOF

chmod +x /usr/local/bin/perrypicks-cron.sh
```

Then use wrapper in cron:
```cron
0 18 * * * /usr/local/bin/perrypicks-cron.sh --models pregame
```

---

## Testing

Before scheduling cron jobs, test manually:

```bash
# Test pregame
./schedule_predictions.py --models pregame

# Check logs
cat logs/pregame.log

# Test halftime
./schedule_predictions.py --models halftime
# Check logs
cat logs/halftime.log

# Test Q3
./schedule_predictions.py --models q3
# Check logs
cat logs/q3.log
```

---

## Quick Reference

### Cron Time Format

```
* * * * * command
│ │ │ │ │
│ │ │ │ └─ Day of week (0-6, 0=Sunday)
│ │ │ └─── Month (1-12)
│ │ └───── Day of month (1-31)
│ └─────── Hour (0-23)
└───────── Minute (0-59)
```

### Common Cron Intervals

| Cron Expression | Description |
|----------------|-------------|
| `0 18 * * *` | Daily at 6:00 PM |
| `*/5 19-23 * * *` | Every 5 minutes from 7-11 PM |
| `*/10 * * * *` | Every 10 minutes |
| `0 * * * *` | Every hour |
| `0 0 * * *` | Daily at midnight |
| `0 0 * * 1` | Weekly on Monday at midnight |
| `0 0 1 * *` | Monthly on 1st at midnight |

---

## Support

For issues or questions:
1. Check logs: `tail -f logs/*.log`
2. Run manual test: `./schedule_predictions.py --dry-run`
3. Review this guide
4. Open issue on GitHub

---

Last Updated: 2026-02-07
Version: 1.0
