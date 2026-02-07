# PerryPicks V3 - Automation Summary

This document summarizes the complete automation system for all three prediction models.

---

## Files Created

| File | Purpose | Type |
|-------|-----------|-------|
| `README_MODELS.md` | Complete model documentation | Documentation |
| `CRON_SETUP.md` | Cron job and Task Scheduler setup | Guide |
| `schedule_predictions.py` | Unified prediction scheduler | Script |
| `run_automated_predictions.py` | Continuous game monitoring | Script |

---

## Quick Start

### Run All Models (Manual)

```bash
# Today's games
python schedule_predictions.py

# Specific date
python schedule_predictions.py --date 2026-02-05

# Specific games (testing)
python schedule_predictions.py --games 0022500733 0022500734
```

### Run Specific Models

```bash
# Pregame only
python schedule_predictions.py --models pregame

# Halftime only
python schedule_predictions.py --models halftime

# Q3 only
python schedule_predictions.py --models q3
```

### Dry Run (Preview Commands)

```bash
python schedule_predictions.py --dry-run
```

---

## Cron Job Setup

### Basic Daily Schedule

```cron
# Run pregame at 6:00 PM (before games)
0 18 * * * cd /path/to/PerryPicks v3 && /usr/local/bin/uv run python schedule_predictions.py --models pregame >> logs/pregame.log 2>&1

# Check halftime every 5 minutes (7 PM - 11 PM)
*/5 19-23 * * * cd /path/to/PerryPicks v3 && /usr/local/bin/uv run python schedule_predictions.py --models halftime >> logs/halftime.log 2>&1

# Check Q3 every 5 minutes (8 PM - 11 PM)
*/5 20-23 * * * cd /path/to/PerryPicks v3 && /usr/local/bin/uv run python schedule_predictions.py --models q3 >> logs/q3.log 2>&1
```

### Setup Steps

```bash
# 1. Edit crontab
crontab -e

# 2. Add cron jobs (see examples above)

# 3. Save and exit

# 4. Create logs directory
mkdir -p logs
chmod 755 logs

# 5. Monitor logs
tail -f logs/pregame.log
tail -f logs/halftime.log
tail -f logs/q3.log
```

---

## Model Reference

| Model | Script | Champion | Target | When to Run |
|-------|---------|-----------|--------|-------------|
| **Pregame** | `run_pregame_predictions.py` | Neural Network | Final game (~225 pts) | 1-2 hours before tipoff |
| **Halftime** | `run_halftime_predictions.py` | XGBoost | Final game from H1 (~220 pts) | At halftime (end of Q2) |
| **Q3** | `run_q3_predictions.py` | Neural Network | Final game from Q3 (~195-257 pts) | After Q3 (end of Q3) |

---

## Output Examples

### Pregame Output

```
Game ID      | Away   @ Home   | Predicted Total | Predicted Margin | Winner  
----------------------------------------------------------------------------------------------------
0022500733   | WAS    @ DET    | 223.8           | -3.4             | WAS     
0022500734   | BKN    @ ORL    | 215.6           | +12.1            | ORL     
```

### Halftime Output

```
Game ID      | Away   @ Home   | H1         | Pred 2H     | Pred Final      | Margin   | Winner  
----------------------------------------------------------------------------------------------------
0022500733   | WAS    @ DET    | 56-52      | 60.8-54.0   | 116.8-106.0     | -6.8     | WAS     
0022500734   | BKN    @ ORL    | 40-56      | 45.8-64.3   | 85.8-120.3      | +18.5    | ORL     
```

### Q3 Output

```
Game ID      | Away   @ Home   | Q3 Cum       | Est Q4        | Pred Final         | Margin   | Winner  
----------------------------------------------------------------------------------------------------
0022500733   | WAS    @ DET    | 95.0-84.0    | 30.8-26.4     | 125.8-110.4        | -15.4    | WAS     
0022500734   | BKN    @ ORL    | 67.0-88.0    | 20.6-29.0     | 87.6-117.0         | +29.4    | ORL     
```

---

## Troubleshooting

### Cron Job Not Running

```bash
# Check cron service
sudo service cron status  # Linux
brew services list           # macOS

# Check cron logs
sudo grep CRON /var/log/syslog  # Linux
log show --predicate 'process == "cron"' --last 1h  # macOS
```

### Script Fails with Import Error

```bash
# Use uv to run
uv run python schedule_predictions.py

# Or activate virtual environment
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate      # Windows
python schedule_predictions.py
```

### API Rate Limit

- Scripts automatically add 1-second delays between game requests
- Schedule adds 10-second delays between models
- If rate-limited, wait 5-10 minutes and retry

---

## Best Practices

1. **Use full paths in cron** - Don't rely on $PATH
2. **Log everything** - Redirect output to log files with `>> logs/model.log 2>&1`
3. **Test manually first** - Run scripts manually before adding to cron
4. **Monitor logs** - Check logs daily for errors
5. **Backup predictions** - Save output to database or CSV for analysis

---

## Documentation Links

- **Complete Model Guide**: `README_MODELS.md`
- **Cron Setup Guide**: `CRON_SETUP.md`
- **Individual Scripts**: 
  - `run_pregame_predictions.py`
  - `run_halftime_predictions.py`
  - `run_q3_predictions.py`
  - `schedule_predictions.py`
  - `run_automated_predictions.py`

---

## Support

For issues or questions:
1. Check logs: `tail -f logs/*.log`
2. Run manual test: `python schedule_predictions.py --dry-run`
3. Review documentation: `README_MODELS.md`, `CRON_SETUP.md`
4. Open issue on GitHub

---

Last Updated: 2026-02-07
Version: 1.0
