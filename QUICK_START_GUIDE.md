# PerryPicks V3 - Quick Start Guide

## 🚀 Running the Automation System

### Option 1: Single Day (Original Runner)
Best for testing or specific days.

```bash
# Run for today's games
python -m worker.runner

# Run for a specific date
python -m worker.runner --date 2026-02-03

# Test run (no Discord posts)
python -m worker.runner --dry-run

# Run single cycle and exit
python -m worker.runner --once
```

### Option 2: Multi-Day (Recommended for Production) ⭐
Automatically transitions between days. No restarts needed!

```bash
# Start multi-day automation
python -m worker.unified_runner

# Test run (no Discord posts)
python -m worker.unified_runner --dry-run

# Run single cycle and exit
python -m worker.unified_runner --once
```

## 📊 Monitoring the System

### Check Database Status
```bash
sqlite3 data/automation.db "SELECT game_date, COUNT(*) FROM games GROUP BY game_date;"
```

### Check Trigger Status
```bash
sqlite3 data/automation.db "SELECT trigger_type, COUNT(*) as total FROM triggers GROUP BY trigger_type;"
```

### View Recent Logs
```bash
# Unified runner logs
tail -f logs/unified_automation.log

# Original runner logs
tail -f logs/automation.log

# All errors
grep -i error logs/*.log
```

## 🧪 Testing Predictions

```bash
# Test pregame prediction
python -c "from src.predict_api import predict_game; print(predict_game('0022500715', 'pregame'))"

# Test halftime prediction
python -c "from src.predict_api import predict_game; print(predict_game('0022500715', 'halftime'))"
```

## 🛠️ Troubleshooting

### Issue: No games scheduled
Check NBA API is working and reschedule:
```bash
python -c "from src.data.game_data import fetch_games_by_date; print(fetch_games_by_date('2026-02-03'))"
```

### Issue: Predictions failing
Check models exist:
```bash
ls -la models_v3/pregame/
ls -la models_v3/q3/
```

### Issue: Discord posts not working
Verify webhook URL:
```bash
cat .env | grep DISCORD
```

## ✅ Daily Operations Checklist

### Morning
- [ ] Check runner is running: `ps aux | grep worker`
- [ ] Review logs for errors
- [ ] Verify games scheduled for today

### During Games
- [ ] Monitor triggers firing in logs
- [ ] Watch Discord for prediction posts

### Evening
- [ ] Review prediction accuracy
- [ ] Backup database: `cp data/automation.db data/automation.db.backup`

---

**Last Updated:** 2026-02-03
**For detailed information, see:** SYSTEM_REVIEW_DOCUMENTATION.md
