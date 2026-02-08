# 🚀 PerryPicks v3 - Quick Start Guide

## One-Click Startup

**Just double-click `start_all.command`** and everything starts automatically!

---

## What Gets Started

| Service | Description | URL/Access |
|----------|-------------|------------|
| **Streamlit UI** | Frontend web interface | http://localhost:8501 |
| **Game State Monitor** | Live game monitoring & auto-posting | Background service |

---

## How to Use

### 1. Start Everything
```bash
# Just double-click this file:
start_all.command
```

### 2. Open Your Browser
```
http://localhost:8501
```

### 3. Ready to Go!
- All services running in background
- Game State Monitor monitoring NBA games
- Predictions will auto-post at halftime/Q3-5min

---

## Stopping Services

Press **Ctrl+C** in the terminal window to stop all services.

---

## Monitoring Logs

```bash
# Streamlit logs
tail -f logs/streamlit.log

# Game State Monitor logs
tail -f logs/game_state_monitor.log
```

---

## Services Overview

### Streamlit UI (http://localhost:8501)
- View predictions
- Manage automation
- Run full day automation
- Access all features

### Game State Monitor
- Polls NBA API every 30 seconds
- Detects halftime and Q3-5min
- Auto-generates predictions
- Auto-posts to Discord
- Zero manual intervention needed

---

## Troubleshooting

### Port Already in Use
```
Error: Port 8501 is already in use
```
**Solution**: Stop any running Streamlit instance or change port in `start_all.command`

### Virtual Environment Not Found
```
❌ Error: .venv not found
```
**Solution**:
```bash
cd "/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Services Not Starting
- Check logs in `logs/` directory
- Ensure Python dependencies are installed
- Verify internet connection for NBA API

---

## Tips

1. **Start early**: Run before games begin to ensure monitoring is active
2. **Check logs**: Monitor logs to verify triggers are firing correctly
3. **Keep running**: Don't stop during active games
4. **Test first**: Run in dry-run mode if testing

---

## Advanced: Start Individually

If you prefer to start services separately:

### Streamlit Only
```bash
source .venv/bin/activate
streamlit run Home_Page.py
```

### Game State Monitor Only
```bash
source .venv/bin/activate
python scripts/start_game_state_monitor.py
```

---

## Files

| File | Description |
|------|-------------|
| `start_all.command` | Main startup script (double-click this!) |
| `logs/streamlit.log` | Streamlit logs |
| `logs/game_state_monitor.log` | Game state monitor logs |

---

## Need Help?

- Check logs for errors
- Review documentation in `docs/`
- Verify all dependencies are installed

---

## 🐶 That's It!

Just double-click `start_all.command` and you're set!

Everything runs automatically - predictions will post at the right times!
