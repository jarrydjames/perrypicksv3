# PerryPicks v3 - Automation Monitoring Portal

A Streamlit-based web dashboard to monitor the automation system and manually trigger predictions.

## Features

### 📊 Automation Status
- **Process Status**: Shows if automation is RUNNING or STOPPED
- **Database Status**: Confirms database connectivity
- **Last Log**: Shows when logs were last updated

### 📅 Game Monitoring
- **Today's Games**: Lists all games scheduled for the current day
- **Game Stats**: Total, Scheduled, In Progress, Completed counts
- **Game Details**: Expand each game to see:
  - Game ID
  - Status (Scheduled, In Progress, Final)
  - Current/Final score
  - Start time and date

### ⏰ Trigger Monitoring
- **Scheduled Triggers**: Shows all prediction triggers for each game
- **Live Countdown**: Real-time countdown to next prediction
- **Trigger Status**:
  - ✅ Fired - Already executed
  - ⏳ Pending - Waiting to execute
  - 📅 Scheduled - Upcoming
  - 🔴 Next - The next upcoming trigger with countdown

### 🎮 Manual Triggers
- **Pre-Game Prediction**: Manually trigger pre-game analysis
- **Halftime Prediction**: Manually trigger halftime prediction

## Running the Monitor

### Development (Local)
```bash
cd /Users/jarrydhawley/Desktop/Predictor/PerryPicks\ v3
.venv/bin/python -m streamlit run monitoring/automation_monitor.py
```

### Background Mode
```bash
cd /Users/jarrydhawley/Desktop/Predictor/PerryPicks\ v3
.venv/bin/python -m streamlit run monitoring/automation_monitor.py &
```

### Accessing the Dashboard
- **Local**: http://localhost:8502
- **Network**: http://192.168.4.40:8502 (from other devices on your network)

## Dependencies

The monitor requires:
- `streamlit` - Web UI framework
- `pytz` - Timezone support
- `sqlite3` - Database access (built-in)

Install dependencies:
```bash
.venv/bin/pip install streamlit pytz
```

## Architecture

```
monitoring/
├── automation_monitor.py  # Streamlit dashboard
└── README.md              # This file
```

The monitor connects to:
- `data/automation.db` - SQLite database with games and triggers
- `logs/automation.log` - Automation log file
- `worker/triggers.py` - Trigger firing module for manual triggers

## Usage Tips

### Checking Automation Status
1. Open the dashboard
2. Check the "Automation Status" section at the top
3. If it shows 🔴 STOPPED, the automation needs to be started

### Monitoring a Game
1. Find the game in the list
2. Click to expand the game details
3. Check the "Triggers" section for:
   - 🔴 Next prediction countdown
   - All scheduled trigger times

### Manually Triggering a Prediction
1. Expand the game you want to predict
2. Scroll to "Manual Trigger" section
3. Click either:
   - "Pre-Game Prediction" for pre-game analysis
   - "Halftime Prediction" for halftime analysis
4. The dashboard will show success/error message and refresh

### Understanding Countdowns
The next trigger countdown shows:
- `2h 15m 30s` - More than 1 hour
- `15m 30s` - Less than 1 hour
- `30s` - Less than 1 minute
- `NOW` - Trigger is due

## Troubleshooting

### Dashboard Shows "No games scheduled"
- The automation may not have detected games yet
- Check that automation is running
- Check that today is a game day

### Manual Trigger Fails
- Check that the game ID exists in the database
- Check that the database is accessible
- Check logs for detailed error messages

### Automation Shows as STOPPED
- Start the automation: `.venv/bin/python -m worker.runner &`
- Check process: `ps aux | grep "python -m worker.runner"`
- Check logs: `tail -f logs/automation.log`

## Auto-Refresh

The dashboard indicates it auto-refreshes every 30 seconds. To refresh manually:
- Click the "Rerun" button in the top-right corner
- Or press `R` in your browser

## Security

The dashboard:
- Runs locally on your machine
- Does not expose credentials
- Requires local file system access
- Cannot be accessed from outside your network

## Future Enhancements

Potential improvements:
- Historical game viewing
- Performance metrics
- Trigger scheduling UI
- Prediction accuracy tracking
- Alert configuration

## Support

For issues or questions:
1. Check automation logs: `tail -f logs/automation.log`
2. Verify database: `sqlite3 data/automation.db`
3. Check automation status: `ps aux | grep worker.runner`

---

Built with Streamlit - Data apps for Python
