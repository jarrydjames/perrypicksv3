# Game State Monitoring System

Complete hands-off automation for live NBA game monitoring and prediction posting.

## 🎯 Overview

The Game State Monitoring System runs as a background service that:

1. **Monitors live NBA games** in real-time (every 30 seconds)
2. **Detects trigger conditions** (halftime, Q3-5min)
3. **Generates predictions** automatically when conditions are met
4. **Processes queue** and posts to Discord automatically
5. **Zero manual intervention** required

## 🚀 Quick Start

### macOS (Recommended)

**Option 1: Double-click (Easiest)**
```bash
# Just double-click this file:
scripts/start_game_state_monitor.command
```

**Option 2: Terminal**
```bash
bash scripts/start_game_state_monitor.sh
```

### Cross-Platform (Python)
```bash
python scripts/start_game_state_monitor.py
```

## ⚙️ Configuration

### Environment Variables

```bash
# Poll interval in seconds (default: 30)
export GAME_STATE_POLL_INTERVAL=30

# Platforms to post to (default: all enabled)
export GAME_STATE_PLATFORMS=discord,twitter,bluesky

# Dry run mode - generate predictions but don't post (default: false)
export GAME_STATE_DRY_RUN=false
```

### Command-Line Options

```bash
# Run with custom poll interval
python scripts/start_game_state_monitor.py --interval 60

# Run in dry run mode (testing)
python scripts/start_game_state_monitor.py --dry-run

# Specify platforms
python scripts/start_game_state_monitor.py --platforms discord
```

## 🎮 How It Works

### Monitoring Loop

The service runs continuously and:

1. **Polls NBA API** for current game states
2. **Tracks period and time** for all active games
3. **Evaluates triggers** against game state
4. **Fires predictions** when conditions met
5. **Auto-processes queue** to post predictions
6. **Repeats** every 30 seconds

### Trigger Conditions

| Trigger | Condition | Action |
|---------|-----------|--------|
| **Halftime** | `period=2` AND `time_remaining=0:00` | Generate halftime prediction & post |
| **Q3-5min** | `period=3` AND `time_remaining≈5:00` | Generate Q3 prediction & post |

### Game State Tracking

The system tracks for each game:
- **Status**: scheduled, live, halftime, finished
- **Period**: Current quarter (1-4)
- **Time Remaining**: e.g., "5:32", "0:00"
- **Scores**: Home and away team scores
- **Last Updated**: Timestamp of last update

## 🔧 Architecture

```
┌─────────────────────────────────────────────────────────┐
│         GameStateService (Main)                         │
│  - Orchestrates all components                         │
│  - Runs monitoring loop                                  │
│  - Handles graceful shutdown                             │
└─────────────────────────────────────────────────────────┘
                       ↓
        ┌──────────────┴──────────────┐
        ↓                             ↓
┌─────────────────┐         ┌─────────────────┐
│ GameStateMonitor│         │  TriggerEngine  │
│                 │         │                 │
│ • Polls NBA API │         │ • Halftime Rule │
│ • Updates state │         │ • Q3-5min Rule  │
│ • Tracks games  │         │ • Prevent dupes  │
└─────────────────┘         └─────────────────┘
        ↓                             ↓
        └──────────────┬──────────────┘
                       ↓
┌──────────────────────────────────────┐
│   AutoQueueProcessor                │
│                                      │
│ • Queue predictions                 │
│ • Auto-process queue                │
│ • Post to Discord/Twitter/Bluesky   │
└──────────────────────────────────────┘
```

## 📊 Features

### ✅ Core Features

- **Real-time monitoring**: Polls NBA API every 30 seconds
- **Precise timing**: Posts at exact game moments
- **Duplicate prevention**: Won't post same trigger twice
- **Auto queue processing**: No manual intervention needed
- **Error handling**: Retries on API failures
- **Comprehensive logging**: Full audit trail

### ✅ Safety Features

- **Duplicate trigger tracking**: Prevents redundant posts
- **Error recovery**: Exponential backoff on failures
- **Graceful shutdown**: Handles SIGINT/SIGTERM properly
- **Dry run mode**: Test without posting

## 🖥️ UI Integration

### Automation Manager Tab

1. Navigate to **Automation Manager** page
2. Click **Game State** tab
3. **Start Service** button launches the monitoring service
4. **Configuration** options for poll interval and dry run
5. **Instructions** for manual startup

## 📝 Logs

### Log Files

Logs are saved to `logs/game_state_monitor_YYYYMMDD_HHMMSS.log`

```bash
# View latest log
tail -f logs/game_state_monitor_*.log

# Search for errors
grep ERROR logs/game_state_monitor_*.log
```

### Log Levels

- **INFO**: Normal operations, game state updates
- **WARNING**: Non-fatal issues, retries
- **ERROR**: Failed operations, API errors

## 🔍 Troubleshooting

### Service Won't Start

**Problem**: Virtual environment not found
```
❌ Error: .venv not found
```
**Solution**: Ensure virtual environment exists and activate it
```bash
cd "/Users/jarrydhawley/Desktop/Predictor/PerryPicks v3"
source .venv/bin/activate
```

### No Games Found

**Problem**: "No games found for today"
```
⚠️ No games found for today
```
**Solution**: Check if there are NBA games today, or verify internet connection

### Trigger Not Firing

**Problem**: Expected trigger didn't fire
**Solution**:
- Check logs for warnings/errors
- Verify game time is correct
- Check if trigger already fired (duplicate prevention)

### Posts Not Appearing

**Problem**: Predictions generated but not posting
**Solution**:
- Check if in dry run mode
- Verify Discord webhook is configured
- Check logs for post errors

## 🧪 Testing

### Dry Run Mode

Run without actually posting to test:
```bash
python scripts/start_game_state_monitor.py --dry-run
```

### Manual Testing

1. Start in dry run mode
2. Monitor logs for game state updates
3. Verify triggers fire when conditions met
4. Check predictions are generated
5. Verify queue processing works

## 📚 API Reference

### GameStateMonitor

```python
from src.automation import GameStateMonitor

monitor = GameStateMonitor(poll_interval_seconds=30)
monitor.start()  # Runs in loop
monitor.stop()   # Stops monitoring
```

### TriggerEngine

```python
from src.automation import TriggerEngine

engine = TriggerEngine(
    game_state_monitor=monitor,
    queue_processor=processor,
)

fired_events = engine.evaluate_all(platforms=["discord"])
```

### GameStateService

```python
from src.automation import GameStateService

service = GameStateService(
    poll_interval_seconds=30,
    platforms=["discord"],
    dry_run=False,
)
service.start()
```

## 🎓 Best Practices

### Production Deployment

1. **Start before games begin**: Ensure service is running
2. **Monitor logs**: Check periodically for errors
3. **Set up log rotation**: Prevent log files growing too large
4. **Test dry run first**: Verify configuration before live use
5. **Keep service running**: Don't stop during active games

### Performance Tuning

- **Poll interval**: 30s default, increase to 60s if API rate limited
- **Max retries**: 3 default, increase if connection unstable
- **Dry run**: Use for testing to avoid unnecessary posts

## 🔗 Related Components

- **AutomationManager**: UI for controlling automation
- **SocialMediaManager**: Handles posting to platforms
- **PostQueue**: Manages pending posts
- **PredictionAPI**: Generates predictions

## 📝 Example Workflow

### Typical Game Day

```
1. Morning (Pre-Game):
   - Start Game State Monitor
   - Service begins monitoring
   - No games live yet

2. During Games (Live):
   - Service polls every 30s
   - Tracks period and time
   - Logs game state updates

3. Halftime Triggers:
   - Game reaches end of Q2
   - Service detects period=2, time=0:00
   - Generates halftime prediction
   - Auto-processes queue
   - Posts to Discord
   - Marks trigger as fired

4. Q3 Triggers:
   - Game reaches 5 min left in Q3
   - Service detects period=3, time≈5:00
   - Generates Q3 prediction
   - Auto-processes queue
   - Posts to Discord
   - Marks trigger as fired

5. After Games:
   - All games finished
   - Service continues monitoring
   - Ready for next day
```

## 🆘 Support

For issues or questions:

1. Check logs for errors
2. Review this documentation
3. Test in dry run mode
4. Verify configuration

## 📄 License

Part of PerryPicks v3 - See project license