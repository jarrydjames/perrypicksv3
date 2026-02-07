# PerryPicks v3 - Social Media Automation

**Status:** COMPLETE ✅  
**Created:** February 8, 2026  
**Author:** Perry (code-puppy)

---

## 🚀 Overview

End-to-end automation system that:
- Generates predictions for NBA games
- Formats posts for Twitter/X, Bluesky, and Discord
- Posts to multiple social platforms automatically
- Detects and prevents duplicate posts
- Handles errors with retry logic
- Supports both scheduled and one-off posting

---

## 📁 File Structure

```
PerryPicks v3/
├── src/automation/              # Automation modules
│   ├── __init__.py
│   ├── twitter_client.py       # Twitter/X API client
│   ├── bluesky_client.py      # Bluesky API client
│   ├── post_generator.py       # Post formatting
│   ├── post_queue.py          # Queue + deduplication
│   ├── social_media_manager.py  # Platform orchestration
│   └── automation_orchestrator.py  # Main coordinator
├── scripts/automation/
│   └── social_poster.py     # CLI entry point
├── config/
│   └── env.example           # Configuration template
├── requirements-automation.txt   # Python dependencies
├── data/
│   └── post_queue.json     # Post queue (auto-created)
└── docs/
    └── social_media_automation.md  # This file
```

**Total:** 1,500+ lines of production-ready code

---

## 🔧 Installation

### 1. Install Dependencies

```bash
# Install automation dependencies
pip install -r requirements-automation.txt


# Or install specific packages
pip install tweepy atproto schedule pendulum
```

### 2. Configure Environment

```bash
# Copy example environment file
cp config/env.example .env

# Edit with your credentials
vim .env
```

### 3. Required Credentials

#### Discord (Required for posting)
```
DISCORD_WEBHOOK_URL=https://discordapp.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN
```

#### Twitter/X (Optional)
```
# Option 1: OAuth 1.0a (User Context)
TWITTER_CONSUMER_KEY=your_consumer_key
TWITTER_CONSUMER_SECRET=your_consumer_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret

# Option 2: OAuth 2.0 (Bearer Token)
TWITTER_BEARER_TOKEN=your_bearer_token
```

#### Bluesky (Optional)
```
BLUESKY_HANDLE=your_handle.bsky.social
BLUESKY_APP_PASSWORD=your_app_password
```

---

## 🎯 Usage

### Mode 1: Scheduled Automation (Continuous)

Run in continuous mode with automatic queue processing:

```bash
python scripts/automation/social_poster.py --schedule --poll-interval 15
```

**Options:**
- `--poll-interval 15`: Process queue every 15 minutes (default: 15)
- `--platforms twitter bluesky`: Post only to these platforms
- `--dry-run`: Simulate posting without actually posting


### Mode 2: One-Off Predictions

Run predictions for specific games:

```bash
python scripts/automation/social_poster.py --games 0022500747 0022500748 --trigger-type pregame
```

**Options:**
- `--trigger-type pregame|halftime|q3`: Prediction trigger type (default: pregame)
- `--mode auto|pregame|halftime|q3`: Prediction mode (default: auto)
- `--platforms twitter bluesky discord`: Target platforms

### Mode 3: Process Queue Only

Process pending posts from queue:

```bash
python scripts/automation/social_poster.py --process-queue
```

### Mode 4: View Statistics

Check automation status:

```bash
python scripts/automation/social_poster.py --stats
```

**Output:**
```
============================================================
AUTOMATION STATISTICS
============================================================
Processed predictions: 15
Enabled platforms: twitter, bluesky, discord

Queue stats:
  Total: 12
  Pending: 3
  Posted: 8
  Failed: 1
============================================================
```

---

## 🔒 Duplicate Detection

The system automatically prevents duplicate posts:

### How It Works
1. **Deduplication Window:** 24 hours (configurable)
2. **Check:** Before posting, check if same game/trigger/platform was posted in window
3. **Skip:** If duplicate detected, skip posting and log warning
4. **Track:** Maintain history of all posts for deduplication

### Example
```
Time 10:00 AM: Posted pregame prediction for 0022500747
Time 10:05 AM: Attempting to post pregame for 0022500747
Result: DUPLICATE DETECTED (skipped)
```

---

## 🔄 Error Handling & Retry Logic

### Automatic Retries
- **Max Retries:** 3 (configurable)
- **Backoff:** Exponential (2s, 4s, 8s delays)
- **States:** pending → posting → posted/retrying/failed

### Failure Recovery
- **Retryable Errors:** Network issues, rate limits, temporary failures
- **Permanent Errors:** Invalid credentials, deleted accounts
- **Dead Letter Queue:** Failed posts saved for manual review

---

## 📊 Post Generation

### Platform-Specific Optimization

#### Twitter/X
- **Character Limit:** 280 chars
- **Emojis:** ✅ Supported
- **Hashtags:** #NBA #NBAPredictions #PerryPicks + team tags
- **Format:** Thread support for long content


#### Bluesky
- **Character Limit:** No hard limit (300 chars recommended)
- **Emojis:** ✅ Supported
- **Hashtags:** 2-3 optimal
- **Format:** Full posts (no thread needed)

#### Discord
- **Character Limit:** No limit
- **Emojis:** ✅ Supported
- **Hashtags:** Not needed
- **Format:** Embed support

### Post Templates

#### Pregame Prediction
```
🏀 PREGAME PREDICTION
WAS @ BKN
Projected: 109.3 - 110.9
Total: 220.2 | Margin: -1.7
Winner: BKN (56.9% confidence)
Odds: Spread -1.5 | O/U 221.5

#NBA #NBAPredictions #PerryPicks
```

#### Halftime Update
```
🔥 HALFTIME UPDATE
WAS @ BKN
Halftime: WAS 52 - 58 BKN
Projected 2H: WAS 57.3 - 52.9 BKN
Projected Final: WAS 109.3 - 110.9 BKN

#NBAPredictions #PerryPicks
```

#### Q3 Update
```
⚡ Q3 UPDATE
WAS @ BKN
Q3 Cumulative: WAS 78.2 - 87.6 BKN
Estimated Q4: WAS 31.1 - 23.3 BKN
Projected Final: WAS 109.3 - 110.9 BKN

#NBAPredictions #PerryPicks
```

---

## 🧪 Integration with Existing System

### Automation Components

| Component | Status | Description |
|-----------|---------|-------------|
| **Post Generator** | ✅ COMPLETE | Formats predictions into platform-optimized posts |
| **Twitter/X Client** | ✅ COMPLETE | Twitter API v2 with OAuth support |
| **Bluesky Client** | ✅ COMPLETE | Bluesky Social API with app password auth |
| **Social Media Manager** | ✅ COMPLETE | Orchestrates posting across all platforms |
| **Post Queue** | ✅ COMPLETE | Queue management with deduplication |
| **Automation Orchestrator** | ✅ COMPLETE | Main coordinator with scheduling support |

### Integration Points

1. **Prediction Generation:** Uses `src/predict_api.predict_game()`
2. **Storage:** Uses `core.storage.GameStorage` for persistence
3. **Discord:** Uses existing `core.discord_client.DiscordWebhookClient`
4. **Environment:** Uses `core.env.load_environment()` for config

---

## 🚦 Deployment

### Cron Job Setup (Recommended)

```bash
# Edit crontab
crontab -e

# Add entry (process queue every 15 minutes)
*/15 * * * * cd /path/to/PerryPicks\ v3 && /usr/bin/env python3 scripts/automation/social_poster.py --schedule --poll-interval 15

# Or use absolute path
*/15 * * * * /usr/bin/python3 /full/path/to/PerryPicks\ v3/scripts/automation/social_poster.py --schedule --poll-interval 15
```

### systemd Service (Alternative)

Create `/etc/systemd/system/perrypicks-automation.service`:

```ini
[Unit]
Description=PerryPicks Social Media Automation
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/PerryPicks v3
EnvironmentFile=/path/to/PerryPicks v3/.env
ExecStart=/usr/bin/python3 /path/to/PerryPicks v3/scripts/automation/social_poster.py --schedule --poll-interval 15
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable perrypicks-automation
sudo systemctl start perrypicks-automation
sudo systemctl status perrypicks-automation
```

---

## 🔍 Troubleshooting

### Twitter API Errors

**Error:** `401 Unauthorized`
**Fix:** Check credentials (consumer key, secret, access tokens)

**Error:** `429 Too Many Requests`
**Fix:** System automatically handles rate limits with `wait_on_rate_limit=True`

### Bluesky API Errors

**Error:** `Authentication failed`
**Fix:** Verify handle format (`@handle.bsky.social`) and app password
**Error:** `Rate limit exceeded`
**Fix:** Bluesky has generous limits; wait 1-2 minutes

### Duplicate Posts

**Issue:** Same game posted multiple times
**Fix:** Check dedupe window (`POST_DEDUPE_WINDOW_HOURS`), clear queue if needed

### Posts Not Posting

**Checklist:**
1. Are credentials set in `.env`?
2. Are platform clients enabled? (Check logs for warnings)
3. Is queue being processed? (Run `--stats`)
4. Check logs for errors (`tail logs/automation.log`)

---

## 📈 Monitoring

### Logs
All operations are logged with timestamps:
```
2026-02-08 10:15:23 | INFO | __main__ | Starting scheduler mode (poll interval: 15min)
2026-02-08 10:15:24 | INFO | src.automation.social_media_manager | Social Media Manager initialized. Enabled: {'twitter', 'bluesky', 'discord'}. Dry run: False
2026-02-08 10:15:24 | INFO | src.automation.twitter_client | Twitter client initialized successfully
2026-02-08 10:15:24 | INFO | src.automation.bluesky_client | Bluesky client initialized: @perrypicks.bsky.social
2026-02-08 10:30:00 | INFO | src.automation.post_queue | Post enqueued: 0022500747_pregame_20260208103000_abc123 (twitter)
2026-02-08 10:30:02 | INFO | src.automation.twitter_client | Tweet posted successfully: ID=1234567890
```

### Metrics
- **Posts Queued:** Total posts in queue
- **Posts Posted:** Successfully posted posts
- **Posts Failed:** Failed posts (with retry info)
- **Duplicate Skips:** Posts blocked as duplicates
- **Queue Age:** Oldest pending post age

---

## ✅ Requirements Fulfilled

| Requirement | Status | Implementation |
|-----------|---------|----------------|
| **1. Post Generator** | ✅ COMPLETE | `post_generator.py` formats platform-optimized posts |
| **2. Social Media Integration** | ✅ COMPLETE | `twitter_client.py`, `bluesky_client.py` with full API support |
| **3. Posting Scheduler** | ✅ COMPLETE | `social_poster.py` with scheduled mode + cron support |
| **4. Duplicate Detection** | ✅ COMPLETE | `post_queue.py` with 24h dedupe window + history tracking |
| **5. Error Handling** | ✅ COMPLETE | Retry logic (3 attempts) + exponential backoff + DLQ |

---

## 🎉 Ready to Deploy!

All automation components are complete and production-ready:

✅ Post Generator - Formats optimized posts for all platforms  
✅ Twitter/X Integration - Full API v2 support with OAuth  
✅ Bluesky Integration - Complete Social API support  
✅ Social Media Manager - Orchestration across all platforms  
✅ Posting Scheduler - Continuous mode with cron support  
✅ Duplicate Detection - 24h dedupe window + history tracking  
✅ Error Handling - Retry logic + exponential backoff + DLQ  

**Deployment Guide:** See "Deployment" section above


---

**Author:** Perry (code-puppy)  
**Created:** February 8, 2026  
**Version:** 1.0.0  
**License:** MIT
