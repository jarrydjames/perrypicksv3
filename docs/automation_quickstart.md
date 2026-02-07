# PerryPicks v3 - Automation Quickstart Guide

**Get automated predictions posting to social media in 5 minutes!** 🚀

---

## 1️⃣ Install Dependencies

```bash
# Navigate to project directory
cd "PerryPicks v3"

# Install automation dependencies
pip install -r requirements-automation.txt

# Or with uv
uv pip install tweepy atproto schedule
```

---

## 2️⃣ Configure Environment

```bash
# Copy example config
cp config/env.example .env

# Edit with your credentials
vim .env  # or use your favorite editor
```

### Required (Discord)
```
DISCORD_WEBHOOK_URL=https://discordapp.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN
```

### Optional (Twitter/X)
```
# OAuth 1.0a (User Context)
TWITTER_CONSUMER_KEY=your_key
TWITTER_CONSUMER_SECRET=your_secret
TWITTER_ACCESS_TOKEN=your_token
TWITTER_ACCESS_TOKEN_SECRET=your_token_secret

# OR OAuth 2.0 (App-only)
TWITTER_BEARER_TOKEN=your_bearer_token
```

### Optional (Bluesky)
```
BLUESKY_HANDLE=your_handle.bsky.social
BLUESKY_APP_PASSWORD=your_app_password
```

---

## 3️⃣ Test Your Setup

### Check Status
```bash
# View current configuration
python scripts/automation/social_poster.py --stats
```

### Dry Run Test
```bash
# Test posting without actually posting
python scripts/automation/social_poster.py --stats --dry-run
```

---

## 4️⃣ Start Automation

### Option A: Continuous Scheduler (Recommended)

```bash
# Run in continuous mode (processes queue every 15 min)
python scripts/automation/social_poster.py --schedule --poll-interval 15
```

### Option B: Cron Job

```bash
# Edit crontab
crontab -e

# Add this line (process queue every 15 minutes)
*/15 * * * * cd /path/to/PerryPicks\ v3 && python3 scripts/automation/social_poster.py --schedule --poll-interval 15
```

### Option C: systemd Service

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

## 5️⃣ One-Off Predictions

```bash
# Post predictions for specific games
python scripts/automation/social_poster.py \
  --games 0022500747 0022500748 \
  --trigger-type pregame \
  --platforms discord
```

### Available Options
- `--trigger-type pregame|halftime|q3`: Prediction trigger
- `--mode auto|pregame|halftime|q3`: Prediction mode
- `--platforms twitter bluesky discord`: Target platforms
- `--dry-run`: Test without posting

---

## 🔍 Monitor & Debug

### View Statistics
```bash
python scripts/automation/social_poster.py --stats
```

### Process Queue Manually
```bash
python scripts/automation/social_poster.py --process-queue
```

### Verbose Logging
```bash
python scripts/automation/social_poster.py --schedule --verbose
```

---

## 📋 Common Issues

### Twitter: 401 Unauthorized
- **Fix:** Check credentials (consumer keys, access tokens)

### Bluesky: Authentication failed
- **Fix:** Verify handle format and app password

### Posts not appearing
- **Check 1:** Are credentials set in `.env`?
- **Check 2:** Are platforms enabled? (Check logs)
- **Check 3:** Is queue being processed? (Run `--stats`)

### Duplicate posts
- **System** automatically prevents duplicates within 24h window
- **Check:** Dedupe window with `POST_DEDUPE_WINDOW_HOURS`

---

## 📊 Architecture Overview

```
PerryPicks v3/
├── src/automation/
│   ├── post_generator.py       # Post formatting
│   ├── twitter_client.py        # Twitter/X API
│   ├── bluesky_client.py        # Bluesky API
│   ├── post_queue.py           # Queue + deduplication
│   ├── social_media_manager.py  # Platform orchestration
│   └── automation_orchestrator.py  # Main coordinator
├── scripts/automation/
│   └── social_poster.py     # CLI tool
└── data/
    └── automation.db        # Storage (auto-created)
```

---

## 🎯 Features

✅ **Multi-Platform Support** - Twitter/X, Bluesky, Discord  
✅ **Duplicate Detection** - Prevents repeat posts within 24h  
✅ **Error Handling** - Retry logic with exponential backoff  
✅ **Queue Management** - Post asynchronously, handle failures gracefully  
✅ **Flexible Scheduling** - Continuous mode, cron, or systemd  
✅ **Platform Optimization** - Platform-specific character limits & formatting  
✅ **Dry Run Mode** - Test without posting  
✅ **Statistics & Monitoring** - Track performance easily  

---

## 📖 Full Documentation

See `docs/social_media_automation.md` for complete documentation.

---

**Need Help?** Check the logs in `data/automation.db` or enable verbose logging.

🐶 *Built with love by Perry (code-puppy)*
