# 🤖 PerryPicks v3 Automation Manager

**A beautiful Streamlit GUI for managing NBA prediction automation!**

---

## ⚡ Quick Start

```bash
cd "PerryPicks v3"
streamlit run pages/04_Automation_Manager.py
```

Open browser to: http://localhost:8501

---

## 📦 What Is This?

A **Streamlit-based GUI** that lets you:

- ✅ **Trigger predictions manually** for specific games
- ✅ **View and manage** queued posts
- ✅ **See real-time status** of all platforms
- ✅ **Browse post history** with search
- ✅ **Configure settings** visually
- ✅ **Monitor automation** with statistics

**Separate from main PerryPicks v3 app** - does not impact manual research or reviews!

---

## 🎯 Use Cases

### 1. Manual Trigger
Need to post a prediction for a specific game?

1. Open Automation Manager
2. Go to **Manual** tab
3. Select game, trigger type, platforms
4. Click **🚀 Run Prediction**
5. Done!

### 2. Queue Management
Need to check what's queued for posting?

1. Go to **Queue** tab
2. View all pending posts
3. Filter by status, platform, or game ID
4. Process or clear queue

### 3. Status Check
Want to see if automation is running?

1. Go to **Dashboard** tab
2. Check status cards (Pending, Posted, Failed)
3. View platform status (Twitter, Bluesky, Discord)
4. See recent activity

### 4. History Review
Want to see what was posted recently?

1. Go to **History** tab
2. Browse all posted predictions
3. Click any post to expand
4. View full content and metadata

---

## 🖥️ Interface Tour

### Dashboard
```
┌─────────────────────────────────────────┐
│ 📊 Dashboard                          │
├─────────────────────────────────────────┤
│ Status Cards:                          │
│ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ │
│ │  15 │ │  3  │ │  12 │ │  0  │ │
│ │Proc.│ │Pend.│ │Post.│ │Fail.│ │
│ └─────┘ └─────┘ └─────┘ └─────┘ │
├─────────────────────────────────────────┤
│ Platform Status:                       │
│ 🐦 Twitter  🦋 Bluesky  💬 Discord  │
│ ✅ Enabled  ✅ Enabled  ✅ Enabled    │
├─────────────────────────────────────────┤
│ Quick Actions:                        │
│ [🔄 Process] [📋 View Queue] [⚙️]  │
├─────────────────────────────────────────┤
│ Recent Activity:                       │
│ Game ID | Platform | Status            │
│ 00225007 | Discord | Pending           │
│ 00225008 | Twitter | Posted            │
└─────────────────────────────────────────┘
```

### Manual Predictions
```
┌─────────────────────────────────────────┐
│ 🎮 Manual Predictions                 │
├─────────────────────────────────────────┤
│ Select Game:                          │
│ [0022500747 ▼]                       │
├─────────────────────────────────────────┤
│ Trigger Type:                         │
│ [pregame ▼]                           │
├─────────────────────────────────────────┤
│ Select Platforms:                     │
│ ☑ Twitter  ☑ Bluesky  ☑ Discord    │
├─────────────────────────────────────────┤
│ ☐ Dry Run (don't actually post)     │
├─────────────────────────────────────────┤
│ [🚀 Run Prediction]                  │
└─────────────────────────────────────────┘
```

---

## 🔧 Setup

### 1. Configure Platforms

Create/Edit `.env` file:

```env
# Discord (Required)
DISCORD_WEBHOOK_URL=https://discordapp.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN

# Twitter/X (Optional)
TWITTER_CONSUMER_KEY=your_key
TWITTER_CONSUMER_SECRET=your_secret
TWITTER_ACCESS_TOKEN=your_token
TWITTER_ACCESS_TOKEN_SECRET=your_token_secret

# Bluesky (Optional)
BLUESKY_HANDLE=your_handle.bsky.social
BLUESKY_APP_PASSWORD=your_app_password
```

### 2. Run Automation Manager

```bash
streamlit run pages/04_Automation_Manager.py
```

### 3. Access UI

Open browser to: http://localhost:8501

---

## 📋 Tabs Explained

| Tab | Description |
|-----|-------------|
| **Dashboard** | Real-time status, stats, quick actions |
| **Manual** | Trigger predictions manually |
| **Queue** | View/manage queued posts |
| **History** | Browse post history |
| **Settings** | View configuration |
| **Logs** | View logs (coming soon) |

---

## 🔄 How It Works

### Manual Prediction Flow
```
User selects game → User clicks Run
                      ↓
            predict_game() called
                      ↓
            Post Generator formats content
                      ↓
            Posts enqueued for each platform
                      ↓
            Queue Manager processes posts
                      ↓
            Posts sent to APIs (Twitter, Bluesky, Discord)
```

### Queue Processing
```
Posts enqueued → Queue Manager polls
                      ↓
            Post selected (pending status)
                      ↓
            Attempt to post to platform
                      ↓
            Success → Mark as posted
            Failure → Retry (3 attempts with backoff)
```

---

## 🔍 Features

### Dashboard
- ✅ Real-time status cards
- ✅ Platform status indicators
- ✅ Quick action buttons
- ✅ Recent activity feed

### Manual Predictions
- ✅ Game selector (today's games)
- ✅ Trigger type selection (pregame, halftime, q3)
- ✅ Platform checkboxes
- ✅ Dry-run mode for testing
- ✅ One-click posting

### Queue Manager
- ✅ View all queued posts
- ✅ Filter by status, platform, game ID
- ✅ Process queue manually
- ✅ Clear queue
- ✅ Post preview

### History
- ✅ Browse all posted predictions
- ✅ Expand posts to see full content
- ✅ View metadata (timestamps, message IDs)

### Settings
- ✅ View current configuration
- ✅ Platform status
- ✅ Environment variable guidance

---

## ⚙️ Configuration

All settings managed via `.env` file:

```env
# Platform selection
SOCIAL_MEDIA_PLATFORMS=twitter bluesky discord

# Deduplication window (default: 24h)
POST_DEDUPE_WINDOW_HOURS=24

# Retry settings
DISCORD_MAX_RETRIES=3
DISCORD_RETRY_BACKOFF_SECONDS=1.5
```

---

## 🚀 Deployment Options

### Local Development
```bash
streamlit run pages/04_Automation_Manager.py
```

### Streamlit Cloud
1. Push to GitHub
2. Deploy to Streamlit Cloud
3. Configure environment variables
4. Access via cloud URL

### Docker
```bash
docker build -t perrypicks-automation .
docker run -p 8501:8501 -e DISCORD_WEBHOOK_URL=... perrypicks-automation
```

---

## 📖 Documentation

- **Full Guide:** `docs/automation_gui_guide.md`
- **Automation Docs:** `docs/social_media_automation.md`
- **Quickstart:** `docs/automation_quickstart.md`

---

## 🐛 Troubleshooting

### GUI Won't Start
```bash
# Check Streamlit installation
pip install streamlit

# Try different port
streamlit run pages/04_Automation_Manager.py --server.port 8502
```

### No Games Available
```bash
# Check NBA API
python -c "from src.predict_api import fetch_todays_games; print(fetch_todays_games())"
```

### Platforms Not Enabled
```bash
# Check .env file
cat .env | grep -E 'TWITTER|BLUESKY|DISCORD'

# Refresh configuration in GUI
# Click "Refresh Configuration" in Settings
```

---

## 📞 Support

- Check logs in GUI (Logs tab)
- Check console output when running GUI
- See documentation in `docs/` folder

---

## 🎉 Summary

**The Automation Manager provides:**

✅ **Beautiful GUI** - Streamlit-based interface  
✅ **Manual Control** - Trigger predictions on demand  
✅ **Queue Management** - View and manage posts  
✅ **Real-time Status** - See what's happening now  
✅ **History Tracking** - Browse all posted predictions  
✅ **Configuration** - Visual settings management  
✅ **Separate App** - Does not impact main PerryPicks v3 app  

**Start using it today:**

```bash
streamlit run pages/04_Automation_Manager.py
```

---

**Author:** Perry (code-puppy)  
**Created:** February 8, 2026  
**Version:** 1.0.0  

🐶 *Built with love and plenty of fetch time!*
