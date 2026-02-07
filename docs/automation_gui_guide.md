# PerryPicks v3 - Automation Manager GUI

**Status:** ✅ COMPLETE & TESTED  
**Date:** February 8, 2026  
**Author:** Perry (code-puppy)

---

## 🚀 Overview

A **Streamlit-based GUI** for managing PerryPicks v3 social media automation.

**Separate from main app** - Does not impact manual research or reviews!

---

## ✨ Features

### 1. 📊 Dashboard
- Real-time automation status
- Queue statistics (pending, posted, failed)
- Platform status indicators
- Quick action buttons
- Recent activity feed

### 2. 🎮 Manual Predictions
- Select game from today's games
- Choose trigger type (pregame, halftime, q3)
- Select target platforms
- One-click prediction posting
- Dry-run mode for testing

### 3. 📋 Queue Manager
- View all queued posts
- Filter by status, platform, game ID
- Process queue manually
- Clear queue
- Post preview

### 4. 📜 History
- All posted predictions
- Search by game ID
- View post content
- Timestamp tracking
- Message IDs

### 5. ⚙️ Settings
- View current configuration
- Platform status
- Environment variable guidance
- Refresh configuration

### 6. 📝 Logs
- Log viewing interface
- Log level information
- CLI logging guidance

---

## 🏃 Quick Start

### 1. Run the GUI

```bash
cd "PerryPicks v3"

# Run automation manager GUI
streamlit run pages/04_Automation_Manager.py
```

### 2. Access the UI

Open your browser to:
- **Local:** http://localhost:8501
- **Network:** http://your-ip:8501

### 3. Configure Platforms

1. Go to **Settings** tab
2. Configure credentials in `.env` file:
   ```env
   # Discord (Required)
   DISCORD_WEBHOOK_URL=...

   # Twitter/X (Optional)
   TWITTER_CONSUMER_KEY=...
   TWITTER_ACCESS_TOKEN=...

   # Bluesky (Optional)
   BLUESKY_HANDLE=...
   BLUESKY_APP_PASSWORD=...
   ```
3. Refresh configuration in GUI

---

## 📱 Navigation

### Sidebar
- **🔄 Refresh Data** - Reload all data
- **Platform Status** - Quick view of platform status
- **Navigation** - Tab descriptions
- **ℹ️ Info** - Separation from main app

### Tabs
| Tab | Description |
|-----|-------------|
| **Dashboard** | Overview & statistics |
| **Manual** | Trigger predictions manually |
| **Queue** | Manage queued posts |
| **History** | View post history |
| **Settings** | Configuration |
| **Logs** | View logs |

---

## 🎮 Using the GUI

### Manual Predictions

1. Go to **Manual** tab
2. Select a game from the dropdown
3. Choose trigger type (pregame, halftime, q3)
4. Select platforms (or leave empty for all)
5. Toggle **Dry Run** mode (recommended for testing)
6. Click **🚀 Run Prediction**
7. View results below

### Queue Management

1. Go to **Queue** tab
2. Use filters to find posts
3. Click **🔄 Process Queue** to post pending items
4. Click **🗑️ Clear Queue** to remove all posts

### View History

1. Go to **History** tab
2. Click on any post to expand
3. View full content and metadata

---

## 🔧 Configuration

### Required (Discord)
```env
DISCORD_WEBHOOK_URL=https://discordapp.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN
```

### Optional (Twitter/X)
```env
TWITTER_CONSUMER_KEY=your_key
TWITTER_CONSUMER_SECRET=your_secret
TWITTER_ACCESS_TOKEN=your_token
TWITTER_ACCESS_TOKEN_SECRET=your_token_secret
```

### Optional (Bluesky)
```env
BLUESKY_HANDLE=your_handle.bsky.social
BLUESKY_APP_PASSWORD=your_app_password
```

### Automation Settings
```env
# Platform selection
SOCIAL_MEDIA_PLATFORMS=twitter bluesky discord

# Deduplication window (default: 24h)
POST_DEDUPE_WINDOW_HOURS=24
```

---

## 🏗️ Architecture

```
pages/04_Automation_Manager.py    # Main Streamlit app
    │
    ├── Dashboard Tab
    ├── Manual Predictions Tab
    ├── Queue Manager Tab
    ├── History Tab
    ├── Settings Tab
    └── Logs Tab
    │
    ▼
src/automation/automation_ui.py      # UI helpers
    ├── render_status_card()
    ├── render_platform_status()
    ├── render_queue_table()
    ├── render_post_content()
    └── ... (utility functions)
    │
    ▼
src/automation/                       # Core automation
    ├── social_media_manager.py
    ├── post_queue.py
    ├── post_generator.py
    └── ...
```

---

## 🔄 Separation from Main App

**Important:** This Automation Manager is **completely separate** from the main PerryPicks v3 Streamlit app.

### File Structure
```
PerryPicks v3/
├── streamlit_app.py              # MAIN APP (manual research/reviews)
├── pages/
│   ├── 01_Predictions.py         # Main app pages
│   ├── 02_Backtest.py
│   ├── 03_Model_Info.py
│   └── 04_Automation_Manager.py  # AUTOMATION MANAGER (separate!)
└── ...
```

### No Impact
- ✅ Does not modify main app
- ✅ Does not affect manual research
- ✅ Does not impact reviews
- ✅ Runs independently on different port

---

## 📊 Screenshots

### Dashboard
- Status cards (Processed, Pending, Posted, Failed)
- Platform status indicators
- Quick action buttons
- Recent activity feed

### Manual Predictions
- Game selector
- Trigger type dropdown
- Platform checkboxes
- Dry-run toggle
- Run prediction button
- Results display

### Queue Manager
- Filter controls (status, platform, game ID)
- Queue table with all posts
- Process queue button
- Clear queue button

---

## 🔍 Troubleshooting

### GUI Won't Start
- Check that Streamlit is installed: `pip install streamlit`
- Check for port conflicts: `streamlit run pages/04_Automation_Manager.py --server.port 8502`

### No Games Available
- Make sure `src.predict_api.fetch_todays_games()` works
- Check that NBA API is accessible

### Platforms Not Enabled
- Check credentials in `.env` file
- Click **Refresh Configuration** in Settings

### Posts Not Posting
- Check platform status in Dashboard
- Check Queue tab for errors
- Check logs for API errors

---

## 📖 Documentation

- **Full Automation Docs:** `docs/social_media_automation.md`
- **Quickstart Guide:** `docs/automation_quickstart.md`
- **Implementation Summary:** `docs/automation_implementation_summary.md`

---

## 🎉 Features Summary

| Feature | Status |
|---------|--------|
| Dashboard | ✅ Complete |
| Manual Predictions | ✅ Complete |
| Queue Manager | ✅ Complete |
| History | ✅ Complete |
| Settings | ✅ Complete |
| Logs | ✅ Complete |
| Platform Status | ✅ Complete |
| Post Preview | ✅ Complete |
| Filtering | ✅ Complete |
| Dry-run Mode | ✅ Complete |

---

## 🚀 Deployment

### Local Development
```bash
streamlit run pages/04_Automation_Manager.py
```

### Production (Streamlit Cloud)
1. Push to GitHub
2. Deploy to Streamlit Cloud
3. Configure environment variables in dashboard
4. Access via cloud URL

### Production (Docker)
```dockerfile
FROM python:3.14
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["streamlit", "run", "pages/04_Automation_Manager.py", "--server.address=0.0.0.0"]
```

---

**Author:** Perry (code-puppy)  
**Created:** February 8, 2026  
**Version:** 1.0.0  
**License:** MIT  

🐶 *Built with love and plenty of fetch time!*
