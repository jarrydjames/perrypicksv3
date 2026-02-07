# How to Generate and Post Predictions - Step by Step Guide
**Status:** ✅ Complete Guide
**Date:** February 7, 2026

---

## 🔍 The Issue

User reported: "I was able to get a prediction to process, but not pending or posted. or failed."

**Root Cause:** Posts are being **queued** but not yet **processed**. The automation system has two stages:
1. **Queue Stage:** Add posts to queue (status: `pending`)
2. **Process Stage:** Send posts from queue to platforms (status: `posting` → `posted`)

**What Happens When You Generate Predictions:**
1. ✅ Predictions are generated
2. ✅ Posts are added to queue (status: `pending`)
3. ❌ Posts are NOT yet sent to platforms
4. ⏳ Posts stay in queue waiting to be processed

---

## 📋 Step-by-Step Instructions

### Step 1: Start the Automation System

**Option A: Double-click (Easiest)**
```bash
# macOS:
./start_automation.command

# Windows:
start_automation.bat

# Linux:
bash start_automation.sh
```

**Option B: Manual Start**
```bash
# Terminal 1: Backend (scheduler)
python scripts/automation/social_poster.py --schedule

# Terminal 2: Frontend
streamlit run pages/04_Automation_Manager.py
```

### Step 2: Open Automation Manager

Browser should open to: `http://localhost:8501`

---

### Step 3: Generate Predictions (Queue Them)

1. Go to **Manual** tab
2. Select a date
3. Select mode: "Generate All Pregame Predictions"
4. **Uncheck** "🧪 Dry Run" (if you want actual posts)
5. Select platforms (e.g., Discord)
6. Click **"🚀 Generate Pregame Predictions for All N Games"**

**What You'll See:**
```
[████████████████████] 100%
🔄 Processing 0012400221 (1/10)...
🔄 Predicting 0012400221...
🔄 Posting 0012400221 to social media...
🔄 ✓ Completed 0012400221 (1 queued)
...

### Result
**Summary:**
- Total games: 10
- Predictions generated: 10
- Posts queued: 10
- Errors: 0
- Skipped (already processed): 0

🎉 All 10 predictions generated successfully!

✅ Queued 10 post(s)

📋 Post #1: 0012400221 (pregame)
[Shows post details]

### 📋 Queue Verification
**Current Queue Status:**
- Total posts in queue: 10
- Pending/posting: 10

**Recent Posts in Queue:**
- `0012400221` → `discord` (pending)
- `0012400222` → `discord` (pending)
...
```

**Important:** At this point, posts are **queued** (status: `pending`) but NOT yet sent to platforms!

---

### Step 4: Process the Queue (Send Posts to Platforms)

You have TWO options:

#### Option A: Manual Processing (One-time)

1. Go to **Dashboard** tab
2. Click **"🔄 Process Queue"** button
3. Watch as posts are sent:

```
Processing queue...
```

**What Happens:**
- Posts are sent from queue to platforms
- Each post's status changes: `pending` → `posting` → `posted`
- Posts are actually sent to Discord/Twitter/Bluesky


**Result:**
```
✅ Processed 10 posts!
```

4. Go to **Queue** tab to verify:

```
### 📋 Queue Manager
**Filter:** [pending, posting] | Platform: All | Game ID: 

**Showing 10 posts**

| Post ID | Game ID | Platform | Status | Created |
|----------|----------|----------|--------|---------|
| abc123... | 0012400221 | discord | posted | 2026-02-07 14:30 |
| def456... | 0012400222 | discord | posted | 2026-02-07 14:31 |
...
```

#### Option B: Automatic Processing (Scheduled)

If you want the scheduler to automatically process the queue:

**Important:** The scheduler is already running if you started with `--schedule` flag!

**How Scheduler Works:**
- Runs every 15 minutes (default)
- Processes up to 10 posts per cycle
- Checks for pending posts in queue
- Sends them to platforms
- Marks as posted/failed

**To Verify Scheduler is Running:**
1. Check terminal where you ran:
```bash
python scripts/automation/social_poster.py --schedule
```

2. Look for logs:
```bash
Starting automation scheduler...
Scheduled prediction at 14:00: pregame
Queue cycle: Processed=10, Success=10, Failed=0
Queue cycle: Processed=0, Success=0, Failed=0  # No new posts
```

3. Scheduler will automatically process your queued posts!

---

### Step 5: Verify Posts Were Sent

#### Check Queue Tab

1. Go to **Queue** tab
2. Set status filter to: `[posted]`
3. You should see all your posts with status `posted`

#### Check Your Platform

**For Discord:**
1. Go to your Discord channel
2. See the posts from PerryPicks

**For Twitter/X:**
1. Check your Twitter/X profile
2. See the posted tweets

**For Bluesky:**
1. Check your Bluesky profile
2. See the posted skeets

---

## 🔧 Troubleshooting

### Problem: Posts are queued but never sent

**Possible Causes:**
1. Scheduler not running
2. No manual "Process Queue" clicked
3. Posts are duplicates (skipped)
4. Platform not configured

**Solutions:**

#### Check if Scheduler is Running

```bash
# Check if process is running
ps aux | grep "social_poster"
```

If not running:
```bash
# Start it
python scripts/automation/social_poster.py --schedule
```

#### Check Queue Status

1. Go to **Queue** tab
2. Look at status filters:
   - If only `[pending]` posts → Need to process queue
   - If `[posted]` posts → Everything worked!
   - If `[failed]` posts → Something wrong (see errors)
   - If no posts → Something wrong with queue

#### Check Dry Run Mode

- If "🧪 Dry Run" is CHECKED → Posts will NOT be sent!
- **Uncheck** to actually send posts

#### Check Platform Configuration

Check `.env` file:
```env
# Discord (Required)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Twitter/X (Optional)
TWITTER_CONSUMER_KEY=...
TWITTER_CONSUMER_SECRET=...
TWITTER_ACCESS_TOKEN=...
TWITTER_ACCESS_TOKEN_SECRET=...
# Bluesky (Optional)
BLUESKY_HANDLE=...
BLUESKY_APP_PASSWORD=...
```

---

## 💡 Quick Reference

| Action | How |
|--------|------|
| **Generate predictions** | Manual tab → Generate All Pregame Predictions |
| **Process queue (one-time)** | Dashboard tab → Process Queue |
| **Process queue (auto)** | Start scheduler with `--schedule` flag |
| **Check if posted** | Queue tab → Filter by status `posted` |
| **Check platform** | Go to Discord/Twitter/Bluesky |

---

## 🎯 Best Practices

### For Testing
1. Keep "🧪 Dry Run" CHECKED
2. Generate predictions
3. Check Queue tab to see queued posts
4. Uncheck Dry Run
5. Process Queue
6. Verify posts appear in platform

### For Production
1. Start scheduler with `--schedule` flag
2. Generate predictions (Dry Run UNCHECKED)
3. Scheduler automatically processes every 15 minutes
4. Check Queue tab periodically
5. Posts appear automatically!

### For One-off Posts
1. Generate predictions (Dry Run UNCHECKED)
2. Immediately click "Process Queue"
3. Posts sent right away!

---

## 📊 Workflow Summary

```
┌─────────────────────────────────────────────┐
│ 1. Generate Predictions             │
│    - Predictions created               │
│    - Posts queued (status: pending)    │
│                                     │
│ 2. Process Queue (Manual or Auto)      │
│    - Posts sent to platforms            │
│    - Status: posting → posted         │
│                                     │
│ 3. Verify                           │
│    - Check Queue tab (status: posted)  │
│    - Check platform (see posts)       │
└─────────────────────────────────────────────┘
```

---

**Author:** Perry (code-puppy)
**Created:** February 7, 2026
**Status:** ✅ Complete

🐶 *Queue workflow explained step by step!* 🚀