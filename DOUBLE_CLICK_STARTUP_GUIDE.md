# 🚀 Double-Click Startup Guide - PerryPicks v3

**Start the complete automation system with a single double-click!**

---

## 📁 Files Created

### macOS Users
- `start_automation.command` - Double-clickable macOS script
  - Opens a new Terminal window
  - Runs the Python startup script
  - Keeps window open for viewing logs

### Windows Users  
- `start_automation.bat` - Double-clickable Windows batch script
  - Opens a new Command Prompt window
  - Runs the Python startup script
  - Keeps window open for viewing logs

### Linux/Mac Users (Alternative)
- `start_automation.sh` - Double-clickable bash script (if file manager supports it)
  - Opens in default terminal
  - Runs the bash startup script
  - Keeps window open for viewing logs

---

## 🚀 How to Use

### macOS (Double-Click Method)

1. **Double-click** `start_automation.command`
2. A new Terminal window will open
3. Watch the automation start:
   - ✅ Dependencies checked/installed
   - ✅ Backend automation started
   - ✅ Frontend GUI started
   - ✅ Browser opens to http://localhost:8501
4. Press `Ctrl+C` to stop
5. Press `Enter` to close the window

**If you get "Permission Denied" error:**

```bash
chmod +x start_automation.command
```

Then double-click again.

### Windows (Double-Click Method)

1. **Double-click** `start_automation.bat`
2. A new Command Prompt window will open
3. Watch the automation start:
   - ✅ Python check
   - ✅ Dependencies checked/installed
   - ✅ Backend automation started
   - ✅ Frontend GUI started
   - ✅ Browser opens to http://localhost:8501
4. Press `Ctrl+C` to stop
5. Press any key to close the window

**If you get "Python is not installed" error:**

- Install Python 3.8 or later from https://python.org
- Make sure to check "Add Python to PATH" during installation
- Double-click the .bat file again

### Linux (Double-Click Method)

**Option 1: Double-click (if supported by file manager)**

1. **Double-click** `start_automation.sh`
2. Select "Run in Terminal" if prompted
3. Watch the automation start
4. Press `Ctrl+C` to stop
5. Press `Enter` to close

**Option 2: Right-click context menu**

1. Right-click `start_automation.sh`
2. Select "Run in Terminal" or "Execute"
3. Watch the automation start
4. Press `Ctrl+C` to stop
5. Press `Enter` to close

**Option 3: Command line**

```bash
cd "PerryPicks v3"
./start_automation.sh
```

---

## ⚙️ Customizing the Startup

### macOS & Linux

You can customize the startup by editing the `.command` or `.sh` file:

```bash
# Open in text editor
open -e start_automation.command  # macOS
# or
vim start_automation.sh  # Linux
```

Add options to the startup command:

```bash
# Example: Frontend only, custom port
python start_automation.py --frontend-only --port 8502

# Example: Backend only, 30 minute poll interval
python start_automation.py --backend-only --poll-interval 30

# Example: Dry run, verbose
python start_automation.py --dry-run --verbose
```

### Windows

You can customize the startup by editing the `.bat` file:

```bash
# Open in Notepad
notepad start_automation.bat
```

Add options to the startup command:

```batch
# Example: Frontend only, custom port
python start_automation.py --frontend-only --port 8502

# Example: Backend only, 30 minute poll interval
python start_automation.py --backend-only --poll-interval 30

# Example: Dry run, verbose
python start_automation.py --dry-run --verbose
```

---

## 🎯 What Happens When You Double-Click

### 1. Open Terminal/Command Prompt

A new window opens with:

```
============================================================

   ╔═════════════════════════════════════════════════════════════╗
   ║                                                               ║
   ║    🤖 PerryPicks v3 - Automation System 🤖                  ║
   ║                                                               ║
   ║    Complete social media automation for NBA predictions            ║
   ║                                                               ║
   ╚═════════════════════════════════════════════════════════════╝
    
============================================================
```

### 2. Check Dependencies

```
Checking dependencies...
✅ Using uv
✅ streamlit is installed
✅ tweepy is installed
✅ atproto is installed
✅ schedule is installed
✅ All dependencies are already installed
```

### 3. Start Backend

```
Starting backend automation: uv run python scripts/automation/social_poster.py --schedule --poll-interval 15
✅ Backend automation started (PID: 12345)
```

### 4. Start Frontend

```
Starting frontend GUI: uv run streamlit run pages/04_Automation_Manager.py --server.port 8501
✅ Frontend GUI started on http://localhost:8501 (PID: 12346)
```

### 5. Show Status

```
============================================================
PerryPicks v3 - Automation System
============================================================

Status:
  Backend: ✅ Running (PID: 12345)
  Frontend: ✅ Running (PID: 12346)
  Frontend URL: http://localhost:8501

Press Ctrl+C to stop
============================================================
```

### 6. Browser Opens

Your default browser should open automatically to:

```
http://localhost:8501
```

This is the Automation Manager GUI where you can:
- View real-time status
- Trigger predictions manually
- Manage the queue
- View history
- Access logs
- Change settings

---

## 🛑 Stopping the Automation

### macOS & Linux

1. Press `Ctrl+C` in the Terminal window
2. Wait for graceful shutdown:
   ```
   Received shutdown signal...
   Stopping frontend GUI (PID: 12346)...
   Stopping backend automation (PID: 12345)...
   ✅ Shutdown complete
   ```
3. Press `Enter` to close the window

### Windows

1. Press `Ctrl+C` in the Command Prompt window
2. Wait for graceful shutdown:
   ```
   Received shutdown signal...
   Stopping frontend GUI (PID: 12346)...
   Stopping backend automation (PID: 12345)...
   ✅ Shutdown complete
   ```
3. Press any key to close the window

---

## 🐛 Troubleshooting

### macOS: "Permission Denied"

**Error:**
```
bash: start_automation.command: Permission denied
```

**Fix:**
```bash
chmod +x start_automation.command
```

Then double-click again.

### Windows: "Python is not installed"

**Error:**
```
❌ Error: Python is not installed or not in PATH
Please install Python 3.8 or later from https://python.org
```

**Fix:**
1. Install Python from https://python.org
2. During installation, check "Add Python to PATH"
3. Restart Command Prompt
4. Double-click the .bat file again

### Window Closes Immediately

**Problem:** Window closes too fast to see errors

**Fix:** Open Terminal/Command Prompt manually and run:

**macOS/Linux:**
```bash
cd "PerryPicks v3"
./start_automation.sh
```

**Windows:**
```batch
cd "PerryPicks v3"
start_automation.bat
```

This keeps the window open so you can see error messages.

### Port Already in Use

**Error:**
```
Port 8501 is already in use
```

**Fix:** Edit the startup file and change the port:

```bash
python start_automation.py --port 8502
```

---

## 📊 Platform Comparison

| Platform | File | Double-Click Support | Auto-Open Browser | Keep Window Open |
|----------|------|---------------------|-------------------|------------------|
| **macOS** | `.command` | ✅ Native | ✅ Yes | ✅ Yes |
| **Windows** | `.bat` | ✅ Native | ✅ Yes | ✅ Yes |
| **Linux** | `.sh` | ⚠️ Depends on file manager | ✅ Yes | ✅ Yes |

---

## 🎉 Summary

**You now have 3 ways to start the automation:**

### Method 1: Double-Click (Easiest!)

**macOS:** Double-click `start_automation.command`
**Windows:** Double-click `start_automation.bat`
**Linux:** Double-click `start_automation.sh`

### Method 2: Startup Scripts

```bash
# Python script (cross-platform)
python start_automation.py

# Bash script (Mac/Linux)
bash start_automation.sh
```

### Method 3: Manual

```bash
# Backend only
python scripts/automation/social_poster.py --schedule --poll-interval 15

# Frontend only
streamlit run pages/04_Automation_Manager.py
```

---

## 📖 Related Documentation

- `AUTOMATION_STARTUP_README.md` - Startup script documentation
- `AUTOMATION_STARTUP_SCRIPTS_SUMMARY.md` - Quick reference
- `AUTOMATION_COMPLETE.md` - Complete automation overview

---

**Author:** Perry (code-puppy)  
**Created:** February 7, 2026  
**Version:** 1.0.0  

🐶 *Double-click and go!*
