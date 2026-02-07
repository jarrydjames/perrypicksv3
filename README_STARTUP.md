# 🚀 PerryPicks v3 - Double-Click Startup

**Start the complete automation system with one double-click!**

---

## 🚀 Quick Start (Double-Click!)

### 🍎 macOS Users

**Just double-click:** `start_automation.command`

A Terminal window will open and automatically:
- ✅ Check Python (uv → python3 → python)
- ✅ Check/install dependencies (gracefully)
- ✅ Start backend automation
- ✅ Start frontend GUI
- ✅ Open browser to http://localhost:8501

### 🪟 Windows Users

**Just double-click:** `start_automation.bat`

A Command Prompt window will open and automatically:
- ✅ Check Python (python → python3)
- ✅ Check/install dependencies (gracefully)
- ✅ Start backend automation
- ✅ Start frontend GUI
- ✅ Open browser to http://localhost:8501

### 🐧 Linux Users

**Just double-click:** `start_automation.sh`

A terminal window will open and automatically:
- ✅ Check Python (uv → python3 → python)
- ✅ Check/install dependencies (gracefully)
- ✅ Start backend automation
- ✅ Start frontend GUI
- ✅ Open browser to http://localhost:8501

---

## 📁 Files Available

| Platform | File | Size | Description |
|----------|------|------|-------------|
| **macOS** | `start_automation.command` | 2.0 KB | Double-clickable macOS script |
| **Windows** | `start_automation.bat` | 2.0 KB | Double-clickable Windows batch script |
| **Linux/Mac** | `start_automation.sh` | 7.6 KB | Double-clickable bash script |
| **Python** | `start_automation.py` | 12.7 KB | Cross-platform Python script |

---

## 🎯 What to Expect

When you double-click the startup file, you'll see:

### 1. Banner
```
============================================================

   ╔═══════════════════════════════════════════════════════════╗
   ║                                                               ║
   ║    🤖 PerryPicks v3 - Automation System 🤖                  ║
   ║                                                               ║
   ║    Complete social media automation for NBA predictions            ║
   ║                                                               ║
   ╚═══════════════════════════════════════════════════════════╝
    
============================================================
```

### 2. Dependency Check
```
Checking dependencies...
✅ Using uv
✅ streamlit is installed
✅ tweepy is installed
✅ atproto is installed
✅ schedule is installed
✅ All dependencies are already installed
```

### 3. Backend Started
```
Starting backend automation...
✅ Backend automation started (PID: 12345)
```

### 4. Frontend Started
```
Starting frontend GUI...
✅ Frontend GUI started on http://localhost:8501 (PID: 12346)
```

### 5. Status Display
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

Your default browser opens automatically to:
```
http://localhost:8501
```

This is the Automation Manager GUI where you can:\- View real-time status
- Trigger predictions manually
- Manage the queue
- View history
- Access logs
- Change settings

---

## 🛑 How to Stop

### macOS & Linux

1. Press `Ctrl+C` in the Terminal window
2. Wait for graceful shutdown:
   ```
   Received shutdown signal...
   Stopping frontend GUI...
   Stopping backend automation...
   ✅ Shutdown complete
   ```
3. Press `Enter` to close the window

### Windows

1. Press `Ctrl+C` in the Command Prompt window
2. Wait for graceful shutdown
3. Press any key to close the window

---

## ⚙️ Advanced Options

You can customize the startup by editing the startup file:

### Example: Frontend Only

Edit the file and change:
```bash
python start_automation.py --frontend-only
```

### Example: Backend Only with 30-minute poll interval

Edit the file and change:
```bash
python start_automation.py --backend-only --poll-interval 30
```

### Example: Dry Run Mode (Testing)

Edit the file and change:
```bash
python start_automation.py --dry-run --verbose
```

### Example: Custom Port

Edit the file and change:
```bash
python start_automation.py --port 8502
```

---

## 🐛 Troubleshooting

### macOS: "Permission Denied"

**Error:** `bash: start_automation.command: Permission denied`

**Fix:** Open Terminal and run:
```bash
cd "PerryPicks v3"
chmod +x start_automation.command
```

Then double-click again.

### Windows: "Python is not installed"

**Error:** `Python is not installed or not in PATH`

**Fix:** 
1. Install Python 3.8+ from https://python.org
2. During installation, check "Add Python to PATH"
3. Restart Command Prompt
4. Double-click the .bat file again

### Window Closes Immediately

**Problem:** Window closes too fast to see errors

**Fix:** Open terminal manually:

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

### Port Already in Use

**Error:** `Port 8501 is already in use`

**Fix:** Edit the startup file and change the port:
```bash
python start_automation.py --port 8502
```

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| `DOUBLE_CLICK_STARTUP_GUIDE.md` | Comprehensive guide (14 KB) |
| `STARTUP_FILES_SUMMARY.md` | Quick reference (3.5 KB) |
| `AUTOMATION_STARTUP_README.md` | Startup script documentation (6.5 KB) |
| `AUTOMATION_COMPLETE.md` | Complete automation overview (6.9 KB) |

---

## 🎉 Summary

**You now have 3 double-clickable files:**

- ✅ **macOS:** `start_automation.command` - Just double-click!
- ✅ **Windows:** `start_automation.bat` - Just double-click!
- ✅ **Linux/Mac:** `start_automation.sh` - Just double-click!

**What they do:**

✅ Check/install dependencies automatically  
✅ Start backend automation  
✅ Start frontend GUI  
✅ Open browser to Automation Manager  
✅ Keep window open for logs  
✅ Graceful shutdown on Ctrl+C  
✅ Cross-platform support  

---

## 🚀 Start Now!

Just **double-click** your startup file:

- 🍎 **macOS:** `start_automation.command`
- 🪟 **Windows:** `start_automation.bat`
- 🐧 **Linux:** `start_automation.sh`

**That's it!** 🎉

---

**Author:** Perry (code-puppy)  
**Created:** February 7, 2026  
**Version:** 1.0.0  

🐶 *Double-click and go!*