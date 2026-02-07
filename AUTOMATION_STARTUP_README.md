# 🚀 PerryPicks v3 - Automation Startup Script

**One-stop script to start the complete automation system!**

---

## 🎯 What It Does

The `start_automation.py` script handles everything:

1. ✅ **Check and install dependencies** (automatically)
2. ✅ **Start backend automation** (CLI scheduler)
3. ✅ **Start frontend GUI** (Streamlit)
4. ✅ **Handle graceful shutdown** (Ctrl+C)
5. ✅ **Support uv** (uses uv if available)
6. ✅ **Verbose logging** (optional)

---

## 🚀 Quick Start

### Start Everything (Backend + Frontend)

```bash
cd "PerryPicks v3"
python start_automation.py
```

That's it! The script will:
- Check dependencies
- Start backend automation
- Start frontend GUI
- Open browser to http://localhost:8501

### Start Frontend Only

```bash
python start_automation.py --frontend-only
```

### Start Backend Only

```bash
python start_automation.py --backend-only
```

### Dry Run Mode

```bash
python start_automation.py --dry-run
```

---

## ⚙️ Options

| Option | Description | Default |
|--------|-------------|----------|
| `--port PORT` | Port for Streamlit GUI | 8501 |
| `--poll-interval MINUTES` | Backend poll interval | 15 |
| `--backend-only` | Start only backend | - |
| `--frontend-only` | Start only frontend | - |
| `--dry-run` | Run backend in dry-run mode | - |
| `--headless` | Run frontend in headless mode | - |
| `--no-deps` | Skip dependency check | - |
| `--verbose` | Enable verbose logging | - |
| `--help` | Show help message | - |

---

## 📋 Examples

### Start with custom port
```bash
python start_automation.py --port 8502
```

### Start with custom poll interval
```bash
python start_automation.py --poll-interval 30
```

### Start backend only (no GUI)
```bash
python start_automation.py --backend-only --poll-interval 15
```

### Dry run (testing)
```bash
python start_automation.py --dry-run --verbose
```

### Skip dependency check
```bash
python start_automation.py --no-deps
```

---

## 🔧 How It Works

### 1. Dependency Check

```bash
Checking dependencies...
✅ All dependencies are already installed
```

If packages are missing, the script will:
- Check for `uv` (preferred) or use system Python
- Install from `requirements-automation.txt`
- Install from `requirements.txt`
- Install any remaining missing packages

### 2. Start Backend

```bash
Starting backend automation: uv run python scripts/automation/social_poster.py --schedule --poll-interval 15
✅ Backend automation started
```

The backend automation:
- Runs in background
- Processes queue every N minutes (default: 15)
- Posts to Twitter, Bluesky, Discord

### 3. Start Frontend

```bash
Starting frontend GUI: uv run streamlit run pages/04_Automation_Manager.py --server.port 8501
✅ Frontend GUI started on http://localhost:8501
```

The frontend GUI:
- Dashboard with status cards
- Manual prediction triggers
- Queue management
- History viewer
- Settings
- Logs

### 4. Print Status

```
============================================================
PerryPicks v3 - Automation System
============================================================

Status:
  Backend: ✅ Running
  Frontend: ✅ Running

  Frontend URL: http://localhost:8501

Press Ctrl+C to stop
============================================================
```

---

## 🛑 Stopping the System

Press `Ctrl+C` to gracefully stop:

```
Received signal 2, shutting down...
Stopping frontend GUI...
Stopping backend automation...
✅ Shutdown complete
```

---

## 🔍 Troubleshooting

### Port Already in Use

```bash
# Use different port
python start_automation.py --port 8502
```

### Dependencies Not Installing

```bash
# Check Python version
python --version

# Install manually
pip install -r requirements-automation.txt
# Then skip dependency check
python start_automation.py --no-deps
```

### Backend Not Starting

```bash
# Enable verbose logging
python start_automation.py --verbose
# Check error messages
# Verify scripts/automation/social_poster.py exists
```

### Frontend Not Starting

```bash
# Enable verbose logging
python start_automation.py --verbose --frontend-only
# Check error messages
# Verify pages/04_Automation_Manager.py exists
```

---

## 🐛 Common Errors

### ModuleNotFoundError

**Error:** `ModuleNotFoundError: No module named 'streamlit'`

**Fix:** The script will automatically install dependencies. If it fails:
```bash
pip install -r requirements-automation.txt
```

### Permission Denied

**Error:** `PermissionError: [Errno 13] Permission denied`

**Fix:** Make script executable:
```bash
chmod +x start_automation.py
```

### uv Not Found

**Error:** `uv: command not found`

**Fix:** The script will use system Python if uv is not available. This is fine.

---

## 📊 Architecture

```
start_automation.py
    │
    ├── Check Dependencies
    │   ├── Check for uv
    │   ├── Install from requirements files
    │   └── Verify packages
    │
    ├── Start Backend
    │   ├── social_poster.py
    │   └── --schedule --poll-interval 15
    │
    └── Start Frontend
        ├── Automation_Manager.py (Streamlit)
        └── --server.port 8501
```

---

## 🎯 Use Cases

### Development
```bash
# Start everything with verbose logging
python start_automation.py --verbose
```

### Production
```bash
# Start backend only (GUI optional)
python start_automation.py --backend-only --poll-interval 15
```

### Testing
```bash
# Dry run mode
python start_automation.py --dry-run --frontend-only
```

### Manual Control
```bash
# Frontend only (manual control via GUI)
python start_automation.py --frontend-only
```

---
## 📖 Related Documentation

- `AUTOMATION_COMPLETE.md` - Complete automation system overview
- `AUTOMATION_MANAGER_README.md` - GUI documentation
- `docs/automation_gui_guide.md` - Complete GUI guide
- `docs/automation_quickstart.md` - 5-minute quickstart

---

## 🎉 Summary

**The startup script provides:**

✅ **One-command start** - `python start_automation.py`  
✅ **Auto dependency install** - Checks and installs missing packages  
✅ **Backend + Frontend** - Starts both automatically  
✅ **uv support** - Uses uv if available  
✅ **Graceful shutdown** - Handles Ctrl+C properly  
✅ **Flexible options** - Port, poll interval, dry-run, verbose  
✅ **Status monitoring** - Shows running processes  
✅ **Error handling** - Catches and reports errors  

**Start your automation in one command:**

```bash
cd "PerryPicks v3"
python start_automation.py
```

---

**Author:** Perry (code-puppy)  
**Created:** February 8, 2026  
**Version:** 1.0.0  

🐶 *Built with love and plenty of fetch time!*
