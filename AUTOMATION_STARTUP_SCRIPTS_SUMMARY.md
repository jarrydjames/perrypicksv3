# 🚀 PerryPicks v3 - Startup Scripts

**One-command startup for the complete automation system!**

---

## 🎯 What You Get

Two startup scripts that handle everything:

1. ✅ **Check and install dependencies** (automatically)
2. ✅ **Start backend automation** (CLI scheduler)
3. ✅ **Start frontend GUI** (Streamlit)
4. ✅ **Handle graceful shutdown** (Ctrl+C)
5. ✅ **Support uv** (uses uv if available)
6. ✅ **Flexible options** (port, poll interval, dry-run)

---

## 📁 Files Created

```
PerryPicks v3/
├── start_automation.py          # Python startup script (recommended)
├── start_automation.sh          # Bash startup script (alternative)
└── AUTOMATION_STARTUP_README.md  # Startup documentation
```

---

## 🚀 Usage

### Method 1: Python Script (Recommended)

```bash
cd "PerryPicks v3"
python start_automation.py
```

### Method 2: Bash Script

```bash
cd "PerryPicks v3"
bash start_automation.sh
```

### Method 3: Direct Execution

```bash
cd "PerryPicks v3"
./start_automation.py
# or
./start_automation.sh
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

### Start Everything
```bash
python start_automation.py
# or
bash start_automation.sh
```

### Frontend Only (Manual Control)
```bash
python start_automation.py --frontend-only
```

### Backend Only (Production)
```bash
python start_automation.py --backend-only --poll-interval 30
```

### Dry Run (Testing)
```bash
python start_automation.py --dry-run --verbose
```

### Custom Port
```bash
python start_automation.py --port 8502
```

---

## 🔍 How It Works

### Step 1: Dependency Check

```bash
Checking dependencies...
✅ All dependencies are already installed
```

If packages are missing, the script will:
- Check for `uv` (preferred) or use system Python
- Install from `requirements-automation.txt`
- Install from `requirements.txt`
- Install any remaining missing packages

### Step 2: Start Backend

```bash
Starting backend automation: uv run python scripts/automation/social_poster.py --schedule --poll-interval 15
✅ Backend automation started
```

### Step 3: Start Frontend

```bash
Starting frontend GUI: uv run streamlit run pages/04_Automation_Manager.py --server.port 8501
✅ Frontend GUI started on http://localhost:8501
```

### Step 4: Print Status

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

## 🛑 Stopping

Press `Ctrl+C` to gracefully stop:

```bash
Received signal 2, shutting down...
Stopping frontend GUI...
Stopping backend automation...
✅ Shutdown complete
```

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Use different port
python start_automation.py --port 8502
```

### Dependencies Not Installing
```bash
# Install manually
pip install -r requirements-automation.txt
# Then skip dependency check
python start_automation.py --no-deps
```

### Script Not Executable
```bash
# Make executable
chmod +x start_automation.py
chmod +x start_automation.sh
```

---

## 📊 Differences Between Scripts

| Feature | Python Script | Bash Script |
|---------|---------------|-------------|
| **Dependencies** | Checks imports | Checks imports |
| **uv Support** | ✅ Yes | ✅ Yes |
| **Process Monitoring** | ✅ Yes | ✅ Yes |
| **Graceful Shutdown** | ✅ Yes | ✅ Yes |
| **Error Handling** | ✅ Comprehensive | ✅ Basic |
| **Verbose Logging** | ✅ Yes | ✅ Yes |
| **Cross-Platform** | ✅ Windows/Mac/Linux | ✅ Mac/Linux only |

---

## 📖 Related Documentation

- `AUTOMATION_STARTUP_README.md` - Detailed startup documentation
- `AUTOMATION_COMPLETE.md` - Complete automation system overview
- `AUTOMATION_MANAGER_README.md` - GUI documentation
- `docs/automation_gui_guide.md` - Complete GUI guide

---

## 🎉 Summary

**The startup scripts provide:**

✅ **One-command start** - `python start_automation.py`  
✅ **Auto dependency install** - Checks and installs missing packages  
✅ **Backend + Frontend** - Starts both automatically  
✅ **uv support** - Uses uv if available  
✅ **Graceful shutdown** - Handles Ctrl+C properly  
✅ **Flexible options** - Port, poll interval, dry-run, verbose  
✅ **Status monitoring** - Shows running processes  
✅ **Error handling** - Catches and reports errors  
✅ **Two versions** - Python (cross-platform) and Bash (Mac/Linux)  

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
