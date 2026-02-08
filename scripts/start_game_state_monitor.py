"""Start Game State Monitoring Service (Cross-platform)

This script starts the live game state monitoring service.
Works on macOS, Windows, and Linux.

Usage:
    python start_game_state_monitor.py [options]

Options:
    --dry-run       Run without actually posting (for testing)
    --interval N     Poll interval in seconds (default: 30)
    --platforms X,Y  Comma-separated platforms (default: all enabled)
    --help           Show this help message

Environment variables:
    GAME_STATE_POLL_INTERVAL  Poll interval in seconds
    GAME_STATE_PLATFORMS       Comma-separated platforms
    GAME_STATE_DRY_RUN        "true" to run without posting
"""

import sys
import os
import signal
import subprocess
import platform
from pathlib import Path

# Configuration defaults
POLL_INTERVAL = 30
DRY_RUN = False
PLATFORMS = None
# Parse command-line arguments
args = sys.argv[1:]
i = 0
while i < len(args):
    arg = args[i]
    
    if arg == "--dry-run":
        DRY_RUN = True
    elif arg == "--interval" and i + 1 < len(args):
        try:
            POLL_INTERVAL = int(args[i + 1])
        except ValueError:
            print("Error: Interval must be a number")
            sys.exit(1)
        i += 1
    elif arg == "--platforms" and i + 1 < len(args):
        PLATFORMS = args[i + 1].split(",")
        i += 1
    elif arg in ["-h", "--help"]:
        print(__doc__)
        sys.exit(0)
    else:
        print(f"Unknown option: {arg}")
        print("Use --help for usage")
        sys.exit(1)
    
    i += 1

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent

cd = lambda: os.chdir(PROJECT_ROOT)
print(f"Project root: {PROJECT_ROOT}")
cd()
# Activate virtual environment
venv_path = PROJECT_ROOT / ".venv"
if venv_path.exists():
    print("Activating virtual environment...")
    python_path = venv_path / ("Scripts" if platform.system() == "Windows" else "bin") / "python"
else:
    print("❌ Error: .venv not found")
    print("Please create a virtual environment first")
    if platform.system() == "Windows":
        input("Press Enter to close...")
    else:
        input("Press Enter to close...")
    sys.exit(1)
# Set environment variables
os.environ["GAME_STATE_POLL_INTERVAL"] = str(POLL_INTERVAL)
os.environ["GAME_STATE_PLATFORMS"] = ",".join(PLATFORMS) if PLATFORMS else ""
os.environ["GAME_STATE_DRY_RUN"] = "true" if DRY_RUN else "false"
# Log directory
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
from datetime import datetime
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"game_state_monitor_{TIMESTAMP}.log"
# Banner
print()
print("="*60)
print("  GAME STATE MONITORING SERVICE")
print("="*60)
print()
print(f"  Poll Interval: {POLL_INTERVAL}s")
print(f"  Platforms: {'All Enabled' if not PLATFORMS else ', '.join(PLATFORMS)}")
print(f"  Dry Run: {DRY_RUN}")
print(f"  Log File: {LOG_FILE}")
print()
print("  Starting service...")
print("  Press Ctrl+C to stop")
print()
print("="*60)
print()
# Setup signal handlers
def signal_handler(signum, frame):
    print()
    print("="*60)
    print("  SHUTTING DOWN...")
    print("="*60)
    print()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)
if platform.system() != "Windows":
    signal.signal(signal.SIGTERM, signal_handler)
# Start the service as a subprocess (to handle terminal issues)
try:
    with open(LOG_FILE, "a") as log_file:
        process = subprocess.Popen(
            [str(python_path), "-m", "src.automation.game_state_service"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
        )
        
        # Stream output to both console and log file
        try:
            for line in process.stdout:
                print(line, end='')  # Print to console
                log_file.write(line)  # Write to log file
                log_file.flush()  # Ensure it's written immediately
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Error streaming output: {e}")
    
    # Wait for process to complete
    EXIT_CODE = process.wait()
except Exception as e:
    print(f"Error starting service: {e}")
    EXIT_CODE = 1
except KeyboardInterrupt:
    print()
    print("="*60)
    print("  INTERRUPTED")
    print("="*60)
    EXIT_CODE = 130
# Cleanup on exit
print()
print("="*60)
print("  SERVICE STOPPED")
print("="*60)
print()
print(f"  Exit code: {EXIT_CODE}")
print(f"  Log saved to: {LOG_FILE}")
print()
print("  To restart, run this script again")
print(f"  To view logs: tail -f {LOG_FILE}")
print()
if platform.system() == "Windows":
    input("Press Enter to close...")
else:
    input("Press Enter to close...")

sys.exit(EXIT_CODE)
