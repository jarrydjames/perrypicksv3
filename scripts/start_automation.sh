#!/bin/bash
#
# PerryPicks Automation Startup Script
# Starts the automation runner in continuous mode
#
# IMPORTANT:
# The legacy runner (worker.runner) may not include DAILY_SUMMARY payload.date guard.
# Use unified_runner to ensure DAILY_SUMMARY only fires for the current CST day.
#

DATE_ARG=${1:-today}

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=========================================="
echo "PerryPicks Automation Startup"
echo "=========================================="
echo ""

# Activate virtual environment
echo "[1/3] Activating virtual environment..."
source "$PROJECT_DIR/.venv/bin/activate"
echo "✅ Virtual environment activated"
echo ""

# Change to project directory
cd "$PROJECT_DIR"

# Check if automation is already running
if pgrep -f "python -m worker.unified_runner" > /dev/null; then
    echo "⚠️  Automation is already running!"
    echo "   Process ID: $(pgrep -f 'python -m worker.runner')"
    echo ""
    echo "To stop the existing automation, run:"
    echo "  pkill -f 'python -m worker.runner'"
    echo ""
    exit 1
fi

# Start automation in continuous mode
echo "[2/3] Starting automation in continuous mode..."
# Unified runner (preferred)
nohup python -m worker.unified_runner --date $DATE_ARG >> logs/automation.out 2>&1 &
PID=$!
echo "✅ Automation started with PID: $PID"
echo ""

# Wait a moment and check if it's still running
sleep 2
if ps -p $PID > /dev/null 2>&1; then
    echo "[3/3] Verifying automation is running..."
    echo "✅ Automation is running successfully!"
    echo ""
    echo "=========================================="
    echo "Automation Status"
    echo "=========================================="
    echo "  Status: RUNNING"
    echo "  PID: $PID"
    echo "  Date: $DATE_ARG"
    echo "  Log file: logs/automation.log"
    echo "  Output: logs/automation.out"
    echo ""
    echo "To monitor logs:"
    echo "  tail -f logs/automation.out"
    echo ""
    echo "To stop automation:"
    echo "  kill $PID"
    echo "  or"
    echo "  pkill -f 'python -m worker.unified_runner'"
    echo ""
    echo "=========================================="
else
    echo "❌ Automation failed to start!"
    echo "   Check logs/automation.out for details"
    exit 1
fi
