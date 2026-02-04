#!/bin/bash
#
# PerryPicks Automation Stop Script
# Stops the automation runner
#

echo "=========================================="
echo "PerryPicks Automation Stop"
echo "=========================================="
echo ""

# Check if automation is running
if ! pgrep -f "python -m worker.runner" > /dev/null; then
    echo "⚠️  Automation is not running!"
    echo ""
    exit 0
fi

# Get PID
PID=$(pgrep -f "python -m worker.runner")
echo "[1/2] Found automation process: PID $PID"
echo ""

# Stop automation
echo "[2/2] Stopping automation..."
kill $PID
echo "✅ Automation stopped"
echo ""

# Wait and verify
sleep 1
if ps -p $PID > /dev/null 2>&1; then
    echo "⚠️  Process still running, forcing stop..."
    kill -9 $PID
    sleep 1
fi

if ps -p $PID > /dev/null 2>&1; then
    echo "❌ Failed to stop automation!"
    exit 1
else
    echo "✅ Automation stopped successfully!"
    echo ""
    echo "=========================================="
    echo "To start automation again:"
    echo "  ./scripts/start_automation.sh"
    echo "=========================================="
fi
