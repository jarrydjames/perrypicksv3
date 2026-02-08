#!/bin/bash
# Start Game State Monitoring Service (macOS)
#
# This script starts live game state monitoring service that:
# - Monitors NBA games in real-time
# - Automatically generates predictions at halftime and Q3-5min
# - Automatically processes queue to post to platforms
# - Runs hands-off, no manual intervention needed
#
# Usage:
#   Double-click this file
#
# Environment variables:
#   GAME_STATE_POLL_INTERVAL  Poll interval in seconds
#   GAME_STATE_PLATFORMS       Comma-separated platforms
#   GAME_STATE_DRY_RUN        "true" to run without posting
#
# Make sure this file has execute permissions:
#   chmod +x start_game_state_monitor.command

set -e  # Exit on error

# Get script directory (this is the .command file location)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Default values
POLL_INTERVAL=30
DRY_RUN=false
PLATFORMS=""

# Check for config file
CONFIG_FILE=".game_state_monitor_config"
if [ -f "$CONFIG_FILE" ]; then
    echo "Loading config from $CONFIG_FILE"
    source "$CONFIG_FILE"
fi
# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --interval)
            POLL_INTERVAL="$2"
            shift 2
            ;;
        --platforms)
            PLATFORMS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --dry-run       Run without actually posting (for testing)"
            echo "  --interval N     Poll interval in seconds (default: 30)"
            echo "  --platforms X,Y  Comma-separated platforms (default: all enabled)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done
# Activate virtual environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "❌ Error: .venv not found"
    echo "Please create a virtual environment first"
    read -p "Press Enter to close"
    exit 1
fi
# Set environment variables
export GAME_STATE_POLL_INTERVAL="$POLL_INTERVAL"
export GAME_STATE_PLATFORMS="$PLATFORMS"
export GAME_STATE_DRY_RUN="$DRY_RUN"
# Log directory (macOS-friendly)
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/game_state_monitor_$TIMESTAMP.log"
# Banner
echo ""
echo "="
echo "  GAME STATE MONITORING SERVICE (macOS)"
echo "="
echo ""
echo "  Poll Interval: ${POLL_INTERVAL}s"
echo "  Platforms: ${PLATFORMS:-'All Enabled'}"
echo "  Dry Run: $DRY_RUN"
echo "  Log File: $LOG_FILE"
echo ""
echo "  Starting service..."
echo "  Press Ctrl+C to stop"
echo ""
echo "="
echo ""
# Create Terminal window if running from Finder (double-click)
if [ ! -t 0 ]; then
    # We're not in a terminal, so we must have been double-clicked
    # Open a new Terminal window and run the script there
    osascript -e 'tell application "Terminal" to do script ("'$0'")
    echo ""
    echo "ℹ️  Service started in new Terminal window"
    echo "   You can close this window"
    read -p "Press Enter to close this window"
    exit 0
fi
# Start service
echo "Monitoring live..."
echo ""
python -m src.automation.game_state_service 2>&1 | tee "$LOG_FILE"
# Cleanup on exit
EXIT_CODE=$?
echo ""
echo "="
echo "  SERVICE STOPPED"
echo "="
echo ""
echo "  Exit code: $EXIT_CODE"
echo "  Log saved to: $LOG_FILE"
echo ""
echo "  To restart, double-click this file again"
echo "  To view logs: tail -f $LOG_FILE"
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    read -p "Press Enter to close"
else
    echo "⚠️  Service exited with error code $EXIT_CODE"
    read -p "Press Enter to close"
fi
exit $EXIT_CODE
