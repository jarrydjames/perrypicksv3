#!/bin/bash
# Start Game State Monitoring Service
#
# This script starts the live game state monitoring service that:
# - Monitors NBA games in real-time
# - Automatically generates predictions at halftime and Q3-5min
# - Automatically processes queue to post to platforms
# - Runs hands-off, no manual intervention needed
#
# Usage:
#   ./start_game_state_monitor.sh [options]
#
# Options:
#   --dry-run       Run without actually posting (for testing)
#   --interval N     Poll interval in seconds (default: 30)
#   --platforms X,Y  Comma-separated platforms (default: all enabled)
#
# Environment variables:
#   GAME_STATE_POLL_INTERVAL  Poll interval in seconds
#   GAME_STATE_PLATFORMS       Comma-separated platforms
#   GAME_STATE_DRY_RUN        "true" to run without posting

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Default values
POLL_INTERVAL=30
DRY_RUN=false
PLATFORMS=""

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
    echo "Warning: .venv not found"
fi

# Set environment variables
export GAME_STATE_POLL_INTERVAL="$POLL_INTERVAL"
export GAME_STATE_PLATFORMS="$PLATFORMS"
export GAME_STATE_DRY_RUN="$DRY_RUN"

# Log file
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/game_state_monitor_$(date +%Y%m%d_%H%M%S).log"

# Banner
echo "="
echo "GAME STATE MONITORING SERVICE"
echo "="
echo "Poll Interval: ${POLL_INTERVAL}s"
echo "Platforms: ${PLATFORMS:-'All enabled'}"
echo "Dry Run: $DRY_RUN"
echo "Log File: $LOG_FILE"
echo "="
echo ""
echo "Starting service..."
echo "Press Ctrl+C to stop"
echo ""

# Start the service
python -m src.automation.game_state_service 2>&1 | tee "$LOG_FILE"

# Cleanup on exit
echo ""
echo "Service stopped. Log saved to: $LOG_FILE"
echo "To restart, run: $0"
echo "To view logs, run: tail -f $LOG_FILE"
