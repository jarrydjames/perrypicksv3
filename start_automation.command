#!/bin/bash
# PerryPicks v3 - Automation Startup (macOS Double-Click)
# 
# Double-click this file to start the complete automation system
# 
# This will open a new Terminal window and start both backend and frontend


# Get the directory where this file is located
SCRIPT_DIR="$(cd "$(dirname "$BASH_SOURCE[0]")" && pwd)"

# Change to script directory
cd "$SCRIPT_DIR" || exit 1

# Clear screen
clear

# Print banner
echo ""
echo "============================================================"
echo ""
echo "   ╔═════════════════════════════════════════════════════════════╗"
echo "   ║                                                               ║"
echo "   ║    🤖 PerryPicks v3 - Automation System 🤖                  ║"
echo "   ║                                                               ║"
echo "   ║    Complete social media automation for NBA predictions            ║"
echo "   ║                                                               ║"
echo "   ╚═════════════════════════════════════════════════════════════╝"
echo ""
echo "============================================================"
echo ""
echo "Starting automation system..."
echo ""

# Make sure scripts are executable
chmod +x start_automation.py 2>/dev/null || true
chmod +x start_automation.sh 2>/dev/null || true

# Detect Python command
if command -v uv &> /dev/null; then
    PYTHON_CMD="uv run python"
    echo "✅ Using uv"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "✅ Using python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    echo "✅ Using python"
else
    echo "❌ Error: Python not found!"
    echo ""
    echo "Please install Python 3.8 or later:"
    echo "  1. Using Homebrew: brew install python3"
    echo "  2. Or download from: https://python.org"
    echo ""
    read -p "Press Enter to close..."
    exit 1
fi

# Start automation (Python script preferred)
echo "Starting automation..."
echo ""

if [ -f "start_automation.py" ]; then
    echo "Using Python startup script..."
    echo "$PYTHON_CMD start_automation.py"
    echo ""
    $PYTHON_CMD start_automation.py
else
    echo "❌ Error: start_automation.py not found!"
    echo ""
    read -p "Press Enter to close..."
    exit 1
fi

# If script exits, keep Terminal window open
echo ""
echo "============================================================"
echo ""
echo "Automation stopped."
echo "Press Enter to close this window."
echo ""
read -p ""
