#!/bin/bash
# PerryPicks v3 - Automation Startup Script (Bash Version)
# 
# One-stop script to start the complete automation system
# Usage: ./start_automation.sh [options]

# Options:
#   --port PORT              Port for Streamlit GUI (default: 8501)
#   --poll-interval MINUTES   Backend poll interval (default: 15)
#   --backend-only            Start only backend
#   --frontend-only           Start only frontend
#   --dry-run                Run backend in dry-run mode
#   --no-deps                Skip dependency check
#   --verbose                Enable verbose logging
#   --help                   Show help message


set -e  # Exit on error


# Default values
PORT=8501
POLL_INTERVAL=15
BACKEND_ONLY=false
FRONTEND_ONLY=false
DRY_RUN=false
NO_DEPS=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --poll-interval)
            POLL_INTERVAL="$2"
            shift 2
            ;;
        --backend-only)
            BACKEND_ONLY=true
            shift
            ;;
        --frontend-only)
            FRONTEND_ONLY=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-deps)
            NO_DEPS=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "PerryPicks v3 - Automation Startup Script"
            echo ""
            echo "Usage: ./start_automation.sh [options]"
            echo ""
            echo "Options:"
            echo "  --port PORT              Port for Streamlit GUI (default: 8501)"
            echo "  --poll-interval MINUTES   Backend poll interval (default: 15)"
            echo "  --backend-only            Start only backend"
            echo "  --frontend-only           Start only frontend"
            echo "  --dry-run                Run backend in dry-run mode"
            echo "  --no-deps                Skip dependency check"
            echo "  --verbose                Enable verbose logging"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

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

# Check dependencies
if [ "$NO_DEPS" = false ]; then
    echo "Checking dependencies..."
    
    # Check for uv or python
    if command -v uv &> /dev/null; then
        PYTHON_CMD="uv run python"
        echo "✅ Using uv"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        echo "✅ Using python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
        echo "✅ Using system Python"
    else
        echo "❌ Error: Python not found!"
        echo "Please install Python 3.8 or later:"
        echo "  1. Using Homebrew: brew install python3"
        echo "  2. Or download from: https://python.org"
        exit 1
    fi
    
    # Check for required packages
    MISSING_PACKAGES=()
    
    for package in streamlit tweepy atproto schedule; do
        if $PYTHON_CMD -c "import $package" 2>/dev/null; then
            echo "✅ $package is installed"
        else
            echo "❌ $package is missing"
            MISSING_PACKAGES+=("$package")
        fi
    done
    
    # Install missing packages
    if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
        echo ""
        echo "Installing missing packages..."
        
        # Try to install from requirements files (gracefully)
        if command -v uv &> /dev/null; then
            for req_file in requirements-automation.txt requirements.txt; do
                if [ -f "$req_file" ]; then
                    echo "Installing from $req_file..."
                    if uv pip install -q -r "$req_file" 2>/dev/null; then
                        echo "✅ Installed from $req_file"
                    else
                        echo "⚠️  Failed to install from $req_file, will try individually"
                    fi
                fi
            done
            
            # Check which packages are still missing and install individually
            STILL_MISSING=()
            for package in "${MISSING_PACKAGES[@]}"; do
                if ! $PYTHON_CMD -c "import $package" 2>/dev/null; then
                    STILL_MISSING+=("$package")
                fi
            done
            
            if [ ${#STILL_MISSING[@]} -gt 0 ]; then
                echo "Installing remaining packages: ${STILL_MISSING[*]}"
                uv pip install -q "${STILL_MISSING[@]}" || {
                    echo "❌ Failed to install packages"
                    exit 1
                }
            fi
        else
            for req_file in requirements-automation.txt requirements.txt; do
                if [ -f "$req_file" ]; then
                    echo "Installing from $req_file..."
                    if pip install -q -r "$req_file" 2>/dev/null; then
                        echo "✅ Installed from $req_file"
                    else
                        echo "⚠️  Failed to install from $req_file, will try individually"
                    fi
                fi
            done
            
            # Check which packages are still missing and install individually
            STILL_MISSING=()
            for package in "${MISSING_PACKAGES[@]}"; do
                if ! $PYTHON_CMD -c "import $package" 2>/dev/null; then
                    STILL_MISSING+=("$package")
                fi
            done
            
            if [ ${#STILL_MISSING[@]} -gt 0 ]; then
                echo "Installing remaining packages: ${STILL_MISSING[*]}"
                pip install -q "${STILL_MISSING[@]}" || {
                    echo "❌ Failed to install packages"
                    exit 1
                }
            fi
        fi
        
        echo "✅ Dependencies installed"
    else
        echo "✅ All dependencies are already installed"
    fi
    echo ""
fi

# Build commands
if [[ "$PYTHON_CMD" == uv* ]]; then
    BACKEND_CMD="uv run python scripts/automation/social_poster.py --schedule --poll-interval $POLL_INTERVAL"
    FRONTEND_CMD="uv run streamlit run pages/04_Automation_Manager.py --server.port $PORT"
else
    BACKEND_CMD="$PYTHON_CMD scripts/automation/social_poster.py --schedule --poll-interval $POLL_INTERVAL"
    FRONTEND_CMD="$PYTHON_CMD -m streamlit run pages/04_Automation_Manager.py --server.port $PORT"
fi

# Add dry-run flag
if [ "$DRY_RUN" = true ]; then
    BACKEND_CMD="$BACKEND_CMD --dry-run"
fi
# Add verbose flag
if [ "$VERBOSE" = true ]; then
    BACKEND_CMD="$BACKEND_CMD --verbose"
    FRONTEND_CMD="$FRONTEND_CMD --verbose"
fi

# Start backend
if [ "$FRONTEND_ONLY" = false ]; then
    echo "Starting backend automation..."
    echo "$BACKEND_CMD"
    $BACKEND_CMD &
    BACKEND_PID=$!
    echo "✅ Backend automation started (PID: $BACKEND_PID)"
    echo ""
fi

# Start frontend
if [ "$BACKEND_ONLY" = false ]; then
    echo "Starting frontend GUI..."
    echo "$FRONTEND_CMD"
    $FRONTEND_CMD &
    FRONTEND_PID=$!
    echo "✅ Frontend GUI started on http://localhost:$PORT (PID: $FRONTEND_PID)"
    echo ""
fi

# Print status
echo "============================================================"
echo "PerryPicks v3 - Automation System"
echo "============================================================"
echo ""
echo "Status:"
if [ "$FRONTEND_ONLY" = false ]; then
    echo "  Backend: ✅ Running (PID: $BACKEND_PID)"
else
    echo "  Backend: ❌ Not running"
fi
if [ "$BACKEND_ONLY" = false ]; then
    echo "  Frontend: ✅ Running (PID: $FRONTEND_PID)"
    echo "  Frontend URL: http://localhost:$PORT"
else
    echo "  Frontend: ❌ Not running"
fi
echo ""
echo "Press Ctrl+C to stop"
echo "============================================================"
echo ""

# Setup trap for graceful shutdown
trap cleanup INT TERM

cleanup() {
    echo ""
    echo "Received shutdown signal..."
    
    if [ "$FRONTEND_ONLY" = false ] && [ -n "$BACKEND_PID" ]; then
        echo "Stopping backend automation (PID: $BACKEND_PID)..."
        kill $BACKEND_PID 2>/dev/null || true
    fi
    
    if [ "$BACKEND_ONLY" = false ] && [ -n "$FRONTEND_PID" ]; then
        echo "Stopping frontend GUI (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    
    echo "✅ Shutdown complete"
    echo ""
    echo "Press Enter to close..."
    read -p ""
    exit 0
}

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID
