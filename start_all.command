#!/bin/bash
# PerryPicks v3 - All-In-One Startup Script
#
# This script starts everything with one click:
# 1. Streamlit Frontend UI
# 2. Backend Services
# 3. Game State Monitor
#
# Usage: Double-click this file
#
# Press Ctrl+C to stop all services

set -e  # Exit on errors

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# PIDs for background processes
STREAMLIT_PID=""
GAME_STATE_PID=""

# Cleanup function
cleanup() {
    echo ""
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}  STOPPING ALL SERVICES${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # Stop Streamlit
    if [ ! -z "$STREAMLIT_PID" ]; then
        echo -e "${YELLOW}Stopping Streamlit (PID: $STREAMLIT_PID)...${NC}"
        kill $STREAMLIT_PID 2>/dev/null || true
        wait $STREAMLIT_PID 2>/dev/null || true
        echo -e "${GREEN}✓ Streamlit stopped${NC}"
    fi
    
    # Stop Game State Monitor
    if [ ! -z "$GAME_STATE_PID" ]; then
        echo -e "${YELLOW}Stopping Game State Monitor (PID: $GAME_STATE_PID)...${NC}"
        kill $GAME_STATE_PID 2>/dev/null || true
        wait $GAME_STATE_PID 2>/dev/null || true
        echo -e "${GREEN}✓ Game State Monitor stopped${NC}"
    fi
    
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  ALL SERVICES STOPPED${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    read -p "Press Enter to close..."
    exit 0
}

# Trap signals for graceful shutdown
trap cleanup SIGINT SIGTERM

# Banner
function print_banner() {
    clear
    echo -e "${PURPLE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${PURPLE}║${NC}  ${CYAN}🐶 PerryPicks v3 - All-In-One Startup${NC}                   ${PURPLE}║${NC}"
    echo -e "${PURPLE}║${NC}                                                              ${PURPLE}║${NC}"
    echo -e "${PURPLE}║${NC}  ${GREEN}Frontend UI  | Backend Services | Game Monitor${NC}        ${PURPLE}║${NC}"
    echo -e "${PURPLE}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Check virtual environment
function check_venv() {
    if [ ! -d ".venv" ]; then
        echo -e "${RED}❌ Error: .venv not found${NC}"
        echo ""
        echo "Please create a virtual environment first:"
        echo "  python3 -m venv .venv"
        echo "  source .venv/bin/activate"
        echo "  pip install -r requirements.txt"
        echo ""
        read -p "Press Enter to close..."
        exit 1
    fi
    
    echo -e "${GREEN}✓ Virtual environment found${NC}"
}

# Activate virtual environment
function activate_venv() {
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source .venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
}

# Start Streamlit
function start_streamlit() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Starting Streamlit Frontend UI${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # Log directory
    LOG_DIR="logs"
    mkdir -p "$LOG_DIR"
    STREAMLIT_LOG="$LOG_DIR/streamlit.log"
    
    # Start Streamlit in background
    streamlit run Home_Page.py \
        --server.port=8501 \
        --server.headless=true \
        --server.address=localhost \
        --browser.gatherUsageStats=false \
        --logger.level=info \
        > "$STREAMLIT_LOG" 2>&1 &
    
    STREAMLIT_PID=$!
    
    # Wait a moment for Streamlit to start
    sleep 3
    
    # Check if still running
    if ps -p $STREAMLIT_PID > /dev/null; then
        echo -e "${GREEN}✓ Streamlit started (PID: $STREAMLIT_PID)${NC}"
        echo -e "${CYAN}  URL: http://localhost:8501${NC}"
        echo -e "${CYAN}  Log: $STREAMLIT_LOG${NC}"
    else
        echo -e "${RED}✗ Streamlit failed to start${NC}"
        echo -e "${YELLOW}Check log: $STREAMLIT_LOG${NC}"
        return 1
    fi
}

# Start Game State Monitor
function start_game_state_monitor() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Starting Game State Monitor${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # Log directory
    LOG_DIR="logs"
    mkdir -p "$LOG_DIR"
    GAME_STATE_LOG="$LOG_DIR/game_state_monitor.log"
    
    # Start game state monitor in background
    python -m src.automation.game_state_service \
        > "$GAME_STATE_LOG" 2>&1 &
    
    GAME_STATE_PID=$!
    
    # Wait a moment for service to start
    sleep 2
    
    # Check if still running
    if ps -p $GAME_STATE_PID > /dev/null; then
        echo -e "${GREEN}✓ Game State Monitor started (PID: $GAME_STATE_PID)${NC}"
        echo -e "${CYAN}  Log: $GAME_STATE_LOG${NC}"
        echo -e "${CYAN}  Poll Interval: 30s${NC}"
        echo -e "${CYAN}  Platforms: All enabled${NC}"
    else
        echo -e "${RED}✗ Game State Monitor failed to start${NC}"
        echo -e "${YELLOW}Check log: $GAME_STATE_LOG${NC}"
        return 1
    fi
}

# Show status
function show_status() {
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  🚀 ALL SERVICES RUNNING${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${CYAN}📱 Frontend UI:${NC}"
    echo -e "   ${GREEN}Running${NC} (PID: $STREAMLIT_PID)"
    echo -e "   ${CYAN}URL:${NC}     http://localhost:8501"
    echo ""
    echo -e "${CYAN}🎮 Game State Monitor:${NC}"
    echo -e "   ${GREEN}Running${NC} (PID: $GAME_STATE_PID)"
    echo -e "   ${CYAN}Status:${NC}  Monitoring NBA games every 30s"
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  🔍 Monitor Logs${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${CYAN}Streamlit:${NC}     tail -f logs/streamlit.log"
    echo -e "${CYAN}Game Monitor:${NC} tail -f logs/game_state_monitor.log"
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  ⏹  To Stop${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "   Press ${RED}Ctrl+C${NC} to stop all services"
    echo ""
}

# Monitor services
function monitor_services() {
    echo -e "${GREEN}Monitoring services... (Press Ctrl+C to stop)${NC}"
    echo ""
    
    # Wait for any service to stop
    while true; do
        # Check Streamlit
        if ! ps -p $STREAMLIT_PID > /dev/null 2>&1; then
            echo -e "${RED}⚠️  Streamlit stopped unexpectedly!${NC}"
            cleanup
        fi
        
        # Check Game State Monitor
        if ! ps -p $GAME_STATE_PID > /dev/null 2>&1; then
            echo -e "${RED}⚠️  Game State Monitor stopped unexpectedly!${NC}"
            cleanup
        fi
        
        # Sleep
        sleep 5
    done
}

# Main execution
main() {
    # Print banner
    print_banner
    
    # Check venv
    check_venv
    
    # Activate venv
    activate_venv
    
    # Start services
    start_streamlit
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to start Streamlit${NC}"
        read -p "Press Enter to close..."
        exit 1
    fi
    
    start_game_state_monitor
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to start Game State Monitor${NC}"
        cleanup
    fi
    
    # Show status
    show_status
    
    # Monitor services
    monitor_services
}

# Run main
main
