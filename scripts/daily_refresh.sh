#!/bin/bash

###############################################################################
# PerryPicks v3 - Daily Refresh Pipeline
#
# Automated daily refresh for temporal features and model training.
# This script fetches the latest games, rebuilds temporal features,
# merges with halftime stats, retrains the model, and deploys to production.
#
# Usage:
#   ./scripts/daily_refresh.sh                    # Default: last 2 days
#   ./scripts/daily_refresh.sh --days-filter 180  # Last 180 days (current season)
#   ./scripts/daily_refresh.sh --date "2026-01-29"  # Specific date
#
# Schedule:
#   Cron: 0 8 * * * /path/to/PerryPicks v3/scripts/daily_refresh.sh
#   Or run manually: ./scripts/daily_refresh.sh
###############################################################################

set -e  # Exit on error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

###############################################################################
# PARSING ARGUMENTS
###############################################################################

DATE=""
DAYS_FILTER=180
DRY_RUN=false
SKIP_FETCH=false
SKIP_BUILD=false
SKIP_MERGE=false
SKIP_TRAIN=false
SKIP_DEPLOY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --date)
            DATE="$2"
            shift 2
            ;;
        --days-filter)
            DAYS_FILTER="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-fetch)
            SKIP_FETCH=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --skip-merge)
            SKIP_MERGE=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip-deploy)
            SKIP_DEPLOY=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown argument: $1${NC}"
            exit 1
            ;;
    esac
done

###############################################################################
# CONFIGURATION
###############################################################################

# Paths
CACHE_DIR="$PROJECT_DIR/data/raw"
PROCESSED_DIR="$PROJECT_DIR/data/processed"
MODELS_DIR="$PROJECT_DIR/models_v3/halftime"
PRODUCTION_DIR="$PROJECT_DIR/models_v3/production"

# Today's date (or specified date)
if [ -n "$DATE" ]; then
    TODAY_DATE=$(date +"%Y-%m-%d")
else
    TODAY_DATE="$DATE"
fi

echo -e "${BLUE}========================================================================${NC}"
echo -e "${BLUE}PERRY PICKS V3 - DAILY REFRESH${NC}"
echo -e "${BLUE}========================================================================${NC}"
echo -e "${BLUE}Date: $TODAY_DATE${NC}"
echo -e "${BLUE}Days Filter: $DAYS_FILTER days${NC}"
echo -e "${BLUE}========================================================================${NC}"

###############################################################################
# STEP 1: FETCH TODAY'S GAMES
###############################################################################

if [ "$SKIP_FETCH" = false ]; then
    echo -e "\n${YELLOW}[1/5]${NC} ${GREEN}Fetching today's games...${NC}"
    
    FETCH_CMD="python3 \"$PROJECT_DIR/src/fetch_today_games.py\" \
        --days-before 2 \
        --days-after 1"
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}  Dry run: $FETCH_CMD${NC}"
    else
        cd "$PROJECT_DIR"
        eval "$FETCH_CMD"
        
        # Check if fetch succeeded
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}  ✅ Fetch complete${NC}"
        else
            echo -e "${RED}  ❌ Fetch failed${NC}"
            exit 1
        fi
    fi
else
    echo -e "\n${YELLOW}[1/5]${NC} ${YELLOW}Skipping fetch (--skip-fetch)${NC}"
fi

###############################################################################
# STEP 2: BUILD TEMPORAL FEATURES
###############################################################################

if [ "$SKIP_BUILD" = false ]; then
    echo -e "\n${YELLOW}[2/5]${NC} ${GREEN}Updating temporal features...${NC}"
    
    BUILD_CMD="python3 \"$PROJECT_DIR/src/build_temporal_features.py\" \
        --data-dir \"$CACHE_DIR/box\" \
        --output-dir \"$PROCESSED_DIR\" \
        --days-filter $DAYS_FILTER"
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}  Dry run: $BUILD_CMD${NC}"
    else
        cd "$PROJECT_DIR"
        eval "$BUILD_CMD"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}  ✅ Temporal features updated${NC}"
        else
            echo -e "${RED}  ❌ Temporal features build failed${NC}"
            exit 1
        fi
    fi
else
    echo -e "\n${YELLOW}[2/5]${NC} ${YELLOW}Skipping build (--skip-build)${NC}"
fi

###############################################################################
# STEP 3: MERGE TEMPORAL + HALFTIME
###############################################################################

if [ "$SKIP_MERGE" = false ]; then
    echo -e "\n${YELLOW}[3/5]${NC} ${GREEN}Merging temporal features with halftime stats...${NC}"
    
    MERGE_CMD="python3 \"$PROJECT_DIR/src/merge_temporal_halftime.py\""
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}  Dry run: $MERGE_CMD${NC}"
    else
        cd "$PROJECT_DIR"
        eval "$MERGE_CMD"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}  ✅ Merge complete${NC}"
        else
            echo -e "${RED}  ❌ Merge failed${NC}"
            exit 1
        fi
    fi
else
    echo -e "\n${YELLOW}[3/5]${NC} ${YELLOW}Skipping merge (--skip-merge)${NC}"
fi

###############################################################################
# STEP 4: RETRAIN MODEL
###############################################################################

if [ "$SKIP_TRAIN" = false ]; then
    echo -e "\n${YELLOW}[4/5]${NC} ${GREEN}Retraining halftime model...${NC}"
    
    # Note: train_halftime_model.py needs to be updated to use temporal features
    # For now, we'll run the existing training
    TRAIN_CMD="python3 \"$PROJECT_DIR/src/train_halftime_model.py\" \
        --dataset \"$PROCESSED_DIR/halftime_with_temporal_features.parquet\""
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}  Dry run: $TRAIN_CMD${NC}"
    else
        cd "$PROJECT_DIR"
        eval "$TRAIN_CMD"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}  ✅ Training complete${NC}"
        else
            echo -e "${RED}  ❌ Training failed${NC}"
            exit 1
        fi
    fi
else
    echo -e "\n${YELLOW}[4/5]${NC} ${YELLOW}Skipping training (--skip-train)${NC}"
fi

###############################################################################
# STEP 5: DEPLOY TO PRODUCTION
###############################################################################

if [ "$SKIP_DEPLOY" = false ]; then
    echo -e "\n${YELLOW}[5/5]${NC} ${GREEN}Deploying updated model to production...${NC}"
    
    # Create production directory
    mkdir -p "$PRODUCTION_DIR"
    
    # Copy model files
    cp "$MODELS_DIR"/* "$PRODUCTION_DIR/"
    
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}  Dry run: Would copy models to $PRODUCTION_DIR${NC}"
    else
        echo -e "${GREEN}  ✅ Models deployed to $PRODUCTION_DIR${NC}"
        ls -lh "$PRODUCTION_DIR"
    fi
else
    echo -e "\n${YELLOW}[5/5]${NC} ${YELLOW}Skipping deployment (--skip-deploy)${NC}"
fi

###############################################################################
# SUMMARY
###############################################################################

echo -e "\n${BLUE}========================================================================${NC}"
echo -e "${BLUE}DAILY REFRESH COMPLETE${NC}"
echo -e "${BLUE}========================================================================${NC}"
echo -e "${BLUE}Date: $TODAY_DATE${NC}"
echo -e "${BLUE}Days Filter: $DAYS_FILTER days${NC}"
echo -e "\n${GREEN}Steps completed:${NC}"
echo -e "${GREEN}  1. Fetch games${NC}"
echo -e "${GREEN}  2. Build temporal features${NC}"
echo -e "${GREEN}  3. Merge with halftime stats${NC}"
echo -e "${GREEN}  4. Retrain model${NC}"
echo -e "${GREEN}  5. Deploy to production${NC}"
echo -e "\n${GREEN}Production models: $PRODUCTION_DIR${NC}"
echo -e "\n${BLUE}========================================================================${NC}"

# Exit
exit 0

