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
POST_DISCORD=false

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
        --post-discord)
            POST_DISCORD=true
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
DISCORD_POSTER="$PROJECT_DIR/scripts/automation/discord_poster.py"

# Today's date (or specified date)
if [ -n "$DATE" ]; then
    TODAY_DATE="$DATE"
else
    TODAY_DATE=$(date +"%Y-%m-%d")
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
    
    # Train halftime model using the merged temporal feature dataset
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
# STEP 6: DATA QUALITY GATES
###############################################################################

echo -e "
${YELLOW}[6/6]${NC} ${GREEN}Running data freshness + readiness gates...${NC}"

AUDIT_CMD="python3 "$PROJECT_DIR/src/data/data_freshness_audit.py" \
    --policy "$PROJECT_DIR/config/data_freshness_policy_v1.json" \
    --out "$PROJECT_DIR/reports/champion_runs/data_freshness_audit.json" \
    --strict"

READINESS_CMD="python3 "$PROJECT_DIR/src/pipelines/champion_refresh_cycle.py" \
    --policy "$PROJECT_DIR/config/champion_refresh_policy_v1.json" \
    --data-policy "$PROJECT_DIR/config/data_freshness_policy_v1.json" \
    --out "$PROJECT_DIR/reports/champion_runs/refresh_readiness.json" \
    --data-audit-out "$PROJECT_DIR/reports/champion_runs/data_freshness_audit.json""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}  Dry run: $AUDIT_CMD${NC}"
    echo -e "${YELLOW}  Dry run: $READINESS_CMD${NC}"
else
    cd "$PROJECT_DIR"
    eval "$AUDIT_CMD"
    eval "$READINESS_CMD"
    echo -e "${GREEN}  ✅ Data quality gates passed${NC}"
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
echo -e "${GREEN}  6. Data freshness + readiness gates${NC}"
echo -e "\n${GREEN}Production models: $PRODUCTION_DIR${NC}"
if [ "$POST_DISCORD" = true ]; then
    echo -e "${GREEN}Discord posting: enabled${NC}"
fi
echo -e "\n${BLUE}========================================================================${NC}"

###############################################################################
# OPTIONAL: POST PREGAME PREDICTIONS TO DISCORD
###############################################################################

if [ "$POST_DISCORD" = true ]; then
    if [ -z "$DISCORD_WEBHOOK_URL" ]; then
        echo -e "${RED}DISCORD_WEBHOOK_URL is required for --post-discord${NC}"
        exit 1
    fi
    if [ -f "$CACHE_DIR/todays_games.json" ]; then
        GAME_IDS=$(python3 - << 'EOF'
import json
from pathlib import Path

data = json.loads(Path("data/raw/todays_games.json").read_text())
game_ids = [g.get("gameId") for g in data.get("games", []) if g.get("gameId")]
print(" ".join(game_ids))
EOF
)
        if [ -n "$GAME_IDS" ]; then
            python3 "$DISCORD_POSTER" $GAME_IDS --mode pregame
        else
            echo -e "${YELLOW}No game IDs found in todays_games.json${NC}"
        fi
    else
        echo -e "${YELLOW}Missing data/raw/todays_games.json; skipping Discord post${NC}"
    fi
fi

# Exit
exit 0
