#!/bin/bash
# Run Q3 dataset builder continuously until reaching 2000 games

echo "============================================================"
echo "🏀 Q3 DATASET BUILDER - CONTINUOUS SHELL MODE"
echo "============================================================"
echo "Target: 2000 games"
echo "Starting..."
echo "============================================================"

COUNTER=0
MAX_RUNS=30  # Safety limit

while [ $COUNTER -lt $MAX_RUNS ]; do
    echo ""
    echo "Run #$(($COUNTER + 1))"
    echo "------------------------------------------------------------"
    
    # Check current progress
    python3 -c "
import pandas as pd
df = pd.read_parquet('data/processed/q3_team_v2.parquet')
print(f'Current: {len(df)} games')
if len(df) >= 2000:
    print('🎉 TARGET REACHED!')
    exit(1)
"
    
    STATUS=$?
    
    if [ $STATUS -eq 1 ]; then
        echo ""
        echo "============================================================"
        echo "🎉 TARGET REACHED: 2000 games!"
        echo "============================================================"
        exit 0
    fi
    
    # Run one batch
    python3 src/build_q3_incremental_fast.py
    
    COUNTER=$(($COUNTER + 1))
done

echo ""
echo "============================================================"
echo "✅ BUILD COMPLETE (Max runs reached)"
echo "============================================================"
