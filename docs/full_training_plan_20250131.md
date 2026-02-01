# Full Training Plan - 3 Season Expansion

**Date:** 2025-01-31  
**Status:** In Progress - Building Datasets

---

## 📊 Background Summary

### Halftime Champion Model (CONFIRMED)

| Model | MAE | DM Statistic | P-Value | Decision |
|--------|------|--------------|-----------|----------|
| **Ridge** | 11.60 | — | — | ✅ **CHAMPION** |
| XGBoost | 12.32 | 6.47 | 1.01e-08 | ❌ Rejected |

**Diebold-Mariano Test:**
- **P-Value: 1.01e-08** (extremely significant!)
- **Conclusion:** Ridge is SIGNIFICANTLY BETTER than XGBoost
- **Confidence:** 99.999999% that Ridge beats XGBoost

**Model Specs:**
```
Type: Linear Regression (L2 Regularized)
Alpha: 2.0
Features: 12 h1_* features
Target: h2_total (second half total)
Dataset: 2,796 games (23-24, 24-25)
```

---

## 🎯 Training Plan

### Phase 1: Data Collection ✅
- [x] Combined all 3 seasons into unified game IDs file
  - 23-24: 1,230 games
  - 24-25: 1,230 games
  - 25-26: 1,230 games
  - Total: **3,690 unique games**

### Phase 2: Dataset Building (IN PROGRESS)
- [ ] Pregame dataset: 3,520 completed games → target ~3,500
  - Status: Running in background (PID: 2964)
  - Log: /tmp/pregame_build_background.log
  
- [ ] Q3 dataset: All games → target 4,000 games
  - Status: Running in background (PID: 2969)
  - Log: /tmp/q3_build_background.log

### Phase 3: Model Training (PENDING)
For each model (Pregame, Q3), train 3 model types:
1. Ridge Regression (alpha=2.0, like halftime champion)
2. Random Forest
3. Gradient Boosting Trees (GBT)

### Phase 4: Model Selection (PENDING)
Use same methodology as halftime:
- Walk-forward temporal cross-validation (11 folds)
- Test size: 200 games per fold
- Diebold-Mariano statistical significance test
- Select champion based on significance (p < 0.05)

### Phase 5: Champion Retraining (PENDING)
- Retrain champion model on FULL dataset (all 3 seasons)
- Calibrate 80% confidence intervals
- Create model card documentation

### Phase 6: Game-State Integration (PENDING)
Implement automatic model switching:
- PreGame → Use pregame model
- Q1-Q2 → Use pregame model with clock adjustment
- Halftime → Use halftime model
- Q3 → Use halftime model with Q3 features
- Q4 → Use Q3 model

---

## 📈 Expected Dataset Sizes (After Full Build)

| Model | Current Games | Target Games | Increase |
|--------|--------------|---------------|----------|
| Pregame | 724 | ~3,500 | +400% |
| Q3 | 2,000 | ~4,000 | +100% |
| Halftime | 2,796 | 2,796 | — (already full) |

---

## 🔄 Current Status

### Background Processes
```
Process | PID | Description | Log File
---------|-----|-------------|----------
Pregame Build | 2964 | Building 3-season dataset | /tmp/pregame_build_background.log
Q3 Build | 2969 | Building 3-season dataset | /tmp/q3_build_background.log
```

### Monitoring Commands
```bash
# Check Pregame build progress
tail -f /tmp/pregame_build_background.log

# Check Q3 build progress
tail -f /tmp/q3_build_background.log

# Check dataset sizes periodically
PYTHONPATH=. .venv/bin/python -c "
import pandas as pd
pg = pd.read_parquet('data/processed/pregame_team_v2.parquet')
q3 = pd.read_parquet('data/processed/q3_team_v2.parquet')
print(f'Pregame: {len(pg)} | Q3: {len(q3)}')
"
```

---

## 🎯 Next Steps (After Datasets Complete)

1. **Stop background builds** (if needed)
   ```bash
   kill 2964 2969
   ```

2. **Verify dataset sizes**
   ```bash
   PYTHONPATH=. .venv/bin/python -c "
   import pandas as pd
   pg = pd.read_parquet('data/processed/pregame_team_v2.parquet')
   q3 = pd.read_parquet('data/processed/q3_team_v2.parquet')
   print(f'Pregame: {len(pg)}')
   print(f'Q3: {len(q3)}')
   "
   ```

3. **Run model comparison/backtest**
   - Need to create backtest scripts for Pregame and Q3
   - Use same methodology as halftime

4. **Select champions and train final models**
   - Based on Diebold-Mariano tests
   - Document with model cards

5. **Implement game-state-aware model switching** in `predict_from_game_id_v3_runtime.py`

---

## 📝 Model Comparison Template (To Be Created)

```python
# Model comparison output structure (like model_comparison.csv)
Rank,Model,MAE (train),MAE (test),RMSE (test),R² (test),DM vs Ridge,P-Value
1,Ridge,?,?,?,?,baseline,1.0
2,GBT,?,?,?,?,0.??,?
3,RandomForest,?,?,?,?,0.??,?
```

**Key:** Ridge will be the baseline (like halftime champion)

---

