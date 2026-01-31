# Comprehensive Execution & Requirements Specification for the NBA Forecasting Platform

This document is the **single, comprehensive specification** for how the coding platform must:

- validate data robustness/completeness before training,
- compute features without leakage (including season-reset rolling),
- run nested walkforward backtests,
- train and compare a full registry of candidate models (GBDT / NN / distributional / hybrid),
- calibrate uncertainty with conformal methods,
- measure accuracy with statistically valid tests (block bootstrap + Diebold–Mariano),
- pre-register experiments (6–10),
- and define production retraining + drift monitoring rules.

It also includes **both sets of lists** requested:

1. the **Information Request List** (what you feed the platform to generate code and configs), and
2. the **System Execution Coverage List** (the mandatory modules and outputs the platform must implement).

---

The platform must implement and expose these modules end-to-end:

1. **Exact model architecture (LightGBM / NN / hybrid)**
2. **Leakage-safe feature pipeline (forward-only, season-reset)**
3. **Nested walkforward CV code**
4. **Statistical testing module (bootstrap + DM)**
5. **Conformal uncertainty wrapper**
6. **Experiment plan (6–10 experiments, pre-registered)**
7. **Production retraining + drift monitoring rules**

Each module must be configurable, logged, and reproducible.

---

# 0. Platform Invariants (Non-negotiable)

## 0.1 Temporal validity is sacred

- No lookahead: any feature for game *g* must only depend on information available at or before the prediction cut for *g*.
- Any historical/rolling feature must only use games strictly prior to *g*.

## 0.2 Deterministic ordering

- Sort key must be deterministic even under tied timestamps.
- Always sort by `(gameTimeUTC, season_end_yy, game_id)` and persist fold indices.

## 0.3 Paired evaluation only

- Baseline and challengers must share identical folds and rows.
- All decisions use paired per-game loss differentials, not unpaired aggregates.

## 0.4 No silent changes

Every run must log:

- dataset hash/version
- feature set version
- fold indices
- model config and random seeds
- predictions and residuals
- metrics + uncertainty metrics
- statistical test outputs

---

# 1. Data Validation (PASS/FAIL Gate)

This step runs before any training or backtesting.

## 1.1 Load data

- Load main dataset
- Load rolling features dataset (if separate)

## 1.2 Schema & dtype checks (hard fail)

**Must include** IDs/time, targets, feature columns. Validate types:

- `gameTimeUTC` must be timezone-aware UTC datetime
- IDs integer-like
- features numeric

## 1.3 Primary key integrity (hard fail)

- Primary key: `(season_end_yy, game_id)`
- No duplicate keys
- Home team != away team

## 1.4 Missingness & completeness (hard fail thresholds)

- Targets: 0.0% missing
- Baseline halftime features: ≤ 0.1% missing each
- Temporal features: ≤ 2% missing each (expected early season)

Additionally:

- Produce a per-season and per-month missingness heatmap (artifact) to detect join regressions.

## 1.5 Temporal ordering integrity (hard fail)

- Count tied timestamps and confirm stable tie-break
- Verify reproducible ordering across repeated runs (same sorted order checksum)

## 1.6 Season/regime diagnostics (warning report)

- Report games per season
- Flag if playoffs mixed with regular season
- Flag cross-season rolling enabled

## 1.7 Leakage sentinels (hard fail)

### Sentinel A: forward-only rolling check

For each team, for each game i:

- Confirm every game contributing to rolling window has index < i (global stable index)

### Sentinel B: suspicious correlation check

- Compute |corr(feature, target)| and flag > 0.95 for review

### Sentinel C: time-shift placebo test

- Train a small model to predict next-game label (y\_{t+1}) from features at time t. Performance must collapse to noise.
- If not, investigate leakage.

**If any sentinel fails → FAIL and stop.**

---

# 2. Leakage-safe Feature Pipeline (Forward-only + Season-reset)

## 2.1 Feature gates (availability)

Each feature must be tagged:

- `PREGAME`
- `HALFTIME`
- `END_Q3`

During dataset creation for a given cut, the platform must:

- include only features whose gate ≤ cut
- exclude all later-cut features

## 2.2 Forward-only rolling features

Compute per team using a single forward pass:

- For row i, rolling stats use rows `[i-k, …, i-1]` only.

## 2.3 Season-reset rolling features (required option)

Rolling must be computed within `(team_id, season_end_yy)` groups.

- If fewer than k games exist within season, use expanding mean within season.

## 2.4 Optional opponent-strength adjustment

If enabled, rolling offensive/defensive ratings should be opponent-adjusted.

- Example: adjusted offensive rating = points per 100 poss relative to opponent defensive baseline.

## 2.5 Early-season missing handling

Choose one (configurable):

- expanding mean
- league-average prior + shrinkage
- prior-season carryover (explicitly flagged, never implicit)

## 2.6 Feature list canonicalization

To avoid accidental train/test mismatch:

- store a canonical ordered feature list in artifacts
- assert train and test use identical feature columns and preprocessing

---

# 3. Walkforward Backtesting (Outer CV)

## 3.1 Fold generation (default)

- `min_train = 500`
- `test_size = 200`
- `step = 200`
- contiguous blocks in time

Algorithm:

1. stable sort
2. for fold f=0..:
   - train = `[0, min_train + f*step)`
   - test  = `[train_end, train_end + test_size)`
   - stop when test\_end > N

## 3.2 Fold persistence

- Save folds as index arrays to disk (e.g., parquet/json)
- All models must reuse these indices

## 3.3 Per-fold outputs

For each fold and model:

- predictions per game
- per-game losses
- summary metrics
- calibration residuals (if needed)

---

# 4. Nested Walkforward Tuning (If used, MUST be nested)

For each outer fold:

1. Split outer-train into:
   - inner\_train: first 80–90%
   - inner\_val: last 10–20%
2. Tune hyperparameters only on inner\_val
3. Retrain best config on full outer-train
4. Evaluate once on outer-test

**Tuning budget (recommended):**

- max\_trials: 30–50 per model per fold (or tune once on an early representative period and freeze)
- early stop: stop search after 10 trials without improvement ≥ ε

---

# 5. Metrics, Accuracy Measurement, and Statistical Validity

## 5.1 Primary & secondary metrics

**Primary:** Total MAE (or explicitly configured primary) **Secondary:** Margin MAE, Total RMSE, Margin RMSE

Also recommended if probabilistic outputs exist:

- NLL / CRPS
- calibration error measures

## 5.2 Paired loss differentials

For each game i:

- `L_base_i = loss(y_i, yhat_base_i)`
- `L_new_i  = loss(y_i, yhat_new_i)`
- `d_i = L_new_i - L_base_i`

## 5.3 Block bootstrap (time-valid CI)

- Choose block size B (default 200)
- Sample contiguous blocks with replacement until length N
- Repeat R times (default 1000)
- Compute mean(d) distribution

Outputs:

- mean(d)
- 95% CI
- P(improvement) = P(mean(d\*) < 0)

## 5.4 Diebold–Mariano test (forecast accuracy)

- Apply DM to loss differential series d\_i
- Use Newey–West variance estimate to handle autocorrelation

Outputs:

- DM statistic
- p-value

## 5.5 Go / No-Go decision rule (pre-registered)

A model change may ship only if:

- Statistical: CI upper bound < 0 AND DM p < 0.05
- Practical: improvement ≥ pre-set threshold (e.g., ≥1% MAE reduction or ≥0.10 points)
- Safety: no material degradation in secondary targets
- Uncertainty: calibrated coverage meets targets (see Section 6)

---

# 6. Uncertainty Quantification (Default + Fallback)

## 6.1 Production default: Sliding-window conformal (time-series safe)

For each outer fold:

1. Choose calibration window W inside outer-train (default 1000 most recent games)
2. Predict on calibration window
3. Residuals r\_j = |y\_j - yhat\_j|
4. For target coverage (1-α), compute q = Quantile\_{1-α}(r)
5. Interval for each test point: [yhat - q, yhat + q]

No leakage: calibration data must be strictly earlier than test.

## 6.2 Fallback: Gaussian baseline (not preferred)

- Use only as fallback with explicit warning that coverage may fail under drift.

## 6.3 Optional: normalized conformal (heteroscedastic)

If the model provides scale σ(x):

- normalized residuals r\_j = |y\_j - yhat\_j| / σhat\_j
- interval: [yhat ± q \* σhat]

## 6.4 Required evaluation table (every run)

For each target:

- coverage at 50/60/70/80/90/95
- average width
- Winkler / interval score

## 6.5 Conditional coverage (required)

Compute coverage and width by bins:

- close games: |halftime\_margin| < 5
- blowouts: |halftime\_margin| > 15
- high pace: top 20% pace proxy
- low pace: bottom 20% pace proxy

---

# 7. Model Registry (All Candidates + Config + Calibration + Backtesting)

The platform must support these candidates, each with a unified interface:

## 7.1 Always-run baselines

1. Ridge Regression (fast linear baseline)
2. Random Forest
3. Sklearn GradientBoosting / HistGradientBoosting

## 7.2 Strong tabular candidates (must be tested)

4. LightGBM
5. CatBoost
6. XGBoost

Expose config:

- max\_depth, num\_leaves (or analog)
- learning\_rate
- n\_estimators
- min\_data\_in\_leaf
- subsample / colsample
- L1/L2 regularization

## 7.3 Distributional candidates (for better UQ)

7. NGBoost (predict μ, σ)
8. Quantile GBDT (q10/q50/q90)
9. Mean model + variance model (two-stage)

## 7.4 Neural candidates (tabular)

10. Tabular MLP + team embeddings
11. Two-stage model: pregame prior + in-game updater network

Neural config:

- embedding\_dim
- hidden\_layers, width
- dropout, weight\_decay
- optimizer and LR schedule
- early stopping patience
- loss: MAE/MSE or Gaussian/Student-t NLL

## 7.5 Hybrid candidates

12. Simple averaging ensemble
13. Time-safe stacking (meta-learner trained only on out-of-fold predictions)

Stacking leakage rule:

- meta-model trains only on OOF predictions generated strictly in time order

---

# 8. Exact Model Architectures (LightGBM / NN / Hybrid)

## 8.1 LightGBM architecture (two-head)

Train two regressors:

- Model\_T: predicts total target
- Model\_M: predicts margin target

Starting config (tune nested):

- objective: regression\_l1
- num\_leaves: 31
- learning\_rate: 0.03–0.07
- n\_estimators: up to 2000 with early stopping
- subsample: 0.8
- colsample\_bytree: 0.8
- reg\_lambda: 1.0
- min\_data\_in\_leaf: 50

## 8.2 Neural net architecture (tabular MLP)

Inputs:

- numeric features
- embeddings: home\_team\_id, away\_team\_id (optional season embedding)

Network:

- concat([X, emb\_home, emb\_away])
- Dense(256) + ReLU + Dropout
- Dense(128) + ReLU + Dropout
- Dense(64)  + ReLU
- Output:
  - 2-d mean head for (total, margin)
  - optional variance head for heteroscedastic NLL

Training:

- AdamW
- LR 1e-3 with decay
- early stopping on inner\_val MAE or NLL

## 8.3 Hybrid stacking architecture

1. Base models produce time-ordered OOF predictions
2. Meta features = base predictions (+ optional simple meta features)
3. Meta model (ridge or lgbm) trained on OOF
4. Evaluate on outer-test only

---

# 9. Pre-registered Experiment Plan (6–10)

Each experiment must declare: hypothesis, change, splits, metrics, statistical test, and go/no-go rule.

1. **Evaluation hardening**: implement paired deltas + block bootstrap + DM
2. **Conformal on current champion**: replace Gaussian intervals with sliding conformal
3. **Season-reset rolling features**: recompute temporal features with season reset
4. **LightGBM challenger**: nested tuning, compare vs current GBT
5. **CatBoost w/ categorical team IDs**: test team latent effects
6. **NGBoost + conformal**: distributional mean/scale then normalized conformal
7. **Tabular NN w/ embeddings**: strict regularization + nested tuning
8. **Simple ensemble**: average best two models
9. **Time-safe stacking**: OOF-based meta model
10. **Retraining cadence simulation**: expanding vs rolling windows and impact on late folds

---

# 10. Production Retraining + Drift Monitoring Rules

## 10.1 Retraining cadence

- In-season: weekly minimum (daily if compute allows)
- Off-season: retrain only when new season data arrives

## 10.2 Window strategy

- default: expanding window
- optional: rolling window on drift trigger

## 10.3 Drift monitoring signals

- PSI on key features (pace proxy, eFG proxy, foul rate)
- CUSUM on residual mean (bias drift)
- calibration drift: conformal coverage over last N games

## 10.4 Trigger rules

Retrain if any holds:

- PSI > 0.2 on 2+ key features
- rolling MAE increases > 5% for 3 consecutive evaluation blocks
- coverage deviates > 3% from target for 2 consecutive checks

## 10.5 Canary deployment

- shadow run candidate model 2–4 weeks
- promote only if live shadow matches backtest thresholds

---

# 11. Required Artifacts Per Run (Reproducibility)

The platform must output:

- Data validation report (PASS/FAIL + caveats)
- Stable sorted dataset checksum
- Fold indices artifact
- Feature list artifact
- Model configs and seeds
- Predictions per game and residuals
- Metrics per fold + aggregate
- Statistical test report (bootstrap CI + DM p-values)
- Uncertainty report (coverage table + conditional bins + interval score)
- Model card summarizing results + ship decision

---

# 12. Implementation Skeleton (function-level execution plan)

## 12.1 Orchestrator

```
run_experiment(config):
  df = load_data(config.data_paths)
  validation = validate_data(df, config.validation)
  if validation.status == FAIL: raise

  df = stable_sort(df, keys=[gameTimeUTC, season_end_yy, game_id])
  folds = make_walkforward_folds(df, config.folds)
  persist_folds(folds)

  Xy = build_features_targets(df, config.features, config.targets)

  results = {}
  for model in config.model_registry:
    preds = backtest_model(model, Xy, folds, config.tuning)
    results[model.name] = preds

  paired = compute_paired_deltas(results, baseline=config.baseline)
  stats  = run_block_bootstrap(paired, config.stats.bootstrap)
  dm     = run_dm_test(paired, config.stats.dm)

  uq = {}
  for model in results:
    uq[model] = conformalize(results[model], folds, config.uq)

  report = build_report(validation, results, stats, dm, uq)
  save_artifacts(report, results, uq, folds, config)
  return report
```

## 12.2 Key modules

- `validate_data()` implements Section 1
- `build_features_targets()` implements Section 2
- `make_walkforward_folds()` implements Section 3
- `nested_tune()` implements Section 4
- `run_block_bootstrap()` + `run_dm_test()` implements Section 5
- `conformalize()` implements Section 6
- `model_registry` implements Section 7/8
- `drift_monitor()` implements Section 10

---

# 13. Final Principle

**A model change is not allowed into production unless it:**

1. Passes the data validation gate
2. Improves out-of-sample accuracy with statistical significance
3. Meets pre-registered practical thresholds
4. Maintains calibrated uncertainty (including conditional bins)
5. Is reproducible end-to-end with saved artifacts

