# Model Validation Audit (All 7 Models × Pregame/Halftime/Q3)

**Date:** 2026-02-11  
**Scope:** Review model testing, calibration, and backtesting process across all seven candidate models for all three states (pregame, halftime, Q3).

---

## Executive assessment

Your concern is valid. The current documentation and pipelines are **not internally consistent**, and the champion-selection process appears to mix:

- different model pools (sometimes 3 models, sometimes 7),
- different validation designs (random split, walk-forward, readout parsing), and
- in at least one place, **hard-coded assumptions** that can force a narrative.

This means the reported champion outcomes (especially extremely high pregame winner accuracy) are at high risk of being overstated or non-reproducible.

---

## Key flaws found

## 1) “All models tested” claims conflict with what was actually compared

- The comprehensive report states all models were trained/backtested, but the state-level champion summary still lists only Ridge/RF/GBT for pregame and Q3.  
- Halftime includes a 7-model sweep, but pregame/Q3 championing still references the 3-model readouts.

**Why this matters:** champion selection is not apples-to-apples across all three states.

---

## 2) Champion compilation script is structurally biased toward a limited model set

The `compile_comprehensive_results.py` process:

- parses only `Ridge`, `Random Forest`, `GBT`, and `Ridge (Agg)` from readouts,
- maps champions only to `{ridge, random_forest, gbt}` model files,
- includes hard-coded ranking text (e.g., “Ridge is best for all states”).

**Why this matters:** even if a 7-model run produced stronger non-Ridge models, this compiler can still output Ridge-centric conclusions.

---

## 3) Random train/test splits are used in key 7-model scripts

Both 7-model training scripts use randomized `train_test_split(..., random_state=42)` instead of strict temporal walk-forward evaluation.

**Why this matters:** NBA data is time-dependent; random splits can leak future distributional information and inflate estimates relative to real deployment.

---

## 4) Material contradictions in model quality across docs/logs

Example contradiction:

- One “champion” narrative gives pregame Ridge MAE around **3.5** with very high R².
- A separate 7-model run log shows pregame total MAE roughly **~9.6–10.4** for top models on that run configuration.

These are too far apart to treat as a normal variance effect. They likely reflect different data definitions, targets, leakage controls, or evaluation code paths.

---

## 5) 90.9% pregame winner accuracy is based on a very small sample (33 games)

A 30/33 hit rate can occur by chance in short windows, especially if game mix is skewed (heavy favorites, same-day correlation, selection effects).

**Why this matters:** this is not enough evidence to conclude true >90% winner skill. You need confidence intervals and larger rolling OOS windows.

---

## 6) ROI claims are likely optimistic and not robust enough for model selection

Some backtest summaries show very high ROI with 100% positive folds, which is unusual in market-like settings.

Potential causes:
- no realistic slippage/line movement/limits,
- no robust odds/line availability checks at prediction timestamp,
- threshold/selection rules tuned on same backtest windows.

---

## 7) Calibration process is narrow and not integrated into champion criteria

Current pipeline calibrates XGBoost intervals in one flow, while championing often uses MAE-only logic and separate report pipelines.

**Why this matters:** model ranking should include calibration quality (ECE/Brier/coverage) and decision quality, not only point error.

---

## Recommended retest protocol (robust champion selection)

## A) Freeze data + reproducibility baseline

1. Create immutable dataset manifests per state (pregame/halftime/Q3): row count, date range, feature list, hash.  
2. Version all feature-generation code and model config in one manifest JSON.  
3. For each run, persist: git SHA, command, seed, dependency versions.

---

## B) Single evaluation design for all states and all 7 models

Use one common protocol:

- **Outer loop:** rolling walk-forward folds (time-ordered, no shuffling).
- **Inner loop:** temporal CV for hyperparameter tuning only on training window.
- **Models per state:** all seven (Ridge, RF, XGBoost, MLP, ElasticNet, GBT, LightGBM).
- **Targets:** both total and margin in each state.

Suggested outer folds:
- expanding train, fixed test horizon (e.g., 200 games) with step size 100–200.
- keep final untouched holdout block (e.g., most recent 10–15% of games) used once.

---

## C) Primary metrics and champion rule

For each state/target, select champion by:

1. **Primary:** mean fold MAE on outer folds (and median MAE for robustness),
2. **Tie-breakers:** RMSE, tail-risk metric (e.g., P90 absolute error),
3. **Significance:** paired test on fold errors (Diebold-Mariano or paired bootstrap),
4. **Stability:** std across folds + worst-fold performance.

If top models are statistically indistinguishable, prefer:
- better calibration,
- lower complexity/latency,
- better robustness on latest holdout.

---

## D) Winner-probability calibration and reporting

For winner/spread/total-over decisions:

- Convert model outputs to probabilities,
- Calibrate with isotonic or Platt scaling using only training folds,
- Report **Brier**, **ECE**, reliability plots,
- Track coverage at multiple confidence bins.

Do not report raw hit rate without uncertainty bands.

For the 30/33 result, include Wilson CI (roughly very wide around this estimate) and avoid “true 90%+” claims.

---

## E) Market realism for betting metrics

If using ROI in selection:

- enforce timestamp-valid lines,
- include vig, line movement assumptions, and bet delay,
- cap stake and add bankroll/risk controls,
- report number of bets and turnover,
- never use ROI alone as champion criterion.

---

## F) Deployment process

1. Promote one champion per state/target from the robust protocol.  
2. Run shadow mode for 2–4 weeks (no production reliance).  
3. Track drift and rolling degradation.  
4. Trigger automatic re-evaluation if MAE/calibration crosses thresholds.

---

## Immediate next actions (high priority)

1. **Unify truth source:** retire compiler paths that hard-code Ridge-centric summaries.  
2. **Re-run all 7 models** for pregame/halftime/Q3 under the same temporal nested protocol.  
3. **Publish one canonical leaderboard** (all states, all models, all targets) from a single result artifact.  
4. **Recompute and contextualize 90.9% claim** with confidence intervals and larger OOS sample.  
5. **Gate champion promotion** on both accuracy and calibration stability.

---

## Expected outcome if you follow this

- You’ll likely see lower but more believable winner accuracy for pregame.
- Champion picks may differ by state/target (and possibly by operational objective).
- Reported results will be reproducible, defensible, and much harder to accidentally overstate.

