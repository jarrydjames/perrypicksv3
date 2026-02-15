# Backtest Recency Pipeline Plan (Post `a2e09ee` Review)

## Findings from `a2e09ee`

The ESPN+NBA CDN approach fixed game discovery and ID mapping reliability, but halftime backtest quality remained constrained by one major issue: the script used hard-coded placeholder recency/team-form fields instead of pulling leakage-safe, pre-game recency values from the feature store.

## What was fixed now

`scripts/halftime_backtest_espn.py` has been updated to:

1. Load the historical feature store (`data/processed/halftime_with_temporal_features_total.parquet`) once.
2. Extract `teamId` from NBA CDN boxscore payload.
3. For each team, locate the latest prior game row (< target date) and copy recency features from the correct side (home/away).
4. Fall back to robust historical medians when a specific recency value is unavailable.
5. Keep strict halftime guardrails (Q1/Q2-only in-game features) while improving model context fidelity.

This removes the primary source of train/serve skew in the backtest path.

## Recommended robust pipeline for future backtests

### A. Data freshness gate (must pass before backtest)

Run:

```bash
python -m src.pipelines.champion_refresh_cycle --policy config/champion_refresh_policy_v1.json --out reports/champion_runs/refresh_readiness.json
```

Gate rules:
- Dataset exists
- Dataset has fresh `latest_game_date`
- Leaderboards exist for all states (pregame, halftime, q3)

If `recommendation = block_promotion_and_fix_data`, do not run a benchmark backtest for promotion decisions.

### B. Deterministic feature contract gate

Before scoring:
- Build training feature list via `feature_columns(train_df)`.
- Assert every required feature is either present in backtest inputs or has deterministic fallback logic.
- Fail loudly if any non-approved fallback usage exceeds threshold (e.g., >5% rows).

### C. Leakage-safe recency join

For each game/team:
- Resolve team IDs from authoritative source (NBA CDN boxscore/schedule).
- Pull latest prior row by `game_date < target_date`.
- Map side-aware columns (`home_*` / `away_*`) into standardized prediction frame.

### D. Reproducibility and observability

Persist with every backtest run:
- input date and game IDs
- model params and fold id
- per-feature fallback counts
- per-game prediction trace
- summary metrics JSON + detailed CSV

### E. Promotion policy alignment

Backtest output should plug into `config/champion_refresh_policy_v1.json` gates:
- MAE regression threshold
- calibration quality
- minimum fold/significance requirements

## Dataset update plan for latest data (future games readiness)

### Immediate refresh sequence

1. Pull latest games and caches.
2. Rebuild temporal/recency features.
3. Rebuild merged halftime dataset.
4. Retrain + calibrate.
5. Regenerate leaderboards and readiness report.

Operationally, use the orchestrators already in repo:

```bash
python scripts/run_pipeline.py pregame
python scripts/run_pipeline.py halftime
python scripts/run_pipeline.py q3
python -m src.pipelines.champion_refresh_cycle --policy config/champion_refresh_policy_v1.json --out reports/champion_runs/refresh_readiness.json
```

## Ongoing cadence + automation design

### Recommended cadence

- **Every 6 hours**: drift/readiness check (`champion_refresh_cycle`).
- **Daily (post-games)**: data ingestion + dataset rebuild for all states.
- **Weekly (Mon)**: full retrain + leaderboard rebuild + shadow deploy.
- **Weekly (Thu)**: calibration-only refresh when full retrain threshold is not met.

This aligns with the policy file crons:
- full retrain: `0 7 * * MON`
- calibration: `0 7 * * THU`
- drift checks: `0 */6 * * *`

### Automation implementation options

1. **GitHub Actions first-class pipeline**
   - Add a scheduled workflow dedicated to data refresh/model refresh (separate from nightly QoL reports).
   - Store readiness/report artifacts on each run.
   - Alert when freshness or leaderboard checks fail.

2. **Host cron fallback**
   - Keep `scripts/daily_refresh.sh` for local/VM cron environments.
   - Ensure logs and non-zero exits are wired to alerting.

3. **Promotion safety**
   - Auto-promote only when readiness + policy gates pass.
   - Otherwise keep champion unchanged and produce remediation report.

## Success criteria

- Backtests use real recency features (not placeholders).
- `refresh_readiness.json` reports `ok: true` for all states before promotion.
- Data currency is within holdout/freshness windows.
- Automated cadence executes without manual intervention and logs artifacts for audit.
