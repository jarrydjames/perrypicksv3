# Data Freshness + Automation Guardrails (System-Wide)

This update hardens the full pipeline (not only the ESPN halftime test) with enforceable data-quality gates before model refresh decisions.

## What changed

1. Added `config/data_freshness_policy_v1.json` to define strict dataset expectations for pregame/halftime/q3.
2. Added `src/data/data_freshness_audit.py` to audit:
   - dataset existence,
   - row minimums,
   - date freshness windows,
   - required seasons (including 25 and 26),
   - current-season row minimums,
   - required recency feature coverage,
   - null-rate thresholds on recency fields.
3. Upgraded `src/pipelines/champion_refresh_cycle.py` to run the data-freshness audit and include it in refresh-readiness recommendations.
4. Updated `scripts/daily_refresh.sh` to execute strict quality gates after deploy and fail pipeline when freshness/coverage conditions are not met.

## Failsafes now in place

- **Season coverage check**: fails if required seasons are missing from audited datasets.
- **Freshness check**: fails when latest game dates are stale beyond policy limits.
- **Feature coverage check**: fails when key recency features are missing or too null-heavy.
- **Promotion block**: refresh readiness recommendation remains `block_promotion_and_fix_data` when audits fail.

## Operations commands

Run data audit directly:

```bash
python src/data/data_freshness_audit.py \
  --policy config/data_freshness_policy_v1.json \
  --out reports/champion_runs/data_freshness_audit.json \
  --strict
```

Run combined readiness + audit:

```bash
python src/pipelines/champion_refresh_cycle.py \
  --policy config/champion_refresh_policy_v1.json \
  --data-policy config/data_freshness_policy_v1.json \
  --out reports/champion_runs/refresh_readiness.json \
  --data-audit-out reports/champion_runs/data_freshness_audit.json
```

Run daily pipeline with quality gates:

```bash
./scripts/daily_refresh.sh
```


## New full-structure automation

- Added `scripts/full_refresh_with_gates.sh` to run **pregame + halftime + q3** pipelines before strict data quality gates.
- Added `.github/workflows/model_refresh_guardrails.yml` to schedule recurring execution (6-hour drift cadence + Monday retrain + Thursday calibration cadence) with artifact upload of readiness/audit reports.

Run full structure locally:

```bash
bash scripts/full_refresh_with_gates.sh --dry-run
```
