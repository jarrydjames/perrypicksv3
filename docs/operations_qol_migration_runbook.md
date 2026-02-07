# QoL Platform Migration & Operations Runbook

## Purpose
Bootstrap and verify QoL platform tables/modules before first production nightly reporting cycle.

## 1) Preconditions
- Python environment has dependencies installed (`requirements.txt`).
- `AUTOMATION_DB_PATH` points to the production automation DB.
- Scheduler secrets include `REPORTS_DISCORD_WEBHOOK_URL` (or `DISCORD_WEBHOOK_URL`) if auto-delivery is enabled.
- Optional S3 sink: `REPORTS_S3_BUCKET` and `REPORTS_S3_PREFIX` if object storage delivery is desired.
- App/runtime has write access to `reports/`.

## 2) Database migration bootstrap
Run once per environment (or each deploy if safe):

```bash
python -c "from pathlib import Path; from core.storage import init_database; init_database(Path('${AUTOMATION_DB_PATH:-data/automation.db}'))"
```

This ensures the following tables/columns exist:
- `discord_post_dlq`
- `clv_tracking`
- `feature_telemetry`
- `experiments`
- `miss_explanations`
- picks metadata columns (`pick_id`, `confidence_tier`, `interval_low`, `interval_high`, `model_version`, `market_line`, `fair_line`, `experiment_id`)

## 3) Health verification

```bash
python scripts/healthcheck.py
```

Expected keys:
- `db_read_write=true`
- `models_present=true`
- `dlq_backlog_ok=true`
- `degraded_mode=false` (unless intentionally enabled)

## 4) Nightly reports
Generate nightly QoL reports:

```bash
bash scripts/run_nightly_reports.sh
```

Artifacts:
- `reports/clv_report.md`
- `reports/experiment_report.md`
- `reports/nightly_snapshot.md`

## 5) Scheduling
Use one scheduler in production:
- GitHub Actions: `.github/workflows/nightly_reports.yml` (expects repository secrets for DB path + webhooks; can also include S3 env vars).
- or cron/systemd invoking `scripts/run_nightly_reports.sh`

## 6) Incident response
- If DLQ backlog rises above threshold, inspect `discord_post_dlq` rows and retry manually.
- If degraded mode appears, verify upstream odds/API health and environment flag `DEGRADED_MODE`.

## 7) Rollback guidance
- Disable nightly workflow/scheduler.
- Revert to prior application image.
- Keep DB schema changes (additive and backward-compatible).
