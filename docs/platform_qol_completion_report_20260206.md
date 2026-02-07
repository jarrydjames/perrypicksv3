# Platform QoL Completion Report

## Implemented in code
- Ops dashboard module: `src/ui/ops_dashboard.py` and app navigation wiring in `app_v3.py`.
- Healthcheck CLI expanded with DLQ backlog threshold and degraded-mode signal.
- CLV reporting script: `scripts/clv_report.py`.
- Experiment coverage script: `scripts/experiment_report.py`.
- Nightly snapshot publisher: `scripts/publish_nightly_snapshot.py`.
- Nightly scheduler entrypoint: `scripts/run_nightly_reports.sh`.
- GitHub Actions nightly workflow: `.github/workflows/nightly_reports.yml`.
- Degraded-mode guard helper and runner wiring to label Discord outputs when upstream data is limited.
- Integration tests for seeded SQLite fixtures covering CLV, experiment coverage, and nightly snapshot outputs.
- Migration/operations runbook for bootstrap + verification.

## Remaining work
1. Configure real repository/environment secrets in target deployment (`AUTOMATION_DB_PATH`, `REPORTS_DISCORD_WEBHOOK_URL` / `DISCORD_WEBHOOK_URL`, and optional `REPORTS_S3_BUCKET` / `REPORTS_S3_PREFIX`).
2. Optional: add email delivery sink if required (Discord + optional S3 are now supported).
