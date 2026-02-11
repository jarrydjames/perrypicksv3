# Vibe Platform Execution Guide (Champion Pipeline)

This is the push-to-run checklist for executing the champion system cleanly on a vibe-coding platform.

## One-command entrypoint

```bash
scripts/vibe_run_champion_pipeline.sh --dry-run --skip-checks
```


> Note: In local/dev dry-run mode (`--dry-run --skip-checks`), preflight failures are treated as non-blocking so orchestration wiring can still be validated. Production runs remain strict.

For real execution:

```bash
scripts/vibe_run_champion_pipeline.sh
```

For gated promotion:

```bash
scripts/vibe_run_champion_pipeline.sh --promote
```

## What this command does

1. Runs preflight environment checks via `scripts/vibe_preflight.py`.
2. Runs recurring readiness + champion testing via `src/pipelines/run_champion_ops_cycle.py`.

## Required platform capabilities

- Python 3.10+
- Installed modules: `pandas`, `numpy`, `pyarrow`, `sklearn`
- Write access to `reports/champion_runs/`
- Access to repository datasets in `data/processed/`

## Success artifacts

- `reports/champion_runs/preflight.json`
- `reports/champion_runs/refresh_readiness.json`
- `reports/champion_runs/latest/run_report.json`
- `reports/champion_runs/latest/champion_candidates.json`

If `--promote` is used and all gates pass:

- `data/processed/champion_models.json`

## CI suggestion (recommended)

Wire this command in your platform build/deploy pipeline:

```bash
scripts/vibe_run_champion_pipeline.sh --dry-run --skip-checks
```

and fail the workflow if exit code is non-zero.
