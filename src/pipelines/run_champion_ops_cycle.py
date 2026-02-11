from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def _print_result(step: str, cp: subprocess.CompletedProcess) -> None:
    print(f"\n[{step}] return_code={cp.returncode}")
    if cp.stdout.strip():
        print(cp.stdout.strip())
    if cp.stderr.strip():
        print(cp.stderr.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run refresh readiness + canonical champion testing in one operational cycle.")
    parser.add_argument("--testing-config", default="config/champion_testing_v1.json")
    parser.add_argument("--refresh-policy", default="config/champion_refresh_policy_v1.json")
    parser.add_argument("--refresh-report", default="reports/champion_runs/refresh_readiness.json")
    parser.add_argument("--promote", action="store_true", help="Promote champions when all checks pass")
    parser.add_argument("--dry-run", action="store_true", help="Dry run champion testing")
    parser.add_argument("--skip-checks", action="store_true", help="Skip dataframe checks in champion testing")
    args = parser.parse_args()

    refresh_cmd = [
        sys.executable,
        "src/pipelines/champion_refresh_cycle.py",
        "--policy",
        str(args.refresh_policy),
        "--out",
        str(args.refresh_report),
    ]
    if args.skip_checks:
        refresh_cmd.append("--skip-checks")
    cp_refresh = _run(refresh_cmd)
    _print_result("refresh-readiness", cp_refresh)
    if cp_refresh.returncode != 0:
        raise SystemExit(cp_refresh.returncode)

    refresh_payload = json.loads(Path(args.refresh_report).read_text(encoding="utf-8"))
    recommendation = str(refresh_payload.get("recommendation", "unknown"))
    print(f"\n[refresh-readiness] recommendation={recommendation}")

    test_cmd = [
        sys.executable,
        "src/pipelines/champion_e2e.py",
        "--config",
        str(args.testing_config),
    ]
    if args.dry_run:
        test_cmd.append("--dry-run")
    if args.skip_checks:
        test_cmd.append("--skip-checks")

    # Only promote when explicit flag is set.
    if args.promote:
        test_cmd.append("--promote")

    cp_test = _run(test_cmd)
    _print_result("champion-testing", cp_test)
    if cp_test.returncode != 0:
        raise SystemExit(cp_test.returncode)

    print("\nChampion ops cycle completed successfully.")
    print("Next step: inspect reports/champion_runs/latest/run_report.json before production deploy.")


if __name__ == "__main__":
    main()
