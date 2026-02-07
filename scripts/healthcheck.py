"""Platform health check CLI."""

from pathlib import Path
import sys
import os
import sqlite3
import json
import argparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.env import load_environment


REQUIRED_ENV_KEYS = ["ODDS_API_KEY", "DISCORD_WEBHOOK_URL"]


def _default_require_env() -> bool:
    return os.getenv("HEALTHCHECK_REQUIRE_ENV", "true").strip().lower() in {"1", "true", "yes"}


def run_healthcheck(db_path: Path, require_env: bool | None = None) -> dict:
    if require_env is None:
        require_env = _default_require_env()

    loaded_env_path = load_environment()

    results = {
        "db_read_write": False,
        "env_complete": False,
        "models_present": False,
        "api_configured": False,
        "dlq_backlog_ok": False,
        "degraded_mode": False,
        "pendulum_available": False,
        "require_env": require_env,
        "env_file_loaded": str(loaded_env_path) if loaded_env_path else None,
    }

    try:
        import pendulum  # noqa: F401

        results["pendulum_available"] = True
    except Exception:
        results["pendulum_available"] = False

    if results["pendulum_available"]:
        from core.storage import init_database

        init_database(db_path)
    else:
        db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT 1")
    cur.fetchone()
    results["db_read_write"] = True

    dlq_limit = int(os.getenv("DLQ_BACKLOG_LIMIT", "25"))
    try:
        cur.execute("SELECT COUNT(*) FROM discord_post_dlq")
        dlq_count = int(cur.fetchone()[0])
    except sqlite3.Error:
        dlq_count = 0
    results["dlq_backlog_ok"] = dlq_count <= dlq_limit

    results["degraded_mode"] = os.getenv("DEGRADED_MODE", "0") in {"1", "true", "TRUE"}
    conn.close()

    env_present = all(os.getenv(k) for k in REQUIRED_ENV_KEYS)
    results["api_configured"] = env_present
    results["env_complete"] = env_present if require_env else True

    model_paths = [Path("models_v3/pregame"), Path("models_v3/halftime"), Path("models_v3/q3")]
    results["models_present"] = all(p.exists() for p in model_paths)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run platform health checks")
    parser.add_argument(
        "--db-path",
        default=os.getenv("AUTOMATION_DB_PATH", "data/automation.db"),
        help="Path to automation SQLite DB",
    )
    parser.add_argument(
        "--require-env",
        dest="require_env",
        action="store_true",
        help="Fail the healthcheck when required API env vars are missing",
    )
    parser.add_argument(
        "--no-require-env",
        dest="require_env",
        action="store_false",
        help="Do not fail the healthcheck when API env vars are missing",
    )
    parser.set_defaults(require_env=None)

    args = parser.parse_args()
    out = run_healthcheck(Path(args.db_path), require_env=args.require_env)
    print(json.dumps(out, indent=2, sort_keys=True))

    must_pass = ["db_read_write", "models_present", "dlq_backlog_ok", "pendulum_available"]
    if out["require_env"]:
        must_pass.extend(["env_complete", "api_configured"])

    return 0 if all(out.get(k) for k in must_pass) else 1


if __name__ == "__main__":
    raise SystemExit(main())
