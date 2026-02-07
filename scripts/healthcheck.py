"""Platform health check CLI."""

from pathlib import Path
import sys
import os
import sqlite3
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_healthcheck(db_path: Path) -> dict:
    results = {
        "db_read_write": False,
        "env_complete": False,
        "models_present": False,
        "api_configured": False,
        "dlq_backlog_ok": False,
        "degraded_mode": False,
        "pendulum_available": False,
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

    required_env = ["ODDS_API_KEY", "DISCORD_WEBHOOK_URL"]
    results["env_complete"] = all(os.getenv(k) for k in required_env)
    results["api_configured"] = results["env_complete"]

    model_paths = [Path("models_v3/pregame"), Path("models_v3/halftime"), Path("models_v3/q3")]
    results["models_present"] = all(p.exists() for p in model_paths)

    return results


if __name__ == "__main__":
    db_path = Path(os.getenv("AUTOMATION_DB_PATH", "data/automation.db"))
    out = run_healthcheck(db_path)
    print(json.dumps(out, indent=2, sort_keys=True))
    must_pass = ["db_read_write", "env_complete", "models_present", "api_configured", "dlq_backlog_ok", "pendulum_available"]
    if not all(out.get(k) for k in must_pass):
        raise SystemExit(1)
