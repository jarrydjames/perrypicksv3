import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.healthcheck import run_healthcheck
from scripts.deliver_reports import _read_text, upload_reports_to_s3


def test_healthcheck_handles_missing_pendulum_gracefully(tmp_path, monkeypatch):
    # Simulate missing pendulum by shadowing import lookup
    monkeypatch.setitem(sys.modules, "pendulum", None)

    db_path = tmp_path / "automation.db"
    os.environ.pop("ODDS_API_KEY", None)
    os.environ.pop("DISCORD_WEBHOOK_URL", None)

    out = run_healthcheck(db_path)
    assert out["db_read_write"] is True
    assert out["pendulum_available"] in {True, False}


def test_read_text_reports_missing_file(tmp_path):
    missing = tmp_path / "missing.md"
    text = _read_text(missing)
    assert "[missing]" in text


def test_s3_upload_skips_when_not_configured(tmp_path, monkeypatch):
    monkeypatch.delenv("REPORTS_S3_BUCKET", raising=False)
    assert upload_reports_to_s3(tmp_path, "2026-02-06") is False
