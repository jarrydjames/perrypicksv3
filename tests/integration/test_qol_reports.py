import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.clv_report import build_report as build_clv_report
from scripts.experiment_report import build_report as build_experiment_report
from scripts.publish_nightly_snapshot import build_snapshot


def _seed_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE clv_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pick_id TEXT NOT NULL,
            game_id TEXT NOT NULL,
            trigger_type TEXT NOT NULL,
            market_type TEXT,
            side TEXT,
            opening_line REAL,
            posted_line REAL,
            closing_line REAL,
            clv_points REAL,
            created_at_utc TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT NOT NULL UNIQUE,
            model_version TEXT,
            calibration_version TEXT,
            bet_policy_version TEXT,
            output_template_version TEXT,
            created_at_utc TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE picks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE games (
            game_id TEXT PRIMARY KEY,
            away_team TEXT,
            home_team TEXT,
            score_away INTEGER,
            score_home INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE miss_explanations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            trigger_type TEXT NOT NULL,
            explanation_bullets_json TEXT NOT NULL,
            created_at_utc TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        INSERT INTO clv_tracking (pick_id, game_id, trigger_type, clv_points, created_at_utc)
        VALUES
            ('p1', 'g1', 'PRE_GAME', 1.5, datetime('now')),
            ('p2', 'g2', 'HALFTIME', -0.5, datetime('now'))
        """
    )
    cur.execute(
        """
        INSERT INTO experiments (experiment_id, model_version, calibration_version, bet_policy_version, output_template_version, created_at_utc)
        VALUES ('exp-a', 'v3', 'cal-a', 'policy-a', 'tpl-a', datetime('now'))
        """
    )
    cur.execute("INSERT INTO picks (experiment_id) VALUES ('exp-a'), ('exp-a')")
    cur.execute("INSERT INTO games (game_id, away_team, home_team, score_away, score_home) VALUES ('g1', 'PHI', 'BOS', 101, 99)")
    cur.execute(
        """
        INSERT INTO miss_explanations (game_id, trigger_type, explanation_bullets_json, created_at_utc)
        VALUES ('g1', 'PRE_GAME', '["What we expected: x", "What changed live: y", "Why this was path deviation (not model collapse): z"]', datetime('now'))
        """
    )

    conn.commit()
    conn.close()


def test_clv_report_includes_trigger_breakdown(tmp_path: Path):
    db = tmp_path / "automation.db"
    _seed_db(db)

    report = build_clv_report(db, days=7)

    assert "CLV Report" in report
    assert "PRE_GAME" in report
    assert "HALFTIME" in report


def test_experiment_report_includes_pick_counts(tmp_path: Path):
    db = tmp_path / "automation.db"
    _seed_db(db)

    report = build_experiment_report(db)

    assert "Experiment Coverage Report" in report
    assert "exp-a" in report
    assert "| 2 |" in report


def test_nightly_snapshot_renders_three_bullets(tmp_path: Path):
    db = tmp_path / "automation.db"
    _seed_db(db)

    report = build_snapshot(db, date_str="2099-01-01")

    # For non-matching date we should still get the heading
    assert "Nightly Snapshot" in report

    from datetime import date
    today_report = build_snapshot(db, date_str=date.today().isoformat())
    assert "What we expected:" in today_report
    assert "What changed live:" in today_report
    assert "Why this was path deviation" in today_report
