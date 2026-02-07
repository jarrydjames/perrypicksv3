"""Deliver generated report artifacts to configured channels (Discord webhook)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable



def _read_text(path: Path, max_chars: int = 1800) -> str:
    if not path.exists():
        return f"[missing] {path}"
    txt = path.read_text(encoding="utf-8", errors="replace").strip()
    if len(txt) <= max_chars:
        return txt
    return txt[:max_chars] + "\n... (truncated)"


def post_to_discord(webhook_url: str, title: str, blocks: Iterable[str]) -> None:
    lines = [f"**{title}**", ""]
    for b in blocks:
        lines.append(b)
        lines.append("")
    content = "\n".join(lines)[:3800]

    import requests
    resp = requests.post(webhook_url, json={"content": content}, timeout=15)
    resp.raise_for_status()


def upload_reports_to_s3(report_dir: Path, report_date: str) -> bool:
    bucket = os.getenv("REPORTS_S3_BUCKET")
    prefix = os.getenv("REPORTS_S3_PREFIX", "perrypicksv3/nightly")
    if not bucket:
        return False

    try:
        import boto3
    except Exception:
        print("boto3 not available; skipping S3 upload.")
        return False

    s3 = boto3.client("s3")
    files = ["clv_report.md", "experiment_report.md", "nightly_snapshot.md"]
    for name in files:
        path = report_dir / name
        if not path.exists():
            continue
        key = f"{prefix}/{report_date}/{name}"
        s3.upload_file(str(path), bucket, key)
    print(f"Uploaded nightly reports to s3://{bucket}/{prefix}/{report_date}/")
    return True


def deliver_reports(report_dir: Path, report_date: str) -> None:
    webhook = os.getenv("REPORTS_DISCORD_WEBHOOK_URL") or os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook:
        print("No report webhook configured; skipping Discord delivery.")
        upload_reports_to_s3(report_dir, report_date)
        return

    clv = _read_text(report_dir / "clv_report.md")
    exp = _read_text(report_dir / "experiment_report.md")
    nightly = _read_text(report_dir / "nightly_snapshot.md")

    post_to_discord(
        webhook,
        f"Nightly QoL Reports — {report_date}",
        [
            f"__CLV__\n```md\n{clv}\n```",
            f"__Experiments__\n```md\n{exp}\n```",
            f"__Nightly Snapshot__\n```md\n{nightly}\n```",
        ],
    )
    print("Delivered nightly reports to Discord webhook.")

    upload_reports_to_s3(report_dir, report_date)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    deliver_reports(Path(args.report_dir), args.date)
