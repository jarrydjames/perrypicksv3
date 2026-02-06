from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ArtifactRecord:
    kind: str
    name: str
    path: str
    version: str
    sha256: Optional[str]
    created_at: str
    metadata: Dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ArtifactRegistry:
    def __init__(self, registry_dir: Path) -> None:
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / "artifacts.jsonl"

    def record(
        self,
        *,
        kind: str,
        name: str,
        path: Path,
        version: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactRecord:
        record = ArtifactRecord(
            kind=kind,
            name=name,
            path=str(path),
            version=version,
            sha256=_sha256(path),
            created_at=_utc_now(),
            metadata=dict(metadata or {}),
        )
        with self.registry_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.__dict__, sort_keys=True) + "\n")
        return record

    def record_dataset(self, name: str, path: Path, version: str, **metadata: Any) -> ArtifactRecord:
        return self.record(kind="dataset", name=name, path=path, version=version, metadata=metadata)

    def record_model(self, name: str, path: Path, version: str, **metadata: Any) -> ArtifactRecord:
        return self.record(kind="model", name=name, path=path, version=version, metadata=metadata)

    def record_report(self, name: str, path: Path, version: str, **metadata: Any) -> ArtifactRecord:
        return self.record(kind="report", name=name, path=path, version=version, metadata=metadata)
