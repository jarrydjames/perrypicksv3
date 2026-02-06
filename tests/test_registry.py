from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.registry.artifacts import ArtifactRegistry


def test_registry_records(tmp_path: Path) -> None:
    registry = ArtifactRegistry(tmp_path)
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("data", encoding="utf-8")

    record = registry.record_dataset("artifact", artifact, "v1", notes="test")

    assert record.kind == "dataset"
    assert record.name == "artifact"
    assert record.version == "v1"
    assert record.sha256 is not None
    assert (tmp_path / "artifacts.jsonl").exists()
