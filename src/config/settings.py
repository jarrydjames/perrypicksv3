from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class PipelineArtifacts:
    datasets: List[str]
    models: List[str]
    reports: List[str]


@dataclass(frozen=True)
class PipelineConfig:
    steps: List[str]
    artifacts: PipelineArtifacts


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    models_dir: Path
    processed_dir: Path
    registry_dir: Path
    pipeline_version: str
    pregame_pipeline: PipelineConfig
    halftime_pipeline: PipelineConfig
    q3_pipeline: PipelineConfig


def _load_defaults() -> Dict[str, Any]:
    defaults_path = Path(__file__).with_name("defaults.json")
    with defaults_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_pipeline(entry: Dict[str, Any]) -> PipelineConfig:
    artifacts = entry.get("artifacts", {})
    return PipelineConfig(
        steps=list(entry.get("steps", [])),
        artifacts=PipelineArtifacts(
            datasets=list(artifacts.get("datasets", [])),
            models=list(artifacts.get("models", [])),
            reports=list(artifacts.get("reports", [])),
        ),
    )


def load_settings() -> Settings:
    data = _load_defaults()
    root = Path(data.get("project_root", ".")).resolve()
    return Settings(
        project_root=root,
        data_dir=root / data.get("data_dir", "data"),
        models_dir=root / data.get("models_dir", "data/models"),
        processed_dir=root / data.get("processed_dir", "data/processed"),
        registry_dir=root / data.get("registry_dir", "data/registry"),
        pipeline_version=str(data.get("pipeline_version", "v1")),
        pregame_pipeline=_parse_pipeline(data.get("pregame_pipeline", {})),
        halftime_pipeline=_parse_pipeline(data.get("halftime_pipeline", {})),
        q3_pipeline=_parse_pipeline(data.get("q3_pipeline", {})),
    )
