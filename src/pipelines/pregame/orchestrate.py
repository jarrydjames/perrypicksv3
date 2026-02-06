from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

from src.config.settings import load_settings
from src.registry.artifacts import ArtifactRegistry
from src.validation.contracts import load_contract, validate_parquet


def _run_steps(steps: List[str]) -> None:
    for step in steps:
        completed = subprocess.run(step, shell=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"Pipeline step failed: {step}")


def run_pregame_pipeline() -> None:
    settings = load_settings()
    pipeline = settings.pregame_pipeline

    if not pipeline.steps:
        raise RuntimeError("No pipeline steps configured for pregame pipeline.")

    _run_steps(pipeline.steps)

    registry = ArtifactRegistry(settings.registry_dir)
    version = settings.pipeline_version

    for dataset_path in pipeline.artifacts.datasets:
        path = settings.project_root / dataset_path
        if not path.exists():
            raise RuntimeError(f"Expected dataset not found: {path}")
        registry.record_dataset(path.name, path, version)

    for model_path in pipeline.artifacts.models:
        path = settings.project_root / model_path
        if not path.exists():
            raise RuntimeError(f"Expected model not found: {path}")
        registry.record_model(path.name, path, version)

    for report_path in pipeline.artifacts.reports:
        path = settings.project_root / report_path
        if path.exists():
            registry.record_report(path.name, path, version)

    contract_path = settings.project_root / "data/contracts/pregame_team_v2.json"
    dataset_path = settings.project_root / "data/processed/pregame_team_v2.parquet"
    if contract_path.exists() and dataset_path.exists():
        contract = load_contract(contract_path)
        validate_parquet(dataset_path, contract)


if __name__ == "__main__":
    run_pregame_pipeline()
