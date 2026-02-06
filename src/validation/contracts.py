from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class DatasetContract:
    name: str
    required_columns: Sequence[str]


def load_contract(path: Path) -> DatasetContract:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return DatasetContract(
        name=str(payload.get("name", path.stem)),
        required_columns=list(payload.get("required_columns", [])),
    )


def validate_columns(actual_columns: Iterable[str], contract: DatasetContract) -> List[str]:
    missing = [col for col in contract.required_columns if col not in actual_columns]
    return missing


def validate_parquet(path: Path, contract: DatasetContract) -> None:
    import pandas as pd

    df = pd.read_parquet(path)
    missing = validate_columns(df.columns, contract)
    if missing:
        missing_cols = ", ".join(missing)
        raise ValueError(f"Dataset {contract.name} missing columns: {missing_cols}")
