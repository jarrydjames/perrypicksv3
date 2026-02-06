from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation.contracts import DatasetContract, load_contract, validate_columns


def test_validate_columns() -> None:
    contract = DatasetContract(name="test", required_columns=["a", "b"])
    missing = validate_columns(["a", "c"], contract)
    assert missing == ["b"]


def test_load_contract(tmp_path: Path) -> None:
    payload = {"name": "demo", "required_columns": ["x", "y"]}
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    contract = load_contract(path)
    assert contract.name == "demo"
    assert list(contract.required_columns) == ["x", "y"]
