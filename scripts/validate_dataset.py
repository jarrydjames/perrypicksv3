from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation.contracts import load_contract, validate_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate dataset against a contract")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("contract", type=Path)
    args = parser.parse_args()

    contract = load_contract(args.contract)
    validate_parquet(args.dataset, contract)
    print(f"OK: {args.dataset} satisfies {contract.name} contract")


if __name__ == "__main__":
    main()
