#!/usr/bin/env python3
"""Check local Python environment for halftime backtest dependencies."""

from __future__ import annotations

import importlib.util
import sys

REQUIRED = [
    "numpy",
    "pandas",
    "requests",
    "scipy",
    "sklearn",
    "catboost",
    "pyarrow",
]


def main() -> int:
    missing = [pkg for pkg in REQUIRED if importlib.util.find_spec(pkg) is None]
    if missing:
        print("❌ Missing required packages for scripts/halftime_backtest_espn.py:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nInstall project dependencies in the intended virtualenv, then rerun.")
        return 1

    print("✅ Environment check passed for halftime backtest dependencies.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
