from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.pregame.orchestrate import run_pregame_pipeline
from src.pipelines.halftime.orchestrate import run_halftime_pipeline
from src.pipelines.q3.orchestrate import run_q3_pipeline


PIPELINES = {
    "pregame": run_pregame_pipeline,
    "halftime": run_halftime_pipeline,
    "q3": run_q3_pipeline,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PerryPicks pipelines")
    parser.add_argument("pipeline", choices=PIPELINES.keys())
    args = parser.parse_args()

    PIPELINES[args.pipeline]()


if __name__ == "__main__":
    main()
