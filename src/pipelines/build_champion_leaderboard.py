from __future__ import annotations

import argparse
from pathlib import Path


def _safe_import_pandas():
    try:
        import pandas as pd  # type: ignore

        return pd
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pandas is required to build champion leaderboard") from exc


def build_leaderboard(input_csv: Path, output_csv: Path, state: str) -> Path:
    pd = _safe_import_pandas()

    if not input_csv.exists():
        raise FileNotFoundError(f"Fold metrics CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    required = {"model", "mae_total", "mae_margin", "rmse_total", "brier_win"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Input metrics missing required columns: {sorted(missing)}")

    agg = (
        df.groupby("model", as_index=False)
        .agg(
            mae_total=("mae_total", "mean"),
            mae_margin=("mae_margin", "mean"),
            rmse_total=("rmse_total", "mean"),
            brier_win=("brier_win", "mean"),
            stability_std_mae_total=("mae_total", "std"),
            folds=("fold", "nunique") if "fold" in df.columns else ("model", "count"),
        )
    )

    agg["state"] = state
    agg["ece_win"] = float("nan")  # placeholder until explicit probability bucket eval is persisted
    agg = agg.sort_values(["mae_total", "mae_margin", "rmse_total"], ascending=True).reset_index(drop=True)
    agg["rank"] = agg.index + 1

    cols = [
        "state",
        "rank",
        "model",
        "mae_total",
        "mae_margin",
        "rmse_total",
        "ece_win",
        "brier_win",
        "stability_std_mae_total",
        "folds",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    agg[cols].to_csv(output_csv, index=False)
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate fold metrics into canonical champion leaderboard.")
    parser.add_argument("--input", type=Path, required=True, help="Fold-level metrics CSV path")
    parser.add_argument("--output", type=Path, required=True, help="Leaderboard CSV path")
    parser.add_argument("--state", type=str, required=True, help="State label (pregame|halftime|q3)")
    args = parser.parse_args()

    out = build_leaderboard(args.input, args.output, args.state)
    print(f"Champion leaderboard written to: {out}")


if __name__ == "__main__":
    main()
