"""Calibrate halftime intervals for models_v3 outputs.

Computes q10/q90 quantiles from halftime residuals to generate 80% bands.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main() -> None:
    df = pd.read_parquet("data/processed/halftime_team_v2.parquet").dropna(subset=["h2_total", "h2_margin"])

    obj = joblib.load("models_v3/halftime/gbt_twohead.joblib")
    features = obj["features"]

    Xt = df[features]
    Xm = df[features]

    pred_t = obj["total"]["model"].predict(Xt)
    pred_m = obj["margin"]["model"].predict(Xm)

    resid_t = df["h2_total"].values - pred_t
    resid_m = df["h2_margin"].values - pred_m

    q_t = np.quantile(resid_t, [0.1, 0.9])
    q_m = np.quantile(resid_m, [0.1, 0.9])

    intervals = {
        "resid_total_q10": float(q_t[0]),
        "resid_total_q90": float(q_t[1]),
        "resid_margin_q10": float(q_m[0]),
        "resid_margin_q90": float(q_m[1]),
    }

    Path("models_v3/halftime").mkdir(parents=True, exist_ok=True)
    joblib.dump(intervals, "models_v3/halftime/halftime_intervals.joblib")

    print("Saved models_v3/halftime/halftime_intervals.joblib")
    print("Total resid q10/q90:", q_t)
    print("Margin resid q10/q90:", q_m)


if __name__ == "__main__":
    main()
