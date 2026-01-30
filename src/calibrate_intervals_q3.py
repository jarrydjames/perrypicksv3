"""Calibrate Q3 intervals - follows same methodology as halftime calibration.

This script computes q10/q90 quantiles from Q3 prediction residuals
to generate 80% confidence bands (same approach as v2 halftime calibration).
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    """Calibrate Q3 intervals using residuals."""
    # Q3 dataset (same structure as halftime dataset)
    df = pd.read_parquet("data/processed/q3_team_v2.parquet").dropna()
    
    # Load Q3 trained models
    obj_t = joblib.load("models_v3/q3/q3_total_twohead.joblib")
    obj_m = joblib.load("models_v3/q3/q3_margin_twohead.joblib")
    
    Xt = df[obj_t["features"]]
    Xm = df[obj_m["features"]]
    
    # Predict on training data to get residuals
    pred_t = obj_t["model"].predict(Xt)
    pred_m = obj_m["model"].predict(Xm)
    
    # Compute residuals
    resid_t = df["q3_total"].values - pred_t
    resid_m = df["q3_margin"].values - pred_m
    
    # Compute q10/q90 quantiles (same as halftime)
    q_t = np.quantile(resid_t, [0.1, 0.9])
    q_m = np.quantile(resid_m, [0.1, 0.9])
    
    # Save calibrated intervals
    intervals = {
        "resid_total_q10": float(q_t[0]),
        "resid_total_q90": float(q_t[1]),
        "resid_margin_q10": float(q_m[0]),
        "resid_margin_q90": float(q_m[1]),
    }
    
    # Ensure models directory exists
    Path("models_v3/q3").mkdir(parents=True, exist_ok=True)
    
    joblib.dump(intervals, "models_v3/q3/q3_intervals.joblib")
    
    print("Saved models_v3/q3/q3_intervals.joblib")
    print("Total resid q10/q90:", q_t)
    print("Margin resid q10/q90:", q_m)


if __name__ == "__main__":
    main()
