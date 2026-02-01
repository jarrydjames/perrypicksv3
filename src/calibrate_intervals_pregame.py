"""Calibrate pregame intervals - follows same methodology as halftime/Q3 calibration.

This script computes q10/q90 quantiles from pregame prediction residuals
to generate 80% confidence bands (same approach as halftime/Q3).
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    """Calibrate pregame intervals using residuals."""
    # Pregame dataset (same structure as halftime/Q3 dataset)
    df = pd.read_parquet("data/processed/pregame_team_v2.parquet").dropna()
    
    # Load pregame trained models (using gbt as default)
    obj = joblib.load("models_v3/pregame/gbt_twohead.joblib")
    
    # Extract features (same for both total and margin)
    features = obj["features"]
    Xt = df[features]
    Xm = df[features]
    
    # Predict on training data to get residuals
    pred_t = obj["total"]["model"].predict(Xt)
    pred_m = obj["margin"]["model"].predict(Xm)
    
    # Compute residuals
    resid_t = df["total"].values - pred_t
    resid_m = df["margin"].values - pred_m
    
    # Compute q10/q90 quantiles (same as halftime/Q3)
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
    Path("models_v3/pregame").mkdir(parents=True, exist_ok=True)
    
    joblib.dump(intervals, "models_v3/pregame/pregame_intervals.joblib")
    
    print("Saved models_v3/pregame/pregame_intervals.joblib")
    print("Total resid q10/q90:", q_t)
    print("Margin resid q10/q90:", q_m)


if __name__ == "__main__":
    main()
