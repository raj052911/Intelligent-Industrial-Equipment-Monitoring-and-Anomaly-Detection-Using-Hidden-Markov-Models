from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.anomaly_detection import (
    detect_anomalies,
    expand_window_flags
)

from src.threshold_calculation import calculate_threshold


def run_detection():

    print("Starting HMM Anomaly Detection...")

    # Load trained HMM model
    model = joblib.load("models/best_hmm_model.pkl")

    # Load normal training data for threshold calibration
    X_train = np.load("data/processed/X_train.npy")

    # Load test data
    X_test = np.load("data/processed/X_test.npy")

    # Calculate anomaly threshold
    threshold, normal_scores = calculate_threshold(
        model,
        X_train,
        window_size=50,
        percentile=5
    )

    print(f"Anomaly Threshold: {threshold:.4f}")

    # Detect anomalies
    results = detect_anomalies(
        model,
        X_test,
        threshold,
        window_size=50
    )

    # Convert window-level results to sample-level results
    sample_flags = expand_window_flags(
        results,
        total_samples=len(X_test)
    )

    # Load original test dataset to retain timestamps and labels
    test_frame = pd.read_csv(
        "data/processed/test_data.csv"
    )

    test_frame["anomaly_score"] = np.nan
    test_frame["predicted_anomaly"] = sample_flags.astype(int)

    # Store each window score at its starting sample
    test_frame.loc[
        results["window_starts"],
        "anomaly_score"
    ] = results["scores"]

    # Save detection results
    output_path = Path("data/processed/detection_results.csv")

    test_frame.to_csv(output_path, index=False)

    print("\nDETECTION COMPLETED")
    print("Total sensor readings:", len(test_frame))
    print("Detected anomalous readings:", int(sample_flags.sum()))
    print("Anomaly windows:", int(results["anomaly_flags"].sum()))
    print("Saved results:", output_path)


if __name__ == "__main__":
    run_detection()
