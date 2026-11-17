"""
Sensor Data Preprocessing Module

Responsibilities:
1. Load sensor datasets.
2. Validate sensor observations.
3. Split normal data chronologically.
4. Perform feature scaling.
5. Save processed datasets.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# Project directories

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")


# Sensor features

FEATURES = [
    "temperature",
    "vibration",
    "pressure"
]


def load_sensor_data(file_path):
    """Load and validate a sensor CSV dataset."""

    frame = pd.read_csv(file_path)

    missing_columns = set(FEATURES) - set(frame.columns)

    if missing_columns:
        raise ValueError(
            f"Missing sensor columns: {missing_columns}"
        )

    if frame[FEATURES].isnull().any().any():
        raise ValueError(
            "Dataset contains missing sensor values."
        )

    sensor_values = frame[FEATURES].to_numpy(dtype=float)

    if not np.isfinite(sensor_values).all():
        raise ValueError(
            "Dataset contains invalid numerical values."
        )

    return frame


def split_normal_data(frame, train_ratio=0.80):
    """
    Split normal observations chronologically.

    First 80%: Training
    Remaining 20%: Validation
    """

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    split_index = int(len(frame) * train_ratio)

    train_data = frame.iloc[:split_index].copy()

    validation_data = frame.iloc[split_index:].copy()

    return train_data, validation_data


def fit_scaler(train_data):
    """Fit StandardScaler using training observations only."""

    scaler = StandardScaler()

    scaler.fit(
        train_data[FEATURES].to_numpy(dtype=float)
    )

    return scaler


def transform_data(frame, scaler):
    """Transform sensor features using the fitted scaler."""

    values = frame[FEATURES].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Non-finite sensor values detected.")

    return scaler.transform(values)


def save_processed_data(
    train_data,
    validation_data,
    test_data,
    scaler
):
    """Save processed datasets and scaler."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Transform sensor features

    X_train = transform_data(train_data, scaler)

    X_validation = transform_data(validation_data, scaler)

    X_test = transform_data(test_data, scaler)

    # Save numerical arrays

    np.save(PROCESSED_DIR / "X_train.npy", X_train)

    np.save(PROCESSED_DIR / "X_validation.npy", X_validation)

    np.save(PROCESSED_DIR / "X_test.npy", X_test)

    # Preserve original readings and labels

    train_data.to_csv(
        PROCESSED_DIR / "train_data.csv",
        index=False
    )

    validation_data.to_csv(
        PROCESSED_DIR / "validation_data.csv",
        index=False
    )

    test_data.to_csv(
        PROCESSED_DIR / "test_data.csv",
        index=False
    )

    # Save scaler for future incoming sensor readings

    joblib.dump(
        scaler,
        MODELS_DIR / "sensor_scaler.pkl"
    )

    print("\nProcessed datasets saved successfully.")


def prepare_datasets():
    """Execute the complete preprocessing pipeline."""

    print("\nLoading sensor datasets...")

    normal_data = load_sensor_data(
        RAW_DIR / "normal_sensor_data.csv"
    )

    test_data = load_sensor_data(
        RAW_DIR / "test_sensor_data.csv"
    )

    print("Normal dataset:", normal_data.shape)

    print("Test dataset:", test_data.shape)

    # Chronological splitting

    train_data, validation_data = split_normal_data(
        normal_data
    )

    print("\nDataset splitting completed.")

    print("Training observations:", len(train_data))

    print("Validation observations:", len(validation_data))

    # Fit scaler only on normal training data

    scaler = fit_scaler(train_data)

    print("\nFeature scaling completed.")

    print("Scaler mean:", scaler.mean_)

    print("Scaler scale:", scaler.scale_)

    # Save processed datasets

    save_processed_data(
        train_data,
        validation_data,
        test_data,
        scaler
    )

    print("\nPREPROCESSING COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    prepare_datasets()