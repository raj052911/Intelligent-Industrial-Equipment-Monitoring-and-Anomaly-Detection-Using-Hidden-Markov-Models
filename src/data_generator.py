"""Generate reproducible simulated machine sensor sequences."""

from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ["temperature", "vibration", "pressure"]

def generate_normal_data(n_samples=3000, seed=42):
    rng = np.random.default_rng(seed)
    time = np.arange(n_samples)

    temperature = 65 + 3 * np.sin(time / 90) + rng.normal(0, 0.8, n_samples)
    vibration = 0.35 + 0.06 * np.sin(time / 35) + rng.normal(0, 0.025, n_samples)
    pressure = 100 + 2 * np.sin(time / 120) + rng.normal(0, 0.7, n_samples)

    return pd.DataFrame({
        "timestamp": time,
        "temperature": temperature,
        "vibration": vibration,
        "pressure": pressure,
        "is_anomaly": 0
    })

def generate_test_data(n_samples=1500, seed=123):
    rng = np.random.default_rng(seed)
    frame = generate_normal_data(n_samples, seed)
    frame["temperature"] += rng.normal(0, 0.4, n_samples)

    # Inject several abnormal operating intervals.
    intervals = [(300, 360), (750, 820), (1150, 1210)]

    for start, end in intervals:
        frame.loc[start:end, "temperature"] += 12
        frame.loc[start:end, "vibration"] += 0.45
        frame.loc[start:end, "pressure"] -= 12
        frame.loc[start:end, "is_anomaly"] = 1

    return frame

def save_datasets(output_dir="data/raw"):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    generate_normal_data().to_csv(path / "normal_sensor_data.csv", index=False)
    generate_test_data().to_csv(path / "test_sensor_data.csv", index=False)

    return path
