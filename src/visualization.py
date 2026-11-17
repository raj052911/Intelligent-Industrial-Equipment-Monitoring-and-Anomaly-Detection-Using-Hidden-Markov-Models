from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")

SENSOR_COLUMNS = [
    "temperature",
    "vibration",
    "pressure"
]


def plot_anomalies(
    data_path="data/processed/detection_results.csv",
    output_path="results/anomaly_detection_plot.png"
):
    """
    Visualize sensor readings and detected anomaly regions.
    """

    data = pd.read_csv(data_path)

    required_columns = (
        SENSOR_COLUMNS + ["predicted_anomaly"]
    )

    missing = set(required_columns) - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamps = (
        data["timestamp"]
        if "timestamp" in data.columns
        else np.arange(len(data))
    )

    anomaly_mask = (
        data["predicted_anomaly"].to_numpy(dtype=bool)
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15, 11),
        sharex=True
    )

    for axis, column in zip(axes, SENSOR_COLUMNS):

        axis.plot(
            timestamps,
            data[column],
            label=column.capitalize(),
            linewidth=1
        )

        axis.fill_between(
            np.arange(len(data)),
            data[column].min(),
            data[column].max(),
            where=anomaly_mask,
            alpha=0.25,
            color="red",
            label="Detected Anomaly"
        )

        axis.set_ylabel(column.capitalize())
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right")

    axes[0].set_title(
        "HMM-Based Industrial Sensor Anomaly Detection"
    )

    axes[-1].set_xlabel("Sensor Reading")

    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()

    print("Visualization generated successfully.")
    print("Saved at:", output_path)


if __name__ == "__main__":
    plot_anomalies()
