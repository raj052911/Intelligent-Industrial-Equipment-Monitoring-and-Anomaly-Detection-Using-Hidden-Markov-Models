import numpy as np


def detect_anomalies(
    model,
    observations,
    threshold,
    window_size=50
):
    """
    Detect anomalies using sliding-window HMM log-likelihood.
    """

    if len(observations) < window_size:
        raise ValueError("Observations are smaller than window size.")

    scores = []
    anomaly_flags = []
    window_starts = []

    for start in range(len(observations) - window_size + 1):

        window = observations[start:start + window_size]

        score = model.score(window) / window_size

        is_anomaly = score < threshold

        scores.append(score)
        anomaly_flags.append(is_anomaly)
        window_starts.append(start)

    return {
        "scores": np.asarray(scores),
        "anomaly_flags": np.asarray(anomaly_flags),
        "window_starts": np.asarray(window_starts),
        "threshold": threshold,
        "window_size": window_size
    }


def expand_window_flags(
    detection_results,
    total_samples,
    min_anomalous_windows=10
):
    """
    Convert window-level anomaly flags into sample-level flags.

    A reading is classified as anomalous only when it belongs
    to at least min_anomalous_windows anomalous windows.
    """

    window_size = detection_results["window_size"]

    anomaly_counts = np.zeros(total_samples, dtype=int)

    for start, is_anomaly in zip(
        detection_results["window_starts"],
        detection_results["anomaly_flags"]
    ):

        if is_anomaly:
            anomaly_counts[start:start + window_size] += 1

    sample_flags = anomaly_counts >= min_anomalous_windows

    return sample_flags
