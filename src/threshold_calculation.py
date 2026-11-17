import numpy as np


def calculate_threshold(
    model_or_scores,
    normal_data=None,
    window_size=50,
    percentile=5
):
    """
    Calculate an anomaly threshold.

    Supports two modes:

    1. Calculate threshold directly from existing scores.
    2. Calculate threshold using HMM normal sensor windows.
    """

    # Mode 1: Threshold from precomputed scores
    if normal_data is None:

        scores = np.asarray(model_or_scores, dtype=float)

        if scores.size == 0:
            raise ValueError("Scores cannot be empty.")

        if not np.isfinite(scores).all():
            raise ValueError("Scores must contain finite values.")

        return float(np.percentile(scores, percentile))

    # Mode 2: Threshold from HMM normal sensor data
    model = model_or_scores

    if len(normal_data) < window_size:
        raise ValueError(
            "Normal data is smaller than window size."
        )

    scores = []

    for start in range(len(normal_data) - window_size + 1):

        window = normal_data[start:start + window_size]

        score = model.score(window) / window_size

        scores.append(score)

    scores = np.asarray(scores)

    threshold = np.percentile(scores, percentile)

    return float(threshold), scores
