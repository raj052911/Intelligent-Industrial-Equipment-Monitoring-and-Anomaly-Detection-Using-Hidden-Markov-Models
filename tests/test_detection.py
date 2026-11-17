import numpy as np
from src.threshold_calculation import calculate_threshold
from src.anomaly_detection import detect_anomalies, expand_window_flags

class FakeModel:
    def score(self, observations):
        return -float(np.sum(observations ** 2))

def test_threshold_and_detection():
    threshold = calculate_threshold([-1.0, -2.0, -3.0], percentile=50)
    assert threshold == -2.0

    observations = np.ones((20, 2))
    observations[8:12] = 10

    result = detect_anomalies(
        FakeModel(), observations, threshold=-5, window_size=4
    )
    mask = expand_window_flags(result, len(observations))

    assert mask.shape == (20,)
