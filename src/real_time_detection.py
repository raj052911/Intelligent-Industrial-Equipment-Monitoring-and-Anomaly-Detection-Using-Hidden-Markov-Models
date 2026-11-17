from collections import deque


class RealTimeDetector:
    """
    Detect anomalies from incoming sensor readings
    using a trained Gaussian HMM.
    """

    def __init__(self, model, threshold, window_size=50):

        self.model = model
        self.threshold = threshold
        self.window_size = window_size

        self.buffer = deque(maxlen=window_size)

    def process_reading(self, reading):

        self.buffer.append(reading)

        if len(self.buffer) < self.window_size:
            return None

        window = list(self.buffer)

        score = self.model.score(window) / self.window_size

        return {
            "log_likelihood": score,
            "is_anomaly": score < self.threshold
        }
