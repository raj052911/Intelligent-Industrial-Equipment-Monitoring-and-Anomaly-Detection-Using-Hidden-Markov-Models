import numpy as np
import pandas as pd
from src.data_preprocessing import fit_scaler, transform_data

def test_scaler_output_shape():
    frame = pd.DataFrame({
        "temperature": [60, 61, 62],
        "vibration": [0.2, 0.3, 0.4],
        "pressure": [99, 100, 101]
    })
    scaler = fit_scaler(frame)
    transformed = transform_data(frame, scaler)

    assert transformed.shape == (3, 3)
    assert np.isfinite(transformed).all()
