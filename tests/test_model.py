import numpy as np
from src.train_model import train_hmm

def test_model_training():
    rng = np.random.default_rng(42)
    observations = rng.normal(size=(300, 3))

    model = train_hmm(observations, n_states=2)

    assert model.n_components == 2
    assert np.isfinite(model.score(observations))
