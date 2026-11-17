from pathlib import Path

import joblib
import numpy as np
from hmmlearn.hmm import GaussianHMM

from src.model_selection import compare_hidden_states


MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")


def train_hmm(sequences, n_states=3, random_state=42):
    """
    Train a Gaussian Hidden Markov Model.
    """

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=200,
        tol=0.001,
        random_state=random_state
    )

    model.fit(sequences)

    return model


def train_best_model():
    """
    Compare different HMM configurations and save
    the best-performing model.
    """

    # Load preprocessed datasets
    X_train = np.load("data/processed/X_train.npy")
    X_validation = np.load("data/processed/X_validation.npy")

    # Compare hidden-state configurations
    best_model, comparison = compare_hidden_states(
        X_train,
        X_validation,
        state_counts=(2, 3, 4, 5)
    )

    # Create directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save trained model
    model_path = MODEL_DIR / "best_hmm_model.pkl"
    joblib.dump(best_model, model_path)

    # Save comparison results
    comparison.to_csv(
        RESULTS_DIR / "hmm_model_comparison.csv",
        index=False
    )

    print("\nMODEL SELECTION RESULTS")
    print(comparison.to_string(index=False))

    print("\nBEST MODEL SELECTED")
    print("Hidden states:", best_model.n_components)
    print("Model saved at:", model_path)

    return best_model


if __name__ == "__main__":
    train_best_model()