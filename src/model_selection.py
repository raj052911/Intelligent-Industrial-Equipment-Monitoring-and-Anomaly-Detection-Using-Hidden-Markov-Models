import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def compare_hidden_states(
    X_train,
    X_validation,
    state_counts=(2, 3, 4, 5),
    random_state=42
):
    """
    Train Gaussian HMM models with different hidden-state counts
    and compare their validation log-likelihood.
    """

    results = []

    for n_states in state_counts:

        print(f"\nTraining HMM with {n_states} hidden states...")

        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=200,
            tol=0.001,
            random_state=random_state
        )

        try:
            model.fit(X_train)

            train_score = model.score(X_train) / len(X_train)
            validation_score = (
                model.score(X_validation) / len(X_validation)
            )

            results.append({
                "hidden_states": n_states,
                "train_log_likelihood": train_score,
                "validation_log_likelihood": validation_score,
                "converged": model.monitor_.converged,
                "model": model
            })

            print(f"Training score: {train_score:.4f}")
            print(f"Validation score: {validation_score:.4f}")
            print(f"Converged: {model.monitor_.converged}")

        except Exception as error:
            print(f"Model with {n_states} states failed: {error}")

    if not results:
        raise RuntimeError("No HMM model could be trained successfully.")

    results.sort(
        key=lambda item: item["validation_log_likelihood"],
        reverse=True
    )

    best_model = results[0]["model"]

    comparison = pd.DataFrame([
        {
            "hidden_states": item["hidden_states"],
            "train_log_likelihood": item["train_log_likelihood"],
            "validation_log_likelihood": item["validation_log_likelihood"],
            "converged": item["converged"]
        }
        for item in results
    ])

    return best_model, comparison
