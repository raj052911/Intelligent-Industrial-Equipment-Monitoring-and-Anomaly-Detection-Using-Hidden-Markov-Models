import pandas as pd

from src.evaluation import (
    evaluate_model,
    save_evaluation_results
)


def run_evaluation():

    print("Starting model evaluation...")

    data = pd.read_csv(
        "data/processed/detection_results.csv"
    )

    y_true = data["is_anomaly"]
    y_pred = data["predicted_anomaly"]

    metrics = evaluate_model(y_true, y_pred)

    save_evaluation_results(metrics)

    print("\nMODEL EVALUATION COMPLETED")


if __name__ == "__main__":
    run_evaluation()
