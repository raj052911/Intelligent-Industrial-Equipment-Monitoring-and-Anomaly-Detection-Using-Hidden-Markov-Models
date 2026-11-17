import csv
from pathlib import Path

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(y_true, y_pred):
    """
    Evaluate HMM anomaly detection performance.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    matrix = confusion_matrix(y_true, y_pred)

    print("\n========== MODEL EVALUATION ==========")
    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": matrix
    }


def save_evaluation_results(
    metrics,
    output_path="results/evaluation_report.csv"
):
    """
    Save model evaluation metrics into a CSV file.
    """

    output_file = Path(output_path)
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(["Metric", "Value"])

        writer.writerow([
            "Accuracy",
            metrics["accuracy"]
        ])

        writer.writerow([
            "Precision",
            metrics["precision"]
        ])

        writer.writerow([
            "Recall",
            metrics["recall"]
        ])

        writer.writerow([
            "F1 Score",
            metrics["f1_score"]
        ])

    print("\nEvaluation report saved:", output_file)
