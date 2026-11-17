from pathlib import Path

import joblib
import pandas as pd

from flask import Blueprint, jsonify, render_template


main = Blueprint("main", __name__)

RESULTS_FILE = Path("data/processed/detection_results.csv")
MODEL_FILE = Path("models/best_hmm_model.pkl")


@main.route("/")
def dashboard():
    return render_template("dashboard.html")


@main.route("/monitoring")
def monitoring():
    return render_template("monitoring.html")


@main.route("/model-analysis")
def model_analysis():
    return render_template("model_analysis.html")


@main.route("/reports")
def reports():
    return render_template("reports.html")


def load_detection_data():
    if not RESULTS_FILE.exists():
        return None

    return pd.read_csv(RESULTS_FILE)


@main.route("/api/metrics")
def get_metrics():

    data = load_detection_data()

    if data is None:
        return jsonify({"error": "Detection results not found"}), 404

    total = len(data)
    anomalies = int(data["predicted_anomaly"].sum())
    normal = total - anomalies

    return jsonify({
        "total_readings": total,
        "normal_readings": normal,
        "anomalies": anomalies,
        "anomaly_rate": round(
            (anomalies / total) * 100, 2
        ) if total else 0
    })


@main.route("/api/sensor-data")
def get_sensor_data():

    data = load_detection_data()

    if data is None:
        return jsonify({"error": "Detection results not found"}), 404

    columns = [
        "timestamp",
        "temperature",
        "vibration",
        "pressure",
        "predicted_anomaly",
        "anomaly_score"
    ]

    available_columns = [
        column for column in columns
        if column in data.columns
    ]

    data = data[available_columns].copy()

    data = data.where(pd.notna(data), None)

    return jsonify(data.to_dict(orient="records"))


@main.route("/api/model-info")
def get_model_info():

    if not MODEL_FILE.exists():
        return jsonify({"error": "Trained model not found"}), 404

    model = joblib.load(MODEL_FILE)

    return jsonify({
        "model_type": "Gaussian Hidden Markov Model",
        "hidden_states": int(model.n_components),
        "covariance_type": model.covariance_type,
        "model_status": "Trained"
    })
