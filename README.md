# HMM-Based Machine Sensor Anomaly Detection System

## Project Overview

An original Python application for detecting unusual machine operating
patterns using a Gaussian Hidden Markov Model (HMM).

The model learns normal sensor behaviour and evaluates incoming sequences
using log-likelihood scores. A sliding-window detector flags windows whose
scores fall below a threshold calculated from normal training data.

## Main Features

- Gaussian Hidden Markov Model using hmmlearn.
- Simulated machine sensor data.
- Normal-data model training.
- Hidden-state model comparison.
- Likelihood-based anomaly threshold.
- Sliding-window anomaly detection.
- Interactive Flask dashboard.
- Sensor visualizations and evaluation metrics.

## Technology Stack

Python, Flask, hmmlearn, NumPy, Pandas, Scikit-learn,
Matplotlib, Chart.js, Bootstrap 5 and JavaScript.

## Project Structure

- app/: Flask routes, templates and frontend assets.
- src/: Data processing, HMM training and anomaly detection.
- data/: Raw and processed sensor datasets.
- models/: Persisted trained models.
- results/: Generated charts and reports.
- config/: Application settings.
- tests/: Automated tests.

## Setup

Create and activate a Python virtual environment.

Install dependencies:

    pip install -r requirements.txt

Generate datasets and run the model pipeline (implementation steps
will be documented as the pipeline is completed).

Start the application:

    python run.py

Open:

    http://127.0.0.1:5000

## Methodology

1. Generate or load normal sensor observations.
2. Fit a Gaussian HMM on normal training sequences.
3. Compare candidate hidden-state counts.
4. Calculate normal-window likelihood scores.
5. Set a low-percentile likelihood threshold.
6. Evaluate overlapping windows in the test stream.
7. Visualize detected anomaly regions.
8. Evaluate predictions using precision, recall and F1-score.

## Academic Integrity

The implementation and project documentation are developed specifically
for this assignment. External libraries are credited in the requirements.
Any external datasets used in the final project will be acknowledged.

## Current Development Status

Initial project scaffold created. The complete end-to-end pipeline,
dashboard integration and testing will be developed and verified in stages.
