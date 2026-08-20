import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def load_data(path):
    """
    Load dataset from CSV.
    """
    return pd.read_csv(path)


def calculate_metrics(y_true, y_pred, y_probability):
    """
    Calculate classification metrics.
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_probability)
    }

    return metrics


def save_model(model, path):
    """
    Save trained model.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)


def load_model(path):
    """
    Load trained model.
    """

    return joblib.load(path)


def save_metrics(metrics, path):
    """
    Save metrics as JSON.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)