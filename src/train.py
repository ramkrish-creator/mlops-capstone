import os
import json
import shutil

import mlflow
import mlflow.sklearn

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.base import clone


DATA_PATH = "data/breast_cancer.csv"
MODEL_DIR = "models/best_model"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "precision": precision_score(
            y_test,
            predictions
        ),

        "recall": recall_score(
            y_test,
            predictions
        ),

        "f1_score": f1_score(
            y_test,
            predictions
        ),

        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        )
    }

    return metrics


def main():

    print("=" * 60)
    print("MLOps Training Pipeline")
    print("=" * 60)

    # --------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------

    print("\n[1] Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)

    # --------------------------------------------------
    # 2. SPLIT FEATURES AND TARGET
    # --------------------------------------------------

    X = df.drop("target", axis=1)
    y = df["target"]

    # --------------------------------------------------
    # 3. TRAIN TEST SPLIT
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # --------------------------------------------------
    # 4. DEFINE MODELS
    # --------------------------------------------------

    models = {

        "Logistic Regression": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE
                )
            )
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=RANDOM_STATE
        ),

        "SVM": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                SVC(
                    probability=True,
                    kernel="rbf",
                    random_state=RANDOM_STATE
                )
            )
        ])
    }

    # --------------------------------------------------
    # 5. MLflow SETUP
    # --------------------------------------------------

    mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

    mlflow.set_experiment(
        "Breast Cancer Classification"
    )

    best_model = None
    best_model_name = None
    best_f1 = -1
    best_metrics = None

    # --------------------------------------------------
    # 6. TRAIN EACH MODEL
    # --------------------------------------------------

    for model_name, model in models.items():

        print("\n" + "=" * 50)
        print(f"Training: {model_name}")
        print("=" * 50)

        with mlflow.start_run(
            run_name=model_name
        ):

            # Train
            model.fit(
                X_train,
                y_train
            )

            # Evaluate
            metrics = evaluate_model(
                model,
                X_test,
                y_test
            )

            print("\nMetrics:")

            for metric_name, value in metrics.items():

                print(
                    f"{metric_name}: "
                    f"{value:.4f}"
                )

            # --------------------------------------------------
            # MLflow PARAMETERS
            # --------------------------------------------------

            mlflow.log_param(
                "model",
                model_name
            )

            mlflow.log_param(
                "test_size",
                TEST_SIZE
            )

            mlflow.log_param(
                "random_state",
                RANDOM_STATE
            )

            # --------------------------------------------------
            # MLflow METRICS
            # --------------------------------------------------

            for metric_name, value in metrics.items():

                mlflow.log_metric(
                    metric_name,
                    value
                )

            # --------------------------------------------------
            # MLflow MODEL
            # --------------------------------------------------

            mlflow.sklearn.log_model(
                model,
                "model"
            )

            # --------------------------------------------------
            # FIND BEST MODEL
            # --------------------------------------------------

            if metrics["f1_score"] > best_f1:

                best_f1 = metrics["f1_score"]

                best_model = model

                best_model_name = model_name

                best_metrics = metrics

    # --------------------------------------------------
    # 7. SAVE BEST MODEL
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("BEST MODEL")
    print("=" * 60)

    print("Model:", best_model_name)

    print(
        "F1 Score:",
        round(best_f1, 4)
    )

    if os.path.exists(MODEL_DIR):

        shutil.rmtree(
            MODEL_DIR
        )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # Save model
    model_path = os.path.join(
        MODEL_DIR,
        "model.joblib"
    )

    import joblib

    joblib.dump(
        best_model,
        model_path
    )

    # Save metadata
    metadata = {

        "model_name": best_model_name,

        "metrics": best_metrics,

        "feature_names": list(X.columns)
    }

    with open(
        os.path.join(
            MODEL_DIR,
            "metadata.json"
        ),
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    print(
        f"\nBest model saved to: "
        f"{model_path}"
    )


if __name__ == "__main__":

    main()