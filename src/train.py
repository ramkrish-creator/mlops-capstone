import os
import json
import shutil

import joblib
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


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/breast_cancer.csv"

MODEL_DIR = "models/best_model"

RANDOM_STATE = 42

TEST_SIZE = 0.2

EXPERIMENT_NAME = "Breast Cancer Classification"

REGISTERED_MODEL_NAME = "BreastCancerClassifier"


# ============================================================
# MLflow configuration
# ============================================================

# GitHub Actions automatically provides CI=true.
# Locally, CI will normally not exist.

IS_CI = os.getenv(
    "CI",
    "false"
).lower() == "true"


# If MLFLOW_TRACKING_URI is provided, use it.
#
# Local:
#     http://127.0.0.1:5000
#
# GitHub Actions:
#     sqlite:///mlflow.db
#
# Default:
#     sqlite:///mlflow.db

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///mlflow.db"
)


# ============================================================
# Evaluation function
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    metrics = {

        "accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        )
    }

    return metrics


# ============================================================
# Main training pipeline
# ============================================================

def main():

    print("=" * 60)

    print(
        "MLOps Training Pipeline"
    )

    print("=" * 60)

    print()

    # ========================================================
    # ENVIRONMENT
    # ========================================================

    print(
        f"Running in CI: {IS_CI}"
    )

    print(
        f"MLflow URI: {MLFLOW_TRACKING_URI}"
    )

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    print()

    print(
        "[1] Loading dataset..."
    )

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    # ========================================================
    # 2. FEATURES / TARGET
    # ========================================================

    X = df.drop(
        "target",
        axis=1
    )

    y = df["target"]

    print(
        f"Number of features: {X.shape[1]}"
    )

    print(
        f"Number of samples: {X.shape[0]}"
    )

    # ========================================================
    # 3. TRAIN / TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    # ========================================================
    # 4. DEFINE MODELS
    # ========================================================

    models = {

        "Logistic Regression":

            Pipeline(
                [
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
                ]
            ),

        "Random Forest":

            RandomForestClassifier(

                n_estimators=200,

                random_state=RANDOM_STATE,

                n_jobs=-1
            ),

        "SVM":

            Pipeline(
                [
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
                ]
            )
    }

    # ========================================================
    # 5. MLflow SETUP
    # ========================================================

    print()

    print(
        "[2] Connecting to MLflow..."
    )

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    print(
        "[3] Setting MLflow experiment..."
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print(
        "[4] MLflow setup completed."
    )

    # ========================================================
    # BEST MODEL VARIABLES
    # ========================================================

    best_model = None

    best_model_name = None

    best_metrics = None

    best_f1 = -1

    # ========================================================
    # 6. TRAIN EACH MODEL
    # ========================================================

    for model_name, model in models.items():

        print()

        print("=" * 50)

        print(
            f"Training: {model_name}"
        )

        print("=" * 50)

        # ----------------------------------------------------
        # Start MLflow run
        # ----------------------------------------------------

        with mlflow.start_run(
            run_name=model_name
        ):

            # ------------------------------------------------
            # TRAIN
            # ------------------------------------------------

            print(
                f"[{model_name}] Training..."
            )

            model.fit(
                X_train,
                y_train
            )

            print(
                f"[{model_name}] Training completed."
            )

            # ------------------------------------------------
            # EVALUATE
            # ------------------------------------------------

            print(
                f"[{model_name}] Evaluating..."
            )

            metrics = evaluate_model(

                model,

                X_test,

                y_test
            )

            # ------------------------------------------------
            # DISPLAY METRICS
            # ------------------------------------------------

            print()

            print(
                f"Results for {model_name}:"
            )

            for metric_name, value in metrics.items():

                print(
                    f"{metric_name}: {value:.4f}"
                )

            # ------------------------------------------------
            # MLflow PARAMETERS
            # ------------------------------------------------

            mlflow.log_param(
                "model",
                model_name
            )

            mlflow.log_param(
                "random_state",
                RANDOM_STATE
            )

            mlflow.log_param(
                "test_size",
                TEST_SIZE
            )

            mlflow.log_param(
                "number_of_features",
                X.shape[1]
            )

            # ------------------------------------------------
            # MLflow METRICS
            # ------------------------------------------------

            for metric_name, value in metrics.items():

                mlflow.log_metric(
                    metric_name,
                    value
                )

            # ------------------------------------------------
            # MLflow MODEL ARTIFACT
            # ------------------------------------------------

            print(
                f"[{model_name}] Logging model to MLflow..."
            )

            mlflow.sklearn.log_model(
                model,
                "model"
            )

            print(
                f"[{model_name}] MLflow logging completed."
            )

            # ------------------------------------------------
            # CHECK BEST MODEL
            # ------------------------------------------------

            if metrics["f1_score"] > best_f1:

                best_f1 = metrics[
                    "f1_score"
                ]

                best_model = model

                best_model_name = model_name

                best_metrics = metrics.copy()

                print(
                    f"[{model_name}] Current best model."
                )

    # ========================================================
    # 7. DISPLAY BEST MODEL
    # ========================================================

    print()

    print("=" * 60)

    print(
        "BEST MODEL"
    )

    print("=" * 60)

    print(
        f"Model: {best_model_name}"
    )

    print(
        f"F1 Score: {best_f1:.4f}"
    )

    print()

    print(
        "Best model metrics:"
    )

    for metric_name, value in best_metrics.items():

        print(
            f"{metric_name}: {value:.4f}"
        )

    # ========================================================
    # 8. SAVE BEST MODEL
    # ========================================================

    print()

    print(
        "[5] Saving best model..."
    )

    if os.path.exists(
        MODEL_DIR
    ):

        shutil.rmtree(
            MODEL_DIR
        )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model_path = os.path.join(
        MODEL_DIR,
        "model.joblib"
    )

    joblib.dump(
        best_model,
        model_path
    )

    # ========================================================
    # 9. SAVE METADATA
    # ========================================================

    metadata = {

        "model_name":
            best_model_name,

        "metrics":
            best_metrics,

        "feature_names":
            list(X.columns),

        "number_of_features":
            X.shape[1],

        "random_state":
            RANDOM_STATE,

        "test_size":
            TEST_SIZE
    }

    metadata_path = os.path.join(
        MODEL_DIR,
        "metadata.json"
    )

    with open(
        metadata_path,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    print(
        f"Model saved to: {model_path}"
    )

    print(
        f"Metadata saved to: {metadata_path}"
    )

    # ========================================================
    # 10. REGISTER BEST MODEL
    # ========================================================

    if IS_CI:

        print()

        print(
            "[6] CI environment detected."
        )

        print(
            "Skipping MLflow Model Registry."
        )

        print(
            "Model registration is performed locally."
        )

    else:

        print()

        print(
            "[6] Registering best model in MLflow..."
        )

        # Create a separate MLflow run specifically
        # for the selected best model.

        with mlflow.start_run(
            run_name="Best Model"
        ):

            mlflow.log_param(
                "best_model",
                best_model_name
            )

            for metric_name, value in best_metrics.items():

                mlflow.log_metric(
                    metric_name,
                    value
                )

            mlflow.sklearn.log_model(

                best_model,

                "best_model",

                registered_model_name=
                    REGISTERED_MODEL_NAME
            )

        print(
            "Best model registered successfully."
        )

        print(
            f"Registered model: "
            f"{REGISTERED_MODEL_NAME}"
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()

    print("=" * 60)

    print(
        "TRAINING PIPELINE COMPLETED SUCCESSFULLY"
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()