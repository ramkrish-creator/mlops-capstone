import os
import joblib
import pandas as pd


MODEL_PATH = "models/best_model/model.joblib"


def load_prediction_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(
        MODEL_PATH
    )


def predict(features):

    model = load_prediction_model()

    df = pd.DataFrame(
        [features]
    )

    prediction = model.predict(df)[0]

    probability = None

    if hasattr(
        model,
        "predict_proba"
    ):

        probability = model.predict_proba(
            df
        )[0].tolist()

    return {
        "prediction": int(prediction),
        "probability": probability
    }