from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.predict import predict


app = FastAPI(
    title="Breast Cancer Prediction API",
    description="MLOps Breast Cancer Classification API",
    version="1.0.0"
)


class PredictionRequest(BaseModel):

    features: List[float]


@app.get("/")
def root():

    return {
        "message": "Breast Cancer Prediction API",
        "status": "running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/predict")
def make_prediction(
    request: PredictionRequest
):

    try:

        result = predict(
            request.features
        )

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )