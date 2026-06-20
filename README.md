import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

app = FastAPI(
    title="NBA Player Longevity Prediction API",
    description="Gaussian Naive Bayes model for NBA career longevity prediction",
    version="1.0.0"
)

MODEL_PATH = "nb_nba_model.joblib"

model = None
expected_features = None


@app.on_event("startup")
def load_model():
    global model, expected_features

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found")

    payload = joblib.load(MODEL_PATH)

    model = payload.get("model")
    expected_features = payload.get("features")

    if model is None or expected_features is None:
        raise ValueError("Model file missing required keys: 'model' or 'features'")


class PlayerFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    fg: float = Field(..., description="Field Goal Percentage")
    three_p: float = Field(..., alias="3p")
    ft: float
    reb: float
    ast: float
    stl: float
    blk: float
    tov: float
    total_points: float
    efficiency: float


class PredictionResponse(BaseModel):
    prediction: int
    probability_class_0: float
    probability_class_1: float
    scouting_guidance: str


@app.get("/")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(player: PlayerFeatures):
    try:
        data = player.model_dump(by_alias=True)

        # safer feature extraction
        try:
            X = np.array([data[feat] for feat in expected_features]).reshape(1, -1)
        except KeyError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Missing feature in request or mismatch: {str(e)}"
            )

        pred = int(model.predict(X)[0])

        # safer probability handling
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
        else:
            probs = [0.5, 0.5]

        guidance = (
            "HIGH CONFIDENCE PROSPECT: Fast-track scouting."
            if pred == 1
            else "POTENTIAL RISK: Requires deeper scouting validation."
        )

        return PredictionResponse(
            prediction=pred,
            probability_class_0=float(probs[0]),
            probability_class_1=float(probs[1]),
            scouting_guidance=guidance
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))