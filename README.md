from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import joblib
from datetime import datetime

try:
    model_data = joblib.load("nba_gnb_model.pkl")

    MODEL = model_data["model"]
    SCALER = model_data["scaler"]
    FEATURE_NAMES = model_data["feature_names"]
    METRICS = model_data["metrics"]
    FEATURE_IMPORTANCE = model_data["feature_importance"]

except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

app = FastAPI(
    title="NBA Player Longevity Prediction API",
    description="Predict whether an NBA rookie will have a 5+ year career.",
    version="1.0.0"
)

class PlayerStats(BaseModel):

    GP: float = Field(..., ge=0, le=82)

    MIN: float = Field(..., ge=0, le=48)

    PTS: float = Field(..., ge=0, le=50)

    FGM: float = Field(..., ge=0)

    FGA: float = Field(..., ge=0)

    FG_Percent: float = Field(..., ge=0, le=100, alias="FG%")

    ThreeP_Made: float = Field(..., ge=0, alias="3P Made")

    ThreePA: float = Field(..., ge=0, alias="3PA")

    ThreeP_Percent: float = Field(..., ge=0, le=100, alias="3P%")

    FTM: float = Field(..., ge=0)

    FTA: float = Field(..., ge=0)

    FT_Percent: float = Field(..., ge=0, le=100, alias="FT%")

    OREB: float = Field(..., ge=0)

    DREB: float = Field(..., ge=0)

    REB: float = Field(..., ge=0)

    AST: float = Field(..., ge=0)

    STL: float = Field(..., ge=0)

    BLK: float = Field(..., ge=0)

    TOV: float = Field(..., ge=0)

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):

    player_id: Optional[str] = None

    prediction: int

    prediction_label: str

    confidence: float

    confidence_tier: str

    risk_assessment: str

    feature_contributions: Dict[str, float]

    model_version: str = "gnb-v1.0"

    timestamp: str


class BatchPredictionRequest(BaseModel):

    players: List[PlayerStats]


class BatchPredictionResponse(BaseModel):

    predictions: List[PredictionResponse]

    summary: Dict[str, int]


class ModelMetrics(BaseModel):

    accuracy: float

    precision: float

    recall: float

    f1_score: float

    auc_roc: float

    confusion_matrix: List[List[int]]

    feature_importance: Dict[str, float]

    total_predictions: int = 0


class HealthCheck(BaseModel):

    status: str

    model_loaded: bool

    model_version: str

    timestamp: str


def calculate_feature_contributions(features_scaled):

    contributions = {}

    for i, feature in enumerate(FEATURE_NAMES):

        mean_diff = abs(
            MODEL.theta_[1, i] - MODEL.theta_[0, i]
        )

        contributions[feature] = round(
            float(mean_diff * abs(features_scaled[0, i])),
            4
        )

    total = sum(contributions.values())

    if total > 0:

        contributions = {
            k: round(v / total * 100, 2)
            for k, v in contributions.items()
        }

    return dict(
        sorted(
            contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
    )


def get_confidence_tier(probability):

    if probability >= 0.85 or probability <= 0.15:

        return "High"

    elif probability >= 0.65 or probability <= 0.35:

        return "Medium"

    return "Low"


def get_risk_assessment(prediction, confidence):

    if prediction == 1:

        if confidence > 0.85:

            return "STRONG RECOMMEND"

        elif confidence > 0.65:

            return "RECOMMEND"

        else:

            return "CONDITIONAL"

    else:

        if confidence > 0.85:

            return "AVOID"

        elif confidence > 0.65:

            return "CAUTION"

        else:

            return "RE-EVALUATE"


@app.get("/", response_model=HealthCheck)

async def root():

    return HealthCheck(

        status="healthy",

        model_loaded=True,

        model_version="gnb-v1.0",

        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/health", response_model=HealthCheck)

async def health_check():

    return await root()


@app.post("/predict", response_model=PredictionResponse)

async def predict(
        player: PlayerStats,
        player_id: Optional[str] = None
):

    try:

        features = np.array([[
            player.GP,
            player.MIN,
            player.PTS,
            player.FGM,
            player.FGA,
            player.FG_Percent,
            player.ThreeP_Made,
            player.ThreePA,
            player.ThreeP_Percent,
            player.FTM,
            player.FTA,
            player.FT_Percent,
            player.OREB,
            player.DREB,
            player.REB,
            player.AST,
            player.STL,
            player.BLK,
            player.TOV
        ]])

        features_scaled = SCALER.transform(features)

        prediction = int(
            MODEL.predict(features_scaled)[0]
        )

        probabilities = MODEL.predict_proba(features_scaled)[0]

        confidence = float(probabilities[prediction])

        contributions = calculate_feature_contributions(features_scaled)

        return PredictionResponse(

            player_id=player_id,

            prediction=prediction,

            prediction_label=(
                "5+ Year Career"
                if prediction == 1
                else "< 5 Year Career"
            ),

            confidence=round(confidence, 4),

            confidence_tier=get_confidence_tier(confidence),

            risk_assessment=get_risk_assessment(
                prediction,
                confidence
            ),

            feature_contributions=contributions,

            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse
)

async def predict_batch(
        request: BatchPredictionRequest
):

    predictions = []

    summary = {
        "five_plus_years": 0,
        "less_than_five": 0,
        "high_confidence": 0
    }

    for idx, player in enumerate(request.players):

        result = await predict(
            player,
            player_id=f"player_{idx+1}"
        )

        predictions.append(result)

        if result.prediction == 1:

            summary["five_plus_years"] += 1

        else:

            summary["less_than_five"] += 1

        if result.confidence_tier == "High":

            summary["high_confidence"] += 1

    return BatchPredictionResponse(
        predictions=predictions,
        summary=summary
    )


@app.get(
    "/metrics",
    response_model=ModelMetrics
)

async def get_metrics():

    return ModelMetrics(

        accuracy=METRICS["accuracy"],

        precision=METRICS["precision"],

        recall=METRICS["recall"],

        f1_score=METRICS["