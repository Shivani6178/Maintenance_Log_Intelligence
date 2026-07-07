from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np
import joblib
import logging
from tensorflow.keras.models import load_model

from src.api.schemas import (
    QueryRequest, QueryResponse,
    RULPredictionRequest, RULPredictionResponse
)
from src.generation.rag_service import answer_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ml_models = {}

BILSTM_MODEL_PATH = "models/bilstm_model.h5"
SCALER_PATH = "models/scaler.pkl"
WINDOW_LENGTH = 30
N_FEATURES = 14
RUL_CEILING = 125


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading Bi-LSTM RUL model and scaler")
    try:
        ml_models["bilstm"] = load_model(BILSTM_MODEL_PATH)
        ml_models["scaler"] = joblib.load(SCALER_PATH)
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        ml_models["bilstm"] = None
        ml_models["scaler"] = None

    yield

    logger.info("Shutting down, clearing model references")
    ml_models.clear()


app = FastAPI(
    title="Aircraft Maintenance Intelligence API",
    description="RUL prediction and RAG-based maintenance log retrieval",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "rul_model_loaded": ml_models.get("bilstm") is not None
    }


@app.post("/query", response_model=QueryResponse)
def query_maintenance_logs(request: QueryRequest):
    try:
        result = answer_query(request.query, top_k=request.top_k)
        return result
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process query")


@app.post("/predict-rul", response_model=RULPredictionResponse)
def predict_rul(request: RULPredictionRequest):
    if ml_models.get("bilstm") is None:
        raise HTTPException(status_code=503, detail="RUL model is not available")

    readings = np.array(request.sensor_readings, dtype=np.float32)

    if readings.shape != (WINDOW_LENGTH, N_FEATURES):
        raise HTTPException(
            status_code=422,
            detail=f"Expected shape ({WINDOW_LENGTH}, {N_FEATURES}), got {readings.shape}"
        )

    try:
        scaler = ml_models["scaler"]
        scaled = scaler.transform(readings)
        model_input = scaled.reshape(1, WINDOW_LENGTH, N_FEATURES)

        prediction = ml_models["bilstm"].predict(model_input, verbose=0)
        predicted_rul = float(prediction[0][0])
        predicted_rul = min(predicted_rul, RUL_CEILING)

        if predicted_rul < 20:
            warning = "critical"
        elif predicted_rul < 50:
            warning = "warning"
        else:
            warning = "normal"

        return RULPredictionResponse(
            predicted_rul=round(predicted_rul, 2),
            cycles_remaining=int(round(predicted_rul)),
            warning_level=warning
        )
    except Exception as e:
        logger.error(f"RUL prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")