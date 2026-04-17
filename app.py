"""
app.py — FastAPI service for Botnet Detection.

Loads pre-trained ensemble (RF + XGBoost + LightGBM) and autoencoder models
from ModelData/ and exposes prediction endpoints.
"""

import os
import logging
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from tensorflow.keras.models import load_model as keras_load_model

from model import (
    NetworkFlowInput,
    EnsemblePrediction,
    AutoencoderPrediction,
    CombinedPrediction,
    HealthResponse,
)

# ── Logging ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global model references (populated at startup) ─────────────
ensemble_model = None
autoencoder = None
FEATURE_COLUMNS = None
AE_FEATURE_COLUMNS = None
ENGINEERED_FEATURES = None
scaler_ae = None
pca_ae = None
threshold = None
_models_loaded = False

MODEL_DIR = os.getenv("MODEL_DIR", "ModelData")


# ── Feature engineering helpers ─────────────────────────────────


def safe_log(x):
    """Log1p with clip to prevent overflow."""
    return np.log1p(np.clip(x, 0, 1e6))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-driven network traffic features."""
    df = df.copy()
    df["packet_ratio"] = safe_log(df["spkts"] / (df["dpkts"] + 1))
    df["byte_ratio"] = safe_log(df["sbytes"] / (df["dbytes"] + 1))
    df["traffic_intensity"] = safe_log(df["rate"] * df["dur"])
    df["avg_packet_size"] = safe_log(
        (df["sbytes"] + df["dbytes"]) / (df["spkts"] + df["dpkts"] + 1)
    )
    df["ttl_difference"] = safe_log(abs(df["sttl"] - df["dttl"]))
    df["packet_rate"] = safe_log(
        (df["spkts"] + df["dpkts"]) / (df["dur"] + 0.001)
    )
    return df


# ── Inference helpers ───────────────────────────────────────────


def prepare_input(input_dict: dict) -> pd.DataFrame:
    """
    Convert a raw network-flow dict into a model-ready DataFrame.
    Steps: OHE → add engineered features → align to training columns.
    """
    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df)
    df = add_features(df)
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return df


def recon_error(X: np.ndarray) -> np.ndarray:
    """Compute log-normalised MAE reconstruction error."""
    recon = autoencoder.predict(X, verbose=0)
    return np.log1p(np.maximum(np.mean(np.abs(X - recon), axis=1), 0))


def _predict_ensemble(input_dict: dict) -> dict:
    """Run ensemble prediction, returning prediction label + confidence."""
    X = prepare_input(input_dict)
    prob = float(ensemble_model.predict_proba(X)[0][1])

    if prob > 0.60:
        prediction = "BOTNET"
    elif prob < 0.45:
        prediction = "NORMAL"
    else:
        prediction = "SUSPICIOUS"

    return {"prediction": prediction, "confidence": round(prob, 4)}


def _predict_autoencoder(input_dict: dict) -> dict:
    """Run autoencoder anomaly detection."""
    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df)
    df = df.reindex(columns=AE_FEATURE_COLUMNS, fill_value=0)
    X = scaler_ae.transform(df)
    X = pca_ae.transform(X)
    error = float(recon_error(X)[0])

    return {
        "prediction": "BOTNET" if error > threshold else "NORMAL",
        "anomaly_score": round(error, 6),
    }


# ── Startup / Shutdown ──────────────────────────────────────────


def load_models():
    """Load all model artifacts from MODEL_DIR."""
    global ensemble_model, autoencoder
    global FEATURE_COLUMNS, AE_FEATURE_COLUMNS, ENGINEERED_FEATURES
    global scaler_ae, pca_ae, threshold, _models_loaded

    logger.info("Loading models from %s …", MODEL_DIR)

    ensemble_path = os.path.join(MODEL_DIR, "botnet_ensemble.pkl")
    meta_path = os.path.join(MODEL_DIR, "botnet_metadata.pkl")
    ae_path = os.path.join(MODEL_DIR, "botnet_autoencoder.keras")

    ensemble_model = joblib.load(ensemble_path)
    logger.info("  ✓ Ensemble model loaded")

    meta = joblib.load(meta_path)
    FEATURE_COLUMNS = meta["feature_columns"]
    AE_FEATURE_COLUMNS = meta["ae_feature_columns"]
    ENGINEERED_FEATURES = meta["engineered_features"]
    scaler_ae = meta["scaler_ae"]
    pca_ae = meta["pca_ae"]
    threshold = meta["ae_threshold"]
    logger.info("  ✓ Metadata loaded  (threshold=%.6f)", threshold)

    autoencoder = keras_load_model(ae_path)
    logger.info("  ✓ Autoencoder loaded")

    _models_loaded = True
    logger.info("All models loaded successfully.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    load_models()
    yield
    logger.info("Shutting down …")


# ── FastAPI app ─────────────────────────────────────────────────

app = FastAPI(
    title="Botnet Detection API",
    description=(
        "Serves pre-trained ensemble (RF + XGBoost + LightGBM) and autoencoder "
        "models for real-time botnet traffic detection on UNSW-NB15 features."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ───────────────────────────────────────────────────


@app.get("/", tags=["General"])
def root():
    """Welcome / root endpoint."""
    return {
        "service": "Botnet Detection API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    """Readiness probe."""
    return HealthResponse(status="ok", models_loaded=_models_loaded)


@app.post(
    "/predict/ensemble",
    response_model=EnsemblePrediction,
    tags=["Predictions"],
    summary="Ensemble classifier prediction",
)
def predict_ensemble(flow: NetworkFlowInput):
    """
    Predict using the soft-voting ensemble (Random Forest + XGBoost + LightGBM).

    Returns a three-class label (BOTNET / SUSPICIOUS / NORMAL) and the
    botnet-class probability as confidence.
    """
    if not _models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    result = _predict_ensemble(flow.to_dict())
    return EnsemblePrediction(**result)


@app.post(
    "/predict/autoencoder",
    response_model=AutoencoderPrediction,
    tags=["Predictions"],
    summary="Autoencoder anomaly detection",
)
def predict_autoencoder(flow: NetworkFlowInput):
    """
    Predict using the autoencoder anomaly detector.

    Returns a binary label (BOTNET / NORMAL) and the reconstruction-error
    anomaly score.
    """
    if not _models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    result = _predict_autoencoder(flow.to_dict())
    return AutoencoderPrediction(**result)


@app.post(
    "/predict/combined",
    response_model=CombinedPrediction,
    tags=["Predictions"],
    summary="Combined prediction (both models)",
)
def predict_combined(flow: NetworkFlowInput):
    """
    Run both the ensemble classifier and the autoencoder anomaly detector
    on the same input and return combined results.
    """
    if not _models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    input_dict = flow.to_dict()
    ens_result = _predict_ensemble(input_dict)
    ae_result = _predict_autoencoder(input_dict)

    return CombinedPrediction(
        ensemble=EnsemblePrediction(**ens_result),
        autoencoder=AutoencoderPrediction(**ae_result),
    )
