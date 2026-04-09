# src/predict.py — Inference utilities for the ETA prediction system

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import MODEL_DIR, API_CONFIG, FEATURE_CONFIG
from src.feature_engineering import build_inference_features, haversine_distance


# ─────────────────────────────────────────────────────────────
# Model Loader (singleton-style cache)
# ─────────────────────────────────────────────────────────────

_model_cache: Dict[str, Any] = {}


def _load(filename: str):
    if filename not in _model_cache:
        path = os.path.join(MODEL_DIR, filename)
        with open(path, "rb") as f:
            _model_cache[filename] = pickle.load(f)
    return _model_cache[filename]


def load_model(model_name: str = "xgboost"):
    """Load model and scaler from disk (cached after first load)."""
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    with open(meta_path) as f:
        metadata = json.load(f)

    model_file = metadata["models"][model_name]["file"]
    model  = _load(model_file)
    scaler = _load("scaler.pkl")
    return model, scaler, metadata


def load_metadata() -> dict:
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    with open(meta_path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────

def predict_single(payload: dict, model_name: str = None) -> dict:
    """
    Predict ETA for a single delivery request.

    Parameters
    ----------
    payload : dict
        Keys: pickup_lat, pickup_lon, drop_lat, drop_lon,
              order_timestamp (optional), traffic_level (optional)
    model_name : str
        "xgboost" (default) or "random_forest"

    Returns
    -------
    dict  — prediction result with metadata
    """
    if model_name is None:
        model_name = API_CONFIG["default_model"]

    model, scaler, metadata = load_model(model_name)

    # Build feature row
    features_df = build_inference_features(payload)

    # Align columns to training order
    train_cols = metadata["feature_columns"]
    for col in train_cols:
        if col not in features_df.columns:
            features_df[col] = 0  # fill missing with 0
    features_df = features_df[train_cols]

    # XGBoost works on raw features; RF was trained on scaled features
    if model_name == "random_forest":
        X = scaler.transform(features_df)
    else:
        X = features_df.values

    eta = float(model.predict(X)[0])
    eta = max(5.0, round(eta, 1))  # floor at 5 min

    distance_km = float(haversine_distance(
        payload["pickup_lat"], payload["pickup_lon"],
        payload["drop_lat"],   payload["drop_lon"]
    ))

    margin = eta * API_CONFIG["confidence_margin_pct"]

    return {
        "predicted_eta_minutes": eta,
        "distance_km":           round(distance_km, 3),
        "traffic_level":         payload.get("traffic_level", "medium"),
        "is_peak_hour":          bool(features_df["is_peak_hour"].values[0]),
        "model_used":            model_name,
        "confidence_range": {
            "lower": round(max(0, eta - margin), 1),
            "upper": round(eta + margin, 1),
        },
        "timestamp": datetime.now().isoformat(),
    }


def predict_batch(payloads: List[dict], model_name: str = None) -> List[dict]:
    """
    Predict ETA for a list of delivery requests.
    Returns a list of prediction results in the same order.
    """
    return [predict_single(p, model_name) for p in payloads]


# ─────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────

REQUIRED_FIELDS = ["pickup_lat", "pickup_lon", "drop_lat", "drop_lon"]
VALID_TRAFFIC   = {"low", "medium", "high", "very_high"}

def validate_payload(payload: dict) -> (bool, str):
    """
    Validate an incoming prediction payload.
    Returns (is_valid: bool, error_message: str).
    """
    for field in REQUIRED_FIELDS:
        if field not in payload:
            return False, f"Missing required field: '{field}'"

    for coord in REQUIRED_FIELDS:
        val = payload[coord]
        if not isinstance(val, (int, float)):
            return False, f"'{coord}' must be a number."

    lat1, lon1 = payload["pickup_lat"], payload["pickup_lon"]
    lat2, lon2 = payload["drop_lat"], payload["drop_lon"]

    if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90):
        return False, "Latitude must be between -90 and 90."
    if not (-180 <= lon1 <= 180 and -180 <= lon2 <= 180):
        return False, "Longitude must be between -180 and 180."

    traffic = payload.get("traffic_level", "medium")
    if traffic not in VALID_TRAFFIC:
        return False, f"'traffic_level' must be one of {sorted(VALID_TRAFFIC)}."

    return True, ""
