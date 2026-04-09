# app.py — Flask REST API for the Delivery ETA Prediction System

import os
import sys
import time
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify, abort

from config import API_CONFIG, MODEL_DIR
from src.predict import predict_single, predict_batch, validate_payload, load_metadata

# ─────────────────────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# Track request count for monitoring
_request_counter = {"total": 0, "errors": 0}
_start_time = time.time()


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """API welcome message."""
    return jsonify({
        "service": "Delivery ETA Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "POST /predict":        "Single delivery ETA prediction",
            "POST /predict/batch":  "Batch ETA predictions",
            "GET  /health":         "Health check",
            "GET  /model/info":     "Model metadata and performance",
        },
        "docs": "See README.md or /model/info for usage details.",
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check — confirms models are loaded and API is live."""
    uptime = round(time.time() - _start_time, 1)
    model_files = ["xgboost_model.pkl", "random_forest_model.pkl", "scaler.pkl"]
    models_present = all(
        os.path.exists(os.path.join(MODEL_DIR, f)) for f in model_files
    )

    status = "healthy" if models_present else "degraded"
    return jsonify({
        "status":         status,
        "uptime_seconds": uptime,
        "models_loaded":  models_present,
        "requests_served": _request_counter["total"],
        "errors":         _request_counter["errors"],
        "timestamp":      datetime.now().isoformat(),
    }), 200 if models_present else 503


@app.route("/model/info", methods=["GET"])
def model_info():
    """Return model metadata, feature list, and evaluation metrics."""
    try:
        metadata = load_metadata()
        return jsonify({
            "model_info": metadata,
            "available_models": list(metadata["models"].keys()),
            "features":         metadata["feature_columns"],
            "target":           metadata["target_column"],
        })
    except FileNotFoundError:
        return jsonify({"error": "Model metadata not found. Run train.py first."}), 503


@app.route("/predict", methods=["POST"])
def predict():
    """
    Single ETA prediction endpoint.

    Request JSON:
    {
        "pickup_lat":       12.9716,
        "pickup_lon":       77.5946,
        "drop_lat":         13.0012,
        "drop_lon":         77.6141,
        "order_timestamp":  "2024-03-15 18:30:00",  // optional
        "traffic_level":    "high",                  // optional
        "model":            "xgboost"                // optional
    }
    """
    _request_counter["total"] += 1
    t0 = time.time()

    if not request.is_json:
        _request_counter["errors"] += 1
        return jsonify({"error": "Content-Type must be application/json"}), 400

    payload = request.get_json()
    model_name = payload.pop("model", API_CONFIG["default_model"])

    is_valid, err_msg = validate_payload(payload)
    if not is_valid:
        _request_counter["errors"] += 1
        return jsonify({"error": err_msg}), 422

    try:
        result = predict_single(payload, model_name=model_name)
        result["latency_ms"] = round((time.time() - t0) * 1000, 2)
        logger.info(
            f"Prediction: ETA={result['predicted_eta_minutes']}min  "
            f"Dist={result['distance_km']}km  Traffic={result['traffic_level']}"
        )
        return jsonify(result), 200

    except Exception as e:
        _request_counter["errors"] += 1
        logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed", "detail": str(e)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch_endpoint():
    """
    Batch ETA prediction endpoint.

    Request JSON:
    {
        "orders": [
            {"pickup_lat": ..., "pickup_lon": ..., "drop_lat": ..., "drop_lon": ..., ...},
            ...
        ],
        "model": "xgboost"  // optional
    }
    """
    _request_counter["total"] += 1
    t0 = time.time()

    if not request.is_json:
        _request_counter["errors"] += 1
        return jsonify({"error": "Content-Type must be application/json"}), 400

    body = request.get_json()
    orders     = body.get("orders", [])
    model_name = body.get("model", API_CONFIG["default_model"])

    if not isinstance(orders, list) or len(orders) == 0:
        return jsonify({"error": "'orders' must be a non-empty list."}), 422

    if len(orders) > 200:
        return jsonify({"error": "Batch size limit is 200 orders per request."}), 422

    errors = []
    for i, order in enumerate(orders):
        valid, msg = validate_payload(order)
        if not valid:
            errors.append({"index": i, "error": msg})

    if errors:
        _request_counter["errors"] += 1
        return jsonify({"error": "Validation failed for some orders", "details": errors}), 422

    try:
        results = predict_batch(orders, model_name=model_name)
        elapsed = round((time.time() - t0) * 1000, 2)
        return jsonify({
            "predictions":      results,
            "count":            len(results),
            "total_latency_ms": elapsed,
            "avg_latency_ms":   round(elapsed / len(results), 2),
        }), 200

    except Exception as e:
        _request_counter["errors"] += 1
        logger.exception("Batch prediction failed")
        return jsonify({"error": "Batch prediction failed", "detail": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# Error Handlers
# ─────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found", "code": 404}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed", "code": 405}), 405


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting Delivery ETA Prediction API...")
    logger.info(f"Model directory: {MODEL_DIR}")
    app.run(
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        debug=API_CONFIG["debug"],
    )
