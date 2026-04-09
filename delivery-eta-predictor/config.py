# config.py — Central configuration for the Delivery ETA Predictor

import os

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_data.csv")

# ─────────────────────────────────────────────
# Data Generation
# ─────────────────────────────────────────────
DATA_CONFIG = {
    "n_samples": 10_000,
    "random_seed": 42,
    # Bounding box: Bangalore city area (good for demo)
    "lat_min": 12.85,
    "lat_max": 13.10,
    "lon_min": 77.45,
    "lon_max": 77.75,
    "traffic_levels": ["low", "medium", "high", "very_high"],
    "traffic_weights": [0.25, 0.35, 0.25, 0.15],  # probability distribution
}

# ─────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────
FEATURE_CONFIG = {
    # Rush hour windows (24-hour format)
    "morning_peak_start": 8,
    "morning_peak_end": 10,
    "evening_peak_start": 17,
    "evening_peak_end": 20,

    # Base speed (km/h) per traffic level
    "traffic_speed_map": {
        "low": 40,
        "medium": 28,
        "high": 18,
        "very_high": 10,
    },

    # Traffic multiplier for delivery time
    "traffic_multiplier": {
        "low": 1.0,
        "medium": 1.4,
        "high": 1.9,
        "very_high": 2.8,
    },

    # Encoding map for traffic levels (ordinal)
    "traffic_encoding": {
        "low": 0,
        "medium": 1,
        "high": 2,
        "very_high": 3,
    },

    # Feature columns used for training
    "feature_columns": [
        "haversine_distance",
        "bearing",
        "hour_of_day",
        "day_of_week",
        "is_peak_hour",
        "is_weekend",
        "traffic_encoded",
        "distance_x_traffic",
        "speed_estimate",
        "pickup_lat",
        "pickup_lon",
        "drop_lat",
        "drop_lon",
    ],

    "target_column": "delivery_time_minutes",
}

# ─────────────────────────────────────────────
# Model Training
# ─────────────────────────────────────────────
TRAINING_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,

    "xgboost_params": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    },

    "random_forest_params": {
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    },

    # Hyperparameter search space for tuning
    "xgb_param_grid": {
        "n_estimators": [200, 300],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    },
}

# ─────────────────────────────────────────────
# Flask API
# ─────────────────────────────────────────────
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": False,
    "default_model": "xgboost",  # "xgboost" or "random_forest"
    # Confidence interval width (±minutes)
    "confidence_margin_pct": 0.15,
}
