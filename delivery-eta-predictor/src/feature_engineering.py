# src/feature_engineering.py — Feature extraction & transformation pipeline

import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import FEATURE_CONFIG


# ─────────────────────────────────────────────────────────────
# Geographic Features
# ─────────────────────────────────────────────────────────────

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Vectorised Haversine formula — returns distance in km.
    Works with scalars or numpy arrays.
    """
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def compute_bearing(lat1, lon1, lat2, lon2):
    """
    Compute the initial compass bearing (degrees) from point A to point B.
    Captures direction of delivery which correlates with traffic corridors.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360


# ─────────────────────────────────────────────────────────────
# Temporal Features
# ─────────────────────────────────────────────────────────────

def extract_temporal_features(timestamp_series):
    """
    Extract hour, day-of-week, peak-hour flag, and weekend flag
    from a pandas Series of timestamps.
    """
    ts = pd.to_datetime(timestamp_series)
    hour        = ts.dt.hour
    day_of_week = ts.dt.dayofweek  # Monday=0, Sunday=6

    morning_peak = (hour >= FEATURE_CONFIG["morning_peak_start"]) & \
                   (hour <= FEATURE_CONFIG["morning_peak_end"])
    evening_peak = (hour >= FEATURE_CONFIG["evening_peak_start"]) & \
                   (hour <= FEATURE_CONFIG["evening_peak_end"])
    is_peak_hour = (morning_peak | evening_peak).astype(int)
    is_weekend   = (day_of_week >= 5).astype(int)

    return hour, day_of_week, is_peak_hour, is_weekend


# ─────────────────────────────────────────────────────────────
# Traffic Features
# ─────────────────────────────────────────────────────────────

def encode_traffic(traffic_series):
    """Ordinal encode traffic level: low=0, medium=1, high=2, very_high=3."""
    encoding = FEATURE_CONFIG["traffic_encoding"]
    return traffic_series.map(encoding).fillna(1).astype(int)


def estimate_speed(traffic_series):
    """Return expected speed (km/h) based on traffic level."""
    speed_map = FEATURE_CONFIG["traffic_speed_map"]
    return traffic_series.map(speed_map).fillna(28)


# ─────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
    """
    End-to-end feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe. Must contain:
        pickup_lat, pickup_lon, drop_lat, drop_lon,
        order_timestamp, traffic_level
    training : bool
        If True, expects 'delivery_time_minutes' column.

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataframe ready for model input.
    """
    df = df.copy()

    # ── Geographic ──────────────────────────────────────────
    df["haversine_distance"] = haversine_distance(
        df["pickup_lat"], df["pickup_lon"],
        df["drop_lat"],   df["drop_lon"]
    ).round(4)

    df["bearing"] = compute_bearing(
        df["pickup_lat"], df["pickup_lon"],
        df["drop_lat"],   df["drop_lon"]
    ).round(2)

    # ── Temporal ────────────────────────────────────────────
    if "order_timestamp" in df.columns:
        hour, dow, is_peak, is_weekend = extract_temporal_features(df["order_timestamp"])
        df["hour_of_day"]  = hour.values
        df["day_of_week"]  = dow.values
        df["is_peak_hour"] = is_peak.values
        df["is_weekend"]   = is_weekend.values

    # ── Traffic ─────────────────────────────────────────────
    df["traffic_encoded"] = encode_traffic(df["traffic_level"])
    df["speed_estimate"]  = estimate_speed(df["traffic_level"])

    # ── Interaction Features ─────────────────────────────────
    df["distance_x_traffic"] = df["haversine_distance"] * df["traffic_encoded"]

    # ── Derived ETA estimate (physics-based baseline) ────────
    # Useful as a feature to help the model calibrate
    df["eta_physics_estimate"] = (df["haversine_distance"] / df["speed_estimate"]) * 60

    feature_cols = FEATURE_CONFIG["feature_columns"] + ["eta_physics_estimate"]

    if training:
        return df[feature_cols + [FEATURE_CONFIG["target_column"]]]
    else:
        # Return only available feature columns (for inference)
        available = [c for c in feature_cols if c in df.columns]
        return df[available]


def build_inference_features(payload: dict) -> pd.DataFrame:
    """
    Build features from a single API request payload (dict).
    Returns a one-row DataFrame ready for model.predict().
    """
    row = {
        "pickup_lat":       payload["pickup_lat"],
        "pickup_lon":       payload["pickup_lon"],
        "drop_lat":         payload["drop_lat"],
        "drop_lon":         payload["drop_lon"],
        "order_timestamp":  payload.get("order_timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "traffic_level":    payload.get("traffic_level", "medium"),
    }
    df = pd.DataFrame([row])
    return build_features(df, training=False)
