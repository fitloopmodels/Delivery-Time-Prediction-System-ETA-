# data/generate_data.py — Synthetic delivery dataset generator

import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_CONFIG, FEATURE_CONFIG

np.random.seed(DATA_CONFIG["random_seed"])


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance (km) between two points
    on Earth using the Haversine formula.
    """
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def compute_delivery_time(distance_km, traffic_level, hour, is_weekend):
    """
    Simulate realistic delivery time with noise and conditional delays.
    Base time = distance / speed + preparation time + random noise.
    """
    speed = FEATURE_CONFIG["traffic_speed_map"][traffic_level]
    multiplier = FEATURE_CONFIG["traffic_multiplier"][traffic_level]

    # Base travel time in minutes
    base_time = (distance_km / speed) * 60

    # Peak-hour penalty
    is_peak = (
        (FEATURE_CONFIG["morning_peak_start"] <= hour <= FEATURE_CONFIG["morning_peak_end"])
        or (FEATURE_CONFIG["evening_peak_start"] <= hour <= FEATURE_CONFIG["evening_peak_end"])
    )
    peak_penalty = np.random.uniform(2, 6) if is_peak else 0

    # Weekend reduction (lighter traffic overall)
    weekend_factor = 0.88 if is_weekend else 1.0

    # Preparation / restaurant wait time
    prep_time = np.random.uniform(3, 12)

    # Stochastic noise (real-world variability)
    noise = np.random.normal(0, 2.5)

    delivery_time = (base_time * multiplier * weekend_factor) + prep_time + peak_penalty + noise
    return max(5.0, round(delivery_time, 2))  # minimum 5 minutes


def generate_dataset(n_samples=None):
    cfg = DATA_CONFIG
    n = n_samples or cfg["n_samples"]

    print(f"[DataGen] Generating {n:,} synthetic delivery records...")

    # Coordinates
    pickup_lat = np.random.uniform(cfg["lat_min"], cfg["lat_max"], n)
    pickup_lon = np.random.uniform(cfg["lon_min"], cfg["lon_max"], n)
    drop_lat   = np.random.uniform(cfg["lat_min"], cfg["lat_max"], n)
    drop_lon   = np.random.uniform(cfg["lon_min"], cfg["lon_max"], n)

    # Timestamps — spread over 1 year
    start_ts = pd.Timestamp("2023-01-01")
    end_ts   = pd.Timestamp("2024-01-01")
    seconds_range = int((end_ts - start_ts).total_seconds())
    random_seconds = np.random.randint(0, seconds_range, n)
    timestamps = [start_ts + pd.Timedelta(seconds=int(s)) for s in random_seconds]

    hours       = np.array([t.hour for t in timestamps])
    days_of_week = np.array([t.dayofweek for t in timestamps])
    is_weekend  = (days_of_week >= 5).astype(int)

    # Traffic levels
    traffic = np.random.choice(
        cfg["traffic_levels"],
        size=n,
        p=cfg["traffic_weights"],
    )

    # Distances
    distances = haversine(pickup_lat, pickup_lon, drop_lat, drop_lon)

    # Delivery times (target)
    delivery_times = np.array([
        compute_delivery_time(distances[i], traffic[i], hours[i], is_weekend[i])
        for i in range(n)
    ])

    df = pd.DataFrame({
        "order_id":              [f"ORD-{i+1:06d}" for i in range(n)],
        "order_timestamp":       timestamps,
        "pickup_lat":            pickup_lat,
        "pickup_lon":            pickup_lon,
        "drop_lat":              drop_lat,
        "drop_lon":              drop_lon,
        "traffic_level":         traffic,
        "hour_of_day":           hours,
        "day_of_week":           days_of_week,
        "is_weekend":            is_weekend,
        "haversine_distance":    distances.round(4),
        "delivery_time_minutes": delivery_times,
    })

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "raw_data.csv")
    df.to_csv(out_path, index=False)
    print(f"[DataGen] Saved {n:,} records → {out_path}")
    print(f"[DataGen] Delivery time stats:\n{df['delivery_time_minutes'].describe().round(2)}\n")
    return df


if __name__ == "__main__":
    generate_dataset()
