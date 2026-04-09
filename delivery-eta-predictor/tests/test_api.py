# tests/test_api.py — Unit and integration tests for the ETA API

import sys
import os
import json
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


# ─────────────────────────────────────────────────────────────
# Feature Engineering Tests
# ─────────────────────────────────────────────────────────────

class TestFeatureEngineering:
    """Tests for the feature engineering pipeline."""

    def test_haversine_distance_same_point(self):
        from src.feature_engineering import haversine_distance
        dist = haversine_distance(12.9716, 77.5946, 12.9716, 77.5946)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_haversine_distance_known(self):
        """Bangalore to Mysore is ~145 km."""
        from src.feature_engineering import haversine_distance
        dist = haversine_distance(12.9716, 77.5946, 12.2958, 76.6394)
        assert 120 < dist < 160

    def test_bearing_north(self):
        from src.feature_engineering import compute_bearing
        # Moving north: same lon, higher lat
        bearing = compute_bearing(12.0, 77.0, 13.0, 77.0)
        assert 355 < bearing or bearing < 5  # ~0 degrees

    def test_peak_hour_detection(self):
        from src.feature_engineering import extract_temporal_features
        import pandas as pd
        # Evening peak at 18:00
        ts = pd.Series(["2024-01-15 18:30:00"])
        _, _, is_peak, _ = extract_temporal_features(ts)
        assert is_peak.values[0] == 1

    def test_non_peak_hour(self):
        from src.feature_engineering import extract_temporal_features
        import pandas as pd
        # Midday, not peak
        ts = pd.Series(["2024-01-15 12:00:00"])
        _, _, is_peak, _ = extract_temporal_features(ts)
        assert is_peak.values[0] == 0

    def test_weekend_detection(self):
        from src.feature_engineering import extract_temporal_features
        import pandas as pd
        ts = pd.Series(["2024-01-13 14:00:00"])  # Saturday
        _, _, _, is_weekend = extract_temporal_features(ts)
        assert is_weekend.values[0] == 1

    def test_traffic_encoding(self):
        from src.feature_engineering import encode_traffic
        import pandas as pd
        traffic = pd.Series(["low", "medium", "high", "very_high"])
        encoded = encode_traffic(traffic)
        assert list(encoded) == [0, 1, 2, 3]

    def test_build_inference_features_shape(self):
        from src.feature_engineering import build_inference_features
        payload = {
            "pickup_lat": 12.9716,
            "pickup_lon": 77.5946,
            "drop_lat":   13.0012,
            "drop_lon":   77.6141,
            "order_timestamp": "2024-03-15 18:30:00",
            "traffic_level": "high",
        }
        df = build_inference_features(payload)
        assert len(df) == 1
        assert "haversine_distance" in df.columns
        assert "is_peak_hour" in df.columns
        assert "traffic_encoded" in df.columns


# ─────────────────────────────────────────────────────────────
# Validation Tests
# ─────────────────────────────────────────────────────────────

class TestValidation:
    """Tests for payload validation logic."""

    def _valid_payload(self):
        return {
            "pickup_lat": 12.9716,
            "pickup_lon": 77.5946,
            "drop_lat":   13.0012,
            "drop_lon":   77.6141,
        }

    def test_valid_payload(self):
        from src.predict import validate_payload
        ok, msg = validate_payload(self._valid_payload())
        assert ok is True
        assert msg == ""

    def test_missing_field(self):
        from src.predict import validate_payload
        payload = self._valid_payload()
        del payload["pickup_lat"]
        ok, msg = validate_payload(payload)
        assert ok is False
        assert "pickup_lat" in msg

    def test_invalid_traffic(self):
        from src.predict import validate_payload
        payload = self._valid_payload()
        payload["traffic_level"] = "gridlock"
        ok, msg = validate_payload(payload)
        assert ok is False

    def test_invalid_latitude(self):
        from src.predict import validate_payload
        payload = self._valid_payload()
        payload["pickup_lat"] = 200  # invalid
        ok, msg = validate_payload(payload)
        assert ok is False

    def test_invalid_type(self):
        from src.predict import validate_payload
        payload = self._valid_payload()
        payload["drop_lon"] = "not_a_number"
        ok, msg = validate_payload(payload)
        assert ok is False


# ─────────────────────────────────────────────────────────────
# Flask API Tests (requires trained models)
# ─────────────────────────────────────────────────────────────

MODELS_AVAILABLE = all(
    os.path.exists(os.path.join("models", f))
    for f in ["xgboost_model.pkl", "random_forest_model.pkl", "scaler.pkl", "model_metadata.json"]
)


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Trained models not found — run train.py first")
class TestFlaskAPI:
    """Integration tests for the Flask endpoints."""

    @pytest.fixture
    def client(self):
        from app import app
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c

    def _valid_body(self):
        return {
            "pickup_lat": 12.9716,
            "pickup_lon": 77.5946,
            "drop_lat":   13.0012,
            "drop_lon":   77.6141,
            "traffic_level": "medium",
            "order_timestamp": "2024-03-15 18:30:00",
        }

    def test_health_endpoint(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"

    def test_index_endpoint(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_predict_valid(self, client):
        resp = client.post(
            "/predict",
            data=json.dumps(self._valid_body()),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "predicted_eta_minutes" in data
        assert data["predicted_eta_minutes"] > 0
        assert "confidence_range" in data
        assert "distance_km" in data

    def test_predict_missing_field(self, client):
        body = self._valid_body()
        del body["drop_lat"]
        resp = client.post(
            "/predict",
            data=json.dumps(body),
            content_type="application/json",
        )
        assert resp.status_code == 422

    def test_predict_non_json(self, client):
        resp = client.post("/predict", data="not json")
        assert resp.status_code == 400

    def test_batch_predict(self, client):
        body = {"orders": [self._valid_body(), self._valid_body()]}
        resp = client.post(
            "/predict/batch",
            data=json.dumps(body),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2

    def test_model_info(self, client):
        resp = client.get("/model/info")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "features" in data

    def test_invalid_endpoint(self, client):
        resp = client.get("/nonexistent")
        assert resp.status_code == 404

    def test_random_forest_model(self, client):
        body = self._valid_body()
        body["model"] = "random_forest"
        resp = client.post(
            "/predict",
            data=json.dumps(body),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["model_used"] == "random_forest"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
