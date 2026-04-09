# 🚚 Delivery Time Prediction System (ETA)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-green.svg)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end machine learning solution to predict delivery time (ETA) using geospatial and temporal data, simulating real-world logistics and last-mile delivery scenarios.

---

## 📌 Project Overview

This project builds a **real-time ETA prediction system** that estimates accurate delivery durations using:
- 📍 Pickup and drop coordinates (latitude/longitude)
- 🕐 Order timestamps and time-based patterns
- 🚦 Traffic conditions and congestion levels
- 📏 Haversine distance calculation for precise geographic measurement

---

## 🏗️ Project Structure

```
delivery-eta-predictor/
├── data/
│   ├── generate_data.py         # Synthetic dataset generator
│   └── raw_data.csv             # Generated dataset (after running generator)
├── src/
│   ├── feature_engineering.py   # Feature extraction & transformation pipeline
│   ├── model_training.py        # Model training, tuning & evaluation
│   └── predict.py               # Inference / prediction utilities
├── models/
│   ├── xgboost_model.pkl        # Trained XGBoost model
│   ├── random_forest_model.pkl  # Trained Random Forest model
│   └── scaler.pkl               # Feature scaler
├── notebooks/
│   └── EDA_and_Modeling.ipynb   # Exploratory Data Analysis notebook
├── tests/
│   └── test_api.py              # API unit tests
├── static/
│   └── swagger.json             # API documentation
├── app.py                       # Flask REST API
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── train.py                     # Training entry point (CLI)
└── README.md
```

---

## ⚙️ Features

### Feature Engineering Pipeline
| Feature | Description |
|---|---|
| `haversine_distance` | Great-circle distance between pickup & drop |
| `bearing` | Direction of travel (azimuth angle) |
| `hour_of_day` | Hour extracted from order timestamp |
| `day_of_week` | Day of week (0=Monday, 6=Sunday) |
| `is_peak_hour` | Flag for rush hours (8–10 AM, 5–8 PM) |
| `is_weekend` | Flag for Saturday/Sunday |
| `traffic_encoded` | Encoded traffic level (low/medium/high/very_high) |
| `distance_x_traffic` | Interaction feature: distance × traffic |
| `speed_estimate` | Estimated speed based on time/traffic |

### Models Trained
- **XGBoost Regressor** — Gradient boosted trees with hyperparameter tuning
- **Random Forest Regressor** — Ensemble of decision trees

### Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/delivery-eta-predictor.git
cd delivery-eta-predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Synthetic Data
```bash
python data/generate_data.py
```

### 4. Train Models
```bash
python train.py
```

### 5. Run the Flask API
```bash
python app.py
```

The API will be live at `http://localhost:5000`

---

## 📡 API Endpoints

### `POST /predict`
Predict delivery time for a single order.

**Request Body:**
```json
{
  "pickup_lat": 12.9716,
  "pickup_lon": 77.5946,
  "drop_lat": 13.0012,
  "drop_lon": 77.6141,
  "order_timestamp": "2024-03-15 18:30:00",
  "traffic_level": "high"
}
```

**Response:**
```json
{
  "predicted_eta_minutes": 34.7,
  "distance_km": 4.21,
  "traffic_level": "high",
  "is_peak_hour": true,
  "model_used": "xgboost",
  "confidence_range": {
    "lower": 29.5,
    "upper": 39.9
  }
}
```

### `POST /predict/batch`
Predict delivery times for multiple orders at once.

### `GET /health`
Health check endpoint.

### `GET /model/info`
Returns model metadata and performance metrics.

---

## 📊 Model Performance

| Model | MAE (min) | RMSE (min) | R² |
|---|---|---|---|
| XGBoost | ~3.2 | ~4.8 | ~0.94 |
| Random Forest | ~3.8 | ~5.4 | ~0.92 |

---

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

---

## 🔧 Configuration

Edit `config.py` to adjust:
- Traffic multipliers
- Peak hour windows
- Model hyperparameters
- API settings

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙋 Author

Built as a demonstration of end-to-end ML system design, covering feature engineering, model development, REST API deployment, and scalable architecture.
