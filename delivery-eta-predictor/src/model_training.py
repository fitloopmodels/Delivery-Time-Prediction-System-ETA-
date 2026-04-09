# src/model_training.py — Model training, evaluation, and persistence

import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import TRAINING_CONFIG, FEATURE_CONFIG, MODEL_DIR, RAW_DATA_PATH
from src.feature_engineering import build_features


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def load_and_prepare_data(data_path: str = RAW_DATA_PATH):
    """Load raw CSV and run the feature engineering pipeline."""
    print(f"[Training] Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"[Training] Raw data shape: {df.shape}")

    featured_df = build_features(df, training=True)
    print(f"[Training] Featured data shape: {featured_df.shape}")

    target = FEATURE_CONFIG["target_column"]
    X = featured_df.drop(columns=[target])
    y = featured_df[target]
    return X, y


def compute_metrics(y_true, y_pred, label=""):
    """Return a dict of regression metrics."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    if label:
        print(f"\n  [{label}]")
        print(f"    MAE  : {mae:.4f} min")
        print(f"    RMSE : {rmse:.4f} min")
        print(f"    R²   : {r2:.4f}")
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


def save_artifact(obj, filename: str):
    """Pickle an object to the models/ directory."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[Training] Saved → {path}")
    return path


def load_artifact(filename: str):
    """Load a pickled artifact from models/."""
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────
# Training Routines
# ─────────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, tune: bool = False):
    """Train XGBoost regressor, optionally with grid search."""
    cfg = TRAINING_CONFIG

    if tune:
        print("\n[Training] Running GridSearchCV for XGBoost...")
        base = XGBRegressor(random_state=cfg["random_state"], n_jobs=-1, verbosity=0)
        gs = GridSearchCV(
            base,
            param_grid=cfg["xgb_param_grid"],
            cv=cfg["cv_folds"],
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            verbose=1,
        )
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        print(f"[Training] Best XGB params: {gs.best_params_}")
    else:
        model = XGBRegressor(**cfg["xgboost_params"], verbosity=0)
        model.fit(X_train, y_train)

    return model


def train_random_forest(X_train, y_train):
    """Train Random Forest regressor."""
    cfg = TRAINING_CONFIG
    model = RandomForestRegressor(**cfg["random_forest_params"])
    model.fit(X_train, y_train)
    return model


# ─────────────────────────────────────────────────────────────
# Feature Importance
# ─────────────────────────────────────────────────────────────

def print_feature_importance(model, feature_names, top_n=10):
    """Print top-N most important features."""
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_names)
        imp = imp.sort_values(ascending=False).head(top_n)
        print(f"\n  Top-{top_n} Feature Importances:")
        for feat, score in imp.items():
            bar = "█" * int(score * 50)
            print(f"    {feat:<28} {score:.4f} {bar}")


# ─────────────────────────────────────────────────────────────
# Cross-Validation
# ─────────────────────────────────────────────────────────────

def cross_validate_model(model, X, y, label="Model"):
    """Run k-fold CV and print mean MAE."""
    cfg = TRAINING_CONFIG
    scores = cross_val_score(
        model, X, y,
        cv=cfg["cv_folds"],
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    mae_scores = -scores
    print(f"\n  [{label}] {cfg['cv_folds']}-Fold CV MAE: "
          f"{mae_scores.mean():.4f} ± {mae_scores.std():.4f} min")
    return mae_scores


# ─────────────────────────────────────────────────────────────
# Full Training Pipeline
# ─────────────────────────────────────────────────────────────

def run_training(data_path: str = RAW_DATA_PATH, tune_xgb: bool = False):
    """
    Full training pipeline:
    1. Load & engineer features
    2. Train/test split + scaling
    3. Train XGBoost and Random Forest
    4. Evaluate on test set
    5. Save models, scaler, metadata
    """
    cfg = TRAINING_CONFIG
    print("=" * 60)
    print("  Delivery ETA Predictor — Training Pipeline")
    print("=" * 60)

    # ── Data ────────────────────────────────────────────────
    X, y = load_and_prepare_data(data_path)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
    )
    print(f"\n[Training] Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Scaling ─────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    save_artifact(scaler, "scaler.pkl")

    # ── XGBoost ─────────────────────────────────────────────
    print("\n[Training] Training XGBoost...")
    t0 = time.time()
    xgb_model = train_xgboost(X_train, y_train, tune=tune_xgb)
    print(f"[Training] XGBoost trained in {time.time()-t0:.1f}s")

    xgb_pred = xgb_model.predict(X_test)
    xgb_metrics = compute_metrics(y_test, xgb_pred, "XGBoost — Test Set")
    print_feature_importance(xgb_model, feature_names)
    cross_validate_model(xgb_model, X, y, "XGBoost")
    save_artifact(xgb_model, "xgboost_model.pkl")

    # ── Random Forest ────────────────────────────────────────
    print("\n[Training] Training Random Forest...")
    t0 = time.time()
    rf_model = train_random_forest(X_train_sc, y_train)
    print(f"[Training] Random Forest trained in {time.time()-t0:.1f}s")

    rf_pred = rf_model.predict(X_test_sc)
    rf_metrics = compute_metrics(y_test, rf_pred, "Random Forest — Test Set")
    print_feature_importance(rf_model, feature_names)
    save_artifact(rf_model, "random_forest_model.pkl")

    # ── Save Metadata ────────────────────────────────────────
    metadata = {
        "feature_columns": feature_names,
        "target_column": FEATURE_CONFIG["target_column"],
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "models": {
            "xgboost": {"file": "xgboost_model.pkl", "metrics": xgb_metrics},
            "random_forest": {"file": "random_forest_model.pkl", "metrics": rf_metrics},
        },
    }
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[Training] Metadata saved → {meta_path}")

    print("\n" + "=" * 60)
    print("  Training Complete ✓")
    print(f"  XGBoost  MAE={xgb_metrics['mae']} min  RMSE={xgb_metrics['rmse']} min  R²={xgb_metrics['r2']}")
    print(f"  RandForest MAE={rf_metrics['mae']} min  RMSE={rf_metrics['rmse']} min  R²={rf_metrics['r2']}")
    print("=" * 60)

    return xgb_model, rf_model, scaler, metadata


if __name__ == "__main__":
    run_training()
