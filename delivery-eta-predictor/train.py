#!/usr/bin/env python3
# train.py — CLI entry point for the full training pipeline

import argparse
import os
import sys

from config import RAW_DATA_PATH
from data.generate_data import generate_dataset
from src.model_training import run_training


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Delivery ETA Prediction models",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate a fresh synthetic dataset before training.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10_000,
        help="Number of synthetic records to generate (default: 10000).",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run GridSearchCV hyperparameter tuning for XGBoost (slower).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=RAW_DATA_PATH,
        help=f"Path to training CSV (default: {RAW_DATA_PATH}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  🚚 Delivery ETA Predictor — Training Entry Point")
    print("=" * 60 + "\n")

    # ── Step 1: Generate data if requested or missing ─────────
    if args.generate_data or not os.path.exists(args.data_path):
        print("[Setup] Generating synthetic dataset...")
        generate_dataset(n_samples=args.n_samples)
    else:
        print(f"[Setup] Using existing data: {args.data_path}")

    # ── Step 2: Run the training pipeline ─────────────────────
    run_training(data_path=args.data_path, tune_xgb=args.tune)

    print("\n[Done] Models saved to /models/")
    print("[Done] Run `python app.py` to start the prediction API.\n")


if __name__ == "__main__":
    main()
