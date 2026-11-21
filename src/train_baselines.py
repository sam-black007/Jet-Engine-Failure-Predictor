"""Train baseline regression models for the Jet Engine Failure Predictor.

Trains:
1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

Uses scikit-learn on flattened sliding-window sequences.
Evaluates on the validation set using:
- Sliding-window validation (sequence-level)
- Per-engine truncated-trajectory evaluation
Saves trained models to models/baselines/ and updates models/performance_metrics.json.
"""

import json
import os
import pathlib
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from src.data_loader import load_dataset
from src.evaluation import compute_metrics, evaluate_per_engine


def train_and_evaluate_baselines():
    project_root = pathlib.Path(__file__).resolve().parent.parent
    processed_dir = project_root / "dataset" / "processed"
    models_dir = project_root / "models"
    baselines_dir = models_dir / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    with open(processed_dir / "preprocess_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    sensor_cols = meta["selected_sensors"]
    # Reconstruct means & stds as numpy arrays
    sensor_means = np.array([meta["sensor_means"][s] for s in sensor_cols], dtype=np.float32)
    sensor_stds = np.array([meta["sensor_stds"][s] for s in sensor_cols], dtype=np.float32)
    val_engine_ids = meta["val_engine_ids"]
    window = meta["window_length"]
    rul_clip = meta["rul_clip"]
    
    # Load raw data for per-engine evaluation
    raw_df = load_dataset(meta.get("dataset_subset", "FD001"), kind="train")
    
    # Load processed arrays
    print("Loading preprocessed training and validation sequences...")
    X_train = np.load(processed_dir / "X_train.npy")  # (N, 30, 17)
    y_train = np.load(processed_dir / "y_train.npy")
    X_val = np.load(processed_dir / "X_val.npy")      # (M, 30, 17)
    y_val = np.load(processed_dir / "y_val.npy")
    
    # Flatten 3D windows for sklearn estimators: (samples, 30 * 17)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=60, max_depth=5, learning_rate=0.1, random_state=42)
    }
    
    # Load existing performance metrics if any
    metrics_file = models_dir / "performance_metrics.json"
    if metrics_file.is_file():
        with open(metrics_file, "r", encoding="utf-8") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}
        
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train_flat, y_train)
        
        # Save model
        save_path = baselines_dir / f"{name}.pkl"
        joblib.dump(model, save_path)
        print(f"Saved model to {save_path}")
        
        # 1. Sliding-window sequence evaluation
        y_pred_val = model.predict(X_val_flat)
        seq_metrics = compute_metrics(y_val, y_pred_val)
        print(f"Sequence-level Validation Metrics:")
        print(f"  MAE: {seq_metrics['MAE']}")
        print(f"  RMSE: {seq_metrics['RMSE']}")
        print(f"  NASA Score: {seq_metrics['NASA_score']}")
        
        # 2. Per-engine truncated-trajectory evaluation
        engine_eval = evaluate_per_engine(
            model=model,
            raw_df=raw_df,
            val_engine_ids=val_engine_ids,
            sensor_cols=sensor_cols,
            sensor_means=sensor_means,
            sensor_stds=sensor_stds,
            window_length=window,
            rul_clip=rul_clip,
            is_3d=False
        )
        eng_metrics = engine_eval["metrics"]
        print(f"Per-engine Truncated-trajectory Metrics ({engine_eval['num_engines_evaluated']} engines):")
        print(f"  MAE: {eng_metrics['MAE']}")
        print(f"  RMSE: {eng_metrics['RMSE']}")
        print(f"  NASA Score: {eng_metrics['NASA_score']}")
        
        all_metrics[name] = {
            "sequence_level": seq_metrics,
            "per_engine": eng_metrics
        }
        
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nUpdated {metrics_file} successfully.")


if __name__ == "__main__":
    train_and_evaluate_baselines()
