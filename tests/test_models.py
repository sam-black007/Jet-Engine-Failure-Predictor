"""Unit tests for trained model loading, inference shapes, and prediction consistency."""

import json
import pathlib
import joblib
import numpy as np
import pytest
import torch

from src.train_lstm import LSTMPredictor
from src.evaluation import evaluate_per_engine
from src.data_loader import load_dataset


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def test_baseline_models_exist_and_predict():
    """Verify all three baseline regression models load and generate valid predictions."""
    baselines = ["linear_regression", "random_forest", "gradient_boosting"]
    dummy_input = np.zeros((2, 30 * 17), dtype=np.float32)

    for name in baselines:
        model_path = MODELS_DIR / "baselines" / f"{name}.pkl"
        assert model_path.is_file(), f"Model file {model_path} missing!"
        
        model = joblib.load(model_path)
        assert hasattr(model, "predict"), f"{name} must implement predict()"
        
        preds = model.predict(dummy_input)
        assert len(preds) == 2, f"Expected 2 predictions, got {len(preds)}"
        assert isinstance(preds[0], (float, np.floating)), f"Prediction should be float, got {type(preds[0])}"


def test_lstm_model_checkpoint_and_inference():
    """Verify PyTorch LSTM model checkpoint loads and produces correct output shapes."""
    lstm_path = MODELS_DIR / "lstm" / "lstm_model.pt"
    assert lstm_path.is_file(), f"LSTM checkpoint {lstm_path} missing!"

    checkpoint = torch.load(lstm_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in checkpoint, "Checkpoint must contain model_state_dict"

    model = LSTMPredictor(input_dim=17)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Test 3D numpy array input (batch=3, seq_len=30, features=17)
    dummy_3d = np.random.randn(3, 30, 17).astype(np.float32)
    preds = model.predict(dummy_3d)
    assert preds.shape == (3,), f"Expected shape (3,), got {preds.shape}"
    assert not np.isnan(preds).any(), "LSTM predictions must not contain NaN"

    # Test 2D flattened input (batch=2, 510)
    dummy_2d = np.random.randn(2, 510).astype(np.float32)
    preds_2d = model.predict(dummy_2d)
    assert preds_2d.shape == (2,), f"Expected shape (2,), got {preds_2d.shape}"


def test_per_engine_evaluation_function():
    """Verify evaluate_per_engine computes valid metrics on truncated engine trajectories."""
    meta_path = PROJECT_ROOT / "dataset" / "processed" / "preprocess_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    sensor_cols = meta["selected_sensors"]
    means = np.array([meta["sensor_means"][s] for s in sensor_cols], dtype=np.float32)
    stds = np.array([meta["sensor_stds"][s] for s in sensor_cols], dtype=np.float32)
    raw_df = load_dataset("FD001", kind="train")

    lr = joblib.load(MODELS_DIR / "baselines" / "linear_regression.pkl")
    res = evaluate_per_engine(
        model=lr,
        raw_df=raw_df,
        val_engine_ids=meta["val_engine_ids"][:2],
        sensor_cols=sensor_cols,
        sensor_means=means,
        sensor_stds=stds,
        window_length=30,
        rul_clip=125,
        is_3d=False
    )
    assert res["num_engines_evaluated"] == 2
    assert "metrics" in res
    assert res["metrics"]["MAE"] >= 0.0
    assert res["metrics"]["RMSE"] >= 0.0
    assert res["metrics"]["NASA_score"] >= 0.0


def test_performance_metrics_json_integrity():
    """Verify performance_metrics.json has non-null numbers for all 4 models."""
    metrics_path = MODELS_DIR / "performance_metrics.json"
    assert metrics_path.is_file(), "performance_metrics.json missing!"

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for model_name in ["linear_regression", "random_forest", "gradient_boosting", "lstm"]:
        assert model_name in data, f"Missing {model_name} in performance_metrics.json"
        seq = data[model_name]["sequence_level"]
        eng = data[model_name]["per_engine"]
        assert seq["MAE"] > 0, f"{model_name} sequence MAE must be > 0"
        assert seq["RMSE"] > 0, f"{model_name} sequence RMSE must be > 0"
        assert seq["NASA_score"] > 0, f"{model_name} sequence NASA score must be > 0"
        assert eng["MAE"] > 0, f"{model_name} per-engine MAE must be > 0"
