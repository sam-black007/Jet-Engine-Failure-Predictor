"""Unit tests for ExperimentTracker and ModelRegistry."""

import pytest
from pathlib import Path
import pandas as pd

from src.config_manager import MasterConfig
from src.experiment_tracker import ExperimentTracker
from src.model_registry import ModelRegistry


def test_experiment_tracker_crud(tmp_path):
    """Verify logging, retrieval, metric updating, and CSV export in ExperimentTracker."""
    db_path = tmp_path / "test_experiments.db"
    tracker = ExperimentTracker(db_path)

    cfg = MasterConfig(experiment_id="test_exp_001", profile_name="test_profile")
    train_res = {
        "best_epoch": 12,
        "training_duration_seconds": 45.2,
        "best_val_nasa": 350.25,
        "final_val_mae": 11.2,
        "checkpoint_path": "models/test.pt",
        "history": {"val_loss": [12.0, 10.5]}
    }
    test_res = {
        "metrics_raw": {"NASA_score": 380.5, "MAE": 12.1, "RMSE": 16.4},
        "metrics_clipped": {"NASA_score": 290.0}
    }

    exp_id = tracker.log_experiment(cfg, train_res, test_res)
    assert exp_id == "test_exp_001"

    row = tracker.get_experiment("test_exp_001")
    assert row is not None
    assert row["model_architecture"] == cfg.model.architecture
    assert row["test_nasa_raw"] == 380.5

    df = tracker.list_experiments()
    assert len(df) == 1
    assert df.iloc[0]["experiment_id"] == "test_exp_001"

    csv_out = tracker.export_csv(tmp_path / "exported.csv")
    assert csv_out.exists()


def test_model_registry_crud(tmp_path):
    """Verify model registration, active tag management, and listing in ModelRegistry."""
    reg_path = tmp_path / "test_registry.json"
    registry = ModelRegistry(reg_path)

    entry = registry.register_model(
        tag="test_model",
        name="Test Model Architecture",
        architecture="attention_lstm",
        checkpoint_path="models/test.pt",
        metrics={"NASA_score": 350.0},
        is_active=True
    )
    assert entry["tag"] == "test_model"
    assert registry.get_active_tag() == "test_model"

    models = registry.list_models()
    assert len(models) == 1
    assert models[0]["tag"] == "test_model"
