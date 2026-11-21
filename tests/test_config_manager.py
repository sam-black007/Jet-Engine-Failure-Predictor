"""Unit and validation tests for Centralized Configuration Manager (JETWISE AI Phase 2)."""

import pytest
from pathlib import Path
from pydantic import ValidationError

from src.config_manager import (
    ConfigManager,
    MasterConfig,
    DatasetConfig,
    SensorConfig,
    PreprocessingConfig,
    ModelConfig,
    LossConfig,
    TrainingConfig,
    PredictionConfig,
    EvaluationConfig,
)


def test_default_config_instantiation():
    """Verify default MasterConfig instantiates with valid defaults."""
    cfg = MasterConfig()
    assert cfg.version == "2.0.0"
    assert cfg.dataset.subset == "FD001"
    assert cfg.model.architecture == "lstm"
    assert cfg.loss.name == "pinball"
    assert cfg.loss.pinball_quantile == 0.35
    assert cfg.training.epochs >= 1
    assert cfg.prediction.strategy == "standard"


def test_transformer_head_validation():
    """Verify validator catches d_model not divisible by nhead."""
    with pytest.raises(ValidationError):
        ModelConfig(
            architecture="transformer",
            transformer_d_model=65,  # 65 not divisible by 4
            transformer_nhead=4
        )


def test_pinball_quantile_range():
    """Verify pinball quantile must be strictly in (0, 1)."""
    with pytest.raises(ValidationError):
        LossConfig(name="pinball", pinball_quantile=1.5)

    with pytest.raises(ValidationError):
        LossConfig(name="pinball", pinball_quantile=-0.1)


def test_deep_merge_and_overrides(tmp_path):
    """Verify ConfigManager loads and deep merges runtime overrides correctly."""
    cm = ConfigManager(tmp_path)
    overrides = {
        "dataset": {"subset": "FD002"},
        "model": {"architecture": "tcn", "tcn_kernel_size": 5},
        "training": {"learning_rate": 0.005}
    }
    cfg = cm.load_config(overrides=overrides)
    assert cfg.dataset.subset == "FD002"
    assert cfg.model.architecture == "tcn"
    assert cfg.model.tcn_kernel_size == 5
    assert cfg.training.learning_rate == 0.005
    # Non-overridden values must retain defaults
    assert cfg.preprocessing.window_size == 30
    assert cfg.loss.pinball_quantile == 0.35


def test_snapshot_saving(tmp_path):
    """Verify frozen snapshots can be saved and reloaded."""
    cm = ConfigManager(tmp_path)
    cfg = cm.load_config(overrides={"profile_name": "test_profile"})
    snap_yaml = cm.save_snapshot(cfg, tmp_path / "snapshots")
    assert snap_yaml.exists()

    reloaded_dict = cm.load_yaml(snap_yaml)
    assert reloaded_dict["profile_name"] == "test_profile"
    reloaded_cfg = MasterConfig.model_validate(reloaded_dict)
    assert reloaded_cfg.profile_name == "test_profile"


def test_list_and_load_profiles():
    """Verify all repository profiles in configs/experiments/ load and validate cleanly."""
    cm = ConfigManager()
    profiles = cm.list_profiles()
    assert len(profiles) >= 4
    for prof_name in profiles:
        cfg = cm.load_profile(prof_name)
        assert cfg.profile_name == prof_name
        assert cfg.model.architecture in [
            "lstm", "attention_lstm", "tcn", "transformer", 
            "linear_regression", "random_forest", "gradient_boosting"
        ]
