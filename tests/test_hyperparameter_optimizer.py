"""Unit tests for HyperparameterOptimizer with Optuna."""

import pytest
from pathlib import Path
from src.config_manager import MasterConfig
from src.hyperparameter_optimizer import HyperparameterOptimizer


def test_hyperparameter_optimizer_smoke(tmp_path):
    """Verify Optuna study creates trials and records best parameters."""
    cfg = MasterConfig()
    cfg.dataset.subset = "FD001"
    optimizer = HyperparameterOptimizer(
        base_config=cfg,
        study_name="test_study",
        storage_dir=tmp_path
    )

    results = optimizer.optimize(
        architecture="lstm",
        objective_metric="nasa_score",
        n_trials=2,
        max_epochs_per_trial=1,
        sampler_type="random"
    )

    assert "best_params" in results
    assert "best_value" in results
    assert results["n_trials_completed"] == 2
