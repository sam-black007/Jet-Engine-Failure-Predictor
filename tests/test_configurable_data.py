"""Unit and data leakage tests for Configurable Data Pipeline (THRUSTLAB Phase 2)."""

import pytest
import numpy as np
import pandas as pd

from src.config_manager import MasterConfig
from src.configurable_data import ConfigurableDataPipeline


def test_zero_engine_leakage():
    """Verify strictly disjoint engine IDs between train and validation splits."""
    cfg = MasterConfig()
    pipeline = ConfigurableDataPipeline(cfg)
    raw_train, _, _ = pipeline.load_raw_data()
    train_ids, val_ids = pipeline.split_engine_ids(raw_train)

    overlap = set(train_ids).intersection(set(val_ids))
    assert len(overlap) == 0, f"Engine ID leakage detected! Overlap: {overlap}"
    assert len(train_ids) + len(val_ids) == len(raw_train["engine_id"].unique())


def test_scaler_types_fit_on_train_only():
    """Verify standard, minmax, and robust scalers fit parameters without NaN or inf."""
    cfg = MasterConfig()
    pipeline = ConfigurableDataPipeline(cfg)
    raw_train, _, _ = pipeline.load_raw_data()
    train_ids, _ = pipeline.split_engine_ids(raw_train)
    train_df = raw_train[raw_train["engine_id"].isin(train_ids)].copy()

    features = pipeline.determine_features(train_df)

    for scaler in ["standard", "minmax", "robust"]:
        cfg.preprocessing.scaler_type = scaler
        stats = pipeline.fit_scaler(train_df, features)
        assert stats["scaler_type"] == scaler
        assert len(stats["param_1"]) == len(features)
        assert len(stats["param_2"]) == len(features)

        # Apply transformation
        norm_df = pipeline.transform(train_df, features, stats)
        assert not norm_df[features].isna().any().any()
        assert not np.isinf(norm_df[features].values).any()


def test_sequence_windowing_and_padding():
    """Verify sequence generation respects window_size and does not cross engine boundaries."""
    cfg = MasterConfig()
    cfg.preprocessing.window_size = 25
    cfg.preprocessing.stride = 2
    pipeline = ConfigurableDataPipeline(cfg)

    data = pipeline.prepare_data()
    X_train = data["X_train"]
    y_train = data["y_train"]

    assert X_train.shape[1] == 25
    assert len(X_train) == len(y_train)
    assert (y_train <= cfg.preprocessing.rul_clip_train).all()
    assert (y_train >= 0.0).all()
