"""Unit and Integration Tests for Dataset Adapters (FD001-FD004, N-CMAPSS)."""

import pytest
import numpy as np
import pandas as pd

from src.dataset_adapters.cmapss_adapter import CMAPSSAdapter
from src.dataset_adapters.ncmapss_adapter import NCMAPSSAdapter


def test_cmapss_adapter_initialization_and_invalid_name():
    adapter = CMAPSSAdapter("FD001")
    assert adapter.dataset_name == "FD001"

    with pytest.raises(ValueError, match="Unsupported C-MAPSS subset"):
        CMAPSSAdapter("FD999")


def test_cmapss_adapter_fd001_loading_and_trajectory():
    adapter = CMAPSSAdapter("FD001")
    train_df = adapter.load_train()
    assert isinstance(train_df, pd.DataFrame)
    assert not train_df.empty
    assert "engine_id" in train_df.columns
    assert "cycle" in train_df.columns

    engine_ids = adapter.get_engine_ids(kind="train")
    assert len(engine_ids) == 100
    assert engine_ids[0] == 1

    traj = adapter.get_engine_trajectory(engine_id=1, kind="train")
    assert len(traj) > 0
    assert traj["cycle"].iloc[0] == 1
    assert traj["cycle"].is_monotonic_increasing


def test_cmapss_adapter_active_sensors():
    adapter = CMAPSSAdapter("FD001")
    active = adapter.get_active_sensors()
    # In FD001, invariant sensors (1, 5, 10, 16, 18, 19) have std == 0
    assert len(active) in {14, 15}
    assert "sensor_1" not in active  # Invariant in FD001
    assert "sensor_2" in active      # Active in FD001


def test_cmapss_adapter_fd002_regime_detection():
    adapter = CMAPSSAdapter("FD002")
    train_df = adapter.load_train()
    assert len(train_df) > 0

    regime_df = adapter.detect_flight_regimes(train_df.head(500))
    assert "regime_id" in regime_df.columns
    # Check that regimes are mapped to discrete integers
    assert regime_df["regime_id"].nunique() >= 1


def test_ncmapss_adapter_synthetic_benchmark():
    adapter = NCMAPSSAdapter()
    assert adapter.dataset_name == "N-CMAPSS"

    train_df = adapter.load_train()
    assert isinstance(train_df, pd.DataFrame)
    assert "T24" in train_df.columns
    assert "T50" in train_df.columns
    assert "alt" in train_df.columns

    meta = adapter.get_metadata()
    assert meta["category"] == "Commercial Turbofan (90,000 lbf)"
    assert meta["available_sensors_count"] == 14

    traj = adapter.get_engine_trajectory(engine_id=1, kind="train")
    assert len(traj) > 0
    assert traj["cycle"].is_monotonic_increasing
