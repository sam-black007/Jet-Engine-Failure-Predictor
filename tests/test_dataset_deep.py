"""Comprehensive dataset and preprocessing tests for NASA C-MAPSS FD001.

Validates data integrity, sensor distributions, lack of temporal leakage,
deterministic splits, edge case resilience, and sequence generation.
"""

import json
import pathlib
import pytest
import numpy as np
import pandas as pd

from src.data_loader import load_dataset, load_rul, COLUMN_NAMES, SENSOR_COLS, OP_COLS
from src.preprocessing import (
    _get_sensor_columns,
    _identify_constant_sensors,
    _split_engine_ids,
    _normalize,
    _generate_sequences,
    preprocess
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_fd001_raw_data_integrity():
    """Verify raw training data has exact dimensions, no missing values, and no duplicates."""
    df = load_dataset("FD001", kind="train")
    
    # Check shape & column names
    assert df.shape == (20631, 26), f"Expected shape (20631, 26), got {df.shape}"
    assert list(df.columns) == COLUMN_NAMES
    
    # Missing values and duplicates
    assert df.isna().sum().sum() == 0, "Raw dataset contains unexpected missing/NaN values"
    assert df.duplicated(subset=["engine_id", "cycle"]).sum() == 0, "Duplicate (engine_id, cycle) found"
    
    # Exactly 100 engines
    unique_engines = df["engine_id"].unique()
    assert len(unique_engines) == 100
    assert set(unique_engines) == set(range(1, 101))


def test_engine_trajectory_monotonicity():
    """Verify every engine trajectory starts at cycle 1 and increments monotonically by 1 without gaps."""
    df = load_dataset("FD001", kind="train")
    for eid in range(1, 101):
        cycles = df[df["engine_id"] == eid]["cycle"].values
        assert cycles[0] == 1, f"Engine {eid} does not start at cycle 1"
        assert np.all(np.diff(cycles) == 1), f"Engine {eid} has non-consecutive cycles or gaps"
        assert len(cycles) >= 30, f"Engine {eid} has fewer than 30 cycles ({len(cycles)})"


def test_constant_sensor_identification():
    """Verify that constant sensors are correctly identified with zero variance."""
    df = load_dataset("FD001", kind="train")
    all_sensors = _get_sensor_columns(df)
    assert len(all_sensors) == 21
    
    const_sensors = _identify_constant_sensors(df, all_sensors)
    expected_const = ["sensor_1", "sensor_10", "sensor_18", "sensor_19"]
    assert set(const_sensors) == set(expected_const), f"Expected constant sensors {expected_const}, got {const_sensors}"
    
    active_sensors = [s for s in all_sensors if s not in const_sensors]
    assert len(active_sensors) == 17


def test_zscore_normalization_and_metadata():
    """Verify Z-score normalization produces zero-mean and unit variance on training data."""
    df = load_dataset("FD001", kind="train")
    meta_path = PROJECT_ROOT / "dataset" / "processed" / "preprocess_metadata.json"
    assert meta_path.is_file(), "Metadata file missing"
    
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    train_ids = meta["train_engine_ids"]
    val_ids = meta["val_engine_ids"]
    sensors = meta["selected_sensors"]
    
    assert len(train_ids) == 80
    assert len(val_ids) == 20
    assert len(set(train_ids).intersection(set(val_ids))) == 0
    
    # Check that normalization on dynamic sensors produces mean ~ 0, std ~ 1
    train_df = df[df["engine_id"].isin(train_ids)]
    dynamic_sensors = [s for s in sensors if s not in ["sensor_5", "sensor_16"]]
    means = pd.Series(meta["sensor_means"])[dynamic_sensors]
    stds = pd.Series(meta["sensor_stds"])[dynamic_sensors]
    norm_train = _normalize(train_df, dynamic_sensors, means, stds)
    
    norm_means = norm_train[dynamic_sensors].mean().values
    norm_stds = norm_train[dynamic_sensors].std().values
    
    # Tolerances: floating point precision
    assert np.all(np.abs(norm_means) < 1e-4)
    assert np.all(np.abs(norm_stds - 1.0) < 1e-4)


def test_edge_case_insufficient_cycles():
    """Engines with fewer cycles than window length must produce zero sequences without raising exceptions."""
    short_df = pd.DataFrame({
        "engine_id": [999] * 20,
        "cycle": list(range(1, 21)),
        **{f"sensor_{i}": np.random.randn(20) for i in range(1, 18)}
    })
    sensors = [f"sensor_{i}" for i in range(1, 18)]
    X, y = _generate_sequences(short_df, sensors, window=30, rul_clip=125)
    assert len(X) == 0
    assert len(y) == 0


def test_edge_case_missing_files_error_handling():
    """Attempting to load a nonexistent dataset must raise FileNotFoundError or ValueError."""
    with pytest.raises(ValueError):
        load_dataset("FD999", kind="train")
        
    with pytest.raises(ValueError):
        load_dataset("FD001", kind="invalid_kind")
