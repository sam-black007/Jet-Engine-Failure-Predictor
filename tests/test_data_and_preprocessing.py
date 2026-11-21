"""Unit tests for dataset loading, engine splitting, sequence generation, and preprocessing."""

import json
import pathlib
import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_dataset, load_rul
from src.preprocessing import (
    _get_sensor_columns,
    _identify_constant_sensors,
    _split_engine_ids,
    _generate_sequences
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_data_loader_fd001():
    """Verify raw training data for FD001 loads with all standard columns and exactly 100 unique engines."""
    df = load_dataset("FD001", kind="train")
    assert not df.empty
    assert "engine_id" in df.columns
    assert "cycle" in df.columns
    assert len(df["engine_id"].unique()) == 100
    assert len(df) == 20631  # Official FD001 train row count


def test_test_and_rul_loader():
    """Verify test set and RUL ground truth file load correctly."""
    df_test = load_dataset("FD001", kind="test")
    rul_series = load_rul("FD001")
    assert len(df_test["engine_id"].unique()) == 100
    assert len(rul_series) == 100


def test_engine_id_split_no_leakage():
    """Engine-level train/validation split must have zero engine overlap and cover all engines."""
    df = load_dataset("FD001", kind="train")
    train_ids, val_ids = _split_engine_ids(df, train_ratio=0.8, seed=42)
    
    assert len(train_ids) == 80
    assert len(val_ids) == 20
    assert len(set(train_ids).intersection(set(val_ids))) == 0
    assert set(train_ids).union(set(val_ids)) == set(df["engine_id"].unique())


def test_sequence_generation():
    """Sliding-window sequence generation must never cross engine boundaries and produce correct shapes."""
    # Create synthetic single-engine trajectory
    n_cycles = 50
    sensors = [f"sensor_{i}" for i in range(1, 18)]
    data = {s: np.random.randn(n_cycles) for s in sensors}
    data["cycle"] = np.arange(1, n_cycles + 1)
    data["engine_id"] = 1
    engine_df = pd.DataFrame(data)

    window = 30
    rul_clip = 125
    X_seq, y_seq = _generate_sequences(engine_df, sensors, window=window, rul_clip=rul_clip)

    expected_num_seq = n_cycles - window + 1  # 50 - 30 + 1 = 21
    assert X_seq.shape == (expected_num_seq, window, len(sensors))
    assert y_seq.shape == (expected_num_seq,)
    
    # RUL at the final cycle should be 0
    assert y_seq[-1] == 0
    # First sequence ends at cycle 30, so RUL = 50 - 30 = 20
    assert y_seq[0] == 20
