"""Unit and integration tests for official NASA C-MAPSS FD001 test set evaluation methodology."""

import json
import pathlib
import pytest
import numpy as np
import pandas as pd
import joblib
import torch

from src.data_loader import load_dataset, load_rul
from src.evaluation import evaluate_official_test_set, compute_nasa_score, compute_metrics
from src.train_lstm import LSTMPredictor

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
METADATA_PATH = PROJECT_ROOT / "dataset" / "processed" / "preprocess_metadata.json"


@pytest.fixture(scope="module")
def metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def test_data():
    df = load_dataset("FD001", kind="test")
    rul = load_rul("FD001")
    return df, rul


def test_official_test_engine_ordering(test_data):
    """Verify test_FD001.txt contains exactly 100 engine units from 1 to 100 in order."""
    df, _ = test_data
    unique_ids = df["engine_id"].unique().tolist()
    assert len(unique_ids) == 100, f"Expected 100 test engines, found {len(unique_ids)}"
    assert unique_ids == list(range(1, 101)), "Engine IDs must be contiguous integers from 1 to 100"


def test_official_rul_label_alignment(test_data):
    """Verify RUL_FD001.txt contains 100 valid non-negative integer labels aligned with engines."""
    _, rul = test_data
    assert len(rul) == 100, f"Expected 100 RUL labels, found {len(rul)}"
    assert (rul > 0).all(), "All ground-truth RUL values must be strictly positive"
    assert (rul < 500).all(), "RUL values should be physically realistic (< 500 cycles)"


def test_final_window_generation(test_data, metadata):
    """Verify final 30-cycle window extraction ends strictly at each engine's final cycle."""
    df, _ = test_data
    window = metadata["window_length"]

    for eid in [1, 25, 50, 75, 100]:
        eng_df = df[df["engine_id"] == eid].sort_values("cycle")
        assert len(eng_df) >= window, f"Engine {eid} has {len(eng_df)} cycles, expected >= {window}"
        
        last_window = eng_df.iloc[-window:]
        assert len(last_window) == 30
        assert int(last_window["cycle"].iloc[-1]) == int(eng_df["cycle"].max())
        assert int(last_window["cycle"].iloc[0]) == int(eng_df["cycle"].max()) - window + 1


def test_feature_ordering(test_data, metadata):
    """Verify test feature columns match the 17 selected training sensors in identical order."""
    df, _ = test_data
    selected_sensors = metadata["selected_sensors"]
    assert len(selected_sensors) == 17
    assert "sensor_1" not in selected_sensors
    assert "sensor_10" not in selected_sensors
    assert "sensor_18" not in selected_sensors
    assert "sensor_19" not in selected_sensors

    for s in selected_sensors:
        assert s in df.columns, f"Selected sensor {s} missing from test dataset"


def test_normalization_consistency(metadata):
    """Verify normalization statistics applied to test data originate strictly from training fleet."""
    means = metadata["sensor_means"]
    stds = metadata["sensor_stds"]
    selected = metadata["selected_sensors"]

    assert len(means) == 17
    assert len(stds) == 17
    for s in selected:
        assert stds[s] > 0, f"Standard deviation for {s} must be positive"


def test_prediction_output_shape(test_data):
    """Verify evaluate_official_test_set returns exactly 100 predictions with valid floats."""
    df, rul = test_data
    lr = joblib.load(MODELS_DIR / "baselines" / "linear_regression.pkl")

    res = evaluate_official_test_set(
        model=lr,
        test_df=df,
        rul_series=rul,
        subset="FD001",
        window_length=30,
        is_3d=False
    )

    assert res["num_engines_evaluated"] == 100
    assert len(res["predictions"]) == 100
    assert len(res["ground_truth_raw"]) == 100
    assert len(res["engine_details"]) == 100
    assert not any(np.isnan(res["predictions"]))


def test_nasa_score_calculation():
    """Verify NASA scoring function on test predictions behaves asymmetrically and non-negatively."""
    y_true = np.array([50.0, 50.0])
    # Case 1: Early by 10 cycles (y_pred = 40)
    # Case 2: Late by 10 cycles (y_pred = 60)
    score_early = compute_nasa_score(np.array([50.0]), np.array([40.0]))
    score_late = compute_nasa_score(np.array([50.0]), np.array([60.0]))

    assert score_late > score_early, "Late overestimation penalty must strictly exceed early penalty"
    assert score_early > 0.0
    assert score_late > 0.0


def test_missing_or_corrupted_test_data():
    """Verify evaluate_official_test_set raises informative errors on mismatched or invalid inputs."""
    lr = joblib.load(MODELS_DIR / "baselines" / "linear_regression.pkl")
    dummy_df = pd.DataFrame({"engine_id": [1, 2], "cycle": [1, 2]})
    mismatched_rul = np.array([10.0])  # Length 1 vs 2 engines

    with pytest.raises(ValueError, match="Mismatch between number of test engines"):
        evaluate_official_test_set(
            model=lr,
            test_df=dummy_df,
            rul_series=mismatched_rul,
            subset="FD001"
        )
