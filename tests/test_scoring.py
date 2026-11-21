"""Unit tests for NASA C-MAPSS asymmetric scoring function and regression metrics."""

import math
import numpy as np
import pytest
from src.evaluation import compute_nasa_score, compute_metrics


def test_zero_error():
    """Zero prediction error must produce a score of exactly 0.0."""
    y_true = np.array([50.0, 100.0, 125.0])
    y_pred = np.array([50.0, 100.0, 125.0])
    score = compute_nasa_score(y_true, y_pred)
    assert score == pytest.approx(0.0, abs=1e-7)
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["NASA_score"] == 0.0


def test_asymmetric_property():
    """A late prediction (overestimation) must receive a strictly harsher penalty than an early prediction (underestimation) of the same magnitude."""
    error = 10.0
    
    # Early prediction: y_pred = 40, y_true = 50 -> error d = -10
    early_true = np.array([50.0])
    early_pred = np.array([40.0])
    early_score = compute_nasa_score(early_true, early_pred)
    expected_early = math.exp(10.0 / 13.0) - 1.0
    assert early_score == pytest.approx(expected_early, rel=1e-5)
    
    # Late prediction: y_pred = 60, y_true = 50 -> error d = +10
    late_true = np.array([50.0])
    late_pred = np.array([60.0])
    late_score = compute_nasa_score(late_true, late_pred)
    expected_late = math.exp(10.0 / 10.0) - 1.0
    assert late_score == pytest.approx(expected_late, rel=1e-5)
    
    # Verify asymmetry: late score > early score
    assert late_score > early_score
    # Specifically: e^1 - 1 (~1.718) > e^(10/13) - 1 (~1.158)
    assert late_score > 1.7
    assert early_score < 1.2


def test_strictly_non_negative():
    """Score must be non-negative for arbitrary positive, negative, and mixed errors."""
    rng = np.random.RandomState(42)
    for _ in range(10):
        y_true = rng.uniform(10, 125, size=20)
        y_pred = rng.uniform(0, 140, size=20)
        score = compute_nasa_score(y_true, y_pred)
        assert score >= 0.0


def test_vectorized_sum():
    """Vectorized scoring across multiple samples must equal sum of individual sample penalties."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([5.0, 20.0, 35.0])
    
    s1 = math.exp(5.0 / 13.0) - 1.0    # d = -5
    s2 = 0.0                           # d = 0
    s3 = math.exp(5.0 / 10.0) - 1.0    # d = +5
    expected_total = s1 + s2 + s3
    
    actual = compute_nasa_score(y_true, y_pred)
    assert actual == pytest.approx(expected_total, rel=1e-5)
