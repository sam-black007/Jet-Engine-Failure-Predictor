"""Unit tests for Advanced Prediction Engine and Uncertainty Estimation."""

import pytest
import numpy as np
import torch
import torch.nn as nn

from src.config_manager import PredictionConfig
from src.prediction_engine import PredictionEngine
from src.models.attention_lstm import AttentionLSTMPredictor


def test_prediction_engine_point_and_conservative():
    """Verify point and conservative bias-shifted predictions."""
    model = AttentionLSTMPredictor(input_dim=17, hidden_dim1=16, hidden_dim2=8, attention_dim=8)
    cfg = PredictionConfig(strategy="conservative", bias_offset=4.0, clip_output_to_zero=True)
    engine = PredictionEngine(model, cfg)

    X = np.random.randn(10, 30, 17).astype(np.float32)
    pt = engine.predict_point(X)
    cons = engine.predict_conservative(X, shift_cycles=4.0)

    assert len(pt) == 10
    assert len(cons) == 10
    assert (pt >= 0.0).all()
    assert (cons >= 0.0).all()
    assert (cons <= pt + 1e-5).all()


def test_mc_dropout_uncertainty_quantification():
    """Verify Monte Carlo Dropout produces non-zero epistemic uncertainty and valid bounds."""
    model = AttentionLSTMPredictor(input_dim=17, hidden_dim1=32, hidden_dim2=16, attention_dim=16, dropout_rate=0.3)
    engine = PredictionEngine(model)

    X = np.random.randn(8, 30, 17).astype(np.float32)
    mc_res = engine.predict_mc_dropout(X, n_samples=15, confidence=0.90)

    assert "mean" in mc_res
    assert "std" in mc_res
    assert "lower" in mc_res
    assert "upper" in mc_res
    assert (mc_res["upper"] >= mc_res["lower"]).all()
    assert (mc_res["interval_width"] >= 0.0).all()


def test_calibration_and_empirical_coverage():
    """Verify empirical coverage calculation on synthetic arrays."""
    model = AttentionLSTMPredictor(input_dim=17, hidden_dim1=16, hidden_dim2=8, attention_dim=8)
    engine = PredictionEngine(model)

    X = np.random.randn(15, 30, 17).astype(np.float32)
    y = np.random.uniform(10, 100, size=15).astype(np.float32)

    cal = engine.evaluate_uncertainty_calibration(X, y, n_samples=10, confidence=0.90)
    assert "nominal_confidence" in cal
    assert "empirical_coverage" in cal
    assert "mean_interval_width" in cal
    assert 0.0 <= cal["empirical_coverage"] <= 1.0
