"""Unit and gradient flow tests for Model Factory and Phase 2 Architectures."""

import pytest
import numpy as np
import torch
import torch.nn as nn

from src.config_manager import ModelConfig
from src.model_factory import create_model, is_deep_learning_model, count_parameters
from src.models.attention_lstm import AttentionLSTMPredictor, BahdanauAttention
from src.models.tcn import TemporalConvNetPredictor, TemporalBlock, Chomp1d
from src.models.transformer import TransformerRULPredictor, PositionalEncoding


def test_attention_lstm_gradient_flow():
    """Verify forward and backward gradient propagation in Attention-LSTM."""
    model = AttentionLSTMPredictor(
        input_dim=17, hidden_dim1=32, hidden_dim2=16, attention_dim=16, dropout_rate=0.1
    )
    x = torch.randn(4, 30, 17, requires_grad=True)
    out, weights = model(x, return_attention=True)
    assert out.shape == (4,)
    assert weights.shape == (4, 30, 1)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (4, 30, 17)


def test_tcn_gradient_flow_and_receptive_field():
    """Verify forward and backward gradient propagation in TCN."""
    model = TemporalConvNetPredictor(
        input_dim=17, num_channels=[16, 32], kernel_size=3, dropout=0.1
    )
    x = torch.randn(4, 30, 17, requires_grad=True)
    out = model(x)
    assert out.shape == (4,)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (4, 30, 17)


def test_transformer_gradient_flow():
    """Verify forward and backward gradient propagation in Transformer."""
    model = TransformerRULPredictor(
        input_dim=17, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1
    )
    x = torch.randn(4, 30, 17, requires_grad=True)
    out = model(x)
    assert out.shape == (4,)

    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (4, 30, 17)


def test_model_factory_deep_and_baselines():
    """Verify ModelFactory instantiates both PyTorch and Scikit-Learn models correctly."""
    # Deep Models
    for arch in ["lstm", "attention_lstm", "tcn", "transformer"]:
        cfg = ModelConfig(
            architecture=arch,
            input_dim=17,
            transformer_d_model=32,
            transformer_nhead=4
        )
        model = create_model(cfg, input_dim=17, seq_len=30)
        assert is_deep_learning_model(model)
        assert count_parameters(model) > 0

    # Baseline Models
    for arch in ["linear_regression", "random_forest", "gradient_boosting"]:
        cfg = ModelConfig(architecture=arch)
        model = create_model(cfg, input_dim=17, seq_len=30)
        assert not is_deep_learning_model(model)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
