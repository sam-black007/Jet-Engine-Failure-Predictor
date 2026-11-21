"""Unified Model Factory for Jet Engine Failure Predictor (JETWISE AI Phase 2).

Provides a single factory interface to instantiate, configure, and inspect any
supported prognostic model (Deep Learning or Classical ML) via ModelConfig.
"""

from typing import Any, Dict, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.config_manager import ModelConfig
from src.train_lstm import LSTMPredictor
from src.models.attention_lstm import AttentionLSTMPredictor
from src.models.tcn import TemporalConvNetPredictor
from src.models.transformer import TransformerRULPredictor


def create_model(
    config: Optional[ModelConfig] = None,
    input_dim: int = 14,
    seq_len: int = 30
) -> Any:
    """Instantiate a prognostic model based on ModelConfig.
    
    Parameters
    ----------
    config : ModelConfig, optional
        Architecture specification and hyperparameters. Defaults to default ModelConfig.
    input_dim : int, default 14
        Number of active input sensor features.
    seq_len : int, default 30
        Observation window length.
        
    Returns
    -------
    nn.Module or sklearn BaseEstimator
    """
    cfg = config or ModelConfig()
    arch = cfg.architecture.lower()

    if arch in ("lstm", "asymmetric_lstm", "pinball_lstm"):
        return LSTMPredictor(
            input_dim=input_dim,
            hidden_dim1=cfg.lstm_hidden_dim1,
            hidden_dim2=cfg.lstm_hidden_dim2,
            dropout_rate=cfg.dropout
        )

    elif arch == "attention_lstm":
        return AttentionLSTMPredictor(
            input_dim=input_dim,
            hidden_dim1=cfg.lstm_hidden_dim1,
            hidden_dim2=cfg.lstm_hidden_dim2,
            attention_dim=cfg.attention_dim,
            dropout_rate=cfg.dropout,
            bidirectional=cfg.bidirectional
        )

    elif arch == "tcn":
        return TemporalConvNetPredictor(
            input_dim=input_dim,
            num_channels=cfg.tcn_num_channels,
            kernel_size=cfg.tcn_kernel_size,
            dropout=cfg.tcn_dropout
        )

    elif arch == "transformer":
        return TransformerRULPredictor(
            input_dim=input_dim,
            d_model=cfg.transformer_d_model,
            nhead=cfg.transformer_nhead,
            num_layers=cfg.transformer_num_layers,
            dim_feedforward=cfg.transformer_dim_feedforward,
            dropout=cfg.transformer_dropout,
            max_seq_len=max(seq_len * 2, 100)
        )

    elif arch == "linear_regression":
        return LinearRegression()

    elif arch == "random_forest":
        return RandomForestRegressor(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            random_state=42
        )

    elif arch == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth or 5,
            random_state=42
        )

    else:
        raise ValueError(f"Unknown model architecture: '{arch}'")


def is_deep_learning_model(model: Any) -> bool:
    """Return True if model is a PyTorch neural network module."""
    return isinstance(model, nn.Module)


def count_parameters(model: Any) -> int:
    """Return total number of trainable parameters in a PyTorch model."""
    if isinstance(model, nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 0
