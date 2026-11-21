"""Time-Series Transformer Encoder for Turbofan Engine RUL Prediction.

Applies multi-head self-attention across temporal sensor trajectories:
- Sinusoidal or learnable positional encoding to capture degradation progression.
- Multi-head self-attention with configurable attention heads and feedforward depth.
- Pre-LayerNorm / residual design for training stability.
- Mean-pooling sequence aggregation followed by regression head.
"""

import math
from typing import Optional, Union
import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time-series sequences."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerRULPredictor(nn.Module):
    """Transformer Encoder Architecture for Turbofan RUL Estimation."""

    def __init__(
        self,
        input_dim: int = 14,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")

        self.input_dim = input_dim
        self.d_model = d_model

        # Project input sensor features into d_model dimension
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_seq_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, seq_len, input_dim)
        
        Returns
        -------
        torch.Tensor of shape (batch_size,)
        """
        # (B, T, D_in) -> (B, T, D_model)
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        # Pass through Transformer encoder
        encoded = self.transformer_encoder(h)  # (B, T, D_model)
        # Pool across sequence dimension (mean pooling captures global degraded state)
        pooled = torch.mean(encoded, dim=1)  # (B, D_model)
        out = self.head(pooled).squeeze(-1)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Inference helper accepting 2D or 3D numpy arrays, returning 1D numpy predictions."""
        self.eval()
        if X.ndim == 2:
            X = X.reshape(X.shape[0], -1, self.input_dim)
        elif X.ndim != 3:
            raise ValueError(f"Expected 2D or 3D input, got shape {X.shape}")

        with torch.no_grad():
            device = next(self.parameters()).device
            t_in = torch.tensor(X, dtype=torch.float32, device=device)
            preds = self.forward(t_in)
            return preds.cpu().numpy().flatten()
