"""Temporal Convolutional Network (TCN) for Turbofan RUL Prediction.

Based on Bai, Kolter, and Koltun (2018):
- Dilated causal 1D convolutions preventing future information leakage.
- Exponentially expanding receptive field (dilation = 1, 2, 4, 8, ...).
- Weight normalization and residual connections.
- Flexible channel depth and kernel dimensions.
"""

from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn as nn

try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """Causal padding truncation: removes future timesteps from right padding."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Residual causal convolutional block with dilation and weight normalization."""

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2
    ):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNetPredictor(nn.Module):
    """Full Temporal Convolutional Network for RUL estimation."""

    def __init__(
        self,
        input_dim: int = 14,
        num_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        num_channels = num_channels or [32, 64, 64]
        self.num_channels = num_channels
        self.kernel_size = kernel_size

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(num_channels[-1], 32),
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
        # PyTorch Conv1d expects (batch_size, channels, seq_len)
        x_trans = x.transpose(1, 2)
        y = self.network(x_trans)
        # Take the features of the last time step (causal output)
        last_step_feat = y[:, :, -1]
        out = self.head(last_step_feat).squeeze(-1)
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
