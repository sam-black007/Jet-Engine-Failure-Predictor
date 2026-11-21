"""Attention-enhanced Deep LSTM for Turbofan Engine Remaining Useful Life Prediction.

Integrates a sequence self-attention / Bahdanau attention mechanism over the hidden states
of a multi-layer LSTM. Allows the network to learn which temporal degradation stages
(e.g., healthy baseline vs accelerated wear phase) carry the highest prognostic importance.
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """Additive Bahdanau Attention mechanism over sequence hidden states."""

    def __init__(self, hidden_dim: int, attention_dim: int = 32):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attention_dim, bias=True)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention weights and context vector.
        
        Parameters
        ----------
        hidden_states : torch.Tensor of shape (batch_size, seq_len, hidden_dim)
        
        Returns
        -------
        context : torch.Tensor of shape (batch_size, hidden_dim)
        weights : torch.Tensor of shape (batch_size, seq_len, 1)
        """
        # Score: (B, T, attention_dim) -> (B, T, 1)
        score = self.v(torch.tanh(self.W(hidden_states)))
        weights = F.softmax(score, dim=1)
        context = torch.sum(weights * hidden_states, dim=1)
        return context, weights


class AttentionLSTMPredictor(nn.Module):
    """Attention-augmented LSTM network for turbofan RUL prognosis."""

    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim1: int = 64,
        hidden_dim2: int = 32,
        attention_dim: int = 32,
        dropout_rate: float = 0.2,
        bidirectional: bool = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim1,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.dropout1 = nn.Dropout(dropout_rate)

        self.lstm2 = nn.LSTM(
            input_size=hidden_dim1 * num_directions,
            hidden_size=hidden_dim2,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.dropout2 = nn.Dropout(dropout_rate)

        lstm2_out_dim = hidden_dim2 * num_directions
        self.attention = BahdanauAttention(hidden_dim=lstm2_out_dim, attention_dim=attention_dim)

        self.head = nn.Sequential(
            nn.Linear(lstm2_out_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, seq_len, input_dim)
        return_attention : bool, default False
            If True, returns (pred, attention_weights).
        """
        out1, _ = self.lstm1(x)
        out1 = self.dropout1(out1)

        out2, _ = self.lstm2(out1)
        out2 = self.dropout2(out2)

        context, weights = self.attention(out2)
        pred = self.head(context).squeeze(-1)

        if return_attention:
            return pred, weights
        return pred

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
