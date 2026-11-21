"""Asymmetric and prognostic loss functions for Jet Engine Failure Predictor.

Implements:
1. NASAAsymmetricLoss: Numerically stable Log-Mean-Exp formulation directly aligned
   with the official NASA C-MAPSS asymmetric scoring function.
2. AsymmetricPinballLoss: Configurable quantile / pinball loss penalizing overestimation
   (d >= 0) more heavily than underestimation (d < 0).
3. SmoothPinballLoss: Smooth Huberized asymmetric pinball loss for continuous optimization.
"""

from typing import Optional, Union, Any, Dict
import torch
import torch.nn as nn


class NASAAsymmetricLoss(nn.Module):
    """Numerically stable NASA C-MAPSS Asymmetric Loss using Log-Mean-Exp formulation.

    Mathematical Motivation:
    The official NASA C-MAPSS scoring function is defined as:
        s_i = exp(d_i / 10) - 1   if d_i >= 0 (late prediction / overestimation)
        s_i = exp(-d_i / 13) - 1  if d_i < 0  (early prediction / underestimation)
    where d_i = y_pred_i - y_true_i.

    Direct optimization of sum(exp(z_i) - 1) is prone to catastrophic numerical overflow
    (float32 overflows when z > 88, i.e., d > 880 cycles, and gradients explode exponentially
    early in training).

    To achieve stable, gradient-friendly optimization, we implement the Log-Mean-Exp loss:
        z_i = where(d_i >= 0, d_i / 10, -d_i / 13)
        Loss = logsumexp(z) - log(batch_size) = log( (1 / batch_size) * sum(exp(z_i)) )

    Key Properties:
    1. Monotonic Alignment: Minimizing log( (1/B) * sum(exp(z_i)) ) is strictly monotonically
       equivalent to minimizing the exponential NASA penalty sum(exp(z_i) - 1).
    2. Numerical Stability: Uses PyTorch's logsumexp (subtracting max(z_i)), preventing any
       single-precision floating point overflow.
    3. Bounded Gradients: The gradient with respect to prediction y_pred_i is given by:
           grad_i = softmax(z)_i * (1/10 if d_i >= 0 else -1/13)
       Since softmax(z)_i in (0, 1), the maximum gradient magnitude is bounded by 0.10,
       completely eliminating exploding gradients while focusing updates on the worst outliers.
    4. Non-Negativity: Loss >= 0, with Loss == 0 if and only if d_i == 0 for all i in batch.
    """

    def __init__(self, late_penalty: float = 10.0, early_penalty: float = 13.0):
        super().__init__()
        if late_penalty <= 0 or early_penalty <= 0:
            raise ValueError("Penalty scales must be positive numbers.")
        self.late_penalty = float(late_penalty)
        self.early_penalty = float(early_penalty)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute the Log-Mean-Exp NASA Asymmetric Loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted Remaining Useful Life (RUL) of any shape (B, ...).
        y_true : torch.Tensor
            Ground-truth Remaining Useful Life (RUL) matching y_pred.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        pred_flat = y_pred.reshape(-1)
        true_flat = y_true.reshape(-1)
        
        d = pred_flat - true_flat
        z = torch.where(d >= 0, d / self.late_penalty, -d / self.early_penalty)
        
        batch_size = torch.tensor(z.numel(), dtype=z.dtype, device=z.device)
        loss = torch.logsumexp(z, dim=0) - torch.log(batch_size)
        return loss


class AsymmetricPinballLoss(nn.Module):
    """Configurable Asymmetric Pinball (Quantile) Loss.

    Mathematical Formulation:
    For residual d = y_pred - y_true:
        loss_i = (1 - q) * d_i   if d_i >= 0 (overestimation)
        loss_i = -q * d_i       if d_i < 0  (underestimation)
    where q in (0, 1) is the target quantile.

    Industrial Rationale:
    When q = 0.40:
        Overestimation penalty scale: 1 - 0.40 = 0.60
        Underestimation penalty scale: 0.40
        Ratio: 0.60 / 0.40 = 1.50
    This penalizes late overestimation 1.5x more severely than early underestimation,
    closely mirroring NASA's asymmetry while retaining a linear, stable gradient.

    Parameters
    ----------
    quantile : float, default 0.40
        Target quantile in (0, 1). Values < 0.50 penalize overestimation more heavily.
    reduction : str, default 'mean'
        One of 'mean', 'sum', or 'none'.
    """

    def __init__(self, quantile: float = 0.40, reduction: str = "mean"):
        super().__init__()
        if not (0.0 < quantile < 1.0):
            raise ValueError(f"Quantile must be in (0, 1), got {quantile}")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Invalid reduction '{reduction}'. Expected 'mean', 'sum', or 'none'.")
        self.quantile = float(quantile)
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute Asymmetric Pinball Loss.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted RUL tensor.
        y_true : torch.Tensor
            Ground-truth RUL tensor.

        Returns
        -------
        torch.Tensor
            Loss tensor reduced according to self.reduction.
        """
        pred_flat = y_pred.reshape(-1)
        true_flat = y_true.reshape(-1)
        
        d = pred_flat - true_flat
        loss = torch.where(d >= 0, (1.0 - self.quantile) * d, -self.quantile * d)
        
        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        return loss


class SmoothPinballLoss(nn.Module):
    """Smooth (Huberized) Asymmetric Pinball Loss.

    Provides a continuously differentiable loss around d = 0 with threshold delta.
    """

    def __init__(self, quantile: float = 0.40, delta: float = 1.0, reduction: str = "mean"):
        super().__init__()
        if not (0.0 < quantile < 1.0):
            raise ValueError("Quantile must be in (0, 1).")
        self.quantile = float(quantile)
        self.delta = float(delta)
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        pred_flat = y_pred.reshape(-1)
        true_flat = y_true.reshape(-1)
        d = pred_flat - true_flat
        abs_d = torch.abs(d)
        
        # Huber smooth transition
        smooth = torch.where(abs_d <= self.delta, 0.5 * (d ** 2) / self.delta, abs_d - 0.5 * self.delta)
        weight = torch.where(d >= 0, 1.0 - self.quantile, self.quantile)
        loss = weight * smooth
        
        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        return loss


def get_loss_function(config: Any) -> nn.Module:
    """Instantiate a PyTorch loss module from a LossConfig, dict, or string name.
    
    Parameters
    ----------
    config : LossConfig or dict or str
        Loss configuration.
        
    Returns
    -------
    nn.Module
    """
    if isinstance(config, str):
        name = config.lower()
        cfg_dict = {"name": name}
    elif hasattr(config, "name"):
        name = config.name.lower()
        cfg_dict = config.__dict__ if hasattr(config, "__dict__") else {}
    elif isinstance(config, dict):
        name = config.get("name", "mse").lower()
        cfg_dict = config
    else:
        name = "mse"
        cfg_dict = {}

    if name == "mse":
        return nn.MSELoss()
    elif name == "mae":
        return nn.L1Loss()
    elif name == "huber":
        delta = getattr(config, "huber_delta", cfg_dict.get("huber_delta", 1.0))
        return nn.HuberLoss(delta=float(delta))
    elif name == "nasa_asymmetric":
        late_p = getattr(config, "late_penalty", cfg_dict.get("late_penalty", 10.0))
        early_p = getattr(config, "early_penalty", cfg_dict.get("early_penalty", 13.0))
        return NASAAsymmetricLoss(late_penalty=float(late_p), early_penalty=float(early_p))
    elif name == "pinball":
        q = getattr(config, "pinball_quantile", cfg_dict.get("pinball_quantile", 0.35))
        return AsymmetricPinballLoss(quantile=float(q))
    elif name == "smooth_pinball":
        q = getattr(config, "pinball_quantile", cfg_dict.get("pinball_quantile", 0.35))
        beta = getattr(config, "smooth_beta", cfg_dict.get("smooth_beta", 0.1))
        return SmoothPinballLoss(quantile=float(q), delta=float(beta))
    else:
        raise ValueError(f"Unknown loss function name: '{name}'")


