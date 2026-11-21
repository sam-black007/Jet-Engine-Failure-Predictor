"""Advanced Prediction Engine and Uncertainty Estimation (JETWISE AI Phase 2).

Provides comprehensive inference strategies:
1. Standard Point Prediction: Deterministic single-pass inference.
2. Conservative Prediction: Safety-oriented negative bias shift to minimize catastrophic late overestimation.
3. Monte Carlo Dropout (Gal & Ghahramani, 2016): Stochastic dropout forward passes estimating epistemic uncertainty.
4. Deep Ensemble Prediction: Aggregation across multiple diverse model checkpoints.
5. Uncertainty Calibration & Empirical Coverage Assessment on held-out validation sets.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from src.config_manager import MasterConfig, PredictionConfig


class PredictionEngine:
    """Multi-strategy prediction engine with rigorous uncertainty quantification."""

    def __init__(
        self,
        model: Any,
        config: Optional[Union[MasterConfig, PredictionConfig]] = None
    ):
        self.model = model
        if isinstance(config, MasterConfig):
            self.pred_cfg = config.prediction
        elif isinstance(config, PredictionConfig):
            self.pred_cfg = config
        else:
            self.pred_cfg = PredictionConfig()

    def _enable_dropout(self, model: nn.Module) -> None:
        """Force dropout modules into train mode while freezing Batch/LayerNorm."""
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def predict_point(self, X: np.ndarray) -> np.ndarray:
        """Standard deterministic point prediction.
        
        Parameters
        ----------
        X : np.ndarray of shape (N, T, D) or (N, D)
        
        Returns
        -------
        preds : np.ndarray of shape (N,)
        """
        if hasattr(self.model, "predict"):
            preds = self.model.predict(X)
        elif isinstance(self.model, nn.Module):
            self.model.eval()
            with torch.no_grad():
                device = next(self.model.parameters()).device
                t_in = torch.tensor(X, dtype=torch.float32, device=device)
                preds = self.model(t_in).cpu().numpy().flatten()
        else:
            raise TypeError("Model must be an nn.Module or expose a .predict() method.")

        if self.pred_cfg.clip_output_to_zero:
            preds = np.maximum(0.0, preds)
        return np.asarray(preds, dtype=np.float32)

    def predict_conservative(self, X: np.ndarray, shift_cycles: Optional[float] = None) -> np.ndarray:
        """Conservative prediction applying safety offset to prevent late maintenance.
        
        Parameters
        ----------
        X : np.ndarray
        shift_cycles : float, optional
            Cycles subtracted from point estimate. Defaults to config bias_offset (or 3.0).
        """
        point_preds = self.predict_point(X)
        shift = shift_cycles if shift_cycles is not None else max(0.0, self.pred_cfg.bias_offset or 3.0)
        conservative = point_preds - shift
        if self.pred_cfg.clip_output_to_zero:
            conservative = np.maximum(0.0, conservative)
        return np.asarray(conservative, dtype=np.float32)

    def predict_mc_dropout(
        self,
        X: np.ndarray,
        n_samples: Optional[int] = None,
        confidence: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """Perform Monte Carlo Dropout uncertainty estimation.
        
        Parameters
        ----------
        X : np.ndarray of shape (N, T, D)
        n_samples : int, optional
            Number of stochastic forward passes (default: 30).
        confidence : float, optional
            Nominal confidence interval coverage level (default: 0.90).
            
        Returns
        -------
        dict containing:
            - 'mean': Expected RUL point estimate
            - 'std': Epistemic uncertainty standard deviation
            - 'lower': Lower confidence bound
            - 'upper': Upper confidence bound
            - 'interval_width': upper - lower
            - 'samples': Raw sample matrix of shape (n_samples, N)
        """
        if not isinstance(self.model, nn.Module):
            # Fallback for Scikit-Learn baselines
            point = self.predict_point(X)
            return {
                "mean": point, "std": np.zeros_like(point),
                "lower": point, "upper": point,
                "interval_width": np.zeros_like(point),
                "samples": point[np.newaxis, :]
            }

        k = n_samples or self.pred_cfg.mc_samples
        alpha = confidence or self.pred_cfg.confidence_level
        z_crit = float(stats.norm.ppf((1.0 + alpha) / 2.0))

        self.model.eval()
        self._enable_dropout(self.model)

        device = next(self.model.parameters()).device
        t_in = torch.tensor(X, dtype=torch.float32, device=device)

        sample_list = []
        with torch.no_grad():
            for _ in range(k):
                out = self.model(t_in).cpu().numpy().flatten()
                if self.pred_cfg.clip_output_to_zero:
                    out = np.maximum(0.0, out)
                sample_list.append(out)

        samples = np.array(sample_list)  # (k, N)
        mean_preds = np.mean(samples, axis=0)
        std_preds = np.std(samples, axis=0)
        lower_bound = np.maximum(0.0, mean_preds - z_crit * std_preds)
        upper_bound = mean_preds + z_crit * std_preds

        return {
            "mean": mean_preds.astype(np.float32),
            "std": std_preds.astype(np.float32),
            "lower": lower_bound.astype(np.float32),
            "upper": upper_bound.astype(np.float32),
            "interval_width": (upper_bound - lower_bound).astype(np.float32),
            "samples": samples
        }

    def predict_ensemble(
        self,
        X: np.ndarray,
        models: List[Any],
        weights: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """Deep Ensemble prediction combining multiple model checkpoints.
        
        Parameters
        ----------
        X : np.ndarray
        models : List[Any]
            List of trained models.
        weights : List[float], optional
            Linear combination weights.
        """
        if not models:
            raise ValueError("Ensemble models list must contain at least one model.")

        if weights is None:
            w = np.ones(len(models)) / len(models)
        else:
            w = np.array(weights) / sum(weights)

        preds_matrix = []
        for m in models:
            sub_engine = PredictionEngine(m, self.pred_cfg)
            p = sub_engine.predict_point(X)
            preds_matrix.append(p)

        preds_matrix = np.array(preds_matrix)  # (M, N)
        ensemble_mean = np.sum(w[:, np.newaxis] * preds_matrix, axis=0)
        ensemble_std = np.std(preds_matrix, axis=0)

        return {
            "mean": ensemble_mean.astype(np.float32),
            "std": ensemble_std.astype(np.float32),
            "lower": np.maximum(0.0, np.min(preds_matrix, axis=0)),
            "upper": np.max(preds_matrix, axis=0),
            "individual_predictions": preds_matrix
        }

    def evaluate_uncertainty_calibration(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_samples: int = 30,
        confidence: float = 0.90
    ) -> Dict[str, float]:
        """Compute empirical coverage and calibration metrics on held-out validation data.
        
        Returns:
            dict containing:
                - 'nominal_confidence': Desired coverage (e.g. 0.90)
                - 'empirical_coverage': Measured percentage of y_val in [lower, upper]
                - 'mean_interval_width': Average width of uncertainty intervals
                - 'coverage_error': empirical_coverage - nominal_confidence
        """
        unc_res = self.predict_mc_dropout(X_val, n_samples=n_samples, confidence=confidence)
        lower = unc_res["lower"]
        upper = unc_res["upper"]

        inside = (y_val >= lower) & (y_val <= upper)
        empirical_cov = float(np.mean(inside))
        mean_width = float(np.mean(upper - lower))

        return {
            "nominal_confidence": round(float(confidence), 3),
            "empirical_coverage": round(empirical_cov, 4),
            "mean_interval_width": round(mean_width, 2),
            "coverage_error": round(empirical_cov - confidence, 4)
        }
