"""Evaluation metrics and scoring functions for Jet Engine Failure Predictor.

Implements standard regression metrics (MAE, RMSE) and the official NASA C-MAPSS
asymmetric scoring function used in the PHM'08 prognostic challenge.

Mathematical formulation of NASA Asymmetric Scoring Function:
Let d_i = y_pred_i - y_true_i be the prediction error for sample i.
    s_i = exp(-d_i / 13) - 1   if d_i < 0  (early prediction: penalty factor 13)
    s_i = exp( d_i / 10) - 1   if d_i >= 0 (late prediction: penalty factor 10)
Total Score S = sum(s_i) for i = 1 to N.

Key characteristics:
1. Penalizes late predictions (d_i > 0, overestimating RUL) much more heavily than
   early predictions (d_i < 0, underestimating RUL), reflecting industrial maintenance
   priorities where unexpected in-flight failure is far more catastrophic than
   premature engine servicing.
2. The score is strictly non-negative: S >= 0, with S = 0 for perfect predictions.
"""

from typing import Dict, Union, List, Any
import numpy as np
import pandas as pd


def compute_nasa_score(y_true: Union[np.ndarray, List[float]], 
                       y_pred: Union[np.ndarray, List[float]]) -> float:
    """Compute the official NASA C-MAPSS asymmetric score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth Remaining Useful Life (RUL).
    y_pred : array-like of shape (n_samples,)
        Predicted Remaining Useful Life (RUL).

    Returns
    -------
    float
        Total asymmetric penalty score S >= 0.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    # d = y_pred - y_true
    d = y_pred - y_true
    
    # Early prediction penalty (d < 0): exp(-d / 13) - 1
    # Late prediction penalty (d >= 0): exp(d / 10) - 1
    score = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    
    return float(np.sum(score))


def compute_metrics(y_true: Union[np.ndarray, List[float]], 
                    y_pred: Union[np.ndarray, List[float]]) -> Dict[str, float]:
    """Compute MAE, RMSE, and NASA asymmetric score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth RUL values.
    y_pred : array-like of shape (n_samples,)
        Predicted RUL values.

    Returns
    -------
    dict
        Dictionary with keys 'MAE', 'RMSE', 'NASA_score'.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    nasa = compute_nasa_score(y_true, y_pred)
    
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "NASA_score": round(nasa, 4)
    }


def evaluate_per_engine(
    model: Any,
    raw_df: pd.DataFrame,
    val_engine_ids: List[int],
    sensor_cols: List[str],
    sensor_means: np.ndarray,
    sensor_stds: np.ndarray,
    window_length: int = 30,
    rul_clip: int = 125,
    is_3d: bool = False
) -> Dict[str, Any]:
    """Evaluate model performance on a per-engine basis using truncated trajectories.

    For each engine, we take the final available window of sensor readings
    up to the end of the observed trajectory, normalize using training stats,
    predict the RUL, and compare against the engine's true remaining life.

    Parameters
    ----------
    model : fitted estimator or PyTorch/Keras model
        The predictive maintenance model.
    raw_df : pd.DataFrame
        Complete dataset containing engine trajectories.
    val_engine_ids : list of int
        Engine IDs belonging to the validation split.
    sensor_cols : list of str
        Names of sensor features used by the model.
    sensor_means : np.ndarray
        Training feature means for normalization.
    sensor_stds : np.ndarray
        Training feature standard deviations for normalization.
    window_length : int, default 30
        Sequence window size.
    rul_clip : int, default 125
        Upper clipping bound for target RUL.
    is_3d : bool, default False
        Whether the model expects 3D inputs (batch, time, features) or 2D (batch, time*features).

    Returns
    -------
    dict
        Dictionary containing overall metrics and per-engine details.
    """
    true_ruls = []
    pred_ruls = []
    engine_details = []

    for eid in val_engine_ids:
        eng_df = raw_df[raw_df["engine_id"] == eid].sort_values("cycle").reset_index(drop=True)
        if len(eng_df) < window_length:
            continue
        
        # Take the final window ending at the trajectory's last cycle
        last_window = eng_df[sensor_cols].iloc[-window_length:].values.astype(np.float32)
        norm_window = (last_window - sensor_means) / sensor_stds
        
        # Target RUL at final cycle of run-to-failure engine is 0 (or actual ground truth)
        max_cycle = eng_df["cycle"].max()
        last_cycle = eng_df["cycle"].iloc[-1]
        actual_rul = min(max_cycle - last_cycle, rul_clip)
        
        # Model prediction
        if is_3d:
            # 3D: (1, window, features)
            if hasattr(model, "predict"):
                pred = float(model.predict(norm_window[np.newaxis, ...]).squeeze())
            else:
                # PyTorch
                import torch
                model.eval()
                with torch.no_grad():
                    t_in = torch.tensor(norm_window[np.newaxis, ...], dtype=torch.float32)
                    pred = float(model(t_in).item())
        else:
            # 2D flattened: (1, window * features)
            flat = norm_window.reshape(1, -1)
            pred = float(model.predict(flat)[0])
            
        true_ruls.append(actual_rul)
        pred_ruls.append(pred)
        engine_details.append({
            "engine_id": int(eid),
            "actual_rul": float(actual_rul),
            "predicted_rul": float(pred),
            "error": float(pred - actual_rul)
        })

    metrics = compute_metrics(true_ruls, pred_ruls)
    return {
        "metrics": metrics,
        "engine_details": engine_details,
        "num_engines_evaluated": len(true_ruls)
    }
