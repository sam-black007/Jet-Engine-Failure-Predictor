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

from typing import Dict, Union, List, Any, Optional
import os
import json
import pathlib
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


def evaluate_official_test_set(
    model: Any,
    test_df: Optional[pd.DataFrame] = None,
    rul_series: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
    subset: str = "FD001",
    window_length: int = 30,
    rul_clip: int = 125,
    is_3d: bool = False,
    metadata_path: Optional[Union[str, pathlib.Path]] = None
) -> Dict[str, Any]:
    """Evaluate model performance on the official NASA C-MAPSS test set.

    In the official C-MAPSS test set (test_FD001.txt), each engine trajectory is
    truncated at a random point in time prior to failure. The true remaining
    cycles after the final observed cycle are provided in RUL_FD001.txt.

    This function extracts the final `window_length` (30) cycles for each test engine,
    normalizes using training set statistics from `preprocess_metadata.json`,
    runs model prediction, and evaluates both:
    1. Raw Target Metrics: Directly against official NASA ground-truth RUL values.
    2. Clipped Target Metrics: Against RUL clipped to `rul_clip` cycles (125).

    Parameters
    ----------
    model : fitted estimator or PyTorch model
        The predictive maintenance model.
    test_df : pd.DataFrame, optional
        Loaded test dataset. If None, loaded via data_loader.load_dataset(subset, kind="test").
    rul_series : pd.Series or np.ndarray or list, optional
        Official RUL labels. If None, loaded via data_loader.load_rul(subset).
    subset : str, default "FD001"
        Dataset subset name.
    window_length : int, default 30
        Sequence window length.
    rul_clip : int, default 125
        Upper bound for clipped target evaluation.
    is_3d : bool, default False
        Whether the model expects 3D inputs (batch, time, features) or 2D (batch, time*features).
    metadata_path : str or Path, optional
        Path to preprocess_metadata.json.

    Returns
    -------
    dict
        Dictionary containing:
        - 'metrics_raw': Dict with MAE, RMSE, NASA_score on raw ground-truth RUL
        - 'metrics_clipped': Dict with MAE, RMSE, NASA_score on clipped ground-truth RUL
        - 'predictions': List of predicted RUL for each engine (in ascending engine_id order)
        - 'ground_truth_raw': List of true RUL values
        - 'ground_truth_clipped': List of clipped true RUL values
        - 'engine_details': Detailed record per engine
        - 'num_engines_evaluated': Number of evaluated engines
    """
    project_root = pathlib.Path(__file__).resolve().parent.parent
    if metadata_path is None:
        metadata_path = project_root / "dataset" / "processed" / "preprocess_metadata.json"
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    sensor_cols = meta["selected_sensors"]
    sensor_means = np.array([meta["sensor_means"][s] for s in sensor_cols], dtype=np.float32)
    sensor_stds = np.array([meta["sensor_stds"][s] for s in sensor_cols], dtype=np.float32)

    if test_df is None:
        from src.data_loader import load_dataset
        test_df = load_dataset(subset, kind="test")

    if rul_series is None:
        from src.data_loader import load_rul
        rul_series = load_rul(subset)

    rul_values = np.asarray(rul_series, dtype=np.float32).flatten()
    engine_ids = sorted(test_df["engine_id"].unique())

    if len(rul_values) != len(engine_ids):
        raise ValueError(
            f"Mismatch between number of test engines ({len(engine_ids)}) and RUL labels ({len(rul_values)})"
        )

    true_ruls_raw = []
    true_ruls_clipped = []
    pred_ruls = []
    engine_details = []

    for idx, eid in enumerate(engine_ids):
        actual_rul_raw = float(rul_values[idx])
        actual_rul_clip = float(min(actual_rul_raw, rul_clip))

        eng_df = test_df[test_df["engine_id"] == eid].sort_values("cycle").reset_index(drop=True)
        if len(eng_df) == 0:
            continue

        last_cycle = int(eng_df["cycle"].iloc[-1])

        # Handle trajectory length (pad if shorter than window_length)
        if len(eng_df) < window_length:
            pad_count = window_length - len(eng_df)
            first_row = eng_df[sensor_cols].iloc[[0]]
            pad_df = pd.concat([first_row] * pad_count, ignore_index=True)
            sensor_vals = pd.concat([pad_df, eng_df[sensor_cols]], ignore_index=True).values.astype(np.float32)
        else:
            sensor_vals = eng_df[sensor_cols].iloc[-window_length:].values.astype(np.float32)

        # Normalize with training stats
        norm_window = (sensor_vals - sensor_means) / sensor_stds

        # Predict RUL
        if is_3d:
            if hasattr(model, "predict"):
                pred = float(model.predict(norm_window[np.newaxis, ...]).squeeze())
            else:
                import torch
                model.eval()
                with torch.no_grad():
                    t_in = torch.tensor(norm_window[np.newaxis, ...], dtype=torch.float32)
                    pred = float(model(t_in).item())
        else:
            flat = norm_window.reshape(1, -1)
            pred = float(model.predict(flat)[0])

        true_ruls_raw.append(actual_rul_raw)
        true_ruls_clipped.append(actual_rul_clip)
        pred_ruls.append(pred)

        engine_details.append({
            "engine_id": int(eid),
            "final_observed_cycle": last_cycle,
            "actual_rul_raw": actual_rul_raw,
            "actual_rul_clipped": actual_rul_clip,
            "predicted_rul": float(pred),
            "error_raw": float(pred - actual_rul_raw),
            "error_clipped": float(pred - actual_rul_clip)
        })

    metrics_raw = compute_metrics(true_ruls_raw, pred_ruls)
    metrics_clipped = compute_metrics(true_ruls_clipped, pred_ruls)

    return {
        "metrics_raw": metrics_raw,
        "metrics_clipped": metrics_clipped,
        "predictions": pred_ruls,
        "ground_truth_raw": true_ruls_raw,
        "ground_truth_clipped": true_ruls_clipped,
        "engine_details": engine_details,
        "num_engines_evaluated": len(true_ruls_raw)
    }
