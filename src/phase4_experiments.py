"""Phase 4 Systematic Experiments: Evaluation Audit and Regime-Adaptive Normalization.

Executes EXP-001 through EXP-006:
- EXP-001: Global Normalization (Baseline)
- EXP-002: Condition-wise StandardScaler
- EXP-003: Condition-wise MinMaxScaler
- EXP-004: Condition-wise RobustScaler
- EXP-005: Regime-Aware Feature Engineering (Condition-wise Scaler + Regime Indicators)
- EXP-006: Adaptive Normalization + Regime Variance Selection

Evaluates across FD001 (control), FD002 (6 regimes), FD003 (2 faults), FD004 (6 regimes + 2 faults).
"""

import json
import time
import pathlib
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from src.data_loader import load_dataset, load_rul, SENSOR_COLS, OP_COLS
from src.evaluation import compute_metrics, compute_nasa_score
from src.regime_normalizer import RegimeAdaptiveScaler


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


def prepare_sequences_or_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_length: int = 30,
    kind: str = "train"
) -> Tuple[np.ndarray, np.ndarray]:
    """Fast vectorized window extraction for tabular estimators."""
    from numpy.lib.stride_tricks import sliding_window_view
    X_list = []
    y_list = []

    for eid in sorted(df["engine_id"].unique()):
        eng_df = df[df["engine_id"] == eid].sort_values("cycle").reset_index(drop=True)
        vals = eng_df[feature_cols].values.astype(np.float32)
        n_cycles = len(vals)

        if kind == "train":
            if n_cycles < window_length:
                continue
            # Fast vectorized sliding window: shape (N - W + 1, features, W)
            w = sliding_window_view(vals, window_shape=window_length, axis=0)
            # Reshape to (N - W + 1, W * features)
            w_flat = np.swapaxes(w, 1, 2).reshape(len(w), -1)
            X_list.append(w_flat)

            # RUL targets
            cycles = eng_df["cycle"].values[window_length - 1:]
            ruls = np.clip(n_cycles - cycles, 0, 125).astype(np.float32)
            y_list.append(ruls)
        else:
            # Test set: take single final window ending at last cycle
            if n_cycles >= window_length:
                w = vals[-window_length:].flatten()
            else:
                pad_count = window_length - n_cycles
                pad = np.repeat(vals[0:1], pad_count, axis=0)
                w = np.vstack([pad, vals]).flatten()
            X_list.append(w.reshape(1, -1))

    if kind == "train":
        if X_list:
            return np.vstack(X_list), np.concatenate(y_list)
        return np.empty((0, window_length * len(feature_cols)), dtype=np.float32), np.empty(0, dtype=np.float32)
    else:
        return np.vstack(X_list), np.empty(0, dtype=np.float32)


def run_experiment(
    exp_id: str,
    exp_name: str,
    scaler_type: str,
    include_regime_features: bool,
    subset: str = "FD002",
    model_type: str = "gbr"
) -> Dict[str, Any]:
    """Execute a single experimental run."""
    print(f"\n--- Running {exp_id} ({exp_name}) on {subset} ---")
    t0 = time.time()

    train_df = load_dataset(subset, kind="train")
    test_df = load_dataset(subset, kind="test")
    test_rul = load_rul(subset).values.astype(np.float32)

    # Feature columns: active sensors with variance > 0
    stds = train_df[SENSOR_COLS].std()
    active_sensors = [s for s in SENSOR_COLS if stds[s] > 1e-4]
    candidate_features = active_sensors.copy()

    # Preprocessing transformation
    if exp_id == "EXP-001":
        # Global StandardScaler
        means = train_df[candidate_features].mean()
        stds_g = train_df[candidate_features].std().replace(0.0, 1.0)
        norm_train = train_df.copy()
        norm_test = test_df.copy()
        norm_train[candidate_features] = (train_df[candidate_features] - means) / stds_g
        norm_test[candidate_features] = (test_df[candidate_features] - means) / stds_g
        final_feature_cols = candidate_features

    elif exp_id in ("EXP-002", "EXP-003", "EXP-004", "EXP-005", "EXP-006"):
        scaler = RegimeAdaptiveScaler(
            scaler_type=scaler_type,
            regime_method="kmeans",
            n_clusters=6,
            include_regime_features=include_regime_features
        )
        norm_train = scaler.fit_transform(train_df, candidate_features)
        norm_test = scaler.transform(test_df)

        if include_regime_features:
            regime_cols = [c for c in norm_train.columns if c.startswith("regime_")]
            final_feature_cols = candidate_features + regime_cols
        else:
            final_feature_cols = candidate_features

    # Subsample training windows for speedy, reproducible comparison
    X_train, y_train = prepare_sequences_or_windows(
        norm_train,
        feature_cols=final_feature_cols,
        window_length=30,
        kind="train"
    )
    X_test, _ = prepare_sequences_or_windows(
        norm_test,
        feature_cols=final_feature_cols,
        window_length=30,
        kind="test"
    )

    # Subsample 1 out of every 3 cycles if training dataset is large (FD002/FD004)
    if len(X_train) > 20000:
        sub_indices = np.arange(0, len(X_train), 3)
        X_train_sub = X_train[sub_indices]
        y_train_sub = y_train[sub_indices]
    else:
        X_train_sub = X_train
        y_train_sub = y_train

    # Train estimator
    if model_type == "gbr":
        model = HistGradientBoostingRegressor(
            max_iter=60,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
    else:
        model = Ridge(alpha=1.0)

    model.fit(X_train_sub, y_train_sub)
    preds = model.predict(X_test)
    preds_clipped = np.clip(preds, 0, 125)

    duration = time.time() - t0

    # Metrics against raw official test RUL
    raw_metrics = compute_metrics(test_rul, preds)
    clipped_metrics = compute_metrics(np.clip(test_rul, 0, 125), preds_clipped)

    print(f"  Duration: {duration:.2f}s | Features: {len(final_feature_cols)}")
    print(f"  Raw Target:     MAE: {raw_metrics['MAE']:.2f}, RMSE: {raw_metrics['RMSE']:.2f}, NASA: {raw_metrics['NASA_score']:.2f}")
    print(f"  Clipped Target: MAE: {clipped_metrics['MAE']:.2f}, RMSE: {clipped_metrics['RMSE']:.2f}, NASA: {clipped_metrics['NASA_score']:.2f}")

    return {
        "exp_id": exp_id,
        "exp_name": exp_name,
        "subset": subset,
        "scaler_type": scaler_type,
        "include_regime_features": include_regime_features,
        "feature_count": len(final_feature_cols),
        "duration_seconds": round(duration, 2),
        "metrics_raw": raw_metrics,
        "metrics_clipped": clipped_metrics
    }


def run_all_phase4_experiments():
    """Run full benchmark matrix across FD001, FD002, FD004, FD003."""
    experiments = [
        ("EXP-001", "Global Normalization (Baseline)", "standard", False),
        ("EXP-002", "Condition-wise StandardScaler", "standard", False),
        ("EXP-003", "Condition-wise MinMaxScaler", "minmax", False),
        ("EXP-004", "Condition-wise RobustScaler", "robust", False),
        ("EXP-005", "Regime-Aware Feature Engineering", "standard", True),
        ("EXP-006", "Adaptive Robust Normalization + Features", "robust", True),
    ]

    all_results = {}

    # Target multi-regime datasets: FD002 first, then FD004, and control on FD001
    for subset in ["FD002", "FD004", "FD001", "FD003"]:
        print(f"\n=======================================================")
        print(f"BENCHMARKING PHASE 4 ON DATASET {subset}")
        print(f"=======================================================")
        subset_res = []
        for exp_id, exp_name, stype, inc_reg in experiments:
            res = run_experiment(exp_id, exp_name, stype, inc_reg, subset=subset)
            subset_res.append(res)
        all_results[subset] = subset_res

    out_file = PROJECT_ROOT / "results" / "phase4_normalization_experiments.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[x] All Phase 4 experiments completed. Saved to {out_file}")
    return all_results


if __name__ == "__main__":
    run_all_phase4_experiments()
