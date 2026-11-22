"""Configurable Dataset Loading and Preprocessing Pipeline (THRUSTLAB Phase 2).

Provides an end-to-end, zero-leakage, configurable pipeline supporting:
1. Datasets: FD001, FD002, FD003, FD004.
2. Sensor Selection: auto-variance filtering, manual lists, or all available channels.
3. Normalization: StandardScaler (Z-score), MinMaxScaler ([0, 1]), and RobustScaler (Median/IQR),
   fitted strictly on training engines.
4. Sequence Generation: configurable window length, sliding stride, and short-trajectory padding.
5. RUL Target Formulation: separate train clipping and evaluation bounds.
"""

from __future__ import annotations

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd

from src.data_loader import load_dataset, load_rul, SENSOR_COLS, OP_COLS
from src.config_manager import MasterConfig


class ConfigurableDataPipeline:
    """End-to-end dataset loader and preprocessor parameterized by MasterConfig."""

    def __init__(self, config: Optional[MasterConfig] = None):
        self.config = config or MasterConfig()
        self.dataset_cfg = self.config.dataset
        self.sensor_cfg = self.config.sensor
        self.prep_cfg = self.config.preprocessing

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Load raw training data, test data, and test RUL for configured subset."""
        subset = self.dataset_cfg.subset
        train_df = load_dataset(subset, kind="train")
        test_df = load_dataset(subset, kind="test")
        test_rul = load_rul(subset)
        return train_df, test_df, test_rul

    def split_engine_ids(self, df: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """Perform disjoint engine-level train/validation split (zero sequence overlap)."""
        engine_ids = sorted(df["engine_id"].unique().tolist())
        rng = random.Random(self.dataset_cfg.seed)
        rng.shuffle(engine_ids)
        split_idx = int(len(engine_ids) * self.dataset_cfg.train_val_split)
        train_ids = sorted(engine_ids[:split_idx])
        val_ids = sorted(engine_ids[split_idx:])
        return train_ids, val_ids

    def determine_features(self, train_df: pd.DataFrame) -> List[str]:
        """Determine active input feature columns strictly using training engines."""
        mode = self.sensor_cfg.selection_mode
        candidate_sensors = [c for c in train_df.columns if c.startswith("sensor_")]

        if mode == "manual" and self.sensor_cfg.manual_sensors:
            selected = [s for s in self.sensor_cfg.manual_sensors if s in candidate_sensors]
        elif mode == "all":
            selected = candidate_sensors
        else:
            # "auto_variance"
            selected = []
            for s in candidate_sensors:
                if s in self.sensor_cfg.default_excluded:
                    continue
                std_val = train_df[s].std()
                if std_val > self.sensor_cfg.variance_threshold:
                    selected.append(s)

        if self.sensor_cfg.include_operational_settings:
            op_cols = [c for c in train_df.columns if c.startswith("op_setting_")]
            features = op_cols + selected
        else:
            features = selected

        if not features:
            raise ValueError("No feature columns selected. Verify sensor configuration.")
        return features

    def fit_scaler(self, train_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        """Fit normalization parameters strictly on training engines.
        
        Supports:
        - 'standard': mean and standard deviation
        - 'minmax': minimum and maximum (scaled to [0, 1])
        - 'robust': median and interquartile range (IQR = Q75 - Q25)
        """
        scaler_type = self.prep_cfg.scaler_type
        stats: Dict[str, Any] = {"scaler_type": scaler_type, "features": feature_cols}

        if scaler_type == "standard":
            means = train_df[feature_cols].mean().to_dict()
            stds = train_df[feature_cols].std().replace(0.0, 1.0).to_dict()
            stats["param_1"] = means  # center
            stats["param_2"] = stds   # scale
        elif scaler_type == "minmax":
            mins = train_df[feature_cols].min().to_dict()
            maxs = train_df[feature_cols].max().to_dict()
            diffs = {f: max(maxs[f] - mins[f], 1e-6) for f in feature_cols}
            stats["param_1"] = mins   # min
            stats["param_2"] = diffs  # max - min
        elif scaler_type == "robust":
            medians = train_df[feature_cols].median().to_dict()
            q25 = train_df[feature_cols].quantile(0.25).to_dict()
            q75 = train_df[feature_cols].quantile(0.75).to_dict()
            iqrs = {f: max(q75[f] - q25[f], 1e-6) for f in feature_cols}
            stats["param_1"] = medians  # center
            stats["param_2"] = iqrs     # scale
        else:
            raise ValueError(f"Unsupported scaler_type '{scaler_type}'")

        return stats

    def transform(
        self, df: pd.DataFrame, feature_cols: List[str], scaler_stats: Dict[str, Any]
    ) -> pd.DataFrame:
        """Apply fitted normalization parameters to dataframe without data leakage."""
        df_norm = df.copy()
        scaler_type = scaler_stats["scaler_type"]
        p1 = pd.Series(scaler_stats["param_1"])
        p2 = pd.Series(scaler_stats["param_2"])

        if scaler_type in ("standard", "robust"):
            df_norm[feature_cols] = (df[feature_cols] - p1) / p2
        elif scaler_type == "minmax":
            df_norm[feature_cols] = (df[feature_cols] - p1) / p2
        return df_norm

    def generate_sequences(
        self,
        df: pd.DataFrame,
        engine_ids: List[int],
        feature_cols: List[str],
        window_size: int,
        stride: int,
        rul_clip: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate sliding-window sequence tensors for given engine IDs.
        
        Returns:
            X: np.ndarray of shape (N, window_size, len(feature_cols))
            y: np.ndarray of shape (N,)
        """
        X_list = []
        y_list = []

        for eid in engine_ids:
            eng_df = df[df["engine_id"] == eid].sort_values("cycle").reset_index(drop=True)
            cycles = eng_df["cycle"].values
            if len(cycles) == 0:
                continue
            max_cycle = cycles[-1]
            features = eng_df[feature_cols].values

            n_cycles = len(eng_df)
            if n_cycles < window_size:
                if self.prep_cfg.padding_mode == "first":
                    pad_count = window_size - n_cycles
                    pad_block = np.repeat(features[[0], :], pad_count, axis=0)
                    padded_feat = np.vstack([pad_block, features])
                    X_list.append(padded_feat[np.newaxis, ...])
                    y_list.append([min(max_cycle - cycles[-1], rul_clip)])
                continue

            for start in range(0, n_cycles - window_size + 1, stride):
                end = start + window_size
                window_data = features[start:end, :]
                last_cycle = cycles[end - 1]
                rul = min(max_cycle - last_cycle, rul_clip)
                X_list.append(window_data)
                y_list.append(rul)

        if not X_list:
            return np.empty((0, window_size, len(feature_cols)), dtype=np.float32), np.empty((0,), dtype=np.float32)

        X = np.array(X_list, dtype=np.float32)
        if X.ndim == 4:
            X = X.squeeze(1)
        y = np.array(y_list, dtype=np.float32).flatten()
        return X, y

    def prepare_data(self) -> Dict[str, Any]:
        """Execute full preprocessing pipeline according to configuration.
        
        Returns dictionary containing:
            X_train, y_train, X_val, y_val, feature_cols, scaler_stats,
            train_engine_ids, val_engine_ids, metadata
        """
        raw_train, raw_test, test_rul = self.load_raw_data()
        raw_train = raw_train.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

        train_ids, val_ids = self.split_engine_ids(raw_train)
        train_df_raw = raw_train[raw_train["engine_id"].isin(train_ids)].copy()
        val_df_raw = raw_train[raw_train["engine_id"].isin(val_ids)].copy()

        feature_cols = self.determine_features(train_df_raw)
        scaler_stats = self.fit_scaler(train_df_raw, feature_cols)

        train_norm = self.transform(train_df_raw, feature_cols, scaler_stats)
        val_norm = self.transform(val_df_raw, feature_cols, scaler_stats)

        X_train, y_train = self.generate_sequences(
            train_norm, train_ids, feature_cols,
            self.prep_cfg.window_size, self.prep_cfg.stride, self.prep_cfg.rul_clip_train
        )
        X_val, y_val = self.generate_sequences(
            val_norm, val_ids, feature_cols,
            self.prep_cfg.window_size, self.prep_cfg.stride, self.prep_cfg.rul_clip_train
        )

        metadata = {
            "subset": self.dataset_cfg.subset,
            "feature_cols": feature_cols,
            "num_features": len(feature_cols),
            "window_size": self.prep_cfg.window_size,
            "stride": self.prep_cfg.stride,
            "scaler_type": self.prep_cfg.scaler_type,
            "rul_clip_train": self.prep_cfg.rul_clip_train,
            "rul_clip_eval": self.prep_cfg.rul_clip_eval,
            "num_train_sequences": int(X_train.shape[0]),
            "num_val_sequences": int(X_val.shape[0]),
            "train_engine_count": len(train_ids),
            "val_engine_count": len(val_ids)
        }

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "feature_cols": feature_cols,
            "scaler_stats": scaler_stats,
            "train_engine_ids": train_ids,
            "val_engine_ids": val_ids,
            "metadata": metadata
        }
