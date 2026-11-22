"""Operating Regime Detection and Condition-Wise Adaptive Normalization.

Provides condition-wise scaling for multi-regime turbofan datasets (FD002, FD004)
to eliminate aerodynamic and environmental sensor variance induced by altitude,
Mach number, and throttle resolver angle (TRA).
"""

from typing import Dict, List, Optional, Any, Tuple, Literal
import json
import pathlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.data_loader import OP_COLS, SENSOR_COLS


class RegimeDetector:
    """Identifies discrete operating flight conditions from operational settings."""

    def __init__(
        self,
        method: Literal["kmeans", "discrete_bins"] = "kmeans",
        n_clusters: int = 6,
        random_state: int = 42
    ):
        self.method = method
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans: Optional[KMeans] = None
        self.cluster_centers: Optional[np.ndarray] = None

    def fit(self, train_op_df: pd.DataFrame) -> "RegimeDetector":
        """Fit cluster boundaries strictly on training operational settings."""
        op_data = train_op_df[OP_COLS].values.astype(np.float32)

        # In single-condition datasets (FD001, FD003), setting variance is zero
        std_ops = np.std(op_data, axis=0)
        if np.all(std_ops < 1e-3):
            # Trivial 1-condition dataset
            self.n_clusters = 1
            self.cluster_centers = np.mean(op_data, axis=0, keepdims=True)
            return self

        if self.method == "kmeans":
            self.kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            self.kmeans.fit(op_data)
            self.cluster_centers = self.kmeans.cluster_centers_
        else:
            # Discrete rounding method
            self.n_clusters = 6
        return self

    def predict(self, op_df: pd.DataFrame) -> np.ndarray:
        """Assign records to their corresponding regime ID."""
        op_data = op_df[OP_COLS].values.astype(np.float32)
        if self.n_clusters == 1 or self.cluster_centers is None:
            return np.zeros(len(op_df), dtype=int)

        if self.kmeans is not None:
            return self.kmeans.predict(op_data).astype(int)
        else:
            # Binning based on Altitude / Mach / TRA
            alt = (op_data[:, 0] / 5.0).round() * 5.0
            mach = (op_data[:, 1] * 10.0).round() / 10.0
            tra = (op_data[:, 2] / 10.0).round() * 10.0
            keys = [f"{a}_{m}_{t}" for a, m, t in zip(alt, mach, tra)]
            unique_keys = sorted(list(set(keys)))
            mapping = {k: idx for idx, k in enumerate(unique_keys)}
            return np.array([mapping.get(k, 0) for k in keys], dtype=int)

    def fit_predict(self, op_df: pd.DataFrame) -> np.ndarray:
        return self.fit(op_df).predict(op_df)


class RegimeAdaptiveScaler:
    """Condition-wise normalizer scaling features within each operating regime."""

    def __init__(
        self,
        scaler_type: Literal["standard", "minmax", "robust"] = "standard",
        regime_method: Literal["kmeans", "discrete_bins"] = "kmeans",
        n_clusters: int = 6,
        include_regime_features: bool = False,
        random_state: int = 42
    ):
        self.scaler_type = scaler_type.lower()
        self.include_regime_features = include_regime_features
        self.detector = RegimeDetector(
            method=regime_method,
            n_clusters=n_clusters,
            random_state=random_state
        )
        self.feature_cols: List[str] = []
        self.stats_by_regime: Dict[int, Dict[str, Dict[str, float]]] = {}
        self.is_fitted: bool = False

    def fit(self, train_df: pd.DataFrame, feature_cols: List[str]) -> "RegimeAdaptiveScaler":
        """Fit regime detector and calculate per-regime scaling statistics."""
        self.feature_cols = feature_cols.copy()
        self.detector.fit(train_df[OP_COLS])
        regime_ids = self.detector.predict(train_df[OP_COLS])

        unique_regimes = sorted(np.unique(regime_ids))
        self.stats_by_regime = {}

        for r in unique_regimes:
            sub = train_df.iloc[regime_ids == r][self.feature_cols]
            r_int = int(r)
            self.stats_by_regime[r_int] = {}

            if self.scaler_type == "standard":
                means = sub.mean()
                stds = sub.std().replace(0.0, 1.0).fillna(1.0)
                self.stats_by_regime[r_int]["center"] = means.to_dict()
                self.stats_by_regime[r_int]["scale"] = stds.to_dict()

            elif self.scaler_type == "minmax":
                mins = sub.min()
                maxs = sub.max()
                ranges = (maxs - mins).replace(0.0, 1.0).fillna(1.0)
                self.stats_by_regime[r_int]["center"] = mins.to_dict()
                self.stats_by_regime[r_int]["scale"] = ranges.to_dict()

            elif self.scaler_type == "robust":
                medians = sub.median()
                q25 = sub.quantile(0.25)
                q75 = sub.quantile(0.75)
                iqr = (q75 - q25).replace(0.0, 1.0).fillna(1.0)
                self.stats_by_regime[r_int]["center"] = medians.to_dict()
                self.stats_by_regime[r_int]["scale"] = iqr.to_dict()

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize dataframe using regime-specific parameters."""
        if not self.is_fitted:
            raise RuntimeError("RegimeAdaptiveScaler must be fitted before transform.")

        regime_ids = self.detector.predict(df[OP_COLS])
        norm_df = df.copy()

        # Ensure float dtype on features to prevent lossy setitem errors
        for col in self.feature_cols:
            norm_df[col] = norm_df[col].astype(np.float64)

        # Vectorized or grouped transformation
        for r, stats in self.stats_by_regime.items():
            mask = regime_ids == r
            if not np.any(mask):
                continue

            centers = pd.Series(stats["center"], dtype=np.float64)
            scales = pd.Series(stats["scale"], dtype=np.float64)

            sub_vals = norm_df.loc[mask, self.feature_cols]
            norm_vals = (sub_vals - centers) / scales
            norm_df.loc[mask, self.feature_cols] = norm_vals.values

        if self.include_regime_features:
            # Append one-hot encoded regime indicators
            for r in sorted(self.stats_by_regime.keys()):
                norm_df[f"regime_{r}"] = (regime_ids == r).astype(float)

        return norm_df

    def fit_transform(self, train_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Fit scaler on training data and return transformed dataframe."""
        return self.fit(train_df, feature_cols).transform(train_df)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration and statistics for experiment reproducibility."""
        return {
            "scaler_type": self.scaler_type,
            "feature_cols": self.feature_cols,
            "n_regimes": len(self.stats_by_regime),
            "include_regime_features": self.include_regime_features,
            "stats_by_regime": self.stats_by_regime
        }
