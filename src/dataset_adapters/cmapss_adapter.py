"""NASA C-MAPSS Dataset Adapter (FD001, FD002, FD003, FD004).

Provides unified data loading, trajectory extraction, active sensor filtering,
operating regime identification, and validation for all 4 NASA C-MAPSS subsets.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from src.dataset_adapters.base_adapter import BaseDatasetAdapter
from src.data_loader import load_dataset, load_rul, SENSOR_COLS, OP_COLS, COLUMN_NAMES
from src.dataset_metadata import (
    CMAPSS_SENSOR_METADATA,
    CMAPSS_OP_SETTINGS_METADATA,
    DATASET_PROFILES,
)
from src.dataset_validator import DatasetValidator


class CMAPSSAdapter(BaseDatasetAdapter):
    """Concrete adapter for NASA C-MAPSS FD001-FD004 turbofan datasets."""

    def __init__(self, subset: str = "FD001"):
        subset_upper = subset.upper()
        if subset_upper not in {"FD001", "FD002", "FD003", "FD004"}:
            raise ValueError(f"Unsupported C-MAPSS subset '{subset}'. Must be one of FD001, FD002, FD003, FD004.")
        super().__init__(dataset_name=subset_upper)
        self._train_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame] = None
        self._rul_series: Optional[pd.Series] = None
        self._active_sensors: Optional[List[str]] = None

    def load_train(self, reload: bool = False) -> pd.DataFrame:
        """Load or retrieve cached run-to-failure training data."""
        if self._train_df is None or reload:
            self._train_df = load_dataset(self.dataset_name, kind="train")
        return self._train_df

    def load_test(self, reload: bool = False) -> pd.DataFrame:
        """Load or retrieve cached truncated test data."""
        if self._test_df is None or reload:
            self._test_df = load_dataset(self.dataset_name, kind="test")
        return self._test_df

    def load_rul(self, reload: bool = False) -> pd.Series:
        """Load or retrieve cached ground-truth RUL vector for test fleet."""
        if self._rul_series is None or reload:
            self._rul_series = load_rul(self.dataset_name)
        return self._rul_series

    def get_metadata(self) -> Dict[str, Any]:
        """Return standardized dataset profile and engineering descriptions."""
        profile = DATASET_PROFILES.get(self.dataset_name, {}).copy()
        profile["available_sensors_count"] = len(SENSOR_COLS)
        profile["operating_settings"] = OP_COLS
        profile["active_sensors"] = self.get_active_sensors()
        profile["active_sensors_count"] = len(profile["active_sensors"])
        return profile

    def get_engine_ids(self, kind: str = "train") -> List[int]:
        """Return sorted list of engine IDs in the requested split."""
        df = self.load_train() if kind == "train" else self.load_test()
        return sorted([int(x) for x in df["engine_id"].unique()])

    def get_engine_trajectory(self, engine_id: int, kind: str = "test") -> pd.DataFrame:
        """Extract all time-series records for a specific engine."""
        df = self.load_test() if kind == "test" else self.load_train()
        engine_df = df[df["engine_id"] == engine_id].sort_values("cycle").copy()
        if engine_df.empty:
            raise ValueError(f"Engine ID {engine_id} not found in {self.dataset_name} ({kind}).")
        return engine_df

    def get_feature_names(self) -> List[str]:
        """Return all standard column features (3 settings + 21 sensors)."""
        return OP_COLS + SENSOR_COLS

    def get_active_sensors(self) -> List[str]:
        """Detect informative sensor channels with non-zero variance on training data."""
        if self._active_sensors is None:
            train_df = self.load_train()
            stds = train_df[SENSOR_COLS].std()
            self._active_sensors = [col for col in SENSOR_COLS if stds[col] > 1e-5]
        return self._active_sensors

    def get_op_settings(self) -> List[str]:
        """Return operational setting column names."""
        return OP_COLS.copy()

    def get_sensor_info(self, sensor_col: str) -> Dict[str, Any]:
        """Return physical metadata (symbol, name, units, description) for a sensor."""
        return CMAPSS_SENSOR_METADATA.get(sensor_col, {
            "symbol": sensor_col,
            "name": sensor_col,
            "units": "—",
            "description": "Unspecified sensor channel."
        })

    def detect_flight_regimes(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Categorize records into discrete operating regimes for multi-regime subsets.

        For FD002 and FD004 (6 flight regimes), rounds altitude, Mach, and TRA
        to cluster into the 6 discrete operating regimes identified in literature.
        """
        target_df = self.load_train() if df is None else df.copy()

        # Altitude ~ [0, 10, 20, 25, 35, 42] kft
        # Mach ~ [0.0, 0.25, 0.40, 0.62, 0.70, 0.84]
        # TRA ~ [0, 20, 40, 60, 80, 100]
        regime_alt = (target_df["op_setting_1"] / 5.0).round() * 5.0
        regime_mach = (target_df["op_setting_2"] * 10.0).round() / 10.0
        regime_tra = (target_df["op_setting_3"] / 10.0).round() * 10.0

        regime_key = regime_alt.astype(str) + "_" + regime_mach.astype(str) + "_" + regime_tra.astype(str)
        unique_keys = {key: idx + 1 for idx, key in enumerate(regime_key.unique())}

        result = target_df.copy()
        result["regime_id"] = regime_key.map(unique_keys)
        return result

    def validate(self) -> Dict[str, Any]:
        """Run comprehensive automated data validation."""
        train_df = self.load_train()
        test_df = self.load_test()
        rul_series = self.load_rul()
        report = DatasetValidator.validate_dataset(train_df, test_df, rul_series)
        report["dataset_name"] = self.dataset_name
        return report
