"""Base Dataset Adapter Interface for Turbofan Predictive Maintenance.

Defines the abstract contract for loading, inspecting, filtering, and validating
turbofan engine sensor telemetry across heterogeneous datasets (NASA C-MAPSS, N-CMAPSS, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd


class BaseDatasetAdapter(ABC):
    """Abstract Base Class for turbofan dataset ingestion and normalization."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name.upper()

    @abstractmethod
    def load_train(self) -> pd.DataFrame:
        """Load the run-to-failure training trajectories."""
        pass

    @abstractmethod
    def load_test(self) -> pd.DataFrame:
        """Load the truncated testing trajectories."""
        pass

    @abstractmethod
    def load_rul(self) -> pd.Series:
        """Load ground-truth Remaining Useful Life labels for test engines."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return dataset specifications, operating conditions, and fault descriptions."""
        pass

    @abstractmethod
    def get_engine_ids(self, kind: str = "train") -> List[int]:
        """Return unique engine identifier list for specified split."""
        pass

    @abstractmethod
    def get_engine_trajectory(self, engine_id: int, kind: str = "test") -> pd.DataFrame:
        """Extract continuous time-series records for a specific engine."""
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return full list of candidate feature columns (settings + sensors)."""
        pass

    @abstractmethod
    def get_active_sensors(self) -> List[str]:
        """Return list of informative, non-invariant sensor channels."""
        pass

    @abstractmethod
    def get_op_settings(self) -> List[str]:
        """Return operational flight condition setting column names."""
        pass

    @abstractmethod
    def validate(self) -> Dict[str, Any]:
        """Run automated data quality and integrity audit."""
        pass
