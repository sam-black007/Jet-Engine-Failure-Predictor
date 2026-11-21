"""Historical Dataset Replay and Simulated Engine Telemetry Engine.

Provides cycle-by-cycle simulated telemetry streaming, sliding window extraction,
real-time RUL degradation tracking, and automated threshold alerts.

NOTE: This simulates telemetry playback from historical benchmark datasets
(e.g., NASA C-MAPSS) for predictive maintenance verification and training.
"""

from typing import Dict, List, Optional, Any, Iterator, Tuple
import time
import numpy as np
import pandas as pd

from src.dataset_registry import DatasetRegistry
from src.dataset_adapters.base_adapter import BaseDatasetAdapter


class EngineReplaySimulator:
    """Historical telemetry replay and simulated condition monitoring engine."""

    def __init__(self, dataset_name: str = "FD001"):
        self.dataset_name = dataset_name.upper()
        self.adapter: BaseDatasetAdapter = DatasetRegistry.get_adapter(self.dataset_name)
        self.current_engine_id: Optional[int] = None
        self.current_cycle: int = 0
        self.trajectory_df: Optional[pd.DataFrame] = None
        self.max_cycles: int = 0
        self.is_paused: bool = False
        self.speed_factor: float = 1.0  # 1.0 = normal, 2.0 = 2x, etc.
        self.ground_truth_final_rul: Optional[int] = None
        self.prediction_history: List[Dict[str, Any]] = []

    def load_engine(self, engine_id: int, kind: str = "test") -> Dict[str, Any]:
        """Initialize replay session for a specific engine."""
        self.current_engine_id = engine_id
        self.trajectory_df = self.adapter.get_engine_trajectory(engine_id=engine_id, kind=kind)
        self.max_cycles = int(self.trajectory_df["cycle"].max())
        self.current_cycle = 1
        self.prediction_history = []

        # Retrieve ground truth RUL if available
        self.ground_truth_final_rul = None
        if kind == "test":
            try:
                rul_series = self.adapter.load_rul()
                engine_ids = self.adapter.get_engine_ids(kind="test")
                if engine_id in engine_ids:
                    idx = engine_ids.index(engine_id)
                    self.ground_truth_final_rul = int(rul_series.iloc[idx])
            except Exception:
                pass

        return {
            "engine_id": engine_id,
            "dataset": self.dataset_name,
            "kind": kind,
            "total_cycles": self.max_cycles,
            "ground_truth_final_rul": self.ground_truth_final_rul,
            "active_sensors": self.adapter.get_active_sensors()
        }

    def get_cycle_snapshot(self, cycle: int) -> Dict[str, Any]:
        """Retrieve single operational snapshot for specified cycle."""
        if self.trajectory_df is None:
            raise ValueError("No engine trajectory loaded. Call load_engine() first.")

        row_df = self.trajectory_df[self.trajectory_df["cycle"] == cycle]
        if row_df.empty:
            raise ValueError(f"Cycle {cycle} not found in engine {self.current_engine_id} trajectory.")

        row = row_df.iloc[0].to_dict()

        # Calculate exact ground-truth RUL for this cycle if test engine
        cycle_ground_truth_rul = None
        if self.ground_truth_final_rul is not None:
            cycles_remaining_to_test_end = self.max_cycles - cycle
            cycle_ground_truth_rul = self.ground_truth_final_rul + cycles_remaining_to_test_end

        return {
            "engine_id": self.current_engine_id,
            "cycle": cycle,
            "telemetry": row,
            "cycle_ground_truth_rul": cycle_ground_truth_rul,
            "is_last_observed_cycle": cycle == self.max_cycles
        }

    def get_sliding_window(
        self,
        cycle: int,
        window_size: int = 30,
        feature_cols: Optional[List[str]] = None
    ) -> np.ndarray:
        """Extract historical sliding window up to the specified cycle.

        If cycle < window_size, pads front with the earliest available cycle values
        (zero-leakage edge padding).
        """
        if self.trajectory_df is None:
            raise ValueError("No engine trajectory loaded.")

        if feature_cols is None:
            feature_cols = self.adapter.get_active_sensors()

        sub_df = self.trajectory_df[self.trajectory_df["cycle"] <= cycle].sort_values("cycle")
        available_rows = sub_df[feature_cols].values

        if len(available_rows) >= window_size:
            window = available_rows[-window_size:]
        else:
            # Pad front with earliest observed cycle
            pad_count = window_size - len(available_rows)
            pad_values = np.repeat(available_rows[0:1], pad_count, axis=0)
            window = np.vstack([pad_values, available_rows])

        return window.astype(np.float32)

    @staticmethod
    def classify_risk(predicted_rul: float) -> Tuple[str, str]:
        """Classify predictive maintenance risk category and UI color indicator."""
        if predicted_rul > 50.0:
            return "HEALTHY", "#22c55e"       # Green
        elif predicted_rul > 25.0:
            return "MONITOR", "#eab308"       # Yellow
        elif predicted_rul > 10.0:
            return "WARNING", "#f97316"       # Orange
        else:
            return "CRITICAL", "#ef4444"      # Red

    def record_prediction(self, cycle: int, predicted_rul: float, uncertainty_std: Optional[float] = None) -> Dict[str, Any]:
        """Store historical prediction result with timestamp, risk assessment, and drift."""
        risk_level, color = self.classify_risk(predicted_rul)
        snapshot = self.get_cycle_snapshot(cycle)
        gt_rul = snapshot["cycle_ground_truth_rul"]

        error = float(predicted_rul - gt_rul) if gt_rul is not None else None

        entry = {
            "cycle": cycle,
            "predicted_rul": float(predicted_rul),
            "uncertainty_std": float(uncertainty_std) if uncertainty_std is not None else None,
            "ground_truth_rul": gt_rul,
            "error": error,
            "risk_level": risk_level,
            "indicator_color": color
        }
        self.prediction_history.append(entry)
        return entry

    def step(self) -> Optional[Dict[str, Any]]:
        """Advance replay by one operational cycle."""
        if self.current_cycle > self.max_cycles:
            return None
        snapshot = self.get_cycle_snapshot(self.current_cycle)
        self.current_cycle += 1
        return snapshot

    def reset(self):
        """Reset replay playback to cycle 1."""
        self.current_cycle = 1
        self.prediction_history = []
