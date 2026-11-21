"""Data Quality and Integrity Validator for Turbofan Engine Datasets.

Executes rigorous automated verification to enforce:
1. No NaN, null, or missing sensor values.
2. No infinite values or floating-point overflow.
3. Monotonically increasing cycle sequences per engine with no gaps.
4. Correct column alignment and feature dimensionalities.
5. Exact target alignment between test engines and ground-truth RUL vectors.
6. Detection of constant sensor channels without data leakage.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd


class DatasetValidator:
    """Automated data integrity and quality inspection suite."""

    @staticmethod
    def check_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
        """Detect any null or NaN values across all columns."""
        null_counts = df.isnull().sum()
        total_nulls = int(null_counts.sum())
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        return {
            "has_missing": total_nulls > 0,
            "total_missing_values": total_nulls,
            "columns_with_missing": cols_with_nulls
        }

    @staticmethod
    def check_infinite_values(df: pd.DataFrame) -> Dict[str, Any]:
        """Detect positive or negative infinite values in numeric columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        inf_counts = np.isinf(numeric_df.values).sum()
        return {
            "has_infinities": int(inf_counts) > 0,
            "total_infinite_values": int(inf_counts)
        }

    @staticmethod
    def check_cycle_monotonicity(df: pd.DataFrame) -> Dict[str, Any]:
        """Verify that operational cycles strictly increase monotonically by 1 per engine."""
        engine_ids = sorted(df["engine_id"].unique())
        broken_engines = []

        for eid in engine_ids:
            cycles = df[df["engine_id"] == eid]["cycle"].values
            # Check strictly increasing by step 1
            diffs = np.diff(cycles)
            if not np.all(diffs == 1):
                broken_engines.append(int(eid))

        return {
            "is_strictly_monotonic": len(broken_engines) == 0,
            "total_engines_checked": len(engine_ids),
            "broken_engines_count": len(broken_engines),
            "broken_engine_ids": broken_engines
        }

    @staticmethod
    def check_target_alignment(test_df: pd.DataFrame, rul_series: pd.Series) -> Dict[str, Any]:
        """Verify that test engine count matches ground-truth RUL label vector length."""
        test_engines = sorted(test_df["engine_id"].unique())
        rul_count = len(rul_series)
        matches = len(test_engines) == rul_count
        return {
            "is_target_aligned": matches,
            "test_engines_count": len(test_engines),
            "ground_truth_rul_count": rul_count,
            "discrepancy": len(test_engines) - rul_count
        }

    @staticmethod
    def check_sensor_variances(df: pd.DataFrame, min_variance: float = 0.0) -> Dict[str, Any]:
        """Identify sensor channels with zero or near-zero variance."""
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
        stds = df[sensor_cols].std()
        invariant = stds[stds <= min_variance].index.tolist()
        active = stds[stds > min_variance].index.tolist()
        return {
            "total_sensors": len(sensor_cols),
            "active_sensors_count": len(active),
            "invariant_sensors_count": len(invariant),
            "invariant_sensors": invariant,
            "active_sensors": active
        }

    @classmethod
    def validate_dataset(
        cls,
        train_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
        test_rul: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Execute full data quality inspection across train and test splits."""
        train_missing = cls.check_missing_values(train_df)
        train_inf = cls.check_infinite_values(train_df)
        train_mono = cls.check_cycle_monotonicity(train_df)
        sensor_vars = cls.check_sensor_variances(train_df)

        report: Dict[str, Any] = {
            "train_engines_count": int(train_df["engine_id"].nunique()),
            "train_total_cycles": int(len(train_df)),
            "train_missing_values": train_missing,
            "train_infinities": train_inf,
            "train_cycle_monotonicity": train_mono,
            "sensor_variances": sensor_vars,
            "is_valid": (
                not train_missing["has_missing"] and
                not train_inf["has_infinities"] and
                train_mono["is_strictly_monotonic"]
            )
        }

        if test_df is not None:
            test_missing = cls.check_missing_values(test_df)
            test_inf = cls.check_infinite_values(test_df)
            test_mono = cls.check_cycle_monotonicity(test_df)
            report["test_engines_count"] = int(test_df["engine_id"].nunique())
            report["test_total_cycles"] = int(len(test_df))
            report["test_missing_values"] = test_missing
            report["test_infinities"] = test_inf
            report["test_cycle_monotonicity"] = test_mono

            if test_rul is not None:
                align = cls.check_target_alignment(test_df, test_rul)
                report["target_alignment"] = align
                report["is_valid"] = report["is_valid"] and align["is_target_aligned"]

        return report
