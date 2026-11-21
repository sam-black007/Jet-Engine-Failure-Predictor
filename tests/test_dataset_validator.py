"""Unit Tests for DatasetValidator."""

import numpy as np
import pandas as pd
import pytest

from src.dataset_validator import DatasetValidator


def test_validator_missing_values():
    clean_df = pd.DataFrame({"engine_id": [1, 1], "cycle": [1, 2], "sensor_1": [10.0, 11.0]})
    res = DatasetValidator.check_missing_values(clean_df)
    assert not res["has_missing"]
    assert res["total_missing_values"] == 0

    dirty_df = clean_df.copy()
    dirty_df.loc[0, "sensor_1"] = np.nan
    res_dirty = DatasetValidator.check_missing_values(dirty_df)
    assert res_dirty["has_missing"]
    assert res_dirty["total_missing_values"] == 1


def test_validator_infinities():
    clean_df = pd.DataFrame({"engine_id": [1, 1], "cycle": [1, 2], "val": [1.0, 2.0]})
    res = DatasetValidator.check_infinite_values(clean_df)
    assert not res["has_infinities"]

    dirty_df = clean_df.copy()
    dirty_df.loc[1, "val"] = np.inf
    res_dirty = DatasetValidator.check_infinite_values(dirty_df)
    assert res_dirty["has_infinities"]


def test_validator_cycle_monotonicity():
    monotonic_df = pd.DataFrame({
        "engine_id": [1, 1, 1, 2, 2],
        "cycle": [1, 2, 3, 1, 2]
    })
    res = DatasetValidator.check_cycle_monotonicity(monotonic_df)
    assert res["is_strictly_monotonic"]
    assert res["broken_engines_count"] == 0

    broken_df = pd.DataFrame({
        "engine_id": [1, 1, 1],
        "cycle": [1, 3, 4]  # Gap: missed 2
    })
    res_broken = DatasetValidator.check_cycle_monotonicity(broken_df)
    assert not res_broken["is_strictly_monotonic"]
    assert 1 in res_broken["broken_engine_ids"]


def test_validator_sensor_variances():
    df = pd.DataFrame({
        "engine_id": [1, 1, 1],
        "cycle": [1, 2, 3],
        "sensor_1": [5.0, 5.0, 5.0],      # invariant
        "sensor_2": [10.0, 15.0, 20.0]    # active
    })
    res = DatasetValidator.check_sensor_variances(df)
    assert res["invariant_sensors_count"] == 1
    assert "sensor_1" in res["invariant_sensors"]
    assert "sensor_2" in res["active_sensors"]
