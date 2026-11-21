"""Unit Tests for EngineReplaySimulator."""

import pytest
import numpy as np
from src.data_replay import EngineReplaySimulator


def test_replay_simulator_load_and_snapshot():
    sim = EngineReplaySimulator(dataset_name="FD001")
    info = sim.load_engine(engine_id=1, kind="train")
    assert info["engine_id"] == 1
    assert info["total_cycles"] > 30

    snapshot = sim.get_cycle_snapshot(cycle=1)
    assert snapshot["cycle"] == 1
    assert "telemetry" in snapshot
    assert "sensor_2" in snapshot["telemetry"]


def test_replay_simulator_sliding_window_padding():
    sim = EngineReplaySimulator(dataset_name="FD001")
    sim.load_engine(engine_id=1, kind="train")
    n_features = len(sim.adapter.get_active_sensors())

    # Cycle 5 < window_size 30: should pad to (30, active_sensors)
    window_early = sim.get_sliding_window(cycle=5, window_size=30)
    assert window_early.shape == (30, n_features)

    # Cycle 35 >= window_size 30: should extract final 30 rows
    window_late = sim.get_sliding_window(cycle=35, window_size=30)
    assert window_late.shape == (30, n_features)


def test_replay_simulator_risk_classification():
    risk_h, col_h = EngineReplaySimulator.classify_risk(65.0)
    assert risk_h == "HEALTHY"

    risk_m, col_m = EngineReplaySimulator.classify_risk(40.0)
    assert risk_m == "MONITOR"

    risk_w, col_w = EngineReplaySimulator.classify_risk(18.0)
    assert risk_w == "WARNING"

    risk_c, col_c = EngineReplaySimulator.classify_risk(8.0)
    assert risk_c == "CRITICAL"


def test_replay_simulator_stepping_and_reset():
    sim = EngineReplaySimulator(dataset_name="FD001")
    sim.load_engine(engine_id=1, kind="train")

    s1 = sim.step()
    assert s1["cycle"] == 1

    s2 = sim.step()
    assert s2["cycle"] == 2

    sim.record_prediction(cycle=2, predicted_rul=45.0)
    assert len(sim.prediction_history) == 1
    assert sim.prediction_history[0]["risk_level"] == "MONITOR"

    sim.reset()
    assert sim.current_cycle == 1
    assert len(sim.prediction_history) == 0
