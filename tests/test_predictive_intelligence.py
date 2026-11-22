"""Tests for Predictive Maintenance Intelligence Engine (JETWISE AI Phase 7).

Verifies:
1. Monte Carlo Dropout Uncertainty Estimation
2. Isolation Forest Anomaly Detection
3. Sequence Feature Attribution (XAI)
4. Maintenance Decision Engine & Priority Index
5. SQLite Database Persistence & Work Order Creation
"""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tempfile
import pathlib

from src.predictive_intelligence import (
    UncertaintyEstimator,
    EngineAnomalyDetector,
    SequenceFeatureAttribution,
    MaintenanceDecisionEngine,
    MaintenanceDatabase,
    HEALTH_STATES
)


class DummyModel(nn.Module):
    """Simple model with Dropout for testing MC uncertainty and XAI."""
    def __init__(self, input_dim=17, seq_len=30):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x_last = x[:, -1, :]
        h = torch.relu(self.fc1(x_last))
        h = self.drop(h)
        out = torch.relu(self.fc2(h)) * 100.0
        return out


@pytest.fixture
def dummy_model():
    return DummyModel(input_dim=17, seq_len=30)


@pytest.fixture
def sample_feature_cols():
    return [f"sensor_{i}" for i in range(1, 18)]


def test_uncertainty_estimator(dummy_model):
    estimator = UncertaintyEstimator(dummy_model, num_samples=15)
    sample_window = np.random.randn(30, 17).astype(np.float32)

    res = estimator.predict_with_uncertainty(sample_window)
    assert "mean_rul" in res
    assert "std_dev" in res
    assert "q10" in res
    assert "q90" in res
    assert "confidence_score" in res
    assert 0.0 <= res["confidence_score"] <= 1.0
    assert res["num_samples"] == 15


def test_anomaly_detector(sample_feature_cols):
    detector = EngineAnomalyDetector(contamination=0.05, random_state=42)

    # Synthetic baseline dataframe
    rng = np.random.RandomState(42)
    data = rng.randn(200, 17)
    df = pd.DataFrame(data, columns=sample_feature_cols)

    detector.fit(df, sample_feature_cols)
    assert detector.is_fitted

    # Test nominal window
    nominal_window = rng.randn(30, 17).astype(np.float32)
    res_nom = detector.detect_window_anomaly(nominal_window)
    assert "is_anomaly" in res_nom
    assert "anomaly_score" in res_nom
    assert 0.0 <= res_nom["anomaly_score"] <= 1.0

    # Test extreme outlier window
    outlier_window = nominal_window.copy()
    outlier_window[-1, 2] = 25.0  # extreme deviation on sensor_3
    res_out = detector.detect_window_anomaly(outlier_window)
    assert res_out["severity"] in ("WARNING", "CRITICAL")
    assert len(res_out["top_anomalous_sensors"]) > 0


def test_feature_attribution(dummy_model, sample_feature_cols):
    xai = SequenceFeatureAttribution(dummy_model, sample_feature_cols)
    window = np.random.randn(30, 17).astype(np.float32)

    res = xai.attribute_window(window, num_steps=5)
    assert "sensor_attributions" in res
    assert len(res["sensor_attributions"]) == 17
    assert "top_3_drivers" in res
    assert len(res["top_3_drivers"]) == 3
    assert "temporal_weights" in res
    assert len(res["temporal_weights"]) == 30


def test_maintenance_decision_engine():
    engine = MaintenanceDecisionEngine()

    # Critical case
    res_crit = engine.evaluate_health(predicted_rul=12.0, uncertainty_std=8.0, anomaly_score=0.85)
    assert res_crit["health_state"] == "CRITICAL"
    assert res_crit["priority"] in ("URGENT", "HIGH")
    assert res_crit["maintenance_priority_index"] >= 60.0
    assert len(res_crit["recommended_actions"]) > 0

    # Healthy case
    res_healthy = engine.evaluate_health(predicted_rul=110.0, uncertainty_std=2.0, anomaly_score=0.1)
    assert res_healthy["health_state"] == "HEALTHY"
    assert res_healthy["priority"] == "LOW"
    assert res_healthy["maintenance_priority_index"] < 40.0


def test_maintenance_database_operations():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = pathlib.Path(tmpdir) / "test_maint.db"
        db = MaintenanceDatabase(db_path)

        # Log prediction
        unc = {"mean_rul": 15.0, "std_dev": 3.0, "q10": 10.0, "q90": 20.0}
        anom = {"anomaly_score": 0.82}
        health = {
            "health_state": "CRITICAL",
            "maintenance_priority_index": 88.5,
            "priority": "URGENT",
            "recommended_actions": [{"action": "Urgent Overhaul", "subsystem": "HPT", "rationale": "High wear"}]
        }

        pred_id = db.log_prediction(
            engine_id=101,
            dataset_subset="FD001",
            cycle=185,
            predicted_rul=15.0,
            uncertainty=unc,
            anomaly=anom,
            health_eval=health
        )
        assert pred_id > 0

        # Verify summary
        summary = db.get_fleet_summary("FD001")
        assert summary["health_distribution"].get("CRITICAL", 0) == 1
        assert summary["work_orders"]["total_orders"] >= 1

        # Verify work orders
        orders = db.get_work_orders("FD001")
        assert len(orders) >= 1
        assert orders[0]["engine_id"] == 101
        assert orders[0]["priority"] == "URGENT"
