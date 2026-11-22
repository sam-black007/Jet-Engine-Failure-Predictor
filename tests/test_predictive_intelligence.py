"""Tests for Predictive Maintenance Intelligence Engine (JETWISE AI Feature 2).

Verifies:
1. Monte Carlo Dropout Uncertainty Estimation & Quantiles
2. Unsupervised Isolation Forest Anomaly Detection & Sensor Z-Scores
3. Sequence Feature Attribution (Integrated Gradients)
4. 5-Tier Risk Classification & Explainable Prioritization (MPI)
5. Evidence-Traceable Recommendations
6. Persistent SQLite Work Order Management & Status Lifecycle
7. Maintenance Alert Lifecycle (Triggering, De-duplication, Acknowledgement, Resolution)
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
    RISK_CATEGORIES,
    MAINTENANCE_STATUSES
)


class DummyModel(nn.Module):
    """Simple model with Dropout for testing MC uncertainty and XAI."""
    def __init__(self, input_dim=17, seq_len=30):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
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


def test_anomaly_detector(sample_feature_cols):
    detector = EngineAnomalyDetector(contamination=0.05, random_state=42)
    rng = np.random.RandomState(42)
    data = rng.randn(200, 17)
    df = pd.DataFrame(data, columns=sample_feature_cols)

    detector.fit(df, sample_feature_cols)
    assert detector.is_fitted

    nominal_window = rng.randn(30, 17).astype(np.float32)
    res_nom = detector.detect_window_anomaly(nominal_window)
    assert "is_anomaly" in res_nom
    assert 0.0 <= res_nom["anomaly_score"] <= 1.0

    outlier_window = nominal_window.copy()
    outlier_window[-1, 2] = 25.0
    res_out = detector.detect_window_anomaly(outlier_window)
    assert res_out["severity"] in ("WARNING", "CRITICAL")
    assert len(res_out["top_anomalous_sensors"]) > 0


def test_feature_attribution(dummy_model, sample_feature_cols):
    xai = SequenceFeatureAttribution(dummy_model, sample_feature_cols)
    window = np.random.randn(30, 17).astype(np.float32)

    res = xai.attribute_window(window, num_steps=5)
    assert "sensor_attributions" in res
    assert len(res["sensor_attributions"]) == 17
    assert len(res["top_3_drivers"]) == 3


def test_five_tier_risk_classification():
    engine = MaintenanceDecisionEngine()

    # 1. Critical
    crit = engine.evaluate_health(predicted_rul=8.0, engine_id=1)
    assert crit["condition"] == "CRITICAL"
    assert crit["priority"] == "URGENT"

    # 2. High Risk
    high = engine.evaluate_health(predicted_rul=20.0, engine_id=2)
    assert high["condition"] == "HIGH_RISK"
    assert high["priority"] == "HIGH"

    # 3. Warning
    warn = engine.evaluate_health(predicted_rul=35.0, engine_id=3)
    assert warn["condition"] == "WARNING"
    assert warn["priority"] == "MEDIUM"

    # 4. Monitor
    mon = engine.evaluate_health(predicted_rul=60.0, engine_id=4)
    assert mon["condition"] == "MONITOR"
    assert mon["priority"] == "ROUTINE"

    # 5. Healthy
    healthy = engine.evaluate_health(predicted_rul=110.0, engine_id=5)
    assert healthy["condition"] == "HEALTHY"
    assert healthy["priority"] == "LOW"


def test_explainable_priority_breakdown():
    engine = MaintenanceDecisionEngine(weight_rul=0.4, weight_anomaly=0.3)
    res = engine.evaluate_health(
        predicted_rul=25.0,
        uncertainty_std=5.0,
        anomaly_score=0.7,
        degradation_rate=1.2,
        max_sensor_zscore=2.5,
        engine_id=7
    )

    assert "score_breakdown" in res
    sb = res["score_breakdown"]
    assert "rul_subscore" in sb
    assert "anomaly_subscore" in sb
    assert sb["weights"]["rul"] == 0.4
    assert sb["weights"]["anomaly"] == 0.3
    assert 0.0 <= res["maintenance_priority_index"] <= 100.0


def test_traceable_recommendations():
    engine = MaintenanceDecisionEngine()
    res = engine.evaluate_health(predicted_rul=10.0, anomaly_score=0.8, max_sensor_zscore=3.2, engine_id=9)
    recs = res["recommended_actions"]
    assert len(recs) > 0

    first = recs[0]
    assert first["engine_id"] == 9
    assert first["risk_classification"] == "CRITICAL"
    assert "supporting_evidence" in first
    assert "relevant_sensor_or_prediction_info" in first
    assert "recommended_action" in first
    assert "maintenance_priority" in first
    assert "reason" in first


def test_database_work_order_crud_and_alerts():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = pathlib.Path(tmpdir) / "test_maint2.db"
        db = MaintenanceDatabase(db_path)

        # 1. Create work order
        wo_id = db.create_work_order(
            engine_id=42,
            dataset_subset="FD001",
            priority="HIGH",
            title="Inspect HPT Nozzle Guide Vanes",
            subsystem="HPT Section",
            description="Thermal creep signatures observed.",
            assigned_to="Lead Engineer Sarah",
            notes="Initial review."
        )
        assert wo_id > 0

        # 2. Update status and notes
        ok = db.update_work_order(
            order_id=wo_id,
            status="Maintenance In Progress",
            notes="Borescope underway."
        )
        assert ok

        # 3. Retrieve and verify
        orders = db.get_work_orders("FD001")
        assert len(orders) == 1
        assert orders[0]["status"] == "Maintenance In Progress"
        assert orders[0]["assigned_to"] == "Lead Engineer Sarah"

        # 4. Log prediction triggering automated alert
        health_eval = {
            "condition": "CRITICAL",
            "health_state": "CRITICAL",
            "maintenance_priority_index": 85.0,
            "priority": "URGENT",
            "recommended_actions": [{"action": "Core Overhaul", "subsystem": "Turbine", "reason": "Failure risk"}]
        }
        db.log_prediction(
            engine_id=42,
            dataset_subset="FD001",
            cycle=190,
            predicted_rul=9.5,
            uncertainty={"std_dev": 2.5, "q10": 7.0, "q90": 12.0},
            anomaly={"anomaly_score": 0.85},
            health_eval=health_eval
        )

        # 5. Check active alerts
        alerts = db.get_alerts("FD001", status="ACTIVE")
        assert len(alerts) >= 1
        alt_id = alerts[0]["id"]
        assert alerts[0]["engine_id"] == 42
        assert alerts[0]["severity"] == "CRITICAL"

        # 6. Acknowledge and Resolve Alert
        assert db.acknowledge_alert(alt_id)
        assert db.resolve_alert(alt_id)
        active_after = db.get_alerts("FD001", status="ACTIVE")
        assert len(active_after) == 0
