"""Unit and Integration Tests for Phase 11 Advanced Aerospace Intelligence Modules."""

import pytest
import numpy as np
import pandas as pd
import pathlib

from src.fdir_engine import SensorFDIREngine
from src.physics_engine import PhysicsInformedEngine
from src.model_trust import ModelTrustEngine
from src.uncertainty_calibration import ConformalCalibrator
from src.maintenance_scheduler import MaintenanceSchedulingOptimizer
from src.fleet_simulator import FleetDigitalTwinSimulator
from src.model_monitor import ModelHealthMonitor
from src.engineering_review import EngineeringReviewEngine
from src.database_migrations import apply_phase11_12_migrations


@pytest.fixture(scope="module")
def setup_test_db(tmp_path_factory):
    db_file = tmp_path_factory.mktemp("db") / "test_maint.db"
    apply_phase11_12_migrations(db_file)
    return db_file


def test_sensor_fdir_stuck_at_and_drift(setup_test_db):
    """Verify FDIR identifies stuck-at sensors and linear drift."""
    fdir = SensorFDIREngine(stuck_var_threshold=1e-4, drift_slope_threshold=0.05, db_path=setup_test_db)
    
    # 30 cycles of dummy data
    cycles = 30
    df = pd.DataFrame({
        "sensor_2": np.linspace(642.0, 642.5, cycles),  # Nominal
        "sensor_3": np.full(cycles, 1585.0),            # Stuck-at (var = 0)
        "sensor_4": np.linspace(1400.0, 1430.0, cycles), # Drifting significantly
        "sensor_7": np.random.normal(553.0, 0.2, cycles) # Nominal
    })

    res = fdir.analyze_sensor_window(
        engine_id=1,
        dataset_subset="FD001",
        current_cycle=30,
        telemetry_window_df=df
    )

    assert "sensor_status_matrix" in res
    fault_types = [f["fault_type"] for f in res["fault_reports"]]
    assert "STUCK_AT" in fault_types
    assert "DRIFT" in fault_types
    assert res["overall_sensor_health_index"] < 1.0


def test_physics_engine_aerothermodynamics():
    """Verify thermodynamic equations, compressor polytropic efficiency, and virtual sensors."""
    physics = PhysicsInformedEngine(nominal_hpc_eta=0.89)

    sensor_dict = {
        "sensor_1": 518.67,  # T2
        "sensor_2": 642.0,   # T24
        "sensor_3": 1590.0,  # T30
        "sensor_4": 1400.0,  # T50
        "sensor_5": 14.69,   # P2
        "sensor_6": 21.61,   # P24
        "sensor_7": 553.0,   # Ps30
        "sensor_9": 1130.0,  # T48 (EGT)
        "sensor_11": 521.0,  # phi
        "sensor_12": 8.4     # BPR
    }

    eval_res = physics.evaluate_thermodynamics(sensor_dict)

    # Check compression metrics
    assert eval_res["compression"]["cpr"] > 10.0
    assert 0.70 <= eval_res["compression"]["observed_hpc_efficiency"] <= 0.99
    assert eval_res["consistency_checks"]["temperature_monotonicity_valid"] is True
    assert eval_res["consistency_checks"]["pressure_monotonicity_valid"] is True

    # Check virtual sensors
    assert eval_res["virtual_sensors"]["virtual_t4_tit_rankine"] > eval_res["compression"]["t30_actual_rankine"]
    assert eval_res["virtual_sensors"]["is_research_approximation"] is True
    assert eval_res["thermal_margins"]["egt_margin_rankine"] > 0.0


def test_model_trust_and_mahalanobis_ood(setup_test_db):
    """Verify OOD detection via Mahalanobis distance and Isolation Forest."""
    trust = ModelTrustEngine(mahalanobis_threshold_p95=10.0, db_path=setup_test_db)

    # Fit synthetic baseline training features
    np.random.seed(42)
    train_X = np.random.normal(loc=0.0, scale=1.0, size=(100, 5))
    feature_names = [f"sensor_{i}" for i in range(1, 6)]
    trust.fit_reference_distribution(train_X, feature_names, dataset_name="FD001")

    # In-domain observation
    in_vec = np.random.normal(loc=0.0, scale=1.0, size=5)
    in_res = trust.evaluate_observation_trust(1, "FD001", 50, "test_model", in_vec)
    assert in_res["status_classification"] in ("IN_DOMAIN", "DISTRIBUTION_SHIFT_DETECTED")
    assert in_res["model_trust_pct"] >= 50.0

    # Extreme OOD observation
    ood_vec = np.array([12.0, -15.0, 20.0, 18.0, -10.0])
    ood_res = trust.evaluate_observation_trust(2, "FD001", 50, "test_model", ood_vec)
    assert ood_res["status_classification"] == "OUT_OF_DISTRIBUTION"
    assert ood_res["mahalanobis_distance"] > 10.0


def test_uncertainty_conformal_calibrator(setup_test_db):
    """Verify conformal prediction intervals provide bounded empirical coverage."""
    calibrator = ConformalCalibrator(target_coverage=0.90, db_path=setup_test_db)

    np.random.seed(42)
    y_true = np.linspace(10.0, 100.0, 50)
    y_pred = y_true + np.random.normal(0, 5.0, 50)

    fit_res = calibrator.fit_calibration_residuals(y_true, y_pred, "pinball_lstm", "FD001")
    assert fit_res["empirical_coverage"] >= 0.85
    assert fit_res["conformal_quantile_margin"] > 0.0

    # Test prediction interval
    pred_int = calibrator.predict_conformal_interval(point_prediction=45.0)
    assert pred_int["conformal_lower"] < 45.0
    assert pred_int["conformal_upper"] > 45.0
    assert pred_int["interval_width"] == round(pred_int["conformal_upper"] - pred_int["conformal_lower"], 1)


def test_maintenance_scheduling_optimizer(setup_test_db):
    """Verify scheduling solver respects hangar bay capacity and technician limits."""
    scheduler = MaintenanceSchedulingOptimizer(
        max_daily_bay_capacity=2,
        technician_hours_per_day=32.0,
        db_path=setup_test_db
    )

    candidates = [
        {"engine_id": 1, "predicted_rul": 5.0, "mpi": 88.0, "required_subsystem": "HPT Section"},
        {"engine_id": 2, "predicted_rul": 15.0, "mpi": 72.0, "required_subsystem": "Combustor"},
        {"engine_id": 3, "predicted_rul": 22.0, "mpi": 65.0, "required_subsystem": "HPC Section"},
        {"engine_id": 4, "predicted_rul": 40.0, "mpi": 45.0, "required_subsystem": "Bearing Compartments"}
    ]

    res = scheduler.optimize_fleet_schedule(candidates, horizon_days=7)
    assert res["total_engines_scheduled"] >= 2
    for day_stat in res["daily_utilization"]:
        assert day_stat["bays_used"] <= 2
        assert day_stat["tech_hours_used"] <= 32.0

    # Test persistence
    sched_id = scheduler.persist_approved_schedule(res)
    assert sched_id > 0


def test_fleet_digital_twin_simulator(setup_test_db):
    """Verify fleet simulation computes degradation trajectories and counterfactual groundings."""
    sim = FleetDigitalTwinSimulator(db_path=setup_test_db)

    fleet = [
        {"engine_id": i, "initial_rul": float(10 + i * 5), "degradation_rate": 1.0, "initial_mpi": 50.0}
        for i in range(1, 11)
    ]

    res_baseline = sim.run_fleet_scenario(fleet, scenario_name="Baseline", policy="status_quo", horizon_cycles=50)
    res_cbm = sim.run_fleet_scenario(fleet, scenario_name="CBM_Policy", policy="condition_based", horizon_cycles=50)

    assert "fleet_rul_trajectory" in res_cbm
    assert len(res_cbm["fleet_rul_trajectory"]) > 0
    # CBM should have equal or fewer groundings than status quo
    assert res_cbm["scenario_unplanned_groundings"] <= res_baseline["scenario_unplanned_groundings"]


def test_model_health_monitor(setup_test_db):
    """Verify drift detection and champion-challenger formal comparison."""
    monitor = ModelHealthMonitor(db_path=setup_test_db)

    np.random.seed(42)
    b_df = pd.DataFrame({"sensor_1": np.random.normal(500, 10, 50)})
    c_df = pd.DataFrame({"sensor_1": np.random.normal(550, 10, 50)})  # Severe shift

    drift_res = monitor.evaluate_feature_drift(b_df, c_df, "pinball_lstm", "FD001")
    assert drift_res["features_evaluated"] == 1
    assert drift_res["overall_drift_status"] in ("WARNING", "CRITICAL_DRIFT")

    # Champion vs Challenger promotion test
    champ = {"NASA_score": 537.0, "MAE": 11.69}
    chall = {"NASA_score": 362.0, "MAE": 12.50}  # ~32.6% improvement
    promo_res = monitor.evaluate_champion_challenger(champ, chall)
    assert promo_res["decision"] == "PROMOTED"
    assert promo_res["nasa_score_improvement_pct"] > 30.0


def test_engineering_review_engine(setup_test_db):
    """Verify human review queue and formal review submission."""
    engine = EngineeringReviewEngine(db_path=setup_test_db)

    rev_id = engine.submit_engineering_review(
        engine_id=1,
        dataset_subset="FD001",
        cycle=150,
        predicted_rul=12.5,
        decision="BORESCOPE_REQUESTED",
        reviewer_name="Chief Inspector A. Smith",
        engineering_notes="Observed high EGT margin loss and HPC pressure oscillation.",
        physical_inspection_findings="Pending hangar inspection",
        is_validated_ground_truth=True
    )
    assert rev_id > 0

    history = engine.get_review_history(engine_id=1)
    assert len(history) >= 1
    assert history[0]["decision"] == "BORESCOPE_REQUESTED"
    assert history[0]["reviewer_name"] == "Chief Inspector A. Smith"
