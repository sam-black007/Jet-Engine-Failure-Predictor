"""JETWISE AI — Phase 17 Automated Comprehensive Test Suite.

Validates:
1. Plain-Language & Glossary mapping for Simple vs Advanced modes.
2. 14 Standardized Scenario Presets Library (bounds, parameters, conversions, custom saving).
3. Live 17-Stage Simulation Pipeline tracking and state representation.
4. SQLite Simulation History Manager (reproducibility tracking, schema, query/filter).
5. Multi-Format Aerospace Engineering Report Exporter (PDF, MATLAB .mat binary, CSV, JSON).
"""

import os
import json
import tempfile
import pathlib
import pytest
import numpy as np
import pandas as pd

from src.ui.contextual_help import (
    PLAIN_LANGUAGE_MAP,
    AEROSPACE_GLOSSARY,
    translate_term,
    get_term_definition
)
from src.ui.scenario_presets_library import (
    ScenarioPresetTemplate,
    ScenarioPresetsManager,
    STANDARD_PRESETS
)
from src.ui.live_simulation_monitor import (
    SimulationStage,
    SimulationExecutionContext,
    LiveSimulationMonitor
)
from src.simulation_history_manager import SimulationHistoryManager
from src.copilot_pdf_report_generator import AerospaceReportPackageExporter
from src.copilot_scenarios import (
    EngineeringScenario,
    ScenarioResult,
    SimulationMode,
    VerificationStatus,
    ComponentVisualState
)
from src.copilot_scorecard import ScorecardCalculator


# -----------------------------------------------------------------------------
# 1. Plain-Language & Glossary Tests
# -----------------------------------------------------------------------------
def test_plain_language_mapping():
    assert len(PLAIN_LANGUAGE_MAP) >= 8
    assert "rul_prediction" in PLAIN_LANGUAGE_MAP
    assert PLAIN_LANGUAGE_MAP["rul_prediction"]["simple"] == "Estimated Remaining Engine Life"

    # Test translate function
    assert translate_term("rul_prediction", "simple") == "Estimated Remaining Engine Life"
    assert translate_term("rul_prediction", "advanced") == "RUL Quantile Regression Prediction (Pinball q10-q90)"
    assert translate_term("unknown_term", "simple") == "unknown_term"


def test_aerospace_glossary():
    assert len(AEROSPACE_GLOSSARY) >= 8
    egt_def = get_term_definition("EGT")
    assert egt_def is not None
    assert "Exhaust Gas Temperature" in egt_def["term"]

    rul_def = get_term_definition("RUL")
    assert rul_def is not None
    assert "Remaining Useful Life" in rul_def["term"]


# -----------------------------------------------------------------------------
# 2. Scenario Presets Library Tests (14 Presets)
# -----------------------------------------------------------------------------
def test_presets_catalog_completeness():
    manager = ScenarioPresetsManager()
    presets = manager.list_all_presets()
    assert len(presets) >= 14

    preset_ids = [p.preset_id for p in presets]
    expected_ids = [
        "PRESET_01_HEALTHY_BASELINE",
        "PRESET_02_INCREASED_FUEL_FLOW",
        "PRESET_03_INCREASED_TURBINE_TEMP",
        "PRESET_04_COMPRESSOR_EFFICIENCY_DEGRADATION",
        "PRESET_05_TURBINE_DEGRADATION",
        "PRESET_06_FAN_DEGRADATION",
        "PRESET_07_PRESSURE_SENSOR_BIAS",
        "PRESET_08_TEMPERATURE_SENSOR_BIAS",
        "PRESET_09_SENSOR_DROPOUT",
        "PRESET_10_GRADUAL_SENSOR_DRIFT",
        "PRESET_11_MULTIPLE_SENSOR_FAULTS",
        "PRESET_12_HIGH_AMBIENT_TEMPERATURE",
        "PRESET_13_OPERATING_CONDITION_CHANGES",
        "PRESET_14_COMBINED_ENGINE_DEGRADATION"
    ]
    for eid in expected_ids:
        assert eid in preset_ids, f"Missing required scenario preset: {eid}"


def test_preset_conversion_and_validation():
    manager = ScenarioPresetsManager()
    p2 = manager.get_preset_by_id("PRESET_02_INCREASED_FUEL_FLOW")
    assert p2 is not None
    assert p2.category == "Operational Tuning"
    assert "Combustion Chamber" in p2.affected_components

    # Test conversion to EngineeringScenario
    scen = manager.to_engineering_scenario(p2, engine_id=5, dataset="FD001")
    assert scen.engine_id == 5
    assert scen.dataset_id == "FD001"
    assert len(scen.parameter_changes) >= 1


def test_custom_preset_saving():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        custom_path = tf.name
        tf.close()

    try:
        manager = ScenarioPresetsManager(custom_storage_path=pathlib.Path(custom_path))
        custom_preset = ScenarioPresetTemplate(
            preset_id="CUSTOM_TEST_SCENARIO_99",
            name="Custom High-Altitude Test",
            category="Operational Tuning",
            description="Testing custom presets persistence.",
            affected_components=["Combustion Chamber"],
            parameters={"fuel_flow_change_pct": 8.0, "duration_cycles": 75},
            parameter_units={"fuel_flow_change_pct": "%", "duration_cycles": "Cycles"},
            parameter_ranges={"fuel_flow_change_pct": {"min": 1.0, "max": 15.0}, "duration_cycles": {"min": 10, "max": 200}},
            supported_modes=["MODE_B_PHYSICAL_PARAMETRIC", "MODE_C_MATLAB_SIMULINK"],
            expected_outputs="Elevated EGT and RUL variation.",
            validation_criteria=["Combustor energy balance verified"]
        )
        saved = manager.save_custom_preset(custom_preset)
        assert saved is True

        # Re-read
        manager_new = ScenarioPresetsManager(custom_storage_path=pathlib.Path(custom_path))
        p = manager_new.get_preset_by_id("CUSTOM_TEST_SCENARIO_99")
        assert p is not None
        assert p.name == "Custom High-Altitude Test"
    finally:
        if os.path.exists(custom_path):
            os.remove(custom_path)


# -----------------------------------------------------------------------------
# 3. Live 17-Stage Simulation Monitor Tests
# -----------------------------------------------------------------------------
def test_live_simulation_stages():
    # Verify 17 distinct stages exist in SimulationStage
    stages = list(SimulationStage)
    assert len(stages) == 17
    stage_values = [s.value for s in stages]
    assert "QUEUED" in stage_values
    assert "INITIALISING" in stage_values
    assert "VALIDATING_PARAMETERS" in stage_values
    assert "CONNECTING_TO_MATLAB" in stage_values
    assert "RUNNING_SIMULINK" in stage_values
    assert "EXECUTING_FALLBACK_PHYSICS" in stage_values
    assert "RUNNING_AI_INFERENCE" in stage_values
    assert "UPDATING_DIGITAL_TWIN" in stage_values
    assert "GENERATING_REPORT" in stage_values
    assert "COMPLETED" in stage_values
    assert "CANCELLED" in stage_values

    # Test context initialization
    ctx = SimulationExecutionContext(
        simulation_id="SIM-TEST-001",
        scenario_name="Thermal Spallation",
        engine_id=3,
        simulation_backend="NASA Brayton Fallback"
    )
    assert ctx.current_stage == SimulationStage.QUEUED
    assert ctx.is_paused is False
    assert ctx.is_cancelled is False


# -----------------------------------------------------------------------------
# 4. SQLite Simulation History Manager Tests
# -----------------------------------------------------------------------------
def test_simulation_history_persistence():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name
        tf.close()

    try:
        mgr = SimulationHistoryManager(db_path=pathlib.Path(db_path))

        scen = EngineeringScenario(
            scenario_name="Preset Verification Test",
            user_query="Testing SQLite history manager",
            simulation_mode=SimulationMode.MODE_B_DYNAMIC_ENGINE_WHAT_IF,
            engine_id=7,
            dataset="FD001",
            duration_cycles=80
        )
        res = ScenarioResult(
            scenario_id="TEST-SCEN-HIST-1",
            scenario_name=scen.scenario_name,
            simulation_mode=scen.simulation_mode,
            baseline_rul=125.0,
            modified_rul=88.5,
            rul_delta=-36.5,
            baseline_uncertainty={"std_dev": 4.0, "q10": 115.0, "q90": 135.0},
            modified_uncertainty={"std_dev": 5.2, "q10": 78.0, "q90": 98.0},
            baseline_telemetry={"sensor_4": 1400.0},
            modified_telemetry={"cycle": [1, 80], "sensor_4": [1400.0, 1435.0]},
            affected_components=["High-Pressure Turbine"],
            component_states={"High-Pressure Turbine": ComponentVisualState.WARNING},
            verification_status=VerificationStatus.PASS,
            verification_checks=[{"name": "Conservation", "status": "PASS", "details": "Verified"}],
            engineering_explanation="Verified historical persistence.",
            simulation_source="Parametric Degradation Simulator"
        )

        sim_id = mgr.record_simulation(
            scenario=scen,
            result=res,
            original_prompt="Simulate turbine degradation",
            matlab_info={"version": "R2026b / Virtual", "simulink_version": "26.2"}
        )
        assert sim_id is not None
        assert sim_id.startswith("SIM-")

        # Query back
        runs = mgr.list_simulations(engine_id=7)
        assert len(runs) == 1
        record = runs[0]
        assert record["engine_id"] == 7
        assert record["scenario_name"] == "Preset Verification Test"
        assert record["predicted_rul"] == 88.5
        assert record["rul_delta"] == -36.5

        # Query by ID
        single_record = mgr.get_simulation_by_id(sim_id)
        assert single_record is not None
        assert single_record["simulation_id"] == sim_id
    finally:
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass


# -----------------------------------------------------------------------------
# 5. Multi-Format Report Exporter Tests (PDF, MAT, CSV, JSON)
# -----------------------------------------------------------------------------
def test_export_package_generation():
    scen = EngineeringScenario(
        scenario_name="HPT Thermal Spallation Verification",
        user_query="Phase 17 Verification Test",
        simulation_mode=SimulationMode.MODE_B_DYNAMIC_ENGINE_WHAT_IF,
        engine_id=2,
        dataset="FD001",
        duration_cycles=60
    )
    res = ScenarioResult(
        scenario_id="TEST-SCEN-EXP-1",
        scenario_name=scen.scenario_name,
        simulation_mode=scen.simulation_mode,
        baseline_rul=125.0,
        modified_rul=75.4,
        rul_delta=-49.6,
        baseline_uncertainty={"std_dev": 4.0, "q10": 115.0, "q90": 135.0},
        modified_uncertainty={"std_dev": 6.0, "q10": 65.0, "q90": 85.0},
        baseline_telemetry={"sensor_4": 1400.0, "sensor_2": 642.0},
        modified_telemetry={"cycle": [1, 60], "sensor_4": [1400.0, 1445.0], "sensor_2": [642.0, 643.5]},
        affected_components=["High-Pressure Turbine"],
        component_states={"High-Pressure Turbine": ComponentVisualState.CRITICAL},
        verification_status=VerificationStatus.PASS,
        verification_checks=[{"name": "Thermodynamics", "status": "PASS", "details": "Within envelope"}],
        engineering_explanation="Significant EGT thermal rise observed.",
        simulation_source="Aerothermal Physics Engine",
        matlab_status="ONLINE"
    )

    exporter = AerospaceReportPackageExporter(
        scenario=scen,
        result=res,
        original_prompt="Increase turbine temperature by 40K"
    )

    # 1. Verify JSON Scenario
    json_out = exporter.generate_json_scenario()
    assert json_out is not None
    data = json.loads(json_out)
    assert data["scenario"]["name"] == "HPT Thermal Spallation Verification"
    assert data["scorecard"]["predicted_rul"] == 75.4

    # 2. Verify CSV Telemetry
    csv_out = exporter.generate_csv_telemetry()
    assert csv_out is not None
    lines = csv_out.strip().split("\n")
    assert len(lines) >= 2
    assert "cycle" in lines[0]

    # 3. Verify MATLAB .mat binary workspace export
    mat_bytes = exporter.generate_mat_workspace()
    assert mat_bytes is not None
    assert len(mat_bytes) > 50

    # 4. Verify PDF report bytes
    pdf_bytes = exporter.generate_pdf_report()
    assert pdf_bytes is not None
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF")
