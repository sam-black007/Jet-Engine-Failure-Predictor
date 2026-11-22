"""THRUSTLAB — Comprehensive Test Suite for Conversational Aerospace Engineering Copilot.

Verifies all 12 required conversational engineering workflows:
1. Increase turbine temperature by 5%
2. Reduce compressor pressure by 3%
3. Increase fuel flow by 5%
4. Simulate gradual fan degradation over 100 cycles
5. Introduce a sensor bias
6. Introduce multiple abnormalities (compound scenario)
7. Compare healthy and degraded scenarios
8. Run a MATLAB/Simulink scenario (probes status and validates physics fallback)
9. Compare AI predictions with physical simulation results
10. Reopen a previous scenario and verify state restoration
11. Request an explanation of an abnormal reading (autonomous investigation)
12. Handle invalid or physically inconsistent instructions safely
Also verifies mathematical reproducibility and deterministic execution.
"""

import pytest
import numpy as np
import pandas as pd

from src.copilot_nlu import AerospaceCopilotNLU, ConversationalIntent
from src.copilot_scenarios import (
    EngineeringScenario,
    ScenarioParameterChange,
    SimulationMode,
    ComponentVisualState,
    VerificationStatus,
    save_scenario_to_db,
    list_saved_scenarios
)
from src.copilot_simulation import CopilotSimulationEngine
from src.copilot_investigation import AutonomousInvestigationEngine, ScenarioComparisonEngine
from src.matlab_simulink_bridge import MATLABSimulinkBridge, AerothermalPhysicsSimulator
from src.dataset_registry import DatasetRegistry


@pytest.fixture(scope="module")
def copilot_setup():
    """Initializes shared test fixtures for the conversational copilot."""
    nlu = AerospaceCopilotNLU()
    engine = CopilotSimulationEngine()
    inv = AutonomousInvestigationEngine()
    adapter = DatasetRegistry.get_adapter("FD001")
    trajectory = adapter.get_engine_trajectory(1, kind="train")
    return {
        "nlu": nlu,
        "engine": engine,
        "inv": inv,
        "trajectory": trajectory
    }


# ---------------------------------------------------------------------------
# Test 1: Increase turbine temperature by 5%
# ---------------------------------------------------------------------------
def test_workflow_01_increase_turbine_temp(copilot_setup):
    nlu = copilot_setup["nlu"]
    engine = copilot_setup["engine"]

    query = "Increase turbine temperature by 5%"
    intent, scen, _ = nlu.parse_query(query)

    assert intent == ConversationalIntent.SIMULATE_SCENARIO
    assert len(scen.parameters) == 1
    p = scen.parameters[0]
    assert p.parameter_key == "t50"
    assert p.change_value == 5.0
    assert p.target_subsystem == "hpt"

    res = engine.execute_scenario(scen)
    assert res.verification_status in (VerificationStatus.PASS, VerificationStatus.WARNING)
    assert "hpt" in res.affected_components
    assert res.modified_rul >= 0.0
    assert res.component_states.get("hpt") in (ComponentVisualState.WARNING.value, ComponentVisualState.MODIFIED.value, ComponentVisualState.CRITICAL.value)


# ---------------------------------------------------------------------------
# Test 2: Reduce compressor pressure by 3%
# ---------------------------------------------------------------------------
def test_workflow_02_reduce_compressor_pressure(copilot_setup):
    nlu = copilot_setup["nlu"]
    engine = copilot_setup["engine"]

    query = "Reduce compressor pressure by 3%"
    intent, scen, _ = nlu.parse_query(query)

    assert len(scen.parameters) == 1
    p = scen.parameters[0]
    assert p.parameter_key == "p30"
    assert p.change_value == -3.0
    assert p.target_subsystem == "hpc"

    res = engine.execute_scenario(scen)
    assert "hpc" in res.affected_components
    assert res.verification_status == VerificationStatus.PASS


# ---------------------------------------------------------------------------
# Test 3: Increase fuel flow by 5%
# ---------------------------------------------------------------------------
def test_workflow_03_increase_fuel_flow(copilot_setup):
    nlu = copilot_setup["nlu"]
    engine = copilot_setup["engine"]

    query = "Increase fuel flow by 5% and check how the engine responds"
    intent, scen, _ = nlu.parse_query(query)

    assert len(scen.parameters) == 1
    p = scen.parameters[0]
    assert p.parameter_key == "fuel_flow"
    assert p.change_value == 5.0
    assert p.target_subsystem == "combustor"

    res = engine.execute_scenario(scen)
    assert "combustor" in res.affected_components


# ---------------------------------------------------------------------------
# Test 4: Simulate gradual fan degradation over 100 cycles
# ---------------------------------------------------------------------------
def test_workflow_04_gradual_fan_degradation(copilot_setup):
    nlu = copilot_setup["nlu"]
    engine = copilot_setup["engine"]

    query = "Simulate gradual fan degradation over 100 operating cycles"
    intent, scen, _ = nlu.parse_query(query)

    assert scen.simulation_mode == SimulationMode.MODE_B_DYNAMIC_ENGINE_WHAT_IF
    assert scen.duration_cycles == 100
    assert len(scen.parameters) == 1
    assert scen.parameters[0].parameter_key == "fan_degradation"
    assert scen.parameters[0].target_subsystem == "fan_assembly"

    res = engine.execute_scenario(scen)
    assert res.time_series_data is not None
    assert len(res.time_series_data["cycle"]) == 100
    # Final RUL should be lower than initial RUL
    assert res.modified_rul <= res.baseline_rul


# ---------------------------------------------------------------------------
# Test 5: Introduce a sensor bias
# ---------------------------------------------------------------------------
def test_workflow_05_sensor_bias(copilot_setup):
    nlu = copilot_setup["nlu"]
    engine = copilot_setup["engine"]

    query = "Introduce a pressure sensor bias"
    intent, scen, _ = nlu.parse_query(query)

    assert len(scen.parameters) >= 1
    p = scen.parameters[0]
    assert p.change_type == "bias"
    assert p.parameter_key == "p30"

    res = engine.execute_scenario(scen)
    assert res.verification_status == VerificationStatus.PASS


# ---------------------------------------------------------------------------
# Test 6: Introduce multiple abnormalities (Compound scenario)
# ---------------------------------------------------------------------------
def test_workflow_06_multiple_abnormalities(copilot_setup):
    nlu = copilot_setup["nlu"]
    engine = copilot_setup["engine"]

    query = "Increase turbine temperature by 5% and reduce compressor pressure by 3%"
    intent, scen, _ = nlu.parse_query(query)

    assert len(scen.parameters) == 2
    keys = {p.parameter_key for p in scen.parameters}
    assert "t50" in keys
    assert "p30" in keys

    res = engine.execute_scenario(scen)
    assert "hpt" in res.affected_components
    assert "hpc" in res.affected_components
    assert len(res.component_states) >= 2


# ---------------------------------------------------------------------------
# Test 7: Compare healthy and degraded scenarios
# ---------------------------------------------------------------------------
def test_workflow_07_compare_healthy_and_degraded(copilot_setup):
    nlu = copilot_setup["nlu"]
    engine = copilot_setup["engine"]

    # Baseline healthy
    _, scen_healthy, _ = nlu.parse_query("Create a healthy baseline engine")
    res_healthy = engine.execute_scenario(scen_healthy)

    # Degraded
    _, scen_deg, _ = nlu.parse_query("Simulate a compressor degradation scenario")
    res_deg = engine.execute_scenario(scen_deg)

    # Compare
    comp = ScenarioComparisonEngine.compare_scenarios([res_healthy, res_deg])
    assert comp["num_scenarios"] == 2
    assert "comparison_matrix" in comp
    assert len(comp["comparison_matrix"]) == 2
    assert comp["best_scenario"] == res_healthy.scenario_name


# ---------------------------------------------------------------------------
# Test 8: Run a MATLAB/Simulink scenario
# ---------------------------------------------------------------------------
def test_workflow_08_matlab_simulink_scenario(copilot_setup):
    nlu = copilot_setup["nlu"]
    engine = copilot_setup["engine"]

    query = "Run the same condition in MATLAB Simulink and compare the results"
    intent, scen, _ = nlu.parse_query(query)

    assert scen.simulation_mode == SimulationMode.MODE_C_MATLAB_SIMULINK_PHYSICS
    res = engine.execute_scenario(scen)

    # Must explicitly state MATLAB status without faking
    bridge = MATLABSimulinkBridge()
    _valid_installed_statuses = (
        "ONLINE",               # MATLAB started and confirmed in time
        "PHYSICS_ENGINE_ACTIVE",  # MATLAB found but timed out/failed — physics engine active
        "ERROR",                # legacy — kept for backwards compat
        "EMULATED",             # legacy — kept for backwards compat
    )
    if bridge.is_matlab_installed:
        assert res.matlab_status in _valid_installed_statuses, (
            f"Unexpected matlab_status '{res.matlab_status}' when MATLAB is installed. "
            f"Expected one of {_valid_installed_statuses}"
        )
    else:
        assert res.matlab_status == "OFFLINE_NOT_INSTALLED"
    assert "Aerothermal" in res.simulation_source or "MATLAB" in res.simulation_source
    assert res.verification_status == VerificationStatus.PASS


# ---------------------------------------------------------------------------
# Test 9: Compare AI predictions with simulation results
# ---------------------------------------------------------------------------
def test_workflow_09_compare_ai_with_simulation(copilot_setup):
    engine = copilot_setup["engine"]

    # Run Mode A (pure AI feature test)
    scen_a = EngineeringScenario(
        scenario_name="AI Sensitivity",
        simulation_mode=SimulationMode.MODE_A_AI_SENSOR_WHAT_IF,
        parameters=[ScenarioParameterChange(
            parameter_key="t50", target_subsystem="hpt", change_type="percentage", change_value=5.0
        )]
    )
    res_a = engine.execute_scenario(scen_a)

    # Run Mode C (Physics balance)
    scen_c = EngineeringScenario(
        scenario_name="Physics Simulation",
        simulation_mode=SimulationMode.MODE_C_MATLAB_SIMULINK_PHYSICS
    )
    res_c = engine.execute_scenario(scen_c)

    assert res_a.modified_rul is not None
    assert res_c.modified_rul is not None
    assert res_a.simulation_source != res_c.simulation_source


# ---------------------------------------------------------------------------
# Test 10: Reopen a previous scenario & Database persistence
# ---------------------------------------------------------------------------
def test_workflow_10_reopen_previous_scenario(copilot_setup):
    engine = copilot_setup["engine"]

    test_scen = EngineeringScenario(
        scenario_name="Persisted Test Scenario",
        user_query="Increase turbine temperature by 4%",
        parameters=[ScenarioParameterChange(
            parameter_key="t50", target_subsystem="hpt", change_type="percentage", change_value=4.0
        )]
    )
    res = engine.execute_scenario(test_scen)
    save_scenario_to_db(test_scen, res)

    saved_list = list_saved_scenarios(limit=10)
    assert len(saved_list) >= 1
    found = any(s["scenario_id"] == test_scen.scenario_id for s in saved_list)
    assert found, f"Scenario {test_scen.scenario_id} should be persisted in database"


# ---------------------------------------------------------------------------
# Test 11: Request explanation of an abnormal reading (Autonomous Investigation)
# ---------------------------------------------------------------------------
def test_workflow_11_investigate_abnormal_reading(copilot_setup):
    inv = copilot_setup["inv"]
    trajectory = copilot_setup["trajectory"]

    diag = inv.investigate_anomaly(trajectory, target_symptom="abnormal temperature readings")
    assert diag["status"] == "DIAGNOSIS_COMPLETE"
    assert len(diag["hypotheses"]) >= 1
    top = diag["hypotheses"][0]
    assert "confidence_score" in top
    assert top["confidence_score"] > 0.50
    assert "recommended_action" in top
    assert "parameter_to_adjust" in top


# ---------------------------------------------------------------------------
# Test 12: Invalid or physically inconsistent instructions handled safely
# ---------------------------------------------------------------------------
def test_workflow_12_invalid_or_extreme_instruction(copilot_setup):
    nlu = copilot_setup["nlu"]
    engine = copilot_setup["engine"]

    # Extreme +150% temperature instruction (physically invalid)
    query = "Increase turbine temperature by 150%"
    _, scen, _ = nlu.parse_query(query)

    res = engine.execute_scenario(scen)
    # Range check should flag warning
    assert res.verification_status in (VerificationStatus.WARNING, VerificationStatus.FAIL)
    range_warnings = [c for c in res.verification_checks if c["status"] in ("WARNING", "FAIL")]
    assert len(range_warnings) >= 1


# ---------------------------------------------------------------------------
# Test 13: Deterministic Reproducibility
# ---------------------------------------------------------------------------
def test_deterministic_reproducibility(copilot_setup):
    engine = copilot_setup["engine"]

    scen = EngineeringScenario(
        scenario_name="Deterministic Check",
        parameters=[ScenarioParameterChange(
            parameter_key="p30", target_subsystem="hpc", change_type="percentage", change_value=-3.0
        )]
    )
    res1 = engine.execute_scenario(scen)
    res2 = engine.execute_scenario(scen)

    assert res1.modified_rul == res2.modified_rul
    assert res1.rul_delta == res2.rul_delta
    assert res1.affected_components == res2.affected_components
