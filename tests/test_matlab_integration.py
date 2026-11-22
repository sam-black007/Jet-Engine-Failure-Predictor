"""THRUSTLAB — Automated Test Suite for MATLAB & Simulink Aerospace Simulation Integration.

Tests:
1. Environment detector (probing, releases, toolboxes, offline status handling).
2. Input parameter specification, bound enforcement, and SI/Imperial unit conversions.
3. Simulation outputs schema and Brayton cycle station states.
4. Thermodynamic scientific integrity validator (First Law checks, stability, warnings).
5. Telemetry adapter & feature mapping to C-MAPSS 17-sensor space.
6. Prognostic AI inference over simulated physical telemetry.
7. SimulinkRunner execution under certified aerothermal fallback (zero fake data).
8. Baseline vs modified simulation delta comparison.
9. SQLite scenario persistence and reproducibility.
"""

import pytest
import numpy as np
import os
import json
import pathlib
import tempfile
from unittest.mock import MagicMock, patch

from src.matlab_integration.simulation_inputs import SimulationInputs
from src.matlab_integration.simulation_outputs import SimulationOutputs
from src.matlab_integration.simulation_validator import SimulationValidator
from src.matlab_integration.matlab_detector import MATLABDetector
from src.matlab_integration.matlab_engine_manager import MATLABEngineManager
from src.matlab_integration.telemetry_adapter import TelemetryAdapter
from src.matlab_integration.simulink_runner import SimulinkRunner
from src.matlab_simulink_bridge import AerothermalPhysicsSimulator, MATLABSimulinkBridge


# =============================================================================
# 1. PARAMETER INPUTS & UNIT CONVERSIONS TESTS
# =============================================================================

def test_simulation_inputs_defaults_and_conversions():
    """Verify default parameters and mathematical unit conversions."""
    inp = SimulationInputs()
    
    # Check default sea-level values
    assert inp.ambient_temp_k == 288.15
    assert inp.ambient_press_kpa == 101.325
    assert inp.altitude_m == 0.0
    assert inp.mach_number == 0.0
    assert inp.throttle_pct == 100.0

    # Conversions
    assert abs(inp.ambient_temp_r - 518.67) < 0.1
    assert abs(inp.ambient_press_psia - 14.696) < 0.01
    assert abs(inp.altitude_ft - 0.0) < 0.01
    assert abs(inp.fuel_flow_pps - 2.315) < 0.05

    # Serialization
    d = inp.to_dict()
    assert "ambient_temp_k" in d
    assert "ambient_press_psia" in d
    assert d["scenario_name"] == "Standard Operating Condition"


def test_simulation_inputs_from_imperial():
    """Verify instantiation from Imperial engineering units."""
    inp = SimulationInputs.from_imperial(
        ambient_temp_r=518.67,
        ambient_press_psia=14.696,
        altitude_ft=32808.4,
        fuel_flow_pps=1.146
    )
    assert abs(inp.ambient_temp_k - 288.15) < 0.1
    assert abs(inp.ambient_press_kpa - 101.325) < 0.1
    assert abs(inp.altitude_m - 10000.0) < 1.0
    assert abs(inp.fuel_flow_kg_s - 0.52) < 0.02


def test_simulation_inputs_presets():
    """Verify flight operating condition presets."""
    inp = SimulationInputs()
    
    inp.apply_preset("Cruise (35,000 ft, M 0.82)")
    assert inp.operating_preset.startswith("Cruise")
    assert inp.mach_number == 0.82
    assert abs(inp.altitude_m - 10668.0) < 1.0
    assert inp.throttle_pct == 85.0

    inp.apply_preset("Sea Level Static (Takeoff)")
    assert inp.altitude_m == 0.0
    assert inp.throttle_pct == 100.0


# =============================================================================
# 2. VALIDATION ENGINE & THERMODYNAMIC CONSTRAINTS TESTS
# =============================================================================

def test_input_validation_nominal_and_out_of_bounds():
    """Verify input parameter boundaries enforcement."""
    validator = SimulationValidator()

    # Nominal inputs should pass
    valid_inp = SimulationInputs()
    is_valid, errs, warns = validator.validate_inputs(valid_inp)
    assert is_valid is True
    assert len(errs) == 0

    # Negative temperature should fail
    invalid_inp = SimulationInputs(ambient_temp_k=150.0)
    is_valid, errs, warns = validator.validate_inputs(invalid_inp)
    assert is_valid is False
    assert any("ambient_temp_k" in e for e in errs)

    # Supersonic Mach should fail civil turbofan bound
    invalid_mach = SimulationInputs(mach_number=1.8)
    is_valid, errs, warns = validator.validate_inputs(invalid_mach)
    assert is_valid is False
    assert any("mach_number" in e for e in errs)

    # Severe degradation should generate warning
    degraded_inp = SimulationInputs(hpc_eff_modifier=0.75)
    is_valid, errs, warns = validator.validate_inputs(degraded_inp)
    assert is_valid is True
    assert any("severe degradation" in w.lower() for w in warns)


def test_output_validation_brayton_cycle():
    """Verify thermodynamic gas path continuity checks."""
    validator = SimulationValidator()

    # Nominal outputs should pass
    nominal_out = SimulationOutputs()
    status, msgs = validator.validate_outputs(nominal_out)
    assert status == "PASS"

    # Temperature inversion: HPC cooling instead of heating (T30 < T24)
    invalid_t30 = SimulationOutputs(t24_k=400.0, t30_k=350.0)
    status, msgs = validator.validate_outputs(invalid_t30)
    assert status == "FAIL"
    assert any("Thermodynamic violation" in m for m in msgs)

    # Combustor flameout: T40 <= T30
    flameout_out = SimulationOutputs(t30_k=850.0, t40_k=800.0)
    status, msgs = validator.validate_outputs(flameout_out)
    assert status == "FAIL"
    assert any("Combustor flameout" in m for m in msgs)

    # Non-finite values
    nan_out = SimulationOutputs(t50_k=float("nan"))
    status, msgs = validator.validate_outputs(nan_out)
    assert status == "FAIL"
    assert any("Non-finite" in m for m in msgs)

    # Extreme turbine temperature warning
    hot_out = SimulationOutputs(t40_k=1900.0)
    status, msgs = validator.validate_outputs(hot_out)
    assert status == "WARNING"
    assert any("Extreme Turbine Inlet Temperature" in m for m in msgs)


# =============================================================================
# 3. ENVIRONMENT DETECTOR & STATUS AUDIT TESTS
# =============================================================================

def test_matlab_detector_offline_environment():
    """Verify detection when MATLAB is absent from host PATH (zero fake data)."""
    # Detector with mocked absent executable
    detector = MATLABDetector()
    with patch.object(detector, "_find_matlab_executable", return_value=None):
        report = detector.detect_matlab(force_refresh=True)

        assert "status" in report
        assert "is_installed" in report
        assert "simulink_status" in report
        assert "toolboxes" in report
        assert "python_version" in report
        assert "setup_instructions_link" in report

        # Verify offline classification
        assert report["is_installed"] is False
        assert report["status"] == "MISSING"
        assert report["can_execute"] is False
        assert report["simulink_status"] == "MISSING"


def test_matlab_detector_mocked_available():
    """Verify detector parsing when MATLAB executable is present."""
    detector = MATLABDetector()
    with patch.object(detector, "_find_matlab_executable", return_value="C:\\Fake\\MATLAB\\bin\\matlab.exe"):
        with patch.object(detector, "_probe_via_cli", return_value={
            "success": True,
            "release": "R2024a",
            "version": "24.1.0",
            "matlabroot": "C:\\Fake\\MATLAB",
            "toolboxes": {"Simulink": "24.1", "Simscape": "24.1"},
            "licenses": ["SIMULINK", "MATLAB"],
            "licensed": True,
            "license_type": "Student License"
        }):
            report = detector.detect_matlab(force_refresh=True)
            assert report["is_installed"] is True
            assert report["status"] == "AVAILABLE"
            assert report["release"] == "R2024a"
            assert report["simulink_status"] == "AVAILABLE"
            assert report["can_execute"] is True


def test_matlab_detector_mocked_license_unavailable():
    """Verify detector handles unlicensed MATLAB without failing silently."""
    detector = MATLABDetector()
    with patch.object(detector, "_find_matlab_executable", return_value="C:\\Fake\\MATLAB\\bin\\matlab.exe"):
        with patch.object(detector, "_probe_via_cli", return_value={
            "success": False,
            "error": "License checkout failed for feature MATLAB"
        }):
            report = detector.detect_matlab(force_refresh=True)
            assert report["is_installed"] is True
            assert report["status"] == "LICENSE_UNAVAILABLE"
            assert report["can_execute"] is False


# =============================================================================
# 4. TELEMETRY ADAPTER & AI PROGNOSTICS LINKAGE TESTS
# =============================================================================

def test_telemetry_adapter_feature_mapping():
    """Verify physical simulation telemetry converts cleanly to C-MAPSS feature space."""
    adapter = TelemetryAdapter()
    outputs = SimulationOutputs(
        t24_k=350.0,
        t30_k=850.0,
        t50_k=860.0,
        p30_kpa=3800.0,
        fan_speed_rpm=2388.0,
        core_speed_rpm=9050.0
    )
    
    feats = adapter.map_telemetry_to_features(outputs)
    assert "sensor_2" in feats
    assert "sensor_3" in feats
    assert "sensor_4" in feats
    assert "sensor_7" in feats
    assert "sensor_8" in feats
    assert "sensor_9" in feats

    # Test unit conversion values
    assert abs(feats["sensor_2"] - 350.0 * 1.8) < 1e-2
    assert abs(feats["sensor_3"] - 850.0 * 1.8) < 1e-2
    assert abs(feats["sensor_4"] - 860.0 * 1.8) < 1e-2
    assert abs(feats["sensor_7"] - 3800.0 * 0.145038) < 1e-2
    assert feats["sensor_8"] == 2388.0
    assert feats["sensor_9"] == 9050.0


def test_telemetry_adapter_prognostic_inference():
    """Verify PyTorch model inference over physical simulation outputs."""
    adapter = TelemetryAdapter()
    outputs = SimulationOutputs()
    
    prog = adapter.evaluate_prognostics(outputs)
    assert "predicted_rul_cycles" in prog
    assert "rul_confidence_interval" in prog
    assert "epistemic_uncertainty_std" in prog
    assert "health_index" in prog
    assert "anomaly_score" in prog
    assert "maintenance_urgency" in prog
    assert "maintenance_recommendation" in prog

    assert 0.0 <= prog["predicted_rul_cycles"] <= 125.0
    assert 0.0 <= prog["health_index"] <= 1.0
    assert prog["maintenance_urgency"] in ["LOW", "MEDIUM", "HIGH"]


# =============================================================================
# 5. SIMULINK RUNNER & SCENARIO EXECUTION TESTS
# =============================================================================

def test_simulink_runner_fallback_execution():
    """Verify SimulinkRunner executes certified aerothermal fallback when MATLAB is offline."""
    runner = SimulinkRunner()
    inputs = SimulationInputs()
    inputs.apply_preset("Takeoff")

    res = runner.run_simulation(inputs, force_fallback=True)
    assert res["success"] is True
    assert "SIM-" in res["scenario_id"]
    assert res["execution_summary"]["is_matlab_native"] is False
    assert "Python Aerothermal Physics" in res["execution_summary"]["execution_engine"]
    assert res["validation"]["status"] in ["PASS", "WARNING"]
    assert len(res["stations"]) == 7
    assert "time_seconds" in res["time_series"]
    assert res["prognostics"]["adapter_status"] == "SUCCESS"


def test_simulink_runner_scenario_comparison():
    """Verify delta comparison between baseline and modified simulations."""
    runner = SimulinkRunner()

    base_inputs = SimulationInputs()
    base_res = runner.run_simulation(base_inputs, force_fallback=True)

    degraded_inputs = SimulationInputs(hpt_eff_modifier=0.88)
    deg_res = runner.run_simulation(degraded_inputs, force_fallback=True)

    comp = runner.compare_simulations(base_res, deg_res)
    assert "metrics_comparison" in comp
    assert "prognostics_comparison" in comp
    assert "engineering_explanation" in comp
    assert comp["baseline_scenario_id"] == base_res["scenario_id"]
    assert comp["modified_scenario_id"] == deg_res["scenario_id"]


def test_simulink_runner_persistence():
    """Verify simulation scenarios persist into SQLite database."""
    runner = SimulinkRunner()
    inputs = SimulationInputs(scenario_name="Test Persistence Scenario")
    res = runner.run_simulation(inputs, force_fallback=True)
    scen_id = res["scenario_id"]

    # Verify retrieval
    loaded = runner.get_simulation_by_id(scen_id)
    assert loaded is not None
    assert loaded["scenario_id"] == scen_id
    assert loaded["scenario_name"] == "Test Persistence Scenario"
    assert "inputs" in loaded
    assert "steady_state" in loaded
    assert "prognostics" in loaded


# =============================================================================
# 6. AUTHENTIC MATLAB ASSETS INTEGRITY TESTS
# =============================================================================

def test_matlab_assets_exist_and_valid():
    """Verify that required MATLAB simulation script and configuration files exist."""
    project_root = pathlib.Path(__file__).resolve().parents[1]
    
    script_m = project_root / "matlab_models" / "scripts" / "run_turbofan_simulation.m"
    builder_m = project_root / "matlab_models" / "models" / "create_turbofan_slx.m"
    config_json = project_root / "matlab_models" / "configuration" / "default_parameters.json"

    assert script_m.exists(), "run_turbofan_simulation.m must exist"
    assert builder_m.exists(), "create_turbofan_slx.m must exist"
    assert config_json.exists(), "default_parameters.json must exist"

    # Check configuration content
    with open(config_json, "r") as f:
        cfg = json.load(f)
    assert "subsystems" in cfg
    assert len(cfg["subsystems"]) == 9
    assert "Fan Assembly" in cfg["subsystems"]
    assert "High-Pressure Compressor (HPC)" in cfg["subsystems"]
    assert "High-Pressure Turbine (HPT)" in cfg["subsystems"]


# =============================================================================
# 7. EMULATION & ADVANCED HOST CONFIGURATION TESTS
# =============================================================================

def test_matlab_detector_emulation_mode():
    """Verify virtual MATLAB & Simulink emulation mode provides full capability audit."""
    detector = MATLABDetector()
    detector.set_emulation_mode(True)
    report = detector.detect_matlab()

    assert report["status"] == "AVAILABLE"
    assert report["is_installed"] is True
    assert report["is_emulated"] is True
    assert report["simulink_status"] == "AVAILABLE"
    assert report["simscape_status"] == "AVAILABLE"
    assert report["engine_api_status"] == "AVAILABLE"
    assert report["can_execute"] is True
    assert "Simulink" in report["toolboxes"]
    assert "Simscape Fluids" in report["toolboxes"]


def test_matlab_detector_custom_path():
    """Verify custom MATLAB path setting and clearing."""
    detector = MATLABDetector()
    detector.set_custom_path(r"C:\Custom\MATLAB\R2024a\bin\matlab.exe")
    assert detector.custom_matlab_path == r"C:\Custom\MATLAB\R2024a\bin\matlab.exe"
    detector.set_custom_path("")
    assert detector.custom_matlab_path is None
