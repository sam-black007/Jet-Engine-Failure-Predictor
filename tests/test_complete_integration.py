"""JETWISE AI — Comprehensive Master Integration Test Suite.

Verifies end-to-end functionality across all 11 phases:
1. Phase 0: Security & Credential Configuration (Zero leakage, masking, safe offline fallback)
2. Phase 1 & 2: MATLAB Student Environment Detection, 6 formal statuses & Calculation Test
3. Phase 4: 10-Component 3D Turbofan Digital Twin, 7 Selectable Modes, Canvas Legend, Spool Speeds
4. Phase 4F: Aerothermal & Mechanical Threshold Configuration System & Provenance
5. Phase 6: Multi-Provider Aerospace Engineering Agent (Gemini, Hugging Face, OpenAI, Local Fallback)
"""

import pytest
import numpy as np
import json
import pathlib

# Security & Credentials
from src.security.credential_manager import (
    CredentialManager,
    get_credential_manager,
    ProviderInfo
)

# Multi-Provider Chatbot
from src.chatbot.multi_provider_agent import (
    MultiProviderAerospaceAgent,
    ProviderType,
    AgentResponse,
    get_aerospace_agent
)

# Digital Twin & Thresholds
from src.digital_twin import (
    COMPONENT_DEFINITIONS,
    generate_turbofan_3d_figure,
    render_threejs_digital_twin_html
)
from src.digital_twin_thresholds import (
    ThresholdManager,
    get_threshold_manager,
    EngineeringThreshold,
    DEFAULT_THRESHOLDS
)

# MATLAB Integration
from src.matlab_integration.matlab_detector import MATLABDetector
from src.matlab_integration.matlab_engine_manager import MATLABEngineManager


# ---------------------------------------------------------------------------
# 1. PHASE 0: SECURITY & CREDENTIAL MANAGEMENT
# ---------------------------------------------------------------------------

def test_credential_manager_masking():
    """Verify that credentials are masked properly without leaking secrets."""
    masked = CredentialManager.mask_key("AIzaSyD-842719X9a1B7c")
    assert "AIzaSy" in masked
    assert "1B7c" in masked
    assert "842719" not in masked
    assert CredentialManager.mask_key("short") == "••••••••"
    assert CredentialManager.mask_key(None) == "Not Set"


def test_credential_error_sanitization():
    """Verify that raw keys are scrubbed from error messages."""
    secret_key = "AIzaSySecretApiKey123456"
    raw_error = f"HTTP 403 Forbidden for key {secret_key}: Quota exceeded."
    clean_error = CredentialManager.sanitize_error(raw_error, [secret_key])
    assert secret_key not in clean_error
    assert "[MASKED_KEY]" in clean_error


def test_credential_manager_providers():
    """Verify standard providers are registered in CredentialManager."""
    cm = get_credential_manager()
    providers = cm.get_all_providers()
    assert "google_gemini" in providers
    assert "huggingface" in providers
    assert "openai" in providers
    assert "matlab_local" in providers


# ---------------------------------------------------------------------------
# 2. PHASE 4: 10-COMPONENT 3D DIGITAL TWIN & 7 MODES
# ---------------------------------------------------------------------------

def test_all_ten_turbofan_components_defined():
    """Verify that all 10 turbofan subassemblies (including Sensors) are defined."""
    expected_components = [
        "fan_assembly",
        "lpc",
        "hpc",
        "combustor",
        "hpt",
        "lpt",
        "exhaust_nozzle",
        "engine_casing",
        "central_shaft",
        "sensors"
    ]
    assert len(COMPONENT_DEFINITIONS) == 10
    for comp in expected_components:
        assert comp in COMPONENT_DEFINITIONS
        meta = COMPONENT_DEFINITIONS[comp]
        assert "name" in meta
        assert "station" in meta
        assert "sensors" in meta
        assert "degradation_mode" in meta
        assert "nominal_temp" in meta


def test_plotly_ten_components_rendering():
    """Verify Plotly 3D digital twin generates traces for 10 subassemblies."""
    fig = generate_turbofan_3d_figure(
        telemetry={"sensor_2": 642.5, "sensor_3": 1588.0, "sensor_4": 1406.0},
        highlight_component="sensors",
        exploded_fraction=0.3
    )
    trace_names = [t.name for t in fig.data if hasattr(t, "name") and t.name]
    assert "Sensors & Instrumentation Harness" in trace_names
    assert "High-Pressure Compressor (HPC)" in trace_names
    assert len(fig.data) >= 9


def test_threejs_seven_modes_and_canvas_legend():
    """Verify Three.js HTML incorporates 7 modes, 7-state canvas legend, and spool rotation."""
    tel = {
        "sensor_2": 643.0,
        "sensor_3": 1595.0,
        "sensor_4": 1420.0,
        "sensor_8": 8410.0,
        "sensor_9": 2390.0
    }
    html = render_threejs_digital_twin_html(
        engine_id=1,
        cycle=120,
        predicted_rul=45.0,
        health_state="WARNING",
        telemetry=tel,
        twin_mode="Engine Health Mode"
    )

    # 7-State Canvas Legend Bar
    assert "legend-bar" in html
    assert "BLUE: Cold" in html
    assert "GREEN: Normal" in html
    assert "YELLOW: Caution" in html
    assert "ORANGE: Warning" in html
    assert "RED: Critical" in html
    assert "PURPLE: Sensor Fault" in html
    assert "GREY: Unsimulated" in html

    # 7 Twin Modes
    assert "Engine Health Mode" in html
    assert "Thermal Mode" in html
    assert "Pressure Mode" in html
    assert "Component Degradation Mode" in html
    assert "Sensor Monitoring Mode" in html
    assert "Predictive Maintenance Mode" in html
    assert "Baseline Comparison Mode" in html

    # Component 10 Sensors Harness in Three.js
    assert 'name = "sensors"' in html or 'sensorsGrp' in html
    assert "10. Sensors & Instrumentation" in html

    # Dual-Spool Rotation
    assert "lpRotSpeed" in html
    assert "hpRotSpeed" in html


# ---------------------------------------------------------------------------
# 3. PHASE 4F: AEROTHERMAL THRESHOLD CONFIGURATION SYSTEM
# ---------------------------------------------------------------------------

def test_threshold_manager_provenance_and_defaults():
    """Verify threshold catalogue provenance metadata and limits."""
    tm = get_threshold_manager()
    assert "T30" in tm.thresholds
    t30 = tm.thresholds["T30"]
    assert t30.unit == "°R"
    assert t30.caution_yellow_limit == 1635.0
    assert t30.critical_red_limit == 1700.0
    assert "NASA" in t30.provenance_source or "Commercial" in t30.provenance_source
    assert t30.documented_reference != ""


def test_threshold_manager_seven_states_evaluation():
    """Verify 7-state threshold evaluation produces proper hex colors and statuses."""
    tm = get_threshold_manager()

    # Normal state
    col, state, _ = tm.evaluate_component_health_color("hpc", {"sensor_3": 1585.0})
    assert state == "GREEN"
    assert col == "#10B981"

    # Caution state
    col, state, _ = tm.evaluate_component_health_color("hpc", {"sensor_3": 1640.0})
    assert state == "YELLOW"
    assert col == "#FBBF24"

    # Warning state
    col, state, _ = tm.evaluate_component_health_color("hpc", {"sensor_3": 1670.0})
    assert state == "ORANGE"
    assert col == "#F97316"

    # Critical state
    col, state, _ = tm.evaluate_component_health_color("hpc", {"sensor_3": 1720.0})
    assert state == "RED"
    assert col == "#EF4444"

    # Cold state
    col, state, _ = tm.evaluate_component_health_color("hpc", {"sensor_3": 1400.0})
    assert state == "BLUE"
    assert col == "#3B82F6"

    # Sensor fault state
    col, state, _ = tm.evaluate_component_health_color("sensors", {}, is_sensor_fault=True)
    assert state == "PURPLE"
    assert col == "#A855F7"


# ---------------------------------------------------------------------------
# 4. PHASE 6: MULTI-PROVIDER AEROSPACE ENGINEERING AGENT
# ---------------------------------------------------------------------------

def test_multi_provider_agent_local_parser():
    """Verify MultiProviderAerospaceAgent processes queries deterministically."""
    agent = get_aerospace_agent(default_provider=ProviderType.LOCAL)
    query = "Increase turbine temperature by 5% and reduce compressor pressure by 3%."
    resp = agent.process_query(query, preferred_provider=ProviderType.LOCAL)

    assert isinstance(resp, AgentResponse)
    assert resp.intent == "SIMULATE_SCENARIO"
    assert resp.scenario is not None
    assert len(resp.scenario.parameters) == 2
    assert resp.provider_used == "Local Deterministic Engine"
    assert resp.latency_ms > 0.0
    assert not resp.is_fallback


def test_multi_provider_agent_fallback():
    """Verify fallback when an unconfigured external LLM is requested."""
    agent = get_aerospace_agent()
    # Temporarily ask for Gemini when running in headless test
    resp = agent.process_query("Simulate gradual fan degradation over 100 cycles.", preferred_provider=ProviderType.GEMINI)

    assert isinstance(resp, AgentResponse)
    assert resp.scenario is not None
    # If GEMINI_API_KEY is not set or network fails, it must fall back cleanly
    if resp.is_fallback:
        assert "fallback" in resp.explanation.lower()
        assert resp.provider_used == "Local Deterministic Engine"


# ---------------------------------------------------------------------------
# 5. PHASE 1 & 2: MATLAB STUDENT DETECTION & CALCULATION TEST
# ---------------------------------------------------------------------------

def test_matlab_formal_status_reported():
    """Verify MATLABDetector returns one of the 6 formal statuses."""
    detector = MATLABDetector()
    report = detector.detect_matlab()
    formal_statuses = {
        "INSTALLED AND LICENSED",
        "INSTALLED BUT NOT LICENSED",
        "NOT INSTALLED",
        "NOT REQUIRED",
        "NOT VERIFIED",
        "INCOMPATIBLE"
    }
    assert "formal_status" in report
    assert report["formal_status"] in formal_statuses


def test_matlab_calculation_test():
    """Verify execute_calculation_test performs mathematical verification."""
    detector = MATLABDetector()
    engine_mgr = MATLABEngineManager(detector=detector)
    res = engine_mgr.execute_calculation_test()

    assert res["status"] in ["CALCULATION_TEST_PASSED", "CALCULATION_TEST_FAILED", "NOT_INSTALLED"]
    assert "latency_ms" in res
    assert "operations" in res
    if res["success"]:
        assert len(res["operations"]) >= 2
        for op in res["operations"]:
            assert op["status"] == "PASS"
