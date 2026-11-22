"""Unit Tests for 3D Turbofan Digital Twin (JETWISE AI Feature 1).

Verifies:
1. Definition of all 9 core turbofan subassemblies
2. Sensor mappings and degradation descriptions
3. Plotly 3D Figure construction in assembled and exploded states
4. Three.js WebGL HTML generation with interactive controls & HUD
5. Inferred diagnostic disclaimer inclusion
"""

import pytest
import numpy as np
import plotly.graph_objects as go

from src.digital_twin import (
    COMPONENT_DEFINITIONS,
    generate_turbofan_3d_figure,
    render_threejs_digital_twin_html
)


def test_nine_turbofan_components_defined():
    required_components = [
        "fan_assembly",
        "lpc",
        "hpc",
        "combustor",
        "hpt",
        "lpt",
        "exhaust_nozzle",
        "engine_casing",
        "central_shaft"
    ]
    for comp in required_components:
        assert comp in COMPONENT_DEFINITIONS, f"Missing required component: {comp}"
        meta = COMPONENT_DEFINITIONS[comp]
        assert "name" in meta
        assert "station" in meta
        assert "sensors" in meta
        assert "degradation_mode" in meta


def test_plotly_assembled_and_exploded_figure():
    tel = {"sensor_2": 643.0, "sensor_3": 1590.0, "sensor_4": 1410.0, "sensor_14": 8140.0}
    
    # Assembled
    fig_assembled = generate_turbofan_3d_figure(
        telemetry=tel,
        health_state="HEALTHY",
        predicted_rul=95.0,
        mpi=20.0,
        exploded_fraction=0.0
    )
    assert isinstance(fig_assembled, go.Figure)
    assert len(fig_assembled.data) >= 8

    # Exploded
    fig_exploded = generate_turbofan_3d_figure(
        telemetry=tel,
        health_state="CRITICAL",
        predicted_rul=14.0,
        mpi=82.0,
        highlight_component="hpc",
        exploded_fraction=0.75
    )
    assert isinstance(fig_exploded, go.Figure)
    assert len(fig_exploded.data) >= 8


def test_threejs_html_generation_and_features():
    tel = {"sensor_2": 645.0, "sensor_3": 1600.0, "sensor_4": 1420.0}
    anom = {"is_anomaly": True, "severity": "WARNING"}
    
    html = render_threejs_digital_twin_html(
        engine_id=1,
        cycle=165,
        predicted_rul=24.5,
        health_state="WARNING",
        mpi=62.0,
        telemetry=tel,
        anomaly_info=anom,
        max_cycles=210
    )

    assert isinstance(html, str)
    assert len(html) > 5000
    # Verify presence of Three.js and OrbitControls
    assert "three.min.js" in html
    assert "OrbitControls.js" in html
    # Verify presence of all 9 components in embedded JSON
    for comp_key in COMPONENT_DEFINITIONS.keys():
        assert comp_key in html
    # Verify presence of Exploded View and Camera controls
    assert "btn-exploded" in html
    assert "btn-assembled" in html
    assert "resetCamera" in html
    # Verify presence of diagnostic disclaimer
    assert "Inferred Diagnostic" in html
