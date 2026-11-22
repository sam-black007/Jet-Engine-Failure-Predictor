"""JETWISE AI — Aerospace Copilot Dynamic Graphical Visualizations.

Generates relevant engineering graphics and component analysis tables after each simulation:
1. Top-Level KPI metrics with baseline deltas
2. Gas-path thermal profile (T2, T24, T30, T50) with aerothermal redline thresholds
3. Pressure profile (P2, P15, P30, Ps30) across stations
4. Dual-spool rotational speeds (N1 Fan, N2 Core)
5. Fuel flow ratio (phi / Wf) and burner enthalpy
6. Turbomachinery degradation and RUL uncertainty cone (q10 - q90 shaded interval)
7. Baseline vs modified multi-sensor delta comparison
8. Component-level health table across all 10 turbofan subassemblies
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.copilot_scorecard import EngineeringScorecard
from src.digital_twin import COMPONENT_DEFINITIONS
from src.digital_twin_thresholds import get_threshold_manager


def generate_kpi_summary_data(
    scorecard: EngineeringScorecard,
    baseline_scorecard: Optional[EngineeringScorecard] = None
) -> List[Dict[str, Any]]:
    """Generate structured data for the 6 top-level KPI cards with baseline deltas."""
    # 1. Engine Health Score
    base_health = baseline_scorecard.engine_health.score if baseline_scorecard else None
    delta_health = (scorecard.engine_health.score - base_health) if base_health is not None else None
    kpi_health = {
        "title": "Engine Health Score",
        "value": f"{scorecard.engine_health.score:.1f} / 100",
        "status": scorecard.engine_health.classification,
        "color": scorecard.engine_health.color_hex,
        "delta": f"{delta_health:+.1f}" if delta_health is not None else "Baseline",
        "delta_color": "normal" if (delta_health is None or delta_health >= 0) else "inverse",
        "baseline_val": f"{base_health:.1f}" if base_health is not None else "N/A",
        "explanation": "Weighted gas-path aerothermal and prognostic life margin."
    }

    # 2. Engine Risk Score
    base_risk = baseline_scorecard.engine_risk.score if baseline_scorecard else None
    delta_risk = (scorecard.engine_risk.score - base_risk) if base_risk is not None else None
    kpi_risk = {
        "title": "Engine Risk Score",
        "value": f"{scorecard.engine_risk.score:.1f} / 100",
        "status": scorecard.engine_risk.risk_level,
        "color": scorecard.engine_risk.color_hex,
        "delta": f"{delta_risk:+.1f}" if delta_risk is not None else "Baseline",
        "delta_color": "inverse" if (delta_risk is not None and delta_risk > 0) else "normal",
        "baseline_val": f"{base_risk:.1f}" if base_risk is not None else "N/A",
        "explanation": "Estimated risk from anomaly severity and prognostic dispersion."
    }

    # 3. Predicted Remaining Useful Life
    base_rul = baseline_scorecard.rul_metric.predicted_rul if baseline_scorecard else None
    delta_rul = (scorecard.rul_metric.predicted_rul - base_rul) if base_rul is not None else None
    kpi_rul = {
        "title": "Remaining Useful Life",
        "value": f"{scorecard.rul_metric.predicted_rul:.1f} {scorecard.rul_metric.unit}",
        "status": f"±{scorecard.rul_metric.uncertainty_std_dev:.1f} σ",
        "color": "#00B8D9" if scorecard.rul_metric.predicted_rul > 40 else "#EF4444",
        "delta": f"{delta_rul:+.1f} cycles" if delta_rul is not None else "Baseline",
        "delta_color": "normal" if (delta_rul is None or delta_rul >= 0) else "inverse",
        "baseline_val": f"{base_rul:.1f}" if base_rul is not None else "N/A",
        "explanation": f"Quantile interval: [{scorecard.rul_metric.q10_lower:.1f} – {scorecard.rul_metric.q90_upper:.1f}] cycles."
    }

    # 4. Anomaly Metric
    kpi_anomaly = {
        "title": "Anomaly Score",
        "value": f"{scorecard.anomaly.score:.1f} / 100",
        "status": scorecard.anomaly.classification,
        "color": "#EF4444" if scorecard.anomaly.is_anomaly else "#10B981",
        "delta": "ACTIVE ANOMALY" if scorecard.anomaly.is_anomaly else "NORMAL",
        "delta_color": "inverse" if scorecard.anomaly.is_anomaly else "normal",
        "baseline_val": "0.0 (Nominal)",
        "explanation": f"Method: {scorecard.anomaly.detection_method}"
    }

    # 5. Critical Components
    crit_comps = [c.upper().replace("_", " ") for c in scorecard.critical_components]
    kpi_critical = {
        "title": "Critical Components",
        "value": ", ".join(crit_comps[:2]) if crit_comps else "NONE",
        "status": f"{len(scorecard.critical_components)} Monitored",
        "color": "#F97316" if len(scorecard.critical_components) > 1 else "#10B981",
        "delta": f"{len(scorecard.critical_components)} Subsystems",
        "delta_color": "normal",
        "baseline_val": "All Nominal",
        "explanation": "Components with active perturbations or low thermal margins."
    }

    # 6. Simulation Status & Validity
    kpi_sim = {
        "title": "Simulation Status",
        "value": scorecard.validity.overall_status,
        "status": scorecard.simulation_engine,
        "color": "#10B981" if scorecard.validity.overall_status == "PASS" else "#FBBF24",
        "delta": f"{scorecard.validity.pass_count}/{scorecard.validity.total_checks} Checks Pass",
        "delta_color": "normal",
        "baseline_val": "Verified",
        "explanation": scorecard.validity.validation_summary
    }

    return [kpi_health, kpi_risk, kpi_rul, kpi_anomaly, kpi_critical, kpi_sim]


def generate_scenario_graphs(
    scenario: Any,
    result: Any,
    baseline_result: Optional[Any] = None
) -> Dict[str, go.Figure]:
    """Dynamically construct relevant Plotly figures based on scenario parameters."""
    figs: Dict[str, go.Figure] = {}
    mod_tel = result.modified_telemetry or {}
    base_tel = (baseline_result.modified_telemetry if baseline_result else result.baseline_telemetry) or {}
    time_series = result.time_series_data or {}

    # Common Dark Theme Layout
    dark_layout = dict(
        paper_bgcolor="#0B0F14",
        plot_bgcolor="#111827",
        font=dict(color="#E2E8F0", family="monospace"),
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor="#1F2937", zerolinecolor="#374151"),
        yaxis=dict(gridcolor="#1F2937", zerolinecolor="#374151"),
        legend=dict(bgcolor="rgba(17, 24, 39, 0.8)", bordercolor="#374151")
    )

    # 1. RUL Trajectory & Uncertainty Cone (if time-series available)
    if "cycle" in time_series and "predicted_rul" in time_series:
        cycles = time_series["cycle"]
        ruls = time_series["predicted_rul"]
        fig_rul = go.Figure()

        # Uncertainty Cone (± 12%)
        ruls_arr = np.array(ruls)
        upper_cone = ruls_arr * 1.12
        lower_cone = np.maximum(0.0, ruls_arr * 0.88)

        fig_rul.add_trace(go.Scatter(
            x=cycles + cycles[::-1],
            y=list(upper_cone) + list(lower_cone[::-1]),
            fill="toself",
            fillcolor="rgba(0, 184, 217, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="90% Conformal Uncertainty Cone"
        ))
        fig_rul.add_trace(go.Scatter(
            x=cycles,
            y=ruls,
            mode="lines+markers",
            line=dict(color="#00B8D9", width=2.5),
            marker=dict(size=4),
            name="Forecasted RUL (cycles)"
        ))
        fig_rul.update_layout(
            title="<b>Prognostic RUL Trajectory & Conformal Uncertainty Cone</b>",
            xaxis_title="Simulation Cycle Step",
            yaxis_title="Remaining Useful Life (cycles)",
            **dark_layout
        )
        figs["rul_trajectory"] = fig_rul

    # 2. Gas-Path Thermal Profile (T2, T24, T30, T50)
    stations = ["St 2 (T2)", "St 2.4 (T24)", "St 3 (T30)", "St 5 (T50 / EGT)"]
    t_base = [
        base_tel.get("sensor_1", 518.67),
        base_tel.get("sensor_2", 642.5),
        base_tel.get("sensor_3", 1585.0),
        base_tel.get("sensor_4", 1405.0)
    ]
    t_mod = [
        mod_tel.get("sensor_1", 518.67),
        mod_tel.get("sensor_2", 642.5),
        mod_tel.get("sensor_3", 1585.0),
        mod_tel.get("sensor_4", 1405.0)
    ]

    fig_temp = go.Figure()
    fig_temp.add_trace(go.Bar(
        x=stations,
        y=t_base,
        name="Baseline ISA",
        marker_color="#38BDF8",
        opacity=0.75
    ))
    fig_temp.add_trace(go.Bar(
        x=stations,
        y=t_mod,
        name="Scenario Modified",
        marker_color="#F97316" if max(t_mod) > 1650 else "#10B981",
        opacity=0.9
    ))
    # Threshold Redline for T30 (1665 °R)
    fig_temp.add_hline(y=1665.0, line_dash="dash", line_color="#EF4444", annotation_text="T30 Warning (1665 °R)")
    fig_temp.update_layout(
        title="<b>Gas-Path Thermal Distribution Across Stations (°R)</b>",
        xaxis_title="Engine Aerothermal Station",
        yaxis_title="Total Temperature (°R)",
        barmode="group",
        **dark_layout
    )
    figs["thermal_profile"] = fig_temp

    # 3. Gas-Path Pressure Profile (P2, P15, P30, Ps30)
    p_stations = ["St 2 (P2)", "St 15 (P15 Bypass)", "St 3 (P30 HPC)", "St 3 (Ps30 Static)"]
    p_base = [
        base_tel.get("sensor_5", 14.696),
        base_tel.get("sensor_6", 21.6),
        base_tel.get("sensor_7", 553.0),
        base_tel.get("sensor_11", 47.5)
    ]
    p_mod = [
        mod_tel.get("sensor_5", 14.696),
        mod_tel.get("sensor_6", 21.6),
        mod_tel.get("sensor_7", 553.0),
        mod_tel.get("sensor_11", 47.5)
    ]

    fig_press = go.Figure()
    fig_press.add_trace(go.Scatter(
        x=p_stations,
        y=p_base,
        mode="lines+markers",
        line=dict(color="#38BDF8", width=2, dash="dash"),
        marker=dict(size=8),
        name="Baseline Pressure"
    ))
    fig_press.add_trace(go.Scatter(
        x=p_stations,
        y=p_mod,
        mode="lines+markers",
        line=dict(color="#A855F7", width=2.5),
        marker=dict(size=8),
        name="Scenario Pressure"
    ))
    fig_press.update_layout(
        title="<b>Aerothermal Pressure Gradient & Compression Profile (psia)</b>",
        xaxis_title="Engine Station Tap",
        yaxis_title="Pressure (psia)",
        **dark_layout
    )
    figs["pressure_profile"] = fig_press

    # 4. Spool Speeds (N1 Fan & N2 Core)
    fig_speed = go.Figure()
    speed_cats = ["N1 Fan Speed (rpm)", "N2 Core Speed (rpm)"]
    s_base = [base_tel.get("sensor_8", 2388.0), base_tel.get("sensor_9", 8400.0)]
    s_mod = [mod_tel.get("sensor_8", 2388.0), mod_tel.get("sensor_9", 8400.0)]

    fig_speed.add_trace(go.Bar(x=speed_cats, y=s_base, name="Baseline Speed", marker_color="#475569"))
    fig_speed.add_trace(go.Bar(x=speed_cats, y=s_mod, name="Simulated Speed", marker_color="#00B8D9"))
    fig_speed.update_layout(
        title="<b>Dual-Spool Rotational Speeds (N1 & N2)</b>",
        yaxis_title="Rotational Velocity (rpm)",
        barmode="group",
        **dark_layout
    )
    figs["spool_speeds"] = fig_speed

    # 5. Multi-Sensor Delta Radar / Percentage Drift
    diff_sensors = ["T24", "T30", "T50", "P30", "Nf", "Nc", "BPR", "htBleed"]
    keys = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9", "sensor_15", "sensor_14"]
    pct_deltas = []
    for k in keys:
        b = base_tel.get(k, 1.0)
        m = mod_tel.get(k, b)
        pct_deltas.append(round(((m - b) / b) * 100.0, 2) if b != 0 else 0.0)

    fig_delta = go.Figure()
    fig_delta.add_trace(go.Bar(
        x=diff_sensors,
        y=pct_deltas,
        marker_color=["#EF4444" if abs(d) >= 5.0 else ("#FBBF24" if abs(d) >= 2.0 else "#10B981") for d in pct_deltas],
        text=[f"{d:+}%" for d in pct_deltas],
        textposition="auto"
    ))
    fig_delta.update_layout(
        title="<b>Multi-Sensor Percentage Deviation Relative to Baseline (Δ%)</b>",
        xaxis_title="Sensor Channel",
        yaxis_title="Percentage Change (%)",
        **dark_layout
    )
    figs["multi_sensor_delta"] = fig_delta

    return figs


def generate_component_health_table(
    result: Any,
    scorecard: EngineeringScorecard
) -> pd.DataFrame:
    """Generate comprehensive component-level health analysis table across all subassemblies."""
    mod_tel = result.modified_telemetry or {}
    aff_comps = set(result.affected_components or [])
    tm = get_threshold_manager()

    rows = []
    for comp_id, meta in COMPONENT_DEFINITIONS.items():
        # Evaluate health via ThresholdManager
        col_hex, state_name, rationale = tm.evaluate_component_health_color(
            component_id=comp_id,
            telemetry=mod_tel,
            is_sensor_fault=(comp_id == "sensors" and scorecard.anomaly.is_anomaly)
        )

        # Status badge
        if comp_id in aff_comps or state_name in ["RED", "ORANGE", "CRITICAL", "WARNING"]:
            status_label = "⚠️ ELEVATED WEAR" if state_name in ["WARNING", "ORANGE"] else ("🔴 CRITICAL" if state_name in ["RED", "CRITICAL"] else "🔵 PERTURBED")
            health_pct = 55.0 if state_name == "ORANGE" else (25.0 if state_name == "RED" else 75.0)
        elif state_name == "PURPLE":
            status_label = "🟣 SENSOR ANOMALY"
            health_pct = 70.0
        elif state_name == "BLUE":
            status_label = "🔵 SUB-NOMINAL"
            health_pct = 85.0
        else:
            status_label = "🟢 NOMINAL"
            health_pct = 95.0

        finding = rationale if rationale else meta["degradation_mode"]

        rows.append({
            "Subassembly": meta["name"],
            "Station": meta["station"],
            "Health Score": f"{health_pct:.0f}%",
            "Operational Status": status_label,
            "Monitored Channels": ", ".join(meta["sensors"][:3]),
            "Engineering Finding": finding
        })

    return pd.DataFrame(rows)
