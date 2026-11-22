"""JETWISE AI — Aerospace Engineering Copilot Interactive UI Workspace.

Provides the 3-column conversational engineering console:
- Left: Scenario history, session tracking, suggested aerospace commands.
- Center: Natural language chat stream, structured scenario summary, synchronized 3D twin, engineering graphs, and MATLAB/Simulink mathematical centre.
- Right: Engine status HUD, active simulation mode, MATLAB connectivity, and verification status.
"""

from typing import Dict, Any, List, Optional
import time
import json
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from src.copilot_nlu import AerospaceCopilotNLU, ConversationalIntent
from src.copilot_scenarios import (
    EngineeringScenario,
    ScenarioResult,
    SimulationMode,
    ComponentVisualState,
    VerificationStatus,
    PARAMETER_MAP,
    save_scenario_to_db,
    list_saved_scenarios
)
from src.copilot_simulation import CopilotSimulationEngine
from src.copilot_investigation import AutonomousInvestigationEngine, ScenarioComparisonEngine
from src.digital_twin import render_threejs_digital_twin_html
from src.dataset_registry import DatasetRegistry


def init_copilot_session():
    """Initialize Streamlit session state variables for the Copilot."""
    if "copilot_history" not in st.session_state:
        st.session_state["copilot_history"] = [
            {
                "sender": "AGENT",
                "text": (
                    "👋 **Welcome to JETWISE AI — Aerospace Engineering Copilot.**\n\n"
                    "I am your conversational aerospace simulation architect and prognostic assistant. "
                    "You can communicate with me in natural English to simulate physical scenarios, "
                    "test prognostic model sensitivity, investigate anomalies, or run physical gas-path simulations.\n\n"
                    "**Try an instruction below or type your own command!**"
                ),
                "timestamp": time.strftime("%H:%M:%S")
            }
        ]
    if "copilot_scenarios" not in st.session_state:
        st.session_state["copilot_scenarios"] = []
    if "active_scenario" not in st.session_state:
        st.session_state["active_scenario"] = None
    if "active_result" not in st.session_state:
        st.session_state["active_result"] = None
    if "baseline_scenario" not in st.session_state:
        st.session_state["baseline_scenario"] = None
    if "baseline_result" not in st.session_state:
        st.session_state["baseline_result"] = None


def render_copilot_workspace():
    """Main rendering entrypoint for the Aerospace Engineering Copilot."""
    init_copilot_session()

    st.markdown("""
    <div style="background: #151D27; border: 1px solid #1E293B; border-left: 4px solid #00B8D9; padding: 14px 20px; border-radius: 6px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="color: #F8FAFC; margin: 0; font-size: 20px;">🧑‍✈️ JETWISE AI — Aerospace Engineering Copilot</h3>
                <p style="color: #94A3B8; margin: 4px 0 0 0; font-size: 13px;">
                    Conversational Aerospace Simulation, Autonomous Verification & Synchronized 3D Digital Twin Intelligence
                </p>
            </div>
            <div style="display: flex; gap: 8px;">
                <span style="background: rgba(0, 184, 217, 0.15); color: #00B8D9; border: 1px solid #00B8D9; padding: 4px 10px; border-radius: 4px; font-size: 11px; font-weight: bold;">
                    COGNITIVE NLU ACTIVE
                </span>
                <span style="background: rgba(33, 199, 122, 0.15); color: #21C77A; border: 1px solid #21C77A; padding: 4px 10px; border-radius: 4px; font-size: 11px; font-weight: bold;">
                    3D TWIN LINKED
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nlu = AerospaceCopilotNLU()
    sim_engine = CopilotSimulationEngine()
    inv_engine = AutonomousInvestigationEngine()

    # Three-Column Industrial Console
    col_left, col_center, col_right = st.columns([1.1, 2.4, 1.1])

    # ---------------------------------------------------------------------------
    # LEFT COLUMN: Scenario History & Suggested Commands
    # ---------------------------------------------------------------------------
    with col_left:
        st.markdown("#### 📜 Scenario History")

        scen_list = st.session_state["copilot_scenarios"]
        if scen_list:
            scen_names = [f"#{i+1}: {s.scenario_name[:26]}" for i, (s, r) in enumerate(scen_list)]
            selected_idx = st.selectbox("Reopen Past Scenario", range(len(scen_names)), format_func=lambda i: scen_names[i])
            if st.button("📂 Load Selected Scenario", use_container_width=True):
                chosen_s, chosen_r = scen_list[selected_idx]
                st.session_state["active_scenario"] = chosen_s
                st.session_state["active_result"] = chosen_r
                st.success(f"Restored Scenario: {chosen_s.scenario_name}")
        else:
            st.caption("No scenarios executed in this session yet.")

        st.markdown("---")
        st.markdown("#### 💡 Suggested Commands")
        suggested_queries = [
            "Increase turbine temperature by 5% and reduce compressor pressure by 3%.",
            "Simulate gradual fan degradation over 200 operating cycles.",
            "Increase fuel flow by 5% and check how the engine responds.",
            "Introduce a pressure sensor abnormality.",
            "Simulate a compressor degradation scenario and estimate the remaining useful life.",
            "Compare this scenario with a healthy engine.",
            "Run the same condition in MATLAB Simulink and compare the results.",
            "Show me the changes in the 3D engine digital twin.",
            "Check whether this engine condition is within acceptable operating limits.",
            "Which parameter should I adjust to investigate this abnormality?",
            "Investigate why this engine is showing abnormal temperature readings.",
            "Create a healthy baseline engine."
        ]

        preset_click = None
        for sq in suggested_queries:
            if st.button(sq, key=f"sq_{sq[:25]}", use_container_width=True):
                preset_click = sq

    # ---------------------------------------------------------------------------
    # RIGHT COLUMN: Engine Telemetry HUD & Simulation Status
    # ---------------------------------------------------------------------------
    with col_right:
        st.markdown("#### ✈️ Engine Status")

        # Global dataset & engine selector
        available_datasets = DatasetRegistry.get_available_datasets()
        active_dataset = st.selectbox("Fleet Dataset", available_datasets, index=0)
        adapter = DatasetRegistry.get_adapter(active_dataset)
        engine_ids = adapter.get_engine_ids(kind="train")
        active_engine = st.selectbox("Engine Unit", engine_ids, index=0)

        # Baseline info
        if st.session_state["active_result"]:
            r = st.session_state["active_result"]
            st.metric("Predicted RUL", f"{r.modified_rul:.1f} Cycles", delta=f"{r.rul_delta:+.1f} Cycles")
            st.metric("Prediction Uncertainty", f"± {r.modified_uncertainty.get('std_dev', 0.0):.1f} Cycles", delta="90% Conformal")

            # Verification badge
            v_color = "#21C77A" if r.verification_status == VerificationStatus.PASS else ("#F59E0B" if r.verification_status == VerificationStatus.WARNING else "#EF4444")
            st.markdown(f"""
            <div style="background: #151D27; border: 1px solid #1E293B; border-left: 4px solid {v_color}; padding: 8px 12px; border-radius: 4px; margin-top: 10px;">
                <span style="color: #94A3B8; font-size: 11px;">Verification Status:</span><br>
                <b style="color: {v_color}; font-size: 13px;">{r.verification_status.value}</b>
            </div>
            """, unsafe_allow_html=True)

            # Simulation Source
            st.markdown(f"""
            <div style="background: #151D27; border: 1px solid #1E293B; padding: 8px 12px; border-radius: 4px; margin-top: 8px; font-size: 11px;">
                <span style="color: #94A3B8;">Simulation Engine:</span><br>
                <b style="color: #00B8D9;">{r.simulation_source}</b><br>
                <span style="color: #94A3B8;">Mode:</span> <code>{r.simulation_mode.value}</code>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Execute a command to view live prognostic predictions.")

        # MATLAB / Simulink Bridge Status
        st.markdown("---")
        matlab_status = sim_engine.matlab_bridge.get_status()
        m_color = "#21C77A" if matlab_status["is_installed"] else "#F59E0B"
        st.markdown(f"""
        <div style="background: #151D27; border: 1px solid #1E293B; border-left: 4px solid {m_color}; padding: 8px 12px; border-radius: 4px; font-size: 11px;">
            <div style="display: flex; justify-content: space-between;">
                <b style="color: #F8FAFC;">MATLAB / Simulink:</b>
                <span style="color: {m_color}; font-weight: bold;">{matlab_status['status_code']}</span>
            </div>
            <p style="color: #94A3B8; margin: 4px 0 0 0; font-size: 10px;">
                {matlab_status['message']}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------------------------------------
    # CENTER COLUMN: Chat Interface, Scenario Execution & Synced Views
    # ---------------------------------------------------------------------------
    with col_center:
        st.markdown("#### 💬 Conversational Engineering Console")

        # Chat display container
        chat_box = st.container(height=360)
        with chat_box:
            for msg in st.session_state["copilot_history"]:
                if msg["sender"] == "USER":
                    st.markdown(f"""
                    <div style="display: flex; justify-content: flex-end; margin-bottom: 8px;">
                        <div style="background: #1E293B; border: 1px solid #334155; padding: 10px 14px; border-radius: 8px; max-width: 80%; font-size: 13px;">
                            <b style="color: #00B8D9;">🧑‍✈️ Aerospace Engineer</b> <span style="font-size: 10px; color: #94A3B8;">[{msg.get('timestamp', '')}]</span>
                            <div style="color: #F8FAFC; margin-top: 4px;">{msg['text']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="display: flex; justify-content: flex-start; margin-bottom: 8px;">
                        <div style="background: #151D27; border: 1px solid #1E293B; border-left: 3px solid #00B8D9; padding: 10px 14px; border-radius: 8px; max-width: 85%; font-size: 13px;">
                            <b style="color: #21C77A;">🤖 JETWISE AI Copilot</b> <span style="font-size: 10px; color: #94A3B8;">[{msg.get('timestamp', '')}]</span>
                            <div style="color: #CBD5E1; margin-top: 4px;">{msg['text']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Input row
        user_input = st.chat_input("Enter natural language engineering command (e.g. 'Increase turbine temperature by 5% and reduce compressor pressure by 3%')...")
        query_to_run = user_input or preset_click

        if query_to_run:
            # Append User message
            st.session_state["copilot_history"].append({
                "sender": "USER",
                "text": query_to_run,
                "timestamp": time.strftime("%H:%M:%S")
            })

            # Parse query
            intent, parsed_scen, err = nlu.parse_query(
                query_to_run,
                current_scenario=st.session_state["active_scenario"],
                default_dataset=active_dataset,
                default_engine=active_engine
            )

            # Route by Intent
            if intent == ConversationalIntent.INVESTIGATE_ANOMALY:
                # Run autonomous investigation
                traj = adapter.get_engine_trajectory(active_engine, kind="train")
                diag = inv_engine.investigate_anomaly(traj, target_symptom=query_to_run)
                reply_text = f"**Diagnostic Investigation Report:**\n\n{diag['top_explanation']}\n\n"
                for h in diag["hypotheses"]:
                    reply_text += f"- **Rank {h['rank']} ({h['subsystem']}):** {h['hypothesis']} (Confidence: {h['confidence_score']*100:.0f}%)\n  - *Action:* {h['recommended_action']}\n"
                st.session_state["copilot_history"].append({
                    "sender": "AGENT",
                    "text": reply_text,
                    "timestamp": time.strftime("%H:%M:%S")
                })
                st.rerun()

            elif intent == ConversationalIntent.COMPARE_SCENARIOS:
                # Compare scenarios in session
                scen_list = st.session_state["copilot_scenarios"]
                if len(scen_list) >= 2:
                    comp_res = ScenarioComparisonEngine.compare_scenarios([r for s, r in scen_list])
                    reply_text = f"**Multi-Scenario Comparative Evaluation:**\n\n{comp_res['summary']}\n\nEvaluated {comp_res['num_scenarios']} scenarios across the fleet."
                else:
                    reply_text = "Need at least two executed scenarios in this session to perform a side-by-side comparison. Execute another scenario and try again!"
                st.session_state["copilot_history"].append({
                    "sender": "AGENT",
                    "text": reply_text,
                    "timestamp": time.strftime("%H:%M:%S")
                })
                st.rerun()

            else:
                # Execute simulation scenario
                with st.spinner(f"Executing {parsed_scen.simulation_mode.value}..."):
                    result = sim_engine.execute_scenario(parsed_scen)

                # Persist to session
                st.session_state["copilot_scenarios"].append((parsed_scen, result))
                st.session_state["active_scenario"] = parsed_scen
                st.session_state["active_result"] = result

                if parsed_scen.is_healthy_baseline or st.session_state["baseline_scenario"] is None:
                    st.session_state["baseline_scenario"] = parsed_scen
                    st.session_state["baseline_result"] = result

                # Save to database
                save_scenario_to_db(parsed_scen, result)

                # Append Agent explanation
                st.session_state["copilot_history"].append({
                    "sender": "AGENT",
                    "text": result.engineering_explanation,
                    "timestamp": time.strftime("%H:%M:%S")
                })
                st.rerun()

        # -----------------------------------------------------------------------
        # Active Scenario Execution & Multi-View Workspace
        # -----------------------------------------------------------------------
        if st.session_state["active_result"]:
            active_s = st.session_state["active_scenario"]
            active_r = st.session_state["active_result"]

            st.markdown("---")
            st.markdown(f"#### 📊 Scenario Outputs: `{active_s.scenario_name}`")

            # Structured Scenario Parameter Summary
            with st.expander("📋 View Structured Scenario Specification & Verification Checks", expanded=False):
                spec_col1, spec_col2 = st.columns(2)
                with spec_col1:
                    st.markdown("**Configured Parameters:**")
                    if active_s.parameters:
                        for p in active_s.parameters:
                            st.write(f"- `{p.parameter_key}`: `{p.change_value:+}%` ({p.change_type}) on `{p.target_subsystem}`")
                    else:
                        st.write("- Factory nominal telemetry (no perturbations)")
                    st.write(f"- **Simulation Mode:** `{active_s.simulation_mode.value}`")
                    st.write(f"- **Operating Duration:** `{active_s.duration_cycles} cycles`")
                with spec_col2:
                    st.markdown("**Verification Checks:**")
                    for c in active_r.verification_checks:
                        badge = "✅" if c.get("status") == "PASS" else ("⚠️" if c.get("status") == "WARNING" else "❌")
                        st.write(f"{badge} **{c.get('name')}:** {c.get('details')}")

            # 4 Synchronized Workspaces
            t1, t2, t3, t4 = st.tabs([
                "🌐 3D Digital Twin (Synced)",
                "📈 Telemetry & RUL Graphs",
                "⚙️ MATLAB & Simulink Centre",
                "⚖️ Cross-Scenario Comparison"
            ])

            # TAB 1: Synchronized 3D Digital Twin
            with t1:
                st.caption("3D Turbofan Digital Twin synchronized with active scenario. Perturbed components are highlighted in real-time.")
                twin_html = render_threejs_digital_twin_html(
                    engine_id=active_s.engine_id,
                    cycle=active_s.cycle,
                    predicted_rul=active_r.modified_rul,
                    health_state="CRITICAL" if active_r.modified_rul < 15 else ("WARNING" if active_r.modified_rul < 30 else "HEALTHY"),
                    mpi=max(0.0, min(100.0, (125.0 - active_r.modified_rul) * 0.7)),
                    telemetry=active_r.modified_telemetry,
                    highlight_components=active_r.affected_components,
                    component_visual_states=active_r.component_states,
                    scenario_info={
                        "name": active_s.scenario_name,
                        "mode": active_s.simulation_mode.value,
                        "rul_delta": active_r.rul_delta
                    }
                )
                st.components.v1.html(twin_html, height=540, scrolling=False)

            # TAB 2: Synchronized Engineering Charts
            with t2:
                g_col1, g_col2 = st.columns(2)
                with g_col1:
                    # Sensor comparison chart (Baseline vs Modified)
                    comp_sensors = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9"]
                    labels = ["T24", "T30", "T50 (EGT)", "P30", "Nf", "Nc"]
                    b_vals = [active_r.baseline_telemetry.get(s, 0.0) for s in comp_sensors]
                    m_vals = [active_r.modified_telemetry.get(s, 0.0) for s in comp_sensors]

                    fig_tel = go.Figure()
                    fig_tel.add_trace(go.Bar(x=labels, y=b_vals, name="Baseline Engine", marker_color="#38BDF8"))
                    fig_tel.add_trace(go.Bar(x=labels, y=m_vals, name=f"Scenario: {active_s.scenario_name[:15]}", marker_color="#F59E0B"))
                    fig_tel.update_layout(
                        barmode="group",
                        title="Baseline vs Modified Sensor Telemetry",
                        template="plotly_dark",
                        paper_bgcolor="#151D27",
                        plot_bgcolor="#0B0F14",
                        height=300
                    )
                    st.plotly_chart(fig_tel, use_container_width=True)

                with g_col2:
                    # RUL & Uncertainty Comparison
                    r_labels = ["Baseline RUL", "Scenario RUL"]
                    r_vals = [active_r.baseline_rul, active_r.modified_rul]
                    r_err = [active_r.baseline_uncertainty.get("std_dev", 0.0), active_r.modified_uncertainty.get("std_dev", 0.0)]

                    fig_rul = go.Figure()
                    fig_rul.add_trace(go.Bar(
                        x=r_labels, y=r_vals,
                        error_y=dict(type="data", array=r_err, visible=True),
                        marker_color=["#21C77A", "#EF4444" if active_r.rul_delta < 0 else "#00B8D9"],
                        text=[f"{v:.1f} Cycles" for v in r_vals],
                        textposition="inside"
                    ))
                    fig_rul.update_layout(
                        title=f"Remaining Useful Life Impact (Δ = {active_r.rul_delta:+.1f} Cycles)",
                        template="plotly_dark",
                        paper_bgcolor="#151D27",
                        plot_bgcolor="#0B0F14",
                        height=300
                    )
                    st.plotly_chart(fig_rul, use_container_width=True)

                # Time series if Mode B
                if active_r.time_series_data:
                    st.markdown("##### Dynamic Trajectory Progression")
                    ts_df = pd.DataFrame(active_r.time_series_data)
                    fig_ts = px.line(
                        ts_df, x=ts_df.columns[0], y=ts_df.columns[1],
                        title="Scenario Forward Trajectory Over Operational Cycles",
                        template="plotly_dark"
                    )
                    fig_ts.update_layout(paper_bgcolor="#151D27", plot_bgcolor="#0B0F14", height=280)
                    st.plotly_chart(fig_ts, use_container_width=True)

            # TAB 3: MATLAB & Simulink Mathematical Engineering Centre
            with t3:
                st.markdown("##### ⚙️ MATLAB & Simulink Physics Simulation Centre")
                st.markdown(f"**Execution Status:** `{active_r.matlab_status}` | **Active Engine:** `{active_r.simulation_source}`")

                if active_r.matlab_message:
                    st.info(active_r.matlab_message)

                m_c1, m_c2 = st.columns(2)
                with m_c1:
                    st.markdown("**Physical Equations Applied:**")
                    st.latex(r"CPR = \frac{P_{30}}{P_{24}}, \quad \eta_{\text{poly}} = \frac{\ln(P_{30}/P_{24})^{\frac{\gamma-1}{\gamma}}}{\ln(T_{30}/T_{24})}")
                    st.latex(r"\dot{m}_{\text{fuel}} \cdot \text{LHV} \cdot \eta_{\text{comb}} = \dot{m}_{\text{core}} c_p (T_{40} - T_{30})")
                    st.latex(r"W_{\text{HPT}} = \dot{m}_{\text{core}} c_p (T_{40} - T_{45}) = W_{\text{HPC}}")

                with m_c2:
                    st.markdown("**Simulated Gas Path Stations:**")
                    st.write("- **Station 2 (Fan Inlet):** Total Temp $T_2 = 518.7^\circ\text{R}$, Pressure $P_2 = 14.7\text{ psia}$")
                    st.write("- **Station 2.4 (LPC Exit):** Total Temp $T_{24} = 642.5^\circ\text{R}$, Pressure $P_{24} = 24.2\text{ psia}$")
                    st.write("- **Station 3 (HPC Exit):** Total Temp $T_{30} = 1585.0^\circ\text{R}$, Pressure $P_{30} = 553.5\text{ psia}$")
                    st.write("- **Station 5 (LPT Exhaust / EGT):** $T_{50} = 1405.0^\circ\text{R}$")

            # TAB 4: Cross-Scenario Comparison
            with t4:
                st.markdown("##### ⚖️ Cross-Scenario Comparison Matrix")
                scen_list = st.session_state["copilot_scenarios"]
                if scen_list:
                    comp_data = ScenarioComparisonEngine.compare_scenarios([r for s, r in scen_list])
                    st.dataframe(pd.DataFrame(comp_data["comparison_matrix"]), use_container_width=True, hide_index=True)
                    st.markdown(f"**Summary:** {comp_data['summary']}")
                else:
                    st.caption("No scenarios available for comparison.")
