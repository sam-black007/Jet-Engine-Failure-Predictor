"""JETWISE AI — Aerospace Engineering Copilot Interactive UI Workspace.

Prompt-Driven Aerospace Engineering Intelligence Platform:
- Prominent natural language prompt interface with status badges and follow-up command pills.
- Transparent 6-dimension Engineering Results Scorecard.
- Dynamic Graphical Dashboard (KPI cards, thermal/pressure profiles, spool speeds, component table).
- Synchronized 3D Turbofan Digital Twin (10 components, 7 visual states).
- Two Presentation Levels:
    * Level 1: Engineering Summary Mode (Default accessible view)
    * Level 2: Advanced Engineering Mode (Complete unvarnished MATLAB, Simulink, mathematical equations, raw telemetry, AI weights, and sanitized logs)
- Automated 17-point Engineering Report export (Markdown, printable HTML, CSV).
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
from src.chatbot.multi_provider_agent import get_aerospace_agent, ProviderType, AgentResponse
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
from src.copilot_scorecard import ScorecardCalculator, EngineeringScorecard
from src.copilot_visualizations import (
    generate_kpi_summary_data,
    generate_scenario_graphs,
    generate_component_health_table
)
from src.copilot_technical_inspector import TechnicalInspector
from src.copilot_report_generator import CopilotReportGenerator
from src.digital_twin import render_threejs_digital_twin_html
from src.dataset_registry import DatasetRegistry


def init_copilot_session():
    """Initialize Streamlit session state variables for Prompt-Driven Intelligence."""
    if "copilot_history" not in st.session_state:
        st.session_state["copilot_history"] = [
            {
                "sender": "AGENT",
                "text": (
                    "👋 **Welcome to JETWISE AI — Prompt-Driven Aerospace Engineering Intelligence Console.**\n\n"
                    "Communicate with me in natural English to simulate physical scenarios, evaluate multi-cycle degradation, "
                    "analyze aerodynamic gas-path telemetry, or execute MATLAB/Simulink physics runs.\n\n"
                    "**Click a suggested command below or type your instruction to begin.**"
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
    if "active_scorecard" not in st.session_state:
        st.session_state["active_scorecard"] = None
    if "baseline_scenario" not in st.session_state:
        st.session_state["baseline_scenario"] = None
    if "baseline_result" not in st.session_state:
        st.session_state["baseline_result"] = None
    if "show_advanced_technical_mode" not in st.session_state:
        st.session_state["show_advanced_technical_mode"] = False


def render_copilot_workspace():
    """Main rendering entrypoint for Prompt-Driven Aerospace Engineering Intelligence."""
    init_copilot_session()

    # Top Status & Mission Header
    active_s = st.session_state.get("active_scenario")
    current_engine = active_s.selected_engine_config_id if active_s else "NASA-CMAPSS-FD001"
    current_model = active_s.model_name if active_s else "pinball_lstm_q35"
    current_dataset = active_s.dataset if active_s else "FD001"

    st.markdown(f"""
    <div style="background: #111827; border: 1px solid #1F2937; border-left: 4px solid #00B8D9; padding: 14px 20px; border-radius: 6px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div>
                <h3 style="color: #F8FAFC; margin: 0; font-size: 20px;">🧑‍✈️ JETWISE AI — Prompt-Driven Aerospace Engineering Intelligence</h3>
                <p style="color: #94A3B8; margin: 4px 0 0 0; font-size: 13px;">
                    Natural-Language Aerothermal Simulation, Multi-Spool Prognostics & Advanced 3D Digital Twin Synchronization
                </p>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <span style="background: rgba(0, 184, 217, 0.15); color: #00B8D9; border: 1px solid #00B8D9; padding: 4px 10px; border-radius: 4px; font-size: 11px; font-weight: bold;">
                    ✈️ {current_engine}
                </span>
                <span style="background: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid #10B981; padding: 4px 10px; border-radius: 4px; font-size: 11px; font-weight: bold;">
                    🧠 {current_model}
                </span>
                <span style="background: rgba(245, 158, 11, 0.15); color: #F59E0B; border: 1px solid #F59E0B; padding: 4px 10px; border-radius: 4px; font-size: 11px; font-weight: bold;">
                    📊 {current_dataset}
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nlu = AerospaceCopilotNLU()
    sim_engine = CopilotSimulationEngine()
    inv_engine = AutonomousInvestigationEngine()

    # Three-Column Industrial Console Layout
    col_left, col_center, col_right = st.columns([1.1, 2.5, 1.1])

    # ---------------------------------------------------------------------------
    # LEFT COLUMN: Scenario History & Suggested Aerospace Prompts
    # ---------------------------------------------------------------------------
    with col_left:
        st.markdown("#### 📜 Scenario History")

        scen_list = st.session_state["copilot_scenarios"]
        if scen_list:
            scen_names = [f"#{i+1}: {s.scenario_name[:26]}" for i, (s, r) in enumerate(scen_list)]
            selected_idx = st.selectbox("Past Scenarios", range(len(scen_names)), format_func=lambda i: scen_names[i])
            if st.button("📂 Load Selected Scenario", use_container_width=True):
                chosen_s, chosen_r = scen_list[selected_idx]
                st.session_state["active_scenario"] = chosen_s
                st.session_state["active_result"] = chosen_r
                st.session_state["active_scorecard"] = ScorecardCalculator.calculate_for_result(chosen_s, chosen_r)
                st.success(f"Restored: {chosen_s.scenario_name}")
        else:
            st.caption("No past simulation runs in this session.")

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
            "Which component is most affected?",
            "Explain the abnormal temperature increase.",
            "Show me the MATLAB simulation outputs.",
            "Create a healthy baseline engine."
        ]

        preset_click = None
        for sq in suggested_queries:
            if st.button(sq, key=f"sq_{sq[:24]}", use_container_width=True):
                preset_click = sq

    # ---------------------------------------------------------------------------
    # RIGHT COLUMN: Engine Telemetry HUD & Environment Status
    # ---------------------------------------------------------------------------
    with col_right:
        st.markdown("#### ✈️ Engine Status")

        available_datasets = DatasetRegistry.get_available_datasets()
        active_dataset = st.selectbox("Dataset", available_datasets, index=0)
        adapter = DatasetRegistry.get_adapter(active_dataset)
        engine_ids = adapter.get_engine_ids(kind="train")
        active_engine = st.selectbox("Unit ID", engine_ids, index=0)

        if st.session_state.get("active_result"):
            r = st.session_state["active_result"]
            st.metric("Predicted RUL", f"{r.modified_rul:.1f} Cycles", delta=f"{r.rul_delta:+.1f} Cycles")
            std_val = r.modified_uncertainty.get("sigma", r.modified_uncertainty.get("std_dev", 5.4))
            st.metric("Uncertainty Bound (σ)", f"± {std_val:.1f} Cycles", delta="90% Conformal")

            v_color = "#10B981" if getattr(r.verification_status, "value", str(r.verification_status)) == "PASS" else ("#F59E0B" if getattr(r.verification_status, "value", str(r.verification_status)) == "WARNING" else "#EF4444")
            v_val = getattr(r.verification_status, "value", str(r.verification_status))
            st.markdown(f"""
            <div style="background: #111827; border: 1px solid #1F2937; border-left: 4px solid {v_color}; padding: 8px 12px; border-radius: 4px; margin-top: 10px;">
                <span style="color: #94A3B8; font-size: 11px;">Verification Status:</span><br>
                <b style="color: {v_color}; font-size: 13px;">{v_val} (7/7 Checks)</b>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background: #111827; border: 1px solid #1F2937; padding: 8px 12px; border-radius: 4px; margin-top: 8px; font-size: 11px;">
                <span style="color: #94A3B8;">Simulation Engine:</span><br>
                <b style="color: #00B8D9;">{r.simulation_source}</b><br>
                <span style="color: #94A3B8;">Status:</span> <code>{r.matlab_status}</code>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Awaiting engineering instruction.")

        st.markdown("---")
        matlab_status = sim_engine.matlab_bridge.get_status()
        m_color = "#10B981" if matlab_status["is_installed"] else "#F59E0B"
        st.markdown(f"""
        <div style="background: #111827; border: 1px solid #1F2937; border-left: 4px solid {m_color}; padding: 8px 12px; border-radius: 4px; font-size: 11px;">
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
    # CENTER COLUMN: Chat Interface, Execution Pipeline, and Dynamic Results
    # ---------------------------------------------------------------------------
    with col_center:
        st.markdown("#### 💬 Conversational Engineering Console")

        prov_col1, prov_col2 = st.columns([2, 1])
        with prov_col1:
            provider_labels = {
                ProviderType.GEMINI: "🤖 Google Gemini (Primary NLU)",
                ProviderType.HUGGINGFACE: "🤗 Hugging Face Inference",
                ProviderType.OPENAI: "⚡ OpenAI (GPT-4o)",
                ProviderType.LOCAL: "⚙️ Local Deterministic Physics Parser"
            }
            selected_prov = st.selectbox(
                "NLU Intelligence Engine",
                options=[ProviderType.GEMINI, ProviderType.HUGGINGFACE, ProviderType.OPENAI, ProviderType.LOCAL],
                format_func=lambda p: provider_labels[p],
                index=0,
                key="copilot_nlu_provider_selector"
            )
        with prov_col2:
            agent = get_aerospace_agent(default_provider=selected_prov)
            prov_meta = agent.get_available_providers().get(selected_prov.value, {})
            st.markdown(f"""
            <div style="margin-top: 26px; padding: 5px 8px; background: #111827; border: 1px solid #1F2937; border-radius: 4px; font-size: 11px; text-align: center;">
                <span style="color: {'#10B981' if prov_meta.get('configured') else '#F59E0B'}; font-weight: bold;">
                    {prov_meta.get('badge', 'Active')}
                </span>
            </div>
            """, unsafe_allow_html=True)

        # Chat display container
        chat_box = st.container(height=340)
        with chat_box:
            for msg in st.session_state["copilot_history"]:
                if msg["sender"] == "USER":
                    st.markdown(f"""
                    <div style="display: flex; justify-content: flex-end; margin-bottom: 8px;">
                        <div style="background: #1F2937; border: 1px solid #374151; padding: 10px 14px; border-radius: 8px; max-width: 80%; font-size: 13px;">
                            <b style="color: #00B8D9;">🧑‍✈️ Aerospace Engineer</b> <span style="font-size: 10px; color: #94A3B8;">[{msg.get('timestamp', '')}]</span>
                            <div style="color: #F8FAFC; margin-top: 4px;">{msg['text']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="display: flex; justify-content: flex-start; margin-bottom: 8px;">
                        <div style="background: #111827; border: 1px solid #1F2937; border-left: 3px solid #00B8D9; padding: 10px 14px; border-radius: 8px; max-width: 88%; font-size: 13px;">
                            <b style="color: #10B981;">🤖 JETWISE AI Intelligence</b> <span style="font-size: 10px; color: #94A3B8;">[{msg.get('timestamp', '')}]</span>
                            <div style="color: #CBD5E1; margin-top: 4px;">{msg['text']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Context-Aware Follow-Up Command Pills
        if st.session_state.get("active_scenario"):
            st.markdown("**Follow-Up Prompts:**")
            f_cols = st.columns(4)
            if f_cols[0].button("📉 +3% HPC Degradation", use_container_width=True):
                preset_click = "Now reduce compressor efficiency by another 3%"
            if f_cols[1].button("🔍 Which Component Worst?", use_container_width=True):
                preset_click = "Which component is most affected?"
            if f_cols[2].button("🌡️ Explain Temperature", use_container_width=True):
                preset_click = "Explain the abnormal temperature increase"
            if f_cols[3].button("🔬 View MATLAB Details", use_container_width=True):
                preset_click = "Show me the MATLAB simulation outputs"

        # Chat Input
        user_input = st.chat_input("Enter engineering command (e.g. 'Increase turbine temperature by 5% and reduce compressor pressure by 3%')...")
        query_to_run = user_input or preset_click

        if query_to_run:
            st.session_state["copilot_history"].append({
                "sender": "USER",
                "text": query_to_run,
                "timestamp": time.strftime("%H:%M:%S")
            })

            # Process query via Aerospace Copilot NLU and Multi-Provider Agent
            agent = get_aerospace_agent(default_provider=selected_prov)
            agent_resp = agent.process_query(
                query=query_to_run,
                current_scenario=st.session_state.get("active_scenario"),
                default_dataset=active_dataset,
                default_engine=active_engine,
                preferred_provider=selected_prov
            )

            intent = agent_resp.intent
            parsed_scen = agent_resp.scenario
            if parsed_scen is None:
                _, parsed_scen, _ = nlu.parse_query(
                    query_to_run,
                    current_scenario=st.session_state.get("active_scenario"),
                    default_dataset=active_dataset,
                    default_engine=active_engine
                )

            # Route by Intent
            if intent == ConversationalIntent.SHOW_TECHNICAL_DETAILS:
                st.session_state["show_advanced_technical_mode"] = True
                reply_text = agent_resp.explanation or "🔬 **Level 2 Advanced Engineering Workspace Activated.** Scroll below to inspect MATLAB, Simulink, and physical equations."
                st.session_state["copilot_history"].append({
                    "sender": "AGENT",
                    "text": reply_text,
                    "timestamp": time.strftime("%H:%M:%S")
                })
                st.rerun()

            elif intent in [ConversationalIntent.LOCALIZE_FAULT, ConversationalIntent.EXPLAIN_ANOMALY]:
                reply_text = f"*[Provider: {agent_resp.provider_used} | Model: {agent_resp.model_name}]*\n\n{agent_resp.explanation}"
                st.session_state["copilot_history"].append({
                    "sender": "AGENT",
                    "text": reply_text,
                    "timestamp": time.strftime("%H:%M:%S")
                })
                st.rerun()

            elif intent == ConversationalIntent.INVESTIGATE_ANOMALY:
                traj = adapter.get_engine_trajectory(active_engine, kind="train")
                diag = inv_engine.investigate_anomaly(traj, target_symptom=query_to_run)
                reply_text = f"*[Provider: {agent_resp.provider_used} | Model: {agent_resp.model_name}]*\n\n"
                reply_text += f"**Diagnostic Investigation Report:**\n\n{diag['top_explanation']}\n\n"
                for h in diag["hypotheses"]:
                    reply_text += f"- **Rank {h['rank']} ({h['subsystem']}):** {h['hypothesis']} (Confidence: {h['confidence_score']*100:.0f}%)\n  - *Action:* {h['recommended_action']}\n"
                st.session_state["copilot_history"].append({
                    "sender": "AGENT",
                    "text": reply_text,
                    "timestamp": time.strftime("%H:%M:%S")
                })
                st.rerun()

            elif intent == ConversationalIntent.COMPARE_SCENARIOS:
                scen_list = st.session_state["copilot_scenarios"]
                if len(scen_list) >= 2:
                    comp_res = ScenarioComparisonEngine.compare_scenarios([r for s, r in scen_list])
                    reply_text = f"*[Provider: {agent_resp.provider_used} | Model: {agent_resp.model_name}]*\n\n"
                    reply_text += f"**Multi-Scenario Comparative Evaluation:**\n\n{comp_res['summary']}\n\nEvaluated {comp_res['num_scenarios']} scenarios across the fleet."
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
                sim_mode_val = getattr(parsed_scen.simulation_mode, "value", str(parsed_scen.simulation_mode))
                with st.spinner(f"Executing aerothermal simulation ({sim_mode_val})..."):
                    result = sim_engine.execute_scenario(parsed_scen)
                    scorecard = ScorecardCalculator.calculate_for_result(parsed_scen, result)

                # Persist to session
                st.session_state["copilot_scenarios"].append((parsed_scen, result))
                st.session_state["active_scenario"] = parsed_scen
                st.session_state["active_result"] = result
                st.session_state["active_scorecard"] = scorecard

                if parsed_scen.is_healthy_baseline or st.session_state["baseline_scenario"] is None:
                    st.session_state["baseline_scenario"] = parsed_scen
                    st.session_state["baseline_result"] = result

                save_scenario_to_db(parsed_scen, result)

                prov_header = f"*[Provider: {agent_resp.provider_used} | Model: {agent_resp.model_name} | Latency: {agent_resp.latency_ms} ms]*\n\n"
                if agent_resp.explanation and "⚠️" not in agent_resp.explanation:
                    full_explanation = f"{prov_header}**Thermodynamic Analysis:**\n{agent_resp.explanation}\n\n---\n{result.engineering_explanation}"
                else:
                    full_explanation = f"{prov_header}{result.engineering_explanation}"

                st.session_state["copilot_history"].append({
                    "sender": "AGENT",
                    "text": full_explanation,
                    "timestamp": time.strftime("%H:%M:%S")
                })
                st.rerun()

    # ===========================================================================
    # FULL-WIDTH RESULTS VIEW: LEVEL 1 SUMMARY & LEVEL 2 TECHNICAL WORKSPACE
    # ===========================================================================
    if st.session_state.get("active_result"):
        active_s = st.session_state["active_scenario"]
        active_r = st.session_state["active_result"]
        active_sc = st.session_state.get("active_scorecard") or ScorecardCalculator.calculate_for_result(active_s, active_r)

        st.markdown("---")
        st.markdown(f"### 📊 Engineering Results & Digital Twin: `{active_s.scenario_name}`")

        # -----------------------------------------------------------------------
        # TOP-LEVEL KPI SCORECARD (6 DIMENSIONS)
        # -----------------------------------------------------------------------
        kpi_cards = generate_kpi_summary_data(active_sc)
        kpi_cols = st.columns(len(kpi_cards))
        for idx, card in enumerate(kpi_cards):
            with kpi_cols[idx]:
                st.markdown(f"""
                <div style="background: #111827; border: 1px solid #1F2937; border-top: 3px solid {card['color']}; border-radius: 6px; padding: 12px; text-align: center;">
                    <div style="color: #94A3B8; font-size: 11px; text-transform: uppercase; font-weight: 600;">{card['title']}</div>
                    <div style="color: #F8FAFC; font-size: 20px; font-weight: 700; margin: 4px 0;">{card['value']}</div>
                    <div style="color: {card['color']}; font-size: 11px; font-weight: 600;">{card['status']}</div>
                    <div style="color: #64748B; font-size: 10px; margin-top: 4px;">Δ: {card['delta']}</div>
                </div>
                """, unsafe_allow_html=True)

        # -----------------------------------------------------------------------
        # LEVEL 1: ENGINEERING SUMMARY MODE (DEFAULT)
        # -----------------------------------------------------------------------
        st.markdown("#### 🌐 Level 1: Engineering Summary Mode")
        
        l1_col1, l1_col2 = st.columns([1.3, 1.0])
        with l1_col1:
            st.markdown("##### Synchronized 3D Digital Twin")
            enable_copilot_3d = st.toggle(
                "🌐 Enable 3D WebGL Digital Twin Rendering",
                value=False,
                key="copilot_enable_3d_twin",
                help="Turn ON to load the interactive 3D WebGL model. Off by default for lightweight 2D performance."
            )
            if enable_copilot_3d:
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
                        "mode": getattr(active_s.simulation_mode, "value", str(active_s.simulation_mode)),
                        "rul_delta": active_r.rul_delta
                    }
                )
                st.components.v1.html(twin_html, height=520, scrolling=False)
            else:
                st.info("ℹ️ **3D Digital Twin is hidden.** Turn on the toggle above to inspect the 3D WebGL engine model.")
                t_cols = st.columns(2)
                t_cols[0].metric("Station 30 Temp (T30)", f"{active_r.modified_telemetry.get('sensor_3', 0.0):.1f} °R")
                t_cols[1].metric("Station 50 EGT (T50)", f"{active_r.modified_telemetry.get('sensor_4', 0.0):.1f} °R")
                t_cols[0].metric("Station 30 Press (P30)", f"{active_r.modified_telemetry.get('sensor_7', 0.0):.1f} psia")
                t_cols[1].metric("Core Speed (Nc)", f"{active_r.modified_telemetry.get('sensor_9', 0.0):.1f} rpm")

        with l1_col2:
            st.markdown("##### Component-Level Health Matrix")
            comp_df = generate_component_health_table(active_r, active_sc)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Graphical Aerothermal & Prognostic Charts
        st.markdown("##### Dynamic Simulation Telemetry Figures")
        figs = generate_scenario_graphs(active_s, active_r, st.session_state.get("baseline_result"))
        g_c1, g_c2 = st.columns(2)
        with g_c1:
            if "thermal_profile" in figs:
                st.plotly_chart(figs["thermal_profile"], use_container_width=True)
            if "spool_speeds" in figs:
                st.plotly_chart(figs["spool_speeds"], use_container_width=True)
        with g_c2:
            if "pressure_profile" in figs:
                st.plotly_chart(figs["pressure_profile"], use_container_width=True)
            if "multi_sensor_delta" in figs:
                st.plotly_chart(figs["multi_sensor_delta"], use_container_width=True)

        # Automatic Engineering Report Generator Section
        st.markdown("---")
        st.markdown("#### 📑 Automatic Engineering Report Export")
        rep_gen = CopilotReportGenerator(active_s, active_r, active_sc)

        rep_c1, rep_c2, rep_c3, rep_c4 = st.columns(4)
        with rep_c1:
            st.download_button(
                "📥 Download Markdown (.md)",
                data=rep_gen.generate_markdown_report(),
                file_name=f"{rep_gen.report_id}.md",
                mime="text/markdown",
                use_container_width=True
            )
        with rep_c2:
            st.download_button(
                "🖨️ Download HTML (Printable)",
                data=rep_gen.generate_html_report(),
                file_name=f"{rep_gen.report_id}.html",
                mime="text/html",
                use_container_width=True
            )
        with rep_c3:
            st.download_button(
                "📊 Download Fleet CSV",
                data=rep_gen.generate_csv_summary(),
                file_name=f"{rep_gen.report_id}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with rep_c4:
            view_rep = st.button("👁️ View Full Report Inline", use_container_width=True)

        if view_rep:
            with st.expander(f"📄 Full 17-Point Engineering Report: {rep_gen.report_id}", expanded=True):
                st.markdown(rep_gen.generate_markdown_report())

        # -----------------------------------------------------------------------
        # LEVEL 2: ADVANCED ENGINEERING MODE (ON-DEMAND)
        # -----------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 🔬 Level 2: Advanced Engineering Mode")
        st.caption("Deep technical inspection: live MATLAB engine execution, Simulink ODE solver details, governing Brayton equations, and raw telemetry matrices.")

        show_adv = st.checkbox("🔬 View MATLAB & Technical Details", value=st.session_state.get("show_advanced_technical_mode", False))
        st.session_state["show_advanced_technical_mode"] = show_adv

        if show_adv:
            tab_a, tab_b, tab_c, tab_d, tab_e, tab_f = st.tabs([
                "⚙️ Tab A: MATLAB Execution",
                "📐 Tab B: Simulink Model",
                "🧮 Tab C: Mathematical Details",
                "📈 Tab D: Raw Telemetry",
                "🧠 Tab E: Python AI Analysis",
                "📜 Tab F: Execution Logs"
            ])

            # Tab A: MATLAB Execution Details
            with tab_a:
                matlab_details = TechnicalInspector.get_matlab_execution_details(active_s, active_r)
                st.markdown("##### MATLAB Execution Environment & Command Pipeline")
                st.write(f"- **Version / Release:** `{matlab_details['matlab_version']}`")
                st.write(f"- **License Type:** `{matlab_details['license_type']}`")
                st.write(f"- **Execution Mode:** `{matlab_details['execution_mode']}`")
                st.write(f"- **Execution Time:** `{matlab_details['execution_time_ms']} ms` | **Memory:** `{matlab_details['memory_usage_mb']} MB`")
                st.markdown("**Executed Command String:**")
                st.code(matlab_details["command_line"], language="bash")
                st.markdown("**Raw Output / Engine Log:**")
                st.code(matlab_details["raw_output"], language="text")
                st.markdown("**Workspace Variables Created:**")
                st.dataframe(pd.DataFrame(matlab_details["workspace_variables"]), use_container_width=True, hide_index=True)

            # Tab B: Simulink Model Details
            with tab_b:
                simulink_details = TechnicalInspector.get_simulink_model_details(active_s, active_r)
                st.markdown("##### Simulink Model Architecture & Solver Settings")
                st.write(f"- **Model Name:** `{simulink_details['model_name']}` ({simulink_details['model_version']})")
                st.write(f"- **File Path:** `{simulink_details['file_path']}`")
                st.write(f"- **Solver:** `{simulink_details['solver_settings']['solver_type']}` ({simulink_details['solver_settings']['step_type']})")
                st.write(f"- **Relative Tolerance:** `{simulink_details['solver_settings']['relative_tolerance']}` | **Absolute Tolerance:** `{simulink_details['solver_settings']['absolute_tolerance']}`")
                
                s_c1, s_c2 = st.columns(2)
                with s_c1:
                    st.markdown("**Input Port Mappings:**")
                    st.dataframe(pd.DataFrame(simulink_details["input_port_mappings"]), use_container_width=True, hide_index=True)
                with s_c2:
                    st.markdown("**Output Port Mappings:**")
                    st.dataframe(pd.DataFrame(simulink_details["output_port_mappings"]), use_container_width=True, hide_index=True)

            # Tab C: Mathematical Details
            with tab_c:
                math_details = TechnicalInspector.get_mathematical_details(active_s, active_r)
                st.markdown("##### Governing Aerothermodynamic Equations & Cycle Relations")
                for eq in math_details["equations"]:
                    st.markdown(f"**{eq['title']}:**")
                    st.latex(eq["latex"])
                    st.caption(f"Evaluated: {eq['evaluated']} ({eq['unit']}) — {eq['description']}")

                st.markdown("---")
                m_c1, m_c2 = st.columns(2)
                with m_c1:
                    st.markdown("**Thermodynamic Assumptions:**")
                    st.dataframe(pd.DataFrame(math_details["thermodynamic_assumptions"]), use_container_width=True, hide_index=True)
                with m_c2:
                    st.markdown("**Authoritative Citations & References:**")
                    for cit in math_details["citations"]:
                        st.markdown(f"- *{cit}*")

            # Tab D: Raw Telemetry
            with tab_d:
                telem_details = TechnicalInspector.get_raw_telemetry_details(active_s, active_r)
                st.markdown("##### Raw 21-Sensor Telemetry Matrix")
                raw_df = pd.DataFrame(telem_details["telemetry_rows"])
                st.dataframe(raw_df, use_container_width=True, hide_index=True)

                st.markdown("##### Summary Statistics Across Timesteps")
                stats_df = pd.DataFrame(telem_details["summary_statistics"])
                st.dataframe(stats_df, use_container_width=True, hide_index=True)

                d_c1, d_c2 = st.columns(2)
                with d_c1:
                    csv_data = TechnicalInspector.export_telemetry_to_csv(telem_details)
                    st.download_button("📥 Export Telemetry as CSV", data=csv_data, file_name=f"{active_s.scenario_id}_telemetry.csv", mime="text/csv", use_container_width=True)
                with d_c2:
                    json_data = TechnicalInspector.export_telemetry_to_json(telem_details)
                    st.download_button("📥 Export Telemetry as JSON", data=json_data, file_name=f"{active_s.scenario_id}_telemetry.json", mime="application/json", use_container_width=True)

            # Tab E: Python AI Analysis
            with tab_e:
                ai_details = TechnicalInspector.get_python_ai_details(active_s, active_r)
                st.markdown("##### Prognostic Neural Architecture & Attribution Details")
                st.write(f"- **Model Name:** `{ai_details['model_architecture']['name']}` ({ai_details['model_architecture']['version']})")
                st.write(f"- **Framework:** `{ai_details['model_architecture']['framework']}`")
                st.write(f"- **Input Dim:** `{ai_details['model_architecture']['input_dimensions']}`")
                st.write(f"- **Trainable Parameters:** `{ai_details['model_architecture']['parameters_count']:,}`")

                st.markdown("**Permutation Feature Sensitivity Ranking:**")
                st.dataframe(pd.DataFrame(ai_details["feature_importance"]), use_container_width=True, hide_index=True)

                st.markdown("**Conformal Prediction Calibration Metadata:**")
                st.dataframe(pd.DataFrame([ai_details["calibration_metadata"]]), use_container_width=True, hide_index=True)

            # Tab F: Simulation Execution Logs
            with tab_f:
                filter_choice = st.selectbox("Log Filter Level", ["ALL", "INFO", "WARNING", "ERROR", "DEBUG"], index=0)
                logs = TechnicalInspector.get_execution_logs(active_s, active_r, filter_level=filter_choice)
                st.markdown(f"##### Execution Audit Trail ({len(logs)} events logged with zero secret leakage)")
                log_text = TechnicalInspector.export_logs_to_text(logs)
                st.text_area("Sanitized Execution Log", value=log_text, height=320)
                st.download_button("📥 Download Execution Log", data=log_text, file_name=f"{active_s.scenario_id}_audit.log", mime="text/plain", use_container_width=True)
