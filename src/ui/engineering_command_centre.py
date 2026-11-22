"""JETWISE AI — Engineering Command Centre (Simulate Hub).

Implements Section 2.2 Hub 2, Section 8, Section 9, Section 10, and Section 12 requirements:
Unified simulation workflow consolidating Conversational Prompts, 14 Reusable Scenario Presets,
Parameter Validation, Live 17-Stage Monitor, Execution Controls (Start, Pause, Resume, Cancel),
and Immediate Results preview with synchronized 3D twin.
"""

from typing import Dict, Any, List, Optional
import time
import pandas as pd
import streamlit as st

from src.copilot_scenarios import EngineeringScenario, SimulationMode
from src.copilot_simulation import CopilotSimulationEngine
from src.copilot_scorecard import ScorecardCalculator
from src.copilot_nlu import AerospaceCopilotNLU, ConversationalIntent
from src.chatbot.multi_provider_agent import get_aerospace_agent, ProviderType
from src.ui.scenario_presets_library import ScenarioPresetsManager, ScenarioPresetTemplate
from src.ui.live_simulation_monitor import LiveSimulationMonitor, SimulationExecutionContext, SimulationStage
from src.simulation_history_manager import SimulationHistoryManager
from src.ui.copilot_workspace import (
    render_threejs_digital_twin_html,
    generate_kpi_summary_data,
    generate_component_health_table,
    generate_scenario_graphs
)
from src.copilot_technical_inspector import TechnicalInspector
from src.copilot_pdf_report_generator import AerospaceReportPackageExporter
from src.dataset_registry import DatasetRegistry


def render_engineering_command_centre():
    """Render the central unified simulation workspace."""
    sim_engine = CopilotSimulationEngine()
    presets_mgr = ScenarioPresetsManager()
    history_mgr = SimulationHistoryManager()
    nlu = AerospaceCopilotNLU()

    # Presentation mode check
    app_mode = st.session_state.get("app_presentation_mode", "simple")

    # Header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #101720 0%, #151D27 100%); border-bottom: 2px solid #00B8D9; padding: 14px 20px; border-radius: 6px; margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="color: #F8FAFC; margin: 0; font-size: 20px;">⚡ JETWISE ENGINEERING COMMAND CENTRE</h3>
                <span style="color: #94A3B8; font-size: 12px;">Unified Conversational Simulation, Preset Library & Multi-Spool Telemetry Execution</span>
            </div>
            <div>
                <span style="background: rgba(0, 184, 217, 0.15); color: #00B8D9; border: 1px solid #00B8D9; padding: 4px 10px; border-radius: 4px; font-weight: 700; font-size: 11px;">
                    CO-SIMULATION DESK
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top Controls Bar: Engine, Dataset, Simulation Mode & AI Provider
    c_eng, c_ds, c_mode, c_prov = st.columns(4)

    with c_eng:
        available_datasets = DatasetRegistry.get_available_datasets()
        active_dataset = st.selectbox("Active Dataset", available_datasets, index=0, key="ecc_dataset")
        adapter = DatasetRegistry.get_adapter(active_dataset)
        engine_ids = adapter.get_engine_ids(kind="train")

    with c_ds:
        active_engine = st.selectbox("Engine Unit ID", engine_ids, index=0, key="ecc_engine_id")

    with c_mode:
        mode_options = [
            SimulationMode.MODE_B_PHYSICAL_PARAMETRIC,
            SimulationMode.MODE_C_MATLAB_SIMULINK,
            SimulationMode.MODE_A_DATASET_REPLAY,
            SimulationMode.MODE_D_BRAYTON_FALLBACK
        ]
        selected_mode = st.selectbox(
            "Simulation Backend",
            mode_options,
            format_func=lambda m: getattr(m, "value", str(m)),
            index=0,
            key="ecc_sim_mode"
        )

    with c_prov:
        prov_options = [ProviderType.LOCAL, ProviderType.GEMINI, ProviderType.HUGGINGFACE, ProviderType.OPENAI]
        selected_provider = st.selectbox(
            "NLU Intelligence",
            prov_options,
            format_func=lambda p: f"🤖 {p.value.title()}" if p != ProviderType.LOCAL else "⚙️ Local Offline Parser",
            index=0,
            key="ecc_nlu_provider"
        )

    st.markdown("---")

    # Dual Input Tabs: 1. Conversational Chatbot | 2. Scenario Presets Library
    input_tab1, input_tab2 = st.tabs([
        "💬 Conversational Simulation Prompt",
        "📚 Reusable Scenario Presets Library (14 Templates)"
    ])

    query_to_execute = None
    preset_scenario_to_execute = None

    # TAB 1: Conversational Prompt
    with input_tab1:
        st.caption("Enter natural-language flight instructions to simulate degradation, sensor bias, or operational changes.")

        # Preset Quick Buttons
        p_c1, p_c2, p_c3, p_c4 = st.columns(4)
        if p_c1.button("📉 +5% Fuel Flow", use_container_width=True):
            query_to_execute = "Increase fuel flow by 5% and check how the engine responds."
        if p_c2.button("🌡️ +5% Turbine Temp", use_container_width=True):
            query_to_execute = "Increase turbine temperature by 5% and reduce compressor pressure by 3%."
        if p_c3.button("🌀 -5% Compressor Eff", use_container_width=True):
            query_to_execute = "Reduce compressor efficiency by 5% and simulate."
        if p_c4.button("🌿 Healthy Baseline", use_container_width=True):
            query_to_execute = "Run a healthy baseline simulation."

        # Chat Prompt Box
        prompt_input = st.text_input(
            "Enter Engineering Command:",
            value=query_to_execute or "",
            placeholder="e.g. 'Simulate gradual fan degradation over 200 operating cycles and run using MATLAB'...",
            key="ecc_chat_prompt_input"
        )
        run_chat_btn = st.button("🚀 Interpret & Execute Simulation", type="primary", use_container_width=True)

        if run_chat_btn and prompt_input:
            query_to_execute = prompt_input

    # TAB 2: Scenario Presets Library
    with input_tab2:
        st.caption("Select, configure, or duplicate any of the 14 certified aerospace simulation templates.")
        all_presets = presets_mgr.list_all_presets()

        preset_cats = list(set(p.category for p in all_presets))
        selected_cat = st.selectbox("Filter Presets by Category", ["All Categories"] + preset_cats)
        
        filtered_presets = all_presets if selected_cat == "All Categories" else [p for p in all_presets if p.category == selected_cat]

        preset_names = [f"[{p.category}] {p.name}" for p in filtered_presets]
        chosen_p_idx = st.selectbox("Select Scenario Preset", range(len(preset_names)), format_func=lambda i: preset_names[i])
        chosen_preset = filtered_presets[chosen_p_idx]

        # Preset Details Card
        st.markdown(f"""
        <div style="background: #111827; border: 1px solid #1F2937; border-left: 4px solid #00B8D9; border-radius: 6px; padding: 12px 16px; margin: 10px 0;">
            <b style="color: #F8FAFC; font-size: 15px;">{chosen_preset.name}</b> <span style="color: #64748B; font-size: 11px;">(ID: {chosen_preset.preset_id})</span>
            <p style="color: #94A3B8; font-size: 12px; margin: 4px 0;">{chosen_preset.description}</p>
            <div style="font-size: 11px; color: #CBD5E1;">
                <b>Affected Components:</b> {', '.join(chosen_preset.affected_components)}<br/>
                <b>Expected Behavior:</b> {chosen_preset.expected_outputs}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Parameter Editor
        st.markdown("##### ⚙️ Preset Parameter Configuration")
        param_cols = st.columns(max(1, len(chosen_preset.parameters)))
        edited_params = {}
        for idx, (p_key, p_val) in enumerate(chosen_preset.parameters.items()):
            unit = chosen_preset.parameter_units.get(p_key, "")
            rng = chosen_preset.parameter_ranges.get(p_key, {"min": 0.0, "max": 100.0})
            with param_cols[min(idx, len(param_cols) - 1)]:
                if isinstance(p_val, float):
                    edited_params[p_key] = st.number_input(f"{p_key} ({unit})", min_value=float(rng.get("min", -50.0)), max_value=float(rng.get("max", 100.0)), value=float(p_val), step=0.5)
                elif isinstance(p_val, int):
                    edited_params[p_key] = st.number_input(f"{p_key} ({unit})", min_value=int(rng.get("min", 1)), max_value=int(rng.get("max", 500)), value=int(p_val), step=10)
                else:
                    edited_params[p_key] = st.text_input(f"{p_key} ({unit})", value=str(p_val))

        p_act_c1, p_act_c2, p_act_c3 = st.columns(3)
        with p_act_c1:
            if st.button("▶️ Execute Selected Preset", type="primary", use_container_width=True):
                chosen_preset.parameters = edited_params
                preset_scenario_to_execute = chosen_preset

        with p_act_c2:
            if st.button("📋 Duplicate Preset as Custom", use_container_width=True):
                dup = ScenarioPresetTemplate(
                    preset_id=f"CUSTOM_{int(time.time())}",
                    name=f"Custom: {chosen_preset.name}",
                    category="Custom",
                    description=f"User modified copy of {chosen_preset.name}",
                    affected_components=list(chosen_preset.affected_components),
                    parameters=dict(edited_params),
                    parameter_units=dict(chosen_preset.parameter_units),
                    parameter_ranges=dict(chosen_preset.parameter_ranges),
                    supported_modes=list(chosen_preset.supported_modes),
                    expected_outputs=chosen_preset.expected_outputs,
                    validation_criteria=list(chosen_preset.validation_criteria),
                    is_custom=True
                )
                presets_mgr.save_custom_preset(dup)
                st.success(f"Preset duplicated as custom preset: '{dup.name}'!")

        with p_act_c3:
            st.caption(f"Category: **{chosen_preset.category}**")

    # ===========================================================================
    # EXECUTION PIPELINE ROUTING
    # ===========================================================================
    target_scenario: Optional[EngineeringScenario] = None

    if preset_scenario_to_execute:
        target_scenario = presets_mgr.to_engineering_scenario(
            preset_scenario_to_execute,
            engine_id=active_engine,
            dataset=active_dataset
        )
        target_scenario.simulation_mode = selected_mode

    elif query_to_execute:
        agent = get_aerospace_agent(default_provider=selected_provider)
        agent_resp = agent.process_query(
            query=query_to_execute,
            current_scenario=st.session_state.get("active_scenario"),
            default_dataset=active_dataset,
            default_engine=active_engine,
            preferred_provider=selected_provider
        )
        target_scenario = agent_resp.scenario
        if target_scenario is None:
            _, target_scenario, _ = nlu.parse_query(
                query_to_execute,
                current_scenario=st.session_state.get("active_scenario"),
                default_dataset=active_dataset,
                default_engine=active_engine
            )
        target_scenario.simulation_mode = selected_mode

    if target_scenario:
        # Create execution context
        sim_id = f"SIM-{time.strftime('%Y%m%d-%H%M%S')}"
        ctx = SimulationExecutionContext(
            simulation_id=sim_id,
            scenario_name=target_scenario.scenario_name,
            engine_id=active_engine,
            simulation_backend=getattr(target_scenario.simulation_mode, "value", str(target_scenario.simulation_mode)),
            current_stage=SimulationStage.INITIALISING,
            current_operation="Validating aerodynamic parameters..."
        )

        # Progress pipeline simulation
        LiveSimulationMonitor.render_pipeline_card(ctx)

        with st.spinner("Executing multi-spool physical simulation & deep AI prognostics..."):
            t_start = time.perf_counter()
            result = sim_engine.execute_scenario(target_scenario)
            scorecard = ScorecardCalculator.calculate_for_result(target_scenario, result)
            ctx.elapsed_seconds = round(time.perf_counter() - t_start, 2)
            ctx.current_stage = SimulationStage.COMPLETED
            ctx.current_operation = "Simulation complete. Scorecard & twin updated."

            # Persist to database
            history_mgr.record_simulation(
                scenario=target_scenario,
                result=result,
                scorecard=scorecard,
                original_prompt=query_to_execute or target_scenario.scenario_name
            )

            # Persist in session
            st.session_state["active_scenario"] = target_scenario
            st.session_state["active_result"] = result
            st.session_state["active_scorecard"] = scorecard
            st.success(f"Simulation {sim_id} completed successfully in {ctx.elapsed_seconds}s!")

    # ===========================================================================
    # RESULTS VIEW (LEVEL 1 SUMMARY & LEVEL 2 ADVANCED ENGINEERING)
    # ===========================================================================
    if st.session_state.get("active_result"):
        active_s = st.session_state["active_scenario"]
        active_r = st.session_state["active_result"]
        active_sc = st.session_state.get("active_scorecard") or ScorecardCalculator.calculate_for_result(active_s, active_r)

        st.markdown("---")
        st.markdown(f"### 📊 Engineering Results & Digital Twin: `{active_s.scenario_name}`")

        # 6-Dimension KPI Scorecard
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
        # LEVEL 1: ENGINEERING SUMMARY MODE
        # -----------------------------------------------------------------------
        st.markdown("#### 🌐 Level 1: Engineering Summary Mode")
        l1_col1, l1_col2 = st.columns([1.3, 1.0])

        with l1_col1:
            st.markdown("##### Synchronized 3D Digital Twin (10 Turbofan Subassemblies)")
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

        with l1_col2:
            st.markdown("##### Component-Level Health Matrix")
            comp_df = generate_component_health_table(active_r, active_sc)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Aerothermal Figures
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

        # -----------------------------------------------------------------------
        # MULTI-FORMAT REPORT EXPORTER SECTION
        # -----------------------------------------------------------------------
        st.markdown("---")
        st.markdown("#### 📑 Engineering Report & Data Package Export")
        pkg_exporter = AerospaceReportPackageExporter(active_s, active_r, active_sc)

        rep_c1, rep_c2, rep_c3, rep_c4 = st.columns(4)
        with rep_c1:
            st.download_button(
                "📕 Download PDF Dossier",
                data=pkg_exporter.generate_pdf_report(),
                file_name=f"{pkg_exporter.report_id}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        with rep_c2:
            st.download_button(
                "🖨️ Download Printable HTML",
                data=pkg_exporter.core_gen.generate_html_report(),
                file_name=f"{pkg_exporter.report_id}.html",
                mime="text/html",
                use_container_width=True
            )
        with rep_c3:
            st.download_button(
                "💾 Download MATLAB (.mat)",
                data=pkg_exporter.generate_mat_workspace(),
                file_name=f"{pkg_exporter.report_id}.mat",
                mime="application/octet-stream",
                use_container_width=True
            )
        with rep_c4:
            st.download_button(
                "📊 Download Telemetry (CSV)",
                data=pkg_exporter.generate_csv_telemetry(),
                file_name=f"{pkg_exporter.report_id}_telemetry.csv",
                mime="text/csv",
                use_container_width=True
            )

        # -----------------------------------------------------------------------
        # LEVEL 2: ADVANCED ENGINEERING MODE
        # -----------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 🔬 Level 2: Advanced Engineering Mode")
        st.caption("Deep technical inspection: live MATLAB engine execution, Simulink ODE solver details, governing Brayton equations, and raw telemetry matrices.")

        show_adv = st.checkbox(
            "🔬 View MATLAB & Technical Details",
            value=st.session_state.get("show_advanced_technical_mode", False),
            key="ecc_show_adv_cb"
        )
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

            with tab_b:
                simulink_details = TechnicalInspector.get_simulink_model_details(active_s, active_r)
                st.markdown("##### Simulink Model Architecture & ODE Solver Dynamics")
                st.write(f"- **Model Name:** `{simulink_details['model_name']}` (Version {simulink_details['model_version']})")
                st.write(f"- **ODE Numerical Solver:** `{simulink_details['ode_solver']}` ({simulink_details['solver_type']})")
                st.write(f"- **Step Size:** `{simulink_details['step_size']}` | **Relative Tolerance:** `{simulink_details['relative_tolerance']}`")
                st.markdown("**Subsystem Hierarchy:**")
                st.dataframe(pd.DataFrame(simulink_details["subsystems"]), use_container_width=True, hide_index=True)
                st.markdown("**I/O Port Bindings:**")
                st.dataframe(pd.DataFrame(simulink_details["io_ports"]), use_container_width=True, hide_index=True)

            with tab_c:
                equations = TechnicalInspector.get_mathematical_equations()
                st.markdown("##### Governing Aerothermal Thermodynamics & Gas Dynamics")
                for eq in equations:
                    st.markdown(f"**{eq['name']}** `[{eq['category']}]`")
                    st.latex(eq["latex"])
                    st.write(f"*{eq['description']}*")
                    st.caption(f"Units: `{eq['units']}` | Reference: *{eq['reference']}*")
                    st.markdown("---")

            with tab_d:
                telem_data = TechnicalInspector.get_raw_telemetry_matrix(active_r)
                st.markdown(f"##### Raw Sensor Telemetry Matrix ({telem_data['total_samples']} samples)")
                telem_df = pd.DataFrame(telem_data["telemetry_dict"])
                st.dataframe(telem_df, use_container_width=True, height=280)
                st.markdown("##### Telemetry Channel Statistical Summary")
                st.dataframe(pd.DataFrame(telem_data["statistics"]), use_container_width=True, hide_index=True)

            with tab_e:
                ai_details = TechnicalInspector.get_python_ai_analysis(active_s, active_r, active_sc)
                st.markdown("##### Python Deep Prognostic Neural Architecture")
                st.write(f"- **Model Architecture:** `{ai_details['model_architecture']}`")
                st.write(f"- **Loss Formulation:** `{ai_details['loss_formulation']}`")
                st.write(f"- **Input Shape:** `{ai_details['input_shape']}` | **Training Epochs:** `{ai_details['training_epochs']}`")
                st.write(f"- **90% Conformal Interval:** `[{ai_details['conformal_coverage_90']['q10_cycles']} to {ai_details['conformal_coverage_90']['q90_cycles']} cycles]`")
                st.markdown("##### Permutation Feature Importance & Sensor Attribution")
                st.dataframe(pd.DataFrame(ai_details["feature_attributions"]), use_container_width=True, hide_index=True)

            with tab_f:
                logs = TechnicalInspector.get_sanitized_execution_logs(active_s, active_r)
                st.markdown("##### Sanitized Simulation Execution Trail")
                log_level = st.selectbox("Filter Log Level", ["ALL", "INFO", "WARNING", "ERROR"], key="ecc_log_filter")
                filtered_logs = logs if log_level == "ALL" else [l for l in logs if l["level"] == log_level]
                st.dataframe(pd.DataFrame(filtered_logs), use_container_width=True, hide_index=True)
