"""JETWISE AI — Industrial Operations SCADA & Replay View.

Consolidates the industrial SCADA dashboard tabs:
- Tab 1: Sensor Trajectories
- Tab 2: Engine Health Degradation Curve
- Tab 3: Historical Telemetry Replay Simulation
- Tab 4: 3D Turbofan Digital Twin (Three.js & Plotly)
- Tab 5: Unsupervised Anomaly Drift & Residuals (Isolation Forest)
- Tab 6: Sequence Explainable AI (Integrated Gradients)
- Tab 7: Predictive Maintenance Actions & SQLite Work Orders
- Tab 8: Automated Engineering Diagnostic Dossier
Plus Validation Fleet Evaluation, Model Benchmarks, and Dispatch Alerts.
"""

import datetime
from typing import Any, Dict
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.data_replay import EngineReplaySimulator
from src.predictive_intelligence import (
    EngineAnomalyDetector,
    SequenceFeatureAttribution,
)
from src.digital_twin import (
    COMPONENT_DEFINITIONS,
    generate_turbofan_3d_figure,
    render_threejs_digital_twin_html,
)


def render_operations_scada_view(ctx: Dict[str, Any]):
    """Render the operations SCADA view using the provided execution context."""
    engine_trajectory = ctx.get("engine_trajectory")
    ds_meta = ctx.get("ds_meta", {})
    selected_dataset = ctx.get("selected_dataset", "FD001")
    selected_engine = ctx.get("selected_engine", 1)
    max_observed_cycle = ctx.get("max_observed_cycle", 100)
    rul_clip = ctx.get("rul_clip", 125.0)
    can_predict = ctx.get("can_predict", False)
    window_len = ctx.get("window_len", 30)
    sensor_cols = ctx.get("sensor_cols", [])
    sensor_means = ctx.get("sensor_means", np.zeros(14))
    sensor_stds = ctx.get("sensor_stds", np.ones(14))
    selected_model = ctx.get("selected_model")
    is_3d = ctx.get("is_3d", False)
    predict_rul = ctx.get("predict_rul")
    health_state = ctx.get("health_state", "HEALTHY")
    predicted_rul = ctx.get("predicted_rul", 112.5)
    anom_res = ctx.get("anom_res", {"anomaly_score": 0.05, "severity": "NORMAL", "top_anomalous_sensors": []})
    norm_window = ctx.get("norm_window")
    raw_df = ctx.get("raw_df")
    FEATURE_COLS = ctx.get("FEATURE_COLS", [])
    unc_res = ctx.get("unc_res", {"std_dev": 5.0, "q10": 100.0, "q90": 125.0, "confidence_score": 0.9})
    decision_engine = ctx.get("decision_engine")
    db = ctx.get("db")
    metadata = ctx.get("metadata", {})
    all_engines = ctx.get("all_engines", [1])
    adapter = ctx.get("adapter")
    alert_threshold = ctx.get("alert_threshold", 30)
    metrics_data = ctx.get("metrics_data", {})
    model_display_names = ctx.get("model_display_names", {})

    st.subheader("Operations SCADA, Telemetry & Degradation Trajectory")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Sensor Trajectories",
        "📉 Engine Health Degradation Curve",
        "⏱️ Historical Telemetry Replay Simulation",
        "🌐 3D Turbofan Digital Twin",
        "⚠️ Anomaly Drift & Residuals",
        "🧠 Sequence Explainable AI",
        "🔧 Maintenance Actions & Work Orders",
        "📄 Diagnostic Dossier"
    ])

    # TAB 1: SENSOR TRAJECTORIES
    with tab1:
        available_sensors = [c for c in engine_trajectory.columns if c.startswith("sensor_") or c in ds_meta.get("active_sensors", [])]
        if not available_sensors:
            available_sensors = [c for c in engine_trajectory.columns if c not in ("engine_id", "cycle")]
        selected_sensors = st.multiselect(
            "Select Sensor Measurements to Inspect",
            options=available_sensors,
            default=available_sensors[:4] if len(available_sensors) >= 4 else available_sensors
        )
        if selected_sensors:
            plot_df = engine_trajectory[["cycle"] + selected_sensors].set_index("cycle")
            st.line_chart(plot_df, height=350)
        else:
            st.info("Select one or more sensors from the multiselect box above to view telemetry.")

    # TAB 2: DEGRADATION CURVE
    with tab2:
        degradation_df = engine_trajectory[["cycle"]].copy()
        degradation_df["True_RUL"] = np.clip(max_observed_cycle - degradation_df["cycle"], 0, rul_clip)
        st.line_chart(degradation_df.set_index("cycle"), height=350)
        st.caption(f"Run-to-failure degradation trajectory for Engine #{selected_engine} ({selected_dataset}).")

    # TAB 3: HISTORICAL REPLAY SIMULATION
    with tab3:
        st.markdown("""
        <div class="simulation-banner">
            <strong>⏱️ Mode: Historical Dataset Replay / Simulated Monitoring</strong><br>
            Simulating cycle-by-cycle telemetry stream playback from validated benchmark dataset.
            Enables flight engineers to step through operational cycles and observe real-time RUL degradation.
        </div>
        """, unsafe_allow_html=True)

        sim_col1, sim_col2 = st.columns([1, 2])
        with sim_col1:
            playback_cycle = st.slider(
                "Playback Operational Cycle Scrubber",
                min_value=1,
                max_value=max_observed_cycle,
                value=min(max_observed_cycle, 35),
                step=1,
                help="Step or slide through historical flight cycles to observe telemetry and RUL changes."
            )

            sim_engine = EngineReplaySimulator(dataset_name=selected_dataset)
            sim_engine.load_engine(engine_id=selected_engine, kind="train")
            cycle_info = sim_engine.get_cycle_snapshot(playback_cycle)
            cycle_gt_rul = max_observed_cycle - playback_cycle

            if can_predict and predict_rul is not None:
                sim_window = sim_engine.get_sliding_window(
                    cycle=playback_cycle,
                    window_size=window_len,
                    feature_cols=sensor_cols
                )
                sim_window_norm = (sim_window - sensor_means) / sensor_stds
                sim_pred_rul = predict_rul(selected_model, sim_window_norm, is_3d=is_3d)
                risk_cat, badge_color = EngineReplaySimulator.classify_risk(sim_pred_rul)
            else:
                sim_pred_rul = float(cycle_gt_rul)
                risk_cat, badge_color = EngineReplaySimulator.classify_risk(sim_pred_rul)

            st.metric("Simulated Flight Cycle", f"{playback_cycle} / {max_observed_cycle}")
            st.metric("Real-Time Predicted RUL", f"{sim_pred_rul:.1f} Cycles")
            st.metric("Ground-Truth Remaining Cycles", f"{cycle_gt_rul} Cycles")
            st.markdown(f"**Prognostic Status:** <span style='color:{badge_color}; font-weight:bold;'>{risk_cat}</span>", unsafe_allow_html=True)

        with sim_col2:
            sim_history_cycles = list(range(1, playback_cycle + 1))
            sim_history_gt = [max_observed_cycle - c for c in sim_history_cycles]

            chart_history = pd.DataFrame({
                "Cycle": sim_history_cycles,
                "Ground Truth RUL": sim_history_gt,
            }).set_index("Cycle")

            st.markdown(f"##### Telemetry & RUL Degradation at Cycle {playback_cycle}")
            st.line_chart(chart_history, height=280)

    # TAB 4: 3D DIGITAL TWIN
    with tab4:
        st.markdown("#### Interactive WebGL Turbofan 3D Digital Twin & Thermodynamic Surface Mesh")

        enable_main_3d = st.toggle(
            "🌐 Enable 3D WebGL Digital Twin Rendering",
            value=False,
            key="main_enable_3d_twin",
            help="Turn ON to load the interactive Three.js 3D turbofan engine. Off by default for lightweight 2D performance."
        )

        if not enable_main_3d:
            st.info("ℹ️ **3D Digital Twin is currently hidden.** Turn on the toggle above anytime you want to inspect the 3D WebGL engine model and exploded CAD views.")
            st.markdown("##### 2D Station Telemetry Overview")
            latest_tel = engine_trajectory.iloc[-1].to_dict()
            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            m_c1.metric("Station 24 (LPC Temp)", f"{latest_tel.get('sensor_2', 0):.1f} °R")
            m_c2.metric("Station 30 (HPC Temp)", f"{latest_tel.get('sensor_3', 0):.1f} °R")
            m_c3.metric("Station 50 (EGT)", f"{latest_tel.get('sensor_4', 0):.1f} °R")
            m_c4.metric("Station 30 (HPC Press)", f"{latest_tel.get('sensor_7', 0):.1f} psia")
        else:
            twin_ctrl_col1, twin_ctrl_col2 = st.columns([1, 1])
            with twin_ctrl_col1:
                exploded_val = st.slider(
                    "3D Exploded View Displacement",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                    help="Displace internal engine subassemblies along the longitudinal thrust axis to inspect inner core disks and flame tube."
                )
            with twin_ctrl_col2:
                comp_keys = [
                    ("all", "All Assembled Subassemblies"),
                    ("fan_assembly", "1. Fan Assembly (Intake & Blades)"),
                    ("lpc", "2. Low-Pressure Compressor (Booster)"),
                    ("hpc", "3. High-Pressure Compressor (HPC Stages)"),
                    ("combustor", "4. Combustion Chamber (Flame Tube)"),
                    ("hpt", "5. High-Pressure Turbine (HPT Stages)"),
                    ("lpt", "6. Low-Pressure Turbine (LPT Stages)"),
                    ("exhaust_nozzle", "7. Exhaust Nozzle & Mixer"),
                    ("engine_casing", "8. Outer Nacelle Casing"),
                    ("central_shaft", "9. Central Dual-Spool Shaft Assembly")
                ]
                selected_comp_tuple = st.selectbox(
                    "Select Subassembly for Detailed Diagnostic Inspection",
                    options=comp_keys,
                    format_func=lambda x: x[1]
                )
                selected_comp_key = None if selected_comp_tuple[0] == "all" else selected_comp_tuple[0]

            twin_c1, twin_c2 = st.columns([3, 2])
            with twin_c1:
                latest_tel = engine_trajectory.iloc[-1].to_dict()
                three_html = render_threejs_digital_twin_html(
                    engine_id=selected_engine,
                    cycle=max_observed_cycle,
                    predicted_rul=predicted_rul,
                    health_state=health_state,
                    mpi=max(0.0, min(100.0, (125.0 - predicted_rul) * 0.7)),
                    telemetry=latest_tel,
                    anomaly_info=anom_res,
                    max_cycles=max_observed_cycle
                )
                components.html(three_html, height=580)

            with twin_c2:
                st.markdown("##### Aerothermal Heat-Map & Station Inspection Points")
                twin_fig = generate_turbofan_3d_figure(
                    telemetry=latest_tel,
                    health_state=health_state,
                    predicted_rul=predicted_rul,
                    mpi=max(0.0, min(100.0, (125.0 - predicted_rul) * 0.7)),
                    highlight_component=selected_comp_key,
                    exploded_fraction=exploded_val
                )
                st.plotly_chart(twin_fig, use_container_width=True)

        # Component Diagnostic Card with Disclaimer
        inspect_key = selected_comp_key if enable_main_3d and 'selected_comp_key' in locals() and selected_comp_key else "hpc"
        c_meta = COMPONENT_DEFINITIONS.get(inspect_key, COMPONENT_DEFINITIONS["hpc"])
        st.markdown(f"""
        <div style="background: #151D27; border: 1px solid #1E293B; border-left: 4px solid #00B8D9; border-radius: 6px; padding: 14px 18px; margin-top: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <b style="color: #00B8D9; font-size: 1.05rem;">{c_meta['name']} ({c_meta['station']})</b>
                <span style="color: #94A3B8; font-size: 0.85rem;">Subsystem: {c_meta['subsystem']}</span>
            </div>
            <div style="margin-top: 6px; font-size: 0.88rem; color: #E2E8F0;">
                <strong>Mapped Physical Sensors:</strong> {c_meta['sensor_descriptions']}<br>
                <strong>Primary Degradation Mode:</strong> {c_meta['degradation_mode']}<br>
                <strong>Inferred Behavioral Indicators:</strong> {c_meta['inferred_indicators']}
            </div>
            <div style="margin-top: 8px; font-size: 0.78rem; color: #FFB547; background: rgba(255, 181, 71, 0.1); padding: 6px 10px; border-radius: 4px; border: 1px solid rgba(255, 181, 71, 0.25);">
                ⚠️ <strong>Diagnostic Verification Note:</strong> Component health status and 3D highlighting are inferred analytical representations derived from sensor correlation dynamics and AI residual predictions. They must not be interpreted as physical teardown damage without physical borescope verification.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # TAB 5: ANOMALY DETECTION
    with tab5:
        st.markdown("#### Unsupervised Sensor Residual Anomaly Detection (Isolation Forest)")
        anom_c1, anom_c2 = st.columns(2)
        with anom_c1:
            if can_predict and norm_window is not None and raw_df is not None and all(c in raw_df.columns for c in FEATURE_COLS):
                detector = EngineAnomalyDetector(contamination=0.05)
                detector.fit(raw_df[raw_df["cycle"] <= 25], FEATURE_COLS)
                anom_res = detector.detect_window_anomaly(norm_window)
                st.metric("Anomaly Score", f"{anom_res['anomaly_score']:.3f} / 1.000")
                st.metric("Severity", anom_res["severity"])
            else:
                st.info(f"Anomaly detector calibrated for NASA C-MAPSS feature space. Active dataset: `{selected_dataset}`.")
                st.metric("Anomaly Score", f"{anom_res['anomaly_score']:.3f} / 1.000")
                st.metric("Severity", anom_res["severity"])
        with anom_c2:
            st.markdown("##### Outlier Sensor Deviations (Z-Scores)")
            if anom_res.get("top_anomalous_sensors"):
                st.table(pd.DataFrame(anom_res["top_anomalous_sensors"]))
            else:
                st.success("All sensors are operating within nominal baseline bounds (|z| < 1.5).")

    # TAB 6: EXPLAINABLE AI
    with tab6:
        st.markdown("#### Sequence Feature Attribution (Integrated Gradients)")
        if is_3d and can_predict and norm_window is not None and all(c in engine_trajectory.columns for c in FEATURE_COLS):
            xai = SequenceFeatureAttribution(selected_model, FEATURE_COLS)
            xai_out = xai.attribute_window(norm_window, num_steps=12)
            attr_df = pd.DataFrame(xai_out["sensor_attributions"])
            st.bar_chart(attr_df.set_index("sensor")["importance_pct"], height=320)
            st.markdown(f"**Top Physical Drivers of Degradation:** {', '.join(xai_out['top_3_drivers'])}")
        elif not can_predict:
            st.info(f"Feature attribution is available for models matching active dataset schema (`{selected_dataset}` schema differs).")
        else:
            st.info("Feature attribution is enabled for deep neural sequence models.")

    # TAB 7: MAINTENANCE ACTIONS & WORK ORDERS
    with tab7:
        st.markdown("#### Aerospace Predictive Maintenance Intelligence & Planning Centre")

        if decision_engine is not None:
            m_eval = decision_engine.evaluate_health(
                predicted_rul=predicted_rul,
                uncertainty_std=unc_res["std_dev"],
                anomaly_score=anom_res["anomaly_score"],
                current_cycle=max_observed_cycle,
                degradation_rate=1.2,
                max_sensor_zscore=float(anom_res["top_anomalous_sensors"][0]["z_score"]) if anom_res.get("top_anomalous_sensors") else 1.2,
                engine_id=selected_engine
            )
        else:
            m_eval = {"recommended_actions": [], "score_breakdown": {}}

        m_subtab1, m_subtab2, m_subtab3, m_subtab4 = st.tabs([
            "📋 Traceable Recommendations",
            "🛠️ Maintenance Planning & Work Orders",
            "🚨 Active Alerts Center",
            "🌐 Fleet Prioritization Matrix"
        ])

        # 1. Traceable Recommendations
        with m_subtab1:
            st.markdown(f"##### Evidence-Traceable Recommendations for Engine #{selected_engine}")
            for rec in m_eval.get("recommended_actions", []):
                st.markdown(f"""
                <div style="background: #151D27; border: 1px solid #1E293B; border-left: 4px solid {'#FF5252' if rec.get('maintenance_priority') in ('URGENT', 'HIGH') else '#00B8D9'}; padding: 12px 16px; margin-bottom: 10px; border-radius: 6px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <b style="color: #F8FAFC; font-size: 1rem;">{rec.get('recommended_action', rec.get('action'))}</b>
                        <span style="font-size: 0.8rem; font-weight: bold; color: {'#FF5252' if rec.get('maintenance_priority') in ('URGENT', 'HIGH') else '#21C77A'}; border: 1px solid currentColor; padding: 2px 6px; border-radius: 4px;">{rec.get('maintenance_priority')}</span>
                    </div>
                    <div style="color: #00B8D9; font-size: 0.85rem; margin-top: 4px;">Subsystem: {rec.get('subsystem')} | Risk: {rec.get('risk_classification')}</div>
                    <div style="color: #CBD5E1; font-size: 0.85rem; margin-top: 6px;">
                        <strong>Supporting Diagnostic Evidence:</strong> {rec.get('supporting_evidence')}<br>
                        <strong>Engineering Rationale:</strong> {rec.get('reason')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("##### Explainable Maintenance Priority Index Breakdown")
            sb = m_eval.get("score_breakdown", {})
            sb_col1, sb_col2, sb_col3, sb_col4, sb_col5 = st.columns(5)
            sb_col1.metric("RUL Impact (35%)", f"{sb.get('rul_subscore', 0):.1f}")
            sb_col2.metric("Uncertainty (15%)", f"{sb.get('uncertainty_subscore', 0):.1f}")
            sb_col3.metric("Anomaly (20%)", f"{sb.get('anomaly_subscore', 0):.1f}")
            sb_col4.metric("Degradation (15%)", f"{sb.get('degradation_rate_subscore', 0):.1f}")
            sb_col5.metric("Sensor Z-Score (15%)", f"{sb.get('sensor_abnormality_subscore', 0):.1f}")

        # 2. Maintenance Planning & Work Orders
        with m_subtab2:
            st.markdown("##### Maintenance Work Order Management (SQLite Persistent)")
            wo_action_col1, wo_action_col2 = st.columns([2, 1])

            with wo_action_col1:
                status_filter = st.selectbox(
                    "Filter Work Orders by Status",
                    ["All", "Pending", "Inspection Scheduled", "Under Inspection", "Maintenance In Progress", "Completed", "Deferred"]
                )
                filter_arg = None if status_filter == "All" else status_filter
                all_wo = db.get_work_orders(dataset_subset=selected_dataset, status=filter_arg) if db else []

                if all_wo:
                    wo_df = pd.DataFrame(all_wo)[["id", "engine_id", "priority", "title", "subsystem", "status", "assigned_to", "created_at"]]
                    st.dataframe(wo_df, use_container_width=True, height=240)

                    # CSV Export
                    csv_bytes = wo_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Export Work Orders (.csv)",
                        data=csv_bytes,
                        file_name=f"JETWISE_Work_Orders_{selected_dataset}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info(f"No work orders found with status '{status_filter}'.")

            with wo_action_col2:
                st.markdown("###### Update Work Order Status")
                if all_wo and db:
                    wo_ids = [w["id"] for w in all_wo]
                    target_wo_id = st.selectbox("Select Order ID", wo_ids)
                    new_status = st.selectbox(
                        "New Operational Status",
                        ["Inspection Scheduled", "Under Inspection", "Maintenance In Progress", "Completed", "Deferred"]
                    )
                    engineer_note = st.text_input("Engineering Log Note", "Routine inspection logged.")
                    if st.button("Update Order"):
                        db.update_work_order(target_wo_id, status=new_status, notes=engineer_note)
                        st.success(f"Work order #{target_wo_id} updated to {new_status}!")
                        st.rerun()

            # Create New Work Order Form
            with st.expander("➕ Create New Aircraft Maintenance Work Order"):
                with st.form("create_wo_form"):
                    f_title = st.text_input("Work Order Title", f"Scheduled Core Borescope // Unit #{selected_engine}")
                    f_subsystem = st.selectbox("Subsystem", ["High-Pressure Turbine", "High-Pressure Compressor", "Combustor Chamber", "Fan Assembly", "Bearing Compartment", "Exhaust Nozzle"])
                    f_priority = st.selectbox("Priority", ["LOW", "ROUTINE", "MEDIUM", "HIGH", "URGENT"])
                    f_assignee = st.text_input("Assigned Inspection Lead", "Aerospace Propulsion Team A")
                    f_desc = st.text_area("Scope & Supporting Evidence", f"Diagnostic indicators at cycle {max_observed_cycle} show RUL forecast of {predicted_rul:.1f} cycles.")
                    if st.form_submit_button("Submit Work Order"):
                        if db:
                            new_id = db.create_work_order(
                                engine_id=selected_engine,
                                dataset_subset=selected_dataset,
                                priority=f_priority,
                                title=f_title,
                                subsystem=f_subsystem,
                                description=f_desc,
                                assigned_to=f_assignee
                            )
                            st.success(f"Work Order #{new_id} created and persisted in SQLite!")
                            st.rerun()

        # 3. Active Alerts Center
        with m_subtab3:
            st.markdown("##### Active Flight & Health Telemetry Alerts")
            active_alerts = db.get_alerts(dataset_subset=selected_dataset, status="ACTIVE") if db else []
            if active_alerts and db:
                alert_df = pd.DataFrame(active_alerts)[["id", "timestamp", "engine_id", "alert_type", "severity", "message"]]
                st.dataframe(alert_df, use_container_width=True, height=220)

                act_col1, act_col2 = st.columns(2)
                with act_col1:
                    sel_alert_id = st.selectbox("Select Alert ID to Action", [a["id"] for a in active_alerts])
                with act_col2:
                    btn_c1, btn_c2 = st.columns(2)
                    with btn_c1:
                        if st.button("Acknowledge Alert"):
                            db.acknowledge_alert(sel_alert_id)
                            st.info(f"Alert #{sel_alert_id} acknowledged.")
                            st.rerun()
                    with btn_c2:
                        if st.button("Resolve Alert"):
                            db.resolve_alert(sel_alert_id)
                            st.success(f"Alert #{sel_alert_id} resolved.")
                            st.rerun()
            else:
                st.success("No active critical telemetry alerts. Fleet is operating within normal limits.")

        # 4. Fleet Prioritization Matrix
        with m_subtab4:
            st.markdown("##### Fleet-Wide Maintenance Prioritization Matrix (Top Units)")
            fleet_summary = db.get_fleet_summary(dataset_subset=selected_dataset) if db else {}
            top_engines = fleet_summary.get("top_priority_engines", [])
            if top_engines:
                fleet_rank_df = pd.DataFrame(top_engines)
                st.dataframe(fleet_rank_df, use_container_width=True, height=300)
            else:
                st.info("Fleet summary records loading from database...")

    # TAB 8: DIAGNOSTIC DOSSIER
    with tab8:
        st.markdown("#### Automated Engineering Diagnostic Summary")
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        report_text = f"""
================================================================================
JETWISE AI // AIRCRAFT ENGINE HEALTH & PROGNOSTIC DIAGNOSTIC REPORT
Generated: {now_str}
================================================================================

1. ENGINE IDENTIFICATION:
   - Aircraft Unit: Engine #{selected_engine}
   - Operational Fleet: {selected_dataset}
   - Operating Conditions: {ds_meta.get('conditions_count', 1)} profile(s)
   - Fault Category: {', '.join(ds_meta.get('fault_modes', ['Single']))}
   - Total Operational Cycles: {max_observed_cycle}

2. PROGNOSTIC HEALTH STATE:
   - Status: {health_state}
   - Predicted Remaining Useful Life: {predicted_rul:.1f} Cycles
   - 90% Prediction Interval: [{unc_res['q10']:.1f} to {unc_res['q90']:.1f}] Cycles
   - Model Epistemic Confidence: {unc_res.get('confidence_score', 0.8) * 100:.1f}%

3. PRESCRIPTIVE ACTIONS:
"""
        for rec in m_eval.get("recommended_actions", []):
            subsys = rec.get("subsystem", "General")
            act = rec.get("action", rec.get("recommended_action", "Inspection"))
            rat = rec.get("rationale", rec.get("reason", "Prognostic risk indicator"))
            report_text += f"   * [{subsys}] {act}\n     Rationale: {rat}\n"

        st.text_area("Engineering Diagnostic Dossier", report_text, height=280)
        st.download_button(
            "📥 Download Diagnostic Dossier (.txt)",
            data=report_text,
            file_name=f"JETWISE_Engine_{selected_engine}_Diagnostic.txt",
            mime="text/plain"
        )

    st.markdown("---")

    # Section 4: Validation Fleet Evaluation
    st.subheader("Validation Fleet Evaluation (Actual vs Predicted)")
    val_ids = metadata.get("val_engine_ids", [])[:20] if selected_dataset == "FD001" else all_engines[:20]
    val_preds = []
    val_actuals = []
    engine_records = []

    if can_predict and predict_rul is not None and adapter is not None:
        for eid in val_ids:
            e_df = adapter.get_engine_trajectory(eid, kind="train")
            if len(e_df) < window_len:
                continue
            w = e_df[sensor_cols].iloc[-window_len:].values.astype(np.float32)
            w_norm = (w - sensor_means) / sensor_stds
            p = predict_rul(selected_model, w_norm, is_3d=is_3d)
            act = min(e_df["cycle"].max() - e_df["cycle"].iloc[-1], rul_clip)

            val_preds.append(p)
            val_actuals.append(act)
            engine_records.append({
                "Engine ID": eid,
                "Actual RUL": act,
                "Predicted RUL": round(p, 1),
                "Error (Cycles)": round(p - act, 1),
                "Maintenance Alert": "🚨 REQUIRED" if p <= alert_threshold else "✅ OK"
            })

        val_table_df = pd.DataFrame(engine_records)

        vcol1, vcol2 = st.columns([2, 1])
        with vcol1:
            chart_data = pd.DataFrame({
                "Engine ID": [f"Unit #{r['Engine ID']}" for r in engine_records],
                "Actual RUL": val_actuals,
                "Predicted RUL": val_preds
            }).set_index("Engine ID")
            st.bar_chart(chart_data, height=320)
        with vcol2:
            st.dataframe(val_table_df[["Engine ID", "Predicted RUL", "Maintenance Alert"]], height=320, use_container_width=True)
    else:
        st.info("Validation fleet evaluation available for models matching active dataset schema.")

    st.markdown("---")

    # Section 5: Model Comparison & Benchmark
    st.subheader("Baseline vs. Deep Prognostic Architecture Benchmark")
    if metrics_data:
        summary_rows = []
        for m_name in ["linear_regression", "random_forest", "gradient_boosting", "lstm"]:
            if m_name in metrics_data:
                s_met = metrics_data[m_name].get("sequence_level", {})
                e_met = metrics_data[m_name].get("per_engine", {})
                summary_rows.append({
                    "Model": model_display_names.get(m_name, m_name),
                    "Seq MAE": s_met.get("MAE", "-"),
                    "Seq RMSE": s_met.get("RMSE", "-"),
                    "Seq NASA Score": s_met.get("NASA_score", "-"),
                    "Engine MAE": e_met.get("MAE", "-"),
                    "Engine RMSE": e_met.get("RMSE", "-"),
                    "Engine NASA Score": e_met.get("NASA_score", "-")
                })
        st.table(pd.DataFrame(summary_rows))
    else:
        st.info("Performance metrics file not found.")

    # Section 6: Maintenance Dispatch Alerts
    st.subheader("Active Fleet Maintenance Dispatch Alerts")
    urgent_engines = [r for r in engine_records if r["Predicted RUL"] <= alert_threshold]
    if urgent_engines:
        st.warning(f"⚠️ {len(urgent_engines)} units in the fleet require immediate hangar servicing!")
        st.dataframe(pd.DataFrame(urgent_engines), use_container_width=True)
    else:
        st.success(f"All inspected fleet engines have predicted RUL above current safety threshold ({alert_threshold} cycles).")
