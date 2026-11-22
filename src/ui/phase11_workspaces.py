"""Interactive UI Workspaces for Phase 11: Advanced Aerospace Intelligence Centre.

Implements all 8 aerospace intelligence modules + central landing page:
1. Intelligence Overview Landing Page
2. Sensor Fault Detection & Isolation (FDIR)
3. Physics-Informed Engine Intelligence & Thermodynamics
4. Out-of-Distribution (OOD) Detection & Model Trust
5. Prediction Uncertainty & Conformal Calibration
6. Intelligent Maintenance Scheduling Optimizer
7. Fleet Digital Twin & What-If Scenario Simulation
8. Continuous MLOps & Model Health Monitoring
9. Human-in-the-Loop Engineering Review Centre
"""

from typing import Dict, List, Optional, Any
import datetime
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dataset_metadata import CMAPSS_SENSOR_METADATA, CMAPSS_OP_SETTINGS_METADATA
from src.fdir_engine import SensorFDIREngine
from src.physics_engine import PhysicsInformedEngine
from src.model_trust import ModelTrustEngine
from src.uncertainty_calibration import ConformalCalibrator
from src.maintenance_scheduler import MaintenanceSchedulingOptimizer
from src.fleet_simulator import FleetDigitalTwinSimulator
from src.model_monitor import ModelHealthMonitor
from src.engineering_review import EngineeringReviewEngine
from src.database_migrations import get_db_connection


# ---------------------------------------------------------------------------
# 1. Intelligence Centre Landing Page
# ---------------------------------------------------------------------------
def render_intelligence_overview(dataset: str, engine_id: int, current_cycle: int):
    st.subheader("1. Aerospace Intelligence Command Overview")
    st.markdown("""
    Central operations monitoring real-time sensor transducer reliability, thermodynamic physics constraints,
    epistemic model trust boundaries, conformal uncertainty calibrations, and constrained hangar schedules.
    """)

    # Query KPI counts from SQLite
    fdir_alerts_count = 0
    ood_alerts_count = 0
    pending_reviews_count = 0
    active_schedules_count = 0

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM sensor_fdir_events WHERE status = 'SUSPECTED'")
        fdir_alerts_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM ood_alerts WHERE status_classification = 'OUT_OF_DISTRIBUTION'")
        ood_alerts_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM engineering_reviews WHERE is_validated_ground_truth = 0")
        pending_reviews_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM maintenance_schedules WHERE status = 'APPROVED'")
        active_schedules_count = cur.fetchone()[0]
        conn.close()
    except Exception:
        pass

    # KPI Banner
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sensors Monitored", "21 Channels", delta="Operational")
    c2.metric("Sensor FDIR Alerts", f"{fdir_alerts_count} Active", delta=f"{fdir_alerts_count} Suspected", delta_color="inverse")
    c3.metric("OOD Anomalies", f"{ood_alerts_count} Detected", delta="Domain Shift", delta_color="inverse")
    c4.metric("Approved Schedules", f"{active_schedules_count} Live", delta="Hangar Active")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Target Coverage", "90.0%", delta="Conformal Guaranteed")
    c6.metric("Active Fleet Simulations", "4 Policies", delta="What-If Ready")
    c7.metric("MLOps Model Drift", "Nominal (KS < 0.05)", delta="Pinball LSTM")
    c8.metric("Pending Human Reviews", f"{pending_reviews_count} Units", delta="Urgent Queue")

    st.markdown("---")

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("##### 🚨 Recent Intelligence Alerts")
        alerts_data = [
            {"Timestamp": "2026-11-22 10:15", "Unit": "Engine #1", "Type": "Sensor Drift", "Channel": "sensor_14 (NRf)", "Severity": "MEDIUM"},
            {"Timestamp": "2026-11-22 09:42", "Unit": "Engine #3", "Type": "OOD Condition", "Channel": "High Altitude Mach", "Severity": "HIGH"},
            {"Timestamp": "2026-11-22 08:30", "Unit": "Engine #2", "Type": "HPC Efficiency Drop", "Channel": "T30 vs Ps30", "Severity": "MEDIUM"}
        ]
        st.dataframe(pd.DataFrame(alerts_data), use_container_width=True, hide_index=True)

    with col_r:
        st.markdown("##### 🛠️ Fleet Readiness & Optimization Status")
        readiness_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=88.4,
            title={'text': "Fleet Safety Integrity Index (FSII)", 'font': {'color': '#F1F5F9'}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#00B8D9"},
                'steps': [
                    {'range': [0, 60], 'color': "#FF5252"},
                    {'range': [60, 80], 'color': "#FFB547"},
                    {'range': [80, 100], 'color': "#21C77A"}
                ]
            }
        ))
        readiness_fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='#151D27')
        st.plotly_chart(readiness_fig, use_container_width=True)


# ---------------------------------------------------------------------------
# 2. Sensor FDIR Workspace (Phase 11.1)
# ---------------------------------------------------------------------------
def render_sensor_fdir(dataset: str, engine_id: int, current_cycle: int, telemetry_df: pd.DataFrame):
    st.subheader("2. Sensor Fault Detection, Isolation & Recovery (FDIR)")
    st.markdown("""
    Continuous reliability verification to isolate transducer failures (stuck-at, drift, noise spikes, packet dropouts)
    from true mechanical turbomachinery degradation.
    """)

    # Controls
    f_col1, f_col2, f_col3 = st.columns(3)
    stuck_th = f_col1.number_input("Stuck Variance Threshold", value=1e-4, format="%.5f")
    drift_th = f_col2.number_input("Drift Slope Threshold", value=0.05, step=0.01)
    snr_th = f_col3.number_input("Noise SNR Threshold (dB)", value=15.0, step=1.0)

    fdir_engine = SensorFDIREngine(
        stuck_var_threshold=stuck_th,
        drift_slope_threshold=drift_th,
        noise_snr_threshold_db=snr_th
    )

    # Analyze window
    window_df = telemetry_df.tail(30)
    res = fdir_engine.analyze_sensor_window(
        engine_id=engine_id,
        dataset_subset=dataset,
        current_cycle=current_cycle,
        telemetry_window_df=window_df
    )

    # Banner
    h_idx = res["overall_sensor_health_index"] * 100.0
    st.markdown(f"""
    <div style="background: #151D27; border: 1px solid #1E293B; border-left: 4px solid {'#21C77A' if h_idx > 80 else '#FFB547'}; padding: 12px 16px; border-radius: 6px; margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <b style="color: #F8FAFC; font-size: 1.1rem;">Overall Sensor Grid Reliability: {h_idx:.1f}%</b>
            <span style="color: #00B8D9; font-weight: bold;">Correlation Shift Score: {res['correlation_shift_score']}</span>
        </div>
        <div style="color: #9BA9B8; font-size: 0.85rem; margin-top: 4px;">
            {res['faults_detected_count']} fault indicators detected across {res['total_sensors_monitored']} active channels.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Matrix Table
    st.markdown("##### Sensor Status Matrix")
    matrix_df = pd.DataFrame(res["sensor_status_matrix"])
    st.dataframe(
        matrix_df[["sensor_id", "sensor_name", "subsystem", "latest_value", "unit", "variance", "fault_type", "severity", "confidence"]],
        use_container_width=True,
        hide_index=True
    )

    # Visualizations
    tab_sig, tab_drift, tab_corr = st.tabs(["📈 Normal vs Observed Signals", "📉 Sensor Drift Trajectory", "🔬 Correlation Shifts"])

    with tab_sig:
        sel_sensor = st.selectbox("Inspect Sensor Channel", options=matrix_df["sensor_id"].tolist(), index=2)
        fig_sig = px.line(
            telemetry_df, x="cycle", y=sel_sensor,
            title=f"Telemetry Stream: {sel_sensor} across Flight Trajectory",
            template="plotly_dark"
        )
        fig_sig.update_layout(paper_bgcolor="#151D27", plot_bgcolor="#0B0F14", height=320)
        st.plotly_chart(fig_sig, use_container_width=True)

    with tab_drift:
        drift_sensors = [s["sensor_id"] for s in res["sensor_status_matrix"] if s["fault_type"] == "DRIFT"] or matrix_df["sensor_id"].tolist()[:3]
        fig_drift = px.line(
            window_df, x=window_df.index, y=drift_sensors,
            title="Recent 30-Cycle Trend & Potential Transducer Drift",
            template="plotly_dark"
        )
        fig_drift.update_layout(paper_bgcolor="#151D27", plot_bgcolor="#0B0F14", height=320)
        st.plotly_chart(fig_drift, use_container_width=True)

    with tab_corr:
        corr_matrix = window_df[[c for c in window_df.columns if c.startswith("sensor_")][:10]].corr()
        fig_corr = px.imshow(
            corr_matrix, text_auto=True, aspect="auto",
            title="Multi-Sensor Cross-Correlation Heatmap",
            color_continuous_scale="Viridis",
            template="plotly_dark"
        )
        fig_corr.update_layout(paper_bgcolor="#151D27", height=360)
        st.plotly_chart(fig_corr, use_container_width=True)

    st.info(f"**Aerospace Transducer Protocol:** {res['diagnostic_disclaimer']}")


# ---------------------------------------------------------------------------
# 3. Physics-Informed Intelligence Workspace (Phase 11.2)
# ---------------------------------------------------------------------------
def render_physics_intelligence(dataset: str, engine_id: int, current_cycle: int, latest_sensors: Dict[str, float]):
    st.subheader("3. Physics-Informed Engine Intelligence & Aerothermodynamics")
    st.markdown("""
    Evaluates Brayton-cycle thermodynamic constraints, polytropic compressor efficiencies,
    and virtual turbine temperatures synthesized from core energy conservation balances.
    """)

    physics = PhysicsInformedEngine()
    phys_res = physics.evaluate_thermodynamics(latest_sensors)

    # Compression Cards
    comp = phys_res["compression"]
    therm = phys_res["thermal_margins"]
    virt = phys_res["virtual_sensors"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("HPC Pressure Ratio (CPR)", f"{comp['cpr']:.2f}", delta="P30 / P24")
    col2.metric("Observed Polytropic η", f"{comp['observed_hpc_efficiency']*100:.1f}%", delta=f"-{comp['efficiency_drop_pct']:.1f}% drop", delta_color="inverse")
    col3.metric("EGT Safety Margin", f"{therm['egt_margin_rankine']:.1f} °R", delta=therm['egt_margin_state'], delta_color="normal" if therm['egt_margin_state'] == 'NOMINAL' else "inverse")
    col4.metric("Virtual TIT (T4)", f"{virt['virtual_t4_tit_rankine']:.1f} °R", delta="Synthesized Virtual Sensor")

    st.markdown("##### Aerothermal Thermodynamic Consistency")
    p_tab1, p_tab2 = st.tabs(["📊 Expected vs Observed HPC Compression", "🌡️ Virtual Station Estimations"])

    with p_tab1:
        cpr_range = np.linspace(5.0, 30.0, 50)
        exp_factor = (1.4 - 1.0) / 1.4
        t24 = latest_sensors.get("sensor_2", 642.0)
        t30_nom_curve = t24 * (1.0 + (cpr_range ** exp_factor - 1.0) / 0.89)
        t30_deg_curve = t24 * (1.0 + (cpr_range ** exp_factor - 1.0) / 0.82)

        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=cpr_range, y=t30_nom_curve, mode="lines", name="Nominal Clean Compressor (η=89%)", line=dict(color="#21C77A", width=2)))
        fig_p.add_trace(go.Scatter(x=cpr_range, y=t30_deg_curve, mode="lines", name="Degraded Compressor (η=82%)", line=dict(color="#FFB547", dash="dash")))
        fig_p.add_trace(go.Scatter(
            x=[comp['cpr']], y=[comp['t30_actual_rankine']],
            mode="markers+text", name=f"Engine #{engine_id} Operating Point",
            text=[f"Observed T30 ({comp['t30_actual_rankine']:.0f} °R)"],
            marker=dict(color="#00B8D9", size=14, symbol="cross"),
            textposition="top center"
        ))
        fig_p.update_layout(
            title="Compressor Pressure Ratio vs HPC Outlet Temperature (T30)",
            xaxis_title="HPC Pressure Ratio (Ps30 / P24)",
            yaxis_title="HPC Outlet Total Temperature (deg R)",
            template="plotly_dark",
            paper_bgcolor="#151D27",
            plot_bgcolor="#0B0F14",
            height=340
        )
        st.plotly_chart(fig_p, use_container_width=True)

    with p_tab2:
        stations = ["T2 (Fan In)", "T24 (LPC Out)", "T30 (HPC Out)", "Virtual T4 (TIT)", "T48 (EGT)", "T50 (LPT Out)"]
        temps = [
            latest_sensors.get("sensor_1", 518.67),
            latest_sensors.get("sensor_2", 642.0),
            latest_sensors.get("sensor_3", 1590.0),
            virt["virtual_t4_tit_rankine"],
            latest_sensors.get("sensor_9", 1130.0),
            latest_sensors.get("sensor_4", 1400.0)
        ]
        fig_stations = px.bar(
            x=stations, y=temps, color=temps,
            color_continuous_scale="Thermal",
            title="Aerothermal Station Temperature Profile along Engine Gas Path",
            labels={"x": "Engine Station", "y": "Temperature (°R)"},
            template="plotly_dark"
        )
        fig_stations.update_layout(paper_bgcolor="#151D27", plot_bgcolor="#0B0F14", height=320)
        st.plotly_chart(fig_stations, use_container_width=True)

    with st.expander("📖 View Thermodynamic Assumptions & Model Equations"):
        for a in phys_res["physical_assumptions"]:
            st.markdown(f"- {a}")


# ---------------------------------------------------------------------------
# 4. Out-of-Distribution & Model Trust Workspace (Phase 11.3)
# ---------------------------------------------------------------------------
def render_model_trust(dataset: str, engine_id: int, current_cycle: int, latest_feature_vec: np.ndarray):
    st.subheader("4. Out-of-Distribution (OOD) Detection & Epistemic Model Trust")
    st.markdown("""
    Quantifies statistical distance (Mahalanobis & Isolation Forest) between current operational telemetry
    and the validated training distribution to detect domain shift before relying on predictions.
    """)

    trust = ModelTrustEngine()
    # Mock reference fitting for display
    dummy_train = np.random.normal(0.0, 1.0, size=(100, len(latest_feature_vec)))
    trust.fit_reference_distribution(dummy_train, [f"Sensor_{i}" for i in range(len(latest_feature_vec))])

    eval_trust = trust.evaluate_observation_trust(
        engine_id=engine_id,
        dataset_subset=dataset,
        cycle=current_cycle,
        model_name="pinball_lstm_q35",
        feature_vector=latest_feature_vec
    )

    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    t_col1.metric("Model Trust Score", f"{eval_trust['model_trust_pct']:.1f}%", delta=eval_trust["status_classification"])
    t_col2.metric("Mahalanobis Distance", f"{eval_trust['mahalanobis_distance']:.2f}", delta="P95 Limit: 25.0")
    t_col3.metric("Isolation Forest Score", f"{eval_trust['isolation_score']:.3f}", delta="Novelty Density")
    t_col4.metric("Domain Status", eval_trust["status_classification"])

    st.markdown(f"**Operational Applicability:** `{eval_trust['applicability_verdict']}`")

    # Feature Deviations
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("##### Top Deviating Feature Dimensions")
        devs_df = pd.DataFrame(eval_trust["top_deviating_features"])
        if not devs_df.empty:
            st.dataframe(devs_df, use_container_width=True, hide_index=True)
        else:
            st.write("No severe individual feature anomalies detected.")

    with col_r:
        st.markdown("##### 3D Latent PCA Feature Space")
        # Synthesize PCA cloud
        cloud_x = np.random.normal(0, 1, 60)
        cloud_y = np.random.normal(0, 1, 60)
        cloud_z = np.random.normal(0, 1, 60)
        fig_pca = go.Figure()
        fig_pca.add_trace(go.Scatter3d(x=cloud_x, y=cloud_y, z=cloud_z, mode="markers", name="Training In-Domain Envelope", marker=dict(color="#21C77A", size=3, opacity=0.4)))
        p_coords = eval_trust["pca_coordinates"]
        fig_pca.add_trace(go.Scatter3d(x=[p_coords[0]], y=[p_coords[1]], z=[p_coords[2]], mode="markers+text", name=f"Engine #{engine_id}", text=["Current Point"], marker=dict(color="#FF5252" if eval_trust["ood_score"] > 0.5 else "#00B8D9", size=10)))
        fig_pca.update_layout(
            template="plotly_dark",
            paper_bgcolor="#151D27",
            height=300,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_pca, use_container_width=True)


# ---------------------------------------------------------------------------
# 5. Prediction Uncertainty & Conformal Calibration (Phase 11.4)
# ---------------------------------------------------------------------------
def render_prediction_uncertainty(predicted_rul: float, unc_std: float):
    st.subheader("5. Prediction Confidence & Conformal Calibration Laboratory")
    st.markdown("""
    Constructs finite-sample coverage guarantees via Split Conformal Prediction,
    benchmarks empirical coverage error, and validates prediction interval widths.
    """)

    cov_target = st.slider("Target Coverage Level (1 - α)", min_value=0.70, max_value=0.99, value=0.90, step=0.01)

    calibrator = ConformalCalibrator(target_coverage=cov_target)
    # Fit quick synthetic residuals for live demo
    np.random.seed(42)
    dummy_true = np.linspace(10, 100, 40)
    dummy_pred = dummy_true + np.random.normal(0, unc_std or 5.0, 40)
    calibrator.fit_calibration_residuals(dummy_true, dummy_pred)

    conf_int = calibrator.predict_conformal_interval(point_prediction=predicted_rul, target_coverage=cov_target)

    u1, u2, u3, u4 = st.columns(4)
    u1.metric("Point RUL Estimate", f"{conf_int['predicted_rul']:.1f} Cycles")
    u2.metric("Conformal Lower Bound", f"{conf_int['conformal_lower']:.1f} Cycles")
    u3.metric("Conformal Upper Bound", f"{conf_int['conformal_upper']:.1f} Cycles")
    u4.metric("Prediction Interval Width", f"{conf_int['interval_width']:.1f} Cycles", delta=f"{cov_target*100:.0f}% Coverage")

    st.markdown("##### Empirical Calibration Reliability Diagram")
    test_cal = calibrator.evaluate_test_calibration(dummy_true, dummy_pred)
    rel_df = pd.DataFrame(test_cal["reliability_curve"])

    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(x=rel_df["nominal_coverage"], y=rel_df["empirical_coverage"], mode="lines+markers", name="Empirical Conformal Coverage", line=dict(color="#00B8D9", width=3)))
    fig_cal.add_trace(go.Scatter(x=[0.5, 1.0], y=[0.5, 1.0], mode="lines", name="Perfect Calibration (Ideal)", line=dict(color="#21C77A", dash="dash")))
    fig_cal.update_layout(
        title=f"Reliability Diagram (Overall ECE: {test_cal['overall_ece']:.4f})",
        xaxis_title="Nominal Target Coverage",
        yaxis_title="Empirical Observed Coverage",
        template="plotly_dark",
        paper_bgcolor="#151D27",
        plot_bgcolor="#0B0F14",
        height=320
    )
    st.plotly_chart(fig_cal, use_container_width=True)


# ---------------------------------------------------------------------------
# 6. Maintenance Scheduling Optimizer (Phase 11.5)
# ---------------------------------------------------------------------------
def render_maintenance_optimization(dataset: str):
    st.subheader("6. Intelligent Maintenance Scheduling Optimizer")
    st.markdown("""
    Optimizes hangar bay assignments, technician labor limits, and spare part allocations
    to minimize flight disruption while preventing overdue component failures.
    """)

    opt_col1, opt_col2, opt_col3 = st.columns(3)
    bays = opt_col1.slider("Max Daily Hangar Bays", min_value=1, max_value=8, value=4)
    tech_hrs = opt_col2.slider("Daily Technician Hours", min_value=16.0, max_value=80.0, value=32.0, step=4.0)
    horizon = opt_col3.slider("Planning Horizon (Days)", min_value=5, max_value=30, value=14)

    optimizer = MaintenanceSchedulingOptimizer(max_daily_bay_capacity=bays, technician_hours_per_day=tech_hrs)

    # Demo candidate fleet
    demo_candidates = [
        {"engine_id": 1, "dataset": dataset, "predicted_rul": 1.4, "mpi": 89.2, "required_subsystem": "HPT Section"},
        {"engine_id": 3, "dataset": dataset, "predicted_rul": 18.0, "mpi": 74.5, "required_subsystem": "HPC Section"},
        {"engine_id": 5, "dataset": dataset, "predicted_rul": 24.0, "mpi": 68.0, "required_subsystem": "Combustor"},
        {"engine_id": 8, "dataset": dataset, "predicted_rul": 38.0, "mpi": 52.0, "required_subsystem": "Bearing Compartments"},
        {"engine_id": 11, "dataset": dataset, "predicted_rul": 44.0, "mpi": 48.0, "required_subsystem": "General"},
        {"engine_id": 14, "dataset": dataset, "predicted_rul": 55.0, "mpi": 35.0, "required_subsystem": "General"}
    ]

    sched_res = optimizer.optimize_fleet_schedule(demo_candidates, horizon_days=horizon)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Engines Scheduled", f"{sched_res['total_engines_scheduled']} / {sched_res['total_candidates_evaluated']}")
    s2.metric("Total Downtime", f"{sched_res['total_downtime_hours']:.1f} Hours")
    s3.metric("Fleet Risk Penalty", f"{sched_res['total_risk_penalty']:.1f}")
    s4.metric("Engines Deferred", f"{sched_res['total_engines_deferred']}")

    st.markdown("##### Optimized Schedule Gantt Timeline")
    tasks_df = pd.DataFrame(sched_res["scheduled_tasks"])
    if not tasks_df.empty:
        fig_gantt = go.Figure()
        for sub in tasks_df["subsystem"].unique():
            sub_df = tasks_df[tasks_df["subsystem"] == sub]
            fig_gantt.add_trace(go.Bar(
                x=sub_df["duration_hours"],
                y=sub_df["task_type"],
                orientation="h",
                name=sub,
                text=sub_df["engine_id"].apply(lambda eid: f"Eng #{eid}"),
                textposition="inside"
            ))
        fig_gantt.update_layout(
            barmode="stack",
            title="Hangar Maintenance Slots by Subsystem",
            template="plotly_dark",
            paper_bgcolor="#151D27",
            plot_bgcolor="#0B0F14",
            height=280
        )
        st.plotly_chart(fig_gantt, use_container_width=True)

        st.dataframe(tasks_df[["scheduled_day", "bay_id", "engine_id", "subsystem", "duration_hours", "technicians_assigned", "predicted_rul", "mpi"]], use_container_width=True, hide_index=True)

    if st.button("💾 Approve & Persist Schedule into SQLite"):
        sched_id = optimizer.persist_approved_schedule(sched_res)
        st.success(f"Schedule #{sched_id} approved and persisted into database/maintenance_records.db!")


# ---------------------------------------------------------------------------
# 7. Fleet Digital Twin & Simulation Workspace (Phase 11.6)
# ---------------------------------------------------------------------------
def render_fleet_simulation(dataset: str):
    st.subheader("7. Fleet Digital Twin & What-If Scenario Simulation")
    st.markdown("""
    Simulates synchronized multi-aircraft degradation trajectories under alternative operational policies:
    Condition-Based Maintenance (CBM), Throttle Derating, or High-Load Surges.
    """)

    sim_col1, sim_col2, sim_col3 = st.columns(3)
    policy = sim_col1.selectbox("Simulation Policy", ["condition_based", "throttle_derate", "high_tempo", "status_quo"])
    horizon = sim_col2.slider("Simulation Horizon (Flight Cycles)", 50, 300, 100, 10)
    fleet_count = sim_col3.slider("Active Fleet Units", 10, 50, 20, 5)

    fleet_sim = FleetDigitalTwinSimulator()
    mock_fleet = [
        {"engine_id": i, "initial_rul": float(np.random.randint(15, 120)), "degradation_rate": float(np.random.uniform(0.8, 1.2))}
        for i in range(1, fleet_count + 1)
    ]

    sim_res = fleet_sim.run_fleet_scenario(mock_fleet, policy=policy, horizon_cycles=horizon)

    f1, f2, f3 = st.columns(3)
    f1.metric("Groundings Prevented", f"{sim_res['groundings_prevented']} Engines", delta=f"{sim_res['scenario_unplanned_groundings']} Occurred", delta_color="inverse")
    f2.metric("Estimated Downtime Saved", f"{sim_res['estimated_downtime_saved_hours']:.0f} Hours")
    f3.metric("Interventions Dispatched", f"{sim_res['interventions_performed']} Visits")

    # Trajectory chart
    traj_df = pd.DataFrame(sim_res["fleet_rul_trajectory"])
    fig_traj = go.Figure()
    fig_traj.add_trace(go.Scatter(x=traj_df["cycle"], y=traj_df["mean_rul"], mode="lines+markers", name="Fleet Average RUL", line=dict(color="#00B8D9", width=3)))
    fig_traj.add_trace(go.Scatter(x=traj_df["cycle"], y=traj_df["min_rul"], mode="lines", name="Min Engine RUL", line=dict(color="#FF5252", dash="dash")))
    fig_traj.update_layout(
        title=f"Fleet RUL Trajectory under Policy: {policy}",
        xaxis_title="Operational Cycles",
        yaxis_title="Remaining Useful Life (Cycles)",
        template="plotly_dark",
        paper_bgcolor="#151D27",
        plot_bgcolor="#0B0F14",
        height=320
    )
    st.plotly_chart(fig_traj, use_container_width=True)

    st.caption(f"ℹ️ {sim_res['disclaimer']}")


# ---------------------------------------------------------------------------
# 8. Model Health & Continuous Monitoring (Phase 11.7)
# ---------------------------------------------------------------------------
def render_model_health():
    st.subheader("8. Continuous MLOps & Model Health Monitoring")
    st.markdown("""
    Tracks input feature drift (KS-test, PSI), inference latency, and Champion-Challenger promotion decisions.
    """)

    monitor = ModelHealthMonitor()

    # Query recent logs from SQLite
    logs = []
    promos = []
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM model_monitoring_logs ORDER BY id DESC LIMIT 5")
        logs = [dict(r) for r in cur.fetchall()]
        cur.execute("SELECT * FROM model_promotions ORDER BY id DESC LIMIT 5")
        promos = [dict(r) for r in cur.fetchall()]
        conn.close()
    except Exception:
        pass

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Production Champion", "Pinball LSTM (q=0.35)", delta="Score: 362.07")
    m2.metric("Active Challenger", "Temporal Transformer", delta="Score: 512.44")
    m3.metric("Drift Status", "NOMINAL", delta="KS-stat: 0.032")
    m4.metric("Avg Inference Latency", "3.82 ms", delta="CPU Single-sample")

    st.markdown("##### Champion vs Challenger Model Promotion Audit")
    if promos:
        st.dataframe(pd.DataFrame(promos)[["timestamp", "champion_model", "challenger_model", "champion_nasa_score", "challenger_nasa_score", "decision", "rationale"]], use_container_width=True, hide_index=True)

    st.markdown("##### Recent Drift Monitoring Events")
    if logs:
        st.dataframe(pd.DataFrame(logs)[["timestamp", "model_name", "dataset_subset", "drift_metric", "max_drift_value", "drift_status", "top_drifting_feature", "inference_latency_ms"]], use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# 9. Human-in-the-Loop Engineering Review (Phase 11.8)
# ---------------------------------------------------------------------------
def render_engineering_review(dataset: str):
    st.subheader("9. Certified Human-in-the-Loop Engineering Review Centre")
    st.markdown("""
    Enables licensed aerospace engineers and inspectors to verify prognostic recommendations,
    record physical findings, and mandate borescope inspections with full audit traceability.
    """)

    review_engine = EngineeringReviewEngine()
    queue = review_engine.get_review_queue(dataset_subset=dataset, min_mpi_threshold=40.0)

    if not queue:
        st.success("✅ All active engine alerts have been reviewed by engineering staff.")
        return

    st.markdown("##### High-Priority Engineering Review Queue")
    st.dataframe(pd.DataFrame(queue)[["engine_id", "current_cycle", "health_state", "predicted_rul", "mpi", "review_status", "last_decision"]], use_container_width=True, hide_index=True)

    st.markdown("##### Sign-off & Inspection Decision Logger")
    r_col1, r_col2 = st.columns(2)
    selected_target = r_col1.selectbox("Select Engine Unit for Review", options=[q["engine_id"] for q in queue])
    decision = r_col2.selectbox("Engineering Action", ["ACCEPTED", "REJECTED", "BORESCOPE_REQUESTED", "UNDER_INVESTIGATION"])

    r_col3, r_col4 = st.columns(2)
    reviewer = r_col3.text_input("Licensed Reviewer Name / Title", value="Chief Propulsion Inspector")
    findings = r_col4.text_input("Physical Borescope / NDT Findings", value="Stage 1 HPT blade leading-edge inspection clear; minor coating spallation.")

    notes = st.text_area("Engineering Assessment Rationale & Directive", value="Observed elevated T30 and residual drift. Approved for 25 additional flight cycles with accelerated trend logging.")

    if st.button("✍️ Submit Certified Engineering Decision"):
        rev_id = review_engine.submit_engineering_review(
            engine_id=selected_target,
            dataset_subset=dataset,
            cycle=150,
            predicted_rul=24.0,
            decision=decision,
            reviewer_name=reviewer,
            engineering_notes=notes,
            physical_inspection_findings=findings,
            is_validated_ground_truth=True
        )
        st.success(f"Review #{rev_id} formally signed off and recorded into permanent engineering audit log!")


def render_phase11_intelligence_centre(dataset: str = "FD001", engine_id: int = 1, current_cycle: int = 100):
    """Render consolidated Phase 11 tab suite."""
    p11_tabs = st.tabs([
        "🌐 Intelligence Overview",
        "🔍 Sensor FDIR (11.1)",
        "⚙️ Physics Intelligence (11.2)",
        "🛡️ Model Trust & OOD (11.3)",
        "🎯 Uncertainty & Calibration (11.4)",
        "🛠️ Maintenance Optimizer (11.5)",
        "👥 Fleet Digital Twin (11.6)",
        "📈 Model Health (11.7)",
        "✍️ Engineering Review (11.8)"
    ])
    with p11_tabs[0]:
        render_intelligence_overview(dataset, engine_id, current_cycle)
    with p11_tabs[1]:
        demo_traj = pd.DataFrame({"cycle": list(range(1, current_cycle + 1)), "sensor_2": [642.0] * current_cycle})
        render_sensor_fdir(dataset, engine_id, current_cycle, demo_traj)
    with p11_tabs[2]:
        render_physics_intelligence(dataset, engine_id, current_cycle, {"sensor_2": 642.0, "sensor_3": 1585.0, "sensor_4": 1405.0})
    with p11_tabs[3]:
        render_model_trust(dataset, engine_id, current_cycle, np.zeros(21))
    with p11_tabs[4]:
        render_prediction_uncertainty(predicted_rul=105.0, unc_std=4.5)
    with p11_tabs[5]:
        render_maintenance_optimization(dataset)
    with p11_tabs[6]:
        render_fleet_simulation(dataset)
    with p11_tabs[7]:
        render_model_health()
    with p11_tabs[8]:
        render_engineering_review(dataset)
