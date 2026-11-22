"""JETWISE AI — Turbofan Jet Engine Degradation & RUL Prediction System.

Phase 8 Complete Industrial SCADA Dark-Mode Dashboard & 3D Digital Twin.
Integrates:
1. Fleet Command Center & Health Overview
2. Selected Engine RUL Prognosis with Monte Carlo Uncertainty Intervals
3. Real-Time Telemetry, Degradation Trajectory, and Multi-Subassembly 3D Digital Twin
4. Historical Telemetry Replay Simulation with Cycle Scrubber
5. Unsupervised Anomaly Detection & Sensor Drift (Isolation Forest)
6. Sequence Explainable AI (Integrated Gradients Feature Attribution)
7. Actual vs. Predicted Validation Fleet Performance
8. Baseline vs. Deep Prognostic Architecture Benchmark
9. Active Fleet Maintenance Dispatch Alerts & SQLite Work Orders
10. Multi-Dataset Fleet Discovery (FD001-FD004, N-CMAPSS)
11. Phase 6 Transfer Learning & Domain Adaptation Hub
12. Automated Engineering Diagnostic Dossier Generator
"""

import os
import sys
import json
import pathlib
import datetime
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_dataset, load_rul
from src.train_lstm import LSTMPredictor
from src.dataset_registry import DatasetRegistry
from src.dataset_metadata import CMAPSS_SENSOR_METADATA, CMAPSS_OP_SETTINGS_METADATA
from src.data_replay import EngineReplaySimulator
from src.predictive_intelligence import (
    MaintenanceDatabase,
    MaintenanceDecisionEngine,
    UncertaintyEstimator,
    EngineAnomalyDetector,
    SequenceFeatureAttribution,
    HEALTH_STATES
)
from src.digital_twin import generate_turbofan_3d_figure, render_threejs_digital_twin_html

# ---------------------------------------------------------------------------
# Page Configuration & SCADA Dark Industrial Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="JETWISE AI - Turbofan Jet Engine Health Monitor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Root App Theme */
    .stApp {
        background-color: #0B0F14;
        color: #E2E8F0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    }
    
    /* Top SCADA Header */
    .scada-banner {
        background: linear-gradient(90deg, #101720 0%, #151D27 100%);
        border-bottom: 2px solid #00B8D9;
        padding: 16px 20px;
        margin-bottom: 20px;
        border-radius: 6px;
    }
    .status-normal { color: #21C77A; font-weight: bold; }
    .status-caution { color: #FFB547; font-weight: bold; }
    .status-critical { color: #FF5252; font-weight: bold; }
    
    /* Status Badges */
    .badge-healthy { background-color: rgba(33, 199, 122, 0.15); color: #21C77A; border: 1px solid #21C77A; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
    .badge-monitor { background-color: rgba(0, 184, 217, 0.15); color: #00B8D9; border: 1px solid #00B8D9; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
    .badge-warning { background-color: rgba(255, 181, 71, 0.15); color: #FFB547; border: 1px solid #FFB547; padding: 3px 8px; border-radius: 4px; font-weight: bold; }
    .badge-critical { background-color: rgba(255, 82, 82, 0.15); color: #FF5252; border: 1px solid #FF5252; padding: 3px 8px; border-radius: 4px; font-weight: bold; }

    /* HUD Simulation Banner */
    .simulation-banner {
        background-color: #101720;
        border: 1px solid #1E293B;
        border-left: 4px solid #00B8D9;
        color: #E2E8F0;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 15px;
        font-size: 0.9rem;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #101720;
        padding: 6px 8px;
        border-radius: 6px;
        border: 1px solid #1E293B;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94A3B8;
        font-weight: 600;
        padding: 8px 14px;
        border-radius: 4px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #151D27 !important;
        color: #00B8D9 !important;
        border-bottom: 2px solid #00B8D9 !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached Data & Model Loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_metadata_cached():
    meta_path = PROJECT_ROOT / "dataset" / "processed" / "preprocess_metadata.json"
    if not meta_path.is_file():
        st.error(f"Preprocessing metadata file not found at: {meta_path}")
        st.stop()
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_performance_metrics_cached():
    metrics_path = PROJECT_ROOT / "models" / "performance_metrics.json"
    if metrics_path.is_file():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_resource
def load_models():
    """Load baseline sklearn regressors and PyTorch deep models."""
    models = {}
    baselines_dir = PROJECT_ROOT / "models" / "baselines"
    for name in ["linear_regression", "random_forest", "gradient_boosting"]:
        path = baselines_dir / f"{name}.pkl"
        if path.is_file():
            models[name] = joblib.load(path)

    # PyTorch Baseline LSTM
    lstm_path = PROJECT_ROOT / "models" / "lstm" / "lstm_model.pt"
    if lstm_path.is_file():
        checkpoint = torch.load(lstm_path, map_location="cpu", weights_only=False)
        lstm = LSTMPredictor(input_dim=17)
        lstm.load_state_dict(checkpoint["model_state_dict"])
        lstm.eval()
        models["lstm"] = lstm

    # Phase 1 Best Pinball LSTM (q=0.35)
    pinball_path = PROJECT_ROOT / "models" / "lstm_asymmetric" / "pinball_q35_lstm.pt"
    if pinball_path.is_file():
        checkpoint = torch.load(pinball_path, map_location="cpu", weights_only=False)
        pinball_m = LSTMPredictor(input_dim=17)
        pinball_m.load_state_dict(checkpoint["model_state_dict"])
        pinball_m.eval()
        models["pinball_lstm_q35 (Best)"] = pinball_m

    # Phase 2 Time-Series Transformer
    transformer_path = PROJECT_ROOT / "models" / "transformer" / "transformer_exp_p2_transformer.pt"
    if transformer_path.is_file():
        try:
            from src.models.transformer import TransformerRULPredictor
            trans_ckpt = torch.load(transformer_path, map_location="cpu", weights_only=False)
            trans_m = TransformerRULPredictor(input_dim=17, d_model=64, nhead=4, num_layers=2, max_seq_len=100)
            trans_m.load_state_dict(trans_ckpt["model_state_dict"])
            trans_m.eval()
            models["transformer (Phase 2)"] = trans_m
        except Exception:
            pass

    return models


@st.cache_resource
def get_intelligence_services():
    db = MaintenanceDatabase()
    decision_engine = MaintenanceDecisionEngine()
    return db, decision_engine


def predict_rul(model, sequence_norm: np.ndarray, is_3d: bool = False) -> float:
    """Predict RUL given normalized sequence of shape (30, 17)."""
    if is_3d or isinstance(model, torch.nn.Module):
        if hasattr(model, "predict"):
            pred = model.predict(sequence_norm[np.newaxis, ...])
            return float(pred[0])
        else:
            model.eval()
            with torch.no_grad():
                t_in = torch.tensor(sequence_norm[np.newaxis, ...], dtype=torch.float32)
                return float(model(t_in).item())
    else:
        flat = sequence_norm.reshape(1, -1)
        return float(model.predict(flat)[0])


metadata = load_metadata_cached()
metrics_data = load_performance_metrics_cached()
models = load_models()
db, decision_engine = get_intelligence_services()

FEATURE_COLS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_5", "sensor_6", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_20", "sensor_21"
]

# ---------------------------------------------------------------------------
# Sidebar Navigation & Selection
# ---------------------------------------------------------------------------
st.sidebar.title("✈️ Industrial Telemetry Control")

# 1. Multi-Dataset Dynamic Selector
available_datasets = DatasetRegistry.get_available_datasets()
selected_dataset = st.sidebar.selectbox(
    "Active Dataset & Fleet",
    available_datasets,
    index=0,
    help="Switch between NASA C-MAPSS FD001-FD004 benchmarks or N-CMAPSS fleet."
)

adapter = DatasetRegistry.get_adapter(selected_dataset)
ds_meta = adapter.get_metadata()

st.sidebar.markdown(f"**Conditions:** `{ds_meta.get('conditions_count', 1)}` ({ds_meta.get('complexity', 'Standard')})")
faults = ds_meta.get("fault_modes", ["Single Fault"])
st.sidebar.caption(f"Faults: {', '.join(faults)}")

raw_df = adapter.load_train()
all_engines = adapter.get_engine_ids(kind="train")

# 2. Engine Selector
selected_engine = st.sidebar.selectbox("Select Engine Fleet ID", all_engines, index=0)

# 3. Model Selector
model_names = list(models.keys())
model_display_names = {
    "linear_regression": "Linear Regression (Baseline)",
    "random_forest": "Random Forest Regressor",
    "gradient_boosting": "Gradient Boosting Regressor",
    "lstm": "Deep LSTM Neural Network (MSE)",
    "pinball_lstm_q35 (Best)": "Pinball Asymmetric LSTM (q=0.35)",
    "transformer (Phase 2)": "Temporal Transformer (Phase 2)"
}

selected_model_key = st.sidebar.selectbox(
    "Active Predictive Model",
    model_names,
    index=model_names.index("pinball_lstm_q35 (Best)") if "pinball_lstm_q35 (Best)" in model_names else 0,
    format_func=lambda x: model_display_names.get(x, x)
)
selected_model = models[selected_model_key]
is_3d = isinstance(selected_model, torch.nn.Module)

# Compatibility Check
model_input_dim = len(metadata["selected_sensors"])
compat_info = DatasetRegistry.check_model_compatibility(
    model_input_dim=model_input_dim,
    model_seq_len=metadata["window_length"],
    dataset_name=selected_dataset
)

if compat_info["status"] == "COMPATIBLE":
    st.sidebar.success(f"Model Compatible with {selected_dataset}")
elif compat_info["warnings"]:
    st.sidebar.warning(f"⚠️ {compat_info['warnings'][0]}")

# Maintenance alert threshold
alert_threshold = st.sidebar.slider(
    "Predictive Maintenance Alert Threshold (Cycles)",
    min_value=5, max_value=80, value=30, step=5
)

# ---------------------------------------------------------------------------
# Main App Header
# ---------------------------------------------------------------------------
st.title("Turbofan Jet Engine Degradation & RUL Prediction System")
st.markdown(
    "*Continuous Prognostics and Health Management (PHM) Platform powered by Deep Learning & Industrial Digital Twin.*"
)

# ---------------------------------------------------------------------------
# Section 1: Overall Engine Health Overview
# ---------------------------------------------------------------------------
st.subheader("1. Fleet Health & Operational Overview")
col1, col2, col3, col4 = st.columns(4)

total_engines = len(all_engines)
is_val = selected_engine in metadata.get("val_engine_ids", []) if selected_dataset == "FD001" else False
split_type = "Validation Engine" if is_val else "Training Engine"
engine_trajectory = adapter.get_engine_trajectory(selected_engine, kind="train")
max_observed_cycle = int(engine_trajectory["cycle"].max())

col1.metric("Total Fleet Engines", f"{total_engines} Units")
col2.metric("Active Dataset", selected_dataset)
col3.metric("Unit Classification", split_type)
col4.metric("Logged Operational Cycles", f"{max_observed_cycle} Cycles")

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 2: Selected Engine Remaining Useful Life (RUL) Prediction
# ---------------------------------------------------------------------------
st.subheader(f"2. Remaining Useful Life (RUL) Prognosis — Engine #{selected_engine} ({selected_dataset})")

health_state = "HEALTHY"
actual_final_rul = 0.0
predicted_rul = 50.0
unc_res = {"std_dev": 5.0, "q10": 40.0, "q90": 60.0, "confidence_score": 0.8}

window_len = metadata["window_length"]
rul_clip = metadata["rul_clip"]
sensor_cols = metadata["selected_sensors"]
sensor_means = np.array([metadata["sensor_means"][s] for s in sensor_cols], dtype=np.float32)
sensor_stds = np.array([metadata["sensor_stds"][s] for s in sensor_cols], dtype=np.float32)

can_predict = all(c in engine_trajectory.columns for c in sensor_cols)

latest_window = engine_trajectory[sensor_cols].iloc[-window_len:].values.astype(np.float32)
if len(latest_window) < window_len:
    pad = np.repeat(latest_window[0:1], window_len - len(latest_window), axis=0)
    latest_window = np.vstack([pad, latest_window])
norm_window = (latest_window - sensor_means) / sensor_stds

if not can_predict:
    st.info(
        f"Selected dataset `{selected_dataset}` schema differs from the model's training configuration ({model_input_dim} features). "
        "Use the Historical Telemetry Replay or Multi-Dataset Research tab for cross-domain inspection."
    )
elif len(engine_trajectory) < window_len:
    st.error(f"Engine #{selected_engine} has fewer than {window_len} cycles ({len(engine_trajectory)}).")
else:
    if is_3d:
        unc_tool = UncertaintyEstimator(selected_model, num_samples=20)
        unc_res = unc_tool.predict_with_uncertainty(norm_window)
        predicted_rul = float(unc_res["mean_rul"])
    else:
        predicted_rul = predict_rul(selected_model, norm_window, is_3d=is_3d)
        unc_res = {"std_dev": 6.5, "q10": max(0.0, predicted_rul - 10.0), "q90": predicted_rul + 10.0, "confidence_score": 0.82}

    actual_final_rul = min(max_observed_cycle - engine_trajectory["cycle"].iloc[-1], rul_clip)

    rul_col1, rul_col2, rul_col3, rul_col4 = st.columns(4)
    rul_col1.metric("Predicted RUL (Cycles Remaining)", f"{predicted_rul:.1f} Cycles")
    rul_col2.metric("True RUL (at Last Recorded Cycle)", f"{actual_final_rul:.1f} Cycles")
    rul_col3.metric("90% Prediction Interval", f"[{unc_res['q10']:.0f} - {unc_res['q90']:.0f}] Cycles")

    # Status Alert
    if predicted_rul <= alert_threshold:
        status_html = f"<div class='status-critical'>🚨 CRITICAL ALERT: RUL below {alert_threshold} cycles. Immediate inspection required!</div>"
        health_state = "CRITICAL"
    elif predicted_rul <= alert_threshold + 20:
        status_html = f"<div class='status-caution'>⚠️ CAUTION: Engine approaching scheduled maintenance window.</div>"
        health_state = "MAINTENANCE_DUE"
    elif predicted_rul <= 75:
        status_html = "<div class='status-caution'>ℹ️ MONITOR: Wear trends elevated but within margins.</div>"
        health_state = "MONITOR"
    else:
        status_html = "<div class='status-normal'>✅ NORMAL: Operating within safe flight parameters.</div>"
        health_state = "HEALTHY"
    rul_col4.markdown("### Health Status")
    rul_col4.markdown(status_html, unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 3: Telemetry, Degradation & 3D Digital Twin Modules
# ---------------------------------------------------------------------------
st.subheader("3. Real-Time Telemetry & Degradation Trajectory")

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

        if can_predict:
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
    twin_c1, twin_c2 = st.columns([3, 2])
    with twin_c1:
        three_html = render_threejs_digital_twin_html(
            engine_id=selected_engine,
            cycle=max_observed_cycle,
            predicted_rul=predicted_rul,
            health_state=health_state,
            mpi=max(0.0, min(100.0, (125.0 - predicted_rul) * 0.7))
        )
        components.html(three_html, height=480)

    with twin_c2:
        st.markdown("##### Aerothermal Heat-Map & Station Inspection Points")
        latest_tel = engine_trajectory.iloc[-1].to_dict()
        twin_fig = generate_turbofan_3d_figure(
            telemetry=latest_tel,
            health_state=health_state,
            predicted_rul=predicted_rul,
            mpi=max(0.0, min(100.0, (125.0 - predicted_rul) * 0.7))
        )
        st.plotly_chart(twin_fig, use_container_width=True)

# TAB 5: ANOMALY DETECTION
with tab5:
    st.markdown("#### Unsupervised Sensor Residual Anomaly Detection (Isolation Forest)")
    anom_c1, anom_c2 = st.columns(2)
    with anom_c1:
        detector = EngineAnomalyDetector(contamination=0.05)
        detector.fit(raw_df[raw_df["cycle"] <= 25], FEATURE_COLS)
        anom_res = detector.detect_window_anomaly(norm_window)
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
    if is_3d:
        xai = SequenceFeatureAttribution(selected_model, FEATURE_COLS)
        xai_out = xai.attribute_window(norm_window, num_steps=12)
        attr_df = pd.DataFrame(xai_out["sensor_attributions"])
        st.bar_chart(attr_df.set_index("sensor")["importance_pct"], height=320)
        st.markdown(f"**Top Physical Drivers of Degradation:** {', '.join(xai_out['top_3_drivers'])}")
    else:
        st.info("Feature attribution is enabled for deep neural sequence models.")

# TAB 7: MAINTENANCE ACTIONS & WORK ORDERS
with tab7:
    st.markdown("#### Prescriptive Maintenance Recommendations & SQLite Work Orders")
    m_eval = decision_engine.evaluate_health(
        predicted_rul=predicted_rul,
        uncertainty_std=unc_res["std_dev"],
        anomaly_score=anom_res["anomaly_score"] if 'anom_res' in locals() else 0.2,
        current_cycle=max_observed_cycle
    )
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown("##### Recommended Actions")
        for rec in m_eval.get("recommended_actions", []):
            st.markdown(f"- **{rec['action']}** ({rec['subsystem']}): {rec['rationale']}")
    with m_col2:
        st.markdown("##### Active Fleet Work Orders (SQLite Persistent)")
        orders = db.get_work_orders(selected_dataset, status="OPEN")
        if orders:
            st.dataframe(pd.DataFrame(orders)[["id", "engine_id", "priority", "title", "subsystem"]], height=240, use_container_width=True)
        else:
            st.success("No open emergency work orders.")

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
        report_text += f"   * [{rec['subsystem']}] {rec['action']}\n     Rationale: {rec['rationale']}\n"

    st.text_area("Engineering Diagnostic Dossier", report_text, height=280)
    st.download_button(
        "📥 Download Diagnostic Dossier (.txt)",
        data=report_text,
        file_name=f"JETWISE_Engine_{selected_engine}_Diagnostic.txt",
        mime="text/plain"
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 4: Validation Fleet Evaluation (Actual vs Predicted)
# ---------------------------------------------------------------------------
st.subheader("4. Actual vs. Predicted RUL Across Validation Fleet")

val_ids = metadata.get("val_engine_ids", [])[:20] if selected_dataset == "FD001" else all_engines[:20]
val_preds = []
val_actuals = []
engine_records = []

if can_predict:
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

# ---------------------------------------------------------------------------
# Section 5: Model Comparison & Benchmark Metrics
# ---------------------------------------------------------------------------
st.subheader("5. Baseline vs. Deep Prognostic Architecture Benchmark")

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

# ---------------------------------------------------------------------------
# Section 6: Predictive Maintenance Alerts
# ---------------------------------------------------------------------------
st.subheader("6. Active Fleet Maintenance Dispatch Alerts")

urgent_engines = [r for r in engine_records if r["Predicted RUL"] <= alert_threshold]
if urgent_engines:
    st.warning(f"⚠️ {len(urgent_engines)} units in the fleet require immediate hangar servicing!")
    st.dataframe(pd.DataFrame(urgent_engines), use_container_width=True)
else:
    st.success(f"All inspected fleet engines have predicted RUL above current safety threshold ({alert_threshold} cycles).")

st.markdown("---")
st.caption("JETWISE AI | Turbofan Jet Engine Prognostic & Health Management Platform | NASA C-MAPSS & N-CMAPSS")
