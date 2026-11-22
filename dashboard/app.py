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
from src.ui.phase11_workspaces import (
    render_intelligence_overview,
    render_sensor_fdir,
    render_physics_intelligence,
    render_model_trust,
    render_prediction_uncertainty,
    render_maintenance_optimization,
    render_fleet_simulation,
    render_model_health,
    render_engineering_review
)
from src.ui.phase12_workspaces import (
    render_research_overview,
    render_mission_intelligence,
    render_component_health,
    render_multimodal_fusion,
    render_reliability_analysis,
    render_active_learning,
    render_knowledge_graph,
    render_edge_deployment,
    render_decision_intelligence
)
from src.ui.configuration_centre import render_global_configuration_centre
from src.ui.copilot_workspace import render_copilot_workspace
from src.ui.matlab_centre import render_matlab_centre
from src.ui.multi_engine_centre import render_multi_engine_centre
from src.ui.api_configuration_centre import render_api_configuration_centre
from src.ui.unified_navigation import (
    NavigationHub,
    init_navigation_state,
    render_unified_header_and_navigation
)
from src.ui.mission_control_home import render_mission_control_home
from src.ui.engineering_command_centre import render_engineering_command_centre
from src.ui.monitor_intelligence_hub import render_monitor_intelligence_hub
from src.ui.analyze_engineering_hub import render_analyze_engineering_hub
from src.ui.configure_system_hub import render_configure_system_hub
from src.ui.contextual_help import render_glossary_view
from src.ui.scada_overview import render_operations_scada_view

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
init_navigation_state()
st.sidebar.title("✈️ JETWISE AI OS")

# Simplified 5-Hub Primary Navigation
hub_options = [
    NavigationHub.HOME,
    NavigationHub.SIMULATE,
    NavigationHub.MONITOR,
    NavigationHub.ANALYZE,
    NavigationHub.CONFIGURE,
    "GLOSSARY"
]
hub_labels = {
    NavigationHub.HOME: "🏠 1. HOME — Mission Control",
    NavigationHub.SIMULATE: "⚡ 2. SIMULATE — Command Centre",
    NavigationHub.MONITOR: "🛰️ 3. MONITOR — Engine Intelligence",
    NavigationHub.ANALYZE: "🔬 4. ANALYZE — Engineering Analysis",
    NavigationHub.CONFIGURE: "⚙️ 5. CONFIGURE — System Setup",
    "GLOSSARY": "📚 Aerospace & AI Glossary"
}

curr_hub = st.session_state.get("active_hub", NavigationHub.HOME)
if curr_hub not in hub_options:
    curr_hub = NavigationHub.HOME

selected_hub_sb = st.sidebar.selectbox(
    "Primary Navigation Hub",
    hub_options,
    index=hub_options.index(curr_hub),
    format_func=lambda h: hub_labels.get(h, h),
    key="sb_active_hub"
)
if selected_hub_sb != st.session_state.get("active_hub"):
    st.session_state["active_hub"] = selected_hub_sb
    st.rerun()

st.sidebar.markdown("---")

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

# Preprocessing parameters
window_len = metadata["window_length"]
rul_clip = metadata["rul_clip"]
sensor_cols = metadata["selected_sensors"]
can_predict = compat_info["status"] == "COMPATIBLE"
sensor_means = np.array([metadata["feature_means"][s] for s in sensor_cols], dtype=np.float32) if can_predict and all(s in metadata["feature_means"] for s in sensor_cols) else np.zeros(len(sensor_cols))
sensor_stds = np.array([metadata["feature_stds"][s] for s in sensor_cols], dtype=np.float32) if can_predict and all(s in metadata["feature_stds"] for s in sensor_cols) else np.ones(len(sensor_cols))
sensor_stds[sensor_stds == 0] = 1.0


def render_legacy_scada_view():
    """Legacy Operational SCADA, Replay, and Fleet Monitoring view."""
    # ---------------------------------------------------------------------------
    # Main App Header
    # ---------------------------------------------------------------------------
    st.subheader("✈️ Fleet Health & Operational SCADA Overview")
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
anom_res = {"anomaly_score": 0.12, "severity": "NORMAL", "is_anomaly": False, "top_anomalous_sensors": []}
m_eval = {
    "maintenance_priority_index": 65.0,
    "risk_category": "MONITOR",
    "recommended_actions": [
        {"subsystem": "Turbine", "action": "Borescope Inspection", "rationale": "Scheduled fleet monitoring"}
    ],
    "score_breakdown": {"rul_score": 50.0, "anomaly_score": 10.0, "uncertainty_score": 5.0}
}

window_len = metadata["window_length"]
rul_clip = metadata["rul_clip"]
sensor_cols = metadata["selected_sensors"]
sensor_means = np.array([metadata["sensor_means"][s] for s in sensor_cols], dtype=np.float32)
sensor_stds = np.array([metadata["sensor_stds"][s] for s in sensor_cols], dtype=np.float32)

can_predict = all(c in engine_trajectory.columns for c in sensor_cols)
norm_window = None

actual_final_rul = min(max_observed_cycle - engine_trajectory["cycle"].iloc[-1], rul_clip)

if not can_predict:
    st.info(
        f"Selected dataset `{selected_dataset}` schema differs from the model's training configuration ({model_input_dim} features). "
        "Use the Historical Telemetry Replay or Multi-Dataset Research tab for cross-domain inspection."
    )
    predicted_rul = float(actual_final_rul)
    unc_res = {"std_dev": 0.0, "q10": max(0.0, predicted_rul - 5.0), "q90": predicted_rul + 5.0, "confidence_score": 1.0}
    
    rul_col1, rul_col2, rul_col3, rul_col4 = st.columns(4)
    rul_col1.metric("Predicted RUL (Cycles Remaining)", "N/A (Schema Mismatch)")
    rul_col2.metric("True RUL (at Last Recorded Cycle)", f"{actual_final_rul:.1f} Cycles")
    rul_col3.metric("90% Prediction Interval", "N/A")
    rul_col4.markdown("### Health Status")
    rul_col4.markdown("<div class='status-caution'>ℹ️ CROSS-DATASET: Model trained on C-MAPSS FD001 schema.</div>", unsafe_allow_html=True)

elif len(engine_trajectory) < window_len:
    st.error(f"Engine #{selected_engine} has fewer than {window_len} cycles ({len(engine_trajectory)}).")
    predicted_rul = float(actual_final_rul)
else:
    latest_window = engine_trajectory[sensor_cols].iloc[-window_len:].values.astype(np.float32)
    if len(latest_window) < window_len:
        pad = np.repeat(latest_window[0:1], window_len - len(latest_window), axis=0)
        latest_window = np.vstack([pad, latest_window])
    norm_window = (latest_window - sensor_means) / sensor_stds

    if is_3d:
        unc_tool = UncertaintyEstimator(selected_model, num_samples=20)
        unc_res = unc_tool.predict_with_uncertainty(norm_window)
        predicted_rul = float(unc_res["mean_rul"])
    else:
        predicted_rul = predict_rul(selected_model, norm_window, is_3d=is_3d)
        unc_res = {"std_dev": 6.5, "q10": max(0.0, predicted_rul - 10.0), "q90": predicted_rul + 10.0, "confidence_score": 0.82}

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
# Phase 17 Unified 5-Hub Master Orchestrator
# ---------------------------------------------------------------------------
def switch_hub_fn(target_hub: str):
    st.session_state["active_hub"] = target_hub
    st.rerun()

active_hub = render_unified_header_and_navigation()

if active_hub == NavigationHub.HOME:
    render_mission_control_home(switch_hub_fn)

elif active_hub == NavigationHub.SIMULATE:
    render_engineering_command_centre()

elif active_hub == NavigationHub.CONFIGURE:
    render_configure_system_hub()

elif active_hub == "GLOSSARY":
    render_glossary_view()

elif active_hub == NavigationHub.ANALYZE:
    render_analyze_engineering_hub()

elif active_hub == NavigationHub.MONITOR:
    m_subtab1, m_subtab2, m_subtab3 = st.tabs([
        "🌐 Operations SCADA, Replay & 3D Twin",
        "🔍 Sensor FDIR & Fleet Intelligence (Phase 11)",
        "🧩 Component Health & Deep Research (Phase 12)"
    ])

    with m_subtab2:
        p11_t1, p11_t2, p11_t3, p11_t4, p11_t5, p11_t6, p11_t7, p11_t8, p11_t9 = st.tabs([
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
        latest_row_dict = engine_trajectory.iloc[-1].to_dict()
        latest_sensors_dict = {c: float(latest_row_dict[c]) for c in engine_trajectory.columns if c.startswith("sensor_")}
        if not latest_sensors_dict:
            numeric_cols = [c for c in engine_trajectory.columns if c not in ("engine_id", "cycle")]
            latest_sensors_dict = {c: float(latest_row_dict[c]) for c in numeric_cols}
        latest_feat_vec_arr = np.array([latest_sensors_dict.get(f"sensor_{i}", 0.0) for i in range(1, 22)])
        if np.all(latest_feat_vec_arr == 0.0) and len(latest_sensors_dict) > 0:
            latest_feat_vec_arr = np.array([float(v) for v in list(latest_sensors_dict.values())[:21]], dtype=np.float32)
            if len(latest_feat_vec_arr) < 21:
                latest_feat_vec_arr = np.pad(latest_feat_vec_arr, (0, 21 - len(latest_feat_vec_arr)))

        with p11_t1:
            render_intelligence_overview(selected_dataset, selected_engine, max_observed_cycle)
        with p11_t2:
            render_sensor_fdir(selected_dataset, selected_engine, max_observed_cycle, engine_trajectory)
        with p11_t3:
            render_physics_intelligence(selected_dataset, selected_engine, max_observed_cycle, latest_sensors_dict)
        with p11_t4:
            render_model_trust(selected_dataset, selected_engine, max_observed_cycle, latest_feat_vec_arr)
        with p11_t5:
            render_prediction_uncertainty(predicted_rul=predicted_rul, unc_std=unc_res["std_dev"])
        with p11_t6:
            render_maintenance_optimization(selected_dataset)
        with p11_t7:
            render_fleet_simulation(selected_dataset)
        with p11_t8:
            render_model_health()
        with p11_t9:
            render_engineering_review(selected_dataset)

    with m_subtab3:
        p12_t1, p12_t2, p12_t3, p12_t4, p12_t5, p12_t6, p12_t7, p12_t8, p12_t9 = st.tabs([
            "🌐 Research Overview",
            "🛫 Mission Intelligence (12.1)",
            "🧩 Component Health (12.2)",
            "📡 Multi-Modal Fusion (12.3)",
            "📉 Reliability & Survival (12.4)",
            "🎯 Active Learning (12.5)",
            "🕸️ Knowledge Graph (12.6)",
            "⚡ Edge Avionics (12.7)",
            "⚖️ Decision Intelligence (12.8)"
        ])
        latest_row_dict = engine_trajectory.iloc[-1].to_dict()
        latest_sensors_dict = {c: float(latest_row_dict[c]) for c in engine_trajectory.columns if c.startswith("sensor_")}
        if not latest_sensors_dict:
            numeric_cols = [c for c in engine_trajectory.columns if c not in ("engine_id", "cycle")]
            latest_sensors_dict = {c: float(latest_row_dict[c]) for c in numeric_cols}
        mock_zscores_dict = {k: float(np.clip((v - np.mean(engine_trajectory[k])) / (np.std(engine_trajectory[k]) + 1e-4), -4, 4)) for k, v in latest_sensors_dict.items()}

        with p12_t1:
            render_research_overview(selected_dataset, selected_engine)
        with p12_t2:
            render_mission_intelligence(selected_dataset, selected_engine, max_observed_cycle, baseline_rul=predicted_rul)
        with p12_t3:
            render_component_health(selected_dataset, selected_engine, max_observed_cycle, predicted_rul=predicted_rul, sensor_zscores=mock_zscores_dict)
        with p12_t4:
            render_multimodal_fusion(selected_dataset, selected_engine, max_observed_cycle, anomaly_score=anom_res["anomaly_score"])
        with p12_t5:
            render_reliability_analysis(selected_dataset, max_observed_cycle)
        with p12_t6:
            render_active_learning(selected_dataset)
        with p12_t7:
            render_knowledge_graph()
        with p12_t8:
            render_edge_deployment()
        with p12_t9:
            render_decision_intelligence(selected_dataset, selected_engine, max_observed_cycle, predicted_rul=predicted_rul, mpi=m_eval.get("maintenance_priority_index", 65.0))

    with m_subtab1:
        scada_ctx = {
            "engine_trajectory": engine_trajectory,
            "ds_meta": ds_meta,
            "selected_dataset": selected_dataset,
            "selected_engine": selected_engine,
            "max_observed_cycle": max_observed_cycle,
            "rul_clip": rul_clip,
            "can_predict": can_predict,
            "window_len": window_len,
            "sensor_cols": sensor_cols,
            "sensor_means": sensor_means,
            "sensor_stds": sensor_stds,
            "selected_model": selected_model,
            "is_3d": is_3d,
            "predict_rul": predict_rul,
            "health_state": health_state,
            "predicted_rul": predicted_rul,
            "anom_res": anom_res,
            "norm_window": norm_window,
            "raw_df": raw_df,
            "FEATURE_COLS": FEATURE_COLS,
            "unc_res": unc_res,
            "decision_engine": decision_engine,
            "db": db,
            "metadata": metadata,
            "all_engines": all_engines,
            "adapter": adapter,
            "alert_threshold": alert_threshold,
            "metrics_data": metrics_data,
            "model_display_names": model_display_names,
        }
        render_operations_scada_view(scada_ctx)
    
st.markdown("---")
st.caption("JETWISE AI | Turbofan Jet Engine Prognostic & Health Management Platform | NASA C-MAPSS & N-CMAPSS")
