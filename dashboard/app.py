"""Streamlit Industrial Dashboard for Jet Engine Failure Predictor (JETWISE AI).

Interactive monitoring interface for turbofan engine degradation forecasting,
Remaining Useful Life (RUL) estimation, predictive maintenance scheduling,
multi-dataset operational switching, and historical telemetry replay simulation.
"""

import os
import sys
import json
import pathlib
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

# Ensure project root is in python path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_dataset
from src.train_lstm import LSTMPredictor
from src.dataset_registry import DatasetRegistry
from src.dataset_metadata import CMAPSS_SENSOR_METADATA, CMAPSS_OP_SETTINGS_METADATA
from src.data_replay import EngineReplaySimulator


# ---------------------------------------------------------------------------
# Page Configuration & Industrial Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="JETWISE AI - Jet Engine Health Monitor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-left: 5px solid #0d6efd;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 10px;
    }
    .status-normal { color: #198754; font-weight: bold; }
    .status-caution { color: #fd7e14; font-weight: bold; }
    .status-critical { color: #dc3545; font-weight: bold; }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-healthy { background-color: #d1fae5; color: #065f46; }
    .badge-monitor { background-color: #fef3c7; color: #92400e; }
    .badge-warning { background-color: #ffedd5; color: #9a3412; }
    .badge-critical { background-color: #fee2e2; color: #991b1b; }
    .simulation-banner {
        background-color: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e40af;
        padding: 10px 16px;
        border-radius: 6px;
        margin-bottom: 15px;
        font-size: 0.9rem;
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
def load_raw_dataset_cached(subset: str = "FD001"):
    adapter = DatasetRegistry.get_adapter(subset)
    return adapter.load_train()


@st.cache_data
def load_performance_metrics_cached():
    metrics_path = PROJECT_ROOT / "models" / "performance_metrics.json"
    if metrics_path.is_file():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_resource
def load_models():
    """Load baseline sklearn models and PyTorch deep models (LSTM, Pinball, Transformer)."""
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


# ---------------------------------------------------------------------------
# Data Preprocessing for Inference
# ---------------------------------------------------------------------------
def predict_rul(model, sequence_norm: np.ndarray, is_3d: bool = False) -> float:
    """Predict RUL given normalized sequence of shape (30, 17)."""
    if is_3d or isinstance(model, torch.nn.Module):
        # 3D: (1, 30, 17)
        if hasattr(model, "predict"):
            pred = model.predict(sequence_norm[np.newaxis, ...])
            return float(pred[0])
        else:
            model.eval()
            with torch.no_grad():
                t_in = torch.tensor(sequence_norm[np.newaxis, ...], dtype=torch.float32)
                return float(model(t_in).item())
    else:
        # 2D flattened: (1, 510)
        flat = sequence_norm.reshape(1, -1)
        return float(model.predict(flat)[0])


# ---------------------------------------------------------------------------
# Main App Flow
# ---------------------------------------------------------------------------
metadata = load_metadata_cached()
metrics_data = load_performance_metrics_cached()
models = load_models()

# Sidebar Navigation & Selection
st.sidebar.title("✈️ Industrial Telemetry Control")

# Multi-Dataset Dynamic Selector
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
selected_engine = st.sidebar.selectbox("Select Engine Fleet ID", all_engines, index=0)

model_names = list(models.keys())
if not model_names:
    st.sidebar.error("No trained models found in models/ directory!")
    st.stop()

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

st.title("Turbofan Jet Engine Degradation & RUL Prediction System")
st.markdown(
    "*Continuous Prognostics and Health Management (PHM) Platform powered by Deep Learning.*"
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

window_len = metadata["window_length"]
rul_clip = metadata["rul_clip"]
sensor_cols = metadata["selected_sensors"]
sensor_means = np.array([metadata["sensor_means"][s] for s in sensor_cols], dtype=np.float32)
sensor_stds = np.array([metadata["sensor_stds"][s] for s in sensor_cols], dtype=np.float32)

can_predict = compat_info["is_compatible"] and all(c in engine_trajectory.columns for c in sensor_cols)

if not can_predict:
    st.info(
        f"Selected dataset `{selected_dataset}` schema differs from the model's training configuration ({model_input_dim} features). "
        "Use the Historical Telemetry Replay or Multi-Dataset Research tab for cross-domain inspection."
    )
    predicted_rul = None
elif len(engine_trajectory) < window_len:
    st.error(f"Engine #{selected_engine} has fewer than {window_len} cycles ({len(engine_trajectory)}).")
    predicted_rul = None
else:
    # Prepare latest sequence window
    latest_window = engine_trajectory[sensor_cols].iloc[-window_len:].values.astype(np.float32)
    norm_window = (latest_window - sensor_means) / sensor_stds
    predicted_rul = predict_rul(selected_model, norm_window, is_3d=is_3d)
    
    # Ground truth RUL at the end of observed run-to-failure trajectory is 0
    actual_final_rul = min(max_observed_cycle - engine_trajectory["cycle"].iloc[-1], rul_clip)
    
    rul_col1, rul_col2, rul_col3 = st.columns(3)
    rul_col1.metric("Predicted RUL (Cycles Remaining)", f"{predicted_rul:.1f} Cycles")
    rul_col2.metric("True RUL (at Last Recorded Cycle)", f"{actual_final_rul:.1f} Cycles")
    
    # Status Alert
    if predicted_rul <= alert_threshold:
        status_html = f"<div class='status-critical'>🚨 CRITICAL ALERT: RUL below {alert_threshold} cycles. Immediate inspection required!</div>"
    elif predicted_rul <= alert_threshold + 20:
        status_html = f"<div class='status-caution'>⚠️ CAUTION: Engine approaching scheduled maintenance window.</div>"
    else:
        status_html = "<div class='status-normal'>✅ NORMAL: Operating within safe flight parameters.</div>"
    rul_col3.markdown("### Health Status")
    rul_col3.markdown(status_html, unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 3: Telemetry, Degradation & Historical Replay Simulation
# ---------------------------------------------------------------------------
st.subheader("3. Real-Time Telemetry & Degradation Trajectory")

tab1, tab2, tab3 = st.tabs([
    "📊 Sensor Trajectories",
    "📉 Engine Health Degradation Curve",
    "⏱️ Historical Telemetry Replay Simulation"
])

with tab1:
    available_sensors = [c for c in engine_trajectory.columns if c.startswith("sensor_") or c in ds_meta.get("active_sensors", [])]
    selected_sensors = st.multiselect(
        "Select Sensor Measurements to Inspect",
        options=available_sensors,
        default=available_sensors[:3] if len(available_sensors) >= 3 else available_sensors
    )
    if selected_sensors:
        plot_df = engine_trajectory[["cycle"] + selected_sensors].set_index("cycle")
        st.line_chart(plot_df, height=350)
    else:
        st.info("Select one or more sensors from the multiselect box above to view telemetry.")

with tab2:
    degradation_df = engine_trajectory[["cycle"]].copy()
    degradation_df["True_RUL"] = np.clip(max_observed_cycle - degradation_df["cycle"], 0, rul_clip)
    st.line_chart(degradation_df.set_index("cycle"), height=350)
    st.caption(f"Run-to-failure degradation trajectory for Engine #{selected_engine} ({selected_dataset}).")

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

        # Compute real-time model prediction at this playback cycle
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
        # Multi-cycle degradation trajectory up to current playback cycle
        sim_history_cycles = list(range(1, playback_cycle + 1))
        sim_history_gt = [max_observed_cycle - c for c in sim_history_cycles]

        chart_history = pd.DataFrame({
            "Cycle": sim_history_cycles,
            "Ground Truth RUL": sim_history_gt,
        }).set_index("Cycle")

        st.markdown(f"##### Telemetry & RUL Degradation at Cycle {playback_cycle}")
        st.line_chart(chart_history, height=280)

        # Show key physical sensor readings at current cycle
        tel = cycle_info["telemetry"]
        tel_cols = [k for k in tel.keys() if k.startswith("sensor_") or k in ds_meta.get("active_sensors", [])][:6]
        if tel_cols:
            kpi_cols = st.columns(len(tel_cols))
            for i, col_name in enumerate(tel_cols):
                s_meta = CMAPSS_SENSOR_METADATA.get(col_name, {})
                label = f"{s_meta.get('symbol', col_name)} ({s_meta.get('units', '')})"
                val = tel[col_name]
                kpi_cols[i].metric(label, f"{val:.2f}" if isinstance(val, (int, float)) else str(val))

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
        st.dataframe(val_table_df[["Engine ID", "Predicted RUL", "Maintenance Alert"]], height=320, width="stretch")
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
    st.info("Performance metrics file not found. Run train_baselines.py and train_lstm.py first.")

# ---------------------------------------------------------------------------
# Section 6: Predictive Maintenance Alerts
# ---------------------------------------------------------------------------
st.subheader("6. Active Fleet Maintenance Dispatch Alerts")

urgent_engines = [r for r in engine_records if r["Predicted RUL"] <= alert_threshold]
if urgent_engines:
    st.warning(f"⚠️ {len(urgent_engines)} units in the fleet require immediate hangar servicing!")
    st.dataframe(pd.DataFrame(urgent_engines), width="stretch")
else:
    st.success(f"All inspected fleet engines have predicted RUL above current safety threshold ({alert_threshold} cycles).")

st.markdown("---")
st.caption("JETWISE AI | Turbofan Jet Engine Prognostic & Health Management Platform | NASA C-MAPSS & N-CMAPSS")
