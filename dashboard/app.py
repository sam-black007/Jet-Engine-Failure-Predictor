"""Streamlit Industrial Dashboard for Jet Engine Failure Predictor (NASA C-MAPSS).

Interactive monitoring interface for turbofan engine degradation forecasting,
Remaining Useful Life (RUL) estimation, and predictive maintenance scheduling.
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


# ---------------------------------------------------------------------------
# Page Configuration & Industrial Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Turbofan AI - Jet Engine Health Monitor",
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
    return load_dataset(subset, kind="train")


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
        pred = model.predict(sequence_norm[np.newaxis, ...])
        return float(pred[0])
    else:
        # 2D flattened: (1, 510)
        flat = sequence_norm.reshape(1, -1)
        return float(model.predict(flat)[0])


# ---------------------------------------------------------------------------
# Main App Flow
# ---------------------------------------------------------------------------
metadata = load_metadata_cached()
raw_df = load_raw_dataset_cached(metadata.get("dataset_subset", "FD001"))
metrics_data = load_performance_metrics_cached()
models = load_models()

sensor_cols = metadata["selected_sensors"]
sensor_means = np.array([metadata["sensor_means"][s] for s in sensor_cols], dtype=np.float32)
sensor_stds = np.array([metadata["sensor_stds"][s] for s in sensor_cols], dtype=np.float32)
window_len = metadata["window_length"]
rul_clip = metadata["rul_clip"]

# Sidebar Navigation & Selection
st.sidebar.title("✈️ Industrial Telemetry Control")
st.sidebar.markdown("**NASA C-MAPSS FD001 Fleet Telemetry**")

all_engines = sorted(metadata["train_engine_ids"] + metadata["val_engine_ids"])
selected_engine = st.sidebar.selectbox("Select Engine Fleet ID", all_engines, index=0)

model_names = list(models.keys())
if not model_names:
    st.sidebar.error("No trained models found in models/ directory!")
    st.stop()

model_display_names = {
    "linear_regression": "Linear Regression (Baseline)",
    "random_forest": "Random Forest Regressor",
    "gradient_boosting": "Gradient Boosting Regressor",
    "lstm": "Deep LSTM Neural Network (Recommended)"
}

selected_model_key = st.sidebar.selectbox(
    "Active Predictive Model",
    model_names,
    index=model_names.index("lstm") if "lstm" in model_names else 0,
    format_func=lambda x: model_display_names.get(x, x)
)
selected_model = models[selected_model_key]
is_3d = isinstance(selected_model, torch.nn.Module)

# Maintenance alert threshold
alert_threshold = st.sidebar.slider(
    "Predictive Maintenance Alert Threshold (Cycles)",
    min_value=5, max_value=80, value=30, step=5
)

st.title("Turbofan Jet Engine Degradation & RUL Prediction System")
st.markdown(
    "*Continuous Prognostics and Health Management (PHM) Dashboard powered by Deep Learning.*"
)

# ---------------------------------------------------------------------------
# Section 1: Overall Engine Health Overview
# ---------------------------------------------------------------------------
st.subheader("1. Fleet Health & Operational Overview")
col1, col2, col3, col4 = st.columns(4)

total_engines = len(all_engines)
split_type = "Validation Engine" if selected_engine in metadata["val_engine_ids"] else "Training Engine"
engine_trajectory = raw_df[raw_df["engine_id"] == selected_engine].sort_values("cycle").reset_index(drop=True)
max_observed_cycle = int(engine_trajectory["cycle"].max())

col1.metric("Total Fleet Engines", f"{total_engines} Units")
col2.metric("Dataset Subset", metadata.get("dataset_subset", "FD001"))
col3.metric("Selected Unit Classification", split_type)
col4.metric("Logged Operational Cycles", f"{max_observed_cycle} Cycles")

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 2 & 3: Selected Engine Remaining Useful Life (RUL) Prediction
# ---------------------------------------------------------------------------
st.subheader(f"2. Remaining Useful Life (RUL) Prognosis — Engine #{selected_engine}")

if len(engine_trajectory) < window_len:
    st.error(f"Engine #{selected_engine} has fewer than {window_len} cycles ({len(engine_trajectory)}). Cannot form sequence.")
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
# Section 4 & 5: Sensor Monitoring & Engine Degradation Visualization
# ---------------------------------------------------------------------------
st.subheader("3. Real-Time Telemetry & Degradation Trajectory")

tab1, tab2 = st.tabs(["📊 Sensor Trajectories", "📉 Engine Health Degradation Curve"])

with tab1:
    selected_sensors = st.multiselect(
        "Select Sensor Measurements to Inspect",
        options=sensor_cols,
        default=[sensor_cols[0], sensor_cols[1], sensor_cols[2]] if len(sensor_cols) >= 3 else sensor_cols[:1]
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
    st.caption("Linear run-to-failure degradation trajectory clipped at 125 cycles.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 6: Validation Fleet Evaluation (Actual vs Predicted)
# ---------------------------------------------------------------------------
st.subheader("4. Actual vs. Predicted RUL Across Validation Fleet")

val_ids = metadata["val_engine_ids"]
val_preds = []
val_actuals = []
engine_records = []

for eid in val_ids:
    e_df = raw_df[raw_df["engine_id"] == eid].sort_values("cycle").reset_index(drop=True)
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

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 7 & 8: Model Comparison & Benchmark Metrics
# ---------------------------------------------------------------------------
st.subheader("5. Baseline vs. LSTM Architectural Benchmark")

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
# Section 9: Predictive Maintenance Alerts
# ---------------------------------------------------------------------------
st.subheader("6. Active Fleet Maintenance Dispatch Alerts")

urgent_engines = [r for r in engine_records if r["Predicted RUL"] <= alert_threshold]
if urgent_engines:
    st.warning(f"⚠️ {len(urgent_engines)} units in the validation fleet require immediate hangar servicing!")
    st.dataframe(pd.DataFrame(urgent_engines), use_container_width=True)
else:
    st.success(f"All validation engines have predicted RUL above current safety threshold ({alert_threshold} cycles).")

st.markdown("---")
st.caption("Turbofan Jet Engine Failure Predictor | NASA C-MAPSS Benchmark | Built with PyTorch, Scikit-Learn & Streamlit")
