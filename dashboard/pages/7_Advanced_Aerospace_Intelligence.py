"""THRUSTLAB — Advanced Aerospace Intelligence Centre (Phase 11 Multipage Dashboard)."""

import streamlit as st
import pandas as pd
import numpy as np

from src.dataset_registry import DatasetRegistry
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

st.set_page_config(page_title="THRUSTLAB - Advanced Aerospace Intelligence", page_icon="🚀", layout="wide")

# Theme styling
st.markdown("""
<style>
    .stApp { background-color: #0B0F14; color: #E2E8F0; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Advanced Aerospace Intelligence Centre")

# Sidebar shared controls
st.sidebar.title("✈️ Intelligence Navigation")
available_datasets = DatasetRegistry.get_available_datasets()
selected_dataset = st.sidebar.selectbox("Active Dataset", available_datasets, index=0)
adapter = DatasetRegistry.get_adapter(selected_dataset)
all_engines = adapter.get_engine_ids(kind="train")
selected_engine = st.sidebar.selectbox("Select Engine Fleet ID", all_engines, index=0)

# Load engine telemetry
engine_trajectory = adapter.get_engine_trajectory(selected_engine, kind="train")
if engine_trajectory.empty:
    raw_df = adapter.load_train()
    id_col = "engine_id" if "engine_id" in raw_df.columns else "unit_number"
    engine_trajectory = raw_df[raw_df[id_col] == selected_engine].sort_values("cycle")
current_cycle = int(engine_trajectory["cycle"].max())
latest_row = engine_trajectory.iloc[-1]
latest_sensors = {c: float(latest_row[c]) for c in engine_trajectory.columns if c.startswith("sensor_")}
latest_feat_vec = np.array([latest_sensors.get(f"sensor_{i}", 0.0) for i in range(1, 22)])

# Module tabs
tab_names = [
    "🌐 Intelligence Overview",
    "🔍 Sensor FDIR (11.1)",
    "⚙️ Physics Intelligence (11.2)",
    "🛡️ Model Trust & OOD (11.3)",
    "🎯 Uncertainty & Calibration (11.4)",
    "🛠️ Maintenance Optimizer (11.5)",
    "👥 Fleet Digital Twin (11.6)",
    "📈 Model Health (11.7)",
    "✍️ Engineering Review (11.8)"
]

t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs(tab_names)

with t1:
    render_intelligence_overview(selected_dataset, selected_engine, current_cycle)
with t2:
    render_sensor_fdir(selected_dataset, selected_engine, current_cycle, engine_trajectory)
with t3:
    render_physics_intelligence(selected_dataset, selected_engine, current_cycle, latest_sensors)
with t4:
    render_model_trust(selected_dataset, selected_engine, current_cycle, latest_feat_vec)
with t5:
    render_prediction_uncertainty(predicted_rul=24.5, unc_std=4.8)
with t6:
    render_maintenance_optimization(selected_dataset)
with t7:
    render_fleet_simulation(selected_dataset)
with t8:
    render_model_health()
with t9:
    render_engineering_review(selected_dataset)
