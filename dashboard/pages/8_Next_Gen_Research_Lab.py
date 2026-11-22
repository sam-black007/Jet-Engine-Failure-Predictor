"""THRUSTLAB — Next-Generation Research Laboratory (Phase 12 Multipage Dashboard)."""

import streamlit as st
import pandas as pd
import numpy as np

from src.dataset_registry import DatasetRegistry
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

st.set_page_config(page_title="THRUSTLAB - Next-Gen Research Lab", page_icon="🔬", layout="wide")

# Theme styling
st.markdown("""
<style>
    .stApp { background-color: #0B0F14; color: #E2E8F0; }
</style>
""", unsafe_allow_html=True)

st.title("🔬 Next-Generation Aerospace Research Laboratory")

# Sidebar shared controls
st.sidebar.title("🔬 Research Controls")
available_datasets = DatasetRegistry.get_available_datasets()
selected_dataset = st.sidebar.selectbox("Active Dataset", available_datasets, index=0)
adapter = DatasetRegistry.get_adapter(selected_dataset)
all_engines = adapter.get_engine_ids(kind="train")
selected_engine = st.sidebar.selectbox("Select Engine Unit", all_engines, index=0)

# Load engine telemetry
engine_trajectory = adapter.get_engine_trajectory(selected_engine, kind="train")
if engine_trajectory.empty:
    raw_df = adapter.load_train()
    id_col = "engine_id" if "engine_id" in raw_df.columns else "unit_number"
    engine_trajectory = raw_df[raw_df[id_col] == selected_engine].sort_values("cycle")
current_cycle = int(engine_trajectory["cycle"].max())
latest_row = engine_trajectory.iloc[-1]
latest_sensors = {c: float(latest_row[c]) for c in engine_trajectory.columns if c.startswith("sensor_")}
mock_zscores = {k: float(np.clip((v - np.mean(engine_trajectory[k])) / (np.std(engine_trajectory[k]) + 1e-4), -4, 4)) for k, v in latest_sensors.items()}

# Module tabs
tab_names = [
    "🌐 Research Overview",
    "🛫 Mission Intelligence (12.1)",
    "🧩 Component Health (12.2)",
    "📡 Multi-Modal Fusion (12.3)",
    "📉 Reliability & Survival (12.4)",
    "🎯 Active Learning (12.5)",
    "🕸️ Knowledge Graph (12.6)",
    "⚡ Edge Avionics (12.7)",
    "⚖️ Decision Intelligence (12.8)"
]

t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs(tab_names)

with t1:
    render_research_overview(selected_dataset, selected_engine)
with t2:
    render_mission_intelligence(selected_dataset, selected_engine, current_cycle, baseline_rul=35.0)
with t3:
    render_component_health(selected_dataset, selected_engine, current_cycle, predicted_rul=35.0, sensor_zscores=mock_zscores)
with t4:
    render_multimodal_fusion(selected_dataset, selected_engine, current_cycle, anomaly_score=0.45)
with t5:
    render_reliability_analysis(selected_dataset, current_cycle)
with t6:
    render_active_learning(selected_dataset)
with t7:
    render_knowledge_graph()
with t8:
    render_edge_deployment()
with t9:
    render_decision_intelligence(selected_dataset, selected_engine, current_cycle, predicted_rul=35.0, mpi=68.5)
