"""JETWISE AI — Experiment Centre (Streamlit Dashboard).

Direct interface to persistent SQLite experiment database:
- Filter, sort, and inspect historical training runs
- Compare hyperparameters against validation and test outcomes
- Export complete run logs to CSV and JSON
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
from src.experiment_tracker import ExperimentTracker

st.set_page_config(
    page_title="JETWISE AI — Experiment Centre",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 JETWISE AI — Experiment Management Centre")
st.markdown("Centralized audit trail of all model training runs, hyperparameters, and benchmarks.")

tracker = ExperimentTracker()
df = tracker.list_experiments()

if df.empty:
    st.info("No experiment records found in database. Train models to populate.")
else:
    # Summary Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Experiments", len(df))
    with c2:
        best_run = tracker.get_best_experiment("test_nasa_raw")
        val = f"{best_run['test_nasa_raw']:.2f}" if best_run else "N/A"
        st.metric("Best Official NASA Score", val)
    with c3:
        n_archs = df["model_architecture"].nunique()
        st.metric("Architectures Tested", n_archs)
    with c4:
        best_mae = tracker.get_best_experiment("test_mae")
        mae_val = f"{best_mae['test_mae']:.2f}" if best_mae else "N/A"
        st.metric("Lowest Test MAE", mae_val)

    st.markdown("---")

    # Filters
    f1, f2 = st.columns(2)
    with f1:
        selected_arch = st.multiselect(
            "Filter by Architecture",
            options=df["model_architecture"].unique().tolist(),
            default=df["model_architecture"].unique().tolist()
        )
    with f2:
        selected_loss = st.multiselect(
            "Filter by Loss",
            options=df["loss_name"].dropna().unique().tolist(),
            default=df["loss_name"].dropna().unique().tolist()
        )

    filtered_df = df[
        df["model_architecture"].isin(selected_arch) &
        df["loss_name"].isin(selected_loss)
    ]

    st.subheader(f"Experiment History ({len(filtered_df)} runs)")
    st.dataframe(
        filtered_df[[
            "experiment_id", "profile_name", "model_architecture",
            "loss_name", "pinball_quantile", "learning_rate",
            "val_nasa", "test_nasa_raw", "test_mae", "training_duration_seconds",
            "git_commit", "created_at"
        ]],
        use_container_width=True
    )

    # Export
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Filtered Experiments (CSV)",
        csv_bytes,
        "experiments_export.csv",
        "text/csv"
    )
