"""THRUSTLAB — Model Comparison & Benchmark Suite (Streamlit Dashboard).

Interactive visual comparison of all evaluated architectures across:
- Official NASA FD001 100-engine test benchmark
- Raw and Clipped NASA asymmetric scores
- Mean absolute error (MAE) and root mean squared error (RMSE)
- Mean prediction bias and overestimation counts
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

st.set_page_config(
    page_title="THRUSTLAB — Model Comparison",
    page_icon="📊",
    layout="wide"
)

st.title("📊 THRUSTLAB — Architecture Benchmark & Comparison")
st.markdown("Rigorous empirical evaluation on the official **NASA C-MAPSS FD001 test set (100 engines)**.")

# Load Phase 2 comparison CSV if available
comp_csv = PROJECT_ROOT / "results" / "phase2" / "phase2_model_comparison.csv"
if comp_csv.exists():
    df_p2 = pd.read_csv(comp_csv)
else:
    df_p2 = pd.DataFrame()

# Master Benchmark Table (Combining Phase 0, Phase 1, and Phase 2)
benchmark_data = [
    {
        "Phase": "Phase 1",
        "Architecture": "Pinball LSTM (q=0.35)",
        "Raw NASA Score": 362.07,
        "Clipped NASA Score": 277.23,
        "MAE": 12.59,
        "RMSE": 17.06,
        "Mean Bias": -1.80,
        "Overestimates": "42/100",
        "Status": "⭐ CURRENT BEST"
    },
    {
        "Phase": "Phase 2",
        "Architecture": "Time-Series Transformer (q=0.35)",
        "Raw NASA Score": 428.51,
        "Clipped NASA Score": 400.35,
        "MAE": 13.24,
        "RMSE": 17.10,
        "Mean Bias": -6.31,
        "Overestimates": "28/100",
        "Status": "Top Transformer"
    },
    {
        "Phase": "Phase 1",
        "Architecture": "NASA Asymmetric LSTM",
        "Raw NASA Score": 435.34,
        "Clipped NASA Score": 359.25,
        "MAE": 12.38,
        "RMSE": 16.51,
        "Mean Bias": -4.74,
        "Overestimates": "41/100",
        "Status": "Conservative"
    },
    {
        "Phase": "Phase 0",
        "Architecture": "Gradient Boosting Regressor",
        "Raw NASA Score": 385.90,
        "Clipped NASA Score": 352.62,
        "MAE": 11.74,
        "RMSE": 15.44,
        "Mean Bias": -0.85,
        "Overestimates": "46/100",
        "Status": "Best Baseline"
    },
    {
        "Phase": "Phase 0",
        "Architecture": "Deep LSTM (MSE Baseline)",
        "Raw NASA Score": 537.01,
        "Clipped NASA Score": 497.15,
        "MAE": 11.69,
        "RMSE": 16.33,
        "Mean Bias": +0.05,
        "Overestimates": "49/100",
        "Status": "Baseline"
    },
    {
        "Phase": "Phase 2",
        "Architecture": "Attention-LSTM (q=0.35)",
        "Raw NASA Score": 684.48,
        "Clipped NASA Score": 639.79,
        "MAE": 13.86,
        "RMSE": 18.02,
        "Mean Bias": +1.67,
        "Overestimates": "54/100",
        "Status": "Evaluated"
    },
    {
        "Phase": "Phase 0",
        "Architecture": "Linear Regression",
        "Raw NASA Score": 494.68,
        "Clipped NASA Score": 462.10,
        "MAE": 14.30,
        "RMSE": 17.54,
        "Mean Bias": -1.20,
        "Overestimates": "48/100",
        "Status": "Baseline"
    },
    {
        "Phase": "Phase 0",
        "Architecture": "Random Forest",
        "Raw NASA Score": 894.51,
        "Clipped NASA Score": 841.20,
        "MAE": 13.44,
        "RMSE": 17.81,
        "Mean Bias": +2.10,
        "Overestimates": "56/100",
        "Status": "Baseline"
    }
]

df_master = pd.DataFrame(benchmark_data)

st.subheader("Official FD001 Test Set Master Leaderboard")
st.dataframe(df_master, use_container_width=True)

# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
col_v1, col_v2 = st.columns(2)

with col_v1:
    st.subheader("NASA Asymmetric Score Comparison")
    try:
        fig_bar = px.bar(
            df_master.sort_values("Raw NASA Score", ascending=True),
            x="Architecture",
            y="Raw NASA Score",
            color="Phase",
            text="Raw NASA Score",
            title="Official Test Set Raw NASA Score (Lower is Better)"
        )
    except Exception:
        df_sorted = df_master.sort_values("Raw NASA Score", ascending=True)
        fig_bar = go.Figure()
        for phase in df_sorted["Phase"].unique():
            sub = df_sorted[df_sorted["Phase"] == phase]
            fig_bar.add_trace(go.Bar(
                x=sub["Architecture"],
                y=sub["Raw NASA Score"],
                name=phase,
                text=sub["Raw NASA Score"],
                textposition="auto"
            ))
        fig_bar.update_layout(title="Official Test Set Raw NASA Score (Lower is Better)")
    fig_bar.update_layout(xaxis_tickangle=-45, template="plotly_dark")
    st.plotly_chart(fig_bar, use_container_width=True)

with col_v2:
    st.subheader("MAE vs NASA Score Trade-off")
    try:
        fig_scatter = px.scatter(
            df_master,
            x="MAE",
            y="Raw NASA Score",
            color="Phase",
            text="Architecture",
            title="MAE vs NASA Asymmetric Penalty Trade-off",
            size=[25 if "BEST" in str(s) else 15 for s in df_master["Status"]]
        )
    except Exception:
        fig_scatter = go.Figure()
        for phase in df_master["Phase"].unique():
            sub = df_master[df_master["Phase"] == phase]
            fig_scatter.add_trace(go.Scatter(
                x=sub["MAE"],
                y=sub["Raw NASA Score"],
                mode="markers+text",
                name=phase,
                text=sub["Architecture"],
                textposition="top center"
            ))
        fig_scatter.update_layout(title="MAE vs NASA Asymmetric Penalty Trade-off", xaxis_title="MAE", yaxis_title="Raw NASA Score")
    fig_scatter.update_traces(textposition='top center')
    fig_scatter.update_layout(template="plotly_dark")
    st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------------------------------
# Per-Engine Residuals Inspector
# ---------------------------------------------------------------------------
st.subheader("Test Engine Predictions Inspector")
pred_files = {
    "Transformer (Phase 2)": PROJECT_ROOT / "results" / "phase2" / "test_predictions_transformer.csv",
    "Attention-LSTM (Phase 2)": PROJECT_ROOT / "results" / "phase2" / "test_predictions_attention_lstm.csv",
    "Pinball LSTM q=0.35 (Phase 1)": PROJECT_ROOT / "results" / "phase1" / "test_predictions.csv"
}

avail_preds = {k: v for k, v in pred_files.items() if v.exists()}
if avail_preds:
    sel_pred = st.selectbox("Select Model for Trajectory Breakdown", list(avail_preds.keys()))
    df_pred = pd.read_csv(avail_preds[sel_pred])
    st.dataframe(df_pred.head(20), use_container_width=True)

    fig_err = px.histogram(
        df_pred,
        x="error_raw" if "error_raw" in df_pred.columns else df_pred.columns[df_pred.columns.str.contains("error")][0],
        nbins=25,
        title=f"Prediction Error Distribution: {sel_pred}",
        labels={"x": "Prediction Error (Predicted - Actual RUL in Cycles)"}
    )
    fig_err.add_vline(x=0, line_dash="dash", line_color="red")
    fig_err.update_layout(template="plotly_dark")
    st.plotly_chart(fig_err, use_container_width=True)
