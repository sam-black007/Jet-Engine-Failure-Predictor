"""Phase 1 Interactive Results Viewer: Asymmetric Loss Formulations.

Displays:
- Baseline MSE LSTM vs. NASA Asymmetric LSTM vs. Pinball LSTM
- Official NASA C-MAPSS FD001 Test Set Metrics (100 engines)
- Interactive Vega-Lite / Altair and Plotly visualizations
- Per-engine error residuals and NASA penalty breakdown
"""

import json
import pathlib
import pandas as pd
import numpy as np
import streamlit as st

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "phase1"

st.set_page_config(
    page_title="Phase 1: Asymmetric Loss Prognostics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("✈️ Phase 1: Asymmetric Loss RUL Prognostics Explorer")
st.markdown("**NASA C-MAPSS FD001 Benchmark | Mitigating Critical Overestimation with Asymmetric Loss Functions**")

# Load metrics
metrics_path = RESULTS_DIR / "phase1_metrics.json"
pred_csv_path = RESULTS_DIR / "test_predictions.csv"

if not metrics_path.is_file() or not pred_csv_path.is_file():
    st.error("Phase 1 artifacts not found. Please run src/evaluate_phase1.py first.")
    st.stop()

with open(metrics_path, "r", encoding="utf-8") as f:
    metrics_data = json.load(f)

pred_df = pd.read_csv(pred_csv_path)

# Top metric summary cards
st.subheader("1. Official FD001 Test Set Benchmark Summary (100 Engines)")

c1, c2, c3, c4, c5 = st.columns(5)
base_m = metrics_data["models"]["baseline_lstm"]
asymm_m = metrics_data["models"]["nasa_asymmetric_lstm"]
impr = metrics_data["improvements_vs_baseline_lstm"]["nasa_asymmetric_lstm"]

c1.metric(
    "Baseline NASA Score",
    f"{base_m['raw_target']['NASA_score']:.1f}",
    f"{base_m['clipped_target']['NASA_score']:.1f} (Clipped)",
    delta_color="off"
)
c2.metric(
    "Asymmetric NASA Score",
    f"{asymm_m['raw_target']['NASA_score']:.1f}",
    f"-{impr['raw_nasa_reduction_pct']}% penalty",
    delta_color="inverse"
)
c3.metric(
    "Clipped Target NASA",
    f"{asymm_m['clipped_target']['NASA_score']:.1f}",
    f"-{impr['clip_nasa_reduction_pct']}% penalty",
    delta_color="inverse"
)
c4.metric(
    "Prediction Bias",
    f"{asymm_m['mean_bias']:+.2f} cyc",
    f"{impr['bias_shift_cycles']:+.1f} cyc shift",
    delta_color="inverse"
)
c5.metric(
    "Overestimated Units",
    f"{asymm_m['overestimation_count']}/100",
    f"-{base_m['overestimation_count'] - asymm_m['overestimation_count']} units",
    delta_color="inverse"
)

st.markdown("---")

# Section 2: Model Comparison Table
st.subheader("2. Comprehensive Architecture Benchmark (Official Test Fleet)")
table_rows = []
for k, v in metrics_data["models"].items():
    table_rows.append({
        "Architecture": v["name"],
        "Test MAE (Raw)": f"{v['raw_target']['MAE']:.2f}",
        "Test MAE (Clip125)": f"{v['clipped_target']['MAE']:.2f}",
        "Test RMSE": f"{v['raw_target']['RMSE']:.2f}",
        "Official NASA Score (Raw)": f"{v['raw_target']['NASA_score']:.1f}",
        "Official NASA Score (Clip)": f"{v['clipped_target']['NASA_score']:.1f}",
        "Mean Bias (Cycles)": f"{v['mean_bias']:+.2f}",
        "Overestimated Units": f"{v['overestimation_count']}/100",
        "Latency": f"{v['latency_ms']:.2f} ms"
    })
st.table(pd.DataFrame(table_rows))

st.markdown("---")

# Section 3: Graphical Inspection Tabs
st.subheader("3. Technical Visualizations & Residual Diagnostics")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Model Performance Comparison",
    "🎯 Actual vs. Predicted RUL",
    "📈 100-Engine Residual Errors",
    "⚠️ Overestimation Mitigation",
    "📉 Training Dynamics & Loss"
])

with tab1:
    st.image(str(RESULTS_DIR / "model_performance_comparison.png"), use_column_width=True)

with tab2:
    st.image(str(RESULTS_DIR / "actual_vs_predicted_scatter.png"), use_column_width=True)

with tab3:
    st.image(str(RESULTS_DIR / "per_engine_comparison.png"), use_column_width=True)

with tab4:
    st.image(str(RESULTS_DIR / "overestimation_underestimation_analysis.png"), use_column_width=True)
    st.image(str(RESULTS_DIR / "nasa_penalty_distribution.png"), use_column_width=True)

with tab5:
    st.image(str(RESULTS_DIR / "training_curves.png"), use_column_width=True)

st.markdown("---")

# Section 4: Granular Engine Table
st.subheader("4. 100-Engine Granular Inspection & Penalty Table")
st.dataframe(pred_df, use_container_width=True, height=350)

st.caption("Jet Engine Failure Predictor — Phase 1 Asymmetric Loss Research | Evaluated on NASA C-MAPSS FD001 Test Set")
