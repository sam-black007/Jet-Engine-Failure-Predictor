"""Streamlit Multi-Dataset Research and Physical Sensor Intelligence (JETWISE AI).

Presents multi-dataset comparative architectures, physical sensor ontologies,
automated data quality validation, and cross-dataset generalization benchmarks.
"""

import sys
import json
import pathlib
import pandas as pd
import numpy as np
import streamlit as st

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.dataset_metadata import (
    CMAPSS_SENSOR_METADATA,
    CMAPSS_OP_SETTINGS_METADATA,
    DATASET_PROFILES
)
from src.dataset_registry import DatasetRegistry
from src.dataset_validator import DatasetValidator

st.set_page_config(
    page_title="Multi-Dataset Research & Domain Transfer - JETWISE AI",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Multi-Dataset Research & Sensor Telemetry Intelligence")
st.markdown(
    "*Cross-dataset evaluation, physical sensor telemetry ontology, automated quality validation, and domain-shift benchmarks.*"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Dataset Comparison Matrix",
    "🩺 Physical Sensor Telemetry Dictionary",
    "🛡️ Automated Data Quality Validator",
    "🌐 Cross-Dataset Generalization Benchmark"
])

# ---------------------------------------------------------------------------
# Tab 1: Dataset Comparison Matrix
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("1. Comprehensive Predictive Maintenance Dataset Catalog")
    st.markdown(
        "JETWISE AI incorporates multi-dataset ingestion supporting single/multi-condition flight profiles, "
        "single/compound fault modes, and next-generation continuous 1 Hz commercial turbofan telemetry."
    )

    rows = []
    for code, prof in DATASET_PROFILES.items():
        rows.append({
            "Dataset": prof["name"],
            "Category": prof["category"],
            "Operating Regimes": f"{prof['conditions_count']} ({prof['conditions_description']})",
            "Fault Modes": ", ".join(prof["fault_modes"]),
            "Train Fleet": f"{prof['train_engines']} Units",
            "Test Fleet": f"{prof['test_engines']} Units",
            "Complexity Tier": prof["complexity"]
        })

    # Add related rotating machinery datasets researched
    rows.append({
        "Dataset": "NASA PHM 2008 Challenge",
        "Category": "Aircraft Turbofan",
        "Operating Regimes": "1 (Sea Level Static)",
        "Fault Modes": "HPC Blade Tip Clearance & Flow Degradation",
        "Train Fleet": "218 Units",
        "Test Fleet": "218 Units",
        "Complexity Tier": "Historical Precursor to C-MAPSS FD001-FD004"
    })
    rows.append({
        "Dataset": "IMS Bearing Dataset (NASA/UC)",
        "Category": "Industrial Bearings",
        "Operating Regimes": "1 (Constant 2000 RPM, 6000 lbs radial load)",
        "Fault Modes": "Outer Race, Inner Race & Roller Element Flaking",
        "Train Fleet": "3 Run-to-Failure Tests (4 Bearings each)",
        "Test Fleet": "Continuous Vibration Logs",
        "Complexity Tier": "High-Frequency Vibration Telemetry (20 kHz)"
    })
    rows.append({
        "Dataset": "FEMTO-ST / PRONOSTIA (IEEE PHM 2012)",
        "Category": "Rotating Machinery",
        "Operating Regimes": "3 Operating Conditions (Speed & Load variations)",
        "Fault Modes": "Accelerated Bearing Degradation",
        "Train Fleet": "6 Run-to-Failure Experiments",
        "Test Fleet": "11 Truncated Validation Runs",
        "Complexity Tier": "Multi-Condition Accelerated Life Testing"
    })

    st.dataframe(pd.DataFrame(rows), width="stretch")

# ---------------------------------------------------------------------------
# Tab 2: Physical Sensor Dictionary
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("2. Physical Sensor Telemetry Dictionary & Engineering Units")
    st.markdown(
        "Canonical thermodynamic and mechanical sensor definitions based on NASA C-MAPSS technical specifications. "
        "Highlights invariant channels in FD001 and core degradation indicators."
    )

    sensor_records = []
    for s_id, s_info in CMAPSS_SENSOR_METADATA.items():
        sensor_records.append({
            "Sensor ID": s_id,
            "Symbol": s_info["symbol"],
            "Parameter Name": s_info["name"],
            "Units": s_info["units"],
            "Nominal Value": s_info["nominal_value"],
            "FD001 Invariant": "⚠️ Constant (Filtered)" if s_info["is_invariant_fd001"] else "✅ Active Indicator",
            "Physical Engineering Role": s_info["description"]
        })

    st.dataframe(pd.DataFrame(sensor_records), width="stretch")

    st.markdown("##### Flight Condition Operating Settings")
    op_records = []
    for op_id, op_info in CMAPSS_OP_SETTINGS_METADATA.items():
        op_records.append({
            "Setting ID": op_id,
            "Engineering Parameter": op_info["name"],
            "Units": op_info["units"],
            "Operational Range": f"{op_info['range'][0]} to {op_info['range'][1]} {op_info['units']}",
            "Physical Role": op_info["description"]
        })
    st.table(pd.DataFrame(op_records))

# ---------------------------------------------------------------------------
# Tab 3: Automated Data Quality Validator
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("3. Automated Dataset Quality & Integrity Validator")
    st.markdown(
        "Runs zero-leakage automated inspection verifying absence of missing values, floating-point infinities, "
        "strictly monotonic cycle sequence progression, and target label alignment."
    )

    val_dataset = st.selectbox(
        "Select Dataset to Inspect",
        DatasetRegistry.get_available_datasets(),
        index=0
    )

    if st.button(f"🚀 Execute Quality Audit on {val_dataset}"):
        with st.spinner("Validating dataset..."):
            report = DatasetRegistry.validate_dataset(val_dataset)

            col_v1, col_v2, col_v3 = st.columns(3)
            col_v1.metric("Validation Status", "PASSED ✅" if report.get("is_valid") else "FAILED ❌")
            col_v2.metric("Train Fleet Engines", f"{report.get('train_engines_count', 0)} Units")
            col_v3.metric("Total Operational Snapshots", f"{report.get('train_total_cycles', 0):,} Cycles")

            st.json(report)

# ---------------------------------------------------------------------------
# Tab 4: Cross-Dataset Generalization Benchmark
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("4. Empirical Cross-Dataset Generalization & Domain Shift Benchmark")
    st.markdown(
        "Empirical evaluation of a Deep LSTM model trained strictly on **FD001** (Sea level, single HPC fault) "
        "and deployed zero-shot across **FD003** (compound faults), **FD002** (6 flight regimes), and **FD004** (compound complex)."
    )

    results_file = PROJECT_ROOT / "results" / "cross_dataset_evaluation.json"
    if results_file.exists():
        with open(results_file, "r", encoding="utf-8") as f:
            bench_data = json.load(f)

        comps = bench_data.get("comparisons", {})
        comp_df = pd.DataFrame(comps).T
        comp_df.index.name = "Dataset"
        comp_df = comp_df.reset_index()

        st.markdown("##### Performance Degradation Summary Across Environments")
        st.dataframe(comp_df[[
            "Dataset", "mae_raw", "mae_clipped", "rmse", "nasa_score",
            "delta_mae", "mae_relative_increase_pct"
        ]], width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            chart_mae = pd.DataFrame({
                "Dataset": comp_df["Dataset"],
                "MAE Raw (Cycles)": comp_df["mae_raw"],
                "MAE Clipped (Cycles)": comp_df["mae_clipped"],
                "RMSE (Cycles)": comp_df["rmse"]
            }).set_index("Dataset")
            st.markdown("##### Error Metric Progression")
            st.bar_chart(chart_mae, height=320)

        with c2:
            chart_rel = pd.DataFrame({
                "Dataset": comp_df["Dataset"],
                "Relative MAE Increase (%)": comp_df["mae_relative_increase_pct"]
            }).set_index("Dataset")
            st.markdown("##### Relative Performance Drop (% vs FD001)")
            st.bar_chart(chart_rel, height=320)

        st.markdown("""
        > [!IMPORTANT]
        > **Key Domain Shift Findings:**
        > 1. **Compound Fault Shift (FD003):** MAE increases by **+129.4%** (11.69 → 26.83 cycles) and NASA score explodes from 537 to 92,746. Unmodeled Fan assembly failure causes the model to severely underestimate late-stage degradation.
        > 2. **Multi-Regime Condition Shift (FD002):** MAE increases by **+277.9%** (11.69 → 44.19 cycles). Flight condition variations (Mach 0-0.84, Altitude 0-42k ft) introduce massive baseline sensor drift that masks true thermal degradation without regime normalization.
        > 3. **Compound Complexity (FD004):** MAE increases by **+354.4%** (11.69 → 53.13 cycles), demonstrating why multi-regime normalization and fault-adaptive architectures are mandatory for production aerospace deployments.
        """)
    else:
        st.info("Cross-dataset evaluation results not found. Run python -m src.cross_dataset_experiment to generate.")
