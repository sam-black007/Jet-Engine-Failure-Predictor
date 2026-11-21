"""JETWISE AI — AI Configuration Centre (Streamlit Dashboard).

Comprehensive 10-section configuration and research console:
1. Dataset Configuration
2. Preprocessing Configuration
3. Model Configuration
4. Loss Configuration
5. Training Configuration
6. Hyperparameter Optimization
7. Prediction Configuration & Uncertainty
8. Evaluation Configuration
9. Experiment Manager
10. Model Registry
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config_manager import ConfigManager, MasterConfig
from src.configurable_data import ConfigurableDataPipeline
from src.model_registry import ModelRegistry
from src.experiment_tracker import ExperimentTracker
from src.hyperparameter_optimizer import HyperparameterOptimizer
from src.prediction_engine import PredictionEngine
from src.model_factory import count_parameters, is_deep_learning_model

st.set_page_config(
    page_title="JETWISE AI — AI Configuration Centre",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Aerospace Dark Theme Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .main-title {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #38bdf8;
        font-weight: 700;
        letter-spacing: 1.5px;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 25px;
    }
    .config-card {
        background-color: #161e2e;
        border: 1px solid #1e293b;
        border-left: 4px solid #38bdf8;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .metric-badge {
        background-color: #1e293b;
        color: #38bdf8;
        padding: 4px 8px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# State Initialization
# ---------------------------------------------------------------------------
cm = ConfigManager()
if "active_cfg" not in st.session_state:
    st.session_state.active_cfg = cm.load_config()

cfg: MasterConfig = st.session_state.active_cfg
tracker = ExperimentTracker()
registry = ModelRegistry()

# ---------------------------------------------------------------------------
# Header & Profile Quick Load
# ---------------------------------------------------------------------------
st.markdown("<h1 class='main-title'>✈️ JETWISE AI — AI CONFIGURATION CENTRE</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Centralized Prognostics Configuration, Architecture Laboratory & MLOps Control</p>", unsafe_allow_html=True)

col_top1, col_top2, col_top3 = st.columns([2, 2, 2])

with col_top1:
    profiles = cm.list_profiles()
    selected_prof = st.selectbox(
        "Load Configuration Profile",
        ["(Select Profile...)"] + profiles,
        help="Quickly populate full pipeline settings from saved YAML profile"
    )
    if selected_prof != "(Select Profile...)":
        st.session_state.active_cfg = cm.load_profile(selected_prof)
        st.success(f"Loaded profile: '{selected_prof}'")
        st.rerun()

with col_top2:
    st.metric("Schema Version", cfg.version)

with col_top3:
    st.metric("Active Model Tag", registry.get_active_tag())

st.markdown("---")

# ---------------------------------------------------------------------------
# 10 Configurable Subsections (Tabbed / Grouped Interface)
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "1. Dataset",
    "2. Preprocessing",
    "3. Architecture",
    "4. Loss Function",
    "5. Training Engine",
    "6. HPO Tuning",
    "7. Prediction & Uncertainty",
    "8. Evaluation",
    "9. Experiment Manager",
    "10. Model Registry"
])

# ---------------------------------------------------------------------------
# TAB 1: DATASET CONFIGURATION
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("1. Dataset & Sensor Channel Selection")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        st.markdown("**Dataset Parameters**")
        subset = st.selectbox(
            "C-MAPSS Subset",
            ["FD001", "FD002", "FD003", "FD004"],
            index=["FD001", "FD002", "FD003", "FD004"].index(cfg.dataset.subset),
            help="FD001: Sea Level, Single Fault. FD002/004: Multi-Condition."
        )
        cfg.dataset.subset = subset

        split_ratio = st.slider(
            "Engine Train/Validation Split",
            min_value=0.50, max_value=0.90, value=cfg.dataset.train_val_split, step=0.05,
            help="Split performed strictly on engine IDs to prevent sequence leakage."
        )
        cfg.dataset.train_val_split = split_ratio

        seed = st.number_input("Split Random Seed", value=cfg.dataset.seed, step=1)
        cfg.dataset.seed = int(seed)

    with col_d2:
        st.markdown("**Sensor Selection Mode**")
        mode = st.selectbox(
            "Selection Mode",
            ["auto_variance", "manual", "all"],
            index=["auto_variance", "manual", "all"].index(cfg.sensor.selection_mode)
        )
        cfg.sensor.selection_mode = mode

        var_thresh = st.number_input(
            "Variance Threshold", min_value=0.0, value=cfg.sensor.variance_threshold, step=0.001,
            format="%.4f"
        )
        cfg.sensor.variance_threshold = float(var_thresh)

        include_ops = st.checkbox(
            "Include Operational Settings (op_setting_1..3)",
            value=cfg.sensor.include_operational_settings
        )
        cfg.sensor.include_operational_settings = include_ops

# ---------------------------------------------------------------------------
# TAB 2: PREPROCESSING CONFIGURATION
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("2. Preprocessing & Temporal Sequencing")
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        scaler = st.selectbox(
            "Normalization Method (Fitted strictly on train engines)",
            ["standard", "minmax", "robust"],
            index=["standard", "minmax", "robust"].index(cfg.preprocessing.scaler_type)
        )
        cfg.preprocessing.scaler_type = scaler

        window_size = st.slider(
            "Sequence Window Length (Cycles)",
            min_value=10, max_value=100, value=cfg.preprocessing.window_size, step=5
        )
        cfg.preprocessing.window_size = window_size

        stride = st.number_input("Sliding Window Stride", min_value=1, max_value=10, value=cfg.preprocessing.stride)
        cfg.preprocessing.stride = int(stride)

    with col_p2:
        padding_mode = st.selectbox(
            "Short-Trajectory Padding Mode",
            ["first", "zero", "none"],
            index=["first", "zero", "none"].index(cfg.preprocessing.padding_mode)
        )
        cfg.preprocessing.padding_mode = padding_mode

        clip_train = st.slider(
            "Training Target RUL Clip (Cycles)",
            min_value=50, max_value=200, value=cfg.preprocessing.rul_clip_train, step=5
        )
        cfg.preprocessing.rul_clip_train = clip_train

        clip_eval = st.slider(
            "Evaluation RUL Clip (Cycles)",
            min_value=50, max_value=200, value=cfg.preprocessing.rul_clip_eval, step=5
        )
        cfg.preprocessing.rul_clip_eval = clip_eval

# ---------------------------------------------------------------------------
# TAB 3: MODEL ARCHITECTURE CONFIGURATION
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("3. Model Architecture & Hyperparameters")
    col_m1, col_m2 = st.columns([1, 2])

    with col_m1:
        arch = st.selectbox(
            "Target Architecture",
            ["lstm", "attention_lstm", "tcn", "transformer", "linear_regression", "random_forest", "gradient_boosting"],
            index=["lstm", "attention_lstm", "tcn", "transformer", "linear_regression", "random_forest", "gradient_boosting"].index(cfg.model.architecture)
        )
        cfg.model.architecture = arch

        dropout = st.slider("Dropout Rate", 0.0, 0.6, value=float(cfg.model.dropout), step=0.05)
        cfg.model.dropout = dropout

    with col_m2:
        if arch in ("lstm", "attention_lstm"):
            st.markdown("**LSTM Hyperparameters**")
            c1, c2, c3 = st.columns(3)
            with c1:
                h1 = st.selectbox("Hidden Dim 1", [32, 64, 128], index=[32, 64, 128].index(cfg.model.lstm_hidden_dim1) if cfg.model.lstm_hidden_dim1 in [32, 64, 128] else 1)
                cfg.model.lstm_hidden_dim1 = h1
            with c2:
                h2 = st.selectbox("Hidden Dim 2", [16, 32, 64], index=[16, 32, 64].index(cfg.model.lstm_hidden_dim2) if cfg.model.lstm_hidden_dim2 in [16, 32, 64] else 1)
                cfg.model.lstm_hidden_dim2 = h2
            with c3:
                bidi = st.checkbox("Bidirectional", value=cfg.model.bidirectional)
                cfg.model.bidirectional = bidi

            if arch == "attention_lstm":
                att_dim = st.slider("Attention Layer Dimension", 8, 128, value=cfg.model.attention_dim, step=8)
                cfg.model.attention_dim = att_dim

        elif arch == "tcn":
            st.markdown("**Temporal Convolutional Network Hyperparameters**")
            t_k = st.selectbox("Kernel Size", [2, 3, 5, 7], index=[2, 3, 5, 7].index(cfg.model.tcn_kernel_size))
            cfg.model.tcn_kernel_size = t_k
            t_drop = st.slider("TCN Dropout", 0.0, 0.5, value=float(cfg.model.tcn_dropout), step=0.05)
            cfg.model.tcn_dropout = t_drop

        elif arch == "transformer":
            st.markdown("**Time-Series Transformer Hyperparameters**")
            c1, c2, c3 = st.columns(3)
            with c1:
                nhead = st.selectbox("Attention Heads", [2, 4, 8], index=[2, 4, 8].index(cfg.model.transformer_nhead))
                cfg.model.transformer_nhead = nhead
            with c2:
                d_model = st.selectbox("Embedding Dim (d_model)", [32, 64, 128], index=[32, 64, 128].index(cfg.model.transformer_d_model) if cfg.model.transformer_d_model in [32, 64, 128] else 1)
                if d_model % nhead != 0:
                    st.error(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
                else:
                    cfg.model.transformer_d_model = d_model
            with c3:
                n_layers = st.selectbox("Encoder Layers", [1, 2, 3, 4], index=[1, 2, 3, 4].index(cfg.model.transformer_num_layers))
                cfg.model.transformer_num_layers = n_layers

# ---------------------------------------------------------------------------
# TAB 4: LOSS FUNCTION CONFIGURATION
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("4. Loss Function Configuration")
    col_l1, col_l2 = st.columns(2)

    with col_l1:
        loss_name = st.selectbox(
            "Loss Formulation",
            ["pinball", "nasa_asymmetric", "smooth_pinball", "mse", "mae", "huber"],
            index=["pinball", "nasa_asymmetric", "smooth_pinball", "mse", "mae", "huber"].index(cfg.loss.name)
        )
        cfg.loss.name = loss_name

        if loss_name in ("pinball", "smooth_pinball"):
            q_val = st.slider(
                "Pinball Quantile (q < 0.50 penalizes late overestimation)",
                min_value=0.10, max_value=0.90, value=float(cfg.loss.pinball_quantile), step=0.05
            )
            cfg.loss.pinball_quantile = float(q_val)
            ratio = (1.0 - q_val) / max(1e-4, q_val)
            st.info(f"Overestimation Penalty Ratio: **{ratio:.2f}x** more severe than underestimation.")

    with col_l2:
        if loss_name == "nasa_asymmetric":
            st.markdown("**Official NASA Asymmetric Divisors**")
            lp = st.number_input("Late Penalty Divisor (d >= 0)", value=cfg.loss.late_penalty)
            ep = st.number_input("Early Penalty Divisor (d < 0)", value=cfg.loss.early_penalty)
            cfg.loss.late_penalty = float(lp)
            cfg.loss.early_penalty = float(ep)

        elif loss_name == "huber":
            delta = st.number_input("Huber Transition Delta", value=cfg.loss.huber_delta)
            cfg.loss.huber_delta = float(delta)

# ---------------------------------------------------------------------------
# TAB 5: TRAINING ENGINE CONFIGURATION
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("5. Training Engine & Optimization Routine")
    col_t1, col_t2, col_t3 = st.columns(3)

    with col_t1:
        epochs = st.number_input("Max Epochs", min_value=1, max_value=200, value=cfg.training.epochs)
        cfg.training.epochs = int(epochs)

        batch_size = st.selectbox(
            "Batch Size", [16, 32, 64, 128, 256],
            index=[16, 32, 64, 128, 256].index(cfg.training.batch_size) if cfg.training.batch_size in [16, 32, 64, 128, 256] else 2
        )
        cfg.training.batch_size = int(batch_size)

        lr = st.number_input(
            "Learning Rate", min_value=1e-5, max_value=0.1, value=float(cfg.training.learning_rate),
            format="%.5f"
        )
        cfg.training.learning_rate = float(lr)

    with col_t2:
        optimizer = st.selectbox(
            "Optimizer", ["adam", "adamw", "rmsprop", "sgd"],
            index=["adam", "adamw", "rmsprop", "sgd"].index(cfg.training.optimizer)
        )
        cfg.training.optimizer = optimizer

        scheduler = st.selectbox(
            "Learning Rate Scheduler", ["plateau", "step", "cosine", "none"],
            index=["plateau", "step", "cosine", "none"].index(cfg.training.scheduler)
        )
        cfg.training.scheduler = scheduler

        grad_clip = st.number_input("Gradient Clip Norm", value=cfg.training.gradient_clip_norm)
        cfg.training.gradient_clip_norm = float(grad_clip)

    with col_t3:
        es_enabled = st.checkbox("Enable Early Stopping", value=cfg.training.early_stopping)
        cfg.training.early_stopping = es_enabled

        es_patience = st.number_input("Early Stopping Patience", value=cfg.training.early_stopping_patience)
        cfg.training.early_stopping_patience = int(es_patience)

# ---------------------------------------------------------------------------
# TAB 6: HYPERPARAMETER OPTIMIZATION (OPTUNA)
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("6. Automated Hyperparameter Optimization (Optuna)")
    st.markdown("Tune hyperparameters strictly on validation engines (Zero Test Set Contamination).")

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        hpo_arch = st.selectbox("HPO Architecture", ["lstm", "attention_lstm", "tcn", "transformer"])
        hpo_obj = st.selectbox("Objective Metric", ["nasa_score", "mae", "balanced"])
        hpo_trials = st.slider("Number of Trials", min_value=2, max_value=30, value=5)
        hpo_epochs = st.slider("Max Epochs per Trial", min_value=1, max_value=15, value=3)

    with col_h2:
        hpo_sampler = st.selectbox("Sampler Strategy", ["TPE (Bayesian)", "Random"])
        if st.button("🚀 Run HPO Study", key="run_hpo_btn"):
            with st.spinner("Executing Optuna study over validation split..."):
                optimizer = HyperparameterOptimizer(cfg, study_name=f"ui_{hpo_arch}")
                hpo_res = optimizer.optimize(
                    architecture=hpo_arch,
                    objective_metric=hpo_obj,
                    n_trials=hpo_trials,
                    max_epochs_per_trial=hpo_epochs,
                    sampler_type="tpe" if "TPE" in hpo_sampler else "random"
                )
                st.success(f"Study completed! Best validation {hpo_obj}: **{hpo_res['best_value']}**")
                st.json(hpo_res["best_params"])

# ---------------------------------------------------------------------------
# TAB 7: PREDICTION STRATEGIES & UNCERTAINTY
# ---------------------------------------------------------------------------
with tabs[6]:
    st.subheader("7. Prediction Engine & Uncertainty Quantification")
    col_u1, col_u2 = st.columns(2)

    with col_u1:
        strat = st.selectbox(
            "Inference Strategy",
            ["standard", "conservative", "quantile", "mc_dropout", "ensemble"],
            index=["standard", "conservative", "quantile", "mc_dropout", "ensemble"].index(cfg.prediction.strategy)
        )
        cfg.prediction.strategy = strat

        bias_shift = st.slider(
            "Conservative Bias Shift (Cycles)",
            min_value=0.0, max_value=15.0, value=float(cfg.prediction.bias_offset or 3.0), step=0.5,
            help="Subtracted from point estimate to enforce safety margin."
        )
        cfg.prediction.bias_offset = float(bias_shift)

    with col_u2:
        mc_samples = st.slider("MC Dropout Forward Passes", min_value=5, max_value=100, value=cfg.prediction.mc_samples, step=5)
        cfg.prediction.mc_samples = mc_samples

        conf_level = st.slider("Nominal Confidence Interval", min_value=0.80, max_value=0.99, value=float(cfg.prediction.confidence_level), step=0.01)
        cfg.prediction.confidence_level = float(conf_level)

        clip_zero = st.checkbox("Clip Predictions to Non-Negative [0, inf)", value=cfg.prediction.clip_output_to_zero)
        cfg.prediction.clip_output_to_zero = clip_zero

# ---------------------------------------------------------------------------
# TAB 8: EVALUATION CONFIGURATION
# ---------------------------------------------------------------------------
with tabs[7]:
    st.subheader("8. Official Benchmark & Metric Selection")
    st.markdown("NASA C-MAPSS FD001 Official Benchmark Protocol.")
    st.checkbox("Compute Raw NASA Asymmetric Penalty", value=True, disabled=True)
    st.checkbox("Compute Clipped-125 NASA Asymmetric Penalty", value=True, disabled=True)
    st.checkbox("Compute Standard MAE & RMSE", value=True, disabled=True)
    st.checkbox("Log Engine-Level Residuals and Bias Tracking", value=True, disabled=True)

# ---------------------------------------------------------------------------
# TAB 9: EXPERIMENT MANAGER
# ---------------------------------------------------------------------------
with tabs[8]:
    st.subheader("9. Persistent Experiment Management")
    st.markdown("All completed training and evaluation runs stored in SQLite `experiments/experiments.db`.")

    exp_df = tracker.list_experiments()
    if not exp_df.empty:
        st.dataframe(
            exp_df[[
                "experiment_id", "profile_name", "model_architecture",
                "loss_name", "val_nasa", "test_nasa_raw", "test_mae", "created_at"
            ]],
            use_container_width=True
        )

        csv_bytes = exp_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export Experiment History to CSV",
            csv_bytes,
            "jetwise_experiments.csv",
            "text/csv"
        )
    else:
        st.info("No recorded experiments in database yet.")

    # Snapshot exporter
    if st.button("💾 Save Current Configuration Snapshot"):
        out_snap = cm.save_snapshot(cfg, PROJECT_ROOT / "results" / "snapshots")
        st.success(f"Snapshot saved to `{out_snap}`")

# ---------------------------------------------------------------------------
# TAB 10: MODEL REGISTRY
# ---------------------------------------------------------------------------
with tabs[9]:
    st.subheader("10. Deployed Models & Checkpoint Registry")
    st.markdown("Registered prognostic checkpoints cataloged in `models/model_registry.json`.")

    models_list = registry.list_models()
    if models_list:
        model_records = []
        for m in models_list:
            tag = m["tag"]
            is_active = (tag == registry.get_active_tag())
            metrics = m.get("metrics", {})
            model_records.append({
                "Tag": tag,
                "Model Name": m["name"],
                "Architecture": m["architecture"],
                "Active Status": "⭐ ACTIVE (Production)" if is_active else "Standby",
                "Raw NASA Score": metrics.get("NASA_score_raw", "N/A"),
                "MAE": metrics.get("MAE", "N/A"),
                "Registered": m.get("registered_at", "")[:19]
            })
        st.dataframe(pd.DataFrame(model_records), use_container_width=True)

        # Set active model
        model_tags = [m["tag"] for m in models_list]
        selected_active = st.selectbox("Set Production Active Model", model_tags, index=model_tags.index(registry.get_active_tag()))
        if st.button("Apply Active Production Model"):
            registry.set_active_model(selected_active)
            st.success(f"Model '{selected_active}' is now the primary active production model!")
            st.rerun()
    else:
        st.info("No models registered in ModelRegistry.")
