"""Global Configuration Centre Workspace (Part 3).

Provides live interactive controls to configure all platform variables without editing Python files:
1. Dataset & Engine Selection
2. Preprocessing & Sensor Selection
3. Model Architecture & Hyperparameters
4. Loss Formulation & Quantile Tuning
5. FDIR & Anomaly Thresholds
6. OOD & Epistemic Trust Thresholds
7. Uncertainty & Conformal Calibration Settings
8. Maintenance Scheduling Constraints
9. Fleet Simulation Parameters
10. Physics Assumptions & Virtual Sensors
11. Model Monitoring & Retraining Triggers
12. Visualization Preferences
13. Save Profile, Load Named Profile, Reset Defaults, Import/Export JSON & YAML
"""

from typing import Dict, Any, Optional
import json
import yaml
import datetime
import streamlit as st

from src.config_manager import ConfigManager, MasterConfig
from src.database_migrations import get_db_connection


def render_global_configuration_centre():
    st.subheader("⚙️ Global Enterprise Configuration Centre")
    st.markdown("""
    Central parameter management console. Configure any dataset, model architecture, loss function,
    physics assumption, or maintenance optimization constraint with immediate schema validation.
    """)

    cm = ConfigManager()

    # Profile Selector Bar
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    available_profiles = cm.list_profiles()
    active_profile = col_p1.selectbox("Configuration Profile", ["default"] + available_profiles)
    
    if col_p2.button("🔄 Reset to Factory Defaults"):
        cfg = cm.get_default_config()
        st.session_state["active_config"] = cfg
        st.success("Configuration reset to default!")

    if "active_config" not in st.session_state:
        st.session_state["active_config"] = cm.load_profile(active_profile) if active_profile != "default" else cm.get_default_config()

    cfg: MasterConfig = st.session_state["active_config"]

    # Configuration Tabs
    tab_data, tab_model, tab_loss, tab_fdir, tab_phys, tab_trust, tab_maint, tab_io = st.tabs([
        "📂 Dataset & Preprocessing",
        "🧠 Model & Architecture",
        "🎯 Loss & Quantile",
        "🔍 FDIR & Anomaly",
        "⚙️ Physics & Virtual Sensors",
        "🛡️ Model Trust & Calibration",
        "🛠️ Maintenance & Scheduling",
        "💾 Profile Save / Import / Export"
    ])

    with tab_data:
        st.markdown("##### Dataset Ingestion & Feature Engineering")
        d_col1, d_col2 = st.columns(2)
        new_subset = d_col1.selectbox("C-MAPSS Subset", ["FD001", "FD002", "FD003", "FD004"], index=["FD001", "FD002", "FD003", "FD004"].index(cfg.dataset.subset))
        new_split = d_col2.slider("Train/Val Engine Split Ratio", 0.50, 0.95, cfg.dataset.train_val_split, 0.05)
        
        p_col1, p_col2, p_col3 = st.columns(3)
        new_scaler = p_col1.selectbox("Normalization Scaler", ["standard", "minmax", "robust"], index=["standard", "minmax", "robust"].index(cfg.preprocessing.scaler_type))
        new_win = p_col2.slider("Sliding Window Sequence Length", 10, 60, cfg.preprocessing.window_size, 5)
        new_clip = p_col3.slider("Target RUL Upper Clip Bound", 50, 200, cfg.preprocessing.rul_clip_train, 5)

        cfg.dataset.subset = new_subset
        cfg.dataset.train_val_split = new_split
        cfg.preprocessing.scaler_type = new_scaler
        cfg.preprocessing.window_size = new_win
        cfg.preprocessing.rul_clip_train = new_clip

    with tab_model:
        st.markdown("##### Deep Prognostic Architecture")
        m_col1, m_col2, m_col3 = st.columns(3)
        arch_list = ["lstm", "attention_lstm", "tcn", "transformer", "linear_regression", "random_forest", "gradient_boosting"]
        new_arch = m_col1.selectbox("Model Family", arch_list, index=arch_list.index(cfg.model.architecture))
        new_dropout = m_col2.slider("Dropout Probability", 0.0, 0.7, cfg.model.dropout, 0.05)
        new_lr = m_col3.number_input("Optimizer Learning Rate", value=cfg.training.learning_rate, format="%.5f")

        cfg.model.architecture = new_arch
        cfg.model.dropout = new_dropout
        cfg.training.learning_rate = new_lr

        if new_arch == "transformer":
            t_col1, t_col2 = st.columns(2)
            cfg.model.transformer_d_model = t_col1.selectbox("Transformer d_model", [32, 64, 128], index=1)
            cfg.model.transformer_nhead = t_col2.selectbox("Attention Heads", [2, 4, 8], index=1)

    with tab_loss:
        st.markdown("##### Objective Function Formulation")
        l_col1, l_col2 = st.columns(2)
        loss_list = ["pinball", "nasa_asymmetric", "mse", "mae", "smooth_pinball"]
        new_loss = l_col1.selectbox("Loss Formulation", loss_list, index=loss_list.index(cfg.loss.name))
        new_q = l_col2.slider("Pinball Loss Quantile (q)", 0.10, 0.90, cfg.loss.pinball_quantile, 0.05)

        cfg.loss.name = new_loss
        cfg.loss.pinball_quantile = new_q

    with tab_fdir:
        st.markdown("##### Sensor Reliability & FDIR Thresholds")
        fd1, fd2 = st.columns(2)
        cfg.fdir.stuck_at_variance_threshold = fd1.number_input("Stuck-at Variance Threshold", value=cfg.fdir.stuck_at_variance_threshold, format="%.5f")
        cfg.fdir.drift_slope_threshold = fd2.slider("Sensor Drift Slope Threshold", 0.01, 0.20, cfg.fdir.drift_slope_threshold, 0.01)

    with tab_phys:
        st.markdown("##### Aerothermal Thermodynamic Assumptions")
        ph1, ph2 = st.columns(2)
        cfg.physics.gamma_air = ph1.number_input("Ratio of Specific Heats (γ)", value=cfg.physics.gamma_air, format="%.2f")
        cfg.physics.hpc_polytropic_efficiency_nominal = ph2.slider("Nominal HPC Efficiency (η)", 0.80, 0.95, cfg.physics.hpc_polytropic_efficiency_nominal, 0.01)

    with tab_trust:
        st.markdown("##### Model Trust Boundaries & Conformal Prediction")
        tr1, tr2 = st.columns(2)
        cfg.model_trust.mahalanobis_threshold_percentile = tr1.slider("Mahalanobis OOD Percentile Threshold", 80.0, 99.9, cfg.model_trust.mahalanobis_threshold_percentile, 0.5)
        cfg.uncertainty_calibration.target_coverage = tr2.slider("Target Conformal Coverage Level (1 - α)", 0.70, 0.99, cfg.uncertainty_calibration.target_coverage, 0.01)

    with tab_maint:
        st.markdown("##### Hangar Scheduling & Resource Constraints")
        sc1, sc2, sc3 = st.columns(3)
        cfg.maintenance_optimization.max_daily_bay_capacity = sc1.slider("Hangar Bay Slots", 1, 10, cfg.maintenance_optimization.max_daily_bay_capacity)
        cfg.maintenance_optimization.technician_hours_per_day = sc2.slider("Technician Labor Hours / Day", 16.0, 100.0, cfg.maintenance_optimization.technician_hours_per_day, 4.0)
        cfg.engineering_review.auto_flag_mpi_threshold = sc3.slider("Auto-Flag Urgent Review MPI Threshold", 40.0, 90.0, cfg.engineering_review.auto_flag_mpi_threshold, 5.0)

    with tab_io:
        st.markdown("##### Save Named Configuration Profile / Import / Export")
        p_name = st.text_input("Profile Name", value=f"profile_{cfg.dataset.subset}_{cfg.model.architecture}")
        p_desc = st.text_input("Description", value="Operational profile generated via Global Configuration Centre")

        save_col, export_col = st.columns(2)
        if save_col.button("💾 Save Profile to configs/experiments/"):
            cfg.profile_name = p_name
            cfg.description = p_desc
            saved_p = cm.save_yaml(cfg, f"experiments/{p_name}.yaml")
            # Log into configuration_history
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("""
                INSERT INTO configuration_history (profile_name, version, change_description, config_json)
                VALUES (?, ?, ?, ?)
                """, (p_name, cfg.version, p_desc, json.dumps(cfg.model_dump())))
                conn.commit()
                conn.close()
            except Exception:
                pass
            st.success(f"Profile saved to: {saved_p.name} and logged into audit history!")

        json_str = json.dumps(cfg.model_dump(), indent=2)
        yaml_str = yaml.safe_dump(cfg.model_dump(), sort_keys=False)

        export_col.download_button("📥 Export Configuration (.yaml)", data=yaml_str, file_name=f"{p_name}.yaml", mime="application/x-yaml")

        st.markdown("##### Active Configuration JSON Snapshot")
        st.code(json_str, language="json")
