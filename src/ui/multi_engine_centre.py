"""JETWISE AI — Multi-Manufacturer Engine Configuration & Simulation Centre Workspace.

Comprehensive interactive user interface for browsing manufacturers, building engine configurations,
inspecting imported engineering models, running cross-engine comparative analytics, validating AI compatibility,
and exploring architecture-specific 3D Digital Twins.
"""

from typing import Dict, List, Any, Optional
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import pathlib
import datetime

from src.engine_configuration.models import (
    Manufacturer,
    EngineFamily,
    EngineConfiguration,
    ImportedEngineModel,
    PropertyWithProvenance,
    ProvenanceSource,
    EngineValidationStatus,
    ShaftArrangement,
    EngineType
)
from src.engine_configuration.repository import EngineRepository
from src.engine_configuration.validation import EngineValidator
from src.engine_configuration.model_importer import ModelImporter, EngineModelAdapter
from src.engine_configuration.engine_comparator import EngineComparator
from src.engine_configuration.multi_twin_generator import render_multi_engine_3d_twin_html


def render_multi_engine_centre():
    """Render the full Multi-Manufacturer Engine Configuration Centre."""
    
    # Initialize repository
    if "engine_repo" not in st.session_state:
        st.session_state.engine_repo = EngineRepository()
    repo: EngineRepository = st.session_state.engine_repo

    # Top Aerospace Header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #0B132B 0%, #1C2541 100%); padding: 18px 24px; border-radius: 8px; border-left: 5px solid #00B8D9; margin-bottom: 20px;">
        <h2 style="margin: 0; color: #F8FAFC; display: flex; align-items: center; gap: 10px;">
            🌐 Multi-Manufacturer Engine Configuration & Simulation Centre
        </h2>
        <p style="margin: 6px 0 0 0; color: #94A3B8; font-size: 0.95rem;">
            Extensible multi-engine aerospace architecture: Rolls-Royce (3-spool), Pratt & Whitney (Geared GTF), GE Aerospace, Honeywell, CFM, Safran, and NASA benchmarks.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 6 Interactive Tabs
    tab_lib, tab_build, tab_imp, tab_comp, tab_ai, tab_twin = st.tabs([
        "🏢 Manufacturer Library",
        "⚙️ Engine Configuration Builder",
        "📥 Model & Dataset Importer",
        "⚖️ Multi-Engine Comparison Lab",
        "🧠 AI Model Compatibility",
        "🌐 Multi-Engine 3D Digital Twin"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: MANUFACTURER LIBRARY
    # -------------------------------------------------------------------------
    with tab_lib:
        st.markdown("### Aerospace Engine Manufacturer & Family Catalogue")
        st.caption("Extensible library of gas turbine original equipment manufacturers (OEMs) and certified engine families.")

        manufacturers = repo.get_all_manufacturers()

        m_cols = st.columns(3)
        for idx, m in enumerate(manufacturers):
            col = m_cols[idx % 3]
            with col:
                st.markdown(f"""
                <div style="background: #111827; border: 1px solid #1F2937; border-top: 3px solid #00B8D9; padding: 14px; border-radius: 6px; margin-bottom: 12px; height: 210px; display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div style="font-size: 1.1rem; font-weight: bold; color: #F8FAFC;">{m.name}</div>
                        <div style="color: #9CA3AF; font-size: 0.75rem; margin-top: 2px;">📍 {m.headquarters} ({m.country})</div>
                        <div style="color: #CBD5E1; font-size: 0.8rem; margin-top: 8px; line-height: 1.3;">{m.description[:110]}...</div>
                    </div>
                    <div>
                        <div style="color: #00B8D9; font-size: 0.75rem; font-weight: bold;">Families: {', '.join(m.engine_families[:3])}</div>
                        <a href="{m.website}" target="_blank" style="color: #38BDF8; font-size: 0.7rem; text-decoration: none;">Visit OEM Portal ↗</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### ➕ Add New Aircraft Engine Manufacturer")

        with st.expander("Register New Aerospace Manufacturer"):
            with st.form("form_add_manufacturer"):
                c1, c2 = st.columns(2)
                with c1:
                    new_m_id = st.text_input("Manufacturer ID (slug)", placeholder="e.g. ihi-aerospace")
                    new_m_name = st.text_input("Full Manufacturer Name", placeholder="e.g. IHI Aerospace")
                    new_m_country = st.text_input("Country of Origin", placeholder="e.g. Japan")
                with c2:
                    new_m_hq = st.text_input("Headquarters Location", placeholder="e.g. Tokyo, Japan")
                    new_m_web = st.text_input("Official Website", placeholder="https://www.ihi.co.jp")
                    new_m_desc = st.text_area("Description & Propulsion Specialties", placeholder="Key architectural competencies and military/civil programs...")
                
                new_families_raw = st.text_input("Engine Families (comma separated)", placeholder="e.g. F7, V2500, XF9")
                
                if st.form_submit_button("Register Manufacturer", type="primary"):
                    if new_m_id and new_m_name:
                        fam_list = [f.strip() for f in new_families_raw.split(",") if f.strip()]
                        m_obj = Manufacturer(
                            id=new_m_id.lower().strip(),
                            name=new_m_name.strip(),
                            country=new_m_country.strip(),
                            headquarters=new_m_hq.strip(),
                            description=new_m_desc.strip(),
                            website=new_m_web.strip(),
                            engine_families=fam_list,
                            is_custom=True
                        )
                        if repo.add_manufacturer(m_obj):
                            st.success(f"Manufacturer '{new_m_name}' registered successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to add manufacturer to repository.")
                    else:
                        st.warning("Please provide Manufacturer ID and Name.")

    # -------------------------------------------------------------------------
    # TAB 2: ENGINE CONFIGURATION BUILDER
    # -------------------------------------------------------------------------
    with tab_build:
        st.markdown("### Aircraft Engine Configuration Builder")
        st.caption("Create and customize detailed gas-path configurations with explicit provenance and scientific validation.")

        configs = repo.get_all_engine_configurations()
        config_options = ["➕ Create New Custom Engine Configuration"] + [f"{c.manufacturer} - {c.engine_model} ({c.config_id})" for c in configs]
        
        selected_option = st.selectbox("Select Engine Configuration to View / Edit", config_options)
        
        is_new = selected_option.startswith("➕")
        active_cfg: Optional[EngineConfiguration] = None
        if not is_new:
            selected_id = selected_option.split("(")[-1].replace(")", "").strip()
            active_cfg = repo.get_engine_configuration(selected_id)

        # Form for creating or editing
        with st.form("engine_builder_form"):
            st.markdown("#### 1. Identity & Manufacturer Association")
            id_col1, id_col2, id_col3 = st.columns(3)
            with id_col1:
                manuf_names = [m.name for m in manufacturers]
                m_default = manuf_names.index(active_cfg.manufacturer) if (active_cfg and active_cfg.manufacturer in manuf_names) else 0
                f_manuf = st.selectbox("Manufacturer", manuf_names, index=m_default)
            with id_col2:
                f_family = st.text_input("Engine Family", value=active_cfg.engine_family if active_cfg else "Generic Family")
            with id_col3:
                f_model = st.text_input("Engine Model Designation", value=active_cfg.engine_model if active_cfg else "Model-100")

            cfg_id_val = active_cfg.config_id if active_cfg else f"{f_manuf[:2].upper()}-{f_model.upper().replace(' ', '-')}"
            f_cfg_id = st.text_input("Configuration ID", value=cfg_id_val, disabled=not is_new)

            st.markdown("#### 2. Spool Mechanical Architecture")
            arch_col1, arch_col2, arch_col3 = st.columns(3)
            with arch_col1:
                f_type = st.selectbox(
                    "Engine Type",
                    [e.value for e in EngineType],
                    index=[e.value for e in EngineType].index(active_cfg.engine_type) if (active_cfg and active_cfg.engine_type in [e.value for e in EngineType]) else 0
                )
            with arch_col2:
                f_shaft = st.selectbox(
                    "Shaft Mechanical Arrangement",
                    [s.value for s in ShaftArrangement],
                    index=[s.value for s in ShaftArrangement].index(active_cfg.shaft_arrangement) if (active_cfg and active_cfg.shaft_arrangement in [s.value for s in ShaftArrangement]) else 1
                )
            with arch_col3:
                num_shafts = 3 if "Triple" in f_shaft else (2 if "Dual" in f_shaft or "Geared" in f_shaft else 1)
                f_num_shafts = st.number_input("Number of Concentric Shafts", min_value=1, max_value=3, value=num_shafts)

            st.markdown("#### 3. Key Physical Specifications & Provenance")
            st.caption("Assign explicit source provenance to every parameter to maintain scientific integrity.")

            p_col1, p_col2 = st.columns(2)
            prov_options = [p.value for p in ProvenanceSource]

            with p_col1:
                # Fan Diameter
                fan_v = float(active_cfg.fan_diameter_m.value) if active_cfg else 2.1
                f_fan_d = st.number_input("Fan Tip Diameter [m]", min_value=0.4, max_value=4.0, value=fan_v, step=0.05)
                f_fan_src = st.selectbox("Fan Diameter Provenance Source", prov_options, index=0)

                # Bypass Ratio
                bpr_v = float(active_cfg.bypass_ratio.value) if active_cfg else 9.0
                f_bpr = st.number_input("Bypass Ratio [-]", min_value=0.0, max_value=20.0, value=bpr_v, step=0.1)
                f_bpr_src = st.selectbox("Bypass Ratio Provenance Source", prov_options, index=0)

                # Overall Pressure Ratio
                opr_v = float(active_cfg.overall_pressure_ratio.value) if active_cfg else 40.0
                f_opr = st.number_input("Overall Pressure Ratio (OPR) [-]", min_value=5.0, max_value=70.0, value=opr_v, step=0.5)
                f_opr_src = st.selectbox("OPR Provenance Source", prov_options, index=0)

            with p_col2:
                # Takeoff Thrust
                th_v = float(active_cfg.takeoff_thrust_kn.value) if active_cfg else 120.0
                f_thrust = st.number_input("Takeoff Thrust [kN]", min_value=5.0, max_value=600.0, value=th_v, step=1.0)
                f_thrust_src = st.selectbox("Thrust Provenance Source", prov_options, index=0)

                # Cruise SFC
                sfc_v = float(active_cfg.cruise_sfc_kg_kn_hr.value) if active_cfg else 0.30
                f_sfc = st.number_input("Cruise Specific Fuel Consumption [kg/(kN*hr)]", min_value=0.15, max_value=0.80, value=sfc_v, step=0.01)
                f_sfc_src = st.selectbox("Cruise SFC Provenance Source", prov_options, index=1)

                # Dry Weight
                wt_v = float(active_cfg.dry_weight_kg.value) if active_cfg else 3000.0
                f_weight = st.number_input("Bare Engine Dry Weight [kg]", min_value=200.0, max_value=12000.0, value=wt_v, step=50.0)
                f_weight_src = st.selectbox("Dry Weight Provenance Source", prov_options, index=0)

            st.markdown("#### 4. 3D Digital Twin & Simulation Linkage")
            t_col1, t_col2 = st.columns(2)
            with t_col1:
                twin_templates = ["standard_twin_spool", "three_spool_trent", "geared_turbofan", "regional_turbofan"]
                twin_idx = twin_templates.index(active_cfg.digital_twin_template) if (active_cfg and active_cfg.digital_twin_template in twin_templates) else 0
                f_twin_tmpl = st.selectbox("3D Digital Twin CAD Template", twin_templates, index=twin_idx)
            with t_col2:
                sim_options = ["nasa_turbofan_brayton", "rr_trent_3spool_research", "pw_gtf_reduction_gear_model", "None (Catalogue Only)"]
                f_sim_model = st.selectbox("Compatible Simulation Model", sim_options, index=0)

            btn_label = "➕ Create Engine Configuration" if is_new else "💾 Update Engine Configuration"
            submitted = st.form_submit_button(btn_label, type="primary")

            if submitted:
                new_cfg = EngineConfiguration(
                    config_id=f_cfg_id.strip(),
                    manufacturer=f_manuf,
                    engine_family=f_family.strip(),
                    engine_model=f_model.strip(),
                    engine_type=f_type,
                    shaft_arrangement=f_shaft,
                    number_of_shafts=int(f_num_shafts),
                    fan_diameter_m=PropertyWithProvenance("Fan Diameter", f_fan_d, "m", ProvenanceSource(f_fan_src)),
                    bypass_ratio=PropertyWithProvenance("Bypass Ratio", f_bpr, "-", ProvenanceSource(f_bpr_src)),
                    overall_pressure_ratio=PropertyWithProvenance("Overall Pressure Ratio", f_opr, "-", ProvenanceSource(f_opr_src)),
                    takeoff_thrust_kn=PropertyWithProvenance("Takeoff Thrust", f_thrust, "kN", ProvenanceSource(f_thrust_src)),
                    cruise_sfc_kg_kn_hr=PropertyWithProvenance("Cruise SFC", f_sfc, "kg/(kN*hr)", ProvenanceSource(f_sfc_src)),
                    dry_weight_kg=PropertyWithProvenance("Dry Weight", f_weight, "kg", ProvenanceSource(f_weight_src)),
                    digital_twin_template=f_twin_tmpl,
                    supported_simulation_model_id=f_sim_model if "None" not in f_sim_model else None,
                    is_user_defined=True
                )
                
                # Scientific validation check
                status, errs, warns = EngineValidator.validate_configuration(new_cfg)
                new_cfg.validation_status = status.value
                new_cfg.validation_notes = warns + errs

                if repo.save_engine_configuration(new_cfg):
                    st.success(f"Configuration '{new_cfg.config_id}' saved successfully with status: {new_cfg.validation_status}!")
                    st.rerun()
                else:
                    st.error("Failed to save configuration.")

        # Package Export Option
        if active_cfg:
            st.markdown("---")
            if st.button(f"📦 Export '{active_cfg.config_id}' as Reusable Package"):
                pkg_file = repo.export_package(active_cfg.config_id)
                if pkg_file:
                    st.success(f"Package exported to: `{pkg_file}`")
                    with open(pkg_file, "r") as pf:
                        st.download_button(
                            "⬇️ Download Engine Package JSON",
                            data=pf.read(),
                            file_name=pathlib.Path(pkg_file).name,
                            mime="application/json"
                        )

    # -------------------------------------------------------------------------
    # TAB 3: MODEL & DATASET IMPORTER
    # -------------------------------------------------------------------------
    with tab_imp:
        st.markdown("### Multi-Engine Model & Telemetry Importer")
        st.caption("Import external MATLAB (.m), Simulink (.slx), FMU, Python modules, MAT, or CSV telemetry files.")

        imp_col1, imp_col2 = st.columns(2)

        with imp_col1:
            st.markdown("#### 1. Import Model File")
            model_file_path = st.text_input(
                "Local File Path to Model / Telemetry Dataset",
                placeholder="e.g. C:/Models/turbofan_trent.m or C:/Data/engine_telemetry.csv"
            )
            assoc_engine = st.selectbox(
                "Associate with Engine Configuration",
                [c.config_id for c in configs]
            )

            if st.button("Inspect & Validate Model", type="primary"):
                if model_file_path:
                    ok, imp_model, msgs = ModelImporter.inspect_and_validate(model_file_path, assoc_engine)
                    if ok and imp_model:
                        repo.save_imported_model(imp_model)
                        st.success(f"Model '{imp_model.model_name}' successfully validated and registered!")
                        st.json(imp_model.to_dict())
                    else:
                        st.error(f"Validation failed: {'; '.join(msgs)}")
                else:
                    st.warning("Please provide a valid file path.")

            st.markdown("#### 2. Import Engine Package JSON")
            uploaded_pkg = st.file_uploader("Upload Reusable Engine Package (.json)", type=["json"])
            if uploaded_pkg is not None:
                try:
                    pkg_dict = json.load(uploaded_pkg)
                    temp_pkg = pathlib.Path("engine_packages") / uploaded_pkg.name
                    with open(temp_pkg, "w") as tf:
                        json.dump(pkg_dict, tf)
                    imported_cfg = repo.import_package(str(temp_pkg))
                    if imported_cfg:
                        st.success(f"Successfully imported configuration '{imported_cfg.config_id}' from package!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Import error: {e}")

        with imp_col2:
            st.markdown("#### Registered External Models & Telemetry Streams")
            models_list = repo.get_imported_models()
            if models_list:
                for m in models_list:
                    st.markdown(f"""
                    <div style="background: #111827; border: 1px solid #1F2937; border-left: 4px solid #8B5CF6; padding: 12px; border-radius: 6px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between;">
                            <b>{m.model_name}</b>
                            <span style="color: #A78BFA; font-size: 0.8rem;">{m.model_type}</span>
                        </div>
                        <div style="color: #9CA3AF; font-size: 0.75rem; margin-top: 4px;">Engine: <code>{m.associated_engine_config_id or 'Unassigned'}</code></div>
                        <div style="color: #6B7280; font-size: 0.75rem;">Inputs: {len(m.input_variables)} | Outputs: {len(m.output_variables)} | Freq: {m.sampling_frequency_hz} Hz</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No external models registered yet. Native MATLAB Brayton cycle model is active.")

    # -------------------------------------------------------------------------
    # TAB 4: MULTI-ENGINE COMPARISON LAB
    # -------------------------------------------------------------------------
    with tab_comp:
        st.markdown("### Multi-Engine Architectural & Performance Comparison Laboratory")
        st.caption("Compare physical dimensions, pressure ratios, bypass ratios, and sensor telemetry across distinct engine types.")

        all_cfg_ids = [c.config_id for c in configs]
        selected_for_comp = st.multiselect(
            "Select 2 or More Engine Configurations to Compare",
            all_cfg_ids,
            default=[all_cfg_ids[0], all_cfg_ids[1]] if len(all_cfg_ids) >= 2 else all_cfg_ids
        )

        if len(selected_for_comp) >= 2:
            cfgs_to_compare = [repo.get_engine_configuration(cid) for cid in selected_for_comp if repo.get_engine_configuration(cid)]
            comp_res = EngineComparator.compare_configurations(cfgs_to_compare)

            # Engineering Narrative Callout
            st.markdown(f"""
            <div style="background: #111827; border: 1px solid #1F2937; border-left: 4px solid #00B8D9; padding: 14px 18px; border-radius: 6px; margin: 12px 0;">
                <div style="color: #F8FAFC; font-weight: bold; margin-bottom: 4px;">🔬 Aerospace Comparative Assessment</div>
                <div style="color: #94A3B8; font-size: 0.9rem; line-height: 1.4;">{comp_res['engineering_narrative']}</div>
            </div>
            """, unsafe_allow_html=True)

            # Comparison Table
            matrix_df = pd.DataFrame(comp_res["comparison_matrix"])
            st.dataframe(
                matrix_df[[
                    "engine_label", "architecture", "spools_count",
                    "fan_diameter_m", "bypass_ratio", "overall_pressure_ratio",
                    "takeoff_thrust_kn", "dry_weight_kg", "thrust_to_weight_ratio",
                    "cruise_sfc", "validation_status"
                ]].rename(columns={
                    "engine_label": "Engine",
                    "architecture": "Architecture",
                    "spools_count": "Shafts",
                    "fan_diameter_m": "Fan Ø (m)",
                    "bypass_ratio": "BPR",
                    "overall_pressure_ratio": "OPR",
                    "takeoff_thrust_kn": "Thrust (kN)",
                    "dry_weight_kg": "Weight (kg)",
                    "thrust_to_weight_ratio": "T/W",
                    "cruise_sfc": "SFC",
                    "validation_status": "Status"
                }),
                use_container_width=True
            )

            # Visual Comparison Charts
            ch_col1, ch_col2 = st.columns(2)
            with ch_col1:
                fig_thrust = px.bar(
                    matrix_df,
                    x="engine_model",
                    y="takeoff_thrust_kn",
                    color="manufacturer",
                    title="Takeoff Thrust Comparison (kN)",
                    template="plotly_dark",
                    text_auto=".1f"
                )
                fig_thrust.update_layout(height=320, margin=dict(l=30, r=20, t=40, b=30))
                st.plotly_chart(fig_thrust, use_container_width=True)

            with ch_col2:
                fig_bpr = px.scatter(
                    matrix_df,
                    x="overall_pressure_ratio",
                    y="bypass_ratio",
                    size="fan_diameter_m",
                    color="architecture",
                    text="engine_model",
                    title="Bypass Ratio vs Overall Pressure Ratio (OPR)",
                    template="plotly_dark"
                )
                fig_bpr.update_layout(height=320, margin=dict(l=30, r=20, t=40, b=30))
                st.plotly_chart(fig_bpr, use_container_width=True)

            # Sensor Overlap Matrix
            st.markdown("#### Sensor Telemetry Overlap Matrix")
            overlap = comp_res["sensor_overlap"]
            st.markdown(f"Found **{overlap['common_count']}** common channels across selected engines: `{', '.join(overlap['common_sensors'])}`")
        else:
            st.info("Select at least 2 engines from the dropdown above to view side-by-side comparative analytics.")

    # -------------------------------------------------------------------------
    # TAB 5: AI MODEL COMPATIBILITY
    # -------------------------------------------------------------------------
    with tab_ai:
        st.markdown("### Cross-Engine AI Prognostic Compatibility Matrix")
        st.caption("Verifies whether deep learning RUL models trained on NASA datasets can safely generalize to other engine architectures.")

        ai_c1, ai_c2 = st.columns(2)
        with ai_c1:
            target_engine_id = st.selectbox("Select Target Engine Configuration", all_cfg_ids, index=0)
            target_engine = repo.get_engine_configuration(target_engine_id)

            tested_models = [
                "pinball_lstm_q35 (Best)",
                "lstm",
                "transformer (Phase 2)",
                "random_forest",
                "linear_regression"
            ]
            eval_model_name = st.selectbox("Select Active AI Prognostic Model", tested_models)

        with ai_c2:
            if target_engine:
                compat_result = EngineValidator.check_ai_model_engine_compatibility(target_engine, eval_model_name)
                
                badge_bg = "rgba(33, 199, 122, 0.2)" if compat_result["is_compatible"] else "rgba(255, 82, 82, 0.2)"
                badge_color = "#21C77A" if compat_result["is_compatible"] else "#FF5252"
                
                st.markdown(f"""
                <div style="background: #111827; border: 1px solid #1F2937; border-left: 5px solid {badge_color}; padding: 16px; border-radius: 6px;">
                    <div style="font-size: 1.1rem; font-weight: bold; color: {badge_color};">
                        Status: {compat_result['status']}
                    </div>
                    <div style="color: #9CA3AF; font-size: 0.85rem; margin-top: 6px;">
                        Risk Level: <b>{compat_result['risk_level']}</b> | Confidence Multiplier: <b>{compat_result['confidence_multiplier'] * 100:.0f}%</b>
                    </div>
                    <div style="color: #E2E8F0; font-size: 0.85rem; margin-top: 10px; line-height: 1.4;">
                        {compat_result['message']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Multi-Engine Compatibility Matrix Across Fleet")
        fleet_matrix_data = []
        for c in configs:
            chk = EngineValidator.check_ai_model_engine_compatibility(c, eval_model_name)
            fleet_matrix_data.append({
                "Engine": f"{c.manufacturer} {c.engine_model}",
                "Architecture": c.shaft_arrangement,
                "Model Compatibility": "✅ Fully Validated" if chk["is_compatible"] else "⚠️ Cross-Engine Warning",
                "Generalization Risk": chk["risk_level"],
                "Confidence": f"{chk['confidence_multiplier']*100:.0f}%",
                "Adaptation Required": "No" if chk["is_compatible"] else "Domain Adaptation Required"
            })
        st.table(pd.DataFrame(fleet_matrix_data))

    # -------------------------------------------------------------------------
    # TAB 6: MULTI-ENGINE 3D DIGITAL TWIN
    # -------------------------------------------------------------------------
    with tab_twin:
        st.markdown("### Architecture-Specific 3D Turbofan Digital Twin")
        st.caption("Custom mechanical CAD representations: Rolls-Royce 3-spool, Pratt & Whitney Geared GTF, and Standard Twin-Spool.")

        twin_cfg_id = st.selectbox("Select Engine for 3D Twin Visualization", all_cfg_ids, index=0)
        twin_engine = repo.get_engine_configuration(twin_cfg_id)

        if twin_engine:
            twin_html = render_multi_engine_3d_twin_html(
                engine_cfg=twin_engine,
                height_px=620
            )
            st.components.v1.html(twin_html, height=640)
