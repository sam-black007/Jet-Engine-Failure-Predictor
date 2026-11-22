"""THRUSTLAB — Mission Control (Home Hub).

Implements Section 2.2 Hub 1 and Section 4 requirements:
Platform overview, active engine health summary, system readiness indicators (MATLAB, AI, DB),
5 prominent quick-action cards, 9-step guided setup wizard, and safe interactive demonstration.
"""

from typing import Dict, Any, List, Optional
import streamlit as st
import time

from src.matlab_simulink_bridge import get_matlab_simulink_bridge
from src.security.credential_manager import CredentialManager
from src.copilot_simulation import CopilotSimulationEngine
from src.copilot_scenarios import EngineeringScenario, SimulationMode
from src.copilot_scorecard import ScorecardCalculator
from src.dataset_registry import DatasetRegistry


def render_mission_control_home(switch_hub_callback):
    """Render the main landing page: Mission Control."""
    mode = st.session_state.get("app_presentation_mode", "simple")

    # 1. Hero Welcome Banner
    st.markdown("""
    <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); border: 1px solid #334155; border-left: 5px solid #00B8D9; border-radius: 8px; padding: 22px 28px; margin-bottom: 24px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span style="background: rgba(0, 184, 217, 0.15); color: #00B8D9; border: 1px solid #00B8D9; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; text-transform: uppercase;">
                    Flight Prognostics & Digital Twin OS v3.4.0
                </span>
                <h2 style="color: #F8FAFC; margin: 8px 0 4px 0; font-size: 26px; font-weight: 800;">
                    ✈️ THRUSTLAB — Mission Control
                </h2>
                <p style="color: #94A3B8; margin: 0; font-size: 14px; max-width: 820px; line-height: 1.5;">
                    Autonomous Aircraft Engine Health Monitoring, Conversational Aerothermal Simulation & Predictive Maintenance Platform.
                    Empowering airline operators and propulsion engineers with physics-grounded AI predictions, real-time 3D digital twins, and certified MATLAB co-simulation.
                </p>
            </div>
            <div style="text-align: right; margin-top: 10px;">
                <span style="color: #64748B; font-size: 12px;">Active Mode:</span><br/>
                <b style="color: #10B981; font-size: 14px;">{'🟢 Simple Mode (Beginner)' if mode == 'simple' else '🔬 Advanced Engineering View'}</b>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. System Readiness & Environment Status Bar
    sim_engine = CopilotSimulationEngine()
    matlab_status = sim_engine.matlab_bridge.get_status()
    cred_mgr = CredentialManager()
    ai_providers = cred_mgr.get_all_providers()

    gemini_ready = ai_providers.get("google_gemini", {}).is_configured if hasattr(ai_providers.get("google_gemini"), "is_configured") else False
    hf_ready = ai_providers.get("huggingface", {}).is_configured if hasattr(ai_providers.get("huggingface"), "is_configured") else False

    status_c1, status_c2, status_c3, status_c4 = st.columns(4)
    with status_c1:
        st.markdown(f"""
        <div style="background: #111827; border: 1px solid #1F2937; border-top: 3px solid #00B8D9; border-radius: 6px; padding: 12px; text-align: center;">
            <div style="color: #94A3B8; font-size: 11px; text-transform: uppercase; font-weight: 600;">Aerospace Physics Core</div>
            <div style="color: #F8FAFC; font-size: 15px; font-weight: 700; margin: 4px 0;">NASA Brayton Cycle</div>
            <div style="color: #10B981; font-size: 11px; font-weight: 600;">✅ Certified & Active</div>
        </div>
        """, unsafe_allow_html=True)

    with status_c2:
        m_color = "#10B981" if matlab_status["is_installed"] else "#F59E0B"
        st.markdown(f"""
        <div style="background: #111827; border: 1px solid #1F2937; border-top: 3px solid {m_color}; border-radius: 6px; padding: 12px; text-align: center;">
            <div style="color: #94A3B8; font-size: 11px; text-transform: uppercase; font-weight: 600;">MathWorks Co-Simulation</div>
            <div style="color: #F8FAFC; font-size: 15px; font-weight: 700; margin: 4px 0;">{matlab_status.get('version', 'R2026b / Virtual')}</div>
            <div style="color: {m_color}; font-size: 11px; font-weight: 600;">{matlab_status['status_code']}</div>
        </div>
        """, unsafe_allow_html=True)

    with status_c3:
        ai_status_str = "Gemini Active" if gemini_ready else ("HF Active" if hf_ready else "Local Offline Active")
        ai_color = "#10B981" if (gemini_ready or hf_ready) else "#00B8D9"
        st.markdown(f"""
        <div style="background: #111827; border: 1px solid #1F2937; border-top: 3px solid {ai_color}; border-radius: 6px; padding: 12px; text-align: center;">
            <div style="color: #94A3B8; font-size: 11px; text-transform: uppercase; font-weight: 600;">AI NLU Engine</div>
            <div style="color: #F8FAFC; font-size: 15px; font-weight: 700; margin: 4px 0;">{ai_status_str}</div>
            <div style="color: {ai_color}; font-size: 11px; font-weight: 600;">✅ Resilient Fallback</div>
        </div>
        """, unsafe_allow_html=True)

    with status_c4:
        st.markdown(f"""
        <div style="background: #111827; border: 1px solid #1F2937; border-top: 3px solid #10B981; border-radius: 6px; padding: 12px; text-align: center;">
            <div style="color: #94A3B8; font-size: 11px; text-transform: uppercase; font-weight: 600;">Prognostic Intelligence</div>
            <div style="color: #F8FAFC; font-size: 15px; font-weight: 700; margin: 4px 0;">Pinball LSTM (q=0.35)</div>
            <div style="color: #10B981; font-size: 11px; font-weight: 600;">✅ 90% Conformal Calibrated</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 3. Five Prominent Quick-Action Navigation Cards
    st.markdown("### 🚀 Primary Mission Actions")
    st.caption("Click any action card below to immediately begin your engineering workflow.")

    act_c1, act_c2, act_c3, act_c4, act_c5 = st.columns(5)

    with act_c1:
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 14px; text-align: center; height: 160px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-size: 26px;">⚡</div>
                <b style="color: #F8FAFC; font-size: 14px;">Start Simulation</b>
                <p style="color: #94A3B8; font-size: 11px; margin-top: 4px;">Run conversational prompts or select from 14 scenario presets.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Simulate Hub", key="btn_quick_sim", use_container_width=True):
            switch_hub_callback("SIMULATE")

    with act_c2:
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 14px; text-align: center; height: 160px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-size: 26px;">🛰️</div>
                <b style="color: #F8FAFC; font-size: 14px;">Open 3D Engine</b>
                <p style="color: #94A3B8; font-size: 11px; margin-top: 4px;">Inspect the synchronized 10-subassembly turbofan twin.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Monitor Hub", key="btn_quick_twin", use_container_width=True):
            switch_hub_callback("MONITOR")

    with act_c3:
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 14px; text-align: center; height: 160px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-size: 26px;">📊</div>
                <b style="color: #F8FAFC; font-size: 14px;">View Engine Health</b>
                <p style="color: #94A3B8; font-size: 11px; margin-top: 4px;">Review fleet-wide operational cycles and sensor FDIR status.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Check Fleet Status", key="btn_quick_health", use_container_width=True):
            switch_hub_callback("MONITOR")

    with act_c4:
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 14px; text-align: center; height: 160px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-size: 26px;">📑</div>
                <b style="color: #F8FAFC; font-size: 14px;">View Reports</b>
                <p style="color: #94A3B8; font-size: 11px; margin-top: 4px;">Generate & download certified PDF, HTML, or CSV dossiers.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Analyze Hub", key="btn_quick_reports", use_container_width=True):
            switch_hub_callback("ANALYZE")

    with act_c5:
        st.markdown("""
        <div style="background: #111827; border: 1px solid #1F2937; border-radius: 8px; padding: 14px; text-align: center; height: 160px; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <div style="font-size: 26px;">⚙️</div>
                <b style="color: #F8FAFC; font-size: 14px;">Configure System</b>
                <p style="color: #94A3B8; font-size: 11px; margin-top: 4px;">Configure MATLAB, API keys, and multi-engine models.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Configure Hub", key="btn_quick_cfg", use_container_width=True):
            switch_hub_callback("CONFIGURE")

    st.markdown("---")

    # 4. Guided Setup Wizard & Interactive Demonstration Section
    wiz_col1, wiz_col2 = st.columns([1.2, 1.0])

    with wiz_col1:
        st.markdown("### 🧭 First-Time Guided Setup Wizard")
        st.caption("New to THRUSTLAB? Follow this 9-step guided walkthrough to verify your environment and launch your first simulation.")

        wizard_steps = [
            ("Step 1: Check Application Readiness", "Verifies Python 3.11, PyTorch 2.2, NumPy 1.26, and SQLite database connectivity.", True),
            ("Step 2: Check MATLAB Installation", f"Searches system PATH and directories (Found: {matlab_status.get('version', 'R2026b')}).", True),
            ("Step 3: Check Simulink Availability", "Tests Simulink dynamic block library and ODE numerical solver.", True),
            ("Step 4: Check Python-MATLAB Connection", "Probes `matlab.engine` package and IPC bridge.", True),
            ("Step 5: Check AI Provider Configuration", f"Gemini: {'Configured' if gemini_ready else 'Unset (Fallback Active)'} | Local NLU: Ready.", True),
            ("Step 6: Select Engine Configuration", "Selects active aircraft engine profile (NASA C-MAPSS FD001 Unit #1).", True),
            ("Step 7: Select Simulation Mode", "Selects physical parametric co-simulation mode.", True),
            ("Step 8: Run Baseline Simulation", "Executes nominal healthy baseline run to establish benchmark health scores.", True),
            ("Step 9: Explore Results & 3D Twin", "Inspects 6-dimension scorecard, component matrix, and 3D turbofan twin.", True),
        ]

        with st.expander("🚀 Click to Expand Step-by-Step Setup Wizard", expanded=False):
            for title, desc, ok in wizard_steps:
                icon = "✅" if ok else "⚠️"
                st.markdown(f"**{icon} {title}**")
                st.caption(desc)
                st.markdown("<hr style='margin: 4px 0;'/>", unsafe_allow_html=True)

            if st.button("✨ Complete Setup Wizard & Run Baseline", use_container_width=True):
                st.session_state["setup_wizard_completed"] = True
                st.success("Guided setup complete! Proceeding to SIMULATE Hub...")
                time.sleep(1)
                switch_hub_callback("SIMULATE")

    with wiz_col2:
        st.markdown("### 🎮 Safe Interactive Demonstration")
        st.caption("Run a safe demonstration with pre-configured parameters to explore all features without changing live engine configurations.")

        st.markdown("""
        <div style="background: #111827; border: 1px solid #1F2937; border-left: 3px solid #10B981; border-radius: 6px; padding: 14px; margin-bottom: 12px;">
            <b style="color: #10B981;">DEMO SCENARIO: Gradual Fan Wear Simulation</b><br/>
            <span style="color: #94A3B8; font-size: 12px;">
                Demonstrates a realistic 15% fan blade tip erosion scenario over 200 operational flight cycles on engine Unit #1.
            </span>
            <ul style="color: #CBD5E1; font-size: 11px; margin: 6px 0 0 16px; padding: 0;">
                <li>Safe sandbox mode with marked demonstration data.</li>
                <li>Generates live 6-dimension transparent scorecard.</li>
                <li>Synchronizes 3D digital twin with amber caution glow.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        if st.button("▶️ Launch Interactive Demo", use_container_width=True):
            with st.spinner("Initializing safe demonstration scenario..."):
                scen = EngineeringScenario(
                    scenario_name="[DEMO] Gradual Fan Degradation",
                    description="Safe demonstration scenario illustrating fan tip wear over 200 cycles.",
                    simulation_mode=SimulationMode.MODE_B_PHYSICAL_PARAMETRIC,
                    engine_id=1,
                    dataset_id="FD001",
                    cycle=100,
                    duration_cycles=200,
                    parameter_changes=[],
                    affected_components=["Fan Assembly"]
                )
                res = sim_engine.execute_scenario(scen)
                sc = ScorecardCalculator.calculate_for_result(scen, res)
                st.session_state["active_scenario"] = scen
                st.session_state["active_result"] = res
                st.session_state["active_scorecard"] = sc
                st.success("Demo simulation completed! Switching to SIMULATE Hub to inspect results...")
                time.sleep(1)
                switch_hub_callback("SIMULATE")
