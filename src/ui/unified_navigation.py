"""JETWISE AI — Unified Navigation Architecture & Command Palette.

Implements Section 2.2, Section 3, Section 5.3, and Section 5.4 requirements:
5-Hub Navigation (HOME, SIMULATE, MONITOR, ANALYZE, CONFIGURE),
Simple Mode vs Advanced Engineering Mode switcher,
Global Command Search Palette, and Persistent Engineering Context.
"""

from typing import Dict, Any, List, Optional, Callable
import streamlit as st

from src.ui.contextual_help import render_glossary_view


class NavigationHub:
    HOME = "HOME"
    SIMULATE = "SIMULATE"
    MONITOR = "MONITOR"
    ANALYZE = "ANALYZE"
    CONFIGURE = "CONFIGURE"


# Global Search Catalog
SEARCHABLE_COMMANDS: List[Dict[str, str]] = [
    {"title": "Start Simulation", "hub": NavigationHub.SIMULATE, "desc": "Open Engineering Command Centre to run a prompt or preset."},
    {"title": "Run Healthy Baseline", "hub": NavigationHub.SIMULATE, "desc": "Execute nominal baseline simulation."},
    {"title": "Inspect 3D Digital Twin", "hub": NavigationHub.MONITOR, "desc": "Open interactive 10-subassembly 3D turbofan twin."},
    {"title": "Fleet Health Overview", "hub": NavigationHub.MONITOR, "desc": "View fleet operational cycles and degradation curves."},
    {"title": "Sensor Fault & FDIR", "hub": NavigationHub.MONITOR, "desc": "Inspect sensor fault detection, isolation & recovery."},
    {"title": "Model Comparison Benchmark", "hub": NavigationHub.ANALYZE, "desc": "Compare Baseline, LSTM, Pinball, and Transformer metrics."},
    {"title": "RUL Uncertainty & Conformal Bounds", "hub": NavigationHub.ANALYZE, "desc": "View 90% conformal intervals and quantile predictions."},
    {"title": "Out-of-Distribution (OOD) Safety", "hub": NavigationHub.ANALYZE, "desc": "Check if engine is operating outside trained AI envelope."},
    {"title": "Download Engineering PDF Report", "hub": NavigationHub.ANALYZE, "desc": "Generate printable aerospace engineering dossier."},
    {"title": "MATLAB & Simulink Setup", "hub": NavigationHub.CONFIGURE, "desc": "Verify MathWorks installation, release, and licences."},
    {"title": "API Configuration (Gemini / HF / OpenAI)", "hub": NavigationHub.CONFIGURE, "desc": "Manage secure API keys and test connections."},
    {"title": "Multi-Engine Manufacturer Library", "hub": NavigationHub.CONFIGURE, "desc": "Explore Rolls-Royce, GE, CFM, Pratt & Whitney engines."},
    {"title": "Aerospace & AI Glossary", "hub": "GLOSSARY", "desc": "Search technical definitions and aerospace units."}
]


def init_navigation_state():
    """Ensure all required navigation and context keys exist in st.session_state."""
    if "active_hub" not in st.session_state:
        st.session_state["active_hub"] = NavigationHub.HOME
    if "app_presentation_mode" not in st.session_state:
        st.session_state["app_presentation_mode"] = "simple"
    if "global_search_query" not in st.session_state:
        st.session_state["global_search_query"] = ""


def render_unified_header_and_navigation() -> str:
    """Render the simplified top navigation bar, presentation mode switch, and command palette."""
    init_navigation_state()

    # Top Navigation Row
    top_col1, top_col2, top_col3 = st.columns([1.8, 1.2, 1.0])

    with top_col1:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; padding: 4px 0;">
            <span style="font-size: 26px;">✈️</span>
            <div>
                <b style="color: #F8FAFC; font-size: 18px; letter-spacing: 0.5px;">JETWISE AI</b>
                <span style="color: #00B8D9; font-size: 11px; margin-left: 6px; font-weight: 700; border: 1px solid #00B8D9; padding: 1px 6px; border-radius: 3px;">PHASE 17 OS</span>
                <div style="color: #94A3B8; font-size: 11px;">Autonomous Flight Prognostics & Multi-Engine Digital Twin</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with top_col2:
        # Presentation Mode Selector
        current_mode = st.session_state["app_presentation_mode"]
        mode_choice = st.radio(
            "Presentation Mode",
            ["🟢 Simple Mode (Beginner)", "🔬 Advanced Engineering View"],
            index=0 if current_mode == "simple" else 1,
            horizontal=True,
            label_visibility="collapsed",
            key="radio_presentation_mode"
        )
        new_mode = "simple" if "Simple" in mode_choice else "advanced"
        st.session_state["app_presentation_mode"] = new_mode

    with top_col3:
        # Search & Help Shortcuts
        search_kw = st.text_input("🔍 Quick Command / Search...", "", placeholder="Search action, tool, RUL...", key="unified_global_search")
        if search_kw:
            st.session_state["global_search_query"] = search_kw

    # If Search Query active, render instant jump results
    if st.session_state.get("global_search_query"):
        kw = st.session_state["global_search_query"].lower()
        matches = [m for m in SEARCHABLE_COMMANDS if kw in m["title"].lower() or kw in m["desc"].lower()]
        if matches:
            st.markdown(f"**Search Results ({len(matches)} matches):**")
            res_cols = st.columns(min(3, len(matches)))
            for idx, match in enumerate(matches[:3]):
                with res_cols[idx]:
                    if st.button(f"👉 {match['title']}", key=f"s_match_{idx}", use_container_width=True):
                        st.session_state["active_hub"] = match["hub"]
                        st.session_state["global_search_query"] = ""
                        st.rerun()

    # Primary 5-Hub Button Bar
    st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
    hub_cols = st.columns(5)
    hubs = [
        (NavigationHub.HOME, "🏠 1. HOME", "Mission Control"),
        (NavigationHub.SIMULATE, "⚡ 2. SIMULATE", "Command Centre"),
        (NavigationHub.MONITOR, "🛰️ 3. MONITOR", "Engine Intelligence"),
        (NavigationHub.ANALYZE, "🔬 4. ANALYZE", "Engineering Analysis"),
        (NavigationHub.CONFIGURE, "⚙️ 5. CONFIGURE", "System Setup")
    ]

    for idx, (hub_key, hub_title, hub_sub) in enumerate(hubs):
        with hub_cols[idx]:
            is_active = (st.session_state["active_hub"] == hub_key)
            btn_type = "primary" if is_active else "secondary"
            if st.button(f"{hub_title}\n{hub_sub}", key=f"nav_btn_{hub_key}", type=btn_type, use_container_width=True):
                st.session_state["active_hub"] = hub_key
                st.rerun()

    # Persistent Engineering Context Bar
    scen = st.session_state.get("active_scenario")
    res = st.session_state.get("active_result")

    eng_id = scen.engine_id if scen else 1
    ds_id = scen.dataset_id if scen else "FD001"
    sim_backend = getattr(scen.simulation_mode, "value", "BRAYTON_PHYSICS") if scen else "NASA Brayton Cycle"
    rul_str = f"{res.modified_rul:.1f} Cycles" if res else "112.5 Cycles (Baseline)"

    st.markdown(f"""
    <div style="background: #0B0F14; border: 1px solid #1E293B; border-radius: 4px; padding: 6px 14px; margin: 10px 0 16px 0; display: flex; justify-content: space-between; align-items: center; font-size: 11px;">
        <div>
            <span style="color: #64748B;">Fleet Context:</span>
            <b style="color: #00B8D9; margin-left: 4px;">Unit #{eng_id}</b>
            <span style="color: #475569; margin: 0 8px;">|</span>
            <span style="color: #64748B;">Dataset:</span>
            <b style="color: #F8FAFC; margin-left: 4px;">{ds_id}</b>
            <span style="color: #475569; margin: 0 8px;">|</span>
            <span style="color: #64748B;">Backend:</span>
            <b style="color: #10B981; margin-left: 4px;">{sim_backend}</b>
        </div>
        <div>
            <span style="color: #64748B;">Current RUL:</span>
            <b style="color: #F8FAFC; margin-left: 4px;">{rul_str}</b>
            <span style="color: #475569; margin: 0 8px;">|</span>
            <span style="color: #64748B;">View:</span>
            <b style="color: #E2E8F0; margin-left: 4px;">{'Simple (Beginner)' if st.session_state['app_presentation_mode'] == 'simple' else 'Advanced Engineering'}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    return st.session_state["active_hub"]
