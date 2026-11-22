"""JETWISE AI — System Setup (Configure Hub).

Implements Section 2.2 Hub 5 and Section 14/15 requirements:
Consolidates MATLAB & Simulink Setup, API Provider Configuration,
Engine Configuration, Dataset Configuration, and User Preferences.
"""

from typing import Dict, Any, List, Optional
import streamlit as st

from src.ui.api_configuration_centre import render_api_configuration_centre
from src.ui.matlab_centre import render_matlab_centre
from src.ui.multi_engine_centre import render_multi_engine_centre
from src.ui.configuration_centre import render_global_configuration_centre


def render_configure_system_hub():
    """Render the consolidated Configure Hub."""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #101720 0%, #151D27 100%); border-bottom: 2px solid #00B8D9; padding: 14px 20px; border-radius: 6px; margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="color: #F8FAFC; margin: 0; font-size: 20px;">⚙️ JETWISE SYSTEM & INTEGRATION CONFIGURATION</h3>
                <span style="color: #94A3B8; font-size: 12px;">Secure API Key Management, MathWorks Co-Simulation Setup, Multi-Engine Library & Global Profiles</span>
            </div>
            <div>
                <span style="background: rgba(0, 184, 217, 0.15); color: #00B8D9; border: 1px solid #00B8D9; padding: 4px 10px; border-radius: 4px; font-weight: 700; font-size: 11px;">
                    ADMIN DESK
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    config_tabs = st.tabs([
        "🔐 API Configuration Centre",
        "🎛️ MATLAB & Simulink Environment Setup",
        "🌐 Multi-Manufacturer Engine Library",
        "⚙️ Global Parameters & Preferences"
    ])

    with config_tabs[0]:
        render_api_configuration_centre()

    with config_tabs[1]:
        render_matlab_centre()

    with config_tabs[2]:
        render_multi_engine_centre()

    with config_tabs[3]:
        render_global_configuration_centre()
