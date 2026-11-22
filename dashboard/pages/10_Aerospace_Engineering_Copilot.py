"""THRUSTLAB — Aerospace Engineering Copilot (Multipage Dashboard)."""

import streamlit as st
from src.ui.copilot_workspace import render_copilot_workspace

st.set_page_config(
    page_title="THRUSTLAB - Aerospace Engineering Copilot",
    page_icon="🧑‍✈️",
    layout="wide"
)

# Dark theme styling
st.markdown("""
<style>
    .stApp { background-color: #0B0F14; color: #E2E8F0; }
</style>
""", unsafe_allow_html=True)

render_copilot_workspace()
