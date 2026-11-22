"""JETWISE AI — MATLAB & Simulink Aerospace Simulation Centre (Multipage Dashboard)."""

import streamlit as st
from src.ui.matlab_centre import render_matlab_centre

st.set_page_config(
    page_title="JETWISE AI - MATLAB & Simulink Simulation Centre",
    page_icon="🎛️",
    layout="wide"
)

# Dark theme styling
st.markdown("""
<style>
    .stApp { background-color: #0B0F14; color: #E2E8F0; }
</style>
""", unsafe_allow_html=True)

render_matlab_centre()
