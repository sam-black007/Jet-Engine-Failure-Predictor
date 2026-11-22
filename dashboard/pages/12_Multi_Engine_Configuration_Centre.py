"""JETWISE AI — Multi-Manufacturer Engine Configuration Centre (Multipage Dashboard)."""

import streamlit as st
from src.ui.multi_engine_centre import render_multi_engine_centre

st.set_page_config(
    page_title="JETWISE AI - Multi-Engine Configuration Centre",
    page_icon="🌐",
    layout="wide"
)

# Dark theme styling
st.markdown("""
<style>
    .stApp { background-color: #0B0F14; color: #E2E8F0; }
</style>
""", unsafe_allow_html=True)

render_multi_engine_centre()
