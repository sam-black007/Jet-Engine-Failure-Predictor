"""THRUSTLAB — Global Enterprise Configuration Centre (Part 3 Multipage Dashboard)."""

import streamlit as st
from src.ui.configuration_centre import render_global_configuration_centre

st.set_page_config(page_title="THRUSTLAB - Global Configuration Centre", page_icon="⚙️", layout="wide")

# Theme styling
st.markdown("""
<style>
    .stApp { background-color: #0B0F14; color: #E2E8F0; }
</style>
""", unsafe_allow_html=True)

st.title("⚙️ Global Enterprise Configuration Centre")
render_global_configuration_centre()
