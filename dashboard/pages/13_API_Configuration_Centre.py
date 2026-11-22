"""Streamlit Multipage Entrypoint — API & Credential Configuration Centre."""

import streamlit as st
from src.ui.api_configuration_centre import render_api_configuration_centre

st.set_page_page_config = getattr(st, "set_page_config", None)
if st.set_page_page_config:
    try:
        st.set_page_config(
            page_title="API Configuration Centre | THRUSTLAB",
            page_icon="🔐",
            layout="wide"
        )
    except Exception:
        pass

render_api_configuration_centre()
