"""JETWISE AI — Aerospace Intelligence & API Configuration Centre Workspace.

Comprehensive interactive dashboard interface for managing external AI credentials (Google Gemini,
Hugging Face, OpenAI) and local MATLAB/Simulink engine connectivity with zero-leakage security safeguards.
"""

import streamlit as st
import datetime
from src.security.credential_manager import (
    CredentialManager,
    ProviderType,
    ProviderStatus,
    get_credential_manager
)


def render_api_configuration_centre():
    """Render the full API & Integration Configuration Centre."""
    cred_mgr = get_credential_manager()

    # Header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #0B132B 0%, #1C2541 100%); padding: 18px 24px; border-radius: 8px; border-left: 5px solid #10B981; margin-bottom: 20px;">
        <h2 style="margin: 0; color: #F8FAFC; display: flex; align-items: center; gap: 10px;">
            🔐 Aerospace Intelligence & API Configuration Centre
        </h2>
        <p style="margin: 6px 0 0 0; color: #94A3B8; font-size: 0.95rem;">
            Centralized credential management, provider status monitoring, and connection verification with strict zero-leakage security.
        </p>
    </div>
    """, unsafe_allow_html=True)

    providers = cred_mgr.get_all_providers()
    gemini_info = providers.get(ProviderType.GEMINI.value)
    hf_info = providers.get(ProviderType.HUGGINGFACE.value)
    openai_info = providers.get(ProviderType.OPENAI.value)
    matlab_info = providers.get(ProviderType.MATLAB.value)

    # Top KPI summary cards
    connected_count = sum(1 for p in providers.values() if p.status == ProviderStatus.CONNECTED)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registered Providers", f"{len(providers)} Services")
    c2.metric("Verified Live Connections", f"{connected_count} / {len(providers)}")
    c3.metric("Local Offline Fallback", "100% OPERATIONAL")
    c4.metric("Security & Zero Leakage", "ACTIVE (Masked)")

    st.markdown("---")

    # Security & Offline Policy Notice
    st.info(
        "🛡️ **Security Policy & Air-Gapped Operation:** All credentials are processed backend-only and masked before UI rendering. "
        "JETWISE AI does **NOT** require external APIs to operate: when APIs are unconfigured or unavailable, the platform automatically "
        "falls back to internal PyTorch sequence models (Pinball LSTM, Transformer), deterministic aerospace NLU, and NASA Brayton-cycle thermodynamic physics."
    )

    # 4 Provider Columns / Cards
    st.markdown("### Integrated AI & Engineering Simulation Providers")

    tab_gemini, tab_hf, tab_openai, tab_matlab = st.tabs([
        "🤖 Google Gemini",
        "🤗 Hugging Face Hub",
        "⚡ OpenAI (Optional)",
        "🎛️ MATLAB / Simulink Engine"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: GOOGLE GEMINI
    # -------------------------------------------------------------------------
    with tab_gemini:
        st.markdown("#### Google Gemini AI Integration")
        st.caption("Powers conversational aerospace engineering dialogue, scenario translation, and diagnostic report generation.")

        g_col1, g_col2 = st.columns([2, 1])
        with g_col1:
            st.markdown(f"**Connection Status:** `{gemini_info.status.value}`")
            st.markdown(f"**Masked Key:** `{gemini_info.masked_key}`")
            st.markdown(f"**Active Model:** `{gemini_info.model_id}`")
            if gemini_info.last_tested:
                st.caption(f"Last Tested: {gemini_info.last_tested} ({gemini_info.latency_ms} ms)")
            if gemini_info.error_message:
                st.error(f"Last Error: {gemini_info.error_message}")

            # Runtime key update input
            with st.expander("🔑 Update / Set Gemini API Key (Runtime In-Memory)"):
                new_key = st.text_input(
                    "Enter Replacement GEMINI_API_KEY",
                    type="password",
                    placeholder="AIzaSy...",
                    help="Keys are stored in memory during session and masked. To persist, add GEMINI_API_KEY to your local .env file."
                )
                selected_model = st.selectbox("Select Gemini Model", gemini_info.available_models, index=0)
                if st.button("Apply & Save Credential", key="btn_save_gemini"):
                    if new_key:
                        cred_mgr.update_runtime_credential(ProviderType.GEMINI.value, new_key, model_id=selected_model)
                        st.success("Updated runtime Gemini credential!")
                        st.rerun()

            if st.button("⚡ Test Gemini Live Connection", key="btn_test_gemini", use_container_width=True):
                with st.spinner("Testing Google Gemini connection..."):
                    res = cred_mgr.test_gemini_connection(model=gemini_info.model_id)
                    if res["success"]:
                        st.success(f"✅ {res['message']}")
                    else:
                        st.error(f"❌ {res['message']}")
                    st.rerun()

        with g_col2:
            st.markdown("""
            <div style="background: #111827; border: 1px solid #1F2937; border-left: 3px solid #38BDF8; padding: 14px; border-radius: 6px;">
                <b style="color: #38BDF8;">📘 Setup Instructions:</b>
                <ol style="color: #CBD5E1; font-size: 0.85rem; padding-left: 18px; margin-top: 8px;">
                    <li>Obtain a replacement key at <a href="https://aistudio.google.com/" target="_blank" style="color: #38BDF8;">Google AI Studio</a>.</li>
                    <li>Add <code>GEMINI_API_KEY=your_key</code> to your local <code>.env</code> file.</li>
                    <li>Or enter the replacement key in the runtime box on the left.</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 2: HUGGING FACE
    # -------------------------------------------------------------------------
    with tab_hf:
        st.markdown("#### Hugging Face Hub & Model Repository")
        st.caption("Access public and gated aerospace predictive maintenance models, datasets, and open-source inference.")

        hf_col1, hf_col2 = st.columns([2, 1])
        with hf_col1:
            st.markdown(f"**Connection Status:** `{hf_info.status.value}`")
            st.markdown(f"**Masked Token:** `{hf_info.masked_key}`")
            if hf_info.auth_identity:
                st.markdown(f"**Authenticated Identity:** `{hf_info.auth_identity}`")
            if hf_info.last_tested:
                st.caption(f"Last Tested: {hf_info.last_tested} ({hf_info.latency_ms} ms)")
            if hf_info.error_message:
                st.error(f"Last Error: {hf_info.error_message}")

            with st.expander("🔑 Update / Set Hugging Face Token (Runtime In-Memory)"):
                new_hf = st.text_input(
                    "Enter Replacement HF_TOKEN",
                    type="password",
                    placeholder="hf_...",
                    help="Create token with Read permissions at https://huggingface.co/settings/tokens"
                )
                if st.button("Apply HF Token", key="btn_save_hf"):
                    if new_hf:
                        cred_mgr.update_runtime_credential(ProviderType.HUGGINGFACE.value, new_hf)
                        st.success("Updated runtime Hugging Face token!")
                        st.rerun()

            if st.button("⚡ Test Hugging Face Connection", key="btn_test_hf", use_container_width=True):
                with st.spinner("Testing Hugging Face Hub connection..."):
                    res = cred_mgr.test_huggingface_connection()
                    if res["success"]:
                        st.success(f"✅ {res['message']}")
                    else:
                        st.error(f"❌ {res['message']}")
                    st.rerun()

        with hf_col2:
            st.markdown("""
            <div style="background: #111827; border: 1px solid #1F2937; border-left: 3px solid #F59E0B; padding: 14px; border-radius: 6px;">
                <b style="color: #F59E0B;">📘 Setup Instructions:</b>
                <ol style="color: #CBD5E1; font-size: 0.85rem; padding-left: 18px; margin-top: 8px;">
                    <li>Generate a fine-grained or read token at <a href="https://huggingface.co/settings/tokens" target="_blank" style="color: #F59E0B;">Hugging Face Tokens</a>.</li>
                    <li>Add <code>HF_TOKEN=your_token</code> to your local <code>.env</code> file.</li>
                    <li>Local PyTorch inference functions independently even if HF token is omitted.</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 3: OPENAI (OPTIONAL)
    # -------------------------------------------------------------------------
    with tab_openai:
        st.markdown("#### OpenAI API Integration (Optional)")
        st.caption("Configurable alternative LLM provider for engineering copilot comparisons and reasoning benchmarks.")

        o_col1, o_col2 = st.columns([2, 1])
        with o_col1:
            st.markdown(f"**Connection Status:** `{openai_info.status.value}`")
            st.markdown(f"**Masked Key:** `{openai_info.masked_key}`")
            st.markdown(f"**Active Model:** `{openai_info.model_id}`")
            if openai_info.last_tested:
                st.caption(f"Last Tested: {openai_info.last_tested} ({openai_info.latency_ms} ms)")
            if openai_info.error_message:
                st.error(f"Last Error: {openai_info.error_message}")

            with st.expander("🔑 Update / Set OpenAI API Key (Runtime In-Memory)"):
                new_o_key = st.text_input(
                    "Enter OPENAI_API_KEY",
                    type="password",
                    placeholder="sk-proj-...",
                    help="Optional provider. Enter key if you wish to use OpenAI models."
                )
                selected_o_model = st.selectbox("Select OpenAI Model", openai_info.available_models, index=0)
                if st.button("Apply OpenAI Key", key="btn_save_openai"):
                    if new_o_key:
                        cred_mgr.update_runtime_credential(ProviderType.OPENAI.value, new_o_key, model_id=selected_o_model)
                        st.success("Updated runtime OpenAI key!")
                        st.rerun()

            if st.button("⚡ Test OpenAI Live Connection", key="btn_test_openai", use_container_width=True):
                with st.spinner("Testing OpenAI connection..."):
                    res = cred_mgr.test_openai_connection()
                    if res["success"]:
                        st.success(f"✅ {res['message']}")
                    else:
                        st.error(f"❌ {res['message']}")
                    st.rerun()

        with o_col2:
            st.markdown("""
            <div style="background: #111827; border: 1px solid #1F2937; border-left: 3px solid #10B981; padding: 14px; border-radius: 6px;">
                <b style="color: #10B981;">📘 Setup Instructions:</b>
                <ol style="color: #CBD5E1; font-size: 0.85rem; padding-left: 18px; margin-top: 8px;">
                    <li>Obtain an API key at <a href="https://platform.openai.com/api-keys" target="_blank" style="color: #10B981;">OpenAI Platform</a>.</li>
                    <li>Add <code>OPENAI_API_KEY=your_key</code> to your local <code>.env</code> file.</li>
                    <li>OpenAI integration is optional; Gemini or Local NLU can be used as defaults.</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 4: MATLAB & SIMULINK LOCAL ENGINE
    # -------------------------------------------------------------------------
    with tab_matlab:
        st.markdown("#### MATLAB & Simulink Local Engine Connection")
        st.caption("Local desktop simulation engine using MathWorks MATLAB Student / Campus license and MATLAB Engine API for Python.")

        m_col1, m_col2 = st.columns([2, 1])
        with m_col1:
            st.markdown(f"**Engine Status:** `{matlab_info.status.value}`")
            st.markdown(f"**Identified Release:** `{matlab_info.model_id}`")
            st.markdown(f"**Executable Location:** `{matlab_info.masked_key}`")
            if matlab_info.auth_identity:
                st.markdown(f"**License Authentication:** `{matlab_info.auth_identity}`")

            if st.button("⚡ Test Local MATLAB Engine API Connection", key="btn_test_matlab", use_container_width=True):
                with st.spinner("Auditing MATLAB environment & Engine API..."):
                    res = cred_mgr.test_matlab_connection()
                    if res["success"]:
                        st.success(f"✅ {res['message']}")
                    else:
                        st.info(f"ℹ️ {res['message']}")
                    st.rerun()

        with m_col2:
            st.markdown("""
            <div style="background: #111827; border: 1px solid #1F2937; border-left: 3px solid #EC4899; padding: 14px; border-radius: 6px;">
                <b style="color: #EC4899;">📘 Local Activation Guidelines:</b>
                <p style="color: #CBD5E1; font-size: 0.82rem; margin-top: 6px; line-height: 1.4;">
                    MATLAB runs locally via your licensed MathWorks desktop installation. <b>No cloud API key is required.</b>
                    Python 3.11 communicates directly with MATLAB through the official <code>matlab.engine</code> API.
                </p>
            </div>
            """, unsafe_allow_html=True)
