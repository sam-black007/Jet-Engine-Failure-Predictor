# THRUSTLAB — API & MATLAB Environment Setup Guide

## Overview

This guide details the procedure for configuring API providers (Google Gemini, Hugging Face, OpenAI) and linking MathWorks MATLAB / Simulink (Student Version or Campus License) to the **THRUSTLAB — Propulsion Health Monitoring & Aerospace Simulation Platform**.

---

## 1. Zero-Credential Leakage Architecture

THRUSTLAB enforces strict aerospace security controls:
- **No Raw Secrets in Logs or UI**: Keys are always masked in UI representations (`AIzaSy••••••••1B7c`).
- **Backend-Only Access**: Credentials are never dispatched to client-side browsers, web storage, or unencrypted cookies.
- **Git Protection**: `.env`, `.env.*`, `credentials.json`, `*.pem`, `*.key` are guarded by `.gitignore`.
- **Autonomous Fallback**: If an external AI provider is unreachable, exhausted, or missing credentials, THRUSTLAB routes queries directly through the built-in deterministic Aerospace NLU engine.

---

## 2. API Provider Configuration

### A. Google Gemini (Recommended Primary NLU)
1. Navigate to [Google AI Studio](https://aistudio.google.com/) and generate an API key.
2. Open THRUSTLAB Dashboard and navigate to **🔐 API Configuration Centre** (`pages/13_API_Configuration_Centre.py`).
3. Select the **🤖 Google Gemini** tab.
4. Paste your API key and click **Save & Test Connection**.
5. Alternatively, add the key directly to your local `.env` file:
   ```bash
   GEMINI_API_KEY="your-new-gemini-api-key"
   GEMINI_MODEL="gemini-1.5-flash"
   ```

### B. Hugging Face Inference
1. Navigate to [Hugging Face Settings > Access Tokens](https://huggingface.co/settings/tokens) and generate a `Read` token.
2. In the **🔐 API Configuration Centre**, select **🤗 Hugging Face** and enter the token.
3. Or update `.env`:
   ```bash
   HF_TOKEN="hf_yourAccessTokenHere"
   HF_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
   ```

### C. OpenAI (Optional Secondary)
1. Obtain an API key from [OpenAI Platform](https://platform.openai.com/api-keys).
2. Enter it into the **⚡ OpenAI** tab in the configuration center or add to `.env`:
   ```bash
   OPENAI_API_KEY="sk-proj-yourOpenAIKey"
   OPENAI_MODEL="gpt-4o-mini"
   ```

---

## 3. MATLAB & Simulink Student Version Setup

THRUSTLAB distinguishes six formal MATLAB operational states:
1. `INSTALLED AND LICENSED`: MATLAB executable found, licensed, and callable.
2. `INSTALLED BUT NOT LICENSED`: Executable present, but MathWorks license file expired or missing.
3. `NOT INSTALLED`: Executable not detected on host.
4. `NOT REQUIRED`: Pure Python certified aerothermal physics engine active.
5. `NOT VERIFIED`: Executable detected, pending execution benchmark.
6. `INCOMPATIBLE`: Release/architecture mismatch with Python 3.11 64-bit.

### Linking MATLAB to THRUSTLAB
1. Locate your MATLAB installation bin directory (e.g. `C:\Program Files\MATLAB\R2024a\bin\matlab.exe` or `D:\MATLAB\R2023b\bin\matlab.exe`).
2. In the Dashboard, open **🎛️ MATLAB & Simulink Centre** -> **🖥️ Environment & Toolboxes**.
3. Under **Connect Custom MATLAB Installation Path**, paste the path and click **🔗 Link & Verify**.
4. To test direct calculation capability, expand **🧮 Mathematical Engine Calculation Benchmark** and click **▶ Run Engine Calculation Benchmark**.
5. **Virtual Emulation Toggle**: If testing without a live MathWorks license, toggle **🧪 Virtual MATLAB R2024a Bridge** to activate certified aerothermal simulation emulation.

---

## 4. Automated Verification Wizard

Run the automated verification wizard from PowerShell or bash:
```bash
python setup_integration.py
```
To enter credentials interactively without writing files manually:
```bash
python setup_integration.py --interactive
```

---

## 5. Verification Checklist

| Component | Target Status | Verification Command |
| :--- | :--- | :--- |
| **Python Environment** | Python 3.11 64-bit | `python setup_integration.py` |
| **Pytest Test Suite** | 165+ tests passing | `python -m pytest tests/` |
| **Digital Twin** | 10 subassemblies active | View in Dashboard tab `🌐 3D Digital Twin` |
| **Calculation Test** | `CALCULATION_TEST_PASSED` | Click Benchmark button in MATLAB Centre |
