# MATLAB & Simulink Setup Guide for JETWISE AI
### Connecting MATLAB Student Version to the JETWISE AI Intelligence Platform

---

## 1. Overview & Compatibility

**JETWISE AI** integrates mathematical gas path aerothermodynamic simulations executed in MathWorks **MATLAB & Simulink** with deep learning Remaining Useful Life (RUL) prognostics and an interactive 3D digital twin.

### System Compatibility Matrix
- **Operating System:** Windows 10 / 11 (64-bit)
- **Active JETWISE AI Python Runtime:** `Python 3.11.9` (64-bit)
- **Recommended MATLAB Releases:**
  - **MATLAB R2024b:** Fully certified with Python 3.11 & 3.12
  - **MATLAB R2024a:** Fully certified with Python 3.11
  - **MATLAB R2023b:** Fully certified with Python 3.11
  - *Note:* Releases R2022b and earlier do **not** support Python 3.11. If using an older release, update your MATLAB Student Version via MathWorks Account.

---

## 2. Recommended MathWorks Toolboxes

For full physical fidelity, ensure your MATLAB Student Version includes:
1. **MATLAB** (Base mathematical computation and scripting engine)
2. **Simulink** (Dynamic state-space solver and block diagram modeler)
3. **Simscape** *(Optional / Recommended)* (Physical multi-domain modeling)
4. **Simscape Fluids** *(Optional / Recommended)* (Thermal gas path and hydraulic systems)
5. **Aerospace Blockset** *(Optional)* (Standard atmospheric lapse rate and aerodynamic coefficients)

> **Note on Licensing:** A standard Student Version or Named User License is sufficient. You do **not** need a MathWorks Personal Access Token (PAT) for local desktop execution.

---

## 3. Step 1: Add MATLAB Executable to Windows System PATH

To allow JETWISE AI to auto-detect and run MATLAB in headless batch mode:

1. Locate your MATLAB installation `bin` directory. Common paths:
   - `C:\Program Files\MATLAB\R2024a\bin`
   - `C:\Program Files\MATLAB\R2023b\bin`
2. Open Windows Start Menu, type `env`, and select **Edit the system environment variables**.
3. Click **Environment Variables...**
4. Under **User variables** (or **System variables**), select `Path` and click **Edit...**
5. Click **New** and paste your path (e.g., `C:\Program Files\MATLAB\R2024a\bin`).
6. Click **OK** on all dialogs.
7. Open a new PowerShell terminal and verify:
   ```powershell
   where.exe matlab
   matlab -batch "disp('MATLAB is configured on PATH'); exit;"
   ```

---

## 4. Step 2: Install MATLAB Engine API for Python

The `matlab.engine` package enables direct in-memory communication between Python and MATLAB.

1. Open PowerShell or Command Prompt.
2. Navigate to the Python engine setup directory inside your MATLAB installation:
   ```powershell
   cd "C:\Program Files\MATLAB\R2024a\extern\engines\python"
   ```
   *(Replace `R2024a` with your installed release).*
3. Install the package into your current Python environment:
   ```powershell
   python -m pip install .
   ```
4. Verify the Python installation:
   ```powershell
   python -c "import matlab.engine; print('MATLAB Engine API for Python is successfully installed!')"
   ```

---

## 5. Step 3: Verify Turbofan Models in JETWISE AI

JETWISE AI contains authentic simulation scripts in `matlab_models/`:
- `matlab_models/scripts/run_turbofan_simulation.m`: Full Brayton cycle gas-path physics script modeling 9 subsystems (Fan, LPC, HPC, Combustor, HPT, LPT, Nozzle, LP Shaft, HP Shaft).
- `matlab_models/models/create_turbofan_slx.m`: Programmatic builder for `turbofan_engine_model.slx`.
- `matlab_models/configuration/default_parameters.json`: Nominal operating points and boundaries.

Test running the MATLAB simulation script directly from PowerShell:
```powershell
cd "c:\Users\samjo\Downloads\simulation\Jet-Engine-Failure-Predictor"
matlab -batch "addpath('matlab_models/scripts'); run_turbofan_simulation; exit;"
```
Verify that `simulation_result.json` is generated with complete steady-state and time-series telemetry.

---

## 6. Step 4: Launch the JETWISE AI Simulation Centre

Start the JETWISE AI application:
```powershell
cd "c:\Users\samjo\Downloads\simulation\Jet-Engine-Failure-Predictor"
python -m streamlit run dashboard/app.py
```

1. Open your browser to `http://localhost:8501`.
2. In the left sidebar, select **Active Workspace** &rarr; **🎛️ MATLAB & Simulink Simulation Centre** (or navigate to page `11_MATLAB_Simulink_Centre`).
3. Under the **🖥️ Environment & Toolboxes** tab, click **Refresh Environment Audit**.
4. The status badge will now show **🟢 AVAILABLE (ONLINE)** with your detected release and toolboxes.
5. Go to the **⚙️ Engine Parameters** tab, configure your operating conditions, and click **🚀 Run Turbofan Simulation** in the **🚀 Simulation Execution** tab!

---

## 7. Troubleshooting & FAQ

### Q1: The dashboard shows `OFFLINE_NOT_INSTALLED`. Will the app crash?
**No.** JETWISE AI is designed with graceful scientific degradation. When MATLAB is offline, it activates the certified internal Python Aerothermal Gas Path Physics Engine (based on NASA Brayton cycle thermodynamic relations) and clearly labels all data sources. No crashes occur and zero fake data is generated.

### Q2: `ModuleNotFoundError: No module named 'matlab'`
This means the MATLAB Engine Python package is not yet installed in your active Python environment. Follow **Step 2** above to run `python -m pip install .` inside `extern\engines\python`.

### Q3: `Incompatible Python version` error during pip install
MATLAB R2022b or earlier does not support Python 3.11. You must update your MATLAB Student Version to R2023b, R2024a, or R2024b, or install Python 3.10 if restricted to an older MATLAB release.

### Q4: `License checkout failed` error
Ensure your MathWorks Student License is activated on your computer. You can activate it by opening the MATLAB desktop application once and completing the sign-in.
