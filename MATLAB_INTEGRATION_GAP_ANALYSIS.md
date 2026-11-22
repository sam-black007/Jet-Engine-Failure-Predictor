# THRUSTLAB — MATLAB & Simulink Aerospace Simulation Integration
## Architecture Audit & Gap Analysis

**Date:** 2026-11-22  
**Platform:** THRUSTLAB — AI-Powered Aircraft Engine Health Monitoring, Digital Twin & Predictive Maintenance Intelligence Platform  
**System Environment:** Windows 11 x64, Python 3.11.9  
**Repository Branch:** `main` (clean working tree)

---

## 1. Executive Summary

This document establishes the architecture audit and gap analysis for integrating MATLAB and Simulink into the **THRUSTLAB** platform. The objective is to allocate responsibilities with scientific rigor:
- **MATLAB/Simulink:** Dynamic physical aerothermodynamic gas path simulation of two-spool turbofan engines, transient mechanics, and multi-station sensor generation.
- **Python Backend:** High-speed AI inference (LSTM, Attention-LSTM, BiLSTM, 1D-CNN, Transformer), conformal uncertainty calibration, Remaining Useful Life (RUL) estimation, and predictive maintenance scheduling.
- **Interactive 3D Digital Twin:** Dynamic physical state visualization, component thermomechanical mapping, and structural inspection.
- **Natural-Language Copilot:** Conversational engineering interaction translating high-level flight profiles and fault injections into validated physical simulations.

In strict adherence to the project's **Scientific Integrity Directive**, the platform will never fabricate MATLAB execution when MATLAB is offline. When MATLAB/Simulink is detected, it executes natively via the MATLAB Engine API or headless batch runner; when offline or uninstalled, it transparently tags the simulation status as `OFFLINE_NOT_INSTALLED`, runs the certified Python Aerothermal Gas Path Physics Engine, and provides actionable setup guidance for connecting the user's MATLAB Student Version.

---

## 2. Existing Repository Audit

### 2.1 Backend Modules (`src/`)
| Module | Capability | Integration Touchpoint |
|---|---|---|
| `src/data_loader.py` & `src/configurable_data.py` | C-MAPSS FD001–FD004 ingestion, multi-regime normalization | Benchmark comparison with physical simulation outputs |
| `src/preprocessing.py` | Sliding window feature extraction ($W=30$), sensor scaling | Telemetry adapter input formatting |
| `src/models/` & `src/model_factory.py` | PyTorch RUL predictors (LSTM, Asymmetric LSTM, Transformer, CNN) | AI prognostic inference over simulation telemetry |
| `src/uncertainty_calibration.py` | Conformal prediction intervals, Monte Carlo dropout | Quantifying uncertainty of physics-driven predictions |
| `src/digital_twin.py` | Three.js turbofan digital twin, 9 subassemblies, dynamic shader coloring | Real-time synchronization with $T_{30}, T_{40}, T_{50}, N_f, N_c$ |
| `src/copilot_nlu.py` & `src/copilot_simulation.py` | Conversational NLP intent parser and scenario generator | Natural-language execution of MATLAB/Simulink simulations |
| `src/database_migrations.py` | SQLite schema management (`database/maintenance_records.db`) | Adding `matlab_simulations` table for scenario persistence |

### 2.2 Model Checkpoints
- `models/checkpoints/best_lstm.pt`: Optimized PyTorch LSTM model trained on official NASA FD001 sequences ($W=30$).
- `models/checkpoints/transformer_model.pt`: Self-attention prognostic model.
- `models/baselines/`: Linear Regression, Random Forest, SVR, and MLP baseline predictors.

---

## 3. Host MATLAB & Simulink Environment Audit

### 3.1 Python Runtime Verification
- **Host Python Version:** `Python 3.11.9 (tags/v3.11.9:de5405b, Apr  2 2024, 18:05:07) [MSC v.1938 64 bit (AMD64)]`
- **MATLAB Engine for Python Compatibility Matrix:**
  - MATLAB R2023a: Supports Python 3.8, 3.9, 3.10 (Update 3+ added experimental 3.11 support)
  - MATLAB R2023b: Officially supports Python 3.9, 3.10, 3.11
  - MATLAB R2024a: Officially supports Python 3.9, 3.10, 3.11
  - MATLAB R2024b: Officially supports Python 3.9, 3.10, 3.11, 3.12

### 3.2 Detection Probing Results
1. `shutil.which("matlab")`: Returned `None` (not on Windows system `PATH`).
2. Windows Registry (`HKLM:\SOFTWARE\MathWorks\MATLAB`): No system-wide MATLAB keys detected.
3. Common Directory Probing:
   - `C:\Program Files\MATLAB`: Not present.
   - `C:\MATLAB`: Not present.
   - `D:\MATLAB`: Not present.
4. Python Module `matlab.engine`: `ModuleNotFoundError: No module named 'matlab'`.

### 3.3 Status Classification
- **MATLAB Executable:** `OFFLINE_NOT_INSTALLED`
- **Simulink:** `MISSING`
- **Simscape Fluids:** `MISSING`
- **Aerospace Blockset:** `MISSING`
- **Python Engine API:** `MISSING`

---

## 4. Architectural Gap Analysis & Solutions

### Gap 1: Standalone Detection & Capability Matrix
- **Current State:** Basic check in `src/matlab_simulink_bridge.py` only checked `shutil.which` and `import matlab.engine`.
- **Required Solution:** Dedicated `src/matlab_integration/matlab_detector.py` performing deep registry scans, environment variable checks (`MATLAB_HOME`, `MATLABROOT`), dynamic execution checks (`matlab -batch "ver; license('inuse'); exit;"`), and returning structured capability states (`AVAILABLE`, `MISSING`, `INCOMPATIBLE`, `LICENSE_UNAVAILABLE`).

### Gap 2: Authentic Turbofan Physics Models
- **Current State:** Embedded Python calculations in `matlab_simulink_bridge.py`.
- **Required Solution:** High-fidelity MATLAB/Simulink assets in `matlab_models/`:
  - `matlab_models/scripts/run_turbofan_simulation.m`: Full Brayton cycle gas-path physics script modeling 9 engine subsystems (Fan, LPC, HPC, Combustor, HPT, LPT, Nozzle, LP Shaft, HP Shaft) with dynamic acceleration, efficiency modifiers, and export to `.mat` and `.json`.
  - `matlab_models/models/create_turbofan_slx.m`: Programmatic Simulink block diagram generator creating `turbofan_engine_model.slx`.
  - `matlab_models/configuration/default_parameters.json`: Engineering units, default operating points (Sea Level Static, Cruise, Climb), and safe ranges.

### Gap 3: Telemetry Adapter & C-MAPSS Feature Space Mapping
- **Current State:** No formal conversion between physical simulation units and C-MAPSS feature space.
- **Required Solution:** `src/matlab_integration/telemetry_adapter.py`:
  - Explicit physical unit conversions:
    - Temperature: $\text{Kelvin} \times 1.8 = ^\circ\text{Rankine}$
    - Pressure: $\text{kPa} \times 0.145038 = \text{psia}$
    - Speed: $\text{rad/s} \times \frac{60}{2\pi} = \text{RPM}$
  - Sliding window buffer ($W=30$ cycles) for sequence generation.
  - Transparent boundary validation with explicit distinction between compatible signals ($T_{24}, T_{30}, T_{50}, P_{30}, N_f, N_c$, fuel flow) and unmodeled static signals.

### Gap 4: Dedicated Simulation Centre User Interface
- **Current State:** Copilot workspace in Page 10 had a simulation tab, but lacks a dedicated full-scale MATLAB Simulation Centre.
- **Required Solution:** `dashboard/pages/11_MATLAB_Simulink_Centre.py` backed by `src/ui/matlab_centre.py` featuring:
  - Connection & Toolboxes Status Monitor.
  - Interactive Parameter Configuration Panel.
  - Execution Controls (MATLAB vs Certified Python Fallback).
  - Multi-Station Gas Path Visualizations ($T\text{-}s$ diagram, timeseries telemetry).
  - Baseline vs Modified Simulation Comparison.
  - AI Prognostic RUL Evaluation.
  - Synchronized 3D Digital Twin Viewer.

### Gap 5: Scenario Persistence & Reproducibility
- **Current State:** Copilot history persisted in `copilot_scenarios` table.
- **Required Solution:** Dedicated `matlab_simulations` table in SQLite tracking `scenario_id`, parameter configs, execution mode, telemetry outputs, validation flags, and AI RUL predictions.

---

## 5. Implementation Roadmap

1. **Step 1:** Establish `src/matlab_integration/` module package (`matlab_detector.py`, `simulation_inputs.py`, `simulation_outputs.py`, `simulation_validator.py`, `telemetry_adapter.py`, `matlab_engine_manager.py`, `simulink_runner.py`).
2. **Step 2:** Develop authentic MATLAB/Simulink models and scripts in `matlab_models/`.
3. **Step 3:** Implement SQLite migration for `matlab_simulations`.
4. **Step 4:** Construct `src/ui/matlab_centre.py` and `dashboard/pages/11_MATLAB_Simulink_Centre.py`.
5. **Step 5:** Connect 3D Digital Twin and Conversational Copilot to the new integration layer.
6. **Step 6:** Author automated test suite in `tests/test_matlab_integration.py`.
7. **Step 7:** Create documentation: `MATLAB_SETUP_FOR_THRUSTLAB_AI.md` and `THRUSTLAB_AI_MATLAB_SIMULINK_INTEGRATION_REPORT.md`.
8. **Step 8:** Run Playwright end-to-end verification and capture UI screenshots.
9. **Step 9:** Stage, commit, and push clean code to GitHub repository.
