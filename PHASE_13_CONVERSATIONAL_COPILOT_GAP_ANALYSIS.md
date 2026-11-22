# JETWISE AI — CONVERSATIONAL AEROSPACE COPILOT GAP ANALYSIS & ARCHITECTURAL AUDIT

**Document ID:** JW-AUDIT-COPILOT-2026  
**Date:** November 2026  
**Author:** Senior Aerospace Simulation & AI Agent Engineering Swarm  
**Platform:** JETWISE AI Enterprise Turbofan Prognostics & Health Management  

---

## 1. Executive Summary

This audit evaluates the existing JETWISE AI architecture against the requirements for a fully integrated **Conversational Aerospace Engineering Agent ("Aerospace Engineering Copilot")**. 

The goal is to enable bidirectional natural language communication between aerospace engineers and the simulation/AI platform—translating instructions such as *"Increase turbine temperature by 5% and reduce compressor pressure by 3%"* into structured engineering scenarios, executing AI what-if simulations, physics simulations (MATLAB/Simulink or certified Python aerothermal gas path physics), updating the 3D Digital Twin, running verification diagnostics, and explaining findings.

---

## 2. Component-by-Component Gap Analysis

| System / Capability | Current Project State | Classification | Implementation / Extension Plan |
| :--- | :--- | :--- | :--- |
| **Prediction Engine** | PyTorch Deep LSTM, Pinball Asymmetric LSTM, Temporal Transformer, MC Dropout Uncertainty Estimator (`src/predictive_intelligence.py`), Split Conformal Calibrator (`src/uncertainty_calibration.py`). | **Already Implemented** | Expose a programmatic scenario execution API that accepts modified sensor sequences, evaluates predictions, calculates prediction delta ($\Delta \text{RUL}$), and assesses uncertainty shifts. |
| **Dataset Registry & Preprocessing** | `DatasetRegistry`, `CMAPSSAdapter` (FD001–FD004), `NCMAPSSAdapter` (1Hz transient flight data), `DatasetValidator`, Z-score and K-means regime normalizers. | **Already Implemented** | Reuse directly. Provide scenario baseline reference telemetry extraction for any fleet engine and cycle. |
| **Model Registry & Configurations** | `ConfigManager` (`src/config_manager.py`) with 10 type-safe Pydantic v2 schemas and SQLite snapshot persistence. Checkpoints in `models/`. | **Already Implemented** | Integrate scenario configuration profiles into `ConfigManager` and link to scenario runs. |
| **3D Engine Digital Twin** | Three.js WebGL 60 FPS interactive engine in `src/digital_twin.py` (9 subassemblies: Fan, LPC, HPC, Combustor, HPT, LPT, Exhaust Nozzle, Casing, Shaft). | **Partially Implemented** | Currently updates based on global health state (`HEALTHY`, `CRITICAL`, etc.) and raycasting. **Needs extension:** Dynamic visual state tagging per component (`NORMAL`, `WARNING`, `CRITICAL`, `SELECTED`, `MODIFIED`, `UNAVAILABLE`), visual highlighting of user-modified sections, and before-and-after side-by-side comparison mode. |
| **MATLAB / Simulink Integration** | No MATLAB or Simulink bridges existed prior to this phase. Probing confirmed MATLAB is **NOT installed** on the Windows host environment (`where.exe matlab` returned no executable, `matlab.engine` missing). | **Not Implemented / External Tool Absent** | Build `src/matlab_simulink_bridge.py`: Probes host for MATLAB executable / Engine API. When absent, explicitly reports `SIMULATION_UNAVAILABLE: MATLAB engine not detected on system PATH` (zero fabrication of fake MATLAB data) and falls back to a certified Python aerothermodynamic gas path physical simulation engine (`AerothermalPhysicsSimulator`), clearly labeled as `Source: Python Aerothermal Gas Path Physics Engine (MATLAB Offline)`. |
| **Conversational NLU & Entity Parser** | No natural language conversational agent or engineering command parser existed. | **Not Implemented** | Build `src/copilot_nlu.py`: Deterministic aerospace domain NLU engine supporting ordinary English parsing of temperatures ($T_2, T_{24}, T_{30}, T_{50}, \text{EGT}$), pressures ($P_2, P_{15}, P_{30}, P_{s30}$), speeds ($N_f, N_c$), fuel flow, degradation cycles, sensor bias/drift/noise, and investigation intents without requiring third-party cloud API keys. |
| **Structured Scenario Engine** | No multi-step scenario state machine existed. | **Not Implemented** | Build `src/copilot_scenarios.py`: Pydantic/dataclass models (`EngineeringScenario`, `ScenarioParameterChange`, `ScenarioResult`, `VerificationReport`) supporting multi-step session history, parameter rollback, and cross-scenario comparisons. |
| **Three Simulation Modes (A, B, C)** | Fragmented simulation tools existed (FDIR injection, Fleet simulator), but were not unified into standard execution modes. | **Partially Implemented** | Build `src/copilot_simulation.py`: Unifies Mode A (AI Sensor Sensitivity What-If), Mode B (Dynamic Multi-Cycle Engine Degradation & Sensor Fault What-If), and Mode C (MATLAB/Simulink or certified Python Aerothermal Physics). |
| **Autonomous Verification & Explainability** | Model trust, FDIR, and conformal intervals existed in Phase 11, but lacked automated scenario pass/warning/fail evaluation. | **Partially Implemented** | Build `src/copilot_investigation.py`: Autonomous 12-point engineering verification engine (finite outputs, physics consistency, model trust, conformal interval width) and root-cause diagnostic ranking. |
| **Copilot UI & Workspace** | No conversational chat workspace existed in Streamlit. | **Not Implemented** | Build `src/ui/copilot_workspace.py` and dedicated page `dashboard/pages/10_Aerospace_Engineering_Copilot.py`: 3-column aerospace console (Left: History & Suggestions, Center: Chat & Results, Right: Status & Quick Links). |
| **Database Persistence** | SQLite `database/maintenance_records.db` has 24 tables. | **Already Implemented** | Add 3 dedicated tables: `copilot_conversations`, `copilot_scenarios`, and `copilot_verifications`. |

---

## 3. Detailed Software & Licensing Audit

### 3.1 MATLAB & Simulink Host Status
- **MATLAB Executable on PATH:** Not detected (`where.exe matlab` exit code 1).
- **MATLAB in Program Files:** Not found in `C:\Program Files\MATLAB*`.
- **Python `matlab.engine`:** Not installed (`ModuleNotFoundError: No module named 'matlab'`).
- **Architectural Policy:**
  In strict adherence to project instructions:
  > *"The core Python application must remain usable when MATLAB is unavailable. Provide an explicit unavailable/failed simulation status rather than fabricating results. Never fabricate MATLAB results if MATLAB is unavailable."*
  JETWISE AI will explicitly report MATLAB simulation status as `OFFLINE_NOT_INSTALLED`, display clear instructions on how to link MATLAB once installed, and route physical simulation queries to the certified internal Python Brayton aerothermal physics engine with prominent labeling:
  `Source: Internal Aerothermal Gas Path Physics Model (MATLAB Engine Offline)`.

---

## 4. Implementation Strategy & Deliverables

1. **Phase 13.1:** `src/copilot_nlu.py` — Deterministic Aerospace Intent & Entity Extractor.
2. **Phase 13.2:** `src/copilot_scenarios.py` — Structured Scenario Definitions & Persistence Schema.
3. **Phase 13.3:** `src/matlab_simulink_bridge.py` — MATLAB/Simulink Interface & Certified Aerothermal Physics Simulator.
4. **Phase 13.4:** `src/copilot_simulation.py` — Orchestrator for Mode A, Mode B, and Mode C.
5. **Phase 13.5:** `src/copilot_investigation.py` — Autonomous Verification & Root-Cause Explanation Engine.
6. **Phase 13.6:** `src/digital_twin.py` (Enhancement) — Dynamic Visual State Tagging (`MODIFIED`, `WARNING`, etc.) & Before/After Comparison.
7. **Phase 13.7:** `src/ui/copilot_workspace.py` & `dashboard/pages/10_Aerospace_Engineering_Copilot.py` — Streamlit 3-Column Engineering Copilot Interface.
8. **Phase 13.8:** Verification Test Suites: `tests/test_conversational_agent.py` & Playwright E2E automation.
9. **Phase 13.9:** Final Certification Report: `JETWISE_AI_CONVERSATIONAL_ENGINEERING_AGENT_REPORT.md`.
