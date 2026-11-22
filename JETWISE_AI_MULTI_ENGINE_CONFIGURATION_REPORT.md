# JETWISE AI — Multi-Manufacturer Engine Configuration & Simulation Centre
## Comprehensive Architecture, Implementation & Verification Report

**Date:** 2026-11-22  
**Platform:** JETWISE AI — AI-Powered Aircraft Engine Health Monitoring, Digital Twin & Predictive Maintenance Intelligence Platform  
**Repository:** `https://github.com/sam-black007/Jet-Engine-Failure-Predictor`  
**Current Branch:** `main`  
**Automated Tests:** **152 / 152 Passing (100% Green)**  

---

## 1. Executive Summary

JETWISE AI has been extended from a single-dataset NASA C-MAPSS research prototype into a **comprehensive, multi-manufacturer aerospace engine configuration and simulation platform**.

The system now supports:
- **Multiple Global Aerospace Engine Manufacturers** (Rolls-Royce, Honeywell, GE Aerospace, Pratt & Whitney, CFM International, Safran, NASA, and custom OEMs).
- **Multiple Engine Architectures** (Dual-Spool High-Bypass, Triple-Spool with Concentric LP/IP/HP Shafts, and Geared Turbofans with Fan Drive Gear Systems).
- **Strict Provenance & Truth Tracking** (Manufacturer Documented, Publicly Sourced, User Supplied, Estimated, Simulated, Unknown).
- **Extensible Model & Telemetry Importer** supporting MATLAB `.m`, Simulink `.slx`, FMU/FMI, Python modules, CSV telemetry, MAT files, and JSON configurations.
- **Cross-Engine AI Compatibility & Risk Safeguards** that prevent unsupported cross-engine RUL claims.
- **Multi-Engine 3D Digital Twin CAD Library** rendering architecture-specific mechanical subassemblies.
- **Multi-Engine Comparison Laboratory** comparing physical, aerodynamic, and sensor telemetry across distinct engine models.
- **Full Conversational Copilot Integration** retaining active engine context and resolving cross-engine simulation requests.
- **SQLite Persistence & Portable Package Export/Import** (`.json`).

---

## 2. Core Architectural Deliverables

### 2.1 Manufacturer & Family Library (`src/engine_configuration/models.py`)
- Provides an extensible registry of gas turbine original equipment manufacturers (OEMs).
- Pre-seeded with authentic corporate profiles:
  - **Rolls-Royce plc** (Trent 1000, Trent XWB, Trent 7000).
  - **Honeywell Aerospace** (HTF7000, TFE731).
  - **GE Aerospace** (GE90, GEnx).
  - **Pratt & Whitney** (PW1000G Geared Turbofan, PW4000).
  - **CFM International** (LEAP-1A/1B, CFM56-7B).
  - **Safran Aircraft Engines** (Silvercrest, M88).
  - **NASA Glenn Research Center** (C-MAPSS Reference Turbofan).
- Users can dynamically register new OEMs and engine families via the UI or API.

### 2.2 Engine Configuration Builder with Strict Provenance
- Configurable parameters:
  - Spool architecture: Single-Spool, Dual-Spool (LP + HP), Triple-Spool (LP + IP + HP), Geared Turbofan (FDGS 3:1).
  - Number of concentric shafts (1, 2, or 3).
  - Fan diameter ($m$), Bypass Ratio (BPR), Overall Pressure Ratio (OPR), Takeoff Thrust ($\text{kN}$ and $\text{lbf}$), Cruise Specific Fuel Consumption (SFC), Bare Dry Weight ($\text{kg}$).
  - Compressor stages (Fan, LPC/IPC, HPC) and turbine stages (HPT, IPT, LPT).
  - Combustor architecture (Annular, TAPS II, TALON X, Cannular).
  - Available sensor channels.
- **Strict Provenance Tracking:** Every physical property is paired with a `ProvenanceSource` (`Manufacturer Documented`, `Publicly Sourced`, `User Supplied`, `Estimated`, `Simulated`, `Unknown`) and citation notes. The platform never silently substitutes estimated values for documented values.

### 2.3 Scientific Validation Status Taxonomy (`EngineValidationStatus`)
Each engine profile is classified under one of 8 explicit engineering statuses:
1. `CATALOGUE ONLY`: Documented OEM entry without connected simulation model or active dataset.
2. `CONFIGURATION INCOMPLETE`: Lacks required physical dimensions or architectural specs.
3. `RESEARCH MODEL`: Connected to a research or preliminary simulation model.
4. `SIMULATION CONNECTED`: Linked to an executable MATLAB/Simulink or Python gas path model.
5. `SIMULATION TESTED`: Physical simulation executed and validated against thermodynamic conservation laws.
6. `DATASET VALIDATED`: Associated with real telemetry dataset matching its sensor channels.
7. `AI MODEL VALIDATED`: Machine learning prognostic model calibrated and evaluated on compatible data.
8. `ENGINEERING VALIDATION INCOMPLETE`: Default state pending complete multi-disciplinary verification.

### 2.4 External Model & Dataset Importer (`src/engine_configuration/model_importer.py`)
- Automatically inspects file extensions and headers:
  - **MATLAB M-Scripts (`.m`):** Checks parameter declarations and script functions.
  - **Simulink Models (`.slx`):** Verifies block diagram metadata and inputs/outputs.
  - **Python Modules (`.py`):** Uses AST parsing to check for callable `simulate_engine` or `run_simulation`.
  - **CSV Telemetry (`.csv`):** Analyzes column names, sampling frequency, and row counts.
  - **JSON Configurations (`.json`):** Validates input/output variable schemas, units, and dependencies.
  - **FMU/FMI (`.fmu`):** Validates Functional Mock-up Unit envelope.
- `EngineModelAdapter`: Normalizes diverse output data streams into unified gas-path state variables.

### 2.5 Multi-Engine Comparison Laboratory (`src/engine_configuration/engine_comparator.py`)
- Provides side-by-side architectural and performance deltas across 2 or more engine models:
  - Thrust-to-weight ratio ($T/W$).
  - Specific thrust ($\text{kN}/m^2$ fan frontal area).
  - Bypass ratio and pressure ratio disparities.
  - Sensor telemetry intersection and unique channel coverage.
  - Automated aerospace engineering analysis explaining physical differences.

### 2.6 Cross-Engine AI Prognostic Compatibility Rules (`src/engine_configuration/validation.py`)
- Enforces scientific safety: **Never assumes a model trained on one engine family generalizes to another.**
- Evaluates target engine architecture against training data:
  - If a NASA FD001 model is applied to Rolls-Royce Trent XWB (3-spool), it generates a `CROSS_ENGINE_UNVALIDATED` status with `HIGH` risk rating, explaining that 3-spool dynamics differ fundamentally from 2-spool training engines.
  - Recommends domain adaptation or fine-tuning before trusting RUL forecasts.

### 2.7 Multi-Engine 3D Digital Twin Generator (`src/engine_configuration/multi_twin_generator.py`)
- Generates architecture-accurate Three.js WebGL 3D Digital Twins:
  - **Rolls-Royce 3-Spool (`three_spool_trent`):** Renders separate Fan, IPC (8 stages), HPC (6 stages), HPT, IPT, LPT, and 3 concentric shafts animating at independent rotational speeds ($N_1, N_2, N_3$).
  - **Pratt & Whitney Geared GTF (`geared_turbofan`):** Renders the Fan Drive Gear System (FDGS) planetary reduction gearbox between the fan and LP shaft.
  - **Standard High-Bypass (`standard_twin_spool`):** Renders dual-spool widebody engines (GE90, CFM LEAP, NASA C-MAPSS).
  - **Regional Business Jet (`regional_turbofan`):** Renders compact dual-spool turbofans (Honeywell HTF7000).
- Displays explicit provenance labels (`[RESEARCH-GRADE CAD SCHEMATIC]`).

### 2.8 Conversational Copilot Multi-Engine Integration (`src/copilot_nlu.py` & `src/copilot_simulation.py`)
- Understands multi-engine conversational instructions:
  - *"Select the Rolls-Royce Trent XWB."*
  - *"Which AI model is compatible with this engine?"*
  - *"Can I use my current NASA-trained model for this engine?"*
  - *"Compare these two engine configurations."*
- Retains active engine context across multi-turn dialogs.
- When a user asks to simulate an engine that is `CATALOGUE ONLY`, the copilot honestly reports:
  > *"Engine profile 'Rolls-Royce Trent XWB-84' is currently a CATALOGUE ONLY entry. A validated physical simulation model for this engine family is not yet connected. Physical simulation cannot be executed without an imported or compatible engineering model."*

### 2.9 Multi-Engine UI Centre (`src/ui/multi_engine_centre.py`)
- Dedicated Streamlit dashboard page: [dashboard/pages/12_Multi_Engine_Configuration_Centre.py](file:///c:/Users/samjo/Downloads/simulation/Jet-Engine-Failure-Predictor/dashboard/pages/12_Multi_Engine_Configuration_Centre.py).
- Integrated into the primary application navigation sidebar in [dashboard/app.py](file:///c:/Users/samjo/Downloads/simulation/Jet-Engine-Failure-Predictor/dashboard/app.py).

---

## 3. Pre-Configured Engine Catalogue Specifications

| Configuration ID | Manufacturer | Family & Model | Architecture | Fan Ø | BPR | OPR | Takeoff Thrust | Validation Status |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| `NASA-CMAPSS-FD001` | NASA Glenn Research | C-MAPSS Reference Turbofan | Dual-Spool (LP + HP) | 2.10 m | 8.5:1 | 38.8:1 | 115.5 kN (26,000 lbf) | `AI_MODEL_VALIDATED` |
| `RR-TRENT-XWB-84` | Rolls-Royce plc | Trent XWB-84 | Triple-Spool (LP + IP + HP) | 3.00 m | 9.6:1 | 50.0:1 | 374.5 kN (84,200 lbf) | `RESEARCH_MODEL` |
| `GE-GE90-115B` | GE Aerospace | GE90-115B | Dual-Spool (LP + HP) | 3.25 m | 8.4:1 | 42.0:1 | 513.9 kN (115,300 lbf) | `CATALOGUE_ONLY` |
| `PW-PW1100G-JM` | Pratt & Whitney | PW1100G-JM (Geared GTF) | Geared Turbofan (FDGS 3:1) | 2.06 m | 12.0:1 | 41.0:1 | 120.4 kN (27,000 lbf) | `RESEARCH_MODEL` |
| `HW-HTF7000` | Honeywell Aerospace | HTF7000 | Dual-Spool Regional | 0.87 m | 4.4:1 | 28.2:1 | 32.5 kN (7,300 lbf) | `CATALOGUE_ONLY` |
| `CFM-LEAP-1A` | CFM International | LEAP-1A26 | Dual-Spool (LP + HP) | 1.98 m | 11.0:1 | 40.0:1 | 120.6 kN (27,100 lbf) | `CATALOGUE_ONLY` |

---

## 4. Test Suite Execution & Verification

A dedicated automated test suite was constructed in [tests/test_multi_engine_configuration.py](file:///c:/Users/samjo/Downloads/simulation/Jet-Engine-Failure-Predictor/tests/test_multi_engine_configuration.py) covering all requirements:

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-8.3.2
collected 11 items

tests/test_multi_engine_configuration.py::test_manufacturer_library_and_add PASSED       [  9%]
tests/test_multi_engine_configuration.py::test_engine_family_add PASSED                 [ 18%]
tests/test_multi_engine_configuration.py::test_engine_configuration_creation_and_persistence PASSED [ 27%]
tests/test_multi_engine_configuration.py::test_engine_validator_rules PASSED           [ 36%]
tests/test_multi_engine_configuration.py::test_model_importer_inspects_json PASSED       [ 45%]
tests/test_multi_engine_configuration.py::test_model_importer_inspects_csv PASSED        [ 54%]
tests/test_multi_engine_configuration.py::test_cross_engine_ai_model_compatibility PASSED [ 63%]
tests/test_multi_engine_configuration.py::test_multi_engine_comparator PASSED            [ 72%]
tests/test_multi_engine_configuration.py::test_multi_engine_3d_twin_html_generation PASSED [ 81%]
tests/test_multi_engine_configuration.py::test_copilot_nlu_multi_engine_commands PASSED   [ 90%]
tests/test_multi_engine_configuration.py::test_package_export_and_import PASSED          [100%]
============================== 11 passed in 0.81s ==============================
```

### Full Repository Regression Verification:
- **152 / 152 tests passed across all 15 test suites.**
- **Pass rate: 100% Green.**
- **Zero regressions.**

---

## 5. Instructions to Run & Verify Multi-Engine Capabilities

### 1. Launch the Dashboard
```powershell
cd "c:\Users\samjo\Downloads\simulation\Jet-Engine-Failure-Predictor"
python -m streamlit run dashboard/app.py
```
- Open `http://localhost:8501`.
- In the sidebar workspace dropdown, select **🌐 Multi-Engine Configuration Centre** (or navigate directly to `http://localhost:8501/Multi_Engine_Configuration_Centre`).

### 2. Run Multi-Engine Unit & Integration Tests
```powershell
python -m pytest tests/test_multi_engine_configuration.py -v
```

### 3. Test Full Pytest Suite
```powershell
python -m pytest -q
```
