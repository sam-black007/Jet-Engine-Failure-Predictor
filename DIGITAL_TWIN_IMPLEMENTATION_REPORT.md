# THRUSTLAB — 3D TURBOFAN ENGINE DIGITAL TWIN
## IMPLEMENTATION & VERIFICATION REPORT

**Feature:** Mandatory Intelligence Feature 1 — Interactive 3D Aircraft Engine Digital Twin  
**Repository:** `Jet-Engine-Failure-Predictor`  
**Status:** FULLY IMPLEMENTED, INTEGRATED & VERIFIED  
**Authors:** THRUSTLAB Architecture & Engineering Team  
**Date:** November 22, 2026  

---

## 1. EXECUTIVE SUMMARY

The **Interactive 3D Aircraft Engine Digital Twin** has been developed as an integrated, real-time spatial intelligence component within the **THRUSTLAB** platform. It provides maintenance engineers, fleet operators, and reliability analysts with an interactive, geometrically accurate 3D visualization of a dual-spool high-bypass commercial turbofan engine.

The Digital Twin translates multi-dimensional sensor telemetry, deep learning Remaining Useful Life (RUL) predictions, unsupervised anomaly scores, and feature attributions into physical engine subsystems.

```
+-----------------------------------------------------------------------------------------+
|                               THRUSTLAB DIGITAL TWIN ARCHITECTURE                      |
+-----------------------------------------------------------------------------------------+
|  [NASA C-MAPSS / N-CMAPSS Telemetry]  -->  [Deep Neural Predictor (Pinball LSTM / TCN)] |
|                                                    |                                    |
|                                                    v                                    |
|                                      [Uncertainty & Anomaly Engine]                     |
|                                                    |                                    |
|       +--------------------------------------------+-----------------------------+      |
|       |                                                                          |      |
|       v                                                                          v      |
|  [Three.js WebGL Engine]                                              [Plotly 3D Meshes]|
|  - Real-time 60 FPS WebGL canvas                                      - Mathematical    |
|  - Dual-spool rotating animations                                       parametric 3D   |
|  - Interactive exploded view slider                                     mesh projection |
|  - Raycasting component click & inspection                            - Thermal gradient|
|  - Time-series degradation playback                                     heat-mapping    |
+-----------------------------------------------------------------------------------------+
```

---

## 2. 3D TURBOFAN ENGINE ARCHITECTURE

The 3D model decomposes the turbofan engine into **nine distinct physical subassemblies**, each individually modeled, positioned along the longitudinal thrust axis ($Z$-axis), and dynamically rendered:

| Index | Subassembly Name | Code Identifier | Axial Range ($Z$) | Radius Range ($R$) | Primary Associated Sensors | Inferred Failure Mode |
|:---:|:---|:---|:---:|:---:|:---|:---|
| **1** | **Fan Assembly** | `fan` | $Z \in [-5.5, -4.0]$ | $R \in [0.8, 3.8]$ | `s_1` (T2), `s_5` (P2), `s_13` (Nf) | Fan blade tip rubbing, FOD, unbalance |
| **2** | **Low-Pressure Compressor (LPC)** | `lpc` | $Z \in [-4.0, -2.5]$ | $R \in [0.6, 2.2]$ | `s_2` (T24), `s_6` (P24), `s_14` (NRf) | Booster stage aerodynamic stall, blade erosion |
| **3** | **High-Pressure Compressor (HPC)** | `hpc` | $Z \in [-2.5, -0.5]$ | $R \in [0.4, 1.8]$ | `s_3` (T30), `s_7` (Ps30), `s_11` (phi) | HPC blade tip clearance growth, severe fouling |
| **4** | **Combustion Chamber** | `combustor` | $Z \in [-0.5, 1.0]$ | $R \in [0.5, 1.5]$ | `s_4` (T50), `s_12` (BPR), `s_15` (NRc) | Thermal liner hot-spots, nozzle spray coking |
| **5** | **High-Pressure Turbine (HPT)** | `hpt` | $Z \in [1.0, 2.5]$ | $R \in [0.4, 1.7]$ | `s_8` (Nc), `s_9` (T48), `s_17` (htBleed) | Thermal barrier coating spallation, blade creep |
| **6** | **Low-Pressure Turbine (LPT)** | `lpt` | $Z \in [2.5, 4.2]$ | $R \in [0.5, 2.0]$ | `s_10` (P50), `s_16` (FarB) | Aerodynamic efficiency loss, seal degradation |
| **7** | **Exhaust Nozzle** | `exhaust_nozzle`| $Z \in [4.2, 5.8]$ | $R \in [0.6, 2.4]$ | `s_18` (P15), `s_19` (Nf_dmd) | Core jet velocity dissipation, plume anomalies |
| **8** | **Engine Nacelle / Casing** | `engine_casing` | $Z \in [-5.8, 4.0]$ | $R \in [3.7, 4.2]$ | `s_20` (PCNFR_dmd), `s_21` (W31) | Nacelle acoustic liner debonding, cowl damage |
| **9** | **Central Shaft Assembly** | `shaft_assembly`| $Z \in [-5.0, 5.0]$ | $R \in [0.0, 0.35]$| `s_8` (Nc), `s_13` (Nf) | Bearing race spalling, shaft bow, eccentricity |

---

## 3. RENDERING ENGINE & INTERACTIVE CAPABILITIES

The Digital Twin is implemented via two complementary rendering modalities:

### A. Three.js High-Performance WebGL Engine
Embedded directly via `src/digital_twin.py::render_threejs_digital_twin_html` into Streamlit:
1. **Real-Time 3D Interaction:** Full 60 FPS OrbitControls supporting unrestricted 3D rotation, pitch, yaw, pan, and smooth exponential zoom.
2. **Camera Reset Action:** Dedicated UI reset triggers instant camera realignment to default isometric perspective $(X=12, Y=8, Z=15)$.
3. **Exploded View Transformation:** Axial translation along the thrust vector:
   $$Z_{\text{exploded}} = Z_{\text{nominal}} + \delta_i \cdot \alpha_{\text{sep}}$$
   where $\delta_i$ is the component separation weight vector and $\alpha_{\text{sep}} \in [0.0, 1.0]$ is controlled smoothly by an interactive slider.
4. **Dynamic Raycasting Click Detection:** User mouse clicks on any 3D subassembly cast a ray from the camera, immediately selecting the component, highlighting its bounding mesh, and projecting its diagnostic HUD.
5. **Component Inspection HUD:** Displays component status, associated physical sensors, live readings, and inferred wear indicators.
6. **Time-Based Degradation Playback:** Dedicated controls (`Play`, `Pause`, `Step -1`, `Step +1`, `Reset`) allow scrubbing and simulating degradation evolution from cycle 1 to end-of-life.

### B. Plotly 3D Parametric Surface Mesh
Implemented in `src/digital_twin.py::generate_turbofan_3d_figure`:
1. **Mathematical Representation:** Exact cylindrical and conical meshes parameterized by:
   $$X(\theta, z) = R(z) \cos(\theta), \quad Y(\theta, z) = R(z) \sin(\theta), \quad Z(\theta, z) = z$$
2. **Thermal & Strain Heat-Mapping:** Surface color gradients dynamically reflect real-time temperature, pressure ratios, and computed degradation severity.

---

## 4. INTEGRATION WITH REAL TELEMETRY & AI PREDICTIONS

The Digital Twin is natively connected to the backend inference and data pipelines:
- **Telemetry Ingestion:** Receives synchronized real-time multi-sensor matrices from `src.data_loader` and `src.data_replay`.
- **RUL Prognostics:** Translates RUL forecasts from the model registry (MSE LSTM, Pinball Asymmetric LSTM, Temporal Transformer) into component wear progressions.
- **Anomaly Detection:** Residual anomalies computed by `EngineAnomalyDetector` highlight aberrant components with diagnostic pulsing warnings.
- **Explainable Attribution (XAI):** Gradient and perturbation attributions identify which physical component was the primary driver of the RUL prediction.

---

## 5. DIAGNOSTIC INTERPRETATION NOTICE

> [!IMPORTANT]
> **AEROSPACE DIAGNOSTIC DISCLAIMER**  
> Component highlighting in this 3D Digital Twin represents an **inferred diagnostic visualization** derived from multi-sensor statistical correlations and deep learning prognostic models. It does **not** represent confirmed physical structural damage until validated by physical borescope inspection, eddy current non-destructive testing (NDT), or oil debris spectrometry in accordance with FAA/EASA certified maintenance manuals.

---

## 6. VERIFICATION & PLAYWRIGHT TEST RESULTS

The Digital Twin was verified through automated tests:

### Unit & Integration Tests (`tests/test_digital_twin.py`)
- `test_nine_turbofan_components_defined`: **PASSED** (all 9 required subassemblies verified).
- `test_plotly_assembled_and_exploded_figure`: **PASSED** (geometry and axial offset mathematics validated).
- `test_threejs_html_generation_and_features`: **PASSED** (WebGL canvas, OrbitControls, raycaster click events, and HUD verified).

### Browser End-to-End Automation (`run_playwright_suite.py`)
- Assembled View Render: **PASSED** (captured to `results/playwright/digital_twin_assembled.png`).
- Exploded View Separation: **PASSED** (captured to `results/playwright/digital_twin_exploded.png`).
- Zero Browser Console Errors: **0 console errors, 0 failed network requests**.
