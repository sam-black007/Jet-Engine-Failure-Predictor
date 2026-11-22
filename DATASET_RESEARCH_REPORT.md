# THRUSTLAB: COMPREHENSIVE DATASET RESEARCH & SOURCE VERIFICATION REPORT
## Aircraft Engine Degradation, Turbofan Telemetry & Predictive Maintenance Benchmarks

**Document Version:** 1.0.0  
**Research Date:** 2026-11-21  
**Lead Researcher:** Senior Data Engineering Architect & ML Research Specialist  
**Project:** Jet Engine Failure Predictor (THRUSTLAB)  

---

## 1. RESEARCH OBJECTIVE & METHODOLOGY

To build a truly dynamic, multi-dataset predictive maintenance platform for aircraft propulsion, this report investigates publicly available datasets beyond NASA C-MAPSS FD001. Each candidate dataset has been evaluated against:
1. Provenance and official hosting authority.
2. Aerodynamic and thermodynamic relevance to gas turbine / turbofan propulsion.
3. Availability of ground-truth Remaining Useful Life (RUL) or verifiable degradation progression.
4. Licensing, distribution terms, file format, and integration complexity.
5. Compatibility with deep time-series neural architectures (LSTM, TCN, Transformers).

---

## 2. COMPREHENSIVE DATASET CATALOG & COMPARATIVE MATRIX

| Dataset Name | Official Publisher | Category | Format | Engines / Units | Sensors | Conditions | Failure Modes | RUL Ground Truth | License & Availability | Integration Feasibility |
|---|---|---|---|---|---|---|---|---|---|---|
| **NASA C-MAPSS FD001** | NASA Ames PCoE (Saxena et al., 2008) | Aircraft Turbofan | Space-delimited `.txt` | 100 Train / 100 Test | 21 sensors + 3 op settings | 1 (Sea Level) | 1 (HPC Degradation) | Available (`RUL_FD001.txt`) | Public Domain (NASA Open Data) | **Fully Supported** (Active Primary) |
| **NASA C-MAPSS FD002** | NASA Ames PCoE (Saxena et al., 2008) | Aircraft Turbofan | Space-delimited `.txt` | 260 Train / 259 Test | 21 sensors + 3 op settings | 6 Operating Regimes | 1 (HPC Degradation) | Available (`RUL_FD002.txt`) | Public Domain (NASA Open Data) | **High** (Local raw files present) |
| **NASA C-MAPSS FD003** | NASA Ames PCoE (Saxena et al., 2008) | Aircraft Turbofan | Space-delimited `.txt` | 100 Train / 100 Test | 21 sensors + 3 op settings | 1 (Sea Level) | 2 (HPC + Fan Degradation) | Available (`RUL_FD003.txt`) | Public Domain (NASA Open Data) | **High** (Local raw files present) |
| **NASA C-MAPSS FD004** | NASA Ames PCoE (Saxena et al., 2008) | Aircraft Turbofan | Space-delimited `.txt` | 248 Train / 249 Test | 21 sensors + 3 op settings | 6 Operating Regimes | 2 (HPC + Fan Degradation) | Available (`RUL_FD004.txt`) | Public Domain (NASA Open Data) | **High** (Local raw files present) |
| **NASA N-CMAPSS** | NASA Ames PCoE / ETH Zurich (Chao et al., 2021) | Commercial Turbofan | HDF5 (`.h5`) | 9 to 15 per subset (`DS01`–`DS08`) | 14 physical + virtual + health ($\theta$) | Realistic Flight Profiles (Climb/Cruise/Descent) | Multi-component Flow & Efficiency ($F, E$) | Available (per flight cycle) | Public Research License (NASA/PHM Society) | **Medium** (HDF5 parsing required) |
| **PHM 2008 Challenge** | PHM Society (2008) | Turbofan Engine | Space-delimited `.txt` | 218 Engines (Fleet total) | 21 sensors + 3 op settings | 6 Operating Regimes | Multi-fault | Proprietary competition test set | Historical Challenge Archive | **Identified as C-MAPSS Precursor** |
| **IMS Bearing Dataset** | Univ. of Cincinnati / NASA PCoE (Qiu et al., 2006) | Rotating Machinery | ASCII CSV (Vibration) | 4 Bearings per test (3 tests) | 4–8 Accelerometers (20 kHz) | Constant Speed (2000 rpm) | Bearing Outer/Inner Race / Roller Failure | End-of-life run to failure | Public Domain | **Partial** (Transfer Anomaly Detection only) |
| **FEMTO-ST PRONOSTIA** | AS2M / FEMTO-ST / IEEE PHM 2012 | Rolling Element Bearings | CSV (Vibration + Temp) | 17 Bearings (6 train / 11 test) | 2 Accelerometers (25.6 kHz) + RTD Temp | 3 Load / Speed Operating Regimes | Accelerated Race Wear | Available | Academic Open Access | **Partial** (Transfer Anomaly Detection only) |

---

## 3. DEEP RESEARCH ANALYSIS OF CANDIDATE DATASETS

### 3.1 NASA N-CMAPSS (Next Generation C-MAPSS)
- **Official Source:** [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) | [Data in Brief 2021 Article](https://doi.org/10.1016/j.dib.2021.107050)
- **Citation:** M. Arias Chao, C. S. Kulkarni, K. Goebel, and O. Fink, *"Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics,"* Data in Brief, vol. 35, p. 107050, 2021.
- **Physical Characteristics:**
  - Simulates a 90,000 lbf thrust commercial turbofan engine under complete, realistic flight profiles: Take-off, Climb, Cruise, and Descent.
  - Subsets (`DS01`–`DS08`) contain individual engine flight histories spanning up to 10 million data points per engine sampled at 1 Hz.
- **Variables Provided:**
  1. **Scenario Descriptors ($W$):** Flight altitude ($h$), Mach number ($M$), Throttle Resolver Angle ($TRA$), and Ambient Temperature ($T_{\text{amb}}$).
  2. **Sensor Measurements ($X_s$):** 14 onboard physical measurements including Fan inlet total temp ($T_2$), LPC outlet temp ($T_{24}$), HPC outlet temp ($T_{30}$), HPT outlet temp ($T_{48}$), pressures ($P_2, P_{15}, P_{21}, P_{30}, P_{50}$), speeds ($N_f, N_c$), and fuel flow ($W_f$).
  3. **Virtual Sensors ($X_v$):** Unmeasured performance estimates computed by the engine control unit.
  4. **Health Parameters ($\theta$):** Unobservable degradation states affecting efficiency ($E$) and flow capacity ($F$) across Fan, LPC, HPC, HPT, and LPT.
- **Integration Assessment:**
  - Format: Large HDF5 (`.h5`) files (several gigabytes per subset).
  - Verdict: The gold standard for modern flight-regime prognostics. Due to multi-gigabyte footprints, our platform implements a dedicated `NCMAPSSAdapter` capable of parsing `.h5` files with subsampling to prevent host memory exhaustion.

### 3.2 PHM 2008 Prognostics Challenge Dataset
- **Official Source:** International Conference on Prognostics and Health Management 2008 Data Challenge.
- **Provenance Investigation:**
  - Our investigation confirms that the PHM 2008 challenge dataset is the **direct historical origin of the C-MAPSS dataset**.
  - In 2008, Saxena and Goebel generated 218 engine trajectories. After the competition concluded, NASA PCoE cleaned, standardized, and partitioned this exact simulation codebase into the four canonical subsets: **FD001, FD002, FD003, and FD004**.
  - Therefore, PHM 2008 is **not an independent physical dataset**, but rather the uncurated predecessor to C-MAPSS.
  - Scientific Recommendation: Standardize on C-MAPSS FD001–FD004 rather than legacy PHM08 to avoid data duplication.

### 3.3 NASA C-MAPSS Subsets FD002, FD003, and FD004
- **Official Source:** NASA Ames Research Center Prognostics Data Repository.
- **Current Local Status:** Raw files are **already present** in `dataset/raw/CMAPSSData/`:
  - `train_FD002.txt` (9.08 MB, 260 engines), `test_FD002.txt` (5.73 MB, 259 engines), `RUL_FD002.txt` (259 labels)
  - `train_FD003.txt` (4.21 MB, 100 engines), `test_FD003.txt` (2.83 MB, 100 engines), `RUL_FD003.txt` (100 labels)
  - `train_FD004.txt` (10.35 MB, 248 engines), `test_FD004.txt` (6.96 MB, 249 engines), `RUL_FD004.txt` (249 labels)
- **Domain Complexity & Regime Breakdown:**
  - **FD001:** 1 Condition (Sea Level), 1 Fault Mode (HPC Degradation) — Single-regime baseline.
  - **FD002:** 6 Operating Regimes, 1 Fault Mode (HPC Degradation) — Requires regime-aware normalization.
  - **FD003:** 1 Condition (Sea Level), 2 Fault Modes (HPC + Fan Degradation) — Tests multi-component degradation.
  - **FD004:** 6 Operating Regimes, 2 Fault Modes (HPC + Fan Degradation) — Most complex C-MAPSS challenge.

---

## 4. FEATURE HARMONIZATION: C-MAPSS SENSOR TELEMETRY METADATA

The table below establishes the canonical physical metadata and engineering units for all 21 sensor channels across the C-MAPSS turbofan family:

| Sensor ID | Symbol | Physical Measurement Description | Units | Nominal Sea Level Baseline | Degradation Role |
|---|---|---|---|---|---|
| `sensor_1` | $T_2$ | Total temperature at fan inlet | °R | 518.67 | Invariant in FD001 (ambient boundary condition) |
| `sensor_2` | $T_{24}$ | Total temperature at LPC outlet | °R | ~642.5 | LPC discharge thermal tracking |
| `sensor_3` | $T_{30}$ | Total temperature at HPC outlet | °R | ~1585.0 | HPC thermal efficiency degradation indicator |
| `sensor_4` | $T_{50}$ | Total temperature at LPT outlet | °R | ~1405.0 | Exhaust gas temperature (EGT) margin erosion |
| `sensor_5` | $P_2$ | Total pressure at fan inlet | psia | 14.696 | Invariant in FD001 (ambient boundary condition) |
| `sensor_6` | $P_{15}$ | Total pressure in bypass-duct | psia | ~21.6 | Bypass duct airflow pressure |
| `sensor_7` | $P_{30}$ | Total pressure at HPC outlet | psia | ~553.0 | High-pressure compressor pressure ratio |
| `sensor_8` | $N_f$ | Physical fan spool speed | rpm | ~2388.0 | Low-pressure rotor mechanical RPM |
| `sensor_9` | $N_c$ | Physical core spool speed | rpm | ~9050.0 | High-pressure rotor mechanical RPM |
| `sensor_10` | $epr$ | Engine pressure ratio ($P_{50} / P_2$) | — | 1.00 | Pressure expansion ratio (Constant in FD001) |
| `sensor_11` | $Ps_{30}$ | Static pressure at HPC outlet | psia | ~47.2 | Combustor inlet static pressure |
| `sensor_12` | $\Phi$ | Ratio of fuel flow to $Ps_{30}$ | pps/psia | ~521.5 | Combustor fuel scheduling ratio |
| `sensor_13` | $NR_f$ | Corrected fan speed | rpm | ~2388.0 | Corrected low-pressure spool speed |
| `sensor_14` | $NR_c$ | Corrected core speed | rpm | ~8130.0 | Corrected high-pressure spool speed |
| `sensor_15` | $BPR$ | Bypass ratio | — | ~8.45 | Ratio of bypass airflow to core airflow |
| `sensor_16` | $farB$ | Burner fuel-air ratio | — | 0.03 | Invariant fuel metering ratio |
| `sensor_17` | $htBleed$ | Bleed enthalpy | BTU/lbm | ~392.0 | High-pressure turbine cooling bleed enthalpy |
| `sensor_18` | $Nf_{\text{dmd}}$ | Demanded fan speed | rpm | 2388.0 | Control loop setpoint (Invariant in FD001) |
| `sensor_19` | $PCNfR_{\text{dmd}}$ | Demanded corrected fan speed | rpm | 100.0 | Control loop setpoint (Invariant in FD001) |
| `sensor_20` | $W_{31}$ | High-pressure turbine coolant bleed | lbm/s | ~38.8 | Turbine blade cooling mass flow |
| `sensor_21` | $W_{32}$ | Low-pressure turbine coolant bleed | lbm/s | ~23.3 | LPT cooling mass flow |
| `op_setting_1` | $Alt$ | Flight altitude | ft | 0 to 42,000 | Operational flight regime |
| `op_setting_2` | $Mach$ | Flight Mach number | Mach | 0.0 to 0.84 | Aircraft aerodynamic velocity |
| `op_setting_3` | $TRA$ | Throttle Resolver Angle | % | 0 to 100 | Pilot power lever angle demand |

---

## 5. ARCHITECTURAL STRATEGY FOR MULTI-DATASET INTEGRATION

1. **Dataset Registry (`src/dataset_registry.py`)**: A centralized singleton registry maintaining formal metadata, file paths, engine counts, and supported capabilities for every dataset.
2. **Abstract Dataset Adapter (`src/dataset_adapters/base_adapter.py`)**: Defines standard interface for loading, trajectory extraction, and feature extraction.
3. **C-MAPSS Adapter (`src/dataset_adapters/cmapss_adapter.py`)**: Unified adapter handling FD001, FD002, FD003, and FD004, including regime clustering and condition normalization.
4. **Data Integrity Validator (`src/dataset_validator.py`)**: Verifies no duplicate engine IDs, cycle strictly monotonic, no NaNs/infinities, and no scaler leakage.
5. **Dynamic Dashboard Binding**: Streamlit UI components dynamically re-query the `DatasetRegistry` whenever the user selects a dataset, ensuring sensor dropdowns, engine counts, and compatible models automatically refresh.
