# JETWISE AI: MULTI-DATASET INTEGRATION & DOMAIN GENERALIZATION REPORT

**Document ID:** JW-DIR-2026-V3  
**Status:** COMPLETED & VERIFIED  
**Author:** JETWISE AI Architecture & ML Engineering Team  
**Repository:** `https://github.com/sam-black007/Jet-Engine-Failure-Predictor`  
**Target Environment:** NASA C-MAPSS (FD001–FD004), NASA N-CMAPSS (Next Gen Fleet), NASA PHM 2008  

---

## 1. Executive Summary

JETWISE AI has been expanded from a single-dataset prototype into an industrial-grade, multi-dataset predictive maintenance and prognostics platform. 

This report documents the architectural design, implementation, and rigorous experimental validation of the dynamic dataset subsystem, encompassing:
1. **Unified Dataset Adapters (`src/dataset_adapters/`):** Standardized ingestion layer supporting NASA C-MAPSS subsets (FD001, FD002, FD003, FD004) and NASA N-CMAPSS.
2. **Central Dataset Registry (`src/dataset_registry.py`):** Unified catalog, caching, metadata discovery, and automated model-to-dataset compatibility auditing.
3. **Automated Data Quality & Integrity Validator (`src/dataset_validator.py`):** Strict verification of cycle monotonicity, zero NaN/infinities, variance thresholds, and test-target alignment without data leakage.
4. **Historical Telemetry Replay Simulation Engine (`src/data_replay.py`):** Real-time cycle-by-cycle playback, edge-padded sliding window generation, prognostic risk classification (`HEALTHY`, `MONITOR`, `WARNING`, `CRITICAL`), and alert triggering.
5. **Empirical Cross-Dataset Generalization Benchmark (`src/cross_dataset_experiment.py`):** Zero-shot transfer evaluation of models trained on FD001 across compound faults (FD003) and multi-regime flight conditions (FD002, FD004), establishing exact empirical domain shift quantification.

---

## 2. Multi-Dataset Architecture & Ingestion Layer

```
                        ┌────────────────────────────────────────┐
                        │          DatasetRegistry               │
                        │ (Caching, Compatibility, Validation)   │
                        └──────────────────┬─────────────────────┘
                                           │
                ┌──────────────────────────┴──────────────────────────┐
                │                                                     │
   ┌────────────▼────────────┐                           ┌────────────▼────────────┐
   │      CMAPSSAdapter      │                           │      NCMAPSSAdapter     │
   ├─────────────────────────┤                           ├─────────────────────────┤
   │ Subsets: FD001 - FD004  │                           │ High-Fidelity Turbofan  │
   │ 3 Settings + 21 Sensors │                           │ 4 Settings + 14 Sensors │
   │ Regime Clustering       │                           │ Flight Phases (TO/CL/CR)│
   │ Variance Identification │                           │ Continuous Degradation  │
   └────────────┬────────────┘                           └────────────┬────────────┘
                │                                                     │
                └──────────────────────────┬──────────────────────────┘
                                           │
                        ┌──────────────────▼─────────────────────┐
                        │        EngineReplaySimulator           │
                        │ (Historical Playback & Telemetry Stream│
                        └──────────────────┬─────────────────────┘
                                           │
                        ┌──────────────────▼─────────────────────┐
                        │      Streamlit JETWISE AI Dashboard    │
                        │ (app.py & 6_Multi_Dataset_Research.py) │
                        └────────────────────────────────────────┘
```

### 2.1 Abstract Adapter Contract (`BaseDatasetAdapter`)

All dataset adapters inherit from `BaseDatasetAdapter` in `src/dataset_adapters/base_adapter.py`, enforcing an immutable contract:
- `load_train(reload=False) -> pd.DataFrame`
- `load_test(reload=False) -> pd.DataFrame`
- `load_rul(reload=False) -> pd.Series`
- `get_metadata() -> Dict[str, Any]`
- `get_engine_ids(kind="train") -> List[int]`
- `get_engine_trajectory(engine_id: int, kind="test") -> pd.DataFrame`
- `get_feature_names() -> List[str]`
- `get_active_sensors() -> List[str]`
- `get_op_settings() -> List[str]`
- `validate() -> Dict[str, Any]`

### 2.2 Supported Datasets Specification Matrix

| Dataset Identifier | Physical Machinery | Operating Flight Conditions | Fault Modes Present | Fleet Size (Train/Test) | Sampling Nature |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FD001** | Turbofan Core | 1 (Sea Level Static) | 1 (HPC Blade Degradation) | 100 / 100 Engines | 1 snapshot / cycle |
| **FD002** | Turbofan Core | 6 (0–42k ft, Mach 0–0.84, TRA 0–100%) | 1 (HPC Blade Degradation) | 260 / 259 Engines | 1 snapshot / cycle |
| **FD003** | Turbofan Core | 1 (Sea Level Static) | 2 (HPC + Fan Assembly Degradation) | 100 / 100 Engines | 1 snapshot / cycle |
| **FD004** | Turbofan Core | 6 (0–42k ft, Mach 0–0.84, TRA 0–100%) | 2 (HPC + Fan Assembly Degradation) | 248 / 249 Engines | 1 snapshot / cycle |
| **N-CMAPSS** | 90,000 lbf Turbofan | Continuous Full Flight Profiles | 5 (Fan, LPC, HPC, HPT, LPT Efficiency/Flow) | 10 / 8 Sub-Fleets | Continuous 1 Hz Profile |

---

## 3. Physical Sensor Telemetry Schema & Ontology

### 3.1 C-MAPSS Physical Sensor Mapping (`src/dataset_metadata.py`)

| Sensor Channel | Thermodynamic Symbol | Engineering Parameter Description | Nominal Standard Value | Physical Units | FD001 Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `sensor_1` | $T_2$ | Total Temperature at Fan Inlet | 518.67 | $^\circ\text{R}$ | Invariant (Constant) |
| `sensor_2` | $T_{24}$ | Total Temperature at LPC Outlet | 642.50 | $^\circ\text{R}$ | **Active Core Indicator** |
| `sensor_3` | $T_{30}$ | Total Temperature at HPC Outlet | 1585.00 | $^\circ\text{R}$ | **Active Core Indicator** |
| `sensor_4` | $T_{50}$ | Total Temperature at LPT Outlet (EGT) | 1405.00 | $^\circ\text{R}$ | **Primary Thermal Indicator** |
| `sensor_5` | $P_2$ | Total Pressure at Fan Inlet | 14.70 | $\text{psia}$ | Invariant in FD001 |
| `sensor_6` | $P_{15}$ | Total Pressure in Bypass Duct | 21.60 | $\text{psia}$ | Active Indicator |
| `sensor_7` | $P_{30}$ | Total Pressure at HPC Outlet (Burner) | 553.00 | $\text{psia}$ | **Active Core Indicator** |
| `sensor_8` | $N_f$ | Physical Fan Spool Speed | 2388.00 | $\text{rpm}$ | **Active Dynamic Indicator** |
| `sensor_9` | $N_c$ | Physical Core Spool Speed | 9050.00 | $\text{rpm}$ | **Active Dynamic Indicator** |
| `sensor_10` | $\text{epr}$ | Engine Pressure Ratio ($P_{50} / P_2$) | 1.00 | — | Invariant in FD001 |
| `sensor_11` | $\text{Ps}_{30}$ | Static Pressure at HPC Outlet | 47.20 | $\text{psia}$ | **Active Core Indicator** |
| `sensor_12` | $\Phi$ | Fuel Flow Ratio to $\text{Ps}_{30}$ | 521.50 | $\text{pps/psia}$ | **Active Fuel Governor Indicator** |
| `sensor_13` | $\text{NR}_f$ | Corrected Fan Spool Speed | 2388.00 | $\text{rpm}$ | **Active Dynamic Indicator** |
| `sensor_14` | $\text{NR}_c$ | Corrected Core Spool Speed | 8130.00 | $\text{rpm}$ | **Active Dynamic Indicator** |
| `sensor_15` | $\text{BPR}$ | Bypass Ratio | 8.45 | — | **Active Bypass Indicator** |
| `sensor_16` | $\text{farB}$ | Burner Fuel-Air Ratio | 0.03 | — | Invariant in FD001 |
| `sensor_17` | $\text{htBleed}$ | Bleed Enthalpy | 392.00 | $\text{BTU/lbm}$ | **Active Bleed Indicator** |
| `sensor_18` | $N_{f\_dmd}$ | Demanded Fan Speed | 2388.00 | $\text{rpm}$ | Invariant (Control Setpoint) |
| `sensor_19` | $\text{PCNfR}_{dmd}$ | Demanded Corrected Fan Speed | 100.00 | $\text{rpm}$ | Invariant (Control Setpoint) |
| `sensor_20` | $W_{31}$ | HPT Coolant Bleed Mass Flow | 38.80 | $\text{lbm/s}$ | **Active Cooling Indicator** |
| `sensor_21` | $W_{32}$ | LPT Coolant Bleed Mass Flow | 23.30 | $\text{lbm/s}$ | **Active Cooling Indicator** |

---

## 4. Empirical Cross-Dataset Generalization & Domain Shift Benchmark

To evaluate the generalization boundaries of deep prognostic architectures, a Deep LSTM network trained strictly on **FD001** (Sea level, single HPC fault) was evaluated zero-shot across **FD003**, **FD002**, and **FD004** test fleets.

### 4.1 Measured Benchmark Results

All figures below are actual, measured values executed via `src/cross_dataset_experiment.py` and serialized to `results/cross_dataset_evaluation.json`.

| Target Dataset | Test Fleet Size | Flight Conditions | Fault Modes | MAE Raw (Cycles) | MAE Clipped (Cycles) | RMSE (Cycles) | Official NASA Score | $\Delta$ MAE vs FD001 | Relative Degradation (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FD001** (In-Domain Baseline) | 100 | 1 | 1 (HPC) | **11.6931** | **10.6231** | **16.3288** | **537.01** | *Baseline* | *0.0%* |
| **FD003** (Compound Fault Shift) | 100 | 1 | 2 (HPC + Fan) | **26.8253** | **25.2653** | **37.2977** | **92,746.94** | **+15.1322** | **+129.41%** |
| **FD002** (Multi-Regime Shift) | 259 | 6 | 1 (HPC) | **44.1860** | **36.6918** | **55.4104** | **382,109.54** | **+32.4929** | **+277.88%** |
| **FD004** (Compound Multi-Regime) | 249 | 6 | 2 (HPC + Fan) | **53.1337** | **44.4401** | **65.7647** | **4,828,844.59** | **+41.4406** | **+354.40%** |

### 4.2 Engineering Analysis of Domain Shift Phenomena

1. **Compound Fault Shift (FD001 $\rightarrow$ FD003):**
   - **Performance Drop:** MAE increases by $+129.41\%$; NASA asymmetric penalty explodes from $537.01$ to $92,746.94$ ($+17,171\%$).
   - **Thermodynamic Cause:** In FD001, the network learned that degradation is exclusively heralded by HPC outlet temperature ($T_{30}$) and burner pressure drop ($P_{30}$). When presented with Fan assembly degradation in FD003, the early degradation indicators reside in Fan spool dynamics ($N_f, \text{NR}_f, P_{15}$) while HPC temperatures remain nominally stable. The FD001-trained model is blind to Fan mechanical fatigue, leading to severely delayed failure predictions that trigger massive exponential penalties under the NASA scoring function ($e^{d/10}-1$).

2. **Operating Flight Regime Shift (FD001 $\rightarrow$ FD002):**
   - **Performance Drop:** MAE increases by $+277.88\%$ ($11.69 \rightarrow 44.19$ cycles).
   - **Aerodynamic Cause:** In FD002, aircraft cruise across 6 operating conditions (altitudes up to 42,000 ft, Mach up to 0.84). At high altitudes, ambient air density drops, causing raw temperature ($T_{24}, T_{30}$) and static pressure ($\text{Ps}_{30}$) to decrease by orders of magnitude compared to sea-level static operations. Because the baseline model normalizes inputs with sea-level fixed means and variances, operational altitude variations are misinterpreted as catastrophic degradation or extreme health, shattering prediction accuracy.

3. **Compound Complexity Shift (FD001 $\rightarrow$ FD004):**
   - **Performance Drop:** MAE degrades by $+354.40\%$ ($11.69 \rightarrow 53.13$ cycles); NASA score reaches $4.83 \times 10^6$.
   - **Architectural Implication:** Confirms that for real-world multi-mission aircraft, models must either incorporate flight regime normalization (clustering operational settings or using condition-aware normalization layers) or utilize multi-task adaptive neural networks.

---

## 5. Historical Replay Simulation Engine (`src/data_replay.py`)

### 5.1 Architecture & Capabilities
The `EngineReplaySimulator` operates as a cycle-by-cycle virtual flight simulator:
- **Zero-Leakage Windowing:** At playback cycle $t$, extracts strictly historical cycles $[t-W+1, t]$. For cycles $t < W$, front edge padding replicates the initial flight condition without lookahead leakage.
- **Dynamic Risk Categorization:**
  - $\text{RUL} > 50$ cycles: `HEALTHY` (Green `#22c55e`)
  - $25 < \text{RUL} \le 50$ cycles: `MONITOR` (Yellow `#eab308`)
  - $10 < \text{RUL} \le 25$ cycles: `WARNING` (Orange `#f97316`)
  - $\text{RUL} \le 10$ cycles: `CRITICAL` (Red `#ef4444`)
- **Simulated Pacing Controls:** Supports interactive scrubber slider and step-by-step cycle progression with residual error tracking against ground truth.

---

## 6. Verification and Automated Testing

All new adapters, validators, and replay systems have been verified through automated unit tests (`tests/test_dataset_adapters.py`, `tests/test_dataset_validator.py`, `tests/test_dataset_registry.py`, `tests/test_data_replay.py`), achieving 100% pass rate across 79 tests in the repository suite.
