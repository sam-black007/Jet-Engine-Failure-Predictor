# THRUSTLAB — Standardized Scenario Library & Test Preset Guide

**Document Reference:** `THRUSTLAB-GUIDE-PHASE17-PRESETS-01`  
**Classification:** Aerospace Simulation Benchmark Standard & Preset Catalog  
**Date:** November 22, 2026  
**Auditor:** Principal Aerospace Prognostics Architect & Propulsion Test Group  
**Target Environment:** Hub 2: SIMULATE — Preset Test Library  

---

## 1. Executive Overview

To ensure rigorous repeatability across aerospace test campaigns, THRUSTLAB provides **14 standardized simulation templates** covering baseline operations, gas-path thermal stresses, component mechanical degradation, instrumentation failures, and environmental extremes.

Every template defines strict parameter boundaries, affected turbofan subassemblies, supported simulation modes, and aerothermal validation criteria.

---

## 2. Complete 14-Preset Master Catalog

### Preset 01: Healthy Engine Baseline (`PRESET_01_HEALTHY_BASELINE`)
- **Category:** Baseline Operation
- **Target Subassemblies:** All 10 Subassemblies (Nominal state)
- **Parameters:** `duration_cycles`: 100 cycles, `degradation_rate`: 0.0%/cycle
- **Supported Modes:** `MODE_A`, `MODE_B`, `MODE_C`, `MODE_D` (Brayton Fallback)
- **Expected Outputs:** Predicted RUL $> 120$ cycles; Health Score $> 85$; nominal inter-stage pressure ratios.
- **Validation Criteria:** $T_{48} < 1665^\circ\text{R}$; monotonic compression $P_2 < P_{24} < P_{30}$; mass flow continuity $dM \le 1.0\%$.

### Preset 02: Increased Fuel Flow (+5%) (`PRESET_02_INCREASED_FUEL_FLOW`)
- **Category:** Operational Tuning
- **Target Subassemblies:** Combustion Chamber, High-Pressure Turbine, Exhaust Nozzle
- **Parameters:** `fuel_flow_change_pct`: $+5.0\%$ (Range: $+1.0\%$ to $+15.0\%$), `duration_cycles`: 80 cycles
- **Supported Modes:** `MODE_B`, `MODE_C`, `MODE_D`
- **Expected Outputs:** Elevated $T_{48} / T_{50}$ ($+25$ to $+40^\circ\text{R}$); increased core rotor speed $N_2$; slight RUL penalty ($-3$ to $-6$ cycles).
- **Validation Criteria:** Combustor energy balance verified; fuel-air ratio within stoichiometric limits.

### Preset 03: Increased Turbine Temperature (+5%) (`PRESET_03_INCREASED_TURBINE_TEMP`)
- **Category:** Thermal Stress
- **Target Subassemblies:** High-Pressure Turbine, Low-Pressure Turbine, Exhaust Nozzle
- **Parameters:** `turbine_temp_change_pct`: $+5.0\%$ (Range: $+1.0\%$ to $+10.0\%$), `duration_cycles`: 100 cycles
- **Supported Modes:** `MODE_B`, `MODE_C`, `MODE_D`
- **Expected Outputs:** Accelerated blade thermal oxidation; RUL reduction ($-8$ to $-14$ cycles); elevated EGT.
- **Validation Criteria:** Verified against structural creep redline ($T_{48} < 1665^\circ\text{R}$); isentropic turbine efficiency $\eta_t > 0.70$.

### Preset 04: Compressor Efficiency Degradation (-5%) (`PRESET_04_COMPRESSOR_EFFICIENCY_DEGRADATION`)
- **Category:** Gas-Path Degradation
- **Target Subassemblies:** High-Pressure Compressor, Low-Pressure Compressor
- **Parameters:** `compressor_efficiency_change_pct`: $-5.0\%$ (Range: $-15.0\%$ to $-1.0\%$), `duration_cycles`: 120 cycles
- **Supported Modes:** `MODE_B`, `MODE_C`, `MODE_D`
- **Expected Outputs:** Reduced pressure ratio $P_{30}/P_{25}$; elevated compressor exit temperature $T_{30}$; reduced surge margin.
- **Validation Criteria:** Compressor stall margin $> 12\%$; strictly monotonic pressure rise across core.

### Preset 05: Turbine Degradation (-5%) (`PRESET_05_TURBINE_DEGRADATION`)
- **Category:** Gas-Path Degradation
- **Target Subassemblies:** High-Pressure Turbine, Central Dual-Spool Shaft
- **Parameters:** `turbine_efficiency_change_pct`: $-5.0\%$ (Range: $-12.0\%$ to $-1.0\%$), `duration_cycles`: 100 cycles
- **Supported Modes:** `MODE_B`, `MODE_C`, `MODE_D`
- **Expected Outputs:** Reduced work extraction per stage; higher residual EGT ($T_{50}$); core spool speed drop.
- **Validation Criteria:** Shaft torque balance satisfied; core spool rotational stability verified.

### Preset 06: Fan Blade Degradation (+15% Wear) (`PRESET_06_FAN_DEGRADATION`)
- **Category:** Mechanical Wear
- **Target Subassemblies:** Fan Assembly, Engine Nacelle & Casing
- **Parameters:** `fan_degradation_pct`: $+15.0\%$ (Range: $+2.0\%$ to $+25.0\%$), `duration_cycles`: 200 cycles
- **Supported Modes:** `MODE_B`, `MODE_C`, `MODE_D`
- **Expected Outputs:** Bypass pressure ratio decay; $N_1$ rotor speed fluctuations; moderate RUL decline ($-5$ to $-10$ cycles).
- **Validation Criteria:** Fan tip Mach number $< 1.45$; bypass flow continuity maintained.

### Preset 07: Pressure Sensor Bias (+3% on P30) (`PRESET_07_PRESSURE_SENSOR_BIAS`)
- **Category:** Sensor Fault
- **Target Subassemblies:** Sensors & Instrumentation Harness, High-Pressure Compressor
- **Parameters:** `sensor_target`: `sensor_8` ($P_{30}$), `bias_percentage`: $+3.0\%$ (Range: $-10.0\%$ to $+10.0\%$), `duration_cycles`: 60 cycles
- **Supported Modes:** `MODE_A`, `MODE_B`, `MODE_D`
- **Expected Outputs:** Sensor FDIR isolates single-channel bias; anomaly flag raised; mechanical health preserved.
- **Validation Criteria:** FDIR parity equations distinguish transducer calibration error from core aerothermal collapse.

### Preset 08: Temperature Sensor Bias (+15 °R on T50) (`PRESET_08_TEMPERATURE_SENSOR_BIAS`)
- **Category:** Sensor Fault
- **Target Subassemblies:** Sensors & Instrumentation Harness, Low-Pressure Turbine
- **Parameters:** `sensor_target`: `sensor_4` ($T_{50}$), `bias_absolute`: $+15.0^\circ\text{R}$ (Range: $-50.0$ to $+50.0^\circ\text{R}$), `duration_cycles`: 80 cycles
- **Supported Modes:** `MODE_A`, `MODE_B`, `MODE_D`
- **Expected Outputs:** Apparent EGT rise flagged as instrumentation bias rather than turbine hot-section distress.
- **Validation Criteria:** Analytical redundancy between $T_{48}$ and $P_{50}$ exposes thermocouple drift.

### Preset 09: Intermittent Sensor Dropout (T24) (`PRESET_09_SENSOR_DROPOUT`)
- **Category:** Sensor Fault
- **Target Subassemblies:** Sensors & Instrumentation Harness, Low-Pressure Compressor
- **Parameters:** `sensor_target`: `sensor_2` ($T_{24}$), `dropout_probability`: $0.25$ (Range: $0.05$ to $0.60$), `duration_cycles`: 50 cycles
- **Supported Modes:** `MODE_A`, `MODE_B`
- **Expected Outputs:** Real-time signal loss alert; virtual sensor neural synthesis activated to prevent NaN propagation.
- **Validation Criteria:** Virtual neural reconstruction fills missing time-series frames with $< 2.0\%$ error.

### Preset 10: Gradual Sensor Drift (+0.1%/Cycle on P2) (`PRESET_10_GRADUAL_SENSOR_DRIFT`)
- **Category:** Sensor Fault
- **Target Subassemblies:** Sensors & Instrumentation Harness, Fan Assembly
- **Parameters:** `sensor_target`: `sensor_1` ($P_2$), `drift_rate_per_cycle`: $+0.1\%/\text{cycle}$, `duration_cycles`: 100 cycles
- **Supported Modes:** `MODE_A`, `MODE_B`
- **Expected Outputs:** Cumulative drift crosses $3\sigma$ statistical threshold at cycle 30, triggering calibration advisory.
- **Validation Criteria:** Cumulative Sum (CUSUM) residual detector triggers alert prior to cycle 40.

### Preset 11: Multiple Correlated Sensor Faults (`PRESET_11_MULTIPLE_SENSOR_FAULTS`)
- **Category:** Sensor Fault
- **Target Subassemblies:** Sensors & Instrumentation Harness, Combustion Chamber
- **Parameters:** `sensor_1`: `sensor_8` ($+3.0\%$ bias), `sensor_2`: `sensor_4` ($+0.15\%/\text{cycle}$ drift), `duration_cycles`: 80 cycles
- **Supported Modes:** `MODE_A`, `MODE_B`
- **Expected Outputs:** Dual-channel sensor FDIR isolation without false-positive engine teardown alert.
- **Validation Criteria:** Cross-channel correlation matrix isolates both faulty transducers.

### Preset 12: Hot-Day Ambient Takeoff (+15 °C) (`PRESET_12_HIGH_AMBIENT_TEMPERATURE`)
- **Category:** Environmental Extreme
- **Target Subassemblies:** Fan Assembly, High-Pressure Compressor, High-Pressure Turbine
- **Parameters:** `ambient_temp_delta_c`: $+15.0^\circ\text{C}$ (ISA+15), `duration_cycles`: 40 cycles
- **Supported Modes:** `MODE_B`, `MODE_C`, `MODE_D`
- **Expected Outputs:** Increased corrected temperatures across all stations; elevated baseline EGT; thermal throttling.
- **Validation Criteria:** Inlet total temperature $T_2$ satisfies aerodynamic ambient recovery formula.

### Preset 13: High-Altitude Cruise Transfer (35,000 ft) (`PRESET_13_OPERATING_CONDITION_CHANGES`)
- **Category:** Operational Tuning
- **Target Subassemblies:** Fan Assembly, Combustion Chamber, Exhaust Nozzle
- **Parameters:** `altitude_feet`: $35,000\text{ ft}$, `mach_number`: $0.82\text{ Mach}$, `duration_cycles`: 60 cycles
- **Supported Modes:** `MODE_B`, `MODE_C`, `MODE_D`
- **Expected Outputs:** Lower total pressure across all stations; normalized non-dimensional mass flow maintained.
- **Validation Criteria:** Complies with Standard Atmosphere (ISA) pressure ratio $\delta_{amb}$ and temperature ratio $\theta_{amb}$.

### Preset 14: Combined High-Time Engine Degradation (`PRESET_14_COMBINED_ENGINE_DEGRADATION`)
- **Category:** Combined Stress
- **Target Subassemblies:** Fan Assembly, High-Pressure Compressor, High-Pressure Turbine, Exhaust Nozzle
- **Parameters:** HPC eff: $-4.0\%$, HPT eff: $-3.0\%$, Fan wear: $+8.0\%$, `duration_cycles`: 150 cycles
- **Supported Modes:** `MODE_B`, `MODE_C`, `MODE_D`
- **Expected Outputs:** Severe RUL reduction ($-25$ to $-45$ cycles); Risk Score $> 65$; critical dispatch advisory.
- **Validation Criteria:** All thermodynamic stations remain within non-choking gas flow bounds.

---

## 3. Preset Management Workflows

Operators can manage presets through four core workflows:
1. **1-Click Execution:** Select preset card and click **🚀 Run Preset** to execute immediately.
2. **Interactive Tuning:** Modify sliders to adjust fuel flow, thermal deltas, or cycles within pre-validated boundary clamps.
3. **Save Custom Preset:** Click **💾 Save Custom Preset** to persist modified parameters into local SQLite/JSON storage for future campaigns.
4. **Compare Scenarios:** Select multiple presets or custom runs and click **⚖️ Compare Selected** to generate side-by-side RUL and aerothermal delta matrices.
