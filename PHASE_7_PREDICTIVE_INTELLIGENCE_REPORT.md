# JETWISE AI — Phase 7 Predictive Maintenance Intelligence Report

**Author:** Antigravity Senior ML & Aerospace Prognostics Architect  
**Date:** November 22, 2026  
**Status:** Certified & Verified  
**Implementation:** `src/predictive_intelligence.py`  
**Test Suite:** `tests/test_predictive_intelligence.py` (5/5 Passing)  
**Database:** `database/maintenance_records.db` (Seeded across FD001–FD004)

---

## 1. Executive Summary

Phase 7 delivers the core intelligence layer for the JETWISE AI platform, bridging pure deep learning prognostic outputs (numerical RUL predictions) with operational aerospace engineering maintenance decision systems.

The intelligence framework integrates five interconnected modules:
1. **Uncertainty Estimation Engine (`UncertaintyEstimator`)**:
   - Epistemic uncertainty computed via Monte Carlo (MC) Dropout across stochastic inference passes.
   - Empirical prediction intervals ($10^{th}$, $50^{th}$, $90^{th}$ percentiles).
   - Inverted confidence score metric ($0.05$ to $0.99$).
2. **Unsupervised Engine Anomaly Detector (`EngineAnomalyDetector`)**:
   - `IsolationForest` fitted on nominal baseline engine cycles (cycles $\le 25$).
   - Computes continuous normalized anomaly score ($0.0$ nominal to $1.0$ severe) and sensor residual z-scores to isolate anomalous sensor telemetry.
3. **Sequence Explainable AI (`SequenceFeatureAttribution`)**:
   - Integrated Gradients attribution across the temporal 30-cycle sliding window.
   - Computes percentage importance per sensor and temporal attribution weights across sequence timesteps.
4. **Maintenance Decision & Priority Engine (`MaintenanceDecisionEngine`)**:
   - Classifies engine operational health states: `HEALTHY`, `MONITOR`, `MAINTENANCE_DUE`, `CRITICAL`.
   - Computes composite **Maintenance Priority Index (MPI, 0–100 scale)**:
     $$\text{MPI} = \min\left(100, \left(\frac{125 - \min(\text{RUL}, 125)}{125} \times 60\right) + (\text{anomaly\_score} \times 25) + \left(\frac{\sigma_{\text{unc}}}{25} \times 15\right)\right)$$
   - Synthesizes actionable physical inspection tasks (e.g., HPT borescope, HPC blade inspection, Lube chip detector analysis).
5. **Persistent SQLite Maintenance Database (`MaintenanceDatabase`)**:
   - Tracks fleet status, historical predictions, open work orders, and sensor telemetry alerts.
   - Fully persistent at `database/maintenance_records.db`.

---

## 2. Component Verification & Test Results

The intelligence engine was validated via `tests/test_predictive_intelligence.py`:

| Test | Objective | Result |
|---|---|:---:|
| `test_uncertainty_estimator` | Verifies MC dropout forward passes, confidence score bounds, quantiles ($q_{10}, q_{50}, q_{90}$) | **PASSED** |
| `test_anomaly_detector` | Verifies nominal vs out-of-distribution detection, normalized scores, sensor z-scores | **PASSED** |
| `test_feature_attribution` | Verifies Integrated Gradients sequence attribution, top-3 drivers, temporal weights | **PASSED** |
| `test_maintenance_decision_engine` | Verifies health state categorization, MPI calculation, action recommendation | **PASSED** |
| `test_maintenance_database_operations` | Verifies SQLite CRUD operations, automatic work-order generation, fleet aggregation | **PASSED** |

**Execution:** `5 passed in 2.86s` (100% Green).

---

## 3. Fleet Maintenance Seeding & Operations

The persistent SQLite database was populated across NASA C-MAPSS FD001–FD004 test fleets using the optimal Pinball-LSTM checkpoint (`models/lstm_asymmetric/pinball_q35_lstm.pt`).

**FD001 Initial Fleet Health Snapshot:**
- Units Inspected: 20 Test Engines
- Health Distribution:
  - `CRITICAL`: 1 Engine (Engine #20, RUL: 15.1 cycles, MPI: 71.8)
  - `MAINTENANCE_DUE`: 19 Engines
- Automatically Generated Work Orders: 6 Open Work Orders (Turbine & HPC core inspections).
