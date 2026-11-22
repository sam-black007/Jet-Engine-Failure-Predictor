# THRUSTLAB — PREDICTIVE MAINTENANCE INTELLIGENCE ENGINE
## IMPLEMENTATION & VERIFICATION REPORT

**Feature:** Mandatory Intelligence Feature 2 — Advanced Predictive Maintenance Intelligence Engine  
**Repository:** `Jet-Engine-Failure-Predictor`  
**Status:** FULLY IMPLEMENTED, INTEGRATED & VERIFIED  
**Authors:** THRUSTLAB Architecture & Engineering Team  
**Date:** November 22, 2026  

---

## 1. EXECUTIVE SUMMARY

The **Advanced Predictive Maintenance Intelligence Engine** bridges the gap between deep learning prognostic models and physical airline fleet maintenance operations.

Rather than providing isolated numerical RUL outputs, the engine integrates:
1. **Epistemic & Aleatoric Uncertainty Estimation** (Monte Carlo Dropout and quantile prediction intervals).
2. **Unsupervised Outlier & Anomaly Detection** (Isolation Forest on standardized sensor residuals).
3. **Temporal Explainable AI (XAI)** (sequence feature attributions).
4. **Configurable 5-Tier Risk Classification & Explainable Scoring** (0–100 Maintenance Priority Index).
5. **Evidence-Traceable Prescriptive Recommendations**.
6. **SQLite-Backed Maintenance Work Order Management & Alert Lifecycles**.
7. **Fleet-Wide Prioritization Matrix**.

```
+-----------------------------------------------------------------------------------------+
|                    THRUSTLAB PREDICTIVE MAINTENANCE PIPELINE                           |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|  [Predicted RUL]       [Uncertainty Std]      [Anomaly Score]   [Degradation]  [Sensor] |
|        |                      |                      |                 |           |    |
|        v                      v                      v                 v           v    |
|   (Weight 35%)           (Weight 15%)           (Weight 20%)      (Weight 15%) (Weight 15%) |
|        \                      |                      |                 /           /    |
|         +---------------------+----------------------+----------------+-----------+     |
|                                           |                                             |
|                                           v                                             |
|                     [Maintenance Priority Index (MPI: 0 - 100)]                         |
|                                           |                                             |
|                                           v                                             |
|                     [5-Tier Risk Classification Taxonomy]                               |
|                         (CRITICAL / HIGH_RISK / WARNING / MONITOR / HEALTHY)            |
|                                           |                                             |
|                     +---------------------+---------------------+                       |
|                     |                                           |                       |
|                     v                                           v                       |
|        [Evidence-Traceable Actions]                [Persistent SQLite Records]          |
|        - Subsystem Targeting                       - `work_orders` Table                |
|        - Concrete Sensor Evidence                  - `maintenance_alerts` Lifecycle     |
|        - Engineering Rationale                     - CSV Export & Filter Audit          |
+-----------------------------------------------------------------------------------------+
```

---

## 2. 5-TIER RISK CLASSIFICATION TAXONOMY

The engine enforces an aerospace-grade 5-tier risk taxonomy:

| Risk Tier | Priority | RUL Range (Cycles) | Urgency Window | Color Code | Operational Meaning & Required Action |
|:---|:---:|:---:|:---:|:---:|:---|
| **`CRITICAL`** | `URGENT` | $0.0 \le \text{RUL} < 12.0$ | $\le 2$ Days | `#FF5252` (Red) | Severe risk; immediate flight grounding and engine core shop visit required. Teardown inspection. |
| **`HIGH_RISK`** | `HIGH` | $12.0 \le \text{RUL} < 25.0$ | $\le 7$ Days | `#FF9800` (Orange) | Significant degradation detected; prioritize HPC/HPT borescope inspection within 7 days. |
| **`WARNING`** | `MEDIUM` | $25.0 \le \text{RUL} < 45.0$ | $\le 21$ Days | `#FFB547` (Amber) | Degradation initiated; schedule visual borescope and lube spectrometry at next transit layover. |
| **`MONITOR`** | `ROUTINE` | $45.0 \le \text{RUL} < 75.0$ | $\le 45$ Days | `#00B8D9` (Cyan) | Early parameter drift; increase telemetry trend logging frequency to 10 Hz; routine turnaround checks. |
| **`HEALTHY`** | `LOW` | $\ge 75.0$ | $\le 90$ Days | `#21C77A` (Green) | Nominal flight clearance; standard scheduled maintenance line maintenance. |

---

## 3. EXPLAINABLE MAINTENANCE PRIORITY INDEX (MPI) FORMULATION

The Maintenance Priority Index ($\text{MPI} \in [0, 100]$) combines five critical physical and prognostic factors with fully configurable weights:

$$\text{MPI} = w_{\text{rul}} S_{\text{rul}} + w_{\text{unc}} S_{\text{unc}} + w_{\text{anom}} S_{\text{anom}} + w_{\text{deg}} S_{\text{deg}} + w_{\text{sens}} S_{\text{sens}}$$

Where the default operational weights are:
- $w_{\text{rul}} = 0.35$ (35% RUL Depletion Impact)
- $w_{\text{unc}} = 0.15$ (15% Model Prediction Uncertainty / Dispersion)
- $w_{\text{anom}} = 0.20$ (20% Unsupervised Outlier Severity)
- $w_{\text{deg}} = 0.15$ (15% Estimated Degradation Velocity)
- $w_{\text{sens}} = 0.15$ (15% Extreme Sensor Z-Score Abnormality)

Every subscore is normalized to $[0, 100]$ and exposed directly in the dashboard UI and API responses for complete audit transparency:

```json
"score_breakdown": {
    "rul_subscore": 34.2,
    "uncertainty_subscore": 11.5,
    "anomaly_subscore": 16.0,
    "degradation_rate_subscore": 9.8,
    "sensor_abnormality_subscore": 12.1
}
```

---

## 4. EVIDENCE-TRACEABLE PRESCRIPTIVE RECOMMENDATIONS

Each generated maintenance action links actionable engineering directives with verifiable sensor telemetry:

1. **Engine Identification:** Specific unit number and fleet identifier.
2. **Risk Classification & Priority:** Standardized urgency tier.
3. **Target Subassembly:** Explicit subsystem (e.g., `HPT Section`, `Combustor Liner`, `Lube System Compartment 3/4`).
4. **Supporting Diagnostic Evidence:** Quantified measurements (e.g., `"Sensor outlier z-score reached 3.42 sigma with elevated EGT; anomaly score 0.82"`).
5. **Relevant Sensor & Prediction Info:** Exact dictionary containing numerical model outputs.
6. **Recommended Action:** Concrete MRO procedure (e.g., `"Borescope Inspection of HPT Stage 1 Blades & Shrouds"`).
7. **Engineering Rationale:** Underlying physical justification (e.g., `"Verify thermal barrier coating integrity and check for leading-edge creep"`).

---

## 5. SQLITE MAINTENANCE DATABASE SCHEMA

The persistent database is located at `database/maintenance_records.db` and manages four primary tables:

```sql
-- 1. FLEET ASSET REGISTRY
CREATE TABLE fleet_engines (
    engine_id INTEGER,
    dataset_subset TEXT,
    cycles_logged INTEGER,
    last_known_status TEXT,
    current_rul_estimate REAL,
    maintenance_priority_index REAL,
    updated_at TIMESTAMP,
    PRIMARY KEY (engine_id, dataset_subset)
);

-- 2. HISTORICAL PREDICTION & PROGNOSTIC AUDIT LOG
CREATE TABLE predictions_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    engine_id INTEGER,
    dataset_subset TEXT,
    cycle INTEGER,
    predicted_rul REAL,
    confidence_std REAL,
    anomaly_score REAL,
    health_state TEXT,
    maintenance_priority_index REAL,
    model_name TEXT,
    details_json TEXT
);

-- 3. INTERACTIVE WORK ORDERS (CRUD)
CREATE TABLE work_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    engine_id INTEGER,
    dataset_subset TEXT,
    priority TEXT,
    title TEXT,
    subsystem TEXT,
    description TEXT,
    status TEXT DEFAULT 'Pending',
    assigned_to TEXT,
    notes TEXT
);

-- 4. ACTIVE ALERT LIFECYCLE MANAGEMENT
CREATE TABLE maintenance_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    engine_id INTEGER,
    dataset_subset TEXT,
    alert_level TEXT,
    message TEXT,
    acknowledged INTEGER DEFAULT 0,
    acknowledged_at TIMESTAMP,
    resolved INTEGER DEFAULT 0,
    resolved_at TIMESTAMP,
    resolution_notes TEXT
);
```

### Work Order Status Workflow:
$$\text{Pending} \longrightarrow \text{Inspection Scheduled} \longrightarrow \text{Under Inspection} \longrightarrow \text{Maintenance In Progress} \longrightarrow \text{Completed} \quad (\text{or Deferred})$$

---

## 6. FLEET-WIDE PRIORITIZATION MATRIX

The dashboard Tab 7 Subtab 4 automatically aggregates and ranks all active fleet engines by their Maintenance Priority Index (MPI):
- Highlights highest-urgency units at the top of the queue.
- Computes fleet-wide counts: Critical, Warning, Monitor, and Healthy.
- Enables single-click work order dispatch and engineer assignment.
- Exports full work order logs to CSV (`THRUSTLAB_Work_Orders.csv`).

---

## 7. VERIFICATION & PLAYWRIGHT TEST RESULTS

The Predictive Maintenance Engine has undergone comprehensive automated verification:

### Unit & Integration Tests (`tests/test_predictive_intelligence.py`)
- `test_uncertainty_estimator`: **PASSED** (MC Dropout and interval generation verified).
- `test_anomaly_detector`: **PASSED** (Isolation Forest residual fit and thresholding validated).
- `test_feature_attribution`: **PASSED** (Sequence gradient/perturbation attributions validated).
- `test_five_tier_risk_classification`: **PASSED** (Boundaries and categories verified).
- `test_explainable_priority_breakdown`: **PASSED** (Mathematical weights and MPI verified).
- `test_traceable_recommendations`: **PASSED** (Evidence linkage and rationale confirmed).
- `test_database_work_order_crud_and_alerts`: **PASSED** (Full SQLite lifecycle verified).

### API REST Integration (`tests/test_api.py`)
- `test_predict_with_engine_id`: **PASSED** (Inference + health evaluation + SQLite logging).
- `test_work_orders_crud`: **PASSED** (Create, status update, notes logging via REST).

### Browser End-to-End Automation (`run_playwright_suite.py`)
- Recommendations Render: **PASSED** (captured to `results/playwright/predictive_maintenance_recommendations.png`).
- Work Orders Table & Dispatch: **PASSED** (captured to `results/playwright/maintenance_work_orders.png`).
- Alert Lifecycle Center: **PASSED** (captured to `results/playwright/maintenance_alerts_center.png`).
- Fleet Prioritization Matrix: **PASSED** (captured to `results/playwright/maintenance_fleet_matrix.png`).
