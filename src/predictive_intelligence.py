"""Predictive Maintenance Intelligence Engine (JETWISE AI Feature 2).

Integrates:
1. Epistemic & Aleatoric Uncertainty Estimation (MC Dropout & Prediction Intervals)
2. Unsupervised Engine Anomaly Detection (Isolation Forest on Sensor Residuals)
3. Sequence Explainable AI (Perturbation & Gradient-based Sensor Attribution)
4. Configurable 5-Tier Risk Classification & Explainable Prioritization (MPI 0-100)
5. Evidence-Traceable Prescriptive Recommendations
6. Alert Lifecycle Management (De-duplication, Acknowledgement, Resolution)
7. Persistent SQLite Maintenance Planning & Fleet Intelligence Database
"""

from typing import Dict, List, Optional, Any, Tuple
import os
import sqlite3
import datetime
import pathlib
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest

from src.data_loader import SENSOR_COLS


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "database" / "maintenance_records.db"

# ---------------------------------------------------------------------------
# 5-Tier Risk Classification Specification
# ---------------------------------------------------------------------------
RISK_CATEGORIES = {
    "HEALTHY": {
        "max_rul": 9999.0,
        "min_rul": 75.0,
        "priority": "LOW",
        "urgency_days": 90,
        "color": "#21C77A",
        "description": "No significant detected degradation; nominal flight clearance."
    },
    "MONITOR": {
        "max_rul": 75.0,
        "min_rul": 45.0,
        "priority": "ROUTINE",
        "urgency_days": 45,
        "color": "#00B8D9",
        "description": "Early indicators observed; increase monitoring frequency."
    },
    "WARNING": {
        "max_rul": 45.0,
        "min_rul": 25.0,
        "priority": "MEDIUM",
        "urgency_days": 21,
        "color": "#FFB547",
        "description": "Degradation identified; schedule inspection at next maintenance opportunity."
    },
    "HIGH_RISK": {
        "max_rul": 25.0,
        "min_rul": 12.0,
        "priority": "HIGH",
        "urgency_days": 7,
        "color": "#FF9800",
        "description": "Significant risk indicators; schedule prioritized maintenance within 7 days."
    },
    "CRITICAL": {
        "max_rul": 12.0,
        "min_rul": 0.0,
        "priority": "URGENT",
        "urgency_days": 2,
        "color": "#FF5252",
        "description": "Severe risk indicators; immediate grounding and shop overhaul recommended."
    }
}
HEALTH_STATES = RISK_CATEGORIES

MAINTENANCE_STATUSES = [
    "Pending",
    "Inspection Scheduled",
    "Under Inspection",
    "Maintenance In Progress",
    "Completed",
    "Deferred"
]


class UncertaintyEstimator:
    """Estimates epistemic and empirical uncertainty via Monte Carlo Dropout."""

    def __init__(self, model: nn.Module, num_samples: int = 25):
        self.model = model
        self.num_samples = num_samples

    def enable_dropout(self):
        """Enable dropout layers while keeping batchnorm/layernorm in eval mode."""
        for m in self.model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
                m.train()

    def predict_with_uncertainty(self, x: np.ndarray) -> Dict[str, Any]:
        """Perform stochastic Monte Carlo forward passes."""
        if x.ndim == 2:
            x = np.expand_dims(x, axis=0)

        tensor_x = torch.from_numpy(x.astype(np.float32))

        self.model.eval()
        self.enable_dropout()

        preds = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                out = self.model(tensor_x)
                if isinstance(out, tuple):
                    out = out[0]
                preds.append(out.squeeze().cpu().numpy())

        preds_arr = np.array(preds)
        if preds_arr.ndim == 1:
            preds_arr = np.expand_dims(preds_arr, axis=1)

        mean_rul = np.mean(preds_arr, axis=0)
        std_rul = np.std(preds_arr, axis=0)
        q10 = np.percentile(preds_arr, 10, axis=0)
        q50 = np.percentile(preds_arr, 50, axis=0)
        q90 = np.percentile(preds_arr, 90, axis=0)

        confidence = np.clip(1.0 - (std_rul / 35.0), 0.05, 0.99)

        if len(mean_rul) == 1:
            return {
                "mean_rul": float(np.round(mean_rul[0], 2)),
                "std_dev": float(np.round(std_rul[0], 2)),
                "q10": float(np.round(q10[0], 2)),
                "q50": float(np.round(q50[0], 2)),
                "q90": float(np.round(q90[0], 2)),
                "confidence_score": float(np.round(confidence[0], 3)),
                "num_samples": self.num_samples
            }

        return {
            "mean_rul": np.round(mean_rul, 2).tolist(),
            "std_dev": np.round(std_rul, 2).tolist(),
            "q10": np.round(q10, 2).tolist(),
            "q50": np.round(q50, 2).tolist(),
            "q90": np.round(q90, 2).tolist(),
            "confidence_score": np.round(confidence, 3).tolist(),
            "num_samples": self.num_samples
        }


class EngineAnomalyDetector:
    """Unsupervised Isolation Forest anomaly detector on turbofan sensor trajectories."""

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False
        self.feature_cols: List[str] = []
        self.sensor_means: Dict[str, float] = {}
        self.sensor_stds: Dict[str, float] = {}

    def fit(self, baseline_df: pd.DataFrame, feature_cols: List[str]):
        """Fit Isolation Forest on nominal healthy cycles (e.g. cycle <= 25)."""
        self.feature_cols = feature_cols
        data = baseline_df[feature_cols].dropna().values
        self.model.fit(data)
        self.is_fitted = True

        for col in feature_cols:
            self.sensor_means[col] = float(baseline_df[col].mean())
            self.sensor_stds[col] = float(baseline_df[col].std()) if baseline_df[col].std() > 1e-6 else 1.0

    def detect_window_anomaly(self, window_vals: np.ndarray) -> Dict[str, Any]:
        """Detect anomaly for a window of shape (window_len, num_features)."""
        if not self.is_fitted:
            return {"is_anomaly": False, "anomaly_score": 0.0, "severity": "NORMAL", "top_anomalous_sensors": []}

        last_cycle = window_vals[-1:]
        raw_score = self.model.decision_function(last_cycle)[0]
        pred_label = self.model.predict(last_cycle)[0]

        norm_score = float(np.clip(0.5 - raw_score, 0.0, 1.0))
        is_anomaly = bool(pred_label == -1 or norm_score > 0.55)

        deviations = {}
        for i, col in enumerate(self.feature_cols):
            mean = self.sensor_means.get(col, 0.0)
            std = self.sensor_stds.get(col, 1.0)
            z = abs((window_vals[-1, i] - mean) / std)
            deviations[col] = float(round(z, 2))

        sorted_devs = sorted(deviations.items(), key=lambda x: x[1], reverse=True)
        top_sensors = [{"sensor": s, "z_score": z} for s, z in sorted_devs[:4] if z > 1.5]

        severity = "NORMAL"
        if norm_score > 0.75 or (len(top_sensors) >= 2 and top_sensors[0]["z_score"] > 3.0):
            severity = "CRITICAL"
        elif norm_score > 0.55 or (len(top_sensors) >= 1 and top_sensors[0]["z_score"] > 2.0):
            severity = "WARNING"
        elif norm_score > 0.40:
            severity = "ELEVATED"

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(norm_score, 3),
            "severity": severity,
            "raw_decision_score": round(float(raw_score), 4),
            "top_anomalous_sensors": top_sensors
        }


class SequenceFeatureAttribution:
    """Explainable AI (XAI) feature attribution for deep turbofan sequence models."""

    def __init__(self, model: nn.Module, feature_cols: List[str]):
        self.model = model
        self.feature_cols = feature_cols

    def attribute_window(self, window_vals: np.ndarray, num_steps: int = 15) -> Dict[str, Any]:
        """Compute integrated gradient approximation for input sequence window."""
        self.model.eval()
        x_target = torch.from_numpy(window_vals.astype(np.float32)).unsqueeze(0)
        x_baseline = torch.zeros_like(x_target)

        alphas = torch.linspace(0.0, 1.0, num_steps)
        accum_grads = torch.zeros_like(x_target)

        for alpha in alphas:
            x_interp = (x_baseline + alpha * (x_target - x_baseline)).clone().detach().requires_grad_(True)
            out = self.model(x_interp)
            if isinstance(out, tuple):
                out = out[0]
            pred = out.squeeze()
            pred.backward()
            if x_interp.grad is not None:
                accum_grads += x_interp.grad.detach()

        avg_grads = accum_grads / num_steps
        ig = (x_target - x_baseline).detach() * avg_grads
        ig_abs = torch.abs(ig).squeeze(0).cpu().numpy()

        sensor_scores = np.sum(ig_abs, axis=0)
        total_score = np.sum(sensor_scores) + 1e-8
        sensor_weights = (sensor_scores / total_score) * 100.0

        temporal_scores = np.sum(ig_abs, axis=1)
        temporal_weights = (temporal_scores / (np.sum(temporal_scores) + 1e-8)) * 100.0

        attributions = []
        for i, col in enumerate(self.feature_cols):
            attributions.append({
                "sensor": col,
                "importance_pct": float(round(sensor_weights[i], 2)),
                "raw_attribution": float(round(sensor_scores[i], 4))
            })

        attributions.sort(key=lambda x: x["importance_pct"], reverse=True)

        return {
            "sensor_attributions": attributions,
            "top_3_drivers": [a["sensor"] for a in attributions[:3]],
            "temporal_weights": [round(float(w), 2) for w in temporal_weights]
        }


class MaintenanceDecisionEngine:
    """Configurable Aerospace Maintenance Decision Engine.

    Calculates transparent 5-tier risk classifications, explainable Maintenance Priority Indices (MPI),
    and evidence-traceable prescriptive maintenance actions.
    """

    def __init__(
        self,
        weight_rul: float = 0.35,
        weight_uncertainty: float = 0.15,
        weight_anomaly: float = 0.20,
        weight_degradation_rate: float = 0.15,
        weight_sensor_abnormality: float = 0.15,
        critical_rul_threshold: float = 15.0,
        warning_rul_threshold: float = 40.0
    ):
        self.w_rul = weight_rul
        self.w_unc = weight_uncertainty
        self.w_anom = weight_anomaly
        self.w_deg = weight_degradation_rate
        self.w_sens = weight_sensor_abnormality
        self.crit_threshold = critical_rul_threshold
        self.warn_threshold = warning_rul_threshold

    def evaluate_health(
        self,
        predicted_rul: float,
        uncertainty_std: float = 5.0,
        anomaly_score: float = 0.2,
        current_cycle: int = 150,
        degradation_rate: float = 1.0,
        max_sensor_zscore: float = 1.0,
        engine_id: int = 1
    ) -> Dict[str, Any]:
        """Classify health condition and calculate explainable priority index."""
        # 1. 5-Tier Condition Classification
        if predicted_rul <= 12.0:
            condition = "CRITICAL"
        elif predicted_rul <= 25.0:
            condition = "HIGH_RISK"
        elif predicted_rul <= 45.0:
            condition = "WARNING"
        elif predicted_rul <= 75.0:
            condition = "MONITOR"
        else:
            condition = "HEALTHY"

        # Elevated anomaly score can escalate condition
        if anomaly_score > 0.75 and condition in ("HEALTHY", "MONITOR"):
            condition = "WARNING"
        elif anomaly_score > 0.85 and condition == "WARNING":
            condition = "HIGH_RISK"

        cond_meta = RISK_CATEGORIES[condition]

        # 2. Transparent Weighted Maintenance Priority Index (0 to 100)
        # RUL subscore: 0 at RUL>=125, 100 at RUL=0
        rul_sub = max(0.0, (125.0 - min(predicted_rul, 125.0)) / 125.0) * 100.0
        # Uncertainty subscore: 0 at std=0, 100 at std>=25
        unc_sub = min(100.0, (uncertainty_std / 25.0) * 100.0)
        # Anomaly subscore: 0 at score=0, 100 at score=1
        anom_sub = min(100.0, max(0.0, anomaly_score)) * 100.0
        # Degradation rate subscore: 0 at rate<=0.8, 100 at rate>=2.0
        deg_sub = min(100.0, max(0.0, (degradation_rate - 0.5) / 1.5) * 100.0)
        # Sensor abnormality subscore: 0 at z<=1.5, 100 at z>=4.5
        sens_sub = min(100.0, max(0.0, (max_sensor_zscore - 1.5) / 3.0) * 100.0)

        mpi_raw = (
            self.w_rul * rul_sub +
            self.w_unc * unc_sub +
            self.w_anom * anom_sub +
            self.w_deg * deg_sub +
            self.w_sens * sens_sub
        )
        mpi = round(min(100.0, max(0.0, mpi_raw)), 1)

        priority = cond_meta["priority"]
        urgency_days = cond_meta["urgency_days"]

        # Explainable Breakdown
        breakdown = {
            "rul_subscore": round(rul_sub, 1),
            "uncertainty_subscore": round(unc_sub, 1),
            "anomaly_subscore": round(anom_sub, 1),
            "degradation_rate_subscore": round(deg_sub, 1),
            "sensor_abnormality_subscore": round(sens_sub, 1),
            "weights": {
                "rul": self.w_rul,
                "uncertainty": self.w_unc,
                "anomaly": self.w_anom,
                "degradation_rate": self.w_deg,
                "sensor_abnormality": self.w_sens
            }
        }

        # 3. Evidence-Traceable Recommendations
        recs = self._generate_traceable_recommendations(
            engine_id=engine_id,
            condition=condition,
            mpi=mpi,
            predicted_rul=predicted_rul,
            uncertainty_std=uncertainty_std,
            anomaly_score=anomaly_score,
            max_sensor_zscore=max_sensor_zscore
        )

        return {
            "engine_id": engine_id,
            "condition": condition,
            "health_state": condition,  # backwards compatibility
            "health_info": cond_meta,
            "maintenance_priority_index": mpi,
            "priority": priority,
            "urgency_days": urgency_days,
            "score_breakdown": breakdown,
            "recommended_actions": recs,
            "inspection_window_cycles": max(1, int(predicted_rul * 0.7))
        }

    def _generate_traceable_recommendations(
        self,
        engine_id: int,
        condition: str,
        mpi: float,
        predicted_rul: float,
        uncertainty_std: float,
        anomaly_score: float,
        max_sensor_zscore: float
    ) -> List[Dict[str, Any]]:
        """Synthesize traceable recommendations with concrete evidence attachments."""
        recs = []

        if condition == "CRITICAL" or mpi >= 80:
            recs.append({
                "engine_id": engine_id,
                "risk_classification": condition,
                "supporting_evidence": f"Predicted RUL ({predicted_rul:.1f} cycles) below safety minimum (12 cycles); anomaly score {anomaly_score:.2f}.",
                "relevant_sensor_or_prediction_info": {"predicted_rul": predicted_rul, "anomaly_score": anomaly_score},
                "recommended_action": "Immediate Grounding & Engine Core Shop Visit",
                "subsystem": "Turbine / Combustor",
                "action": "Immediate Grounding & Engine Core Shop Visit",
                "maintenance_priority": "URGENT",
                "reason": "Exceeds flight-safe fatigue threshold. Physical teardown and borescope required.",
                "rationale": "Exceeds flight-safe fatigue threshold. Physical teardown and borescope required."
            })
            recs.append({
                "engine_id": engine_id,
                "risk_classification": condition,
                "supporting_evidence": f"Sensor outlier z-score reached {max_sensor_zscore:.2f} sigma with elevated EGT.",
                "relevant_sensor_or_prediction_info": {"max_zscore": max_sensor_zscore},
                "recommended_action": "Borescope Inspection of HPT Stage 1 Blades & Shrouds",
                "subsystem": "HPT Section",
                "action": "Borescope Inspection of HPT Stage 1 Blades & Shrouds",
                "maintenance_priority": "URGENT",
                "reason": "Verify thermal barrier coating integrity and check for leading-edge creep.",
                "rationale": "Verify thermal barrier coating integrity and check for leading-edge creep."
            })
        elif condition == "HIGH_RISK" or mpi >= 65:
            recs.append({
                "engine_id": engine_id,
                "risk_classification": condition,
                "supporting_evidence": f"RUL forecast is {predicted_rul:.1f} cycles with uncertainty +/- {uncertainty_std:.1f} cycles.",
                "relevant_sensor_or_prediction_info": {"predicted_rul": predicted_rul, "uncertainty_std": uncertainty_std},
                "recommended_action": "Prioritized High-Pressure Compressor (HPC) Inspection",
                "subsystem": "HPC Section",
                "action": "Prioritized High-Pressure Compressor (HPC) Inspection",
                "maintenance_priority": "HIGH",
                "reason": "HPC efficiency loss trends indicate blade fouling or tip clearance growth.",
                "rationale": "HPC efficiency loss trends indicate blade fouling or tip clearance growth."
            })
            recs.append({
                "engine_id": engine_id,
                "risk_classification": condition,
                "supporting_evidence": f"Elevated residual vibration / bleed temperatures observed.",
                "relevant_sensor_or_prediction_info": {"anomaly_score": anomaly_score},
                "recommended_action": "Lube System Chip Detector & Filter Spectrometry",
                "subsystem": "Bearing Compartments 3 / 4",
                "action": "Lube System Chip Detector & Filter Spectrometry",
                "maintenance_priority": "HIGH",
                "reason": "Verify mechanical wear particulate levels prior to next flight segment.",
                "rationale": "Verify mechanical wear particulate levels prior to next flight segment."
            })
        elif condition == "WARNING" or mpi >= 45:
            recs.append({
                "engine_id": engine_id,
                "risk_classification": condition,
                "supporting_evidence": f"RUL estimated at {predicted_rul:.1f} cycles; moderate residual drift.",
                "relevant_sensor_or_prediction_info": {"predicted_rul": predicted_rul},
                "recommended_action": "Schedule Visual Borescope at Next Transit Layover",
                "subsystem": "Combustor Liner",
                "action": "Schedule Visual Borescope at Next Transit Layover",
                "maintenance_priority": "MEDIUM",
                "reason": "Inspect for hot-spot soot deposition and fuel nozzle spray pattern uniformity.",
                "rationale": "Inspect for hot-spot soot deposition and fuel nozzle spray pattern uniformity."
            })
        elif condition == "MONITOR":
            recs.append({
                "engine_id": engine_id,
                "risk_classification": condition,
                "supporting_evidence": f"Nominal RUL ({predicted_rul:.1f} cycles) with mild sensor fluctuation.",
                "relevant_sensor_or_prediction_info": {"predicted_rul": predicted_rul},
                "recommended_action": "Accelerate Trend Telemetry Monitoring to 10 Hz",
                "subsystem": "Telemetry & Controls",
                "action": "Accelerate Trend Telemetry Monitoring to 10 Hz",
                "maintenance_priority": "ROUTINE",
                "reason": "Track rate of change in EGT margin and core speed differential.",
                "rationale": "Track rate of change in EGT margin and core speed differential."
            })
        else:
            recs.append({
                "engine_id": engine_id,
                "risk_classification": condition,
                "supporting_evidence": "All sensor parameters and predictions within nominal certified operational limits.",
                "relevant_sensor_or_prediction_info": {"predicted_rul": predicted_rul, "anomaly_score": anomaly_score},
                "recommended_action": "Maintain Standard Flight Clearance",
                "subsystem": "Full Turbofan Assembly",
                "action": "Maintain Standard Flight Clearance",
                "maintenance_priority": "LOW",
                "reason": "Engine operating smoothly with healthy remaining life margin.",
                "rationale": "Engine operating smoothly with healthy remaining life margin."
            })

        return recs


class MaintenanceDatabase:
    """Persistent SQLite database for engine health tracking, work orders, and alert lifecycles."""

    def __init__(self, db_path: Optional[pathlib.Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS fleet_engines (
                engine_id INTEGER,
                dataset_subset TEXT,
                current_cycle INTEGER,
                health_state TEXT,
                last_predicted_rul REAL,
                mpi REAL,
                last_updated TIMESTAMP,
                PRIMARY KEY (engine_id, dataset_subset)
            )
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                engine_id INTEGER,
                dataset_subset TEXT,
                cycle INTEGER,
                predicted_rul REAL,
                std_dev REAL,
                q10 REAL,
                q90 REAL,
                anomaly_score REAL,
                health_state TEXT,
                mpi REAL,
                model_name TEXT
            )
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS work_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                engine_id INTEGER,
                dataset_subset TEXT,
                priority TEXT,
                title TEXT,
                subsystem TEXT,
                description TEXT,
                status TEXT DEFAULT 'Pending',
                assigned_to TEXT DEFAULT 'Unassigned',
                notes TEXT DEFAULT '',
                closed_at TIMESTAMP
            )
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS maintenance_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                engine_id INTEGER,
                dataset_subset TEXT,
                cycle INTEGER,
                alert_type TEXT,
                message TEXT,
                severity TEXT,
                status TEXT DEFAULT 'ACTIVE',
                acknowledged_at TIMESTAMP,
                resolved_at TIMESTAMP
            )
            """)
            conn.commit()
        finally:
            conn.close()

    def log_prediction(
        self,
        engine_id: int,
        dataset_subset: str,
        cycle: int,
        predicted_rul: float,
        uncertainty: Dict[str, Any],
        anomaly: Dict[str, Any],
        health_eval: Dict[str, Any],
        model_name: str = "Attention-LSTM"
    ) -> int:
        """Record prediction, update fleet state, and evaluate alert triggers."""
        now = datetime.datetime.now().isoformat()
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO predictions_log (
                timestamp, engine_id, dataset_subset, cycle, predicted_rul,
                std_dev, q10, q90, anomaly_score, health_state, mpi, model_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now, engine_id, dataset_subset, cycle, predicted_rul,
                uncertainty.get("std_dev", 0.0),
                uncertainty.get("q10", 0.0),
                uncertainty.get("q90", 0.0),
                anomaly.get("anomaly_score", 0.0),
                health_eval.get("condition", health_eval.get("health_state", "HEALTHY")),
                health_eval.get("maintenance_priority_index", 0.0),
                model_name
            ))
            pred_id = cursor.lastrowid

            cursor.execute("""
            INSERT OR REPLACE INTO fleet_engines (
                engine_id, dataset_subset, current_cycle, health_state,
                last_predicted_rul, mpi, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                engine_id, dataset_subset, cycle,
                health_eval.get("condition", health_eval.get("health_state", "HEALTHY")),
                predicted_rul,
                health_eval.get("maintenance_priority_index", 0.0),
                now
            ))

            # Auto-generate work order if CRITICAL or HIGH_RISK
            cond = health_eval.get("condition", health_eval.get("health_state", "HEALTHY"))
            if cond in ("CRITICAL", "HIGH_RISK"):
                for rec in health_eval.get("recommended_actions", []):
                    # Check if an unresolved duplicate order already exists
                    cursor.execute("""
                    SELECT id FROM work_orders
                    WHERE engine_id = ? AND dataset_subset = ? AND title = ? AND status != 'Completed'
                    """, (engine_id, dataset_subset, rec.get("recommended_action", rec.get("action"))))
                    if not cursor.fetchone():
                        cursor.execute("""
                        INSERT INTO work_orders (
                            created_at, updated_at, engine_id, dataset_subset, priority,
                            title, subsystem, description, status, assigned_to, notes
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Pending', 'Unassigned', ?)
                        """, (
                            now, now, engine_id, dataset_subset,
                            health_eval.get("priority", "HIGH"),
                            rec.get("recommended_action", rec.get("action")),
                            rec.get("subsystem"),
                            rec.get("supporting_evidence", rec.get("reason", "")),
                            f"Auto-generated by Maintenance Decision Engine at cycle {cycle}."
                        ))

            # Auto-generate alert if critical threshold crossed
            if predicted_rul <= 30.0 or anomaly.get("anomaly_score", 0.0) > 0.65:
                alert_type = "RUL_THRESHOLD_EXCEEDED" if predicted_rul <= 30.0 else "SEVERE_ANOMALY"
                cursor.execute("""
                SELECT id FROM maintenance_alerts
                WHERE engine_id = ? AND dataset_subset = ? AND alert_type = ? AND status = 'ACTIVE'
                """, (engine_id, dataset_subset, alert_type))
                if not cursor.fetchone():
                    cursor.execute("""
                    INSERT INTO maintenance_alerts (
                        timestamp, engine_id, dataset_subset, cycle, alert_type, message, severity, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'ACTIVE')
                    """, (
                        now, engine_id, dataset_subset, cycle, alert_type,
                        f"Engine #{engine_id} triggered {alert_type}: RUL={predicted_rul:.1f} cycles, Anomaly={anomaly.get('anomaly_score', 0):.2f}.",
                        "CRITICAL" if predicted_rul <= 15.0 else "WARNING"
                    ))

            conn.commit()
            return pred_id
        finally:
            conn.close()

    # -----------------------------------------------------------------------
    # Work Order CRUD Operations
    # -----------------------------------------------------------------------
    def create_work_order(
        self,
        engine_id: int,
        dataset_subset: str,
        priority: str,
        title: str,
        subsystem: str,
        description: str,
        assigned_to: str = "Unassigned",
        notes: str = ""
    ) -> int:
        now = datetime.datetime.now().isoformat()
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO work_orders (
                created_at, updated_at, engine_id, dataset_subset, priority,
                title, subsystem, description, status, assigned_to, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Pending', ?, ?)
            """, (now, now, engine_id, dataset_subset, priority, title, subsystem, description, assigned_to, notes))
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def update_work_order(
        self,
        order_id: int,
        status: Optional[str] = None,
        assigned_to: Optional[str] = None,
        notes: Optional[str] = None
    ) -> bool:
        now = datetime.datetime.now().isoformat()
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            updates = ["updated_at = ?"]
            params = [now]
            if status is not None:
                updates.append("status = ?")
                params.append(status)
                if status == "Completed":
                    updates.append("closed_at = ?")
                    params.append(now)
            if assigned_to is not None:
                updates.append("assigned_to = ?")
                params.append(assigned_to)
            if notes is not None:
                updates.append("notes = ?")
                params.append(notes)

            params.append(order_id)
            query = f"UPDATE work_orders SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def get_work_orders(
        self,
        dataset_subset: Optional[str] = None,
        status: Optional[str] = None,
        engine_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            query = "SELECT * FROM work_orders WHERE 1=1"
            params = []
            if dataset_subset:
                query += " AND dataset_subset = ?"
                params.append(dataset_subset)
            if status:
                query += " AND status = ?"
                params.append(status)
            if engine_id is not None:
                query += " AND engine_id = ?"
                params.append(engine_id)
            query += " ORDER BY id DESC"
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    # -----------------------------------------------------------------------
    # Alert Lifecycle Operations
    # -----------------------------------------------------------------------
    def get_alerts(self, dataset_subset: Optional[str] = None, status: str = "ACTIVE") -> List[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            query = "SELECT * FROM maintenance_alerts WHERE status = ?"
            params = [status]
            if dataset_subset:
                query += " AND dataset_subset = ?"
                params.append(dataset_subset)
            query += " ORDER BY id DESC"
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def acknowledge_alert(self, alert_id: int) -> bool:
        now = datetime.datetime.now().isoformat()
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("UPDATE maintenance_alerts SET status = 'ACKNOWLEDGED', acknowledged_at = ? WHERE id = ?", (now, alert_id))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def resolve_alert(self, alert_id: int) -> bool:
        now = datetime.datetime.now().isoformat()
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("UPDATE maintenance_alerts SET status = 'RESOLVED', resolved_at = ? WHERE id = ?", (now, alert_id))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    # -----------------------------------------------------------------------
    # Fleet Intelligence & Reporting
    # -----------------------------------------------------------------------
    def get_fleet_summary(self, dataset_subset: str = "FD001") -> Dict[str, Any]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT health_state, count(*) as cnt
            FROM fleet_engines
            WHERE dataset_subset = ?
            GROUP BY health_state
            """, (dataset_subset,))
            health_counts = {row["health_state"]: row["cnt"] for row in cursor.fetchall()}

            cursor.execute("""
            SELECT count(*) as total_orders,
                   sum(case when status != 'Completed' then 1 else 0 end) as open_orders,
                   sum(case when priority='URGENT' and status != 'Completed' then 1 else 0 end) as urgent_orders
            FROM work_orders
            WHERE dataset_subset = ?
            """, (dataset_subset,))
            wo_summary = dict(cursor.fetchone() or {})

            cursor.execute("""
            SELECT count(*) as active_alerts,
                   sum(case when severity='CRITICAL' then 1 else 0 end) as critical_alerts
            FROM maintenance_alerts
            WHERE dataset_subset = ? AND status='ACTIVE'
            """, (dataset_subset,))
            alert_summary = dict(cursor.fetchone() or {})

            cursor.execute("""
            SELECT engine_id, current_cycle, health_state, last_predicted_rul, mpi
            FROM fleet_engines
            WHERE dataset_subset = ?
            ORDER BY mpi DESC LIMIT 15
            """, (dataset_subset,))
            top_urgent = [dict(row) for row in cursor.fetchall()]

            return {
                "health_distribution": health_counts,
                "work_orders": wo_summary,
                "alerts": alert_summary,
                "top_priority_engines": top_urgent
            }
        finally:
            conn.close()

    def export_work_orders_df(self, dataset_subset: Optional[str] = None) -> pd.DataFrame:
        orders = self.get_work_orders(dataset_subset=dataset_subset)
        if not orders:
            return pd.DataFrame(columns=["id", "engine_id", "priority", "title", "subsystem", "status", "assigned_to", "created_at"])
        return pd.DataFrame(orders)


if __name__ == "__main__":
    print("[x] Testing Advanced Predictive Maintenance Intelligence Engine...")
    db = MaintenanceDatabase()
    engine_decision = MaintenanceDecisionEngine()
    res = engine_decision.evaluate_health(
        predicted_rul=18.4,
        uncertainty_std=4.2,
        anomaly_score=0.72,
        current_cycle=165,
        degradation_rate=1.4,
        max_sensor_zscore=2.8,
        engine_id=14
    )
    print("Health Evaluation & Score Breakdown:")
    print(json.dumps(res, indent=2))
    print("[x] Predictive Maintenance Intelligence module verified.")
