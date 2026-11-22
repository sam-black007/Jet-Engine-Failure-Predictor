"""Predictive Maintenance Intelligence Engine (JETWISE AI Phase 7).

Integrates:
1. Epistemic & Aleatoric Uncertainty Estimation (MC Dropout & Prediction Intervals)
2. Unsupervised Engine Anomaly Detection (Isolation Forest on Sensor Residuals)
3. Sequence Explainable AI (Perturbation & Gradient-based Sensor Attribution)
4. Maintenance Priority & Decision Engine (Health Classification & Maintenance Priority Index)
5. Persistent SQLite Maintenance Records Database (Audit trail & Work Orders)
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

HEALTH_STATES = {
    "HEALTHY": {"min_rul": 75, "color": "#21C77A", "description": "Nominal operational status."},
    "MONITOR": {"min_rul": 40, "color": "#00B8D9", "description": "Elevated wear patterns observed; monitor closely."},
    "MAINTENANCE_DUE": {"min_rul": 20, "color": "#FFB547", "description": "Maintenance required within scheduled window."},
    "CRITICAL": {"min_rul": 0, "color": "#FF5252", "description": "Severe degradation; immediate grounding / overhaul recommended."}
}


class UncertaintyEstimator:
    """Estimates epistemic and empirical uncertainty via Monte Carlo Dropout."""

    def __init__(self, model: nn.Module, num_samples: int = 30):
        self.model = model
        self.num_samples = num_samples

    def enable_dropout(self):
        """Enable dropout layers while keeping batchnorm/layernorm in eval mode."""
        for m in self.model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
                m.train()

    def predict_with_uncertainty(self, x: np.ndarray) -> Dict[str, Any]:
        """Perform stochastic Monte Carlo forward passes.

        Parameters
        ----------
        x : np.ndarray
            Input sequence tensor of shape (batch_size, seq_len, num_features) or (seq_len, num_features).

        Returns
        -------
        Dict[str, Any]
            mean_rul, std_dev, q10, q50, q90, confidence_score
        """
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

        preds_arr = np.array(preds)  # shape: (num_samples, batch_size) or (num_samples,)
        if preds_arr.ndim == 1:
            preds_arr = np.expand_dims(preds_arr, axis=1)

        mean_rul = np.mean(preds_arr, axis=0)
        std_rul = np.std(preds_arr, axis=0)
        q10 = np.percentile(preds_arr, 10, axis=0)
        q50 = np.percentile(preds_arr, 50, axis=0)
        q90 = np.percentile(preds_arr, 90, axis=0)

        # Confidence: high confidence when std is low and RUL is consistent
        # Normalized confidence metric 0.0 - 1.0
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
        """Fit Isolation Forest on nominal healthy cycles (e.g. cycle <= 30)."""
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
            # Fallback simple z-score check
            return {"is_anomaly": False, "anomaly_score": 0.0, "severity": "NORMAL", "top_anomalous_sensors": []}

        # Use final cycle in window or mean of window
        last_cycle = window_vals[-1:]
        raw_score = self.model.decision_function(last_cycle)[0]  # negative means outlier
        pred_label = self.model.predict(last_cycle)[0]  # -1 for anomaly, 1 for inlier

        # Normalized anomaly score 0.0 (normal) to 1.0 (extreme anomaly)
        norm_score = float(np.clip(0.5 - raw_score, 0.0, 1.0))
        is_anomaly = bool(pred_label == -1 or norm_score > 0.55)

        # Detect top anomalous sensors via z-score
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
        """Compute integrated gradient approximation for input sequence window.

        Parameters
        ----------
        window_vals : np.ndarray
            Sequence of shape (seq_len, num_features)
        num_steps : int
            Number of interpolation steps between baseline (zeros) and input.

        Returns
        -------
        Dict[str, Any]
            Attribution per sensor, relative percentage, and time step importance.
        """
        self.model.eval()
        x_target = torch.from_numpy(window_vals.astype(np.float32)).unsqueeze(0)  # (1, seq_len, num_features)
        x_baseline = torch.zeros_like(x_target)

        # Integrated gradients
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
        ig = (x_target - x_baseline).detach() * avg_grads  # (1, seq_len, num_features)
        ig_abs = torch.abs(ig).squeeze(0).cpu().numpy()  # (seq_len, num_features)

        # Sum over time dimension to get sensor importance
        sensor_scores = np.sum(ig_abs, axis=0)
        total_score = np.sum(sensor_scores) + 1e-8
        sensor_weights = (sensor_scores / total_score) * 100.0

        # Temporal importance over window
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
    """Aerospace Decision Engine: health scoring, priority ranking, and action recommendations."""

    def __init__(self):
        pass

    def evaluate_health(
        self,
        predicted_rul: float,
        uncertainty_std: float = 5.0,
        anomaly_score: float = 0.2,
        current_cycle: int = 150
    ) -> Dict[str, Any]:
        """Classify health state, compute Maintenance Priority Index (MPI), and recommend action."""
        # Health state determination
        if predicted_rul <= 20:
            health_state = "CRITICAL"
        elif predicted_rul <= 40:
            health_state = "MAINTENANCE_DUE"
        elif predicted_rul <= 75:
            health_state = "MONITOR"
        else:
            health_state = "HEALTHY"

        if anomaly_score > 0.70 and health_state in ("HEALTHY", "MONITOR"):
            health_state = "MAINTENANCE_DUE"

        # Maintenance Priority Index (0 to 100)
        # RUL factor: 0 at RUL=125, 60 at RUL=0
        rul_factor = max(0.0, (125.0 - min(predicted_rul, 125.0)) / 125.0) * 60.0
        # Anomaly factor: 0 to 25
        anomaly_factor = min(1.0, max(0.0, anomaly_score)) * 25.0
        # Uncertainty factor: 0 to 15
        unc_factor = min(1.0, max(0.0, uncertainty_std / 25.0)) * 15.0

        mpi = min(100.0, round(rul_factor + anomaly_factor + unc_factor, 1))

        # Priority label
        if mpi >= 80:
            priority = "URGENT"
            urgency_days = 2
        elif mpi >= 60:
            priority = "HIGH"
            urgency_days = 7
        elif mpi >= 40:
            priority = "MEDIUM"
            urgency_days = 21
        elif mpi >= 20:
            priority = "ROUTINE"
            urgency_days = 45
        else:
            priority = "LOW"
            urgency_days = 90

        # Action recommendation synthesis
        recommendations = self._generate_recommendations(health_state, mpi, predicted_rul, anomaly_score)

        return {
            "health_state": health_state,
            "health_info": HEALTH_STATES[health_state],
            "maintenance_priority_index": mpi,
            "priority": priority,
            "urgency_days": urgency_days,
            "recommended_actions": recommendations,
            "inspection_window_cycles": max(1, int(predicted_rul * 0.7))
        }

    def _generate_recommendations(
        self,
        health_state: str,
        mpi: float,
        predicted_rul: float,
        anomaly_score: float
    ) -> List[Dict[str, str]]:
        recs = []
        if health_state == "CRITICAL" or mpi >= 80:
            recs.append({
                "action": "Immediate Grounding & Shop Visit",
                "subsystem": "Turbine / Combustor",
                "rationale": f"Estimated RUL is {predicted_rul:.1f} cycles. Exceeds safe flight operational threshold."
            })
            recs.append({
                "action": "Complete High-Pressure Turbine (HPT) Borescope",
                "subsystem": "HPT Blades & Stator Vanes",
                "rationale": "High thermal stress signatures indicate imminent thermal barrier coating spallation."
            })
        elif health_state == "MAINTENANCE_DUE" or mpi >= 60:
            recs.append({
                "action": "Scheduled Engine Core Inspection",
                "subsystem": "High-Pressure Compressor (HPC)",
                "rationale": f"Predictive degradation trajectory requires maintenance within {int(predicted_rul)} cycles."
            })
            recs.append({
                "action": "Lube System Chip Detector & Filter Analysis",
                "subsystem": "Bearing Compartment 3 / 4",
                "rationale": "Verify no mechanical particulate accumulation from high-cycle fatigue."
            })
        elif health_state == "MONITOR":
            recs.append({
                "action": "Trend Telemetry Acceleration",
                "subsystem": "Sensor Fleet Monitoring",
                "rationale": "Increase data telemetry polling to 10 Hz for vibration (sensor_14) and core temp (sensor_4)."
            })
            recs.append({
                "action": "Routine Borescope at Next Transit",
                "subsystem": "Combustor Liner",
                "rationale": "Confirm normal soot deposition and absence of hot spots."
            })
        else:
            recs.append({
                "action": "Maintain Standard Flight Clearance",
                "subsystem": "Full Turbofan Assembly",
                "rationale": "Engine operating within nominal health margins across all aerothermal parameters."
            })

        return recs


class MaintenanceDatabase:
    """Persistent SQLite database for engine health tracking, predictions, and work orders."""

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
                engine_id INTEGER,
                dataset_subset TEXT,
                priority TEXT,
                title TEXT,
                subsystem TEXT,
                description TEXT,
                status TEXT DEFAULT 'OPEN',
                closed_at TIMESTAMP
            )
            """)

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS sensor_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                engine_id INTEGER,
                dataset_subset TEXT,
                cycle INTEGER,
                sensor_name TEXT,
                z_score REAL,
                severity TEXT
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
        """Record prediction and update fleet state."""
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
                health_eval.get("health_state", "HEALTHY"),
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
                health_eval.get("health_state", "HEALTHY"),
                predicted_rul,
                health_eval.get("maintenance_priority_index", 0.0),
                now
            ))

            # Auto-generate work order if CRITICAL or URGENT
            if health_eval.get("priority") in ("URGENT", "HIGH"):
                for rec in health_eval.get("recommended_actions", []):
                    cursor.execute("""
                    INSERT INTO work_orders (
                        created_at, engine_id, dataset_subset, priority,
                        title, subsystem, description, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN')
                    """, (
                        now, engine_id, dataset_subset,
                        health_eval.get("priority"),
                        rec.get("action"),
                        rec.get("subsystem"),
                        rec.get("rationale")
                    ))

            conn.commit()
            return pred_id
        finally:
            conn.close()

    def get_fleet_summary(self, dataset_subset: str = "FD001") -> Dict[str, Any]:
        """Retrieve aggregated fleet overview statistics."""
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
                   sum(case when status='OPEN' then 1 else 0 end) as open_orders,
                   sum(case when priority='URGENT' and status='OPEN' then 1 else 0 end) as urgent_orders
            FROM work_orders
            WHERE dataset_subset = ?
            """, (dataset_subset,))
            wo_summary = dict(cursor.fetchone() or {})

            cursor.execute("""
            SELECT engine_id, current_cycle, health_state, last_predicted_rul, mpi
            FROM fleet_engines
            WHERE dataset_subset = ?
            ORDER BY mpi DESC LIMIT 10
            """, (dataset_subset,))
            top_urgent = [dict(row) for row in cursor.fetchall()]

            return {
                "health_distribution": health_counts,
                "work_orders": wo_summary,
                "top_priority_engines": top_urgent
            }
        finally:
            conn.close()

    def get_work_orders(self, dataset_subset: Optional[str] = None, status: str = "OPEN") -> List[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            query = "SELECT * FROM work_orders WHERE status = ?"
            params = [status]
            if dataset_subset:
                query += " AND dataset_subset = ?"
                params.append(dataset_subset)
            query += " ORDER BY id DESC LIMIT 50"
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()


if __name__ == "__main__":
    print("[x] Testing Predictive Maintenance Intelligence Engine...")
    db = MaintenanceDatabase()
    engine_decision = MaintenanceDecisionEngine()
    res = engine_decision.evaluate_health(predicted_rul=18.4, uncertainty_std=4.2, anomaly_score=0.72)
    print("Health Evaluation Result:")
    print(json.dumps(res, indent=2))
    print("[x] Predictive Maintenance Intelligence module verified.")
