"""Sensor Fault Detection, Isolation, and Recovery (FDIR) Engine (Phase 11.1).

Provides multi-modal sensor reliability diagnostics:
1. Stuck-at fault detection (near-zero variance across active flight windows)
2. Sensor drift detection (monotonic residual drift slope via linear trend)
3. Packet drop / missing measurement detection
4. High-frequency noise & spike detection (Signal-to-Noise Ratio SNR analysis)
5. Intermittent step-jump anomaly detection
6. Multivariate correlation shift monitoring (Frobenius norm of cross-sensor correlations)
7. Clear distinction: suspected sensor fault vs physical engine hardware damage
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import json
import pathlib
import sqlite3
from dataclasses import dataclass, asdict

from src.data_loader import SENSOR_COLS
from src.dataset_metadata import CMAPSS_SENSOR_METADATA
from src.database_migrations import get_db_connection


@dataclass
class SensorFaultReport:
    """Diagnostic fault record for an individual sensor."""
    sensor_id: str
    sensor_name: str
    fault_type: str  # 'NOMINAL', 'STUCK_AT', 'DRIFT', 'NOISE', 'PACKET_DROP', 'CORRELATION_INCONSISTENCY'
    severity: str    # 'NONE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    confidence: float
    measured_value: float
    expected_value: float
    residual: float
    details: Dict[str, Any]
    status: str      # 'SUSPECTED_SENSOR_FAULT', 'NOMINAL'


class SensorFDIREngine:
    """Aerospace Sensor Reliability & Fault Isolation Engine."""

    def __init__(
        self,
        stuck_var_threshold: float = 1e-4,
        drift_slope_threshold: float = 0.05,
        noise_snr_threshold_db: float = 15.0,
        missing_packet_tolerance: int = 1,
        z_score_threshold: float = 3.0,
        db_path: Optional[pathlib.Path] = None
    ):
        self.stuck_var_threshold = stuck_var_threshold
        self.drift_slope_threshold = drift_slope_threshold
        self.noise_snr_threshold_db = noise_snr_threshold_db
        self.missing_packet_tolerance = missing_packet_tolerance
        self.z_score_threshold = z_score_threshold
        self.db_path = db_path

    def analyze_sensor_window(
        self,
        engine_id: int,
        dataset_subset: str,
        current_cycle: int,
        telemetry_window_df: pd.DataFrame,
        baseline_means: Optional[Dict[str, float]] = None,
        baseline_stds: Optional[Dict[str, float]] = None,
        baseline_corr: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """Analyze a multi-cycle telemetry window (e.g. 30 cycles) for sensor faults.
        
        Returns:
            Dictionary containing:
            - 'sensor_status_matrix': List of per-sensor health summaries
            - 'fault_reports': List of active fault reports
            - 'correlation_shift_score': Frobenius norm difference from baseline
            - 'overall_sensor_health_index': Float [0.0, 1.0]
        """
        sensor_matrix = []
        fault_reports = []

        active_cols = [c for c in telemetry_window_df.columns if c in SENSOR_COLS]
        if not active_cols:
            active_cols = [c for c in telemetry_window_df.columns if c.startswith("sensor_")]

        num_samples = len(telemetry_window_df)
        time_steps = np.arange(num_samples)

        for s_col in active_cols:
            series = telemetry_window_df[s_col].values
            meta = CMAPSS_SENSOR_METADATA.get(s_col, {})
            s_name = meta.get("name", s_col)
            unit = meta.get("units", "")

            # 1. Missing / Packet Drop check
            nan_count = np.isnan(series).sum() + np.isinf(series).sum()
            zero_count = (series == 0.0).sum() if s_col not in ["sensor_1", "sensor_5"] else 0

            latest_val = float(series[-1]) if len(series) > 0 and not np.isnan(series[-1]) else 0.0
            var_val = float(np.var(series)) if len(series) > 1 else 0.0
            mean_val = float(np.mean(series)) if len(series) > 0 else 0.0

            b_mean = baseline_means.get(s_col, mean_val) if baseline_means else mean_val
            b_std = baseline_stds.get(s_col, 1.0) if baseline_stds and baseline_stds.get(s_col, 0) > 0 else 1.0
            z_score = abs(latest_val - b_mean) / b_std

            fault_type = "NOMINAL"
            severity = "NONE"
            confidence = 1.0
            expected_val = b_mean
            evidence = {}

            # Check 1: Missing or Dropout
            if nan_count > self.missing_packet_tolerance or zero_count > self.missing_packet_tolerance:
                fault_type = "PACKET_DROP"
                severity = "HIGH"
                confidence = 0.95
                evidence = {"nan_count": int(nan_count), "zero_count": int(zero_count)}

            # Check 2: Stuck-at Sensor (Invariant in dynamic flight)
            elif var_val < self.stuck_var_threshold and s_col not in ["sensor_1", "sensor_5", "sensor_10", "sensor_18", "sensor_19"]:
                fault_type = "STUCK_AT"
                severity = "HIGH"
                confidence = 0.92
                evidence = {"window_variance": var_val, "threshold": self.stuck_var_threshold}

            # Check 3: Linear Sensor Drift (High residual slope)
            elif len(series) >= 10:
                slope, _ = np.polyfit(time_steps, series, 1)
                normalized_slope = abs(slope) / (b_std + 1e-6)
                if normalized_slope > self.drift_slope_threshold:
                    fault_type = "DRIFT"
                    severity = "MEDIUM" if normalized_slope < 0.15 else "HIGH"
                    confidence = min(0.98, 0.70 + normalized_slope * 2.0)
                    evidence = {"drift_slope": float(slope), "norm_slope": float(normalized_slope)}

            # Check 4: Excessive High-Frequency Noise (Spurious variance)
            if fault_type == "NOMINAL" and len(series) >= 5:
                # Difference sequence
                diffs = np.diff(series)
                noise_power = np.var(diffs) + 1e-9
                signal_power = np.var(series) + 1e-9
                snr_ratio = signal_power / noise_power
                snr_db = 10.0 * np.log10(snr_ratio) if snr_ratio > 0 else 0.0

                if snr_db < self.noise_snr_threshold_db and var_val > self.stuck_var_threshold:
                    fault_type = "NOISE"
                    severity = "LOW" if snr_db > 8.0 else "MEDIUM"
                    confidence = 0.85
                    evidence = {"snr_db": float(snr_db), "noise_power": float(noise_power)}

            # Check 5: Statistical Outlier Spike
            if fault_type == "NOMINAL" and z_score > self.z_score_threshold:
                fault_type = "CORRELATION_INCONSISTENCY"
                severity = "MEDIUM" if z_score < 4.5 else "HIGH"
                confidence = min(0.95, 0.60 + (z_score / 10.0))
                evidence = {"z_score": float(z_score), "baseline_mean": b_mean, "baseline_std": b_std}

            report = SensorFaultReport(
                sensor_id=s_col,
                sensor_name=s_name,
                fault_type=fault_type,
                severity=severity,
                confidence=float(confidence),
                measured_value=latest_val,
                expected_value=float(expected_val),
                residual=float(latest_val - expected_val),
                details=evidence,
                status="SUSPECTED_SENSOR_FAULT" if fault_type != "NOMINAL" else "NOMINAL"
            )

            if fault_type != "NOMINAL":
                fault_reports.append(report)

            sensor_matrix.append({
                "sensor_id": s_col,
                "sensor_name": s_name,
                "unit": unit,
                "latest_value": round(latest_val, 2),
                "expected_value": round(expected_val, 2),
                "residual": round(latest_val - expected_val, 3),
                "variance": round(var_val, 5),
                "z_score": round(z_score, 2),
                "health_status": "FAULT_DETECTED" if fault_type != "NOMINAL" else "HEALTHY",
                "fault_type": fault_type,
                "severity": severity,
                "confidence": round(confidence * 100, 1),
                "subsystem": meta.get("subsystem", "General")
            })

        # Multivariate Correlation Shift
        corr_shift_score = 0.0
        if len(telemetry_window_df) >= 5:
            current_corr = telemetry_window_df[active_cols].corr().fillna(0.0)
            if baseline_corr is not None:
                # Align columns
                common_cols = [c for c in active_cols if c in baseline_corr.columns]
                diff = current_corr.loc[common_cols, common_cols].values - baseline_corr.loc[common_cols, common_cols].values
                corr_shift_score = float(np.linalg.norm(diff, ord="fro"))

        total_sensors = len(active_cols)
        faulty_count = len(fault_reports)
        health_index = max(0.0, 1.0 - (faulty_count / max(1, total_sensors)))

        # Optionally persist faults to database
        if fault_reports:
            self._persist_fault_events(engine_id, dataset_subset, current_cycle, fault_reports)

        return {
            "engine_id": engine_id,
            "dataset_subset": dataset_subset,
            "cycle": current_cycle,
            "sensor_status_matrix": sensor_matrix,
            "fault_reports": [asdict(f) for f in fault_reports],
            "correlation_shift_score": round(corr_shift_score, 3),
            "overall_sensor_health_index": round(health_index, 3),
            "faults_detected_count": faulty_count,
            "total_sensors_monitored": total_sensors,
            "diagnostic_disclaimer": (
                "NOTICE: Sensor fault alerts indicate suspected electrical, line, or telemetry transducer abnormalities. "
                "They do not constitute verified engine hardware damage until calibrated with shop multimeter / loop checks."
            )
        }

    def _persist_fault_events(self, engine_id: int, dataset_subset: str, cycle: int, fault_reports: List[SensorFaultReport]):
        """Store fault events in SQLite sensor_fdir_events table with de-duplication."""
        try:
            conn = get_db_connection(self.db_path)
            try:
                cur = conn.cursor()
                for f in fault_reports:
                    # Check recent duplicate within 5 cycles
                    cur.execute("""
                    SELECT id FROM sensor_fdir_events
                    WHERE engine_id = ? AND dataset_subset = ? AND sensor_id = ? AND fault_type = ? AND cycle >= ?
                    """, (engine_id, dataset_subset, f.sensor_id, f.fault_type, max(0, cycle - 5)))
                    if not cur.fetchone():
                        cur.execute("""
                        INSERT INTO sensor_fdir_events (
                            engine_id, dataset_subset, cycle, sensor_id, fault_type,
                            severity, confidence, measured_value, expected_value, residual, evidence_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            engine_id, dataset_subset, cycle, f.sensor_id, f.fault_type,
                            f.severity, f.confidence, f.measured_value, f.expected_value, f.residual, json.dumps(f.details)
                        ))
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass  # Silent fail-safe for database logging
