"""Component-Level Health Intelligence & Subsystem Degradation Engine (Phase 12.2).

Performs hierarchical condition monitoring across the 7 principal turbofan subassemblies:
1. Fan Assembly (T2, P2, Nf)
2. Low-Pressure Compressor / LPC (T24, P24, NRf)
3. High-Pressure Compressor / HPC (T30, Ps30, phi)
4. Combustion Chamber (T4, BPR, fuel nozzle uniformity)
5. High-Pressure Turbine / HPT (T48 / EGT, Nc, HPT bleed)
6. Low-Pressure Turbine / LPT (T50, P50, FarB)
7. Exhaust System & Nozzle (P15, Core jet velocity)

Computes per-subsystem health indices [0.0 - 1.0], wear rates, and anomaly drivers.
Directly communicates with the 3D Digital Twin visualization.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import pathlib

from src.database_migrations import get_db_connection


class ComponentHealthEngine:
    """Subsystem-Level Degradation & Component Health Analyzer."""

    SUBSYSTEM_MAP = {
        "Fan": {
            "name": "Fan Assembly",
            "sensors": ["sensor_1", "sensor_5", "sensor_13"],
            "failure_modes": ["Blade untwist", "FOD leading-edge nicking", "Tip casing rubbing"],
            "weight": 0.10
        },
        "LPC": {
            "name": "Low-Pressure Compressor (Booster)",
            "sensors": ["sensor_2", "sensor_6", "sensor_14"],
            "failure_modes": ["Stator vane erosion", "Rotor tip clearance growth"],
            "weight": 0.10
        },
        "HPC": {
            "name": "High-Pressure Compressor",
            "sensors": ["sensor_3", "sensor_7", "sensor_11", "sensor_15"],
            "failure_modes": ["Compressor fouling", "Aerodynamic tip rubbing", "Bleed valve leakage"],
            "weight": 0.25
        },
        "Combustor": {
            "name": "Combustion Chamber",
            "sensors": ["sensor_4", "sensor_12"],
            "failure_modes": ["Thermal liner coking", "Fuel nozzle streak hot-spots"],
            "weight": 0.15
        },
        "HPT": {
            "name": "High-Pressure Turbine",
            "sensors": ["sensor_8", "sensor_9", "sensor_17"],
            "failure_modes": ["Thermal barrier coating spallation", "Blade creep & oxidation"],
            "weight": 0.25
        },
        "LPT": {
            "name": "Low-Pressure Turbine",
            "sensors": ["sensor_10", "sensor_16"],
            "failure_modes": ["Seal leakage", "Aerodynamic stage efficiency loss"],
            "weight": 0.10
        },
        "Nozzle": {
            "name": "Exhaust System & Core Nozzle",
            "sensors": ["sensor_20", "sensor_21"],
            "failure_modes": ["Plume pressure distortion", "Nozzle acoustic liner debonding"],
            "weight": 0.05
        }
    }

    def __init__(self, db_path: Optional[pathlib.Path] = None):
        self.db_path = db_path

    def evaluate_subsystem_health(
        self,
        engine_id: int,
        dataset_subset: str,
        current_cycle: int,
        sensor_zscores: Dict[str, float],
        predicted_rul: float
    ) -> Dict[str, Any]:
        """Compute health indices, degradation rates, and dominant anomaly drivers per subsystem."""
        subsystem_reports = []
        overall_health_accumulator = 0.0

        for sub_key, meta in self.SUBSYSTEM_MAP.items():
            sensors = meta["sensors"]
            # Extract z-scores for associated sensors
            z_vals = [abs(sensor_zscores.get(s, 0.0)) for s in sensors]
            max_z = float(np.max(z_vals)) if z_vals else 0.0
            mean_z = float(np.mean(z_vals)) if z_vals else 0.0

            # Find dominant sensor
            dominant_idx = int(np.argmax(z_vals)) if z_vals else 0
            dominant_sensor = sensors[dominant_idx] if sensors else "None"

            # Wear degradation formula: combines RUL depletion + sensor residual z-score
            # RUL score: 1.0 at 125 cycles, 0.0 at 0 cycles
            rul_factor = max(0.0, min(1.0, predicted_rul / 100.0))
            z_factor = max(0.0, 1.0 - (mean_z / 4.0))

            health_idx = float(0.60 * rul_factor + 0.40 * z_factor)
            # Add specific sensitivity for HPC and HPT
            if sub_key in ("HPC", "HPT") and max_z > 2.5:
                health_idx = max(0.0, health_idx - 0.15)

            health_idx = round(min(1.0, max(0.0, health_idx)), 3)
            wear_rate = round((1.0 - health_idx) / max(1, current_cycle) * 100.0, 3)

            status = "HEALTHY" if health_idx >= 0.75 else ("MONITOR" if health_idx >= 0.50 else ("WARNING" if health_idx >= 0.25 else "CRITICAL"))

            report = {
                "subsystem_key": sub_key,
                "subsystem_name": meta["name"],
                "health_index": health_idx,
                "wear_rate_pct_per_100_cycles": wear_rate,
                "status": status,
                "dominant_sensor": dominant_sensor,
                "max_sensor_zscore": round(max_z, 2),
                "weight": meta["weight"],
                "likely_failure_mechanism": meta["failure_modes"][0] if health_idx < 0.60 else "Nominal operating wear",
                "supporting_evidence": f"Sensor {dominant_sensor} exhibited z-score residual deviation of {max_z:.2f} sigma."
            }
            subsystem_reports.append(report)
            overall_health_accumulator += health_idx * meta["weight"]

            # Persist to database
            self._persist_component_health(engine_id, dataset_subset, current_cycle, report)

        return {
            "engine_id": engine_id,
            "dataset_subset": dataset_subset,
            "current_cycle": current_cycle,
            "overall_engine_health_index": round(overall_health_accumulator, 3),
            "subsystems": subsystem_reports,
            "lowest_health_subsystem": min(subsystem_reports, key=lambda s: s["health_index"])
        }

    def _persist_component_health(self, engine_id: int, dataset_subset: str, cycle: int, report: Dict[str, Any]):
        """Record component health assessment in SQLite."""
        try:
            conn = get_db_connection(self.db_path)
            try:
                cur = conn.cursor()
                cur.execute("""
                INSERT INTO component_health_logs (
                    engine_id, dataset_subset, cycle, subsystem, health_index, wear_rate, dominant_sensor, supporting_evidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    engine_id, dataset_subset, cycle, report["subsystem_name"],
                    report["health_index"], report["wear_rate_pct_per_100_cycles"],
                    report["dominant_sensor"], report["supporting_evidence"]
                ))
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass
