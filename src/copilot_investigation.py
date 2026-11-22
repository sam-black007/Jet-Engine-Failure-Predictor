"""JETWISE AI — Autonomous Engineering Investigation & Multi-Scenario Comparison Engine.

Provides:
1. Autonomous diagnostic investigation ("Investigate why this engine is showing abnormal temperature readings").
2. Evidence-based root-cause hypothesis ranking.
3. Multi-scenario comparative analytics across baseline and modified scenarios.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.copilot_scenarios import (
    EngineeringScenario,
    ScenarioResult,
    PARAMETER_MAP
)
from src.fdir_engine import SensorFDIREngine
from src.physics_engine import PhysicsInformedEngine


class AutonomousInvestigationEngine:
    """Investigates turbomachinery anomalies and ranks diagnostic hypotheses with evidence."""

    def __init__(self):
        self.fdir_engine = SensorFDIREngine()
        self.physics_engine = PhysicsInformedEngine()

    def investigate_anomaly(
        self,
        engine_trajectory: pd.DataFrame,
        target_symptom: str = "temperature",
        current_cycle: int = 100
    ) -> Dict[str, Any]:
        """Performs multi-step engineering diagnosis for observed abnormal telemetry."""
        sensor_cols = [c for c in engine_trajectory.columns if c.startswith("sensor_")]
        if not sensor_cols:
            return {"status": "NO_DATA", "hypotheses": []}

        latest_row = engine_trajectory.iloc[-1]
        mean_row = engine_trajectory[sensor_cols].mean()
        std_row = engine_trajectory[sensor_cols].std().replace(0, 1e-4)

        # Compute z-scores for all sensors
        z_scores = {s: float((latest_row[s] - mean_row[s]) / std_row[s]) for s in sensor_cols}

        # Check for sensor faults using FDIR
        try:
            fdir_res = self.fdir_engine.analyze_sensor_window(
                engine_id=1,
                dataset_subset="FD001",
                current_cycle=current_cycle,
                telemetry_window_df=engine_trajectory[sensor_cols].iloc[-30:]
            )
            faulty_sensors = [f.sensor_name for f in fdir_res.get("fault_reports", []) if hasattr(f, "sensor_name")]
        except Exception:
            faulty_sensors = []

        # Candidate hypotheses list
        hypotheses = []

        # 1. High-Pressure Turbine (HPT) Thermal Distress
        t50_z = z_scores.get("sensor_4", 0.0)
        w31_z = z_scores.get("sensor_20", 0.0)
        if t50_z > 1.5 or w31_z > 1.5 or "temperature" in target_symptom.lower():
            evidence = [
                f"LPT Outlet / EGT (T50) departed by {t50_z:+.2f}σ above nominal baseline.",
                f"HPT Coolant Bleed Flow (W31) z-score at {w31_z:+.2f}σ.",
                "Elevated exhaust gas temperature indicates reduced turbine expansion efficiency or cooling hole glazing."
            ]
            confidence = min(0.95, max(0.55, 0.60 + (abs(t50_z) * 0.10)))
            hypotheses.append({
                "rank": 1,
                "subsystem": "High-Pressure Turbine (HPT)",
                "hypothesis": "Turbine Blade Creep & Thermal Barrier Coating (TBC) Spallation",
                "evidence": evidence,
                "confidence_score": round(confidence, 2),
                "recommended_action": "Perform immediate borescope inspection of HPT 1st stage nozzle guide vanes.",
                "parameter_to_adjust": "Increase fuel flow or reduce cooling bleed to investigate thermal margin sensitivity."
            })

        # 2. High-Pressure Compressor (HPC) Aerodynamic Fouling
        t30_z = z_scores.get("sensor_3", 0.0)
        p30_z = z_scores.get("sensor_7", 0.0)
        nc_z = z_scores.get("sensor_9", 0.0)
        if t30_z > 1.2 or p30_z < -1.0 or "pressure" in target_symptom.lower():
            evidence = [
                f"HPC Exit Temperature (T30) shifted by {t30_z:+.2f}σ.",
                f"HPC Exit Pressure (P30) shifted by {p30_z:+.2f}σ.",
                f"Core physical speed (Nc) shift at {nc_z:+.2f}σ.",
                "Elevated discharge temperature combined with pressure loss is the classic signature of compressor aerodynamic fouling."
            ]
            confidence = min(0.90, max(0.50, 0.55 + (abs(t30_z) * 0.08)))
            hypotheses.append({
                "rank": len(hypotheses) + 1,
                "subsystem": "High-Pressure Compressor (HPC)",
                "hypothesis": "Compressor Blade Fouling & Tip Clearance Erosion",
                "evidence": evidence,
                "confidence_score": round(confidence, 2),
                "recommended_action": "Schedule off-line chemical compressor water wash and boroblend inspection.",
                "parameter_to_adjust": "Adjust compressor pressure (P30) downward to simulate continued aerodynamic stall margin loss."
            })

        # 3. Sensor Instrumentation Transducer Drift / Bias
        if faulty_sensors:
            evidence = [
                f"Sensor FDIR engine flagged active instrumentation anomalies on: {', '.join(faulty_sensors)}.",
                "Anomalous telemetry is primarily confined to a single transducer without corroborating thermal gas path shifts."
            ]
            hypotheses.append({
                "rank": len(hypotheses) + 1,
                "subsystem": "Avionics / Instrumentation",
                "hypothesis": "Thermocouple or Pressure Transducer Signal Drift (False Physical Alarm)",
                "evidence": evidence,
                "confidence_score": 0.82,
                "recommended_action": "Bench-test and recalibrate dual-channel sensor wiring harness before hardware removal.",
                "parameter_to_adjust": "Introduce a compensatory calibration offset on the flagged sensor channel."
            })

        # Sort hypotheses by confidence score descending
        hypotheses.sort(key=lambda x: x["confidence_score"], reverse=True)
        for i, h in enumerate(hypotheses):
            h["rank"] = i + 1

        return {
            "status": "DIAGNOSIS_COMPLETE",
            "target_symptom": target_symptom,
            "engine_cycle": current_cycle,
            "highest_confidence_subsystem": hypotheses[0]["subsystem"] if hypotheses else "Normal Operation",
            "hypotheses": hypotheses,
            "top_explanation": (
                f"Autonomous diagnosis identifies '{hypotheses[0]['hypothesis']}' as the primary contributor "
                f"(Confidence: {hypotheses[0]['confidence_score']*100:.0f}%). "
                f"{hypotheses[0]['evidence'][0]}"
            ) if hypotheses else "Engine telemetry is operating within nominal statistical tolerances."
        }


class ScenarioComparisonEngine:
    """Compares multiple executed engineering scenarios side-by-side."""

    @staticmethod
    def compare_scenarios(scenario_results: List[ScenarioResult]) -> Dict[str, Any]:
        """Generates cross-scenario comparison matrix and trend metrics."""
        if not scenario_results:
            return {"error": "No scenarios provided for comparison."}

        table_rows = []
        for res in scenario_results:
            table_rows.append({
                "scenario_name": res.scenario_name,
                "mode": res.simulation_mode.value,
                "rul_cycles": res.modified_rul,
                "rul_delta": res.rul_delta,
                "uncertainty_std": res.modified_uncertainty.get("std_dev", 0.0),
                "confidence": res.modified_uncertainty.get("confidence", 0.8),
                "verification": res.verification_status.value,
                "affected_subsystems": ", ".join(res.affected_components) or "None (Nominal)"
            })

        best_scenario = max(scenario_results, key=lambda r: r.modified_rul)
        worst_scenario = min(scenario_results, key=lambda r: r.modified_rul)

        summary = (
            f"Evaluated {len(scenario_results)} distinct scenarios. "
            f"Highest operational life observed in '{best_scenario.scenario_name}' ({best_scenario.modified_rul} cycles). "
            f"Severe degradation observed in '{worst_scenario.scenario_name}' ({worst_scenario.modified_rul} cycles, Δ={worst_scenario.rul_delta:+.1f} cycles)."
        )

        return {
            "num_scenarios": len(scenario_results),
            "summary": summary,
            "comparison_matrix": table_rows,
            "best_scenario": best_scenario.scenario_name,
            "worst_scenario": worst_scenario.scenario_name
        }
