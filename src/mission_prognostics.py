"""Mission-Aware Engine Prognostics Engine (Phase 12.1).

Models how varying operating duty cycles, mission profiles, and environmental
flight conditions modulate engine degradation rates and remaining useful life:
1. Configurable aerospace operational profiles:
   - Nominal (Standard mixed commercial fleet baseline)
   - Short-Haul (High takeoff/climb cyclic thermal fatigue)
   - Long-Haul (High steady-state cruise creep, lower cyclic fatigue)
   - Hot & High (Elevated ambient temperatures, thin air, elevated EGT)
   - Severe Dust / Sand (Particulate abrasive erosion & cooling hole coking)
2. Scenario RUL projection & baseline comparison
3. Accelerated degradation curves & sensor trajectory synthesis
4. Explicit labeling: Hypothetical scenario projections vs Historical ground truth
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import pathlib
import json

from src.database_migrations import get_db_connection


class MissionAwarePrognostics:
    """Mission Profile Degradation Modeler & Scenario Comparator."""

    PROFILES = {
        "nominal": {
            "name": "Standard Commercial Cruise",
            "wear_multiplier": 1.0,
            "egt_offset_rankine": 0.0,
            "cyclic_fatigue_factor": 1.0,
            "description": "Balanced short-to-medium haul commercial airline flight profile."
        },
        "short_haul": {
            "name": "High-Frequency Short-Haul",
            "wear_multiplier": 1.25,
            "egt_offset_rankine": 15.0,
            "cyclic_fatigue_factor": 1.45,
            "description": "Frequent low-altitude takeoffs and climbs inducing elevated thermal low-cycle fatigue (LCF)."
        },
        "long_haul": {
            "name": "Extended Trans-Oceanic Long-Haul",
            "wear_multiplier": 0.85,
            "egt_offset_rankine": -10.0,
            "cyclic_fatigue_factor": 0.70,
            "description": "Prolonged steady cruise at high altitude; reduced cyclic thermal shock, steady creep dominance."
        },
        "hot_high": {
            "name": "Hot & High Altitude Operations",
            "wear_multiplier": 1.40,
            "egt_offset_rankine": 45.0,
            "cyclic_fatigue_factor": 1.30,
            "description": "High ambient airfield temperatures and low air density requiring maximum thrust ratings and elevated EGT."
        },
        "severe_dust": {
            "name": "Arid / Desert Dust & Sand Ingestion",
            "wear_multiplier": 1.65,
            "egt_offset_rankine": 60.0,
            "cyclic_fatigue_factor": 1.50,
            "description": "Abrasive particulate ingestion causing rapid compressor blade tip clearance erosion and turbine cooling hole clogging."
        }
    }

    def __init__(self, db_path: Optional[pathlib.Path] = None):
        self.db_path = db_path

    def evaluate_mission_impact(
        self,
        engine_id: int,
        dataset_subset: str,
        current_cycle: int,
        baseline_predicted_rul: float,
        selected_profile: str = "short_haul"
    ) -> Dict[str, Any]:
        """Project degradation trajectory and remaining life delta under a designated mission profile."""
        profile_key = selected_profile.lower().replace(" ", "_")
        if profile_key not in self.PROFILES:
            profile_key = "nominal"

        spec = self.PROFILES[profile_key]
        multiplier = spec["wear_multiplier"]

        # Projected RUL under new mission rate: RUL_new = RUL_baseline / multiplier
        projected_rul = max(0.0, baseline_predicted_rul / multiplier)
        rul_delta = projected_rul - baseline_predicted_rul

        # Synthesize multi-cycle degradation curve
        cycles_ahead = int(min(200, max(20, baseline_predicted_rul * 1.5)))
        trajectory_steps = list(range(current_cycle, current_cycle + cycles_ahead + 1, 5))

        baseline_curve = []
        mission_curve = []

        for c in trajectory_steps:
            elapsed = c - current_cycle
            r_base = max(0.0, baseline_predicted_rul - elapsed)
            r_miss = max(0.0, baseline_predicted_rul - (elapsed * multiplier))
            baseline_curve.append({"cycle": c, "rul": round(r_base, 1)})
            mission_curve.append({"cycle": c, "rul": round(r_miss, 1)})

        result = {
            "engine_id": engine_id,
            "dataset_subset": dataset_subset,
            "current_cycle": current_cycle,
            "profile_key": profile_key,
            "profile_name": spec["name"],
            "description": spec["description"],
            "wear_acceleration_factor": multiplier,
            "egt_thermal_penalty_rankine": spec["egt_offset_rankine"],
            "cyclic_fatigue_factor": spec["cyclic_fatigue_factor"],
            "baseline_rul": round(baseline_predicted_rul, 1),
            "projected_rul": round(projected_rul, 1),
            "rul_delta_cycles": round(rul_delta, 1),
            "baseline_trajectory": baseline_curve,
            "mission_trajectory": mission_curve,
            "disclaimer": (
                "RESEARCH NOTICE: Mission degradation modulations are analytical thermodynamic approximations. "
                "They represent comparative research scenarios, not certified maintenance intervals."
            )
        }

        # Persist scenario to SQLite
        self._persist_mission_scenario(result)
        return result

    def _persist_mission_scenario(self, res: Dict[str, Any]):
        """Store mission scenario evaluation in SQLite."""
        try:
            conn = get_db_connection(self.db_path)
            try:
                cur = conn.cursor()
                cur.execute("""
                INSERT INTO mission_scenarios (
                    engine_id, dataset_subset, profile_name, baseline_rul,
                    projected_rul, rul_delta, degradation_acceleration_factor, assumptions_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    res["engine_id"], res["dataset_subset"], res["profile_name"],
                    res["baseline_rul"], res["projected_rul"], res["rul_delta_cycles"],
                    res["wear_acceleration_factor"], json.dumps(res)
                ))
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass
