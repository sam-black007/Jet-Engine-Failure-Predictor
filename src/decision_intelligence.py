"""Action-Conditioned Decision Intelligence Engine (Phase 12.8).

Evaluates and compares counterfactual operational and maintenance decisions:
1. Four standardized aerospace actions:
   - Status Quo (Continue normal flight clearance)
   - Throttle Derating (10% takeoff power reduction -> 20% wear reduction)
   - Line Borescope & Minor Servicing (Clear fouling, water wash, replace seal)
   - Immediate Engine Core Overhaul (Heavy shop visit, full life restoration)
2. Multi-attribute evaluation:
   - Expected RUL extension (cycles)
   - In-flight abort risk reduction (%)
   - Aircraft downtime (hours)
   - MRO resource cost index (0 - 100)
3. Radar / Spider chart trade-off projection
4. Independent recording of human final decision in SQLite (engineering_decisions)
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pathlib

from src.database_migrations import get_db_connection


class DecisionIntelligenceEngine:
    """Action-Conditioned Multi-Criteria Engineering Decision Evaluator."""

    ACTIONS = {
        "status_quo": {
            "name": "Continue Normal Flight Clearance",
            "rul_delta_cycles": 0.0,
            "risk_reduction_pct": 0.0,
            "downtime_hours": 0.0,
            "cost_index": 0.0,
            "description": "Maintain existing flight schedule without proactive intervention."
        },
        "throttle_derate": {
            "name": "Throttle De-rate (10% Power Reduction)",
            "rul_delta_cycles": 18.5,
            "risk_reduction_pct": 35.0,
            "downtime_hours": 0.5,
            "cost_index": 12.0,
            "description": "FADEC software limit derating peak turbine inlet temperature and thermal fatigue."
        },
        "minor_servicing": {
            "name": "Compressor Water Wash & Borescope",
            "rul_delta_cycles": 28.0,
            "risk_reduction_pct": 65.0,
            "downtime_hours": 8.0,
            "cost_index": 35.0,
            "description": "Transit layover servicing: remove aerodynamic fouling and inspect Stage 1 HPT vanes."
        },
        "shop_overhaul": {
            "name": "Immediate Engine Core Overhaul",
            "rul_delta_cycles": 120.0,
            "risk_reduction_pct": 98.0,
            "downtime_hours": 72.0,
            "cost_index": 95.0,
            "description": "Full off-wing shop visit with rotor teardown, blade recoating, and bearing replacement."
        }
    }

    def __init__(self, db_path: Optional[pathlib.Path] = None):
        self.db_path = db_path

    def evaluate_decision_matrix(
        self,
        engine_id: int,
        dataset_subset: str,
        current_cycle: int,
        current_predicted_rul: float,
        current_mpi: float
    ) -> Dict[str, Any]:
        """Project multi-attribute outcomes for all candidate operational actions."""
        action_evaluations = []

        for key, spec in self.ACTIONS.items():
            projected_rul = current_predicted_rul + spec["rul_delta_cycles"]

            # Calculate adjusted risk: MPI scaled by risk reduction
            adjusted_mpi = max(5.0, current_mpi * (1.0 - spec["risk_reduction_pct"] / 100.0))

            # Composite Decision Value Score (Higher is better balance)
            # Value = RUL benefit + Risk reduction - normalized cost - normalized downtime
            rul_norm = min(100.0, (spec["rul_delta_cycles"] / 120.0) * 100.0)
            risk_norm = spec["risk_reduction_pct"]
            cost_norm = spec["cost_index"]
            down_norm = min(100.0, (spec["downtime_hours"] / 72.0) * 100.0)

            # Balanced utility function: 35% life, 35% safety, 15% downtime, 15% cost
            utility_score = (0.35 * rul_norm + 0.35 * risk_norm) - (0.15 * down_norm + 0.15 * cost_norm)

            action_evaluations.append({
                "action_key": key,
                "action_name": spec["name"],
                "projected_rul": round(projected_rul, 1),
                "rul_delta_cycles": spec["rul_delta_cycles"],
                "in_flight_risk_reduction_pct": spec["risk_reduction_pct"],
                "projected_mpi": round(adjusted_mpi, 1),
                "downtime_hours": spec["downtime_hours"],
                "cost_index": spec["cost_index"],
                "utility_score": round(utility_score, 1),
                "description": spec["description"]
            })

        # Find recommended optimal trade-off action
        best_action = max(action_evaluations, key=lambda a: a["utility_score"])

        return {
            "engine_id": engine_id,
            "dataset_subset": dataset_subset,
            "current_cycle": current_cycle,
            "baseline_predicted_rul": current_predicted_rul,
            "baseline_mpi": current_mpi,
            "action_evaluations": action_evaluations,
            "ai_recommended_action": best_action["action_name"],
            "recommendation_rationale": (
                f"Action '{best_action['action_name']}' delivers the optimal multi-criteria trade-off "
                f"(Utility score {best_action['utility_score']}) with +{best_action['rul_delta_cycles']:.0f} cycles life extension "
                f"and {best_action['in_flight_risk_reduction_pct']:.0f}% risk reduction."
            )
        }

    def record_human_decision(
        self,
        engine_id: int,
        dataset_subset: str,
        cycle: int,
        action_chosen: str,
        projected_rul: float,
        rationale: str,
        engineer_id: str
    ) -> int:
        """Persist confirmed engineering decision separately from AI suggestion in SQLite."""
        record_id = -1
        try:
            conn = get_db_connection(self.db_path)
            try:
                cur = conn.cursor()
                cur.execute("""
                INSERT INTO engineering_decisions (
                    engine_id, dataset_subset, cycle, action_chosen,
                    projected_rul_outcome, decision_rationale, engineer_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    engine_id, dataset_subset, cycle, action_chosen,
                    projected_rul, rationale, engineer_id
                ))
                record_id = cur.lastrowid
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass
        return record_id
