"""Active Learning & Intelligent Inspection Engine (Phase 12.5).

Ranks uninspected or partially run fleet engines to identify the most informative
observations for physical teardown inspection and model retraining:
1. Epistemic uncertainty acquisition (MC Dropout standard deviation)
2. Operational regime novelty acquisition (sparse coverage in Altitude-Mach-TRA)
3. Anomaly residual dispersion acquisition
4. Priority ranking of candidate units for ground truth verification
5. Integration with certified human review feedback loop
6. Persistent query storage in SQLite database (active_learning_queries)
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import pathlib

from src.database_migrations import get_db_connection


class ActiveLearningEngine:
    """Aerospace Active Learning & Ground-Truth Acquisition Optimizer."""

    def __init__(
        self,
        weight_uncertainty: float = 0.50,
        weight_regime_novelty: float = 0.30,
        weight_anomaly: float = 0.20,
        db_path: Optional[pathlib.Path] = None
    ):
        self.w_unc = weight_uncertainty
        self.w_reg = weight_regime_novelty
        self.w_anom = weight_anomaly
        self.db_path = db_path

    def rank_inspection_candidates(
        self,
        fleet_units: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Compute acquisition scores and rank fleet units for physical teardown/inspection.
        
        Expected fields per unit:
            - engine_id: int
            - dataset: str
            - current_cycle: int
            - uncertainty_std: float (e.g. 2.0 to 18.0 cycles)
            - ood_score: float (0.0 to 1.0)
            - anomaly_score: float (0.0 to 1.0)
            - predicted_rul: float
        """
        candidates = []

        for unit in fleet_units:
            eid = unit["engine_id"]
            ds = unit.get("dataset", "FD001")
            cycle = unit.get("current_cycle", 100)
            unc = float(unit.get("uncertainty_std", 5.0))
            ood = float(unit.get("ood_score", 0.2))
            anom = float(unit.get("anomaly_score", 0.2))
            rul = float(unit.get("predicted_rul", 50.0))

            # Normalize uncertainty score [0 - 1]: 15 cycles std is max
            unc_norm = min(1.0, unc / 15.0)

            # Acquisition Function: weighted combination
            acquisition_score = (
                self.w_unc * unc_norm +
                self.w_reg * ood +
                self.w_anom * anom
            )

            # Determine primary query reason
            if unc_norm >= ood and unc_norm >= anom:
                reason = "HIGH_EPISTEMIC_UNCERTAINTY"
                explanation = f"Model displays high prediction dispersion (std={unc:.1f} cycles)."
            elif ood >= anom:
                reason = "UNDERREPRESENTED_FLIGHT_REGIME"
                explanation = f"Flight telemetry lies in low-density training envelope (OOD={ood:.2f})."
            else:
                reason = "UNEXPLAINED_SENSOR_RESIDUALS"
                explanation = f"Unsupervised anomaly score {anom:.2f} warrants ground-truth verification."

            candidates.append({
                "engine_id": eid,
                "dataset_subset": ds,
                "current_cycle": cycle,
                "predicted_rul": round(rul, 1),
                "uncertainty_std": round(unc, 1),
                "ood_score": round(ood, 2),
                "anomaly_score": round(anom, 2),
                "acquisition_score": round(float(acquisition_score * 100.0), 1),
                "query_reason": reason,
                "rationale": explanation
            })

        # Sort descending by acquisition score
        ranked = sorted(candidates, key=lambda c: c["acquisition_score"], reverse=True)
        for rank, c in enumerate(ranked, 1):
            c["priority_rank"] = rank

        top_candidates = ranked[:top_k]
        self._persist_candidates(top_candidates)
        return top_candidates

    def _persist_candidates(self, candidates: List[Dict[str, Any]]):
        """Store top queries in SQLite active_learning_queries table."""
        try:
            conn = get_db_connection(self.db_path)
            try:
                cur = conn.cursor()
                for c in candidates:
                    cur.execute("""
                    SELECT id FROM active_learning_queries
                    WHERE engine_id = ? AND dataset_subset = ? AND status = 'PENDING_INSPECTION'
                    """, (c["engine_id"], c["dataset_subset"]))
                    if not cur.fetchone():
                        cur.execute("""
                        INSERT INTO active_learning_queries (
                            engine_id, dataset_subset, acquisition_score, query_reason, priority_rank, status
                        ) VALUES (?, ?, ?, ?, ?, 'PENDING_INSPECTION')
                        """, (
                            c["engine_id"], c["dataset_subset"],
                            c["acquisition_score"], c["query_reason"], c["priority_rank"]
                        ))
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass
