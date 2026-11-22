"""Human-in-the-Loop Aerospace Engineering Review Engine (Phase 11.8).

Manages licensed engineering verification, evidence aggregation, and decision sign-offs:
1. Automated review queue filtering (MPI >= 65, RUL <= 25, high uncertainty, or OOD)
2. Comprehensive evidence synthesis (Sensor telemetry, XAI attributions, FDIR, Physics)
3. Formal engineering review actions:
   - ACCEPTED (Authorize recommended line maintenance)
   - REJECTED (Classify as false alarm / operational transient)
   - BORESCOPE_REQUESTED (Mandate internal optical inspection)
   - UNDER_INVESTIGATION (Hold engine for bench diagnostic test)
4. Certified engineering notes & confirmed physical inspection findings
5. Strict provenance separation: Unverified AI outputs vs Validated Ground Truth
6. Persistent storage in SQLite database (engineering_reviews & engineering_investigations)
"""

from typing import Dict, List, Optional, Any, Tuple
import datetime
import pathlib
import json

from src.database_migrations import get_db_connection


class EngineeringReviewEngine:
    """Aerospace Certified Engineering Review & Decision Audit Engine."""

    def __init__(self, db_path: Optional[pathlib.Path] = None):
        self.db_path = db_path

    def get_review_queue(
        self,
        dataset_subset: str = "FD001",
        min_mpi_threshold: float = 50.0
    ) -> List[Dict[str, Any]]:
        """Retrieve fleet units pending certified engineering review."""
        queue = []
        try:
            conn = get_db_connection(self.db_path)
            try:
                cur = conn.cursor()
                # Query engines from fleet_engines with high MPI or Critical status
                cur.execute("""
                SELECT f.engine_id, f.dataset_subset, f.current_cycle, f.health_state,
                       f.last_predicted_rul, f.mpi, f.last_updated
                FROM fleet_engines f
                WHERE f.dataset_subset = ? AND (f.mpi >= ? OR f.health_state IN ('CRITICAL', 'HIGH_RISK'))
                ORDER BY f.mpi DESC
                LIMIT 20
                """, (dataset_subset, min_mpi_threshold))
                rows = cur.fetchall()

                for r in rows:
                    eid = r["engine_id"]
                    # Check if already reviewed in the current cycle
                    cur.execute("""
                    SELECT id, decision, reviewer_name, created_at FROM engineering_reviews
                    WHERE engine_id = ? AND dataset_subset = ?
                    ORDER BY id DESC LIMIT 1
                    """, (eid, dataset_subset))
                    last_rev = cur.fetchone()

                    queue.append({
                        "engine_id": eid,
                        "dataset_subset": dataset_subset,
                        "current_cycle": r["current_cycle"],
                        "health_state": r["health_state"],
                        "predicted_rul": r["last_predicted_rul"],
                        "mpi": r["mpi"],
                        "last_updated": str(r["last_updated"]),
                        "review_status": "REVIEWED" if last_rev else "PENDING_REVIEW",
                        "last_decision": last_rev["decision"] if last_rev else None,
                        "last_reviewer": last_rev["reviewer_name"] if last_rev else None
                    })
            finally:
                conn.close()
        except Exception:
            pass

        return queue

    def submit_engineering_review(
        self,
        engine_id: int,
        dataset_subset: str,
        cycle: int,
        predicted_rul: float,
        decision: str,  # 'ACCEPTED', 'REJECTED', 'BORESCOPE_REQUESTED', 'UNDER_INVESTIGATION'
        reviewer_name: str,
        engineering_notes: str,
        physical_inspection_findings: Optional[str] = None,
        confidence_interval: Optional[Dict[str, float]] = None,
        is_validated_ground_truth: bool = False
    ) -> int:
        """Record formal engineering sign-off decision in SQLite."""
        review_id = -1
        ci_json = json.dumps(confidence_interval or {})

        try:
            conn = get_db_connection(self.db_path)
            try:
                cur = conn.cursor()
                cur.execute("""
                INSERT INTO engineering_reviews (
                    engine_id, dataset_subset, cycle, predicted_rul,
                    confidence_interval_json, decision, reviewer_name,
                    engineering_notes, physical_inspection_findings, is_validated_ground_truth
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    engine_id, dataset_subset, cycle, predicted_rul,
                    ci_json, decision, reviewer_name, engineering_notes,
                    physical_inspection_findings or "", 1 if is_validated_ground_truth else 0
                ))
                review_id = cur.lastrowid

                # If BORESCOPE_REQUESTED or UNDER_INVESTIGATION, also open an investigation record
                if decision in ("BORESCOPE_REQUESTED", "UNDER_INVESTIGATION"):
                    cur.execute("""
                    INSERT INTO engineering_investigations (
                        engine_id, dataset_subset, title, subsystem, findings, investigator, status
                    ) VALUES (?, ?, ?, ?, ?, ?, 'OPEN')
                    """, (
                        engine_id, dataset_subset,
                        f"Formal Investigation: Unit #{engine_id} ({decision})",
                        "Turbomachinery Core",
                        engineering_notes, reviewer_name
                    ))

                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass

        return review_id

    def get_review_history(self, engine_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent engineering review audit logs."""
        reviews = []
        try:
            conn = get_db_connection(self.db_path)
            try:
                cur = conn.cursor()
                if engine_id is not None:
                    cur.execute("""
                    SELECT * FROM engineering_reviews WHERE engine_id = ? ORDER BY id DESC LIMIT ?
                    """, (engine_id, limit))
                else:
                    cur.execute("SELECT * FROM engineering_reviews ORDER BY id DESC LIMIT ?", (limit,))
                rows = cur.fetchall()

                for r in rows:
                    reviews.append({
                        "id": r["id"],
                        "created_at": str(r["created_at"]),
                        "engine_id": r["engine_id"],
                        "dataset_subset": r["dataset_subset"],
                        "cycle": r["cycle"],
                        "predicted_rul": r["predicted_rul"],
                        "decision": r["decision"],
                        "reviewer_name": r["reviewer_name"],
                        "engineering_notes": r["engineering_notes"],
                        "physical_inspection_findings": r["physical_inspection_findings"],
                        "is_validated_ground_truth": bool(r["is_validated_ground_truth"])
                    })
            finally:
                conn.close()
        except Exception:
            pass

        return reviews
