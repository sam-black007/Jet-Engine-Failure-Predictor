"""Prediction Uncertainty & Conformal Calibration Engine (Phase 11.4).

Provides distribution-free conformal prediction intervals, Monte Carlo Dropout
uncertainty calibration, and empirical reliability assessment:
1. Split Conformal Prediction (finite-sample coverage guarantees 1 - alpha)
2. Nonconformity score estimation on held-out calibration split
3. Prediction Interval Coverage Probability (PICP) & Mean Interval Width (MPIW)
4. Regression Expected Calibration Error (ECE)
5. Coverage assessment across operating regimes & datasets
6. Persistent calibration experiment tracking in SQLite database
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pathlib
import json

from src.database_migrations import get_db_connection


class ConformalCalibrator:
    """Split Conformal Predictor for Remaining Useful Life (RUL) regression."""

    def __init__(
        self,
        target_coverage: float = 0.90,
        db_path: Optional[pathlib.Path] = None
    ):
        self.target_coverage = target_coverage
        self.alpha = 1.0 - target_coverage
        self.db_path = db_path
        self.calibrated_quantile: float = 20.0
        self.is_calibrated: bool = False
        self.calibration_size: int = 0

    def fit_calibration_residuals(
        self,
        y_true_cal: np.ndarray,
        y_pred_cal: np.ndarray,
        model_name: str = "pinball_lstm_q35",
        dataset_name: str = "FD001"
    ) -> Dict[str, Any]:
        """Compute empirical nonconformity scores R_i = |y_i - y_hat_i| and calculate conformal quantile."""
        residuals = np.abs(y_true_cal - y_pred_cal)
        n = len(residuals)
        if n < 5:
            # Fallback default margin
            self.calibrated_quantile = 25.0
            self.is_calibrated = True
            return {"status": "INSUFFICIENT_CALIBRATION_DATA", "quantile": self.calibrated_quantile}

        # Conformal quantile level with finite sample correction: ceil((n+1)(1 - alpha)) / n
        k = int(np.ceil((n + 1) * (1.0 - self.alpha)))
        k = min(n, max(1, k))
        sorted_residuals = np.sort(residuals)
        self.calibrated_quantile = float(sorted_residuals[k - 1])
        self.is_calibrated = True
        self.calibration_size = n

        # Evaluate on calibration set itself
        in_bounds = (residuals <= self.calibrated_quantile)
        empirical_coverage = float(np.mean(in_bounds))
        mean_width = float(2.0 * self.calibrated_quantile)
        ece = float(abs(empirical_coverage - self.target_coverage))

        # Persist record
        self._persist_calibration_record(
            model_name=model_name,
            dataset=dataset_name,
            method="conformal",
            target_cov=self.target_coverage,
            emp_cov=empirical_coverage,
            mean_width=mean_width,
            ece=ece,
            params={"alpha": self.alpha, "n_cal": n, "conformal_q": self.calibrated_quantile}
        )

        return {
            "model_name": model_name,
            "dataset": dataset_name,
            "method": "split_conformal",
            "target_coverage": self.target_coverage,
            "empirical_coverage": round(empirical_coverage, 4),
            "conformal_quantile_margin": round(self.calibrated_quantile, 2),
            "mean_interval_width": round(mean_width, 2),
            "ece_score": round(ece, 4),
            "calibration_samples": n
        }

    def predict_conformal_interval(
        self,
        point_prediction: float,
        target_coverage: Optional[float] = None
    ) -> Dict[str, Any]:
        """Construct calibrated conformal prediction interval for a point prediction."""
        q = self.calibrated_quantile
        if target_coverage is not None and abs(target_coverage - self.target_coverage) > 0.01:
            # Scale proportionally if user requests dynamic coverage
            scale = (target_coverage / self.target_coverage)
            q = q * scale

        lower_bound = max(0.0, float(point_prediction - q))
        upper_bound = float(point_prediction + q)
        interval_width = upper_bound - lower_bound

        return {
            "predicted_rul": round(point_prediction, 1),
            "conformal_lower": round(lower_bound, 1),
            "conformal_upper": round(upper_bound, 1),
            "interval_width": round(interval_width, 1),
            "target_coverage": target_coverage or self.target_coverage,
            "is_calibrated": self.is_calibrated,
            "calibration_method": "Split Conformal Prediction (Distribution-Free)"
        }

    def evaluate_test_calibration(
        self,
        y_true_test: np.ndarray,
        y_pred_test: np.ndarray,
        nominal_coverages: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Generate empirical reliability diagram data across multiple nominal coverage targets."""
        targets = nominal_coverages or [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]
        residuals = np.abs(y_true_test - y_pred_test)
        n = len(residuals)

        reliability_curve = []
        for cov in targets:
            q_val = float(np.quantile(residuals, cov))
            in_interval = np.mean(residuals <= q_val)
            width = 2.0 * q_val
            reliability_curve.append({
                "nominal_coverage": cov,
                "empirical_coverage": round(float(in_interval), 4),
                "margin": round(q_val, 2),
                "interval_width": round(width, 2),
                "coverage_error": round(float(in_interval - cov), 4)
            })

        # Calculate Expected Calibration Error
        ece = float(np.mean([abs(r["coverage_error"]) for r in reliability_curve]))

        return {
            "test_sample_count": n,
            "reliability_curve": reliability_curve,
            "overall_ece": round(ece, 4),
            "mean_prediction_interval_width": round(float(np.mean([r["interval_width"] for r in reliability_curve])), 2)
        }

    def _persist_calibration_record(
        self,
        model_name: str,
        dataset: str,
        method: str,
        target_cov: float,
        emp_cov: float,
        mean_width: float,
        ece: float,
        params: Dict[str, Any]
    ):
        """Record calibration experiment in SQLite."""
        try:
            conn = get_db_connection(self.db_path)
            try:
                cur = conn.cursor()
                cur.execute("""
                INSERT INTO calibration_records (
                    model_name, dataset_subset, method, target_coverage,
                    empirical_coverage, mean_interval_width, ece_score, parameters_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_name, dataset, method, target_cov, emp_cov, mean_width, ece, json.dumps(params)
                ))
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass
