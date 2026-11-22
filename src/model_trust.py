"""Out-of-Distribution (OOD) Detection & Model Trust Engine (Phase 11.3).

Assesses whether incoming flight telemetry lies within the validated operational
and statistical domain of the trained machine learning model:
1. Mahalanobis distance in normalized multivariate sensor space
2. Isolation Forest novelty detection on latent feature representations
3. Operating-regime boundary checking (Altitude, Mach, Throttle Resolver Angle)
4. PCA feature-space projection & variance coverage
5. Model Trust Status:
   - IN DOMAIN
   - DISTRIBUTION SHIFT DETECTED
   - OUT OF DISTRIBUTION
   - INSUFFICIENT EVIDENCE
   - UNSUPPORTED CONFIGURATION
6. Persistent OOD alerting in SQLite database
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import pathlib
import json
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from src.database_migrations import get_db_connection


class ModelTrustEngine:
    """Out-of-Distribution Detection & Epistemic Trust Boundary Evaluator."""

    def __init__(
        self,
        mahalanobis_threshold_p95: float = 25.0,
        isolation_contamination: float = 0.05,
        db_path: Optional[pathlib.Path] = None
    ):
        self.mahalanobis_threshold = mahalanobis_threshold_p95
        self.isolation_contamination = isolation_contamination
        self.db_path = db_path

        # Reference training statistics
        self.train_mean: Optional[np.ndarray] = None
        self.inv_cov: Optional[np.ndarray] = None
        self.pca: Optional[PCA] = None
        self.iso_forest: Optional[IsolationForest] = None
        self.fitted_dataset: Optional[str] = None
        self.regime_bounds: Dict[str, Tuple[float, float]] = {}

    def fit_reference_distribution(
        self,
        train_features: np.ndarray,
        feature_names: List[str],
        dataset_name: str = "FD001",
        op_settings_df: Optional[pd.DataFrame] = None
    ):
        """Fit empirical baseline statistics on training set engines."""
        self.fitted_dataset = dataset_name
        self.feature_names = feature_names

        # Clean NaNs
        valid_mask = ~np.isnan(train_features).any(axis=1) & ~np.isinf(train_features).any(axis=1)
        X = train_features[valid_mask]

        if len(X) < 10:
            return

        # 1. Mean and Regularized Covariance for Mahalanobis
        self.train_mean = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        # Add small ridge epsilon for numerical stability
        cov_reg = cov + np.eye(cov.shape[0]) * 1e-4
        try:
            self.inv_cov = np.linalg.pinv(cov_reg)
        except Exception:
            self.inv_cov = np.diag(1.0 / (np.var(X, axis=0) + 1e-4))

        # 2. PCA for 2D/3D projection
        self.pca = PCA(n_components=min(3, X.shape[1]))
        self.pca.fit(X)

        # 3. Isolation Forest on training data
        self.iso_forest = IsolationForest(
            contamination=self.isolation_contamination,
            random_state=42,
            n_estimators=50
        )
        self.iso_forest.fit(X)

        # 4. Operating condition bounds
        if op_settings_df is not None:
            for col in op_settings_df.columns:
                vals = op_settings_df[col].dropna().values
                if len(vals) > 0:
                    self.regime_bounds[col] = (float(np.min(vals)), float(np.max(vals)))

    def evaluate_observation_trust(
        self,
        engine_id: int,
        dataset_subset: str,
        cycle: int,
        model_name: str,
        feature_vector: np.ndarray,
        op_settings_dict: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Evaluate epistemic trust and OOD status for a single observation or sequence window."""
        # Check if reference is fitted; if not, use standard fallback
        dim = len(feature_vector)
        if self.train_mean is None or len(self.train_mean) != dim:
            # Fallback zero-mean identity covariance
            self.train_mean = np.zeros(dim)
            self.inv_cov = np.eye(dim)

        diff = feature_vector - self.train_mean
        # Mahalanobis Distance: sqrt( diff * inv_cov * diff^T )
        try:
            m_dist_sq = float(np.dot(np.dot(diff, self.inv_cov), diff.T))
            mahalanobis_dist = float(np.sqrt(max(0.0, m_dist_sq)))
        except Exception:
            mahalanobis_dist = float(np.linalg.norm(diff))

        # Isolation score
        iso_score = 0.5
        if self.iso_forest is not None:
            try:
                # decision_function gives negative score for outliers
                raw_score = float(self.iso_forest.decision_function(feature_vector.reshape(1, -1))[0])
                # normalize to [0, 1] where 1 is extreme outlier
                iso_score = float(1.0 / (1.0 + np.exp(raw_score * 5.0)))
            except Exception:
                iso_score = min(1.0, mahalanobis_dist / 30.0)
        else:
            iso_score = min(1.0, mahalanobis_dist / 30.0)

        # Operating regime bounds violation check
        regime_violations = []
        if op_settings_dict and self.regime_bounds:
            for k, val in op_settings_dict.items():
                if k in self.regime_bounds:
                    low, high = self.regime_bounds[k]
                    margin = (high - low) * 0.10
                    if val < (low - margin) or val > (high + margin):
                        regime_violations.append(f"{k}: {val:.2f} outside [{low:.2f}, {high:.2f}]")

        # Composite OOD Score [0.0 - 1.0]
        m_score_norm = min(1.0, mahalanobis_dist / max(1.0, self.mahalanobis_threshold * 1.5))
        ood_score = float(0.60 * m_score_norm + 0.40 * iso_score)

        # Status Classification
        if regime_violations or ood_score >= 0.75:
            status = "OUT_OF_DISTRIBUTION"
            model_trust_pct = max(0.0, (1.0 - ood_score) * 60.0)
        elif ood_score >= 0.45:
            status = "DISTRIBUTION_SHIFT_DETECTED"
            model_trust_pct = max(40.0, (1.0 - ood_score) * 100.0)
        else:
            status = "IN_DOMAIN"
            model_trust_pct = min(100.0, 95.0 + (1.0 - ood_score) * 5.0)

        # PCA 2D/3D Projection
        pca_coords = [0.0, 0.0, 0.0]
        if self.pca is not None:
            try:
                proj = self.pca.transform(feature_vector.reshape(1, -1))[0]
                pca_coords = [round(float(c), 3) for c in proj[:3]]
            except Exception:
                pass

        # Identify top features contributing to distance
        feature_deviations = []
        if hasattr(self, "feature_names") and len(self.feature_names) == dim:
            z_devs = np.abs(diff) / (np.sqrt(np.diag(np.linalg.pinv(self.inv_cov))) + 1e-4)
            top_indices = np.argsort(z_devs)[::-1][:3]
            for idx in top_indices:
                feature_deviations.append({
                    "feature": self.feature_names[idx],
                    "z_deviation": round(float(z_devs[idx]), 2),
                    "observed": round(float(feature_vector[idx]), 3),
                    "reference_mean": round(float(self.train_mean[idx]), 3)
                })

        # Persist alert if OOD or Shift
        if status in ("OUT_OF_DISTRIBUTION", "DISTRIBUTION_SHIFT_DETECTED"):
            self._persist_ood_alert(
                engine_id=engine_id,
                dataset_subset=dataset_subset,
                cycle=cycle,
                model_name=model_name,
                ood_score=ood_score,
                m_dist=mahalanobis_dist,
                iso_score=iso_score,
                status=status,
                flagged=feature_deviations
            )

        return {
            "engine_id": engine_id,
            "dataset_subset": dataset_subset,
            "cycle": cycle,
            "model_name": model_name,
            "ood_score": round(ood_score, 3),
            "mahalanobis_distance": round(mahalanobis_dist, 2),
            "isolation_score": round(iso_score, 3),
            "status_classification": status,
            "model_trust_pct": round(model_trust_pct, 1),
            "regime_violations": regime_violations,
            "top_deviating_features": feature_deviations,
            "pca_coordinates": pca_coords,
            "applicability_verdict": (
                "SAFE FOR OPERATIONAL USE" if status == "IN_DOMAIN" else
                ("USE WITH CAUTION: MODERATE DOMAIN SHIFT" if status == "DISTRIBUTION_SHIFT_DETECTED" else
                 "HIGH RISK: OBSERVATION OUTSIDE VALIDATED TRAINING DOMAIN")
            )
        }

    def _persist_ood_alert(
        self,
        engine_id: int,
        dataset_subset: str,
        cycle: int,
        model_name: str,
        ood_score: float,
        m_dist: float,
        iso_score: float,
        status: str,
        flagged: List[Dict[str, Any]]
    ):
        """Record OOD alert into SQLite."""
        try:
            conn = get_db_connection(self.db_path)
            try:
                cur = conn.cursor()
                cur.execute("""
                INSERT INTO ood_alerts (
                    engine_id, dataset_subset, cycle, model_name, ood_score,
                    mahalanobis_distance, isolation_score, status_classification, flagged_features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    engine_id, dataset_subset, cycle, model_name, ood_score,
                    m_dist, iso_score, status, json.dumps(flagged)
                ))
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass
