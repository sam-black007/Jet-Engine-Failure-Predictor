"""Persistent Experiment Tracker for Jet Engine Failure Predictor (JETWISE AI Phase 2).

Provides an SQLite-backed experiment repository recording full reproducibility metadata,
validation metrics, official test benchmarks, checkpoint paths, and configuration snapshots.
"""

from __future__ import annotations

import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import pandas as pd
from src.config_manager import MasterConfig


class ExperimentTracker:
    """SQLite-backed experiment tracking and metadata repository."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path or "experiments/experiments.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create experiments table if it doesn't already exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    profile_name TEXT,
                    model_architecture TEXT,
                    dataset_subset TEXT,
                    sensor_count INTEGER,
                    window_size INTEGER,
                    loss_name TEXT,
                    pinball_quantile REAL,
                    learning_rate REAL,
                    batch_size INTEGER,
                    epochs_trained INTEGER,
                    training_duration_seconds REAL,
                    val_loss REAL,
                    val_mae REAL,
                    val_nasa REAL,
                    test_nasa_raw REAL,
                    test_nasa_clipped REAL,
                    test_mae REAL,
                    test_rmse REAL,
                    checkpoint_path TEXT,
                    config_snapshot_path TEXT,
                    git_commit TEXT,
                    created_at TEXT,
                    notes TEXT
                )
            """)
            conn.commit()

    def log_experiment(
        self,
        config: MasterConfig,
        training_res: Dict[str, Any],
        test_res: Optional[Dict[str, Any]] = None,
        config_snapshot_path: Optional[str] = None,
        notes: str = ""
    ) -> str:
        """Log a newly completed experiment into the database.
        
        Returns:
            experiment_id string
        """
        exp_id = config.experiment_id
        now_str = datetime.utcnow().isoformat()

        # Extract test metrics if available
        test_nasa_raw = None
        test_nasa_clipped = None
        test_mae = None
        test_rmse = None

        if test_res:
            m_raw = test_res.get("metrics_raw", {})
            m_clip = test_res.get("metrics_clipped", {})
            test_nasa_raw = float(m_raw.get("NASA_score", 0.0))
            test_nasa_clipped = float(m_clip.get("NASA_score", 0.0))
            test_mae = float(m_raw.get("MAE", 0.0))
            test_rmse = float(m_raw.get("RMSE", 0.0))

        record = (
            exp_id,
            config.profile_name,
            config.model.architecture,
            config.dataset.subset,
            config.model.input_dim,
            config.preprocessing.window_size,
            config.loss.name,
            config.loss.pinball_quantile if config.loss.name == "pinball" else None,
            config.training.learning_rate,
            config.training.batch_size,
            training_res.get("best_epoch", 1),
            training_res.get("training_duration_seconds", 0.0),
            training_res["history"]["val_loss"][-1] if training_res.get("history", {}).get("val_loss") else None,
            training_res.get("final_val_mae", None),
            training_res.get("best_val_nasa", None),
            test_nasa_raw,
            test_nasa_clipped,
            test_mae,
            test_rmse,
            training_res.get("checkpoint_path", ""),
            config_snapshot_path or "",
            config.git_commit,
            now_str,
            notes
        )

        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO experiments VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, record)
            conn.commit()

        return exp_id

    def update_test_metrics(self, experiment_id: str, test_res: Dict[str, Any]) -> None:
        """Update an existing experiment with official test benchmark results."""
        m_raw = test_res.get("metrics_raw", {})
        m_clip = test_res.get("metrics_clipped", {})
        raw_nasa = float(m_raw.get("NASA_score", 0.0))
        clip_nasa = float(m_clip.get("NASA_score", 0.0))
        mae = float(m_raw.get("MAE", 0.0))
        rmse = float(m_raw.get("RMSE", 0.0))

        with self._get_connection() as conn:
            conn.execute("""
                UPDATE experiments SET
                    test_nasa_raw = ?,
                    test_nasa_clipped = ?,
                    test_mae = ?,
                    test_rmse = ?
                WHERE experiment_id = ?
            """, (raw_nasa, clip_nasa, mae, rmse, experiment_id))
            conn.commit()

    def list_experiments(self) -> pd.DataFrame:
        """Retrieve all recorded experiments as a pandas DataFrame."""
        with self._get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM experiments ORDER BY created_at DESC", conn)
        return df

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve single experiment record as dictionary."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,))
            row = cur.fetchone()
            if row:
                return dict(row)
        return None

    def get_best_experiment(self, metric: str = "test_nasa_raw") -> Optional[Dict[str, Any]]:
        """Retrieve the best experiment record for the specified metric (ascending order)."""
        valid_metrics = ["test_nasa_raw", "test_nasa_clipped", "val_nasa", "test_mae", "val_mae"]
        if metric not in valid_metrics:
            metric = "test_nasa_raw"

        with self._get_connection() as conn:
            cur = conn.execute(f"""
                SELECT * FROM experiments 
                WHERE {metric} IS NOT NULL AND {metric} > 0 
                ORDER BY {metric} ASC LIMIT 1
            """)
            row = cur.fetchone()
            if row:
                return dict(row)
        return None

    def export_csv(self, output_path: Union[str, Path] = "experiments/experiments.csv") -> Path:
        """Export experiment history to a standalone CSV file."""
        df = self.list_experiments()
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        return p
