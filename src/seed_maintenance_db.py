"""Seed the persistent Maintenance Database with predictions, health states, and work orders."""

import pathlib
import numpy as np
import pandas as pd
import torch

from src.data_loader import load_dataset, load_rul
from src.predictive_intelligence import (
    MaintenanceDatabase,
    MaintenanceDecisionEngine,
    UncertaintyEstimator,
    EngineAnomalyDetector
)
from src.train_lstm import LSTMPredictor

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


def seed_database():
    print("[>] Initializing and seeding maintenance database...")
    db = MaintenanceDatabase()
    decision_engine = MaintenanceDecisionEngine()

    # Load trained model for inference
    model_path = PROJECT_ROOT / "models" / "lstm_asymmetric" / "pinball_q35_lstm.pt"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "models" / "lstm" / "lstm_model.pt"

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model = LSTMPredictor(input_dim=17)
    model.load_state_dict({k: v for k, v in state.items() if k in model.state_dict()})
    model.eval()

    unc_estimator = UncertaintyEstimator(model, num_samples=15)

    subsets = ["FD001", "FD002", "FD003", "FD004"]
    feature_cols = [
        "sensor_2", "sensor_3", "sensor_4", "sensor_5", "sensor_6", "sensor_7", "sensor_8",
        "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
        "sensor_16", "sensor_17", "sensor_20", "sensor_21"
    ]

    for subset in subsets:
        print(f"  [+] Seeding records for subset {subset}...")
        try:
            train_df = load_dataset(subset, kind="train")
            test_df = load_dataset(subset, kind="test")
            test_rul = load_rul(subset).values.astype(np.float32)
        except Exception as e:
            print(f"      Skipping {subset}: {e}")
            continue

        # Fit anomaly detector on baseline nominal cycles (cycle <= 25)
        baseline = train_df[train_df["cycle"] <= 25]
        detector = EngineAnomalyDetector(contamination=0.05)
        detector.fit(baseline, feature_cols)

        # Compute predictions for first 20 engines
        eids = sorted(test_df["engine_id"].unique())[:20]
        means = train_df[feature_cols].mean().values
        stds = train_df[feature_cols].std().replace(0, 1.0).values

        for i, eid in enumerate(eids):
            eng = test_df[test_df["engine_id"] == eid].sort_values("cycle")
            max_c = int(eng["cycle"].max())
            raw_w = eng[feature_cols].iloc[-30:].values if len(eng) >= 30 else eng[feature_cols].values
            if len(raw_w) < 30:
                pad = np.repeat(raw_w[0:1], 30 - len(raw_w), axis=0)
                raw_w = np.vstack([pad, raw_w])

            norm_w = (raw_w - means) / stds

            # Uncertainty & Prediction
            unc = unc_estimator.predict_with_uncertainty(norm_w)
            pred_rul = max(0.0, unc["mean_rul"])

            # Anomaly
            anom = detector.detect_window_anomaly(norm_w)

            # Health Evaluation
            health = decision_engine.evaluate_health(
                predicted_rul=pred_rul,
                uncertainty_std=unc["std_dev"],
                anomaly_score=anom["anomaly_score"],
                current_cycle=max_c
            )

            # Log to SQLite DB
            db.log_prediction(
                engine_id=int(eid),
                dataset_subset=subset,
                cycle=max_c,
                predicted_rul=pred_rul,
                uncertainty=unc,
                anomaly=anom,
                health_eval=health,
                model_name="Pinball-LSTM (q=0.35)"
            )

    print("[x] Database seeded successfully!")
    summary = db.get_fleet_summary("FD001")
    print(f"FD001 Fleet Summary: {summary}")


if __name__ == "__main__":
    seed_database()
