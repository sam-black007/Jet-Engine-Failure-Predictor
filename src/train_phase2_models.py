"""Phase 2 Model Training and Official FD001 Test Evaluation Script.

Trains, evaluates, and registers:
1. Attention-LSTM
2. Temporal Convolutional Network (TCN)
3. Time-Series Transformer Encoder

Evaluates all models against the official NASA C-MAPSS FD001 test dataset (100 engines)
using the exact official evaluation protocol and logs results to ExperimentTracker and ModelRegistry.
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

from src.config_manager import ConfigManager, MasterConfig
from src.configurable_data import ConfigurableDataPipeline
from src.training_engine import TrainingEngine
from src.evaluation import evaluate_official_test_set, compute_metrics
from src.experiment_tracker import ExperimentTracker
from src.model_registry import ModelRegistry


def train_and_evaluate_phase2():
    print("=" * 78)
    print(" THRUSTLAB — PHASE 2: ADVANCED ARCHITECTURES TRAINING & OFFICIAL BENCHMARK ")
    print("=" * 78)

    cm = ConfigManager()
    tracker = ExperimentTracker()
    registry = ModelRegistry()

    # 1. Prepare Data
    base_cfg = cm.load_config()
    pipeline = ConfigurableDataPipeline(base_cfg)
    data = pipeline.prepare_data()

    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    input_dim = X_train.shape[2]
    seq_len = X_train.shape[1]

    print(f"Data Prepared: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"               X_val={X_val.shape}, y_val={y_val.shape}")
    print(f"               Features ({input_dim}): {data['feature_cols']}")
    print("-" * 78)

    # Output directory for Phase 2 results
    res_dir = repo_root / "results" / "phase2"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Architectures to train
    experiments_to_run = [
        {
            "tag": "attention_lstm",
            "name": "Phase 2 Attention-LSTM (Pinball q=0.35)",
            "profile": "attention_lstm_experiment",
            "checkpoint_sub": "models/attention_lstm",
            "overrides": {
                "model": {
                    "architecture": "attention_lstm",
                    "input_dim": input_dim,
                    "lstm_hidden_dim1": 64,
                    "lstm_hidden_dim2": 32,
                    "attention_dim": 32,
                    "dropout": 0.2
                },
                "loss": {"name": "pinball", "pinball_quantile": 0.35},
                "training": {"epochs": 30, "batch_size": 64, "learning_rate": 0.001, "early_stopping_patience": 8}
            }
        },
        {
            "tag": "tcn",
            "name": "Phase 2 Temporal ConvNet (Pinball q=0.35)",
            "profile": "tcn_experiment",
            "checkpoint_sub": "models/tcn",
            "overrides": {
                "model": {
                    "architecture": "tcn",
                    "input_dim": input_dim,
                    "tcn_num_channels": [32, 64, 64],
                    "tcn_kernel_size": 3,
                    "tcn_dropout": 0.2
                },
                "loss": {"name": "pinball", "pinball_quantile": 0.35},
                "training": {"epochs": 30, "batch_size": 64, "learning_rate": 0.001, "early_stopping_patience": 8}
            }
        },
        {
            "tag": "transformer",
            "name": "Phase 2 Time-Series Transformer (Pinball q=0.35)",
            "profile": "transformer_experiment",
            "checkpoint_sub": "models/transformer",
            "overrides": {
                "model": {
                    "architecture": "transformer",
                    "input_dim": input_dim,
                    "transformer_d_model": 64,
                    "transformer_nhead": 4,
                    "transformer_num_layers": 2,
                    "transformer_dim_feedforward": 128,
                    "transformer_dropout": 0.1
                },
                "loss": {"name": "pinball", "pinball_quantile": 0.35},
                "training": {"epochs": 30, "batch_size": 64, "learning_rate": 0.001, "early_stopping_patience": 8}
            }
        }
    ]

    benchmark_records = []

    for exp in experiments_to_run:
        tag = exp["tag"]
        name = exp["name"]
        ckpt_dir = repo_root / exp["checkpoint_sub"]
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[TRAINING] Starting {name}...")
        cfg = cm.load_config(overrides=exp["overrides"])
        cfg.experiment_id = f"exp_p2_{tag}"
        cfg.profile_name = exp["profile"]

        # Save config snapshot
        snap_path = cm.save_snapshot(cfg, res_dir / "snapshots")

        # Train model
        engine = TrainingEngine(cfg, checkpoint_dir=ckpt_dir)
        train_res = engine.train(X_train, y_train, X_val, y_val)
        model = train_res["model"]

        print(f" -> Trained in {train_res['training_duration_seconds']}s (Best Epoch: {train_res['best_epoch']})")
        print(f" -> Validation NASA Score: {train_res['best_val_nasa']:.2f} | Val MAE: {train_res['final_val_mae']:.2f}")

        # Official Test Set Evaluation (100 engines)
        print(f"[EVALUATION] Evaluating {tag} on Official NASA FD001 Test Set...")
        eval_res = evaluate_official_test_set(
            model=model,
            subset="FD001",
            window_length=seq_len,
            rul_clip=125,
            is_3d=True
        )

        raw_nasa = eval_res["metrics_raw"]["NASA_score"]
        clip_nasa = eval_res["metrics_clipped"]["NASA_score"]
        raw_mae = eval_res["metrics_raw"]["MAE"]
        raw_rmse = eval_res["metrics_raw"]["RMSE"]

        # Calculate Mean Bias and Overestimates
        diffs = [e["error_raw"] for e in eval_res["engine_details"]]
        mean_bias = float(np.mean(diffs))
        overestimates = sum(1 for d in diffs if d > 0)

        print(f" -> Official Test Raw NASA Score:    {raw_nasa:.2f}")
        print(f" -> Official Test Clipped NASA Score: {clip_nasa:.2f}")
        print(f" -> Official Test MAE:               {raw_mae:.2f}")
        print(f" -> Mean Prediction Bias:           {mean_bias:+.2f} cycles")
        print(f" -> Overestimates:                  {overestimates} / 100 engines")

        # Log to ExperimentTracker
        tracker.log_experiment(
            config=cfg,
            training_res=train_res,
            test_res=eval_res,
            config_snapshot_path=str(snap_path),
            notes=f"Phase 2 {tag} architecture training with Pinball q=0.35"
        )

        # Register in ModelRegistry
        registry.register_model(
            tag=f"phase2_{tag}",
            name=name,
            architecture=exp["overrides"]["model"]["architecture"],
            checkpoint_path=train_res["checkpoint_path"],
            config=cfg,
            metrics={
                "NASA_score_raw": raw_nasa,
                "NASA_score_clipped": clip_nasa,
                "MAE": raw_mae,
                "RMSE": raw_rmse,
                "Mean_Bias": round(mean_bias, 2),
                "Overestimates": overestimates,
                "Val_NASA_score": round(train_res["best_val_nasa"], 2)
            },
            input_dim=input_dim,
            seq_len=seq_len
        )

        # Save test engine predictions CSV
        pred_df = pd.DataFrame(eval_res["engine_details"])
        pred_df.to_csv(res_dir / f"test_predictions_{tag}.csv", index=False)

        benchmark_records.append({
            "Architecture": name,
            "Tag": tag,
            "Raw NASA Score": round(raw_nasa, 2),
            "Clipped NASA Score": round(clip_nasa, 2),
            "MAE": round(raw_mae, 2),
            "RMSE": round(raw_rmse, 2),
            "Mean Bias (cycles)": round(mean_bias, 2),
            "Overestimates": f"{overestimates}/100",
            "Training Time (s)": train_res["training_duration_seconds"],
            "Checkpoint": str(train_res["checkpoint_path"])
        })

    # Summary table
    bench_df = pd.DataFrame(benchmark_records)
    bench_df.to_csv(res_dir / "phase2_model_comparison.csv", index=False)
    tracker.export_csv(res_dir / "experiments_history.csv")

    print("\n" + "=" * 78)
    print(" PHASE 2 OFFICIAL FD001 TEST SET BENCHMARK RESULTS ")
    print("=" * 78)
    print(bench_df[["Architecture", "Raw NASA Score", "Clipped NASA Score", "MAE", "RMSE", "Mean Bias (cycles)"]].to_string(index=False))
    print("=" * 78)

    return bench_df


if __name__ == "__main__":
    train_and_evaluate_phase2()
