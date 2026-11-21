"""Training Pipeline for Phase 1: Asymmetric Loss Formulations for Deep LSTM.

Executes:
1. Experiment A: NASA Asymmetric Exponential Loss (Log-Mean-Exp)
2. Experiment B: Asymmetric Pinball (Quantile) Loss (q=0.40 primary, plus q=0.35, q=0.45)
3. Seed variability analysis across multiple random seeds (42, 100, 2026)

Monitors validation NASA score, MAE, RMSE, overestimation rate, and prediction bias.
Saves best checkpoints under models/lstm_asymmetric/ without modifying baseline models.
Exports training history and configs to models/lstm_asymmetric/ and results/phase1/.
"""

import os
import sys
import json
import time
import random
import pathlib
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Project setup
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.losses import NASAAsymmetricLoss, AsymmetricPinballLoss
from src.evaluation import compute_nasa_score, compute_metrics
from src.train_lstm import LSTMPredictor

DATASET_DIR = PROJECT_ROOT / "dataset" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ASYMM_MODELS_DIR = MODELS_DIR / "lstm_asymmetric"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase1"
ASYMM_MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_single_model(
    loss_name: str,
    criterion: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    max_epochs: int = 35,
    patience: int = 8,
    save_path: pathlib.Path = None
) -> Dict[str, Any]:
    """Train a single LSTM with specified loss function and track full epoch metrics."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMPredictor(input_dim=X_train.shape[2]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=False
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_rmse": [],
        "val_nasa_score": [],
        "val_overestimation_count": [],
        "val_underestimation_count": [],
        "val_mean_bias": []
    }

    best_val_nasa = float("inf")
    best_epoch = -1
    best_state_dict = None
    patience_counter = 0

    start_time = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_total = 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)

            if torch.isnan(loss) or torch.isinf(loss):
                raise FloatingPointError(f"NaN/Inf loss detected at epoch {epoch} with loss {loss_name}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_total += loss.item() * batch_x.size(0)

        train_loss = train_loss_total / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss_total = 0.0
        val_preds_list = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss_total += loss.item() * batch_x.size(0)
                val_preds_list.append(preds.cpu().numpy())

        val_loss = val_loss_total / len(val_loader.dataset)
        val_preds = np.concatenate(val_preds_list, axis=0)

        scheduler.step(val_loss)

        # Validation error metrics
        residuals = val_preds - y_val
        val_metrics = compute_metrics(y_val, val_preds)
        val_mae = val_metrics["MAE"]
        val_rmse = val_metrics["RMSE"]
        val_nasa = val_metrics["NASA_score"]
        over_count = int(np.sum(residuals > 0))
        under_count = int(np.sum(residuals <= 0))
        mean_bias = float(np.mean(residuals))

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_mae"].append(round(val_mae, 4))
        history["val_rmse"].append(round(val_rmse, 4))
        history["val_nasa_score"].append(round(val_nasa, 4))
        history["val_overestimation_count"].append(over_count)
        history["val_underestimation_count"].append(under_count)
        history["val_mean_bias"].append(round(mean_bias, 4))

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{max_epochs:02d} | Train Loss: {train_loss:.4f} | Val NASA: {val_nasa:9.1f} | Val MAE: {val_mae:5.2f} | Bias: {mean_bias:+5.2f}")

        # Checkpoint based on Validation NASA Score
        if val_nasa < best_val_nasa:
            best_val_nasa = val_nasa
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered at epoch {epoch}. Best Val NASA: {best_val_nasa:.2f} (Epoch {best_epoch})")
                break

    training_time = time.perf_counter() - start_time

    # Load best checkpoint
    model.load_state_dict(best_state_dict)
    model.eval()

    # Measure inference latency on 100 validation samples
    sample_tensor = torch.tensor(X_val[:100], dtype=torch.float32).to(device)
    with torch.no_grad():
        # Warmup
        _ = model(sample_tensor)
        t0 = time.perf_counter()
        for _ in range(50):
            _ = model(sample_tensor)
        t1 = time.perf_counter()
    latency_per_100 = ((t1 - t0) / 50.0) * 1000.0  # ms
    latency_single = latency_per_100 / 100.0

    # Save checkpoint if path provided
    if save_path:
        torch.save({
            "model_state_dict": best_state_dict,
            "input_dim": X_train.shape[2],
            "hidden_dim1": 64,
            "hidden_dim2": 32,
            "dropout_rate": 0.2,
            "best_epoch": best_epoch,
            "best_val_nasa": best_val_nasa,
            "loss_name": loss_name,
            "seed": seed,
            "training_time_sec": round(training_time, 2),
            "inference_latency_ms": round(latency_single, 3)
        }, save_path)
        print(f"  Saved best checkpoint to {save_path.name} (Val NASA: {best_val_nasa:.2f})")

    return {
        "loss_name": loss_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_nasa": best_val_nasa,
        "final_val_mae": history["val_mae"][best_epoch - 1],
        "final_val_rmse": history["val_rmse"][best_epoch - 1],
        "final_mean_bias": history["val_mean_bias"][best_epoch - 1],
        "final_overestimation_pct": (history["val_overestimation_count"][best_epoch - 1] / len(y_val)) * 100.0,
        "training_time_sec": round(training_time, 2),
        "latency_single_ms": round(latency_single, 3),
        "history": history,
        "model": model
    }


def run_phase1_experiments():
    print("=" * 80)
    print("EXECUTING PHASE 1: ASYMMETRIC LOSS EXPERIMENTS FOR DEEP LSTM")
    print("=" * 80)

    # 1. Load preprocessed training and validation sequences
    X_train = np.load(DATASET_DIR / "X_train.npy").astype(np.float32)
    y_train = np.load(DATASET_DIR / "y_train.npy").astype(np.float32)
    X_val = np.load(DATASET_DIR / "X_val.npy").astype(np.float32)
    y_val = np.load(DATASET_DIR / "y_val.npy").astype(np.float32)

    print(f"Loaded training sequences: {X_train.shape}, labels: {y_train.shape}")
    print(f"Loaded validation sequences: {X_val.shape}, labels: {y_val.shape}")

    all_experiments = {}

    # -------------------------------------------------------------------------
    # Experiment A: NASA Asymmetric Exponential Loss (Primary)
    # -------------------------------------------------------------------------
    print("\n[EXPERIMENT A] Training Deep LSTM with NASA Asymmetric Exponential Loss (Log-Mean-Exp)...")
    exp_a = train_single_model(
        loss_name="nasa_asymmetric",
        criterion=NASAAsymmetricLoss(late_penalty=10.0, early_penalty=13.0),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        seed=42,
        save_path=ASYMM_MODELS_DIR / "nasa_asymmetric_lstm.pt"
    )
    all_experiments["nasa_asymmetric"] = exp_a

    # -------------------------------------------------------------------------
    # Experiment B: Asymmetric Pinball / Quantile Loss (q=0.40 Primary)
    # -------------------------------------------------------------------------
    print("\n[EXPERIMENT B1] Training Deep LSTM with Asymmetric Pinball Loss (q=0.40)...")
    exp_b1 = train_single_model(
        loss_name="pinball_q40",
        criterion=AsymmetricPinballLoss(quantile=0.40),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        seed=42,
        save_path=ASYMM_MODELS_DIR / "pinball_lstm.pt"
    )
    all_experiments["pinball_q40"] = exp_b1

    # Additional Quantile Explorations (q=0.35, q=0.45)
    print("\n[EXPERIMENT B2] Training Deep LSTM with Asymmetric Pinball Loss (q=0.35)...")
    exp_b2 = train_single_model(
        loss_name="pinball_q35",
        criterion=AsymmetricPinballLoss(quantile=0.35),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        seed=42,
        save_path=ASYMM_MODELS_DIR / "pinball_q35_lstm.pt"
    )
    all_experiments["pinball_q35"] = exp_b2

    print("\n[EXPERIMENT B3] Training Deep LSTM with Asymmetric Pinball Loss (q=0.45)...")
    exp_b3 = train_single_model(
        loss_name="pinball_q45",
        criterion=AsymmetricPinballLoss(quantile=0.45),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        seed=42,
        save_path=ASYMM_MODELS_DIR / "pinball_q45_lstm.pt"
    )
    all_experiments["pinball_q45"] = exp_b3

    # -------------------------------------------------------------------------
    # Multi-Seed Stability Check for Primary Asymmetric Model (Seeds 100, 2026)
    # -------------------------------------------------------------------------
    print("\n[SEED STABILITY AUDIT] Running NASA Asymmetric Loss across additional seeds...")
    seed_runs = [exp_a["best_val_nasa"]]
    for test_seed in [100, 2026]:
        print(f"  Training with Seed {test_seed}...")
        res_seed = train_single_model(
            loss_name=f"nasa_asymmetric_seed_{test_seed}",
            criterion=NASAAsymmetricLoss(),
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            seed=test_seed,
            save_path=None
        )
        seed_runs.append(res_seed["best_val_nasa"])

    seed_mean = float(np.mean(seed_runs))
    seed_std = float(np.std(seed_runs))
    print(f"  Multi-seed Val NASA Scores: {seed_runs} -> Mean: {seed_mean:.2f} +/- {seed_std:.2f}")

    # -------------------------------------------------------------------------
    # Save Training Config and History
    # -------------------------------------------------------------------------
    config_record = {
        "architecture": {
            "model_type": "LSTMPredictor",
            "input_dim": 17,
            "hidden_dim1": 64,
            "hidden_dim2": 32,
            "dropout_rate": 0.20,
            "output_dim": 1
        },
        "training_params": {
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "batch_size": 64,
            "max_epochs": 35,
            "patience": 8,
            "gradient_clipping": 1.0
        },
        "experiments_summary": {
            k: {
                "loss_name": v["loss_name"],
                "best_epoch": v["best_epoch"],
                "best_val_nasa": v["best_val_nasa"],
                "val_mae": v["final_val_mae"],
                "val_rmse": v["final_val_rmse"],
                "val_bias": v["final_mean_bias"],
                "overestimation_pct": v["final_overestimation_pct"],
                "training_time_sec": v["training_time_sec"],
                "latency_single_ms": v["latency_single_ms"]
            }
            for k, v in all_experiments.items()
        },
        "seed_stability": {
            "seeds_tested": [42, 100, 2026],
            "val_nasa_scores": seed_runs,
            "mean_val_nasa": seed_mean,
            "std_val_nasa": seed_std
        }
    }

    with open(ASYMM_MODELS_DIR / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)
    with open(RESULTS_DIR / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)

    history_export = {
        k: v["history"] for k, v in all_experiments.items()
    }
    with open(ASYMM_MODELS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history_export, f, indent=2)
    with open(RESULTS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history_export, f, indent=2)

    print(f"\nAll Phase 1 training artifacts saved to {ASYMM_MODELS_DIR} and {RESULTS_DIR}.")
    return all_experiments, config_record


if __name__ == "__main__":
    run_phase1_experiments()
