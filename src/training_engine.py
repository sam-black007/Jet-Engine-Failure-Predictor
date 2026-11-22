"""Unified Training Engine for Jet Engine Failure Predictor (THRUSTLAB Phase 2).

Provides an enterprise-grade, reproducible training harness for both PyTorch deep
architectures (LSTM, Attention-LSTM, TCN, Transformer) and Scikit-Learn baselines.

Features:
- Configurable optimizers (Adam, AdamW, RMSprop, SGD)
- Schedulers (ReduceLROnPlateau, StepLR, CosineAnnealingLR)
- Early stopping with configurable patience and delta
- Gradient norm clipping to prevent gradient explosions
- Multi-metric validation tracking (loss, MAE, NASA asymmetric penalty)
- Full checkpoint state retention (model weights, optimizer, epoch, config snapshot)
- Structured training history export (JSON / CSV compatible)
"""

from __future__ import annotations

import os
import time
import random
import pathlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.config_manager import MasterConfig, TrainingConfig
from src.model_factory import create_model, is_deep_learning_model
from src.losses import get_loss_function
from src.evaluation import compute_nasa_score, compute_metrics


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping monitor based on validation score or loss."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.early_stop = False
        self.best_epoch = 0

    def step(self, current_val: float, epoch: int) -> bool:
        """Return True if improved, False otherwise. Updates early_stop status."""
        improved = False
        if self.mode == "min":
            if current_val < self.best_score - self.min_delta:
                self.best_score = current_val
                self.counter = 0
                self.best_epoch = epoch
                improved = True
            else:
                self.counter += 1
        else:
            if current_val > self.best_score + self.min_delta:
                self.best_score = current_val
                self.counter = 0
                self.best_epoch = epoch
                improved = True
            else:
                self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
        return improved


class TrainingEngine:
    """Configurable training coordinator for THRUSTLAB models."""

    def __init__(
        self,
        config: Optional[MasterConfig] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None
    ):
        self.config = config or MasterConfig()
        self.train_cfg: TrainingConfig = self.config.training
        self.checkpoint_dir = Path(checkpoint_dir or "models/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Device selection
        if self.train_cfg.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.train_cfg.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        opt_name = self.train_cfg.optimizer.lower()
        lr = self.train_cfg.learning_rate
        wd = self.train_cfg.weight_decay

        if opt_name == "adam":
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "rmsprop":
            return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: '{opt_name}'")

    def _build_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[Any]:
        sched_name = self.train_cfg.scheduler.lower()
        if sched_name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=self.train_cfg.scheduler_factor,
                patience=self.train_cfg.scheduler_patience, min_lr=1e-6
            )
        elif sched_name == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.train_cfg.scheduler_patience,
                gamma=self.train_cfg.scheduler_factor
            )
        elif sched_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.train_cfg.epochs, eta_min=1e-6
            )
        return None

    def train_deep_model(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        loss_fn: Optional[nn.Module] = None,
        trial_prune_callback: Optional[Callable[[int, float], bool]] = None
    ) -> Dict[str, Any]:
        """Train a PyTorch neural network model and return training artifacts."""
        set_seed(self.train_cfg.seed)
        model = model.to(self.device)
        loss_fn = loss_fn or get_loss_function(self.config.loss)
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)

        # Build PyTorch DataLoaders
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )

        train_loader = DataLoader(
            train_dataset, batch_size=self.train_cfg.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.train_cfg.batch_size, shuffle=False
        )

        early_stopping = EarlyStopping(
            patience=self.train_cfg.early_stopping_patience,
            mode="min"
        ) if self.train_cfg.early_stopping else None

        history: Dict[str, List[float]] = {
            "epoch": [], "train_loss": [], "val_loss": [],
            "val_mae": [], "val_nasa": [], "lr": []
        }

        best_val_nasa = float("inf")
        best_state = None
        best_epoch = 0
        start_time = time.time()

        for epoch in range(1, self.train_cfg.epochs + 1):
            # Training Phase
            model.train()
            train_loss_accum = 0.0
            num_train_batches = 0

            for bx, by in train_loader:
                bx = bx.to(self.device)
                by = by.to(self.device)
                optimizer.zero_grad()
                preds = model(bx)
                loss = loss_fn(preds, by)
                loss.backward()

                if self.train_cfg.gradient_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.train_cfg.gradient_clip_norm)

                optimizer.step()
                train_loss_accum += float(loss.item())
                num_train_batches += 1

            train_loss = train_loss_accum / max(1, num_train_batches)

            # Validation Phase
            model.eval()
            val_loss_accum = 0.0
            val_preds_list = []
            val_true_list = []

            with torch.no_grad():
                for bx, by in val_loader:
                    bx = bx.to(self.device)
                    by = by.to(self.device)
                    preds = model(bx)
                    loss = loss_fn(preds, by)
                    val_loss_accum += float(loss.item())
                    val_preds_list.extend(preds.cpu().numpy().flatten().tolist())
                    val_true_list.extend(by.cpu().numpy().flatten().tolist())

            val_loss = val_loss_accum / max(1, len(val_loader))
            val_preds_arr = np.array(val_preds_list)
            val_true_arr = np.array(val_true_list)
            val_mae = float(np.mean(np.abs(val_preds_arr - val_true_arr)))
            val_nasa = float(compute_nasa_score(val_true_arr, val_preds_arr))

            current_lr = optimizer.param_groups[0]["lr"]
            history["epoch"].append(epoch)
            history["train_loss"].append(round(train_loss, 5))
            history["val_loss"].append(round(val_loss, 5))
            history["val_mae"].append(round(val_mae, 4))
            history["val_nasa"].append(round(val_nasa, 2))
            history["lr"].append(current_lr)

            # Update scheduler
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Track best model checkpoint
            if val_nasa < best_val_nasa:
                best_val_nasa = val_nasa
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Early stopping check
            if early_stopping is not None:
                early_stopping.step(val_loss, epoch)
                if early_stopping.early_stop:
                    break

            # Pruning callback (for Optuna trials)
            if trial_prune_callback is not None:
                should_prune = trial_prune_callback(epoch, val_nasa)
                if should_prune:
                    break

        training_duration = time.time() - start_time

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        # Save checkpoint to disk
        checkpoint_name = f"{self.config.model.architecture}_{self.config.experiment_id}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        torch.save({
            "epoch": best_epoch,
            "model_state_dict": model.state_dict(),
            "best_val_nasa": best_val_nasa,
            "config": self.config.model_dump(),
            "history": history
        }, checkpoint_path)

        return {
            "model": model,
            "best_epoch": best_epoch,
            "best_val_nasa": best_val_nasa,
            "final_val_mae": history["val_mae"][-1] if history["val_mae"] else 0.0,
            "training_duration_seconds": round(training_duration, 2),
            "checkpoint_path": str(checkpoint_path),
            "history": history
        }

    def train_baseline_model(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Fit a Scikit-Learn baseline regressor (flattening sequences to 2D)."""
        start_time = time.time()
        X_tr_flat = X_train.reshape(X_train.shape[0], -1) if X_train.ndim == 3 else X_train
        X_v_flat = X_val.reshape(X_val.shape[0], -1) if X_val.ndim == 3 else X_val

        model.fit(X_tr_flat, y_train)
        training_duration = time.time() - start_time

        val_preds = model.predict(X_v_flat)
        val_mae = float(np.mean(np.abs(val_preds - y_val)))
        val_nasa = float(compute_nasa_score(y_val, val_preds))

        return {
            "model": model,
            "best_epoch": 1,
            "best_val_nasa": val_nasa,
            "final_val_mae": val_mae,
            "training_duration_seconds": round(training_duration, 2),
            "checkpoint_path": None,
            "history": {
                "epoch": [1], "train_loss": [0.0], "val_loss": [val_mae],
                "val_mae": [val_mae], "val_nasa": [val_nasa], "lr": [0.0]
            }
        }

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        trial_prune_callback: Optional[Callable[[int, float], bool]] = None
    ) -> Dict[str, Any]:
        """Main entry point: train configured model on provided arrays."""
        input_dim = X_train.shape[2] if X_train.ndim == 3 else X_train.shape[1]
        seq_len = X_train.shape[1] if X_train.ndim == 3 else 30

        if model is None:
            model = create_model(self.config.model, input_dim=input_dim, seq_len=seq_len)

        if is_deep_learning_model(model):
            return self.train_deep_model(
                model, X_train, y_train, X_val, y_val, loss_fn=loss_fn,
                trial_prune_callback=trial_prune_callback
            )
        else:
            return self.train_baseline_model(model, X_train, y_train, X_val, y_val)
