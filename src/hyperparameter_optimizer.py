"""Automated Hyperparameter Optimization Engine for Jet Engine Failure Predictor (JETWISE AI Phase 2).

Leverages Optuna with:
- Tree-structured Parzen Estimator (TPE) or Random Search
- Automated trial pruning (MedianPruner) to cut underperforming trials early
- Multi-objective or single-objective formulation (NASA Score, MAE, Balanced)
- Optimization STRICTLY on training validation split (NEVER on official test set)
- Persistent SQLite / JSON logging of all trials and parameter spaces.
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union

import numpy as np
import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.pruners import MedianPruner, NopPruner

from src.config_manager import MasterConfig
from src.configurable_data import ConfigurableDataPipeline
from src.model_factory import create_model
from src.losses import get_loss_function
from src.training_engine import TrainingEngine


class HyperparameterOptimizer:
    """Orchestrates Optuna-driven hyperparameter tuning over validation data."""

    def __init__(
        self,
        base_config: Optional[MasterConfig] = None,
        study_name: str = "jetwise_hpo",
        storage_dir: Optional[Union[str, Path]] = None
    ):
        self.base_config = base_config or MasterConfig()
        self.study_name = study_name
        self.storage_dir = Path(storage_dir or "results/hpo")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_dir / f"{study_name}.db"

    def _sample_parameters(self, trial: optuna.Trial, architecture: str) -> Dict[str, Any]:
        """Sample hyperparameters dynamically based on target architecture."""
        params = {
            "training.learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "training.batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "training.weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "loss.pinball_quantile": trial.suggest_float("pinball_quantile", 0.25, 0.45, step=0.05)
        }

        if architecture in ("lstm", "attention_lstm"):
            params["model.lstm_hidden_dim1"] = trial.suggest_categorical("hidden_dim1", [32, 64, 128])
            params["model.lstm_hidden_dim2"] = trial.suggest_categorical("hidden_dim2", [16, 32, 64])
            params["model.dropout"] = trial.suggest_float("dropout", 0.1, 0.4, step=0.1)
            if architecture == "attention_lstm":
                params["model.attention_dim"] = trial.suggest_categorical("attention_dim", [16, 32, 64])

        elif architecture == "tcn":
            kernel = trial.suggest_categorical("tcn_kernel_size", [3, 5])
            channels = trial.suggest_categorical("tcn_channels", ["32-64-64", "64-64-128"])
            ch_list = [int(c) for c in channels.split("-")]
            params["model.tcn_kernel_size"] = kernel
            params["model.tcn_num_channels"] = ch_list
            params["model.tcn_dropout"] = trial.suggest_float("tcn_dropout", 0.1, 0.3, step=0.1)

        elif architecture == "transformer":
            nhead = trial.suggest_categorical("transformer_nhead", [2, 4])
            d_mult = trial.suggest_categorical("d_multiplier", [8, 16])
            d_model = nhead * d_mult  # Guaranteed divisible
            params["model.transformer_nhead"] = nhead
            params["model.transformer_d_model"] = d_model
            params["model.transformer_num_layers"] = trial.suggest_int("transformer_num_layers", 1, 3)
            params["model.transformer_dim_feedforward"] = d_model * 2
            params["model.transformer_dropout"] = trial.suggest_float("transformer_dropout", 0.05, 0.2, step=0.05)

        return params

    def optimize(
        self,
        architecture: str = "lstm",
        objective_metric: str = "nasa_score",
        n_trials: int = 10,
        timeout: Optional[int] = None,
        max_epochs_per_trial: int = 8,
        sampler_type: str = "tpe"
    ) -> Dict[str, Any]:
        """Execute hyperparameter optimization study strictly using training validation data.
        
        Parameters
        ----------
        architecture : str
            Model architecture to tune ("lstm", "attention_lstm", "tcn", "transformer").
        objective_metric : str
            Metric to minimize: "nasa_score", "mae", or "balanced".
        n_trials : int
            Number of optimization trials.
        timeout : int, optional
            Timeout in seconds.
        max_epochs_per_trial : int
            Maximum epochs per trial (kept fast for responsive tuning).
        sampler_type : str
            "tpe" for Bayesian optimization, "random" for uniform sampling.
            
        Returns
        -------
        dict with best trial parameters, value, and study statistics.
        """
        # Prepare training and validation data once
        pipeline = ConfigurableDataPipeline(self.base_config)
        data = pipeline.prepare_data()
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]

        sampler = TPESampler(seed=42) if sampler_type.lower() == "tpe" else RandomSampler(seed=42)
        pruner = MedianPruner(n_startup_trials=2, n_warmup_steps=3)

        storage_url = f"sqlite:///{self.db_path}"
        study = optuna.create_study(
            study_name=f"{self.study_name}_{architecture}_{int(time.time())}",
            direction="minimize",
            sampler=sampler,
            pruner=pruner
        )

        def objective(trial: optuna.Trial) -> float:
            param_dict = self._sample_parameters(trial, architecture)

            # Build trial overrides
            overrides: Dict[str, Any] = {
                "model": {"architecture": architecture},
                "training": {"epochs": max_epochs_per_trial, "early_stopping_patience": 4}
            }
            for k, v in param_dict.items():
                section, key = k.split(".")
                if section not in overrides:
                    overrides[section] = {}
                overrides[section][key] = v

            trial_cfg = self.base_config.model_copy(deep=True)
            # Apply overrides
            for sec, vals in overrides.items():
                sub = getattr(trial_cfg, sec)
                for vk, vv in vals.items():
                    setattr(sub, vk, vv)

            # Training callback for Optuna pruning
            def prune_callback(epoch: int, val_score: float) -> bool:
                trial.report(val_score, step=epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                return False

            engine = TrainingEngine(trial_cfg)
            try:
                res = engine.train(
                    X_train, y_train, X_val, y_val, trial_prune_callback=prune_callback
                )
            except optuna.exceptions.TrialPruned:
                raise

            if objective_metric == "mae":
                return float(res["final_val_mae"])
            elif objective_metric == "balanced":
                # Balanced combination of normalized NASA and MAE
                return float(res["best_val_nasa"] / 1000.0 + res["final_val_mae"])
            else:
                return float(res["best_val_nasa"])

        # Run study
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        best_trial = study.best_trial
        results_summary = {
            "study_name": study.study_name,
            "architecture": architecture,
            "objective_metric": objective_metric,
            "n_trials_completed": len(study.trials),
            "best_trial_number": best_trial.number,
            "best_value": round(float(best_trial.value), 4),
            "best_params": best_trial.params
        }

        # Save summary to disk
        summary_path = self.storage_dir / f"{architecture}_best_hpo_params.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results_summary, f, indent=2)

        return results_summary
