"""Transfer Learning and Domain Adaptation Engine (JETWISE AI Phase 6).

Implements transfer learning strategies across turbofan operating environments:
1. Zero-Shot Transfer (Source Model evaluated directly on Target Domain).
2. Frozen Backbone Fine-Tuning (Feature extractor frozen; head trained on target).
3. Full Fine-Tuning with Differential Learning Rates (Small LR on backbone, higher on head).
4. Cold-Start Training from Scratch on Target Domain.

Supports few-shot adaptation where only a small fraction of target engines is available.
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
from typing import Dict, List, Optional, Any, Tuple, Literal
import os
import json
import time
import pathlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.data_loader import load_dataset, load_rul, SENSOR_COLS
from src.evaluation import compute_metrics, compute_nasa_score
from src.train_lstm import LSTMPredictor
from src.regime_normalizer import RegimeAdaptiveScaler


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


def prepare_target_tensors(
    subset: str,
    feature_cols: List[str],
    scaler: Optional[RegimeAdaptiveScaler] = None,
    window_length: int = 30,
    target_engine_fraction: float = 1.0,
    seed: int = 42
) -> Tuple[TensorDataset, np.ndarray, np.ndarray]:
    """Load and process target dataset into PyTorch tensors with zero leakage."""
    train_df = load_dataset(subset, kind="train")
    test_df = load_dataset(subset, kind="test")
    test_rul = load_rul(subset).values.astype(np.float32)

    # If few-shot fraction < 1.0, subsample target training engines
    all_target_eids = sorted(train_df["engine_id"].unique())
    if target_engine_fraction < 1.0:
        rng = np.random.RandomState(seed)
        n_keep = max(5, int(len(all_target_eids) * target_engine_fraction))
        keep_eids = set(rng.choice(all_target_eids, size=n_keep, replace=False))
        train_df = train_df[train_df["engine_id"].isin(keep_eids)].copy()

    # Apply regime adaptive scaling if multi-regime, else condition-wise
    if scaler is None:
        scaler = RegimeAdaptiveScaler(scaler_type="standard", n_clusters=6 if subset in ("FD002", "FD004") else 1)
        scaler.fit(train_df, feature_cols)

    norm_train = scaler.transform(train_df)
    norm_test = scaler.transform(test_df)

    # Extract sequence windows
    X_train_list, y_train_list = [], []
    for eid in sorted(norm_train["engine_id"].unique()):
        eng = norm_train[norm_train["engine_id"] == eid].sort_values("cycle").reset_index(drop=True)
        vals = eng[feature_cols].values.astype(np.float32)
        n_c = len(vals)
        if n_c < window_length:
            continue
        for t in range(window_length, n_c + 1):
            X_train_list.append(vals[t - window_length:t])
            y_train_list.append(min(n_c - eng["cycle"].iloc[t - 1], 125))

    # Test windows: single final window per test engine
    X_test_list = []
    for eid in sorted(norm_test["engine_id"].unique()):
        eng = norm_test[norm_test["engine_id"] == eid].sort_values("cycle").reset_index(drop=True)
        vals = eng[feature_cols].values.astype(np.float32)
        n_c = len(vals)
        if n_c >= window_length:
            X_test_list.append(vals[-window_length:])
        else:
            pad_count = window_length - n_c
            pad = np.repeat(vals[0:1], pad_count, axis=0)
            X_test_list.append(np.vstack([pad, vals]))

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.float32)
    X_test = np.array(X_test_list, dtype=np.float32)

    # Subsample training windows if very large
    if len(X_train) > 15000:
        sub_idx = np.arange(0, len(X_train), 2)
        X_train = X_train[sub_idx]
        y_train = y_train[sub_idx]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    return train_ds, X_test, test_rul


class TransferLearningEngine:
    """Orchestrates source model transfer and adaptation to target fleets."""

    def __init__(
        self,
        source_model_path: Optional[str] = None,
        source_subset: str = "FD001",
        feature_cols: Optional[List[str]] = None
    ):
        self.source_subset = source_subset
        self.project_root = PROJECT_ROOT
        self.feature_cols = feature_cols or [
            "sensor_2", "sensor_3", "sensor_4", "sensor_5", "sensor_6", "sensor_7", "sensor_8",
            "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
            "sensor_16", "sensor_17", "sensor_20", "sensor_21"
        ]

        if source_model_path is None:
            # Default to best Pinball LSTM or baseline LSTM
            cand = [
                self.project_root / "models" / "lstm_asymmetric" / "pinball_q35_lstm.pt",
                self.project_root / "models" / "lstm" / "lstm_model.pt"
            ]
            for p in cand:
                if p.exists():
                    source_model_path = str(p)
                    break

        self.source_model_path = source_model_path

    def _load_source_model(self) -> LSTMPredictor:
        """Instantiate LSTMPredictor with weights from source checkpoint."""
        ckpt = torch.load(self.source_model_path, map_location="cpu", weights_only=False)
        input_dim = ckpt.get("input_dim", len(self.feature_cols)) if isinstance(ckpt, dict) else len(self.feature_cols)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

        model = LSTMPredictor(input_dim=input_dim, hidden_dim1=64, hidden_dim2=32)
        # Adapt state_dict if input dimension differs
        curr_state = model.state_dict()
        filtered_state = {k: v for k, v in state_dict.items() if k in curr_state and v.shape == curr_state[k].shape}
        curr_state.update(filtered_state)
        model.load_state_dict(curr_state)
        return model

    def evaluate_model(self, model: nn.Module, X_test: np.ndarray, test_rul: np.ndarray) -> Dict[str, Any]:
        """Compute metrics on target test set."""
        model.eval()
        with torch.no_grad():
            t_in = torch.from_numpy(X_test)
            preds = model(t_in).squeeze().numpy()

        preds_clipped = np.clip(preds, 0, 125)
        raw_m = compute_metrics(test_rul, preds)
        clip_m = compute_metrics(np.clip(test_rul, 0, 125), preds_clipped)
        return {
            "metrics_raw": raw_m,
            "metrics_clipped": clip_m
        }

    def run_transfer_strategy(
        self,
        target_subset: str,
        strategy: Literal["zero_shot", "freeze_backbone", "full_fine_tune", "from_scratch"],
        target_engine_fraction: float = 0.5,
        epochs: int = 12,
        batch_size: int = 64,
        seed: int = 42
    ) -> Dict[str, Any]:
        """Execute specified domain transfer strategy."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        t0 = time.time()
        train_ds, X_test, test_rul = prepare_target_tensors(
            subset=target_subset,
            feature_cols=self.feature_cols,
            target_engine_fraction=target_engine_fraction,
            seed=seed
        )

        if strategy == "zero_shot":
            model = self._load_source_model()
            duration = time.time() - t0
            eval_res = self.evaluate_model(model, X_test, test_rul)
            eval_res["duration_seconds"] = round(duration, 2)
            eval_res["strategy"] = strategy
            eval_res["target_subset"] = target_subset
            return eval_res

        elif strategy == "from_scratch":
            model = LSTMPredictor(input_dim=len(self.feature_cols), hidden_dim1=64, hidden_dim2=32)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        elif strategy == "freeze_backbone":
            model = self._load_source_model()
            # Freeze LSTM layer 1 & layer 2
            for param in model.lstm1.parameters():
                param.requires_grad = False
            for param in model.lstm2.parameters():
                param.requires_grad = False
            # Train only fully connected head
            optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

        elif strategy == "full_fine_tune":
            model = self._load_source_model()
            # Differential learning rates: 1e-4 on backbone, 1e-3 on head
            optimizer = torch.optim.Adam([
                {"params": model.lstm1.parameters(), "lr": 1e-4},
                {"params": model.lstm2.parameters(), "lr": 1e-4},
                {"params": model.fc.parameters(), "lr": 1e-3}
            ])

        # Training loop on target domain
        criterion = nn.MSELoss()
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        model.train()
        for epoch in range(epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                out = model(bx).squeeze()
                loss = criterion(out, by)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        duration = time.time() - t0
        eval_res = self.evaluate_model(model, X_test, test_rul)
        eval_res["duration_seconds"] = round(duration, 2)
        eval_res["strategy"] = strategy
        eval_res["target_subset"] = target_subset
        eval_res["target_engine_fraction"] = target_engine_fraction

        # Save checkpoint
        save_dir = self.project_root / "models" / "transfer"
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / f"transfer_{self.source_subset}_to_{target_subset}_{strategy}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "source_subset": self.source_subset,
            "target_subset": target_subset,
            "strategy": strategy,
            "feature_cols": self.feature_cols
        }, ckpt_path)
        eval_res["checkpoint"] = str(ckpt_path.relative_to(self.project_root))

        return eval_res

    def run_all_transfer_experiments(self) -> Dict[str, Any]:
        """Execute full transfer learning matrix across target domains."""
        targets = ["FD003", "FD002", "FD004"]
        strategies = ["zero_shot", "freeze_backbone", "full_fine_tune", "from_scratch"]

        results = {}
        for target in targets:
            print(f"\n=======================================================")
            print(f"RUNNING TRANSFER EXPERIMENTS: {self.source_subset} -> {target}")
            print(f"=======================================================")
            results[target] = {}
            for strat in strategies:
                print(f"  [>] Strategy: {strat}...")
                res = self.run_transfer_strategy(
                    target_subset=target,
                    strategy=strat,
                    target_engine_fraction=0.5,
                    epochs=10
                )
                raw = res["metrics_raw"]
                clip = res["metrics_clipped"]
                print(f"      Result: Raw MAE={raw['MAE']:.2f}, NASA={raw['NASA_score']:.2f} | Clip MAE={clip['MAE']:.2f}")
                results[target][strat] = res

        out_path = self.project_root / "results" / "phase6_transfer_learning_experiments.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"\n[x] All transfer learning experiments saved to {out_path}")
        return results


if __name__ == "__main__":
    engine = TransferLearningEngine()
    engine.run_all_transfer_experiments()
