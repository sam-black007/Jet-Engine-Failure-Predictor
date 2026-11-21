"""Cross-Dataset Generalization Experiment Suite.

Evaluates in-domain vs out-of-domain transferability of FD001-trained turbofan
RUL prognostic models across NASA C-MAPSS subsets (FD002, FD003, FD004).

Quantifies:
1. Baseline In-Domain performance on FD001.
2. Multi-Fault Domain Shift on FD003 (HPC + Fan degradation).
3. Multi-Regime Flight Condition Domain Shift on FD002 (6 operating conditions).
4. Compound Complexity Shift on FD004 (6 regimes + 2 fault modes).
"""

import os
import json
import pathlib
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch

from src.train_lstm import LSTMPredictor
from src.evaluation import evaluate_official_test_set, compute_metrics
from src.dataset_registry import DatasetRegistry


class CrossDatasetEvaluator:
    """Evaluates prognostic model robustness across multiple turbofan operating environments."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "lstm",
        metadata_path: Optional[str] = None
    ):
        self.project_root = pathlib.Path(__file__).resolve().parent.parent
        self.model_type = model_type.lower()
        self.metadata_path = metadata_path or str(
            self.project_root / "dataset" / "processed" / "preprocess_metadata.json"
        )

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.sensor_cols = self.meta["selected_sensors"]
        self.sensor_means = np.array([self.meta["sensor_means"][s] for s in self.sensor_cols], dtype=np.float32)
        self.sensor_stds = np.array([self.meta["sensor_stds"][s] for s in self.sensor_cols], dtype=np.float32)

        # Load PyTorch LSTM model
        if model_path is None:
            # Prefer asymmetric or standard Deep LSTM
            cand_paths = [
                self.project_root / "models" / "lstm" / "lstm_model.pt",
                self.project_root / "models" / "lstm_asymmetric" / "nasa_asymmetric_lstm.pt"
            ]
            for p in cand_paths:
                if p.exists():
                    model_path = str(p)
                    break

        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

        self.model_path = model_path
        checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            input_dim = checkpoint.get("input_dim", len(self.sensor_cols))
            hidden_dim1 = checkpoint.get("hidden_dim1", 64)
            hidden_dim2 = checkpoint.get("hidden_dim2", 32)
            dropout_rate = checkpoint.get("dropout_rate", 0.2)
            self.model = LSTMPredictor(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, dropout_rate=dropout_rate)
        else:
            state_dict = checkpoint
            self.model = LSTMPredictor(input_dim=len(self.sensor_cols), hidden_dim1=64, hidden_dim2=32)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def evaluate_dataset(self, subset_name: str) -> Dict[str, Any]:
        """Evaluate FD001-trained model on target subset test fleet."""
        adapter = DatasetRegistry.get_adapter(subset_name)
        test_df = adapter.load_test()
        rul_series = adapter.load_rul()

        eval_result = evaluate_official_test_set(
            model=self.model,
            test_df=test_df,
            rul_series=rul_series,
            subset=subset_name,
            window_length=30,
            rul_clip=125,
            is_3d=True,
            metadata_path=self.metadata_path
        )
        return eval_result

    def run_full_cross_evaluation(self, subsets: Optional[List[str]] = None) -> Dict[str, Any]:
        """Execute cross-evaluation across all C-MAPSS subsets and compute domain shift gaps."""
        if subsets is None:
            subsets = ["FD001", "FD003", "FD002", "FD004"]

        results = {}
        for sub in subsets:
            print(f"Evaluating {sub}...")
            res = self.evaluate_dataset(sub)
            results[sub] = {
                "num_engines": res["num_engines_evaluated"],
                "metrics_raw": res["metrics_raw"],
                "metrics_clipped": res["metrics_clipped"],
                "sample_predictions": res["engine_details"][:5]
            }

        # Compute Domain Shift Degradation relative to FD001 baseline
        base_raw_mae = results["FD001"]["metrics_raw"]["MAE"]
        base_raw_rmse = results["FD001"]["metrics_raw"]["RMSE"]
        base_raw_nasa = results["FD001"]["metrics_raw"]["NASA_score"]

        comparisons = {}
        for sub in subsets:
            raw = results[sub]["metrics_raw"]
            clip = results[sub]["metrics_clipped"]
            comparisons[sub] = {
                "mae_raw": raw["MAE"],
                "mae_clipped": clip["MAE"],
                "rmse": raw["RMSE"],
                "nasa_score": raw["NASA_score"],
                "delta_mae": round(raw["MAE"] - base_raw_mae, 4),
                "delta_rmse": round(raw["RMSE"] - base_raw_rmse, 4),
                "delta_nasa_score": round(raw["NASA_score"] - base_raw_nasa, 4),
                "mae_relative_increase_pct": round(((raw["MAE"] - base_raw_mae) / base_raw_mae) * 100, 2)
            }

        output_data = {
            "model_path": str(self.model_path),
            "feature_dim": len(self.sensor_cols),
            "sequence_window": 30,
            "comparisons": comparisons,
            "detailed_subsets": results
        }

        # Save to results/
        results_dir = self.project_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        out_file = results_dir / "cross_dataset_evaluation.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)

        print(f"Cross-dataset evaluation complete. Saved to {out_file}")
        return output_data


if __name__ == "__main__":
    evaluator = CrossDatasetEvaluator()
    evaluator.run_full_cross_evaluation()
