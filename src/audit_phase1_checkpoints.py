"""Audit script to resolve Pinball LSTM score discrepancy and inspect all checkpoints."""

import json
import pathlib
import torch
import numpy as np
import pandas as pd

from src.train_lstm import LSTMPredictor
from src.evaluation import evaluate_official_test_set, compute_nasa_score, compute_metrics

project_root = pathlib.Path(__file__).resolve().parent.parent

checkpoints = {
    "baseline_lstm": project_root / "models" / "lstm" / "lstm_model.pt",
    "nasa_asymmetric": project_root / "models" / "lstm_asymmetric" / "nasa_asymmetric_lstm.pt",
    "pinball_lstm": project_root / "models" / "lstm_asymmetric" / "pinball_lstm.pt",
    "pinball_q35_lstm": project_root / "models" / "lstm_asymmetric" / "pinball_q35_lstm.pt",
    "pinball_q45_lstm": project_root / "models" / "lstm_asymmetric" / "pinball_q45_lstm.pt",
}

print("=" * 80)
print("AUDITING PHASE 1 CHECKPOINTS ON OFFICIAL FD001 TEST FLEET (100 ENGINES)")
print("=" * 80)

audit_records = {}

for name, path in checkpoints.items():
    if not path.exists():
        print(f"[-] {name}: Checkpoint NOT found at {path}")
        continue

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    input_dim = ckpt.get("input_dim", 17) if isinstance(ckpt, dict) else 17
    hidden_dim1 = ckpt.get("hidden_dim1", 64) if isinstance(ckpt, dict) else 64
    hidden_dim2 = ckpt.get("hidden_dim2", 32) if isinstance(ckpt, dict) else 32
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    model = LSTMPredictor(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2)
    model.load_state_dict(state_dict)
    model.eval()

    eval_result = evaluate_official_test_set(
        model=model,
        subset="FD001",
        window_length=30,
        rul_clip=125,
        is_3d=True
    )

    m_raw = eval_result["metrics_raw"]
    m_clip = eval_result["metrics_clipped"]

    # Also compute on predictions if predictions themselves are clipped to 125
    preds = np.array(eval_result["predictions"])
    gt_raw = np.array(eval_result["ground_truth_raw"])
    gt_clip = np.array(eval_result["ground_truth_clipped"])

    preds_clipped_125 = np.clip(preds, 0, 125)
    nasa_preds_clipped_gt_raw = compute_nasa_score(gt_raw, preds_clipped_125)
    nasa_preds_clipped_gt_clip = compute_nasa_score(gt_clip, preds_clipped_125)
    mae_preds_clipped_gt_clip = float(np.mean(np.abs(preds_clipped_125 - gt_clip)))

    audit_records[name] = {
        "checkpoint": str(path.relative_to(project_root)),
        "input_dim": input_dim,
        "eval_raw_target": {
            "MAE": round(m_raw["MAE"], 4),
            "RMSE": round(m_raw["RMSE"], 4),
            "NASA_Score": round(m_raw["NASA_score"], 4)
        },
        "eval_clipped_target": {
            "MAE": round(m_clip["MAE"], 4),
            "RMSE": round(m_clip["RMSE"], 4),
            "NASA_Score": round(m_clip["NASA_score"], 4)
        },
        "eval_both_clipped": {
            "MAE": round(mae_preds_clipped_gt_clip, 4),
            "NASA_Score": round(nasa_preds_clipped_gt_clip, 4)
        }
    }

    print(f"\n[+] Model: {name} (File: {path.name})")
    print(f"    Raw Ground-Truth Target:     MAE = {m_raw['MAE']:.2f}, RMSE = {m_raw['RMSE']:.2f}, NASA Score = {m_raw['NASA_score']:.2f}")
    print(f"    Clipped (125) GT Target:     MAE = {m_clip['MAE']:.2f}, RMSE = {m_clip['RMSE']:.2f}, NASA Score = {m_clip['NASA_score']:.2f}")
    print(f"    Both Pred & GT Clipped(125): MAE = {mae_preds_clipped_gt_clip:.2f}, NASA Score = {nasa_preds_clipped_gt_clip:.2f}")

# Also inspect training history if available
hist_path = project_root / "models" / "lstm_asymmetric" / "training_history.json"
if hist_path.exists():
    with open(hist_path, "r", encoding="utf-8") as f:
        hist = json.load(f)
    print("\n" + "=" * 80)
    print("INSPECTING training_history.json:")
    print("=" * 80)
    for model_k, data in hist.items():
        val_m = data.get("val_metrics", {})
        test_m = data.get("official_test_metrics", {})
        print(f"  Model: {model_k}")
        print(f"    Val Metrics:  {val_m}")
        print(f"    Test Metrics: {test_m}")

# Save audit to results/
out_audit = project_root / "results" / "phase1_pinball_audit.json"
with open(out_audit, "w", encoding="utf-8") as f:
    json.dump(audit_records, f, indent=2)
print(f"\n[x] Audit complete. Saved to {out_audit}")
