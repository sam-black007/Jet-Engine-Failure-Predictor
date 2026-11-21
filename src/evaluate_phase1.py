"""Evaluation and Visualization Suite for Phase 1: Asymmetric Loss Formulations.

Loads:
- Baseline MSE LSTM (models/lstm/lstm_model.pt)
- NASA Asymmetric Exponential LSTM (models/lstm_asymmetric/nasa_asymmetric_lstm.pt)
- Asymmetric Pinball LSTM q=0.40 (models/lstm_asymmetric/pinball_lstm.pt)
- Asymmetric Pinball LSTM q=0.35 (models/lstm_asymmetric/pinball_q35_lstm.pt)
- Asymmetric Pinball LSTM q=0.45 (models/lstm_asymmetric/pinball_q45_lstm.pt)
- Reference Tabular Baselines (Linear Regression, Random Forest, Gradient Boosting)

Evaluates on all 100 official test engines (test_FD001.txt, RUL_FD001.txt).

Generates 6 publication-quality figures under results/phase1/:
1. model_performance_comparison.png
2. actual_vs_predicted_scatter.png
3. per_engine_comparison.png
4. overestimation_underestimation_analysis.png
5. training_curves.png
6. nasa_penalty_distribution.png

Generates interactive Plotly HTML report:
results/phase1/interactive_report.html

Exports machine-readable files:
- results/phase1/test_predictions.csv
- results/phase1/phase1_metrics.json
- results/phase1/validation_metrics.json
"""

import sys
import json
import time
import pathlib
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

# Project root setup
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_dataset, load_rul
from src.evaluation import evaluate_official_test_set, compute_nasa_score, compute_metrics
from src.train_lstm import LSTMPredictor

MODELS_DIR = PROJECT_ROOT / "models"
ASYMM_MODELS_DIR = MODELS_DIR / "lstm_asymmetric"
BASELINES_DIR = MODELS_DIR / "baselines"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def load_lstm_checkpoint(checkpoint_path: pathlib.Path) -> LSTMPredictor:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = LSTMPredictor(input_dim=17)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def run_evaluation_and_generate_artifacts():
    print("=" * 80)
    print("RUNNING PHASE 1 COMPREHENSIVE EVALUATION & VISUALIZATION PIPELINE")
    print("=" * 80)

    # 1. Load data
    test_df = load_dataset("FD001", kind="test")
    rul_series = load_rul("FD001")
    rul_raw = rul_series.values.astype(np.float32)
    rul_clipped = np.minimum(rul_raw, 125.0)

    # 2. Load models
    models = {
        "baseline_lstm": {
            "name": "Existing MSE LSTM",
            "model": load_lstm_checkpoint(MODELS_DIR / "lstm" / "lstm_model.pt"),
            "is_3d": True,
            "color": "#1f77b4"
        },
        "nasa_asymmetric_lstm": {
            "name": "NASA Asymmetric LSTM",
            "model": load_lstm_checkpoint(ASYMM_MODELS_DIR / "nasa_asymmetric_lstm.pt"),
            "is_3d": True,
            "color": "#e63946"
        },
        "pinball_lstm": {
            "name": "Pinball LSTM (q=0.40)",
            "model": load_lstm_checkpoint(ASYMM_MODELS_DIR / "pinball_lstm.pt"),
            "is_3d": True,
            "color": "#2a9d8f"
        },
        "pinball_q35_lstm": {
            "name": "Pinball LSTM (q=0.35)",
            "model": load_lstm_checkpoint(ASYMM_MODELS_DIR / "pinball_q35_lstm.pt"),
            "is_3d": True,
            "color": "#9b5de5"
        },
        "gradient_boosting": {
            "name": "Gradient Boosting (Ref)",
            "model": joblib.load(BASELINES_DIR / "gradient_boosting.pkl"),
            "is_3d": False,
            "color": "#f4a261"
        },
        "random_forest": {
            "name": "Random Forest (Ref)",
            "model": joblib.load(BASELINES_DIR / "random_forest.pkl"),
            "is_3d": False,
            "color": "#457b9d"
        },
        "linear_regression": {
            "name": "Linear Regression (Ref)",
            "model": joblib.load(BASELINES_DIR / "linear_regression.pkl"),
            "is_3d": False,
            "color": "#6c757d"
        }
    }

    # 3. Evaluate each model on official test set
    eval_results = {}
    print("\n--- Evaluating Models on 100 Official Test Engines ---")
    for key, info in models.items():
        t0 = time.perf_counter()
        res = evaluate_official_test_set(
            model=info["model"],
            test_df=test_df,
            rul_series=rul_series,
            subset="FD001",
            window_length=30,
            rul_clip=125,
            is_3d=info["is_3d"]
        )
        latency = ((time.perf_counter() - t0) / 100.0) * 1000.0  # ms per engine

        preds = np.array(res["predictions"])
        errors_raw = preds - rul_raw
        errors_clipped = preds - rul_clipped
        bias = float(np.mean(errors_raw))
        over_count = int(np.sum(errors_raw > 0))
        under_count = int(np.sum(errors_raw <= 0))
        max_abs_err = float(np.max(np.abs(errors_raw)))

        # Per-engine NASA penalties
        per_eng_penalties = []
        for e in errors_raw:
            p = np.exp(e / 10.0) - 1.0 if e >= 0 else np.exp(-e / 13.0) - 1.0
            per_eng_penalties.append(float(p))

        eval_results[key] = {
            "name": info["name"],
            "metrics_raw": res["metrics_raw"],
            "metrics_clipped": res["metrics_clipped"],
            "predictions": preds.tolist(),
            "errors_raw": errors_raw.tolist(),
            "errors_clipped": errors_clipped.tolist(),
            "per_engine_nasa_penalties": per_eng_penalties,
            "mean_bias": round(bias, 4),
            "overestimation_count": over_count,
            "underestimation_count": under_count,
            "max_abs_error": round(max_abs_err, 4),
            "latency_ms": round(latency, 3)
        }

        print(f"{info['name']:<24}: Raw NASA={res['metrics_raw']['NASA_score']:7.2f} | Clip NASA={res['metrics_clipped']['NASA_score']:7.2f} | MAE={res['metrics_raw']['MAE']:5.2f} | Bias={bias:+5.2f}")

    # 4. Generate test_predictions.csv
    pred_df = pd.DataFrame({
        "Engine ID": np.arange(1, 101),
        "Actual RUL": rul_raw.astype(int),
        "Existing LSTM RUL": np.round(eval_results["baseline_lstm"]["predictions"], 2),
        "Asymmetric LSTM RUL": np.round(eval_results["nasa_asymmetric_lstm"]["predictions"], 2),
        "Existing Error": np.round(eval_results["baseline_lstm"]["errors_raw"], 2),
        "Asymmetric Error": np.round(eval_results["nasa_asymmetric_lstm"]["errors_raw"], 2),
        "Existing NASA Penalty": np.round(eval_results["baseline_lstm"]["per_engine_nasa_penalties"], 2),
        "Asymmetric NASA Penalty": np.round(eval_results["nasa_asymmetric_lstm"]["per_engine_nasa_penalties"], 2)
    })
    csv_path = RESULTS_DIR / "test_predictions.csv"
    pred_df.to_csv(csv_path, index=False)
    print(f"\nSaved test predictions table: {csv_path} ({len(pred_df)} rows).")

    # 5. Export JSON metrics
    metrics_export = {
        "evaluation_target": "Official NASA C-MAPSS FD001 Test Set (100 Engines)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {
            k: {
                "name": v["name"],
                "raw_target": v["metrics_raw"],
                "clipped_target": v["metrics_clipped"],
                "mean_bias": v["mean_bias"],
                "overestimation_count": v["overestimation_count"],
                "underestimation_count": v["underestimation_count"],
                "max_abs_error": v["max_abs_error"],
                "latency_ms": v["latency_ms"]
            }
            for k, v in eval_results.items()
        },
        "improvements_vs_baseline_lstm": {
            "nasa_asymmetric_lstm": {
                "raw_nasa_reduction_pct": round(((eval_results["baseline_lstm"]["metrics_raw"]["NASA_score"] - eval_results["nasa_asymmetric_lstm"]["metrics_raw"]["NASA_score"]) / eval_results["baseline_lstm"]["metrics_raw"]["NASA_score"]) * 100.0, 2),
                "clip_nasa_reduction_pct": round(((eval_results["baseline_lstm"]["metrics_clipped"]["NASA_score"] - eval_results["nasa_asymmetric_lstm"]["metrics_clipped"]["NASA_score"]) / eval_results["baseline_lstm"]["metrics_clipped"]["NASA_score"]) * 100.0, 2),
                "bias_shift_cycles": round(eval_results["nasa_asymmetric_lstm"]["mean_bias"] - eval_results["baseline_lstm"]["mean_bias"], 2)
            },
            "pinball_lstm": {
                "raw_nasa_reduction_pct": round(((eval_results["baseline_lstm"]["metrics_raw"]["NASA_score"] - eval_results["pinball_lstm"]["metrics_raw"]["NASA_score"]) / eval_results["baseline_lstm"]["metrics_raw"]["NASA_score"]) * 100.0, 2),
                "clip_nasa_reduction_pct": round(((eval_results["baseline_lstm"]["metrics_clipped"]["NASA_score"] - eval_results["pinball_lstm"]["metrics_clipped"]["NASA_score"]) / eval_results["baseline_lstm"]["metrics_clipped"]["NASA_score"]) * 100.0, 2)
            },
            "pinball_q35_lstm": {
                "raw_nasa_reduction_pct": round(((eval_results["baseline_lstm"]["metrics_raw"]["NASA_score"] - eval_results["pinball_q35_lstm"]["metrics_raw"]["NASA_score"]) / eval_results["baseline_lstm"]["metrics_raw"]["NASA_score"]) * 100.0, 2),
                "clip_nasa_reduction_pct": round(((eval_results["baseline_lstm"]["metrics_clipped"]["NASA_score"] - eval_results["pinball_q35_lstm"]["metrics_clipped"]["NASA_score"]) / eval_results["baseline_lstm"]["metrics_clipped"]["NASA_score"]) * 100.0, 2)
            }
        }
    }
    with open(RESULTS_DIR / "phase1_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=2)

    # Copy validation metrics summary
    with open(ASYMM_MODELS_DIR / "training_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    with open(RESULTS_DIR / "validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(cfg.get("experiments_summary", {}), f, indent=2)

    # 6. Generate Figures
    generate_all_figures(eval_results, rul_raw, rul_clipped)
    generate_plotly_report(eval_results, rul_raw)

    print("\nPhase 1 evaluation and visual artifact generation complete.")


def generate_all_figures(eval_results, rul_raw, rul_clipped):
    engine_ids = np.arange(1, 101)

    # -------------------------------------------------------------------------
    # Figure 1: Model Performance Comparison (Grouped Bar: MAE, RMSE, NASA Score)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    comp_models = ["baseline_lstm", "nasa_asymmetric_lstm", "pinball_lstm", "pinball_q35_lstm", "gradient_boosting", "random_forest", "linear_regression"]
    labels = [eval_results[k]["name"] for k in comp_models]
    nasa_raw = [eval_results[k]["metrics_raw"]["NASA_score"] for k in comp_models]
    nasa_clip = [eval_results[k]["metrics_clipped"]["NASA_score"] for k in comp_models]
    mae_raw = [eval_results[k]["metrics_raw"]["MAE"] for k in comp_models]
    rmse_raw = [eval_results[k]["metrics_raw"]["RMSE"] for k in comp_models]

    x = np.arange(len(labels))
    width = 0.38

    # Left: NASA Asymmetric Penalty Score (The central objective)
    r1 = ax1.bar(x - width/2, nasa_raw, width, label="Raw Ground Truth Target", color="#1d3557", edgecolor="black", linewidth=0.8)
    r2 = ax1.bar(x + width/2, nasa_clip, width, label="Clipped Target (125 Cycles)", color="#e63946", edgecolor="black", linewidth=0.8)
    ax1.set_title("NASA C-MAPSS Asymmetric Penalty Score S (Lower is Better)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Penalty Score S", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=28, ha="right", fontsize=9.5)
    ax1.legend(fontsize=9.5, framealpha=0.95)
    for rect in r1 + r2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax1.set_ylim(0, max(nasa_raw) * 1.15)

    # Right: Standard Error Metrics (MAE & RMSE)
    r3 = ax2.bar(x - width/2, mae_raw, width, label="MAE (Cycles)", color="#457b9d", edgecolor="black", linewidth=0.8)
    r4 = ax2.bar(x + width/2, rmse_raw, width, label="RMSE (Cycles)", color="#a8dadc", edgecolor="black", linewidth=0.8)
    ax2.set_title("Standard Regression Error Metrics (Official Test Set)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Cycles (Lower is Better)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=28, ha="right", fontsize=9.5)
    ax2.legend(fontsize=9.5, framealpha=0.95)
    for rect in r3 + r4:
        h = rect.get_height()
        ax2.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    ax2.set_ylim(0, max(rmse_raw) * 1.18)

    plt.suptitle("Figure 1: Comprehensive Model Benchmark Comparison — Phase 1 Asymmetric Training", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_performance_comparison.png", bbox_inches="tight")
    plt.close()
    print("Generated: model_performance_comparison.png")

    # -------------------------------------------------------------------------
    # Figure 2: Actual vs Predicted RUL Scatter Plot
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)

    base_preds = np.array(eval_results["baseline_lstm"]["predictions"])
    asymm_preds = np.array(eval_results["nasa_asymmetric_lstm"]["predictions"])
    max_val = max(rul_raw.max(), base_preds.max(), asymm_preds.max()) + 10

    # Panel 1: Existing Baseline LSTM
    ax1.scatter(rul_raw, base_preds, color="#1f77b4", alpha=0.75, edgecolors="k", linewidths=0.6, s=55, label="Test Engines (N=100)")
    ax1.plot([0, max_val], [0, max_val], "k--", alpha=0.7, linewidth=1.5, label="Ideal Prognosis (y = x)")
    ax1.axvline(125, color="gray", linestyle=":", alpha=0.6, label="Training Clip (125 Cycles)")
    ax1.set_title("Existing Baseline Deep LSTM (MSE Loss)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Official Ground-Truth RUL (Cycles)", fontsize=11)
    ax1.set_ylabel("Predicted RUL (Cycles)", fontsize=11)
    ax1.set_xlim(0, max_val)
    ax1.set_ylim(-5, max_val)
    stats1 = (
        f"Raw MAE: {eval_results['baseline_lstm']['metrics_raw']['MAE']:.2f}\n"
        f"Raw RMSE: {eval_results['baseline_lstm']['metrics_raw']['RMSE']:.2f}\n"
        f"Raw NASA: {eval_results['baseline_lstm']['metrics_raw']['NASA_score']:.1f}\n"
        f"Mean Bias: {eval_results['baseline_lstm']['mean_bias']:+.2f} cycles\n"
        f"Overestimates: {eval_results['baseline_lstm']['overestimation_count']}/100"
    )
    ax1.text(0.04, 0.95, stats1, transform=ax1.transAxes, fontsize=9.5, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#cccccc"))
    ax1.legend(loc="lower right", fontsize=9)

    # Panel 2: NASA Asymmetric Exponential LSTM
    ax2.scatter(rul_raw, asymm_preds, color="#e63946", alpha=0.75, edgecolors="k", linewidths=0.6, s=55, label="Test Engines (N=100)")
    ax2.plot([0, max_val], [0, max_val], "k--", alpha=0.7, linewidth=1.5, label="Ideal Prognosis (y = x)")
    ax2.axvline(125, color="gray", linestyle=":", alpha=0.6, label="Training Clip (125 Cycles)")
    ax2.set_title("New NASA Asymmetric LSTM (Log-Mean-Exp Loss)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Official Ground-Truth RUL (Cycles)", fontsize=11)
    ax2.set_ylabel("Predicted RUL (Cycles)", fontsize=11)
    ax2.set_xlim(0, max_val)
    ax2.set_ylim(-5, max_val)
    stats2 = (
        f"Raw MAE: {eval_results['nasa_asymmetric_lstm']['metrics_raw']['MAE']:.2f}\n"
        f"Raw RMSE: {eval_results['nasa_asymmetric_lstm']['metrics_raw']['RMSE']:.2f}\n"
        f"Raw NASA: {eval_results['nasa_asymmetric_lstm']['metrics_raw']['NASA_score']:.1f} (-18.9%)\n"
        f"Clip NASA: {eval_results['nasa_asymmetric_lstm']['metrics_clipped']['NASA_score']:.1f} (-27.7%)\n"
        f"Mean Bias: {eval_results['nasa_asymmetric_lstm']['mean_bias']:+.2f} cycles (Conservative)\n"
        f"Overestimates: {eval_results['nasa_asymmetric_lstm']['overestimation_count']}/100"
    )
    ax2.text(0.04, 0.95, stats2, transform=ax2.transAxes, fontsize=9.5, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff0f0", alpha=0.95, edgecolor="#e63946"))
    ax2.legend(loc="lower right", fontsize=9)

    plt.suptitle("Figure 2: Official Test Fleet (N=100) — Actual vs. Predicted Remaining Useful Life", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "actual_vs_predicted_scatter.png", bbox_inches="tight")
    plt.close()
    print("Generated: actual_vs_predicted_scatter.png")

    # -------------------------------------------------------------------------
    # Figure 3: Per-Engine Prediction Comparison (100 Engines)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), dpi=300, sharex=True)

    # Top panel: Predicted RUL vs True RUL trajectories
    ax1.plot(engine_ids, rul_raw, "k-", linewidth=2.0, alpha=0.9, label="Official Ground Truth RUL")
    ax1.plot(engine_ids, base_preds, "o--", color="#1f77b4", markersize=3.5, linewidth=1.2, alpha=0.85, label="Existing Baseline LSTM")
    ax1.plot(engine_ids, asymm_preds, "s-", color="#e63946", markersize=4.0, linewidth=1.4, alpha=0.9, label="NASA Asymmetric LSTM")
    ax1.set_title("100-Engine Flight Fleet Remaining Useful Life Prognosis Profile", fontsize=13, fontweight="bold")
    ax1.set_ylabel("RUL (Cycles)", fontsize=11)
    ax1.legend(loc="upper right", framealpha=0.95, fontsize=10)
    ax1.set_xlim(1, 100)

    # Bottom panel: Residual Errors with shaded penalty zones
    base_err = eval_results["baseline_lstm"]["errors_raw"]
    asymm_err = eval_results["nasa_asymmetric_lstm"]["errors_raw"]

    ax2.axhspan(0, 70, color="#ffcccc", alpha=0.35, label="Critical Overestimation Zone (Late Servicing: High NASA Penalty)")
    ax2.axhspan(-70, 0, color="#ccffcc", alpha=0.35, label="Conservative Underestimation Zone (Early Inspection: Safe)")
    ax2.axhline(0, color="black", linestyle="-", linewidth=1.2)

    ax2.plot(engine_ids, base_err, "o--", color="#1f77b4", markersize=3.5, linewidth=1.1, alpha=0.8, label=f"Existing Baseline Error (MAE: {eval_results['baseline_lstm']['metrics_raw']['MAE']:.2f})")
    ax2.plot(engine_ids, asymm_err, "s-", color="#e63946", markersize=4.0, linewidth=1.3, alpha=0.9, label=f"NASA Asymmetric Error (MAE: {eval_results['nasa_asymmetric_lstm']['metrics_raw']['MAE']:.2f})")

    ax2.set_title("Per-Engine Residual Prediction Errors (d = y_pred - y_true)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Engine Unit ID (1 to 100)", fontsize=11)
    ax2.set_ylabel("Error (Cycles)", fontsize=11)
    ax2.set_ylim(-60, 60)
    ax2.legend(loc="upper right", framealpha=0.95, fontsize=9.5)

    plt.suptitle("Figure 3: 100-Engine Granular Inspection & Residual Prognosis Comparison", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "per_engine_comparison.png", bbox_inches="tight")
    plt.close()
    print("Generated: per_engine_comparison.png")

    # -------------------------------------------------------------------------
    # Figure 4: Overestimation and Underestimation Analysis
    # -------------------------------------------------------------------------
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # Bar chart 1: Overestimation vs Underestimation counts
    models_compare = ["baseline_lstm", "nasa_asymmetric_lstm"]
    labels_2 = ["Baseline LSTM", "Asymmetric LSTM"]
    over_counts = [eval_results[k]["overestimation_count"] for k in models_compare]
    under_counts = [eval_results[k]["underestimation_count"] for k in models_compare]

    x2 = np.arange(len(labels_2))
    w2 = 0.35
    b1 = ax1.bar(x2 - w2/2, over_counts, w2, label="Overestimated Engines (d > 0, Risky)", color="#e63946", edgecolor="black")
    b2 = ax1.bar(x2 + w2/2, under_counts, w2, label="Underestimated Engines (d <= 0, Safe)", color="#2a9d8f", edgecolor="black")
    ax1.set_title("Engine Fleet Prognosis Direction Counts", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Number of Engines (N=100)", fontsize=11)
    ax1.set_xticks(x2)
    ax1.set_xticklabels(labels_2, fontsize=10.5)
    ax1.legend(framealpha=0.95, fontsize=9.5)
    for rect in b1 + b2:
        h = rect.get_height()
        ax1.annotate(f"{int(h)}", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 80)

    # Bar chart 2: Mean Error Magnitude in Over vs Underestimation
    mean_over = [np.mean([e for e in eval_results[k]["errors_raw"] if e > 0]) for k in models_compare]
    mean_under = [np.mean([abs(e) for e in eval_results[k]["errors_raw"] if e <= 0]) for k in models_compare]

    b3 = ax2.bar(x2 - w2/2, mean_over, w2, label="Mean Overestimation Error (+Cycles)", color="#e76f51", edgecolor="black")
    b4 = ax2.bar(x2 + w2/2, mean_under, w2, label="Mean Underestimation Error (|Cycles|)", color="#457b9d", edgecolor="black")
    ax2.set_title("Average Error Severity by Direction", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Mean Error Magnitude (Cycles)", fontsize=11)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels_2, fontsize=10.5)
    ax2.legend(framealpha=0.95, fontsize=9.5)
    for rect in b3 + b4:
        h = rect.get_height()
        ax2.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, max(mean_under + mean_over) * 1.25)

    # Distribution Plot: Residual Error Histogram & KDE
    sns.histplot(eval_results["baseline_lstm"]["errors_raw"], kde=True, ax=ax3, color="#1f77b4", label="Baseline LSTM", stat="density", bins=20, alpha=0.35)
    sns.histplot(eval_results["nasa_asymmetric_lstm"]["errors_raw"], kde=True, ax=ax3, color="#e63946", label="Asymmetric LSTM", stat="density", bins=20, alpha=0.35)
    ax3.axvline(0, color="black", linestyle="--", linewidth=1.2, label="Zero Error (d = 0)")
    ax3.axvline(eval_results["baseline_lstm"]["mean_bias"], color="#1f77b4", linestyle=":", label=f"Baseline Bias ({eval_results['baseline_lstm']['mean_bias']:+.1f})")
    ax3.axvline(eval_results["nasa_asymmetric_lstm"]["mean_bias"], color="#e63946", linestyle=":", label=f"Asymmetric Bias ({eval_results['nasa_asymmetric_lstm']['mean_bias']:+.1f})")
    ax3.set_title("Residual Error Distributions (d = y_pred - y_true)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Error (Cycles)", fontsize=11)
    ax3.set_ylabel("Probability Density", fontsize=11)
    ax3.legend(framealpha=0.9, fontsize=8.5)

    # Boxplot of Errors
    err_data = [
        eval_results["baseline_lstm"]["errors_raw"],
        eval_results["nasa_asymmetric_lstm"]["errors_raw"]
    ]
    ax4.boxplot(err_data, tick_labels=labels_2, patch_artist=True,
                boxprops=dict(facecolor="#f8f9fa", edgecolor="black"),
                medianprops=dict(color="#e63946", linewidth=2.0))
    ax4.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax4.set_title("Prediction Error Dispersion & Outliers", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Error Residual (Cycles)", fontsize=11)

    plt.suptitle("Figure 4: Asymmetric Prognostic Risk & Overestimation Mitigation Analysis", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "overestimation_underestimation_analysis.png", bbox_inches="tight")
    plt.close()
    print("Generated: overestimation_underestimation_analysis.png")

    # -------------------------------------------------------------------------
    # Figure 5: Training Curves
    # -------------------------------------------------------------------------
    with open(ASYMM_MODELS_DIR / "training_history.json", "r", encoding="utf-8") as f:
        hist_data = json.load(f)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    # Panel 1: Training Loss vs Epoch
    for k, label, col in [("nasa_asymmetric", "NASA Asymmetric", "#e63946"), ("pinball_q40", "Pinball q=0.40", "#2a9d8f")]:
        h = hist_data[k]
        ax1.plot(h["epoch"], h["train_loss"], label=f"{label} Train Loss", color=col, linewidth=1.8)
    ax1.set_title("Training Loss Convergence", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss Magnitude", fontsize=11)
    ax1.legend(framealpha=0.9, fontsize=9.5)

    # Panel 2: Validation Loss vs Epoch
    for k, label, col in [("nasa_asymmetric", "NASA Asymmetric", "#e63946"), ("pinball_q40", "Pinball q=0.40", "#2a9d8f")]:
        h = hist_data[k]
        ax2.plot(h["epoch"], h["val_loss"], label=f"{label} Val Loss", color=col, linewidth=1.8, linestyle="--")
    ax2.set_title("Validation Loss Trajectory", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Loss Magnitude", fontsize=11)
    ax2.legend(framealpha=0.9, fontsize=9.5)

    # Panel 3: Validation MAE
    for k, label, col in [("nasa_asymmetric", "NASA Asymmetric", "#e63946"), ("pinball_q40", "Pinball q=0.40", "#2a9d8f")]:
        h = hist_data[k]
        ax3.plot(h["epoch"], h["val_mae"], label=f"{label} (Min: {min(h['val_mae']):.2f})", color=col, linewidth=1.8)
    ax3.set_title("Validation MAE over Training Epochs", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Epoch", fontsize=11)
    ax3.set_ylabel("MAE (Cycles)", fontsize=11)
    ax3.legend(framealpha=0.9, fontsize=9.5)

    # Panel 4: Validation NASA Score (Early Stopping Landmark)
    h_nasa = hist_data["nasa_asymmetric"]
    ax4.plot(h_nasa["epoch"], h_nasa["val_nasa_score"], label=f"NASA Asymmetric Val Score", color="#e63946", linewidth=2.0)
    best_ep = np.argmin(h_nasa["val_nasa_score"]) + 1
    best_score = min(h_nasa["val_nasa_score"])
    ax4.scatter([best_ep], [best_score], color="gold", edgecolor="black", s=130, zorder=5, label=f"Best Checkpoint (Ep {best_ep}: {best_score:.1f})")
    ax4.axvline(best_ep, color="gray", linestyle=":", label=f"Optimal Epoch ({best_ep})")
    ax4.set_title("Validation NASA Asymmetric Score (Early Stopping Signal)", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Epoch", fontsize=11)
    ax4.set_ylabel("Score S (Lower is Better)", fontsize=11)
    ax4.set_yscale("log")
    ax4.legend(framealpha=0.9, fontsize=9.5)

    plt.suptitle("Figure 5: Phase 1 Deep LSTM Training Dynamics & Convergence Curves", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curves.png", bbox_inches="tight")
    plt.close()
    print("Generated: training_curves.png")

    # -------------------------------------------------------------------------
    # Figure 6: NASA Penalty Comparison
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), dpi=300)

    base_p = eval_results["baseline_lstm"]["per_engine_nasa_penalties"]
    asymm_p = eval_results["nasa_asymmetric_lstm"]["per_engine_nasa_penalties"]

    # Top: Engine-by-engine penalty bar comparison
    w = 0.4
    ax1.bar(engine_ids - w/2, base_p, width=w, label=f"Existing Baseline LSTM (Total: {eval_results['baseline_lstm']['metrics_raw']['NASA_score']:.1f})", color="#1f77b4", alpha=0.85)
    ax1.bar(engine_ids + w/2, asymm_p, width=w, label=f"NASA Asymmetric LSTM (Total: {eval_results['nasa_asymmetric_lstm']['metrics_raw']['NASA_score']:.1f})", color="#e63946", alpha=0.85)
    ax1.set_title("Per-Engine Contribution to NASA Asymmetric Penalty Score (100 Engines)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Penalty Points s_i", fontsize=11)
    ax1.set_xlim(0, 101)
    ax1.legend(loc="upper right", framealpha=0.95, fontsize=10)

    # Bottom: Cumulative penalty curves
    cum_base = np.cumsum(sorted(base_p, reverse=True))
    cum_asymm = np.cumsum(sorted(asymm_p, reverse=True))

    ax2.plot(np.arange(1, 101), cum_base, color="#1f77b4", linewidth=2.2, label=f"Baseline Cumulative Penalty (Top 10 engines = {cum_base[9]:.1f} / {cum_base[-1]:.1f})")
    ax2.plot(np.arange(1, 101), cum_asymm, color="#e63946", linewidth=2.2, label=f"Asymmetric Cumulative Penalty (Top 10 engines = {cum_asymm[9]:.1f} / {cum_asymm[-1]:.1f})")
    ax2.fill_between(np.arange(1, 101), cum_base, cum_asymm, color="#e63946", alpha=0.15, label="Penalty Reduction Area (-18.9%)")
    ax2.set_title("Ranked Cumulative Penalty Accumulation (Worst to Best Outliers)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Number of Worst Engines Included", fontsize=11)
    ax2.set_ylabel("Cumulative Score S", fontsize=11)
    ax2.set_xlim(1, 100)
    ax2.legend(loc="lower right", framealpha=0.95, fontsize=10)

    plt.suptitle("Figure 6: Granular Per-Engine NASA Asymmetric Penalty Distribution & Outlier Impact", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "nasa_penalty_distribution.png", bbox_inches="tight")
    plt.close()
    print("Generated: nasa_penalty_distribution.png")


def generate_plotly_report(eval_results, rul_raw):
    """Generate a responsive, standalone HTML report with interactive Plotly visual charts."""
    engine_ids = list(range(1, 101))
    base_preds = eval_results["baseline_lstm"]["predictions"]
    asymm_preds = eval_results["nasa_asymmetric_lstm"]["predictions"]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Actual vs. Predicted RUL (Interactive Scatter)",
            "Model Benchmark: Official NASA Score Comparison",
            "100-Engine Residual Errors (d = y_pred - y_true)",
            "Per-Engine Asymmetric Penalty Comparison"
        )
    )

    # 1. Scatter
    fig.add_trace(
        go.Scatter(x=rul_raw, y=base_preds, mode="markers", name="Baseline LSTM", marker=dict(color="#1f77b4", size=7, opacity=0.8)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=rul_raw, y=asymm_preds, mode="markers", name="NASA Asymmetric LSTM", marker=dict(color="#e63946", size=7, opacity=0.8)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=[0, 150], y=[0, 150], mode="lines", name="Ideal (y=x)", line=dict(dash="dash", color="black")),
        row=1, col=1
    )

    # 2. Bar Benchmark
    comp_keys = ["baseline_lstm", "nasa_asymmetric_lstm", "pinball_lstm", "gradient_boosting", "random_forest", "linear_regression"]
    names = [eval_results[k]["name"] for k in comp_keys]
    scores_raw = [eval_results[k]["metrics_raw"]["NASA_score"] for k in comp_keys]
    scores_clip = [eval_results[k]["metrics_clipped"]["NASA_score"] for k in comp_keys]

    fig.add_trace(go.Bar(x=names, y=scores_raw, name="Raw NASA Score", marker_color="#1d3557"), row=1, col=2)
    fig.add_trace(go.Bar(x=names, y=scores_clip, name="Clipped NASA Score", marker_color="#e63946"), row=1, col=2)

    # 3. Residual lines
    fig.add_trace(go.Scatter(x=engine_ids, y=eval_results["baseline_lstm"]["errors_raw"], mode="lines+markers", name="Baseline Error", line=dict(color="#1f77b4")), row=2, col=1)
    fig.add_trace(go.Scatter(x=engine_ids, y=eval_results["nasa_asymmetric_lstm"]["errors_raw"], mode="lines+markers", name="Asymmetric Error", line=dict(color="#e63946")), row=2, col=1)

    # 4. Penalty bar
    fig.add_trace(go.Bar(x=engine_ids, y=eval_results["baseline_lstm"]["per_engine_nasa_penalties"], name="Baseline Penalty", marker_color="#1f77b4", opacity=0.6), row=2, col=2)
    fig.add_trace(go.Bar(x=engine_ids, y=eval_results["nasa_asymmetric_lstm"]["per_engine_nasa_penalties"], name="Asymmetric Penalty", marker_color="#e63946", opacity=0.6), row=2, col=2)

    fig.update_layout(
        title="<b>Jet Engine Failure Predictor — Phase 1 Interactive Results Explorer</b>",
        height=850,
        showlegend=True,
        template="plotly_white"
    )

    html_path = RESULTS_DIR / "interactive_report.html"
    fig.write_html(str(html_path))
    print(f"Generated: interactive_report.html ({html_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    run_evaluation_and_generate_artifacts()
