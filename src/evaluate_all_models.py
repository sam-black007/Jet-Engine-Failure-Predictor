"""Official NASA C-MAPSS FD001 Test Set Evaluation & Visualization Suite.

Evaluates all four trained models:
1. Linear Regression (Tabular OLS baseline)
2. Random Forest Regressor (Ensemble baseline)
3. Gradient Boosting Regressor (Boosting baseline)
4. Deep LSTM (PyTorch 2-layer Recurrent Network)

Calculates:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- NASA C-MAPSS Asymmetric Scoring Function (S)
Both on raw ground-truth RUL (RUL_FD001.txt) and clipped ground-truth RUL (capped at 125 cycles).

Generates and saves the 5 required technical figures under results/evaluation/:
- actual_vs_predicted_scatter.png
- per_engine_error_chart.png
- rul_prediction_distribution.png
- nasa_score_comparison.png
- model_performance_comparison.png

Updates models/performance_metrics.json while preserving historical validation metrics.
"""

import os
import json
import pathlib
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Project paths
import sys
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
MODELS_DIR = PROJECT_ROOT / "models"
BASELINES_DIR = MODELS_DIR / "baselines"
LSTM_DIR = MODELS_DIR / "lstm"
RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from src.data_loader import load_dataset, load_rul
from src.evaluation import compute_metrics, evaluate_official_test_set
from src.train_lstm import LSTMPredictor

# Professional plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def run_full_evaluation():
    print("=" * 80)
    print("RUNNING OFFICIAL NASA C-MAPSS FD001 TEST SET EVALUATION")
    print("=" * 80)

    # 1. Load data
    test_df = load_dataset("FD001", kind="test")
    rul_series = load_rul("FD001")
    rul_raw = rul_series.values.astype(np.float32)
    rul_clipped = np.minimum(rul_raw, 125.0)

    print(f"Loaded test dataset: {len(test_df)} observations across {test_df['engine_id'].nunique()} test engines.")
    print(f"Loaded ground-truth RUL: {len(rul_raw)} engines (Min: {rul_raw.min()}, Max: {rul_raw.max()}, Mean: {rul_raw.mean():.2f}).")

    # 2. Load models
    models = {
        "linear_regression": {
            "name": "Linear Regression",
            "model": joblib.load(BASELINES_DIR / "linear_regression.pkl"),
            "is_3d": False,
            "color": "#1f77b4"
        },
        "random_forest": {
            "name": "Random Forest",
            "model": joblib.load(BASELINES_DIR / "random_forest.pkl"),
            "is_3d": False,
            "color": "#2ca02c"
        },
        "gradient_boosting": {
            "name": "Gradient Boosting",
            "model": joblib.load(BASELINES_DIR / "gradient_boosting.pkl"),
            "is_3d": False,
            "color": "#ff7f0e"
        }
    }

    # Load PyTorch LSTM
    lstm_ckpt = torch.load(LSTM_DIR / "lstm_model.pt", weights_only=False, map_location="cpu")
    lstm_model = LSTMPredictor(input_dim=17)
    lstm_model.load_state_dict(lstm_ckpt["model_state_dict"])
    lstm_model.eval()

    models["lstm"] = {
        "name": "Deep LSTM (PyTorch)",
        "model": lstm_model,
        "is_3d": True,
        "color": "#d62728"
    }

    # 3. Evaluate each model
    eval_results = {}
    for key, info in models.items():
        print(f"\nEvaluating {info['name']}...")
        res = evaluate_official_test_set(
            model=info["model"],
            test_df=test_df,
            rul_series=rul_series,
            subset="FD001",
            window_length=30,
            rul_clip=125,
            is_3d=info["is_3d"]
        )
        eval_results[key] = res
        print(f"  Raw Target:     MAE = {res['metrics_raw']['MAE']:6.2f} | RMSE = {res['metrics_raw']['RMSE']:6.2f} | NASA = {res['metrics_raw']['NASA_score']:9.2f}")
        print(f"  Clipped Target: MAE = {res['metrics_clipped']['MAE']:6.2f} | RMSE = {res['metrics_clipped']['RMSE']:6.2f} | NASA = {res['metrics_clipped']['NASA_score']:9.2f}")

    # 4. Update performance_metrics.json
    metrics_path = MODELS_DIR / "performance_metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        stored_metrics = json.load(f)

    for key in models.keys():
        if key not in stored_metrics:
            stored_metrics[key] = {}
        stored_metrics[key]["official_test_set"] = {
            "raw_target": eval_results[key]["metrics_raw"],
            "clipped_target": eval_results[key]["metrics_clipped"],
            "num_engines": eval_results[key]["num_engines_evaluated"]
        }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(stored_metrics, f, indent=2)
    print(f"\nSuccessfully updated {metrics_path} with official test metrics.")

    # Save detailed evaluation json
    details_path = RESULTS_DIR / "official_test_metrics.json"
    clean_details = {}
    for k, v in eval_results.items():
        clean_details[k] = {
            "metrics_raw": v["metrics_raw"],
            "metrics_clipped": v["metrics_clipped"],
            "predictions": v["predictions"],
            "ground_truth_raw": v["ground_truth_raw"],
            "ground_truth_clipped": v["ground_truth_clipped"]
        }
    with open(details_path, "w", encoding="utf-8") as f:
        json.dump(clean_details, f, indent=2)
    print(f"Saved detailed predictions to {details_path}.")

    # 5. Generate Visualizations
    generate_visualizations(models, eval_results, rul_raw, rul_clipped)
    print("\nAll visualizations successfully generated and saved to results/evaluation/.")

    # 6. Print summary table
    print_summary_table(models, stored_metrics)


def generate_visualizations(models, eval_results, rul_raw, rul_clipped):
    engine_ids = np.arange(1, 101)

    # -------------------------------------------------------------------------
    # Figure 1: Actual vs Predicted RUL Scatter Plot (2x2 Grid)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    axes = axes.flatten()

    for idx, (key, info) in enumerate(models.items()):
        ax = axes[idx]
        preds = np.array(eval_results[key]["predictions"])
        m_raw = eval_results[key]["metrics_raw"]
        m_clip = eval_results[key]["metrics_clipped"]

        # Scatter points
        ax.scatter(rul_raw, preds, color=info["color"], alpha=0.7, edgecolors="k", linewidths=0.5, s=45, label="Test Engines (N=100)")
        
        # Identity line
        max_val = max(rul_raw.max(), preds.max()) + 10
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.7, linewidth=1.5, label="Ideal (y = x)")
        ax.axvline(125, color="gray", linestyle=":", alpha=0.6, label="Training RUL Clip (125)")

        ax.set_title(f"{info['name']}", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Official Ground-Truth RUL (Cycles)", fontsize=11)
        ax.set_ylabel("Predicted RUL (Cycles)", fontsize=11)
        ax.set_xlim(0, max_val)
        ax.set_ylim(min(-5, preds.min() - 5), max_val)

        stats_box = (
            f"Raw MAE: {m_raw['MAE']:.2f}\n"
            f"Raw RMSE: {m_raw['RMSE']:.2f}\n"
            f"Raw NASA: {m_raw['NASA_score']:.1f}\n"
            f"Clip125 MAE: {m_clip['MAE']:.2f}\n"
            f"Clip125 NASA: {m_clip['NASA_score']:.1f}"
        )
        ax.text(0.04, 0.95, stats_box, transform=ax.transAxes, fontsize=9.5,
                verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="#dddddd"))
        ax.legend(loc="lower right", fontsize=8.5)

    plt.suptitle("NASA C-MAPSS FD001 Official Test Set: Actual vs. Predicted RUL", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig_path = RESULTS_DIR / "actual_vs_predicted_scatter.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Generated: {fig_path.name}")

    # -------------------------------------------------------------------------
    # Figure 2: Per-Engine Prediction Error Chart
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(15, 7), dpi=300)

    # Shaded penalty zones
    ax.axhspan(0, 80, color="#ffdddd", alpha=0.3, label="Late Prediction Zone (Overestimation: Severe Penalty x1.72)")
    ax.axhspan(-80, 0, color="#ddffdd", alpha=0.3, label="Early Prediction Zone (Underestimation: Conservative Penalty x1.16)")
    ax.axhline(0, color="black", linestyle="-", linewidth=1.2)

    for key, info in models.items():
        preds = np.array(eval_results[key]["predictions"])
        errors = preds - rul_raw
        ax.plot(engine_ids, errors, marker="o" if key == "lstm" else "s", markersize=3.5 if key != "lstm" else 4.5,
                linewidth=1.2, alpha=0.85, label=f"{info['name']} (MAE: {eval_results[key]['metrics_raw']['MAE']:.2f})", color=info["color"])

    ax.set_title("Per-Engine Prediction Residuals on Official Test Set (100 Engines)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Engine Unit ID (1 to 100)", fontsize=12)
    ax.set_ylabel("Prediction Error: d = y_pred - y_true (Cycles)", fontsize=12)
    ax.set_xlim(1, 100)
    ax.set_ylim(-60, 60)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=10)
    plt.tight_layout()
    fig_path = RESULTS_DIR / "per_engine_error_chart.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Generated: {fig_path.name}")

    # -------------------------------------------------------------------------
    # Figure 3: RUL Prediction Distribution
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    bins = np.linspace(0, 160, 33)

    ax.hist(rul_raw, bins=bins, color="black", alpha=0.25, density=True, label="Ground Truth RUL", edgecolor="black", linewidth=1.2)
    for key, info in models.items():
        preds = np.array(eval_results[key]["predictions"])
        sns.kdeplot(preds, ax=ax, label=f"{info['name']}", color=info["color"], linewidth=2.0)

    ax.axvline(125, color="gray", linestyle="--", linewidth=1.2, label="RUL Upper Clipping Bound (125)")
    ax.set_title("RUL Prediction Probability Density vs. Ground Truth Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Remaining Useful Life (Cycles)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_xlim(0, 160)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=10)
    plt.tight_layout()
    fig_path = RESULTS_DIR / "rul_prediction_distribution.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Generated: {fig_path.name}")

    # -------------------------------------------------------------------------
    # Figure 4: NASA Asymmetric Score Comparison
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    model_labels = [models[k]["name"] for k in models.keys()]
    raw_scores = [eval_results[k]["metrics_raw"]["NASA_score"] for k in models.keys()]
    clipped_scores = [eval_results[k]["metrics_clipped"]["NASA_score"] for k in models.keys()]

    x = np.arange(len(model_labels))
    width = 0.35

    rects1 = ax.bar(x - width/2, raw_scores, width, label="Official Raw Ground Truth", color="#2b5c8f", edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x + width/2, clipped_scores, width, label="Clipped Target (125 Cycles)", color="#e07a5f", edgecolor="black", linewidth=0.8)

    ax.set_ylabel("NASA Asymmetric Penalty Score S (Lower is Better)", fontsize=12)
    ax.set_title("NASA C-MAPSS Asymmetric Penalty Score Benchmark Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=11)
    ax.legend(framealpha=0.95, fontsize=11)

    # Bar value labels
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}", xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_ylim(0, max(raw_scores) * 1.15)
    plt.tight_layout()
    fig_path = RESULTS_DIR / "nasa_score_comparison.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Generated: {fig_path.name}")

    # -------------------------------------------------------------------------
    # Figure 5: Comprehensive Model Performance Comparison (Grouped Bar: MAE, RMSE)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    maes_raw = [eval_results[k]["metrics_raw"]["MAE"] for k in models.keys()]
    rmses_raw = [eval_results[k]["metrics_raw"]["RMSE"] for k in models.keys()]
    maes_clip = [eval_results[k]["metrics_clipped"]["MAE"] for k in models.keys()]
    rmses_clip = [eval_results[k]["metrics_clipped"]["RMSE"] for k in models.keys()]

    # Left: Raw Target MAE & RMSE
    rects_mae1 = ax1.bar(x - width/2, maes_raw, width, label="MAE", color="#457b9d", edgecolor="black")
    rects_rmse1 = ax1.bar(x + width/2, rmses_raw, width, label="RMSE", color="#1d3557", edgecolor="black")
    ax1.set_title("Standard Error Metrics: Raw Ground Truth", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Cycles (Lower is Better)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, fontsize=10)
    ax1.legend(framealpha=0.9)
    for rect in rects_mae1 + rects_rmse1:
        h = rect.get_height()
        ax1.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    ax1.set_ylim(0, max(rmses_raw) * 1.18)

    # Right: Clipped Target MAE & RMSE
    rects_mae2 = ax2.bar(x - width/2, maes_clip, width, label="MAE", color="#a8dadc", edgecolor="black")
    rects_rmse2 = ax2.bar(x + width/2, rmses_clip, width, label="RMSE", color="#e63946", edgecolor="black")
    ax2.set_title("Standard Error Metrics: Clipped Target (125 Cycles)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Cycles (Lower is Better)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels, fontsize=10)
    ax2.legend(framealpha=0.9)
    for rect in rects_mae2 + rects_rmse2:
        h = rect.get_height()
        ax2.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    ax2.set_ylim(0, max(rmses_raw) * 1.18)

    plt.suptitle("Official FD001 Test Set Regression Accuracy Comparison", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig_path = RESULTS_DIR / "model_performance_comparison.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Generated: {fig_path.name}")


def print_summary_table(models, stored_metrics):
    print("\n" + "=" * 105)
    print(f"{'MODEL ARCHITECTURE':<26} | {'OFFICIAL TEST MAE':<18} | {'OFFICIAL TEST RMSE':<18} | {'OFFICIAL NASA SCORE':<20} | {'VAL SEQ MAE':<12}")
    print(f"{'':<26} | {'(Raw / Clip125)':<18} | {'(Raw / Clip125)':<18} | {'(Raw / Clip125)':<20} | {'':<12}")
    print("-" * 105)
    for key, info in models.items():
        test_raw = stored_metrics[key]["official_test_set"]["raw_target"]
        test_clip = stored_metrics[key]["official_test_set"]["clipped_target"]
        val_seq = stored_metrics[key]["sequence_level"]["MAE"]
        print(f"{info['name']:<26} | {test_raw['MAE']:5.2f} / {test_clip['MAE']:5.2f} cycles | {test_raw['RMSE']:5.2f} / {test_clip['RMSE']:5.2f} cycles | {test_raw['NASA_score']:7.1f} / {test_clip['NASA_score']:7.1f}     | {val_seq:5.2f} cycles")
    print("=" * 105)


if __name__ == "__main__":
    run_full_evaluation()
