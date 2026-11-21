"""Empirical performance benchmark for Jet Engine Failure Predictor models and dashboard."""

import sys
import time
import json
import pathlib
import joblib
import tracemalloc
import numpy as np
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.train_lstm import LSTMPredictor
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "dataset" / "processed"


def benchmark_system():
    print("==================================================")
    print("JET ENGINE FAILURE PREDICTOR — PERFORMANCE BENCHMARK")
    print("==================================================")

    results = {}
    tracemalloc.start()

    # 1. Model Loading Times
    print("\n[1] Measuring Model Deserialization & Loading Times:")
    load_times = {}

    t0 = time.perf_counter()
    lr = joblib.load(MODELS_DIR / "baselines" / "linear_regression.pkl")
    load_times["Linear Regression"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    rf = joblib.load(MODELS_DIR / "baselines" / "random_forest.pkl")
    load_times["Random Forest"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    gb = joblib.load(MODELS_DIR / "baselines" / "gradient_boosting.pkl")
    load_times["Gradient Boosting"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    ckpt = torch.load(MODELS_DIR / "lstm" / "lstm_model.pt", map_location="cpu", weights_only=False)
    lstm = LSTMPredictor(input_dim=17)
    lstm.load_state_dict(ckpt["model_state_dict"])
    lstm.eval()
    load_times["Deep LSTM (PyTorch)"] = (time.perf_counter() - t0) * 1000

    for model_name, ms in load_times.items():
        print(f"  - {model_name:25s}: {ms:8.2f} ms")
    results["model_loading_times_ms"] = load_times

    # 2. Inference Latency (Single Sequence & Batch 100)
    print("\n[2] Measuring Inference Latency (100 Repetitions):")
    single_flat = np.random.randn(1, 510).astype(np.float32)
    single_3d = np.random.randn(1, 30, 17).astype(np.float32)
    batch_flat = np.random.randn(100, 510).astype(np.float32)
    batch_3d = np.random.randn(100, 30, 17).astype(np.float32)

    inference_latencies = {}
    models_dict = {
        "Linear Regression": (lr, single_flat, batch_flat),
        "Random Forest": (rf, single_flat, batch_flat),
        "Gradient Boosting": (gb, single_flat, batch_flat),
        "Deep LSTM (PyTorch)": (lstm, single_3d, batch_3d),
    }

    N_RUNS = 100
    for name, (m, s_in, b_in) in models_dict.items():
        # Warmup
        m.predict(s_in)
        
        # Single sequence latency
        single_times = []
        for _ in range(N_RUNS):
            t_start = time.perf_counter()
            m.predict(s_in)
            single_times.append((time.perf_counter() - t_start) * 1000)

        # Batch 100 latency
        batch_times = []
        for _ in range(N_RUNS):
            t_start = time.perf_counter()
            m.predict(b_in)
            batch_times.append((time.perf_counter() - t_start) * 1000)

        s_mean, s_p95 = np.mean(single_times), np.percentile(single_times, 95)
        b_mean, b_p95 = np.mean(batch_times), np.percentile(batch_times, 95)
        
        print(f"  - {name:25s}: Single: {s_mean:6.3f} ms (p95: {s_p95:6.3f} ms) | Batch-100: {b_mean:6.2f} ms")
        inference_latencies[name] = {
            "single_mean_ms": round(float(s_mean), 3),
            "single_p95_ms": round(float(s_p95), 3),
            "batch100_mean_ms": round(float(b_mean), 3),
        }
    results["inference_latencies"] = inference_latencies

    # 3. Memory Profiling
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print("\n[3] Memory Consumption (Tracemalloc):")
    print(f"  - Current Heap Allocation : {current_mem / (1024 * 1024):.2f} MB")
    print(f"  - Peak Memory Consumption : {peak_mem / (1024 * 1024):.2f} MB")
    results["memory_profile_mb"] = {
        "current_heap_mb": round(current_mem / (1024 * 1024), 2),
        "peak_memory_mb": round(peak_mem / (1024 * 1024), 2)
    }

    with open(PROJECT_ROOT / "results" / "metrics" / "performance_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark metrics to results/metrics/performance_benchmark.json")


if __name__ == "__main__":
    benchmark_system()
