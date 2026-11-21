"""Automated Performance and Latency Benchmarking Suite.

Measures:
1. Data loading and preprocessing latency per dataset (FD001-FD004, N-CMAPSS).
2. Model inference latency (P50, P95, P99, Mean) per sample and batch (batch size 1 and 32).
3. Memory footprint of dataframes and PyTorch state_dicts.
"""

import time
import json
import pathlib
import sys
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.dataset_registry import DatasetRegistry
from src.model_factory import create_model
from src.config_manager import ModelConfig


def benchmark_data_loading():
    results = {}
    for ds in DatasetRegistry.get_available_datasets():
        adapter = DatasetRegistry.get_adapter(ds)
        t0 = time.perf_counter()
        df = adapter.load_train(reload=True)
        t_load = time.perf_counter() - t0
        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        results[ds] = {
            "rows": len(df),
            "columns": len(df.columns),
            "load_time_ms": round(t_load * 1000, 2),
            "memory_mb": round(mem_mb, 2)
        }
    return results


def benchmark_model_inference():
    seq_len = 30
    input_dim = 14
    batch_single = torch.randn(1, seq_len, input_dim)
    batch_32 = torch.randn(32, seq_len, input_dim)

    architectures = [
        ("lstm", ModelConfig(architecture="lstm")),
        ("attention_lstm", ModelConfig(architecture="attention_lstm")),
        ("tcn", ModelConfig(architecture="tcn")),
        ("transformer", ModelConfig(architecture="transformer", transformer_d_model=64, transformer_nhead=4))
    ]

    results = {}
    num_runs = 200

    for name, cfg in architectures:
        model = create_model(cfg, input_dim=input_dim, seq_len=seq_len)
        model.eval()

        # Warmup
        for _ in range(20):
            with torch.no_grad():
                _ = model(batch_single)

        latencies_single = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(batch_single)
            latencies_single.append((time.perf_counter() - t0) * 1000)

        latencies_batch = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(batch_32)
            latencies_batch.append((time.perf_counter() - t0) * 1000)

        results[name] = {
            "single_sample": {
                "mean_ms": round(float(np.mean(latencies_single)), 3),
                "p50_ms": round(float(np.percentile(latencies_single, 50)), 3),
                "p95_ms": round(float(np.percentile(latencies_single, 95)), 3),
                "p99_ms": round(float(np.percentile(latencies_single, 99)), 3)
            },
            "batch_32": {
                "mean_ms": round(float(np.mean(latencies_batch)), 3),
                "throughput_samples_per_sec": round(32.0 / (np.mean(latencies_batch) / 1000), 1)
            }
        }
    return results


def run_benchmark():
    print("Running Data Loading Benchmark...")
    data_res = benchmark_data_loading()
    print("Running Model Inference Benchmark...")
    model_res = benchmark_model_inference()

    combined = {
        "data_loading": data_res,
        "inference_latency": model_res
    }

    out_file = PROJECT_ROOT / "results" / "performance_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"Benchmark completed successfully. Output written to {out_file}")
    return combined


if __name__ == "__main__":
    run_benchmark()
