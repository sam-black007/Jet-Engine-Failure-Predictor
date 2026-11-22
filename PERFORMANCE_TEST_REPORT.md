# THRUSTLAB: SYSTEM PERFORMANCE, LATENCY & RESOURCE PROFILING REPORT

**Document ID:** JW-PRF-2026-V3  
**Status:** VERIFIED & MEASURED  
**Execution Timestamp:** 2026-11-21  
**Hardware Environment:** Intel Core / AMD64 Architecture, Windows 11 Host, Python 3.11.9  
**Target Deployments:** Aerospace Onboard Edge (FADEC / EHM) vs. Ground Maintenance Hangar Server  

---

## 1. Executive Summary

This report documents measured performance benchmarks for data loading, memory footprint, single-sequence inference latency, and batch throughput across all supported datasets and neural architectures in THRUSTLAB.

All figures presented below are empirically measured using `src/benchmark_performance.py` over 200 consecutive inference runs and serialized to `results/performance_benchmark.json`.

---

## 2. Data Loading & Ingestion Profiling

Each dataset subset was loaded from raw space-delimited text files or high-fidelity schemas.

| Dataset Identifier | Total Fleet Records | Feature Columns | Disk Ingestion Time (ms) | In-Memory Footprint (MB) | Read Throughput (records/sec) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FD001** (Baseline) | 20,631 | 26 | **200.62 ms** | 4.09 MB | 102,836 |
| **FD002** (6 Regimes) | 53,759 | 26 | **558.96 ms** | 10.66 MB | 96,177 |
| **FD003** (2 Faults) | 24,720 | 26 | **250.59 ms** | 4.90 MB | 98,647 |
| **FD004** (Complex) | 61,249 | 26 | **674.31 ms** | 12.15 MB | 90,832 |
| **N-CMAPSS** (Sample) | 849 | 21 | **10.83 ms** | 0.14 MB | 78,393 |

*Key Takeaway:* Even the largest C-MAPSS subset (FD004 with 61k rows) loads from raw disk in under 675 milliseconds with an in-memory footprint of only 12.15 MB, well within standard embedded aviation avionics hardware constraints.

---

## 3. Deep Prognostic Architecture Latency Profiling

Each model architecture was evaluated on single-sequence inference (sequence length = 30 cycles, feature dimension = 14) over 200 iterations following 20 warmup cycles.

### 3.1 Single-Sequence Latency Percentiles (Batch Size = 1)

| Model Architecture | Parameter Count | Mean Latency (ms) | P50 Median (ms) | P95 Latency (ms) | P99 Peak (ms) | Edge Feasibility |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Temporal Transformer** | ~75,000 | **0.937 ms** | **0.904 ms** | **1.125 ms** | **1.180 ms** | **Ultra-Fast (<1 ms)** |
| **Temporal ConvNet (TCN)** | ~45,000 | **1.598 ms** | **1.552 ms** | **1.864 ms** | **2.057 ms** | **Real-Time (<2 ms)** |
| **Deep LSTM (2-Layer)** | ~32,000 | **2.181 ms** | **2.147 ms** | **2.427 ms** | **2.743 ms** | **Real-Time (<3 ms)** |
| **Attention LSTM** | ~38,000 | **2.450 ms** | **2.399 ms** | **2.853 ms** | **3.307 ms** | **Real-Time (<3.5 ms)** |

```
Inference Latency Comparison (P50 Median in ms):
Transformer   ████ 0.90 ms
TCN           ███████ 1.55 ms
Deep LSTM     ██████████ 2.15 ms
AttentionLSTM ███████████ 2.40 ms
```

### 3.2 High-Throughput Batch Processing (Batch Size = 32)

For batch offline fleet screening (evaluating 100+ engines simultaneously in ground maintenance hangars):

| Model Architecture | Batch Latency (ms) | Batch Throughput (Engines/sec) |
| :--- | :---: | :---: |
| **Temporal Transformer** | **2.338 ms** | **13,686.4 engines/sec** |
| **Temporal ConvNet (TCN)** | **3.592 ms** | **8,909.9 engines/sec** |
| **Deep LSTM** | **4.231 ms** | **7,562.7 engines/sec** |
| **Attention LSTM** | **4.688 ms** | **6,826.1 engines/sec** |

---

## 4. Classical ML Baselines Latency Comparison

| Model Architecture | Model Size on Disk (KB) | Single Inference Latency (ms) | Memory Impact |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 2.1 KB | **0.08 ms** | Minimal (< 1 KB) |
| **Gradient Boosting** | 215.4 KB | **0.62 ms** | Low (< 5 MB) |
| **Random Forest (100 Trees)**| 1,842.0 KB | **8.45 ms** | Moderate (~15 MB) |

---

## 5. Deployment Recommendations for Aerospace Hardware

1. **Onboard Aircraft FADEC (Full Authority Digital Engine Control):**
   - **Recommended Architecture:** Temporal ConvNet (TCN) or Deep LSTM (quantized to INT8).
   - **Rationale:** Deterministic execution time, zero dynamic attention memory overhead, fixed causal receptive field, and sub-2 ms inference time fits comfortably inside standard 20 ms avionics control loop deadlines.
2. **Ground-Based Fleet Maintenance Hangar Servers:**
   - **Recommended Architecture:** Pinball Asymmetric LSTM (q=0.35) or Temporal Transformer.
   - **Rationale:** Highest prognostic safety margin against late failures (penalizes catastrophic over-prediction) and massive batch throughput exceeding 13,000 engines/second.
