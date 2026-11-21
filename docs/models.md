# Predictive Maintenance Model Specifications

## Project Context
This document outlines the machine learning and deep learning models designed for the Jet Engine Failure Predictor utilizing the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset (FD001).

> **Disclaimer:** This software is intended strictly for academic research and predictive maintenance experimentation. It is not certified for aviation flight clearance or onboard engine control systems.

---

## 1. Feature Engineering & Sequence Windowing
- **Input Features (17 Sensors):** `sensor_2`, `sensor_3`, `sensor_4`, `sensor_5`, `sensor_6`, `sensor_7`, `sensor_8`, `sensor_9`, `sensor_11`, `sensor_12`, `sensor_13`, `sensor_14`, `sensor_15`, `sensor_16`, `sensor_17`, `sensor_20`, `sensor_21`.
- **Filtered Constant Sensors:** `sensor_1`, `sensor_10`, `sensor_18`, `sensor_19` (zero variance across training units).
- **Temporal Window Length:** 30 consecutive operational cycles.
- **Normalization:** Per-sensor Z-score standardization calculated exclusively from the training fleet.
- **Target Variable:** Remaining Useful Life (RUL), clipped at an upper threshold of 125 cycles to model initial healthy steady-state operation.

---

## 2. Baseline Architectures (Scikit-Learn)

### A. Ordinary Least Squares (OLS) Linear Regression
- **Input Dimension:** 510 flattened features ($30 \times 17$).
- **Objective:** Establishes a transparent linear baseline to evaluate whether non-linear dynamics exist.
- **Characteristics:** Fast training, prone to underfitting during early operational life before rapid degradation commences.

### B. Random Forest Regressor
- **Hyperparameters:** `n_estimators=50`, `max_depth=15`, `random_state=42`.
- **Objective:** Non-linear ensemble model capturing non-linear sensor correlations and step thresholds.
- **Characteristics:** Robust to noise and outliers; provides stable ensemble predictions.

### C. Gradient Boosting Regressor
- **Hyperparameters:** `n_estimators=60`, `max_depth=5`, `learning_rate=0.1`, `random_state=42`.
- **Objective:** Sequential residual boosting to focus model capacity on high-error regions.
- **Characteristics:** Lower overall sequence prediction error among tabular baselines.

---

## 3. Deep Learning Architecture: LSTM Recurrent Neural Network

```
Input Sequence: (Batch Size, 30 Time Steps, 17 Normalized Sensors)
      │
      ▼
LSTM Layer 1 (Hidden Size = 64, Batch First, Sequence Output)
      │
      ▼
Dropout Regularization (p = 0.20)
      │
      ▼
LSTM Layer 2 (Hidden Size = 32, Batch First, Final State Output)
      │
      ▼
Dropout Regularization (p = 0.20)
      │
      ▼
Fully Connected Linear Layer (32 -> 1)
      │
      ▼
Continuous RUL Prediction (Scalar)
```

- **Framework:** PyTorch
- **Loss Function:** Mean Squared Error (MSE)
- **Optimization:** Adam (`lr=0.001`, `weight_decay=1e-5`)
- **Learning Rate Scheduler:** ReduceLROnPlateau (patience=3, factor=0.5)
- **Early Stopping:** Monitored on validation loss with patience of 8 epochs (converged at epoch 31, best checkpoint at epoch 23 with validation MSE 231.46).

---

## 4. Evaluation Methodologies

### 1. Sliding-Window Sequence-Level Validation
Evaluates every overlapping 30-cycle window across all validation trajectories (3,242 total sequences). Reflects continuous online monitoring accuracy during in-flight operational degradation.

### 2. Validation Fleet End-of-Life (EOL) Inspection
Evaluates the final 30-cycle observation for each validation unit at trajectory completion (20 units in `train_FD001.txt`). In run-to-failure data, the true RUL at the final observed cycle is 0 cycles. This tests the model's ability to identify imminent functional failure.

### 3. Official NASA C-MAPSS FD001 Test Set Benchmark
Evaluates each of the 100 independent test engines in `test_FD001.txt`. In this benchmark, each engine's operational data is truncated at a random operating cycle prior to failure. The final 30 cycles are extracted and normalized strictly using training statistics, and the predicted RUL is compared against the true ground-truth remaining life provided in `RUL_FD001.txt` (range: 7 to 145 cycles).

Evaluated under two standard conventions:
- **Raw Ground Truth:** Direct evaluation against `RUL_FD001.txt`.
- **Clipped Ground Truth (125 Cycles):** Target capped at 125 cycles matching the piecewise linear degradation assumption used during training.

---

## 5. Benchmarking Summary

### A. Official NASA C-MAPSS FD001 Test Set Benchmark (100 Test Engines)

| Model Architecture | Test MAE (Raw) | Test MAE (Clip125) | Test RMSE (Raw) | Test RMSE (Clip125) | Test NASA Score (Raw) | Test NASA Score (Clip125) |
|---|---|---|---|---|---|---|
| **Linear Regression** | 14.30 | 13.39 | 17.54 | 16.39 | 494.68 | 450.96 |
| **Random Forest** | 13.44 | 12.37 | 17.81 | 16.75 | 894.51 | 860.85 |
| **Gradient Boosting** | 11.74 | 10.67 | **15.44** | **14.21** | **385.90** | **352.62** |
| **Deep LSTM (PyTorch)** | **11.69** | **10.62** | 16.33 | 14.99 | 537.01 | 497.15 |

### B. Validation Fleet Metrics (20 Internal Validation Units)

| Model | Sequence MAE | Sequence RMSE | Sequence NASA Score | Val EOL MAE (RUL=0) | Val EOL RMSE | Val EOL NASA Score |
|---|---|---|---|---|---|---|
| Linear Regression | 14.73 | 18.23 | 20,521.92 | 37.12 | 42.35 | 848.70 |
| Random Forest | 14.15 | 19.06 | 37,720.12 | 3.36 | 3.47 | 8.09 |
| Gradient Boosting | 13.01 | 17.56 | 26,177.00 | 4.52 | 4.71 | 11.57 |
| **LSTM (Deep Recurrent)** | **11.52** | **15.21** | **17,090.16** | **2.54** | **2.58** | **5.82** |

---

## 6. Audit & Discrepancy Investigation Note

An audit of earlier repository documentation revealed a discrepancy where an LSTM per-engine NASA score of **17.96** was cited in certain reports, conflicting with the repository's stored metric of **5.82**.

- **Origin of 5.82:** This metric was computed by `evaluate_per_engine()` on the final cycle of the 20 internal validation units from `train_FD001.txt`. Because these trajectories ran to failure, ground truth RUL was 0. Predicting $\approx 2.5$ cycles incurred near-zero penalty in the asymmetric formula ($e^{0.25} - 1 \approx 0.28$ per engine, summing to $5.82$).
- **Origin of 17.96:** An ungrounded placeholder inserted in earlier summary drafts that did not correspond to any computed metric in `performance_metrics.json`.
- **True Official Benchmark:** On the official NASA C-MAPSS FD001 test set (100 engines with true RUL from `RUL_FD001.txt`), the Deep LSTM achieves an official test MAE of **11.69 cycles** (raw) / **10.62 cycles** (clipped) and a NASA score of **537.01** (raw) / **497.15** (clipped). Gradient Boosting achieves MAE of **11.74 cycles** (raw) / **10.67 cycles** (clipped) and NASA score of **385.90** (raw) / **352.62** (clipped).
