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

### 1. Sliding-Window Sequence-Level Evaluation
Evaluates every overlapping 30-cycle window across all validation trajectories (3,242 total sequences). Reflects continuous online monitoring accuracy.

### 2. Per-Engine Truncated-Trajectory Evaluation
Evaluates the final 30-cycle observation for each validation unit at trajectory completion (20 units). Reflects end-of-life inspection accuracy before overhaul.

---

## 5. Benchmarking Summary

| Model | Sequence MAE | Sequence RMSE | Sequence NASA Score | Per-Engine MAE | Per-Engine RMSE | Per-Engine NASA Score |
|---|---|---|---|---|---|---|
| Linear Regression | 14.73 | 18.23 | 20,521.92 | 37.12 | 42.35 | 848.70 |
| Random Forest | 14.15 | 19.06 | 37,720.12 | 3.36 | 3.47 | 8.09 |
| Gradient Boosting | 13.01 | 17.56 | 26,177.00 | 4.52 | 4.71 | 11.57 |
| **LSTM (Deep Recurrent)** | **11.52** | **15.21** | **17,090.16** | **2.54** | **2.58** | **5.82** |
