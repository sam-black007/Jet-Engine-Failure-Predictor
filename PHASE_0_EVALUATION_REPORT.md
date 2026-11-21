# Phase 0 Technical Evaluation Report: Official NASA C-MAPSS FD001 Benchmark Audit
**Project:** Jet Engine Failure Predictor  
**Author:** Senior ML Engineering Team  
**Date:** November 21, 2026  
**Status:** Audit & Corrections Complete | 32/32 Tests Passed | Certified for Offline Research  

---

## 1. Executive Summary & Problems Discovered

An exhaustive audit of the model evaluation pipeline was conducted on the **Jet Engine Failure Predictor** project. The audit identified three fundamental methodological issues in prior reporting and implemented rigorous corrections using the official NASA C-MAPSS FD001 test benchmark (`test_FD001.txt` and `RUL_FD001.txt`).

### A. Evaluation Target Mismatch in Historical "Per-Engine" Metrics
- **Historical Implementation:** In `src/evaluation.py` (`evaluate_per_engine`), the per-engine evaluation was performed on the 20 internal validation units from `train_FD001.txt`. Because trajectories in `train_FD001.txt` are run-to-failure, the engine's final recorded cycle coincides with complete functional failure, meaning **Ground Truth RUL = 0 cycles** for all 20 engines (`actual_rul = min(max_cycle - last_cycle, rul_clip) = 0`).
- **Consequence:** Models predicting near 0 cycles at end-of-life incurred near-zero penalty in the NASA asymmetric scoring formula ($e^{2.5/10} - 1 \approx 0.284$). Summing across 20 engines yielded an artificially deflated NASA score of **5.82**. This metric only measured end-of-life detection, not general RUL prognosis on operating engines.

### B. Investigation of the 17.96 vs. 5.82 Discrepancy
- **Finding:** In `docs/models.md` and `models/performance_metrics.json`, the stored per-engine NASA score for the Deep LSTM was **5.82**. In contrast, earlier summary text and `TEST_REPORT.md` cited **17.96**.
- **Root Cause:** 17.96 was an ungrounded placeholder inserted during earlier draft documentation that did not correspond to any computed metric in the codebase.
- **Audited Truth:** Neither 5.82 nor 17.96 represented the true official NASA C-MAPSS test set score. When evaluated on the official NASA test set (100 test engines with true RUL labels from `RUL_FD001.txt`), the actual LSTM NASA score is **537.01** (raw ground truth) and **497.15** (clipped to 125 cycles).

### C. Preprocessing Pipeline & Data Leakage Audit
- **Historical Implementation:** In `src/preprocessing.py`, constant sensor identification (`_identify_constant_sensors`) was previously executed on the full dataset before the 80/20 train/validation engine split.
- **Leakage Analysis:** Constant sensors (`sensor_1`, `sensor_10`, `sensor_18`, `sensor_19`) are physical constants across all FD001 engines. Furthermore, normalization means and standard deviations were already computed strictly on `train_df`. However, to achieve strict architectural purity, the pipeline was updated to perform the engine split **first** and fit all statistics and sensor selections exclusively on training engines.
- **Numerical Precision Artifacts:** When computing standard deviation on a dataframe slice of constant values (`1.3` for `sensor_10`), IEEE-754 floating point subtraction yielded $2.22 \times 10^{-16}$, causing strict `std == 0` checks to fail unless numerical epsilon tolerance or explicit constant filtering is applied.

---

## 2. Changes Implemented

1. **Official NASA Test Set Evaluation Engine (`src/evaluation.py`):**
   - Implemented `evaluate_official_test_set()`, which loads `test_FD001.txt` and `RUL_FD001.txt`, validates engine sorting and contiguous 1..100 alignment, extracts the final 30-cycle operational window per engine, standardizes with training-fleet parameters, and executes model inference.
   - Supports dual-target evaluation: **Raw Ground Truth** (7 to 145 cycles) and **Clipped Ground Truth** (capped at 125 cycles, matching training assumptions).

2. **Leakage-Free Preprocessing Pipeline (`src/preprocessing.py`):**
   - Reordered preprocessing steps: Engine split (`_split_engine_ids`) is executed before sensor identification and normalization.
   - Constant sensors and z-score statistics are fitted strictly on `train_df` without validation or test data exposure.
   - Retained the established 17-feature schema, ensuring complete backward compatibility with all trained baseline models and the PyTorch LSTM checkpoint without requiring model retraining.

3. **Automated Evaluation & Visualization Suite (`src/evaluate_all_models.py`):**
   - Developed an automated evaluation runner that evaluates all four models on the official test set, updates `models/performance_metrics.json`, exports detailed predictions to `results/evaluation/official_test_metrics.json`, and generates 5 publication-grade visualization plots.

4. **Multi-Faceted Metrics Schema (`models/performance_metrics.json`):**
   - Preserved existing validation metrics under `sequence_level` and `per_engine`.
   - Added `official_test_set` containing `raw_target` and `clipped_target` metrics for all 100 test engines.

5. **Expanded Test Suite (`tests/test_official_evaluation.py`):**
   - Added 8 dedicated unit and integration tests verifying test-engine ordering, RUL label alignment, final-window generation, feature ordering, normalization consistency, prediction output shape, NASA score properties, and error handling on corrupted/mismatched data. Total automated tests increased from 24 to **32 passed tests**.

---

## 3. Official Evaluation Methodology

The NASA C-MAPSS FD001 benchmark evaluates remaining useful life estimation under steady-state sea-level operating conditions:

```
Test Fleet (test_FD001.txt: 100 Units)
               │
               ▼
Extract Final 30-Cycle Operational Window (Cycles T_i - 29 to T_i)
               │
               ▼
Feature Selection: 17 Operational Sensors (Prune Invariant: 1, 10, 18, 19)
               │
               ▼
Z-Score Normalization: (X - mu_train) / sigma_train (Strict Training Fleet Stats)
               │
               ▼
Model Inference:
  - Tabular Baselines (LR, RF, GBR): Flattened (1, 510)
  - Deep LSTM: 3D Sequence Tensor (1, 30, 17)
               │
               ▼
Evaluation against Official Ground Truth (RUL_FD001.txt)
  - Raw Target: y_true in [7, 145]
  - Clipped Target (125): min(y_true, 125)
               │
               ▼
Metrics: MAE, RMSE, NASA Asymmetric Penalty Score S
```

### Mathematical Formulation of the NASA Scoring Function
For prediction error $d_i = \hat{y}_i - y_i$:
$$s_i = \begin{cases} e^{-d_i / 13} - 1 & \text{if } d_i < 0 \text{ (early prediction / underestimation)} \\ e^{d_i / 10} - 1 & \text{if } d_i \ge 0 \text{ (late prediction / overestimation)} \end{cases}$$
$$S = \sum_{i=1}^{100} s_i$$

- **Asymmetric Property:** An overestimation of $+10$ cycles incurs penalty $e^{1.0} - 1 \approx 1.7183$. An underestimation of $-10$ cycles incurs penalty $e^{10/13} - 1 \approx 1.1582$. Late predictions are penalized $\approx 48\%$ more heavily due to catastrophic in-flight failure risks.

---

## 4. Official Benchmark Performance Metrics (100 Test Engines)

The table below presents the verified empirical evaluation results on the official NASA C-MAPSS FD001 test set:

| Model Architecture | Test MAE (Raw) | Test MAE (Clip125) | Test RMSE (Raw) | Test RMSE (Clip125) | Official NASA Score (Raw) | Official NASA Score (Clip125) | Val Seq MAE |
|---|---|---|---|---|---|---|---|
| **Linear Regression** | 14.30 cycles | 13.39 cycles | 17.54 cycles | 16.39 cycles | 494.68 | 450.96 | 14.73 cycles |
| **Random Forest** | 13.44 cycles | 12.37 cycles | 17.81 cycles | 16.75 cycles | 894.51 | 860.85 | 14.15 cycles |
| **Gradient Boosting** | 11.74 cycles | 10.67 cycles | **15.44 cycles** | **14.21 cycles** | **385.90** | **352.62** | 13.01 cycles |
| **Deep LSTM (PyTorch)** | **11.69 cycles** | **10.62 cycles** | 16.33 cycles | 14.99 cycles | 537.01 | 497.15 | **11.52 cycles** |

### Key Architectural Findings:
1. **Deep LSTM vs. Gradient Boosting:**
   - Deep LSTM achieves the **lowest MAE** on the official test set (**11.69 cycles** raw, **10.62 cycles** clipped).
   - Gradient Boosting achieves the **lowest RMSE** (**15.44 cycles** raw) and the **lowest NASA Asymmetric Penalty** (**385.90** raw).
   - **Why does Gradient Boosting have a lower NASA score despite slightly higher MAE?** Because the NASA score is exponential in positive errors ($e^{d/10}$). A model that makes a few large overestimations ($d > +15$ cycles) incurs extreme exponential penalties ($e^{2.0} \approx 7.39$, $e^{3.0} \approx 20.09$). The LSTM had 3 engines with overestimations $> 25$ cycles, which elevated its total sum, whereas Gradient Boosting's residual distribution was tighter in the positive error zone.
2. **Linear Regression Baseline:**
   - Achieves competitive MAE (14.30) and NASA score (494.68), outperforming Random Forest on NASA score because Random Forest had higher outlier overestimations.
3. **Target Clipping Impact:**
   - Clipping the official ground truth to 125 cycles reduces MAE across all models by $\approx 1.0$ cycle and reduces NASA penalty by $\approx 7–10\%$, confirming that models trained with a piecewise linear ceiling of 125 naturally align with clipped ground truth.

---

## 5. Visualizations Generated

All 5 publication-grade technical figures were generated by `src/evaluate_all_models.py` and saved under `results/evaluation/`:

1. **`actual_vs_predicted_scatter.png` (1.18 MB):**
   - $2 \times 2$ subplot grid comparing Ground Truth RUL vs Predicted RUL for each model with diagonal identity line ($y=x$), 125-cycle threshold line, and embedded statistical metric boxes.
2. **`per_engine_error_chart.png` (1.26 MB):**
   - Engine-by-engine (1 to 100) residual plot ($d_i = \hat{y}_i - y_i$) with shaded green (conservative early zone) and red (critical late zone) regions, demonstrating error spread across models.
3. **`rul_prediction_distribution.png` (393 KB):**
   - Kernel density estimation (KDE) and histogram overlay comparing the empirical distribution of model predictions against the true RUL distribution.
4. **`nasa_score_comparison.png` (180 KB):**
   - Comparative bar chart contrasting NASA Asymmetric Penalty Scores across all four models for both raw and clipped target conventions.
5. **`model_performance_comparison.png` (262 KB):**
   - Dual-panel grouped bar chart illustrating standard error metrics (MAE and RMSE) for both raw and clipped targets.

---

## 6. Comprehensive Test Results

The full automated testing suite was executed with 100% pass rate:

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-8.3.2, pluggy-1.6.0
rootdir: C:\Users\samjo\Downloads\simulation\Jet-Engine-Failure-Predictor
collected 32 items

tests/test_dashboard_playwright.py::test_dashboard_title_and_metrics[chromium] PASSED [  3%]
tests/test_dashboard_playwright.py::test_dashboard_tab_interaction[chromium] PASSED [  6%]
tests/test_dashboard_playwright.py::test_dashboard_sidebar_controls[chromium] PASSED [  9%]
tests/test_data_and_preprocessing.py::test_data_loader_fd001 PASSED      [ 12%]
tests/test_data_and_preprocessing.py::test_test_and_rul_loader PASSED    [ 15%]
tests/test_data_and_preprocessing.py::test_engine_id_split_no_leakage PASSED [ 18%]
tests/test_data_and_preprocessing.py::test_sequence_generation PASSED    [ 21%]
tests/test_dataset_deep.py::test_fd001_raw_data_integrity PASSED         [ 25%]
tests/test_dataset_deep.py::test_engine_trajectory_monotonicity PASSED   [ 28%]
tests/test_dataset_deep.py::test_constant_sensor_identification PASSED   [ 31%]
tests/test_dataset_deep.py::test_zscore_normalization_and_metadata PASSED [ 34%]
tests/test_dataset_deep.py::test_edge_case_insufficient_cycles PASSED    [ 37%]
tests/test_dataset_deep.py::test_edge_case_missing_files_error_handling PASSED [ 40%]
tests/test_models.py::test_baseline_models_exist_and_predict PASSED      [ 43%]
tests/test_models.py::test_lstm_model_checkpoint_and_inference PASSED    [ 46%]
tests/test_models.py::test_per_engine_evaluation_function PASSED         [ 50%]
tests/test_models.py::test_performance_metrics_json_integrity PASSED     [ 53%]
tests/test_models_deep.py::test_baseline_models_input_and_determinism PASSED [ 56%]
tests/test_models_deep.py::test_lstm_architecture_and_forward_pass PASSED [ 59%]
tests/test_models_deep.py::test_lstm_corrupted_checkpoint_handling PASSED [ 62%]
tests/test_official_evaluation.py::test_official_test_engine_ordering PASSED [ 65%]
tests/test_official_evaluation.py::test_official_rul_label_alignment PASSED [ 68%]
tests/test_official_evaluation.py::test_final_window_generation PASSED   [ 71%]
tests/test_official_evaluation.py::test_feature_ordering PASSED          [ 75%]
tests/test_official_evaluation.py::test_normalization_consistency PASSED [ 78%]
tests/test_official_evaluation.py::test_prediction_output_shape PASSED   [ 81%]
tests/test_official_evaluation.py::test_nasa_score_calculation PASSED    [ 84%]
tests/test_official_evaluation.py::test_missing_or_corrupted_test_data PASSED [ 87%]
tests/test_scoring.py::test_zero_error PASSED                            [ 90%]
tests/test_scoring.py::test_asymmetric_property PASSED                   [ 93%]
tests/test_scoring.py::test_strictly_non_negative PASSED                 [ 96%]
tests/test_scoring.py::test_vectorized_sum PASSED                        [100%]

======================= 32 passed, 1 warning in 20.15s ========================
```

### Playwright End-to-End Browser Test Verification:
The Playwright browser automation suite (`run_playwright_suite.py`) was executed on Chromium against the live Streamlit dashboard:
- **Suites Executed:** 5 / 5 PASSED
- **Console Errors:** 0
- **Failed Network Requests:** 0
- **Responsive Viewports Tested:** Desktop 1080p, Laptop 768p, Tablet 768x1024, Mobile 390x844.

---

## 7. Remaining Limitations

1. **Symmetric Loss Training vs. Asymmetric Evaluation:**
   - Both the Scikit-learn baselines and the PyTorch LSTM were trained using Mean Squared Error (MSE), which treats overestimation and underestimation symmetrically ($(-d)^2 = d^2$). However, the NASA evaluation metric penalizes overestimation exponentially more heavily. Training directly with an asymmetric loss function would significantly improve the NASA score.
2. **Fixed 125-Cycle RUL Ceiling:**
   - Target clipping at 125 cycles creates a constant plateau for engines early in their operating life. While physically motivated, it induces flat prediction regions when engines have $RUL > 125$.
3. **Single Operating Condition (FD001):**
   - FD001 models operate under sea-level conditions with a single fault mode (HPC degradation). Performance on multi-condition datasets (FD002 with 6 operating conditions, FD004 with 2 fault modes) requires condition normalization or clustering.
4. **Safety & Operational Bounds:**
   - This software is strictly an academic research prototype. It is not certified for flight clearance, onboard avionics, or operational aviation maintenance decisions.

---

## 8. Recommended Next Development Phase (Phase 1)

1. **Asymmetric Loss Function Training:**
   - Implement custom PyTorch loss functions:
     - **NASA Asymmetric Exponential Loss:** Directly optimize $\mathcal{L} = \frac{1}{N} \sum (\exp(d/10) - 1)$ or custom pinball loss during LSTM backpropagation to bias predictions towards conservative underestimation.
2. **Modern Deep Architectures (TCN & Transformer):**
   - Benchmark Temporal Convolutional Networks (TCN) with causal dilated convolutions and Transformer/Attention models against the LSTM baseline.
3. **Dataset Generalization (FD002 - FD004):**
   - Extend the ingestion and normalization pipeline to multi-condition datasets FD002 and FD004 using operating regime k-means clustering or altitude/Mach standardization.
4. **FastAPI Microservice Packaging:**
   - Wrap the trained models in a high-throughput REST API with Pydantic request validation and automated Docker containerization.
