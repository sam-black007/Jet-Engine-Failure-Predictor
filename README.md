# Jet Engine Failure Predictor (NASA C-MAPSS)

An end-to-end predictive maintenance and prognostic health management (PHM) system built with PyTorch, Scikit-Learn, and Streamlit to forecast the Remaining Useful Life (RUL) of turbofan engines using simulated degradation telemetry.

---

## 1. Project Overview & Problem Statement
High-bypass turbofan engines operate under severe thermal and mechanical stresses. Unscheduled engine removals cause flight cancellations and catastrophic flight safety hazards, while premature servicing incurs excessive maintenance overhead. 

Predictive maintenance leverages continuous sensor streams to estimate Remaining Useful Life (RUL)—the number of operational cycles an engine can safely fly before requiring depot-level overhaul.

---

## 2. Dataset Description
This project utilizes the **NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)** dataset:
- **Subset:** FD001 (sea-level, single operating condition, high-pressure compressor degradation).
- **Fleet Size:** 100 engine units in the training set (run until functional failure).
- **Telemetry:** 3 operating settings and 21 sensor measurements per operational flight cycle.
- **Data Source:** NASA Prognostics Center of Excellence (PCoE).

---

## 3. Data Preprocessing Pipeline
Implemented in `src/preprocessing.py`:
1. **Engine-Wise Partitioning:** Split 80 training units (14,489 sequences) and 20 validation units (3,242 sequences) with zero engine overlap to eliminate temporal data leakage.
2. **Sensor Pruning:** 4 invariant sensors with zero variance (`sensor_1`, `sensor_10`, `sensor_18`, `sensor_19`) are dropped. 17 active sensor channels are retained.
3. **Z-Score Normalization:** Calculated strictly using training-engine statistics:
   $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
4. **Sliding Windowing:** Formulated into sequences of 30 cycles per window. Sequences never cross engine unit boundaries.
5. **RUL Target Clipping:** Target RUL is capped at 125 cycles to reflect constant health during initial wear before failure mode propagation begins.

---

## 4. Modeling Architectures

### Baseline Models
- **Linear Regression:** Standard ordinary least squares over flattened 510-dimensional window vectors ($30 \times 17$).
- **Random Forest Regressor:** 50 decision trees (`max_depth=15`, `random_state=42`).
- **Gradient Boosting Regressor:** 60 boosted stages (`max_depth=5`, `learning_rate=0.1`, `random_state=42`).

### Deep LSTM Recurrent Neural Network
Implemented in `src/train_lstm.py` using PyTorch:
- **Input:** Tensor of shape `(Batch, 30, 17)`.
- **LSTM Layer 1:** 64 hidden units, batch-first.
- **Dropout 1:** $p = 0.20$.
- **LSTM Layer 2:** 32 hidden units, returns final sequence state.
- **Dropout 2:** $p = 0.20$.
- **Dense Linear Head:** 32 units mapped to a single scalar RUL prediction.
- **Optimizer:** Adam (`lr=0.001`, `weight_decay=1e-5`) with `ReduceLROnPlateau`.
- **Checkpointing:** Best checkpoint saved at epoch 23 (`models/lstm/lstm_model.pt`).

---

## 5. Model Evaluation Methodologies & Benchmark

We report two distinct validation methodologies:
1. **Sliding-Window Validation (Sequence-Level):** Evaluates all 3,242 sliding windows in the validation fleet. Reflects continuous online health tracking.
2. **Per-Engine Truncated-Trajectory Evaluation:** Evaluates the final available 30-cycle observation for each validation unit at trajectory completion (20 engines). Reflects end-of-life maintenance triage.

### Official NASA Asymmetric Scoring Function
For error $d_i = \hat{y}_i - y_i$:
- If $d_i < 0$ (early prediction): $s_i = e^{-d_i / 13} - 1$
- If $d_i \ge 0$ (late prediction): $s_i = e^{d_i / 10} - 1$
Total penalty $S = \sum s_i$ heavily penalizes overestimating remaining engine life (late prediction) over conservative underestimation.

### Performance Benchmarking (From `models/performance_metrics.json`)

#### 1. Official NASA C-MAPSS FD001 Test Set Benchmark (100 Test Engines)
Evaluated on `test_FD001.txt` trajectories truncated prior to failure, compared against ground-truth remaining useful life in `RUL_FD001.txt`:

| Model Architecture | Official Test MAE (Raw / Clip125) | Official Test RMSE (Raw / Clip125) | Official NASA Score (Raw / Clip125) |
|---|---|---|---|
| **Linear Regression** | 14.30 / 13.39 cycles | 17.54 / 16.39 cycles | 494.68 / 450.96 |
| **Random Forest** | 13.44 / 12.37 cycles | 17.81 / 16.75 cycles | 894.51 / 860.85 |
| **Gradient Boosting** | 11.74 / 10.67 cycles | **15.44** / **14.21** cycles | **385.90** / **352.62** |
| **Deep LSTM (PyTorch)** | **11.69** / **10.62** cycles | 16.33 / 14.99 cycles | 537.01 / 497.15 |

*Note: Raw Target evaluates against the unclipped NASA ground-truth RUL (7–145 cycles); Clipped Target evaluates with an upper threshold of 125 cycles matching the piecewise linear degradation assumption used during training.*

#### 2. Engine-Disjoint Validation Fleet Metrics (20 Validation Engines)
Evaluated on the internal 80/20 train/validation split across 3,242 sliding sequence windows and 20 end-of-life inspection points:

| Model | Sequence MAE | Sequence RMSE | Sequence NASA Score | Val EOL MAE* | Val EOL RMSE* | Val EOL NASA Score* |
|---|---|---|---|---|---|---|
| **Linear Regression** | 14.73 | 18.23 | 20,521.92 | 37.12 | 42.35 | 848.70 |
| **Random Forest** | 14.15 | 19.06 | 37,720.12 | 3.36 | 3.47 | 8.09 |
| **Gradient Boosting** | 13.01 | 17.56 | 26,177.00 | 4.52 | 4.71 | 11.57 |
| **Deep LSTM (PyTorch)** | **11.52** | **15.21** | **17,090.16** | **2.54** | **2.58** | **5.82** |

*\*Val EOL represents final observed window in run-to-failure training data where ground truth RUL = 0.*

---

## 6. Streamlit Industrial Dashboard
The application (`dashboard/app.py`) provides:
- **Fleet Health Overview:** Displays total unit counts, active telemetry statistics, and dataset configurations.
- **Engine Selection:** Interactive dropdown for all 100 fleet engines.
- **Real-Time RUL Prognosis:** Instant RUL predictions with health status categorization (Normal / Caution / Critical).
- **Sensor Telemetry Explorer:** Multi-sensor interactive time-series plots.
- **Degradation Trajectory View:** Plots the ground-truth wear curve over cycles.
- **Fleet Actual vs. Predicted Validation:** Comparative bar plots and error tables across the validation fleet.
- **Model Comparison Table:** Dynamic benchmarking between Baselines and LSTM.
- **Predictive Maintenance Dispatch:** Automated alerts flagging engines with RUL below a configurable safety margin (default 30 cycles).

---

## 7. Installation & Quickstart

### Prerequisites
- Python 3.10 or 3.11

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd Jet-Engine-Failure-Predictor

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Dashboard
```bash
streamlit run dashboard/app.py
```

### Executing Tests
```bash
python -m pytest tests/ -v
```

---

## 8. Repository Structure

```text
Jet-Engine-Failure-Predictor/
│
├── .github/
│   └── workflows/
│       └── tests.yml            # CI automated testing pipeline
│
├── dashboard/
│   ├── app.py                   # Streamlit industrial monitoring interface
│   └── requirements.txt         # Dashboard dependencies
│
├── dataset/
│   ├── raw/                     # Raw NASA C-MAPSS text files & archive
│   └── processed/
│       ├── preprocess_metadata.json # Sensor statistics & partition IDs
│       └── (X_train, X_val, y_train, y_val).npy
│
├── docs/
│   └── models.md                # In-depth architectural documentation
│
├── models/
│   ├── baselines/               # Fitted Scikit-learn baseline models (.pkl)
│   ├── lstm/                    # Trained PyTorch LSTM checkpoint (.pt)
│   └── performance_metrics.json # Validated evaluation metrics
│
├── notebooks/
│   └── exploratory_analysis.ipynb # Sensor EDA and variance plots
│
├── src/
│   ├── data_loader.py           # Robust C-MAPSS ingestion utility
│   ├── preprocessing.py         # Windowing, clipping, and normalization
│   ├── evaluation.py            # NASA scoring and per-engine metrics
│   ├── train_baselines.py       # Baseline models training script
│   └── train_lstm.py            # PyTorch LSTM network training script
│
├── tests/
│   ├── test_data_and_preprocessing.py
│   ├── test_models.py
│   └── test_scoring.py
│
├── .gitignore
├── LICENSE                      # Apache-2.0 open-source license
├── README.md                    # Technical documentation
├── requirements.txt             # Project dependencies
└── status.md                    # Project development status tracker
```

---

## 9. Limitations & Future Work
- **Single Operating Regime:** Evaluated on FD001 (sea-level, steady condition). Future work includes adapting models to FD002 and FD004 (6 operating flight conditions).
- **Composite Degradation Modes:** Investigating multi-head attention mechanisms and Transformer-based time-series architectures for multi-fault regimes (HPC + Fan degradation).

---

## 10. Disclaimer & Citation
This software is developed for predictive maintenance research and engineering benchmarking. It is not approved for actual flight dispatch or critical avionics operations.

Dataset credit: *A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set", NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.*
