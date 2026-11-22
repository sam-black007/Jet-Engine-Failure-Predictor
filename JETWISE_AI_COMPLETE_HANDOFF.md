# THRUSTLAB — Autonomous Master Project Handoff Prompt

> **Instructions for User:** Copy and paste the prompt below into any fresh ChatGPT, Claude, or coding agent session to immediately resume, inspect, or build upon the **THRUSTLAB** platform with complete historical context and zero loss of momentum.

***

```markdown
# THRUSTLAB — AIRCRAFT ENGINE HEALTH MONITORING & DIGITAL TWIN PLATFORM (MASTER CONTEXT)

You are acting as the Senior Principal ML Engineer, Aerospace Prognostics Architect, and Full-Stack AI Lead taking ownership of the THRUSTLAB repository.

## 1. REPOSITORY & ENVIRONMENT CONTEXT
- Repository: https://github.com/sam-black007/Jet-Engine-Failure-Predictor
- Local Workspace: c:\Users\samjo\Downloads\simulation\Jet-Engine-Failure-Predictor
- Main Branch: main
- Python Version: Python 3.11.9
- PyTorch: 2.4.0+cpu
- Streamlit Dashboard: Running live on http://localhost:8501
- FastAPI Server: Ready on port 8000 via `python -m uvicorn api.main:app --port 8000`
- Test Suite: 101/101 Pytest tests passing (100% Green in ~52s)
- Browser Automation: 8/8 Playwright Chromium E2E scenarios passing across 5 responsive viewports (1920x1080 to 390x844)
- Two Mandatory Core Intelligence Features:
  1. Interactive 3D Aircraft Engine Digital Twin (9 physical subassemblies, Three.js WebGL canvas, OrbitControls, exploded view separation, raycasting click & HUD, Plotly 3D meshes, degradation playback, diagnostic notice).
  2. Advanced Predictive Maintenance Intelligence Engine (5-tier risk taxonomy, explainable MPI scoring breakdown, evidence-traceable recommendations, SQLite work orders & alert lifecycles, fleet prioritization matrix).

## 2. COMPLETED ARCHITECTURAL PHASES
All 10 project development phases have been implemented, mathematically audited, evaluated on official test benchmarks, and certified:

1. **Phase 0 (Evaluation Audit & Ground-Truth Alignment):**
   - Official NASA C-MAPSS FD001 test benchmark: 100 engines (`test_FD001.txt`, `RUL_FD001.txt`).
   - Fixed zero-leakage normalization (fit exclusively on training split before sequence windowing).
   - Standardized 17-sensor schema and verified exact label alignment.

2. **Phase 1 (Asymmetric & Quantile Losses):**
   - Implemented Log-Mean-Exp NASA loss and Pinball Quantile loss ($q=0.35$).
   - Slashed NASA asymmetric penalty from 537.01 down to 362.07 (-32.6%) while maintaining MAE ~12.5.
   - Checkpoint: `models/lstm_asymmetric/pinball_q35_lstm.pt` (Best official model).

3. **Phase 2 (Configurable Deep Learning Architectures & MLOps):**
   - Implemented Attention-LSTM, Temporal Convolutional Networks (TCN), and Temporal Transformer (`src/models/`).
   - Added Optuna Bayesian hyperparameter optimization and configuration manager (`configs/`).
   - Automated model registry and experiment tracking (`experiments/`).

4. **Phase 3 (Multi-Dataset Fleet Architecture & Replay Simulator):**
   - Unified adapter layer for NASA C-MAPSS FD001, FD002, FD003, FD004, and N-CMAPSS.
   - Standardized 21 physical sensor metadata dictionary with engineering units and nominal operating values.
   - Cycle-by-cycle historical telemetry replay simulator (`src/data_replay.py`).

5. **Phase 4 (Regime-Adaptive Multi-Condition Normalization):**
   - Resolved 304.70 vs 362.07 evaluation provenance discrepancy.
   - Implemented `RegimeDetector` (K-Means $k=6$ clustering on Altitude, Mach, and TRA) and `RegimeAdaptiveScaler`.
   - On FD002 (6 operating regimes), condition-wise MinMaxScaler slashed NASA asymmetric penalty by **51.4%** ($22,160 \rightarrow 10,763$).
   - On FD004, condition-wise RobustScaler dropped clipped MAE from 14.76 to 12.95 and NASA score by **37.2%**.

6. **Phase 5 (CI/CD, Docker & Reproducibility):**
   - Production Docker container (`Dockerfile`) with bundled Playwright Chromium, persistent data volumes, and healthcheck.
   - `docker-compose.yml` orchestrating dashboard (8501) and API (8000).
   - Multi-job GitHub Actions CI/CD matrix (`.github/workflows/ci.yml`).

7. **Phase 6 (Transfer Learning & Cross-Domain Adaptation):**
   - Zero-shot, frozen backbone, differential fine-tuning, and cold-start transfer matrix (`src/transfer_learning.py`).
   - Evaluated FD001 source model transferring to FD003, FD002, and FD004 with 50% target engine data.
   - Differential fine-tuning ($10^{-4}$ on temporal backbone, $10^{-3}$ on head) outperformed cold-start scratch training by **58–63%** in MAE and reduced NASA asymmetric penalties by **89–97%**.

8. **Phase 7 (Predictive Maintenance Intelligence Engine):**
   - `UncertaintyEstimator`: Monte Carlo Dropout with 25 stochastic passes yielding epistemic uncertainty std dev and $[q_{10}, q_{90}]$ prediction intervals.
   - `EngineAnomalyDetector`: Unsupervised `IsolationForest` on baseline residuals detecting sensor drift and anomalous z-score outliers.
   - `SequenceFeatureAttribution`: Integrated Gradients sequence attribution for deep models, highlighting top 3 physical degradation drivers.
   - `MaintenanceDecisionEngine`: Health classification (`HEALTHY`, `MONITOR`, `MAINTENANCE_DUE`, `CRITICAL`), composite Maintenance Priority Index (MPI, 0–100), and physical inspection recommendations (e.g. HPT blade borescope, lube chip detector check).
   - `MaintenanceDatabase`: Persistent SQLite database (`database/maintenance_records.db`) with tables for fleet engines, predictions log, sensor alerts, and automated work orders.

9. **Phase 8 (SCADA Dark-Mode Dashboard & 3D Turbofan Digital Twin):**
   - Aerospace SCADA dark tokens (`#0B0F14` background, `#101720` sidebar, `#00B8D9` cyan accent, semantic health badges).
   - Interactive **WebGL Three.js 3D Turbofan Digital Twin**: Real-time rotating fan blades, cutaway nacelle casing, compressor spool discs, glowing combustor core, and diagnostic status HUD overlay.
   - **Plotly 3D Aerothermal Surface Heat-Map**: 3D conical surface meshes representing Fan, LPC, HPC, Combustor, HPT, LPT, and Exhaust Nozzle with surface temperatures ($T_{24}, T_{30}, T_{50}$) and station inspection markers.
   - 12 unified modules in `dashboard/app.py`.

10. **Phase 9 (FastAPI Backend & Real-Time Telemetry Streaming):**
    - Production REST endpoints (`/api/health`, `/api/datasets`, `/api/fleet/{dataset}`, `/api/engine/{dataset}/{id}`, `/api/predict`, `/api/maintenance/work-orders`).
    - Server-Sent Events (SSE) streaming endpoint (`/api/stream/telemetry/{dataset}/{id}`) emitting historical flight cycles one by one with live predicted RUL.
    - Verified via 9 unit tests in `tests/test_api.py`.

11. **Phase 10 (Production Hardening & Certification):**
    - 93/93 Pytest test suite passing 100% green.
    - Full Playwright responsive browser suite certified across 5 viewports with 11 screenshot artifacts in `results/playwright/`.
    - Complete validation reports: `FINAL_THRUSTLAB_AI_VALIDATION_REPORT.md`, `PROJECT_MASTER_ROADMAP.md`.

## 3. HOW TO RUN THE PLATFORM LOCALLY

### Start the SCADA Dark-Mode Dashboard
```powershell
cd "c:\Users\samjo\Downloads\simulation\Jet-Engine-Failure-Predictor"
python -m streamlit run dashboard/app.py --server.port 8501
```
Open your browser at `http://localhost:8501`.

### Start the FastAPI Backend & Telemetry Streaming Server
```powershell
cd "c:\Users\samjo\Downloads\simulation\Jet-Engine-Failure-Predictor"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Interactive Swagger API documentation: `http://localhost:8000/docs`.

### Run the Full Test Suite
```powershell
cd "c:\Users\samjo\Downloads\simulation\Jet-Engine-Failure-Predictor"
python -m pytest tests/ -v
```

### Run Playwright Browser Automation & Capture Screenshots
```powershell
cd "c:\Users\samjo\Downloads\simulation\Jet-Engine-Failure-Predictor"
python run_playwright_suite.py
```

## 4. KEY REPOSITORY FILE MAP
- `dashboard/app.py`: Industrial SCADA dark-mode Streamlit dashboard with 3D digital twin.
- `src/digital_twin.py`: Three.js WebGL & Plotly 3D mesh generator with thermal surface mapping.
- `src/predictive_intelligence.py`: Uncertainty (MC Dropout), Anomaly (Isolation Forest), XAI (Integrated Gradients), MPI decision engine, SQLite database.
- `src/regime_normalizer.py`: K-Means $k=6$ clustering on flight conditions and condition-wise scalers.
- `src/transfer_learning.py`: Cross-domain adaptation engine across FD001-FD004.
- `src/dataset_registry.py` & `src/dataset_adapters/`: Unified dataset loaders for FD001-FD004 & N-CMAPSS.
- `src/train_asymmetric_lstm.py` & `src/losses.py`: NASA Log-Mean-Exp & Pinball quantile losses.
- `api/main.py`: Production FastAPI service and Server-Sent Events (SSE) telemetry stream.
- `models/lstm_asymmetric/pinball_q35_lstm.pt`: Best official NASA score checkpoint (362.07 raw, 277.23 clipped).
- `database/maintenance_records.db`: SQLite persistent fleet and work order store.
- `results/playwright/`: High-resolution visual screenshots across 5 responsive viewports.

Your mission is to maintain, extend, deploy, or run advanced experiments on this industrial aerospace intelligence platform following the existing rigorous zero-leakage standards.
```
