# PHASE 5 TECHNICAL REPORT: CI/CD, DOCKER & REPRODUCIBLE ENVIRONMENTS

**Document ID:** JW-P5-CICD-2026  
**Status:** COMPLETED & VERIFIED  
**Author:** JETWISE AI Architecture & DevOps Engineering Team  
**Repository:** `https://github.com/sam-black007/Jet-Engine-Failure-Predictor`  
**Date:** 2026-11-22  

---

## 1. Executive Summary

Phase 5 establishes a containerized, reproducible deployment and automated continuous integration pipeline for JETWISE AI:
1. **Production Dockerfile:** Multi-stage image based on `python:3.11-slim` with bundled system rendering dependencies, pinned dependencies from `requirements.txt`, headless Chromium for browser automation, persistent data volume mount points, and automated health checks.
2. **Docker Compose (`docker-compose.yml`):** Multi-service orchestration managing the Streamlit industrial dashboard (Port 8501) and the FastAPI streaming telemetry backend (Port 8000) with volume persistence for `./models`, `./experiments`, `./results`, and `./dataset`.
3. **GitHub Actions CI/CD Pipeline (`.github/workflows/ci.yml`):** Multi-job matrix pipeline automating code syntax verification, full 79-test backend pytest execution, live background Streamlit startup, Playwright Chromium browser automation across multiple resolutions, artifact archiving, and automated Docker image build caching.
4. **Dependency Pinning (`requirements.txt`):** Pinned ranges for NumPy, PyTorch, Pandas, Scikit-Learn, Pydantic v2, PyYAML, Optuna, Streamlit, Plotly, Playwright, and FastAPI.

---

## 2. Container Architecture & Orchestration

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DOCKER COMPOSE TOPOLOGY                         │
│                                                                        │
│  ┌───────────────────────────────┐   ┌───────────────────────────────┐ │
│  │     jetwise_ai_dashboard      │   │        jetwise_ai_api         │ │
│  │   Streamlit Industrial UI     │   │      FastAPI Telemetry        │ │
│  │       Port: 8501:8501         │   │       Port: 8000:8000         │ │
│  │   Healthcheck: /_stcore/health│   │    WebSockets & REST APIs     │ │
│  └───────────────┬───────────────┘   └───────────────┬───────────────┘ │
│                  │                                   │                 │
│                  └─────────────────┬─────────────────┘                 │
│                                    ▼                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     HOST PERSISTENT VOLUMES                      │  │
│  │   ./models  │  ./experiments  │  ./results  │  ./dataset         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Dockerfile Specifications
- **Base Image:** `python:3.11-slim`
- **System Libraries:** Minimal headless rendering packages (`libgl1`, `libnss3`, `libatk1.0-0`, `libpango-1.0-0`, `libcairo2`, `libasound2`, `curl`, `git`) for both Streamlit charting and Chromium headless testing.
- **Port Bindings:** `8501` (Dashboard), `8000` (FastAPI).
- **Healthcheck Probe:** `curl -f http://localhost:8501/_stcore/health || exit 1` every 30s.

### 2.2 Docker Compose Services
- `dashboard`: Runs `python -m streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true`.
- `api`: Runs `python -m uvicorn api.main:app --host 0.0.0.0 --port 8000` (connecting to Phase 9 streaming service).
- **Shared Volumes:**
  - `./models`: Preserves trained deep learning checkpoints (`models/lstm/`, `models/lstm_asymmetric/`, `models/transformer/`, `models/model_registry.json`).
  - `./experiments`: Preserves SQLite database (`experiments/experiments.db`).
  - `./results`: Preserves evaluation JSONs, Plotly figures, and Playwright screenshot evidence.
  - `./dataset`: Mounts raw C-MAPSS and processed parquet files.

---

## 3. GitHub Actions CI/CD Matrix Pipeline

The workflow file `.github/workflows/ci.yml` executes three concurrent and staged jobs on every push and pull request to `main`:

1. **Job 1: `test-backend`:**
   - Installs dependencies from `requirements.txt`.
   - Runs syntax compilation check across `src/*.py` and `dashboard/*.py`.
   - Executes the 79-test backend test suite (`python -m pytest tests/ -v`).
2. **Job 2: `test-e2e-playwright`:**
   - Provisions headless Chromium via `playwright install --with-deps chromium`.
   - Launches the Streamlit server in the background on port 8501.
   - Executes `run_playwright_suite.py` verifying DOM loading, tab interactions, model swapping, dataset switching, and responsive viewport rendering.
   - Archives visual evidence (`results/playwright/*.png`) as downloadable GitHub build artifacts.
3. **Job 3: `docker-build-validation`:**
   - Executes a GitHub Actions Buildx build against `Dockerfile` using GitHub Actions layer cache (`type=gha`) to ensure zero broken container builds reach production.

---

## 4. Verification and Next Steps

- Docker configuration and CI workflows verified.
- Proceeding automatically to **Phase 6: Transfer Learning & Domain Adaptation**.
