# ===========================================================================
# JETWISE AI: Production Dockerfile
# AI-Powered Aircraft Engine Health Monitoring & Predictive Maintenance Platform
# ===========================================================================

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PORT=8501

WORKDIR /app

# Install system utilities and minimal libraries required for Playwright/Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Install pinned Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install Playwright browser engine (Chromium only for lightweight footprint)
RUN python -m playwright install chromium

# Copy codebase
COPY . /app/

# Create persistent storage directories
RUN mkdir -p /app/models /app/experiments /app/results /app/dataset/processed

# Expose Streamlit dashboard (8501) and FastAPI service (8000)
EXPOSE 8501 8000

# Docker Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default Command: Start the JETWISE AI Industrial Dashboard
CMD ["python", "-m", "streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
