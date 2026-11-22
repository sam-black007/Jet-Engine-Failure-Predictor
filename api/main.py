"""FastAPI Backend and Real-Time Telemetry Streaming Server (JETWISE AI Phase 9).

Provides:
- REST API for fleet management, engine telemetry, and deep prognostic RUL inference
- Monte Carlo uncertainty and anomaly detection endpoints
- Persistent SQLite maintenance work order dispatch
- Server-Sent Events (SSE) real-time flight cycle telemetry streaming
"""

import sys
import json
import asyncio
import pathlib
import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.dataset_registry import DatasetRegistry
from src.data_replay import EngineReplaySimulator
from src.train_lstm import LSTMPredictor
from src.predictive_intelligence import (
    MaintenanceDatabase,
    MaintenanceDecisionEngine,
    UncertaintyEstimator,
    EngineAnomalyDetector,
    HEALTH_STATES
)

# ---------------------------------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="JETWISE AI — Aerospace Prognostics & Digital Twin API",
    description="Production REST & Telemetry Streaming API for Turbofan Jet Engine RUL Prognosis and Predictive Maintenance.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database & Intelligence instances
db = MaintenanceDatabase()
decision_engine = MaintenanceDecisionEngine()

FEATURE_COLS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_5", "sensor_6", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_20", "sensor_21"
]

# Load default neural network
_MODEL_CACHE: Dict[str, Any] = {}


def get_model(model_name: str = "pinball_lstm_q35") -> nn.Module if "nn" in globals() else Any:
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    pinball_path = PROJECT_ROOT / "models" / "lstm_asymmetric" / "pinball_q35_lstm.pt"
    lstm_path = PROJECT_ROOT / "models" / "lstm" / "lstm_model.pt"

    path = pinball_path if pinball_path.exists() else lstm_path
    if not path.exists():
        raise RuntimeError("No pre-trained PyTorch model found in models/ directory.")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    m = LSTMPredictor(input_dim=17)
    m.load_state_dict({k: v for k, v in state.items() if k in m.state_dict()})
    m.eval()
    _MODEL_CACHE[model_name] = m
    return m


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    dataset: str = Field(default="FD001", description="Dataset identifier (FD001-FD004 or N-CMAPSS)")
    engine_id: Optional[int] = Field(default=None, description="Engine ID to pull latest telemetry window")
    sequence: Optional[List[List[float]]] = Field(default=None, description="Optional custom 30x17 sequence matrix")
    model_name: str = Field(default="pinball_lstm_q35", description="Model architecture key")


class PredictResponse(BaseModel):
    engine_id: Optional[int]
    dataset: str
    predicted_rul: float
    uncertainty: Dict[str, Any]
    anomaly: Dict[str, Any]
    health_state: str
    maintenance_priority_index: float
    priority: str
    recommended_actions: List[Dict[str, Any]]


class WorkOrderCreate(BaseModel):
    engine_id: int
    dataset_subset: str
    priority: str
    title: str
    subsystem: str
    description: str


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["General"])
def root():
    return {
        "platform": "JETWISE AI",
        "description": "Aerospace Turbofan Digital Twin & Prognostics API",
        "version": "1.0.0",
        "documentation": "/docs"
    }


@app.get("/api/health", tags=["General"])
def health_check():
    return {
        "status": "HEALTHY",
        "timestamp": datetime.datetime.now().isoformat(),
        "available_datasets": DatasetRegistry.get_available_datasets(),
        "database_connected": True
    }


@app.get("/api/datasets", tags=["Fleet & Datasets"])
def list_datasets():
    datasets = []
    for d_name in DatasetRegistry.get_available_datasets():
        adapter = DatasetRegistry.get_adapter(d_name)
        meta = adapter.get_metadata()
        datasets.append({
            "name": d_name,
            "complexity": meta.get("complexity", "Standard"),
            "conditions_count": meta.get("conditions_count", 1),
            "fault_modes": meta.get("fault_modes", []),
            "train_engines_count": len(adapter.get_engine_ids(kind="train"))
        })
    return {"datasets": datasets}


@app.get("/api/fleet/{dataset}", tags=["Fleet & Datasets"])
def get_fleet(dataset: str):
    if dataset not in DatasetRegistry.get_available_datasets():
        raise HTTPException(status_code=404, detail=f"Dataset {dataset} not found.")

    adapter = DatasetRegistry.get_adapter(dataset)
    engine_ids = adapter.get_engine_ids(kind="train")
    summary = db.get_fleet_summary(dataset)

    return {
        "dataset": dataset,
        "total_engines": len(engine_ids),
        "engine_ids": engine_ids,
        "fleet_summary": summary
    }


@app.get("/api/engine/{dataset}/{engine_id}", tags=["Fleet & Datasets"])
def get_engine_details(dataset: str, engine_id: int):
    adapter = DatasetRegistry.get_adapter(dataset)
    try:
        traj = adapter.get_engine_trajectory(engine_id, kind="train")
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "engine_id": engine_id,
        "dataset": dataset,
        "total_cycles": int(traj["cycle"].max()),
        "first_cycle_telemetry": traj.iloc[0].to_dict(),
        "latest_cycle_telemetry": traj.iloc[-1].to_dict()
    }


@app.post("/api/predict", response_model=PredictResponse, tags=["Prognostics & Inference"])
def predict_rul_endpoint(req: PredictRequest):
    adapter = DatasetRegistry.get_adapter(req.dataset)
    train_all = adapter.load_train()

    sensor_means = train_all[FEATURE_COLS].mean().values.astype(np.float32)
    sensor_stds = train_all[FEATURE_COLS].std().replace(0, 1.0).values.astype(np.float32)

    if req.sequence is not None:
        raw_w = np.array(req.sequence, dtype=np.float32)
        if raw_w.shape != (30, 17):
            raise HTTPException(status_code=400, detail=f"Sequence matrix must be shape (30, 17), got {raw_w.shape}")
        eid = req.engine_id or 0
        current_cycle = 100
    elif req.engine_id is not None:
        traj = adapter.get_engine_trajectory(req.engine_id, kind="train")
        eid = req.engine_id
        current_cycle = int(traj["cycle"].max())
        raw_w = traj[FEATURE_COLS].iloc[-30:].values.astype(np.float32)
        if len(raw_w) < 30:
            pad = np.repeat(raw_w[0:1], 30 - len(raw_w), axis=0)
            raw_w = np.vstack([pad, raw_w])
    else:
        raise HTTPException(status_code=400, detail="Must provide either 'engine_id' or 'sequence'.")

    norm_w = (raw_w - sensor_means) / sensor_stds

    # Model inference & uncertainty
    model = get_model(req.model_name)
    unc_tool = UncertaintyEstimator(model, num_samples=20)
    unc = unc_tool.predict_with_uncertainty(norm_w)
    pred_rul = max(0.0, float(unc["mean_rul"]))

    # Anomaly detection
    detector = EngineAnomalyDetector(contamination=0.05)
    detector.fit(train_all[train_all["cycle"] <= 25], FEATURE_COLS)
    anom = detector.detect_window_anomaly(norm_w)

    # Health evaluation
    health = decision_engine.evaluate_health(
        predicted_rul=pred_rul,
        uncertainty_std=float(unc["std_dev"]),
        anomaly_score=anom["anomaly_score"],
        current_cycle=current_cycle
    )

    # Log to SQLite
    db.log_prediction(
        engine_id=eid,
        dataset_subset=req.dataset,
        cycle=current_cycle,
        predicted_rul=pred_rul,
        uncertainty=unc,
        anomaly=anom,
        health_eval=health,
        model_name=req.model_name
    )

    return PredictResponse(
        engine_id=eid,
        dataset=req.dataset,
        predicted_rul=pred_rul,
        uncertainty=unc,
        anomaly=anom,
        health_state=health["health_state"],
        maintenance_priority_index=health["maintenance_priority_index"],
        priority=health["priority"],
        recommended_actions=health["recommended_actions"]
    )


@app.get("/api/maintenance/work-orders", tags=["Predictive Maintenance"])
def get_work_orders(dataset: Optional[str] = None, status: str = "OPEN"):
    orders = db.get_work_orders(dataset_subset=dataset, status=status)
    return {"count": len(orders), "work_orders": orders}


@app.post("/api/maintenance/work-orders", tags=["Predictive Maintenance"])
def create_work_order(order: WorkOrderCreate):
    conn = db._get_connection()
    try:
        cursor = conn.cursor()
        now = datetime.datetime.now().isoformat()
        cursor.execute("""
        INSERT INTO work_orders (
            created_at, engine_id, dataset_subset, priority,
            title, subsystem, description, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN')
        """, (
            now, order.engine_id, order.dataset_subset, order.priority,
            order.title, order.subsystem, order.description
        ))
        conn.commit()
        order_id = cursor.lastrowid
        return {"status": "CREATED", "work_order_id": order_id}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Server-Sent Events (SSE) Real-Time Telemetry Streaming
# ---------------------------------------------------------------------------
@app.get("/api/stream/telemetry/{dataset}/{engine_id}", tags=["Real-Time Streaming"])
async def stream_telemetry(dataset: str, engine_id: int, speed: float = Query(0.1, ge=0.01, le=2.0)):
    """Stream historical flight cycles one by one via Server-Sent Events (SSE)."""
    adapter = DatasetRegistry.get_adapter(dataset)
    traj = adapter.get_engine_trajectory(engine_id, kind="train")
    max_c = int(traj["cycle"].max())
    model = get_model()

    train_all = adapter.load_train()
    means = train_all[FEATURE_COLS].mean().values.astype(np.float32)
    stds = train_all[FEATURE_COLS].std().replace(0, 1.0).values.astype(np.float32)

    async def event_generator():
        for c in range(1, max_c + 1):
            sub = traj[traj["cycle"] <= c]
            curr_row = sub.iloc[-1].to_dict()
            raw_w = sub[FEATURE_COLS].iloc[-30:].values.astype(np.float32)
            if len(raw_w) < 30:
                pad = np.repeat(raw_w[0:1], 30 - len(raw_w), axis=0)
                raw_w = np.vstack([pad, raw_w])
            norm_w = (raw_w - means) / stds

            with torch.no_grad():
                tin = torch.tensor(norm_w[np.newaxis, ...], dtype=torch.float32)
                pred_rul = float(model(tin).item())

            payload = {
                "engine_id": engine_id,
                "dataset": dataset,
                "cycle": c,
                "max_cycle": max_c,
                "predicted_rul": round(pred_rul, 1),
                "true_remaining_cycles": max(0, max_c - c),
                "telemetry": {k: round(v, 3) if isinstance(v, float) else v for k, v in curr_row.items() if k in FEATURE_COLS or k.startswith("setting_")}
            }

            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(speed)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    print("[x] Starting JETWISE AI FastAPI Server on port 8000...")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
