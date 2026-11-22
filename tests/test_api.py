"""Integration and Unit Tests for JETWISE AI FastAPI Server (Phase 9).

Verifies:
1. Root & Health Endpoints
2. Dataset Registry & Fleet Discovery Endpoints
3. Engine Trajectory Details Endpoint
4. Real-Time Deep Inference & Uncertainty Prediction Endpoint
5. Work Order SQLite CRUD Dispatch Endpoints
6. Server-Sent Events (SSE) Streaming Endpoint
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_root_endpoint():
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["platform"] == "JETWISE AI"
    assert "version" in data


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert "available_datasets" in data
    assert len(data["available_datasets"]) >= 4


def test_list_datasets_endpoint():
    res = client.get("/api/datasets")
    assert res.status_code == 200
    data = res.json()
    assert "datasets" in data
    names = [d["name"] for d in data["datasets"]]
    assert "FD001" in names
    assert "FD002" in names
    assert "FD003" in names
    assert "FD004" in names


def test_get_fleet_endpoint():
    res = client.get("/api/fleet/FD001")
    assert res.status_code == 200
    data = res.json()
    assert data["dataset"] == "FD001"
    assert data["total_engines"] == 100
    assert 1 in data["engine_ids"]


def test_get_engine_details_endpoint():
    res = client.get("/api/engine/FD001/1")
    assert res.status_code == 200
    data = res.json()
    assert data["engine_id"] == 1
    assert data["dataset"] == "FD001"
    assert data["total_cycles"] > 0
    assert "first_cycle_telemetry" in data
    assert "latest_cycle_telemetry" in data


def test_predict_with_engine_id():
    payload = {
        "dataset": "FD001",
        "engine_id": 1,
        "model_name": "pinball_lstm_q35"
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["engine_id"] == 1
    assert data["dataset"] == "FD001"
    assert "predicted_rul" in data
    assert data["predicted_rul"] >= 0.0
    assert "uncertainty" in data
    assert "anomaly" in data
    assert "health_state" in data
    assert "maintenance_priority_index" in data
    assert "recommended_actions" in data


def test_predict_with_custom_sequence():
    seq = np.random.randn(30, 17).tolist()
    payload = {
        "dataset": "FD001",
        "sequence": seq,
        "model_name": "pinball_lstm_q35"
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_rul" in data
    assert "health_state" in data


def test_work_orders_crud():
    # Create work order
    order_data = {
        "engine_id": 999,
        "dataset_subset": "FD001",
        "priority": "HIGH",
        "title": "API Test Work Order",
        "subsystem": "HPT Blades",
        "description": "Routine test dispatch from automated suite."
    }
    res_post = client.post("/api/maintenance/work-orders", json=order_data)
    assert res_post.status_code == 200
    res_data = res_post.json()
    assert res_data["status"] == "CREATED"
    order_id = res_data["work_order_id"]
    assert order_id > 0

    # Retrieve work orders
    res_get = client.get("/api/maintenance/work-orders?dataset=FD001")
    assert res_get.status_code == 200
    orders = res_get.json()["work_orders"]
    matching = [o for o in orders if o["id"] == order_id]
    assert len(matching) == 1
    assert matching[0]["engine_id"] == 999


def test_streaming_telemetry_sse():
    with client.stream("GET", "/api/stream/telemetry/FD001/1?speed=0.01") as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        chunks = []
        for line in response.iter_lines():
            if line and line.startswith("data: "):
                chunks.append(line)
                if len(chunks) >= 3:
                    break
        assert len(chunks) >= 3
