"""Tests for the FastAPI server endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Create a test client with the RAG warm-up and pipeline mocked out."""

    mock_pipeline_result = {
        "root_cause": "routing loop congestion",
        "confidence": 0.84,
        "recommended_fix": "restart OSPF process on R17",
        "anomaly_summary": "Packet drop and OSPF flapping detected on R17.",
        "evidence": ["OSPF adjacency flapping", "CPU at 92%"],
        "remediation_steps": ["ssh to R17", "clear ip ospf process"],
        "risk_level": "low",
        "estimated_resolution_time": "5 minutes",
    }

    # Patch the lifespan warm-up and the pipeline
    with patch("rag.retrieval._get_manager"), \
         patch("agents.rca_workflow.run_rca_pipeline", return_value=mock_pipeline_result):
        from api.server import app
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------

def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"rca_request_total" in response.content or b"rca_" in response.content


# ---------------------------------------------------------------------------
# /analyze_network
# ---------------------------------------------------------------------------

def test_analyze_network_success(client):
    payload = {"logs": "router R17 experiencing packet loss"}
    response = client.post("/analyze_network", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "root_cause" in data
    assert "confidence" in data
    assert "recommended_fix" in data
    assert 0.0 <= data["confidence"] <= 1.0


def test_analyze_network_returns_correct_fields(client):
    payload = {"logs": "network latency spike on SW04"}
    response = client.post("/analyze_network", json=payload)
    assert response.status_code == 200
    data = response.json()
    expected_keys = {
        "root_cause", "confidence", "recommended_fix",
        "anomaly_summary", "evidence", "remediation_steps",
        "risk_level", "estimated_resolution_time",
    }
    assert expected_keys.issubset(data.keys())


def test_analyze_network_empty_logs(client):
    """Empty / too-short logs should return 422."""
    response = client.post("/analyze_network", json={"logs": "ab"})
    assert response.status_code == 422


def test_analyze_network_missing_logs_field(client):
    """Missing 'logs' field should return 422."""
    response = client.post("/analyze_network", json={})
    assert response.status_code == 422


def test_analyze_network_pipeline_error(client):
    """Pipeline exceptions should be wrapped in a 500 response."""
    with patch("api.server.run_rca_pipeline", side_effect=RuntimeError("LLM error")):
        response = client.post("/analyze_network", json={"logs": "some log text here"})
    assert response.status_code == 500
    assert "RCA pipeline error" in response.json()["detail"]
