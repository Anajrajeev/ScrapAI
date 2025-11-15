"""Tests for API module."""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_health_endpoint():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "predictor_loaded" in data


def test_predict_endpoint_no_file():
    """Test predict endpoint without file."""
    response = client.post("/predict")
    assert response.status_code != 200  # Should fail without file

