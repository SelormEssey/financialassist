"""Tests for the public health endpoint."""

from fastapi.testclient import TestClient

from app.main import app


def test_health_returns_healthy_json() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {"status": "healthy"}
