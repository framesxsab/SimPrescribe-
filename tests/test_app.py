from fastapi.testclient import TestClient

from simpliscribe.main import app


client = TestClient(app)


def test_dashboard_route():
    response = client.get("/")
    assert response.status_code == 200


def test_history_api_route():
    response = client.get("/api/history")
    assert response.status_code == 200
    assert "analyses" in response.json()