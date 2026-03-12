from fastapi.testclient import TestClient

from simpliscribe.local_model_server import app


client = TestClient(app)


def test_local_model_server_health_route():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_local_model_server_extract_route(monkeypatch):
    def fake_generate_output(raw_text: str, prompt_override: str | None = None) -> str:
        assert raw_text == "Paracetamol 650 tab od 5 days"
        assert prompt_override
        return '{"medications": [{"name": "Paracetamol", "category": "Analgesic", "type": "Tablet", "dosage": "650 mg", "frequency": "once daily", "duration": "5 days", "insight": "Use as directed."}]}'

    monkeypatch.setattr("simpliscribe.local_model_server.generate_output", fake_generate_output)

    response = client.post(
        "/extract",
        json={
            "input": "Paracetamol 650 tab od 5 days",
            "prompt": "Return only JSON.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"]
    assert payload["medications"][0]["name"] == "Paracetamol"