from pathlib import Path

from fastapi.testclient import TestClient

from simpliscribe.retrieval import FastPrescriptionRetriever
from simpliscribe.main import app


def test_favicon_redirects_to_existing_static_asset():
    response = TestClient(app).get("/favicon.ico", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/static/favicon.svg"


def test_clean_install_can_run_without_a_generated_similarity_index(tmp_path, monkeypatch):
    retriever = FastPrescriptionRetriever(index_path=tmp_path / "absent.npz")
    monkeypatch.setattr("simpliscribe.main.get_retriever", lambda: retriever)
    client = TestClient(app)

    assert client.get("/api/live").status_code == 200
    assert client.get("/api/health").status_code == 200
    response = client.get("/api/retrieval/similar?q=Paracetamol")
    assert response.status_code == 200
    assert response.json()["results"] == []
    assert not retriever.is_ready()


def test_similarity_index_is_ignored_and_excluded_from_distributions():
    root = Path(__file__).resolve().parents[1]
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    manifest = (root / "MANIFEST.in").read_text(encoding="utf-8")

    assert "data/embeddings/*.npz" in gitignore
    assert "*.npz" in manifest
    assert "prune synthetic_prescription_dataset" in manifest
