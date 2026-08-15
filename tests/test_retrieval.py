from pathlib import Path

import numpy as np
from starlette.testclient import TestClient

from simpliscribe.main import app
from simpliscribe.retrieval import FastPrescriptionRetriever, PrescriptionEmbedder, VectorIndex


def test_embedder_single_and_batch():
    embedder = PrescriptionEmbedder(dim=128)
    vec = embedder.embed("Paracetamol 500mg tablet twice daily")
    assert vec.shape == (128,)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)

    batch_vecs = embedder.embed_batch([
        "Paracetamol 500mg tablet",
        "Amoxicillin 250mg capsule",
    ])
    assert batch_vecs.shape == (2, 128)
    assert np.isclose(np.linalg.norm(batch_vecs[0]), 1.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(batch_vecs[1]), 1.0, atol=1e-5)


def test_embedder_semantic_similarity():
    embedder = PrescriptionEmbedder(dim=256)
    v1 = embedder.embed("Paracetamol 650mg tablet once daily")
    v2 = embedder.embed("Paracetamol 650 tab od")
    v3 = embedder.embed("Azithromycin 500mg suspension for 3 days")

    sim_1_2 = float(np.dot(v1, v2))
    sim_1_3 = float(np.dot(v1, v3))

    assert sim_1_2 > 0.4
    assert sim_1_2 > sim_1_3


def test_vector_index_add_and_search():
    index = VectorIndex(dim=64)
    v1 = np.ones(64, dtype=np.float32)
    v2 = np.zeros(64, dtype=np.float32)
    v2[0] = 1.0

    index.add("item1", v1, {"name": "Item 1"})
    index.add("item2", v2, {"name": "Item 2"})

    assert len(index) == 2

    # Query with vector close to item2
    results = index.search(v2, top_k=2)
    assert len(results) == 2
    assert results[0].item_id == "item2"
    assert np.isclose(results[0].score, 1.0, atol=1e-4)
    assert results[0].metadata["name"] == "Item 2"


def test_vector_index_save_and_load(tmp_path: Path):
    index = VectorIndex(dim=32)
    embedder = PrescriptionEmbedder(dim=32)

    t1 = "Metformin 500mg 1-0-1"
    t2 = "Cetirizine 10mg 0-0-1"

    index.add("case1", embedder.embed(t1), {"text": t1, "meds": ["Metformin"]})
    index.add("case2", embedder.embed(t2), {"text": t2, "meds": ["Cetirizine"]})

    save_path = tmp_path / "test_index.npz"
    index.save(save_path)
    assert save_path.exists()

    loaded = VectorIndex.load(save_path)
    assert len(loaded) == 2
    assert loaded.item_ids == ["case1", "case2"]

    results = loaded.search(embedder.embed(t1), top_k=1)
    assert len(results) == 1
    assert results[0].item_id == "case1"
    assert results[0].metadata["meds"] == ["Metformin"]


def test_fast_prescription_retriever(tmp_path: Path):
    index_file = tmp_path / "prescriptions.npz"
    retriever = FastPrescriptionRetriever(index_path=index_file)

    retriever.index_prescription(
        prescription_id="case_para",
        raw_text="Paracetamol 650mg 1 tablet daily for 5 days",
        medicines=[{"name": "Paracetamol", "dosage": "650mg"}],
        source="golden",
    )
    retriever.index_prescription(
        prescription_id="case_amox",
        raw_text="Amoxicillin 500mg capsule twice daily",
        medicines=[{"name": "Amoxicillin", "dosage": "500mg"}],
        source="golden",
    )

    matches = retriever.query_similar("Paracetamol 650 tablet", top_k=2, min_similarity=0.2)
    assert len(matches) >= 1
    assert matches[0]["id"] == "case_para"
    assert matches[0]["medicines"][0]["name"] == "Paracetamol"

    retriever.save()
    assert index_file.exists()


def test_similar_prescriptions_api():
    client = TestClient(app)
    response = client.get("/api/retrieval/similar?q=Paracetamol&limit=3")
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "results" in data
    assert isinstance(data["results"], list)


def test_vector_cache_exact_and_semantic():
    from simpliscribe.retrieval import VectorCache

    cache = VectorCache(max_memory_entries=100)
    cache.clear()

    text = "Metformin 500mg tab 1-0-1 for 30 days"
    sample_payload = {"medications": [{"name": "Metformin", "dosage": "500 mg"}], "patient_name": "John Doe"}

    # Initially miss
    assert cache.lookup(text) is None

    # Store
    cache.store(text, sample_payload)

    # Exact hit (similarity 1.0)
    hit, sim = cache.lookup(text)
    assert hit is not None
    assert sim == 1.0
    assert hit["patient_name"] == "John Doe"

    # Near semantic match (minor whitespace/punctuation variance)
    near_text = "Metformin 500mg tab 1-0-1 for 30 days."
    hit_near, sim_near = cache.lookup(near_text, threshold=0.95)
    assert hit_near is not None
    assert sim_near >= 0.95
    assert hit_near["medications"][0]["name"] == "Metformin"


def test_cache_stats_and_clear_api():
    client = TestClient(app)
    stats_res = client.get("/api/cache/stats")
    assert stats_res.status_code == 200
    stats = stats_res.json()
    assert "in_memory_entries" in stats
    assert "session_hits" in stats

    clear_res = client.post("/api/cache/clear")
    assert clear_res.status_code == 200
    assert clear_res.json()["status"] == "cleared"

