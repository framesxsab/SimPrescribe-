from __future__ import annotations

import argparse
import csv
import json
import logging
import time
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from simpliscribe.retrieval import FastPrescriptionRetriever, PrescriptionEmbedder, VectorIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_embeddings")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN_PATH = BASE_DIR / "data" / "golden_cases.v1.json"
DEFAULT_SYNTHETIC_LABELS = BASE_DIR / "synthetic_prescription_dataset" / "labels.csv"
DEFAULT_OUTPUT_INDEX = BASE_DIR / "data" / "embeddings" / "prescriptions_index.npz"


def load_golden_cases(golden_path: Path) -> list[dict[str, Any]]:
    if not golden_path.exists():
        logger.warning("Golden cases file not found: %s", golden_path)
        return []
    payload = json.loads(golden_path.read_text(encoding="utf-8"))
    cases = payload.get("cases", []) if isinstance(payload, dict) else payload
    records: list[dict[str, Any]] = []
    for c in cases:
        raw_text = str(c.get("raw_text") or "").strip()
        if not raw_text:
            continue
        records.append(
            {
                "id": f"golden:{c.get('id')}",
                "raw_text": raw_text,
                "medicines": c.get("expected_medications", []),
                "source": "golden_cases_v1",
                "tags": c.get("tags", []),
            }
        )
    logger.info("Loaded %d golden cases from %s", len(records), golden_path)
    return records


def load_synthetic_prescriptions(labels_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not labels_path.exists():
        logger.warning("Synthetic labels CSV not found: %s", labels_path)
        return []

    # Group medicines by image
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with labels_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = str(row.get("image") or "").strip()
            med = str(row.get("medicine") or "").strip()
            dos = str(row.get("dosage") or "").strip()
            freq = str(row.get("frequency") or "").strip()
            if img and med:
                grouped[img].append({"name": med, "dosage": dos, "frequency": freq})

    records: list[dict[str, Any]] = []
    for img_name, meds in grouped.items():
        if limit is not None and len(records) >= limit:
            break
        # Build composite prescription raw text representation
        lines = [f"{m['name']} {m['dosage']} {m['frequency']}".strip() for m in meds]
        raw_text = "\n".join(lines)
        records.append(
            {
                "id": f"synthetic:{img_name}",
                "raw_text": raw_text,
                "medicines": meds,
                "source": "synthetic_dataset",
                "tags": ["synthetic", "precomputed"],
            }
        )

    logger.info("Loaded %d synthetic prescriptions from %s", len(records), labels_path)
    return records


def benchmark_retrieval(retriever: FastPrescriptionRetriever, query_cases: list[dict[str, Any]], iterations: int = 1000) -> None:
    if not query_cases:
        logger.warning("No query cases available for benchmarking.")
        return

    logger.info("Running retrieval benchmark (%d iterations across %d cases)...", iterations, len(query_cases))
    queries = [c["raw_text"] for c in query_cases]
    latencies_ms: list[float] = []

    start_total = time.perf_counter()
    for i in range(iterations):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        results = retriever.query_similar(q, top_k=5, min_similarity=0.1)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
    total_time = time.perf_counter() - start_total

    latencies_ms.sort()
    p50 = latencies_ms[int(len(latencies_ms) * 0.50)]
    p95 = latencies_ms[int(len(latencies_ms) * 0.95)]
    p99 = latencies_ms[int(len(latencies_ms) * 0.99)]
    qps = iterations / total_time

    print("\n" + "=" * 50)
    print("FAST RETRIEVAL BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Total Index Size:    {len(retriever.index)} items")
    print(f"Iterations:          {iterations}")
    print(f"Total Time:          {total_time:.3f} s")
    print(f"Throughput (QPS):    {qps:.1f} queries/sec")
    print(f"Latency P50 (median):{p50:.3f} ms")
    print(f"Latency P95:         {p95:.3f} ms")
    print(f"Latency P99:         {p99:.3f} ms")
    print("=" * 50 + "\n")


def build_and_save_index(
    golden_path: Path = DEFAULT_GOLDEN_PATH,
    labels_path: Path = DEFAULT_SYNTHETIC_LABELS,
    output_path: Path = DEFAULT_OUTPUT_INDEX,
    synthetic_limit: int | None = None,
    run_benchmark: bool = False,
) -> None:
    records = load_golden_cases(golden_path)
    records.extend(load_synthetic_prescriptions(labels_path, limit=synthetic_limit))

    logger.info("Building embeddings for %d total prescription cases...", len(records))
    t0 = time.perf_counter()
    embedder = PrescriptionEmbedder()
    index = VectorIndex(dim=embedder.dim)

    ids = [r["id"] for r in records]
    texts = [r["raw_text"] for r in records]
    metadatas = [{"raw_text": r["raw_text"], "medicines": r["medicines"], "source": r["source"], "tags": r["tags"]} for r in records]

    vectors = embedder.embed_batch(texts)
    index.add_batch(ids, vectors, metadatas)
    index.save(output_path)
    build_time = time.perf_counter() - t0
    logger.info("Successfully built & saved vector index (%d items) to %s in %.3f s", len(index), output_path, build_time)

    retriever = FastPrescriptionRetriever(index_path=output_path, embedder=embedder)

    # Test sample lookup
    sample_query = "Paracetamol 650 tab od 5 days"
    sample_matches = retriever.query_similar(sample_query, top_k=3)
    logger.info("Sample query: '%s'", sample_query)
    for m in sample_matches:
        logger.info("  -> Match: %s (sim: %.4f, meds: %s)", m["id"], m["similarity"], [med.get("name") for med in m["medicines"]])

    if run_benchmark:
        benchmark_retrieval(retriever, records, iterations=1000)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and benchmark prescription embeddings index.")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_PATH, help="Path to golden cases JSON.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_SYNTHETIC_LABELS, help="Path to synthetic labels CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_INDEX, help="Output path for embeddings index (.npz).")
    parser.add_argument("--sample", type=int, default=None, help="Limit number of synthetic prescriptions to index.")
    parser.add_argument("--benchmark", action="store_true", help="Run retrieval latency and throughput benchmark.")
    args = parser.parse_args()

    build_and_save_index(
        golden_path=args.golden,
        labels_path=args.labels,
        output_path=args.output,
        synthetic_limit=args.sample,
        run_benchmark=args.benchmark,
    )


if __name__ == "__main__":
    main()
