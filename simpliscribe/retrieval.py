from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_DIM = 384
DEFAULT_INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "embeddings" / "prescriptions_index.npz"


def _tokenize_text(text: str) -> list[str]:
    cleaned = re.sub(r"[^\w\s\-\./]", " ", text.lower())
    raw_tokens = [tok.strip(".,;:!?\"'()") for tok in cleaned.split()]
    tokens = [tok for tok in raw_tokens if tok]
    features: list[str] = []
    for tok in tokens:
        features.append(f"w:{tok}")
        # Subword character n-grams (3-4 chars) for robust spelling and dosage matching
        if len(tok) >= 3:
            for n in (3, 4):
                for i in range(len(tok) - n + 1):
                    features.append(f"c:{tok[i:i+n]}")
    # Word bigrams
    for i in range(len(tokens) - 1):
        features.append(f"bg:{tokens[i]}_{tokens[i+1]}")
    return features


def _feature_hash_vector(features: list[str], dim: int = DEFAULT_EMBEDDING_DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    if not features:
        return vec

    # Feature frequency weighting with Murmur3-like SHA256 hashed projection
    for feat in features:
        digest = hashlib.sha256(feat.encode("utf-8")).digest()
        # Derive primary bucket index and secondary sign bit
        idx = int.from_bytes(digest[:4], byteorder="little") % dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        # Weight word-level tokens slightly higher than character n-grams
        weight = 1.5 if feat.startswith("w:") else (1.2 if feat.startswith("bg:") else 0.8)
        vec[idx] += sign * weight

    # L2 normalize vector for cosine similarity via dot product
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec /= norm
    return vec


class PrescriptionEmbedder:
    """Computes dense, normalized semantic vector representations for prescription text."""

    def __init__(self, dim: int = DEFAULT_EMBEDDING_DIM) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        features = _tokenize_text(text)
        return _feature_hash_vector(features, dim=self.dim)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vectors = np.stack([self.embed(t) for t in texts], axis=0)
        return vectors.astype(np.float32)


@dataclass
class SearchResult:
    item_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorIndex:
    """In-memory and file-backed dense vector index with cosine similarity search."""

    def __init__(self, dim: int = DEFAULT_EMBEDDING_DIM) -> None:
        self.dim = dim
        self.item_ids: list[str] = []
        self.vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self.metadatas: list[dict[str, Any]] = []
        self._id_to_idx: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.item_ids)

    def add(self, item_id: str, vector: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        if vector.shape != (self.dim,):
            raise ValueError(f"Expected vector of shape ({self.dim},), got {vector.shape}")

        norm = np.linalg.norm(vector)
        normed_vector = (vector / norm) if norm > 1e-12 else vector
        normed_vector = normed_vector.astype(np.float32)

        if item_id in self._id_to_idx:
            idx = self._id_to_idx[item_id]
            self.vectors[idx] = normed_vector
            self.metadatas[idx] = metadata or {}
        else:
            idx = len(self.item_ids)
            self.item_ids.append(item_id)
            self._id_to_idx[item_id] = idx
            if self.vectors.shape[0] == 0:
                self.vectors = np.expand_dims(normed_vector, axis=0)
            else:
                self.vectors = np.vstack([self.vectors, normed_vector])
            self.metadatas.append(metadata or {})

    def add_batch(
        self,
        item_ids: list[str],
        vectors: np.ndarray,
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        if len(item_ids) != vectors.shape[0]:
            raise ValueError(f"Mismatch between number of IDs ({len(item_ids)}) and vectors ({vectors.shape[0]})")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vector dim {self.dim}, got {vectors.shape[1]}")

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        normed_vectors = (vectors / norms).astype(np.float32)
        meta_list = metadatas or [{} for _ in item_ids]

        for i, (item_id, vec, meta) in enumerate(zip(item_ids, normed_vectors, meta_list)):
            if item_id in self._id_to_idx:
                idx = self._id_to_idx[item_id]
                self.vectors[idx] = vec
                self.metadatas[idx] = meta
            else:
                idx = len(self.item_ids)
                self.item_ids.append(item_id)
                self._id_to_idx[item_id] = idx
                if self.vectors.shape[0] == 0:
                    self.vectors = np.expand_dims(vec, axis=0)
                else:
                    self.vectors = np.vstack([self.vectors, vec])
                self.metadatas.append(meta)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[SearchResult]:
        if len(self.item_ids) == 0:
            return []

        if query_vector.shape != (self.dim,):
            raise ValueError(f"Expected query vector of shape ({self.dim},), got {query_vector.shape}")

        norm = np.linalg.norm(query_vector)
        q_norm = (query_vector / norm) if norm > 1e-12 else query_vector
        q_norm = q_norm.astype(np.float32)

        # Fast cosine similarity via dot product against normalized vectors
        scores = np.dot(self.vectors, q_norm)

        # Top-k indices
        k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        sorted_top_indices = top_indices[np.argsort(-scores[top_indices])]

        results: list[SearchResult] = []
        for idx in sorted_top_indices:
            score = float(scores[idx])
            if score < min_similarity:
                continue
            results.append(
                SearchResult(
                    item_id=self.item_ids[idx],
                    score=round(score, 4),
                    metadata=self.metadatas[idx],
                )
            )
        return results

    def save(self, path: Path | str) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_json = json.dumps(self.metadatas, ensure_ascii=False)
        np.savez_compressed(
            file_path,
            dim=np.array([self.dim], dtype=np.int32),
            item_ids=np.array(self.item_ids, dtype=object),
            vectors=self.vectors,
            metadata_json=np.array([metadata_json], dtype=object),
        )
        logger.info("Saved vector index (%d items) to %s", len(self), file_path)

    @classmethod
    def load(cls, path: Path | str) -> VectorIndex:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Vector index file not found: {file_path}")

        with np.load(file_path, allow_pickle=True) as data:
            dim = int(data["dim"][0])
            index = cls(dim=dim)
            index.item_ids = list(data["item_ids"])
            index.vectors = data["vectors"].astype(np.float32)
            meta_json = str(data["metadata_json"][0])
            index.metadatas = json.loads(meta_json)
            index._id_to_idx = {item_id: i for i, item_id in enumerate(index.item_ids)}
        logger.info("Loaded vector index (%d items) from %s", len(index), file_path)
        return index


class FastPrescriptionRetriever:
    """High-level manager for prescription embeddings index, cached lookups, and fast similarity retrieval."""

    def __init__(self, index_path: Path | str = DEFAULT_INDEX_PATH, embedder: PrescriptionEmbedder | None = None) -> None:
        self.index_path = Path(index_path)
        self.embedder = embedder or PrescriptionEmbedder()
        self.index = VectorIndex(dim=self.embedder.dim)
        if self.index_path.exists():
            try:
                self.index = VectorIndex.load(self.index_path)
            except Exception as exc:
                logger.warning("Could not load vector index from %s: %s. Initialized empty index.", self.index_path, exc)

    def is_ready(self) -> bool:
        return len(self.index) > 0

    def index_prescription(
        self,
        prescription_id: str,
        raw_text: str,
        medicines: list[dict[str, Any]] | None = None,
        source: str = "custom",
        tags: list[str] | None = None,
    ) -> None:
        vec = self.embedder.embed(raw_text)
        metadata = {
            "raw_text": raw_text,
            "medicines": medicines or [],
            "source": source,
            "tags": tags or [],
        }
        self.index.add(prescription_id, vec, metadata)

    def query_similar(
        self,
        query_text: str,
        top_k: int = 5,
        min_similarity: float = 0.3,
    ) -> list[dict[str, Any]]:
        query_vec = self.embedder.embed(query_text)
        results = self.index.search(query_vec, top_k=top_k, min_similarity=min_similarity)
        return [
            {
                "id": res.item_id,
                "similarity": res.score,
                "raw_text": res.metadata.get("raw_text", ""),
                "medicines": res.metadata.get("medicines", []),
                "source": res.metadata.get("source", ""),
                "tags": res.metadata.get("tags", []),
            }
            for res in results
        ]

    def save(self) -> None:
        self.index.save(self.index_path)


# Module-level singletons for fast shared access across application requests
_global_retriever: FastPrescriptionRetriever | None = None
_global_vector_cache: VectorCache | None = None


def get_retriever() -> FastPrescriptionRetriever:
    global _global_retriever
    if _global_retriever is None:
        _global_retriever = FastPrescriptionRetriever()
    return _global_retriever


class VectorCache:
    """Semantic vector cache combining an in-memory high-speed vector index with database persistence."""

    def __init__(self, embedder: PrescriptionEmbedder | None = None, max_memory_entries: int = 1000) -> None:
        self.embedder = embedder or PrescriptionEmbedder()
        self.index = VectorIndex(dim=self.embedder.dim)
        self.max_memory_entries = max_memory_entries
        self._exact_hash_cache: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self.warm_from_db()

    def warm_from_db(self) -> None:
        try:
            from .storage import load_vector_cache_entries
            entries = load_vector_cache_entries(limit=self.max_memory_entries)
            for entry in entries:
                vec = np.array(entry["vector"], dtype=np.float32)
                self.index.add(entry["id"], vec, {"payload": entry["payload"], "raw_text": entry["raw_text"]})
                self._exact_hash_cache[entry["text_hash"]] = entry["payload"]
            logger.info("Warmed VectorCache with %d entries from database.", len(entries))
        except Exception as exc:
            logger.warning("Could not warm VectorCache from DB: %s", exc)

    def _hash_text(self, text: str) -> str:
        cleaned = " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())
        return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()

    def lookup(self, raw_text: str, threshold: float = 0.98) -> tuple[dict[str, Any], float] | None:
        if not raw_text or not raw_text.strip():
            return None

        text_hash = self._hash_text(raw_text)
        # 1. Exact hash check (0 ms O(1))
        if text_hash in self._exact_hash_cache:
            self.hits += 1
            return deepcopy(self._exact_hash_cache[text_hash]), 1.0

        # 2. Semantic vector cosine similarity check (<1 ms)
        if len(self.index) > 0:
            query_vec = self.embedder.embed(raw_text)
            matches = self.index.search(query_vec, top_k=1, min_similarity=threshold)
            if matches:
                top = matches[0]
                self.hits += 1
                try:
                    from .storage import increment_vector_cache_hit
                    increment_vector_cache_hit(top.item_id)
                except Exception:
                    pass
                return deepcopy(top.metadata.get("payload", {})), top.score

        self.misses += 1
        return None

    def store(self, raw_text: str, payload: dict[str, Any], vector: np.ndarray | None = None) -> None:
        if not raw_text or not raw_text.strip():
            return
        text_hash = self._hash_text(raw_text)
        entry_id = str(uuid.uuid4())
        vec = vector if vector is not None else self.embedder.embed(raw_text)

        # Store in memory
        self._exact_hash_cache[text_hash] = deepcopy(payload)
        self.index.add(entry_id, vec, {"payload": payload, "raw_text": raw_text})

        # Persist to database
        try:
            from .storage import save_vector_cache_entry
            save_vector_cache_entry(
                entry_id=entry_id,
                text_hash=text_hash,
                raw_text=raw_text,
                vector_json=json.dumps(vec.tolist()),
                payload_json=json.dumps(payload),
            )
        except Exception as exc:
            logger.warning("Could not persist vector cache entry: %s", exc)

    def stats(self) -> dict[str, Any]:
        from .storage import get_vector_cache_stats
        db_stats = {}
        try:
            db_stats = get_vector_cache_stats()
        except Exception:
            pass
        return {
            "in_memory_entries": len(self.index),
            "exact_cache_size": len(self._exact_hash_cache),
            "session_hits": self.hits,
            "session_misses": self.misses,
            "hit_ratio": round((self.hits / (self.hits + self.misses)), 4) if (self.hits + self.misses) > 0 else 0.0,
            **db_stats,
        }

    def clear(self) -> None:
        self.index = VectorIndex(dim=self.embedder.dim)
        self._exact_hash_cache.clear()
        self.hits = 0
        self.misses = 0
        try:
            from .storage import clear_vector_cache_db
            clear_vector_cache_db()
        except Exception as exc:
            logger.warning("Could not clear DB vector cache: %s", exc)


def get_vector_cache() -> VectorCache:
    global _global_vector_cache
    if _global_vector_cache is None:
        _global_vector_cache = VectorCache()
    return _global_vector_cache

