from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path
from threading import Lock
from typing import Any, Mapping
from urllib.parse import quote

from .config import settings

INDEX_SCHEMA_VERSION = "1"
INDEX_BEHAVIOR_VERSION = "1"
INDEX_FILENAME = "medicine_optional_references.sqlite"
SOURCE_DATASET = settings.medicine_database_dataset
OPTIONAL_COLUMNS = (
    ("substitutes", "substitute", 5),
    ("uses", "use", 5),
    ("side_effects", "sideEffect", 42),
)
_SOURCE_LABEL = "Medicine Database"

_cache_lock = Lock()
_cache_signature: tuple[tuple[int, int, int] | None, tuple[int, int, int] | None] | None = None
_cache_index: OptionalReferenceIndex | None = None


def default_index_path() -> Path:
    return settings.data_dir / INDEX_FILENAME


def source_fingerprint(source_path: Path | None = None) -> str:
    source = source_path or SOURCE_DATASET
    digest = hashlib.sha256()
    digest.update(f"schema={INDEX_SCHEMA_VERSION}\nbehavior={INDEX_BEHAVIOR_VERSION}\n".encode())
    digest.update(source.name.encode("utf-8"))
    if not source.is_file():
        digest.update(b"\nmissing\n")
        return digest.hexdigest()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()



def _collect_optional_fields(source: Path) -> tuple[dict[str, dict[str, Any]], int]:
    if not source.is_file():
        raise FileNotFoundError("Medicine reference source dataset is unavailable.")
    from .inference import collect_series, clean_value, normalize_text

    fields: dict[str, dict[str, Any]] = {}
    rows = 0
    with source.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            name = clean_value(row.get("name"))
            if not name:
                continue
            key = normalize_text(name)
            current = fields.setdefault(key, {
                "name": name,
                "substitutes": (),
                "uses": (),
                "side_effects": (),
                "provenance": (_SOURCE_LABEL,),
            })
            for output, prefix, limit in OPTIONAL_COLUMNS:
                values = collect_series(row, prefix, limit)
                current[output] = tuple(dict.fromkeys((*current[output], *values)))
    return fields, rows


def write_index(path: Path, fields: Mapping[str, Mapping[str, Any]], fingerprint: str, source_rows: int) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=FULL;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE medicine_references (
                normalized_name TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                substitutes_json TEXT NOT NULL,
                uses_json TEXT NOT NULL,
                side_effects_json TEXT NOT NULL,
                provenance_json TEXT NOT NULL
            ) WITHOUT ROWID;
            """
        )
        connection.executemany(
            "INSERT INTO medicine_references VALUES (?, ?, ?, ?, ?, ?)",
            (
                (
                    name,
                    value["name"],
                    json.dumps(list(value["substitutes"]), ensure_ascii=False, separators=(",", ":")),
                    json.dumps(list(value["uses"]), ensure_ascii=False, separators=(",", ":")),
                    json.dumps(list(value["side_effects"]), ensure_ascii=False, separators=(",", ":")),
                    json.dumps(list(value["provenance"]), ensure_ascii=False, separators=(",", ":")),
                )
                for name, value in fields.items()
            ),
        )
        reference_count = sum(
            len(value[field])
            for value in fields.values()
            for field in ("substitutes", "uses", "side_effects")
        )
        metadata = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "behavior_version": INDEX_BEHAVIOR_VERSION,
            "source_fingerprint": fingerprint,
            "source_rows": str(source_rows),
            "medicine_count": str(len(fields)),
            "reference_count": str(reference_count),
        }
        connection.executemany("INSERT INTO metadata VALUES (?, ?)", metadata.items())
        connection.commit()
        check = connection.execute("PRAGMA quick_check").fetchone()
        if not check or check[0] != "ok":
            raise sqlite3.DatabaseError("Generated optional reference index failed integrity check.")
    finally:
        connection.close()
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(path),
        "medicines": len(fields),
        "source_rows": source_rows,
        "reference_values": reference_count,
        "bytes": path.stat().st_size,
        "fingerprint": fingerprint,
    }


def build_index(path: Path | None = None, source_path: Path | None = None) -> dict[str, Any]:
    output = path or default_index_path()
    source = source_path or SOURCE_DATASET
    before = source_fingerprint(source)
    fields, source_rows = _collect_optional_fields(source)
    after = source_fingerprint(source)
    if before != after:
        raise RuntimeError("Medicine reference dataset changed while building the optional index.")
    return write_index(output, fields, before, source_rows)


def _readonly_connection(path: Path) -> sqlite3.Connection:
    encoded_path = quote(path.resolve().as_posix(), safe="/:")
    return sqlite3.connect(f"file:{encoded_path}?mode=ro", uri=True)


def is_current(path: Path | None = None, source_path: Path | None = None) -> bool:
    index_path = path or default_index_path()
    source = source_path or SOURCE_DATASET
    if not index_path.is_file() or not source.is_file():
        return False
    try:
        source_signature = _stat_signature(source)
        with closing(_readonly_connection(index_path)) as connection:
            check = connection.execute("PRAGMA quick_check").fetchone()
            if not check or check[0] != "ok":
                return False
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(medicine_references)")
            }
        return (
            source_signature is not None
            and source_signature == _stat_signature(source)
            and columns == {
                "normalized_name",
                "display_name",
                "substitutes_json",
                "uses_json",
                "side_effects_json",
                "provenance_json",
            }
            and metadata.get("schema_version") == INDEX_SCHEMA_VERSION
            and metadata.get("behavior_version") == INDEX_BEHAVIOR_VERSION
            and metadata.get("source_fingerprint") == source_fingerprint(source)
        )
    except (OSError, sqlite3.Error, ValueError):
        return False


class OptionalReferenceIndex:
    def __init__(self, path: Path):
        self.path = path

    def lookup(self, normalized_name: str) -> dict[str, Any] | None:
        with closing(_readonly_connection(self.path)) as connection:
            row = connection.execute(
                """
                SELECT substitutes_json, uses_json, side_effects_json, provenance_json
                FROM medicine_references
                WHERE normalized_name = ?
                """,
                (normalized_name,),
            ).fetchone()
        if row is None:
            return None
        decoded = [json.loads(value) for value in row]
        if any(not isinstance(values, list) or any(not isinstance(value, str) for value in values) for values in decoded):
            raise ValueError("Optional reference index contains invalid fields.")
        return {
            "substitutes": tuple(decoded[0]),
            "uses": tuple(decoded[1]),
            "side_effects": tuple(decoded[2]),
            "provenance": tuple(decoded[3]),
        }


def open_current_index(path: Path | None = None, source_path: Path | None = None) -> OptionalReferenceIndex | None:
    index_path = path or default_index_path()
    return OptionalReferenceIndex(index_path) if is_current(index_path, source_path) else None


def _stat_signature(path: Path) -> tuple[int, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def load_current_index() -> OptionalReferenceIndex | None:
    global _cache_signature, _cache_index

    path = default_index_path()
    source = SOURCE_DATASET
    signature = (_stat_signature(path), _stat_signature(source))
    with _cache_lock:
        if signature == _cache_signature:
            return _cache_index
        _cache_index = open_current_index(path, source)
        _cache_signature = signature
        return _cache_index


def clear_index_cache() -> None:
    global _cache_signature, _cache_index
    with _cache_lock:
        _cache_signature = None
        _cache_index = None
