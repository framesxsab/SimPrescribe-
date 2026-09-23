from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from threading import local
from pathlib import Path
from typing import Any, Mapping

from .config import settings

INDEX_SCHEMA_VERSION = "1"
INDEX_BEHAVIOR_VERSION = "1"
INDEX_FILENAME = "medicine_lexicon.sqlite"
SOURCE_DATASETS = (settings.india_medicine_dataset, settings.medicine_database_dataset)


def default_index_path() -> Path:
    return settings.data_dir / INDEX_FILENAME


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    digest.update(f"schema={INDEX_SCHEMA_VERSION}\nbehavior={INDEX_BEHAVIOR_VERSION}\n".encode())
    for path in SOURCE_DATASETS:
        digest.update(path.name.encode())
        if not path.is_file():
            digest.update(b"missing\n")
            continue
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _entry_row(entry: Any) -> tuple[Any, ...]:
    return (
        entry.name,
        entry.composition,
        entry.category,
        entry.dosage_form,
        entry.manufacturer,
        entry.pack_size,
        entry.therapeutic_class,
        entry.chemical_class,
        entry.action_class,
        json.dumps(list(entry.sources), separators=(",", ":")),
    )


def write_index(path: Path, lexicon: Mapping[str, Any], fingerprint: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = f".{path.name}.{os.getpid()}.tmp"
    temp_path = path.with_name(temp_name)
    temp_path.unlink(missing_ok=True)
    entries: dict[Any, int] = {}
    entry_rows: list[tuple[Any, ...]] = []
    alias_rows: list[tuple[Any, ...]] = []

    for ordinal, (alias, entry) in enumerate(lexicon.items()):
        entry_id = entries.get(entry)
        if entry_id is None:
            entry_id = len(entries) + 1
            entries[entry] = entry_id
            entry_rows.append((entry_id, *_entry_row(entry)))
        alias_rows.append((ordinal, alias, alias[:3], entry_id))

    connection = sqlite3.connect(temp_path)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=FULL;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE medicines (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                composition TEXT NOT NULL,
                category TEXT NOT NULL,
                dosage_form TEXT NOT NULL,
                manufacturer TEXT NOT NULL,
                pack_size TEXT NOT NULL,
                therapeutic_class TEXT NOT NULL,
                chemical_class TEXT NOT NULL,
                action_class TEXT NOT NULL,
                sources_json TEXT NOT NULL
            );
            CREATE TABLE aliases (
                ordinal INTEGER PRIMARY KEY,
                alias TEXT NOT NULL UNIQUE,
                prefix TEXT NOT NULL,
                medicine_id INTEGER NOT NULL REFERENCES medicines(id)
            );
            CREATE INDEX aliases_prefix_ordinal ON aliases(prefix, ordinal);
            """
        )
        connection.executemany(
            "INSERT INTO medicines VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            entry_rows,
        )
        connection.executemany(
            "INSERT INTO aliases VALUES (?, ?, ?, ?)",
            alias_rows,
        )
        metadata = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "behavior_version": INDEX_BEHAVIOR_VERSION,
            "source_fingerprint": fingerprint,
            "alias_count": str(len(alias_rows)),
            "entry_count": str(len(entry_rows)),
        }
        connection.executemany("INSERT INTO metadata VALUES (?, ?)", metadata.items())
        connection.commit()
    finally:
        connection.close()
    os.replace(temp_path, path)
    return {
        "path": str(path),
        "aliases": len(alias_rows),
        "entries": len(entry_rows),
        "bytes": path.stat().st_size,
    }


def build_index(path: Path | None = None) -> dict[str, Any]:
    from .inference import _load_medicine_lexicon

    output = path or default_index_path()
    before = source_fingerprint()
    lexicon = _load_medicine_lexicon()
    after = source_fingerprint()
    if before != after:
        raise RuntimeError("Medicine datasets changed while building the lexicon index.")
    result = write_index(output, lexicon, before)
    result["generation_ms"] = None
    result["fingerprint"] = before
    return result


def is_current(path: Path | None = None) -> bool:
    index_path = path or default_index_path()
    if not index_path.is_file():
        return False
    try:
        with sqlite3.connect(index_path) as connection:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        return (
            metadata.get("schema_version") == INDEX_SCHEMA_VERSION
            and metadata.get("behavior_version") == INDEX_BEHAVIOR_VERSION
            and metadata.get("source_fingerprint") == source_fingerprint()
        )
    except (OSError, sqlite3.Error):
        return False


class RequiredLexiconIndex:
    def __init__(self, path: Path):
        self.path = path
        self._connections = local()

    def _connect(self) -> sqlite3.Connection:
        connection = getattr(self._connections, "connection", None)
        if connection is None:
            uri = f"file:{self.path.resolve().as_posix()}?mode=ro"
            connection = sqlite3.connect(uri, uri=True)
            self._connections.connection = connection
        return connection

    def _row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "name": row[0],
            "composition": row[1],
            "category": row[2],
            "dosage_form": row[3],
            "manufacturer": row[4],
            "pack_size": row[5],
            "therapeutic_class": row[6],
            "chemical_class": row[7],
            "action_class": row[8],
            "sources": tuple(json.loads(row[9])),
        }

    def exact(self, alias: str) -> dict[str, Any] | None:
        row = self._connect().execute(
            """
            SELECT m.name, m.composition, m.category, m.dosage_form, m.manufacturer,
                   m.pack_size, m.therapeutic_class, m.chemical_class, m.action_class,
                   m.sources_json
            FROM aliases a JOIN medicines m ON m.id = a.medicine_id
            WHERE a.alias = ?
            """,
            (alias,),
        ).fetchone()
        return self._row(row) if row else None

    def prefix(self, prefix: str) -> list[tuple[str, dict[str, Any]]]:
        rows = self._connect().execute(
            """
            SELECT a.alias, m.name, m.composition, m.category, m.dosage_form,
                   m.manufacturer, m.pack_size, m.therapeutic_class,
                   m.chemical_class, m.action_class, m.sources_json
            FROM aliases a JOIN medicines m ON m.id = a.medicine_id
            WHERE a.prefix = ? ORDER BY a.ordinal
            """,
            (prefix,),
        )
        result = []
        for row in rows:
            result.append((row[0], self._row(row[1:])))
        return result


def open_current_index(path: Path | None = None) -> RequiredLexiconIndex | None:
    index_path = path or default_index_path()
    return RequiredLexiconIndex(index_path) if is_current(index_path) else None
