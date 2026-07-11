import json
import os
from threading import Lock
from typing import Any

from .config import settings


_history_lock = Lock()

for directory in (settings.static_dir, settings.uploads_dir, settings.data_dir):
    directory.mkdir(exist_ok=True)


def load_history() -> list[dict[str, Any]]:
    if not settings.history_file.exists():
        return []

    try:
        return json.loads(settings.history_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_history(history: list[dict[str, Any]]) -> None:
    temporary = settings.history_file.with_suffix(".tmp")
    temporary.write_text(json.dumps(history, indent=2), encoding="utf-8")
    os.replace(temporary, settings.history_file)


def append_history(record: dict[str, Any], limit: int = 25) -> None:
    # ponytail: one process lock is enough for the local JSON store; use a real
    # authenticated database before running multiple workers.
    with _history_lock:
        history = load_history()
        history.insert(0, record)
        save_history(history[:limit])


def get_analysis_record(analysis_id: str) -> dict[str, Any] | None:
    for record in load_history():
        if record.get("id") == analysis_id:
            return record
    return None
