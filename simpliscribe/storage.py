import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import Column, DateTime, MetaData, String, Table, Text, create_engine, delete, insert, select, text, update

from .config import settings

logger = logging.getLogger(__name__)


metadata = MetaData()
analyses = Table(
    "analyses",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("owner_id", String(255), nullable=False, index=True),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
    Column("expires_at", DateTime(timezone=True), nullable=False, index=True),
    Column("payload", Text, nullable=False),
)
audit_events = Table(
    "audit_events",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("owner_id", String(255), nullable=False, index=True),
    Column("analysis_id", String(36), nullable=True, index=True),
    Column("event_type", String(64), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("metadata_json", Text, nullable=False, default="{}"),
)

engine = create_engine(settings.database_url, pool_pre_ping=True)


def ensure_schema() -> None:
    metadata.create_all(engine)


def ping_database() -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.exception("Database liveness check failed.")
        return False


def _now() -> datetime:
    return datetime.now(timezone.utc)


def purge_expired() -> int:
    with engine.begin() as connection:
        result = connection.execute(delete(analyses).where(analyses.c.expires_at < _now()))
        return int(result.rowcount or 0)


def load_history(owner_id: str = "local") -> list[dict[str, Any]]:
    purge_expired()
    query = select(analyses.c.payload).where(analyses.c.owner_id == owner_id).order_by(analyses.c.created_at.desc())
    with engine.connect() as connection:
        return [json.loads(row.payload) for row in connection.execute(query)]


def save_history(history: list[dict[str, Any]], owner_id: str = "local") -> None:
    with engine.begin() as connection:
        connection.execute(delete(analyses).where(analyses.c.owner_id == owner_id))
        for record in history:
            _insert_record(connection, record, owner_id)


def _insert_record(connection, record: dict[str, Any], owner_id: str) -> None:
    created_at = datetime.fromisoformat(str(record.get("created_at") or _now().isoformat()))
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    connection.execute(insert(analyses).values(
        id=str(record["id"]), owner_id=owner_id, created_at=created_at,
        expires_at=created_at + timedelta(days=settings.retention_days),
        payload=json.dumps(record),
    ))


def append_history(record: dict[str, Any], limit: int = 25, owner_id: str = "local") -> None:
    with engine.begin() as connection:
        _insert_record(connection, record, owner_id)
        stale_ids = connection.execute(
            select(analyses.c.id).where(analyses.c.owner_id == owner_id)
            .order_by(analyses.c.created_at.desc()).offset(limit)
        ).scalars().all()
        if stale_ids:
            connection.execute(delete(analyses).where(analyses.c.id.in_(stale_ids)))


def try_append_history(record: dict[str, Any], limit: int = 25, owner_id: str = "local", attempts: int = 2) -> bool:
    for _ in range(max(attempts, 1)):
        try:
            append_history(record, limit=limit, owner_id=owner_id)
            return True
        except Exception:
            logger.exception("Failed to persist analysis history.")
    return False


def get_analysis_record(analysis_id: str, owner_id: str = "local") -> dict[str, Any] | None:
    query = select(analyses.c.payload).where(analyses.c.id == analysis_id, analyses.c.owner_id == owner_id)
    with engine.connect() as connection:
        payload = connection.execute(query).scalar_one_or_none()
    return json.loads(payload) if payload else None


def update_analysis_record(
    analysis_id: str,
    owner_id: str,
    record: dict[str, Any],
    expected_record: dict[str, Any] | None = None,
) -> bool:
    conditions = [analyses.c.id == analysis_id, analyses.c.owner_id == owner_id]
    if expected_record is not None:
        conditions.append(analyses.c.payload == json.dumps(expected_record))
    with engine.begin() as connection:
        result = connection.execute(
            update(analyses).where(*conditions)
            .values(payload=json.dumps(record))
        )
        return bool(result.rowcount)


def append_audit_event(event_id: str, owner_id: str, event_type: str, analysis_id: str | None = None, **safe_metadata: Any) -> None:
    with engine.begin() as connection:
        connection.execute(insert(audit_events).values(
            id=event_id, owner_id=owner_id, analysis_id=analysis_id, event_type=event_type,
            created_at=_now(), metadata_json=json.dumps(safe_metadata),
        ))


def load_audit_events(owner_id: str = "local", limit: int = 100) -> list[dict[str, Any]]:
    query = (
        select(audit_events)
        .where(audit_events.c.owner_id == owner_id)
        .order_by(audit_events.c.created_at.desc())
        .limit(min(max(limit, 1), 100))
    )
    with engine.connect() as connection:
        return [
            {
                "id": row.id,
                "analysis_id": row.analysis_id,
                "event_type": row.event_type,
                "created_at": row.created_at.isoformat(),
                "metadata": json.loads(row.metadata_json),
            }
            for row in connection.execute(query)
        ]
