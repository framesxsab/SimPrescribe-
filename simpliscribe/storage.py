import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, MetaData, String, Table, Text, UniqueConstraint, create_engine, delete, insert, select, text, update

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
vector_cache_table = Table(
    "vector_cache",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("text_hash", String(64), nullable=False, index=True),
    Column("raw_text", Text, nullable=False),
    Column("vector_json", Text, nullable=False),
    Column("payload_json", Text, nullable=False),
    Column("hit_count", Integer, nullable=False, default=0),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
    Column("last_accessed_at", DateTime(timezone=True), nullable=False),
)
users = Table(
    "users", metadata,
    Column("id", String(36), primary_key=True),
    Column("email", String(255), nullable=False, unique=True, index=True),
    Column("password_hash", Text, nullable=False),
    Column("role", String(32), nullable=False, index=True),
    Column("full_name", String(255), nullable=False),
    Column("phone", String(32), nullable=False),
    Column("pin_code", String(6), nullable=False, index=True),
    Column("active", Boolean, nullable=False, default=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
)
pharmacies = Table(
    "pharmacies", metadata,
    Column("id", String(36), primary_key=True),
    Column("user_id", String(36), ForeignKey("users.id"), nullable=False, unique=True, index=True),
    Column("business_name", String(255), nullable=False),
    Column("license_number", String(128), nullable=False, unique=True),
    Column("address", Text, nullable=False),
    Column("pin_code", String(6), nullable=False, index=True),
    Column("serviceable_pins_json", Text, nullable=False, default="[]"),
    Column("supports_pickup", Boolean, nullable=False, default=True),
    Column("supports_delivery", Boolean, nullable=False, default=False),
    Column("approval_status", String(32), nullable=False, default="pending", index=True),
    Column("approved_at", DateTime(timezone=True), nullable=True),
)
inventory = Table(
    "inventory", metadata,
    Column("id", String(36), primary_key=True),
    Column("pharmacy_id", String(36), ForeignKey("pharmacies.id"), nullable=False, index=True),
    Column("medicine_name", String(255), nullable=False),
    Column("normalized_name", String(255), nullable=False),
    Column("unit_label", String(128), nullable=False),
    Column("price_paise", Integer, nullable=False),
    Column("stock_quantity", Integer, nullable=False),
    Column("active", Boolean, nullable=False, default=True),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    UniqueConstraint("pharmacy_id", "normalized_name", name="uq_inventory_pharmacy_medicine"),
)
prescription_files = Table(
    "prescription_files", metadata,
    Column("id", String(36), primary_key=True),
    Column("analysis_id", String(36), ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, unique=True, index=True),
    Column("owner_id", String(255), nullable=False, index=True),
    Column("storage_name", String(255), nullable=False, unique=True),
    Column("original_name", String(255), nullable=False),
    Column("content_type", String(128), nullable=False),
    Column("sha256", String(64), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False, index=True),
)
orders = Table(
    "orders", metadata,
    Column("id", String(36), primary_key=True),
    Column("analysis_id", String(36), nullable=False, index=True),
    Column("patient_id", String(255), nullable=False, index=True),
    Column("pharmacy_id", String(36), ForeignKey("pharmacies.id"), nullable=False, index=True),
    Column("status", String(32), nullable=False, index=True),
    Column("fulfillment_mode", String(32), nullable=False),
    Column("delivery_address", Text, nullable=False, default=""),
    Column("total_paise", Integer, nullable=False, default=0),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False, index=True),
)
order_items = Table(
    "order_items", metadata,
    Column("id", String(36), primary_key=True),
    Column("order_id", String(36), ForeignKey("orders.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("inventory_id", String(36), ForeignKey("inventory.id"), nullable=True),
    Column("medicine_name", String(255), nullable=False),
    Column("prescription_json", Text, nullable=False),
    Column("generic_inquiry", Boolean, nullable=False, default=False),
    Column("availability", String(32), nullable=False, default="pending"),
    Column("verified_quantity", Integer, nullable=True),
    Column("unit_price_paise", Integer, nullable=True),
    Column("pharmacist_note", Text, nullable=False, default=""),
)
order_events = Table(
    "order_events", metadata,
    Column("id", String(36), primary_key=True),
    Column("order_id", String(36), ForeignKey("orders.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("actor_id", String(255), nullable=False),
    Column("status", String(32), nullable=False),
    Column("note", Text, nullable=False, default=""),
    Column("created_at", DateTime(timezone=True), nullable=False),
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


def save_vector_cache_entry(
    entry_id: str,
    text_hash: str,
    raw_text: str,
    vector_json: str,
    payload_json: str,
) -> None:
    now = _now()
    with engine.begin() as connection:
        # Check if already exists by text_hash
        existing = connection.execute(
            select(vector_cache_table.c.id).where(vector_cache_table.c.text_hash == text_hash)
        ).scalar_one_or_none()
        if existing:
            connection.execute(
                update(vector_cache_table)
                .where(vector_cache_table.c.id == existing)
                .values(
                    raw_text=raw_text,
                    vector_json=vector_json,
                    payload_json=payload_json,
                    last_accessed_at=now,
                )
            )
        else:
            connection.execute(
                insert(vector_cache_table).values(
                    id=entry_id,
                    text_hash=text_hash,
                    raw_text=raw_text,
                    vector_json=vector_json,
                    payload_json=payload_json,
                    hit_count=0,
                    created_at=now,
                    last_accessed_at=now,
                )
            )


def load_vector_cache_entries(limit: int = 1000) -> list[dict[str, Any]]:
    query = (
        select(vector_cache_table)
        .order_by(vector_cache_table.c.hit_count.desc(), vector_cache_table.c.last_accessed_at.desc())
        .limit(limit)
    )
    with engine.connect() as connection:
        return [
            {
                "id": row.id,
                "text_hash": row.text_hash,
                "raw_text": row.raw_text,
                "vector": json.loads(row.vector_json),
                "payload": json.loads(row.payload_json),
                "hit_count": row.hit_count,
            }
            for row in connection.execute(query)
        ]


def increment_vector_cache_hit(entry_id: str) -> None:
    now = _now()
    with engine.begin() as connection:
        connection.execute(
            update(vector_cache_table)
            .where(vector_cache_table.c.id == entry_id)
            .values(
                hit_count=vector_cache_table.c.hit_count + 1,
                last_accessed_at=now,
            )
        )


def clear_vector_cache_db() -> int:
    with engine.begin() as connection:
        res = connection.execute(delete(vector_cache_table))
        return int(res.rowcount or 0)


def get_vector_cache_stats() -> dict[str, Any]:
    with engine.connect() as connection:
        count = connection.execute(select(text("count(*)")).select_from(vector_cache_table)).scalar() or 0
        total_hits = connection.execute(select(text("coalesce(sum(hit_count), 0)")).select_from(vector_cache_table)).scalar() or 0
        return {"total_db_entries": int(count), "total_db_hits": int(total_hits)}


def get_user_by_email(email: str) -> dict[str, Any] | None:
    query = select(users).where(users.c.email == email.strip().lower())
    with engine.connect() as connection:
        row = connection.execute(query).mappings().one_or_none()
    if not row:
        return None
    result = dict(row)
    if result["role"] == "pharmacy":
        pharmacy = get_pharmacy_by_user(result["id"])
        result["approval_status"] = pharmacy["approval_status"] if pharmacy else "pending"
    return result


def get_user(user_id: str) -> dict[str, Any] | None:
    with engine.connect() as connection:
        row = connection.execute(select(users).where(users.c.id == user_id)).mappings().one_or_none()
    return dict(row) if row else None


def create_user(user: dict[str, Any], pharmacy: dict[str, Any] | None = None) -> None:
    with engine.begin() as connection:
        connection.execute(insert(users).values(**user))
        if pharmacy:
            connection.execute(insert(pharmacies).values(**pharmacy))


def get_pharmacy_by_user(user_id: str) -> dict[str, Any] | None:
    with engine.connect() as connection:
        row = connection.execute(select(pharmacies).where(pharmacies.c.user_id == user_id)).mappings().one_or_none()
    if not row:
        return None
    result = dict(row)
    result["serviceable_pins"] = json.loads(result.pop("serviceable_pins_json"))
    return result


def list_pharmacies(status: str | None = None) -> list[dict[str, Any]]:
    query = select(pharmacies, users.c.email, users.c.phone).join(users, users.c.id == pharmacies.c.user_id)
    if status:
        query = query.where(pharmacies.c.approval_status == status)
    query = query.order_by(pharmacies.c.business_name)
    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()
    results = []
    for row in rows:
        item = dict(row)
        item["serviceable_pins"] = json.loads(item.pop("serviceable_pins_json"))
        results.append(item)
    return results


def set_pharmacy_approval(pharmacy_id: str, status: str) -> bool:
    values: dict[str, Any] = {"approval_status": status, "approved_at": _now() if status == "approved" else None}
    with engine.begin() as connection:
        result = connection.execute(update(pharmacies).where(pharmacies.c.id == pharmacy_id).values(**values))
    return bool(result.rowcount)


def save_prescription_file(record: dict[str, Any]) -> None:
    with engine.begin() as connection:
        connection.execute(insert(prescription_files).values(**record))


def get_prescription_file(analysis_id: str) -> dict[str, Any] | None:
    with engine.connect() as connection:
        row = connection.execute(select(prescription_files).where(prescription_files.c.analysis_id == analysis_id)).mappings().one_or_none()
    return dict(row) if row else None


def purge_expired_marketplace() -> tuple[list[str], int]:
    now = _now()
    with engine.begin() as connection:
        expired_files = connection.execute(
            select(prescription_files.c.storage_name).where(prescription_files.c.expires_at < now)
        ).scalars().all()
        if expired_files:
            connection.execute(delete(prescription_files).where(prescription_files.c.expires_at < now))
        expired_order_ids = connection.execute(select(orders.c.id).where(orders.c.expires_at < now)).scalars().all()
        if expired_order_ids:
            connection.execute(delete(order_events).where(order_events.c.order_id.in_(expired_order_ids)))
            connection.execute(delete(order_items).where(order_items.c.order_id.in_(expired_order_ids)))
        expired_orders = connection.execute(delete(orders).where(orders.c.expires_at < now))
    return list(expired_files), int(expired_orders.rowcount or 0)


def seed_test_pharmacies() -> None:
    """Seed approved test pharmacies in development mode when none exist.

    Creates three pharmacies with realistic inventory covering common PIN codes
    (560001 Bangalore, 110001 Delhi, 400001 Mumbai) so the upload-to-order flow
    can be tested without manual registration or admin approval.
    """
    import uuid as _uuid
    from .security import hash_password

    with engine.connect() as connection:
        count = connection.execute(select(text("count(*)")).select_from(pharmacies)).scalar() or 0
    if count > 0:
        return

    logger.info("Seeding test pharmacies for development mode...")
    pharmacy_configs = [
        {
            "email": "medplus-test@localhost",
            "full_name": "MedPlus Test Pharmacy",
            "phone": "9876543210",
            "pin_code": "560001",
            "business_name": "MedPlus Pharmacy (Test)",
            "license_number": "TEST-KA-PH-001",
            "address": "123 MG Road, Bengaluru, Karnataka 560001",
            "serviceable_pins": ["560001", "560002", "560003"],
            "supports_pickup": True,
            "supports_delivery": True,
            "inventory": [
                ("Paracetamol 500mg", "tablet", 120, 50),
                ("Amoxicillin 250mg", "capsule", 350, 30),
                ("Cetirizine 10mg", "tablet", 85, 100),
                ("Azithromycin 500mg", "tablet", 450, 20),
                ("Pantoprazole 40mg", "tablet", 180, 40),
            ],
        },
        {
            "email": "apollo-test@localhost",
            "full_name": "Apollo Pharmacy Test",
            "phone": "9876543211",
            "pin_code": "110001",
            "business_name": "Apollo Pharmacy (Test)",
            "license_number": "TEST-DL-PH-002",
            "address": "45 Connaught Place, New Delhi 110001",
            "serviceable_pins": ["110001", "110002"],
            "supports_pickup": True,
            "supports_delivery": False,
            "inventory": [
                ("Paracetamol 500mg", "tablet", 100, 80),
                ("Ibuprofen 400mg", "tablet", 150, 60),
                ("Metformin 500mg", "tablet", 200, 45),
                ("Cetirizine 10mg", "tablet", 75, 120),
                ("Omeprazole 20mg", "capsule", 220, 35),
            ],
        },
        {
            "email": "netmeds-test@localhost",
            "full_name": "Netmeds Test Pharmacy",
            "phone": "9876543212",
            "pin_code": "400001",
            "business_name": "Netmeds Pharmacy (Test)",
            "license_number": "TEST-MH-PH-003",
            "address": "78 Marine Drive, Mumbai, Maharashtra 400001",
            "serviceable_pins": ["400001", "400002", "400003"],
            "supports_pickup": True,
            "supports_delivery": True,
            "inventory": [
                ("Paracetamol 500mg", "tablet", 110, 70),
                ("Amoxicillin 250mg", "capsule", 320, 25),
                ("Pantoprazole 40mg", "tablet", 165, 50),
                ("Atorvastatin 10mg", "tablet", 280, 30),
                ("Azithromycin 500mg", "tablet", 420, 15),
            ],
        },
    ]

    now = _now()
    password_hash = hash_password("test-password-dev-only")
    for config in pharmacy_configs:
        user_id = str(_uuid.uuid4())
        pharmacy_id = str(_uuid.uuid4())
        try:
            with engine.begin() as connection:
                connection.execute(insert(users).values(
                    id=user_id, email=config["email"], password_hash=password_hash,
                    role="pharmacy", full_name=config["full_name"], phone=config["phone"],
                    pin_code=config["pin_code"], active=True, created_at=now,
                ))
                connection.execute(insert(pharmacies).values(
                    id=pharmacy_id, user_id=user_id, business_name=config["business_name"],
                    license_number=config["license_number"], address=config["address"],
                    pin_code=config["pin_code"],
                    serviceable_pins_json=json.dumps(sorted(set(config["serviceable_pins"]))),
                    supports_pickup=config["supports_pickup"],
                    supports_delivery=config["supports_delivery"],
                    approval_status="approved", approved_at=now,
                ))
                for med_name, unit, price, stock in config["inventory"]:
                    normalized = med_name.strip().upper()
                    connection.execute(insert(inventory).values(
                        id=str(_uuid.uuid4()), pharmacy_id=pharmacy_id,
                        medicine_name=med_name, normalized_name=normalized,
                        unit_label=unit, price_paise=price, stock_quantity=stock,
                        active=True, updated_at=now,
                    ))
            logger.info("  Seeded: %s (%s)", config["business_name"], config["pin_code"])
        except Exception:
            logger.warning("  Skipping %s (already exists or error)", config["business_name"])

