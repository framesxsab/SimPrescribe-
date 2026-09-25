from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import insert, select, update
from sqlalchemy.exc import OperationalError

from .config import settings
from .storage import (
    analyses,
    engine,
    inventory,
    order_events,
    order_items,
    orders,
    pharmacies,
    prescription_files,
    users,
)

logger = logging.getLogger(__name__)
MAX_INTEGER = 2**31 - 1


class MarketplaceConflict(Exception):
    pass


def now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_medicine_name(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def valid_pin(value: str) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[1-9][0-9]{5}", value.strip()))


def _is_expired(value: datetime | str | None, at: datetime | None = None) -> bool:
    if value is None:
        return True
    timestamp = datetime.fromisoformat(value) if isinstance(value, str) else value
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    comparison = at or now()
    if comparison.tzinfo is None:
        comparison = comparison.replace(tzinfo=timezone.utc)
    return timestamp <= comparison


def _require_active_user(connection, user_id: str) -> None:
    active = connection.execute(select(users.c.active).where(users.c.id == user_id)).scalar_one_or_none()
    if active is not True:
        raise PermissionError("An active account is required.")


def pharmacy_for_user(user_id: str) -> dict[str, Any] | None:
    with engine.connect() as connection:
        row = connection.execute(
            select(pharmacies, users.c.active.label("user_active"))
            .join(users, users.c.id == pharmacies.c.user_id)
            .where(pharmacies.c.user_id == user_id)
        ).mappings().one_or_none()
    return dict(row) if row else None


def _require_approved_pharmacy(connection, user_id: str) -> dict[str, Any]:
    pharmacy = connection.execute(
        select(pharmacies, users.c.active.label("user_active"))
        .join(users, users.c.id == pharmacies.c.user_id)
        .where(
            pharmacies.c.user_id == user_id,
            pharmacies.c.approval_status == "approved",
            users.c.active.is_(True),
        )
    ).mappings().one_or_none()
    if pharmacy is None:
        raise PermissionError("An active, approved pharmacy account is required.")
    return dict(pharmacy)


def _approved_pharmacy(connection, pharmacy_id: str) -> dict[str, Any] | None:
    row = connection.execute(
        select(pharmacies)
        .join(users, users.c.id == pharmacies.c.user_id)
        .where(
            pharmacies.c.id == pharmacy_id,
            pharmacies.c.approval_status == "approved",
            users.c.active.is_(True),
        )
    ).mappings().one_or_none()
    return dict(row) if row else None


def save_inventory_item(user_id: str, payload: dict[str, Any], item_id: str | None = None) -> dict[str, Any]:
    name = str(payload.get("medicine_name") or "").strip()[:255]
    normalized = normalize_medicine_name(name)
    unit_label = str(payload.get("unit_label") or "pack").strip()[:128]
    price = payload.get("price_paise")
    stock = payload.get("stock_quantity")
    active = payload.get("active", True)
    if (
        not normalized
        or type(price) is not int
        or type(stock) is not int
        or price < 0
        or price > MAX_INTEGER
        or stock < 0
        or stock > MAX_INTEGER
        or type(active) is not bool
    ):
        raise ValueError("Medicine name, integer price, non-negative stock, and valid active state are required.")
    values = {
        "medicine_name": name,
        "normalized_name": normalized,
        "unit_label": unit_label,
        "price_paise": price,
        "stock_quantity": stock,
        "active": active,
        "updated_at": now(),
    }
    with engine.begin() as connection:
        pharmacy = _require_approved_pharmacy(connection, user_id)
        if item_id:
            result = connection.execute(
                update(inventory)
                .where(inventory.c.id == item_id, inventory.c.pharmacy_id == pharmacy["id"])
                .values(**values)
            )
            if not result.rowcount:
                raise LookupError("Inventory item not found.")
            saved_id = item_id
        else:
            saved_id = str(uuid.uuid4())
            connection.execute(insert(inventory).values(id=saved_id, pharmacy_id=pharmacy["id"], **values))
    return {"id": saved_id, **values}


def pharmacy_inventory(user_id: str, *, active_only: bool = False) -> list[dict[str, Any]]:
    with engine.connect() as connection:
        pharmacy = _require_approved_pharmacy(connection, user_id)
        query = select(inventory).where(inventory.c.pharmacy_id == pharmacy["id"])
        if active_only:
            query = query.where(inventory.c.active.is_(True))
        return [
            dict(row)
            for row in connection.execute(query.order_by(inventory.c.medicine_name)).mappings()
        ]


def deactivate_inventory_item(user_id: str, item_id: str) -> None:
    with engine.begin() as connection:
        pharmacy = _require_approved_pharmacy(connection, user_id)
        result = connection.execute(
            update(inventory)
            .where(inventory.c.id == item_id, inventory.c.pharmacy_id == pharmacy["id"])
            .values(active=False, updated_at=now())
        )
    if not result.rowcount:
        raise LookupError("Inventory item not found.")


def matching_pharmacies(patient_pin: str, medicine_names: list[str]) -> list[dict[str, Any]]:
    patient_pin = str(patient_pin or "").strip()
    if not valid_pin(patient_pin):
        raise ValueError("Enter a valid six-digit PIN code.")
    requested = [
        (str(name), normalize_medicine_name(str(name)))
        for name in medicine_names
        if normalize_medicine_name(str(name))
    ]
    with engine.connect() as connection:
        pharmacy_rows = connection.execute(
            select(pharmacies)
            .join(users, users.c.id == pharmacies.c.user_id)
            .where(pharmacies.c.approval_status == "approved", users.c.active.is_(True))
        ).mappings().all()
        stock_rows = connection.execute(
            select(inventory).where(inventory.c.active.is_(True))
        ).mappings().all()
    stock_by_pharmacy: dict[str, dict[str, dict[str, Any]]] = {}
    for row in stock_rows:
        stock_by_pharmacy.setdefault(row["pharmacy_id"], {})[row["normalized_name"]] = dict(row)
    matches = []
    for row in pharmacy_rows:
        try:
            serviceable = json.loads(row["serviceable_pins_json"] or "[]")
        except (TypeError, json.JSONDecodeError):
            logger.warning("pharmacy_id=%s error_code=SERVICEABLE_PINS_INVALID", row["id"])
            continue
        if not isinstance(serviceable, list):
            continue
        if patient_pin != row["pin_code"] and patient_pin not in serviceable:
            continue
        pharmacy_stock = stock_by_pharmacy.get(row["id"], {})
        availability = []
        for display_name, normalized_name in requested:
            item = pharmacy_stock.get(normalized_name)
            status = "pharmacist_check_required"
            if item:
                status = (
                    "unavailable"
                    if item["stock_quantity"] == 0
                    else "low_stock"
                    if item["stock_quantity"] <= 5
                    else "in_stock"
                )
            availability.append({
                "medicine_name": display_name,
                "status": status,
                "inventory_id": item["id"] if item else None,
                "price_paise": item["price_paise"] if item else None,
                "unit_label": item["unit_label"] if item else "",
            })
        matches.append({
            "id": row["id"],
            "business_name": row["business_name"],
            "address": row["address"],
            "pin_code": row["pin_code"],
            "supports_pickup": row["supports_pickup"],
            "supports_delivery": row["supports_delivery"],
            "exact_pin": patient_pin == row["pin_code"],
            "availability": availability,
            "available_count": sum(
                item["status"] in {"in_stock", "low_stock"} for item in availability
            ),
        })
    return sorted(
        matches,
        key=lambda item: (
            -int(item["exact_pin"]),
            -item["available_count"],
            item["business_name"].casefold(),
        ),
    )


def _prescription_source_is_current(connection, patient_id: str, analysis_id: str, timestamp: datetime) -> bool:
    source = connection.execute(
        select(prescription_files.c.expires_at).where(
            prescription_files.c.analysis_id == analysis_id,
            prescription_files.c.owner_id == patient_id,
        )
    ).scalar_one_or_none()
    return source is not None and not _is_expired(source, timestamp)


def confirmed_prescription(patient_id: str, analysis_id: str) -> dict[str, Any]:
    with engine.connect() as connection:
        row = connection.execute(
            select(analyses.c.payload, analyses.c.expires_at).where(
                analyses.c.id == analysis_id, analyses.c.owner_id == patient_id
            )
        ).mappings().one_or_none()
    if row is None:
        raise LookupError("Prescription analysis not found.")
    if _is_expired(row["expires_at"]):
        raise MarketplaceConflict("This prescription has expired.")
    analysis = json.loads(row["payload"])
    if (
        analysis.get("patient_review_status") not in {"confirmed", "corrected"}
        or analysis.get("prescription_state") != "confirmed"
    ):
        raise MarketplaceConflict("Confirm the prescription before requesting an order.")
    return analysis


def create_order(
    patient_id: str,
    analysis_id: str,
    pharmacy_id: str,
    fulfillment_mode: str,
    delivery_address: str,
    generic_inquiries: list[int],
    *,
    return_created: bool = False,
) -> str | tuple[str, bool]:
    if fulfillment_mode not in {"pickup", "delivery"}:
        raise ValueError("Fulfillment mode must be pickup or delivery.")
    if not isinstance(delivery_address, str) or len(delivery_address) > 1000:
        raise ValueError("Enter a valid delivery address.")
    address = delivery_address.strip() if fulfillment_mode == "delivery" else ""
    if fulfillment_mode == "delivery" and not address:
        raise ValueError("A delivery address is required for delivery.")
    if not isinstance(generic_inquiries, list) or len(generic_inquiries) > 50:
        raise ValueError("Generic inquiries must refer to requested prescription lines.")
    if any(type(index) is not int or index < 0 for index in generic_inquiries):
        raise ValueError("Generic inquiries must refer to requested prescription lines.")
    if len(set(generic_inquiries)) != len(generic_inquiries):
        raise ValueError("Generic inquiry lines must not be repeated.")

    timestamp = now()
    with engine.begin() as connection:
        _require_active_user(connection, patient_id)
        patient = connection.execute(
            select(users.c.pin_code).where(users.c.id == patient_id)
        ).scalar_one_or_none()
        # Serialize requests for this prescription so concurrent retries cannot create
        # duplicate order/event rows. The write locks the analysis row on both SQLite
        # and PostgreSQL before the existing request check.
        locked = connection.execute(
            update(analyses)
            .where(analyses.c.id == analysis_id, analyses.c.owner_id == patient_id)
            .values(payload=analyses.c.payload)
        )
        if not locked.rowcount:
            raise LookupError("Prescription analysis not found.")
        analysis_row = connection.execute(
            select(analyses.c.payload, analyses.c.expires_at).where(
                analyses.c.id == analysis_id, analyses.c.owner_id == patient_id
            )
        ).mappings().one()
        if _is_expired(analysis_row["expires_at"], timestamp):
            raise MarketplaceConflict("This prescription has expired.")
        analysis = json.loads(analysis_row["payload"])
        if (
            analysis.get("patient_review_status") not in {"confirmed", "corrected"}
            or analysis.get("prescription_state") != "confirmed"
        ):
            raise MarketplaceConflict("Confirm the prescription before requesting an order.")
        medications = analysis.get("medications")
        if not isinstance(medications, list) or not medications or len(medications) > 50:
            raise ValueError("A confirmed prescription with medicine lines is required.")
        if any(
            not isinstance(medication, dict)
            or not str(medication.get("name") or "").strip()
            for medication in medications
        ):
            raise ValueError("Every prescription line must have a medicine name.")
        if any(index >= len(medications) for index in generic_inquiries):
            raise ValueError("Generic inquiries must refer to requested prescription lines.")

        pharmacy = _approved_pharmacy(connection, pharmacy_id)
        if pharmacy is None:
            raise LookupError("Approved pharmacy not found.")
        if patient:
            try:
                serviceable_pins = json.loads(pharmacy["serviceable_pins_json"] or "[]")
            except (TypeError, json.JSONDecodeError) as exc:
                logger.warning("pharmacy_id=%s error_code=SERVICEABLE_PINS_INVALID", pharmacy_id)
                raise MarketplaceConflict("Pharmacy service coverage is unavailable.") from exc
            patient_pin = str(patient).strip()
            if patient_pin != pharmacy["pin_code"] and patient_pin not in serviceable_pins:
                raise MarketplaceConflict("This pharmacy does not serve the patient's PIN code.")
        if fulfillment_mode == "pickup" and not pharmacy["supports_pickup"]:
            raise ValueError("This pharmacy does not offer pickup.")
        if fulfillment_mode == "delivery" and not pharmacy["supports_delivery"]:
            raise ValueError("This pharmacy does not offer delivery.")

        source = connection.execute(
            select(prescription_files).where(
                prescription_files.c.analysis_id == analysis_id,
                prescription_files.c.owner_id == patient_id,
            )
        ).mappings().one_or_none()
        if source is None or _is_expired(source["expires_at"], timestamp):
            raise MarketplaceConflict("The original prescription is unavailable; upload it again before ordering.")

        existing_order = connection.execute(
            select(orders).where(
                orders.c.analysis_id == analysis_id,
                orders.c.patient_id == patient_id,
                orders.c.pharmacy_id == pharmacy_id,
            )
        ).mappings().one_or_none()
        requested_lines = sorted(
            (
                json.dumps(medication, sort_keys=True, separators=(",", ":")),
                index in set(generic_inquiries),
            )
            for index, medication in enumerate(medications)
        )
        if existing_order is not None:
            existing_items = connection.execute(
                select(order_items.c.prescription_json, order_items.c.generic_inquiry)
                .where(order_items.c.order_id == existing_order["id"])
            ).all()
            existing_lines = sorted(
                (json.dumps(json.loads(row[0]), sort_keys=True, separators=(",", ":")), bool(row[1]))
                for row in existing_items
            )
            same_request = (
                existing_order["fulfillment_mode"] == fulfillment_mode
                and existing_order["delivery_address"] == address
                and existing_lines == requested_lines
            )
            if not same_request:
                raise MarketplaceConflict("A request already exists for this prescription and pharmacy.")
            order_id = existing_order["id"]
            return (order_id, False) if return_created else order_id

        order_id = str(uuid.uuid4())
        connection.execute(insert(orders).values(
            id=order_id,
            analysis_id=analysis_id,
            patient_id=patient_id,
            pharmacy_id=pharmacy_id,
            status="requested",
            fulfillment_mode=fulfillment_mode,
            delivery_address=address,
            total_paise=0,
            created_at=timestamp,
            updated_at=timestamp,
            expires_at=timestamp + timedelta(days=settings.order_retention_days),
        ))
        for index, medication in enumerate(medications):
            connection.execute(insert(order_items).values(
                id=str(uuid.uuid4()),
                order_id=order_id,
                medicine_name=str(medication["name"]).strip()[:255],
                prescription_json=json.dumps(medication),
                generic_inquiry=index in set(generic_inquiries),
                availability="pending",
                pharmacist_note="",
            ))
        connection.execute(insert(order_events).values(
            id=str(uuid.uuid4()),
            order_id=order_id,
            actor_id=patient_id,
            status="requested",
            note="Order request submitted.",
            created_at=timestamp,
        ))
    return (order_id, True) if return_created else order_id


def _order_row(order_id: str) -> dict[str, Any] | None:
    with engine.connect() as connection:
        row = connection.execute(select(orders).where(orders.c.id == order_id)).mappings().one_or_none()
    return dict(row) if row else None


def _public_pharmacy(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "business_name": row["business_name"],
        "address": row["address"],
        "pin_code": row["pin_code"],
        "supports_pickup": row["supports_pickup"],
        "supports_delivery": row["supports_delivery"],
    }


def order_detail(order_id: str, user: dict[str, str]) -> dict[str, Any]:
    row = _order_row(order_id)
    if row is None:
        raise LookupError("Order not found.")
    if user.get("role") == "pharmacy":
        pharmacy = pharmacy_for_user(user["id"])
        if (
            pharmacy is None
            or pharmacy["approval_status"] != "approved"
            or pharmacy["user_active"] is not True
            or row["pharmacy_id"] != pharmacy["id"]
        ):
            raise PermissionError("Order access denied.")
    elif user.get("role") == "patient":
        with engine.connect() as connection:
            _require_active_user(connection, user["id"])
        if row["patient_id"] != user["id"]:
            raise PermissionError("Order access denied.")
    elif user.get("role") not in {"admin", "reviewer"}:
        raise PermissionError("Order access denied.")

    with engine.connect() as connection:
        items = [
            dict(item)
            for item in connection.execute(
                select(order_items).where(order_items.c.order_id == order_id)
            ).mappings()
        ]
        events = [
            dict(event)
            for event in connection.execute(
                select(order_events)
                .where(order_events.c.order_id == order_id)
                .order_by(order_events.c.created_at, order_events.c.id)
            ).mappings()
        ]
        pharmacy_row = connection.execute(
            select(
                pharmacies.c.id,
                pharmacies.c.business_name,
                pharmacies.c.address,
                pharmacies.c.pin_code,
                pharmacies.c.supports_pickup,
                pharmacies.c.supports_delivery,
            ).where(pharmacies.c.id == row["pharmacy_id"])
        ).mappings().one()
    for item in items:
        item["prescription"] = json.loads(item.pop("prescription_json"))
    row.update(items=items, events=events, pharmacy=dict(pharmacy_row))
    return row


def list_orders_for(user: dict[str, str]) -> list[dict[str, Any]]:
    with engine.connect() as connection:
        query = select(orders)
        if user["role"] == "patient":
            _require_active_user(connection, user["id"])
            query = query.where(orders.c.patient_id == user["id"])
        elif user["role"] == "pharmacy":
            pharmacy = _require_approved_pharmacy(connection, user["id"])
            query = query.where(orders.c.pharmacy_id == pharmacy["id"])
        elif user["role"] not in {"admin", "reviewer"}:
            return []
        rows = [
            dict(row)
            for row in connection.execute(query.order_by(orders.c.created_at.desc())).mappings()
        ]
        for row in rows:
            row["pharmacy_name"] = connection.execute(
                select(pharmacies.c.business_name).where(pharmacies.c.id == row["pharmacy_id"])
            ).scalar_one_or_none() or "Pharmacy"
            row["medicine_names"] = connection.execute(
                select(order_items.c.medicine_name)
                .where(order_items.c.order_id == row["id"])
                .order_by(order_items.c.id)
            ).scalars().all()
        return rows


def quote_order(user_id: str, order_id: str, quoted_items: list[dict[str, Any]]) -> None:
    if not isinstance(quoted_items, list) or not quoted_items or len(quoted_items) > 50:
        raise ValueError("A quote must include each requested medicine exactly once.")
    timestamp = now()
    with engine.begin() as connection:
        pharmacy = _require_approved_pharmacy(connection, user_id)
        order = connection.execute(
            select(orders)
            .where(orders.c.id == order_id, orders.c.pharmacy_id == pharmacy["id"])
            .with_for_update()
        ).mappings().one_or_none()
        if order is None:
            raise PermissionError("Order access denied.")
        if _is_expired(order["expires_at"], timestamp):
            raise MarketplaceConflict("This order has expired.")
        if order["status"] != "requested":
            raise MarketplaceConflict("Only requested orders can be quoted.")

        if not _prescription_source_is_current(connection, order["patient_id"], order["analysis_id"], timestamp):
            raise MarketplaceConflict("The original prescription is unavailable; request a current source before quoting.")
        existing = {
            row["id"]: dict(row)
            for row in connection.execute(
                select(order_items).where(order_items.c.order_id == order_id)
            ).mappings()
        }
        item_ids = [item.get("id") for item in quoted_items if isinstance(item, dict)]
        if (
            len(item_ids) != len(quoted_items)
            or any(not isinstance(item_id, str) or not item_id for item_id in item_ids)
            or len(set(item_ids)) != len(item_ids)
            or set(existing) != set(item_ids)
        ):
            raise ValueError("Quote must include every order item exactly once.")

        total = 0
        normalized_quotes = []
        for quote in quoted_items:
            availability = quote.get("availability")
            if availability not in {"available", "unavailable"}:
                raise ValueError("Availability must be available or unavailable.")
            note = quote.get("pharmacist_note") or ""
            if not isinstance(note, str):
                raise ValueError("Pharmacist notes must be text.")
            note = note.strip()[:1000]
            item = existing[quote["id"]]
            if availability == "available":
                inventory_id = quote.get("inventory_id")
                quantity = quote.get("verified_quantity")
                price = quote.get("unit_price_paise")
                if (
                    not isinstance(inventory_id, str)
                    or type(quantity) is not int
                    or type(price) is not int
                    or quantity < 1
                    or quantity > MAX_INTEGER
                    or price < 0
                    or price > MAX_INTEGER
                ):
                    raise ValueError("Available lines require an inventory item, positive integer quantity, and non-negative integer price.")
                inv = connection.execute(
                    select(inventory).where(
                        inventory.c.id == inventory_id,
                        inventory.c.pharmacy_id == pharmacy["id"],
                        inventory.c.active.is_(True),
                    )
                ).mappings().one_or_none()
                if not inv or inv["normalized_name"] != normalize_medicine_name(item["medicine_name"]):
                    raise ValueError("Available items require an active exact-name inventory match.")
                if inv["stock_quantity"] < quantity:
                    raise MarketplaceConflict("Inventory changed; revise the verified quantity before submitting the quote.")
                total += quantity * price
                if total > MAX_INTEGER:
                    raise ValueError("Quote total exceeds the supported integer-paise range.")
                normalized_quotes.append((
                    item["id"],
                    inventory_id,
                    availability,
                    quantity,
                    price,
                    note,
                ))
            else:
                normalized_quotes.append((
                    item["id"],
                    None,
                    availability,
                    None,
                    None,
                    note,
                ))
        if not any(item[2] == "available" for item in normalized_quotes):
            raise ValueError("A quote requires at least one available medicine; decline the request instead.")

        for item_id, inventory_id, availability, quantity, price, note in normalized_quotes:
            updated = connection.execute(
                update(order_items)
                .where(order_items.c.id == item_id, order_items.c.order_id == order_id)
                .values(
                    inventory_id=inventory_id,
                    availability=availability,
                    verified_quantity=quantity,
                    unit_price_paise=price,
                    pharmacist_note=note,
                )
            )
            if updated.rowcount != 1:
                raise MarketplaceConflict("The requested medicine lines changed. Reload the order.")
        result = connection.execute(
            update(orders)
            .where(
                orders.c.id == order_id,
                orders.c.pharmacy_id == pharmacy["id"],
                orders.c.status == "requested",
                orders.c.expires_at > timestamp,
            )
            .values(status="quoted", total_paise=total, updated_at=timestamp)
        )
        if result.rowcount != 1:
            raise MarketplaceConflict("The order changed before the quote could be submitted.")
        connection.execute(insert(order_events).values(
            id=str(uuid.uuid4()),
            order_id=order_id,
            actor_id=user_id,
            status="quoted",
            note="Pharmacy submitted a verified quote.",
            created_at=timestamp,
        ))


def decline_order(user_id: str, order_id: str, note: str) -> None:
    _record_transition(
        order_id,
        user_id,
        "declined",
        note or "Pharmacy declined the request.",
        {"requested"},
        pharmacy_user_id=user_id,
    )


def _insert_event(connection, order_id: str, actor_id: str, status: str, note: str, timestamp: datetime) -> None:
    connection.execute(insert(order_events).values(
        id=str(uuid.uuid4()),
        order_id=order_id,
        actor_id=actor_id,
        status=status,
        note=note[:1000],
        created_at=timestamp,
    ))


def _reset_quote_for_requote(order_id: str, patient_id: str, timestamp: datetime) -> None:
    with engine.begin() as connection:
        result = connection.execute(
            update(orders)
            .where(
                orders.c.id == order_id,
                orders.c.patient_id == patient_id,
                orders.c.status == "quoted",
            )
            .values(status="requested", total_paise=0, updated_at=timestamp)
        )
        if result.rowcount:
            _insert_event(
                connection,
                order_id,
                patient_id,
                "requested",
                "Inventory changed; pharmacy must submit a new quote.",
                timestamp,
            )


def accept_order(patient_id: str, order_id: str) -> None:
    timestamp = now()
    stock_changed = False
    try:
        with engine.begin() as connection:
            _require_active_user(connection, patient_id)
            order = connection.execute(
                select(orders)
                .where(orders.c.id == order_id, orders.c.patient_id == patient_id)
                .with_for_update()
            ).mappings().one_or_none()
            if order is None:
                raise PermissionError("Order access denied.")
            if _is_expired(order["expires_at"], timestamp):
                raise MarketplaceConflict("This quote has expired.")
            if order["status"] != "quoted":
                raise MarketplaceConflict("Only quoted orders can be accepted.")
            pharmacy = _approved_pharmacy(connection, order["pharmacy_id"])
            if pharmacy is None:
                raise MarketplaceConflict("This pharmacy is no longer eligible to fulfill the order.")
            if not _prescription_source_is_current(connection, patient_id, order["analysis_id"], timestamp):
                raise MarketplaceConflict("The original prescription is unavailable; acceptance needs a current source.")

            items = connection.execute(
                select(order_items)
                .where(order_items.c.order_id == order_id)
                .order_by(order_items.c.id)
            ).mappings().all()
            available_items = [item for item in items if item["availability"] == "available"]
            if not available_items:
                raise MarketplaceConflict("The quote has no available medicine lines.")
            calculated_total = 0
            for item in available_items:
                quantity = item["verified_quantity"]
                price = item["unit_price_paise"]
                if (
                    type(quantity) is not int
                    or type(price) is not int
                    or quantity < 1
                    or price < 0
                    or not item["inventory_id"]
                ):
                    raise MarketplaceConflict("The quote contains an invalid verified line.")
                calculated_total += quantity * price
            if calculated_total > MAX_INTEGER or calculated_total != order["total_paise"]:
                raise MarketplaceConflict("The quote total is no longer valid.")

            claimed = connection.execute(
                update(orders)
                .where(
                    orders.c.id == order_id,
                    orders.c.patient_id == patient_id,
                    orders.c.status == "quoted",
                    orders.c.expires_at > timestamp,
                )
                .values(status="accepted", updated_at=timestamp)
            )
            if claimed.rowcount != 1:
                raise MarketplaceConflict("The quote changed before it could be accepted.")

            stock_savepoint = connection.begin_nested()
            for item in available_items:
                result = connection.execute(
                    update(inventory)
                    .where(
                        inventory.c.id == item["inventory_id"],
                        inventory.c.pharmacy_id == order["pharmacy_id"],
                        inventory.c.active.is_(True),
                        inventory.c.stock_quantity >= item["verified_quantity"],
                    )
                    .values(
                        stock_quantity=inventory.c.stock_quantity - item["verified_quantity"],
                        updated_at=timestamp,
                    )
                )
                if result.rowcount != 1:
                    stock_savepoint.rollback()
                    connection.execute(
                        update(orders)
                        .where(orders.c.id == order_id, orders.c.status == "accepted")
                        .values(status="requested", total_paise=0, updated_at=timestamp)
                    )
                    _insert_event(
                        connection,
                        order_id,
                        patient_id,
                        "requested",
                        "Inventory changed; pharmacy must submit a new quote.",
                        timestamp,
                    )
                    stock_changed = True
                    break
            if not stock_changed:
                stock_savepoint.commit()
                _insert_event(
                    connection,
                    order_id,
                    patient_id,
                    "accepted",
                    "Patient accepted the quote.",
                    timestamp,
                )
    except OperationalError as exc:
        message = str(exc).lower()
        if "locked" not in message and "deadlock" not in message and "serialization" not in message:
            raise
        try:
            _reset_quote_for_requote(order_id, patient_id, timestamp)
        except OperationalError:
            logger.warning("order_id=%s error_code=ACCEPTANCE_CONTENTION", order_id)
        raise MarketplaceConflict("Inventory or order state changed; review a new quote before accepting.") from exc
    if stock_changed:
        raise MarketplaceConflict("Stock changed. The pharmacy must submit a new quote.")


def cancel_order(patient_id: str, order_id: str, note: str = "") -> None:
    _record_transition(
        order_id,
        patient_id,
        "cancelled",
        note or "Patient cancelled the order.",
        {"requested", "quoted", "accepted"},
        patient_id=patient_id,
    )


def pharmacy_transition(user_id: str, order_id: str, target: str, note: str = "") -> None:
    allowed = {
        "preparing": {"accepted"},
        "ready_for_pickup": {"preparing"},
        "out_for_delivery": {"preparing"},
        "fulfilled": {"ready_for_pickup", "out_for_delivery"},
        "cancelled": {"accepted", "preparing"},
        "expired": {"requested", "quoted"},
    }
    if target not in allowed:
        raise ValueError("Invalid fulfillment status.")
    _record_transition(
        order_id,
        user_id,
        target,
        note or target.replace("_", " ").title(),
        allowed[target],
        pharmacy_user_id=user_id,
    )


def _record_transition(
    order_id: str,
    actor_id: str,
    status: str,
    note: str,
    source: set[str],
    *,
    patient_id: str | None = None,
    pharmacy_user_id: str | None = None,
) -> None:
    timestamp = now()
    with engine.begin() as connection:
        pharmacy = None
        if pharmacy_user_id is not None:
            pharmacy = _require_approved_pharmacy(connection, pharmacy_user_id)
            query = select(orders).where(
                orders.c.id == order_id,
                orders.c.pharmacy_id == pharmacy["id"],
            )
        else:
            query = select(orders).where(orders.c.id == order_id)
        order = connection.execute(query.with_for_update()).mappings().one_or_none()
        if order is None:
            raise PermissionError("Order access denied.")
        if patient_id is not None:
            _require_active_user(connection, patient_id)
            if order["patient_id"] != patient_id:
                raise PermissionError("Order access denied.")
        if pharmacy_user_id is None and patient_id is None:
            raise PermissionError("Order access denied.")
        if status != "expired" and _is_expired(order["expires_at"], timestamp):
            raise MarketplaceConflict("This order has expired.")
        if order["status"] not in source:
            raise MarketplaceConflict(f"Order cannot move from {order['status']} to {status}.")

        if status == "ready_for_pickup" and order["fulfillment_mode"] != "pickup":
            raise MarketplaceConflict("Delivery orders cannot be marked ready for pickup.")
        if status == "out_for_delivery" and order["fulfillment_mode"] != "delivery":
            raise MarketplaceConflict("Pickup orders cannot be marked out for delivery.")
        if status == "fulfilled":
            expected_mode = "pickup" if order["status"] == "ready_for_pickup" else "delivery"
            if order["fulfillment_mode"] != expected_mode:
                raise MarketplaceConflict("The fulfillment state does not match this order's mode.")

        result = connection.execute(
            update(orders)
            .where(orders.c.id == order_id, orders.c.status == order["status"])
            .values(status=status, updated_at=timestamp)
        )
        if result.rowcount != 1:
            raise MarketplaceConflict("The order changed before the status update.")

        if status == "cancelled" and order["status"] in {"accepted", "preparing"}:
            accepted_items = connection.execute(
                select(order_items).where(
                    order_items.c.order_id == order_id,
                    order_items.c.availability == "available",
                )
            ).mappings().all()
            for item in accepted_items:
                if not item["inventory_id"] or type(item["verified_quantity"]) is not int:
                    raise MarketplaceConflict("Accepted inventory could not be restored safely.")
                restored = connection.execute(
                    update(inventory)
                    .where(
                        inventory.c.id == item["inventory_id"],
                        inventory.c.pharmacy_id == order["pharmacy_id"],
                    )
                    .values(
                        stock_quantity=inventory.c.stock_quantity + item["verified_quantity"],
                        updated_at=timestamp,
                    )
                )
                if restored.rowcount != 1:
                    raise MarketplaceConflict("Accepted inventory could not be restored safely.")

        _insert_event(connection, order_id, actor_id, status, note, timestamp)


def pharmacy_can_access_analysis(user_id: str, analysis_id: str) -> bool:
    timestamp = now()
    with engine.connect() as connection:
        pharmacy = connection.execute(
            select(pharmacies.c.id)
            .join(users, users.c.id == pharmacies.c.user_id)
            .where(
                pharmacies.c.user_id == user_id,
                pharmacies.c.approval_status == "approved",
                users.c.active.is_(True),
            )
        ).scalar_one_or_none()
        if pharmacy is None:
            return False
        return connection.execute(
            select(orders.c.id)
            .where(
                orders.c.analysis_id == analysis_id,
                orders.c.pharmacy_id == pharmacy,
                orders.c.expires_at > timestamp,
                orders.c.status.not_in({"declined", "cancelled", "expired"}),
            )
            .limit(1)
        ).scalar_one_or_none() is not None
