from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import insert, select, update

from .config import settings
from .storage import analyses, engine, inventory, order_events, order_items, orders, pharmacies, prescription_files


class MarketplaceConflict(Exception):
    pass


def now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_medicine_name(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def valid_pin(value: str) -> bool:
    return bool(re.fullmatch(r"[1-9][0-9]{5}", value.strip()))


def pharmacy_for_user(user_id: str) -> dict[str, Any] | None:
    with engine.connect() as connection:
        row = connection.execute(select(pharmacies).where(pharmacies.c.user_id == user_id)).mappings().one_or_none()
    return dict(row) if row else None


def save_inventory_item(user_id: str, payload: dict[str, Any], item_id: str | None = None) -> dict[str, Any]:
    pharmacy = pharmacy_for_user(user_id)
    if not pharmacy or pharmacy["approval_status"] != "approved":
        raise PermissionError("An approved pharmacy account is required.")
    name = str(payload.get("medicine_name") or "").strip()[:255]
    normalized = normalize_medicine_name(name)
    unit_label = str(payload.get("unit_label") or "pack").strip()[:128]
    price = int(payload.get("price_paise", -1))
    stock = int(payload.get("stock_quantity", -1))
    if not normalized or price < 0 or stock < 0:
        raise ValueError("Medicine name, non-negative price, and non-negative stock are required.")
    values = {
        "medicine_name": name, "normalized_name": normalized, "unit_label": unit_label,
        "price_paise": price, "stock_quantity": stock, "active": bool(payload.get("active", True)), "updated_at": now(),
    }
    with engine.begin() as connection:
        if item_id:
            result = connection.execute(update(inventory).where(
                inventory.c.id == item_id, inventory.c.pharmacy_id == pharmacy["id"]
            ).values(**values))
            if not result.rowcount:
                raise LookupError("Inventory item not found.")
            saved_id = item_id
        else:
            saved_id = str(uuid.uuid4())
            connection.execute(insert(inventory).values(id=saved_id, pharmacy_id=pharmacy["id"], **values))
    return {"id": saved_id, **values}


def pharmacy_inventory(user_id: str) -> list[dict[str, Any]]:
    pharmacy = pharmacy_for_user(user_id)
    if not pharmacy:
        return []
    with engine.connect() as connection:
        return [dict(row) for row in connection.execute(
            select(inventory).where(inventory.c.pharmacy_id == pharmacy["id"]).order_by(inventory.c.medicine_name)
        ).mappings()]


def deactivate_inventory_item(user_id: str, item_id: str) -> None:
    pharmacy = pharmacy_for_user(user_id)
    if not pharmacy:
        raise PermissionError("Pharmacy access denied.")
    with engine.begin() as connection:
        result = connection.execute(update(inventory).where(
            inventory.c.id == item_id, inventory.c.pharmacy_id == pharmacy["id"]
        ).values(active=False, updated_at=now()))
    if not result.rowcount:
        raise LookupError("Inventory item not found.")


def matching_pharmacies(patient_pin: str, medicine_names: list[str]) -> list[dict[str, Any]]:
    requested = [(name, normalize_medicine_name(name)) for name in medicine_names if normalize_medicine_name(name)]
    with engine.connect() as connection:
        pharmacy_rows = connection.execute(select(pharmacies).where(pharmacies.c.approval_status == "approved")).mappings().all()
        stock_rows = connection.execute(select(inventory).where(inventory.c.active.is_(True))).mappings().all()
    stock_by_pharmacy: dict[str, dict[str, dict[str, Any]]] = {}
    for row in stock_rows:
        stock_by_pharmacy.setdefault(row["pharmacy_id"], {})[row["normalized_name"]] = dict(row)
    matches = []
    for row in pharmacy_rows:
        serviceable = json.loads(row["serviceable_pins_json"] or "[]")
        if patient_pin != row["pin_code"] and patient_pin not in serviceable:
            continue
        pharmacy_stock = stock_by_pharmacy.get(row["id"], {})
        availability = []
        for display_name, name in requested:
            item = pharmacy_stock.get(name)
            status = "pharmacist_check_required"
            if item:
                status = "unavailable" if item["stock_quantity"] == 0 else "low_stock" if item["stock_quantity"] <= 5 else "in_stock"
            availability.append({"medicine_name": display_name, "status": status, "inventory_id": item["id"] if item else None,
                                 "price_paise": item["price_paise"] if item else None, "unit_label": item["unit_label"] if item else ""})
        result = dict(row)
        result.pop("serviceable_pins_json", None)
        result["exact_pin"] = patient_pin == row["pin_code"]
        result["availability"] = availability
        result["available_count"] = sum(item["status"] in {"in_stock", "low_stock"} for item in availability)
        matches.append(result)
    return sorted(matches, key=lambda item: (-int(item["exact_pin"]), -item["available_count"], item["business_name"].lower()))


def create_order(patient_id: str, analysis_id: str, pharmacy_id: str, fulfillment_mode: str,
                 delivery_address: str, generic_inquiries: list[int]) -> str:
    with engine.begin() as connection:
        analysis_payload = connection.execute(select(analyses.c.payload).where(
            analyses.c.id == analysis_id, analyses.c.owner_id == patient_id
        )).scalar_one_or_none()
        pharmacy = connection.execute(select(pharmacies).where(
            pharmacies.c.id == pharmacy_id, pharmacies.c.approval_status == "approved"
        )).mappings().one_or_none()
        if not analysis_payload:
            raise LookupError("Prescription analysis not found.")
        if not pharmacy:
            raise LookupError("Approved pharmacy not found.")
        analysis = json.loads(analysis_payload)
        if analysis.get("patient_review_status") not in {"confirmed", "corrected"} or analysis.get("prescription_state", "confirmed") != "confirmed":
            raise ValueError("Confirm the prescription before requesting an order.")
        retained_source = connection.execute(select(prescription_files.c.id).where(
            prescription_files.c.analysis_id == analysis_id, prescription_files.c.expires_at >= now()
        )).scalar_one_or_none()
        if not retained_source:
            raise ValueError("The original prescription is unavailable; upload it again before ordering.")
        if fulfillment_mode not in {"pickup", "delivery"}:
            raise ValueError("Fulfillment mode must be pickup or delivery.")
        if fulfillment_mode == "pickup" and not pharmacy["supports_pickup"]:
            raise ValueError("This pharmacy does not offer pickup.")
        if fulfillment_mode == "delivery" and (not pharmacy["supports_delivery"] or not delivery_address.strip()):
            raise ValueError("A delivery address is required for a pharmacy offering delivery.")
        order_id = str(uuid.uuid4())
        timestamp = now()
        connection.execute(insert(orders).values(
            id=order_id, analysis_id=analysis_id, patient_id=patient_id, pharmacy_id=pharmacy_id,
            status="requested", fulfillment_mode=fulfillment_mode, delivery_address=delivery_address.strip()[:1000],
            total_paise=0, created_at=timestamp, updated_at=timestamp,
            expires_at=timestamp + timedelta(days=settings.order_retention_days),
        ))
        for index, medication in enumerate(analysis.get("medications") or []):
            connection.execute(insert(order_items).values(
                id=str(uuid.uuid4()), order_id=order_id, medicine_name=str(medication.get("name") or "Unknown")[:255],
                prescription_json=json.dumps(medication), generic_inquiry=index in generic_inquiries,
                availability="pending", pharmacist_note="",
            ))
        connection.execute(insert(order_events).values(
            id=str(uuid.uuid4()), order_id=order_id, actor_id=patient_id, status="requested",
            note="Order request submitted.", created_at=timestamp,
        ))
    return order_id


def _order_row(order_id: str) -> dict[str, Any] | None:
    with engine.connect() as connection:
        row = connection.execute(select(orders).where(orders.c.id == order_id)).mappings().one_or_none()
    return dict(row) if row else None


def order_detail(order_id: str, user: dict[str, str]) -> dict[str, Any]:
    row = _order_row(order_id)
    if not row:
        raise LookupError("Order not found.")
    pharmacy = pharmacy_for_user(user["id"]) if user["role"] == "pharmacy" else None
    if user["role"] not in {"admin", "reviewer"} and row["patient_id"] != user["id"] and (not pharmacy or row["pharmacy_id"] != pharmacy["id"]):
        raise PermissionError("Order access denied.")
    with engine.connect() as connection:
        items = [dict(item) for item in connection.execute(select(order_items).where(order_items.c.order_id == order_id)).mappings()]
        events = [dict(event) for event in connection.execute(
            select(order_events).where(order_events.c.order_id == order_id).order_by(order_events.c.created_at)
        ).mappings()]
        pharmacy_row = connection.execute(select(pharmacies).where(pharmacies.c.id == row["pharmacy_id"])).mappings().one()
    for item in items:
        item["prescription"] = json.loads(item.pop("prescription_json"))
    row.update(items=items, events=events, pharmacy=dict(pharmacy_row))
    return row


def list_orders_for(user: dict[str, str]) -> list[dict[str, Any]]:
    query = select(orders)
    if user["role"] == "patient":
        query = query.where(orders.c.patient_id == user["id"])
    elif user["role"] == "pharmacy":
        pharmacy = pharmacy_for_user(user["id"])
        if not pharmacy:
            return []
        query = query.where(orders.c.pharmacy_id == pharmacy["id"])
    elif user["role"] not in {"admin", "reviewer"}:
        return []
    with engine.connect() as connection:
        return [dict(row) for row in connection.execute(query.order_by(orders.c.created_at.desc())).mappings()]


def quote_order(user_id: str, order_id: str, quoted_items: list[dict[str, Any]]) -> None:
    pharmacy = pharmacy_for_user(user_id)
    order = _order_row(order_id)
    if not pharmacy or not order or order["pharmacy_id"] != pharmacy["id"]:
        raise PermissionError("Order access denied.")
    if order["status"] != "requested":
        raise MarketplaceConflict("Only requested orders can be quoted.")
    total = 0
    timestamp = now()
    with engine.begin() as connection:
        existing = {row["id"]: dict(row) for row in connection.execute(
            select(order_items).where(order_items.c.order_id == order_id)
        ).mappings()}
        if set(existing) != {str(item.get("id")) for item in quoted_items}:
            raise ValueError("Quote must include every order item exactly once.")
        for quote in quoted_items:
            item = existing[str(quote["id"])]
            availability = str(quote.get("availability") or "")
            quantity = int(quote.get("verified_quantity") or 0)
            price = int(quote.get("unit_price_paise") or 0)
            inventory_id = quote.get("inventory_id")
            if availability not in {"available", "unavailable"}:
                raise ValueError("Availability must be available or unavailable.")
            if availability == "available":
                inv = connection.execute(select(inventory).where(
                    inventory.c.id == inventory_id, inventory.c.pharmacy_id == pharmacy["id"], inventory.c.active.is_(True)
                )).mappings().one_or_none()
                if not inv or inv["normalized_name"] != normalize_medicine_name(item["medicine_name"]):
                    raise ValueError("Available items require an exact-name inventory match.")
                if quantity < 1 or price < 0:
                    raise ValueError("Available items require a verified quantity and non-negative price.")
                total += quantity * price
            connection.execute(update(order_items).where(order_items.c.id == item["id"]).values(
                inventory_id=inventory_id if availability == "available" else None,
                availability=availability, verified_quantity=quantity if availability == "available" else None,
                unit_price_paise=price if availability == "available" else None,
                pharmacist_note=str(quote.get("pharmacist_note") or "")[:1000],
            ))
        if not any(str(item.get("availability")) == "available" for item in quoted_items):
            raise ValueError("A quote requires at least one available medicine; decline the request instead.")
        connection.execute(update(orders).where(orders.c.id == order_id).values(status="quoted", total_paise=total, updated_at=timestamp))
        connection.execute(insert(order_events).values(id=str(uuid.uuid4()), order_id=order_id, actor_id=user_id,
                                                               status="quoted", note="Pharmacy submitted a verified quote.", created_at=timestamp))


def decline_order(user_id: str, order_id: str, note: str) -> None:
    _pharmacy_transition(user_id, order_id, {"requested"}, "declined", note or "Pharmacy declined the request.")


def accept_order(patient_id: str, order_id: str) -> None:
    order = _order_row(order_id)
    if not order or order["patient_id"] != patient_id:
        raise PermissionError("Order access denied.")
    if order["status"] != "quoted":
        raise MarketplaceConflict("Only quoted orders can be accepted.")
    failed = False
    timestamp = now()
    with engine.begin() as connection:
        items = connection.execute(select(order_items).where(
            order_items.c.order_id == order_id, order_items.c.availability == "available"
        )).mappings().all()
        for item in items:
            stock = connection.execute(select(inventory.c.stock_quantity).where(
                inventory.c.id == item["inventory_id"], inventory.c.active.is_(True)
            )).scalar_one_or_none()
            if stock is None or stock < item["verified_quantity"]:
                failed = True
                break
        if not failed:
            for item in items:
                connection.execute(update(inventory).where(inventory.c.id == item["inventory_id"]).values(
                    stock_quantity=inventory.c.stock_quantity - item["verified_quantity"], updated_at=timestamp
                ))
            connection.execute(update(orders).where(orders.c.id == order_id).values(status="accepted", updated_at=timestamp))
            connection.execute(insert(order_events).values(id=str(uuid.uuid4()), order_id=order_id, actor_id=patient_id,
                                                                   status="accepted", note="Patient accepted the quote.", created_at=timestamp))
    if failed:
        with engine.begin() as connection:
            connection.execute(update(orders).where(orders.c.id == order_id).values(status="requested", total_paise=0, updated_at=timestamp))
            connection.execute(insert(order_events).values(id=str(uuid.uuid4()), order_id=order_id, actor_id=patient_id,
                                                                   status="requested", note="Stock changed; pharmacy must submit a new quote.", created_at=timestamp))
        raise MarketplaceConflict("Stock changed. The pharmacy must submit a new quote.")


def cancel_order(patient_id: str, order_id: str, note: str = "") -> None:
    order = _order_row(order_id)
    if not order or order["patient_id"] != patient_id:
        raise PermissionError("Order access denied.")
    if order["status"] not in {"requested", "quoted", "accepted"}:
        raise MarketplaceConflict("This order can no longer be cancelled by the patient.")
    _record_transition(order_id, patient_id, "cancelled", note or "Patient cancelled the order.")


def pharmacy_transition(user_id: str, order_id: str, target: str, note: str = "") -> None:
    allowed = {
        "preparing": {"accepted"}, "ready_for_pickup": {"preparing"}, "out_for_delivery": {"preparing"},
        "fulfilled": {"ready_for_pickup", "out_for_delivery"}, "cancelled": {"accepted", "preparing"},
        "expired": {"requested", "quoted"},
    }
    if target not in allowed:
        raise ValueError("Invalid fulfillment status.")
    _pharmacy_transition(user_id, order_id, allowed[target], target, note or target.replace("_", " ").title())


def _pharmacy_transition(user_id: str, order_id: str, source: set[str], target: str, note: str) -> None:
    pharmacy = pharmacy_for_user(user_id)
    order = _order_row(order_id)
    if not pharmacy or not order or order["pharmacy_id"] != pharmacy["id"]:
        raise PermissionError("Order access denied.")
    if order["status"] not in source:
        raise MarketplaceConflict(f"Order cannot move from {order['status']} to {target}.")
    _record_transition(order_id, user_id, target, note)


def _record_transition(order_id: str, actor_id: str, status: str, note: str) -> None:
    timestamp = now()
    with engine.begin() as connection:
        previous = connection.execute(select(orders.c.status).where(orders.c.id == order_id)).scalar_one()
        if status == "cancelled" and previous in {"accepted", "preparing"}:
            accepted_items = connection.execute(select(order_items).where(
                order_items.c.order_id == order_id, order_items.c.availability == "available"
            )).mappings().all()
            for item in accepted_items:
                connection.execute(update(inventory).where(inventory.c.id == item["inventory_id"]).values(
                    stock_quantity=inventory.c.stock_quantity + item["verified_quantity"], updated_at=timestamp
                ))
        connection.execute(update(orders).where(orders.c.id == order_id).values(status=status, updated_at=timestamp))
        connection.execute(insert(order_events).values(id=str(uuid.uuid4()), order_id=order_id, actor_id=actor_id,
                                                               status=status, note=note[:1000], created_at=timestamp))


def pharmacy_can_access_analysis(user_id: str, analysis_id: str) -> bool:
    pharmacy = pharmacy_for_user(user_id)
    if not pharmacy:
        return False
    with engine.connect() as connection:
        return connection.execute(select(orders.c.id).where(
            orders.c.analysis_id == analysis_id, orders.c.pharmacy_id == pharmacy["id"]
        ).limit(1)).scalar_one_or_none() is not None
