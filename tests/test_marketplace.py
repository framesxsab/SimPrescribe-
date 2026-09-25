import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import delete, func, select, update
from fastapi.testclient import TestClient

from simpliscribe.main import _login_times, _marketplace_http_error, app
from simpliscribe.marketplace import (MarketplaceConflict, accept_order, cancel_order, create_order, decline_order, list_orders_for, matching_pharmacies, order_detail, pharmacy_can_access_analysis, pharmacy_transition, quote_order, save_inventory_item)
from simpliscribe.security import hash_password, verify_password
from simpliscribe.config import settings
from simpliscribe.storage import analyses, append_history, create_user, engine, get_analysis_record, inventory, order_events, orders, pharmacies, prescription_files, save_prescription_file, users


def unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def account(role: str, pin: str = "400001") -> dict:
    user_id = str(uuid.uuid4())
    user = {"id": user_id, "email": f"{unique(role)}@example.test", "password_hash": hash_password("correct horse battery"),
            "role": role, "full_name": role.title(), "phone": "9999999999", "pin_code": pin, "active": True,
            "created_at": datetime.now(timezone.utc)}
    pharmacy = None
    if role == "pharmacy":
        pharmacy = {"id": str(uuid.uuid4()), "user_id": user_id, "business_name": unique("Pharmacy"),
                    "license_number": unique("LIC"), "address": "Test address", "pin_code": pin,
                    "serviceable_pins_json": json.dumps(["400002"]), "supports_pickup": True,
                    "supports_delivery": True, "approval_status": "approved", "approved_at": datetime.now(timezone.utc)}
    create_user(user, pharmacy)
    return {**user, "pharmacy": pharmacy}


def prescription(
    patient_id: str,
    *,
    state: str | None = "confirmed",
    review_status: str = "confirmed",
    expires_in_days: int = 30,
) -> str:
    analysis_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    record = {
        "id": analysis_id,
        "created_at": now.isoformat(),
        "filename": "rx.png",
        "patient_review_status": review_status,
        "medications": [{"name": "Paracetamol 500", "dosage": "500 mg",
                         "frequency": "once daily", "duration": "3 days", "type": "Tablet"}],
    }
    if state is not None:
        record["prescription_state"] = state
    append_history(record, owner_id=patient_id)
    expiry = now + timedelta(days=expires_in_days)
    if expires_in_days < 0:
        with engine.begin() as connection:
            connection.execute(update(analyses).where(analyses.c.id == analysis_id).values(expires_at=expiry))
    save_prescription_file({
        "id": str(uuid.uuid4()),
        "analysis_id": analysis_id,
        "owner_id": patient_id,
        "storage_name": unique("rx") + ".png",
        "original_name": "rx.png",
        "content_type": "image/png",
        "sha256": "0" * 64,
        "created_at": now,
        "expires_at": expiry,
    })
    return analysis_id


def login_client(user: dict) -> TestClient:
    _login_times.clear()
    client = TestClient(app)
    page = client.get("/login")
    token = page.text.split('name="csrf" value="', 1)[1].split('"', 1)[0]
    response = client.post("/login", data={"csrf": token, "email": user["email"], "password": "correct horse battery"}, follow_redirects=False)
    assert response.status_code == 303
    return client


def test_marketplace_is_visible_and_links_to_patient_review():
    patient = account("patient")
    analysis_id = prescription(patient["id"])
    client = login_client(patient)
    page = client.get("/marketplace")
    assert page.status_code == 200
    assert "Medicine Marketplace" in page.text
    assert f"/details/{analysis_id}#medicine-marketplace" in page.text
    assert 'href="/marketplace"' in page.text


def test_passwords_use_scrypt_and_verify():
    encoded = hash_password("correct horse battery")
    assert encoded.startswith("scrypt$")
    assert "correct horse battery" not in encoded
    assert verify_password("correct horse battery", encoded)
    assert not verify_password("wrong password", encoded)


def test_patient_registration_and_login():
    _login_times.clear()
    client = TestClient(app)
    csrf = client.get("/register/patient").cookies.get("session")
    assert csrf
    # Obtain the token rendered into the form; registration must not bypass CSRF.
    page = client.get("/register/patient")
    token = page.text.split('name="csrf" value="', 1)[1].split('"', 1)[0]
    email = f"{unique('patient')}@example.test"
    response = client.post("/register/patient", data={"csrf": token, "full_name": "Test Patient", "email": email,
                           "phone": "9999999999", "pin_code": "400001", "password": "correct horse battery"})
    assert response.status_code == 201
    login = client.get("/login")
    token = login.text.split('name="csrf" value="', 1)[1].split('"', 1)[0]
    assert client.post("/login", data={"csrf": token, "email": email, "password": "correct horse battery"}, follow_redirects=False).status_code == 303


def test_exact_pin_marketplace_order_quote_and_stock_acceptance():
    patient = account("patient")
    pharmacy_user = account("pharmacy")
    item = save_inventory_item(pharmacy_user["id"], {"medicine_name": "Paracetamol 500", "unit_label": "strip",
                               "price_paise": 2500, "stock_quantity": 10, "active": True})
    analysis_id = prescription(patient["id"])
    matches = matching_pharmacies("400001", ["Paracetamol 500", "Unknown medicine"])
    match = next(row for row in matches if row["id"] == pharmacy_user["pharmacy"]["id"])
    assert match["exact_pin"] is True
    assert [row["status"] for row in match["availability"]] == ["in_stock", "pharmacist_check_required"]

    order_id = create_order(patient["id"], analysis_id, pharmacy_user["pharmacy"]["id"], "pickup", "", [0])
    detail = order_detail(order_id, {"id": pharmacy_user["id"], "role": "pharmacy"})
    assert detail["items"][0]["generic_inquiry"] is True
    quote_order(pharmacy_user["id"], order_id, [{"id": detail["items"][0]["id"], "availability": "available",
                "inventory_id": item["id"], "verified_quantity": 2, "unit_price_paise": 2500, "pharmacist_note": "Verified"}])
    accept_order(patient["id"], order_id)
    accepted = order_detail(order_id, {"id": patient["id"], "role": "patient"})
    assert accepted["status"] == "accepted"
    assert accepted["total_paise"] == 5000
    with engine.connect() as connection:
        assert connection.execute(inventory.select().where(inventory.c.id == item["id"])).mappings().one()["stock_quantity"] == 8


def test_order_rejects_unconfirmed_prescription_and_cross_patient_access():
    patient = account("patient")
    other = account("patient")
    pharmacy_user = account("pharmacy")
    analysis_id = str(uuid.uuid4())
    append_history({"id": analysis_id, "created_at": datetime.now(timezone.utc).isoformat(), "patient_review_status": "needs_review",
                    "medications": [{"name": "Medicine"}]}, owner_id=patient["id"])
    with pytest.raises(MarketplaceConflict, match="Confirm"):
        create_order(patient["id"], analysis_id, pharmacy_user["pharmacy"]["id"], "pickup", "", [])
    confirmed_id = prescription(patient["id"])
    order_id = create_order(patient["id"], confirmed_id, pharmacy_user["pharmacy"]["id"], "pickup", "", [])
    with pytest.raises(PermissionError):
        order_detail(order_id, {"id": other["id"], "role": "patient"})


def test_stock_change_returns_order_for_requote():
    patient = account("patient")
    pharmacy_user = account("pharmacy")
    item = save_inventory_item(pharmacy_user["id"], {"medicine_name": "Paracetamol 500", "unit_label": "strip",
                               "price_paise": 1000, "stock_quantity": 1, "active": True})
    order_id = create_order(patient["id"], prescription(patient["id"]), pharmacy_user["pharmacy"]["id"], "pickup", "", [])
    detail = order_detail(order_id, {"id": pharmacy_user["id"], "role": "pharmacy"})
    quote_order(pharmacy_user["id"], order_id, [{"id": detail["items"][0]["id"], "availability": "available",
                "inventory_id": item["id"], "verified_quantity": 1, "unit_price_paise": 1000}])
    with engine.begin() as connection:
        connection.execute(inventory.update().where(inventory.c.id == item["id"]).values(stock_quantity=0))
    with pytest.raises(MarketplaceConflict, match="Stock changed"):
        accept_order(patient["id"], order_id)
    assert order_detail(order_id, {"id": patient["id"], "role": "patient"})["status"] == "requested"


def test_patient_corrections_preserve_original_medication_version():
    patient = account("patient")
    analysis_id = str(uuid.uuid4())
    original = {"name": "Paracetmol", "type": "Tablet", "dosage": "500 mg", "frequency": "once daily", "duration": "3 days"}
    append_history({"id": analysis_id, "created_at": datetime.now(timezone.utc).isoformat(), "patient_review_status": "needs_review",
                    "patient_review_versions": [], "medications": [original]}, owner_id=patient["id"])
    client = login_client(patient)
    page = client.get(f"/details/{analysis_id}")
    token = page.text.split("'X-CSRF-Token':'", 1)[1].split("'", 1)[0]
    corrected = {**original, "name": "Paracetamol"}
    response = client.patch(f"/api/analyses/{analysis_id}/patient-review", headers={"X-CSRF-Token": token},
                            json={"status": "corrected", "medications": [corrected]})
    assert response.status_code == 200
    stored = get_analysis_record(analysis_id, patient["id"])
    assert stored["original_medications"][0]["name"] == "Paracetmol"
    assert stored["medications"][0]["name"] == "Paracetamol"
    assert stored["patient_review_versions"][0]["medications"][0]["name"] == "Paracetmol"


def test_patient_confirmation_is_owner_scoped_and_cannot_repeat():
    patient, other = account("patient"), account("patient")
    analysis_id = str(uuid.uuid4())
    original = {"name": "Paracetamol", "dosage": "500 mg", "frequency": "N/A", "duration": "3 days", "type": "Tablet"}
    append_history({"id": analysis_id, "created_at": datetime.now(timezone.utc).isoformat(),
                    "prescription_state": "review_required", "patient_review_status": "needs_review",
                    "medications": [original], "original_medications": [original]}, owner_id=patient["id"])
    other_client = login_client(other)
    other_token = other_client.get("/").text.split('name="csrf" value="', 1)[1].split('"', 1)[0]
    payload = {"status": "corrected", "medications": [{**original, "frequency": "once daily"}]}
    assert other_client.patch(f"/api/analyses/{analysis_id}/patient-review", headers={"X-CSRF-Token": other_token}, json=payload).status_code == 404
    assert other_client.get(f"/api/analyses/{analysis_id}/status").status_code == 404
    assert other_client.post(f"/api/analyses/{analysis_id}/process", headers={"X-CSRF-Token": other_token}).status_code == 404

    client = login_client(patient)
    page = client.get(f"/details/{analysis_id}")
    assert "Prescription review" in page.text
    token = page.text.split("'X-CSRF-Token':'", 1)[1].split("'", 1)[0]
    first = client.patch(f"/api/analyses/{analysis_id}/patient-review", headers={"X-CSRF-Token": token}, json=payload)
    assert first.status_code == 200
    assert client.patch(f"/api/analyses/{analysis_id}/patient-review", headers={"X-CSRF-Token": token}, json=payload).status_code == 409
    saved = get_analysis_record(analysis_id, patient["id"])
    assert saved["prescription_state"] == "confirmed"
    assert saved["original_medications"][0]["frequency"] == "N/A"
    assert saved["medications"][0]["frequency"] == "once daily"
    assert len(saved["patient_review_versions"]) == 1
    assert client.get(f"/details/{analysis_id}").status_code == 200


def test_patient_draft_survives_refresh_without_changing_original():
    patient = account("patient")
    analysis_id = str(uuid.uuid4())
    original = {"name": "Paracetamol", "dosage": "500 mg", "frequency": "N/A", "duration": "N/A", "type": "Tablet"}
    append_history({"id": analysis_id, "created_at": datetime.now(timezone.utc).isoformat(),
                    "prescription_state": "review_required", "patient_review_status": "needs_review",
                    "medications": [original], "original_medications": [original]}, owner_id=patient["id"])
    client = login_client(patient)
    page = client.get(f"/details/{analysis_id}")
    token = page.text.split("'X-CSRF-Token':'", 1)[1].split("'", 1)[0]
    changed = {**original, "frequency": "once daily"}
    response = client.patch(f"/api/analyses/{analysis_id}/patient-draft", headers={"X-CSRF-Token": token}, json={"medications": [changed]})
    assert response.status_code == 200
    assert 'value="once daily"' in client.get(f"/details/{analysis_id}").text
    stored = get_analysis_record(analysis_id, patient["id"])
    assert stored["original_medications"][0]["frequency"] == "N/A"
    assert stored["medications"][0]["frequency"] == "N/A"
    assert stored["patient_draft_medications"][0]["frequency"] == "once daily"
    assert len(stored["patient_draft_versions"]) == 1


def test_stale_processing_refresh_becomes_recoverable_without_new_analysis():
    patient = account("patient")
    analysis_id = str(uuid.uuid4())
    append_history({"id": analysis_id, "created_at": datetime.now(timezone.utc).isoformat(),
                    "processing_started_at": (datetime.now(timezone.utc) - timedelta(minutes=6)).isoformat(),
                    "prescription_state": "processing", "processing_stage": "ocr", "medications": []}, owner_id=patient["id"])
    client = login_client(patient)
    first = client.get(f"/details/{analysis_id}")
    second = client.get(f"/details/{analysis_id}")
    assert first.status_code == second.status_code == 200
    assert "Try reading again" in second.text
    stored = get_analysis_record(analysis_id, patient["id"])
    assert stored["prescription_state"] == "processing_failed"
    assert stored["error_code"] == "PROCESSING_INTERRUPTED"


def test_protected_source_is_owner_scoped():
    patient, other = account("patient"), account("patient")
    analysis_id = str(uuid.uuid4())
    append_history({"id": analysis_id, "created_at": datetime.now(timezone.utc).isoformat(), "medications": []}, owner_id=patient["id"])
    settings.prescription_storage_dir.mkdir(parents=True, exist_ok=True)
    storage_name = unique("protected") + ".png"
    path = settings.prescription_storage_dir / storage_name
    path.write_bytes(b"private-prescription")
    save_prescription_file({"id": str(uuid.uuid4()), "analysis_id": analysis_id, "owner_id": patient["id"],
                            "storage_name": storage_name, "original_name": "rx.png", "content_type": "image/png",
                            "sha256": "0" * 64, "created_at": datetime.now(timezone.utc),
                            "expires_at": datetime.now(timezone.utc) + timedelta(days=30)})
    try:
        assert login_client(patient).get(f"/api/analyses/{analysis_id}/source").status_code == 200
        assert login_client(other).get(f"/api/analyses/{analysis_id}/source").status_code == 403
    finally:
        path.unlink(missing_ok=True)


def test_marketplace_requires_explicit_confirmed_current_prescription():
    patient = account("patient")
    pharmacy_user = account("pharmacy")
    cases = [
        (None, "confirmed", 30),
        ("processing", "needs_review", 30),
        ("review_required", "needs_review", 30),
        ("processing_failed", "needs_review", 30),
        ("confirmed", "needs_review", 30),
        ("confirmed", "confirmed", -1),
    ]
    for state, review_status, expiry in cases:
        analysis_id = prescription(
            patient["id"],
            state=state,
            review_status=review_status,
            expires_in_days=expiry,
        )
        with pytest.raises(MarketplaceConflict):
            create_order(patient["id"], analysis_id, pharmacy_user["pharmacy"]["id"], "pickup", "", [])


def test_only_approved_active_pharmacies_appear_and_manage_inventory():
    patient = account("patient")
    analysis_id = prescription(patient["id"])
    approved = account("pharmacy")
    pending = account("pharmacy")
    rejected = account("pharmacy")
    inactive = account("pharmacy")
    with engine.begin() as connection:
        connection.execute(update(pharmacies).where(pharmacies.c.id == pending["pharmacy"]["id"]).values(approval_status="pending"))
        connection.execute(update(pharmacies).where(pharmacies.c.id == rejected["pharmacy"]["id"]).values(approval_status="rejected"))
        connection.execute(update(users).where(users.c.id == inactive["id"]).values(active=False))
    discovered = {row["id"] for row in matching_pharmacies("400001", ["Paracetamol 500"])}
    assert approved["pharmacy"]["id"] in discovered
    assert pending["pharmacy"]["id"] not in discovered
    assert rejected["pharmacy"]["id"] not in discovered
    assert inactive["pharmacy"]["id"] not in discovered
    for account_to_reject in (pending, rejected, inactive):
        with pytest.raises((PermissionError, LookupError)):
            save_inventory_item(account_to_reject["id"], {
                "medicine_name": "Paracetamol 500",
                "price_paise": 100,
                "stock_quantity": 1,
                "active": True,
            })
        with pytest.raises((PermissionError, LookupError)):
            create_order(patient["id"], analysis_id, account_to_reject["pharmacy"]["id"], "pickup", "", [])
    save_inventory_item(approved["id"], {
        "medicine_name": "Paracetamol 500",
        "price_paise": 100,
        "stock_quantity": 1,
        "active": True,
    })


def test_pin_matching_ranks_exact_then_coverage_and_hides_private_pharmacy_fields():
    patient = account("patient", pin="400001")
    exact_empty = account("pharmacy", pin="400001")
    exact_stock = account("pharmacy", pin="400001")
    exact_tie = account("pharmacy", pin="400001")
    serviceable = account("pharmacy", pin="400003")
    outside = account("pharmacy", pin="400004")
    with engine.begin() as connection:
        connection.execute(update(pharmacies).where(pharmacies.c.id == exact_empty["pharmacy"]["id"]).values(business_name="A Exact Empty"))
        connection.execute(update(pharmacies).where(pharmacies.c.id == exact_stock["pharmacy"]["id"]).values(business_name="Z Exact Stock"))
        connection.execute(update(pharmacies).where(pharmacies.c.id == exact_tie["pharmacy"]["id"]).values(business_name="B Exact Empty"))
        connection.execute(update(pharmacies).where(pharmacies.c.id == serviceable["pharmacy"]["id"]).values(
            business_name="A Serviceable", serviceable_pins_json='["400001"]'
        ))
        connection.execute(update(pharmacies).where(pharmacies.c.id == outside["pharmacy"]["id"]).values(business_name="Outside"))
    save_inventory_item(exact_stock["id"], {
        "medicine_name": "Paracetamol 500", "price_paise": 125, "stock_quantity": 9, "active": True,
    })
    matches = matching_pharmacies("400001", ["Paracetamol 500"])
    ids = [row["id"] for row in matches]
    expected_ids = {
        exact_stock["pharmacy"]["id"], exact_empty["pharmacy"]["id"],
        exact_tie["pharmacy"]["id"], serviceable["pharmacy"]["id"],
    }
    scoped = [row for row in matches if row["id"] in expected_ids]
    assert [row["id"] for row in scoped] == [
        exact_stock["pharmacy"]["id"], exact_empty["pharmacy"]["id"],
        exact_tie["pharmacy"]["id"], serviceable["pharmacy"]["id"],
    ]
    assert outside["pharmacy"]["id"] not in ids
    assert scoped[0]["exact_pin"] is True
    assert scoped[-1]["exact_pin"] is False
    assert all("user_id" not in row and "license_number" not in row for row in matches)
    assert scoped[1]["availability"][0]["status"] == "pharmacist_check_required"
    assert scoped[2]["availability"][0]["status"] == "pharmacist_check_required"
    analysis_id = prescription(patient["id"])
    with pytest.raises(MarketplaceConflict, match="PIN"):
        create_order(patient["id"], analysis_id, outside["pharmacy"]["id"], "pickup", "", [])
    assert create_order(patient["id"], analysis_id, serviceable["pharmacy"]["id"], "pickup", "", [])


def test_inventory_matching_is_exact_and_excludes_inactive_stock():
    patient = account("patient")
    pharmacy_user = account("pharmacy")
    save_inventory_item(pharmacy_user["id"], {
        "medicine_name": "Paracetamol 500 mg", "price_paise": 100, "stock_quantity": 10, "active": True,
    })
    save_inventory_item(pharmacy_user["id"], {
        "medicine_name": "Paracetamol 500", "price_paise": 100, "stock_quantity": 10, "active": False,
    })
    match = next(
        row for row in matching_pharmacies("400001", ["Paracetamol 500"])
        if row["id"] == pharmacy_user["pharmacy"]["id"]
    )
    assert match["availability"][0]["status"] == "pharmacist_check_required"
    assert match["availability"][0]["inventory_id"] is None


def test_quote_request_is_idempotent_and_creates_one_order_event():
    patient = account("patient")
    pharmacy_user = account("pharmacy")
    analysis_id = prescription(patient["id"])
    first_id, first_created = create_order(
        patient["id"], analysis_id, pharmacy_user["pharmacy"]["id"], "pickup", "", [], return_created=True,
    )
    retry_id, retry_created = create_order(
        patient["id"], analysis_id, pharmacy_user["pharmacy"]["id"], "pickup", "", [], return_created=True,
    )
    assert (retry_id, retry_created) == (first_id, False)
    with engine.connect() as connection:
        assert connection.execute(select(func.count()).select_from(orders).where(orders.c.analysis_id == analysis_id)).scalar_one() == 1
        assert connection.execute(select(func.count()).select_from(order_events).where(order_events.c.order_id == first_id)).scalar_one() == 1


def test_competing_duplicate_order_submissions_create_one_request_and_event():
    patient = account("patient")
    pharmacist = account("pharmacy")
    analysis_id = prescription(patient["id"])
    gate = Barrier(2)

    def submit():
        gate.wait(timeout=10)
        return create_order(
            patient["id"],
            analysis_id,
            pharmacist["pharmacy"]["id"],
            "pickup",
            "",
            [],
            return_created=True,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: submit(), range(2)))
    assert results[0][0] == results[1][0]
    assert sorted(created for _, created in results) == [False, True]
    with engine.connect() as connection:
        assert connection.execute(select(func.count()).select_from(orders).where(
            orders.c.analysis_id == analysis_id
        )).scalar_one() == 1
        assert connection.execute(select(func.count()).select_from(order_events).where(
            order_events.c.order_id == results[0][0], order_events.c.status == "requested"
        )).scalar_one() == 1


def test_quote_validation_exact_inventory_integer_price_and_server_total():
    patient = account("patient")
    pharmacist = account("pharmacy")
    other_pharmacist = account("pharmacy")
    item = save_inventory_item(pharmacist["id"], {
        "medicine_name": "Paracetamol 500", "price_paise": 777, "stock_quantity": 4, "active": True,
    })
    foreign_item = save_inventory_item(other_pharmacist["id"], {
        "medicine_name": "Paracetamol 500", "price_paise": 1, "stock_quantity": 100, "active": True,
    })
    order_id = create_order(patient["id"], prescription(patient["id"]), pharmacist["pharmacy"]["id"], "pickup", "", [])
    line_id = order_detail(order_id, {"id": pharmacist["id"], "role": "pharmacy"})["items"][0]["id"]
    for invalid_line in (
        {"id": line_id, "availability": "available", "inventory_id": item["id"], "verified_quantity": 0, "unit_price_paise": 777},
        {"id": line_id, "availability": "available", "inventory_id": item["id"], "verified_quantity": -1, "unit_price_paise": 777},
        {"id": line_id, "availability": "available", "inventory_id": item["id"], "verified_quantity": 1.5, "unit_price_paise": 777},
        {"id": line_id, "availability": "available", "inventory_id": foreign_item["id"], "verified_quantity": 1, "unit_price_paise": 777},
        {"id": line_id, "availability": "available", "inventory_id": item["id"], "verified_quantity": 1, "unit_price_paise": -1},
    ):
        with pytest.raises((ValueError, MarketplaceConflict)):
            quote_order(pharmacist["id"], order_id, [invalid_line])
    with pytest.raises(ValueError, match="at least one available"):
        quote_order(pharmacist["id"], order_id, [{
            "id": line_id,
            "availability": "unavailable",
            "pharmacist_note": "Temporarily unavailable",
        }])
    quote_order(pharmacist["id"], order_id, [{
        "id": line_id,
        "availability": "available",
        "inventory_id": item["id"],
        "verified_quantity": 2,
        "unit_price_paise": 777,
        "pharmacist_note": "Verified",
        "client_total_paise": 1,
    }])
    quoted = order_detail(order_id, {"id": patient["id"], "role": "patient"})
    assert quoted["total_paise"] == 1554
    with engine.connect() as connection:
        assert connection.execute(select(func.count()).select_from(order_events).where(
            order_events.c.order_id == order_id, order_events.c.status == "quoted"
        )).scalar_one() == 1
    with pytest.raises(MarketplaceConflict, match="Only requested"):
        quote_order(pharmacist["id"], order_id, [{
            "id": line_id,
            "availability": "available",
            "inventory_id": item["id"],
            "verified_quantity": 2,
            "unit_price_paise": 777,
        }])


def test_quote_schema_rejects_fractional_quantities_and_line_total_manipulation():
    from pydantic import ValidationError

    from simpliscribe.schemas import QuoteLineRequest

    with pytest.raises(ValidationError):
        QuoteLineRequest.model_validate({
            "id": "line",
            "availability": "available",
            "inventory_id": "inventory",
            "verified_quantity": 1.5,
            "unit_price_paise": 100,
        })
    with pytest.raises(ValidationError):
        QuoteLineRequest.model_validate({
            "id": "line",
            "availability": "available",
            "inventory_id": "inventory",
            "verified_quantity": 1,
            "unit_price_paise": 100,
            "total_paise": 1,
        })


def test_pharmacist_source_is_private_until_owned_active_request_exists():
    original_auth = settings.auth_required
    object.__setattr__(settings, "auth_required", True)
    patient = account("patient")
    other_patient = account("patient")
    pharmacist = account("pharmacy")
    other_pharmacist = account("pharmacy")
    analysis_id = prescription(patient["id"])
    storage_name = unique("source") + ".png"
    source_path = settings.prescription_storage_dir / storage_name
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"synthetic-prescription-source")
    now = datetime.now(timezone.utc)
    with engine.begin() as connection:
        connection.execute(update(prescription_files).where(
            prescription_files.c.analysis_id == analysis_id
        ).values(storage_name=storage_name, expires_at=now + timedelta(days=30)))
    try:
        assert login_client(pharmacist).get(f"/api/analyses/{analysis_id}/source").status_code == 403
        assert TestClient(app).get(f"/api/analyses/{analysis_id}/source").status_code == 401
        order_id = create_order(patient["id"], analysis_id, pharmacist["pharmacy"]["id"], "pickup", "", [])
        assert pharmacy_can_access_analysis(pharmacist["id"], analysis_id) is True
        assert pharmacy_can_access_analysis(other_pharmacist["id"], analysis_id) is False
        assert login_client(pharmacist).get(f"/api/analyses/{analysis_id}/source").status_code == 200
        assert login_client(other_pharmacist).get(f"/api/analyses/{analysis_id}/source").status_code == 403
        assert login_client(other_patient).get(f"/api/analyses/{analysis_id}/source").status_code == 403
        assert login_client(patient).get(f"/api/analyses/{analysis_id}/source").status_code == 200
        assert order_detail(order_id, {"id": pharmacist["id"], "role": "pharmacy"})["id"] == order_id
    finally:
        source_path.unlink(missing_ok=True)
        object.__setattr__(settings, "auth_required", original_auth)


def test_order_state_machine_permissions_fulfillment_and_cancel_refund():
    patient = account("patient")
    pharmacist = account("pharmacy")
    other_pharmacist = account("pharmacy")
    item = save_inventory_item(pharmacist["id"], {
        "medicine_name": "Paracetamol 500", "price_paise": 500, "stock_quantity": 3, "active": True,
    })
    order_id = create_order(patient["id"], prescription(patient["id"]), pharmacist["pharmacy"]["id"], "pickup", "", [])
    line_id = order_detail(order_id, {"id": pharmacist["id"], "role": "pharmacy"})["items"][0]["id"]
    with pytest.raises(MarketplaceConflict):
        pharmacy_transition(pharmacist["id"], order_id, "fulfilled")
    with pytest.raises(PermissionError):
        quote_order(other_pharmacist["id"], order_id, [{
            "id": line_id, "availability": "available", "inventory_id": item["id"],
            "verified_quantity": 1, "unit_price_paise": 500,
        }])
    quote_order(pharmacist["id"], order_id, [{
        "id": line_id, "availability": "available", "inventory_id": item["id"],
        "verified_quantity": 1, "unit_price_paise": 500,
    }])
    with pytest.raises(MarketplaceConflict):
        pharmacy_transition(pharmacist["id"], order_id, "preparing")
    accept_order(patient["id"], order_id)
    with pytest.raises(MarketplaceConflict, match="Only quoted"):
        accept_order(patient["id"], order_id)
    with pytest.raises(PermissionError):
        order_detail(order_id, {"id": other_pharmacist["id"], "role": "pharmacy"})
    with pytest.raises(PermissionError):
        pharmacy_transition(other_pharmacist["id"], order_id, "preparing")
    pharmacy_transition(pharmacist["id"], order_id, "preparing")
    with pytest.raises(MarketplaceConflict, match="Pickup orders"):
        pharmacy_transition(pharmacist["id"], order_id, "out_for_delivery")
    with pytest.raises(MarketplaceConflict):
        cancel_order(patient["id"], order_id)
    pharmacy_transition(pharmacist["id"], order_id, "cancelled")
    with pytest.raises(MarketplaceConflict):
        pharmacy_transition(pharmacist["id"], order_id, "cancelled")
    with engine.connect() as connection:
        assert connection.execute(select(inventory.c.stock_quantity).where(inventory.c.id == item["id"])).scalar_one() == 3
        assert connection.execute(select(func.count()).select_from(order_events).where(
            order_events.c.order_id == order_id, order_events.c.status == "cancelled"
        )).scalar_one() == 1


def test_delivery_state_path_requires_address_and_blocks_pickup_states():
    patient = account("patient")
    pharmacist = account("pharmacy")
    item = save_inventory_item(pharmacist["id"], {
        "medicine_name": "Paracetamol 500", "price_paise": 100, "stock_quantity": 2, "active": True,
    })
    analysis_id = prescription(patient["id"])
    with pytest.raises(ValueError, match="address"):
        create_order(patient["id"], analysis_id, pharmacist["pharmacy"]["id"], "delivery", "", [])
    order_id = create_order(patient["id"], analysis_id, pharmacist["pharmacy"]["id"], "delivery", "12 Test Street", [])
    line_id = order_detail(order_id, {"id": pharmacist["id"], "role": "pharmacy"})["items"][0]["id"]
    quote_order(pharmacist["id"], order_id, [{
        "id": line_id, "availability": "available", "inventory_id": item["id"],
        "verified_quantity": 1, "unit_price_paise": 100,
    }])
    accept_order(patient["id"], order_id)
    pharmacy_transition(pharmacist["id"], order_id, "preparing")
    with pytest.raises(MarketplaceConflict, match="Delivery orders"):
        pharmacy_transition(pharmacist["id"], order_id, "ready_for_pickup")
    pharmacy_transition(pharmacist["id"], order_id, "out_for_delivery")
    pharmacy_transition(pharmacist["id"], order_id, "fulfilled")
    assert order_detail(order_id, {"id": patient["id"], "role": "patient"})["status"] == "fulfilled"


def test_two_competing_orders_cannot_oversell_last_stock():
    patients = (account("patient"), account("patient"))
    pharmacist = account("pharmacy")
    item = save_inventory_item(pharmacist["id"], {
        "medicine_name": "Paracetamol 500", "price_paise": 300, "stock_quantity": 1, "active": True,
    })
    order_ids = []
    for patient in patients:
        order_id = create_order(patient["id"], prescription(patient["id"]), pharmacist["pharmacy"]["id"], "pickup", "", [])
        line_id = order_detail(order_id, {"id": pharmacist["id"], "role": "pharmacy"})["items"][0]["id"]
        quote_order(pharmacist["id"], order_id, [{
            "id": line_id, "availability": "available", "inventory_id": item["id"],
            "verified_quantity": 1, "unit_price_paise": 300,
        }])
        order_ids.append(order_id)

    gate = Barrier(2)

    def accept_after_barrier(patient_id, order_id):
        gate.wait(timeout=10)
        try:
            accept_order(patient_id, order_id)
            return "accepted"
        except MarketplaceConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(
            lambda args: accept_after_barrier(*args),
            [(patients[0]["id"], order_ids[0]), (patients[1]["id"], order_ids[1])],
        ))
    assert sorted(outcomes) == ["accepted", "conflict"]
    with engine.connect() as connection:
        assert connection.execute(select(inventory.c.stock_quantity).where(inventory.c.id == item["id"])).scalar_one() == 0
        statuses = connection.execute(select(orders.c.status).where(orders.c.id.in_(order_ids))).scalars().all()
    assert statuses.count("accepted") == 1
    assert statuses.count("requested") == 1
    with engine.connect() as connection:
        assert connection.execute(select(func.count()).select_from(order_events).where(
            order_events.c.order_id.in_(order_ids), order_events.c.status == "accepted"
        )).scalar_one() == 1


def test_order_history_includes_pharmacy_medicines_and_isolated_queues():
    patient = account("patient")
    other_patient = account("patient")
    pharmacist = account("pharmacy")
    other_pharmacist = account("pharmacy")
    first = create_order(patient["id"], prescription(patient["id"]), pharmacist["pharmacy"]["id"], "pickup", "", [])
    other = create_order(other_patient["id"], prescription(other_patient["id"]), other_pharmacist["pharmacy"]["id"], "pickup", "", [])
    patient_orders = list_orders_for({"id": patient["id"], "role": "patient"})
    pharmacy_orders = list_orders_for({"id": pharmacist["id"], "role": "pharmacy"})
    assert [row["id"] for row in patient_orders] == [first]
    assert patient_orders[0]["pharmacy_name"] == pharmacist["pharmacy"]["business_name"]
    assert patient_orders[0]["medicine_names"] == ["Paracetamol 500"]
    assert [row["id"] for row in pharmacy_orders] == [first]
    assert other not in {row["id"] for row in pharmacy_orders}



def test_matching_reports_in_stock_low_stock_unavailable_and_check_required():
    pharmacy_user = account("pharmacy")
    inventory_payloads = [
        ("Healthy med", 6, True),
        ("Low stock med", 5, True),
        ("Unavailable med", 0, True),
        ("Check required med", 7, False),
    ]
    for name, stock, active in inventory_payloads:
        save_inventory_item(pharmacy_user["id"], {
            "medicine_name": name,
            "price_paise": 100,
            "stock_quantity": stock,
            "active": active,
        })
    match = next(
        row for row in matching_pharmacies("400001", [name for name, _, _ in inventory_payloads])
        if row["id"] == pharmacy_user["pharmacy"]["id"]
    )
    assert [item["status"] for item in match["availability"]] == [
        "in_stock", "low_stock", "unavailable", "pharmacist_check_required",
    ]


def test_declined_order_is_terminal_and_cannot_be_quoted_or_fulfilled():
    patient = account("patient")
    pharmacist = account("pharmacy")
    order_id = create_order(patient["id"], prescription(patient["id"]), pharmacist["pharmacy"]["id"], "pickup", "", [])
    detail = order_detail(order_id, {"id": pharmacist["id"], "role": "pharmacy"})
    decline_order(pharmacist["id"], order_id, "Synthetic test decline")
    with pytest.raises(MarketplaceConflict):
        quote_order(pharmacist["id"], order_id, [{
            "id": detail["items"][0]["id"], "availability": "available",
            "inventory_id": "missing", "verified_quantity": 1, "unit_price_paise": 100,
        }])
    with pytest.raises(MarketplaceConflict):
        pharmacy_transition(pharmacist["id"], order_id, "preparing")
    with pytest.raises(MarketplaceConflict):
        accept_order(patient["id"], order_id)
    with engine.connect() as connection:
        assert connection.execute(select(func.count()).select_from(order_events).where(
            order_events.c.order_id == order_id, order_events.c.status == "declined"
        )).scalar_one() == 1


def test_missing_or_expired_source_blocks_order_quote_and_acceptance():
    patient = account("patient")
    pharmacist = account("pharmacy")
    item = save_inventory_item(pharmacist["id"], {
        "medicine_name": "Paracetamol 500", "price_paise": 250, "stock_quantity": 2, "active": True,
    })
    missing_analysis = prescription(patient["id"])
    with engine.begin() as connection:
        connection.execute(delete(prescription_files).where(prescription_files.c.analysis_id == missing_analysis))
    with pytest.raises(MarketplaceConflict, match="original prescription"):
        create_order(patient["id"], missing_analysis, pharmacist["pharmacy"]["id"], "pickup", "", [])

    analysis_id = prescription(patient["id"])
    order_id = create_order(patient["id"], analysis_id, pharmacist["pharmacy"]["id"], "pickup", "", [])
    line_id = order_detail(order_id, {"id": pharmacist["id"], "role": "pharmacy"})["items"][0]["id"]
    with engine.begin() as connection:
        connection.execute(update(prescription_files).where(
            prescription_files.c.analysis_id == analysis_id
        ).values(expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)))
    with pytest.raises(MarketplaceConflict, match="original prescription"):
        quote_order(pharmacist["id"], order_id, [{
            "id": line_id, "availability": "available", "inventory_id": item["id"],
            "verified_quantity": 1, "unit_price_paise": 250,
        }])
    with engine.begin() as connection:
        connection.execute(update(prescription_files).where(
            prescription_files.c.analysis_id == analysis_id
        ).values(expires_at=datetime.now(timezone.utc) + timedelta(days=30)))
    quote_order(pharmacist["id"], order_id, [{
        "id": line_id, "availability": "available", "inventory_id": item["id"],
        "verified_quantity": 1, "unit_price_paise": 250,
    }])
    with engine.begin() as connection:
        connection.execute(update(prescription_files).where(
            prescription_files.c.analysis_id == analysis_id
        ).values(expires_at=datetime.now(timezone.utc) - timedelta(seconds=1)))
    with pytest.raises(MarketplaceConflict, match="original prescription"):
        accept_order(patient["id"], order_id)
    assert order_detail(order_id, {"id": patient["id"], "role": "patient"})["status"] == "quoted"
    with engine.connect() as connection:
        assert connection.execute(select(inventory.c.stock_quantity).where(inventory.c.id == item["id"])).scalar_one() == 2


def test_deactivated_inventory_after_quote_requires_requote_without_decrement():
    patient = account("patient")
    pharmacist = account("pharmacy")
    item = save_inventory_item(pharmacist["id"], {
        "medicine_name": "Paracetamol 500", "price_paise": 300, "stock_quantity": 2, "active": True,
    })
    order_id = create_order(patient["id"], prescription(patient["id"]), pharmacist["pharmacy"]["id"], "pickup", "", [])
    line_id = order_detail(order_id, {"id": pharmacist["id"], "role": "pharmacy"})["items"][0]["id"]
    quote_order(pharmacist["id"], order_id, [{
        "id": line_id, "availability": "available", "inventory_id": item["id"],
        "verified_quantity": 1, "unit_price_paise": 300,
    }])
    with engine.begin() as connection:
        connection.execute(update(inventory).where(inventory.c.id == item["id"]).values(active=False))
    with pytest.raises(MarketplaceConflict, match="Stock changed"):
        accept_order(patient["id"], order_id)
    assert order_detail(order_id, {"id": patient["id"], "role": "patient"})["status"] == "requested"
    with engine.connect() as connection:
        assert connection.execute(select(inventory.c.stock_quantity).where(inventory.c.id == item["id"])).scalar_one() == 2


def test_corrupt_persisted_marketplace_data_does_not_leak_parser_diagnostics():
    import json

    response = _marketplace_http_error(json.JSONDecodeError("invalid value", "{", 0))
    assert response.status_code == 500
    assert response.detail == "The marketplace request could not be completed."
    assert "invalid value" not in response.detail


def test_pharmacy_approval_is_rechecked_at_patient_acceptance():
    patient = account("patient")
    pharmacist = account("pharmacy")
    item = save_inventory_item(pharmacist["id"], {
        "medicine_name": "Paracetamol 500", "price_paise": 500, "stock_quantity": 1, "active": True,
    })
    order_id = create_order(patient["id"], prescription(patient["id"]), pharmacist["pharmacy"]["id"], "pickup", "", [])
    line_id = order_detail(order_id, {"id": pharmacist["id"], "role": "pharmacy"})["items"][0]["id"]
    quote_order(pharmacist["id"], order_id, [{
        "id": line_id, "availability": "available", "inventory_id": item["id"],
        "verified_quantity": 1, "unit_price_paise": 500,
    }])
    with engine.begin() as connection:
        connection.execute(update(pharmacies).where(
            pharmacies.c.id == pharmacist["pharmacy"]["id"]
        ).values(approval_status="pending"))
    with pytest.raises(MarketplaceConflict, match="no longer eligible"):
        accept_order(patient["id"], order_id)
    assert order_detail(order_id, {"id": patient["id"], "role": "patient"})["status"] == "quoted"
    with engine.connect() as connection:
        assert connection.execute(select(inventory.c.stock_quantity).where(inventory.c.id == item["id"])).scalar_one() == 1
