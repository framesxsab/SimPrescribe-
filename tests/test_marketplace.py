import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from simpliscribe.main import _login_times, app
from simpliscribe.marketplace import MarketplaceConflict, accept_order, create_order, matching_pharmacies, order_detail, quote_order, save_inventory_item
from simpliscribe.security import hash_password, verify_password
from simpliscribe.config import settings
from simpliscribe.storage import append_history, create_user, engine, get_analysis_record, inventory, save_prescription_file


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


def prescription(patient_id: str) -> str:
    analysis_id = str(uuid.uuid4())
    append_history({"id": analysis_id, "created_at": datetime.now(timezone.utc).isoformat(), "filename": "rx.png",
                    "patient_review_status": "confirmed", "medications": [{"name": "Paracetamol 500", "dosage": "500 mg",
                    "frequency": "once daily", "duration": "3 days", "type": "Tablet"}]}, owner_id=patient_id)
    save_prescription_file({"id": str(uuid.uuid4()), "analysis_id": analysis_id, "owner_id": patient_id,
                            "storage_name": unique("rx") + ".png", "original_name": "rx.png", "content_type": "image/png",
                            "sha256": "0" * 64, "created_at": datetime.now(timezone.utc),
                            "expires_at": datetime.now(timezone.utc) + timedelta(days=30)})
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
    with pytest.raises(ValueError, match="Confirm"):
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
