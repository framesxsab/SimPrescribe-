from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tests.a11y_app import app as _app  # noqa: F401

from simpliscribe.marketplace import save_inventory_item
from simpliscribe.security import hash_password
from simpliscribe.storage import append_history, create_user, save_prescription_file


def seed_marketplace_fixture() -> dict[str, str]:
    now = datetime.now(timezone.utc)
    patient_id = str(uuid.uuid4())
    pharmacist_id = str(uuid.uuid4())
    pharmacy_id = str(uuid.uuid4())
    patient_email = os.environ["MARKETPLACE_PATIENT_EMAIL"]
    patient_password = os.environ["MARKETPLACE_PATIENT_PASSWORD"]
    pharmacy_email = os.environ["MARKETPLACE_PHARMACY_EMAIL"]
    pharmacy_password = os.environ["MARKETPLACE_PHARMACY_PASSWORD"]

    create_user({
        "id": patient_id,
        "email": patient_email,
        "password_hash": hash_password(patient_password),
        "role": "patient",
        "full_name": "Synthetic Marketplace Patient",
        "phone": "9999999911",
        "pin_code": "400001",
        "active": True,
        "created_at": now,
    })
    create_user({
        "id": pharmacist_id,
        "email": pharmacy_email,
        "password_hash": hash_password(pharmacy_password),
        "role": "pharmacy",
        "full_name": "Synthetic Marketplace Pharmacist",
        "phone": "9999999912",
        "pin_code": "400001",
        "active": True,
        "created_at": now,
    }, {
        "id": pharmacy_id,
        "user_id": pharmacist_id,
        "business_name": "Synthetic Marketplace Pharmacy",
        "license_number": "SYNTH-" + uuid.uuid4().hex[:12],
        "address": "Synthetic test address",
        "pin_code": "400001",
        "serviceable_pins_json": '["400001"]',
        "supports_pickup": True,
        "supports_delivery": True,
        "approval_status": "approved",
        "approved_at": now,
    })
    save_inventory_item(pharmacist_id, {
        "medicine_name": "Paracetamol 500",
        "unit_label": "strip",
        "price_paise": 617,
        "stock_quantity": 3,
        "active": True,
    })

    analysis_id = str(uuid.uuid4())
    medication = {
        "name": "Paracetamol 500",
        "type": "Tablet",
        "dosage": "500 mg",
        "frequency": "Once daily",
        "duration": "3 days",
    }
    append_history({
        "id": analysis_id,
        "created_at": now.isoformat(),
        "filename": "Synthetic confirmed prescription.png",
        "prescription_state": "confirmed",
        "patient_review_status": "confirmed",
        "patient_confirmed_at": now.isoformat(),
        "medications": [medication],
    }, owner_id=patient_id)

    storage_name = "synthetic-marketplace-" + uuid.uuid4().hex + ".png"
    storage_dir = Path(os.environ["PRESCRIPTION_STORAGE_DIR"])
    storage_dir.mkdir(parents=True, exist_ok=True)
    (storage_dir / storage_name).write_bytes(b"synthetic-prescription-source")
    save_prescription_file({
        "id": str(uuid.uuid4()),
        "analysis_id": analysis_id,
        "owner_id": patient_id,
        "storage_name": storage_name,
        "original_name": "synthetic-prescription.png",
        "content_type": "image/png",
        "sha256": "0" * 64,
        "created_at": now,
        "expires_at": now + timedelta(days=30),
    })
    return {
        "MARKETPLACE_ANALYSIS_ID": analysis_id,
        "MARKETPLACE_PHARMACY_ID": pharmacy_id,
    }


if __name__ == "__main__":
    print(json.dumps(seed_marketplace_fixture()))
