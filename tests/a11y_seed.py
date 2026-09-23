from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone

from simpliscribe.main import app as _app  # noqa: F401
from simpliscribe.marketplace import create_order
from simpliscribe.security import hash_password
from simpliscribe.storage import append_history, create_user, save_prescription_file


def seed_accessibility_fixture() -> dict[str, str]:
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    patient_id = str(uuid.uuid4())
    pharmacy_user_id = str(uuid.uuid4())
    pharmacy_id = str(uuid.uuid4())
    email = os.environ["A11Y_EMAIL"]
    password = os.environ["A11Y_PASSWORD"]

    create_user({
        "id": patient_id,
        "email": email,
        "password_hash": hash_password(password),
        "role": "patient",
        "full_name": "Synthetic Accessibility Patient",
        "phone": "9999999999",
        "pin_code": "400001",
        "active": True,
        "created_at": now,
    })
    create_user({
        "id": pharmacy_user_id,
        "email": "a11y-pharmacy-" + uuid.uuid4().hex[:12] + "@example.test",
        "password_hash": hash_password(uuid.uuid4().hex + "Test!"),
        "role": "pharmacy",
        "full_name": "Synthetic Accessibility Pharmacist",
        "phone": "9999999998",
        "pin_code": "400001",
        "active": True,
        "created_at": now,
    }, {
        "id": pharmacy_id,
        "user_id": pharmacy_user_id,
        "business_name": "Synthetic Accessibility Pharmacy",
        "license_number": "A11Y-" + uuid.uuid4().hex[:16],
        "address": "Synthetic test address",
        "pin_code": "400001",
        "serviceable_pins_json": '["400001"]',
        "supports_pickup": True,
        "supports_delivery": True,
        "approval_status": "approved",
        "approved_at": now,
    })

    medication = {
        "name": "Paracetamol 500",
        "type": "Tablet",
        "dosage": "500 mg",
        "frequency": "Once daily",
        "duration": "3 days",
    }
    analysis_ids: dict[str, str] = {}
    records = {
        "processing": {
            "prescription_state": "processing",
            "processing_stage": "ocr",
            "processing_started_at": now_iso,
            "patient_review_status": "needs_review",
            "medications": [],
        },
        "review": {
            "prescription_state": "review_required",
            "processing_stage": "review_ready",
            "patient_review_status": "needs_review",
            "medications": [medication],
            "original_medications": [medication],
        },
        "details": {
            "prescription_state": "confirmed",
            "patient_review_status": "confirmed",
            "medications": [medication],
        },
        "order": {
            "prescription_state": "confirmed",
            "patient_review_status": "confirmed",
            "medications": [medication],
        },
    }
    for state, payload in records.items():
        analysis_id = str(uuid.uuid4())
        analysis_ids[state] = analysis_id
        append_history({
            "id": analysis_id,
            "created_at": now_iso,
            "filename": "Synthetic accessibility " + state + " record.png",
            **payload,
        }, owner_id=patient_id)

    order_analysis_id = analysis_ids["order"]
    save_prescription_file({
        "id": str(uuid.uuid4()),
        "analysis_id": order_analysis_id,
        "owner_id": patient_id,
        "storage_name": "a11y-" + uuid.uuid4().hex + ".png",
        "original_name": "synthetic-accessibility-prescription.png",
        "content_type": "image/png",
        "sha256": "0" * 64,
        "created_at": now,
        "expires_at": now + timedelta(days=30),
    })
    order_id = create_order(patient_id, order_analysis_id, pharmacy_id, "pickup", "", [0])
    return {
        "A11Y_PROCESSING_ID": analysis_ids["processing"],
        "A11Y_REVIEW_ID": analysis_ids["review"],
        "A11Y_DETAILS_ID": analysis_ids["details"],
        "A11Y_ORDER_ID": order_id,
    }


if __name__ == "__main__":
    print(json.dumps(seed_accessibility_fixture()))
