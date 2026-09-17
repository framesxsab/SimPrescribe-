"""Create one approved local demo pharmacy and exact-name inventory."""
import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from simpliscribe.marketplace import save_inventory_item, valid_pin
from simpliscribe.security import hash_password
from simpliscribe.storage import create_user, ensure_schema, get_user_by_email


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", default="pharmacy@example.test")
    parser.add_argument("--password", required=True)
    parser.add_argument("--pin", default="400001")
    args = parser.parse_args()
    if not valid_pin(args.pin):
        parser.error("--pin must be a six-digit Indian PIN code")
    ensure_schema()
    if get_user_by_email(args.email):
        parser.error("that email is already registered")
    user_id, pharmacy_id = str(uuid.uuid4()), str(uuid.uuid4())
    create_user(
        {"id": user_id, "email": args.email.lower(), "password_hash": hash_password(args.password), "role": "pharmacy",
         "full_name": "Demo Pharmacist", "phone": "9999999999", "pin_code": args.pin, "active": True,
         "created_at": datetime.now(timezone.utc)},
        {"id": pharmacy_id, "user_id": user_id, "business_name": "SimpliScribe Demo Pharmacy",
         "license_number": f"DEMO-{uuid.uuid4().hex[:8].upper()}", "address": "Demo local pharmacy",
         "pin_code": args.pin, "serviceable_pins_json": "[]", "supports_pickup": True,
         "supports_delivery": True, "approval_status": "approved", "approved_at": datetime.now(timezone.utc)},
    )
    for medicine, price, stock in (("Paracetamol 500", 2500, 20), ("Amoxycillin 500", 8000, 10)):
        save_inventory_item(user_id, {"medicine_name": medicine, "unit_label": "strip", "price_paise": price,
                                      "stock_quantity": stock, "active": True})
    print(f"Created approved demo pharmacy {args.email} serving PIN {args.pin}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
