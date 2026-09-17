"""Native accounts and prescription marketplace.

Revision ID: 0002_marketplace
Revises: 0001_initial
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_marketplace"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "vector_cache",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("text_hash", sa.String(64), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("vector_json", sa.Text(), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("hit_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_vector_cache_text_hash", "vector_cache", ["text_hash"])
    op.create_index("ix_vector_cache_created_at", "vector_cache", ["created_at"])
    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("full_name", sa.String(255), nullable=False),
        sa.Column("phone", sa.String(32), nullable=False),
        sa.Column("pin_code", sa.String(6), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"])
    op.create_index("ix_users_role", "users", ["role"])
    op.create_index("ix_users_pin_code", "users", ["pin_code"])
    op.create_table(
        "pharmacies",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False, unique=True),
        sa.Column("business_name", sa.String(255), nullable=False),
        sa.Column("license_number", sa.String(128), nullable=False, unique=True),
        sa.Column("address", sa.Text(), nullable=False),
        sa.Column("pin_code", sa.String(6), nullable=False),
        sa.Column("serviceable_pins_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("supports_pickup", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("supports_delivery", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("approval_status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_pharmacies_user_id", "pharmacies", ["user_id"])
    op.create_index("ix_pharmacies_pin_code", "pharmacies", ["pin_code"])
    op.create_index("ix_pharmacies_approval_status", "pharmacies", ["approval_status"])
    op.create_table(
        "inventory",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("pharmacy_id", sa.String(36), sa.ForeignKey("pharmacies.id"), nullable=False),
        sa.Column("medicine_name", sa.String(255), nullable=False),
        sa.Column("normalized_name", sa.String(255), nullable=False),
        sa.Column("unit_label", sa.String(128), nullable=False),
        sa.Column("price_paise", sa.Integer(), nullable=False),
        sa.Column("stock_quantity", sa.Integer(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("pharmacy_id", "normalized_name", name="uq_inventory_pharmacy_medicine"),
    )
    op.create_index("ix_inventory_pharmacy_id", "inventory", ["pharmacy_id"])
    op.create_table(
        "prescription_files",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("analysis_id", sa.String(36), sa.ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("owner_id", sa.String(255), nullable=False),
        sa.Column("storage_name", sa.String(255), nullable=False, unique=True),
        sa.Column("original_name", sa.String(255), nullable=False),
        sa.Column("content_type", sa.String(128), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_prescription_files_analysis_id", "prescription_files", ["analysis_id"])
    op.create_index("ix_prescription_files_owner_id", "prescription_files", ["owner_id"])
    op.create_index("ix_prescription_files_expires_at", "prescription_files", ["expires_at"])
    op.create_table(
        "orders",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("analysis_id", sa.String(36), nullable=False),
        sa.Column("patient_id", sa.String(255), nullable=False),
        sa.Column("pharmacy_id", sa.String(36), sa.ForeignKey("pharmacies.id"), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("fulfillment_mode", sa.String(32), nullable=False),
        sa.Column("delivery_address", sa.Text(), nullable=False, server_default=""),
        sa.Column("total_paise", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    for column in ("analysis_id", "patient_id", "pharmacy_id", "status", "expires_at"):
        op.create_index(f"ix_orders_{column}", "orders", [column])
    op.create_table(
        "order_items",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("order_id", sa.String(36), sa.ForeignKey("orders.id", ondelete="CASCADE"), nullable=False),
        sa.Column("inventory_id", sa.String(36), sa.ForeignKey("inventory.id"), nullable=True),
        sa.Column("medicine_name", sa.String(255), nullable=False),
        sa.Column("prescription_json", sa.Text(), nullable=False),
        sa.Column("generic_inquiry", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("availability", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("verified_quantity", sa.Integer(), nullable=True),
        sa.Column("unit_price_paise", sa.Integer(), nullable=True),
        sa.Column("pharmacist_note", sa.Text(), nullable=False, server_default=""),
    )
    op.create_index("ix_order_items_order_id", "order_items", ["order_id"])
    op.create_table(
        "order_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("order_id", sa.String(36), sa.ForeignKey("orders.id", ondelete="CASCADE"), nullable=False),
        sa.Column("actor_id", sa.String(255), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("note", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_order_events_order_id", "order_events", ["order_id"])


def downgrade() -> None:
    for table in ("order_events", "order_items", "orders", "prescription_files", "inventory", "pharmacies", "users", "vector_cache"):
        op.drop_table(table)
