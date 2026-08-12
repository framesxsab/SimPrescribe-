"""Initial schema: analyses and audit_events

Revision ID: 0001_initial
Revises:
Create Date: 2026-08-12

Matches the tables created by simpliscribe.storage metadata.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "analyses",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("owner_id", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", sa.Text(), nullable=False),
    )
    op.create_index("ix_analyses_owner_id", "analyses", ["owner_id"])
    op.create_index("ix_analyses_created_at", "analyses", ["created_at"])
    op.create_index("ix_analyses_expires_at", "analyses", ["expires_at"])

    op.create_table(
        "audit_events",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("owner_id", sa.String(length=255), nullable=False),
        sa.Column("analysis_id", sa.String(length=36), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=False, server_default="{}"),
    )
    op.create_index("ix_audit_events_owner_id", "audit_events", ["owner_id"])
    op.create_index("ix_audit_events_analysis_id", "audit_events", ["analysis_id"])
    op.create_index("ix_audit_events_created_at", "audit_events", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_audit_events_created_at", table_name="audit_events")
    op.drop_index("ix_audit_events_analysis_id", table_name="audit_events")
    op.drop_index("ix_audit_events_owner_id", table_name="audit_events")
    op.drop_table("audit_events")

    op.drop_index("ix_analyses_expires_at", table_name="analyses")
    op.drop_index("ix_analyses_created_at", table_name="analyses")
    op.drop_index("ix_analyses_owner_id", table_name="analyses")
    op.drop_table("analyses")
