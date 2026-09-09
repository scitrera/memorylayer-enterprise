"""SQLAlchemy Core table definitions mirroring migration 001.

These ``Table`` objects are used by the PostgreSQL-backed catalog / provider
store for CRUD via SQLAlchemy Core (no ORM/declarative Base). They must stay in
sync with ``db/migrations/versions/001_initial_schema.py`` — alembic remains the
source of truth for the actual DDL; this metadata is only a query surface.
"""
from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

metadata = sa.MetaData()

providers_table = sa.Table(
    "providers",
    metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("workspace_id", sa.Text(), nullable=False),
    sa.Column("name", sa.Text(), nullable=False),
    sa.Column("provider_type", sa.Text(), nullable=False),
    sa.Column("description", sa.Text(), nullable=True),
    sa.Column("enabled", sa.Boolean(), nullable=False),
    sa.Column("connection_args", JSONB(), nullable=False),
    sa.Column("encrypted_args", JSONB(), nullable=False),
    sa.Column("schedule", sa.Text(), nullable=True),
    sa.Column("last_sync_at", sa.TIMESTAMP(timezone=True), nullable=True),
    sa.Column("metadata", JSONB(), nullable=False),
    sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
    sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False),
)

vfs_entries_table = sa.Table(
    "vfs_entries",
    metadata,
    sa.Column("vfs_ref", sa.Text(), primary_key=True),
    sa.Column("workspace_id", sa.Text(), nullable=False),
    sa.Column("connector_id", sa.Text(), nullable=True),
    sa.Column("source_path", sa.Text(), nullable=False),
    sa.Column("content_hash", sa.Text(), nullable=False),
    sa.Column("content_type", sa.Text(), nullable=True),
    sa.Column("size_bytes", sa.BigInteger(), nullable=True),
    sa.Column("blob_key", sa.Text(), nullable=True),
    sa.Column("ml_doc_id", sa.Text(), nullable=True),
    sa.Column("ml_job_id", sa.Text(), nullable=True),
    sa.Column("metadata", JSONB(), nullable=False),
    sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
    sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False),
)

sync_checkpoints_table = sa.Table(
    "sync_checkpoints",
    metadata,
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("provider_id", sa.Text(), nullable=False),
    sa.Column("workspace_id", sa.Text(), nullable=False),
    sa.Column("checkpoint_data", JSONB(), nullable=False),
    sa.Column("entries_count", sa.Integer(), nullable=False),
    sa.Column("synced_at", sa.TIMESTAMP(timezone=True), nullable=False),
    sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
)
