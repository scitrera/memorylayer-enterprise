"""SQLAlchemy ORM models for MemoryLayer.ai (PostgreSQL + pgvector)."""
import os
import uuid
from datetime import datetime
from typing import Any

# Configurable vector dimensions - read at import time for ORM model definition
_EMBEDDING_DIM = int(os.environ.get('MEMORYLAYER_EMBEDDING_DIMENSIONS', '1536'))
_MULTIVECTOR_DIM = int(os.environ.get('MEMORYLAYER_MULTIVECTOR_DIMENSIONS', '128'))

from pgvector.sqlalchemy import Vector, HALFVEC
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .database import Base


class WorkspaceModel(Base):
    """Workspace table - tenant boundary for multi-tenancy."""

    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    settings: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default="{}")

    # Discovery tags for tag-based workspace lookup (e.g. 'knowledge', 'topic:finance')
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")

    # Cold tier tiering configuration (overrides global TieringConfig)
    # Expected structure:
    # {
    #   "cold_tier_enabled": bool,        # Enable/disable cold tier for this workspace
    #   "archival_age_days": int,         # Days since last access before archival
    #   "importance_threshold": float,    # Max importance for archival (0.0-1.0)
    #   "access_count_threshold": int,    # Max access count for archival
    #   "warmup_access_threshold": int,   # Cold accesses before warm-up promotion
    #   "cold_tier_search_enabled": bool  # Enable cold tier search in recall
    # }
    tiering_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    contexts: Mapped[list["ContextModel"]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan"
    )
    memories: Mapped[list["MemoryModel"]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan"
    )
    sessions: Mapped[list["SessionModel"]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan"
    )

    __table_args__ = (
        # GIN index backs tag-based workspace lookup (tags @> / && operators)
        Index("idx_workspaces_tags", "tags", postgresql_using="gin"),
    )


class ContextModel(Base):
    """Contexts - logical grouping within workspace (e.g., 'project-alpha')."""

    __tablename__ = "contexts"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        Text, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    settings: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    workspace: Mapped["WorkspaceModel"] = relationship(back_populates="contexts")
    memories: Mapped[list["MemoryModel"]] = relationship(
        back_populates="context", cascade="all, delete-orphan"
    )

    # Constraints
    __table_args__ = (UniqueConstraint("workspace_id", "name", name="uq_workspace_context_name"),)


class MemoryModel(Base):
    """Memory entries - core memory storage with pgvector embedding."""

    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        Text, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False, server_default="_default")
    # RPG uses context rows to isolate the canonical graph from task/intent
    # overlays. Other memory paths may continue to leave this nullable.
    context_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("contexts.id", ondelete="SET NULL"), nullable=True
    )
    user_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    logical_key: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Content
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)

    # Classification
    type: Mapped[str] = mapped_column(Text, nullable=False)  # episodic, semantic, procedural, working
    subtype: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # Solution, Problem, CodePattern, etc.
    importance: Mapped[float] = mapped_column(
        Float, nullable=False, server_default="0.5"
    )
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, server_default="{}")
    refinement_meta: Mapped[dict[str, Any]] = mapped_column(
        "refinement_metadata", JSONB, nullable=False, server_default="{}"
    )

    # Hierarchical memory fields (aligned with OSS Memory model)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    overview: Mapped[str | None] = mapped_column(Text, nullable=True)
    session_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_memory_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="active")
    pinned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Entity attribution (v3) - "who remembers what about whom"
    observer_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    subject_id: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Document provenance - traces memory back to source document/page
    source_document_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )
    source_page_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("document_pages.id", ondelete="SET NULL"), nullable=True
    )
    source_dataset_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_thread_id: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Vector embedding (dimensions configured via MEMORYLAYER_EMBEDDING_DIMENSIONS)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(_EMBEDDING_DIM), nullable=True)

    # Multi-vector embedding for ColPali (array of _MULTIVECTOR_DIM-dim vectors)
    # Used for late interaction (MaxSim) retrieval. Stored as halfvec (16-bit):
    # measured equal-quality to float32 (1b) at half the bytes; see
    # migrations 006/009 and max_sim(halfvec[], halfvec[]) in 001.
    multivector: Mapped[list[list[float]] | None] = mapped_column(
        ARRAY(HALFVEC(_MULTIVECTOR_DIM)), nullable=True
    )

    # Lifecycle
    access_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    last_accessed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    decay_factor: Mapped[float] = mapped_column(Float, nullable=False, server_default="1.0")
    archived_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    deleted_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    etag: Mapped[str] = mapped_column(Text, nullable=False, server_default="")

    # Temporal: when the memory's content is about (event time), distinct from created_at
    event_time: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    workspace: Mapped["WorkspaceModel"] = relationship(back_populates="memories")
    context: Mapped["ContextModel | None"] = relationship(back_populates="memories")
    fragments: Mapped[list["MemoryFragmentModel"]] = relationship(
        back_populates="memory", cascade="all, delete-orphan"
    )
    cue_anchors: Mapped[list["CueAnchorModel"]] = relationship(
        back_populates="memory", cascade="all, delete-orphan"
    )
    source_associations: Mapped[list["MemoryAssociationModel"]] = relationship(
        foreign_keys="MemoryAssociationModel.source_id",
        back_populates="source_memory",
        cascade="all, delete-orphan",
    )
    target_associations: Mapped[list["MemoryAssociationModel"]] = relationship(
        foreign_keys="MemoryAssociationModel.target_id",
        back_populates="target_memory",
        cascade="all, delete-orphan",
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "type IN ('episodic', 'semantic', 'procedural', 'working')", name="ck_memory_type"
        ),
        CheckConstraint("importance >= 0 AND importance <= 1", name="ck_memory_importance"),
        Index("idx_memories_workspace", "workspace_id", postgresql_where=(deleted_at.is_(None))),
        Index(
            "idx_memories_workspace_type",
            "workspace_id",
            "type",
            postgresql_where=(deleted_at.is_(None)),
        ),
        Index(
            "idx_memories_workspace_user",
            "workspace_id",
            "user_id",
            postgresql_where=(deleted_at.is_(None)),
        ),
        Index(
            "idx_memories_workspace_logical_key_global",
            "workspace_id",
            "logical_key",
            unique=True,
            postgresql_where=text("user_id IS NULL AND logical_key IS NOT NULL"),
        ),
        Index(
            "idx_memories_workspace_user_logical_key",
            "workspace_id",
            "user_id",
            "logical_key",
            unique=True,
            postgresql_where=text("user_id IS NOT NULL AND logical_key IS NOT NULL"),
        ),
        Index(
            "idx_memories_tags",
            "tags",
            postgresql_using="gin",
            postgresql_where=(deleted_at.is_(None)),
        ),
        Index(
            "idx_memories_rpg_context_subtype",
            "workspace_id",
            "context_id",
            "subtype",
            postgresql_where=text(
                "deleted_at IS NULL AND status = 'active' "
                "AND tags @> ARRAY['rpg']::text[]"
            ),
        ),
        Index(
            "idx_memories_rpg_metadata",
            "metadata",
            postgresql_using="gin",
            postgresql_ops={"metadata": "jsonb_path_ops"},
            postgresql_where=text(
                "deleted_at IS NULL AND status = 'active' "
                "AND tags @> ARRAY['rpg']::text[]"
            ),
        ),
        Index(
            "idx_memories_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_where=(embedding.is_not(None) & deleted_at.is_(None)),
        ),
        # Note: pgvector doesn't support HNSW on ARRAY(Vector) directly.
        # MaxSim queries will use sequential scan with workspace_id filter.
        # The workspace index above provides adequate filtering for multivector queries.
        Index(
            "idx_memories_observer",
            "workspace_id",
            "observer_id",
            postgresql_where=(observer_id.is_not(None) & deleted_at.is_(None)),
        ),
        Index(
            "idx_memories_subject",
            "workspace_id",
            "subject_id",
            postgresql_where=(subject_id.is_not(None) & deleted_at.is_(None)),
        ),
        Index(
            "idx_memories_source_document",
            "workspace_id",
            "source_document_id",
            postgresql_where=(source_document_id.is_not(None) & deleted_at.is_(None)),
        ),
        Index(
            "idx_memories_source_page",
            "source_page_id",
            postgresql_where=(source_page_id.is_not(None) & deleted_at.is_(None)),
        ),
        Index(
            "idx_memories_source_dataset",
            "source_dataset_id",
            postgresql_where=(source_dataset_id.is_not(None) & deleted_at.is_(None)),
        ),
        Index(
            "idx_memories_source_thread",
            "source_thread_id",
            postgresql_where=(source_thread_id.is_not(None) & deleted_at.is_(None)),
        ),
    )


class MemoryRevisionModel(Base):
    """Immutable snapshot for one accepted native semantic-memory mutation."""

    __tablename__ = "memory_revisions"

    sequence: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False)
    workspace_id: Mapped[str] = mapped_column(Text, nullable=False)
    memory_id: Mapped[str] = mapped_column(Text, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    operation_id: Mapped[str] = mapped_column(Text, nullable=False)
    request_hash: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "workspace_id", "memory_id", "revision",
            name="uq_memory_revision",
        ),
        Index(
            "idx_memory_revisions_resource",
            "tenant_id", "workspace_id", "memory_id", "sequence",
        ),
    )


class MemoryOperationModel(Base):
    """Idempotency key mapped to its exact immutable memory result."""

    __tablename__ = "memory_operations"

    tenant_id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(Text, primary_key=True)
    operation_id: Mapped[str] = mapped_column(Text, primary_key=True)
    request_hash: Mapped[str] = mapped_column(Text, nullable=False)
    memory_id: Mapped[str] = mapped_column(Text, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)


class MemoryAssociationModel(Base):
    """Memory associations - semantic graph edges between memories."""

    __tablename__ = "memory_associations"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(Text, nullable=False)
    source_id: Mapped[str] = mapped_column(
        Text, ForeignKey("memories.id", ondelete="CASCADE"), nullable=False
    )
    target_id: Mapped[str] = mapped_column(
        Text, ForeignKey("memories.id", ondelete="CASCADE"), nullable=False
    )
    relation_type: Mapped[str] = mapped_column(
        "relationship", Text, nullable=False
    )  # CAUSES, SOLVES, RELATED_TO, etc.
    strength: Mapped[float] = mapped_column(Float, nullable=False, server_default="0.5")
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    source_memory: Mapped["MemoryModel"] = relationship(
        foreign_keys=[source_id], back_populates="source_associations"
    )
    target_memory: Mapped["MemoryModel"] = relationship(
        foreign_keys=[target_id], back_populates="target_associations"
    )

    # Constraints
    __table_args__ = (
        CheckConstraint("strength >= 0 AND strength <= 1", name="ck_association_strength"),
        UniqueConstraint("source_id", "target_id", "relationship", name="uq_association"),
        Index("idx_associations_workspace", "workspace_id"),
        Index("idx_associations_source", "source_id"),
        Index("idx_associations_target", "target_id"),
        Index(
            "idx_associations_rpg_source",
            "workspace_id",
            "source_id",
            "relationship",
            postgresql_where=text(
                "relationship IN ('contains', 'contained_by', 'inherits', "
                "'inherited_by', 'invokes', 'invoked_by', 'imports', "
                "'imported_by', 'composes', 'composed_by', 'data_flow', "
                "'data_flow_from')"
            ),
        ),
        Index(
            "idx_associations_rpg_target",
            "workspace_id",
            "target_id",
            "relationship",
            postgresql_where=text(
                "relationship IN ('contains', 'contained_by', 'inherits', "
                "'inherited_by', 'invokes', 'invoked_by', 'imports', "
                "'imported_by', 'composes', 'composed_by', 'data_flow', "
                "'data_flow_from')"
            ),
        ),
    )


class MemoryFragmentModel(Base):
    """Memory fragments - chunked content for large memories."""

    __tablename__ = "memory_fragments"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    memory_id: Mapped[str] = mapped_column(
        Text, ForeignKey("memories.id", ondelete="CASCADE"), nullable=False
    )
    workspace_id: Mapped[str] = mapped_column(Text, nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(_EMBEDDING_DIM), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    memory: Mapped["MemoryModel"] = relationship(back_populates="fragments")

    # Constraints
    __table_args__ = (
        UniqueConstraint("memory_id", "sequence", name="uq_fragment_sequence"),
        Index("idx_fragments_memory", "memory_id"),
        Index(
            "idx_fragments_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class CueAnchorModel(Base):
    """Cue anchors - short "[entity] + [aspect]" semantic keys per memory.

    Memora-inspired abstraction+cue indexing: each memory can have 1-3 cue
    anchors generated at ingest (see ExtractionService.generate_cue_anchors),
    embedded, and stored here. At recall the cue embeddings are searched by
    vector similarity and dereferenced back to their parent ``memory_id`` for the
    RRF cue fusion arm (MemoryService._fuse_cue_results). Shares the shape of
    MemoryFragmentModel. Ships DARK (rows only written when the cue channel is on).
    """

    __tablename__ = "cue_anchors"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        Text, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    memory_id: Mapped[str] = mapped_column(
        Text, ForeignKey("memories.id", ondelete="CASCADE"), nullable=False
    )
    cue: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_cue: Mapped[str] = mapped_column(Text, nullable=False)
    # Canonical Entity the cue names, when resolvable against the entity registry
    # at ingest (else NULL). Links the cue arm and the entity-anchored arm to a
    # shared entity vocabulary. SET NULL on entity delete so a cue survives its
    # entity being merged/removed. (Migration 011.)
    entity_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("entities.id", ondelete="SET NULL"), nullable=True
    )
    embedding: Mapped[list[float] | None] = mapped_column(Vector(_EMBEDDING_DIM), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    memory: Mapped["MemoryModel"] = relationship(back_populates="cue_anchors")

    # Constraints
    __table_args__ = (
        Index("idx_cue_anchors_workspace", "workspace_id"),
        Index("idx_cue_anchors_memory", "memory_id"),
        Index("idx_cue_anchors_normalized", "normalized_cue"),
        Index("idx_cue_anchors_entity", "entity_id"),
        Index(
            "idx_cue_anchors_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class SessionModel(Base):
    """Sessions - working memory context."""

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        Text, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False, server_default="_default")
    # context_id — RESERVED / unused as a filter today; NOT NULL + "_default"
    # sentinel (no FK on this table). See MemoryModel.context_id for rationale.
    context_id: Mapped[str] = mapped_column(Text, nullable=False, server_default="_default")
    auto_commit: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    committed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, server_default="{}")
    expires_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    workspace: Mapped["WorkspaceModel"] = relationship(back_populates="sessions")
    context_entries: Mapped[list["SessionContextModel"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class SessionContextModel(Base):
    """Session context - key-value working memory."""

    __tablename__ = "session_context"

    session_id: Mapped[str] = mapped_column(
        Text, ForeignKey("sessions.id", ondelete="CASCADE"), primary_key=True
    )
    key: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    ttl_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    session: Mapped["SessionModel"] = relationship(back_populates="context_entries")


class SessionCheckpointModel(Base):
    """Durable raw transcript capture recorded before derived processing."""

    __tablename__ = "session_checkpoints"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        Text, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    session_id: Mapped[str] = mapped_column(
        Text, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    raw_memory_id: Mapped[str] = mapped_column(
        Text, ForeignKey("memories.id", ondelete="CASCADE"), nullable=False
    )
    source_kind: Mapped[str] = mapped_column(Text, nullable=False)
    source_sequence: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    source_boundary: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    byte_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    capture_status: Mapped[str] = mapped_column(Text, nullable=False)
    index_status: Mapped[str] = mapped_column(Text, nullable=False)
    enrichment_status: Mapped[str] = mapped_column(Text, nullable=False)
    idempotency_key: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint("session_id", "idempotency_key", name="uq_session_checkpoint_idempotency"),
        Index("idx_session_checkpoints_scope", "workspace_id", "session_id", "created_at"),
    )


class SessionContextEventModel(Base):
    """Monotonic session-context change log used by opaque delta cursors."""

    __tablename__ = "session_context_events"

    sequence: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    workspace_id: Mapped[str] = mapped_column(
        Text, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    session_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=True
    )
    event_kind: Mapped[str] = mapped_column(Text, nullable=False)
    subject_kind: Mapped[str] = mapped_column(Text, nullable=False)
    subject_id: Mapped[str] = mapped_column(Text, nullable=False)
    event_time: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, server_default="{}")

    __table_args__ = (
        Index("idx_session_context_events_scope", "workspace_id", "session_id", "sequence"),
        Index("idx_session_context_events_retention", "event_time"),
    )


class EntityRelationModel(Base):
    """Canonical typed edge between two workspace entities."""

    __tablename__ = "entity_relations"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        Text, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    source_entity_id: Mapped[str] = mapped_column(
        Text, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    target_entity_id: Mapped[str] = mapped_column(
        Text, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    relationship: Mapped[str] = mapped_column(Text, nullable=False)
    direction: Mapped[str] = mapped_column(Text, nullable=False, server_default="outgoing")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, server_default="1.0")
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint("source_entity_id <> target_entity_id", name="ck_entity_relation_not_self"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_entity_relation_confidence"),
        Index(
            "uq_entity_relations_active_edge",
            "workspace_id",
            "source_entity_id",
            "target_entity_id",
            "relationship",
            unique=True,
            postgresql_where=text("active = true"),
        ),
        Index("idx_entity_relations_source", "workspace_id", "source_entity_id", "relationship"),
        Index("idx_entity_relations_target", "workspace_id", "target_entity_id", "relationship"),
    )


class EntityRelationEvidenceModel(Base):
    """Inspectable source evidence supporting a structural entity edge."""

    __tablename__ = "entity_relation_evidence"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        Text, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    relation_id: Mapped[str] = mapped_column(
        Text, ForeignKey("entity_relations.id", ondelete="CASCADE"), nullable=False
    )
    source_memory_id: Mapped[str] = mapped_column(
        Text, ForeignKey("memories.id", ondelete="CASCADE"), nullable=False
    )
    evidence_kind: Mapped[str] = mapped_column(Text, nullable=False)
    source_span_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_span_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    excerpt_hash: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, server_default="1.0")
    extraction_method: Mapped[str] = mapped_column(Text, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_entity_relation_evidence_confidence"),
        Index(
            "uq_entity_relation_evidence_source_span",
            "relation_id",
            "source_memory_id",
            text("COALESCE(source_span_start, -1)"),
            text("COALESCE(source_span_end, -1)"),
            "excerpt_hash",
            unique=True,
        ),
        Index("idx_entity_relation_evidence_relation", "workspace_id", "relation_id", "active"),
        Index("idx_entity_relation_evidence_memory", "workspace_id", "source_memory_id", "active"),
    )


class MemoryAccessLogModel(Base):
    """Memory access log - analytics and decay calculation."""

    __tablename__ = "memory_access_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    workspace_id: Mapped[str] = mapped_column(Text, nullable=False)
    memory_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    access_type: Mapped[str] = mapped_column(
        Text, nullable=False
    )  # recall, reflect, associate
    query: Mapped[str | None] = mapped_column(Text, nullable=True)
    relevance_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    accessed_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    # Indexes
    __table_args__ = (
        Index("idx_access_log_workspace", "workspace_id"),
        Index("idx_access_log_accessed_at", "accessed_at"),
    )


class ContradictionModel(Base):
    """Contradiction records - detected conflicts between memories."""

    __tablename__ = "contradictions"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(Text, nullable=False)
    memory_a_id: Mapped[str] = mapped_column(Text, nullable=False)
    memory_b_id: Mapped[str] = mapped_column(Text, nullable=False)
    contradiction_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, server_default="0.0")
    detection_method: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    detected_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    resolved_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    resolution: Mapped[str | None] = mapped_column(Text, nullable=True)
    merged_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Which of the two memories is the CURRENT one. Without it a record says the pair
    # conflicts but not which side is stale, so recall-side supersession has nothing to
    # act on. NULL = direction unknown, which supersedes nothing (never guessed at).
    newer_memory_id: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("idx_contradictions_workspace", "workspace_id"),
        Index(
            "idx_contradictions_unresolved",
            "workspace_id",
            postgresql_where=(resolved_at.is_(None)),
        ),
        # Backs get_superseded_memory_ids: "of these ids, which are the stale side of an
        # unresolved contradiction".
        Index(
            "idx_contradictions_superseded",
            "workspace_id",
            "newer_memory_id",
            postgresql_where=(resolved_at.is_(None)),
        ),
    )


class LeannGraphModel(Base):
    """LEANN cold tier graph - compressed neighbor graph structure for cold storage.

    Stores the pruned neighbor graph in CSR (Compressed Sparse Row) format,
    enabling 90%+ storage reduction by eliminating stored embeddings.
    """

    __tablename__ = "leann_graphs"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(Text, nullable=False)

    # Graph structure stored as serialized CSR format (binary)
    graph_data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    # Graph statistics
    node_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    edge_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    # List of memory IDs included in this graph
    memory_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")

    # Metadata for compression/retrieval tuning
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, server_default="{}")

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    documents: Mapped[list["LeannDocumentModel"]] = relationship(
        back_populates="graph", cascade="all, delete-orphan"
    )

    # Indexes
    __table_args__ = (
        Index("idx_leann_graphs_workspace", "workspace_id"),
        Index(
            "idx_leann_graphs_memory_ids",
            "memory_ids",
            postgresql_using="gin",
        ),
    )


class LeannDocumentModel(Base):
    """LEANN cold tier document - archived memory content without embedding.

    Stores the original memory content and position in the neighbor graph,
    allowing on-demand embedding computation during cold tier retrieval.
    """

    __tablename__ = "leann_documents"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    graph_id: Mapped[str] = mapped_column(
        Text, ForeignKey("leann_graphs.id", ondelete="CASCADE"), nullable=False
    )
    workspace_id: Mapped[str] = mapped_column(Text, nullable=False)  # Denormalized for queries

    # Reference to original memory (may be null if memory was deleted)
    memory_id: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Original memory content preserved for on-demand embedding
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Position/index in the neighbor graph for graph-guided search
    position: Mapped[int] = mapped_column(Integer, nullable=False)

    # Track cold tier access for automatic warm-up to hot tier
    cold_access_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    last_cold_access_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)

    # Original memory metadata preserved
    memory_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    memory_subtype: Mapped[str | None] = mapped_column(Text, nullable=True)
    importance: Mapped[float] = mapped_column(Float, nullable=False, server_default="0.5")
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, server_default="{}")

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    graph: Mapped["LeannGraphModel"] = relationship(back_populates="documents")

    # Constraints and Indexes
    __table_args__ = (
        UniqueConstraint("graph_id", "position", name="uq_leann_document_position"),
        Index("idx_leann_documents_workspace", "workspace_id"),
        Index("idx_leann_documents_graph", "graph_id"),
        Index("idx_leann_documents_memory", "memory_id"),
        Index(
            "idx_leann_documents_tags",
            "tags",
            postgresql_using="gin",
        ),
        Index(
            "idx_leann_documents_cold_access",
            "workspace_id",
            "cold_access_count",
        ),
    )


class DocumentModel(Base):
    """Documents - uploaded files for ingestion."""
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        Text, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False, server_default="_default")
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    document_type: Mapped[str] = mapped_column(Text, nullable=False)  # pdf, markdown, text, html, docx, pptx
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    source_vfs_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    mime_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    # target_context_id flows to each extracted memory's context_id — RESERVED /
    # unused as a filter today. See MemoryModel.context_id for rationale.
    target_context_id: Mapped[str] = mapped_column(Text, nullable=False, server_default="_default")
    extraction_options: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default="{}")
    page_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    memory_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")
    # Knowledge phase, deliberately separate from ``status`` (which tracks
    # retrieval readiness only). Existing rows default to not_applicable: they
    # predate the split and nothing was recorded as scheduled for them, so
    # claiming their enrichment is "complete" would be a fabrication.
    enrichment_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default="not_applicable"
    )
    enrichment_memory_ids: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default="{}"
    )
    deduplicated_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    storage_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    retain_original: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, server_default="{}")
    extracted_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    processing_started_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    processing_completed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)

    # Relationships
    workspace: Mapped["WorkspaceModel"] = relationship()
    pages: Mapped[list["DocumentPageModel"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            # Must match the DocumentStatus enum (models/document.py) AND
            # migration 016. 'pending_fetch' is the data-connectors byte-fetch
            # lifecycle state — omitting it here breaks create_all-built DBs.
            "status IN ('pending', 'pending_fetch', 'processing', 'completed', 'failed', 'partial')",
            name="ck_document_status"
        ),
        CheckConstraint(
            "document_type IN ('pdf', 'markdown', 'text', 'html', 'docx', 'pptx')",
            name="ck_document_type"
        ),
        Index("idx_documents_workspace", "workspace_id"),
        # Same-bytes-→-one-doc invariant (LINK decision). DB-enforces what the
        # ingest entry-point (find_document_by_hash / doc_added) relies on. The
        # old non-unique idx_documents_content_hash had IDENTICAL leading
        # columns (workspace_id, content_hash), so this constraint's implicit
        # index fully supersedes it — the old index was dropped (migration 024)
        # to avoid a redundant duplicate index.
        UniqueConstraint(
            "workspace_id", "content_hash", name="uq_documents_workspace_content_hash"
        ),
        Index("idx_documents_status", "workspace_id", "status"),
    )


class DocumentPageModel(Base):
    """Document pages - persisted page registry for ingested documents."""
    __tablename__ = "document_pages"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    document_id: Mapped[str] = mapped_column(
        Text, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    workspace_id: Mapped[str] = mapped_column(Text, nullable=False)
    page_no: Mapped[int] = mapped_column(Integer, nullable=False)
    image_storage_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Single-vector text embedding (dimensions configured via
    # MEMORYLAYER_EMBEDDING_DIMENSIONS) — mirrors MemoryModel.embedding exactly so
    # the same embed model serves both. Replaces the brittle metadata['_embedding']
    # JSON stash; binds normally through the ORM (a single vector(N), unlike the
    # multivector ARRAY(Vector) below). See DESIGN_idempotent_ingestion.md P2.4.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(_EMBEDDING_DIM), nullable=True)
    multivector: Mapped[list[list[float]] | None] = mapped_column(
        ARRAY(Vector(_MULTIVECTOR_DIM)), nullable=True
    )
    transcript_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    transcript_attempts: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )
    visual_tokens: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, server_default="{}"
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    document: Mapped["DocumentModel"] = relationship(back_populates="pages")

    __table_args__ = (
        UniqueConstraint("document_id", "page_no", name="uq_document_page"),
        Index("idx_document_pages_workspace", "workspace_id"),
        Index("idx_document_pages_document", "document_id"),
        Index("idx_document_pages_workspace_document", "workspace_id", "document_id"),
    )


class IngestionJobModel(Base):
    """Ingestion jobs - track document processing progress."""
    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(Text, nullable=False)
    document_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="queued")
    progress_percent: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    documents_processed: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    total_memories_created: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    webhook_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    errors: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default="[]")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'failed', 'cancelled')",
            name="ck_job_status"
        ),
        Index("idx_jobs_workspace", "workspace_id"),
        Index("idx_jobs_status", "workspace_id", "status"),
        Index("idx_jobs_created_at", "created_at"),
    )


# ------------------------------------------------------------------ #
# Dataset models
# ------------------------------------------------------------------ #

class DatasetModel(Base):
    """Datasets - uploaded tabular data for profiling and memory extraction."""
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        Text, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False, server_default="_default")
    name: Mapped[str] = mapped_column(Text, nullable=False)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    format: Mapped[str] = mapped_column(Text, nullable=False)  # csv, tsv, parquet, jsonl, xlsx
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    storage_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    original_storage_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    # target_context_id flows to each extracted memory's context_id — RESERVED /
    # unused as a filter today. See MemoryModel.context_id for rationale.
    target_context_id: Mapped[str] = mapped_column(Text, nullable=False, server_default="_default")
    profiling_options: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default="{}")
    row_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    column_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    columns: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default="[]")
    memory_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")
    profile_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    profiling_started_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    profiling_completed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)

    # Relationships
    workspace: Mapped["WorkspaceModel"] = relationship()

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'profiling', 'summarizing', 'completed', 'failed')",
            name="ck_dataset_status"
        ),
        CheckConstraint(
            "format IN ('csv', 'tsv', 'parquet', 'jsonl', 'xlsx')",
            name="ck_dataset_format"
        ),
        Index("idx_datasets_workspace", "workspace_id"),
        Index("idx_datasets_content_hash", "workspace_id", "content_hash"),
        Index("idx_datasets_status", "workspace_id", "status"),
    )


class DatasetJobModel(Base):
    """Dataset jobs - track dataset profiling/summarization progress."""
    __tablename__ = "dataset_jobs"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(Text, nullable=False)
    dataset_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="queued")
    progress_percent: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    datasets_processed: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    total_memories_created: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    errors: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default="[]")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'failed', 'cancelled')",
            name="ck_dataset_job_status"
        ),
        Index("idx_dataset_jobs_workspace", "workspace_id"),
        Index("idx_dataset_jobs_status", "workspace_id", "status"),
    )


class ChatThreadModel(Base):
    """Chat threads for conversation history."""

    __tablename__ = "chat_threads"

    # Opaque surrogate primary key (text uuid). Internal-only: it is the join
    # target for chat_messages and NEVER appears in any API response or event.
    # The client-facing identifier is ``id`` (stored verbatim), scoped per owner
    # by the (workspace_id, COALESCE(user_id,''), id) unique index below — so a
    # shared client id like "_default" is unique per user without being mangled.
    row_id: Mapped[str] = mapped_column(Text, primary_key=True, default=lambda: uuid.uuid4().hex)
    id: Mapped[str] = mapped_column(Text, nullable=False)
    workspace_id: Mapped[str] = mapped_column(
        Text, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False, server_default="_default")
    # context_id — RESERVED / unused as a filter today; NOT NULL + "_default"
    # sentinel (no FK on this table). See MemoryModel.context_id for rationale.
    context_id: Mapped[str] = mapped_column(Text, nullable=False, server_default="_default")
    user_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    observer_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    subject_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, server_default="{}")
    message_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    last_decomposed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    last_decomposed_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    # Surface scope — separates web-app threads from Office add-in threads.
    # NULL is treated as "web" at read time; no backfill required.
    scope: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Ownership discriminator — separates user-owned threads (the user-session-scoped
    # right rail in the web app) from workspace-shared threads. Defaults to 'user';
    # existing rows backfill to 'user' via server_default — matches the
    # "leave legacy per-workspace _default threads alone" decision.
    ownership: Mapped[str] = mapped_column(String(16), nullable=False, server_default="user")
    # Per-thread idle policy: NULL = never idles out; 'hide' = archive once idle
    # past the server idle threshold; 'delete' = remove. Idle is measured on
    # updated_at (bumped on every append).
    idle_action: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # Archive flag: set when hidden (manually or by the idle 'hide' policy);
    # cleared on revival (a new append un-hides). NULL = visible.
    hidden_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    # Sub-thread parent. NULL = top-level (listings default to top-level only).
    # A child always shares its parent's workspace + ownership (enforced at the
    # service layer). No DB-level self-FK; child cascade is handled in delete_thread.
    parent_thread: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    messages: Mapped[list["ChatMessageModel"]] = relationship(
        back_populates="thread", cascade="all, delete-orphan"
    )

    # Indexes — composite (tenant, user, ownership) supports the cross-workspace
    # list_user_threads query path; mirrors the user-scope patterns on SkillModel.
    # The UNIQUE(workspace_id, id) constraint is defense-in-depth: the application
    # now routes all user-scoped threads through the USER_CHAT_HOME_WORKSPACE
    # sentinel, but the constraint guarantees that even if a code path bypasses
    # the sentinel, the database refuses to create duplicate (workspace_id, id)
    # tuples — preventing the cross-workspace _default-thread collision identified
    # in the chat-rail audit. Existing data is satisfiable: legacy per-workspace
    # _default threads have distinct workspace_ids, so the constraint is consistent
    # on every deployed database.
    __table_args__ = (
        Index(
            "idx_chat_threads_tenant_user_ownership",
            "tenant_id",
            "user_id",
            "ownership",
            postgresql_where=(text("user_id IS NOT NULL")),
        ),
        # Idle-policy scan (delete/hide sweeps) and post-archive grace purge.
        Index(
            "idx_chat_threads_idle",
            "updated_at",
            postgresql_where=(text("idle_action IS NOT NULL")),
        ),
        Index(
            "idx_chat_threads_hidden",
            "hidden_at",
            postgresql_where=(text("hidden_at IS NOT NULL")),
        ),
        # Sub-thread child lookup.
        Index(
            "idx_chat_threads_parent",
            "parent_thread",
            postgresql_where=(text("parent_thread IS NOT NULL")),
        ),
        # Owner-scoped identity: the client id is unique PER OWNER, not globally.
        # COALESCE(user_id,'') scopes user-owned threads (in the _user_chat
        # sentinel) by their OBO subject, while workspace-owned threads (user_id
        # NULL) are scoped by (workspace_id, id). This replaces the old global
        # UNIQUE(workspace_id, id) + the ``::u::<hash>`` id materialization.
        Index(
            "uq_chat_threads_ws_user_id",
            "workspace_id",
            text("COALESCE(user_id, '')"),
            "id",
            unique=True,
        ),
    )


class ChatMessageModel(Base):
    """Chat messages within threads."""

    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    # References the thread's opaque surrogate (chat_threads.row_id), NOT its
    # client-facing id — so a per-owner client id like "_default" is unambiguous.
    thread_id: Mapped[str] = mapped_column(
        Text, ForeignKey("chat_threads.row_id", ondelete="CASCADE"), nullable=False
    )
    message_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[dict[str, Any] | str] = mapped_column(JSONB, nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    thread: Mapped["ChatThreadModel"] = relationship(back_populates="messages")


class UserModel(Base):
    """Platform users."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False)
    email: Mapped[str] = mapped_column(Text, nullable=False)
    display_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    first_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    licensed: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "email", name="uq_users_tenant_email"),
        Index("idx_users_tenant", "tenant_id"),
        Index("idx_users_email", "email"),
    )


class ApplicationModel(Base):
    """Registered applications."""

    __tablename__ = "applications"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    app_type: Mapped[str] = mapped_column(Text, nullable=False, server_default="generic")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default="{}")
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, server_default="{}")
    # Tenant-level capability defaults (resolved per workspace at load time).
    default_skill_names: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default="{}"
    )
    default_mcp_server_names: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default="{}"
    )
    default_tool_names: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default="{}"
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("idx_applications_tenant", "tenant_id"),
    )


class WorkspaceApplicationModel(Base):
    """Junction table linking applications to workspaces."""

    __tablename__ = "workspace_applications"

    workspace_id: Mapped[str] = mapped_column(
        Text, ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True
    )
    application_id: Mapped[str] = mapped_column(
        Text, ForeignKey("applications.id", ondelete="CASCADE"), primary_key=True
    )
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    config_overrides: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default="{}")
    # Workspace-level capability overrides.
    tool_names: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default="{}"
    )
    # 'merge' (default) unions defaults with binding rows; 'replace' uses only binding rows.
    skill_override_mode: Mapped[str] = mapped_column(Text, nullable=False, server_default="merge")
    mcp_override_mode: Mapped[str] = mapped_column(Text, nullable=False, server_default="merge")
    tool_override_mode: Mapped[str] = mapped_column(Text, nullable=False, server_default="merge")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    # Capability binding relationships (eager via selectinload at the service layer).
    skill_bindings: Mapped[list["WorkspaceApplicationSkillModel"]] = relationship(
        back_populates="binding",
        cascade="all, delete-orphan",
        primaryjoin=(
            "and_("
            "WorkspaceApplicationModel.workspace_id == foreign(WorkspaceApplicationSkillModel.workspace_id),"
            "WorkspaceApplicationModel.application_id == foreign(WorkspaceApplicationSkillModel.application_id)"
            ")"
        ),
    )
    mcp_bindings: Mapped[list["WorkspaceApplicationMcpServerModel"]] = relationship(
        back_populates="binding",
        cascade="all, delete-orphan",
        primaryjoin=(
            "and_("
            "WorkspaceApplicationModel.workspace_id == foreign(WorkspaceApplicationMcpServerModel.workspace_id),"
            "WorkspaceApplicationModel.application_id == foreign(WorkspaceApplicationMcpServerModel.application_id)"
            ")"
        ),
    )

    __table_args__ = (
        Index("idx_workspace_apps_workspace", "workspace_id"),
        Index("idx_workspace_apps_application", "application_id"),
    )


class WorkspaceApplicationSkillModel(Base):
    """Workspace-level skill bindings for an application."""

    __tablename__ = "workspace_application_skills"

    workspace_id: Mapped[str] = mapped_column(Text, primary_key=True)
    application_id: Mapped[str] = mapped_column(Text, primary_key=True)
    skill_id: Mapped[str] = mapped_column(
        Text, ForeignKey("skills.id", ondelete="CASCADE"), primary_key=True
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    binding: Mapped["WorkspaceApplicationModel"] = relationship(
        back_populates="skill_bindings",
        primaryjoin=(
            "and_("
            "foreign(WorkspaceApplicationSkillModel.workspace_id) == WorkspaceApplicationModel.workspace_id,"
            "foreign(WorkspaceApplicationSkillModel.application_id) == WorkspaceApplicationModel.application_id"
            ")"
        ),
        viewonly=True,
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["workspace_id", "application_id"],
            ["workspace_applications.workspace_id", "workspace_applications.application_id"],
            ondelete="CASCADE",
            name="fk_wa_skills_binding",
        ),
        Index("idx_wa_skills_binding", "workspace_id", "application_id"),
        Index("idx_wa_skills_skill", "skill_id"),
    )


class WorkspaceApplicationMcpServerModel(Base):
    """Workspace-level MCP server bindings for an application."""

    __tablename__ = "workspace_application_mcp_servers"

    workspace_id: Mapped[str] = mapped_column(Text, primary_key=True)
    application_id: Mapped[str] = mapped_column(Text, primary_key=True)
    mcp_server_id: Mapped[str] = mapped_column(
        Text, ForeignKey("mcp_servers.id", ondelete="CASCADE"), primary_key=True
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    binding: Mapped["WorkspaceApplicationModel"] = relationship(
        back_populates="mcp_bindings",
        primaryjoin=(
            "and_("
            "foreign(WorkspaceApplicationMcpServerModel.workspace_id) == WorkspaceApplicationModel.workspace_id,"
            "foreign(WorkspaceApplicationMcpServerModel.application_id) == WorkspaceApplicationModel.application_id"
            ")"
        ),
        viewonly=True,
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["workspace_id", "application_id"],
            ["workspace_applications.workspace_id", "workspace_applications.application_id"],
            ondelete="CASCADE",
            name="fk_wa_mcp_binding",
        ),
        Index("idx_wa_mcp_binding", "workspace_id", "application_id"),
        Index("idx_wa_mcp_server", "mcp_server_id"),
    )


class CollectionItemModel(Base):
    """Vector collection items for tools, skills, procedures, etc."""

    __tablename__ = "collection_items"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False)
    workspace_id: Mapped[str] = mapped_column(
        Text, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    collection_name: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    item_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, server_default="{}")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    embedding: Mapped[list[float] | None] = mapped_column(Vector(_EMBEDDING_DIM), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("idx_collection_items_workspace", "workspace_id"),
        Index("idx_collection_items_collection", "workspace_id", "collection_name"),
        Index("idx_collection_items_tags", "tags", postgresql_using="gin"),
        Index(
            "idx_collection_items_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_where=(embedding.is_not(None)),
        ),
    )


class DataProviderModel(Base):
    """Data provider registry for document ingestion sources."""

    __tablename__ = "data_providers"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False)
    workspace_id: Mapped[str] = mapped_column(
        Text, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    provider_type: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    connection_args: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default="{}")
    encrypted_args: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default="{}")
    schedule: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_sync_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("idx_data_providers_workspace", "workspace_id"),
        Index("idx_data_providers_type", "workspace_id", "provider_type"),
    )


class SkillModel(Base):
    """Agent skills - knowledge units stored in MemoryLayer."""

    __tablename__ = "skills"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    workspace_id: Mapped[str] = mapped_column(
        Text, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[str] = mapped_column(Text, nullable=False, server_default="0.1.0")
    license: Mapped[str | None] = mapped_column(Text, nullable=True)
    compatibility: Mapped[str | None] = mapped_column(Text, nullable=True)
    allowed_tools: Mapped[str | None] = mapped_column(Text, nullable=True)
    body: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, server_default="{}")
    source_mode: Mapped[str] = mapped_column(Text, nullable=False, server_default="server")
    manifest_hash: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    bundle_hash: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    revision: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    etag: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    deleted_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)

    # Relationships
    files: Mapped[list["SkillFileModel"]] = relationship(
        back_populates="skill", cascade="all, delete-orphan"
    )

    __table_args__ = (
        # Workspace-scoped skills unique by name (no user scope)
        Index(
            "idx_skills_workspace_name_global",
            "workspace_id",
            "name",
            unique=True,
            postgresql_where=(text("user_id IS NULL")),
        ),
        # User-scoped skills unique by name per user
        Index(
            "idx_skills_workspace_user_name",
            "workspace_id",
            "user_id",
            "name",
            unique=True,
            postgresql_where=(text("user_id IS NOT NULL")),
        ),
        # Cross-scope name lookup
        Index("idx_skills_name", "name"),
        # Tenant + workspace filtering
        Index("idx_skills_tenant_workspace", "tenant_id", "workspace_id"),
        Index("idx_skills_workspace", "workspace_id"),
        Index("idx_skills_active", "workspace_id", "deleted_at"),
    )


class SkillFileModel(Base):
    """Skill files - individual files within a skill bundle."""

    __tablename__ = "skill_files"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    skill_id: Mapped[str] = mapped_column(
        Text, ForeignKey("skills.id", ondelete="CASCADE"), nullable=False
    )
    path: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    mime_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    skill: Mapped["SkillModel"] = relationship(back_populates="files")

    __table_args__ = (
        UniqueConstraint("skill_id", "path", name="uq_skill_file_path"),
        Index("idx_skill_files_skill", "skill_id"),
    )


class SkillRevisionModel(Base):
    """Immutable snapshot for one accepted native skill-manifest mutation."""

    __tablename__ = "skill_revisions"

    sequence: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False)
    workspace_id: Mapped[str] = mapped_column(Text, nullable=False)
    skill_id: Mapped[str] = mapped_column(Text, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    operation_id: Mapped[str] = mapped_column(Text, nullable=False)
    request_hash: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "workspace_id", "skill_id", "revision",
            name="uq_skill_revision",
        ),
        Index(
            "idx_skill_revisions_resource",
            "tenant_id", "workspace_id", "skill_id", "sequence",
        ),
    )


class SkillOperationModel(Base):
    """Idempotency key mapped to its exact immutable skill result."""

    __tablename__ = "skill_operations"

    tenant_id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(Text, primary_key=True)
    operation_id: Mapped[str] = mapped_column(Text, primary_key=True)
    request_hash: Mapped[str] = mapped_column(Text, nullable=False)
    skill_id: Mapped[str] = mapped_column(Text, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)


class McpServerModel(Base):
    """MCP server registry — one row per server, scoped by workspace + user."""

    __tablename__ = "mcp_servers"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False, server_default="_default")
    workspace_id: Mapped[str] = mapped_column(
        Text, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    transport: Mapped[str] = mapped_column(Text, nullable=False)
    command: Mapped[str | None] = mapped_column(Text, nullable=True)
    args: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, server_default="{}")
    env: Mapped[dict[str, str]] = mapped_column(JSONB, nullable=False, server_default="{}")
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    headers: Mapped[dict[str, str]] = mapped_column(JSONB, nullable=False, server_default="{}")
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, server_default="{}")
    source_mode: Mapped[str] = mapped_column(Text, nullable=False, server_default="server")
    manifest_hash: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        # Workspace-scoped servers unique by name (no user scope)
        Index(
            "idx_mcp_servers_workspace_name_global",
            "workspace_id",
            "name",
            unique=True,
            postgresql_where=(text("user_id IS NULL")),
        ),
        # User-scoped servers unique by name per user
        Index(
            "idx_mcp_servers_workspace_user_name",
            "workspace_id",
            "user_id",
            "name",
            unique=True,
            postgresql_where=(text("user_id IS NOT NULL")),
        ),
        # Cross-scope name lookup
        Index("idx_mcp_servers_name", "name"),
        # Tenant + workspace filtering
        Index("idx_mcp_servers_tenant_workspace", "tenant_id", "workspace_id"),
        Index("idx_mcp_servers_workspace", "workspace_id"),
    )


class VersionedResourceHeadModel(Base):
    """Authoritative head for an internal typed MemoryLayer resource."""

    __tablename__ = "versioned_resource_heads"

    tenant_id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(Text, primary_key=True)
    namespace: Mapped[str] = mapped_column(Text, primary_key=True)
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    resource_key: Mapped[str] = mapped_column(Text, nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, server_default="{}")
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    sequence: Mapped[int] = mapped_column(BigInteger, nullable=False)
    state_hash: Mapped[str] = mapped_column(Text, nullable=False)
    etag: Mapped[str] = mapped_column(Text, nullable=False)
    created_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "workspace_id",
            "namespace",
            "resource_key",
            name="uq_versioned_resource_logical_key",
        ),
        Index(
            "idx_versioned_resource_heads_list",
            "tenant_id",
            "workspace_id",
            "namespace",
            "sequence",
        ),
    )


class VersionedResourceRevisionModel(Base):
    """Immutable snapshot written for every accepted resource mutation."""

    __tablename__ = "versioned_resource_revisions"

    sequence: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False)
    workspace_id: Mapped[str] = mapped_column(Text, nullable=False)
    namespace: Mapped[str] = mapped_column(Text, nullable=False)
    id: Mapped[str] = mapped_column(Text, nullable=False)
    resource_key: Mapped[str] = mapped_column(Text, nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, server_default="{}")
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    state_hash: Mapped[str] = mapped_column(Text, nullable=False)
    etag: Mapped[str] = mapped_column(Text, nullable=False)
    created_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_by: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    operation_id: Mapped[str] = mapped_column(Text, nullable=False)
    request_hash: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "workspace_id",
            "namespace",
            "id",
            "revision",
            name="uq_versioned_resource_revision",
        ),
        Index(
            "idx_versioned_resource_revisions_resource",
            "tenant_id",
            "workspace_id",
            "namespace",
            "id",
            "sequence",
        ),
    )


class VersionedResourceOperationModel(Base):
    """Accepted idempotency key mapped to its immutable result revision."""

    __tablename__ = "versioned_resource_operations"

    tenant_id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(Text, primary_key=True)
    namespace: Mapped[str] = mapped_column(Text, primary_key=True)
    operation_id: Mapped[str] = mapped_column(Text, primary_key=True)
    request_hash: Mapped[str] = mapped_column(Text, nullable=False)
    resource_id: Mapped[str] = mapped_column(Text, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)


class AuditEventModel(Base):
    """Audit events - structured log of security and operational events."""

    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False)
    workspace_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    resource_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    resource_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )

    __table_args__ = (
        Index("idx_audit_events_tenant", "tenant_id"),
        Index("idx_audit_events_workspace", "workspace_id"),
        Index("idx_audit_events_user", "user_id"),
        Index("idx_audit_events_event_type", "event_type"),
        Index("idx_audit_events_timestamp", "timestamp"),
    )


class KnowledgebaseArticleModel(Base):
    """Knowledgebase articles - generated KB content scoped per workspace.

    Mirrors the SQLite ``knowledgebase_articles`` table (storage/sqlite.py).
    Keyed by (workspace_id, article_id); the reserved ``article_id="index"``
    row holds the index article and is upserted like any other.
    """

    __tablename__ = "knowledgebase_articles"

    workspace_id: Mapped[str] = mapped_column(
        Text, ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True
    )
    article_id: Mapped[str] = mapped_column(Text, primary_key=True)
    article_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_md: Mapped[str | None] = mapped_column(Text, nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, server_default="{}"
    )
    generated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("idx_kb_articles_workspace", "workspace_id"),
        Index("idx_kb_articles_workspace_type", "workspace_id", "article_type"),
    )


class GraphAnalysisModel(Base):
    """Cached graph-analysis results, one row per workspace.

    Mirrors the SQLite ``graph_analyses`` table (storage/sqlite.py): the
    analysis payload is stored as a JSON blob and upserted on the workspace
    primary key.
    """

    __tablename__ = "graph_analyses"

    workspace_id: Mapped[str] = mapped_column(
        Text, ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True
    )
    analysis_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )
    generated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )


class EntityModel(Base):
    """Canonical, workspace-scoped entity nodes (entity registry slice 1).

    Mirrors the SQLite ``entities`` table. The exact-match index is a PARTIAL
    unique index over active rows only (``WHERE status = 'active'``) so merged
    tombstones do not collide with live rows.
    """

    __tablename__ = "entities"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        Text, ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    entity_type: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_name: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_name: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, server_default="1.0")
    provenance: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default="{}")
    representative_memory_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="active")
    merged_into: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Embedding of the canonical/normalized name for the enterprise
    # embedding-fuzzy resolution tier (ANN over same-type, same-workspace active
    # entities). Nullable: OSS/SQLite never populates it, and the enterprise
    # backend degrades gracefully to exact+alias+create when no embedding
    # service is available. Dimensions match the memory/fragment embedding
    # columns (MEMORYLAYER_EMBEDDING_DIMENSIONS) so a single embed model serves
    # both. (Migration 023.)
    name_embedding: Mapped[list[float] | None] = mapped_column(Vector(_EMBEDDING_DIM), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("idx_entities_workspace", "workspace_id"),
        Index(
            "uq_entities_active_norm",
            "workspace_id",
            "entity_type",
            "normalized_name",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
        # HNSW cosine index over name_embedding for the embedding-fuzzy tier.
        # Mirrors idx_memories_embedding (m=16, ef_construction=64,
        # vector_cosine_ops); partial so only active, embedded rows are indexed.
        Index(
            "idx_entities_name_embedding",
            "name_embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"name_embedding": "vector_cosine_ops"},
            postgresql_where=text("name_embedding IS NOT NULL AND status = 'active'"),
        ),
    )


class EntityAliasModel(Base):
    """Alternate surface forms for a canonical entity (entity registry slice 1).

    Mirrors the SQLite ``entity_aliases`` table. Idempotent on
    (entity_id, normalized_alias).
    """

    __tablename__ = "entity_aliases"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(Text, nullable=False)
    entity_id: Mapped[str] = mapped_column(
        Text, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    alias: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_alias: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str] = mapped_column(Text, nullable=False, server_default="manual")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("entity_id", "normalized_alias", name="uq_entity_aliases_entity_norm"),
        Index("idx_entity_aliases_norm", "workspace_id", "normalized_alias"),
    )


class EntityMemberModel(Base):
    """Membership edges: a memory mentions/belongs to an entity (slice 1).

    Mirrors the SQLite ``entity_members`` table and the
    ``MemoryAssociationModel`` edge style: CASCADE FKs, named indexes, a unique
    constraint on (entity_id, memory_id, role).
    """

    __tablename__ = "entity_members"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    workspace_id: Mapped[str] = mapped_column(Text, nullable=False)
    entity_id: Mapped[str] = mapped_column(
        Text, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False
    )
    memory_id: Mapped[str] = mapped_column(
        Text, ForeignKey("memories.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(Text, nullable=False, server_default="mention")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, server_default="1.0")
    meta: Mapped[dict[str, Any]] = mapped_column("meta", JSONB, nullable=False, server_default="{}")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("entity_id", "memory_id", "role", name="uq_entity_member"),
        Index("idx_entity_members_entity", "workspace_id", "entity_id"),
        Index("idx_entity_members_memory", "workspace_id", "memory_id"),
    )
