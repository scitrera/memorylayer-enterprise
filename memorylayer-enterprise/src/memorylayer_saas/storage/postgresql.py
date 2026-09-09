"""PostgreSQL storage backend with pgvector support and LEANN cold tier."""
import asyncio
import hashlib
import json
import os
import re
import uuid

from datetime import datetime, timedelta, timezone
from logging import Logger
from pathlib import Path
from typing import Optional, Any

from scitrera_app_framework import Variables, get_extension, get_logger

from sqlalchemy import (select, delete, update, func, and_, or_, text, desc, column, bindparam, literal_column, case)
from sqlalchemy.types import Float
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import aliased, defer
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from memorylayer_server.services.storage.base import StoragePluginBase
from memorylayer_server.config import (
    DEFAULT_MEMORYLAYER_CONTEXT_EVENT_RETENTION_DAYS,
    MEMORYLAYER_CONTEXT_EVENT_RETENTION_DAYS,
)
from memorylayer_server.models.session import Session, WorkingMemory
from memorylayer_server.models.context_pack import (
    CheckpointCaptureStatus,
    CheckpointWorkStatus,
    ContextEventKind,
    SessionCheckpoint,
    SessionCheckpointInput,
    SessionContextEvent,
)
from memorylayer_server.models.entity_relation import (
    EntityRelation,
    EntityRelationEvidence,
    EntityRelationPath,
)
from memorylayer_server.models.memory import (
    MemoryMutation,
    MemoryMutationResult,
    MemoryRevision,
    MemoryStatus,
    MemoryType,
    RecallInput,
    RecallResult,
    RememberInput,
)
from memorylayer_server.models.association import (
    Association, AssociateInput, GraphQueryInput, GraphQueryResult, GraphPath
)
from memorylayer_server.models.workspace import Workspace, Context, normalize_tags
from memorylayer_server.models.versioned_resource import (
    VersionedResource,
    VersionedResourceConflictError,
    VersionedResourceMutation,
    VersionedResourceMutationResult,
    VersionedResourceNotFoundError,
    VersionedResourcePreconditionFailedError,
    VersionedResourceRevision,
)
from memorylayer_server.models.skill import Skill, SkillMutation, SkillMutationResult, SkillRevision
from memorylayer_server.services.skills.versioning import canonical_hash, manifest_etag, manifest_state
from memorylayer_server.services.memory.versioning import (
    SEMANTIC_MEMORY_FIELDS,
    memory_etag,
    memory_revision_snapshot,
    memory_semantic_state,
)
from memorylayer_server.utils import generate_id
# Use enterprise Memory model with multivector support
from ..models.memory import Memory

from .base import ColdTierStorageBackend
from .leann import LeannStorage, CSRGraph
from .models import (
    WorkspaceModel,
    ContextModel,
    MemoryModel,
    MemoryOperationModel,
    MemoryRevisionModel,
    MemoryAssociationModel,
    MemoryFragmentModel,
    CueAnchorModel,
    ContradictionModel,
    LeannGraphModel,
    LeannDocumentModel,
    SessionModel,
    SessionContextModel,
    SessionCheckpointModel,
    SessionContextEventModel,
    EntityRelationModel,
    EntityRelationEvidenceModel,
    MemoryAccessLogModel,
    DocumentModel,
    DocumentPageModel,
    IngestionJobModel,
    _EMBEDDING_DIM,
    DatasetModel,
    DatasetJobModel,
    ChatThreadModel,
    ChatMessageModel,
    UserModel,
    ApplicationModel,
    WorkspaceApplicationModel,
    CollectionItemModel,
    DataProviderModel,
    SkillModel,
    SkillFileModel,
    SkillRevisionModel,
    SkillOperationModel,
    McpServerModel,
    AuditEventModel,
    KnowledgebaseArticleModel,
    GraphAnalysisModel,
    EntityModel,
    EntityAliasModel,
    EntityMemberModel,
)

from memorylayer_server.services.contradiction.base import ContradictionRecord
from memorylayer_server.models.chat import ChatThread, ChatMessage, MessageInput

from ..models.document import (
    Document,
    DocumentEnrichmentStatus,
    IngestionJob,
    DocumentPage,
    DocumentStatus,
    JobStatus,
    DocumentExtractionOptions,
)
from ..models.dataset import Dataset, DatasetJob, DatasetColumn, DatasetFormat, DatasetProfilingOptions, DatasetStatus
from .database import Base
from .versioned_resources import PostgreSQLVersionedResourceStore

# PostgreSQL plugin configuration
MEMORYLAYER_POSTGRESQL_URL = 'MEMORYLAYER_POSTGRESQL_URL'
DEFAULT_MEMORYLAYER_POSTGRESQL_URL = 'postgresql://localhost:5432/memorylayer'

MEMORYLAYER_POSTGRESQL_POOL_SIZE = 'MEMORYLAYER_POSTGRESQL_POOL_SIZE'
DEFAULT_MEMORYLAYER_POSTGRESQL_POOL_SIZE = '10'

# Sentinel distinguishing "field not supplied" from an explicit None in
# **updates-style** methods (e.g. clearing multivector with multivector=None).
_UNSET = object()

# Name of the pg_textsearch BM25 index on memories.fts_content. Must match
# migrations/005_pg_textsearch_bm25.sql; referenced by full_text_search via
# to_bm25query(...) so parameterized queries resolve the index.
_FTS_BM25_INDEX = "memories_fts_bm25"

# Loader option that leaves ``memories.embedding`` off a MemoryModel query.
#
# Bulk read paths (search, timeline, decay, document listing) return rows whose
# vectors nobody reads: ranking already happened in SQL via pgvector, and the
# API layer strips ``embedding`` before serializing. Materializing them anyway
# was the single largest per-request allocation in the server — a dim-1920
# vector costs ~7.7 KB on the wire and as a numpy array, but ~61 KB once
# ``_memory_model_to_domain`` turns it into a Python ``list[float]`` (a float
# object is 24 bytes plus 8 for the list slot). A 200-row candidate pool
# therefore carried ~12 MB of boxed floats per in-flight request.
#
# Deferring skips the column in the SELECT entirely, so the cost disappears
# from the wire, the driver, and the heap alike. Only apply this to queries
# whose results are never asked for their embedding — see
# ``_memory_model_to_domain``, which returns ``None`` rather than triggering a
# lazy load (which would raise ``MissingGreenlet`` under asyncio anyway).
#
# Built per call rather than bound once at import: loader options are tied to
# the specific mapped class, and tests exercise these PostgreSQL code paths
# against SQLite by patching this module's ``MemoryModel`` global with a
# SQLite-compatible mapper. An import-time constant would still point at the
# original class and SQLAlchemy would reject it as not applying to the query's
# root entity.
def _defer_embedding():
    """Loader option leaving ``memories.embedding`` out of a MemoryModel query."""
    return defer(MemoryModel.embedding)


def _loaded_embedding(model) -> Optional[list[float]]:
    """Return a model's embedding as a list, or ``None`` if it wasn't loaded.

    Queries carrying :func:`_defer_embedding` leave the column unloaded.
    Touching it there would emit a lazy SELECT — which under asyncio raises
    ``MissingGreenlet`` rather than quietly doing IO — so check loaded state
    first and report the vector as absent, exactly as it is for a memory that
    genuinely has none.
    """
    if "embedding" in sa_inspect(model).unloaded:
        return None
    return list(model.embedding) if model.embedding is not None else None


# pgvector 0.4.x's ARRAY(Vector) bind processor is broken for a list-of-vectors
# (raises "expected ndim to be 1"), so multivectors cannot be written or queried
# through ORM/parameter binding. They must round-trip through an explicit
# text[] -> vector(dim)[] cast. The two helpers below are the single source of
# truth for that codec, shared by the page and memory multivector paths.

def _multivector_text_literals(multivector: list[list[float]]) -> list[str]:
    """Format a multi-vector as ``text[]`` literals for ``CAST(:mv AS vector(dim)[])``.

    Each element becomes a ``"[f0,f1,...]"`` string; bind the returned list as a
    single text-array parameter and cast it server-side (see ``update_memory`` /
    ``update_page``). Floats only, so ``repr`` is injection-safe.
    """
    return ["[" + ",".join(repr(float(x)) for x in vec) + "]" for vec in multivector]


def _vec_to_list(v) -> list[float]:
    """Coerce a stored vector/halfvec element to a plain list[float].

    pgvector returns ``HalfVector`` value objects for halfvec columns (not
    iterable), and numpy arrays / lists for vector columns. ``.to_list()``
    covers the pgvector value classes; ``list()`` covers arrays/lists.
    """
    if hasattr(v, "to_list"):
        return v.to_list()
    return list(v)


def _multivector_sql_array(multivector: list[list[float]]) -> str:
    """Format a multi-vector as an inline ``ARRAY[...]::vector(dim)[]`` SQL literal.

    Used for the query side of MaxSim where the value is embedded directly in the
    SQL text (asyncpg cannot infer the vector-array type for a bound param).
    Floats only, so the inline interpolation is injection-safe.
    """
    dim = len(multivector[0]) if multivector else 0
    inner = ",".join(
        "'[%s]'::vector(%d)" % (",".join(repr(float(x)) for x in vec), dim)
        for vec in multivector
    )
    return "ARRAY[%s]::vector(%d)[]" % (inner, dim)


class PostgreSQLBackend(ColdTierStorageBackend):
    """PostgreSQL storage backend with pgvector support and LEANN cold tier."""

    def __init__(self, v: Variables = None, connection_string: str = None, pool_size: int = 10,
                 compression_service=None):
        """
        Initialize PostgreSQL backend.

        Args:
            v: Variables instance for logger context.
            connection_string: PostgreSQL connection string (asyncpg format)
            pool_size: Connection pool size
            compression_service: Optional CompressionService for embedding-based cold tier search.
        """
        super().__init__(v)
        # Retained so instance methods can resolve sibling extensions at call
        # time (e.g. the blob-storage service in ``get_page_image_b64``).
        self._v = v
        self.logger = get_logger(v, name=self.__class__.__name__)
        self.connection_string = connection_string
        self.pool_size = pool_size
        self._engine = None
        self._session_factory = None
        self._leann_storage: Optional[LeannStorage] = None
        self._compression_service = compression_service
        self._versioned_resource_store: PostgreSQLVersionedResourceStore | None = None
        self._context_event_retention_days = (
            v.environ(
                MEMORYLAYER_CONTEXT_EVENT_RETENTION_DAYS,
                default=DEFAULT_MEMORYLAYER_CONTEXT_EVENT_RETENTION_DAYS,
                type_fn=int,
            )
            if v is not None
            else DEFAULT_MEMORYLAYER_CONTEXT_EVENT_RETENTION_DAYS
        )

    @property
    def session_factory(self):
        """Public access to the async session factory for dependent services."""
        return self._session_factory

    async def connect(self) -> None:
        """Initialize storage connection.

        Idempotent: if an engine/session_factory already exists (e.g. the
        backend was connected early so a dependent service like the AGE graph
        backend could read ``session_factory`` during sync plugin init), a
        second ``connect()`` from the framework's ``async_ready`` hook is a
        no-op rather than leaking a second engine and re-running schema setup.
        """
        if self._engine is not None and self._session_factory is not None:
            return
        self._engine = create_async_engine(
            self.connection_string,
            pool_size=self.pool_size,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        self._versioned_resource_store = PostgreSQLVersionedResourceStore(self._session_factory)

        # Schema management — three layers. The ORDER of the first two depends
        # on whether the DB is already tracked by alembic; see
        # :meth:`_schema_bootstrap_order` for why, and the tests in
        # tests/unit/test_alembic_action_decision.py that pin it.
        #
        # 1. ``Base.metadata.create_all`` — bootstraps tables that don't yet
        #    exist, using the CURRENT ORM definitions.  Cheap on subsequent
        #    runs (no-ops when tables exist).  This handles fresh installs
        #    where alembic hasn't been pre-run, AND any tables that aren't
        #    yet covered by an alembic migration.
        # 2. Alembic ``upgrade head`` — applies ALTER-style migrations against
        #    existing tables.  Required when an ORM column was added in a
        #    later release: ``create_all`` does not modify existing tables,
        #    so without this step the runtime SQLAlchemy INSERT would fail
        #    with "column does not exist" until ops manually ran alembic.
        #    On a fresh DB, we stamp alembic to ``head`` instead of upgrading
        #    — the schema already matches the latest ORM, so applying every
        #    migration would double-create.
        # 3. SQL migrations under ``storage/migrations/*.sql`` — auxiliary
        #    objects (custom functions, e.g. ``max_sim``) that aren't part
        #    of the ORM/alembic flow.  Idempotent (``CREATE … IF NOT
        #    EXISTS``).  Runs every startup.
        # Detect a genuinely-fresh DB BEFORE create_all bootstraps tables, so the
        # alembic step can tell "fresh DB (safe to stamp head)" apart from a
        # pre-existing, untracked DB. create_all does NOT add missing columns to
        # existing tables, so stamping head on a pre-existing DB would silently
        # hide schema drift (the "column ... does not exist" trap).
        from sqlalchemy import inspect as sa_inspect
        async with self._engine.connect() as conn:
            pre_tables = await conn.run_sync(lambda c: sa_inspect(c).get_table_names())
        db_was_empty = not [t for t in pre_tables if t != "alembic_version"]

        async def _create_all() -> None:
            async with self._engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

        if self._auto_migrate_enabled():
            for step in self._schema_bootstrap_order(db_was_empty):
                if step == "create_all":
                    await _create_all()
                else:
                    await self._apply_alembic_migrations(db_was_empty=db_was_empty)
        else:
            # An out-of-band migrator (the ArgoCD PreSync Job) owns the schema.
            # Do NOT silently trust it: if that Job was skipped — ArgoCD selfHeal
            # can update a Deployment's image without running Sync hooks — we
            # would run new code against an old schema and fail deep in request
            # handling with "column ... does not exist". Fail loudly at startup
            # instead.
            await self._verify_schema_at_head()
        # Auxiliary SQL objects (e.g. max_sim) are idempotent and cheap, and the
        # runtime needs them regardless of who owns the versioned schema, so they
        # run on every startup in both modes.
        await self._run_migrations()

        # Initialize LEANN storage for cold tier
        self._leann_storage = LeannStorage(session_factory=self._session_factory)

        self.logger.info("Connected to PostgreSQL database with LEANN cold tier support")

    @staticmethod
    def _resolve_alembic_paths() -> tuple[str, str]:
        """Return ``(script_location, ini_path)`` for alembic.

        Default: walk up from this file to the project root. postgresql.py lives
        at ``<root>/src/memorylayer_saas/storage/postgresql.py``, so
        ``parents[3]`` is the project root containing ``migrations/`` and
        ``alembic.ini``. Production images may lay this out differently, hence
        the ``MEMORYLAYER_ALEMBIC_DIR`` / ``MEMORYLAYER_ALEMBIC_CONFIG``
        overrides (the container sets them to /app).
        """
        default_root = Path(__file__).resolve().parents[3]
        script_location = os.environ.get(
            "MEMORYLAYER_ALEMBIC_DIR", str(default_root / "migrations"),
        )
        ini_path = os.environ.get(
            "MEMORYLAYER_ALEMBIC_CONFIG", str(default_root / "alembic.ini"),
        )
        return script_location, ini_path

    @staticmethod
    def _auto_migrate_enabled() -> bool:
        """Whether THIS process manages the schema.

        Default ``True`` (every replica bootstraps + migrates on startup), which
        is fine for a single replica but races once there are several: N pods
        run ``create_all`` and ``upgrade head`` concurrently with no lock.

        Set ``MEMORYLAYER_AUTO_MIGRATE=0`` when an out-of-band migrator owns the
        schema (the ArgoCD PreSync Job), leaving exactly one writer. Replicas
        then only VERIFY the schema is at head (see
        :meth:`_verify_schema_at_head`).
        """
        return os.environ.get("MEMORYLAYER_AUTO_MIGRATE", "1").strip().lower() not in {
            "0", "false", "no", "off",
        }

    async def _verify_schema_at_head(self) -> None:
        """Fail fast unless the DB's alembic revision is the code's head.

        Guards the gap the migration Job cannot: if the Job did not run (a
        skipped hook, a hand-rolled rollout) the pod would otherwise serve
        traffic against a stale schema and fail deep in request handling.
        """
        from alembic.config import Config
        from alembic.script import ScriptDirectory

        script_location, ini_path = self._resolve_alembic_paths()
        cfg = Config(ini_path)
        cfg.set_main_option("script_location", script_location)
        head = ScriptDirectory.from_config(cfg).get_current_head()

        async with self._engine.connect() as conn:
            result = await conn.execute(text("SELECT version_num FROM alembic_version LIMIT 1"))
            row = result.first()
        current = row[0] if row else None

        if current != head:
            raise RuntimeError(
                "MemoryLayer schema is not at head: database is at "
                f"{current or '<no alembic_version row>'}, code expects {head}. "
                "MEMORYLAYER_AUTO_MIGRATE is disabled, so this process will not "
                "migrate. Run the memorylayer-migrate Job (ArgoCD PreSync hook) "
                "against this database, or set MEMORYLAYER_AUTO_MIGRATE=1 to let "
                "the pod migrate itself.",
            )
        self.logger.info("Schema verified at alembic head %s (auto-migrate disabled)", head)

    @staticmethod
    def _schema_bootstrap_order(db_was_empty: bool) -> tuple[str, ...]:
        """Return the order to run ``create_all`` and alembic in.

        On a TRACKED DB (anything not brand-new) alembic MUST run first.
        ``create_all`` bootstraps every table in the ORM metadata — including
        tables a pending migration is about to ``op.create_table``. Running it
        first makes that migration raise ``DuplicateTableError``, which rolls
        back the ENTIRE migration (losing its ``add_column``s too), leaves
        ``alembic_version`` behind, and blocks every later revision. A restart
        cannot recover it: ``create_all`` simply re-creates the tables before
        alembic runs again. That is exactly how 032/033 wedged production —
        ``skills.revision`` and ``memories.logical_key`` stayed missing while
        alembic sat at 031.

        Running migrations first lets them create their own tables; the
        subsequent ``create_all`` is then a no-op for those, and still acts as
        the safety net for ORM models that no migration covers.

        On a genuinely FRESH DB the ORM builds the head schema directly and
        alembic only records it (``stamp``), so ``create_all`` goes first —
        there is nothing for a migration to collide with.
        """
        if db_was_empty:
            return ("create_all", "alembic")
        return ("alembic", "create_all")

    @staticmethod
    def _decide_alembic_action(version_present: bool, db_was_empty: bool, force_stamp: bool) -> str:
        """Decide the alembic action for the current DB state.

        Returns one of ``"upgrade"``, ``"stamp"``, ``"refuse"``.

          * ``alembic_version`` present -> ``upgrade`` head (apply pending deltas).
          * absent + genuinely fresh DB -> ``stamp`` head (``create_all`` built the
            head schema; no DDL needed).
          * absent + pre-existing tables -> ``refuse`` (the DB is untracked and may
            be missing newer columns; stamping head would hide that drift). Caller
            raises unless ``force_stamp`` overrides, in which case ``stamp``.
        """
        if version_present:
            return "upgrade"
        if db_was_empty:
            return "stamp"
        if force_stamp:
            return "stamp"
        return "refuse"

    async def _apply_alembic_migrations(self, db_was_empty: bool) -> None:
        """Stamp or upgrade alembic to ``head`` to keep the DB schema in sync.

        Detection (see :meth:`_decide_alembic_action`):
          * If ``alembic_version`` has a row, ``upgrade head`` applies any pending
            revisions. ``create_all`` already ran and is a no-op for existing
            tables, so the upgrade only handles ALTER-style deltas
            (``add_column``, ``create_index``) that ``create_all`` cannot.
          * If ``alembic_version`` is absent AND the DB was empty before
            ``create_all`` (a genuinely fresh install), ``stamp head`` declares
            the schema at the latest revision without re-running DDL.
          * If ``alembic_version`` is absent but the DB already had tables, we
            REFUSE (raise) rather than stamp — stamping head on a pre-existing,
            untracked DB silently hides missing columns (create_all does not ALTER
            existing tables). Set ``MEMORYLAYER_ALEMBIC_STAMP_UNTRACKED=1`` to force
            the old stamp-head behavior when the schema is known to match head.

        Alembic's CLI is sync; we run it on a worker thread via
        ``asyncio.to_thread`` so the async event loop isn't blocked while
        DDL executes.

        The migrations directory and alembic ini location can be overridden
        via ``MEMORYLAYER_ALEMBIC_DIR`` (script_location) and
        ``MEMORYLAYER_ALEMBIC_CONFIG`` (path to alembic.ini).  The defaults
        resolve to the source-tree paths relative to this file, which works
        for the dev container setup; production images should COPY the
        ``migrations/`` directory and set the env vars if the layout
        differs.
        """
        from sqlalchemy import inspect as sa_inspect

        async def _alembic_version_present() -> bool:
            """Return True iff alembic_version table exists and has a row."""
            async with self._engine.connect() as conn:
                tables = await conn.run_sync(
                    lambda sync_conn: sa_inspect(sync_conn).get_table_names()
                )
                if "alembic_version" not in tables:
                    return False
                result = await conn.execute(text("SELECT version_num FROM alembic_version LIMIT 1"))
                row = result.first()
                return row is not None

        version_present = await _alembic_version_present()

        script_location, ini_path = self._resolve_alembic_paths()

        if not Path(script_location).is_dir():
            self.logger.warning(
                "Alembic script_location %s does not exist; skipping alembic step. "
                "Set MEMORYLAYER_ALEMBIC_DIR if migrations live elsewhere.",
                script_location,
            )
            return

        # The connection string is whatever the user configured (e.g.
        # ``postgresql+asyncpg://…``).  alembic's env.py supports async via
        # ``async_engine_from_config`` so we can pass it through unchanged.
        # ``DATABASE_URL`` env var is also honoured by env.py — set both for
        # belt-and-braces.
        sqlalchemy_url = self.connection_string
        force_stamp = os.environ.get("MEMORYLAYER_ALEMBIC_STAMP_UNTRACKED", "").strip().lower() in (
            "1", "true", "yes", "on",
        )
        action = self._decide_alembic_action(
            version_present=version_present, db_was_empty=db_was_empty, force_stamp=force_stamp,
        )
        if action == "refuse":
            raise RuntimeError(
                "Database has application tables but no alembic_version row — it is not "
                "alembic-tracked. Refusing to 'stamp head' because that would silently hide "
                "missing columns (create_all does not ALTER existing tables; this is the "
                "\"column ... does not exist\" trap). Baseline it: run `alembic stamp "
                "<revision matching the current schema>` then restart so 'upgrade head' "
                "applies the rest — or set MEMORYLAYER_ALEMBIC_STAMP_UNTRACKED=1 if the "
                "schema is already known to match head."
            )

        def _run_alembic() -> None:
            # Late import: alembic is heavy (pulls SQLAlchemy plumbing), and
            # only the postgres backend needs it.
            from alembic import command
            from alembic.config import Config

            cfg = Config(ini_path) if Path(ini_path).is_file() else Config()
            # Do NOT let alembic reconfigure Python logging in-process. env.py
            # honors this attribute and skips fileConfig(alembic.ini), which
            # would otherwise reset the root logger to [logger_root] level=WARN
            # with alembic's console handler — clobbering the app's SAF INFO
            # handler and silencing every app INFO log for the life of the
            # server process.
            cfg.attributes["configure_logger"] = False
            cfg.set_main_option("script_location", script_location)
            cfg.set_main_option("sqlalchemy.url", sqlalchemy_url)
            # env.py also reads DATABASE_URL; provide it so both paths
            # converge on the same target regardless of which the
            # operator's alembic.ini references.
            os.environ.setdefault("DATABASE_URL", sqlalchemy_url)

            if action == "upgrade":
                command.upgrade(cfg, "head")
            else:  # "stamp" (fresh DB, or forced)
                command.stamp(cfg, "head")

        try:
            await asyncio.to_thread(_run_alembic)
        except Exception:  # noqa: BLE001
            # Failure to apply alembic is loud-fatal-worthy: leaving the
            # backend running with schema drift will cause every INSERT to
            # fail mysteriously (the original "column 'scope' does not
            # exist" symptom).  Log and re-raise so connect() fails and
            # the caller sees the cause.
            self.logger.exception(
                "Alembic %s failed; database schema may be out of sync with the ORM. "
                "Resolve via ``alembic upgrade head`` against the dev DB or rebuild the volume.",
                "stamp" if not version_present else "upgrade",
            )
            raise

        self.logger.info(
            "Alembic %s complete (script_location=%s)",
            "stamp head" if not version_present else "upgrade head",
            script_location,
        )

    async def _run_migrations(self) -> bool:
        """Run auxiliary SQL migrations (custom functions, etc.).

        Distinct from :meth:`_apply_alembic_migrations`: this path is for
        SQL that isn't expressible cleanly via SQLAlchemy ORM / alembic
        (e.g. user-defined functions like ``max_sim``).  Files are
        idempotent and run every startup.

        Returns:
            True if migration files were found and executed, False otherwise.
        """
        migrations_path = Path(__file__).parent / "migrations"
        if not migrations_path.exists():
            return False
        sql_files = sorted(migrations_path.glob("*.sql"))
        if not sql_files:
            return False
        # SQLAlchemy's text() uses asyncpg prepared statements, which only
        # support a single SQL statement per call. Get the raw asyncpg
        # connection to use its execute() which handles multiple statements.
        async with self._engine.connect() as conn:
            raw = (await conn.get_raw_connection()).driver_connection  # type: ignore[union-attr]
            for sql_file in sql_files:
                self.logger.info("Running migration: %s", sql_file.name)
                sql = sql_file.read_text()
                await raw.execute(sql)
        return True

    async def disconnect(self) -> None:
        """Close storage connection."""
        if self._engine:
            await self._engine.dispose()
            self._versioned_resource_store = None
            self.logger.info("Disconnected from PostgreSQL database")

    async def health_check(self) -> bool:
        """Check if storage is healthy."""
        try:
            async with self._session_factory() as session:
                result = await session.execute(select(func.count()).select_from(WorkspaceModel))
                result.scalar()
                return True
        except Exception as e:
            self.logger.error("Health check failed: %s", e)
            return False

    # Memory operations
    async def create_memory(
        self,
        workspace_id: str,
        input: RememberInput,
        embedding: Optional[list] = None,
        multivector: Optional[list] = None,
    ) -> Memory:
        """Store a new memory.

        ``embedding`` / ``multivector`` accept pre-computed vectors (e.g. from
        the document-ingest pipeline) so the caller can bypass re-embedding. The
        single-vector ``embedding`` binds through the ORM; ``multivector``
        (halfvec(dim)[]) must round-trip through the text[] -> halfvec(dim)[]
        cast — pgvector's ARRAY(Vector) bind raises "expected ndim to be 1"
        (see the codec helpers / update_memory / create_page).
        """
        # Compute content hash
        content_hash = hashlib.sha256(input.content.encode()).hexdigest()

        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        memory = Memory(
            id=memory_id,
            workspace_id=workspace_id,
            tenant_id=getattr(input, 'tenant_id', None) or 'default_tenant',
            context_id=input.context_id,
            user_id=input.user_id,
            logical_key=input.logical_key,
            content=input.content,
            content_hash=content_hash,
            type=input.type.value if input.type else MemoryType.SEMANTIC.value,
            subtype=input.subtype if input.subtype else None,
            importance=input.importance,
            tags=input.tags,
            metadata=input.metadata,
            refinement_metadata=input.refinement_metadata,
            embedding=embedding,
            multivector=multivector,
            pinned=input.pinned,
            session_id=getattr(input, "session_id", None),
            source_memory_id=getattr(input, "source_memory_id", None),
            observer_id=getattr(input, 'observer_id', None),
            subject_id=getattr(input, 'subject_id', None),
            source_document_id=getattr(input, 'source_document_id', None),
            source_page_id=getattr(input, 'source_page_id', None),
            source_dataset_id=getattr(input, 'source_dataset_id', None),
            source_thread_id=getattr(input, 'source_thread_id', None),
            event_time=getattr(input, 'event_time', None),
            created_at=now,
            updated_at=now,
        )
        result = await self.mutate_memory(
            MemoryMutation(
                action="create",
                memory=memory,
                operation_id=generate_id("op"),
                request_hash=canonical_hash(
                    {"action": "create", "state": memory_semantic_state(memory)}
                ),
                expected_etag="*",
            )
        )
        if multivector:
            return await self.update_memory(
                workspace_id, memory_id, multivector=multivector
            )
        return result.memory

    async def get_memory(
        self,
        workspace_id: str,
        memory_id: str,
        track_access: bool = True,
        include_deleted: bool = False,
    ) -> Optional[Memory]:
        """Get memory by ID."""
        conditions = [
            MemoryModel.id == memory_id,
            MemoryModel.workspace_id == workspace_id,
        ]
        if not include_deleted:
            conditions.append(MemoryModel.deleted_at.is_(None))
        async with self._session_factory() as session:
            result = await session.execute(
                select(MemoryModel).where(and_(*conditions))
            )
            memory_model = result.scalar_one_or_none()

            if not memory_model:
                return None

            if track_access:
                # Update access tracking
                memory_model.access_count += 1
                memory_model.last_accessed_at = datetime.now(timezone.utc)

            # Convert to the domain object BEFORE committing: the access-tracking
            # UPDATE expires server-managed columns (updated_at), and reading an
            # expired attribute in an async session triggers implicit sync IO,
            # raising MissingGreenlet. Convert while attributes are still loaded.
            memory = self._memory_model_to_domain(memory_model)
            if track_access:
                await session.commit()
            return memory

    async def get_memory_by_id(
        self,
        memory_id: str,
        track_access: bool = True,
        include_deleted: bool = False,
    ) -> Optional[Memory]:
        """Get memory by ID without workspace filter. Memory IDs are globally unique."""
        conditions = [MemoryModel.id == memory_id]
        if not include_deleted:
            conditions.append(MemoryModel.deleted_at.is_(None))
        async with self._session_factory() as session:
            result = await session.execute(
                select(MemoryModel).where(and_(*conditions))
            )
            memory_model = result.scalar_one_or_none()

            if not memory_model:
                return None

            if track_access:
                memory_model.access_count += 1
                memory_model.last_accessed_at = datetime.now(timezone.utc)

            # Convert BEFORE committing — see get_memory: the access-tracking
            # UPDATE expires server-managed columns and reading them in an async
            # session triggers implicit sync IO (MissingGreenlet).
            memory = self._memory_model_to_domain(memory_model)
            if track_access:
                await session.commit()
            return memory

    async def get_memories_by_ids(
        self,
        workspace_id: str,
        memory_ids: list[str],
        *,
        include_deleted: bool = False,
    ) -> list[Memory]:
        """Batch-hydrate memories in caller order without N+1 queries."""
        if not memory_ids:
            return []
        conditions = [
            MemoryModel.workspace_id == workspace_id,
            MemoryModel.id.in_(list(dict.fromkeys(memory_ids))),
        ]
        if not include_deleted:
            conditions.append(MemoryModel.deleted_at.is_(None))
        async with self._session_factory() as session:
            result = await session.execute(
                select(MemoryModel).options(_defer_embedding()).where(and_(*conditions))
            )
            by_id = {
                model.id: self._memory_model_to_domain(model)
                for model in result.scalars().all()
            }
        return [by_id[memory_id] for memory_id in memory_ids if memory_id in by_id]

    async def list_source_memories(
        self,
        workspace_id: str,
        source_memory_id: str,
    ) -> list[Memory]:
        """Return active deterministic children for one raw source memory."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(MemoryModel)
                .where(
                    and_(
                        MemoryModel.workspace_id == workspace_id,
                        MemoryModel.source_memory_id == source_memory_id,
                        MemoryModel.deleted_at.is_(None),
                    )
                )
                .order_by(MemoryModel.created_at, MemoryModel.id)
            )
            return [self._memory_model_to_domain(model) for model in result.scalars().all()]

    async def update_memory(self, workspace_id: str, memory_id: str, **updates) -> Optional[Memory]:
        """Update memory fields."""
        # multivector cannot be written through the ORM (see the codec helpers);
        # pop it and write via an explicit text[] -> vector(dim)[] cast below.
        multivector = updates.pop("multivector", _UNSET)

        if set(updates) & SEMANTIC_MEMORY_FIELDS:
            current = await self.get_memory(
                workspace_id, memory_id, track_access=False
            )
            if current is None:
                return None
            normalized = dict(updates)
            if "content" in normalized and "content_hash" not in normalized:
                normalized["content_hash"] = hashlib.sha256(
                    normalized["content"].encode()
                ).hexdigest()
            if isinstance(normalized.get("type"), str):
                normalized["type"] = MemoryType(normalized["type"])
            if isinstance(normalized.get("status"), str):
                normalized["status"] = MemoryStatus(normalized["status"])
            if "pinned" in normalized:
                normalized["pinned"] = bool(normalized["pinned"])
            normalized["updated_at"] = datetime.now(timezone.utc)
            desired = current.model_copy(update=normalized)
            result = await self.mutate_memory(
                MemoryMutation(
                    action="replace",
                    memory=desired,
                    operation_id=generate_id("op"),
                    request_hash=canonical_hash(
                        {
                            "action": "replace",
                            "state": memory_semantic_state(desired),
                            "expected_etag": current.etag,
                        }
                    ),
                    expected_etag=current.etag,
                )
            )
            if multivector is not _UNSET:
                return await self.update_memory(
                    workspace_id, memory_id, multivector=multivector
                )
            return result.memory

        async with self._session_factory() as session:
            result = await session.execute(
                select(MemoryModel).where(
                    and_(
                        MemoryModel.id == memory_id,
                        MemoryModel.workspace_id == workspace_id,
                        MemoryModel.deleted_at.is_(None),
                    )
                )
            )
            memory_model = result.scalar_one_or_none()

            if not memory_model:
                return None

            for key, value in updates.items():
                orm_key = "meta" if key == "metadata" else key
                if hasattr(memory_model, orm_key):
                    if isinstance(value, (MemoryType, MemoryStatus)):
                        value = value.value
                    setattr(memory_model, orm_key, value)

            if multivector is not _UNSET:
                if multivector is None:
                    memory_model.multivector = None
                else:
                    dim = len(multivector[0]) if multivector else 0
                    await session.flush()
                    await session.execute(
                        text(
                            "UPDATE memories SET multivector = "
                            "CAST(:mv AS halfvec(%d)[]) WHERE id = :mid" % dim
                        ),
                        {"mv": _multivector_text_literals(multivector), "mid": memory_id},
                    )

            memory_model.updated_at = datetime.now(timezone.utc)
            await self._add_context_event(
                session,
                workspace_id,
                memory_model.session_id,
                ContextEventKind.MEMORY_UPSERT.value,
                "memory",
                memory_id,
                {"revision": memory_model.revision},
            )
            await session.commit()
            await session.refresh(memory_model)

            return self._memory_model_to_domain(memory_model)

    async def delete_memory(self, workspace_id: str, memory_id: str, hard: bool = False) -> bool:
        """Soft or hard delete memory."""
        if hard:
            async with self._session_factory() as session:
                memory_result = await session.execute(
                    select(MemoryModel).where(
                        and_(
                            MemoryModel.id == memory_id,
                            MemoryModel.workspace_id == workspace_id,
                        )
                    )
                )
                memory_model = memory_result.scalar_one_or_none()
                if memory_model is None:
                    return False
                await self._deactivate_relation_evidence_in_session(
                    session, workspace_id, memory_id
                )
                await session.execute(
                    delete(MemoryOperationModel).where(
                        MemoryOperationModel.workspace_id == workspace_id,
                        MemoryOperationModel.memory_id == memory_id,
                    )
                )
                await session.execute(
                    delete(MemoryRevisionModel).where(
                        MemoryRevisionModel.workspace_id == workspace_id,
                        MemoryRevisionModel.memory_id == memory_id,
                    )
                )
                result = await session.execute(
                    delete(MemoryModel).where(
                        and_(
                            MemoryModel.id == memory_id,
                            MemoryModel.workspace_id == workspace_id,
                        )
                    )
                )
                await self._add_context_event(
                    session,
                    workspace_id,
                    memory_model.session_id,
                    ContextEventKind.MEMORY_DELETE.value,
                    "memory",
                    memory_id,
                    {"hard": True},
                )
                await session.commit()
                return result.rowcount > 0
        current = await self.get_memory(workspace_id, memory_id, track_access=False)
        if current is None:
            return False
        now = datetime.now(timezone.utc)
        desired = current.model_copy(update={"deleted_at": now, "updated_at": now})
        await self.mutate_memory(
            MemoryMutation(
                action="delete",
                memory=desired,
                operation_id=generate_id("op"),
                request_hash=canonical_hash(
                    {"action": "delete", "id": memory_id, "expected_etag": current.etag}
                ),
                expected_etag=current.etag,
            )
        )
        return True

    async def mutate_memory(self, mutation: MemoryMutation) -> MemoryMutationResult:
        """Atomically mutate a locked memory head plus revision/idempotency rows."""
        from sqlalchemy.exc import IntegrityError

        desired = mutation.memory.model_copy(deep=True)
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    replay = await self._get_memory_operation_in_session(
                        session,
                        desired.tenant_id,
                        desired.workspace_id,
                        mutation.operation_id,
                    )
                    if replay is not None:
                        return self._validate_memory_replay(
                            replay, mutation.request_hash
                        )

                    result = await session.execute(
                        select(MemoryModel)
                        .where(
                            MemoryModel.id == desired.id,
                            MemoryModel.workspace_id == desired.workspace_id,
                        )
                        .with_for_update()
                    )
                    current_model = result.scalar_one_or_none()
                    current = (
                        self._memory_model_to_domain(current_model)
                        if current_model is not None
                        else None
                    )
                    if mutation.action == "create":
                        if mutation.expected_etag != "*":
                            raise VersionedResourcePreconditionFailedError(
                                "create requires If-None-Match: *"
                            )
                        if current is not None:
                            raise VersionedResourceConflictError(
                                "memory id already exists"
                            )
                        final = desired.model_copy(
                            update={"revision": 1, "deleted_at": None}
                        )
                        if final.context_id == "_default":
                            final = final.model_copy(update={"context_id": None})
                        final = final.model_copy(
                            update={"etag": memory_etag(1, final)}
                        )
                        current_model = self._memory_domain_to_model(final)
                        session.add(current_model)
                    else:
                        if current is None or current.tenant_id != desired.tenant_id:
                            raise VersionedResourceNotFoundError("memory not found")
                        if mutation.expected_etag != current.etag:
                            raise VersionedResourcePreconditionFailedError(
                                "ETag does not match current revision"
                            )
                        if mutation.action == "restore":
                            if current.deleted_at is None:
                                raise VersionedResourceConflictError(
                                    "memory is not deleted"
                                )
                        elif current.deleted_at is not None:
                            raise VersionedResourceNotFoundError("memory not found")
                        if (
                            desired.logical_key != current.logical_key
                            or desired.user_id != current.user_id
                            or desired.workspace_id != current.workspace_id
                        ):
                            raise VersionedResourceConflictError(
                                "memory logical key and ownership are immutable"
                            )
                        final = desired.model_copy(
                            update={
                                "created_at": current.created_at,
                                "revision": current.revision + 1,
                            }
                        )
                        final = final.model_copy(
                            update={"etag": memory_etag(final.revision, final)}
                        )
                        self._apply_memory_domain(current_model, final)

                    snapshot = memory_revision_snapshot(final).model_dump(mode="json")
                    snapshot["multivector"] = None
                    session.add(
                        MemoryRevisionModel(
                            tenant_id=final.tenant_id,
                            workspace_id=final.workspace_id,
                            memory_id=final.id,
                            revision=final.revision,
                            snapshot=snapshot,
                            action=mutation.action,
                            operation_id=mutation.operation_id,
                            request_hash=mutation.request_hash,
                        )
                    )
                    session.add(
                        MemoryOperationModel(
                            tenant_id=final.tenant_id,
                            workspace_id=final.workspace_id,
                            operation_id=mutation.operation_id,
                            request_hash=mutation.request_hash,
                            memory_id=final.id,
                            revision=final.revision,
                        )
                    )
                    if mutation.action in ("replace", "delete"):
                        await self._deactivate_relation_evidence_in_session(
                            session, final.workspace_id, final.id
                        )
                    await self._add_context_event(
                        session,
                        final.workspace_id,
                        final.session_id,
                        (
                            ContextEventKind.MEMORY_DELETE.value
                            if mutation.action == "delete"
                            else ContextEventKind.MEMORY_UPSERT.value
                        ),
                        "memory",
                        final.id,
                        {"revision": final.revision, "action": mutation.action},
                    )
                    await session.flush()
                    return MemoryMutationResult(memory=final)
        except IntegrityError as exc:
            replay = await self.get_memory_operation(
                desired.tenant_id,
                desired.workspace_id,
                mutation.operation_id,
                mutation.request_hash,
            )
            if replay is not None:
                return replay
            if mutation.action == "create":
                raise VersionedResourceConflictError(
                    "memory id or scoped logical_key already exists"
                ) from exc
            raise

    async def get_memory_operation(
        self,
        tenant_id: str,
        workspace_id: str,
        operation_id: str,
        request_hash: str,
    ) -> MemoryMutationResult | None:
        async with self._session_factory() as session:
            revision = await self._get_memory_operation_in_session(
                session, tenant_id, workspace_id, operation_id
            )
        if revision is None:
            return None
        return self._validate_memory_replay(revision, request_hash)

    async def _get_memory_operation_in_session(
        self,
        session,
        tenant_id: str,
        workspace_id: str,
        operation_id: str,
    ) -> MemoryRevision | None:
        result = await session.execute(
            select(MemoryRevisionModel)
            .join(
                MemoryOperationModel,
                and_(
                    MemoryRevisionModel.tenant_id == MemoryOperationModel.tenant_id,
                    MemoryRevisionModel.workspace_id == MemoryOperationModel.workspace_id,
                    MemoryRevisionModel.memory_id == MemoryOperationModel.memory_id,
                    MemoryRevisionModel.revision == MemoryOperationModel.revision,
                ),
            )
            .where(
                MemoryOperationModel.tenant_id == tenant_id,
                MemoryOperationModel.workspace_id == workspace_id,
                MemoryOperationModel.operation_id == operation_id,
            )
        )
        model = result.scalar_one_or_none()
        return self._memory_revision_to_domain(model) if model is not None else None

    @staticmethod
    def _validate_memory_replay(
        revision: MemoryRevision,
        request_hash: str,
    ) -> MemoryMutationResult:
        if revision.request_hash != request_hash:
            raise VersionedResourceConflictError(
                "idempotency key was already used for a different request"
            )
        return MemoryMutationResult(memory=revision.memory, replayed=True)

    async def list_memory_revisions(
        self,
        tenant_id: str,
        workspace_id: str,
        memory_id: str,
        *,
        limit: int,
        before_sequence: int | None = None,
    ) -> list[MemoryRevision]:
        conditions = [
            MemoryRevisionModel.tenant_id == tenant_id,
            MemoryRevisionModel.workspace_id == workspace_id,
            MemoryRevisionModel.memory_id == memory_id,
        ]
        if before_sequence is not None:
            conditions.append(MemoryRevisionModel.sequence < before_sequence)
        async with self._session_factory() as session:
            result = await session.execute(
                select(MemoryRevisionModel)
                .where(and_(*conditions))
                .order_by(MemoryRevisionModel.sequence.desc())
                .limit(limit)
            )
            return [
                self._memory_revision_to_domain(model)
                for model in result.scalars().all()
            ]

    async def search_memories(
            self,
            workspace_id: str,
            query_embedding: list[float],
            limit: int = 10,
            offset: int = 0,
            min_relevance: float = 0.5,
            types: Optional[list[str]] = None,
            subtypes: Optional[list[str]] = None,
            tags: Optional[list[str]] = None,
            include_archived: bool = False,
            observer_id: Optional[str] = None,
            subject_id: Optional[str] = None,
            created_after: Optional[str] = None,
            created_before: Optional[str] = None,
            user_id: Optional[str] = None,
    ) -> list[tuple[Memory, float]]:
        """Vector similarity search using pgvector."""
        async with self._session_factory() as session:
            # Build query
            query = select(
                MemoryModel,
                (1 - MemoryModel.embedding.cosine_distance(query_embedding)).label("relevance")
            ).where(
                and_(
                    MemoryModel.workspace_id == workspace_id,
                    MemoryModel.deleted_at.is_(None),
                    MemoryModel.embedding.isnot(None),
                )
            )

            # Entity attribution filters
            if observer_id is not None:
                query = query.where(MemoryModel.observer_id == observer_id)
            if subject_id is not None:
                query = query.where(MemoryModel.subject_id == subject_id)
            if user_id is not None:
                # NULL user_id = workspace-shared (visible to every member); a
                # non-NULL user_id keeps a memory private to that user. Match the
                # caller's own memories OR the workspace-shared ones — otherwise
                # document-ingested memories (user_id NULL) are never recalled
                # while preserving the cross-user boundary on user-scoped memories.
                query = query.where(
                    or_(MemoryModel.user_id == user_id, MemoryModel.user_id.is_(None))
                )

            # Apply filters
            if types:
                query = query.where(MemoryModel.type.in_(types))
            if subtypes:
                query = query.where(MemoryModel.subtype.in_(subtypes))
            if tags:
                for tag in tags:
                    query = query.where(MemoryModel.tags.contains([tag]))

            # Date range filters
            if created_after is not None:
                query = query.where(MemoryModel.created_at >= created_after)
            if created_before is not None:
                query = query.where(MemoryModel.created_at <= created_before)

            # Order by relevance and apply limits
            query = query.order_by(desc("relevance")).limit(limit).offset(offset)

            query = query.options(_defer_embedding())
            result = await session.execute(query)
            rows = result.all()

            # Filter by minimum relevance and convert to domain models
            results = []
            for memory_model, relevance in rows:
                if relevance >= min_relevance:
                    memory = self._memory_model_to_domain(memory_model)
                    results.append((memory, float(relevance)))

            return results

    async def search_memories_by_entities(
            self,
            workspace_id: str,
            query_embedding: list[float],
            entities: list[str],
            limit: int = 10,
    ) -> list[tuple[Memory, float]]:
        """Entity-anchored vector search (see StorageBackend.search_memories_by_entities).

        Restrict the candidate set to active, embedded memories whose
        ``meta->>'speaker'`` is one of ``entities`` (the speaker is the
        discriminating, validated signal — see the recall-side note in
        MemoryService._fuse_entity_results; the broader ``meta->'entities'``
        mention set dilutes recall), then rank that set by vector distance to
        ``query_embedding`` (``embedding <=> query``, ascending = best-first).
        Powers the entity-anchored RRF fusion channel.
        """
        if not entities:
            return []

        async with self._session_factory() as session:
            stmt = (
                select(
                    MemoryModel,
                    (1 - MemoryModel.embedding.cosine_distance(query_embedding)).label("relevance"),
                )
                .where(
                    and_(
                        MemoryModel.workspace_id == workspace_id,
                        MemoryModel.deleted_at.is_(None),
                        MemoryModel.embedding.isnot(None),
                        text("metadata->>'speaker' = ANY(:entities)"),
                    )
                )
                .order_by(MemoryModel.embedding.cosine_distance(query_embedding).asc())
                .limit(limit)
            ).params(entities=list(entities))

            stmt = stmt.options(_defer_embedding())
            result = await session.execute(stmt)
            rows = result.all()

            return [(self._memory_model_to_domain(m), float(relevance)) for m, relevance in rows]

    async def store_cue_anchors(
            self,
            workspace_id: str,
            memory_id: str,
            cues: list[dict],
    ) -> None:
        """Persist structured cue-anchor rows for a memory (cue retrieval channel).

        Inserts one ``cue_anchors`` row per cue anchor generated for ``memory_id``
        (see StorageBackend.store_cue_anchors). Each cue dict carries
        ``{"cue": str, "embedding": list[float], "entity_id": str | None}``; the
        optional ``entity_id`` links the cue to the canonical Entity it names.
        Blank cue texts are skipped. Called at ingest behind the cue-channel flag;
        ships DARK (no rows written when the channel is off).
        """
        if not cues:
            return None

        from memorylayer_server.utils import generate_id

        now = datetime.now(timezone.utc)
        async with self._session_factory() as session:
            for cue in cues:
                normalized = (cue.get("cue") or "").strip()
                if not normalized:
                    continue
                session.add(
                    CueAnchorModel(
                        id=generate_id("cue"),
                        workspace_id=workspace_id,
                        memory_id=memory_id,
                        cue=normalized,
                        normalized_cue=self._normalize_alias(normalized),
                        entity_id=cue.get("entity_id"),
                        embedding=cue.get("embedding"),
                        created_at=now,
                    )
                )
            await session.commit()
        self.logger.debug("Stored %d cue anchor(s) for memory %s", len(cues), memory_id)
        return None

    async def search_cue_anchors(
            self,
            workspace_id: str,
            query_embedding: list[float],
            limit: int,
    ) -> list[tuple[Memory, float]]:
        """Cue-anchor vector search, dereferenced to primary memories (best-first).

        Searches ``cue_anchors.embedding`` by cosine distance to
        ``query_embedding``, collapses to one row per parent memory keeping the
        BEST (nearest) cue via ``MIN`` + ``GROUP BY memory_id``, then joins back to
        the ``memories`` row (active, not-deleted) and returns ``(memory,
        relevance)`` tuples ranked best-first (``relevance = 1 - distance``). Powers
        the Memora-inspired RRF cue fusion channel
        (MemoryService._fuse_cue_results). Uses the same cosine distance operator
        (``<=>`` via SQLAlchemy ``cosine_distance``) as the other vector arms.
        """
        async with self._session_factory() as session:
            distance = CueAnchorModel.embedding.cosine_distance(query_embedding)
            # Collapse multiple cues per memory to the single best (min) distance.
            # entity_id is aggregated through so the canonical-entity link is
            # queryable on the returned rows (callers do not consume it yet;
            # cross-arm entity boosting is a follow-up).
            best = (
                select(
                    CueAnchorModel.memory_id.label("memory_id"),
                    func.min(distance).label("distance"),
                    func.min(CueAnchorModel.entity_id).label("entity_id"),
                )
                .where(
                    and_(
                        CueAnchorModel.workspace_id == workspace_id,
                        CueAnchorModel.embedding.isnot(None),
                    )
                )
                .group_by(CueAnchorModel.memory_id)
                .subquery()
            )

            stmt = (
                select(MemoryModel, (1 - best.c.distance).label("relevance"))
                .join(best, best.c.memory_id == MemoryModel.id)
                .where(
                    and_(
                        MemoryModel.workspace_id == workspace_id,
                        MemoryModel.deleted_at.is_(None),
                        MemoryModel.status == "active",
                    )
                )
                .order_by(best.c.distance.asc())
                .limit(limit)
            )

            stmt = stmt.options(_defer_embedding())
            result = await session.execute(stmt)
            rows = result.all()

            return [(self._memory_model_to_domain(m), float(relevance)) for m, relevance in rows]

    async def expand_via_cues(
            self,
            workspace_id: str,
            memory_ids: list[str],
            *,
            limit: int = 20,
            threshold: float = 0.85,
    ) -> list[tuple[Memory, float]]:
        """Reach OTHER memories sharing a thematic cue with ``memory_ids`` (best-first).

        The "relaxed frontier" source for agentic EXPAND (see
        StorageBackend.expand_via_cues). Self-joins ``cue_anchors``: for every cue
        of an input memory (``src``), finds cue anchors of DIFFERENT memories
        (``tgt``) within cosine distance ``<= (1 - threshold)``, collapses to one
        row per target memory keeping the BEST (nearest) cue via ``MIN`` +
        ``GROUP BY memory_id``, joins back to the active/non-deleted ``memories``
        row, and returns ``(memory, relevance)`` tuples ranked best-first
        (``relevance = 1 - distance``), capped at ``limit``. Uses the same cosine
        operator (``<=>`` via ``cosine_distance``) as the other vector arms.
        """
        if not memory_ids:
            return []

        async with self._session_factory() as session:
            src = aliased(CueAnchorModel)
            tgt = aliased(CueAnchorModel)
            distance = tgt.embedding.cosine_distance(src.embedding)

            # Self-join: source cues (input memories) -> target cues (other
            # memories) within the cosine radius. Collapse to the best distance
            # per target memory.
            best = (
                select(
                    tgt.memory_id.label("memory_id"),
                    func.min(distance).label("distance"),
                )
                .select_from(src)
                .join(
                    tgt,
                    and_(
                        tgt.workspace_id == workspace_id,
                        tgt.embedding.isnot(None),
                        tgt.memory_id.notin_(memory_ids),
                    ),
                )
                .where(
                    and_(
                        src.workspace_id == workspace_id,
                        src.memory_id.in_(memory_ids),
                        src.embedding.isnot(None),
                        distance <= (1 - threshold),
                    )
                )
                .group_by(tgt.memory_id)
                .subquery()
            )

            stmt = (
                select(MemoryModel, (1 - best.c.distance).label("relevance"))
                .join(best, best.c.memory_id == MemoryModel.id)
                .where(
                    and_(
                        MemoryModel.workspace_id == workspace_id,
                        MemoryModel.deleted_at.is_(None),
                        MemoryModel.status == "active",
                    )
                )
                .order_by(best.c.distance.asc())
                .limit(limit)
            )

            stmt = stmt.options(_defer_embedding())
            result = await session.execute(stmt)
            rows = result.all()

            return [(self._memory_model_to_domain(m), float(relevance)) for m, relevance in rows]

    async def full_text_search(
            self,
            workspace_id: str,
            query: str,
            limit: int = 10,
            offset: int = 0,
            context_id: str | None = None,
    ) -> list[Memory]:
        """Full-text search using pg_textsearch BM25 (indexed), ranked best-first.

        Replaces the legacy unindexed to_tsvector / ts_rank_cd path. The BM25
        index lives on the generated ``fts_content`` column (content + folded
        metadata aliases — see migrations/005_pg_textsearch_bm25.sql), so this
        mirrors the OSS SQLite FTS arm: only documents that actually match the
        query terms are returned (the inverted index excludes non-matches), and
        BM25 relevance ordering feeds a meaningful rank into hybrid Reciprocal
        Rank Fusion (see MemoryService._fuse_keyword_results).

        The ``<@>`` operator returns negative BM25 scores (lower = better), so
        results are ordered ascending; ORDER BY ... LIMIT enables pg_textsearch's
        Block-Max WAND top-k optimization.

        The query text is wrapped in ``to_bm25query(text, index_name)`` rather
        than passed as a bare ``<@>`` right operand: pg_textsearch only auto-
        detects the BM25 index for a *literal* constant, and the SQLAlchemy/
        asyncpg path binds it as a parameter ($1). The explicit form names the
        index so parameterized (prepared) queries resolve it correctly.
        """
        if not query or not query.strip():
            return []

        async with self._session_factory() as session:
            # fts_content is a generated column not mapped on the ORM model, so
            # reference it via a raw column expression.
            bm25_rank = column("fts_content").op("<@>", return_type=Float)(
                func.to_bm25query(bindparam("bm25_query", query), _FTS_BM25_INDEX)
            )

            conditions = [
                MemoryModel.workspace_id == workspace_id,
                MemoryModel.deleted_at.is_(None),
            ]
            # RPG canonical and overlay graphs are isolated by context row ID.
            if context_id is not None:
                conditions.append(MemoryModel.context_id == context_id)

            query_sql = (
                select(MemoryModel)
                .where(and_(*conditions))
                .order_by(bm25_rank.asc())
                .limit(limit)
                .offset(offset)
            )

            query_sql = query_sql.options(_defer_embedding())
            result = await session.execute(query_sql)
            memory_models = result.scalars().all()

            return [self._memory_model_to_domain(m) for m in memory_models]

    async def get_timeline(
        self,
        workspace_id: str,
        event_after: str | None = None,
        event_before: str | None = None,
        ascending: bool = True,
        limit: int = 50,
        offset: int = 0,
        types: list[str] | None = None,
        include_archived: bool = False,
    ) -> list[Memory]:
        """Return memories ordered by effective event time (event_time or created_at).

        Mirrors the OSS contract: COALESCE(event_time, created_at) is the timeline
        position, so undated memories fall back to creation time. event_after/
        event_before are canonical UTC ISO strings (parsed to timestamps here so
        asyncpg binds them as timestamps, not text).
        """
        effective = func.coalesce(MemoryModel.event_time, MemoryModel.created_at)
        conditions = [MemoryModel.workspace_id == workspace_id, MemoryModel.deleted_at.is_(None)]
        if not include_archived:
            conditions.append(MemoryModel.status == "active")
        if types:
            conditions.append(MemoryModel.type.in_(types))
        if event_after is not None:
            conditions.append(effective >= datetime.fromisoformat(event_after))
        if event_before is not None:
            conditions.append(effective <= datetime.fromisoformat(event_before))

        order = effective.asc() if ascending else effective.desc()
        async with self._session_factory() as session:
            query_sql = select(MemoryModel).where(and_(*conditions)).order_by(order).limit(limit).offset(offset)
            query_sql = query_sql.options(_defer_embedding())
            result = await session.execute(query_sql)
            return [self._memory_model_to_domain(m) for m in result.scalars().all()]

    async def get_recent_memories(
            self,
            workspace_id: str,
            created_after: datetime,
            limit: int = 10,
            detail_level: str = "abstract",
            offset: int = 0,
    ) -> list:
        """Get recent memories ordered by creation time (newest first).

        Args:
            workspace_id: Workspace boundary.
            created_after: Only return memories created after this time.
            limit: Maximum number of memories to return.
            detail_level: Level of detail - "abstract", "overview", or "full".
            offset: Number of memories to skip (for pagination).

        Returns:
            List of dicts with memory data, newest first.
        """
        async with self._session_factory() as session:
            query = (
                select(MemoryModel)
                .where(
                    and_(
                        MemoryModel.workspace_id == workspace_id,
                        MemoryModel.created_at > created_after,
                        MemoryModel.deleted_at.is_(None),
                    )
                )
                .order_by(MemoryModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )

            result = await session.execute(query)
            memory_models = result.scalars().all()

            results = []
            for m in memory_models:
                entry = {
                    "id": m.id,
                    "workspace_id": m.workspace_id,
                    "type": m.type,
                    "importance": m.importance,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                    "tags": m.tags or [],
                }
                if detail_level in ("abstract", "overview", "full"):
                    entry["abstract"] = m.abstract if hasattr(m, "abstract") else None
                if detail_level in ("overview", "full"):
                    entry["overview"] = m.overview if hasattr(m, "overview") else None
                if detail_level == "full":
                    entry["content"] = m.content
                results.append(entry)

            return results

    async def search_memories_multivector(
            self,
            workspace_id: str,
            query_multivector: list[list[float]],
            limit: int = 10,
            min_relevance: float = 0.0,
            types: Optional[list[str]] = None,
            subtypes: Optional[list[str]] = None,
            tags: Optional[list[str]] = None,
            observer_id: Optional[str] = None,
            subject_id: Optional[str] = None,
    ) -> list[tuple[Memory, float]]:
        """Vector similarity search using MaxSim for multi-vector embeddings.

        Uses the max_sim() function from pgvector for late interaction scoring.
        MaxSim computes the maximum similarity between query vectors and document vectors,
        which is the core mechanism for ColPali-style retrieval.

        Args:
            workspace_id: Workspace identifier.
            query_multivector: List of query vectors (e.g., from ColPali query encoder).
            limit: Maximum number of results to return.
            min_relevance: Minimum relevance score threshold (0.0-1.0).
            types: Filter by memory types.
            subtypes: Filter by memory subtypes.
            tags: Filter by tags (AND logic).
            observer_id: Filter by observer entity.
            subject_id: Filter by subject entity.

        Returns:
            List of (memory, relevance_score) tuples sorted by relevance descending.
        """
        if not query_multivector:
            return []

        async with self._session_factory() as session:
            # Build query using max_sim function for late interaction scoring.
            # max_sim computes: sum over query vectors of max similarity to any
            # document vector. The query multivector is inlined as a vector(dim)[]
            # SQL literal because asyncpg can't bind a list-of-vectors param.
            query_array_sql = _multivector_sql_array(query_multivector)
            # multivector is stored as halfvec[]; cast the query to halfvec[] so
            # the max_sim(halfvec[], halfvec[]) overload is used.
            relevance = literal_column(
                "max_sim(multivector, (%s)::halfvec(128)[])" % query_array_sql
            ).label("relevance")
            query = select(
                MemoryModel,
                relevance,
            ).where(
                and_(
                    MemoryModel.workspace_id == workspace_id,
                    MemoryModel.deleted_at.is_(None),
                    MemoryModel.multivector.isnot(None),
                )
            )

            # Apply filters
            if types:
                query = query.where(MemoryModel.type.in_(types))
            if subtypes:
                query = query.where(MemoryModel.subtype.in_(subtypes))
            if tags:
                for tag in tags:
                    query = query.where(MemoryModel.tags.contains([tag]))

            # Entity attribution filters
            if observer_id is not None:
                query = query.where(MemoryModel.observer_id == observer_id)
            if subject_id is not None:
                query = query.where(MemoryModel.subject_id == subject_id)

            # Order by relevance and apply limits
            query = query.order_by(desc("relevance")).limit(limit)

            query = query.options(_defer_embedding())
            result = await session.execute(query)
            rows = result.all()

            # Filter by minimum relevance and convert to domain models
            results = []
            for memory_model, relevance in rows:
                if relevance >= min_relevance:
                    memory = self._memory_model_to_domain(memory_model)
                    results.append((memory, float(relevance)))

            return results

    async def index_memory_multivector(
            self,
            workspace_id: str,
            memory_id: str,
            multivector: list[list[float]] | None,
    ) -> None:
        """Populate the ColBERT token-flatten index for a memory's multivector.

        Stores one row per token-vector in ``memory_multivector_tokens`` (the
        HNSW-indexed satellite table from 006_multivector_token_index.sql),
        enabling indexed two-stage MaxSim retrieval. Replaces any tokens already
        stored for the memory; an empty/None multivector just clears them.
        """
        async with self._session_factory() as session:
            await session.execute(
                text("DELETE FROM memory_multivector_tokens WHERE memory_id = :mid"),
                {"mid": memory_id},
            )
            if multivector:
                literals = _multivector_text_literals(multivector)
                await session.execute(
                    text(
                        "INSERT INTO memory_multivector_tokens "
                        "(memory_id, workspace_id, token_index, embedding) "
                        "VALUES (:mid, :ws, :idx, CAST(:emb AS halfvec(128)))"
                    ),
                    [
                        {"mid": memory_id, "ws": workspace_id, "idx": i, "emb": lit}
                        for i, lit in enumerate(literals)
                    ],
                )
            await session.commit()

    async def search_memories_multivector_indexed(
            self,
            workspace_id: str,
            query_multivector: list[list[float]],
            limit: int = 10,
            min_relevance: float = 0.0,
            tokens_per_query: int = 20,
            ann_precision: str = "full",
            types: Optional[list[str]] = None,
            subtypes: Optional[list[str]] = None,
            tags: Optional[list[str]] = None,
            observer_id: Optional[str] = None,
            subject_id: Optional[str] = None,
    ) -> list[tuple[Memory, float]]:
        """Indexed late-interaction (ColBERT) retrieval — two stages.

        Stage 1: for each query token-vector, ANN-search the HNSW token index to
        gather candidate memory_ids (the index prunes the corpus to a small set).
        Stage 2: exact ``max_sim`` rerank over only those candidates' full
        multivectors. Approximates brute-force MaxSim
        (:meth:`search_memories_multivector`) at a fraction of the scan cost, so
        it scales with corpus size.

        ``tokens_per_query`` is the per-query-token ANN candidate pool (recall vs
        cost dial). Token vectors are stored as halfvec (16-bit). ``ann_precision``
        selects the stage-1 candidate index:
          - default ("half"): halfvec HNSW (006) on the stored halfvec tokens.
          - "bit": binary-quantized Hamming index (008) — 32x smaller candidate
            index, lossier gen; the exact halfvec rerank recovers quality.
        Stage-2 exact MaxSim rerank always runs on the stored halfvec vectors.
        The query multivector is inlined as a vector(128)[] literal (asyncpg
        can't bind a list-of-vectors param).
        """
        if not query_multivector:
            return []

        query_array_sql = _multivector_sql_array(query_multivector)
        # Stage-1 ANN distance. Token vectors are stored as halfvec (006/009):
        #   default -> halfvec cosine on the halfvec HNSW (cast the query token).
        #   bit     -> binary_quantize + Hamming (008 index): 32x smaller candidate
        #              index, very lossy gen; the exact rerank recovers quality.
        if ann_precision == "bit":
            ann_distance = (
                "binary_quantize(embedding::vector(128))::bit(128) "
                "<~> binary_quantize(q.qv)::bit(128)"
            )
        else:
            ann_distance = "embedding <=> q.qv::halfvec(128)"

        async with self._session_factory() as session:
            # Stage 1: candidate memory_ids via per-query-token HNSW ANN.
            cand_rows = await session.execute(
                text(
                    "SELECT DISTINCT t.memory_id "
                    "FROM unnest(%s) AS q(qv) "
                    "CROSS JOIN LATERAL ("
                    "  SELECT memory_id FROM memory_multivector_tokens "
                    "  WHERE workspace_id = :ws "
                    "  ORDER BY %s LIMIT :pool"
                    ") AS t" % (query_array_sql, ann_distance)
                ),
                {"ws": workspace_id, "pool": tokens_per_query},
            )
            candidate_ids = [r[0] for r in cand_rows.all()]
            if not candidate_ids:
                return []

            # Stage 2: exact MaxSim rerank over the candidate set only. multivector
            # is stored as halfvec[]; cast the query to halfvec[] for the
            # max_sim(halfvec[], halfvec[]) overload.
            relevance = literal_column(
                "max_sim(multivector, (%s)::halfvec(128)[])" % query_array_sql
            ).label("relevance")
            query = select(
                MemoryModel,
                relevance,
            ).where(
                and_(
                    MemoryModel.id.in_(candidate_ids),
                    MemoryModel.workspace_id == workspace_id,
                    MemoryModel.deleted_at.is_(None),
                    MemoryModel.multivector.isnot(None),
                )
            )
            if types:
                query = query.where(MemoryModel.type.in_(types))
            if subtypes:
                query = query.where(MemoryModel.subtype.in_(subtypes))
            if tags:
                for tag in tags:
                    query = query.where(MemoryModel.tags.contains([tag]))
            if observer_id is not None:
                query = query.where(MemoryModel.observer_id == observer_id)
            if subject_id is not None:
                query = query.where(MemoryModel.subject_id == subject_id)
            query = query.order_by(desc("relevance")).limit(limit)

            query = query.options(_defer_embedding())
            result = await session.execute(query)
            results = []
            for memory_model, relevance in result.all():
                if relevance >= min_relevance:
                    results.append((self._memory_model_to_domain(memory_model), float(relevance)))
            return results

    # Association operations
    async def create_association(self, workspace_id: str, input: AssociateInput) -> Association:
        """Create graph edge between memories."""
        association_id = f"assoc_{uuid.uuid4().hex[:12]}"

        association_model = MemoryAssociationModel(
            id=association_id,
            workspace_id=workspace_id,
            source_id=input.source_id,
            target_id=input.target_id,
            # AssociateInput.relationship is a str (e.g. auto_associate's
            # "similar_to"); tolerate an enum too. Parity with the sqlite backend,
            # which stores input.relationship directly.
            relation_type=getattr(input.relationship, "value", input.relationship),
            strength=input.strength,
            meta=input.metadata,
            created_at=datetime.now(timezone.utc),
        )

        async with self._session_factory() as session:
            session.add(association_model)
            await session.commit()
            await session.refresh(association_model)

        return self._association_model_to_domain(association_model)

    async def get_associations(
            self,
            workspace_id: str,
            memory_id: str,
            direction: str = "both",
            relationships: Optional[list[str]] = None,
    ) -> list[Association]:
        """Get associations for a memory."""
        async with self._session_factory() as session:
            # Build query based on direction
            conditions = [MemoryAssociationModel.workspace_id == workspace_id]

            if direction == "outgoing":
                conditions.append(MemoryAssociationModel.source_id == memory_id)
            elif direction == "incoming":
                conditions.append(MemoryAssociationModel.target_id == memory_id)
            else:  # both
                conditions.append(
                    or_(
                        MemoryAssociationModel.source_id == memory_id,
                        MemoryAssociationModel.target_id == memory_id,
                    )
                )

            if relationships:
                conditions.append(MemoryAssociationModel.relation_type.in_(relationships))

            query = select(MemoryAssociationModel).where(and_(*conditions))
            result = await session.execute(query)
            association_models = result.scalars().all()

            return [self._association_model_to_domain(a) for a in association_models]

    async def get_associations_batch(
            self,
            workspace_id: str,
            memory_ids: list[str],
            direction: str = "outgoing",
            relationships: Optional[list[str]] = None,
    ) -> list[Association]:
        """Get associations for multiple memories in a single set-based query.

        Overrides the base N+1 loop with one IN-based query per chunk (chunks
        of 30 000 ids to stay under parameter limits).  Results are deduplicated
        by association id — direction="both" can match the same edge twice (once
        as source, once as target) so we collapse those before returning.

        Semantics are identical to the per-id ``get_associations`` loop:
        - workspace_id scoping
        - direction outgoing / incoming / both
        - optional relationships filter on the ``relation_type`` column
        """
        if not memory_ids:
            return []

        _CHUNK = 30_000
        seen: set[str] = set()
        result_list: list[Association] = []

        async with self._session_factory() as session:
            for chunk_start in range(0, len(memory_ids), _CHUNK):
                chunk = memory_ids[chunk_start: chunk_start + _CHUNK]

                conditions = [MemoryAssociationModel.workspace_id == workspace_id]

                if direction == "outgoing":
                    conditions.append(MemoryAssociationModel.source_id.in_(chunk))
                elif direction == "incoming":
                    conditions.append(MemoryAssociationModel.target_id.in_(chunk))
                else:  # both
                    conditions.append(
                        or_(
                            MemoryAssociationModel.source_id.in_(chunk),
                            MemoryAssociationModel.target_id.in_(chunk),
                        )
                    )

                if relationships:
                    conditions.append(MemoryAssociationModel.relation_type.in_(relationships))

                query = select(MemoryAssociationModel).where(and_(*conditions))
                db_result = await session.execute(query)
                for model in db_result.scalars().all():
                    if model.id not in seen:
                        seen.add(model.id)
                        result_list.append(self._association_model_to_domain(model))

        return result_list

    async def delete_association(self, workspace_id: str, association_id: str) -> bool:
        """Delete an association by ID."""
        async with self._session_factory() as session:
            result = await session.execute(
                delete(MemoryAssociationModel).where(
                    and_(
                        MemoryAssociationModel.id == association_id,
                        MemoryAssociationModel.workspace_id == workspace_id,
                    )
                )
            )
            await session.commit()
            return result.rowcount > 0

    async def update_association(
            self,
            workspace_id: str,
            association_id: str,
            *,
            strength: float | None = None,
            metadata: dict | None = None,
    ) -> bool:
        """Update an existing association's strength and/or metadata.

        Only ``strength`` and ``metadata`` are updatable; ``relationship``
        (relation_type) is immutable to preserve graph semantics.

        Watermark visibility: ``get_workspace_change_watermark`` tracks
        ``max(memory.updated_at)`` — the association table has no ``updated_at``
        column, so an edge-only update is invisible to the KB skip-generate gate
        and the AGE materialize gate, which would then serve stale edge data.
        To make this change watermark-visible we bump ``updated_at`` on the
        association's source AND target memories immediately after the edge write,
        within the same transaction. This is safe because Phase 1 fix 1.6
        decoupled the recency boost from ``updated_at``; bumping it no longer
        affects relevance scoring.
        """
        if strength is None and metadata is None:
            return False

        async with self._session_factory() as session:
            # Fetch the edge first so we know source/target for the watermark bump.
            row = await session.execute(
                select(MemoryAssociationModel).where(
                    and_(
                        MemoryAssociationModel.id == association_id,
                        MemoryAssociationModel.workspace_id == workspace_id,
                    )
                )
            )
            assoc = row.scalar_one_or_none()
            if assoc is None:
                return False

            # Build the SET clause for the edge update.
            edge_values: dict = {}
            if strength is not None:
                edge_values["strength"] = strength
            if metadata is not None:
                edge_values["meta"] = metadata

            await session.execute(
                update(MemoryAssociationModel)
                .where(
                    and_(
                        MemoryAssociationModel.id == association_id,
                        MemoryAssociationModel.workspace_id == workspace_id,
                    )
                )
                .values(**edge_values)
            )

            # Bump updated_at on source + target memories so this change is
            # visible to the KB/AGE change-watermark (see docstring above).
            now = datetime.now(timezone.utc)
            memory_ids = list({assoc.source_id, assoc.target_id})
            await session.execute(
                update(MemoryModel)
                .where(MemoryModel.id.in_(memory_ids))
                .values(updated_at=now)
            )

            await session.commit()
            self.logger.debug("Updated association: %s", association_id)
            return True

    async def traverse_graph(
        self,
        workspace_id: str,
        start_id: str,
        max_depth: int = 3,
        relationships: Optional[list[str]] = None,
        direction: str = "both",
    ) -> GraphQueryResult:
        """Multi-hop traversal with direction semantics matching OSS storage."""
        if direction == "outgoing":
            base_start_condition = "source_id = :start_id"
            base_current_node = "target_id"
            recursive_join = "a.source_id = gt.current_node"
            next_node = "a.target_id"
        elif direction == "incoming":
            base_start_condition = "target_id = :start_id"
            base_current_node = "source_id"
            recursive_join = "a.target_id = gt.current_node"
            next_node = "a.source_id"
        else:
            base_start_condition = "(source_id = :start_id OR target_id = :start_id)"
            base_current_node = (
                "CASE WHEN source_id = :start_id THEN target_id ELSE source_id END"
            )
            recursive_join = (
                "(a.source_id = gt.current_node OR a.target_id = gt.current_node)"
            )
            next_node = (
                "CASE WHEN a.source_id = gt.current_node THEN a.target_id ELSE a.source_id END"
            )

        async with self._session_factory() as session:
            params = {
                "workspace_id": workspace_id,
                "start_id": start_id,
                "max_depth": max_depth,
            }

            base_rel_filter = ""
            recursive_rel_filter = ""
            if relationships:
                rel_placeholders = ", ".join(
                    f":rel_{i}" for i in range(len(relationships))
                )
                base_rel_filter = f"AND relationship IN ({rel_placeholders})"
                recursive_rel_filter = (
                    f"AND a.relationship IN ({rel_placeholders})"
                )
                for i, rel in enumerate(relationships):
                    params[f"rel_{i}"] = rel

            cte_sql = text(f"""
            WITH RECURSIVE graph_traverse(
                id, source_id, target_id, relationship, strength, metadata,
                created_at, depth, current_node, path
            ) AS (
                -- Base case: every eligible edge adjacent in the requested direction.
                SELECT
                    id,
                    source_id,
                    target_id,
                    relationship,
                    strength,
                    metadata,
                    created_at,
                    1 as depth,
                    {base_current_node} as current_node,
                    ARRAY[
                        CAST(:start_id AS text),
                        {base_current_node}
                    ] as path
                FROM memory_associations
                WHERE workspace_id = :workspace_id
                  AND {base_start_condition}
                  {base_rel_filter}

                UNION ALL

                -- Recursive case: continue from the last visited node.
                SELECT
                    a.id,
                    a.source_id,
                    a.target_id,
                    a.relationship,
                    a.strength,
                    a.metadata,
                    a.created_at,
                    gt.depth + 1,
                    {next_node} as current_node,
                    gt.path || {next_node}
                FROM memory_associations a
                INNER JOIN graph_traverse gt ON {recursive_join}
                WHERE a.workspace_id = :workspace_id
                  {recursive_rel_filter}
                  AND gt.depth < :max_depth
                  AND NOT ({next_node} = ANY(gt.path))
            )
            SELECT * FROM graph_traverse;
            """)

            result = await session.execute(cte_sql, params)
            # Use .mappings() so rows are dict-like and string subscripting
            # (row["path"], row["id"], ...) works. A SQLAlchemy 2.0 Core Row
            # does NOT support string subscripting and raises TypeError, which
            # made traverse_graph throw on any non-empty result.
            rows = result.mappings().all()

            # Build paths from results
            paths = []
            unique_nodes = set([start_id])

            for row in rows:
                path_nodes = row["path"]
                unique_nodes.update(path_nodes)

                # Create association edge
                edge = Association(
                    id=row["id"],
                    workspace_id=workspace_id,
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    relationship=row["relationship"],
                    strength=row["strength"],
                    metadata=row["metadata"],
                    created_at=row.get("created_at") or datetime.now(timezone.utc),
                )

                path = GraphPath(
                    nodes=path_nodes,
                    edges=[edge],
                    total_strength=row["strength"],
                    depth=row["depth"],
                )
                paths.append(path)

            return GraphQueryResult(
                paths=paths,
                total_paths=len(paths),
                unique_nodes=list(unique_nodes),
                query_latency_ms=0,
            )

    async def count_associations_by_relationship(self, workspace_id: str) -> dict[str, int]:
        """Per-relationship edge counts via a single GROUP BY aggregate.

        Powers ``GraphQueryService.relationship_rollup`` without loading all
        edges into Python. ``relation_type`` is the ORM attr for the
        ``relationship`` column.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(
                    MemoryAssociationModel.relation_type,
                    func.count().label("cnt"),
                )
                .where(MemoryAssociationModel.workspace_id == workspace_id)
                .group_by(MemoryAssociationModel.relation_type)
            )
            return {row[0]: row[1] for row in result.all()}

    # Workspace operations
    async def create_workspace(self, workspace: Workspace) -> Workspace:
        """Create workspace (upsert: updates settings if workspace already exists)."""
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        now = workspace.updated_at or datetime.now(timezone.utc)

        async with self._session_factory() as session:
            tags = normalize_tags(workspace.tags)
            stmt = pg_insert(WorkspaceModel).values(
                id=workspace.id,
                tenant_id=workspace.tenant_id,
                name=workspace.name,
                settings=workspace.settings,
                tags=tags,
                created_at=workspace.created_at,
                updated_at=now,
            ).on_conflict_do_update(
                index_elements=['id'],
                set_=dict(
                    name=workspace.name,
                    settings=workspace.settings,
                    tags=tags,
                    updated_at=now,
                ),
            )
            await session.execute(stmt)
            await session.commit()
            result = await session.execute(
                select(WorkspaceModel).where(WorkspaceModel.id == workspace.id)
            )
            workspace_model = result.scalar_one()

        return self._workspace_model_to_domain(workspace_model)

    async def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """Get workspace by ID."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(WorkspaceModel).where(WorkspaceModel.id == workspace_id)
            )
            workspace_model = result.scalar_one_or_none()

            if not workspace_model:
                return None

            return self._workspace_model_to_domain(workspace_model)

    async def list_workspaces(
        self,
        *,
        tags: list[str] | None = None,
        match: str = "all",
    ) -> list[Workspace]:
        """List workspaces, optionally filtered by tag.

        Uses PostgreSQL array operators: 'all' -> tags @> [t] per tag (containment),
        'any' -> tags && query (overlap).
        """
        async with self._session_factory() as session:
            stmt = select(WorkspaceModel).order_by(WorkspaceModel.created_at)

            query_tags = normalize_tags(tags)
            if query_tags:
                if match == "any":
                    stmt = stmt.where(WorkspaceModel.tags.overlap(query_tags))
                else:  # default: 'all' — workspace must carry every requested tag
                    stmt = stmt.where(WorkspaceModel.tags.contains(query_tags))

            result = await session.execute(stmt)
            workspace_models = result.scalars().all()
            return [self._workspace_model_to_domain(m) for m in workspace_models]

    async def list_all_workspace_ids(self) -> list[str]:
        """Get all workspace IDs."""
        async with self._session_factory() as session:
            result = await session.execute(select(WorkspaceModel.id))
            return [row[0] for row in result.all()]

    async def update_workspace(self, workspace_id: str, **updates) -> Optional[Workspace]:
        """Update workspace fields."""
        if not updates:
            return await self.get_workspace(workspace_id)

        async with self._session_factory() as session:
            result = await session.execute(
                select(WorkspaceModel).where(WorkspaceModel.id == workspace_id)
            )
            workspace_model = result.scalar_one_or_none()
            if not workspace_model:
                return None

            for key, value in updates.items():
                if not hasattr(workspace_model, key):
                    continue
                if key == "tags":
                    value = normalize_tags(value)
                setattr(workspace_model, key, value)

            workspace_model.updated_at = datetime.now(timezone.utc)
            await session.commit()
            await session.refresh(workspace_model)
            return self._workspace_model_to_domain(workspace_model)

    async def search_memories_by_filter(
        self,
        workspace_id: str,
        *,
        subtypes: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        metadata_filter: Optional[dict[str, str]] = None,
        status: str = "active",
        context_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Memory]:
        """Search memories by subtype, tags, and/or metadata without requiring embeddings."""
        async with self._session_factory() as session:
            conditions = [
                MemoryModel.workspace_id == workspace_id,
                MemoryModel.deleted_at.is_(None),
            ]

            # RPG canonical and overlay graphs are isolated by context row ID.
            if context_id is not None:
                conditions.append(MemoryModel.context_id == context_id)

            # Forced user-scope partition filter (the user-global read boundary):
            # the same forced user_id filter the recall fan-out uses to keep one
            # user's user-global memories private from another.
            if user_id is not None:
                conditions.append(MemoryModel.user_id == user_id)

            if status:
                conditions.append(MemoryModel.status == status)

            if subtypes:
                conditions.append(MemoryModel.subtype.in_(subtypes))

            if tags:
                conditions.append(MemoryModel.tags.contains(tags))

            if metadata_filter:
                conditions.append(MemoryModel.meta.contains(metadata_filter))

            query = (
                select(MemoryModel)
                .where(and_(*conditions))
                .order_by(desc(MemoryModel.created_at))
                .limit(limit)
                .offset(offset)
            )
            query = query.options(_defer_embedding())
            result = await session.execute(query)
            memory_models = result.scalars().all()
            return [self._memory_model_to_domain(m) for m in memory_models]

    # Context operations
    async def create_context(self, workspace_id: str, context: Context) -> Context:
        """Create a context within a workspace."""
        context_model = ContextModel(
            id=context.id,
            workspace_id=workspace_id,
            name=context.name,
            description=context.description,
            settings=context.settings,
            created_at=context.created_at,
        )

        async with self._session_factory() as session:
            session.add(context_model)
            await session.commit()
            await session.refresh(context_model)

        return self._context_model_to_domain(context_model)

    async def get_context(self, workspace_id: str, context_id: str) -> Optional[Context]:
        """Get context by ID."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(ContextModel).where(
                    ContextModel.id == context_id,
                    ContextModel.workspace_id == workspace_id
                )
            )
            context_model = result.scalar_one_or_none()

            if not context_model:
                return None

            return self._context_model_to_domain(context_model)

    async def list_contexts(self, workspace_id: str) -> list[Context]:
        """List all contexts in a workspace."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(ContextModel)
                .where(ContextModel.workspace_id == workspace_id)
                .order_by(ContextModel.created_at)
            )
            context_models = result.scalars().all()

            return [self._context_model_to_domain(m) for m in context_models]

    async def delete_context(self, workspace_id: str, context_id: str) -> bool:
        """Hard-delete a context within a workspace.

        Contexts are hard-deleted (no soft-delete column). The FK from
        memories.context_id is ON DELETE SET NULL, so deleting a context does
        not remove its memories — it only clears their (reserved/unused)
        context_id. Returns True if a row was deleted, False if the context did
        not exist in this workspace.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                delete(ContextModel).where(
                    and_(
                        ContextModel.id == context_id,
                        ContextModel.workspace_id == workspace_id,
                    )
                )
            )
            await session.commit()
            deleted = result.rowcount > 0
            if deleted:
                self.logger.debug("Deleted context: %s", context_id)
            return deleted

    def _context_model_to_domain(self, model: ContextModel) -> Context:
        """Convert ORM model to domain model."""
        return Context(
            id=model.id,
            workspace_id=model.workspace_id,
            name=model.name,
            description=model.description,
            settings=model.settings or {},
            created_at=model.created_at,
        )

    # Statistics
    async def get_workspace_stats(self, workspace_id: str) -> dict:
        """Get memory statistics for workspace."""
        async with self._session_factory() as session:
            # Count memories by type
            type_counts_query = select(
                MemoryModel.type,
                func.count().label("count")
            ).where(
                and_(
                    MemoryModel.workspace_id == workspace_id,
                    MemoryModel.deleted_at.is_(None),
                )
            ).group_by(MemoryModel.type)

            type_counts_result = await session.execute(type_counts_query)
            type_counts = {row[0]: row[1] for row in type_counts_result}

            # Count associations
            assoc_count_query = select(func.count()).select_from(MemoryAssociationModel).where(
                MemoryAssociationModel.workspace_id == workspace_id
            )
            assoc_count_result = await session.execute(assoc_count_query)
            assoc_count = assoc_count_result.scalar()

            return {
                "total_memories": sum(type_counts.values()),
                "memories_by_type": type_counts,
                "total_associations": assoc_count,
                "total_categories": 0,
            }

    async def get_workspace_change_watermark(
        self, workspace_id: str
    ) -> tuple[str, str, int, int] | None:
        """Deterministic dirty-watermark over active memories + all associations.

        Two aggregate queries (one per table), no full scan into Python. See the
        ABC docstring for the staleness rationale: association_count catches an
        edge delete that advances no timestamp. Returns None on any error so
        callers fall back to doing the full work (fail-safe).
        """
        try:
            async with self._session_factory() as session:
                mem_result = await session.execute(
                    select(
                        func.max(MemoryModel.updated_at),
                        func.count(),
                    ).where(
                        and_(
                            MemoryModel.workspace_id == workspace_id,
                            MemoryModel.deleted_at.is_(None),
                        )
                    )
                )
                max_updated, mem_count = mem_result.one()

                assoc_result = await session.execute(
                    select(
                        func.max(MemoryAssociationModel.created_at),
                        func.count(),
                    ).where(MemoryAssociationModel.workspace_id == workspace_id)
                )
                max_created, assoc_count = assoc_result.one()
        except Exception as e:  # fail-safe: ambiguous watermark -> do the work
            self.logger.debug("Could not compute change watermark for %s: %s", workspace_id, e)
            return None

        return (
            max_updated.isoformat() if max_updated else "",
            max_created.isoformat() if max_created else "",
            int(mem_count or 0),
            int(assoc_count or 0),
        )

    async def get_memory_by_hash(self, workspace_id: str, content_hash: str) -> Optional[Memory]:
        """Get memory by content hash for deduplication.

        Tolerates duplicate hashes rather than asserting uniqueness. Nothing
        enforces one row per (workspace_id, content_hash): dedup is check-then-
        act, so two concurrent ingests of identical content both look, both miss,
        and both insert. Under decomposition that happens readily -- many facts
        from parallel tasks carry the same text.

        ``scalar_one_or_none()`` raised MultipleResultsFound on those rows, which
        killed dedup for that content PERMANENTLY: every later ingest of the same
        text hit the same exception. Returning the oldest match instead answers
        the question actually being asked ("is this content already stored?") and
        keeps the collapse deterministic, so repeated calls converge on the same
        survivor rather than picking arbitrarily.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(MemoryModel).where(
                    and_(
                        MemoryModel.workspace_id == workspace_id,
                        MemoryModel.content_hash == content_hash,
                        MemoryModel.deleted_at.is_(None),
                    )
                ).order_by(MemoryModel.created_at.asc(), MemoryModel.id.asc()).limit(1)
            )
            memory_model = result.scalar_one_or_none()

            if not memory_model:
                return None

            return self._memory_model_to_domain(memory_model)

    # Cold Tier Storage operations
    async def archive_memory(
            self,
            workspace_id: str,
            memory_id: str,
    ) -> bool:
        """
        Archive a memory to cold tier storage.

        Moves the memory from hot tier (with full embeddings) to cold tier
        (compressed graph structure without embeddings). The memory's embedding
        is removed after archival to save storage space.

        Args:
            workspace_id: Workspace identifier.
            memory_id: Memory identifier to archive.

        Returns:
            True if successfully archived, False if memory not found or already archived.
        """
        async with self._session_factory() as session:
            # Get the memory to archive
            result = await session.execute(
                select(MemoryModel).where(
                    and_(
                        MemoryModel.id == memory_id,
                        MemoryModel.workspace_id == workspace_id,
                        MemoryModel.deleted_at.is_(None),
                    )
                )
            )
            memory_model = result.scalar_one_or_none()

            if not memory_model:
                self.logger.warning("Memory %s not found for archival", memory_id)
                return False

            # Check if already archived (has archived_at timestamp)
            if hasattr(memory_model, 'archived_at') and memory_model.archived_at is not None:
                self.logger.warning("Memory %s is already archived", memory_id)
                return False

            # Find or create a LEANN graph for this workspace
            graph_result = await session.execute(
                select(LeannGraphModel)
                .where(LeannGraphModel.workspace_id == workspace_id)
                .order_by(LeannGraphModel.created_at.desc())
                .limit(1)
            )
            graph_model = graph_result.scalar_one_or_none()

            # Create a new graph if none exists or current graph has enough nodes
            max_nodes_per_graph = 10000  # Configurable threshold
            if not graph_model or graph_model.node_count >= max_nodes_per_graph:
                import numpy as np
                # Create a new minimal graph with just this memory
                graph_id = f"leann_{uuid.uuid4().hex[:12]}"
                csr_graph = CSRGraph(
                    indptr=np.array([0, 0], dtype=np.int64),
                    indices=np.array([], dtype=np.int64),
                    data=np.array([], dtype=np.float32),
                    node_ids=[memory_id],
                )
                graph_data = self._leann_storage.serialize_graph(csr_graph)

                graph_model = LeannGraphModel(
                    id=graph_id,
                    workspace_id=workspace_id,
                    graph_data=graph_data,
                    node_count=1,
                    edge_count=0,
                    memory_ids=[memory_id],
                    metadata={},
                )
                session.add(graph_model)
                await session.flush()
                position = 0
            else:
                # Add to existing graph
                position = graph_model.node_count
                # Update graph's memory_ids list
                memory_ids = list(graph_model.memory_ids) if graph_model.memory_ids else []
                memory_ids.append(memory_id)
                graph_model.memory_ids = memory_ids
                graph_model.node_count = position + 1

                # Update graph_data to include new node
                import numpy as np
                csr_graph = self._leann_storage.deserialize_graph(graph_model.graph_data)
                # Extend indptr for new node (no edges)
                new_indptr = np.append(csr_graph.indptr, csr_graph.indptr[-1])
                csr_graph = CSRGraph(
                    indptr=new_indptr,
                    indices=csr_graph.indices,
                    data=csr_graph.data,
                    node_ids=memory_ids,
                )
                graph_model.graph_data = self._leann_storage.serialize_graph(csr_graph)

            # Create the cold tier document
            doc_id = f"ldoc_{uuid.uuid4().hex[:12]}"
            doc_model = LeannDocumentModel(
                id=doc_id,
                graph_id=graph_model.id,
                workspace_id=workspace_id,
                memory_id=memory_id,
                content=memory_model.content,
                position=position,
                cold_access_count=0,
                memory_type=memory_model.type,
                memory_subtype=memory_model.subtype,
                importance=memory_model.importance,
                tags=memory_model.tags or [],
                metadata=memory_model.meta or {},
            )
            session.add(doc_model)

            # Mark the original memory as archived and clear embedding
            if hasattr(memory_model, 'archived_at'):
                memory_model.archived_at = datetime.now(timezone.utc)
            memory_model.embedding = None  # Remove embedding to save space

            await session.commit()

            self.logger.info(
                "Archived memory %s to cold tier (graph %s, position %d)",
                memory_id,
                graph_model.id,
                position,
            )
            return True

    async def restore_memory(
            self,
            workspace_id: str,
            memory_id: str,
    ) -> bool:
        """
        Restore a memory from cold tier to hot tier.

        The memory is moved back to hot tier. Note that embeddings will need
        to be regenerated separately by the embedding service.

        Args:
            workspace_id: Workspace identifier.
            memory_id: Memory identifier to restore.

        Returns:
            True if successfully restored, False if memory not found or not in cold tier.
        """
        async with self._session_factory() as session:
            # Find the cold tier document for this memory
            doc_result = await session.execute(
                select(LeannDocumentModel).where(
                    and_(
                        LeannDocumentModel.memory_id == memory_id,
                        LeannDocumentModel.workspace_id == workspace_id,
                    )
                )
            )
            doc_model = doc_result.scalar_one_or_none()

            if not doc_model:
                self.logger.warning("Memory %s not found in cold tier", memory_id)
                return False

            # Get the original memory
            memory_result = await session.execute(
                select(MemoryModel).where(
                    and_(
                        MemoryModel.id == memory_id,
                        MemoryModel.workspace_id == workspace_id,
                    )
                )
            )
            memory_model = memory_result.scalar_one_or_none()

            if memory_model:
                # Clear the archived_at flag to mark as hot tier
                if hasattr(memory_model, 'archived_at'):
                    memory_model.archived_at = None
                # Embedding will need to be regenerated by embedding service
            else:
                # Memory was deleted, need to recreate from cold tier
                workspace = await session.get(WorkspaceModel, workspace_id)
                tenant_id = workspace.tenant_id if workspace else 'default_tenant'
                now = datetime.now(timezone.utc)
                memory_model = MemoryModel(
                    id=memory_id,
                    workspace_id=workspace_id,
                    tenant_id=tenant_id,
                    content=doc_model.content,
                    content_hash=hashlib.sha256(doc_model.content.encode()).hexdigest(),
                    type=doc_model.memory_type or MemoryType.SEMANTIC.value,
                    subtype=doc_model.memory_subtype,
                    importance=doc_model.importance,
                    tags=doc_model.tags,
                    meta=doc_model.metadata,
                    created_at=now,
                    updated_at=now,
                )
                session.add(memory_model)

            # Remove the cold tier document
            await session.execute(
                delete(LeannDocumentModel).where(
                    LeannDocumentModel.id == doc_model.id
                )
            )

            await session.commit()

            self.logger.info("Restored memory %s from cold tier to hot tier", memory_id)
            return True

    async def search_cold_memories(
            self,
            workspace_id: str,
            query_embedding: list[float],
            limit: int = 10,
            min_relevance: float = 0.5,
    ) -> list[tuple[Memory, float]]:
        """
        Search memories in cold tier storage.

        Uses embedding-based graph-guided search when a compression service is
        available. Falls back to importance-based ranking otherwise.

        Args:
            workspace_id: Workspace identifier.
            query_embedding: Query vector for similarity search.
            limit: Maximum number of results to return.
            min_relevance: Minimum relevance score threshold.

        Returns:
            List of (memory, relevance_score) tuples.
        """
        if self._compression_service is not None:
            return await self._search_cold_memories_embedding(
                workspace_id, query_embedding, limit, min_relevance,
            )
        return await self._search_cold_memories_importance(
            workspace_id, query_embedding, limit, min_relevance,
        )

    async def _search_cold_memories_embedding(
            self,
            workspace_id: str,
            query_embedding: list[float],
            limit: int = 10,
            min_relevance: float = 0.5,
    ) -> list[tuple[Memory, float]]:
        """Search cold tier using embedding-based graph-guided retrieval."""
        results: list[tuple[Memory, float]] = []

        # Get all LEANN graphs for this workspace
        graphs = await self._leann_storage.get_graphs_by_workspace(workspace_id)
        if not graphs:
            return results

        for graph_meta in graphs:
            # Load the full CSR graph
            graph_result = await self._leann_storage.get_graph(workspace_id, graph_meta.id)
            if graph_result is None:
                continue
            _, csr_graph = graph_result

            if csr_graph.node_count == 0:
                continue

            # Load documents for this graph and build content lookup
            documents = await self._leann_storage.get_documents_by_graph(
                workspace_id, graph_meta.id,
            )
            content_lookup: dict[str, str] = {}
            doc_by_memory_id: dict[str, 'LeannDocument'] = {}
            for doc in documents:
                mem_id = doc.memory_id or doc.id
                content_lookup[mem_id] = doc.content
                doc_by_memory_id[mem_id] = doc

            if not content_lookup:
                continue

            # Use compression service for real embedding search
            cold_result = await self._compression_service.retrieve_cold_with_on_demand_embedding(
                graph=csr_graph,
                query_embedding=query_embedding,
                content_lookup=content_lookup,
                limit=limit,
            )

            self.logger.debug(
                "Cold embedding search on graph %s: %d candidates -> %d results in %d ms",
                graph_meta.id,
                cold_result.candidates_evaluated,
                len(cold_result.memory_ids),
                cold_result.total_retrieval_time_ms,
            )

            # Convert ColdRetrievalResult to Memory objects
            for mem_id, similarity in zip(cold_result.memory_ids, cold_result.similarities):
                if similarity < min_relevance:
                    continue

                doc = doc_by_memory_id.get(mem_id)
                if doc is None:
                    continue

                memory = Memory(
                    id=doc.memory_id or doc.id,
                    workspace_id=doc.workspace_id,
                    context_id=None,
                    user_id=None,
                    content=doc.content,
                    content_hash=hashlib.sha256(doc.content.encode()).hexdigest(),
                    type=MemoryType(doc.memory_type) if doc.memory_type else MemoryType.SEMANTIC,
                    subtype=doc.memory_subtype if doc.memory_subtype else None,
                    importance=doc.importance,
                    tags=doc.tags or [],
                    metadata={
                        **(doc.metadata or {}),
                        "_cold_tier": True,
                        "_graph_id": doc.graph_id,
                        "_position": doc.position,
                    },
                    embedding=None,
                    access_count=doc.cold_access_count,
                    last_accessed_at=doc.last_cold_access_at,
                    decay_factor=1.0,
                    created_at=doc.created_at,
                    updated_at=doc.created_at,
                )
                results.append((memory, similarity))

        # Sort by relevance descending across all graphs
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:limit]

        # Update cold access counts for returned results
        if results:
            returned_memory_ids = {m.id for m, _ in results}
            async with self._session_factory() as session:
                now = datetime.now(timezone.utc)
                await session.execute(
                    update(LeannDocumentModel)
                    .where(
                        and_(
                            LeannDocumentModel.workspace_id == workspace_id,
                            or_(
                                LeannDocumentModel.memory_id.in_(returned_memory_ids),
                                LeannDocumentModel.id.in_(returned_memory_ids),
                            ),
                        )
                    )
                    .values(
                        cold_access_count=LeannDocumentModel.cold_access_count + 1,
                        last_cold_access_at=now,
                    )
                )
                await session.commit()

        return results

    async def _search_cold_memories_importance(
            self,
            workspace_id: str,
            query_embedding: list[float],
            limit: int = 10,
            min_relevance: float = 0.5,
    ) -> list[tuple[Memory, float]]:
        """Fallback cold tier search using importance as a proxy for relevance."""
        import numpy as np

        results: list[tuple[Memory, float]] = []

        async with self._session_factory() as session:
            # Get cold tier documents ordered by importance
            doc_result = await session.execute(
                select(LeannDocumentModel)
                .where(LeannDocumentModel.workspace_id == workspace_id)
                .order_by(LeannDocumentModel.importance.desc())
                .limit(limit * 10)  # Get more candidates for filtering
            )
            doc_models = doc_result.scalars().all()

            if not doc_models:
                return results

            for doc_model in doc_models:
                # Use importance as a proxy relevance score
                relevance = doc_model.importance

                if relevance >= min_relevance:
                    memory = Memory(
                        id=doc_model.memory_id or doc_model.id,
                        workspace_id=doc_model.workspace_id,
                        context_id=None,
                        user_id=None,
                        content=doc_model.content,
                        content_hash=hashlib.sha256(doc_model.content.encode()).hexdigest(),
                        type=MemoryType(doc_model.memory_type) if doc_model.memory_type else MemoryType.SEMANTIC,
                        subtype=doc_model.memory_subtype if doc_model.memory_subtype else None,
                        importance=doc_model.importance,
                        tags=doc_model.tags or [],
                        metadata={
                            **(doc_model.metadata or {}),
                            "_cold_tier": True,
                            "_graph_id": doc_model.graph_id,
                            "_position": doc_model.position,
                        },
                        embedding=None,
                        access_count=doc_model.cold_access_count,
                        last_accessed_at=doc_model.last_cold_access_at,
                        decay_factor=1.0,
                        created_at=doc_model.created_at,
                        updated_at=doc_model.created_at,
                    )
                    results.append((memory, relevance))

                    # Update cold access count
                    doc_model.cold_access_count += 1
                    doc_model.last_cold_access_at = datetime.now(timezone.utc)

                    if len(results) >= limit:
                        break

            await session.commit()

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    async def get_hot_promotion_candidates(
            self,
            workspace_id: str,
            min_access_count: int = 10,
            limit: int = 100,
    ) -> list[Memory]:
        """Get frequently accessed cold tier documents for hot promotion."""
        docs = await self._leann_storage.get_frequently_accessed_documents(
            workspace_id=workspace_id,
            min_access_count=min_access_count,
            limit=limit,
        )

        candidates = []
        for doc in docs:
            memory = Memory(
                id=doc.memory_id or doc.id,
                workspace_id=doc.workspace_id,
                context_id=None,
                user_id=None,
                content=doc.content,
                content_hash=hashlib.sha256(doc.content.encode()).hexdigest(),
                type=MemoryType(doc.memory_type) if doc.memory_type else MemoryType.SEMANTIC,
                subtype=doc.memory_subtype if doc.memory_subtype else None,
                importance=doc.importance,
                tags=doc.tags or [],
                metadata={
                    **(doc.metadata or {}),
                    "_cold_tier": True,
                    "_graph_id": doc.graph_id,
                    "_position": doc.position,
                },
                embedding=None,
                access_count=doc.cold_access_count,
                last_accessed_at=doc.last_cold_access_at,
                decay_factor=1.0,
                created_at=doc.created_at,
                updated_at=doc.created_at,
            )
            candidates.append(memory)

        return candidates

    async def get_archival_candidates(
            self,
            workspace_id: str,
            max_importance: float = 0.3,
            max_access_count: int = 5,
            older_than_days: int = 90,
            limit: int = 100,
    ) -> list[Memory]:
        """
        Get memories that are candidates for archival to cold tier.

        Returns memories that meet archival criteria based on:
        - importance score below threshold
        - access count below threshold
        - last accessed older than specified days

        Args:
            workspace_id: Workspace identifier.
            max_importance: Maximum importance score threshold.
            max_access_count: Maximum access count threshold.
            older_than_days: Minimum days since last access.
            limit: Maximum number of candidates to return.

        Returns:
            List of memories eligible for archival.
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=older_than_days)

        async with self._session_factory() as session:
            # Build query for archival candidates
            query = (
                select(MemoryModel)
                .where(
                    and_(
                        MemoryModel.workspace_id == workspace_id,
                        MemoryModel.deleted_at.is_(None),
                        MemoryModel.importance <= max_importance,
                        MemoryModel.access_count <= max_access_count,
                        or_(
                            MemoryModel.last_accessed_at.is_(None),
                            MemoryModel.last_accessed_at < cutoff_date,
                        ),
                        # Only include memories not already archived
                        MemoryModel.embedding.isnot(None),
                    )
                )
                .order_by(MemoryModel.importance.asc(), MemoryModel.access_count.asc())
                .limit(limit)
            )

            # Add archived_at check if the column exists
            try:
                query = query.where(
                    MemoryModel.archived_at.is_(None)
                )
            except AttributeError:
                # archived_at column may not exist in older schemas
                pass

            query = query.options(_defer_embedding())
            result = await session.execute(query)
            memory_models = result.scalars().all()

            return [self._memory_model_to_domain(m) for m in memory_models]

    async def get_cold_storage_stats(self, workspace_id: str) -> dict:
        """
        Get cold tier storage statistics for workspace.

        Returns statistics about hot and cold tier storage usage including
        memory counts, storage bytes, and compression ratio.

        Args:
            workspace_id: Workspace identifier.

        Returns:
            Dict with storage statistics.
        """
        async with self._session_factory() as session:
            # Count hot tier memories. Hot = live and not archived. We key on
            # archived_at (the tier marker set by archive_memory), NOT on
            # ``embedding IS NOT NULL`` — a memory can be hot yet lack an
            # embedding (e.g. embedding generation disabled), and counting by
            # embedding presence made hot read 0 for those workspaces.
            hot_query = select(func.count()).select_from(MemoryModel).where(
                and_(
                    MemoryModel.workspace_id == workspace_id,
                    MemoryModel.deleted_at.is_(None),
                    MemoryModel.archived_at.is_(None),
                )
            )
            hot_result = await session.execute(hot_query)
            hot_count = hot_result.scalar() or 0

            # Count cold tier documents
            cold_query = select(func.count()).select_from(LeannDocumentModel).where(
                LeannDocumentModel.workspace_id == workspace_id
            )
            cold_result = await session.execute(cold_query)
            cold_count = cold_result.scalar() or 0

            # Calculate cold tier storage bytes (graph data)
            cold_bytes_query = select(
                func.coalesce(func.sum(func.length(LeannGraphModel.graph_data)), 0)
            ).where(LeannGraphModel.workspace_id == workspace_id)
            cold_bytes_result = await session.execute(cold_bytes_query)
            cold_bytes = cold_bytes_result.scalar() or 0

            # Estimate hot tier storage bytes (embeddings + content). Each
            # embedding is the configured ``_EMBEDDING_DIM`` float32s (not a
            # hardcoded 1536), so this tracks MEMORYLAYER_EMBEDDING_DIMENSIONS.
            embedding_bytes_per_memory = _EMBEDDING_DIM * 4
            hot_bytes = hot_count * embedding_bytes_per_memory

            # Add content size estimation for hot tier
            hot_content_query = select(
                func.coalesce(func.sum(func.length(MemoryModel.content)), 0)
            ).where(
                and_(
                    MemoryModel.workspace_id == workspace_id,
                    MemoryModel.deleted_at.is_(None),
                    MemoryModel.archived_at.is_(None),
                )
            )
            hot_content_result = await session.execute(hot_content_query)
            hot_content_bytes = hot_content_result.scalar() or 0
            hot_bytes += hot_content_bytes

            # Add content size for cold tier
            cold_content_query = select(
                func.coalesce(func.sum(func.length(LeannDocumentModel.content)), 0)
            ).where(LeannDocumentModel.workspace_id == workspace_id)
            cold_content_result = await session.execute(cold_content_query)
            cold_content_bytes = cold_content_result.scalar() or 0
            cold_bytes += cold_content_bytes

            # Calculate compression ratio
            # Equivalent hot storage = cold memories with embeddings
            equivalent_hot_bytes = cold_count * embedding_bytes_per_memory + cold_content_bytes
            compression_ratio = cold_bytes / equivalent_hot_bytes if equivalent_hot_bytes > 0 else 0.0

            # Calculate savings
            savings_bytes = equivalent_hot_bytes - cold_bytes if equivalent_hot_bytes > cold_bytes else 0

            return {
                "cold_memory_count": cold_count,
                "cold_storage_bytes": cold_bytes,
                "hot_memory_count": hot_count,
                "hot_storage_bytes": hot_bytes,
                "compression_ratio": compression_ratio,
                "estimated_savings_bytes": savings_bytes,
            }

    async def get_admin_tiering_stats(self) -> dict:
        """Aggregate cold-tier statistics across ALL workspaces.

        The tenant-wide analogue of :meth:`get_cold_storage_stats` (which is
        per-workspace). Used by the ``GET /v1/admin/tiering`` endpoint so the
        admin console can show one tenant-wide hot/cold picture instead of a
        single workspace's numbers. Same estimation model as the per-workspace
        version; hot = memories, cold = LEANN cold-tier tables.

        Storage sizes are MEASURED on-disk bytes via ``pg_total_relation_size``
        (heap + TOAST + every index) — not a content-length estimate. This
        captures what a byte model misses entirely: the pgvector indexes and the
        memory/page ``multivector``s (often larger than the raw vectors). Sizes
        are whole-table (deployment-wide), matching this admin overview's scope,
        and include index + bloat (real disk). ``to_regclass`` → NULL for an
        absent table → coalesced to 0, so it's safe on partially-migrated
        schemas.
          hot      = ``memories`` table
          cold     = LEANN cold-tier (``leann_documents`` + ``leann_graphs``)
          document = ``documents`` + ``document_pages`` (pages hold transcripts +
                     the page multivectors). Document CHUNKS are extracted into
                     ``memories`` (``source_document_id``) → already under hot;
                     not re-added here.

        ``compression_ratio`` / ``estimated_savings_bytes`` stay a LOGICAL model:
        they measure the tiering *benefit* of LEANN vs. keeping cold memories in
        the hot tier (content length + the configured embedding dimension), not
        disk — so they intentionally differ in KIND from the measured sizes.
        """
        # Embedding size for the tiering-benefit model follows the configured
        # dimension (a memory's vector is ``_EMBEDDING_DIM`` float32s), not 1536.
        embedding_bytes_per_memory = _EMBEDDING_DIM * 4

        async with self._session_factory() as session:
            async def _relation_size(table_name: str) -> int:
                return (await session.execute(
                    select(func.coalesce(
                        func.pg_total_relation_size(func.to_regclass(table_name)), 0
                    ))
                )).scalar() or 0

            hot_count = (await session.execute(
                select(func.count()).select_from(MemoryModel).where(
                    and_(
                        MemoryModel.deleted_at.is_(None),
                        MemoryModel.archived_at.is_(None),
                    )
                )
            )).scalar() or 0
            cold_count = (await session.execute(
                select(func.count()).select_from(LeannDocumentModel)
            )).scalar() or 0

            # Measured on-disk storage per category (heap + TOAST + indexes).
            hot_bytes = await _relation_size(MemoryModel.__tablename__)
            cold_bytes = (
                await _relation_size(LeannDocumentModel.__tablename__)
                + await _relation_size(LeannGraphModel.__tablename__)
            )
            documents_table_bytes = await _relation_size(DocumentModel.__tablename__)
            document_pages_bytes = await _relation_size(DocumentPageModel.__tablename__)
            document_storage_bytes = documents_table_bytes + document_pages_bytes

            # Logical tiering-benefit model (unchanged in kind): LEANN cold-tier
            # size vs. the equivalent hot cost of those same memories. Content
            # length + configured embedding size — deliberately NOT relation size.
            cold_graph_bytes = (await session.execute(
                select(func.coalesce(func.sum(func.length(LeannGraphModel.graph_data)), 0))
            )).scalar() or 0
            cold_content_bytes = (await session.execute(
                select(func.coalesce(func.sum(func.length(LeannDocumentModel.content)), 0))
            )).scalar() or 0
            cold_logical_bytes = cold_graph_bytes + cold_content_bytes
            equivalent_hot_bytes = cold_count * embedding_bytes_per_memory + cold_content_bytes
            compression_ratio = cold_logical_bytes / equivalent_hot_bytes if equivalent_hot_bytes > 0 else 0.0
            savings_bytes = equivalent_hot_bytes - cold_logical_bytes if equivalent_hot_bytes > cold_logical_bytes else 0

            return {
                "cold_memory_count": cold_count,
                "cold_storage_bytes": cold_bytes,
                "hot_memory_count": hot_count,
                "hot_storage_bytes": hot_bytes,
                "compression_ratio": compression_ratio,
                "estimated_savings_bytes": savings_bytes,
                "document_storage_bytes": document_storage_bytes,
                "documents_table_bytes": documents_table_bytes,
                "document_pages_bytes": document_pages_bytes,
            }

    # Session operations
    async def create_session(self, workspace_id: str, session: Session) -> Session:
        """Store a new session."""
        session_model = SessionModel(
            id=session.id,
            workspace_id=workspace_id,
            tenant_id=session.tenant_id,
            context_id=session.context_id,
            user_id=session.user_id,
            meta=session.metadata,
            auto_commit=session.auto_commit,
            committed_at=session.committed_at,
            expires_at=session.expires_at,
            created_at=session.created_at,
        )

        async with self._session_factory() as db_session:
            db_session.add(session_model)
            await db_session.commit()
            await db_session.refresh(session_model)

        return self._session_model_to_domain(session_model)

    async def get_session(self, workspace_id: str, session_id: str) -> Optional[Session]:
        """Get session by ID (returns None if not found or expired)."""
        async with self._session_factory() as db_session:
            result = await db_session.execute(
                select(SessionModel).where(
                    and_(
                        SessionModel.id == session_id,
                        SessionModel.workspace_id == workspace_id,
                    )
                )
            )
            session_model = result.scalar_one_or_none()

            if not session_model:
                return None

            # Check if expired
            session = self._session_model_to_domain(session_model)
            if session.is_expired:
                return None

            return session

    async def get_session_by_id(self, session_id: str) -> Optional[Session]:
        """Get session by ID without workspace filter."""
        async with self._session_factory() as db_session:
            result = await db_session.execute(
                select(SessionModel).where(SessionModel.id == session_id)
            )
            session_model = result.scalar_one_or_none()

            if not session_model:
                return None

            session = self._session_model_to_domain(session_model)
            if session.is_expired:
                return None

            return session

    async def delete_session(self, workspace_id: str, session_id: str) -> bool:
        """Delete session and all its context."""
        async with self._session_factory() as db_session:
            result = await db_session.execute(
                delete(SessionModel).where(
                    and_(
                        SessionModel.id == session_id,
                        SessionModel.workspace_id == workspace_id,
                    )
                )
            )
            await db_session.commit()
            return result.rowcount > 0

    async def set_working_memory(
            self,
            workspace_id: str,
            session_id: str,
            key: str,
            value: Any,
            ttl_seconds: Optional[int] = None
    ) -> WorkingMemory:
        """Set context key-value within session."""
        now = datetime.now(timezone.utc)

        async with self._session_factory() as db_session:
            # First verify the session exists and is not expired
            session_result = await db_session.execute(
                select(SessionModel).where(
                    and_(
                        SessionModel.id == session_id,
                        SessionModel.workspace_id == workspace_id,
                    )
                )
            )
            session_model = session_result.scalar_one_or_none()

            if not session_model:
                raise ValueError(f"Session {session_id} not found")

            if session_model.expires_at < now:
                raise ValueError(f"Session {session_id} has expired")

            # Check if context entry already exists
            existing_result = await db_session.execute(
                select(SessionContextModel).where(
                    and_(
                        SessionContextModel.session_id == session_id,
                        SessionContextModel.key == key,
                    )
                )
            )
            existing = existing_result.scalar_one_or_none()

            if existing:
                # Update existing entry
                existing.value = value
                existing.ttl_seconds = ttl_seconds
                existing.updated_at = now
                await self._add_context_event(
                    db_session,
                    workspace_id,
                    session_id,
                    ContextEventKind.WORKING_UPSERT.value,
                    "working_memory",
                    key,
                )
                await db_session.commit()
                await db_session.refresh(existing)
                return self._session_context_model_to_domain(existing)
            else:
                # Create new entry
                context_model = SessionContextModel(
                    session_id=session_id,
                    key=key,
                    value=value,
                    ttl_seconds=ttl_seconds,
                    created_at=now,
                    updated_at=now,
                )
                db_session.add(context_model)
                await self._add_context_event(
                    db_session,
                    workspace_id,
                    session_id,
                    ContextEventKind.WORKING_UPSERT.value,
                    "working_memory",
                    key,
                )
                await db_session.commit()
                await db_session.refresh(context_model)
                return self._session_context_model_to_domain(context_model)

    async def get_working_memory(
            self,
            workspace_id: str,
            session_id: str,
            key: str
    ) -> Optional[WorkingMemory]:
        """Get specific context entry."""
        async with self._session_factory() as db_session:
            # Verify session exists and belongs to workspace
            session_result = await db_session.execute(
                select(SessionModel).where(
                    and_(
                        SessionModel.id == session_id,
                        SessionModel.workspace_id == workspace_id,
                    )
                )
            )
            session_model = session_result.scalar_one_or_none()

            if not session_model:
                return None

            # Check if session is expired
            if session_model.expires_at < datetime.now(timezone.utc):
                return None

            # Get the context entry
            result = await db_session.execute(
                select(SessionContextModel).where(
                    and_(
                        SessionContextModel.session_id == session_id,
                        SessionContextModel.key == key,
                    )
                )
            )
            context_model = result.scalar_one_or_none()

            if not context_model:
                return None

            return self._session_context_model_to_domain(context_model)

    async def get_all_working_memory(
            self,
            workspace_id: str,
            session_id: str
    ) -> list[WorkingMemory]:
        """Get all context entries for session."""
        async with self._session_factory() as db_session:
            # Verify session exists and belongs to workspace
            session_result = await db_session.execute(
                select(SessionModel).where(
                    and_(
                        SessionModel.id == session_id,
                        SessionModel.workspace_id == workspace_id,
                    )
                )
            )
            session_model = session_result.scalar_one_or_none()

            if not session_model:
                return []

            # Check if session is expired
            if session_model.expires_at < datetime.now(timezone.utc):
                return []

            # Get all context entries
            result = await db_session.execute(
                select(SessionContextModel).where(
                    SessionContextModel.session_id == session_id
                ).order_by(SessionContextModel.created_at)
            )
            context_models = result.scalars().all()

            return [self._session_context_model_to_domain(c) for c in context_models]

    async def cleanup_expired_sessions(self, workspace_id: str) -> int:
        """Delete all expired sessions. Returns number cleaned up."""
        now = datetime.now(timezone.utc)

        async with self._session_factory() as db_session:
            result = await db_session.execute(
                delete(SessionModel).where(
                    and_(
                        SessionModel.workspace_id == workspace_id,
                        SessionModel.expires_at < now,
                    )
                )
            )
            await db_session.commit()
            return result.rowcount

    async def cleanup_all_expired_sessions(self) -> int:
        """Delete all expired sessions across all workspaces. Returns number cleaned up."""
        now = datetime.now(timezone.utc)

        async with self._session_factory() as db_session:
            result = await db_session.execute(
                delete(SessionModel).where(
                    SessionModel.expires_at < now,
                )
            )
            await db_session.commit()
            return result.rowcount

    # Decay service support
    async def get_memories_for_decay(
        self,
        workspace_id: str,
        min_age_days: int = 7,
        exclude_pinned: bool = True,
    ) -> list[Memory]:
        """Get memories eligible for importance decay."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=min_age_days)

        async with self._session_factory() as session:
            conditions = [
                MemoryModel.workspace_id == workspace_id,
                MemoryModel.deleted_at.is_(None),
                MemoryModel.created_at < cutoff,
            ]
            if exclude_pinned:
                conditions.append(MemoryModel.pinned == False)

            result = await session.execute(
                select(MemoryModel).options(_defer_embedding()).where(and_(*conditions))
            )
            return [self._memory_model_to_domain(m) for m in result.scalars().all()]

    # Session optional methods
    async def list_expired_sessions(self, limit: int = 100) -> list[Session]:
        """List expired sessions across all workspaces."""
        now = datetime.now(timezone.utc)

        async with self._session_factory() as db_session:
            result = await db_session.execute(
                select(SessionModel)
                .where(SessionModel.expires_at < now)
                .order_by(SessionModel.expires_at)
                .limit(limit)
            )
            return [self._session_model_to_domain(m) for m in result.scalars().all()]

    async def update_session(
            self,
            workspace_id: str,
            session_id: str,
            **updates
    ) -> Optional[Session]:
        """Update session fields."""
        async with self._session_factory() as db_session:
            result = await db_session.execute(
                select(SessionModel).where(
                    and_(
                        SessionModel.id == session_id,
                        SessionModel.workspace_id == workspace_id,
                    )
                )
            )
            session_model = result.scalar_one_or_none()

            if not session_model:
                return None

            for key, value in updates.items():
                if hasattr(session_model, key):
                    setattr(session_model, key, value)

            if updates.get("committed_at") is not None:
                await self._add_context_event(
                    db_session,
                    workspace_id,
                    session_id,
                    ContextEventKind.COMMIT.value,
                    "session",
                    session_id,
                )

            await db_session.commit()
            await db_session.refresh(session_model)
            return self._session_model_to_domain(session_model)

    async def list_sessions(
            self,
            workspace_id: str,
            context_id: str | None = None,
            include_expired: bool = False,
    ) -> list[Session]:
        """List sessions for a workspace."""
        now = datetime.now(timezone.utc)

        async with self._session_factory() as db_session:
            conditions = [SessionModel.workspace_id == workspace_id]
            if not include_expired:
                conditions.append(SessionModel.expires_at >= now)
            # RESERVED filter — no caller passes a non-None context_id today;
            # kept for the future cross-concern-filtering feature. Effectively
            # dead. See MemoryModel.context_id for the full rationale.
            if context_id is not None:
                conditions.append(SessionModel.context_id == context_id)

            result = await db_session.execute(
                select(SessionModel)
                .where(and_(*conditions))
                .order_by(SessionModel.created_at.desc())
            )
            return [self._session_model_to_domain(m) for m in result.scalars().all()]

    def supports_capability(self, capability: str) -> bool:
        return capability in {
            "session_checkpoints",
            "session_context_events",
            "entity_relations",
        }

    async def _add_context_event(
        self,
        db_session: AsyncSession,
        workspace_id: str,
        session_id: str | None,
        event_kind: str,
        subject_kind: str,
        subject_id: str,
        metadata: dict | None = None,
    ) -> SessionContextEventModel:
        model = SessionContextEventModel(
            workspace_id=workspace_id,
            session_id=session_id,
            event_kind=event_kind,
            subject_kind=subject_kind,
            subject_id=subject_id,
            event_time=datetime.now(timezone.utc),
            meta=metadata or {},
        )
        db_session.add(model)
        if self._context_event_retention_days > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(
                days=self._context_event_retention_days
            )
            await db_session.execute(
                delete(SessionContextEventModel).where(
                    SessionContextEventModel.event_time < cutoff
                )
            )
        await db_session.flush()
        return model

    async def append_context_event(
        self,
        workspace_id: str,
        session_id: str | None,
        event_kind: str,
        subject_kind: str,
        subject_id: str,
        metadata: dict | None = None,
    ) -> SessionContextEvent:
        async with self._session_factory() as session:
            async with session.begin():
                model = await self._add_context_event(
                    session,
                    workspace_id,
                    session_id,
                    event_kind,
                    subject_kind,
                    subject_id,
                    metadata,
                )
                event = self._context_event_model_to_domain(model)
        return event

    async def list_context_events(
        self,
        workspace_id: str,
        session_id: str,
        *,
        after_sequence: int,
        limit: int,
    ) -> list[SessionContextEvent]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(SessionContextEventModel)
                .where(
                    and_(
                        SessionContextEventModel.workspace_id == workspace_id,
                        or_(
                            SessionContextEventModel.session_id == session_id,
                            SessionContextEventModel.session_id.is_(None),
                        ),
                        SessionContextEventModel.sequence > after_sequence,
                    )
                )
                .order_by(SessionContextEventModel.sequence)
                .limit(limit)
            )
            return [
                self._context_event_model_to_domain(model)
                for model in result.scalars().all()
            ]

    async def get_context_event_bounds(
        self,
        workspace_id: str,
        session_id: str,
    ) -> tuple[int, int]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(
                    func.coalesce(func.min(SessionContextEventModel.sequence), 0),
                    func.coalesce(func.max(SessionContextEventModel.sequence), 0),
                ).where(
                    and_(
                        SessionContextEventModel.workspace_id == workspace_id,
                        or_(
                            SessionContextEventModel.session_id == session_id,
                            SessionContextEventModel.session_id.is_(None),
                        ),
                    )
                )
            )
            minimum, maximum = result.one()
            return int(minimum), int(maximum)

    async def create_session_checkpoint(
        self,
        workspace_id: str,
        session_id: str,
        input: SessionCheckpointInput,
    ) -> tuple[SessionCheckpoint, bool]:
        session_domain = await self.get_session(workspace_id, session_id)
        if session_domain is None:
            raise ValueError(f"Session {session_id} not found in workspace {workspace_id}")
        digest = hashlib.sha256(
            f"{workspace_id}\0{session_id}\0{input.idempotency_key}".encode()
        ).hexdigest()
        checkpoint_id = f"chk_{digest[:32]}"
        raw_memory_id = f"mem_chk_{digest[:32]}"
        existing = await self.get_session_checkpoint(
            workspace_id, session_id, checkpoint_id
        )
        if existing is not None:
            if existing.content_hash != input.content_hash:
                raise ValueError(
                    "idempotency key was already used for different checkpoint content"
                )
            return existing, True

        now = datetime.now(timezone.utc)
        raw = Memory(
            id=raw_memory_id,
            tenant_id=session_domain.tenant_id,
            workspace_id=workspace_id,
            context_id=session_domain.context_id,
            session_id=session_id,
            content=input.transcript_segment,
            content_hash=input.content_hash,
            type=MemoryType.EPISODIC,
            subtype="checkpoint",
            metadata={
                "source": "session_checkpoint",
                "source_kind": input.source_kind,
                "source_sequence": input.source_sequence,
                "source_boundary": input.source_boundary,
            },
            created_at=now,
            updated_at=now,
        )
        # The ordinary memory validator trims surrounding whitespace; raw
        # checkpoint content must remain byte-for-byte identical to its hash.
        raw = raw.model_copy(update={"content": input.transcript_segment})
        await self.mutate_memory(
            MemoryMutation(
                action="create",
                memory=raw,
                operation_id=f"checkpoint-raw:{digest}",
                request_hash=canonical_hash(
                    {"action": "checkpoint_raw", "content_hash": input.content_hash}
                ),
                expected_etag="*",
            )
        )

        try:
            async with self._session_factory() as session:
                async with session.begin():
                    model = SessionCheckpointModel(
                        id=checkpoint_id,
                        workspace_id=workspace_id,
                        session_id=session_id,
                        raw_memory_id=raw_memory_id,
                        source_kind=input.source_kind,
                        source_sequence=input.source_sequence,
                        source_boundary=input.source_boundary,
                        content_hash=input.content_hash,
                        byte_count=len(input.transcript_segment.encode("utf-8")),
                        capture_status=CheckpointCaptureStatus.DURABLE.value,
                        index_status=CheckpointWorkStatus.PENDING.value,
                        enrichment_status=CheckpointWorkStatus.SKIPPED.value,
                        idempotency_key=input.idempotency_key,
                        created_at=now,
                        updated_at=now,
                    )
                    session.add(model)
                    await self._add_context_event(
                        session,
                        workspace_id,
                        session_id,
                        ContextEventKind.CHECKPOINT.value,
                        "checkpoint",
                        checkpoint_id,
                        {"raw_memory_id": raw_memory_id},
                    )
        except Exception:
            existing = await self.get_session_checkpoint(
                workspace_id, session_id, checkpoint_id
            )
            if existing is not None and existing.content_hash == input.content_hash:
                return existing, True
            raise
        checkpoint = await self.get_session_checkpoint(
            workspace_id, session_id, checkpoint_id
        )
        return checkpoint, False

    async def get_session_checkpoint(
        self,
        workspace_id: str,
        session_id: str,
        checkpoint_id: str,
    ) -> SessionCheckpoint | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(SessionCheckpointModel).where(
                    and_(
                        SessionCheckpointModel.id == checkpoint_id,
                        SessionCheckpointModel.workspace_id == workspace_id,
                        SessionCheckpointModel.session_id == session_id,
                    )
                )
            )
            model = result.scalar_one_or_none()
            return self._checkpoint_model_to_domain(model) if model else None

    async def list_session_checkpoints(
        self,
        workspace_id: str,
        session_id: str,
        *,
        limit: int = 20,
    ) -> list[SessionCheckpoint]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(SessionCheckpointModel)
                .where(
                    and_(
                        SessionCheckpointModel.workspace_id == workspace_id,
                        SessionCheckpointModel.session_id == session_id,
                    )
                )
                .order_by(
                    SessionCheckpointModel.created_at.desc(),
                    SessionCheckpointModel.id.desc(),
                )
                .limit(limit)
            )
            return [
                self._checkpoint_model_to_domain(model)
                for model in result.scalars().all()
            ]

    async def update_session_checkpoint_status(
        self,
        workspace_id: str,
        session_id: str,
        checkpoint_id: str,
        *,
        index_status: str | None = None,
        enrichment_status: str | None = None,
    ) -> SessionCheckpoint | None:
        values: dict[str, Any] = {"updated_at": datetime.now(timezone.utc)}
        if index_status is not None:
            values["index_status"] = index_status
        if enrichment_status is not None:
            values["enrichment_status"] = enrichment_status
        async with self._session_factory() as session:
            await session.execute(
                update(SessionCheckpointModel)
                .where(
                    and_(
                        SessionCheckpointModel.id == checkpoint_id,
                        SessionCheckpointModel.workspace_id == workspace_id,
                        SessionCheckpointModel.session_id == session_id,
                    )
                )
                .values(**values)
            )
            await session.commit()
        return await self.get_session_checkpoint(
            workspace_id, session_id, checkpoint_id
        )

    async def upsert_entity_relation(
        self,
        relation: EntityRelation,
        evidence: EntityRelationEvidence,
    ) -> tuple[EntityRelation, bool]:
        """Upsert one canonical edge and one idempotent source-evidence row."""
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        if evidence.workspace_id != relation.workspace_id:
            raise ValueError("relation evidence must use the edge workspace")
        async with self._session_factory() as session:
            async with session.begin():
                endpoints = await session.execute(
                    select(func.count(EntityModel.id)).where(
                        and_(
                            EntityModel.workspace_id == relation.workspace_id,
                            EntityModel.id.in_([
                                relation.source_entity_id,
                                relation.target_entity_id,
                            ]),
                            EntityModel.status == "active",
                        )
                    )
                )
                if int(endpoints.scalar_one()) != 2:
                    raise ValueError(
                        "relation endpoints must be active entities in the same workspace"
                    )
                source_memory = await session.execute(
                    select(MemoryModel.id).where(
                        and_(
                            MemoryModel.id == evidence.source_memory_id,
                            MemoryModel.workspace_id == relation.workspace_id,
                            MemoryModel.deleted_at.is_(None),
                        )
                    )
                )
                if source_memory.scalar_one_or_none() is None:
                    raise ValueError(
                        "relation evidence memory must be active in the same workspace"
                    )

                edge_values = {
                    "id": relation.id,
                    "workspace_id": relation.workspace_id,
                    "source_entity_id": relation.source_entity_id,
                    "target_entity_id": relation.target_entity_id,
                    "relationship": relation.relationship,
                    "direction": relation.direction,
                    "confidence": relation.confidence,
                    "active": True,
                    "created_at": relation.created_at,
                    "updated_at": relation.updated_at,
                }
                await session.execute(
                    pg_insert(EntityRelationModel)
                    .values(**edge_values)
                    .on_conflict_do_nothing()
                )
                result = await session.execute(
                    select(EntityRelationModel)
                    .where(
                        and_(
                            EntityRelationModel.workspace_id == relation.workspace_id,
                            EntityRelationModel.source_entity_id == relation.source_entity_id,
                            EntityRelationModel.target_entity_id == relation.target_entity_id,
                            EntityRelationModel.relationship == relation.relationship,
                            EntityRelationModel.active.is_(True),
                        )
                    )
                    .with_for_update()
                )
                relation_model = result.scalar_one()

                span_start_condition = (
                    EntityRelationEvidenceModel.source_span_start.is_(None)
                    if evidence.source_span_start is None
                    else EntityRelationEvidenceModel.source_span_start
                    == evidence.source_span_start
                )
                span_end_condition = (
                    EntityRelationEvidenceModel.source_span_end.is_(None)
                    if evidence.source_span_end is None
                    else EntityRelationEvidenceModel.source_span_end
                    == evidence.source_span_end
                )
                evidence_values = {
                    "id": evidence.id,
                    "workspace_id": relation.workspace_id,
                    "relation_id": relation_model.id,
                    "source_memory_id": evidence.source_memory_id,
                    "evidence_kind": evidence.evidence_kind.value,
                    "source_span_start": evidence.source_span_start,
                    "source_span_end": evidence.source_span_end,
                    "excerpt_hash": evidence.excerpt_hash,
                    "confidence": evidence.confidence,
                    "extraction_method": evidence.extraction_method,
                    "active": True,
                    "created_at": evidence.created_at,
                }
                inserted = await session.execute(
                    pg_insert(EntityRelationEvidenceModel)
                    .values(**evidence_values)
                    .on_conflict_do_nothing()
                    .returning(EntityRelationEvidenceModel.id)
                )
                inserted_evidence_id = inserted.scalar_one_or_none()
                evidence_result = await session.execute(
                    select(EntityRelationEvidenceModel)
                    .where(
                        and_(
                            EntityRelationEvidenceModel.relation_id == relation_model.id,
                            EntityRelationEvidenceModel.source_memory_id
                            == evidence.source_memory_id,
                            span_start_condition,
                            span_end_condition,
                            EntityRelationEvidenceModel.excerpt_hash
                            == evidence.excerpt_hash,
                        )
                    )
                    .with_for_update()
                )
                evidence_model = evidence_result.scalar_one_or_none()
                if evidence_model is None:
                    raise RuntimeError("relation evidence upsert did not produce a row")
                duplicate = inserted_evidence_id is None and evidence_model.active
                evidence_model.active = True
                evidence_model.confidence = evidence.confidence
                evidence_model.evidence_kind = evidence.evidence_kind.value
                evidence_model.extraction_method = evidence.extraction_method
                await session.flush()

                aggregate = await session.execute(
                    select(func.avg(EntityRelationEvidenceModel.confidence)).where(
                        and_(
                            EntityRelationEvidenceModel.relation_id == relation_model.id,
                            EntityRelationEvidenceModel.active.is_(True),
                        )
                    )
                )
                relation_model.confidence = float(aggregate.scalar_one())
                relation_model.active = True
                relation_model.updated_at = datetime.now(timezone.utc)
                await session.flush()
                domain = self._entity_relation_model_to_domain(relation_model)
        return domain, duplicate

    async def traverse_entity_relations(
        self,
        workspace_id: str,
        seed_entity_ids: list[str],
        *,
        relationships: list[str] | None,
        direction: str,
        max_hops: int,
        max_edges: int,
    ) -> list[EntityRelationPath]:
        if not seed_entity_ids or max_edges <= 0:
            return []
        frontier = list(dict.fromkeys(seed_entity_ids))
        paths: list[tuple[str, list[str], list[EntityRelation]]] = [
            (seed, [seed], []) for seed in frontier
        ]
        completed: list[tuple[str, list[str], list[EntityRelation]]] = []
        remaining = max_edges
        async with self._session_factory() as session:
            for _hop in range(min(max_hops, 2)):
                if not frontier or remaining <= 0:
                    break
                endpoint_conditions = []
                if direction in ("outgoing", "both"):
                    endpoint_conditions.append(
                        EntityRelationModel.source_entity_id.in_(frontier)
                    )
                if direction in ("incoming", "both"):
                    endpoint_conditions.append(
                        EntityRelationModel.target_entity_id.in_(frontier)
                    )
                conditions = [
                    EntityRelationModel.workspace_id == workspace_id,
                    EntityRelationModel.active.is_(True),
                    or_(*endpoint_conditions),
                ]
                if relationships:
                    conditions.append(
                        EntityRelationModel.relationship.in_(relationships)
                    )
                result = await session.execute(
                    select(EntityRelationModel)
                    .where(and_(*conditions))
                    .order_by(EntityRelationModel.id)
                    .limit(remaining)
                )
                edges = [
                    self._entity_relation_model_to_domain(model)
                    for model in result.scalars().all()
                ]
                remaining -= len(edges)
                next_paths: list[tuple[str, list[str], list[EntityRelation]]] = []
                next_frontier: list[str] = []
                for seed, nodes, path_edges in paths:
                    current = nodes[-1]
                    for edge in edges:
                        target: str | None = None
                        if (
                            edge.source_entity_id == current
                            and direction in ("outgoing", "both")
                        ):
                            target = edge.target_entity_id
                        elif (
                            edge.target_entity_id == current
                            and direction in ("incoming", "both")
                        ):
                            target = edge.source_entity_id
                        if target is None or target in nodes:
                            continue
                        candidate = (
                            seed,
                            [*nodes, target],
                            [*path_edges, edge],
                        )
                        completed.append(candidate)
                        next_paths.append(candidate)
                        next_frontier.append(target)
                paths = next_paths
                frontier = list(dict.fromkeys(next_frontier))

            relation_ids = list(
                dict.fromkeys(
                    edge.id
                    for _seed, _nodes, path_edges in completed
                    for edge in path_edges
                )
            )
            evidence_by_relation: dict[str, list[tuple[str, str]]] = {}
            if relation_ids:
                evidence_result = await session.execute(
                    select(
                        EntityRelationEvidenceModel.id,
                        EntityRelationEvidenceModel.relation_id,
                        EntityRelationEvidenceModel.source_memory_id,
                    )
                    .where(
                        and_(
                            EntityRelationEvidenceModel.workspace_id == workspace_id,
                            EntityRelationEvidenceModel.relation_id.in_(relation_ids),
                            EntityRelationEvidenceModel.active.is_(True),
                        )
                    )
                    .order_by(
                        EntityRelationEvidenceModel.relation_id,
                        EntityRelationEvidenceModel.id,
                    )
                )
                for evidence_id, relation_id, memory_id in evidence_result.all():
                    evidence_by_relation.setdefault(relation_id, []).append(
                        (evidence_id, memory_id)
                    )

        results: list[EntityRelationPath] = []
        for seed, nodes, edges in completed:
            evidence_rows = [
                item
                for edge in edges
                for item in evidence_by_relation.get(edge.id, [])
            ]
            if not evidence_rows:
                continue
            results.append(
                EntityRelationPath(
                    seed_entity_id=seed,
                    entity_ids=nodes,
                    relations=edges,
                    evidence_ids=list(
                        dict.fromkeys(item[0] for item in evidence_rows)
                    ),
                    evidence_memory_ids=list(
                        dict.fromkeys(item[1] for item in evidence_rows)
                    ),
                )
            )
        return results

    async def _deactivate_relation_evidence_in_session(
        self,
        session: AsyncSession,
        workspace_id: str,
        memory_id: str,
    ) -> int:
        relation_result = await session.execute(
            select(EntityRelationEvidenceModel.relation_id).where(
                and_(
                    EntityRelationEvidenceModel.workspace_id == workspace_id,
                    EntityRelationEvidenceModel.source_memory_id == memory_id,
                    EntityRelationEvidenceModel.active.is_(True),
                )
            )
        )
        relation_ids = list(dict.fromkeys(relation_result.scalars().all()))
        result = await session.execute(
            update(EntityRelationEvidenceModel)
            .where(
                and_(
                    EntityRelationEvidenceModel.workspace_id == workspace_id,
                    EntityRelationEvidenceModel.source_memory_id == memory_id,
                    EntityRelationEvidenceModel.active.is_(True),
                )
            )
            .values(active=False)
        )
        await self._refresh_relation_aggregates(session, workspace_id, relation_ids)
        return int(result.rowcount or 0)

    async def _refresh_relation_aggregates(
        self,
        session: AsyncSession,
        workspace_id: str,
        relation_ids: list[str],
    ) -> None:
        if not relation_ids:
            return
        aggregate_result = await session.execute(
            select(
                EntityRelationEvidenceModel.relation_id,
                func.avg(EntityRelationEvidenceModel.confidence),
            )
            .where(
                and_(
                    EntityRelationEvidenceModel.workspace_id == workspace_id,
                    EntityRelationEvidenceModel.relation_id.in_(relation_ids),
                    EntityRelationEvidenceModel.active.is_(True),
                )
            )
            .group_by(EntityRelationEvidenceModel.relation_id)
        )
        averages = {
            relation_id: float(average)
            for relation_id, average in aggregate_result.all()
        }
        now = datetime.now(timezone.utc)
        for relation_id in relation_ids:
            values: dict[str, Any] = {
                "active": relation_id in averages,
                "updated_at": now,
            }
            if relation_id in averages:
                values["confidence"] = averages[relation_id]
            await session.execute(
                update(EntityRelationModel)
                .where(
                    and_(
                        EntityRelationModel.id == relation_id,
                        EntityRelationModel.workspace_id == workspace_id,
                    )
                )
                .values(**values)
            )

    async def deactivate_relation_evidence_for_memory(
        self,
        workspace_id: str,
        memory_id: str,
    ) -> int:
        async with self._session_factory() as session:
            async with session.begin():
                return await self._deactivate_relation_evidence_in_session(
                    session, workspace_id, memory_id
                )

    async def reconcile_entity_relations_after_merge(
        self,
        workspace_id: str,
        source_entity_id: str,
        target_entity_id: str,
    ) -> int:
        async with self._session_factory() as session:
            async with session.begin():
                result = await session.execute(
                    select(EntityRelationModel)
                    .where(
                        and_(
                            EntityRelationModel.workspace_id == workspace_id,
                            EntityRelationModel.active.is_(True),
                            or_(
                                EntityRelationModel.source_entity_id
                                == source_entity_id,
                                EntityRelationModel.target_entity_id
                                == source_entity_id,
                            ),
                        )
                    )
                    .order_by(EntityRelationModel.id)
                    .with_for_update()
                )
                source_relations = result.scalars().all()
                changed = 0
                refresh_ids: list[str] = []
                for relation_model in source_relations:
                    new_source = (
                        target_entity_id
                        if relation_model.source_entity_id == source_entity_id
                        else relation_model.source_entity_id
                    )
                    new_target = (
                        target_entity_id
                        if relation_model.target_entity_id == source_entity_id
                        else relation_model.target_entity_id
                    )
                    if new_source == new_target:
                        await session.execute(
                            update(EntityRelationEvidenceModel)
                            .where(
                                EntityRelationEvidenceModel.relation_id
                                == relation_model.id
                            )
                            .values(active=False)
                        )
                        relation_model.active = False
                        relation_model.updated_at = datetime.now(timezone.utc)
                        changed += 1
                        continue

                    collision_result = await session.execute(
                        select(EntityRelationModel)
                        .where(
                            and_(
                                EntityRelationModel.workspace_id == workspace_id,
                                EntityRelationModel.source_entity_id == new_source,
                                EntityRelationModel.target_entity_id == new_target,
                                EntityRelationModel.relationship
                                == relation_model.relationship,
                                EntityRelationModel.active.is_(True),
                                EntityRelationModel.id != relation_model.id,
                            )
                        )
                        .with_for_update()
                    )
                    collision = collision_result.scalar_one_or_none()
                    if collision is None:
                        relation_model.source_entity_id = new_source
                        relation_model.target_entity_id = new_target
                        relation_model.updated_at = datetime.now(timezone.utc)
                        refresh_ids.append(relation_model.id)
                        changed += 1
                        continue

                    existing_result = await session.execute(
                        select(EntityRelationEvidenceModel).where(
                            EntityRelationEvidenceModel.relation_id == collision.id
                        )
                    )
                    existing_keys = {
                        (
                            row.source_memory_id,
                            row.source_span_start,
                            row.source_span_end,
                            row.excerpt_hash,
                        )
                        for row in existing_result.scalars().all()
                    }
                    moving_result = await session.execute(
                        select(EntityRelationEvidenceModel).where(
                            EntityRelationEvidenceModel.relation_id
                            == relation_model.id
                        )
                    )
                    for evidence_model in moving_result.scalars().all():
                        key = (
                            evidence_model.source_memory_id,
                            evidence_model.source_span_start,
                            evidence_model.source_span_end,
                            evidence_model.excerpt_hash,
                        )
                        if key in existing_keys:
                            await session.delete(evidence_model)
                        else:
                            evidence_model.relation_id = collision.id
                            existing_keys.add(key)
                    await session.flush()
                    await session.delete(relation_model)
                    refresh_ids.append(collision.id)
                    changed += 1
                await session.flush()
                await self._refresh_relation_aggregates(
                    session,
                    workspace_id,
                    list(dict.fromkeys(refresh_ids)),
                )
                return changed

    # Contradiction operations
    async def create_contradiction(self, contradiction: ContradictionRecord) -> ContradictionRecord:
        """Store a contradiction record."""
        model = ContradictionModel(
            id=contradiction.id,
            workspace_id=contradiction.workspace_id,
            memory_a_id=contradiction.memory_a_id,
            memory_b_id=contradiction.memory_b_id,
            contradiction_type=contradiction.contradiction_type,
            confidence=contradiction.confidence,
            detection_method=contradiction.detection_method,
            detected_at=contradiction.detected_at,
            resolved_at=contradiction.resolved_at,
            resolution=contradiction.resolution,
            merged_content=contradiction.merged_content,
            newer_memory_id=contradiction.newer_memory_id,
        )

        async with self._session_factory() as session:
            session.add(model)
            await session.commit()
            await session.refresh(model)

        return self._contradiction_model_to_domain(model)

    async def get_contradiction(self, workspace_id: str, contradiction_id: str) -> Optional[ContradictionRecord]:
        """Get a specific contradiction."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(ContradictionModel).where(
                    and_(
                        ContradictionModel.id == contradiction_id,
                        ContradictionModel.workspace_id == workspace_id,
                    )
                )
            )
            model = result.scalar_one_or_none()
            if not model:
                return None
            return self._contradiction_model_to_domain(model)

    async def get_unresolved_contradictions(self, workspace_id: str, limit: int = 10) -> list[ContradictionRecord]:
        """Get unresolved contradictions for a workspace."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(ContradictionModel)
                .where(
                    and_(
                        ContradictionModel.workspace_id == workspace_id,
                        ContradictionModel.resolved_at.is_(None),
                    )
                )
                .order_by(ContradictionModel.detected_at.desc())
                .limit(limit)
            )
            return [self._contradiction_model_to_domain(m) for m in result.scalars().all()]

    async def resolve_contradiction(
        self,
        workspace_id: str,
        contradiction_id: str,
        resolution: str,
        merged_content: Optional[str] = None,
    ) -> Optional[ContradictionRecord]:
        """Resolve a contradiction."""
        now = datetime.now(timezone.utc)

        async with self._session_factory() as session:
            result = await session.execute(
                select(ContradictionModel).where(
                    and_(
                        ContradictionModel.id == contradiction_id,
                        ContradictionModel.workspace_id == workspace_id,
                    )
                )
            )
            model = result.scalar_one_or_none()
            if not model:
                return None

            model.resolved_at = now
            model.resolution = resolution
            model.merged_content = merged_content

            await session.commit()
            await session.refresh(model)
            return self._contradiction_model_to_domain(model)

    def _contradiction_model_to_domain(self, model: ContradictionModel) -> ContradictionRecord:
        """Convert ORM model to domain dataclass."""
        return ContradictionRecord(
            id=model.id,
            workspace_id=model.workspace_id,
            memory_a_id=model.memory_a_id,
            memory_b_id=model.memory_b_id,
            contradiction_type=model.contradiction_type,
            confidence=model.confidence,
            detection_method=model.detection_method,
            detected_at=model.detected_at,
            resolved_at=model.resolved_at,
            resolution=model.resolution,
            merged_content=model.merged_content,
            newer_memory_id=model.newer_memory_id,
        )

    async def get_superseded_memory_ids(self, workspace_id: str, memory_ids: list[str]) -> set[str]:
        """Indexed form of the base scan — see ``StorageBackend.get_superseded_memory_ids``.

        Backed by ``idx_contradictions_superseded``. Records with no recorded direction are
        excluded in SQL: they say the pair conflicts, not which side is current, and that
        is never guessed at. Resolved contradictions are excluded for the same reason the
        base implementation excludes them — resolution means the operator has dealt with
        it, and continuing to penalise would make the resolution invisible.
        """
        if not memory_ids:
            return set()
        async with self._session_factory() as session:
            result = await session.execute(
                select(
                    ContradictionModel.memory_a_id,
                    ContradictionModel.memory_b_id,
                    ContradictionModel.newer_memory_id,
                ).where(
                    and_(
                        ContradictionModel.workspace_id == workspace_id,
                        ContradictionModel.resolved_at.is_(None),
                        ContradictionModel.newer_memory_id.isnot(None),
                        or_(
                            ContradictionModel.memory_a_id.in_(memory_ids),
                            ContradictionModel.memory_b_id.in_(memory_ids),
                        ),
                    )
                )
            )
            wanted = set(memory_ids)
            superseded: set[str] = set()
            for memory_a_id, memory_b_id, newer in result.all():
                for candidate in (memory_a_id, memory_b_id):
                    if candidate in wanted and candidate != newer:
                        superseded.add(candidate)
            return superseded

    # Chat history operations

    @staticmethod
    async def _resolve_thread_row_id(
        db_session, workspace_id: str, client_id: str, user_id: Optional[str]
    ) -> Optional[str]:
        """Resolve the opaque surrogate ``row_id`` for an owner-scoped thread.

        SECURITY: threads are identified by ``(workspace_id, COALESCE(user_id,''),
        id)`` so a shared client id like ``"_default"`` in the ``_user_chat``
        sentinel resolves to the correct owner's thread. ``user_id=None`` matches
        rows with a NULL ``user_id`` (workspace-owned). Returns ``None`` when no
        such thread exists. The returned row_id is INTERNAL — never exposed to
        callers — and is the only value ``chat_messages.thread_id`` references.
        """
        result = await db_session.execute(
            select(ChatThreadModel.row_id).where(
                and_(
                    ChatThreadModel.workspace_id == workspace_id,
                    ChatThreadModel.id == client_id,
                    func.coalesce(ChatThreadModel.user_id, "") == (user_id or ""),
                )
            )
        )
        return result.scalar_one_or_none()

    async def create_thread(self, thread: ChatThread) -> ChatThread:
        """Store a new chat thread. row_id (the surrogate PK) is auto-generated."""
        async with self._session_factory() as db_session:
            model = ChatThreadModel(
                id=thread.id,
                workspace_id=thread.workspace_id,
                tenant_id=thread.tenant_id,
                context_id=thread.context_id or "_default",
                user_id=thread.user_id,
                observer_id=thread.observer_id,
                subject_id=thread.subject_id,
                title=thread.title,
                meta=thread.metadata or {},
                message_count=thread.message_count or 0,
                last_decomposed_at=thread.last_decomposed_at,
                last_decomposed_index=thread.last_decomposed_index,
                expires_at=thread.expires_at,
                created_at=thread.created_at,
                updated_at=thread.updated_at,
                scope=thread.scope,
                ownership=thread.ownership or 'user',
                idle_action=thread.idle_action,
                hidden_at=thread.hidden_at,
                parent_thread=thread.parent_thread,
            )
            db_session.add(model)
            await db_session.commit()
            await db_session.refresh(model)
            return self._thread_model_to_domain(model)

    async def get_thread(
        self, workspace_id: str, thread_id: str, user_id: Optional[str] = None
    ) -> Optional[ChatThread]:
        """Get chat thread by ID, scoped by owner (workspace_id, user_id, id)."""
        async with self._session_factory() as db_session:
            result = await db_session.execute(
                select(ChatThreadModel).where(
                    and_(
                        ChatThreadModel.id == thread_id,
                        ChatThreadModel.workspace_id == workspace_id,
                        func.coalesce(ChatThreadModel.user_id, "") == (user_id or ""),
                    )
                )
            )
            model = result.scalar_one_or_none()
            if not model:
                return None
            return self._thread_model_to_domain(model)

    async def list_threads(
        self,
        workspace_id: str,
        user_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        scope_filter: Optional[str] = None,
        ownership_filter: Optional[str] = None,
        include_hidden: bool = False,
        parent_thread: Optional[str] = None,
    ) -> list[ChatThread]:
        """List chat threads in a workspace."""
        async with self._session_factory() as db_session:
            conditions = [ChatThreadModel.workspace_id == workspace_id]
            if not include_hidden:
                conditions.append(ChatThreadModel.hidden_at.is_(None))
            # Default: top-level only (parent_thread IS NULL); a value lists children.
            if parent_thread is None:
                conditions.append(ChatThreadModel.parent_thread.is_(None))
            else:
                conditions.append(ChatThreadModel.parent_thread == parent_thread)
            if user_id is not None:
                conditions.append(ChatThreadModel.user_id == user_id)

            # scope_filter="web"  → scope='web' OR scope IS NULL (NULL ≡ web)
            # scope_filter="office" → scope='office'
            # scope_filter=None   → no scope restriction (return all)
            if scope_filter == "web":
                from sqlalchemy import or_
                conditions.append(
                    or_(ChatThreadModel.scope == "web", ChatThreadModel.scope.is_(None))
                )
            elif scope_filter is not None:
                conditions.append(ChatThreadModel.scope == scope_filter)

            if ownership_filter is not None:
                conditions.append(ChatThreadModel.ownership == ownership_filter)

            result = await db_session.execute(
                select(ChatThreadModel)
                .where(and_(*conditions))
                .order_by(ChatThreadModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return [self._thread_model_to_domain(m) for m in result.scalars().all()]

    async def list_user_threads(
        self,
        tenant_id: str,
        user_id: str,
        ownership: str = 'user',
        scope_filter: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        include_hidden: bool = False,
        parent_thread: Optional[str] = None,
    ) -> list[ChatThread]:
        """List chat threads owned by a user across all workspaces.

        Filters by (tenant_id, user_id, ownership) — does NOT constrain workspace_id.
        Used for the user-session-scoped right rail; each returned thread carries
        its existing metadata (where callers may stash a preferredWorkspace).
        """
        async with self._session_factory() as db_session:
            conditions = [
                ChatThreadModel.tenant_id == tenant_id,
                ChatThreadModel.user_id == user_id,
                ChatThreadModel.ownership == ownership,
            ]
            if not include_hidden:
                conditions.append(ChatThreadModel.hidden_at.is_(None))
            if parent_thread is None:
                conditions.append(ChatThreadModel.parent_thread.is_(None))
            else:
                conditions.append(ChatThreadModel.parent_thread == parent_thread)

            if scope_filter == "web":
                from sqlalchemy import or_
                conditions.append(
                    or_(ChatThreadModel.scope == "web", ChatThreadModel.scope.is_(None))
                )
            elif scope_filter is not None:
                conditions.append(ChatThreadModel.scope == scope_filter)

            result = await db_session.execute(
                select(ChatThreadModel)
                .where(and_(*conditions))
                .order_by(ChatThreadModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return [self._thread_model_to_domain(m) for m in result.scalars().all()]

    async def update_thread(
        self, workspace_id: str, thread_id: str, user_id: Optional[str] = None, **updates
    ) -> Optional[ChatThread]:
        """Update thread fields, scoped by owner."""
        async with self._session_factory() as db_session:
            result = await db_session.execute(
                select(ChatThreadModel).where(
                    and_(
                        ChatThreadModel.id == thread_id,
                        ChatThreadModel.workspace_id == workspace_id,
                        func.coalesce(ChatThreadModel.user_id, "") == (user_id or ""),
                    )
                )
            )
            model = result.scalar_one_or_none()
            if not model:
                return None

            for key, value in updates.items():
                attr = 'meta' if key == 'metadata' else key
                if hasattr(model, attr):
                    setattr(model, attr, value)

            model.updated_at = datetime.now(timezone.utc)
            await db_session.commit()
            await db_session.refresh(model)
            return self._thread_model_to_domain(model)

    async def delete_thread(
        self, workspace_id: str, thread_id: str, user_id: Optional[str] = None
    ) -> bool:
        """Delete a thread and all its messages, cascading to sub-threads.

        Owner-scoped by (workspace_id, user_id, id). Children (parent_thread ==
        thread_id, same owner) are removed first so a parent delete never orphans
        sub-threads; recurses for arbitrary depth. ``parent_thread`` is a client-id
        string, so children are resolved within the same (workspace, user).
        Messages cascade at the DB level via the chat_messages FK (row_id).
        """
        async with self._session_factory() as db_session:
            child_rows = await db_session.execute(
                select(ChatThreadModel.id).where(
                    and_(
                        ChatThreadModel.parent_thread == thread_id,
                        ChatThreadModel.workspace_id == workspace_id,
                        func.coalesce(ChatThreadModel.user_id, "") == (user_id or ""),
                    )
                )
            )
            child_ids = [r[0] for r in child_rows.all()]
        for child_id in child_ids:
            await self.delete_thread(workspace_id, child_id, user_id=user_id)

        async with self._session_factory() as db_session:
            result = await db_session.execute(
                delete(ChatThreadModel).where(
                    and_(
                        ChatThreadModel.id == thread_id,
                        ChatThreadModel.workspace_id == workspace_id,
                        func.coalesce(ChatThreadModel.user_id, "") == (user_id or ""),
                    )
                )
            )
            await db_session.commit()
            return result.rowcount > 0

    def _strip_null_bytes(self, obj):
        if isinstance(obj, str):
            return obj.replace('\x00', '')
        if isinstance(obj, list):
            return [self._strip_null_bytes(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self._strip_null_bytes(v) for k, v in obj.items()}
        return obj

    async def append_messages(
        self,
        workspace_id: str,
        thread_id: str,
        messages: list[MessageInput],
        user_id: Optional[str] = None,
    ) -> list[ChatMessage]:
        """Append messages to a thread, scoped by owner.

        Messages reference the thread's surrogate ``row_id`` (not its client id),
        so a shared client id like ``"_default"`` in the ``_user_chat`` sentinel is
        unambiguous per owner.
        """
        import uuid

        async with self._session_factory() as db_session:
            # Verify thread exists (owner-scoped) and capture its surrogate row_id.
            thread_result = await db_session.execute(
                select(ChatThreadModel).where(
                    and_(
                        ChatThreadModel.id == thread_id,
                        ChatThreadModel.workspace_id == workspace_id,
                        func.coalesce(ChatThreadModel.user_id, "") == (user_id or ""),
                    )
                )
            )
            thread_model = thread_result.scalar_one_or_none()
            if not thread_model:
                return []

            row_id = thread_model.row_id
            now = datetime.now(timezone.utc)
            current_count = thread_model.message_count or 0
            created = []

            for i, msg in enumerate(messages):
                msg_id = msg.id or str(uuid.uuid4())
                content = msg.content if isinstance(msg.content, str) else [c.model_dump() for c in msg.content] if isinstance(msg.content, list) else msg.content
                content = self._strip_null_bytes(content)
                model = ChatMessageModel(
                    id=msg_id,
                    thread_id=row_id,
                    message_index=current_count + i,
                    role=msg.role,
                    content=content,
                    meta=msg.metadata or {},
                    created_at=now,
                )
                db_session.add(model)
                created.append(model)

            thread_model.message_count = (thread_model.message_count or 0) + len(messages)
            thread_model.updated_at = now
            # Clear hidden_at so a new message revives (un-archives) a hidden thread.
            thread_model.hidden_at = None

            await db_session.commit()
            for m in created:
                await db_session.refresh(m)
            return [self._message_model_to_domain(m) for m in created]

    async def get_messages(
        self,
        workspace_id: str,
        thread_id: str,
        limit: int = 100,
        offset: int = 0,
        after_index: Optional[int] = None,
        order: str = "asc",
        user_id: Optional[str] = None,
    ) -> list[ChatMessage]:
        """Get messages from a thread, scoped by owner."""
        async with self._session_factory() as db_session:
            # Resolve the owner-scoped thread to its surrogate row_id; messages
            # are keyed on row_id, NOT the client id.
            row_id = await self._resolve_thread_row_id(
                db_session, workspace_id, thread_id, user_id
            )
            if row_id is None:
                return []

            query = select(ChatMessageModel).where(
                ChatMessageModel.thread_id == row_id
            )

            if order == "desc":
                query = query.order_by(ChatMessageModel.created_at.desc())
            else:
                query = query.order_by(ChatMessageModel.created_at.asc())

            query = query.limit(limit).offset(offset)

            result = await db_session.execute(query)
            return [self._message_model_to_domain(m) for m in result.scalars().all()]

    async def get_message_count(
        self, workspace_id: str, thread_id: str, user_id: Optional[str] = None
    ) -> int:
        """Get total message count for a thread, scoped by owner."""
        async with self._session_factory() as db_session:
            row_id = await self._resolve_thread_row_id(
                db_session, workspace_id, thread_id, user_id
            )
            if row_id is None:
                return 0
            result = await db_session.execute(
                select(func.count(ChatMessageModel.id)).where(
                    ChatMessageModel.thread_id == row_id
                )
            )
            return result.scalar_one() or 0

    async def delete_message(
        self, workspace_id: str, thread_id: str, message_id: str, user_id: Optional[str] = None
    ) -> bool:
        """Delete a single message by ID within an owner-scoped thread."""
        async with self._session_factory() as db_session:
            # Resolve the owner-scoped thread to its surrogate row_id before deleting.
            row_id = await self._resolve_thread_row_id(
                db_session, workspace_id, thread_id, user_id
            )
            if row_id is None:
                return False

            result = await db_session.execute(
                delete(ChatMessageModel).where(
                    and_(
                        ChatMessageModel.id == message_id,
                        ChatMessageModel.thread_id == row_id,
                    )
                )
            )
            await db_session.commit()
            return result.rowcount > 0

    async def list_expired_threads(self, limit: int = 100) -> list[ChatThread]:
        """List expired chat threads across all workspaces."""
        now = datetime.now(timezone.utc)
        async with self._session_factory() as db_session:
            result = await db_session.execute(
                select(ChatThreadModel)
                .where(
                    and_(
                        ChatThreadModel.expires_at.isnot(None),
                        ChatThreadModel.expires_at < now,
                    )
                )
                .limit(limit)
            )
            return [self._thread_model_to_domain(m) for m in result.scalars().all()]

    async def list_idle_threads(
        self,
        updated_before: datetime,
        *,
        idle_action: str,
        only_hidden: Optional[bool] = None,
        limit: int = 100,
    ) -> list[ChatThread]:
        """List threads whose idle policy is due (idle_action set, updated_at < cutoff)."""
        async with self._session_factory() as db_session:
            conditions = [
                ChatThreadModel.idle_action == idle_action,
                ChatThreadModel.updated_at < updated_before,
            ]
            if only_hidden is True:
                conditions.append(ChatThreadModel.hidden_at.isnot(None))
            elif only_hidden is False:
                conditions.append(ChatThreadModel.hidden_at.is_(None))
            result = await db_session.execute(
                select(ChatThreadModel)
                .where(and_(*conditions))
                .order_by(ChatThreadModel.updated_at.asc())
                .limit(limit)
            )
            return [self._thread_model_to_domain(m) for m in result.scalars().all()]

    async def list_hidden_threads(self, hidden_before: datetime, limit: int = 100) -> list[ChatThread]:
        """List hidden/archived threads hidden before ``hidden_before`` (grace purge)."""
        async with self._session_factory() as db_session:
            result = await db_session.execute(
                select(ChatThreadModel)
                .where(
                    and_(
                        ChatThreadModel.hidden_at.isnot(None),
                        ChatThreadModel.hidden_at < hidden_before,
                    )
                )
                .order_by(ChatThreadModel.hidden_at.asc())
                .limit(limit)
            )
            return [self._thread_model_to_domain(m) for m in result.scalars().all()]

    async def hide_thread(
        self, workspace_id: str, thread_id: str, user_id: Optional[str] = None
    ) -> Optional[ChatThread]:
        """Archive (hide) a thread; sets hidden_at WITHOUT bumping updated_at. Owner-scoped."""
        async with self._session_factory() as db_session:
            result = await db_session.execute(
                select(ChatThreadModel).where(
                    and_(
                        ChatThreadModel.id == thread_id,
                        ChatThreadModel.workspace_id == workspace_id,
                        func.coalesce(ChatThreadModel.user_id, "") == (user_id or ""),
                    )
                )
            )
            model = result.scalar_one_or_none()
            if not model:
                return None
            model.hidden_at = datetime.now(timezone.utc)
            await db_session.commit()
            await db_session.refresh(model)
            return self._thread_model_to_domain(model)

    async def unhide_thread(
        self, workspace_id: str, thread_id: str, user_id: Optional[str] = None
    ) -> Optional[ChatThread]:
        """Restore (un-archive) a thread; clears hidden_at. Owner-scoped."""
        async with self._session_factory() as db_session:
            result = await db_session.execute(
                select(ChatThreadModel).where(
                    and_(
                        ChatThreadModel.id == thread_id,
                        ChatThreadModel.workspace_id == workspace_id,
                        func.coalesce(ChatThreadModel.user_id, "") == (user_id or ""),
                    )
                )
            )
            model = result.scalar_one_or_none()
            if not model:
                return None
            model.hidden_at = None
            await db_session.commit()
            await db_session.refresh(model)
            return self._thread_model_to_domain(model)

    def _thread_model_to_domain(self, model: ChatThreadModel) -> ChatThread:
        """Convert ORM model to domain model."""
        return ChatThread(
            id=model.id,
            workspace_id=model.workspace_id,
            tenant_id=model.tenant_id,
            context_id=model.context_id,
            user_id=model.user_id,
            observer_id=model.observer_id,
            subject_id=model.subject_id,
            title=model.title,
            metadata=model.meta or {},
            message_count=model.message_count or 0,
            last_decomposed_at=model.last_decomposed_at,
            last_decomposed_index=model.last_decomposed_index,
            expires_at=model.expires_at,
            idle_action=model.idle_action,
            hidden_at=model.hidden_at,
            parent_thread=model.parent_thread,
            created_at=model.created_at,
            updated_at=model.updated_at,
            scope=model.scope,
            ownership=model.ownership or 'user',
        )

    def _message_model_to_domain(self, model: ChatMessageModel) -> ChatMessage:
        """Convert ORM model to domain model."""
        return ChatMessage(
            id=model.id,
            thread_id=model.thread_id,
            message_index=model.message_index,
            role=model.role,
            content=model.content,
            metadata=model.meta or {},
            created_at=model.created_at,
        )

    # Helper methods to convert ORM models to domain models
    def _session_model_to_domain(self, model: SessionModel) -> Session:
        """Convert ORM model to domain model."""
        return Session(
            id=model.id,
            workspace_id=model.workspace_id,
            tenant_id=model.tenant_id,
            context_id=model.context_id,
            user_id=model.user_id,
            metadata=model.meta or {},
            auto_commit=model.auto_commit,
            committed_at=model.committed_at,
            expires_at=model.expires_at,
            created_at=model.created_at,
        )

    def _session_context_model_to_domain(self, model: SessionContextModel) -> WorkingMemory:
        """Convert ORM model to domain model."""
        return WorkingMemory(
            session_id=model.session_id,
            key=model.key,
            value=model.value,
            ttl_seconds=model.ttl_seconds,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    @staticmethod
    def _context_event_model_to_domain(model: SessionContextEventModel) -> SessionContextEvent:
        return SessionContextEvent(
            sequence=model.sequence,
            workspace_id=model.workspace_id,
            session_id=model.session_id,
            event_kind=ContextEventKind(model.event_kind),
            subject_kind=model.subject_kind,
            subject_id=model.subject_id,
            event_time=model.event_time,
            metadata=model.meta or {},
        )

    @staticmethod
    def _checkpoint_model_to_domain(model: SessionCheckpointModel) -> SessionCheckpoint:
        return SessionCheckpoint(
            id=model.id,
            workspace_id=model.workspace_id,
            session_id=model.session_id,
            raw_memory_id=model.raw_memory_id,
            source_kind=model.source_kind,
            source_sequence=model.source_sequence,
            source_boundary=model.source_boundary,
            content_hash=model.content_hash,
            byte_count=model.byte_count,
            capture_status=CheckpointCaptureStatus(model.capture_status),
            index_status=CheckpointWorkStatus(model.index_status),
            enrichment_status=CheckpointWorkStatus(model.enrichment_status),
            idempotency_key=model.idempotency_key,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    @staticmethod
    def _entity_relation_model_to_domain(model: EntityRelationModel) -> EntityRelation:
        return EntityRelation(
            id=model.id,
            workspace_id=model.workspace_id,
            source_entity_id=model.source_entity_id,
            target_entity_id=model.target_entity_id,
            relationship=model.relationship,
            direction=model.direction,
            confidence=model.confidence,
            active=model.active,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    def _memory_model_to_domain(self, model: MemoryModel) -> Memory:
        """Convert ORM model to domain model."""
        # Convert multivector from pgvector array to list of lists
        multivector = None
        if model.multivector is not None:
            multivector = [_vec_to_list(v) for v in model.multivector]

        memory = Memory(
            id=model.id,
            logical_key=getattr(model, "logical_key", None),
            workspace_id=model.workspace_id,
            tenant_id=model.tenant_id,
            context_id=model.context_id,
            user_id=model.user_id,
            content=model.content,
            content_hash=model.content_hash,
            type=MemoryType(model.type),
            subtype=model.subtype if model.subtype else None,
            importance=model.importance,
            tags=model.tags or [],
            metadata=model.meta or {},
            refinement_metadata=getattr(model, "refinement_meta", None) or {},
            abstract=model.abstract,
            overview=model.overview,
            session_id=model.session_id,
            source_memory_id=model.source_memory_id,
            category=model.category,
            status=model.status,
            pinned=model.pinned,
            observer_id=model.observer_id,
            subject_id=model.subject_id,
            source_document_id=getattr(model, 'source_document_id', None),
            source_page_id=getattr(model, 'source_page_id', None),
            source_dataset_id=getattr(model, 'source_dataset_id', None),
            source_thread_id=getattr(model, 'source_thread_id', None),
            embedding=_loaded_embedding(model),
            multivector=multivector,
            access_count=model.access_count,
            last_accessed_at=model.last_accessed_at,
            decay_factor=model.decay_factor,
            event_time=model.event_time,
            revision=getattr(model, "revision", 0),
            etag=getattr(model, "etag", ""),
            created_at=model.created_at,
            updated_at=model.updated_at,
            deleted_at=model.deleted_at,
        )
        if (model.meta or {}).get("source") == "session_checkpoint":
            memory = memory.model_copy(update={"content": model.content})
        return memory

    @staticmethod
    def _memory_domain_to_model(memory: Memory) -> MemoryModel:
        context_id = memory.context_id
        if context_id == "_default":
            context_id = None
        return MemoryModel(
            id=memory.id,
            logical_key=memory.logical_key,
            workspace_id=memory.workspace_id,
            tenant_id=memory.tenant_id,
            context_id=context_id,
            user_id=memory.user_id,
            content=memory.content,
            content_hash=memory.content_hash,
            type=memory.type.value,
            subtype=memory.subtype,
            importance=memory.importance,
            tags=memory.tags,
            meta=memory.metadata,
            refinement_meta=memory.refinement_metadata,
            abstract=memory.abstract,
            overview=memory.overview,
            session_id=memory.session_id,
            source_memory_id=memory.source_memory_id,
            category=memory.category,
            status=memory.status.value,
            pinned=memory.pinned,
            observer_id=memory.observer_id,
            subject_id=memory.subject_id,
            source_document_id=memory.source_document_id,
            source_page_id=memory.source_page_id,
            source_dataset_id=memory.source_dataset_id,
            source_thread_id=memory.source_thread_id,
            embedding=memory.embedding,
            multivector=None,
            access_count=memory.access_count,
            last_accessed_at=memory.last_accessed_at,
            decay_factor=memory.decay_factor,
            deleted_at=memory.deleted_at,
            revision=memory.revision,
            etag=memory.etag,
            event_time=memory.event_time,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
        )

    @staticmethod
    def _apply_memory_domain(model: MemoryModel, memory: Memory) -> None:
        model.content = memory.content
        model.content_hash = memory.content_hash
        model.type = memory.type.value
        model.subtype = memory.subtype
        model.tags = memory.tags
        model.refinement_meta = memory.refinement_metadata
        model.pinned = memory.pinned
        model.embedding = memory.embedding
        model.revision = memory.revision
        model.etag = memory.etag
        model.updated_at = memory.updated_at
        model.deleted_at = memory.deleted_at

    @staticmethod
    def _memory_revision_to_domain(model: MemoryRevisionModel) -> MemoryRevision:
        return MemoryRevision(
            memory=Memory.model_validate(model.snapshot),
            sequence=model.sequence,
            action=model.action,
            operation_id=model.operation_id,
            request_hash=model.request_hash,
        )

    def _association_model_to_domain(self, model: MemoryAssociationModel) -> Association:
        """Convert ORM model to domain model."""
        return Association(
            id=model.id,
            workspace_id=model.workspace_id,
            source_id=model.source_id,
            target_id=model.target_id,
            relationship=model.relation_type,
            strength=model.strength,
            metadata=model.meta or {},
            created_at=model.created_at,
        )

    def _workspace_model_to_domain(self, model: WorkspaceModel) -> Workspace:
        """Convert ORM model to domain model."""
        return Workspace(
            id=model.id,
            tenant_id=model.tenant_id,
            name=model.name,
            settings=model.settings or {},
            tags=list(model.tags or []),
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    # === Document Operations ===

    async def create_document(self, document: Document) -> Document:
        """Create a new document record.

        Args:
            document: Document domain model with all fields populated.

        Returns:
            The created document domain model with server-generated timestamps.
        """
        doc_model = DocumentModel(
            id=document.id,
            workspace_id=document.workspace_id,
            tenant_id=document.tenant_id,
            filename=document.filename,
            document_type=document.document_type.value,
            content_hash=document.content_hash,
            source_vfs_ref=document.source_vfs_ref,
            size_bytes=document.size_bytes,
            mime_type=document.mime_type,
            status=document.status.value,
            target_context_id=document.target_context_id,
            extraction_options=document.extraction_options.model_dump(),
            page_count=document.page_count,
            chunk_count=document.chunk_count,
            memory_ids=document.memory_ids,
            enrichment_status=document.enrichment_status.value,
            enrichment_memory_ids=document.enrichment_memory_ids,
            deduplicated_count=document.deduplicated_count,
            storage_path=document.storage_path,
            retain_original=document.retain_original,
            meta=document.metadata,
            extracted_metadata=document.extracted_metadata,
            created_at=document.created_at,
            processing_started_at=document.processing_started_at,
            processing_completed_at=document.processing_completed_at,
        )

        async with self._session_factory() as session:
            session.add(doc_model)
            await session.commit()
            await session.refresh(doc_model)

        return self._document_model_to_domain(doc_model)

    async def get_document(self, document_id: str, workspace_id: str = None) -> Document | None:
        """Get document by ID, optionally scoped to a workspace.

        Args:
            document_id: Document identifier.
            workspace_id: Optional workspace scope for multi-tenant safety.

        Returns:
            Document domain model, or None if not found.
        """
        async with self._session_factory() as session:
            conditions = [DocumentModel.id == document_id]
            if workspace_id is not None:
                conditions.append(DocumentModel.workspace_id == workspace_id)

            result = await session.execute(
                select(DocumentModel).where(and_(*conditions))
            )
            doc_model = result.scalar_one_or_none()

            if not doc_model:
                return None

            return self._document_model_to_domain(doc_model)

    async def get_documents(
        self, document_ids: list[str], workspace_id: str = None,
    ) -> list[Document]:
        """Get several documents by ID in one query.

        :meth:`get_document` for a set. Ids that do not exist (or fall outside
        ``workspace_id``) are simply absent from the result — callers that need
        to distinguish "missing" must compare against what they asked for.

        Args:
            document_ids: Document identifiers. Empty returns ``[]``.
            workspace_id: Optional workspace scope for multi-tenant safety.

        Returns:
            Document domain models, in the order the database returned them.
        """
        if not document_ids:
            return []

        async with self._session_factory() as session:
            conditions = [DocumentModel.id.in_(document_ids)]
            if workspace_id is not None:
                conditions.append(DocumentModel.workspace_id == workspace_id)

            result = await session.execute(
                select(DocumentModel).where(and_(*conditions))
            )
            return [
                self._document_model_to_domain(m) for m in result.scalars().all()
            ]

    async def list_documents(
        self,
        workspace_id: str,
        status: str = None,
        document_type: str = None,
        created_after: str = None,
        created_before: str = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Document], int]:
        """List documents with optional filters.

        Args:
            workspace_id: Workspace identifier.
            status: Optional status filter (e.g., 'pending', 'completed').
            document_type: Optional document type filter (e.g., 'pdf').
            created_after: Optional ISO datetime lower bound for created_at.
            created_before: Optional ISO datetime upper bound for created_at.
            limit: Maximum number of results.
            offset: Pagination offset.

        Returns:
            Tuple of (documents list, total count matching the filter).
        """
        async with self._session_factory() as session:
            conditions = [DocumentModel.workspace_id == workspace_id]
            if status is not None:
                conditions.append(DocumentModel.status == status)
            if document_type is not None:
                conditions.append(DocumentModel.document_type == document_type)
            if created_after is not None:
                from datetime import datetime as _dt, timezone as _tz
                dt = _dt.fromisoformat(created_after.replace("Z", "+00:00"))
                conditions.append(DocumentModel.created_at >= dt)
            if created_before is not None:
                from datetime import datetime as _dt, timezone as _tz
                dt = _dt.fromisoformat(created_before.replace("Z", "+00:00"))
                conditions.append(DocumentModel.created_at <= dt)

            # Get total count
            count_query = select(func.count()).select_from(DocumentModel).where(
                and_(*conditions)
            )
            count_result = await session.execute(count_query)
            total_count = count_result.scalar() or 0

            # Get paginated results
            query = (
                select(DocumentModel)
                .where(and_(*conditions))
                .order_by(DocumentModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await session.execute(query)
            doc_models = result.scalars().all()

            documents = [self._document_model_to_domain(m) for m in doc_models]
            return documents, total_count

    async def update_document(self, document_id: str, **updates) -> Document:
        """Update document fields by ID.

        Accepts arbitrary keyword arguments matching Document model fields.
        Enum values are automatically converted to their string representation.

        Args:
            document_id: Document identifier.
            **updates: Field names and new values to set.

        Returns:
            Updated document domain model.

        Raises:
            ValueError: If the document is not found.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(DocumentModel).where(DocumentModel.id == document_id)
            )
            doc_model = result.scalar_one_or_none()

            if not doc_model:
                raise ValueError("Document %s not found" % document_id)

            for key, value in updates.items():
                if hasattr(doc_model, key):
                    # Convert enum values to strings for ORM storage
                    if isinstance(value, (DocumentStatus, DocumentEnrichmentStatus)):
                        value = value.value
                    elif isinstance(value, DocumentExtractionOptions):
                        value = value.model_dump()
                    setattr(doc_model, key, value)
                elif key == 'metadata' and hasattr(doc_model, 'meta'):
                    doc_model.meta = value
                else:
                    self.logger.warning(
                        "Skipping unknown document field: %s", key
                    )

            await session.commit()
            await session.refresh(doc_model)

            return self._document_model_to_domain(doc_model)

    async def try_claim_document(
        self, document_id: str, workspace_id: str, ttl_seconds: int,
    ) -> bool:
        """Atomically claim a document for (re)processing via a conditional UPDATE.

        A single conditional ``UPDATE documents SET status='processing',
        processing_started_at=now() WHERE id=:id AND workspace_id=:ws AND (status
        != 'processing' OR processing_started_at IS NULL OR processing_started_at
        < now() - ttl)``. Returns ``True`` iff exactly one row matched (the caller
        won the claim); ``False`` means the doc is owned by a fresh in-flight
        worker — a NO-OP for the loser. Closes the read-then-write race the
        freshness check alone leaves open (two workers both passing the read).

        Args:
            document_id: Document identifier.
            workspace_id: Workspace identifier (scopes the claim).
            ttl_seconds: In-flight freshness window; a PROCESSING doc older than
                this is treated as orphaned and re-claimable.

        Returns:
            True if this caller claimed the document, False otherwise.
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=ttl_seconds)
        processing = DocumentStatus.PROCESSING.value
        async with self._session_factory() as session:
            result = await session.execute(
                update(DocumentModel)
                .where(
                    and_(
                        DocumentModel.id == document_id,
                        DocumentModel.workspace_id == workspace_id,
                        or_(
                            DocumentModel.status != processing,
                            DocumentModel.processing_started_at.is_(None),
                            DocumentModel.processing_started_at < cutoff,
                        ),
                    )
                )
                .values(status=processing, processing_started_at=now)
            )
            await session.commit()
            return result.rowcount == 1

    async def delete_document(self, document_id: str) -> None:
        """Delete a document record by ID.

        This is a hard delete. Blob storage cleanup should be handled
        separately by the ingestion service.

        Args:
            document_id: Document identifier.
        """
        async with self._session_factory() as session:
            await session.execute(
                delete(DocumentModel).where(DocumentModel.id == document_id)
            )
            await session.commit()

    async def find_document_by_hash(
        self, workspace_id: str, content_hash: str
    ) -> Document | None:
        """Find a document by content hash within a workspace.

        Used for deduplication to avoid re-ingesting identical files.

        Args:
            workspace_id: Workspace identifier.
            content_hash: SHA-256 hash of the document content.

        Returns:
            Existing document with the same hash, or None. When multiple rows
            share the hash (the pre-``UNIQUE(workspace_id, content_hash)``
            window — the constraint is a later phase), the oldest is returned
            deterministically rather than raising.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(DocumentModel).where(
                    and_(
                        DocumentModel.workspace_id == workspace_id,
                        DocumentModel.content_hash == content_hash,
                    )
                ).order_by(DocumentModel.created_at.asc())
            )
            # .first() (not scalar_one_or_none) so a pre-unique-constraint
            # duplicate hash does not raise MultipleResultsFound. The
            # uq_documents_workspace_content_hash constraint (migration 024)
            # now backs this query so at most one row can match post-migration;
            # .first() is kept (harmless either way) because the constraint is
            # not guaranteed applied in every environment yet.
            doc_model = result.scalars().first()

            if not doc_model:
                return None

            return self._document_model_to_domain(doc_model)

    async def find_document_by_vfs_ref(
        self, vfs_ref: str
    ) -> Document | None:
        """Find a document by its source VFS reference.

        Used for dedup-by-vfs-ref lookups in the data-connectors ingestion path.

        Args:
            vfs_ref: VFS reference from data-connectors.

        Returns:
            Existing document linked to this VFS ref, or None.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(DocumentModel).where(
                    DocumentModel.source_vfs_ref == vfs_ref,
                ).order_by(DocumentModel.created_at.asc())
            )
            # .first() (not scalar_one_or_none) because idx_documents_source_vfs_ref
            # is NON-unique (migration 016, unique=False): a duplicate vfs_ref would
            # otherwise raise MultipleResultsFound and permanently wedge doc_added.
            # Returning the oldest match deterministically keeps dedup stable.
            doc_model = result.scalars().first()

            if not doc_model:
                return None

            return self._document_model_to_domain(doc_model)

    async def get_document_memories(
        self, workspace_id: str, doc_id: str,
    ) -> list[Memory]:
        """Get all (non-deleted) memories created from a document.

        Mirrors the abstract base / sqlite signature (absent in PG until now).
        Uses the partial index on ``source_document_id``
        (``idx_memories_source_document``).

        Args:
            workspace_id: Workspace identifier.
            doc_id: Source document identifier.

        Returns:
            List of Memory domain models, oldest first.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(MemoryModel).options(_defer_embedding())
                .where(
                    and_(
                        MemoryModel.workspace_id == workspace_id,
                        MemoryModel.source_document_id == doc_id,
                        MemoryModel.deleted_at.is_(None),
                    )
                )
                .order_by(MemoryModel.created_at.asc())
            )
            return [self._memory_model_to_domain(m) for m in result.scalars().all()]

    async def get_memory_source_page_ids(
        self, workspace_id: str, document_id: str,
    ) -> set[str]:
        """Return the set of ``source_page_id``s already represented by a memory.

        The store-phase idempotency primitive: a transcribed page whose id is
        NOT in this set is missing its composite memory. Uses the partial index
        on ``source_page_id`` (``idx_memories_source_page``).

        Args:
            workspace_id: Workspace identifier.
            document_id: Source document identifier.

        Returns:
            Set of non-null ``source_page_id`` values for live memories of the
            document.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(MemoryModel.source_page_id).where(
                    and_(
                        MemoryModel.workspace_id == workspace_id,
                        MemoryModel.source_document_id == document_id,
                        MemoryModel.source_page_id.is_not(None),
                        MemoryModel.deleted_at.is_(None),
                    )
                )
            )
            return {pid for (pid,) in result.all() if pid is not None}

    async def get_fact_memory_parent_ids(
        self, workspace_id: str, parent_memory_ids: set[str],
    ) -> set[str]:
        """Return which of ``parent_memory_ids`` already have ≥1 derived fact.

        The fact-gap primitive (doc_verify, Phase 3): a decomposable page's
        composite memory whose id is NOT in this set never had its
        ``decompose_facts`` produce any atomic fact (e.g. the task was scheduled
        but the worker crashed). ``decompose_facts`` creates each fact via
        ``ingest_fact(source_memory_id=<parent>)`` and archives the parent, so a
        live ``subtype='fact'`` memory chaining to a parent (via
        ``source_memory_id``) is proof that decomposition ran for it.

        Facts do not inherit ``source_document_id`` (``ingest_fact`` only stamps
        ``source_memory_id``), so the lookup keys off the composite memory ids
        the caller already collected — not the document id.

        Args:
            workspace_id: Workspace identifier.
            parent_memory_ids: Candidate composite memory ids (the document's
                page memories) to test for derived facts.

        Returns:
            The subset of ``parent_memory_ids`` that have ≥1 live fact memory.
        """
        if not parent_memory_ids:
            return set()
        async with self._session_factory() as session:
            result = await session.execute(
                select(MemoryModel.source_memory_id)
                .where(
                    and_(
                        MemoryModel.workspace_id == workspace_id,
                        MemoryModel.subtype == "fact",
                        MemoryModel.source_memory_id.in_(parent_memory_ids),
                        MemoryModel.deleted_at.is_(None),
                    )
                )
                .distinct()
            )
            return {pid for (pid,) in result.all() if pid is not None}

    def _document_model_to_domain(self, model: DocumentModel) -> Document:
        """Convert a DocumentModel ORM instance to a Document domain model.

        Args:
            model: SQLAlchemy DocumentModel instance.

        Returns:
            Document domain model.
        """
        return Document(
            id=model.id,
            workspace_id=model.workspace_id,
            tenant_id=model.tenant_id,
            filename=model.filename,
            document_type=model.document_type,
            content_hash=model.content_hash,
            source_vfs_ref=model.source_vfs_ref,
            size_bytes=model.size_bytes,
            mime_type=model.mime_type,
            status=model.status,
            target_context_id=model.target_context_id,
            extraction_options=DocumentExtractionOptions(**(model.extraction_options or {})),
            page_count=model.page_count,
            chunk_count=model.chunk_count,
            memory_ids=model.memory_ids or [],
            enrichment_status=model.enrichment_status or DocumentEnrichmentStatus.NOT_APPLICABLE.value,
            enrichment_memory_ids=model.enrichment_memory_ids or [],
            deduplicated_count=model.deduplicated_count,
            storage_path=model.storage_path,
            retain_original=model.retain_original,
            metadata=model.meta or {},
            extracted_metadata=model.extracted_metadata or {},
            created_at=model.created_at,
            processing_started_at=model.processing_started_at,
            processing_completed_at=model.processing_completed_at,
        )

    def _page_model_to_domain(self, model: DocumentPageModel) -> DocumentPage:
        """Convert a DocumentPageModel ORM instance to a DocumentPage domain model."""
        multivector = None
        if model.multivector is not None:
            multivector = [_vec_to_list(v) for v in model.multivector]

        return DocumentPage(
            id=model.id,
            document_id=model.document_id,
            workspace_id=model.workspace_id,
            page_no=model.page_no,
            image_storage_path=model.image_storage_path,
            transcript=model.transcript,
            embedding=_loaded_embedding(model),
            multivector=multivector,
            transcript_model=model.transcript_model,
            transcript_attempts=model.transcript_attempts or {},
            visual_tokens=model.visual_tokens,
            metadata=model.meta or {},
            created_at=model.created_at,
        )

    # === Document Page Operations ===

    async def create_page(
        self, workspace_id: str, document_id: str, page: DocumentPage,
    ) -> DocumentPage:
        """Persist a document page.

        Args:
            workspace_id: Workspace identifier.
            document_id: Parent document identifier.
            page: DocumentPage with data to persist.

        Returns:
            Persisted DocumentPage with generated id and timestamps.
        """
        page_id = page.id or ("page_%s" % uuid.uuid4().hex[:12])

        # multivector cannot be written through the ORM (pgvector's ARRAY(Vector)
        # bind processor raises "expected ndim to be 1"); insert the row without
        # it, then write it via the explicit text[] -> vector(dim)[] cast — same
        # codec as update_memory / update_page.
        multivector = page.multivector

        page_model = DocumentPageModel(
            id=page_id,
            document_id=document_id,
            workspace_id=workspace_id,
            page_no=page.page_no,
            image_storage_path=page.image_storage_path,
            transcript=page.transcript,
            # Single vector(N) binds normally through the ORM (unlike multivector,
            # below, which needs the text[] -> vector(dim)[] cast).
            embedding=page.embedding,
            multivector=None,
            transcript_model=page.transcript_model,
            transcript_attempts=page.transcript_attempts or {},
            visual_tokens=page.visual_tokens,
            meta=page.metadata if hasattr(page, 'metadata') else {},
        )

        async with self._session_factory() as session:
            session.add(page_model)
            if multivector:
                dim = len(multivector[0])
                await session.flush()
                await session.execute(
                    text(
                        "UPDATE document_pages SET multivector = "
                        "CAST(:mv AS vector(%d)[]) WHERE id = :pid" % dim
                    ),
                    {"mv": _multivector_text_literals(multivector), "pid": page_id},
                )
            await session.commit()
            await session.refresh(page_model)

        return self._page_model_to_domain(page_model)

    async def get_pages(
        self, document_id: str, workspace_id: str = None,
    ) -> list[DocumentPage]:
        """Get all pages for a document, ordered by page number.

        Args:
            document_id: Document identifier.
            workspace_id: Optional workspace scope.

        Returns:
            List of DocumentPage domain models.
        """
        async with self._session_factory() as session:
            conditions = [DocumentPageModel.document_id == document_id]
            if workspace_id is not None:
                conditions.append(DocumentPageModel.workspace_id == workspace_id)

            result = await session.execute(
                select(DocumentPageModel)
                .where(and_(*conditions))
                .order_by(DocumentPageModel.page_no)
            )
            page_models = result.scalars().all()
            return [self._page_model_to_domain(m) for m in page_models]

    async def get_pages_for_documents(
        self,
        document_ids: list[str],
        workspace_id: str = None,
        *,
        limit: int = None,
        offset: int = 0,
    ) -> list[DocumentPage]:
        """Get pages for SEVERAL documents in one query.

        The per-document :meth:`get_pages` forces any caller working over a set
        of documents — assembling a transcript across a bid's proposal files,
        say — to fan out one request per document. This collapses that to a
        single round trip.

        Ordered by ``(document_id, page_no)`` so each document's pages stay
        grouped and in reading order, which is what transcript assembly relies
        on for deterministic output.

        Args:
            document_ids: Document identifiers. Empty returns ``[]`` without a
                query.
            workspace_id: Optional workspace scope.
            limit: Optional page cap, for chunking very large document sets.
            offset: Offset, for use with ``limit``.

        Returns:
            List of DocumentPage domain models.
        """
        if not document_ids:
            return []

        async with self._session_factory() as session:
            conditions = [DocumentPageModel.document_id.in_(document_ids)]
            if workspace_id is not None:
                conditions.append(DocumentPageModel.workspace_id == workspace_id)

            stmt = (
                select(DocumentPageModel)
                .where(and_(*conditions))
                .order_by(DocumentPageModel.document_id, DocumentPageModel.page_no)
            )
            if limit is not None:
                stmt = stmt.limit(limit).offset(offset)

            result = await session.execute(stmt)
            page_models = result.scalars().all()
            return [self._page_model_to_domain(m) for m in page_models]

    async def get_memory_source_page_ids_for_documents(
        self, workspace_id: str, document_ids: list[str],
    ) -> set[str]:
        """:meth:`get_memory_source_page_ids` over several documents at once.

        Lets batch gap analysis resolve the store phase for a whole document
        set in one query rather than one per document.
        """
        if not document_ids:
            return set()

        async with self._session_factory() as session:
            result = await session.execute(
                select(MemoryModel.source_page_id).where(
                    and_(
                        MemoryModel.workspace_id == workspace_id,
                        MemoryModel.source_document_id.in_(document_ids),
                        MemoryModel.source_page_id.is_not(None),
                        MemoryModel.deleted_at.is_(None),
                    )
                )
            )
            return {pid for (pid,) in result.all() if pid is not None}

    async def delete_pages(
        self, document_id: str, workspace_id: str = None,
    ) -> int:
        """Delete all pages for a document.

        Used when reprocessing a document from the render phase, where the
        prior page rows must be cleared before re-rendering to avoid violating
        the ``uq_document_page (document_id, page_no)`` unique constraint.

        Args:
            document_id: Document identifier.
            workspace_id: Optional workspace scope.

        Returns:
            Number of page rows deleted.
        """
        async with self._session_factory() as session:
            conditions = [DocumentPageModel.document_id == document_id]
            if workspace_id is not None:
                conditions.append(DocumentPageModel.workspace_id == workspace_id)

            result = await session.execute(
                delete(DocumentPageModel).where(and_(*conditions))
            )
            await session.commit()
            return result.rowcount

    async def get_pages_by_ids(self, page_ids: list[str]) -> list[DocumentPage]:
        """Get pages by id, returned in the same order as ``page_ids``.

        Used by the grounded document-chat endpoint's explicit-``pages`` context
        item. Each returned page carries ``workspace_id`` so the caller can
        authorize per-workspace; missing ids are simply absent from the result.
        """
        if not page_ids:
            return []
        async with self._session_factory() as session:
            result = await session.execute(
                select(DocumentPageModel).where(DocumentPageModel.id.in_(page_ids))
            )
            by_id = {m.id: self._page_model_to_domain(m) for m in result.scalars().all()}
        # Preserve the caller's requested order (and dedup) for stable, cacheable prompts.
        seen: set[str] = set()
        ordered: list[DocumentPage] = []
        for pid in page_ids:
            if pid in by_id and pid not in seen:
                seen.add(pid)
                ordered.append(by_id[pid])
        return ordered

    async def get_page(self, page_id: str) -> DocumentPage | None:
        """Get a single page by ID.

        Args:
            page_id: Page identifier.

        Returns:
            DocumentPage or None if not found.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(DocumentPageModel).where(DocumentPageModel.id == page_id)
            )
            page_model = result.scalar_one_or_none()
            if not page_model:
                return None
            return self._page_model_to_domain(page_model)

    async def get_page_image_b64(self, page_id: str) -> str | None:
        """Return a page's rendered image as base64, or None if unavailable.

        Duck-typed hook consumed by the OSS ``decompose_facts`` handler to feed
        the page image into the vision model for OCR-free (transcript-less)
        document-page memories. Resolves the page's ``image_storage_path``
        through the enterprise blob-storage service so it works across the local
        filesystem and blobgw/S3 backends alike. Best-effort: returns None (never
        raises) when the page, its image, or the blob service is absent, so the
        handler falls back to text decomposition.
        """
        try:
            page = await self.get_page(page_id)
            if not page or not page.image_storage_path:
                return None
            # Delayed import: the blob-storage ext constant lives in the document
            # services package; importing it lazily avoids a storage<->document
            # import cycle at module load.
            from memorylayer_saas.services.document import EXT_BLOB_STORAGE_SERVICE

            blob_storage = get_extension(EXT_BLOB_STORAGE_SERVICE, self._v)
            if blob_storage is None:
                return None
            data = await blob_storage.retrieve_file(page.image_storage_path)
            import base64

            image_b64 = base64.b64encode(data).decode("ascii")
            # Shape the copy sent to the multimodal extraction LLM: downscale +
            # JPEG-encode (the stored render stays full-res). Keeps the request body
            # under the AI-gateway ingress client_max_body_size (the nginx 413) and
            # cuts multimodal latency/cost. Config-knobbed; 0 = no downscale.
            from memorylayer_saas.services.document.image_util import to_llm_image
            from memorylayer_saas import config

            max_dim = int(self._v.environ(
                config.MEMORYLAYER_LLM_IMAGE_MAX_DIM,
                default=str(config.DEFAULT_MEMORYLAYER_LLM_IMAGE_MAX_DIM),
            ))
            quality = int(self._v.environ(
                config.MEMORYLAYER_LLM_IMAGE_JPEG_QUALITY,
                default=str(config.DEFAULT_MEMORYLAYER_LLM_IMAGE_JPEG_QUALITY),
            ))
            return to_llm_image(image_b64, max_dim=max_dim, quality=quality)
        except Exception as exc:  # noqa: BLE001 - best-effort image resolution
            self.logger.warning(
                "get_page_image_b64 failed for page %s: %s", page_id, exc,
            )
            return None

    async def update_page(self, page_id: str, **updates) -> DocumentPage:
        """Update document page fields by ID.

        Accepts arbitrary keyword arguments matching DocumentPage model fields.
        The 'metadata' key is mapped to the ORM 'meta' column.

        Args:
            page_id: Page identifier.
            **updates: Field names and new values to set.

        Returns:
            Updated DocumentPage domain model.

        Raises:
            ValueError: If the page is not found.
        """
        # pgvector 0.4.x's ARRAY(Vector) bind processor is broken for a
        # list-of-vectors (raises "expected ndim to be 1"), so multivector
        # cannot be written through the ORM. Pop it and write it via an
        # explicit text[] -> vector(dim)[] cast below.
        multivector = updates.pop("multivector", _UNSET)

        async with self._session_factory() as session:
            result = await session.execute(
                select(DocumentPageModel).where(DocumentPageModel.id == page_id)
            )
            page_model = result.scalar_one_or_none()

            if not page_model:
                raise ValueError("Page %s not found" % page_id)

            for key, value in updates.items():
                if hasattr(page_model, key):
                    setattr(page_model, key, value)
                elif key == 'metadata' and hasattr(page_model, 'meta'):
                    page_model.meta = value
                else:
                    self.logger.warning("Skipping unknown page field: %s", key)

            if multivector is not _UNSET:
                if multivector is None:
                    page_model.multivector = None
                else:
                    dim = len(multivector[0]) if multivector else 0
                    await session.flush()
                    await session.execute(
                        text(
                            "UPDATE document_pages SET multivector = "
                            "CAST(:mv AS vector(%d)[]) WHERE id = :pid" % dim
                        ),
                        {"mv": _multivector_text_literals(multivector), "pid": page_id},
                    )

            await session.commit()
            await session.refresh(page_model)

            return self._page_model_to_domain(page_model)

    async def search_pages_by_maxsim(
        self,
        workspace_id: str,
        query_multivector: list[list[float]],
        limit: int = 10,
        doc_ids: list[str] | None = None,
    ) -> list[tuple[DocumentPage, float]]:
        """Search document pages using MaxSim (ColBERT-style late interaction).

        Computes the sum of maximum cosine similarities between each query
        vector and all document vectors (late interaction scoring).

        Args:
            workspace_id: Workspace scope.
            query_multivector: Query multi-vector embedding (list of 128-dim vectors).
            limit: Maximum results to return.
            doc_ids: Optional list of document IDs to restrict search.

        Returns:
            List of (DocumentPage, score) tuples ordered by descending score.
        """
        # Format query vectors for SQL (asyncpg can't bind a list-of-vectors param)
        query_vectors_sql = _multivector_sql_array(query_multivector)

        # Build WHERE clause
        where_parts = [
            "workspace_id = :workspace_id",
            "multivector IS NOT NULL",
        ]
        params: dict = {"workspace_id": workspace_id, "limit": limit}

        if doc_ids:
            where_parts.append("document_id = ANY(:doc_ids)")
            params["doc_ids"] = doc_ids

        where_clause = " AND ".join(where_parts)

        # MaxSim scoring: for each query vector, find max cosine sim against
        # all document vectors, then sum those maxima.
        # This uses a lateral subquery pattern for correctness.
        sql = text("""
            SELECT dp.*,
                   (SELECT SUM(max_cos)
                    FROM (
                        SELECT MAX(1 - (dv <=> qv)) AS max_cos
                        FROM unnest(dp.multivector) AS dv,
                             unnest(%(query)s) AS qv
                        GROUP BY qv
                    ) sub
                   ) AS score
            FROM document_pages dp
            WHERE %(where)s
            ORDER BY score DESC NULLS LAST
            LIMIT :limit
        """ % {"query": query_vectors_sql, "where": where_clause})

        async with self._session_factory() as session:
            result = await session.execute(sql, params)
            rows = result.fetchall()

            results = []
            for row in rows:
                # Map row back to ORM-like object for converter
                page_model = DocumentPageModel(
                    id=row.id,
                    document_id=row.document_id,
                    workspace_id=row.workspace_id,
                    page_no=row.page_no,
                    image_storage_path=row.image_storage_path,
                    transcript=row.transcript,
                    # The raw text() maxsim query returns vector(128)[] WITHOUT
                    # the pgvector result codec, so row.multivector is a list of
                    # vector *strings* (which _page_model_to_domain would explode
                    # into per-character lists). Search results don't need the
                    # raw vectors, so drop them.
                    multivector=None,
                    transcript_model=row.transcript_model,
                    transcript_attempts=row.transcript_attempts or {},
                    visual_tokens=row.visual_tokens,
                    meta=row.metadata if hasattr(row, 'metadata') else {},
                    created_at=row.created_at,
                )
                page = self._page_model_to_domain(page_model)
                score = float(row.score) if row.score is not None else 0.0
                results.append((page, score))

            return results

    # === Ingestion Job Operations ===

    async def create_job(self, job: IngestionJob) -> IngestionJob:
        """Create a new ingestion job record.

        Args:
            job: IngestionJob domain model with all fields populated.

        Returns:
            The created job domain model with server-generated timestamps.
        """
        job_model = IngestionJobModel(
            id=job.id,
            workspace_id=job.workspace_id,
            document_ids=job.document_ids,
            status=job.status.value,
            progress_percent=job.progress_percent,
            documents_processed=job.documents_processed,
            total_memories_created=job.total_memories_created,
            webhook_url=job.webhook_url,
            errors=job.errors,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
        )

        async with self._session_factory() as session:
            session.add(job_model)
            await session.commit()
            await session.refresh(job_model)

        return self._job_model_to_domain(job_model)

    async def get_job(self, job_id: str) -> IngestionJob | None:
        """Get ingestion job by ID.

        Args:
            job_id: Job identifier.

        Returns:
            IngestionJob domain model, or None if not found.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(IngestionJobModel).where(IngestionJobModel.id == job_id)
            )
            job_model = result.scalar_one_or_none()

            if not job_model:
                return None

            return self._job_model_to_domain(job_model)

    async def update_job(self, job_id: str, **updates) -> IngestionJob:
        """Update ingestion job fields by ID.

        Accepts arbitrary keyword arguments matching IngestionJob model fields.
        Enum values are automatically converted to their string representation.

        Args:
            job_id: Job identifier.
            **updates: Field names and new values to set.

        Returns:
            Updated IngestionJob domain model.

        Raises:
            ValueError: If the job is not found.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(IngestionJobModel).where(IngestionJobModel.id == job_id)
            )
            job_model = result.scalar_one_or_none()

            if not job_model:
                raise ValueError("Ingestion job %s not found" % job_id)

            for key, value in updates.items():
                if hasattr(job_model, key):
                    # Convert enum values to strings for ORM storage
                    if isinstance(value, JobStatus):
                        value = value.value
                    setattr(job_model, key, value)
                else:
                    self.logger.warning(
                        "Skipping unknown job field: %s", key
                    )

            await session.commit()
            await session.refresh(job_model)

            return self._job_model_to_domain(job_model)

    async def list_jobs(
        self,
        workspace_id: str,
        status: str = None,
        limit: int = 50,
    ) -> list[IngestionJob]:
        """List ingestion jobs for a workspace.

        Args:
            workspace_id: Workspace identifier.
            status: Optional status filter (e.g., 'queued', 'running').
            limit: Maximum number of results.

        Returns:
            List of IngestionJob domain models, newest first.
        """
        async with self._session_factory() as session:
            conditions = [IngestionJobModel.workspace_id == workspace_id]
            if status is not None:
                conditions.append(IngestionJobModel.status == status)

            query = (
                select(IngestionJobModel)
                .where(and_(*conditions))
                .order_by(IngestionJobModel.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(query)
            job_models = result.scalars().all()

            return [self._job_model_to_domain(m) for m in job_models]

    def _job_model_to_domain(self, model: IngestionJobModel) -> IngestionJob:
        """Convert an IngestionJobModel ORM instance to an IngestionJob domain model.

        Args:
            model: SQLAlchemy IngestionJobModel instance.

        Returns:
            IngestionJob domain model.
        """
        return IngestionJob(
            id=model.id,
            workspace_id=model.workspace_id,
            document_ids=model.document_ids or [],
            status=model.status,
            progress_percent=model.progress_percent,
            documents_processed=model.documents_processed,
            total_memories_created=model.total_memories_created,
            webhook_url=model.webhook_url,
            errors=model.errors if isinstance(model.errors, list) else [],
            created_at=model.created_at,
            started_at=model.started_at,
            completed_at=model.completed_at,
        )

    async def list_active_jobs_for_documents(
        self,
        document_ids: list[str],
        statuses: tuple[str, ...] = ("queued", "running"),
    ) -> list[IngestionJob]:
        """Return non-terminal jobs whose ``document_ids`` overlap ``document_ids``.

        Used to coalesce jobs on create (supersede overlapping in-flight jobs)
        and to reconcile the losers when a document reaches ``completed``.
        """
        if not statuses:
            return []
        return await self.list_jobs_for_documents(document_ids, statuses=statuses)

    async def list_jobs_for_documents(
        self,
        document_ids: list[str],
        statuses: tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list[IngestionJob]:
        """Return jobs whose ``document_ids`` overlap ``document_ids``, newest first.

        Uses the Postgres array overlap operator (``&&``) against the
        ``ingestion_jobs.document_ids`` array column. ``statuses=None`` means ANY
        status, which is what an operator inspecting a single document wants:
        the failed and superseded attempts are usually the interesting ones.
        """
        if not document_ids:
            return []

        async with self._session_factory() as session:
            conditions = [IngestionJobModel.document_ids.overlap(list(document_ids))]
            if statuses:
                conditions.append(IngestionJobModel.status.in_(list(statuses)))
            query = (
                select(IngestionJobModel)
                .where(and_(*conditions))
                .order_by(IngestionJobModel.created_at.desc())
            )
            if limit is not None:
                query = query.limit(limit)
            result = await session.execute(query)
            return [self._job_model_to_domain(m) for m in result.scalars().all()]

    async def cancel_orphaned_ingestion_jobs(self) -> int:
        """Cancel queued/running jobs whose referenced documents are all completed.

        A single set-based UPDATE: a job is orphaned iff there is no referenced
        document that is missing or not ``completed`` (``IS DISTINCT FROM`` keeps
        a NULL/missing document out of the orphan set, so such jobs are left
        untouched). Returns the number of jobs cancelled.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                text(
                    "UPDATE ingestion_jobs SET status = 'cancelled', completed_at = now() "
                    "WHERE status IN ('running', 'queued') "
                    # A job with no document_ids has undeterminable liveness — never
                    # treat the empty array as "all docs completed"; leave it untouched.
                    "AND cardinality(document_ids) > 0 "
                    "AND NOT EXISTS ("
                    "  SELECT 1 FROM unnest(document_ids) AS d "
                    "  LEFT JOIN documents doc ON doc.id = d "
                    "  WHERE doc.status IS DISTINCT FROM 'completed'"
                    ")"
                )
            )
            await session.commit()
            return result.rowcount or 0


    # ------------------------------------------------------------------ #
    # Dataset CRUD
    # ------------------------------------------------------------------ #

    async def create_dataset(self, dataset: Dataset) -> Dataset:
        """Create a new dataset record."""
        ds_model = DatasetModel(
            id=dataset.id,
            workspace_id=dataset.workspace_id,
            tenant_id=dataset.tenant_id,
            name=dataset.name,
            filename=dataset.filename,
            format=dataset.format.value,
            content_hash=dataset.content_hash,
            size_bytes=dataset.size_bytes,
            status=dataset.status.value,
            storage_path=dataset.storage_path,
            original_storage_path=dataset.original_storage_path,
            target_context_id=dataset.target_context_id,
            profiling_options=dataset.profiling_options.model_dump(),
            row_count=dataset.row_count,
            column_count=dataset.column_count,
            columns=[c.model_dump() for c in dataset.columns],
            memory_ids=dataset.memory_ids,
            profile_summary=dataset.profile_summary,
            meta=dataset.metadata,
            created_at=dataset.created_at,
            profiling_started_at=dataset.profiling_started_at,
            profiling_completed_at=dataset.profiling_completed_at,
        )

        async with self._session_factory() as session:
            session.add(ds_model)
            await session.commit()
            await session.refresh(ds_model)

        return self._dataset_model_to_domain(ds_model)

    async def get_dataset(self, dataset_id: str, workspace_id: str = None) -> Dataset | None:
        """Get dataset by ID, optionally scoped to a workspace."""
        async with self._session_factory() as session:
            conditions = [DatasetModel.id == dataset_id]
            if workspace_id is not None:
                conditions.append(DatasetModel.workspace_id == workspace_id)

            result = await session.execute(
                select(DatasetModel).where(and_(*conditions))
            )
            ds_model = result.scalar_one_or_none()

            if not ds_model:
                return None

            return self._dataset_model_to_domain(ds_model)

    async def find_dataset_by_hash(self, workspace_id: str, content_hash: str) -> Dataset | None:
        """Find a dataset by content hash within a workspace."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(DatasetModel).where(
                    and_(
                        DatasetModel.workspace_id == workspace_id,
                        DatasetModel.content_hash == content_hash,
                    )
                )
            )
            ds_model = result.scalar_one_or_none()

            if not ds_model:
                return None

            return self._dataset_model_to_domain(ds_model)

    async def list_datasets(
        self,
        workspace_id: str,
        status: str = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Dataset], int]:
        """List datasets with optional status filter."""
        async with self._session_factory() as session:
            conditions = [DatasetModel.workspace_id == workspace_id]
            if status is not None:
                conditions.append(DatasetModel.status == status)

            count_query = select(func.count()).select_from(DatasetModel).where(
                and_(*conditions)
            )
            count_result = await session.execute(count_query)
            total_count = count_result.scalar() or 0

            query = (
                select(DatasetModel)
                .where(and_(*conditions))
                .order_by(DatasetModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await session.execute(query)
            ds_models = result.scalars().all()

            datasets = [self._dataset_model_to_domain(m) for m in ds_models]
            return datasets, total_count

    async def update_dataset(self, dataset_id: str, **updates) -> None:
        """Update dataset fields by ID."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(DatasetModel).where(DatasetModel.id == dataset_id)
            )
            ds_model = result.scalar_one_or_none()

            if not ds_model:
                raise ValueError("Dataset %s not found" % dataset_id)

            for key, value in updates.items():
                if hasattr(ds_model, key):
                    if isinstance(value, DatasetStatus):
                        value = value.value
                    elif isinstance(value, DatasetProfilingOptions):
                        value = value.model_dump()
                    setattr(ds_model, key, value)
                elif key == 'metadata' and hasattr(ds_model, 'meta'):
                    ds_model.meta = value
                else:
                    self.logger.warning("Skipping unknown dataset field: %s", key)

            await session.commit()

    async def delete_dataset(self, dataset_id: str) -> None:
        """Delete a dataset record by ID."""
        async with self._session_factory() as session:
            await session.execute(
                delete(DatasetModel).where(DatasetModel.id == dataset_id)
            )
            await session.commit()

    async def create_dataset_job(self, job: DatasetJob) -> DatasetJob:
        """Create a new dataset job record."""
        job_model = DatasetJobModel(
            id=job.id,
            workspace_id=job.workspace_id,
            dataset_ids=job.dataset_ids,
            status=job.status,
            progress_percent=job.progress_percent,
            datasets_processed=job.datasets_processed,
            total_memories_created=job.total_memories_created,
            errors=job.errors,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
        )

        async with self._session_factory() as session:
            session.add(job_model)
            await session.commit()
            await session.refresh(job_model)

        return self._dataset_job_model_to_domain(job_model)

    async def get_dataset_job(self, job_id: str) -> DatasetJob | None:
        """Get dataset job by ID."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(DatasetJobModel).where(DatasetJobModel.id == job_id)
            )
            job_model = result.scalar_one_or_none()

            if not job_model:
                return None

            return self._dataset_job_model_to_domain(job_model)

    async def list_dataset_jobs(
        self,
        workspace_id: str,
        status: str = None,
        limit: int = 50,
    ) -> list[DatasetJob]:
        """List dataset jobs for a workspace."""
        async with self._session_factory() as session:
            conditions = [DatasetJobModel.workspace_id == workspace_id]
            if status is not None:
                conditions.append(DatasetJobModel.status == status)

            query = (
                select(DatasetJobModel)
                .where(and_(*conditions))
                .order_by(DatasetJobModel.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(query)
            job_models = result.scalars().all()

            return [self._dataset_job_model_to_domain(m) for m in job_models]

    async def update_dataset_job(self, job_id: str, **updates) -> None:
        """Update dataset job fields by ID."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(DatasetJobModel).where(DatasetJobModel.id == job_id)
            )
            job_model = result.scalar_one_or_none()

            if not job_model:
                raise ValueError("Dataset job %s not found" % job_id)

            for key, value in updates.items():
                if hasattr(job_model, key):
                    setattr(job_model, key, value)
                else:
                    self.logger.warning("Skipping unknown dataset job field: %s", key)

            await session.commit()

    def _dataset_model_to_domain(self, model: DatasetModel) -> Dataset:
        """Convert a DatasetModel ORM instance to a Dataset domain model."""
        columns_data = model.columns if isinstance(model.columns, list) else []
        columns = [DatasetColumn(**c) for c in columns_data]

        return Dataset(
            id=model.id,
            workspace_id=model.workspace_id,
            tenant_id=model.tenant_id,
            name=model.name,
            filename=model.filename,
            format=model.format,
            content_hash=model.content_hash,
            size_bytes=model.size_bytes,
            status=model.status,
            storage_path=model.storage_path,
            original_storage_path=model.original_storage_path,
            target_context_id=model.target_context_id,
            profiling_options=DatasetProfilingOptions(**(model.profiling_options or {})),
            row_count=model.row_count,
            column_count=model.column_count,
            columns=columns,
            memory_ids=model.memory_ids or [],
            profile_summary=model.profile_summary,
            metadata=model.meta or {},
            created_at=model.created_at,
            profiling_started_at=model.profiling_started_at,
            profiling_completed_at=model.profiling_completed_at,
        )

    def _dataset_job_model_to_domain(self, model: DatasetJobModel) -> DatasetJob:
        """Convert a DatasetJobModel ORM instance to a DatasetJob domain model."""
        return DatasetJob(
            id=model.id,
            workspace_id=model.workspace_id,
            dataset_ids=model.dataset_ids or [],
            status=model.status,
            progress_percent=model.progress_percent,
            datasets_processed=model.datasets_processed,
            total_memories_created=model.total_memories_created,
            errors=model.errors if isinstance(model.errors, list) else [],
            created_at=model.created_at,
            started_at=model.started_at,
            completed_at=model.completed_at,
        )

    # ------------------------------------------------------------------ #
    # Admin cross-workspace queries
    # ------------------------------------------------------------------ #

    async def get_admin_stats(self) -> dict[str, int]:
        """Get aggregate counts across all workspaces."""
        async with self._session_factory() as session:
            workspace_count = (await session.execute(
                select(func.count()).select_from(WorkspaceModel)
            )).scalar() or 0

            memory_count = (await session.execute(
                select(func.count()).select_from(MemoryModel).where(
                    MemoryModel.deleted_at.is_(None)
                )
            )).scalar() or 0

            session_count = (await session.execute(
                select(func.count()).select_from(SessionModel)
            )).scalar() or 0

            document_count = (await session.execute(
                select(func.count()).select_from(DocumentModel)
            )).scalar() or 0

            dataset_count = (await session.execute(
                select(func.count()).select_from(DatasetModel)
            )).scalar() or 0

            token_count = 0
            try:
                from .models import TokenModel
                token_count = (await session.execute(
                    select(func.count()).select_from(TokenModel)
                )).scalar() or 0
            except Exception:
                pass

            return {
                "workspace_count": workspace_count,
                "memory_count": memory_count,
                "session_count": session_count,
                "document_count": document_count,
                "dataset_count": dataset_count,
                "token_count": token_count,
            }

    async def admin_list_memories(
        self,
        workspace_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list, int]:
        """List memories across all workspaces (or filtered by workspace_id)."""
        async with self._session_factory() as session:
            conditions = [MemoryModel.deleted_at.is_(None)]
            if workspace_id is not None:
                conditions.append(MemoryModel.workspace_id == workspace_id)
            if status is not None:
                conditions.append(MemoryModel.status == status)

            total = (await session.execute(
                select(func.count()).select_from(MemoryModel).where(and_(*conditions))
            )).scalar() or 0

            query = (
                select(MemoryModel)
                .where(and_(*conditions))
                .order_by(MemoryModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await session.execute(query)
            rows = result.scalars().all()
            memories = [
                {
                    "id": m.id,
                    "workspace_id": m.workspace_id,
                    "content": m.content[:200] if m.content else "",
                    "type": m.type,
                    "subtype": m.subtype,
                    "importance": m.importance,
                    "status": m.status,
                    "tags": m.tags or [],
                    "access_count": m.access_count,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                    "updated_at": m.updated_at.isoformat() if m.updated_at else None,
                }
                for m in rows
            ]
            return memories, total

    async def admin_list_sessions(
        self,
        workspace_id: str | None = None,
        include_expired: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list, int]:
        """List sessions across all workspaces (or filtered by workspace_id)."""
        now = datetime.now(timezone.utc)
        async with self._session_factory() as session:
            conditions = []
            if workspace_id is not None:
                conditions.append(SessionModel.workspace_id == workspace_id)
            if not include_expired:
                conditions.append(SessionModel.expires_at >= now)

            where_clause = and_(*conditions) if conditions else True

            total = (await session.execute(
                select(func.count()).select_from(SessionModel).where(where_clause)
            )).scalar() or 0

            query = (
                select(SessionModel)
                .where(where_clause)
                .order_by(SessionModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await session.execute(query)
            rows = result.scalars().all()
            sessions = [
                {
                    "id": m.id,
                    "workspace_id": m.workspace_id,
                    "context_id": m.context_id,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                    "expires_at": m.expires_at.isoformat() if m.expires_at else None,
                    "last_accessed_at": m.last_accessed_at.isoformat() if m.last_accessed_at else None,
                }
                for m in rows
            ]
            return sessions, total

    async def admin_list_documents(
        self,
        workspace_id: str | None = None,
        status: str | None = None,
        enrichment_status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list, int]:
        """List documents across all workspaces (or filtered by workspace_id).

        ``status`` and ``enrichment_status`` filter the two INDEPENDENT phases:
        ``status`` is retrieval readiness (pages/embeddings/memories durable),
        ``enrichment_status`` is the knowledge phase (fact decomposition), which
        routinely trails it. Filtering on both is how "ingested but never
        decomposed" is isolated — see ``DocumentEnrichmentStatus``.
        """
        async with self._session_factory() as session:
            conditions = []
            if workspace_id is not None:
                conditions.append(DocumentModel.workspace_id == workspace_id)
            if status is not None:
                conditions.append(DocumentModel.status == status)
            if enrichment_status is not None:
                conditions.append(DocumentModel.enrichment_status == enrichment_status)

            where_clause = and_(*conditions) if conditions else True

            total = (await session.execute(
                select(func.count()).select_from(DocumentModel).where(where_clause)
            )).scalar() or 0

            query = (
                select(DocumentModel)
                .where(where_clause)
                .order_by(DocumentModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await session.execute(query)
            rows = result.scalars().all()
            documents = [
                {
                    "id": m.id,
                    "workspace_id": m.workspace_id,
                    "filename": m.filename,
                    "document_type": m.document_type,
                    "status": m.status,
                    # The knowledge phase, independent of ``status``. Emitted so
                    # a caller can tell "not retrievable yet" from "retrievable
                    # but facts never extracted" without a second query.
                    "enrichment_status": (
                        m.enrichment_status or DocumentEnrichmentStatus.NOT_APPLICABLE.value
                    ),
                    # How many memories were HANDED to decomposition at ingest,
                    # not how many are still outstanding: doc_verify converges
                    # ``enrichment_status`` but deliberately never rewrites this
                    # list, so it stays the scheduled set for the document's
                    # life. Naming it "pending" would report a large count on a
                    # fully COMPLETE document. Count only — the ids run to
                    # thousands and no list view needs them.
                    "enrichment_scheduled_count": len(m.enrichment_memory_ids or []),
                    "size_bytes": m.size_bytes,
                    "page_count": m.page_count,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                    # DocumentModel has no ``updated_at``; these are its only
                    # other timestamps. Already declared by the superadmin
                    # client type but never actually sent by this endpoint —
                    # they make "processing" legible as "stuck since <t>".
                    "processing_started_at": (
                        m.processing_started_at.isoformat() if m.processing_started_at else None
                    ),
                    "processing_completed_at": (
                        m.processing_completed_at.isoformat() if m.processing_completed_at else None
                    ),
                }
                for m in rows
            ]
            return documents, total

    async def admin_list_datasets(
        self,
        workspace_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list, int]:
        """List datasets across all workspaces (or filtered by workspace_id)."""
        async with self._session_factory() as session:
            conditions = []
            if workspace_id is not None:
                conditions.append(DatasetModel.workspace_id == workspace_id)
            if status is not None:
                conditions.append(DatasetModel.status == status)

            where_clause = and_(*conditions) if conditions else True

            total = (await session.execute(
                select(func.count()).select_from(DatasetModel).where(where_clause)
            )).scalar() or 0

            query = (
                select(DatasetModel)
                .where(where_clause)
                .order_by(DatasetModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await session.execute(query)
            rows = result.scalars().all()
            datasets = [
                {
                    "id": m.id,
                    "workspace_id": m.workspace_id,
                    "name": m.name,
                    "filename": m.filename,
                    "format": m.format,
                    "status": m.status,
                    "size_bytes": m.size_bytes,
                    "row_count": m.row_count,
                    "column_count": m.column_count,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                }
                for m in rows
            ]
            return datasets, total

    async def admin_list_jobs(
        self,
        workspace_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list, int]:
        """List all jobs (document + dataset) across workspaces."""
        async with self._session_factory() as session:
            # Document ingestion jobs
            doc_conditions = []
            if workspace_id is not None:
                doc_conditions.append(IngestionJobModel.workspace_id == workspace_id)
            if status is not None:
                doc_conditions.append(IngestionJobModel.status == status)
            doc_where = and_(*doc_conditions) if doc_conditions else True

            doc_total = (await session.execute(
                select(func.count()).select_from(IngestionJobModel).where(doc_where)
            )).scalar() or 0

            doc_query = (
                select(IngestionJobModel)
                .where(doc_where)
                .order_by(IngestionJobModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            doc_result = await session.execute(doc_query)
            doc_rows = doc_result.scalars().all()

            # Dataset jobs
            ds_conditions = []
            if workspace_id is not None:
                ds_conditions.append(DatasetJobModel.workspace_id == workspace_id)
            if status is not None:
                ds_conditions.append(DatasetJobModel.status == status)
            ds_where = and_(*ds_conditions) if ds_conditions else True

            ds_total = (await session.execute(
                select(func.count()).select_from(DatasetJobModel).where(ds_where)
            )).scalar() or 0

            ds_query = (
                select(DatasetJobModel)
                .where(ds_where)
                .order_by(DatasetJobModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            ds_result = await session.execute(ds_query)
            ds_rows = ds_result.scalars().all()

            jobs = []
            for m in doc_rows:
                jobs.append({
                    "id": m.id,
                    "workspace_id": m.workspace_id,
                    "job_type": "document",
                    "status": m.status,
                    "progress_percent": m.progress_percent,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                    "started_at": m.started_at.isoformat() if m.started_at else None,
                    "completed_at": m.completed_at.isoformat() if m.completed_at else None,
                })
            for m in ds_rows:
                jobs.append({
                    "id": m.id,
                    "workspace_id": m.workspace_id,
                    "job_type": "dataset",
                    "status": m.status,
                    "progress_percent": m.progress_percent,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                    "started_at": m.started_at.isoformat() if m.started_at else None,
                    "completed_at": m.completed_at.isoformat() if m.completed_at else None,
                })

            jobs.sort(key=lambda j: j["created_at"] or "", reverse=True)
            return jobs, doc_total + ds_total

    # === Admin catalog listings (chat threads / skills / MCP servers / apps) ===
    #
    # Cross-workspace read-only projections for the platform admin console.
    # Same ``tuple[list[dict], int]`` contract as admin_list_sessions/memories:
    # a trimmed row dict for the table plus a total for server pagination. The
    # ``admin_get_*`` companions return the full row (incl. JSON blobs) for the
    # per-row detail drawer, or ``None`` when the id is unknown.

    async def admin_list_chat_threads(
        self,
        workspace_id: str | None = None,
        ownership: str | None = None,
        include_hidden: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list, int]:
        """List chat threads across all workspaces (admin console)."""
        async with self._session_factory() as session:
            conditions = []
            if workspace_id is not None:
                conditions.append(ChatThreadModel.workspace_id == workspace_id)
            if ownership is not None:
                conditions.append(ChatThreadModel.ownership == ownership)
            if not include_hidden:
                conditions.append(ChatThreadModel.hidden_at.is_(None))
            where_clause = and_(*conditions) if conditions else True

            total = (await session.execute(
                select(func.count()).select_from(ChatThreadModel).where(where_clause)
            )).scalar() or 0

            query = (
                select(ChatThreadModel)
                .where(where_clause)
                .order_by(ChatThreadModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            rows = (await session.execute(query)).scalars().all()
            items = [
                {
                    "id": m.id,
                    "workspace_id": m.workspace_id,
                    "tenant_id": m.tenant_id,
                    "user_id": m.user_id,
                    "title": m.title,
                    "ownership": m.ownership,
                    # NULL scope reads as "web" (see model comment).
                    "scope": m.scope or "web",
                    "message_count": m.message_count,
                    "parent_thread": m.parent_thread,
                    "idle_action": m.idle_action,
                    "hidden_at": m.hidden_at.isoformat() if m.hidden_at else None,
                    "expires_at": m.expires_at.isoformat() if m.expires_at else None,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                    "updated_at": m.updated_at.isoformat() if m.updated_at else None,
                }
                for m in rows
            ]
            return items, total

    async def admin_get_chat_thread(self, thread_id: str) -> dict | None:
        """Full chat-thread row for the admin detail drawer (or None)."""
        async with self._session_factory() as session:
            m = (await session.execute(
                select(ChatThreadModel).where(ChatThreadModel.id == thread_id)
            )).scalar_one_or_none()
            if m is None:
                return None
            return {
                "id": m.id,
                "workspace_id": m.workspace_id,
                "tenant_id": m.tenant_id,
                "context_id": m.context_id,
                "user_id": m.user_id,
                "observer_id": m.observer_id,
                "subject_id": m.subject_id,
                "title": m.title,
                "ownership": m.ownership,
                "scope": m.scope or "web",
                "message_count": m.message_count,
                "parent_thread": m.parent_thread,
                "idle_action": m.idle_action,
                "last_decomposed_index": m.last_decomposed_index,
                "last_decomposed_at": m.last_decomposed_at.isoformat() if m.last_decomposed_at else None,
                "metadata": m.meta or {},
                "hidden_at": m.hidden_at.isoformat() if m.hidden_at else None,
                "expires_at": m.expires_at.isoformat() if m.expires_at else None,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "updated_at": m.updated_at.isoformat() if m.updated_at else None,
            }

    async def admin_list_skills(
        self,
        workspace_id: str | None = None,
        enabled: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list, int]:
        """List skills across all workspaces (admin console)."""
        async with self._session_factory() as session:
            conditions = []
            if workspace_id is not None:
                conditions.append(SkillModel.workspace_id == workspace_id)
            if enabled is not None:
                conditions.append(SkillModel.enabled == enabled)
            where_clause = and_(*conditions) if conditions else True

            total = (await session.execute(
                select(func.count()).select_from(SkillModel).where(where_clause)
            )).scalar() or 0

            query = (
                select(SkillModel)
                .where(where_clause)
                .order_by(SkillModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            rows = (await session.execute(query)).scalars().all()
            items = [
                {
                    "id": m.id,
                    "workspace_id": m.workspace_id,
                    "tenant_id": m.tenant_id,
                    "user_id": m.user_id,
                    "name": m.name,
                    "description": m.description,
                    "version": m.version,
                    "source_mode": m.source_mode,
                    "enabled": m.enabled,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                    "updated_at": m.updated_at.isoformat() if m.updated_at else None,
                }
                for m in rows
            ]
            return items, total

    async def admin_get_skill(self, skill_id: str) -> dict | None:
        """Full skill row for the admin detail drawer (or None)."""
        async with self._session_factory() as session:
            m = (await session.execute(
                select(SkillModel).where(SkillModel.id == skill_id)
            )).scalar_one_or_none()
            if m is None:
                return None
            return {
                "id": m.id,
                "workspace_id": m.workspace_id,
                "tenant_id": m.tenant_id,
                "user_id": m.user_id,
                "name": m.name,
                "description": m.description,
                "version": m.version,
                "license": m.license,
                "compatibility": m.compatibility,
                "allowed_tools": m.allowed_tools,
                "body": m.body,
                "source_mode": m.source_mode,
                "manifest_hash": m.manifest_hash,
                "bundle_hash": m.bundle_hash,
                "metadata": m.meta or {},
                "enabled": m.enabled,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "updated_at": m.updated_at.isoformat() if m.updated_at else None,
            }

    async def admin_list_mcp_servers(
        self,
        workspace_id: str | None = None,
        transport: str | None = None,
        enabled: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list, int]:
        """List MCP servers across all workspaces (admin console)."""
        async with self._session_factory() as session:
            conditions = []
            if workspace_id is not None:
                conditions.append(McpServerModel.workspace_id == workspace_id)
            if transport is not None:
                conditions.append(McpServerModel.transport == transport)
            if enabled is not None:
                conditions.append(McpServerModel.enabled == enabled)
            where_clause = and_(*conditions) if conditions else True

            total = (await session.execute(
                select(func.count()).select_from(McpServerModel).where(where_clause)
            )).scalar() or 0

            query = (
                select(McpServerModel)
                .where(where_clause)
                .order_by(McpServerModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            rows = (await session.execute(query)).scalars().all()
            items = [
                {
                    "id": m.id,
                    "workspace_id": m.workspace_id,
                    "tenant_id": m.tenant_id,
                    "user_id": m.user_id,
                    "name": m.name,
                    "description": m.description,
                    "transport": m.transport,
                    "enabled": m.enabled,
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                    "updated_at": m.updated_at.isoformat() if m.updated_at else None,
                }
                for m in rows
            ]
            return items, total

    async def admin_get_mcp_server(self, mcp_id: str) -> dict | None:
        """Full MCP-server row for the admin detail drawer (or None)."""
        async with self._session_factory() as session:
            m = (await session.execute(
                select(McpServerModel).where(McpServerModel.id == mcp_id)
            )).scalar_one_or_none()
            if m is None:
                return None
            return {
                "id": m.id,
                "workspace_id": m.workspace_id,
                "tenant_id": m.tenant_id,
                "user_id": m.user_id,
                "name": m.name,
                "description": m.description,
                "transport": m.transport,
                "command": m.command,
                "args": list(m.args or []),
                "env": m.env or {},
                "url": m.url,
                "headers": m.headers or {},
                "metadata": m.meta or {},
                "source_mode": m.source_mode,
                "manifest_hash": m.manifest_hash,
                "enabled": m.enabled,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "updated_at": m.updated_at.isoformat() if m.updated_at else None,
            }

    async def admin_list_applications(
        self,
        enabled: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list, int]:
        """List registered applications (tenant-scoped; admin console).

        Applications carry no ``workspace_id`` (they are tenant-level and
        associated to workspaces via a join table), so there is no
        workspace filter here — the per-tenant ML instance already scopes it.
        """
        async with self._session_factory() as session:
            conditions = []
            if enabled is not None:
                conditions.append(ApplicationModel.enabled == enabled)
            where_clause = and_(*conditions) if conditions else True

            total = (await session.execute(
                select(func.count()).select_from(ApplicationModel).where(where_clause)
            )).scalar() or 0

            query = (
                select(ApplicationModel)
                .where(where_clause)
                .order_by(ApplicationModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            rows = (await session.execute(query)).scalars().all()
            items = [
                {
                    "id": m.id,
                    "tenant_id": m.tenant_id,
                    "name": m.name,
                    "description": m.description,
                    "app_type": m.app_type,
                    "enabled": m.enabled,
                    "skill_count": len(m.default_skill_names or []),
                    "mcp_server_count": len(m.default_mcp_server_names or []),
                    "tool_count": len(m.default_tool_names or []),
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                    "updated_at": m.updated_at.isoformat() if m.updated_at else None,
                }
                for m in rows
            ]
            return items, total

    async def admin_get_application(self, app_id: str) -> dict | None:
        """Full application row for the admin detail drawer (or None)."""
        async with self._session_factory() as session:
            m = (await session.execute(
                select(ApplicationModel).where(ApplicationModel.id == app_id)
            )).scalar_one_or_none()
            if m is None:
                return None
            return {
                "id": m.id,
                "tenant_id": m.tenant_id,
                "name": m.name,
                "description": m.description,
                "app_type": m.app_type,
                "enabled": m.enabled,
                "config": m.config or {},
                "metadata": m.meta or {},
                "default_skill_names": list(m.default_skill_names or []),
                "default_mcp_server_names": list(m.default_mcp_server_names or []),
                "default_tool_names": list(m.default_tool_names or []),
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "updated_at": m.updated_at.isoformat() if m.updated_at else None,
            }

    # === Data Provider Operations ===

    def _data_provider_model_to_domain(self, model: "DataProviderModel") -> "DataProvider":
        """Convert ORM model to domain model."""
        # Delayed import: avoid circular dependency at module level
        from memorylayer_server.models.data_provider import DataProvider
        return DataProvider(
            id=model.id,
            tenant_id=model.tenant_id,
            workspace_id=model.workspace_id,
            name=model.name,
            provider_type=model.provider_type,
            description=model.description,
            enabled=model.enabled,
            connection_args=model.connection_args or {},
            schedule=model.schedule,
            last_sync_at=model.last_sync_at,
            metadata=model.meta or {},
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    async def create_data_provider(self, workspace_id: str, provider: "DataProvider") -> "DataProvider":
        """Store a data provider."""
        now = datetime.now(timezone.utc)

        async with self._session_factory() as session:
            model = DataProviderModel(
                id=provider.id,
                tenant_id=getattr(provider, 'tenant_id', ''),
                workspace_id=workspace_id,
                name=provider.name,
                provider_type=provider.provider_type,
                description=provider.description,
                enabled=provider.enabled,
                connection_args=provider.connection_args or {},
                encrypted_args={},
                schedule=provider.schedule,
                last_sync_at=provider.last_sync_at,
                meta=provider.metadata or {},
                created_at=now,
                updated_at=now,
            )
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return self._data_provider_model_to_domain(model)

    async def get_data_provider(self, workspace_id: str, provider_id: str) -> "DataProvider | None":
        """Get data provider by ID."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(DataProviderModel).where(
                    and_(
                        DataProviderModel.id == provider_id,
                        DataProviderModel.workspace_id == workspace_id,
                    )
                )
            )
            model = result.scalar_one_or_none()
            return self._data_provider_model_to_domain(model) if model else None

    async def list_data_providers(
        self, workspace_id: str, limit: int = 50, offset: int = 0,
    ) -> tuple[list["DataProvider"], int]:
        """List data providers for a workspace."""
        async with self._session_factory() as session:
            conditions = [DataProviderModel.workspace_id == workspace_id]

            count_result = await session.execute(
                select(func.count(DataProviderModel.id)).where(and_(*conditions))
            )
            total = count_result.scalar_one()

            result = await session.execute(
                select(DataProviderModel)
                .where(and_(*conditions))
                .order_by(DataProviderModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            providers = [self._data_provider_model_to_domain(m) for m in result.scalars().all()]
            return providers, total

    async def update_data_provider(self, workspace_id: str, provider_id: str, **updates) -> "DataProvider | None":
        """Update data provider fields."""
        if not updates:
            return await self.get_data_provider(workspace_id, provider_id)

        async with self._session_factory() as session:
            result = await session.execute(
                select(DataProviderModel).where(
                    and_(
                        DataProviderModel.id == provider_id,
                        DataProviderModel.workspace_id == workspace_id,
                    )
                )
            )
            model = result.scalar_one_or_none()
            if not model:
                return None

            for key, value in updates.items():
                if key == "metadata":
                    setattr(model, "meta", value)
                elif hasattr(model, key):
                    setattr(model, key, value)
            model.updated_at = datetime.now(timezone.utc)

            await session.commit()
            await session.refresh(model)
            return self._data_provider_model_to_domain(model)

    async def delete_data_provider(self, workspace_id: str, provider_id: str) -> bool:
        """Delete a data provider."""
        async with self._session_factory() as session:
            result = await session.execute(
                delete(DataProviderModel).where(
                    and_(
                        DataProviderModel.id == provider_id,
                        DataProviderModel.workspace_id == workspace_id,
                    )
                )
            )
            await session.commit()
            return result.rowcount > 0

    # === Skill Operations ===

    def _skill_model_to_domain(self, model: "SkillModel") -> "Skill":
        """Convert ORM model to Skill domain model."""
        from memorylayer_server.models.skill import Skill
        return Skill(
            id=model.id,
            tenant_id=model.tenant_id,
            workspace_id=model.workspace_id,
            user_id=model.user_id,
            name=model.name,
            description=model.description,
            version=model.version,
            license=model.license,
            compatibility=model.compatibility,
            allowed_tools=model.allowed_tools,
            body=model.body,
            metadata=model.meta or {},
            source_mode=model.source_mode,
            manifest_hash=model.manifest_hash,
            bundle_hash=model.bundle_hash,
            enabled=model.enabled,
            revision=model.revision,
            etag=model.etag,
            created_at=model.created_at,
            updated_at=model.updated_at,
            deleted_at=model.deleted_at,
        )

    def _skill_file_model_to_domain(self, model: "SkillFileModel") -> "SkillFile":
        """Convert ORM model to SkillFile domain model."""
        from memorylayer_server.models.skill import SkillFile
        return SkillFile(
            id=model.id,
            skill_id=model.skill_id,
            path=model.path,
            kind=model.kind,
            content=model.content,
            content_hash=model.content_hash,
            size_bytes=model.size_bytes,
            mime_type=model.mime_type,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    async def create_skill(self, skill: "Skill") -> "Skill":
        """Store a new skill."""
        result = await self.mutate_skill(
            SkillMutation(
                action="create",
                skill=skill,
                operation_id=generate_id("op"),
                request_hash=canonical_hash({"action": "create", "state": manifest_state(skill)}),
                expected_etag="*",
            )
        )
        return result.skill

    async def get_skill(
        self,
        workspace_id: str,
        skill_id: str,
        *,
        include_deleted: bool = False,
    ) -> "Skill | None":
        """Get skill by ID within a workspace."""
        conditions = [
            SkillModel.id == skill_id,
            SkillModel.workspace_id == workspace_id,
        ]
        if not include_deleted:
            conditions.append(SkillModel.deleted_at.is_(None))
        async with self._session_factory() as session:
            result = await session.execute(
                select(SkillModel).where(and_(*conditions))
            )
            model = result.scalar_one_or_none()
            return self._skill_model_to_domain(model) if model else None

    async def get_skill_by_name(
        self, workspace_id: str, name: str, user_id: str | None = None
    ) -> "Skill | None":
        """Get skill by name within a workspace, optionally scoped to a user."""
        async with self._session_factory() as session:
            if user_id is not None:
                conditions = and_(
                    SkillModel.workspace_id == workspace_id,
                    SkillModel.name == name,
                    SkillModel.user_id == user_id,
                    SkillModel.deleted_at.is_(None),
                )
            else:
                conditions = and_(
                    SkillModel.workspace_id == workspace_id,
                    SkillModel.name == name,
                    SkillModel.user_id.is_(None),
                    SkillModel.deleted_at.is_(None),
                )
            result = await session.execute(select(SkillModel).where(conditions))
            model = result.scalar_one_or_none()
            return self._skill_model_to_domain(model) if model else None

    async def list_skills(
        self,
        workspace_id: str,
        user_id: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        enabled: bool | None = None,
        limit: int = 100,
        offset: int = 0,
        include_global: bool = False,
    ) -> list["Skill"]:
        """List skills for a workspace with optional filters.

        When ``include_global`` is set, tenant-shared ``_global`` skills
        (``user_id IS NULL``) are unioned into the result.
        """
        from memorylayer_server.config import GLOBAL_WORKSPACE_ID

        # Workspace scope: the current workspace, plus optionally the shared
        # _global workspace (global skills are never user-scoped).
        if include_global and user_id is None and workspace_id != GLOBAL_WORKSPACE_ID:
            conditions = [
                or_(
                    SkillModel.workspace_id == workspace_id,
                    and_(
                        SkillModel.workspace_id == GLOBAL_WORKSPACE_ID,
                        SkillModel.user_id.is_(None),
                    ),
                )
            ]
        else:
            conditions = [SkillModel.workspace_id == workspace_id]

        conditions.append(SkillModel.deleted_at.is_(None))

        if user_id is not None:
            conditions.append(SkillModel.user_id == user_id)
        if name is not None:
            conditions.append(SkillModel.name == name)
        if enabled is not None:
            conditions.append(SkillModel.enabled == enabled)
        # tags filter not stored as array on skills; skip silently if provided

        async with self._session_factory() as session:
            result = await session.execute(
                select(SkillModel)
                .where(and_(*conditions))
                .order_by(SkillModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return [self._skill_model_to_domain(m) for m in result.scalars().all()]

    async def find_skills_by_name(self, name: str, scope_filters: list[dict]) -> list["Skill"]:
        """Find skills by name across multiple scope filters for resolution.

        Each filter dict has workspace_id and optional user_id keys.
        Returns all matches; the resolution service handles precedence ordering.
        """
        if not scope_filters:
            return []

        conditions = []
        for sf in scope_filters:
            ws = sf.get("workspace_id")
            uid = sf.get("user_id")
            if uid is not None:
                conditions.append(
                    and_(SkillModel.workspace_id == ws, SkillModel.user_id == uid)
                )
            else:
                conditions.append(
                    and_(SkillModel.workspace_id == ws, SkillModel.user_id.is_(None))
                )

        async with self._session_factory() as session:
            result = await session.execute(
                select(SkillModel)
                .where(and_(SkillModel.name == name, SkillModel.deleted_at.is_(None), or_(*conditions)))
                .order_by(SkillModel.updated_at.desc())
            )
            return [self._skill_model_to_domain(m) for m in result.scalars().all()]

    async def update_skill(self, workspace_id: str, skill_id: str, updates: dict) -> "Skill | None":
        """Update skill fields."""
        if not updates:
            return await self.get_skill(workspace_id, skill_id)
        current = await self.get_skill(workspace_id, skill_id)
        if current is None:
            return None
        desired = current.model_copy(
            update={key: value for key, value in updates.items() if hasattr(current, key)}
        ).model_copy(update={"updated_at": datetime.now(timezone.utc)})
        semantic_updates = set(updates) - {"bundle_hash", "updated_at"}
        if not semantic_updates:
            async with self._session_factory() as session:
                result = await session.execute(
                    select(SkillModel).where(
                        SkillModel.id == skill_id,
                        SkillModel.workspace_id == workspace_id,
                        SkillModel.deleted_at.is_(None),
                    )
                )
                model = result.scalar_one_or_none()
                if model is None:
                    return None
                model.bundle_hash = desired.bundle_hash
                model.updated_at = desired.updated_at
                await session.commit()
                await session.refresh(model)
                return self._skill_model_to_domain(model)
        result = await self.mutate_skill(
            SkillMutation(
                action="replace",
                skill=desired,
                operation_id=generate_id("op"),
                request_hash=canonical_hash(
                    {"action": "replace", "state": manifest_state(desired), "expected_etag": current.etag}
                ),
                expected_etag=current.etag,
            )
        )
        return result.skill

    async def delete_skill(self, workspace_id: str, skill_id: str) -> bool:
        """Write a durable tombstone and retain child files for restoration."""
        current = await self.get_skill(workspace_id, skill_id)
        if current is None:
            return False
        now = datetime.now(timezone.utc)
        desired = current.model_copy(update={"updated_at": now, "deleted_at": now})
        await self.mutate_skill(
            SkillMutation(
                action="delete",
                skill=desired,
                operation_id=generate_id("op"),
                request_hash=canonical_hash(
                    {"action": "delete", "id": skill_id, "expected_etag": current.etag}
                ),
                expected_etag=current.etag,
            )
        )
        self.logger.debug("Deleted skill: %s", skill_id)
        return True

    async def mutate_skill(self, mutation: SkillMutation) -> SkillMutationResult:
        """Atomically mutate a locked native head plus revision/idempotency rows."""
        from sqlalchemy.exc import IntegrityError

        desired = mutation.skill.model_copy(deep=True)
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    replay = await self._get_skill_operation_in_session(
                        session,
                        desired.tenant_id,
                        desired.workspace_id,
                        mutation.operation_id,
                    )
                    if replay is not None:
                        return self._validate_skill_replay(replay, mutation.request_hash)

                    result = await session.execute(
                        select(SkillModel)
                        .where(
                            SkillModel.id == desired.id,
                            SkillModel.workspace_id == desired.workspace_id,
                        )
                        .with_for_update()
                    )
                    current_model = result.scalar_one_or_none()
                    current = (
                        self._skill_model_to_domain(current_model)
                        if current_model is not None
                        else None
                    )
                    if mutation.action == "create":
                        if mutation.expected_etag != "*":
                            raise VersionedResourcePreconditionFailedError(
                                "create requires If-None-Match: *"
                            )
                        if current is not None:
                            raise VersionedResourceConflictError("skill id already exists")
                        final = desired.model_copy(update={"revision": 1, "deleted_at": None})
                        final = final.model_copy(update={"etag": manifest_etag(1, final)})
                        current_model = self._skill_domain_to_model(final)
                        session.add(current_model)
                    else:
                        if current is None or current.tenant_id != desired.tenant_id:
                            raise VersionedResourceNotFoundError("skill not found")
                        if mutation.expected_etag != current.etag:
                            raise VersionedResourcePreconditionFailedError(
                                "ETag does not match current revision"
                            )
                        if mutation.action == "restore":
                            if current.deleted_at is None:
                                raise VersionedResourceConflictError("skill is not deleted")
                        elif current.deleted_at is not None:
                            raise VersionedResourceNotFoundError("skill not found")
                        if desired.name != current.name or desired.user_id != current.user_id:
                            raise VersionedResourceConflictError(
                                "skill name and ownership are immutable"
                            )
                        final = desired.model_copy(
                            update={
                                "created_at": current.created_at,
                                "revision": current.revision + 1,
                            }
                        )
                        final = final.model_copy(
                            update={"etag": manifest_etag(final.revision, final)}
                        )
                        self._apply_skill_domain(current_model, final)

                    revision = SkillRevisionModel(
                        tenant_id=final.tenant_id,
                        workspace_id=final.workspace_id,
                        skill_id=final.id,
                        revision=final.revision,
                        snapshot=final.model_dump(mode="json"),
                        action=mutation.action,
                        operation_id=mutation.operation_id,
                        request_hash=mutation.request_hash,
                    )
                    session.add(revision)
                    session.add(
                        SkillOperationModel(
                            tenant_id=final.tenant_id,
                            workspace_id=final.workspace_id,
                            operation_id=mutation.operation_id,
                            request_hash=mutation.request_hash,
                            skill_id=final.id,
                            revision=final.revision,
                        )
                    )
                    await session.flush()
                    return SkillMutationResult(skill=final)
        except IntegrityError as exc:
            replay = await self.get_skill_operation(
                desired.tenant_id,
                desired.workspace_id,
                mutation.operation_id,
                mutation.request_hash,
            )
            if replay is not None:
                return replay
            if mutation.action == "create":
                raise VersionedResourceConflictError(
                    "skill id or scoped name already exists"
                ) from exc
            raise

    async def get_skill_operation(
        self,
        tenant_id: str,
        workspace_id: str,
        operation_id: str,
        request_hash: str,
    ) -> SkillMutationResult | None:
        async with self._session_factory() as session:
            revision = await self._get_skill_operation_in_session(
                session, tenant_id, workspace_id, operation_id
            )
        if revision is None:
            return None
        return self._validate_skill_replay(revision, request_hash)

    async def _get_skill_operation_in_session(
        self,
        session,
        tenant_id: str,
        workspace_id: str,
        operation_id: str,
    ) -> SkillRevision | None:
        result = await session.execute(
            select(SkillRevisionModel)
            .join(
                SkillOperationModel,
                and_(
                    SkillRevisionModel.tenant_id == SkillOperationModel.tenant_id,
                    SkillRevisionModel.workspace_id == SkillOperationModel.workspace_id,
                    SkillRevisionModel.skill_id == SkillOperationModel.skill_id,
                    SkillRevisionModel.revision == SkillOperationModel.revision,
                ),
            )
            .where(
                SkillOperationModel.tenant_id == tenant_id,
                SkillOperationModel.workspace_id == workspace_id,
                SkillOperationModel.operation_id == operation_id,
            )
        )
        model = result.scalar_one_or_none()
        return self._skill_revision_to_domain(model) if model is not None else None

    @staticmethod
    def _validate_skill_replay(
        revision: SkillRevision,
        request_hash: str,
    ) -> SkillMutationResult:
        if revision.request_hash != request_hash:
            raise VersionedResourceConflictError(
                "idempotency key was already used for a different request"
            )
        return SkillMutationResult(skill=revision.skill, replayed=True)

    async def list_skill_revisions(
        self,
        tenant_id: str,
        workspace_id: str,
        skill_id: str,
        *,
        limit: int,
        before_sequence: int | None = None,
    ) -> list[SkillRevision]:
        conditions = [
            SkillRevisionModel.tenant_id == tenant_id,
            SkillRevisionModel.workspace_id == workspace_id,
            SkillRevisionModel.skill_id == skill_id,
        ]
        if before_sequence is not None:
            conditions.append(SkillRevisionModel.sequence < before_sequence)
        async with self._session_factory() as session:
            result = await session.execute(
                select(SkillRevisionModel)
                .where(and_(*conditions))
                .order_by(SkillRevisionModel.sequence.desc())
                .limit(limit)
            )
            return [
                self._skill_revision_to_domain(model)
                for model in result.scalars().all()
            ]

    @staticmethod
    def _skill_domain_to_model(skill: Skill) -> SkillModel:
        return SkillModel(
            id=skill.id,
            tenant_id=skill.tenant_id,
            workspace_id=skill.workspace_id,
            user_id=skill.user_id,
            name=skill.name,
            description=skill.description,
            version=skill.version,
            license=skill.license,
            compatibility=skill.compatibility,
            allowed_tools=skill.allowed_tools,
            body=skill.body,
            meta=skill.metadata,
            source_mode=skill.source_mode,
            manifest_hash=skill.manifest_hash,
            bundle_hash=skill.bundle_hash,
            enabled=skill.enabled,
            revision=skill.revision,
            etag=skill.etag,
            created_at=skill.created_at,
            updated_at=skill.updated_at,
            deleted_at=skill.deleted_at,
        )

    @staticmethod
    def _apply_skill_domain(model: SkillModel, skill: Skill) -> None:
        model.description = skill.description
        model.version = skill.version
        model.license = skill.license
        model.compatibility = skill.compatibility
        model.allowed_tools = skill.allowed_tools
        model.body = skill.body
        model.meta = skill.metadata
        model.source_mode = skill.source_mode
        model.manifest_hash = skill.manifest_hash
        model.bundle_hash = skill.bundle_hash
        model.enabled = skill.enabled
        model.revision = skill.revision
        model.etag = skill.etag
        model.updated_at = skill.updated_at
        model.deleted_at = skill.deleted_at

    @staticmethod
    def _skill_revision_to_domain(model: SkillRevisionModel) -> SkillRevision:
        return SkillRevision(
            skill=Skill.model_validate(model.snapshot),
            sequence=model.sequence,
            action=model.action,
            operation_id=model.operation_id,
            request_hash=model.request_hash,
        )

    async def upsert_skill_file(self, skill_file: "SkillFile") -> "SkillFile":
        """Insert or update a skill file by (skill_id, path)."""
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        now = datetime.now(timezone.utc)
        async with self._session_factory() as session:
            stmt = (
                pg_insert(SkillFileModel)
                .values(
                    id=skill_file.id,
                    skill_id=skill_file.skill_id,
                    path=skill_file.path,
                    kind=skill_file.kind,
                    content=skill_file.content,
                    content_hash=skill_file.content_hash,
                    size_bytes=skill_file.size_bytes,
                    mime_type=skill_file.mime_type,
                    created_at=skill_file.created_at or now,
                    updated_at=skill_file.updated_at or now,
                )
                .on_conflict_do_update(
                    constraint="uq_skill_file_path",
                    set_=dict(
                        id=skill_file.id,
                        kind=skill_file.kind,
                        content=skill_file.content,
                        content_hash=skill_file.content_hash,
                        size_bytes=skill_file.size_bytes,
                        mime_type=skill_file.mime_type,
                        updated_at=now,
                    ),
                )
                .returning(SkillFileModel)
            )
            result = await session.execute(stmt)
            await session.commit()
            row = result.fetchone()
            # Fetch refreshed model for clean domain conversion
            fetch_result = await session.execute(
                select(SkillFileModel).where(
                    and_(
                        SkillFileModel.skill_id == skill_file.skill_id,
                        SkillFileModel.path == skill_file.path,
                    )
                )
            )
            model = fetch_result.scalar_one()
            return self._skill_file_model_to_domain(model)

    async def get_skill_file(self, skill_id: str, path: str) -> "SkillFile | None":
        """Get a skill file by skill_id and path."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(SkillFileModel).where(
                    and_(
                        SkillFileModel.skill_id == skill_id,
                        SkillFileModel.path == path,
                    )
                )
            )
            model = result.scalar_one_or_none()
            return self._skill_file_model_to_domain(model) if model else None

    async def list_skill_files(self, skill_id: str) -> list["SkillFile"]:
        """List all files in a skill bundle ordered by path."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(SkillFileModel)
                .where(SkillFileModel.skill_id == skill_id)
                .order_by(SkillFileModel.path)
            )
            return [self._skill_file_model_to_domain(m) for m in result.scalars().all()]

    async def delete_skill_file(self, skill_id: str, path: str) -> bool:
        """Delete a single skill file by path."""
        async with self._session_factory() as session:
            result = await session.execute(
                delete(SkillFileModel).where(
                    and_(
                        SkillFileModel.skill_id == skill_id,
                        SkillFileModel.path == path,
                    )
                )
            )
            await session.commit()
            return result.rowcount > 0

    # === MCP Server Operations ===

    def _mcp_server_model_to_domain(self, model: "McpServerModel") -> "McpServer":
        """Convert ORM model to McpServer domain model."""
        from memorylayer_server.models.mcp_server import McpServer
        return McpServer(
            id=model.id,
            tenant_id=model.tenant_id,
            workspace_id=model.workspace_id,
            user_id=model.user_id,
            name=model.name,
            description=model.description,
            transport=model.transport,
            command=model.command,
            args=list(model.args) if model.args else [],
            env=dict(model.env) if model.env else {},
            url=model.url,
            headers=dict(model.headers) if model.headers else {},
            metadata=dict(model.meta) if model.meta else {},
            source_mode=model.source_mode,
            manifest_hash=model.manifest_hash,
            enabled=model.enabled,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    async def create_mcp_server(self, mcp_server: "McpServer") -> "McpServer":
        """Store a new MCP server record."""
        now = datetime.now(timezone.utc)
        async with self._session_factory() as session:
            model = McpServerModel(
                id=mcp_server.id,
                tenant_id=mcp_server.tenant_id or "_default",
                workspace_id=mcp_server.workspace_id,
                user_id=mcp_server.user_id,
                name=mcp_server.name,
                description=mcp_server.description,
                transport=mcp_server.transport,
                command=mcp_server.command,
                args=list(mcp_server.args) if mcp_server.args else [],
                env=dict(mcp_server.env) if mcp_server.env else {},
                url=mcp_server.url,
                headers=dict(mcp_server.headers) if mcp_server.headers else {},
                meta=dict(mcp_server.metadata) if mcp_server.metadata else {},
                source_mode=mcp_server.source_mode,
                manifest_hash=mcp_server.manifest_hash,
                enabled=mcp_server.enabled,
                created_at=now,
                updated_at=now,
            )
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return self._mcp_server_model_to_domain(model)

    async def get_mcp_server(self, workspace_id: str, mcp_server_id: str) -> "McpServer | None":
        """Get MCP server by ID within a workspace."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(McpServerModel).where(
                    and_(
                        McpServerModel.id == mcp_server_id,
                        McpServerModel.workspace_id == workspace_id,
                    )
                )
            )
            model = result.scalar_one_or_none()
            return self._mcp_server_model_to_domain(model) if model else None

    async def get_mcp_server_by_name(
        self, workspace_id: str, name: str, user_id: str | None = None
    ) -> "McpServer | None":
        """Get MCP server by name within a workspace, optionally scoped to a user."""
        async with self._session_factory() as session:
            if user_id is not None:
                conditions = and_(
                    McpServerModel.workspace_id == workspace_id,
                    McpServerModel.name == name,
                    McpServerModel.user_id == user_id,
                )
            else:
                conditions = and_(
                    McpServerModel.workspace_id == workspace_id,
                    McpServerModel.name == name,
                    McpServerModel.user_id.is_(None),
                )
            result = await session.execute(select(McpServerModel).where(conditions))
            model = result.scalar_one_or_none()
            return self._mcp_server_model_to_domain(model) if model else None

    async def list_mcp_servers(
        self,
        workspace_id: str,
        user_id: str | None = None,
        name: str | None = None,
        transport: str | None = None,
        enabled: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list["McpServer"]:
        """List MCP servers for a workspace with optional filters."""
        conditions = [McpServerModel.workspace_id == workspace_id]

        if user_id is not None:
            conditions.append(McpServerModel.user_id == user_id)
        if name is not None:
            conditions.append(McpServerModel.name == name)
        if transport is not None:
            conditions.append(McpServerModel.transport == transport)
        if enabled is not None:
            conditions.append(McpServerModel.enabled == enabled)

        async with self._session_factory() as session:
            result = await session.execute(
                select(McpServerModel)
                .where(and_(*conditions))
                .order_by(McpServerModel.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return [self._mcp_server_model_to_domain(m) for m in result.scalars().all()]

    async def find_mcp_servers_by_name(self, name: str, scope_filters: list[dict]) -> list["McpServer"]:
        """Find MCP servers by name across multiple scope filters for resolution.

        Each filter dict has workspace_id and optional user_id keys.
        Returns all matches; the resolution service handles precedence ordering.
        """
        if not scope_filters:
            return []

        conditions = []
        for sf in scope_filters:
            ws = sf.get("workspace_id")
            uid = sf.get("user_id")
            if uid is not None:
                conditions.append(
                    and_(McpServerModel.workspace_id == ws, McpServerModel.user_id == uid)
                )
            else:
                conditions.append(
                    and_(McpServerModel.workspace_id == ws, McpServerModel.user_id.is_(None))
                )

        async with self._session_factory() as session:
            result = await session.execute(
                select(McpServerModel)
                .where(and_(McpServerModel.name == name, or_(*conditions)))
                .order_by(McpServerModel.updated_at.desc())
            )
            return [self._mcp_server_model_to_domain(m) for m in result.scalars().all()]

    async def update_mcp_server(
        self, workspace_id: str, mcp_server_id: str, updates: dict
    ) -> "McpServer | None":
        """Update MCP server fields."""
        if not updates:
            return await self.get_mcp_server(workspace_id, mcp_server_id)

        async with self._session_factory() as session:
            result = await session.execute(
                select(McpServerModel).where(
                    and_(
                        McpServerModel.id == mcp_server_id,
                        McpServerModel.workspace_id == workspace_id,
                    )
                )
            )
            model = result.scalar_one_or_none()
            if not model:
                return None

            for key, value in updates.items():
                if key == "metadata":
                    model.meta = value
                elif key == "args":
                    model.args = list(value) if value is not None else []
                elif key == "env":
                    model.env = dict(value) if value is not None else {}
                elif key == "headers":
                    model.headers = dict(value) if value is not None else {}
                elif hasattr(model, key):
                    setattr(model, key, value)
            model.updated_at = datetime.now(timezone.utc)

            await session.commit()
            await session.refresh(model)
            return self._mcp_server_model_to_domain(model)

    async def delete_mcp_server(self, workspace_id: str, mcp_server_id: str) -> bool:
        """Delete an MCP server record."""
        async with self._session_factory() as session:
            result = await session.execute(
                delete(McpServerModel).where(
                    and_(
                        McpServerModel.id == mcp_server_id,
                        McpServerModel.workspace_id == workspace_id,
                    )
                )
            )
            await session.commit()
            deleted = result.rowcount > 0
            if deleted:
                self.logger.debug("Deleted MCP server: %s", mcp_server_id)
            return deleted

    # ============================================
    # Knowledgebase Article Operations
    # ============================================

    def _kb_article_model_to_dict(self, model: KnowledgebaseArticleModel) -> dict:
        """Convert a KB article ORM row to the dict shape used by the KB service.

        Mirrors the SQLite ``_row_to_kb_article`` output, including ``generated_at``
        as an ISO-8601 string so both backends round-trip identically.
        """
        return {
            "workspace_id": model.workspace_id,
            "article_id": model.article_id,
            "article_type": model.article_type,
            "title": model.title,
            "content_md": model.content_md,
            "metadata": dict(model.meta) if model.meta else {},
            "generated_at": model.generated_at.isoformat() if model.generated_at else None,
        }

    async def store_kb_article(
        self,
        workspace_id: str,
        article_id: str,
        article_type: str,
        title: str,
        content_md: str,
        metadata: dict | None = None,
    ) -> dict:
        """Store a knowledgebase article (upsert on (workspace_id, article_id))."""
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        now = datetime.now(timezone.utc)
        async with self._session_factory() as session:
            stmt = (
                pg_insert(KnowledgebaseArticleModel)
                .values(
                    workspace_id=workspace_id,
                    article_id=article_id,
                    article_type=article_type,
                    title=title,
                    content_md=content_md,
                    meta=metadata or {},
                    generated_at=now,
                )
                .on_conflict_do_update(
                    index_elements=["workspace_id", "article_id"],
                    # NB: set_ keys are resolved as raw DB column names, not ORM
                    # attributes, so the JSONB column must be referenced by its
                    # actual name ("metadata") rather than the mapped attr ("meta").
                    set_=dict(
                        article_type=article_type,
                        title=title,
                        content_md=content_md,
                        metadata=metadata or {},
                        generated_at=now,
                    ),
                )
            )
            await session.execute(stmt)
            await session.commit()
        self.logger.debug("Stored KB article: %s in workspace: %s", article_id, workspace_id)
        return await self.get_kb_article(workspace_id, article_id)

    async def get_kb_article(self, workspace_id: str, article_id: str) -> dict | None:
        """Get a knowledgebase article by ID."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(KnowledgebaseArticleModel).where(
                    and_(
                        KnowledgebaseArticleModel.workspace_id == workspace_id,
                        KnowledgebaseArticleModel.article_id == article_id,
                    )
                )
            )
            model = result.scalar_one_or_none()
            return self._kb_article_model_to_dict(model) if model else None

    async def list_kb_articles(
        self,
        workspace_id: str,
        article_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """List knowledgebase articles for a workspace, newest first."""
        conditions = [KnowledgebaseArticleModel.workspace_id == workspace_id]
        if article_type is not None:
            conditions.append(KnowledgebaseArticleModel.article_type == article_type)

        async with self._session_factory() as session:
            result = await session.execute(
                select(KnowledgebaseArticleModel)
                .where(and_(*conditions))
                .order_by(KnowledgebaseArticleModel.generated_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return [self._kb_article_model_to_dict(m) for m in result.scalars().all()]

    async def delete_kb_articles(self, workspace_id: str) -> int:
        """Delete all knowledgebase articles for a workspace (for regeneration)."""
        async with self._session_factory() as session:
            result = await session.execute(
                delete(KnowledgebaseArticleModel).where(
                    KnowledgebaseArticleModel.workspace_id == workspace_id
                )
            )
            await session.commit()
            count = result.rowcount or 0
        self.logger.debug("Deleted %d KB articles for workspace: %s", count, workspace_id)
        return count

    async def delete_kb_article(self, workspace_id: str, article_id: str) -> bool:
        """Delete a single knowledgebase article by id (for stale-article GC)."""
        async with self._session_factory() as session:
            result = await session.execute(
                delete(KnowledgebaseArticleModel).where(
                    and_(
                        KnowledgebaseArticleModel.workspace_id == workspace_id,
                        KnowledgebaseArticleModel.article_id == article_id,
                    )
                )
            )
            await session.commit()
            deleted = (result.rowcount or 0) > 0
        self.logger.debug("Deleted KB article %s in workspace %s: %s", article_id, workspace_id, deleted)
        return deleted

    # ============================================
    # Graph Analysis Operations
    # ============================================

    async def store_graph_analysis(self, workspace_id: str, analysis_json: dict) -> dict:
        """Cache a graph analysis result (upsert on workspace_id)."""
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        now = datetime.now(timezone.utc)
        async with self._session_factory() as session:
            stmt = (
                pg_insert(GraphAnalysisModel)
                .values(
                    workspace_id=workspace_id,
                    analysis_json=analysis_json or {},
                    generated_at=now,
                )
                .on_conflict_do_update(
                    index_elements=["workspace_id"],
                    set_=dict(
                        analysis_json=analysis_json or {},
                        generated_at=now,
                    ),
                )
            )
            await session.execute(stmt)
            await session.commit()
        self.logger.debug("Stored graph analysis for workspace: %s", workspace_id)
        return {
            "workspace_id": workspace_id,
            "analysis_json": analysis_json,
            "generated_at": now.isoformat(),
        }

    async def get_graph_analysis(self, workspace_id: str) -> dict | None:
        """Get cached graph analysis for a workspace."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(GraphAnalysisModel).where(
                    GraphAnalysisModel.workspace_id == workspace_id
                )
            )
            model = result.scalar_one_or_none()
            if not model:
                return None
            return {
                "workspace_id": model.workspace_id,
                "analysis_json": dict(model.analysis_json) if model.analysis_json else {},
                "generated_at": model.generated_at.isoformat() if model.generated_at else None,
            }

    # ============================================
    # Entity Registry Operations
    # ============================================

    def _entity_model_to_dict(self, model: EntityModel, aliases: list[str]) -> dict:
        """Convert an EntityModel row + its aliases into the entity dict contract.

        Mirrors the SQLite ``_row_to_entity`` output (ISO-8601 timestamps) so
        both backends round-trip identically through the EntityRegistryService.
        """
        return {
            "id": model.id,
            "workspace_id": model.workspace_id,
            "entity_type": model.entity_type,
            "canonical_name": model.canonical_name,
            "normalized_name": model.normalized_name,
            "aliases": aliases,
            "confidence": model.confidence,
            "provenance": dict(model.provenance) if model.provenance else {},
            "representative_memory_id": model.representative_memory_id,
            "status": model.status,
            "merged_into": model.merged_into,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "updated_at": model.updated_at.isoformat() if model.updated_at else None,
        }

    @staticmethod
    def _normalize_alias(alias: str) -> str:
        from memorylayer_server.services.entity_registry import normalize_entity_name

        return normalize_entity_name(alias)

    async def _entity_alias_surface(self, session, workspace_id: str, entity_id: str) -> list[str]:
        result = await session.execute(
            select(EntityAliasModel.alias)
            .where(
                and_(
                    EntityAliasModel.workspace_id == workspace_id,
                    EntityAliasModel.entity_id == entity_id,
                )
            )
            .order_by(EntityAliasModel.alias)
        )
        return [r[0] for r in result.all()]

    async def store_entity(self, entity: dict) -> dict:
        """Insert a canonical entity row (and its initial aliases).

        Uses a PLAIN INSERT guarded by the ``uq_entities_active_norm`` partial
        unique index (workspace_id, entity_type, normalized_name WHERE
        status='active'), catching the resulting ``IntegrityError`` on a duplicate
        and re-fetching the winner (first-writer-wins; the race loser's provenance
        is discarded).

        We deliberately do NOT use ``INSERT ... ON CONFLICT DO NOTHING`` here.
        Against asyncpg, the ON CONFLICT partial-index INFERENCE fails
        intermittently with ``there is no unique or exclusion constraint matching
        the ON CONFLICT specification`` (SQLSTATE 42P10) EVEN THOUGH the index
        exists and matches — which silently dropped ~95% of inserts during a bulk
        entity accretion / backfill (only the first handful of a run survived). A
        plain INSERT does not depend on that inference and is robust, while the
        partial UNIQUE index still enforces the same active-name uniqueness (a real
        duplicate raises IntegrityError, which we treat as the race path).
        """
        from sqlalchemy.exc import IntegrityError

        from memorylayer_server.utils import generate_id

        now = datetime.now(timezone.utc)
        entity_id = entity.get("id") or generate_id("ent")
        workspace_id = entity["workspace_id"]
        entity_type = entity["entity_type"]
        normalized_name = entity["normalized_name"]

        try:
            async with self._session_factory() as session:
                session.add(
                    EntityModel(
                        id=entity_id,
                        workspace_id=workspace_id,
                        entity_type=entity_type,
                        canonical_name=entity["canonical_name"],
                        normalized_name=normalized_name,
                        confidence=entity.get("confidence", 1.0),
                        provenance=entity.get("provenance") or {},
                        representative_memory_id=entity.get("representative_memory_id"),
                        status=entity.get("status", "active"),
                        merged_into=entity.get("merged_into"),
                        # Optional name embedding for the enterprise embedding-fuzzy
                        # resolution tier (NULL when no embedding service available).
                        name_embedding=entity.get("name_embedding"),
                        created_at=entity.get("created_at") or now,
                        updated_at=entity.get("updated_at") or now,
                    )
                )
                await session.commit()
        except IntegrityError:
            # A concurrent writer won the active-name race (uq_entities_active_norm)
            # OR a genuine constraint was violated (e.g. the workspace FK). Re-fetch
            # the active winner: if there is one, first-writer-wins; otherwise the
            # violation was not the name race, so re-raise it.
            existing = await self.find_entity_by_normalized_name(
                workspace_id, entity_type, normalized_name
            )
            if existing is not None:
                self.logger.debug(
                    "store_entity race on (%s, %s, %s) — returning winner",
                    workspace_id, entity_type, normalized_name,
                )
                return existing
            raise

        for alias in entity.get("aliases") or []:
            await self.add_entity_alias(
                workspace_id, entity_id, alias, self._normalize_alias(alias), source="initial"
            )
        self.logger.debug("Stored entity %s in workspace %s", entity_id, workspace_id)
        return await self.get_entity(workspace_id, entity_id)

    async def get_entity(self, workspace_id: str, entity_id: str) -> dict | None:
        """Get a canonical entity by id (with aliases folded in)."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(EntityModel).where(
                    and_(EntityModel.workspace_id == workspace_id, EntityModel.id == entity_id)
                )
            )
            model = result.scalar_one_or_none()
            if not model:
                return None
            aliases = await self._entity_alias_surface(session, workspace_id, entity_id)
            return self._entity_model_to_dict(model, aliases)

    async def find_entity_by_normalized_name(
        self,
        workspace_id: str,
        entity_type: str,
        normalized_name: str,
    ) -> dict | None:
        """Find the active entity matching (workspace, type, normalized_name)."""
        async with self._session_factory() as session:
            result = await session.execute(
                select(EntityModel).where(
                    and_(
                        EntityModel.workspace_id == workspace_id,
                        EntityModel.entity_type == entity_type,
                        EntityModel.normalized_name == normalized_name,
                        EntityModel.status == "active",
                    )
                )
            )
            model = result.scalar_one_or_none()
            if not model:
                return None
            aliases = await self._entity_alias_surface(session, workspace_id, model.id)
            return self._entity_model_to_dict(model, aliases)

    async def find_entities_by_normalized_name_any_type(
        self,
        workspace_id: str,
        normalized_name: str,
    ) -> list[dict]:
        """Find ALL active entities matching (workspace, normalized_name), any type.

        Ordered PERSON-first (so the accretion name-first resolver can prefer an
        existing PERSON node), then by entity id for a deterministic tie-break.
        Parity with the OSS relational backends.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(EntityModel)
                .where(
                    and_(
                        EntityModel.workspace_id == workspace_id,
                        EntityModel.normalized_name == normalized_name,
                        EntityModel.status == "active",
                    )
                )
                .order_by(
                    case((EntityModel.entity_type == "person", 0), else_=1),
                    EntityModel.id,
                )
            )
            models = result.scalars().all()
            out = []
            for model in models:
                aliases = await self._entity_alias_surface(session, workspace_id, model.id)
                out.append(self._entity_model_to_dict(model, aliases))
            return out

    async def find_entities_by_normalized_alias(
        self,
        workspace_id: str,
        normalized_alias: str,
        entity_type: str | None = None,
    ) -> list[dict]:
        """Find active entities whose aliases include normalized_alias."""
        conditions = [
            EntityModel.workspace_id == workspace_id,
            EntityAliasModel.normalized_alias == normalized_alias,
            EntityModel.status == "active",
        ]
        if entity_type is not None:
            conditions.append(EntityModel.entity_type == entity_type)
        async with self._session_factory() as session:
            result = await session.execute(
                select(EntityModel)
                .join(
                    EntityAliasModel,
                    and_(
                        EntityAliasModel.entity_id == EntityModel.id,
                        EntityAliasModel.workspace_id == EntityModel.workspace_id,
                    ),
                )
                .where(and_(*conditions))
                .order_by(EntityModel.id)
                .distinct()
            )
            models = result.scalars().all()
            out = []
            for model in models:
                aliases = await self._entity_alias_surface(session, workspace_id, model.id)
                out.append(self._entity_model_to_dict(model, aliases))
            return out

    async def find_entities_by_name_embedding(
        self,
        workspace_id: str,
        entity_type: str,
        embedding: list[float],
        *,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> list[dict]:
        """ANN over active same-type entities by ``name_embedding`` (cosine).

        Powers the enterprise embedding-fuzzy resolution tier. Uses pgvector's
        cosine distance operator (``<=>`` via SQLAlchemy ``cosine_distance``);
        cosine similarity = ``1 - distance``. Restricted to ACTIVE entities of
        the given ``entity_type`` in ``workspace_id`` that actually have a
        ``name_embedding`` (the partial HNSW index covers exactly these rows).
        Results are best-first; each dict carries an extra ``score`` key.
        """
        async with self._session_factory() as session:
            distance = EntityModel.name_embedding.cosine_distance(embedding)
            result = await session.execute(
                select(EntityModel, (1 - distance).label("score"))
                .where(
                    and_(
                        EntityModel.workspace_id == workspace_id,
                        EntityModel.entity_type == entity_type,
                        EntityModel.status == "active",
                        EntityModel.name_embedding.is_not(None),
                    )
                )
                .order_by(distance.asc())
                .limit(limit)
            )
            out: list[dict] = []
            for model, score in result.all():
                score = float(score)
                if score < min_score:
                    continue
                aliases = await self._entity_alias_surface(session, workspace_id, model.id)
                d = self._entity_model_to_dict(model, aliases)
                d["score"] = score
                out.append(d)
            return out

    async def add_entity_alias(
        self,
        workspace_id: str,
        entity_id: str,
        alias: str,
        normalized_alias: str,
        source: str = "manual",
    ) -> None:
        """Add an alias to an entity (idempotent on (entity_id, normalized_alias))."""
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        from memorylayer_server.utils import generate_id

        async with self._session_factory() as session:
            stmt = (
                pg_insert(EntityAliasModel)
                .values(
                    id=generate_id("ealias"),
                    workspace_id=workspace_id,
                    entity_id=entity_id,
                    alias=alias,
                    normalized_alias=normalized_alias,
                    source=source,
                    created_at=datetime.now(timezone.utc),
                )
                .on_conflict_do_nothing(index_elements=["entity_id", "normalized_alias"])
            )
            await session.execute(stmt)
            await session.commit()

    async def add_entity_member(
        self,
        workspace_id: str,
        entity_id: str,
        memory_id: str,
        role: str = "mention",
        confidence: float = 1.0,
        meta: dict | None = None,
    ) -> dict:
        """Attach a memory to an entity as a member (idempotent on (entity_id, memory_id, role))."""
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        from memorylayer_server.utils import generate_id

        async with self._session_factory() as session:
            stmt = (
                pg_insert(EntityMemberModel)
                .values(
                    id=generate_id("emem"),
                    workspace_id=workspace_id,
                    entity_id=entity_id,
                    memory_id=memory_id,
                    role=role,
                    confidence=confidence,
                    meta=meta or {},
                    created_at=datetime.now(timezone.utc),
                )
                .on_conflict_do_nothing(index_elements=["entity_id", "memory_id", "role"])
            )
            await session.execute(stmt)
            await session.commit()
        return {"entity_id": entity_id, "memory_id": memory_id, "role": role, "confidence": confidence}

    async def list_entity_members(
        self,
        workspace_id: str,
        entity_id: str,
        role: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """List member rows for an entity, optionally filtered by role."""
        conditions = [
            EntityMemberModel.workspace_id == workspace_id,
            EntityMemberModel.entity_id == entity_id,
        ]
        if role is not None:
            conditions.append(EntityMemberModel.role == role)
        async with self._session_factory() as session:
            result = await session.execute(
                select(EntityMemberModel)
                .where(and_(*conditions))
                .order_by(EntityMemberModel.created_at, EntityMemberModel.memory_id)
                .limit(limit)
            )
            return [
                {
                    "entity_id": m.entity_id,
                    "memory_id": m.memory_id,
                    "role": m.role,
                    "confidence": m.confidence,
                }
                for m in result.scalars().all()
            ]

    async def list_workspace_entities(
        self,
        workspace_id: str,
        *,
        status: str = "active",
        limit: int = 10000,
    ) -> list[dict]:
        """List ALL canonical entities for a workspace, ordered by id.

        Single-pass: one SELECT for the entity rows, one SELECT for ALL aliases
        in the workspace grouped in Python (avoids the per-entity N+1 alias
        lookup that ``get_entity`` does). Aliases are surfaced in the same
        deterministic order (alias text) as ``_entity_alias_surface``.
        """
        async with self._session_factory() as session:
            conditions = [EntityModel.workspace_id == workspace_id]
            if status is not None:
                conditions.append(EntityModel.status == status)
            ent_result = await session.execute(
                select(EntityModel)
                .where(and_(*conditions))
                .order_by(EntityModel.id)
                .limit(limit)
            )
            models = list(ent_result.scalars().all())
            if not models:
                return []

            alias_result = await session.execute(
                select(EntityAliasModel.entity_id, EntityAliasModel.alias)
                .where(EntityAliasModel.workspace_id == workspace_id)
                .order_by(EntityAliasModel.entity_id, EntityAliasModel.alias)
            )
            aliases_by_entity: dict[str, list[str]] = {}
            for entity_id, alias in alias_result.all():
                aliases_by_entity.setdefault(entity_id, []).append(alias)

            return [
                self._entity_model_to_dict(m, aliases_by_entity.get(m.id, []))
                for m in models
            ]

    async def list_workspace_entity_members(
        self,
        workspace_id: str,
        *,
        role: str | None = None,
        limit: int = 100000,
    ) -> list[dict]:
        """List ALL member edges for a workspace, ordered by (entity_id, memory_id, role)."""
        conditions = [EntityMemberModel.workspace_id == workspace_id]
        if role is not None:
            conditions.append(EntityMemberModel.role == role)
        async with self._session_factory() as session:
            result = await session.execute(
                select(EntityMemberModel)
                .where(and_(*conditions))
                .order_by(
                    EntityMemberModel.entity_id,
                    EntityMemberModel.memory_id,
                    EntityMemberModel.role,
                )
                .limit(limit)
            )
            return [
                {
                    "entity_id": m.entity_id,
                    "memory_id": m.memory_id,
                    "role": m.role,
                    "confidence": m.confidence,
                }
                for m in result.scalars().all()
            ]

    async def reassign_entity_members(
        self,
        workspace_id: str,
        source_id: str,
        target_id: str,
    ) -> int:
        """Reassign all members from source_id to target_id (for merge).

        Source members that would collide with an existing target member (same
        (memory_id, role)) are deleted first to honor the unique constraint.
        Returns the number of member rows moved.
        """
        async with self._session_factory() as session:
            # Existing target (memory_id, role) keys.
            target_rows = await session.execute(
                select(EntityMemberModel.memory_id, EntityMemberModel.role).where(
                    and_(
                        EntityMemberModel.workspace_id == workspace_id,
                        EntityMemberModel.entity_id == target_id,
                    )
                )
            )
            target_keys = {(mid, role) for mid, role in target_rows.all()}

            source_rows = await session.execute(
                select(EntityMemberModel).where(
                    and_(
                        EntityMemberModel.workspace_id == workspace_id,
                        EntityMemberModel.entity_id == source_id,
                    )
                )
            )
            moved = 0
            for member in source_rows.scalars().all():
                if (member.memory_id, member.role) in target_keys:
                    await session.delete(member)
                    continue
                member.entity_id = target_id
                moved += 1
            await session.commit()
        return moved

    async def update_entity(
        self,
        workspace_id: str,
        entity_id: str,
        **updates,
    ) -> dict | None:
        """Update mutable entity fields (status/merged_into/confidence/...)."""
        allowed = {
            "entity_type",
            "canonical_name",
            "normalized_name",
            "confidence",
            "provenance",
            "representative_memory_id",
            "status",
            "merged_into",
            "name_embedding",
        }
        clean = {k: v for k, v in updates.items() if k in allowed}
        if not clean:
            return await self.get_entity(workspace_id, entity_id)
        clean["updated_at"] = datetime.now(timezone.utc)
        async with self._session_factory() as session:
            await session.execute(
                update(EntityModel)
                .where(and_(EntityModel.workspace_id == workspace_id, EntityModel.id == entity_id))
                .values(**clean)
            )
            await session.commit()
        return await self.get_entity(workspace_id, entity_id)

    async def delete_workspace_entities(self, workspace_id: str) -> int:
        """Hard-delete ALL canonical entities in a workspace; returns rows deleted.

        Aliases (entity_aliases) and members (entity_members) cascade away via
        their ``entity_id`` FK (ondelete=CASCADE), so one DELETE on ``entities``
        clears the whole registry for the workspace. Used by the entity-registry
        backfill reset to rebuild a clean typed entity graph.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                delete(EntityModel).where(EntityModel.workspace_id == workspace_id)
            )
            await session.commit()
        return int(result.rowcount or 0)

    # ============================================
    # Internal Versioned Resource Operations
    # ============================================

    def _versioned_store(self) -> PostgreSQLVersionedResourceStore:
        if self._versioned_resource_store is None:
            if self._session_factory is None:
                raise RuntimeError("PostgreSQL storage is not connected")
            self._versioned_resource_store = PostgreSQLVersionedResourceStore(
                self._session_factory
            )
        return self._versioned_resource_store

    async def mutate_versioned_resource(
        self,
        mutation: VersionedResourceMutation,
    ) -> VersionedResourceMutationResult:
        return await self._versioned_store().mutate(mutation)

    async def get_versioned_resource_operation(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        operation_id: str,
        request_hash: str,
    ) -> VersionedResourceMutationResult | None:
        return await self._versioned_store().get_operation_result(
            tenant_id, workspace_id, namespace, operation_id, request_hash
        )

    async def get_versioned_resource(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        resource_id: str,
        *,
        include_deleted: bool = False,
    ) -> VersionedResource | None:
        return await self._versioned_store().get(
            tenant_id,
            workspace_id,
            namespace,
            resource_id,
            include_deleted=include_deleted,
        )

    async def get_versioned_resource_by_key(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        resource_key: str,
        *,
        include_deleted: bool = False,
    ) -> VersionedResource | None:
        return await self._versioned_store().get_by_key(
            tenant_id,
            workspace_id,
            namespace,
            resource_key,
            include_deleted=include_deleted,
        )

    async def list_versioned_resources(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        *,
        limit: int,
        before_sequence: int | None = None,
        include_deleted: bool = False,
    ) -> list[VersionedResource]:
        return await self._versioned_store().list(
            tenant_id,
            workspace_id,
            namespace,
            limit=limit,
            before_sequence=before_sequence,
            include_deleted=include_deleted,
        )

    async def list_versioned_resource_revisions(
        self,
        tenant_id: str,
        workspace_id: str,
        namespace: str,
        resource_id: str,
        *,
        limit: int,
        before_sequence: int | None = None,
    ) -> list[VersionedResourceRevision]:
        return await self._versioned_store().list_revisions(
            tenant_id,
            workspace_id,
            namespace,
            resource_id,
            limit=limit,
            before_sequence=before_sequence,
        )


class PostgreSQLStoragePlugin(StoragePluginBase):
    """Plugin for PostgreSQL storage backend with cold tier support."""
    PROVIDER_NAME = 'postgresql'

    def initialize(self, v: Variables, logger: Logger) -> object | None:
        from ..services.compression import get_compression_service

        connection_string = v.environ(
            MEMORYLAYER_POSTGRESQL_URL,
            default=DEFAULT_MEMORYLAYER_POSTGRESQL_URL
        )
        connection_string = connection_string.replace("postgres://", "postgresql://").replace("postgresql://", "postgresql+asyncpg://")

        pool_size = int(v.environ(
            MEMORYLAYER_POSTGRESQL_POOL_SIZE,
            default=DEFAULT_MEMORYLAYER_POSTGRESQL_POOL_SIZE
        ))

        compression_service = get_compression_service(v)
        if compression_service is not None:
            logger.info("Compression service available; cold tier will use embedding-based search")
        else:
            logger.info("No compression service; cold tier will use importance-based fallback")

        logger.info("Initializing PostgreSQL storage backend: %s", connection_string.split('@')[-1])
        return PostgreSQLBackend(
            v=v,
            connection_string=connection_string,
            pool_size=pool_size,
            compression_service=compression_service,
        )
