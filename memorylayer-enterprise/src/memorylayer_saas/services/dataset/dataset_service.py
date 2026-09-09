"""Dataset service -- orchestrates the upload/profile/summarize/embed pipeline.

Pipeline phases:
    1. Upload: validate, hash, dedup, convert to Parquet, store blob, create DB records
    2. Profile: scan with Polars to compute column-level statistics and schema
    3. Summarize: LLM generates natural-language summaries from the profile
    4. Embed: embed summaries and store as memories
    5. Finalize: update dataset and job status
"""
import hashlib
import io
import re
import uuid
from datetime import datetime, timezone
from logging import Logger
from typing import Any, Optional

from scitrera_app_framework import Variables, get_extension, get_logger

from memorylayer_server.services.storage import EXT_STORAGE_BACKEND
from memorylayer_server.services.tasks import EXT_TASK_SERVICE
from memorylayer_server.services.embedding import EXT_EMBEDDING_SERVICE, EmbeddingService
from memorylayer_server.services.llm import EXT_LLM_SERVICE, LLMService
from memorylayer_server.models.memory import RememberInput, MemoryType
from memorylayer_server.models.generation import GenerationActivity

from . import DatasetServicePluginBase, EXT_DATASET_SERVICE
from ..document import EXT_BLOB_STORAGE_SERVICE
from ..document.blob_storage import BlobStorageService
from ...config import (
    MEMORYLAYER_DATASET_MAX_FILE_SIZE,
    DEFAULT_MEMORYLAYER_DATASET_MAX_FILE_SIZE,
)
from ...models.dataset import (
    Dataset,
    DatasetJob,
    DatasetColumn,
    DatasetFormat,
    DatasetProfilingOptions,
    DatasetSliceRequest,
    DatasetSliceResult,
    DatasetStatus,
    ColumnType,
)


class DatasetService:
    """Orchestrates dataset ingestion: upload, profile, summarize, embed."""

    def __init__(
        self,
        v: Variables,
        storage_backend,
        blob_storage: BlobStorageService,
        task_service,
        embedding_service,
        llm_service,
        max_file_size: int,
        logger: Logger,
    ):
        self._v = v
        self._storage = storage_backend
        self._blob = blob_storage
        self._tasks = task_service
        self._embedding = embedding_service
        self._llm = llm_service
        self._max_file_size = max_file_size
        self.logger = logger

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def upload_dataset(
        self,
        workspace_id: str,
        file_data: bytes,
        filename: str,
        name: Optional[str] = None,
        dataset_format: Optional[DatasetFormat] = None,
        profiling_options: Optional[DatasetProfilingOptions] = None,
        metadata: Optional[dict] = None,
    ) -> tuple[Dataset, DatasetJob]:
        """Upload a dataset and schedule profiling.

        Phase 1: validate size, compute hash, detect format, convert to Parquet,
        persist blob, create DB records, schedule background profiling.

        Args:
            workspace_id: Target workspace identifier.
            file_data: Raw file bytes.
            filename: Original filename (used for format detection).
            name: Human-readable dataset name (defaults to filename stem).
            dataset_format: Explicit format override (auto-detected if None).
            profiling_options: Profiling configuration.
            metadata: Arbitrary user metadata.

        Returns:
            Tuple of (Dataset, DatasetJob) created records.

        Raises:
            ValueError: If file exceeds size limit, format is unsupported,
                or a duplicate content hash already exists.
        """
        if len(file_data) > self._max_file_size:
            raise ValueError(
                "File size %d exceeds maximum %d" % (len(file_data), self._max_file_size)
            )

        if dataset_format is None:
            dataset_format = self._detect_format(filename)

        content_hash = hashlib.sha256(file_data).hexdigest()

        existing = await self._storage.find_dataset_by_hash(workspace_id, content_hash)
        if existing:
            raise ValueError("Duplicate dataset: %s has same content hash" % existing.id)

        if profiling_options is None:
            profiling_options = DatasetProfilingOptions()

        if name is None:
            name = filename.rsplit(".", 1)[0] if "." in filename else filename

        ds_id = "ds_%s" % uuid.uuid4().hex[:12]
        job_id = "dsjob_%s" % uuid.uuid4().hex[:12]

        # Store original file
        original_path = self._blob_path(workspace_id, ds_id, filename)
        await self._blob.store_file(original_path, file_data)

        # Convert to Parquet if not already
        if dataset_format == DatasetFormat.PARQUET:
            parquet_path = original_path
            original_storage_path = None
        else:
            parquet_bytes = await self._convert_to_parquet(file_data, dataset_format)
            parquet_filename = filename.rsplit(".", 1)[0] + ".parquet"
            parquet_path = self._blob_path(workspace_id, ds_id, parquet_filename)
            await self._blob.store_file(parquet_path, parquet_bytes)
            original_storage_path = original_path

        ds = Dataset(
            id=ds_id,
            workspace_id=workspace_id,
            name=name,
            filename=filename,
            format=dataset_format,
            content_hash=content_hash,
            size_bytes=len(file_data),
            status=DatasetStatus.PENDING,
            storage_path=parquet_path,
            original_storage_path=original_storage_path,
            target_context_id=profiling_options.target_context_id,
            profiling_options=profiling_options,
            metadata=metadata or {},
        )
        ds = await self._storage.create_dataset(ds)

        job = DatasetJob(
            id=job_id,
            workspace_id=workspace_id,
            dataset_ids=[ds_id],
            status="queued",
        )
        job = await self._storage.create_dataset_job(job)

        await self._tasks.schedule_task(
            "dataset_profile",
            {
                "dataset_id": ds_id,
                "job_id": job_id,
                "workspace_id": workspace_id,
            },
            priority=3,
        )

        self.logger.info(
            "Uploaded dataset %s (%s, %d bytes, format=%s), job %s queued",
            ds_id, filename, len(file_data), dataset_format.value, job_id,
        )
        return ds, job

    async def get_dataset(self, dataset_id: str, workspace_id: Optional[str] = None) -> Optional[Dataset]:
        """Retrieve a dataset record by ID."""
        return await self._storage.get_dataset(dataset_id, workspace_id)

    async def list_datasets(
        self,
        workspace_id: str,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Dataset], int]:
        """List datasets in a workspace with optional status filter."""
        return await self._storage.list_datasets(
            workspace_id, status=status, limit=limit, offset=offset,
        )

    async def delete_dataset(
        self,
        dataset_id: str,
        workspace_id: Optional[str] = None,
        delete_memories: bool = False,
    ) -> None:
        """Delete a dataset and optionally its extracted memories.

        Raises:
            ValueError: If dataset not found.
        """
        ds = await self._storage.get_dataset(dataset_id, workspace_id)
        if not ds:
            raise ValueError("Dataset not found: %s" % dataset_id)

        if delete_memories and ds.memory_ids:
            for mem_id in ds.memory_ids:
                try:
                    await self._storage.delete_memory(ds.workspace_id, mem_id)
                except Exception as exc:
                    self.logger.warning("Failed to delete memory %s: %s", mem_id, exc)

        blob_prefix = "datasets/%s/%s" % (ds.workspace_id, ds.id)
        await self._blob.delete_tree(blob_prefix)

        await self._storage.delete_dataset(dataset_id)
        self.logger.info(
            "Deleted dataset %s (delete_memories=%s)", dataset_id, delete_memories,
        )

    async def get_job(self, job_id: str) -> Optional[DatasetJob]:
        """Get a dataset processing job by ID."""
        return await self._storage.get_dataset_job(job_id)

    async def list_jobs(
        self, workspace_id: str, status: Optional[str] = None, limit: int = 50,
    ) -> list[DatasetJob]:
        """List dataset processing jobs for a workspace."""
        return await self._storage.list_dataset_jobs(
            workspace_id, status=status, limit=limit,
        )

    async def cancel_job(self, job_id: str) -> None:
        """Cancel a queued or running dataset job.

        Raises:
            ValueError: If job not found or already in a terminal state.
        """
        job = await self._storage.get_dataset_job(job_id)
        if not job:
            raise ValueError("Job not found: %s" % job_id)
        if job.status in ("completed", "failed", "cancelled"):
            raise ValueError(
                "Job %s already in terminal state: %s" % (job_id, job.status)
            )
        await self._storage.update_dataset_job(
            job_id,
            status="cancelled",
            completed_at=datetime.now(timezone.utc),
        )
        self.logger.info("Cancelled dataset job %s", job_id)

    # ------------------------------------------------------------------ #
    # Slice / Query API
    # ------------------------------------------------------------------ #

    async def query_slice(
        self,
        dataset_id: str,
        workspace_id: str,
        request: DatasetSliceRequest,
    ) -> DatasetSliceResult:
        """Execute a slice query against the dataset's Parquet file using DuckDB.

        Args:
            dataset_id: Dataset identifier.
            workspace_id: Workspace scope.
            request: Slice request specifying columns, filters, SQL, etc.

        Returns:
            DatasetSliceResult with column names, rows, and metadata.

        Raises:
            ValueError: If dataset not found or not yet profiled.
        """
        import duckdb

        ds = await self._storage.get_dataset(dataset_id, workspace_id)
        if not ds:
            raise ValueError("Dataset not found: %s" % dataset_id)
        if ds.status not in (DatasetStatus.COMPLETED, DatasetStatus.SUMMARIZING):
            raise ValueError(
                "Dataset %s is not ready for queries (status=%s)" % (dataset_id, ds.status.value)
            )

        parquet_bytes = await self._blob.retrieve_file(ds.storage_path)
        parquet_buf = io.BytesIO(parquet_bytes)

        conn = duckdb.connect(":memory:")
        try:
            conn.register("data", conn.read_parquet(parquet_buf))  # noqa: F841

            if request.sql:
                sql = self._sanitize_sql(request.sql)
            else:
                sql = self._build_sql(request)

            # Count total matching rows (without limit/offset)
            count_sql = "SELECT COUNT(*) FROM (%s) _sub" % self._strip_limit(sql)
            total_matching = conn.execute(count_sql).fetchone()[0]

            result = conn.execute(sql)
            columns = [desc[0] for desc in result.description]
            dtypes = [desc[1] for desc in result.description]
            rows = result.fetchall()

            return DatasetSliceResult(
                dataset_id=dataset_id,
                columns=columns,
                dtypes=[str(d) for d in dtypes],
                rows=[list(row) for row in rows],
                total_matching=total_matching,
                returned_count=len(rows),
                sql_executed=sql,
            )
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Pipeline phases (called from task handlers)
    # ------------------------------------------------------------------ #

    async def profile_dataset(self, dataset_id: str, workspace_id: str) -> list[DatasetColumn]:
        """Phase 2: Profile the dataset using Polars.

        Scans the Parquet file and computes per-column statistics.

        Returns:
            List of DatasetColumn with populated statistics.
        """
        import polars as pl

        ds = await self._storage.get_dataset(dataset_id, workspace_id)
        if not ds:
            raise ValueError("Dataset not found: %s" % dataset_id)

        parquet_bytes = await self._blob.retrieve_file(ds.storage_path)
        df = pl.read_parquet(io.BytesIO(parquet_bytes))

        row_count = len(df)
        columns: list[DatasetColumn] = []
        opts = ds.profiling_options

        for col_name in df.columns:
            series = df[col_name]
            col = self._profile_column(col_name, series, opts)
            columns.append(col)

        await self._storage.update_dataset(
            dataset_id,
            row_count=row_count,
            column_count=len(columns),
            columns=[c.model_dump() for c in columns],
        )

        self.logger.info(
            "Profiled dataset %s: %d rows, %d columns",
            dataset_id, row_count, len(columns),
        )
        return columns

    async def summarize_dataset(self, dataset_id: str, workspace_id: str) -> str:
        """Phase 3: Generate LLM natural-language summary from the profile.

        Returns:
            The generated summary text.
        """
        import polars as pl

        ds = await self._storage.get_dataset(dataset_id, workspace_id)
        if not ds:
            raise ValueError("Dataset not found: %s" % dataset_id)

        # Build a profile description for the LLM
        profile_text = self._build_profile_prompt(ds)

        # Include sample rows
        parquet_bytes = await self._blob.retrieve_file(ds.storage_path)
        df = pl.read_parquet(io.BytesIO(parquet_bytes))
        sample_size = min(ds.profiling_options.sample_rows, len(df))
        sample_df = df.head(sample_size)
        sample_csv = sample_df.write_csv()

        prompt = (
            "You are a data analyst. Below is a statistical profile and sample data "
            "from a dataset named '%s' with %d rows and %d columns.\n\n"
            "## Statistical Profile\n%s\n\n"
            "## Sample Data (first %d rows as CSV)\n```csv\n%s```\n\n"
            "Generate a comprehensive but concise summary of this dataset. Include:\n"
            "1. What the dataset appears to contain (domain/topic)\n"
            "2. Key statistical findings per column\n"
            "3. Notable patterns, distributions, or anomalies\n"
            "4. Potential relationships between columns\n"
            "5. Data quality observations (nulls, outliers)\n"
            "6. If temporal data is detected, describe the time range and granularity\n\n"
            "Be specific with numbers. This summary will be stored as a memory for "
            "future retrieval by AI agents."
        ) % (
            ds.name, ds.row_count, ds.column_count,
            profile_text, sample_size, sample_csv,
        )

        summary = await self._llm.synthesize(
            prompt,
            activity=GenerationActivity.SYNTHESIS,
        )

        await self._storage.update_dataset(
            dataset_id, profile_summary=summary,
        )

        self.logger.info("Generated summary for dataset %s (%d chars)", dataset_id, len(summary))
        return summary

    async def embed_and_store_memories(
        self, dataset_id: str, workspace_id: str,
    ) -> list[str]:
        """Phase 4: Embed the summary and store as memories.

        Creates memories from:
        - The overall dataset summary
        - Per-column statistical summaries for important columns

        Returns:
            List of created memory IDs.
        """
        ds = await self._storage.get_dataset(dataset_id, workspace_id)
        if not ds:
            raise ValueError("Dataset not found: %s" % dataset_id)

        memory_ids: list[str] = []
        base_tags = ["dataset", "ds:%s" % ds.id, "tabular_data"]

        # Memory 1: Overall dataset summary
        if ds.profile_summary:
            summary_input = RememberInput(
                content=ds.profile_summary,
                type=MemoryType.SEMANTIC,
                importance=ds.profiling_options.importance,
                tags=base_tags + ["dataset_summary"],
                metadata={
                    "source_dataset_id": ds.id,
                    "source_dataset_name": ds.name,
                    "row_count": ds.row_count,
                    "column_count": ds.column_count,
                    "format": ds.format.value,
                },
                # context_id RESERVED / unused as a filter → persist NULL for the
                # "_default" sentinel (see MemoryModel.context_id).
                context_id=(
                    None if ds.target_context_id in (None, "_default")
                    else ds.target_context_id
                ),
                source_dataset_id=ds.id,
            )

            embedding = await self._embedding.embed(ds.profile_summary)
            memory = await self._storage.create_memory(
                workspace_id=workspace_id,
                input=summary_input,
                embedding=embedding,
            )
            memory_ids.append(memory.id)

        # Memory 2+: Per-column summaries for columns with interesting stats
        for col in ds.columns:
            col_summary = self._build_column_summary(ds.name, col)
            if not col_summary:
                continue

            col_input = RememberInput(
                content=col_summary,
                type=MemoryType.SEMANTIC,
                importance=max(0.3, ds.profiling_options.importance - 0.1),
                tags=base_tags + ["column_profile", "col:%s" % col.name],
                metadata={
                    "source_dataset_id": ds.id,
                    "source_dataset_name": ds.name,
                    "column_name": col.name,
                    "column_type": col.column_type.value,
                },
                # context_id RESERVED / unused as a filter → persist NULL for the
                # "_default" sentinel (see MemoryModel.context_id).
                context_id=(
                    None if ds.target_context_id in (None, "_default")
                    else ds.target_context_id
                ),
                source_dataset_id=ds.id,
            )

            embedding = await self._embedding.embed(col_summary)
            memory = await self._storage.create_memory(
                workspace_id=workspace_id,
                input=col_input,
                embedding=embedding,
            )
            memory_ids.append(memory.id)

        self.logger.info(
            "Created %d memories for dataset %s", len(memory_ids), dataset_id,
        )
        return memory_ids

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _blob_path(self, workspace_id: str, dataset_id: str, filename: str) -> str:
        return "datasets/%s/%s/%s" % (workspace_id, dataset_id, filename)

    @staticmethod
    def _detect_format(filename: str) -> DatasetFormat:
        """Detect dataset format from filename extension.

        Raises:
            ValueError: If the extension is unsupported.
        """
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        mapping = {
            "csv": DatasetFormat.CSV,
            "tsv": DatasetFormat.TSV,
            "parquet": DatasetFormat.PARQUET,
            "pq": DatasetFormat.PARQUET,
            "jsonl": DatasetFormat.JSON_LINES,
            "ndjson": DatasetFormat.JSON_LINES,
            "xlsx": DatasetFormat.EXCEL,
            "xls": DatasetFormat.EXCEL,
        }
        if ext not in mapping:
            raise ValueError("Unsupported dataset format: .%s" % ext)
        return mapping[ext]

    @staticmethod
    async def _convert_to_parquet(file_data: bytes, fmt: DatasetFormat) -> bytes:
        """Convert source file to Parquet bytes using Polars."""
        import asyncio
        import polars as pl

        def _convert() -> bytes:
            buf = io.BytesIO(file_data)
            if fmt == DatasetFormat.CSV:
                df = pl.read_csv(buf)
            elif fmt == DatasetFormat.TSV:
                df = pl.read_csv(buf, separator="\t")
            elif fmt == DatasetFormat.JSON_LINES:
                df = pl.read_ndjson(buf)
            elif fmt == DatasetFormat.EXCEL:
                df = pl.read_excel(buf)
            else:
                raise ValueError("Cannot convert format %s to Parquet" % fmt.value)

            out = io.BytesIO()
            df.write_parquet(out)
            return out.getvalue()

        return await asyncio.to_thread(_convert)

    @staticmethod
    def _profile_column(
        col_name: str,
        series,  # polars.Series
        opts: DatasetProfilingOptions,
    ) -> DatasetColumn:
        """Compute statistics for a single column."""
        import polars as pl

        dtype_str = str(series.dtype)
        null_count = series.null_count()
        total = len(series)
        null_pct = (null_count / total * 100) if total > 0 else 0.0
        unique_count = series.n_unique()

        col = DatasetColumn(
            name=col_name,
            dtype=dtype_str,
            nullable=null_count > 0,
            null_count=null_count,
            null_percent=round(null_pct, 2),
            unique_count=unique_count,
        )

        # Detect column type and compute type-specific stats
        if series.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            col.column_type = ColumnType.INTEGER
            non_null = series.drop_nulls()
            if len(non_null) > 0:
                col.min_value = float(non_null.min())
                col.max_value = float(non_null.max())
                col.mean_value = round(float(non_null.mean()), 4)
                col.median_value = float(non_null.median())
                col.std_value = round(float(non_null.std()), 4) if len(non_null) > 1 else 0.0
                col.p25_value = float(non_null.quantile(0.25))
                col.p75_value = float(non_null.quantile(0.75))

                # Histogram
                try:
                    hist = non_null.cast(pl.Float64).hist(bin_count=opts.histogram_bins)
                    col.histogram = {
                        "bins": hist["breakpoint"].to_list(),
                        "counts": hist["count"].to_list(),
                    }
                except Exception:
                    pass

        elif series.dtype in (pl.Float32, pl.Float64, pl.Decimal):
            col.column_type = ColumnType.FLOAT
            non_null = series.drop_nulls().cast(pl.Float64)
            if len(non_null) > 0:
                col.min_value = round(float(non_null.min()), 6)
                col.max_value = round(float(non_null.max()), 6)
                col.mean_value = round(float(non_null.mean()), 6)
                col.median_value = round(float(non_null.median()), 6)
                col.std_value = round(float(non_null.std()), 6) if len(non_null) > 1 else 0.0
                col.p25_value = round(float(non_null.quantile(0.25)), 6)
                col.p75_value = round(float(non_null.quantile(0.75)), 6)

                try:
                    hist = non_null.hist(bin_count=opts.histogram_bins)
                    col.histogram = {
                        "bins": hist["breakpoint"].to_list(),
                        "counts": hist["count"].to_list(),
                    }
                except Exception:
                    pass

        elif series.dtype == pl.Boolean:
            col.column_type = ColumnType.BOOLEAN
            vc = series.value_counts()
            col.top_values = [
                {"value": str(row[0]), "count": row[1], "percent": round(row[1] / total * 100, 2)}
                for row in vc.iter_rows()
            ]

        elif series.dtype in (pl.Date, pl.Datetime, pl.Time, pl.Duration):
            col.column_type = ColumnType.DATETIME
            col.is_temporal = True
            non_null = series.drop_nulls()
            if len(non_null) > 0:
                col.temporal_range_start = str(non_null.min())
                col.temporal_range_end = str(non_null.max())
                if len(non_null) > 1 and series.dtype in (pl.Datetime, pl.Date):
                    diffs = non_null.sort().diff().drop_nulls()
                    if len(diffs) > 0:
                        try:
                            median_diff = diffs.median()
                            col.temporal_resolution = _detect_resolution(median_diff)
                        except Exception:
                            pass

        elif series.dtype in (pl.Utf8, pl.String, pl.Categorical):
            non_null = series.drop_nulls().cast(pl.Utf8)
            # Check if categorical (low cardinality relative to row count)
            if unique_count <= min(50, total * 0.05) and unique_count > 0:
                col.column_type = ColumnType.CATEGORICAL
                vc = series.value_counts().sort("count", descending=True).head(20)
                col.top_values = [
                    {"value": str(row[0]), "count": row[1], "percent": round(row[1] / total * 100, 2)}
                    for row in vc.iter_rows()
                ]
            else:
                col.column_type = ColumnType.STRING

            if len(non_null) > 0:
                lengths = non_null.str.len_chars()
                col.min_length = int(lengths.min())
                col.max_length = int(lengths.max())
                col.avg_length = round(float(lengths.mean()), 2)

            # Time series detection on string columns
            if opts.detect_time_series and col.column_type == ColumnType.STRING:
                try:
                    parsed = non_null.str.to_datetime(strict=False)
                    if parsed.null_count() < len(parsed) * 0.5:
                        col.is_temporal = True
                        col.column_type = ColumnType.DATETIME
                        valid = parsed.drop_nulls()
                        col.temporal_range_start = str(valid.min())
                        col.temporal_range_end = str(valid.max())
                except Exception:
                    pass
        else:
            col.column_type = ColumnType.UNKNOWN

        return col

    @staticmethod
    def _build_profile_prompt(ds: Dataset) -> str:
        """Build a text description of the dataset profile for the LLM."""
        lines = [
            "Dataset: %s" % ds.name,
            "Format: %s | Rows: %d | Columns: %d" % (ds.format.value, ds.row_count, ds.column_count),
            "",
        ]
        for col in ds.columns:
            line = "- **%s** (%s, %s)" % (col.name, col.column_type.value, col.dtype)
            parts = []
            if col.null_count > 0:
                parts.append("nulls: %d (%.1f%%)" % (col.null_count, col.null_percent))
            parts.append("unique: %d" % col.unique_count)

            if col.column_type in (ColumnType.INTEGER, ColumnType.FLOAT):
                if col.min_value is not None:
                    parts.append("range: [%s, %s]" % (col.min_value, col.max_value))
                if col.mean_value is not None:
                    parts.append("mean: %s, std: %s" % (col.mean_value, col.std_value))

            if col.is_temporal:
                parts.append("temporal: %s to %s" % (col.temporal_range_start, col.temporal_range_end))
                if col.temporal_resolution:
                    parts.append("resolution: %s" % col.temporal_resolution)

            if col.top_values and len(col.top_values) <= 10:
                top_str = ", ".join(
                    "%s (%d)" % (v["value"], v["count"]) for v in col.top_values[:5]
                )
                parts.append("top values: %s" % top_str)

            line += " | " + " | ".join(parts)
            lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def _build_column_summary(dataset_name: str, col: DatasetColumn) -> Optional[str]:
        """Build a natural-language summary for a single column.

        Returns None for columns that don't warrant their own memory.
        """
        # Skip columns with very few unique values or low information content
        if col.column_type == ColumnType.BOOLEAN:
            return None
        if col.column_type == ColumnType.UNKNOWN:
            return None

        parts = [
            "In dataset '%s', column '%s' (%s type, dtype=%s):" % (
                dataset_name, col.name, col.column_type.value, col.dtype
            ),
        ]

        if col.null_count > 0:
            parts.append("Contains %d null values (%.1f%% of rows)." % (col.null_count, col.null_percent))

        parts.append("Has %d unique values." % col.unique_count)

        if col.column_type in (ColumnType.INTEGER, ColumnType.FLOAT):
            if col.min_value is not None:
                parts.append(
                    "Range: %s to %s. Mean: %s, Median: %s, Std: %s. "
                    "25th percentile: %s, 75th percentile: %s." % (
                        col.min_value, col.max_value, col.mean_value,
                        col.median_value, col.std_value, col.p25_value, col.p75_value,
                    )
                )

        if col.is_temporal:
            parts.append(
                "Temporal column spanning %s to %s." % (
                    col.temporal_range_start, col.temporal_range_end,
                )
            )
            if col.temporal_resolution:
                parts.append("Data resolution: %s." % col.temporal_resolution)

        if col.column_type == ColumnType.CATEGORICAL and col.top_values:
            top_str = ", ".join(
                "'%s' (%d, %.1f%%)" % (v["value"], v["count"], v["percent"])
                for v in col.top_values[:5]
            )
            parts.append("Top values: %s." % top_str)

        if col.column_type == ColumnType.STRING:
            if col.min_length is not None:
                parts.append(
                    "String lengths: min=%d, max=%d, avg=%.1f." % (
                        col.min_length, col.max_length, col.avg_length,
                    )
                )

        return " ".join(parts)

    @staticmethod
    def _sanitize_sql(sql: str) -> str:
        """Sanitize user-provided SQL to prevent mutation.

        Only SELECT statements are allowed. The query runs against a
        registered table named 'data'.

        Raises:
            ValueError: If the SQL contains disallowed statements.
        """
        stripped = sql.strip().rstrip(";")
        upper = stripped.upper()

        disallowed = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE", "GRANT", "REVOKE"]
        for keyword in disallowed:
            if re.search(r'\b' + keyword + r'\b', upper):
                raise ValueError("SQL mutation not allowed: %s" % keyword)

        if not upper.startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed")

        return stripped

    @staticmethod
    def _quote_identifier(name: str) -> str:
        """Quote a SQL identifier to prevent injection via column/table names.

        Uses DuckDB's double-quote identifier quoting. Any embedded double
        quotes are escaped by doubling them.
        """
        return '"%s"' % name.replace('"', '""')

    @staticmethod
    def _validate_identifier(name: str) -> None:
        """Validate that a string is a safe SQL identifier (column name).

        Raises ValueError if the name contains characters that should not
        appear in a column identifier, even after quoting.
        """
        if not name or not re.match(r'^[\w\s\-\.]+$', name):
            raise ValueError("Invalid column name: %r" % name)

    @staticmethod
    def _build_sql(request: DatasetSliceRequest) -> str:
        """Build a SQL query from a structured slice request.

        All column names are validated and quoted to prevent SQL injection.
        """
        if request.columns:
            for c in request.columns:
                DatasetService._validate_identifier(c)
            cols = ", ".join(DatasetService._quote_identifier(c) for c in request.columns)
        else:
            cols = "*"
        sql = "SELECT %s FROM data" % cols

        if request.filters:
            conditions = []
            for f in request.filters:
                col = f["column"]
                DatasetService._validate_identifier(col)
                quoted_col = DatasetService._quote_identifier(col)
                op = f["op"]
                val = f["value"]
                if op in ("=", "!=", "<", ">", "<=", ">="):
                    if isinstance(val, str):
                        conditions.append("%s %s '%s'" % (quoted_col, op, val.replace("'", "''")))
                    else:
                        conditions.append("%s %s %s" % (quoted_col, op, val))
                elif op == "in":
                    if isinstance(val, list):
                        vals = ", ".join(
                            "'%s'" % str(v).replace("'", "''") if isinstance(v, str) else str(v)
                            for v in val
                        )
                        conditions.append("%s IN (%s)" % (quoted_col, vals))
                elif op == "like":
                    conditions.append("%s LIKE '%s'" % (quoted_col, str(val).replace("'", "''")))
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)

        if request.order_by:
            DatasetService._validate_identifier(request.order_by)
            sql += " ORDER BY %s" % DatasetService._quote_identifier(request.order_by)
            if request.descending:
                sql += " DESC"

        sql += " LIMIT %d OFFSET %d" % (request.limit, request.offset)
        return sql

    @staticmethod
    def _strip_limit(sql: str) -> str:
        """Remove LIMIT and OFFSET clauses from SQL for counting."""
        return re.sub(r'\s+LIMIT\s+\d+(\s+OFFSET\s+\d+)?', '', sql, flags=re.IGNORECASE)


def _detect_resolution(median_diff) -> str:
    """Detect temporal resolution from median time difference."""
    import polars as pl

    if isinstance(median_diff, pl.Duration):
        total_seconds = median_diff.total_seconds()
    else:
        total_seconds = median_diff.total_seconds() if hasattr(median_diff, 'total_seconds') else 0

    if total_seconds < 1:
        return "subsecond"
    elif total_seconds < 60:
        return "second"
    elif total_seconds < 3600:
        return "minute"
    elif total_seconds < 86400:
        return "hour"
    elif total_seconds < 604800:
        return "day"
    elif total_seconds < 2592000:
        return "week"
    elif total_seconds < 31536000:
        return "month"
    else:
        return "year"


class DatasetServicePlugin(DatasetServicePluginBase):
    """Plugin for the default dataset service."""

    PROVIDER_NAME = "default"

    def initialize(self, v: Variables, logger: Logger) -> DatasetService:
        """Build and return the dataset service with all dependencies."""
        storage_backend = get_extension(EXT_STORAGE_BACKEND, v)
        blob_storage = get_extension(EXT_BLOB_STORAGE_SERVICE, v)
        task_service = get_extension(EXT_TASK_SERVICE, v)
        embedding_service = get_extension(EXT_EMBEDDING_SERVICE, v)
        llm_service = get_extension(EXT_LLM_SERVICE, v)

        max_file_size = int(v.environ(
            MEMORYLAYER_DATASET_MAX_FILE_SIZE,
            default=str(DEFAULT_MEMORYLAYER_DATASET_MAX_FILE_SIZE),
        ))

        logger.info(
            "Initializing dataset service (max_file_size=%d)", max_file_size,
        )
        return DatasetService(
            v=v,
            storage_backend=storage_backend,
            blob_storage=blob_storage,
            task_service=task_service,
            embedding_service=embedding_service,
            llm_service=llm_service,
            max_file_size=max_file_size,
            logger=logger,
        )

    def get_dependencies(self, v: Variables):
        """Declare extension point dependencies."""
        return (
            EXT_BLOB_STORAGE_SERVICE,
            EXT_STORAGE_BACKEND,
            EXT_TASK_SERVICE,
            EXT_EMBEDDING_SERVICE,
            EXT_LLM_SERVICE,
        )
