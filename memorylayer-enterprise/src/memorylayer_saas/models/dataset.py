"""Dataset domain models for tabular data ingestion and profiling."""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class DatasetStatus(str, Enum):
    """Dataset processing status."""
    PENDING = "pending"
    PROFILING = "profiling"
    SUMMARIZING = "summarizing"
    COMPLETED = "completed"
    FAILED = "failed"


class DatasetFormat(str, Enum):
    """Supported dataset file formats."""
    CSV = "csv"
    TSV = "tsv"
    PARQUET = "parquet"
    JSON_LINES = "jsonl"
    EXCEL = "xlsx"


class ColumnType(str, Enum):
    """Detected column data type."""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    UNKNOWN = "unknown"


class DatasetProfilingOptions(BaseModel):
    """Options controlling dataset profiling and memory extraction."""
    # target_context_id sets each extracted memory's context_id. RESERVED /
    # unused as a retrieval filter today — see MemoryModel.context_id.
    target_context_id: str = Field("_default", description="Target context for extracted memories")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Default importance for memories")
    sample_rows: int = Field(1000, ge=1, description="Max rows to include in LLM summary sample")
    histogram_bins: int = Field(20, ge=2, le=100, description="Number of histogram bins for numeric columns")
    detect_time_series: bool = Field(True, description="Attempt to detect temporal columns")
    generate_summaries: bool = Field(True, description="Generate LLM natural-language summaries")


class DatasetColumn(BaseModel):
    """Column-level schema and statistics from profiling."""
    name: str
    dtype: str = Field(..., description="Original dtype string from the dataframe")
    column_type: ColumnType = ColumnType.UNKNOWN
    nullable: bool = False
    null_count: int = 0
    null_percent: float = 0.0
    unique_count: int = 0

    # Numeric stats
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    median_value: Optional[float] = None
    std_value: Optional[float] = None
    p25_value: Optional[float] = None
    p75_value: Optional[float] = None

    # String stats
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    avg_length: Optional[float] = None

    # Categorical stats
    top_values: Optional[list[dict[str, Any]]] = Field(
        None, description="Top values with counts: [{value, count, percent}]"
    )

    # Time series detection
    is_temporal: bool = False
    temporal_resolution: Optional[str] = Field(
        None, description="Detected resolution: second, minute, hour, day, week, month, year"
    )
    temporal_range_start: Optional[str] = None
    temporal_range_end: Optional[str] = None

    # Distribution
    histogram: Optional[dict[str, Any]] = Field(
        None, description="Histogram data: {bins, counts, edges}"
    )


class Dataset(BaseModel):
    """Dataset domain model."""
    model_config = {"from_attributes": True}

    id: str
    workspace_id: str
    tenant_id: str = "default_tenant"
    name: str = Field(..., description="User-provided or filename-derived name")
    filename: str
    format: DatasetFormat
    content_hash: str
    size_bytes: int
    status: DatasetStatus = DatasetStatus.PENDING
    storage_path: Optional[str] = Field(None, description="Blob path to Parquet file")
    original_storage_path: Optional[str] = Field(
        None, description="Blob path to original upload (if converted)"
    )
    target_context_id: str = "_default"
    profiling_options: DatasetProfilingOptions = Field(default_factory=DatasetProfilingOptions)

    # Schema (populated during profiling)
    row_count: int = 0
    column_count: int = 0
    columns: list[DatasetColumn] = Field(default_factory=list)

    # Processing results
    memory_ids: list[str] = Field(default_factory=list)
    profile_summary: Optional[str] = Field(None, description="LLM-generated natural-language summary")

    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    profiling_started_at: Optional[datetime] = None
    profiling_completed_at: Optional[datetime] = None


class DatasetJob(BaseModel):
    """Dataset processing job domain model."""
    model_config = {"from_attributes": True}

    id: str
    workspace_id: str
    dataset_ids: list[str] = Field(default_factory=list)
    status: str = "queued"
    progress_percent: int = 0
    datasets_processed: int = 0
    total_memories_created: int = 0
    errors: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class DatasetSliceRequest(BaseModel):
    """Request for a slice of data from the dataset."""
    sql: Optional[str] = Field(
        None, description="Raw SQL query (SELECT only, table alias is 'data')"
    )
    columns: Optional[list[str]] = Field(None, description="Columns to select")
    filters: Optional[list[dict[str, Any]]] = Field(
        None, description="Filters: [{column, op, value}] where op is =, !=, <, >, <=, >=, in, like"
    )
    order_by: Optional[str] = Field(None, description="Column to order by")
    descending: bool = False
    limit: int = Field(100, ge=1, le=10000, description="Max rows to return")
    offset: int = Field(0, ge=0, description="Row offset for pagination")


class DatasetSliceResult(BaseModel):
    """Result of a dataset slice query."""
    dataset_id: str
    columns: list[str]
    dtypes: list[str] = Field(default_factory=list)
    rows: list[list[Any]]
    total_matching: int
    returned_count: int
    sql_executed: Optional[str] = None
