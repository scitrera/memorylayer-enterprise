"""Unit tests for dataset domain models."""
import pytest

from memorylayer_saas.models.dataset import (
    ColumnType,
    Dataset,
    DatasetColumn,
    DatasetFormat,
    DatasetJob,
    DatasetProfilingOptions,
    DatasetSliceRequest,
    DatasetSliceResult,
    DatasetStatus,
)


class TestDatasetModels:
    """Test dataset domain model creation and validation."""

    def test_dataset_defaults(self):
        ds = Dataset(
            id="ds_test123",
            workspace_id="ws_1",
            name="sales_data",
            filename="sales.csv",
            format=DatasetFormat.CSV,
            content_hash="abc123",
            size_bytes=1024,
        )
        assert ds.status == DatasetStatus.PENDING
        assert ds.row_count == 0
        assert ds.column_count == 0
        assert ds.columns == []
        assert ds.memory_ids == []
        assert ds.profile_summary is None
        assert ds.target_context_id == "_default"

    def test_dataset_format_values(self):
        assert DatasetFormat.CSV.value == "csv"
        assert DatasetFormat.TSV.value == "tsv"
        assert DatasetFormat.PARQUET.value == "parquet"
        assert DatasetFormat.JSON_LINES.value == "jsonl"
        assert DatasetFormat.EXCEL.value == "xlsx"

    def test_dataset_status_values(self):
        assert DatasetStatus.PENDING.value == "pending"
        assert DatasetStatus.PROFILING.value == "profiling"
        assert DatasetStatus.SUMMARIZING.value == "summarizing"
        assert DatasetStatus.COMPLETED.value == "completed"
        assert DatasetStatus.FAILED.value == "failed"

    def test_column_type_values(self):
        assert ColumnType.INTEGER.value == "integer"
        assert ColumnType.FLOAT.value == "float"
        assert ColumnType.STRING.value == "string"
        assert ColumnType.BOOLEAN.value == "boolean"
        assert ColumnType.DATETIME.value == "datetime"
        assert ColumnType.CATEGORICAL.value == "categorical"

    def test_dataset_column_numeric(self):
        col = DatasetColumn(
            name="revenue",
            dtype="Float64",
            column_type=ColumnType.FLOAT,
            unique_count=500,
            min_value=0.0,
            max_value=1000000.0,
            mean_value=50000.0,
            median_value=45000.0,
            std_value=25000.0,
            p25_value=30000.0,
            p75_value=70000.0,
        )
        assert col.column_type == ColumnType.FLOAT
        assert col.min_value == 0.0
        assert col.max_value == 1000000.0
        assert col.is_temporal is False

    def test_dataset_column_temporal(self):
        col = DatasetColumn(
            name="timestamp",
            dtype="Datetime",
            column_type=ColumnType.DATETIME,
            unique_count=1000,
            is_temporal=True,
            temporal_resolution="day",
            temporal_range_start="2024-01-01",
            temporal_range_end="2024-12-31",
        )
        assert col.is_temporal is True
        assert col.temporal_resolution == "day"

    def test_dataset_column_categorical(self):
        col = DatasetColumn(
            name="region",
            dtype="Utf8",
            column_type=ColumnType.CATEGORICAL,
            unique_count=5,
            top_values=[
                {"value": "North", "count": 300, "percent": 30.0},
                {"value": "South", "count": 250, "percent": 25.0},
            ],
        )
        assert col.column_type == ColumnType.CATEGORICAL
        assert len(col.top_values) == 2

    def test_profiling_options_defaults(self):
        opts = DatasetProfilingOptions()
        assert opts.target_context_id == "_default"
        assert opts.importance == 0.5
        assert opts.sample_rows == 1000
        assert opts.histogram_bins == 20
        assert opts.detect_time_series is True
        assert opts.generate_summaries is True

    def test_profiling_options_validation(self):
        with pytest.raises(ValueError):
            DatasetProfilingOptions(importance=1.5)
        with pytest.raises(ValueError):
            DatasetProfilingOptions(importance=-0.1)

    def test_dataset_job_defaults(self):
        job = DatasetJob(
            id="dsjob_test123",
            workspace_id="ws_1",
        )
        assert job.status == "queued"
        assert job.progress_percent == 0
        assert job.datasets_processed == 0
        assert job.total_memories_created == 0
        assert job.errors == []

    def test_slice_request_defaults(self):
        req = DatasetSliceRequest()
        assert req.sql is None
        assert req.columns is None
        assert req.filters is None
        assert req.limit == 100
        assert req.offset == 0
        assert req.descending is False

    def test_slice_request_validation(self):
        with pytest.raises(ValueError):
            DatasetSliceRequest(limit=0)
        with pytest.raises(ValueError):
            DatasetSliceRequest(limit=20000)
        with pytest.raises(ValueError):
            DatasetSliceRequest(offset=-1)

    def test_slice_result(self):
        result = DatasetSliceResult(
            dataset_id="ds_test",
            columns=["a", "b"],
            dtypes=["INTEGER", "VARCHAR"],
            rows=[[1, "x"], [2, "y"]],
            total_matching=100,
            returned_count=2,
            sql_executed="SELECT a, b FROM data LIMIT 2",
        )
        assert result.returned_count == 2
        assert result.total_matching == 100
        assert len(result.rows) == 2

    def test_dataset_with_columns(self):
        cols = [
            DatasetColumn(name="id", dtype="Int64", column_type=ColumnType.INTEGER, unique_count=100),
            DatasetColumn(name="name", dtype="Utf8", column_type=ColumnType.STRING, unique_count=90),
        ]
        ds = Dataset(
            id="ds_test",
            workspace_id="ws_1",
            name="test",
            filename="test.csv",
            format=DatasetFormat.CSV,
            content_hash="abc",
            size_bytes=100,
            columns=cols,
            column_count=2,
            row_count=100,
        )
        assert len(ds.columns) == 2
        assert ds.columns[0].name == "id"
        assert ds.column_count == 2


class TestDatasetServiceHelpers:
    """Test dataset service static helper methods."""

    def test_detect_format_csv(self):
        from memorylayer_saas.services.dataset.dataset_service import DatasetService
        assert DatasetService._detect_format("data.csv") == DatasetFormat.CSV

    def test_detect_format_tsv(self):
        from memorylayer_saas.services.dataset.dataset_service import DatasetService
        assert DatasetService._detect_format("data.tsv") == DatasetFormat.TSV

    def test_detect_format_parquet(self):
        from memorylayer_saas.services.dataset.dataset_service import DatasetService
        assert DatasetService._detect_format("data.parquet") == DatasetFormat.PARQUET
        assert DatasetService._detect_format("data.pq") == DatasetFormat.PARQUET

    def test_detect_format_jsonl(self):
        from memorylayer_saas.services.dataset.dataset_service import DatasetService
        assert DatasetService._detect_format("data.jsonl") == DatasetFormat.JSON_LINES
        assert DatasetService._detect_format("data.ndjson") == DatasetFormat.JSON_LINES

    def test_detect_format_excel(self):
        from memorylayer_saas.services.dataset.dataset_service import DatasetService
        assert DatasetService._detect_format("data.xlsx") == DatasetFormat.EXCEL

    def test_detect_format_unsupported(self):
        from memorylayer_saas.services.dataset.dataset_service import DatasetService
        with pytest.raises(ValueError, match="Unsupported"):
            DatasetService._detect_format("data.xml")

    def test_sanitize_sql_select(self):
        from memorylayer_saas.services.dataset.dataset_service import DatasetService
        sql = DatasetService._sanitize_sql("SELECT * FROM data WHERE x > 5")
        assert sql == "SELECT * FROM data WHERE x > 5"

    def test_sanitize_sql_rejects_mutation(self):
        from memorylayer_saas.services.dataset.dataset_service import DatasetService
        with pytest.raises(ValueError, match="mutation"):
            DatasetService._sanitize_sql("DELETE FROM data")
        with pytest.raises(ValueError, match="mutation"):
            DatasetService._sanitize_sql("DROP TABLE data")
        with pytest.raises(ValueError, match="mutation"):
            DatasetService._sanitize_sql("INSERT INTO data VALUES (1)")

    def test_sanitize_sql_rejects_non_select(self):
        from memorylayer_saas.services.dataset.dataset_service import DatasetService
        with pytest.raises(ValueError, match="SELECT"):
            DatasetService._sanitize_sql("EXPLAIN SELECT * FROM data")

    def test_build_sql_basic(self):
        from memorylayer_saas.services.dataset.dataset_service import DatasetService
        req = DatasetSliceRequest(limit=10, offset=0)
        sql = DatasetService._build_sql(req)
        assert sql == "SELECT * FROM data LIMIT 10 OFFSET 0"

    def test_build_sql_with_columns(self):
        from memorylayer_saas.services.dataset.dataset_service import DatasetService
        req = DatasetSliceRequest(columns=["a", "b"], limit=5, offset=0)
        sql = DatasetService._build_sql(req)
        assert 'SELECT "a", "b" FROM data' in sql

    def test_build_sql_with_filters(self):
        from memorylayer_saas.services.dataset.dataset_service import DatasetService
        req = DatasetSliceRequest(
            filters=[{"column": "age", "op": ">", "value": 18}],
            limit=10,
            offset=0,
        )
        sql = DatasetService._build_sql(req)
        assert 'WHERE "age" > 18' in sql

    def test_build_sql_with_order(self):
        from memorylayer_saas.services.dataset.dataset_service import DatasetService
        req = DatasetSliceRequest(
            order_by="revenue",
            descending=True,
            limit=10,
            offset=0,
        )
        sql = DatasetService._build_sql(req)
        assert 'ORDER BY "revenue" DESC' in sql

    def test_strip_limit(self):
        from memorylayer_saas.services.dataset.dataset_service import DatasetService
        sql = "SELECT * FROM data WHERE x > 5 LIMIT 10 OFFSET 20"
        stripped = DatasetService._strip_limit(sql)
        assert "LIMIT" not in stripped
        assert "OFFSET" not in stripped
        assert "WHERE x > 5" in stripped
