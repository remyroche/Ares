"""Data Format Converter Component
Handles conversion between different data formats, particularly focusing on Parquet operations.
Extracted from step01_5_data_converter.py
"""
import contextlib
import os
from datetime import UTC, datetime
from typing import Any, Optional

import pandas as pd

# Safe import of pyarrow with fallback
try:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    pa = None
    ds = None
    pq = None

from src.core.decorators import traced, validates
from src.utils.logger import system_logger
from src.utils.file_operations import ensure_directory, safe_json_dump, safe_json_load


class DataFormatConverter:
    """Handles conversion between different data formats with focus on Parquet operations.
    
    This class provides functionality for:
    - Writing and reading partitioned Parquet datasets
    - Schema enforcement and validation
    - Metadata management
    - Efficient scanning with filters
    """
    
    def __init__(self, logger=None) -> None:
        self.logger = logger or system_logger.getChild("DataFormatConverter")
        try:
            self.default_batch_size = int(
                os.environ.get("ARES_SCAN_BATCH_SIZE", "262144")
            )
        except Exception:
            self.default_batch_size = 262144
        # Arrow memory pool proxy for visibility if available
        self._proxy_pool = None
        if PYARROW_AVAILABLE:
            try:
                self._memory_pool = pa.default_memory_pool()
                self._proxy_pool = pa.proxy_memory_pool(self._memory_pool)
                pa.set_memory_pool(self._proxy_pool)
            except Exception:
                self._proxy_pool = None

    def _ensure_pyarrow(self) -> None:
        """Ensure pyarrow is available for operations."""
        if not PYARROW_AVAILABLE:
            msg = "pyarrow is required for DataFormatConverter operations"
            raise ImportError(msg)

    @validates(mode="warn", arg_index=1)
    @traced(
        "DataFormatConverter.enforce_schema", log_args=False, log_result_len_only=True
    )
    def enforce_schema(self, df: pd.DataFrame, schema_name: str) -> pd.DataFrame:
        """Enforce a specific schema on the DataFrame.
        
        Args:
            df: DataFrame to enforce schema on
            schema_name: Name of the schema to enforce (klines, aggtrades, futures, split, unified)
            
        Returns:
            DataFrame with enforced schema
        """
        if df is None or df.empty:
            return df

        conversions: dict[str, str] = {}
        optional_columns: dict[str, str] = {}
        
        if schema_name == "klines":
            conversions = {
                "timestamp": "int64",
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
                "volume": "float64",
            }
        elif schema_name == "aggtrades":
            conversions = {
                "timestamp": "int64",
                "price": "float64",
                "quantity": "float64",
                "is_buyer_maker": "bool",
                "agg_trade_id": "int64",
            }
        elif schema_name == "futures":
            conversions = {
                "timestamp": "int64",
                "fundingRate": "float64",
            }
        elif schema_name == "split":
            if "timestamp" in df.columns:
                conversions["timestamp"] = "int64"
            if "label" in df.columns:
                conversions["label"] = "int64"
        elif schema_name == "unified":
            conversions = {
                "timestamp": "int64",
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
                "volume": "float64",
                "exchange": "string",
                "symbol": "string",
                "timeframe": "string",
                "year": "int16",
                "month": "int8",
                "day": "int8",
            }
            optional_columns = {
                "trade_volume": "float64",
                "trade_count": "int64",
                "avg_price": "float64",
                "min_price": "float64",
                "max_price": "float64",
                "volume_ratio": "float64",
                "funding_rate": "float64",
            }

        for col, dtype in optional_columns.items():
            if col in df.columns:
                conversions[col] = dtype

        # Handle timestamp conversion
        if "timestamp" in df.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                    df.loc[:, "timestamp"] = (
                        pd.to_datetime(df["timestamp"], utc=True).astype("int64")
                        // 10**6
                    ).astype("int64")
                else:
                    ts_numeric = pd.to_numeric(df["timestamp"], errors="coerce")
                    if pd.notna(ts_numeric.max()) and float(ts_numeric.max()) > 1e14:
                        df.loc[:, "timestamp"] = (ts_numeric // 10**6).astype("int64")
                    else:
                        df.loc[:, "timestamp"] = ts_numeric.astype("int64")
            except Exception:
                pass

        # Apply conversions
        for col, dtype in conversions.items():
            if col in df.columns:
                try:
                    if dtype == "bool":
                        df.loc[:, col] = df[col].astype("boolean").astype(bool)
                    elif dtype == "string":
                        df.loc[:, col] = df[col].astype("string")
                    else:
                        df.loc[:, col] = pd.to_numeric(df[col], errors="coerce").astype(
                            dtype
                        )
                except Exception:
                    if self.logger:
                        self.logger.debug(
                            f"Schema conversion skipped for column: {col}"
                        )
        return df

    def write_partitioned_dataset(
        self,
        df: pd.DataFrame,
        base_dir: str,
        partition_cols: list[str],
        schema_name: str | None,
        compression: str = "snappy",
        use_dictionary: bool | dict[str, bool] = True,
        min_rows_per_group: int = 50000,
        max_rows_per_file: int = 5_000_000,
        use_threads: bool = True,
        update_manifest: bool = True,
        metadata: dict[str, Any] | None = None,
        auto_add_date_columns: bool = True,
    ) -> None:
        """Write DataFrame as partitioned Parquet dataset.
        
        Args:
            df: DataFrame to write
            base_dir: Base directory for the dataset
            partition_cols: Columns to partition by
            schema_name: Schema to enforce before writing
            compression: Compression algorithm
            use_dictionary: Whether to use dictionary encoding
            min_rows_per_group: Minimum rows per row group
            max_rows_per_file: Maximum rows per file
            use_threads: Whether to use multiple threads
            update_manifest: Whether to update the manifest file
            metadata: Additional metadata to store
            auto_add_date_columns: Whether to automatically add year/month/day columns
        """
        self._ensure_pyarrow()
        ensure_directory(base_dir)

        if min_rows_per_group >= max_rows_per_file:
            min_rows_per_group = max(1000, max_rows_per_file // 10)
            if self.logger:
                self.logger.warning(
                    f"Adjusted min_rows_per_group to {min_rows_per_group} to be < max_rows_per_file ({max_rows_per_file})",
                )

        if schema_name:
            df = self.enforce_schema(df, schema_name)

        # Log dataset info
        try:
            nrows = len(df)
            ncols = len(df.columns)
            cols_preview = ",".join(list(map(str, df.columns[:12])))
            if self.logger:
                self.logger.info(
                    f"Preparing to write dataset: rows={nrows}, cols={ncols}, cols[0..11]=[{cols_preview}] -> {base_dir}",
                )
            if "timestamp" in df.columns:
                ts = pd.to_datetime(
                    df["timestamp"], unit="ms", utc=True, errors="coerce"
                )
                if self.logger:
                    self.logger.info(
                        f"Timestamp coverage: {ts.min()} → {ts.max()} (UTC)"
                    )
        except Exception:
            pass

        # Add date columns if needed
        if "timestamp" in df.columns and auto_add_date_columns:
            ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            if "year" not in df.columns:
                df["year"] = ts.dt.year.astype("int16")
            if "month" not in df.columns:
                df["month"] = ts.dt.month.astype("int8")
            if "day" not in df.columns:
                df["day"] = ts.dt.day.astype("int8")

        table = pa.Table.from_pandas(df, preserve_index=False)

        # Add metadata
        if metadata:
            try:
                meta = {
                    str(k): (str(v) if v is not None else "")
                    for k, v in metadata.items()
                }
                schema_with_meta = table.schema.with_metadata(meta)
                table = table.cast(schema_with_meta)
            except Exception:
                pass

        # Setup partitioning
        partitioning = None
        try:
            if partition_cols:
                fields = []
                for col in partition_cols:
                    if col in df.columns:
                        try:
                            dtype = pa.array(df[col]).type
                        except Exception:
                            dtype = pa.string()
                        fields.append(pa.field(col, dtype))
                    else:
                        fields.append(pa.field(col, pa.string()))
                partition_schema = pa.schema(fields)
                partitioning = ds.partitioning(partition_schema, flavor="hive")
        except Exception:
            partitioning = None

        if self.logger:
            self.logger.info(
                f"Writing partitioned dataset to {base_dir} with compression={compression}"
            )

        # Count files before
        try:
            before_count = 0
            for r, _d, files in os.walk(base_dir):
                before_count += sum(1 for f in files if f.endswith(".parquet"))
        except Exception:
            before_count = None

        def _file_visitor(written_file: Any) -> None:
            try:
                path = getattr(written_file, "path", None) or str(written_file)
            except Exception:
                path = str(written_file)
            if self.logger:
                self.logger.info(f"🆕 Wrote partitioned parquet file: {path}")

        write_args: dict[str, Any] = {
            "base_dir": base_dir,
            "format": "parquet",
            "basename_template": "part-{i}.parquet",
            "file_visitor": _file_visitor,
            "existing_data_behavior": "overwrite_or_ignore",
            "max_rows_per_file": max_rows_per_file,
            "min_rows_per_group": min_rows_per_group,
            "max_rows_per_group": min(max_rows_per_file, 1024 * 1024),
        }
        if partitioning is not None:
            write_args["partitioning"] = partitioning

        ds.write_dataset(table, **write_args)

        # Count files after
        try:
            after_count = 0
            total_bytes = 0
            for r, _d, files in os.walk(base_dir):
                for f in files:
                    if f.endswith(".parquet"):
                        after_count += 1
                        with contextlib.suppress(Exception):
                            total_bytes += os.path.getsize(os.path.join(r, f))
            if self.logger:
                self.logger.info(
                    f"Partitioned write complete: files_before={before_count}, files_after={after_count}, size≈{total_bytes} bytes",
                )
        except Exception:
            pass

        if update_manifest:
            with contextlib.suppress(Exception):
                self.update_manifest(base_dir)

    def scan_dataset(
        self,
        base_dir: str,
        filters: list | None = None,
        columns: list[str] | None = None,
        batch_size: int | None = None,
        to_pandas: bool = True,
        use_threads: bool = True,
        ignore_hidden_temp: bool = True,
    ) -> pd.DataFrame | Any:
        """Scan a partitioned Parquet dataset with optional filters.
        
        Args:
            base_dir: Base directory of the dataset
            filters: Filter expressions as list of tuples
            columns: Columns to read
            batch_size: Batch size for reading
            to_pandas: Whether to convert to pandas DataFrame
            use_threads: Whether to use multiple threads
            ignore_hidden_temp: Whether to ignore hidden/temp files
            
        Returns:
            DataFrame or Arrow Table
        """
        self._ensure_pyarrow()
        if batch_size is None:
            batch_size = self.default_batch_size

        if columns is not None and len(columns) == 0:
            columns = None

        before_bytes = None
        if self._proxy_pool is not None:
            with contextlib.suppress(Exception):
                before_bytes = self._proxy_pool.bytes_allocated()

        # Handle hidden/temp files
        try:
            if ignore_hidden_temp and os.path.isdir(base_dir):
                file_paths: list[str] = []
                for root, _dirs, files in os.walk(base_dir):
                    for name in files:
                        if not name.endswith(".parquet"):
                            continue
                        if name.startswith((".", "_")) or name.endswith(
                            (".tmp", ".partial")
                        ):
                            continue
                        file_paths.append(os.path.join(root, name))
                dataset = (
                    ds.dataset(file_paths, format="parquet")
                    if file_paths
                    else ds.dataset(base_dir, format="parquet")
                )
            else:
                dataset = ds.dataset(base_dir, format="parquet")
        except Exception:
            dataset = ds.dataset(base_dir, format="parquet")

        expr = self._build_filter_expression(filters)
        try:
            table = dataset.to_table(columns=columns, filter=expr)
        except Exception:
            table = dataset.to_table(columns=columns, filter=expr)

        if to_pandas:
            df = table.to_pandas(types_mapper=pd.ArrowDtype)
            with contextlib.suppress(Exception):
                nbytes = getattr(table, "nbytes", None) or 0
                if self.logger:
                    self.logger.info(
                        f"Scan read: rows={len(df)}, cols={len(df.columns)}, bytes≈{nbytes}, filters={bool(filters)}, columns_pruned={columns is not None}",
                    )
            return df

        after_bytes = None
        if self._proxy_pool is not None:
            with contextlib.suppress(Exception):
                after_bytes = self._proxy_pool.bytes_allocated()
        if self.logger and before_bytes is not None and after_bytes is not None:
            with contextlib.suppress(Exception):
                self.logger.debug(
                    f"Arrow memory delta: {after_bytes - before_bytes} bytes (alloc={after_bytes})"
                )
        return table

    def _build_filter_expression(
        self, filters: list | None
    ) -> Optional["ds.Expression"]:
        """Build Arrow filter expression from list of filter tuples."""
        if not filters:
            return None
        try:
            expressions: list[ds.Expression] = []
            for f in filters:
                if isinstance(f, (list, tuple)) and len(f) == 3:
                    field, op, value = f
                    if op == "==":
                        expressions.append(ds.field(field) == value)
                    elif op == "!=":
                        expressions.append(ds.field(field) != value)
                    elif op == ">":
                        expressions.append(ds.field(field) > value)
                    elif op == ">=":
                        expressions.append(ds.field(field) >= value)
                    elif op == "<":
                        expressions.append(ds.field(field) < value)
                    elif op == "<=":
                        expressions.append(ds.field(field) <= value)
            if expressions:
                expr = expressions[0]
                for sub in expressions[1:]:
                    expr = expr & sub
                return expr
        except Exception:
            return None
        return None

    def write_flat_parquet(
        self,
        df: pd.DataFrame,
        file_path: str,
        schema_name: str | None = None,
        compression: str = "snappy",
        use_dictionary: bool | dict[str, bool] = True,
        row_group_size: int = 128_000,
        write_statistics: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write DataFrame as a single Parquet file.
        
        Args:
            df: DataFrame to write
            file_path: Path to write the file
            schema_name: Schema to enforce before writing
            compression: Compression algorithm
            use_dictionary: Whether to use dictionary encoding
            row_group_size: Size of row groups
            write_statistics: Whether to write statistics
            metadata: Additional metadata
        """
        self._ensure_pyarrow()
        ensure_directory(os.path.dirname(file_path))
        if schema_name:
            df = self.enforce_schema(df, schema_name)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if metadata:
            with contextlib.suppress(Exception):
                meta = {
                    str(k): (str(v) if v is not None else "")
                    for k, v in metadata.items()
                }
                table = table.cast(table.schema.with_metadata(meta))
        pq.write_table(
            table,
            file_path,
            compression=compression,
            row_group_size=row_group_size,
            write_statistics=write_statistics,
        )

    def update_manifest(self, base_dir: str, ts_column: str = "timestamp") -> None:
        """Update manifest file with dataset statistics.
        
        Args:
            base_dir: Base directory of the dataset
            ts_column: Name of the timestamp column
        """
        try:
            if not os.path.exists(base_dir):
                return
            manifest_path = os.path.join(base_dir, "_manifest.json")
            manifest: dict[str, Any] = {
                "updated_at": datetime.now(UTC).isoformat(),
                "base_dir": base_dir,
                "timestamp_column": ts_column,
            }
            file_count = 0
            latest_ts: int | None = None
            for root, _dirs, files in os.walk(base_dir):
                for file in files:
                    if not file.endswith(".parquet"):
                        continue
                    file_count += 1
                    file_path = os.path.join(root, file)
                    with contextlib.suppress(Exception):
                        pf = pq.ParquetFile(file_path)
                        # Attempt to read first row group stats
                        md = pf.metadata
                        for rg_idx in range(md.num_row_groups):
                            rg = md.row_group(rg_idx)
                            for col_idx in range(rg.num_columns):
                                col = rg.column(col_idx)
                                if col.path_in_schema == ts_column and hasattr(
                                    col, "statistics"
                                ):
                                    st = col.statistics
                                    if st and st.max is not None:
                                        candidate = int(st.max)
                                        latest_ts = (
                                            candidate
                                            if latest_ts is None
                                            else max(latest_ts, candidate)
                                        )
            manifest["file_count"] = file_count
            manifest["latest_timestamp"] = latest_ts
            safe_json_dump(manifest, manifest_path, indent=2, default=str)
            if self.logger:
                self.logger.info(f"Updated manifest: {manifest_path}")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to update manifest: {e}")

    def get_latest_timestamp(
        self, base_dir: str, ts_column: str = "timestamp"
    ) -> int | None:
        """Get the latest timestamp from manifest or scan dataset.
        
        Args:
            base_dir: Base directory of the dataset
            ts_column: Name of the timestamp column
            
        Returns:
            Latest timestamp in milliseconds or None
        """
        try:
            manifest_path = os.path.join(base_dir, "_manifest.json")
            if os.path.exists(manifest_path):
                manifest = safe_json_load(manifest_path)
                return manifest.get("latest_timestamp")
        except Exception:
            return None
        return None