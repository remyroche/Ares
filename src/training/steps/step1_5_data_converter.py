# src/training/steps/step1_5_data_converter.py

import asyncio
import contextlib
import glob
import os
import sys
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Import centralized decorators for secure data processing
try:
    from src.utils.centralized_decorators import (
        handle_errors,
        handle_file_operations,
        secure_klines_download_operation,
        validate_klines_data_quality,
        secure_data_processing,
        prevent_data_leakage,
        resource_monitor,
        memory_efficient,
        quality_gate,
        circuit_breaker_protection,
    )
except ImportError:
    # Fallback decorators if centralized decorators are not available
    def handle_errors(**kwargs):
        def decorator(func):
            return func
        return decorator

    def handle_file_operations(func):
        return func

    def secure_klines_download_operation(func):
        return func

    def validate_klines_data_quality(func):
        return func

    def secure_data_processing(func):
        return func

    def prevent_data_leakage(func):
        return func

    def resource_monitor(func):
        return func

    def memory_efficient(func):
        return func

    def quality_gate(func):
        return func

    def circuit_breaker_protection(func):
        return func

# Try to import pyarrow components
try:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    PYARROW_AVAILABLE = True
except ImportError:
    pa = None
    pq = None
    ds = None
    PYARROW_AVAILABLE = False

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Handle imports with fallback
CONFIG = None
handle_errors = None
setup_logging = None
system_logger = None

try:
    from src.config import CONFIG
    from src.utils.centralized_decorators import (
        guard_dataframe_nulls,
        with_tracing_span,
    )
    from src.utils.error_handler import (
        handle_errors,
    )
    from src.utils.logger import setup_logging, system_logger
    from src.utils.data_quality_decorators import (
        handle_data_processing_errors,
        validate_klines_data,
        validate_aggtrades_data,
        validate_futures_data,
        format_klines_data,
        format_aggtrades_data,
        format_futures_data,
        log_step_metrics,
    )
    from src.utils.warning_symbols import (
        connection_error,
        critical,
        error,
        execution_error,
        failed,
        initialization_error,
        invalid,
        missing,
        problem,
        timeout,
        validation_error,
        warning,
    )
except ImportError:
    # Fallback configuration
    CONFIG = {
        "SYMBOL": "ETHUSDT",
        "INTERVAL": "1m",
        "LOOKBACK_YEARS": 2,
    }

    # Create fallback functions
    def handle_errors(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def handle_data_processing_errors(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def validate_klines_data(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def validate_aggtrades_data(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def validate_futures_data(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def format_klines_data(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def format_aggtrades_data(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def format_futures_data(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def log_step_metrics(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def guard_dataframe_nulls(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def with_tracing_span(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def setup_logging(config=None):
        try:
            from src.utils.logger import (
                setup_logging as _setup_logging,
            )
            from src.utils.logger import (
                system_logger as _syslog,
            )

            if config is not None:
                _setup_logging(config)
            else:
                _setup_logging()
            return _syslog
        except Exception:
            import logging

            logging.basicConfig(level=logging.INFO)
            return logging.getLogger("Step1.5Fallback")

    system_logger = setup_logging()

    # Define fallback formatting helpers
    def _fmt(prefix, msg):
        try:
            return f"{prefix} {msg}"
        except Exception:
            return str(msg)

    def error(msg):
        return _fmt("ERROR:", msg)

    def warning(msg):
        return _fmt("WARN:", msg)

    def critical(msg):
        return _fmt("CRITICAL:", msg)

    def problem(msg):
        return _fmt("PROBLEM:", msg)

    def failed(msg):
        return _fmt("FAILED:", msg)

    def invalid(msg):
        return _fmt("INVALID:", msg)

    def missing(msg):
        return _fmt("MISSING:", msg)

    def timeout(msg):
        return _fmt("TIMEOUT:", msg)

    def connection_error(msg):
        return _fmt("CONNECTION_ERROR:", msg)

    def validation_error(msg):
        return _fmt("VALIDATION_ERROR:", msg)

    def initialization_error(msg):
        return _fmt("INIT_ERROR:", msg)

    def execution_error(msg):
        return _fmt("EXEC_ERROR:", msg)


class ParquetDatasetManager:
    """High-performance Parquet dataset reader/writer with partitioning, schema control,
    projection, filtering, and caching.

    This class centralizes Parquet I/O best practices to reduce duplicated logic and
    improve memory and CPU efficiency across the pipeline.
    """

    def __init__(self, logger=None) -> None:
        self.logger = logger or system_logger.getChild("ParquetDatasetManager")

        # Default batch size from env; fall back to 256k rows
        try:
            self.default_batch_size = int(
                os.environ.get("ARES_SCAN_BATCH_SIZE", "262144"),
            )
        except Exception:
            self.default_batch_size = 262144

        # In blank/dev mode, prefer smaller batches for stability unless overridden
        try:
            blank_mode = os.environ.get("ARES_BLANK_MODE", "").lower() in (
                "1",
                "true",
                "yes",
            )
            if blank_mode and os.environ.get("ARES_SCAN_BATCH_SIZE") is None:
                self.default_batch_size = 131072
        except Exception:
            pass

        # Optional: set Arrow thread count from env for stability in dev
        try:
            if pa is not None:
                env_threads = os.environ.get("ARROW_NUM_THREADS")
                if env_threads is not None:
                    try:
                        threads = int(env_threads)
                        if hasattr(pa, "set_cpu_count"):
                            pa.set_cpu_count(threads)
                    except Exception:
                        pass
        except Exception:
            pass

        # Optional Arrow memory pool proxy for monitoring
        try:
            if pa is not None:
                self._memory_pool = pa.default_memory_pool()
                self._proxy_pool = pa.proxy_memory_pool(self._memory_pool)
                pa.set_memory_pool(self._proxy_pool)
            else:
                self._memory_pool = None
                self._proxy_pool = None
        except Exception:
            self._memory_pool = None
            self._proxy_pool = None

    def _ensure_pyarrow(self) -> None:
        """Ensure pyarrow is available for operations."""
        if not PYARROW_AVAILABLE:
            msg = "pyarrow is required for ParquetDatasetManager operations"
            raise ImportError(
                msg,
            )

    @guard_dataframe_nulls(mode="warn", arg_index=1)
    @with_tracing_span(
        "ParquetDatasetManager.enforce_schema", log_args=False, log_result_len_only=True,
    )
    def enforce_schema(self, df: pd.DataFrame, schema_name: str) -> pd.DataFrame:
        """Standardize dtypes for known schemas to avoid costly inference and mismatches.

        Supported schema_name values: 'klines', 'aggtrades', 'futures', 'split', 'unified'.
        """
        if df is None or df.empty:
            return df

        conversions: dict[str, str] = {}
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
            # General split schema: ensure timestamp is present if available, label numeric
            if "timestamp" in df.columns:
                conversions["timestamp"] = "int64"
            if "label" in df.columns:
                conversions["label"] = "int64"
        elif schema_name == "unified":
            # Unified schema: comprehensive type enforcement
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
            # Add optional columns if they exist
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

        # Normalize timestamp to milliseconds since epoch if present
        if "timestamp" in df.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                    df.loc[:, "timestamp"] = (
                        pd.to_datetime(df["timestamp"], utc=True).astype("int64")
                        // 10**6
                    ).astype("int64")
                else:
                    # If numeric but likely in nanoseconds, downscale to ms
                    ts_numeric = pd.to_numeric(df["timestamp"], errors="coerce")
                    if pd.notna(ts_numeric.max()) and float(ts_numeric.max()) > 1e14:
                        df.loc[:, "timestamp"] = (ts_numeric // 10**6).astype("int64")
                    else:
                        df.loc[:, "timestamp"] = ts_numeric.astype("int64")
            except Exception:
                pass

        for col, dtype in conversions.items():
            if col in df.columns:
                try:
                    if dtype == "bool":
                        df.loc[:, col] = df[col].astype("boolean").astype(bool)
                    elif dtype == "string":
                        df.loc[:, col] = df[col].astype("string")
                    else:
                        df.loc[:, col] = pd.to_numeric(df[col], errors="coerce").astype(
                            dtype,
                        )
                except Exception:
                    # Leave column as-is if conversion fails; log at debug level
                    if self.logger:
                        self.logger.debug(
                            f"Schema conversion skipped for column: {col}",
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
        """Write a DataFrame into a partitioned Parquet dataset with hive-style layout.

        Partitions should typically include ['exchange','symbol','timeframe','year','month','day']
        where applicable.
        """
        self._ensure_pyarrow()
        os.makedirs(base_dir, exist_ok=True)

        # Ensure min_rows_per_group is less than max_rows_per_file
        if min_rows_per_group >= max_rows_per_file:
            min_rows_per_group = max(1000, max_rows_per_file // 10)
            if self.logger:
                self.logger.warning(
                    f"Adjusted min_rows_per_group to {min_rows_per_group} to be less than max_rows_per_file ({max_rows_per_file})",
                )

        # Enforce schema if provided; otherwise proceed with inferred schema
        if schema_name:
            df = self.enforce_schema(df, schema_name)

        # Verbose logging about the dataframe being written
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
                    df["timestamp"], unit="ms", utc=True, errors="coerce",
                )
                ts_min = ts.min()
                ts_max = ts.max()
                if self.logger:
                    self.logger.info(f"Timestamp coverage: {ts_min} → {ts_max} (UTC)")
        except Exception:
            pass

        # Derive date components if timestamp exists and auto_add_date_columns is enabled
        if "timestamp" in df.columns and auto_add_date_columns:
            # Expect timestamp in ms since epoch
            ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            if "year" not in df.columns:
                df["year"] = ts.dt.year.astype("int16")
            if "month" not in df.columns:
                df["month"] = ts.dt.month.astype("int8")
            if "day" not in df.columns:
                df["day"] = ts.dt.day.astype("int8")

        table = pa.Table.from_pandas(df, preserve_index=False)

        # Attach key_value_metadata for governance if provided
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

        # Create partitioning with specific columns to avoid too many partitions
        try:
            if partition_cols:
                # Build partition schema from df dtypes, defaulting to string
                fields = []
                for col in partition_cols:
                    if col in df.columns:
                        # Map pandas dtype to pyarrow
                        try:
                            dtype = pa.array(df[col]).type
                        except Exception:
                            dtype = pa.string()
                        fields.append(pa.field(col, dtype))
                    else:
                        fields.append(pa.field(col, pa.string()))
                partition_schema = pa.schema(fields)
                partitioning = ds.partitioning(partition_schema, flavor="hive")
            else:
                partitioning = None
        except Exception:
            # Fallback: no partitioning
            partitioning = None

        if self.logger:
            self.logger.info(
                f"Writing partitioned dataset to {base_dir} with compression={compression}",
            )

        # Count existing files before write for delta logging
        try:
            before_count = 0
            for r, _d, files in os.walk(base_dir):
                before_count += sum(1 for f in files if f.endswith(".parquet"))
        except Exception:
            before_count = None

        # File visitor to log each file materialized by the engine
        def _file_visitor(written_file: Any) -> None:
            try:
                path = getattr(written_file, "path", None) or str(written_file)
            except Exception:
                path = str(written_file)
            if self.logger:
                self.logger.info(f"🆕 Wrote partitioned parquet file: {path}")
            with contextlib.suppress(Exception):
                pass

        # Prepare write_dataset arguments with minimal required parameters
        write_args = {
            "base_dir": base_dir,
            "format": "parquet",
            "basename_template": "part-{i}.parquet",
            "file_visitor": _file_visitor,
            "existing_data_behavior": "overwrite_or_ignore",
            "max_rows_per_file": max_rows_per_file,
            "min_rows_per_group": min_rows_per_group,
            "max_rows_per_group": min(
                max_rows_per_file, 1024 * 1024,
            ),  # Ensure max_rows_per_group <= max_rows_per_file
        }

        # Add partitioning only if available
        if partitioning is not None:
            write_args["partitioning"] = partitioning

        # Write the dataset
        ds.write_dataset(table, **write_args)

        # Delta logging after write
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
            try:
                self.update_manifest(base_dir)
            except Exception:
                if self.logger:
                    self.logger.debug("Manifest update skipped")

    def scan_dataset(
        self,
        base_dir: str,
        filters: list | None = None,
        columns: list[str] | None = None,
        batch_size: int | None = None,
        to_pandas: bool = True,
        use_threads: bool = True,
        ignore_hidden_temp: bool = True,
    ) -> pd.DataFrame | pa.Table:
        """Scan a Parquet dataset with projection and predicate pushdown.

        Returns a pandas DataFrame by default; can return an Arrow table if to_pandas=False.
        """
        self._ensure_pyarrow()
        if batch_size is None:
            batch_size = self.default_batch_size

        # Early column pruning: ensure columns is a list or None
        if columns is not None and len(columns) == 0:
            columns = None

        before_bytes = None
        if self._proxy_pool is not None:
            try:
                before_bytes = self._proxy_pool.bytes_allocated()
            except Exception:
                before_bytes = None

        # Build dataset while ignoring hidden/temporary files when requested
        dataset: ds.Dataset
        try:
            if ignore_hidden_temp and os.path.isdir(base_dir):
                file_paths: list[str] = []
                for root, _dirs, files in os.walk(base_dir):
                    for name in files:
                        if not name.endswith(".parquet"):
                            continue
                        if (
                            name.startswith((".", "_")) or name.endswith((".tmp", ".partial"))
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
            # Fallback to direct dataset construction
            dataset = ds.dataset(base_dir, format="parquet")

        # Apply filters via dataset.to_table to ensure pushdown; retry once on transient errors
        expr = self._build_filter_expression(filters)
        try:
            table = dataset.to_table(columns=columns, filter=expr)
        except Exception:
            # Retry without threading
            table = dataset.to_table(columns=columns, filter=expr)

        if to_pandas:
            df = table.to_pandas(types_mapper=pd.ArrowDtype)
            # Log I/O metrics
            try:
                nbytes = getattr(table, "nbytes", None) or 0
                if self.logger:
                    self.logger.info(
                        f"Scan read: rows={len(df)}, cols={len(df.columns)}, bytes≈{nbytes}, batch_size={batch_size}, filters={bool(filters)}, columns_pruned={columns is not None}",
                    )
            except Exception:
                pass
            return df

        after_bytes = None
        if self._proxy_pool is not None:
            try:
                after_bytes = self._proxy_pool.bytes_allocated()
            except Exception:
                after_bytes = None
        if self.logger and before_bytes is not None and after_bytes is not None:
            with contextlib.suppress(Exception):
                self.logger.debug(
                    f"Arrow memory delta: {after_bytes - before_bytes} bytes (alloc={after_bytes})",
                )
        return table

    def _build_filter_expression(
        self, filters: list | None,
    ) -> ds.Expression | None:
        """Build Arrow filter expression from filter list."""
        if not filters:
            return None

        try:
            # Simple filter building - can be extended for more complex filters
            expressions = []
            for filter_item in filters:
                if isinstance(filter_item, list | tuple) and len(filter_item) == 3:
                    field, op, value = filter_item
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
                return (
                    expressions[0]
                    if len(expressions) == 1
                    else expressions[0] & expressions[1]
                    if len(expressions) == 2
                    else expressions[0] & expressions[1] & expressions[2]
                )
        except Exception:
            pass
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
        """Write a DataFrame to a single Parquet file."""
        self._ensure_pyarrow()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        if schema_name:
            df = self.enforce_schema(df, schema_name)

        table = pa.Table.from_pandas(df, preserve_index=False)

        # Attach metadata if provided
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

        if self.logger:
            self.logger.info(
                f"Writing flat parquet: {file_path} (rows={len(df)}, cols={len(df.columns)})",
            )

        pq.write_table(
            table,
            file_path,
            compression=compression,
            row_group_size=row_group_size,
            write_statistics=write_statistics,
        )

    def update_manifest(self, base_dir: str, ts_column: str = "timestamp") -> None:
        """Update manifest file for the dataset."""
        try:
            if not os.path.exists(base_dir):
                return

            manifest_path = os.path.join(base_dir, "_manifest.json")
            manifest = {
                "updated_at": datetime.now(UTC).isoformat(),
                "base_dir": base_dir,
                "timestamp_column": ts_column,
            }

            # Count files and get latest timestamp
            file_count = 0
            latest_ts = None

            for root, _dirs, files in os.walk(base_dir):
                for file in files:
                    if file.endswith(".parquet"):
                        file_count += 1
                        file_path = os.path.join(root, file)
                        try:
                            # Read timestamp from file metadata
                            parquet_file = pq.ParquetFile(file_path)
                            if ts_column in parquet_file.schema_arrow.names:
                                # Get min/max from statistics if available
                                stats = parquet_file.metadata.row_group(0).statistics
                                if stats and ts_column in stats:
                                    col_stats = stats[ts_column]
                                    if (
                                        hasattr(col_stats, "max")
                                        and col_stats.max is not None
                                    ):
                                        latest_ts = (
                                            max(latest_ts, col_stats.max)
                                            if latest_ts is not None
                                            else col_stats.max
                                        )
                        except Exception:
                            pass

            manifest["file_count"] = file_count
            manifest["latest_timestamp"] = latest_ts

            import json

            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2, default=str)

            if self.logger:
                self.logger.info(f"Updated manifest: {manifest_path}")

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to update manifest: {e}")

    def get_latest_timestamp(
        self, base_dir: str, ts_column: str = "timestamp",
    ) -> int | None:
        """Get the latest timestamp from the dataset."""
        try:
            manifest_path = os.path.join(base_dir, "_manifest.json")
            if os.path.exists(manifest_path):
                import json

                with open(manifest_path) as f:
                    manifest = json.load(f)
                return manifest.get("latest_timestamp")
        except Exception:
            pass
        return None

    def get_latest_timestamp_from_manifest(self, base_dir: str) -> int | None:
        """Get the latest timestamp from the manifest file."""
        return self.get_latest_timestamp(base_dir)

    def cached_projection(
        self,
        base_dir: str,
        filters: list | None,
        columns: list[str],
        cache_dir: str,
        cache_key_prefix: str,
        compression: str = "snappy",
        snapshot_version: str | None = None,
        ttl_seconds: int | None = None,
        batch_size: int | None = None,
        arrow_transform: Callable[[pa.Table], pa.Table] | None = None,
    ) -> pd.DataFrame:
        """Read a projection with caching support."""
        self._ensure_pyarrow()

        # Generate cache key
        import hashlib

        cache_key = (
            f"{cache_key_prefix}_{hashlib.md5(str(filters).encode()).hexdigest()[:8]}"
        )
        if snapshot_version:
            cache_key += f"_{snapshot_version}"

        cache_path = os.path.join(cache_dir, f"{cache_key}.parquet")

        # Check if cache is valid
        if os.path.exists(cache_path) and ttl_seconds:
            import time

            if time.time() - os.path.getmtime(cache_path) < ttl_seconds:
                try:
                    return pd.read_parquet(cache_path)
                except Exception:
                    pass

        # Read from source
        df = self.scan_dataset(base_dir, filters, columns, batch_size)

        # Apply transform if provided
        if arrow_transform and not df.empty:
            try:
                table = pa.Table.from_pandas(df)
                transformed_table = arrow_transform(table)
                df = transformed_table.to_pandas(types_mapper=pd.ArrowDtype)
            except Exception:
                pass

        # Cache the result
        try:
            os.makedirs(cache_dir, exist_ok=True)
            df.to_parquet(cache_path, compression=compression)
        except Exception:
            pass

        return df

    def materialize_projection(
        self,
        base_dir: str,
        filters: list | None,
        columns: list[str],
        output_dir: str,
        partition_cols: list[str],
        schema_name: str = "split",
        compression: str = "snappy",
        batch_size: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Materialize a projection to a new partitioned dataset."""
        df = self.scan_dataset(base_dir, filters, columns, batch_size)

        if df.empty:
            msg = "No data found for the specified filters"
            raise ValueError(msg)

        os.makedirs(output_dir, exist_ok=True)

        self.write_partitioned_dataset(
            df=df,
            base_dir=output_dir,
            partition_cols=partition_cols,
            schema_name=schema_name,
            compression=compression,
            metadata=metadata,
        )

        return output_dir

    def compact_dataset(
        self,
        base_dir: str,
        output_dir: str | None = None,
        compression: str = "snappy",
        min_rows_per_group: int | None = None,
        max_rows_per_file: int | None = None,
    ) -> str | None:
        """Compact a dataset by rewriting with optimized settings."""
        if output_dir is None:
            output_dir = f"{base_dir}_compacted"

        try:
            # Read all data
            df = self.scan_dataset(base_dir)

            if df.empty:
                return None

            # Determine partitioning from existing structure
            partition_cols = []
            for col in ["exchange", "symbol", "timeframe", "year", "month", "day"]:
                if col in df.columns:
                    partition_cols.append(col)

            # Write compacted dataset
            self.write_partitioned_dataset(
                df=df,
                base_dir=output_dir,
                partition_cols=partition_cols,
                schema_name="unified",
                compression=compression,
                min_rows_per_group=min_rows_per_group or 50000,
                max_rows_per_file=max_rows_per_file or 5_000_000,
            )

            return output_dir

        except Exception as e:
            if self.logger:
                self.logger.exception(f"Failed to compact dataset: {e}")
            return None

    def migrate_flat_parquet_dir_to_partitioned(
        self,
        src_dir: str,
        dst_base_dir: str,
        schema_name: str,
        static_columns: dict[str, str | int] | None = None,
        compression: str = "snappy",
    ) -> None:
        """Migrate a directory of flat parquet files to partitioned format."""
        self._ensure_pyarrow()

        if not os.path.exists(src_dir):
            msg = f"Source directory not found: {src_dir}"
            raise FileNotFoundError(msg)

        # Find all parquet files
        parquet_files = []
        for root, _dirs, files in os.walk(src_dir):
            for file in files:
                if file.endswith(".parquet"):
                    parquet_files.append(os.path.join(root, file))

        if not parquet_files:
            msg = f"No parquet files found in {src_dir}"
            raise ValueError(msg)

        # Read and combine all files
        dfs = []
        for file_path in parquet_files:
            try:
                df = pd.read_parquet(file_path)
                dfs.append(df)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to read {file_path}: {e}")

        if not dfs:
            msg = "No valid parquet files could be read"
            raise ValueError(msg)

        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)

        # Add static columns if provided
        if static_columns:
            for col, value in static_columns.items():
                combined_df[col] = value

        # Determine partitioning columns
        partition_cols = []
        for col in ["exchange", "symbol", "timeframe", "year", "month", "day"]:
            if col in combined_df.columns:
                partition_cols.append(col)

        # Write partitioned dataset
        self.write_partitioned_dataset(
            df=combined_df,
            base_dir=dst_base_dir,
            partition_cols=partition_cols,
            schema_name=schema_name,
            compression=compression,
        )


class UnifiedDataConverter:
    """Converts existing parquet files to unified consolidated format and sets up
    infrastructure for future data collection.

    This step bridges the gap between the old individual consolidated files
    and the new unified system with daily partitioning.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("UnifiedDataConverter")

        # Configuration
        self.data_cache_dir = "data_cache"
        self.unified_dir = os.path.join(self.data_cache_dir, "unified")
        self.backup_dir = os.path.join(self.data_cache_dir, "backup_pre_unified")

        # Ensure directories exist
        os.makedirs(self.unified_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)

    async def initialize(self) -> None:
        """Initialize the converter."""
        self.logger.info("🚀 Initializing Unified Data Converter...")
        self.logger.info(f"📁 Unified data directory: {self.unified_dir}")
        self.logger.info(f"📁 Backup directory: {self.backup_dir}")

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="unified data conversion",
    )
    async def execute(
        self,
        symbol: str,
        exchange: str,
        timeframe: str = "1m",
        data_dir: str = "data_cache",
        force_rerun: bool = False,
    ) -> bool:
        """Execute the unified data conversion process.

        Args:
            symbol: Trading symbol (e.g., "ETHUSDT")
            exchange: Exchange name (e.g., "BINANCE")
            timeframe: Timeframe (e.g., "1m")
            data_dir: Data directory
            force_rerun: Force re-run even if unified data exists

        Returns:
            bool: True if successful, False otherwise

        """
        try:

            # Update data cache directory to use the passed data_dir
            self.data_cache_dir = data_dir
            self.unified_dir = os.path.join(self.data_cache_dir, "unified")
            self.backup_dir = os.path.join(self.data_cache_dir, "backup_pre_unified")

            # Create directories
            os.makedirs(self.unified_dir, exist_ok=True)
            os.makedirs(self.backup_dir, exist_ok=True)


            self.logger.info("=" * 80)
            self.logger.info("🔄 STEP 1.5: Unified Data Converter")
            self.logger.info("=" * 80)
            self.logger.info(f"🎯 Symbol: {symbol}")
            self.logger.info(f"🏢 Exchange: {exchange}")
            self.logger.info(f"📊 Timeframe: {timeframe}")
            self.logger.info(f"📁 Data directory: {data_dir}")

            # Check if unified data already exists
            unified_exists = await self._check_unified_data_exists(
                symbol, exchange, timeframe,
            )

            if unified_exists:
                if force_rerun:
                    self.logger.info(
                        "🔄 Force rerun requested - will reprocess all data",
                    )
                    # Step 1: Backup existing data
                    await self._backup_existing_data(symbol, exchange, timeframe)
                else:
                    self.logger.info(
                        "✅ Unified data already exists, checking for incremental updates...",
                    )
                    # Check if we need to process only new data
                    incremental_success = await self._process_incremental_updates(
                        symbol, exchange, timeframe,
                    )
                    if incremental_success:
                        self.logger.info("✅ Incremental processing completed")
                        return True
                    self.logger.info("🔄 Full reprocessing required")
                    # Step 1: Backup existing data
                    await self._backup_existing_data(symbol, exchange, timeframe)
            else:
                self.logger.info(
                    "🔄 No existing unified data found - performing initial conversion",
                )

            # Step 2: Convert existing consolidated files
            conversion_success = await self._convert_existing_data(
                symbol, exchange, timeframe,
            )
            if not conversion_success:
                self.logger.error("❌ Failed to convert existing data")
                return False

            # Step 3: Set up infrastructure for future data
            infrastructure_success = await self._setup_future_infrastructure(
                symbol, exchange, timeframe,
            )
            if not infrastructure_success:
                self.logger.error("❌ Failed to set up future infrastructure")
                return False

            # Step 4: Validate unified dataset
            validation_success = await self._validate_unified_dataset(
                symbol, exchange, timeframe,
            )
            if not validation_success:
                self.logger.error("❌ Unified dataset validation failed")
                return False

            # Step 5: Verify data quality and completeness
            verification_success = await self._verify_unified_data_quality(
                symbol, exchange, timeframe,
            )
            if not verification_success:
                self.logger.warning("⚠️ Data quality verification found issues")

            self.logger.info("=" * 80)
            self.logger.info("✅ STEP 1.5 COMPLETED: Unified Data Converter")
            self.logger.info("=" * 80)

            # Clean up memory
            import gc

            gc.collect()

            return True

        except Exception as e:
            self.logger.exception(f"❌ Unified data conversion failed: {e}")
            return False
        finally:
            # Ensure cleanup happens even if there's an error
            try:
                import gc

                gc.collect()
            except:
                pass

    async def _check_unified_data_exists(
        self, symbol: str, exchange: str, timeframe: str,
    ) -> bool:
        """Check if unified data already exists."""
        try:
            unified_base = os.path.join(
                self.unified_dir, exchange.lower(), symbol, timeframe,
            )
            if os.path.exists(unified_base):
                # Check if there are any parquet files
                parquet_files = glob.glob(
                    os.path.join(unified_base, "**/*.parquet"), recursive=True,
                )
                if parquet_files:
                    self.logger.info(
                        f"✅ Found existing unified data: {len(parquet_files)} files",
                    )
                    return True
            return False
        except Exception as e:
            self.logger.warning(f"⚠️ Error checking unified data existence: {e}")
            return False

    async def _process_incremental_updates(
        self, symbol: str, exchange: str, timeframe: str,
    ) -> bool:
        """Process only new data that hasn't been converted yet."""
        try:
            self.logger.info("🔍 Checking for incremental updates...")

            # Get the latest timestamp from existing unified data
            unified_base = os.path.join(
                self.unified_dir, exchange.lower(), symbol, timeframe,
            )

            # Find all existing dates in unified data
            unified_dates = set()
            parquet_files = glob.glob(
                os.path.join(unified_base, "**/*.parquet"), recursive=True,
            )

            if not parquet_files:
                self.logger.info(
                    "⚠️ No existing parquet files found - full reprocessing needed",
                )
                return False

            # Extract dates from file paths
            for file_path in parquet_files:
                try:
                    # Extract date from path like: .../year=2025/month=07/day=15/...
                    path_parts = file_path.split("/")
                    for i, part in enumerate(path_parts):
                        if part.startswith("year="):
                            year = int(part.split("=")[1])
                            month = int(path_parts[i + 1].split("=")[1])
                            day = int(path_parts[i + 2].split("=")[1])
                            unified_dates.add(date(year, month, day))
                            break
                except Exception as e:
                    self.logger.warning(f"⚠️ Error parsing date from {file_path}: {e}")

            if not unified_dates:
                self.logger.info(
                    "⚠️ Could not determine existing unified dates - full reprocessing needed",
                )
                return False

            # Get the date range from source klines data
            klines_data = await self._load_klines_data(symbol, exchange, timeframe)
            if klines_data is None or klines_data.empty:
                self.logger.error(
                    "❌ No klines data available for incremental processing",
                )
                return False

            # Convert timestamps to dates
            klines_data["date"] = pd.to_datetime(
                klines_data["timestamp"], unit="ms", utc=True,
            ).dt.date
            klines_dates = set(klines_data["date"].unique())

            # Find missing dates
            missing_dates = klines_dates - unified_dates
            missing_dates = sorted(missing_dates)

            if not missing_dates:
                self.logger.info(
                    "✅ No missing dates found - unified dataset is complete",
                )
                return True

            self.logger.info(
                f"🔄 Found {len(missing_dates)} missing dates: {missing_dates[:5]}{'...' if len(missing_dates) > 5 else ''}",
            )

            # Process only the missing data
            success = await self._process_data_incrementally(
                klines_data, symbol, exchange, timeframe, start_date=min(missing_dates),
            )

            if success:
                self.logger.info("✅ Incremental processing completed successfully")
                return True
            self.logger.error("❌ Incremental processing failed")
            return False

        except Exception as e:
            self.logger.exception(f"❌ Error during incremental processing: {e}")
            return False

    async def _backup_existing_data(
        self, symbol: str, exchange: str, timeframe: str,
    ) -> None:
        """Backup existing consolidated files."""
        try:
            self.logger.info("📦 Backing up existing consolidated data...")

            # Find existing consolidated files
            patterns = [
                f"klines_{exchange}_{symbol}_{timeframe}_consolidated.*",
                f"aggtrades_{exchange}_{symbol}_consolidated.*",
                f"futures_{exchange}_{symbol}_consolidated.*",
            ]

            backup_count = 0
            for pattern in patterns:
                files = glob.glob(os.path.join(self.data_cache_dir, pattern))
                for file_path in files:
                    try:
                        filename = os.path.basename(file_path)
                        backup_path = os.path.join(self.backup_dir, filename)

                        # Only backup if not already backed up
                        if not os.path.exists(backup_path):
                            import shutil

                            shutil.copy2(file_path, backup_path)
                            backup_count += 1
                            self.logger.info(f"   📦 Backed up: {filename}")
                    except Exception as e:
                        self.logger.warning(f"   ⚠️ Failed to backup {file_path}: {e}")

            self.logger.info(f"✅ Backup completed: {backup_count} files backed up")

        except Exception as e:
            self.logger.warning(f"⚠️ Backup process failed: {e}")

    async def _convert_existing_data(
        self, symbol: str, exchange: str, timeframe: str,
    ) -> bool:
        """Convert existing consolidated files to unified format incrementally."""
        try:
            self.logger.info(
                "🔄 Converting existing consolidated data to unified format incrementally...",
            )

            # Load klines data first (this is our base data)
            klines_data = await self._load_klines_data(symbol, exchange, timeframe)

            if klines_data is None or klines_data.empty:
                self.logger.error(
                    "❌ No klines data found - cannot proceed with conversion",
                )
                return False

            self.logger.info(f"✅ Loaded {len(klines_data)} klines rows")

            # Process data incrementally by date
            success = await self._process_data_incrementally(
                klines_data, symbol, exchange, timeframe,
            )

            if success:
                self.logger.info("✅ Incremental conversion completed successfully")
                return True
            self.logger.error("❌ Incremental conversion failed")
            return False

        except Exception as e:
            self.logger.exception(f"❌ Data conversion failed: {e}")
            return False

    async def _process_data_incrementally(
        self,
        klines_data: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str,
        start_date: date | None = None,
    ) -> bool:
        """Process data incrementally by date to avoid memory issues."""
        try:
            self.logger.info("🔄 Processing data incrementally by date...")

            # Ensure timestamp is in the right format
            if "timestamp" in klines_data.columns:
                klines_data = klines_data.copy()
                # Convert to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(klines_data["timestamp"]):
                    klines_data["timestamp"] = pd.to_datetime(
                        klines_data["timestamp"], utc=True,
                    )

            # Add date columns for partitioning
            ts = klines_data["timestamp"]
            klines_data["year"] = ts.dt.year.astype("int16")
            klines_data["month"] = ts.dt.month.astype("int8")
            klines_data["day"] = ts.dt.day.astype("int8")

            # Get date range from actual data timestamps
            min_date = start_date if start_date else ts.min().date()
            max_date = ts.max().date()
            total_days = (max_date - min_date).days + 1

            if start_date:
                self.logger.info(
                    f"📅 Processing {total_days} days from {min_date} to {max_date} (incremental)",
                )
            else:
                self.logger.info(
                    f"📅 Processing {total_days} days from {min_date} to {max_date}",
                )

            # Define base directory for unified dataset
            base_dir = os.path.join(
                self.unified_dir, exchange.lower(), symbol, timeframe,
            )
            os.makedirs(base_dir, exist_ok=True)

            # Process each day incrementally
            processed_days = 0
            total_rows_processed = 0

            current_date = min_date
            while current_date <= max_date:
                try:
                    self.logger.info(
                        f"📅 Processing date: {current_date} ({processed_days + 1}/{total_days})",
                    )

                    # Filter klines data for current date
                    date_mask = (
                        (klines_data["year"] == current_date.year)
                        & (klines_data["month"] == current_date.month)
                        & (klines_data["day"] == current_date.day)
                    )
                    daily_klines = klines_data[date_mask].copy()

                    if daily_klines.empty:
                        current_date += timedelta(days=1)
                        processed_days += 1
                        continue


                    # Load aggtrades data for this date
                    daily_aggtrades = await self._load_aggtrades_for_date(
                        symbol, exchange, current_date,
                    )

                    # Load futures data for this date
                    daily_futures = await self._load_futures_for_date(
                        symbol, exchange, current_date,
                    )

                    # Merge data for this day
                    daily_unified = await self._merge_daily_data(
                        daily_klines,
                        daily_aggtrades,
                        daily_futures,
                        symbol,
                        exchange,
                        timeframe,
                    )

                    if daily_unified is not None and not daily_unified.empty:
                        # Write daily partition
                        success = await self._write_daily_partition(
                            daily_unified,
                            symbol,
                            exchange,
                            timeframe,
                            current_date,
                            base_dir,
                        )

                        if success:
                            total_rows_processed += len(daily_unified)
                            self.logger.info(
                                f"   ✅ Processed {len(daily_unified)} kline rows for {current_date}",
                            )
                        else:
                            self.logger.error(
                                f"   ❌ Failed to write kline     data for {current_date}",
                            )
                    else:
                        pass

                    processed_days += 1
                    current_date += timedelta(days=1)

                    # Progress update every 10 days
                    if processed_days % 10 == 0:
                        progress_pct = (processed_days / total_days) * 100
                        self.logger.info(
                            f"📊 Progress: {processed_days}/{total_days} days ({progress_pct:.1f}%) - {total_rows_processed:,} total rows",
                        )

                except Exception as e:
                    self.logger.exception(f"   ❌ Error processing {current_date}: {e}")
                    current_date += timedelta(days=1)
                    processed_days += 1
                    continue

            self.logger.info(
                f"✅ Incremental processing completed: {total_rows_processed:,} total rows across {processed_days} days",
            )

            return True

        except Exception as e:
            self.logger.exception(f"❌ Incremental processing failed: {e}")
            return False

    async def _load_aggtrades_for_date(
        self, symbol: str, exchange: str, target_date: datetime.date,
    ) -> pd.DataFrame | None:
        """Load aggtrades data for a specific date."""
        try:
            # Look for aggtrades data in the parquet directory
            parquet_dir = os.path.join(
                self.data_cache_dir, "parquet", f"aggtrades_{exchange}_{symbol}",
            )

            if not os.path.exists(parquet_dir):
                return None

            # Format target date for file matching
            target_date_str = target_date.strftime("%Y-%m-%d")

            # Find files for the target date
            date_files = []
            for root, _dirs, files in os.walk(parquet_dir):
                for file in files:
                    if file.endswith(".parquet") and target_date_str in file:
                        file_path = os.path.join(root, file)
                        date_files.append(file_path)

            if not date_files:
                self.logger.warning(f"⚠️ No aggtrades files found for {target_date_str}")
                return None

            # Load and combine files for this date
            dfs = []
            for file_path in date_files:
                try:
                    df = pd.read_parquet(file_path)
                    dfs.append(df)
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to load {file_path}: {e}")

            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                # Remove duplicates and sort
                combined_df = combined_df.drop_duplicates(
                    subset=["timestamp", "price", "quantity"], keep="first",
                )
                combined_df = combined_df.sort_values("timestamp").reset_index(
                    drop=True,
                )
                self.logger.info(
                    f"✅ Loaded {len(combined_df)} aggtrades rows for {target_date_str}",
                )
                return combined_df

            return None

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load aggtrades for {target_date}: {e}")
            return None

    async def _load_futures_for_date(
        self, symbol: str, exchange: str, target_date: datetime.date,
    ) -> pd.DataFrame | None:
        """Load futures data for a specific date."""
        try:
            # Look for futures data in the parquet directory
            parquet_dir = os.path.join(
                self.data_cache_dir, "parquet", f"futures_{exchange}_{symbol}",
            )

            if not os.path.exists(parquet_dir):
                return None

            # Format target date for file matching
            target_date_str = target_date.strftime("%Y-%m-%d")

            # Find files for the target date
            date_files = []
            for root, _dirs, files in os.walk(parquet_dir):
                for file in files:
                    if file.endswith(".parquet") and target_date_str in file:
                        file_path = os.path.join(root, file)
                        date_files.append(file_path)

            if not date_files:
                self.logger.warning(f"⚠️ No futures files found for {target_date_str}")
                return None

            # Load and combine files for this date
            dfs = []
            for file_path in date_files:
                try:
                    df = pd.read_parquet(file_path)
                    dfs.append(df)
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to load {file_path}: {e}")

            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                combined_df = combined_df.sort_values("timestamp").reset_index(
                    drop=True,
                )
                self.logger.info(
                    f"✅ Loaded {len(combined_df)} futures rows for {target_date_str}",
                )
                return combined_df

            return None

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load futures for {target_date}: {e}")
            return None

    async def _merge_daily_data(
        self,
        daily_klines: pd.DataFrame,
        daily_aggtrades: pd.DataFrame | None,
        daily_futures: pd.DataFrame | None,
        symbol: str,
        exchange: str,
        timeframe: str,
    ) -> pd.DataFrame | None:
        """Merge daily data into unified format."""
        try:
            # Start with klines data as the base
            unified = daily_klines.copy()

            # Add metadata columns
            unified["exchange"] = exchange.upper()
            unified["symbol"] = symbol
            unified["timeframe"] = timeframe

            # Merge aggtrades data if available
            if daily_aggtrades is not None and not daily_aggtrades.empty:
                # Drop existing aggtrades columns if they exist
                aggtrades_cols = [
                    "trade_volume",
                    "trade_count",
                    "avg_price",
                    "min_price",
                    "max_price",
                    "volume_ratio",
                ]
                unified = unified.drop(
                    columns=[col for col in aggtrades_cols if col in unified.columns],
                )
                unified = await self._merge_daily_aggtrades(unified, daily_aggtrades)

            # Merge futures data if available
            if daily_futures is not None and not daily_futures.empty:
                unified = await self._merge_daily_futures(unified, daily_futures)

            # Fill missing values
            unified = await self._fill_missing_values(unified)

            # Sort by timestamp
            if "timestamp" in unified.columns:
                unified = unified.sort_values("timestamp").reset_index(drop=True)

            return unified

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to merge daily data: {e}")
            return None

    async def _merge_daily_aggtrades(
        self, unified: pd.DataFrame, aggtrades_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge daily aggtrades data into unified dataset."""
        try:
            # Ensure aggtrades timestamp is in the right format
            if "timestamp" in aggtrades_data.columns:
                aggtrades_data = aggtrades_data.copy()
                if aggtrades_data["timestamp"].dtype == "object":
                    aggtrades_data["timestamp"] = pd.to_datetime(
                        aggtrades_data["timestamp"], utc=True,
                    )
                # Keep timestamps as-is since they're already in milliseconds

            # Aggregate aggtrades to kline level
            if (
                "timestamp" in aggtrades_data.columns
                and "price" in aggtrades_data.columns
                and "quantity" in aggtrades_data.columns
            ):
                # Convert aggtrades timestamps to kline boundaries (floor to minute)
                aggtrades_data = aggtrades_data.copy()
                # Convert to datetime, floor to minute, then back to milliseconds
                aggtrades_data["kline_timestamp"] = pd.to_datetime(
                    aggtrades_data["timestamp"], unit="ms", utc=True,
                )
                aggtrades_data["kline_timestamp"] = aggtrades_data[
                    "kline_timestamp"
                ].dt.floor("1min")
                aggtrades_data["kline_timestamp"] = aggtrades_data[
                    "kline_timestamp"
                ].apply(lambda x: int(x.timestamp() * 1000))

                # Group by kline timestamp and aggregate
                agg_stats = (
                    aggtrades_data.groupby("kline_timestamp")
                    .agg(
                        {
                            "quantity": ["sum", "count"],
                            "price": ["mean", "min", "max"],
                        },
                    )
                    .reset_index()
                )

                # Flatten column names
                agg_stats.columns = [
                    "timestamp",
                    "trade_volume",
                    "trade_count",
                    "avg_price",
                    "min_price",
                    "max_price",
                ]

                # Merge with unified data
                unified = unified.merge(agg_stats, on="timestamp", how="left")

                # Fill NaN values in trade columns with 0 (no trades occurred)
                trade_columns = [
                    "trade_volume",
                    "trade_count",
                    "avg_price",
                    "min_price",
                    "max_price",
                ]
                for col in trade_columns:
                    if col in unified.columns:
                        unified[col] = unified[col].fillna(0)

                # Calculate additional metrics
                if "trade_volume" in unified.columns and "volume" in unified.columns:
                    unified["volume_ratio"] = (
                        unified["trade_volume"] / unified["volume"]
                    )
                    # Handle division by zero for volume_ratio
                    unified["volume_ratio"] = unified["volume_ratio"].fillna(0)

            return unified

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to merge daily aggtrades: {e}")
            return unified

    async def _merge_daily_futures(
        self, unified: pd.DataFrame, futures_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge daily futures data into unified dataset."""
        try:
            # Ensure futures timestamp is in the right format
            if "timestamp" in futures_data.columns:
                futures_data = futures_data.copy()
                if futures_data["timestamp"].dtype == "object":
                    futures_data["timestamp"] = pd.to_datetime(
                        futures_data["timestamp"], utc=True,
                    )
                if pd.api.types.is_datetime64_any_dtype(futures_data["timestamp"]):
                    futures_data["timestamp"] = (
                        futures_data["timestamp"].astype(np.int64) // 10**6
                    )
                else:
                    futures_data["timestamp"] = futures_data["timestamp"].astype(
                        np.int64,
                    )

            # Forward fill funding rates
            funding_rate_col = None
            if "fundingRate" in futures_data.columns:
                funding_rate_col = "fundingRate"
            elif "funding_rate" in futures_data.columns:
                funding_rate_col = "funding_rate"

            if "timestamp" in futures_data.columns and funding_rate_col:
                futures_data = futures_data.sort_values("timestamp")
                funding_rates = futures_data.set_index("timestamp")[funding_rate_col]
                unified["funding_rate"] = unified["timestamp"].map(funding_rates)
                unified["funding_rate"] = unified["funding_rate"].ffill()

            return unified

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to merge daily futures: {e}")
            return unified

    async def _write_daily_partition(
        self,
        daily_data: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str,
        target_date: datetime.date,
        base_dir: str,
    ) -> bool:
        """Write daily partition to unified dataset."""
        try:
            # Use actual timestamp data to determine the correct partition path
            if "timestamp" in daily_data.columns and not daily_data.empty:
                # Convert timestamp to datetime to get the actual date
                actual_ts = pd.to_datetime(daily_data["timestamp"], unit="ms", utc=True)
                actual_date = actual_ts.iloc[0].date()

                # Use the actual date from the data, not the target_date
                partition_year = actual_date.year
                partition_month = actual_date.month
                partition_day = actual_date.day
            else:
                # Fallback to target_date if no timestamp data
                partition_year = target_date.year
                partition_month = target_date.month
                partition_day = target_date.day

            # Create partition directory structure
            partition_path = os.path.join(
                base_dir,
                f"exchange={exchange.upper()}",
                f"symbol={symbol}",
                f"timeframe={timeframe}",
                f"year={partition_year}",
                f"month={partition_month:02d}",
                f"day={partition_day:02d}",
            )

            os.makedirs(partition_path, exist_ok=True)

            # Write parquet file
            file_path = os.path.join(partition_path, "part-0.parquet")
            daily_data.to_parquet(file_path, compression="snappy", index=False)

            return True

        except Exception as e:
            self.logger.exception(
                f"❌ Failed to write daily partition for {target_date}: {e}",
            )
            return False

    async def _setup_future_infrastructure(
        self, symbol: str, exchange: str, timeframe: str,
    ) -> bool:
        """Set up infrastructure for future data collection."""
        try:
            self.logger.info(
                "🔧 Setting up infrastructure for future data collection...",
            )

            # Create configuration for future data collection
            future_config = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "unified_base_dir": os.path.join(
                    self.unified_dir, exchange.lower(), symbol, timeframe,
                ),
                "partitioning": [
                    "exchange",
                    "symbol",
                    "timeframe",
                    "year",
                    "month",
                    "day",
                ],
                "compression": "snappy",
                "max_rows_per_file": 1_000_000,
                "schema_name": "unified",
                "created_at": datetime.now(UTC).isoformat(),
            }

            # Save configuration
            config_path = os.path.join(
                self.unified_dir, f"{exchange.lower()}_{symbol}_{timeframe}_config.json",
            )
            import json

            with open(config_path, "w") as f:
                json.dump(future_config, f, indent=2)

            self.logger.info(f"✅ Future infrastructure config saved to: {config_path}")
            return True

        except Exception as e:
            self.logger.exception(f"❌ Failed to set up future infrastructure: {e}")
            return False

    async def _validate_unified_dataset(
        self, symbol: str, exchange: str, timeframe: str,
    ) -> bool:
        """Validate the unified dataset."""
        try:
            self.logger.info("🔍 Validating unified dataset...")

            # Use the local ParquetDatasetManager
            try:
                pdm = ParquetDatasetManager(logger=self.logger)
            except Exception as e:
                self.logger.exception(f"❌ ParquetDatasetManager not available: {e}")
                return False

            # Define base directory
            base_dir = os.path.join(
                self.unified_dir, exchange.lower(), symbol, timeframe,
            )

            # Scan the dataset
            try:
                sample_data = pdm.scan_dataset(
                    base_dir=base_dir,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                    batch_size=1000,
                )

                if sample_data is not None and not sample_data.empty:
                    self.logger.info(
                        f"✅ Dataset validation successful: {len(sample_data)} sample rows",
                    )

                    # Check for required columns
                    required_columns = [
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                    ]
                    missing_columns = [
                        col
                        for col in required_columns
                        if col not in sample_data.columns
                    ]

                    if missing_columns:
                        self.logger.error(
                            f"❌ Missing required columns: {missing_columns}",
                        )
                        return False

                    # Check data quality
                    if sample_data["timestamp"].isna().any():
                        self.logger.warning("⚠️ Found null timestamps in sample data")

                    if sample_data["volume"].isna().any():
                        self.logger.warning("⚠️ Found null volumes in sample data")

                    return True
                self.logger.error("❌ No data found in unified dataset")
                return False

            except Exception as e:
                self.logger.exception(f"❌ Failed to scan unified dataset: {e}")
                return False

        except Exception as e:
            self.logger.exception(f"❌ Dataset validation failed: {e}")
            return False

    async def _verify_unified_data_quality(
        self, symbol: str, exchange: str, timeframe: str,
    ) -> bool:
        """Verify data quality and completeness of the unified dataset."""
        try:
            self.logger.info("🔍 Verifying unified data quality...")

            # Get the unified dataset path
            unified_path = self.get_unified_data_path(symbol, exchange, timeframe)

            if not os.path.exists(unified_path):
                self.logger.error(
                    f"❌ Unified dataset path does not exist: {unified_path}",
                )
                return False

            # Check a few sample dates across the dataset
            test_dates = [
                ("2025-01-01", "year=2025/month=01/day=01"),
                ("2025-04-15", "year=2025/month=04/day=15"),
                ("2025-07-15", "year=2025/month=07/day=15"),
                ("2025-08-08", "year=2025/month=08/day=08"),
            ]

            base_path = os.path.join(
                unified_path,
                f"exchange={exchange.upper()}",
                f"symbol={symbol}",
                f"timeframe={timeframe}",
            )

            quality_issues = []

            for date_str, partition_path in test_dates:
                file_path = os.path.join(base_path, partition_path, "part-0.parquet")

                if os.path.exists(file_path):
                    try:
                        df = pd.read_parquet(file_path)

                        # Check data presence
                        klines_present = all(
                            col in df.columns
                            for col in ["open", "high", "low", "close", "volume"]
                        )
                        aggtrades_present = all(
                            col in df.columns
                            for col in [
                                "trade_volume",
                                "trade_count",
                                "avg_price",
                                "min_price",
                                "max_price",
                                "volume_ratio",
                            ]
                        )
                        futures_present = "funding_rate" in df.columns

                        # Check data quality
                        aggtrades_coverage = (
                            (df["trade_volume"] > 0).sum() / len(df) * 100
                            if len(df) > 0
                            else 0
                        )
                        futures_coverage = (
                            df["funding_rate"].notna().sum() / len(df) * 100
                            if len(df) > 0
                            else 0
                        )

                        if not klines_present:
                            quality_issues.append(f"{date_str}: Missing klines data")
                        if not aggtrades_present:
                            quality_issues.append(f"{date_str}: Missing aggtrades data")
                        if not futures_present:
                            quality_issues.append(f"{date_str}: Missing futures data")
                        if aggtrades_coverage < 80:
                            quality_issues.append(
                                f"{date_str}: Low aggtrades coverage ({aggtrades_coverage:.1f}%)",
                            )
                        if futures_coverage < 80:
                            quality_issues.append(
                                f"{date_str}: Low futures coverage ({futures_coverage:.1f}%)",
                            )

                    except Exception as e:
                        quality_issues.append(f"{date_str}: Error reading file - {e}")
                else:
                    quality_issues.append(f"{date_str}: File not found")

            # Report quality status
            if quality_issues:
                self.logger.warning("⚠️ Data quality issues found:")
                for issue in quality_issues:
                    self.logger.warning(f"   - {issue}")
                return False
            self.logger.info(
                "✅ Data quality verification passed - all data types present and well-populated",
            )
            return True

        except Exception as e:
            self.logger.exception(f"❌ Data quality verification failed: {e}")
            return False

    def get_unified_data_path(self, symbol: str, exchange: str, timeframe: str) -> str:
        """Get the path to the unified dataset."""
        return os.path.join(self.unified_dir, exchange.lower(), symbol, timeframe)

    def get_unified_config_path(
        self, symbol: str, exchange: str, timeframe: str,
    ) -> str:
        """Get the path to the unified dataset configuration."""
        return os.path.join(
            self.unified_dir, f"{exchange.lower()}_{symbol}_{timeframe}_config.json",
        )

    @validate_klines_data_quality
    @secure_data_processing
    @prevent_data_leakage
    @resource_monitor
    @memory_efficient
    @quality_gate
    @handle_data_processing_errors(context="klines_data_loading")
    @validate_klines_data(context="loading")
    @format_klines_data(context="loading")
    @log_step_metrics(context="klines_loading")
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="klines data loading",
    )
    async def _load_klines_data(
        self, symbol: str, exchange: str, timeframe: str,
    ) -> pd.DataFrame | None:
        """Load klines data from existing consolidated files or download directly.

        This method is secured with decorators for:
        - Data quality validation
        - Security checks
        - Pipeline monitoring
        - Error handling
        - Resource monitoring
        - Memory efficiency
        - Quality gates
        """
        try:
            # Always check data_cache first (where the data actually is)
            data_cache_dir = "data_cache"

            # Try parquet first in data_cache
            parquet_path = os.path.join(
                data_cache_dir,
                f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet",
            )
            if os.path.exists(parquet_path):
                self.logger.info(f"📊 Loading klines from parquet: {parquet_path}")
                df = pd.read_parquet(parquet_path)
                self.logger.info(f"   ✅ Loaded {len(df)} klines rows")
                return df

            # Try CSV fallback in data_cache
            csv_path = os.path.join(
                data_cache_dir,
                f"klines_{exchange}_{symbol}_{timeframe}_consolidated.csv",
            )
            if os.path.exists(csv_path):
                self.logger.info(f"📊 Loading klines from CSV: {csv_path}")
                df = pd.read_csv(csv_path)
                self.logger.info(f"   ✅ Loaded {len(df)} klines rows")
                return df

            # Try PKL fallback in data_cache
            pkl_path = os.path.join(
                data_cache_dir,
                f"klines_{exchange}_{symbol}_{timeframe}_consolidated_cached_data.pkl",
            )
            if os.path.exists(pkl_path):
                self.logger.info(f"📊 Loading klines from PKL: {pkl_path}")
                df = pd.read_pickle(pkl_path)
                self.logger.info(f"   ✅ Loaded {len(df)} klines rows")
                return df

            # If not found in data_cache, try the passed data_dir as fallback
            if self.data_cache_dir != data_cache_dir:
                parquet_path = os.path.join(
                    self.data_cache_dir,
                    f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet",
                )
                if os.path.exists(parquet_path):
                    self.logger.info(
                        f"📊 Loading klines from fallback path: {parquet_path}",
                    )
                    df = pd.read_parquet(parquet_path)
                    self.logger.info(f"   ✅ Loaded {len(df)} klines rows")
                    return df

            # If no klines data found, try to download it directly
            self.logger.info(
                "🔄 No klines data found, attempting to download klines directly...",
            )
            klines_df = await self._download_klines_data(
                symbol, exchange, timeframe,
            )
            if klines_df is not None and not klines_df.empty:
                self.logger.info(
                    f"✅ Successfully downloaded klines data: {len(klines_df)} rows",
                )
                return klines_df

            self.logger.warning(
                f"⚠️ No klines data found for {exchange}_{symbol}_{timeframe}",
            )
            return None

        except Exception as e:
            self.logger.exception(f"❌ Failed to load klines data: {e}")
            return None

    @secure_klines_download_operation
    @validate_klines_data_quality
    @secure_data_processing
    @prevent_data_leakage
    @resource_monitor
    @memory_efficient
    @quality_gate
    @circuit_breaker_protection
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="klines download operation",
    )
    async def _download_klines_data(
        self, symbol: str, exchange: str, timeframe: str,
    ) -> pd.DataFrame | None:
        """Download klines data directly using the existing step1 download logic.

        This method is secured with comprehensive decorators for:
        - Data quality validation
        - Security checks
        - Pipeline monitoring
        - Error handling
        - Resource monitoring
        - Memory efficiency
        - Quality gates
        - Circuit breaker protection
        """
        try:
            self.logger.info(
                f"🔄 Downloading klines data for {exchange}_{symbol}_{timeframe}",
            )

            # Import the data downloader
            try:
                from src.training.steps.data_downloader import download_all_data_with_consolidation
            except ImportError:
                self.logger.error("❌ Could not import data downloader")
                return None

            # Download klines data
            success = await download_all_data_with_consolidation(
                symbol=symbol,
                exchange_name=exchange,
                interval=timeframe,
            )

            if not success:
                self.logger.error("❌ Failed to download klines data")
                return None

            # After successful download, try to load the data again
            self.logger.info("🔄 Attempting to load downloaded klines data...")

            # Check for the downloaded klines files
            data_cache_dir = "data_cache"

            # Look for monthly klines files that were downloaded
            klines_pattern = f"klines_{exchange}_{symbol}_{timeframe}_*.csv"
            klines_files = glob.glob(os.path.join(data_cache_dir, klines_pattern))

            if not klines_files:
                self.logger.warning(f"⚠️ No klines files found after download: {klines_pattern}")
                return None

            self.logger.info(f"📁 Found {len(klines_files)} klines files after download")

            # Load and combine all klines files
            all_klines = []
            for file_path in sorted(klines_files):
                try:
                    df = pd.read_csv(file_path)
                    if not df.empty:
                        all_klines.append(df)
                        self.logger.debug(
                            f"📊 Loaded {len(df)} rows from {os.path.basename(file_path)}",
                        )
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to load {file_path}: {e}")
                    continue

            if not all_klines:
                self.logger.error("❌ No valid klines data found after download")
                return None

            # Combine all klines data
            combined_klines = pd.concat(all_klines, ignore_index=True)
            self.logger.info(
                f"📊 Combined {len(combined_klines)} total klines rows",
            )

            # Remove duplicates and sort by timestamp
            combined_klines = combined_klines.drop_duplicates().sort_values("timestamp").reset_index(drop=True)

            # Save the consolidated klines data for future use
            output_path = os.path.join(
                data_cache_dir,
                f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet",
            )
            combined_klines.to_parquet(output_path, index=False)
            self.logger.info(f"💾 Saved consolidated klines to: {output_path}")

            return combined_klines

        except Exception as e:
            self.logger.exception(f"❌ Failed to download klines data: {e}")
            return None

    @validate_klines_data_quality
    @secure_data_processing
    @prevent_data_leakage
    @resource_monitor
    @memory_efficient
    @quality_gate
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="deprecated aggtrades to klines conversion",
    )
    async def _create_klines_from_aggtrades(
        self, symbol: str, exchange: str, timeframe: str,
    ) -> pd.DataFrame | None:
        """Create klines data from individual aggtrades files.

        DEPRECATED: This method should not be used. Use _download_klines_data instead
        to download klines directly from the exchange.

        This method is secured with decorators for:
        - Data quality validation
        - Security checks
        - Pipeline monitoring
        - Error handling
        - Resource monitoring
        - Memory efficiency
        - Quality gates
        """
        import warnings
        warnings.warn(
            "_create_klines_from_aggtrades is deprecated. Use _download_klines_data instead.",
            DeprecationWarning,
            stacklevel=2
        )
        try:
            self.logger.info(
                f"🔄 Creating klines from aggtrades for {exchange}_{symbol}_{timeframe}",
            )

            # Find all aggtrades files for this symbol/exchange
            aggtrades_pattern = f"aggtrades_{exchange}_{symbol}_*.parquet"
            aggtrades_files = glob.glob(os.path.join("data_cache", aggtrades_pattern))

            if not aggtrades_files:
                self.logger.warning(
                    f"⚠️ No aggtrades files found matching pattern: {aggtrades_pattern}",
                )
                return None

            self.logger.info(f"📁 Found {len(aggtrades_files)} aggtrades files")

            # Load and combine all aggtrades files
            all_aggtrades = []
            for file_path in sorted(aggtrades_files):
                try:
                    df = pd.read_parquet(file_path)
                    if not df.empty:
                        all_aggtrades.append(df)
                        self.logger.debug(
                            f"📊 Loaded {len(df)} rows from {os.path.basename(file_path)}",
                        )
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to load {file_path}: {e}")
                    continue

            if not all_aggtrades:
                self.logger.error("❌ No valid aggtrades data found")
                return None

            # Combine all aggtrades data
            combined_aggtrades = pd.concat(all_aggtrades, ignore_index=True)
            self.logger.info(
                f"📊 Combined {len(combined_aggtrades)} total aggtrades rows",
            )

            # Convert aggtrades to klines format
            klines_df = self._convert_aggtrades_to_klines(combined_aggtrades, timeframe)

            if klines_df is not None and not klines_df.empty:
                # Save the consolidated klines data for future use
                output_path = os.path.join(
                    "data_cache",
                    f"klines_{exchange}_{symbol}_{timeframe}_consolidated.parquet",
                )
                klines_df.to_parquet(output_path, index=False)
                self.logger.info(f"💾 Saved consolidated klines to: {output_path}")

            return klines_df

        except Exception as e:
            self.logger.exception(f"❌ Failed to create klines from aggtrades: {e}")
            return None

    @validate_klines_data_quality
    @secure_data_processing
    @prevent_data_leakage
    @resource_monitor
    @memory_efficient
    @quality_gate
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="aggtrades to klines conversion",
    )
    def _convert_aggtrades_to_klines(
        self, aggtrades_df: pd.DataFrame, timeframe: str,
    ) -> pd.DataFrame | None:
        """Convert aggtrades data to klines format.

        This method is secured with decorators for:
        - Data quality validation
        - Security checks
        - Pipeline monitoring
        - Error handling
        - Resource monitoring
        - Memory efficiency
        - Quality gates
        """
        try:
            # Ensure we have the required columns
            required_columns = ["timestamp", "price", "quantity"]
            if not all(col in aggtrades_df.columns for col in required_columns):
                self.logger.error(
                    f"❌ Missing required columns. Available: {list(aggtrades_df.columns)}",
                )
                return None

            # Convert timestamp to datetime if needed
            if aggtrades_df["timestamp"].dtype == "int64":
                aggtrades_df["timestamp"] = pd.to_datetime(
                    aggtrades_df["timestamp"], unit="ms",
                )
            else:
                aggtrades_df["timestamp"] = pd.to_datetime(aggtrades_df["timestamp"])

            # Ensure timezone awareness
            if aggtrades_df["timestamp"].dt.tz is None:
                aggtrades_df["timestamp"] = aggtrades_df["timestamp"].dt.tz_localize(
                    "UTC",
                )

            # Sort by timestamp
            aggtrades_df = aggtrades_df.sort_values("timestamp").reset_index(drop=True)

            # Resample to the target timeframe
            aggtrades_df = aggtrades_df.set_index("timestamp")

            # Define resampling rules based on timeframe
            if timeframe == "1m":
                rule = "1T"  # 1 minute
            elif timeframe == "5m":
                rule = "5T"  # 5 minutes
            elif timeframe == "15m":
                rule = "15T"  # 15 minutes
            elif timeframe == "30m":
                rule = "30T"  # 30 minutes
            elif timeframe == "1h":
                rule = "1H"  # 1 hour
            else:
                self.logger.error(f"❌ Unsupported timeframe: {timeframe}")
                return None

            # Resample and aggregate
            resampled = (
                aggtrades_df.resample(rule)
                .agg({"price": ["first", "last", "min", "max"], "quantity": "sum"})
                .dropna()
            )

            # Flatten column names
            resampled.columns = ["open", "close", "low", "high", "volume"]
            resampled = resampled.reset_index()

            # Keep timestamp in datetime format (don't convert to milliseconds here)
            # The main processing will handle the conversion to milliseconds

            self.logger.info(
                f"✅ Converted aggtrades to {timeframe} klines: {len(resampled)} rows",
            )
            return resampled

        except Exception as e:
            self.logger.exception(f"❌ Failed to convert aggtrades to klines: {e}")
            return None

    async def _fill_missing_values(self, unified: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in unified dataset."""
        try:
            # Track what we're filling
            filled_columns = []

            # Fill numeric columns with 0 (except timestamp, date columns, and trade data columns)
            numeric_columns = unified.select_dtypes(include=[np.number]).columns
            trade_data_columns = [
                "trade_volume",
                "trade_count",
                "avg_price",
                "min_price",
                "max_price",
                "volume_ratio",
                "funding_rate",
            ]

            for col in numeric_columns:
                if col not in ["timestamp", "year", "month", "day"]:
                    missing_count = unified[col].isna().sum()
                    if missing_count > 0:
                        # For trade data columns, only fill if they're completely missing (not just some values)
                        if col in trade_data_columns:
                            # Check if this column has any non-zero values
                            non_zero_count = (unified[col] > 0).sum()
                            if non_zero_count == 0:
                                # No trade data exists, fill with 0
                                unified[col] = unified[col].fillna(0)
                                filled_columns.append(
                                    f"{col} ({missing_count} values - no trade data)",
                                )
                            else:
                                # Trade data exists, only fill the truly missing values
                                unified[col] = unified[col].fillna(0)
                                filled_columns.append(
                                    f"{col} ({missing_count} values - preserving trade data)",
                                )
                        else:
                            # Non-trade data columns, fill normally
                            unified[col] = unified[col].fillna(0)
                            filled_columns.append(f"{col} ({missing_count} values)")

            # Fill string columns with empty string
            string_columns = unified.select_dtypes(include=["object"]).columns
            for col in string_columns:
                missing_count = unified[col].isna().sum()
                if missing_count > 0:
                    unified[col] = unified[col].fillna("")
                    filled_columns.append(f"{col} ({missing_count} values)")

            if filled_columns:
                self.logger.debug(
                    f"   ✅ Filled missing values in: {', '.join(filled_columns)}",
                )

            return unified

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to fill missing values: {e}")
            return unified


@handle_errors(
    exceptions=(Exception,),
    default_return=False,
    context="step1.5_data_converter",
)
@secure_data_processing
@prevent_data_leakage
@resource_monitor
@memory_efficient
@quality_gate
@circuit_breaker_protection
@handle_errors(
    exceptions=(Exception,),
    default_return=False,
    context="step1_5_data_converter main execution",
)
async def run_step(
    symbol: str,
    exchange: str,
    timeframe: str = "1m",
    data_dir: str = "data_cache",
    force_rerun: bool = False,
) -> bool:
    """Run the unified data converter step.

    This step is secured with comprehensive decorators for:
    - Secure data processing
    - Data leakage prevention
    - Resource monitoring
    - Memory efficiency
    - Quality gates
    - Circuit breaker protection
    - Error handling

    Args:
        symbol: Trading symbol (e.g., "ETHUSDT")
        exchange: Exchange name (e.g., "BINANCE")
        timeframe: Timeframe (e.g., "1m")
        data_dir: Data directory
        force_rerun: Force re-run even if unified data exists

    Returns:
        bool: True if successful, False otherwise

    """
    try:
        # Initialize converter
        converter = UnifiedDataConverter(CONFIG or {})
        await converter.initialize()

        # Execute conversion
        success = await converter.execute(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            force_rerun=force_rerun,
        )

        if success:
            # Log success information
            unified_path = converter.get_unified_data_path(symbol, exchange, timeframe)
            config_path = converter.get_unified_config_path(symbol, exchange, timeframe)

            system_logger.info("✅ Step 1.5 completed successfully")
            system_logger.info(f"📁 Unified dataset: {unified_path}")
            system_logger.info(f"📁 Configuration: {config_path}")

        return success

    except Exception as e:
        system_logger.exception(f"❌ Step 1.5 failed: {e}")
        return False


if __name__ == "__main__":
    # Parse command line arguments
    import asyncio

    async def main() -> None:
        # Get command line arguments
        if len(sys.argv) >= 4:
            symbol = sys.argv[1]
            exchange = sys.argv[2]
            timeframe = sys.argv[3]
            data_dir = sys.argv[4] if len(sys.argv) > 4 else "data_cache"
            force_rerun = len(sys.argv) > 5 and sys.argv[5].lower() == "true"
        else:
            return


        success = await run_step(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            data_dir=data_dir,
            force_rerun=force_rerun,
        )

        if success:
            pass
        else:
            pass

        # Clean up memory to prevent segmentation fault
        import gc

        gc.collect()

    # Use a more robust approach to prevent segmentation fault
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception:
        pass
    finally:
        # Final cleanup
        import gc

        gc.collect()
