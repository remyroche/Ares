"""
Data loading utilities for partitioned datasets.

This module provides utilities for loading data from partitioned Parquet datasets
in a memory-efficient manner, supporting both full dataset loading and streaming
for large datasets.
"""

import os
import logging
from typing import Optional, List, Dict, Any, Union
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

from src.utils.logger import system_logger


class PartitionedDataLoader:
    """Utility class for loading data from partitioned Parquet datasets."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or system_logger

    def load_partitioned_data(
        self,
        base_dir: str,
        exchange: str,
        symbol: str,
        data_type: str = "aggtrades",
        timeframe: str = "1m",
        filters: Optional[List] = None,
        columns: Optional[List[str]] = None,
        max_rows: Optional[int] = None,
        use_streaming: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load data from partitioned Parquet dataset.

        Args:
            base_dir: Base directory for partitioned data
            exchange: Exchange name
            symbol: Symbol name
            data_type: Type of data (aggtrades, klines, etc.)
            timeframe: Timeframe for the data
            filters: Additional filters to apply
            columns: Columns to load (None for all)
            max_rows: Maximum number of rows to load (None for all)
            use_streaming: Whether to use streaming for large datasets
            **kwargs: Additional arguments

        Returns:
            DataFrame with the loaded data
        """
        try:
            # Construct the dataset path
            dataset_path = os.path.join(base_dir, f"{data_type}_{exchange}_{symbol}")

            if not os.path.exists(dataset_path):
                raise FileNotFoundError(
                    f"Partitioned dataset not found: {dataset_path}"
                )

            # Build filters
            if filters is None:
                filters = []

            # Add exchange and symbol filters if not already present
            exchange_filter = ("exchange", "==", exchange)
            symbol_filter = ("symbol", "==", symbol)

            if exchange_filter not in filters:
                filters.append(exchange_filter)
            if symbol_filter not in filters:
                filters.append(symbol_filter)

            # Add timeframe filter if applicable
            if timeframe and timeframe != "1m":  # Default timeframe
                timeframe_filter = ("timeframe", "==", timeframe)
                if timeframe_filter not in filters:
                    filters.append(timeframe_filter)

            self.logger.info(f"📁 Loading partitioned data from: {dataset_path}")
            self.logger.info(f"🔍 Applying filters: {filters}")

            if use_streaming and PYARROW_AVAILABLE:
                return self._load_with_pyarrow_streaming(
                    dataset_path, filters, columns, max_rows
                )
            elif PYARROW_AVAILABLE:
                return self._load_with_pyarrow(dataset_path, filters, columns, max_rows)
            else:
                return self._load_with_pandas(dataset_path, filters, columns, max_rows)

        except Exception as e:
            self.logger.error(f"Error loading partitioned data: {e}")
            raise

    def _load_with_pyarrow_streaming(
        self,
        dataset_path: str,
        filters: List,
        columns: Optional[List[str]],
        max_rows: Optional[int],
    ) -> pd.DataFrame:
        """Load data using PyArrow with streaming for large datasets."""
        try:
            # Create dataset
            dataset = ds.dataset(dataset_path, format="parquet")

            # Build filter expression
            filter_expr = self._build_filter_expression(filters)

            # Create scanner
            scanner = dataset.scanner(
                filter=filter_expr,
                columns=columns,
                batch_size=10000,  # Small batch size for streaming
            )

            # Stream data in chunks
            chunks = []
            total_rows = 0

            for batch in scanner.to_batches():
                if max_rows and total_rows >= max_rows:
                    break

                chunk_df = batch.to_pandas()
                chunks.append(chunk_df)
                total_rows += len(chunk_df)

                # Memory management: concatenate chunks periodically
                if len(chunks) >= 10:  # Concatenate every 10 chunks
                    chunks = [pd.concat(chunks, ignore_index=True)]

            # Final concatenation
            if chunks:
                result = pd.concat(chunks, ignore_index=True)
                if max_rows and len(result) > max_rows:
                    result = result.head(max_rows)
            else:
                result = pd.DataFrame()

            self.logger.info(f"✅ Loaded {len(result)} rows using PyArrow streaming")
            return result

        except Exception as e:
            self.logger.warning(
                f"PyArrow streaming failed: {e}, falling back to regular PyArrow"
            )
            return self._load_with_pyarrow(dataset_path, filters, columns, max_rows)

    def _load_with_pyarrow(
        self,
        dataset_path: str,
        filters: List,
        columns: Optional[List[str]],
        max_rows: Optional[int],
    ) -> pd.DataFrame:
        """Load data using PyArrow without streaming."""
        try:
            # Create dataset
            dataset = ds.dataset(dataset_path, format="parquet")

            # Build filter expression
            filter_expr = self._build_filter_expression(filters)

            # Load data
            table = dataset.to_table(filter=filter_expr, columns=columns)
            result = table.to_pandas()

            if max_rows and len(result) > max_rows:
                result = result.head(max_rows)

            self.logger.info(f"✅ Loaded {len(result)} rows using PyArrow")
            return result

        except Exception as e:
            self.logger.warning(f"PyArrow loading failed: {e}, falling back to pandas")
            return self._load_with_pandas(dataset_path, filters, columns, max_rows)

    def _load_with_pandas(
        self,
        dataset_path: str,
        filters: List,
        columns: Optional[List[str]],
        max_rows: Optional[int],
    ) -> pd.DataFrame:
        """Load data using pandas (fallback method)."""
        try:
            # Find all parquet files in the dataset
            parquet_files = list(Path(dataset_path).rglob("*.parquet"))

            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found in {dataset_path}")

            self.logger.info(f"📁 Found {len(parquet_files)} parquet files")

            # Load files one by one
            chunks = []
            total_rows = 0

            for file_path in parquet_files:
                if max_rows and total_rows >= max_rows:
                    break

                try:
                    chunk = pd.read_parquet(file_path, columns=columns)
                    chunks.append(chunk)
                    total_rows += len(chunk)
                except Exception as e:
                    self.logger.warning(f"Failed to load {file_path}: {e}")
                    continue

            if chunks:
                result = pd.concat(chunks, ignore_index=True)
                if max_rows and len(result) > max_rows:
                    result = result.head(max_rows)
            else:
                result = pd.DataFrame()

            self.logger.info(f"✅ Loaded {len(result)} rows using pandas")
            return result

        except Exception as e:
            self.logger.error(f"Pandas loading failed: {e}")
            raise

    def _build_filter_expression(self, filters: List) -> Optional[ds.Expression]:
        """Build PyArrow filter expression from filter list."""
        if not filters or not PYARROW_AVAILABLE:
            return None

        try:
            expressions = []
            for field, op, value in filters:
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
                elif op == "in":
                    expressions.append(ds.field(field).isin(value))

            if expressions:
                return (
                    expressions[0]
                    if len(expressions) == 1
                    else expressions[0] & expressions[1:]
                )
            return None

        except Exception as e:
            self.logger.warning(f"Failed to build filter expression: {e}")
            return None

    def get_available_partitions(
        self, base_dir: str, exchange: str, symbol: str, data_type: str = "aggtrades"
    ) -> List[str]:
        """Get list of available partitions for a dataset."""
        try:
            dataset_path = os.path.join(base_dir, f"{data_type}_{exchange}_{symbol}")

            if not os.path.exists(dataset_path):
                return []

            partitions = []
            for year_dir in os.listdir(dataset_path):
                year_path = os.path.join(dataset_path, year_dir)
                if os.path.isdir(year_path) and year_dir.isdigit():
                    for month_dir in os.listdir(year_path):
                        month_path = os.path.join(year_path, month_dir)
                        if os.path.isdir(month_path) and month_dir.isdigit():
                            partitions.append(f"{year_dir}/{month_dir}")

            return sorted(partitions)

        except Exception as e:
            self.logger.error(f"Error getting available partitions: {e}")
            return []

    def estimate_dataset_size(
        self, base_dir: str, exchange: str, symbol: str, data_type: str = "aggtrades"
    ) -> Dict[str, Any]:
        """Estimate the size of a partitioned dataset."""
        try:
            dataset_path = os.path.join(base_dir, f"{data_type}_{exchange}_{symbol}")

            if not os.path.exists(dataset_path):
                return {"total_rows": 0, "total_size_mb": 0, "partitions": 0}

            total_rows = 0
            total_size_mb = 0
            partition_count = 0

            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.endswith(".parquet"):
                        file_path = os.path.join(root, file)
                        try:
                            # Get file size
                            file_size = os.path.getsize(file_path)
                            total_size_mb += file_size / (1024 * 1024)

                            # Estimate rows from file size (rough estimate)
                            if PYARROW_AVAILABLE:
                                try:
                                    parquet_file = pq.ParquetFile(file_path)
                                    total_rows += parquet_file.metadata.num_rows
                                except:
                                    # Fallback estimate: ~1KB per row
                                    total_rows += int(file_size / 1024)
                            else:
                                # Fallback estimate: ~1KB per row
                                total_rows += int(file_size / 1024)

                            partition_count += 1
                        except Exception as e:
                            self.logger.warning(f"Error processing {file_path}: {e}")

            return {
                "total_rows": total_rows,
                "total_size_mb": round(total_size_mb, 2),
                "partitions": partition_count,
            }

        except Exception as e:
            self.logger.error(f"Error estimating dataset size: {e}")
            return {"total_rows": 0, "total_size_mb": 0, "partitions": 0}


# Convenience function for loading data
def load_partitioned_data(
    exchange: str,
    symbol: str,
    data_type: str = "aggtrades",
    timeframe: str = "1m",
    base_dir: str = "data_cache/parquet",
    max_rows: Optional[int] = None,
    use_streaming: bool = True,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Convenience function to load partitioned data.

    Args:
        exchange: Exchange name
        symbol: Symbol name
        data_type: Type of data (aggtrades, klines, etc.)
        timeframe: Timeframe for the data
        base_dir: Base directory for partitioned data
        max_rows: Maximum number of rows to load
        use_streaming: Whether to use streaming for large datasets
        logger: Logger instance

    Returns:
        DataFrame with the loaded data
    """
    loader = PartitionedDataLoader(logger)
    return loader.load_partitioned_data(
        base_dir=base_dir,
        exchange=exchange,
        symbol=symbol,
        data_type=data_type,
        timeframe=timeframe,
        max_rows=max_rows,
        use_streaming=use_streaming,
    )
