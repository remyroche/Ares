"""
Optimized Data Manager for Training Pipeline.

This module provides optimized data storage, access, and management utilities
for maximum performance in machine learning workflows.
"""

import pyarrow as pa
import pyarrow.parquet as pq

import pyarrow.dataset as ds
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Iterator
import logging
from pathlib import Path
import json
import pickle

import gzip
from concurrent.futures import ThreadPoolExecutor
import time

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime

import hashlib
import sqlite3
import os
import uuid
import shutil
import numpy as np
import pandas as pd

try:
    from src.utils.pipeline_standards import PipelineStandards
except Exception:
    PipelineStandards = None  # type: ignore

logger = logging.getLogger(__name__)

@dataclass
class DataMetadata:
    """Comprehensive metadata for stored data."""
    id: str
    name: str
    data_type: str  # 'dataframe', 'numpy_array', 'model', 'other'
    format: str  # 'parquet', 'npy', 'pkl', etc.
    compression: str
    shape: Optional[Tuple[int, ...]] = None
    dtypes: Optional[Dict[str, str]] = None
    size_bytes: int = 0
    size_mb: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    checksum: str = ""
    tags: List[str] = field(default_factory=list)
    description: str = ""
    lineage: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for storage."""
        data = {
            'id': self.id,
            'name': self.name,
            'data_type': self.data_type,
            'format': self.format,
            'compression': self.compression,
            'shape': self.shape,
            'dtypes': self.dtypes,
            'size_bytes': self.size_bytes,
            'size_mb': self.size_mb,
            'created_at': self.created_at.isoformat(),
            'modified_at': self.modified_at.isoformat(),
            'version': self.version,
            'checksum': self.checksum,
            'tags': self.tags,
            'description': self.description,
            'lineage': self.lineage,
            'statistics': self.statistics,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'dependencies': self.dependencies,
            'quality_metrics': self.quality_metrics,
            'custom_metadata': self.custom_metadata
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataMetadata':
        """Create metadata from dictionary."""
        # Handle datetime parsing
        created_at = datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now()
        modified_at = datetime.fromisoformat(data['modified_at']) if data.get('modified_at') else datetime.now()
        last_accessed = datetime.fromisoformat(data['last_accessed']) if data.get('last_accessed') else None

        return cls(
            id=data['id'],
            name=data['name'],
            data_type=data['data_type'],
            format=data['format'],
            compression=data['compression'],
            shape=data.get('shape'),
            dtypes=data.get('dtypes'),
            size_bytes=data.get('size_bytes', 0),
            size_mb=data.get('size_mb', 0.0),
            created_at=created_at,
            modified_at=modified_at,
            version=data.get('version', 1),
            checksum=data.get('checksum', ''),
            tags=data.get('tags', []),
            description=data.get('description', ''),
            lineage=data.get('lineage', {}),
            statistics=data.get('statistics', {}),
            access_count=data.get('access_count', 0),
            last_accessed=last_accessed,
            dependencies=data.get('dependencies', []),
            quality_metrics=data.get('quality_metrics', {}),
            custom_metadata=data.get('custom_metadata', {})
        )

    def update_access(self):
        """Update access tracking."""
        self.access_count += 1
        self.last_accessed = datetime.now()

    def add_tag(self, tag: str):
        """Add a tag to the metadata."""
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str):
        """Remove a tag from the metadata."""
        if tag in self.tags:
            self.tags.remove(tag)

    def update_lineage(self, operation: str, inputs: List[str], parameters: Dict[str, Any] = None):
        """Update data lineage information."""
        lineage_entry = {
            'operation': operation,
            'timestamp': datetime.now().isoformat(),
            'inputs': inputs,
            'parameters': parameters or {}
        }

        if 'operations' not in self.lineage:
            self.lineage['operations'] = []

        self.lineage['operations'].append(lineage_entry)

        # Update dependencies
        self.dependencies.extend(inputs)
        self.dependencies = list(set(self.dependencies))  # Remove duplicates

    def compute_quality_metrics(self, data: Any):
        """Compute quality metrics for the data."""
        try:
            if isinstance(data, pd.DataFrame):
                self._compute_dataframe_quality(data)
            elif isinstance(data, np.ndarray):
                self._compute_array_quality(data)
        except Exception as e:
            logger.warning(f"Failed to compute quality metrics: {e}")

    def _compute_dataframe_quality(self, df: pd.DataFrame):
        """Compute quality metrics for DataFrame."""
        self.quality_metrics = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'null_percentage': df.isnull().mean().mean() * 100,
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100 if len(df) > 0 else 0,
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime']).columns)
        }

        # Column-specific metrics
        column_metrics = {}
        for col in df.columns:
            col_data = df[col]
            col_metrics = {
                'null_count': col_data.isnull().sum(),
                'null_percentage': (col_data.isnull().sum() / len(col_data)) * 100,
                'unique_count': col_data.nunique(),
                'unique_percentage': (col_data.nunique() / len(col_data)) * 100
            }

            if col_data.dtype in ['int64', 'float64']:
                col_metrics.update({
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'median': col_data.median()
                })

            column_metrics[col] = col_metrics

        self.quality_metrics['column_metrics'] = column_metrics

    def _compute_array_quality(self, array: np.ndarray):
        """Compute quality metrics for NumPy array."""
        self.quality_metrics = {
            'shape': array.shape,
            'size': array.size,
            'dtype': str(array.dtype),
            'ndim': array.ndim,
            'mean': float(np.mean(array)) if array.size > 0 else 0,
            'std': float(np.std(array)) if array.size > 0 else 0,
            'min': float(np.min(array)) if array.size > 0 else 0,
            'max': float(np.max(array)) if array.size > 0 else 0,
            'zero_percentage': (np.count_nonzero(array == 0) / array.size) * 100 if array.size > 0 else 0,
            'nan_percentage': (np.count_nonzero(np.isnan(array)) / array.size) * 100 if array.size > 0 else 0,
            'inf_percentage': (np.count_nonzero(np.isinf(array)) / array.size) * 100 if array.size > 0 else 0
        }

class MetadataStore:
    """Advanced metadata storage and query system."""

    def __init__(self, db_path: str = None):
        """Initialize metadata store.

        Args:
            db_path: Path to SQLite database file. If None, uses in-memory database.
        """
        self.db_path = db_path or ":memory:"
        self.connection = None
        self.logger = logging.getLogger(f"{__name__}.MetadataStore")

        self._init_database()

    def _init_database(self):
        """Initialize the metadata database."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self._create_tables()
            self.logger.info(f"📊 Metadata store initialized at {self.db_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize metadata store: {e}")
            raise

    def _create_tables(self):
        """Create database tables."""
        cursor = self.connection.cursor()

        # Main metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                data_type TEXT NOT NULL,
                format TEXT NOT NULL,
                compression TEXT,
                shape TEXT,
                dtypes TEXT,
                size_bytes INTEGER,
                size_mb REAL,
                created_at TEXT,
                modified_at TEXT,
                version INTEGER,
                checksum TEXT,
                tags TEXT,
                description TEXT,
                lineage TEXT,
                statistics TEXT,
                access_count INTEGER,
                last_accessed TEXT,
                dependencies TEXT,
                quality_metrics TEXT,
                custom_metadata TEXT
            )
        ''')

        # Tags table for efficient tag queries
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata_tags (
                metadata_id TEXT,
                tag TEXT,
                FOREIGN KEY (metadata_id) REFERENCES metadata (id)
            )
        ''')

        # Indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_name ON metadata(name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_data_type ON metadata(data_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tags ON metadata_tags(tag)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON metadata(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_access_count ON metadata(access_count)')

        self.connection.commit()

    def store_metadata(self, metadata: DataMetadata):
        """Store metadata in the database."""
        try:
            cursor = self.connection.cursor()

            # Insert or replace metadata
            cursor.execute('''
                INSERT OR REPLACE INTO metadata
                (id, name, data_type, format, compression, shape, dtypes, size_bytes, size_mb,
                 created_at, modified_at, version, checksum, tags, description, lineage,
                 statistics, access_count, last_accessed, dependencies, quality_metrics, custom_metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.id,
                metadata.name,
                metadata.data_type,
                metadata.format,
                metadata.compression,
                json.dumps(metadata.shape) if metadata.shape else None,
                json.dumps(metadata.dtypes) if metadata.dtypes else None,
                metadata.size_bytes,
                metadata.size_mb,
                metadata.created_at.isoformat(),
                metadata.modified_at.isoformat(),
                metadata.version,
                metadata.checksum,
                json.dumps(metadata.tags),
                metadata.description,
                json.dumps(metadata.lineage),
                json.dumps(metadata.statistics),
                metadata.access_count,
                metadata.last_accessed.isoformat() if metadata.last_accessed else None,
                json.dumps(metadata.dependencies),
                json.dumps(metadata.quality_metrics),
                json.dumps(metadata.custom_metadata)
            ))

            # Update tags
            cursor.execute('DELETE FROM metadata_tags WHERE metadata_id = ?', (metadata.id,))
            for tag in metadata.tags:
                cursor.execute('INSERT INTO metadata_tags (metadata_id, tag) VALUES (?, ?)',
                             (metadata.id, tag))

            self.connection.commit()
            self.logger.debug(f"💾 Stored metadata for {metadata.name}")

        except Exception as e:
            self.logger.error(f"Failed to store metadata: {e}")
            raise

    def get_metadata(self, metadata_id: str) -> Optional[DataMetadata]:
        """Retrieve metadata by ID."""
        try:
            cursor = self.connection.cursor()
            cursor.execute('SELECT * FROM metadata WHERE id = ?', (metadata_id,))

            row = cursor.fetchone()
            if row:
                # Convert row to dictionary
                columns = [desc[0] for desc in cursor.description]
                data = dict(zip(columns, row))

                # Parse JSON fields
                for field in ['shape', 'dtypes', 'tags', 'lineage', 'statistics',
                            'dependencies', 'quality_metrics', 'custom_metadata']:
                    if data[field]:
                        data[field] = json.loads(data[field])

                return DataMetadata.from_dict(data)

        except Exception as e:
            self.logger.error(f"Failed to retrieve metadata: {e}")

        return None

    def query_metadata(self, filters: Dict[str, Any] = None,
                      order_by: str = 'created_at',
                      limit: int = 100) -> List[DataMetadata]:
        """Query metadata with flexible filters."""
        try:
            cursor = self.connection.cursor()

            query = "SELECT * FROM metadata WHERE 1=1"
            params = []

            # Apply filters
            if filters:
                for key, value in filters.items():
                    if key == 'tags':
                        # Special handling for tags
                        if isinstance(value, list):
                            for tag in value:
                                query += " AND id IN (SELECT metadata_id FROM metadata_tags WHERE tag = ?)"
                                params.append(tag)
                        else:
                            query += " AND id IN (SELECT metadata_id FROM metadata_tags WHERE tag = ?)"
                            params.append(value)
                    elif key == 'data_type':
                        query += " AND data_type = ?"
                        params.append(value)
                    elif key == 'name_like':
                        query += " AND name LIKE ?"
                        params.append(f"%{value}%")
                    elif key == 'size_range':
                        if isinstance(value, tuple) and len(value) == 2:
                            query += " AND size_mb BETWEEN ? AND ?"
                            params.extend(value)
                    elif key == 'date_range':
                        if isinstance(value, tuple) and len(value) == 2:
                            query += " AND created_at BETWEEN ? AND ?"
                            params.extend([d.isoformat() for d in value])

            # Order by
            if order_by in ['created_at', 'modified_at', 'access_count', 'size_mb', 'name']:
                query += f" ORDER BY {order_by} DESC"

            # Limit
            query += " LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)

            results = []
            columns = [desc[0] for desc in cursor.description]

            for row in cursor.fetchall():
                data = dict(zip(columns, row))

                # Parse JSON fields
                for field in ['shape', 'dtypes', 'tags', 'lineage', 'statistics',
                            'dependencies', 'quality_metrics', 'custom_metadata']:
                    if data[field]:
                        data[field] = json.loads(data[field])

                results.append(DataMetadata.from_dict(data))

            return results

        except Exception as e:
            self.logger.error(f"Failed to query metadata: {e}")
            return []

    def update_access(self, metadata_id: str):
        """Update access tracking for metadata."""
        try:
            cursor = self.connection.cursor()
            cursor.execute('''
                UPDATE metadata
                SET access_count = access_count + 1,
                    last_accessed = ?
                WHERE id = ?
            ''', (datetime.now().isoformat(), metadata_id))

            self.connection.commit()

        except Exception as e:
            self.logger.error(f"Failed to update access: {e}")

    def delete_metadata(self, metadata_id: str) -> bool:
        """Delete metadata by ID."""
        try:
            cursor = self.connection.cursor()

            # Delete from tags table first
            cursor.execute('DELETE FROM metadata_tags WHERE metadata_id = ?', (metadata_id,))

            # Delete from main table
            cursor.execute('DELETE FROM metadata WHERE id = ?', (metadata_id,))

            deleted = cursor.rowcount > 0
            self.connection.commit()

            if deleted:
                self.logger.info(f"🗑️ Deleted metadata for {metadata_id}")

            return deleted

        except Exception as e:
            self.logger.error(f"Failed to delete metadata: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive metadata statistics."""
        try:
            cursor = self.connection.cursor()

            # Total counts
            cursor.execute('SELECT COUNT(*) FROM metadata')
            total_items = cursor.fetchone()[0]

            cursor.execute('SELECT SUM(size_mb) FROM metadata')
            total_size_mb = cursor.fetchone()[0] or 0

            # Data type distribution
            cursor.execute('SELECT data_type, COUNT(*) FROM metadata GROUP BY data_type')
            data_types = {row[0]: row[1] for row in cursor.fetchall()}

            # Tag distribution
            cursor.execute('SELECT tag, COUNT(*) FROM metadata_tags GROUP BY tag ORDER BY COUNT(*) DESC LIMIT 10')
            top_tags = {row[0]: row[1] for row in cursor.fetchall()}

            # Access statistics
            cursor.execute('SELECT SUM(access_count) FROM metadata')
            total_accesses = cursor.fetchone()[0] or 0

            return {
                'total_items': total_items,
                'total_size_mb': total_size_mb,
                'data_type_distribution': data_types,
                'top_tags': top_tags,
                'total_accesses': total_accesses,
                'average_accesses_per_item': total_accesses / total_items if total_items > 0 else 0
            }

        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {}

    def cleanup_old_metadata(self, days_old: int = 30) -> int:
        """Clean up metadata older than specified days."""
        try:
            cursor = self.connection.cursor()

            cutoff_date = (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) -
                          pd.Timedelta(days=days_old)).isoformat()

            # Delete old metadata (only if access_count == 0)
            cursor.execute('''
                DELETE FROM metadata
                WHERE created_at < ? AND access_count = 0
            ''', (cutoff_date,))

            deleted_count = cursor.rowcount
            self.connection.commit()

            if deleted_count > 0:
                self.logger.info(f"🧹 Cleaned up {deleted_count} old metadata entries")

            return deleted_count

        except Exception as e:
            self.logger.error(f"Failed to cleanup old metadata: {e}")
            return 0

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

class OptimizedDataManager:
    """Optimized data manager with efficient storage and access patterns."""

    def __init__(self, base_path: str = "data_cache", compression: str = "snappy",
                 chunk_size: int = 100000, enable_parallel: bool = True,
                 enable_metadata_tracking: bool = True, metadata_db_path: str = None,
                 exchange: Optional[str] = None, asset: Optional[str] = None, path_type: str = "processed_data", **path_kwargs):
        """Initialize optimized data manager.

        Args:
            base_path: Base directory for data storage
            compression: Compression algorithm ('snappy', 'gzip', 'lz4', 'zstd')
            chunk_size: Chunk size for processing
            enable_parallel: Whether to use parallel processing
            enable_metadata_tracking: Whether to track comprehensive metadata
            metadata_db_path: Path to metadata database (None for in-memory)
        """
        # Align to PipelineStandards.build_path() if exchange/asset provided
        resolved_base = None
        try:
            from .pipeline_standards import PipelineStandards
            if exchange and asset:
                resolved = PipelineStandards.build_path(path_type, exchange, asset, **path_kwargs)
                # Trim known subfolders like processed to get the root for this manager
                resolved_base = Path(resolved).parent if path_type in {"processed_data", "reports", "backup", "temp"} else Path(resolved)
        except Exception:
            resolved_base = None

        self.base_path = Path(resolved_base) if resolved_base else Path(base_path)
        self.compression = compression
        self.chunk_size = chunk_size
        self.enable_parallel = enable_parallel
        self.enable_metadata_tracking = enable_metadata_tracking

        # Create directories
        self.base_path.mkdir(parents=True, exist_ok=True)
        (self.base_path / "processed").mkdir(parents=True, exist_ok=True)
        (self.base_path / "features").mkdir(parents=True, exist_ok=True)
        (self.base_path / "models").mkdir(parents=True, exist_ok=True)
        (self.base_path / "cache").mkdir(parents=True, exist_ok=True)
        (self.base_path / "wal").mkdir(exist_ok=True)
        (self.base_path / "wal_archive").mkdir(exist_ok=True)

        # Initialize M1 optimizations
        try:
            from .m1_memory_optimizer import get_m1_memory_optimizer
            from .m1_cpu_optimizer import get_m1_cpu_optimizer
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()
            self.m1_available = True
        except ImportError:
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.m1_available = False

        # Initialize metadata store
        if self.enable_metadata_tracking:
            self.metadata_store = MetadataStore(metadata_db_path)
        else:
            self.metadata_store = None

        self.logger = logger.getChild('OptimizedDataManager')
        self.logger.info(f"🔧 Optimized Data Manager initialized (compression: {compression}, metadata: {enable_metadata_tracking})")

        # Cache for frequently accessed data
        self.cache = {}
        self.cache_metadata = {}

        try:
            self.recover_pending_wal_transactions()
        except Exception as e:
            self.logger.debug(f"WAL recovery skipped: {e}")

    @contextmanager
    def memory_efficient_context(self, operation_name: str = "data_operation"):
        """Context manager for memory-efficient operations."""
        if self.memory_optimizer:
            with self.memory_optimizer.memory_checkpoint(operation_name):
                yield
        else:
            yield

    def optimize_dataframe_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame schema for storage and processing.

        Args:
            df: Input DataFrame

        Returns:
            Optimized DataFrame
        """
        with self.memory_efficient_context("schema_optimization"):
            optimized_df = df.copy()

            # Optimize numeric columns
            for col in optimized_df.select_dtypes(include=[np.number]):
                if optimized_df[col].dtype == np.float64:
                    # Check if float32 is sufficient
                    col_min, col_max = optimized_df[col].min(), optimized_df[col].max()
                    if (col_min >= np.finfo(np.float32).min and
                        col_max <= np.finfo(np.float32).max):
                        optimized_df[col] = optimized_df[col].astype(np.float32)
                elif optimized_df[col].dtype == np.int64:
                    # Use smallest integer type possible
                    col_min, col_max = optimized_df[col].min(), optimized_df[col].max()
                    if col_min >= 0:
                        if col_max <= np.iinfo(np.uint8).max:
                            optimized_df[col] = optimized_df[col].astype(np.uint8)
                        elif col_max <= np.iinfo(np.uint16).max:
                            optimized_df[col] = optimized_df[col].astype(np.uint16)
                        elif col_max <= np.iinfo(np.uint32).max:
                            optimized_df[col] = optimized_df[col].astype(np.uint32)
                    else:
                        if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                            optimized_df[col] = optimized_df[col].astype(np.int8)
                        elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                            optimized_df[col] = optimized_df[col].astype(np.int16)
                        elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                            optimized_df[col] = optimized_df[col].astype(np.int32)

            # Optimize categorical columns
            for col in optimized_df.select_dtypes(include=['object']):
                if optimized_df[col].nunique() / len(optimized_df) < 0.1:  # Less than 10% unique
                    optimized_df[col] = optimized_df[col].astype('category')

            # Enhanced datetime dtype detection and optimization
            for col in optimized_df.columns:
                # Check for various datetime-related dtypes
                col_dtype = optimized_df[col].dtype

                # Handle datetime64 dtypes
                if pd.api.types.is_datetime64_any_dtype(col_dtype):
                    # Convert to consistent datetime64[ns] if not already
                    if col_dtype != 'datetime64[ns]':
                        optimized_df[col] = pd.to_datetime(optimized_df[col], utc=True).dt.tz_convert(None)
                    # Consider converting to datetime64[s] for memory efficiency if precision allows
                    if optimized_df[col].dt.microsecond.eq(0).all() and optimized_df[col].dt.nanosecond.eq(0).all():
                        if optimized_df[col].dt.second.eq(0).all():
                            optimized_df[col] = optimized_df[col].astype('datetime64[s]')  # Second precision
                        else:
                            optimized_df[col] = optimized_df[col].astype('datetime64[ms]')  # Millisecond precision

                # Handle timedelta dtypes
                elif pd.api.types.is_timedelta64_dtype(col_dtype):
                    # Convert to consistent timedelta64[ns] if not already
                    if col_dtype != 'timedelta64[ns]':
                        optimized_df[col] = pd.to_timedelta(optimized_df[col])

                # Handle object columns that might contain datetime strings
                elif col_dtype == 'object':
                    # Sample values to detect datetime patterns
                    sample_values = optimized_df[col].dropna().head(10)
                    if len(sample_values) > 0:
                        try:
                            # Try to infer datetime format
                            pd.to_datetime(sample_values, infer_datetime_format=True, errors='coerce')
                            # If successful, convert the column
                            optimized_df[col] = pd.to_datetime(optimized_df[col], infer_datetime_format=True, errors='coerce')
                        except (ValueError, TypeError):
                            pass  # Not datetime-like, leave as object

            # Additional datetime optimizations
            for col in optimized_df.select_dtypes(include=['datetime64']):
                # Remove timezone info if present (saves memory)
                if hasattr(optimized_df[col], 'dt') and hasattr(optimized_df[col].dt, 'tz'):
                    if optimized_df[col].dt.tz is not None:
                        optimized_df[col] = optimized_df[col].dt.tz_convert(None)

            return optimized_df

    def save_dataframe_optimized(self, df: pd.DataFrame, filename: str,
                               partition_cols: Optional[List[str]] = None,
                               **kwargs) -> str:
        """Save DataFrame with optimization.

        Args:
            df: DataFrame to save
            filename: Filename without extension
            partition_cols: Columns to partition by

        Returns:
            Path to saved file
        """
        filepath = self.base_path / "processed" / f"{filename}.parquet"

        with self.memory_efficient_context("dataframe_save"):
            # Optimize schema before saving
            optimized_df = self.optimize_dataframe_schema(df)

            # Prepare PyArrow table
            table = pa.Table.from_pandas(optimized_df)

            # Set up compression
            compression_kwargs = self._get_compression_kwargs()

            if partition_cols and len(partition_cols) > 0:
                # Partitioned save
                pq.write_to_dataset(
                    table,
                    root_path=str(filepath.parent / filename),
                    partition_cols=partition_cols,
                    **compression_kwargs
                )
                return str(filepath.parent / filename)
            else:
                # Single file save
                pq.write_table(table, filepath, **compression_kwargs)
                saved_path = str(filepath)

            # Create and store metadata if enabled
            if self.enable_metadata_tracking and self.metadata_store:
                try:
                    metadata = self._create_metadata_for_data(filename, optimized_df, 'parquet', self.compression, saved_path)
                    self.metadata_store.store_metadata(metadata)
                except Exception as e:
                    self.logger.warning(f"Failed to create metadata: {e}")

            return saved_path

    def load_dataframe_optimized(self, filepath: str,
                               columns: Optional[List[str]] = None,
                               filters: Optional[List] = None) -> pd.DataFrame:
        """Load DataFrame with optimization.

        Args:
            filepath: Path to data file
            columns: Columns to load
            filters: Row filters to apply

        Returns:
            Loaded DataFrame
        """
        filepath = Path(filepath)

        with self.memory_efficient_context("dataframe_load"):
            # Check cache first
            cache_key = str(filepath)
            if cache_key in self.cache:
                cache_time = self.cache_metadata.get(cache_key, {}).get('timestamp', 0)
                file_time = filepath.stat().st_mtime if filepath.exists() else 0

                if cache_time >= file_time:
                    self.logger.debug(f"📋 Cache hit for {filepath}")
                    return self.cache[cache_key]

            # Load with PyArrow for better performance
            try:
                if filepath.is_dir():
                    # Partitioned dataset
                    dataset = pq.ParquetDataset(filepath, filters=filters)
                    table = dataset.read(columns=columns)
                else:
                    # Single file
                    table = pq.read_table(filepath, columns=columns, filters=filters)

                df = table.to_pandas()

                # Cache the result
                if len(df) < 1000000:  # Only cache smaller datasets
                    self.cache[cache_key] = df.copy()
                    self.cache_metadata[cache_key] = {
                        'timestamp': time.time(),
                        'size': len(df),
                        'columns': list(df.columns)
                    }

                # Update metadata access tracking if enabled
                if self.enable_metadata_tracking and self.metadata_store:
                    try:
                        # Extract filename from path for metadata lookup
                        filename = Path(filepath).stem
                        metadata = self.metadata_store.get_metadata(filename)
                        if metadata:
                            metadata.update_access()
                            self.metadata_store.update_access(metadata.id)
                    except Exception as e:
                        self.logger.debug(f"Failed to update metadata access: {e}")

                return df

            except Exception as e:
                self.logger.warning(f"PyArrow load failed: {e}, falling back to pandas")
                return pd.read_parquet(filepath, columns=columns, filters=filters)

    def save_numpy_array_optimized(self, array: np.ndarray, filename: str) -> str:
        """Save NumPy array with optimization.

        Args:
            array: NumPy array to save
            filename: Filename without extension

        Returns:
            Path to saved file
        """
        filepath = self.base_path / "features" / f"{filename}.npy"

        with self.memory_efficient_context("numpy_save"):
            # Use .npy open_memmap for large arrays to preserve .npy format
            if array.nbytes > 100 * 1024 * 1024:  # > 100MB
                self.logger.info(f"💾 Using open_memmap save for large array ({array.nbytes / 1024**2:.1f}MB)")
                mmap = np.lib.format.open_memmap(
                    filename=filepath,
                    mode='w+',
                    dtype=array.dtype,
                    shape=array.shape
                )
                mmap[:] = array
                del mmap  # Ensure file is flushed and closed
            else:
                np.save(filepath, array)

            return str(filepath)

    def load_numpy_array_optimized(self, filepath: str) -> np.ndarray:
        """Load NumPy array with optimization.

        Args:
            filepath: Path to array file

        Returns:
            Loaded NumPy array
        """
        filepath = Path(filepath)

        with self.memory_efficient_context("numpy_load"):
            # Check cache
            cache_key = f"numpy_{filepath}"
            if cache_key in self.cache:
                return self.cache[cache_key]

            # Load with memory mapping for large files
            file_size = filepath.stat().st_size if filepath.exists() else 0
            if file_size > 50 * 1024 * 1024:  # > 50MB
                self.logger.info(f"💾 Using memory-mapped load for large array ({file_size / 1024**2:.1f}MB)")
                array = np.load(filepath, mmap_mode='r')
            else:
                array = np.load(filepath)

            # Cache small arrays
            if array.nbytes < 50 * 1024 * 1024:
                self.cache[cache_key] = array.copy()

            return array

    def save_model_optimized(self, model: Any, filename: str,
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save model with optimization.

        Args:
            model: Model object to save
            filename: Filename without extension
            metadata: Additional metadata

        Returns:
            Path to saved model
        """
        model_path = self.base_path / "models" / f"{filename}.pkl"

        with self.memory_efficient_context("model_save"):
            # Save with compression
            with gzip.open(model_path, 'wb', compresslevel=6) as f:
                pickle.dump({
                    'model': model,
                    'metadata': metadata or {},
                    'timestamp': time.time(),
                    'version': '1.0'
                }, f)

            return str(model_path)

    def load_model_optimized(self, filepath: str) -> Tuple[Any, Dict[str, Any]]:
        """Load model with optimization.

        Args:
            filepath: Path to model file

        Returns:
            Tuple of (model, metadata)
        """
        filepath = Path(filepath)

        with self.memory_efficient_context("model_load"):
            with gzip.open(filepath, 'rb') as f:
                data = pickle.load(f)

            return data['model'], data.get('metadata', {})

    def parallel_data_processing(self, data_list: List[pd.DataFrame],
                               processing_func: Callable[[pd.DataFrame], pd.DataFrame],
                               max_workers: Optional[int] = None) -> List[pd.DataFrame]:
        """Parallel data processing with optimization.

        Args:
            data_list: List of DataFrames to process
            processing_func: Processing function
            max_workers: Maximum number of workers

        Returns:
            List of processed DataFrames
        """
        if not self.enable_parallel or len(data_list) == 1:
            return [processing_func(df) for df in data_list]

        with self.memory_efficient_context("parallel_processing"):
            if self.cpu_optimizer and max_workers is None:
                max_workers = self.cpu_optimizer.get_optimal_workers_for_task("cpu_bound")

            if self.cpu_optimizer:
                results = self.cpu_optimizer.parallel_process(
                    data_list, processing_func, task_type="cpu_bound"
                )
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(processing_func, data_list))

            return results

    def create_data_pipeline(self, steps: List[Callable[[pd.DataFrame], pd.DataFrame]],
                           cache_intermediate: bool = True) -> Callable[[pd.DataFrame], pd.DataFrame]:
        """Create optimized data processing pipeline.

        Args:
            steps: List of processing steps
            cache_intermediate: Whether to cache intermediate results

        Returns:
            Composed processing function
        """
        def pipeline_processor(data: pd.DataFrame) -> pd.DataFrame:
            result = data.copy()

            for i, step in enumerate(steps):
                step_name = f"step_{i}"
                cache_key = f"pipeline_{step_name}_{hash(str(result.values.tobytes())[:16]):x}"

                # Check cache
                if cache_intermediate and cache_key in self.cache:
                    result = self.cache[cache_key]
                    continue

                with self.memory_efficient_context(step_name):
                    result = step(result)

                # Cache intermediate result
                if cache_intermediate and len(result) < 500000:
                    self.cache[cache_key] = result.copy()

            return result

        return pipeline_processor

    def get_data_stats(self) -> Dict[str, Any]:
        """Get data storage statistics."""
        stats = {
            'total_files': 0,
            'total_size_mb': 0,
            'cache_entries': len(self.cache),
            'compression': self.compression
        }

        # Calculate storage usage
        for pattern in ['*.parquet', '*.npy', '*.pkl.gz']:
            for file_path in self.base_path.rglob(pattern):
                stats['total_files'] += 1
                stats['total_size_mb'] += file_path.stat().st_size / (1024**2)

        return stats

    def cleanup_cache(self, max_age_hours: float = 24,
                     max_cache_size_mb: float = 1000) -> int:
        """Clean up old cache entries.

        Args:
            max_age_hours: Maximum age of cache entries
            max_cache_size_mb: Maximum cache size in MB

        Returns:
            Number of entries cleaned
        """
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        cleaned_count = 0
        cache_size_mb = 0

        # Calculate current cache size
        for key, data in self.cache.items():
            if hasattr(data, 'memory_usage'):
                cache_size_mb += data.memory_usage(deep=True).sum() / (1024**2)
            elif hasattr(data, 'nbytes'):
                cache_size_mb += data.nbytes / (1024**2)

        # Clean old entries
        keys_to_remove = []
        for key, metadata in self.cache_metadata.items():
            if current_time - metadata.get('timestamp', 0) > max_age_seconds:
                keys_to_remove.append(key)
            elif cache_size_mb > max_cache_size_mb:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            if key in self.cache:
                del self.cache[key]
            if key in self.cache_metadata:
                del self.cache_metadata[key]
            cleaned_count += 1

        if cleaned_count > 0:
            self.logger.info(f"🧹 Cleaned {cleaned_count} cache entries")

        return cleaned_count

    def _get_compression_kwargs(self) -> Dict[str, Any]:
        """Get compression kwargs for PyArrow."""
        if self.compression == 'snappy':
            return {'compression': 'snappy', 'use_dictionary': True, 'row_group_size': 100000}
        elif self.compression == 'gzip':
            return {'compression': 'gzip', 'compression_level': 6, 'use_dictionary': True, 'row_group_size': 100000}
        elif self.compression == 'lz4':
            return {'compression': 'lz4', 'use_dictionary': True, 'row_group_size': 100000}
        elif self.compression == 'zstd':
            return {'compression': 'zstd', 'compression_level': 3, 'use_dictionary': True, 'row_group_size': 100000}
        else:
            return {'compression': 'snappy', 'use_dictionary': True, 'row_group_size': 100000}

    def _resolve_unified_base_dir(self, exchange: str, asset: str, timeframe: str) -> Path:
        """Resolve base directory for unified partitioned dataset."""
        try:
            if PipelineStandards is not None:
                built = PipelineStandards.build_path('unified_data', exchange, asset, timeframe = timeframe)
                return Path(built)
        except Exception:
            pass
        return self.base_path / 'unified' / exchange.lower() / asset.lower() / timeframe

    def _ensure_partition_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure year/month/day columns exist based on timestamp (int64 ms)."""
        if 'timestamp' not in df.columns:
            return df
        ts = pd.to_numeric(df['timestamp'], errors = 'coerce')
        if ts.max() > 10 ** 12:
            dt = pd.to_datetime(ts, unit = 'ms', utc = True)
        else:
            dt = pd.to_datetime(ts, unit = 's', utc = True)
        result = df.copy()
        result['year'] = dt.dt.year.astype('int16')
        result['month'] = dt.dt.month.astype('int8')
        result['day'] = dt.dt.day.astype('int8')
        return result

    def _wal_root(self) -> Path:
        return self.base_path / 'wal'

    def _wal_archive_root(self) -> Path:
        return self.base_path / 'wal_archive'

    def _wal_begin(self, dataset_name: str, base_dataset_dir: Path, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
        wal_id = f"{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        wal_dir = self._wal_root() / dataset_name / wal_id
        data_dir = wal_dir / 'data'
        data_dir.mkdir(parents = True, exist_ok = True)
        manifest = {
            'wal_id': wal_id,
            'dataset_name': dataset_name,
            'base_dataset_dir': str(base_dataset_dir),
            'created_at': datetime.utcnow().isoformat(),
            'status': 'open',
            'files': [],
            'config': config or {}
        }
        manifest_path = wal_dir / 'manifest.json'
        with open(manifest_path, 'w', encoding = 'utf-8') as f:
            json.dump(manifest, f, indent = 2)
        return {'wal_id': wal_id, 'wal_dir': wal_dir, 'data_dir': data_dir, 'manifest_path': manifest_path, 'base_dataset_dir': base_dataset_dir}

    def _wal_append_file(self, wal_ctx: Dict[str, Any], tmp_path: Path, final_rel_path: Path) -> None:
        try:
            with open(wal_ctx['manifest_path'], 'r', encoding = 'utf-8') as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}
        files = manifest.get('files', []) if isinstance(manifest, dict) else []
        files.append({'tmp_path': str(tmp_path), 'final_rel_path': str(final_rel_path)})
        if isinstance(manifest, dict):
            manifest['files'] = files
        else:
            manifest = {'files': files}
        with open(wal_ctx['manifest_path'], 'w', encoding = 'utf-8') as f:
            json.dump(manifest, f, indent = 2)

    def _wal_commit(self, wal_ctx: Dict[str, Any]) -> int:
        moved = 0
        try:
            with open(wal_ctx['manifest_path'], 'r', encoding = 'utf-8') as f:
                manifest = json.load(f)
        except Exception as e:
            self.logger.warning(f"WAL manifest load failed: {e}")
            return 0
        files = manifest.get('files', [])
        base_dataset_dir = Path(wal_ctx['base_dataset_dir'])
        for entry in files:
            tmp = Path(entry['tmp_path'])
            final_rel = Path(entry['final_rel_path'])
            final_abs = base_dataset_dir / final_rel
            final_abs.parent.mkdir(parents = True, exist_ok = True)
            final_target = final_abs.with_suffix('') if final_abs.suffix == '.tmp' else final_abs
            try:
                tmp.rename(final_target)
                moved += 1
            except Exception as e:
                self.logger.warning(f"Failed to move WAL file {tmp} -> {final_target}: {e}")
        try:
            manifest['status'] = 'committed'
            manifest['committed_at'] = datetime.utcnow().isoformat()
            with open(wal_ctx['manifest_path'], 'w', encoding = 'utf-8') as f:
                json.dump(manifest, f, indent = 2)
            archive_target = self._wal_archive_root() / wal_ctx['wal_dir'].relative_to(self._wal_root())
            archive_target.parent.mkdir(parents = True, exist_ok = True)
            if archive_target.exists():
                archive_target = archive_target.with_name(archive_target.name + f"-committed-{uuid.uuid4().hex[:4]}")
            shutil.move(str(wal_ctx['wal_dir']), str(archive_target))
        except Exception as e:
            self.logger.debug(f"WAL archive move failed: {e}")
        return moved

    def _wal_abort(self, wal_ctx: Dict[str, Any]) -> None:
        try:
            with open(wal_ctx['manifest_path'], 'r', encoding = 'utf-8') as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}
        if isinstance(manifest, dict):
            manifest['status'] = 'aborted'
            manifest['aborted_at'] = datetime.utcnow().isoformat()
            try:
                with open(wal_ctx['manifest_path'], 'w', encoding = 'utf-8') as f:
                    json.dump(manifest, f, indent = 2)
            except Exception:
                pass
        try:
            archive_target = self._wal_archive_root() / wal_ctx['wal_dir'].relative_to(self._wal_root())
            archive_target = archive_target.with_name(archive_target.name + '-aborted')
            archive_target.parent.mkdir(parents = True, exist_ok = True)
            shutil.move(str(wal_ctx['wal_dir']), str(archive_target))
        except Exception as e:
            self.logger.debug(f"WAL abort archive move failed: {e}")

    def recover_pending_wal_transactions(self) -> int:
        """Recover any open WAL transactions by committing their files."""
        root = self._wal_root()
        if not root.exists():
            return 0
        recovered = 0
        for dataset_dir in root.iterdir():
            if not dataset_dir.is_dir():
                continue
            for wal_dir in dataset_dir.iterdir():
                manifest_path = wal_dir / 'manifest.json'
                data_dir = wal_dir / 'data'
                if not manifest_path.exists() or not data_dir.exists():
                    continue
                try:
                    with open(manifest_path, 'r', encoding = 'utf-8') as f:
                        manifest = json.load(f)
                except Exception:
                    continue
                if not isinstance(manifest, dict):
                    continue
                status = manifest.get('status', 'open')
                base_dataset_dir = Path(manifest.get('base_dataset_dir', self.base_path))
                wal_ctx = {'wal_id': wal_dir.name, 'wal_dir': wal_dir, 'data_dir': data_dir, 'manifest_path': manifest_path, 'base_dataset_dir': base_dataset_dir}
                if status in ('open', 'recover'):
                    recovered += self._wal_commit(wal_ctx)
        if recovered > 0:
            self.logger.info(f"🧾 Recovered {recovered} WAL files")
        return recovered

    def stream_write_partitioned_dataset(self, data: Union[pd.DataFrame, Iterator[pd.DataFrame]], dataset_name: str, exchange: str, asset: str, timeframe: str, max_rows_per_file: int = 500000) -> str:
        """Stream write to a hive-partitioned dataset (year/month/day) with WAL.

        Returns base dataset directory path.
        """
        base_dir = self._resolve_unified_base_dir(exchange, asset, timeframe)
        base_dir.mkdir(parents = True, exist_ok = True)
        wal_ctx = self._wal_begin(dataset_name, base_dir, config = {'exchange': exchange, 'asset': asset, 'timeframe': timeframe, 'max_rows_per_file': max_rows_per_file})
        compression_kwargs = self._get_compression_kwargs()
        try:
            if isinstance(data, pd.DataFrame):
                iter_data: Iterator[pd.DataFrame] = iter([data])
            else:
                iter_data = data  # type: ignore
            for chunk in iter_data:
                if chunk is None or len(chunk) == 0:
                    continue
                chunk2 = self._ensure_partition_columns(chunk)
                if len(chunk2) > max_rows_per_file:
                    for i in range(0, len(chunk2), max_rows_per_file):
                        sub = chunk2.iloc[i:i + max_rows_per_file]
                        self._write_chunk_to_wal(sub, wal_ctx, compression_kwargs)
                else:
                    self._write_chunk_to_wal(chunk2, wal_ctx, compression_kwargs)
            self._wal_commit(wal_ctx)
            return str(base_dir)
        except Exception as e:
            self.logger.error(f"Stream write failed: {e}")
            try:
                self._wal_abort(wal_ctx)
            except Exception:
                pass
            raise

    def _write_chunk_to_wal(self, df: pd.DataFrame, wal_ctx: Dict[str, Any], compression_kwargs: Dict[str, Any]) -> None:
        grouped = df.groupby(['year', 'month', 'day'], sort = False, as_index = False)
        for (year, month, day), subdf in grouped:
            table = pa.Table.from_pandas(subdf, preserve_index = False)
            rel_dir = Path(f"year={int(year)}") / f"month={int(month):02d}" / f"day={int(day):02d}"
            part_name = f"part-{uuid.uuid4().hex[:12]}.parquet"
            tmp_path = wal_ctx['data_dir'] / rel_dir / (part_name + '.tmp')
            tmp_path.parent.mkdir(parents = True, exist_ok = True)
            pq.write_table(table, tmp_path, **compression_kwargs)
            final_rel = rel_dir / part_name
            self._wal_append_file(wal_ctx, tmp_path, final_rel)

    def read_partitioned_dataset_stream(self, exchange: str, asset: str, timeframe: str, columns: Optional[List[str]] = None, filter_expr: Any = None, batch_size: int = 65536) -> Iterator[pd.DataFrame]:
        """Stream-read a partitioned dataset (yields DataFrame batches)."""
        base_dir = self._resolve_unified_base_dir(exchange, asset, timeframe)
        if not base_dir.exists():
            return iter(())
        try:
            dataset = ds.dataset(str(base_dir), format = 'parquet', partitioning = 'hive')
            scanner = ds.Scanner.from_dataset(dataset, columns = columns, filter = filter_expr, batch_size = batch_size)
            for batch in scanner.to_batches():
                tbl = pa.Table.from_batches([batch])
                if hasattr(pd, 'ArrowDtype'):
                    yield tbl.to_pandas(types_mapper = pd.ArrowDtype)
                else:
                    yield tbl.to_pandas()
        except Exception as e:
            self.logger.warning(f"Falling back to pandas reader for streaming: {e}")
            files = sorted(base_dir.rglob('*.parquet'))
            for file_path in files:
                df = pd.read_parquet(file_path)
                if columns:
                    df = df[columns]
                for i in range(0, len(df), batch_size):
                    yield df.iloc[i:i + batch_size]

    def _create_metadata_for_data(self, name: str, data: Any, format: str, compression: str,
                                 filepath: str) -> DataMetadata:
        """Create comprehensive metadata for stored data."""
        data_id = self._generate_data_id(name, data)

        # Get file info
        file_path = Path(filepath)
        size_bytes = file_path.stat().st_size if file_path.exists() else 0
        size_mb = size_bytes / (1024 * 1024)

        # Determine data type and extract information
        if isinstance(data, pd.DataFrame):
            data_type = 'dataframe'
            shape = data.shape
            dtypes = {}

            for col, dtype in data.dtypes.items():
                # Enhanced dtype information for datetime columns
                if pd.api.types.is_datetime64_any_dtype(dtype):
                    # Include timezone and precision information
                    tz_info = None
                    if hasattr(data[col], 'dt') and hasattr(data[col].dt, 'tz'):
                        tz_info = str(data[col].dt.tz) if data[col].dt.tz is not None else None

                    # Determine precision
                    precision = 'ns'  # default
                    if dtype == 'datetime64[s]':
                        precision = 's'
                    elif dtype == 'datetime64[ms]':
                        precision = 'ms'
                    elif dtype == 'datetime64[us]':
                        precision = 'us'

                    dtypes[col] = f"datetime64[{precision}]{f'[{tz_info}]' if tz_info else ''}"
                elif pd.api.types.is_timedelta64_dtype(dtype):
                    dtypes[col] = f"timedelta64[{dtype.name.split('[')[1]}"
                else:
                    dtypes[col] = str(dtype)

        elif isinstance(data, np.ndarray):
            data_type = 'numpy_array'
            shape = data.shape
            dtypes = {'dtype': str(data.dtype)}
        else:
            data_type = 'model' if 'model' in name.lower() else 'other'
            shape = None
            dtypes = None

        # Create metadata
        metadata = DataMetadata(
            id=data_id,
            name=name,
            data_type=data_type,
            format=format,
            compression=compression,
            shape=shape,
            dtypes=dtypes,
            size_bytes=size_bytes,
            size_mb=size_mb,
            description=f"Auto-generated metadata for {name}",
            checksum=self._calculate_checksum(filepath) if file_path.exists() else ""
        )

        # Compute quality metrics
        metadata.compute_quality_metrics(data)

        return metadata

    def _calculate_checksum(self, filepath: str) -> str:
        """Calculate SHA256 checksum of file."""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return ""

    def _generate_data_id(self, name: str, data: Any) -> str:
        """Generate unique ID for data based on content hash."""
        if isinstance(data, pd.DataFrame):
            # Hash DataFrame content
            content_str = str(data.values.tobytes()) + str(data.columns.tolist()) + str(data.dtypes.tolist())
        elif isinstance(data, np.ndarray):
            content_str = str(data.tobytes()) + str(data.shape) + str(data.dtype)
        else:
            content_str = str(data)

        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def query_data_by_metadata(self, filters: Dict[str, Any] = None,
                              order_by: str = 'created_at',
                              limit: int = 100) -> List[Dict[str, Any]]:
        """Query stored data by metadata filters."""
        if not self.enable_metadata_tracking or not self.metadata_store:
            return []

        # Get metadata matches
        metadata_list = self.metadata_store.query_metadata(filters, order_by, limit)

        results = []
        for metadata in metadata_list:
            # Check if file still exists
            filepath = self._get_filepath_for_metadata(metadata)
            if filepath.exists():
                results.append({
                    'metadata': metadata,
                    'filepath': str(filepath),
                    'exists': True
                })
            else:
                # File missing - could clean up metadata here
                results.append({
                    'metadata': metadata,
                    'filepath': str(filepath),
                    'exists': False
                })

        return results

    def _get_filepath_for_metadata(self, metadata: DataMetadata) -> Path:
        """Get file path for metadata."""
        if metadata.data_type == 'dataframe':
            return self.base_path / "processed" / f"{metadata.name}.parquet"
        elif metadata.data_type == 'numpy_array':
            return self.base_path / "features" / f"{metadata.name}.npy"
        elif metadata.data_type == 'model':
            return self.base_path / "models" / f"{metadata.name}.pkl"
        else:
            return self.base_path / f"{metadata.name}"

    def update_data_lineage(self, data_name: str, operation: str,
                           inputs: List[str], parameters: Dict[str, Any] = None):
        """Update lineage information for data."""
        if not self.enable_metadata_tracking or not self.metadata_store:
            return

        metadata = self.metadata_store.get_metadata(data_name)
        if metadata:
            metadata.update_lineage(operation, inputs, parameters)
            self.metadata_store.store_metadata(metadata)

    def add_data_tags(self, data_name: str, tags: List[str]):
        """Add tags to data metadata."""
        if not self.enable_metadata_tracking or not self.metadata_store:
            return

        metadata = self.metadata_store.get_metadata(data_name)
        if metadata:
            for tag in tags:
                metadata.add_tag(tag)
            self.metadata_store.store_metadata(metadata)

    def get_data_dependencies(self, data_name: str) -> List[str]:
        """Get dependencies for data."""
        if not self.enable_metadata_tracking or not self.metadata_store:
            return []

        metadata = self.metadata_store.get_metadata(data_name)
        return metadata.dependencies if metadata else []

    def get_data_quality_report(self, data_name: str) -> Dict[str, Any]:
        """Get quality report for data."""
        if not self.enable_metadata_tracking or not self.metadata_store:
            return {}

        metadata = self.metadata_store.get_metadata(data_name)
        if metadata:
            return {
                'quality_metrics': metadata.quality_metrics,
                'last_updated': metadata.modified_at.isoformat(),
                'data_type': metadata.data_type
            }
        return {}

    def cleanup_orphaned_metadata(self) -> int:
        """Clean up metadata for files that no longer exist."""
        if not self.enable_metadata_tracking or not self.metadata_store:
            return 0

        cleaned_count = 0
        all_metadata = self.metadata_store.query_metadata(limit=10000)

        for metadata in all_metadata:
            filepath = self._get_filepath_for_metadata(metadata)
            if not filepath.exists():
                self.metadata_store.delete_metadata(metadata.id)
                cleaned_count += 1

        if cleaned_count > 0:
            self.logger.info(f"🧹 Cleaned up {cleaned_count} orphaned metadata entries")

        return cleaned_count

    def get_metadata_statistics(self) -> Dict[str, Any]:
        """Get comprehensive metadata statistics."""
        if not self.enable_metadata_tracking or not self.metadata_store:
            return {'metadata_tracking': False}

        stats = self.metadata_store.get_statistics()
        stats['metadata_tracking'] = True

        # Add file system statistics
        file_stats = self.get_data_stats()
        stats.update({
            'files_with_metadata': file_stats.get('total_files', 0),
            'orphaned_metadata_check': self.cleanup_orphaned_metadata()
        })

        return stats

# Global instance
_optimized_data_manager = None

def get_optimized_data_manager() -> OptimizedDataManager:
    """Get global optimized data manager instance."""
    global _optimized_data_manager
    if _optimized_data_manager is None:
        _optimized_data_manager = OptimizedDataManager()
    return _optimized_data_manager

# Convenience functions
def save_dataframe(df: pd.DataFrame, filename: str, **kwargs) -> str:
    """Save DataFrame with optimization."""
    manager = get_optimized_data_manager()
    return manager.save_dataframe_optimized(df, filename, **kwargs)

def load_dataframe(filepath: str, **kwargs) -> pd.DataFrame:
    """Load DataFrame with optimization."""
    manager = get_optimized_data_manager()
    return manager.load_dataframe_optimized(filepath, **kwargs)

def parallel_data_processing(data_list: List[pd.DataFrame],
                           processing_func: Callable[[pd.DataFrame], pd.DataFrame]) -> List[pd.DataFrame]:
    """Parallel data processing."""
    manager = get_optimized_data_manager()
    return manager.parallel_data_processing(data_list, processing_func)

def query_data_by_metadata(filters: Dict[str, Any] = None,
                          order_by: str = 'created_at',
                          limit: int = 100) -> List[Dict[str, Any]]:
    """Query stored data by metadata filters."""
    manager = get_optimized_data_manager()
    return manager.query_data_by_metadata(filters, order_by, limit)

def update_data_lineage(data_name: str, operation: str,
                       inputs: List[str], parameters: Dict[str, Any] = None):
    """Update lineage information for data."""
    manager = get_optimized_data_manager()
    manager.update_data_lineage(data_name, operation, inputs, parameters)

def add_data_tags(data_name: str, tags: List[str]):
    """Add tags to data metadata."""
    manager = get_optimized_data_manager()
    manager.add_data_tags(data_name, tags)

def get_data_dependencies(data_name: str) -> List[str]:
    """Get dependencies for data."""
    manager = get_optimized_data_manager()
    return manager.get_data_dependencies(data_name)

def get_data_quality_report(data_name: str) -> Dict[str, Any]:
    """Get quality report for data."""
    manager = get_optimized_data_manager()
    return manager.get_data_quality_report(data_name)

def get_metadata_statistics() -> Dict[str, Any]:
    """Get comprehensive metadata statistics."""
    manager = get_optimized_data_manager()
    return manager.get_metadata_statistics()
