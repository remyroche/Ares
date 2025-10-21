"""
Enhanced Artifact Manager with Advanced Features

This module provides a comprehensive artifact management system with:
- Multiple storage backends (filesystem, cloud, database)
- Intelligent compression and optimization
- Security and access control
- Performance monitoring
- Data lifecycle management
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
import threading

# Optional dependencies
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
    NUMPY_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    NUMPY_AVAILABLE = False

try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from .logger import system_logger
from .common_operations import ensure_directory


class StorageBackend(Enum):
    """Available storage backends."""
    FILESYSTEM = "filesystem"
    S3 = "s3"
    REDIS = "redis"
    MEMORY = "memory"


class CompressionType(Enum):
    """Available compression types."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"
    PARQUET = "parquet"


@dataclass
class ArtifactMetadata:
    """Rich metadata for artifacts."""
    id: str
    name: str
    type: str
    size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    lineage: List[str] = field(default_factory=list)
    tags: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    checksum: str = ""
    encryption_key_id: Optional[str] = None


@dataclass
class StorageResult:
    """Result of storage operation."""
    success: bool
    key: str
    size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    duration_ms: float
    error: Optional[str] = None


@dataclass
class ArtifactConfig:
    """Configuration for the artifact manager."""
    # Storage settings
    backend: StorageBackend = StorageBackend.FILESYSTEM
    base_path: str = "artifacts"
    max_cache_size_mb: int = 1024
    compression_type: CompressionType = CompressionType.GZIP
    
    # Security settings
    encryption_enabled: bool = False
    encryption_key: Optional[str] = None
    
    # Performance settings
    max_workers: int = 4
    chunk_size_mb: int = 10
    enable_streaming: bool = True
    
    # Lifecycle settings
    retention_days: int = 30
    auto_cleanup: bool = True
    archive_after_days: int = 7
    
    # Monitoring settings
    enable_metrics: bool = True
    metrics_interval_seconds: int = 60


class ArtifactBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    async def store(self, key: str, data: bytes, metadata: dict) -> StorageResult:
        """Store data with the given key."""
        pass
    
    @abstractmethod
    async def retrieve(self, key: str) -> Optional[bytes]:
        """Retrieve data by key."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete data by key."""
        pass
    
    @abstractmethod
    async def list(self, prefix: str = "") -> List[str]:
        """List all keys with optional prefix."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass


class FilesystemBackend(ArtifactBackend):
    """Filesystem-based storage backend."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
    
    async def store(self, key: str, data: bytes, metadata: dict) -> StorageResult:
        start_time = time.time()
        file_path = self.base_path / f"{key}.artifact"
        
        try:
            with self._lock:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, 'wb') as f:
                    f.write(data)
            
            duration_ms = (time.time() - start_time) * 1000
            return StorageResult(
                success=True,
                key=key,
                size_bytes=len(data),
                compressed_size_bytes=len(data),
                compression_ratio=1.0,
                duration_ms=duration_ms
            )
        except Exception as e:
            return StorageResult(
                success=False,
                key=key,
                size_bytes=0,
                compressed_size_bytes=0,
                compression_ratio=0.0,
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    async def retrieve(self, key: str) -> Optional[bytes]:
        file_path = self.base_path / f"{key}.artifact"
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'rb') as f:
                return f.read()
        except Exception:
            return None
    
    async def delete(self, key: str) -> bool:
        file_path = self.base_path / f"{key}.artifact"
        try:
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception:
            return False
    
    async def list(self, prefix: str = "") -> List[str]:
        try:
            pattern = f"{prefix}*.artifact" if prefix else "*.artifact"
            files = list(self.base_path.glob(pattern))
            return [f.stem for f in files]
        except Exception:
            return []
    
    async def exists(self, key: str) -> bool:
        file_path = self.base_path / f"{key}.artifact"
        return file_path.exists()


class S3Backend(ArtifactBackend):
    """AWS S3-based storage backend."""
    
    def __init__(self, bucket: str, region: str = "us-east-1", credentials: Optional[dict] = None):
        if not AWS_AVAILABLE:
            raise ImportError("boto3 is required for S3 backend")
        
        self.bucket = bucket
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region, **credentials or {})
    
    async def store(self, key: str, data: bytes, metadata: dict) -> StorageResult:
        start_time = time.time()
        try:
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                Metadata=metadata
            )
            
            duration_ms = (time.time() - start_time) * 1000
            return StorageResult(
                success=True,
                key=key,
                size_bytes=len(data),
                compressed_size_bytes=len(data),
                compression_ratio=1.0,
                duration_ms=duration_ms
            )
        except ClientError as e:
            return StorageResult(
                success=False,
                key=key,
                size_bytes=0,
                compressed_size_bytes=0,
                compression_ratio=0.0,
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    async def retrieve(self, key: str) -> Optional[bytes]:
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            return response['Body'].read()
        except ClientError:
            return None
    
    async def delete(self, key: str) -> bool:
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False
    
    async def list(self, prefix: str = "") -> List[str]:
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )
            return [obj['Key'] for obj in response.get('Contents', [])]
        except ClientError:
            return []
    
    async def exists(self, key: str) -> bool:
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False


class CompressionEngine:
    """Intelligent compression engine."""
    
    def __init__(self, compression_type: CompressionType = CompressionType.GZIP):
        self.compression_type = compression_type
        self._strategies = {
            CompressionType.GZIP: self._gzip_compress,
            CompressionType.LZ4: self._lz4_compress,
            CompressionType.ZSTD: self._zstd_compress,
            CompressionType.PARQUET: self._parquet_compress,
        }
    
    def compress(self, data: bytes, data_type: str = "binary") -> tuple[bytes, float]:
        """Compress data and return compressed bytes and ratio."""
        if self.compression_type == CompressionType.NONE:
            return data, 1.0
        
        strategy = self._strategies.get(self.compression_type)
        if not strategy:
            return data, 1.0
        
        try:
            compressed = strategy(data, data_type)
            ratio = len(compressed) / len(data) if len(data) > 0 else 1.0
            return compressed, ratio
        except Exception:
            return data, 1.0
    
    def decompress(self, data: bytes, data_type: str = "binary") -> bytes:
        """Decompress data."""
        if self.compression_type == CompressionType.NONE:
            return data
        
        # Implement decompression logic here
        return data
    
    def _gzip_compress(self, data: bytes, data_type: str) -> bytes:
        import gzip
        return gzip.compress(data)
    
    def _lz4_compress(self, data: bytes, data_type: str) -> bytes:
        try:
            import lz4.frame
            return lz4.frame.compress(data)
        except ImportError:
            return data
    
    def _zstd_compress(self, data: bytes, data_type: str) -> bytes:
        try:
            import zstandard as zstd
            cctx = zstd.ZstdCompressor()
            return cctx.compress(data)
        except ImportError:
            return data
    
    def _parquet_compress(self, data: bytes, data_type: str) -> bytes:
        if data_type == "dataframe" and PANDAS_AVAILABLE:
            # Convert bytes back to DataFrame and compress as Parquet
            # This is a simplified example
            return data
        return data


class SecurityManager:
    """Security and encryption manager."""
    
    def __init__(self, encryption_key: Optional[str] = None):
        self.encryption_key = encryption_key
        self.encryption_enabled = encryption_key is not None
    
    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data if encryption is enabled."""
        if not self.encryption_enabled:
            return data
        
        # Implement encryption logic here
        # For now, return data as-is
        return data
    
    def decrypt(self, data: bytes) -> bytes:
        """Decrypt data if encryption is enabled."""
        if not self.encryption_enabled:
            return data
        
        # Implement decryption logic here
        # For now, return data as-is
        return data


class MetricsCollector:
    """Metrics collection for monitoring."""
    
    def __init__(self):
        self.metrics = {
            'artifacts_stored': 0,
            'artifacts_retrieved': 0,
            'total_storage_bytes': 0,
            'total_compressed_bytes': 0,
            'average_compression_ratio': 0.0,
            'average_access_latency_ms': 0.0,
            'error_count': 0
        }
        self._lock = threading.Lock()
    
    def record_storage(self, size_bytes: int, compressed_bytes: int, duration_ms: float):
        """Record storage metrics."""
        with self._lock:
            self.metrics['artifacts_stored'] += 1
            self.metrics['total_storage_bytes'] += size_bytes
            self.metrics['total_compressed_bytes'] += compressed_bytes
            
            # Update average compression ratio
            if self.metrics['total_storage_bytes'] > 0:
                self.metrics['average_compression_ratio'] = (
                    self.metrics['total_compressed_bytes'] / self.metrics['total_storage_bytes']
                )
    
    def record_retrieval(self, duration_ms: float):
        """Record retrieval metrics."""
        with self._lock:
            self.metrics['artifacts_retrieved'] += 1
            # Update average latency
            total_retrievals = self.metrics['artifacts_retrieved']
            current_avg = self.metrics['average_access_latency_ms']
            self.metrics['average_access_latency_ms'] = (
                (current_avg * (total_retrievals - 1) + duration_ms) / total_retrievals
            )
    
    def record_error(self):
        """Record error occurrence."""
        with self._lock:
            self.metrics['error_count'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        with self._lock:
            return self.metrics.copy()


class EnhancedArtifactManager:
    """Enhanced artifact manager with advanced features."""
    
    def __init__(self, config: ArtifactConfig):
        self.config = config
        self.logger = system_logger.getChild("EnhancedArtifactManager")
        
        # Initialize components
        self.backend = self._create_backend()
        self.compression_engine = CompressionEngine(config.compression_type)
        self.security_manager = SecurityManager(config.encryption_key)
        self.metrics_collector = MetricsCollector()
        
        # Cache for frequently accessed artifacts
        self.cache = {}
        self.cache_size_bytes = 0
        self.max_cache_size_bytes = config.max_cache_size_mb * 1024 * 1024
        
        # Thread pool for I/O operations
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_workers)
        
        # Metadata store
        self.metadata_store = {}
        
        self.logger.info(f"Enhanced Artifact Manager initialized with {config.backend.value} backend")
    
    def _create_backend(self) -> ArtifactBackend:
        """Create storage backend based on configuration."""
        if self.config.backend == StorageBackend.FILESYSTEM:
            return FilesystemBackend(self.config.base_path)
        elif self.config.backend == StorageBackend.S3:
            if not AWS_AVAILABLE:
                raise ImportError("boto3 is required for S3 backend")
            return S3Backend("ares-artifacts", "us-east-1")
        elif self.config.backend == StorageBackend.REDIS:
            if not REDIS_AVAILABLE:
                raise ImportError("redis is required for Redis backend")
            # Implement Redis backend
            raise NotImplementedError("Redis backend not implemented yet")
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")
    
    async def store(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> StorageResult:
        """Store artifact with advanced features."""
        start_time = time.time()
        
        try:
            # Serialize data
            serialized_data = await self._serialize_data(data)
            
            # Compress data
            compressed_data, compression_ratio = self.compression_engine.compress(
                serialized_data, self._detect_data_type(data)
            )
            
            # Encrypt data if enabled
            if self.config.encryption_enabled:
                compressed_data = self.security_manager.encrypt(compressed_data)
            
            # Store in backend
            result = await self.backend.store(key, compressed_data, metadata or {})
            
            # Update metrics
            duration_ms = (time.time() - start_time) * 1000
            self.metrics_collector.record_storage(
                len(serialized_data), len(compressed_data), duration_ms
            )
            
            # Store metadata
            artifact_metadata = ArtifactMetadata(
                id=str(uuid.uuid4()),
                name=key,
                type=self._detect_data_type(data),
                size_bytes=len(serialized_data),
                compressed_size_bytes=len(compressed_data),
                compression_ratio=compression_ratio,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                checksum=hashlib.sha256(serialized_data).hexdigest(),
                tags=metadata or {}
            )
            
            self.metadata_store[key] = artifact_metadata
            
            # Update cache if enabled
            if self._should_cache(len(compressed_data)):
                self._add_to_cache(key, compressed_data)
            
            self.logger.info(f"Stored artifact {key} ({len(serialized_data)} bytes, {compression_ratio:.2f} compression ratio)")
            return result
            
        except Exception as e:
            self.metrics_collector.record_error()
            self.logger.error(f"Failed to store artifact {key}: {e}")
            return StorageResult(
                success=False,
                key=key,
                size_bytes=0,
                compressed_size_bytes=0,
                compression_ratio=0.0,
                duration_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve artifact with caching and optimization."""
        start_time = time.time()
        
        try:
            # Check cache first
            if key in self.cache:
                self.logger.debug(f"Retrieved artifact {key} from cache")
                return await self._deserialize_data(self.cache[key])
            
            # Retrieve from backend
            compressed_data = await self.backend.retrieve(key)
            if compressed_data is None:
                return None
            
            # Decrypt if enabled
            if self.config.encryption_enabled:
                compressed_data = self.security_manager.decrypt(compressed_data)
            
            # Decompress
            serialized_data = self.compression_engine.decompress(
                compressed_data, self._detect_data_type_from_key(key)
            )
            
            # Update metrics
            duration_ms = (time.time() - start_time) * 1000
            self.metrics_collector.record_retrieval(duration_ms)
            
            # Update metadata
            if key in self.metadata_store:
                self.metadata_store[key].accessed_at = datetime.now()
                self.metadata_store[key].access_count += 1
            
            # Add to cache if enabled
            if self._should_cache(len(compressed_data)):
                self._add_to_cache(key, compressed_data)
            
            # Deserialize and return
            data = await self._deserialize_data(serialized_data)
            self.logger.debug(f"Retrieved artifact {key} ({len(serialized_data)} bytes)")
            return data
            
        except Exception as e:
            self.metrics_collector.record_error()
            self.logger.error(f"Failed to retrieve artifact {key}: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete artifact."""
        try:
            # Remove from cache
            if key in self.cache:
                del self.cache[key]
            
            # Remove from backend
            success = await self.backend.delete(key)
            
            # Remove metadata
            if key in self.metadata_store:
                del self.metadata_store[key]
            
            if success:
                self.logger.info(f"Deleted artifact {key}")
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to delete artifact {key}: {e}")
            return False
    
    async def list_artifacts(self, prefix: str = "") -> List[str]:
        """List all artifacts with optional prefix."""
        try:
            return await self.backend.list(prefix)
        except Exception as e:
            self.logger.error(f"Failed to list artifacts: {e}")
            return []
    
    async def get_metadata(self, key: str) -> Optional[ArtifactMetadata]:
        """Get metadata for an artifact."""
        return self.metadata_store.get(key)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.metrics_collector.get_metrics()
    
    async def cleanup_old_artifacts(self) -> int:
        """Clean up old artifacts based on retention policy."""
        if not self.config.auto_cleanup:
            return 0
        
        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
        cleaned_count = 0
        
        for key, metadata in list(self.metadata_store.items()):
            if metadata.created_at < cutoff_date:
                if await self.delete(key):
                    cleaned_count += 1
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old artifacts")
        
        return cleaned_count
    
    async def _serialize_data(self, data: Any) -> bytes:
        """Serialize data to bytes."""
        if isinstance(data, bytes):
            return data
        elif isinstance(data, str):
            return data.encode('utf-8')
        elif PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            buffer = data.to_parquet()
            return buffer
        elif NUMPY_AVAILABLE and isinstance(data, np.ndarray):
            buffer = data.tobytes()
            return buffer
        else:
            # JSON serialization for other types
            json_str = json.dumps(data, default=str)
            return json_str.encode('utf-8')
    
    async def _deserialize_data(self, data: bytes) -> Any:
        """Deserialize bytes back to original data type."""
        try:
            # Try JSON first
            json_str = data.decode('utf-8')
            return json.loads(json_str)
        except (UnicodeDecodeError, json.JSONDecodeError):
            # Return as bytes if not JSON
            return data
    
    def _detect_data_type(self, data: Any) -> str:
        """Detect data type for compression optimization."""
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            return "dataframe"
        elif NUMPY_AVAILABLE and isinstance(data, np.ndarray):
            return "array"
        elif isinstance(data, (dict, list)):
            return "json"
        elif isinstance(data, str):
            return "text"
        else:
            return "binary"
    
    def _detect_data_type_from_key(self, key: str) -> str:
        """Detect data type from key name."""
        if key.endswith('.df') or 'dataframe' in key:
            return "dataframe"
        elif key.endswith('.arr') or 'array' in key:
            return "array"
        elif key.endswith('.json'):
            return "json"
        else:
            return "binary"
    
    def _should_cache(self, size_bytes: int) -> bool:
        """Determine if data should be cached."""
        return (self.cache_size_bytes + size_bytes) <= self.max_cache_size_bytes
    
    def _add_to_cache(self, key: str, data: bytes):
        """Add data to cache with LRU eviction."""
        # Simple LRU implementation
        if key in self.cache:
            del self.cache[key]
        
        # Evict oldest if cache is full
        while (self.cache_size_bytes + len(data)) > self.max_cache_size_bytes and self.cache:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = data
        self.cache_size_bytes += len(data)
    
    async def close(self):
        """Close the artifact manager and cleanup resources."""
        self.thread_pool.shutdown(wait=True)
        self.logger.info("Enhanced Artifact Manager closed")


# Factory function for easy creation
def create_artifact_manager(
    backend: StorageBackend = StorageBackend.FILESYSTEM,
    base_path: str = "artifacts",
    compression: CompressionType = CompressionType.GZIP,
    encryption_key: Optional[str] = None
) -> EnhancedArtifactManager:
    """Create an enhanced artifact manager with the specified configuration."""
    config = ArtifactConfig(
        backend=backend,
        base_path=base_path,
        compression_type=compression,
        encryption_key=encryption_key
    )
    return EnhancedArtifactManager(config)

def get_artifact_manager(config: Optional[Dict[str, Any]] = None) -> EnhancedArtifactManager:
    """Get an instance of the enhanced artifact manager."""
    return create_enhanced_artifact_manager(config)