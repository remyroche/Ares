"""Unified artifact and path management for reads/writes.

Provides a single place to resolve data, reports, cache, optimization, and tmp
paths based on configuration. Ensures directories exist before use.

Enhanced with memory optimization and computational efficiency features.
"""

from __future__ import annotations

import gc
import hashlib
import io
import json
import pickle
import time
import threading
import asyncio
import psutil
from contextlib import nullcontext, asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any, Dict, List, Union, Callable
from collections import OrderedDict
from datetime import datetime, timedelta

# Optional dependencies for optimization
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
    NUMPY_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    NUMPY_AVAILABLE = False

try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

from .logger import system_logger
from .common_operations import ensure_directory
from .version_manager import get_version_manager

@dataclass
class SpillStrategy:
	"""Configuration for artifact spill strategies."""
	enable_spilling: bool = True
	spill_threshold_mb: float = 100.0
	compression_type: str = "lz4"  # lz4, gzip, zstd, snappy
	enable_column_pruning: bool = True
	prune_threshold: float = 0.1  # Remove columns with >90% nulls
	enable_parquet_optimization: bool = True
	parquet_compression: str = "snappy"
	enable_lazy_loading: bool = True
	lazy_cache_size_mb: int = 256
	lazy_ttl_hours: int = 24
	enable_version_checks: bool = True

@dataclass
class MemoryProfile:
	"""Memory usage profile for artifacts."""
	artifact_id: str
	memory_usage_mb: float
	spilled: bool = False
	compression_ratio: float = 1.0
	column_count: int = 0
	row_count: int = 0
	access_count: int = 0
	last_accessed: datetime = field(default_factory=datetime.now)
	created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ArtifactManager:
	config: dict
	
	# Memory optimization settings
	max_cache_size_mb: int = 512
	enable_compression: bool = True
	compression_threshold_mb: float = 1.0
	enable_data_type_optimization: bool = True
	enable_aggressive_cleanup: bool = True
	cleanup_interval_seconds: int = 300
	
	# Enhanced spill and profiling settings
	spill_strategy: SpillStrategy = field(default_factory=SpillStrategy)
	enable_memory_profiling: bool = True
	enable_lazy_loading: bool = True
	enable_thread_safety: bool = True

	def __post_init__(self) -> None:
		self.logger = system_logger.getChild("ArtifactManager")
		paths = self.config.get("paths", {}) if isinstance(self.config, dict) else {}
		self._data_dir = Path(paths.get("data_dir", "data"))
		self._reports_dir = Path(paths.get("reports_dir", "reports"))
		self._cache_dir = Path(paths.get("cache_dir", "data_cache"))
		self._optimization_dir = Path(paths.get("optimization_dir", self._data_dir / "optimization"))
		self._tmp_dir = Path(paths.get("tmp_dir", "tmp"))

		# Ensure base directories exist
		for d in (self._data_dir, self._reports_dir, self._cache_dir, self._optimization_dir, self._tmp_dir):
			ensure_directory(str(d))

		# Initialize version manager
		self.version_manager = get_version_manager()
		
		# Initialize memory optimization components
		self._cache = OrderedDict()  # LRU cache
		self._cache_size_bytes = 0
		self._max_cache_size_bytes = self.max_cache_size_mb * 1024 * 1024
		# Thread safety with nullcontext fallback
		if self.enable_thread_safety:
			self._lock = threading.RLock()
			self._lock_context = nullcontext
		else:
			self._lock = None
			self._lock_context = nullcontext
		self._last_cleanup = time.time()
		
		# Track compression method per key
		self._compression_method: Dict[str, str] = {}
		self._performance_metrics = {
			'cache_hits': 0,
			'cache_misses': 0,
			'compression_savings_mb': 0.0,
			'optimization_savings_mb': 0.0,
			'spill_operations': 0,
			'lazy_loads': 0,
			'memory_profiling_enabled': self.enable_memory_profiling
		}
		
		# Enhanced features
		self._memory_profiles: Dict[str, MemoryProfile] = {}
		self._lazy_cache = OrderedDict() if self.enable_lazy_loading else None
		self._lazy_cache_size_bytes = 0
		self._max_lazy_cache_size_bytes = self.spill_strategy.lazy_cache_size_mb * 1024 * 1024
		self._spill_dir = self._cache_dir / "spilled"
		self._spill_dir.mkdir(parents=True, exist_ok=True)
		
		# Initialize KlinesParquetManager for large dataframes
		try:
			from src.utils.data.klines_parquet import KlinesParquetManager
			self._parquet_manager = KlinesParquetManager(str(self._spill_dir))
		except ImportError:
			self._parquet_manager = None
			self.logger.warning("KlinesParquetManager not available - parquet optimization disabled")
		
		# Thread safety
		self._async_lock = asyncio.Lock() if self.enable_thread_safety else None

	def get_data_dir(self, *subdirs: str) -> Path:
		return self._ensure(self._data_dir, *subdirs)

	def get_reports_dir(self, *subdirs: str) -> Path:
		return self._ensure(self._reports_dir, *subdirs)

	def get_cache_dir(self, *subdirs: str) -> Path:
		return self._ensure(self._cache_dir, *subdirs)

	def get_optimization_dir(self, *subdirs: str) -> Path:
		return self._ensure(self._optimization_dir, *subdirs)

	def get_tmp_dir(self, *subdirs: str) -> Path:
		return self._ensure(self._tmp_dir, *subdirs)

	def get_tmp_path(self, filename: str) -> Path:
		return self.get_tmp_dir() / filename

	def _ensure(self, base: Path, *subdirs: str) -> Path:
		path = base
		for s in subdirs:
			path = path / s
		ensure_directory(str(path))
		return path

	def get_versioned_filename(self, base_name: str, extension: str = ".pkl") -> str:
		"""Generate a versioned filename with timestamp.

		Args:
			base_name: Base name for the file
			extension: File extension

		Returns:
			Versioned filename
		"""
		version = self.version_manager.get_ares_version()
		timestamp = self.version_manager.generate_timestamp()
		return f"{base_name}_{version}_{timestamp}{extension}"

	def get_ares_version(self) -> str:
		"""Get the current Ares version.

		Returns:
			Current Ares version
		"""
		return self.version_manager.get_ares_version()
	
	# Memory optimization methods
	def _optimize_dataframe(self, df: Any) -> Any:
		"""Optimize DataFrame data types for memory efficiency."""
		if not PANDAS_AVAILABLE or not isinstance(df, pd.DataFrame) or not self.enable_data_type_optimization:
			return df
		
		optimized_df = df.copy()
		original_memory = optimized_df.memory_usage(deep=True).sum() / (1024 * 1024)
		
		for col in optimized_df.select_dtypes(include=[np.number]).columns:
			col_data = optimized_df[col]
			
			# Compute min/max once per column
			col_min = col_data.min()
			col_max = col_data.max()
			has_nans = col_data.isna().any()
			
			# Handle integer columns with NaNs using nullable dtypes
			if col_data.dtype == np.int64:
				if has_nans:
					# Use pandas nullable dtypes for integers with NaNs
					if col_max < np.iinfo(np.int32).max and col_min > np.iinfo(np.int32).min:
						optimized_df[col] = col_data.astype('Int32')
					elif col_max < np.iinfo(np.int16).max and col_min > np.iinfo(np.int16).min:
						optimized_df[col] = col_data.astype('Int16')
					elif col_max < np.iinfo(np.int8).max and col_min > np.iinfo(np.int8).min:
						optimized_df[col] = col_data.astype('Int8')
				else:
					# No NaNs, can use regular downcast
					if col_max < np.iinfo(np.int32).max and col_min > np.iinfo(np.int32).min:
						optimized_df[col] = col_data.astype(np.int32)
					elif col_max < np.iinfo(np.int16).max and col_min > np.iinfo(np.int16).min:
						optimized_df[col] = col_data.astype(np.int16)
					elif col_max < np.iinfo(np.int8).max and col_min > np.iinfo(np.int8).min:
						optimized_df[col] = col_data.astype(np.int8)
			
			# Handle float columns with safe downcast (NaNs are safe)
			elif col_data.dtype == np.float64:
				if col_max < np.finfo(np.float32).max and col_min > np.finfo(np.float32).min:
					optimized_df[col] = col_data.astype(np.float32)
		
		# Optimize object columns to category if beneficial (with sampling for large DataFrames)
		for col in optimized_df.select_dtypes(include=['object']).columns:
			if len(optimized_df) > 100000:  # Large DataFrame
				# Sample for estimation
				sample_size = min(10000, len(optimized_df))
				sample = optimized_df[col].sample(n=sample_size, random_state=42)
				uniqueness_ratio = sample.nunique() / len(sample)
				if uniqueness_ratio < 0.5 and sample.nunique() < 10000:  # Gate by absolute cardinality
					optimized_df[col] = optimized_df[col].astype('category')
			else:
				uniqueness_ratio = optimized_df[col].nunique() / len(optimized_df)
				if uniqueness_ratio < 0.5 and optimized_df[col].nunique() < 10000:
					optimized_df[col] = optimized_df[col].astype('category')
		
		# Use convert_dtypes for efficient extension dtypes
		try:
			optimized_df = optimized_df.convert_dtypes()
		except Exception:
			pass  # Fallback if convert_dtypes fails
		
		optimized_memory = optimized_df.memory_usage(deep=True).sum() / (1024 * 1024)
		savings = original_memory - optimized_memory
		
		if savings > 0.1:  # Only log if significant savings
			with self._lock_context():
				self._performance_metrics['optimization_savings_mb'] += savings
			self.logger.debug(f"💾 DataFrame optimization: {savings:.1f}MB saved ({savings/original_memory*100:.1f}% reduction)")
		
		return optimized_df
	
	def _compress_data(self, data: bytes, compression_method: str = "lz4") -> tuple[bytes, float, str]:
		"""Compress data if beneficial."""
		if not self.enable_compression or len(data) < (self.compression_threshold_mb * 1024 * 1024):
			return data, 1.0, "none"
		
		try:
			# Use memoryview to reduce copies
			data_view = memoryview(data)
			
			if compression_method == "lz4" and LZ4_AVAILABLE:
				compressed = lz4.frame.compress(data_view, compression_level=1)
				ratio = len(compressed) / len(data) if len(data) > 0 else 1.0
				return compressed, ratio, "lz4"
			elif compression_method == "gzip":
				import gzip
				compressed = gzip.compress(data_view, compresslevel=6)
				ratio = len(compressed) / len(data) if len(data) > 0 else 1.0
				return compressed, ratio, "gzip"
			else:
				# Fallback to gzip if lz4 not available
				import gzip
				compressed = gzip.compress(data_view, compresslevel=6)
				ratio = len(compressed) / len(data) if len(data) > 0 else 1.0
				return compressed, ratio, "gzip"
		except Exception:
			return data, 1.0, "none"
	
	def _decompress_data(self, data: bytes, compression_method: str = "lz4") -> bytes:
		"""Decompress data using the specified method."""
		try:
			if compression_method == "lz4" and LZ4_AVAILABLE:
				return lz4.frame.decompress(data)
			elif compression_method == "gzip":
				import gzip
				return gzip.decompress(data)
			elif compression_method == "none":
				return data
			else:
				# Try to auto-detect
				if LZ4_AVAILABLE:
					try:
						return lz4.frame.decompress(data)
					except:
						import gzip
						return gzip.decompress(data)
				else:
					import gzip
					return gzip.decompress(data)
		except Exception:
			return data
	
	def _add_to_cache(self, key: str, data: bytes):
		"""Add data to LRU cache with size management."""
		with self._lock_context():
			# Remove if already exists and compute size first
			if key in self._cache:
				old_data = self._cache[key]
				old_size = len(old_data)
				del self._cache[key]
				self._cache_size_bytes -= old_size
			
			# Evict items if cache is full
			while (self._cache_size_bytes + len(data)) > self._max_cache_size_bytes and self._cache:
				oldest_key, oldest_data = self._cache.popitem(last=False)
				oldest_size = len(oldest_data)
				self._cache_size_bytes -= oldest_size
				# Remove compression method tracking
				if oldest_key in self._compression_method:
					del self._compression_method[oldest_key]
			
			# Add new item
			self._cache[key] = data
			self._cache_size_bytes += len(data)
	
	def _get_from_cache(self, key: str) -> Optional[bytes]:
		"""Get data from cache."""
		with self._lock_context():
			if key in self._cache:
				# Move to end (most recently used)
				data = self._cache.pop(key)
				self._cache[key] = data
				return data
			return None
	
	def store_optimized(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
		"""Store data with memory optimization."""
		try:
			# Optimize data types
			optimized_data = self._optimize_dataframe(data)
			
			# Serialize data with proper type handling
			if isinstance(optimized_data, bytes):
				serialized_data = optimized_data
				data_type = "raw"
			elif PANDAS_AVAILABLE and isinstance(optimized_data, pd.DataFrame):
				# Use BytesIO buffer for parquet with engine detection
				buf = io.BytesIO()
				try:
					# Try pyarrow first, then fastparquet, then fallback to pickle
					optimized_data.to_parquet(buf, index=False, compression='snappy', engine='pyarrow')
					serialized_data = buf.getvalue()
					data_type = "parquet"
				except ImportError:
					try:
						optimized_data.to_parquet(buf, index=False, compression='snappy', engine='fastparquet')
						serialized_data = buf.getvalue()
						data_type = "parquet"
					except ImportError:
						# Fallback to pickle if no parquet engines available
						serialized_data = pickle.dumps(optimized_data, protocol=pickle.HIGHEST_PROTOCOL)
						data_type = "pickle"
				except Exception:
					# Fallback to pickle if parquet fails for other reasons
					serialized_data = pickle.dumps(optimized_data, protocol=pickle.HIGHEST_PROTOCOL)
					data_type = "pickle"
			elif NUMPY_AVAILABLE and isinstance(optimized_data, np.ndarray):
				# Use np.save for proper dtype/shape preservation
				buf = io.BytesIO()
				np.save(buf, optimized_data, allow_pickle=False)
				serialized_data = buf.getvalue()
				data_type = "npy"
			else:
				serialized_data = pickle.dumps(optimized_data, protocol=pickle.HIGHEST_PROTOCOL)
				data_type = "pickle"
			
			# Compress if beneficial
			compressed_data, compression_ratio, compression_method = self._compress_data(serialized_data)
			
			# Store compression method for later decompression
			self._compression_method[key] = compression_method
			
			# Store to cache
			self._add_to_cache(key, compressed_data)
			
			# Update metrics
			if compression_ratio < 1.0:
				savings = len(serialized_data) - len(compressed_data)
				with self._lock_context():
					self._performance_metrics['compression_savings_mb'] += savings / (1024 * 1024)
			
			# Periodic cleanup
			self._periodic_cleanup()
			
			self.logger.debug(f"Stored optimized artifact {key} ({len(serialized_data)} bytes, {compression_ratio:.2f} compression, {data_type})")
			return True
			
		except Exception as e:
			self.logger.error(f"Failed to store optimized artifact {key}: {e}")
			return False
	
	def retrieve_optimized(self, key: str) -> Optional[Any]:
		"""Retrieve data with cache optimization."""
		try:
			# Check cache first
			cached_data = self._get_from_cache(key)
			if cached_data is not None:
				with self._lock_context():
					self._performance_metrics['cache_hits'] += 1
				
				# Get compression method for this key
				compression_method = self._compression_method.get(key, "lz4")
				
				# Decompress before deserializing
				decompressed_data = self._decompress_data(cached_data, compression_method)
				
				self.logger.debug(f"Retrieved artifact {key} from cache")
				return self._deserialize_data(decompressed_data)
			
			# Cache miss
			with self._lock_context():
				self._performance_metrics['cache_misses'] += 1
			
			return None
			
		except Exception as e:
			self.logger.error(f"Failed to retrieve optimized artifact {key}: {e}")
			return None
	
	def _deserialize_data(self, data: bytes) -> Any:
		"""Deserialize data from bytes with type detection."""
		try:
			# Try to detect data type from header
			if len(data) > 4:
				# Check for numpy format
				if data[:6] == b'\x93NUMPY':
					if NUMPY_AVAILABLE:
						return np.load(io.BytesIO(data), allow_pickle=False)
					else:
						return pickle.loads(data)
				
				# Check for parquet format (more robust detection)
				if (data[:4] == b'PAR1' or 
					data[-4:] == b'PAR1' or 
					b'PAR1' in data[:100]):  # Check first 100 bytes for parquet magic
					if PANDAS_AVAILABLE:
						try:
							return pd.read_parquet(io.BytesIO(data))
						except Exception:
							# Fallback to pickle if parquet reading fails
							return pickle.loads(data)
					else:
						return pickle.loads(data)
			
			# Try pickle first (most common)
			return pickle.loads(data)
		except (pickle.PickleError, EOFError):
			try:
				# Try JSON
				json_str = data.decode('utf-8')
				return json.loads(json_str)
			except (UnicodeDecodeError, json.JSONDecodeError):
				return data
	
	def _periodic_cleanup(self):
		"""Perform periodic memory cleanup with partial eviction."""
		current_time = time.time()
		if current_time - self._last_cleanup > self.cleanup_interval_seconds:
			if self.enable_aggressive_cleanup:
				# Check both cache size and system memory
				system_memory_percent = psutil.virtual_memory().percent
				cache_memory_ratio = self._cache_size_bytes / self._max_cache_size_bytes
				
				if cache_memory_ratio > 0.8 or system_memory_percent > 85:
					# Partial eviction instead of full clear
					target_ratio = 0.6  # Evict to 60% of max
					target_size = int(self._max_cache_size_bytes * target_ratio)
					
					with self._lock_context():
						while self._cache_size_bytes > target_size and self._cache:
							oldest_key, oldest_data = self._cache.popitem(last=False)
							oldest_size = len(oldest_data)
							self._cache_size_bytes -= oldest_size
							# Remove compression method tracking
							if oldest_key in self._compression_method:
								del self._compression_method[oldest_key]
					
					self.logger.debug(f"Partial cache eviction: reduced to {self._cache_size_bytes / (1024*1024):.1f}MB")
				
				# Force garbage collection
				collected = gc.collect()
				if collected > 0:
					self.logger.debug(f"Garbage collection freed {collected} objects")
			
			self._last_cleanup = current_time
	
	def get_performance_metrics(self) -> Dict[str, Any]:
		"""Get performance metrics."""
		with self._lock_context():
			total_requests = self._performance_metrics['cache_hits'] + self._performance_metrics['cache_misses']
			cache_hit_ratio = (
				self._performance_metrics['cache_hits'] / total_requests
				if total_requests > 0 else 0
			)
			
			return {
				'cache_hits': self._performance_metrics['cache_hits'],
				'cache_misses': self._performance_metrics['cache_misses'],
				'cache_hit_ratio': cache_hit_ratio,
				'cache_size_mb': self._cache_size_bytes / (1024 * 1024),
				'max_cache_size_mb': self.max_cache_size_mb,
				'compression_savings_mb': self._performance_metrics['compression_savings_mb'],
				'optimization_savings_mb': self._performance_metrics['optimization_savings_mb'],
				'compressed_artifacts': len(self._compression_method),
				'system_memory_percent': psutil.virtual_memory().percent
			}
	
	def clear_cache(self):
		"""Clear the cache."""
		with self._lock_context():
			self._cache.clear()
			self._cache_size_bytes = 0
			self._compression_method.clear()
		self.logger.debug("Cache cleared")
	
	# Enhanced memory profiling and spill strategies
	def _profile_memory_usage(self, artifact_id: str, data: Any) -> MemoryProfile:
		"""Profile memory usage of an artifact."""
		if not self.enable_memory_profiling:
			return None
		
		memory_usage_mb = 0
		column_count = 0
		row_count = 0
		
		if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
			memory_usage_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
			column_count = len(data.columns)
			row_count = len(data)
		elif NUMPY_AVAILABLE and isinstance(data, np.ndarray):
			memory_usage_mb = data.nbytes / (1024 * 1024)
			row_count = data.shape[0] if len(data.shape) > 0 else 0
			column_count = data.shape[1] if len(data.shape) > 1 else 1
		else:
			# Estimate for other types
			try:
				import sys
				memory_usage_mb = sys.getsizeof(data) / (1024 * 1024)
			except:
				memory_usage_mb = 0
		
		profile = MemoryProfile(
			artifact_id=artifact_id,
			memory_usage_mb=memory_usage_mb,
			column_count=column_count,
			row_count=row_count
		)
		
		self._memory_profiles[artifact_id] = profile
		return profile
	
	def _should_spill_artifact(self, profile: MemoryProfile) -> bool:
		"""Determine if an artifact should be spilled to disk."""
		if not self.spill_strategy.enable_spilling:
			return False
		
		return profile.memory_usage_mb > self.spill_strategy.spill_threshold_mb
	
	def _spill_artifact(self, artifact_id: str, data: Any, profile: MemoryProfile) -> bool:
		"""Spill artifact to disk with optimization."""
		try:
			spill_path = self._spill_dir / f"{artifact_id}.spilled"
			
			if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
				# Use KlinesParquetManager for large DataFrames
				if self._parquet_manager and self.spill_strategy.enable_parquet_optimization:
					# Apply column pruning if enabled
					if self.spill_strategy.enable_column_pruning:
						data = self._prune_columns(data)
					
					# Save as optimized parquet
					data.to_parquet(
						spill_path.with_suffix('.parquet'),
						compression=self.spill_strategy.parquet_compression,
						engine='pyarrow'
					)
					profile.spilled = True
					profile.compression_ratio = 0.3  # Estimate for parquet
				else:
					# Fallback to compressed pickle
					compressed_data, ratio, method = self._compress_data(pickle.dumps(data))
					with open(spill_path, 'wb') as f:
						f.write(compressed_data)
					profile.spilled = True
					profile.compression_ratio = ratio
					# Store compression method
					self._compression_method[artifact_id] = method
			else:
				# Generic spill for other data types
				compressed_data, ratio, method = self._compress_data(pickle.dumps(data))
				with open(spill_path, 'wb') as f:
					f.write(compressed_data)
				profile.spilled = True
				profile.compression_ratio = ratio
				# Store compression method
				self._compression_method[artifact_id] = method
			
			# Update performance metrics
			self._performance_metrics['spill_operations'] += 1
			self.logger.info(f"Spilled artifact {artifact_id} ({profile.memory_usage_mb:.2f}MB -> {profile.memory_usage_mb * profile.compression_ratio:.2f}MB)")
			return True
			
		except Exception as e:
			self.logger.error(f"Failed to spill artifact {artifact_id}: {e}")
			return False
	
	def _prune_columns(self, df: pd.DataFrame) -> pd.DataFrame:
		"""Prune columns with high null ratios."""
		if not self.spill_strategy.enable_column_pruning:
			return df
		
		threshold = self.spill_strategy.prune_threshold
		pruned_df = df.copy()
		
		for col in df.columns:
			null_ratio = df[col].isnull().sum() / len(df)
			if null_ratio > threshold:
				pruned_df = pruned_df.drop(columns=[col])
				self.logger.debug(f"Pruned column {col} (null ratio: {null_ratio:.2%})")
		
		return pruned_df
	
	def _load_spilled_artifact(self, artifact_id: str) -> Optional[Any]:
		"""Load a spilled artifact from disk."""
		try:
			spill_path = self._spill_dir / f"{artifact_id}.spilled"
			parquet_path = self._spill_dir / f"{artifact_id}.parquet"
			
			# Check for parquet file first
			if parquet_path.exists() and PANDAS_AVAILABLE:
				df = pd.read_parquet(parquet_path)
				self._performance_metrics['lazy_loads'] += 1
				return df
			
			# Fallback to spilled file
			if spill_path.exists():
				with open(spill_path, 'rb') as f:
					data = f.read()
				deserialized = self._deserialize_data(data)
				self._performance_metrics['lazy_loads'] += 1
				return deserialized
			
			return None
			
		except Exception as e:
			self.logger.error(f"Failed to load spilled artifact {artifact_id}: {e}")
			return None
	
	# Lazy loading with TTL and version checks
	def _is_lazy_cache_valid(self, artifact_id: str) -> bool:
		"""Check if lazy cache entry is still valid."""
		if not self.enable_lazy_loading or artifact_id not in self._lazy_cache:
			return False
		
		profile = self._memory_profiles.get(artifact_id)
		if not profile:
			return False
		
		# Check TTL
		ttl_hours = self.spill_strategy.lazy_ttl_hours
		if datetime.now() - profile.last_accessed > timedelta(hours=ttl_hours):
			return False
		
		# Check version if enabled
		if self.spill_strategy.enable_version_checks:
			# This would need to be implemented based on your versioning system
			pass
		
		return True
	
	def _add_to_lazy_cache(self, artifact_id: str, data: Any):
		"""Add data to lazy cache with size management."""
		if not self.enable_lazy_loading:
			return
		
		data_size = len(pickle.dumps(data))
		
		# Evict items if cache is full
		while (self._lazy_cache_size_bytes + data_size) > self._max_lazy_cache_size_bytes and self._lazy_cache:
			oldest_key, oldest_data = self._lazy_cache.popitem(last=False)
			oldest_size = len(pickle.dumps(oldest_data))
			self._lazy_cache_size_bytes -= oldest_size
		
		# Add new item
		self._lazy_cache[artifact_id] = data
		self._lazy_cache_size_bytes += data_size
	
	def _get_from_lazy_cache(self, artifact_id: str) -> Optional[Any]:
		"""Get data from lazy cache."""
		if not self.enable_lazy_loading or not self._is_lazy_cache_valid(artifact_id):
			return None
		
		if artifact_id in self._lazy_cache:
			# Move to end (most recently used)
			data = self._lazy_cache.pop(artifact_id)
			self._lazy_cache[artifact_id] = data
			
			# Update access tracking
			profile = self._memory_profiles.get(artifact_id)
			if profile:
				profile.access_count += 1
				profile.last_accessed = datetime.now()
			
			return data
		
		return None
	
	# Context manager for auto-cleanup
	@asynccontextmanager
	async def run_context(self, run_id: str):
		"""Async context manager for automatic run directory cleanup."""
		run_dir = self._cache_dir / f"run_{run_id}"
		run_dir.mkdir(parents=True, exist_ok=True)
		
		try:
			yield run_dir
		finally:
			# Auto-cleanup run directory
			try:
				import shutil
				shutil.rmtree(run_dir, ignore_errors=True)
				self.logger.info(f"Cleaned up run directory: {run_dir}")
			except Exception as e:
				self.logger.warning(f"Failed to cleanup run directory {run_dir}: {e}")
	
	# Synchronous context manager for non-async usage
	@contextmanager
	def run_context_sync(self, run_id: str):
		"""Synchronous context manager for automatic run directory cleanup."""
		run_dir = self._cache_dir / f"run_{run_id}"
		run_dir.mkdir(parents=True, exist_ok=True)
		
		try:
			yield run_dir
		finally:
			# Auto-cleanup run directory
			try:
				import shutil
				shutil.rmtree(run_dir, ignore_errors=True)
				self.logger.info(f"Cleaned up run directory: {run_dir}")
			except Exception as e:
				self.logger.warning(f"Failed to cleanup run directory {run_dir}: {e}")
	
	# Enhanced store method with profiling and spilling
	def store_enhanced(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
		"""Store artifact with enhanced profiling and spill strategies."""
		try:
			# Profile memory usage
			profile = self._profile_memory_usage(key, data)
			
			# Check if we should spill
			if self._should_spill_artifact(profile):
				# Spill to disk
				if self._spill_artifact(key, data, profile):
					# Add to lazy cache for quick access
					self._add_to_lazy_cache(key, data)
					self.logger.info(f"Stored and spilled artifact {key} ({profile.memory_usage_mb:.2f}MB)")
					return True
				else:
					# Fallback to regular storage
					return self.store_optimized(key, data, metadata)
			else:
				# Regular in-memory storage
				return self.store_optimized(key, data, metadata)
				
		except Exception as e:
			self.logger.error(f"Failed to store enhanced artifact {key}: {e}")
			return False
	
	# Enhanced retrieve method with lazy loading
	def retrieve_enhanced(self, key: str) -> Optional[Any]:
		"""Retrieve artifact with lazy loading and spill support."""
		try:
			# Check lazy cache first
			cached_data = self._get_from_lazy_cache(key)
			if cached_data is not None:
				return cached_data
			
			# Check if artifact is spilled
			profile = self._memory_profiles.get(key)
			if profile and profile.spilled:
				# Load from spill
				spilled_data = self._load_spilled_artifact(key)
				if spilled_data is not None:
					# Add back to lazy cache
					self._add_to_lazy_cache(key, spilled_data)
					return spilled_data
			
			# Fallback to regular retrieval
			return self.retrieve_optimized(key)
			
		except Exception as e:
			self.logger.error(f"Failed to retrieve enhanced artifact {key}: {e}")
			return None
	
	# Memory profiling and analytics
	def get_memory_analytics(self) -> Dict[str, Any]:
		"""Get comprehensive memory analytics."""
		total_memory_mb = sum(profile.memory_usage_mb for profile in self._memory_profiles.values())
		spilled_count = sum(1 for profile in self._memory_profiles.values() if profile.spilled)
		avg_compression_ratio = np.mean([p.compression_ratio for p in self._memory_profiles.values() if p.spilled]) if spilled_count > 0 else 1.0
		
		return {
			'total_artifacts': len(self._memory_profiles),
			'total_memory_mb': total_memory_mb,
			'spilled_artifacts': spilled_count,
			'in_memory_artifacts': len(self._memory_profiles) - spilled_count,
			'average_compression_ratio': avg_compression_ratio,
			'cache_utilization': (self._cache_size_bytes / self._max_cache_size_bytes) * 100,
			'lazy_cache_utilization': (self._lazy_cache_size_bytes / self._max_lazy_cache_size_bytes) * 100 if self.enable_lazy_loading else 0,
			'performance_metrics': self._performance_metrics
		}
