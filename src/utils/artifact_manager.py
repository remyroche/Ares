"""Enhanced unified artifact and path management for reads/writes.

Provides a single place to resolve data, reports, cache, optimization, and tmp
paths based on configuration. Ensures directories exist before use.

Enhanced with:
- Robust error handling with retry mechanisms
- Compression support for storage optimization
- Comprehensive metadata tracking and artifact lineage
- Thread safety and concurrent access protection
- Performance monitoring and metrics collection
- Intelligent caching strategies with memory management
- Step-category based artifact organization
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
import uuid
import shutil
from contextlib import nullcontext, asynccontextmanager, contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any, Dict, List, Union, Callable, Set, Tuple
from collections import OrderedDict, defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum

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

try:
    import gzip
    GZIP_AVAILABLE = True
except ImportError:
    GZIP_AVAILABLE = False

from .logger import system_logger
from .common_operations import ensure_directory
from .version_manager import get_version_manager


class CompressionType(Enum):
    """Supported compression algorithms."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"
    AUTO = "auto"  # Automatically choose best compression


class OperationType(Enum):
    """Types of artifact operations."""
    SAVE = "save"
    LOAD = "load"
    DELETE = "delete"
    LIST = "list"


class RetryStrategy(Enum):
    """Retry strategies for failed operations."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retryable_exceptions: Tuple[type, ...] = (OSError, IOError, ConnectionError)


@dataclass
class CompressionConfig:
    """Configuration for artifact compression."""
    enabled: bool = True
    algorithm: CompressionType = CompressionType.AUTO
    min_size_mb: float = 10.0
    compression_level: int = 6
    enable_for_memory: bool = True
    enable_for_disk: bool = True


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    max_memory_mb: float = 2000.0
    cache_memory_mb: float = 500.0
    spill_threshold_mb: float = 150.0
    cleanup_interval_seconds: float = 300.0
    enable_gc_collection: bool = True


@dataclass
class ArtifactMetadata:
    """Enhanced metadata for artifacts."""
    artifact_key: str
    step_name: str
    artifact_type: str
    size_bytes: int
    compressed_size_bytes: Optional[int] = None
    checksum: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)
    compression_used: CompressionType = CompressionType.NONE
    storage_location: str = "memory"
    parent_artifacts: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    version: str = "1.0"


@dataclass
class OperationMetrics:
    """Metrics for artifact operations."""
    operation_type: OperationType
    artifact_key: str
    step_name: str
    duration_seconds: float
    success: bool
    error_message: Optional[str] = None
    retry_count: int = 0
    bytes_processed: int = 0
    compression_ratio: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    artifact_key: str
    data: Any
    metadata: ArtifactMetadata
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    memory_size_mb: float = 0.0


# Step category mapping for organized artifact storage
STEP_CATEGORIES = {
    'data_collection': ['step01', 'data_downloader', 'klines_downloading_processing'],
    'market_analysis': ['step02', 'market_analysis', 'sr_detection', 'regime_discovery'],
    'pre_training': ['step02_5', 'feature_generation', 'pre_training'],
    'models_training': ['step03', 'model_training', 'analyst_models', 'tactician_models'],
    'backtesting': ['step04', 'backtesting', 'real_parameters_optimization']
}


def get_step_category(step_name: str) -> str:
    """Determine the category for a step based on its name."""
    step_name_lower = step_name.lower()
    for category, patterns in STEP_CATEGORIES.items():
        if any(pattern.lower() in step_name_lower for pattern in patterns):
            return category
    return 'pre_training'  # Default fallback

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
	
	# Enhanced configuration options
	compression: CompressionConfig = field(default_factory=CompressionConfig)
	memory: MemoryConfig = field(default_factory=MemoryConfig)
	retry: RetryConfig = field(default_factory=RetryConfig)
	
	# Performance and monitoring
	enable_metrics: bool = True
	enable_caching: bool = True
	
	# Cleanup and maintenance
	enable_health_checks: bool = True
	
	# Enhanced file naming and path management
	include_symbol_in_filename: bool = True
	include_exchange_in_filename: bool = True
	include_datetime_in_filename: bool = True
	include_information_in_filename: bool = True
	include_direction_in_filename: bool = True
	include_model_in_filename: bool = True
	use_joint_parquet_format: bool = True
	generate_json_metadata: bool = True

	def __post_init__(self) -> None:
		self.logger = system_logger.getChild("ArtifactManager")
		paths = self.config.get("paths", {}) if isinstance(self.config, dict) else {}
		self._data_dir = Path(paths.get("data_dir", "data"))
		self._reports_dir = Path(paths.get("reports_dir", "reports"))
		self._cache_dir = Path(paths.get("cache_dir", "data_cache"))
		self._optimization_dir = Path(paths.get("optimization_dir", self._data_dir / "optimization"))
		self._tmp_dir = Path(paths.get("tmp_dir", "tmp"))
		
		# Enhanced artifacts directory with step categories
		self._artifacts_dir = Path("artifacts")
		self._artifacts_dir.mkdir(parents=True, exist_ok=True)

		# Ensure base directories exist
		for d in (self._data_dir, self._reports_dir, self._cache_dir, self._optimization_dir, self._tmp_dir):
			ensure_directory(str(d))

		# Initialize version manager
		self.version_manager = get_version_manager()
		
		# Initialize memory optimization components
		self._cache = OrderedDict()  # LRU cache
		self._cache_lru = deque()  # LRU queue for cache management
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
		
		# Enhanced artifact storage
		self._artifacts: Dict[str, Dict[str, Any]] = {}
		self._metadata: Dict[str, Dict[str, Any]] = {}
		self._artifact_registry: Dict[str, ArtifactMetadata] = {}
		self._current_run_id: Optional[str] = None
		self._run_dir: Optional[Path] = None
		
		# Performance monitoring
		self._metrics: List[OperationMetrics] = []
		self._operation_counts: Dict[OperationType, int] = defaultdict(int)
		self._error_counts: Dict[str, int] = defaultdict(int)
		
		# Memory management
		self._memory_usage_mb = 0.0
		
		# Compression utilities
		self._compressor = self.ArtifactCompressor()
		
		# Enhanced file naming and path management
		self._current_symbol: Optional[str] = None
		self._current_exchange: Optional[str] = None
		self._current_datetime: Optional[datetime] = None
		self._current_information: Optional[str] = None
		self._current_direction: str = "long"  # Default direction
		self._current_model: str = "Analyst"  # Default model
		self._current_step_name: Optional[str] = None
		
		# Initialize KlinesParquetManager for large dataframes
		try:
			from src.utils.data.klines_parquet import KlinesParquetManager
			self._parquet_manager = KlinesParquetManager(str(self._spill_dir))
		except ImportError:
			self._parquet_manager = None
			self.logger.warning("KlinesParquetManager not available - parquet optimization disabled")
		
		# Thread safety
		self._async_lock = asyncio.Lock() if self.enable_thread_safety else None
		
		# Start background tasks
		self._start_background_tasks()
		
		self.reset_run()

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
	
	# ------------------------------------------------------------------
	# Enhanced Context Management for Step-Category Organization
	# ------------------------------------------------------------------
	
	def set_context(self, step_name: str, symbol: Optional[str] = None, exchange: Optional[str] = None, 
	               datetime: Optional[datetime] = None, information: Optional[str] = None,
	               direction: str = "long", model: str = "Analyst") -> None:
		"""Set the current context for file naming and path management."""
		self._current_step_name = step_name
		self._current_symbol = symbol
		self._current_exchange = exchange
		from datetime import datetime as dt
		self._current_datetime = datetime or dt.utcnow()
		self._current_information = information
		self._current_direction = direction
		self._current_model = model
		
		self.logger.info(f"📁 Context set: step={step_name}, symbol={symbol}, exchange={exchange}, datetime={self._current_datetime}, information={information}, direction={direction}, model={model}")

	def _generate_enhanced_filename(self, key: str, step_name: str, file_extension: str = "parquet") -> str:
		"""Generate enhanced filename with information + symbol + exchange + datetime + direction + model."""
		parts = []
		
		# Add information prefix if configured and available
		if self.include_information_in_filename and self._current_information:
			parts.append(self._current_information)
		
		# Add step name
		parts.append(step_name)
		
		# Add key
		parts.append(key)
		
		# Add symbol if configured and available
		if self.include_symbol_in_filename and self._current_symbol:
			parts.append(self._current_symbol)
		
		# Add exchange if configured and available
		if self.include_exchange_in_filename and self._current_exchange:
			parts.append(self._current_exchange)
		
		# Add direction if configured
		if self.include_direction_in_filename and self._current_direction:
			parts.append(self._current_direction)
		
		# Add model if configured
		if self.include_model_in_filename and self._current_model:
			parts.append(self._current_model)
		
		# Add datetime if configured
		if self.include_datetime_in_filename and self._current_datetime:
			datetime_str = self._current_datetime.strftime("%Y%m%d_%H%M%S")
			parts.append(datetime_str)
		
		# Join parts with underscores and add extension
		filename = "_".join(parts) + f".{file_extension}"
		
		self.logger.debug(f"📁 Generated filename: {filename}")
		return filename

	def _get_enhanced_path(self, step_name: str, key: str, file_extension: str = "parquet") -> Path:
		"""Get enhanced path with proper directory structure and filename using step categories."""
		# Determine step category
		step_category = get_step_category(step_name)
		
		# Create directory structure: artifacts/step_category/symbol/exchange/direction/model/step_name/
		path_parts = [self._artifacts_dir, step_category]
		
		if self._current_symbol:
			path_parts.append(self._current_symbol)
		
		if self._current_exchange:
			path_parts.append(self._current_exchange)
		
		if self._current_direction:
			path_parts.append(self._current_direction)
		
		if self._current_model:
			path_parts.append(self._current_model)
		
		path_parts.append(step_name)
		
		step_dir = Path(*path_parts)
		step_dir.mkdir(parents=True, exist_ok=True)
		
		# Generate enhanced filename
		filename = self._generate_enhanced_filename(key, step_name, file_extension)
		full_path = step_dir / filename
		
		self.logger.info(f"📁 Full path created: {full_path}")
		return full_path
	
	def ensure_step_category_directories(self) -> None:
		"""Ensure all step category directories exist."""
		try:
			# Ensure base artifacts directory exists
			self._artifacts_dir.mkdir(parents=True, exist_ok=True)
			
			# Ensure all step category directories exist
			for category in STEP_CATEGORIES.keys():
				category_dir = self._artifacts_dir / category
				category_dir.mkdir(parents=True, exist_ok=True)
				self.logger.debug(f"📁 Ensured directory exists: {category_dir}")
			
			self.logger.info(f"📁 All step category directories ensured in: {self._artifacts_dir}")
		except Exception as e:
			self.logger.error(f"Failed to ensure step category directories: {e}")
			raise

	def _log_file_operation(self, operation: str, path: Path, success: bool = True) -> None:
		"""Log file operations with full path information."""
		status = "✅" if success else "❌"
		self.logger.info(f"{status} {operation}: {path}")
		
		# Also print to console for visibility
		print(f"{status} {operation}: {path}")

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
	
	# ------------------------------------------------------------------
	# Enhanced Artifact Storage and Retrieval (BaseStep Compatible)
	# ------------------------------------------------------------------
	
	def save(self, data: Any, artifact_name: str, 
	         artifact_type: str = "data", 
	         compression: str = "auto",
	         metadata: Optional[Dict] = None) -> str:
		"""
		Save an artifact with automatic CSV generation for small datasets.
		
		This is the default save method used by BaseStep. It automatically:
		- Saves as Parquet (always)
		- Saves as CSV if the data has < 2000 rows (for DataFrames)
		
		Args:
			data: Data to save (DataFrame, dict, model, etc.)
			artifact_name: Name for the artifact
			artifact_type: Type of artifact ("data", "model", "metadata", etc.)
			compression: Compression method for Parquet ("auto", "gzip", "lz4", "none")
			metadata: Additional metadata to store with artifact
			
		Returns:
			Path where the primary artifact (Parquet) was saved
		"""
		try:
			# Save as Parquet (primary format)
			parquet_path = self._save_artifact_to_parquet(
				data=data,
				artifact_name=artifact_name,
				artifact_type=artifact_type,
				compression=compression,
				metadata=metadata
			)
			
			# Automatically save as CSV if it's a DataFrame with < 2000 rows
			if isinstance(data, pd.DataFrame) and len(data) < 2000:
				try:
					csv_path = self._get_enhanced_path(self._current_step_name, artifact_name, "csv")
					data.to_csv(csv_path, index=True)
					self.logger.info(f"📊 Auto-saved CSV artifact (rows < 2000): {artifact_name} -> {csv_path}")
				except Exception as e:
					self.logger.warning(f"Failed to auto-save CSV for {artifact_name}: {e}")
			elif isinstance(data, pd.DataFrame) and len(data) >= 2000:
				self.logger.info(f"📊 Skipping CSV auto-save for {artifact_name} (rows >= 2000: {len(data)})")
			
			return parquet_path
			
		except Exception as e:
			self.logger.error(f"Failed to save artifact {artifact_name}: {e}")
			raise
	
	def _save_artifact_to_parquet(self, data: Any, artifact_name: str, 
	                             artifact_type: str = "data", 
	                             compression: str = "auto",
	                             metadata: Optional[Dict] = None) -> str:
		"""
		Save an artifact as Parquet file.
		
		Args:
			data: Data to save
			artifact_name: Name for the artifact
			artifact_type: Type of artifact
			compression: Compression method
			metadata: Additional metadata
			
		Returns:
			Path where artifact was saved
		"""
		try:
			# Generate enhanced filename and path
			file_extension = "parquet"
			enhanced_path = self._get_enhanced_path(
				self._current_step_name, artifact_name, file_extension
			)
			
			# Ensure directory exists
			enhanced_path.parent.mkdir(parents=True, exist_ok=True)
			
			# Save the data
			if isinstance(data, pd.DataFrame):
				# Save DataFrame as Parquet
				data.to_parquet(enhanced_path, compression='snappy')
			elif isinstance(data, dict):
				# Convert dict to DataFrame and save as Parquet
				df = pd.DataFrame([data])
				df.to_parquet(enhanced_path, compression='snappy')
			else:
				# For other data types, save as pickle first, then convert to parquet
				temp_pickle_path = enhanced_path.with_suffix('.pkl')
				with open(temp_pickle_path, 'wb') as f:
					pickle.dump(data, f)
				# For now, just keep as pickle for non-DataFrame data
				temp_pickle_path.rename(enhanced_path.with_suffix('.pkl'))
				enhanced_path = enhanced_path.with_suffix('.pkl')
			
			# Store metadata
			if metadata is None:
				metadata = {}
			
			metadata.update({
				'artifact_name': artifact_name,
				'artifact_type': artifact_type,
				'file_path': str(enhanced_path),
				'file_size': enhanced_path.stat().st_size if enhanced_path.exists() else 0,
				'timestamp': datetime.now().isoformat(),
				'compression': compression
			})
			
			# Persist metadata
			self._persist_metadata_to_disk(artifact_name, metadata)
			
			self._log_file_operation("Saved artifact", enhanced_path, success=True)
			return str(enhanced_path)
			
		except Exception as e:
			self.logger.error(f"Failed to save Parquet artifact {artifact_name}: {e}")
			raise
	
	def get_artifact(self, artifact_name: str, 
	                artifact_type: str = "data") -> Any:
		"""
		Retrieve an artifact using multiple fallback mechanisms for backward compatibility.
		
		This method implements a comprehensive fallback strategy:
		1. Try step-category structure (artifacts/STEP-CATEGORY/)
		2. Try general artifacts/ directory search
		3. Try with model type and direction variations
		4. Try fuzzy matching for similar names
		
		Args:
			artifact_name: Name of the artifact to retrieve
			artifact_type: Type of artifact to retrieve
			
		Returns:
			Retrieved data or None if not found
		"""
		try:
			# Primary: Try step-category structure
			step_category = get_step_category(self._current_step_name)
			artifact_path = self._find_artifact_in_category(
				step_category, artifact_name, artifact_type
			)
			
			if artifact_path and artifact_path.exists():
				data = self._load_artifact_from_path(artifact_path)
				self._log_file_operation("Retrieved artifact from category", artifact_path, success=True)
				return data
			
			# Fallback 1: Direct artifacts/ directory search
			fallback_path = self._find_artifact_in_fallback(artifact_name, artifact_type)
			if fallback_path and fallback_path.exists():
				data = self._load_artifact_from_path(fallback_path)
				self._log_file_operation("Retrieved artifact from fallback 1", fallback_path, success=True)
				return data
			
			# Fallback 2: Search with model type and direction variations
			variation_path = self._find_artifact_with_variations(artifact_name, artifact_type)
			if variation_path and variation_path.exists():
				data = self._load_artifact_from_path(variation_path)
				self._log_file_operation("Retrieved artifact from variations", variation_path, success=True)
				return data
			
			# Fallback 3: Search in all subdirectories with fuzzy matching
			fuzzy_path = self._find_artifact_fuzzy(artifact_name, artifact_type)
			if fuzzy_path and fuzzy_path.exists():
				data = self._load_artifact_from_path(fuzzy_path)
				self._log_file_operation("Retrieved artifact from fuzzy search", fuzzy_path, success=True)
				return data
			
			self.logger.warning(f"Artifact not found with any fallback method: {artifact_name}")
			return None
			
		except Exception as e:
			self.logger.error(f"Failed to retrieve artifact {artifact_name}: {e}")
			return None
	
	def _find_artifact_fuzzy(self, artifact_name: str, artifact_type: str) -> Optional[Path]:
		"""Find artifact using fuzzy matching across all directories."""
		try:
			if not self._artifacts_dir.exists():
				return None
			
			# Search in all subdirectories
			for file_path in self._artifacts_dir.rglob("*"):
				if file_path.is_file():
					# Check if the filename is similar to the artifact name
					if self._is_similar_name(artifact_name, file_path.stem):
						# Additional check: ensure it's the right type of file
						if self._is_correct_file_type(file_path, artifact_type):
							return file_path
			
			return None
		except Exception as e:
			self.logger.warning(f"Failed to search with fuzzy matching: {e}")
			return None
	
	def _is_correct_file_type(self, file_path: Path, artifact_type: str) -> bool:
		"""Check if the file type matches the expected artifact type."""
		try:
			file_extension = file_path.suffix.lower()
			
			# Map artifact types to expected file extensions
			type_mappings = {
				"data": [".parquet", ".csv", ".json"],
				"model": [".pkl", ".joblib", ".h5", ".onnx"],
				"metadata": [".json", ".yaml", ".yml"],
				"image": [".png", ".jpg", ".jpeg", ".svg"],
				"text": [".txt", ".md", ".log"]
			}
			
			expected_extensions = type_mappings.get(artifact_type, [".parquet", ".csv", ".json", ".pkl"])
			return file_extension in expected_extensions
		except Exception:
			return True  # Default to True if we can't determine
	
	def _find_artifact_in_category(self, step_category: str, artifact_name: str, 
	                              artifact_type: str) -> Optional[Path]:
		"""Find artifact in step-category structure."""
		try:
			category_dir = self._artifacts_dir / step_category
			if not category_dir.exists():
				return None
			
			# Search recursively for the artifact
			for file_path in category_dir.rglob(f"*{artifact_name}*"):
				if file_path.is_file():
					return file_path
			
			return None
		except Exception as e:
			self.logger.warning(f"Failed to search in category {step_category}: {e}")
			return None
	
	def _find_artifact_in_fallback(self, artifact_name: str, artifact_type: str) -> Optional[Path]:
		"""Find artifact in fallback artifacts/ directory with enhanced search patterns."""
		try:
			if not self._artifacts_dir.exists():
				return None
			
			# Define search patterns for better matching
			search_patterns = [
				f"*{artifact_name}*",  # Original pattern
				f"*{artifact_name}*.parquet",  # Specific parquet files
				f"*{artifact_name}*.csv",  # Specific CSV files
				f"*{artifact_name}*.pkl",  # Specific pickle files
				f"*{artifact_name}*.json",  # Specific JSON files
			]
			
			# Search with multiple patterns
			for pattern in search_patterns:
				for file_path in self._artifacts_dir.rglob(pattern):
					if file_path.is_file():
						# Check if the filename actually contains the artifact name
						if artifact_name.lower() in file_path.name.lower():
							return file_path
			
			# Additional search: look for files with similar names (fuzzy matching)
			for file_path in self._artifacts_dir.rglob("*"):
				if file_path.is_file():
					# Check for partial matches
					filename_lower = file_path.name.lower()
					artifact_lower = artifact_name.lower()
					
					# Check if artifact name is contained in filename or vice versa
					if (artifact_lower in filename_lower or 
						filename_lower in artifact_lower or
						self._is_similar_name(artifact_name, file_path.stem)):
						return file_path
			
			return None
		except Exception as e:
			self.logger.warning(f"Failed to search in fallback: {e}")
			return None
	
	def _is_similar_name(self, name1: str, name2: str) -> bool:
		"""Check if two names are similar (for fuzzy matching)."""
		try:
			# Simple similarity check - can be enhanced with more sophisticated algorithms
			name1_clean = name1.lower().replace('_', '').replace('-', '')
			name2_clean = name2.lower().replace('_', '').replace('-', '')
			
			# Check if one is contained in the other
			if name1_clean in name2_clean or name2_clean in name1_clean:
				return True
			
			# Check for common patterns
			common_patterns = ['data', 'model', 'result', 'output', 'input']
			for pattern in common_patterns:
				if pattern in name1_clean and pattern in name2_clean:
					return True
			
			return False
		except Exception:
			return False
	
	def _find_artifact_with_variations(self, artifact_name: str, artifact_type: str) -> Optional[Path]:
		"""Find artifact with model type and direction variations."""
		try:
			if not self._artifacts_dir.exists():
				return None
			
			# Define variations to try
			model_variations = ["Analyst", "Tactician", "analyst", "tactician", ""]
			direction_variations = ["long", "short", "Long", "Short", ""]
			
			# Try different combinations
			for model in model_variations:
				for direction in direction_variations:
					# Create search pattern with variations
					pattern_parts = [artifact_name]
					if model:
						pattern_parts.append(model)
					if direction:
						pattern_parts.append(direction)
					
					# Try different separators
					for separator in ["_", "-", ""]:
						search_pattern = separator.join(pattern_parts)
						
						# Search for files matching this pattern
						for file_path in self._artifacts_dir.rglob(f"*{search_pattern}*"):
							if file_path.is_file():
								return file_path
			
			return None
		except Exception as e:
			self.logger.warning(f"Failed to search with variations: {e}")
			return None
	
	def _load_artifact_from_path(self, path: Path) -> Any:
		"""Load artifact from file path."""
		try:
			if path.suffix == '.parquet':
				return pd.read_parquet(path)
			elif path.suffix == '.csv':
				return pd.read_csv(path, index_col=0)
			elif path.suffix == '.pkl':
				with open(path, 'rb') as f:
					return pickle.load(f)
			elif path.suffix == '.json':
				with open(path, 'r') as f:
					return json.load(f)
			else:
				self.logger.warning(f"Unknown file extension: {path.suffix}")
				return None
		except Exception as e:
			self.logger.error(f"Failed to load artifact from {path}: {e}")
			return None
	
	def _persist_metadata_to_disk(self, artifact_name: str, metadata: Dict[str, Any]) -> None:
		"""Persist metadata to disk for long-term storage with enhanced path management."""
		try:
			# Create enhanced directory structure
			artifacts_dir = self._get_enhanced_path(self._current_step_name, "metadata", "dir").parent
			artifacts_dir.mkdir(parents=True, exist_ok=True)

			# Save metadata to disk with enhanced naming
			metadata_file = self._get_enhanced_path(self._current_step_name, f"{artifact_name}_metadata", "json")
			with open(metadata_file, 'w') as f:
				json.dump(metadata, f, indent=2, default=str)
			self._log_file_operation("Persisted metadata", metadata_file, success=True)
				
			self.logger.debug(f"Persisted metadata to disk for {artifact_name}")
		except Exception as e:
			self.logger.warning(f"Failed to persist metadata to disk: {e}")
	
	# ------------------------------------------------------------------
	# Lifecycle Management
	# ------------------------------------------------------------------
	
	def reset_run(self) -> None:
		"""Clear all artifacts and prepare storage for a new pipeline run."""
		with self._lock:
			# Clear all storage
			self._artifacts.clear()
			self._metadata.clear()
			self._artifact_registry.clear()

			# Reset cache and memory tracking
			self._cache.clear()
			self._cache_lru.clear()
			self._memory_usage_mb = 0.0

			# Reset metrics
			self._metrics.clear()
			self._operation_counts.clear()
			self._error_counts.clear()

			# Create new run directory
			self._current_run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:8]}"
			self._run_dir = self._artifacts_dir / self._current_run_id

			# Clean up old run directory if it exists
			if self._run_dir.exists():
				try:
					shutil.rmtree(self._run_dir, ignore_errors=True)
				except Exception as e:
					self.logger.warning(f"Failed to remove old run directory: {e}")

			# Create new run directory
			self._run_dir.mkdir(parents=True, exist_ok=True)

			# Clean up old runs
			self._cleanup_old_runs()

			self.logger.debug(f"Initialized enhanced artifact storage for run {self._current_run_id}")

	def get_run_id(self) -> Optional[str]:
		"""Get the current run ID."""
		return self._current_run_id

	def get_run_dir(self) -> Optional[Path]:
		"""Get the current run directory."""
		return self._run_dir

	def _cleanup_old_runs(self) -> None:
		"""Remove old run directories beyond the retention limit."""
		try:
			run_dirs = sorted(
				[p for p in self._artifacts_dir.iterdir() if p.is_dir()],
				key=lambda p: p.stat().st_mtime,
				reverse=True,
			)
			for old_run in run_dirs[3:]:  # Keep last 3 runs
				shutil.rmtree(old_run, ignore_errors=True)
		except Exception as exc:
			self.logger.debug("Failed to cleanup old artifact runs: %s", exc, exc_info=True)

	def _start_background_tasks(self) -> None:
		"""Start background maintenance tasks."""
		if self.enable_health_checks:
			# Start cleanup task
			cleanup_task = threading.Thread(
				target=self._background_cleanup,
				daemon=True,
				name="ArtifactManagerCleanup"
			)
			cleanup_task.start()
			self.logger.debug("Background cleanup task started")

	def _background_cleanup(self) -> None:
		"""Background task for periodic cleanup and maintenance."""
		while True:
			try:
				time.sleep(self.cleanup_interval_seconds)
				self._perform_cleanup()
			except Exception as e:
				self.logger.error(f"Background cleanup failed: {e}")

	def _perform_cleanup(self) -> None:
		"""Perform cleanup operations."""
		with self._lock:
			current_time = time.time()

			# Check if cleanup is needed
			if (current_time - self._last_cleanup) < self.cleanup_interval_seconds:
				return

			# Clean up old cache entries
			self._cleanup_cache()

			# Clean up old run directories
			self._cleanup_old_runs()

			# Force garbage collection if enabled
			if self.memory.enable_gc_collection:
				import gc
				gc.collect()

			self._last_cleanup = current_time

	def _cleanup_cache(self) -> None:
		"""Clean up old cache entries based on LRU and memory usage."""
		if not self.enable_caching:
			return

		max_memory_mb = self.memory.cache_memory_mb
		current_memory_mb = self._memory_usage_mb

		# Remove LRU entries if memory usage is high
		while current_memory_mb > max_memory_mb and self._cache_lru:
			oldest_key = self._cache_lru.popleft()
			if oldest_key in self._cache:
				entry = self._cache.pop(oldest_key)
				current_memory_mb -= entry.memory_size_mb
				self.logger.debug(f"Evicted cache entry: {oldest_key}")

		# Remove entries not accessed recently (older than 1 hour)
		cutoff_time = datetime.utcnow() - timedelta(hours=1)
		to_remove = []

		for key, entry in self._cache.items():
			if entry.last_accessed < cutoff_time and entry.access_count < 2:
				to_remove.append(key)
				current_memory_mb -= entry.memory_size_mb

		for key in to_remove:
			self._cache.pop(key, None)

		self._memory_usage_mb = current_memory_mb

	# ------------------------------------------------------------------
	# Helper Classes and Methods
	# ------------------------------------------------------------------

	class ArtifactCompressor:
		"""Utility class for artifact compression."""

		def __init__(self):
			self._logger = system_logger.getChild("ArtifactManager.ArtifactCompressor")

		def should_compress(self, data_size_bytes: int, config: CompressionConfig) -> bool:
			"""Determine if data should be compressed based on configuration."""
			if not config.enabled:
				return False

			min_size_bytes = int(config.min_size_mb * 1024 * 1024)
			return data_size_bytes >= min_size_bytes

		def choose_compression(self, data_size_bytes: int, config: CompressionConfig) -> CompressionType:
			"""Choose the best compression algorithm for the data."""
			if not config.enabled:
				return CompressionType.NONE

			if config.algorithm != CompressionType.AUTO:
				return config.algorithm

			# Auto-select based on data size and characteristics
			if data_size_bytes > 100 * 1024 * 1024:  # > 100MB
				return CompressionType.LZ4  # Fast compression for large data
			else:
				return CompressionType.GZIP  # Better compression ratio for smaller data

		def compress_data(self, data: Any, compression_type: CompressionType) -> bytes:
			"""Compress data using the specified algorithm."""
			try:
				if compression_type == CompressionType.GZIP:
					return gzip.compress(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
				elif compression_type == CompressionType.LZ4:
					return lz4.frame.compress(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
				else:
					return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
			except Exception as e:
				self._logger.warning(f"Compression failed, falling back to no compression: {e}")
				return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

		def decompress_data(self, compressed_data: bytes, compression_type: CompressionType) -> Any:
			"""Decompress data using the specified algorithm."""
			try:
				if compression_type == CompressionType.GZIP:
					return pickle.loads(gzip.decompress(compressed_data))
				elif compression_type == CompressionType.LZ4:
					return pickle.loads(lz4.frame.decompress(compressed_data))
				else:
					return pickle.loads(compressed_data)
			except Exception as e:
				self._logger.error(f"Decompression failed: {e}")
				raise

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
