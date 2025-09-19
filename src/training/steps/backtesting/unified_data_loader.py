"""
Unified Data Loading Mechanism for Backtesting

This module provides a standardized data loading interface that ensures consistency
across all backtesting steps and integrates with the existing data infrastructure
used in model training and market analysis.

Key Features:
- Unified data loading interface across all backtesting steps
- Integration with standardized_parquet_handler
- Memory optimization using hardware utilities
- Consistent data validation and preprocessing
- Caching mechanism for frequently accessed data
- Automatic cleanup and garbage collection
"""

import logging
import time
import gc
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import weakref

# Optional imports with fallback
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# Import existing infrastructure
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
from src.utils.common_operations import (
    safe_json_load, safe_file_exists, ensure_directory, get_current_datetime
)
from src.utils.math_validation import validate_finite, validate_positive

# Import klines parquet utilities
try:
    from src.utils.data.klines_parquet import KlinesParquetManager, get_klines_manager
    KLINES_UTILS_AVAILABLE = True
except ImportError:
    KLINES_UTILS_AVAILABLE = False

# Import hardware optimization tools
try:
    from src.utils.hardware.advanced_memory_optimizer import AdvancedMemoryOptimizer
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager, WorkloadType, OptimizationLevel
    from src.utils.matrix_operations.unified_operations import UnifiedMatrixOperations
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """Types of data sources for backtesting."""
    KLINES_PARQUET = "klines_parquet"
    CONSOLIDATED_PARQUET = "consolidated_parquet"
    INDIVIDUAL_FILES = "individual_files"
    CACHED_DATA = "cached_data"
    LIVE_DATA = "live_data"


class DataLoadingMode(Enum):
    """Data loading modes for different use cases."""
    FULL = "full"              # Load all available data
    STREAMING = "streaming"    # Load data in chunks
    MEMORY_OPTIMIZED = "memory_optimized"  # Load with memory constraints
    CACHED = "cached"          # Use cached data when available


@dataclass
class DataLoadingConfig:
    """Configuration for unified data loading."""
    # Basic parameters
    symbol: str
    exchange: str
    timeframe: str
    data_dir: str
    
    # Date range
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Loading behavior
    loading_mode: DataLoadingMode = DataLoadingMode.MEMORY_OPTIMIZED
    data_source_priority: List[DataSourceType] = field(default_factory=lambda: [
        DataSourceType.CACHED_DATA,
        DataSourceType.KLINES_PARQUET,
        DataSourceType.CONSOLIDATED_PARQUET,
        DataSourceType.INDIVIDUAL_FILES
    ])
    
    # Memory management
    enable_memory_optimization: bool = True
    memory_limit_mb: float = 1000.0
    chunk_size: int = 10000
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    
    # Data validation
    validate_data: bool = True
    required_columns: List[str] = field(default_factory=lambda: [
        'timestamp', 'open', 'high', 'low', 'close', 'volume'
    ])
    min_data_points: int = 100
    
    # Performance settings
    enable_parallel_loading: bool = True
    max_workers: int = 4


@dataclass
class LoadedData:
    """Container for loaded data with metadata."""
    data: pd.DataFrame
    metadata: Dict[str, Any]
    source_type: DataSourceType
    load_time: datetime
    memory_usage_mb: float
    data_quality_score: float = 0.0


class DataCache:
    """Memory-optimized data cache with automatic cleanup."""
    
    def __init__(self, max_size_mb: float = 500.0, ttl_seconds: int = 3600):
        """Initialize data cache.
        
        Args:
            max_size_mb: Maximum cache size in MB
            ttl_seconds: Time-to-live for cached items
        """
        self.max_size_mb = max_size_mb
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[LoadedData, datetime]] = {}
        self.access_times: Dict[str, datetime] = {}
        self.logger = logger.getChild('DataCache')
        
        # Initialize memory optimizer if available
        self.memory_optimizer = None
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            try:
                self.memory_optimizer = get_m1_memory_optimizer()
            except Exception as e:
                self.logger.warning(f"Could not initialize memory optimizer: {e}")
    
    def _get_cache_key(self, config: DataLoadingConfig) -> str:
        """Generate cache key for configuration."""
        key_parts = [
            config.symbol,
            config.exchange,
            config.timeframe,
            config.start_date.isoformat() if config.start_date else "None",
            config.end_date.isoformat() if config.end_date else "None"
        ]
        return "_".join(key_parts)
    
    def _calculate_memory_usage(self, data: pd.DataFrame) -> float:
        """Calculate memory usage of DataFrame in MB."""
        try:
            return data.memory_usage(deep=True).sum() / 1024 / 1024
        except Exception:
            return 0.0
    
    def _cleanup_expired(self) -> None:
        """Remove expired cache entries."""
        current_time = datetime.now()
        expired_keys = []
        
        for key, (cached_data, cache_time) in self.cache.items():
            if (current_time - cache_time).total_seconds() > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
            self.logger.debug(f"Removed expired cache entry: {key}")
    
    def _remove_entry(self, key: str) -> None:
        """Remove cache entry and cleanup memory."""
        if key in self.cache:
            cached_data, _ = self.cache[key]
            del cached_data.data  # Explicitly delete DataFrame
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            
            # Force garbage collection
            gc.collect()
    
    def _enforce_size_limit(self) -> None:
        """Enforce cache size limit by removing least recently used items."""
        current_size = sum(
            self._calculate_memory_usage(cached_data.data) 
            for cached_data, _ in self.cache.values()
        )
        
        if current_size <= self.max_size_mb:
            return
        
        # Sort by access time (least recently used first)
        sorted_keys = sorted(
            self.access_times.keys(),
            key=lambda k: self.access_times[k]
        )
        
        # Remove entries until under size limit
        for key in sorted_keys:
            if current_size <= self.max_size_mb:
                break
            
            if key in self.cache:
                cached_data, _ = self.cache[key]
                current_size -= self._calculate_memory_usage(cached_data.data)
                self._remove_entry(key)
                self.logger.debug(f"Removed cache entry due to size limit: {key}")
    
    def get(self, config: DataLoadingConfig) -> Optional[LoadedData]:
        """Get data from cache if available and valid."""
        self._cleanup_expired()
        
        key = self._get_cache_key(config)
        if key not in self.cache:
            return None
        
        cached_data, cache_time = self.cache[key]
        self.access_times[key] = datetime.now()
        
        self.logger.debug(f"Cache hit for key: {key}")
        return cached_data
    
    def put(self, config: DataLoadingConfig, data: LoadedData) -> None:
        """Store data in cache."""
        if not config.enable_caching:
            return
        
        key = self._get_cache_key(config)
        current_time = datetime.now()
        
        self.cache[key] = (data, current_time)
        self.access_times[key] = current_time
        
        self._enforce_size_limit()
        self.logger.debug(f"Cached data for key: {key}")
    
    def clear(self) -> None:
        """Clear all cached data."""
        for key in list(self.cache.keys()):
            self._remove_entry(key)
        gc.collect()
        self.logger.info("Cache cleared")


class UnifiedDataLoader:
    """Unified data loader for all backtesting operations."""
    
    def __init__(self, config: Optional[DataLoadingConfig] = None):
        """Initialize unified data loader.
        
        Args:
            config: Optional default configuration
        """
        self.default_config = config
        self.logger = logger.getChild('UnifiedDataLoader')
        
        # Initialize cache
        self.cache = DataCache()
        
        # Initialize hardware optimization
        self.hardware_manager = None
        self.memory_optimizer = None
        self.matrix_ops = None
        
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            try:
                self.hardware_manager = UnifiedHardwareManager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.matrix_ops = UnifiedMatrixOperations()
                self.logger.info("✅ Unified hardware optimization enabled")
            except Exception as e:
                self.logger.warning(f"Hardware optimization not available: {e}")
        
        # Initialize klines manager
        self.klines_manager = None
        if KLINES_UTILS_AVAILABLE:
            try:
                self.klines_manager = get_klines_manager(config.data_dir if config else "historical_data")
                self.logger.info("✅ Klines parquet utilities enabled")
            except Exception as e:
                self.logger.warning(f"Klines utilities not available: {e}")
        
        # Track loaded data for cleanup
        self.loaded_data_refs: List[weakref.ref] = []
        
        self.logger.info("✅ UnifiedDataLoader initialized")
    
    @contextmanager
    def memory_optimized_loading(self, config: DataLoadingConfig):
        """Context manager for memory-optimized data loading using unified hardware manager."""
        if self.hardware_manager and config.enable_memory_optimization:
            # Configure hardware for backtesting workload
            hardware_config = {
                'workload_type': WorkloadType.BACKTESTING,
                'optimization_level': OptimizationLevel.BALANCED,
                'memory_limit_gb': config.memory_limit_mb / 1024
            }
            
            with self.hardware_manager.optimize_for_workload(**hardware_config):
                yield
        elif self.memory_optimizer and config.enable_memory_optimization:
            # Fallback to basic memory optimization
            with self.memory_optimizer.optimization_context(
                memory_limit_gb=config.memory_limit_mb / 1024
            ):
                yield
        else:
            yield
    
    def _validate_config(self, config: DataLoadingConfig) -> None:
        """Validate data loading configuration."""
        if not config.symbol or not config.exchange:
            raise ValueError("Symbol and exchange must be specified")
        
        if not config.data_dir or not Path(config.data_dir).exists():
            raise ValueError(f"Data directory does not exist: {config.data_dir}")
        
        if config.start_date and config.end_date and config.start_date >= config.end_date:
            raise ValueError("Start date must be before end date")
    
    def _validate_data(self, data: pd.DataFrame, config: DataLoadingConfig) -> float:
        """Validate loaded data and return quality score.
        
        Args:
            data: Loaded DataFrame
            config: Loading configuration
            
        Returns:
            Data quality score (0.0 to 1.0)
        """
        if not config.validate_data:
            return 1.0
        
        quality_score = 1.0
        issues = []
        
        # Check if data is empty
        if data.empty:
            raise ValueError("Loaded data is empty")
        
        # Check required columns
        missing_columns = [col for col in config.required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check minimum data points
        if len(data) < config.min_data_points:
            raise ValueError(f"Insufficient data points: {len(data)} < {config.min_data_points}")
        
        # Check for missing values
        missing_values = data[config.required_columns].isnull().sum().sum()
        if missing_values > 0:
            missing_ratio = missing_values / (len(data) * len(config.required_columns))
            quality_score -= missing_ratio * 0.3
            issues.append(f"Missing values: {missing_values} ({missing_ratio:.2%})")
        
        # Check for duplicate timestamps
        if 'timestamp' in data.columns:
            duplicates = data['timestamp'].duplicated().sum()
            if duplicates > 0:
                duplicate_ratio = duplicates / len(data)
                quality_score -= duplicate_ratio * 0.2
                issues.append(f"Duplicate timestamps: {duplicates} ({duplicate_ratio:.2%})")
        
        # Check for invalid prices (negative or zero)
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                invalid_prices = (data[col] <= 0).sum()
                if invalid_prices > 0:
                    invalid_ratio = invalid_prices / len(data)
                    quality_score -= invalid_ratio * 0.3
                    issues.append(f"Invalid {col} prices: {invalid_prices} ({invalid_ratio:.2%})")
        
        # Check for extreme outliers
        if 'close' in data.columns:
            close_prices = data['close'].dropna()
            if len(close_prices) > 0:
                q1, q3 = close_prices.quantile([0.25, 0.75])
                iqr = q3 - q1
                outliers = ((close_prices < (q1 - 3 * iqr)) | 
                           (close_prices > (q3 + 3 * iqr))).sum()
                if outliers > 0:
                    outlier_ratio = outliers / len(close_prices)
                    if outlier_ratio > 0.05:  # More than 5% outliers is concerning
                        quality_score -= outlier_ratio * 0.1
                        issues.append(f"Price outliers: {outliers} ({outlier_ratio:.2%})")
        
        quality_score = max(0.0, quality_score)
        
        if issues:
            self.logger.warning(f"Data quality issues found: {'; '.join(issues)}")
        
        self.logger.info(f"Data quality score: {quality_score:.2f}")
        return quality_score
    
    def _load_from_klines_parquet(self, config: DataLoadingConfig) -> Optional[pd.DataFrame]:
        """Load data from klines parquet utilities."""
        if not self.klines_manager:
            self.logger.debug("Klines manager not available")
            return None
        
        try:
            self.logger.info(f"📁 Loading data via klines parquet: {config.symbol} {config.timeframe}")
            
            # Try to load processed data first, then raw data
            data = self.klines_manager.read_data(
                symbol=config.symbol,
                interval=config.timeframe,
                start_date=config.start_date,
                end_date=config.end_date,
                data_type="processed"
            )
            
            if data is None or data.empty:
                # Fallback to raw data
                self.logger.debug("Processed data not found, trying raw data")
                data = self.klines_manager.read_data(
                    symbol=config.symbol,
                    interval=config.timeframe,
                    start_date=config.start_date,
                    end_date=config.end_date,
                    data_type="raw"
                )
            
            if data is not None and not data.empty:
                self.logger.info(f"✅ Loaded {len(data):,} records from klines parquet")
                return data
            else:
                self.logger.debug("No data found in klines parquet")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error loading from klines parquet: {e}")
            return None
    
    def _load_from_consolidated_parquet(self, config: DataLoadingConfig) -> Optional[pd.DataFrame]:
        """Load data from consolidated parquet file."""
        consolidated_file = Path(config.data_dir) / f"aggtrades_{config.exchange}_{config.symbol}_consolidated.parquet"
        
        if not safe_file_exists(consolidated_file):
            self.logger.debug(f"Consolidated file not found: {consolidated_file}")
            return None
        
        try:
            self.logger.info(f"📁 Loading consolidated data: {consolidated_file}")
            data = standardized_parquet_handler.read_parquet_standardized(consolidated_file)
            
            # Filter by date range if specified
            if config.start_date or config.end_date:
                data = self._filter_by_date_range(data, config.start_date, config.end_date)
            
            self.logger.info(f"✅ Loaded {len(data):,} records from consolidated file")
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Error loading consolidated file: {e}")
            return None
    
    def _load_from_individual_files(self, config: DataLoadingConfig) -> Optional[pd.DataFrame]:
        """Load data from individual parquet files."""
        data_dir = Path(config.data_dir)
        
        # Look for individual files matching the pattern
        pattern = f"*{config.exchange}*{config.symbol}*.parquet"
        files = list(data_dir.glob(pattern))
        
        if not files:
            self.logger.debug(f"No individual files found matching pattern: {pattern}")
            return None
        
        try:
            self.logger.info(f"📁 Loading {len(files)} individual files")
            
            dataframes = []
            for file_path in sorted(files):
                try:
                    df = standardized_parquet_handler.read_parquet_standardized(file_path)
                    if not df.empty:
                        dataframes.append(df)
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not load {file_path}: {e}")
                    continue
            
            if not dataframes:
                self.logger.warning("No valid data found in individual files")
                return None
            
            # Concatenate all dataframes
            data = pd.concat(dataframes, ignore_index=True)
            
            # Sort by timestamp if available
            if 'timestamp' in data.columns:
                data = data.sort_values('timestamp').reset_index(drop=True)
            
            # Filter by date range if specified
            if config.start_date or config.end_date:
                data = self._filter_by_date_range(data, config.start_date, config.end_date)
            
            self.logger.info(f"✅ Loaded {len(data):,} records from {len(dataframes)} individual files")
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Error loading individual files: {e}")
            return None
    
    def _filter_by_date_range(
        self, 
        data: pd.DataFrame, 
        start_date: Optional[datetime], 
        end_date: Optional[datetime]
    ) -> pd.DataFrame:
        """Filter data by date range."""
        if not start_date and not end_date:
            return data
        
        if 'timestamp' not in data.columns:
            self.logger.warning("Cannot filter by date: no timestamp column")
            return data
        
        try:
            # Convert timestamp to datetime if needed
            if data['timestamp'].dtype == 'int64':
                # Assume millisecond timestamps
                timestamp_col = pd.to_datetime(data['timestamp'], unit='ms')
            else:
                timestamp_col = pd.to_datetime(data['timestamp'])
            
            mask = pd.Series(True, index=data.index)
            
            if start_date:
                mask &= timestamp_col >= start_date
            
            if end_date:
                mask &= timestamp_col <= end_date
            
            filtered_data = data[mask].reset_index(drop=True)
            
            if len(filtered_data) < len(data):
                self.logger.info(f"📅 Filtered data: {len(data):,} → {len(filtered_data):,} records")
            
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"❌ Error filtering by date range: {e}")
            return data
    
    def _optimize_dataframe_memory(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        if not self.matrix_ops:
            return data
        
        try:
            # Use matrix operations for memory optimization
            optimized_data = self.matrix_ops.optimize_dataframe_memory(data)
            
            original_memory = data.memory_usage(deep=True).sum() / 1024 / 1024
            optimized_memory = optimized_data.memory_usage(deep=True).sum() / 1024 / 1024
            
            if optimized_memory < original_memory:
                self.logger.info(f"🧠 Memory optimized: {original_memory:.1f}MB → {optimized_memory:.1f}MB")
                return optimized_data
            
        except Exception as e:
            self.logger.warning(f"⚠️ Memory optimization failed: {e}")
        
        return data
    
    def load_data(self, config: Optional[DataLoadingConfig] = None) -> LoadedData:
        """Load data using unified loading mechanism.
        
        Args:
            config: Loading configuration (uses default if None)
            
        Returns:
            LoadedData object with data and metadata
        """
        config = config or self.default_config
        if not config:
            raise ValueError("No configuration provided")
        
        self._validate_config(config)
        
        # Check cache first
        if config.enable_caching:
            cached_data = self.cache.get(config)
            if cached_data:
                self.logger.info("📦 Using cached data")
                return cached_data
        
        start_time = time.time()
        
        with self.memory_optimized_loading(config):
            # Try each data source in priority order
            data = None
            source_type = None
            
            for source in config.data_source_priority:
                if source == DataSourceType.KLINES_PARQUET:
                    data = self._load_from_klines_parquet(config)
                    if data is not None:
                        source_type = source
                        break
                elif source == DataSourceType.CONSOLIDATED_PARQUET:
                    data = self._load_from_consolidated_parquet(config)
                    if data is not None:
                        source_type = source
                        break
                elif source == DataSourceType.INDIVIDUAL_FILES:
                    data = self._load_from_individual_files(config)
                    if data is not None:
                        source_type = source
                        break
                # Add other source types as needed
            
            if data is None:
                raise ValueError("Could not load data from any available source")
            
            # Optimize memory usage
            if config.enable_memory_optimization:
                data = self._optimize_dataframe_memory(data)
            
            # Validate data quality
            quality_score = self._validate_data(data, config)
            
            # Calculate memory usage
            memory_usage = data.memory_usage(deep=True).sum() / 1024 / 1024
            
            # Create loaded data object
            loaded_data = LoadedData(
                data=data,
                metadata={
                    'symbol': config.symbol,
                    'exchange': config.exchange,
                    'timeframe': config.timeframe,
                    'start_date': config.start_date.isoformat() if config.start_date else None,
                    'end_date': config.end_date.isoformat() if config.end_date else None,
                    'record_count': len(data),
                    'columns': list(data.columns),
                    'data_types': {col: str(dtype) for col, dtype in data.dtypes.items()},
                    'loading_time_seconds': time.time() - start_time
                },
                source_type=source_type,
                load_time=datetime.now(),
                memory_usage_mb=memory_usage,
                data_quality_score=quality_score
            )
            
            # Cache the data
            if config.enable_caching:
                self.cache.put(config, loaded_data)
            
            # Track for cleanup
            self.loaded_data_refs.append(weakref.ref(loaded_data))
            
            self.logger.info(f"✅ Data loaded successfully:")
            self.logger.info(f"   📊 Records: {len(data):,}")
            self.logger.info(f"   🧠 Memory: {memory_usage:.1f}MB")
            self.logger.info(f"   ⏱️ Time: {time.time() - start_time:.2f}s")
            self.logger.info(f"   🎯 Quality: {quality_score:.2f}")
            self.logger.info(f"   📁 Source: {source_type.value}")
            
            return loaded_data
    
    def cleanup(self) -> None:
        """Clean up loaded data and free memory."""
        # Clear cache
        self.cache.clear()
        
        # Clean up tracked references
        active_refs = []
        for ref in self.loaded_data_refs:
            obj = ref()
            if obj is not None:
                try:
                    del obj.data
                except AttributeError:
                    pass
            else:
                active_refs.append(ref)
        
        self.loaded_data_refs = active_refs
        
        # Force garbage collection
        if self.memory_optimizer:
            self.memory_optimizer.force_cleanup()
        else:
            gc.collect()
        
        self.logger.info("🧹 Data loader cleanup completed")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        stats = {
            'cache_size_mb': sum(
                data.memory_usage_mb for data, _ in self.cache.cache.values()
            ),
            'cache_entries': len(self.cache.cache),
            'active_references': len([ref for ref in self.loaded_data_refs if ref() is not None])
        }
        
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            stats['process_memory_mb'] = process.memory_info().rss / 1024 / 1024
            stats['process_memory_percent'] = process.memory_percent()
        
        return stats


# Global instance for easy access
_unified_data_loader = None

def get_unified_data_loader(config: Optional[DataLoadingConfig] = None) -> UnifiedDataLoader:
    """Get global unified data loader instance."""
    global _unified_data_loader
    if _unified_data_loader is None:
        _unified_data_loader = UnifiedDataLoader(config)
    return _unified_data_loader


# Convenience functions for backward compatibility
def load_backtesting_data(
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    **kwargs
) -> LoadedData:
    """Convenience function to load backtesting data."""
    config = DataLoadingConfig(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )
    
    loader = get_unified_data_loader()
    return loader.load_data(config)


def cleanup_data_loader() -> None:
    """Cleanup global data loader."""
    global _unified_data_loader
    if _unified_data_loader:
        _unified_data_loader.cleanup()
        _unified_data_loader = None