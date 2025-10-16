"""
Multi-Timeframe Data Synchronization Utilities for Hybrid NAS-TAS Regime Detection.

Provides comprehensive multi-timeframe data synchronization and alignment
using existing utils for robust cross-timeframe analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass
import time
from datetime import datetime, timedelta
from enum import Enum
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import existing utilities
try:
    from src.utils.common_operations import (
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer
    )
    HARDWARE_UTILS_AVAILABLE = True
except ImportError:
    HARDWARE_UTILS_AVAILABLE = False

try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_enhanced_matrix_operations,
        get_batch_matrix_processor
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False

try:
    from src.utils.data.klines_parquet import get_klines_manager
    KLINES_MANAGER_AVAILABLE = True
except ImportError:
    KLINES_MANAGER_AVAILABLE = False

logger = logging.getLogger(__name__)

class SyncMethod(Enum):
    """Synchronization methods available."""
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    INTERPOLATE = "interpolate"
    AGGREGATE = "aggregate"
    ALIGN = "align"

@dataclass
class MultiTimeframeConfig:
    """Configuration for multi-timeframe synchronization."""
    primary_timeframe: str = "15m"
    secondary_timeframes: List[str] = None
    sync_method: SyncMethod = SyncMethod.FORWARD_FILL
    aggregation_method: str = "mean"  # "mean", "median", "sum", "last", "first"
    handle_missing: bool = True
    fill_limit: int = 10
    use_hardware_acceleration: bool = True
    use_matrix_operations: bool = True
    batch_size: int = 1000
    memory_limit_gb: float = 8.0

    def __post_init__(self):
        if self.secondary_timeframes is None:
            self.secondary_timeframes = ["1m", "5m", "1h", "4h", "1d"]

@dataclass
class MultiTimeframeResult:
    """Result from multi-timeframe synchronization."""
    synchronized_data: Dict[str, pd.DataFrame]
    sync_metadata: Dict[str, Any]
    alignment_info: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    hardware_optimization_applied: bool = False
    matrix_operations_used: bool = False

class MultiTimeframeSynchronizer:
    """Advanced multi-timeframe synchronizer with hardware acceleration."""

    def __init__(self, config: MultiTimeframeConfig):
        """Initialize the multi-timeframe synchronizer.

        Args:
            config: Multi-timeframe configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize hardware acceleration if available
        self.hardware_accelerator = None
        self.memory_optimizer = None
        self.cpu_optimizer = None

        if HARDWARE_UTILS_AVAILABLE and config.use_hardware_acceleration:
            try:
                self.hardware_accelerator = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()
                self.logger.info("✅ Hardware acceleration initialized for multi-timeframe sync")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware acceleration not available: {e}")

        # Initialize matrix operations if available
        self.matrix_ops = None
        self.vectorized_core = None
        self.enhanced_ops = None
        self.batch_processor = None

        if MATRIX_OPERATIONS_AVAILABLE and config.use_matrix_operations:
            try:
                self.matrix_ops = get_unified_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                self.enhanced_ops = get_enhanced_matrix_operations()
                self.batch_processor = get_batch_matrix_processor()
                self.logger.info("✅ Matrix operations initialized for multi-timeframe sync")
            except Exception as e:
                self.logger.warning(f"⚠️ Matrix operations not available: {e}")

        self.logger.info("✅ Multi-Timeframe Synchronizer initialized")
        self.logger.info(f"   Primary timeframe: {config.primary_timeframe}")
        self.logger.info(f"   Secondary timeframes: {config.secondary_timeframes}")
        self.logger.info(f"   Sync method: {config.sync_method.value}")

    def synchronize_timeframes(self, data_dict: Dict[str, pd.DataFrame],
                             symbol: str,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> MultiTimeframeResult:
        """Synchronize multiple timeframes to a common timeline.

        Args:
            data_dict: Dictionary mapping timeframes to DataFrames
            symbol: Symbol for data retrieval if needed
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            MultiTimeframeResult with synchronized data
        """
        start_time = time.time()

        try:
            self.logger.info("🔄 Starting multi-timeframe synchronization")
            self.logger.info(f"   Available timeframes: {list(data_dict.keys())}")
            self.logger.info(f"   Primary timeframe: {self.config.primary_timeframe}")

            # Load additional data if needed
            if self.config.primary_timeframe not in data_dict:
                self.logger.info(f"📊 Loading primary timeframe data: {self.config.primary_timeframe}")
                primary_data = self._load_timeframe_data(
                    symbol, self.config.primary_timeframe, start_date, end_date
                )
                if primary_data is not None:
                    data_dict[self.config.primary_timeframe] = primary_data

            # Load secondary timeframes if needed
            for timeframe in self.config.secondary_timeframes:
                if timeframe not in data_dict:
                    self.logger.info(f"📊 Loading secondary timeframe data: {timeframe}")
                    secondary_data = self._load_timeframe_data(
                        symbol, timeframe, start_date, end_date
                    )
                    if secondary_data is not None:
                        data_dict[timeframe] = secondary_data

            # Create primary timeline
            primary_timeline = self._create_primary_timeline(data_dict)

            # Synchronize all timeframes to primary timeline
            synchronized_data = {}
            alignment_info = {}

            for timeframe, data in data_dict.items():
                if timeframe == self.config.primary_timeframe:
                    synchronized_data[timeframe] = data
                    alignment_info[timeframe] = {
                        'original_shape': data.shape,
                        'synchronized_shape': data.shape,
                        'alignment_method': 'primary',
                        'missing_values': 0,
                        'alignment_quality': 1.0
                    }
                else:
                    synced_data, align_info = self._synchronize_to_timeline(
                        data, primary_timeline, timeframe
                    )
                    synchronized_data[timeframe] = synced_data
                    alignment_info[timeframe] = align_info

            processing_time = time.time() - start_time

            sync_metadata = {
                'primary_timeframe': self.config.primary_timeframe,
                'secondary_timeframes': self.config.secondary_timeframes,
                'sync_method': self.config.sync_method.value,
                'total_timeframes': len(synchronized_data),
                'primary_timeline_length': len(primary_timeline),
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }

            self.logger.info(f"✅ Multi-timeframe synchronization completed in {processing_time:.2f}s")
            self.logger.info(f"   Synchronized timeframes: {len(synchronized_data)}")

            return MultiTimeframeResult(
                synchronized_data=synchronized_data,
                sync_metadata=sync_metadata,
                alignment_info=alignment_info,
                processing_time=processing_time,
                success=True,
                hardware_optimization_applied=self.hardware_accelerator is not None,
                matrix_operations_used=self.matrix_ops is not None
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Multi-timeframe synchronization failed: {e}")

            return MultiTimeframeResult(
                synchronized_data={},
                sync_metadata={'error': str(e)},
                alignment_info={},
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )

    def _load_timeframe_data(self, symbol: str, timeframe: str,
                           start_date: Optional[str], end_date: Optional[str]) -> Optional[pd.DataFrame]:
        """Load data for a specific timeframe.

        Args:
            symbol: Symbol to load
            timeframe: Timeframe to load
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            Loaded DataFrame or None
        """
        try:
            if KLINES_MANAGER_AVAILABLE:
                manager = get_klines_manager()

                # Parse dates if provided
                start_dt = None
                end_dt = None
                if start_date:
                    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                if end_date:
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

                data = manager.read_data(
                    symbol, timeframe,
                    start_date=start_dt,
                    end_date=end_dt,
                    data_type="processed"
                )

                if data is not None and not len(data) == 0:
                    self.logger.info(f"✅ Loaded {timeframe} data: {data.shape}")
                    return data
                else:
                    self.logger.warning(f"⚠️ No data available for {symbol} {timeframe}")
                    return None
            else:
                self.logger.warning("⚠️ Klines manager not available")
                return None

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load {timeframe} data: {e}")
            return None

    def _create_primary_timeline(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        """Create primary timeline from primary timeframe data.

        Args:
            data_dict: Dictionary of timeframe data

        Returns:
            Primary timeline as DatetimeIndex
        """
        try:
            if self.config.primary_timeframe in data_dict:
                primary_data = data_dict[self.config.primary_timeframe]
                if hasattr(primary_data.index, 'to_pydatetime'):
                    return primary_data.index
                else:
                    # Create timeline from data
                    return pd.date_range(
                        start=primary_data.index[0],
                        end=primary_data.index[-1],
                        freq=self.config.primary_timeframe
                    )
            else:
                # Create default timeline
                return pd.date_range(
                    start='2023-01-01',
                    end='2023-12-31',
                    freq=self.config.primary_timeframe
                )

        except Exception as e:
            self.logger.warning(f"⚠️ Primary timeline creation failed: {e}")
            # Fallback to default timeline
            return pd.date_range(
                start='2023-01-01',
                end='2023-12-31',
                freq=self.config.primary_timeframe
            )

    def _synchronize_to_timeline(self, data: pd.DataFrame,
                                primary_timeline: pd.DatetimeIndex,
                                timeframe: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Synchronize data to primary timeline.

        Args:
            data: Data to synchronize
            primary_timeline: Target timeline
            timeframe: Source timeframe

        Returns:
            Tuple of (synchronized_data, alignment_info)
        """
        try:
            original_shape = data.shape

            # Ensure data has datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                if 'timestamp' in data.columns:
                    data = data.set_index('timestamp')
                else:
                    # Create datetime index
                    data.index = pd.date_range(
                        start='2023-01-01',
                        periods=len(data),
                        freq=timeframe
                    )

            # Create synchronized DataFrame with primary timeline
            synced_data = pd.DataFrame(index=primary_timeline)

            # Copy relevant columns
            for col in data.columns:
                if col in ['open', 'high', 'low', 'close', 'volume']:
                    synced_data[col] = np.nan

            # Align data based on sync method
            if self.config.sync_method == SyncMethod.FORWARD_FILL:
                synced_data = self._forward_fill_align(data, synced_data)
            elif self.config.sync_method == SyncMethod.BACKWARD_FILL:
                synced_data = self._backward_fill_align(data, synced_data)
            elif self.config.sync_method == SyncMethod.INTERPOLATE:
                synced_data = self._interpolate_align(data, synced_data)
            elif self.config.sync_method == SyncMethod.AGGREGATE:
                synced_data = self._aggregate_align(data, synced_data)
            else:  # ALIGN
                synced_data = self._align_data(data, synced_data)

            # Calculate alignment info
            missing_values = synced_data.isnull().sum().sum()
            total_values = synced_data.size
            alignment_quality = 1.0 - (missing_values / total_values) if total_values > 0 else 0.0

            alignment_info = {
                'original_shape': original_shape,
                'synchronized_shape': synced_data.shape,
                'alignment_method': self.config.sync_method.value,
                'missing_values': int(missing_values),
                'alignment_quality': float(alignment_quality),
                'timeframe': timeframe
            }

            return synced_data, alignment_info

        except Exception as e:
            self.logger.warning(f"⚠️ Timeline synchronization failed for {timeframe}: {e}")
            return data, {'error': str(e), 'timeframe': timeframe}

    def _forward_fill_align(self, source_data: pd.DataFrame,
                          target_data: pd.DataFrame) -> pd.DataFrame:
        """Forward fill alignment."""
        try:
            # Reindex and forward fill
            aligned_data = source_data.reindex(target_data.index, method='ffill', limit=self.config.fill_limit)
            target_data.update(aligned_data)
            return target_data

        except Exception as e:
            self.logger.warning(f"⚠️ Forward fill alignment failed: {e}")
            return target_data

    def _backward_fill_align(self, source_data: pd.DataFrame,
                            target_data: pd.DataFrame) -> pd.DataFrame:
        """Backward fill alignment."""
        try:
            # Reindex and backward fill
            aligned_data = source_data.reindex(target_data.index, method='bfill', limit=self.config.fill_limit)
            target_data.update(aligned_data)
            return target_data

        except Exception as e:
            self.logger.warning(f"⚠️ Backward fill alignment failed: {e}")
            return target_data

    def _interpolate_align(self, source_data: pd.DataFrame,
                         target_data: pd.DataFrame) -> pd.DataFrame:
        """Interpolation alignment."""
        try:
            # Reindex and interpolate
            aligned_data = source_data.reindex(target_data.index).interpolate(method='linear')
            target_data.update(aligned_data)
            return target_data

        except Exception as e:
            self.logger.warning(f"⚠️ Interpolation alignment failed: {e}")
            return target_data

    def _aggregate_align(self, source_data: pd.DataFrame,
                        target_data: pd.DataFrame) -> pd.DataFrame:
        """Aggregation alignment."""
        try:
            # Resample to target frequency and aggregate
            if self.config.aggregation_method == "mean":
                aggregated = source_data.resample(target_data.index.freq).mean()
            elif self.config.aggregation_method == "median":
                aggregated = source_data.resample(target_data.index.freq).median()
            elif self.config.aggregation_method == "sum":
                aggregated = source_data.resample(target_data.index.freq).sum()
            elif self.config.aggregation_method == "last":
                aggregated = source_data.resample(target_data.index.freq).last()
            elif self.config.aggregation_method == "first":
                aggregated = source_data.resample(target_data.index.freq).first()
            else:
                aggregated = source_data.resample(target_data.index.freq).mean()

            # Align with target index
            aligned_data = aggregated.reindex(target_data.index)
            target_data.update(aligned_data)
            return target_data

        except Exception as e:
            self.logger.warning(f"⚠️ Aggregation alignment failed: {e}")
            return target_data

    def _align_data(self, source_data: pd.DataFrame,
                   target_data: pd.DataFrame) -> pd.DataFrame:
        """Direct alignment."""
        try:
            # Direct reindexing
            aligned_data = source_data.reindex(target_data.index)
            target_data.update(aligned_data)
            return target_data

        except Exception as e:
            self.logger.warning(f"⚠️ Direct alignment failed: {e}")
            return target_data

    def get_synchronization_statistics(self, result: MultiTimeframeResult) -> Dict[str, Any]:
        """Get statistics about synchronization results.

        Args:
            result: MultiTimeframeResult

        Returns:
            Synchronization statistics
        """
        try:
            stats = {
                'total_timeframes': len(result.synchronized_data),
                'primary_timeframe': result.sync_metadata.get('primary_timeframe'),
                'sync_method': result.sync_metadata.get('sync_method'),
                'processing_time': result.processing_time,
                'timeframe_statistics': {}
            }

            for timeframe, data in result.synchronized_data.items():
                align_info = result.alignment_info.get(timeframe, {})
                stats['timeframe_statistics'][timeframe] = {
                    'shape': data.shape,
                    'alignment_quality': align_info.get('alignment_quality', 0.0),
                    'missing_values': align_info.get('missing_values', 0),
                    'alignment_method': align_info.get('alignment_method', 'unknown')
                }

            return stats

        except Exception as e:
            self.logger.warning(f"⚠️ Statistics calculation failed: {e}")
            return {'error': str(e)}

def create_multi_timeframe_synchronizer(config: Optional[MultiTimeframeConfig] = None) -> MultiTimeframeSynchronizer:
    """Create a multi-timeframe synchronizer instance.

    Args:
        config: Optional multi-timeframe configuration

    Returns:
        MultiTimeframeSynchronizer instance
    """
    if config is None:
        config = MultiTimeframeConfig()
    return MultiTimeframeSynchronizer(config)

def quick_synchronize(data_dict: Dict[str, pd.DataFrame],
                    primary_timeframe: str = "15m",
                    sync_method: SyncMethod = SyncMethod.FORWARD_FILL) -> MultiTimeframeResult:
    """Quick multi-timeframe synchronization with default settings.

    Args:
        data_dict: Dictionary mapping timeframes to DataFrames
        primary_timeframe: Primary timeframe
        sync_method: Synchronization method

    Returns:
        MultiTimeframeResult
    """
    config = MultiTimeframeConfig(
        primary_timeframe=primary_timeframe,
        sync_method=sync_method
    )
    synchronizer = MultiTimeframeSynchronizer(config)
    return synchronizer.synchronize_timeframes(data_dict, "BTCUSDT")
