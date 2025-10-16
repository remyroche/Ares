"""
Data Pipeline Utilities for Hybrid NAS-TAS Regime Detection.

Provides common data processing utilities used by both NAS and TAS regime detection systems.
Uses the same data source as hmm_regime_discovery.py (klines_parquet) but operates independently
without direct dependency on hmm_regime_discovery.py.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import time
from datetime import datetime, timedelta
import asyncio
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

try:
    from src.utils.data.klines_parquet import get_klines_manager
    KLINES_MANAGER_AVAILABLE = True
except ImportError:
    KLINES_MANAGER_AVAILABLE = False

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
    from src.utils.hardware import (
        get_hardware_accelerator,
        get_memory_manager,
        get_performance_monitor
    )
    HARDWARE_ACCELERATION_AVAILABLE = True
except ImportError:
    HARDWARE_ACCELERATION_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class DataPipelineConfig:
    """Configuration for data pipeline operations."""
    symbol: str
    timeframe: str = "15m"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    data_type: str = "processed"  # "raw" or "processed"
    use_hardware_acceleration: bool = True
    use_matrix_operations: bool = True
    batch_size: int = 1000
    memory_limit_gb: float = 8.0
    validation_enabled: bool = True

@dataclass
class DataPipelineResult:
    """Result from data pipeline operations."""
    data: pd.DataFrame
    metadata: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    hardware_optimization_applied: bool = False
    matrix_operations_used: bool = False

class MarketDataProcessor:
    """Market data processor with hardware acceleration and matrix operations."""

    def __init__(self, config: DataPipelineConfig):
        """Initialize the market data processor.

        Args:
            config: Data pipeline configuration
        """
        tprint_info("Initializing Market Data Processor")
        tprint_debug(f"Configuration: {config}")
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize hardware acceleration if available
        self.hardware_accelerator = None
        self.memory_manager = None
        self.performance_monitor = None

        if HARDWARE_ACCELERATION_AVAILABLE and config.use_hardware_acceleration:
            try:
                tprint_info("Initializing hardware acceleration")
                self.hardware_accelerator = get_hardware_accelerator()
                self.memory_manager = get_memory_manager()
                self.performance_monitor = get_performance_monitor()
                tprint_success("Hardware acceleration initialized")
                self.logger.info("✅ Hardware acceleration initialized for data processing")
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
                self.logger.info("✅ Matrix operations initialized for data processing")
            except Exception as e:
                self.logger.warning(f"⚠️ Matrix operations not available: {e}")

    async def load_market_data(self) -> DataPipelineResult:
        """Load market data using klines_parquet manager.

        Returns:
            DataPipelineResult with loaded data and metadata
        """
        tprint_info(f"Loading market data for {self.config.symbol} {self.config.timeframe}")
        start_time = time.time()

        try:
            self.logger.info(f"📊 Loading market data for {self.config.symbol} {self.config.timeframe}")

            # Monitor performance
            if self.performance_monitor:
                tprint_debug("Starting performance monitoring")
                self.performance_monitor.start_monitoring("market_data_loading")

            # Load data using klines_parquet manager
            if KLINES_MANAGER_AVAILABLE:
                tprint_info("Using klines manager for data loading")
                data = await self._load_with_klines_manager()
            else:
                tprint_warning("Klines manager not available, using fallback data loading")
                data = await self._load_fallback_data()

            if data is None or len(data) == 0:
                tprint_error(f"No data available for {self.config.symbol} {self.config.timeframe}")
                raise ValueError(f"No data available for {self.config.symbol} {self.config.timeframe}")

            tprint_success(f"Data loaded successfully: {data.shape}")

            # Validate data
            if self.config.validation_enabled:
                tprint_info("Validating loaded data")
                validation_results = self._validate_data(data)
                if not validation_results['is_valid']:
                    tprint_warning(f"Data validation issues: {validation_results['issues']}")
                    self.logger.warning(f"⚠️ Data validation issues: {validation_results['issues']}")
                else:
                    tprint_success("Data validation passed")

            # Apply hardware optimizations if available
            if self.memory_manager:
                tprint_info("Applying memory optimizations")
                memory_config = self._optimize_memory_usage(data)
                tprint_success(f"Memory optimization applied: {memory_config}")
                self.logger.info(f"💾 Memory optimization applied: {memory_config}")

            processing_time = time.time() - start_time
            tprint_performance("Market Data Loading", processing_time)

            # Stop performance monitoring
            perf_metrics = {}
            if self.performance_monitor:
                tprint_debug("Stopping performance monitoring")
                perf_metrics = self.performance_monitor.stop_monitoring("market_data_loading")

            metadata = {
                'symbol': self.config.symbol,
                'timeframe': self.config.timeframe,
                'data_shape': data.shape,
                'columns': list(data.columns),
                'date_range': {
                    'start': data.index.min().isoformat() if not len(data) == 0 else None,
                    'end': data.index.max().isoformat() if not len(data) == 0 else None
                },
                'performance_metrics': perf_metrics,
                'hardware_optimization': self.hardware_accelerator is not None,
                'matrix_operations': self.matrix_ops is not None
            }

            self.logger.info(f"✅ Market data loaded: {data.shape} in {processing_time:.2f}s")

            return DataPipelineResult(
                data=data,
                metadata=metadata,
                processing_time=processing_time,
                success=True,
                hardware_optimization_applied=self.hardware_accelerator is not None,
                matrix_operations_used=self.matrix_ops is not None
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Market data loading failed: {e}")
            return DataPipelineResult(
                data=pd.DataFrame(),
                metadata={'error': str(e)},
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )

    async def _load_with_klines_manager(self) -> Optional[pd.DataFrame]:
        """Load data using klines_parquet manager."""
        try:
            manager = get_klines_manager()

            # Parse date filters if provided
            start_date = None
            end_date = None
            if self.config.start_date:
                start_date = datetime.strptime(self.config.start_date, '%Y-%m-%d')
            if self.config.end_date:
                end_date = datetime.strptime(self.config.end_date, '%Y-%m-%d')

            # If date filtering is requested, load all data first to determine available range
            if start_date or end_date:
                self.logger.info(f"🔍 Date filtering requested, loading all data first to determine available range")
                # Load all data first without date filtering
                all_data = manager.read_data(
                    self.config.symbol,
                    self.config.timeframe,
                    start_date=None,
                    end_date=None,
                    data_type=self.config.data_type
                )

                if all_data is not None and not all_len(data) == 0:
                    # Determine the last 10 days of available data
                    if 'timestamp' in all_data.columns:
                        # Convert timestamp to datetime
                        timestamps = pd.to_datetime(all_data['timestamp'], unit='s')
                        max_date = timestamps.max()
                        min_date = timestamps.min()

                        self.logger.info(f"📅 Available data range: {min_date.date()} to {max_date.date()}")

                        # Use the last 10 days of available data instead of hardcoded dates
                        if start_date is None:
                            start_date = max_date - timedelta(days=10)
                        if end_date is None:
                            end_date = max_date

                        self.logger.info(f"📅 Using date range: {start_date.date()} to {end_date.date()}")

                        # Apply date filtering to the loaded data
                        mask = (timestamps >= start_date) & (timestamps <= end_date)
                        data = all_data[mask]

                        if len(data) == 0:
                            self.logger.warning(f"⚠️ No data found in date range {start_date.date()} to {end_date.date()}")
                            # Fall back to the last 10 days of available data
                            last_10_days = max_date - timedelta(days=10)
                            mask = timestamps >= last_10_days
                            data = all_data[mask]
                            self.logger.info(f"📅 Fallback: Using last 10 days from {last_10_days.date()}")
                    else:
                        # No timestamp column, use the original data
                        data = all_data
                else:
                    # No data available, try with original date filtering
                    data = manager.read_data(
                        self.config.symbol,
                        self.config.timeframe,
                        start_date=start_date,
                        end_date=end_date,
                        data_type=self.config.data_type
                    )
            else:
                # No date filtering requested, load all data
                data = manager.read_data(
                    self.config.symbol,
                    self.config.timeframe,
                    start_date=None,
                    end_date=None,
                    data_type=self.config.data_type
                )

            if data is None or len(data) == 0:
                # Fallback to raw data
                self.logger.info(f"📊 No processed data found, trying raw data")
                data = manager.read_data(
                    self.config.symbol,
                    self.config.timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    data_type="raw"
                )

            return data

        except Exception as e:
            self.logger.error(f"❌ Klines manager loading failed: {e}")
            return None

    async def _load_fallback_data(self) -> Optional[pd.DataFrame]:
        """Fallback data loading when klines_parquet is not available."""
        try:
            # Generate synthetic data for testing
            self.logger.warning("⚠️ Using fallback synthetic data - klines_parquet not available")

            # Create synthetic OHLCV data
            n_samples = 1000
            dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='15min')

            # Generate synthetic price data
            np.random.seed(42)
            base_price = 100.0
            returns = np.random.normal(0, 0.01, n_samples)
            prices = base_price * np.exp(np.cumsum(returns))

            # Generate OHLCV data
            data = pd.DataFrame(index=dates)
            data['open'] = prices
            data['high'] = prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples)))
            data['low'] = prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples)))
            data['close'] = prices * (1 + np.random.normal(0, 0.002, n_samples))
            data['volume'] = np.random.uniform(1000, 10000, n_samples)

            return data

        except Exception as e:
            self.logger.error(f"❌ Fallback data generation failed: {e}")
            return None

    def _validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality.

        Args:
            data: Input DataFrame

        Returns:
            Validation results
        """
        try:
            validation_results = {
                'is_valid': True,
                'issues': [],
                'statistics': {}
            }

            # Check for empty DataFrame
            if len(data) == 0:
                validation_results['issues'].append('DataFrame is empty')
                validation_results['is_valid'] = False
                return validation_results

            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                validation_results['issues'].append(f'Missing required columns: {missing_columns}')
                validation_results['is_valid'] = False

            # Check for missing values
            missing_counts = data.isnull().sum()
            if missing_counts.sum() > 0:
                validation_results['issues'].append(f'Missing values: {missing_counts.sum()}')

            # Check for infinite values
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                inf_counts = np.isinf(data[numeric_cols]).sum()
                if inf_counts.sum() > 0:
                    validation_results['issues'].append(f'Infinite values: {inf_counts.sum()}')

            # Calculate basic statistics
            if len(numeric_cols) > 0:
                validation_results['statistics'] = data[numeric_cols].describe().to_dict()

            return validation_results

        except Exception as e:
            self.logger.warning(f"⚠️ Data validation failed: {e}")
            return {
                'is_valid': False,
                'issues': [f'Validation error: {str(e)}'],
                'error': str(e)
            }

    def _optimize_memory_usage(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Optimize memory usage for data processing.

        Args:
            data: Input DataFrame

        Returns:
            Memory optimization configuration
        """
        try:
            if not self.memory_manager:
                return {'optimization_applied': False}

            # Get current memory usage
            memory_info = self.memory_manager.get_memory_usage()

            # Calculate optimal chunk size
            data_size_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
            available_memory_gb = memory_info.get('available_memory_gb', 4.0)

            # Calculate optimal batch size
            optimal_batch_size = min(
                self.config.batch_size,
                int(available_memory_gb * 1024 * 0.8 / (data_size_mb / len(data)))
            )

            memory_config = {
                'optimization_applied': True,
                'data_size_mb': data_size_mb,
                'available_memory_gb': available_memory_gb,
                'optimal_batch_size': optimal_batch_size,
                'memory_limit_gb': self.config.memory_limit_gb
            }

            return memory_config

        except Exception as e:
            self.logger.warning(f"⚠️ Memory optimization failed: {e}")
            return {'optimization_applied': False, 'error': str(e)}

class DataPipelineManager:
    """Manager for data pipeline operations with coordination between NAS and TAS."""

    def __init__(self, config: DataPipelineConfig):
        """Initialize the data pipeline manager.

        Args:
            config: Data pipeline configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.processor = MarketDataProcessor(config)

        self.logger.info("✅ Data Pipeline Manager initialized")

    async def collect_raw_data(self) -> DataPipelineResult:
        """Collect raw market data for both NAS and TAS regime detection.

        Returns:
            DataPipelineResult with collected data
        """
        try:
            self.logger.info("📊 Collecting raw data for hybrid NAS-TAS regime detection")

            # Load market data
            result = await self.processor.load_market_data()

            if not result.success:
                self.logger.error(f"❌ Raw data collection failed: {result.error_message}")
                return result

            # Add hybrid-specific metadata
            result.metadata['pipeline_type'] = 'hybrid_nas_tas'
            result.metadata['data_source'] = 'klines_parquet' if KLINES_MANAGER_AVAILABLE else 'synthetic'
            result.metadata['collection_timestamp'] = datetime.now().isoformat()

            self.logger.info(f"✅ Raw data collected: {result.data.shape}")
            return result

        except Exception as e:
            self.logger.error(f"❌ Data collection failed: {e}")
            return DataPipelineResult(
                data=pd.DataFrame(),
                metadata={'error': str(e)},
                processing_time=0.0,
                success=False,
                error_message=str(e)
            )

    async def prepare_data_for_nas(self, raw_data: pd.DataFrame) -> DataPipelineResult:
        """Prepare data specifically for NAS regime detection.

        Args:
            raw_data: Raw market data

        Returns:
            DataPipelineResult with NAS-prepared data
        """
        try:
            self.logger.info("🧠 Preparing data for NAS regime detection")

            # NAS-specific data preparation
            nas_data = raw_data.copy()

            # Add NAS-specific features if needed
            # (This would be expanded based on NAS requirements)

            processing_time = 0.1  # Placeholder

            metadata = {
                'preparation_type': 'nas_regime',
                'original_shape': raw_data.shape,
                'prepared_shape': nas_data.shape,
                'preparation_timestamp': datetime.now().isoformat()
            }

            self.logger.info(f"✅ Data prepared for NAS: {nas_data.shape}")

            return DataPipelineResult(
                data=nas_data,
                metadata=metadata,
                processing_time=processing_time,
                success=True
            )

        except Exception as e:
            self.logger.error(f"❌ NAS data preparation failed: {e}")
            return DataPipelineResult(
                data=pd.DataFrame(),
                metadata={'error': str(e)},
                processing_time=0.0,
                success=False,
                error_message=str(e)
            )

    async def prepare_data_for_tas(self, raw_data: pd.DataFrame) -> DataPipelineResult:
        """Prepare data specifically for TAS regime detection.

        Args:
            raw_data: Raw market data

        Returns:
            DataPipelineResult with TAS-prepared data
        """
        try:
            self.logger.info("🌳 Preparing data for TAS regime detection")

            # TAS-specific data preparation
            tas_data = raw_data.copy()

            # Add TAS-specific features if needed
            # (This would be expanded based on TAS requirements)

            processing_time = 0.1  # Placeholder

            metadata = {
                'preparation_type': 'tas_regime',
                'original_shape': raw_data.shape,
                'prepared_shape': tas_data.shape,
                'preparation_timestamp': datetime.now().isoformat()
            }

            self.logger.info(f"✅ Data prepared for TAS: {tas_data.shape}")

            return DataPipelineResult(
                data=tas_data,
                metadata=metadata,
                processing_time=processing_time,
                success=True
            )

        except Exception as e:
            self.logger.error(f"❌ TAS data preparation failed: {e}")
            return DataPipelineResult(
                data=pd.DataFrame(),
                metadata={'error': str(e)},
                processing_time=0.0,
                success=False,
                error_message=str(e)
            )

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status.

        Returns:
            Pipeline status information
        """
        try:
            status = {
                'pipeline_active': True,
                'config': {
                    'symbol': self.config.symbol,
                    'timeframe': self.config.timeframe,
                    'data_type': self.config.data_type
                },
                'capabilities': {
                    'hardware_acceleration': self.processor.hardware_accelerator is not None,
                    'matrix_operations': self.processor.matrix_ops is not None,
                    'klines_manager': KLINES_MANAGER_AVAILABLE
                },
                'timestamp': datetime.now().isoformat()
            }

            return status

        except Exception as e:
            self.logger.error(f"❌ Status retrieval failed: {e}")
            return {'pipeline_active': False, 'error': str(e)}

def create_data_pipeline_manager(config: DataPipelineConfig) -> DataPipelineManager:
    """Create a data pipeline manager instance.

    Args:
        config: Data pipeline configuration

    Returns:
        DataPipelineManager instance
    """
    return DataPipelineManager(config)
