"""
NAS-TAS Clustering Component.

This component performs advanced regime clustering using combined Neural Architecture Search (NAS)
and Tree-based Architecture Search (TAS) approaches. It leverages the unified clustering algorithms
from the hybrid NAS-TAS regime system for superior clustering quality and economic awareness.

Enhanced with comprehensive error handling, tprint logging, and full utility integration.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import traceback
import sys
import os

# Import base component
from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult

# Import utility modules
from src.utils.tprint import (
    tprint, tprint_info, tprint_error, tprint_warning, tprint_success, 
    tprint_debug, tprint_performance, tprint_progress
)
from src.utils.common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
    safe_apply_function, create_summary_statistics, safe_drop_columns,
    safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, safe_filter_dataframe, create_data_quality_report,
    safe_to_parquet, safe_read_parquet, optimize_dataframe_dtypes,
    safe_fillna, safe_divide, safe_log, safe_sqrt, safe_power,
    validate_finite, validate_positive, validate_range, safe_mean, safe_std,
    safe_float, safe_int, safe_kelly_calculation, safe_weighted_average,
    safe_percentage_change, safe_correlation, safe_covariance, safe_percentile,
    validate_correlation_matrix, safe_matrix_inverse, math_safe,
    ensure_directory, safe_file_exists, safe_json_dump, safe_json_load,
    get_current_datetime, format_datetime, safe_sleep, timed_operation,
    format_bytes, chunked_iterable, parallel_map, validate_file_path,
    get_file_size, check_disk_space, memory_checkpoint, gpu_context,
    optimize_memory, get_memory_usage, integrate_with_m1_optimizers,
    cleanup_m1_optimizers, get_m1_gpu_manager, get_m1_memory_optimizer,
    get_m1_cpu_optimizer
)
from src.utils.common_utilities import (
    CommonUtilities, safe_dataframe_operation as safe_df_op,
    validate_dataframe_columns as validate_df_cols, safe_convert_dtypes as safe_conv_dtypes,
    calculate_data_quality_metrics as calc_quality_metrics, safe_merge_dataframes as safe_merge,
    safe_groupby_operation as safe_groupby, safe_apply_function as safe_apply,
    create_summary_statistics as create_summary, safe_drop_columns as safe_drop,
    safe_rename_columns as safe_rename, validate_timestamp_column as validate_ts,
    safe_timestamp_conversion as safe_ts_conv, get_dataframe_info as get_df_info,
    safe_filter_dataframe as safe_filter, create_data_quality_report as create_quality_report
)
from src.utils.math_validation import (
    MathValidation, safe_divide as math_safe_divide, safe_log as math_safe_log,
    safe_sqrt as math_safe_sqrt, safe_power as math_safe_power,
    validate_finite as math_validate_finite, validate_positive as math_validate_positive,
    validate_range as math_validate_range, validate_numeric_array as math_validate_numeric_array,
    safe_kelly_calculation as math_safe_kelly, safe_weighted_average as math_safe_weighted,
    safe_percentage_change as math_safe_pct_change, safe_correlation as math_safe_corr,
    safe_covariance as math_safe_cov, safe_mean as math_safe_mean,
    safe_std as math_safe_std, safe_percentile as math_safe_percentile,
    validate_correlation_matrix as math_validate_corr_matrix,
    safe_matrix_inverse as math_safe_matrix_inv, math_safe as math_safe_func,
    MathValidationError
)
from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)
from src.utils.data.klines_parquet import (
    KlinesParquetManager, get_klines_manager, read_ethusdt_data,
    save_klines_to_parquet, load_klines_from_parquet, validate_klines_data
)
from src.utils.hardware.m1_gpu_utils import (
    M1GPUManager, get_m1_gpu_manager as get_m1_gpu, is_m1_available,
    is_mps_available, optimize_dataframe_for_m1, create_m1_optimized_array,
    m1_backtesting_simulate, m1_monte_carlo_simulate
)

# Import ML utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer, TPEConfig
    )
    BAYESIAN_TPE_AVAILABLE = True
except ImportError:
    BAYESIAN_TPE_AVAILABLE = False
    tprint_warning("⚠️ Bayesian TPE optimizer not available")

# Import matrix operations
try:
    from src.utils.matrix_operations import MatrixOperations
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False
    tprint_warning("⚠️ Matrix operations not available")

# Import NAS-TAS utilities
try:
    from src.utils.nas_tas import NASTASManager, NASTASConfig
    NAS_TAS_AVAILABLE = True
except ImportError:
    NAS_TAS_AVAILABLE = False
    tprint_warning("⚠️ NAS-TAS utilities not available")

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class NASTASClusteringConfig(ComponentConfig):
    """Enhanced configuration for NAS-TAS clustering component with comprehensive validation."""
    symbol: str = "ETHUSDT"
    timeframe: str = "15m"
    exchange: str = "binance"

    # Clustering parameters
    n_regimes: int = 8
    algorithm_type: str = "adaptive_clustering"
    enable_economic_clustering: bool = True
    enable_ensemble_clustering: bool = True

    # Economic clustering weights
    economic_weight: float = 0.3
    momentum_weight: float = 0.25
    volume_weight: float = 0.25

    # Feature configuration
    feature_categories: List[str] = None
    use_standardized_features: bool = True

    # Output configuration
    output_dir: str = "data_cache"
    save_intermediate_results: bool = True

    # Enhanced configuration options
    enable_m1_optimization: bool = True
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_bayesian_optimization: bool = True
    
    # Performance tuning
    max_memory_usage_gb: float = 8.0
    chunk_size: int = 10000
    parallel_workers: int = 4
    
    # Validation settings
    enable_input_validation: bool = True
    enable_output_validation: bool = True
    validation_strict_mode: bool = False
    
    # Error handling
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    enable_fallback_mode: bool = True
    
    # Logging configuration
    log_level: str = "INFO"
    enable_performance_logging: bool = True
    enable_debug_logging: bool = False

    def __post_init__(self):
        """Post-initialization validation and setup."""
        try:
            # Validate and set default feature categories
            if self.feature_categories is None:
                self.feature_categories = ['momentum', 'volatility', 'volume', 'trend', 'price_action']
            
            # Validate numeric parameters
            self.n_regimes = math_validate_positive(self.n_regimes, "n_regimes")
            self.economic_weight = math_validate_range(self.economic_weight, 0.0, 1.0, "economic_weight")
            self.momentum_weight = math_validate_range(self.momentum_weight, 0.0, 1.0, "momentum_weight")
            self.volume_weight = math_validate_range(self.volume_weight, 0.0, 1.0, "volume_weight")
            
            # Validate timeframe
            valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
            if self.timeframe not in valid_timeframes:
                raise ValueError(f"Invalid timeframe: {self.timeframe}. Must be one of {valid_timeframes}")
            
            # Validate symbol format
            if not isinstance(self.symbol, str) or len(self.symbol) < 3:
                raise ValueError(f"Invalid symbol: {self.symbol}. Must be a valid trading symbol.")
            
            # Ensure output directory exists
            ensure_directory(self.output_dir)
            
            tprint_info(f"✅ NAS-TAS Clustering configuration validated successfully")
            
        except Exception as e:
            tprint_error(f"❌ Configuration validation failed: {e}")
            raise ValueError(f"Invalid configuration: {e}") from e


class NASTASClusteringComponent(BaseMarketAnalysisComponent):
    """
    Enhanced NAS-TAS Clustering Component.

    Performs advanced regime clustering using combined NAS and TAS approaches
    with comprehensive error handling, utility integration, and M1 optimization.
    """

    def __init__(self, config: Optional[NASTASClusteringConfig] = None):
        """Initialize the enhanced NAS-TAS clustering component."""
        try:
            super().__init__(config)
            
            # Initialize utility managers
            self.common_utils = CommonUtilities()
            self.math_validator = MathValidation()
            self.serializer = UniversalSerializer()
            self.klines_manager = get_klines_manager()
            
            # Initialize M1 optimizations
            self.m1_gpu_manager = get_m1_gpu() if is_m1_available() else None
            self.m1_memory_optimizer = get_m1_memory_optimizer() if is_m1_available() else None
            self.m1_cpu_optimizer = get_m1_cpu_optimizer() if is_m1_available() else None
            
            # Initialize ML utilities
            self.bayesian_optimizer = None
            if BAYESIAN_TPE_AVAILABLE and config and config.enable_bayesian_optimization:
                try:
                    self.bayesian_optimizer = BayesianTPEOptimizer()
                    tprint_info("✅ Bayesian TPE optimizer initialized")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to initialize Bayesian optimizer: {e}")
            
            # Initialize matrix operations
            self.matrix_ops = None
            if MATRIX_OPS_AVAILABLE:
                try:
                    self.matrix_ops = MatrixOperations()
                    tprint_info("✅ Matrix operations initialized")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to initialize matrix operations: {e}")
            
            # Initialize NAS-TAS manager
            self.nas_tas_manager = None
            if NAS_TAS_AVAILABLE:
                try:
                    self.nas_tas_manager = NASTASManager()
                    tprint_info("✅ NAS-TAS manager initialized")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to initialize NAS-TAS manager: {e}")
            
            # Component state
            self.unified_clustering = None
            self.clustering_result = None
            self.execution_metadata = {}
            self.performance_metrics = {}
            self.error_count = 0
            self.retry_count = 0
            
            # Initialize M1 optimizations if available
            if self.config and self.config.enable_m1_optimization and is_m1_available():
                try:
                    integration_result = integrate_with_m1_optimizers()
                    if integration_result.get('success', False):
                        tprint_success("✅ M1 optimizations integrated successfully")
                    else:
                        tprint_warning(f"⚠️ M1 integration failed: {integration_result.get('error', 'Unknown error')}")
                except Exception as e:
                    tprint_warning(f"⚠️ M1 optimization setup failed: {e}")
            
            tprint_success("✅ NAS-TAS Clustering Component initialized successfully")
            
        except Exception as e:
            tprint_error(f"❌ Failed to initialize NAS-TAS Clustering Component: {e}")
            tprint_error(f"❌ Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Component initialization failed: {e}") from e

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['nas_tas_clustering_result']

    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute enhanced NAS-TAS clustering with comprehensive error handling and utility integration.

        Args:
            data: Market data for clustering
            pipeline_state: Current pipeline state

        Returns:
            ComponentResult with clustering results
        """
        start_time = get_current_datetime()
        tprint_info('🚀 Starting Enhanced NAS-TAS Clustering')

        try:
            # Initialize execution metadata
            self.execution_metadata = {
                'start_time': start_time,
                'symbol': self.config.symbol if self.config else 'UNKNOWN',
                'timeframe': self.config.timeframe if self.config else '15m',
                'component': 'nas_tas_clustering',
                'version': '2.0_enhanced',
                'retry_count': self.retry_count,
                'error_count': self.error_count
            }

            # Validate inputs
            if self.config and self.config.enable_input_validation:
                validation_result = self._validate_inputs(data, pipeline_state)
                if not validation_result['valid']:
                    raise ValueError(f"Input validation failed: {validation_result['errors']}")

            # Check memory usage before processing
            if self.config and self.config.enable_memory_optimization:
                memory_usage = get_memory_usage()
                memory_gb = memory_usage / (1024 ** 3)
                if memory_gb > self.config.max_memory_usage_gb:
                    tprint_warning(f"⚠️ High memory usage: {memory_gb:.2f}GB (limit: {self.config.max_memory_usage_gb}GB)")
                    optimize_memory()

            # Load and validate market data
            with memory_checkpoint("data_loading"):
                market_data = await self._load_market_data_enhanced(data)
                if market_data is None or market_data.empty:
                    raise ValueError("No market data available for clustering")

            # Prepare features with enhanced processing
            with memory_checkpoint("feature_preparation"):
                features = await self._prepare_features_enhanced(market_data)
                if features is None:
                    raise ValueError("Failed to prepare features for clustering")

            # Initialize clustering with enhanced configuration
            clustering_config = self._create_enhanced_clustering_config()
            self.unified_clustering = await self._initialize_unified_clustering_enhanced(clustering_config)

            # Perform clustering with error handling and retries
            clustering_result = await self._perform_clustering_with_retry(
                features, market_data, clustering_config
            )

            if not clustering_result or not getattr(clustering_result, 'success', False):
                raise ValueError(f"Clustering failed: {getattr(clustering_result, 'error_message', 'Unknown error')}")

            self.clustering_result = clustering_result
            regime_count = len(set(clustering_result.labels)) if clustering_result.labels is not None else 0
            tprint_success(f"✅ NAS-TAS Clustering completed: {regime_count} regimes discovered")

            # Generate enhanced outputs
            with memory_checkpoint("output_generation"):
                outputs = await self._generate_enhanced_outputs(market_data, clustering_result)

            # Update execution metadata with performance metrics
            end_time = get_current_datetime()
            execution_time = (end_time - start_time).total_seconds()
            
            self.execution_metadata.update({
                'end_time': end_time,
                'execution_time': execution_time,
                'success': True,
                'regime_count': regime_count,
                'algorithm_used': getattr(clustering_result, 'algorithm_used', 'unknown'),
                'quality_metrics': getattr(clustering_result, 'quality_metrics', {}),
                'performance_metrics': self.performance_metrics,
                'memory_usage_gb': get_memory_usage() / (1024 ** 3),
                'output_files': outputs.get('output_files', [])
            })

            # Log performance metrics
            if self.config and self.config.enable_performance_logging:
                tprint_performance("NAS-TAS Clustering", execution_time)
                tprint_info(f"📊 Memory usage: {self.execution_metadata['memory_usage_gb']:.2f}GB")
                tprint_info(f"📊 Regimes discovered: {regime_count}")
                tprint_info(f"📊 Data points processed: {len(market_data)}")

            cluster_assignments = clustering_result.labels.tolist() if clustering_result.labels is not None else []
            cluster_centers = (
                clustering_result.cluster_centers.tolist()
                if hasattr(clustering_result, 'cluster_centers') and clustering_result.cluster_centers is not None
                else []
            )
            probabilities = (
                clustering_result.probabilities.tolist()
                if hasattr(clustering_result, 'probabilities') and clustering_result.probabilities is not None
                else []
            )
            regime_characteristics = getattr(clustering_result, 'regime_characteristics', {})
            nas_clustering_metrics = getattr(clustering_result, 'quality_metrics', {})

            return ComponentResult(
                success=True,
                artifacts={
                    'nas_tas_clustering_result': {
                        'regime_count': regime_count,
                        'total_samples': len(clustering_result.labels) if clustering_result.labels is not None else 0,
                        'regime_assignments': cluster_assignments,
                        'cluster_centers': cluster_centers,
                        'probabilities': probabilities,
                        'quality_metrics': nas_clustering_metrics,
                        'algorithm_used': getattr(clustering_result, 'algorithm_used', 'unknown'),
                        'execution_time': execution_time,
                        'configuration': asdict(self.config) if self.config else {},
                        'execution_info': self.execution_metadata,
                        'performance_metrics': self.performance_metrics,
                        'nas_clusters': {
                            'regime_assignments': cluster_assignments,
                            'cluster_centers': cluster_centers,
                            'regime_characteristics': regime_characteristics,
                            'probabilities': probabilities
                        },
                        'nas_clustering_metrics': nas_clustering_metrics,
                        'cluster_assignments': cluster_assignments
                    }
                },
                metadata={
                    'symbol': self.config.symbol if self.config else 'UNKNOWN',
                    'timeframe': self.config.timeframe if self.config else '15m',
                    'data_points_processed': len(market_data),
                    'regime_count': regime_count,
                    'algorithm_used': getattr(clustering_result, 'algorithm_used', 'unknown'),
                    'execution_successful': True,
                    'execution_time': execution_time,
                    'memory_usage_gb': self.execution_metadata['memory_usage_gb'],
                    'retry_count': self.retry_count,
                    'error_count': self.error_count
                }
            )

        except Exception as e:
            self.error_count += 1
            tprint_error(f'❌ NAS-TAS Clustering failed: {e}')
            tprint_error(f'❌ Traceback: {traceback.format_exc()}')

            # Attempt retry if configured
            if (self.config and self.config.enable_fallback_mode and 
                self.retry_count < self.config.max_retries):
                return await self._handle_retry(data, pipeline_state, e)

            self.execution_metadata.update({
                'end_time': get_current_datetime(),
                'success': False,
                'error': str(e),
                'error_count': self.error_count,
                'retry_count': self.retry_count
            })

            return ComponentResult(
                success=False,
                artifacts={},
                error_message=f"NAS-TAS clustering failed: {str(e)}",
                metadata={
                    'error_count': self.error_count,
                    'retry_count': self.retry_count,
                    'execution_time': (get_current_datetime() - start_time).total_seconds()
                }
            )

    def _validate_inputs(self, data: Any, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data and pipeline state."""
        try:
            errors = []
            warnings = []
            
            # Validate data
            if data is None:
                errors.append("Input data is None")
            elif isinstance(data, pd.DataFrame) and data.empty:
                errors.append("Input DataFrame is empty")
            
            # Validate pipeline state
            if not isinstance(pipeline_state, dict):
                errors.append("Pipeline state must be a dictionary")
            
            # Validate configuration
            if not self.config:
                errors.append("Configuration is required")
            else:
                # Validate required fields
                if not self.config.symbol:
                    errors.append("Symbol is required")
                if not self.config.timeframe:
                    errors.append("Timeframe is required")
                if self.config.n_regimes < 2:
                    errors.append("Number of regimes must be at least 2")
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings
            }
            
        except Exception as e:
            tprint_error(f"❌ Input validation failed: {e}")
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': []
            }

    def _create_enhanced_clustering_config(self) -> Dict[str, Any]:
        """Create enhanced clustering configuration with validation."""
        try:
            config = self.config if self.config else NASTASClusteringConfig()

            clustering_config = {
                'n_regimes': math_validate_positive(config.n_regimes, "n_regimes"),
                'algorithm_type': config.algorithm_type,
                'enable_economic_clustering': config.enable_economic_clustering,
                'enable_ensemble_clustering': config.enable_ensemble_clustering,
                'economic_weight': math_validate_range(config.economic_weight, 0.0, 1.0, "economic_weight"),
                'momentum_weight': math_validate_range(config.momentum_weight, 0.0, 1.0, "momentum_weight"),
                'volume_weight': math_validate_range(config.volume_weight, 0.0, 1.0, "volume_weight"),
                'feature_categories': config.feature_categories,
                'use_standardized_features': config.use_standardized_features,
                'enable_m1_optimization': config.enable_m1_optimization,
                'enable_gpu_acceleration': config.enable_gpu_acceleration,
                'enable_memory_optimization': config.enable_memory_optimization,
                'enable_bayesian_optimization': config.enable_bayesian_optimization,
                'max_memory_usage_gb': config.max_memory_usage_gb,
                'chunk_size': config.chunk_size,
                'parallel_workers': config.parallel_workers
            }

            tprint_info(f"📊 Enhanced clustering configuration: {config.n_regimes} regimes, algorithm: {config.algorithm_type}")
            return clustering_config

        except Exception as e:
            tprint_warning(f"⚠️ Failed to create enhanced clustering config: {e}, using defaults")
            return {
                'n_regimes': 8,
                'algorithm_type': 'adaptive_clustering',
                'enable_economic_clustering': True,
                'enable_ensemble_clustering': True,
                'economic_weight': 0.3,
                'momentum_weight': 0.25,
                'volume_weight': 0.25,
                'feature_categories': ['momentum', 'volatility', 'volume', 'trend', 'price_action'],
                'use_standardized_features': True,
                'enable_m1_optimization': True,
                'enable_gpu_acceleration': True,
                'enable_memory_optimization': True,
                'enable_bayesian_optimization': True,
                'max_memory_usage_gb': 8.0,
                'chunk_size': 10000,
                'parallel_workers': 4
            }

    async def _load_market_data_enhanced(self, data: Any) -> Optional[pd.DataFrame]:
        """Load and validate market data with enhanced error handling."""
        try:
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                tprint_warning("⚠️ No market data provided, attempting to load from klines_parquet")

                symbol = self.config.symbol if self.config else 'ETHUSDT'
                timeframe = self.config.timeframe if self.config else '15m'

                tprint_info(f"📊 Loading {symbol} {timeframe} data using enhanced klines_parquet manager")

                # Try processed data first
                market_data = self.klines_manager.read_data(symbol, timeframe, data_type="processed")

                if market_data is None or market_data.empty:
                    # Fallback to raw data
                    market_data = self.klines_manager.read_data(symbol, timeframe, data_type="raw")

                if market_data is None or market_data.empty:
                    tprint_error(f"❌ No data available for {symbol} {timeframe}")
                    return None

                # Validate loaded data
                validation_result = validate_klines_data(market_data)
                if not validation_result['valid']:
                    tprint_warning(f"⚠️ Data validation issues: {validation_result['errors']}")
                    if validation_result['warnings']:
                        tprint_warning(f"⚠️ Data warnings: {validation_result['warnings']}")

                tprint_success(f"✅ Loaded {len(market_data)} rows of {symbol} {timeframe} data")
                return market_data

            # If data is already a DataFrame, validate and optimize it
            if isinstance(data, pd.DataFrame):
                tprint_info(f"📊 Using provided DataFrame with {len(data)} rows")
                
                # Validate the data
                validation_result = validate_klines_data(data)
                if not validation_result['valid']:
                    tprint_warning(f"⚠️ Data validation issues: {validation_result['errors']}")
                
                # Optimize for M1 if available
                if self.config and self.config.enable_m1_optimization and is_m1_available():
                    data = optimize_dataframe_for_m1(data)
                    tprint_info("✅ DataFrame optimized for M1")
                
                return data.copy()

            return None

        except Exception as e:
            tprint_error(f"❌ Error loading market data: {e}")
            tprint_error(f"❌ Traceback: {traceback.format_exc()}")
            return None

    async def _prepare_features_enhanced(self, market_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Prepare features for clustering with enhanced processing."""
        try:
            if market_data is None or market_data.empty:
                tprint_error("❌ Market data is None or empty")
                return None

            tprint_info("🔧 Preparing enhanced features for clustering...")
            
            features = []
            feature_names = []

            # Price-based features with validation
            if 'close' in market_data.columns:
                try:
                    # Returns
                    returns = market_data['close'].pct_change().fillna(0)
                    returns = math_validate_numeric_array(returns.values, "returns")
                    features.append(returns.reshape(-1, 1))
                    feature_names.append('returns')

                    # Volatility (rolling std)
                    volatility = returns.rolling(20).std().fillna(0)
                    volatility = math_validate_numeric_array(volatility.values, "volatility")
                    features.append(volatility.reshape(-1, 1))
                    feature_names.append('volatility')

                    # Moving averages ratio
                    sma_20 = market_data['close'].rolling(20).mean().fillna(market_data['close'].iloc[0])
                    ma_ratio = (market_data['close'] / sma_20 - 1).fillna(0)
                    ma_ratio = math_validate_numeric_array(ma_ratio.values, "ma_ratio")
                    features.append(ma_ratio.reshape(-1, 1))
                    feature_names.append('ma_ratio')

                except Exception as e:
                    tprint_warning(f"⚠️ Error processing price features: {e}")

            # Volume features with validation
            if 'volume' in market_data.columns:
                try:
                    volume_ma = market_data['volume'].rolling(20).mean().fillna(market_data['volume'].mean())
                    volume_ratio = (market_data['volume'] / volume_ma).fillna(1)
                    volume_ratio = math_validate_numeric_array(volume_ratio.values, "volume_ratio")
                    features.append(volume_ratio.reshape(-1, 1))
                    feature_names.append('volume_ratio')
                except Exception as e:
                    tprint_warning(f"⚠️ Error processing volume features: {e}")

            # High-low spread with validation
            if all(col in market_data.columns for col in ['high', 'low', 'close']):
                try:
                    hl_spread = ((market_data['high'] - market_data['low']) / market_data['close']).fillna(0)
                    hl_spread = math_validate_numeric_array(hl_spread.values, "hl_spread")
                    features.append(hl_spread.reshape(-1, 1))
                    feature_names.append('hl_spread')
                except Exception as e:
                    tprint_warning(f"⚠️ Error processing HL spread features: {e}")

            # Combine features with validation
            if features:
                try:
                    feature_array = np.hstack(features)
                    
                    # Remove any NaN or infinite values
                    feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    # Validate final feature array
                    feature_array = math_validate_numeric_array(feature_array, "feature_array")
                    
                    tprint_success(f"✅ Prepared {feature_array.shape[1]} features: {feature_names}")
                    return feature_array
                    
                except Exception as e:
                    tprint_error(f"❌ Error combining features: {e}")
                    return None
            else:
                tprint_warning("⚠️ No features could be created, using dummy features")
                return np.random.randn(len(market_data), 5)

        except Exception as e:
            tprint_error(f"❌ Failed to prepare enhanced features: {e}")
            tprint_error(f"❌ Traceback: {traceback.format_exc()}")
            return None

    async def _initialize_unified_clustering_enhanced(self, clustering_config: Dict[str, Any]):
        """Initialize unified clustering algorithm with enhanced error handling."""
        try:
            # Import the unified clustering algorithm
            from src.utils.nas_tas.shared_utils.unified_clustering_algorithms import (
                UnifiedClusteringAlgorithm
            )

            clustering = UnifiedClusteringAlgorithm(clustering_config)
            tprint_success("✅ Enhanced unified clustering algorithm initialized")
            return clustering

        except ImportError as e:
            tprint_error(f"❌ Failed to import unified clustering: {e}")
            raise ValueError(f"Cannot import unified clustering algorithm: {e}")
        except Exception as e:
            tprint_error(f"❌ Failed to initialize unified clustering: {e}")
            raise RuntimeError(f"Clustering initialization failed: {e}") from e

    async def _perform_clustering_with_retry(self, features: np.ndarray, market_data: pd.DataFrame, clustering_config: Dict[str, Any]):
        """Perform clustering with retry logic and error handling."""
        max_retries = self.config.max_retries if self.config else 3
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    tprint_info(f"🔄 Retry attempt {attempt}/{max_retries}")
                    await asyncio.sleep(self.config.retry_delay_seconds if self.config else 1.0)
                
                # Perform clustering
                clustering_result = self.unified_clustering.cluster_features(
                    features=features,
                    market_data=market_data
                )
                
                if clustering_result and getattr(clustering_result, 'success', False):
                    tprint_success(f"✅ Clustering completed successfully on attempt {attempt + 1}")
                    return clustering_result
                else:
                    error_msg = getattr(clustering_result, 'error_message', 'Unknown clustering error')
                    tprint_warning(f"⚠️ Clustering failed on attempt {attempt + 1}: {error_msg}")
                    
                    if attempt < max_retries:
                        continue
                    else:
                        raise ValueError(f"Clustering failed after {max_retries + 1} attempts: {error_msg}")
                        
            except Exception as e:
                tprint_warning(f"⚠️ Clustering attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries:
                    continue
                else:
                    raise RuntimeError(f"Clustering failed after {max_retries + 1} attempts: {e}") from e
        
        return None

    async def _handle_retry(self, data: Any, pipeline_state: Dict[str, Any], error: Exception):
        """Handle retry logic for failed executions."""
        try:
            self.retry_count += 1
            tprint_info(f"🔄 Attempting retry {self.retry_count}/{self.config.max_retries}")
            
            # Wait before retry
            await asyncio.sleep(self.config.retry_delay_seconds)
            
            # Attempt execution again
            return await self.execute(data, pipeline_state)
            
        except Exception as retry_error:
            tprint_error(f"❌ Retry {self.retry_count} failed: {retry_error}")
            raise retry_error

    async def _generate_enhanced_outputs(self, market_data: pd.DataFrame, clustering_result) -> Dict[str, Any]:
        """Generate enhanced output files and data structures."""
        try:
            tprint_info("📁 Generating enhanced output files...")

            outputs = {
                'clustering_report': None,
                'regime_assignments': None,
                'cluster_characteristics': None,
                'performance_metrics': None,
                'output_files': []
            }

            # Save clustering report
            if clustering_result:
                report_file = await self._save_enhanced_clustering_report(clustering_result)
                outputs['clustering_report'] = report_file
                outputs['output_files'].append(report_file)

                # Generate regime assignments
                regime_data = await self._generate_enhanced_regime_assignments(market_data, clustering_result)
                if regime_data is not None:
                    regime_file = await self._save_enhanced_regime_assignments(regime_data)
                    outputs['regime_assignments'] = regime_file
                    outputs['output_files'].append(regime_file)

                # Generate cluster characteristics
                characteristics = await self._generate_enhanced_cluster_characteristics(market_data, clustering_result)
                if characteristics:
                    char_file = await self._save_enhanced_cluster_characteristics(characteristics)
                    outputs['cluster_characteristics'] = char_file
                    outputs['output_files'].append(char_file)

                # Generate performance metrics
                performance_metrics = await self._generate_performance_metrics(clustering_result)
                if performance_metrics:
                    self.performance_metrics = performance_metrics
                    perf_file = await self._save_performance_metrics(performance_metrics)
                    outputs['performance_metrics'] = perf_file
                    outputs['output_files'].append(perf_file)

            tprint_success(f"✅ Generated {len(outputs['output_files'])} output files")
            return outputs

        except Exception as e:
            tprint_error(f"❌ Failed to generate enhanced outputs: {e}")
            tprint_error(f"❌ Traceback: {traceback.format_exc()}")
            return outputs

    async def _save_enhanced_clustering_report(self, clustering_result) -> str:
        """Save enhanced clustering report to file."""
        try:
            output_dir = Path(self.config.output_dir) / "nas_tas_clustering" / (self.config.symbol if self.config else 'UNKNOWN')
            ensure_directory(output_dir)

            timestamp = format_datetime(get_current_datetime(), '%Y%m%d_%H%M%S')
            filename = f"nas_tas_clustering_report_enhanced_{timestamp}.json"
            filepath = output_dir / filename

            report_data = {
                'clustering_result': {
                    'regime_count': len(set(clustering_result.labels)) if clustering_result.labels is not None else 0,
                    'algorithm_used': getattr(clustering_result, 'algorithm_used', 'unknown'),
                    'quality_metrics': getattr(clustering_result, 'quality_metrics', {}),
                    'execution_time': getattr(clustering_result, 'execution_time', 0.0),
                    'success': getattr(clustering_result, 'success', False)
                },
                'metadata': self.execution_metadata,
                'config': asdict(self.config) if self.config else {},
                'performance_metrics': self.performance_metrics,
                'system_info': {
                    'memory_usage_gb': get_memory_usage() / (1024 ** 3),
                    'm1_available': is_m1_available(),
                    'mps_available': is_mps_available(),
                    'error_count': self.error_count,
                    'retry_count': self.retry_count
                }
            }

            success = safe_json_dump(report_data, filepath, indent=2, default=str)
            if success:
                tprint_success(f"💾 Enhanced clustering report saved to: {filepath}")
                return str(filepath)
            else:
                tprint_error(f"❌ Failed to save clustering report to: {filepath}")
                return ""

        except Exception as e:
            tprint_error(f"❌ Failed to save enhanced clustering report: {e}")
            return ""

    async def _save_enhanced_regime_assignments(self, regime_data: pd.DataFrame) -> str:
        """Save enhanced regime assignments to file."""
        try:
            output_dir = Path(self.config.output_dir) / "nas_tas_clustering" / (self.config.symbol if self.config else 'UNKNOWN')
            ensure_directory(output_dir)

            timestamp = format_datetime(get_current_datetime(), '%Y%m%d_%H%M%S')
            filename = f"nas_tas_regime_assignments_enhanced_{timestamp}.parquet"
            filepath = output_dir / filename

            success = safe_to_parquet(regime_data, filepath, compression='snappy')
            if success:
                tprint_success(f"💾 Enhanced regime assignments saved to: {filepath}")
                return str(filepath)
            else:
                tprint_error(f"❌ Failed to save regime assignments to: {filepath}")
                return ""

        except Exception as e:
            tprint_error(f"❌ Failed to save enhanced regime assignments: {e}")
            return ""

    async def _save_enhanced_cluster_characteristics(self, characteristics: Dict) -> str:
        """Save enhanced cluster characteristics to file."""
        try:
            output_dir = Path(self.config.output_dir) / "nas_tas_clustering" / (self.config.symbol if self.config else 'UNKNOWN')
            ensure_directory(output_dir)

            timestamp = format_datetime(get_current_datetime(), '%Y%m%d_%H%M%S')
            filename = f"nas_tas_cluster_characteristics_enhanced_{timestamp}.json"
            filepath = output_dir / filename

            success = safe_json_dump(characteristics, filepath, indent=2, default=str)
            if success:
                tprint_success(f"💾 Enhanced cluster characteristics saved to: {filepath}")
                return str(filepath)
            else:
                tprint_error(f"❌ Failed to save cluster characteristics to: {filepath}")
                return ""

        except Exception as e:
            tprint_error(f"❌ Failed to save enhanced cluster characteristics: {e}")
            return ""

    async def _generate_enhanced_regime_assignments(self, market_data: pd.DataFrame, clustering_result) -> Optional[pd.DataFrame]:
        """Generate enhanced regime assignments DataFrame."""
        try:
            if clustering_result.labels is None or len(clustering_result.labels) == 0:
                tprint_warning("⚠️ No clustering labels available")
                return None

            # Create DataFrame with regime assignments
            regime_data = pd.DataFrame({
                'timestamp': market_data.index,
                'regime_id': clustering_result.labels,
                'regime_prob': clustering_result.probabilities if hasattr(clustering_result, 'probabilities') and clustering_result.probabilities is not None else np.zeros(len(market_data))
            }).set_index('timestamp')

            # Add additional features if available
            if 'close' in market_data.columns:
                regime_data['price'] = market_data['close']
                regime_data['returns'] = market_data['close'].pct_change().fillna(0)
            
            if 'volume' in market_data.columns:
                regime_data['volume'] = market_data['volume']

            # Optimize DataFrame for M1 if available
            if self.config and self.config.enable_m1_optimization and is_m1_available():
                regime_data = optimize_dataframe_for_m1(regime_data)

            tprint_success(f"✅ Generated regime assignments for {len(regime_data)} data points")
            return regime_data

        except Exception as e:
            tprint_error(f"❌ Failed to generate enhanced regime assignments: {e}")
            return None

    async def _generate_enhanced_cluster_characteristics(self, market_data: pd.DataFrame, clustering_result) -> Dict[str, Any]:
        """Generate enhanced cluster characteristics."""
        try:
            characteristics = {}
            unique_regimes = set(clustering_result.labels) if clustering_result.labels is not None else set()

            for regime_id in unique_regimes:
                regime_mask = clustering_result.labels == regime_id
                regime_data = market_data.iloc[regime_mask] if regime_mask.any() else pd.DataFrame()

                if len(regime_data) > 0:
                    # Calculate comprehensive statistics
                    regime_stats = {
                        'sample_count': len(regime_data),
                        'percentage_of_total': (len(regime_data) / len(market_data)) * 100,
                        'date_range': {
                            'start': regime_data.index.min().isoformat() if hasattr(regime_data.index.min(), 'isoformat') else str(regime_data.index.min()),
                            'end': regime_data.index.max().isoformat() if hasattr(regime_data.index.max(), 'isoformat') else str(regime_data.index.max())
                        }
                    }

                    # Price statistics
                    if 'close' in regime_data.columns:
                        returns = regime_data['close'].pct_change().fillna(0)
                        regime_stats.update({
                            'avg_return': math_safe_mean(returns.values),
                            'volatility': math_safe_std(returns.values),
                            'avg_price': math_safe_mean(regime_data['close'].values),
                            'price_range': {
                                'min': regime_data['close'].min(),
                                'max': regime_data['close'].max()
                            }
                        })

                    # Volume statistics
                    if 'volume' in regime_data.columns:
                        regime_stats.update({
                            'avg_volume': math_safe_mean(regime_data['volume'].values),
                            'volume_volatility': math_safe_std(regime_data['volume'].values),
                            'volume_range': {
                                'min': regime_data['volume'].min(),
                                'max': regime_data['volume'].max()
                            }
                        })

                    # High-Low statistics
                    if all(col in regime_data.columns for col in ['high', 'low']):
                        hl_spread = (regime_data['high'] - regime_data['low']) / regime_data['close']
                        regime_stats.update({
                            'avg_hl_spread': math_safe_mean(hl_spread.values),
                            'hl_spread_volatility': math_safe_std(hl_spread.values)
                        })

                    characteristics[f'regime_{regime_id}'] = regime_stats

            tprint_success(f"✅ Generated characteristics for {len(characteristics)} regimes")
            return characteristics

        except Exception as e:
            tprint_error(f"❌ Failed to generate enhanced cluster characteristics: {e}")
            return {}

    async def _generate_performance_metrics(self, clustering_result) -> Dict[str, Any]:
        """Generate performance metrics for the clustering."""
        try:
            metrics = {
                'clustering_quality': getattr(clustering_result, 'quality_metrics', {}),
                'execution_time': getattr(clustering_result, 'execution_time', 0.0),
                'regime_count': len(set(clustering_result.labels)) if clustering_result.labels is not None else 0,
                'total_samples': len(clustering_result.labels) if clustering_result.labels is not None else 0,
                'algorithm_used': getattr(clustering_result, 'algorithm_used', 'unknown'),
                'success': getattr(clustering_result, 'success', False),
                'system_metrics': {
                    'memory_usage_gb': get_memory_usage() / (1024 ** 3),
                    'm1_optimization_enabled': self.config.enable_m1_optimization if self.config else False,
                    'gpu_acceleration_enabled': self.config.enable_gpu_acceleration if self.config else False,
                    'error_count': self.error_count,
                    'retry_count': self.retry_count
                }
            }

            # Add M1-specific metrics if available
            if is_m1_available():
                metrics['m1_metrics'] = {
                    'm1_available': True,
                    'mps_available': is_mps_available(),
                    'optimization_applied': self.config.enable_m1_optimization if self.config else False
                }

            return metrics

        except Exception as e:
            tprint_error(f"❌ Failed to generate performance metrics: {e}")
            return {}

    async def _save_performance_metrics(self, performance_metrics: Dict[str, Any]) -> str:
        """Save performance metrics to file."""
        try:
            output_dir = Path(self.config.output_dir) / "nas_tas_clustering" / (self.config.symbol if self.config else 'UNKNOWN')
            ensure_directory(output_dir)

            timestamp = format_datetime(get_current_datetime(), '%Y%m%d_%H%M%S')
            filename = f"nas_tas_performance_metrics_{timestamp}.json"
            filepath = output_dir / filename

            success = safe_json_dump(performance_metrics, filepath, indent=2, default=str)
            if success:
                tprint_success(f"💾 Performance metrics saved to: {filepath}")
                return str(filepath)
            else:
                tprint_error(f"❌ Failed to save performance metrics to: {filepath}")
                return ""

        except Exception as e:
            tprint_error(f"❌ Failed to save performance metrics: {e}")
            return ""

    def get_status(self) -> Dict[str, Any]:
        """Get enhanced component status."""
        return {
            'component': 'nas_tas_clustering_enhanced',
            'version': '2.0_enhanced',
            'initialized': self.unified_clustering is not None,
            'has_results': self.clustering_result is not None,
            'execution_metadata': self.execution_metadata,
            'performance_metrics': self.performance_metrics,
            'error_count': self.error_count,
            'retry_count': self.retry_count,
            'utility_status': {
                'common_utils': self.common_utils is not None,
                'math_validator': self.math_validator is not None,
                'serializer': self.serializer is not None,
                'klines_manager': self.klines_manager is not None,
                'm1_gpu_manager': self.m1_gpu_manager is not None,
                'm1_memory_optimizer': self.m1_memory_optimizer is not None,
                'm1_cpu_optimizer': self.m1_cpu_optimizer is not None,
                'bayesian_optimizer': self.bayesian_optimizer is not None,
                'matrix_ops': self.matrix_ops is not None,
                'nas_tas_manager': self.nas_tas_manager is not None
            },
            'system_status': {
                'm1_available': is_m1_available(),
                'mps_available': is_mps_available(),
                'memory_usage_gb': get_memory_usage() / (1024 ** 3)
            }
        }

    def validate_inputs(self) -> List[str]:
        """Enhanced input validation."""
        errors = []

        if not self.config:
            errors.append("Configuration is required")
            return errors

        # Validate required fields
        if not self.config.symbol:
            errors.append("Symbol is required")
        elif not isinstance(self.config.symbol, str) or len(self.config.symbol) < 3:
            errors.append("Symbol must be a valid trading symbol (at least 3 characters)")

        if not self.config.timeframe:
            errors.append("Timeframe is required")
        else:
            valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
            if self.config.timeframe not in valid_timeframes:
                errors.append(f"Invalid timeframe: {self.config.timeframe}. Must be one of {valid_timeframes}")

        if self.config.n_regimes < 2:
            errors.append("Number of regimes must be at least 2")
        elif self.config.n_regimes > 50:
            errors.append("Number of regimes should not exceed 50 for performance reasons")

        # Validate weights
        total_weight = self.config.economic_weight + self.config.momentum_weight + self.config.volume_weight
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"Weights must sum to 1.0, got {total_weight:.3f}")

        # Validate memory settings
        if self.config.max_memory_usage_gb <= 0:
            errors.append("Max memory usage must be positive")
        elif self.config.max_memory_usage_gb > 100:
            errors.append("Max memory usage should not exceed 100GB")

        # Validate performance settings
        if self.config.chunk_size <= 0:
            errors.append("Chunk size must be positive")
        if self.config.parallel_workers <= 0:
            errors.append("Parallel workers must be positive")
        elif self.config.parallel_workers > 32:
            errors.append("Parallel workers should not exceed 32")

        return errors

    def cleanup(self):
        """Cleanup resources and optimizers."""
        try:
            tprint_info("🧹 Cleaning up NAS-TAS Clustering Component...")
            
            # Cleanup M1 optimizers
            if is_m1_available():
                cleanup_m1_optimizers()
                tprint_info("✅ M1 optimizers cleaned up")
            
            # Clear large objects
            self.unified_clustering = None
            self.clustering_result = None
            self.performance_metrics = {}
            
            # Force garbage collection
            import gc
            gc.collect()
            
            tprint_success("✅ NAS-TAS Clustering Component cleanup completed")
            
        except Exception as e:
            tprint_warning(f"⚠️ Cleanup failed: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during destruction
