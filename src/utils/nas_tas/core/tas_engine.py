"""
Trading Architecture Search (TAS) Engine

This module provides the core TAS engine with extensive integration of utility modules
for optimal trading strategy search, data processing, and hardware optimization.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd

# Extensive use of common utilities
from ...common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
    safe_apply_function, create_summary_statistics, safe_drop_columns,
    safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, safe_filter_dataframe, create_data_quality_report,
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
    validate_finite, validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, optimize_dataframe_dtypes,
    safe_to_parquet, safe_read_parquet, integrate_with_m1_optimizers,
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    cleanup_m1_optimizers, memory_checkpoint, gpu_context, optimize_memory,
    get_memory_usage, safe_copy, safe_deepcopy, safe_resample, align_dataframes,
    validate_dataframe_schema, guard_dataframe_nulls, timed_operation,
    format_bytes, parallel_map, chunked_iterable, safe_rolling, safe_groupby_operation,
    safe_apply_function as co_safe_apply_function, create_summary_statistics as co_create_summary_statistics
)

from ...common_utilities import (
    CommonUtilities, safe_dataframe_operation as cu_safe_dataframe_operation,
    validate_dataframe_columns as cu_validate_dataframe_columns,
    calculate_data_quality_metrics as cu_calculate_data_quality_metrics,
    safe_merge_dataframes as cu_safe_merge_dataframes,
    safe_groupby_operation as cu_safe_groupby_operation,
    safe_apply_function as cu_safe_apply_function,
    create_summary_statistics as cu_create_summary_statistics,
    safe_drop_columns as cu_safe_drop_columns,
    safe_rename_columns as cu_safe_rename_columns,
    validate_timestamp_column as cu_validate_timestamp_column,
    safe_timestamp_conversion as cu_safe_timestamp_conversion,
    get_dataframe_info as cu_get_dataframe_info,
    safe_filter_dataframe as cu_safe_filter_dataframe,
    create_data_quality_report as cu_create_data_quality_report
)

from ...math_validation import (
    MathValidation, safe_divide as mv_safe_divide, safe_log as mv_safe_log,
    safe_sqrt as mv_safe_sqrt, safe_power as mv_safe_power,
    validate_finite as mv_validate_finite, validate_positive as mv_validate_positive,
    validate_range as mv_validate_range, safe_kelly_calculation as mv_safe_kelly_calculation,
    safe_weighted_average as mv_safe_weighted_average, safe_percentage_change as mv_safe_percentage_change,
    safe_correlation, safe_covariance, safe_mean as mv_safe_mean, safe_std as mv_safe_std,
    safe_percentile, validate_correlation_matrix, safe_matrix_inverse, math_safe,
    validate_numeric_array
)

from ...tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_structured,
    tprint_with_level, tprint_timer, tprint_logged, configure_tprint,
    get_tprint_config, tprint_context, LogLevel
)

from ...data.klines_parquet import (
    KlinesParquetManager, get_klines_manager, read_ethusdt_data,
    save_klines_to_parquet, load_klines_from_parquet, validate_klines_data
)

from ...serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

# Import data processing utilities
from ...data.processing.data_processing import DataProcessor
from ...data.basic_returns_engineer import BasicReturnsEngineer
from ...data.feature_engineer import FeatureEngineer
from ...data.gap_detector import GapDetector
from ...data.unified_data_utils import UnifiedDataUtils

# Import ML common optimization utilities
from ...ml_common.optimization.bayesian_entry_timing_optimizer import BayesianEntryTimingOptimizer
from ...ml_common.optimization.grid_utils import GridSearchOptimizer
from ...ml_common.optimization.hpo_utils import HPOUtils
from ...ml_common.optimization.hierarchical_hpo import HierarchicalHPO
from ...ml_common.optimization.regime_specific_tpsl_optimizer import RegimeSpecificTPSLOptimizer

# Import matrix operations
from ...matrix_operations.unified_operations import UnifiedMatrixOperations
from ...matrix_operations.enhanced_operations import EnhancedMatrixOperations
from ...matrix_operations.batch_operations import BatchMatrixOperations
from ...matrix_operations.vectorized_core import VectorizedProcessingCore
from ...matrix_operations.convenience import MatrixConvenience

# Import hardware utilities
from ...hardware.m1_gpu_utils import M1GPUManager, is_m1_available, is_mps_available
from ...hardware.m1_memory_optimizer import M1MemoryOptimizer
from ...hardware.m1_cpu_optimizer import M1CPUOptimizer

# Setup logging with tprint integration
logger = logging.getLogger(__name__)

@tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
class TASEngine:
    """
    Trading Architecture Search Engine with extensive utility integration.

    This engine provides comprehensive TAS capabilities with:
    - Extensive use of common operations for data processing
    - Math validation for safe computations
    - Comprehensive logging with tprint
    - Data management with klines parquet utilities
    - Serialization for strategy persistence
    - M1 hardware optimization
    - Matrix operations for high-performance computations
    - Advanced optimization algorithms
    - Data processing pipeline integration
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the TAS Engine with extensive utility integration.

        Args:
            config: Configuration dictionary for TAS engine
        """
        tprint_info("🚀 Initializing TAS Engine with extensive utility integration")
        tprint_debug(f"📋 Configuration provided: {'Yes' if config else 'No'}")

        # Initialize configuration
        self.config = config or {}
        tprint_debug(f"⚙️ TAS Engine config keys: {list(self.config.keys()) if self.config else 'None'}")
        self.logger = logger.getChild("TASEngine")
        tprint_debug(f"📝 Logger initialized: {self.logger.name}")

        # Initialize utility classes
        tprint_debug("🔧 Initializing utility classes")
        self.common_ops = CommonUtilities()
        tprint_debug("✅ CommonUtilities initialized")
        self.math_validator = MathValidation()
        tprint_debug("✅ MathValidation initialized")
        self.klines_manager = get_klines_manager()
        tprint_debug("✅ KlinesParquetManager initialized")
        self.serializer = UniversalSerializer()
        tprint_debug("✅ UniversalSerializer initialized")

        # Initialize data processing utilities
        tprint_debug("🔧 Initializing data processing utilities")
        self.data_processor = DataProcessor()
        tprint_debug("✅ DataProcessor initialized")
        self.returns_engineer = BasicReturnsEngineer()
        tprint_debug("✅ BasicReturnsEngineer initialized")
        self.feature_engineer = FeatureEngineer()
        tprint_debug("✅ FeatureEngineer initialized")
        self.gap_detector = GapDetector()
        tprint_debug("✅ GapDetector initialized")
        self.unified_data_utils = UnifiedDataUtils()
        tprint_debug("✅ UnifiedDataUtils initialized")

        # Initialize matrix operations
        tprint_debug("🔧 Initializing matrix operations")
        self.matrix_ops = UnifiedMatrixOperations()
        tprint_debug("✅ UnifiedMatrixOperations initialized")
        self.enhanced_matrix_ops = EnhancedMatrixOperations()
        tprint_debug("✅ EnhancedMatrixOperations initialized")
        self.batch_matrix_ops = BatchMatrixOperations()
        tprint_debug("✅ BatchMatrixOperations initialized")
        self.vectorized_core = VectorizedProcessingCore()
        tprint_debug("✅ VectorizedProcessingCore initialized")
        self.matrix_convenience = MatrixConvenience()
        tprint_debug("✅ MatrixConvenience initialized")

        # Initialize M1 hardware optimizations
        tprint_debug("🔧 Initializing M1 hardware optimizations")
        self.m1_integration = integrate_with_m1_optimizers()
        tprint_debug(f"🔍 M1 integration result: {self.m1_integration}")
        if self.m1_integration['success']:
            tprint_success("✅ M1 integration successful")
            self.gpu_manager = get_m1_gpu_manager()
            tprint_debug("✅ M1 GPU Manager initialized")
            self.memory_optimizer = get_m1_memory_optimizer()
            tprint_debug("✅ M1 Memory Optimizer initialized")
            self.cpu_optimizer = get_m1_cpu_optimizer()
            tprint_debug("✅ M1 CPU Optimizer initialized")
        else:
            tprint_warning("⚠️ M1 integration failed, using fallback")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            tprint_debug("🔄 Using fallback configurations")

        # Initialize optimization components
        tprint_debug("🔧 Initializing optimization components")
        self.bayesian_optimizer = BayesianEntryTimingOptimizer()
        tprint_debug("✅ BayesianEntryTimingOptimizer initialized")
        self.grid_optimizer = GridSearchOptimizer()
        tprint_debug("✅ GridSearchOptimizer initialized")
        self.hpo_utils = HPOUtils
        tprint_debug("✅ HPOUtils initialized")
        self.hierarchical_hpo = HierarchicalHPO()
        tprint_debug("✅ HierarchicalHPO initialized")
        self.regime_tpsl_optimizer = RegimeSpecificTPSLOptimizer()
        tprint_debug("✅ RegimeSpecificTPSLOptimizer initialized")

        # Initialize performance tracking
        self.performance_metrics = {}
        tprint_debug("✅ Performance metrics tracking initialized")
        self.strategy_history = []
        tprint_debug("✅ Strategy history tracking initialized")
        self.trading_metrics = {}
        tprint_debug("✅ Trading metrics tracking initialized")

        tprint_success("✅ TAS Engine initialized successfully")
        tprint_info(f"📊 Engine components: {len([attr for attr in dir(self) if not attr.startswith('_')])} public attributes")
        tprint_structured({
            'engine_type': 'TAS',
            'initialization_time': time.time(),
            'm1_integration': self.m1_integration['success'],
            'components_initialized': {
                'utility_classes': True,
                'data_processing': True,
                'matrix_operations': True,
                'hardware_optimization': self.m1_integration['success'],
                'optimization_components': True
            }
        }, LogLevel.INFO)

    @tprint_timer("Data Loading and Processing")
    def load_and_process_data(
        self,
        symbol: str = "ETHUSDT",
        interval: str = "1m",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        apply_feature_engineering: bool = True
    ) -> Optional[pd.DataFrame]:
        """Load and process data using extensive utility integration.

        Args:
            symbol: Trading symbol to load
            interval: Data interval
            start_date: Start date for data loading
            end_date: End date for data loading
            apply_feature_engineering: Whether to apply feature engineering

        Returns:
            Processed DataFrame or None if loading fails
        """
        tprint_info(f"📊 Loading and processing data for {symbol} {interval}")

        try:
            # Load data using klines parquet manager
            tprint_debug(f"📊 Loading data with parameters: symbol={symbol}, interval={interval}")
            tprint_debug(f"📅 Date range: {start_date} to {end_date}")

            with memory_checkpoint("data_loading"):
                tprint_debug("🔍 Accessing klines manager for data retrieval")
                data = self.klines_manager.read_data(
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date,
                    data_type="processed"
                )
                tprint_debug(f"📊 Raw data retrieved: {len(data) if data is not None else 0} records")

            if data is None or len(data) == 0:
                tprint_error(f"❌ No data loaded for {symbol} {interval}")
                tprint_debug(f"🔍 Data check: data is None={data is None}, len(data) == 0={len(data) == 0 if data is not None else 'N/A'}")
                return None

            tprint_info(f"📊 Loaded {len(data)} records")
            tprint_debug(f"📋 Data columns: {list(data.columns)}")
            tprint_debug(f"📅 Data date range: {data.index.min()} to {data.index.max()}")
            tprint_structured({
                'data_loading': {
                    'symbol': symbol,
                    'interval': interval,
                    'records_loaded': len(data),
                    'columns_count': len(data.columns),
                    'memory_usage': get_memory_usage()
                }
            }, LogLevel.DEBUG)

            # Validate data using common utilities
            tprint_debug("🔍 Validating data quality")
            validation_result = validate_klines_data(data)
            tprint_debug(f"📋 Validation result: {validation_result}")

            if not validation_result['valid']:
                tprint_error(f"❌ Data validation failed: {validation_result['errors']}")
                tprint_structured({
                    'validation_failure': {
                        'errors': validation_result['errors'],
                        'data_shape': data.shape,
                        'data_types': data.dtypes.to_dict()
                    }
                }, LogLevel.ERROR)
                return None

            tprint_success("✅ Data validation passed")

            # Apply data quality metrics
            tprint_debug("📊 Calculating data quality metrics")
            quality_metrics = calculate_data_quality_metrics(data)
            tprint_info(f"📈 Data quality metrics: {quality_metrics}")
            tprint_structured({
                'data_quality': quality_metrics,
                'data_characteristics': {
                    'shape': data.shape,
                    'null_counts': data.isnull().sum().to_dict(),
                    'memory_usage': data.memory_usage(deep=True).sum()
                }
            }, LogLevel.INFO)

            # Process data using unified data utilities
            tprint_debug(f"🔧 Starting data processing with feature engineering: {apply_feature_engineering}")
            with memory_checkpoint("data_processing"):
                processed_data = self._process_trading_data(data, apply_feature_engineering)

            if processed_data is None or processed_len(data) == 0:
                tprint_error("❌ Data processing failed")
                tprint_debug(f"🔍 Processed data check: data is None={processed_data is None}, len(data) == 0={processed_len(data) == 0 if processed_data is not None else 'N/A'}")
                return None

            tprint_info(f"✅ Data processing completed: {len(processed_data)} records")
            tprint_debug(f"📋 Processed data columns: {list(processed_data.columns)}")

            # Optimize data types for memory efficiency
            tprint_debug("🔧 Optimizing data types")
            memory_before = processed_data.memory_usage(deep=True).sum()
            processed_data = optimize_dataframe_dtypes(processed_data)
            memory_after = processed_data.memory_usage(deep=True).sum()
            tprint_debug(f"💾 Memory optimization: {memory_before} -> {memory_after} bytes ({(memory_after/memory_before-1)*100:.1f}% change)")

            # Guard against null values
            tprint_debug("🛡️ Applying null value guards")
            null_counts_before = processed_data.isnull().sum().sum()
            processed_data = guard_dataframe_nulls(processed_data, threshold=0.1)
            null_counts_after = processed_data.isnull().sum().sum()
            tprint_debug(f"🔍 Null values: {null_counts_before} -> {null_counts_after}")

            # Create final data quality report
            tprint_debug("📊 Creating final data quality report")
            final_quality_report = create_data_quality_report(processed_data)
            tprint_structured(final_quality_report, LogLevel.INFO)

            tprint_success(f"✅ Data loaded and processed: {len(processed_data)} records")
            tprint_info(f"📊 Final data summary: {processed_data.shape[0]} rows × {processed_data.shape[1]} columns")
            tprint_structured({
                'data_processing_summary': {
                    'final_shape': processed_data.shape,
                    'memory_usage': get_memory_usage(),
                    'processing_completed': True,
                    'feature_engineering_applied': apply_feature_engineering
                }
            }, LogLevel.SUCCESS)
            return processed_data

        except Exception as e:
            tprint_error(f"❌ Error loading and processing data: {e}")
            self.logger.exception("Data loading and processing error")
            return None

    def _process_trading_data(
        self,
        data: pd.DataFrame,
        apply_feature_engineering: bool = True
    ) -> Optional[pd.DataFrame]:
        """Process trading data using extensive utility integration."""
        try:
            tprint_debug("🔧 Processing trading data with feature engineering")

            # Make a copy to avoid modifying original data
            processed_data = safe_copy(data)

            # Apply basic returns engineering
            with memory_checkpoint("returns_engineering"):
                processed_data = self.returns_engineer.add_basic_returns(processed_data)

            # Detect gaps in data
            with memory_checkpoint("gap_detection"):
                gaps = self.gap_detector.detect_gaps(processed_data)
                if gaps:
                    tprint_info(f"🔍 Detected {len(gaps)} gaps in data")

            # Apply feature engineering if requested
            if apply_feature_engineering:
                with memory_checkpoint("feature_engineering"):
                    processed_data = self.feature_engineer.add_technical_indicators(processed_data)
                    processed_data = self.feature_engineer.add_price_features(processed_data)
                    processed_data = self.feature_engineer.add_volume_features(processed_data)
                    processed_data = self.feature_engineer.add_time_features(processed_data)

            # Apply unified data processing
            with memory_checkpoint("unified_processing"):
                processed_data = self.unified_data_utils.standardize_data(processed_data)
                processed_data = self.unified_data_utils.add_derived_features(processed_data)

            # Validate processed data
            if not validate_dataframe_columns(processed_data, ['open', 'high', 'low', 'close', 'volume']):
                tprint_error("❌ Processed data missing required columns")
                return None

            tprint_debug(f"🔧 Processed data shape: {processed_data.shape}")
            tprint_debug(f"🔧 Processed data columns: {list(processed_data.columns)}")

            return processed_data

        except Exception as e:
            tprint_error(f"❌ Error processing trading data: {e}")
            return None

    @tprint_timer("Strategy Search")
    def search_strategies(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        optimization_method: str = "bayesian_tpe",
        n_trials: int = 100,
        include_regime_specific: bool = True
    ) -> Dict[str, Any]:
        """Search for optimal trading strategies using extensive utility integration.

        Args:
            data: Input data for strategy search
            search_space: Strategy search space
            optimization_method: Optimization method (bayesian_tpe, grid, hierarchical)
            n_trials: Number of optimization trials
            include_regime_specific: Whether to include regime-specific optimization

        Returns:
            Dictionary with search results and best strategy
        """
        tprint_info(f"🔍 Starting strategy search with {optimization_method}")

        try:
            # Validate input data
            tprint_debug("🔍 Validating input data for strategy search")
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            tprint_debug(f"📋 Required columns: {required_columns}")
            tprint_debug(f"📊 Available columns: {list(data.columns)}")

            if not validate_dataframe_columns(data, required_columns):
                tprint_error("❌ Invalid data columns for strategy search")
                tprint_structured({
                    'validation_error': {
                        'required_columns': required_columns,
                        'available_columns': list(data.columns),
                        'missing_columns': [col for col in required_columns if col not in data.columns]
                    }
                }, LogLevel.ERROR)
                return {}

            tprint_success("✅ Data validation passed for strategy search")

            # Initialize search results
            tprint_debug("📊 Initializing search results structure")
            search_results = {
                'method': optimization_method,
                'n_trials': n_trials,
                'trials': [],
                'best_strategy': None,
                'best_score': -np.inf,
                'search_time': 0,
                'performance_metrics': {},
                'regime_analysis': None
            }
            tprint_structured({
                'search_configuration': {
                    'optimization_method': optimization_method,
                    'n_trials': n_trials,
                    'include_regime_specific': include_regime_specific,
                    'data_shape': data.shape,
                    'search_space_keys': list(search_space.keys()) if search_space else []
                }
            }, LogLevel.INFO)

            start_time = time.time()
            tprint_debug(f"⏰ Search start time: {start_time}")

            # Perform regime analysis if requested
            if include_regime_specific:
                tprint_debug("🔍 Performing regime analysis")
                with tprint_timer("Regime Analysis"):
                    regime_analysis = self._analyze_regimes(data)

                if regime_analysis:
                    tprint_info(f"📊 Regime analysis completed: {len(regime_analysis.get('regime_stats', {}))} regimes detected")
                    tprint_structured({
                        'regime_analysis_summary': {
                            'regimes_detected': len(regime_analysis.get('regime_stats', {})),
                            'regime_types': list(regime_analysis.get('regime_stats', {}).keys()),
                            'analysis_successful': True
                        }
                    }, LogLevel.INFO)
                else:
                    tprint_warning("⚠️ Regime analysis returned empty results")

                search_results['regime_analysis'] = regime_analysis
            else:
                tprint_debug("⏭️ Skipping regime analysis as requested")

            # Use M1 GPU context if available
            context_type = "GPU" if self.gpu_manager else "Memory"
            tprint_debug(f"🔧 Using {context_type} context for strategy search")

            with gpu_context("strategy_search") if self.gpu_manager else memory_checkpoint("strategy_search"):

                if optimization_method == "bayesian_tpe":
                    tprint_info("🧠 Using Bayesian TPE optimization")
                    tprint_debug(f"🔧 Bayesian TPE parameters: n_trials={n_trials}, search_space_size={len(search_space)}")
                    with tprint_timer("Bayesian TPE Strategy Search"):
                        best_strategy, best_score, trials = self._bayesian_strategy_search(
                            data, search_space, n_trials, regime_analysis
                        )
                elif optimization_method == "grid":
                    tprint_info("🔧 Using Grid Search optimization")
                    tprint_debug(f"🔧 Grid Search parameters: n_trials={n_trials}, search_space_size={len(search_space)}")
                    with tprint_timer("Grid Search Strategy Search"):
                        best_strategy, best_score, trials = self._grid_strategy_search(
                            data, search_space, n_trials, regime_analysis
                        )
                elif optimization_method == "hierarchical":
                    tprint_info("🏗️ Using Hierarchical HPO optimization")
                    tprint_debug(f"🔧 Hierarchical HPO parameters: n_trials={n_trials}, search_space_size={len(search_space)}")
                    with tprint_timer("Hierarchical HPO Strategy Search"):
                        best_strategy, best_score, trials = self._hierarchical_strategy_search(
                            data, search_space, n_trials, regime_analysis
                        )
                else:
                    tprint_error(f"❌ Unknown optimization method: {optimization_method}")
                    tprint_debug(f"📋 Available methods: ['bayesian_tpe', 'grid', 'hierarchical']")
                    return {}

                search_results.update({
                    'best_strategy': best_strategy,
                    'best_score': best_score,
                    'trials': trials
                })

                tprint_info(f"📊 Search results updated: {len(trials)} trials completed")
                tprint_debug(f"🏆 Best strategy found: {bool(best_strategy)}")
                tprint_debug(f"📈 Best score: {best_score:.6f}")

            search_time = time.time() - start_time
            search_results['search_time'] = search_time

            # Calculate performance metrics
            tprint_debug("📊 Calculating strategy performance metrics")
            search_results['performance_metrics'] = self._calculate_strategy_metrics(trials)

            tprint_success(f"✅ Strategy search completed in {search_time:.2f}s")
            tprint_info(f"🏆 Best score: {best_score:.4f}")
            tprint_info(f"📊 Total trials: {len(trials)}")

            # Log comprehensive search summary
            tprint_structured({
                'search_summary': {
                    'method': optimization_method,
                    'total_trials': len(trials),
                    'best_score': best_score,
                    'search_time_seconds': search_time,
                    'trials_per_second': len(trials) / search_time if search_time > 0 else 0,
                    'regime_analysis_included': include_regime_specific,
                    'performance_metrics_available': bool(search_results['performance_metrics'])
                }
            }, LogLevel.SUCCESS)

            return search_results

        except Exception as e:
            tprint_error(f"❌ Error in strategy search: {e}")
            self.logger.exception("Strategy search error")
            return {}

    def _analyze_regimes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market regimes using matrix operations and data processing utilities."""
        try:
            tprint_debug("🔍 Analyzing market regimes")

            # Extract price data for regime analysis
            price_data = data[['open', 'high', 'low', 'close', 'volume']].values

            # Use matrix operations for regime detection
            with memory_checkpoint("regime_analysis"):
                # Calculate rolling statistics using matrix operations
                rolling_returns = self.matrix_ops.calculate_rolling_returns(price_data)
                volatility = self.matrix_ops.calculate_rolling_volatility(rolling_returns)
                trend_strength = self.matrix_ops.calculate_trend_strength(price_data)

                # Combine features for regime classification
                regime_features = np.column_stack([volatility, trend_strength])

                # Use vectorized operations for regime classification
                regimes = self.vectorized_core.classify_regimes(regime_features)

            # Calculate regime statistics
            regime_stats = {}
            unique_regimes = np.unique(regimes)

            for regime in unique_regimes:
                regime_mask = regimes == regime
                regime_data = data[regime_mask]

                if not regime_len(data) == 0:
                    regime_stats[f'regime_{regime}'] = {
                        'count': len(regime_data),
                        'percentage': safe_divide(len(regime_data), len(data)) * 100,
                        'avg_volatility': safe_mean(volatility[regime_mask]),
                        'avg_trend': safe_mean(trend_strength[regime_mask]),
                        'avg_return': safe_mean(rolling_returns[regime_mask])
                    }

            tprint_info(f"🔍 Detected {len(unique_regimes)} market regimes")
            return {
                'regimes': regimes,
                'regime_stats': regime_stats,
                'features': {
                    'volatility': volatility,
                    'trend_strength': trend_strength,
                    'returns': rolling_returns
                }
            }

        except Exception as e:
            tprint_error(f"❌ Error in regime analysis: {e}")
            return {}

    def _bayesian_strategy_search(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        n_trials: int,
        regime_analysis: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """Perform Bayesian TPE strategy search with extensive utility integration."""
        tprint_debug("🧠 Starting Bayesian TPE strategy search")

        trials = []
        best_score = -np.inf
        best_strategy = None

        try:
            # Configure Bayesian optimizer
            self.bayesian_optimizer.configure(
                search_space=search_space,
                n_trials=n_trials,
                random_state=42
            )

            for trial_idx in range(n_trials):
                tprint_progress(trial_idx, n_trials, f"Bayesian TPE trial {trial_idx}")

                # Get next trial parameters
                trial_params = self.bayesian_optimizer.suggest()

                # Evaluate strategy
                with tprint_timer(f"Strategy trial {trial_idx} evaluation"):
                    score = self._evaluate_strategy(data, trial_params, regime_analysis)

                # Record trial
                trial_result = {
                    'trial_idx': trial_idx,
                    'params': trial_params,
                    'score': score,
                    'timestamp': time.time()
                }
                trials.append(trial_result)

                # Update best if improved
                if score > best_score:
                    best_score = score
                    best_strategy = trial_params.copy()
                    tprint_info(f"🏆 New best score: {best_score:.4f} at trial {trial_idx}")

                # Update optimizer
                self.bayesian_optimizer.update(trial_params, score)

                # Memory optimization
                if trial_idx % 10 == 0:
                    optimize_memory()

            tprint_success(f"✅ Bayesian strategy search completed: {len(trials)} trials")
            return best_strategy, best_score, trials

        except Exception as e:
            tprint_error(f"❌ Error in Bayesian strategy search: {e}")
            return {}, -np.inf, []

    def _grid_strategy_search(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        n_trials: int,
        regime_analysis: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """Perform Grid Search strategy search with extensive utility integration."""
        tprint_debug("🔧 Starting Grid Search strategy search")

        trials = []
        best_score = -np.inf
        best_strategy = None

        try:
            # Generate grid parameters
            grid_params = self.grid_optimizer.generate_grid(search_space, max_trials=n_trials)

            if not grid_params:
                tprint_warning("⚠️ Grid strategy search received an empty parameter grid; skipping")
                return {}, -np.inf, []

            total_trials = len(grid_params)
            tprint_info(f"🔧 Grid search: {total_trials} parameter combinations")

            for trial_idx, params in enumerate(grid_params):
                tprint_progress(trial_idx, total_trials, f"Grid search trial {trial_idx}")

                # Evaluate strategy
                with tprint_timer(f"Grid trial {trial_idx} evaluation"):
                    score = self._evaluate_strategy(data, params, regime_analysis)

                # Record trial
                trial_result = {
                    'trial_idx': trial_idx,
                    'params': params,
                    'score': score,
                    'timestamp': time.time()
                }
                trials.append(trial_result)

                # Update best if improved
                if score > best_score:
                    best_score = score
                    best_strategy = params.copy()
                    tprint_info(f"🏆 New best score: {best_score:.4f} at trial {trial_idx}")

                # Memory optimization
                if trial_idx % 10 == 0:
                    optimize_memory()

            tprint_success(f"✅ Grid strategy search completed: {len(trials)} trials")
            return best_strategy, best_score, trials

        except Exception as e:
            tprint_error(f"❌ Error in Grid strategy search: {e}")
            return {}, -np.inf, []

    def _hierarchical_strategy_search(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        n_trials: int,
        regime_analysis: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """Perform Hierarchical HPO strategy search with extensive utility integration."""
        tprint_debug("🏗️ Starting Hierarchical HPO strategy search")

        trials = []
        best_score = -np.inf
        best_strategy = None

        try:
            # Configure hierarchical HPO
            self.hierarchical_hpo.configure(
                search_space=search_space,
                n_trials=n_trials,
                hierarchy_levels=3
            )

            for trial_idx in range(n_trials):
                tprint_progress(trial_idx, n_trials, f"Hierarchical HPO trial {trial_idx}")

                # Get next trial parameters
                trial_params = self.hierarchical_hpo.suggest()

                # Evaluate strategy
                with tprint_timer(f"Hierarchical trial {trial_idx} evaluation"):
                    score = self._evaluate_strategy(data, trial_params, regime_analysis)

                # Record trial
                trial_result = {
                    'trial_idx': trial_idx,
                    'params': trial_params,
                    'score': score,
                    'timestamp': time.time()
                }
                trials.append(trial_result)

                # Update best if improved
                if score > best_score:
                    best_score = score
                    best_strategy = trial_params.copy()
                    tprint_info(f"🏆 New best score: {best_score:.4f} at trial {trial_idx}")

                # Update hierarchical HPO
                self.hierarchical_hpo.update(trial_params, score)

                # Memory optimization
                if trial_idx % 10 == 0:
                    optimize_memory()

            tprint_success(f"✅ Hierarchical strategy search completed: {len(trials)} trials")
            return best_strategy, best_score, trials

        except Exception as e:
            tprint_error(f"❌ Error in Hierarchical strategy search: {e}")
            return {}, -np.inf, []

    @tprint_timer("Strategy Evaluation")
    def _evaluate_strategy(
        self,
        data: pd.DataFrame,
        strategy_params: Dict[str, Any],
        regime_analysis: Optional[Dict[str, Any]] = None
    ) -> float:
        """Evaluate strategy performance with extensive utility integration.

        Args:
            data: Input data for evaluation
            strategy_params: Strategy parameters to evaluate
            regime_analysis: Optional regime analysis results

        Returns:
            Strategy performance score
        """
        try:
            # Validate strategy parameters
            validated_params = {}
            for param, value in strategy_params.items():
                try:
                    if isinstance(value, (int, float)):
                        validated_value = validate_finite(value, param)
                        validated_params[param] = validated_value
                    else:
                        validated_params[param] = value
                except ValueError as e:
                    tprint_warning(f"⚠️ Invalid parameter {param}: {e}")
                    continue

            # Prepare data for evaluation
            with memory_checkpoint("strategy_data_preparation"):
                # Create feature matrix using matrix operations
                feature_matrix = self._create_strategy_feature_matrix(data)

                # Validate feature matrix
                if not validate_correlation_matrix(feature_matrix):
                    tprint_warning("⚠️ Invalid feature matrix correlation structure")
                    return 0.0

            # Simulate strategy evaluation (placeholder for actual strategy evaluation)
            with gpu_context("strategy_evaluation") if self.gpu_manager else memory_checkpoint("strategy_evaluation"):
                # Use matrix operations for evaluation
                score = self._compute_strategy_score(feature_matrix, validated_params, regime_analysis)

            # Validate score
            score = validate_finite(score, "strategy_score")

            tprint_debug(f"🔍 Strategy evaluation score: {score:.4f}")
            return score

        except Exception as e:
            tprint_error(f"❌ Error evaluating strategy: {e}")
            return 0.0

    def _create_strategy_feature_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """Create strategy feature matrix using matrix operations utilities."""
        try:
            # Extract numeric columns for strategy features
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            feature_data = data[numeric_cols].values

            # Handle NaN values
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)

            # Validate numeric array
            feature_data = validate_numeric_array(feature_data, "strategy_features")

            # Use matrix operations for feature engineering
            # Normalize features
            normalized_features = self.matrix_ops.normalize_matrix(feature_data)

            # Add technical indicator features
            technical_features = self.enhanced_matrix_ops.add_technical_features(
                normalized_features
            )

            # Add trading-specific features
            trading_features = self.matrix_convenience.add_trading_features(
                technical_features
            )

            return trading_features

        except Exception as e:
            tprint_error(f"❌ Error creating strategy feature matrix: {e}")
            return np.array([])

    def _compute_strategy_score(
        self,
        feature_matrix: np.ndarray,
        params: Dict[str, Any],
        regime_analysis: Optional[Dict[str, Any]] = None
    ) -> float:
        """Compute strategy score using matrix operations and regime analysis."""
        try:
            # Extract key parameters
            entry_threshold = params.get('entry_threshold', 0.5)
            exit_threshold = params.get('exit_threshold', 0.5)
            risk_factor = params.get('risk_factor', 1.0)
            position_size = params.get('position_size', 0.1)

            # Compute base score using matrix operations
            base_score = self.vectorized_core.compute_strategy_performance(
                feature_matrix, entry_threshold, exit_threshold
            )

            # Apply regime-specific adjustments if available
            regime_adjustment = 1.0
            if regime_analysis and 'regime_stats' in regime_analysis:
                regime_adjustment = self._calculate_regime_adjustment(
                    regime_analysis['regime_stats'], params
                )

            # Apply parameter-based adjustments
            risk_adjustment = safe_power(risk_factor, 0.5)
            position_adjustment = safe_sqrt(position_size)

            # Combine factors using math validation utilities
            adjusted_score = safe_weighted_average(
                [base_score, regime_adjustment, risk_adjustment, position_adjustment],
                [0.6, 0.2, 0.1, 0.1]
            )

            return adjusted_score

        except Exception as e:
            tprint_error(f"❌ Error computing strategy score: {e}")
            return 0.0

    def _calculate_regime_adjustment(
        self,
        regime_stats: Dict[str, Any],
        params: Dict[str, Any]
    ) -> float:
        """Calculate regime-specific adjustment factor."""
        try:
            # Simple regime adjustment based on volatility and trend
            adjustments = []

            for regime_key, stats in regime_stats.items():
                if isinstance(stats, dict) and 'avg_volatility' in stats:
                    volatility = stats['avg_volatility']
                    trend = stats.get('avg_trend', 0.0)
                    percentage = stats.get('percentage', 0.0)

                    # Calculate regime-specific performance factor
                    regime_factor = safe_weighted_average(
                        [volatility, abs(trend)],
                        [0.7, 0.3]
                    )

                    # Weight by regime percentage
                    weighted_factor = safe_divide(regime_factor * percentage, 100.0)
                    adjustments.append(weighted_factor)

            if adjustments:
                return safe_mean(np.array(adjustments))
            else:
                return 1.0

        except Exception as e:
            tprint_warning(f"⚠️ Error calculating regime adjustment: {e}")
            return 1.0

    def _calculate_strategy_metrics(self, trials: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate strategy search performance metrics using math validation utilities."""
        try:
            if not trials:
                return {}

            # Extract scores
            scores = [trial['score'] for trial in trials]
            scores_array = np.array(scores)

            # Calculate metrics using math validation utilities
            metrics = {
                'mean_score': safe_mean(scores_array),
                'std_score': safe_std(scores_array),
                'max_score': np.max(scores_array),
                'min_score': np.min(scores_array),
                'median_score': safe_percentile(scores_array, 50.0),
                'q25_score': safe_percentile(scores_array, 25.0),
                'q75_score': safe_percentile(scores_array, 75.0),
                'improvement_rate': self._calculate_strategy_improvement_rate(scores),
                'convergence_metric': self._calculate_strategy_convergence_metric(scores),
                'risk_adjusted_score': self._calculate_risk_adjusted_score(scores)
            }

            return metrics

        except Exception as e:
            tprint_error(f"❌ Error calculating strategy metrics: {e}")
            return {}

    def _calculate_strategy_improvement_rate(self, scores: List[float]) -> float:
        """Calculate strategy improvement rate using math validation utilities."""
        try:
            if len(scores) < 2:
                return 0.0

            improvements = 0
            for i in range(1, len(scores)):
                if scores[i] > scores[i-1]:
                    improvements += 1

            return safe_divide(improvements, len(scores) - 1)

        except Exception:
            return 0.0

    def _calculate_strategy_convergence_metric(self, scores: List[float]) -> float:
        """Calculate strategy convergence metric using math validation utilities."""
        try:
            if len(scores) < 10:
                return 0.0

            # Use last 20% of trials for convergence analysis
            last_portion = max(1, len(scores) // 5)
            recent_scores = scores[-last_portion:]

            # Calculate coefficient of variation
            mean_score = safe_mean(np.array(recent_scores))
            std_score = safe_std(np.array(recent_scores))

            if mean_score == 0:
                return 0.0

            cv = safe_divide(std_score, abs(mean_score))
            return 1.0 - cv  # Lower CV means better convergence

        except Exception:
            return 0.0

    def _calculate_risk_adjusted_score(self, scores: List[float]) -> float:
        """Calculate risk-adjusted score using Kelly criterion."""
        try:
            if len(scores) < 2:
                return 0.0

            # Calculate win rate and average win/loss
            positive_scores = [s for s in scores if s > 0]
            negative_scores = [s for s in scores if s < 0]

            if not positive_scores or not negative_scores:
                return safe_mean(np.array(scores))

            win_rate = safe_divide(len(positive_scores), len(scores))
            avg_win = safe_mean(np.array(positive_scores))
            avg_loss = abs(safe_mean(np.array(negative_scores)))

            # Use Kelly criterion for risk adjustment
            kelly_fraction = safe_kelly_calculation(win_rate, avg_win, avg_loss)

            # Apply Kelly adjustment to mean score
            mean_score = safe_mean(np.array(scores))
            risk_adjusted = mean_score * (1 + kelly_fraction)

            return risk_adjusted

        except Exception:
            return safe_mean(np.array(scores))

    @tprint_timer("Results Serialization")
    def save_results(
        self,
        results: Dict[str, Any],
        filepath: str
    ) -> bool:
        """Save strategy search results using serialization utilities.

        Args:
            results: Strategy search results to save
            filepath: Path to save results

        Returns:
            True if successful, False otherwise
        """
        try:
            tprint_info(f"💾 Saving strategy results to {filepath}")

            # Add metadata
            results_with_metadata = {
                'results': results,
                'metadata': {
                    'timestamp': time.time(),
                    'tas_engine_version': '1.0.0',
                    'm1_integration': self.m1_integration,
                    'memory_usage': get_memory_usage(),
                    'trading_metrics': self.trading_metrics
                }
            }

            # Save using universal serializer
            success = self.serializer.save(results_with_metadata, filepath)

            if success:
                tprint_success(f"✅ Strategy results saved successfully to {filepath}")
            else:
                tprint_error(f"❌ Failed to save strategy results to {filepath}")

            return success

        except Exception as e:
            tprint_error(f"❌ Error saving strategy results: {e}")
            return False

    def load_results(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Load strategy search results using serialization utilities.

        Args:
            filepath: Path to load results from

        Returns:
            Loaded results or None if loading fails
        """
        try:
            tprint_info(f"📂 Loading strategy results from {filepath}")

            # Load using universal serializer
            results = self.serializer.load(filepath)

            if results:
                tprint_success(f"✅ Strategy results loaded successfully from {filepath}")
                return results
            else:
                tprint_error(f"❌ Failed to load strategy results from {filepath}")
                return None

        except Exception as e:
            tprint_error(f"❌ Error loading strategy results: {e}")
            return None

    def cleanup(self):
        """Cleanup resources and M1 optimizations."""
        try:
            tprint_info("🧹 Cleaning up TAS Engine resources")

            # Get memory usage before cleanup
            memory_before = get_memory_usage()
            tprint_debug(f"💾 Memory usage before cleanup: {memory_before}")

            # Cleanup M1 optimizers
            tprint_debug("🔧 Cleaning up M1 optimizers")
            cleanup_m1_optimizers()
            tprint_debug("✅ M1 optimizers cleaned up")

            # Clear strategy history
            strategy_count = len(self.strategy_history)
            tprint_debug(f"📊 Clearing {strategy_count} strategy history entries")
            self.strategy_history.clear()

            # Clear trading metrics
            metrics_count = len(self.trading_metrics)
            tprint_debug(f"📊 Clearing {metrics_count} trading metrics entries")
            self.trading_metrics.clear()

            # Clear performance metrics
            perf_count = len(self.performance_metrics)
            tprint_debug(f"📊 Clearing {perf_count} performance metrics entries")
            self.performance_metrics.clear()

            # Get memory usage after cleanup
            memory_after = get_memory_usage()
            tprint_debug(f"💾 Memory usage after cleanup: {memory_after}")

            tprint_success("✅ TAS Engine cleanup completed")
            tprint_structured({
                'cleanup_summary': {
                    'strategy_history_cleared': strategy_count,
                    'trading_metrics_cleared': metrics_count,
                    'performance_metrics_cleared': perf_count,
                    'memory_before': memory_before,
                    'memory_after': memory_after,
                    'cleanup_successful': True
                }
            }, LogLevel.INFO)

        except Exception as e:
            tprint_error(f"❌ Error during cleanup: {e}")
            tprint_structured({
                'cleanup_error': {
                    'error_message': str(e),
                    'error_type': type(e).__name__,
                    'cleanup_failed': True
                }
            }, LogLevel.ERROR)

    def __enter__(self):
        """Context manager entry."""
        tprint_debug("🚪 Entering TAS Engine context manager")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        tprint_debug("🚪 Exiting TAS Engine context manager")

        if exc_type is not None:
            tprint_error(f"❌ Exception in context manager: {exc_type.__name__}: {exc_val}")
            tprint_structured({
                'context_manager_exception': {
                    'exception_type': exc_type.__name__,
                    'exception_value': str(exc_val),
                    'traceback_available': exc_tb is not None
                }
            }, LogLevel.ERROR)
        else:
            tprint_debug("✅ Context manager exited normally")

        self.cleanup()

# Convenience function for quick TAS usage
def create_tas_engine(config: Optional[Dict[str, Any]] = None) -> TASEngine:
    """Create a TAS engine instance with default configuration.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured TASEngine instance
    """
    tprint_info("🏭 Creating TAS Engine instance")
    tprint_debug(f"📋 Configuration provided: {'Yes' if config else 'No'}")

    if config:
        tprint_debug(f"⚙️ Config keys: {list(config.keys())}")

    engine = TASEngine(config)
    tprint_success("✅ TAS Engine instance created successfully")
    return engine

# Example usage
if __name__ == "__main__":
    # Configure tprint for better output
    tprint_info("🚀 Starting TAS Engine example")
    from ...tprint import TPrintConfig, configure_tprint

    tprint_debug("⚙️ Configuring tprint for enhanced output")
    config = TPrintConfig(
        use_colors=True,
        output_to_console=True,
        enable_structured_logging=True,
        min_log_level=LogLevel.DEBUG
    )
    configure_tprint(config)
    tprint_success("✅ Tprint configuration applied")

    # Create and use TAS engine
    tprint_info("🏭 Creating TAS engine for example usage")
    with create_tas_engine() as tas_engine:
        tprint_info("📊 Starting data loading and processing example")

        # Load and process data
        data = tas_engine.load_and_process_data("ETHUSDT", "1m", apply_feature_engineering=True)

        if data is not None:
            tprint_success("✅ Data loaded successfully, proceeding with strategy search")

            # Define search space
            tprint_debug("🔧 Defining strategy search space")
            search_space = {
                'entry_threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                'exit_threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                'risk_factor': [0.5, 1.0, 1.5, 2.0],
                'position_size': [0.05, 0.1, 0.15, 0.2, 0.25],
                'stop_loss': [0.01, 0.02, 0.03, 0.04, 0.05],
                'take_profit': [0.02, 0.03, 0.04, 0.05, 0.06]
            }

            tprint_info(f"📋 Search space defined: {len(search_space)} parameters")
            tprint_structured({
                'search_space_summary': {
                    'parameter_count': len(search_space),
                    'parameter_names': list(search_space.keys()),
                    'total_combinations': np.prod([len(v) for v in search_space.values()])
                }
            }, LogLevel.INFO)

            # Perform strategy search
            tprint_info("🔍 Starting strategy search")
            with tprint_timer("Complete Strategy Search Example"):
                results = tas_engine.search_strategies(
                    data=data,
                    search_space=search_space,
                    optimization_method="bayesian_tpe",
                    n_trials=50,
                    include_regime_specific=True
                )

            # Save results
            if results:
                tprint_info("💾 Saving strategy search results")
                success = tas_engine.save_results(results, "tas_results.json")
                if success:
                    tprint_success("✅ Results saved successfully")
                else:
                    tprint_warning("⚠️ Failed to save results")

                tprint_info("📊 Displaying search results summary")
                tprint_structured(results, LogLevel.INFO)
            else:
                tprint_error("❌ No results to save")
        else:
            tprint_error("❌ Failed to load data, skipping strategy search")

    tprint_success("✅ TAS Engine example completed")
