"""
Neural Architecture Search (NAS) Engine

This module provides the core NAS engine with extensive integration of utility modules
for optimal performance, data processing, and hardware optimization.
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
    format_bytes, parallel_map, chunked_iterable
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
    safe_percentile, validate_correlation_matrix, safe_matrix_inverse, math_safe
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

from ...ml_common.optimization.bayesian_entry_timing_optimizer import BayesianEntryTimingOptimizer
from ...ml_common.optimization.grid_utils import GridSearchOptimizer
from ...ml_common.optimization.hpo_utils import HPOUtils
from ...ml_common.optimization.hierarchical_hpo import HierarchicalHPO

from ...matrix_operations.unified_operations import UnifiedMatrixOperations
from ...matrix_operations.enhanced_operations import EnhancedMatrixOperations
from ...matrix_operations.batch_operations import BatchMatrixOperations
from ...matrix_operations.vectorized_core import VectorizedProcessingCore

from ...hardware.m1_gpu_utils import M1GPUManager, is_m1_available, is_mps_available
from ...hardware.m1_memory_optimizer import M1MemoryOptimizer
from ...hardware.m1_cpu_optimizer import M1CPUOptimizer

# Setup logging with tprint integration
logger = logging.getLogger(__name__)

@tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
class NASEngine:
    """
    Neural Architecture Search Engine with extensive utility integration.

    This engine provides comprehensive NAS capabilities with:
    - Extensive use of common operations for data processing
    - Math validation for safe computations
    - Comprehensive logging with tprint
    - Data management with klines parquet utilities
    - Serialization for model persistence
    - M1 hardware optimization
    - Matrix operations for high-performance computations
    - Advanced optimization algorithms
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the NAS Engine with extensive utility integration.

        Args:
            config: Configuration dictionary for NAS engine
        """
        tprint_info("🚀 Initializing NAS Engine with extensive utility integration")
        tprint_debug(f"📋 Configuration provided: {'Yes' if config else 'No'}")

        # Initialize configuration
        self.config = config or {}
        tprint_debug(f"⚙️ NAS Engine config keys: {list(self.config.keys()) if self.config else 'None'}")
        self.logger = logger.getChild("NASEngine")
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

        # Initialize performance tracking
        self.performance_metrics = {}
        tprint_debug("✅ Performance metrics tracking initialized")
        self.search_history = []
        tprint_debug("✅ Search history tracking initialized")

        tprint_success("✅ NAS Engine initialized successfully")
        tprint_info(f"📊 Engine components: {len([attr for attr in dir(self) if not attr.startswith('_')])} public attributes")
        tprint_structured({
            'engine_type': 'NAS',
            'initialization_time': time.time(),
            'm1_integration': self.m1_integration['success'],
            'components_initialized': {
                'utility_classes': True,
                'matrix_operations': True,
                'hardware_optimization': self.m1_integration['success'],
                'optimization_components': True
            }
        }, LogLevel.INFO)

    @tprint_timer("Data Loading and Validation")
    def load_and_validate_data(
        self,
        symbol: str = "ETHUSDT",
        interval: str = "1m",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Load and validate data using extensive utility integration.

        Args:
            symbol: Trading symbol to load
            interval: Data interval
            start_date: Start date for data loading
            end_date: End date for data loading

        Returns:
            Validated DataFrame or None if loading fails
        """
        tprint_info(f"📊 Loading data for {symbol} {interval}")

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

            if data is None or data.empty:
                tprint_error(f"❌ No data loaded for {symbol} {interval}")
                tprint_debug(f"🔍 Data check: data is None={data is None}, data.empty={data.empty if data is not None else 'N/A'}")
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

            # Optimize data types for memory efficiency
            tprint_debug("🔧 Optimizing data types")
            memory_before = data.memory_usage(deep=True).sum()
            data = optimize_dataframe_dtypes(data)
            memory_after = data.memory_usage(deep=True).sum()
            tprint_debug(f"💾 Memory optimization: {memory_before} -> {memory_after} bytes ({(memory_after/memory_before-1)*100:.1f}% change)")

            # Guard against null values
            tprint_debug("🛡️ Applying null value guards")
            null_counts_before = data.isnull().sum().sum()
            data = guard_dataframe_nulls(data, threshold=0.1)
            null_counts_after = data.isnull().sum().sum()
            tprint_debug(f"🔍 Null values: {null_counts_before} -> {null_counts_after}")

            tprint_success(f"✅ Data loaded and validated: {len(data)} records")
            tprint_info(f"📊 Final data summary: {data.shape[0]} rows × {data.shape[1]} columns")
            tprint_structured({
                'data_validation_summary': {
                    'final_shape': data.shape,
                    'memory_usage': get_memory_usage(),
                    'validation_completed': True
                }
            }, LogLevel.SUCCESS)
            return data

        except Exception as e:
            tprint_error(f"❌ Error loading data: {e}")
            self.logger.exception("Data loading error")
            return None

    @tprint_timer("Architecture Search")
    def search_architectures(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        optimization_method: str = "bayesian_tpe",
        n_trials: int = 100
    ) -> Dict[str, Any]:
        """Search for optimal architectures using extensive utility integration.

        Args:
            data: Input data for architecture search
            search_space: Architecture search space
            optimization_method: Optimization method (bayesian_tpe, grid, hierarchical)
            n_trials: Number of optimization trials

        Returns:
            Dictionary with search results and best architecture
        """
        tprint_info(f"🔍 Starting architecture search with {optimization_method}")

        try:
            # Validate input data
            tprint_debug("🔍 Validating input data for architecture search")
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            tprint_debug(f"📋 Required columns: {required_columns}")
            tprint_debug(f"📊 Available columns: {list(data.columns)}")

            if not validate_dataframe_columns(data, required_columns):
                tprint_error("❌ Invalid data columns for architecture search")
                tprint_structured({
                    'validation_error': {
                        'required_columns': required_columns,
                        'available_columns': list(data.columns),
                        'missing_columns': [col for col in required_columns if col not in data.columns]
                    }
                }, LogLevel.ERROR)
                return {}

            tprint_success("✅ Data validation passed for architecture search")

            # Initialize search results
            tprint_debug("📊 Initializing search results structure")
            search_results = {
                'method': optimization_method,
                'n_trials': n_trials,
                'trials': [],
                'best_architecture': None,
                'best_score': -np.inf,
                'search_time': 0,
                'performance_metrics': {}
            }
            tprint_structured({
                'search_configuration': {
                    'optimization_method': optimization_method,
                    'n_trials': n_trials,
                    'data_shape': data.shape,
                    'search_space_keys': list(search_space.keys()) if search_space else []
                }
            }, LogLevel.INFO)

            start_time = time.time()
            tprint_debug(f"⏰ Search start time: {start_time}")

            # Use M1 GPU context if available
            context_type = "GPU" if self.gpu_manager else "Memory"
            tprint_debug(f"🔧 Using {context_type} context for architecture search")

            with gpu_context("architecture_search") if self.gpu_manager else memory_checkpoint("architecture_search"):

                if optimization_method == "bayesian_tpe":
                    tprint_info("🧠 Using Bayesian TPE optimization")
                    tprint_debug(f"🔧 Bayesian TPE parameters: n_trials={n_trials}, search_space_size={len(search_space)}")
                    with tprint_timer("Bayesian TPE Architecture Search"):
                        best_architecture, best_score, trials = self._bayesian_search(
                            data, search_space, n_trials
                        )
                elif optimization_method == "grid":
                    tprint_info("🔧 Using Grid Search optimization")
                    tprint_debug(f"🔧 Grid Search parameters: n_trials={n_trials}, search_space_size={len(search_space)}")
                    with tprint_timer("Grid Search Architecture Search"):
                        best_architecture, best_score, trials = self._grid_search(
                            data, search_space, n_trials
                        )
                elif optimization_method == "hierarchical":
                    tprint_info("🏗️ Using Hierarchical HPO optimization")
                    tprint_debug(f"🔧 Hierarchical HPO parameters: n_trials={n_trials}, search_space_size={len(search_space)}")
                    with tprint_timer("Hierarchical HPO Architecture Search"):
                        best_architecture, best_score, trials = self._hierarchical_search(
                            data, search_space, n_trials
                        )
                else:
                    tprint_error(f"❌ Unknown optimization method: {optimization_method}")
                    tprint_debug(f"📋 Available methods: ['bayesian_tpe', 'grid', 'hierarchical']")
                    return {}

                search_results.update({
                    'best_architecture': best_architecture,
                    'best_score': best_score,
                    'trials': trials
                })

                tprint_info(f"📊 Search results updated: {len(trials)} trials completed")
                tprint_debug(f"🏆 Best architecture found: {bool(best_architecture)}")
                tprint_debug(f"📈 Best score: {best_score:.6f}")

            search_time = time.time() - start_time
            search_results['search_time'] = search_time

            # Calculate performance metrics
            tprint_debug("📊 Calculating architecture search performance metrics")
            search_results['performance_metrics'] = self._calculate_search_metrics(trials)

            tprint_success(f"✅ Architecture search completed in {search_time:.2f}s")
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
                    'performance_metrics_available': bool(search_results['performance_metrics'])
                }
            }, LogLevel.SUCCESS)

            return search_results

        except Exception as e:
            tprint_error(f"❌ Error in architecture search: {e}")
            self.logger.exception("Architecture search error")
            return {}

    def _bayesian_search(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        n_trials: int
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """Perform Bayesian TPE search with extensive utility integration."""
        tprint_debug("🧠 Starting Bayesian TPE search")

        trials = []
        best_score = -np.inf
        best_architecture = None

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

                # Evaluate architecture
                with tprint_timer(f"Trial {trial_idx} evaluation"):
                    score = self._evaluate_architecture(data, trial_params)

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
                    best_architecture = trial_params.copy()
                    tprint_info(f"🏆 New best score: {best_score:.4f} at trial {trial_idx}")

                # Update optimizer
                self.bayesian_optimizer.update(trial_params, score)

                # Memory optimization
                if trial_idx % 10 == 0:
                    optimize_memory()

            tprint_success(f"✅ Bayesian search completed: {len(trials)} trials")
            return best_architecture, best_score, trials

        except Exception as e:
            tprint_error(f"❌ Error in Bayesian search: {e}")
            return {}, -np.inf, []

    def _grid_search(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        n_trials: int
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """Perform Grid Search with extensive utility integration."""
        tprint_debug("🔧 Starting Grid Search")

        trials = []
        best_score = -np.inf
        best_architecture = None

        try:
            # Generate grid parameters
            grid_params = self.grid_optimizer.generate_grid(search_space, max_trials=n_trials)

            if not grid_params:
                tprint_warning("⚠️ Grid search received an empty parameter grid; skipping")
                return {}, -np.inf, []

            total_trials = len(grid_params)
            tprint_info(f"🔧 Grid search: {total_trials} parameter combinations")

            for trial_idx, params in enumerate(grid_params):
                tprint_progress(trial_idx, total_trials, f"Grid search trial {trial_idx}")

                # Evaluate architecture
                with tprint_timer(f"Grid trial {trial_idx} evaluation"):
                    score = self._evaluate_architecture(data, params)

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
                    best_architecture = params.copy()
                    tprint_info(f"🏆 New best score: {best_score:.4f} at trial {trial_idx}")

                # Memory optimization
                if trial_idx % 10 == 0:
                    optimize_memory()

            tprint_success(f"✅ Grid search completed: {len(trials)} trials")
            return best_architecture, best_score, trials

        except Exception as e:
            tprint_error(f"❌ Error in Grid search: {e}")
            return {}, -np.inf, []

    def _hierarchical_search(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        n_trials: int
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """Perform Hierarchical HPO search with extensive utility integration."""
        tprint_debug("🏗️ Starting Hierarchical HPO search")

        trials = []
        best_score = -np.inf
        best_architecture = None

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

                # Evaluate architecture
                with tprint_timer(f"Hierarchical trial {trial_idx} evaluation"):
                    score = self._evaluate_architecture(data, trial_params)

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
                    best_architecture = trial_params.copy()
                    tprint_info(f"🏆 New best score: {best_score:.4f} at trial {trial_idx}")

                # Update hierarchical HPO
                self.hierarchical_hpo.update(trial_params, score)

                # Memory optimization
                if trial_idx % 10 == 0:
                    optimize_memory()

            tprint_success(f"✅ Hierarchical search completed: {len(trials)} trials")
            return best_architecture, best_score, trials

        except Exception as e:
            tprint_error(f"❌ Error in Hierarchical search: {e}")
            return {}, -np.inf, []

    @tprint_timer("Architecture Evaluation")
    def _evaluate_architecture(
        self,
        data: pd.DataFrame,
        architecture_params: Dict[str, Any]
    ) -> float:
        """Evaluate architecture performance with extensive utility integration.

        Args:
            data: Input data for evaluation
            architecture_params: Architecture parameters to evaluate

        Returns:
            Architecture performance score
        """
        try:
            # Validate architecture parameters
            validated_params = {}
            for param, value in architecture_params.items():
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
            with memory_checkpoint("data_preparation"):
                # Create feature matrix using matrix operations
                feature_matrix = self._create_feature_matrix(data)

                # Validate feature matrix
                if not validate_correlation_matrix(feature_matrix):
                    tprint_warning("⚠️ Invalid feature matrix correlation structure")
                    return 0.0

            # Simulate architecture evaluation (placeholder for actual model evaluation)
            with gpu_context("architecture_evaluation") if self.gpu_manager else memory_checkpoint("architecture_evaluation"):
                # Use matrix operations for evaluation
                score = self._compute_architecture_score(feature_matrix, validated_params)

            # Validate score
            score = validate_finite(score, "architecture_score")

            tprint_debug(f"🔍 Architecture evaluation score: {score:.4f}")
            return score

        except Exception as e:
            tprint_error(f"❌ Error evaluating architecture: {e}")
            return 0.0

    def _create_feature_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """Create feature matrix using matrix operations utilities."""
        try:
            # Extract numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            feature_data = data[numeric_cols].values

            # Handle NaN values
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)

            # Use matrix operations for feature engineering
            # Normalize features
            normalized_features = self.matrix_ops.normalize_matrix(feature_data)

            # Add polynomial features
            polynomial_features = self.enhanced_matrix_ops.add_polynomial_features(
                normalized_features, degree=2
            )

            return polynomial_features

        except Exception as e:
            tprint_error(f"❌ Error creating feature matrix: {e}")
            return np.array([])

    def _compute_architecture_score(
        self,
        feature_matrix: np.ndarray,
        params: Dict[str, Any]
    ) -> float:
        """Compute architecture score using matrix operations."""
        try:
            # Use vectorized operations for efficient computation
            # Simulate model performance based on architecture parameters

            # Extract key parameters
            complexity = params.get('complexity', 1.0)
            depth = params.get('depth', 1)
            width = params.get('width', 1)

            # Compute base score using matrix operations
            base_score = self.vectorized_core.compute_performance_metric(
                feature_matrix, complexity, depth, width
            )

            # Apply parameter-based adjustments
            complexity_factor = safe_power(complexity, 0.5)
            depth_factor = safe_log(depth + 1)
            width_factor = safe_sqrt(width)

            # Combine factors
            adjusted_score = safe_weighted_average(
                [base_score, complexity_factor, depth_factor, width_factor],
                [0.7, 0.1, 0.1, 0.1]
            )

            return adjusted_score

        except Exception as e:
            tprint_error(f"❌ Error computing architecture score: {e}")
            return 0.0

    def _calculate_search_metrics(self, trials: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate search performance metrics using math validation utilities."""
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
                'improvement_rate': self._calculate_improvement_rate(scores),
                'convergence_metric': self._calculate_convergence_metric(scores)
            }

            return metrics

        except Exception as e:
            tprint_error(f"❌ Error calculating search metrics: {e}")
            return {}

    def _calculate_improvement_rate(self, scores: List[float]) -> float:
        """Calculate improvement rate using math validation utilities."""
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

    def _calculate_convergence_metric(self, scores: List[float]) -> float:
        """Calculate convergence metric using math validation utilities."""
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

    @tprint_timer("Results Serialization")
    def save_results(
        self,
        results: Dict[str, Any],
        filepath: str
    ) -> bool:
        """Save search results using serialization utilities.

        Args:
            results: Search results to save
            filepath: Path to save results

        Returns:
            True if successful, False otherwise
        """
        try:
            tprint_info(f"💾 Saving results to {filepath}")

            # Add metadata
            results_with_metadata = {
                'results': results,
                'metadata': {
                    'timestamp': time.time(),
                    'nas_engine_version': '1.0.0',
                    'm1_integration': self.m1_integration,
                    'memory_usage': get_memory_usage()
                }
            }

            # Save using universal serializer
            success = self.serializer.save(results_with_metadata, filepath)

            if success:
                tprint_success(f"✅ Results saved successfully to {filepath}")
            else:
                tprint_error(f"❌ Failed to save results to {filepath}")

            return success

        except Exception as e:
            tprint_error(f"❌ Error saving results: {e}")
            return False

    def load_results(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Load search results using serialization utilities.

        Args:
            filepath: Path to load results from

        Returns:
            Loaded results or None if loading fails
        """
        try:
            tprint_info(f"📂 Loading results from {filepath}")

            # Load using universal serializer
            results = self.serializer.load(filepath)

            if results:
                tprint_success(f"✅ Results loaded successfully from {filepath}")
                return results
            else:
                tprint_error(f"❌ Failed to load results from {filepath}")
                return None

        except Exception as e:
            tprint_error(f"❌ Error loading results: {e}")
            return None

    def cleanup(self):
        """Cleanup resources and M1 optimizations."""
        try:
            tprint_info("🧹 Cleaning up NAS Engine resources")

            # Get memory usage before cleanup
            memory_before = get_memory_usage()
            tprint_debug(f"💾 Memory usage before cleanup: {memory_before}")

            # Cleanup M1 optimizers
            tprint_debug("🔧 Cleaning up M1 optimizers")
            cleanup_m1_optimizers()
            tprint_debug("✅ M1 optimizers cleaned up")

            # Clear search history
            search_count = len(self.search_history)
            tprint_debug(f"📊 Clearing {search_count} search history entries")
            self.search_history.clear()

            # Clear performance metrics
            perf_count = len(self.performance_metrics)
            tprint_debug(f"📊 Clearing {perf_count} performance metrics entries")
            self.performance_metrics.clear()

            # Get memory usage after cleanup
            memory_after = get_memory_usage()
            tprint_debug(f"💾 Memory usage after cleanup: {memory_after}")

            tprint_success("✅ NAS Engine cleanup completed")
            tprint_structured({
                'cleanup_summary': {
                    'search_history_cleared': search_count,
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
        tprint_debug("🚪 Entering NAS Engine context manager")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        tprint_debug("🚪 Exiting NAS Engine context manager")

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

# Convenience function for quick NAS usage
def create_nas_engine(config: Optional[Dict[str, Any]] = None) -> NASEngine:
    """Create a NAS engine instance with default configuration.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured NASEngine instance
    """
    tprint_info("🏭 Creating NAS Engine instance")
    tprint_debug(f"📋 Configuration provided: {'Yes' if config else 'No'}")

    if config:
        tprint_debug(f"⚙️ Config keys: {list(config.keys())}")

    engine = NASEngine(config)
    tprint_success("✅ NAS Engine instance created successfully")
    return engine

# Example usage
if __name__ == "__main__":
    # Configure tprint for better output
    tprint_info("🚀 Starting NAS Engine example")
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

    # Create and use NAS engine
    tprint_info("🏭 Creating NAS engine for example usage")
    with create_nas_engine() as nas_engine:
        tprint_info("📊 Starting data loading and validation example")

        # Load data
        data = nas_engine.load_and_validate_data("ETHUSDT", "1m")

        if data is not None:
            tprint_success("✅ Data loaded successfully, proceeding with architecture search")

            # Define search space
            tprint_debug("🔧 Defining architecture search space")
            search_space = {
                'complexity': [1.0, 1.5, 2.0, 2.5, 3.0],
                'depth': [1, 2, 3, 4, 5],
                'width': [8, 16, 32, 64, 128],
                'activation': ['relu', 'tanh', 'sigmoid']
            }

            tprint_info(f"📋 Search space defined: {len(search_space)} parameters")
            tprint_structured({
                'search_space_summary': {
                    'parameter_count': len(search_space),
                    'parameter_names': list(search_space.keys()),
                    'total_combinations': np.prod([len(v) for v in search_space.values()])
                }
            }, LogLevel.INFO)

            # Perform architecture search
            tprint_info("🔍 Starting architecture search")
            with tprint_timer("Complete Architecture Search Example"):
                results = nas_engine.search_architectures(
                    data=data,
                    search_space=search_space,
                    optimization_method="bayesian_tpe",
                    n_trials=50
                )

            # Save results
            if results:
                tprint_info("💾 Saving architecture search results")
                success = nas_engine.save_results(results, "nas_results.json")
                if success:
                    tprint_success("✅ Results saved successfully")
                else:
                    tprint_warning("⚠️ Failed to save results")

                tprint_info("📊 Displaying search results summary")
                tprint_structured(results, LogLevel.INFO)
            else:
                tprint_error("❌ No results to save")
        else:
            tprint_error("❌ Failed to load data, skipping architecture search")

    tprint_success("✅ NAS Engine example completed")
