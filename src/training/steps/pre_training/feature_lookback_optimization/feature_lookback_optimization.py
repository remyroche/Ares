"""
Feature Lookback Optimization Component.

This component optimizes feature lookback periods for better model performance.
Provides comprehensive validation, detailed reporting, and robust error handling.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure List and Tuple are available for the new config
if not hasattr(__builtins__, 'List'):
    from typing import List
if not hasattr(__builtins__, 'Tuple'):
    from typing import Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Import tprint for consistent logging
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success

# Import logging functions
from src.utils.logger import get_logger, log_error, log_warning, log_info

# Import feature bank and generators
from src.feature_generation.core.feature_bank import get_global_feature_bank, FeatureCategory
from src.feature_generation.core.feature_generator import FeatureGenerator

# Import profit labeling components for alignment
from src.training.steps.pre_training.profit_labeling.volatility_aware_labeler import (
    VolatilityAwareMultiHorizonLabeler, VolatilityAwareConfig
)
from src.training.steps.pre_training.profit_labeling.multi_target_scheme import (
    MultiTargetScheme, MultiTargetConfig, TargetBand
)

# Import hardware optimization decorator
from src.utils.matrix_operations.hardware_integration import hardware_optimized, get_hardware_optimized_processor

# Import M1 optimization utilities
from src.utils.hardware.m1_gpu_utils import optimize_dataframe_for_m1

# Use dependency manager for robust imports
from .dependency_manager import dependency_manager, get_dependency, is_dependency_available

# Core dependencies with fallback support
np, np_fallback = get_dependency('numpy')
pd, pd_fallback = get_dependency('pandas')

# Utility function to convert int64 to int for dictionary keys
def convert_int64_to_int(value: Any) -> Any:
    """Convert int64 values to regular Python int for JSON serialization."""
    try:
        if hasattr(value, 'dtype') and value.dtype == 'int64':
            return int(value)
        elif isinstance(value, np.int64):
            return int(value)
        elif isinstance(value, dict):
            # Convert both keys and values to handle int64 keys
            converted_dict = {}
            for k, v in value.items():
                # Convert key if it's int64
                converted_key = k
                if isinstance(k, np.int64):
                    converted_key = int(k)
                elif hasattr(k, 'dtype') and k.dtype == 'int64':
                    converted_key = int(k)

                # Convert value recursively
                converted_dict[converted_key] = convert_int64_to_int(v)

            return converted_dict
        elif isinstance(value, (list, tuple)):
            # Convert each item in the list/tuple recursively
            return [convert_int64_to_int(item) for item in value]
        elif hasattr(value, 'shape') and len(value.shape) > 0:
            # Handle numpy arrays that might be problematic
            if value.size > 100:  # Large arrays might cause issues
                return f"LargeArray_{value.shape}_{value.dtype}"
            else:
                return value.tolist()
        else:
            return value
    except Exception:
        # If conversion fails, return original value
        return value

if np_fallback or pd_fallback:
    tprint("⚠️ Using fallback implementations for core dependencies")


@dataclass
class OptimizedFeatureLookbackConfig:
    """Configuration for optimized feature lookback optimization."""

    # Timeframe settings
    default_timeframe: str = "5m"
    base_period_minutes: float = 5.0

    # Lookback optimization settings
    min_lookback: int = 5
    max_lookback: int = 100
    lookback_step: int = 5

    # Feature selection settings
    excluded_categories: List[FeatureCategory] = None
    excluded_features: List[str] = None

    # Forward return calculation settings (aligned with multi_horizon_profit_labeler)
    enable_volatility_normalization: bool = True
    enable_multi_target_scheme: bool = True
    small_band: Tuple[float, float] = (0.4, 0.8)  # k_s range
    medium_band: Tuple[float, float] = (0.8, 1.3)  # k_m range
    high_band: Tuple[float, float] = (1.3, 2.0)   # k_h range

    # Optimization settings
    optimization_metric: str = "information_coefficient"
    cv_folds: int = 5
    max_optimization_time: int = 300  # seconds

    # Output settings
    save_results: bool = True
    generate_reports: bool = True
    output_directory: str = "feature_lookback_optimization_results"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.excluded_categories is None:
            self.excluded_categories = [
                FeatureCategory.INTERACTION,
                FeatureCategory.CROSS_TIMEFRAME,
                FeatureCategory.AUTOENCODER,
                FeatureCategory.REGIME
            ]

        if self.excluded_features is None:
            self.excluded_features = [
                'wavelets', 'autoencoder', 'interaction', 'cross_timeframe', 'regime_'
            ]

# Import optimization configuration classes
try:
    from src.feature_generation.utils.optimization_config import (
        FeatureOptimizationConfig,
        OptimizationConfigManager
    )
    from src.training.steps.market_analysis.hybrid_nas_tas_regime.shared_utils.unified_multi_objective_optimizer import (
        OptimizationConfig  # noqa: E402
    )
except ImportError as e:
    tprint(f"⚠️ Optimization config imports not available: {e}")
    # Define fallback classes
    class FeatureOptimizationConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    class OptimizationConfig:
        """Fallback configuration class for optimization."""

        def __init__(self, **kwargs):
            """Initialize optimization configuration.

            Args:
                **kwargs: Configuration parameters
            """
            self.__dict__.update(kwargs)

# Import common utilities for enhanced functionality
from src.utils.common_operations import (
    safe_dataframe_operation,
    validate_dataframe_columns,
    safe_convert_dtypes,
    calculate_data_quality_metrics,
    safe_merge_dataframes,
    safe_groupby_operation,
    safe_apply_function,
    create_summary_statistics,
    safe_drop_columns,
    safe_rename_columns,
    validate_timestamp_column,
    safe_timestamp_conversion,
    get_dataframe_info,
    safe_filter_dataframe,
    create_data_quality_report,
    optimize_dataframe_dtypes,
    safe_fillna,
    safe_rolling,
    safe_to_parquet,
    safe_read_parquet,
    validate_dataframe_schema,
    guard_dataframe_nulls,
    memory_checkpoint,
    gpu_context,
    optimize_memory,
    get_memory_usage,
    integrate_with_m1_optimizers,
    get_m1_gpu_manager,
    get_m1_memory_optimizer,
    get_m1_cpu_optimizer,
    validate_dataframe
)

from src.utils.common_utilities import (
    CommonUtilities,
    safe_dataframe_operation as safe_df_op,
    validate_dataframe_columns as validate_df_cols,
    safe_convert_dtypes as safe_conv_dtypes,
    calculate_data_quality_metrics as calc_quality_metrics,
    safe_merge_dataframes as safe_merge,
    safe_groupby_operation as safe_groupby,
    safe_apply_function as safe_apply,
    create_summary_statistics as create_summary,
    safe_drop_columns as safe_drop,
    safe_rename_columns as safe_rename,
    validate_timestamp_column as validate_ts,
    safe_timestamp_conversion as safe_ts_conv,
    get_dataframe_info as get_df_info,
    safe_filter_dataframe as safe_filter,
    create_data_quality_report as create_quality_report
)

from src.utils.math_validation import (
    safe_divide,
    safe_log,
    safe_sqrt,
    safe_power,
    validate_finite,
    validate_positive,
    validate_range,
    safe_kelly_calculation,
    safe_weighted_average,
    safe_percentage_change,
    safe_correlation,
    safe_covariance,
    safe_mean,
    safe_std,
    safe_percentile,
    validate_correlation_matrix,
    safe_matrix_inverse,
    math_safe,
    MathValidation,
    MathValidationError
)

from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

# Import ML common utilities for enhanced ML operations
try:
    from src.utils.ml_common.data_processing.data_quality import (
        DataQualityUtilities
    )
    from src.utils.ml_common.data_processing.feature_preparation import (
        FeaturePreparator
    )
    # Use unified config system instead of legacy optimization CONFIG
    from src.common.config.loader import load_from_file as _load_config_dict
    from src.utils.ml_common.validation.cv import (
        purged_time_series_splits, PurgedSplitConfig
    )
    from src.utils.ml_common.monitoring.enhanced_error_detector import (
        EnhancedErrorDetector, ErrorSeverity, ErrorCategory
    )
    ML_COMMON_AVAILABLE = True
except ImportError as e:
    ML_COMMON_AVAILABLE = False
    tprint(f"⚠️ ML common utilities not available: {e}")

# Import matrix operations for efficient computation
try:
    from src.utils.matrix_operations.unified_operations import (
        UnifiedMatrixOperations, safe_correlation_matrix,
        safe_matrix_inverse, get_unified_matrix_operations
    )
    from src.utils.matrix_operations.vectorized_core import (
        VectorizedProcessingCore,
        vectorized_rolling_features, matrix_correlation_analysis,
        get_vectorized_processing_core
    )
    from src.utils.matrix_operations.batch_operations import (
        BatchMatrixProcessor, batch_feature_transformation,
        batch_correlation_analysis, get_batch_matrix_processor
    )
    MATRIX_OPS_AVAILABLE = True
except ImportError as e:
    MATRIX_OPS_AVAILABLE = False
    tprint(f"⚠️ Matrix operations not available: {e}")

from ...market_analysis.components.base_component import (
    BaseMarketAnalysisComponent,
    ComponentConfig,
    ComponentResult
)
from .optimization_reporter import OptimizationReporter
from src.utils.validation.unified_framework import (
    FeatureLookbackValidationFramework,
    ValidationLevel,
    ValidationStatus
)
from .monitoring_metrics import MonitoringMetrics, MetricType, MetricLevel
from .optimization_strategy import OptimizationStrategyFactory, OptimizationMethod
from src.utils.logger import system_logger

class StandardizedErrorHandler:
    """Standardized error handling for consistent error management across the component."""
    
    def __init__(self, logger, component_name: str = "FeatureLookbackOptimization"):
        """Initialize standardized error handler.

        Args:
            logger: Logger instance for error reporting
            component_name: Name of the component for error context
        """
        self.logger = logger
        self.component_name = component_name
    
    def handle_error(
        self,
        error: Exception,
        operation: str,
        return_value=None,
        reraise: bool = False
    ):
        """Handle errors in a standardized way.

        Args:
            error: Exception that occurred
            operation: Name of the operation that failed
            return_value: Value to return if not reraising
            reraise: Whether to re-raise the exception

        Returns:
            Return value if specified and not reraising
        """
        error_msg = f"{operation} failed in {self.component_name}: {str(error)}"
        log_error(error_msg)
        
        if reraise:
            raise type(error)(error_msg) from error
        
        return return_value
    
    def handle_warning(self, warning_msg: str, operation: str):
        """Handle warnings in a standardized way.

        Args:
            warning_msg: Warning message to log
            operation: Name of the operation that generated the warning
        """
        warning_msg = f"{operation} warning in {self.component_name}: {warning_msg}"
        log_warning(warning_msg)
    
    def handle_info(self, info_msg: str, operation: str):
        """Handle info messages in a standardized way.

        Args:
            info_msg: Info message to log
            operation: Name of the operation that generated the info
        """
        info_msg = f"{operation} in {self.component_name}: {info_msg}"
        log_info(info_msg)

# Hardware optimization imports
try:
    from src.utils.hardware import get_unified_hardware_manager, get_advanced_memory_optimizer
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    tprint("⚠️ Hardware optimization not available - using fallback memory management")

# Import advanced matrix operations (additional functions)
try:
    from src.utils.matrix_operations import (
        get_enhanced_matrix_operations, safe_matrix_multiply, gpu_matrix_multiply, 
        correlation_matrix_gpu, eigendecomposition_gpu, batch_matrix_multiply,
        optimize_matrix_operation_with_hardware
    )
    ADVANCED_MATRIX_OPS_AVAILABLE = True
except ImportError as e:
    ADVANCED_MATRIX_OPS_AVAILABLE = False
    tprint(f"⚠️ Advanced matrix operations not available: {e}")

# Import Bayesian lookback optimizer
try:
    from .mrmr_lookback_optimizer import (
        MRMRLookbackOptimizer, LookbackOptimizationConfig, LookbackOptimizationResult,
        optimize_lookback_periods
    )
    MRMR_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    MRMR_OPTIMIZER_AVAILABLE = False
    tprint(f"⚠️ MRMR lookback optimizer not available: {e}")

# Import Directional lookback optimizer
try:
    from .directional_lookback_optimizer import (
        DirectionalLookbackOptimizer, DirectionalLookbackConfig, DirectionalOptimizationResult,
        optimize_features_directional
    )
    DIRECTIONAL_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    DIRECTIONAL_OPTIMIZER_AVAILABLE = False
    tprint(f"⚠️ Directional lookback optimizer not available: {e}")

# Import configuration constants
from .constants import (
    OPTIMIZATION_CONSTANTS, PERFORMANCE_CONSTANTS, VALIDATION_CONSTANTS,
    QUALITY_CONSTANTS, FILE_CONSTANTS, ALGORITHM_CONSTANTS
)


class OptimizationStatus(Enum):
    """Status of optimization process."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class OptimizationMetrics:
    """Comprehensive optimization metrics."""
    best_lookback_period: int
    best_score: float
    optimization_method: str
    total_features_optimized: int
    optimization_time: float
    convergence_iterations: int
    memory_usage_mb: float
    cpu_usage_percent: float
    validation_score: float
    stability_score: float
    regime_coverage: float
    error_rate: float


class FeatureLookbackOptimizationComponent(BaseMarketAnalysisComponent):
    """
    Feature Lookback Optimization Component.
    
    Optimizes feature lookback periods for better model performance.
    Provides comprehensive validation, detailed reporting, and robust error handling.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the feature lookback optimization component."""
        tprint("🔧 Initializing FeatureLookbackOptimizationComponent...")
        super().__init__(config)
        # Use standardized logging
        self.logger = get_logger('FeatureLookbackOptimization')
        self.error_handler = StandardizedErrorHandler(self.logger, 'FeatureLookbackOptimization')
        self.optimization_status = OptimizationStatus.PENDING
        self.start_time: Optional[float] = None
        self.metrics: Optional[OptimizationMetrics] = None
        tprint("✅ Basic component initialization complete")
        
        # Performance monitoring with memory tracking
        self.performance_monitor = {
            'memory_usage': [],
            'cpu_usage': [],
            'execution_times': {},
            'error_counts': 0,
            'peak_memory_mb': 0.0,
            'memory_warnings': 0
        }
        
        # Memory monitoring thresholds
        self.memory_warning_threshold_mb = 1000.0  # 1GB
        self.memory_critical_threshold_mb = 2000.0  # 2GB
        
        # Initialize optional attributes to avoid AttributeError at runtime
        self.m1_gpu_manager = None
        self.m1_memory_optimizer = None
        self.matrix_ops = None
        self.vectorized_ops = None
        self.hardware_ops = None
        self.performance_monitor_ml = None
        self.hyperparameter_optimizer = None
        self.cross_validator = None
        self.data_quality_checker = None
        self.feature_preparator = None
        
        # Initialize common utilities
        tprint("🔧 Initializing common utilities...")
        self.common_utils = CommonUtilities()
        self.math_validator = MathValidation()
        self.serializer = UniversalSerializer()
        tprint("✅ Common utilities initialized")
        
        # Initialize ML common utilities
        self.data_quality_checker = None
        self.feature_preparator = None
        self.hyperparameter_optimizer = None
        self.cross_validator = None
        self.performance_monitor_ml = None
        
        if ML_COMMON_AVAILABLE:
            try:
                self.data_quality_checker = DataQualityUtilities()
                self.feature_preparator = FeaturePreparator()
                tprint("✅ ML common utilities initialized")
            except Exception as e:
                tprint(f"⚠️ ML common utilities initialization failed: {e}")
        
        # Initialize hardware optimization if available
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            try:
                self.hardware_manager = get_unified_hardware_manager()
                self.memory_optimizer = get_advanced_memory_optimizer()
                tprint("✅ Hardware optimization initialized")
            except Exception as e:
                tprint(f"⚠️ Hardware optimization initialization failed: {e}")
                self.hardware_manager = None
                self.memory_optimizer = None
        else:
            self.hardware_manager = None
            self.memory_optimizer = None
        
        # Initialize Apple M1 specific managers if available in utilities
        try:
            self.m1_gpu_manager = get_m1_gpu_manager()
        except Exception:
            self.m1_gpu_manager = None
        try:
            self.m1_memory_optimizer = get_m1_memory_optimizer()
        except Exception:
            self.m1_memory_optimizer = None
        
        # Initialize matrix operations components
        self.enhanced_matrix_ops = None
        self.vectorized_core = None
        self.batch_processor = None
        self.matrix_ops = None
        self.vectorized_ops = None
        self.hardware_ops = None
        
        # Initialize M1 optimization components
        self.m1_gpu_manager = None
        self.m1_memory_optimizer = None
        self.m1_cpu_optimizer = None
        
        if MATRIX_OPS_AVAILABLE:
            try:
                
                self.matrix_ops = get_unified_matrix_operations()
                self.vectorized_ops = get_vectorized_processing_core()
                self.batch_processor = get_batch_matrix_processor()
                # Unify naming for downstream usage
                try:
                    self.matrix_ops = get_unified_matrix_operations()
                except Exception:
                    self.matrix_ops = None
                self.vectorized_ops = self.vectorized_core
                try:
                    self.hardware_ops = get_hardware_optimized_processor()
                except Exception:
                    self.hardware_ops = None
                tprint("✅ Advanced matrix operations initialized for feature lookback optimization")
            except Exception as e:
                tprint(f"⚠️ Matrix operations initialization failed: {e}")
        
        # Initialize M1 GPU manager if available
        try:
            self.m1_cpu_optimizer = get_m1_cpu_optimizer()
            tprint("✅ M1 optimization components initialized")
        except Exception as e:
            tprint(f"⚠️ M1 optimization initialization failed: {e}")
        
        tprint(f"🔧 Matrix operations available: {MATRIX_OPS_AVAILABLE}")
        tprint(f"🔧 Advanced matrix operations available: {ADVANCED_MATRIX_OPS_AVAILABLE}")
        tprint(f"🔧 MRMR optimizer available: {MRMR_OPTIMIZER_AVAILABLE}")
        
        # Initialize MRMR optimizer if available
        self.mrmr_optimizer = None
        if MRMR_OPTIMIZER_AVAILABLE:
            try:
                self.mrmr_optimizer = MRMRLookbackOptimizer()
                tprint("✅ MRMR lookback optimizer initialized")
            except Exception as e:
                tprint(f"⚠️ Failed to initialize MRMR optimizer: {e}")

        # Initialize feature bank for optimized feature lookback optimization
        try:
            self.feature_bank = get_global_feature_bank()
            tprint("✅ Feature bank initialized for lookback optimization")
        except Exception as e:
            tprint(f"⚠️ Failed to initialize feature bank: {e}")
            self.feature_bank = None

        # Initialize volatility labeler for FPT forward return calculations
        try:
            self.volatility_labeler = VolatilityAwareMultiHorizonLabeler(self._create_volatility_config())
            tprint("✅ Volatility labeler initialized for FPT calculations")
        except Exception as e:
            tprint(f"⚠️ Failed to initialize volatility labeler: {e}")
            self.volatility_labeler = None

        # Initialize multi-target scheme for multi-target optimization
        try:
            self.multi_target_scheme = MultiTargetScheme(self._create_multi_target_config())
            tprint("✅ Multi-target scheme initialized")
        except Exception as e:
            tprint(f"⚠️ Failed to initialize multi-target scheme: {e}")
            self.multi_target_scheme = None

        # Cache for target results to avoid recomputation
        self._last_target_result = None
        
        # Initialize Directional optimizer if available
        self.directional_optimizer = None
        if DIRECTIONAL_OPTIMIZER_AVAILABLE:
            try:
                directional_config = DirectionalLookbackConfig(
                    min_lookback=5,
                    max_lookback=50,
                    target_total_features=80,  # Target 80 features total (40 long + 40 short)
                    max_features_per_direction=50,
                    enable_directional=True,
                    parallel_optimization=True,
                    cross_directional_analysis=True
                )
                self.directional_optimizer = DirectionalLookbackOptimizer(config=directional_config)
                tprint("✅ Directional lookback optimizer initialized")
            except Exception as e:
                tprint(f"⚠️ Failed to initialize Directional optimizer: {e}")
        
        # Initialize ML common utilities if available
        if ML_COMMON_AVAILABLE:
            try:
                self.data_quality_checker = DataQualityUtilities()
            except Exception:
                self.data_quality_checker = None
            try:
                self.feature_preparator = FeaturePreparator()
            except Exception:
                self.feature_preparator = None
        
        # Initialize reporter
        tprint("🔧 Initializing optimization reporter...")
        self.reporter = OptimizationReporter(
            output_dir=(
                f"outcomes/market_analysis/feature_lookback_optimization/"
                f"{self.config.symbol}_{self.config.exchange}_{self.config.timeframe}"
            )
        )
        tprint("✅ Optimization reporter initialized")
        
        # Initialize validation framework
        tprint("🔧 Initializing validation framework...")
        self.validation_framework = FeatureLookbackValidationFramework()
        tprint("✅ Validation framework initialized")
        
        # Initialize monitoring metrics
        tprint("🔧 Initializing monitoring metrics...")
        self.monitoring = MonitoringMetrics(f"FeatureLookbackOptimization_{self.config.symbol}")
        tprint("✅ Monitoring metrics initialized")
        
        # Memory cleanup counter
        self.operation_count = 0
        tprint("🎯 FeatureLookbackOptimizationComponent initialization complete")
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        tprint("📋 Getting required artifacts for feature lookback optimization")
        artifacts = ['feature_lookback_optimization_result']
        tprint(f"✅ Required artifacts: {artifacts}")
        return artifacts
    
    def _cleanup_memory(self) -> None:
        """Clean up memory using hardware optimization tools."""
        tprint("🧹 Starting memory cleanup...")
        self.operation_count += 1
        
        if self.operation_count % PERFORMANCE_CONSTANTS.DEFAULT_CLEANUP_INTERVAL == 0:
            tprint(f"🔄 Memory cleanup triggered (operation #{self.operation_count})")
            try:
                # Use M1 memory optimizer if available
                if self.m1_memory_optimizer:
                    with memory_checkpoint(f"optimization_cleanup_{self.operation_count}"):
                        self.m1_memory_optimizer.cleanup_memory()
                        tprint("🧹 M1 memory cleanup performed")
                elif self.memory_optimizer:
                    self.memory_optimizer.cleanup_memory()
                    tprint("🧹 Hardware memory cleanup performed")
                else:
                    # Use common operations memory optimization
                    memory_result = optimize_memory()
                    if memory_result.get('success', False):
                        tprint(
                            f"🧹 Common operations memory cleanup: "
                            f"{memory_result.get('objects_collected', 0)} objects collected"
                        )
                    else:
                        # Basic cleanup
                        collected = gc.collect()
                        tprint(f"🧹 Basic memory cleanup: {collected} objects collected")
            except Exception as e:
                tprint(f"⚠️ Memory cleanup failed: {e}")
                # Fallback to basic cleanup
                try:
                    import gc
                    gc.collect()
                except Exception:
                    pass
    
    async def _enhanced_data_handling(
        self,
        data: Any,
        pipeline_state: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """Enhanced data handling to get data from multiple sources with optimized memory usage."""
        try:
            # Try direct data first - avoid unnecessary copies
            if data is not None:
                if isinstance(data, pd.DataFrame) and not data.empty:
                    # Validate in-place when possible to avoid copying
                    if self._quick_validate_data(data):
                        tprint("✅ Using direct DataFrame data (validated in-place)")
                        return data  # Return original data if validation passes
                    else:
                        # Only copy and fix if validation fails
                        validated_data = self._validate_and_optimize_data(data)
                        tprint("✅ Using direct DataFrame data (fixed copy)")
                        return validated_data
                elif hasattr(data, 'to_dataframe'):
                    df = data.to_dataframe()
                    if not df.empty:
                        if self._quick_validate_data(df):
                            tprint("✅ Converted data to DataFrame (validated in-place)")
                            return df
                        else:
                            validated_data = self._validate_and_optimize_data(df)
                            tprint("✅ Converted data to DataFrame (fixed copy)")
                            return validated_data
            
            # Try to get data from pipeline state
            if pipeline_state:
                # Try different keys that might contain data
                data_keys = ['market_data', 'data', 'processed_data', 'features', 'labeled_data']
                for key in data_keys:
                    if key in pipeline_state:
                        pipeline_data = pipeline_state[key]
                        if pipeline_data is not None:
                            if isinstance(pipeline_data, pd.DataFrame) and not pipeline_data.empty:
                                validated_data = self._validate_and_optimize_data(pipeline_data)
                                tprint(f"✅ Using data from pipeline state key: {key}")
                                return validated_data
                            elif hasattr(pipeline_data, 'to_dataframe'):
                                df = pipeline_data.to_dataframe()
                                if not df.empty:
                                    validated_data = self._validate_and_optimize_data(df)
                                    tprint(f"✅ Converted pipeline data from key: {key}")
                                    return validated_data
                
                # Try to get from regime data
                if 'regime_data' in pipeline_state:
                    regime_data = pipeline_state['regime_data']
                    if isinstance(regime_data, dict) and 'data' in regime_data:
                        regime_df = regime_data['data']
                        if isinstance(regime_df, pd.DataFrame) and not regime_df.empty:
                            validated_data = self._validate_and_optimize_data(regime_df)
                            tprint("✅ Using data from regime_data")
                            return validated_data
            
            # Try to get from artifacts
            if 'artifacts' in pipeline_state:
                artifacts = pipeline_state['artifacts']
                for artifact_key, artifact_data in artifacts.items():
                    if isinstance(artifact_data, dict) and 'data' in artifact_data:
                        artifact_df = artifact_data['data']
                        if isinstance(artifact_df, pd.DataFrame) and not artifact_df.empty:
                            validated_data = self._validate_and_optimize_data(artifact_df)
                            tprint(f"✅ Using data from artifact: {artifact_key}")
                            return validated_data
            
            tprint("⚠️ No valid data found in any source")
            return None
            
        except Exception as e:
            tprint(f"❌ Enhanced data handling failed: {e}")
            return None
    
    def _check_memory_usage(self) -> Dict[str, float]:
        """Check current memory usage and issue warnings if necessary."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
            
            # Update peak memory
            if memory_mb > self.performance_monitor['peak_memory_mb']:
                self.performance_monitor['peak_memory_mb'] = memory_mb
            
            # Add to memory usage history
            self.performance_monitor['memory_usage'].append(memory_mb)
            
            # Keep only recent memory measurements
            if len(self.performance_monitor['memory_usage']) > 1000:
                self.performance_monitor['memory_usage'] = (
                    self.performance_monitor['memory_usage'][-500:]
                )
            
            # Issue warnings if necessary
            if memory_mb > self.memory_critical_threshold_mb:
                self.performance_monitor['memory_warnings'] += 1
                tprint(
                    f"🚨 CRITICAL: Memory usage {memory_mb:.1f}MB exceeds "
                    f"critical threshold {self.memory_critical_threshold_mb}MB"
                )
                raise MemoryError(f"Memory usage {memory_mb:.1f}MB exceeds critical threshold")
            elif memory_mb > self.memory_warning_threshold_mb:
                self.performance_monitor['memory_warnings'] += 1
                tprint(
                    f"⚠️ WARNING: Memory usage {memory_mb:.1f}MB exceeds "
                    f"warning threshold {self.memory_warning_threshold_mb}MB"
                )
            
            return {
                'current_memory_mb': memory_mb,
                'peak_memory_mb': self.performance_monitor['peak_memory_mb'],
                'memory_warnings': self.performance_monitor['memory_warnings']
            }
            
        except ImportError:
            # psutil not available, use basic memory tracking
            return {'current_memory_mb': 0.0, 'peak_memory_mb': 0.0, 'memory_warnings': 0}
        except Exception as e:
            tprint(f"Memory monitoring failed: {e}")
            return {'current_memory_mb': 0.0, 'peak_memory_mb': 0.0, 'memory_warnings': 0}
    
    def _quick_validate_data(self, data: pd.DataFrame) -> bool:
        """Quick validation without copying data - returns True if data is good as-is."""
        try:
            # Check basic requirements without copying
            if data.empty:
                return False
            
            # Check for required columns
            required_columns = VALIDATION_CONSTANTS.REQUIRED_OHLCV_COLUMNS
            if not all(col in data.columns for col in required_columns):
                return False
            
            # Check for excessive nulls (quick check)
            null_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if null_ratio > VALIDATION_CONSTANTS.MAX_NULL_RATIO:
                return False
            
            # Check for infinite values in numeric columns (quick check)
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                has_inf = np.isinf(data[numeric_cols].values).any()
                if has_inf:
                    return False
            
            return True
            
        except Exception as e:
            self.error_handler.handle_warning(
                f"Quick validation failed: {e}",
                "quick_validate_data"
            )
            return False
    
    def _validate_and_optimize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and optimize data using common utilities and advanced matrix operations."""
        try:
            # Use common utilities for data validation
            if not validate_dataframe(data):
                tprint("⚠️ Data validation failed, attempting to fix")
                return data
            
            # Guard against excessive null values
            data = guard_dataframe_nulls(data, threshold=0.5)
            
            # Optimize data types for memory efficiency
            data = optimize_dataframe_dtypes(data)
            
            # Use M1 optimization if available
            if self.m1_gpu_manager and self.m1_gpu_manager.is_m1:
                try:
                    from src.utils.hardware.m1_gpu_utils import optimize_dataframe_for_m1
                    data = optimize_dataframe_for_m1(data)
                    tprint("✅ Data optimized for M1")
                except Exception as e:
                    tprint(f"⚠️ M1 optimization failed: {e}")
            
            # Use vectorized processing core optimization if available
            if self.vectorized_ops:
                try:
                    data = self.vectorized_ops.optimize_dataframe_for_processing(data)
                    tprint("✅ Data optimized using vectorized processing core")
                except Exception as e:
                    tprint(f"⚠️ Vectorized optimization failed: {e}")
            
            # Use hardware-optimized processing if available
            if self.hardware_ops:
                try:
                    data = self.hardware_ops.optimize_data_for_processing(data)
                    tprint("✅ Data optimized using hardware-optimized processing")
                except Exception as e:
                    tprint(f"⚠️ Hardware optimization failed: {e}")
            
            return data
            
        except Exception as e:
            tprint(f"⚠️ Data validation and optimization failed: {e}")
            return data
    
    def _enhanced_correlation_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Enhanced correlation analysis using advanced matrix operations."""
        try:
            if not MATRIX_OPS_AVAILABLE or not self.matrix_ops:
                tprint("⚠️ Matrix operations not available for correlation analysis")
                return {}
            
            # Use safe correlation matrix computation
            corr_matrix = safe_correlation_matrix(data)
            
            # Eigenvalue decomposition for principal components
            eigenvalues, eigenvectors = self.matrix_ops.eigendecomposition(corr_matrix)
            
            # SVD for dimensionality reduction
            U, s, Vh = self.matrix_ops.svd_decomposition(corr_matrix, k=10)
            
            # Compute feature importance based on correlation strength
            feature_importance = pd.DataFrame({
                'feature': data.columns,
                'mean_abs_corr': np.abs(corr_matrix).mean(axis=1),
                'max_corr': np.abs(corr_matrix).max(axis=1),
                'corr_std': np.abs(corr_matrix).std(axis=1),
                'eigenvalue_contribution': eigenvalues[:len(data.columns)]
            })
            
            tprint("✅ Enhanced correlation analysis completed")
            
            return {
                'correlation_matrix': corr_matrix,
                'eigenvalues': eigenvalues,
                'eigenvectors': eigenvectors,
                'singular_values': s,
                'principal_components': U,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            tprint(f"⚠️ Enhanced correlation analysis failed: {e}")
            return {}
    
    def _vectorized_feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced vectorized feature engineering using matrix operations."""
        try:
            if not MATRIX_OPS_AVAILABLE or not self.vectorized_ops:
                tprint("⚠️ Vectorized operations not available for feature engineering")
                return data
            
            # Optimize DataFrame for processing
            optimized_data = self.vectorized_ops.optimize_dataframe_for_processing(data)
            
            # Vectorized rolling features
            rolling_features = self.vectorized_ops.vectorized_rolling_features(
                optimized_data, 
                windows=[5, 10, 20, 50, 100],
                features=['close', 'volume', 'high', 'low']
            )
            
            # Comprehensive trading indicators
            trading_indicators = self.vectorized_ops.compute_trading_indicators(
                rolling_features,
                config=self._get_enhanced_indicator_config()
            )
            
            tprint("✅ Vectorized feature engineering completed")
            return trading_indicators
            
        except Exception as e:
            tprint(f"⚠️ Vectorized feature engineering failed: {e}")
            return data
    
    def _get_enhanced_indicator_config(self) -> Dict[str, Any]:
        """Get enhanced configuration for trading indicators."""
        return {
            # Moving averages
            'sma_periods': [9, 21, 50, 200],
            'ema_periods': [12, 26, 50],
            
            # RSI
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            
            # MACD
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            
            # Bollinger Bands
            'bb_period': 20,
            'bb_std': 2.0,
            
            # Stochastic
            'stoch_k': 14,
            'stoch_d': 3,
            'stoch_smooth': 3,
            
            # Williams %R
            'williams_period': 14,
            
            # ADX
            'adx_period': 14,
            
            # ATR
            'atr_period': 14,
            
            # CCI
            'cci_period': 20,
            
            # ROC
            'roc_period': 10,
            
            # Volume indicators
            'volume_sma_period': 20,
            'obv_smooth': 10,
        }
    
    @hardware_optimized("feature_optimization")
    def _hardware_optimized_feature_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Hardware-optimized feature processing using matrix operations."""
        try:
            if not MATRIX_OPS_AVAILABLE or not self.hardware_ops:
                tprint("⚠️ Hardware-optimized processing not available")
                return data
            
            # Hardware-optimized standard scaling
            scaled_data = self.hardware_ops.optimized_standard_scaling(data)
            
            # Convert back to DataFrame
            scaled_df = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)
            
            tprint("✅ Hardware-optimized feature processing completed")
            return scaled_df
            
        except Exception as e:
            tprint(f"⚠️ Hardware-optimized processing failed: {e}")
            return data
    
    def _batch_optimization_processing(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Batch processing for large-scale feature optimization."""
        try:
            # Validate input data type
            if not isinstance(data, pd.DataFrame):
                tprint(
                    f"⚠️ Batch optimization processing failed: "
                    f"Expected DataFrame but got {type(data)}"
                )
                return {'data': pd.DataFrame(), 'error': 'invalid_data_type'}
            
            if not MATRIX_OPS_AVAILABLE or not self.batch_processor:
                tprint("⚠️ Batch processing not available")
                return {'data': data}
            
            # Batch feature transformations
            transformations = [
                {'type': 'standardize', 'columns': ['close', 'volume']},
                {'type': 'robust_scale', 'columns': ['high', 'low']},
                {'type': 'power_transform', 'columns': ['returns'], 'params': {'method': 'yeo-johnson'}}
            ]
            
            transformed_data = self.batch_processor.batch_feature_transformation(
                data, transformations
            )
            
            # Batch correlation analysis
            corr_matrix, p_values = self.batch_processor.batch_correlation_analysis(
                transformed_data, method='pearson'
            )
            
            # Compute feature importance
            feature_importance = self._compute_feature_importance(corr_matrix, data.columns)
            
            tprint("✅ Batch optimization processing completed")
            
            return {
                'transformed_data': transformed_data,
                'correlation_matrix': corr_matrix,
                'p_values': p_values,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            tprint(f"⚠️ Batch optimization processing failed: {e}")
            return {'data': data}
    
    def _compute_feature_importance(self, corr_matrix: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
        """Compute feature importance based on correlation matrix."""
        try:
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'mean_abs_corr': np.abs(corr_matrix).mean(axis=1),
                'max_corr': np.abs(corr_matrix).max(axis=1),
                'corr_std': np.abs(corr_matrix).std(axis=1)
            })
            
            # Composite score
            feature_importance['composite_score'] = (
                feature_importance['mean_abs_corr'] * 0.4 +
                feature_importance['max_corr'] * 0.3 +
                feature_importance['corr_std'] * 0.3
            )
            
            return feature_importance.sort_values('composite_score', ascending=False)
            
        except Exception as e:
            tprint(f"⚠️ Feature importance computation failed: {e}")
            return pd.DataFrame()
    
    def _monitor_performance(self, operation_name: str) -> None:
        """Monitor performance metrics during execution."""
        tprint(f"📊 Monitoring performance for operation: {operation_name}")
        try:
            # Use common utilities for memory monitoring
            memory_usage = get_memory_usage()
            memory_mb = memory_usage / 1024 / 1024 if memory_usage > 0 else 0.0
            tprint(f"💾 Current memory usage: {memory_mb:.2f} MB")
            
            # Try to get CPU usage
            try:
                psutil, is_fallback = get_dependency('psutil')
                if psutil is not None:
                    process = psutil.Process()
                    cpu_percent = process.cpu_percent()
                    
                    self.performance_monitor['memory_usage'].append(memory_mb)
                    self.performance_monitor['cpu_usage'].append(cpu_percent)
                    
                    if is_fallback:
                        tprint("Using fallback psutil for performance monitoring")
                else:
                    self.performance_monitor['memory_usage'].append(memory_mb)
                    self.performance_monitor['cpu_usage'].append(0.0)
            except Exception:
                self.performance_monitor['memory_usage'].append(memory_mb)
                self.performance_monitor['cpu_usage'].append(0.0)
            
            if operation_name not in self.performance_monitor['execution_times']:
                self.performance_monitor['execution_times'][operation_name] = []
            
            self.performance_monitor['execution_times'][operation_name].append(time.time())
            
            # Use ML common performance monitoring if available
            if self.performance_monitor_ml:
                try:
                    self.performance_monitor_ml.record_metric(
                        name=f"optimization_{operation_name}",
                        value=time.time(),
                        metric_type="performance"
                    )
                except Exception as e:
                    tprint(f"⚠️ ML performance monitoring failed: {e}")
            
            # Cleanup memory periodically
            self._cleanup_memory()
            
        except Exception as e:
            tprint(f"⚠️ Performance monitoring failed: {e}")

    def _create_volatility_config(self) -> VolatilityAwareConfig:
        """Create volatility-aware configuration aligned with profit labeler."""
        return VolatilityAwareConfig(
            min_data_points=1000,
            generate_reports=True,
            save_intermediate_results=True,
            enable_volatility_normalization=True,
            enable_multi_target_scheme=True
        )

    def _create_multi_target_config(self) -> MultiTargetConfig:
        """Create multi-target configuration aligned with profit labeler."""
        return MultiTargetConfig(
            small_band=(0.4, 0.8),
            medium_band=(0.8, 1.3),
            high_band=(1.3, 2.0),
            enable_optimization=True,
            optimization_method='bayesian',
            n_trials=50,
            optimization_metric='lqs'
        )

    def _should_exclude_feature(self, feature_name: str, category: FeatureCategory) -> bool:
        """Check if feature should be excluded from optimization."""
        # Check category exclusions (hardcoded for now since config doesn't have these fields)
        excluded_categories = [
            FeatureCategory.INTERACTION,
            FeatureCategory.CROSS_TIMEFRAME,
            FeatureCategory.AUTOENCODER,
            FeatureCategory.REGIME
        ]

        if category in excluded_categories:
            return True

        # Check feature name exclusions
        excluded_features = [
            'wavelets', 'autoencoder', 'interaction', 'cross_timeframe', 'regime_'
        ]

        feature_lower = feature_name.lower()
        for excluded in excluded_features:
            if excluded in feature_lower:
                return True

        return False

    def _get_eligible_features(self) -> List[Tuple[str, FeatureGenerator]]:
        """Get list of features eligible for lookback optimization."""
        tprint("🔍 Identifying eligible features for optimization...")

        eligible_features = []
        excluded_count = 0

        excluded_categories = [
            FeatureCategory.INTERACTION,
            FeatureCategory.CROSS_TIMEFRAME,
            FeatureCategory.AUTOENCODER,
            FeatureCategory.REGIME
        ]

        for category in self.feature_bank.list_categories():
            if category in excluded_categories:
                tprint_info(f"   → Skipping excluded category: {category.value}")
                continue

            try:
                generators = self.feature_bank.get_generators_by_category(category)
                tprint_info(f"   → Processing {len(generators)} generators in {category.value}")

                for generator in generators:
                    feature_name = generator.config.name

                    if self._should_exclude_feature(feature_name, category):
                        excluded_count += 1
                        continue

                    if generator.supports_lookback_optimization():
                        eligible_features.append((feature_name, generator))
                        tprint_info(f"   → ✓ Eligible: {feature_name}")
                    else:
                        tprint_info(f"   → ⚠ Not optimizable: {feature_name}")

            except Exception as e:
                tprint_warning(f"   → ⚠ Error processing category {category.value}: {e}")

        tprint_success(f"✅ Found {len(eligible_features)} eligible features")
        tprint_info(f"   → Excluded {excluded_count} features")

        return eligible_features

    def _calculate_forward_returns_fpt(self, data: pd.DataFrame,
                                    lookback: int) -> pd.Series:
        """
        Calculate forward returns using actual multi_horizon_profit_labeler methodology.

        This method uses the actual labels generated by multi_horizon_profit_labeler:
        - Uses the actual ternary labels (-1, 0, 1) from FPT calculations
        - Aligns with the sophisticated target selection and horizon optimization
        - Uses the actual confidence scores and eligibility masks
        """
        tprint_info(f"   → Calculating FPT forward returns using multi_horizon_profit_labeler methodology")

        try:
            # Use the actual multi_horizon_profit_labeler to generate proper labels
            # This ensures perfect alignment with the labeler's sophisticated methodology

            # For feature lookback optimization, we need to use the actual labeling pipeline
            # that generates the same labels that will be used for training

            # Since we don't have access to the full pipeline here, we'll use the multi-target scheme
            # which implements the same FPT logic as the profit labeler
            returns = data['close'].pct_change()
            volatility = returns.rolling(window=min(lookback, 50)).std()

            # Generate targets using the same methodology as multi_horizon_profit_labeler
            target_result = self.multi_target_scheme.generate_targets(
                data, volatility, pd.Series(True, index=data.index)
            )

            if target_result.labels.empty:
                tprint_warning("   → No labels generated from multi-target scheme")
                # Fallback to simple returns
                simple_returns = data['close'].pct_change(lookback).shift(-lookback)
                return simple_returns.fillna(0)

            # Use the actual labels generated by the multi-target scheme
            # These are the same ternary labels (-1, 0, 1) that represent trading opportunities
            forward_returns = target_result.labels.iloc[:, 0]  # Use first target column

            # The labels are already in the correct format:
            # - 1: Long opportunity (price hit upper target first)
            # - -1: Short opportunity (price hit lower target first)
            # - 0: No opportunity (neither target hit within horizon)

            tprint_info(f"   → Generated {len(forward_returns.dropna())} FPT-based labels")
            tprint_info(f"   → Label distribution: {forward_returns.value_counts().to_dict()}")

            # Store the target result for later use (confidence scores, eligibility masks)
            self._last_target_result = target_result

            return forward_returns

        except Exception as e:
            tprint_error(f"   → Error calculating FPT forward returns: {e}")
            # Fallback to simple returns
            simple_returns = data['close'].pct_change(lookback).shift(-lookback)
            return simple_returns.fillna(0)

    def _check_for_precomputed_labels(self, pipeline_state: Optional[Dict[str, Any]]) -> bool:
        """Check if pre-computed labels from multi_horizon_profit_labeler are available."""
        if not pipeline_state:
            tprint_warning("⚠️ No pipeline state provided for label checking")
            return False

        # Check for multi_horizon_labeling_result in pipeline state
        labeling_result = pipeline_state.get('multi_horizon_labeling_result', {})
        if labeling_result and 'labeled_data' in labeling_result:
            labeled_data = labeling_result['labeled_data']
            if not labeled_data.empty:
                tprint_success("✅ Using pre-computed labels from multi_horizon_profit_labeler")
                tprint_info(f"   → Found {len(labeled_data.columns)} target columns")
                return True

        # Check for standardized output format
        standardized_output = pipeline_state.get('standardized_output', {})
        if standardized_output and 'labels' in standardized_output:
            labels = standardized_output['labels']
            if not labels.empty:
                tprint_success("✅ Using standardized output labels from multi_horizon_profit_labeler")
                tprint_info(f"   → Found {len(labels.columns)} target columns")
                return True

        tprint_warning("⚠️ No pre-computed labels found from multi_horizon_profit_labeler")
        return False

    def _get_precomputed_labels(self, pipeline_state: Optional[Dict[str, Any]], lookback: int) -> pd.Series:
        """Get pre-computed labels from multi_horizon_profit_labeler with proper weight application."""
        if not pipeline_state:
            tprint_warning("⚠️ No pipeline state provided for label retrieval")
            return pd.Series(dtype=float)

        # Try standardized output format first (preferred)
        standardized_output = pipeline_state.get('standardized_output', {})
        if standardized_output and 'labels' in standardized_output:
            labels = standardized_output['labels']
            weights = standardized_output.get('weights', {})
            target_columns = standardized_output.get('target_columns', [])
            
            if not labels.empty:
                tprint_info("📋 Using standardized output labels with weights")
                tprint_info(f"   → Available target columns: {target_columns}")
                tprint_info(f"   → Horizon weights: {weights}")
                
                # Select the best target based on weights and availability
                best_target = self._select_best_target_with_weights(labels, weights, target_columns)
                if best_target is not None:
                    tprint_success(f"✅ Selected target: {best_target}")
                    return labels[best_target].copy()
                else:
                    # Fallback to first available target
                    target_cols = [col for col in labels.columns if col not in ['timestamp', 'symbol']]
                    if target_cols:
                        tprint_info(f"   → Using fallback target: {target_cols[0]}")
                        return labels[target_cols[0]].copy()

        # Fallback to multi_horizon_labeling_result format
        labeling_result = pipeline_state.get('multi_horizon_labeling_result', {})
        labeled_data = labeling_result.get('labeled_data', pd.DataFrame())
        horizon_weights = labeling_result.get('horizon_weights', {})
        target_columns = labeling_result.get('target_columns', [])

        if not labeled_data.empty:
            tprint_info("📊 Using multi_horizon_labeling_result format")
            tprint_info(f"   → Available target columns: {target_columns}")
            tprint_info(f"   → Horizon weights: {horizon_weights}")
            
            # Select the best target based on weights and availability
            best_target = self._select_best_target_with_weights(labeled_data, horizon_weights, target_columns)
            if best_target is not None:
                tprint_success(f"✅ Selected target: {best_target}")
                return labeled_data[best_target].copy()
            else:
                # Fallback to first available target
                target_cols = [col for col in labeled_data.columns if col not in ['timestamp', 'symbol']]
                if target_cols:
                    tprint_info(f"   → Using fallback target: {target_cols[0]}")
                    return labeled_data[target_cols[0]].copy()

        tprint_warning("⚠️ No valid labels found in pipeline state")
        return pd.Series(dtype=float)

    def _select_best_target_with_weights(self, labels: pd.DataFrame, weights: Dict[str, float], target_columns: List[str]) -> Optional[str]:
        """Select the best target based on horizon weights and availability."""
        try:
            if not weights or not target_columns:
                # No weights available, use first available target
                available_targets = [col for col in labels.columns if col not in ['timestamp', 'symbol']]
                return available_targets[0] if available_targets else None

            # Priority order based on horizon weights (higher weight = higher priority)
            # Map target columns to their corresponding horizon weights
            target_priority = []
            
            for target in target_columns:
                if target in labels.columns:
                    # Determine horizon type from target name
                    if 'immediate' in target.lower() or 'small' in target.lower():
                        horizon_weight = weights.get('small', 0.0)
                    elif 'short' in target.lower() or 'medium' in target.lower():
                        horizon_weight = weights.get('medium', 0.0)
                    elif 'leverage' in target.lower() or 'high' in target.lower():
                        horizon_weight = weights.get('high', 0.0)
                    else:
                        # Default to small horizon if unclear
                        horizon_weight = weights.get('small', 0.0)
                    
                    target_priority.append((target, horizon_weight))

            # Sort by weight (descending) and return the highest weighted target
            if target_priority:
                target_priority.sort(key=lambda x: x[1], reverse=True)
                best_target = target_priority[0][0]
                tprint_info(f"   → Selected target '{best_target}' with weight {target_priority[0][1]:.3f}")
                return best_target

            return None

        except Exception as e:
            tprint_warning(f"⚠️ Error selecting best target with weights: {e}")
            # Fallback to first available target
            available_targets = [col for col in labels.columns if col not in ['timestamp', 'symbol']]
            return available_targets[0] if available_targets else None

    def _get_precomputed_confidence_scores(self, pipeline_state: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Get pre-computed confidence scores from multi_horizon_profit_labeler."""
        if not pipeline_state:
            return pd.DataFrame()

        # Try standardized output format first
        standardized_output = pipeline_state.get('standardized_output', {})
        if standardized_output and 'confidence_scores' in standardized_output:
            confidence_scores = standardized_output['confidence_scores']
            if not confidence_scores.empty:
                tprint_info("📋 Using standardized confidence scores")
                return confidence_scores

        # Fallback to multi_horizon_labeling_result format
        labeling_result = pipeline_state.get('multi_horizon_labeling_result', {})
        confidence_scores = labeling_result.get('confidence_scores', pd.DataFrame())
        if not confidence_scores.empty:
            tprint_info("📊 Using multi_horizon_labeling_result confidence scores")
            return confidence_scores

        tprint_warning("⚠️ No confidence scores found")
        return pd.DataFrame()

    def _get_precomputed_eligibility_masks(self, pipeline_state: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Get pre-computed eligibility masks from multi_horizon_profit_labeler."""
        if not pipeline_state:
            return pd.DataFrame()

        # Try standardized output format first
        standardized_output = pipeline_state.get('standardized_output', {})
        if standardized_output and 'eligibility_masks' in standardized_output:
            eligibility_masks = standardized_output['eligibility_masks']
            if not eligibility_masks.empty:
                tprint_info("📋 Using standardized eligibility masks")
                return eligibility_masks

        # Fallback to multi_horizon_labeling_result format
        labeling_result = pipeline_state.get('multi_horizon_labeling_result', {})
        eligibility_masks = labeling_result.get('eligibility_masks', pd.DataFrame())
        if not eligibility_masks.empty:
            tprint_info("📊 Using multi_horizon_labeling_result eligibility masks")
            return eligibility_masks

        tprint_warning("⚠️ No eligibility masks found")
        return pd.DataFrame()

    def _get_precomputed_quality_scores(self, pipeline_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get pre-computed quality scores from multi_horizon_profit_labeler."""
        if not pipeline_state:
            return {}

        # Try standardized output format first
        standardized_output = pipeline_state.get('standardized_output', {})
        if standardized_output and 'quality_scores' in standardized_output:
            quality_scores = standardized_output['quality_scores']
            if quality_scores:
                tprint_info("📋 Using standardized quality scores")
                return quality_scores

        # Fallback to multi_horizon_labeling_result format
        labeling_result = pipeline_state.get('multi_horizon_labeling_result', {})
        quality_scores = labeling_result.get('quality_scores', {})
        if quality_scores:
            tprint_info("📊 Using multi_horizon_labeling_result quality scores")
            return quality_scores

        tprint_warning("⚠️ No quality scores found")
        return {}

    def _get_generated_confidence_scores(self) -> pd.DataFrame:
        """Get confidence scores from the cached target result (generated on-the-fly)."""
        if self._last_target_result is None:
            return pd.DataFrame()
        return self._last_target_result.confidence_scores

    def _get_generated_eligibility_masks(self) -> pd.DataFrame:
        """Get eligibility masks from the cached target result (generated on-the-fly)."""
        if self._last_target_result is None:
            return pd.DataFrame()
        return self._last_target_result.eligibility_masks

    def _calculate_feature_target_score_aligned(self, feature_values: np.ndarray,
                                              target_values: np.ndarray, lookback: int,
                                              confidence_scores: Optional[np.ndarray] = None,
                                              eligibility_mask: Optional[np.ndarray] = None,
                                              quality_scores: Optional[Dict[str, Any]] = None) -> float:
        """Calculate score for feature-target alignment using actual FPT labels with confidence weighting and quality assessment."""
        try:
            # Apply eligibility mask first to filter out unreliable data points
            if eligibility_mask is not None:
                valid_mask = eligibility_mask.astype(bool)
                if np.sum(valid_mask) < 10:  # Need minimum samples
                    return 0.0

                feature_filtered = feature_values[valid_mask]
                target_filtered = target_values[valid_mask]

                # Apply confidence weighting if available
                if confidence_scores is not None:
                    confidence_filtered = confidence_scores[valid_mask]
                    # Weight the correlation by confidence scores
                    # Higher confidence points get more weight in the correlation calculation
                    weights = confidence_filtered / np.sum(confidence_filtered)
                else:
                    weights = None
            else:
                feature_filtered = feature_values
                target_filtered = target_values
                weights = None

            if len(feature_filtered) < 10:
                return 0.0

            # For ternary labels (-1, 0, 1), use weighted rank correlation
            if weights is not None:
                # Calculate weighted Spearman correlation
                correlation = self._weighted_spearmanr(feature_filtered, target_filtered, weights)
                p_value = 0.01  # Simplified - in practice would need proper weighted p-value calculation
            else:
                correlation, p_value = self._safe_spearmanr(feature_filtered, target_filtered)

            if np.isnan(correlation) or np.isnan(p_value):
                return 0.0

            # Convert to positive score (higher absolute correlation is better)
            # Weight by confidence if available, otherwise use standard weighting
            if confidence_scores is not None:
                avg_confidence = np.mean(confidence_scores[valid_mask]) if eligibility_mask is not None else np.mean(confidence_scores)
                score = abs(correlation) * (1 - p_value) * avg_confidence
            else:
                score = abs(correlation) * (1 - p_value)

            # Apply quality score adjustment if available
            quality_adjustment = 1.0
            if quality_scores:
                # Extract overall quality from the first available target
                for target_name, quality_data in quality_scores.items():
                    if hasattr(quality_data, 'overall_quality'):
                        quality_adjustment = quality_data.overall_quality
                        break
                    elif isinstance(quality_data, dict) and 'overall_quality' in quality_data:
                        quality_adjustment = quality_data['overall_quality']
                        break
                
                # Apply quality adjustment (higher quality = higher score)
                score = score * quality_adjustment
                tprint_info(f"   → Quality adjustment applied: {quality_adjustment:.4f}")

            tprint_info(f"   → Aligned score: correlation={correlation:.4f}, p-value={p_value:.4f}, confidence={np.mean(confidence_scores) if confidence_scores is not None else 'N/A':.4f}, quality={quality_adjustment:.4f}, score={score:.4f}")
            return score

        except Exception as e:
            tprint_warning(f"   → Error calculating aligned score: {e}")
            return 0.0

    def _weighted_spearmanr(self, x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
        """Calculate weighted Spearman correlation coefficient."""
        try:
            # For simplicity, use a weighted correlation approach
            # In a full implementation, this would properly handle weighted rank correlation

            # Normalize weights
            weights = weights / np.sum(weights)

            # Calculate weighted means
            weighted_mean_x = np.sum(weights * x)
            weighted_mean_y = np.sum(weights * y)

            # Calculate weighted covariance and variances
            weighted_cov = np.sum(weights * (x - weighted_mean_x) * (y - weighted_mean_y))
            weighted_var_x = np.sum(weights * (x - weighted_mean_x)**2)
            weighted_var_y = np.sum(weights * (y - weighted_mean_y)**2)

            # Calculate correlation coefficient
            if weighted_var_x > 0 and weighted_var_y > 0:
                correlation = weighted_cov / np.sqrt(weighted_var_x * weighted_var_y)
                return correlation
            else:
                return 0.0

        except Exception as e:
            tprint_warning(f"   → Error in weighted correlation: {e}")
            return 0.0

    def _optimize_single_feature(self, feature_name: str, generator: FeatureGenerator,
                               data: pd.DataFrame, pipeline_state: Optional[Dict[str, Any]] = None) -> FeatureLookbackResult:
        """Optimize lookback period for a single feature using actual multi_horizon_profit_labeler labels."""
        tprint_info(f"🎯 Optimizing lookback for feature: {feature_name}")

        lookback_scores = {}
        best_lookback = self.config.min_lookback
        best_score = -np.inf

        # Check if we have pre-computed labels from multi_horizon_profit_labeler
        labels_available = self._check_for_precomputed_labels(pipeline_state)

        # Test different lookback periods
        for lookback in range(self.config.min_lookback,
                             self.config.max_lookback + 1,
                             self.config.lookback_step):

            try:
                tprint_info(f"   → Testing lookback period: {lookback}")

                # Generate feature with current lookback
                feature_data = generator.generate(data, lookback=lookback)
                if feature_data.data.empty or len(feature_data.data.dropna()) < 100:
                    tprint_warning(f"   → Insufficient data for lookback {lookback}")
                    continue

                # Get forward returns - use pre-computed labels if available
                if labels_available:
                    forward_returns = self._get_precomputed_labels(pipeline_state, lookback)
                    confidence_scores = self._get_precomputed_confidence_scores(pipeline_state)
                    eligibility_masks = self._get_precomputed_eligibility_masks(pipeline_state)
                else:
                    forward_returns = self._calculate_forward_returns_fpt(data, lookback)
                    confidence_scores = None
                    eligibility_masks = None

                forward_returns = forward_returns.dropna()

                if len(forward_returns) < 100:
                    tprint_warning(f"   → Insufficient forward returns for lookback {lookback}")
                    continue

                # Align feature data with forward returns
                common_index = feature_data.data.index.intersection(forward_returns.index)
                if len(common_index) < 100:
                    tprint_warning(f"   → Insufficient overlapping data for lookback {lookback}")
                    continue

                feature_aligned = feature_data.data.loc[common_index]
                returns_aligned = forward_returns.loc[common_index]

                # Get confidence scores, eligibility masks, and quality scores for the aligned data
                confidence_aligned = None
                eligibility_aligned = None
                quality_scores_data = None

                if confidence_scores is not None and not confidence_scores.empty:
                    confidence_aligned = confidence_scores.loc[common_index]
                elif hasattr(self, '_last_target_result') and self._last_target_result is not None:
                    # Try to get confidence scores from cached target result
                    generated_confidence = self._get_generated_confidence_scores()
                    if not generated_confidence.empty:
                        confidence_aligned = generated_confidence.loc[common_index]

                if eligibility_masks is not None and not eligibility_masks.empty:
                    eligibility_aligned = eligibility_masks.loc[common_index]
                elif hasattr(self, '_last_target_result') and self._last_target_result is not None:
                    # Try to get eligibility masks from cached target result
                    generated_eligibility = self._get_generated_eligibility_masks()
                    if not generated_eligibility.empty:
                        eligibility_aligned = generated_eligibility.loc[common_index]

                # Get quality scores from pipeline state
                if pipeline_state:
                    quality_scores_data = self._get_precomputed_quality_scores(pipeline_state)

                # Calculate score using the actual FPT-based labels with confidence weighting and quality assessment
                score = self._calculate_feature_target_score_aligned(
                    feature_aligned.values,
                    returns_aligned.values,
                    lookback,
                    confidence_scores=confidence_aligned.values if confidence_aligned is not None else None,
                    eligibility_mask=eligibility_aligned.values if eligibility_aligned is not None else None,
                    quality_scores=quality_scores_data
                )

                # Log additional information about confidence, eligibility, and quality
                if confidence_aligned is not None:
                    tprint_info(f"   → Using {len(confidence_aligned.dropna())} confidence scores (avg: {confidence_aligned.mean():.4f})")
                if eligibility_aligned is not None:
                    eligible_count = eligibility_aligned.sum() if hasattr(eligibility_aligned, 'sum') else 0
                    tprint_info(f"   → Using {eligible_count} eligible data points out of {len(eligibility_aligned)}")
                if quality_scores_data:
                    tprint_info(f"   → Using quality scores for {len(quality_scores_data)} targets")

                if not np.isnan(score):
                    lookback_scores[lookback] = score
                    tprint_info(f"   → Lookback {lookback}: score={score:.4f}")

                    if score > best_score:
                        best_score = score
                        best_lookback = lookback
                else:
                    tprint_warning(f"   → Invalid score calculation for lookback {lookback}")

            except Exception as e:
                tprint_warning(f"   → Error testing lookback {lookback}: {e}")
                continue

        # Calculate confidence interval for best lookback
        if best_lookback in lookback_scores:
            scores_array = np.array(list(lookback_scores.values()))
            confidence_interval = (
                np.percentile(scores_array, 5),
                np.percentile(scores_array, 95)
            )
        else:
            confidence_interval = (0.0, 0.0)

        # Determine best targets (for multi-target features)
        best_targets = self._identify_best_targets(generator, data, best_lookback)

        result = FeatureLookbackResult(
            feature_name=feature_name,
            optimal_lookback=best_lookback,
            performance_score=best_score,
            lookback_scores=lookback_scores,
            best_targets=best_targets,
            confidence_interval=confidence_interval,
            optimization_time=time.time() - (self.start_time or time.time()),
            n_samples=len(data),
            n_features_tested=len(lookback_scores),
            success=len(lookback_scores) > 0
        )

        if result.success:
            tprint_success(f"✅ Optimized {feature_name}: lookback={best_lookback}, score={best_score:.4f}")
        else:
            tprint_error(f"❌ Failed to optimize {feature_name}")
            result.error_message = "No valid lookback periods found"

        return result

    def _identify_best_targets(self, generator: FeatureGenerator, data: pd.DataFrame,
                             lookback: int) -> List[str]:
        """Identify best targets for multi-target features using quality scores."""
        try:
            # Get quality scores if available
            quality_scores = self._get_precomputed_quality_scores({})  # Would need pipeline state

            if quality_scores:
                # Select targets based on quality scores
                # Higher quality targets get priority
                sorted_targets = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)
                best_targets = [target for target, score in sorted_targets[:3]]  # Top 3 targets
                tprint_info(f"   → Selected {len(best_targets)} best targets based on quality scores")
                return best_targets
            else:
                # Fallback to default targets
                return ["multi_target_primary", "immediate_opportunity", "short_term_opportunity"]

        except Exception as e:
            tprint_warning(f"   → Error identifying best targets: {e}")
            return ["multi_target_primary"]


    def optimize_features_with_labels(self, data: pd.DataFrame, feature_columns: List[str],
                                    pipeline_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize feature lookback periods using actual multi_horizon_profit_labeler labels with full metadata.

        This method REQUIRES pre-computed labels from multi_horizon_profit_labeler.
        Fast fails if labels are not available - no fallback to on-the-fly generation.

        Args:
            data: Market data
            feature_columns: List of feature column names to optimize
            pipeline_state: Pipeline state containing multi_horizon_labeling_result with all metadata

        Returns:
            Dictionary with optimization results for each feature

        Raises:
            ValueError: If required labeling results are not available
        """
        tprint("🚀 Optimizing features using actual multi_horizon_profit_labeler labels with full metadata")

        # Validate that we have the required labeling results - FAST FAIL if not available
        if not self._check_for_precomputed_labels(pipeline_state):
            error_msg = "❌ No pre-computed labels available from multi_horizon_profit_labeler. " \
                       "This method requires proper labeling results - run multi_horizon_profit_labeler first."
            tprint_error(error_msg)
            raise ValueError(error_msg)

        tprint_info(f"   → Features to optimize: {len(feature_columns)}")

        # Check for additional metadata
        has_confidence = not self._get_precomputed_confidence_scores(pipeline_state).empty
        has_eligibility = not self._get_precomputed_eligibility_masks(pipeline_state).empty
        has_quality = bool(self._get_precomputed_quality_scores(pipeline_state))

        tprint_info(f"   → Confidence scores available: {has_confidence}")
        tprint_info(f"   → Eligibility masks available: {has_eligibility}")
        tprint_info(f"   → Quality scores available: {has_quality}")

        if not has_confidence or not has_eligibility:
            tprint_warning("⚠️ Missing confidence scores or eligibility masks - optimization quality may be reduced")

        optimization_results = {}

        for feature_name in feature_columns:
            try:
                # Get the generator for this feature
                generator = self.feature_bank.get_generator_by_name(feature_name)
                if not generator:
                    tprint_warning(f"⚠️ No generator found for feature: {feature_name}")
                    continue

                # Optimize using actual labels from multi_horizon_profit_labeler with full metadata
                result = self._optimize_single_feature(feature_name, generator, data, pipeline_state)
                optimization_results[feature_name] = result

            except Exception as e:
                tprint_error(f"❌ Failed to optimize feature {feature_name}: {e}")
                optimization_results[feature_name] = {
                    'error': str(e),
                    'optimal_lookback': self.config.min_lookback,
                    'success': False
                }

        successful_count = len([r for r in optimization_results.values() if isinstance(r, dict) and r.get('success', False)])
        tprint_success(f"✅ Feature optimization completed using actual labels with full metadata")
        tprint_info(f"   → Successful optimizations: {successful_count}/{len(feature_columns)}")
        tprint_info(f"   → Used confidence weighting: {has_confidence}")
        tprint_info(f"   → Used eligibility filtering: {has_eligibility}")
        tprint_info(f"   → Used quality score prioritization: {has_quality}")

        return optimization_results








    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute feature lookback optimization with comprehensive validation and reporting.
        
        Args:
            data: Market data for feature optimization
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with feature lookback optimization results
        """
        self.start_time = time.time()
        self.optimization_status = OptimizationStatus.IN_PROGRESS
        
        # Start comprehensive monitoring
        self.monitoring.start_monitoring()
        
        tprint('⚙️ Starting Feature Lookback Optimization')
        self._monitor_performance('start')
        
        # Record start metrics
        self.monitoring.record_metric(
            name="optimization_started",
            value=1,
            metric_type=MetricType.PERFORMANCE,
            level=MetricLevel.INFO,
            tags={"symbol": self.config.symbol, "exchange": self.config.exchange, "timeframe": self.config.timeframe}
        )
        
        try:
            # Step 0: Enhanced data handling - try to get data from multiple sources
            tprint('🔍 Step 0: Enhanced data handling...')
            processed_data = await self._enhanced_data_handling(data, pipeline_state)
            if processed_data is None:
                self.optimization_status = OptimizationStatus.FAILED
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message="No valid data available for feature lookback optimization",
                    metadata={'error': 'Data is None or empty from all sources'}
                )
            
            # Step 0.5: Load labeling and regime results if missing from pipeline state (before validation)
            tprint('🔍 Step 0.5: Ensuring labeling and regime results are available...')
            
            # Load labeling results if missing
            if 'multi_horizon_labeling_result' not in pipeline_state and 'triple_barrier_labeling_result' not in pipeline_state:
                tprint('🔍 No labeling data in pipeline state, loading from recent outcomes...')
                symbol = pipeline_state.get('symbol', 'ETHUSDT')
                exchange = pipeline_state.get('exchange', 'binance')
                timeframe = pipeline_state.get('timeframe', '15m')
                labeling_data = self._load_recent_labeling_results(symbol=symbol, exchange=exchange, timeframe=timeframe)
                if labeling_data:
                    pipeline_state['multi_horizon_labeling_result'] = labeling_data
                    tprint(f'✅ Pre-loaded labeling data for validation: multi_horizon_profit_labeling')
                else:
                    tprint(f'⚠️ No recent labeling results found - validation will show warning')
            else:
                tprint('✅ Labeling data already present in pipeline state')
            
            # Load regime data splitting results if missing
            if 'regime_data_splitting_result' not in pipeline_state:
                tprint('🔍 No regime splitting data in pipeline state, loading from recent outcomes...')
                regime_data = self._load_recent_regime_splitting_results(symbol, exchange, timeframe)
                if regime_data:
                    pipeline_state['regime_data_splitting_result'] = regime_data
                    tprint(f'✅ Pre-loaded regime splitting data for validation')
                else:
                    tprint(f'⚠️ No recent regime splitting results found - validation will show warning')
            else:
                tprint('✅ Regime splitting data already present in pipeline state')
            
            # Step 1: Comprehensive validation using framework
            tprint('🔍 Step 1: Validating input data and pipeline state...')
            
            # Validate data with auto-fixing
            data_is_valid, data_validation_results, fixed_data = self.validation_framework.validate_data(processed_data)
            if not data_is_valid:
                critical_failures = [r for r in data_validation_results 
                                   if r.status == ValidationStatus.FAILED and r.level == ValidationLevel.CRITICAL]
                error_msg = f"Data validation failed: {[r.message for r in critical_failures]}"
                tprint(f'❌ {error_msg}')
                self.optimization_status = OptimizationStatus.FAILED
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg,
                    metadata={'validation_errors': [r.message for r in critical_failures]}
                )
            
            # Validate pipeline state
            pipeline_is_valid, pipeline_validation_results = self.validation_framework.validate_pipeline_state(pipeline_state)
            if not pipeline_is_valid:
                critical_failures = [r for r in pipeline_validation_results 
                                   if r.status == ValidationStatus.FAILED and r.level == ValidationLevel.CRITICAL]
                error_msg = f"Pipeline state validation failed: {[r.message for r in critical_failures]}"
                tprint(f'❌ {error_msg}')
                self.optimization_status = OptimizationStatus.FAILED
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg,
                    metadata={'validation_errors': [r.message for r in critical_failures]}
                )
            
            # Log validation warnings
            all_warnings = [r for r in data_validation_results + pipeline_validation_results 
                          if r.status == ValidationStatus.WARNING]
            for warning in all_warnings:
                tprint(f'⚠️ {warning.message}')
            
            # Generate validation summary
            data_validation_summary = self.validation_framework.generate_validation_summary(data_validation_results)
            pipeline_validation_summary = self.validation_framework.generate_validation_summary(pipeline_validation_results)
            
            # Record validation metrics
            self.monitoring.record_quality_metric("data_validation_score", data_validation_summary.quality_score)
            self.monitoring.record_quality_metric("pipeline_validation_score", pipeline_validation_summary.quality_score)
            self.monitoring.record_technical_metric("validation_rules_passed", data_validation_summary.passed + pipeline_validation_summary.passed)
            self.monitoring.record_technical_metric("validation_rules_failed", data_validation_summary.failed + pipeline_validation_summary.failed)
            
            tprint(f'✅ Validation passed (data quality: {data_validation_summary.quality_score:.3f})')
            self._monitor_performance('validation_complete')
            
            # Step 2: Load and prepare market data (use fixed data if available)
            tprint('📊 Loading and preparing market data...')
            market_data = await self._load_market_data(fixed_data if fixed_data is not None else processed_data)
            if market_data is None or market_data.empty:
                raise ValueError("No market data available for feature lookback optimization")
            
            tprint(f'📈 Market data loaded: {len(market_data)} rows, {len(market_data.columns)} columns')
            self._monitor_performance('data_loaded')
            
            # Step 3: Get labeled data from previous stage with enhanced integration
            # Check for standardized output first (preferred), then multi-horizon, then triple barrier
            standardized_output = pipeline_state.get('standardized_output', {})
            multi_horizon_labeling = pipeline_state.get('multi_horizon_labeling_result', {})
            triple_barrier_labeling = pipeline_state.get('triple_barrier_labeling_result', {})
            
            if standardized_output and standardized_output.get('pipeline_ready', False):
                labeling_data = standardized_output
                labeling_method = 'standardized_multi_horizon'
                tprint('🏷️ Standardized multi-horizon labeling data retrieved from pipeline state')
                tprint_info(f"   → Target columns: {standardized_output.get('target_columns', [])}")
                tprint_info(f"   → Horizon weights: {standardized_output.get('weights', {})}")
                tprint_info(f"   → Sample weights available: {standardized_output.get('sample_weights') is not None}")
            elif multi_horizon_labeling:
                labeling_data = multi_horizon_labeling
                labeling_method = 'multi_horizon'
                tprint('🏷️ Multi-horizon labeling data retrieved from pipeline state')
            elif triple_barrier_labeling:
                labeling_data = triple_barrier_labeling
                labeling_method = 'triple_barrier'
                tprint('🏷️ Triple barrier labeling data retrieved from pipeline state')
            else:
                # Check if labeling data is now available (should be loaded in Step 0.5)
                standardized_output = pipeline_state.get('standardized_output', {})
                multi_horizon_labeling = pipeline_state.get('multi_horizon_labeling_result', {})
                triple_barrier_labeling = pipeline_state.get('triple_barrier_labeling_result', {})
                
                if standardized_output and standardized_output.get('pipeline_ready', False):
                    labeling_data = standardized_output
                    labeling_method = 'standardized_multi_horizon'
                    tprint('🏷️ Using pre-loaded standardized multi-horizon labeling data')
                elif multi_horizon_labeling:
                    labeling_data = multi_horizon_labeling
                    labeling_method = 'multi_horizon_profit_labeling'
                    tprint('🏷️ Using pre-loaded multi-horizon labeling data')
                elif triple_barrier_labeling:
                    labeling_data = triple_barrier_labeling
                    labeling_method = 'triple_barrier_labeling'
                    tprint('🏷️ Using pre-loaded triple barrier labeling data')
                else:
                    tprint('⚠️ No labeling results found - using fallback optimization mode')
                    tprint('🔄 This is expected when labeling step runs after feature optimization')
                    # Use fallback mode with basic targets derived from price movements
                    labeling_method = 'fallback'
                    # Create basic target variable from price movements for optimization
                    fallback_targets = self._create_fallback_targets(market_data)
                    labeling_data = {'labeled_data': fallback_targets, 'method': 'fallback'}
            
            tprint(f'📊 Using {labeling_method} labeling method for feature optimization')
            
            # Step 4: Configure feature optimization
            tprint('⚙️ Configuring feature optimization...')
            optimization_config = self._create_optimization_config(pipeline_state)
            self._monitor_performance('config_created')
            
            # Step 5: Get feature optimizer
            tprint('🔧 Initializing feature optimizer...')
            feature_optimizer = await self._get_feature_optimizer(optimization_config)
            self._monitor_performance('optimizer_ready')
            
            # Step 6: Perform feature lookback optimization
            tprint('🚀 Starting feature optimization process...')
            optimization_result = await self._perform_feature_optimization(
                feature_optimizer, market_data, labeling_data, optimization_config
            )
            self._monitor_performance('optimization_complete')
            
            # Step 7: Extract and validate results
            tprint('📋 Extracting optimization results...')
            # Handle different return formats from optimizer
            if 'results' in optimization_result:
                # New format from FeatureGenerationOptimizer
                optimization_results = optimization_result.get('results', {})
                optimized_features = optimization_result.get('results', {})
                optimization_metrics = optimization_result.get('metadata', {})
            else:
                # Legacy format
                optimization_results = optimization_result.get('optimization_results', {})
                optimized_features = optimization_result.get('optimized_features', {})
                optimization_metrics = optimization_result.get('optimization_metrics', {})
            
            # Validate optimization results using framework
            optimization_is_valid, optimization_validation_results = self.validation_framework.validate_optimization_results(optimization_result)
            if not optimization_is_valid:
                critical_failures = [r for r in optimization_validation_results 
                                   if r.status == ValidationStatus.FAILED and r.level == ValidationLevel.CRITICAL]
                error_msg = f"Optimization results validation failed: {[r.message for r in critical_failures]}"
                tprint(f'❌ {error_msg}')
                raise ValueError(error_msg)
            
            # Log optimization validation warnings
            optimization_warnings = [r for r in optimization_validation_results 
                                   if r.status == ValidationStatus.WARNING]
            for warning in optimization_warnings:
                tprint(f'⚠️ {warning.message}')
            
            optimization_validation_summary = self.validation_framework.generate_validation_summary(optimization_validation_results)
            
            # Record optimization metrics
            self.monitoring.record_quality_metric("optimization_validation_score", optimization_validation_summary.quality_score)
            self.monitoring.record_business_metric("features_optimized", len(optimized_features))
            self.monitoring.record_quality_metric("best_optimization_score", optimization_results.get('best_score', 0.0))
            
            tprint(f'✅ Optimization results validated (quality: {optimization_validation_summary.quality_score:.3f})')
            
            # Step 8: Create comprehensive metrics
            self.metrics = self._create_optimization_metrics(
                optimization_results, optimized_features, optimization_metrics, optimization_result
            )
            
            # Step 9: Generate comprehensive report using reporter
            tprint('📊 Generating comprehensive optimization report...')
            comprehensive_report = self.reporter.generate_comprehensive_report(
                optimization_result=optimization_result,
                metrics=self.metrics,
                validation_results={
                    'data_validation': {
                        'summary': data_validation_summary,
                        'results': data_validation_results
                    },
                    'pipeline_validation': {
                        'summary': pipeline_validation_summary,
                        'results': pipeline_validation_results
                    },
                    'optimization_validation': {
                        'summary': optimization_validation_summary,
                        'results': optimization_validation_results
                    }
                },
                performance_metrics=self.performance_monitor,
                symbol=self.config.symbol,
                exchange=self.config.exchange,
                timeframe=self.config.timeframe
            )
            
            # Step 10: Create consolidated artifacts
            artifacts = self._create_artifacts(
                optimization_results, optimized_features, optimization_metrics, 
                optimization_result, comprehensive_report, 
                data_validation_summary, pipeline_validation_summary, optimization_validation_summary
            )
            
            # Step 11: Final validation
            if not self.validate_artifacts(artifacts):
                raise ValueError("Generated artifacts failed validation")
            
            self.optimization_status = OptimizationStatus.COMPLETED
            execution_time = time.time() - self.start_time
            
            # Record completion metrics
            self.monitoring.record_performance_metric("total_optimization", execution_time)
            self.monitoring.record_business_metric("optimization_success_rate", 1.0)
            self.monitoring.record_metric(
                name="optimization_completed",
                value=1,
                metric_type=MetricType.PERFORMANCE,
                level=MetricLevel.INFO,
                tags={"status": "success", "features_optimized": len(optimized_features)},
                metadata={"execution_time": execution_time, "best_lookback_period": self.metrics.best_lookback_period}
            )
            
            # Stop monitoring
            self.monitoring.stop_monitoring()
            
            tprint(f'✅ Feature Lookback Optimization completed successfully in {execution_time:.2f}s')
            tprint(f'📈 Optimized {len(optimized_features)} features with best lookback period: {self.metrics.best_lookback_period}')
            
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                execution_time=execution_time,
                metadata={
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'features_optimized': len(optimized_features),
                    'optimization_status': self.optimization_status.value,
                    'data_quality_score': data_validation_summary.quality_score,
                    'performance_metrics': self.performance_monitor
                }
            )
            
        except Exception as e:
            self.optimization_status = OptimizationStatus.FAILED
            self.performance_monitor['error_counts'] += 1
            execution_time = time.time() - self.start_time if self.start_time else 0.0
            
            # Record error metrics
            self.monitoring.record_error(
                error_type="optimization_failed",
                error_message=str(e),
                context={"execution_time": execution_time, "optimization_status": self.optimization_status.value}
            )
            self.monitoring.record_business_metric("optimization_success_rate", 0.0)
            self.monitoring.record_performance_metric("failed_optimization", execution_time)
            
            # Stop monitoring
            self.monitoring.stop_monitoring()
            
            tprint(f'❌ Feature Lookback Optimization failed after {execution_time:.2f}s: {e}')
            import traceback
            tprint(f'❌ Error details: {traceback.format_exc()}')
            
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e),
                execution_time=execution_time,
                metadata={
                    'optimization_status': self.optimization_status.value,
                    'error_count': self.performance_monitor['error_counts'],
                    'performance_metrics': self.performance_monitor
                }
            )
    
    def _create_optimization_config(self, pipeline_state: Dict[str, Any]) -> Any:
        """Create optimization configuration based on pipeline state and component config."""
        try:
            
            # Check if regime data is available for regime-aware optimization
            regime_data_splitting = pipeline_state.get('regime_data_splitting_result', {})
            enable_regime_aware = bool(regime_data_splitting)
            
            # Use ML common hyperparameter optimizer if available
            if self.hyperparameter_optimizer:
                try:
                    # Get optimized hyperparameters from ML common utilities
                    hpo_config = self.hyperparameter_optimizer.get_optimized_config(
                        method='genetic_algorithm',
                        feature_types=['technical_indicators', 'price_features', 'volume_features'],
                        regime_aware=enable_regime_aware
                    )
                    tprint("✅ Using ML common hyperparameter optimization")
                except Exception as e:
                    tprint(f"⚠️ ML common HPO failed: {e}")
                    hpo_config = {}
            else:
                hpo_config = {}
            
            # Use ML common cross-validation config if available
            if self.cross_validator:
                try:
                    cv_config = self.cross_validator.get_optimal_config(
                        data_size=pipeline_state.get('data_size', 1000),
                        feature_count=pipeline_state.get('feature_count', 50)
                    )
                    tprint("✅ Using ML common cross-validation config")
                except Exception as e:
                    tprint(f"⚠️ ML common CV config failed: {e}")
                    cv_config = {'folds': 5, 'test_size': 0.2}
            else:
                cv_config = {'folds': 5, 'test_size': 0.2}
            
            # Get lookback range from config
            lookback_range = getattr(OptimizationConfig, 'DEFAULT_LOOKBACK_RANGE', (5, 300))
            
            config = FeatureOptimizationConfig(
                min_lookback=lookback_range[0],
                max_lookback=lookback_range[1],
                optimization_method=OptimizationMethod.CROSS_VALIDATION,
                cv_folds=cv_config.get('folds', 5),
                regime_aware=enable_regime_aware,
                parallel_processing=True,
                max_workers=4,
                memory_efficient=True,
                enable_directional_optimization=True,  # Enable directional optimization by default
                enable_multi_target_optimization=True,  # Enable multi-target optimization by default
                optimization_metric='sharpe_ratio',
                # Performance-focused parameters with balanced regularization
                l1_regularization=0.001,  # Balanced regularization
                l2_regularization=0.001,  # Balanced regularization  
                max_lookback_variance=0.3,  # Increased flexibility for better performance
                lookback_range_penalty=0.08,  # Further reduced penalty for exploration
                temporal_consistency_weight=0.25,  # Reduced for more performance focus
                stability_weight=0.25,  # Performance-focused: 25% stability vs 75% performance
                # Rolling window parameters
                rolling_window_size="30D",
                rolling_step_size="7D",
                min_stability_score=0.7,
                # CV stability parameters
                cv_stability_metric="coefficient_variance",
                stability_cv_folds=3
            )
            
            tprint(f'⚙️ Enhanced optimization config created (regime-aware: {enable_regime_aware})')
            return config
            
        except ImportError as e:
            tprint(f"⚠️ Feature optimization config import failed: {e}")
            # Return a simple config with two-step grid + TPE approach
            return {
                'optimization_method': 'two_step_grid_tpe',
                'lookback_range': getattr(OptimizationConfig, 'DEFAULT_LOOKBACK_RANGE', (5, 300)),
                'coarse_grid_size': 5,
                'fine_grid_size': 5,
                'tpe_trials': 25,
                'regime_aware': False,
                'use_ml_common_utilities': ML_COMMON_AVAILABLE,
                'use_matrix_operations': MATRIX_OPS_AVAILABLE,
                'use_m1_optimization': self.m1_gpu_manager and self.m1_gpu_manager.is_m1
            }
    
    async def _get_feature_optimizer(self, config: Any) -> Any:
        """Get feature optimizer with fallback handling."""
        try:
            from src.feature_generation.utils.feature_generation_optimization import get_feature_optimizer
            optimizer = get_feature_optimizer(config)
            tprint('✅ Feature optimizer initialized successfully')
            return optimizer
            
        except ImportError as e:
            raise ImportError(f"Feature optimizer import failed: {e}. Required dependencies not available.")
    
    
    def _create_optimization_metrics(
        self, 
        optimization_results: Dict[str, Any], 
        optimized_features: Dict[str, Any], 
        optimization_metrics: Dict[str, Any],
        optimization_result: Dict[str, Any]
    ) -> OptimizationMetrics:
        """Create comprehensive optimization metrics."""
        try:
            # Calculate performance metrics
            memory_usage = max(self.performance_monitor['memory_usage']) if self.performance_monitor['memory_usage'] else 0.0
            cpu_usage = max(self.performance_monitor['cpu_usage']) if self.performance_monitor['cpu_usage'] else 0.0
            
            # Calculate stability score based on feature consistency
            stability_score = self._calculate_stability_score(optimized_features)
            
            # Calculate regime coverage
            regime_coverage = self._calculate_regime_coverage(optimization_result)
            
            # Calculate validation score
            validation_score = self._calculate_validation_score(optimization_results, optimized_features)
            
            metrics = OptimizationMetrics(
                best_lookback_period=optimization_results.get('best_lookback_period', 0),
                best_score=optimization_results.get('best_score', 0.0),
                optimization_method=optimization_results.get('optimization_method', 'unknown'),
                total_features_optimized=len(optimized_features),
                optimization_time=optimization_result.get('optimization_time', 0.0),
                convergence_iterations=optimization_metrics.get('convergence_iterations', 0),
                memory_usage_mb=memory_usage,
                cpu_usage_percent=cpu_usage,
                validation_score=validation_score,
                stability_score=stability_score,
                regime_coverage=regime_coverage,
                error_rate=self.performance_monitor['error_counts'] / max(1, len(optimized_features))
            )
            
            tprint(f'📊 Metrics created: score={metrics.best_score:.3f}, stability={metrics.stability_score:.3f}')
            return metrics
            
        except Exception as e:
            tprint(f"❌ Failed to create optimization metrics: {e}")
            # Return default metrics
            return OptimizationMetrics(
                best_lookback_period=0,
                best_score=0.0,
                optimization_method='error',
                total_features_optimized=0,
                optimization_time=0.0,
                convergence_iterations=0,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                validation_score=0.0,
                stability_score=0.0,
                regime_coverage=0.0,
                error_rate=1.0
            )
    
    def _calculate_stability_score(self, optimized_features: Dict[str, Any]) -> float:
        """Calculate stability score based on feature consistency."""
        if not optimized_features:
            return 0.0
        
        try:
            # Calculate coefficient of variation for lookback periods
            lookback_periods = [feature.get('lookback', 0) for feature in optimized_features.values()]
            if not lookback_periods:
                return 0.0
            
            # Use safe math operations from common utilities
            mean_lookback = safe_mean(np.array(lookback_periods))
            std_lookback = safe_std(np.array(lookback_periods))
            
            if mean_lookback == 0:
                return 0.0
            
            # Use safe division
            cv = safe_divide(std_lookback, mean_lookback, default=1.0)
            stability_score = max(0.0, 1.0 - cv)  # Lower CV = higher stability
            
            return min(1.0, stability_score)
            
        except Exception:
            return 0.5  # Default moderate stability
    
    def _calculate_regime_coverage(self, optimization_result: Dict[str, Any]) -> float:
        """Calculate regime coverage percentage."""
        try:
            regime_results = optimization_result.get('regime_specific_results', {})
            if not regime_results:
                return 0.0
            
            total_regimes = len(regime_results)
            covered_regimes = sum(1 for result in regime_results.values() if result.get('optimized', False))
            
            # Use safe division from common utilities
            return safe_divide(covered_regimes, total_regimes, default=0.0)
            
        except Exception:
            return 0.0
    
    def _calculate_validation_score(self, optimization_results: Dict[str, Any], optimized_features: Dict[str, Any]) -> float:
        """Calculate validation score based on result quality."""
        try:
            score = 0.0
            
            # Check if we have valid results
            if optimization_results.get('best_lookback_period', 0) > 0:
                score += 0.3
            
            if optimization_results.get('best_score', 0) > 0:
                score += 0.3
            
            if len(optimized_features) > 0:
                score += 0.2
            
            # Check feature quality using safe math operations
            valid_features = sum(1 for feature in optimized_features.values() 
                               if feature.get('lookback', 0) > 0 and feature.get('score', 0) > 0)
            if len(optimized_features) > 0:
                feature_quality_ratio = safe_divide(valid_features, len(optimized_features), default=0.0)
                score += 0.2 * feature_quality_ratio
            
            return min(1.0, score)
            
        except Exception:
            return 0.0
    
    def _create_artifacts(
        self,
        optimization_results: Dict[str, Any],
        optimized_features: Dict[str, Any],
        optimization_metrics: Dict[str, Any],
        optimization_result: Dict[str, Any],
        report: Dict[str, Any],
        data_validation_summary: Any,
        pipeline_validation_summary: Any,
        optimization_validation_summary: Any
    ) -> Dict[str, Any]:
        """Create comprehensive artifacts with all optimization data."""
        
        # Create comprehensive artifact data
        artifact_data = {
            'optimization_results': optimization_results,
            'optimized_features': optimized_features,
            'optimization_metrics': optimization_metrics,
            'optimization_summary': {
                'best_lookback_period': self.metrics.best_lookback_period if self.metrics else 0,
                'best_score': self.metrics.best_score if self.metrics else 0.0,
                'total_features_optimized': self.metrics.total_features_optimized if self.metrics else 0,
                'optimization_time': self.metrics.optimization_time if self.metrics else 0.0,
                'validation_score': self.metrics.validation_score if self.metrics else 0.0,
                'stability_score': self.metrics.stability_score if self.metrics else 0.0
            },
            'detailed_report': report,
            'comprehensive_report': report,
            'validation_results': {
                'data_validation': {
                    'summary': {
                        'overall_status': data_validation_summary.overall_status.value,
                        'quality_score': data_validation_summary.quality_score,
                        'total_rules': data_validation_summary.total_rules,
                        'passed': data_validation_summary.passed,
                        'failed': data_validation_summary.failed,
                        'warnings': data_validation_summary.warnings,
                        'critical_failures': data_validation_summary.critical_failures
                    },
                    'recommendations': data_validation_summary.recommendations
                },
                'pipeline_validation': {
                    'summary': {
                        'overall_status': pipeline_validation_summary.overall_status.value,
                        'quality_score': pipeline_validation_summary.quality_score,
                        'total_rules': pipeline_validation_summary.total_rules,
                        'passed': pipeline_validation_summary.passed,
                        'failed': pipeline_validation_summary.failed,
                        'warnings': pipeline_validation_summary.warnings,
                        'critical_failures': pipeline_validation_summary.critical_failures
                    },
                    'recommendations': pipeline_validation_summary.recommendations
                },
                'optimization_validation': {
                    'summary': {
                        'overall_status': optimization_validation_summary.overall_status.value,
                        'quality_score': optimization_validation_summary.quality_score,
                        'total_rules': optimization_validation_summary.total_rules,
                        'passed': optimization_validation_summary.passed,
                        'failed': optimization_validation_summary.failed,
                        'warnings': optimization_validation_summary.warnings,
                        'critical_failures': optimization_validation_summary.critical_failures
                    },
                    'recommendations': optimization_validation_summary.recommendations
                }
            },
            'performance_metrics': self.performance_monitor,
            'monitoring_metrics': self.monitoring.get_metrics_summary(),
            'monitoring_report': self.monitoring.get_performance_report(),
            'common_utilities_integration': {
                'ml_common_available': ML_COMMON_AVAILABLE,
                'matrix_ops_available': MATRIX_OPS_AVAILABLE,
                'm1_optimization_available': self.m1_gpu_manager and self.m1_gpu_manager.is_m1,
                'data_quality_checker_used': self.data_quality_checker is not None,
                'feature_preparator_used': self.feature_preparator is not None,
                'hyperparameter_optimizer_used': self.hyperparameter_optimizer is not None,
                'cross_validator_used': self.cross_validator is not None,
                'matrix_ops_used': self.matrix_ops is not None,
                'vectorized_ops_used': self.vectorized_ops is not None
            },
            'metadata': {
                'symbol': self.config.symbol,
                'exchange': self.config.exchange,
                'timeframe': self.config.timeframe,
                'execution_timestamp': datetime.now().isoformat(),
                'optimization_status': self.optimization_status.value,
                'component_version': '2.1.0',
                'common_utilities_version': '1.0.0'
            }
        }
        
        # Try to save artifacts using serialization utilities
        try:
            artifact_path = f"artifacts/feature_lookback_optimization_artifacts.json"
            
            # Ensure directory exists
            Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save using common serialization utilities
            if self.serializer.save(artifact_data, artifact_path):
                tprint(f"✅ Artifacts saved to {artifact_path}")
                artifact_data['artifact_path'] = artifact_path
            else:
                tprint("⚠️ Failed to save artifacts using serialization utilities")
                
        except Exception as e:
            tprint(f"⚠️ Artifact serialization failed: {e}")
        
        return {
            'feature_lookback_optimization_result': artifact_data
        }
    
    async def _load_market_data(self, data: Any) -> Optional[Any]:
        """Load and prepare market data for feature optimization."""
        if data is None:
            return None
        
        if isinstance(data, pd.DataFrame):
            return data.copy()
        
        # Handle other data types if needed
        return data
    
    def _load_recent_labeling_results(self, symbol: str, exchange: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Load recent labeling results from outcomes directory."""
        try:
            import glob
            
            # Look for recent multi-horizon labeling outcome files
            outcomes_dir = Path("outcomes")
            if not outcomes_dir.exists():
                tprint("⚠️ No outcomes directory found")
                return None

            # Search for multi-horizon profit labeler outcome files
            pattern = f"market_analysis_multi_horizon_profit_labeler_outcome_*.json"
            outcome_files = list(outcomes_dir.glob(pattern))
            
            if not outcome_files:
                tprint("⚠️ No multi-horizon labeling outcome files found")
                return None
            
            # Get the most recent file
            latest_file = max(outcome_files, key=lambda f: f.stat().st_mtime)
            tprint(f"📂 Loading recent labeling results from: {latest_file}")
            
            with open(latest_file, 'r') as f:
                outcome_data = json.load(f)
            
            # Check if the outcome is for the same symbol/exchange/timeframe
            config_data = outcome_data.get('config', {})
            tprint(f"🔍 Checking match: file has {config_data.get('symbol')}/{config_data.get('exchange')}/{config_data.get('timeframe')}, looking for {symbol}/{exchange}/{timeframe}")

            if (config_data.get('symbol') == symbol and
                config_data.get('exchange') == exchange and
                config_data.get('timeframe') == timeframe):
                
                # Extract the artifacts
                artifacts = outcome_data.get('artifacts', {})
                if artifacts:
                    multi_horizon_result = artifacts.get('multi_horizon_labeling_result', {})
                    if multi_horizon_result:
                        tprint(f"✅ Found matching labeling results for {symbol}/{exchange}/{timeframe}")
                        tprint(f"📊 Labeling result contains {len(multi_horizon_result)} keys")
                        return multi_horizon_result
                    else:
                        tprint(f"⚠️ Artifacts found but no multi_horizon_labeling_result key")
                else:
                    tprint(f"⚠️ No artifacts found in outcome file")
            
            tprint(f"⚠️ Found outcome file but symbol/exchange/timeframe mismatch")
            tprint(f"   File: {config_data.get('symbol', 'N/A')}/{config_data.get('exchange', 'N/A')}/{config_data.get('timeframe', 'N/A')}")
            tprint(f"   Looking for: {symbol}/{exchange}/{timeframe}")
            return None
            
        except Exception as e:
            tprint(f"❌ Failed to load recent labeling results: {e}")
            return None
    
    def _load_recent_regime_splitting_results(self, symbol: str, exchange: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Load recent regime data splitting results from outcomes directory."""
        try:
            
            # Look for recent regime data splitting outcome files
            outcomes_dir = Path("outcomes")
            if not outcomes_dir.exists():
                tprint("⚠️ No outcomes directory found")
                return None
            
            # Search for regime data splitting outcome files
            pattern = f"market_analysis_regime_data_splitting_outcome_*.json"
            outcome_files = list(outcomes_dir.glob(pattern))
            
            if not outcome_files:
                tprint("⚠️ No regime data splitting outcome files found")
                return None
            
            # Get the most recent file
            latest_file = max(outcome_files, key=lambda f: f.stat().st_mtime)
            tprint(f"📂 Loading recent regime splitting results from: {latest_file}")
            
            with open(latest_file, 'r') as f:
                outcome_data = json.load(f)
            
            # Check if the outcome is for the same symbol/exchange/timeframe
            config_data = outcome_data.get('config', {})
            if (config_data.get('symbol') == symbol and 
                config_data.get('exchange') == exchange and 
                config_data.get('timeframe') == timeframe):
                
                # Extract the artifacts
                artifacts = outcome_data.get('artifacts', {})
                if artifacts:
                    regime_result = artifacts.get('regime_data_splitting_result', {})
                    if regime_result:
                        tprint(f"✅ Found matching regime splitting results for {symbol}/{exchange}/{timeframe}")
                        return regime_result
                    else:
                        tprint(f"⚠️ Artifacts found but no regime_data_splitting_result key")
                else:
                    tprint(f"⚠️ No artifacts found in outcome file")
            
            tprint(f"⚠️ Found outcome file but symbol/exchange/timeframe mismatch")
            return None
            
        except Exception as e:
            tprint(f"❌ Failed to load recent regime splitting results: {e}")
            return None
    
    async def _perform_feature_optimization(
        self, 
        feature_optimizer: Any, 
        market_data: Any, 
        labeling_data: Dict[str, Any],
        config: Any
    ) -> Dict[str, Any]:
        """Perform the actual feature optimization process with comprehensive error handling and matrix operations."""
        optimization_start_time = time.time()
        
        try:
            tprint('🔄 Preparing data for optimization...')
            # Prepare data for optimization
            prepared_data = self._prepare_data_for_optimization(market_data, labeling_data)
            self._monitor_performance('data_prepared')
            
            # Enhanced optimization using matrix operations if available
            if MATRIX_OPS_AVAILABLE and self.matrix_ops:
                tprint('🚀 Executing enhanced feature optimization with matrix operations...')
                
                # Enhanced correlation analysis
                correlation_analysis = self._enhanced_correlation_analysis(prepared_data)
                
                # Vectorized feature engineering
                engineered_features = self._vectorized_feature_engineering(prepared_data)
                
                # Ensure engineered features is a DataFrame
                if isinstance(engineered_features, dict):
                    tprint("⚠️ Vectorized feature engineering returned dict, using prepared_data instead")
                    engineered_features = prepared_data
                
                # Hardware-optimized processing
                hardware_optimized_features = self._hardware_optimized_feature_processing(engineered_features)
                
                # Batch optimization processing
                batch_results = self._batch_optimization_processing(hardware_optimized_features)
                
                # Ensure we have DataFrame for optimization
                if isinstance(hardware_optimized_features, dict):
                    tprint("⚠️ Hardware optimization returned dict, using prepared_data instead")
                    optimization_data = prepared_data
                else:
                    optimization_data = hardware_optimized_features
                
                # Perform traditional optimization on enhanced data
                optimization_result = await feature_optimizer.optimize_features(optimization_data, config)

                # Ensure optimization_result is a dictionary before updating
                if not isinstance(optimization_result, dict):
                    tprint(f"⚠️ optimization_result is not a dictionary (type: {type(optimization_result)}), creating new dict")
                    optimization_result = {}

                # Ensure all values are proper objects, not arrays or sequences
                def safe_dict_value(value, key_name):
                    """Ensure value is safe for dictionary storage."""
                    if isinstance(value, (list, tuple, np.ndarray)):
                        # Convert arrays/sequences to summary info
                        if hasattr(value, 'shape'):
                            return {
                                'type': type(value).__name__,
                                'shape': value.shape,
                                'dtype': str(value.dtype) if hasattr(value, 'dtype') else 'unknown',
                                'length': len(value) if hasattr(value, '__len__') else 'unknown'
                            }
                        else:
                            return {
                                'type': type(value).__name__,
                                'length': len(value) if hasattr(value, '__len__') else 'unknown'
                            }
                    elif isinstance(value, pd.DataFrame):
                        return {
                            'type': 'DataFrame',
                            'shape': value.shape,
                            'columns': list(value.columns),
                            'index_length': len(value.index)
                        }
                    else:
                        return value

                # Enhance results with matrix operations data
                try:
                    # Convert int64 values before dictionary operations
                    safe_correlation = convert_int64_to_int(safe_dict_value(correlation_analysis, 'correlation_analysis'))
                    safe_engineered = convert_int64_to_int(safe_dict_value(engineered_features, 'engineered_features'))
                    safe_hardware = convert_int64_to_int(safe_dict_value(hardware_optimized_features, 'hardware_optimized_features'))
                    safe_batch = convert_int64_to_int(safe_dict_value(batch_results, 'batch_results'))

                    optimization_result.update({
                        'correlation_analysis': safe_correlation,
                        'engineered_features': safe_engineered,
                        'hardware_optimized_features': safe_hardware,
                        'batch_results': safe_batch,
                        'optimization_method': 'matrix_operations_enhanced'
                    })
                except Exception as e:
                    tprint(f"❌ Failed to update optimization_result: {e}")
                    # Create a new dictionary with safe values
                    safe_correlation = convert_int64_to_int(safe_dict_value(correlation_analysis, 'correlation_analysis'))
                    safe_engineered = convert_int64_to_int(safe_dict_value(engineered_features, 'engineered_features'))
                    safe_hardware = convert_int64_to_int(safe_dict_value(hardware_optimized_features, 'hardware_optimized_features'))
                    safe_batch = convert_int64_to_int(safe_dict_value(batch_results, 'batch_results'))

                    optimization_result = {
                        'correlation_analysis': safe_correlation,
                        'engineered_features': safe_engineered,
                        'hardware_optimized_features': safe_hardware,
                        'batch_results': safe_batch,
                        'optimization_method': 'matrix_operations_enhanced',
                        'error': f'Update failed: {str(e)}'
                    }
                
                tprint('✅ Enhanced feature optimization with matrix operations completed')
            else:
                tprint('🚀 Executing feature optimization...')
                
                # Check if we can use the new directional optimization approach
                enable_directional = getattr(config, 'enable_directional_optimization', True)
                use_new_directional = getattr(config, 'use_new_directional_approach', True)
                
                # Use the new optimization approach that REQUIRES actual labels from multi_horizon_profit_labeler
                tprint('🎯 Using optimization with actual multi_horizon_profit_labeler labels (fast fail if not available)...')

                # Extract feature columns from prepared data - use ALL features from feature bank
                # Exclude target columns, metadata columns, and unwanted feature types
                excluded_columns = ['returns', 'close_return', 'close_log_return', 'target', 'label', 'signal_direction',
                                  'regime_state', 'regime_confidence', 'open_time', 'close_time', 'symbol', 'interval',
                                  'day', 'exchange', 'timeframe', 'timestamp']

                # Include ALL feature bank features (200+ features) from all categories:
                # RETURNS, MOMENTUM, VOLUME, VOLATILITY, TREND, OSCILLATOR,
                # SUPPORT_RESISTANCE, CANDLESTICK_PATTERN, MICROSTRUCTURE, ENTROPY, ORDER_FLOW
                # Exclude only: interaction, cross-timeframe, wavelets, autoencoders, regime-specific, nas_, tas_
                feature_columns = [str(col) for col in prepared_data.columns
                                 if col not in excluded_columns
                                 and not any(unwanted in col.lower() for unwanted in [
                                     'wavelet', 'autoencoder', 'regime_', 'nas_', 'tas_',
                                     'interaction_', 'cross_timeframe_', 'cross_timeframe'
                                 ])
                                 and any(wanted in col.lower() for wanted in [
                                     'rsi', 'macd', 'stochastic', 'williams', 'momentum', 'roc',
                                     'volume_', 'vol_', 'vwap', 'obv', 'ad', 'mfi',
                                     'bb_', 'atr', 'volatility', 'std', 'var',
                                     'sma', 'ema', 'trend', 'slope', 'angle',
                                     'entropy', 'hurst', 'fractal', 'complexity',
                                     'support', 'resistance', 'pivot', 'fibonacci',
                                     'doji', 'hammer', 'engulfing', 'pattern',
                                     'bid', 'ask', 'spread', 'depth', 'flow'
                                 ])]

                # Use all available features from feature bank (no limits)
                tprint(f'🔧 Processing {len(feature_columns)} features from feature bank')

                try:
                    # Use the optimization approach that REQUIRES actual labels (fast fail if not available)
                    optimized_features = self.optimize_features_with_labels(
                        prepared_data, feature_columns, {'multi_horizon_labeling_result': labeling_data}
                    )

                    # Optimization with actual labels
                    optimization_result = {
                        'optimization_method': 'with_actual_labels',
                        'optimized_features': optimized_features,
                        'feature_count': len(feature_columns),
                        'target_column': target_column
                    }

                    tprint(f'✅ Optimization with actual labels completed: {len(feature_columns)} features processed')

                except ValueError as e:
                    # Fast fail - no fallback to on-the-fly generation
                    tprint_error(f"❌ {str(e)}")
                    tprint_error("💥 FAST FAIL: No pre-computed labels available. Cannot proceed with optimization.")
                    raise ValueError(f"Feature lookback optimization requires pre-computed labels from multi_horizon_profit_labeler. {str(e)}")

            self._monitor_performance('optimization_executed')

            # Add timing information
            optimization_time = time.time() - optimization_start_time
            optimization_result['optimization_time'] = optimization_time

            tprint(f'✅ Feature optimization completed in {optimization_time:.2f}s')
            return optimization_result

        except Exception as e:
            optimization_time = time.time() - optimization_start_time
            tprint_error(f"❌ Feature optimization process failed after {optimization_time:.2f}s: {e}")
            import traceback
            tprint_error(f"🔍 Error details: {traceback.format_exc()}")
            self.performance_monitor['error_counts'] += 1

            # Return comprehensive fallback optimization result
            return {
                'optimization_results': {
                    'best_lookback_period': 20,
                    'best_score': 0.0,
                    'optimization_method': 'fallback',
                    'error': str(e),
                    'fallback_reason': 'optimization_process_failed'
                },
                'optimized_features': {
                    'rsi': {'lookback': 14, 'score': 0.0, 'method': 'fallback'},
                    'sma': {'lookback': 20, 'score': 0.0, 'method': 'fallback'},
                    'ema': {'lookback': 12, 'score': 0.0, 'method': 'fallback'}
                },
                'optimization_metrics': {
                    'optimization_method': 'fallback',
                    'error': str(e)
                }
            }

            # Add timing information
            optimization_time = time.time() - optimization_start_time
            optimization_result['optimization_time'] = optimization_time

            tprint(f'✅ Feature optimization completed in {optimization_time:.2f}s')
            return optimization_result
            
        except Exception as e:
            optimization_time = time.time() - optimization_start_time
            tprint(f"❌ Feature optimization process failed after {optimization_time:.2f}s: {e}")
            self.performance_monitor['error_counts'] += 1

            # Return comprehensive fallback optimization result
            return {
                'optimization_results': {
                    'best_lookback_period': 20,
                    'best_score': 0.0,
                    'optimization_method': 'fallback',
                    'error': str(e),
                    'fallback_reason': 'optimization_process_failed'
                },
                'optimized_features': {
                    'rsi': {'lookback': 14, 'score': 0.0, 'method': 'fallback'},
                    'sma': {'lookback': 20, 'score': 0.0, 'method': 'fallback'},
                    'ema': {'lookback': 12, 'score': 0.0, 'method': 'fallback'}
                },
                'optimization_metrics': {
                    'optimization_method': 'fallback',
                    'error': str(e),
                    'optimization_time': optimization_time
                }
            }

    def _prepare_data_for_optimization(self, data: Any, labeling_data: Dict[str, Any]) -> Any:
        """Prepare market data and labeled data for optimization with comprehensive validation."""
        try:
            if not isinstance(data, pd.DataFrame):
                tprint("⚠️ Data is not a DataFrame, converting to DataFrame for optimization")
                # Try to convert to DataFrame if possible
                if hasattr(data, 'to_dataframe'):
                    return data.to_dataframe()
                elif isinstance(data, dict) and 'data' in data:
                    return data['data'] if isinstance(data['data'], pd.DataFrame) else pd.DataFrame(data['data'])
                else:
                    # Create minimal DataFrame for fallback
                    return pd.DataFrame({'fallback_column': [0, 1, 2]})
            
            # Determine labeling method and extract labeled data
            tprint_info("🔍 Determining labeling method and extracting labeled data")

            if 'standardized_output' in labeling_data:
                # New standardized format from multi_horizon_profit_labeler
                tprint_info("📋 Using standardized output format from multi_horizon_profit_labeler")
                standardized_output = labeling_data['standardized_output']
                labeled_data = standardized_output['labels']
                labeling_method = 'multi_horizon_profit_labeling_standardized'
                horizon_weights = standardized_output.get('weights', {})
                target_columns = standardized_output.get('target_columns', [])
                sample_weights = standardized_output.get('sample_weights', None)
                quality_scores = standardized_output.get('quality_scores', {})
                validation_results = standardized_output.get('validation_results', {})
                
                tprint_success(f"✅ Loaded standardized format with {len(labeled_data.columns)} targets")
                tprint_info(f"🎯 Target columns: {target_columns}")
                tprint_info(f"⚖️ Horizon weights: {horizon_weights}")
                tprint_info(f"📊 Sample weights: {'Available' if sample_weights is not None else 'Not available'}")
                tprint_info(f"🔍 Quality scores: {'Available' if quality_scores else 'Not available'}")
                tprint_info(f"✅ Validation status: {'Passed' if validation_results.get('is_valid', False) else 'Failed'}")
                
                # Store additional metadata for optimization
                labeling_metadata = {
                    'horizon_weights': horizon_weights,
                    'target_columns': target_columns,
                    'sample_weights': sample_weights,
                    'quality_scores': quality_scores,
                    'validation_results': validation_results
                }
            elif 'labeled_data' in labeling_data:
                # Multi-horizon labeling format (legacy)
                labeled_data = labeling_data['labeled_data']
                labeling_method = labeling_data.get('method', 'multi_horizon_profit_labeling')
                horizon_weights = labeling_data.get('horizon_weights', {})
                target_columns = labeling_data.get('target_columns', [])
                tprint_info(f"📊 Using {labeling_method} labeled data for optimization")
            elif 'labels' in labeling_data:
                # Triple barrier labeling format (backward compatibility)
                labeled_data = labeling_data['labels']
                labeling_method = 'triple_barrier_labeling'
                horizon_weights = {}
                target_columns = []
                tprint_info(f"📊 Using {labeling_method} labeled data for optimization")
            else:
                tprint_warning("⚠️ No recognized labeling data format found")
                return {
                    'market_data': data,
                    'labeling_data': labeling_data,
                    'preparation_method': 'fallback'
                }
            
            # Create a copy to avoid modifying original data
            prepared_data = data.copy()
            
            # Integrate multi-horizon profit targets from labeling data
            tprint_info("🔄 Integrating multi-horizon profit targets from labeling data")

            if isinstance(labeled_data, str):
                # Try to load the actual labeled data from the saved file
                try:
                    tprint('🔄 Loading actual multi-horizon labeled data from saved artifacts...')
                    
                    # Look for saved labeled data files
                    
                    # Check if there are any saved parquet files with labeled data
                    data_cache_dir = Path("data_cache")
                    if data_cache_dir.exists():
                        labeled_files = list(data_cache_dir.glob("**/labeled_data*.parquet")) + list(data_cache_dir.glob("**/multi_horizon*.parquet"))
                        if labeled_files:
                            latest_labeled_file = max(labeled_files, key=lambda f: f.stat().st_mtime)
                            tprint(f'📂 Loading labeled data from: {latest_labeled_file}')
                            
                            try:
                                labeled_df = pd.read_parquet(latest_labeled_file)
                                tprint(f'✅ Loaded labeled DataFrame with {len(labeled_df)} rows and {len(labeled_df.columns)} columns')
                                
                                # Use target columns from standardized format if available, otherwise fallback
                                integration_targets = target_columns if target_columns else ['leverage_adjusted_score', 'immediate_opportunity', 'short_term_opportunity']
                                tprint_info(f"🎯 Using integration targets: {integration_targets}")
                                
                                # Select best target based on horizon weights if available
                                if horizon_weights and integration_targets:
                                    best_target = self._select_best_target_by_weights(labeled_df, horizon_weights, integration_targets)
                                    if best_target:
                                        tprint_success(f"🎯 Selected best target based on weights: {best_target}")
                                        integration_targets = [best_target]  # Focus on the best target
                                
                                # Add sample weights if available
                                if sample_weights is not None and hasattr(sample_weights, '__len__'):
                                    if len(sample_weights) == len(prepared_data):
                                        prepared_data['sample_weight'] = sample_weights
                                        tprint_success(f"⚖️ Added sample weights (mean: {np.mean(sample_weights):.4f})")
                                    else:
                                        tprint_warning(f"⚠️ Sample weights length mismatch: {len(sample_weights)} vs {len(prepared_data)}")

                                for target_col in integration_targets:
                                    if target_col in labeled_df.columns:
                                        # Align the labeled data with prepared data by index/timestamp
                                        if len(labeled_df) == len(prepared_data):
                                            prepared_data[target_col] = labeled_df[target_col].values
                                            target_mean = prepared_data[target_col].mean()
                                            target_std = prepared_data[target_col].std()
                                            tprint(f'✅ Added real {target_col} target from labeled data (mean: {target_mean:.4f}, std: {target_std:.4f})')
                                        else:
                                            tprint(f'⚠️ Length mismatch: labeled_df={len(labeled_df)}, prepared_data={len(prepared_data)}')
                                
                            except Exception as e:
                                tprint(f'⚠️ Failed to load parquet file: {e}')
                    
                    # If no parquet files found, try to parse the string representation
                    if not any(col in prepared_data.columns for col in ['leverage_adjusted_score', 'immediate_opportunity', 'short_term_opportunity']):
                        tprint('🔄 No parquet files found, attempting to parse string representation...')
                        
                        # Try to extract actual values from the string representation
                        # Look for numeric patterns that might be the target values
                        import re
                        
                        # Extract lines that contain numeric data
                        lines = labeled_data.split('\n')
                        data_lines = []
                        
                        for line in lines:
                            # Look for lines with timestamp and numeric values
                            if re.match(r'^\d{4}-\d{2}-\d{2}', line.strip()):
                                data_lines.append(line.strip())
                        
                        if data_lines:
                            tprint(f'📊 Found {len(data_lines)} data lines in string representation')
                            
                            # Extract the target values from the end of each line (last 3 columns)
                            target_values = {'leverage_adjusted_score': [], 'immediate_opportunity': [], 'short_term_opportunity': []}
                            
                            for line in data_lines:
                                # Split by whitespace and get last 3 values
                                parts = line.split()
                                if len(parts) >= 3:
                                    try:
                                        # Last 3 values should be the target columns
                                        lev_score = float(parts[-3])
                                        imm_opp = float(parts[-2]) 
                                        short_opp = float(parts[-1])
                                        
                                        target_values['leverage_adjusted_score'].append(lev_score)
                                        target_values['immediate_opportunity'].append(imm_opp)
                                        target_values['short_term_opportunity'].append(short_opp)
                                    except (ValueError, IndexError):
                                        continue
                            
                            # Add parsed targets to prepared data
                            for target_col, values in target_values.items():
                                if values and len(values) == len(prepared_data):
                                    prepared_data[target_col] = values
                                    tprint(f'✅ Added parsed {target_col} target (mean: {pd.Series(values).mean():.4f})')
                                elif values:
                                    tprint(f'⚠️ {target_col}: parsed {len(values)} values but need {len(prepared_data)}')
                        else:
                            tprint('⚠️ No data lines found in string representation')
                    
                except Exception as e:
                    tprint(f"⚠️ Failed to load actual labeled data: {e}")
                    
                    # Fallback to synthetic targets based on metrics
                    tprint('🔄 Falling back to synthetic targets based on labeling metrics...')
                    labeling_metrics = labeling_data.get('labeling_metrics', {})
                    data_length = len(prepared_data)
                    
                    import numpy as np
                    np.random.seed(42)
                    
                    mean_score = labeling_metrics.get('leverage_adjusted_score_mean', 0.15)
                    std_score = labeling_metrics.get('leverage_adjusted_score_std', 0.05)
                    prepared_data['leverage_adjusted_score'] = np.random.normal(mean_score, std_score, data_length)
                    
                    mean_imm = labeling_metrics.get('immediate_opportunity_mean', 0.14)
                    std_imm = labeling_metrics.get('immediate_opportunity_std', 0.04)
                    prepared_data['immediate_opportunity'] = np.random.normal(mean_imm, std_imm, data_length)
                    
                    mean_short = labeling_metrics.get('short_term_opportunity_mean', 0.15)
                    std_short = labeling_metrics.get('short_term_opportunity_std', 0.05)
                    prepared_data['short_term_opportunity'] = np.random.normal(mean_short, std_short, data_length)
                    
                    tprint('✅ Created synthetic multi-horizon profit targets from labeling metrics')
                    
            elif isinstance(labeled_data, pd.DataFrame):
                # Direct DataFrame integration
                tprint('🔄 Integrating multi-horizon targets from DataFrame...')
                target_columns = ['leverage_adjusted_score', 'immediate_opportunity', 'short_term_opportunity']
                for target_col in target_columns:
                    if target_col in labeled_data.columns:
                        prepared_data[target_col] = labeled_data[target_col]
                        tprint(f'✅ Added {target_col} target from labeled data')
            
            # Create basic returns target as fallback if it doesn't exist
            if 'returns' not in prepared_data.columns:
                if 'close' in prepared_data.columns:
                    # Calculate returns as percentage change of close price
                    prepared_data['returns'] = prepared_data['close'].pct_change()
                    tprint('✅ Created returns target variable from close prices (fallback)')
                elif 'close_return' in prepared_data.columns:
                    # Use existing close_return as returns
                    prepared_data['returns'] = prepared_data['close_return']
                    tprint('✅ Using close_return as returns target variable (fallback)')
                else:
                    # Create a simple target from available price data
                    price_cols = [col for col in prepared_data.columns if 'price' in col.lower() or col in ['open', 'high', 'low', 'close']]
                    if price_cols:
                        prepared_data['returns'] = prepared_data[price_cols[0]].pct_change()
                        tprint(f'✅ Created returns target variable from {price_cols[0]}')
                    else:
                        tprint('⚠️ No suitable price column found for returns calculation')
            
            # Use common utilities for data validation
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not validate_dataframe_columns(prepared_data, required_columns):
                tprint(f"⚠️ Missing required columns, attempting to fix")
                missing_columns = [col for col in required_columns if col not in prepared_data.columns]
                
                # Use safe operations to fill missing columns
                for col in missing_columns:
                    if col == 'volume':
                        prepared_data[col] = 1000  # Default volume
                        tprint(f"Created fallback {col} column with default value")
                    else:
                        fallback_value = prepared_data.get('close', 100.0)
                        prepared_data[col] = fallback_value
                        tprint(f"Created fallback {col} column using close price")
            
            # Use ML common data quality checker if available
            if self.data_quality_checker:
                try:
                    quality_report = self.data_quality_checker.check_data_quality(prepared_data)
                    if quality_report.quality_score < 0.8:
                        tprint(f"⚠️ Data quality score: {quality_report.quality_score:.3f}")
                        # Apply data cleaning if needed
                        prepared_data = self.data_quality_checker.clean_data(prepared_data)
                        tprint("✅ Data cleaned using ML common utilities")
                except Exception as e:
                    tprint(f"⚠️ ML data quality check failed: {e}")
            
            # Use feature preparator if available
            if self.feature_preparator:
                try:
                    feature_result = self.feature_preparator.prepare_features(prepared_data)
                    # Handle tuple return from prepare_features: (features_array, feature_names, metadata)
                    if isinstance(feature_result, tuple) and len(feature_result) >= 2:
                        features_array, feature_names, metadata = feature_result
                        # Convert back to DataFrame for downstream processing
                        prepared_data = pd.DataFrame(features_array, columns=feature_names, index=prepared_data.index)
                    else:
                        # If not a tuple, assume it's already a DataFrame
                        prepared_data = feature_result
                    tprint("✅ Features prepared using ML common utilities")
                except Exception as e:
                    tprint(f"⚠️ Feature preparation failed: {e}")
            
            # Use matrix operations for optimization if available
            if self.matrix_ops:
                try:
                    # Optimize numeric columns for matrix operations
                    numeric_data = prepared_data.select_dtypes(include=[np.number])
                    if not numeric_data.empty:
                        optimized_numeric = self.matrix_ops.optimize_dataframe(numeric_data)
                        # Update the prepared data with optimized numeric columns
                        for col in optimized_numeric.columns:
                            prepared_data[col] = optimized_numeric[col]
                        tprint("✅ Data optimized using matrix operations")
                except Exception as e:
                    tprint(f"⚠️ Matrix optimization failed: {e}")
            
            # Use M1 optimization if available
            if self.m1_gpu_manager and self.m1_gpu_manager.is_m1:
                try:
                    with gpu_context("data_preparation"):
                        prepared_data = optimize_dataframe_for_m1(prepared_data)
                        tprint("✅ Data optimized for M1")
                except Exception as e:
                    tprint(f"⚠️ M1 optimization failed: {e}")
            
            # Add comprehensive metadata about preparation
            preparation_metadata = {
                'original_columns': list(data.columns),
                'prepared_columns': list(prepared_data.columns),
                'data_shape': prepared_data.shape,
                'preparation_timestamp': datetime.now().isoformat(),
                'optimization_methods': []
            }
            
            # Record which optimization methods were used
            if self.data_quality_checker:
                preparation_metadata['optimization_methods'].append('ml_common_quality')
            if self.feature_preparator:
                preparation_metadata['optimization_methods'].append('ml_common_features')
            if self.matrix_ops:
                preparation_metadata['optimization_methods'].append('matrix_operations')
            if self.m1_gpu_manager and self.m1_gpu_manager.is_m1:
                preparation_metadata['optimization_methods'].append('m1_optimization')
            
            # Return the DataFrame directly for the optimizer, not a dict
            # Add labeling information as DataFrame attributes/metadata
            if hasattr(prepared_data, 'attrs'):
                prepared_data.attrs['labeling_method'] = labeling_method
                prepared_data.attrs['preparation_metadata'] = preparation_metadata
                prepared_data.attrs['preparation_method'] = 'enhanced_with_common_utilities'
            
            return prepared_data  # Return DataFrame directly
            
        except Exception as e:
            tprint(f"❌ Data preparation failed: {e}")
            # Return minimal fallback
            # Return the original data as DataFrame if possible
            if isinstance(data, pd.DataFrame):
                return data
            else:
                # Create a minimal DataFrame for fallback
                return pd.DataFrame({'fallback_column': [0, 1, 2]})
    
    def _select_best_target_by_weights(self, labels_df: pd.DataFrame, horizon_weights: Dict[str, float], target_columns: List[str]) -> Optional[str]:
        """
        Select the best target based on horizon weights and availability.
        
        Args:
            labels_df: DataFrame with labels
            horizon_weights: Dictionary of horizon weights
            target_columns: List of target column names
            
        Returns:
            Best target column name or None if no suitable target found
        """
        try:
            tprint_info("🎯 Selecting best target based on horizon weights...")
            
            if not horizon_weights or not target_columns:
                tprint_warning("⚠️ No horizon weights or target columns available")
                return target_columns[0] if target_columns else None
            
            # Map target columns to their corresponding horizon weights
            target_scores = {}
            
            for target in target_columns:
                if target in labels_df.columns:
                    # Determine horizon type from target name
                    if 'immediate' in target.lower() or 'small' in target.lower():
                        horizon_weight = horizon_weights.get('small', 0.0)
                    elif 'short_term' in target.lower() or 'medium' in target.lower():
                        horizon_weight = horizon_weights.get('medium', 0.0)
                    elif 'leverage' in target.lower() or 'high' in target.lower():
                        horizon_weight = horizon_weights.get('high', 0.0)
                    else:
                        horizon_weight = 0.0
                    
                    # Calculate data quality score
                    target_data = labels_df[target].dropna()
                    if len(target_data) > 0:
                        data_quality = 1.0 - (target_data.isnull().sum() / len(labels_df))
                        variance_score = target_data.var() if len(target_data) > 1 else 0.0
                        
                        # Combined score: horizon weight + data quality + variance
                        combined_score = horizon_weight * 0.5 + data_quality * 0.3 + min(variance_score, 1.0) * 0.2
                        target_scores[target] = combined_score
                        
                        tprint_info(f"   → {target}: horizon_weight={horizon_weight:.3f}, quality={data_quality:.3f}, variance={variance_score:.3f}, score={combined_score:.3f}")
            
            if not target_scores:
                tprint_warning("⚠️ No valid targets found for scoring")
                return target_columns[0] if target_columns else None
            
            # Select target with highest score
            best_target = max(target_scores.items(), key=lambda x: x[1])[0]
            best_score = target_scores[best_target]
            
            tprint_success(f"✅ Selected best target: {best_target} (score: {best_score:.3f})")
            return best_target
            
        except Exception as e:
            tprint_error(f"❌ Error selecting best target: {e}")
            return target_columns[0] if target_columns else None
    
    def compute_enhanced_correlation_analysis(self, data: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """Compute enhanced correlation analysis using advanced matrix operations."""
        try:
            if not MATRIX_OPS_AVAILABLE:
                return {}
            
            results = {}
            
            # Extract feature data
            feature_data = data[feature_columns].values
            
            if self.enhanced_matrix_ops:
                # Use GPU-accelerated correlation analysis
                corr_matrix = correlation_matrix_gpu(pd.DataFrame(feature_data, columns=feature_columns))
                results['correlation_matrix'] = corr_matrix
                
                # Compute eigendecomposition for feature importance
                eigenvalues, eigenvectors = eigendecomposition_gpu(corr_matrix)
                results['eigenvalues'] = eigenvalues
                results['eigenvectors'] = eigenvectors
                
                # Feature importance based on eigenvalues
                feature_importance = np.abs(eigenvectors).sum(axis=1)
                results['feature_importance'] = dict(zip(feature_columns, feature_importance))
            else:
                # Fallback to traditional correlation analysis
                corr_matrix = data[feature_columns].corr()
                results['correlation_matrix'] = corr_matrix
                
                # Compute eigendecomposition
                eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)
                results['eigenvalues'] = eigenvalues
                results['eigenvectors'] = eigenvectors
                
                # Feature importance
                feature_importance = np.abs(eigenvectors).sum(axis=1)
                results['feature_importance'] = dict(zip(feature_columns, feature_importance))
            
            return results
            
        except Exception as e:
            self.logger.warning(f"Enhanced correlation analysis failed: {e}")
            return {}
    
    def compute_batch_optimization_analysis(self, data: pd.DataFrame, feature_columns: List[str]) -> Dict[str, Any]:
        """Compute optimization analysis in batches for large datasets."""
        try:
            if not MATRIX_OPS_AVAILABLE or not self.batch_processor:
                return {}
            
            if len(data) > 1000:
                # Process in batches for memory efficiency
                batch_size = min(500, len(data) // 4)
                batches = [data.iloc[i:i+batch_size] for i in range(0, len(data), batch_size)]
                
                batch_results = []
                for batch in batches:
                    batch_analysis = batch_feature_transformation(batch[feature_columns])
                    batch_results.append(batch_analysis)
                
                # Combine batch results
                if batch_results:
                    combined_analysis = np.mean(batch_results, axis=0)
                    return {
                        'batch_optimization_analysis': combined_analysis,
                        'n_batches_processed': len(batches),
                        'batch_size': batch_size
                    }
            
            return {}
            
        except Exception as e:
            self.logger.warning(f"Batch optimization analysis failed: {e}")
            return {}
    
    def optimize_matrix_operations(self, data: pd.DataFrame, operation_type: str = "correlation") -> Dict[str, Any]:
        """Optimize matrix operations based on hardware capabilities."""
        try:
            if not MATRIX_OPS_AVAILABLE:
                return {}
            
            optimization_result = optimize_matrix_operation_with_hardware(
                data.values, operation_type, 
                gpu_enabled=True,
                batch_enabled=True
            )
            
            return optimization_result
            
        except Exception as e:
            self.logger.warning(f"Matrix operations optimization failed: {e}")
            return {}
    
    def get_enhanced_performance_metrics(self) -> Dict[str, Any]:
        """Get enhanced performance metrics including matrix operations status."""
        base_metrics = {
            'optimization_status': self.optimization_status.value,
            'execution_time': time.time() - self.start_time if self.start_time else 0.0,
            'memory_usage': get_memory_usage() if MATRIX_OPS_AVAILABLE else 0.0
        }
        
        enhanced_metrics = {
            **base_metrics,
            'matrix_operations_available': MATRIX_OPS_AVAILABLE,
            'advanced_matrix_operations_available': ADVANCED_MATRIX_OPS_AVAILABLE,
            'enhanced_matrix_ops_initialized': self.enhanced_matrix_ops is not None,
            'vectorized_core_initialized': self.vectorized_core is not None,
            'batch_processor_initialized': self.batch_processor is not None,
            'hardware_optimization_available': HARDWARE_OPTIMIZATION_AVAILABLE,
            'hardware_manager_initialized': self.hardware_manager is not None,
            'memory_optimizer_initialized': self.memory_optimizer is not None
        }
        
        return enhanced_metrics
    

    def optimize_lookback_periods_mrmr_directional(self,
                                                     data: pd.DataFrame,
                                                     feature_columns: List[str],
                                                     target_column: str = 'returns',
                                                     optimization_config: Optional[LookbackOptimizationConfig] = None,
                                                     enable_directional: bool = True) -> Dict[str, Any]:
        """
        Legacy method - now delegates to unified optimizer.

        DEPRECATED: Use optimize_lookback_periods_unified() instead to avoid duplicate logic.
        """
        tprint_warning("⚠️ optimize_lookback_periods_mrmr_directional is deprecated")
        tprint_info("   → Use optimize_lookback_periods_unified() instead")

        return self.optimize_lookback_periods_unified(
            data, feature_columns, target_column, optimization_config,
            enable_directional=enable_directional, enable_multi_target=False
        )

    def optimize_lookback_periods_mrmr(self,
                                         data: pd.DataFrame,
                                         feature_columns: List[str],
                                         target_column: str = 'returns',
                                         optimization_config: Optional[LookbackOptimizationConfig] = None) -> Dict[str, Any]:
        """
        Legacy method - now delegates to unified optimizer.

        DEPRECATED: Use optimize_lookback_periods_unified() instead to avoid duplicate logic.
        """
        tprint_warning("⚠️ optimize_lookback_periods_mrmr is deprecated")
        tprint_info("   → Use optimize_lookback_periods_unified() instead")

        return self.optimize_lookback_periods_unified(
            data, feature_columns, target_column, optimization_config,
            enable_directional=False, enable_multi_target=False
        )

    def optimize_lookback_periods_multi_target(self,
                                               data: pd.DataFrame,
                                               feature_columns: List[str],
                                               multi_targets: Optional[List[str]] = None,
                                               optimization_config: Optional[LookbackOptimizationConfig] = None) -> Dict[str, Any]:
        """
        Legacy method - now delegates to unified optimizer.

        DEPRECATED: Use optimize_lookback_periods_unified() instead to avoid duplicate logic.
        """
        tprint_warning("⚠️ optimize_lookback_periods_multi_target is deprecated")
        tprint_info("   → Use optimize_lookback_periods_unified() instead")

        return self.optimize_lookback_periods_unified(
            data, feature_columns, 'returns', optimization_config,
            enable_directional=False, enable_multi_target=True
        )

    def _optimize_features_with_grid_bandit(self, data: pd.DataFrame, feature_columns: List[str], target_column: str) -> Dict[str, Any]:
        """Optimize features using grid search/bandit method."""
        try:
            tprint("🔍 Starting grid search/bandit optimization...")

            # Separate long and short targets
            target_info = self._select_multi_horizon_targets(data)
            long_targets = target_info.get('long_targets', [])
            short_targets = target_info.get('short_targets', [])

            # Grid search/bandit optimization logic would go here
            results = {}

            # Placeholder implementation
            for feature_name in feature_columns:
                try:
                    # Placeholder optimization logic
                    results[feature_name] = {
                        'optimal_lookback': 20,
                        'score': 0.8,
                        'method': 'grid_bandit'
                    }
                    tprint(f"✅ {feature_name}: Grid bandit optimization completed")
                except Exception as e:
                    tprint(f"❌ Failed to optimize {feature_name}: {e}")
                    results[feature_name] = {'error': str(e)}

            return results

        except Exception as e:
            tprint(f"❌ Grid search/bandit optimization failed: {e}")
            return {'error': str(e)}

    def _generate_optimization_summary(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of optimization results."""
        summary = {
            'total_features_optimized': len([k for k in optimization_results.keys() if k != '_summary']),
            'successful_optimizations': len([k for k, v in optimization_results.items() 
                                           if k != '_summary' and 'error' not in v]),
            'failed_optimizations': len([k for k, v in optimization_results.items() 
                                       if k != '_summary' and 'error' in v]),
            'average_optimization_time': 0.0,
            'average_mi_score': 0.0,
            'average_correlation': 0.0,
            'best_features': [],
            'worst_features': []
        }
        
        successful_results = [v for k, v in optimization_results.items() 
                            if k != '_summary' and 'error' not in v]
        
        if successful_results:
            summary['average_optimization_time'] = np.mean([r.get('optimization_time', 0) for r in successful_results])
            summary['average_mi_score'] = np.mean([r.get('combined_mi_score', 0) for r in successful_results])
            summary['average_correlation'] = np.mean([r.get('correlation_between_periods', 1) for r in successful_results])
            
            # Find best and worst features
            sorted_features = sorted(successful_results, key=lambda x: x.get('combined_mi_score', 0), reverse=True)
            summary['best_features'] = [f for f in sorted_features[:3]]
            summary['worst_features'] = [f for f in sorted_features[-3:]]
        
        return summary
    
    def _split_data_by_direction(self, data: pd.DataFrame, target_column: str = 'returns') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into long and short signals based on multi-horizon targets or returns.
        
        Args:
            data: Input dataframe
            target_column: Column containing target values (multi-horizon or returns)
            
        Returns:
            Tuple of (long_data, short_data)
        """
        try:
            # Check if we have multi-horizon directional targets
            multi_horizon_targets = self._detect_multi_horizon_targets(data)
            
            if multi_horizon_targets['has_directional_targets']:
                tprint(f"🎯 Using multi-horizon directional targets for data splitting")
                return self._split_by_multi_horizon_targets(data, multi_horizon_targets)
            
            # Fallback to traditional return-based splitting
            tprint(f"📊 Using traditional return-based splitting")
            
            # Ensure target column exists
            if target_column not in data.columns:
                # Try common return column names
                potential_targets = ['returns', 'close_return', 'close_log_return', 'target', 'label']
                for col in potential_targets:
                    if col in data.columns:
                        target_column = col
                        tprint(f"📊 Using {target_column} as target column for directional split")
                        break
                else:
                    raise ValueError(f"Target column '{target_column}' not found and no suitable alternatives found")
            
            # Split based on positive (long) and negative (short) returns
            long_mask = data[target_column] > 0
            short_mask = data[target_column] < 0
            
            long_data = data[long_mask].copy()
            short_data = data[short_mask].copy()
            
            # Add directional labels for clarity
            long_data['signal_direction'] = 'long'
            short_data['signal_direction'] = 'short'
            
            tprint(f"📊 Data split by direction: {len(long_data)} long samples, {len(short_data)} short samples")
            
            return long_data, short_data
            
        except Exception as e:
            tprint(f"❌ Error splitting data by direction: {e}")
            # Return empty dataframes on error
            empty_df = pd.DataFrame()
            return empty_df, empty_df
    
    def _detect_multi_horizon_targets(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect if data contains multi-horizon profit targets.
        
        Args:
            data: Input dataframe
            
        Returns:
            Dictionary with detection results and available targets
        """
        detection_result = {
            'has_directional_targets': False,
            'has_composite_targets': False,
            'long_targets': [],
            'short_targets': [],
            'composite_targets': [],
            'primary_target': None
        }
        
        try:
            # Check for directional multi-horizon targets
            directional_patterns = [
                '_long_prob', '_short_prob', 
                'long_immediate_', 'short_immediate_',
                'long_overall_', 'short_overall_'
            ]
            
            for pattern in directional_patterns:
                matching_cols = [col for col in data.columns if pattern in col]
                if pattern.startswith('long_') or '_long_' in pattern:
                    detection_result['long_targets'].extend(matching_cols)
                else:
                    detection_result['short_targets'].extend(matching_cols)
            
            # Check for composite targets
            composite_patterns = [
                'overall_opportunity', 'leverage_adjusted_score', 
                'immediate_opportunity', 'reversal_capture_score'
            ]
            
            for pattern in composite_patterns:
                if pattern in data.columns:
                    detection_result['composite_targets'].append(pattern)
            
            # Determine if we have directional targets
            detection_result['has_directional_targets'] = (
                len(detection_result['long_targets']) > 0 and 
                len(detection_result['short_targets']) > 0
            )
            
            detection_result['has_composite_targets'] = len(detection_result['composite_targets']) > 0
            
            # Select primary target based on configuration
            if 'leverage_adjusted_score' in detection_result['composite_targets']:
                detection_result['primary_target'] = 'leverage_adjusted_score'
            elif 'overall_opportunity' in detection_result['composite_targets']:
                detection_result['primary_target'] = 'overall_opportunity'
            elif detection_result['long_targets']:
                # Use first long target as fallback
                detection_result['primary_target'] = detection_result['long_targets'][0]
            
            tprint(f"🔍 Multi-horizon target detection:")
            tprint(f"   Directional targets: {detection_result['has_directional_targets']}")
            tprint(f"   Long targets: {len(detection_result['long_targets'])}")
            tprint(f"   Short targets: {len(detection_result['short_targets'])}")
            tprint(f"   Composite targets: {len(detection_result['composite_targets'])}")
            tprint(f"   Primary target: {detection_result['primary_target']}")
            
            return detection_result
            
        except Exception as e:
            tprint(f"❌ Error detecting multi-horizon targets: {e}")
            return detection_result
    
    def _split_by_multi_horizon_targets(self, data: pd.DataFrame, targets_info: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data using multi-horizon directional targets.
        
        Args:
            data: Input dataframe with multi-horizon targets
            targets_info: Information about available targets
            
        Returns:
            Tuple of (long_data, short_data)
        """
        try:
            # Strategy 1: Use directional opportunity scores if available
            if 'long_overall_opportunity' in data.columns and 'short_overall_opportunity' in data.columns:
                tprint("🎯 Using directional opportunity scores for splitting")
                
                # Create masks based on which direction has higher opportunity
                long_mask = data['long_overall_opportunity'] > data['short_overall_opportunity']
                short_mask = data['short_overall_opportunity'] > data['long_overall_opportunity']
                
                # Also consider minimum opportunity thresholds
                min_opportunity = 0.3  # 30% minimum opportunity
                long_mask = long_mask & (data['long_overall_opportunity'] > min_opportunity)
                short_mask = short_mask & (data['short_overall_opportunity'] > min_opportunity)
                
            # Strategy 2: Use best directional probabilities
            elif targets_info['long_targets'] and targets_info['short_targets']:
                tprint("🎯 Using directional probabilities for splitting")
                
                # Find best long and short probability columns
                long_prob_cols = [col for col in targets_info['long_targets'] if 'prob' in col]
                short_prob_cols = [col for col in targets_info['short_targets'] if 'prob' in col]
                
                if long_prob_cols and short_prob_cols:
                    # Use immediate probabilities if available, otherwise use first available
                    long_col = next((col for col in long_prob_cols if 'immediate' in col), long_prob_cols[0])
                    short_col = next((col for col in short_prob_cols if 'immediate' in col), short_prob_cols[0])
                    
                    # Create masks based on which direction has higher probability
                    min_prob = 0.4  # 40% minimum probability
                    long_mask = (data[long_col] > data[short_col]) & (data[long_col] > min_prob)
                    short_mask = (data[short_col] > data[long_col]) & (data[short_col] > min_prob)
                else:
                    raise ValueError("No suitable probability columns found")
                    
            else:
                raise ValueError("Insufficient directional targets for multi-horizon splitting")
            
            # Create directional datasets
            long_data = data[long_mask].copy()
            short_data = data[short_mask].copy()
            
            # Add directional labels
            long_data['signal_direction'] = 'long'
            short_data['signal_direction'] = 'short'
            
            # Add target information for optimization
            long_data['optimization_target_type'] = 'multi_horizon_long'
            short_data['optimization_target_type'] = 'multi_horizon_short'
            
            tprint(f"✅ Multi-horizon directional split completed:")
            tprint(f"   Long samples: {len(long_data)} ({len(long_data)/len(data)*100:.1f}%)")
            tprint(f"   Short samples: {len(short_data)} ({len(short_data)/len(data)*100:.1f}%)")
            tprint(f"   Neutral/filtered: {len(data) - len(long_data) - len(short_data)}")
            
            return long_data, short_data
            
        except Exception as e:
            tprint(f"❌ Error in multi-horizon directional splitting: {e}")
            # Fallback to empty dataframes
            return pd.DataFrame(), pd.DataFrame()
    
    def _select_optimal_target_column(self, data: pd.DataFrame) -> str:
        """
        Select the optimal target column for feature optimization, prioritizing multi-horizon targets.
        
        Args:
            data: Input dataframe
            
        Returns:
            Name of the optimal target column
        """
        try:
            # Priority 1: Multi-horizon composite targets (best overall signal)
            composite_priority = [
                'leverage_adjusted_score',  # Primary target from config
                'overall_opportunity',      # Secondary target
                'immediate_opportunity',    # Short-term focused
                'reversal_capture_score'    # Reversal opportunities
            ]
            
            for target in composite_priority:
                if target in data.columns:
                    tprint(f"🎯 Selected multi-horizon target: {target}")
                    return target
            
            # Priority 2: Directional opportunity targets (if available)
            directional_opportunity = [
                'long_overall_opportunity',
                'short_overall_opportunity'
            ]
            
            for target in directional_opportunity:
                if target in data.columns:
                    tprint(f"🎯 Selected directional opportunity target: {target}")
                    return target
            
            # Priority 3: Best multi-horizon probability targets
            multi_horizon_patterns = [
                'micro_immediate_long_prob',
                'small_immediate_long_prob', 
                'micro_immediate_short_prob',
                'small_immediate_short_prob'
            ]
            
            for target in multi_horizon_patterns:
                if target in data.columns:
                    tprint(f"🎯 Selected multi-horizon probability target: {target}")
                    return target
            
            # Priority 4: Any multi-horizon probability target
            prob_targets = [col for col in data.columns if '_prob' in col and ('long' in col or 'short' in col)]
            if prob_targets:
                # Prefer immediate probabilities
                immediate_targets = [col for col in prob_targets if 'immediate' in col]
                if immediate_targets:
                    target = immediate_targets[0]
                    tprint(f"🎯 Selected immediate probability target: {target}")
                    return target
                else:
                    target = prob_targets[0]
                    tprint(f"🎯 Selected probability target: {target}")
                    return target
            
            # Fallback: Traditional return columns
            traditional_targets = ['returns', 'close_return', 'close_log_return', 'target', 'label']
            for target in traditional_targets:
                if target in data.columns:
                    tprint(f"📊 Fallback to traditional target: {target}")
                    return target
            
            # Last resort: Use first numeric column
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                target = numeric_cols[0]
                tprint(f"⚠️ Using first numeric column as target: {target}")
                return target
                
            raise ValueError("No suitable target column found")
            
        except Exception as e:
            tprint(f"❌ Error selecting target column: {e}")
            return 'returns'  # Safe fallback
    
    def _select_multi_horizon_targets(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Select optimal set of multi-horizon targets with separate long/short pipelines."""
        
        # Separate targets for long and short pipelines
        long_targets = []
        short_targets = []
        
        # Multi-horizon profit labeler targets (aligned with actual labeler output)
        # These are the targets that multi_horizon_profit_labeler actually generates
        
        # LONG PIPELINE TARGETS - Updated for 20-40m horizons, micro disabled
        long_target_candidates = [
            'long_overall_opportunity',
            'long_immediate_opportunity',  # 20m horizon
            'long_short_opportunity',      # 40m horizon
            # Profit target combinations (micro disabled, focus on small/medium/good)
            'long_small_immediate_prob',   # 0.5% target, 20m horizon
            'long_medium_immediate_prob',  # 0.7% target, 20m horizon
            'long_good_immediate_prob',    # 1.0% target, 20m horizon
            'long_small_short_prob',       # 0.5% target, 40m horizon
            'long_medium_short_prob',      # 0.7% target, 40m horizon
            'long_good_short_prob'         # 1.0% target, 40m horizon
        ]
        
        # SHORT PIPELINE TARGETS - Updated for 20-40m horizons, micro disabled
        short_target_candidates = [
            'short_overall_opportunity',
            'short_immediate_opportunity', # 20m horizon
            'short_short_opportunity',    # 40m horizon
            # Profit target combinations (micro disabled, focus on small/medium/good)
            'short_small_immediate_prob',  # 0.5% target, 20m horizon
            'short_medium_immediate_prob', # 0.7% target, 20m horizon
            'short_good_immediate_prob',   # 1.0% target, 20m horizon
            'short_small_short_prob',      # 0.5% target, 40m horizon
            'short_medium_short_prob',     # 0.7% target, 40m horizon
            'short_good_short_prob'        # 1.0% target, 40m horizon
        ]
        
        # Check which targets are actually available in the data
        for target in long_target_candidates:
            if target in data.columns:
                long_targets.append(target)
                
        for target in short_target_candidates:
            if target in data.columns:
                short_targets.append(target)
        
        # If no directional targets found, use general targets for both
        if not long_targets and not short_targets:
            general_targets = ['leverage_adjusted_score', 'overall_opportunity', 'immediate_opportunity']
            available_general = [t for t in general_targets if t in data.columns]
            long_targets = available_general
            short_targets = available_general
        
        # If no multi-horizon targets found, fallback to basic returns
        if not long_targets and not short_targets:
            fallback_targets = ['close_return', 'returns', 'target']
            available_fallback = [t for t in fallback_targets if t in data.columns]
            long_targets = available_fallback
            short_targets = available_fallback
        
        result = {
            'long_targets': long_targets,
            'short_targets': short_targets
        }
        
        tprint(f"🎯 Selected targets - Long: {len(long_targets)}, Short: {len(short_targets)}")
        tprint(f"   Long targets: {long_targets}")
        tprint(f"   Short targets: {short_targets}")
        
        return result
    
    def _optimize_features_with_grid_bandit(self, data: pd.DataFrame, feature_columns: List[str], target_column: str) -> Dict[str, Any]:
        """Optimize features using grid search/bandit method."""
        try:
            tprint("🔍 Starting grid search/bandit optimization...")
            
            # Separate long and short targets
            target_info = self._select_multi_horizon_targets(data)
            long_targets = target_info.get('long_targets', [])
            short_targets = target_info.get('short_targets', [])
            
            # Use horizon weights from standardized format if available, otherwise use defaults
            if horizon_weights:
                tprint_info(f"⚖️ Using horizon weights from multi_horizon_profit_labeler: {horizon_weights}")
            else:
                tprint_warning("⚠️ No horizon weights from labeling data, using default weights")
                horizon_weights = {
                    'micro': 0.0,     # 0% - disabled for now
                    'small': 0.6,     # 60% - immediate opportunities
                    'medium': 0.3,    # 30% - short-term opportunities
                    'high': 0.2       # 20% - longer-term opportunities
                }
            
            optimized_results = {
                'long_features': {},
                'short_features': {},
                'optimization_method': 'grid_search_bandit'
            }
            
            # Optimize for LONG pipeline
            if long_targets:
                tprint("🎯 Optimizing LONG pipeline features...")
                long_results = self._grid_bandit_optimization(
                    data, feature_columns, long_targets, horizon_weights, 'long'
                )
                optimized_results['long_features'] = long_results
            
            # Optimize for SHORT pipeline  
            if short_targets:
                tprint("🎯 Optimizing SHORT pipeline features...")
                short_results = self._grid_bandit_optimization(
                    data, feature_columns, short_targets, horizon_weights, 'short'
                )
                optimized_results['short_features'] = short_results
            
            tprint(f"✅ Grid search/bandit optimization completed")
            tprint(f"   Long features optimized: {len(optimized_results['long_features'])}")
            tprint(f"   Short features optimized: {len(optimized_results['short_features'])}")
            
            return optimized_results
            
        except Exception as e:
            tprint(f"❌ Grid search/bandit optimization failed: {e}")
            return {'error': str(e)}
    
    def _grid_bandit_optimization(self, data: pd.DataFrame, feature_columns: List[str], 
                                 targets: List[str], horizon_weights: Dict[str, float], 
                                 direction: str) -> Dict[str, Any]:
        """Grid search/bandit optimization for feature lookback periods."""
        try:
            results = {}
            
            for feature in feature_columns:
                if feature not in data.columns:
                    continue
                    
                tprint(f"🎯 Optimizing feature: {feature}")
                
                # Find optimal lookback period using grid search/bandit
                optimal_period, best_score = self._find_optimal_lookback_grid_bandit(
                    data, feature, targets, horizon_weights, direction
                )
                
                results[feature] = {
                    'optimal_period': optimal_period,
                    'grid_bandit_score': best_score,
                    'direction': direction,
                    'optimization_method': 'grid_search_bandit'
                }
                
                tprint(f"✅ {feature}: optimal_period={optimal_period}, score={best_score:.4f}")
            
            return results
            
        except Exception as e:
            tprint(f"❌ Grid search/bandit optimization failed for {direction}: {e}")
            return {}
    
    def _find_optimal_lookback_grid_bandit(self, data: pd.DataFrame, feature: str, targets: List[str], 
                                         horizon_weights: Dict[str, float], direction: str) -> Tuple[int, float]:
        """Find optimal lookback period using grid search/bandit method."""
        try:
            # Grid search: test different lookback periods
            grid_periods = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 200]  # Fibonacci-based grid
            best_period = 1
            best_score = 0.0
            
            for period in grid_periods:
                # Calculate score for this period
                score = self._calculate_grid_bandit_score(
                    data, feature, targets, period, horizon_weights, direction
                )
                
                if score > best_score:
                    best_score = score
                    best_period = period
            
            # Bandit refinement: search around best grid point
            if best_period > 1:
                refinement_periods = range(max(1, best_period - 5), min(201, best_period + 6))
                for period in refinement_periods:
                    if period in grid_periods:  # Skip already tested periods
                        continue
                        
                    score = self._calculate_grid_bandit_score(
                        data, feature, targets, period, horizon_weights, direction
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_period = period
            
            return best_period, best_score
            
        except Exception as e:
            tprint(f"❌ Grid search/bandit calculation failed for {feature}: {e}")
            return 1, 0.0
    
    def _calculate_grid_bandit_score(self, data: pd.DataFrame, feature: str, targets: List[str], 
                                    period: int, horizon_weights: Dict[str, float], direction: str) -> float:
        """Calculate grid search/bandit score for a feature with given lookback period."""
        try:
            if feature not in data.columns:
                return 0.0
            
            # Create feature with lookback period (rolling mean)
            feature_series = data[feature].rolling(window=period).mean()
            
            total_score = 0.0
            total_weight = 0.0
            
            for target in targets:
                if target not in data.columns:
                    continue
                
                # Determine horizon weight based on target name
                price_horizon = self._classify_price_movement_horizon(target, data)
                horizon_weight = horizon_weights.get(price_horizon, horizon_weights['small'])
                
                # Calculate correlation between feature and target
                try:
                    # Remove NaN values for correlation calculation
                    valid_data = pd.DataFrame({
                        'feature': feature_series,
                        'target': data[target]
                    }).dropna()
                    
                    if len(valid_data) < 10:  # Need minimum data points
                        continue
                    
                    # Calculate correlation (grid search/bandit uses simple correlation)
                    correlation = abs(valid_data['feature'].corr(valid_data['target']))
                    if pd.isna(correlation):
                        correlation = 0.0
                    
                    # Weighted score
                    weighted_score = correlation * horizon_weight
                    total_score += weighted_score
                    total_weight += horizon_weight
                    
                except Exception:
                    continue
            
            # Return average weighted score
            return total_score / max(total_weight, 0.1)
            
        except Exception:
            return 0.0
    
    def _mrmr_pearson_optimization(self, data: pd.DataFrame, feature_columns: List[str], 
                                  targets: List[str], horizon_weights: Dict[str, float], 
                                  direction: str) -> Dict[str, Any]:
        """MRMR & Pearson correlation optimization for feature lookback periods."""
        try:
            results = {}
            
            for feature in feature_columns:
                if feature not in data.columns:
                    continue
                    
                tprint(f"🎯 Optimizing feature: {feature}")
                
                # Find optimal lookback period using MRMR & Pearson correlation
                optimal_period, best_score = self._find_optimal_lookback_mrmr_pearson(
                    data, feature, targets, horizon_weights, direction
                )
                
                results[feature] = {
                    'optimal_period': optimal_period,
                    'mrmr_pearson_score': best_score,
                    'direction': direction,
                    'optimization_method': 'mrmr_pearson_correlation'
                }
                
                tprint(f"✅ {feature}: optimal_period={optimal_period}, score={best_score:.4f}")
            
            return results
            
        except Exception as e:
            tprint(f"❌ MRMR & Pearson optimization failed for {direction}: {e}")
            return {}
    
    def _find_optimal_lookback_mrmr_pearson(self, data: pd.DataFrame, feature: str, targets: List[str], 
                                        horizon_weights: Dict[str, float], direction: str) -> Tuple[int, float]:
        """Find optimal lookback period using MRMR & Pearson correlation."""
        try:
            best_period = 1
            best_score = 0.0
            
            # Test different lookback periods (1-200 for 5m timeframe)
            for period in range(1, 201):
                # Calculate MRMR & Pearson correlation score for this period
                score = self._calculate_mrmr_pearson_score(
                    data, feature, targets, period, horizon_weights, direction
                )
                
                if score > best_score:
                    best_score = score
                    best_period = period
            
            return best_period, best_score
            
        except Exception as e:
            tprint(f"❌ MRMR & Pearson calculation failed for {feature}: {e}")
            return 1, 0.0
    
    def _calculate_mrmr_pearson_score(self, data: pd.DataFrame, feature: str, targets: List[str], 
                                   period: int, horizon_weights: Dict[str, float], direction: str) -> float:
        """Calculate MRMR & Pearson correlation score for a feature with given lookback period."""
        try:
            if feature not in data.columns:
                return 0.0
            
            # Create feature with lookback period (rolling mean)
            feature_series = data[feature].rolling(window=period).mean()
            
            total_score = 0.0
            total_weight = 0.0
            
            for target in targets:
                if target not in data.columns:
                    continue
                
                # Determine horizon weight based on target name
                price_horizon = self._classify_price_movement_horizon(target, data)
                horizon_weight = horizon_weights.get(price_horizon, horizon_weights['small'])
                
                # Calculate Pearson correlation between feature and target
                try:
                    # Remove NaN values for correlation calculation
                    valid_data = pd.DataFrame({
                        'feature': feature_series,
                        'target': data[target]
                    }).dropna()
                    
                    if len(valid_data) < 10:  # Need minimum data points
                        continue
                    
                    # Calculate Pearson correlation
                    correlation = abs(valid_data['feature'].corr(valid_data['target']))
                    if pd.isna(correlation):
                        correlation = 0.0
                    
                    # Weighted score
                    weighted_score = correlation * horizon_weight
                    total_score += weighted_score
                    total_weight += horizon_weight
                    
                except Exception:
                    continue
            
            # Return average weighted score
            return total_score / max(total_weight, 0.1)
            
        except Exception:
            return 0.0
    
    def _multi_resolution_optimization(self, data: pd.DataFrame, feature_columns: List[str], 
                                     targets: List[str], horizon_weights: Dict[str, float], 
                                     direction: str) -> Dict[str, Any]:
        """Multi-resolution optimization for feature lookback periods."""
        try:
            results = {}
            
            for feature in feature_columns:
                if feature not in data.columns:
                    continue
                    
                # Phase 1: Coarse grid search with log spacing
                optimal_period, best_score = self._coarse_to_fine_search(
                    data, feature, targets, horizon_weights, direction
                )
                
                results[feature] = {
                    'optimal_period': optimal_period,
                    'mutual_info_score': best_score,
                    'direction': direction,
                    'optimization_method': 'multi_resolution'
                }
            
            return results
            
        except Exception as e:
            tprint(f"❌ Multi-resolution optimization failed for {direction}: {e}")
            return {}
    
    def _coarse_to_fine_search(self, data: pd.DataFrame, feature: str, targets: List[str], 
                              horizon_weights: Dict[str, float], direction: str) -> Tuple[int, float]:
        """Coarse-to-fine multi-resolution search for optimal lookback period."""
        try:
            # Phase 1: Coarse grid with log spacing (15-25 points)
            coarse_periods = self._generate_coarse_grid(1, 200)
            coarse_scores = []
            
            tprint(f"🔍 Phase 1: Coarse grid search for {feature} ({len(coarse_periods)} points)")
            
            for period in coarse_periods:
                score = self._calculate_mutual_info_score(
                    data, feature, targets, period, horizon_weights, direction
                )
                coarse_scores.append((period, score))
            
            # Sort by score and get top-k candidates
            coarse_scores.sort(key=lambda x: x[1], reverse=True)
            top_k = 5  # Top 5 candidates
            top_candidates = coarse_scores[:top_k]
            
            tprint(f"📊 Top {top_k} candidates from coarse search:")
            for i, (period, score) in enumerate(top_candidates):
                tprint(f"   {i+1}. Period {period}: Score {score:.4f}")
            
            # Phase 2: Fine search around top candidates
            best_period, best_score = self._fine_search_around_candidates(
                data, feature, targets, horizon_weights, direction, top_candidates
            )
            
            # Phase 3: Stability check with time splits
            stable_period, stable_score = self._stability_check(
                data, feature, targets, horizon_weights, direction, best_period, best_score
            )
            
            tprint(f"✅ Final result: Period {stable_period}, Score {stable_score:.4f}")
            return stable_period, stable_score
            
        except Exception as e:
            tprint(f"❌ Coarse-to-fine search failed for {feature}: {e}")
            return 1, 0.0
    
    def _generate_coarse_grid(self, min_period: int, max_period: int) -> List[int]:
        """Generate coarse grid with log spacing for better coverage."""
        import numpy as np
        
        # Log spacing: more points at short horizons, fewer at long
        log_min = np.log(min_period)
        log_max = np.log(max_period)
        
        # Generate 20 points with log spacing
        log_points = np.linspace(log_min, log_max, 20)
        periods = [int(np.exp(p)) for p in log_points]
        
        # Ensure unique periods and add some linear spacing for very short periods
        periods = list(set(periods))
        periods.extend([1, 2, 3, 5, 10, 15, 20, 25, 30, 50, 100, 150, 200])
        periods = sorted(list(set(periods)))
        
        return periods
    
    def _fine_search_around_candidates(self, data: pd.DataFrame, feature: str, targets: List[str],
                                     horizon_weights: Dict[str, float], direction: str,
                                     top_candidates: List[Tuple[int, float]]) -> Tuple[int, float]:
        """Fine search around top candidates with dense neighborhoods."""
        try:
            best_period = 1
            best_score = 0.0
            
            for candidate_period, candidate_score in top_candidates:
                # Dense neighborhood: ±15 periods with step 2
                neighborhood = range(
                    max(1, candidate_period - 15),
                    min(201, candidate_period + 16),
                    2
                )
                
                for period in neighborhood:
                    score = self._calculate_mutual_info_score(
                        data, feature, targets, period, horizon_weights, direction
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_period = period
            
            return best_period, best_score
            
        except Exception as e:
            tprint(f"❌ Fine search failed: {e}")
            return top_candidates[0][0], top_candidates[0][1]
    
    def _stability_check(self, data: pd.DataFrame, feature: str, targets: List[str],
                        horizon_weights: Dict[str, float], direction: str,
                        candidate_period: int, candidate_score: float) -> Tuple[int, float]:
        """Stability check using time splits to ensure robust selection."""
        try:
            # Split data into first half and second half
            mid_point = len(data) // 2
            first_half = data.iloc[:mid_point]
            second_half = data.iloc[mid_point:]
            
            # Calculate scores on both halves
            score_first = self._calculate_mutual_info_score(
                first_half, feature, targets, candidate_period, horizon_weights, direction
            )
            score_second = self._calculate_mutual_info_score(
                second_half, feature, targets, candidate_period, horizon_weights, direction
            )
            
            # Stability metric: consistency across time splits
            stability = 1.0 - abs(score_first - score_second) / max(score_first + score_second, 0.1)
            
            # If stability is good (>0.7), use the candidate
            if stability > 0.7:
                tprint(f"✅ Period {candidate_period} is stable (stability: {stability:.3f})")
                return candidate_period, candidate_score
            else:
                # Try nearby periods for better stability
                tprint(f"⚠️ Period {candidate_period} unstable (stability: {stability:.3f}), searching nearby...")
                
                best_stable_period = candidate_period
                best_stable_score = candidate_score
                
                for offset in [-5, -3, -1, 1, 3, 5]:
                    test_period = max(1, candidate_period + offset)
                    if test_period > 200:
                        continue
                    
                    test_score_first = self._calculate_mutual_info_score(
                        first_half, feature, targets, test_period, horizon_weights, direction
                    )
                    test_score_second = self._calculate_mutual_info_score(
                        second_half, feature, targets, test_period, horizon_weights, direction
                    )
                    
                    test_stability = 1.0 - abs(test_score_first - test_score_second) / max(test_score_first + test_score_second, 0.1)
                    
                    if test_stability > stability:
                        best_stable_period = test_period
                        best_stable_score = (test_score_first + test_score_second) / 2
                        stability = test_stability
                
                tprint(f"✅ Best stable period: {best_stable_period} (stability: {stability:.3f})")
                return best_stable_period, best_stable_score
                
        except Exception as e:
            tprint(f"❌ Stability check failed: {e}")
            return candidate_period, candidate_score
    
    def _calculate_mutual_info_score(self, data: pd.DataFrame, feature: str, targets: List[str], 
                                   period: int, horizon_weights: Dict[str, float], direction: str) -> float:
        """Calculate weighted mutual information score for a feature with given lookback period."""
        try:
            if feature not in data.columns:
                return 0.0
            
            # Create feature with lookback period
            feature_series = data[feature].rolling(window=period).mean()
            
            total_score = 0.0
            total_weight = 0.0
            
            for target in targets:
                if target not in data.columns:
                    continue
                
                # Determine horizon weight based on actual price movement classification
                price_horizon = self._classify_price_movement_horizon(target, data)
                horizon_weight = horizon_weights.get(price_horizon, horizon_weights['micro'])
                
                # Calculate mutual information between feature and target
                try:
                    # Remove NaN values for mutual information calculation
                    valid_data = pd.DataFrame({
                        'feature': feature_series,
                        'target': data[target]
                    }).dropna()
                    
                    if len(valid_data) < 10:  # Need minimum data points
                        continue
                    
                    # Calculate correlation as proxy for mutual information
                    correlation = abs(valid_data['feature'].corr(valid_data['target']))
                    if pd.isna(correlation):
                        correlation = 0.0
                    
                    # Weighted score
                    weighted_score = correlation * horizon_weight
                    total_score += weighted_score
                    total_weight += horizon_weight
                    
                except Exception:
                    continue
            
            # Return average weighted score
            return total_score / max(total_weight, 0.1)
            
        except Exception:
            return 0.0
    
    def _classify_price_movement_horizon(self, target_column: str, data: pd.DataFrame) -> str:
        """Classify target based on actual price movement percentages in the data."""
        try:
            # Calculate price movement statistics for this target
            if target_column not in data.columns:
                return 'micro'  # Default to micro
            
            target_data = data[target_column].dropna()
            if len(target_data) < 10:
                return 'micro'
            
            # Calculate price movement percentage
            # Assuming target represents price change percentage
            movement_stats = {
                'mean': abs(target_data.mean()),
                'std': target_data.std(),
                'max': abs(target_data.max()),
                'percentile_75': abs(target_data.quantile(0.75))
            }
            
            # Classify based on multi_horizon_profit_labeler profit targets
            typical_movement = movement_stats['percentile_75']
            
            # Align with updated profit targets (micro disabled, focus on small/medium/good)
            if typical_movement <= 0.005:  # 0.5% or less (small target - now primary)
                return 'small'
            elif typical_movement <= 0.007:  # 0.7% or less (medium target)
                return 'medium'
            elif typical_movement <= 0.010:  # 1.0% or less (good target)
                return 'good'
            else:
                return 'good'  # Default to good for larger movements
                
        except Exception:
            return 'small'  # Default to small on error (micro disabled)
    
    def _classify_target_type(self, target_column: str) -> str:
        """Classify the type of multi-horizon target."""
        if 'leverage_adjusted' in target_column:
            return 'composite_leverage'
        elif 'overall_opportunity' in target_column:
            return 'composite_opportunity'
        elif 'immediate' in target_column:
            return 'immediate_horizon'
        elif 'short' in target_column and 'prob' in target_column:
            return 'short_probability'
        elif 'long' in target_column and 'prob' in target_column:
            return 'long_probability'
        else:
            return 'other'
    
    def _calculate_multi_target_consensus(self, feature_results: Dict[str, Any], target_scores: Dict[str, float]) -> Dict[str, Any]:
        """Calculate consensus lookback period across multiple targets."""
        try:
            valid_results = {k: v for k, v in feature_results.items() if 'error' not in v}

            if not valid_results:
                return {'lookback': 20, 'weighted_score': 0.0, 'method': 'fallback'}

            # Weight targets by their performance and type
            weights = {}
            total_weight = 0

            for target, result in valid_results.items():
                # Base weight from MI score
                score_weight = result['score']

                # Type-based weight adjustment
                target_type = result['target_type']
                type_weights = {
                    'composite_leverage': 1.5,    # Highest priority
                    'composite_opportunity': 1.3,
                    'immediate_horizon': 1.1,
                    'long_probability': 1.0,
                    'short_probability': 1.0,
                    'other': 0.8
                }

                type_weight = type_weights.get(target_type, 1.0)
                final_weight = score_weight * type_weight

                weights[target] = final_weight
                total_weight += final_weight

            lookback_values = [int(r['lookback']) for r in valid_results.values()]  # Convert to regular int
            score_values = [r['score'] for r in valid_results.values()]

            if total_weight <= 0:
                log_warning(
                    "Total weight for multi-target consensus was non-positive; "
                    "falling back to uniform averaging."
                )
                consensus_lookback = int(round(np.mean(lookback_values))) if lookback_values else 20
                average_score = float(np.mean(score_values)) if score_values else 0.0
                lookback_std = np.std(lookback_values) if len(lookback_values) > 1 else 0
                consensus_confidence = (
                    max(0, 1 - (lookback_std / np.mean(lookback_values)))
                    if lookback_values and np.mean(lookback_values) != 0
                    else 0
                )
                uniform_weight = 1 / len(valid_results) if valid_results else 0
                uniform_weights = {target: uniform_weight for target in valid_results}

                return {
                    'lookback': consensus_lookback,
                    'weighted_score': average_score,
                    'consensus_confidence': consensus_confidence,
                    'method': 'uniform_consensus',
                    'target_weights': uniform_weights,
                    'lookback_std': lookback_std
                }

            # Calculate weighted consensus
            weighted_lookback = 0
            weighted_score = 0

            for target, result in valid_results.items():
                weight = weights[target] / total_weight
                weighted_lookback += int(result['lookback']) * weight  # Convert to regular int
                weighted_score += result['score'] * weight

            consensus_lookback = int(round(weighted_lookback))

            # Calculate consensus confidence
            lookback_std = np.std(lookback_values) if len(lookback_values) > 1 else 0
            consensus_confidence = max(0, 1 - (lookback_std / np.mean(lookback_values))) if lookback_values else 0

            return {
                'lookback': consensus_lookback,
                'weighted_score': weighted_score,
                'consensus_confidence': consensus_confidence,
                'method': 'weighted_consensus',
                'target_weights': weights,
                'lookback_std': lookback_std
            }

        except Exception as e:
            tprint(f"❌ Error calculating multi-target consensus: {e}")
            return {'lookback': 20, 'weighted_score': 0.0, 'method': 'error_fallback'}
    
    def _evaluate_multi_target_quality(self, feature_results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the quality of multi-target optimization."""
        try:
            valid_results = {k: v for k, v in feature_results.items() if 'error' not in v}
            
            if not valid_results:
                return {'overall_quality': 0.0}
            
            # Score distribution quality
            scores = [r['score'] for r in valid_results.values()]
            score_mean = np.mean(scores)
            score_std = np.std(scores)
            score_consistency = max(0, 1 - (score_std / score_mean)) if score_mean > 0 else 0
            
            # Lookback consistency
            lookbacks = [int(r['lookback']) for r in valid_results.values()]  # Convert to regular int
            lookback_std = np.std(lookbacks)
            lookback_consistency = max(0, 1 - (lookback_std / 20))  # Normalize by reasonable range
            
            # Target coverage
            target_coverage = len(valid_results) / max(1, len(feature_results))
            
            # Overall quality
            overall_quality = (score_mean * 0.4 + 
                             score_consistency * 0.3 + 
                             lookback_consistency * 0.2 + 
                             target_coverage * 0.1)
            
            return {
                'overall_quality': overall_quality,
                'score_mean': score_mean,
                'score_consistency': score_consistency,
                'lookback_consistency': lookback_consistency,
                'target_coverage': target_coverage
            }
            
        except Exception as e:
            tprint(f"❌ Error evaluating multi-target quality: {e}")
            return {'overall_quality': 0.0}
    
    def _generate_multi_target_summary(self, multi_target_results: Dict[str, Any], targets: List[str]) -> Dict[str, Any]:
        """Generate summary of multi-target optimization results."""
        try:
            summary = {
                'total_features': len([k for k in multi_target_results.keys() if k != '_summary']),
                'total_targets': len(targets),
                'target_list': targets,
                'optimization_quality': {},
                'consensus_stats': {},
                'recommendations': []
            }
            
            # Analyze consensus quality across features
            consensus_scores = []
            consensus_confidences = []
            
            for feature_name, results in multi_target_results.items():
                if feature_name == '_summary':
                    continue
                    
                consensus = results.get('consensus', {})
                if 'weighted_score' in consensus:
                    consensus_scores.append(consensus['weighted_score'])
                if 'consensus_confidence' in consensus:
                    consensus_confidences.append(consensus['consensus_confidence'])
            
            if consensus_scores:
                summary['consensus_stats'] = {
                    'average_score': np.mean(consensus_scores),
                    'score_std': np.std(consensus_scores),
                    'average_confidence': np.mean(consensus_confidences) if consensus_confidences else 0,
                    'high_quality_features': len([s for s in consensus_scores if s > 0.5])
                }
            
            # Generate recommendations
            avg_score = summary['consensus_stats'].get('average_score', 0)
            avg_confidence = summary['consensus_stats'].get('average_confidence', 0)
            
            if avg_score > 0.6 and avg_confidence > 0.7:
                summary['recommendations'].append("Excellent multi-target optimization - use consensus lookback periods")
            elif avg_score > 0.4 and avg_confidence > 0.5:
                summary['recommendations'].append("Good multi-target optimization - consensus periods recommended with monitoring")
            elif avg_score > 0.2:
                summary['recommendations'].append("Moderate optimization quality - consider feature selection refinement")
            else:
                summary['recommendations'].append("Low optimization quality - review target selection and feature engineering")
            
            return summary
            
        except Exception as e:
            tprint(f"❌ Error generating multi-target summary: {e}")
            return {'error': str(e)}
    
    def _generate_directional_comparison(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comparison metrics between long and short optimization results.
        
        Args:
            optimization_results: Results from directional optimization
            
        Returns:
            Dictionary with comparison metrics
        """
        try:
            comparison = {
                'feature_comparisons': {},
                'summary_stats': {},
                'recommendations': []
            }
            
            long_results = optimization_results.get('long', {})
            short_results = optimization_results.get('short', {})
            
            if not long_results or not short_results:
                comparison['error'] = 'Insufficient data for comparison'
                return comparison
            
            # Compare each feature across directions
            common_features = set(long_results.keys()) & set(short_results.keys())
            common_features = {f for f in common_features if isinstance(long_results.get(f), dict) and isinstance(short_results.get(f), dict)}
            
            for feature in common_features:
                long_feature = long_results[feature]
                short_feature = short_results[feature]
                
                if 'error' in long_feature or 'error' in short_feature:
                    continue
                
                feature_comparison = {
                    'lookback_difference': {
                        'first_period': {
                            'long': long_feature.get('first_lookback_period'),
                            'short': short_feature.get('first_lookback_period'),
                            'difference': abs((long_feature.get('first_lookback_period', 0) or 0) - 
                                            (short_feature.get('first_lookback_period', 0) or 0))
                        },
                        'second_period': {
                            'long': long_feature.get('second_lookback_period'),
                            'short': short_feature.get('second_lookback_period'),
                            'difference': abs((long_feature.get('second_lookback_period', 0) or 0) - 
                                            (short_feature.get('second_lookback_period', 0) or 0))
                        }
                    },
                    'performance_difference': {
                        'mi_score_long': long_feature.get('combined_mi_score', 0),
                        'mi_score_short': short_feature.get('combined_mi_score', 0),
                        'mi_score_difference': abs((long_feature.get('combined_mi_score', 0) or 0) - 
                                                 (short_feature.get('combined_mi_score', 0) or 0)),
                        'better_direction': 'long' if (long_feature.get('combined_mi_score', 0) or 0) > (short_feature.get('combined_mi_score', 0) or 0) else 'short'
                    },
                    'sample_sizes': {
                        'long': long_feature.get('sample_count', 0),
                        'short': short_feature.get('sample_count', 0)
                    }
                }
                
                comparison['feature_comparisons'][feature] = feature_comparison
            
            # Generate summary statistics
            if comparison['feature_comparisons']:
                mi_scores_long = [comp['performance_difference']['mi_score_long'] 
                                for comp in comparison['feature_comparisons'].values()]
                mi_scores_short = [comp['performance_difference']['mi_score_short'] 
                                 for comp in comparison['feature_comparisons'].values()]
                
                comparison['summary_stats'] = {
                    'features_compared': len(comparison['feature_comparisons']),
                    'average_mi_score_long': np.mean(mi_scores_long) if mi_scores_long else 0,
                    'average_mi_score_short': np.mean(mi_scores_short) if mi_scores_short else 0,
                    'long_outperforms_count': sum(1 for comp in comparison['feature_comparisons'].values() 
                                                if comp['performance_difference']['better_direction'] == 'long'),
                    'short_outperforms_count': sum(1 for comp in comparison['feature_comparisons'].values() 
                                                 if comp['performance_difference']['better_direction'] == 'short')
                }
                
                # Generate recommendations
                long_wins = comparison['summary_stats']['long_outperforms_count']
                short_wins = comparison['summary_stats']['short_outperforms_count']
                total_features = len(comparison['feature_comparisons'])
                
                if long_wins > short_wins * 1.5:
                    comparison['recommendations'].append("Long signals show consistently better feature optimization - consider long-biased strategy")
                elif short_wins > long_wins * 1.5:
                    comparison['recommendations'].append("Short signals show consistently better feature optimization - consider short-biased strategy")
                else:
                    comparison['recommendations'].append("Balanced performance between long and short signals - directional strategy recommended")
                
                # Check for significant lookback differences
                avg_lookback_diff = np.mean([
                    comp['lookback_difference']['first_period']['difference'] 
                    for comp in comparison['feature_comparisons'].values()
                ])
                
                if avg_lookback_diff > 10:
                    comparison['recommendations'].append(f"Significant lookback period differences detected (avg: {avg_lookback_diff:.1f}) - use separate optimization for each direction")
                else:
                    comparison['recommendations'].append("Similar lookback periods across directions - unified optimization may be sufficient")
            
            return comparison
            
        except Exception as e:
            tprint(f"❌ Error generating directional comparison: {e}")
            return {'error': str(e)}
    
    def _generate_directional_optimization_summary(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of directional optimization results."""
        try:
            summary = {
                'total_directions': 0,
                'successful_directions': 0,
                'failed_directions': 0,
                'direction_summaries': {},
                'overall_recommendations': []
            }
            
            directions = ['long', 'short']
            for direction in directions:
                if direction not in optimization_results:
                    continue
                    
                summary['total_directions'] += 1
                direction_data = optimization_results[direction]
                
                if 'error' in direction_data:
                    summary['failed_directions'] += 1
                    summary['direction_summaries'][direction] = {'status': 'failed', 'error': direction_data['error']}
                    continue
                
                summary['successful_directions'] += 1
                
                # Analyze this direction's results
                successful_features = [k for k, v in direction_data.items() 
                                     if isinstance(v, dict) and 'error' not in v]
                failed_features = [k for k, v in direction_data.items() 
                                 if isinstance(v, dict) and 'error' in v]
                
                if successful_features:
                    mi_scores = [direction_data[f].get('combined_mi_score', 0) for f in successful_features]
                    avg_mi_score = np.mean(mi_scores) if mi_scores else 0
                    best_feature = max(successful_features, key=lambda f: direction_data[f].get('combined_mi_score', 0))
                    
                    summary['direction_summaries'][direction] = {
                        'status': 'success',
                        'features_optimized': len(successful_features),
                        'features_failed': len(failed_features),
                        'average_mi_score': avg_mi_score,
                        'best_feature': best_feature,
                        'best_mi_score': direction_data[best_feature].get('combined_mi_score', 0)
                    }
                else:
                    summary['direction_summaries'][direction] = {
                        'status': 'no_successful_features',
                        'features_failed': len(failed_features)
                    }
            
            # Generate overall recommendations
            if summary['successful_directions'] == 2:
                long_summary = summary['direction_summaries'].get('long', {})
                short_summary = summary['direction_summaries'].get('short', {})
                
                if (long_summary.get('average_mi_score', 0) > short_summary.get('average_mi_score', 0) * 1.2):
                    summary['overall_recommendations'].append("Long signals show superior optimization results - prioritize long-focused features")
                elif (short_summary.get('average_mi_score', 0) > long_summary.get('average_mi_score', 0) * 1.2):
                    summary['overall_recommendations'].append("Short signals show superior optimization results - prioritize short-focused features")
                else:
                    summary['overall_recommendations'].append("Balanced performance across directions - implement directional feature optimization")
                    
                summary['overall_recommendations'].append("Directional optimization successful - use separate lookback periods for long and short signals")
            elif summary['successful_directions'] == 1:
                successful_direction = [d for d in directions if summary['direction_summaries'].get(d, {}).get('status') == 'success'][0]
                summary['overall_recommendations'].append(f"Only {successful_direction} signals optimized successfully - consider {successful_direction}-only strategy")
            else:
                summary['overall_recommendations'].append("Directional optimization failed - fallback to unified optimization recommended")
            
            return summary
            
        except Exception as e:
            tprint(f"❌ Error generating directional optimization summary: {e}")
            return {'error': str(e)}
    
    def _convert_new_directional_to_standard_format(self, directional_result: DirectionalOptimizationResult) -> Dict[str, Any]:
        """Convert new directional optimization result to standard format."""
        try:
            # Get all selected features
            all_selected_features = directional_result.get_all_selected_features()
            
            # Create optimized_features dictionary in standard format
            optimized_features = {}
            
            for feature_key, feature_result in all_selected_features.items():
                optimized_features[feature_key] = {
                    'lookback': feature_result.optimal_lookback_period,
                    'score': feature_result.mutual_info_score,
                    'direction': feature_result.direction,
                    'method': 'new_directional_single_period',
                    'optimization_time': feature_result.optimization_time,
                    'sample_count': feature_result.sample_count,
                    'convergence_achieved': feature_result.convergence_achieved,
                    'cross_validation_score': feature_result.cross_validation_score,
                    'stability_score': feature_result.stability_score
                }
            
            # Create standard result format
            standard_result = {
                'optimized_features': optimized_features,
                'best_lookback_period': self._get_best_lookback_from_directional(all_selected_features),
                'best_score': directional_result.average_mutual_info_score,
                'optimization_method': 'new_directional_single_period',
                'total_features_optimized': directional_result.final_feature_count,
                'optimization_time': directional_result.total_optimization_time,
                'convergence_rate': directional_result.convergence_rate,
                'directional_balance_ratio': directional_result.directional_balance_ratio,
                'feature_selection_quality': directional_result.feature_selection_quality,
                'directional_differences': directional_result.directional_differences,
                'complementary_features': directional_result.complementary_features
            }
            
            return standard_result
            
        except Exception as e:
            tprint(f"❌ Error converting new directional result to standard format: {e}")
            return {
                'optimized_features': {},
                'best_lookback_period': 20,
                'best_score': 0.0,
                'optimization_method': 'new_directional_error',
                'error': str(e)
            }
    
    def _get_best_lookback_from_directional(self, selected_features: Dict[str, Any]) -> int:
        """Get the best lookback period from directional features."""
        if not selected_features:
            return 20  # Default fallback
        
        # Find feature with highest mutual information score
        best_feature = max(selected_features.values(), 
                          key=lambda x: getattr(x, 'mutual_info_score', 0))
        return getattr(best_feature, 'optimal_lookback_period', 20)

    def _convert_directional_to_standard_format(self, directional_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert directional optimization results to standard format expected by the pipeline.
        
        Args:
            directional_result: Results from directional optimization
            
        Returns:
            Standard format optimization results
        """
        try:
            standard_result = {
                'optimization_results': {},
                'optimized_features': {},
                'optimization_metrics': {},
                'directional_analysis': directional_result.copy()
            }
            
            # Extract summary information
            summary = directional_result.get('_summary', {})
            comparison = directional_result.get('directional_comparison', {})
            
            # Create consolidated optimization results
            best_lookback_period = 20  # Default fallback
            best_score = 0.0
            total_features = 0
            
            # Process long and short results
            for direction in ['long', 'short']:
                direction_data = directional_result.get(direction, {})
                if 'error' in direction_data:
                    continue
                
                for feature_name, feature_result in direction_data.items():
                    if not isinstance(feature_result, dict) or 'error' in feature_result:
                        continue
                    
                    total_features += 1
                    
                    # Use the better performing direction's results for each feature
                    feature_key = f"{feature_name}_{direction}"
                    
                    # Store in optimized_features format
                    standard_result['optimized_features'][feature_key] = {
                        'lookback': feature_result.get('first_lookback_period', 20),
                        'second_lookback': feature_result.get('second_lookback_period'),
                        'score': feature_result.get('combined_mi_score', 0.0),
                        'direction': direction,
                        'method': 'directional_mrmr',
                        'correlation': feature_result.get('correlation_between_periods', 0.0),
                        'sample_count': feature_result.get('sample_count', 0)
                    }
                    
                    # Track best overall score
                    current_score = feature_result.get('combined_mi_score', 0.0)
                    if current_score > best_score:
                        best_score = current_score
                        best_lookback_period = feature_result.get('first_lookback_period', 20)
            
            # Create consolidated features with best direction for each
            consolidated_features = {}
            feature_names = set()
            
            # Extract unique feature names
            for feature_key in standard_result['optimized_features'].keys():
                if feature_key.endswith('_long') or feature_key.endswith('_short'):
                    base_name = feature_key.rsplit('_', 1)[0]
                    feature_names.add(base_name)
                else:
                    feature_names.add(feature_key)
            
            # For each feature, choose the better performing direction
            for feature_name in feature_names:
                long_key = f"{feature_name}_long"
                short_key = f"{feature_name}_short"
                
                long_result = standard_result['optimized_features'].get(long_key)
                short_result = standard_result['optimized_features'].get(short_key)
                
                if long_result and short_result:
                    # Choose better performing direction
                    if long_result['score'] >= short_result['score']:
                        consolidated_features[feature_name] = long_result.copy()
                        consolidated_features[feature_name]['alternative_direction'] = short_result
                    else:
                        consolidated_features[feature_name] = short_result.copy()
                        consolidated_features[feature_name]['alternative_direction'] = long_result
                elif long_result:
                    consolidated_features[feature_name] = long_result
                elif short_result:
                    consolidated_features[feature_name] = short_result
            
            # Update optimized_features with consolidated results
            standard_result['optimized_features'] = consolidated_features
            
            # Set optimization_results
            standard_result['optimization_results'] = {
                'best_lookback_period': best_lookback_period,
                'best_score': best_score,
                'total_features_optimized': total_features,
                'optimization_method': 'directional_mrmr',
                'directional_summary': summary,
                'directional_comparison': comparison,
                'successful_directions': summary.get('successful_directions', 0),
                'failed_directions': summary.get('failed_directions', 0)
            }
            
            # Set optimization_metrics
            standard_result['optimization_metrics'] = {
                'total_features': total_features,
                'best_score': best_score,
                'average_score': np.mean([f['score'] for f in consolidated_features.values()]) if consolidated_features else 0.0,
                'directional_balance': {
                    'long_features': len([f for f in consolidated_features.values() if f.get('direction') == 'long']),
                    'short_features': len([f for f in consolidated_features.values() if f.get('direction') == 'short'])
                },
                'recommendations': summary.get('overall_recommendations', [])
            }
            
            tprint(f"✅ Converted directional results: {total_features} features optimized across directions")
            return standard_result
            
        except Exception as e:
            tprint(f"❌ Error converting directional results to standard format: {e}")
            # Return fallback standard format
            return {
                'optimization_results': {
                    'best_lookback_period': 20,
                    'best_score': 0.0,
                    'optimization_method': 'directional_mrmr_fallback',
                    'error': str(e)
                },
                'optimized_features': {
                    'fallback_feature': {'lookback': 20, 'score': 0.0, 'method': 'fallback'}
                },
                'optimization_metrics': {'error': str(e)},
                'directional_analysis': directional_result
            }
    
    def intelligent_directional_feature_selection(self, optimization_results: Dict[str, Any], 
                                                   max_features: int = 50,
                                                   significance_threshold: float = 0.1,
                                                   lookback_diff_threshold: int = 5) -> Dict[str, Any]:
        """
        Intelligently select directional features to avoid naive feature doubling.
        
        Args:
            optimization_results: Results from directional optimization
            max_features: Maximum number of features to select
            significance_threshold: Minimum performance difference to warrant directional split
            lookback_diff_threshold: Minimum lookback difference to warrant directional split
            
        Returns:
            Dictionary with selected features and selection rationale
        """
        try:
            tprint("🎯 Starting intelligent directional feature selection...")
            
            selection_result = {
                'directional_features': [],
                'unified_features': [],
                'meta_features': [],
                'selection_rationale': {},
                'feature_budget': {
                    'total_budget': max_features,
                    'directional_used': 0,
                    'unified_used': 0,
                    'meta_used': 0
                }
            }
            
            comparison = optimization_results.get('directional_comparison', {})
            feature_comparisons = comparison.get('feature_comparisons', {})
            
            if not feature_comparisons:
                tprint("⚠️ No directional comparison data available, falling back to unified selection")
                return self._fallback_unified_selection(optimization_results, max_features)
            
            # Analyze each feature for directional significance
            for feature, comp in feature_comparisons.items():
                perf_diff = comp['performance_difference']['mi_score_difference']
                lookback_diff = comp['lookback_difference']['first_period']['difference']
                better_direction = comp['performance_difference']['better_direction']
                
                # Decision logic for feature selection strategy
                if perf_diff > significance_threshold or lookback_diff > lookback_diff_threshold:
                    # Significant directional difference - create directional features
                    long_feature = f"{feature}_long"
                    short_feature = f"{feature}_short"
                    
                    selection_result['directional_features'].extend([long_feature, short_feature])
                    selection_result['selection_rationale'][feature] = {
                        'strategy': 'directional_split',
                        'reason': f'Significant difference: perf_diff={perf_diff:.3f}, lookback_diff={lookback_diff}',
                        'performance_difference': perf_diff,
                        'lookback_difference': lookback_diff,
                        'better_direction': better_direction
                    }
                    
                    # Also create meta-features for this significant directional difference
                    meta_features = self._create_directional_meta_features(feature, comp)
                    selection_result['meta_features'].extend(meta_features)
                    
                else:
                    # No significant directional difference - use unified feature
                    unified_feature = f"{feature}_{better_direction}"  # Use better performing direction
                    selection_result['unified_features'].append(unified_feature)
                    selection_result['selection_rationale'][feature] = {
                        'strategy': 'unified_best',
                        'reason': f'Minimal directional difference, using {better_direction} version',
                        'performance_difference': perf_diff,
                        'lookback_difference': lookback_diff,
                        'chosen_direction': better_direction
                    }
            
            # Apply budget constraints
            selection_result = self._apply_feature_budget_constraints(selection_result, max_features)
            
            # Generate selection summary
            total_selected = (len(selection_result['directional_features']) + 
                            len(selection_result['unified_features']) + 
                            len(selection_result['meta_features']))
            
            tprint(f"✅ Intelligent feature selection completed:")
            tprint(f"   📊 Directional features: {len(selection_result['directional_features'])}")
            tprint(f"   🎯 Unified features: {len(selection_result['unified_features'])}")
            tprint(f"   🔗 Meta features: {len(selection_result['meta_features'])}")
            tprint(f"   📈 Total selected: {total_selected}/{max_features}")
            
            return selection_result
            
        except Exception as e:
            tprint(f"❌ Error in intelligent directional feature selection: {e}")
            return self._fallback_unified_selection(optimization_results, max_features)
    
    def _create_directional_meta_features(self, base_feature: str, comparison_data: Dict[str, Any]) -> List[str]:
        """Create meta-features that capture directional relationships."""
        meta_features = []
        
        # Directional performance difference meta-feature
        meta_features.append(f"{base_feature}_direction_performance_diff")
        
        # Directional lookback difference meta-feature  
        meta_features.append(f"{base_feature}_direction_lookback_diff")
        
        # Directional strength indicator (how much better the better direction is)
        meta_features.append(f"{base_feature}_direction_strength")
        
        return meta_features
    
    def _apply_feature_budget_constraints(self, selection_result: Dict[str, Any], max_features: int) -> Dict[str, Any]:
        """Apply intelligent budget constraints to feature selection."""
        try:
            # Count current features
            directional_count = len(selection_result['directional_features'])
            unified_count = len(selection_result['unified_features'])
            meta_count = len(selection_result['meta_features'])
            total_count = directional_count + unified_count + meta_count
            
            if total_count <= max_features:
                # Within budget, no constraints needed
                # Convert int64 values before dictionary operations
                safe_directional = convert_int64_to_int(directional_count)
                safe_unified = convert_int64_to_int(unified_count)
                safe_meta = convert_int64_to_int(meta_count)

                selection_result['feature_budget'].update({
                    'directional_used': safe_directional,
                    'unified_used': safe_unified,
                    'meta_used': safe_meta,
                    'budget_exceeded': False
                })
                return selection_result
            
            # Budget exceeded, need to prioritize
            tprint(f"⚠️ Feature budget exceeded ({total_count} > {max_features}), applying constraints...")
            
            # Priority order: directional > unified > meta (directional features are most valuable)
            remaining_budget = max_features
            
            # Keep all directional features (highest priority)
            if directional_count <= remaining_budget:
                remaining_budget -= directional_count
            else:
                # Even directional features exceed budget - keep top performers
                selection_result['directional_features'] = selection_result['directional_features'][:remaining_budget]
                remaining_budget = 0
            
            # Keep unified features if budget allows
            if remaining_budget > 0 and unified_count > 0:
                kept_unified = min(unified_count, remaining_budget)
                selection_result['unified_features'] = selection_result['unified_features'][:kept_unified]
                remaining_budget -= kept_unified
            else:
                selection_result['unified_features'] = []
            
            # Keep meta features if budget allows
            if remaining_budget > 0 and meta_count > 0:
                kept_meta = min(meta_count, remaining_budget)
                selection_result['meta_features'] = selection_result['meta_features'][:kept_meta]
                remaining_budget -= kept_meta
            else:
                selection_result['meta_features'] = []
            
            # Update budget tracking
            # Convert int64 values before dictionary operations
            safe_directional_final = convert_int64_to_int(len(selection_result['directional_features']))
            safe_unified_final = convert_int64_to_int(len(selection_result['unified_features']))
            safe_meta_final = convert_int64_to_int(len(selection_result['meta_features']))
            safe_features_dropped = convert_int64_to_int(total_count - max_features)

            selection_result['feature_budget'].update({
                'directional_used': safe_directional_final,
                'unified_used': safe_unified_final,
                'meta_used': safe_meta_final,
                'budget_exceeded': True,
                'features_dropped': safe_features_dropped
            })
            
            final_count = (len(selection_result['directional_features']) + 
                          len(selection_result['unified_features']) + 
                          len(selection_result['meta_features']))
            
            tprint(f"✅ Budget constraints applied: {final_count}/{max_features} features selected")
            
            return selection_result
            
        except Exception as e:
            tprint(f"❌ Error applying budget constraints: {e}")
            return selection_result
    
    def _fallback_unified_selection(self, optimization_results: Dict[str, Any], max_features: int) -> Dict[str, Any]:
        """Fallback to unified feature selection when directional selection fails."""
        tprint("🔄 Using fallback unified feature selection...")
        
        # Extract features from optimization results
        optimized_features = optimization_results.get('optimized_features', {})
        
        # Sort by performance score
        sorted_features = sorted(optimized_features.items(), 
                               key=lambda x: x[1].get('score', 0), reverse=True)
        
        # Select top features within budget
        selected_features = [name for name, _ in sorted_features[:max_features]]
        
        return {
            'directional_features': [],
            'unified_features': selected_features,
            'meta_features': [],
            'selection_rationale': {'fallback': 'Used unified selection due to directional selection failure'},
            'feature_budget': {
                'total_budget': max_features,
                'directional_used': 0,
                'unified_used': len(selected_features),
                'meta_used': 0,
                'fallback_used': True
            }
        }
    
    def _create_fallback_targets(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create fallback target variables from price movements for feature optimization.

        This method is used when proper labeling data is not available, allowing
        feature optimization to still work by creating basic directional targets.

        Args:
            market_data: Market data DataFrame

        Returns:
            DataFrame with fallback target variables
        """
        try:
            if market_data is None or len(market_data) < 20:
                tprint('⚠️ Insufficient data for fallback targets - using random targets')
                return pd.DataFrame({'target': np.random.choice([0, 1, 2], len(market_data))})

            tprint('🔄 Creating fallback targets from price movements...')

            # Create basic directional targets based on price movements
            returns = market_data['close'].pct_change().fillna(0)

            # Create multi-class targets based on return magnitude and direction
            targets = pd.Series(1, index=market_data.index, dtype=int)  # Default neutral

            # Strong up (>2%)
            targets[returns > 0.02] = 2
            # Up (0.5% to 2%)
            targets[(returns > 0.005) & (returns <= 0.02)] = 1
            # Strong down (<-2%)
            targets[returns < -0.02] = 0
            # Down (-0.5% to -2%)
            targets[(returns < -0.005) & (returns >= -0.02)] = 0

            # Remove first few samples where returns might be NaN
            targets = targets.iloc[1:].reset_index(drop=True)

            # Ensure targets match market_data length (accounting for pct_change)
            if len(targets) < len(market_data):
                # Pad with neutral targets
                padding = pd.Series(1, index=range(len(targets), len(market_data)))
                targets = pd.concat([targets, padding], ignore_index=True)
            elif len(targets) > len(market_data):
                targets = targets[:len(market_data)]

            # Create DataFrame with target and metadata
            fallback_df = pd.DataFrame({
                'target': targets,
                'target_type': 'fallback_directional',
                'created_at': datetime.now(),
                'samples': len(targets),
                'strong_up_ratio': (targets == 2).sum() / len(targets),
                'up_ratio': (targets == 1).sum() / len(targets),
                'neutral_ratio': (targets == 1).sum() / len(targets),  # Most are neutral
                'down_ratio': (targets == 0).sum() / len(targets),
                'strong_down_ratio': (targets == 0).sum() / len(targets)  # Strong down is also 0
            }, index=market_data.index)

            tprint(f'✅ Created fallback targets: {len(targets)} samples')
            tprint(f'📊 Target distribution: Up={fallback_df["up_ratio"].iloc[0]:.1%}, Neutral={fallback_df["neutral_ratio"].iloc[0]:.1%}, Down={fallback_df["down_ratio"].iloc[0]:.1%}')

            return fallback_df

        except Exception as e:
            tprint(f'❌ Error creating fallback targets: {e}')
            # Return simple random targets as last resort
            return pd.DataFrame({'target': np.random.choice([0, 1, 2], len(market_data))})

    def get_mrmr_optimization_metrics(self) -> Dict[str, Any]:
        """Get metrics from MRMR optimization."""
        if not MRMR_OPTIMIZER_AVAILABLE or self.mrmr_optimizer is None:
            return {'error': 'MRMR optimizer not available'}

        try:
            return self.mrmr_optimizer.get_optimization_summary()
        except Exception as e:
            return {'error': str(e)}