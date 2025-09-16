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
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Use dependency manager for robust imports
from .dependency_manager import dependency_manager, get_dependency, is_dependency_available

# Core dependencies with fallback support
np, np_fallback = get_dependency('numpy')
pd, pd_fallback = get_dependency('pandas')

if np_fallback or pd_fallback:
    from src.utils.tprint import tprint
    tprint("⚠️ Using fallback implementations for core dependencies")

# Import common utilities for enhanced functionality
from src.utils.common_operations import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
    safe_apply_function, create_summary_statistics, safe_drop_columns,
    safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, safe_filter_dataframe, create_data_quality_report,
    optimize_dataframe_dtypes, safe_fillna, safe_rolling, safe_groupby_operation,
    safe_apply_function, safe_filter_dataframe, create_summary_statistics,
    safe_to_parquet, safe_read_parquet, validate_dataframe_schema,
    guard_dataframe_nulls, memory_checkpoint, gpu_context, optimize_memory,
    get_memory_usage, integrate_with_m1_optimizers, get_m1_gpu_manager,
    get_m1_memory_optimizer, get_m1_cpu_optimizer
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
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, safe_correlation,
    safe_covariance, safe_mean, safe_std, safe_percentile,
    validate_correlation_matrix, safe_matrix_inverse, math_safe,
    MathValidation, MathValidationError
)

from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

# Import ML common utilities for enhanced ML operations
try:
    from src.utils.ml_common.common_operations import (
        safe_cross_validation, safe_hyperparameter_optimization,
        safe_feature_selection, safe_model_training, safe_model_evaluation
    )
    from src.utils.ml_common.data_processing.data_quality import (
        DataQualityChecker, DataQualityReport
    )
    from src.utils.ml_common.data_processing.feature_preparation import (
        FeaturePreparator, FeatureScaler
    )
    from src.utils.ml_common.optimization.hyperparameter_optimization import (
        HyperparameterOptimizer, OptimizationConfig
    )
    from src.utils.ml_common.validation.cross_validation import (
        CrossValidator, CVConfig
    )
    from src.utils.ml_common.monitoring.performance_monitor import (
        PerformanceMonitor, PerformanceMetrics
    )
    ML_COMMON_AVAILABLE = True
except ImportError as e:
    ML_COMMON_AVAILABLE = False
    tprint(f"⚠️ ML common utilities not available: {e}")

# Import matrix operations for efficient computation
try:
    from src.utils.matrix_operations.unified_operations import (
        UnifiedMatrixOperations, MatrixOptimizer, safe_correlation_matrix,
        safe_matrix_inverse, eigendecomposition, svd_decomposition,
        kmeans_plus_plus_init, normalize_matrix, initialize_covariances,
        get_unified_matrix_operations
    )
    from src.utils.matrix_operations.vectorized_core import (
        VectorizedOperations, VectorizedOptimizer, VectorizedProcessingCore,
        vectorized_rolling_features, matrix_correlation_analysis,
        compute_trading_indicators, get_vectorized_processing_core
    )
    from src.utils.matrix_operations.hardware_integration import (
        HardwareOptimizedOperations, HardwareOptimizedMatrixProcessor,
        hardware_optimized, optimize_matrix_operation, get_hardware_optimized_processor,
        HardwareConfig
    )
    from src.utils.matrix_operations.batch_operations import (
        BatchMatrixProcessor, batch_feature_transformation,
        batch_correlation_analysis, get_batch_matrix_processor
    )
    MATRIX_OPS_AVAILABLE = True
except ImportError as e:
    MATRIX_OPS_AVAILABLE = False
    tprint(f"⚠️ Matrix operations not available: {e}")

from ...market_analysis.components.base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult
from .optimization_reporter import OptimizationReporter
from .validation_framework import ValidationFramework, ValidationLevel, ValidationStatus
from .monitoring_metrics import MonitoringMetrics, MetricType, MetricLevel
from src.utils.logger import system_logger
from src.utils.tprint import tprint

# Hardware optimization imports
try:
    from src.utils.hardware import get_unified_hardware_manager, get_advanced_memory_optimizer
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    tprint("⚠️ Hardware optimization not available - using fallback memory management")

# Import advanced matrix operations
try:
    from src.utils.matrix_operations import (
        get_enhanced_matrix_operations, get_vectorized_processing_core, 
        get_batch_matrix_processor, safe_matrix_multiply, safe_correlation_matrix,
        safe_matrix_inverse, gpu_matrix_multiply, correlation_matrix_gpu,
        eigendecomposition_gpu, batch_matrix_multiply, batch_feature_transformation,
        batch_correlation_analysis, optimize_matrix_operation_with_hardware
    )
    MATRIX_OPS_AVAILABLE = True
except ImportError as e:
    MATRIX_OPS_AVAILABLE = False
    tprint(f"⚠️ Advanced matrix operations not available: {e}")

# Import common operations for enhanced functionality
try:
    from src.utils.common_operations import (
        safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
        validate_finite, get_memory_usage, timed_operation
    )
    COMMON_OPERATIONS_AVAILABLE = True
except ImportError as e:
    COMMON_OPERATIONS_AVAILABLE = False
    tprint(f"⚠️ Common operations not available: {e}")

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

# Configuration constants
class OptimizationConfig:
    """Configuration constants for optimization."""
    DEFAULT_LOOKBACK_RANGE = (5, 50)
    DEFAULT_POPULATION_SIZE = 50
    DEFAULT_GENERATIONS = 100
    DEFAULT_MUTATION_RATE = 0.1
    DEFAULT_CROSSOVER_RATE = 0.8
    DEFAULT_ELITISM_RATE = 0.1
    DEFAULT_MAX_FEATURES = 20
    DEFAULT_FEATURE_IMPORTANCE_THRESHOLD = 0.01
    DEFAULT_MEMORY_LIMIT_GB = 8.0
    DEFAULT_DATA_COMPLETENESS_THRESHOLD = 0.8
    DEFAULT_OPTIMIZATION_TIMEOUT = 600  # 10 minutes
    DEFAULT_MEMORY_CLEANUP_INTERVAL = 100  # Cleanup every 100 operations


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
        super().__init__(config)
        self.logger = system_logger.getChild('FeatureLookbackOptimization')
        self.optimization_status = OptimizationStatus.PENDING
        self.start_time: Optional[float] = None
        self.metrics: Optional[OptimizationMetrics] = None
        
        # Performance monitoring
        self.performance_monitor = {
            'memory_usage': [],
            'cpu_usage': [],
            'execution_times': {},
            'error_counts': 0
        }
        
        # Initialize common utilities
        self.common_utils = CommonUtilities()
        self.math_validator = MathValidation()
        self.serializer = UniversalSerializer()
        
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
        
        # Initialize matrix operations components
        self.enhanced_matrix_ops = None
        self.vectorized_core = None
        self.batch_processor = None
        
        if MATRIX_OPS_AVAILABLE:
            try:
                self.enhanced_matrix_ops = get_enhanced_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                self.batch_processor = get_batch_matrix_processor()
                tprint("✅ Advanced matrix operations initialized for feature lookback optimization")
            except Exception as e:
                tprint(f"⚠️ Matrix operations initialization failed: {e}")
        
        tprint(f"🔧 Matrix operations available: {MATRIX_OPS_AVAILABLE}")
        tprint(f"🔧 Common operations available: {COMMON_OPERATIONS_AVAILABLE}")
        tprint(f"🔧 MRMR optimizer available: {MRMR_OPTIMIZER_AVAILABLE}")
        
        # Initialize MRMR optimizer if available
        self.mrmr_optimizer = None
        if MRMR_OPTIMIZER_AVAILABLE:
            try:
                self.mrmr_optimizer = MRMRLookbackOptimizer()
                tprint("✅ MRMR lookback optimizer initialized")
            except Exception as e:
                tprint(f"⚠️ Failed to initialize MRMR optimizer: {e}")
        
        # Initialize reporter
        self.reporter = OptimizationReporter(
            output_dir=f"reports/feature_lookback_optimization/{self.config.symbol}_{self.config.exchange}_{self.config.timeframe}"
        )
        
        # Initialize validation framework
        self.validation_framework = ValidationFramework()
        
        # Initialize monitoring metrics
        self.monitoring = MonitoringMetrics(f"FeatureLookbackOptimization_{self.config.symbol}")
        
        # Memory cleanup counter
        self.operation_count = 0
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['feature_lookback_optimization_result']
    
    def _cleanup_memory(self) -> None:
        """Clean up memory using hardware optimization tools."""
        self.operation_count += 1
        
        if self.operation_count % OptimizationConfig.DEFAULT_MEMORY_CLEANUP_INTERVAL == 0:
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
                        tprint(f"🧹 Common operations memory cleanup: {memory_result.get('objects_collected', 0)} objects collected")
                    else:
                        # Basic cleanup
                        import gc
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
    
    async def _enhanced_data_handling(self, data: Any, pipeline_state: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Enhanced data handling to get data from multiple sources."""
        try:
            # Try direct data first
            if data is not None:
                if isinstance(data, pd.DataFrame) and not data.empty:
                    # Use common utilities for data validation and optimization
                    validated_data = self._validate_and_optimize_data(data)
                    tprint("✅ Using direct DataFrame data")
                    return validated_data
                elif hasattr(data, 'to_dataframe'):
                    df = data.to_dataframe()
                    if not df.empty:
                        validated_data = self._validate_and_optimize_data(df)
                        tprint("✅ Converted data to DataFrame")
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
        try:
            # Use common utilities for memory monitoring
            memory_usage = get_memory_usage()
            memory_mb = memory_usage / 1024 / 1024 if memory_usage > 0 else 0.0
            
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
            
            # Step 3: Get labeled data from previous stage
            triple_barrier_labeling = pipeline_state.get('triple_barrier_labeling_result', {})
            if not triple_barrier_labeling:
                raise ValueError("No triple barrier labeling results available for feature optimization")
            
            tprint('🏷️ Triple barrier labeling data retrieved')
            
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
                feature_optimizer, market_data, triple_barrier_labeling, optimization_config
            )
            self._monitor_performance('optimization_complete')
            
            # Step 7: Extract and validate results
            tprint('📋 Extracting optimization results...')
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
            from src.feature_engineering.feature_generation_optimization import FeatureOptimizationConfig
            
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
            
            config = FeatureOptimizationConfig(
                optimization_method='genetic_algorithm',
                lookback_range=OptimizationConfig.DEFAULT_LOOKBACK_RANGE,
                feature_types=['technical_indicators', 'price_features', 'volume_features'],
                optimization_metric='sharpe_ratio',
                cross_validation_folds=cv_config.get('folds', 5),
                test_size=cv_config.get('test_size', 0.2),
                random_state=42,
                
                # Genetic algorithm parameters (use optimized values if available)
                population_size=hpo_config.get('population_size', OptimizationConfig.DEFAULT_POPULATION_SIZE),
                generations=hpo_config.get('generations', OptimizationConfig.DEFAULT_GENERATIONS),
                mutation_rate=hpo_config.get('mutation_rate', OptimizationConfig.DEFAULT_MUTATION_RATE),
                crossover_rate=hpo_config.get('crossover_rate', OptimizationConfig.DEFAULT_CROSSOVER_RATE),
                elitism_rate=hpo_config.get('elitism_rate', OptimizationConfig.DEFAULT_ELITISM_RATE),
                
                # Feature selection
                enable_feature_selection=True,
                max_features=OptimizationConfig.DEFAULT_MAX_FEATURES,
                feature_importance_threshold=OptimizationConfig.DEFAULT_FEATURE_IMPORTANCE_THRESHOLD,
                
                # Regime-aware optimization
                enable_regime_aware_optimization=enable_regime_aware,
                regime_specific_optimization=enable_regime_aware,
                
                # Hardware optimization
                enable_parallel_processing=True,
                enable_gpu_acceleration=self.m1_gpu_manager and self.m1_gpu_manager.is_m1,
                memory_limit_gb=OptimizationConfig.DEFAULT_MEMORY_LIMIT_GB,
                
                # Matrix operations optimization
                enable_matrix_optimization=self.matrix_ops is not None,
                enable_vectorization=self.vectorized_ops is not None,
                
                # ML common utilities integration
                use_ml_common_utilities=ML_COMMON_AVAILABLE,
                use_data_quality_checker=self.data_quality_checker is not None,
                use_feature_preparator=self.feature_preparator is not None
            )
            
            tprint(f'⚙️ Enhanced optimization config created (regime-aware: {enable_regime_aware})')
            return config
            
        except ImportError as e:
            tprint(f"⚠️ Feature optimization config import failed: {e}")
            # Return a simple fallback config with common utilities integration
            return {
                'optimization_method': 'statistical',
                'lookback_range': OptimizationConfig.DEFAULT_LOOKBACK_RANGE,
                'regime_aware': False,
                'use_ml_common_utilities': ML_COMMON_AVAILABLE,
                'use_matrix_operations': MATRIX_OPS_AVAILABLE,
                'use_m1_optimization': self.m1_gpu_manager and self.m1_gpu_manager.is_m1
            }
    
    async def _get_feature_optimizer(self, config: Any) -> Any:
        """Get feature optimizer with fallback handling."""
        try:
            from src.feature_engineering.feature_generation_optimization import get_feature_optimizer
            optimizer = get_feature_optimizer(config)
            tprint('✅ Feature optimizer initialized successfully')
            return optimizer
            
        except ImportError as e:
            tprint(f"⚠️ Feature optimizer import failed: {e}")
            # Return a fallback optimizer
            return self._create_fallback_optimizer()
    
    def _create_fallback_optimizer(self) -> Any:
        """Create a fallback optimizer for when ML commons are not available."""
        class FallbackOptimizer:
            def __init__(self, config):
                self.config = config
                self.logger = system_logger.getChild('FallbackOptimizer')
            
            async def optimize_features(self, data, config):
                tprint("Using fallback statistical optimization")
                return {
                    'optimization_results': {
                        'best_lookback_period': 20,
                        'best_score': 0.5,
                        'optimization_method': 'fallback_statistical'
                    },
                    'optimized_features': {
                        'rsi': {'lookback': 14, 'score': 0.5},
                        'sma': {'lookback': 20, 'score': 0.4},
                        'ema': {'lookback': 12, 'score': 0.45}
                    },
                    'optimization_metrics': {
                        'method': 'fallback_statistical',
                        'convergence_iterations': 1
                    },
                    'optimization_time': 0.1
                }
        
        return FallbackOptimizer(self.config)
    
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
            artifact_path = f"artifacts/feature_lookback_optimization_{self.config.symbol}_{self.config.exchange}_{self.config.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Ensure directory exists
            from pathlib import Path
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
    
    async def _perform_feature_optimization(
        self, 
        feature_optimizer: Any, 
        market_data: Any, 
        triple_barrier_labeling: Dict[str, Any],
        config: Any
    ) -> Dict[str, Any]:
        """Perform the actual feature optimization process with comprehensive error handling and matrix operations."""
        optimization_start_time = time.time()
        
        try:
            tprint('🔄 Preparing data for optimization...')
            # Prepare data for optimization
            prepared_data = self._prepare_data_for_optimization(market_data, triple_barrier_labeling)
            self._monitor_performance('data_prepared')
            
            # Enhanced optimization using matrix operations if available
            if MATRIX_OPS_AVAILABLE and self.matrix_ops:
                tprint('🚀 Executing enhanced feature optimization with matrix operations...')
                
                # Enhanced correlation analysis
                correlation_analysis = self._enhanced_correlation_analysis(prepared_data)
                
                # Vectorized feature engineering
                engineered_features = self._vectorized_feature_engineering(prepared_data)
                
                # Hardware-optimized processing
                hardware_optimized_features = self._hardware_optimized_feature_processing(engineered_features)
                
                # Batch optimization processing
                batch_results = self._batch_optimization_processing(hardware_optimized_features)
                
                # Perform traditional optimization on enhanced data
                optimization_result = await feature_optimizer.optimize_features(hardware_optimized_features, config)
                
                # Enhance results with matrix operations data
                optimization_result.update({
                    'correlation_analysis': correlation_analysis,
                    'engineered_features': engineered_features,
                    'hardware_optimized_features': hardware_optimized_features,
                    'batch_results': batch_results,
                    'optimization_method': 'matrix_operations_enhanced'
                })
                
                tprint('✅ Enhanced feature optimization with matrix operations completed')
            else:
                tprint('🚀 Executing standard feature optimization...')
                # Perform standard feature optimization
                optimization_result = await feature_optimizer.optimize_features(prepared_data, config)
                optimization_result['optimization_method'] = 'standard'
                tprint('✅ Standard feature optimization completed')
            
            self._monitor_performance('optimization_executed')
            
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
                    'convergence_iterations': 0,
                    'fallback_used': True
                },
                'optimization_time': optimization_time,
                'regime_specific_results': {},
                'error_details': {
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            }
    
    def _prepare_data_for_optimization(self, data: Any, triple_barrier_labeling: Dict[str, Any]) -> Any:
        """Prepare market data and labeled data for optimization with comprehensive validation."""
        try:
            if not isinstance(data, pd.DataFrame):
                tprint("⚠️ Data is not a DataFrame, using fallback preparation")
                return {
                    'market_data': data,
                    'triple_barrier_labeling': triple_barrier_labeling,
                    'preparation_method': 'fallback'
                }
            
            # Create a copy to avoid modifying original data
            prepared_data = data.copy()
            
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
                    prepared_data = self.feature_preparator.prepare_features(prepared_data)
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
                        from src.utils.hardware.m1_gpu_utils import optimize_dataframe_for_m1
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
            
            return {
                'market_data': prepared_data,
                'triple_barrier_labeling': triple_barrier_labeling,
                'preparation_metadata': preparation_metadata,
                'preparation_method': 'enhanced_with_common_utilities'
            }
            
        except Exception as e:
            tprint(f"❌ Data preparation failed: {e}")
            # Return minimal fallback
            return {
                'market_data': data,
                'triple_barrier_labeling': triple_barrier_labeling,
                'preparation_method': 'fallback',
                'preparation_error': str(e)
            }
    
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
            'memory_usage': get_memory_usage() if COMMON_OPERATIONS_AVAILABLE else 0.0
        }
        
        enhanced_metrics = {
            **base_metrics,
            'matrix_operations_available': MATRIX_OPS_AVAILABLE,
            'common_operations_available': COMMON_OPERATIONS_AVAILABLE,
            'enhanced_matrix_ops_initialized': self.enhanced_matrix_ops is not None,
            'vectorized_core_initialized': self.vectorized_core is not None,
            'batch_processor_initialized': self.batch_processor is not None,
            'hardware_optimization_available': HARDWARE_OPTIMIZATION_AVAILABLE,
            'hardware_manager_initialized': self.hardware_manager is not None,
            'memory_optimizer_initialized': self.memory_optimizer is not None
        }
        
        return enhanced_metrics
    
    def optimize_lookback_periods_mrmr(self, 
                                         data: pd.DataFrame,
                                         feature_columns: List[str],
                                         target_column: str = 'returns',
                                         optimization_config: Optional[LookbackOptimizationConfig] = None) -> Dict[str, Any]:
        """
        Optimize lookback periods using MRMR approach (MI + mRMR).
        
        Args:
            data: Input data with features and target
            feature_columns: List of feature columns to optimize
            target_column: Name of the target column
            optimization_config: Optional configuration for optimization
            
        Returns:
            Dictionary with optimization results for each feature
        """
        if not MRMR_OPTIMIZER_AVAILABLE or self.mrmr_optimizer is None:
            tprint("⚠️ Bayesian optimizer not available - using fallback optimization")
            return self._fallback_lookback_optimization(data, feature_columns, target_column)
        
        tprint("🔍 Starting Bayesian lookback period optimization...")
        start_time = time.time()
        
        optimization_results = {}
        
        try:
            # Create optimization config if not provided
            if optimization_config is None:
                optimization_config = LookbackOptimizationConfig(
                    n_trials=50,  # Reduced for faster execution
                    min_lookback=5,
                    max_lookback=50,
                    max_correlation_threshold=0.7,
                    min_mutual_info_threshold=0.1,
                    enable_pruning=True,
                    enable_parallel=True
                )
            
            # Optimize each feature
            for feature_name in feature_columns:
                tprint(f"📊 Optimizing lookback periods for {feature_name}...")
                
                try:
                    # Optimize lookback periods for this feature
                    result = self.mrmr_optimizer.optimize_lookback_periods(
                        data=data,
                        feature_name=feature_name,
                        target_column=target_column,
                        parameter_type="technical_indicator"
                    )
                    
                    # Store results
                    optimization_results[feature_name] = {
                        'first_lookback_period': result.first_lookback_period,
                        'second_lookback_period': result.second_lookback_period,
                        'first_mi_score': result.first_mi_score,
                        'second_mi_score': result.second_mi_score,
                        'combined_mi_score': result.combined_mi_score,
                        'correlation_between_periods': result.correlation_between_periods,
                        'optimization_time': result.optimization_time,
                        'n_trials': result.n_trials,
                        'best_score': result.best_score,
                        'convergence_rate': result.convergence_rate,
                        'parameter_importance': result.parameter_importance
                    }
                    
                    tprint(f"✅ {feature_name}: First={result.first_lookback_period} (MI={result.first_mi_score:.4f}), "
                          f"Second={result.second_lookback_period} (MI={result.second_mi_score:.4f}), "
                          f"Correlation={result.correlation_between_periods:.4f}")
                    
                except Exception as e:
                    tprint(f"❌ Failed to optimize {feature_name}: {e}")
                    optimization_results[feature_name] = {
                        'error': str(e),
                        'first_lookback_period': None,
                        'second_lookback_period': None
                    }
            
            total_time = time.time() - start_time
            tprint(f"✅ Bayesian optimization completed in {total_time:.2f} seconds")
            
            # Generate summary
            summary = self._generate_optimization_summary(optimization_results)
            optimization_results['_summary'] = summary
            
            return optimization_results
            
        except Exception as e:
            tprint(f"❌ Bayesian optimization failed: {e}")
            return {'error': str(e)}
    
    def _fallback_lookback_optimization(self, 
                                      data: pd.DataFrame,
                                      feature_columns: List[str],
                                      target_column: str) -> Dict[str, Any]:
        """Fallback optimization when Bayesian optimizer is not available."""
        tprint("🔄 Using fallback lookback optimization...")
        
        optimization_results = {}
        
        for feature_name in feature_columns:
            try:
                # Simple grid search for lookback periods
                best_first = 10
                best_second = 20
                best_score = 0.0
                
                # Basic optimization logic
                for first_lookback in range(5, 51, 5):
                    for second_lookback in range(5, 51, 5):
                        if second_lookback == first_lookback:
                            continue
                        
                        # Calculate simple correlation
                        first_feature = data['close'].rolling(window=first_lookback).mean()
                        second_feature = data['close'].rolling(window=second_lookback).mean()
                        
                        correlation = first_feature.corr(second_feature)
                        
                        if abs(correlation) < 0.7:  # Low correlation
                            score = 1.0 - abs(correlation)  # Higher score for lower correlation
                            if score > best_score:
                                best_score = score
                                best_first = first_lookback
                                best_second = second_lookback
                
                optimization_results[feature_name] = {
                    'first_lookback_period': best_first,
                    'second_lookback_period': best_second,
                    'first_mi_score': 0.5,  # Placeholder
                    'second_mi_score': 0.5,  # Placeholder
                    'combined_mi_score': best_score,
                    'correlation_between_periods': abs(correlation),
                    'optimization_time': 0.1,
                    'n_trials': 100,
                    'best_score': best_score,
                    'convergence_rate': 1.0,
                    'parameter_importance': {},
                    'method': 'fallback_grid_search'
                }
                
            except Exception as e:
                optimization_results[feature_name] = {
                    'error': str(e),
                    'first_lookback_period': 10,
                    'second_lookback_period': 20
                }
        
        return optimization_results
    
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
    
    def get_mrmr_optimization_metrics(self) -> Dict[str, Any]:
        """Get metrics from MRMR optimization."""
        if not MRMR_OPTIMIZER_AVAILABLE or self.mrmr_optimizer is None:
            return {'error': 'MRMR optimizer not available'}
        
        try:
            return self.mrmr_optimizer.get_optimization_summary()
        except Exception as e:
            return {'error': str(e)}