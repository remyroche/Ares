"""
Analysis Components - Comprehensive Market Analysis Framework

This module provides comprehensive analysis components for market analysis,
integrating various utilities for data processing, ML operations, hardware optimization,
and advanced analytics capabilities.

Features:
- Unified data processing and validation
- Advanced ML model analysis and evaluation
- M1 hardware optimization for Apple Silicon
- Comprehensive confidence metrics and calibration
- Matrix operations and mathematical validation
- Serialization and data persistence
- Enhanced logging and monitoring
"""

import logging
import time
import asyncio
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Core utilities
from ....utils.common_operations import (
    safe_json_dump, safe_json_load, ensure_directory, safe_file_exists,
    validate_dataframe, validate_dataframe_columns, safe_dataframe_operation,
    safe_merge_dataframes, safe_groupby_operation, safe_apply_function,
    create_summary_statistics, safe_drop_columns, safe_rename_columns,
    validate_timestamp_column, safe_timestamp_conversion, get_dataframe_info,
    safe_filter_dataframe, create_data_quality_report, safe_to_parquet,
    safe_read_parquet, optimize_dataframe_dtypes, safe_resample,
    align_dataframes, validate_dataframe_schema, guard_dataframe_nulls,
    safe_copy, safe_deepcopy, get_latest_outcome_file,
    load_latest_optimal_regime_clustering_outcome, integrate_with_m1_optimizers,
    cleanup_m1_optimizers, memory_checkpoint, gpu_context, optimize_memory,
    get_memory_usage, validate_file_path, get_file_size, check_disk_space,
    CommonUtilities
)

from ....utils.common_utilities import (
    safe_dataframe_operation as safe_df_op, validate_dataframe_columns as validate_df_cols,
    safe_convert_dtypes, calculate_data_quality_metrics, safe_merge_dataframes as safe_merge,
    safe_groupby_operation as safe_groupby, safe_apply_function as safe_apply,
    create_summary_statistics as create_summary, safe_drop_columns as safe_drop,
    safe_rename_columns as safe_rename, validate_timestamp_column as validate_ts,
    safe_timestamp_conversion as safe_ts_conv, get_dataframe_info as get_df_info,
    safe_filter_dataframe as safe_filter, create_data_quality_report as create_quality_report,
    CommonUtilities as CommonUtils
)

from ....utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, validate_numeric_array,
    safe_kelly_calculation, safe_weighted_average, safe_percentage_change,
    safe_correlation, safe_covariance, safe_mean, safe_std, safe_percentile,
    validate_correlation_matrix, safe_matrix_inverse, math_safe,
    MathValidation, MathValidationError
)

from ....utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)

from ....utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_structured,
    tprint_with_level, tprint_batch, tprint_timer, tprint_logged,
    configure_tprint, get_tprint_config, tprint_context, cleanup_tprint,
    enable_auto_print_logging, set_print_log_level, TPrintConfig, LogLevel
)

# Data utilities
try:
    from ....utils.data.unified_data_utils import (
        load_market_data, validate_market_data, preprocess_market_data,
        create_feature_engineering_pipeline, apply_feature_engineering,
        calculate_technical_indicators, detect_market_regimes,
        create_regime_aware_features, optimize_data_for_analysis
    )
    DATA_UTILS_AVAILABLE = True
except ImportError:
    DATA_UTILS_AVAILABLE = False
    tprint_warning("Data utilities not available")

# Matrix operations
try:
    from ....utils.matrix_operations.unified_operations import (
        MatrixOperations, VectorizedOperations, BatchOperations,
        create_matrix_operations, create_vectorized_operations,
        create_batch_operations, optimize_matrix_operations
    )
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False
    tprint_warning("Matrix operations not available")

# Hardware optimization
from ....utils.hardware.m1_gpu_utils import (
    get_m1_gpu_manager, is_m1_available, is_mps_available,
    optimize_dataframe_for_m1, create_m1_optimized_array,
    m1_backtesting_simulate, m1_monte_carlo_simulate
)

from ....utils.hardware.m1_memory_optimizer import (
    get_m1_memory_optimizer, start_m1_memory_monitoring,
    stop_m1_memory_monitoring, optimize_dataframe_memory,
    optimize_memory, get_memory_usage as get_memory_stats
)

from ....utils.hardware.m1_cpu_optimizer import (
    get_m1_cpu_optimizer, optimize_function_for_m1, parallel_map_m1,
    create_m1_optimized_thread_pool, run_cpu_intensive_task,
    parallel_backtesting_worker, create_parallel_backtesting_pool,
    parallel_monte_carlo_simulation, run_monte_carlo_batch
)

# ML utilities
try:
    from ....utils.ml_common.common_operations import get_ml_common_operations
    from ....utils.ml_common.confidence_metrics import (
        calculate_confidence_metrics, calculate_calibration_metrics,
        calculate_expected_calibration_error, calculate_prediction_distribution,
        log_confidence_metrics, get_confidence_summary, ModelConfidenceCalibration,
        calibrate_model_confidence
    )
    from ....utils.ml_common.feature_selection import (
        FeatureSelector, create_feature_selector, select_features,
        evaluate_feature_importance, create_feature_engineering_pipeline
    )
    ML_UTILS_AVAILABLE = True
except ImportError:
    ML_UTILS_AVAILABLE = False
    tprint_warning("ML utilities not available")

# Unified evaluation framework (if available)
try:
    from src.utils.nas_tas.unified_evaluator import (
        UnifiedEvaluationFramework, EvaluationConfig
    )
    from src.utils.nas_tas.unified_regime_config import ArchitectureType
    UNIFIED_FRAMEWORK_AVAILABLE = True
except ImportError:
    UNIFIED_FRAMEWORK_AVAILABLE = False
    UnifiedEvaluationFramework = None
    EvaluationConfig = None
    ArchitectureType = None

# Setup logging
logger = logging.getLogger(__name__)

class AnalysisComponents:
    """
    Comprehensive Analysis Components for Market Analysis.
    
    This class provides a unified interface for market analysis components,
    integrating data processing, ML operations, hardware optimization,
    and advanced analytics capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Analysis Components.
        
        Args:
            config: Configuration dictionary for analysis components
        """
        self.config = config or {}
        self.logger = logger.getChild('AnalysisComponents')
        
        # Initialize utilities
        self.common_ops = CommonUtilities()
        self.math_validator = MathValidation()
        self.serializer = UniversalSerializer()
        
        # Initialize hardware optimizers
        self.gpu_manager = get_m1_gpu_manager()
        self.memory_optimizer = get_m1_memory_optimizer()
        self.cpu_optimizer = get_m1_cpu_optimizer()
        
        # Initialize ML utilities
        if ML_UTILS_AVAILABLE:
            self.ml_ops = get_ml_common_operations()
            self.confidence_calibrator = ModelConfidenceCalibration()
        else:
            self.ml_ops = None
            self.confidence_calibrator = None
        
        # Initialize matrix operations
        if MATRIX_OPS_AVAILABLE:
            self.matrix_ops = create_matrix_operations()
            self.vectorized_ops = create_vectorized_operations()
            self.batch_ops = create_batch_operations()
        else:
            self.matrix_ops = None
            self.vectorized_ops = None
            self.batch_ops = None
        
        # Initialize unified framework if available
        if UNIFIED_FRAMEWORK_AVAILABLE:
            self.evaluator = UnifiedEvaluationFramework(
                architecture_type=ArchitectureType.TAS,
                config=EvaluationConfig()
            )
        else:
            self.evaluator = None
            self.logger.warning("Unified evaluation framework not available")
        
        # Setup M1 optimizations
        self._setup_m1_optimizations()
        
        # Initialize data quality monitoring
        self.data_quality_threshold = self.config.get('data_quality_threshold', 0.8)
        self.performance_monitoring = self.config.get('performance_monitoring', True)
        
        self.logger.info("✅ Analysis Components initialized successfully")
    
    def _setup_m1_optimizations(self):
        """Setup M1 hardware optimizations."""
        try:
            # Integrate with M1 optimizers
            integration_result = integrate_with_m1_optimizers()
            if integration_result.get('success', False):
                self.logger.info("🧠 M1 optimizations integrated successfully")
                
                # Start memory monitoring
                start_m1_memory_monitoring()
                
                # Optimize numpy operations
                self.cpu_optimizer.optimize_numpy_operations()
                
            else:
                self.logger.warning("⚠️ M1 optimizations not available")
                
        except Exception as e:
            self.logger.warning(f"⚠️ M1 optimization setup failed: {e}")
    
    def analyze_components(self, components: Any, X_test: np.ndarray, 
                         y_test: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Analyze components using comprehensive evaluation framework.
        
        Args:
            components: Analysis components to evaluate
            X_test: Test features
            y_test: Test labels
            **kwargs: Additional analysis parameters
            
        Returns:
            Dictionary containing analysis results
        """
        start_time = time.time()
        self.logger.info("🔍 Starting comprehensive component analysis")
        
        try:
            # Validate inputs
            self._validate_analysis_inputs(components, X_test, y_test)
            
            # Setup analysis context
            with memory_checkpoint("component_analysis"):
                with gpu_context("analysis_computation"):
                    # Perform analysis
                    results = self._perform_component_analysis(
                        components, X_test, y_test, **kwargs
                    )
            
            # Calculate performance metrics
            execution_time = time.time() - start_time
            results['execution_time'] = execution_time
            results['timestamp'] = datetime.now().isoformat()
            
            # Log results
            self._log_analysis_results(results)
            
            self.logger.info(f"✅ Component analysis completed in {execution_time:.3f}s")
            return results
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Component analysis failed after {execution_time:.3f}s: {e}")
            return {
                'error': str(e),
                'execution_time': execution_time,
                'success': False
            }
    
    def _validate_analysis_inputs(self, components: Any, X_test: np.ndarray, 
                                 y_test: np.ndarray) -> None:
        """Validate analysis inputs."""
        if components is None:
            raise ValueError("Components cannot be None")
        
        if X_test is None or len(X_test) == 0:
            raise ValueError("X_test cannot be None or empty")
        
        if y_test is None or len(y_test) == 0:
            raise ValueError("y_test cannot be None or empty")
        
        if len(X_test) != len(y_test):
            raise ValueError("X_test and y_test must have the same length")
        
        # Validate numeric arrays
        X_test = validate_numeric_array(X_test, "X_test")
        y_test = validate_numeric_array(y_test, "y_test")
        
        self.logger.debug(f"✅ Input validation passed - X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    def _perform_component_analysis(self, components: Any, X_test: np.ndarray,
                                   y_test: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Perform the actual component analysis."""
        results = {
            'success': True,
            'component_type': type(components).__name__,
            'data_shape': X_test.shape,
            'analysis_metrics': {}
        }
        
        # Basic component analysis
        if hasattr(components, 'predict'):
            # ML model analysis
            results.update(self._analyze_ml_model(components, X_test, y_test, **kwargs))
        elif hasattr(components, 'transform'):
            # Transformer analysis
            results.update(self._analyze_transformer(components, X_test, y_test, **kwargs))
        else:
            # Generic component analysis
            results.update(self._analyze_generic_component(components, X_test, y_test, **kwargs))
        
        # Add confidence metrics if applicable
        if 'predict_proba' in dir(components):
            confidence_metrics = self._calculate_confidence_metrics(components, X_test, y_test)
            results['confidence_metrics'] = confidence_metrics
        
        # Add performance metrics
        results['performance_metrics'] = self._calculate_performance_metrics(components, X_test, y_test)
        
        return results
    
    def _analyze_ml_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, 
                         **kwargs) -> Dict[str, Any]:
        """Analyze ML model components."""
        self.logger.info("🤖 Analyzing ML model component")
        
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate basic metrics
            accuracy = np.mean(y_pred == y_test)
            
            # Calculate additional metrics if available
            metrics = {
                'accuracy': float(accuracy),
                'predictions_count': len(y_pred),
                'unique_predictions': len(np.unique(y_pred))
            }
            
            # Add probability predictions if available
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
                metrics['has_probability_predictions'] = True
                metrics['probability_shape'] = y_pred_proba.shape
            else:
                metrics['has_probability_predictions'] = False
            
            return {
                'component_analysis': 'ml_model',
                'metrics': metrics,
                'predictions': y_pred.tolist() if len(y_pred) < 1000 else 'large_array'
            }
            
        except Exception as e:
            self.logger.error(f"❌ ML model analysis failed: {e}")
            return {
                'component_analysis': 'ml_model',
                'error': str(e),
                'metrics': {}
            }
    
    def _analyze_transformer(self, transformer: Any, X_test: np.ndarray, y_test: np.ndarray,
                            **kwargs) -> Dict[str, Any]:
        """Analyze transformer components."""
        self.logger.info("🔄 Analyzing transformer component")
        
        try:
            # Transform data
            X_transformed = transformer.transform(X_test)
            
            # Calculate transformation metrics
            metrics = {
                'input_shape': X_test.shape,
                'output_shape': X_transformed.shape,
                'transformation_ratio': X_transformed.shape[1] / X_test.shape[1],
                'data_type': str(X_transformed.dtype)
            }
            
            # Calculate data quality metrics
            if hasattr(X_transformed, 'isnull'):
                null_count = X_transformed.isnull().sum().sum()
                metrics['null_count'] = int(null_count)
                metrics['null_percentage'] = float(null_count / X_transformed.size * 100)
            
            return {
                'component_analysis': 'transformer',
                'metrics': metrics,
                'transformed_data_shape': X_transformed.shape
            }
            
        except Exception as e:
            self.logger.error(f"❌ Transformer analysis failed: {e}")
            return {
                'component_analysis': 'transformer',
                'error': str(e),
                'metrics': {}
            }
    
    def _analyze_generic_component(self, component: Any, X_test: np.ndarray, y_test: np.ndarray,
                                  **kwargs) -> Dict[str, Any]:
        """Analyze generic components."""
        self.logger.info("🔧 Analyzing generic component")
        
        try:
            # Basic component information
            component_info = {
                'type': type(component).__name__,
                'module': getattr(component, '__module__', 'unknown'),
                'attributes': [attr for attr in dir(component) if not attr.startswith('_')]
            }
            
            # Try to call common methods
            methods_tested = []
            for method_name in ['fit', 'predict', 'transform', 'score', 'evaluate']:
                if hasattr(component, method_name):
                    try:
                        method = getattr(component, method_name)
                        methods_tested.append(method_name)
                    except Exception:
                        pass
            
            return {
                'component_analysis': 'generic',
                'component_info': component_info,
                'methods_available': methods_tested,
                'metrics': {
                    'input_shape': X_test.shape,
                    'component_type': type(component).__name__
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Generic component analysis failed: {e}")
            return {
                'component_analysis': 'generic',
                'error': str(e),
                'metrics': {}
            }
    
    def _calculate_confidence_metrics(self, model: Any, X_test: np.ndarray, 
                                     y_test: np.ndarray) -> Dict[str, Any]:
        """Calculate confidence metrics for the model."""
        try:
            if not hasattr(model, 'predict_proba'):
                return {'error': 'Model does not support probability predictions'}
            
            if not ML_UTILS_AVAILABLE:
                return {'error': 'ML utilities not available for confidence metrics'}
            
            y_pred_proba = model.predict_proba(X_test)
            confidence_metrics = calculate_confidence_metrics(y_test, y_pred_proba)
            
            return confidence_metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Confidence metrics calculation failed: {e}")
            return {'error': str(e)}
    
    def _calculate_performance_metrics(self, component: Any, X_test: np.ndarray,
                                      y_test: np.ndarray) -> Dict[str, Any]:
        """Calculate performance metrics for the component."""
        try:
            start_time = time.time()
            
            # Test component performance
            if hasattr(component, 'predict'):
                _ = component.predict(X_test)
            elif hasattr(component, 'transform'):
                _ = component.transform(X_test)
            
            execution_time = time.time() - start_time
            
            # Get memory usage
            memory_usage = get_memory_usage()
            
            return {
                'execution_time': execution_time,
                'memory_usage': memory_usage,
                'throughput': len(X_test) / execution_time if execution_time > 0 else 0
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Performance metrics calculation failed: {e}")
            return {'error': str(e)}
    
    def _log_analysis_results(self, results: Dict[str, Any]) -> None:
        """Log analysis results in a structured format."""
        try:
            if results.get('success', False):
                self.logger.info("📊 Analysis Results Summary:")
                
                # Log basic metrics
                if 'metrics' in results:
                    metrics = results['metrics']
                    self.logger.info(f"  📈 Component Type: {results.get('component_type', 'Unknown')}")
                    if 'accuracy' in metrics:
                        self.logger.info(f"  📈 Accuracy: {metrics['accuracy']:.4f}")
                    if 'execution_time' in metrics:
                        self.logger.info(f"  ⏱️ Execution Time: {metrics['execution_time']:.3f}s")
                
                # Log confidence metrics
                if 'confidence_metrics' in results:
                    confidence = results['confidence_metrics']
                    if 'mean_confidence' in confidence:
                        self.logger.info(f"  🎯 Mean Confidence: {confidence['mean_confidence']:.3f}")
                    if 'calibration_quality' in confidence:
                        self.logger.info(f"  📊 Calibration Quality: {confidence['calibration_quality']}")
                
            else:
                self.logger.error(f"❌ Analysis failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to log analysis results: {e}")
    
    def analyze_dataframe(self, df: pd.DataFrame, analysis_type: str = 'comprehensive') -> Dict[str, Any]:
        """
        Analyze DataFrame using comprehensive data analysis.
        
        Args:
            df: DataFrame to analyze
            analysis_type: Type of analysis ('comprehensive', 'basic', 'quality')
            
        Returns:
            Dictionary containing analysis results
        """
        start_time = time.time()
        self.logger.info(f"📊 Starting {analysis_type} DataFrame analysis")
        
        try:
            # Validate DataFrame
            if not validate_dataframe(df):
                raise ValueError("Invalid DataFrame provided")
            
            results = {
                'success': True,
                'analysis_type': analysis_type,
                'dataframe_shape': df.shape,
                'timestamp': datetime.now().isoformat()
            }
            
            # Basic analysis
            if analysis_type in ['comprehensive', 'basic']:
                results['basic_info'] = get_dataframe_info(df)
                results['summary_statistics'] = create_summary_statistics(df)
            
            # Quality analysis
            if analysis_type in ['comprehensive', 'quality']:
                results['data_quality'] = create_data_quality_report(df)
                results['quality_metrics'] = calculate_data_quality_metrics(df)
            
            # Advanced analysis
            if analysis_type == 'comprehensive':
                results['advanced_analysis'] = self._perform_advanced_dataframe_analysis(df)
            
            # Optimize DataFrame for M1 if available
            if is_m1_available():
                optimized_df = optimize_dataframe_for_m1(df.copy())
                results['m1_optimization'] = {
                    'original_memory': df.memory_usage(deep=True).sum(),
                    'optimized_memory': optimized_df.memory_usage(deep=True).sum(),
                    'memory_saved': df.memory_usage(deep=True).sum() - optimized_df.memory_usage(deep=True).sum()
                }
            
            execution_time = time.time() - start_time
            results['execution_time'] = execution_time
            
            self.logger.info(f"✅ DataFrame analysis completed in {execution_time:.3f}s")
            return results
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ DataFrame analysis failed after {execution_time:.3f}s: {e}")
            return {
                'error': str(e),
                'execution_time': execution_time,
                'success': False
            }
    
    def _perform_advanced_dataframe_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform advanced DataFrame analysis."""
        try:
            analysis = {}
            
            # Correlation analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                analysis['correlation_matrix'] = {
                    'shape': corr_matrix.shape,
                    'high_correlations': self._find_high_correlations(corr_matrix),
                    'correlation_stats': {
                        'mean_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()),
                        'max_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()),
                        'min_correlation': float(corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min())
                    }
                }
            
            # Temporal analysis if datetime index
            if isinstance(df.index, pd.DatetimeIndex):
                analysis['temporal_analysis'] = self._analyze_temporal_patterns(df)
            
            # Outlier detection
            analysis['outlier_analysis'] = self._detect_outliers(df)
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"⚠️ Advanced DataFrame analysis failed: {e}")
            return {'error': str(e)}
    
    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find high correlations in correlation matrix."""
        high_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_corrs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(corr_value)
                    })
        return high_corrs
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns in the DataFrame."""
        try:
            # Basic temporal statistics
            time_span = df.index.max() - df.index.min()
            time_gaps = df.index.to_series().diff().dropna()
            
            return {
                'time_span_days': time_span.days,
                'total_periods': len(df),
                'avg_time_gap': str(time_gaps.mean()),
                'time_gap_std': str(time_gaps.std()),
                'missing_periods': self._detect_missing_periods(df)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _detect_missing_periods(self, df: pd.DataFrame) -> int:
        """Detect missing periods in temporal data."""
        try:
            if not isinstance(df.index, pd.DatetimeIndex):
                return 0
            
            # Create expected index
            freq = pd.infer_freq(df.index)
            if freq:
                expected_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
                missing_count = len(expected_index) - len(df)
                return max(0, missing_count)
            return 0
        except Exception:
            return 0
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in the DataFrame."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outlier_info = {}
            
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_info[col] = {
                    'outlier_count': len(outliers),
                    'outlier_percentage': len(outliers) / len(df) * 100,
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }
            
            return outlier_info
        except Exception as e:
            return {'error': str(e)}
    
    def optimize_for_m1(self, data: Any) -> Any:
        """
        Optimize data for M1 hardware.
        
        Args:
            data: Data to optimize
            
        Returns:
            Optimized data
        """
        try:
            if isinstance(data, pd.DataFrame):
                return optimize_dataframe_for_m1(data)
            elif isinstance(data, np.ndarray):
                return create_m1_optimized_array(data)
            else:
                return data
        except Exception as e:
            self.logger.warning(f"⚠️ M1 optimization failed: {e}")
            return data
    
    def save_analysis_results(self, results: Dict[str, Any], filepath: str, 
                             format: str = 'auto') -> bool:
        """
        Save analysis results to file.
        
        Args:
            results: Analysis results to save
            filepath: Path to save the results
            format: File format ('auto', 'json', 'pickle', 'parquet')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            ensure_directory(Path(filepath).parent)
            
            # Save using universal serializer
            success = self.serializer.save(results, filepath, format)
            
            if success:
                self.logger.info(f"💾 Analysis results saved to {filepath}")
            else:
                self.logger.error(f"❌ Failed to save analysis results to {filepath}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Error saving analysis results: {e}")
            return False
    
    def load_analysis_results(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load analysis results from file.
        
        Args:
            filepath: Path to the results file
            
        Returns:
            Loaded results or None if failed
        """
        try:
            if not safe_file_exists(filepath):
                self.logger.warning(f"⚠️ Results file does not exist: {filepath}")
                return None
            
            results = self.serializer.load(filepath)
            
            if results:
                self.logger.info(f"📂 Analysis results loaded from {filepath}")
            else:
                self.logger.error(f"❌ Failed to load analysis results from {filepath}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error loading analysis results: {e}")
            return None
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        try:
            system_info = {
                'timestamp': datetime.now().isoformat(),
                'm1_available': is_m1_available(),
                'mps_available': is_mps_available(),
                'memory_usage': get_memory_usage(),
                'cpu_info': self.cpu_optimizer.get_cpu_info(),
                'gpu_info': self.gpu_manager.get_gpu_info(),
                'unified_framework_available': UNIFIED_FRAMEWORK_AVAILABLE,
                'ml_utils_available': ML_UTILS_AVAILABLE,
                'matrix_ops_available': MATRIX_OPS_AVAILABLE,
                'data_utils_available': DATA_UTILS_AVAILABLE
            }
            
            return system_info
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get system info: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """Cleanup resources and stop monitoring."""
        try:
            # Stop M1 memory monitoring
            stop_m1_memory_monitoring()
            
            # Cleanup M1 optimizers
            cleanup_m1_optimizers()
            
            # Cleanup tprint
            cleanup_tprint()
            
            self.logger.info("🧹 Analysis Components cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Cleanup failed: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# Convenience functions for easy usage
def create_analysis_components(config: Optional[Dict[str, Any]] = None) -> AnalysisComponents:
    """Create and return AnalysisComponents instance."""
    return AnalysisComponents(config)


def analyze_components(components: Any, X_test: np.ndarray, y_test: np.ndarray,
                      config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Convenience function for component analysis."""
    with AnalysisComponents(config) as analyzer:
        return analyzer.analyze_components(components, X_test, y_test, **kwargs)


def analyze_dataframe(df: pd.DataFrame, analysis_type: str = 'comprehensive',
                     config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convenience function for DataFrame analysis."""
    with AnalysisComponents(config) as analyzer:
        return analyzer.analyze_dataframe(df, analysis_type)


# Export main classes and functions
__all__ = [
    'AnalysisComponents',
    'create_analysis_components',
    'analyze_components',
    'analyze_dataframe',
    'UNIFIED_FRAMEWORK_AVAILABLE',
    'ML_UTILS_AVAILABLE',
    'MATRIX_OPS_AVAILABLE',
    'DATA_UTILS_AVAILABLE'
]