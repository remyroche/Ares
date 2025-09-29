"""
Architecture Evaluator

This module provides comprehensive architecture evaluation for Neural Architecture Search
and Trading Architecture Search with extensive integration of utility modules
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
    safe_rename_columns as cu_rename_columns,
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

# Import matrix operations
from ...matrix_operations.unified_operations import MatrixOperations
from ...matrix_operations.enhanced_operations import EnhancedMatrixOperations
from ...matrix_operations.batch_operations import BatchMatrixOperations
from ...matrix_operations.vectorized_core import VectorizedCore
from ...matrix_operations.convenience import MatrixConvenience

# Import hardware utilities
from ...hardware.m1_gpu_utils import M1GPUManager, is_m1_available, is_mps_available
from ...hardware.m1_memory_optimizer import M1MemoryOptimizer
from ...hardware.m1_cpu_optimizer import M1CPUOptimizer

# Setup logging with tprint integration
logger = logging.getLogger(__name__)

@tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
class ArchitectureEvaluator:
    """
    Architecture Evaluator with extensive utility integration.
    
    This evaluator provides comprehensive architecture evaluation capabilities with:
    - Extensive use of common operations for data processing
    - Math validation for safe computations
    - Comprehensive logging with tprint
    - Data management with klines parquet utilities
    - Serialization for evaluation persistence
    - M1 hardware optimization
    - Matrix operations for high-performance computations
    - Multiple evaluation metrics and methods
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Architecture Evaluator with extensive utility integration.
        
        Args:
            config: Configuration dictionary for evaluator
        """
        tprint_info("🚀 Initializing Architecture Evaluator with extensive utility integration")
        
        # Initialize configuration
        self.config = config or {}
        self.logger = logger.getChild("ArchitectureEvaluator")
        
        # Initialize utility classes
        tprint_debug("🔧 Initializing utility classes")
        self.common_ops = CommonUtilities()
        self.math_validator = MathValidation()
        self.klines_manager = get_klines_manager()
        self.serializer = UniversalSerializer()
        
        # Initialize matrix operations
        tprint_debug("🔧 Initializing matrix operations")
        self.matrix_ops = MatrixOperations()
        self.enhanced_matrix_ops = EnhancedMatrixOperations()
        self.batch_matrix_ops = BatchMatrixOperations()
        self.vectorized_core = VectorizedCore()
        self.matrix_convenience = MatrixConvenience()
        
        # Initialize M1 hardware optimizations
        tprint_debug("🔧 Initializing M1 hardware optimizations")
        self.m1_integration = integrate_with_m1_optimizers()
        if self.m1_integration['success']:
            tprint_success("✅ M1 integration successful")
            self.gpu_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()
        else:
            tprint_warning("⚠️ M1 integration failed, using fallback")
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
        
        # Initialize evaluation state
        self.evaluation_history = []
        self.performance_metrics = {}
        self.best_architecture = None
        self.best_score = -np.inf
        
        tprint_success("✅ Architecture Evaluator initialized successfully")
    
    @tprint_timer("Architecture Evaluation")
    def evaluate_architecture(
        self,
        data: pd.DataFrame,
        architecture_params: Dict[str, Any],
        evaluation_method: str = "comprehensive",
        validation_split: float = 0.2,
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """Evaluate architecture using extensive utility integration.
        
        Args:
            data: Input data for evaluation
            architecture_params: Architecture parameters to evaluate
            evaluation_method: Method of evaluation (comprehensive, basic, performance)
            validation_split: Fraction of data to use for validation
            metrics: List of metrics to calculate
            
        Returns:
            Dictionary with evaluation results
        """
        tprint_info(f"🔍 Evaluating architecture with method: {evaluation_method}")
        
        try:
            # Validate input data
            if not validate_dataframe_columns(data, ['open', 'high', 'low', 'close', 'volume']):
                tprint_error("❌ Invalid data columns for architecture evaluation")
                return {}
            
            # Default metrics
            if metrics is None:
                metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'sharpe_ratio', 'max_drawdown']
            
            # Split data for validation
            with memory_checkpoint("data_splitting"):
                train_data, val_data = self._split_data(data, validation_split)
            
            if train_data is None or val_data is None:
                tprint_error("❌ Data splitting failed")
                return {}
            
            tprint_info(f"📊 Training data: {len(train_data)} records, Validation data: {len(val_data)} records")
            
            # Initialize evaluation results
            evaluation_results = {
                'architecture_params': architecture_params,
                'evaluation_method': evaluation_method,
                'metrics': {},
                'performance_metrics': {},
                'evaluation_time': 0,
                'data_quality': {},
                'validation_results': {}
            }
            
            start_time = time.time()
            
            # Use M1 GPU context if available
            with gpu_context("architecture_evaluation") if self.gpu_manager else memory_checkpoint("architecture_evaluation"):
                
                # Evaluate architecture based on method
                if evaluation_method == "comprehensive":
                    results = self._comprehensive_evaluation(train_data, val_data, architecture_params, metrics)
                elif evaluation_method == "basic":
                    results = self._basic_evaluation(train_data, val_data, architecture_params, metrics)
                elif evaluation_method == "performance":
                    results = self._performance_evaluation(train_data, val_data, architecture_params, metrics)
                else:
                    tprint_error(f"❌ Unknown evaluation method: {evaluation_method}")
                    return {}
                
                evaluation_results.update(results)
            
            evaluation_time = time.time() - start_time
            evaluation_results['evaluation_time'] = evaluation_time
            
            # Calculate data quality metrics
            evaluation_results['data_quality'] = calculate_data_quality_metrics(data)
            
            # Store evaluation history
            self.evaluation_history.append({
                'timestamp': time.time(),
                'architecture_params': architecture_params,
                'evaluation_method': evaluation_method,
                'results': evaluation_results
            })
            
            tprint_success(f"✅ Architecture evaluation completed in {evaluation_time:.2f}s")
            tprint_info(f"🏆 Best score: {evaluation_results.get('overall_score', 0.0):.4f}")
            
            return evaluation_results
            
        except Exception as e:
            tprint_error(f"❌ Error in architecture evaluation: {e}")
            self.logger.exception("Architecture evaluation error")
            return {}
    
    def _split_data(self, data: pd.DataFrame, validation_split: float) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Split data into training and validation sets using safe operations."""
        try:
            tprint_debug(f"🔧 Splitting data with validation split: {validation_split}")
            
            # Validate input
            if not validate_dataframe_columns(data, ['open', 'high', 'low', 'close', 'volume']):
                tprint_error("❌ Invalid data for splitting")
                return None, None
            
            # Calculate split index
            total_size = len(data)
            val_size = int(total_size * validation_split)
            train_size = total_size - val_size
            
            if train_size <= 0 or val_size <= 0:
                tprint_error("❌ Invalid split sizes")
                return None, None
            
            # Split data safely
            train_data = safe_copy(data.iloc[:train_size])
            val_data = safe_copy(data.iloc[train_size:])
            
            # Validate split data
            if train_data.empty or val_data.empty:
                tprint_error("❌ Empty data after splitting")
                return None, None
            
            tprint_info(f"📊 Split: {len(train_data)} train, {len(val_data)} validation")
            return train_data, val_data
            
        except Exception as e:
            tprint_error(f"❌ Error splitting data: {e}")
            return None, None
    
    def _comprehensive_evaluation(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        architecture_params: Dict[str, Any],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Perform comprehensive architecture evaluation."""
        tprint_debug("🔍 Performing comprehensive evaluation")
        
        try:
            # Create feature matrices
            with memory_checkpoint("feature_matrix_creation"):
                train_features = self._create_evaluation_feature_matrix(train_data)
                val_features = self._create_evaluation_feature_matrix(val_data)
            
            # Validate feature matrices
            if not validate_correlation_matrix(train_features) or not validate_correlation_matrix(val_features):
                tprint_warning("⚠️ Invalid feature matrix correlation structure")
                return {}
            
            # Calculate all metrics
            evaluation_metrics = {}
            
            for metric in metrics:
                tprint_debug(f"🔧 Calculating metric: {metric}")
                
                if metric == 'accuracy':
                    evaluation_metrics[metric] = self._calculate_accuracy(train_features, val_features, architecture_params)
                elif metric == 'precision':
                    evaluation_metrics[metric] = self._calculate_precision(train_features, val_features, architecture_params)
                elif metric == 'recall':
                    evaluation_metrics[metric] = self._calculate_recall(train_features, val_features, architecture_params)
                elif metric == 'f1_score':
                    evaluation_metrics[metric] = self._calculate_f1_score(train_features, val_features, architecture_params)
                elif metric == 'sharpe_ratio':
                    evaluation_metrics[metric] = self._calculate_sharpe_ratio(train_features, val_features, architecture_params)
                elif metric == 'max_drawdown':
                    evaluation_metrics[metric] = self._calculate_max_drawdown(train_features, val_features, architecture_params)
                elif metric == 'information_ratio':
                    evaluation_metrics[metric] = self._calculate_information_ratio(train_features, val_features, architecture_params)
                elif metric == 'calmar_ratio':
                    evaluation_metrics[metric] = self._calculate_calmar_ratio(train_features, val_features, architecture_params)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(evaluation_metrics)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(train_features, val_features, architecture_params)
            
            return {
                'metrics': evaluation_metrics,
                'overall_score': overall_score,
                'performance_metrics': performance_metrics,
                'validation_results': {
                    'train_score': self._calculate_train_score(train_features, architecture_params),
                    'val_score': self._calculate_val_score(val_features, architecture_params)
                }
            }
            
        except Exception as e:
            tprint_error(f"❌ Error in comprehensive evaluation: {e}")
            return {}
    
    def _basic_evaluation(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        architecture_params: Dict[str, Any],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Perform basic architecture evaluation."""
        tprint_debug("🔍 Performing basic evaluation")
        
        try:
            # Create feature matrices
            with memory_checkpoint("feature_matrix_creation"):
                train_features = self._create_evaluation_feature_matrix(train_data)
                val_features = self._create_evaluation_feature_matrix(val_data)
            
            # Calculate basic metrics
            evaluation_metrics = {}
            
            for metric in metrics:
                if metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                    evaluation_metrics[metric] = self._calculate_basic_metric(
                        train_features, val_features, architecture_params, metric
                    )
            
            # Calculate overall score
            overall_score = safe_mean(np.array(list(evaluation_metrics.values())))
            
            return {
                'metrics': evaluation_metrics,
                'overall_score': overall_score,
                'performance_metrics': {},
                'validation_results': {
                    'train_score': self._calculate_train_score(train_features, architecture_params),
                    'val_score': self._calculate_val_score(val_features, architecture_params)
                }
            }
            
        except Exception as e:
            tprint_error(f"❌ Error in basic evaluation: {e}")
            return {}
    
    def _performance_evaluation(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        architecture_params: Dict[str, Any],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Perform performance-focused architecture evaluation."""
        tprint_debug("🔍 Performing performance evaluation")
        
        try:
            # Create feature matrices
            with memory_checkpoint("feature_matrix_creation"):
                train_features = self._create_evaluation_feature_matrix(train_data)
                val_features = self._create_evaluation_feature_matrix(val_data)
            
            # Calculate performance metrics
            evaluation_metrics = {}
            
            for metric in metrics:
                if metric in ['sharpe_ratio', 'max_drawdown', 'information_ratio', 'calmar_ratio']:
                    evaluation_metrics[metric] = self._calculate_performance_metric(
                        train_features, val_features, architecture_params, metric
                    )
            
            # Calculate overall score
            overall_score = safe_mean(np.array(list(evaluation_metrics.values())))
            
            return {
                'metrics': evaluation_metrics,
                'overall_score': overall_score,
                'performance_metrics': evaluation_metrics,
                'validation_results': {
                    'train_score': self._calculate_train_score(train_features, architecture_params),
                    'val_score': self._calculate_val_score(val_features, architecture_params)
                }
            }
            
        except Exception as e:
            tprint_error(f"❌ Error in performance evaluation: {e}")
            return {}
    
    def _create_evaluation_feature_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """Create evaluation feature matrix using matrix operations utilities."""
        try:
            # Extract numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            feature_data = data[numeric_cols].values
            
            # Handle NaN values
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Validate numeric array
            feature_data = validate_numeric_array(feature_data, "evaluation_features")
            
            # Use matrix operations for feature engineering
            # Normalize features
            normalized_features = self.matrix_ops.normalize_matrix(feature_data)
            
            # Add polynomial features
            polynomial_features = self.enhanced_matrix_ops.add_polynomial_features(
                normalized_features, degree=2
            )
            
            # Add evaluation-specific features
            evaluation_features = self.matrix_convenience.add_evaluation_features(
                polynomial_features
            )
            
            return evaluation_features
            
        except Exception as e:
            tprint_error(f"❌ Error creating evaluation feature matrix: {e}")
            return np.array([])
    
    def _calculate_accuracy(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray,
        architecture_params: Dict[str, Any]
    ) -> float:
        """Calculate accuracy metric using matrix operations."""
        try:
            # Simulate accuracy calculation
            base_accuracy = self.vectorized_core.compute_accuracy_metric(
                train_features, val_features, architecture_params
            )
            
            # Apply architecture parameter adjustments
            complexity = architecture_params.get('complexity', 1.0)
            depth = architecture_params.get('depth', 1)
            
            complexity_factor = safe_power(complexity, 0.3)
            depth_factor = safe_log(depth + 1)
            
            adjusted_accuracy = safe_weighted_average(
                [base_accuracy, complexity_factor, depth_factor],
                [0.7, 0.2, 0.1]
            )
            
            return validate_finite(adjusted_accuracy, "accuracy")
            
        except Exception as e:
            tprint_error(f"❌ Error calculating accuracy: {e}")
            return 0.0
    
    def _calculate_precision(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray,
        architecture_params: Dict[str, Any]
    ) -> float:
        """Calculate precision metric using matrix operations."""
        try:
            # Simulate precision calculation
            base_precision = self.vectorized_core.compute_precision_metric(
                train_features, val_features, architecture_params
            )
            
            return validate_finite(base_precision, "precision")
            
        except Exception as e:
            tprint_error(f"❌ Error calculating precision: {e}")
            return 0.0
    
    def _calculate_recall(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray,
        architecture_params: Dict[str, Any]
    ) -> float:
        """Calculate recall metric using matrix operations."""
        try:
            # Simulate recall calculation
            base_recall = self.vectorized_core.compute_recall_metric(
                train_features, val_features, architecture_params
            )
            
            return validate_finite(base_recall, "recall")
            
        except Exception as e:
            tprint_error(f"❌ Error calculating recall: {e}")
            return 0.0
    
    def _calculate_f1_score(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray,
        architecture_params: Dict[str, Any]
    ) -> float:
        """Calculate F1 score metric using matrix operations."""
        try:
            # Get precision and recall
            precision = self._calculate_precision(train_features, val_features, architecture_params)
            recall = self._calculate_recall(train_features, val_features, architecture_params)
            
            # Calculate F1 score
            f1_score = safe_divide(2 * precision * recall, precision + recall, default=0.0)
            
            return validate_finite(f1_score, "f1_score")
            
        except Exception as e:
            tprint_error(f"❌ Error calculating F1 score: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray,
        architecture_params: Dict[str, Any]
    ) -> float:
        """Calculate Sharpe ratio using matrix operations."""
        try:
            # Simulate returns calculation
            returns = self.vectorized_core.compute_returns(train_features, val_features)
            
            # Calculate Sharpe ratio
            mean_return = safe_mean(returns)
            std_return = safe_std(returns)
            
            if std_return == 0:
                return 0.0
            
            sharpe_ratio = safe_divide(mean_return, std_return)
            
            return validate_finite(sharpe_ratio, "sharpe_ratio")
            
        except Exception as e:
            tprint_error(f"❌ Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_max_drawdown(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray,
        architecture_params: Dict[str, Any]
    ) -> float:
        """Calculate maximum drawdown using matrix operations."""
        try:
            # Simulate cumulative returns calculation
            cumulative_returns = self.vectorized_core.compute_cumulative_returns(
                train_features, val_features
            )
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(cumulative_returns)
            
            # Calculate drawdown
            drawdown = running_max - cumulative_returns
            
            # Calculate maximum drawdown
            max_drawdown = np.max(drawdown)
            
            return validate_finite(max_drawdown, "max_drawdown")
            
        except Exception as e:
            tprint_error(f"❌ Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_information_ratio(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray,
        architecture_params: Dict[str, Any]
    ) -> float:
        """Calculate information ratio using matrix operations."""
        try:
            # Simulate strategy and benchmark returns
            strategy_returns = self.vectorized_core.compute_strategy_returns(
                train_features, val_features, architecture_params
            )
            benchmark_returns = self.vectorized_core.compute_benchmark_returns(
                train_features, val_features
            )
            
            # Calculate excess returns
            excess_returns = strategy_returns - benchmark_returns
            
            # Calculate information ratio
            mean_excess = safe_mean(excess_returns)
            std_excess = safe_std(excess_returns)
            
            if std_excess == 0:
                return 0.0
            
            information_ratio = safe_divide(mean_excess, std_excess)
            
            return validate_finite(information_ratio, "information_ratio")
            
        except Exception as e:
            tprint_error(f"❌ Error calculating information ratio: {e}")
            return 0.0
    
    def _calculate_calmar_ratio(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray,
        architecture_params: Dict[str, Any]
    ) -> float:
        """Calculate Calmar ratio using matrix operations."""
        try:
            # Get annualized return and max drawdown
            annualized_return = self._calculate_annualized_return(train_features, val_features, architecture_params)
            max_drawdown = self._calculate_max_drawdown(train_features, val_features, architecture_params)
            
            if max_drawdown == 0:
                return 0.0
            
            calmar_ratio = safe_divide(annualized_return, max_drawdown)
            
            return validate_finite(calmar_ratio, "calmar_ratio")
            
        except Exception as e:
            tprint_error(f"❌ Error calculating Calmar ratio: {e}")
            return 0.0
    
    def _calculate_annualized_return(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray,
        architecture_params: Dict[str, Any]
    ) -> float:
        """Calculate annualized return using matrix operations."""
        try:
            # Simulate returns calculation
            returns = self.vectorized_core.compute_returns(train_features, val_features)
            
            # Calculate annualized return
            mean_return = safe_mean(returns)
            annualized_return = mean_return * 252  # Assuming daily returns
            
            return validate_finite(annualized_return, "annualized_return")
            
        except Exception as e:
            tprint_error(f"❌ Error calculating annualized return: {e}")
            return 0.0
    
    def _calculate_basic_metric(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray,
        architecture_params: Dict[str, Any],
        metric: str
    ) -> float:
        """Calculate basic metric using matrix operations."""
        try:
            # Use vectorized operations for basic metrics
            base_score = self.vectorized_core.compute_basic_metric(
                train_features, val_features, architecture_params, metric
            )
            
            return validate_finite(base_score, metric)
            
        except Exception as e:
            tprint_error(f"❌ Error calculating {metric}: {e}")
            return 0.0
    
    def _calculate_performance_metric(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray,
        architecture_params: Dict[str, Any],
        metric: str
    ) -> float:
        """Calculate performance metric using matrix operations."""
        try:
            # Use vectorized operations for performance metrics
            base_score = self.vectorized_core.compute_performance_metric(
                train_features, val_features, architecture_params, metric
            )
            
            return validate_finite(base_score, metric)
            
        except Exception as e:
            tprint_error(f"❌ Error calculating {metric}: {e}")
            return 0.0
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall score using math validation utilities."""
        try:
            if not metrics:
                return 0.0
            
            # Extract scores
            scores = list(metrics.values())
            scores_array = np.array(scores)
            
            # Calculate weighted average
            weights = [1.0] * len(scores)  # Equal weights for now
            overall_score = safe_weighted_average(scores_array, weights)
            
            return validate_finite(overall_score, "overall_score")
            
        except Exception as e:
            tprint_error(f"❌ Error calculating overall score: {e}")
            return 0.0
    
    def _calculate_performance_metrics(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray,
        architecture_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate performance metrics using math validation utilities."""
        try:
            # Calculate various performance metrics
            performance_metrics = {
                'train_score': self._calculate_train_score(train_features, architecture_params),
                'val_score': self._calculate_val_score(val_features, architecture_params),
                'overfitting_metric': self._calculate_overfitting_metric(train_features, val_features, architecture_params),
                'stability_metric': self._calculate_stability_metric(train_features, val_features, architecture_params),
                'efficiency_metric': self._calculate_efficiency_metric(train_features, val_features, architecture_params)
            }
            
            return performance_metrics
            
        except Exception as e:
            tprint_error(f"❌ Error calculating performance metrics: {e}")
            return {}
    
    def _calculate_train_score(self, train_features: np.ndarray, architecture_params: Dict[str, Any]) -> float:
        """Calculate training score using matrix operations."""
        try:
            score = self.vectorized_core.compute_train_score(train_features, architecture_params)
            return validate_finite(score, "train_score")
        except Exception as e:
            tprint_error(f"❌ Error calculating train score: {e}")
            return 0.0
    
    def _calculate_val_score(self, val_features: np.ndarray, architecture_params: Dict[str, Any]) -> float:
        """Calculate validation score using matrix operations."""
        try:
            score = self.vectorized_core.compute_val_score(val_features, architecture_params)
            return validate_finite(score, "val_score")
        except Exception as e:
            tprint_error(f"❌ Error calculating val score: {e}")
            return 0.0
    
    def _calculate_overfitting_metric(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray,
        architecture_params: Dict[str, Any]
    ) -> float:
        """Calculate overfitting metric using math validation utilities."""
        try:
            train_score = self._calculate_train_score(train_features, architecture_params)
            val_score = self._calculate_val_score(val_features, architecture_params)
            
            # Calculate overfitting as the difference between train and val scores
            overfitting = train_score - val_score
            
            return validate_finite(overfitting, "overfitting_metric")
            
        except Exception as e:
            tprint_error(f"❌ Error calculating overfitting metric: {e}")
            return 0.0
    
    def _calculate_stability_metric(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray,
        architecture_params: Dict[str, Any]
    ) -> float:
        """Calculate stability metric using math validation utilities."""
        try:
            # Calculate stability as the consistency of performance
            train_score = self._calculate_train_score(train_features, architecture_params)
            val_score = self._calculate_val_score(val_features, architecture_params)
            
            # Stability is higher when train and val scores are close
            stability = 1.0 - abs(train_score - val_score)
            
            return validate_finite(stability, "stability_metric")
            
        except Exception as e:
            tprint_error(f"❌ Error calculating stability metric: {e}")
            return 0.0
    
    def _calculate_efficiency_metric(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray,
        architecture_params: Dict[str, Any]
    ) -> float:
        """Calculate efficiency metric using math validation utilities."""
        try:
            # Calculate efficiency as the ratio of performance to complexity
            performance = self._calculate_val_score(val_features, architecture_params)
            complexity = architecture_params.get('complexity', 1.0)
            
            efficiency = safe_divide(performance, complexity)
            
            return validate_finite(efficiency, "efficiency_metric")
            
        except Exception as e:
            tprint_error(f"❌ Error calculating efficiency metric: {e}")
            return 0.0
    
    @tprint_timer("Results Serialization")
    def save_results(
        self,
        results: Dict[str, Any],
        filepath: str
    ) -> bool:
        """Save evaluation results using serialization utilities.
        
        Args:
            results: Evaluation results to save
            filepath: Path to save results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            tprint_info(f"💾 Saving evaluation results to {filepath}")
            
            # Add metadata
            results_with_metadata = {
                'results': results,
                'metadata': {
                    'timestamp': time.time(),
                    'evaluator_version': '1.0.0',
                    'm1_integration': self.m1_integration,
                    'memory_usage': get_memory_usage(),
                    'evaluation_history_length': len(self.evaluation_history)
                }
            }
            
            # Save using universal serializer
            success = self.serializer.save(results_with_metadata, filepath)
            
            if success:
                tprint_success(f"✅ Evaluation results saved successfully to {filepath}")
            else:
                tprint_error(f"❌ Failed to save evaluation results to {filepath}")
            
            return success
            
        except Exception as e:
            tprint_error(f"❌ Error saving evaluation results: {e}")
            return False
    
    def load_results(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Load evaluation results using serialization utilities.
        
        Args:
            filepath: Path to load results from
            
        Returns:
            Loaded results or None if loading fails
        """
        try:
            tprint_info(f"📂 Loading evaluation results from {filepath}")
            
            # Load using universal serializer
            results = self.serializer.load(filepath)
            
            if results:
                tprint_success(f"✅ Evaluation results loaded successfully from {filepath}")
                return results
            else:
                tprint_error(f"❌ Failed to load evaluation results from {filepath}")
                return None
                
        except Exception as e:
            tprint_error(f"❌ Error loading evaluation results: {e}")
            return None
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get comprehensive evaluation summary using utility integration."""
        try:
            summary = {
                'evaluation_history_length': len(self.evaluation_history),
                'performance_metrics_count': len(self.performance_metrics),
                'm1_integration': self.m1_integration,
                'memory_usage': get_memory_usage(),
                'best_architecture': self.best_architecture,
                'best_score': self.best_score,
                'recent_evaluations': self.evaluation_history[-5:] if self.evaluation_history else []
            }
            
            return summary
            
        except Exception as e:
            tprint_error(f"❌ Error getting evaluation summary: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup resources and M1 optimizations."""
        try:
            tprint_info("🧹 Cleaning up Architecture Evaluator resources")
            
            # Cleanup M1 optimizers
            cleanup_m1_optimizers()
            
            # Clear evaluation history
            self.evaluation_history.clear()
            self.performance_metrics.clear()
            
            tprint_success("✅ Architecture Evaluator cleanup completed")
            
        except Exception as e:
            tprint_error(f"❌ Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()


# Convenience function for quick evaluator usage
def create_architecture_evaluator(config: Optional[Dict[str, Any]] = None) -> ArchitectureEvaluator:
    """Create an Architecture Evaluator instance with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured ArchitectureEvaluator instance
    """
    return ArchitectureEvaluator(config)


# Example usage
if __name__ == "__main__":
    # Configure tprint for better output
    from ...tprint import TPrintConfig, configure_tprint
    
    config = TPrintConfig(
        use_colors=True,
        output_to_console=True,
        enable_structured_logging=True
    )
    configure_tprint(config)
    
    # Create and use evaluator
    with create_architecture_evaluator() as evaluator:
        # Load sample data
        data = evaluator.klines_manager.read_data("ETHUSDT", "1m")
        
        if data is not None:
            # Define architecture parameters
            architecture_params = {
                'complexity': 2.0,
                'depth': 3,
                'width': 64,
                'activation': 'relu'
            }
            
            # Perform evaluation
            results = evaluator.evaluate_architecture(
                data=data,
                architecture_params=architecture_params,
                evaluation_method="comprehensive",
                metrics=['accuracy', 'precision', 'recall', 'f1_score', 'sharpe_ratio', 'max_drawdown']
            )
            
            # Save results
            if results:
                evaluator.save_results(results, "architecture_evaluation_results.json")
                tprint_structured(results, LogLevel.INFO)