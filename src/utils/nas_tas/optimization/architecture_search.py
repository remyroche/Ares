"""
Architecture Search Optimizer

This module provides comprehensive architecture search optimization with extensive
integration of utility modules for optimal performance, data processing, and hardware optimization.
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

# Import ML common optimization utilities
from ..ml_common.optimization.bayesian_entry_timing_optimizer import BayesianEntryTimingOptimizer
from ..ml_common.optimization.grid_utils import GridSearchOptimizer
from ..ml_common.optimization.hpo_utils import HPOUtils
from ..ml_common.optimization.hierarchical_hpo import HierarchicalHPO

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
class ArchitectureSearchOptimizer:
    """
    Architecture Search Optimizer with extensive utility integration.
    
    This optimizer provides comprehensive architecture search capabilities with:
    - Extensive use of common operations for data processing
    - Math validation for safe computations
    - Comprehensive logging with tprint
    - Data management with klines parquet utilities
    - Serialization for model persistence
    - M1 hardware optimization
    - Matrix operations for high-performance computations
    - Advanced optimization algorithms (Grid + Bayesian TPE)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Architecture Search Optimizer with extensive utility integration.
        
        Args:
            config: Configuration dictionary for optimizer
        """
        tprint_info("🚀 Initializing Architecture Search Optimizer with extensive utility integration")
        
        # Initialize configuration
        self.config = config or {}
        self.logger = logger.getChild("ArchitectureSearchOptimizer")
        
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
        
        # Initialize optimization components
        tprint_debug("🔧 Initializing optimization components")
        self.bayesian_optimizer = BayesianEntryTimingOptimizer()
        self.grid_optimizer = GridSearchOptimizer()
        self.hpo_utils = HPOUtils()
        self.hierarchical_hpo = HierarchicalHPO()
        
        # Initialize search state
        self.search_history = []
        self.performance_metrics = {}
        self.best_architecture = None
        self.best_score = -np.inf
        
        tprint_success("✅ Architecture Search Optimizer initialized successfully")
    
    @tprint_timer("Architecture Search")
    def optimize_architecture(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        optimization_method: str = "bayesian_tpe",
        n_trials: int = 100,
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """Optimize architecture using extensive utility integration.
        
        Args:
            data: Input data for architecture optimization
            search_space: Architecture search space
            optimization_method: Optimization method (bayesian_tpe, grid, hierarchical)
            n_trials: Number of optimization trials
            validation_split: Fraction of data to use for validation
            
        Returns:
            Dictionary with optimization results and best architecture
        """
        tprint_info(f"🔍 Starting architecture optimization with {optimization_method}")
        
        try:
            # Validate input data
            if not validate_dataframe_columns(data, ['open', 'high', 'low', 'close', 'volume']):
                tprint_error("❌ Invalid data columns for architecture optimization")
                return {}
            
            # Calculate data quality metrics
            quality_metrics = calculate_data_quality_metrics(data)
            tprint_info(f"📈 Data quality metrics: {quality_metrics}")
            
            # Split data for validation
            with memory_checkpoint("data_splitting"):
                train_data, val_data = self._split_data(data, validation_split)
            
            if train_data is None or val_data is None:
                tprint_error("❌ Data splitting failed")
                return {}
            
            tprint_info(f"📊 Training data: {len(train_data)} records, Validation data: {len(val_data)} records")
            
            # Initialize optimization results
            optimization_results = {
                'method': optimization_method,
                'n_trials': n_trials,
                'trials': [],
                'best_architecture': None,
                'best_score': -np.inf,
                'optimization_time': 0,
                'performance_metrics': {},
                'data_quality': quality_metrics
            }
            
            start_time = time.time()
            
            # Use M1 GPU context if available
            with gpu_context("architecture_optimization") if self.gpu_manager else memory_checkpoint("architecture_optimization"):
                
                if optimization_method == "bayesian_tpe":
                    tprint_info("🧠 Using Bayesian TPE optimization")
                    best_architecture, best_score, trials = self._bayesian_optimization(
                        train_data, val_data, search_space, n_trials
                    )
                elif optimization_method == "grid":
                    tprint_info("🔧 Using Grid Search optimization")
                    best_architecture, best_score, trials = self._grid_optimization(
                        train_data, val_data, search_space, n_trials
                    )
                elif optimization_method == "hierarchical":
                    tprint_info("🏗️ Using Hierarchical HPO optimization")
                    best_architecture, best_score, trials = self._hierarchical_optimization(
                        train_data, val_data, search_space, n_trials
                    )
                else:
                    tprint_error(f"❌ Unknown optimization method: {optimization_method}")
                    return {}
                
                optimization_results.update({
                    'best_architecture': best_architecture,
                    'best_score': best_score,
                    'trials': trials
                })
            
            optimization_time = time.time() - start_time
            optimization_results['optimization_time'] = optimization_time
            
            # Calculate performance metrics
            optimization_results['performance_metrics'] = self._calculate_optimization_metrics(trials)
            
            # Update internal state
            self.best_architecture = best_architecture
            self.best_score = best_score
            self.search_history.extend(trials)
            
            tprint_success(f"✅ Architecture optimization completed in {optimization_time:.2f}s")
            tprint_info(f"🏆 Best score: {best_score:.4f}")
            
            return optimization_results
            
        except Exception as e:
            tprint_error(f"❌ Error in architecture optimization: {e}")
            self.logger.exception("Architecture optimization error")
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
    
    def _bayesian_optimization(
        self, 
        train_data: pd.DataFrame, 
        val_data: pd.DataFrame,
        search_space: Dict[str, Any], 
        n_trials: int
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """Perform Bayesian TPE optimization with extensive utility integration."""
        tprint_debug("🧠 Starting Bayesian TPE optimization")
        
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
                    score = self._evaluate_architecture(train_data, val_data, trial_params)
                
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
            
            tprint_success(f"✅ Bayesian optimization completed: {len(trials)} trials")
            return best_architecture, best_score, trials
            
        except Exception as e:
            tprint_error(f"❌ Error in Bayesian optimization: {e}")
            return {}, -np.inf, []
    
    def _grid_optimization(
        self, 
        train_data: pd.DataFrame, 
        val_data: pd.DataFrame,
        search_space: Dict[str, Any], 
        n_trials: int
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """Perform Grid Search optimization with extensive utility integration."""
        tprint_debug("🔧 Starting Grid Search optimization")
        
        trials = []
        best_score = -np.inf
        best_architecture = None
        
        try:
            # Generate grid parameters
            grid_params = self.grid_optimizer.generate_grid(search_space, max_trials=n_trials)
            
            total_trials = len(grid_params)
            tprint_info(f"🔧 Grid search: {total_trials} parameter combinations")
            
            for trial_idx, params in enumerate(grid_params):
                tprint_progress(trial_idx, total_trials, f"Grid search trial {trial_idx}")
                
                # Evaluate architecture
                with tprint_timer(f"Grid trial {trial_idx} evaluation"):
                    score = self._evaluate_architecture(train_data, val_data, params)
                
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
            
            tprint_success(f"✅ Grid optimization completed: {len(trials)} trials")
            return best_architecture, best_score, trials
            
        except Exception as e:
            tprint_error(f"❌ Error in Grid optimization: {e}")
            return {}, -np.inf, []
    
    def _hierarchical_optimization(
        self, 
        train_data: pd.DataFrame, 
        val_data: pd.DataFrame,
        search_space: Dict[str, Any], 
        n_trials: int
    ) -> Tuple[Dict[str, Any], float, List[Dict[str, Any]]]:
        """Perform Hierarchical HPO optimization with extensive utility integration."""
        tprint_debug("🏗️ Starting Hierarchical HPO optimization")
        
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
                    score = self._evaluate_architecture(train_data, val_data, trial_params)
                
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
            
            tprint_success(f"✅ Hierarchical optimization completed: {len(trials)} trials")
            return best_architecture, best_score, trials
            
        except Exception as e:
            tprint_error(f"❌ Error in Hierarchical optimization: {e}")
            return {}, -np.inf, []
    
    @tprint_timer("Architecture Evaluation")
    def _evaluate_architecture(
        self, 
        train_data: pd.DataFrame, 
        val_data: pd.DataFrame,
        architecture_params: Dict[str, Any]
    ) -> float:
        """Evaluate architecture performance with extensive utility integration.
        
        Args:
            train_data: Training data for architecture evaluation
            val_data: Validation data for architecture evaluation
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
            with memory_checkpoint("architecture_data_preparation"):
                # Create feature matrices using matrix operations
                train_features = self._create_architecture_feature_matrix(train_data)
                val_features = self._create_architecture_feature_matrix(val_data)
                
                # Validate feature matrices
                if not validate_correlation_matrix(train_features) or not validate_correlation_matrix(val_features):
                    tprint_warning("⚠️ Invalid feature matrix correlation structure")
                    return 0.0
            
            # Simulate architecture evaluation (placeholder for actual model evaluation)
            with gpu_context("architecture_evaluation") if self.gpu_manager else memory_checkpoint("architecture_evaluation"):
                # Use matrix operations for evaluation
                train_score = self._compute_architecture_score(train_features, validated_params)
                val_score = self._compute_architecture_score(val_features, validated_params)
                
                # Combine train and validation scores
                combined_score = safe_weighted_average([train_score, val_score], [0.3, 0.7])
            
            # Validate score
            score = validate_finite(combined_score, "architecture_score")
            
            tprint_debug(f"🔍 Architecture evaluation score: {score:.4f} (train: {train_score:.4f}, val: {val_score:.4f})")
            return score
            
        except Exception as e:
            tprint_error(f"❌ Error evaluating architecture: {e}")
            return 0.0
    
    def _create_architecture_feature_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """Create architecture feature matrix using matrix operations utilities."""
        try:
            # Extract numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            feature_data = data[numeric_cols].values
            
            # Handle NaN values
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Validate numeric array
            feature_data = validate_numeric_array(feature_data, "architecture_features")
            
            # Use matrix operations for feature engineering
            # Normalize features
            normalized_features = self.matrix_ops.normalize_matrix(feature_data)
            
            # Add polynomial features
            polynomial_features = self.enhanced_matrix_ops.add_polynomial_features(
                normalized_features, degree=2
            )
            
            # Add architecture-specific features
            architecture_features = self.matrix_convenience.add_architecture_features(
                polynomial_features
            )
            
            return architecture_features
            
        except Exception as e:
            tprint_error(f"❌ Error creating architecture feature matrix: {e}")
            return np.array([])
    
    def _compute_architecture_score(
        self, 
        feature_matrix: np.ndarray, 
        params: Dict[str, Any]
    ) -> float:
        """Compute architecture score using matrix operations."""
        try:
            # Extract key parameters
            complexity = params.get('complexity', 1.0)
            depth = params.get('depth', 1)
            width = params.get('width', 1)
            activation = params.get('activation', 'relu')
            
            # Compute base score using matrix operations
            base_score = self.vectorized_core.compute_architecture_performance(
                feature_matrix, complexity, depth, width
            )
            
            # Apply parameter-based adjustments
            complexity_factor = safe_power(complexity, 0.5)
            depth_factor = safe_log(depth + 1)
            width_factor = safe_sqrt(width)
            
            # Activation function adjustment
            activation_factors = {
                'relu': 1.0,
                'tanh': 0.9,
                'sigmoid': 0.8,
                'leaky_relu': 1.1
            }
            activation_factor = activation_factors.get(activation, 1.0)
            
            # Combine factors using math validation utilities
            adjusted_score = safe_weighted_average(
                [base_score, complexity_factor, depth_factor, width_factor, activation_factor],
                [0.6, 0.1, 0.1, 0.1, 0.1]
            )
            
            return adjusted_score
            
        except Exception as e:
            tprint_error(f"❌ Error computing architecture score: {e}")
            return 0.0
    
    def _calculate_optimization_metrics(self, trials: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate optimization performance metrics using math validation utilities."""
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
                'convergence_metric': self._calculate_convergence_metric(scores),
                'efficiency_metric': self._calculate_efficiency_metric(scores)
            }
            
            return metrics
            
        except Exception as e:
            tprint_error(f"❌ Error calculating optimization metrics: {e}")
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
    
    def _calculate_efficiency_metric(self, scores: List[float]) -> float:
        """Calculate efficiency metric based on score improvement over time."""
        try:
            if len(scores) < 5:
                return 0.0
            
            # Calculate score improvement rate
            initial_scores = scores[:len(scores)//4]
            final_scores = scores[-len(scores)//4:]
            
            initial_mean = safe_mean(np.array(initial_scores))
            final_mean = safe_mean(np.array(final_scores))
            
            if initial_mean == 0:
                return 0.0
            
            improvement_rate = safe_divide(final_mean - initial_mean, abs(initial_mean))
            return max(0.0, improvement_rate)
            
        except Exception:
            return 0.0
    
    @tprint_timer("Results Serialization")
    def save_results(
        self, 
        results: Dict[str, Any], 
        filepath: str
    ) -> bool:
        """Save optimization results using serialization utilities.
        
        Args:
            results: Optimization results to save
            filepath: Path to save results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            tprint_info(f"💾 Saving optimization results to {filepath}")
            
            # Add metadata
            results_with_metadata = {
                'results': results,
                'metadata': {
                    'timestamp': time.time(),
                    'optimizer_version': '1.0.0',
                    'm1_integration': self.m1_integration,
                    'memory_usage': get_memory_usage(),
                    'search_history_length': len(self.search_history)
                }
            }
            
            # Save using universal serializer
            success = self.serializer.save(results_with_metadata, filepath)
            
            if success:
                tprint_success(f"✅ Optimization results saved successfully to {filepath}")
            else:
                tprint_error(f"❌ Failed to save optimization results to {filepath}")
            
            return success
            
        except Exception as e:
            tprint_error(f"❌ Error saving optimization results: {e}")
            return False
    
    def load_results(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Load optimization results using serialization utilities.
        
        Args:
            filepath: Path to load results from
            
        Returns:
            Loaded results or None if loading fails
        """
        try:
            tprint_info(f"📂 Loading optimization results from {filepath}")
            
            # Load using universal serializer
            results = self.serializer.load(filepath)
            
            if results:
                tprint_success(f"✅ Optimization results loaded successfully from {filepath}")
                return results
            else:
                tprint_error(f"❌ Failed to load optimization results from {filepath}")
                return None
                
        except Exception as e:
            tprint_error(f"❌ Error loading optimization results: {e}")
            return None
    
    def cleanup(self):
        """Cleanup resources and M1 optimizations."""
        try:
            tprint_info("🧹 Cleaning up Architecture Search Optimizer resources")
            
            # Cleanup M1 optimizers
            cleanup_m1_optimizers()
            
            # Clear search history
            self.search_history.clear()
            self.performance_metrics.clear()
            
            tprint_success("✅ Architecture Search Optimizer cleanup completed")
            
        except Exception as e:
            tprint_error(f"❌ Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()


# Convenience function for quick optimizer usage
def create_architecture_search_optimizer(config: Optional[Dict[str, Any]] = None) -> ArchitectureSearchOptimizer:
    """Create an Architecture Search Optimizer instance with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured ArchitectureSearchOptimizer instance
    """
    return ArchitectureSearchOptimizer(config)


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
    
    # Create and use optimizer
    with create_architecture_search_optimizer() as optimizer:
        # Load sample data
        data = optimizer.klines_manager.read_data("ETHUSDT", "1m")
        
        if data is not None:
            # Define search space
            search_space = {
                'complexity': [1.0, 1.5, 2.0, 2.5, 3.0],
                'depth': [1, 2, 3, 4, 5],
                'width': [8, 16, 32, 64, 128],
                'activation': ['relu', 'tanh', 'sigmoid', 'leaky_relu']
            }
            
            # Perform optimization
            results = optimizer.optimize_architecture(
                data=data,
                search_space=search_space,
                optimization_method="bayesian_tpe",
                n_trials=50
            )
            
            # Save results
            if results:
                optimizer.save_results(results, "architecture_optimization_results.json")
                tprint_structured(results, LogLevel.INFO)