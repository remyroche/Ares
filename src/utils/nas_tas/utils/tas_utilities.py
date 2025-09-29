"""
TAS Utilities

This module provides comprehensive utilities for Trading Architecture Search
with extensive integration of utility modules for optimal performance,
data processing, and hardware optimization.
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

from ...matrix_operations.unified_operations import MatrixOperations
from ...matrix_operations.enhanced_operations import EnhancedMatrixOperations
from ...matrix_operations.batch_operations import BatchMatrixOperations
from ...matrix_operations.vectorized_core import VectorizedCore

from ...hardware.m1_gpu_utils import M1GPUManager, is_m1_available, is_mps_available
from ...hardware.m1_memory_optimizer import M1MemoryOptimizer
from ...hardware.m1_cpu_optimizer import M1CPUOptimizer

# Setup logging with tprint integration
logger = logging.getLogger(__name__)

@tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
class TASUtilities:
    """
    TAS Utilities with extensive utility integration.
    
    This utility class provides comprehensive TAS capabilities with:
    - Extensive use of common operations for data processing
    - Math validation for safe computations
    - Comprehensive logging with tprint
    - M1 hardware optimization
    - Matrix operations for high-performance computations
    - Trading strategy utilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize TAS Utilities with extensive utility integration."""
        tprint_info("🚀 Initializing TAS Utilities with extensive utility integration")
        
        self.config = config or {}
        self.logger = logger.getChild("TASUtilities")
        
        # Initialize utility classes
        self.math_validator = MathValidation()
        self.matrix_ops = MatrixOperations()
        self.enhanced_matrix_ops = EnhancedMatrixOperations()
        self.batch_matrix_ops = BatchMatrixOperations()
        self.vectorized_core = VectorizedCore()
        
        # Initialize M1 hardware optimizations
        self.m1_integration = integrate_with_m1_optimizers()
        if self.m1_integration['success']:
            self.gpu_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()
        else:
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
        
        tprint_success("✅ TAS Utilities initialized successfully")
    
    @tprint_timer("Strategy Generation")
    def generate_strategy(
        self,
        search_space: Dict[str, Any],
        method: str = "random",
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate trading strategy using extensive utility integration."""
        tprint_info(f"🔧 Generating strategy with method: {method}")
        
        try:
            if method == "random":
                return self._generate_random_strategy(search_space, constraints)
            elif method == "grid":
                return self._generate_grid_strategy(search_space, constraints)
            elif method == "bayesian":
                return self._generate_bayesian_strategy(search_space, constraints)
            else:
                tprint_error(f"❌ Unknown generation method: {method}")
                return {}
                
        except Exception as e:
            tprint_error(f"❌ Error generating strategy: {e}")
            return {}
    
    def _generate_random_strategy(
        self,
        search_space: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate random strategy with constraints."""
        try:
            strategy = {}
            
            for param, values in search_space.items():
                if isinstance(values, list):
                    selected_value = np.random.choice(values)
                elif isinstance(values, dict) and 'min' in values and 'max' in values:
                    min_val = values['min']
                    max_val = values['max']
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        selected_value = np.random.randint(min_val, max_val + 1)
                    else:
                        selected_value = np.random.uniform(min_val, max_val)
                else:
                    selected_value = values
                
                # Apply constraints
                if constraints and param in constraints:
                    constraint = constraints[param]
                    if isinstance(constraint, dict):
                        if 'min' in constraint:
                            selected_value = max(selected_value, constraint['min'])
                        if 'max' in constraint:
                            selected_value = min(selected_value, constraint['max'])
                
                strategy[param] = selected_value
            
            tprint_success(f"✅ Generated random strategy: {strategy}")
            return strategy
            
        except Exception as e:
            tprint_error(f"❌ Error generating random strategy: {e}")
            return {}
    
    def _generate_grid_strategy(
        self,
        search_space: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate grid strategy with constraints."""
        try:
            strategy = {}
            
            for param, values in search_space.items():
                if isinstance(values, list):
                    selected_value = values[0]  # First value for grid
                else:
                    selected_value = values
                
                strategy[param] = selected_value
            
            tprint_success(f"✅ Generated grid strategy: {strategy}")
            return strategy
            
        except Exception as e:
            tprint_error(f"❌ Error generating grid strategy: {e}")
            return {}
    
    def _generate_bayesian_strategy(
        self,
        search_space: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate Bayesian strategy with constraints."""
        try:
            strategy = {}
            
            for param, values in search_space.items():
                if isinstance(values, list):
                    selected_value = np.random.choice(values)
                else:
                    selected_value = values
                
                strategy[param] = selected_value
            
            tprint_success(f"✅ Generated Bayesian strategy: {strategy}")
            return strategy
            
        except Exception as e:
            tprint_error(f"❌ Error generating Bayesian strategy: {e}")
            return {}
    
    @tprint_timer("Strategy Validation")
    def validate_strategy(
        self,
        strategy: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate strategy using extensive utility integration."""
        tprint_info("🔍 Validating strategy")
        
        try:
            validation_results = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'score': 0.0
            }
            
            # Validate each parameter
            for param, value in strategy.items():
                try:
                    # Validate finite values
                    if isinstance(value, (int, float)):
                        validated_value = validate_finite(value, param)
                        if validated_value != value:
                            validation_results['warnings'].append(f"Parameter {param} adjusted from {value} to {validated_value}")
                    
                    # Apply constraints
                    if constraints and param in constraints:
                        constraint = constraints[param]
                        if isinstance(constraint, dict):
                            if 'min' in constraint and value < constraint['min']:
                                validation_results['errors'].append(f"Parameter {param} below minimum: {value} < {constraint['min']}")
                                validation_results['valid'] = False
                            if 'max' in constraint and value > constraint['max']:
                                validation_results['errors'].append(f"Parameter {param} above maximum: {value} > {constraint['max']}")
                                validation_results['valid'] = False
                
                except ValueError as e:
                    validation_results['errors'].append(f"Invalid parameter {param}: {e}")
                    validation_results['valid'] = False
            
            # Calculate validation score
            if validation_results['valid']:
                validation_results['score'] = 1.0
            else:
                validation_results['score'] = 0.0
            
            tprint_info(f"🔍 Strategy validation: {'✅ Valid' if validation_results['valid'] else '❌ Invalid'}")
            return validation_results
            
        except Exception as e:
            tprint_error(f"❌ Error validating strategy: {e}")
            return {'valid': False, 'errors': [str(e)], 'warnings': [], 'score': 0.0}
    
    @tprint_timer("Strategy Scoring")
    def score_strategy(
        self,
        strategy: Dict[str, Any],
        data: pd.DataFrame,
        scoring_method: str = "comprehensive"
    ) -> float:
        """Score strategy using extensive utility integration."""
        tprint_info(f"🔍 Scoring strategy with method: {scoring_method}")
        
        try:
            # Validate input data
            if not validate_dataframe_columns(data, ['open', 'high', 'low', 'close', 'volume']):
                tprint_error("❌ Invalid data columns for strategy scoring")
                return 0.0
            
            # Create feature matrix
            with memory_checkpoint("feature_matrix_creation"):
                feature_matrix = self._create_strategy_feature_matrix(data)
            
            if feature_matrix.size == 0:
                tprint_error("❌ Empty feature matrix")
                return 0.0
            
            # Score strategy based on method
            if scoring_method == "comprehensive":
                score = self._comprehensive_strategy_score(feature_matrix, strategy)
            elif scoring_method == "basic":
                score = self._basic_strategy_score(feature_matrix, strategy)
            elif scoring_method == "performance":
                score = self._performance_strategy_score(feature_matrix, strategy)
            else:
                tprint_error(f"❌ Unknown scoring method: {scoring_method}")
                return 0.0
            
            # Validate score
            score = validate_finite(score, "strategy_score")
            
            tprint_info(f"🔍 Strategy score: {score:.4f}")
            return score
            
        except Exception as e:
            tprint_error(f"❌ Error scoring strategy: {e}")
            return 0.0
    
    def _create_strategy_feature_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """Create strategy feature matrix using matrix operations utilities."""
        try:
            # Extract numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            feature_data = data[numeric_cols].values
            
            # Handle NaN values
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Validate numeric array
            feature_data = validate_numeric_array(feature_data, "strategy_features")
            
            # Use matrix operations for feature engineering
            normalized_features = self.matrix_ops.normalize_matrix(feature_data)
            technical_features = self.enhanced_matrix_ops.add_technical_features(normalized_features)
            
            return technical_features
            
        except Exception as e:
            tprint_error(f"❌ Error creating strategy feature matrix: {e}")
            return np.array([])
    
    def _comprehensive_strategy_score(
        self,
        feature_matrix: np.ndarray,
        strategy: Dict[str, Any]
    ) -> float:
        """Calculate comprehensive strategy score."""
        try:
            # Extract parameters
            entry_threshold = strategy.get('entry_threshold', 0.5)
            exit_threshold = strategy.get('exit_threshold', 0.5)
            risk_factor = strategy.get('risk_factor', 1.0)
            position_size = strategy.get('position_size', 0.1)
            
            # Calculate base score using matrix operations
            base_score = self.vectorized_core.compute_strategy_performance(
                feature_matrix, entry_threshold, exit_threshold
            )
            
            # Apply parameter-based adjustments
            risk_adjustment = safe_power(risk_factor, 0.5)
            position_adjustment = safe_sqrt(position_size)
            
            # Combine factors
            adjusted_score = safe_weighted_average(
                [base_score, risk_adjustment, position_adjustment],
                [0.7, 0.2, 0.1]
            )
            
            return adjusted_score
            
        except Exception as e:
            tprint_error(f"❌ Error calculating comprehensive strategy score: {e}")
            return 0.0
    
    def _basic_strategy_score(
        self,
        feature_matrix: np.ndarray,
        strategy: Dict[str, Any]
    ) -> float:
        """Calculate basic strategy score."""
        try:
            # Simple scoring based on strategy parameters
            entry_threshold = strategy.get('entry_threshold', 0.5)
            exit_threshold = strategy.get('exit_threshold', 0.5)
            risk_factor = strategy.get('risk_factor', 1.0)
            
            # Basic score calculation
            score = safe_weighted_average([entry_threshold, exit_threshold, risk_factor], [0.4, 0.4, 0.2])
            
            return score
            
        except Exception as e:
            tprint_error(f"❌ Error calculating basic strategy score: {e}")
            return 0.0
    
    def _performance_strategy_score(
        self,
        feature_matrix: np.ndarray,
        strategy: Dict[str, Any]
    ) -> float:
        """Calculate performance-focused strategy score."""
        try:
            # Performance-based scoring
            base_score = self.vectorized_core.compute_strategy_performance(
                feature_matrix, strategy
            )
            
            return base_score
            
        except Exception as e:
            tprint_error(f"❌ Error calculating performance strategy score: {e}")
            return 0.0
    
    def cleanup(self):
        """Cleanup resources and M1 optimizations."""
        try:
            tprint_info("🧹 Cleaning up TAS Utilities resources")
            cleanup_m1_optimizers()
            tprint_success("✅ TAS Utilities cleanup completed")
        except Exception as e:
            tprint_error(f"❌ Error during cleanup: {e}")


# Convenience function
def create_tas_utilities(config: Optional[Dict[str, Any]] = None) -> TASUtilities:
    """Create TAS Utilities instance with default configuration."""
    return TASUtilities(config)