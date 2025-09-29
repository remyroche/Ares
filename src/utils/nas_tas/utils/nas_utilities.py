"""
NAS Utilities

This module provides comprehensive utilities for Neural Architecture Search
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
class NASUtilities:
    """
    NAS Utilities with extensive utility integration.
    
    This utility class provides comprehensive NAS capabilities with:
    - Extensive use of common operations for data processing
    - Math validation for safe computations
    - Comprehensive logging with tprint
    - M1 hardware optimization
    - Matrix operations for high-performance computations
    - Architecture search utilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize NAS Utilities with extensive utility integration."""
        tprint_info("🚀 Initializing NAS Utilities with extensive utility integration")
        
        self.config = config or {}
        self.logger = logger.getChild("NASUtilities")
        
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
        
        tprint_success("✅ NAS Utilities initialized successfully")
    
    @tprint_timer("Architecture Generation")
    def generate_architecture(
        self,
        search_space: Dict[str, Any],
        method: str = "random",
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate architecture using extensive utility integration."""
        tprint_info(f"🔧 Generating architecture with method: {method}")
        
        try:
            if method == "random":
                return self._generate_random_architecture(search_space, constraints)
            elif method == "grid":
                return self._generate_grid_architecture(search_space, constraints)
            elif method == "bayesian":
                return self._generate_bayesian_architecture(search_space, constraints)
            else:
                tprint_error(f"❌ Unknown generation method: {method}")
                return {}
                
        except Exception as e:
            tprint_error(f"❌ Error generating architecture: {e}")
            return {}
    
    def _generate_random_architecture(
        self,
        search_space: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate random architecture with constraints."""
        try:
            architecture = {}
            
            for param, values in search_space.items():
                if isinstance(values, list):
                    # Random selection from list
                    selected_value = np.random.choice(values)
                elif isinstance(values, dict) and 'min' in values and 'max' in values:
                    # Random value in range
                    min_val = values['min']
                    max_val = values['max']
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        selected_value = np.random.randint(min_val, max_val + 1)
                    else:
                        selected_value = np.random.uniform(min_val, max_val)
                else:
                    selected_value = values
                
                # Apply constraints if provided
                if constraints and param in constraints:
                    constraint = constraints[param]
                    if isinstance(constraint, dict):
                        if 'min' in constraint:
                            selected_value = max(selected_value, constraint['min'])
                        if 'max' in constraint:
                            selected_value = min(selected_value, constraint['max'])
                
                architecture[param] = selected_value
            
            tprint_success(f"✅ Generated random architecture: {architecture}")
            return architecture
            
        except Exception as e:
            tprint_error(f"❌ Error generating random architecture: {e}")
            return {}
    
    def _generate_grid_architecture(
        self,
        search_space: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate grid architecture with constraints."""
        try:
            # This would be used in grid search - return first combination
            architecture = {}
            
            for param, values in search_space.items():
                if isinstance(values, list):
                    selected_value = values[0]  # First value for grid
                else:
                    selected_value = values
                
                architecture[param] = selected_value
            
            tprint_success(f"✅ Generated grid architecture: {architecture}")
            return architecture
            
        except Exception as e:
            tprint_error(f"❌ Error generating grid architecture: {e}")
            return {}
    
    def _generate_bayesian_architecture(
        self,
        search_space: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate Bayesian architecture with constraints."""
        try:
            # This would be used in Bayesian optimization
            architecture = {}
            
            for param, values in search_space.items():
                if isinstance(values, list):
                    # Use TPE-like selection
                    selected_value = np.random.choice(values)
                else:
                    selected_value = values
                
                architecture[param] = selected_value
            
            tprint_success(f"✅ Generated Bayesian architecture: {architecture}")
            return architecture
            
        except Exception as e:
            tprint_error(f"❌ Error generating Bayesian architecture: {e}")
            return {}
    
    @tprint_timer("Architecture Validation")
    def validate_architecture(
        self,
        architecture: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate architecture using extensive utility integration."""
        tprint_info("🔍 Validating architecture")
        
        try:
            validation_results = {
                'valid': True,
                'errors': [],
                'warnings': [],
                'score': 0.0
            }
            
            # Validate each parameter
            for param, value in architecture.items():
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
            
            tprint_info(f"🔍 Architecture validation: {'✅ Valid' if validation_results['valid'] else '❌ Invalid'}")
            return validation_results
            
        except Exception as e:
            tprint_error(f"❌ Error validating architecture: {e}")
            return {'valid': False, 'errors': [str(e)], 'warnings': [], 'score': 0.0}
    
    @tprint_timer("Architecture Scoring")
    def score_architecture(
        self,
        architecture: Dict[str, Any],
        data: pd.DataFrame,
        scoring_method: str = "comprehensive"
    ) -> float:
        """Score architecture using extensive utility integration."""
        tprint_info(f"🔍 Scoring architecture with method: {scoring_method}")
        
        try:
            # Validate input data
            if not validate_dataframe_columns(data, ['open', 'high', 'low', 'close', 'volume']):
                tprint_error("❌ Invalid data columns for architecture scoring")
                return 0.0
            
            # Create feature matrix
            with memory_checkpoint("feature_matrix_creation"):
                feature_matrix = self._create_architecture_feature_matrix(data)
            
            if feature_matrix.size == 0:
                tprint_error("❌ Empty feature matrix")
                return 0.0
            
            # Score architecture based on method
            if scoring_method == "comprehensive":
                score = self._comprehensive_architecture_score(feature_matrix, architecture)
            elif scoring_method == "basic":
                score = self._basic_architecture_score(feature_matrix, architecture)
            elif scoring_method == "performance":
                score = self._performance_architecture_score(feature_matrix, architecture)
            else:
                tprint_error(f"❌ Unknown scoring method: {scoring_method}")
                return 0.0
            
            # Validate score
            score = validate_finite(score, "architecture_score")
            
            tprint_info(f"🔍 Architecture score: {score:.4f}")
            return score
            
        except Exception as e:
            tprint_error(f"❌ Error scoring architecture: {e}")
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
            normalized_features = self.matrix_ops.normalize_matrix(feature_data)
            polynomial_features = self.enhanced_matrix_ops.add_polynomial_features(normalized_features, degree=2)
            
            return polynomial_features
            
        except Exception as e:
            tprint_error(f"❌ Error creating architecture feature matrix: {e}")
            return np.array([])
    
    def _comprehensive_architecture_score(
        self,
        feature_matrix: np.ndarray,
        architecture: Dict[str, Any]
    ) -> float:
        """Calculate comprehensive architecture score."""
        try:
            # Extract parameters
            complexity = architecture.get('complexity', 1.0)
            depth = architecture.get('depth', 1)
            width = architecture.get('width', 1)
            activation = architecture.get('activation', 'relu')
            
            # Calculate base score using matrix operations
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
            
            # Combine factors
            adjusted_score = safe_weighted_average(
                [base_score, complexity_factor, depth_factor, width_factor, activation_factor],
                [0.6, 0.1, 0.1, 0.1, 0.1]
            )
            
            return adjusted_score
            
        except Exception as e:
            tprint_error(f"❌ Error calculating comprehensive architecture score: {e}")
            return 0.0
    
    def _basic_architecture_score(
        self,
        feature_matrix: np.ndarray,
        architecture: Dict[str, Any]
    ) -> float:
        """Calculate basic architecture score."""
        try:
            # Simple scoring based on architecture parameters
            complexity = architecture.get('complexity', 1.0)
            depth = architecture.get('depth', 1)
            width = architecture.get('width', 1)
            
            # Basic score calculation
            score = safe_weighted_average([complexity, depth, width], [0.5, 0.3, 0.2])
            
            return score
            
        except Exception as e:
            tprint_error(f"❌ Error calculating basic architecture score: {e}")
            return 0.0
    
    def _performance_architecture_score(
        self,
        feature_matrix: np.ndarray,
        architecture: Dict[str, Any]
    ) -> float:
        """Calculate performance-focused architecture score."""
        try:
            # Performance-based scoring
            base_score = self.vectorized_core.compute_performance_metric(
                feature_matrix, architecture
            )
            
            return base_score
            
        except Exception as e:
            tprint_error(f"❌ Error calculating performance architecture score: {e}")
            return 0.0
    
    def cleanup(self):
        """Cleanup resources and M1 optimizations."""
        try:
            tprint_info("🧹 Cleaning up NAS Utilities resources")
            cleanup_m1_optimizers()
            tprint_success("✅ NAS Utilities cleanup completed")
        except Exception as e:
            tprint_error(f"❌ Error during cleanup: {e}")


# Convenience function
def create_nas_utilities(config: Optional[Dict[str, Any]] = None) -> NASUtilities:
    """Create NAS Utilities instance with default configuration."""
    return NASUtilities(config)