"""
Neural Architecture Search (NAS) Search Space Implementation

This module provides a comprehensive SearchSpace class for defining and managing
neural architecture search spaces, with integration to all relevant utilities
from the src/utils/ directory.
"""

import logging
import json
import time
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd

# Import utilities from src/utils
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, ensure_directory, 
    safe_divide, safe_log, safe_sqrt, safe_power,
    validate_finite, validate_positive, validate_range,
    get_current_datetime, format_datetime,
    optimize_dataframe_dtypes, create_data_quality_report,
    integrate_with_m1_optimizers, cleanup_m1_optimizers
)
from src.utils.common_utilities import (
    validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, get_dataframe_info
)
from src.utils.math_validation import (
    safe_correlation, safe_covariance, safe_mean, safe_std,
    safe_percentile, validate_correlation_matrix, safe_matrix_inverse
)
from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
)
from src.utils.tprint import (
    tprint, tprint_info, tprint_debug, tprint_warning, tprint_error,
    tprint_success, tprint_performance, tprint_structured
)

# Import M1 hardware utilities
try:
    from src.utils.hardware.m1_gpu_utils import (
        get_m1_gpu_manager, is_m1_available, is_mps_available,
        optimize_dataframe_for_m1, create_m1_optimized_array
    )
    M1_GPU_AVAILABLE = True
except ImportError:
    M1_GPU_AVAILABLE = False
    tprint_warning("M1 GPU utilities not available")

try:
    from src.utils.hardware.m1_memory_optimizer import (
        get_m1_memory_optimizer, optimize_memory, get_memory_usage
    )
    M1_MEMORY_AVAILABLE = True
except ImportError:
    M1_MEMORY_AVAILABLE = False
    tprint_warning("M1 memory optimizer not available")

try:
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    M1_CPU_AVAILABLE = True
except ImportError:
    M1_CPU_AVAILABLE = False
    tprint_warning("M1 CPU optimizer not available")

# Import ML common utilities
try:
    from src.utils.ml_common import (
        UnifiedCrossValidator, perform_cross_validation,
        ParetoOptimizer, ParetoFront,
        MemoryOptimizer, ParallelProcessor,
        LookaheadProtection, MLTrainingSafeguards
    )
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False
    tprint_warning("ML common utilities not available")

# Setup logging
logger = logging.getLogger(__name__)

class SearchSpaceType(Enum):
    """Types of search spaces."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    MIXED = "mixed"

class OptimizationStrategy(Enum):
    """Optimization strategies for search."""
    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"
    TPE = "tpe"
    EVOLUTIONARY = "evolutionary"
    HYPERBAND = "hyperband"

@dataclass
class ParameterRange:
    """Defines a parameter range for search space."""
    name: str
    param_type: SearchSpaceType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[Any]] = None
    default: Optional[Any] = None
    constraints: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate parameter range after initialization."""
        if self.param_type == SearchSpaceType.CONTINUOUS:
            if self.min_value is None or self.max_value is None:
                raise ValueError("Continuous parameters require min_value and max_value")
            if self.min_value >= self.max_value:
                raise ValueError("min_value must be less than max_value")
        elif self.param_type == SearchSpaceType.DISCRETE:
            if self.min_value is None or self.max_value is None:
                raise ValueError("Discrete parameters require min_value and max_value")
            if self.step is None:
                self.step = 1
        elif self.param_type == SearchSpaceType.CATEGORICAL:
            if not self.choices:
                raise ValueError("Categorical parameters require choices")
            if self.default is None:
                self.default = self.choices[0]

@dataclass
class SearchSpaceConfig:
    """Configuration for search space."""
    name: str
    description: str = ""
    max_iterations: int = 100
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    lookahead_protection: bool = True
    memory_optimization: bool = True
    hardware_optimization: bool = True
    parallel_processing: bool = True
    max_parallel_jobs: int = 4
    save_intermediate_results: bool = True
    results_dir: str = "nas_results"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not 0 < self.validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1")
        if self.cross_validation_folds < 2:
            raise ValueError("cross_validation_folds must be at least 2")

class SearchSpace:
    """
    Comprehensive Neural Architecture Search Space implementation.
    
    This class provides a complete framework for defining, managing, and optimizing
    neural architecture search spaces with integration to all relevant utilities.
    """
    
    def __init__(self, config: SearchSpaceConfig):
        """Initialize search space with configuration."""
        self.config = config
        self.parameters: Dict[str, ParameterRange] = {}
        self.results: List[Dict[str, Any]] = []
        self.best_result: Optional[Dict[str, Any]] = None
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Initialize utilities
        self._setup_logging()
        self._setup_hardware_optimization()
        self._setup_ml_utilities()
        
        # Create results directory
        ensure_directory(self.config.results_dir)
        
        tprint_info(f"Initialized SearchSpace: {self.config.name}")
    
    def _setup_logging(self):
        """Setup logging for search space."""
        self.logger = logging.getLogger(f"{__name__}.{self.config.name}")
        self.logger.setLevel(logging.INFO)
    
    def _setup_hardware_optimization(self):
        """Setup hardware optimization utilities."""
        if M1_GPU_AVAILABLE:
            self.gpu_manager = get_m1_gpu_manager()
            tprint_info("M1 GPU optimization enabled")
        else:
            self.gpu_manager = None
            
        if M1_MEMORY_AVAILABLE:
            self.memory_optimizer = get_m1_memory_optimizer()
            tprint_info("M1 memory optimization enabled")
        else:
            self.memory_optimizer = None
            
        if M1_CPU_AVAILABLE:
            self.cpu_optimizer = get_m1_cpu_optimizer()
            tprint_info("M1 CPU optimization enabled")
        else:
            self.cpu_optimizer = None
    
    def _setup_ml_utilities(self):
        """Setup ML utilities."""
        if ML_COMMON_AVAILABLE:
            self.cross_validator = UnifiedCrossValidator()
            self.memory_optimizer_ml = MemoryOptimizer()
            self.parallel_processor = ParallelProcessor()
            self.lookahead_protection = LookaheadProtection()
            self.training_safeguards = MLTrainingSafeguards()
            tprint_info("ML common utilities enabled")
        else:
            self.cross_validator = None
            self.memory_optimizer_ml = None
            self.parallel_processor = None
            self.lookahead_protection = None
            self.training_safeguards = None
    
    def add_parameter(self, param: ParameterRange) -> None:
        """Add a parameter to the search space."""
        self.parameters[param.name] = param
        tprint_debug(f"Added parameter: {param.name} ({param.param_type.value})")
    
    def add_continuous_parameter(self, name: str, min_val: float, max_val: float, 
                                default: Optional[float] = None) -> None:
        """Add a continuous parameter."""
        param = ParameterRange(
            name=name,
            param_type=SearchSpaceType.CONTINUOUS,
            min_value=min_val,
            max_value=max_val,
            default=default
        )
        self.add_parameter(param)
    
    def add_discrete_parameter(self, name: str, min_val: int, max_val: int, 
                              step: int = 1, default: Optional[int] = None) -> None:
        """Add a discrete parameter."""
        param = ParameterRange(
            name=name,
            param_type=SearchSpaceType.DISCRETE,
            min_value=min_val,
            max_value=max_val,
            step=step,
            default=default
        )
        self.add_parameter(param)
    
    def add_categorical_parameter(self, name: str, choices: List[Any], 
                                  default: Optional[Any] = None) -> None:
        """Add a categorical parameter."""
        param = ParameterRange(
            name=name,
            param_type=SearchSpaceType.CATEGORICAL,
            choices=choices,
            default=default
        )
        self.add_parameter(param)
    
    def get_parameter_space_size(self) -> int:
        """Calculate the total size of the parameter space."""
        total_size = 1
        for param in self.parameters.values():
            if param.param_type == SearchSpaceType.CONTINUOUS:
                # For continuous, estimate based on step size
                if param.step:
                    total_size *= int((param.max_value - param.min_value) / param.step) + 1
                else:
                    total_size *= 100  # Default estimation
            elif param.param_type == SearchSpaceType.DISCRETE:
                total_size *= int((param.max_value - param.min_value) / param.step) + 1
            elif param.param_type == SearchSpaceType.CATEGORICAL:
                total_size *= len(param.choices)
        
        return total_size
    
    def sample_parameters(self, n_samples: int = 1) -> List[Dict[str, Any]]:
        """Sample parameters from the search space."""
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            for name, param in self.parameters.items():
                if param.param_type == SearchSpaceType.CONTINUOUS:
                    value = np.random.uniform(param.min_value, param.max_value)
                    if param.step:
                        value = round(value / param.step) * param.step
                    sample[name] = value
                elif param.param_type == SearchSpaceType.DISCRETE:
                    value = np.random.randint(param.min_value, param.max_value + 1, param.step)
                    sample[name] = value
                elif param.param_type == SearchSpaceType.CATEGORICAL:
                    sample[name] = np.random.choice(param.choices)
            
            samples.append(sample)
        
        return samples
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate parameter values against constraints."""
        for name, value in params.items():
            if name not in self.parameters:
                tprint_warning(f"Unknown parameter: {name}")
                return False
            
            param = self.parameters[name]
            
            if param.param_type == SearchSpaceType.CONTINUOUS:
                if not (param.min_value <= value <= param.max_value):
                    tprint_warning(f"Parameter {name} out of range: {value}")
                    return False
            elif param.param_type == SearchSpaceType.DISCRETE:
                if not (param.min_value <= value <= param.max_value):
                    tprint_warning(f"Parameter {name} out of range: {value}")
                    return False
            elif param.param_type == SearchSpaceType.CATEGORICAL:
                if value not in param.choices:
                    tprint_warning(f"Parameter {name} not in choices: {value}")
                    return False
        
        return True
    
    def optimize(self, objective_function: Callable, 
                 data: Optional[Any] = None) -> Dict[str, Any]:
        """
        Optimize the search space using the specified strategy.
        
        Args:
            objective_function: Function to optimize (should return a score)
            data: Optional data to pass to objective function
            
        Returns:
            Dictionary containing optimization results
        """
        tprint_info(f"Starting optimization with strategy: {self.config.optimization_strategy.value}")
        
        start_time = time.time()
        best_score = float('-inf')
        best_params = None
        patience_counter = 0
        
        # Initialize optimization history
        self.optimization_history = []
        
        for iteration in range(self.config.max_iterations):
            tprint_progress(iteration + 1, self.config.max_iterations, 
                          f"Optimization iteration {iteration + 1}")
            
            # Sample parameters based on strategy
            if self.config.optimization_strategy == OptimizationStrategy.RANDOM:
                params = self.sample_parameters(1)[0]
            elif self.config.optimization_strategy == OptimizationStrategy.GRID:
                params = self._grid_search_sample(iteration)
            elif self.config.optimization_strategy == OptimizationStrategy.BAYESIAN:
                params = self._bayesian_sample(iteration)
            else:
                params = self.sample_parameters(1)[0]
            
            # Validate parameters
            if not self.validate_parameters(params):
                continue
            
            # Evaluate objective function
            try:
                if data is not None:
                    score = objective_function(params, data)
                else:
                    score = objective_function(params)
                
                # Record result
                result = {
                    'iteration': iteration + 1,
                    'parameters': params.copy(),
                    'score': score,
                    'timestamp': get_current_datetime().isoformat()
                }
                
                self.results.append(result)
                self.optimization_history.append(result)
                
                # Update best result
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    self.best_result = result
                    patience_counter = 0
                    tprint_success(f"New best score: {score:.4f}")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= self.config.early_stopping_patience:
                    tprint_info(f"Early stopping at iteration {iteration + 1}")
                    break
                    
            except Exception as e:
                tprint_error(f"Error in objective function: {e}")
                continue
        
        # Save results
        if self.config.save_intermediate_results:
            self._save_results()
        
        optimization_time = time.time() - start_time
        
        tprint_performance("Search space optimization", optimization_time)
        
        return {
            'best_score': best_score,
            'best_parameters': best_params,
            'total_iterations': len(self.results),
            'optimization_time': optimization_time,
            'strategy': self.config.optimization_strategy.value
        }
    
    def _grid_search_sample(self, iteration: int) -> Dict[str, Any]:
        """Sample parameters for grid search."""
        # Simple grid search implementation
        params = {}
        for name, param in self.parameters.items():
            if param.param_type == SearchSpaceType.CONTINUOUS:
                # Create grid points
                if param.step:
                    grid_points = np.arange(param.min_value, param.max_value + param.step, param.step)
                else:
                    grid_points = np.linspace(param.min_value, param.max_value, 10)
                params[name] = grid_points[iteration % len(grid_points)]
            elif param.param_type == SearchSpaceType.DISCRETE:
                values = list(range(param.min_value, param.max_value + 1, param.step))
                params[name] = values[iteration % len(values)]
            elif param.param_type == SearchSpaceType.CATEGORICAL:
                params[name] = param.choices[iteration % len(param.choices)]
        
        return params
    
    def _bayesian_sample(self, iteration: int) -> Dict[str, Any]:
        """Sample parameters for Bayesian optimization."""
        # Simplified Bayesian sampling (would integrate with actual Bayesian optimizer)
        if iteration == 0:
            return self.sample_parameters(1)[0]
        
        # Use previous results to guide sampling
        if len(self.results) > 0:
            # Simple heuristic: sample around best results
            best_result = max(self.results, key=lambda x: x['score'])
            params = best_result['parameters'].copy()
            
            # Add some noise
            for name, param in self.parameters.items():
                if param.param_type == SearchSpaceType.CONTINUOUS:
                    noise = np.random.normal(0, (param.max_value - param.min_value) * 0.1)
                    params[name] = np.clip(params[name] + noise, param.min_value, param.max_value)
                elif param.param_type == SearchSpaceType.DISCRETE:
                    if np.random.random() < 0.3:  # 30% chance to change
                        params[name] = np.random.randint(param.min_value, param.max_value + 1, param.step)
                elif param.param_type == SearchSpaceType.CATEGORICAL:
                    if np.random.random() < 0.3:  # 30% chance to change
                        params[name] = np.random.choice(param.choices)
        else:
            params = self.sample_parameters(1)[0]
        
        return params
    
    def _save_results(self) -> None:
        """Save optimization results to file."""
        try:
            results_file = Path(self.config.results_dir) / f"{self.config.name}_results.json"
            results_data = {
                'config': {
                    'name': self.config.name,
                    'description': self.config.description,
                    'max_iterations': self.config.max_iterations,
                    'optimization_strategy': self.config.optimization_strategy.value,
                    'parameters': {name: {
                        'type': param.param_type.value,
                        'min_value': param.min_value,
                        'max_value': param.max_value,
                        'choices': param.choices
                    } for name, param in self.parameters.items()}
                },
                'results': self.results,
                'best_result': self.best_result,
                'optimization_history': self.optimization_history
            }
            
            safe_json_dump(results_data, results_file)
            tprint_info(f"Results saved to {results_file}")
            
        except Exception as e:
            tprint_error(f"Failed to save results: {e}")
    
    def load_results(self, results_file: str) -> bool:
        """Load optimization results from file."""
        try:
            results_data = safe_json_load(results_file)
            if results_data:
                self.results = results_data.get('results', [])
                self.best_result = results_data.get('best_result')
                self.optimization_history = results_data.get('optimization_history', [])
                tprint_info(f"Results loaded from {results_file}")
                return True
            return False
        except Exception as e:
            tprint_error(f"Failed to load results: {e}")
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of search space and results."""
        return {
            'config': {
                'name': self.config.name,
                'description': self.config.description,
                'max_iterations': self.config.max_iterations,
                'optimization_strategy': self.config.optimization_strategy.value
            },
            'parameters': {
                'count': len(self.parameters),
                'types': {name: param.param_type.value for name, param in self.parameters.items()},
                'space_size': self.get_parameter_space_size()
            },
            'results': {
                'total_evaluations': len(self.results),
                'best_score': self.best_result['score'] if self.best_result else None,
                'best_parameters': self.best_result['parameters'] if self.best_result else None
            },
            'hardware_optimization': {
                'm1_gpu_available': M1_GPU_AVAILABLE,
                'm1_memory_available': M1_MEMORY_AVAILABLE,
                'm1_cpu_available': M1_CPU_AVAILABLE,
                'ml_common_available': ML_COMMON_AVAILABLE
            }
        }


def get_default_search_space() -> SearchSpace:
    """
    Create a default search space with common neural architecture parameters.
    
    Returns:
        SearchSpace instance with default parameters
    """
    # Create default configuration
    config = SearchSpaceConfig(
        name="default_nas_search_space",
        description="Default neural architecture search space",
        max_iterations=100,
        optimization_strategy=OptimizationStrategy.BAYESIAN,
        early_stopping_patience=10,
        validation_split=0.2,
        cross_validation_folds=5,
        lookahead_protection=True,
        memory_optimization=True,
        hardware_optimization=True,
        parallel_processing=True,
        max_parallel_jobs=4,
        save_intermediate_results=True,
        results_dir="nas_results"
    )
    
    # Create search space
    search_space = SearchSpace(config)
    
    # Add common neural architecture parameters
    search_space.add_continuous_parameter("learning_rate", 1e-5, 1e-1, 1e-3)
    search_space.add_discrete_parameter("hidden_layers", 1, 10, 1, 3)
    search_space.add_discrete_parameter("hidden_units", 32, 1024, 32, 128)
    search_space.add_categorical_parameter("activation", ["relu", "tanh", "sigmoid", "leaky_relu"], "relu")
    search_space.add_categorical_parameter("optimizer", ["adam", "sgd", "rmsprop", "adamw"], "adam")
    search_space.add_continuous_parameter("dropout_rate", 0.0, 0.8, 0.2)
    search_space.add_discrete_parameter("batch_size", 16, 512, 16, 32)
    search_space.add_categorical_parameter("regularization", ["l1", "l2", "elastic_net", "none"], "l2")
    search_space.add_continuous_parameter("l1_alpha", 1e-6, 1e-2, 1e-4)
    search_space.add_continuous_parameter("l2_alpha", 1e-6, 1e-2, 1e-4)
    
    tprint_info("Created default search space with common neural architecture parameters")
    
    return search_space


# Example usage and testing
if __name__ == "__main__":
    # Create default search space
    search_space = get_default_search_space()
    
    # Print summary
    summary = search_space.get_summary()
    tprint_structured(summary)
    
    # Example objective function
    def example_objective(params):
        """Example objective function for testing."""
        # Simple scoring based on parameter values
        score = 0.0
        
        # Learning rate scoring (prefer moderate values)
        lr = params.get('learning_rate', 1e-3)
        if 1e-4 <= lr <= 1e-2:
            score += 0.3
        
        # Hidden layers scoring (prefer moderate complexity)
        layers = params.get('hidden_layers', 3)
        if 2 <= layers <= 5:
            score += 0.2
        
        # Activation function scoring
        activation = params.get('activation', 'relu')
        if activation == 'relu':
            score += 0.2
        
        # Add some randomness
        score += np.random.normal(0, 0.1)
        
        return score
    
    # Run optimization
    tprint_info("Running example optimization...")
    results = search_space.optimize(example_objective)
    
    tprint_structured(results)