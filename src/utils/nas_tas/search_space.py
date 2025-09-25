"""
Comprehensive Search Space Implementation for Neural Architecture Search
Integrates with shared utilities from src/utils/
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime
from pathlib import Path
import json
import itertools

# Import shared utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, ensure_directory,
    safe_divide, safe_log, safe_sqrt, validate_finite, validate_positive,
    get_current_datetime, optimize_dataframe_dtypes
)
from src.utils.common_utilities import (
    validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, get_dataframe_info
)
from src.utils.math_validation import (
    safe_correlation, safe_covariance, safe_mean, safe_std,
    safe_percentile, validate_correlation_matrix
)
from src.utils.tprint import (
    tprint, tprint_info, tprint_debug, tprint_warning, tprint_error,
    tprint_success, tprint_performance, tprint_structured
)
from src.utils.serialization_utils import JSONSerializer, UniversalSerializer

# Import ML optimization utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer
    from src.utils.ml_common.optimization.grid_utils import build_coarse_grid_from_search_space
    TPE_GRID_AVAILABLE = True
except ImportError:
    TPE_GRID_AVAILABLE = False
    tprint_warning("TPE/Grid optimization utilities not available")

# Import hardware optimization
try:
    from src.utils.hardware.m1_gpu_utils import (
        get_m1_gpu_manager, is_m1_available, optimize_dataframe_for_m1
    )
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer, optimize_memory
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    M1_HARDWARE_AVAILABLE = True
except ImportError:
    M1_HARDWARE_AVAILABLE = False
    tprint_warning("M1 hardware optimization not available")

# Import matrix operations
try:
    from src.utils.matrix_operations import MatrixOperations, OptimizedMatrixOps
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False
    tprint_warning("Matrix operations utilities not available")

logger = logging.getLogger(__name__)


class SearchSpaceType(Enum):
    """Types of search spaces."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    MIXED = "mixed"
    HIERARCHICAL = "hierarchical"


class OptimizationStrategy(Enum):
    """Optimization strategies for search."""
    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"
    TPE = "tpe"
    EVOLUTIONARY = "evolutionary"
    HYPERBAND = "hyperband"
    GRID_TPE = "grid_tpe"  # Combined grid + TPE strategy
    HIERARCHICAL = "hierarchical"


@dataclass
class ParameterRange:
    """Defines a parameter range for search space with validation."""
    
    name: str
    param_type: SearchSpaceType
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    step: Optional[Union[int, float]] = None
    choices: Optional[List[Any]] = None
    default: Optional[Any] = None
    constraints: Optional[Dict[str, Any]] = None
    log_scale: bool = False
    description: str = ""
    
    def __post_init__(self):
        """Validate parameter range after initialization."""
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate parameter configuration."""
        if self.param_type == SearchSpaceType.CONTINUOUS:
            if self.min_value is None or self.max_value is None:
                raise ValueError(f"Parameter {self.name}: Continuous parameters require min_value and max_value")
            if not validate_finite(np.array([self.min_value, self.max_value])).all():
                raise ValueError(f"Parameter {self.name}: min_value and max_value must be finite")
            if self.min_value >= self.max_value:
                raise ValueError(f"Parameter {self.name}: min_value must be less than max_value")
                
        elif self.param_type == SearchSpaceType.DISCRETE:
            if self.min_value is None or self.max_value is None:
                raise ValueError(f"Parameter {self.name}: Discrete parameters require min_value and max_value")
            if self.step is None:
                self.step = 1
            if not isinstance(self.min_value, int) or not isinstance(self.max_value, int):
                raise ValueError(f"Parameter {self.name}: Discrete parameters require integer bounds")
                
        elif self.param_type == SearchSpaceType.CATEGORICAL:
            if not self.choices:
                raise ValueError(f"Parameter {self.name}: Categorical parameters require choices")
            if self.default is None:
                self.default = self.choices[0]
                
        tprint_debug(f"Validated parameter {self.name} ({self.param_type.value})")
    
    def sample(self) -> Any:
        """Sample a value from the parameter range."""
        try:
            if self.param_type == SearchSpaceType.CONTINUOUS:
                if self.log_scale:
                    min_log = safe_log(self.min_value) if self.min_value > 0 else np.log(1e-10)
                    max_log = safe_log(self.max_value) if self.max_value > 0 else np.log(1.0)
                    return np.exp(np.random.uniform(min_log, max_log))
                else:
                    value = np.random.uniform(self.min_value, self.max_value)
                    if self.step:
                        value = round(value / self.step) * self.step
                    return value
                    
            elif self.param_type == SearchSpaceType.DISCRETE:
                if self.step:
                    n_steps = int((self.max_value - self.min_value) / self.step) + 1
                    step_idx = np.random.randint(0, n_steps)
                    return self.min_value + step_idx * self.step
                else:
                    return np.random.randint(self.min_value, self.max_value + 1)
                    
            elif self.param_type == SearchSpaceType.CATEGORICAL:
                return np.random.choice(self.choices)
            else:
                raise ValueError(f"Unknown parameter type: {self.param_type}")
                
        except Exception as e:
            tprint_warning(f"Failed to sample parameter {self.name}: {e}")
            return self.default
    
    def is_valid(self, value: Any) -> bool:
        """Check if a value is valid for this parameter range."""
        try:
            if self.param_type == SearchSpaceType.CONTINUOUS:
                if not validate_finite(np.array([value])).all():
                    return False
                return self.min_value <= value <= self.max_value
                
            elif self.param_type == SearchSpaceType.DISCRETE:
                if not isinstance(value, (int, np.integer)):
                    return False
                return self.min_value <= value <= self.max_value
                
            elif self.param_type == SearchSpaceType.CATEGORICAL:
                return value in self.choices
            else:
                return False
                
        except Exception:
            return False


@dataclass
class SearchSpaceConfig:
    """Configuration for search space with comprehensive settings."""
    
    name: str
    description: str = ""
    max_iterations: int = 100
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.GRID_TPE
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    
    # Hardware optimization
    enable_m1_optimization: bool = True
    enable_parallel_processing: bool = True
    max_parallel_jobs: int = 4
    
    # Memory management
    enable_memory_optimization: bool = True
    max_memory_usage_gb: float = 8.0
    
    # Results and logging
    save_intermediate_results: bool = True
    results_dir: str = "search_space_results"
    log_level: str = "INFO"
    
    # Grid + TPE specific settings
    grid_phase_ratio: float = 0.3  # 30% for grid search
    tpe_phase_ratio: float = 0.7   # 70% for TPE
    grid_sample_density: int = 5   # Grid density per dimension
    tpe_n_startup_trials: int = 10
    
    # Advanced features
    enable_hierarchical_search: bool = False
    enable_meta_learning: bool = False
    enable_transfer_learning: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not 0 < self.validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1")
        if self.cross_validation_folds < 2:
            raise ValueError("cross_validation_folds must be at least 2")
        if not 0 < self.grid_phase_ratio < 1:
            raise ValueError("grid_phase_ratio must be between 0 and 1")
        if not 0 < self.tpe_phase_ratio < 1:
            raise ValueError("tpe_phase_ratio must be between 0 and 1")
        if abs(self.grid_phase_ratio + self.tpe_phase_ratio - 1.0) > 1e-6:
            raise ValueError("grid_phase_ratio + tpe_phase_ratio must equal 1.0")


class SearchSpace:
    """
    Comprehensive Neural Architecture Search Space implementation.
    
    Integrates with all shared utilities and provides advanced optimization strategies
    including Grid + TPE, hardware optimization, and memory management.
    """
    
    def __init__(self, config: SearchSpaceConfig):
        """Initialize search space with comprehensive configuration."""
        self.config = config
        self.parameters: Dict[str, ParameterRange] = {}
        self.results: List[Dict[str, Any]] = []
        self.best_result: Optional[Dict[str, Any]] = None
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Initialize utilities and hardware
        self._setup_logging()
        self._setup_serialization()
        self._setup_hardware_optimization()
        self._setup_matrix_operations()
        self._setup_optimization_utilities()
        
        # Create results directory
        ensure_directory(self.config.results_dir)
        
        tprint_info(f"🚀 SearchSpace '{self.config.name}' initialized with shared utilities")
        tprint_info(f"📊 Strategy: {self.config.optimization_strategy.value}")
        if self.config.enable_m1_optimization and M1_HARDWARE_AVAILABLE:
            tprint_info("⚡ M1 hardware optimization enabled")
    
    def _setup_logging(self):
        """Setup comprehensive logging."""
        self.logger = logging.getLogger(f"{__name__}.{self.config.name}")
        self.logger.setLevel(getattr(logging, self.config.log_level))
    
    def _setup_serialization(self):
        """Setup serialization utilities."""
        self.json_serializer = JSONSerializer()
        self.universal_serializer = UniversalSerializer()
    
    def _setup_hardware_optimization(self):
        """Setup M1 hardware optimization if available."""
        if self.config.enable_m1_optimization and M1_HARDWARE_AVAILABLE:
            try:
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()
                tprint_info("✅ M1 hardware optimization setup complete")
            except Exception as e:
                tprint_warning(f"M1 optimization setup failed: {e}")
                self.gpu_manager = None
                self.memory_optimizer = None
                self.cpu_optimizer = None
        else:
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
    
    def _setup_matrix_operations(self):
        """Setup optimized matrix operations if available."""
        if MATRIX_OPS_AVAILABLE:
            try:
                self.matrix_ops = OptimizedMatrixOps()
                tprint_info("✅ Optimized matrix operations enabled")
            except Exception as e:
                tprint_warning(f"Matrix operations setup failed: {e}")
                self.matrix_ops = None
        else:
            self.matrix_ops = None
    
    def _setup_optimization_utilities(self):
        """Setup optimization utilities (TPE, Grid, etc.)."""
        if TPE_GRID_AVAILABLE:
            self.tpe_optimizer = None  # Initialize when needed
            tprint_info("✅ TPE and Grid optimization utilities available")
        else:
            self.tpe_optimizer = None
    
    def add_parameter(self, param: ParameterRange) -> None:
        """Add a parameter to the search space with validation."""
        if param.name in self.parameters:
            tprint_warning(f"Parameter {param.name} already exists, overwriting")
        
        self.parameters[param.name] = param
        tprint_debug(f"Added parameter: {param.name} ({param.param_type.value})")
    
    def add_continuous_parameter(self, name: str, min_val: float, max_val: float,
                                step: Optional[float] = None, log_scale: bool = False,
                                default: Optional[float] = None, description: str = "") -> None:
        """Add a continuous parameter with validation."""
        param = ParameterRange(
            name=name,
            param_type=SearchSpaceType.CONTINUOUS,
            min_value=float(min_val),
            max_value=float(max_val),
            step=step,
            log_scale=log_scale,
            default=default,
            description=description
        )
        self.add_parameter(param)
    
    def add_discrete_parameter(self, name: str, min_val: int, max_val: int,
                              step: int = 1, default: Optional[int] = None,
                              description: str = "") -> None:
        """Add a discrete parameter with validation."""
        param = ParameterRange(
            name=name,
            param_type=SearchSpaceType.DISCRETE,
            min_value=int(min_val),
            max_value=int(max_val),
            step=int(step),
            default=default,
            description=description
        )
        self.add_parameter(param)
    
    def add_categorical_parameter(self, name: str, choices: List[Any],
                                 default: Optional[Any] = None,
                                 description: str = "") -> None:
        """Add a categorical parameter with validation."""
        param = ParameterRange(
            name=name,
            param_type=SearchSpaceType.CATEGORICAL,
            choices=list(choices),
            default=default,
            description=description
        )
        self.add_parameter(param)
    
    def get_parameter_space_size(self) -> int:
        """Calculate the total size of the parameter space."""
        if not self.parameters:
            return 0
        
        total_size = 1
        for param in self.parameters.values():
            try:
                if param.param_type == SearchSpaceType.CONTINUOUS:
                    if param.step:
                        param_size = int((param.max_value - param.min_value) / param.step) + 1
                    else:
                        param_size = 1000  # Default estimation for continuous
                elif param.param_type == SearchSpaceType.DISCRETE:
                    param_size = int((param.max_value - param.min_value) / param.step) + 1
                elif param.param_type == SearchSpaceType.CATEGORICAL:
                    param_size = len(param.choices)
                else:
                    param_size = 1
                
                total_size *= param_size
                
                # Prevent overflow
                if total_size > 1e12:
                    return int(1e12)
                    
            except Exception as e:
                tprint_warning(f"Error calculating space size for {param.name}: {e}")
                continue
        
        return int(total_size)
    
    def sample_parameters(self, n_samples: int = 1, use_hardware_optimization: bool = True) -> List[Dict[str, Any]]:
        """Sample parameters from the search space with optional hardware optimization."""
        if not self.parameters:
            tprint_warning("No parameters defined in search space")
            return []
        
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            for name, param in self.parameters.items():
                try:
                    value = param.sample()
                    sample[name] = value
                except Exception as e:
                    tprint_warning(f"Failed to sample parameter {name}: {e}")
                    sample[name] = param.default
            
            samples.append(sample)
        
        # Optimize for hardware if available and requested
        if use_hardware_optimization and self.memory_optimizer and len(samples) > 100:
            try:
                samples = self._optimize_samples_for_hardware(samples)
            except Exception as e:
                tprint_warning(f"Hardware optimization failed: {e}")
        
        return samples
    
    def _optimize_samples_for_hardware(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize samples for hardware processing."""
        if not self.memory_optimizer:
            return samples
        
        try:
            # Convert to DataFrame for optimization
            df = pd.DataFrame(samples)
            optimized_df = optimize_dataframe_dtypes(df)
            
            if self.gpu_manager and is_m1_available():
                optimized_df = optimize_dataframe_for_m1(optimized_df)
            
            return optimized_df.to_dict('records')
        except Exception as e:
            tprint_warning(f"Hardware optimization failed: {e}")
            return samples
    
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate parameter values against constraints with detailed feedback."""
        if not self.parameters:
            return False, ["No parameters defined in search space"]
        
        errors = []
        
        for name, value in params.items():
            if name not in self.parameters:
                errors.append(f"Unknown parameter: {name}")
                continue
            
            param = self.parameters[name]
            if not param.is_valid(value):
                errors.append(f"Parameter {name} invalid value {value} (type: {param.param_type.value})")
        
        # Check for missing required parameters
        for name, param in self.parameters.items():
            if name not in params and param.default is None:
                errors.append(f"Missing required parameter: {name}")
        
        is_valid = len(errors) == 0
        if not is_valid:
            tprint_warning(f"Parameter validation failed: {errors}")
        
        return is_valid, errors
    
    def optimize(self, objective_function: Callable[..., float],
                 data: Optional[Any] = None,
                 use_hardware_acceleration: bool = True) -> Dict[str, Any]:
        """
        Optimize the search space using the configured strategy.
        
        Args:
            objective_function: Function to optimize (should return a score)
            data: Optional data to pass to objective function
            use_hardware_acceleration: Enable hardware acceleration
            
        Returns:
            Dictionary containing optimization results
        """
        if not self.parameters:
            raise ValueError("No parameters defined in search space")
        
        tprint_info(f"🚀 Starting optimization: {self.config.optimization_strategy.value}")
        tprint_info(f"📊 Search space size: {self.get_parameter_space_size()}")
        
        start_time = time.time()
        
        # Choose optimization strategy
        if self.config.optimization_strategy == OptimizationStrategy.GRID_TPE:
            results = self._optimize_grid_tpe(objective_function, data, use_hardware_acceleration)
        elif self.config.optimization_strategy == OptimizationStrategy.GRID:
            results = self._optimize_grid(objective_function, data, use_hardware_acceleration)
        elif self.config.optimization_strategy == OptimizationStrategy.TPE:
            results = self._optimize_tpe(objective_function, data, use_hardware_acceleration)
        elif self.config.optimization_strategy == OptimizationStrategy.RANDOM:
            results = self._optimize_random(objective_function, data, use_hardware_acceleration)
        elif self.config.optimization_strategy == OptimizationStrategy.HIERARCHICAL:
            results = self._optimize_hierarchical(objective_function, data, use_hardware_acceleration)
        else:
            tprint_warning(f"Unknown strategy {self.config.optimization_strategy}, using random")
            results = self._optimize_random(objective_function, data, use_hardware_acceleration)
        
        # Save results if configured
        if self.config.save_intermediate_results:
            self._save_results()
        
        optimization_time = time.time() - start_time
        results['optimization_time'] = optimization_time
        results['search_space_size'] = self.get_parameter_space_size()
        
        tprint_performance("Search Space Optimization", optimization_time)
        tprint_success(f"✅ Best score: {results.get('best_score', 'N/A')}")
        
        return results
    
    def _optimize_grid_tpe(self, objective_function: Callable, data: Optional[Any],
                          use_hardware_acceleration: bool) -> Dict[str, Any]:
        """Optimize using Grid + TPE strategy."""
        tprint_info("🔍 Grid + TPE optimization strategy")
        
        # Phase 1: Grid search
        grid_iterations = int(self.config.max_iterations * self.config.grid_phase_ratio)
        tprint_info(f"Phase 1: Grid search ({grid_iterations} iterations)")
        
        grid_results = self._run_grid_phase(objective_function, data, grid_iterations, use_hardware_acceleration)
        
        # Phase 2: TPE optimization
        tpe_iterations = self.config.max_iterations - grid_iterations
        tprint_info(f"Phase 2: TPE optimization ({tpe_iterations} iterations)")
        
        tpe_results = self._run_tpe_phase(objective_function, data, tpe_iterations, 
                                         grid_results.get('best_parameters'), use_hardware_acceleration)
        
        # Combine results
        all_evaluations = grid_results.get('evaluations', []) + tpe_results.get('evaluations', [])
        best_result = max(all_evaluations, key=lambda x: x.get('score', -np.inf)) if all_evaluations else None
        
        return {
            'strategy': 'grid_tpe',
            'grid_phase': grid_results,
            'tpe_phase': tpe_results,
            'best_score': best_result.get('score') if best_result else None,
            'best_parameters': best_result.get('parameters') if best_result else None,
            'total_evaluations': len(all_evaluations),
            'evaluations': all_evaluations
        }
    
    def _run_grid_phase(self, objective_function: Callable, data: Optional[Any],
                       n_iterations: int, use_hardware_acceleration: bool) -> Dict[str, Any]:
        """Run grid search phase."""
        # Generate grid points
        grid_points = self._generate_grid_points(n_iterations)
        
        evaluations = []
        best_score = -np.inf
        best_params = None
        
        for i, params in enumerate(grid_points):
            try:
                # Validate parameters
                is_valid, errors = self.validate_parameters(params)
                if not is_valid:
                    tprint_debug(f"Grid point {i} invalid: {errors}")
                    continue
                
                # Evaluate objective function
                if data is not None:
                    score = objective_function(params, data)
                else:
                    score = objective_function(params)
                
                evaluation = {
                    'iteration': i,
                    'parameters': params.copy(),
                    'score': score,
                    'timestamp': get_current_datetime().isoformat(),
                    'phase': 'grid'
                }
                
                evaluations.append(evaluation)
                self.optimization_history.append(evaluation)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    tprint_debug(f"Grid iteration {i}: New best {score:.4f}")
                
            except Exception as e:
                tprint_warning(f"Grid evaluation {i} failed: {e}")
                continue
        
        return {
            'best_score': best_score,
            'best_parameters': best_params,
            'evaluations': evaluations,
            'grid_points_generated': len(grid_points),
            'successful_evaluations': len(evaluations)
        }
    
    def _run_tpe_phase(self, objective_function: Callable, data: Optional[Any],
                      n_iterations: int, initial_best: Optional[Dict[str, Any]],
                      use_hardware_acceleration: bool) -> Dict[str, Any]:
        """Run TPE optimization phase."""
        evaluations = []
        best_score = -np.inf
        best_params = initial_best
        
        if initial_best:
            best_score = 0.0  # Will be updated with actual score
        
        for i in range(n_iterations):
            try:
                # Sample parameters (for now using random, would integrate with actual TPE)
                if i == 0 and initial_best:
                    params = initial_best.copy()
                else:
                    params = self._sample_tpe_parameters(i, evaluations)
                
                # Validate parameters
                is_valid, errors = self.validate_parameters(params)
                if not is_valid:
                    tprint_debug(f"TPE iteration {i} invalid: {errors}")
                    continue
                
                # Evaluate objective function
                if data is not None:
                    score = objective_function(params, data)
                else:
                    score = objective_function(params)
                
                evaluation = {
                    'iteration': i,
                    'parameters': params.copy(),
                    'score': score,
                    'timestamp': get_current_datetime().isoformat(),
                    'phase': 'tpe'
                }
                
                evaluations.append(evaluation)
                self.optimization_history.append(evaluation)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    tprint_debug(f"TPE iteration {i}: New best {score:.4f}")
                
            except Exception as e:
                tprint_warning(f"TPE evaluation {i} failed: {e}")
                continue
        
        return {
            'best_score': best_score,
            'best_parameters': best_params,
            'evaluations': evaluations,
            'successful_evaluations': len(evaluations)
        }
    
    def _generate_grid_points(self, max_points: int) -> List[Dict[str, Any]]:
        """Generate grid points for grid search."""
        if not self.parameters:
            return []
        
        # Create parameter grids
        param_grids = {}
        for name, param in self.parameters.items():
            if param.param_type == SearchSpaceType.CONTINUOUS:
                n_points = min(self.config.grid_sample_density, max_points)
                if param.log_scale and param.min_value > 0:
                    points = np.logspace(safe_log(param.min_value), safe_log(param.max_value), n_points)
                else:
                    points = np.linspace(param.min_value, param.max_value, n_points)
                param_grids[name] = points
                
            elif param.param_type == SearchSpaceType.DISCRETE:
                step = param.step or 1
                points = list(range(param.min_value, param.max_value + 1, step))
                # Limit grid size
                if len(points) > self.config.grid_sample_density:
                    step_size = len(points) // self.config.grid_sample_density
                    points = points[::step_size]
                param_grids[name] = points
                
            elif param.param_type == SearchSpaceType.CATEGORICAL:
                param_grids[name] = param.choices
        
        # Generate all combinations (limited to max_points)
        param_names = list(param_grids.keys())
        param_values = list(param_grids.values())
        
        grid_points = []
        for i, combination in enumerate(itertools.product(*param_values)):
            if i >= max_points:
                break
            params = dict(zip(param_names, combination))
            grid_points.append(params)
        
        tprint_info(f"Generated {len(grid_points)} grid points")
        return grid_points
    
    def _sample_tpe_parameters(self, iteration: int, previous_evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Sample parameters for TPE (simplified version)."""
        # For now, use intelligent random sampling based on previous results
        if previous_evaluations and len(previous_evaluations) > 5:
            # Get best performing parameters
            best_eval = max(previous_evaluations, key=lambda x: x.get('score', -np.inf))
            best_params = best_eval['parameters']
            
            # Add noise around best parameters
            params = {}
            for name, param in self.parameters.items():
                if name in best_params:
                    best_value = best_params[name]
                    
                    if param.param_type == SearchSpaceType.CONTINUOUS:
                        # Add Gaussian noise
                        noise_scale = (param.max_value - param.min_value) * 0.1
                        new_value = best_value + np.random.normal(0, noise_scale)
                        new_value = np.clip(new_value, param.min_value, param.max_value)
                        params[name] = new_value
                    elif param.param_type == SearchSpaceType.DISCRETE:
                        # Add small integer noise
                        noise = np.random.randint(-2, 3)
                        new_value = best_value + noise
                        new_value = np.clip(new_value, param.min_value, param.max_value)
                        params[name] = int(new_value)
                    else:
                        # For categorical, sometimes keep best, sometimes sample
                        if np.random.random() < 0.7:
                            params[name] = best_value
                        else:
                            params[name] = param.sample()
                else:
                    params[name] = param.sample()
        else:
            # Random sampling for early iterations
            params = self.sample_parameters(1)[0]
        
        return params
    
    def _optimize_grid(self, objective_function: Callable, data: Optional[Any],
                      use_hardware_acceleration: bool) -> Dict[str, Any]:
        """Pure grid search optimization."""
        grid_results = self._run_grid_phase(objective_function, data, self.config.max_iterations, use_hardware_acceleration)
        
        return {
            'strategy': 'grid',
            'best_score': grid_results.get('best_score'),
            'best_parameters': grid_results.get('best_parameters'),
            'total_evaluations': len(grid_results.get('evaluations', [])),
            'evaluations': grid_results.get('evaluations', [])
        }
    
    def _optimize_tpe(self, objective_function: Callable, data: Optional[Any],
                     use_hardware_acceleration: bool) -> Dict[str, Any]:
        """Pure TPE optimization."""
        tpe_results = self._run_tpe_phase(objective_function, data, self.config.max_iterations, None, use_hardware_acceleration)
        
        return {
            'strategy': 'tpe',
            'best_score': tpe_results.get('best_score'),
            'best_parameters': tpe_results.get('best_parameters'),
            'total_evaluations': len(tpe_results.get('evaluations', [])),
            'evaluations': tpe_results.get('evaluations', [])
        }
    
    def _optimize_random(self, objective_function: Callable, data: Optional[Any],
                        use_hardware_acceleration: bool) -> Dict[str, Any]:
        """Random search optimization."""
        evaluations = []
        best_score = -np.inf
        best_params = None
        
        for i in range(self.config.max_iterations):
            try:
                params = self.sample_parameters(1)[0]
                
                if data is not None:
                    score = objective_function(params, data)
                else:
                    score = objective_function(params)
                
                evaluation = {
                    'iteration': i,
                    'parameters': params.copy(),
                    'score': score,
                    'timestamp': get_current_datetime().isoformat(),
                    'phase': 'random'
                }
                
                evaluations.append(evaluation)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                
            except Exception as e:
                tprint_warning(f"Random evaluation {i} failed: {e}")
                continue
        
        return {
            'strategy': 'random',
            'best_score': best_score,
            'best_parameters': best_params,
            'total_evaluations': len(evaluations),
            'evaluations': evaluations
        }
    
    def _optimize_hierarchical(self, objective_function: Callable, data: Optional[Any],
                              use_hardware_acceleration: bool) -> Dict[str, Any]:
        """Hierarchical search optimization (placeholder for future implementation)."""
        tprint_warning("Hierarchical optimization not yet implemented, falling back to Grid+TPE")
        return self._optimize_grid_tpe(objective_function, data, use_hardware_acceleration)
    
    def _save_results(self) -> None:
        """Save optimization results with comprehensive data."""
        try:
            timestamp = get_current_datetime().strftime('%Y%m%d_%H%M%S')
            results_file = Path(self.config.results_dir) / f"{self.config.name}_results_{timestamp}.json"
            
            results_data = {
                'config': {
                    'name': self.config.name,
                    'description': self.config.description,
                    'max_iterations': self.config.max_iterations,
                    'optimization_strategy': self.config.optimization_strategy.value,
                    'enable_m1_optimization': self.config.enable_m1_optimization,
                    'enable_parallel_processing': self.config.enable_parallel_processing
                },
                'search_space': {
                    'parameters': {
                        name: {
                            'type': param.param_type.value,
                            'min_value': param.min_value,
                            'max_value': param.max_value,
                            'choices': param.choices,
                            'description': param.description
                        } for name, param in self.parameters.items()
                    },
                    'space_size': self.get_parameter_space_size()
                },
                'optimization_history': self.optimization_history,
                'best_result': self.best_result,
                'hardware_info': {
                    'm1_available': M1_HARDWARE_AVAILABLE,
                    'matrix_ops_available': MATRIX_OPS_AVAILABLE,
                    'tpe_grid_available': TPE_GRID_AVAILABLE
                },
                'timestamp': timestamp
            }
            
            safe_json_dump(results_data, results_file)
            tprint_info(f"💾 Results saved to {results_file}")
            
        except Exception as e:
            tprint_error(f"Failed to save results: {e}")
    
    def load_results(self, results_file: Union[str, Path]) -> bool:
        """Load optimization results from file."""
        try:
            results_data = safe_json_load(results_file)
            if results_data:
                self.optimization_history = results_data.get('optimization_history', [])
                self.best_result = results_data.get('best_result')
                tprint_info(f"📁 Results loaded from {results_file}")
                return True
            return False
        except Exception as e:
            tprint_error(f"Failed to load results: {e}")
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of search space and results."""
        return {
            'config': {
                'name': self.config.name,
                'description': self.config.description,
                'max_iterations': self.config.max_iterations,
                'optimization_strategy': self.config.optimization_strategy.value,
                'enable_m1_optimization': self.config.enable_m1_optimization
            },
            'search_space': {
                'parameter_count': len(self.parameters),
                'parameter_types': {name: param.param_type.value for name, param in self.parameters.items()},
                'space_size': self.get_parameter_space_size(),
                'continuous_params': [name for name, param in self.parameters.items() 
                                    if param.param_type == SearchSpaceType.CONTINUOUS],
                'discrete_params': [name for name, param in self.parameters.items() 
                                  if param.param_type == SearchSpaceType.DISCRETE],
                'categorical_params': [name for name, param in self.parameters.items() 
                                     if param.param_type == SearchSpaceType.CATEGORICAL]
            },
            'optimization_results': {
                'total_evaluations': len(self.optimization_history),
                'best_score': self.best_result.get('score') if self.best_result else None,
                'best_parameters': self.best_result.get('parameters') if self.best_result else None
            },
            'hardware_status': {
                'm1_optimization': M1_HARDWARE_AVAILABLE and self.config.enable_m1_optimization,
                'matrix_operations': MATRIX_OPS_AVAILABLE,
                'tpe_grid_optimization': TPE_GRID_AVAILABLE,
                'parallel_processing': self.config.enable_parallel_processing
            }
        }


def create_default_nas_search_space() -> SearchSpace:
    """
    Create a default Neural Architecture Search space with comprehensive parameters.
    
    Returns:
        SearchSpace instance configured for NAS
    """
    config = SearchSpaceConfig(
        name="default_nas_search_space",
        description="Comprehensive NAS search space with Grid+TPE optimization",
        max_iterations=100,
        optimization_strategy=OptimizationStrategy.GRID_TPE,
        early_stopping_patience=15,
        validation_split=0.2,
        cross_validation_folds=5,
        enable_m1_optimization=True,
        enable_parallel_processing=True,
        max_parallel_jobs=4,
        save_intermediate_results=True,
        results_dir="nas_search_results",
        grid_phase_ratio=0.3,
        tpe_phase_ratio=0.7,
        grid_sample_density=5,
        tpe_n_startup_trials=10
    )
    
    search_space = SearchSpace(config)
    
    # Add neural architecture parameters
    search_space.add_continuous_parameter("learning_rate", 1e-5, 1e-1, log_scale=True, 
                                         description="Learning rate for optimizer")
    search_space.add_discrete_parameter("hidden_layers", 1, 10, description="Number of hidden layers")
    search_space.add_discrete_parameter("hidden_units", 32, 1024, step=32, description="Units per hidden layer")
    search_space.add_categorical_parameter("activation", ["relu", "tanh", "sigmoid", "leaky_relu", "gelu"],
                                          description="Activation function")
    search_space.add_categorical_parameter("optimizer", ["adam", "sgd", "rmsprop", "adamw"],
                                          description="Optimizer type")
    search_space.add_continuous_parameter("dropout_rate", 0.0, 0.8, description="Dropout rate")
    search_space.add_discrete_parameter("batch_size", 16, 512, step=16, description="Training batch size")
    search_space.add_categorical_parameter("regularization", ["l1", "l2", "elastic_net", "none"],
                                          description="Regularization type")
    search_space.add_continuous_parameter("l1_alpha", 1e-6, 1e-2, log_scale=True, description="L1 regularization strength")
    search_space.add_continuous_parameter("l2_alpha", 1e-6, 1e-2, log_scale=True, description="L2 regularization strength")
    
    tprint_info("✅ Created comprehensive NAS search space")
    tprint_structured(search_space.get_summary())
    
    return search_space


def create_tree_search_space() -> SearchSpace:
    """
    Create a search space optimized for tree-based models.
    
    Returns:
        SearchSpace instance configured for tree models
    """
    config = SearchSpaceConfig(
        name="tree_model_search_space",
        description="Search space optimized for tree-based models (RF, XGBoost, LightGBM)",
        max_iterations=80,
        optimization_strategy=OptimizationStrategy.GRID_TPE,
        enable_m1_optimization=True,
        results_dir="tree_search_results",
        grid_phase_ratio=0.4,  # More emphasis on grid for tree models
        tpe_phase_ratio=0.6
    )
    
    search_space = SearchSpace(config)
    
    # Tree-specific parameters
    search_space.add_discrete_parameter("n_estimators", 10, 1000, description="Number of trees")
    search_space.add_discrete_parameter("max_depth", 3, 20, description="Maximum tree depth")
    search_space.add_discrete_parameter("min_samples_split", 2, 100, description="Minimum samples to split")
    search_space.add_discrete_parameter("min_samples_leaf", 1, 50, description="Minimum samples per leaf")
    search_space.add_categorical_parameter("max_features", ["auto", "sqrt", "log2", 0.5, 0.7, 0.9],
                                          description="Maximum features per split")
    search_space.add_continuous_parameter("learning_rate", 0.01, 0.3, description="Learning rate (for boosting)")
    search_space.add_continuous_parameter("subsample", 0.5, 1.0, description="Subsample ratio")
    search_space.add_continuous_parameter("colsample_bytree", 0.5, 1.0, description="Feature subsample ratio")
    search_space.add_continuous_parameter("reg_alpha", 0.0, 10.0, log_scale=True, description="L1 regularization")
    search_space.add_continuous_parameter("reg_lambda", 0.0, 10.0, log_scale=True, description="L2 regularization")
    
    tprint_info("✅ Created tree model search space")
    
    return search_space


# Example usage
if __name__ == "__main__":
    # Create and test default NAS search space
    search_space = create_default_nas_search_space()
    
    # Example objective function
    def example_objective(params: Dict[str, Any]) -> float:
        """Example objective function for testing."""
        score = 0.0
        
        # Scoring based on parameter values
        lr = params.get('learning_rate', 1e-3)
        if 1e-4 <= lr <= 1e-2:
            score += 0.3
        
        layers = params.get('hidden_layers', 3)
        if 2 <= layers <= 6:
            score += 0.2
        
        units = params.get('hidden_units', 128)
        if 64 <= units <= 512:
            score += 0.2
        
        activation = params.get('activation', 'relu')
        if activation in ['relu', 'gelu']:
            score += 0.15
        
        dropout = params.get('dropout_rate', 0.2)
        if 0.1 <= dropout <= 0.5:
            score += 0.15
        
        # Add some randomness to simulate real optimization
        score += np.random.normal(0, 0.05)
        
        return max(0, score)  # Ensure non-negative score
    
    # Run optimization
    tprint_info("🚀 Running example optimization...")
    results = search_space.optimize(example_objective)
    
    # Display results
    tprint_structured(results)