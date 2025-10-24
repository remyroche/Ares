"""
Configuration validation for ML optimization utilities.

This module provides Pydantic-based validation for all optimization configurations,
ensuring type safety and parameter validation.
"""

from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum

# Handle pydantic import gracefully
try:
    from pydantic import BaseModel, Field, validator, root_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Create mock pydantic classes for basic functionality
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    def Field(default=None, **kwargs):
        return default
    
    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def root_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Handle numpy import gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create a mock numpy for basic functionality
    class MockNumpy:
        def __getattr__(self, name):
            raise ImportError("NumPy is required for validation functionality")
    np = MockNumpy()


class OptimizationStrategy(str, Enum):
    """Available optimization strategies."""
    BAYESIAN = "bayesian"
    BOHB = "bohb"
    GRID = "grid"
    RANDOM = "random"
    HIERARCHICAL = "hierarchical"


class AresExecutionMode(str, Enum):
    """Ares launcher execution modes."""
    LIGHT = "light"
    BLANK = "blank"
    FULL = "full"


class PrunerStrategy(str, Enum):
    """Available pruner strategies."""
    ADAPTIVE = "adaptive"
    CONFIDENCE_BASED = "confidence_based"
    MULTI_FIDELITY = "multi_fidelity"
    HYPERBAND = "hyperband"
    SUCCESSIVE_HALVING = "successive_halving"
    MEDIAN = "median"


class ParameterType(str, Enum):
    """Parameter types for search spaces."""
    FLOAT = "float"
    INT = "int"
    CATEGORICAL = "categorical"


class SearchSpaceParameter(BaseModel):
    """Single parameter in search space."""
    type: ParameterType
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log: bool = False
    
    @validator('low', 'high')
    def validate_numeric_bounds(cls, v, values):
        if v is not None and 'type' in values:
            if values['type'] in ['float', 'int'] and v < 0 and values.get('log', False):
                raise ValueError("Log scale parameters must have positive bounds")
        return v
    
    @validator('choices')
    def validate_choices(cls, v, values):
        if 'type' in values and values['type'] == 'categorical' and not v:
            raise ValueError("Categorical parameters must have choices")
        return v
    
    @root_validator
    def validate_parameter_consistency(cls, values):
        param_type = values.get('type')
        low = values.get('low')
        high = values.get('high')
        choices = values.get('choices')
        
        if param_type == 'categorical':
            if low is not None or high is not None:
                raise ValueError("Categorical parameters should not have numeric bounds")
            if not choices:
                raise ValueError("Categorical parameters must have choices")
        elif param_type in ['float', 'int']:
            if low is None or high is None:
                raise ValueError("Numeric parameters must have both low and high bounds")
            if low >= high:
                raise ValueError("Low bound must be less than high bound")
        
        return values


class HPOConfig(BaseModel):
    """Validated configuration for hyperparameter optimization."""
    
    # Basic settings
    n_trials: int = Field(ge=1, le=10000, description="Number of optimization trials")
    timeout: Optional[float] = Field(gt=0, description="Timeout in seconds")
    random_state: Optional[int] = Field(ge=0, description="Random seed for reproducibility")
    
    # Optimization strategy
    strategy: OptimizationStrategy = Field(default=OptimizationStrategy.BAYESIAN)
    
    # Ares launcher integration
    ares_execution_mode: AresExecutionMode = Field(default=AresExecutionMode.FULL)
    enable_mode_scaling: bool = Field(default=True)
    auto_detect_mode: bool = Field(default=True)
    
    # Bayesian optimization settings
    n_startup_trials: int = Field(ge=1, le=100, default=10)
    n_ei_candidates: int = Field(ge=1, le=100, default=24)
    multivariate: bool = Field(default=True)
    group: bool = Field(default=True)
    
    # BOHB settings
    min_budget: float = Field(ge=0.01, le=1.0, default=0.1)
    max_budget: float = Field(ge=0.01, le=1.0, default=1.0)
    reduction_factor: float = Field(ge=1.1, le=10.0, default=3.0)
    n_brackets: int = Field(ge=1, le=10, default=1)
    
    # Grid search settings
    enable_staged_optimization: bool = Field(default=True)
    coarse_grid_points: int = Field(ge=2, le=20, default=5)
    fine_grid_points: int = Field(ge=2, le=20, default=5)
    coarse_grid_trials: int = Field(ge=1, le=1000, default=25)
    fine_grid_trials: int = Field(ge=1, le=1000, default=25)
    tpe_trials: int = Field(ge=1, le=1000, default=50)
    
    # Hardware optimization
    enable_hardware_optimization: bool = Field(default=True)
    enable_vectorbt: bool = Field(default=True)
    enable_parallel: bool = Field(default=True)
    max_workers: int = Field(ge=1, le=32, default=4)
    
    # Monitoring and diagnostics
    enable_monitoring: bool = Field(default=True)
    enable_diagnostics: bool = Field(default=True)
    enable_overfitting_detection: bool = Field(default=True)
    
    # Cross-validation
    cv_folds: int = Field(ge=2, le=20, default=5)
    enable_time_series_cv: bool = Field(default=True)
    scoring: str = Field(default="neg_mean_squared_error")
    
    # Caching and persistence
    enable_caching: bool = Field(default=True)
    cache_dir: str = Field(default="./hpo_cache")
    save_results: bool = Field(default=True)
    results_dir: str = Field(default="./hpo_results")
    
    # Additional configuration
    enable_detailed_logging: bool = Field(default=False)
    overfitting_threshold: float = Field(ge=0.0, le=1.0, default=0.1)
    
    @validator('n_startup_trials')
    def validate_startup_trials(cls, v, values):
        n_trials = values.get('n_trials', 100)
        if v >= n_trials:
            raise ValueError("n_startup_trials must be less than n_trials")
        return v
    
    @validator('min_budget', 'max_budget')
    def validate_budget_range(cls, v, values):
        if 'min_budget' in values and 'max_budget' in values:
            min_budget = values.get('min_budget', 0.1)
            max_budget = values.get('max_budget', 1.0)
            if min_budget >= max_budget:
                raise ValueError("min_budget must be less than max_budget")
        return v
    
    @validator('timeout')
    def validate_timeout_reasonable(cls, v, values):
        if v is not None:
            n_trials = values.get('n_trials', 100)
            # Warn if timeout might be too short for trials
            if v < n_trials * 0.1:  # Less than 0.1 seconds per trial
                import warnings
                warnings.warn(f"Timeout {v}s might be too short for {n_trials} trials")
        return v
    
    @root_validator
    def validate_strategy_consistency(cls, values):
        strategy = values.get('strategy')
        n_trials = values.get('n_trials', 100)
        
        if strategy == OptimizationStrategy.GRID:
            # For grid search, ensure we have reasonable trial counts
            coarse_trials = values.get('coarse_grid_trials', 25)
            fine_trials = values.get('fine_grid_trials', 25)
            tpe_trials = values.get('tpe_trials', 50)
            
            total_staged = coarse_trials + fine_trials + tpe_trials
            if total_staged > n_trials:
                raise ValueError("Total staged trials exceed n_trials")
        
        return values


class HPOPhaseConfig(BaseModel):
    """Validated configuration for hierarchical HPO phases."""
    
    phase_name: str = Field(min_length=1, description="Name of the optimization phase")
    models: Dict[str, Any] = Field(description="Models to optimize in this phase")
    search_spaces: Dict[str, Dict[str, SearchSpaceParameter]] = Field(description="Search spaces for each model")
    n_trials: int = Field(ge=1, le=1000, default=100)
    timeout_seconds: Optional[int] = Field(gt=0, description="Timeout in seconds")
    enable_pruning: bool = Field(default=True)
    cv_folds: int = Field(ge=2, le=20, default=5)
    scoring_metric: str = Field(default="neg_mean_squared_error")
    direction: str = Field(regex="^(maximize|minimize)$", default="maximize")


class PrunerConfig(BaseModel):
    """Validated configuration for pruner system."""
    
    strategy: PrunerStrategy = Field(default=PrunerStrategy.ADAPTIVE)
    ares_mode: AresExecutionMode = Field(default=AresExecutionMode.FULL)
    base_patience: int = Field(ge=1, le=100, default=10)
    improvement_threshold: float = Field(ge=0.0, le=1.0, default=0.001)
    enable_aggressive_pruning: bool = Field(default=False)
    min_resource: int = Field(ge=1, default=1)
    max_resource: int = Field(ge=1, default=100)
    reduction_factor: float = Field(ge=1.1, le=10.0, default=3.0)
    
    @validator('max_resource')
    def validate_resource_range(cls, v, values):
        min_resource = values.get('min_resource', 1)
        if v <= min_resource:
            raise ValueError("max_resource must be greater than min_resource")
        return v


def validate_search_space(search_space: Dict[str, Any]) -> Dict[str, SearchSpaceParameter]:
    """Validate and convert search space to validated format."""
    validated_space = {}
    
    for param_name, param_config in search_space.items():
        try:
            validated_space[param_name] = SearchSpaceParameter(**param_config)
        except Exception as e:
            from .exceptions import SearchSpaceError
            raise SearchSpaceError(f"Invalid parameter '{param_name}': {e}", {
                'parameter': param_name,
                'config': param_config
            })
    
    return validated_space


def validate_hpo_config(config_dict: Dict[str, Any]) -> HPOConfig:
    """Validate HPO configuration dictionary."""
    try:
        return HPOConfig(**config_dict)
    except Exception as e:
        from .exceptions import ConfigurationError
        raise ConfigurationError(f"Invalid HPO configuration: {e}", {
            'config': config_dict,
            'error': str(e)
        })


def validate_pruner_config(config_dict: Dict[str, Any]) -> PrunerConfig:
    """Validate pruner configuration dictionary."""
    try:
        return PrunerConfig(**config_dict)
    except Exception as e:
        from .exceptions import ConfigurationError
        raise ConfigurationError(f"Invalid pruner configuration: {e}", {
            'config': config_dict,
            'error': str(e)
        })