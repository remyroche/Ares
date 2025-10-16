"""
Optimization Configuration for Multi-Horizon Optimizer

This module defines configuration classes and enums for the optimized
multi-horizon optimizer using ml_commons utilities.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

class ModelType(Enum):
    """Model types for optimization."""
    ANALYST = "analyst"
    TACTICIAN = "tactician"
    BOTH = "both"

class OptimizationMethod(Enum):
    """Optimization methods."""
    GRID_SEARCH = "grid_search"
    BAYESIAN_TPE = "bayesian_tpe"
    GRID_BAYESIAN = "grid_bayesian"  # Grid search followed by Bayesian TPE

class ValidationLevel(Enum):
    """Validation levels."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    COMPREHENSIVE = "comprehensive"

@dataclass
class GridSearchConfig:
    """Configuration for grid search optimization."""
    # Coarse grid settings
    coarse_grid_points: int = 5
    coarse_enabled: bool = True

    # Fine grid settings
    fine_grid_points: int = 10
    fine_enabled: bool = True
    fine_range_percentage: float = 0.2  # 20% around best coarse result

    # Grid search parameters
    enable_parallel: bool = True
    max_workers: int = 4
    timeout_seconds: int = 300

@dataclass
class BayesianTPEConfig:
    """Configuration for Bayesian TPE optimization."""
    # TPE settings
    n_trials: int = 50
    n_startup_trials: int = 10
    n_ei_candidates: int = 24

    # Optimization settings
    enable_parallel: bool = True
    max_workers: int = 4
    timeout_seconds: int = 600

    # Pruning settings
    enable_pruning: bool = True
    pruning_patience: int = 5
    pruning_min_trials: int = 10

@dataclass
class ValidationConfig:
    """Configuration for validation framework."""
    # Validation level
    validation_level: ValidationLevel = ValidationLevel.COMPREHENSIVE

    # Cross-validation settings
    cv_folds: int = 5
    cv_strategy: str = "time_series"  # time_series, stratified, kfold

    # Statistical validation
    enable_statistical_validation: bool = True
    min_information_coefficient: float = 0.05
    min_signal_to_noise_ratio: float = 1.0
    min_hit_rate: float = 0.55

    # Economic validation
    enable_economic_validation: bool = True
    max_transaction_cost_ratio: float = 0.1
    min_sharpe_ratio: float = 0.5
    max_drawdown_threshold: float = 0.2

    # Market microstructure validation
    enable_microstructure_validation: bool = True
    min_liquidity_score: float = 0.7
    min_volatility_stability: float = 0.6

@dataclass
class OptimizationConfig:
    """Main optimization configuration."""
    # Model settings
    model_type: ModelType = ModelType.BOTH
    base_timeframe_analyst: int = 15  # 15 minutes
    base_timeframe_tactician: int = 5  # 5 minutes
    horizon_range: Tuple[int, int] = (1, 16)  # 1-16 periods

    # Optimization method
    optimization_method: OptimizationMethod = OptimizationMethod.GRID_BAYESIAN

    # Grid search configuration
    grid_search_config: GridSearchConfig = field(default_factory=GridSearchConfig)

    # Bayesian TPE configuration
    bayesian_tpe_config: BayesianTPEConfig = field(default_factory=BayesianTPEConfig)

    # Validation configuration
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)

    # Performance settings
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    enable_monitoring: bool = True

    # Fast fail settings
    fast_fail_on_optimization_failure: bool = True
    min_optimization_score: float = 0.3
    min_validation_score: float = 0.5

    # Logging settings
    enable_detailed_logging: bool = True
    log_optimization_progress: bool = True

@dataclass
class SearchSpace:
    """Search space definition for optimization."""
    # Time horizons
    horizon_immediate: Dict[str, Any] = field(default_factory=lambda: {
        'type': 'int',
        'low': 1,
        'high': 16
    })

    horizon_short: Dict[str, Any] = field(default_factory=lambda: {
        'type': 'int',
        'low': 1,
        'high': 16
    })

    # Profit targets
    target_micro: Dict[str, Any] = field(default_factory=lambda: {
        'type': 'float',
        'low': 0.001,
        'high': 0.010,
        'log': True
    })

    target_small: Dict[str, Any] = field(default_factory=lambda: {
        'type': 'float',
        'low': 0.002,
        'high': 0.015,
        'log': True
    })

    target_medium: Dict[str, Any] = field(default_factory=lambda: {
        'type': 'float',
        'low': 0.003,
        'high': 0.020,
        'log': True
    })

    target_good: Dict[str, Any] = field(default_factory=lambda: {
        'type': 'float',
        'low': 0.005,
        'high': 0.025,
        'log': True
    })

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ml_commons utilities."""
        return {
            'horizon_immediate': self.horizon_immediate,
            'horizon_short': self.horizon_short,
            'target_micro': self.target_micro,
            'target_small': self.target_small,
            'target_medium': self.target_medium,
            'target_good': self.target_good
        }

@dataclass
class OptimizationResult:
    """Result of optimization process."""
    # Optimization results
    optimal_horizons: Dict[str, int]
    optimal_targets: Dict[str, float]
    optimization_score: float
    validation_score: float

    # Performance metrics
    performance_metrics: Dict[str, float]
    optimization_time: float

    # Method information
    optimization_method: str
    n_trials: int
    convergence_info: Dict[str, Any]

    # Timestamps
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'optimal_horizons': self.optimal_horizons,
            'optimal_targets': self.optimal_targets,
            'optimization_score': self.optimization_score,
            'validation_score': self.validation_score,
            'performance_metrics': self.performance_metrics,
            'optimization_time': self.optimization_time,
            'optimization_method': self.optimization_method,
            'n_trials': self.n_trials,
            'convergence_info': self.convergence_info,
            'timestamp': self.timestamp.isoformat()
        }
