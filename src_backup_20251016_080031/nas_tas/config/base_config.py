"""
Unified Base Configuration for NAS/TAS Systems

This module provides the foundational configuration classes that are shared
between NAS and TAS implementations, eliminating redundancy and ensuring
consistent behavior across both systems.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
import json
from pathlib import Path
from datetime import datetime


class ArchitectureType(Enum):
    """Types of architectures supported by the unified system."""
    NEURAL_ONLY = "neural_only"
    TREE_ONLY = "tree_only"
    HYBRID_NEURAL_TREE = "hybrid_neural_tree"
    ENSEMBLE = "ensemble"
    META_LEARNING = "meta_learning"


class OptimizationMode(Enum):
    """Optimization modes for architecture search."""
    SINGLE_OBJECTIVE = "single_objective"
    MULTI_OBJECTIVE = "multi_objective"
    REGIME_AWARE = "regime_aware"
    REAL_TIME = "real_time"
    CONTINUAL = "continual"


class SearchStrategy(Enum):
    """Search strategies for architecture exploration."""
    RANDOM = "random"
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT = "reinforcement"
    META_LEARNING = "meta_learning"
    HYBRID = "hybrid"


class ValidationMethod(Enum):
    """Validation methods for model evaluation."""
    HOLDOUT = "holdout"
    CROSS_VALIDATION = "cross_validation"
    TIME_SERIES_SPLIT = "time_series_split"
    WALK_FORWARD = "walk_forward"


@dataclass
class UnifiedArchitectureConfig:
    """
    Unified base configuration for both NAS and TAS systems.
    
    This class consolidates common configuration parameters that are shared
    between neural architecture search and tree architecture search systems,
    eliminating redundancy and ensuring consistent behavior.
    """
    
    # Core architecture settings
    architecture_type: ArchitectureType = ArchitectureType.HYBRID_NEURAL_TREE
    optimization_mode: OptimizationMode = OptimizationMode.REGIME_AWARE
    search_strategy: SearchStrategy = SearchStrategy.HYBRID
    
    # Timeframe configuration
    primary_timeframe: str = "15m"
    secondary_timeframe: str = "5m"
    micro_timeframe: str = "1m"
    regime_detection_window: int = 100
    
    # Regime configuration
    n_regimes: int = 8
    min_regime_duration: int = 15  # minutes
    max_regime_duration: int = 180  # minutes
    regime_stability_threshold: float = 0.7
    data_driven_regimes: bool = True
    
    # Search parameters
    max_search_iterations: int = 100
    max_search_time_seconds: int = 3600
    population_size: int = 50
    generations: int = 100
    early_stopping_patience: int = 20
    min_improvement_threshold: float = 0.001
    
    # Validation settings
    validation_method: ValidationMethod = ValidationMethod.TIME_SERIES_SPLIT
    validation_split: float = 0.2
    cv_folds: int = 5
    test_size: float = 0.2
    
    # Performance settings
    n_jobs: int = -1
    random_state: int = 42
    verbose: bool = True
    enable_parallel_processing: bool = True
    
    # Memory and resource management
    max_memory_usage_gb: float = 8.0
    enable_memory_optimization: bool = True
    enable_gpu_acceleration: bool = True
    batch_size: int = 1000
    
    # Financial and trading parameters
    economic_significance_threshold: float = 0.7
    trading_viability_threshold: float = 0.6
    max_drawdown_threshold: float = 0.15
    risk_adjusted_return_threshold: float = 0.1
    transaction_cost_penalty: float = 0.001
    slippage_assumption: float = 0.0005
    
    # Model constraints
    min_model_confidence: float = 0.6
    max_model_complexity: int = 100
    min_position_size: float = 0.01
    max_position_size: float = 0.1
    
    # Output and logging
    save_intermediate_results: bool = True
    save_best_models: bool = True
    output_dir: str = "nas_tas_results"
    log_level: str = "INFO"
    enable_detailed_logging: bool = True
    
    # Advanced features
    enable_uncertainty_quantification: bool = True
    enable_robustness_testing: bool = True
    enable_hyperparameter_optimization: bool = True
    enable_feature_importance: bool = True
    enable_early_stopping: bool = True
    
    # Integration settings
    integrate_with_existing_pipeline: bool = True
    backward_compatibility: bool = True
    output_format: str = "comprehensive"
    
    # Custom parameters (for extensibility)
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        self._validate_configuration()
        self._set_default_paths()
    
    def _validate_configuration(self):
        """Validate configuration parameters."""
        # Validate numeric ranges
        if not (0 < self.validation_split < 1):
            raise ValueError(f"validation_split must be between 0 and 1, got {self.validation_split}")
        
        if not (0 < self.test_size < 1):
            raise ValueError(f"test_size must be between 0 and 1, got {self.test_size}")
        
        if self.n_regimes < 2:
            raise ValueError(f"n_regimes must be at least 2, got {self.n_regimes}")
        
        if self.population_size < 10:
            raise ValueError(f"population_size must be at least 10, got {self.population_size}")
        
        if self.max_memory_usage_gb <= 0:
            raise ValueError(f"max_memory_usage_gb must be positive, got {self.max_memory_usage_gb}")
        
        # Validate thresholds
        thresholds = [
            self.economic_significance_threshold,
            self.trading_viability_threshold,
            self.max_drawdown_threshold,
            self.risk_adjusted_return_threshold,
            self.min_model_confidence,
            self.regime_stability_threshold
        ]
        
        for threshold in thresholds:
            if not (0 <= threshold <= 1):
                raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
    
    def _set_default_paths(self):
        """Set default paths and directories."""
        if not Path(self.output_dir).is_absolute():
            self.output_dir = str(Path.cwd() / self.output_dir)
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {}
        
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, Enum):
                config_dict[field_name] = field_value.value
            elif isinstance(field_value, Path):
                config_dict[field_name] = str(field_value)
            else:
                config_dict[field_name] = field_value
        
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'UnifiedArchitectureConfig':
        """Create configuration from dictionary."""
        # Convert enum values back to enums
        if 'architecture_type' in config_dict:
            config_dict['architecture_type'] = ArchitectureType(config_dict['architecture_type'])
        
        if 'optimization_mode' in config_dict:
            config_dict['optimization_mode'] = OptimizationMode(config_dict['optimization_mode'])
        
        if 'search_strategy' in config_dict:
            config_dict['search_strategy'] = SearchStrategy(config_dict['search_strategy'])
        
        if 'validation_method' in config_dict:
            config_dict['validation_method'] = ValidationMethod(config_dict['validation_method'])
        
        return cls(**config_dict)
    
    def save_to_file(self, filepath: Union[str, Path]) -> bool:
        """Save configuration to file."""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"Failed to save configuration: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'UnifiedArchitectureConfig':
        """Load configuration from file."""
        try:
            filepath = Path(filepath)
            
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            
            return cls.from_dict(config_dict)
        except Exception as e:
            print(f"Failed to load configuration: {e}")
            raise
    
    def update(self, updates: Dict[str, Any]) -> 'UnifiedArchitectureConfig':
        """Create new configuration with updates."""
        current_dict = self.to_dict()
        current_dict.update(updates)
        return self.from_dict(current_dict)
    
    def get_timeframe_config(self) -> Dict[str, Any]:
        """Get timeframe-specific configuration."""
        return {
            'primary_timeframe': self.primary_timeframe,
            'secondary_timeframe': self.secondary_timeframe,
            'micro_timeframe': self.micro_timeframe,
            'regime_detection_window': self.regime_detection_window
        }
    
    def get_regime_config(self) -> Dict[str, Any]:
        """Get regime-specific configuration."""
        return {
            'n_regimes': self.n_regimes,
            'min_regime_duration': self.min_regime_duration,
            'max_regime_duration': self.max_regime_duration,
            'regime_stability_threshold': self.regime_stability_threshold,
            'data_driven_regimes': self.data_driven_regimes
        }
    
    def get_search_config(self) -> Dict[str, Any]:
        """Get search-specific configuration."""
        return {
            'max_search_iterations': self.max_search_iterations,
            'max_search_time_seconds': self.max_search_time_seconds,
            'population_size': self.population_size,
            'generations': self.generations,
            'early_stopping_patience': self.early_stopping_patience,
            'min_improvement_threshold': self.min_improvement_threshold,
            'search_strategy': self.search_strategy.value
        }
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation-specific configuration."""
        return {
            'validation_method': self.validation_method.value,
            'validation_split': self.validation_split,
            'cv_folds': self.cv_folds,
            'test_size': self.test_size
        }
    
    def get_financial_config(self) -> Dict[str, Any]:
        """Get financial and trading-specific configuration."""
        return {
            'economic_significance_threshold': self.economic_significance_threshold,
            'trading_viability_threshold': self.trading_viability_threshold,
            'max_drawdown_threshold': self.max_drawdown_threshold,
            'risk_adjusted_return_threshold': self.risk_adjusted_return_threshold,
            'transaction_cost_penalty': self.transaction_cost_penalty,
            'slippage_assumption': self.slippage_assumption,
            'min_position_size': self.min_position_size,
            'max_position_size': self.max_position_size
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance and resource configuration."""
        return {
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'enable_parallel_processing': self.enable_parallel_processing,
            'max_memory_usage_gb': self.max_memory_usage_gb,
            'enable_memory_optimization': self.enable_memory_optimization,
            'enable_gpu_acceleration': self.enable_gpu_acceleration,
            'batch_size': self.batch_size
        }
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"UnifiedArchitectureConfig(architecture_type={self.architecture_type.value}, " \
               f"optimization_mode={self.optimization_mode.value}, " \
               f"search_strategy={self.search_strategy.value}, " \
               f"n_regimes={self.n_regimes})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"UnifiedArchitectureConfig(\n" \
               f"  architecture_type={self.architecture_type.value},\n" \
               f"  optimization_mode={self.optimization_mode.value},\n" \
               f"  search_strategy={self.search_strategy.value},\n" \
               f"  timeframes=({self.primary_timeframe}, {self.secondary_timeframe}, {self.micro_timeframe}),\n" \
               f"  n_regimes={self.n_regimes},\n" \
               f"  population_size={self.population_size},\n" \
               f"  generations={self.generations},\n" \
               f"  output_dir='{self.output_dir}'\n" \
               f")"


# Configuration presets for common use cases
def create_quick_config() -> UnifiedArchitectureConfig:
    """Create a quick configuration for fast prototyping."""
    return UnifiedArchitectureConfig(
        max_search_iterations=20,
        max_search_time_seconds=300,
        population_size=20,
        generations=50,
        n_regimes=4,
        enable_detailed_logging=False,
        enable_uncertainty_quantification=False,
        enable_robustness_testing=False
    )


def create_comprehensive_config() -> UnifiedArchitectureConfig:
    """Create a comprehensive configuration for thorough analysis."""
    return UnifiedArchitectureConfig(
        max_search_iterations=500,
        max_search_time_seconds=7200,
        population_size=100,
        generations=200,
        n_regimes=12,
        enable_detailed_logging=True,
        enable_uncertainty_quantification=True,
        enable_robustness_testing=True,
        enable_hyperparameter_optimization=True
    )


def create_regime_aware_config() -> UnifiedArchitectureConfig:
    """Create a regime-aware configuration for market regime analysis."""
    return UnifiedArchitectureConfig(
        optimization_mode=OptimizationMode.REGIME_AWARE,
        n_regimes=10,
        regime_stability_threshold=0.8,
        data_driven_regimes=True,
        enable_uncertainty_quantification=True
    )


def create_real_time_config() -> UnifiedArchitectureConfig:
    """Create a real-time configuration for live trading."""
    return UnifiedArchitectureConfig(
        optimization_mode=OptimizationMode.REAL_TIME,
        max_search_iterations=50,
        max_search_time_seconds=600,
        early_stopping_patience=5,
        enable_detailed_logging=False,
        enable_uncertainty_quantification=True
    )