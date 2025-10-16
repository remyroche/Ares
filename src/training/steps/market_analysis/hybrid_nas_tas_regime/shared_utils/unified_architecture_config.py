"""
Unified Architecture Configuration System

This module provides a unified configuration system that consolidates configuration
management for both TAS and NAS architectures, reducing duplication and ensuring
consistency across the hybrid regime detection system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import logging
from abc import ABC, abstractmethod
import json
from pathlib import Path
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)

class ArchitectureType(Enum):
    """Types of architectures supported."""
    TAS = "tas"
    NAS = "nas"
    HYBRID = "hybrid"
    TREE_ONLY = "tree_only"
    NEURAL_ONLY = "neural_only"

class SearchStrategy(Enum):
    """Unified search strategies."""
    RANDOM = "random"
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    META_LEARNING = "meta_learning"
    HYBRID = "hybrid"
    GRID = "grid"

class OptimizationObjective(Enum):
    """Unified optimization objectives."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFITABILITY = "profitability"
    ROBUSTNESS = "robustness"
    EFFICIENCY = "efficiency"
    ECONOMIC_SIGNIFICANCE = "economic_significance"
    TRADING_VIABILITY = "trading_viability"
    COMPUTATIONAL_EFFICIENCY = "computational_efficiency"
    ARCHITECTURE_COMPLEXITY = "architecture_complexity"

class MarketRegime(Enum):
    """Unified market regime types."""
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"
    CRISIS = "crisis"
    NORMAL = "normal"
    UNKNOWN = "unknown"

@dataclass
class BaseArchitectureConfig(ABC):
    """Base configuration class for all architectures."""

    # System identification
    architecture_type: ArchitectureType
    system_name: str
    version: str = "1.0.0"

    # Core search settings
    search_strategy: SearchStrategy = SearchStrategy.BAYESIAN
    max_iterations: int = 100
    max_time_seconds: int = 3600
    early_stopping_patience: int = 10
    min_improvement_threshold: float = 0.001

    # Multi-objective optimization
    enable_multi_objective: bool = True
    primary_objective: OptimizationObjective = OptimizationObjective.ACCURACY
    secondary_objectives: List[OptimizationObjective] = field(default_factory=lambda: [
        OptimizationObjective.ROBUSTNESS,
        OptimizationObjective.EFFICIENCY
    ])
    objective_weights: Dict[OptimizationObjective, float] = field(default_factory=lambda: {
        OptimizationObjective.ACCURACY: 0.6,
        OptimizationObjective.ROBUSTNESS: 0.2,
        OptimizationObjective.EFFICIENCY: 0.2
    })

    # Performance thresholds
    accuracy_threshold: float = 0.9
    economic_significance_threshold: float = 0.7
    trading_viability_threshold: float = 0.6
    regime_stability_threshold: float = 0.8
    max_drawdown_threshold: float = 0.15
    risk_adjusted_return_threshold: float = 0.1

    # Validation settings
    validation_method: str = "holdout"
    validation_split: float = 0.2
    cv_folds: int = 5
    time_series_gap: int = 0

    # Performance settings
    n_jobs: int = -1
    random_state: int = 42
    verbose: bool = True

    # Output settings
    save_results: bool = True
    save_models: bool = True
    output_dir: str = "architecture_results"

    # Timeframe configuration
    primary_timeframe: str = "15m"
    micro_timeframe: str = "5m"
    regime_detection_window: int = 100

    # Regime configuration
    n_regimes: int = 10
    min_regime_duration: int = 15
    max_regime_duration: int = 180
    data_driven_regimes: bool = True

    # Hardware optimization
    enable_hardware_optimization: bool = True
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    max_memory_usage_gb: float = 8.0
    batch_size: int = 1000

    # Advanced features
    enable_uncertainty_estimation: bool = True
    enable_regime_analysis: bool = True
    enable_real_time_adaptation: bool = True
    enable_continual_learning: bool = True
    enable_meta_learning: bool = True

    # ML Common integration
    enable_lookahead_protection: bool = True
    enable_overfitting_prevention: bool = True
    enable_cross_validation: bool = True
    enable_threshold_optimization: bool = True

    def __post_init__(self):
        """Validate and setup configuration after initialization."""
        self._validate_config()
        self._setup_logging()

    def _validate_config(self):
        """Validate configuration parameters."""
        try:
            # Validate objective weights sum to 1.0
            total_weight = sum(self.objective_weights.values())
            if abs(total_weight - 1.0) > 1e-6:
                logger.warning(f"Objective weights sum to {total_weight}, normalizing to 1.0")
                for obj in self.objective_weights:
                    self.objective_weights[obj] /= total_weight

            # Validate thresholds
            thresholds = [
                ('accuracy_threshold', self.accuracy_threshold),
                ('economic_significance_threshold', self.economic_significance_threshold),
                ('trading_viability_threshold', self.trading_viability_threshold),
                ('regime_stability_threshold', self.regime_stability_threshold)
            ]

            for name, value in thresholds:
                if not (0.0 <= value <= 1.0):
                    raise ValueError(f"Invalid {name}: {value}")

            # Validate timeframes
            if self.min_regime_duration >= self.max_regime_duration:
                raise ValueError("Minimum regime duration must be less than maximum")

            logger.info("✅ Configuration validation passed")

        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            raise

    def _setup_logging(self):
        """Setup logging configuration."""
        if self.verbose:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

    @abstractmethod
    def get_architecture_specific_config(self) -> Dict[str, Any]:
        """Get architecture-specific configuration."""
        pass

    def get_unified_config(self) -> Dict[str, Any]:
        """Get unified configuration that can be used by both TAS and NAS."""
        return {
            'architecture_type': self.architecture_type.value,
            'system_name': self.system_name,
            'version': self.version,
            'search_strategy': self.search_strategy.value,
            'max_iterations': self.max_iterations,
            'max_time_seconds': self.max_time_seconds,
            'early_stopping_patience': self.early_stopping_patience,
            'min_improvement_threshold': self.min_improvement_threshold,
            'enable_multi_objective': self.enable_multi_objective,
            'primary_objective': self.primary_objective.value,
            'secondary_objectives': [obj.value for obj in self.secondary_objectives],
            'objective_weights': {obj.value: weight for obj, weight in self.objective_weights.items()},
            'performance_thresholds': {
                'accuracy': self.accuracy_threshold,
                'economic_significance': self.economic_significance_threshold,
                'trading_viability': self.trading_viability_threshold,
                'regime_stability': self.regime_stability_threshold,
                'max_drawdown': self.max_drawdown_threshold,
                'risk_adjusted_return': self.risk_adjusted_return_threshold
            },
            'validation_settings': {
                'method': self.validation_method,
                'split': self.validation_split,
                'cv_folds': self.cv_folds,
                'time_series_gap': self.time_series_gap
            },
            'performance_settings': {
                'n_jobs': self.n_jobs,
                'random_state': self.random_state,
                'verbose': self.verbose
            },
            'output_settings': {
                'save_results': self.save_results,
                'save_models': self.save_models,
                'output_dir': self.output_dir
            },
            'timeframe_settings': {
                'primary_timeframe': self.primary_timeframe,
                'micro_timeframe': self.micro_timeframe,
                'regime_detection_window': self.regime_detection_window
            },
            'regime_settings': {
                'n_regimes': self.n_regimes,
                'min_regime_duration': self.min_regime_duration,
                'max_regime_duration': self.max_regime_duration,
                'data_driven_regimes': self.data_driven_regimes
            },
            'hardware_settings': {
                'enable_hardware_optimization': self.enable_hardware_optimization,
                'enable_gpu_acceleration': self.enable_gpu_acceleration,
                'enable_memory_optimization': self.enable_memory_optimization,
                'max_memory_usage_gb': self.max_memory_usage_gb,
                'batch_size': self.batch_size
            },
            'advanced_features': {
                'enable_uncertainty_estimation': self.enable_uncertainty_estimation,
                'enable_regime_analysis': self.enable_regime_analysis,
                'enable_real_time_adaptation': self.enable_real_time_adaptation,
                'enable_continual_learning': self.enable_continual_learning,
                'enable_meta_learning': self.enable_meta_learning
            },
            'ml_common_integration': {
                'enable_lookahead_protection': self.enable_lookahead_protection,
                'enable_overfitting_prevention': self.enable_overfitting_prevention,
                'enable_cross_validation': self.enable_cross_validation,
                'enable_threshold_optimization': self.enable_threshold_optimization
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config = self.get_unified_config()
        config.update(self.get_architecture_specific_config())
        return config

    def save_config(self, filepath: str):
        """Save configuration to file."""
        try:
            output_path = Path(filepath)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)

            logger.info(f"✅ Configuration saved to {filepath}")

        except Exception as e:
            logger.error(f"❌ Failed to save configuration: {e}")
            raise

    @classmethod
    def load_config(cls, filepath: str) -> 'BaseArchitectureConfig':
        """Load configuration from file."""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)

            # Determine architecture type and create appropriate config
            arch_type = ArchitectureType(config_dict.get('architecture_type', 'hybrid'))

            if arch_type == ArchitectureType.TAS:
                return TASArchitectureConfig.from_dict(config_dict)
            elif arch_type == ArchitectureType.NAS:
                return NASArchitectureConfig.from_dict(config_dict)
            else:
                return HybridArchitectureConfig.from_dict(config_dict)

        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            raise

@dataclass
class TASArchitectureConfig(BaseArchitectureConfig):
    """TAS-specific configuration extending base configuration."""

    # TAS-specific settings
    architecture_type: ArchitectureType = ArchitectureType.TAS
    system_name: str = "Tree Architecture Search"

    # Tree-specific parameters
    min_trees: int = 10
    max_trees: int = 1000
    min_depth: int = 1
    max_depth: int = 20
    min_samples_split: int = 2
    max_samples_split: int = 1000
    min_samples_leaf: int = 1
    max_samples_leaf: int = 100

    # Feature selection
    min_features: int = 1
    max_features: Union[int, float, str] = "auto"
    feature_selection_methods: List[str] = field(default_factory=lambda: [
        "auto", "sqrt", "log2", "none"
    ])

    # Tree model types
    model_types: List[str] = field(default_factory=lambda: [
        "RandomForest", "XGBoost", "LightGBM", "ExtraTrees",
        "GradientBoosting", "AdaBoost", "NGBoost", "DART"
    ])

    # CVLSA-specific parameters
    enable_cvlSA_architecture: bool = True
    cvlSA_cascade_depth: int = 3
    cvlSA_variable_selection_methods: List[str] = field(default_factory=lambda: [
        'variance_threshold', 'mutual_information', 'tree_importance',
        'correlation_filter', 'recursive_elimination'
    ])

    # Micro-regime detection
    enable_micro_regime_detection: bool = True
    micro_regime_sensitivity: float = 0.7
    micro_regime_detection_threshold: float = 0.6

    def get_architecture_specific_config(self) -> Dict[str, Any]:
        """Get TAS-specific configuration."""
        return {
            'tas_settings': {
                'min_trees': self.min_trees,
                'max_trees': self.max_trees,
                'min_depth': self.min_depth,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'max_samples_split': self.max_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'max_samples_leaf': self.max_samples_leaf,
                'min_features': self.min_features,
                'max_features': self.max_features,
                'feature_selection_methods': self.feature_selection_methods,
                'model_types': self.model_types
            },
            'cvlSA_settings': {
                'enable_cvlSA_architecture': self.enable_cvlSA_architecture,
                'cascade_depth': self.cvlSA_cascade_depth,
                'variable_selection_methods': self.cvlSA_variable_selection_methods
            },
            'micro_regime_settings': {
                'enable_micro_regime_detection': self.enable_micro_regime_detection,
                'sensitivity': self.micro_regime_sensitivity,
                'detection_threshold': self.micro_regime_detection_threshold
            }
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TASArchitectureConfig':
        """Create TAS config from dictionary."""
        # Extract base config
        base_config = {k: v for k, v in config_dict.items()
                      if k not in ['tas_settings', 'cvlSA_settings', 'micro_regime_settings']}

        # Convert enum strings back to enums
        if 'architecture_type' in base_config:
            base_config['architecture_type'] = ArchitectureType(base_config['architecture_type'])
        if 'search_strategy' in base_config:
            base_config['search_strategy'] = SearchStrategy(base_config['search_strategy'])
        if 'primary_objective' in base_config:
            base_config['primary_objective'] = OptimizationObjective(base_config['primary_objective'])
        if 'secondary_objectives' in base_config:
            base_config['secondary_objectives'] = [OptimizationObjective(obj) for obj in base_config['secondary_objectives']]

        # Extract TAS-specific settings
        tas_settings = config_dict.get('tas_settings', {})
        cvlSA_settings = config_dict.get('cvlSA_settings', {})
        micro_regime_settings = config_dict.get('micro_regime_settings', {})

        # Create config
        config = cls(**base_config)

        # Set TAS-specific attributes
        for key, value in tas_settings.items():
            if hasattr(config, key):
                setattr(config, key, value)

        for key, value in cvlSA_settings.items():
            attr_name = f"cvlSA_{key}"
            if hasattr(config, attr_name):
                setattr(config, attr_name, value)

        for key, value in micro_regime_settings.items():
            attr_name = f"micro_regime_{key}"
            if hasattr(config, attr_name):
                setattr(config, attr_name, value)

        return config

@dataclass
class NASArchitectureConfig(BaseArchitectureConfig):
    """NAS-specific configuration extending base configuration."""

    # NAS-specific settings
    architecture_type: ArchitectureType = ArchitectureType.NAS
    system_name: str = "Neural Architecture Search"

    # Neural architecture parameters
    min_layers: int = 2
    max_layers: int = 20
    min_hidden_size: int = 32
    max_hidden_size: int = 1024
    min_parameters: int = 1000
    max_parameters: int = 10000000

    # Neural architecture types
    architecture_types: List[str] = field(default_factory=lambda: [
        "NeuralODE", "VisionTransformer", "StateSpaceModel", "LSTM",
        "GRU", "Transformer", "CNN", "MLP"
    ])

    # Neural-specific parameters
    enable_neural_odes: bool = True
    enable_vision_transformers: bool = True
    enable_state_space_models: bool = True

    # Meta-learning specific
    meta_learning_rate: float = 1e-3
    inner_learning_rate: float = 0.01
    num_inner_steps: int = 5
    num_outer_steps: int = 100
    num_shots: int = 5
    num_ways: int = 5

    def get_architecture_specific_config(self) -> Dict[str, Any]:
        """Get NAS-specific configuration."""
        return {
            'nas_settings': {
                'min_layers': self.min_layers,
                'max_layers': self.max_layers,
                'min_hidden_size': self.min_hidden_size,
                'max_hidden_size': self.max_hidden_size,
                'min_parameters': self.min_parameters,
                'max_parameters': self.max_parameters,
                'architecture_types': self.architecture_types
            },
            'neural_components': {
                'enable_neural_odes': self.enable_neural_odes,
                'enable_vision_transformers': self.enable_vision_transformers,
                'enable_state_space_models': self.enable_state_space_models
            },
            'meta_learning_settings': {
                'meta_learning_rate': self.meta_learning_rate,
                'inner_learning_rate': self.inner_learning_rate,
                'num_inner_steps': self.num_inner_steps,
                'num_outer_steps': self.num_outer_steps,
                'num_shots': self.num_shots,
                'num_ways': self.num_ways
            }
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'NASArchitectureConfig':
        """Create NAS config from dictionary."""
        # Extract base config
        base_config = {k: v for k, v in config_dict.items()
                      if k not in ['nas_settings', 'neural_components', 'meta_learning_settings']}

        # Convert enum strings back to enums
        if 'architecture_type' in base_config:
            base_config['architecture_type'] = ArchitectureType(base_config['architecture_type'])
        if 'search_strategy' in base_config:
            base_config['search_strategy'] = SearchStrategy(base_config['search_strategy'])
        if 'primary_objective' in base_config:
            base_config['primary_objective'] = OptimizationObjective(base_config['primary_objective'])
        if 'secondary_objectives' in base_config:
            base_config['secondary_objectives'] = [OptimizationObjective(obj) for obj in base_config['secondary_objectives']]

        # Extract NAS-specific settings
        nas_settings = config_dict.get('nas_settings', {})
        neural_components = config_dict.get('neural_components', {})
        meta_learning_settings = config_dict.get('meta_learning_settings', {})

        # Create config
        config = cls(**base_config)

        # Set NAS-specific attributes
        for key, value in nas_settings.items():
            if hasattr(config, key):
                setattr(config, key, value)

        for key, value in neural_components.items():
            if hasattr(config, key):
                setattr(config, key, value)

        for key, value in meta_learning_settings.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

@dataclass
class HybridArchitectureConfig(BaseArchitectureConfig):
    """Hybrid TAS-NAS configuration extending base configuration."""

    # Hybrid-specific settings
    architecture_type: ArchitectureType = ArchitectureType.HYBRID
    system_name: str = "Hybrid TAS-NAS Architecture"

    # Hybrid weights
    tas_weight: float = 0.5
    nas_weight: float = 0.5
    adaptive_weighting: bool = True

    # Integration settings
    enable_tas_integration: bool = True
    enable_nas_integration: bool = True
    integration_method: str = "ensemble"  # "ensemble", "cascade", "parallel"

    # Combined parameters
    enable_combined_search: bool = True
    combined_search_strategy: SearchStrategy = SearchStrategy.HYBRID

    # Ensemble settings
    ensemble_method: str = "weighted_voting"  # "voting", "stacking", "weighted_voting"
    ensemble_optimization: bool = True

    def get_architecture_specific_config(self) -> Dict[str, Any]:
        """Get hybrid-specific configuration."""
        return {
            'hybrid_settings': {
                'tas_weight': self.tas_weight,
                'nas_weight': self.nas_weight,
                'adaptive_weighting': self.adaptive_weighting,
                'enable_tas_integration': self.enable_tas_integration,
                'enable_nas_integration': self.enable_nas_integration,
                'integration_method': self.integration_method
            },
            'combined_search_settings': {
                'enable_combined_search': self.enable_combined_search,
                'combined_search_strategy': self.combined_search_strategy.value
            },
            'ensemble_settings': {
                'ensemble_method': self.ensemble_method,
                'ensemble_optimization': self.ensemble_optimization
            }
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HybridArchitectureConfig':
        """Create hybrid config from dictionary."""
        # Extract base config
        base_config = {k: v for k, v in config_dict.items()
                      if k not in ['hybrid_settings', 'combined_search_settings', 'ensemble_settings']}

        # Convert enum strings back to enums
        if 'architecture_type' in base_config:
            base_config['architecture_type'] = ArchitectureType(base_config['architecture_type'])
        if 'search_strategy' in base_config:
            base_config['search_strategy'] = SearchStrategy(base_config['search_strategy'])
        if 'primary_objective' in base_config:
            base_config['primary_objective'] = OptimizationObjective(base_config['primary_objective'])
        if 'secondary_objectives' in base_config:
            base_config['secondary_objectives'] = [OptimizationObjective(obj) for obj in base_config['secondary_objectives']]

        # Extract hybrid-specific settings
        hybrid_settings = config_dict.get('hybrid_settings', {})
        combined_search_settings = config_dict.get('combined_search_settings', {})
        ensemble_settings = config_dict.get('ensemble_settings', {})

        # Create config
        config = cls(**base_config)

        # Set hybrid-specific attributes
        for key, value in hybrid_settings.items():
            if hasattr(config, key):
                setattr(config, key, value)

        for key, value in combined_search_settings.items():
            if key == 'combined_search_strategy':
                config.combined_search_strategy = SearchStrategy(value)
            elif hasattr(config, key):
                setattr(config, key, value)

        for key, value in ensemble_settings.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

# Convenience functions for creating configurations
def create_tas_config(**kwargs) -> TASArchitectureConfig:
    """Create TAS configuration with default values."""
    return TASArchitectureConfig(**kwargs)

def create_nas_config(**kwargs) -> NASArchitectureConfig:
    """Create NAS configuration with default values."""
    return NASArchitectureConfig(**kwargs)

def create_hybrid_config(**kwargs) -> HybridArchitectureConfig:
    """Create hybrid configuration with default values."""
    return HybridArchitectureConfig(**kwargs)

def create_quick_config(architecture_type: ArchitectureType = ArchitectureType.HYBRID) -> BaseArchitectureConfig:
    """Create quick configuration for testing."""
    config_params = {
        'max_iterations': 20,
        'max_time_seconds': 300,
        'search_strategy': SearchStrategy.RANDOM,
        'early_stopping_patience': 5,
        'enable_multi_objective': False,
        'enable_hardware_optimization': False,
        'enable_meta_learning': False,
        'verbose': True
    }

    if architecture_type == ArchitectureType.TAS:
        return TASArchitectureConfig(**config_params)
    elif architecture_type == ArchitectureType.NAS:
        return NASArchitectureConfig(**config_params)
    else:
        return HybridArchitectureConfig(**config_params)

def create_comprehensive_config(architecture_type: ArchitectureType = ArchitectureType.HYBRID) -> BaseArchitectureConfig:
    """Create comprehensive configuration for production."""
    config_params = {
        'max_iterations': 500,
        'max_time_seconds': 7200,
        'search_strategy': SearchStrategy.HYBRID,
        'enable_multi_objective': True,
        'enable_hardware_optimization': True,
        'enable_meta_learning': True,
        'enable_uncertainty_estimation': True,
        'enable_regime_analysis': True,
        'enable_real_time_adaptation': True,
        'enable_continual_learning': True,
        'verbose': False
    }

    if architecture_type == ArchitectureType.TAS:
        return TASArchitectureConfig(**config_params)
    elif architecture_type == ArchitectureType.NAS:
        return NASArchitectureConfig(**config_params)
    else:
        return HybridArchitectureConfig(**config_params)
