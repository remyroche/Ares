"""
Centralized Configuration Management for Model Training Pipeline

This module provides:
- Centralized model configuration management
- Training mode parameter scaling
- Dynamic model type registration
- Configuration validation and optimization
- Environment-specific configuration loading
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import yaml
from pathlib import Path

from src.utils.logger import system_logger
from src.utils.tprint import tprint_error, tprint_warning, tprint_success, tprint_info

logger = system_logger.getChild('ConfigManager')

class ModelType(Enum):
    """Supported model types for training."""
    # Analyst Models (5m timeframe)
    TCN = "tcn"  # Temporal Convolutional Network
    CATBOOST = "catboost"  # CatBoost Regressor
    LIGHTGBM = "lightgbm"  # LightGBM Regressor
    ELASTIC_NET = "elastic_net"  # Elastic Net (meta-learner)

    # Tactician Models (1m timeframe)
    NODE = "node"  # Neural Oblivious Decision Ensembles
    XGBOOST = "xgboost"  # XGBoost Regressor
    RANDOM_FOREST = "random_forest"  # Random Forest Regressor
    RIDGE = "ridge"  # Ridge Regression

    # Fallback Models
    LINEAR_REGRESSION = "linear_regression"
    DECISION_TREE = "decision_tree"

@dataclass
class ModelConfig:
    """Configuration for a specific model type."""
    name: str
    model_type: ModelType
    description: str
    default_params: Dict[str, Any]
    hpo_params: Dict[str, Any]
    required_features: List[str] = field(default_factory=list)
    optional_features: List[str] = field(default_factory=list)
    memory_requirement_mb: int = 1024
    training_time_estimate_minutes: int = 30
    fallback_model: Optional[ModelType] = None

    def get_total_features(self) -> int:
        """Get total number of features this model can handle."""
        return len(self.required_features) + len(self.optional_features)

@dataclass
class TrainingModeConfig:
    """Configuration for a training mode."""
    name: str
    description: str
    intensity_percentage: float
    lookback_days: int
    model_configs: Dict[str, Dict[str, Any]]
    validation_configs: Dict[str, Any]
    optimization_configs: Dict[str, Any]
    resource_limits: Dict[str, Any]

    def scale_parameters(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Scale parameters based on intensity percentage."""
        scaled_params = base_params.copy()
        intensity = self.intensity_percentage

        # Scale trial counts
        if 'n_trials' in scaled_params:
            scaled_params['n_trials'] = max(1, int(scaled_params['n_trials'] * intensity))
        if 'max_trials' in scaled_params:
            scaled_params['max_trials'] = max(1, int(scaled_params['max_trials'] * intensity))

        # Scale epochs
        if 'epochs' in scaled_params:
            scaled_params['epochs'] = max(1, int(scaled_params['epochs'] * intensity))

        # Scale validation parameters
        if 'monte_carlo_samples' in scaled_params:
            scaled_params['monte_carlo_samples'] = max(100, int(scaled_params['monte_carlo_samples'] * intensity))
        if 'ab_test_rounds' in scaled_params:
            scaled_params['ab_test_rounds'] = max(1, int(scaled_params['ab_test_rounds'] * intensity))

        return scaled_params

class ConfigurationManager:
    """Centralized configuration management for training pipeline."""

    def __init__(self):
        self.model_configs: Dict[ModelType, ModelConfig] = {}
        self.training_modes: Dict[str, TrainingModeConfig] = {}
        self._register_default_configurations()

    def _register_default_configurations(self):
        """Register all default model and training configurations."""

        # Analyst Models Configuration
        self._register_model_config(ModelConfig(
            name="TCN Analyst",
            model_type=ModelType.TCN,
            description="Temporal Convolutional Network for analyst predictions",
            default_params={
                'kernel_size': 3,
                'filters': 64,
                'dilation_base': 2,
                'dropout': 0.1,
                'batch_size': 32,
                'epochs': 50
            },
            hpo_params={
                'kernel_size': [2, 3, 5],
                'filters': [32, 64, 128],
                'dilation_base': [2, 3, 4],
                'dropout': [0.1, 0.2, 0.3],
                'batch_size': [16, 32, 64]
            },
            required_features=['close', 'volume', 'rsi', 'macd'],
            optional_features=['bb_upper', 'bb_lower', 'ema_20', 'ema_50', 'sma_20', 'sma_50'],
            memory_requirement_mb=2048,
            training_time_estimate_minutes=45,
            fallback_model=ModelType.LIGHTGBM
        ))

        self._register_model_config(ModelConfig(
            name="CatBoost Analyst",
            model_type=ModelType.CATBOOST,
            description="CatBoost for analyst predictions with categorical features",
            default_params={
                'iterations': 1000,
                'learning_rate': 0.1,
                'depth': 6,
                'l2_leaf_reg': 3,
                'verbose': False
            },
            hpo_params={
                'iterations': [500, 1000, 1500],
                'learning_rate': [0.01, 0.1, 0.3],
                'depth': [4, 6, 8],
                'l2_leaf_reg': [1, 3, 5]
            },
            required_features=['close', 'volume', 'rsi'],
            optional_features=['macd', 'bb_upper', 'bb_lower', 'ema_20', 'sma_20'],
            memory_requirement_mb=512,
            training_time_estimate_minutes=20,
            fallback_model=ModelType.RANDOM_FOREST
        ))

        self._register_model_config(ModelConfig(
            name="LightGBM Analyst",
            model_type=ModelType.LIGHTGBM,
            description="LightGBM for fast analyst predictions",
            default_params={
                'n_estimators': 1000,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_child_samples': 20,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1
            },
            hpo_params={
                'n_estimators': [500, 1000, 1500],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [4, 6, 8],
                'min_child_samples': [10, 20, 50],
                'reg_alpha': [0.0, 0.1, 0.3],
                'reg_lambda': [0.0, 0.1, 0.3]
            },
            required_features=['close', 'volume'],
            optional_features=['rsi', 'macd', 'bb_upper', 'bb_lower', 'ema_20', 'sma_20'],
            memory_requirement_mb=256,
            training_time_estimate_minutes=15
        ))

        # Tactician Models Configuration
        self._register_model_config(ModelConfig(
            name="NODE Tactician",
            model_type=ModelType.NODE,
            description="Neural Oblivious Decision Ensembles for tactician timing",
            default_params={
                'n_estimators': 1000,
                'max_depth': 6,
                'min_child_samples': 20,
                'learning_rate': 0.1,
                'verbose': 0
            },
            hpo_params={
                'n_estimators': [500, 1000, 1500],
                'max_depth': [4, 6, 8],
                'min_child_samples': [10, 20, 50],
                'learning_rate': [0.01, 0.1, 0.3]
            },
            required_features=['analyst_signal', 'close', 'volume', 'rsi'],
            optional_features=['macd', 'bb_upper', 'bb_lower', 'price_change_1m', 'volume_ratio'],
            memory_requirement_mb=1024,
            training_time_estimate_minutes=25,
            fallback_model=ModelType.XGBOOST
        ))

        self._register_model_config(ModelConfig(
            name="XGBoost Tactician",
            model_type=ModelType.XGBOOST,
            description="XGBoost for tactician timing decisions",
            default_params={
                'n_estimators': 1000,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0
            },
            hpo_params={
                'n_estimators': [500, 1000, 1500],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'reg_alpha': [0.0, 0.1, 0.3],
                'reg_lambda': [0.5, 1.0, 2.0]
            },
            required_features=['analyst_signal', 'close', 'volume'],
            optional_features=['rsi', 'macd', 'bb_upper', 'bb_lower', 'price_change_1m'],
            memory_requirement_mb=512,
            training_time_estimate_minutes=20,
            fallback_model=ModelType.RANDOM_FOREST
        ))

        # Register training modes
        self._register_training_modes()

    def _register_model_config(self, config: ModelConfig):
        """Register a model configuration."""
        self.model_configs[config.model_type] = config

    def _register_training_modes(self):
        """Register training mode configurations."""

        # Full production mode
        self.training_modes['full'] = TrainingModeConfig(
            name="full",
            description="Production mode - Complete training with full dataset",
            intensity_percentage=1.0,
            lookback_days=730,
            model_configs={
                'analyst_base_models': {
                    'tcn': {'enabled': True, 'priority': 1},
                    'catboost': {'enabled': True, 'priority': 2},
                    'lightgbm': {'enabled': True, 'priority': 3}
                },
                'analyst_ensemble': {
                    'base_models': ['tcn', 'catboost', 'lightgbm'],
                    'meta_learner': 'elastic_net'
                },
                'tactician_base_models': {
                    'node': {'enabled': True, 'priority': 1},
                    'xgboost': {'enabled': True, 'priority': 2},
                    'catboost': {'enabled': True, 'priority': 3}
                },
                'tactician_ensemble': {
                    'base_models': ['node', 'xgboost', 'catboost'],
                    'meta_learner': 'lightgbm'
                }
            },
            validation_configs={
                'monte_carlo_samples': 10000,
                'ab_test_rounds': 10,
                'cross_validation_folds': 5
            },
            optimization_configs={
                'optuna_trials': 200,
                'optuna_timeout': 3600,
                'enable_parallel_optimization': True
            },
            resource_limits={
                'max_memory_gb': 16,
                'max_cpu_percent': 90,
                'timeout_minutes': 240
            }
        )

        # Light development mode
        self.training_modes['light'] = TrainingModeConfig(
            name="light",
            description="Development mode - Minimal data for quick iterations",
            intensity_percentage=0.025,
            lookback_days=730,  # Keep 730 for HMM regime discovery
            model_configs={
                'analyst_base_models': {
                    'lightgbm': {'enabled': True, 'priority': 1}
                },
                'analyst_ensemble': {
                    'base_models': ['lightgbm'],
                    'meta_learner': 'elastic_net'
                },
                'tactician_base_models': {
                    'xgboost': {'enabled': True, 'priority': 1}
                },
                'tactician_ensemble': {
                    'base_models': ['xgboost'],
                    'meta_learner': 'lightgbm'
                }
            },
            validation_configs={
                'monte_carlo_samples': 100,
                'ab_test_rounds': 2,
                'cross_validation_folds': 3
            },
            optimization_configs={
                'optuna_trials': 10,
                'optuna_timeout': 300,
                'enable_parallel_optimization': False
            },
            resource_limits={
                'max_memory_gb': 4,
                'max_cpu_percent': 50,
                'timeout_minutes': 30
            }
        )

        # Blank testing mode
        self.training_modes['blank'] = TrainingModeConfig(
            name="blank",
            description="Testing mode - All features with reduced data",
            intensity_percentage=0.1,
            lookback_days=180,
            model_configs={
                'analyst_base_models': {
                    'lightgbm': {'enabled': True, 'priority': 1},
                    'catboost': {'enabled': True, 'priority': 2}
                },
                'analyst_ensemble': {
                    'base_models': ['lightgbm', 'catboost'],
                    'meta_learner': 'elastic_net'
                },
                'tactician_base_models': {
                    'xgboost': {'enabled': True, 'priority': 1},
                    'catboost': {'enabled': True, 'priority': 2}
                },
                'tactician_ensemble': {
                    'base_models': ['xgboost', 'catboost'],
                    'meta_learner': 'lightgbm'
                }
            },
            validation_configs={
                'monte_carlo_samples': 1000,
                'ab_test_rounds': 5,
                'cross_validation_folds': 3
            },
            optimization_configs={
                'optuna_trials': 50,
                'optuna_timeout': 900,
                'enable_parallel_optimization': False
            },
            resource_limits={
                'max_memory_gb': 8,
                'max_cpu_percent': 70,
                'timeout_minutes': 60
            }
        )

    def get_model_config(self, model_type: ModelType) -> ModelConfig:
        """Get configuration for a specific model type."""
        if model_type not in self.model_configs:
            raise ValueError(f"Model type {model_type.value} not found in configuration")
        return self.model_configs[model_type]

    def get_training_mode_config(self, mode: str) -> TrainingModeConfig:
        """Get configuration for a training mode."""
        if mode not in self.training_modes:
            raise ValueError(f"Training mode {mode} not found in configuration")
        return self.training_modes[mode]

    def get_available_models_for_role(self, role: str) -> List[ModelConfig]:
        """Get all available models for a specific role (analyst/tactician)."""
        if role == 'analyst':
            return [config for config in self.model_configs.values()
                   if config.model_type in [ModelType.TCN, ModelType.CATBOOST, ModelType.LIGHTGBM, ModelType.ELASTIC_NET]]
        elif role == 'tactician':
            return [config for config in self.model_configs.values()
                   if config.model_type in [ModelType.NODE, ModelType.XGBOOST, ModelType.RANDOM_FOREST, ModelType.RIDGE]]
        else:
            raise ValueError(f"Unknown role: {role}")

    def get_models_by_priority(self, role: str, mode: str = 'full') -> List[ModelConfig]:
        """Get models for a role sorted by priority for the given mode."""
        mode_config = self.get_training_mode_config(mode)
        role_configs = mode_config.model_configs.get(f'{role}_base_models', {})

        models = []
        for model_name, config in role_configs.items():
            if config.get('enabled', False):
                # Find model config by name (simple mapping for now)
                model_type = self._name_to_model_type(model_name)
                if model_type:
                    models.append((config.get('priority', 999), self.model_configs[model_type]))

        return [model for priority, model in sorted(models)]

    def _name_to_model_type(self, name: str) -> Optional[ModelType]:
        """Convert model name to ModelType enum."""
        name_mapping = {
            'tcn': ModelType.TCN,
            'catboost': ModelType.CATBOOST,
            'lightgbm': ModelType.LIGHTGBM,
            'elastic_net': ModelType.ELASTIC_NET,
            'node': ModelType.NODE,
            'xgboost': ModelType.XGBOOST,
            'random_forest': ModelType.RANDOM_FOREST,
            'ridge': ModelType.RIDGE
        }
        return name_mapping.get(name.lower())

    def validate_configuration(self, role: str, mode: str) -> List[str]:
        """Validate configuration for a role and mode combination."""
        warnings = []

        try:
            mode_config = self.get_training_mode_config(mode)
            role_configs = mode_config.model_configs.get(f'{role}_base_models', {})

            # Check if at least one model is enabled
            enabled_models = [name for name, config in role_configs.items() if config.get('enabled', False)]
            if not enabled_models:
                warnings.append(f"No models enabled for {role} in {mode} mode")

            # Check resource requirements
            total_memory = sum(self.model_configs[self._name_to_model_type(name)].memory_requirement_mb
                             for name in enabled_models if self._name_to_model_type(name))

            if total_memory > mode_config.resource_limits['max_memory_gb'] * 1024:
                warnings.append(f"Total memory requirement ({total_memory}MB) exceeds limit ({mode_config.resource_limits['max_memory_gb']}GB)")

        except Exception as e:
            warnings.append(f"Configuration validation error: {e}")

        return warnings

# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None

def get_config_manager() -> ConfigurationManager:
    """Get or create the global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager()
    return _config_manager

def get_model_config(model_type: ModelType) -> ModelConfig:
    """Convenience function to get model configuration."""
    return get_config_manager().get_model_config(model_type)

def get_training_mode_config(mode: str) -> TrainingModeConfig:
    """Convenience function to get training mode configuration."""
    return get_config_manager().get_training_mode_config(mode)

def validate_training_setup(role: str, mode: str) -> bool:
    """Validate training setup and report any issues."""
    manager = get_config_manager()
    warnings = manager.validate_configuration(role, mode)

    if warnings:
        tprint_warning(f"⚠️ Configuration warnings for {role} {mode}:")
        for warning in warnings:
            tprint_warning(f"   - {warning}")
        return False
    else:
        tprint_success(f"✅ Configuration validated for {role} {mode}")
        return True