"""
Base Training Configuration

Common configuration patterns shared across all training modules.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BaseTrainingConfig:
    """Base configuration for all training steps with common functionality."""
    
    # Basic configuration
    model_name: str = "base_model"
    timeframe: str = "5m"
    
    # HPO configuration
    enable_hpo: bool = True
    hpo_n_trials: int = 100
    hpo_timeout_seconds: int = 3600
    hpo_cv_folds: int = 5
    
    # Model saving
    save_models: bool = True
    model_save_path: str = "./models"
    save_format: str = "joblib"  # joblib, pickle, h5
    
    # Evaluation configuration
    enable_evaluation: bool = True
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        "mse", "mae", "r2", "mape", "smape"
    ])
    
    # Overfitting prevention
    enable_overfitting_prevention: bool = True
    overfitting_threshold: float = 0.1
    
    # Training configuration
    validation_split: float = 0.2
    test_split: float = 0.1
    enable_cross_validation: bool = True
    cv_folds: int = 5
    
    # Data augmentation
    enable_data_augmentation: bool = True
    augmentation_method: str = "smote"  # smote, adasyn
    augmentation_ratio: float = 1.0
    
    # Regime configuration
    min_samples_per_regime: int = 1000
    enable_regime_merging: bool = True
    regime_merge_threshold: int = 500


@dataclass
class PerRegimeTrainingConfig(BaseTrainingConfig):
    """Configuration for per-regime training steps."""
    
    # Model types to train
    model_types: List[str] = field(default_factory=lambda: [
        "TCN", "CatBoostRegressor", "LGBMRegressor", "RandomForestRegressor"
    ])
    
    # Model-specific HPO search spaces
    hpo_search_spaces: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'TCN': {
            'hidden_size': {'type': 'int', 'low': 32, 'high': 128},
            'num_layers': {'type': 'int', 'low': 1, 'high': 4},
            'dropout': {'type': 'float', 'low': 0.1, 'high': 0.5},
            'learning_rate': {'type': 'float', 'low': 0.001, 'high': 0.1, 'log': True}
        },
        'CatBoostRegressor': {
            'n_estimators': {'type': 'int', 'low': 500, 'high': 2000},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.2, 'log': True},
            'depth': {'type': 'int', 'low': 4, 'high': 10},
            'l2_leaf_reg': {'type': 'float', 'low': 1.0, 'high': 10.0}
        },
        'LGBMRegressor': {
            'n_estimators': {'type': 'int', 'low': 500, 'high': 2000},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.2, 'log': True},
            'max_depth': {'type': 'int', 'low': 4, 'high': 10},
            'reg_alpha': {'type': 'float', 'low': 0.0, 'high': 1.0},
            'reg_lambda': {'type': 'float', 'low': 0.0, 'high': 1.0}
        },
        'RandomForestRegressor': {
            'n_estimators': {'type': 'int', 'low': 100, 'high': 1000},
            'max_depth': {'type': 'int', 'low': 5, 'high': 20},
            'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
            'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10}
        }
    })


@dataclass
class EnsembleTrainingConfig(BaseTrainingConfig):
    """Configuration for ensemble training steps."""

    # Ensemble configuration
    base_models: List[str] = field(default_factory=lambda: [
        "TCN", "CatBoostRegressor", "LGBMRegressor", "RandomForestRegressor"
    ])
    meta_model: str = "Ridge"

    # Intensity configuration (for scaling training parameters)
    intensity_percentage: float = 1.0
    
    # Meta model HPO search space
    meta_model_hpo_space: Dict[str, Any] = field(default_factory=lambda: {
        'alpha': {'type': 'float', 'low': 0.1, 'high': 10.0, 'log': True},
        'solver': {'type': 'categorical', 'choices': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
    })


@dataclass
class TacticianTrainingConfig(BaseTrainingConfig):
    """Configuration for Tactician training steps."""
    
    # Model types to train
    model_types: List[str] = field(default_factory=lambda: [
        "NODE", "CatBoostRegressor", "LGBMRegressor", "Ridge"
    ])
    
    # Analyst integration
    analyst_model_path: str = "./models/analyst_ensemble"
    analyst_output_names: List[str] = field(default_factory=lambda: [
        "signal_strength", "confidence", "risk_score", "regime_label"
    ])
    analyst_threshold: float = 0.6
    
    # Single model training (not per-regime)
    use_single_model: bool = True
    single_model_name: str = "tactician_unified_model"
    
    # Model-specific HPO search spaces
    hpo_search_spaces: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'NODE': {
            'n_d': {'type': 'int', 'low': 32, 'high': 128},
            'n_a': {'type': 'int', 'low': 32, 'high': 128},
            'n_steps': {'type': 'int', 'low': 3, 'high': 8},
            'gamma': {'type': 'float', 'low': 1.0, 'high': 2.0},
            'lambda_sparse': {'type': 'float', 'low': 1e-4, 'high': 1e-2, 'log': True}
        },
        'CatBoostRegressor': {
            'n_estimators': {'type': 'int', 'low': 500, 'high': 2000},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.2, 'log': True},
            'depth': {'type': 'int', 'low': 4, 'high': 10},
            'l2_leaf_reg': {'type': 'float', 'low': 1.0, 'high': 10.0}
        },
        'LGBMRegressor': {
            'n_estimators': {'type': 'int', 'low': 500, 'high': 2000},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.2, 'log': True},
            'max_depth': {'type': 'int', 'low': 4, 'high': 10},
            'reg_alpha': {'type': 'float', 'low': 0.0, 'high': 1.0},
            'reg_lambda': {'type': 'float', 'low': 0.0, 'high': 1.0}
        },
        'Ridge': {
            'alpha': {'type': 'float', 'low': 0.1, 'high': 10.0, 'log': True},
            'solver': {'type': 'categorical', 'choices': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
        }
    })


@dataclass
class HMMTrainingConfig(BaseTrainingConfig):
    """Configuration for HMM training steps."""

    # HMM specific configuration
    n_features: int = 100
    sequence_length: int = 20
    n_regimes: int = 3
    intensity_percentage: float = 1.0
    training_mode_config: Optional[Dict[str, Any]] = None
    model_training: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    optimization: Optional[Dict[str, Any]] = None
    
    # Model types
    model_types: List[str] = field(default_factory=lambda: [
        "logistic_regression", "lightgbm", "tcn"
    ])
    
    # HPO configuration
    hpo_trials: int = 100
    enable_multi_objective: bool = True
    objectives: List[str] = field(default_factory=lambda: [
        "accuracy", "f1_score", "regime_stability"
    ])
    objective_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])