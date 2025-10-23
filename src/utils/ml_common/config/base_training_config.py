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

    # Enhanced training utilities
    enable_enhanced_training: bool = True
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001

    # Lookahead bias detection
    enable_lookahead_bias_detection: bool = True
    lookahead_bias_strict_mode: bool = True

    # Enhanced regularization
    enable_enhanced_regularization: bool = True
    l1_alpha: float = 0.01
    l2_alpha: float = 0.01
    dropout_rate: float = 0.2
    max_depth: Optional[int] = None
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    max_features: str = 'sqrt'  # 'sqrt', 'log2', None, or float

    # Temporal validation
    enable_temporal_validation: bool = True
    enable_purged_cv: bool = True
    cv_purge_pct: float = 0.01  # 1% of data purged between train/test
    cv_gap: int = 0  # Additional gap between train/test

    # Walk-forward validation
    enable_walk_forward_validation: bool = False
    wfv_initial_train_size: int = 1000
    wfv_test_size: int = 100
    wfv_step_size: int = 50
    wfv_expanding_window: bool = True

    # Ensemble diversity monitoring
    enable_ensemble_diversity: bool = False
    diversity_threshold: float = 0.1

    # Universal validation settings (from main branch)
    enable_validation: bool = True
    enable_overfitting_detection: bool = True
    enable_timeframe_validation: bool = True
    validation_failure_threshold: float = 0.5
    fail_on_validation_error: bool = False
    warn_on_validation_issues: bool = True
    save_validation_reports: bool = True
    validation_report_directory: str = "reports/validation"
    enable_validation_logging: bool = True

    # Training configuration
    validation_split: float = 0.2
    test_split: float = 0.1
    enable_cross_validation: bool = True
    cv_folds: int = 5

    # Data augmentation
    enable_data_augmentation: bool = True
    augmentation_method: str = "smote"  # smote, adasyn
    augmentation_ratio: float = 1.0



@dataclass
class EnsembleTrainingConfig(BaseTrainingConfig):
    """Configuration for ensemble training steps."""

    # Ensemble configuration
    base_models: List[str] = field(default_factory=lambda: [
        "NAS", "CatBoostClassifier", "XGBoostClassifier", "LGBMDARTClassifier"
    ])
    meta_models: List[str] = field(default_factory=lambda: [
        "XGBoostClassifier", "CatBoostClassifier", "NAS"
    ])
    meta_model: str = "XGBoostClassifier"  # Default meta model for backward compatibility

    # Enable meta model comparison
    compare_meta_models: bool = True

    # Intensity configuration (for scaling training parameters)
    intensity_percentage: float = 1.0

    # Meta model HPO search spaces for different models
    meta_model_hpo_spaces: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'XGBoostClassifier': {
            'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
            'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0}
        },
        'CatBoostClassifier': {
            'iterations': {'type': 'int', 'low': 50, 'high': 300},
            'depth': {'type': 'int', 'low': 4, 'high': 10},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3},
            'l2_leaf_reg': {'type': 'float', 'low': 1.0, 'high': 10.0}
        },
        'LightGBMClassifier': {
            'n_estimators': {'type': 'int', 'low': 200, 'high': 600},
            'max_depth': {'type': 'int', 'low': 3, 'high': 5},
            'learning_rate': {'type': 'float', 'low': 0.03, 'high': 0.05, 'log': True},
            'num_leaves': {'type': 'int', 'low': 15, 'high': 31},
            'min_data_in_leaf': {'type': 'int', 'low': 50, 'high': 150},
            'feature_fraction': {'type': 'float', 'low': 0.6, 'high': 0.9},
            'bagging_fraction': {'type': 'float', 'low': 0.7, 'high': 0.9},
            'lambda_l1': {'type': 'float', 'low': 0.0, 'high': 0.1},
            'lambda_l2': {'type': 'float', 'low': 0.0, 'high': 0.1}
        },
        'NAS': {
            'learning_rate': {'type': 'float', 'low': 0.001, 'high': 0.1, 'log': True},
            'num_epochs': {'type': 'int', 'low': 10, 'high': 100},
            'hidden_size': {'type': 'int', 'low': 32, 'high': 256},
            'dropout': {'type': 'float', 'low': 0.1, 'high': 0.5}
        }
    })

    # Legacy HPO space for backward compatibility
    meta_model_hpo_space: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': {'type': 'int', 'low': 50, 'high': 300},
        'max_depth': {'type': 'int', 'low': 3, 'high': 10},
        'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3}
    })

@dataclass
class TacticianTrainingConfig(BaseTrainingConfig):
    """Configuration for Tactician training steps."""

    # Model types to train
    model_types: List[str] = field(default_factory=lambda: [
        "XGBoost_custom", "RandomForest", "CatBoostRegressor", "ElasticNet", "RandomSurvivalForest"
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

    # Ensemble training (always enabled for Tactician)
    enable_ensemble_training: bool = True  # Always True for Tactician
    ensemble_method: str = "stacking"  # stacking, voting, blending
    meta_model: str = "LightGBM"  # Use LightGBM as meta-learner
    ensemble_name: str = "tactician_ensemble"

    # Entry timing optimization (focus on optimal entry within 0-0.5% range)
    enable_entry_timing_optimization: bool = True
    entry_timing_objectives: Dict[str, str] = field(default_factory=lambda: {
        'early_entry_penalty': 'min',           # Minimize entering too early
        'late_entry_penalty': 'min',            # Minimize entering too late
        'optimal_entry_reward': 'max',          # Maximize entering at optimal timing
        'entry_timing_efficiency': 'max',       # Maximize profit from optimal entry timing
        'directional_consistency': 'min',       # Minimize directional inconsistency
        'confidence_score': 'max'               # Maximize confidence in optimal timing
    })
    entry_timing_range: float = 0.005  # 0-0.5% range for entry timing optimization
    expected_movement: float = 0.01  # Expected 1% movement in the right direction

    # Model-specific HPO search spaces
    hpo_search_spaces: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'XGBoost_custom': {
            'n_estimators': {'type': 'int', 'low': 500, 'high': 2000},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.2, 'log': True},
            'max_depth': {'type': 'int', 'low': 4, 'high': 8},
            'subsample': {'type': 'float', 'low': 0.7, 'high': 1.0},
            'colsample_bytree': {'type': 'float', 'low': 0.7, 'high': 1.0},
            'reg_alpha': {'type': 'float', 'low': 0.0, 'high': 1.0},
            'reg_lambda': {'type': 'float', 'low': 0.0, 'high': 1.0},
            'min_child_weight': {'type': 'int', 'low': 1, 'high': 7},
            'gamma': {'type': 'float', 'low': 0.0, 'high': 0.3}
        },
        'RandomForest': {
            'n_estimators': {'type': 'int', 'low': 100, 'high': 1000},
            'max_depth': {'type': 'int', 'low': 5, 'high': 20},
            'min_samples_split': {'type': 'int', 'low': 2, 'high': 10},
            'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 4},
            'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2']},
            'bootstrap': {'type': 'categorical', 'choices': [True, False]}
        },
        'CatBoostRegressor': {
            'n_estimators': {'type': 'int', 'low': 500, 'high': 2000},
            'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.2, 'log': True},
            'depth': {'type': 'int', 'low': 4, 'high': 10},
            'l2_leaf_reg': {'type': 'float', 'low': 1.0, 'high': 10.0}
        },
        'ElasticNet': {
            'alpha': {'type': 'float', 'low': 0.001, 'high': 10.0, 'log': True},
            'l1_ratio': {'type': 'float', 'low': 0.1, 'high': 1.0},
            'max_iter': {'type': 'int', 'low': 1000, 'high': 5000}
        },
        'RandomSurvivalForest': {
            'n_estimators': {'type': 'int', 'low': 100, 'high': 1000},
            'max_depth': {'type': 'int', 'low': 5, 'high': 20},
            'min_samples_split': {'type': 'int', 'low': 2, 'high': 10},
            'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 4},
            'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2']},
            'bootstrap': {'type': 'categorical', 'choices': [True, False]},
            'max_samples': {'type': 'float', 'low': 0.5, 'high': 1.0}
        }
    })

@dataclass
class RegimeMetaModelTrainingConfig(BaseTrainingConfig):
    """Configuration for regime meta-model training with enhanced meta-features."""

    # Meta-model configuration
    meta_model_types: List[str] = field(default_factory=lambda: [
        "LightGBMClassifier", "XGBoostClassifier", "CatBoostClassifier"
    ])

    # Meta-features configuration
    enable_meta_features: bool = True
    meta_feature_types: Dict[str, bool] = field(default_factory=lambda: {
        # Disagreement & uncertainty
        'margin': True,
        'entropy': True,
        'gini_impurity': True,
        'pairwise_variance': True,
        'disagreement_rate': True,
        'js_divergence_spread': True,
        # Temporal dynamics
        'probability_slope': True,
        'momentum_confidence': True,
        'flip_pressure': True,
        'duration_prior': True,
        # Calibration & reliability
        'brier_components': True,
        'temperature_proxy': True,
        # Diversity & specialist detection
        'specialist_gating_cues': True,
        'cohens_kappa': True,
        'diversity_metrics': True
    })

    # Meta-feature parameters
    meta_feature_params: Dict[str, Any] = field(default_factory=lambda: {
        'temporal_window': 5,  # Short windows: 3-8 bars
        'momentum_half_life': 5,  # EWMA half-life
        'flip_pressure_window': 8,  # Rolling count window
        'brier_validation_window': 200,  # Shadow validation stream
        'temperature_optimization_window': 100,  # Rolling window for temperature
        'diversity_window': 50,  # Rolling window for diversity metrics
        'max_meta_features': 10  # Use 2-5 of the most important
    })

    # LightGBM meta-model specific configuration
    lightgbm_meta_config: Dict[str, Any] = field(default_factory=lambda: {
        'objective': 'multiclass',
        'num_leaves': [15, 23, 31],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.03, 0.04, 0.05],
        'min_data_in_leaf': [50, 100, 150],
        'feature_fraction': [0.6, 0.75, 0.9],
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': [0, 1e-2, 1e-1],
        'lambda_l2': [0, 1e-2, 1e-1],
        'n_estimators': [200, 400, 600],
        'boosting': 'gbdt',
        'metric': 'multi_logloss',
        # Stable sweet spot
        'stable_sweet_spot': {
            'num_leaves': 23,
            'max_depth': 4,
            'learning_rate': 0.04,
            'min_data_in_leaf': 100,
            'n_estimators': 400
        }
    })

    # Advanced meta-model features
    advanced_features: Dict[str, Any] = field(default_factory=lambda: {
        'enable_uncertainty_quantification': True,
        'uncertainty_methods': ['ensemble_variance', 'monte_carlo_dropout', 'bayesian_approximation'],
        'enable_regime_transitions': True,
        'transition_window': 5,
        'transition_smoothing': True,
        'enable_adaptive_learning': True,
        'adaptation_rate': 0.01,
        'adaptation_window': 100,
        'enable_dynamic_model_selection': True,
        'selection_criteria': ['accuracy', 'stability', 'diversity'],
        'enable_calibration': True,
        'calibration_method': 'platt_scaling',  # platt_scaling, isotonic_regression
        'calibration_window': 200
    })

    # HPO configuration for meta-models
    meta_model_hpo_spaces: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'LightGBMClassifier': {
            'n_estimators': {'type': 'int', 'low': 200, 'high': 600},
            'max_depth': {'type': 'int', 'low': 3, 'high': 5},
            'learning_rate': {'type': 'float', 'low': 0.03, 'high': 0.05, 'log': True},
            'num_leaves': {'type': 'int', 'low': 15, 'high': 31},
            'min_data_in_leaf': {'type': 'int', 'low': 50, 'high': 150},
            'feature_fraction': {'type': 'float', 'low': 0.6, 'high': 0.9},
            'bagging_fraction': {'type': 'float', 'low': 0.7, 'high': 0.9},
            'lambda_l1': {'type': 'float', 'low': 0.0, 'high': 0.1},
            'lambda_l2': {'type': 'float', 'low': 0.0, 'high': 0.1}
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
        "logistic_regression", "lightgbm", "random_forest"
    ])

    # HPO configuration
    hpo_trials: int = 100
    enable_multi_objective: bool = True
    objectives: List[str] = field(default_factory=lambda: [
        "accuracy", "f1_score", "regime_stability"
    ])
    objective_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.2])  # Reduced regime stability weight for 15m short-term predictions
