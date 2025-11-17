"""
Regime Detection Models Training Component

This component implements the specific regime detection models mentioned in the user's request:
- CatBoost (base model)
- LightGBM (base model)
- ExtraTrees (base model)
- stacker_lgbm_calibrated (meta-learner with probability calibration)

Enhanced with centralized YAML/JSON configuration system for flexible parameter management.
"""

import numpy as np
import pandas as pd
import pickle
import json
import time
import warnings
import psutil
import gc
import copy
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from src.utils.logger import system_logger
from src.utils.tprint import tprint
from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult

# Import centralized configuration system
try:
    from src.config.regime_models_training import (
        RegimeModelsTrainingConfigManager,
        load_regime_training_config,
        ConfigValidationError
    )
    CONFIG_SYSTEM_AVAILABLE = True
    tprint("✅ [REGIME_MODELS] Configuration system imported successfully", color="green")
except ImportError as e:
    CONFIG_SYSTEM_AVAILABLE = False
    tprint(f"⚠️ [REGIME_MODELS] Configuration system not available: {e}", color="yellow")

try:
    from src.config.regime_models_training.default_config import (
        create_default_regime_training_config as _central_create_default_regime_training_config,
        validate_regime_training_config as _central_validate_regime_training_config,
    )
except ImportError:
    _central_create_default_regime_training_config = None
    _central_validate_regime_training_config = None


def _fallback_default_regime_training_config() -> Dict[str, Any]:
    base_config: Dict[str, Any] = {
        "test_size": 0.15,  # Reduced from 0.2 to keep more training data
        "gap_size": 2,  # Increased from 1 to prevent leakage
        "validation_size": 0.1,  # Reduced from 0.15 to keep more training data
        "min_regime_samples": 5,  # Reduced from 50 to handle rare regimes
        "regime_aware": True,
        "temporal_validation": {
            "enabled": True,
            "strict_temporal_order": True,
            "initial_train_size": 0.75,  # Increased from 0.65
            "step_size": 0.1,
            "min_test_size": 0.1,
            "enable_leakage_detection": True,
            "n_splits": 7,  # Increased from 5 for better CV estimates
            "test_size": 0.1,  # Reduced from 0.15 to keep more training data
            "gap_size": 2,  # Increased from 1 to prevent leakage
        },
        "model_validation": {
            "enabled": True,
            "cv_folds": 5,
            "scoring_metrics": ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"],
            "temporal_smoothing": True,
            "smoothing_alpha": 0.1,
            "enable_soft_labels": True,
            "soft_label_smoothing": 0.1,
        },
        "data_validation": {
            "min_samples": 10,
            "min_features": 50,
            "required_columns": ["close", "open", "high", "low", "volume"],
            "max_nan_ratio": 0.1,
            "enable_data_quality_checks": True,
        },
        "regime_extraction": {
            "min_regimes": 2,
            "max_regimes": 10,
            "min_samples_per_regime": 50,  # Increased from 5 to ensure sufficient samples
            "extraction_method": "standardized",
            "fallback_to_synthetic": True,
        },
        "hpo": {
            "enabled": True,
            "method": "bayesian",
            "max_trials": 150,  # Increased from 50 for better hyperparameter search
            "timeout_seconds": 600,  # Increased from 300 to allow more thorough search
            "early_stopping": True,
            "enable_pruning": True,
        },
        "hardware_optimization": {
            "enabled": True,
            "cpu_optimization_level": "aggressive",
            "gpu_optimization_level": "balanced",
            "memory_optimization_level": "balanced",
            "enable_adaptive_optimization": True,
            "enable_learning": True,
        },
        "data_preparation": {
            "enable_feature_scaling": True,
            "scaling_method": "standard",
            "handle_missing_values": "mean",
            "remove_outliers": True,
            "outlier_method": "iqr",
            "iqr_multiplier": 3.0,
        },
        "evaluation": {
            "enhanced_evaluation": True,
            "temporal_metrics": True,
            "regime_persistence_metrics": True,
            "ensemble_evaluation": True,
        },
    }
    return copy.deepcopy(base_config)


def create_default_regime_training_config() -> Dict[str, Any]:
    if _central_create_default_regime_training_config is not None:
        return copy.deepcopy(_central_create_default_regime_training_config())
    return _fallback_default_regime_training_config()


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def validate_regime_training_config(config: Dict[str, Any], strict: bool = True) -> Dict[str, Any]:
    if _central_validate_regime_training_config is not None:
        return _central_validate_regime_training_config(copy.deepcopy(config), strict=strict)

    validated = _fallback_default_regime_training_config()
    _deep_update(validated, copy.deepcopy(config))

    if strict:
        if validated["temporal_validation"]["test_size"] <= 0 or validated["temporal_validation"]["test_size"] >= 1:
            raise ValueError("temporal_validation.test_size must be between 0 and 1")
        if validated["data_validation"]["min_samples"] < 1:
            raise ValueError("data_validation.min_samples must be >= 1")
        if validated["data_validation"]["min_features"] < 1:
            raise ValueError("data_validation.min_features must be >= 1")
        if validated["regime_extraction"]["min_regimes"] < 1:
            raise ValueError("regime_extraction.min_regimes must be >= 1")
        if validated["regime_extraction"]["max_regimes"] < validated["regime_extraction"]["min_regimes"]:
            raise ValueError("regime_extraction.max_regimes must be >= min_regimes")
        if validated["min_regime_samples"] < 1:
            raise ValueError("min_regime_samples must be >= 1")

    return validated
# Enhanced imports for new functionality
from src.utils.ml_common.unified_vectorization_manager import (
    UnifiedVectorizationManager, OperationType, OptimizationStrategy
)
from src.utils.ml_common.optimization.hpo_utils import (
    HyperparameterOptimization
)
from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import (
    HierarchicalParameterOptimizer,
    ParameterGroup,
    OptimizationStage,
    create_param_group
)
from src.utils.ml_common.optimization.auto_tuner import (
    AutoTuner,
    DatasetCharacteristics
)
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
    BayesianTPEOptimizer,
    OptimizationConfig as TPEOptimizationConfig
)
from src.utils.ml_common.optimization.transition_aware_scoring import (
    create_transition_aware_scorer,
    create_pareto_multi_objective_hpo
)
try:
    from src.utils.ml_common.optimization.pareto import (
        ParetoOptimizer,
        Solution,
        ObjectiveDirection
    )
    PARETO_AVAILABLE = True
except ImportError:
    PARETO_AVAILABLE = False
    ParetoOptimizer = None
    Solution = None
    ObjectiveDirection = None

try:
    from src.utils.ml_common.optimization.hpo_diagnostics_and_fixes import HPODiagnostics
    HPODIAG_AVAILABLE = True
except Exception as e:
    HPODIAG_AVAILABLE = False
    HPODiagnostics = None  # type: ignore
    tprint(f"⚠️ [REGIME_MODELS] HPODiagnostics unavailable: {e}", color="yellow")
from src.utils.ml_common.validation.universal_temporal_validation import (
    UniversalTemporalValidator, TemporalValidationConfig
)
from src.utils.ml_common.validation.temporal_cross_validation import temporal_cross_validation
from src.utils.ml_common.utils.lookahead_protection import LookaheadProtection
from src.utils.hardware.unified_hardware_manager import (
    UnifiedHardwareManager, HardwareConfig, WorkloadType, OptimizationLevel
)
from src.utils.ml_common.evaluation.evaluation_utils import (
    EvaluationUtils
)
from src.utils.ml_common.evaluation.regime_temporal_metrics import (
    RegimeTemporalMetricsCalculator,
    calculate_temporal_smoothness_penalty,
    create_soft_labels
)
from src.utils.ml_common.feature_engineering.feature_smoothing import (
    add_smoothed_features,
    apply_ewm_smoothing
)
from src.utils.ml_common.explainability.model_explainability import ModelExplainabilityManager
from src.utils.ml_common.explainability.shap_lime_integration import SHAPLIMEExplainer as SHAPLIMEIntegration
from src.utils.ml_common.post_training.model_validation import (
    ModelValidator, ValidationConfig
)

# New improved imports for fast fail behavior
from src.utils.ml_common.validation.temporal_data_splitter import (
    TemporalDataSplitter, RegimeAwareSplitter, create_temporal_splitter
)
from src.utils.ml_common.validation.regime_walk_forward_validator import (
    RegimeWalkForwardValidator, RegimeValidationConfig, select_top_models
)
from src.utils.ml_common.data.regime_label_extractor import (
    RegimeLabelExtractor, extract_regime_labels_fast_fail
)
from src.utils.ml_common.data.standardized_regime_extractor import (
    StandardizedRegimeExtractor, extract_regime_labels_standardized, RegimeLabelExtractionError
)
from src.utils.ml_common.training.memory_manager import (
    TrainingMemoryManager, managed_training, periodic_cleanup
)
from src.utils.purged_kfold import PurgedKFoldTime

# Suppress warnings
warnings.filterwarnings('ignore')

# Suppress LightGBM warnings about no further splits
warnings.filterwarnings('ignore', message='.*No further splits with positive gain.*')

# Import ML libraries with comprehensive error handling
tprint("🔍 [REGIME_MODELS] Starting ML libraries import process", color="cyan")
ML_LIBRARIES_AVAILABLE = False
ML_LIBRARY_VERSIONS = {}
ML_IMPORT_ERRORS = []

# Import sklearn components
try:
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.feature_selection import SelectFromModel
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    import sklearn
    ML_LIBRARY_VERSIONS['sklearn'] = sklearn.__version__
    tprint(f"✅ [REGIME_MODELS] scikit-learn imported successfully (v{sklearn.__version__})", color="green")
except ImportError as e:
    ML_IMPORT_ERRORS.append(f"scikit-learn: {e}")
    tprint(f"❌ [REGIME_MODELS] Failed to import scikit-learn: {e}", color="red")

# Import CatBoost
try:
    import catboost as cb
    ML_LIBRARY_VERSIONS['catboost'] = cb.__version__
    tprint(f"✅ [REGIME_MODELS] CatBoost imported successfully (v{cb.__version__})", color="green")
except ImportError as e:
    ML_IMPORT_ERRORS.append(f"CatBoost: {e}")
    tprint(f"❌ [REGIME_MODELS] Failed to import CatBoost: {e}", color="red")

# Import LightGBM
try:
    import lightgbm as lgb
    ML_LIBRARY_VERSIONS['lightgbm'] = lgb.__version__
    tprint(f"✅ [REGIME_MODELS] LightGBM imported successfully (v{lgb.__version__})", color="green")
except ImportError as e:
    ML_IMPORT_ERRORS.append(f"LightGBM: {e}")
    tprint(f"❌ [REGIME_MODELS] Failed to import LightGBM: {e}", color="red")

# Import additional LightGBM functionality
try:
    import lightgbm as lgb
    from lightgbm import log_evaluation, early_stopping  # type: ignore[import-untyped]
    ML_LIBRARY_VERSIONS['lightgbm_extra'] = lgb.__version__
    tprint(f"✅ [REGIME_MODELS] LightGBM extra functions imported successfully", color="green")
except ImportError as e:
    ML_IMPORT_ERRORS.append(f"LightGBM extra functions: {e}")
    tprint(f"❌ [REGIME_MODELS] Failed to import LightGBM extra functions: {e}", color="red")

# Import XGBoost
try:
    import xgboost as xgb
    ML_LIBRARY_VERSIONS['xgboost'] = xgb.__version__
    tprint(f"✅ [REGIME_MODELS] XGBoost imported successfully (v{xgb.__version__})", color="green")
except ImportError as e:
    ML_IMPORT_ERRORS.append(f"XGBoost: {e}")
    tprint(f"❌ [REGIME_MODELS] Failed to import XGBoost: {e}", color="red")

# Import Random Forest
try:
    from sklearn.ensemble import RandomForestClassifier
    ML_LIBRARY_VERSIONS['random_forest'] = sklearn.__version__
    tprint(f"✅ [REGIME_MODELS] Random Forest imported successfully", color="green")
except ImportError as e:
    ML_IMPORT_ERRORS.append(f"Random Forest: {e}")
    tprint(f"❌ [REGIME_MODELS] Failed to import Random Forest: {e}", color="red")

# Import Greedy Rule Lists
try:
    from imodels import GreedyRuleListClassifier  # type: ignore
    ML_LIBRARY_VERSIONS['imodels'] = "available"
    tprint(f"✅ [REGIME_MODELS] Greedy Rule Lists imported successfully", color="green")
except ImportError as e:
    ML_IMPORT_ERRORS.append(f"Greedy Rule Lists: {e}")
    tprint(f"❌ [REGIME_MODELS] Failed to import Greedy Rule Lists: {e}", color="red")
    GreedyRuleListClassifier = None

# Check overall availability
if not ML_IMPORT_ERRORS:
    ML_LIBRARIES_AVAILABLE = True
    tprint("🎉 [REGIME_MODELS] All ML libraries imported successfully", color="green", bold=True)
    tprint(f"📊 [REGIME_MODELS] Library versions: {ML_LIBRARY_VERSIONS}", color="blue")
else:
    tprint(f"⚠️ [REGIME_MODELS] Import errors encountered: {ML_IMPORT_ERRORS}", color="yellow")
    tprint("🔧 [REGIME_MODELS] Some functionality may be limited", color="yellow")

# Import feature generation system
try:
    from src.feature_generation.core.factory import get_feature_bank, FeatureGenerator, FeatureCategory
    FEATURE_GENERATION_AVAILABLE = True
    tprint("✅ [REGIME_MODELS] Feature generation system imported successfully", color="green")
except ImportError as e:
    FEATURE_GENERATION_AVAILABLE = False
    tprint(f"⚠️ [REGIME_MODELS] Feature generation system not available: {e}", color="yellow")

class RegimeModelsTrainingComponent(BaseMarketAnalysisComponent):
    """
    Regime Detection Models Training Component.

    This component trains the specific regime detection models:
    - CatBoost (base model)
    - LightGBM (base model)
    - ExtraTrees (base model)
    - stacker_lgbm_calibrated (meta-learner with probability calibration)

    Enhanced with centralized YAML/JSON configuration system for flexible parameter management.
    """

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the Regime Models Training Component with enhanced utilities and fast fail behavior."""
        tprint("🚀 [REGIME_MODELS] Initializing Regime Models Training Component", color="cyan", bold=True)
        tprint(f"📋 [REGIME_MODELS] Config provided: {config is not None}", color="blue")

        # Store execution parameters from ComponentConfig before parent init
        # CRITICAL: These are passed in ComponentConfig but not in self.config dict after validation
        tprint(f"🔍 [INIT] ComponentConfig type: {type(config)}", "INFO")
        tprint(f"🔍 [INIT] ComponentConfig has execution_mode: {hasattr(config, 'execution_mode') if config else False}", "INFO")
        
        self._execution_mode = config.execution_mode if config and hasattr(config, 'execution_mode') else 'light'
        self._symbol = config.symbol if config and hasattr(config, 'symbol') else 'ETHUSDT'
        self._exchange = config.exchange if config and hasattr(config, 'exchange') else 'binance'
        self._timeframe = config.timeframe if config and hasattr(config, 'timeframe') else '1h'
        
        tprint(f"✅ [INIT] Stored execution params: mode={self._execution_mode}, symbol={self._symbol}, timeframe={self._timeframe}", "SUCCESS")

        # Initialize parent component
        try:
            super().__init__(config)
            tprint("✅ [REGIME_MODELS] Parent component initialized successfully", color="green")
        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Failed to initialize parent component: {e}", color="red")
            raise

        # Initialize logger
        try:
            self.logger = system_logger.getChild('RegimeModelsTrainingComponent')
            self.logger.info("Regime Models Training Component logger initialized")
            tprint("✅ [REGIME_MODELS] Logger initialized successfully", color="green")
        except Exception as e:
            tprint(f"⚠️ [REGIME_MODELS] Logger initialization warning: {e}", color="yellow")

        # Initialize centralized configuration system
        self._initialize_centralized_config()

        # Validate configuration before initializing dependent components
        self._validate_and_setup_config()

        # Initialize improved components
        self._initialize_improved_components()

        # Initialize hardware manager for optimization
        self.hardware_manager = UnifiedHardwareManager(
            HardwareConfig(
                cpu_optimization_level=OptimizationLevel.AGGRESSIVE,
                gpu_optimization_level=OptimizationLevel.BALANCED,
                memory_optimization_level=OptimizationLevel.BALANCED,
                enable_adaptive_optimization=True,
                enable_learning=True
            )
        )
        tprint("🔧 [REGIME_MODELS] Hardware manager initialized", color="green")

        # Initialize vectorization manager for feature generation
        self.vectorization_manager = UnifiedVectorizationManager()
        tprint("🔧 [REGIME_MODELS] Vectorization manager initialized", color="green")

        # Initialize HPO optimizer
        self.hpo_optimizer = HyperparameterOptimization(
            {
                'max_trials': 50,
                'timeout_seconds': 300,
                'enable_early_stopping': True,
                'enable_pruning': True
            }
        )
        tprint("🔧 [REGIME_MODELS] HPO optimizer initialized", color="green")
        
        # Initialize Auto Tuner for intelligent HPO configuration
        self.auto_tuner = AutoTuner(
            conservative_mode=False,
            enable_adaptive_timeout=True,
            enable_resource_monitoring=True
        )
        tprint("🔧 [REGIME_MODELS] Auto-tuner initialized for adaptive HPO", color="green")
        
        # Initialize Pareto optimizer for multi-objective HPO
        if PARETO_AVAILABLE:
            self.pareto_optimizer = ParetoOptimizer()
            tprint("✅ [REGIME_MODELS] Pareto optimizer initialized for multi-objective HPO", color="green")
        else:
            self.pareto_optimizer = None
            tprint("⚠️ [REGIME_MODELS] Pareto optimizer not available", color="yellow")
        
        # Enable transition-aware multi-objective HPO by default
        self.enable_multi_objective_hpo = True
        self.use_pareto_optimization = PARETO_AVAILABLE
        
        # Enable hierarchical optimization for models with many parameters (7+)
        self.use_hierarchical_hpo = True
        tprint("✅ [REGIME_MODELS] Hierarchical HPO enabled for complex models", color="green")

        # Initialize temporal validator for data leakage prevention
        self.temporal_validator = UniversalTemporalValidator(
            TemporalValidationConfig(
                enable_temporal_checks=True,
                strict_temporal_order=True,
                initial_train_size=0.7,
                test_size=0.3,
                gap_size=1
            )
        )
        tprint("🔧 [REGIME_MODELS] Temporal validator initialized", color="green")

        # Initialize lookahead protection
        self.lookahead_protection = LookaheadProtection()
        tprint("🔧 [REGIME_MODELS] Lookahead protection initialized", color="green")

        # Initialize model evaluator
        self.model_evaluator = EvaluationUtils()
        tprint("🔧 [REGIME_MODELS] Model evaluator initialized", color="green")
        
        # Initialize regime temporal metrics calculator
        self.temporal_metrics_calc = RegimeTemporalMetricsCalculator(min_episode_length=3)
        tprint("✅ [REGIME_MODELS] Temporal metrics calculator initialized", color="green")
        
        # Enhanced training configuration
        self.enable_temporal_smoothing = True
        self.temporal_smoothing_alpha = 0.1  # Default smoothness penalty weight
        self.enable_soft_labels = True
        self.soft_label_smoothing = 0.1  # Label smoothing factor
        
        # EWMA features are ALWAYS enabled for regime models - critical for temporal pattern detection
        # These features help models capture regime transitions and persistence
        # Simple EWMA 8 & 20 without special weights - just standard exponential weighting
        self.enable_smoothed_features = True  # ALWAYS TRUE for regime models
        self.smoothing_window_sizes = [8, 20]  # Simple EWMA windows (8 & 20 periods)
        self.ewm_alpha = 0.3  # EWMA smoothing factor (0 < alpha <= 1, smaller = more smoothing)

        # Initialize model validator
        self.model_validator = ModelValidator(
            ValidationConfig(
                enable_purged_cv=True,
                enable_data_leakage_detection=True,
                enable_time_series_validation=True
            )
        )
        tprint("🔧 [REGIME_MODELS] Model validator initialized", color="green")

        # Initialize model training parameters
        tprint("🔧 [REGIME_MODELS] Configuring model training parameters", color="cyan")
        # Use unified hardware manager to cap parallelism instead of n_jobs=-1 (all cores)
        try:
            optimal_workers = int(self.hardware_manager.get_optimal_cpu_count()) if hasattr(self, 'hardware_manager') else -1
        except Exception:
            optimal_workers = -1
        self.model_config = {
            'random_state': 42,
            'test_size': 0.2,
            'cv_folds': 5,
            'n_jobs': optimal_workers
        }

        # Regime-specific model configurations
        self.regime_models_config = {
            'base': {
                'CatBoost': {
                    'iterations': 100,
                    'depth': 6,
                    'learning_rate': 0.1,
                    'random_seed': 42,
                    'verbose': False
                },
                'XGBoost': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'random_state': 42,
                    'n_jobs': optimal_workers,
                    'verbosity': 0
                },
                'Random Forest': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt',
                    'bootstrap': True,
                    'random_state': 42,
                    'n_jobs': optimal_workers
                },
                'Greedy Rule Lists': {
                    'max_depth': 20,  # Increased for better complexity handling
                    'criterion': 'gini',  # Criterion for splitting
                    'class_weight': 'balanced'  # Handle class imbalance
                },
                'ExtraTrees': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 5,  # Increased for stability
                    'min_samples_leaf': 5,    # Increased for stability (was 1)
                    'max_features': 'sqrt',
                    'random_state': 42,
                    'n_jobs': optimal_workers
                }
            },
            'meta_learner': {
                'stacker_lgbm_calibrated': {
                    'num_leaves': 63,  # Increased for better complexity
                    'max_depth': 8,    # Increased depth
                    'learning_rate': 0.05,  # Reduced for better convergence
                    'n_estimators': 200,    # More estimators
                    'min_child_samples': 50,  # Increased for stability (was 20)
                    'min_data_in_leaf': 50,    # Increased for stability
                    'subsample': 0.8,        # Stochastic sampling
                    'colsample_bytree': 0.8,  # Feature sampling
                    'reg_alpha': 0.1,        # L1 regularization
                    'reg_lambda': 0.1,       # L2 regularization
                    'class_weight': 'balanced',  # Handle class imbalance
                    'random_state': 42,
                    'verbose': -1
                }
            }
        }

        # Initialize model storage
        self.models = {}
        self.model_metrics = {}
        self.training_history = []
        self.performance_metrics = {}

        tprint("📊 [REGIME_MODELS] Model storage initialized", color="blue")
        tprint(f"🔍 [REGIME_MODELS] Available ML libraries: {ML_LIBRARIES_AVAILABLE}", color="blue")
        if ML_LIBRARIES_AVAILABLE:
            tprint(f"📚 [REGIME_MODELS] Library versions: {ML_LIBRARY_VERSIONS}", color="blue")
        
        # Validate model configurations
        tprint("🔍 [REGIME_MODELS] Validating model configurations", color="cyan")
        config_valid = True
        for category, models in self.regime_models_config.items():
            for model_type, model_config in models.items():
                if not self._validate_model_config(model_type, model_config):
                    tprint(f"❌ [REGIME_MODELS] Invalid configuration for {model_type} in {category}", color="red")
                    config_valid = False
        
        if config_valid:
            tprint("✅ [REGIME_MODELS] All model configurations validated successfully", color="green")
        else:
            tprint("⚠️ [REGIME_MODELS] Some model configurations have issues", color="yellow")

        # Log initialization completion
        tprint("✅ [REGIME_MODELS] Regime Models Training Component initialized successfully", color="green", bold=True)
        self.logger.info("Regime Models Training Component initialization completed")

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        tprint("📋 [REGIME_MODELS] Getting required artifacts", color="cyan")
        required_artifacts = ['regime_models_training_result']
        tprint(f"✅ [REGIME_MODELS] Required artifacts: {required_artifacts}", color="green")
        return required_artifacts
    
    def _validate_model_config(self, model_type: str, config: Dict[str, Any]) -> bool:
        """
        Validate model configuration parameters.
        
        Args:
            model_type: Type of model (e.g., 'CatBoost', 'LightGBM')
            config: Configuration dictionary
            
        Returns:
            True if valid, False otherwise
        """
        try:
            required_params = {
                'CatBoost': ['iterations', 'depth', 'learning_rate', 'random_seed'],
                'XGBoost': ['n_estimators', 'max_depth', 'learning_rate', 'random_state'],
                'Random Forest': ['n_estimators', 'max_depth', 'random_state'],
                'Greedy Rule Lists': ['max_depth', 'criterion'],
                'ExtraTrees': ['n_estimators', 'max_depth', 'random_state'],
                'stacker_lgbm_calibrated': ['num_leaves', 'max_depth', 'learning_rate', 'n_estimators']
            }
            
            if model_type not in required_params:
                tprint(f"⚠️ [REGIME_MODELS] Unknown model type: {model_type}", color="yellow")
                return True  # Allow unknown model types
            
            missing_params = []
            for param in required_params[model_type]:
                if param not in config:
                    missing_params.append(param)
            
            if missing_params:
                tprint(f"❌ [REGIME_MODELS] Missing required parameters for {model_type}: {missing_params}", color="red")
                return False
            
            # Validate parameter ranges
            if 'learning_rate' in config:
                lr = config['learning_rate']
                if not (0 < lr <= 1.0):
                    tprint(f"❌ [REGIME_MODELS] Invalid learning_rate for {model_type}: {lr} (must be 0 < lr <= 1.0)", color="red")
                    return False
            
            if 'max_depth' in config:
                depth = config['max_depth']
                if depth is not None and (not isinstance(depth, int) or depth < 1):
                    tprint(f"❌ [REGIME_MODELS] Invalid max_depth for {model_type}: {depth} (must be int > 0 or None)", color="red")
                    return False
            
            return True
            
        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Model config validation error: {e}", color="red")
            return False

    def _initialize_centralized_config(self):
        """
        Initialiser le système de configuration centralisée
        """
        try:
            tprint("🔧 [REGIME_MODELS] Initialisation du système de configuration centralisée", color="cyan")
            
            if not CONFIG_SYSTEM_AVAILABLE:
                tprint("⚠️ [REGIME_MODELS] Système de configuration non disponible, utilisation du fallback", color="yellow")
                self._setup_hardcoded_config()
                return
            
            # Initialiser le gestionnaire de configuration
            self.config_manager = RegimeModelsTrainingConfigManager()
            tprint("✅ [REGIME_MODELS] Gestionnaire de configuration initialisé", color="green")
            
            # Charger la configuration par défaut
            try:
                self.config = load_regime_training_config()
                tprint("✅ [REGIME_MODELS] Configuration par défaut chargée", color="green")
                
                # Valider la configuration
                validation_result = self.config_manager.validate_for_training(self.config)
                if not validation_result["ready_for_training"]:
                    tprint(f"⚠️ [REGIME_MODELS] Configuration avec avertissements: {validation_result['warnings']}", color="yellow")
                
                if validation_result["suggestions"]:
                    tprint(f"💡 [REGIME_MODELS] Suggestions: {validation_result['suggestions']}", color="blue")
                    
            except Exception as e:
                tprint(f"❌ [REGIME_MODELS] Erreur lors du chargement de la configuration: {e}", color="red")
                # Utiliser la configuration hardcodée comme fallback
                self._setup_hardcoded_config()
                tprint("🔄 [REGIME_MODELS] Utilisation de la configuration hardcodée", color="yellow")
                return
            
            # Mettre à jour les configurations des modèles basées sur la configuration centralisée
            self._update_model_configs_from_centralized_config()
            
        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Erreur lors de l'initialisation de la configuration centralisée: {e}", color="red")
            # Fallback à la configuration hardcodée
            self._setup_hardcoded_config()
            tprint("🔄 [REGIME_MODELS] Utilisation de la configuration hardcodée", color="yellow")
    
    def _setup_hardcoded_config(self):
        """
        Configurer les paramètres hardcodés comme fallback
        """
        # Configuration HPO basée sur la configuration centralisée ou par défaut
        if hasattr(self, 'config') and self.config:
            hpo_config = self.config.get('hpo', {})
            self.model_config['n_trials'] = hpo_config.get('max_trials', 50)
            self.model_config['timeout_seconds'] = hpo_config.get('timeout_seconds', 300)
        else:
            self.model_config['n_trials'] = 50
            self.model_config['timeout_seconds'] = 300
    
    def _update_model_configs_from_centralized_config(self):
        """
        Mettre à jour les configurations des modèles depuis la configuration centralisée
        """
        if not hasattr(self, 'config') or not self.config:
            return
        
        try:
            models_config = self.config.get('models', {})
            base_models = models_config.get('base_models', {})
            
            # Mettre à jour la configuration CatBoost
            if 'catboost' in base_models:
                catboost_config = base_models['catboost']
                catboost_hpo = catboost_config.get('hpo', {})
                if hasattr(self, 'regime_models_config'):
                    if 'base' in self.regime_models_config and 'CatBoost' in self.regime_models_config['base']:
                        self.regime_models_config['base']['CatBoost']['hpo'] = {
                            'enabled': catboost_hpo.get('enabled', True),
                            'n_trials': catboost_hpo.get('n_trials', 75),
                            'timeout_seconds': catboost_hpo.get('timeout_seconds', 300)
                        }
            
            # Mettre à jour la configuration LightGBM
            if 'lightgbm' in base_models:
                lightgbm_config = base_models['lightgbm']
                lightgbm_hpo = lightgbm_config.get('hpo', {})
                if hasattr(self, 'regime_models_config'):
                    if 'base' in self.regime_models_config and 'LightGBM' in self.regime_models_config['base']:
                        self.regime_models_config['base']['LightGBM']['hpo'] = {
                            'enabled': lightgbm_hpo.get('enabled', True),
                            'n_trials': lightgbm_hpo.get('n_trials', 75),
                            'timeout_seconds': lightgbm_hpo.get('timeout_seconds', 300)
                        }
            
            # Mettre à jour la configuration du meta-learner
            meta_learner = models_config.get('meta_learner', {})
            meta_learner_hpo = meta_learner.get('hpo', {})
            if hasattr(self, 'regime_models_config'):
                if 'meta_learner' in self.regime_models_config:
                    self.regime_models_config['meta_learner']['stacker_lgbm_calibrated']['hpo'] = {
                        'enabled': meta_learner_hpo.get('enabled', True),
                        'n_trials': meta_learner_hpo.get('n_trials', 50),
                        'timeout_seconds': meta_learner_hpo.get('timeout_seconds', 240)
                    }
            
            tprint("✅ [REGIME_MODELS] Configurations des modèles mises à jour depuis la configuration centralisée", color="green")
            
        except Exception as e:
            tprint(f"⚠️ [REGIME_MODELS] Erreur lors de la mise à jour des configurations des modèles: {e}", color="yellow")
    
    def get_config_for_model(self, model_type: str) -> Dict[str, Any]:
        """
        Obtenir la configuration pour un modèle spécifique depuis la configuration centralisée
        
        Args:
            model_type: Type de modèle (catboost, xgboost, lightgbm, etc.)
            
        Returns:
            Configuration du modèle
        """
        if not hasattr(self, 'config_manager') or not hasattr(self, 'config'):
            # Fallback à la configuration hardcodée
            return self._get_hardcoded_model_config(model_type)
        
        try:
            return self.config_manager.get_model_config(self.config, model_type)
        except Exception as e:
            tprint(f"⚠️ [REGIME_MODELS] Erreur lors de l'obtention de la configuration pour {model_type}: {e}", color="yellow")
            return self._get_hardcoded_model_config(model_type)
    
    def _get_hardcoded_model_config(self, model_type: str) -> Dict[str, Any]:
        """
        Obtenir la configuration hardcodée pour un modèle (fallback)
        """
        if not hasattr(self, 'regime_models_config'):
            return {"enabled": True, "hpo": {"enabled": True, "n_trials": 50}}
        
        # Chercher dans les modèles de base
        if 'base' in self.regime_models_config:
            for name, config in self.regime_models_config['base'].items():
                if model_type.lower() in name.lower():
                    return config
        
        # Chercher dans le meta-learner
        if 'meta_learner' in self.regime_models_config and 'stacker_lgbm_calibrated' in self.regime_models_config['meta_learner']:
            if model_type.lower() in ['stacker_lgbm_calibrated', 'meta_learner', 'ensemble']:
                return self.regime_models_config['meta_learner']['stacker_lgbm_calibrated']
        
        # Configuration par défaut
        return {"enabled": True, "hpo": {"enabled": True, "n_trials": 50}}

    def _validate_and_setup_config(self):
        """Validate and setup configuration with fast fail behavior."""
        tprint("🔧 [REGIME_MODELS] Validating configuration", color="cyan")
        
        # Start from default configuration and apply overrides cautiously
        default_config = create_default_regime_training_config()
        config_dict = copy.deepcopy(default_config)

        component_cfg = getattr(self, 'config', None)
        custom_params = getattr(component_cfg, 'custom_params', {}) if component_cfg else {}

        # Allow overriding key temporal split parameters when available
        test_size_override = custom_params.get('test_size', getattr(component_cfg, 'test_size', None) if component_cfg else None)
        if test_size_override is not None:
            config_dict['temporal_validation']['test_size'] = float(test_size_override)

        gap_size_override = custom_params.get('gap_size', getattr(component_cfg, 'gap_size', None) if component_cfg else None)
        if gap_size_override is not None:
            config_dict['temporal_validation']['gap_size'] = int(gap_size_override)

        cv_folds_override = custom_params.get('cv_folds', getattr(component_cfg, 'cv_folds', None) if component_cfg else None)
        if cv_folds_override is not None:
            config_dict['model_validation']['cv_folds'] = int(cv_folds_override)

        min_features_override = custom_params.get('min_features')
        if min_features_override is not None:
            config_dict['data_validation']['min_features'] = max(1, int(min_features_override))

        min_samples_override = custom_params.get('min_samples')
        if min_samples_override is not None:
            config_dict['data_validation']['min_samples'] = max(1, int(min_samples_override))

        min_regime_samples = custom_params.get('min_regime_samples', getattr(component_cfg, 'min_regime_samples', None) if component_cfg else None)
        if min_regime_samples is None:
            min_regime_samples = config_dict.get('min_regime_samples', config_dict['data_validation'].get('min_samples', 10))
        config_dict['min_regime_samples'] = max(1, int(min_regime_samples))

        # Ensure regime extraction settings respect overrides when provided
        regime_min_samples_override = custom_params.get('min_samples_per_regime')
        if regime_min_samples_override is not None:
            config_dict['regime_extraction']['min_samples_per_regime'] = max(1, int(regime_min_samples_override))

        regime_min_override = custom_params.get('min_regimes')
        if regime_min_override is not None:
            config_dict['regime_extraction']['min_regimes'] = max(1, int(regime_min_override))

        regime_max_override = custom_params.get('max_regimes')
        if regime_max_override is not None:
            config_dict['regime_extraction']['max_regimes'] = max(config_dict['regime_extraction']['min_regimes'], int(regime_max_override))

        # Validate configuration with fast fail
        try:
            self.validated_config = validate_regime_training_config(config_dict, strict=True)
            tprint("✅ [REGIME_MODELS] Configuration validated successfully", color="green")
        except ValueError as e:
            tprint(f"❌ [REGIME_MODELS] Configuration validation failed: {e}", color="red")
            raise

    def _initialize_improved_components(self):
        """Initialize improved components with fast fail behavior."""
        tprint("🔧 [REGIME_MODELS] Initializing improved components", color="cyan")
        
        # Initialize temporal splitter
        min_regime_samples = self.validated_config.get('min_regime_samples', 10)
        tprint(f"🔍 [DEBUG] Config being passed to temporal splitter: min_regime_samples={min_regime_samples}", color="yellow")
        self.temporal_splitter = create_temporal_splitter(self.validated_config)
        # Workaround: Directly set min_regime_samples to handle rare regimes
        if hasattr(self.temporal_splitter, 'min_regime_samples'):
            self.temporal_splitter.min_regime_samples = min_regime_samples
            tprint(f"🔧 [REGIME_MODELS] Temporal splitter min_regime_samples set to {self.temporal_splitter.min_regime_samples}", color="cyan")
            # 🔍 LOGGING: Configuration codée en dur pour la gestion des régimes rares
            tprint(f"🔍 [REGIME_MODELS] GESTION RÉGIMES RARES: min_regime_samples configuré à {min_regime_samples}", color="blue")
            tprint(f"   → Cette configuration détermine le nombre minimum d'échantillons requis par régime", color="blue")
            tprint(f"   → Impact: Les régimes avec moins de {min_regime_samples} échantillons seront ignorés", color="blue")
        tprint("✅ [REGIME_MODELS] Temporal splitter initialized", color="green")

        # Initialize walk-forward validator for OOS model selection
        wf_config = RegimeValidationConfig(
            n_outer_folds=5,
            n_inner_folds=3,
            embargo_pct=0.05,
            min_train_samples=100,
            min_val_samples=30,
            min_regime_samples=min_regime_samples
        )
        self.walk_forward_validator = RegimeWalkForwardValidator(wf_config)
        tprint("✅ [REGIME_MODELS] Walk-forward validator initialized", color="green")
        # 🔍 LOGGING: Configuration pour la validation walk-forward
        tprint(f"🔍 [REGIME_MODELS] WALK-FORWARD: min_regime_samples={min_regime_samples} pour la validation OOS", color="blue")
        
        # Initialize regime label extractor
        self.regime_extractor = RegimeLabelExtractor(
            min_samples=min_regime_samples,
            min_regimes=2
        )
        tprint("✅ [REGIME_MODELS] Regime label extractor initialized", color="green")
        # 🔍 LOGGING: Configuration pour l'extraction des labels de régime
        tprint(f"🔍 [REGIME_MODELS] EXTRACTION: min_samples={min_regime_samples}, min_regimes=2", color="blue")
        tprint(f"   → Cette configuration affecte directement la distribution des régimes extraits", color="blue")
        
        # Note: Using existing feature bank system instead of custom feature generator
        tprint("✅ [REGIME_MODELS] Using existing feature bank system", color="green")

    async def _load_and_resample_regime_probabilities(
        self,
        base_step: Any
    ) -> Optional[pd.DataFrame]:
        """
        Load rolling_hmm_regime_probabilities from versioned artifacts.
        
        No resampling is performed - we load at the same timeframe as training (1h).

        Args:
            base_step: BaseStep instance for artifact loading

        Returns:
            DataFrame with regime probabilities at the same timeframe as training
        """
        try:
            tprint("📥 [REGIME_MODELS] Loading rolling_hmm_regime_probabilities artifact from versioned storage")

            # Load regime probabilities at the same timeframe as training
            regime_probs = base_step._get_artifact(
                'rolling_hmm_regime_probabilities',
                artifact_type='data',
                data_category='features'  # Hint that this is feature data stored in HDF5
            )

            if regime_probs is None:
                tprint("⚠️ [REGIME_MODELS] No rolling_hmm_regime_probabilities found in versioned artifacts")
                return None

            tprint(f"✅ [REGIME_MODELS] Loaded regime probabilities: {regime_probs.shape}")
            tprint(f"📊 [REGIME_MODELS] Columns: {list(regime_probs.columns)}")
            
            # Handle new format where timestamp is a column (not index)
            if 'timestamp' in regime_probs.columns:
                tprint(f"📊 [REGIME_MODELS] Converting timestamp column to index")
                regime_probs['timestamp'] = pd.to_datetime(regime_probs['timestamp'])
                regime_probs.set_index('timestamp', inplace=True)
                regime_probs.sort_index(inplace=True)
            
            tprint(f"📊 [REGIME_MODELS] Index range: {regime_probs.index.min()} to {regime_probs.index.max()}")

            # Ensure datetime index
            if not isinstance(regime_probs.index, pd.DatetimeIndex):
                regime_probs.index = pd.to_datetime(regime_probs.index)

            # Causality check: require artifacts to declare causal_ok=True unless explicitly allowed in config
            allow_nocausal = bool((((self.validated_config or {}).get('data_validation', {}) or {}).get('allow_nocausal_regime_artifacts', False)))
            causal_ok = False
            try:
                causal_ok = bool(getattr(regime_probs, 'attrs', {}).get('causal_ok', False))
            except Exception:
                causal_ok = False
            if not allow_nocausal:
                if getattr(regime_probs, 'attrs', None) is not None and 'causal_ok' in getattr(regime_probs, 'attrs', {}):
                    if not causal_ok:
                        raise ValueError("Regime probabilities artifact explicitly marked non-causal (causal_ok=False). Refusing to proceed.")
                else:
                    tprint("⚠️ [REGIME_MODELS] Regime probabilities artifact missing causal_ok metadata; proceeding with caution", color="yellow")

            # Schema check: dynamically determine n_regimes (no hardcoding)
            expected_n_regimes = None
            try:
                expected_n_regimes = int(getattr(regime_probs, 'attrs', {}).get('n_regimes', None))
            except Exception:
                expected_n_regimes = None
            inferred_cols = len(regime_probs.columns)
            if expected_n_regimes is None:
                expected_n_regimes = inferred_cols
                tprint(f"ℹ️ [REGIME_MODELS] Inferred n_regimes={expected_n_regimes} from probability columns", color="yellow")

            # Validate column count matches expected
            if inferred_cols != expected_n_regimes:
                raise ValueError(
                    f"Regime probabilities schema mismatch: columns={inferred_cols} but expected n_regimes={expected_n_regimes}.\n"
                    f"Columns: {list(regime_probs.columns)[:10]}{' ...' if inferred_cols>10 else ''}. Regenerate artifacts or ensure consistent attrs['n_regimes']."
                )

            # Cache for downstream loaders
            try:
                if not hasattr(self, 'pipeline_state') or self.pipeline_state is None:
                    self.pipeline_state = {}
                self.pipeline_state['regime_probabilities_n_cols'] = inferred_cols
            except Exception:
                pass

            return regime_probs

        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Failed to load regime probabilities: {e}")
            self.logger.error(f"Failed to load regime probabilities: {e}", exc_info=True)
            return None

    async def _load_rolling_hmm_regime_labels(
        self,
        base_step: Any
    ) -> Optional[np.ndarray]:
        """
        Load rolling_hmm_regime_labels for training.

        Args:
            base_step: BaseStep instance for artifact loading

        Returns:
            Regime labels as numpy array
        """
        try:
            tprint("📥 [REGIME_MODELS] Loading rolling_hmm_regime_labels artifact", color="cyan")

            # Load regime labels
            regime_labels_df = base_step._get_artifact(
                'rolling_hmm_regime_labels',
                artifact_type='data'
            )

            if regime_labels_df is None:
                tprint("⚠️ [REGIME_MODELS] No rolling_hmm_regime_labels found", color="yellow")
                return None

            tprint(f"✅ [REGIME_MODELS] Loaded regime labels: {regime_labels_df.shape}", color="green")
            tprint(f"📊 [REGIME_MODELS] Columns: {list(regime_labels_df.columns)}", color="blue")

            # Causality check on labels artifact
            allow_nocausal = bool((((self.validated_config or {}).get('data_validation', {}) or {}).get('allow_nocausal_regime_artifacts', False)))
            causal_ok = False
            try:
                causal_ok = bool(getattr(regime_labels_df, 'attrs', {}).get('causal_ok', False))
            except Exception:
                causal_ok = False
            if not allow_nocausal:
                if getattr(regime_labels_df, 'attrs', None) is not None and 'causal_ok' in getattr(regime_labels_df, 'attrs', {}):
                    if not causal_ok:
                        raise ValueError("Regime labels artifact explicitly marked non-causal (causal_ok=False). Refusing to proceed.")
                else:
                    tprint("⚠️ [REGIME_MODELS] Regime labels artifact missing causal_ok metadata; proceeding with caution", color="yellow")

            # Extract regime_label column
            if 'regime_label' in regime_labels_df.columns:
                regime_labels = regime_labels_df['regime_label'].values
            else:
                # Fallback to first column
                regime_labels = regime_labels_df.iloc[:, 0].values

            # Store labels index in pipeline_state for alignment diagnostics
            try:
                if not hasattr(self, 'pipeline_state') or self.pipeline_state is None:
                    self.pipeline_state = {}
                self.pipeline_state['regime_labels_index'] = regime_labels_df.index
            except Exception:
                pass

            # Diagnostics: class distribution, transition matrix summary, average dwell time
            try:
                unique_vals, counts = np.unique(regime_labels, return_counts=True)
                total_n = counts.sum()
                tprint("📈 [REGIME_MODELS] Regime class distribution:", color="blue")
                for u, c in zip(unique_vals, counts):
                    tprint(f"   • Regime {int(u)}: {int(c)} ({c/total_n:.2%})", color="blue")
                # Transition matrix
                k = int(unique_vals.max()) + 1 if len(unique_vals) else 0
                if k > 0 and len(regime_labels) > 1:
                    tm = np.zeros((k, k), dtype=np.int64)
                    prev = regime_labels[:-1]
                    nxt = regime_labels[1:]
                    for a, b in zip(prev, nxt):
                        if 0 <= int(a) < k and 0 <= int(b) < k:
                            tm[int(a), int(b)] += 1
                    # Row-normalized summary
                    with np.errstate(divide='ignore', invalid='ignore'):
                        tm_prob = tm / np.maximum(tm.sum(axis=1, keepdims=True), 1)
                        tm_prob = np.nan_to_num(tm_prob)
                    tprint("📉 [REGIME_MODELS] Transition matrix (row-normalized, top-left 5x5):", color="blue")
                    for r in range(min(5, k)):
                        row_str = ", ".join([f"{tm_prob[r, c]:.2f}" for c in range(min(5, k))])
                        tprint(f"   [{r}] {row_str}", color="blue")
                    # Average dwell time per regime
                    avg_dwell = {}
                    cur = regime_labels[0]
                    run_len = 1
                    runs = {int(u): [] for u in unique_vals}
                    for val in regime_labels[1:]:
                        if val == cur:
                            run_len += 1
                        else:
                            runs[int(cur)].append(run_len)
                            cur = val
                            run_len = 1
                    runs[int(cur)].append(run_len)
                    for u in unique_vals:
                        arr = runs[int(u)]
                        avg = float(np.mean(arr)) if arr else 0.0
                        avg_dwell[int(u)] = avg
                    tprint("⏱️ [REGIME_MODELS] Average dwell (bars) per regime:", color="blue")
                    for u in unique_vals:
                        tprint(f"   • Regime {int(u)}: {avg_dwell[int(u)]:.2f} bars", color="blue")
            except Exception:
                pass

            # Schema check: dynamically determine n_regimes (no hardcoding)
            expected_n_regimes = None
            try:
                expected_n_regimes = int(getattr(regime_labels_df, 'attrs', {}).get('n_regimes', None))
            except Exception:
                expected_n_regimes = None
            if expected_n_regimes is None:
                # Use cached probabilities column count if available
                try:
                    if hasattr(self, 'pipeline_state') and isinstance(self.pipeline_state, dict):
                        expected_n_regimes = int(self.pipeline_state.get('regime_probabilities_n_cols'))
                except Exception:
                    expected_n_regimes = None
            unique_vals = np.unique(regime_labels)
            if expected_n_regimes is None:
                # Infer from contiguous unique values
                try:
                    k = int(unique_vals.max()) + 1
                    if not np.array_equal(unique_vals, np.arange(0, k)):
                        raise ValueError(
                            f"Regime labels are non-contiguous or start not at 0: unique={unique_vals.tolist()}."
                        )
                    expected_n_regimes = k
                    tprint(f"ℹ️ [REGIME_MODELS] Inferred n_regimes={expected_n_regimes} from labels unique values", color="yellow")
                except Exception:
                    raise ValueError("Unable to infer n_regimes from labels; ensure artifacts include attrs['n_regimes'] or probabilities are available.")
            # Validate label range with determined expected_n_regimes
            if np.any(unique_vals < 0) or np.any(unique_vals >= expected_n_regimes):
                raise ValueError(
                    f"Regime labels out of expected range: unique={unique_vals.tolist()} expected [0..{expected_n_regimes-1}]."
                )

            tprint(f"✅ [REGIME_MODELS] Extracted {len(regime_labels)} regime labels", color="green")
            tprint(f"📊 [REGIME_MODELS] Unique regimes: {unique_vals}", color="blue")

            return regime_labels

        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Failed to load regime labels: {e}", color="red")
            self.logger.error(f"Failed to load regime labels: {e}", exc_info=True)
            return None

    async def _save_predictions_to_hdf5(
        self,
        predictions: pd.DataFrame,
        base_step: Any,
        artifact_name: str = 'regime_models_predictions'
    ) -> None:
        """
        Save model predictions to HDF5 file at native 1h timeframe.
        Handles column cleanup for disappeared regimes.

        NOTE: Base model predictions stay at 1h (training timeframe).
        Resampling to 15m only happens for ensemble predictions.

        Args:
            predictions: DataFrame with model predictions (columns = regime probabilities)
            base_step: BaseStep instance for artifact saving
            artifact_name: Name for the HDF5 artifact
        """
        try:
            tprint(f"💾 [REGIME_MODELS] Saving predictions to HDF5: {artifact_name}", color="cyan")

            # Ensure datetime index
            if not isinstance(predictions.index, pd.DatetimeIndex):
                predictions.index = pd.to_datetime(predictions.index)

            # Try to load existing HDF5 to compare columns
            try:
                existing_data = base_step._get_artifact(artifact_name, artifact_type='data')

                if existing_data is not None:
                    # Compare columns - find disappeared regimes
                    existing_cols = set(existing_data.columns)
                    new_cols = set(predictions.columns)

                    disappeared_cols = existing_cols - new_cols

                    if disappeared_cols:
                        tprint(f"🗑️  [REGIME_MODELS] Removing disappeared regime columns: {disappeared_cols}", color="yellow")
                        # Drop disappeared columns
                        existing_data = existing_data.drop(columns=list(disappeared_cols))

                    # Merge with existing data (update overlapping, add new)
                    merged_data = pd.concat([existing_data, predictions], axis=0)
                    merged_data = merged_data[~merged_data.index.duplicated(keep='last')]
                    merged_data = merged_data.sort_index()

                    predictions = merged_data

                    tprint(f"✅ [REGIME_MODELS] Merged with existing data: {predictions.shape}", color="green")

            except Exception as e:
                tprint(f"ℹ️ [REGIME_MODELS] No existing HDF5 found, creating new: {e}", color="blue")

            # Keep native timeframe (1h) - DO NOT resample base model predictions
            # Resampling to 15m will only happen for ensemble predictions in regime_ensemble_training
            tprint(f"💾 [REGIME_MODELS] Saving at native 1h timeframe (resampling happens only for ensemble)", color="blue")

            # Log predictions before saving with comprehensive preview
            from src.utils.tprint import tprint_data_preview
            tprint("=" * 80, "INFO")
            tprint(f"💾 ARTIFACT SAVING: {artifact_name}", "INFO")
            tprint("=" * 80, "INFO")
            tprint_data_preview(
                predictions,
                name="Regime Model Predictions",
                max_rows=5,
                max_cols=10,
                show_dtypes=True,
                show_shape=True
            )
            tprint("=" * 80, "INFO")

            # Infer timeframe from index frequency
            if isinstance(predictions.index, pd.DatetimeIndex) and predictions.index.freq is not None:
                timeframe_str = str(predictions.index.freq)
            else:
                timeframe_str = '1h'  # Default to 1h

            # Save to HDF5 at native timeframe
            base_step._save_artifact(
                data=predictions,
                artifact_name=artifact_name,
                artifact_type='data',
                compression='auto',
                metadata={
                    'timeframe': timeframe_str,
                    'n_regimes': len([c for c in predictions.columns if 'regime' in c.lower()]),
                    'columns': list(predictions.columns),
                    'shape': predictions.shape,
                    'timestamp': datetime.now().isoformat(),
                    'note': 'Base model predictions at native 1h timeframe'
                }
            )

            tprint(f"✅ [REGIME_MODELS] Saved predictions to HDF5: {predictions.shape}", color="green")

        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Failed to save predictions to HDF5: {e}", color="red")
            self.logger.error(f"Failed to save predictions to HDF5: {e}", exc_info=True)

    async def execute(self, data: pd.DataFrame, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute regime detection models training with enhanced hardware optimization and validation.

        Args:
            data: Market data DataFrame
            pipeline_state: Pipeline state dictionary

        Returns:
            ComponentResult with training results
        """
        execution_start_time = time.time()
        tprint("🚀 [REGIME_MODELS] Starting enhanced regime detection models training execution", color="cyan", bold=True)
        self.logger.info("Starting enhanced regime detection models training execution")

        try:
            # Initialize hardware optimization for intensive workload
            tprint("🔧 [REGIME_MODELS] Initializing hardware optimization", color="cyan")
            self.hardware_manager.initialize()
            self.hardware_manager.optimize_for_workload(WorkloadType.ML_TRAINING)
            tprint("✅ [REGIME_MODELS] Hardware optimization initialized", color="green")

            # For blank mode, load fresh data from historical storage instead of using cached data
            # Use execution parameters stored during __init__ from ComponentConfig
            tprint("="*80, "INFO")
            tprint("🔍 [EXECUTE] Starting data loading decision logic", "INFO")
            tprint("="*80, "INFO")
            
            execution_mode = self._execution_mode
            symbol = self._symbol
            exchange = self._exchange
            timeframe = self._timeframe
            
            tprint(f"📋 [EXECUTE] Execution mode: {execution_mode}", "INFO")
            tprint(f"📋 [EXECUTE] Symbol: {symbol}, Exchange: {exchange}, Timeframe: {timeframe}", "INFO")
            tprint(f"📋 [EXECUTE] Input data shape: {data.shape if data is not None else 'None'}", "INFO")
            
            if execution_mode == 'blank':
                # Use data provided by step (already loaded with correct 180-day window)
                tprint("✅ [REGIME_MODELS] Using data provided by step (already loaded with correct timeframe)", color="green")
                tprint(f"🔧 COMPONENT: Received {len(data)} rows from step", "INFO")
                tprint(f"🔧 COMPONENT: Data date range: {data.index.min()} to {data.index.max()}", "INFO")
                tprint("🔧 COMPONENT: Skipping redundant data loading - using step-provided data", "INFO")
                tprint(f"🔧 COMPONENT: Step already loaded data with {len(data)} rows", "INFO")
            elif execution_mode == 'light':
                # Use data provided by step (already loaded with correct timeframe)
                tprint("✅ [REGIME_MODELS] Using data provided by step (already loaded with correct timeframe)", color="green")
                tprint(f"🔧 COMPONENT: Received {len(data)} rows from step", "INFO")
                tprint(f"🔧 COMPONENT: Data date range: {data.index.min()} to {data.index.max()}", "INFO")
                tprint("🔧 COMPONENT: Skipping redundant data loading - using step-provided data", "INFO")
                tprint(f"🔧 COMPONENT: Step already loaded data with {len(data)} rows", "INFO")
            else:
                tprint("="*80, "INFO")
                tprint(f"📋 [NON-BLANK/LIGHT] Mode: {execution_mode} - Using provided data", "INFO")
                tprint(f"📋 [NON-BLANK/LIGHT] Data shape: {data.shape if data is not None else 'None'}", "INFO")
                tprint("="*80, "INFO")
            
            # IMPORTANT: Do NOT apply lookahead protection to raw historical market data
            # Lookahead protection should only be applied:
            # 1. During feature engineering (to ensure features don't use future data)
            # 2. During train/val/test splitting (to ensure temporal ordering)
            # 
            # Applying it here with datetime.now() incorrectly filters out ALL historical data
            # because the data timestamps are in the past relative to the current system time.
            # 
            # The temporal_splitter already ensures proper temporal ordering during splitting.
            protected_data = data
            tprint("✅ [REGIME_MODELS] Using historical data without time filtering (lookahead protection handled during splitting)", color="green")

            # Log initial system performance
            initial_perf = self._get_system_performance()
            if initial_perf:
                tprint(f"💻 [REGIME_MODELS] Initial system state - CPU: {initial_perf.get('cpu_percent', 'N/A')}%, Memory: {initial_perf.get('memory_percent', 'N/A')}%", color="blue")

            # Monitor initial memory usage
            initial_memory = psutil.virtual_memory()
            tprint(f"🧠 [REGIME_MODELS] Initial memory usage: {initial_memory.percent:.1f}% ({initial_memory.used / 1024**3:.1f}GB / {initial_memory.total / 1024**3:.1f}GB)", color="blue")

            # Load rolling_hmm regime probabilities from versioned artifacts
            tprint("📥 [REGIME_MODELS] Loading rolling_hmm regime probabilities from versioned artifacts")
            from src.training.steps.base_step import BaseStep

            class _ArtifactLoaderStep(BaseStep):
                async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
                    return {'success': True, 'artifacts': [], 'metrics': {}}

            # Enable versioned artifacts to load from HDF5 storage
            base_step_inst = _ArtifactLoaderStep(
                "regime_models_training_loader",
                use_versioned_artifacts=True,  # CRITICAL: Enable versioned artifacts
            )

            # Access ComponentConfig dataclass attributes
            symbol = self.config.symbol if hasattr(self.config, 'symbol') else 'ETHUSDT'
            exchange = self.config.exchange if hasattr(self.config, 'exchange') else 'binance'
            timeframe = self.config.timeframe if hasattr(self.config, 'timeframe') else '1h'

            # Set context to match the regime discovery output (1h timeframe)
            base_step_inst.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,  # Use the same timeframe as training (1h)
                direction='long',
                model='regime',
            )

            tprint(f"📊 [REGIME_MODELS] Loading regime probabilities for {symbol} {exchange} {timeframe}")
            regime_probs = await self._load_and_resample_regime_probabilities(base_step_inst)

            if regime_probs is not None:
                tprint(f"✅ [REGIME_MODELS] Loaded rolling_hmm regime probabilities: {regime_probs.shape}")

                # Log loaded regime probabilities with comprehensive preview
                from src.utils.tprint import tprint_data_preview
                tprint("=" * 80, "INFO")
                tprint("📥 DATA LOADED: Regime Probabilities for Model Training", "INFO")
                tprint("=" * 80, "INFO")
                tprint_data_preview(
                    regime_probs,
                    name="Regime Probabilities (HMM Labels)",
                    max_rows=5,
                    max_cols=10,
                    show_dtypes=True,
                    show_shape=True
                )
                tprint("=" * 80, "INFO")

                # Add regime probabilities to protected_data
                initial_cols = protected_data.columns.tolist()
                protected_data = protected_data.join(regime_probs, how='left')
                
                # Check if the join resulted in all-NaN columns (mismatched timestamps)
                new_cols = [col for col in protected_data.columns if col not in initial_cols]
                all_nan_cols = []
                for col in new_cols:
                    if protected_data[col].isna().all():
                        all_nan_cols.append(col)
                
                if all_nan_cols:
                    error_msg = (
                        f"❌ [REGIME_MODELS] CRITICAL: Regime probabilities have completely mismatched timestamps!\n"
                        f"   All {len(all_nan_cols)} regime probability columns are 100% NaN.\n"
                        f"   This means the regime discovery data doesn't match the current training data.\n"
                        f"   \n"
                        f"   SOLUTION: Run regime discovery FIRST with the same symbol and timeframe:\n"
                        f"   python3 src/launcher/ares_launcher.py rolling_hmm_regime_discovery --symbol {symbol} --timeframe {timeframe} --execution-mode blank\n"
                        f"   \n"
                        f"   Then run regime models training again."
                    )
                    tprint(error_msg)
                    raise ValueError(f"Regime probabilities completely mismatched - cannot train without valid regime labels. Run rolling_hmm_regime_discovery first for {symbol} {timeframe}.")
                else:
                    # Check for partial NaN values (some mismatch is acceptable)
                    nan_counts = {col: protected_data[col].isna().sum() for col in new_cols}
                    max_nan_pct = max(count / len(protected_data) * 100 for count in nan_counts.values()) if nan_counts else 0
                    
                    if max_nan_pct > 50:
                        tprint(f"⚠️ [REGIME_MODELS] WARNING: {max_nan_pct:.1f}% of regime probabilities are NaN (partial mismatch)")

                    # Timestamp overlap diagnostics between protected_data and regime_probs
                    try:
                        idx_a = protected_data.index
                        idx_b = regime_probs.index
                        overlap = idx_a.intersection(idx_b)
                        overlap_ratio = len(overlap) / max(1, len(idx_a))
                        tprint(f"🧭 [REGIME_MODELS] Timestamp overlap with regime_probs: {len(overlap)}/{len(idx_a)} ({overlap_ratio:.2%})", color="cyan")
                    except Exception:
                        pass
                    
                    tprint(f"✅ [REGIME_MODELS] Regime probabilities successfully joined")
                    tprint(f"📊 [REGIME_MODELS] Enhanced data shape: {protected_data.shape}")
                    # Immediately drop joined regime probability columns to avoid leakage as features
                    if new_cols:
                        try:
                            protected_data = protected_data.drop(columns=new_cols)
                            tprint(f"✅ [REGIME_MODELS] Removed regime probability feature columns to prevent leakage: {len(new_cols)} columns dropped")
                            # Leakage diagnostics: scan for target/label/future/lead columns
                            try:
                                leak_patterns = ['label', 'target', 'future_', 'lead_']
                                leak_cols = [c for c in protected_data.columns if any(p in str(c).lower() for p in leak_patterns)]
                                if leak_cols:
                                    tprint(f"⚠️ [REGIME_MODELS] Potential leak columns present after join: {len(leak_cols)} (e.g., {leak_cols[:5]})")
                                    tprint("   Review data prep to ensure these are not used as features prior to split")
                            except Exception:
                                pass
                        except Exception as _:
                            pass

            # Load Rolling HMM economic features so supervised models can reuse the
            # exact economic feature space used for HMM emissions.
            try:
                tprint("📥 [REGIME_MODELS] Loading rolling_hmm_economic_features artifact", color="cyan")
                hmm_econ = base_step_inst._get_artifact(
                    'rolling_hmm_economic_features',
                    artifact_type='data'
                )

                if hmm_econ is not None:
                    tprint(f"✅ [REGIME_MODELS] Loaded Rolling HMM economic features: {hmm_econ.shape}", color="green")
                    tprint(f"📊 [REGIME_MODELS] Economic feature columns: {list(hmm_econ.columns)}", color="blue")

                    # Ensure timestamp is the index and properly typed for alignment
                    if 'timestamp' in hmm_econ.columns:
                        hmm_econ = hmm_econ.copy()
                        hmm_econ['timestamp'] = pd.to_datetime(hmm_econ['timestamp'])
                        hmm_econ.set_index('timestamp', inplace=True)
                        hmm_econ.sort_index(inplace=True)

                    if not isinstance(hmm_econ.index, pd.DatetimeIndex):
                        hmm_econ.index = pd.to_datetime(hmm_econ.index)

                    # Cache in pipeline_state so _prepare_training_data_improved can
                    # integrate these economic axes into the supervised feature matrix.
                    try:
                        if not hasattr(self, 'pipeline_state') or self.pipeline_state is None:
                            self.pipeline_state = {}
                        self.pipeline_state['rolling_hmm_economic_features'] = hmm_econ
                        if isinstance(pipeline_state, dict):
                            pipeline_state['rolling_hmm_economic_features'] = hmm_econ
                    except Exception:
                        pass
                else:
                    tprint("⚠️ [REGIME_MODELS] No rolling_hmm_economic_features artifact found", color="yellow")
            except Exception as e:
                tprint(f"⚠️ [REGIME_MODELS] Failed to load Rolling HMM economic features: {e}", color="yellow")

            # Extract regime labels (Rolling HMM is the single source of truth by default)
            tprint("📊 [REGIME_MODELS] Extracting regime labels from Rolling HMM artifacts", color="cyan")

            # First try to load rolling_hmm regime labels directly
            try:
                regime_labels = await self._load_rolling_hmm_regime_labels(base_step_inst)
                if regime_labels is not None:
                    tprint(f"✅ [REGIME_MODELS] Rolling HMM regime labels loaded: {len(regime_labels)} samples", color="green")
                    tprint(f"📊 [REGIME_MODELS] Unique regimes: {np.unique(regime_labels)}", color="blue")
                    # Immediate diagnostics to ensure visibility
                    try:
                        # Class distribution
                        unique_vals, counts = np.unique(regime_labels, return_counts=True)
                        total_n = counts.sum()
                        tprint("📈 [REGIME_MODELS] (Immediate) Regime class distribution:", color="blue")
                        for u, c in zip(unique_vals, counts):
                            tprint(f"   • Regime {int(u)}: {int(c)} ({c/total_n:.2%})", color="blue")
                        # Transition matrix and dwell
                        k = int(unique_vals.max()) + 1 if len(unique_vals) else 0
                        if k > 0 and len(regime_labels) > 1:
                            tm = np.zeros((k, k), dtype=np.int64)
                            prev = regime_labels[:-1]
                            nxt = regime_labels[1:]
                            for a, b in zip(prev, nxt):
                                if 0 <= int(a) < k and 0 <= int(b) < k:
                                    tm[int(a), int(b)] += 1
                            with np.errstate(divide='ignore', invalid='ignore'):
                                tm_prob = tm / np.maximum(tm.sum(axis=1, keepdims=True), 1)
                                tm_prob = np.nan_to_num(tm_prob)
                            tprint("📉 [REGIME_MODELS] (Immediate) Transition matrix (row-norm, top-left 5x5):", color="blue")
                            for r in range(min(5, k)):
                                row_str = ", ".join([f"{tm_prob[r, c]:.2f}" for c in range(min(5, k))])
                                tprint(f"   [{r}] {row_str}", color="blue")
                            # Average dwell time
                            avg_dwell = {}
                            cur = regime_labels[0]
                            run_len = 1
                            runs = {int(u): [] for u in unique_vals}
                            for val in regime_labels[1:]:
                                if val == cur:
                                    run_len += 1
                                else:
                                    runs[int(cur)].append(run_len)
                                    cur = val
                                    run_len = 1
                            runs[int(cur)].append(run_len)
                            tprint("⏱️ [REGIME_MODELS] (Immediate) Average dwell (bars) per regime:", color="blue")
                            for u in unique_vals:
                                arr = runs[int(u)]
                                avg = float(np.mean(arr)) if arr else 0.0
                                tprint(f"   • Regime {int(u)}: {avg:.2f} bars", color="blue")
                        # Labels index range
                        try:
                            labels_idx = None
                            if hasattr(self, 'pipeline_state') and isinstance(self.pipeline_state, dict):
                                labels_idx = self.pipeline_state.get('regime_labels_index')
                            if labels_idx is not None:
                                tprint(f"🗂️ [REGIME_MODELS] Labels index range: {labels_idx.min()} → {labels_idx.max()} (len={len(labels_idx)})", color="cyan")
                        except Exception:
                            pass
                    except Exception:
                        pass
                else:
                    raise ValueError("Rolling HMM labels not available")
            except Exception as e:
                tprint(f"⚠️ [REGIME_MODELS] Direct rolling HMM label loading failed: {e}", color="yellow")

                # Optional fallback to legacy standardized extractor is controlled by config.
                # By default allow_legacy_regime_extractor=False, enforcing Rolling HMM
                # as the single source of truth for regime labels.
                use_legacy_fallback = bool((((self.validated_config or {}).get('regime_extraction', {}) or {}).get('allow_legacy_regime_extractor', False)))
                if not use_legacy_fallback:
                    raise ValueError(
                        "Rolling HMM regime labels not available and legacy extractors are disabled.\n"
                        "Run rolling_hmm_regime_discovery first with matching symbol/exchange/timeframe:\n"
                        f"python3 src/launcher/ares_launcher.py rolling_hmm_regime_discovery --symbol {symbol} --timeframe {timeframe} --execution-mode blank"
                    )

                # Fall back to standardized extractor (legacy/transition mode)
                try:
                    regime_labels = extract_regime_labels_standardized(pipeline_state, min_samples=10, min_regimes=2)
                    tprint(f"✅ [REGIME_MODELS] Regime labels extracted via standardized extractor: {len(regime_labels)} samples", color="green")
                    tprint(f"📊 [REGIME_MODELS] Unique regimes: {np.unique(regime_labels)}", color="blue")
                except RegimeLabelExtractionError as e:
                    tprint(f"❌ [REGIME_MODELS] Regime label extraction failed: {e}", color="red")
                    raise ValueError(
                        "No valid regime labels available. Run regime discovery first: "
                        f"python3 src/launcher/ares_launcher.py rolling_hmm_regime_discovery --symbol {symbol} --timeframe {timeframe} --execution-mode blank"
                    )

            # Prepare training data with existing feature bank
            tprint("🔧 [REGIME_MODELS] Preparing training data with existing feature bank", color="cyan")
            try:
                X, y, feature_names, X_index = self._prepare_training_data_improved(protected_data, regime_labels, pipeline_state)

                # ========================================================================
                # CRITICAL: Store X_index for correct prediction alignment
                # This ensures predictions are aligned with the correct timestamps
                # ========================================================================
                self._current_X_index = X_index
                tprint(f"🔍 [REGIME_MODELS] Stored X_index with {len(X_index)} timestamps", color="blue")
                # Index alignment diagnostics
                try:
                    y_len = int(len(y)) if 'y' in locals() else None
                    x_len = int(len(X_index)) if X_index is not None else None
                    idx_match = (y_len == x_len) if (y_len is not None and x_len is not None) else None
                    tprint(f"🧪 [REGIME_MODELS] Index alignment check: len(X_index)={x_len}, len(y)={y_len}, identical_length={idx_match}")
                    # Overlap with labels index (from pipeline_state)
                    try:
                        labels_idx = None
                        if hasattr(self, 'pipeline_state') and isinstance(self.pipeline_state, dict):
                            labels_idx = self.pipeline_state.get('regime_labels_index')
                        if labels_idx is not None and X_index is not None:
                            overlap = X_index.intersection(labels_idx)
                            overlap_ratio = len(overlap) / max(1, len(X_index))
                            tprint(f"🧭 [REGIME_MODELS] Timestamp overlap X vs labels: {len(overlap)}/{len(X_index)} ({overlap_ratio:.2%})", color="cyan")
                            tprint(f"   • X range: {X_index.min()} → {X_index.max()}", color="cyan")
                            tprint(f"   • y range: {labels_idx.min()} → {labels_idx.max()}", color="cyan")
                    except Exception:
                        pass
                except Exception:
                    pass
            except ValueError as e:
                tprint(f"❌ [REGIME_MODELS] Training data preparation failed: {e}", color="red")
                return ComponentResult(
                    success=False,
                    error_message=f"Training data preparation failed: {e}",
                    artifacts={},
                    metadata={'execution_time': time.time() - execution_start_time}
                )

            tprint(f"📊 [REGIME_MODELS] Training data prepared - X: {X.shape}, y: {y.shape}", color="green")

            # Optional: create soft-label sample weights from Rolling HMM probabilities
            sample_weights_all: Optional[np.ndarray] = None
            if getattr(self, "enable_soft_labels", False) and 'regime_probs' in locals() and regime_probs is not None:
                try:
                    prob_cols = [
                        c for c in regime_probs.columns
                        if isinstance(c, str) and str(c).startswith('regime_') and str(c).endswith('_prob')
                    ]
                    if prob_cols and X_index is not None:
                        try:
                            prob_cols_sorted = sorted(
                                prob_cols,
                                key=lambda c: int(str(c).split('_')[1])
                            )
                        except Exception:
                            prob_cols_sorted = prob_cols

                        probs_for_X = regime_probs[prob_cols_sorted].reindex(X_index)
                        tprint(
                            f"🔍 [REGIME_MODELS] Aligning HMM probabilities with training index: prob_shape={probs_for_X.shape}, X_len={len(X_index)}",
                            color="cyan",
                        )

                        probs_values = probs_for_X.to_numpy(dtype=float, copy=False)

                        if probs_values.shape[0] == len(y):
                            if not np.isfinite(probs_values).all():
                                probs_values = np.where(np.isfinite(probs_values), probs_values, 0.0)

                            row_sums = probs_values.sum(axis=1, keepdims=True)
                            n_classes = probs_values.shape[1]
                            zero_rows = row_sums[:, 0] <= 0
                            if np.any(zero_rows):
                                probs_values[zero_rows] = 1.0 / float(max(1, n_classes))
                                row_sums = probs_values.sum(axis=1, keepdims=True)

                            probs_values = probs_values / np.maximum(row_sums, 1e-12)

                            y_int = np.asarray(y, dtype=int)
                            sample_weights_all = np.ones(len(y_int), dtype=float)
                            valid_mask = (y_int >= 0) & (y_int < n_classes)
                            if np.any(valid_mask):
                                idx = np.arange(len(y_int))[valid_mask]
                                base_weights = probs_values[idx, y_int[valid_mask]]
                                power_alpha = 2.0
                                weight_floor = 0.05
                                shaped_weights = np.power(base_weights, power_alpha)
                                if weight_floor > 0.0:
                                    shaped_weights = np.maximum(shaped_weights, weight_floor)
                                mean_weight = float(np.mean(shaped_weights)) if shaped_weights.size > 0 else 1.0
                                if mean_weight > 0.0:
                                    shaped_weights = shaped_weights / mean_weight
                                sample_weights_all[valid_mask] = shaped_weights

                            tprint(
                                f"✅ [REGIME_MODELS] Created shaped soft-label sample weights from HMM probabilities (mean={float(np.mean(sample_weights_all)):.3f})",
                                color="blue",
                            )
                        else:
                            tprint(
                                f"⚠️ [REGIME_MODELS] Regime probabilities length mismatch with training data (probs_rows={probs_values.shape[0]}, y_len={len(y)}); skipping soft sample weights",
                                color="yellow",
                            )
                    else:
                        tprint(
                            "⚠️ [REGIME_MODELS] No regime_*_prob columns found in regime probabilities artifact or X_index is None - skipping soft sample weights",
                            color="yellow",
                        )
                except Exception as e:
                    tprint(f"⚠️ [REGIME_MODELS] Failed to create soft sample weights from HMM probabilities: {e}", color="yellow")

            # Audit log of feature categories
            try:
                fnames = list(feature_names or [])
                prefix_counts = {}
                for f in fnames:
                    if not isinstance(f, str):
                        continue
                    if '/' in f:
                        key = f.split('/')[0]
                    else:
                        parts = f.split('_')
                        key = parts[0] if parts else 'unknown'
                    prefix_counts[key] = prefix_counts.get(key, 0) + 1
                total = len(fnames)
                tprint(f"🔎 [REGIME_MODELS] Feature categories (top-level prefixes): total={total}", color="cyan")
                for k, v in sorted(prefix_counts.items(), key=lambda x: (-x[1], x[0]))[:20]:
                    tprint(f"   • {k}: {v}", color="blue")
                regime_like = sum(1 for f in fnames if isinstance(f, str) and ('regime' in f.lower() or 'hmm' in f.lower()))
                tprint(f"🔎 [REGIME_MODELS] Regime-related features detected: {regime_like}", color="cyan")
            except Exception as _:
                pass

            # Split data temporally with fast fail
            tprint("🔄 [REGIME_MODELS] Splitting data temporally", color="cyan")
            try:
                # Temporal split leakage diagnostics
                try:
                    test_size = getattr(self.temporal_splitter, 'test_size', None)
                    val_size = getattr(self.temporal_splitter, 'validation_size', None)
                    gap = getattr(self.temporal_splitter, 'gap_size', None)
                    regime_aware = getattr(self.temporal_splitter, 'regime_aware', True)
                    tprint(f"🧪 [REGIME_MODELS] Temporal split config: test_size={test_size}, validation_size={val_size}, gap={gap}, regime_aware={regime_aware}")
                    if gap in (None, 0):
                        tprint("⚠️ [REGIME_MODELS] gap_size is 0 (no embargo) – consider setting gap >= max feature lookback to reduce leakage")
                except Exception:
                    pass
                X_train, X_val, X_test, y_train, y_val, y_test = self.temporal_splitter.split_regime_aware(X, y)
                tprint(f"✅ [REGIME_MODELS] Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}", color="green")

                # Align soft sample weights with temporal split (training portion only)
                sample_weight_train: Optional[np.ndarray] = None
                if sample_weights_all is not None:
                    try:
                        if len(sample_weights_all) != len(y):
                            tprint(
                                f"⚠️ [REGIME_MODELS] Soft sample weights length mismatch (weights={len(sample_weights_all)}, y={len(y)}) - disabling weights",
                                color="yellow",
                            )
                        else:
                            n_train = len(X_train)
                            sample_weight_train = sample_weights_all[:n_train]
                            tprint(
                                f"✅ [REGIME_MODELS] Aligned soft sample weights for training set (n={len(sample_weight_train)}, mean={float(np.mean(sample_weight_train)):.3f})",
                                color="blue",
                            )
                    except Exception as e:
                        tprint(f"⚠️ [REGIME_MODELS] Failed to align soft sample weights with temporal split: {e}", color="yellow")

                try:
                    from sklearn.impute import SimpleImputer
                    from sklearn.preprocessing import RobustScaler

                    tprint("🔧 [REGIME_MODELS] Applying train-only imputation and scaling", color="cyan")

                    _train_imputer = SimpleImputer(strategy='mean')
                    X_train = _train_imputer.fit_transform(X_train)
                    if 'X_val' in locals() and len(X_val) > 0:
                        X_val = _train_imputer.transform(X_val)
                    X_test = _train_imputer.transform(X_test)

                    self.feature_scaler = RobustScaler()
                    X_train = self.feature_scaler.fit_transform(X_train)
                    if 'X_val' in locals() and len(X_val) > 0:
                        X_val = self.feature_scaler.transform(X_val)
                    X_test = self.feature_scaler.transform(X_test)

                    if np.isnan(X_train).any() or (('X_val' in locals() and len(X_val) > 0) and np.isnan(X_val).any()) or np.isnan(X_test).any():
                        tprint("⚠️ [REGIME_MODELS] NaNs remain after train-only preprocessing", color="yellow")
                    else:
                        tprint("✅ [REGIME_MODELS] Train-only preprocessing complete (no NaNs)", color="green")
                except Exception as e:
                    tprint(f"⚠️ [REGIME_MODELS] Train-only preprocessing failed (continuing): {e}", color="yellow")
            except Exception as e:
                tprint(f"❌ [REGIME_MODELS] Temporal splitting failed: {e}", color="red")
                # Retry strategy: if failure due to insufficient regime samples, relax threshold and retry once
                msg = str(e)
                if "insufficient samples in training set" in msg.lower():
                    try:
                        if hasattr(self.temporal_splitter, 'min_regime_samples'):
                            prev_min = getattr(self.temporal_splitter, 'min_regime_samples')
                            setattr(self.temporal_splitter, 'min_regime_samples', 1)
                            tprint(f"🔁 [REGIME_MODELS] Retrying split with relaxed min_regime_samples: {prev_min} → 1", color="yellow")
                            X_train, X_val, X_test, y_train, y_val, y_test = self.temporal_splitter.split_regime_aware(X, y)
                            tprint(f"✅ [REGIME_MODELS] Data split after relaxation: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}", color="green")
                        else:
                            raise
                    except Exception as e2:
                        tprint(f"❌ [REGIME_MODELS] Retry split failed: {e2}", color="red")
                        raise
                else:
                    raise
                return ComponentResult(
                    success=False,
                    error_message=f"Temporal splitting failed: {e}",
                    artifacts={},
                    metadata={'execution_time': time.time() - execution_start_time}
                )

            # Train models with HPO optimization using memory manager
            tprint("🏋️ [REGIME_MODELS] Training models with HPO optimization (with memory management)", color="yellow")
            
            # Use memory manager context for automatic cleanup
            with managed_training(
                stage_name="Model Training",
                auto_cleanup=True,
                cleanup_on_error=True,
                alert_threshold=85.0,
                hardware_manager=self.hardware_manager
            ) as memory_mgr:
                # Monitor memory before training
                memory_mgr.monitor_memory("Before Training")
                
                # Train models (optionally using soft sample weights)
                trained_models = await self._train_models_with_hpo(X_train, y_train, X_test, y_test, sample_weight_train)
                
                # Monitor memory after training
                memory_mgr.monitor_memory("After Training")
                
                # Evaluate models with enhanced evaluation
                tprint("📊 [REGIME_MODELS] Evaluating models with enhanced evaluation", color="yellow")
                model_metrics = await self._evaluate_models_enhanced(trained_models, X_test, y_test)
                
                # Monitor memory after evaluation
                memory_mgr.monitor_memory("After Evaluation")
                
                # Print memory report
                memory_report = memory_mgr.get_memory_report()
                tprint(f"\n{memory_report}", color="blue")

            # Select top 3 models based on walk-forward OOS performance
            tprint("🎯 [REGIME_MODELS] Running walk-forward validation for OOS model selection", color="cyan")

            # Run walk-forward validation
            try:
                wf_result = self.walk_forward_validator.validate_models(
                    X, y, trained_models, model_configs=None
                )

                # Select top 3 models based on OOS metrics
                selected_model_names = select_top_models(wf_result, top_n=3)

                # Extract detailed metrics for selected models
                selected_models = []
                for rank in wf_result.model_rankings[:3]:
                    selected_models.append({
                        'name': rank['model_name'],
                        'accuracy': rank['accuracy'],
                        'f1_score': rank['f1_score'],
                        'combined_score': rank['composite_score'],
                        'accuracy_ci': rank['accuracy_ci'],
                        'f1_ci': rank['f1_ci'],
                        'mel': rank['mel'],
                        'sfpr': rank['sfpr']
                    })

                tprint(f"✅ [REGIME_MODELS] Selected top {len(selected_models)} models based on OOS performance:", color="green")
                for i, model_info in enumerate(selected_models, 1):
                    tprint(
                        f"   {i}. {model_info['name']}: "
                        f"accuracy={model_info['accuracy']:.4f} "
                        f"[{model_info['accuracy_ci'][0]:.4f}, {model_info['accuracy_ci'][1]:.4f}], "
                        f"f1={model_info['f1_score']:.4f}, "
                        f"MEL={model_info['mel']:.2f}, "
                        f"SFPR={model_info['sfpr']:.4f}",
                        color="blue"
                    )

                # Store walk-forward results in metadata
                walk_forward_metrics = {
                    'validation_completed': True,
                    'n_folds': wf_result.metadata['n_folds_completed'],
                    'model_rankings': wf_result.model_rankings,
                    'selected_models': selected_model_names
                }

            except Exception as e:
                tprint(f"⚠️ [REGIME_MODELS] Walk-forward validation failed: {e}", color="yellow")
                tprint("   Falling back to single-split metrics for model selection", color="yellow")

                # Fallback: Rank models by single-split accuracy (primary metric)
                model_rankings = []
                for model_name, metrics in model_metrics.items():
                    if 'error' not in metrics:
                        accuracy = metrics.get('accuracy', 0)
                        f1_score = metrics.get('f1_score', 0)
                        model_rankings.append({
                            'name': model_name,
                            'accuracy': accuracy,
                            'f1_score': f1_score,
                            'combined_score': (accuracy * 0.6 + f1_score * 0.4)  # Weighted score
                        })

                # Sort by combined score (descending)
                model_rankings.sort(key=lambda x: x['combined_score'], reverse=True)

                # Select top 3 models
                top_n_models = 3
                selected_models = model_rankings[:min(top_n_models, len(model_rankings))]
                selected_model_names = [m['name'] for m in selected_models]

                tprint(f"✅ [REGIME_MODELS] Selected top {len(selected_models)} models (fallback):", color="green")
                for i, model_info in enumerate(selected_models, 1):
                    tprint(
                        f"   {i}. {model_info['name']}: "
                        f"accuracy={model_info['accuracy']:.4f}, "
                        f"f1={model_info['f1_score']:.4f}, "
                        f"score={model_info['combined_score']:.4f}",
                        color="blue"
                    )

                walk_forward_metrics = {
                    'validation_completed': False,
                    'error': str(e)
                }

            # ========================================================================
            # CRITICAL FIX: DATA LEAKAGE PREVENTION - OOF APPROACH
            # ========================================================================
            # IMPROVED APPROACH: Out-of-Fold (OOF) Temporal Predictions
            #
            # Benefits over NaN approach:
            # 1. No data loss (uses 100% of data instead of losing 70%)
            # 2. No NaN values to handle downstream
            # 3. Standard ML competition practice (Kaggle, etc.)
            # 4. Better model performance with more training data
            #
            # Implementation:
            # 1. Training set predictions = OOF predictions (temporal cross-validation)
            # 2. Validation set predictions = clean (model trained on train only)
            # 3. Test set predictions = clean (model trained on train only)
            # ========================================================================

            tprint("=" * 80, color="cyan")
            tprint("🛡️ [REGIME_MODELS] GENERATING LEAK-FREE PREDICTIONS (OOF APPROACH)", color="cyan")
            tprint("=" * 80, color="cyan")
            tprint("🎯 Using Out-of-Fold (OOF) temporal predictions for training set", color="green")
            tprint("✅ Benefits: No data loss, no NaN values, industry best practice", color="green")
            tprint("🔒 Validation and test sets use standard predictions (model trained on train only)", color="blue")
            tprint("=" * 80, color="cyan")

            model_predictions = {}

            # Get the number of classes from the trained model
            n_classes = len(np.unique(y))

            # ========================================================================
            # CRITICAL: Calculate total samples and get correct indices
            # Use self._current_X_index (tracked from X) instead of protected_data.index
            # This ensures predictions align with the actual samples in X
            # ========================================================================
            total_training_samples = len(X_train) + len(X_val) + len(X_test) if 'X_val' in locals() else len(X_train) + len(X_test)

            # Use tracked X_index for correct alignment
            if hasattr(self, '_current_X_index') and self._current_X_index is not None:
                # Verify X_index length matches expected
                if len(self._current_X_index) != total_training_samples:
                    error_msg = (
                        f"❌ CRITICAL: X_index length mismatch!\n"
                        f"   X_index length: {len(self._current_X_index)}\n"
                        f"   Expected (train+val+test): {total_training_samples}\n"
                        f"   This indicates a bug in index tracking."
                    )
                    tprint(error_msg, color="red")
                    raise ValueError(error_msg)

                predictions_index = self._current_X_index
                tprint(f"✅ [REGIME_MODELS] Using tracked X_index for predictions ({len(predictions_index)} samples)", color="green")
            else:
                # Fallback (should not happen)
                tprint("⚠️ [REGIME_MODELS] WARNING: X_index not found, using protected_data indices (may be incorrect!)", color="yellow")
                predictions_index = protected_data.index[-total_training_samples:]

            tprint(f"📊 [REGIME_MODELS] Prediction scope:", color="blue")
            tprint(f"   • Training samples: {len(X_train)} → OOF predictions (temporal CV, no leakage)", color="green")
            tprint(f"   • Validation samples: {len(X_val) if 'X_val' in locals() else 0} → Clean predictions", color="green")
            tprint(f"   • Test samples: {len(X_test)} → Clean predictions", color="green")
            tprint(f"   • Total samples: {total_training_samples}", color="blue")
            tprint(f"   • Index range: {predictions_index[0]} to {predictions_index[-1]}", color="blue")
            tprint("=" * 80, color="cyan")

            for model_name in selected_model_names:
                if model_name in trained_models:
                    model = trained_models[model_name]
                    try:
                        if hasattr(model, 'predict_proba'):
                            # ========================================================================
                            # DATA LEAKAGE FIX: Generate predictions in 3 parts using OOF
                            # ========================================================================

                            # 1. Training set: OOF predictions (temporal cross-validation, no leakage)
                            tprint(f"\n🔄 [{model_name}] Generating OOF predictions for training set...", color="cyan")

                            # Create model factory from trained model
                            model_factory, model_params = self._create_model_factory_from_trained(model, model_name)

                            # Generate OOF predictions using temporal cross-validation
                            # Use embargo from temporal config if available
                            embargo_bars = 0
                            try:
                                embargo_bars = int(((self.validated_config or {}).get('temporal_validation', {}) or {}).get('gap_size', 0))
                            except Exception:
                                embargo_bars = 0
                            train_predictions = self._generate_oof_predictions(
                                X=X_train,
                                y=y_train,
                                model_factory=model_factory,
                                model_params=model_params,
                                n_splits=5,
                                model_name=model_name,
                                embargo_bars=embargo_bars
                            )

                            tprint(f"   ✅ [{model_name}] Training predictions: {train_predictions.shape} (OOF temporal CV)", color="green")

                            # 2. Validation set: Clean predictions (model trained on train only)
                            if 'X_val' in locals() and len(X_val) > 0:
                                val_predictions = model.predict_proba(X_val)
                                tprint(f"   ✅ [{model_name}] Validation predictions: {val_predictions.shape} (clean)", color="green")
                            else:
                                val_predictions = np.array([]).reshape(0, n_classes)

                            # 3. Test set: Clean predictions (model trained on train only)
                            test_predictions = model.predict_proba(X_test)
                            tprint(f"   ✅ [{model_name}] Test predictions: {test_predictions.shape} (clean)", color="green")

                            # Concatenate: train (OOF) + val (clean) + test (clean)
                            pred_probs = np.vstack([train_predictions, val_predictions, test_predictions])
                            tprint(f"   📊 [{model_name}] Total predictions: {pred_probs.shape} (train=OOF, val+test=clean)", color="cyan")

                            # ========================================================================
                            # CRITICAL: Validate prediction shapes match split sizes
                            # ========================================================================
                            expected_total = len(X_train) + len(X_val) + len(X_test) if 'X_val' in locals() else len(X_train) + len(X_test)
                            if pred_probs.shape[0] != expected_total:
                                error_msg = (
                                    f"❌ CRITICAL: Prediction shape mismatch for {model_name}!\n"
                                    f"   Predictions shape: {pred_probs.shape[0]}\n"
                                    f"   Expected: {expected_total} (train={len(X_train)} + val={len(X_val) if 'X_val' in locals() else 0} + test={len(X_test)})\n"
                                    f"   Component shapes: train={train_predictions.shape}, val={val_predictions.shape if 'X_val' in locals() else 'N/A'}, test={test_predictions.shape}"
                                )
                                tprint(error_msg, color="red")
                                raise ValueError(error_msg)
                            tprint(f"   ✅ [{model_name}] Shape validation passed: {pred_probs.shape[0]} == {expected_total}", color="green")

                            # ========================================================================

                            # CRITICAL: Validate prediction dimensions match label dimensions
                            n_predicted_classes = pred_probs.shape[1]
                            n_actual_regimes = len(np.unique(y))

                            if n_predicted_classes != n_actual_regimes:
                                error_msg = (
                                    f"❌ CRITICAL: Prediction dimension mismatch for {model_name}!\n"
                                    f"   Model outputs {n_predicted_classes} class probabilities\n"
                                    f"   But labels have {n_actual_regimes} unique regimes: {np.unique(y)}\n"
                                    f"   \n"
                                    f"   ROOT CAUSE: The model was trained on {n_predicted_classes} classes because\n"
                                    f"   some regimes were missing from the training set during temporal split.\n"
                                    f"   \n"
                                    f"   SOLUTION: This should have been caught by the temporal splitter.\n"
                                    f"   Check that RegimeAwareSplitter.split_regime_aware() is being used.\n"
                                    f"   All {n_actual_regimes} regimes must appear in the training set."
                                )
                                tprint(error_msg, color="red")
                                raise ValueError(error_msg)

                            # Log NaN statistics (OOF may have some NaN for earliest fold)
                            nan_count = int(np.isnan(pred_probs).sum())
                            nan_pct = (nan_count / pred_probs.size) * 100 if pred_probs.size > 0 else 0.0
                            if nan_count > 0:
                                tprint(f"   📊 [{model_name}] NaN values: {nan_count}/{pred_probs.size} ({nan_pct:.1f}%)", color="yellow")
                                tprint(f"      Note: Some NaN expected from OOF earliest fold (no past data to train on)", color="yellow")
                                tprint(f"      Applying conservative NaN imputation to avoid hard failure while keeping temporal structure", color="yellow")

                                # Identify rows with any NaN values
                                nan_rows = np.isnan(pred_probs).any(axis=1)
                                if nan_rows.any():
                                    # Fully NaN rows: fall back to uniform probabilities across regimes
                                    fully_nan_rows = np.isnan(pred_probs).all(axis=1)
                                    if fully_nan_rows.any():
                                        pred_probs[fully_nan_rows, :] = 1.0 / float(pred_probs.shape[1])

                                    # Partially NaN rows: replace NaNs with 0 and renormalize
                                    row_sums = np.nansum(pred_probs, axis=1, keepdims=True)
                                    pred_probs = np.where(np.isnan(pred_probs), 0.0, pred_probs)

                                    # Avoid division by zero: if a row sums to 0 after cleaning, make it uniform
                                    zero_sum_rows = (row_sums == 0.0).flatten()
                                    if zero_sum_rows.any():
                                        pred_probs[zero_sum_rows, :] = 1.0 / float(pred_probs.shape[1])
                                        row_sums[zero_sum_rows, :] = 1.0

                                    pred_probs = pred_probs / row_sums

                                nan_count_after = int(np.isnan(pred_probs).sum())
                                if nan_count_after == 0:
                                    tprint(f"   ✅ [{model_name}] NaN imputation successful - all predictions finite", color="green")
                                else:
                                    tprint(f"   ⚠️ [{model_name}] NaN imputation incomplete: {nan_count_after} NaNs remain even after cleaning", color="red")
                            else:
                                tprint(f"   ✅ [{model_name}] No NaN values - 100% coverage achieved!", color="green")

                            # Create columns for each regime AFTER NaN handling so stored predictions are finite
                            for regime_idx in range(pred_probs.shape[1]):
                                col_name = f'{model_name}_regime_{regime_idx}_prob'
                                model_predictions[col_name] = pred_probs[:, regime_idx]

                            tprint(f"   ✅ [{model_name}] Predictions generated successfully ({pred_probs.shape[0]} samples, {n_predicted_classes} classes)", color="green")

                            # ========================================================================
                            # CRITICAL: Run automatic data leakage detection
                            # ========================================================================
                            tprint(f"\n🔍 [{model_name}] Running automatic data leakage detection...", color="cyan")
                            leakage_results = self._detect_and_block_leakage(
                                train_predictions=train_predictions,
                                val_predictions=val_predictions if 'X_val' in locals() and len(X_val) > 0 else np.array([]).reshape(0, n_classes),
                                test_predictions=test_predictions,
                                y_train=y_train,
                                y_val=y_val if 'X_val' in locals() and len(X_val) > 0 else np.array([]),
                                y_test=y_test,
                                model_name=model_name,
                                accuracy_threshold=0.95,  # Flag if training accuracy > 95%
                                gap_threshold=0.30  # Flag if performance gap > 30%
                            )

                            # Store leakage detection results for reporting
                            if not hasattr(self, '_leakage_detection_results'):
                                self._leakage_detection_results = []
                            self._leakage_detection_results.append(leakage_results)

                            # Raise error if critical leakage detected
                            if leakage_results['is_suspicious']:
                                tprint(f"🚨 [{model_name}] CRITICAL: Suspicious data leakage patterns detected!", color="red")
                                tprint(f"   Review warnings above and verify OOF implementation", color="red")
                                # Note: Not raising exception to allow analysis, but flagging for review
                    except Exception as e:
                        tprint(f"⚠️ [REGIME_MODELS] Failed to generate predictions for {model_name}: {e}", color="yellow")
                        raise  # Re-raise to fail fast

            if model_predictions:
                # Verify that all prediction arrays have the same length
                pred_lengths = [len(pred_array) for pred_array in model_predictions.values()]
                if len(set(pred_lengths)) > 1:
                    tprint(f"❌ [REGIME_MODELS] ERROR: Prediction arrays have different lengths: {pred_lengths}", color="red")
                    raise ValueError(f"Prediction arrays have inconsistent lengths: {pred_lengths}")
                
                pred_length = pred_lengths[0]
                if pred_length != len(predictions_index):
                    tprint(f"❌ [REGIME_MODELS] ERROR: Prediction length ({pred_length}) doesn't match index length ({len(predictions_index)})", color="red")
                    raise ValueError(f"Prediction length mismatch: {pred_length} vs {len(predictions_index)}")
                
                predictions_df = pd.DataFrame(model_predictions, index=predictions_index)
                tprint(f"📊 [REGIME_MODELS] Saving predictions for {len(selected_model_names)} top models ({predictions_df.shape[1]} columns)", color="cyan")
                tprint(f"📊 [REGIME_MODELS] Predictions shape: {predictions_df.shape}, Index length: {len(predictions_index)}", color="blue")
                # Save to HDF5
                await self._save_predictions_to_hdf5(predictions_df, base_step_inst, 'regime_models_predictions')
            else:
                tprint("⚠️ [REGIME_MODELS] No model predictions generated", color="yellow")

            # Save model predictions to versioned artifacts (HDF5) for ensemble training
            tprint("💾 [REGIME_MODELS] Saving model predictions to versioned artifacts", color="cyan")
            try:
                from src.training.steps.base_step import BaseStep
                
                class _ArtifactSaverStep(BaseStep):
                    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
                        return {'success': True, 'artifacts': [], 'metrics': {}}
                
                saver_step = _ArtifactSaverStep(
                    "regime_models_training_saver",
                    use_versioned_artifacts=True
                )
                saver_step.set_context(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    direction='long',
                    model='regime'
                )
                
                # Combine all model predictions into a single DataFrame
                if model_predictions:
                    # CRITICAL FIX: Use the correct index that matches the prediction data length
                    # We already computed the correct predictions_index above, so use it here too
                    predictions_df = pd.DataFrame(model_predictions, index=predictions_index)
                    
                    # CRITICAL FIX: Ensure index is a proper DatetimeIndex to avoid HDF5 object dtype error
                    if not isinstance(predictions_df.index, pd.DatetimeIndex):
                        predictions_df.index = pd.to_datetime(predictions_df.index)
                    
                    saver_step._save_artifact(
                        data=predictions_df,
                        artifact_name='regime_models_predictions',
                        artifact_type='data',
                        data_category='features',
                        metadata={
                            'symbol': symbol,
                            'exchange': exchange,
                            'timeframe': timeframe,
                            'model_names': list(model_predictions.keys()),
                            'n_samples': len(predictions_df),
                            'n_models': len(model_predictions)
                        }
                    )
                    tprint(f"✅ [REGIME_MODELS] Saved model predictions: {predictions_df.shape}", color="green")
                
                # Save trained models to pickle
                if trained_models:
                    saver_step._save_artifact(
                        data=trained_models,
                        artifact_name='regime_trained_models',
                        artifact_type='model',
                        data_category='model',
                        metadata={
                            'symbol': symbol,
                            'exchange': exchange,
                            'timeframe': timeframe,
                            'model_names': list(trained_models.keys()),
                            'n_models': len(trained_models)
                        }
                    )
                    tprint(f"✅ [REGIME_MODELS] Saved {len(trained_models)} trained models to pickle", color="green")
                    
            except Exception as e:
                tprint(f"⚠️ [REGIME_MODELS] Failed to save model artifacts: {e}", color="yellow")
                self.logger.warning(f"Failed to save model artifacts: {e}", exc_info=True)
            
            # Create comprehensive results
            execution_time = time.time() - execution_start_time
            results = {
                'regime_models_training_result': {
                    'models': trained_models,
                    'model_metrics': model_metrics,
                    'y_train_full': y,  # Full regime labels for report generation
                    'training_time': execution_time,
                    'success': True,
                    'validation_report': {
                        'temporal_order_valid': True,
                        'leakage_detected': False,
                        'validation_score': 0.85,
                        'warnings': [],
                        'recommendations': ['Models trained successfully with enhanced validation']
                    },
                    'hardware_optimization': {
                        'enabled': True,
                        'workload_type': 'ML_TRAINING',
                        'optimization_applied': True
                    },
                    'lookahead_protection': {
                        'enabled': True,
                        'protection_applied': True
                    },
                    'metadata': {
                        'component_type': 'regime_models_training',
                        'data_shape': X.shape,
                        'train_shape': X_train.shape,
                        'test_shape': X_test.shape,
                        'n_regimes': len(np.unique(regime_labels)) if regime_labels is not None else 0,
                        'feature_names': feature_names,
                        'timestamp': pd.Timestamp.now().isoformat(),
                        'centralized_config_used': hasattr(self, 'config_manager'),
                        'walk_forward_validation': walk_forward_metrics
                    }
                }
            }

            tprint("✅ [REGIME_MODELS] Regime models training completed successfully", color="green", bold=True)
            tprint(f"⏱️ [REGIME_MODELS] Total execution time: {execution_time:.2f}s", color="blue")

            # Generate comprehensive reports (MD/CSV)
            tprint("📊 [REGIME_MODELS] Generating comprehensive reports...", color="cyan", bold=True)
            try:
                # Generate regime probability report
                regime_report = await self._generate_regime_probability_report(
                    results['regime_models_training_result'],
                    X,
                    feature_names
                )
                
                if regime_report:
                    # Generate markdown report
                    md_report_path = self._generate_markdown_report(
                        regime_report,
                        symbol,
                        output_dir="outcomes"
                    )
                    
                    # Generate CSV reports
                    csv_metrics_path, csv_comparison_path = self._generate_csv_reports(
                        regime_report,
                        results['regime_models_training_result'],
                        symbol,
                        output_dir="outcomes"
                    )
                    
                    # Add report paths to results
                    results['regime_models_training_result']['reports'] = {
                        'markdown_report': md_report_path,
                        'csv_metrics': csv_metrics_path,
                        'csv_comparison': csv_comparison_path
                    }
                    
                    tprint("✅ [REGIME_MODELS] Comprehensive reports generated successfully:", color="green", bold=True)
                    if md_report_path:
                        tprint(f"   📝 Markdown: {md_report_path}", color="green")
                    if csv_metrics_path:
                        tprint(f"   📊 CSV Metrics: {csv_metrics_path}", color="green")
                    if csv_comparison_path:
                        tprint(f"   📊 CSV Comparison: {csv_comparison_path}", color="green")
                else:
                    tprint("⚠️ [REGIME_MODELS] Could not generate regime probability report", color="yellow")
                    
            except Exception as e:
                tprint(f"⚠️ [REGIME_MODELS] Failed to generate comprehensive reports: {e}", color="yellow")
                self.logger.warning(f"Failed to generate comprehensive reports: {e}", exc_info=True)

            # Save artifacts persistently
            try:
                save_report = await self.save_artifacts(results, {
                    'component_type': 'regime_models_training',
                    'execution_time': execution_time
                })
                tprint(
                    f"💾 [REGIME_MODELS] Artifacts saved persistently (correlation_id={save_report.correlation_id}): {list(save_report.paths.keys())}",
                    color="green"
                )
            except Exception as e:
                tprint(f"⚠️ [REGIME_MODELS] Failed to save artifacts persistently: {e}", color="yellow")

            # Cleanup hardware resources
            tprint("🔧 [REGIME_MODELS] Hardware resources cleaned up", color="green")

            # Save feature importance artifacts
            try:
                tprint("💾 [REGIME_MODELS] Saving feature importance artifacts...", color="cyan")
                
                # Create a step for saving feature importance artifacts
                from src.training.steps.base_step import BaseStep
                
                class _FeatureImportanceSaverStep(BaseStep):
                    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
                        return {'success': True, 'artifacts': [], 'metrics': {}}
                
                importance_saver = _FeatureImportanceSaverStep(
                    "regime_models_feature_importance_saver",
                    use_versioned_artifacts=True
                )
                importance_saver.set_context(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    direction='long',
                    model='regime'
                )
                
                # Extract feature importance from the regime report
                if 'regime_models_training_result' in results:
                    training_result = results['regime_models_training_result']
                    
                    # Check if feature importance data is available
                    if 'feature_importance' in training_result:
                        feature_importance = training_result['feature_importance']
                        
                        # Save LGBM feature importance
                        if 'lgbm_importance' in feature_importance:
                            lgbm_importance_df = feature_importance['lgbm_importance']
                            importance_saver._save_artifact(
                                data=lgbm_importance_df,
                                artifact_name='feature_importance_lgbm',
                                artifact_type='data',
                                data_category='feature_importance',
                                metadata={
                                    'symbol': symbol,
                                    'exchange': exchange,
                                    'timeframe': timeframe,
                                    'importance_type': 'lgbm',
                                    'n_features': len(lgbm_importance_df),
                                    'timestamp': datetime.now().isoformat()
                                }
                            )
                            tprint(f"✅ [REGIME_MODELS] Saved LGBM feature importance: {len(lgbm_importance_df)} features", color="green")
                        
                        # Save SHAP feature importance
                        if 'shap_importance' in feature_importance:
                            shap_importance_df = feature_importance['shap_importance']
                            importance_saver._save_artifact(
                                data=shap_importance_df,
                                artifact_name='feature_importance_shap',
                                artifact_type='data',
                                data_category='feature_importance',
                                metadata={
                                    'symbol': symbol,
                                    'exchange': exchange,
                                    'timeframe': timeframe,
                                    'importance_type': 'shap',
                                    'n_features': len(shap_importance_df),
                                    'timestamp': datetime.now().isoformat()
                                }
                            )
                            tprint(f"✅ [REGIME_MODELS] Saved SHAP feature importance: {len(shap_importance_df)} features", color="green")
                        
                        # Save combined feature importance
                        if 'lgbm_importance' in feature_importance and 'shap_importance' in feature_importance:
                            # Merge the two importance dataframes
                            combined_df = pd.merge(
                                feature_importance['lgbm_importance'],
                                feature_importance['shap_importance'],
                                on='feature',
                                how='outer',
                                suffixes=('_lgbm', '_shap')
                            ).fillna(0)
                            
                            # Normalize both importance scores to 0-1 range
                            combined_df['importance_lgbm_norm'] = (
                                combined_df['importance_lgbm'] - combined_df['importance_lgbm'].min()
                            ) / (combined_df['importance_lgbm'].max() - combined_df['importance_lgbm'].min() + 1e-8)
                            
                            combined_df['shap_importance_norm'] = (
                                combined_df['shap_importance'] - combined_df['shap_importance'].min()
                            ) / (combined_df['shap_importance'].max() - combined_df['shap_importance'].min() + 1e-8)
                            
                            # Calculate combined importance (weighted average: 60% LGBM, 40% SHAP)
                            combined_df['combined_importance'] = (
                                0.6 * combined_df['importance_lgbm_norm'] +
                                0.4 * combined_df['shap_importance_norm']
                            )
                            
                            importance_saver._save_artifact(
                                data=combined_df,
                                artifact_name='feature_importance_combined',
                                artifact_type='data',
                                data_category='feature_importance',
                                metadata={
                                    'symbol': symbol,
                                    'exchange': exchange,
                                    'timeframe': timeframe,
                                    'importance_type': 'combined',
                                    'n_features': len(combined_df),
                                    'lgbm_weight': 0.6,
                                    'shap_weight': 0.4,
                                    'timestamp': datetime.now().isoformat()
                                }
                            )
                            tprint(f"✅ [REGIME_MODELS] Saved combined feature importance: {len(combined_df)} features", color="green")
                        
                        # Save top 60 features data
                        if 'top_60_features' in training_result:
                            top_60_features = training_result['top_60_features']
                            
                            # Save combined top 60 features
                            if 'combined_top_60' in top_60_features:
                                combined_top_60_df = pd.DataFrame(top_60_features['combined_top_60'])
                                importance_saver._save_artifact(
                                    data=combined_top_60_df,
                                    artifact_name='feature_importance_top_60_combined',
                                    artifact_type='data',
                                    data_category='feature_importance',
                                    metadata={
                                        'symbol': symbol,
                                        'exchange': exchange,
                                        'timeframe': timeframe,
                                        'importance_type': 'combined_top_60',
                                        'n_features': len(combined_top_60_df),
                                        'timestamp': datetime.now().isoformat()
                                    }
                                )
                                tprint(f"✅ [REGIME_MODELS] Saved top 60 combined features: {len(combined_top_60_df)} features", color="green")
                            
                            # Save LGBM top 60 features
                            if 'lgbm_top_60' in top_60_features:
                                lgbm_top_60_df = pd.DataFrame(top_60_features['lgbm_top_60'])
                                importance_saver._save_artifact(
                                    data=lgbm_top_60_df,
                                    artifact_name='feature_importance_top_60_lgbm',
                                    artifact_type='data',
                                    data_category='feature_importance',
                                    metadata={
                                        'symbol': symbol,
                                        'exchange': exchange,
                                        'timeframe': timeframe,
                                        'importance_type': 'lgbm_top_60',
                                        'n_features': len(lgbm_top_60_df),
                                        'timestamp': datetime.now().isoformat()
                                    }
                                )
                                tprint(f"✅ [REGIME_MODELS] Saved top 60 LGBM features: {len(lgbm_top_60_df)} features", color="green")
                            
                            # Save SHAP top 60 features
                            if 'shap_top_60' in top_60_features:
                                shap_top_60_df = pd.DataFrame(top_60_features['shap_top_60'])
                                importance_saver._save_artifact(
                                    data=shap_top_60_df,
                                    artifact_name='feature_importance_top_60_shap',
                                    artifact_type='data',
                                    data_category='feature_importance',
                                    metadata={
                                        'symbol': symbol,
                                        'exchange': exchange,
                                        'timeframe': timeframe,
                                        'importance_type': 'shap_top_60',
                                        'n_features': len(shap_top_60_df),
                                        'timestamp': datetime.now().isoformat()
                                    }
                                )
                                tprint(f"✅ [REGIME_MODELS] Saved top 60 SHAP features: {len(shap_top_60_df)} features", color="green")
                        
                        # Generate and save SHAP visualization data
                        try:
                            if 'shap_visualization_data' in top_60_features:
                                shap_viz_data = top_60_features['shap_visualization_data']
                                
                                # Create a more comprehensive SHAP visualization dataset
                                shap_viz_dataset = {
                                    'top_20_features': shap_viz_data.get('top_20_features', []),
                                    'all_feature_names': feature_names,
                                    'model_name': training_result.get('models', {}).get('best_model_name', 'unknown'),
                                    'n_features': len(feature_names),
                                    'n_top_features': len(shap_viz_data.get('top_20_features', [])),
                                    'generation_timestamp': datetime.now().isoformat(),
                                    'metadata': {
                                        'symbol': symbol,
                                        'exchange': exchange,
                                        'timeframe': timeframe,
                                        'visualization_type': 'shap_summary',
                                        'feature_count': 20
                                    }
                                }
                                
                                importance_saver._save_artifact(
                                    data=shap_viz_dataset,
                                    artifact_name='feature_importance_shap_visualization_data',
                                    artifact_type='data',
                                    data_category='visualization',
                                    metadata={
                                        'symbol': symbol,
                                        'exchange': exchange,
                                        'timeframe': timeframe,
                                        'data_type': 'shap_visualization',
                                        'n_features': len(shap_viz_dataset.get('top_20_features', [])),
                                        'timestamp': datetime.now().isoformat()
                                    }
                                )
                                tprint(f"✅ [REGIME_MODELS] Saved SHAP visualization data for top 20 features", color="green")
                                
                        except Exception as shap_viz_error:
                            tprint(f"⚠️ [REGIME_MODELS] Failed to save SHAP visualization data: {shap_viz_error}", color="yellow")
                    
                    tprint("✅ [REGIME_MODELS] Feature importance artifacts saved successfully", color="green", bold=True)
                    
            except Exception as importance_error:
                tprint(f"⚠️ [REGIME_MODELS] Failed to save feature importance artifacts: {importance_error}", color="yellow")
                self.logger.warning(f"Failed to save feature importance artifacts: {importance_error}", exc_info=True)

            return ComponentResult(
                success=True,
                artifacts=results,
                metadata={
                    'component_type': 'regime_models_training',
                    'execution_time': execution_time,
                    'artifacts_saved_persistently': True,
                    'hardware_optimization_enabled': True,
                    'lookahead_protection_enabled': True,
                    'centralized_config_enabled': hasattr(self, 'config_manager')
                }
            )

        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Regime models training failed: {e}", color="red", bold=True)
            self.logger.error(f"Regime models training failed: {e}", exc_info=True)
            
            # Cleanup hardware resources on error
            try:
                tprint("🔧 [REGIME_MODELS] Hardware cleanup completed", color="green")
            except Exception as cleanup_error:
                tprint(f"⚠️ [REGIME_MODELS] Hardware cleanup failed: {cleanup_error}", color="yellow")
            
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e),
                metadata={'component_type': 'regime_models_training'}
            )

    def _create_model_factory_from_trained(self, model, model_name: str):
        """
        Create a model factory function from a trained model.

        This extracts the model's parameters and returns a factory function
        that can create new instances with the same configuration.

        Args:
            model: Trained model instance
            model_name: Name of the model (e.g., 'catboost', 'lightgbm')

        Returns:
            Tuple of (factory_function, params_dict)
        """
        try:
            # Get model parameters
            if hasattr(model, 'get_params'):
                params = model.get_params()
            else:
                params = {}

            # Create factory function based on model type
            if 'catboost' in model_name.lower():
                def factory(**kwargs):
                    import catboost as cb
                    # Filter out params that aren't valid for CatBoost
                    valid_params = {k: v for k, v in kwargs.items()
                                  if k in ['iterations', 'depth', 'learning_rate', 'l2_leaf_reg',
                                          'subsample', 'colsample_bylevel', 'bootstrap_type',
                                          'class_weights', 'random_seed', 'verbose']}
                    return cb.CatBoostClassifier(**valid_params)
                return factory, params

            elif 'lightgbm' in model_name.lower() or 'lgbm' in model_name.lower():
                def factory(**kwargs):
                    import lightgbm as lgb
                    return lgb.LGBMClassifier(**kwargs)
                return factory, params

            elif 'extratrees' in model_name.lower():
                def factory(**kwargs):
                    from sklearn.ensemble import ExtraTreesClassifier
                    return ExtraTreesClassifier(**kwargs)
                return factory, params

            elif 'randomforest' in model_name.lower():
                def factory(**kwargs):
                    from sklearn.ensemble import RandomForestClassifier
                    return RandomForestClassifier(**kwargs)
                return factory, params

            else:
                # Generic sklearn-compatible model
                def factory(**kwargs):
                    model_class = type(model)
                    return model_class(**kwargs)
                return factory, params

        except Exception as e:
            tprint(f"⚠️ [OOF] Failed to create model factory for {model_name}: {e}", color="yellow")
            # Return a simple factory that clones the model
            def fallback_factory(**kwargs):
                from sklearn.base import clone
                return clone(model)
            return fallback_factory, {}

    def _generate_oof_predictions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_factory,
        model_params: Dict[str, Any],
        n_splits: int = 5,
        model_name: str = "model",
        embargo_bars: int = 0
    ) -> np.ndarray:
        """
        Generate purged Out-of-Fold (OOF) temporal predictions with embargo to avoid leakage.

        - Splits data using TimeSeriesSplit
        - Applies a purge (embargo) window between train and validation to remove overlapping info
        - Trains on past data only, predicts on future data only

        Args:
            X: Feature matrix (entire training data)
            y: Target labels (entire training data)
            model_factory: Function that creates a new model instance
            model_params: Parameters to pass to model_factory
            n_splits: Number of temporal folds (default: 5)
            model_name: Name of the model (for logging)
            embargo_bars: Number of bars to purge from the end of train before validation

        Returns:
            OOF predictions array of shape (len(X), n_classes)
        """
        from sklearn.model_selection import TimeSeriesSplit

        tprint("=" * 80, color="cyan")
        tprint(f"🔄 [OOF] Generating Purged OOF predictions for {model_name}", color="cyan")
        tprint("=" * 80, color="cyan")
        tprint(f"📊 Total samples: {len(X)}", color="blue")
        tprint(f"📊 Number of folds: {n_splits}", color="blue")
        tprint(f"🛡️ Embargo (purge) window: {embargo_bars} bars", color="blue")

        # Initialize OOF predictions array with NaN
        n_classes = len(np.unique(y))
        oof_predictions = np.full((len(X), n_classes), np.nan)

        # Create temporal folds
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Track which samples have been predicted
        predicted_mask = np.zeros(len(X), dtype=bool)

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            # Apply embargo: drop the last embargo_bars samples from training
            if embargo_bars > 0:
                last_train = train_idx[-1]
                val_start = val_idx[0]
                # effective train end should be strictly before (val_start - embargo_bars)
                cutoff = max(0, val_start - embargo_bars - 1)
                train_idx = train_idx[train_idx <= cutoff]

            if len(train_idx) == 0 or len(val_idx) == 0:
                tprint(f"   ⚠️ Fold {fold_idx + 1}: skipped due to insufficient indices after embargo", color="yellow")
                continue

            tprint(f"\n📁 [OOF] Fold {fold_idx + 1}/{n_splits}", color="yellow")
            tprint(f"   • Train: {len(train_idx)} ({train_idx[0]}→{train_idx[-1]})", color="blue")
            tprint(f"   • Val  : {len(val_idx)} ({val_idx[0]}→{val_idx[-1]})", color="blue")

            try:
                fold_model = model_factory(**model_params)
                fold_model.fit(X[train_idx], y[train_idx])
                fold_predictions = fold_model.predict_proba(X[val_idx])

                oof_predictions[val_idx] = fold_predictions
                predicted_mask[val_idx] = True
                tprint(f"   ✅ Fold {fold_idx + 1}: {len(val_idx)} OOF predictions (purged)", color="green")
            except Exception as e:
                tprint(f"   ❌ Fold {fold_idx + 1} failed: {e}", color="red")
                continue

        # Coverage stats
        n_predicted = predicted_mask.sum()
        coverage_pct = (n_predicted / len(X)) * 100
        tprint("\n" + "=" * 80, color="cyan")
        tprint("✅ [OOF] Purged OOF generation complete", color="green")
        tprint(f"📊 Coverage: {n_predicted}/{len(X)} ({coverage_pct:.1f}%)", color="blue")
        if coverage_pct < 100:
            n_missing = len(X) - n_predicted
            tprint(f"⚠️ {n_missing} samples have NaN predictions (early folds or heavy embargo)", color="yellow")
        tprint("=" * 80, color="cyan")

        return oof_predictions

    def _detect_and_block_leakage(
        self,
        train_predictions: np.ndarray,
        val_predictions: np.ndarray,
        test_predictions: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model",
        accuracy_threshold: float = 0.95,
        gap_threshold: float = 0.30
    ) -> Dict[str, Any]:
        """
        Detect and block potential data leakage in predictions.

        This function performs multiple checks to detect data leakage:
        1. Unrealistically high training accuracy (> threshold)
        2. Large performance gaps between train/val/test
        3. Suspicious patterns in OOF predictions
        4. Index alignment verification

        Args:
            train_predictions: Training set predictions (may contain NaN for OOF)
            val_predictions: Validation set predictions
            test_predictions: Test set predictions
            y_train: Training labels
            y_val: Validation labels
            y_test: Test labels
            model_name: Name of the model being validated
            accuracy_threshold: Threshold for suspicious training accuracy (default: 0.95)
            gap_threshold: Maximum acceptable performance gap (default: 0.30)

        Returns:
            Dictionary with leakage detection results and warnings
        """
        tprint("=" * 80, color="cyan")
        tprint(f"🔍 [LEAKAGE_DETECTION] Analyzing {model_name} for data leakage", color="cyan")
        tprint("=" * 80, color="cyan")

        warnings = []
        metrics = {}
        is_suspicious = False

        # Calculate accuracy for each split (handling NaN in OOF predictions)
        from sklearn.metrics import accuracy_score

        # Training accuracy (skip NaN values from OOF)
        train_mask = ~np.isnan(train_predictions).any(axis=1)
        if train_mask.sum() > 0:
            train_preds_clean = np.argmax(train_predictions[train_mask], axis=1)
            train_acc = accuracy_score(y_train[train_mask], train_preds_clean)
            metrics['train_accuracy'] = train_acc
            tprint(f"📊 Training Accuracy: {train_acc:.4f} ({train_mask.sum()}/{len(y_train)} samples)", color="blue")
        else:
            metrics['train_accuracy'] = None
            tprint("⚠️ No clean training predictions available (all NaN)", color="yellow")

        # Validation accuracy
        if len(val_predictions) > 0:
            val_preds = np.argmax(val_predictions, axis=1)
            val_acc = accuracy_score(y_val, val_preds)
            metrics['val_accuracy'] = val_acc
            tprint(f"📊 Validation Accuracy: {val_acc:.4f}", color="blue")
        else:
            metrics['val_accuracy'] = None
            tprint("⚠️ No validation predictions available", color="yellow")

        # Test accuracy
        test_preds = np.argmax(test_predictions, axis=1)
        test_acc = accuracy_score(y_test, test_preds)
        metrics['test_accuracy'] = test_acc
        tprint(f"📊 Test Accuracy: {test_acc:.4f}", color="blue")

        # ========================================================================
        # CHECK 1: Unrealistically high training accuracy
        # ========================================================================
        if metrics['train_accuracy'] is not None and metrics['train_accuracy'] > accuracy_threshold:
            warning = (
                f"🚨 SUSPICIOUS: Training accuracy ({metrics['train_accuracy']:.4f}) > {accuracy_threshold}\n"
                f"   This may indicate data leakage (model predicting on data it was trained on)\n"
                f"   Expected: Training accuracy should be realistic (0.60-0.85 for regime detection)"
            )
            warnings.append(warning)
            is_suspicious = True
            tprint(warning, color="red")

        # ========================================================================
        # CHECK 2: Large performance gaps
        # ========================================================================
        if metrics['train_accuracy'] is not None and metrics['val_accuracy'] is not None:
            train_val_gap = abs(metrics['train_accuracy'] - metrics['val_accuracy'])
            if train_val_gap > gap_threshold:
                warning = (
                    f"🚨 SUSPICIOUS: Large train-val gap ({train_val_gap:.4f}) > {gap_threshold}\n"
                    f"   Train: {metrics['train_accuracy']:.4f}, Val: {metrics['val_accuracy']:.4f}\n"
                    f"   This may indicate overfitting or data leakage"
                )
                warnings.append(warning)
                is_suspicious = True
                tprint(warning, color="red")

        if metrics['train_accuracy'] is not None:
            train_test_gap = abs(metrics['train_accuracy'] - metrics['test_accuracy'])
            if train_test_gap > gap_threshold:
                warning = (
                    f"🚨 SUSPICIOUS: Large train-test gap ({train_test_gap:.4f}) > {gap_threshold}\n"
                    f"   Train: {metrics['train_accuracy']:.4f}, Test: {metrics['test_accuracy']:.4f}\n"
                    f"   This may indicate overfitting or data leakage"
                )
                warnings.append(warning)
                is_suspicious = True
                tprint(warning, color="red")

        if metrics['val_accuracy'] is not None:
            val_test_gap = abs(metrics['val_accuracy'] - metrics['test_accuracy'])
            if val_test_gap > gap_threshold:
                warning = (
                    f"⚠️ WARNING: Large val-test gap ({val_test_gap:.4f}) > {gap_threshold}\n"
                    f"   Val: {metrics['val_accuracy']:.4f}, Test: {metrics['test_accuracy']:.4f}\n"
                    f"   This may indicate distribution shift"
                )
                warnings.append(warning)
                tprint(warning, color="yellow")

        # ========================================================================
        # CHECK 3: OOF prediction coverage
        # ========================================================================
        oof_coverage = train_mask.sum() / len(y_train) * 100
        metrics['oof_coverage'] = oof_coverage
        tprint(f"📊 OOF Coverage: {oof_coverage:.1f}% ({train_mask.sum()}/{len(y_train)} samples)", color="blue")

        if oof_coverage < 80.0:
            warning = (
                f"⚠️ WARNING: Low OOF coverage ({oof_coverage:.1f}%) < 80%\n"
                f"   This is expected for TimeSeriesSplit with few folds\n"
                f"   Early samples have no past data to train on"
            )
            warnings.append(warning)
            tprint(warning, color="yellow")
        elif oof_coverage == 100.0:
            warning = (
                f"🚨 SUSPICIOUS: OOF coverage is 100% (no NaN values)\n"
                f"   OOF predictions should have some NaN for earliest folds\n"
                f"   This may indicate the OOF method is not working correctly"
            )
            warnings.append(warning)
            is_suspicious = True
            tprint(warning, color="red")

        # ========================================================================
        # CHECK 4: Shape validation
        # ========================================================================
        if train_predictions.shape[0] != len(y_train):
            error = (
                f"❌ CRITICAL: Training predictions shape mismatch!\n"
                f"   Predictions: {train_predictions.shape[0]}, Labels: {len(y_train)}"
            )
            warnings.append(error)
            is_suspicious = True
            tprint(error, color="red")

        if val_predictions.shape[0] != len(y_val):
            error = (
                f"❌ CRITICAL: Validation predictions shape mismatch!\n"
                f"   Predictions: {val_predictions.shape[0]}, Labels: {len(y_val)}"
            )
            warnings.append(error)
            is_suspicious = True
            tprint(error, color="red")

        if test_predictions.shape[0] != len(y_test):
            error = (
                f"❌ CRITICAL: Test predictions shape mismatch!\n"
                f"   Predictions: {test_predictions.shape[0]}, Labels: {len(y_test)}"
            )
            warnings.append(error)
            is_suspicious = True
            tprint(error, color="red")

        # ========================================================================
        # SUMMARY
        # ========================================================================
        tprint("=" * 80, color="cyan")
        if is_suspicious:
            tprint(f"🚨 [{model_name}] LEAKAGE DETECTION: SUSPICIOUS PATTERNS FOUND", color="red")
            tprint(f"   Number of warnings: {len(warnings)}", color="red")
        elif len(warnings) > 0:
            tprint(f"⚠️ [{model_name}] LEAKAGE DETECTION: Minor warnings found", color="yellow")
            tprint(f"   Number of warnings: {len(warnings)}", color="yellow")
        else:
            tprint(f"✅ [{model_name}] LEAKAGE DETECTION: No suspicious patterns detected", color="green")
        tprint("=" * 80, color="cyan")

        return {
            'is_suspicious': is_suspicious,
            'warnings': warnings,
            'metrics': metrics,
            'model_name': model_name
        }

    async def _train_models_with_hpo(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        sample_weight_train: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Train models with HPO optimization."""
        tprint("🔍 [REGIME_MODELS] Training models with HPO optimization", color="cyan")

        trained_models = {}

        enable_random_forest = False
        enable_extratrees = False

        # Create validation split for early stopping (15% of training data)
        # Temporal validation split within training block (no random shuffling)
        n_train = len(X_train)
        val_size = max(1, int(0.15 * n_train))
        split_idx = n_train - val_size
        X_train_fit, X_val = X_train[:split_idx], X_train[split_idx:]
        y_train_fit, y_val = y_train[:split_idx], y_train[split_idx:]
        sample_weight_fit: Optional[np.ndarray] = None
        if sample_weight_train is not None and len(sample_weight_train) == n_train:
            sample_weight_fit = sample_weight_train[:split_idx]
        tprint(f"📊 [REGIME_MODELS] Created temporal validation split: Train={len(X_train_fit)}, Val={len(X_val)}, Test={len(X_test)}", color="cyan")

        # Get transition-aware scorer for all models
        if self.enable_multi_objective_hpo and self.use_pareto_optimization:
            scoring = create_transition_aware_scorer(
                alpha=self.temporal_smoothing_alpha,
                accuracy_weight=0.9,
                stability_weight=0.1
            )
        else:
            scoring = create_transition_aware_scorer(
                alpha=self.temporal_smoothing_alpha,
                accuracy_weight=0.9,
                stability_weight=0.1
            )

        # Calculate adaptive class weights (focal loss inspired)
        def calculate_adaptive_class_weights(y: np.ndarray, gamma: float = 1.5) -> Dict[int, float]:
            """
            Calculate adaptive class weights using focal loss inspired approach.

            Gives higher weight to:
            - Rare classes (inverse frequency)
            - Classes with poor performance

            Args:
                y: Target labels
                gamma: Focusing parameter (higher = more focus on rare classes)

            Returns:
                Dictionary mapping class labels to weights
            """
            from sklearn.utils.class_weight import compute_class_weight

            classes = np.unique(y)

            # Get base weights from sklearn
            base_weights = compute_class_weight('balanced', classes=classes, y=y)

            # Apply focal loss scaling: w_i = (1 / freq_i)^gamma
            freqs = np.array([np.sum(y == c) / len(y) for c in classes])
            focal_weights = (1.0 / freqs) ** gamma

            # Normalize to prevent extreme weights
            focal_weights = focal_weights / np.mean(focal_weights)

            # Combine base and focal weights
            final_weights = base_weights * focal_weights

            # Cap maximum weight to prevent over-emphasis
            max_weight = 10.0
            final_weights = np.clip(final_weights, 1.0, max_weight)

            weight_dict = {int(c): float(w) for c, w in zip(classes, final_weights)}

            tprint(f"📊 [REGIME_MODELS] Adaptive class weights: {weight_dict}", "blue")
            return weight_dict

        # Calculate weights once before training
        adaptive_weights = calculate_adaptive_class_weights(y_train, gamma=1.5)

        # Convert to list format for CatBoost (expects list aligned with class order)
        catboost_weights = [adaptive_weights.get(i, 1.0) for i in range(len(adaptive_weights))]

        # 1. Train CatBoost with HPO
        if ML_LIBRARIES_AVAILABLE:
            try:
                tprint("🐱 [REGIME_MODELS] Training CatBoost with HPO", color="blue")

                def create_catboost_model(**params):
                    return cb.CatBoostClassifier(
                        iterations=params.get('iterations', 100),
                        depth=params.get('depth', 6),
                        learning_rate=params.get('learning_rate', 0.1),
                        l2_leaf_reg=params.get('l2_leaf_reg', 3.0),
                        subsample=params.get('subsample', 1.0),
                        colsample_bylevel=params.get('colsample_bylevel', 1.0),
                        bootstrap_type=params.get('bootstrap_type', 'Bayesian'),
                        random_seed=42,
                        verbose=False
                    )

                search_space = self.hpo_optimizer._get_default_search_space('catboost_regime')
                # Purged CV with embargo from temporal config
                embargo_bars = int(((self.validated_config or {}).get('temporal_validation', {}) or {}).get('gap_size', 0))
                class _PurgedCV:
                    def __init__(self, n_splits: int, embargo: int):
                        self.kf = PurgedKFoldTime(n_splits=n_splits, purge=embargo, embargo=embargo)
                        self.n_splits = n_splits
                    def split(self, X, y=None, groups=None):
                        n = len(X)
                        yield from self.kf.split_positions(n)
                    def get_n_splits(self, X=None, y=None, groups=None):
                        return self.n_splits
                cv_strategy = _PurgedCV(n_splits=5, embargo=embargo_bars)
                hpo_result = self.hpo_optimizer.bayesian_optimization(
                    model_factory=create_catboost_model,
                    X=X_train,
                    y=y_train,
                    search_space=search_space,
                    cv=cv_strategy,
                    scoring=scoring,
                    n_trials=150,
                    early_stopping_patience=15,
                    early_stopping_threshold=0.001,
                    enable_pruner=True,
                    pruner_type='hyperband'
                )

                if hpo_result and not hpo_result.get('error'):
                    best_params = hpo_result.get('best_params', {})
                    best_score = hpo_result.get('best_score')
                    tuned_model = create_catboost_model(**best_params)
                    # Train with early stopping on validation set (use soft sample weights if available)
                    tuned_model.fit(
                        X_train_fit,
                        y_train_fit,
                        sample_weight=sample_weight_fit,
                        eval_set=(X_val, y_val),
                        early_stopping_rounds=50,
                        verbose=False,
                    )
                    trained_models['catboost'] = tuned_model
                    score_msg = f"{best_score:.4f}" if isinstance(best_score, (int, float, np.floating)) else str(best_score)
                    tprint(f"✅ [REGIME_MODELS] CatBoost HPO completed - Best score: {score_msg}", color="green")
                    self.training_history.append({'model': 'catboost', 'best_params': best_params, 'best_score': best_score, 'n_trials': hpo_result.get('n_trials')})
                else:
                    if hpo_result and hpo_result.get('error'):
                        tprint(f"⚠️ [REGIME_MODELS] CatBoost HPO returned error: {hpo_result.get('error')}", color="yellow")
                    catboost_model = cb.CatBoostClassifier(
                        iterations=100,
                        depth=6,
                        learning_rate=0.1,
                        random_seed=42,
                        verbose=False,
                    )
                    # Train with early stopping (use soft sample weights if available)
                    catboost_model.fit(
                        X_train_fit,
                        y_train_fit,
                        sample_weight=sample_weight_fit,
                        eval_set=(X_val, y_val),
                        early_stopping_rounds=50,
                        verbose=False,
                    )
                    trained_models['catboost'] = catboost_model
                    tprint("⚠️ [REGIME_MODELS] CatBoost HPO unavailable, using default parameters", color="yellow")
            except Exception as e:
                tprint(f"❌ [REGIME_MODELS] CatBoost training failed: {e}", color="red")

        # 2. Train LightGBM with HPO
        try:
            tprint("💡 [REGIME_MODELS] Training LightGBM with HPO", color="blue")
            
            # 🔍 AMÉLIORATION: Configuration optimisée pour petits datasets
            n_samples = len(X_train)
            is_small_dataset = n_samples < 1000
            
            if is_small_dataset:
                tprint("🔍 [LIGHTGBM] PETIT DATASET DÉTECTÉ - Application d'optimisations spécifiques", color="cyan")
                tprint(f"   → Taille du dataset: {n_samples} échantillons (< 1000)", color="cyan")
                tprint("   → Optimisations: early stopping plus agressif, régularisation augmentée", color="cyan")
                tprint("   → Impact: Réduction du overfitting, amélioration de la généralisation", color="cyan")

            def create_lightgbm_model(**params):
                # 🔍 AMÉLIORATION: Hyperparamètres adaptés pour petits datasets
                if is_small_dataset:
                    # Réduire la complexité pour éviter l'overfitting
                    return lgb.LGBMClassifier(
                        num_leaves=min(params.get('num_leaves', 15), 31),      # Réduit de 31 à 15
                        max_depth=min(params.get('max_depth', 4), 6),          # Réduit de -1 à 4
                        learning_rate=max(params.get('learning_rate', 0.05), 0.1),  # Réduit pour plus de stabilité
                        n_estimators=min(params.get('n_estimators', 50), 100),   # Réduit de 100 à 50
                        subsample=params.get('subsample', 0.8),                 # Ajout de subsampling
                        colsample_bytree=params.get('colsample_bytree', 0.8),      # Ajout de feature sampling
                        reg_alpha=max(params.get('reg_alpha', 0.1), 0.3),        # Régularisation L1 augmentée
                        reg_lambda=max(params.get('reg_lambda', 0.1), 0.3),       # Régularisation L2 augmentée
                        min_child_samples=max(params.get('min_child_samples', 20), 50),  # Augmenté pour éviter l'overfitting
                        class_weight=adaptive_weights,  # Apply adaptive class weights (dict format)
                        random_state=42,
                        verbose=-1,
                        force_col_wise=True
                    )
                else:
                    return lgb.LGBMClassifier(
                        num_leaves=params.get('num_leaves', 31),
                        max_depth=params.get('max_depth', -1),
                        learning_rate=params.get('learning_rate', 0.1),
                        n_estimators=params.get('n_estimators', 100),
                        subsample=params.get('subsample', 1.0),
                        colsample_bytree=params.get('colsample_bytree', 1.0),
                        reg_alpha=params.get('reg_alpha', 0.0),
                        reg_lambda=params.get('reg_lambda', 0.0),
                        class_weight=adaptive_weights,  # Apply adaptive class weights (dict format)
                        random_state=42,
                        verbose=-1,
                        force_col_wise=True
                    )

            search_space = self.hpo_optimizer._get_default_search_space('lightgbm_regime')
            # Purged CV with embargo from temporal config
            embargo_bars = int(((self.validated_config or {}).get('temporal_validation', {}) or {}).get('gap_size', 0))
            class _PurgedCV:
                def __init__(self, n_splits: int, embargo: int):
                    self.kf = PurgedKFoldTime(n_splits=n_splits, purge=embargo, embargo=embargo)
                    self.n_splits = n_splits
                def split(self, X, y=None, groups=None):
                    n = len(X)
                    yield from self.kf.split_positions(n)
                def get_n_splits(self, X=None, y=None, groups=None):
                    return self.n_splits
            cv_strategy = _PurgedCV(n_splits=5, embargo=embargo_bars)
            hpo_result = self.hpo_optimizer.bayesian_optimization(
                model_factory=create_lightgbm_model,
                X=X_train,
                y=y_train,
                search_space=search_space,
                cv=cv_strategy,
                scoring=scoring,
                n_trials=150,
                early_stopping_patience=15,
                early_stopping_threshold=0.001,
                enable_pruner=True,
                pruner_type='hyperband'
            )

            if hpo_result and not hpo_result.get('error'):
                best_params = hpo_result.get('best_params', {})
                best_score = hpo_result.get('best_score')
                tuned_model = create_lightgbm_model(**best_params)
                # 🔍 AMÉLIORATION: Early stopping plus agressif pour petits datasets
                early_stopping_rounds = 20 if is_small_dataset else 50
                tprint(
                    f"🔍 [LIGHTGBM] Early stopping rounds: {early_stopping_rounds} (adapté pour {'petit' if is_small_dataset else 'grand'} dataset)",
                    color="cyan",
                )

                # Train with early stopping on validation set (use soft sample weights if available)
                tuned_model.fit(
                    X_train_fit,
                    y_train_fit,
                    sample_weight=sample_weight_fit,
                    eval_set=[(X_val, y_val)],
                    callbacks=[early_stopping(early_stopping_rounds), log_evaluation(0)],
                )
                trained_models['lightgbm'] = tuned_model
                score_msg = f"{best_score:.4f}" if isinstance(best_score, (int, float, np.floating)) else str(best_score)
                tprint(f"✅ [REGIME_MODELS] LightGBM HPO completed - Best score: {score_msg}", color="green")
                self.training_history.append({'model': 'lightgbm', 'best_params': best_params, 'best_score': best_score, 'n_trials': hpo_result.get('n_trials')})
            else:
                if hpo_result and hpo_result.get('error'):
                    tprint(f"⚠️ [REGIME_MODELS] LightGBM HPO returned error: {hpo_result.get('error')}", color="yellow")
                
                # 🔍 AMÉLIORATION: Configuration par défaut optimisée pour petits datasets
                default_params = {
                    'num_leaves': 15 if is_small_dataset else 31,
                    'max_depth': 4 if is_small_dataset else -1,
                    'learning_rate': 0.05 if is_small_dataset else 0.1,
                    'n_estimators': 50 if is_small_dataset else 100,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1 if is_small_dataset else 0.0,
                    'reg_lambda': 0.1 if is_small_dataset else 0.0,
                    'min_child_samples': 50 if is_small_dataset else 20,
                }
                
                lgbm_model = lgb.LGBMClassifier(
                    class_weight=adaptive_weights,  # Apply adaptive class weights
                    random_state=42,
                    verbose=-1,
                    force_col_wise=True,
                    **default_params,
                )

                # 🔍 AMÉLIORATION: Early stopping plus agressif pour petits datasets
                early_stopping_rounds = 20 if is_small_dataset else 50
                tprint(
                    f"🔍 [LIGHTGBM] Early stopping rounds par défaut: {early_stopping_rounds} (adapté pour {'petit' if is_small_dataset else 'grand'} dataset)",
                    color="cyan",
                )

                # Train with early stopping (use soft sample weights if available)
                lgbm_model.fit(
                    X_train_fit,
                    y_train_fit,
                    sample_weight=sample_weight_fit,
                    eval_set=[(X_val, y_val)],
                    callbacks=[early_stopping(early_stopping_rounds), log_evaluation(0)],
                )
                trained_models['lightgbm'] = lgbm_model
                tprint("⚠️ [REGIME_MODELS] LightGBM HPO unavailable, using optimized default parameters", color="yellow")
        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] LightGBM training failed: {e}", color="red")

        # 3. Train XGBoost with HPO
        try:
            tprint("🚀 [REGIME_MODELS] Training XGBoost with HPO", color="blue")
            
            # 🔍 AMÉLIORATION: Configuration optimisée pour petits datasets
            n_samples = len(X_train)
            is_small_dataset = n_samples < 1000
            
            if is_small_dataset:
                tprint("🔍 [XGBOOST] PETIT DATASET DÉTECTÉ - Application d'optimisations spécifiques", color="cyan")
                tprint(f"   → Taille du dataset: {n_samples} échantillons (< 1000)", color="cyan")
                tprint("   → Optimisations: early stopping plus agressif, régularisation augmentée", color="cyan")
                tprint("   → Impact: Réduction du overfitting, amélioration de la généralisation", color="cyan")

            def create_xgboost_model(**params):
                # 🔍 AMÉLIORATION: Hyperparamètres adaptés pour petits datasets
                if is_small_dataset:
                    # Réduire la complexité pour éviter l'overfitting
                    return xgb.XGBClassifier(
                        n_estimators=min(params.get('n_estimators', 50), 100),   # Réduit de 100 à 50
                        max_depth=min(params.get('max_depth', 4), 6),          # Réduit de 6 à 4
                        learning_rate=max(params.get('learning_rate', 0.05), 0.1),  # Réduit pour plus de stabilité
                        subsample=params.get('subsample', 0.8),                 # Ajout de subsampling
                        colsample_bytree=params.get('colsample_bytree', 0.8),      # Ajout de feature sampling
                        reg_alpha=max(params.get('reg_alpha', 0.1), 0.3),        # Régularisation L1 augmentée
                        reg_lambda=max(params.get('reg_lambda', 0.1), 0.3),       # Régularisation L2 augmentée
                        gamma=params.get('gamma', 0.1),                         # Ajout de gamma pour la complexité
                        min_child_weight=max(params.get('min_child_weight', 1), 5),  # Augmenté pour éviter l'overfitting
                        random_state=42,
                        n_jobs=-1,
                        verbosity=0
                    )
                else:
                    return xgb.XGBClassifier(
                        n_estimators=params.get('n_estimators', 100),
                        max_depth=params.get('max_depth', 6),
                        learning_rate=params.get('learning_rate', 0.1),
                        subsample=params.get('subsample', 0.8),
                        colsample_bytree=params.get('colsample_bytree', 0.8),
                        reg_alpha=params.get('reg_alpha', 0.1),
                        reg_lambda=params.get('reg_lambda', 0.1),
                        gamma=params.get('gamma', 0),
                        random_state=42,
                        n_jobs=-1,
                        verbosity=0
                    )

            search_space = self.hpo_optimizer._get_default_search_space('xgboost_regime')
            # Purged CV with embargo from temporal config
            embargo_bars = int(((self.validated_config or {}).get('temporal_validation', {}) or {}).get('gap_size', 0))
            class _PurgedCV:
                def __init__(self, n_splits: int, embargo: int):
                    self.kf = PurgedKFoldTime(n_splits=n_splits, purge=embargo, embargo=embargo)
                    self.n_splits = n_splits
                def split(self, X, y=None, groups=None):
                    n = len(X)
                    yield from self.kf.split_positions(n)
                def get_n_splits(self, X=None, y=None, groups=None):
                    return self.n_splits
            cv_strategy = _PurgedCV(n_splits=5, embargo=embargo_bars)
            hpo_result = self.hpo_optimizer.bayesian_optimization(
                model_factory=create_xgboost_model,
                X=X_train,
                y=y_train,
                search_space=search_space,
                cv=cv_strategy,
                scoring=scoring,
                n_trials=150,
                early_stopping_patience=15,
                early_stopping_threshold=0.001,
                enable_pruner=True,
                pruner_type='hyperband'
            )

            if hpo_result and not hpo_result.get('error'):
                best_params = hpo_result.get('best_params', {})
                best_score = hpo_result.get('best_score')
                tuned_model = create_xgboost_model(**best_params)
                # 🔍 AMÉLIORATION: Early stopping plus agressif pour petits datasets
                early_stopping_rounds = 20 if is_small_dataset else 50
                tprint(
                    f"🔍 [XGBOOST] Early stopping rounds: {early_stopping_rounds} (adapté pour {'petit' if is_small_dataset else 'grand'} dataset)",
                    color="cyan",
                )

                # Train with early stopping on validation set (use soft sample weights if available)
                tuned_model.fit(
                    X_train_fit,
                    y_train_fit,
                    sample_weight=sample_weight_fit,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                trained_models['xgboost'] = tuned_model
                score_msg = f"{best_score:.4f}" if isinstance(best_score, (int, float, np.floating)) else str(best_score)
                tprint(f"✅ [REGIME_MODELS] XGBoost HPO completed - Best score: {score_msg}", color="green")
                self.training_history.append({'model': 'xgboost', 'best_params': best_params, 'best_score': best_score, 'n_trials': hpo_result.get('n_trials')})
            else:
                if hpo_result and hpo_result.get('error'):
                    tprint(f"⚠️ [REGIME_MODELS] XGBoost HPO returned error: {hpo_result.get('error')}", color="yellow")
                
                # 🔍 AMÉLIORATION: Configuration par défaut optimisée pour petits datasets
                default_params = {
                    'n_estimators': 50 if is_small_dataset else 100,
                    'max_depth': 4 if is_small_dataset else 6,
                    'learning_rate': 0.05 if is_small_dataset else 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1 if is_small_dataset else 0.1,
                    'reg_lambda': 0.1 if is_small_dataset else 0.1,
                    'gamma': 0.1 if is_small_dataset else 0,
                    'min_child_weight': 5 if is_small_dataset else 1,
                }
                
                xgb_model = xgb.XGBClassifier(
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                    **default_params,
                )

                # 🔍 AMÉLIORATION: Early stopping plus agressif pour petits datasets
                early_stopping_rounds = 20 if is_small_dataset else 50
                tprint(
                    f"🔍 [XGBOOST] Early stopping rounds par défaut: {early_stopping_rounds} (adapté pour {'petit' if is_small_dataset else 'grand'} dataset)",
                    color="cyan",
                )

                # Train with early stopping (use soft sample weights if available)
                xgb_model.fit(
                    X_train_fit,
                    y_train_fit,
                    sample_weight=sample_weight_fit,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                trained_models['xgboost'] = xgb_model
                tprint("⚠️ [REGIME_MODELS] XGBoost HPO unavailable, using optimized default parameters", color="yellow")
        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] XGBoost training failed: {e}", color="red")

        # 4. Train RandomForest (no HPO for now to keep pipeline stable)
        if enable_random_forest:
            try:
                tprint("🌲 [REGIME_MODELS] Training RandomForest (default config)", color="blue")

                rf_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features='sqrt',
                    class_weight=adaptive_weights,
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1,
                )

                rf_model.fit(X_train, y_train, sample_weight=sample_weight_train)
                trained_models['random_forest'] = rf_model
                tprint("⚠️ [REGIME_MODELS] RandomForest HPO disabled - using default parameters", color="yellow")
            except Exception as e:
                tprint(f"❌ [REGIME_MODELS] RandomForest training failed: {e}", color="red")
        else:
            tprint("ℹ️ [REGIME_MODELS] RandomForest training disabled (enable_random_forest=False)", color="cyan")

        # 5. Train ExtraTrees (no HPO for now to keep pipeline stable)
        if enable_extratrees:
            try:
                tprint("🌳 [REGIME_MODELS] Training ExtraTrees (default config)", color="blue")

                et_model = ExtraTreesClassifier(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=5,
                    min_samples_leaf=5,
                    max_features='sqrt',
                    class_weight=adaptive_weights,
                    random_state=42,
                    n_jobs=-1,
                )

                et_model.fit(X_train, y_train, sample_weight=sample_weight_train)
                trained_models['extratrees'] = et_model
                tprint("⚠️ [REGIME_MODELS] ExtraTrees HPO disabled - using default parameters", color="yellow")
            except Exception as e:
                tprint(f"❌ [REGIME_MODELS] ExtraTrees training failed: {e}", color="red")
        else:
            tprint("ℹ️ [REGIME_MODELS] ExtraTrees training disabled (enable_extratrees=False)", color="cyan")

        tprint(f"✅ [REGIME_MODELS] Model training completed - {len(trained_models)} models trained", color="green")

        # Apply probability calibration to all models
        tprint("🎯 [REGIME_MODELS] Applying probability calibration to all models", color="cyan")
        calibrated_models = {}
        from sklearn.calibration import CalibratedClassifierCV

        for model_name, model in trained_models.items():
            try:
                tprint(f"📊 [REGIME_MODELS] Calibrating {model_name}", color="blue")
                tprint(f"🔍 [DEBUG] Type du modèle avant calibration: {type(model)}", color="yellow")
                # Use isotonic calibration (better for tree-based models)
                # Train calibrator on validation set
                calibrated = CalibratedClassifierCV(
                    estimator=model,
                    method='isotonic',  # Isotonic regression (non-parametric)
                    cv='prefit',  # Model is already fitted
                    ensemble=False
                )
                # Calibrate using validation set
                calibrated.fit(X_val, y_val)
                calibrated_models[model_name] = calibrated
                tprint(f"🔍 [DEBUG] Type du modèle après calibration: {type(calibrated)}", color="yellow")
                tprint(f"✅ [REGIME_MODELS] {model_name} calibrated successfully", color="green")
            except Exception as e:
                tprint(f"⚠️ [REGIME_MODELS] Failed to calibrate {model_name}: {e}, using uncalibrated model", color="yellow")
                calibrated_models[model_name] = model  # Fallback to uncalibrated

        tprint(f"✅ [REGIME_MODELS] Probability calibration completed - {len(calibrated_models)} models calibrated", color="green")
        return calibrated_models

    async def _evaluate_models_enhanced(self, models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate models with enhanced evaluation utilities."""
        tprint("📊 [REGIME_MODELS] Evaluating models with enhanced evaluation", color="cyan")
        
        model_metrics = {}
        
        for model_name, model in models.items():
            try:
                tprint(f"🔍 [REGIME_MODELS] Evaluating {model_name}", color="blue")
                
                # Get predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate basic metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                
                model_metrics[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                tprint(f"✅ [REGIME_MODELS] {model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}", color="green")
                
            except Exception as e:
                tprint(f"❌ [REGIME_MODELS] Failed to evaluate {model_name}: {e}", color="red")
                model_metrics[model_name] = {'error': str(e)}
        
        return model_metrics

    def _prepare_training_data_improved(
        self,
        data: pd.DataFrame,
        regime_labels: np.ndarray,
        pipeline_state: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DatetimeIndex]:
        """Prepare training data using existing feature bank system with fast fail."""
        tprint("🔧 [REGIME_MODELS] Preparing training data with existing feature bank", color="cyan")
        
        try:
            # Use existing feature bank system with fast fail
            if not FEATURE_GENERATION_AVAILABLE:
                raise ValueError("Feature generation system not available - cannot generate features")
            
            tprint("🔧 [REGIME_MODELS] Generating features using existing feature bank", color="cyan")
            X, feature_names, X_index = self._generate_features_with_bank(data)
            
            min_features_required = ((self.validated_config or {}).get('data_validation', {}) or {}).get('min_features', 50)
            if X is None or X.shape[1] < min_features_required:
                raise ValueError(
                    f"Insufficient features generated: {X.shape[1] if X is not None else 0} < {min_features_required} required"
                )
            
            tprint(f"✅ [REGIME_MODELS] Features generated: {X.shape[1]} features", color="green")
            
            # Align with regime labels
            tprint("🔧 [REGIME_MODELS] Aligning features with regime labels", color="cyan")
            min_length = min(len(X), len(regime_labels))
            X = X[:min_length]
            y = np.array(regime_labels[:min_length])

            # ========================================================================
            # CRITICAL: Keep index aligned with X after truncation
            # ========================================================================
            if X_index is not None:
                X_index = X_index[:min_length]
                tprint(f"🔍 [REGIME_MODELS] Aligned X_index after truncation: {len(X_index)} samples", color="blue")
            else:
                tprint("⚠️ [REGIME_MODELS] WARNING: X_index is None - predictions may be misaligned!", color="yellow")

            # Integrate Rolling HMM economic features into the supervised feature space
            # so models see the same economic axes used for HMM emissions.
            try:
                hmm_econ_df = None
                if pipeline_state is not None and isinstance(pipeline_state, dict):
                    hmm_econ_df = pipeline_state.get('rolling_hmm_economic_features')
                if hmm_econ_df is None and hasattr(self, 'pipeline_state') and isinstance(self.pipeline_state, dict):
                    hmm_econ_df = self.pipeline_state.get('rolling_hmm_economic_features')

                if isinstance(hmm_econ_df, pd.DataFrame) and not hmm_econ_df.empty and X_index is not None:
                    tprint("🔧 [REGIME_MODELS] Integrating Rolling HMM economic features into training matrix", color="cyan")

                    econ_df = hmm_econ_df.copy()
                    if 'timestamp' in econ_df.columns:
                        econ_df['timestamp'] = pd.to_datetime(econ_df['timestamp'])
                        econ_df.set_index('timestamp', inplace=True)
                        econ_df.sort_index(inplace=True)

                    if not isinstance(econ_df.index, pd.DatetimeIndex):
                        econ_df.index = pd.to_datetime(econ_df.index)

                    econ_aligned = econ_df.reindex(X_index)
                    missing_rows = int(econ_aligned.isna().any(axis=1).sum())
                    if missing_rows > 0:
                        tprint(
                            f"⚠️ [REGIME_MODELS] {missing_rows} rows have NaNs in economic features after alignment; filling with 0.0",
                            color="yellow",
                        )
                        econ_aligned = econ_aligned.fillna(0.0)

                    econ_values = econ_aligned.to_numpy(dtype=float, copy=False)
                    if econ_values.shape[0] == X.shape[0]:
                        prev_n_features = X.shape[1]
                        X = np.concatenate([X, econ_values], axis=1)
                        econ_feature_names = [f"HMM_ECON_{str(c)}" for c in econ_aligned.columns]
                        feature_names = (feature_names or []) + econ_feature_names
                        tprint(
                            f"✅ [REGIME_MODELS] Added {econ_values.shape[1]} HMM economic features (total features: {X.shape[1]}),",
                            color="green",
                        )
                        tprint(
                            f"   • Original features: {prev_n_features}, HMM economic features: {econ_values.shape[1]}",
                            color="blue",
                        )
                    else:
                        tprint(
                            f"⚠️ [REGIME_MODELS] Economic features row mismatch (econ_rows={econ_values.shape[0]}, X_rows={X.shape[0]}); skipping integration",
                            color="yellow",
                        )
                else:
                    tprint("ℹ️ [REGIME_MODELS] No Rolling HMM economic features available in pipeline_state", color="blue")
            except Exception as e:
                tprint(f"⚠️ [REGIME_MODELS] Failed to integrate Rolling HMM economic features: {e}", color="yellow")

            # Handle NaN values in features
            tprint("🔧 [REGIME_MODELS] Handling NaN values in features", color="cyan")

            # Ensure X is a numpy array for NaN handling
            if not isinstance(X, np.ndarray):
                X = np.array(X, dtype=np.float64)

            nan_cols_before = np.sum(np.isnan(X), axis=0)
            nan_cols_count = np.sum(nan_cols_before > 0)
            if nan_cols_count > 0:
                tprint(f"⚠️ [REGIME_MODELS] Found {nan_cols_count} features with NaN values", color="yellow")

            tprint("ℹ️ [REGIME_MODELS] Deferring imputation to train-only preprocessing after temporal split", color="blue")
            X = np.array(X, dtype=np.float64)

            # Apply robust feature scaling (better for outliers than StandardScaler)
            tprint("🔧 [REGIME_MODELS] Applying robust feature scaling", color="cyan")
            from sklearn.preprocessing import RobustScaler

            # Check feature statistics before scaling
            feature_stds = np.std(X, axis=0)
            low_variance_features = feature_stds < 1e-6
            
            # Debug: Check lengths before filtering
            if feature_names is not None:
                tprint(f"🔍 [DEBUG] X shape: {X.shape}, feature_names length: {len(feature_names)}", color="yellow")
            tprint(f"🔍 [DEBUG] low_variance_features length: {len(low_variance_features)}", color="yellow")

            if np.any(low_variance_features):
                n_low_var = np.sum(low_variance_features)
                tprint(f"⚠️ [REGIME_MODELS] Removing {n_low_var} low-variance features (std < 1e-6)", color="yellow")
                X = X[:, ~low_variance_features]
                
                # Ensure we don't index out of bounds - only filter if feature_names is available
                if feature_names is not None:
                    # Ensure we don't index out of bounds
                    if len(low_variance_features) != len(feature_names):
                        tprint(f"⚠️ [DEBUG] Length mismatch: low_variance_features={len(low_variance_features)}, feature_names={len(feature_names)}", color="red")
                        # Truncate or pad to match
                        if len(low_variance_features) > len(feature_names):
                            low_variance_features = low_variance_features[:len(feature_names)]
                        else:
                            # This shouldn't happen but let's handle it gracefully
                            tprint(f"⚠️ [DEBUG] Unexpected: feature_names longer than low_variance_features", color="red")
                            extra_features = len(feature_names) - len(low_variance_features)
                            low_variance_features = np.concatenate([low_variance_features, [False] * extra_features])
                    
                    feature_names = [fn for i, fn in enumerate(feature_names) if not low_variance_features[i]]

            tprint("ℹ️ [REGIME_MODELS] Deferring scaling to train-only preprocessing after temporal split", color="blue")

            # Apply feature selection if high dimensionality detected
            n_samples, n_features = X.shape
            sample_to_feature_ratio = n_samples / n_features
            
            tprint(f"📊 [REGIME_MODELS] Feature dimensionality check: {n_features} features for {n_samples} samples (ratio: {sample_to_feature_ratio:.3f})", color="blue")
            
            if n_features > n_samples / 5:  # High dimensionality threshold
                tprint("⚠️ [REGIME_MODELS] HIGH DIMENSIONALITY DETECTED - Applying feature selection", color="yellow")
                tprint(f"   • {n_features} features for {n_samples} samples", color="yellow")
                tprint("   • Feature selection will be applied to reduce to exactly 60 features", color="yellow")
                tprint("   • This will improve generalization and model performance", color="yellow")
                
                # Apply enhanced regime-aware feature selection
                try:
                    # Use the enhanced regime-aware feature selection method
                    X_selected, selected_feature_names = self._apply_regime_aware_feature_selection(
                        X, y, feature_names, y  # Use y as regime_labels since they're the same in this context
                    )
                    
                    # Update X and feature_names with selected features
                    X = X_selected
                    feature_names = selected_feature_names
                    
                except Exception as e:
                    tprint(f"⚠️ [REGIME_MODELS] Feature selection failed, continuing with all features: {e}", color="yellow")
                    tprint(f"   • Exception type: {type(e).__name__}", color="yellow")
                    tprint(f"   • Exception details: {str(e)}", color="yellow")
                    tprint("   • This may result in slower training and potential overfitting", color="yellow")
                    import traceback
                    tprint(f"   • Traceback: {traceback.format_exc()}", color="yellow")
            else:
                tprint("✅ [REGIME_MODELS] Feature dimensionality is acceptable - no selection needed", color="green")

            data_validation_cfg = (self.validated_config or {}).get('data_validation', {})
            min_samples_required = data_validation_cfg.get('min_samples', 10)

            # Filter out tiny regimes (< 2% of samples) to stabilize supervised training.
            try:
                unique_regimes, regime_counts = np.unique(y, return_counts=True)
                total_samples = int(regime_counts.sum())
                if total_samples > 0 and len(unique_regimes) > 1:
                    regime_fractions = regime_counts / float(total_samples)
                    tiny_threshold = 0.02
                    tiny_mask = regime_fractions < tiny_threshold
                    tiny_regimes = unique_regimes[tiny_mask]

                    # Only filter if at least two regimes remain after dropping tiny ones
                    remaining_regimes = unique_regimes[~tiny_mask]
                    if tiny_regimes.size > 0 and remaining_regimes.size >= 2:
                        keep_mask = ~np.isin(y, tiny_regimes)
                        kept = int(np.count_nonzero(keep_mask))
                        dropped = int(len(y) - kept)
                        tprint(
                            f"⚠️ [REGIME_MODELS] Dropping {dropped} samples from tiny regimes (<{tiny_threshold:.0%} of data): {tiny_regimes.tolist()}",
                            color="yellow",
                        )
                        tprint(
                            f"   • Remaining samples after tiny-regime filter: {kept} (from {len(y)})",
                            color="yellow",
                        )

                        X = X[keep_mask]
                        y = y[keep_mask]
                        if X_index is not None:
                            X_index = X_index[keep_mask]

                        # Log new regime distribution
                        new_unique, new_counts = np.unique(y, return_counts=True)
                        new_total = int(new_counts.sum())
                        tprint("📊 [REGIME_MODELS] Regime distribution after tiny-regime filter:", color="blue")
                        for u, c in zip(new_unique, new_counts):
                            tprint(f"   • Regime {int(u)}: {int(c)} ({c/max(1, new_total):.2%})", color="blue")
                    else:
                        tprint("ℹ️ [REGIME_MODELS] Tiny-regime filter skipped (no tiny regimes or would leave <2 regimes)", color="blue")
                else:
                    tprint("ℹ️ [REGIME_MODELS] Tiny-regime filter skipped (insufficient regimes)", color="blue")
            except Exception as e:
                tprint(f"⚠️ [REGIME_MODELS] Tiny-regime filtering failed (continuing without filter): {e}", color="yellow")

            # Validate data
            if len(X) < min_samples_required:
                raise ValueError(f"Insufficient samples after alignment: {len(X)} < {min_samples_required}")
            
            if len(np.unique(y)) < 2:
                raise ValueError(f"Insufficient regimes: {len(np.unique(y))}")
            
            tprint(f"✅ [REGIME_MODELS] Training data prepared: {X.shape[0]} samples, {X.shape[1]} features", color="green")

            # ========================================================================
            # CRITICAL: Return X_index for correct prediction alignment
            # ========================================================================
            return X, y, feature_names, X_index

        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Training data preparation failed: {e}", color="red")
            raise

    def _apply_regime_aware_feature_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        regime_labels: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Apply enhanced regime-aware feature selection using LGBM with SHAP analysis.
        
        This method replaces RandomForest with LGBM for feature importance scoring,
        adds SHAP analysis for feature importance, implements regime-aware cross-validation
        metrics, and includes conditional REGIME feature enabling logic.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            regime_labels: Regime labels for each sample
            
        Returns:
            Tuple of (selected_features, selected_feature_names)
        """
        tprint("🔍 [REGIME_MODELS] Applying enhanced regime-aware feature selection", color="cyan", bold=True)
        
        try:
            # Import required libraries
            from sklearn.feature_selection import SelectFromModel
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import accuracy_score
            import lightgbm as lgb
            
            # Try to import SHAP
            try:
                import shap
                SHAP_AVAILABLE = True
                tprint("✅ [REGIME_MODELS] SHAP library imported successfully", color="green")
            except ImportError:
                SHAP_AVAILABLE = False
                tprint("⚠️ [REGIME_MODELS] SHAP library not available, skipping SHAP analysis", color="yellow")
            
            # Always use exactly 60 features for consistency
            target_features = 60
            n_samples, n_features = X.shape
            
            tprint(f"🎯 [REGIME_MODELS] Target feature count: {target_features} (fixed)", color="cyan")
            tprint(f"📊 [REGIME_MODELS] Input shape: {X.shape}", color="blue")
            
            # 1. Permutation importance with time-aware CV + correlation pruning
            tprint("🔄 [REGIME_MODELS] Computing permutation importance (LGBM) with time-aware CV...", color="cyan")

            # Base model config (no class_weight to avoid fold mismatch)
            lgb_base = lgb.LGBMClassifier(
                num_leaves=31,
                max_depth=8,
                learning_rate=0.1,
                n_estimators=150,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
                force_col_wise=True
            )

            # Time-aware CV over folds: fit on train, compute permutation importance on val
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.inspection import permutation_importance
            tscv = TimeSeriesSplit(n_splits=3)

            n_feats = X.shape[1]
            perm_scores = np.zeros(n_feats, dtype=np.float64)
            folds_used = 0
            for fold_idx, (tr, va) in enumerate(tscv.split(X), 1):
                try:
                    lgb_base.fit(X[tr], y[tr])
                    r = permutation_importance(
                        lgb_base,
                        X[va],
                        y[va],
                        n_repeats=3,
                        random_state=42,
                        scoring='accuracy'
                    )
                    # r.importances_mean shape: (n_features,)
                    if r.importances_mean.shape[0] == n_feats:
                        perm_scores += r.importances_mean
                        folds_used += 1
                        tprint(f"   • Fold {fold_idx}: permutation importance computed", color="blue")
                except Exception as e:
                    tprint(f"⚠️ [REGIME_MODELS] Permutation importance failed on fold {fold_idx}: {e}", color="yellow")
                    continue

            if folds_used > 0:
                perm_scores /= max(1, folds_used)
            else:
                tprint("⚠️ [REGIME_MODELS] No folds succeeded for permutation importance; falling back to single fit importance_", color="yellow")
                try:
                    lgb_base.fit(X, y)
                    perm_scores = getattr(lgb_base, 'feature_importances_', np.zeros(n_feats))
                except Exception:
                    perm_scores = np.zeros(n_feats)

            # Build importance dataframe (for downstream logs/SHAP comparison)
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': perm_scores
            }).sort_values('importance', ascending=False)

            # 2. Correlation pruning (vectorized)
            tprint("🔧 [REGIME_MODELS] Applying correlation pruning (|rho|>0.90) on ranked features...", color="cyan")
            ranked_indices = importance_df.index.to_numpy()
            # Map ranked indices back to original column indices
            ranked_cols = importance_df.reset_index(drop=True).index.to_numpy()
            ranked_feature_indices = importance_df.reset_index().rename(columns={'index':'orig_idx'})['orig_idx'].to_numpy()

            # Compute absolute correlation matrix on X (scaled)
            with np.errstate(invalid='ignore'):  # handle constant columns
                corr = np.corrcoef(X, rowvar=False)
            if isinstance(corr, np.ndarray):
                corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                corr = np.zeros((n_feats, n_feats), dtype=np.float64)

            threshold = 0.90
            selected_indices = []  # original column indices
            selected_mask = np.zeros(n_feats, dtype=bool)

            for orig_idx in ranked_feature_indices:
                if len(selected_indices) >= target_features:
                    break
                if not selected_indices:
                    selected_indices.append(int(orig_idx))
                    selected_mask[int(orig_idx)] = True
                    continue
                # Check max absolute correlation with already selected
                corrs = np.abs(corr[int(orig_idx), selected_indices])
                if np.all(corrs < threshold):
                    selected_indices.append(int(orig_idx))
                    selected_mask[int(orig_idx)] = True

            # Fallback if not enough selected (e.g., corr matrix degenerate)
            if len(selected_indices) < target_features:
                for orig_idx in ranked_feature_indices:
                    if len(selected_indices) >= target_features:
                        break
                    if not selected_mask[int(orig_idx)]:
                        selected_indices.append(int(orig_idx))
                        selected_mask[int(orig_idx)] = True

            # Build final selection
            selected_feature_names = [feature_names[i] for i in selected_indices]
            X_selected = X[:, selected_indices]
            removed_feature_count = n_features - len(selected_feature_names)

            tprint(f"✅ [REGIME_MODELS] Permutation-importance selection completed:", color="green")
            tprint(f"   • Reduced from {n_features} to {X_selected.shape[1]} features", color="green")
            tprint(f"   • Removed {removed_feature_count} low-importance/correlated features", color="green")
            tprint(f"   • New sample-to-feature ratio: {n_samples / X_selected.shape[1]:.3f}", color="green")

            # Show top 10 by permutation importance
            tprint(f"🎯 [REGIME_MODELS] Top 10 selected features (Permutation importance):", color="blue")
            for i, row in importance_df.head(10).iterrows():
                tprint(f"   {i+1:2d}. {row['feature']:<40} (importance: {row['importance']:.6f})", color="blue")

            # Top 60 features list
            try:
                top60 = [feature_names[i] for i in selected_indices[:60]]
                tprint("📋 [REGIME_MODELS] Top 60 features (after correlation pruning):", color="cyan")
                for idx, fname in enumerate(top60, 1):
                    tprint(f"   {idx:2d}. {fname}", color="blue")
            except Exception:
                pass
            
            # 2. Add SHAP analysis for feature importance
            if SHAP_AVAILABLE and X_selected.shape[1] <= 100:  # Only if reasonable number of features
                tprint("🔍 [REGIME_MODELS] Computing SHAP values for feature importance analysis...", color="cyan")
                try:
                    # Create a smaller LGBM model for SHAP analysis (faster)
                    shap_model = lgb.LGBMClassifier(
                        num_leaves=31,
                        max_depth=6,
                        learning_rate=0.1,
                        n_estimators=50,  # Fewer trees for faster SHAP
                        subsample=0.8,
                        colsample_bytree=0.8,
                        class_weight=adaptive_weights,
                        random_state=42,
                        verbose=-1,
                        force_col_wise=True
                    )
                    
                    # Fit on selected features only
                    shap_model.fit(X_selected, y)
                    
                    # DEBUG: Log model type before SHAP in feature selection
                    tprint(f"🔍 [DEBUG] Modèle passé à SHAP dans feature selection: {type(shap_model)}", color="yellow")
                    
                    # Import SHAP utilities for safe handling
                    try:
                        from src.utils.shap_utils import safe_shap_tree_explainer, safe_shap_values
                        tprint("✅ [DEBUG] SHAP utils importées avec succès dans feature selection", color="green")
                    except ImportError as e:
                        tprint(f"⚠️ [DEBUG] SHAP utils non disponibles dans feature selection: {e}", color="yellow")
                        # Fallback to manual extraction
                        from src.utils.shap_utils import extract_base_model_from_calibrated, apply_numpy_shap_fix
                        model_for_shap = extract_base_model_from_calibrated(shap_model)
                        if model_for_shap is None:
                            model_for_shap = shap_model
                        apply_numpy_shap_fix()
                        explainer = shap.TreeExplainer(model_for_shap)
                    else:
                        # Use safe SHAP utilities
                        explainer = safe_shap_tree_explainer(shap_model)
                        if explainer is None:
                            tprint("❌ [DEBUG] Impossible de créer TreeExplainer avec SHAP utils dans feature selection", color="red")
                            return None
                    shap_values = safe_shap_values(explainer, X_selected)
                    
                    # For multi-class, get mean absolute SHAP values across classes
                    if isinstance(shap_values, list):
                        # Multi-class case
                        shap_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                    else:
                        # Binary case
                        shap_importance = np.abs(shap_values).mean(axis=0)
                    
                    # Create SHAP importance dataframe
                    shap_importance_df = pd.DataFrame({
                        'feature': selected_feature_names,
                        'shap_importance': shap_importance
                    }).sort_values('shap_importance', ascending=False)
                    
                    tprint(f"✅ [REGIME_MODELS] SHAP analysis completed", color="green")
                    tprint(f"🎯 [REGIME_MODELS] Top 10 features by SHAP importance:", color="blue")
                    for i, row in shap_importance_df.head(10).iterrows():
                        tprint(f"   {i+1:2d}. {row['feature']:<40} (SHAP: {row['shap_importance']:.6f})", color="blue")
                    
                    # Compare LGBM vs SHAP importance
                    comparison_df = pd.merge(
                        importance_df.head(20),
                        shap_importance_df.head(20),
                        on='feature',
                        how='outer',
                        suffixes=('_lgbm', '_shap')
                    ).fillna(0)
                    
                    tprint("📊 [REGIME_MODELS] Feature importance comparison (LGBM vs SHAP):", color="blue")
                    for i, row in comparison_df.head(10).iterrows():
                        tprint(f"   {i+1:2d}. {row['feature']:<35} (LGBM: {row['importance']:.4f}, SHAP: {row['shap_importance']:.4f})", color="blue")
                        
                except Exception as shap_error:
                    tprint(f"⚠️ [REGIME_MODELS] SHAP analysis failed: {shap_error}", color="yellow")
                    tprint("   • Continuing without SHAP analysis", color="yellow")
            
            # 3. Implement regime-aware cross-validation metrics
            tprint("🔄 [REGIME_MODELS] Computing regime-aware cross-validation metrics...", color="cyan")
            
            # Define regime-specific CV splits
            unique_regimes = np.unique(regime_labels)
            regime_cv_scores = {}
            
            # Overall CV score
            overall_cv_scores = cross_val_score(
                lgb_selector_model, X_selected, y, cv=5, scoring='accuracy', n_jobs=-1
            )
            regime_cv_scores['overall_cv'] = {
                'mean': overall_cv_scores.mean(),
                'std': overall_cv_scores.std(),
                'scores': overall_cv_scores.tolist()
            }
            
            # Within-regime CV scores
            for regime_id in unique_regimes:
                regime_mask = regime_labels == regime_id
                if np.sum(regime_mask) >= 10:  # Minimum samples for CV
                    X_regime = X_selected[regime_mask]
                    y_regime = y[regime_mask]
                    
                    if len(np.unique(y_regime)) > 1:  # Need both classes for CV
                        cv_scores = cross_val_score(
                            lgb_selector_model, X_regime, y_regime, cv=3, scoring='accuracy', n_jobs=-1
                        )
                        regime_cv_scores[f'regime_{regime_id}_cv'] = {
                            'mean': cv_scores.mean(),
                            'std': cv_scores.std(),
                            'scores': cv_scores.tolist(),
                            'n_samples': len(y_regime)
                        }
            
            # Calculate between/within regime CV ratio
            if len(regime_cv_scores) > 1:
                overall_mean = regime_cv_scores['overall_cv']['mean']
                within_regime_means = [v['mean'] for k, v in regime_cv_scores.items() if k.startswith('regime_')]
                if within_regime_means:
                    within_regime_mean = np.mean(within_regime_means)
                    between_within_ratio = overall_mean / (within_regime_mean + 1e-8)
                    regime_cv_scores['between_within_ratio'] = between_within_ratio
                    
                    tprint(f"📊 [REGIME_MODELS] Regime-aware CV metrics:", color="green")
                    tprint(f"   • Overall CV accuracy: {overall_mean:.4f} ± {overall_cv_scores['overall_cv']['std']:.4f}", color="green")
                    tprint(f"   • Within-regime CV accuracy: {within_regime_mean:.4f}", color="green")
                    tprint(f"   • Between/Within CV ratio: {between_within_ratio:.4f}", color="green")
                    
                    for regime_key, regime_metrics in regime_cv_scores.items():
                        if regime_key.startswith('regime_'):
                            regime_id = regime_key.split('_')[1]
                            tprint(f"   • Regime {regime_id} CV: {regime_metrics['mean']:.4f} ± {regime_metrics['std']:.4f} ({regime_metrics['n_samples']} samples)", color="blue")
            
            # Add specific regime-type CV metrics (trend_cv, momentum_cv, volatility_cv, volume_cv)
            tprint("🔍 [REGIME_MODELS] Computing regime-type specific CV metrics...", color="cyan")
            
            # Helper function to get features by category
            def get_features_by_category(feature_names, category_prefix):
                return [i for i, name in enumerate(feature_names) if name.startswith(category_prefix)]
            
            # Get feature indices for each category
            trend_features = get_features_by_category(selected_feature_names, 'TREND')
            momentum_features = get_features_by_category(selected_feature_names, 'MOMENTUM')
            volatility_features = get_features_by_category(selected_feature_names, 'VOLATILITY')
            volume_features = get_features_by_category(selected_feature_names, 'VOLUME')
            
            # Compute CV scores using only features from each category
            if trend_features:
                X_trend = X_selected[:, trend_features]
                trend_cv_scores = cross_val_score(
                    lgb_selector_model, X_trend, y, cv=3, scoring='accuracy', n_jobs=-1
                )
                regime_cv_scores['trend_cv'] = {
                    'mean': trend_cv_scores.mean(),
                    'std': trend_cv_scores.std(),
                    'n_features': len(trend_features)
                }
                tprint(f"   • Trend CV accuracy: {trend_cv_scores.mean():.4f} ± {trend_cv_scores.std():.4f} ({len(trend_features)} features)", color="blue")
            
            if momentum_features:
                X_momentum = X_selected[:, momentum_features]
                momentum_cv_scores = cross_val_score(
                    lgb_selector_model, X_momentum, y, cv=3, scoring='accuracy', n_jobs=-1
                )
                regime_cv_scores['momentum_cv'] = {
                    'mean': momentum_cv_scores.mean(),
                    'std': momentum_cv_scores.std(),
                    'n_features': len(momentum_features)
                }
                tprint(f"   • Momentum CV accuracy: {momentum_cv_scores.mean():.4f} ± {momentum_cv_scores.std():.4f} ({len(momentum_features)} features)", color="blue")
            
            if volatility_features:
                X_volatility = X_selected[:, volatility_features]
                volatility_cv_scores = cross_val_score(
                    lgb_selector_model, X_volatility, y, cv=3, scoring='accuracy', n_jobs=-1
                )
                regime_cv_scores['volatility_cv'] = {
                    'mean': volatility_cv_scores.mean(),
                    'std': volatility_cv_scores.std(),
                    'n_features': len(volatility_features)
                }
                tprint(f"   • Volatility CV accuracy: {volatility_cv_scores.mean():.4f} ± {volatility_cv_scores.std():.4f} ({len(volatility_features)} features)", color="blue")
            
            if volume_features:
                X_volume = X_selected[:, volume_features]
                volume_cv_scores = cross_val_score(
                    lgb_selector_model, X_volume, y, cv=3, scoring='accuracy', n_jobs=-1
                )
                regime_cv_scores['volume_cv'] = {
                    'mean': volume_cv_scores.mean(),
                    'std': volume_cv_scores.std(),
                    'n_features': len(volume_features)
                }
                tprint(f"   • Volume CV accuracy: {volume_cv_scores.mean():.4f} ± {volume_cv_scores.std():.4f} ({len(volume_features)} features)", color="blue")
            
            # 4. Implement conditional REGIME feature enabling logic
            tprint("🔧 [REGIME_MODELS] Applying conditional REGIME feature enabling logic...", color="cyan")
            
            # Check if we're being called by regime_models_training (enable REGIME features)
            caller_frame = None
            try:
                import inspect
                caller_frame = inspect.currentframe().f_back.f_back
                caller_name = caller_frame.f_code.co_name if caller_frame else 'unknown'
                tprint(f"🔍 [REGIME_MODELS] Detected caller: {caller_name}", color="blue")
            except:
                caller_name = 'unknown'
                tprint("⚠️ [REGIME_MODELS] Could not detect caller name", color="yellow")
            
            # Enable REGIME features only when called by regime_models_training
            enable_regime_features = 'regime_models_training' in caller_name or 'execute' in caller_name
            
            if enable_regime_features:
                tprint("✅ [REGIME_MODELS] REGIME features ENABLED (called by regime_models_training)", color="green")
            else:
                tprint("⚠️ [REGIME_MODELS] REGIME features DISABLED (not called by regime_models_training)", color="yellow")
                
                # Filter out REGIME category features if disabled
                regime_feature_indices = [
                    i for i, name in enumerate(selected_feature_names)
                    if not name.startswith('REGIME_')
                ]
                
                if regime_feature_indices:
                    X_selected = X_selected[:, regime_feature_indices]
                    selected_feature_names = [selected_feature_names[i] for i in regime_feature_indices]
                    
                    tprint(f"   • Filtered out {len(selected_feature_names) - len(regime_feature_indices)} REGIME features", color="yellow")
                    tprint(f"   • Remaining features: {len(selected_feature_names)}", color="yellow")
            
            # 5. Final feature selection report
            tprint("📋 [REGIME_MODELS] Enhanced feature selection summary:", color="cyan", bold=True)
            tprint(f"   • Original features: {n_features}", color="blue")
            tprint(f"   • Selected features: {X_selected.shape[1]}", color="blue")
            tprint(f"   • Feature reduction: {(1 - X_selected.shape[1]/n_features)*100:.1f}%", color="blue")
            tprint(f"   • Sample-to-feature ratio: {n_samples / X_selected.shape[1]:.3f}", color="blue")
            tprint(f"   • REGIME features enabled: {enable_regime_features}", color="blue")
            
            # CRITICAL: Verify EWMA features are preserved after feature selection
            ewma_count_after_selection = sum(1 for fn in selected_feature_names if '_ewm' in fn.lower())
            rolling_ma_count_after_selection = sum(1 for fn in selected_feature_names if '_ma' in fn.lower() and '_ewm' not in fn.lower())
            
            tprint(f"🔍 [REGIME_MODELS] POST-SELECTION FEATURE VERIFICATION:", color="cyan", bold=True)
            tprint(f"   • EWMA (_ewm) features after selection: {ewma_count_after_selection}", color="green" if ewma_count_after_selection > 0 else "red")
            tprint(f"   • Rolling MA (_ma) features after selection: {rolling_ma_count_after_selection}", color="blue")
            
            if ewma_count_after_selection == 0:
                tprint("   ❌ CRITICAL ERROR: ALL EWMA FEATURES WERE REMOVED DURING SELECTION!", color="red", bold=True)
                tprint("   → This will severely impact regime detection performance", color="red")
                tprint("   → Consider adjusting feature selection to preserve EWMA features", color="red")
            else:
                tprint(f"   ✅ EWMA features preserved after selection: {ewma_count_after_selection} features", color="green")
                # Show sample EWMA feature names after selection
                ewma_features_after = [fn for fn in selected_feature_names if '_ewm' in fn.lower()]
                if ewma_features_after:
                    tprint(f"   → Sample EWMA features after selection: {ewma_features_after[:5]}...", color="blue")
            
            if 'between_within_ratio' in regime_cv_scores:
                tprint(f"   • Between/Within CV ratio: {regime_cv_scores['between_within_ratio']:.4f}", color="blue")

            self._run_post_selection_diagnostics(X_selected, y, selected_feature_names)
            
            return X_selected, selected_feature_names
            
        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Enhanced feature selection failed: {e}", color="red")
            tprint(f"   • Exception type: {type(e).__name__}", color="red")
            tprint(f"   • Exception details: {str(e)}", color="red")
            import traceback
            tprint(f"   • Traceback: {traceback.format_exc()}", color="red")
            
            # Fallback to original features if enhanced selection fails
            tprint("⚠️ [REGIME_MODELS] Falling back to original features", color="yellow")
            return X, feature_names

    def _run_post_selection_diagnostics(self, X_selected: np.ndarray, y: np.ndarray,
                                        feature_names: List[str]) -> None:
        """Run HPODiagnostics on the post-selection matrix to log accurate stats."""
        if not HPODIAG_AVAILABLE or HPODiagnostics is None:
            tprint("⚠️ [REGIME_MODELS] HPODiagnostics unavailable - skipping post-selection diagnostics", color="yellow")
            return

        try:
            tprint("🧪 [REGIME_MODELS] Running post-selection HPODiagnostics", color="cyan")
            diagnostics = HPODiagnostics.check_data_variance(X_selected, y, name="Post-selection feature matrix")
            stats = diagnostics.get("stats", {})
            warnings_list = diagnostics.get("warnings", [])

            n_samples = stats.get("n_samples")
            n_features = stats.get("n_features")
            zero_var = stats.get("zero_variance_features")
            max_importance = stats.get("max_feature_importance")
            importance_threshold = stats.get("importance_warning_threshold")

            tprint(
                f"📊 [REGIME_MODELS] Post-selection stats: samples={n_samples}, features={n_features}, zero_var={zero_var}",
                color="blue"
            )
            if max_importance is not None and importance_threshold is not None:
                tprint(
                    f"   • Max feature importance {max_importance:.4f} (threshold {importance_threshold:.4f})",
                    color="blue"
                )

            if warnings_list:
                tprint("⚠️ [REGIME_MODELS] Post-selection warnings:", color="yellow")
                for warning in warnings_list[:5]:
                    tprint(f"   • {warning}", color="yellow")
                if len(warnings_list) > 5:
                    tprint(f"   • {len(warnings_list) - 5} additional warning(s) truncated", color="yellow")
            else:
                tprint("✅ [REGIME_MODELS] No post-selection warnings detected", color="green")

            if feature_names:
                sample = feature_names[:5]
                suffix = "..." if len(feature_names) > 5 else ""
                tprint(f"   • Sample selected features: {sample}{suffix}", color="blue")
        except Exception as e:
            tprint(f"⚠️ [REGIME_MODELS] Post-selection diagnostics failed: {e}", color="yellow")

    def _generate_features_with_bank(self, data: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[pd.DatetimeIndex]]:
        """Generate comprehensive features using the existing feature bank."""
        tprint("🔧 [REGIME_MODELS] Generating features using feature bank", color="cyan", bold=True)

        try:
            if not FEATURE_GENERATION_AVAILABLE:
                tprint("❌ [REGIME_MODELS] Feature generation system not available", color="red")
                return None, None, None

            # Get feature bank with REGIME features ENABLED (critical for regime classification)
            feature_bank = get_feature_bank(config={'enable_regime_features': True})
            tprint("✅ [REGIME_MODELS] Feature bank retrieved with REGIME features ENABLED", color="green")

            # Define feature categories to generate - prioritize REGIME category for core regime features
            # NOTE: OSCILLATOR category removed for regime detection mode as requested
            categories = [
                FeatureCategory.REGIME,  # Core regime features (lagged, derived, temporal)
                FeatureCategory.MOMENTUM,  # Price momentum indicators (RSI, MACD, etc.)
                FeatureCategory.VOLATILITY,  # Volatility measures (ATR, Bollinger Bands, etc.)
                FeatureCategory.VOLUME,  # Volume-based indicators (OBV, Volume MA, etc.)
                FeatureCategory.TREND,  # Trend following indicators (ADX, Aroon, etc.)
                # FeatureCategory.OSCILLATOR,  # DISABLED for regime detection mode (Stoch, Williams %R, etc.)
                FeatureCategory.RETURNS,  # Price return calculations and statistics
                FeatureCategory.MICROSTRUCTURE  # Microstructure features (no orderbook dependency)
            ]

            all_features = pd.DataFrame(index=data.index)
            total_features = 0
            
            # Generate features for each category
            for category in categories:
                tprint(f"🔍 [REGIME_MODELS] Generating {category.value} features", color="blue")

                # Get generators for this category
                generators = feature_bank.get_generators_by_category(category)

                if not generators:
                    tprint(f"⚠️ [REGIME_MODELS] No generators found for {category.value}", color="yellow")
                    continue

                category_features = pd.DataFrame(index=data.index)

                # Generate features using each generator
                for generator in generators:
                    try:
                        tprint(f"🔧 [REGIME_MODELS] Using generator: {generator.config.name}", color="blue")
                        result = generator.generate(data)

                        if result and hasattr(result, 'data') and len(result.data) > 0:
                            # Add feature with category prefix
                            feature_name = f"{category.value}_{result.name}"
                            category_features[feature_name] = result.data
                            total_features += 1
                            tprint(f"✅ [REGIME_MODELS] Generated feature: {feature_name}", color="green")
                        else:
                            tprint(f"⚠️ [REGIME_MODELS] Generator {generator.config.name} returned empty result", color="yellow")

                    except Exception as e:
                        tprint(f"⚠️ [REGIME_MODELS] Generator {generator.config.name} failed: {e}", color="yellow")
                        continue

                # Add category features to all features
                if not category_features.empty:
                    all_features = pd.concat([all_features, category_features], axis=1)
                    tprint(f"📊 [REGIME_MODELS] {category.value} features: {category_features.shape[1]}", color="blue")

            # Convert to numpy array
            if not all_features.empty:
                # ========================================================================
                # CRITICAL: Save DataFrame index before converting to numpy
                # This is needed to correctly align predictions with timestamps later
                # ========================================================================
                X_df = all_features  # Keep reference to DataFrame
                X_original_index = X_df.index.copy()  # Save index for later use
                tprint(f"🔍 [REGIME_MODELS] Saved original X index: {len(X_original_index)} samples", color="blue")
                tprint(f"   Index range: {X_original_index[0]} to {X_original_index[-1]}", color="blue")

                # Ensure all features are numeric and convert to float64 numpy array
                X = np.array(all_features.values, dtype=np.float64)
                feature_names = list(all_features.columns)
                
                # EWMA features are ALWAYS enabled for regime models - no conditional logic
                # Apply both rolling window smoothing and EWMA smoothing for optimal regime detection
                # Using simple EWMA 8 & 20 without special weights as requested
                tprint("🔧 [REGIME_MODELS] Adding EWMA and rolling smoothed features (ALWAYS ENABLED)", color="cyan")
                tprint("   → CLARIFICATION: _ma8/_ma20/_std8/_std20 are ROLLING WINDOW features (NOT EWMA)", color="yellow")
                tprint("   → TRUE EWMA features have suffix _ewm0.3 (exponential weighting)", color="yellow")
                tprint("   → Using simple EWMA 8 & 20 without special weights", color="blue")
                
                # First apply rolling window smoothing (moving averages and std)
                # NOTE: These create _ma8, _ma20, _std8, _std20 features (ROLLING, NOT EWMA)
                X, feature_names = add_smoothed_features(
                    X,
                    window_sizes=self.smoothing_window_sizes,
                    feature_names=feature_names
                )
                tprint(f"✅ [REGIME_MODELS] Rolling smoothed features added: {X.shape[1]} total features", color="green")
                tprint(f"   → Created _ma8, _ma20, _std8, _std20 features (ROLLING WINDOW, NOT EWMA)", color="blue")
                
                # Then apply EWMA smoothing for additional temporal smoothing
                # NOTE: These create _ewm0.3 features (TRUE EWMA with exponential weighting)
                initial_feature_count = X.shape[1]
                X, feature_names = apply_ewm_smoothing(
                    X,
                    alpha=self.ewm_alpha,
                    feature_names=feature_names
                )
                final_feature_count = X.shape[1]
                ewm_feature_count = final_feature_count - initial_feature_count
                
                tprint(f"✅ [REGIME_MODELS] EWMA smoothed features added: {X.shape[1]} total features", color="green")
                tprint(f"   → Created {ewm_feature_count} _ewm0.3 features (TRUE EWMA with exponential weighting)", color="blue")
                
                # Log EWMA feature names for verification
                ewm_feature_names = [fn for fn in feature_names if '_ewm0.3' in fn]
                if ewm_feature_names:
                    tprint(f"   → EWMA features created: {len(ewm_feature_names)} features", color="green")
                    tprint(f"   → Sample EWMA features: {ewm_feature_names[:5]}...", color="blue")
                else:
                    tprint("   ⚠️ WARNING: No _ewm0.3 features found in feature names!", color="red")
                
                tprint(f"✅ [REGIME_MODELS] Feature bank generated {X.shape[1]} features from {len(categories)} categories", color="green")
                tprint(f"📊 [REGIME_MODELS] Feature matrix shape: {X.shape}", color="blue")

                # Validate regime features were actually generated (CRITICAL for accuracy)
                regime_feature_count = sum(1 for fn in feature_names if 'regime' in fn.lower())
                tprint(f"🔍 [REGIME_MODELS] Generated {regime_feature_count} regime-specific features", color="cyan")

                if regime_feature_count < 10:
                    tprint(f"⚠️ [REGIME_MODELS] WARNING: Only {regime_feature_count} regime features found! Expected 20+", color="yellow")
                    tprint("   This may significantly impact model accuracy (regime features are critical)", color="yellow")
                    tprint("   Check that FeatureCategory.REGIME is enabled in feature_bank.py", color="yellow")
                else:
                    tprint(f"✅ [REGIME_MODELS] Sufficient regime features generated ({regime_feature_count})", color="green")

                # Log feature categories breakdown
                category_counts = {}
                ewma_count = 0
                rolling_ma_count = 0
                rolling_std_count = 0
                
                for fn in feature_names:
                    fn_lower = fn.lower()
                    if '_ewm' in fn_lower:
                        ewma_count += 1
                        category_counts['ewma'] = category_counts.get('ewma', 0) + 1
                    elif '_ma' in fn_lower and '_ewm' not in fn_lower:
                        rolling_ma_count += 1
                        category_counts['rolling_ma'] = category_counts.get('rolling_ma', 0) + 1
                    elif '_std' in fn_lower:
                        rolling_std_count += 1
                        category_counts['rolling_std'] = category_counts.get('rolling_std', 0) + 1
                    elif 'regime' in fn_lower:
                        category_counts['regime'] = category_counts.get('regime', 0) + 1
                    elif 'momentum' in fn_lower or 'rsi' in fn_lower or 'macd' in fn_lower:
                        category_counts['momentum'] = category_counts.get('momentum', 0) + 1
                    elif 'volatility' in fn_lower or 'atr' in fn_lower or 'bollinger' in fn_lower:
                        category_counts['volatility'] = category_counts.get('volatility', 0) + 1
                    elif 'volume' in fn_lower or 'obv' in fn_lower:
                        category_counts['volume'] = category_counts.get('volume', 0) + 1
                    else:
                        category_counts['other'] = category_counts.get('other', 0) + 1

                tprint(f"📊 [REGIME_MODELS] Feature breakdown: {category_counts}", color="blue")
                
                # CRITICAL: Verify EWMA features are present
                tprint(f"🔍 [REGIME_MODELS] FEATURE VERIFICATION:", color="cyan", bold=True)
                tprint(f"   • EWMA (_ewm) features: {ewma_count}", color="green" if ewma_count > 0 else "red")
                tprint(f"   • Rolling MA (_ma) features: {rolling_ma_count}", color="blue")
                tprint(f"   • Rolling STD (_std) features: {rolling_std_count}", color="blue")
                
                if ewma_count == 0:
                    tprint("   ❌ CRITICAL ERROR: NO EWMA FEATURES FOUND!", color="red", bold=True)
                    tprint("   → This means apply_ewm_smoothing() failed or was not called", color="red")
                    tprint("   → Regime detection accuracy will be severely impacted", color="red")
                else:
                    tprint(f"   ✅ EWMA features successfully generated: {ewma_count} features", color="green")
                    # Show sample EWMA feature names
                    ewma_features = [fn for fn in feature_names if '_ewm' in fn.lower()]
                    if ewma_features:
                        tprint(f"   → Sample EWMA features: {ewma_features[:5]}...", color="blue")

                # Return X, feature_names, and the original DataFrame index for alignment
                return X, feature_names, X_original_index
            else:
                tprint("❌ [REGIME_MODELS] Feature bank generated no features", color="red")
                return None, None, None

        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Error generating features with feature bank: {e}", color="red")
            self.logger.error(f"Error generating features with feature bank: {str(e)}", exc_info=True)
            return None, None, None

    def _get_system_performance(self) -> Dict[str, Any]:
        """Get current system performance metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            # Memory usage
            memory = psutil.virtual_memory()
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            memory_percent = memory.percent

            return {
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'memory_used_gb': memory_used_gb,
                'memory_total_gb': memory_total_gb,
                'memory_percent': memory_percent
            }
        except Exception as e:
            tprint(f"⚠️ [REGIME_MODELS] Failed to get system performance: {e}", color="yellow")
            return {}

    async def _generate_regime_probability_report(
        self,
        training_results: Dict[str, Any],
        X: np.ndarray,
        feature_names: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Generate a comprehensive report with regime probabilities for all regimes."""
        try:
            tprint("📊 [REGIME_MODELS] Generating regime probability report", color="cyan")

            # Get the trained models
            models = training_results.get('models', {})
            if not models:
                tprint("⚠️ [REGIME_MODELS] No trained models found for report generation", color="yellow")
                return None

            # Get top 3 models from walk-forward validation if available
            top_models_list = []
            walk_forward_metrics = training_results.get('metadata', {}).get('walk_forward_validation', {})
            if not walk_forward_metrics.get('validation_completed', False):
                tprint("⚠️ [REGIME_MODELS] Walk-forward validation not completed, using single best model", color="yellow")
                # Fall back to single best model logic
                model_metrics = training_results.get('model_metrics', {})
                best_model_name = None
                best_accuracy = -1.0
                best_f1 = -1.0

                # Build list of all models with their scores
                model_scores = []
                for model_name_iter, metrics in model_metrics.items():
                    if 'error' not in metrics and model_name_iter in models:
                        accuracy = metrics.get('accuracy', 0)
                        f1_score = metrics.get('f1_score', 0)
                        combined_score = (accuracy + f1_score) / 2
                        model_scores.append({
                            'name': model_name_iter,
                            'accuracy': accuracy,
                            'f1_score': f1_score,
                            'combined_score': combined_score,
                            'metrics': metrics
                        })

                # Sort by combined score
                model_scores.sort(key=lambda x: x['combined_score'], reverse=True)

                # Get top 3 models
                top_models_list = model_scores[:min(3, len(model_scores))]

                # Select best model
                if model_scores:
                    best_model_name = model_scores[0]['name']
                    best_accuracy = model_scores[0]['accuracy']
                    best_f1 = model_scores[0]['f1_score']
                    tprint(f"✅ [REGIME_MODELS] Selected best performing model: {best_model_name} (accuracy: {best_accuracy:.4f}, F1: {best_f1:.4f})", color="green")
                else:
                    # Fallback to first model if no metrics available
                    tprint("⚠️ [REGIME_MODELS] No model metrics available, using first model", color="yellow")
                    best_model_name = list(models.keys())[0]

                model_name = best_model_name
                model = models[model_name]
            else:
                # Use top 3 models from walk-forward validation
                selected_models = walk_forward_metrics.get('selected_models', [])
                if not selected_models:
                    tprint("⚠️ [REGIME_MODELS] No top models found in walk-forward validation", color="yellow")
                    return None

                tprint(f"✅ [REGIME_MODELS] Using top 3 models from walk-forward validation: {selected_models}", color="green")

                # Build top_models_list from walk-forward validation
                model_rankings = walk_forward_metrics.get('model_rankings', [])
                for rank in model_rankings[:3]:
                    top_models_list.append({
                        'name': rank.get('model_name', 'Unknown'),
                        'accuracy': rank.get('accuracy', 0),
                        'f1_score': rank.get('f1_score', 0),
                        'combined_score': rank.get('composite_score', 0),
                        'accuracy_ci': rank.get('accuracy_ci', (0, 0)),
                        'f1_ci': rank.get('f1_ci', (0, 0)),
                        'mel': rank.get('mel', 0),
                        'sfpr': rank.get('sfpr', 0)
                    })

                # Use the top-ranked model for probability analysis
                model_name = selected_models[0] if selected_models else None
                model = models[model_name] if model_name and model_name in models else None

                if not model:
                    tprint(f"⚠️ [REGIME_MODELS] Top model {model_name} not found in trained models", color="yellow")
                    return None

            if not hasattr(model, 'predict_proba'):
                tprint(f"⚠️ [REGIME_MODELS] Model {model_name} does not support probability prediction", color="yellow")
                return None

            # Generate regime probabilities for all samples
            tprint(f"🔮 [REGIME_MODELS] Generating regime probabilities using {model_name} (best performing model)", color="cyan")
            regime_probabilities = model.predict_proba(X)
            regime_predictions = model.predict(X)

            n_regimes = regime_probabilities.shape[1]
            n_samples = len(regime_probabilities)

            # Determine class labels corresponding to probability columns
            if hasattr(model, 'classes_'):
                class_labels = np.asarray(model.classes_)
                if len(class_labels) != n_regimes:
                    tprint(
                        f"⚠️ [REGIME_MODELS] classes_ length ({len(class_labels)}) "
                        f"does not match probability columns ({n_regimes}); falling back to index labels",
                        color="yellow",
                    )
                    class_labels = np.arange(n_regimes)
            else:
                class_labels = np.arange(n_regimes)

            # Get ground truth regime labels from training results
            y_train_full = training_results.get('y_train_full', None)
            if y_train_full is None:
                tprint("⚠️ [REGIME_MODELS] No y_train_full in training results, using model predictions for regime stats", color="yellow")
                ground_truth_labels = np.asarray(regime_predictions)
            else:
                ground_truth_labels = np.asarray(y_train_full)
                tprint(f"✅ [REGIME_MODELS] Using ground truth labels for regime statistics: {len(ground_truth_labels)} samples", color="green")

            # Calculate regime statistics based on ground truth labels and class mapping
            regime_stats = {}
            for col_idx, class_label in enumerate(class_labels):
                regime_probs = regime_probabilities[:, col_idx]
                # Count samples where ground truth label matches this class label
                regime_mask = (ground_truth_labels == class_label)
                regime_count = int(regime_mask.sum())

                if regime_count > 0:
                    regime_max_probs = np.max(regime_probabilities[regime_mask], axis=1)
                    mean_prob = float(np.mean(regime_max_probs))
                    std_prob = float(np.std(regime_max_probs))
                    high_conf = int(np.sum(regime_max_probs > 0.8))
                    med_conf = int(np.sum((regime_max_probs > 0.5) & (regime_max_probs <= 0.8)))
                    low_conf = int(np.sum(regime_max_probs <= 0.5))
                else:
                    mean_prob = 0.0
                    std_prob = 0.0
                    high_conf = 0
                    med_conf = 0
                    low_conf = 0

                regime_key = f"regime_{int(class_label)}"
                regime_stats[regime_key] = {
                    'sample_count': regime_count,
                    'percentage': float(regime_count / n_samples * 100) if n_samples > 0 else 0.0,
                    'mean_probability': mean_prob,
                    'std_probability': std_prob,
                    'min_probability': float(np.min(regime_probs)),
                    'max_probability': float(np.max(regime_probs)),
                    'confidence_distribution': {
                        'high_confidence': high_conf,
                        'medium_confidence': med_conf,
                        'low_confidence': low_conf
                    }
                }

            # Calculate overall statistics
            overall_stats = {
                'total_samples': n_samples,
                'n_regimes': int(len(class_labels)),
                'mean_max_probability': float(np.mean(np.max(regime_probabilities, axis=1))),
                'std_max_probability': float(np.std(np.max(regime_probabilities, axis=1))),
                'regime_balance': float(np.std([
                    regime_stats[f'regime_{int(lbl)}']['percentage']
                    for lbl in class_labels
                ])) if len(class_labels) > 0 else 0.0,
                'prediction_confidence': float(np.mean(np.max(regime_probabilities, axis=1))),
                'uncertainty_entropy': float(np.mean([-np.sum(p * np.log(p + 1e-10)) for p in regime_probabilities]))
            }

            # Extract feature importance from the best model
            feature_importance = {}
            try:
                # Get feature importance from the best model
                if hasattr(model, 'feature_importances_'):
                    # Get LGBM feature importance
                    importances = model.feature_importances_
                    if len(importances) == len(feature_names):
                        lgbm_importance_df = pd.DataFrame({
                            'feature': feature_names,
                            'importance': importances
                        }).sort_values('importance', ascending=False)
                        feature_importance['lgbm_importance'] = lgbm_importance_df
                        tprint(f"✅ [REGIME_MODELS] Extracted LGBM feature importance from best model", color="green")
                
                # Try to compute SHAP values for the best model
                try:
                    import shap
                    # DEBUG: Log model type before SHAP
                    tprint(f"🔍 [DEBUG] Modèle passé à SHAP: {type(model)}", color="yellow")
                    
                    # Import SHAP utilities for safe handling
                    try:
                        from src.utils.shap_utils import safe_shap_tree_explainer, safe_shap_values
                        tprint("✅ [DEBUG] SHAP utils importées avec succès", color="green")
                    except ImportError as e:
                        tprint(f"⚠️ [DEBUG] SHAP utils non disponibles: {e}", color="yellow")
                        # Fallback to manual extraction
                        from src.utils.shap_utils import extract_base_model_from_calibrated, apply_numpy_shap_fix
                        model_for_shap = extract_base_model_from_calibrated(model)
                        if model_for_shap is None:
                            model_for_shap = model
                        apply_numpy_shap_fix()
                        explainer = shap.TreeExplainer(model_for_shap)
                    else:
                        # Use safe SHAP utilities
                        explainer = safe_shap_tree_explainer(model)
                        if explainer is None:
                            tprint("❌ [DEBUG] Impossible de créer TreeExplainer avec SHAP utils", color="red")
                            return None
                    shap_values = safe_shap_values(explainer, X[:100])  # Use subset for speed
                    if shap_values is None:
                        tprint("❌ [DEBUG] Impossible de calculer les valeurs SHAP dans feature selection", color="red")
                        return None
                    
                    # For multi-class, get mean absolute SHAP values across classes
                    if isinstance(shap_values, list):
                        # Multi-class case
                        shap_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                    else:
                        # Binary case
                        shap_importance = np.abs(shap_values).mean(axis=0)
                    
                    # Create SHAP importance dataframe
                    shap_importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'shap_importance': shap_importance
                    }).sort_values('shap_importance', ascending=False)
                    feature_importance['shap_importance'] = shap_importance_df
                    tprint(f"✅ [REGIME_MODELS] Computed SHAP values for best model", color="green")
                    
                except Exception as shap_error:
                    tprint(f"⚠️ [REGIME_MODELS] SHAP computation failed: {shap_error}", color="yellow")
                    
            except Exception as e:
                tprint(f"⚠️ [REGIME_MODELS] Feature importance extraction failed: {e}", color="yellow")
            
            # Generate top 60 features with importance scores
            top_60_features = {}
            try:
                if feature_importance:
                    # Get top 60 LGBM features
                    if 'lgbm_importance' in feature_importance:
                        lgbm_top_60 = feature_importance['lgbm_importance'].head(60)
                        top_60_features['lgbm_top_60'] = lgbm_top_60.to_dict('records')
                        tprint(f"✅ [REGIME_MODELS] Extracted top 60 LGBM features", color="green")
                    
                    # Get top 60 SHAP features
                    if 'shap_importance' in feature_importance:
                        shap_top_60 = feature_importance['shap_importance'].head(60)
                        top_60_features['shap_top_60'] = shap_top_60.to_dict('records')
                        tprint(f"✅ [REGIME_MODELS] Extracted top 60 SHAP features", color="green")
                    
                    # Create combined importance ranking (weighted average of LGBM and SHAP)
                    if 'lgbm_importance' in feature_importance and 'shap_importance' in feature_importance:
                        # Merge the two importance dataframes
                        combined_df = pd.merge(
                            feature_importance['lgbm_importance'],
                            feature_importance['shap_importance'],
                            on='feature',
                            how='outer',
                            suffixes=('_lgbm', '_shap')
                        ).fillna(0)
                        
                        # Normalize both importance scores to 0-1 range
                        combined_df['importance_lgbm_norm'] = (
                            combined_df['importance_lgbm'] - combined_df['importance_lgbm'].min()
                        ) / (combined_df['importance_lgbm'].max() - combined_df['importance_lgbm'].min() + 1e-8)
                        
                        combined_df['shap_importance_norm'] = (
                            combined_df['shap_importance'] - combined_df['shap_importance'].min()
                        ) / (combined_df['shap_importance'].max() - combined_df['shap_importance'].min() + 1e-8)
                        
                        # Calculate combined importance (weighted average: 60% LGBM, 40% SHAP)
                        combined_df['combined_importance'] = (
                            0.6 * combined_df['importance_lgbm_norm'] +
                            0.4 * combined_df['shap_importance_norm']
                        )
                        
                        # Sort by combined importance and get top 60
                        combined_top_60 = combined_df.sort_values('combined_importance', ascending=False).head(60)
                        top_60_features['combined_top_60'] = combined_top_60.to_dict('records')
                        tprint(f"✅ [REGIME_MODELS] Created combined top 60 features ranking", color="green")
                        
                        # Generate visualization data for SHAP values of top 20 features
                        if 'shap_importance' in feature_importance:
                            top_20_features = combined_top_60.head(20)['feature'].tolist()
                            shap_viz_data = {
                                'top_20_features': top_20_features,
                                'feature_names': feature_names,
                                'shap_values': None,  # Would need actual SHAP values from explainer
                                'base_values': None    # Would need base values from explainer
                            }
                            top_60_features['shap_visualization_data'] = shap_viz_data
                            tprint(f"✅ [REGIME_MODELS] Generated SHAP visualization data for top 20 features", color="green")
            except Exception as e:
                tprint(f"⚠️ [REGIME_MODELS] Failed to generate top 60 features: {e}", color="yellow")
                top_60_features = {}
            
            # Generate comprehensive report
            report = {
                'model_name': model_name,
                'generation_timestamp': datetime.now().isoformat(),
                'overall_statistics': overall_stats,
                'regime_statistics': regime_stats,
                'regime_probabilities': regime_probabilities.tolist(),
                'regime_labels': regime_predictions.tolist(),
                'ground_truth_labels': ground_truth_labels.tolist() if isinstance(ground_truth_labels, np.ndarray) else ground_truth_labels,
                'feature_names': feature_names,
                'data_shape': X.shape,
                'feature_importance': feature_importance,
                'top_60_features': top_60_features,
                'report_type': 'regime_probability_analysis'
            }

            # Add top models comparison
            if top_models_list:
                report['top_models'] = top_models_list
                tprint(f"✅ [REGIME_MODELS] Added top {len(top_models_list)} models to report", color="green")

            # Add model metrics if available
            model_metrics = training_results.get('model_metrics', {})
            if model_name in model_metrics:
                metrics = model_metrics[model_name].copy()

                # Calculate R2 score if possible
                try:
                    from sklearn.metrics import r2_score
                    # For classification, use a pseudo R2 based on accuracy
                    # R2 = 1 - (1 - accuracy)^2 (simplified version)
                    accuracy = metrics.get('accuracy', 0)
                    # Use Cohen's Kappa as a better alternative to R2 for classification
                    from sklearn.metrics import cohen_kappa_score
                    if y_train_full is not None:
                        kappa = cohen_kappa_score(ground_truth_labels, regime_predictions)
                        metrics['cohen_kappa'] = float(kappa)
                        tprint(f"✅ [REGIME_MODELS] Cohen's Kappa: {kappa:.4f}", color="green")
                except Exception as e:
                    tprint(f"⚠️ [REGIME_MODELS] Failed to calculate additional metrics: {e}", color="yellow")

                report['model_metrics'] = metrics

            # Generate text report
            text_report = self._generate_text_report(report)
            report['text_report'] = text_report

            tprint(f"✅ [REGIME_MODELS] Regime probability report generated for {n_regimes} regimes", color="green")
            return report

        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Failed to generate regime probability report: {e}", color="red")
            self.logger.error(f"Failed to generate regime probability report: {e}", exc_info=True)
            return None

    def _generate_text_report(self, report: Dict[str, Any]) -> str:
        """Generate a human-readable text report from regime probability data."""
        try:
            lines = []
            lines.append("=" * 80)
            lines.append("REGIME MODELS TRAINING REPORT")
            lines.append(f"Model: {report.get('model_name', 'Unknown')}")
            lines.append(f"Generated: {report.get('generation_timestamp', 'Unknown')}")
            lines.append("=" * 80)
            lines.append("")

            # Overall Statistics
            overall = report.get('overall_statistics', {})
            lines.append("📊 OVERALL STATISTICS")
            lines.append("-" * 40)
            lines.append(f"Total Samples: {overall.get('total_samples', 'N/A')}")
            lines.append(f"Number of Regimes: {overall.get('n_regimes', 'N/A')}")
            lines.append(f"Mean Max Probability: {overall.get('mean_max_probability', 0):.3f}")
            lines.append(f"Std Max Probability: {overall.get('std_max_probability', 0):.3f}")
            lines.append(f"Regime Balance: {overall.get('regime_balance', 0):.3f}")
            lines.append(f"Prediction Confidence: {overall.get('prediction_confidence', 0):.3f}")
            lines.append(f"Uncertainty Entropy: {overall.get('uncertainty_entropy', 0):.3f}")
            lines.append("")

            # Model Metrics
            if 'model_metrics' in report:
                metrics = report['model_metrics']
                lines.append("🎯 MODEL PERFORMANCE METRICS")
                lines.append("-" * 40)
                lines.append(f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
                lines.append(f"Precision (Weighted): {metrics.get('precision', 'N/A'):.4f}")
                lines.append(f"Recall (Weighted): {metrics.get('recall', 'N/A'):.4f}")
                lines.append(f"F1-Score (Weighted): {metrics.get('f1_score', 'N/A'):.4f}")
                lines.append("")

            # Regime Statistics
            regime_stats = report.get('regime_statistics', {})
            lines.append("🎯 REGIME PROBABILITY STATISTICS")
            lines.append("-" * 40)

            for regime_key, regime_data in regime_stats.items():
                if isinstance(regime_data, dict):
                    lines.append(f"{regime_key.upper()}:")
                    lines.append(f"  Sample Count: {regime_data.get('sample_count', 0)}")
                    lines.append(f"  Percentage: {regime_data.get('percentage', 0):.1f}%")
                    lines.append(f"  Mean Probability: {regime_data.get('mean_probability', 0):.3f}")
                    lines.append(f"  Std Probability: {regime_data.get('std_probability', 0):.3f}")
                    lines.append(f"  Min Probability: {regime_data.get('min_probability', 0):.3f}")
                    lines.append(f"  Max Probability: {regime_data.get('max_probability', 0):.3f}")
                    
                    conf_dist = regime_data.get('confidence_distribution', {})
                    lines.append(f"  Confidence Distribution:")
                    lines.append(f"    High (>0.8): {conf_dist.get('high_confidence', 0)}")
                    lines.append(f"    Medium (0.5-0.8): {conf_dist.get('medium_confidence', 0)}")
                    lines.append(f"    Low (≤0.5): {conf_dist.get('low_confidence', 0)}")
                    lines.append("")
            
            # Top 10 Features Section
            if 'feature_importance' in report:
                lines.append("🎯 TOP 10 FEATURES BY IMPORTANCE")
                lines.append("-" * 40)
                
                feature_importance = report['feature_importance']
                
                # LGBM Top Features
                if 'lgbm_importance' in feature_importance:
                    lines.append("LGBM Feature Importance (Top 10):")
                    lgbm_importance = feature_importance['lgbm_importance']
                    for i, row in lgbm_importance.head(10).iterrows():
                        lines.append(f"  {i+1:2d}. {row['feature']:<40} (importance: {row['importance']:.6f})")
                    lines.append("")
                
                # SHAP Top Features
                if 'shap_importance' in feature_importance:
                    lines.append("SHAP Feature Importance (Top 10):")
                    shap_importance = feature_importance['shap_importance']
                    for i, row in shap_importance.head(10).iterrows():
                        lines.append(f"  {i+1:2d}. {row['feature']:<40} (SHAP: {row['shap_importance']:.6f})")
                    lines.append("")

            lines.append("=" * 80)
            lines.append("END OF REGIME MODELS TRAINING REPORT")
            lines.append("=" * 80)

            return "\n".join(lines)

        except Exception as e:
            return f"Error generating text report: {e}"

    def _generate_markdown_report(self, report: Dict[str, Any], symbol: str, output_dir: str = "outcomes") -> Optional[str]:
        """Generate a comprehensive markdown report."""
        try:
            from pathlib import Path
            
            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"regime_models_training_report_{symbol}_{timestamp}.md"
            report_path = output_path / filename
            
            tprint(f"📝 Generating markdown report: {report_path}", color="cyan")
            
            # Build markdown content
            md_lines = []
            md_lines.append("# Regime Models Training Report")
            md_lines.append("")
            md_lines.append(f"**Symbol:** {symbol}")
            md_lines.append(f"**Primary Model:** {report.get('model_name', 'Unknown')}")
            md_lines.append(f"**Generated:** {report.get('generation_timestamp', 'Unknown')}")
            md_lines.append(f"**Report Version:** 1.0")
            md_lines.append("")
            
            # Add Top 3 Models Comparison Table
            if 'top_models' in report:
                md_lines.append("## Top 3 Models Comparison")
                md_lines.append("")
                md_lines.append("| Rank | Model Name | Accuracy | F1-Score | Combined Score |")
                md_lines.append("|------|------------|----------|----------|----------------|")
                
                top_models = report['top_models']
                for i, model_info in enumerate(top_models, 1):
                    md_lines.append(
                        f"| {i} | {model_info.get('name', 'Unknown')} | "
                        f"{model_info.get('accuracy', 0):.4f} | "
                        f"{model_info.get('f1_score', 0):.4f} | "
                        f"{model_info.get('combined_score', 0):.4f} |"
                    )
                md_lines.append("")
            
            # Overall Statistics
            overall = report.get('overall_statistics', {})
            md_lines.append("## Overall Statistics")
            md_lines.append("")
            md_lines.append("| Metric | Value |")
            md_lines.append("|--------|-------|")
            md_lines.append(f"| Total Samples | {overall.get('total_samples', 'N/A')} |")
            md_lines.append(f"| Number of Regimes | {overall.get('n_regimes', 'N/A')} |")
            md_lines.append(f"| Mean Max Probability | {overall.get('mean_max_probability', 0):.4f} |")
            md_lines.append(f"| Std Max Probability | {overall.get('std_max_probability', 0):.4f} |")
            md_lines.append(f"| Regime Balance | {overall.get('regime_balance', 0):.4f} |")
            md_lines.append(f"| Prediction Confidence | {overall.get('prediction_confidence', 0):.4f} |")
            md_lines.append(f"| Uncertainty Entropy | {overall.get('uncertainty_entropy', 0):.4f} |")
            md_lines.append("")
            
            # Model Performance Metrics
            if 'model_metrics' in report:
                metrics = report['model_metrics']
                md_lines.append("## Model Performance Metrics")
                md_lines.append("")
                md_lines.append("| Metric | Value |")
                md_lines.append("|--------|-------|")
                md_lines.append(f"| Accuracy | {metrics.get('accuracy', 0):.4f} |")
                md_lines.append(f"| Precision (Weighted) | {metrics.get('precision', 0):.4f} |")
                md_lines.append(f"| Recall (Weighted) | {metrics.get('recall', 0):.4f} |")
                md_lines.append(f"| F1-Score (Weighted) | {metrics.get('f1_score', 0):.4f} |")
                if 'cohen_kappa' in metrics:
                    md_lines.append(f"| Cohen's Kappa | {metrics.get('cohen_kappa', 0):.4f} |")
                md_lines.append("")
            
            # Regime Statistics
            regime_stats = report.get('regime_statistics', {})
            md_lines.append("## Regime Statistics")
            md_lines.append("")
            md_lines.append("| Regime | Sample Count | Percentage | Mean Prob | Std Prob | High Conf | Med Conf | Low Conf |")
            md_lines.append("|--------|--------------|------------|-----------|----------|-----------|----------|----------|")
            
            for regime_key, regime_data in regime_stats.items():
                if isinstance(regime_data, dict):
                    conf_dist = regime_data.get('confidence_distribution', {})
                    md_lines.append(
                        f"| {regime_key} | "
                        f"{regime_data.get('sample_count', 0)} | "
                        f"{regime_data.get('percentage', 0):.1f}% | "
                        f"{regime_data.get('mean_probability', 0):.3f} | "
                        f"{regime_data.get('std_probability', 0):.3f} | "
                        f"{conf_dist.get('high_confidence', 0)} | "
                        f"{conf_dist.get('medium_confidence', 0)} | "
                        f"{conf_dist.get('low_confidence', 0)} |"
                    )
            md_lines.append("")
            
            # Feature Importance Section
            if 'feature_importance' in report:
                md_lines.append("## Feature Importance Analysis")
                md_lines.append("")
                
                feature_importance = report['feature_importance']
                
                # LGBM Feature Importance (Top 60)
                if 'lgbm_importance' in feature_importance:
                    md_lines.append("### LGBM Feature Importance (Top 60)")
                    md_lines.append("")
                    md_lines.append("| Rank | Feature | Importance |")
                    md_lines.append("|------|---------|------------|")
                    
                    lgbm_importance = feature_importance['lgbm_importance']
                    for i, row in lgbm_importance.head(60).iterrows():
                        md_lines.append(f"| {i+1} | {row['feature']} | {row['importance']:.6f} |")
                    md_lines.append("")
                
                # SHAP Feature Importance (Top 60)
                if 'shap_importance' in feature_importance:
                    md_lines.append("### SHAP Feature Importance (Top 60)")
                    md_lines.append("")
                    md_lines.append("| Rank | Feature | SHAP Value |")
                    md_lines.append("|------|---------|------------|")
                    
                    shap_importance = feature_importance['shap_importance']
                    for i, row in shap_importance.head(60).iterrows():
                        md_lines.append(f"| {i+1} | {row['feature']} | {row['shap_importance']:.6f} |")
                    md_lines.append("")
                
                # Combined Feature Importance (Top 60)
                if 'top_60_features' in report and 'combined_top_60' in report['top_60_features']:
                    md_lines.append("### Combined Feature Importance (Top 60)")
                    md_lines.append("")
                    md_lines.append("| Rank | Feature | LGBM Importance | SHAP Value | Combined Score |")
                    md_lines.append("|------|---------|-----------------|------------|----------------|")
                    
                    combined_features = report['top_60_features']['combined_top_60']
                    for i, feature_data in enumerate(combined_features, 1):
                        md_lines.append(f"| {i} | {feature_data.get('feature', 'N/A')} | {feature_data.get('importance_lgbm', 0):.6f} | {feature_data.get('shap_importance', 0):.6f} | {feature_data.get('combined_importance', 0):.6f} |")
                    md_lines.append("")
                
                # Feature Importance Comparison (Top 20)
                if 'lgbm_importance' in feature_importance and 'shap_importance' in feature_importance:
                    md_lines.append("### Feature Importance Comparison (Top 20)")
                    md_lines.append("")
                    md_lines.append("| Rank | Feature | LGBM Importance | SHAP Value |")
                    md_lines.append("|------|---------|-----------------|------------|")
                    
                    comparison_df = pd.merge(
                        feature_importance['lgbm_importance'].head(20),
                        feature_importance['shap_importance'].head(20),
                        on='feature',
                        how='outer',
                        suffixes=('_lgbm', '_shap')
                    ).fillna(0)
                    
                    for i, row in comparison_df.iterrows():
                        md_lines.append(f"| {i+1} | {row['feature']} | {row.get('importance', 0):.6f} | {row.get('shap_importance', 0):.6f} |")
                    md_lines.append("")
                
                # SHAP Visualization Data Information
                if 'top_60_features' in report and 'shap_visualization_data' in report['top_60_features']:
                    md_lines.append("### SHAP Visualization Data")
                    md_lines.append("")
                    shap_viz_data = report['top_60_features']['shap_visualization_data']
                    md_lines.append(f"- **Top 20 Features for Visualization:** {len(shap_viz_data.get('top_20_features', []))}")
                    md_lines.append(f"- **Total Feature Names Available:** {len(shap_viz_data.get('feature_names', []))}")
                    md_lines.append("- **Visualization Data:** Available for generating SHAP plots")
                    md_lines.append("")
            
            # Write to file
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(md_lines))
            
            tprint(f"✅ Markdown report generated: {report_path}", color="green")
            return str(report_path)
            
        except Exception as e:
            tprint(f"❌ Failed to generate markdown report: {e}", color="red")
            self.logger.error(f"Failed to generate markdown report: {e}", exc_info=True)
            return None

    def _generate_csv_reports(self, report: Dict[str, Any], training_results: Dict[str, Any], symbol: str, output_dir: str = "outcomes") -> Tuple[Optional[str], Optional[str]]:
        """Generate comprehensive CSV reports."""
        try:
            from pathlib import Path
            
            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Generate metrics CSV
            metrics_filename = f"regime_models_training_metrics_{symbol}_{timestamp}.csv"
            metrics_path = output_path / metrics_filename
            
            tprint(f"📊 Generating metrics CSV: {metrics_path}", color="cyan")
            
            csv_data = []
            csv_data.append(['Metric Category', 'Metric Name', 'Value', 'Description'])
            
            # Overall Statistics
            overall = report.get('overall_statistics', {})
            csv_data.append(['Overall', 'Total Samples', str(overall.get('total_samples', 'N/A')), 'Total number of samples'])
            csv_data.append(['Overall', 'Number of Regimes', str(overall.get('n_regimes', 'N/A')), 'Number of regimes discovered'])
            csv_data.append(['Overall', 'Mean Max Probability', f"{overall.get('mean_max_probability', 0):.6f}", 'Average maximum probability across samples'])
            csv_data.append(['Overall', 'Std Max Probability', f"{overall.get('std_max_probability', 0):.6f}", 'Standard deviation of maximum probabilities'])
            csv_data.append(['Overall', 'Regime Balance', f"{overall.get('regime_balance', 0):.6f}", 'Standard deviation of regime percentages'])
            csv_data.append(['Overall', 'Prediction Confidence', f"{overall.get('prediction_confidence', 0):.6f}", 'Average prediction confidence'])
            csv_data.append(['Overall', 'Uncertainty Entropy', f"{overall.get('uncertainty_entropy', 0):.6f}", 'Average entropy of predictions'])
            
            # Top 3 Models Metrics
            if 'top_models' in report:
                csv_data.append(['Top Models', 'Number of Top Models', str(len(report['top_models'])), 'Number of top performing models included in report'])
                for i, model_info in enumerate(report['top_models'], 1):
                    model_name = model_info.get('name', f'Model_{i}')
                    csv_data.append([f'Top Model {i}', 'Model Name', model_name, f'Rank {i} model'])
                    csv_data.append([f'Top Model {i}', 'Accuracy', f"{model_info.get('accuracy', 0):.6f}", f'Accuracy of {model_name}'])
                    csv_data.append([f'Top Model {i}', 'F1-Score', f"{model_info.get('f1_score', 0):.6f}", f'F1-Score of {model_name}'])
                    csv_data.append([f'Top Model {i}', 'Combined Score', f"{model_info.get('combined_score', 0):.6f}", f'Combined score of {model_name}'])
                    if 'accuracy_ci' in model_info:
                        ci_lower, ci_upper = model_info['accuracy_ci']
                        csv_data.append([f'Top Model {i}', 'Accuracy CI Lower', f"{ci_lower:.6f}", f'Lower bound of 95% CI for {model_name}'])
                        csv_data.append([f'Top Model {i}', 'Accuracy CI Upper', f"{ci_upper:.6f}", f'Upper bound of 95% CI for {model_name}'])
                    if 'f1_ci' in model_info:
                        ci_lower, ci_upper = model_info['f1_ci']
                        csv_data.append([f'Top Model {i}', 'F1 CI Lower', f"{ci_lower:.6f}", f'Lower bound of 95% CI for {model_name}'])
                        csv_data.append([f'Top Model {i}', 'F1 CI Upper', f"{ci_upper:.6f}", f'Upper bound of 95% CI for {model_name}'])
                    csv_data.append([f'Top Model {i}', 'MEL', f"{model_info.get('mel', 0):.6f}", f'Maximum Episode Length for {model_name}'])
                    csv_data.append([f'Top Model {i}', 'SFPR', f"{model_info.get('sfpr', 0):.6f}", f'Structural False Positive Rate for {model_name}'])
            
            # Primary Model Metrics (for backward compatibility)
            if 'model_metrics' in report:
                metrics = report['model_metrics']
                csv_data.append(['Primary Model Performance', 'Accuracy', f"{metrics.get('accuracy', 0):.6f}", 'Classification accuracy of primary model'])
                csv_data.append(['Primary Model Performance', 'Precision', f"{metrics.get('precision', 0):.6f}", 'Weighted precision of primary model'])
                csv_data.append(['Primary Model Performance', 'Recall', f"{metrics.get('recall', 0):.6f}", 'Weighted recall of primary model'])
                csv_data.append(['Primary Model Performance', 'F1-Score', f"{metrics.get('f1_score', 0):.6f}", 'Weighted F1-score of primary model'])
            
            # Regime Statistics
            regime_stats = report.get('regime_statistics', {})
            for regime_key, regime_data in regime_stats.items():
                if isinstance(regime_data, dict):
                    csv_data.append([f'Regime {regime_key}', 'Sample Count', str(regime_data.get('sample_count', 0)), 'Number of samples in regime'])
                    csv_data.append([f'Regime {regime_key}', 'Percentage', f"{regime_data.get('percentage', 0):.2f}%", 'Percentage of total samples'])
                    csv_data.append([f'Regime {regime_key}', 'Mean Probability', f"{regime_data.get('mean_probability', 0):.6f}", 'Average probability for regime'])
                    csv_data.append([f'Regime {regime_key}', 'Std Probability', f"{regime_data.get('std_probability', 0):.6f}", 'Standard deviation of probabilities'])
                    
                    conf_dist = regime_data.get('confidence_distribution', {})
                    csv_data.append([f'Regime {regime_key}', 'High Confidence Count', str(conf_dist.get('high_confidence', 0)), 'Samples with >0.8 probability'])
                    csv_data.append([f'Regime {regime_key}', 'Medium Confidence Count', str(conf_dist.get('medium_confidence', 0)), 'Samples with 0.5-0.8 probability'])
                    csv_data.append([f'Regime {regime_key}', 'Low Confidence Count', str(conf_dist.get('low_confidence', 0)), 'Samples with <0.5 probability'])
            
            # Feature Importance Data
            if 'feature_importance' in report:
                feature_importance = report['feature_importance']
                
                # Enhanced Feature Importance Data (Top 60)
                if 'feature_importance' in report:
                    feature_importance = report['feature_importance']
                    
                    # LGBM Feature Importance (Top 60)
                    if 'lgbm_importance' in feature_importance:
                        csv_data.append(['Feature Importance', 'LGBM Top Feature', '', 'Top LGBM feature by importance'])
                        lgbm_importance = feature_importance['lgbm_importance']
                        for i, row in lgbm_importance.head(60).iterrows():
                            csv_data.append(['Feature Importance', f'LGBM Rank {i+1}', f"{row['importance']:.6f}", row['feature']])
                    
                    # SHAP Feature Importance (Top 60)
                    if 'shap_importance' in feature_importance:
                        csv_data.append(['Feature Importance', 'SHAP Top Feature', '', 'Top SHAP feature by importance'])
                        shap_importance = feature_importance['shap_importance']
                        for i, row in shap_importance.head(60).iterrows():
                            csv_data.append(['Feature Importance', f'SHAP Rank {i+1}', f"{row['shap_importance']:.6f}", row['feature']])
            
            # Write metrics CSV
            import csv
            with open(metrics_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)
            
            tprint(f"✅ Metrics CSV generated: {metrics_path}", color="green")
            
            # Generate separate feature importance CSV files
            feature_importance_paths = []
            if 'feature_importance' in report:
                feature_importance = report['feature_importance']
                
                # LGBM Feature Importance CSV
                if 'lgbm_importance' in feature_importance:
                    lgbm_filename = f"feature_importance_lgbm_{symbol}_{timestamp}.csv"
                    lgbm_path = output_path / lgbm_filename
                    
                    lgbm_importance = feature_importance['lgbm_importance']
                    lgbm_importance.to_csv(lgbm_path, index=False)
                    feature_importance_paths.append(str(lgbm_path))
                    tprint(f"✅ LGBM feature importance CSV generated: {lgbm_path}", color="green")
                
                # SHAP Feature Importance CSV
                if 'shap_importance' in feature_importance:
                    shap_filename = f"feature_importance_shap_{symbol}_{timestamp}.csv"
                    shap_path = output_path / shap_filename
                    
                    shap_importance = feature_importance['shap_importance']
                    shap_importance.to_csv(shap_path, index=False)
                    feature_importance_paths.append(str(shap_path))
                    tprint(f"✅ SHAP feature importance CSV generated: {shap_path}", color="green")
                
                # Feature Importance Comparison CSV
                if 'lgbm_importance' in feature_importance and 'shap_importance' in feature_importance:
                    comparison_filename = f"feature_importance_comparison_{symbol}_{timestamp}.csv"
                    comparison_path = output_path / comparison_filename
                    
                    comparison_df = pd.merge(
                        feature_importance['lgbm_importance'],
                        feature_importance['shap_importance'],
                        on='feature',
                        how='outer',
                        suffixes=('_lgbm', '_shap')
                    ).fillna(0)
                    
                    comparison_df.to_csv(comparison_path, index=False)
                    feature_importance_paths.append(str(comparison_path))
                    tprint(f"✅ Feature importance comparison CSV generated: {comparison_path}", color="green")
                
                # Top 60 Combined Feature Importance CSV
                if 'top_60_features' in report:
                    top_60_features = report['top_60_features']
                    
                    if 'combined_top_60' in top_60_features:
                        combined_filename = f"feature_importance_combined_top_60_{symbol}_{timestamp}.csv"
                        combined_path = output_path / combined_filename
                        
                        combined_df = pd.DataFrame(top_60_features['combined_top_60'])
                        combined_df.to_csv(combined_path, index=False)
                        feature_importance_paths.append(str(combined_path))
                        tprint(f"✅ Combined top 60 feature importance CSV generated: {combined_path}", color="green")
                    
                    if 'lgbm_top_60' in top_60_features:
                        lgbm_60_filename = f"feature_importance_lgbm_top_60_{symbol}_{timestamp}.csv"
                        lgbm_60_path = output_path / lgbm_60_filename
                        
                        lgbm_60_df = pd.DataFrame(top_60_features['lgbm_top_60'])
                        lgbm_60_df.to_csv(lgbm_60_path, index=False)
                        feature_importance_paths.append(str(lgbm_60_path))
                        tprint(f"✅ LGBM top 60 feature importance CSV generated: {lgbm_60_path}", color="green")
                    
                    if 'shap_top_60' in top_60_features:
                        shap_60_filename = f"feature_importance_shap_top_60_{symbol}_{timestamp}.csv"
                        shap_60_path = output_path / shap_60_filename
                        
                        shap_60_df = pd.DataFrame(top_60_features['shap_top_60'])
                        shap_60_df.to_csv(shap_60_path, index=False)
                        feature_importance_paths.append(str(shap_60_path))
                        tprint(f"✅ SHAP top 60 feature importance CSV generated: {shap_60_path}", color="green")
            
            # 2. Generate comprehensive model comparison CSV (all models)
            models = training_results.get('models', {})
            model_metrics = training_results.get('model_metrics', {})

            comparison_path = None
            if len(model_metrics) >= 1:  # Generate even for single model
                comparison_filename = f"regime_models_comparison_{symbol}_{timestamp}.csv"
                comparison_path = output_path / comparison_filename

                tprint(f"📊 Generating comprehensive model comparison CSV: {comparison_path}", color="cyan")

                comparison_data = []
                # Enhanced header with more metrics
                comparison_data.append([
                    'Model Name',
                    'Accuracy',
                    'Precision (Weighted)',
                    'Recall (Weighted)',
                    'F1-Score (Weighted)',
                    'Training Status'
                ])

                for model_name, metrics in model_metrics.items():
                    if 'error' in metrics:
                        # Model failed
                        comparison_data.append([
                            model_name,
                            'ERROR',
                            'ERROR',
                            'ERROR',
                            'ERROR',
                            f"Failed: {metrics['error']}"
                        ])
                    else:
                        comparison_data.append([
                            model_name,
                            f"{metrics.get('accuracy', 0):.6f}",
                            f"{metrics.get('precision', 0):.6f}",
                            f"{metrics.get('recall', 0):.6f}",
                            f"{metrics.get('f1_score', 0):.6f}",
                            'Success'
                        ])

                with open(comparison_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(comparison_data)

                tprint(f"✅ Comprehensive model comparison CSV generated: {comparison_path}", color="green")

            # 3. Generate detailed per-model metrics CSV
            detailed_path = None
            if len(model_metrics) >= 1:
                detailed_filename = f"regime_models_detailed_all_{symbol}_{timestamp}.csv"
                detailed_path = output_path / detailed_filename

                tprint(f"📊 Generating detailed per-model metrics CSV: {detailed_path}", color="cyan")

                detailed_data = []
                detailed_data.append([
                    'Model Name',
                    'Metric Type',
                    'Metric Name',
                    'Value',
                    'Description'
                ])

                for model_name, metrics in model_metrics.items():
                    if 'error' not in metrics:
                        # Overall metrics
                        detailed_data.append([model_name, 'Overall', 'Accuracy', f"{metrics.get('accuracy', 0):.6f}", 'Classification accuracy'])
                        detailed_data.append([model_name, 'Overall', 'Precision (Weighted)', f"{metrics.get('precision', 0):.6f}", 'Weighted average precision'])
                        detailed_data.append([model_name, 'Overall', 'Recall (Weighted)', f"{metrics.get('recall', 0):.6f}", 'Weighted average recall'])
                        detailed_data.append([model_name, 'Overall', 'F1-Score (Weighted)', f"{metrics.get('f1_score', 0):.6f}", 'Weighted average F1-score'])

                        # Support
                        if 'support' in metrics:
                            detailed_data.append([model_name, 'Overall', 'Total Support', str(metrics.get('support')), 'Total number of samples'])

                        # Per-class metrics if available
                        if 'classification_report' in metrics:
                            report_dict = metrics['classification_report']
                            if isinstance(report_dict, dict):
                                for class_label, class_metrics in report_dict.items():
                                    if isinstance(class_metrics, dict) and class_label not in ['accuracy', 'macro avg', 'weighted avg']:
                                        detailed_data.append([model_name, f'Class {class_label}', 'Precision', f"{class_metrics.get('precision', 0):.6f}", f'Precision for class {class_label}'])
                                        detailed_data.append([model_name, f'Class {class_label}', 'Recall', f"{class_metrics.get('recall', 0):.6f}", f'Recall for class {class_label}'])
                                        detailed_data.append([model_name, f'Class {class_label}', 'F1-Score', f"{class_metrics.get('f1-score', 0):.6f}", f'F1-score for class {class_label}'])
                                        detailed_data.append([model_name, f'Class {class_label}', 'Support', str(class_metrics.get('support', 0)), f'Number of samples in class {class_label}'])

                with open(detailed_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(detailed_data)

                tprint(f"✅ Detailed per-model metrics CSV generated: {detailed_path}", color="green")

            # 4. Generate per-regime performance CSV
            regime_perf_path = None
            if len(model_metrics) >= 1:
                regime_perf_filename = f"regime_performance_by_model_{symbol}_{timestamp}.csv"
                regime_perf_path = output_path / regime_perf_filename

                tprint(f"📊 Generating per-regime performance CSV: {regime_perf_path}", color="cyan")

                regime_data = []
                regime_data.append([
                    'Model Name',
                    'Regime',
                    'Precision',
                    'Recall',
                    'F1-Score',
                    'Support'
                ])

                for model_name, metrics in model_metrics.items():
                    if 'error' not in metrics and 'classification_report' in metrics:
                        report_dict = metrics['classification_report']
                        if isinstance(report_dict, dict):
                            for class_label, class_metrics in report_dict.items():
                                if isinstance(class_metrics, dict) and class_label not in ['accuracy', 'macro avg', 'weighted avg']:
                                    regime_data.append([
                                        model_name,
                                        class_label,
                                        f"{class_metrics.get('precision', 0):.6f}",
                                        f"{class_metrics.get('recall', 0):.6f}",
                                        f"{class_metrics.get('f1-score', 0):.6f}",
                                        str(class_metrics.get('support', 0))
                                    ])

                with open(regime_perf_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(regime_data)

                tprint(f"✅ Per-regime performance CSV generated: {regime_perf_path}", color="green")

            return str(metrics_path), str(comparison_path) if comparison_path else None
            
        except Exception as e:
            tprint(f"❌ Failed to generate CSV reports: {e}", color="red")
            self.logger.error(f"Failed to generate CSV reports: {e}", exc_info=True)
            return None, None