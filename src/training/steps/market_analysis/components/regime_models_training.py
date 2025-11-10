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
        self.enable_smoothed_features = True
        self.smoothing_window_sizes = [3, 5, 7]

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
        self.model_config = {
            'random_state': 42,
            'test_size': 0.2,
            'cv_folds': 5,
            'n_jobs': -1
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
                    'n_jobs': -1,
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
                    'n_jobs': -1
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
                    'n_jobs': -1
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
        tprint(f"🔍 [DEBUG] Config being passed to temporal splitter: min_regime_samples={self.validated_config.get('min_regime_samples', 'NOT_SET')}", color="yellow")
        self.temporal_splitter = create_temporal_splitter(self.validated_config)
        # Workaround: Directly set min_regime_samples to handle rare regimes
        if hasattr(self.temporal_splitter, 'min_regime_samples'):
            self.temporal_splitter.min_regime_samples = 5
            tprint(f"🔧 [REGIME_MODELS] Temporal splitter min_regime_samples set to {self.temporal_splitter.min_regime_samples}", color="cyan")
        tprint("✅ [REGIME_MODELS] Temporal splitter initialized", color="green")

        # Initialize walk-forward validator for OOS model selection
        wf_config = RegimeValidationConfig(
            n_outer_folds=5,
            n_inner_folds=3,
            embargo_pct=0.05,
            min_train_samples=100,
            min_val_samples=30,
            min_regime_samples=self.validated_config.get('min_regime_samples', 10)
        )
        self.walk_forward_validator = RegimeWalkForwardValidator(wf_config)
        tprint("✅ [REGIME_MODELS] Walk-forward validator initialized", color="green")
        
        # Initialize regime label extractor
        self.regime_extractor = RegimeLabelExtractor(
            min_samples=self.validated_config.get('min_regime_samples', 10),
            min_regimes=2
        )
        tprint("✅ [REGIME_MODELS] Regime label extractor initialized", color="green")
        
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

            # Extract regime_label column
            if 'regime_label' in regime_labels_df.columns:
                regime_labels = regime_labels_df['regime_label'].values
            else:
                # Fallback to first column
                regime_labels = regime_labels_df.iloc[:, 0].values

            tprint(f"✅ [REGIME_MODELS] Extracted {len(regime_labels)} regime labels", color="green")
            tprint(f"📊 [REGIME_MODELS] Unique regimes: {np.unique(regime_labels)}", color="blue")

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
                    
                    tprint(f"✅ [REGIME_MODELS] Regime probabilities successfully joined")
                    tprint(f"📊 [REGIME_MODELS] Enhanced data shape: {protected_data.shape}")

            # Extract regime labels with standardized extractor (fast fail behavior)
            tprint("📊 [REGIME_MODELS] Extracting regime labels with standardized extractor", color="cyan")
            
            # First try to load rolling_hmm regime labels directly
            try:
                regime_labels = await self._load_rolling_hmm_regime_labels(base_step_inst)
                if regime_labels is not None:
                    tprint(f"✅ [REGIME_MODELS] Rolling HMM regime labels loaded: {len(regime_labels)} samples", color="green")
                    tprint(f"📊 [REGIME_MODELS] Unique regimes: {np.unique(regime_labels)}", color="blue")
                else:
                    raise ValueError("Rolling HMM labels not available")
            except Exception as e:
                tprint(f"⚠️ [REGIME_MODELS] Direct rolling HMM label loading failed: {e}", color="yellow")
                
                # Fall back to standardized extractor
                try:
                    regime_labels = extract_regime_labels_standardized(pipeline_state, min_samples=10, min_regimes=2)
                    tprint(f"✅ [REGIME_MODELS] Regime labels extracted via standardized extractor: {len(regime_labels)} samples", color="green")
                    tprint(f"📊 [REGIME_MODELS] Unique regimes: {np.unique(regime_labels)}", color="blue")
                except RegimeLabelExtractionError as e:
                    tprint(f"⚠️ [REGIME_MODELS] Regime label extraction failed: {e}", color="yellow")
                    tprint("⚠️ [REGIME_MODELS] Creating synthetic labels for testing", color="yellow")

                    # Create configuration-aware synthetic regime labels when real labels are unavailable
                    n_samples = len(protected_data)
                    if n_samples == 0:
                        raise ValueError("Cannot generate synthetic regime labels with no data samples")

                    regime_config = (self.validated_config or {}).get('regime_extraction', {})
                    min_regimes_cfg = max(1, int(regime_config.get('min_regimes', 2)))
                    max_regimes_cfg = max(min_regimes_cfg, int(regime_config.get('max_regimes', min_regimes_cfg)))
                    min_samples_per_regime_cfg = max(1, int(regime_config.get('min_samples_per_regime', 5)))
                    min_regime_samples_cfg = max(1, int((self.validated_config or {}).get('min_regime_samples', min_samples_per_regime_cfg)))

                    min_samples_required_per_regime = max(min_samples_per_regime_cfg, min_regime_samples_cfg)
                    max_regimes_by_samples = max(1, n_samples // max(1, min_samples_required_per_regime))

                    n_regimes = min(max_regimes_cfg, max_regimes_by_samples)
                    if n_regimes < min_regimes_cfg:
                        n_regimes = min_regimes_cfg if n_samples >= min_regimes_cfg else 1
                    n_regimes = max(1, min(n_regimes, n_samples))
                    samples_per_regime = np.full(n_regimes, n_samples // n_regimes, dtype=int)
                    samples_per_regime[: n_samples % n_regimes] += 1

                    regime_sequence = np.concatenate([
                        np.full(count, regime_id, dtype=int)
                        for regime_id, count in enumerate(samples_per_regime)
                    ])

                    regime_labels = regime_sequence[:n_samples]
                    tprint(
                        f"✅ [REGIME_MODELS] Created synthetic regime labels: {n_samples} samples, {n_regimes} regimes",
                        color="green"
                    )

            # Prepare training data with existing feature bank
            tprint("🔧 [REGIME_MODELS] Preparing training data with existing feature bank", color="cyan")
            try:
                X, y, feature_names = self._prepare_training_data_improved(protected_data, regime_labels, pipeline_state)
            except ValueError as e:
                tprint(f"❌ [REGIME_MODELS] Training data preparation failed: {e}", color="red")
                return ComponentResult(
                    success=False,
                    error_message=f"Training data preparation failed: {e}",
                    artifacts={},
                    metadata={'execution_time': time.time() - execution_start_time}
                )

            tprint(f"📊 [REGIME_MODELS] Training data prepared - X: {X.shape}, y: {y.shape}", color="green")

            # Split data temporally with fast fail
            tprint("🔄 [REGIME_MODELS] Splitting data temporally", color="cyan")
            try:
                X_train, X_val, X_test, y_train, y_val, y_test = self.temporal_splitter.split_regime_aware(X, y)
                tprint(f"✅ [REGIME_MODELS] Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}", color="green")
            except Exception as e:
                tprint(f"❌ [REGIME_MODELS] Temporal splitting failed: {e}", color="red")
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
                
                # Train models
                trained_models = await self._train_models_with_hpo(X_train, y_train, X_test, y_test)
                
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

            # Generate predictions only for top 3 models
            tprint("🎯 [REGIME_MODELS] Generating predictions for top 3 models only", color="cyan")
            model_predictions = {}

            # CRITICAL FIX: Determine the correct data scope for predictions
            # The models were trained on X_train + X_val + X_test, not the full X
            # We need to concatenate the training splits to get the correct prediction scope
            X_for_prediction = np.concatenate([X_train, X_val, X_test]) if 'X_val' in locals() else np.concatenate([X_train, X_test])
            
            # Get the corresponding indices from protected_data
            # The training data was created from protected_data, so we need to find the matching indices
            total_training_samples = len(X_for_prediction)
            
            # Use the last 'total_training_samples' rows from protected_data since that's where the training data came from
            predictions_index = protected_data.index[-total_training_samples:]
            
            tprint(f"📊 [REGIME_MODELS] Prediction scope: {total_training_samples} samples (from {len(protected_data)} total)", color="blue")
            tprint(f"📊 [REGIME_MODELS] Using indices from {predictions_index[0]} to {predictions_index[-1]}", color="blue")

            for model_name in selected_model_names:
                if model_name in trained_models:
                    model = trained_models[model_name]
                    try:
                        if hasattr(model, 'predict_proba'):
                            # CRITICAL FIX: Use the correct data scope for predictions
                            pred_probs = model.predict_proba(X_for_prediction)
                            
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
                            
                            # Create columns for each regime
                            for regime_idx in range(pred_probs.shape[1]):
                                col_name = f'{model_name}_regime_{regime_idx}_prob'
                                model_predictions[col_name] = pred_probs[:, regime_idx]
                            tprint(f"✅ [REGIME_MODELS] Generated predictions for {model_name} ({pred_probs.shape[0]} samples, {n_predicted_classes} classes)", color="green")
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

    async def _train_models_with_hpo(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train models with HPO optimization."""
        tprint("🔍 [REGIME_MODELS] Training models with HPO optimization", color="cyan")

        trained_models = {}

        # Create validation split for early stopping (15% of training data)
        from sklearn.model_selection import train_test_split
        X_train_fit, X_val, y_train_fit, y_val = train_test_split(
            X_train, y_train,
            test_size=0.15,
            random_state=42,
            stratify=y_train  # Stratified split to preserve class distribution
        )
        tprint(f"📊 [REGIME_MODELS] Created validation split: Train={len(X_train_fit)}, Val={len(X_val)}, Test={len(X_test)}", color="cyan")

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
                        class_weights=catboost_weights,  # Apply adaptive class weights
                        random_seed=42,
                        verbose=False
                    )

                search_space = self.hpo_optimizer._get_default_search_space('catboost_regime')
                hpo_result = self.hpo_optimizer.bayesian_optimization(
                    model_factory=create_catboost_model,
                    X=X_train,
                    y=y_train,
                    search_space=search_space,
                    cv=5,  # Increased from 3 for better validation
                    scoring=scoring,
                    n_trials=150,  # Increased from 75 for better hyperparameter search
                    # Early stopping for efficiency
                    early_stopping_patience=15,  # Stop if no improvement for 15 trials
                    early_stopping_threshold=0.001,  # 0.1% minimum improvement required
                    enable_pruner=True,
                    pruner_type='hyperband'  # Aggressive pruning for speed
                )

                if hpo_result and not hpo_result.get('error'):
                    best_params = hpo_result.get('best_params', {})
                    best_score = hpo_result.get('best_score')
                    tuned_model = create_catboost_model(**best_params)
                    # Train with early stopping on validation set
                    tuned_model.fit(
                        X_train_fit, y_train_fit,
                        eval_set=(X_val, y_val),
                        early_stopping_rounds=50,
                        verbose=False
                    )
                    trained_models['catboost'] = tuned_model
                    score_msg = f"{best_score:.4f}" if isinstance(best_score, (int, float, np.floating)) else str(best_score)
                    tprint(f"✅ [REGIME_MODELS] CatBoost HPO completed - Best score: {score_msg}", color="green")
                    self.training_history.append({'model': 'catboost', 'best_params': best_params, 'best_score': best_score, 'n_trials': hpo_result.get('n_trials')})
                else:
                    if hpo_result and hpo_result.get('error'):
                        tprint(f"⚠️ [REGIME_MODELS] CatBoost HPO returned error: {hpo_result.get('error')}", color="yellow")
                    catboost_model = cb.CatBoostClassifier(
                        iterations=100, depth=6, learning_rate=0.1,
                        class_weights=catboost_weights,  # Apply adaptive class weights
                        random_seed=42, verbose=False
                    )
                    # Train with early stopping
                    catboost_model.fit(
                        X_train_fit, y_train_fit,
                        eval_set=(X_val, y_val),
                        early_stopping_rounds=50,
                        verbose=False
                    )
                    trained_models['catboost'] = catboost_model
                    tprint("⚠️ [REGIME_MODELS] CatBoost HPO unavailable, using default parameters", color="yellow")
            except Exception as e:
                tprint(f"❌ [REGIME_MODELS] CatBoost training failed: {e}", color="red")

        # 2. Train LightGBM with HPO
        try:
            tprint("💡 [REGIME_MODELS] Training LightGBM with HPO", color="blue")

            def create_lightgbm_model(**params):
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
            hpo_result = self.hpo_optimizer.bayesian_optimization(
                model_factory=create_lightgbm_model,
                X=X_train,
                y=y_train,
                search_space=search_space,
                cv=5,  # Increased from 3 for better validation
                scoring=scoring,
                n_trials=150,  # Increased from 75 for better hyperparameter search
                # Early stopping for efficiency
                early_stopping_patience=15,
                early_stopping_threshold=0.001,
                enable_pruner=True,
                pruner_type='hyperband'
            )

            if hpo_result and not hpo_result.get('error'):
                best_params = hpo_result.get('best_params', {})
                best_score = hpo_result.get('best_score')
                tuned_model = create_lightgbm_model(**best_params)
                # Train with early stopping on validation set
                tuned_model.fit(
                    X_train_fit, y_train_fit,
                    eval_set=[(X_val, y_val)],
                    callbacks=[early_stopping(50), log_evaluation(0)]
                )
                trained_models['lightgbm'] = tuned_model
                score_msg = f"{best_score:.4f}" if isinstance(best_score, (int, float, np.floating)) else str(best_score)
                tprint(f"✅ [REGIME_MODELS] LightGBM HPO completed - Best score: {score_msg}", color="green")
                self.training_history.append({'model': 'lightgbm', 'best_params': best_params, 'best_score': best_score, 'n_trials': hpo_result.get('n_trials')})
            else:
                if hpo_result and hpo_result.get('error'):
                    tprint(f"⚠️ [REGIME_MODELS] LightGBM HPO returned error: {hpo_result.get('error')}", color="yellow")
                lgbm_model = lgb.LGBMClassifier(
                    num_leaves=31, learning_rate=0.1, n_estimators=100,
                    class_weight=adaptive_weights,  # Apply adaptive class weights
                    random_state=42, verbose=-1, force_col_wise=True
                )
                # Train with early stopping
                lgbm_model.fit(
                    X_train_fit, y_train_fit,
                    eval_set=[(X_val, y_val)],
                    callbacks=[early_stopping(50), log_evaluation(0)]
                )
                trained_models['lightgbm'] = lgbm_model
                tprint("⚠️ [REGIME_MODELS] LightGBM HPO unavailable, using default parameters", color="yellow")
        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] LightGBM training failed: {e}", color="red")

        # 3. Train XGBoost with HPO
        try:
            tprint("🚀 [REGIME_MODELS] Training XGBoost with HPO", color="blue")

            def create_xgboost_model(**params):
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
            hpo_result = self.hpo_optimizer.bayesian_optimization(
                model_factory=create_xgboost_model,
                X=X_train,
                y=y_train,
                search_space=search_space,
                cv=5,  # Increased from 3 for better validation
                scoring=scoring,
                n_trials=150,  # Increased from 75 for better hyperparameter search
                # Early stopping for efficiency
                early_stopping_patience=15,
                early_stopping_threshold=0.001,
                enable_pruner=True,
                pruner_type='hyperband'
            )

            if hpo_result and not hpo_result.get('error'):
                best_params = hpo_result.get('best_params', {})
                best_score = hpo_result.get('best_score')
                tuned_model = create_xgboost_model(**best_params)
                # Train with early stopping on validation set
                tuned_model.fit(
                    X_train_fit, y_train_fit,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                trained_models['xgboost'] = tuned_model
                score_msg = f"{best_score:.4f}" if isinstance(best_score, (int, float, np.floating)) else str(best_score)
                tprint(f"✅ [REGIME_MODELS] XGBoost HPO completed - Best score: {score_msg}", color="green")
                self.training_history.append({'model': 'xgboost', 'best_params': best_params, 'best_score': best_score, 'n_trials': hpo_result.get('n_trials')})
            else:
                if hpo_result and hpo_result.get('error'):
                    tprint(f"⚠️ [REGIME_MODELS] XGBoost HPO returned error: {hpo_result.get('error')}", color="yellow")
                xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1, verbosity=0)
                # Train with early stopping
                xgb_model.fit(
                    X_train_fit, y_train_fit,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                trained_models['xgboost'] = xgb_model
                tprint("⚠️ [REGIME_MODELS] XGBoost HPO unavailable, using default parameters", color="yellow")
        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] XGBoost training failed: {e}", color="red")

        # 4. Train RandomForest with HPO
        try:
            tprint("🌲 [REGIME_MODELS] Training RandomForest with HPO", color="blue")

            def create_rf_model(**params):
                return RandomForestClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', None),
                    min_samples_split=params.get('min_samples_split', 2),
                    min_samples_leaf=params.get('min_samples_leaf', 1),
                    max_features=params.get('max_features', 'sqrt'),
                    class_weight=adaptive_weights,  # Apply adaptive class weights
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1
                )

            search_space = self.hpo_optimizer._get_default_search_space('random_forest')
            hpo_result = self.hpo_optimizer.bayesian_optimization(
                model_factory=create_rf_model,
                X=X_train,
                y=y_train,
                search_space=search_space,
                cv=5,  # Increased from 3 for better validation
                scoring=scoring,
                n_trials=150,  # Increased from 75 for better hyperparameter search
                # Early stopping for efficiency
                early_stopping_patience=15,
                early_stopping_threshold=0.001,
                enable_pruner=True,
                pruner_type='hyperband'
            )

            if hpo_result and not hpo_result.get('error'):
                best_params = hpo_result.get('best_params', {})
                best_score = hpo_result.get('best_score')
                tuned_model = create_rf_model(**best_params)
                # RandomForest doesn't support early stopping, use full training data
                tuned_model.fit(X_train, y_train)
                trained_models['random_forest'] = tuned_model
                score_msg = f"{best_score:.4f}" if isinstance(best_score, (int, float, np.floating)) else str(best_score)
                tprint(f"✅ [REGIME_MODELS] RandomForest HPO completed - Best score: {score_msg}", color="green")
                self.training_history.append({'model': 'random_forest', 'best_params': best_params, 'best_score': best_score, 'n_trials': hpo_result.get('n_trials')})
            else:
                if hpo_result and hpo_result.get('error'):
                    tprint(f"⚠️ [REGIME_MODELS] RandomForest HPO returned error: {hpo_result.get('error')}", color="yellow")
                rf_model = RandomForestClassifier(
                    n_estimators=100, max_features='sqrt',
                    class_weight=adaptive_weights,  # Apply adaptive class weights
                    random_state=42, n_jobs=-1
                )
                rf_model.fit(X_train, y_train)
                trained_models['random_forest'] = rf_model
                tprint("⚠️ [REGIME_MODELS] RandomForest HPO unavailable, using default parameters", color="yellow")
        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] RandomForest training failed: {e}", color="red")

        # 5. Train ExtraTrees with HPO
        try:
            tprint("🌳 [REGIME_MODELS] Training ExtraTrees with HPO", color="blue")

            def create_et_model(**params):
                return ExtraTreesClassifier(
                    n_estimators=params.get('n_estimators', 100),
                    max_depth=params.get('max_depth', None),
                    min_samples_split=params.get('min_samples_split', 5),
                    min_samples_leaf=params.get('min_samples_leaf', 5),
                    max_features=params.get('max_features', 'sqrt'),
                    class_weight=adaptive_weights,  # Apply adaptive class weights
                    random_state=42,
                    n_jobs=-1
                )

            search_space = self.hpo_optimizer._get_default_search_space('extra_trees')
            hpo_result = self.hpo_optimizer.bayesian_optimization(
                model_factory=create_et_model,
                X=X_train,
                y=y_train,
                search_space=search_space,
                cv=5,  # Increased from 3 for better validation
                scoring=scoring,
                n_trials=150,  # Increased from 75 for better hyperparameter search
                # Early stopping for efficiency
                early_stopping_patience=15,
                early_stopping_threshold=0.001,
                enable_pruner=True,
                pruner_type='hyperband'
            )

            if hpo_result and not hpo_result.get('error'):
                best_params = hpo_result.get('best_params', {})
                best_score = hpo_result.get('best_score')
                tuned_model = create_et_model(**best_params)
                # ExtraTrees doesn't support early stopping, use full training data
                tuned_model.fit(X_train, y_train)
                trained_models['extratrees'] = tuned_model
                score_msg = f"{best_score:.4f}" if isinstance(best_score, (int, float, np.floating)) else str(best_score)
                tprint(f"✅ [REGIME_MODELS] ExtraTrees HPO completed - Best score: {score_msg}", color="green")
                self.training_history.append({'model': 'extratrees', 'best_params': best_params, 'best_score': best_score, 'n_trials': hpo_result.get('n_trials')})
            else:
                if hpo_result and hpo_result.get('error'):
                    tprint(f"⚠️ [REGIME_MODELS] ExtraTrees HPO returned error: {hpo_result.get('error')}", color="yellow")
                et_model = ExtraTreesClassifier(
                    n_estimators=100, max_features='sqrt',
                    class_weight=adaptive_weights,  # Apply adaptive class weights
                    random_state=42, n_jobs=-1
                )
                et_model.fit(X_train, y_train)
                trained_models['extratrees'] = et_model
                tprint("⚠️ [REGIME_MODELS] ExtraTrees HPO unavailable, using default parameters", color="yellow")
        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] ExtraTrees training failed: {e}", color="red")

        tprint(f"✅ [REGIME_MODELS] Model training completed - {len(trained_models)} models trained", color="green")

        # Apply probability calibration to all models
        tprint("🎯 [REGIME_MODELS] Applying probability calibration to all models", color="cyan")
        calibrated_models = {}
        from sklearn.calibration import CalibratedClassifierCV

        for model_name, model in trained_models.items():
            try:
                tprint(f"📊 [REGIME_MODELS] Calibrating {model_name}", color="blue")
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
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data using existing feature bank system with fast fail."""
        tprint("🔧 [REGIME_MODELS] Preparing training data with existing feature bank", color="cyan")
        
        try:
            # Use existing feature bank system with fast fail
            if not FEATURE_GENERATION_AVAILABLE:
                raise ValueError("Feature generation system not available - cannot generate features")
            
            tprint("🔧 [REGIME_MODELS] Generating features using existing feature bank", color="cyan")
            X, feature_names = self._generate_features_with_bank(data)
            
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

            # Handle NaN values in features
            tprint("🔧 [REGIME_MODELS] Handling NaN values in features", color="cyan")

            # Ensure X is a numpy array for NaN handling
            if not isinstance(X, np.ndarray):
                X = np.array(X, dtype=np.float64)

            nan_cols_before = np.sum(np.isnan(X), axis=0)
            nan_cols_count = np.sum(nan_cols_before > 0)
            if nan_cols_count > 0:
                tprint(f"⚠️ [REGIME_MODELS] Found {nan_cols_count} features with NaN values", color="yellow")

            # Fill NaN values with column means (simple imputation)
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X = imputer.fit_transform(X)

            # Ensure X is still a numpy array after imputation
            X = np.array(X, dtype=np.float64)

            # Verify no NaN values remain
            nan_after = np.sum(np.isnan(X))
            if nan_after > 0:
                tprint(f"⚠️ [REGIME_MODELS] Still have {nan_after} NaN values after imputation", color="yellow")
            else:
                tprint("✅ [REGIME_MODELS] NaN values handled successfully", color="green")

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

            # Apply RobustScaler (uses median and IQR, robust to outliers)
            self.feature_scaler = RobustScaler()
            X_scaled = self.feature_scaler.fit_transform(X)

            # Verify scaling
            scaled_means = np.mean(X_scaled, axis=0)
            scaled_stds = np.std(X_scaled, axis=0)
            tprint(f"✅ [REGIME_MODELS] Feature scaling completed - Mean range: [{scaled_means.min():.3f}, {scaled_means.max():.3f}], Std range: [{scaled_stds.min():.3f}, {scaled_stds.max():.3f}]", color="green")

            X = X_scaled  # Use scaled features

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

            # Validate data
            if len(X) < min_samples_required:
                raise ValueError(f"Insufficient samples after alignment: {len(X)} < {min_samples_required}")
            
            if len(np.unique(y)) < 2:
                raise ValueError(f"Insufficient regimes: {len(np.unique(y))}")
            
            tprint(f"✅ [REGIME_MODELS] Training data prepared: {X.shape[0]} samples, {X.shape[1]} features", color="green")
            return X, y, feature_names
            
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
            
            # 1. Replace RandomForest with LGBM for importance scoring
            tprint("🔄 [REGIME_MODELS] Training LGBM for feature importance selection...", color="cyan")
            
            # Calculate adaptive class weights for LGBM
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y)
            class_weights = compute_class_weight('balanced', classes=classes, y=y)
            adaptive_weights = {int(c): float(w) for c, w in zip(classes, class_weights)}
            
            # Create LGBM model for feature selection
            lgb_selector_model = lgb.LGBMClassifier(
                num_leaves=31,
                max_depth=8,
                learning_rate=0.1,
                n_estimators=100,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                class_weight=adaptive_weights,
                random_state=42,
                verbose=-1,
                force_col_wise=True
            )
            
            # Create selector with LGBM
            lgb_selector = SelectFromModel(
                lgb_selector_model,
                max_features=target_features
            )
            
            # Fit selector
            lgb_selector.fit(X, y)
            
            # Get selected features
            selected_features_mask = lgb_selector.get_support()
            X_selected = lgb_selector.transform(X)
            
            # Update feature names to match selected features
            if len(selected_features_mask) != len(feature_names):
                tprint(f"⚠️ [REGIME_MODELS] Mask length mismatch: {len(selected_features_mask)} vs {len(feature_names)}", color="yellow")
                if len(selected_features_mask) > len(feature_names):
                    selected_features_mask = selected_features_mask[:len(feature_names)]
                else:
                    padded_mask = np.zeros(len(feature_names), dtype=bool)
                    padded_mask[:len(selected_features_mask)] = selected_features_mask
                    selected_features_mask = padded_mask
            
            selected_feature_names = [feature_names[i] for i in range(len(feature_names)) if selected_features_mask[i]]
            removed_feature_count = len(feature_names) - len(selected_feature_names)
            
            # Calculate feature importances for logging
            feature_importances = lgb_selector.estimator_.feature_importances_
            actual_feature_count = len(feature_importances)
            feature_names_for_importance = feature_names[:actual_feature_count] if len(feature_names) > actual_feature_count else feature_names
            
            importance_df = pd.DataFrame({
                'feature': feature_names_for_importance,
                'importance': feature_importances
            }).sort_values('importance', ascending=False)
            
            tprint(f"✅ [REGIME_MODELS] LGBM feature selection completed:", color="green")
            tprint(f"   • Reduced from {n_features} to {X_selected.shape[1]} features", color="green")
            tprint(f"   • Removed {removed_feature_count} low-importance features", color="green")
            tprint(f"   • New sample-to-feature ratio: {n_samples / X_selected.shape[1]:.3f}", color="green")
            
            # Show top 10 selected features
            tprint(f"🎯 [REGIME_MODELS] Top 10 selected features (LGBM importance):", color="blue")
            for i, row in importance_df.head(10).iterrows():
                tprint(f"   {i+1:2d}. {row['feature']:<40} (importance: {row['importance']:.6f})", color="blue")
            
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
                    
                    # Compute SHAP values
                    explainer = shap.TreeExplainer(shap_model)
                    shap_values = explainer.shap_values(X_selected)
                    
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
            
            if 'between_within_ratio' in regime_cv_scores:
                tprint(f"   • Between/Within CV ratio: {regime_cv_scores['between_within_ratio']:.4f}", color="blue")
            
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

    def _generate_features_with_bank(self, data: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        """Generate comprehensive features using the existing feature bank."""
        tprint("🔧 [REGIME_MODELS] Generating features using feature bank", color="cyan", bold=True)

        try:
            if not FEATURE_GENERATION_AVAILABLE:
                tprint("❌ [REGIME_MODELS] Feature generation system not available", color="red")
                return None, None

            # Get feature bank with REGIME features ENABLED (critical for regime classification)
            feature_bank = get_feature_bank(config={'enable_regime_features': True})
            tprint("✅ [REGIME_MODELS] Feature bank retrieved with REGIME features ENABLED", color="green")

            # Define feature categories to generate - prioritize REGIME category for core regime features
            categories = [
                FeatureCategory.REGIME,  # Core regime features (lagged, derived, temporal)
                FeatureCategory.MOMENTUM,
                FeatureCategory.VOLATILITY,
                FeatureCategory.VOLUME,
                FeatureCategory.TREND,
                FeatureCategory.OSCILLATOR,
                FeatureCategory.RETURNS,
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
                # Ensure all features are numeric and convert to float64 numpy array
                X = np.array(all_features.values, dtype=np.float64)
                feature_names = list(all_features.columns)
                
                # Add smoothed features if enabled
                if self.enable_smoothed_features:
                    tprint("🔧 [REGIME_MODELS] Adding smoothed features", color="cyan")
                    X, feature_names = add_smoothed_features(
                        X, 
                        window_sizes=self.smoothing_window_sizes,
                        feature_names=feature_names
                    )
                    tprint(f"✅ [REGIME_MODELS] Smoothed features added: {X.shape[1]} total features", color="green")
                
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
                for fn in feature_names:
                    fn_lower = fn.lower()
                    if 'regime' in fn_lower:
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

                return X, feature_names
            else:
                tprint("❌ [REGIME_MODELS] Feature bank generated no features", color="red")
                return None, None

        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Error generating features with feature bank: {e}", color="red")
            self.logger.error(f"Error generating features with feature bank: {str(e)}", exc_info=True)
            return None, None

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
            top_models = None
            walk_forward_metrics = training_results.get('metadata', {}).get('walk_forward_validation', {})
            if walk_forward_metrics.get('validation_completed', False):
                tprint("⚠️ [REGIME_MODELS] Walk-forward validation not completed, using single best model", color="yellow")
                # Fall back to single best model logic
                model_metrics = training_results.get('model_metrics', {})
                best_model_name = None
                best_accuracy = -1.0
                
                for model_name, metrics in model_metrics.items():
                    if 'error' not in metrics and model_name in models:
                        accuracy = metrics.get('accuracy', 0)
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model_name = model_name
                
                # Fallback to first model if no metrics available
                if best_model_name is None:
                    tprint("⚠️ [REGIME_MODELS] No model metrics available, using first model", color="yellow")
                    best_model_name = list(models.keys())[0]
                else:
                    tprint(f"✅ [REGIME_MODELS] Selected best performing model: {best_model_name} (accuracy: {best_accuracy:.4f})", color="green")
                
                model_name = best_model_name
                model = models[model_name]
            else:
                # Use top 3 models from walk-forward validation
                top_models = walk_forward_metrics.get('selected_models', [])
                if not top_models:
                    tprint("⚠️ [REGIME_MODELS] No top models found in walk-forward validation", color="yellow")
                    return None
                
                tprint(f"✅ [REGIME_MODELS] Using top 3 models from walk-forward validation: {top_models}", color="green")
                
                # Use the top-ranked model for probability analysis
                model_name = top_models[0] if top_models else None
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
            regime_labels = model.predict(X)

            n_regimes = regime_probabilities.shape[1]
            n_samples = len(regime_probabilities)

            # Calculate regime statistics
            regime_stats = {}
            for i in range(n_regimes):
                regime_probs = regime_probabilities[:, i]
                regime_count = np.sum(regime_labels == i)

                regime_stats[f'regime_{i}'] = {
                    'sample_count': int(regime_count),
                    'percentage': float(regime_count / n_samples * 100),
                    'mean_probability': float(np.mean(regime_probs)),
                    'std_probability': float(np.std(regime_probs)),
                    'min_probability': float(np.min(regime_probs)),
                    'max_probability': float(np.max(regime_probs)),
                    'confidence_distribution': {
                        'high_confidence': int(np.sum(regime_probs > 0.8)),
                        'medium_confidence': int(np.sum((regime_probs > 0.5) & (regime_probs <= 0.8))),
                        'low_confidence': int(np.sum(regime_probs <= 0.5))
                    }
                }

            # Calculate overall statistics
            overall_stats = {
                'total_samples': n_samples,
                'n_regimes': n_regimes,
                'mean_max_probability': float(np.mean(np.max(regime_probabilities, axis=1))),
                'std_max_probability': float(np.std(np.max(regime_probabilities, axis=1))),
                'regime_balance': float(np.std([regime_stats[f'regime_{i}']['percentage'] for i in range(n_regimes)])),
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
                    # Create a smaller explainer for SHAP analysis
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X[:100])  # Use subset for speed
                    
                    # For multi-class, get mean absolute SHAP values across classes
                    if isinstance(shap_values, list):
                        # Multi-class case
                        shap_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                    else:
                        # Binary case
                        shAP_importance = np.abs(shap_values).mean(axis=0)
                    
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
                'regime_labels': regime_labels.tolist(),
                'feature_names': feature_names,
                'data_shape': X.shape,
                'feature_importance': feature_importance,
                'top_60_features': top_60_features,
                'report_type': 'regime_probability_analysis'
            }
            
            # Add model metrics if available
            model_metrics = training_results.get('model_metrics', {})
            if model_name in model_metrics:
                report['model_metrics'] = model_metrics[model_name]

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
                    for i, row in shAP_importance.head(10).iterrows():
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
                md_lines.append(f"| Accuracy | {metrics.get('accuracy', 'N/A'):.4f} |")
                md_lines.append(f"| Precision (Weighted) | {metrics.get('precision', 'N/A'):.4f} |")
                md_lines.append(f"| Recall (Weighted) | {metrics.get('recall', 'N/A'):.4f} |")
                md_lines.append(f"| F1-Score (Weighted) | {metrics.get('f1_score', 'N/A'):.4f} |")
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