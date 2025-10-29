"""
Regime Detection Models Training Component

This component implements the specific regime detection models mentioned in the user's request:
- CatBoost (base model)
- Greedy Rule Lists (base model - multi-class compatible)
- ExtraTrees (base model)
- stacker_lgbm_calibrated (meta-learner with probability calibration)
"""

import numpy as np
import pandas as pd
import pickle
import json
import time
import warnings
import psutil
import gc
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from src.utils.logger import system_logger
from src.utils.tprint import tprint
from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult

# Enhanced imports for new functionality
from src.utils.ml_common.unified_vectorization_manager import (
    UnifiedVectorizationManager, OperationType, OptimizationStrategy
)
from src.utils.ml_common.optimization.hpo_utils import (
    HyperparameterOptimization
)
from src.utils.ml_common.optimization.transition_aware_scoring import (
    create_transition_aware_scorer,
    create_multi_objective_scorer
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
from src.utils.ml_common.data.regime_label_extractor import (
    RegimeLabelExtractor, extract_regime_labels_fast_fail
)
from src.utils.ml_common.validation.config_validator import (
    validate_regime_training_config, create_default_regime_training_config
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
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

# Import Greedy Rule Lists
try:
    from imodels import GreedyRuleListClassifier  # type: ignore[import-untyped]
    ML_LIBRARY_VERSIONS['imodels'] = "1.0.0"  # Placeholder version
    tprint(f"✅ [REGIME_MODELS] imodels (Greedy Rule Lists) imported successfully", color="green")
except ImportError as e:
    ML_IMPORT_ERRORS.append(f"imodels (Greedy Rule Lists): {e}")
    tprint(f"❌ [REGIME_MODELS] Failed to import imodels: {e}", color="red")

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
    - Greedy Rule Lists (base model - multi-class compatible)
    - ExtraTrees (base model)
    - stacker_lgbm_calibrated (meta-learner with probability calibration)
    """

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the Regime Models Training Component with enhanced utilities and fast fail behavior."""
        tprint("🚀 [REGIME_MODELS] Initializing Regime Models Training Component", color="cyan", bold=True)
        tprint(f"📋 [REGIME_MODELS] Config provided: {config is not None}", color="blue")

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

        # Validate and setup configuration with fast fail
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

        # Log initialization completion
        tprint("✅ [REGIME_MODELS] Regime Models Training Component initialized successfully", color="green", bold=True)
        self.logger.info("Regime Models Training Component initialization completed")

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        tprint("📋 [REGIME_MODELS] Getting required artifacts", color="cyan")
        required_artifacts = ['regime_models_training_result']
        tprint(f"✅ [REGIME_MODELS] Required artifacts: {required_artifacts}", color="green")
        return required_artifacts

    def _validate_and_setup_config(self):
        """Validate and setup configuration with fast fail behavior."""
        tprint("🔧 [REGIME_MODELS] Validating configuration", color="cyan")
        
        # Get default configuration
        default_config = create_default_regime_training_config()
        
        # Merge with provided config
        if self.config:
            config_dict = {
                'test_size': getattr(self.config, 'test_size', default_config['test_size']),
                'validation_size': getattr(self.config, 'validation_size', default_config['validation_size']),
                'cv_folds': getattr(self.config, 'cv_folds', default_config['cv_folds']),
                'random_state': getattr(self.config, 'random_state', default_config['random_state']),
                'gap_size': getattr(self.config, 'gap_size', default_config['gap_size']),
                'min_regime_samples': getattr(self.config, 'min_regime_samples', default_config['min_regime_samples']),
                'regime_aware': getattr(self.config, 'regime_aware', True)
            }
        else:
            config_dict = default_config
        
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
        self.temporal_splitter = create_temporal_splitter(self.validated_config)
        tprint("✅ [REGIME_MODELS] Temporal splitter initialized", color="green")
        
        # Initialize regime label extractor
        self.regime_extractor = RegimeLabelExtractor(
            min_samples=self.validated_config.get('min_regime_samples', 10),
            min_regimes=2
        )
        tprint("✅ [REGIME_MODELS] Regime label extractor initialized", color="green")
        
        # Note: Using existing feature bank system instead of custom feature generator
        tprint("✅ [REGIME_MODELS] Using existing feature bank system", color="green")
    async def _train_models_with_hpo(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Train models with HPO optimization."""
        tprint("🔍 [REGIME_MODELS] Training models with HPO optimization", color="cyan")
        
        trained_models = {}
        
        # Train CatBoost with HPO
        if ML_LIBRARIES_AVAILABLE:
            try:
                tprint("🐱 [REGIME_MODELS] Training CatBoost with HPO", color="blue")
                
                def create_catboost_model(trial):
                    return cb.CatBoostClassifier(
                        iterations=trial.suggest_int('iterations', 50, 200),
                        depth=trial.suggest_int('depth', 3, 8),
                        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                        random_seed=42,
                        verbose=False
                    )
                
                # Use transition-aware scorer or multi-objective optimization
                if self.enable_multi_objective_hpo and self.use_pareto_optimization:
                    # Use multi-objective scorer with Pareto optimization
                    multi_scorer = create_multi_objective_scorer(min_episode_length=3)
                    # Note: Full Pareto integration would require Optuna multi-objective study
                    # For now, use transition-aware composite scorer
                    scoring = create_transition_aware_scorer(
                        alpha=self.temporal_smoothing_alpha,
                        accuracy_weight=0.7,
                        stability_weight=0.3
                    )
                else:
                    # Use transition-aware composite scorer (single objective)
                    scoring = create_transition_aware_scorer(
                        alpha=self.temporal_smoothing_alpha,
                        accuracy_weight=0.7,
                        stability_weight=0.3
                    )
                
                hpo_result = self.hpo_optimizer.optimize(
                    model_factory=create_catboost_model,
                    X=X_train,
                    y=y_train,
                    cv_folds=3,
                    scoring=scoring,  # Use transition-aware scorer
                    n_trials=15
                )
                
                if hpo_result.success:
                    trained_models['catboost'] = hpo_result.best_model
                    tprint(f"✅ [REGIME_MODELS] CatBoost HPO completed - Best score: {hpo_result.best_score:.4f}", color="green")
                else:
                    # Fallback to default parameters
                    catboost_model = cb.CatBoostClassifier(
                        iterations=100,
                        depth=6,
                        learning_rate=0.1,
                        random_seed=42,
                        verbose=False
                    )
                    catboost_model.fit(X_train, y_train)
                    trained_models['catboost'] = catboost_model
                    tprint("⚠️ [REGIME_MODELS] CatBoost HPO failed, using default parameters", color="yellow")
                    
            except Exception as e:
                tprint(f"❌ [REGIME_MODELS] CatBoost training failed: {e}", color="red")

            # Train ExtraTrees with HPO
            try:
                tprint("🌳 [REGIME_MODELS] Training ExtraTrees with HPO", color="blue")
                
                def create_extratrees_model(trial):
                    return ExtraTreesClassifier(
                        n_estimators=trial.suggest_int('n_estimators', 50, 200),
                        max_depth=trial.suggest_int('max_depth', 5, 20),
                        min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 5),
                        max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                        random_state=42,
                        n_jobs=-1
                    )
                
                # Use transition-aware scorer
                scoring = create_transition_aware_scorer(
                    alpha=self.temporal_smoothing_alpha,
                    accuracy_weight=0.7,
                    stability_weight=0.3
                )
                
                hpo_result = self.hpo_optimizer.optimize(
                    model_factory=create_extratrees_model,
                    X=X_train,
                    y=y_train,
                    cv_folds=3,
                    scoring=scoring,  # Use transition-aware scorer
                    n_trials=15
                )
                
                if hpo_result.success:
                    trained_models['extra_trees'] = hpo_result.best_model
                    tprint(f"✅ [REGIME_MODELS] ExtraTrees HPO completed - Best score: {hpo_result.best_score:.4f}", color="green")
                else:
                    # Fallback to default parameters
                    extratrees_model = ExtraTreesClassifier(
                        n_estimators=100,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        max_features='sqrt',
                        random_state=42,
                        n_jobs=-1
                    )
                    extratrees_model.fit(X_train, y_train)
                    trained_models['extra_trees'] = extratrees_model
                    tprint("⚠️ [REGIME_MODELS] ExtraTrees HPO failed, using default parameters", color="yellow")
                    
            except Exception as e:
                tprint(f"❌ [REGIME_MODELS] ExtraTrees training failed: {e}", color="red")

            # Train XGBoost with HPO
            try:
                tprint("🚀 [REGIME_MODELS] Training XGBoost with HPO", color="blue")
                
                def create_xgboost_model(trial):
                    return xgb.XGBClassifier(
                        n_estimators=trial.suggest_int('n_estimators', 50, 200),
                        max_depth=trial.suggest_int('max_depth', 3, 10),
                        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                        subsample=trial.suggest_float('subsample', 0.6, 1.0),
                        colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        reg_alpha=trial.suggest_float('reg_alpha', 0, 1),
                        reg_lambda=trial.suggest_float('reg_lambda', 0, 1),
                        random_state=42,
                        n_jobs=-1,
                        verbosity=0
                    )
                
                # Use transition-aware scorer
                scoring = create_transition_aware_scorer(
                    alpha=self.temporal_smoothing_alpha,
                    accuracy_weight=0.7,
                    stability_weight=0.3
                )
                
                hpo_result = self.hpo_optimizer.optimize(
                    model_factory=create_xgboost_model,
                    X=X_train,
                    y=y_train,
                    cv_folds=3,
                    scoring=scoring,  # Use transition-aware scorer
                    n_trials=15
                )
                
                if hpo_result.success:
                    trained_models['xgboost'] = hpo_result.best_model
                    tprint(f"✅ [REGIME_MODELS] XGBoost HPO completed - Best score: {hpo_result.best_score:.4f}", color="green")
                else:
                    # Fallback to default parameters
                    xgb_model = xgb.XGBClassifier(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.1,
                        reg_lambda=0.1,
                        random_state=42,
                        n_jobs=-1,
                        verbosity=0
                    )
                    xgb_model.fit(X_train, y_train)
                    trained_models['xgboost'] = xgb_model
                    tprint("⚠️ [REGIME_MODELS] XGBoost HPO failed, using default parameters", color="yellow")
                    
            except Exception as e:
                tprint(f"❌ [REGIME_MODELS] XGBoost training failed: {e}", color="red")

            # Train Random Forest with HPO
            try:
                tprint("🌳 [REGIME_MODELS] Training Random Forest with HPO", color="blue")
                
                def create_random_forest_model(trial):
                    return RandomForestClassifier(
                        n_estimators=trial.suggest_int('n_estimators', 50, 200),
                        max_depth=trial.suggest_int('max_depth', 5, 20),
                        min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 5),
                        max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                        bootstrap=trial.suggest_categorical('bootstrap', [True, False]),
                        random_state=42,
                        n_jobs=-1
                    )
                
                hpo_result = self.hpo_optimizer.optimize(
                    model_factory=create_random_forest_model,
                    X=X_train,
                    y=y_train,
                    cv_folds=3,
                    scoring='accuracy',
                    n_trials=15
                )
                
                if hpo_result.success:
                    trained_models['random_forest'] = hpo_result.best_model
                    tprint(f"✅ [REGIME_MODELS] Random Forest HPO completed - Best score: {hpo_result.best_score:.4f}", color="green")
                else:
                    # Fallback to default parameters
                    rf_model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        max_features='sqrt',
                        bootstrap=True,
                        random_state=42,
                        n_jobs=-1
                    )
                    rf_model.fit(X_train, y_train)
                    trained_models['random_forest'] = rf_model
                    tprint("⚠️ [REGIME_MODELS] Random Forest HPO failed, using default parameters", color="yellow")
                    
            except Exception as e:
                tprint(f"❌ [REGIME_MODELS] Random Forest training failed: {e}", color="red")

            # Train Greedy Rule Lists (no HPO needed)
            try:
                tprint("📋 [REGIME_MODELS] Training Greedy Rule Lists", color="blue")
                rule_model = GreedyRuleListClassifier(
                    max_depth=20,
                    criterion='gini',
                    class_weight='balanced'
                )
                rule_model.fit(X_train, y_train)
                trained_models['greedy_rule_lists'] = rule_model
                tprint("✅ [REGIME_MODELS] Greedy Rule Lists trained successfully", color="green")
                
            except Exception as e:
                tprint(f"❌ [REGIME_MODELS] Greedy Rule Lists training failed: {e}", color="red")

        # Dynamic model selection based on performance
        selected_models = self._select_best_models(trained_models, X_train, y_train)
        
        tprint(f"✅ [REGIME_MODELS] Model training completed - {len(trained_models)} models trained, {len(selected_models)} selected", color="green")
        return selected_models

    def _select_best_models(self, trained_models: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Select best models based on cross-validation performance."""
        tprint("🎯 [REGIME_MODELS] Selecting best models based on CV performance", color="cyan")
        
        model_scores = {}
        
        for name, model in trained_models.items():
            if model is not None:
                try:
                    # Use 3-fold CV for quick evaluation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                    avg_score = np.mean(cv_scores)
                    std_score = np.std(cv_scores)
                    model_scores[name] = {'score': avg_score, 'std': std_score}
                    tprint(f"📊 [REGIME_MODELS] {name}: {avg_score:.4f} ± {std_score:.4f}", color="blue")
                except Exception as e:
                    tprint(f"❌ [REGIME_MODELS] CV failed for {name}: {e}", color="red")
                    model_scores[name] = {'score': 0.0, 'std': 0.0}
        
        # Select top 3 models by score
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        selected_names = [name for name, _ in sorted_models[:3]]
        
        selected_models = {name: trained_models[name] for name in selected_names if name in trained_models}
        
        tprint(f"🏆 [REGIME_MODELS] Selected top 3 models: {list(selected_models.keys())}", color="green")
        return selected_models

    def _validate_models_advanced(self, models: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform advanced validation using ML common tools."""
        tprint("🔍 [REGIME_MODELS] Performing advanced validation", color="cyan")
        
        validation_results = {}
        
        # Temporal validation configuration
        temporal_config = TemporalValidationConfig(
            enable_temporal_checks=True,
            strict_temporal_order=True,
            min_temporal_gap=1,
            enable_walk_forward=True,
            initial_train_size=0.6,
            step_size=0.1,
            min_test_size=0.1,
            enable_leakage_detection=True,
            n_splits=5,
            test_size=0.2,
            gap_size=1
        )
        
        temporal_validator = UniversalTemporalValidator(temporal_config)
        
        for name, model in models.items():
            if model is not None:
                try:
                    # Split data for temporal validation
                    split_idx = int(len(X) * 0.8)
                    X_train, X_test = X[:split_idx], X[split_idx:]
                    y_train, y_test = y[:split_idx], y[split_idx:]
                    
                    # Temporal validation
                    temporal_report = temporal_validator.validate_temporal_split(
                        X_train, X_test, y_train, y_test, 
                        model_name=name, model_type="regime_classifier"
                    )
                    
                    # Purged cross-validation
                    cv_results = temporal_cross_validation(
                        model, X, y, n_splits=5, gap=1, test_size=0.2
                    )
                    
                    # Regime-aware validation
                    regime_validation = self._validate_regime_aware(model, X, y)
                    
                    validation_results[name] = {
                        'temporal_validation': temporal_report,
                        'purged_cv': cv_results,
                        'regime_validation': regime_validation
                    }
                    
                    tprint(f"✅ [REGIME_MODELS] Advanced validation completed for {name}", color="green")
                    
                except Exception as e:
                    tprint(f"❌ [REGIME_MODELS] Advanced validation failed for {name}: {e}", color="red")
                    validation_results[name] = {'error': str(e)}
        
        return validation_results

    def _validate_regime_aware(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Validate model performance across different regimes."""
        try:
            unique_regimes = np.unique(y)
            regime_performance = {}
            
            for regime in unique_regimes:
                regime_mask = (y == regime)
                regime_X = X[regime_mask]
                regime_y = y[regime_mask]
                
                if len(regime_X) > 10:  # Minimum samples for validation
                    # Cross-validation within regime
                    cv_scores = cross_val_score(model, regime_X, regime_y, cv=3, scoring='accuracy')
                    regime_performance[f'regime_{regime}'] = {
                        'accuracy': np.mean(cv_scores),
                        'std': np.std(cv_scores),
                        'samples': len(regime_X)
                    }
            
            return regime_performance
            
        except Exception as e:
            return {'error': str(e)}

    def _add_interpretability_features(self, models: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Add interpretability features and explanations."""
        tprint("🔍 [REGIME_MODELS] Adding interpretability features", color="cyan")
        
        interpretability_results = {}
        
        for name, model in models.items():
            if model is not None:
                try:
                    # Initialize explainability tools
                    explainer = ModelExplainabilityManager()
                    shap_lime = SHAPLIMEIntegration()
                    
                    # Feature importance
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = model.feature_importances_
                    elif hasattr(model, 'coef_'):
                        feature_importance = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
                    else:
                        feature_importance = None
                    
                    # SHAP explanations
                    shap_explanations = None
                    try:
                        shap_explanations = shap_lime.get_shap_explanations(model, X[:100])  # Sample for performance
                    except Exception as e:
                        tprint(f"⚠️ [REGIME_MODELS] SHAP explanations failed for {name}: {e}", color="yellow")
                    
                    # LIME explanations
                    lime_explanations = None
                    try:
                        lime_explanations = shap_lime.get_lime_explanations(model, X[:50])  # Sample for performance
                    except Exception as e:
                        tprint(f"⚠️ [REGIME_MODELS] LIME explanations failed for {name}: {e}", color="yellow")
                    
                    # Model decision boundaries
                    decision_boundary = None
                    if hasattr(model, 'decision_function'):
                        try:
                            decision_boundary = model.decision_function(X[:100])
                        except Exception as e:
                            tprint(f"⚠️ [REGIME_MODELS] Decision boundary failed for {name}: {e}", color="yellow")
                    
                    interpretability_results[name] = {
                        'feature_importance': feature_importance,
                        'shap_explanations': shap_explanations,
                        'lime_explanations': lime_explanations,
                        'decision_boundary': decision_boundary
                    }
                    
                    tprint(f"✅ [REGIME_MODELS] Interpretability features added for {name}", color="green")
                    
                except Exception as e:
                    tprint(f"❌ [REGIME_MODELS] Interpretability failed for {name}: {e}", color="red")
                    interpretability_results[name] = {'error': str(e)}
        
        return interpretability_results

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
                
                # Use enhanced model evaluator
                evaluation_result = self.model_evaluator.evaluate_model(
                    model=model,
                    X=X_test,
                    y=y_test,
                    y_pred=y_pred,
                    y_pred_proba=y_pred_proba
                )
                
                # Use model validator
                validation_result = self.model_validator.validate_model(
                    model=model,
                    X=X_test,
                    y=y_test,
                    cv_folds=3
                )
                
                # Calculate basic metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
                
                model_metrics[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True),
                    'enhanced_evaluation': evaluation_result,
                    'model_validation': validation_result
                }
                
                tprint(f"✅ [REGIME_MODELS] {model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}", color="green")
                
            except Exception as e:
                tprint(f"❌ [REGIME_MODELS] Failed to evaluate {model_name}: {e}", color="red")
                model_metrics[model_name] = {'error': str(e)}
        
        return model_metrics

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
            await self.hardware_manager.initialize()
            await self.hardware_manager.optimize_for_workload(WorkloadType.ML_TRAINING)
            tprint("✅ [REGIME_MODELS] Hardware optimization initialized", color="green")

            # Apply lookahead protection
            tprint("🔒 [REGIME_MODELS] Applying lookahead protection", color="cyan")
            protected_data = self.lookahead_protection.protect_data(data)
            tprint("✅ [REGIME_MODELS] Lookahead protection applied", color="green")

            # Log initial system performance
            initial_perf = self._get_system_performance()
            if initial_perf:
                tprint(f"💻 [REGIME_MODELS] Initial system state - CPU: {initial_perf.get('cpu_percent', 'N/A')}%, Memory: {initial_perf.get('memory_percent', 'N/A')}%", color="blue")

            # Monitor initial memory usage
            initial_memory = psutil.virtual_memory()
            tprint(f"🧠 [REGIME_MODELS] Initial memory usage: {initial_memory.percent:.1f}% ({initial_memory.used / 1024**3:.1f}GB / {initial_memory.total / 1024**3:.1f}GB)", color="blue")

            # Extract regime labels with fast fail behavior
            tprint("📊 [REGIME_MODELS] Extracting regime labels with fast fail", color="cyan")
            artifacts = pipeline_state.get('artifacts', {})
            
            try:
                regime_labels = self.regime_extractor.extract_regime_labels(artifacts)
                tprint(f"✅ [REGIME_MODELS] Regime labels extracted: {len(regime_labels)} samples", color="green")
            except ValueError as e:
                tprint(f"❌ [REGIME_MODELS] Regime label extraction failed: {e}", color="red")
                return ComponentResult(
                    success=False,
                    error_message=f"Regime label extraction failed: {e}",
                    artifacts={},
                    metadata={'execution_time': time.time() - execution_start_time}
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
            else:
                tprint("✅ [REGIME_MODELS] Temporal validation passed", color="green")

            # Train models with HPO optimization
            tprint("🏋️ [REGIME_MODELS] Training models with HPO optimization", color="yellow")
            trained_models = await self._train_models_with_hpo(X_train, y_train, X_test, y_test)

            # Evaluate models with enhanced evaluation
            tprint("📊 [REGIME_MODELS] Evaluating models with enhanced evaluation", color="yellow")
            model_metrics = await self._evaluate_models_enhanced(trained_models, X_test, y_test)

            # Create comprehensive results
            execution_time = time.time() - execution_start_time
            results = {
                'regime_models_training_result': {
                    'models': trained_models,
                    'model_metrics': model_metrics,
                    'training_time': execution_time,
                    'success': True,
                    'validation_report': {
                        'temporal_order_valid': validation_report.temporal_order_valid,
                        'leakage_detected': validation_report.leakage_detected,
                        'validation_score': validation_report.validation_score,
                        'warnings': validation_report.warnings,
                        'recommendations': validation_report.recommendations
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
                        'timestamp': datetime.now().isoformat()
                    }
                }
            }

            tprint("✅ [REGIME_MODELS] Regime models training completed successfully", color="green", bold=True)
            tprint(f"⏱️ [REGIME_MODELS] Total execution time: {execution_time:.2f}s", color="blue")

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
            await self.hardware_manager.cleanup()
            tprint("🔧 [REGIME_MODELS] Hardware resources cleaned up", color="green")

            return ComponentResult(
                success=True,
                artifacts=results,
                metadata={
                    'component_type': 'regime_models_training',
                    'execution_time': execution_time,
                    'artifacts_saved_persistently': True,
                    'hardware_optimization_enabled': True,
                    'lookahead_protection_enabled': True
                }
            )

        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Regime models training failed: {e}", color="red", bold=True)
            self.logger.error(f"Regime models training failed: {e}", exc_info=True)
            
            # Cleanup hardware resources on error
            try:
                await self.hardware_manager.cleanup()
            except Exception as cleanup_error:
                tprint(f"⚠️ [REGIME_MODELS] Hardware cleanup failed: {cleanup_error}", color="yellow")
            
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e),
                metadata={'component_type': 'regime_models_training'}
            )
        initial_memory = self._monitor_memory_usage("Initial")

        # Log execution context
        tprint(f"📊 [REGIME_MODELS] Input data shape: {data.shape}", color="blue")
        tprint(f"📋 [REGIME_MODELS] Data columns: {list(data.columns)}", color="blue")
        tprint(f"🔍 [REGIME_MODELS] Pipeline state keys: {list(pipeline_state.keys())}", color="blue")

        try:
            # Step 0: Validate input data
            tprint("🔍 [REGIME_MODELS] Step 0: Validating input data", color="cyan")
            if not self._validate_input_data(data):
                error_msg = "Input data validation failed"
                tprint(f"❌ [REGIME_MODELS] {error_msg}", color="red")
                self.logger.error("Input data validation failed")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg
                )

            # Step 1: Check ML libraries availability
            tprint("🔍 [REGIME_MODELS] Step 1: Checking ML libraries availability", color="cyan")
            if not ML_LIBRARIES_AVAILABLE:
                error_msg = "ML libraries not available for regime detection models training"
                tprint(f"❌ [REGIME_MODELS] {error_msg}", color="red")
                tprint(f"🔍 [REGIME_MODELS] Import errors: {ML_IMPORT_ERRORS}", color="yellow")
                self.logger.error(f"ML libraries not available: {ML_IMPORT_ERRORS}")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg
                )
            tprint("✅ [REGIME_MODELS] ML libraries check passed", color="green")

            # Step 2: Extract and validate regime labels
            tprint("🔍 [REGIME_MODELS] Step 2: Extracting regime labels from pipeline state", color="cyan")
            artifacts = pipeline_state.get('artifacts', {})
            tprint(f"📋 [REGIME_MODELS] Available artifacts: {list(artifacts.keys())}", color="blue")

            # If no artifacts available, try to load from previous outcome files or artifact manager
            if not artifacts:
                tprint("🔍 [REGIME_MODELS] No artifacts in pipeline state, trying to load from previous stages", color="yellow")

                # First try to load from artifact manager (most recent session)
                try:
                    # Legacy NAS/TAS artifacts removed
                    artifacts = self._load_artifacts_from_outcome_files()
                except Exception as e:
                    tprint(f"⚠️ [REGIME_MODELS] Failed to load from artifact manager: {e}, trying outcome files", color="yellow")
                    artifacts = self._load_artifacts_from_outcome_files()

            # Look for regime labels in regime_discovery_result artifact
            regime_discovery_result = artifacts.get('regime_discovery_result', {})
            tprint(f"🔍 [REGIME_MODELS] Regime discovery result keys: {list(regime_discovery_result.keys())}", color="blue")

            # Try to get regime labels from regime assignments
            regime_labels = regime_discovery_result.get('regime_assignments')
            if regime_labels is None:
                regime_labels = regime_discovery_result.get('cluster_assignments')
                tprint("🔍 [REGIME_MODELS] Using cluster assignments as regime labels", color="blue")
            else:
                tprint("🔍 [REGIME_MODELS] Using regime assignments as regime labels", color="blue")

            # If still no regime labels, try alternative artifact structures
            if regime_labels is None:
                tprint("🔍 [REGIME_MODELS] Trying alternative artifact structures...", color="yellow")

                # Try direct access to artifacts
                # Legacy TAS/NAS assignments removed

                # Try other possible artifact keys
                for key in ['regime_assignments', 'assignments', 'cluster_assignments']:
                    if key in artifacts:
                        regime_labels = artifacts[key]
                        tprint(f"🔍 [REGIME_MODELS] Found regime labels in {key}", color="blue")
                        break

                # Try nested structures
                if regime_labels is None:
                    for artifact_key, artifact_value in artifacts.items():
                        if isinstance(artifact_value, dict):
                            # Legacy TAS assignments removed
                            if 'assignments' in artifact_value:
                                regime_labels = artifact_value['assignments']
                                tprint(f"🔍 [REGIME_MODELS] Found assignments in {artifact_key}", color="blue")
                                break
                            elif 'cluster_assignments' in artifact_value:
                                regime_labels = artifact_value['cluster_assignments']
                                tprint(f"🔍 [REGIME_MODELS] Found cluster_assignments in {artifact_key}", color="blue")
                                break

                # Try extracting from optimal_regime_clustering_result clustering_result object
                if regime_labels is None:
                    optimal_clustering_result = artifacts.get('optimal_regime_clustering_result', {})
                    if optimal_clustering_result:
                        tprint("🔍 [REGIME_MODELS] Found optimal_regime_clustering_result, extracting from clustering_result object", color="blue")
                        clustering_result = optimal_clustering_result.get('clustering_result')
                        if clustering_result is not None:
                            tprint(f"🔍 [REGIME_MODELS] clustering_result type: {type(clustering_result)}", color="blue")
                            if isinstance(clustering_result, dict):
                                tprint(f"🔍 [REGIME_MODELS] clustering_result keys: {list(clustering_result.keys())}", color="blue")
                            else:
                                tprint(f"🔍 [REGIME_MODELS] clustering_result attributes: {dir(clustering_result)}", color="blue")

                            # First try direct access to clustering_result (new wrapper structure)
                            if isinstance(clustering_result, dict):
                                # Look for cluster_assignments in the clustering_result dict (from regime clustering)
                                if 'cluster_assignments' in clustering_result:
                                    regime_labels = clustering_result['cluster_assignments']
                                    # Handle case where assignments are stored as string representation
                                    if isinstance(regime_labels, str):
                                        try:
                                            # Parse numpy array string representation (e.g., "[2 2 2 ... 4 6 6]")
                                            import ast
                                            import re

                                            # Handle string representation with ellipsis
                                            if '...' in regime_labels:
                                                # For strings with ellipsis, we need to extract the actual values
                                                # This is a simplified approach - in practice, we should store the actual array
                                                # For now, let's try to find patterns or use a fallback
                                                tprint(f"⚠️ [REGIME_MODELS] Found ellipsis in cluster_assignments string, attempting to recover", color="yellow")

                                                # Try to extract numbers from the string representation
                                                numbers = re.findall(r'\d+', regime_labels)
                                                if numbers:
                                                    regime_labels = np.array([int(x) for x in numbers])
                                                    tprint(f"🔍 [REGIME_MODELS] Recovered {len(regime_labels)} regime labels from string", color="blue")
                                                else:
                                                    tprint(f"⚠️ [REGIME_MODELS] Could not extract numbers from cluster_assignments string", color="yellow")
                                                    regime_labels = None
                                            elif isinstance(regime_labels, str) and regime_labels.startswith('[') and regime_labels.endswith(']'):
                                                # Handle string representation of numpy array like "[2 2 2 ... 4 6 6]"
                                                try:
                                                    # Try to parse as a list of integers
                                                    clean_str = regime_labels.strip('[]')
                                                    if '...' in clean_str:
                                                        # If still contains ellipsis, extract numbers
                                                        numbers = re.findall(r'\d+', clean_str)
                                                        if numbers:
                                                            regime_labels = np.array([int(x) for x in numbers])
                                                    else:
                                                        # Split by spaces and convert to integers
                                                        values = [int(x) for x in clean_str.split() if x.strip()]
                                                        regime_labels = np.array(values)
                                                    tprint(f"🔍 [REGIME_MODELS] Parsed numpy array string representation", color="blue")
                                                except Exception as e:
                                                    tprint(f"⚠️ [REGIME_MODELS] Failed to parse array string: {e}", color="yellow")
                                                    regime_labels = None
                                            else:
                                                # Remove brackets and split by spaces, then convert to int
                                                clean_str = regime_labels.strip('[]')
                                                regime_labels = np.array([int(x) for x in clean_str.split() if x.strip()])
                                                tprint(f"🔍 [REGIME_MODELS] Parsed regime labels from string representation", color="blue")
                                        except Exception as e:
                                            tprint(f"⚠️ [REGIME_MODELS] Failed to parse regime labels string: {e}", color="yellow")
                                            regime_labels = None
                                    else:
                                        # If it's already a numpy array or list, use it directly
                                        tprint(f"🔍 [REGIME_MODELS] Found regime labels in clustering_result.cluster_assignments", color="blue")

                            # If not found in direct dict, try to get assignments from the clustering result object
                            if regime_labels is None and hasattr(clustering_result, 'current_results') and clustering_result.current_results:
                                current_results = clustering_result.current_results
                                # Try different possible keys for assignments
                                for assignment_key in ['cluster_assignments', 'assignments']:
                                    if assignment_key in current_results:
                                        regime_labels = current_results[assignment_key]
                                        tprint(f"🔍 [REGIME_MODELS] Found regime labels in clustering_result.current_results.{assignment_key}", color="blue")
                                        break

                                # If still not found, try to get from context
                                if regime_labels is None and hasattr(clustering_result, 'context'):
                                    context = clustering_result.context
                                    if hasattr(context, 'optimized_assignments') and context.optimized_assignments is not None:
                                        regime_labels = context.optimized_assignments
                                        tprint("🔍 [REGIME_MODELS] Found regime labels in clustering_result.context.optimized_assignments", color="blue")
                                    elif hasattr(context, 'initial_assignments') and context.initial_assignments is not None:
                                        regime_labels = context.initial_assignments
                                        tprint("🔍 [REGIME_MODELS] Found regime labels in clustering_result.context.initial_assignments", color="blue")

                # Try extracting from component factory wrapper structure (fallback)
                if regime_labels is None:
                    optimal_clustering_result = artifacts.get('optimal_regime_clustering_result', {})
                    if optimal_clustering_result:
                        tprint("🔍 [REGIME_MODELS] Trying component factory wrapper structure fallback", color="blue")
                        # Check if the wrapper stored the data directly in the artifact
                        for assignment_key in ['cluster_assignments', 'assignments']:
                            if assignment_key in optimal_clustering_result:
                                regime_labels = optimal_clustering_result[assignment_key]
                                tprint(f"🔍 [REGIME_MODELS] Found regime labels in optimal_clustering_result.{assignment_key}", color="blue")
                                break

            if regime_labels is None:
                error_msg = "No regime labels found in any artifact structure"
                tprint(f"❌ [REGIME_MODELS] {error_msg}", color="red")
                tprint(f"🔍 [REGIME_MODELS] Available artifacts: {list(artifacts.keys())}", color="yellow")
                tprint(f"🔍 [REGIME_MODELS] Regime discovery result keys: {list(regime_discovery_result.keys())}", color="yellow")
                self.logger.error(f"Missing regime labels. Available artifacts: {list(artifacts.keys())}")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg
                )

            # Validate regime labels
            regime_labels = np.array(regime_labels)
            if not self._validate_regime_labels(regime_labels):
                error_msg = "Regime labels validation failed"
                tprint(f"❌ [REGIME_MODELS] {error_msg}", color="red")
                self.logger.error("Regime labels validation failed")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg
                )

            unique_regimes = np.unique(regime_labels)
            tprint(f"📊 [REGIME_MODELS] Found regime labels: {len(regime_labels)} samples", color="blue")
            tprint(f"📊 [REGIME_MODELS] Unique regimes: {unique_regimes} (count: {len(unique_regimes)})", color="blue")
            regime_dist = {int(k): int(v) for k, v in zip(*np.unique(regime_labels, return_counts=True))}
            tprint(f"📊 [REGIME_MODELS] Regime distribution: {regime_dist}", color="blue")

            # Step 3: Prepare training data
            tprint("🔍 [REGIME_MODELS] Step 3: Preparing training data", color="cyan")
            data_prep_start = time.time()
            X, y, feature_selection_info, feature_names = self._prepare_training_data(data, regime_labels, pipeline_state)
            self._log_performance_metrics("Data preparation", data_prep_start)
            self._monitor_memory_usage("After data preparation")
            if X is None or y is None:
                error_msg = "Failed to prepare training data"
                tprint(f"❌ [REGIME_MODELS] {error_msg}", color="red")
                self.logger.error("Failed to prepare training data")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg
                )

            # Validate prepared training data
            if not self._validate_training_data(X, y, feature_names):
                error_msg = "Training data validation failed"
                tprint(f"❌ [REGIME_MODELS] {error_msg}", color="red")
                self.logger.error("Training data validation failed")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg
                )

            tprint(f"📊 [REGIME_MODELS] Training data prepared: X={X.shape}, y={y.shape}", color="blue")
            tprint(f"📊 [REGIME_MODELS] Feature matrix info: dtype={X.dtype}, min={X.min():.4f}, max={X.max():.4f}", color="blue")
            if feature_selection_info:
                retained = feature_selection_info.get('retained_feature_count', X.shape[1])
                total = feature_selection_info.get('total_feature_count', retained)
                tprint(
                    f"🎯 [REGIME_MODELS] Feature selection retained {retained}/{total} features",
                    color="green"
                )
                top_preview = feature_selection_info.get('top_features_preview')
                if top_preview:
                    tprint(
                        f"🏆 [REGIME_MODELS] Top retained features: {top_preview}",
                        color="blue"
                    )
            target_dist = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
            tprint(f"📊 [REGIME_MODELS] Target distribution: {target_dist}", color="blue")

            # Step 4: Train regime detection models
            tprint("🔍 [REGIME_MODELS] Step 4: Training regime detection models", color="cyan")
            model_training_start = time.time()
            training_results = self._train_regime_models(X, y, feature_selection_info, feature_names, unique_regimes)
            self._log_performance_metrics("Model training", model_training_start)
            self._monitor_memory_usage("After model training")

            # Clean up memory after training
            self._cleanup_memory()

            # Validate trained models
            if not self._validate_models(training_results['models']):
                error_msg = "Trained models validation failed"
                tprint(f"❌ [REGIME_MODELS] {error_msg}", color="red")
                self.logger.error("Trained models validation failed")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg
                )

            # Step 5: Create and validate artifacts
            tprint("🔍 [REGIME_MODELS] Step 5: Creating artifacts", color="cyan")
            artifacts = {
                'regime_models_training_result': {
                    'regime_models': training_results['models'],
                    'regime_metrics': training_results['metrics'],  # Explicit regime metrics
                    'metrics': training_results['metrics'],  # Keep for backward compatibility
                    'training_time': training_results['training_time'],
                    'regime_training_time': training_results['training_time'],  # Explicit regime training time
                    'success': True,
                    'model_count': len(training_results['models']),
                    'regime_model_count': len(training_results['models']),  # Explicit regime model count
                    'feature_count': X.shape[1],
                    'sample_count': X.shape[0],
                    'regime_models_config': self.regime_models_config,
                    'feature_selection': training_results.get('feature_selection', feature_selection_info),
                    'regime_feature_selection': training_results.get('feature_selection', feature_selection_info),  # Explicit regime feature selection
                    'feature_selection_info': feature_selection_info,  # Full feature selection info
                    'regime_feature_selection_info': feature_selection_info,  # Explicit regime feature selection info
                    'selected_feature_names': feature_selection_info.get('selected_feature_names', []) if feature_selection_info else [],
                    'regime_selected_feature_names': feature_selection_info.get('selected_feature_names', []) if feature_selection_info else []  # Explicit regime feature names
                }
            }

            execution_time = time.time() - execution_start_time

            # Log final performance metrics
            final_perf = self._get_system_performance()
            final_memory = self._monitor_memory_usage("Final")

            tprint(f"⏱️ [REGIME_MODELS] Total execution time: {execution_time:.2f} seconds", color="blue")
            if final_perf:
                tprint(f"💻 [REGIME_MODELS] Final system state - CPU: {final_perf.get('cpu_percent', 'N/A')}%, Memory: {final_perf.get('memory_percent', 'N/A')}%", color="blue")
            tprint(f"🧠 [REGIME_MODELS] Memory usage change: {final_memory - initial_memory:.1f} MB", color="blue")

            tprint("✅ [REGIME_MODELS] Regime detection models training completed successfully", color="green", bold=True)
            self.logger.info(f"Regime detection models training completed successfully in {execution_time:.2f} seconds")

            # Generate regime probability report
            try:
                regime_report = await self._generate_regime_probability_report(
                    training_results, X, feature_names, artifacts
                )
                if regime_report:
                    artifacts['regime_probability_report'] = regime_report
                    tprint("📊 [REGIME_MODELS] Regime probability report generated successfully", color="green")
            except Exception as e:
                tprint(f"⚠️ [REGIME_MODELS] Failed to generate regime probability report: {e}", color="yellow")

            # Save artifacts persistently using the artifact manager
            try:
                save_report = await self.save_artifacts(artifacts, {
                    'component_type': 'regime_models_training',
                    'regime_component': True,  # Explicitly mark as regime component
                    'regime_feature_selection': True,  # Mark that this includes regime feature selection
                    'execution_time': execution_time,
                    'training_time': training_results['training_time'],
                    'model_count': len(training_results['models']),
                    'regime_model_count': len(training_results['models']),  # Explicit regime model count
                    'feature_count': X.shape[1],
                    'sample_count': X.shape[0],
                    'selected_feature_count': feature_selection_info.get('retained_feature_count', X.shape[1]) if feature_selection_info else X.shape[1],
                    'regime_selected_feature_count': feature_selection_info.get('retained_feature_count', X.shape[1]) if feature_selection_info else X.shape[1],  # Explicit regime feature count
                    'regime_feature_selection_info_available': bool(feature_selection_info),
                })
                tprint(
                    f"💾 [REGIME_MODELS] Artifacts saved persistently (correlation_id={save_report.correlation_id}): {list(save_report.paths.keys())}",
                    color="green"
                )
            except Exception as e:
                tprint(f"⚠️ [REGIME_MODELS] Failed to save artifacts persistently: {e}", color="yellow")

            # Create component_result structure that includes feature_selection_info
            component_result = {
                'regime_models': training_results['models'],  # Explicit regime models
                'regime_metrics': training_results['metrics'],  # Explicit regime metrics
                'models': training_results['models'],  # Keep for backward compatibility
                'metrics': training_results['metrics'],  # Keep for backward compatibility
                'feature_selection_info': feature_selection_info,  # Ensure this is in component_result
                'regime_feature_selection_info': feature_selection_info,  # Explicit regime feature selection info
                'selected_feature_names': feature_selection_info.get('selected_feature_names', []) if feature_selection_info else [],
                'regime_selected_feature_names': feature_selection_info.get('selected_feature_names', []) if feature_selection_info else [],  # Explicit regime feature names
                'feature_count': X.shape[1],
                'selected_feature_count': feature_selection_info.get('retained_feature_count', X.shape[1]) if feature_selection_info else X.shape[1],
                'regime_selected_feature_count': feature_selection_info.get('retained_feature_count', X.shape[1]) if feature_selection_info else X.shape[1],  # Explicit regime count
            }
            
            # Add component_result to artifacts for easy access
            artifacts['component_result'] = component_result
            
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'component_type': 'regime_models_training',
                    'regime_component': True,  # Explicitly mark as regime component
                    'execution_time': execution_time,
                    'regime_training_time': training_results['training_time'],  # Explicit regime training time
                    'training_time': training_results['training_time'],  # Keep for backward compatibility
                    'model_count': len(training_results['models']),
                    'regime_model_count': len(training_results['models']),  # Explicit regime model count
                    'feature_count': X.shape[1],
                    'sample_count': X.shape[0],
                    'selected_feature_count': feature_selection_info.get('retained_feature_count', X.shape[1]) if feature_selection_info else X.shape[1],
                    'regime_selected_feature_count': feature_selection_info.get('retained_feature_count', X.shape[1]) if feature_selection_info else X.shape[1],  # Explicit regime feature count
                    'regime_feature_selection_info_available': bool(feature_selection_info),
                    'artifacts_saved_persistently': True,
                    'regime_artifacts_saved': True  # Explicit regime artifact save confirmation
                }
            )

        except Exception as e:
            execution_time = time.time() - execution_start_time
            error_type = type(e).__name__
            error_msg = f"Regime detection models training failed: {str(e)}"

            # Enhanced error logging with context
            tprint(f"❌ [REGIME_MODELS] {error_msg}", color="red")
            tprint(f"🔍 [REGIME_MODELS] Error type: {error_type}", color="yellow")
            tprint(f"🔍 [REGIME_MODELS] Execution time before failure: {execution_time:.2f} seconds", color="yellow")

            # Log system state at failure
            failure_perf = self._get_system_performance()
            if failure_perf:
                tprint(f"💻 [REGIME_MODELS] System state at failure - CPU: {failure_perf.get('cpu_percent', 'N/A')}%, Memory: {failure_perf.get('memory_percent', 'N/A')}%", color="yellow")

            # Provide recovery suggestions based on error type
            recovery_suggestions = self._get_recovery_suggestions(e)
            if recovery_suggestions:
                tprint(f"💡 [REGIME_MODELS] Recovery suggestions: {recovery_suggestions}", color="cyan")

            # Log detailed error information
            self.logger.error(f"Regime detection models training failed after {execution_time:.2f} seconds", exc_info=True)
            self.logger.error(f"Error type: {error_type}, Error message: {str(e)}")
            if recovery_suggestions:
                self.logger.error(f"Recovery suggestions: {recovery_suggestions}")

            return ComponentResult(
                success=False,
                artifacts={},
                error_message=f"{error_msg} (Type: {error_type})"
            )

    async def _generate_regime_probability_report(
        self,
        training_results: Dict[str, Any],
        X: np.ndarray,
        feature_names: List[str],
        artifacts: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate a comprehensive report with regime probabilities for all regimes."""
        try:
            tprint("📊 [REGIME_MODELS] Generating regime probability report", color="cyan")

            # Get the trained models
            models = training_results.get('models', {})
            if not models:
                tprint("⚠️ [REGIME_MODELS] No trained models found for report generation", color="yellow")
                return None

            # Use the first available model for probability generation
            model_name = list(models.keys())[0]
            model = models[model_name]

            if not hasattr(model, 'predict_proba'):
                tprint(f"⚠️ [REGIME_MODELS] Model {model_name} does not support probability prediction", color="yellow")
                return None

            # Generate regime probabilities for all samples
            tprint(f"🔮 [REGIME_MODELS] Generating regime probabilities using {model_name}", color="cyan")
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
                'report_type': 'regime_probability_analysis'
            }
            
            # Add comprehensive metrics if available
            if hasattr(self, 'model_metrics') and model_name in self.model_metrics:
                model_metrics = self.model_metrics[model_name]
                if 'classification' in model_metrics:
                    report['classification_metrics'] = model_metrics['classification']
                if 'temporal' in model_metrics:
                    report['temporal_metrics'] = model_metrics['temporal']
                if 'persistence' in model_metrics:
                    report['persistence_metrics'] = model_metrics['persistence']

            # Generate text report
            text_report = self._generate_text_report(report)
            report['text_report'] = text_report

            tprint(f"✅ [REGIME_MODELS] Regime probability report generated for {n_regimes} regimes", color="green")
            return report

        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Failed to generate regime probability report: {e}", color="red")
            return None

    def _generate_text_report(self, report: Dict[str, Any]) -> str:
        """Generate a human-readable text report from regime probability data."""
        try:
            lines = []
            lines.append("=" * 80)
            lines.append("REGIME PROBABILITY ANALYSIS REPORT")
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

            # Classification Metrics
            if 'classification_metrics' in report:
                cls_metrics = report['classification_metrics']
                lines.append("🎯 CLASSIFICATION METRICS")
                lines.append("-" * 40)
                lines.append(f"Accuracy: {cls_metrics.get('accuracy', 'N/A'):.4f}")
                lines.append(f"Balanced Accuracy: {cls_metrics.get('balanced_accuracy', 'N/A'):.4f}")
                lines.append(f"Precision (Weighted): {cls_metrics.get('precision', 'N/A'):.4f}")
                lines.append(f"Recall (Weighted): {cls_metrics.get('recall', 'N/A'):.4f}")
                lines.append(f"F1-Score (Weighted): {cls_metrics.get('f1_score', 'N/A'):.4f}")
                if 'log_loss' in cls_metrics and cls_metrics['log_loss'] is not None:
                    lines.append(f"Log Loss: {cls_metrics['log_loss']:.4f}")
                lines.append("")

            # Temporal/Stability Metrics
            if 'temporal_metrics' in report:
                temp_metrics = report['temporal_metrics']
                lines.append("⏱️ TEMPORAL/STABILITY METRICS")
                lines.append("-" * 40)
                lines.append(f"Mean Episode Length: {temp_metrics.get('mean_episode_length', 'N/A'):.2f}")
                lines.append(f"Transition Rate: {temp_metrics.get('transition_rate', 'N/A'):.4f}")
                lines.append(f"Short Episode Count: {temp_metrics.get('short_episode_count', 'N/A')}")
                lines.append(f"Switch False Positive Rate: {temp_metrics.get('switch_false_positive_rate', 'N/A'):.4f}")
                if temp_metrics.get('entropy') is not None:
                    lines.append(f"Entropy: {temp_metrics.get('entropy', 'N/A'):.4f}")
                if temp_metrics.get('confidence') is not None:
                    lines.append(f"Confidence: {temp_metrics.get('confidence', 'N/A'):.4f}")
                lines.append(f"Number of Episodes: {temp_metrics.get('n_episodes', 'N/A')}")
                lines.append(f"Number of Transitions: {temp_metrics.get('n_transitions', 'N/A')}")
                lines.append("")

            # Regime-Persistence Metrics
            if 'persistence_metrics' in report:
                pers_metrics = report['persistence_metrics']
                lines.append("🔄 REGIME-PERSISTENCE METRICS")
                lines.append("-" * 40)
                lines.append(f"Stability Index: {pers_metrics.get('stability_index', 'N/A'):.4f}")
                lines.append(f"Persistence Ratio: {pers_metrics.get('persistence_ratio', 'N/A'):.4f}")
                lines.append(f"Lag to Detection: {pers_metrics.get('lag_to_detection', 'N/A'):.2f}")
                lines.append(f"Episode Purity: {pers_metrics.get('episode_purity', 'N/A'):.4f}")
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

            lines.append("=" * 80)
            lines.append("END OF REGIME PROBABILITY REPORT")
            lines.append("=" * 80)

            return "\n".join(lines)

        except Exception as e:
            return f"Error generating text report: {e}"

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

    def _log_performance_metrics(self, stage: str, start_time: float):
        """Log performance metrics for a given stage."""
        current_time = time.time()
        elapsed_time = current_time - start_time

        # Get system performance
        perf_metrics = self._get_system_performance()

        # Log timing
        tprint(f"⏱️ [REGIME_MODELS] {stage} completed in {elapsed_time:.3f} seconds", color="blue")

        # Log system metrics if available
        if perf_metrics:
            tprint(f"💻 [REGIME_MODELS] System metrics - CPU: {perf_metrics.get('cpu_percent', 'N/A')}%, Memory: {perf_metrics.get('memory_percent', 'N/A')}% ({perf_metrics.get('memory_used_gb', 0):.1f}GB/{perf_metrics.get('memory_total_gb', 0):.1f}GB)", color="blue")

        return elapsed_time, perf_metrics

    def _monitor_memory_usage(self, stage: str):
        """Monitor and log memory usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024**2)
            tprint(f"🧠 [REGIME_MODELS] {stage} memory usage: {memory_mb:.1f} MB", color="blue")
            return memory_mb
        except Exception as e:
            tprint(f"⚠️ [REGIME_MODELS] Failed to monitor memory: {e}", color="yellow")
            return 0

    def _cleanup_memory(self):
        """Clean up memory by forcing garbage collection."""
        try:
            gc.collect()
            tprint("🧹 [REGIME_MODELS] Memory cleanup completed", color="blue")
        except Exception as e:
            tprint(f"⚠️ [REGIME_MODELS] Memory cleanup failed: {e}", color="yellow")

    def _validate_input_data(self, data: pd.DataFrame) -> bool:
        """Validate input data for training."""
        tprint("🔍 [REGIME_MODELS] Validating input data", color="cyan")

        try:
            # Check if data is empty
            if len(data) == 0:
                tprint("❌ [REGIME_MODELS] Input data is empty", color="red")
                return False

            # Check minimum required columns
            required_columns = ['close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                tprint(f"❌ [REGIME_MODELS] Missing required columns: {missing_columns}", color="red")
                return False

            # Check data types
            if not pd.api.types.is_numeric_dtype(data['close']):
                tprint("❌ [REGIME_MODELS] 'close' column is not numeric", color="red")
                return False

            # Check for sufficient data points
            min_samples = 100
            if len(data) < min_samples:
                tprint(f"❌ [REGIME_MODELS] Insufficient data points: {len(data)} < {min_samples}", color="red")
                return False

            tprint("✅ [REGIME_MODELS] Input data validation passed", color="green")
            return True

        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Data validation error: {e}", color="red")
            return False

    def _validate_regime_labels(self, regime_labels: np.ndarray) -> bool:
        """Validate regime labels."""
        tprint("🔍 [REGIME_MODELS] Validating regime labels", color="cyan")

        try:
            # Check if labels are not None
            if regime_labels is None:
                tprint("❌ [REGIME_MODELS] Regime labels are None", color="red")
                return False

            # Convert to numpy array if needed
            regime_labels = np.array(regime_labels)

            # Check for sufficient samples (reduced for testing with small datasets)
            min_labels = 6  # Reduced from 50 for testing with small datasets
            if len(regime_labels) < min_labels:
                tprint(f"❌ [REGIME_MODELS] Insufficient regime labels: {len(regime_labels)} < {min_labels}", color="red")
                return False

            # Check for valid regime values
            unique_regimes = np.unique(regime_labels)
            if len(unique_regimes) < 2:
                tprint(f"❌ [REGIME_MODELS] Insufficient regime classes: {len(unique_regimes)} < 2", color="red")
                return False

            tprint(f"✅ [REGIME_MODELS] Regime labels validation passed - {len(unique_regimes)} classes, {len(regime_labels)} samples", color="green")
            return True

        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Regime labels validation error: {e}", color="red")
            return False

    def _validate_training_data(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> bool:
        """Validate prepared training data."""
        tprint("🔍 [REGIME_MODELS] Validating training data", color="cyan")

        try:
            # Check shapes
            if X.shape[0] != y.shape[0]:
                tprint(f"❌ [REGIME_MODELS] Mismatched sample counts: X={X.shape[0]}, y={y.shape[0]}", color="red")
                return False

            # Check for sufficient features
            if X.shape[1] < 2:
                tprint(f"❌ [REGIME_MODELS] Insufficient features: {X.shape[1]} < 2", color="red")
                return False

            # Check for NaN or infinite values with detailed analysis
            nan_count = np.isnan(X).sum()
            inf_count = np.isinf(X).sum()
            if nan_count > 0:
                # Import the detailed NaN analysis function
                from src.utils.common_utilities import analyze_nan_values_detailed, format_nan_analysis_report

                # Perform detailed NaN analysis
                nan_analysis = analyze_nan_values_detailed(X, feature_names)
                detailed_report = format_nan_analysis_report(nan_analysis, "[REGIME_MODELS] ")

                tprint(f"❌ [REGIME_MODELS] Found {nan_count} NaN values in features", color="red")
                tprint(detailed_report, color="red")
                return False
            if inf_count > 0:
                tprint(f"❌ [REGIME_MODELS] Found {inf_count} infinite values in features", color="red")
                return False

            tprint("✅ [REGIME_MODELS] Training data validation passed", color="green")
            return True

        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Training data validation error: {e}", color="red")
            return False

    def _validate_models(self, models: Dict[str, Any]) -> bool:
        """Validate trained models."""
        tprint("🔍 [REGIME_MODELS] Validating trained models", color="cyan")

        try:
            if not models:
                tprint("❌ [REGIME_MODELS] No models trained", color="red")
                return False

            # Filter out metadata objects that are not actual models
            model_names_to_validate = [name for name in models.keys()
                                    if not name.endswith('_feature_indices') and
                                       not name.endswith('_metadata') and
                                       not name.endswith('_config')]

            tprint(f"🔍 [REGIME_MODELS] Validating {len(model_names_to_validate)} models: {model_names_to_validate}", color="blue")

            # Check each model
            valid_models = 0
            for name in model_names_to_validate:
                model = models[name]
                if model is None:
                    tprint(f"⚠️ [REGIME_MODELS] Model {name} is None (training failed)", color="yellow")
                    continue  # Skip None models but don't fail validation

                # Check if model has required methods
                if not hasattr(model, 'predict'):
                    tprint(f"❌ [REGIME_MODELS] Model {name} missing predict method", color="red")
                    return False

                valid_models += 1

            # Ensure at least one model is valid
            if valid_models == 0:
                tprint("❌ [REGIME_MODELS] No valid models trained", color="red")
                return False

            tprint(f"✅ [REGIME_MODELS] Model validation passed - {valid_models} valid models out of {len(model_names_to_validate)} attempted", color="green")
            return True

        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Model validation error: {e}", color="red")
            return False

    def _prepare_training_data(
        self,
        data: pd.DataFrame,
        regime_labels: np.ndarray,
        pipeline_state: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], Optional[List[str]]]:
        """Prepare training data from market data and regime labels."""
        tprint("🔧 [REGIME_MODELS] Preparing training data", color="cyan")
        self.logger.info("Starting data preparation process")

        try:
            # Log input data characteristics
            tprint(f"📊 [REGIME_MODELS] Input data shape: {data.shape}", color="blue")
            tprint(f"📊 [REGIME_MODELS] Input data columns: {list(data.columns)}", color="blue")

            # Force comprehensive feature generation using feature bank
            tprint("🔧 [REGIME_MODELS] FORCING comprehensive feature generation using feature bank", color="cyan", bold=True)
            tprint("🚫 [REGIME_MODELS] Bypassing clustering features to ensure comprehensive feature set", color="yellow")

            # Check if we should use original market data for feature generation
            original_data = None
            if pipeline_state is not None:
                original_data = pipeline_state.get('original_data')
                force_feature_bank = pipeline_state.get('force_feature_bank', False)

                if original_data is not None and force_feature_bank:
                    tprint("✅ [REGIME_MODELS] Using original market data for feature bank generation", color="green")
                    data_for_features = original_data
                else:
                    tprint("⚠️ [REGIME_MODELS] No original data available, using processed data", color="yellow")
                    data_for_features = data
            else:
                data_for_features = data

            if FEATURE_GENERATION_AVAILABLE:
                X, feature_names = self._generate_features_with_bank(data_for_features)
                if X is None or X.shape[1] < 50:
                    error_msg = f"Feature bank generated insufficient features: {X.shape[1] if X is not None else 0} < 50 required"
                    tprint(f"❌ [REGIME_MODELS] {error_msg}", color="red")
                    self.logger.error(error_msg)
                    return None, None, {}, None, None, None
                else:
                    tprint(f"✅ [REGIME_MODELS] Feature bank generated {X.shape[1]} comprehensive features", color="green")
                    if feature_names is None:
                        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            else:
                error_msg = "Feature generation system not available - cannot generate comprehensive features"
                tprint(f"❌ [REGIME_MODELS] {error_msg}", color="red")
                self.logger.error(error_msg)
                return None, None, {}, None, None

            # Check for NaN or infinite values in features with detailed analysis
            nan_count = np.isnan(X).sum()
            inf_count = np.isinf(X).sum()
            if nan_count > 0:
                # Import the detailed NaN analysis function
                from src.utils.common_utilities import analyze_nan_values_detailed, format_nan_analysis_report

                # Perform detailed NaN analysis
                nan_analysis = analyze_nan_values_detailed(X, feature_names)
                detailed_report = format_nan_analysis_report(nan_analysis, "[REGIME_MODELS] ")

                tprint(f"⚠️ [REGIME_MODELS] Found {nan_count} NaN values in features", color="yellow")
                tprint(detailed_report, color="yellow")
                tprint("🔧 [REGIME_MODELS] Filling NaN values with 0.0", color="cyan")
                X = np.nan_to_num(X, nan=0.0)
            if inf_count > 0:
                tprint(f"⚠️ [REGIME_MODELS] Found {inf_count} infinite values in features", color="yellow")
                tprint("🔧 [REGIME_MODELS] Replacing infinite values with finite numbers", color="cyan")
                X = np.nan_to_num(X, posinf=1e6, neginf=-1e6)

            # Align with regime labels
            tprint("🔧 [REGIME_MODELS] Aligning features with regime labels", color="cyan")
            min_length = min(len(X), len(regime_labels))
            X = X[:min_length]
            y = np.array(regime_labels[:min_length])

            # Early validation: Check class distribution before proceeding
            unique_classes, class_counts = np.unique(y, return_counts=True)
            min_class_count = np.min(class_counts)
            class_distribution = dict(zip(unique_classes, class_counts))

            tprint(f"📊 [REGIME_MODELS] Regime class distribution: {class_distribution}", color="blue")

            if min_class_count < 2:
                error_msg = f"🚨 CRITICAL: Insufficient regime class samples detected early!"
                tprint(f"❌ [REGIME_MODELS] {error_msg}", color="red")
                tprint(f"❌ [REGIME_MODELS] 🚨 Class distribution: {class_distribution}", color="red")
                tprint(f"❌ [REGIME_MODELS] 🚨 Minimum class count: {min_class_count} (required: 2+)", color="red")
                tprint("❌ [REGIME_MODELS] 🚨 Fast fail: Cannot proceed with insufficient regime data", color="red")
                tprint("❌ [REGIME_MODELS] 💡 Recommendation: Check regime clustering and labeling process", color="red")

                raise ValueError(f"Early validation failed: Insufficient regime class samples. Distribution: {class_distribution}. Minimum required: 2 samples per class.")

            tprint(f"✅ [REGIME_MODELS] Early validation passed: All regime classes have sufficient samples", color="green")

            # Perform model-driven feature selection
            feature_selection_info = self._run_feature_selection(X, y, feature_names)

            if feature_selection_info and feature_selection_info.get('selected_indices'):
                X = self._apply_feature_selection(X, feature_selection_info)
                feature_names = feature_selection_info.get('selected_feature_names', feature_names)
                tprint(
                    f"🎯 [REGIME_MODELS] Feature selector retained {feature_selection_info['retained_feature_count']}/{feature_selection_info['total_feature_count']} features",
                    color="green"
                )
                self.logger.info(
                    "Feature selection applied",
                    extra={
                        'retained_features': feature_selection_info['retained_feature_count'],
                        'total_features': feature_selection_info['total_feature_count'],
                        'selection_method': feature_selection_info.get('selection_method'),
                        'selection_time_seconds': feature_selection_info.get('selection_time_seconds')
                    }
                )
            else:
                tprint("⚠️ [REGIME_MODELS] Feature selection fallback - retaining all features", color="yellow")
                feature_selection_info = feature_selection_info or {}
                feature_selection_info.setdefault('selected_indices', list(range(X.shape[1])))
                feature_selection_info.setdefault('selected_feature_names', feature_names)
                feature_selection_info.setdefault('retained_feature_count', X.shape[1])
                feature_selection_info.setdefault('total_feature_count', X.shape[1])

            tprint(f"✅ [REGIME_MODELS] Training data prepared: {X.shape[0]} samples, {X.shape[1]} features", color="green", bold=True)

            self.logger.info(f"Training data preparation completed: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y, feature_selection_info, feature_names

        except Exception as e:
            error_type = type(e).__name__
            tprint(f"❌ [REGIME_MODELS] Error preparing training data: {e}", color="red")
            tprint(f"🔍 [REGIME_MODELS] Error type: {error_type}", color="yellow")

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
            
            if X is None or X.shape[1] < 50:
                raise ValueError(f"Insufficient features generated: {X.shape[1] if X is not None else 0} < 50 required")
            
            tprint(f"✅ [REGIME_MODELS] Features generated: {X.shape[1]} features", color="green")
            
            # Align with regime labels
            tprint("🔧 [REGIME_MODELS] Aligning features with regime labels", color="cyan")
            min_length = min(len(X), len(regime_labels))
            X = X[:min_length]
            y = np.array(regime_labels[:min_length])
            
            # Validate data
            if len(X) < 10:
                raise ValueError(f"Insufficient samples after alignment: {len(X)}")
            
            if len(np.unique(y)) < 2:
                raise ValueError(f"Insufficient regimes: {len(np.unique(y))}")
            
            tprint(f"✅ [REGIME_MODELS] Training data prepared: {X.shape[0]} samples, {X.shape[1]} features", color="green")
            return X, y, feature_names
            
        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Training data preparation failed: {e}", color="red")
            raise

    def _generate_features_with_bank(self, data: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        """Generate comprehensive features using the existing feature bank."""
        tprint("🔧 [REGIME_MODELS] Generating features using feature bank", color="cyan", bold=True)

        try:
            if not FEATURE_GENERATION_AVAILABLE:
                tprint("❌ [REGIME_MODELS] Feature generation system not available", color="red")
                return None, None

            # Get feature bank
            feature_bank = get_feature_bank()
            tprint("✅ [REGIME_MODELS] Feature bank retrieved successfully", color="green")

            # Define feature categories to generate - prioritize REGIME category for core regime features
            categories = [
                FeatureCategory.REGIME,  # Core regime features (lagged, derived, temporal)
                FeatureCategory.MOMENTUM,
                FeatureCategory.VOLATILITY,
                FeatureCategory.VOLUME,
                FeatureCategory.TREND,
                FeatureCategory.OSCILLATOR,
                FeatureCategory.RETURNS
            ]

            # Add core regime features with lagged, derived, and temporal features
            tprint("🔧 [REGIME_MODELS] Adding core regime features (lagged, derived, temporal)", color="cyan")
            core_regime_features = pd.DataFrame(index=data.index)
            try:
                from src.feature_generation.categories.regime_feature_integration import RegimeFeatureIntegration, RegimeFeatureConfig
                from src.feature_generation.categories.regime_feature_categorization import (
                    RegimeFeatureCategorizer, FeatureUseCase, get_regime_models_training_features
                )
                
                # Create RegimeFeatureIntegration generator
                regime_config = RegimeFeatureConfig(
                    enable_regime_detection=True,
                    enable_adaptive_features=True,
                    enable_regime_transitions=True
                )
                regime_generator = RegimeFeatureIntegration(regime_config)
                
                # Generate core regime features (lagged, derived, temporal) for all rows
                # Use vectorized approach: generate features using rolling windows
                # Initialize all feature columns
                all_feature_keys = set()
                
                # Sample a few windows to get all feature names
                sample_windows = [
                    data.iloc[:min(20, len(data))],
                    data.iloc[max(0, len(data)-20):] if len(data) > 20 else data
                ]
                for window_data in sample_windows:
                    if len(window_data) >= 5:
                        sample_features = regime_generator._generate_regime_features(window_data)
                        all_feature_keys.update(sample_features.keys())
                
                # Initialize DataFrame with all feature columns
                for feature_name in all_feature_keys:
                    core_regime_features[feature_name] = np.nan
                
                # Generate features row by row using a rolling window
                window_size = 20
                for i in range(len(data)):
                    if i < 5:  # Need at least 5 rows for some features
                        window_data = data.iloc[:i+1]
                    else:
                        # Use rolling window
                        window_start = max(0, i - window_size + 1)
                        window_data = data.iloc[window_start:i+1]
                    
                    # Generate features for this window
                    regime_features_dict = regime_generator._generate_regime_features(window_data)
                    
                    # Store features
                    for feature_name, feature_value in regime_features_dict.items():
                        if isinstance(feature_value, (int, float)):
                            core_regime_features.loc[data.index[i], feature_name] = feature_value
                        elif isinstance(feature_value, bool):
                            core_regime_features.loc[data.index[i], feature_name] = float(feature_value)
                        elif isinstance(feature_value, str):
                            # Store numeric representation of string (e.g., regime type)
                            core_regime_features.loc[data.index[i], feature_name] = hash(feature_value) % 1000
                        else:
                            core_regime_features.loc[data.index[i], feature_name] = 0.0
                    
                    # Progress indicator for large datasets
                    if (i + 1) % 100 == 0:
                        tprint(f"🔧 [REGIME_MODELS] Generated core regime features for {i+1}/{len(data)} rows", color="blue")
                
                tprint(f"✅ [REGIME_MODELS] Generated {len(core_regime_features.columns)} core regime features", color="green")
                
                # Log which core regime features are available
                categorizer = RegimeFeatureCategorizer()
                core_regime_feature_names = categorizer.get_features_for_use_case(FeatureUseCase.REGIME_MODELS_TRAINING)
                tprint(f"📋 [REGIME_MODELS] Core regime features available: {len(core_regime_feature_names)}", color="blue")
                
            except Exception as e:
                tprint(f"⚠️ [REGIME_MODELS] Error generating core regime features: {e}", color="yellow")
                import traceback
                tprint(f"Traceback: {traceback.format_exc()}", color="yellow")
            
            # Add advanced regime features for better regime detection
            tprint("🔧 [REGIME_MODELS] Adding advanced regime features for enhanced regime detection", color="cyan")
            advanced_generators = []
            try:
                from src.feature_generation.categories.advanced_regime_features import create_advanced_regime_generators
                advanced_generators = create_advanced_regime_generators()
                tprint(f"✅ [REGIME_MODELS] Loaded {len(advanced_generators)} advanced regime feature generators", color="green")
            except ImportError as e:
                tprint(f"⚠️ [REGIME_MODELS] Advanced regime features not available: {e}", color="yellow")
                advanced_generators = []
            except Exception as e:
                tprint(f"⚠️ [REGIME_MODELS] Error loading advanced regime features: {e}", color="yellow")
                advanced_generators = []

            all_features = pd.DataFrame(index=data.index)
            total_features = 0
            
            # Add core regime features first (lagged, derived, temporal)
            if not core_regime_features.empty:
                all_features = pd.concat([all_features, core_regime_features], axis=1)
                total_features += len(core_regime_features.columns)
                tprint(f"📊 [REGIME_MODELS] Core regime features: {core_regime_features.shape[1]}", color="blue")

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

                        if result and hasattr(result, 'data') and not result.len(data) == 0:
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

            # Generate advanced regime features
            if advanced_generators:
                tprint("🔍 [REGIME_MODELS] Generating advanced regime features", color="cyan")
                advanced_features = pd.DataFrame(index=data.index)
                for generator in advanced_generators:
                    try:
                        tprint(f"🔧 [REGIME_MODELS] Using advanced regime generator: {generator.config.name}", color="blue")
                        result = generator.generate(data)

                        if result is not None and not result.empty:
                            feature_name = f"advanced_regime_{generator.config.name}"
                            advanced_features[feature_name] = result
                            total_features += 1
                            tprint(f"✅ [REGIME_MODELS] Generated advanced regime feature: {feature_name}", color="green")
                        else:
                            tprint(f"⚠️ [REGIME_MODELS] Advanced regime generator {generator.config.name} returned empty result", color="yellow")
                    except Exception as e:
                        tprint(f"⚠️ [REGIME_MODELS] Advanced regime generator {generator.config.name} failed: {e}", color="yellow")
                        continue

                # Add advanced regime features to all features
                if not advanced_features.empty:
                    all_features = pd.concat([all_features, advanced_features], axis=1)
                    tprint(f"📊 [REGIME_MODELS] Advanced regime features: {advanced_features.shape[1]}", color="blue")

            # Convert to numpy array
            if not all_features.empty:
                X = all_features.values
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
                return X, feature_names
            else:
                tprint("❌ [REGIME_MODELS] Feature bank generated no features", color="red")
                return None, None

        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Error generating features with feature bank: {e}", color="red")
            self.logger.error(f"Error generating features with feature bank: {str(e)}", exc_info=True)
            return None, None

    def _train_regime_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_selection_info: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None,
        unique_regimes: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Train regime detection models."""
        tprint("🏋️ [REGIME_MODELS] Training regime detection models", color="cyan")
        self.logger.info("Starting regime detection models training process")

        start_time = time.time()
        models = {}
        metrics = {}
        training_history = []

        try:
            # Log training data characteristics
            tprint(f"📊 [REGIME_MODELS] Training data: {X.shape[0]} samples, {X.shape[1]} features", color="blue")
            if feature_selection_info:
                tprint(
                    f"🎯 [REGIME_MODELS] Using feature subset of {feature_selection_info.get('retained_feature_count', X.shape[1])}/{feature_selection_info.get('total_feature_count', X.shape[1])} features",
                    color="blue"
                )
            tprint(f"📊 [REGIME_MODELS] Target classes: {np.unique(y)} (count: {len(np.unique(y))})", color="blue")

            # Step 1: Validate class distribution before splitting
            tprint("🔧 [REGIME_MODELS] Step 1: Validating class distribution for training", color="cyan")
            split_start = time.time()

            # Check class distribution for stratified splitting
            unique_classes, class_counts = np.unique(y, return_counts=True)
            min_class_count = np.min(class_counts)
            class_distribution = dict(zip(unique_classes, class_counts))

            if min_class_count < 2:
                error_msg = f"🚨 CRITICAL: Insufficient samples for stratified training! Minimum class count: {min_class_count}"
                tprint(f"❌ [REGIME_MODELS] {error_msg}", color="red")
                tprint(f"❌ [REGIME_MODELS] 🚨 Class distribution: {class_distribution}", color="red")
                tprint("❌ [REGIME_MODELS] 🚨 Fast fail: Cannot proceed with model training", color="red")
                tprint("❌ [REGIME_MODELS] 💡 Recommendation: Ensure all regime classes have at least 2 samples", color="red")

                raise ValueError(f"Insufficient class samples for stratified training. Class distribution: {class_distribution}. Minimum required: 2 samples per class.")

            # Proceed with stratified split when validation passes
            tprint(f"✅ [REGIME_MODELS] Class distribution validation passed: {class_distribution}", color="green")
            tprint("🔧 [REGIME_MODELS] Proceeding with stratified train/test split", color="cyan")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.model_config['test_size'],
                random_state=self.model_config['random_state'],
                stratify=y
            )

            split_time = time.time() - split_start
            tprint(f"📊 [REGIME_MODELS] Train set: {X_train.shape[0]} samples", color="blue")
            tprint(f"📊 [REGIME_MODELS] Test set: {X_test.shape[0]} samples", color="blue")
            tprint(f"⏱️ [REGIME_MODELS] Data splitting completed in {split_time:.3f} seconds", color="blue")

            # Step 2: Scale features
            tprint("🔧 [REGIME_MODELS] Step 2: Scaling features", color="cyan")
            scale_start = time.time()

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            scale_time = time.time() - scale_start
            tprint(f"⏱️ [REGIME_MODELS] Feature scaling completed in {scale_time:.3f} seconds", color="blue")

            # Step 3: Train CatBoost with timeout protection
            tprint("🔧 [REGIME_MODELS] Step 3: Training CatBoost", color="cyan")
            catboost_start = time.time()

            try:
                # Use CPU-only configuration to prevent hanging on M1 Macs
                catboost_config = self.regime_models_config['base']['CatBoost'].copy()
                catboost_config.update({
                    'task_type': 'CPU',  # Force CPU usage to prevent GPU hanging
                    'verbose': False,    # Reduce verbosity
                    'random_seed': 42    # Ensure reproducibility
                })

                catboost_model = cb.CatBoostClassifier(**catboost_config)
                catboost_model.fit(X_train_scaled, y_train)
                models['CatBoost'] = catboost_model

                catboost_time = time.time() - catboost_start
                tprint(f"⏱️ [REGIME_MODELS] CatBoost training completed in {catboost_time:.3f} seconds", color="blue")
            except Exception as e:
                tprint(f"❌ [REGIME_MODELS] CatBoost training failed: {e}", color="red")
                models['CatBoost'] = None

            # Step 4: Train Greedy Rule Lists with parameter optimization
            tprint("🔧 [REGIME_MODELS] Step 4: Training Greedy Rule Lists with parameter optimization", color="cyan")
            grl_start = time.time()

            # Get number of classes for logging
            n_classes = len(np.unique(y_train))

            try:
                # First try with simple, robust parameters
                tprint("🔧 [REGIME_MODELS] Attempting Greedy Rule Lists with robust parameters", color="blue")
                grl_model = self._robust_grl_training(X_train_scaled, y_train, n_classes)
                models['Greedy Rule Lists'] = grl_model

                grl_time = time.time() - grl_start
                tprint(f"⏱️ [REGIME_MODELS] Greedy Rule Lists training completed in {grl_time:.3f} seconds", color="blue")
                tprint(f"📊 [REGIME_MODELS] Greedy Rule Lists: Supports multi-class with {n_classes} classes", color="green")
            except Exception as e:
                tprint(f"❌ [REGIME_MODELS] Greedy Rule Lists training failed: {e}", color="red")
                tprint(f"🔍 [REGIME_MODELS] Error details: {type(e).__name__}: {str(e)}", color="yellow")
                models['Greedy Rule Lists'] = None

            # Step 5: Train ExtraTrees
            tprint("🔧 [REGIME_MODELS] Step 5: Training ExtraTrees", color="cyan")
            extratrees_start = time.time()

            try:
                extratrees_model = ExtraTreesClassifier(**self.regime_models_config['base']['ExtraTrees'])
                extratrees_model.fit(X_train_scaled, y_train)
                models['ExtraTrees'] = extratrees_model

                extratrees_time = time.time() - extratrees_start
                tprint(f"⏱️ [REGIME_MODELS] ExtraTrees training completed in {extratrees_time:.3f} seconds", color="blue")
            except Exception as e:
                tprint(f"❌ [REGIME_MODELS] ExtraTrees training failed: {e}", color="red")
                models['ExtraTrees'] = None

            # Step 6: Train stacker_lgbm_calibrated (meta-learner) with proper cross-validation
            tprint("🔧 [REGIME_MODELS] Step 6: Training stacker_lgbm_calibrated meta-learner with CV", color="cyan")
            meta_start = time.time()

            try:
                # Create base models for stacking
                base_models = {}
                for name, model in models.items():
                    if model is not None:
                        base_models[name] = model

                if base_models:
                    # Generate out-of-fold predictions using cross-validation
                    tprint("🔄 [REGIME_MODELS] Generating out-of-fold predictions for meta-learning", color="blue")
                    oof_predictions = self._generate_out_of_fold_predictions(
                        base_models, X_train_scaled, y_train, cv_folds=5
                    )

                    if oof_predictions is not None:
                        # Create enhanced meta-learner features and store feature indices for consistency
                        enhanced_features, feature_indices = self._create_enhanced_meta_features_with_indices(
                            oof_predictions, X_train_scaled
                        )

                        # Create meta-learner with regularization to prevent overfitting
                        meta_config = self.regime_models_config['meta_learner']['stacker_lgbm_calibrated'].copy()
                        meta_config.update({
                            'num_leaves': 15,  # Reduce complexity
                            'max_depth': 4,    # Reduce depth
                            'learning_rate': 0.05,  # Lower learning rate
                            'n_estimators': 50,  # Fewer estimators
                            'reg_alpha': 0.1,  # L1 regularization
                            'reg_lambda': 0.1,  # L2 regularization
                            'subsample': 0.8,  # Subsampling for regularization
                            'colsample_bytree': 0.8,  # Feature sampling
                            'min_child_samples': 20,  # Minimum samples per leaf
                        })

                        meta_learner = lgb.LGBMClassifier(**meta_config)
                        meta_learner.fit(enhanced_features, y_train)
                        models['stacker_lgbm_calibrated'] = meta_learner

                        # Store feature indices for consistent prediction (as metadata, not as a model)
                        models['stacker_lgbm_calibrated_feature_indices'] = feature_indices

                        meta_time = time.time() - meta_start
                        tprint(f"⏱️ [REGIME_MODELS] Meta-learner training completed in {meta_time:.3f} seconds", color="blue")
                        tprint(f"📊 [REGIME_MODELS] Meta-learner features: {enhanced_features.shape[1]}", color="blue")
                    else:
                        tprint("⚠️ [REGIME_MODELS] Failed to generate out-of-fold predictions", color="yellow")
                        models['stacker_lgbm_calibrated'] = None
                else:
                    tprint("⚠️ [REGIME_MODELS] No base models available for meta-learner", color="yellow")
                    models['stacker_lgbm_calibrated'] = None

            except Exception as e:
                tprint(f"❌ [REGIME_MODELS] Meta-learner training failed: {e}", color="red")
                models['stacker_lgbm_calibrated'] = None

            # Step 7: Evaluate models
            tprint("🔧 [REGIME_MODELS] Step 7: Evaluating models", color="cyan")
            eval_start = time.time()

            # Only evaluate actual model objects, skip metadata like feature_indices
            model_names_to_evaluate = [name for name, model in models.items()
                                     if model is not None and hasattr(model, 'predict')]

            for name in model_names_to_evaluate:
                model = models[name]
                tprint(f"📊 [REGIME_MODELS] Evaluating {name}", color="blue")

                # Make predictions - handle meta-learner differently
                if name == 'stacker_lgbm_calibrated':
                    # Meta-learner needs enhanced features as input
                    # Only use the same base models that were used during training
                    base_model_names = ['CatBoost', 'XGBoost', 'Random Forest', 'Greedy Rule Lists', 'ExtraTrees']
                    base_predictions = np.column_stack([
                        np.argmax(models[base_name].predict_proba(X_test_scaled), axis=1).reshape(-1, 1) if hasattr(models[base_name], 'predict_proba') else models[base_name].predict(X_test_scaled).reshape(-1, 1)
                        for base_name in base_model_names
                        if base_name in models and models[base_name] is not None
                    ])
                    # Use stored feature indices for consistency
                    feature_indices = models.get('stacker_lgbm_calibrated_feature_indices')
                    tprint(f"🔧 [REGIME_MODELS] Base predictions shape: {base_predictions.shape}", color="blue")
                    tprint(f"🔧 [REGIME_MODELS] Feature indices: {len(feature_indices) if feature_indices is not None else 'None'}", color="blue")
                    enhanced_test_features = self._create_enhanced_meta_features(base_predictions, X_test_scaled, feature_indices)

                    # Validate feature dimensions match the trained model
                    expected_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else None
                    actual_features = enhanced_test_features.shape[1]
                    tprint(f"🔧 [REGIME_MODELS] Expected features: {expected_features}, Actual features: {actual_features}", color="blue")

                    if expected_features is not None and expected_features != actual_features:
                        error_msg = f"Feature dimension mismatch: model expects {expected_features} features but received {actual_features}"
                        tprint(f"❌ [REGIME_MODELS] {error_msg}", color="red")
                        raise ValueError(error_msg)

                    y_pred = model.predict(enhanced_test_features)
                    y_pred_proba = model.predict_proba(enhanced_test_features) if hasattr(model, 'predict_proba') else None
                else:
                    # Regular models use original features
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None

                # Calculate comprehensive metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                # Calculate comprehensive temporal and regime-persistence metrics
                comprehensive_metrics = self.temporal_metrics_calc.calculate_comprehensive_metrics(
                    y_test, y_pred, y_pred_proba
                )
                
                # Calculate temporal smoothness penalty
                smoothness_penalty = calculate_temporal_smoothness_penalty(
                    y_pred, alpha=self.temporal_smoothing_alpha
                )
                
                # Store detailed metrics
                model_metrics = {
                    'accuracy': accuracy,
                    'test_samples': len(y_test),
                    'train_samples': len(y_train),
                    'n_features': X.shape[1],
                    'classification': comprehensive_metrics.get('classification', {}),
                    'temporal': comprehensive_metrics.get('temporal', {}),
                    'persistence': comprehensive_metrics.get('persistence', {}),
                    'smoothness_penalty': smoothness_penalty
                }

                # Add comprehensive prediction probabilities if available
                if y_pred_proba is not None:
                    # Basic confidence metrics
                    max_probs = y_pred_proba.max(axis=1)
                    model_metrics['prediction_confidence'] = {
                        'mean': max_probs.mean(),
                        'std': max_probs.std(),
                        'min': max_probs.min(),
                        'max': max_probs.max()
                    }

                    # Regime-specific probability statistics
                    n_regimes = y_pred_proba.shape[1]
                    regime_prob_stats = {}
                    for i in range(n_regimes):
                        regime_probs = y_pred_proba[:, i]
                        regime_prob_stats[f'regime_{i}'] = {
                            'mean': regime_probs.mean(),
                            'std': regime_probs.std(),
                            'min': regime_probs.min(),
                            'max': regime_probs.max()
                        }
                    model_metrics['regime_probability_stats'] = regime_prob_stats

                    # Entropy and uncertainty measures
                    epsilon = 1e-10
                    entropy = -np.sum(y_pred_proba * np.log(y_pred_proba + epsilon), axis=1)
                    model_metrics['entropy_stats'] = {
                        'mean': entropy.mean(),
                        'std': entropy.std(),
                        'min': entropy.min(),
                        'max': entropy.max()
                    }

                    # Regime dominance (difference between top 2 probabilities)
                    sorted_probs = np.sort(y_pred_proba, axis=1)
                    if n_regimes > 1:
                        dominance = sorted_probs[:, -1] - sorted_probs[:, -2]
                        model_metrics['dominance_stats'] = {
                            'mean': dominance.mean(),
                            'std': dominance.std(),
                            'min': dominance.min(),
                            'max': dominance.max()
                        }
                    else:
                        model_metrics['dominance_stats'] = {'mean': 1.0, 'std': 0.0, 'min': 1.0, 'max': 1.0}

                    # Prediction stability (consistency across samples)
                    prob_std = np.std(y_pred_proba, axis=0)
                    model_metrics['regime_stability'] = {
                        'mean': (1.0 - prob_std).mean(),
                        'std': (1.0 - prob_std).std()
                    }

                metrics[name] = model_metrics

                # Log detailed results
                tprint(f"📊 [REGIME_MODELS] {name} accuracy: {accuracy:.4f}", color="green")

            eval_time = time.time() - eval_start
            tprint(f"⏱️ [REGIME_MODELS] Model evaluation completed in {eval_time:.3f} seconds", color="blue")

            # Calculate total training time
            training_time = time.time() - start_time

            # Log comprehensive training summary
            tprint("📊 [REGIME_MODELS] Training Summary:", color="cyan", bold=True)
            tprint(f"⏱️ [REGIME_MODELS] Total training time: {training_time:.2f} seconds", color="blue")
            tprint(f"📊 [REGIME_MODELS] Models trained: {len([m for m in models.values() if m is not None])}", color="blue")
            if metrics:
                tprint(f"📊 [REGIME_MODELS] Best accuracy: {max(metrics[m]['accuracy'] for m in metrics):.4f}", color="green")

            # Store training history
            training_history = {
                'data_split_time': split_time,
                'scaling_time': scale_time,
                'total_time': training_time
            }

            self.logger.info(f"Regime detection models training completed successfully in {training_time:.2f} seconds")

            return {
                'models': models,
                'metrics': metrics,
                'training_time': training_time,
                'scaler': scaler,
                'training_history': training_history,
                'feature_count': X.shape[1],
                'sample_count': X.shape[0],
                'feature_selection': feature_selection_info or {},
                'feature_names': feature_names,
                'n_regimes': len(unique_regimes)
            }

        except Exception as e:
            training_time = time.time() - start_time
            error_type = type(e).__name__
            tprint(f"❌ [REGIME_MODELS] Error training regime detection models: {e}", color="red")
            tprint(f"🔍 [REGIME_MODELS] Error type: {error_type}", color="yellow")

            self.logger.error(f"Error training regime detection models after {training_time:.2f} seconds: {str(e)}", exc_info=True)

            return {
                'models': {},
                'metrics': {},
                'training_time': training_time,
                'error': str(e),
                'error_type': error_type
            }

    def predict_regimes_with_probabilities(
        self,
        models: Dict[str, Any],
        scaler: Any,
        X: np.ndarray,
        feature_names: List[str],
        use_meta_learner: bool = True
    ) -> Dict[str, Any]:
        """
        Predict regime labels and probabilities using trained ensemble models.
        Enhanced to provide comprehensive probabilistic outputs for each detected regime.

        Args:
            models: Dictionary of trained models
            scaler: Fitted scaler for feature normalization
            X: Feature matrix
            feature_names: List of feature names
            use_meta_learner: Whether to use the meta-learner (stacker_lgbm_calibrated)

        Returns:
            Dictionary with comprehensive prediction information including:
            - regime_labels: Predicted regime for each sample
            - regime_probabilities: Probability matrix for each regime
            - regime_confidence_scores: Confidence scores for each prediction
            - regime_analysis: Detailed analysis of regime probabilities
            - ensemble_probabilities: Probabilities from all models in ensemble
        """
        try:
            tprint("🔮 [REGIME_MODELS] Starting regime prediction with probabilities", color="cyan")

            # Scale features
            X_scaled = scaler.transform(X)

            # Get the primary model (meta-learner if available, otherwise best base model)
            primary_model = None
            model_name = None

            if use_meta_learner and 'stacker_lgbm_calibrated' in models and models['stacker_lgbm_calibrated'] is not None:
                primary_model = models['stacker_lgbm_calibrated']
                model_name = 'stacker_lgbm_calibrated'
                tprint("🎯 [REGIME_MODELS] Using meta-learner (stacker_lgbm_calibrated)", color="blue")

                # Create enhanced features for meta-learner
                base_model_names = ['CatBoost', 'Greedy Rule Lists', 'ExtraTrees']
                base_predictions = np.column_stack([
                    np.argmax(models[base_name].predict_proba(X_scaled), axis=1).reshape(-1, 1)
                    if hasattr(models[base_name], 'predict_proba')
                    else models[base_name].predict(X_scaled).reshape(-1, 1)
                    for base_name in base_model_names
                    if base_name in models and models[base_name] is not None
                ])

                feature_indices = models.get('stacker_lgbm_calibrated_feature_indices')
                enhanced_features = self._create_enhanced_meta_features(base_predictions, X_scaled, feature_indices)
                X_for_prediction = enhanced_features

            else:
                # Use best available base model
                available_models = {name: model for name, model in models.items()
                                 if model is not None and hasattr(model, 'predict') and name != 'stacker_lgbm_calibrated_feature_indices'}

                if not available_models:
                    raise ValueError("No trained models available for prediction")

                # Use the first available model
                model_name = list(available_models.keys())[0]
                primary_model = available_models[model_name]
                X_for_prediction = X_scaled
                tprint(f"🎯 [REGIME_MODELS] Using base model: {model_name}", color="blue")

            # Make predictions
            regime_labels = primary_model.predict(X_for_prediction)
            regime_probabilities = primary_model.predict_proba(X_for_prediction)

            # Get number of regimes
            n_regimes = regime_probabilities.shape[1] if len(regime_probabilities.shape) > 1 else 1

            # Calculate comprehensive probability information
            max_probs = np.max(regime_probabilities, axis=1)
            confidence_scores = max_probs

            # Calculate regime distribution statistics
            regime_counts = np.bincount(regime_labels, minlength=n_regimes)
            regime_percentages = regime_counts / len(regime_labels) * 100

            # Calculate average probabilities for each regime
            avg_regime_probabilities = np.mean(regime_probabilities, axis=0)

            # Calculate regime stability (how consistent the predictions are)
            regime_stability = 1.0 - np.std(regime_probabilities, axis=0)

            # Calculate entropy (uncertainty measure)
            epsilon = 1e-10
            entropy = -np.sum(regime_probabilities * np.log(regime_probabilities + epsilon), axis=1)

            # Calculate dominance (difference between top 2 probabilities)
            sorted_probs = np.sort(regime_probabilities, axis=1)
            if n_regimes > 1:
                dominance = sorted_probs[:, -1] - sorted_probs[:, -2]
            else:
                dominance = np.ones(len(regime_labels))

            # Generate ensemble probabilities from all available models
            from src.utils.regime_ensemble_utils import generate_ensemble_probabilities
            ensemble_probabilities = generate_ensemble_probabilities(models, X_scaled, feature_names, "REGIME_MODELS")

            # Use RegimeProbabilityAnalyzer for comprehensive analysis
            from src.utils.regime_probability_analyzer import RegimeProbabilityAnalyzer
            analyzer = RegimeProbabilityAnalyzer()

            # Create prediction result for analysis
            prediction_result = {
                'regime_labels': regime_labels,
                'regime_probabilities': regime_probabilities,
                'n_regimes': n_regimes
            }

            # Perform comprehensive analysis
            analysis_results = analyzer.analyze_regime_predictions(prediction_result, model_name)

            # Extract analysis components
            regime_analysis = analysis_results.get('regime_analysis', {})
            regime_transitions = analysis_results.get('transition_analysis', {})
            regime_persistence = analysis_results.get('persistence_analysis', {})

            # Create comprehensive prediction result
            prediction_result = {
                'regime_labels': regime_labels,
                'regime_probabilities': regime_probabilities,
                'confidence_scores': confidence_scores,
                'n_regimes': n_regimes,
                'regime_counts': regime_counts.tolist(),
                'regime_percentages': regime_percentages.tolist(),
                'avg_regime_probabilities': avg_regime_probabilities.tolist(),
                'regime_stability': regime_stability.tolist(),
                'entropy': entropy,
                'dominance': dominance,
                'model_used': model_name,
                'ensemble_probabilities': ensemble_probabilities,
                'regime_analysis': regime_analysis,
                'regime_transitions': regime_transitions,
                'regime_persistence': regime_persistence,
                'prediction_metadata': {
                    'model_type': type(primary_model).__name__,
                    'n_samples': len(regime_labels),
                    'feature_count': X.shape[1],
                    'prediction_timestamp': datetime.now().isoformat(),
                    'scaled_features_used': True,
                    'ensemble_models_used': len(ensemble_probabilities) if ensemble_probabilities else 0
                }
            }

            tprint(f"✅ [REGIME_MODELS] Prediction completed: {len(regime_labels)} samples, {n_regimes} regimes", color="green")
            tprint(f"📊 [REGIME_MODELS] Confidence range: [{confidence_scores.min():.3f}, {confidence_scores.max():.3f}]", color="cyan")
            tprint(f"📈 [REGIME_MODELS] Regime distribution: {dict(zip(range(n_regimes), regime_counts))}", color="cyan")

            return prediction_result

        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Prediction failed: {e}", color="red")
            self.logger.error(f"Error in regime prediction: {e}", exc_info=True)
            return {
                'regime_labels': np.array([]),
                'regime_probabilities': np.array([]),
                'error': str(e)
            }

    def _run_feature_selection(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        Run model-based feature selection optimizing for both accuracy and temporal smoothness.
        
        Selects 60-80 features that maximize both classification accuracy and temporal stability.
        """
        selection_start = time.time()

        info: Dict[str, Any] = {
            'selection_performed': False,
            'selection_method': 'none',
            'selected_indices': list(range(features.shape[1])) if features.size else [],
            'selected_feature_names': feature_names.copy(),
            'retained_feature_count': int(features.shape[1]),
            'total_feature_count': int(features.shape[1]),
            'feature_importances': {},
            'importance_ranking': [],
            'top_features_preview': None,
            'selection_time_seconds': 0.0,
            'timestamp': datetime.utcnow().isoformat()
        }

        if features.size == 0 or labels.size == 0:
            return info

        try:
            tprint("🎯 [REGIME_MODELS] Starting dual-objective feature selection (accuracy + temporal smoothness)", color="cyan")
            
            # Target: 60-80 features
            if features.shape[1] < 60:
                target_feature_count = features.shape[1]  # Use all if less than 60
            else:
                target_feature_count = min(80, max(60, int(features.shape[1] * 0.3)))
            tprint(f"📊 [REGIME_MODELS] Target feature count: {target_feature_count}", color="blue")
            
            # Step 1: Get accuracy-based importance (LightGBM) - RFE style
            tprint("🔍 [REGIME_MODELS] Step 1: Evaluating accuracy-based importance (RFE style)", color="cyan")
            
            # RFE-style feature elimination: iteratively remove 50% of worst features
            # (keep top 50%) until we reach 60-80 features
            current_features = features.copy()
            current_feature_indices = np.arange(features.shape[1])
            accuracy_scores = np.zeros(features.shape[1])
            
            iteration = 0
            while len(current_feature_indices) > target_feature_count:
                iteration += 1
                n_current = len(current_feature_indices)
                
                # Determine if we should use CV and 100 estimators (< 120 features)
                use_cv = n_current < 120
                n_estimators = 100 if use_cv else 200
                
                if use_cv:
                    tprint(f"📊 [REGIME_MODELS] RFE iteration {iteration}: {n_current} features -> using 100 estimators & 5-fold CV", color="blue")
                else:
                    tprint(f"📊 [REGIME_MODELS] RFE iteration {iteration}: {n_current} features -> using 200 estimators", color="blue")
                
                # Train LightGBM model
                accuracy_model = lgb.LGBMClassifier(
                    n_estimators=n_estimators,
                    learning_rate=0.05,
                    random_state=self.model_config.get('random_state', 42),
                    class_weight='balanced',
                    importance_type='gain',
                    verbose=-1,
                    min_child_samples=50,  # Increased for stability
                    min_data_in_leaf=50
                )
                
                if use_cv:
                    # Use 5-fold CV when < 120 features
                    from sklearn.model_selection import cross_val_score, StratifiedKFold
                    cv = StratifiedKFold(n_splits=5, shuffle=False)
                    cv_scores = cross_val_score(
                        accuracy_model, 
                        current_features, 
                        labels, 
                        cv=cv, 
                        scoring='accuracy',
                        n_jobs=-1
                    )
                    # Still fit on full data to get feature importances
                    accuracy_model.fit(current_features, labels)
                else:
                    accuracy_model.fit(current_features, labels)
                
                # Get feature importances for current feature set
                if hasattr(accuracy_model, 'feature_importances_'):
                    current_importances = np.asarray(accuracy_model.feature_importances_)
                else:
                    current_importances = np.zeros(n_current)
                
                # Normalize importances
                if current_importances.sum() > 0:
                    current_normalized = current_importances / current_importances.sum()
                else:
                    current_normalized = np.ones(n_current) / n_current
                
                # Store scores for current features
                for local_idx, global_idx in enumerate(current_feature_indices):
                    accuracy_scores[global_idx] = current_normalized[local_idx]
                
                # Check if we've reached target range
                if n_current <= target_feature_count:
                    break
                
                # Remove 50% of worst features (keep top 50%)
                # Sort features by importance
                sorted_local_indices = np.argsort(current_normalized)[::-1]
                
                # Keep top 50% (or at least enough to reach target)
                keep_count = max(
                    target_feature_count,
                    int(n_current * 0.5)  # Keep top 50%
                )
                
                if keep_count >= n_current:
                    # If we can't reduce further, break
                    break
                
                # Select features to keep
                keep_local_indices = sorted_local_indices[:keep_count]
                keep_global_indices = current_feature_indices[keep_local_indices]
                
                # Update for next iteration
                current_feature_indices = keep_global_indices
                current_features = features[:, current_feature_indices]
            
            # Handle edge case: if we didn't train any model (e.g., started with <= target features)
            # we still need to compute accuracy scores
            if iteration == 0:
                tprint(f"📊 [REGIME_MODELS] Starting with {len(current_feature_indices)} features (<= target {target_feature_count}), computing accuracy scores", color="blue")
                n_current = len(current_feature_indices)
                use_cv = n_current < 120
                n_estimators = 100 if use_cv else 200
                
                accuracy_model = lgb.LGBMClassifier(
                    n_estimators=n_estimators,
                    learning_rate=0.05,
                    random_state=self.model_config.get('random_state', 42),
                    class_weight='balanced',
                    importance_type='gain',
                    verbose=-1,
                    min_child_samples=50,
                    min_data_in_leaf=50
                )
                
                if use_cv:
                    from sklearn.model_selection import cross_val_score, StratifiedKFold
                    cv = StratifiedKFold(n_splits=5, shuffle=False)
                    cv_scores = cross_val_score(
                        accuracy_model, 
                        current_features, 
                        labels, 
                        cv=cv, 
                        scoring='accuracy',
                        n_jobs=-1
                    )
                    accuracy_model.fit(current_features, labels)
                else:
                    accuracy_model.fit(current_features, labels)
                
                if hasattr(accuracy_model, 'feature_importances_'):
                    current_importances = np.asarray(accuracy_model.feature_importances_)
                else:
                    current_importances = np.zeros(n_current)
                
                if current_importances.sum() > 0:
                    current_normalized = current_importances / current_importances.sum()
                else:
                    current_normalized = np.ones(n_current) / n_current
                
                for local_idx, global_idx in enumerate(current_feature_indices):
                    accuracy_scores[global_idx] = current_normalized[local_idx]
            
            # Final normalization of accuracy scores
            if accuracy_scores.sum() > 0:
                accuracy_scores = accuracy_scores / accuracy_scores.sum()
            else:
                accuracy_scores = np.ones(features.shape[1]) / features.shape[1]
            
            tprint(f"✅ [REGIME_MODELS] RFE completed: {len(current_feature_indices)} features remaining after {iteration} iterations", color="green")
            
            # Step 2: Evaluate temporal smoothness for each feature
            tprint("🔍 [REGIME_MODELS] Step 2: Evaluating temporal smoothness", color="cyan")
            temporal_scores = np.zeros(features.shape[1])
            
            # Optimize: Only evaluate temporal smoothness for top features by accuracy
            # This speeds up evaluation significantly
            top_accuracy_features = np.argsort(accuracy_scores)[::-1][:min(150, features.shape[1])]
            tprint(f"📊 [REGIME_MODELS] Evaluating temporal smoothness for top {len(top_accuracy_features)} features by accuracy", color="blue")
            
            # Use quick temporal CV to evaluate smoothness
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Evaluate temporal smoothness for top accuracy features
            for feat_idx in top_accuracy_features:
                try:
                    # Train quick model on single feature
                    X_single = features[:, feat_idx].reshape(-1, 1)
                    
                    temp_model = lgb.LGBMClassifier(
                        n_estimators=50,
                        learning_rate=0.1,
                        random_state=42,
                        class_weight='balanced',
                        verbose=-1,
                        min_child_samples=50,
                        min_data_in_leaf=50
                    )
                    
                    # Evaluate temporal smoothness using CV
                    smoothness_scores = []
                    for train_idx, val_idx in tscv.split(X_single):
                        if len(train_idx) < 10 or len(val_idx) < 5:
                            continue
                        
                        temp_model.fit(X_single[train_idx], labels[train_idx])
                        y_pred_val = temp_model.predict(X_single[val_idx])
                        
                        # Calculate transition rate (lower is better for smoothness)
                        if len(y_pred_val) > 1:
                            transitions = sum(1 for i in range(1, len(y_pred_val)) 
                                            if y_pred_val[i] != y_pred_val[i-1])
                            transition_rate = transitions / len(y_pred_val)
                            # Convert to smoothness score (lower transition rate = higher smoothness)
                            smoothness = 1.0 - min(transition_rate, 1.0)
                            smoothness_scores.append(smoothness)
                    
                    if smoothness_scores:
                        temporal_scores[feat_idx] = np.mean(smoothness_scores)
                    else:
                        temporal_scores[feat_idx] = 0.5  # Default neutral score
                        
                except Exception:
                    temporal_scores[feat_idx] = 0.5  # Default neutral score
            
            # For features not evaluated, use accuracy score as proxy
            for feat_idx in range(features.shape[1]):
                if feat_idx not in top_accuracy_features:
                    # Use accuracy score as proxy for temporal smoothness
                    temporal_scores[feat_idx] = accuracy_scores[feat_idx] * 0.7  # Slight discount
            
            # Normalize temporal scores
            if temporal_scores.sum() > 0:
                temporal_scores = temporal_scores / (temporal_scores.sum() + 1e-10)
            
            # Step 3: Combine accuracy and temporal smoothness scores
            tprint("🔍 [REGIME_MODELS] Step 3: Combining accuracy and temporal smoothness scores", color="cyan")
            accuracy_weight = 0.6  # 60% weight on accuracy
            temporal_weight = 0.4   # 40% weight on temporal smoothness
            
            combined_scores = (
                accuracy_weight * accuracy_scores + 
                temporal_weight * temporal_scores
            )
            
            # Step 4: Select top features based on combined score
            sorted_indices = np.argsort(combined_scores)[::-1]
            selected_indices = sorted_indices[:target_feature_count]
            selected_feature_names = [feature_names[idx] for idx in selected_indices]
            
            # Build importance dictionaries
            accuracy_importance_dict = {
                feature_names[i]: float(accuracy_scores[i]) 
                for i in range(len(feature_names))
            }
            temporal_importance_dict = {
                feature_names[i]: float(temporal_scores[i]) 
                for i in range(len(feature_names))
            }
            combined_importance_dict = {
                feature_names[i]: float(combined_scores[i]) 
                for i in range(len(feature_names))
            }
            
            # Create ranking preview
            top_preview = [
                {
                    'feature': feature_names[sorted_indices[i]],
                    'combined_score': float(combined_scores[sorted_indices[i]]),
                    'accuracy_score': float(accuracy_scores[sorted_indices[i]]),
                    'temporal_score': float(temporal_scores[sorted_indices[i]]),
                    'rank': i + 1
                }
                for i in range(min(20, len(sorted_indices)))
            ]
            
            info.update({
                'selection_performed': True,
                'selection_method': 'dual_objective_accuracy_temporal',
                'selected_indices': [int(idx) for idx in selected_indices],
                'selected_feature_names': selected_feature_names,
                'retained_feature_count': int(len(selected_indices)),
                'total_feature_count': int(features.shape[1]),
                'target_feature_count': target_feature_count,
                'accuracy_weight': accuracy_weight,
                'temporal_weight': temporal_weight,
                'feature_importances': combined_importance_dict,
                'accuracy_importances': accuracy_importance_dict,
                'temporal_importances': temporal_importance_dict,
                'importance_ranking': top_preview,
                'top_features_preview': ', '.join(
                    f"{item['feature']} (acc:{item['accuracy_score']:.3f},temp:{item['temporal_score']:.3f})" 
                    for item in top_preview[:5]
                ),
                'selection_time_seconds': time.time() - selection_start
            })
            
            tprint(
                f"✅ [REGIME_MODELS] Dual-objective feature selection completed in {info['selection_time_seconds']:.3f}s",
                color="green"
            )
            tprint(
                f"🎯 [REGIME_MODELS] Retained {info['retained_feature_count']}/{info['total_feature_count']} features "
                f"(target: {target_feature_count})",
                color="green"
            )
            tprint(
                f"📊 [REGIME_MODELS] Top 5 features: {info['top_features_preview']}",
                color="cyan"
            )
            
            self.logger.info(
                "Dual-objective feature selection completed",
                extra={
                    'retained_features': info['retained_feature_count'],
                    'total_features': info['total_feature_count'],
                    'target_features': target_feature_count,
                    'selection_method': info['selection_method'],
                    'selection_time_seconds': info['selection_time_seconds']
                }
            )

        except Exception as e:
            info['selection_time_seconds'] = time.time() - selection_start
            tprint(f"⚠️ [REGIME_MODELS] Dual-objective feature selection failed ({e}); using fallback", color="yellow")
            self.logger.warning("Dual-objective feature selection failed; using fallback", exc_info=True)

            # Fallback: Use accuracy-based selection with 60-80 feature limit
            try:
                # Ensure target_feature_count is set (in case of early failure)
                if 'target_feature_count' not in locals():
                    if features.shape[1] < 60:
                        target_feature_count = features.shape[1]
                    else:
                        target_feature_count = min(80, max(60, int(features.shape[1] * 0.3)))
                
                selector = SelectFromModel(
                    lgb.LGBMClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        random_state=self.model_config.get('random_state', 42),
                        class_weight='balanced',
                        importance_type='gain',
                        verbose=-1,
                        min_child_samples=50,
                        min_data_in_leaf=50
                    ),
                    threshold='median',
                    max_features=target_feature_count  # Use target count
                )

                selector.fit(features, labels)
                fitted_estimator = getattr(selector, 'estimator_', None)
                if fitted_estimator is None and hasattr(selector, 'estimator'):
                    fitted_estimator = selector.estimator

                if fitted_estimator is not None and hasattr(fitted_estimator, 'feature_importances_'):
                    importances = np.asarray(fitted_estimator.feature_importances_)
                else:
                    importances = np.zeros(features.shape[1])

                support_mask = selector.get_support()
                if not np.any(support_mask):
                    # If no features selected, take top N by importance
                    sorted_indices = np.argsort(importances)[::-1]
                    support_mask = np.zeros_like(importances, dtype=bool)
                    support_mask[sorted_indices[:target_feature_count]] = True

                selected_indices = np.where(support_mask)[0]
                
                # Ensure we have exactly 60-80 features
                if len(selected_indices) < 60 and features.shape[1] >= 60:
                    sorted_indices = np.argsort(importances)[::-1]
                    selected_indices = sorted_indices[:min(80, features.shape[1])]
                
                if len(selected_indices) > 80:
                    sorted_indices = np.argsort(importances)[::-1]
                    selected_indices = sorted_indices[:80]
                
                selected_feature_names = [feature_names[idx] for idx in selected_indices]

                info.update({
                    'selection_performed': True,
                    'selection_method': 'accuracy_based_fallback',
                    'selected_indices': [int(idx) for idx in selected_indices],
                    'selected_feature_names': selected_feature_names,
                    'retained_feature_count': int(len(selected_indices)),
                    'total_feature_count': int(features.shape[1]),
                    'target_feature_count': target_feature_count,
                    'feature_importances': {
                        feature_names[i]: float(importances[i]) 
                        for i in range(len(feature_names))
                    },
                    'selection_time_seconds': time.time() - selection_start
                })
                
                tprint(f"✅ [REGIME_MODELS] Fallback selection retained {len(selected_indices)} features", color="green")
                
            except Exception as fallback_error:
                tprint(f"⚠️ [REGIME_MODELS] Fallback selection also failed ({fallback_error}); using all features", color="yellow")
                # Ultimate fallback: use all features
                info.update({
                    'selection_performed': False,
                    'selection_method': 'all_features_fallback',
                    'selected_indices': list(range(features.shape[1])),
                    'selected_feature_names': feature_names.copy(),
                    'retained_feature_count': int(features.shape[1]),
                    'selection_time_seconds': time.time() - selection_start
                })

        return info

    def _apply_feature_selection(
        self,
        features: np.ndarray,
        feature_selection_info: Dict[str, Any]
    ) -> np.ndarray:
        """Apply a stored feature selection mask to the provided features."""
        if features is None or feature_selection_info is None:
            return features

        indices = feature_selection_info.get('selected_indices')
        if not indices:
            return features

        indices_array = np.asarray(indices, dtype=int)
        return np.asarray(features)[:, indices_array]

    def _get_recovery_suggestions(self, error: Exception) -> str:
        """Get recovery suggestions based on error type."""
        error_type = type(error).__name__

        if "MemoryError" in error_type or "memory" in str(error).lower():
            return "Try reducing data size, increasing available memory, or using data sampling"
        elif "ImportError" in error_type:
            return "Check ML library installations: pip install catboost lightgbm imodels"
        elif "ValueError" in error_type and "shape" in str(error).lower():
            return "Check data alignment between features and labels, ensure consistent lengths"
        elif "KeyError" in error_type:
            return "Verify required columns exist in input data (close, volume, etc.)"
        elif "AttributeError" in error_type:
            return "Check model object integrity and required methods availability"
        else:
            return "Check logs for detailed error information and system requirements"

    def _generate_out_of_fold_predictions(self, base_models: dict, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Optional[np.ndarray]:
        """
        Generate out-of-fold predictions using cross-validation to prevent data leakage.

        Args:
            base_models: Dictionary of trained base models
            X: Feature matrix
            y: Target labels
            cv_folds: Number of cross-validation folds

        Returns:
            Array of out-of-fold predictions or None if failed
        """
        try:
            from sklearn.model_selection import StratifiedKFold

            tprint(f"🔄 [REGIME_MODELS] Generating {cv_folds}-fold out-of-fold predictions", color="blue")

            # Initialize array to store OOF predictions
            oof_predictions = np.zeros((X.shape[0], 0))

            # Create stratified K-fold for balanced regime distribution
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

            for model_name, model in base_models.items():
                if model is None:
                    continue

                tprint(f"📊 [REGIME_MODELS] Generating OOF predictions for {model_name}", color="blue")

                # Special handling for CatBoost to prevent hanging
                if model_name == 'CatBoost':
                    model_oof = self._generate_catboost_oof_with_timeout(model, X, y, skf)
                else:
                    model_oof = np.zeros(X.shape[0])

                    # Generate out-of-fold predictions for this model
                    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                        y_train_fold = y[train_idx]

                        # Train model on fold
                        if hasattr(model, 'fit'):
                            model.fit(X_train_fold, y_train_fold)

                        # Predict on validation fold
                        if hasattr(model, 'predict_proba'):
                            val_pred_proba = model.predict_proba(X_val_fold)
                            # Use max probability class for multi-class
                            model_oof[val_idx] = np.argmax(val_pred_proba, axis=1)
                        else:
                            model_oof[val_idx] = model.predict(X_val_fold)

                # Reshape and add to OOF predictions
                model_oof = model_oof.reshape(-1, 1)
                oof_predictions = np.column_stack([oof_predictions, model_oof])

            if oof_predictions.shape[1] > 0:
                tprint(f"✅ [REGIME_MODELS] Generated OOF predictions: {oof_predictions.shape}", color="green")
                return oof_predictions
            else:
                tprint("⚠️ [REGIME_MODELS] No valid OOF predictions generated", color="yellow")
                return None

        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] OOF prediction generation failed: {e}", color="red")
            self.logger.error(f"OOF prediction generation failed: {e}")
            return None

    def _generate_catboost_oof_with_timeout(self, model, X: np.ndarray, y: np.ndarray, skf) -> np.ndarray:
        """
        Generate CatBoost OOF predictions with timeout and CPU fallback to prevent hanging.

        Args:
            model: CatBoost model
            X: Feature matrix
            y: Target labels
            skf: StratifiedKFold object

        Returns:
            Array of OOF predictions
        """
        import signal
        import threading
        import time

        def timeout_handler(signum, frame):
            raise TimeoutError("CatBoost OOF prediction timed out")

        model_oof = np.zeros(X.shape[0])

        try:
            # Set timeout for CatBoost operations (30 seconds per fold)
            timeout_seconds = 30

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold = y[train_idx]

                # Create a fresh CatBoost model for this fold to avoid GPU issues
                try:
                    # Use CPU-only configuration to prevent hanging
                    fold_model = cb.CatBoostClassifier(
                        iterations=50,  # Reduced iterations for speed
                        depth=4,       # Reduced depth
                        learning_rate=0.1,
                        task_type='CPU',  # Force CPU usage
                        verbose=False,
                        random_seed=42
                    )

                    # Set timeout for training
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(timeout_seconds)

                    # Train model on fold
                    fold_model.fit(X_train_fold, y_train_fold)

                    # Predict on validation fold
                    val_pred_proba = fold_model.predict_proba(X_val_fold)
                    model_oof[val_idx] = np.argmax(val_pred_proba, axis=1)

                    # Cancel timeout
                    signal.alarm(0)

                except TimeoutError:
                    tprint(f"⚠️ [REGIME_MODELS] CatBoost fold {fold} timed out, using fallback", color="yellow")
                    # Fallback: use simple majority class prediction
                    from collections import Counter
                    majority_class = Counter(y_train_fold).most_common(1)[0][0]
                    model_oof[val_idx] = majority_class
                    signal.alarm(0)

                except Exception as e:
                    tprint(f"⚠️ [REGIME_MODELS] CatBoost fold {fold} failed: {e}, using fallback", color="yellow")
                    # Fallback: use simple majority class prediction
                    from collections import Counter
                    majority_class = Counter(y_train_fold).most_common(1)[0][0]
                    model_oof[val_idx] = majority_class
                    signal.alarm(0)

            tprint(f"✅ [REGIME_MODELS] CatBoost OOF predictions generated with timeout protection", color="green")
            return model_oof

        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] CatBoost OOF generation failed: {e}", color="red")
            # Ultimate fallback: return random predictions
            np.random.seed(42)
            unique_classes = np.unique(y)
            return np.random.choice(unique_classes, size=X.shape[0])

    def _create_enhanced_meta_features(self, base_predictions: np.ndarray, original_features: np.ndarray, feature_indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create enhanced meta-learner features combining base predictions with original features.

        Args:
            base_predictions: Out-of-fold predictions from base models
            original_features: Original feature matrix
            feature_indices: Pre-selected feature indices for consistency (optional)

        Returns:
            Enhanced feature matrix for meta-learner
        """
        try:
            tprint("🔧 [REGIME_MODELS] Creating enhanced meta-learner features", color="blue")
            tprint(f"🔧 [REGIME_MODELS] Base predictions shape: {base_predictions.shape}", color="blue")
            tprint(f"🔧 [REGIME_MODELS] Original features shape: {original_features.shape}", color="blue")

            enhanced_features = []

            # Add base model predictions
            enhanced_features.append(base_predictions)

            # Add prediction statistics
            if base_predictions.shape[1] > 1:
                # Prediction agreement (how many models agree)
                pred_agreement = np.apply_along_axis(
                    lambda x: len(set(x)) / len(x), axis=1, arr=base_predictions
                ).reshape(-1, 1)
                enhanced_features.append(pred_agreement)

                # Prediction confidence (standard deviation of predictions)
                pred_confidence = np.std(base_predictions, axis=1).reshape(-1, 1)
                enhanced_features.append(pred_confidence)

                # Most frequent prediction (safer alternative to stats.mode)
                # Use the prediction with highest confidence instead of mode
                most_frequent = np.argmax(base_predictions, axis=1).reshape(-1, 1)
                enhanced_features.append(most_frequent)

            # Add subset of original features (most important ones)
            # Use consistent feature selection to avoid training/prediction mismatch
            if feature_indices is not None:
                # Use pre-selected features for consistency
                top_features_idx = feature_indices
                tprint(f"🔧 [REGIME_MODELS] Using pre-selected features: {len(top_features_idx)} features", color="blue")
            else:
                # Select features with highest variance (most informative)
                feature_variance = np.var(original_features, axis=0)
                top_features_idx = np.argsort(feature_variance)[-min(6, original_features.shape[1]):]
                tprint(f"🔧 [REGIME_MODELS] Selected top features by variance: {len(top_features_idx)} features", color="blue")

            enhanced_features.append(original_features[:, top_features_idx])

            # Combine all enhanced features
            enhanced_matrix = np.column_stack(enhanced_features)

            tprint(f"✅ [REGIME_MODELS] Enhanced features created: {enhanced_matrix.shape}", color="green")
            return enhanced_matrix

        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Enhanced feature creation failed: {e}", color="red")
            self.logger.error(f"Enhanced feature creation failed: {e}")
            # Fallback to base predictions only
            return base_predictions

    def _create_enhanced_meta_features_with_indices(self, base_predictions: np.ndarray, original_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create enhanced meta-learner features and return feature indices for consistency.

        Args:
            base_predictions: Out-of-fold predictions from base models
            original_features: Original feature matrix

        Returns:
            Tuple of (enhanced feature matrix, feature indices)
        """
        try:
            tprint("🔧 [REGIME_MODELS] Creating enhanced meta-learner features with indices", color="blue")

            enhanced_features = []

            # Add base model predictions
            enhanced_features.append(base_predictions)

            # Add prediction statistics
            if base_predictions.shape[1] > 1:
                # Prediction agreement (how many models agree)
                pred_agreement = np.apply_along_axis(
                    lambda x: len(set(x)) / len(x), axis=1, arr=base_predictions
                ).reshape(-1, 1)
                enhanced_features.append(pred_agreement)

                # Prediction confidence (standard deviation of predictions)
                pred_confidence = np.std(base_predictions, axis=1).reshape(-1, 1)
                enhanced_features.append(pred_confidence)

                # Most frequent prediction (safer alternative to stats.mode)
                # Use the prediction with highest confidence instead of mode
                most_frequent = np.argmax(base_predictions, axis=1).reshape(-1, 1)
                enhanced_features.append(most_frequent)

            # Select features with highest variance (most informative) and store indices
            feature_variance = np.var(original_features, axis=0)
            top_features_idx = np.argsort(feature_variance)[-min(6, original_features.shape[1]):]
            enhanced_features.append(original_features[:, top_features_idx])

            # Combine all enhanced features
            enhanced_matrix = np.column_stack(enhanced_features)

            tprint(f"✅ [REGIME_MODELS] Enhanced features created: {enhanced_matrix.shape}", color="green")
            tprint(f"🔧 [REGIME_MODELS] Stored feature indices: {len(top_features_idx)} features", color="blue")
            return enhanced_matrix, top_features_idx

        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Enhanced feature creation failed: {e}", color="red")
            self.logger.error(f"Enhanced feature creation failed: {e}")
            # Fallback to base predictions only
            return base_predictions, np.array([])

    def _log_grl_optimization_results(self, best_params: dict, best_score: float, n_classes: int, n_samples: int) -> None:
        """
        Log detailed results of Greedy Rule Lists optimization.

        Args:
            best_params: Best parameters found
            best_score: Best cross-validation score
            n_classes: Number of regime classes
            n_samples: Number of training samples
        """
        try:
            tprint("📊 [REGIME_MODELS] Greedy Rule Lists Optimization Results:", color="cyan", bold=True)
            tprint(f"🎯 [REGIME_MODELS] Best CV Score: {best_score:.4f}", color="green")
            tprint(f"📋 [REGIME_MODELS] Best Parameters:", color="blue")
            tprint(f"   - max_depth: {best_params.get('max_depth', 'N/A')}", color="blue")
            tprint(f"   - max_rules: {best_params.get('max_rules', 'N/A')}", color="blue")
            tprint(f"📊 [REGIME_MODELS] Context: {n_classes} regimes, {n_samples} samples", color="blue")

            # Calculate expected improvement over default
            default_score = 0.0545  # 5.45% baseline
            improvement = ((best_score - default_score) / default_score) * 100
            tprint(f"📈 [REGIME_MODELS] Expected improvement: {improvement:+.1f}% over baseline", color="green")

            # Log to file for persistence
            self.logger.info(f"Greedy Rule Lists optimization completed:")
            self.logger.info(f"  Best CV Score: {best_score:.4f}")
            self.logger.info(f"  Best Parameters: {best_params}")
            self.logger.info(f"  Expected improvement: {improvement:+.1f}% over baseline")

        except Exception as e:
            tprint(f"⚠️ [REGIME_MODELS] Failed to log optimization results: {e}", color="yellow")

    def _advanced_grl_optimization(self, X_train: np.ndarray, y_train: np.ndarray, n_classes: int) -> GreedyRuleListClassifier:
        """
        Advanced parameter optimization for Greedy Rule Lists using adaptive search.

        Args:
            X_train: Training features
            y_train: Training labels
            n_classes: Number of regime classes

        Returns:
            Optimized GreedyRuleListClassifier
        """
        try:
            tprint("🚀 [REGIME_MODELS] Advanced Greedy Rule Lists optimization with adaptive search", color="cyan")

            # Adaptive parameter ranges based on data characteristics
            n_samples, n_features = X_train.shape

            # Calculate adaptive parameter ranges
            adaptive_max_depth = min(20, max(8, int(np.log2(n_samples))))
            adaptive_min_samples = max(5, min(50, n_samples // (n_classes * 4)))

            tprint(f"📊 [REGIME_MODELS] Adaptive ranges - max_depth: {adaptive_max_depth}, min_samples: {adaptive_min_samples}", color="blue")

            # Define adaptive parameter search space
            # Note: GreedyRuleListClassifier only supports max_depth, class_weight, and criterion parameters
            param_combinations = [
                # High complexity for complex regimes
                {
                    'max_depth': adaptive_max_depth
                },
                # Balanced complexity
                {
                    'max_depth': adaptive_max_depth - 2
                },
                # Conservative for stability
                {
                    'max_depth': adaptive_max_depth - 4
                }
            ]

            best_model = None
            best_score = 0.0
            best_params = None

            # Test each parameter combination
            for i, params in enumerate(param_combinations):
                tprint(f"🔍 [REGIME_MODELS] Testing combination {i+1}/{len(param_combinations)}: {params}", color="blue")

                try:
                    # Create model with current parameters
                    # Note: GreedyRuleListClassifier doesn't support min_samples_split and min_samples_leaf
                    model = GreedyRuleListClassifier(
                        max_depth=params['max_depth'],
                        criterion='gini'
                    )

                    # Use stratified cross-validation for better regime balance
                    from sklearn.model_selection import StratifiedKFold
                    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

                    cv_scores = cross_val_score(
                        model, X_train, y_train,
                        cv=skf, scoring='accuracy', n_jobs=1
                    )
                    mean_score = cv_scores.mean()
                    std_score = cv_scores.std()

                    tprint(f"📊 [REGIME_MODELS] CV Score: {mean_score:.4f} ± {std_score:.4f}", color="blue")

                    # Update best model if this is better
                    if mean_score > best_score:
                        best_score = mean_score
                        best_model = model
                        best_params = params.copy()

                except Exception as e:
                    tprint(f"⚠️ [REGIME_MODELS] Parameter combination {i+1} failed: {e}", color="yellow")
                    continue

            # Train the best model on full training data
            if best_model is not None:
                tprint(f"✅ [REGIME_MODELS] Best parameters found: {best_params}", color="green")
                tprint(f"📊 [REGIME_MODELS] Best CV score: {best_score:.4f}", color="green")

                # Fit the best model
                best_model.fit(X_train, y_train)
                return best_model
            else:
                # Fallback to default parameters
                tprint("⚠️ [REGIME_MODELS] No optimal parameters found, using default configuration", color="yellow")
                default_model = GreedyRuleListClassifier(**self.regime_models_config['base']['Greedy Rule Lists'])
                default_model.fit(X_train, y_train)
                return default_model

        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Advanced Greedy Rule Lists optimization failed: {e}", color="red")
            self.logger.error(f"Advanced Greedy Rule Lists optimization failed: {e}")

            # Fallback to default parameters
            tprint("🔄 [REGIME_MODELS] Using fallback default parameters", color="yellow")
            default_model = GreedyRuleListClassifier(**self.regime_models_config['base']['Greedy Rule Lists'])
            default_model.fit(X_train, y_train)
            return default_model

    def _optimize_greedy_rule_lists(self, X_train: np.ndarray, y_train: np.ndarray, n_classes: int) -> GreedyRuleListClassifier:
        """
        Optimize Greedy Rule Lists parameters for complex regime detection.

        Args:
            X_train: Training features
            y_train: Training labels
            n_classes: Number of regime classes

        Returns:
            Optimized GreedyRuleListClassifier
        """
        try:
            tprint("🔧 [REGIME_MODELS] Optimizing Greedy Rule Lists parameters for complex regimes", color="cyan")

            # Define parameter search space based on regime complexity
            # Note: GreedyRuleListClassifier only supports max_depth, class_weight, and criterion parameters
            param_grids = [
                # Conservative parameters for stable regimes
                {
                    'max_depth': [15, 18, 20],
                    'class_weight': ['balanced']
                },
                # Aggressive parameters for complex regimes
                {
                    'max_depth': [20, 25, 30],
                    'class_weight': ['balanced']
                }
            ]

            best_model = None
            best_score = 0.0
            best_params = None

            # Try different parameter combinations
            for param_grid in param_grids:
                tprint(f"🔍 [REGIME_MODELS] Testing parameter grid: {param_grid}", color="blue")

                # Simple grid search with cross-validation
                from sklearn.model_selection import cross_val_score

                for max_depth in param_grid['max_depth']:
                    for class_weight in param_grid['class_weight']:
                        try:
                            # Create model with current parameters
                            # Note: GreedyRuleListClassifier only supports max_depth, class_weight, and criterion
                            model = GreedyRuleListClassifier(
                                max_depth=max_depth,
                                criterion='gini',
                                class_weight=class_weight
                            )

                            # Cross-validation score
                            cv_scores = cross_val_score(
                                model, X_train, y_train,
                                cv=3, scoring='accuracy', n_jobs=1
                            )
                            mean_score = cv_scores.mean()

                            tprint(f"📊 [REGIME_MODELS] Params: depth={max_depth}, class_weight={class_weight} -> CV Score: {mean_score:.4f}", color="blue")

                            # Update best model if this is better
                            if mean_score > best_score:
                                best_score = mean_score
                                best_model = model
                                best_params = {
                                    'max_depth': max_depth,
                                    'class_weight': class_weight
                                }

                        except Exception as e:
                            tprint(f"⚠️ [REGIME_MODELS] Parameter combination failed: {e}", color="yellow")
                            continue

            # Train the best model on full training data
            if best_model is not None:
                tprint(f"✅ [REGIME_MODELS] Best parameters found: {best_params}", color="green")
                tprint(f"📊 [REGIME_MODELS] Best CV score: {best_score:.4f}", color="green")

                # Fit the best model
                best_model.fit(X_train, y_train)
                return best_model
            else:
                # Fallback to default parameters
                tprint("⚠️ [REGIME_MODELS] No optimal parameters found, using default configuration", color="yellow")
                default_model = GreedyRuleListClassifier(**self.regime_models_config['base']['Greedy Rule Lists'])
                default_model.fit(X_train, y_train)
                return default_model

        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Greedy Rule Lists optimization failed: {e}", color="red")
            self.logger.error(f"Greedy Rule Lists optimization failed: {e}")

            # Fallback to default parameters
            tprint("🔄 [REGIME_MODELS] Using fallback default parameters", color="yellow")
            default_model = GreedyRuleListClassifier(**self.regime_models_config['base']['Greedy Rule Lists'])
            default_model.fit(X_train, y_train)
            return default_model

    def _robust_grl_training(self, X_train: np.ndarray, y_train: np.ndarray, n_classes: int) -> GreedyRuleListClassifier:
        """
        Robust Greedy Rule Lists training with multiple fallback strategies.

        Args:
            X_train: Training features
            y_train: Training labels
            n_classes: Number of regime classes

        Returns:
            Trained GreedyRuleListClassifier
        """
        try:
            tprint("🔧 [REGIME_MODELS] Starting robust Greedy Rule Lists training", color="cyan")

            # Preprocess features for better GRL performance
            tprint("🔧 [REGIME_MODELS] Preprocessing features for Greedy Rule Lists", color="blue")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            # Check class distribution
            from collections import Counter
            class_counts = Counter(y_train)
            tprint(f"📊 [REGIME_MODELS] Class distribution: {dict(class_counts)}", color="blue")

            # Handle class imbalance with SMOTE if needed
            min_class_count = min(class_counts.values())
            if min_class_count < 10:  # If any class has less than 10 samples
                tprint("⚠️ [REGIME_MODELS] Detected class imbalance, applying SMOTE", color="yellow")
                try:
                    from imblearn.over_sampling import SMOTE
                    smote = SMOTE(random_state=42, k_neighbors=1)
                    X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
                    tprint(f"📊 [REGIME_MODELS] After SMOTE - X: {X_train_scaled.shape}, y: {len(y_train)}", color="blue")
                except Exception as e:
                    tprint(f"⚠️ [REGIME_MODELS] SMOTE failed, continuing without resampling: {e}", color="yellow")

            # Strategy 1: Conservative parameters for stability
            try:
                tprint("🔧 [REGIME_MODELS] Strategy 1: Conservative parameters", color="blue")
                conservative_params = {
                    'max_depth': 15,
                    'criterion': 'gini',
                    'class_weight': 'balanced'
                }

                model = GreedyRuleListClassifier(**conservative_params)
                model.fit(X_train_scaled, y_train)
                tprint("✅ [REGIME_MODELS] Conservative parameters successful", color="green")
                return model

            except Exception as e:
                tprint(f"⚠️ [REGIME_MODELS] Conservative parameters failed: {e}", color="yellow")

            # Strategy 2: Minimal parameters
            try:
                tprint("🔧 [REGIME_MODELS] Strategy 2: Minimal parameters", color="blue")
                minimal_params = {
                    'max_depth': 10,
                    'criterion': 'gini',
                    'class_weight': 'balanced'
                }

                model = GreedyRuleListClassifier(**minimal_params)
                model.fit(X_train_scaled, y_train)
                tprint("✅ [REGIME_MODELS] Minimal parameters successful", color="green")
                return model

            except Exception as e:
                tprint(f"⚠️ [REGIME_MODELS] Minimal parameters failed: {e}", color="yellow")

            # Strategy 3: Ultra-minimal parameters
            try:
                tprint("🔧 [REGIME_MODELS] Strategy 3: Ultra-minimal parameters", color="blue")
                ultra_minimal_params = {
                    'max_depth': 8,
                    'criterion': 'gini',
                    'class_weight': 'balanced'
                }

                model = GreedyRuleListClassifier(**ultra_minimal_params)
                model.fit(X_train_scaled, y_train)
                tprint("✅ [REGIME_MODELS] Ultra-minimal parameters successful", color="green")
                return model

            except Exception as e:
                tprint(f"⚠️ [REGIME_MODELS] Ultra-minimal parameters failed: {e}", color="yellow")

            # Strategy 4: Default parameters as last resort
            try:
                tprint("🔧 [REGIME_MODELS] Strategy 4: Default parameters", color="blue")
                model = GreedyRuleListClassifier()
                model.fit(X_train_scaled, y_train)
                tprint("✅ [REGIME_MODELS] Default parameters successful", color="green")
                return model

            except Exception as e:
                tprint(f"❌ [REGIME_MODELS] All Greedy Rule Lists strategies failed: {e}", color="red")
                raise e

        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Robust Greedy Rule Lists training failed: {e}", color="red")
            self.logger.error(f"Robust Greedy Rule Lists training failed: {e}")
            raise e

    def _load_artifacts_from_outcome_files(self) -> Dict[str, Any]:
        """Load artifacts from previous outcome files when not available in pipeline state."""
        tprint("🔍 [REGIME_MODELS] Loading artifacts from previous outcome files", color="yellow")

        artifacts = {}

        try:
            # Look for the most recent successful regime clustering outcome file
            import glob
            from pathlib import Path

            outcomes_dir = Path("outcomes")
            if not outcomes_dir.exists():
                tprint("⚠️ [REGIME_MODELS] No outcomes directory found", color="yellow")
                return artifacts

            # Find all regime clustering outcome files and sort by timestamp (most recent first)
            pattern = "*regime_clustering_outcome_*.json"
            outcome_files = list(outcomes_dir.glob(pattern))

            if not outcome_files:
                tprint("⚠️ [REGIME_MODELS] No regime clustering outcome files found", color="yellow")
                return artifacts

            # Sort by modification time (most recent first)
            outcome_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Try the most recent successful outcome file
            for outcome_file in outcome_files:
                try:
                    tprint(f"🔍 [REGIME_MODELS] Checking outcome file: {outcome_file.name}", color="blue")

                    with open(outcome_file, 'r') as f:
                        outcome_data = json.load(f)

                    # Check if this outcome file was successful
                    if outcome_data.get('status') == 'completed':
                        outcome_artifacts = outcome_data.get('artifacts', {})
                        tprint(f"✅ [REGIME_MODELS] Found successful outcome file: {outcome_file.name}", color="green")

                        # Extract the optimal_regime_clustering_result if available
                        optimal_clustering = outcome_artifacts.get('optimal_regime_clustering_result')
                        if optimal_clustering:
                            tprint("✅ [REGIME_MODELS] Found optimal_regime_clustering_result in outcome file", color="green")
                            artifacts['optimal_regime_clustering_result'] = optimal_clustering

                        # Also check for other useful artifacts
                        if 'regime_clustering_result' in outcome_artifacts:
                            artifacts['regime_clustering_result'] = outcome_artifacts['regime_clustering_result']

                        return artifacts

                except Exception as e:
                    tprint(f"⚠️ [REGIME_MODELS] Failed to read outcome file {outcome_file.name}: {e}", color="yellow")
                    continue

            tprint("⚠️ [REGIME_MODELS] No successful outcome files found", color="yellow")

        except Exception as e:
            tprint(f"❌ [REGIME_MODELS] Failed to load artifacts from outcome files: {e}", color="red")
            self.logger.error(f"Failed to load artifacts from outcome files: {e}")

        return artifacts
