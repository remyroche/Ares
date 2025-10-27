"""
Improved Regime Detection Models Training Component.

This component implements the improved regime detection models with:
- Proper temporal validation to prevent data leakage
- Simplified regime label extraction with fast fail
- Robust feature generation with fast fail
- Configuration validation
- Improved error handling
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

# New improved imports
from src.utils.ml_common.validation.temporal_data_splitter import (
    TemporalDataSplitter, RegimeAwareSplitter, create_temporal_splitter
)
from src.utils.ml_common.data.regime_label_extractor import (
    RegimeLabelExtractor, extract_regime_labels_fast_fail
)
from src.utils.ml_common.validation.config_validator import (
    validate_regime_training_config, create_default_regime_training_config
)
from src.utils.ml_common.features.robust_feature_generator import (
    RobustFeatureGenerator, generate_features_fast_fail, FeatureGenerationError
)

# Existing imports
from src.utils.ml_common.unified_vectorization_manager import (
    UnifiedVectorizationManager, OperationType, OptimizationStrategy
)
from src.utils.ml_common.optimization.hpo_utils import (
    HyperparameterOptimization
)
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
from src.utils.ml_common.explainability.model_explainability import ModelExplainability
from src.utils.ml_common.explainability.shap_lime_integration import SHAPLIMEIntegration
from src.utils.ml_common.post_training.model_validation import (
    ModelValidator, ValidationConfig
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Import ML libraries with comprehensive error handling
tprint("🔍 [IMPROVED_REGIME_MODELS] Starting ML libraries import process", color="cyan")
ML_LIBRARIES_AVAILABLE = False
ML_LIBRARY_VERSIONS = {}
ML_IMPORT_ERRORS = []

# Import sklearn components
try:
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.feature_selection import SelectFromModel
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    import sklearn
    ML_LIBRARY_VERSIONS['sklearn'] = sklearn.__version__
    tprint(f"✅ [IMPROVED_REGIME_MODELS] scikit-learn imported successfully (v{sklearn.__version__})", color="green")
except ImportError as e:
    ML_IMPORT_ERRORS.append(f"scikit-learn: {e}")
    tprint(f"❌ [IMPROVED_REGIME_MODELS] Failed to import scikit-learn: {e}", color="red")

# Import CatBoost
try:
    import catboost as cb
    ML_LIBRARY_VERSIONS['catboost'] = cb.__version__
    tprint(f"✅ [IMPROVED_REGIME_MODELS] CatBoost imported successfully (v{cb.__version__})", color="green")
except ImportError as e:
    ML_IMPORT_ERRORS.append(f"CatBoost: {e}")
    tprint(f"❌ [IMPROVED_REGIME_MODELS] Failed to import CatBoost: {e}", color="red")

# Import LightGBM
try:
    import lightgbm as lgb
    ML_LIBRARY_VERSIONS['lightgbm'] = lgb.__version__
    tprint(f"✅ [IMPROVED_REGIME_MODELS] LightGBM imported successfully (v{lgb.__version__})", color="green")
except ImportError as e:
    ML_IMPORT_ERRORS.append(f"LightGBM: {e}")
    tprint(f"❌ [IMPROVED_REGIME_MODELS] Failed to import LightGBM: {e}", color="red")

# Import Greedy Rule Lists
try:
    from imodels import GreedyRuleListClassifier
    ML_LIBRARY_VERSIONS['imodels'] = "1.0.0"
    tprint(f"✅ [IMPROVED_REGIME_MODELS] imodels (Greedy Rule Lists) imported successfully", color="green")
except ImportError as e:
    ML_IMPORT_ERRORS.append(f"imodels (Greedy Rule Lists): {e}")
    tprint(f"❌ [IMPROVED_REGIME_MODELS] Failed to import imodels: {e}", color="red")

# Check if all required libraries are available
if len(ML_IMPORT_ERRORS) == 0:
    ML_LIBRARIES_AVAILABLE = True
    tprint("✅ [IMPROVED_REGIME_MODELS] All ML libraries imported successfully", color="green")
else:
    tprint(f"⚠️ [IMPROVED_REGIME_MODELS] Some ML libraries failed to import: {ML_IMPORT_ERRORS}", color="yellow")

# Feature generation availability check
FEATURE_GENERATION_AVAILABLE = False
try:
    from src.feature_generation.integration.feature_task_integration import get_feature_bank
    from src.feature_generation.categories.regime_feature_categorization import FeatureCategory
    FEATURE_GENERATION_AVAILABLE = True
    tprint("✅ [IMPROVED_REGIME_MODELS] Feature generation system available", color="green")
except ImportError as e:
    tprint(f"⚠️ [IMPROVED_REGIME_MODELS] Feature generation system not available: {e}", color="yellow")


class ImprovedRegimeModelsTrainingComponent(BaseMarketAnalysisComponent):
    """
    Improved Regime Detection Models Training Component.
    
    This component trains regime detection models with:
    - Proper temporal validation to prevent data leakage
    - Simplified regime label extraction with fast fail
    - Robust feature generation with fast fail
    - Configuration validation
    - Improved error handling
    """

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the Improved Regime Models Training Component."""
        tprint("🚀 [IMPROVED_REGIME_MODELS] Initializing Improved Regime Models Training Component", color="cyan", bold=True)
        
        # Initialize parent component
        try:
            super().__init__(config)
            tprint("✅ [IMPROVED_REGIME_MODELS] Parent component initialized successfully", color="green")
        except Exception as e:
            tprint(f"❌ [IMPROVED_REGIME_MODELS] Failed to initialize parent component: {e}", color="red")
            raise

        # Initialize logger
        self.logger = system_logger.getChild('ImprovedRegimeModelsTraining')
        self.logger.info("Improved Regime Models Training Component logger initialized")
        tprint("✅ [IMPROVED_REGIME_MODELS] Logger initialized successfully", color="green")

        # Validate configuration
        self._validate_and_setup_config()

        # Initialize components
        self._initialize_components()

        tprint("✅ [IMPROVED_REGIME_MODELS] Component initialization completed", color="green")

    def _validate_and_setup_config(self):
        """Validate and setup configuration."""
        tprint("🔧 [IMPROVED_REGIME_MODELS] Validating configuration", color="cyan")
        
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
        
        # Validate configuration
        try:
            self.validated_config = validate_regime_training_config(config_dict, strict=True)
            tprint("✅ [IMPROVED_REGIME_MODELS] Configuration validated successfully", color="green")
        except ValueError as e:
            tprint(f"❌ [IMPROVED_REGIME_MODELS] Configuration validation failed: {e}", color="red")
            raise

    def _initialize_components(self):
        """Initialize all required components."""
        tprint("🔧 [IMPROVED_REGIME_MODELS] Initializing components", color="cyan")
        
        # Initialize temporal splitter
        self.temporal_splitter = create_temporal_splitter(self.validated_config)
        tprint("✅ [IMPROVED_REGIME_MODELS] Temporal splitter initialized", color="green")
        
        # Initialize regime label extractor
        self.regime_extractor = RegimeLabelExtractor(
            min_samples=self.validated_config.get('min_regime_samples', 10),
            min_regimes=2
        )
        tprint("✅ [IMPROVED_REGIME_MODELS] Regime label extractor initialized", color="green")
        
        # Initialize feature generator
        self.feature_generator = RobustFeatureGenerator(
            min_total_features=50,
            min_samples=100
        )
        tprint("✅ [IMPROVED_REGIME_MODELS] Feature generator initialized", color="green")
        
        # Initialize hardware manager
        self.hardware_manager = UnifiedHardwareManager(
            HardwareConfig(
                max_memory_usage=0.8,
                enable_optimization=True,
                optimization_level=OptimizationLevel.AGGRESSIVE
            )
        )
        tprint("✅ [IMPROVED_REGIME_MODELS] Hardware manager initialized", color="green")
        
        # Initialize lookahead protection
        self.lookahead_protection = LookaheadProtection()
        tprint("✅ [IMPROVED_REGIME_MODELS] Lookahead protection initialized", color="green")
        
        # Initialize evaluation utilities
        self.evaluation_utils = EvaluationUtils()
        tprint("✅ [IMPROVED_REGIME_MODELS] Evaluation utilities initialized", color="green")

    async def execute(self, data: pd.DataFrame, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute improved regime detection models training.
        
        Args:
            data: Market data DataFrame
            pipeline_state: Pipeline state dictionary
            
        Returns:
            ComponentResult with training results
        """
        execution_start_time = time.time()
        tprint("🚀 [IMPROVED_REGIME_MODELS] Starting improved regime detection models training", color="cyan", bold=True)
        
        try:
            # Step 1: Initialize hardware optimization
            tprint("🔧 [IMPROVED_REGIME_MODELS] Initializing hardware optimization", color="cyan")
            await self.hardware_manager.initialize()
            await self.hardware_manager.optimize_for_workload(WorkloadType.ML_TRAINING)
            tprint("✅ [IMPROVED_REGIME_MODELS] Hardware optimization initialized", color="green")
            
            # Step 2: Apply lookahead protection
            tprint("🔒 [IMPROVED_REGIME_MODELS] Applying lookahead protection", color="cyan")
            protected_data = self.lookahead_protection.protect_data(data)
            tprint("✅ [IMPROVED_REGIME_MODELS] Lookahead protection applied", color="green")
            
            # Step 3: Extract regime labels with fast fail
            tprint("📊 [IMPROVED_REGIME_MODELS] Extracting regime labels", color="cyan")
            try:
                regime_labels = self.regime_extractor.extract_regime_labels(pipeline_state.get('artifacts', {}))
                tprint(f"✅ [IMPROVED_REGIME_MODELS] Regime labels extracted: {len(regime_labels)} samples", color="green")
            except ValueError as e:
                tprint(f"❌ [IMPROVED_REGIME_MODELS] Regime label extraction failed: {e}", color="red")
                return ComponentResult(
                    success=False,
                    error_message=f"Regime label extraction failed: {e}",
                    artifacts={},
                    metadata={'execution_time': time.time() - execution_start_time}
                )
            
            # Step 4: Generate features with fast fail
            tprint("🔧 [IMPROVED_REGIME_MODELS] Generating features", color="cyan")
            try:
                X, feature_names = self.feature_generator.generate_features(protected_data)
                tprint(f"✅ [IMPROVED_REGIME_MODELS] Features generated: {X.shape[1]} features", color="green")
            except FeatureGenerationError as e:
                tprint(f"❌ [IMPROVED_REGIME_MODELS] Feature generation failed: {e}", color="red")
                return ComponentResult(
                    success=False,
                    error_message=f"Feature generation failed: {e}",
                    artifacts={},
                    metadata={'execution_time': time.time() - execution_start_time}
                )
            
            # Step 5: Align features and labels
            tprint("🔧 [IMPROVED_REGIME_MODELS] Aligning features and labels", color="cyan")
            min_length = min(len(X), len(regime_labels))
            X = X[:min_length]
            y = regime_labels[:min_length]
            tprint(f"✅ [IMPROVED_REGIME_MODELS] Data aligned: {X.shape[0]} samples", color="green")
            
            # Step 6: Split data temporally
            tprint("🔧 [IMPROVED_REGIME_MODELS] Splitting data temporally", color="cyan")
            X_train, X_val, X_test, y_train, y_val, y_test = self.temporal_splitter.split_regime_aware(X, y)
            tprint(f"✅ [IMPROVED_REGIME_MODELS] Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}", color="green")
            
            # Step 7: Train models
            tprint("🏋️ [IMPROVED_REGIME_MODELS] Training models", color="cyan")
            trained_models = await self._train_models_improved(X_train, y_train, X_val, y_val)
            if not trained_models:
                tprint("❌ [IMPROVED_REGIME_MODELS] No models trained successfully", color="red")
                return ComponentResult(
                    success=False,
                    error_message="No models trained successfully",
                    artifacts={},
                    metadata={'execution_time': time.time() - execution_start_time}
                )
            tprint(f"✅ [IMPROVED_REGIME_MODELS] Trained {len(trained_models)} models", color="green")
            
            # Step 8: Evaluate models
            tprint("📊 [IMPROVED_REGIME_MODELS] Evaluating models", color="cyan")
            evaluation_results = await self._evaluate_models_improved(trained_models, X_test, y_test)
            tprint("✅ [IMPROVED_REGIME_MODELS] Model evaluation completed", color="green")
            
            # Step 9: Create artifacts
            tprint("📦 [IMPROVED_REGIME_MODELS] Creating artifacts", color="cyan")
            artifacts = self._create_artifacts(trained_models, evaluation_results, feature_names, X.shape)
            tprint("✅ [IMPROVED_REGIME_MODELS] Artifacts created", color="green")
            
            # Step 10: Cleanup
            tprint("🧹 [IMPROVED_REGIME_MODELS] Cleaning up resources", color="cyan")
            await self.hardware_manager.cleanup()
            tprint("✅ [IMPROVED_REGIME_MODELS] Cleanup completed", color="green")
            
            execution_time = time.time() - execution_start_time
            tprint(f"✅ [IMPROVED_REGIME_MODELS] Training completed successfully in {execution_time:.2f}s", color="green", bold=True)
            
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'execution_time': execution_time,
                    'n_models_trained': len(trained_models),
                    'n_features': X.shape[1],
                    'n_samples': X.shape[0],
                    'config': self.validated_config
                }
            )
            
        except Exception as e:
            execution_time = time.time() - execution_start_time
            error_msg = f"Improved regime models training failed: {str(e)}"
            tprint(f"❌ [IMPROVED_REGIME_MODELS] {error_msg}", color="red", bold=True)
            self.logger.error(error_msg, exc_info=True)
            
            # Cleanup on error
            try:
                await self.hardware_manager.cleanup()
            except Exception as cleanup_error:
                tprint(f"⚠️ [IMPROVED_REGIME_MODELS] Cleanup failed: {cleanup_error}", color="yellow")
            
            return ComponentResult(
                success=False,
                error_message=error_msg,
                artifacts={},
                metadata={'execution_time': execution_time}
            )

    async def _train_models_improved(self, X_train: np.ndarray, y_train: np.ndarray, 
                                   X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Train models with improved error handling."""
        tprint("🏋️ [IMPROVED_REGIME_MODELS] Training models with improved error handling", color="cyan")
        
        if not ML_LIBRARIES_AVAILABLE:
            raise RuntimeError("ML libraries not available")
        
        trained_models = {}
        failed_models = []
        
        # Model configurations
        model_configs = {
            'catboost': {
                'class': cb.CatBoostClassifier,
                'params': {
                    'iterations': 100,
                    'depth': 4,
                    'learning_rate': 0.05,
                    'random_seed': self.validated_config['random_state'],
                    'verbose': False
                }
            },
            'extratrees': {
                'class': ExtraTreesClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'random_state': self.validated_config['random_state']
                }
            },
            'greedy_rules': {
                'class': GreedyRuleListClassifier,
                'params': {
                    'max_depth': 3,
                    'random_state': self.validated_config['random_state']
                }
            }
        }
        
        # Train each model
        for model_name, config in model_configs.items():
            try:
                tprint(f"🔧 [IMPROVED_REGIME_MODELS] Training {model_name}", color="blue")
                
                # Create model
                model_class = config['class']
                model_params = config['params']
                model = model_class(**model_params)
                
                # Train model
                model.fit(X_train, y_train)
                
                # Validate model
                if hasattr(model, 'predict'):
                    val_predictions = model.predict(X_val)
                    val_accuracy = accuracy_score(y_val, val_predictions)
                    tprint(f"✅ [IMPROVED_REGIME_MODELS] {model_name} trained - Val accuracy: {val_accuracy:.4f}", color="green")
                    
                    trained_models[model_name] = {
                        'model': model,
                        'val_accuracy': val_accuracy,
                        'config': model_params
                    }
                else:
                    tprint(f"⚠️ [IMPROVED_REGIME_MODELS] {model_name} has no predict method", color="yellow")
                    failed_models.append(model_name)
                    
            except Exception as e:
                tprint(f"❌ [IMPROVED_REGIME_MODELS] {model_name} training failed: {e}", color="red")
                failed_models.append(model_name)
        
        if failed_models:
            tprint(f"⚠️ [IMPROVED_REGIME_MODELS] Failed to train models: {failed_models}", color="yellow")
        
        return trained_models

    async def _evaluate_models_improved(self, models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate models with improved metrics."""
        tprint("📊 [IMPROVED_REGIME_MODELS] Evaluating models with improved metrics", color="cyan")
        
        evaluation_results = {}
        
        for model_name, model_data in models.items():
            try:
                model = model_data['model']
                
                # Get predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                # Regime-specific performance
                regime_performance = self._calculate_regime_performance(y_test, y_pred)
                
                evaluation_results[model_name] = {
                    'accuracy': accuracy,
                    'regime_performance': regime_performance,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                tprint(f"✅ [IMPROVED_REGIME_MODELS] {model_name} evaluated - Accuracy: {accuracy:.4f}", color="green")
                
            except Exception as e:
                tprint(f"❌ [IMPROVED_REGIME_MODELS] Evaluation failed for {model_name}: {e}", color="red")
                evaluation_results[model_name] = {'error': str(e)}
        
        return evaluation_results

    def _calculate_regime_performance(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regime-specific performance metrics."""
        regime_performance = {}
        unique_regimes = np.unique(y_true)
        
        for regime in unique_regimes:
            regime_mask = (y_true == regime)
            regime_true = y_true[regime_mask]
            regime_pred = y_pred[regime_mask]
            
            if len(regime_true) > 0:
                regime_accuracy = accuracy_score(regime_true, regime_pred)
                regime_performance[f'regime_{regime}'] = {
                    'accuracy': regime_accuracy,
                    'samples': len(regime_true)
                }
        
        return regime_performance

    def _create_artifacts(self, models: Dict[str, Any], evaluation_results: Dict[str, Any], 
                         feature_names: List[str], data_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Create artifacts for the trained models."""
        tprint("📦 [IMPROVED_REGIME_MODELS] Creating artifacts", color="cyan")
        
        artifacts = {
            'regime_models_training_result': {
                'models': models,
                'evaluation_results': evaluation_results,
                'feature_names': feature_names,
                'data_shape': data_shape,
                'config': self.validated_config,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return artifacts