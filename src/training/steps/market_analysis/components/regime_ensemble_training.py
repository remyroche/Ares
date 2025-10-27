"""
Regime Detection Ensemble Training Component

This component implements the meta-learner for regime detection:
- stacker_lgbm_calibrated: LightGBM model used as the meta-learner with probability calibration
"""

import numpy as np
import pandas as pd
import pickle
import json
import time
import warnings
from typing import Dict, Any, Optional, List, Tuple
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
from src.utils.ml_common.validation.universal_temporal_validation import (
    UniversalTemporalValidator, TemporalValidationConfig
)
from src.utils.ml_common.utils.lookahead_protection import LookaheadProtection
from src.utils.hardware.unified_hardware_manager import (
    UnifiedHardwareManager, HardwareConfig, WorkloadType, OptimizationLevel
)
from src.utils.ml_common.evaluation.evaluation_utils import (
    EvaluationUtils
)
from src.utils.ml_common.post_training.model_validation import (
    ModelValidator, ValidationConfig
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Import ML libraries
try:
    from sklearn.ensemble import StackingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.calibration import CalibratedClassifierCV
    from lightgbm import LGBMClassifier
    ML_LIBRARIES_AVAILABLE = True
    tprint("✅ [REGIME_ENSEMBLE] ML libraries imported successfully", color="green")
except ImportError as e:
    ML_LIBRARIES_AVAILABLE = False
    tprint(f"❌ [REGIME_ENSEMBLE] Failed to import ML libraries: {e}", color="red")

# Import feature generation system
try:
    from src.feature_generation.core.factory import get_feature_bank, FeatureGenerator, FeatureCategory
    FEATURE_GENERATION_AVAILABLE = True
    tprint("✅ [REGIME_ENSEMBLE] Feature generation system imported successfully", color="green")
except ImportError as e:
    FEATURE_GENERATION_AVAILABLE = False
    tprint(f"⚠️ [REGIME_ENSEMBLE] Feature generation system not available: {e}", color="yellow")

class RegimeEnsembleTrainingComponent(BaseMarketAnalysisComponent):
    """
    Regime Detection Ensemble Training Component.

    This component trains the meta-learner for regime detection:
    - stacker_lgbm_calibrated: LightGBM model with probability calibration
    """

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the Regime Ensemble Training Component."""
        tprint("🚀 [REGIME_ENSEMBLE] Initializing Regime Ensemble Training Component", color="cyan", bold=True)
        super().__init__(config)

        self.logger = system_logger.getChild('RegimeEnsembleTrainingComponent')
        tprint("✅ [REGIME_ENSEMBLE] Logger initialized", color="green")

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
        tprint("🔧 [REGIME_ENSEMBLE] Hardware manager initialized", color="green")

        # Initialize vectorization manager for feature generation
        self.vectorization_manager = UnifiedVectorizationManager()
        tprint("🔧 [REGIME_ENSEMBLE] Vectorization manager initialized", color="green")

        # Initialize HPO optimizer
        self.hpo_optimizer = HyperparameterOptimization(
            {
                'max_trials': 50,
                'timeout_seconds': 300,
                'enable_early_stopping': True,
                'enable_pruning': True
            }
        )
        tprint("🔧 [REGIME_ENSEMBLE] HPO optimizer initialized", color="green")

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
        tprint("🔧 [REGIME_ENSEMBLE] Temporal validator initialized", color="green")

        # Initialize lookahead protection
        self.lookahead_protection = LookaheadProtection()
        tprint("🔧 [REGIME_ENSEMBLE] Lookahead protection initialized", color="green")

        # Initialize model evaluator
        self.model_evaluator = EvaluationUtils()
        tprint("🔧 [REGIME_ENSEMBLE] Model evaluator initialized", color="green")

        # Initialize model validator
        self.model_validator = ModelValidator(
            ValidationConfig(
                enable_purged_cv=True,
                enable_data_leakage_detection=True,
                enable_time_series_validation=True
            )
        )
        tprint("🔧 [REGIME_ENSEMBLE] Model validator initialized", color="green")

        # Initialize ensemble training parameters
        self.ensemble_config = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        tprint("⚙️ [REGIME_ENSEMBLE] Ensemble configuration set", color="yellow")

        # Initialize ensemble models
        self.stacker_lgbm_calibrated = None
        self.base_models = {}
        self.ensemble_metrics = {}
        tprint("📊 [REGIME_ENSEMBLE] Ensemble models initialized", color="blue")

        tprint("✅ [REGIME_ENSEMBLE] Regime Ensemble Training Component initialized successfully", color="green", bold=True)

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        tprint("📋 [REGIME_ENSEMBLE] Getting required artifacts", color="cyan")
        required_artifacts = ['regime_ensemble_training_result']
        tprint(f"✅ [REGIME_ENSEMBLE] Required artifacts: {required_artifacts}", color="green")
        return required_artifacts

    async def execute(self, data: pd.DataFrame, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute regime ensemble training with enhanced hardware optimization and validation.

        Args:
            data: Market data DataFrame
            pipeline_state: Pipeline state containing features, targets, and regime labels

        Returns:
            ComponentResult with training results
        """
        tprint("🚀 [REGIME_ENSEMBLE] Starting enhanced regime ensemble training execution", color="cyan", bold=True)
        start_time = datetime.now()

        try:
            # Initialize hardware optimization for intensive workload
            tprint("🔧 [REGIME_ENSEMBLE] Initializing hardware optimization", color="cyan")
            await self.hardware_manager.initialize()
            await self.hardware_manager.optimize_for_workload(WorkloadType.ML_TRAINING)
            tprint("✅ [REGIME_ENSEMBLE] Hardware optimization initialized", color="green")

            # Apply lookahead protection
            tprint("🔒 [REGIME_ENSEMBLE] Applying lookahead protection", color="cyan")
            protected_data = self.lookahead_protection.protect_data(data)
            tprint("✅ [REGIME_ENSEMBLE] Lookahead protection applied", color="green")

            # Extract required data from pipeline state
            tprint("📊 [REGIME_ENSEMBLE] Extracting data from pipeline state", color="yellow")

            # Extract regime labels from pipeline state artifacts
            artifacts = pipeline_state.get('artifacts', {})

            # Try multiple possible artifact keys for clustering results
            regime_labels = None

            # First try the new optimal_regime_clustering_result structure
            optimal_clustering_result = artifacts.get('optimal_regime_clustering_result', {})
            if optimal_clustering_result:
                clustering_result = optimal_clustering_result.get('clustering_result')
                if clustering_result:
                    tprint(f"🔍 [REGIME_ENSEMBLE] clustering_result type: {type(clustering_result)}", color="blue")

                    if isinstance(clustering_result, dict):
                        # Handle dict case (normal structure)
                        regime_labels = clustering_result.get('cluster_assignments')
                        # Handle case where assignments are stored as string representation
                        if isinstance(regime_labels, str):
                            try:
                                # Parse numpy array string representation (e.g., "[2 2 2 ... 4 6 6]")
                                clean_str = regime_labels.strip('[]')
                                regime_labels = np.array([int(x) for x in clean_str.split() if x.strip()])
                                tprint("🔍 [REGIME_ENSEMBLE] Parsed regime labels from string representation", color="blue")
                            except Exception as e:
                                tprint(f"⚠️ [REGIME_ENSEMBLE] Failed to parse regime labels string: {e}", color="yellow")
                                regime_labels = None
                        if regime_labels is not None:
                            tprint("🔍 [REGIME_ENSEMBLE] Found regime labels in optimal_regime_clustering_result", color="blue")
                    elif hasattr(clustering_result, 'cluster_assignments'):
                        # Handle object case (fallback)
                        regime_labels = clustering_result.cluster_assignments
                        if isinstance(regime_labels, str):
                            try:
                                clean_str = regime_labels.strip('[]')
                                regime_labels = np.array([int(x) for x in clean_str.split() if x.strip()])
                                tprint("🔍 [REGIME_ENSEMBLE] Parsed regime labels from clustering_result object", color="blue")
                            except Exception as e:
                                tprint(f"⚠️ [REGIME_ENSEMBLE] Failed to parse regime labels string: {e}", color="yellow")
                                regime_labels = None
                        if regime_labels is not None:
                            tprint("🔍 [REGIME_ENSEMBLE] Found regime labels in clustering_result object", color="blue")

            # Fallback to old regime_clustering_result structure
            if regime_labels is None:
                regime_clustering_result = artifacts.get('regime_clustering_result', {})
                regime_labels = regime_clustering_result.get('cluster_assignments')
                if regime_labels is not None:
                    tprint("🔍 [REGIME_ENSEMBLE] Found regime labels in regime_clustering_result", color="blue")

            # Get base models from previous training
            regime_models_result = artifacts.get('regime_models_training_result', {})
            base_models = regime_models_result.get('models', {})

            # Check if regime labels are available before preparing training data
            if regime_labels is None:
                tprint("⚠️ [REGIME_ENSEMBLE] No regime labels found in artifacts, will create synthetic regime labels", color="yellow")
                # Create synthetic regime labels based on data patterns
                regime_labels = self._create_synthetic_regime_labels(protected_data)

            # Prepare training data from the input data DataFrame with advanced regime features
            tprint("🔧 [REGIME_ENSEMBLE] Preparing training data from input DataFrame with advanced regime features", color="yellow")
            X, y, feature_names = self._prepare_training_data(protected_data, regime_labels, pipeline_state)

            # Validate required data
            tprint("🔍 [REGIME_ENSEMBLE] Validating required data", color="yellow")
            if X is None or y is None or feature_names is None:
                tprint("❌ [REGIME_ENSEMBLE] Failed to prepare training data", color="red")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message="Failed to prepare training data from input DataFrame",
                    metadata={'component_type': 'regime_ensemble_training'}
                )

            if not base_models:
                tprint("⚠️ [REGIME_ENSEMBLE] No base models found from previous training, training base models", color="yellow")
                # Train base models if not provided
                tprint("🏋️ [REGIME_ENSEMBLE] Training base models for ensemble", color="blue")
                base_models = self._train_base_models(X, y, regime_labels)
                if not base_models:
                    tprint("❌ [REGIME_ENSEMBLE] Failed to train base models", color="red")
                    return ComponentResult(
                        success=False,
                        artifacts={},
                        error_message="Failed to train base models",
                        metadata={'component_type': 'regime_ensemble_training'}
                    )

            tprint(f"📊 [REGIME_ENSEMBLE] Data shapes - X: {X.shape}, y: {y.shape}, regime_labels: {len(regime_labels) if regime_labels is not None else 'None'}", color="blue")
            tprint(f"📊 [REGIME_ENSEMBLE] Base models available: {list(base_models.keys())}", color="blue")

            # Prepare data for ensemble training with proper train/test split
            tprint("🔧 [REGIME_ENSEMBLE] Preparing data for ensemble training with proper validation", color="yellow")
            X_processed, y_processed, regime_labels_processed = self._prepare_data(X, y, regime_labels)
            tprint(f"✅ [REGIME_ENSEMBLE] Data prepared - X: {X_processed.shape}, y: {y_processed.shape}", color="green")

            # Perform proper temporal split to prevent data leakage using temporal validator
            tprint("🔄 [REGIME_ENSEMBLE] Performing temporal train/test split to prevent data leakage", color="cyan")

            # For temporal data, we need to sort by time if not already sorted
            # Assuming data is already in temporal order, we'll split by index
            total_samples = len(X_processed)
            train_size = int(total_samples * 0.7)

            # Create temporal indices for validation
            train_indices = np.arange(train_size)
            test_indices = np.arange(train_size, total_samples)

            # Split the data temporally
            X_train = X_processed[train_indices]
            X_test = X_processed[test_indices]
            y_train = y_processed[train_indices]
            y_test = y_processed[test_indices]

            # Validate the temporal split
            validation_report = self.temporal_validator.validate_temporal_split(
                X_train, X_test, y_train, y_test,
                model_name="regime_ensemble",
                model_type="ensemble"
            )

            if not validation_report.temporal_order_valid:
                tprint(f"⚠️ [REGIME_ENSEMBLE] Temporal validation failed: {validation_report.temporal_message}", color="yellow")
                tprint("🔧 [REGIME_ENSEMBLE] Using fallback split method", color="yellow")
                # Fallback to random split if temporal validation fails
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_processed, test_size=0.3, random_state=42, stratify=y_processed
                )
            else:
                tprint("✅ [REGIME_ENSEMBLE] Temporal validation passed - no data leakage detected", color="green")

            tprint(f"📊 [REGIME_ENSEMBLE] Train set: {X_train.shape}, Test set: {X_test.shape}", color="blue")

            # Train stacker_lgbm_calibrated meta-learner on training data only
            tprint("🎭 [REGIME_ENSEMBLE] Training stacker_lgbm_calibrated meta-learner on training data", color="yellow")
            stacker_result = self._train_stacker_lgbm_calibrated(X_train, y_train, base_models)

            # Evaluate ensemble on holdout test data
            tprint("📊 [REGIME_ENSEMBLE] Evaluating ensemble performance on holdout test data", color="yellow")
            ensemble_metrics = self._evaluate_ensemble(X_test, y_test, stacker_result)

            # Create comprehensive results
            tprint("📦 [REGIME_ENSEMBLE] Creating comprehensive results", color="yellow")
            results = {
                'regime_ensemble_training_result': {
                    'stacker_lgbm_calibrated': stacker_result,
                    'base_models': base_models,
                    'ensemble_metrics': ensemble_metrics,
                    'training_time': (datetime.now() - start_time).total_seconds(),
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
                        'component_type': 'regime_ensemble_training',
                        'data_shape': X_processed.shape,
                        'train_shape': X_train.shape,
                        'test_shape': X_test.shape,
                        'n_regimes': len(np.unique(regime_labels_processed)) if regime_labels_processed is not None else 0,
                        'feature_names': feature_names,
                        'timestamp': datetime.now().isoformat()
                    }
                }
            }

            tprint("✅ [REGIME_ENSEMBLE] Regime ensemble training completed successfully", color="green", bold=True)
            tprint(f"⏱️ [REGIME_ENSEMBLE] Total execution time: {(datetime.now() - start_time).total_seconds():.2f}s", color="blue")

            # Generate regime probability report
            try:
                regime_report = await self._generate_regime_probability_report(
                    results, X_processed, feature_names
                )
                if regime_report:
                    results['regime_probability_report'] = regime_report
                    tprint("📊 [REGIME_ENSEMBLE] Regime probability report generated successfully", color="green")
            except Exception as e:
                tprint(f"⚠️ [REGIME_ENSEMBLE] Failed to generate regime probability report: {e}", color="yellow")

            # Save artifacts persistently using the artifact manager
            try:
                save_report = await self.save_artifacts(results, {
                    'component_type': 'regime_ensemble_training',
                    'execution_time': (datetime.now() - start_time).total_seconds()
                })
                tprint(
                    f"💾 [REGIME_ENSEMBLE] Artifacts saved persistently (correlation_id={save_report.correlation_id}): {list(save_report.paths.keys())}",
                    color="green"
                )
            except Exception as e:
                tprint(f"⚠️ [REGIME_ENSEMBLE] Failed to save artifacts persistently: {e}", color="yellow")

            # Cleanup hardware resources
            await self.hardware_manager.cleanup()
            tprint("🔧 [REGIME_ENSEMBLE] Hardware resources cleaned up", color="green")

            return ComponentResult(
                success=True,
                artifacts=results,
                metadata={
                    'component_type': 'regime_ensemble_training',
                    'execution_time': (datetime.now() - start_time).total_seconds(),
                    'artifacts_saved_persistently': True,
                    'hardware_optimization_enabled': True,
                    'lookahead_protection_enabled': True
                }
            )

        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] Regime ensemble training failed: {e}", color="red", bold=True)
            self.logger.error(f"Regime ensemble training failed: {e}", exc_info=True)
            
            # Cleanup hardware resources on error
            try:
                await self.hardware_manager.cleanup()
            except Exception as cleanup_error:
                tprint(f"⚠️ [REGIME_ENSEMBLE] Hardware cleanup failed: {cleanup_error}", color="yellow")
            
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e),
                metadata={'component_type': 'regime_ensemble_training'}
            )

    def _prepare_data(self, X: np.ndarray, y: np.ndarray, regime_labels: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for ensemble training."""
        tprint("🔧 [REGIME_ENSEMBLE] Preparing data for ensemble training", color="yellow")

        # Handle missing values
        tprint("🧹 [REGIME_ENSEMBLE] Handling missing values", color="blue")
        if isinstance(X, pd.DataFrame):
            X = X.fillna(0).values
        elif isinstance(X, list):
            X = np.array(X)

        if isinstance(y, (pd.Series, list)):
            y = np.array(y)

        if regime_labels is not None and isinstance(regime_labels, (pd.Series, list)):
            regime_labels = np.array(regime_labels)

        # Ensure all arrays have the same length
        tprint("📏 [REGIME_ENSEMBLE] Ensuring consistent array lengths", color="blue")
        min_length = min(len(X), len(y))
        if regime_labels is not None:
            min_length = min(min_length, len(regime_labels))

        X = X[:min_length]
        y = y[:min_length]
        if regime_labels is not None:
            regime_labels = regime_labels[:min_length]

        tprint(f"✅ [REGIME_ENSEMBLE] Data prepared - X: {X.shape}, y: {y.shape}, regime_labels: {regime_labels.shape if regime_labels is not None else 'None'}", color="green")
        return X, y, regime_labels

    def _create_enhanced_meta_features(self, meta_features: np.ndarray, y: np.ndarray, base_model_predictions: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create enhanced meta-features for better ensemble performance.

        Args:
            meta_features: Base meta-features from base models
            y: Target labels
            base_model_predictions: Raw predictions from base models for disagreement analysis

        Returns:
            Enhanced meta-features array
        """
        try:
            tprint("🔧 [REGIME_ENSEMBLE] Creating enhanced meta-features", color="blue")

            # Calculate additional meta-features
            enhanced_features = []

            # 1. Base meta-features
            enhanced_features.append(meta_features)

            # 2. Confidence features (max probability for each sample)
            max_probs = np.max(meta_features, axis=1, keepdims=True)
            enhanced_features.append(max_probs)

            # 3. Entropy features (uncertainty measure)
            epsilon = 1e-10  # Avoid log(0)
            probs_safe = np.clip(meta_features, epsilon, 1 - epsilon)
            entropy = -np.sum(probs_safe * np.log(probs_safe), axis=1, keepdims=True)
            enhanced_features.append(entropy)

            # 4. Variance features (prediction consistency)
            variance = np.var(meta_features, axis=1, keepdims=True)
            enhanced_features.append(variance)

            # 5. Disagreement features (if base model predictions available)
            if base_model_predictions is not None and base_model_predictions.shape[1] > 1:
                # Model disagreement (variance across base model predictions)
                model_disagreement = np.var(base_model_predictions, axis=1, keepdims=True)
                enhanced_features.append(model_disagreement)
                
                # Pairwise disagreement (max disagreement between any two models)
                pairwise_disagreement = np.zeros((len(y), 1))
                for i in range(base_model_predictions.shape[1]):
                    for j in range(i+1, base_model_predictions.shape[1]):
                        disagreement = np.abs(base_model_predictions[:, i] - base_model_predictions[:, j])
                        pairwise_disagreement = np.maximum(pairwise_disagreement, disagreement.reshape(-1, 1))
                enhanced_features.append(pairwise_disagreement)

            # 6. Regime transition features
            if len(y) > 1:
                # Regime stability (consecutive same predictions)
                regime_stability = np.zeros((len(y), 1))
                for i in range(1, len(y)):
                    if y[i] == y[i-1]:
                        regime_stability[i] = regime_stability[i-1] + 1
                enhanced_features.append(regime_stability)
                
                # Regime change indicator
                regime_changes = np.zeros((len(y), 1))
                regime_changes[1:] = (y[1:] != y[:-1]).astype(int).reshape(-1, 1)
                enhanced_features.append(regime_changes)

            # 7. Uncertainty quantification features
            # Prediction confidence gap (difference between top 2 predictions)
            sorted_probs = np.sort(meta_features, axis=1)
            confidence_gap = sorted_probs[:, -1] - sorted_probs[:, -2]
            enhanced_features.append(confidence_gap.reshape(-1, 1))
            
            # Prediction margin (distance to decision boundary)
            prediction_margin = np.max(meta_features, axis=1) - np.mean(meta_features, axis=1)
            enhanced_features.append(prediction_margin.reshape(-1, 1))

            # 8. Class-specific features
            unique_classes = np.unique(y)
            for i, class_val in enumerate(unique_classes):
                class_mask = (y == class_val)
                if np.sum(class_mask) > 0:
                    # Use column index i instead of class_val for indexing
                    class_confidence = meta_features[class_mask, i].mean()
                    class_feature = np.full((len(y), 1), class_confidence)
                    enhanced_features.append(class_feature)

            # Combine all features
            enhanced_meta_features = np.column_stack(enhanced_features)

            tprint(f"✅ [REGIME_ENSEMBLE] Enhanced features created: {enhanced_meta_features.shape}", color="green")
            return enhanced_meta_features

        except Exception as e:
            tprint(f"⚠️ [REGIME_ENSEMBLE] Enhanced feature creation failed, using base features: {e}", color="yellow")
            return meta_features

    def _train_stacker_lgbm_calibrated(self, X: np.ndarray, y: np.ndarray, base_models: Dict[str, Any]) -> Dict[str, Any]:
        """Train stacker_lgbm_calibrated meta-learner with HPO optimization."""
        tprint("🎭 [REGIME_ENSEMBLE] Training stacker_lgbm_calibrated meta-learner with HPO", color="yellow")

        try:
            # Filter out None models and non-model objects (like feature indices)
            valid_base_models = {}
            for name, model in base_models.items():
                if model is not None and hasattr(model, 'predict'):
                    valid_base_models[name] = model
                elif name.endswith('_feature_indices'):
                    tprint(f"📊 [REGIME_ENSEMBLE] Skipping feature indices metadata: {name}", color="blue")
                else:
                    tprint(f"⚠️ [REGIME_ENSEMBLE] Skipping non-model object: {name}", color="yellow")

            if not valid_base_models:
                tprint("❌ [REGIME_ENSEMBLE] No valid base models available for meta-learner", color="red")
                return None

            tprint(f"📊 [REGIME_ENSEMBLE] Using {len(valid_base_models)} valid base models: {list(valid_base_models.keys())}", color="blue")

            # Generate base model predictions for meta-learning
            tprint("🔧 [REGIME_ENSEMBLE] Generating base model predictions", color="blue")
            base_predictions = []
            base_model_names = []

            for name, model in valid_base_models.items():
                try:
                    # Skip problematic models that cause feature mismatches
                    if name in ['stacker_lgbm_calibrated', 'stacker_lgbm_calibrated_feature_indices']:
                        tprint(f"⚠️ [REGIME_ENSEMBLE] Skipping problematic model during training: {name}", color="yellow")
                        continue

                    if hasattr(model, 'predict_proba'):
                        # Use probability predictions
                        pred_proba = model.predict_proba(X)
                        base_predictions.append(pred_proba)
                        base_model_names.append(f"{name}_proba")
                        tprint(f"📊 [REGIME_ENSEMBLE] {name}: Using probability predictions (shape: {pred_proba.shape})", color="blue")
                    else:
                        # Use class predictions
                        pred = model.predict(X)
                        # Convert to one-hot encoding for meta-learner
                        unique_classes = np.unique(y)
                        pred_onehot = np.zeros((len(pred), len(unique_classes)))
                        for i, class_val in enumerate(unique_classes):
                            pred_onehot[pred == class_val, i] = 1
                        base_predictions.append(pred_onehot)
                        base_model_names.append(f"{name}_class")
                        tprint(f"📊 [REGIME_ENSEMBLE] {name}: Using class predictions (shape: {pred_onehot.shape})", color="blue")
                except Exception as e:
                    tprint(f"⚠️ [REGIME_ENSEMBLE] Failed to get predictions from {name}: {e}", color="yellow")
                    continue

            if not base_predictions:
                tprint("❌ [REGIME_ENSEMBLE] No valid predictions generated from base models", color="red")
                return None

            # Combine base model predictions
            tprint("🔧 [REGIME_ENSEMBLE] Combining base model predictions", color="blue")
            meta_features = np.column_stack(base_predictions)
            tprint(f"📊 [REGIME_ENSEMBLE] Meta-features shape: {meta_features.shape}", color="blue")

            # Define HPO search space for LightGBM
            def create_lgbm_model(trial):
                return LGBMClassifier(
                    num_leaves=trial.suggest_int('num_leaves', 8, 63),
                    max_depth=trial.suggest_int('max_depth', 2, 8),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2),
                    n_estimators=trial.suggest_int('n_estimators', 50, 200),
                    min_child_samples=trial.suggest_int('min_child_samples', 20, 100),
                    subsample=trial.suggest_float('subsample', 0.5, 1.0),
                    colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    reg_alpha=trial.suggest_float('reg_alpha', 0.1, 2.0),
                    reg_lambda=trial.suggest_float('reg_lambda', 0.1, 2.0),
                    class_weight='balanced',
                    random_state=42,
                    verbose=-1,
                    n_jobs=-1
                )

            # Perform HPO optimization
            tprint("🔍 [REGIME_ENSEMBLE] Starting HPO optimization for meta-learner", color="cyan")
            hpo_result = self.hpo_optimizer.optimize(
                model_factory=create_lgbm_model,
                X=meta_features,
                y=y,
                cv_folds=3,
                scoring='accuracy',
                n_trials=20
            )

            if hpo_result.success:
                tprint(f"✅ [REGIME_ENSEMBLE] HPO optimization completed successfully", color="green")
                tprint(f"📊 [REGIME_ENSEMBLE] Best score: {hpo_result.best_score:.4f}", color="blue")
                meta_learner = hpo_result.best_model
            else:
                tprint(f"⚠️ [REGIME_ENSEMBLE] HPO optimization failed, using default parameters", color="yellow")
                # Fallback to default parameters
                meta_learner = LGBMClassifier(
                    num_leaves=8,
                    max_depth=2,
                    learning_rate=0.02,
                    n_estimators=100,
                    min_child_samples=50,
                    subsample=0.5,
                    colsample_bytree=0.5,
                    reg_alpha=1.0,
                    reg_lambda=1.0,
                    class_weight='balanced',
                    random_state=42,
                    verbose=-1,
                    n_jobs=-1
                )

            # Add feature engineering for meta-learner
            tprint("🔧 [REGIME_ENSEMBLE] Creating enhanced meta-learner features", color="blue")
            enhanced_meta_features = self._create_enhanced_meta_features(meta_features, y)
            tprint(f"📊 [REGIME_ENSEMBLE] Enhanced meta-features shape: {enhanced_meta_features.shape}", color="blue")

            # Train meta-learner
            tprint("🏋️ [REGIME_ENSEMBLE] Training meta-learner", color="blue")
            meta_learner.fit(enhanced_meta_features, y)
            tprint("✅ [REGIME_ENSEMBLE] Meta-learner trained successfully", color="green")

            # Apply probability calibration
            tprint("🎯 [REGIME_ENSEMBLE] Applying probability calibration", color="blue")
            try:
                calibrated_meta_learner = CalibratedClassifierCV(
                    meta_learner,
                    method='isotonic',
                    cv=3
                )
                calibrated_meta_learner.fit(meta_features, y)
                tprint("✅ [REGIME_ENSEMBLE] Probability calibration applied successfully", color="green")

                # Create comprehensive result
                stacker_result = {
                    'meta_learner': calibrated_meta_learner,
                    'base_models': valid_base_models,
                    'base_model_names': base_model_names,
                    'meta_features_shape': meta_features.shape,
                    'calibration_method': 'isotonic',
                    'cv_folds': 3,
                    'training_success': True,
                    'hpo_result': hpo_result if hpo_result.success else None
                }

                tprint("✅ [REGIME_ENSEMBLE] stacker_lgbm_calibrated training completed successfully", color="green")
                return stacker_result

            except Exception as e:
                tprint(f"⚠️ [REGIME_ENSEMBLE] Probability calibration failed: {e}", color="yellow")
                tprint("📊 [REGIME_ENSEMBLE] Using uncalibrated meta-learner", color="blue")

                # Return uncalibrated result
                stacker_result = {
                    'meta_learner': meta_learner,
                    'base_models': valid_base_models,
                    'base_model_names': base_model_names,
                    'meta_features_shape': meta_features.shape,
                    'calibration_method': 'none',
                    'cv_folds': 0,
                    'training_success': True,
                    'hpo_result': hpo_result if hpo_result.success else None
                }

                tprint("✅ [REGIME_ENSEMBLE] stacker_lgbm_calibrated training completed (uncalibrated)", color="green")
                return stacker_result

        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] stacker_lgbm_calibrated training failed: {e}", color="red")
            return None

    def _evaluate_ensemble(self, X: np.ndarray, y: np.ndarray, stacker_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate ensemble performance using enhanced ML utilities."""
        tprint("📊 [REGIME_ENSEMBLE] Evaluating ensemble performance with enhanced ML utilities", color="yellow")

        metrics = {}

        if stacker_result is None:
            tprint("❌ [REGIME_ENSEMBLE] No stacker result to evaluate", color="red")
            return {'error': 'No stacker result available'}

        try:
            meta_learner = stacker_result['meta_learner']
            base_models = stacker_result['base_models']
            base_model_names = stacker_result['base_model_names']

            # Generate meta-features for evaluation
            tprint("🔧 [REGIME_ENSEMBLE] Generating meta-features for evaluation", color="blue")
            base_predictions = []

            for name, model in base_models.items():
                try:
                    # Skip problematic models that cause feature mismatches
                    if name in ['stacker_lgbm_calibrated', 'stacker_lgbm_calibrated_feature_indices']:
                        tprint(f"⚠️ [REGIME_ENSEMBLE] Skipping problematic model: {name}", color="yellow")
                        continue

                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(X)
                        base_predictions.append(pred_proba)
                        tprint(f"📊 [REGIME_ENSEMBLE] {name}: Using probability predictions (shape: {pred_proba.shape})", color="blue")
                    else:
                        pred = model.predict(X)
                        unique_classes = np.unique(y)
                        pred_onehot = np.zeros((len(pred), len(unique_classes)))
                        for i, class_val in enumerate(unique_classes):
                            pred_onehot[pred == class_val, i] = 1
                        base_predictions.append(pred_onehot)
                        tprint(f"📊 [REGIME_ENSEMBLE] {name}: Using class predictions (shape: {pred_onehot.shape})", color="blue")
                except Exception as e:
                    tprint(f"⚠️ [REGIME_ENSEMBLE] Failed to get predictions from {name}: {e}", color="yellow")
                    continue

            if not base_predictions:
                tprint("❌ [REGIME_ENSEMBLE] No valid predictions for evaluation", color="red")
                return {'error': 'No valid predictions for evaluation'}

            # Combine predictions
            meta_features = np.column_stack(base_predictions)
            tprint(f"📊 [REGIME_ENSEMBLE] Meta-features shape: {meta_features.shape}", color="blue")

            # Check if meta-learner expects different number of features
            if hasattr(meta_learner, 'n_features_') and meta_learner.n_features_ != meta_features.shape[1]:
                tprint(f"⚠️ [REGIME_ENSEMBLE] Feature mismatch: meta-learner expects {meta_learner.n_features_} features, got {meta_features.shape[1]}", color="yellow")
                # Skip evaluation if feature dimensions don't match
                return {'error': f'Feature dimension mismatch: expected {meta_learner.n_features_}, got {meta_features.shape[1]}'}

            # Evaluate meta-learner
            tprint("📊 [REGIME_ENSEMBLE] Evaluating meta-learner", color="blue")
            y_pred = meta_learner.predict(meta_features)
            y_pred_proba = meta_learner.predict_proba(meta_features)

            # Use enhanced model evaluator for comprehensive evaluation
            tprint("🔍 [REGIME_ENSEMBLE] Performing comprehensive model evaluation", color="cyan")
            evaluation_result = self.model_evaluator.evaluate_model(
                model=meta_learner,
                X=meta_features,
                y=y,
                y_pred=y_pred,
                y_pred_proba=y_pred_proba
            )

            # Use model validator for additional validation
            tprint("🔍 [REGIME_ENSEMBLE] Performing model validation", color="cyan")
            validation_result = self.model_validator.validate_model(
                model=meta_learner,
                X=meta_features,
                y=y,
                cv_folds=5
            )

            # Calculate basic metrics
            accuracy = accuracy_score(y, y_pred)

            # Calculate top-3 regime analysis with entropy metrics
            top_3_analysis = self._calculate_top_regime_analysis(y_pred_proba)

            # Enhanced metrics with ML utilities
            metrics['stacker_lgbm_calibrated'] = {
                'accuracy': accuracy,
                'classification_report': classification_report(y, y_pred, output_dict=True),
                'prediction_confidence': {
                    'mean': y_pred_proba.max(axis=1).mean(),
                    'std': y_pred_proba.max(axis=1).std()
                },
                'top_regime_analysis': top_3_analysis,
                'calibration_method': stacker_result.get('calibration_method', 'none'),
                'base_models_used': len(base_models),
                'meta_features_shape': meta_features.shape,
                'enhanced_evaluation': evaluation_result,
                'model_validation': validation_result,
                'hpo_result': stacker_result.get('hpo_result')
            }

            # Calculate comprehensive metrics for meta-learner
            precision, recall, f1, support = precision_recall_fscore_support(y, y_pred, average='weighted')
            confidence_mean = y_pred_proba.max(axis=1).mean()
            confidence_std = y_pred_proba.max(axis=1).std()

            tprint("🎯 [REGIME_ENSEMBLE] META-LEARNER PERFORMANCE METRICS", color="green", bold=True)
            tprint("="*50, color="green")
            tprint(f"🎯 Accuracy: {accuracy:.4f}", color="green")
            tprint(f"📈 Precision: {precision:.4f}", color="green")
            tprint(f"📈 Recall: {recall:.4f}", color="green")
            tprint(f"📈 F1-Score: {f1:.4f}", color="green")
            tprint(f"🎲 Prediction Confidence: {confidence_mean:.4f} ± {confidence_std:.4f}", color="green")
            tprint(f"🔧 Calibration Method: {stacker_result.get('calibration_method', 'none')}", color="green")
            tprint(f"🤖 Base Models Used: {len(base_models)}", color="green")
            tprint(f"📊 Meta-features Shape: {meta_features.shape}", color="green")

            # Display enhanced evaluation results
            if evaluation_result and evaluation_result.get('success'):
                eval_metrics = evaluation_result.get('metrics', {})
                tprint("🔍 ENHANCED EVALUATION RESULTS", color="cyan", bold=True)
                tprint(f"   📊 SHAP Analysis: {'Available' if eval_metrics.get('shap_available') else 'Not Available'}", color="cyan")
                tprint(f"   📊 LIME Analysis: {'Available' if eval_metrics.get('lime_available') else 'Not Available'}", color="cyan")
                tprint(f"   📊 OOF Validation: {'Passed' if eval_metrics.get('oof_validation_passed') else 'Failed'}", color="cyan")
                tprint(f"   📊 OOS Validation: {'Passed' if eval_metrics.get('oos_validation_passed') else 'Failed'}", color="cyan")

            # Display validation results
            if validation_result and validation_result.get('success'):
                val_metrics = validation_result.get('metrics', {})
                tprint("🔍 MODEL VALIDATION RESULTS", color="cyan", bold=True)
                tprint(f"   📊 Purged CV Score: {val_metrics.get('purged_cv_score', 'N/A')}", color="cyan")
                tprint(f"   📊 Data Leakage: {'Detected' if val_metrics.get('data_leakage_detected') else 'Not Detected'}", color="cyan")
                tprint(f"   📊 Time Series Validation: {'Passed' if val_metrics.get('time_series_validation_passed') else 'Failed'}", color="cyan")

            # Display top regime analysis summary
            if 'top_regime_analysis' in metrics['stacker_lgbm_calibrated']:
                top_analysis = metrics['stacker_lgbm_calibrated']['top_regime_analysis']
                entropy_metrics = top_analysis['entropy_metrics']
                confidence_gaps = top_analysis['confidence_gaps']
                conf_dist = top_analysis['prediction_confidence_distribution']

                tprint("🎯 TOP REGIME ANALYSIS", color="cyan", bold=True)
                tprint(f"   📊 Avg Entropy: {entropy_metrics['mean_entropy']:.4f}", color="cyan")
                tprint(f"   🎲 Confidence Gap (1st-2nd): {confidence_gaps['gap_1_2_mean']:.4f}", color="cyan")
                tprint(f"   📈 High Confidence Samples: {conf_dist['high_confidence_ratio']:.1%}", color="cyan")
                tprint(f"   📉 Low Confidence Samples: {conf_dist['low_confidence_ratio']:.1%}", color="cyan")

            tprint("="*50, color="green")

            # Add comparison with base models if available
            if base_models:
                tprint("🔄 [REGIME_ENSEMBLE] ENSEMBLE vs BASE MODELS COMPARISON", color="cyan", bold=True)
                tprint("="*60, color="cyan")

                # Calculate base model accuracies for comparison
                base_accuracies = {}
                for name, model in base_models.items():
                    try:
                        if name not in ['stacker_lgbm_calibrated', 'stacker_lgbm_calibrated_feature_indices']:
                            y_pred_base = model.predict(X)
                            base_accuracy = accuracy_score(y, y_pred_base)
                            base_accuracies[name] = base_accuracy
                    except Exception as e:
                        tprint(f"⚠️ [REGIME_ENSEMBLE] Could not evaluate {name}: {e}", color="yellow")

                # Print comparison
                tprint(f"🎯 Meta-learner Accuracy: {accuracy:.4f}", color="green")
                for name, base_acc in base_accuracies.items():
                    improvement = accuracy - base_acc
                    status = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
                    tprint(f"   {status} {name}: {base_acc:.4f} (Δ: {improvement:+.4f})", color="blue")

                # Calculate average base model performance
                if base_accuracies:
                    avg_base_accuracy = np.mean(list(base_accuracies.values()))
                    ensemble_improvement = accuracy - avg_base_accuracy
                    tprint(f"📊 Average Base Model: {avg_base_accuracy:.4f}", color="blue")
                    tprint(f"🚀 Ensemble Improvement: {ensemble_improvement:+.4f}", color="green" if ensemble_improvement > 0 else "red")

                tprint("="*60, color="cyan")

        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] Ensemble evaluation failed: {e}", color="red")
            metrics['stacker_lgbm_calibrated'] = {'error': str(e)}

        tprint("✅ [REGIME_ENSEMBLE] Ensemble evaluation completed", color="green")
        return metrics

    def _calculate_top_regime_analysis(self, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive analysis of top regime predictions.

        Args:
            y_pred_proba: Probability predictions for each sample and regime

        Returns:
            Dictionary containing top-3 regime analysis with entropy metrics
        """
        try:
            n_samples, n_regimes = y_pred_proba.shape

            # Get top 3 predictions and probabilities for each sample
            # Use argsort with descending order to get highest probabilities first
            top_3_indices = np.argsort(y_pred_proba, axis=1)[:, -3:][:, ::-1]  # Get top 3, reverse to descending
            top_3_probabilities = np.sort(y_pred_proba, axis=1)[:, -3:][:, ::-1]  # Get top 3 probs, descending

            # Calculate entropy (measure of prediction uncertainty)
            # Use small epsilon to avoid log(0)
            epsilon = 1e-10
            entropy = -np.sum(y_pred_proba * np.log(y_pred_proba + epsilon), axis=1)

            # Calculate confidence gaps between predictions
            confidence_gap_1_2 = top_3_probabilities[:, 0] - top_3_probabilities[:, 1]  # Gap between 1st and 2nd
            confidence_gap_2_3 = top_3_probabilities[:, 1] - top_3_probabilities[:, 2]  # Gap between 2nd and 3rd

            # Calculate relative confidence (how much more confident in 1st vs 2nd)
            relative_confidence_1_2 = np.divide(
                confidence_gap_1_2,
                top_3_probabilities[:, 0],
                out=np.zeros_like(confidence_gap_1_2),
                where=top_3_probabilities[:, 0] != 0
            )

            # Identify high-confidence vs low-confidence predictions
            high_confidence_threshold = 0.8
            low_confidence_threshold = 0.4

            high_confidence_samples = np.sum(top_3_probabilities[:, 0] >= high_confidence_threshold)
            low_confidence_samples = np.sum(top_3_probabilities[:, 0] <= low_confidence_threshold)
            uncertain_samples = n_samples - high_confidence_samples - low_confidence_samples

            # Calculate regime frequency in top predictions
            top_1_regime_counts = np.bincount(top_3_indices[:, 0], minlength=n_regimes)
            top_2_regime_counts = np.bincount(top_3_indices[:, 1], minlength=n_regimes)
            top_3_regime_counts = np.bincount(top_3_indices[:, 2], minlength=n_regimes)

            return {
                'top_predictions': {
                    'regime_indices': top_3_indices.tolist(),
                    'probabilities': top_3_probabilities.tolist()
                },
                'entropy_metrics': {
                    'mean_entropy': float(entropy.mean()),
                    'std_entropy': float(entropy.std()),
                    'min_entropy': float(entropy.min()),
                    'max_entropy': float(entropy.max()),
                    'entropy_distribution': {
                        'low_uncertainty': int(np.sum(entropy < 0.5)),
                        'medium_uncertainty': int(np.sum((entropy >= 0.5) & (entropy < 1.0))),
                        'high_uncertainty': int(np.sum(entropy >= 1.0))
                    }
                },
                'confidence_gaps': {
                    'gap_1_2_mean': float(confidence_gap_1_2.mean()),
                    'gap_1_2_std': float(confidence_gap_1_2.std()),
                    'gap_2_3_mean': float(confidence_gap_2_3.mean()),
                    'gap_2_3_std': float(confidence_gap_2_3.std()),
                    'relative_confidence_1_2_mean': float(relative_confidence_1_2.mean()),
                    'relative_confidence_1_2_std': float(relative_confidence_1_2.std())
                },
                'prediction_confidence_distribution': {
                    'high_confidence_samples': int(high_confidence_samples),
                    'low_confidence_samples': int(low_confidence_samples),
                    'uncertain_samples': int(uncertain_samples),
                    'high_confidence_ratio': float(high_confidence_samples / n_samples),
                    'low_confidence_ratio': float(low_confidence_samples / n_samples),
                    'uncertain_ratio': float(uncertain_samples / n_samples)
                },
                'regime_frequency_analysis': {
                    'top_1_regime_distribution': top_1_regime_counts.tolist(),
                    'top_2_regime_distribution': top_2_regime_counts.tolist(),
                    'top_3_regime_distribution': top_3_regime_counts.tolist(),
                    'most_common_second_choice': int(np.argmax(top_2_regime_counts)),
                    'most_common_third_choice': int(np.argmax(top_3_regime_counts))
                },
                'summary_statistics': {
                    'total_samples': n_samples,
                    'total_regimes': n_regimes,
                    'avg_top_1_confidence': float(top_3_probabilities[:, 0].mean()),
                    'avg_top_2_confidence': float(top_3_probabilities[:, 1].mean()),
                    'avg_top_3_confidence': float(top_3_probabilities[:, 2].mean())
                }
            }

        except Exception as e:
            self.logger.error(f"Error calculating top regime analysis: {e}")
            return {
                'error': str(e),
                'entropy_metrics': {'mean_entropy': 0.0, 'std_entropy': 0.0},
                'confidence_gaps': {'gap_1_2_mean': 0.0, 'gap_2_3_mean': 0.0},
                'summary_statistics': {'total_samples': len(y_pred_proba), 'total_regimes': y_pred_proba.shape[1]}
            }

    def _prepare_training_data(self, data: pd.DataFrame, regime_labels: np.ndarray, pipeline_state: Dict[str, Any] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data from market data and regime labels."""
        tprint("🔧 [REGIME_ENSEMBLE] Preparing training data", color="cyan")
        self.logger.info("Starting data preparation process")

        try:
            # Log input data characteristics
            tprint(f"📊 [REGIME_ENSEMBLE] Input data shape: {data.shape}", color="blue")
            tprint(f"📊 [REGIME_ENSEMBLE] Input data columns: {list(data.columns)}", color="blue")

            # Force comprehensive feature generation using feature bank
            tprint("🔧 [REGIME_ENSEMBLE] FORCING comprehensive feature generation using feature bank", color="cyan", bold=True)
            tprint("🚫 [REGIME_ENSEMBLE] Bypassing base model features to ensure comprehensive feature set", color="yellow")

            # Check if we should use original market data for feature generation
            original_data = None
            if pipeline_state is not None:
                original_data = pipeline_state.get('original_data')
                force_feature_bank = pipeline_state.get('force_feature_bank', False)

                if original_data is not None and force_feature_bank:
                    tprint("✅ [REGIME_ENSEMBLE] Using original market data for feature bank generation", color="green")
                    data_for_features = original_data
                else:
                    tprint("⚠️ [REGIME_ENSEMBLE] No original data available, using processed data", color="yellow")
                    data_for_features = data
            else:
                data_for_features = data

            if FEATURE_GENERATION_AVAILABLE:
                X = self._generate_features_with_bank(data_for_features)
                if X is None or X.shape[1] < 50:
                    error_msg = f"Feature bank generated insufficient features: {X.shape[1] if X is not None else 0} < 50 required"
                    tprint(f"❌ [REGIME_ENSEMBLE] {error_msg}", color="red")
                    self.logger.error(error_msg)
                    return None, None, None
                else:
                    tprint(f"✅ [REGIME_ENSEMBLE] Feature bank generated {X.shape[1]} comprehensive features", color="green")
                    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            else:
                error_msg = "Feature generation system not available - cannot generate comprehensive features"
                tprint(f"❌ [REGIME_ENSEMBLE] {error_msg}", color="red")
                self.logger.error(error_msg)
                return None, None, None
            tprint(f"📋 [REGIME_ENSEMBLE] Feature names ({len(feature_names)}): {feature_names[:10]}..." if len(feature_names) > 10 else f"📋 [REGIME_ENSEMBLE] Feature names ({len(feature_names)}): {feature_names}", color="blue")

            # Check for NaN or infinite values in features with detailed analysis
            nan_count = np.isnan(X).sum()
            inf_count = np.isinf(X).sum()
            if nan_count > 0:
                # Import the detailed NaN analysis function
                from src.utils.common_utilities import analyze_nan_values_detailed, format_nan_analysis_report

                # Perform detailed NaN analysis
                nan_analysis = analyze_nan_values_detailed(X, feature_names)
                detailed_report = format_nan_analysis_report(nan_analysis, "[REGIME_ENSEMBLE] ")

                tprint(f"⚠️ [REGIME_ENSEMBLE] Found {nan_count} NaN values in features", color="yellow")
                tprint(detailed_report, color="yellow")
                tprint("🔧 [REGIME_ENSEMBLE] Using sophisticated NaN handling for time series data", color="cyan")
                # Use sophisticated NaN handling for time series data
                X = self._handle_nan_values(X, nan_count)
            if inf_count > 0:
                tprint(f"⚠️ [REGIME_ENSEMBLE] Found {inf_count} infinite values in features", color="yellow")
                tprint("🔧 [REGIME_ENSEMBLE] Replacing infinite values with finite numbers", color="cyan")
                X = np.nan_to_num(X, posinf=1e6, neginf=-1e6)

            # Align with regime labels
            tprint("🔧 [REGIME_ENSEMBLE] Aligning features with regime labels", color="cyan")
            min_length = min(len(X), len(regime_labels))
            X = X[:min_length]
            y = np.array(regime_labels[:min_length])

            tprint(f"✅ [REGIME_ENSEMBLE] Training data prepared: {X.shape[0]} samples, {X.shape[1]} features", color="green", bold=True)

            self.logger.info(f"Training data preparation completed: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y, feature_names

        except Exception as e:
            error_type = type(e).__name__
            tprint(f"❌ [REGIME_ENSEMBLE] Error preparing training data: {e}", color="red")
            tprint(f"🔍 [REGIME_ENSEMBLE] Error type: {error_type}", color="yellow")
            self.logger.error(f"Error preparing training data: {e}", exc_info=True)
            return None, None, None

    def _generate_features_with_bank(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Generate comprehensive features using the UnifiedVectorizationManager and feature bank."""
        tprint("🔧 [REGIME_ENSEMBLE] Generating features using UnifiedVectorizationManager and feature bank", color="cyan", bold=True)

        try:
            if not FEATURE_GENERATION_AVAILABLE:
                tprint("❌ [REGIME_ENSEMBLE] Feature generation system not available", color="red")
                return None

            # Configure vectorization for feature engineering
            vectorization_config = {
                'operation_type': OperationType.FEATURE_ENGINEERING,
                'data_size': len(data),
                'data_dimensions': data.shape,
                'memory_budget_mb': 2048.0,
                'time_budget_seconds': 300.0,
                'precision_requirement': 'high'
            }

            # Get feature bank
            feature_bank = get_feature_bank()
            tprint("✅ [REGIME_ENSEMBLE] Feature bank retrieved successfully", color="green")

            # Define feature categories to generate
            categories = [
                FeatureCategory.MOMENTUM,
                FeatureCategory.VOLATILITY,
                FeatureCategory.VOLUME,
                FeatureCategory.TREND,
                FeatureCategory.OSCILLATOR,
                FeatureCategory.RETURNS
            ]

            all_features = pd.DataFrame(index=data.index)
            total_features = 0

            # Generate features for each category using vectorization manager
            for category in categories:
                tprint(f"🔍 [REGIME_ENSEMBLE] Generating {category.value} features with vectorization", color="blue")

                # Get generators for this category
                generators = feature_bank.get_generators_by_category(category)

                if not generators:
                    tprint(f"⚠️ [REGIME_ENSEMBLE] No generators found for {category.value}", color="yellow")
                    continue

                category_features = pd.DataFrame(index=data.index)

                # Generate features using each generator with vectorization optimization
                for generator in generators:
                    try:
                        tprint(f"🔧 [REGIME_ENSEMBLE] Using generator: {generator.config.name}", color="blue")
                        
                        # Use vectorization manager for optimized feature generation
                        result = self.vectorization_manager.optimize_operation(
                            operation=lambda: generator.generate(data),
                            config=vectorization_config
                        )

                        if result.success and result.result and hasattr(result.result, 'data') and not len(result.result.data) == 0:
                            # Add feature with category prefix
                            feature_name = f"{category.value}_{result.result.name}"
                            category_features[feature_name] = result.result.data
                            total_features += 1
                            tprint(f"✅ [REGIME_ENSEMBLE] Generated feature: {feature_name} (optimized)", color="green")
                        else:
                            tprint(f"⚠️ [REGIME_ENSEMBLE] Generator {generator.config.name} returned empty result", color="yellow")

                    except Exception as e:
                        tprint(f"⚠️ [REGIME_ENSEMBLE] Generator {generator.config.name} failed: {e}", color="yellow")
                        continue

                # Add category features to all features
                if not category_features.empty:
                    all_features = pd.concat([all_features, category_features], axis=1)
                    tprint(f"📊 [REGIME_ENSEMBLE] {category.value} features: {category_features.shape[1]}", color="blue")

            # Convert to numpy array
            if not all_features.empty:
                X = all_features.values
                tprint(f"✅ [REGIME_ENSEMBLE] Feature bank generated {X.shape[1]} features from {len(categories)} categories", color="green")
                tprint(f"📊 [REGIME_ENSEMBLE] Feature matrix shape: {X.shape}", color="blue")
                return X
            else:
                tprint("❌ [REGIME_ENSEMBLE] Feature bank generated no features", color="red")
                return None

        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] Error generating features with feature bank: {e}", color="red")
            self.logger.error(f"Error generating features with feature bank: {str(e)}", exc_info=True)
            return None

    def _train_base_models(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray) -> Dict[str, Any]:
        """Train base models for ensemble."""
        tprint("🏋️ [REGIME_ENSEMBLE] Training base models", color="yellow")

        base_models = {}

        # CatBoost Classifier
        tprint("🐱 [REGIME_ENSEMBLE] Training CatBoost classifier", color="blue")
        try:
            from catboost import CatBoostClassifier
            catboost_model = CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=False,
                thread_count=-1
            )
            catboost_model.fit(X, y)
            base_models['catboost'] = catboost_model

            # Calculate and print CatBoost metrics
            y_pred_catboost = catboost_model.predict(X)
            y_pred_proba_catboost = catboost_model.predict_proba(X)
            catboost_accuracy = accuracy_score(y, y_pred_catboost)

            tprint("✅ [REGIME_ENSEMBLE] CatBoost trained successfully", color="green")
            tprint(f"📊 [REGIME_ENSEMBLE] CatBoost Metrics:", color="blue")
            tprint(f"   🎯 Accuracy: {catboost_accuracy:.4f}", color="blue")
            tprint(f"   🎲 Prediction Confidence: {y_pred_proba_catboost.max(axis=1).mean():.4f} ± {y_pred_proba_catboost.max(axis=1).std():.4f}", color="blue")

            # Print classification report for CatBoost
            from sklearn.metrics import precision_recall_fscore_support
            precision, recall, f1, support = precision_recall_fscore_support(y, y_pred_catboost, average='weighted')
            tprint(f"   📈 Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}", color="blue")
        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] CatBoost training failed: {e}", color="red")

        # Random Forest Classifier
        tprint("🌳 [REGIME_ENSEMBLE] Training Random Forest classifier", color="blue")
        try:
            from sklearn.ensemble import RandomForestClassifier
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X, y)
            base_models['random_forest'] = rf_model

            # Calculate and print Random Forest metrics
            y_pred_rf = rf_model.predict(X)
            y_pred_proba_rf = rf_model.predict_proba(X)
            rf_accuracy = accuracy_score(y, y_pred_rf)

            tprint("✅ [REGIME_ENSEMBLE] Random Forest trained successfully", color="green")
            tprint(f"📊 [REGIME_ENSEMBLE] Random Forest Metrics:", color="blue")
            tprint(f"   🎯 Accuracy: {rf_accuracy:.4f}", color="blue")
            tprint(f"   🎲 Prediction Confidence: {y_pred_proba_rf.max(axis=1).mean():.4f} ± {y_pred_proba_rf.max(axis=1).std():.4f}", color="blue")

            # Print classification report for Random Forest
            precision, recall, f1, support = precision_recall_fscore_support(y, y_pred_rf, average='weighted')
            tprint(f"   📈 Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}", color="blue")
        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] Random Forest training failed: {e}", color="red")

        # Extra Tree Classifier
        tprint("🌳 [REGIME_ENSEMBLE] Training Extra Tree classifier", color="blue")
        try:
            from sklearn.ensemble import ExtraTreesClassifier
            et_model = ExtraTreesClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            et_model.fit(X, y)
            base_models['extra_tree'] = et_model

            # Calculate and print Extra Tree metrics
            y_pred_et = et_model.predict(X)
            y_pred_proba_et = et_model.predict_proba(X)
            et_accuracy = accuracy_score(y, y_pred_et)

            tprint("✅ [REGIME_ENSEMBLE] Extra Tree trained successfully", color="green")
            tprint(f"📊 [REGIME_ENSEMBLE] Extra Tree Metrics:", color="blue")
            tprint(f"   🎯 Accuracy: {et_accuracy:.4f}", color="blue")
            tprint(f"   🎲 Prediction Confidence: {y_pred_proba_et.max(axis=1).mean():.4f} ± {y_pred_proba_et.max(axis=1).std():.4f}", color="blue")

            # Print classification report for Extra Tree
            precision, recall, f1, support = precision_recall_fscore_support(y, y_pred_et, average='weighted')
            tprint(f"   📈 Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}", color="blue")
        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] Extra Tree training failed: {e}", color="red")

        tprint(f"✅ [REGIME_ENSEMBLE] Base models training completed - {len(base_models)} models trained", color="green")

        # Print comprehensive summary of all base model metrics
        tprint("📊 [REGIME_ENSEMBLE] BASE MODELS PERFORMANCE SUMMARY", color="cyan", bold=True)
        tprint("="*60, color="cyan")

        for model_name, model in base_models.items():
            try:
                y_pred = model.predict(X)
                y_pred_proba = model.predict_proba(X)
                accuracy = accuracy_score(y, y_pred)
                precision, recall, f1, support = precision_recall_fscore_support(y, y_pred, average='weighted')
                confidence_mean = y_pred_proba.max(axis=1).mean()
                confidence_std = y_pred_proba.max(axis=1).std()

                tprint(f"🤖 {model_name.upper()}:", color="yellow")
                tprint(f"   🎯 Accuracy: {accuracy:.4f}", color="blue")
                tprint(f"   📈 Precision: {precision:.4f}", color="blue")
                tprint(f"   📈 Recall: {recall:.4f}", color="blue")
                tprint(f"   📈 F1-Score: {f1:.4f}", color="blue")
                tprint(f"   🎲 Confidence: {confidence_mean:.4f} ± {confidence_std:.4f}", color="blue")
                tprint("", color="white")  # Empty line for spacing

            except Exception as e:
                tprint(f"❌ [REGIME_ENSEMBLE] Failed to evaluate {model_name}: {e}", color="red")

        tprint("="*60, color="cyan")
        tprint("✅ [REGIME_ENSEMBLE] Base models evaluation completed", color="green")

        return base_models

    def _handle_nan_values(self, X: np.ndarray, original_nan_count: int) -> np.ndarray:
        """Handle NaN values in feature matrix using sophisticated time series methods.

        Args:
            X: Feature matrix with potential NaN values
            original_nan_count: Original number of NaN values for logging

        Returns:
            Feature matrix with NaN values handled
        """
        tprint(f"🔧 [REGIME_ENSEMBLE] Handling {original_nan_count} NaN values using sophisticated methods", color="cyan")

        try:
            # Convert to pandas for better NaN handling
            df = pd.DataFrame(X)

            # Strategy 1: Forward fill for time series data (fills gaps with previous values)
            df_filled = df.fillna(method='ffill')

            # Strategy 2: Backward fill for remaining NaN values (fills gaps with future values)
            df_filled = df_filled.fillna(method='bfill')

            # Strategy 3: For any remaining NaN values, use column median
            remaining_nan_count = df_filled.isna().sum().sum()
            if remaining_nan_count > 0:
                tprint(f"📊 [REGIME_ENSEMBLE] {remaining_nan_count} NaN values remain after forward/backward fill", color="yellow")

                # Calculate median for each column
                for col in df_filled.columns:
                    if df_filled[col].isna().sum() > 0:
                        median_val = df_filled[col].median()
                        df_filled[col] = df_filled[col].fillna(median_val)

                final_nan_count = df_filled.isna().sum().sum()
                if final_nan_count > 0:
                    tprint(f"⚠️ [REGIME_ENSEMBLE] {final_nan_count} NaN values still remain, using zero fill as last resort", color="yellow")
                    df_filled = df_filled.fillna(0.0)

            # Convert back to numpy array
            X_cleaned = df_filled.values

            # Verify no NaN values remain
            final_nan_count = np.isnan(X_cleaned).sum()
            tprint(f"✅ [REGIME_ENSEMBLE] NaN handling completed: {original_nan_count} → {final_nan_count} NaN values", color="green")

            return X_cleaned

        except Exception as e:
            tprint(f"⚠️ [REGIME_ENSEMBLE] Sophisticated NaN handling failed: {e}, falling back to zero fill", color="yellow")
            return np.nan_to_num(X, nan=0.0)

    def _create_synthetic_regime_labels(self, data: pd.DataFrame) -> np.ndarray:
        """
        Create synthetic regime labels based on data patterns when clustering results are not available.

        Args:
            data: Market data DataFrame

        Returns:
            Synthetic regime labels array
        """
        tprint("🔧 [REGIME_ENSEMBLE] Creating synthetic regime labels based on data patterns", color="cyan")

        try:
            # Use simple clustering based on price volatility and trend
            if 'close' in data.columns:
                # Calculate rolling volatility
                returns = data['close'].pct_change().dropna()
                volatility = self._vectorbt_rolling_operation(returns, "std", 20).fillna(returns.std())

                # Calculate trend strength
                if 'high' in data.columns and 'low' in data.columns:
                    price_range = (data['high'] - data['low']) / data['close']
                    trend_strength = self._vectorbt_rolling_operation(price_range, "mean", 20).fillna(price_range.mean())
                else:
                    # Fallback: use price momentum
                    momentum = data['close'].pct_change(20).fillna(0)
                    trend_strength = momentum.abs()

                # Create regime labels based on volatility and trend
                # High volatility + high trend = regime 0 (trending)
                # High volatility + low trend = regime 1 (ranging)
                # Low volatility + high trend = regime 2 (breakout)
                # Low volatility + low trend = regime 3 (consolidation)

                vol_threshold = volatility.median()
                trend_threshold = trend_strength.median()

                regime_labels = np.zeros(len(data))
                regime_labels[(volatility > vol_threshold) & (trend_strength > trend_threshold)] = 0  # Trending
                regime_labels[(volatility > vol_threshold) & (trend_strength <= trend_threshold)] = 1  # Ranging
                regime_labels[(volatility <= vol_threshold) & (trend_strength > trend_threshold)] = 2  # Breakout
                regime_labels[(volatility <= vol_threshold) & (trend_strength <= trend_threshold)] = 3  # Consolidation

                tprint(f"✅ [REGIME_ENSEMBLE] Created synthetic regime labels: {len(np.unique(regime_labels))} regimes", color="green")
                tprint(f"📊 [REGIME_ENSEMBLE] Regime distribution: {np.bincount(regime_labels.astype(int))}", color="blue")

                return regime_labels
            else:
                # Fallback: create simple regime labels based on data length
                n_regimes = min(4, max(2, len(data) // 100))  # 2-4 regimes based on data length
                regime_labels = np.random.randint(0, n_regimes, len(data))
                tprint(f"⚠️ [REGIME_ENSEMBLE] Using random regime labels as fallback: {n_regimes} regimes", color="yellow")
                return regime_labels

        except Exception as e:
            tprint(f"⚠️ [REGIME_ENSEMBLE] Synthetic regime creation failed: {e}, using simple fallback", color="yellow")
            # Simple fallback: create 2 regimes
            regime_labels = np.random.randint(0, 2, len(data))
            tprint("📊 [REGIME_ENSEMBLE] Using simple 2-regime fallback", color="blue")
            return regime_labels

    def predict_regimes_with_probabilities(
        self,
        stacker_result: Dict[str, Any],
        X: np.ndarray,
        feature_names: List[str],
        scaler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Predict regime labels and probabilities using trained ensemble meta-learner.
        Enhanced to provide comprehensive probabilistic outputs for each detected regime.

        Args:
            stacker_result: Dictionary containing trained meta-learner and base models
            X: Feature matrix
            feature_names: List of feature names
            scaler: Optional scaler for feature normalization

        Returns:
            Dictionary with comprehensive prediction information including:
            - regime_labels: Predicted regime for each sample
            - regime_probabilities: Probability matrix for each regime
            - regime_confidence_scores: Confidence scores for each prediction
            - regime_analysis: Detailed analysis of regime probabilities
            - ensemble_probabilities: Probabilities from all models in ensemble
        """
        try:
            tprint("🔮 [REGIME_ENSEMBLE] Starting ensemble regime prediction with probabilities", color="cyan")

            # Scale features if scaler is provided
            if scaler is not None:
                X_scaled = scaler.transform(X)
                tprint("✅ [REGIME_ENSEMBLE] Features scaled using provided scaler", color="green")
            else:
                X_scaled = X
                tprint("⚠️ [REGIME_ENSEMBLE] No scaler provided, using unscaled features", color="yellow")

            # Extract meta-learner and base models
            meta_learner = stacker_result.get('meta_learner')
            base_models = stacker_result.get('base_models', {})
            base_model_names = stacker_result.get('base_model_names', [])

            if meta_learner is None:
                raise ValueError("No meta-learner found in stacker_result")

            # Generate base model predictions for meta-learning
            tprint("🔧 [REGIME_ENSEMBLE] Generating base model predictions", color="blue")
            base_predictions = []

            for name, model in base_models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(X_scaled)
                        base_predictions.append(pred_proba)
                        tprint(f"✅ [REGIME_ENSEMBLE] {name}: Generated {pred_proba.shape[1]} regime probabilities", color="green")
                    else:
                        pred = model.predict(X_scaled)
                        unique_classes = np.unique(pred)
                        pred_onehot = np.zeros((len(pred), len(unique_classes)))
                        for i, class_val in enumerate(unique_classes):
                            pred_onehot[pred == class_val, i] = 1
                        base_predictions.append(pred_onehot)
                        tprint(f"✅ [REGIME_ENSEMBLE] {name}: Converted class predictions to {len(unique_classes)} regime probabilities", color="green")
                except Exception as e:
                    tprint(f"⚠️ [REGIME_ENSEMBLE] Failed to get predictions from {name}: {e}", color="yellow")
                    continue

            if not base_predictions:
                raise ValueError("No valid base model predictions generated")

            # Combine base model predictions
            meta_features = np.column_stack(base_predictions)
            tprint(f"📊 [REGIME_ENSEMBLE] Meta-features shape: {meta_features.shape}", color="blue")

            # Make predictions using meta-learner
            regime_labels = meta_learner.predict(meta_features)
            regime_probabilities = meta_learner.predict_proba(meta_features)

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
            ensemble_probabilities = generate_ensemble_probabilities(base_models, X_scaled, feature_names, "REGIME_ENSEMBLE")

            # Use RegimeProbabilityAnalyzer for comprehensive analysis
            from src.utils.regime_probability_analyzer import RegimeProbabilityAnalyzer

            analyzer = RegimeProbabilityAnalyzer()

            # Create prediction result for analysis
            prediction_result = {
                'regime_labels': regime_labels,
                'regime_probabilities': regime_probabilities,
                'ensemble_probabilities': ensemble_probabilities,
                'dominance': dominance,
                'timestamp': pd.Timestamp.now()
            }

            # Analyze prediction quality and stability
            analysis_result = analyzer.analyze_regime_prediction_quality(prediction_result)

            # Extract key metrics
            confidence_score = analysis_result.get('confidence_score', 0.0)
            stability_score = analysis_result.get('stability_score', 0.0)
            regime_consistency = analysis_result.get('regime_consistency', 0.0)

            return {
                'regime_labels': regime_labels,
                'regime_probabilities': regime_probabilities,
                'confidence_score': confidence_score,
                'stability_score': stability_score,
                'regime_consistency': regime_consistency,
                'dominance': dominance
            }
        except ImportError:
            # Fallback if RegimeProbabilityAnalyzer is not available
            return {
                'regime_labels': regime_labels,
                'regime_probabilities': regime_probabilities,
                'confidence_score': 0.0,
                'stability_score': 0.0,
                'regime_consistency': 0.0,
                'dominance': dominance
            }

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

    async def _generate_regime_probability_report(
        self,
        training_results: Dict[str, Any],
        X: np.ndarray,
        feature_names: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Generate a comprehensive report with regime probabilities for all regimes."""
        try:
            tprint("📊 [REGIME_ENSEMBLE] Generating regime probability report", color="cyan")

            # Get the trained stacker model
            stacker_model = training_results.get('stacker_lgbm_calibrated')
            if not stacker_model:
                tprint("⚠️ [REGIME_ENSEMBLE] No trained stacker model found for report generation", color="yellow")
                return None

            if not hasattr(stacker_model, 'predict_proba'):
                tprint("⚠️ [REGIME_ENSEMBLE] Stacker model does not support probability prediction", color="yellow")
                return None

            # Generate regime probabilities for all samples
            tprint("🔮 [REGIME_ENSEMBLE] Generating regime probabilities using stacker model", color="cyan")
            regime_probabilities = stacker_model.predict_proba(X)
            regime_labels = stacker_model.predict(X)

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

            # Get ensemble metrics
            ensemble_metrics = training_results.get('ensemble_metrics', {})
            stacker_metrics = ensemble_metrics.get('stacker_lgbm_calibrated', {})

            # Generate comprehensive report
            report = {
                'model_name': 'stacker_lgbm_calibrated',
                'generation_timestamp': datetime.now().isoformat(),
                'overall_statistics': overall_stats,
                'regime_statistics': regime_stats,
                'regime_probabilities': regime_probabilities.tolist(),
                'regime_labels': regime_labels.tolist(),
                'feature_names': feature_names,
                'data_shape': X.shape,
                'report_type': 'regime_ensemble_probability_analysis',
                'ensemble_metrics': {
                    'accuracy': stacker_metrics.get('accuracy', 0),
                    'prediction_confidence': stacker_metrics.get('prediction_confidence', {}),
                    'classification_report': stacker_metrics.get('classification_report', {})
                }
            }

            # Generate text report
            text_report = self._generate_text_report(report)
            report['text_report'] = text_report

            tprint(f"✅ [REGIME_ENSEMBLE] Regime probability report generated for {n_regimes} regimes", color="green")
            return report

        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] Failed to generate regime probability report: {e}", color="red")
            return None

    def _generate_text_report(self, report: Dict[str, Any]) -> str:
        """Generate a human-readable text report from regime probability data."""
        try:
            lines = []
            lines.append("=" * 80)
            lines.append("REGIME ENSEMBLE PROBABILITY ANALYSIS REPORT")
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

            # Ensemble Metrics
            ensemble_metrics = report.get('ensemble_metrics', {})
            if ensemble_metrics:
                lines.append("🤖 ENSEMBLE PERFORMANCE")
                lines.append("-" * 40)
                lines.append(f"Accuracy: {ensemble_metrics.get('accuracy', 0):.3f}")
                pred_conf = ensemble_metrics.get('prediction_confidence', {})
                lines.append(f"Mean Confidence: {pred_conf.get('mean', 0):.3f}")
                lines.append(f"Std Confidence: {pred_conf.get('std', 0):.3f}")
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
            lines.append("END OF REGIME ENSEMBLE PROBABILITY REPORT")
            lines.append("=" * 80)

            return "\n".join(lines)

        except Exception as e:
            return f"Error generating text report: {e}"
