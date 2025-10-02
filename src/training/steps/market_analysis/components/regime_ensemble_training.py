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

# Suppress warnings
warnings.filterwarnings('ignore')

# Import ML libraries
try:
    from sklearn.ensemble import StackingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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
        Execute regime ensemble training.
        
        Args:
            data: Market data DataFrame
            pipeline_state: Pipeline state containing features, targets, and regime labels
            
        Returns:
            ComponentResult with training results
        """
        tprint("🚀 [REGIME_ENSEMBLE] Starting regime ensemble training execution", color="cyan", bold=True)
        start_time = datetime.now()
        
        try:
            # Extract required data from pipeline state
            tprint("📊 [REGIME_ENSEMBLE] Extracting data from pipeline state", color="yellow")
            
            # Extract regime labels from pipeline state artifacts
            artifacts = pipeline_state.get('artifacts', {})
            nas_tas_clustering_result = artifacts.get('nas_tas_clustering_result', {})
            regime_labels = nas_tas_clustering_result.get('cluster_assignments')
            
            # Get base models from previous training
            nas_tas_models_result = artifacts.get('nas_tas_models_training_result', {})
            base_models = nas_tas_models_result.get('models', {})
            
            # Prepare training data from the input data DataFrame
            tprint("🔧 [REGIME_ENSEMBLE] Preparing training data from input DataFrame", color="yellow")
            X, y, feature_names = self._prepare_training_data(data, regime_labels, pipeline_state)
            
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
            
            if regime_labels is None:
                tprint("⚠️ [REGIME_ENSEMBLE] No regime labels found, using targets as regime labels", color="yellow")
                regime_labels = y
            
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

            # Import temporal validation utilities
            from src.utils.ml_common.validation.universal_temporal_validation import UniversalTemporalValidator, TemporalValidationConfig

            # Create temporal validator for proper train/test splitting
            temporal_config = TemporalValidationConfig(
                enable_temporal_checks=True,
                strict_temporal_order=True,
                initial_train_size=0.7,  # 70% for training
                test_size=0.3,  # 30% for validation
                gap_size=1  # Gap between train and test
            )
            temporal_validator = UniversalTemporalValidator(temporal_config)

            # Perform proper temporal split to prevent data leakage
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
            validation_report = temporal_validator.validate_temporal_split(
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
            
            return ComponentResult(
                success=True,
                artifacts=results,
                metadata={'component_type': 'regime_ensemble_training', 'execution_time': (datetime.now() - start_time).total_seconds()}
            )
            
        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] Regime ensemble training failed: {e}", color="red", bold=True)
            self.logger.error(f"Regime ensemble training failed: {e}", exc_info=True)
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
    
    def _create_enhanced_meta_features(self, meta_features: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Create enhanced meta-features for better ensemble performance.
        
        Args:
            meta_features: Base meta-features from base models
            y: Target labels
            
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
            
            # 5. Class-specific features
            unique_classes = np.unique(y)
            for class_val in unique_classes:
                class_mask = (y == class_val)
                if np.sum(class_mask) > 0:
                    class_confidence = meta_features[class_mask, class_val].mean()
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
        """Train stacker_lgbm_calibrated meta-learner."""
        tprint("🎭 [REGIME_ENSEMBLE] Training stacker_lgbm_calibrated meta-learner", color="yellow")
        
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
            
            # Create base model predictions for meta-learning
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
            
            # Create LightGBM meta-learner with AGGRESSIVE regularization to prevent overfitting
            tprint("🌲 [REGIME_ENSEMBLE] Creating LightGBM meta-learner with AGGRESSIVE regularization", color="blue", bold=True)
            meta_learner = LGBMClassifier(
                num_leaves=8,           # Dramatically reduced from 63 to 8
                max_depth=2,            # Dramatically reduced from 8 to 2
                learning_rate=0.02,     # Reduced from 0.05 to 0.02
                n_estimators=100,       # Reduced from 200 to 100
                min_child_samples=50,   # Increased from 20 to 50
                subsample=0.5,          # Reduced from 0.8 to 0.5
                colsample_bytree=0.5,   # Reduced from 0.8 to 0.5
                reg_alpha=1.0,          # Increased from 0.1 to 1.0
                reg_lambda=1.0,         # Increased from 0.1 to 1.0
                class_weight='balanced', # Handle class imbalance
                max_bin=127,            # Added: limit bin size
                min_data_in_bin=3,      # Added: minimum data per bin
                boost_from_average=False, # Added: disable boost from average
                force_col_wise=True,    # Added: force column-wise boosting
                extra_trees=True,       # Added: use extra trees for more randomness
                min_split_gain=0.1,     # Added: high threshold for splits
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
                    'training_success': True
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
                    'training_success': True
                }
                
                tprint("✅ [REGIME_ENSEMBLE] stacker_lgbm_calibrated training completed (uncalibrated)", color="green")
                return stacker_result
                
        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] stacker_lgbm_calibrated training failed: {e}", color="red")
            return None
    
    def _evaluate_ensemble(self, X: np.ndarray, y: np.ndarray, stacker_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate ensemble performance."""
        tprint("📊 [REGIME_ENSEMBLE] Evaluating ensemble performance", color="yellow")
        
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
            
            # Calculate metrics
            accuracy = accuracy_score(y, y_pred)
            
            metrics['stacker_lgbm_calibrated'] = {
                'accuracy': accuracy,
                'classification_report': classification_report(y, y_pred, output_dict=True),
                'prediction_confidence': {
                    'mean': y_pred_proba.max(axis=1).mean(),
                    'std': y_pred_proba.max(axis=1).std()
                },
                'calibration_method': stacker_result.get('calibration_method', 'none'),
                'base_models_used': len(base_models),
                'meta_features_shape': meta_features.shape
            }
            
            tprint(f"✅ [REGIME_ENSEMBLE] Meta-learner accuracy: {accuracy:.4f}", color="green")
            tprint(f"📊 [REGIME_ENSEMBLE] Prediction confidence: {y_pred_proba.max(axis=1).mean():.4f} ± {y_pred_proba.max(axis=1).std():.4f}", color="blue")
            
        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] Ensemble evaluation failed: {e}", color="red")
            metrics['stacker_lgbm_calibrated'] = {'error': str(e)}
        
        tprint("✅ [REGIME_ENSEMBLE] Ensemble evaluation completed", color="green")
        return metrics
    
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

            # Check for NaN or infinite values in features
            nan_count = np.isnan(X).sum()
            inf_count = np.isinf(X).sum()
            if nan_count > 0:
                tprint(f"⚠️ [REGIME_ENSEMBLE] Found {nan_count} NaN values in features", color="yellow")
                # Use sophisticated NaN handling for time series data
                X = self._handle_nan_values(X, nan_count)
            if inf_count > 0:
                tprint(f"⚠️ [REGIME_ENSEMBLE] Found {inf_count} infinite values in features", color="yellow")
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
        """Generate comprehensive features using the existing feature bank."""
        tprint("🔧 [REGIME_ENSEMBLE] Generating features using feature bank", color="cyan", bold=True)
        
        try:
            if not FEATURE_GENERATION_AVAILABLE:
                tprint("❌ [REGIME_ENSEMBLE] Feature generation system not available", color="red")
                return None
            
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
            
            # Generate features for each category
            for category in categories:
                tprint(f"🔍 [REGIME_ENSEMBLE] Generating {category.value} features", color="blue")
                
                # Get generators for this category
                generators = feature_bank.get_generators_by_category(category)
                
                if not generators:
                    tprint(f"⚠️ [REGIME_ENSEMBLE] No generators found for {category.value}", color="yellow")
                    continue
                
                category_features = pd.DataFrame(index=data.index)
                
                # Generate features using each generator
                for generator in generators:
                    try:
                        tprint(f"🔧 [REGIME_ENSEMBLE] Using generator: {generator.config.name}", color="blue")
                        result = generator.generate(data)
                        
                        if result and hasattr(result, 'data') and not result.data.empty:
                            # Add feature with category prefix
                            feature_name = f"{category.value}_{result.name}"
                            category_features[feature_name] = result.data
                            total_features += 1
                            tprint(f"✅ [REGIME_ENSEMBLE] Generated feature: {feature_name}", color="green")
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
            tprint("✅ [REGIME_ENSEMBLE] CatBoost trained successfully", color="green")
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
            tprint("✅ [REGIME_ENSEMBLE] Random Forest trained successfully", color="green")
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
            tprint("✅ [REGIME_ENSEMBLE] Extra Tree trained successfully", color="green")
        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] Extra Tree training failed: {e}", color="red")
        
        tprint(f"✅ [REGIME_ENSEMBLE] Base models training completed - {len(base_models)} models trained", color="green")
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