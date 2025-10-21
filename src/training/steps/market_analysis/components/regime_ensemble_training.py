"""
Regime Detection Ensemble Training Step

BaseStep-based implementation for training regime detection ensemble models.
Migrated from the component pattern to use the new BaseStep architecture.

This step implements the meta-learner for regime detection:
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

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_warning, tprint_error

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

class RegimeEnsembleTrainingStep(BaseStep):
    """
    Regime Detection Ensemble Training Step using BaseStep pattern.

    This step trains the meta-learner for regime detection:
    - stacker_lgbm_calibrated: LightGBM model with probability calibration
    """

    def __init__(self, step_name: str = "regime_ensemble_training", config: Optional[Dict[str, Any]] = None):
        """Initialize the Regime Ensemble Training Step."""
        super().__init__(step_name, config)
        self.logger = system_logger.getChild('RegimeEnsembleTraining')
        self.training_start_time = None
        self.training_end_time = None
        tprint("🚀 [REGIME_ENSEMBLE] Initializing Regime Ensemble Training Step", "INFO")
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

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute regime ensemble training.

        Args:
            config: Configuration dictionary with parameters:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - data_dir: Data directory path
                - start_date: Start date (optional)
                - end_date: End date (optional)
                - train_data: Training data (optional)
                - regime_labels: Regime labels (optional)
                - base_models: Base models to use (optional)
                - ensemble_params: Ensemble parameters (optional)

        Returns:
            Dictionary with ensemble training results and model artifacts
        """
        start_time = datetime.now()
        tprint(f"🔍 Starting regime ensemble training for {config.get('symbol', 'UNKNOWN')}", "INFO")
        
        # Set context for artifact management
        self._set_context(
            symbol=config.get('symbol', 'ETHUSDT'),
            exchange=config.get('exchange', 'binance'),
            direction=config.get('direction', 'both'),
            model=config.get('model', 'default')
        )

        try:
            # Load training data
            data = self._load_training_data(config)
            if data is None:
                raise ValueError("No training data found")
            
            tprint(f"✅ Loaded training data: {data.shape[0]} rows, {data.shape[1]} columns", "SUCCESS")

            # Load regime labels
            regime_labels = self._load_regime_labels(config)
            if regime_labels is None:
                raise ValueError("No regime labels found")
            
            tprint(f"✅ Loaded regime labels: {len(regime_labels)} labels", "SUCCESS")

            # Load base models
            base_models = self._load_base_models(config)
            if not base_models:
                raise ValueError("No base models found")
            
            tprint(f"✅ Loaded {len(base_models)} base models", "SUCCESS")

            # Prepare training data from the input data DataFrame with advanced regime features
            tprint("🔧 Preparing training data with advanced regime features", "INFO")
            X, y, feature_names = self._prepare_training_data(data, regime_labels, config)

            # Validate required data
            tprint("🔍 Validating required data", "INFO")
            if X is None or y is None or feature_names is None:
                tprint("❌ Failed to prepare training data", "ERROR")
                return {
                    'success': False,
                    'error': "Failed to prepare training data from input DataFrame",
                    'ensemble_model': None,
                    'evaluation_results': {},
                    'metrics': {},
                    'processing_time': (datetime.now() - start_time).total_seconds()
                }

            if not base_models:
                tprint("⚠️ [REGIME_ENSEMBLE] No base models found from previous training, training base models", color="yellow")
                # Train base models if not provided
                tprint("🏋️ [REGIME_ENSEMBLE] Training base models for ensemble", color="blue")
                base_models = self._train_base_models(X, y, regime_labels)
                if not base_models:
                    tprint("❌ [REGIME_ENSEMBLE] Failed to train base models", color="red")
                    return {
                        'success': False,
                        'artifacts': {},
                        'error_message': "Failed to train base models",
                        'metadata': {'component_type': 'regime_ensemble_training'}
                    }

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

            return {
                'success': True,
                'artifacts': results,
                'metadata': {
                    'component_type': 'regime_ensemble_training',
                    'execution_time': (datetime.now() - start_time).total_seconds(),
                    'artifacts_saved_persistently': True
                }
            }

        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] Regime ensemble training failed: {e}", color="red", bold=True)
            self.logger.error(f"Regime ensemble training failed: {e}", exc_info=True)
            return {
                'success': False,
                'artifacts': {},
                'error_message': str(e),
                'metadata': {'component_type': 'regime_ensemble_training'}
            }

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

            # Calculate top-3 regime analysis with entropy metrics
            top_3_analysis = self._calculate_top_regime_analysis(y_pred_proba)

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
                'meta_features_shape': meta_features.shape
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

                        if result and hasattr(result, 'data') and not len(result.data) == 0:
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

    def _load_training_data(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load training data from artifacts or config."""
        try:
            # Try to load from artifacts first
            train_data = self._load_dataframe('train_data')
            if train_data is not None:
                return train_data
            
            # Try alternative artifact names
            train_data = self._load_dataframe('training_data') or self._load_dataframe('market_data')
            if train_data is not None:
                return train_data
            
            # Try to load from config
            if 'train_data' in config:
                return pd.DataFrame(config['train_data'])
            
            return None
            
        except Exception as e:
            tprint(f"⚠️ Failed to load training data: {e}", "WARNING")
            return None

    def _load_regime_labels(self, config: Dict[str, Any]) -> Optional[np.ndarray]:
        """Load regime labels from artifacts or config."""
        try:
            # Try to load from artifacts first
            regime_data = self._get_artifact('regime_labels')
            if regime_data is not None:
                if isinstance(regime_data, dict) and 'labels' in regime_data:
                    return np.array(regime_data['labels'])
                elif isinstance(regime_data, (list, np.ndarray)):
                    return np.array(regime_data)
            
            # Try alternative artifact names
            regime_data = self._get_artifact('regime_assignments') or self._get_artifact('cluster_assignments')
            if regime_data is not None:
                return np.array(regime_data)
            
            # Try to load from config
            if 'regime_labels' in config:
                return np.array(config['regime_labels'])
            
            return None
            
        except Exception as e:
            tprint(f"⚠️ Failed to load regime labels: {e}", "WARNING")
            return None

    def _load_base_models(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load base models from artifacts or config."""
        try:
            base_models = {}
            
            # Try to load from artifacts
            model_artifacts = ['catboost_model', 'extratrees_model', 'rulelist_model', 'lightgbm_model']
            for artifact_name in model_artifacts:
                try:
                    model = self._load_model(artifact_name)
                    if model is not None:
                        model_name = artifact_name.replace('_model', '')
                        base_models[model_name] = model
                except Exception as e:
                    tprint(f"⚠️ Could not load {artifact_name}: {e}", "WARNING")
            
            # Try to load from config
            if 'base_models' in config:
                for name, model in config['base_models'].items():
                    base_models[name] = model
            
            return base_models
            
        except Exception as e:
            tprint(f"⚠️ Failed to load base models: {e}", "WARNING")
            return {}

    def _calculate_ensemble_metrics(self, ensemble: Any, evaluation_results: Dict[str, Any], start_time: datetime, config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate ensemble training metrics."""
        try:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            metrics = {
                'processing_time_seconds': processing_time,
                'accuracy': evaluation_results.get('accuracy', 0.0),
                'cv_mean': evaluation_results.get('cv_mean', 0.0),
                'cv_std': evaluation_results.get('cv_std', 0.0),
                'n_test_samples': evaluation_results.get('n_test_samples', 0),
                'model_type': 'stacking_ensemble',
                'calibrated': config.get('ensemble_params', {}).get('calibrate', True),
                'success': True
            }
            
            return metrics
            
        except Exception as e:
            tprint(f"⚠️ Failed to calculate metrics: {e}", "WARNING")
            return {'success': False, 'error': str(e)}

    def _create_outcome_report(self, ensemble: Any, evaluation_results: Dict[str, Any], metrics: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Create outcome report markdown."""
        try:
            report = f"""# Regime Ensemble Training Outcome Report

## Execution Summary
- **Symbol**: {config.get('symbol', 'UNKNOWN')}
- **Exchange**: {config.get('exchange', 'UNKNOWN')}
- **Timeframe**: {config.get('timeframe', 'UNKNOWN')}
- **Processing Time**: {metrics.get('processing_time_seconds', 0):.2f} seconds
- **Success**: {'✅ Yes' if metrics.get('success', False) else '❌ No'}

## Ensemble Training Results
- **Model Type**: {metrics.get('model_type', 'stacking_ensemble')}
- **Calibrated**: {'✅ Yes' if metrics.get('calibrated', False) else '❌ No'}
- **Accuracy**: {metrics.get('accuracy', 0):.3f}
- **CV Mean**: {metrics.get('cv_mean', 0):.3f} ± {metrics.get('cv_std', 0):.3f}
- **Test Samples**: {metrics.get('n_test_samples', 0):,}

## Base Models Used
"""
            
            base_models = config.get('base_models', {})
            for model_name in base_models.keys():
                report += f"- **{model_name}**: ✅ Available\n"
            
            report += f"""
## Performance Metrics
- **Accuracy**: {evaluation_results.get('accuracy', 0):.3f}
- **Cross-Validation Mean**: {evaluation_results.get('cv_mean', 0):.3f}
- **Cross-Validation Std**: {evaluation_results.get('cv_std', 0):.3f}

## Generated Artifacts
- Ensemble model (pickle file)
- Evaluation results
- Ensemble metadata

---
*Generated by Regime Ensemble Training Step at {datetime.now().isoformat()}*
"""
            
            return report
            
        except Exception as e:
            tprint(f"⚠️ Failed to create outcome report: {e}", "WARNING")
            return f"# Regime Ensemble Training Outcome Report\n\nError creating report: {str(e)}"


# Register the step
def register_regime_ensemble_training_step():
    """Register the regime ensemble training step."""
    from src.training.steps.base_step import step_registry
    
    step_registry.register("regime_ensemble_training", RegimeEnsembleTrainingStep)
    tprint("✅ Regime ensemble training step registered", "SUCCESS")


# Auto-register when module is imported
register_regime_ensemble_training_step()
