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
            X = pipeline_state.get('features')
            y = pipeline_state.get('targets')
            
            # Extract regime labels from pipeline state artifacts
            artifacts = pipeline_state.get('artifacts', {})
            nas_tas_clustering_result = artifacts.get('nas_tas_clustering_result', {})
            regime_labels = nas_tas_clustering_result.get('regime_assignments')
            
            # Get base models from previous training
            regime_models_result = artifacts.get('regime_models_training_result', {})
            base_models = regime_models_result.get('regime_models', {})
            
            feature_names = pipeline_state.get('feature_names', [])
            
            # Validate required data
            tprint("🔍 [REGIME_ENSEMBLE] Validating required data", color="yellow")
            if X is None or y is None:
                tprint("❌ [REGIME_ENSEMBLE] Missing features or targets", color="red")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message="Missing features or targets in pipeline state",
                    metadata={'component_type': 'regime_ensemble_training'}
                )
            
            if regime_labels is None:
                tprint("⚠️ [REGIME_ENSEMBLE] No regime labels found, using targets as regime labels", color="yellow")
                regime_labels = y
            
            if not base_models:
                tprint("❌ [REGIME_ENSEMBLE] No base models found from previous training", color="red")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message="No base models found from previous training",
                    metadata={'component_type': 'regime_ensemble_training'}
                )
            
            tprint(f"📊 [REGIME_ENSEMBLE] Data shapes - X: {X.shape}, y: {y.shape}, regime_labels: {len(regime_labels) if regime_labels is not None else 'None'}", color="blue")
            tprint(f"📊 [REGIME_ENSEMBLE] Base models available: {list(base_models.keys())}", color="blue")
            
            # Prepare data for ensemble training
            tprint("🔧 [REGIME_ENSEMBLE] Preparing data for ensemble training", color="yellow")
            X_processed, y_processed, regime_labels_processed = self._prepare_data(X, y, regime_labels)
            tprint(f"✅ [REGIME_ENSEMBLE] Data prepared - X: {X_processed.shape}, y: {y_processed.shape}", color="green")
            
            # Train stacker_lgbm_calibrated meta-learner
            tprint("🎭 [REGIME_ENSEMBLE] Training stacker_lgbm_calibrated meta-learner", color="yellow")
            stacker_result = self._train_stacker_lgbm_calibrated(X_processed, y_processed, base_models)
            
            # Evaluate ensemble
            tprint("📊 [REGIME_ENSEMBLE] Evaluating ensemble performance", color="yellow")
            ensemble_metrics = self._evaluate_ensemble(X_processed, y_processed, stacker_result)
            
            # Create comprehensive results
            tprint("📦 [REGIME_ENSEMBLE] Creating comprehensive results", color="yellow")
            results = {
                'regime_ensemble_training_result': {
                    'stacker_lgbm_calibrated': stacker_result,
                    'base_models': base_models,
                    'ensemble_metrics': ensemble_metrics,
                    'training_time': (datetime.now() - start_time).total_seconds(),
                    'success': True,
                    'metadata': {
                        'component_type': 'regime_ensemble_training',
                        'data_shape': X_processed.shape,
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
    
    def _train_stacker_lgbm_calibrated(self, X: np.ndarray, y: np.ndarray, base_models: Dict[str, Any]) -> Dict[str, Any]:
        """Train stacker_lgbm_calibrated meta-learner."""
        tprint("🎭 [REGIME_ENSEMBLE] Training stacker_lgbm_calibrated meta-learner", color="yellow")
        
        try:
            # Filter out None models
            valid_base_models = {name: model for name, model in base_models.items() if model is not None}
            
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
            
            # Create LightGBM meta-learner
            tprint("🌲 [REGIME_ENSEMBLE] Creating LightGBM meta-learner", color="blue")
            meta_learner = LGBMClassifier(
                num_leaves=31,
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
            
            # Train meta-learner
            tprint("🏋️ [REGIME_ENSEMBLE] Training meta-learner", color="blue")
            meta_learner.fit(meta_features, y)
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
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(X)
                        base_predictions.append(pred_proba)
                    else:
                        pred = model.predict(X)
                        unique_classes = np.unique(y)
                        pred_onehot = np.zeros((len(pred), len(unique_classes)))
                        for i, class_val in enumerate(unique_classes):
                            pred_onehot[pred == class_val, i] = 1
                        base_predictions.append(pred_onehot)
                except Exception as e:
                    tprint(f"⚠️ [REGIME_ENSEMBLE] Failed to get predictions from {name}: {e}", color="yellow")
                    continue
            
            if not base_predictions:
                tprint("❌ [REGIME_ENSEMBLE] No valid predictions for evaluation", color="red")
                return {'error': 'No valid predictions for evaluation'}
            
            # Combine predictions
            meta_features = np.column_stack(base_predictions)
            
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