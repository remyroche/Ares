"""
NAS-TAS Ensemble Training Component

This component implements ensemble training for NAS-TAS (Neural Architecture Search - Tree-based Architecture Search) based regime detection models.
It creates meta-models that combine multiple base models trained on NAS-TAS regime labels.
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
import warnings
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path

from src.utils.tprint import tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_progress, tprint_performance, tprint_timer
from src.utils.logger import system_logger
from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult

# Suppress LightGBM warnings about no further splits
warnings.filterwarnings('ignore', message='.*No further splits with positive gain.*')

# Import ensemble training classes
try:
    from src.analyst.predictive_ensembles.ensemble_orchestrator import RegimePredictiveEnsembles
    from src.analyst.predictive_ensembles.regime_ensembles.volatile_regime_ensemble import VolatileRegimeEnsemble
    ENSEMBLE_AVAILABLE = True
    tprint("✅ [NAS_TAS_ENSEMBLE] Ensemble training classes imported successfully", color="green")
except ImportError as e:
    ENSEMBLE_AVAILABLE = False
    tprint(f"❌ [NAS_TAS_ENSEMBLE] Failed to import ensemble training classes: {e}", color="red")

# Import ML libraries
try:
    from sklearn.ensemble import VotingClassifier, StackingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.calibration import CalibratedClassifierCV
    from lightgbm import LGBMClassifier
    ML_LIBRARIES_AVAILABLE = True
    tprint("✅ [NAS_TAS_ENSEMBLE] ML libraries imported successfully", color="green")
except ImportError as e:
    ML_LIBRARIES_AVAILABLE = False
    tprint(f"❌ [NAS_TAS_ENSEMBLE] Failed to import ML libraries: {e}", color="red")


class NASTASEnsembleTrainingComponent(BaseMarketAnalysisComponent):
    """
    NAS-TAS Ensemble Training Component.
    
    This component trains ensemble models using NAS-TAS regime labels for meta-learning.
    It combines multiple base models into voting and stacking ensembles.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the NAS-TAS Ensemble Training Component."""
        tprint("🚀 [NAS_TAS_ENSEMBLE] Initializing NAS-TAS Ensemble Training Component", color="cyan", bold=True)
        super().__init__(config)
        
        self.logger = system_logger.getChild('NASTASEnsembleTrainingComponent')
        tprint("✅ [NAS_TAS_ENSEMBLE] Logger initialized", color="green")
        
        # Initialize ensemble training parameters
        self.ensemble_config = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1
        }
        tprint("⚙️ [NAS_TAS_ENSEMBLE] Ensemble configuration set", color="yellow")
        
        # Initialize ensemble models
        self.stacker_lgbm_calibrated = None
        self.base_models = {}
        self.ensemble_metrics = {}
        tprint("📊 [NAS_TAS_ENSEMBLE] Ensemble models initialized", color="blue")
        
        tprint("✅ [NAS_TAS_ENSEMBLE] NAS-TAS Ensemble Training Component initialized successfully", color="green", bold=True)
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        tprint("📋 [NAS_TAS_ENSEMBLE] Getting required artifacts", color="cyan")
        required_artifacts = ['nas_tas_ensemble_training_result']
        tprint(f"✅ [NAS_TAS_ENSEMBLE] Required artifacts: {required_artifacts}", color="green")
        return required_artifacts
    
    async def execute(self, data: pd.DataFrame, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute NAS ensemble training.
        
        Args:
            data: Market data DataFrame
            pipeline_state: Pipeline state containing features, targets, and regime labels
            
        Returns:
            ComponentResult with training results
        """
        tprint("🚀 [NAS_TAS_ENSEMBLE] Starting NAS ensemble training execution", color="cyan", bold=True)
        tprint("🎭 [NAS_TAS_ENSEMBLE] Using stacker_lgbm_calibrated meta-learner approach", color="cyan")
        start_time = datetime.now()
        
        try:
            # Extract required data from pipeline state
            tprint("📊 [NAS_TAS_ENSEMBLE] Extracting data from pipeline state", color="yellow")
            X = pipeline_state.get('features')
            y = pipeline_state.get('targets')
            # Extract regime labels from pipeline state artifacts
            artifacts = pipeline_state.get('artifacts', {})
            nas_tas_clustering_result = artifacts.get('nas_tas_clustering_result', {})
            regime_labels = nas_tas_clustering_result.get('regime_assignments')
            feature_names = pipeline_state.get('feature_names', [])
            nas_models = pipeline_state.get('nas_models', {})
            
            # Validate required data
            tprint("🔍 [NAS_TAS_ENSEMBLE] Validating required data", color="yellow")
            if X is None or y is None:
                tprint("❌ [NAS_TAS_ENSEMBLE] Missing features or targets", color="red")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message="Missing features or targets in pipeline state",
                    metadata={'component_type': 'nas_tas_ensemble_training'}
                )
            
            if regime_labels is None:
                tprint("⚠️ [NAS_TAS_ENSEMBLE] No regime labels found, using targets as regime labels", color="yellow")
                regime_labels = y
            
            tprint(f"📊 [NAS_TAS_ENSEMBLE] Data shapes - X: {X.shape}, y: {y.shape}, regime_labels: {len(regime_labels) if regime_labels is not None else 'None'}", color="blue")
            
            # Prepare data for ensemble training
            tprint("🔧 [NAS_TAS_ENSEMBLE] Preparing data for ensemble training", color="yellow")
            X_processed, y_processed, regime_labels_processed = self._prepare_data(X, y, regime_labels)
            tprint(f"✅ [NAS_TAS_ENSEMBLE] Data prepared - X: {X_processed.shape}, y: {y_processed.shape}", color="green")
            
            # Train base models if not provided
            tprint("🏋️ [NAS_TAS_ENSEMBLE] Training base models", color="yellow")
            if not nas_models:
                tprint("📝 [NAS_TAS_ENSEMBLE] No pre-trained NAS models found, training base models", color="blue")
                base_models = self._train_base_models(X_processed, y_processed, regime_labels_processed)
            else:
                tprint("📝 [NAS_TAS_ENSEMBLE] Using pre-trained NAS models", color="blue")
                base_models = nas_models
            
            # Train stacker_lgbm_calibrated meta-learner
            tprint("🎭 [NAS_TAS_ENSEMBLE] Training stacker_lgbm_calibrated meta-learner", color="yellow", bold=True)
            tprint("🎯 [NAS_TAS_ENSEMBLE] Meta-learner will use probability calibration for improved predictions", color="cyan")
            stacker_result = self._train_stacker_lgbm_calibrated(X_processed, y_processed, base_models)
            
            # Evaluate ensemble
            tprint("📊 [NAS_TAS_ENSEMBLE] Evaluating ensemble performance", color="yellow", bold=True)
            tprint("🔍 [NAS_TAS_ENSEMBLE] Computing accuracy, confidence metrics, and classification reports", color="cyan")
            ensemble_metrics = self._evaluate_ensemble(X_processed, y_processed, stacker_result)
            
            # Create comprehensive results
            tprint("📦 [NAS_TAS_ENSEMBLE] Creating comprehensive results", color="yellow")
            results = {
                'nas_tas_ensemble_training_result': {
                    'stacker_lgbm_calibrated': stacker_result,
                    'base_models': base_models,
                    'ensemble_metrics': ensemble_metrics,
                    'training_time': (datetime.now() - start_time).total_seconds(),
                    'success': True,
                    'metadata': {
                        'component_type': 'nas_tas_ensemble_training',
                        'data_shape': X_processed.shape,
                        'n_regimes': len(np.unique(regime_labels_processed)) if regime_labels_processed is not None else 0,
                        'feature_names': feature_names,
                        'timestamp': datetime.now().isoformat()
                    }
                }
            }
            
            tprint("✅ [NAS_TAS_ENSEMBLE] NAS ensemble training completed successfully", color="green", bold=True)
            tprint(f"⏱️ [NAS_TAS_ENSEMBLE] Total execution time: {(datetime.now() - start_time).total_seconds():.2f}s", color="blue")
            
            return ComponentResult(
                success=True,
                artifacts=results,
                metadata={'component_type': 'nas_tas_ensemble_training', 'execution_time': (datetime.now() - start_time).total_seconds()}
            )
            
        except Exception as e:
            tprint(f"❌ [NAS_TAS_ENSEMBLE] NAS ensemble training failed: {e}", color="red", bold=True)
            self.logger.error(f"NAS ensemble training failed: {e}", exc_info=True)
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e),
                metadata={'component_type': 'nas_tas_ensemble_training'}
            )
    
    def _prepare_data(self, X: np.ndarray, y: np.ndarray, regime_labels: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for ensemble training."""
        tprint("🔧 [NAS_TAS_ENSEMBLE] Preparing data for ensemble training", color="yellow")
        
        # Handle missing values
        tprint("🧹 [NAS_TAS_ENSEMBLE] Handling missing values", color="blue")
        if isinstance(X, pd.DataFrame):
            X = X.fillna(0).values
        elif isinstance(X, list):
            X = np.array(X)
        
        if isinstance(y, (pd.Series, list)):
            y = np.array(y)
        
        if regime_labels is not None and isinstance(regime_labels, (pd.Series, list)):
            regime_labels = np.array(regime_labels)
        
        # Ensure all arrays have the same length
        tprint("📏 [NAS_TAS_ENSEMBLE] Ensuring consistent array lengths", color="blue")
        min_length = min(len(X), len(y))
        if regime_labels is not None:
            min_length = min(min_length, len(regime_labels))
        
        X = X[:min_length]
        y = y[:min_length]
        if regime_labels is not None:
            regime_labels = regime_labels[:min_length]
        
        tprint(f"✅ [NAS_TAS_ENSEMBLE] Data prepared - X: {X.shape}, y: {y.shape}, regime_labels: {regime_labels.shape if regime_labels is not None else 'None'}", color="green")
        return X, y, regime_labels
    
    def _train_base_models(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray) -> Dict[str, Any]:
        """Train base models for ensemble."""
        tprint("🏋️ [NAS_TAS_ENSEMBLE] Training base models", color="yellow")
        
        base_models = {}
        
        # CatBoost Classifier
        tprint("🐱 [NAS_TAS_ENSEMBLE] Training CatBoost classifier", color="blue")
        try:
            import catboost as cb
            cb_model = cb.CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                l2_leaf_reg=3.0,
                random_seed=self.ensemble_config['random_state'],
                thread_count=self.ensemble_config['n_jobs'],
                verbose=False
            )
            cb_model.fit(X, y)
            base_models['catboost'] = cb_model
            tprint("✅ [NAS_TAS_ENSEMBLE] CatBoost trained successfully", color="green")
        except Exception as e:
            tprint(f"❌ [NAS_TAS_ENSEMBLE] CatBoost training failed: {e}", color="red")
        
        # Bayesian Rule Lists
        tprint("📋 [NAS_TAS_ENSEMBLE] Training Bayesian Rule Lists", color="blue")
        try:
            from imodels import BayesianRuleListClassifier
            brl_model = BayesianRuleListClassifier(
                max_rules=12,
                max_rule_length=3,
                n_chains=3,
                n_iter=10000
            )
            brl_model.fit(X, y)
            base_models['bayesian_rule_lists'] = brl_model
            tprint("✅ [NAS_TAS_ENSEMBLE] Bayesian Rule Lists trained successfully", color="green")
        except Exception as e:
            tprint(f"❌ [NAS_TAS_ENSEMBLE] Bayesian Rule Lists training failed: {e}", color="red")
        
        # ExtraTrees Classifier
        tprint("🌳 [NAS_TAS_ENSEMBLE] Training ExtraTrees classifier", color="blue")
        try:
            from sklearn.ensemble import ExtraTreesClassifier
            et_model = ExtraTreesClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=self.ensemble_config['random_state'],
                n_jobs=self.ensemble_config['n_jobs']
            )
            et_model.fit(X, y)
            base_models['extratrees'] = et_model
            tprint("✅ [NAS_TAS_ENSEMBLE] ExtraTrees trained successfully", color="green")
        except Exception as e:
            tprint(f"❌ [NAS_TAS_ENSEMBLE] ExtraTrees training failed: {e}", color="red")
        
        tprint(f"✅ [NAS_TAS_ENSEMBLE] Base models training completed - {len(base_models)} models trained", color="green")
        return base_models
    
    def _train_stacker_lgbm_calibrated(self, X: np.ndarray, y: np.ndarray, base_models: Dict[str, Any]) -> Dict[str, Any]:
        """Train stacker_lgbm_calibrated meta-learner."""
        tprint("🎭 [NAS_TAS_ENSEMBLE] Training stacker_lgbm_calibrated meta-learner", color="yellow", bold=True)
        tprint("🔧 [NAS_TAS_ENSEMBLE] Meta-learner configuration: LightGBM with constrained depth + probability calibration", color="cyan")
        
        if not base_models:
            tprint("❌ [NAS_TAS_ENSEMBLE] No base models available for meta-learner", color="red")
            return None
        
        try:
            # Filter out None models
            valid_models = {name: model for name, model in base_models.items() if model is not None}
            
            if not valid_models:
                tprint("❌ [NAS_TAS_ENSEMBLE] No valid base models available", color="red")
                return None
            
            tprint(f"📊 [NAS_TAS_ENSEMBLE] Using {len(valid_models)} valid base models: {list(valid_models.keys())}", color="blue")
            tprint(f"📊 [NAS_TAS_ENSEMBLE] Meta-learner input data: {X.shape[0]} samples, {X.shape[1]} features", color="blue")
            tprint(f"📊 [NAS_TAS_ENSEMBLE] Meta-learner target classes: {np.unique(y)}", color="blue")
            
            # Create base model predictions
            tprint("🔮 [NAS_TAS_ENSEMBLE] Generating base model predictions for meta-features", color="cyan")
            base_predictions = []
            for name, model in valid_models.items():
                try:
                    tprint(f"🔮 [NAS_TAS_ENSEMBLE] Generating predictions from {name}...", color="blue")
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(X)
                        base_predictions.append(pred_proba)
                        tprint(f"✅ [NAS_TAS_ENSEMBLE] {name} predictions generated: {pred_proba.shape}", color="green")
                        tprint(f"📊 [NAS_TAS_ENSEMBLE] {name} prediction confidence: {pred_proba.max(axis=1).mean():.4f}", color="blue")
                    else:
                        pred = model.predict(X)
                        # Convert to one-hot encoding for meta-learner
                        from sklearn.preprocessing import LabelBinarizer
                        lb = LabelBinarizer()
                        pred_proba = lb.fit_transform(pred)
                        base_predictions.append(pred_proba)
                        tprint(f"✅ [NAS_TAS_ENSEMBLE] {name} predictions generated (converted): {pred_proba.shape}", color="green")
                except Exception as e:
                    tprint(f"⚠️ [NAS_TAS_ENSEMBLE] {name} prediction failed: {e}", color="yellow")
            
            if not base_predictions:
                tprint("❌ [NAS_TAS_ENSEMBLE] No valid predictions generated", color="red")
                return None
            
            # Stack predictions
            tprint("🔗 [NAS_TAS_ENSEMBLE] Stacking base model predictions into meta-features", color="cyan")
            X_meta = np.hstack(base_predictions)
            tprint(f"📊 [NAS_TAS_ENSEMBLE] Meta-features shape: {X_meta.shape}", color="blue")
            tprint(f"📊 [NAS_TAS_ENSEMBLE] Meta-features from {len(base_predictions)} base models", color="blue")
            
            # Create LightGBM meta-learner with constrained depth
            tprint("🌲 [NAS_TAS_ENSEMBLE] Creating LightGBM meta-learner with constrained depth", color="cyan")
            meta_learner_params = {
                'num_leaves': 31,
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'random_state': self.ensemble_config['random_state'],
                'n_jobs': self.ensemble_config['n_jobs'],
                'verbosity': -1
            }
            tprint(f"🌲 [NAS_TAS_ENSEMBLE] Meta-learner parameters: {meta_learner_params}", color="blue")
            
            meta_learner = LGBMClassifier(**meta_learner_params)
            
            # Train meta-learner
            tprint("🏋️ [NAS_TAS_ENSEMBLE] Training LightGBM meta-learner", color="blue", bold=True)
            tprint(f"🏋️ [NAS_TAS_ENSEMBLE] Meta-learner training data: {X_meta.shape[0]} samples, {X_meta.shape[1]} meta-features", color="blue")
            meta_learner.fit(X_meta, y)
            tprint("✅ [NAS_TAS_ENSEMBLE] LightGBM meta-learner trained successfully", color="green")
            
            # Apply probability calibration
            tprint("🎯 [NAS_TAS_ENSEMBLE] Applying probability calibration (isotonic method)", color="blue", bold=True)
            tprint("🎯 [NAS_TAS_ENSEMBLE] Calibration will improve probability estimates for better predictions", color="cyan")
            calibrated_meta_learner = CalibratedClassifierCV(meta_learner, method='isotonic', cv=3)
            calibrated_meta_learner.fit(X_meta, y)
            tprint("✅ [NAS_TAS_ENSEMBLE] Probability calibration completed successfully", color="green")
            
            stacker_result = {
                'meta_learner': calibrated_meta_learner,
                'base_models': valid_models,
                'meta_features_shape': X_meta.shape,
                'training_success': True
            }
            
            tprint("✅ [NAS_TAS_ENSEMBLE] stacker_lgbm_calibrated training completed successfully", color="green", bold=True)
            tprint(f"🎭 [NAS_TAS_ENSEMBLE] Meta-learner ready with {len(valid_models)} base models and {X_meta.shape[1]} meta-features", color="cyan")
            return stacker_result
            
        except Exception as e:
            tprint(f"❌ [NAS_TAS_ENSEMBLE] stacker_lgbm_calibrated training failed: {e}", color="red")
            return None
    
    def _evaluate_ensemble(self, X: np.ndarray, y: np.ndarray, stacker_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate ensemble performance."""
        tprint("📊 [NAS_TAS_ENSEMBLE] Evaluating ensemble performance", color="yellow")
        
        metrics = {}
        
        # Evaluate stacker_lgbm_calibrated
        if stacker_result is not None and stacker_result.get('training_success', False):
            tprint("🎭 [NAS_TAS_ENSEMBLE] Evaluating stacker_lgbm_calibrated", color="blue", bold=True)
            tprint("🔍 [NAS_TAS_ENSEMBLE] Computing performance metrics and confidence analysis", color="cyan")
            try:
                meta_learner = stacker_result['meta_learner']
                base_models = stacker_result['base_models']
                
                tprint(f"🔍 [NAS_TAS_ENSEMBLE] Evaluation data: {X.shape[0]} samples, {X.shape[1]} features", color="blue")
                tprint(f"🔍 [NAS_TAS_ENSEMBLE] Using {len(base_models)} base models for meta-features", color="blue")
                
                # Generate meta-features
                tprint("🔮 [NAS_TAS_ENSEMBLE] Generating meta-features from base models", color="cyan")
                base_predictions = []
                for name, model in base_models.items():
                    tprint(f"🔮 [NAS_TAS_ENSEMBLE] Generating meta-features from {name}...", color="blue")
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(X)
                        base_predictions.append(pred_proba)
                        tprint(f"✅ [NAS_TAS_ENSEMBLE] {name} meta-features: {pred_proba.shape}", color="green")
                    else:
                        pred = model.predict(X)
                        from sklearn.preprocessing import LabelBinarizer
                        lb = LabelBinarizer()
                        pred_proba = lb.fit_transform(pred)
                        base_predictions.append(pred_proba)
                        tprint(f"✅ [NAS_TAS_ENSEMBLE] {name} meta-features (converted): {pred_proba.shape}", color="green")
                
                X_meta = np.hstack(base_predictions)
                tprint(f"📊 [NAS_TAS_ENSEMBLE] Final meta-features shape: {X_meta.shape}", color="blue")
                
                # Make predictions
                tprint("🎯 [NAS_TAS_ENSEMBLE] Making predictions with calibrated meta-learner", color="cyan")
                y_pred = meta_learner.predict(X_meta)
                y_pred_proba = meta_learner.predict_proba(X_meta)
                tprint(f"🎯 [NAS_TAS_ENSEMBLE] Predictions generated: {y_pred.shape}, Probabilities: {y_pred_proba.shape}", color="green")
                
                # Calculate metrics
                accuracy = accuracy_score(y, y_pred)
                tprint(f"📊 [NAS_TAS_ENSEMBLE] Accuracy calculated: {accuracy:.4f}", color="blue")
                
                metrics['stacker_lgbm_calibrated'] = {
                    'accuracy': accuracy,
                    'classification_report': classification_report(y, y_pred, output_dict=True),
                    'prediction_confidence': {
                        'mean': y_pred_proba.max(axis=1).mean(),
                        'std': y_pred_proba.max(axis=1).std()
                    }
                }
                tprint(f"✅ [NAS_TAS_ENSEMBLE] stacker_lgbm_calibrated accuracy: {accuracy:.4f}", color="green", bold=True)
                tprint(f"📊 [NAS_TAS_ENSEMBLE] Prediction confidence: {y_pred_proba.max(axis=1).mean():.4f} ± {y_pred_proba.max(axis=1).std():.4f}", color="blue")
                tprint(f"📊 [NAS_TAS_ENSEMBLE] Class distribution in predictions: {dict(zip(*np.unique(y_pred, return_counts=True)))}", color="blue")
                tprint("✅ [NAS_TAS_ENSEMBLE] Evaluation completed successfully", color="green")
                
            except Exception as e:
                tprint(f"❌ [NAS_TAS_ENSEMBLE] stacker_lgbm_calibrated evaluation failed: {e}", color="red")
                metrics['stacker_lgbm_calibrated'] = {'error': str(e)}
        else:
            tprint("❌ [NAS_TAS_ENSEMBLE] No valid stacker_lgbm_calibrated model to evaluate", color="red")
            metrics['stacker_lgbm_calibrated'] = {'error': 'No valid model'}
        
        tprint("✅ [NAS_TAS_ENSEMBLE] Ensemble evaluation completed", color="green")
        return metrics
