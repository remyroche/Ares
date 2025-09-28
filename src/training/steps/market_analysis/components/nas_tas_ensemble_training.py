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
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
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
        self.voting_ensemble = None
        self.stacking_ensemble = None
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
            
            # Train voting ensemble
            tprint("🗳️ [NAS_TAS_ENSEMBLE] Training voting ensemble", color="yellow")
            voting_ensemble = self._train_voting_ensemble(X_processed, y_processed, base_models)
            
            # Train stacking ensemble
            tprint("📚 [NAS_TAS_ENSEMBLE] Training stacking ensemble", color="yellow")
            stacking_ensemble = self._train_stacking_ensemble(X_processed, y_processed, base_models)
            
            # Evaluate ensembles
            tprint("📊 [NAS_TAS_ENSEMBLE] Evaluating ensemble performance", color="yellow")
            ensemble_metrics = self._evaluate_ensembles(X_processed, y_processed, voting_ensemble, stacking_ensemble)
            
            # Create comprehensive results
            tprint("📦 [NAS_TAS_ENSEMBLE] Creating comprehensive results", color="yellow")
            results = {
                'nas_tas_ensemble_training_result': {
                    'voting_ensemble': voting_ensemble,
                    'stacking_ensemble': stacking_ensemble,
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
        
        # LightGBM Classifier
        tprint("🌲 [NAS_TAS_ENSEMBLE] Training LightGBM classifier", color="blue")
        try:
            lgb_model = LGBMClassifier(
                n_estimators=self.ensemble_config['n_estimators'],
                max_depth=self.ensemble_config['max_depth'],
                learning_rate=self.ensemble_config['learning_rate'],
                random_state=self.ensemble_config['random_state'],
                n_jobs=self.ensemble_config['n_jobs'],
                verbose=-1
            )
            lgb_model.fit(X, y)
            base_models['lightgbm'] = lgb_model
            tprint("✅ [NAS_TAS_ENSEMBLE] LightGBM trained successfully", color="green")
        except Exception as e:
            tprint(f"❌ [NAS_TAS_ENSEMBLE] LightGBM training failed: {e}", color="red")
        
        # XGBoost Classifier
        tprint("🚀 [NAS_TAS_ENSEMBLE] Training XGBoost classifier", color="blue")
        try:
            xgb_model = XGBClassifier(
                n_estimators=self.ensemble_config['n_estimators'],
                max_depth=self.ensemble_config['max_depth'],
                learning_rate=self.ensemble_config['learning_rate'],
                random_state=self.ensemble_config['random_state'],
                n_jobs=self.ensemble_config['n_jobs'],
                verbosity=0
            )
            xgb_model.fit(X, y)
            base_models['xgboost'] = xgb_model
            tprint("✅ [NAS_TAS_ENSEMBLE] XGBoost trained successfully", color="green")
        except Exception as e:
            tprint(f"❌ [NAS_TAS_ENSEMBLE] XGBoost training failed: {e}", color="red")
        
        # Random Forest Classifier
        tprint("🌳 [NAS_TAS_ENSEMBLE] Training Random Forest classifier", color="blue")
        try:
            rf_model = RandomForestClassifier(
                n_estimators=self.ensemble_config['n_estimators'],
                max_depth=self.ensemble_config['max_depth'],
                random_state=self.ensemble_config['random_state'],
                n_jobs=self.ensemble_config['n_jobs']
            )
            rf_model.fit(X, y)
            base_models['random_forest'] = rf_model
            tprint("✅ [NAS_TAS_ENSEMBLE] Random Forest trained successfully", color="green")
        except Exception as e:
            tprint(f"❌ [NAS_TAS_ENSEMBLE] Random Forest training failed: {e}", color="red")
        
        # Logistic Regression
        tprint("📈 [NAS_TAS_ENSEMBLE] Training Logistic Regression", color="blue")
        try:
            lr_model = LogisticRegression(
                random_state=self.ensemble_config['random_state'],
                max_iter=1000,
                n_jobs=self.ensemble_config['n_jobs']
            )
            lr_model.fit(X, y)
            base_models['logistic_regression'] = lr_model
            tprint("✅ [NAS_TAS_ENSEMBLE] Logistic Regression trained successfully", color="green")
        except Exception as e:
            tprint(f"❌ [NAS_TAS_ENSEMBLE] Logistic Regression training failed: {e}", color="red")
        
        tprint(f"✅ [NAS_TAS_ENSEMBLE] Base models training completed - {len(base_models)} models trained", color="green")
        return base_models
    
    def _train_voting_ensemble(self, X: np.ndarray, y: np.ndarray, base_models: Dict[str, Any]) -> Any:
        """Train voting ensemble."""
        tprint("🗳️ [NAS_TAS_ENSEMBLE] Training voting ensemble", color="yellow")
        
        if not base_models:
            tprint("❌ [NAS_TAS_ENSEMBLE] No base models available for voting ensemble", color="red")
            return None
        
        try:
            # Create voting classifier
            estimators = [(name, model) for name, model in base_models.items()]
            voting_ensemble = VotingClassifier(estimators=estimators, voting='soft')
            
            # Train the ensemble
            tprint("🏋️ [NAS_TAS_ENSEMBLE] Training voting classifier", color="blue")
            voting_ensemble.fit(X, y)
            
            tprint("✅ [NAS_TAS_ENSEMBLE] Voting ensemble trained successfully", color="green")
            return voting_ensemble
            
        except Exception as e:
            tprint(f"❌ [NAS_TAS_ENSEMBLE] Voting ensemble training failed: {e}", color="red")
            return None
    
    def _train_stacking_ensemble(self, X: np.ndarray, y: np.ndarray, base_models: Dict[str, Any]) -> Any:
        """Train stacking ensemble."""
        tprint("📚 [NAS_TAS_ENSEMBLE] Training stacking ensemble", color="yellow")
        
        if not base_models:
            tprint("❌ [NAS_TAS_ENSEMBLE] No base models available for stacking ensemble", color="red")
            return None
        
        try:
            # Create stacking classifier with meta-learner
            estimators = [(name, model) for name, model in base_models.items()]
            meta_learner = LogisticRegression(random_state=self.ensemble_config['random_state'])
            
            stacking_ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=5,
                n_jobs=self.ensemble_config['n_jobs']
            )
            
            # Train the ensemble
            tprint("🏋️ [NAS_TAS_ENSEMBLE] Training stacking classifier", color="blue")
            stacking_ensemble.fit(X, y)
            
            tprint("✅ [NAS_TAS_ENSEMBLE] Stacking ensemble trained successfully", color="green")
            return stacking_ensemble
            
        except Exception as e:
            tprint(f"❌ [NAS_TAS_ENSEMBLE] Stacking ensemble training failed: {e}", color="red")
            return None
    
    def _evaluate_ensembles(self, X: np.ndarray, y: np.ndarray, voting_ensemble: Any, stacking_ensemble: Any) -> Dict[str, Any]:
        """Evaluate ensemble performance."""
        tprint("📊 [NAS_TAS_ENSEMBLE] Evaluating ensemble performance", color="yellow")
        
        metrics = {}
        
        # Evaluate voting ensemble
        if voting_ensemble is not None:
            tprint("🗳️ [NAS_TAS_ENSEMBLE] Evaluating voting ensemble", color="blue")
            try:
                y_pred_voting = voting_ensemble.predict(X)
                voting_accuracy = accuracy_score(y, y_pred_voting)
                metrics['voting_ensemble'] = {
                    'accuracy': voting_accuracy,
                    'classification_report': classification_report(y, y_pred_voting, output_dict=True)
                }
                tprint(f"✅ [NAS_TAS_ENSEMBLE] Voting ensemble accuracy: {voting_accuracy:.4f}", color="green")
            except Exception as e:
                tprint(f"❌ [NAS_TAS_ENSEMBLE] Voting ensemble evaluation failed: {e}", color="red")
                metrics['voting_ensemble'] = {'error': str(e)}
        
        # Evaluate stacking ensemble
        if stacking_ensemble is not None:
            tprint("📚 [NAS_TAS_ENSEMBLE] Evaluating stacking ensemble", color="blue")
            try:
                y_pred_stacking = stacking_ensemble.predict(X)
                stacking_accuracy = accuracy_score(y, y_pred_stacking)
                metrics['stacking_ensemble'] = {
                    'accuracy': stacking_accuracy,
                    'classification_report': classification_report(y, y_pred_stacking, output_dict=True)
                }
                tprint(f"✅ [NAS_TAS_ENSEMBLE] Stacking ensemble accuracy: {stacking_accuracy:.4f}", color="green")
            except Exception as e:
                tprint(f"❌ [NAS_TAS_ENSEMBLE] Stacking ensemble evaluation failed: {e}", color="red")
                metrics['stacking_ensemble'] = {'error': str(e)}
        
        tprint("✅ [NAS_TAS_ENSEMBLE] Ensemble evaluation completed", color="green")
        return metrics
