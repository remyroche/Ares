"""
HMM Ensemble Training

This module handles the training of ensemble models (meta-models) for HMM regime prediction.
It includes meta-model training, hyperparameter optimization, model saving, and metrics calculation.

Meta-Model:
- XGBoost as meta-learner for combining base models

Note: This module does NOT handle regime tagging - that is done in regime_data_splitting.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from sklearn.ensemble import StackingClassifier, StackingRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Import existing infrastructure - FAST FAIL if not available
try:
    from src.utils.ml_common.optimization.hpo_utils import HyperparameterOptimization
    from src.utils.ml_common.pareto import ParetoOptimizer
    INFRASTRUCTURE_AVAILABLE = True
except ImportError as e:
    raise ImportError(f"Required infrastructure not available: {e}. Cannot proceed without existing tools.")

class HMMEnsembleTraining:
    """HMM ensemble training for regime prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize HMM ensemble training."""
        self.config = config
        self.logger = config.get('logger', None)
        self.ensemble_models = {}
        self.performance_metrics = {}
        
        # Configuration
        self.hpo_trials = config.get('hpo_trials', 100)
        self.cv_folds = config.get('cv_folds', 5)
        
        # Initialize existing infrastructure - FAST FAIL
        self._initialize_infrastructure()
        
    def _initialize_infrastructure(self):
        """Initialize existing infrastructure components - FAST FAIL if not available."""
        # Initialize multi-objective optimization
        hpo_config = {
            'enable_multi_objective': True,
            'objectives': ['accuracy', 'f1_score', 'regime_stability'],
            'objective_weights': [0.4, 0.3, 0.3],
            'n_trials': self.hpo_trials,
            'enable_pruning': True
        }
        self.hpo_optimizer = HyperparameterOptimization(hpo_config)
        
        # Initialize Pareto optimizer
        self.pareto_optimizer = ParetoOptimizer()
    
    def create_ensemble_models(self, base_models: Dict[str, Any], is_classification: bool) -> Dict[str, Any]:
        """Create ensemble models with XGBoost as meta-learner."""
        ensembles = {}
        
        if is_classification:
            # Stacking ensemble with XGBoost as meta-learner
            meta_learner = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, n_jobs=-1
            )
            ensembles['stacking_ensemble'] = StackingClassifier(
                estimators=list(base_models.items()),
                final_estimator=meta_learner,  # XGBoost as meta-learner
                cv=self.cv_folds, n_jobs=-1
            )
            
        else:
            # Stacking ensemble with XGBoost as meta-learner
            meta_learner = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, n_jobs=-1
            )
            ensembles['stacking_ensemble'] = StackingRegressor(
                estimators=list(base_models.items()),
                final_estimator=meta_learner,  # XGBoost as meta-learner
                cv=self.cv_folds, n_jobs=-1
            )
        
        return ensembles
    
    def optimize_meta_learner_hyperparameters(self, X: pd.DataFrame, y: np.ndarray, 
                                            is_classification: bool) -> Dict[str, Any]:
        """Optimize hyperparameters for XGBoost meta-learner."""
        
        def create_meta_learner(params):
            if is_classification:
                return xgb.XGBClassifier(**params)
            else:
                return xgb.XGBRegressor(**params)
        
        # Use existing multi-objective optimization - NO FALLBACK
        optimization_result = self.hpo_optimizer.multi_objective_optimization(
            model_factory=create_meta_learner,
            X=X, y=y,
            objectives=['accuracy', 'f1_score', 'regime_stability'],
            objective_weights=[0.4, 0.3, 0.3],
            n_trials=self.hpo_trials
        )
        
        print(f"✅ Meta-learner hyperparameter optimization completed")
        return optimization_result
    
    def evaluate_ensemble_comprehensive(self, ensemble: Any, X_test: pd.DataFrame, y_test: np.ndarray, 
                                      is_classification: bool) -> Dict[str, Any]:
        """Comprehensive ensemble evaluation."""
        
        y_pred = ensemble.predict(X_test)
        
        if is_classification:
            y_pred_proba = ensemble.predict_proba(X_test) if hasattr(ensemble, 'predict_proba') else None
            
            from sklearn.metrics import (
                accuracy_score, f1_score, precision_score, recall_score,
                classification_report, confusion_matrix, log_loss
            )
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            if y_pred_proba is not None and len(np.unique(y_test)) == 2:
                from sklearn.metrics import roc_auc_score
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            
            if y_pred_proba is not None:
                metrics['log_loss'] = log_loss(y_test, y_pred_proba)
            
        else:
            from sklearn.metrics import mean_squared_error, r2_score
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': np.mean(np.abs(y_test - y_pred)),
                'r2_score': r2_score(y_test, y_pred),
                'explained_variance': 1 - np.var(y_test - y_pred) / np.var(y_test)
            }
        
        return metrics
    
    def train_ensemble_models(self, base_models: Dict[str, Any], X: pd.DataFrame, y: np.ndarray, 
                            is_classification: bool = True) -> Dict[str, Any]:
        """Train ensemble models with XGBoost meta-learner."""
        
        results = {
            'ensemble_models': {},
            'performance': {},
            'best_ensemble': None,
            'best_score': 0.0,
            'meta_learner_optimization': {}
        }
        
        # Create ensemble models
        ensemble_models = self.create_ensemble_models(base_models, is_classification)
        
        # Split data for training and testing
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=y if is_classification else None
        )
        
        # Train ensemble models
        for name, ensemble in ensemble_models.items():
            try:
                print(f"🔄 Training ensemble model: {name}")
                
                # Train ensemble
                ensemble.fit(X_train, y_train)
                
                # Evaluate ensemble
                metrics = self.evaluate_ensemble_comprehensive(ensemble, X_test, y_test, is_classification)
                
                # Store results
                results['ensemble_models'][name] = ensemble
                results['performance'][name] = metrics
                
                # Track best ensemble
                score = metrics.get('accuracy' if is_classification else 'r2_score', 0.0)
                if score > results['best_score']:
                    results['best_score'] = score
                    results['best_ensemble'] = name
                
                print(f"✅ {name} completed: {score:.4f}")
                
            except Exception as e:
                print(f"❌ Error training ensemble {name}: {e}")
                continue
        
        # Optimize meta-learner hyperparameters
        try:
            print("🔄 Optimizing meta-learner hyperparameters...")
            meta_optimization = self.optimize_meta_learner_hyperparameters(X, y, is_classification)
            results['meta_learner_optimization'] = meta_optimization
            print("✅ Meta-learner optimization completed")
        except Exception as e:
            print(f"⚠️ Meta-learner optimization failed: {e}")
            results['meta_learner_optimization'] = {'error': str(e)}
        
        return results
    
    def save_ensemble_models(self, ensemble_models: Dict[str, Any], symbol: str, exchange: str, 
                           timeframe: str, data_dir: str) -> List[str]:
        """Save trained ensemble models."""
        import pickle
        from pathlib import Path
        
        models_dir = Path(data_dir) / 'models' / 'hmm' / 'ensemble_models'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_paths = []
        
        for name, model in ensemble_models.items():
            model_path = models_dir / f'hmm_ensemble_{name}_{symbol}_{exchange}_{timeframe}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            model_paths.append(str(model_path))
        
        print(f"✅ Saved {len(model_paths)} ensemble models")
        return model_paths
    
    def get_ensemble_predictions(self, ensemble: Any, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from ensemble model."""
        predictions = {
            'regime_predictions': ensemble.predict(X),
            'regime_probabilities': ensemble.predict_proba(X) if hasattr(ensemble, 'predict_proba') else None
        }
        
        return predictions