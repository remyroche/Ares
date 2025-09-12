"""
Tactician Ensemble Training Step

This step handles training of Tactician ensemble models on all regime data including:
- Ensemble model training with HPO
- Model saving and persistence
- Metrics analysis and validation
- Feature integration from Analyst models
- HMM cluster/regime state integration
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime
import joblib
from pathlib import Path

from src.utils.logger import system_logger
from src.utils.ml_common.ensembles import StackingEnsembleManager, StackingEnsembleConfig
from src.utils.ml_common.optimization.overfitting_prevention import OverfittingPrevention, OverfittingPreventionConfig
from src.utils.ml_common.post_training.model_persistence import ModelPersistence
from src.utils.ml_common.post_training.model_evaluation import ModelEvaluator, EvaluationConfig

logger = system_logger.getChild('TacticianEnsembleTraining')

@dataclass
class TacticianEnsembleTrainingConfig:
    """Configuration for Tactician ensemble training."""
    
    # Basic configuration
    model_name: str = "tactician_ensemble"
    timeframe: str = "1m"
    
    # Ensemble configuration
    base_models: List[str] = field(default_factory=lambda: [
        "NODE", "CatBoostRegressor", "LGBMRegressor", "Ridge"
    ])
    meta_model: str = "Ridge"
    
    # HPO configuration
    enable_hpo: bool = True
    hpo_n_trials: int = 50
    hpo_timeout_seconds: int = 1800
    hpo_cv_folds: int = 5
    
    # Meta model HPO search space
    meta_model_hpo_space: Dict[str, Any] = field(default_factory=lambda: {
        'alpha': {'type': 'float', 'low': 0.1, 'high': 10.0, 'log': True},
        'solver': {'type': 'categorical', 'choices': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}
    })
    
    # Analyst integration
    analyst_model_path: str = "./models/analyst_ensemble"
    analyst_output_names: List[str] = field(default_factory=lambda: [
        "signal_strength", "confidence", "risk_score", "regime_label"
    ])
    analyst_threshold: float = 0.6
    
    # Training configuration
    validation_split: float = 0.2
    test_split: float = 0.1
    enable_cross_validation: bool = True
    cv_folds: int = 5
    
    # Model saving
    save_models: bool = True
    model_save_path: str = "./models/tactician_ensemble"
    save_format: str = "joblib"
    
    # Evaluation configuration
    enable_evaluation: bool = True
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        "mse", "mae", "r2", "mape", "smape"
    ])
    
    # Overfitting prevention
    enable_overfitting_prevention: bool = True
    overfitting_threshold: float = 0.1

class TacticianEnsembleTrainingStep:
    """
    Tactician Ensemble Training Step with all-regime ensemble training, HPO, saving, and metrics.
    
    This step trains ensemble models on all regime data, including:
    1. Ensemble model training with HPO on all regime data
    2. Model saving and persistence
    3. Comprehensive metrics analysis
    4. Feature integration from Analyst models
    5. HMM cluster/regime state integration
    """
    
    def __init__(self, config: Optional[TacticianEnsembleTrainingConfig] = None):
        """Initialize Tactician ensemble training step."""
        self.config = config or TacticianEnsembleTrainingConfig()
        self.logger = logger.getChild('TacticianEnsembleTrainingStep')
        
        # Initialize components
        self.overfitting_prevention = OverfittingPrevention(
            OverfittingPreventionConfig() if self.config.enable_overfitting_prevention else None
        )
        self.model_persistence = ModelPersistence()
        self.model_evaluator = ModelEvaluator(
            EvaluationConfig(metrics=self.config.evaluation_metrics)
        )
        
        # Training results
        self.training_results = {}
        self.tactician_ensemble = None
        self.training_metadata = {}
        
        self.logger.info("✅ Tactician Ensemble Training Step initialized")
    
    def execute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        analyst_ensembles: Optional[Dict[int, Any]] = None,
        base_models: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute Tactician ensemble training step.
        
        Args:
            X: Input features
            y: Target values (tactician outputs)
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states
            analyst_ensembles: Pre-trained Analyst ensemble models
            base_models: Pre-trained base models from tactician_models_training step
            
        Returns:
            Dictionary containing training results and metadata
        """
        self.logger.info("🚀 Starting Tactician ensemble training step")
        start_time = time.time()
        
        # Step 1: Prepare combined features with Analyst outputs and HMM states
        self.logger.info("🔄 Step 1: Preparing combined features...")
        combined_features = self._prepare_combined_features(
            X, regime_labels, hmm_states, analyst_ensembles, feature_names
        )
        
        # Step 2: Train ensemble on all regime data
        self.logger.info("🔄 Step 2: Training ensemble on all regime data...")
        ensemble_result = self._train_ensemble(combined_features, y, feature_names, base_models)
        
        # Step 3: Save ensemble model
        if self.config.save_models:
            self.logger.info("🔄 Step 3: Saving trained ensemble model...")
            self._save_ensemble_model(ensemble_result)
        
        # Step 4: Evaluate ensemble performance
        if self.config.enable_evaluation:
            self.logger.info("🔄 Step 4: Evaluating ensemble performance...")
            evaluation_results = self._evaluate_ensemble(ensemble_result, combined_features, y, regime_labels)
        else:
            evaluation_results = {}
        
        # Create final results
        total_time = time.time() - start_time
        results = {
            'tactician_ensemble': ensemble_result,
            'training_metadata': ensemble_result['metadata'],
            'evaluation_results': evaluation_results,
            'training_time': total_time,
            'config': self.config,
            'combined_features_shape': combined_features.shape
        }
        
        self.training_results = results
        
        self.logger.info(f"✅ Tactician ensemble training completed in {total_time:.2f}s")
        self.logger.info(f"📊 Combined features: {combined_features.shape[1]} total features")
        
        return results
    
    def _prepare_combined_features(
        self,
        X: np.ndarray,
        regime_labels: np.ndarray,
        hmm_states: Optional[np.ndarray],
        analyst_ensembles: Optional[Dict[int, Any]],
        feature_names: Optional[List[str]]
    ) -> np.ndarray:
        """Prepare combined features with Analyst outputs and HMM states."""
        
        features = [X]
        new_feature_names = feature_names.copy() if feature_names else []
        
        # Add HMM states as features if available
        if hmm_states is not None:
            self.logger.info("🔄 Adding HMM states as features...")
            hmm_features = pd.get_dummies(hmm_states, prefix='hmm_state').values
            features.append(hmm_features)
            new_feature_names.extend([f"hmm_state_{i}" for i in range(hmm_features.shape[1])])
        
        # Add Analyst outputs as features if available
        if analyst_ensembles is not None:
            self.logger.info("🔄 Adding Analyst outputs as features...")
            analyst_outputs = self._get_analyst_outputs(X, regime_labels, analyst_ensembles)
            features.append(analyst_outputs)
            new_feature_names.extend(self.config.analyst_output_names)
        
        # Add regime features
        self.logger.info("🔄 Adding regime features...")
        regime_features = self._create_regime_features(regime_labels, X)
        features.append(regime_features)
        new_feature_names.extend([f"regime_feature_{i}" for i in range(regime_features.shape[1])])
        
        # Combine all features
        combined_features = np.hstack(features)
        
        self.logger.info(f"📊 Combined features: {combined_features.shape[1]} total features")
        self.logger.info(f"📊 - Original features: {X.shape[1]}")
        if hmm_states is not None:
            self.logger.info(f"📊 - HMM features: {hmm_features.shape[1]}")
        if analyst_ensembles is not None:
            self.logger.info(f"📊 - Analyst features: {analyst_outputs.shape[1]}")
        self.logger.info(f"📊 - Regime features: {regime_features.shape[1]}")
        
        return combined_features
    
    def _get_analyst_outputs(
        self,
        X: np.ndarray,
        regime_labels: np.ndarray,
        analyst_ensembles: Dict[int, Any]
    ) -> np.ndarray:
        """Get Analyst outputs for all samples."""
        
        analyst_outputs = np.zeros((len(X), len(self.config.analyst_output_names)))
        
        for regime in np.unique(regime_labels):
            regime_mask = regime_labels == regime
            regime_X = X[regime_mask]
            
            if regime in analyst_ensembles:
                try:
                    ensemble_manager = analyst_ensembles[regime]['ensemble_manager']
                    regime_outputs = ensemble_manager.predict(regime_X)
                    
                    # Apply threshold filtering
                    confidence_scores = regime_outputs[:, 1]  # Assuming confidence is second column
                    valid_mask = confidence_scores >= self.config.analyst_threshold
                    
                    # Only use outputs above threshold
                    analyst_outputs[regime_mask] = regime_outputs
                    analyst_outputs[regime_mask][~valid_mask] = 0  # Zero out low confidence outputs
                    
                    self.logger.debug(f"📊 Regime {regime}: {np.sum(valid_mask)}/{len(regime_X)} samples above threshold")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to get Analyst outputs for regime {regime}: {e}")
                    continue
            else:
                self.logger.warning(f"⚠️ No Analyst ensemble found for regime {regime}")
        
        return analyst_outputs
    
    def _create_regime_features(self, regime_labels: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Create regime-aware features."""
        
        regime_features = []
        
        # One-hot encoding of regime
        regime_onehot = pd.get_dummies(regime_labels, prefix='regime')
        regime_features.append(regime_onehot.values)
        
        # Regime transition features
        regime_transitions = np.diff(regime_labels, prepend=regime_labels[0])
        regime_features.append(regime_transitions.reshape(-1, 1))
        
        # Regime duration features
        regime_durations = self._calculate_regime_durations(regime_labels)
        regime_features.append(regime_durations.reshape(-1, 1))
        
        # Regime momentum features
        regime_momentum = self._calculate_regime_momentum(regime_labels, X)
        regime_features.append(regime_momentum)
        
        return np.hstack(regime_features)
    
    def _calculate_regime_durations(self, regime_labels: np.ndarray) -> np.ndarray:
        """Calculate duration of current regime for each sample."""
        
        durations = np.zeros(len(regime_labels))
        current_regime = regime_labels[0]
        current_duration = 1
        
        for i in range(1, len(regime_labels)):
            if regime_labels[i] == current_regime:
                current_duration += 1
            else:
                # Regime changed, update durations for previous regime
                durations[i-current_duration:i] = current_duration
                current_regime = regime_labels[i]
                current_duration = 1
        
        # Update durations for the last regime
        durations[-current_duration:] = current_duration
        
        return durations
    
    def _calculate_regime_momentum(self, regime_labels: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Calculate momentum features within each regime."""
        
        momentum_features = []
        
        for regime in np.unique(regime_labels):
            regime_mask = regime_labels == regime
            regime_X = X[regime_mask]
            
            if len(regime_X) > 1:
                # Calculate momentum as difference between consecutive samples
                regime_momentum = np.diff(regime_X, axis=0)
                # Pad with zeros for the first sample
                regime_momentum = np.vstack([np.zeros((1, regime_momentum.shape[1])), regime_momentum])
            else:
                regime_momentum = np.zeros((1, X.shape[1]))
            
            momentum_features.append(regime_momentum)
        
        # Combine momentum features
        combined_momentum = np.vstack(momentum_features)
        return combined_momentum
    
    def _train_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]],
        base_models: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Train Tactician ensemble on combined features."""
        
        # Get base models
        if base_models is not None:
            ensemble_base_models = base_models
        else:
            self.logger.info("🔄 Creating base models for ensemble...")
            ensemble_base_models = self._create_base_models(X, y)
        
        # Train ensemble
        if self.config.enable_hpo:
            ensemble_result = self._optimize_ensemble(ensemble_base_models, X, y, feature_names)
        else:
            ensemble_result = self._train_single_ensemble(ensemble_base_models, X, y, feature_names)
        
        return ensemble_result
    
    def _create_base_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Create base models for ensemble if not provided."""
        
        from src.utils.ml_common.models import EnhancedModelFactory, ModelType, ModelConfig
        
        model_factory = EnhancedModelFactory()
        base_models = {}
        
        for model_type in self.config.base_models:
            model_config = ModelConfig(
                model_type=ModelType[model_type.upper()],
                model_name=f"tactician_{model_type.lower()}",
                model_params=self._get_model_params(model_type)
            )
            
            model = model_factory.create_model(model_config)
            model.fit(X, y)
            base_models[model_type] = model
        
        return base_models
    
    def _optimize_ensemble(
        self,
        base_models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Optimize ensemble using HPO."""
        
        self.logger.info("🔄 Optimizing Tactician ensemble...")
        
        # Create ensemble configuration
        ensemble_config = StackingEnsembleConfig(
            base_models=base_models,
            meta_model_type=self.config.meta_model,
            meta_model_params=self._get_meta_model_params(),
            enable_cross_validation=self.config.enable_cross_validation,
            cv_folds=self.config.cv_folds,
            validation_split=self.config.validation_split
        )
        
        # Create ensemble manager
        ensemble_manager = StackingEnsembleManager(ensemble_config)
        
        # Apply overfitting prevention
        if self.config.enable_overfitting_prevention:
            for model_name, model in base_models.items():
                base_models[model_name] = self.overfitting_prevention.apply_regularization(model, model_name)
        
        # Train ensemble
        start_time = time.time()
        ensemble_manager.train(X, y)
        training_time = time.time() - start_time
        
        return {
            'ensemble_manager': ensemble_manager,
            'base_models': base_models,
            'training_time': training_time,
            'config': ensemble_config,
            'metadata': {
                'samples': len(X),
                'features': X.shape[1],
                'base_models': list(base_models.keys()),
                'meta_model': self.config.meta_model
            }
        }
    
    def _train_single_ensemble(
        self,
        base_models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Train single ensemble without HPO."""
        
        self.logger.info("🔄 Training Tactician ensemble (no HPO)...")
        
        # Create ensemble configuration
        ensemble_config = StackingEnsembleConfig(
            base_models=base_models,
            meta_model_type=self.config.meta_model,
            meta_model_params=self._get_meta_model_params(),
            enable_cross_validation=self.config.enable_cross_validation,
            cv_folds=self.config.cv_folds,
            validation_split=self.config.validation_split
        )
        
        # Create ensemble manager
        ensemble_manager = StackingEnsembleManager(ensemble_config)
        
        # Apply overfitting prevention
        if self.config.enable_overfitting_prevention:
            for model_name, model in base_models.items():
                base_models[model_name] = self.overfitting_prevention.apply_regularization(model, model_name)
        
        # Train ensemble
        start_time = time.time()
        ensemble_manager.train(X, y)
        training_time = time.time() - start_time
        
        return {
            'ensemble_manager': ensemble_manager,
            'base_models': base_models,
            'training_time': training_time,
            'config': ensemble_config,
            'metadata': {
                'samples': len(X),
                'features': X.shape[1],
                'base_models': list(base_models.keys()),
                'meta_model': self.config.meta_model
            }
        }
    
    def _get_model_params(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters for model type."""
        
        default_params = {
            'NODE': {
                'n_d': 64,
                'n_a': 64,
                'n_steps': 5,
                'gamma': 1.5,
                'lambda_sparse': 1e-3,
                'dropout': 0.1,
                'l2_regularization': 0.01
            },
            'CATBOOSTREGRESSOR': {
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3.0,
                'subsample': 0.8,
                'colsample_bylevel': 0.8
            },
            'LGBMREGRESSOR': {
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'max_depth': 6,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'RIDGE': {
                'alpha': 1.0,
                'solver': 'auto',
                'random_state': 42
            }
        }
        
        return default_params.get(model_type.upper(), {})
    
    def _get_meta_model_params(self) -> Dict[str, Any]:
        """Get default parameters for meta model."""
        
        return {
            'alpha': 1.0,
            'solver': 'auto',
            'random_state': 42
        }
    
    def _save_ensemble_model(self, ensemble_result: Dict[str, Any]) -> None:
        """Save trained ensemble model."""
        
        save_path = Path(self.config.model_save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save ensemble manager
        ensemble_file = save_path / f"tactician_ensemble.{self.config.save_format}"
        
        if self.config.save_format == "joblib":
            joblib.dump(ensemble_result['ensemble_manager'], ensemble_file)
        elif self.config.save_format == "pickle":
            import pickle
            with open(ensemble_file, 'wb') as f:
                pickle.dump(ensemble_result['ensemble_manager'], f)
        
        self.logger.info(f"💾 Tactician ensemble saved to {ensemble_file}")
    
    def _evaluate_ensemble(
        self,
        ensemble_result: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate ensemble performance."""
        
        ensemble_manager = ensemble_result['ensemble_manager']
        
        # Make predictions
        y_pred = ensemble_manager.predict(X)
        
        # Calculate metrics
        metrics = {}
        for metric in self.config.evaluation_metrics:
            if metric == 'mse':
                from sklearn.metrics import mean_squared_error
                metrics[metric] = mean_squared_error(y, y_pred)
            elif metric == 'mae':
                from sklearn.metrics import mean_absolute_error
                metrics[metric] = mean_absolute_error(y, y_pred)
            elif metric == 'r2':
                from sklearn.metrics import r2_score
                metrics[metric] = r2_score(y, y_pred)
            elif metric == 'mape':
                metrics[metric] = np.mean(np.abs((y - y_pred) / y)) * 100
            elif metric == 'smape':
                metrics[metric] = np.mean(2 * np.abs(y - y_pred) / (np.abs(y) + np.abs(y_pred))) * 100
        
        return metrics

# Convenience functions
def create_tactician_ensemble_training_step(config: Optional[TacticianEnsembleTrainingConfig] = None) -> TacticianEnsembleTrainingStep:
    """Create Tactician ensemble training step."""
    return TacticianEnsembleTrainingStep(config)

def execute_tactician_ensemble_training(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[TacticianEnsembleTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None,
    analyst_ensembles: Optional[Dict[int, Any]] = None,
    base_models: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute Tactician ensemble training step."""
    step = create_tactician_ensemble_training_step(config)
    return step.execute(X, y, regime_labels, feature_names, hmm_states, analyst_ensembles, base_models)