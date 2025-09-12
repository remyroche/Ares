"""
Analyst Ensemble Training Step

This step handles per-regime training of Analyst ensemble models including:
- Ensemble model training with HPO
- Model saving and persistence
- Metrics analysis and validation
- Regime-specific feature integration
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
from src.utils.ml_common.optimization import HierarchicalHPO, HierarchicalHPOConfig, HPOPhaseConfig
from src.utils.ml_common.optimization.overfitting_prevention import OverfittingPrevention, OverfittingPreventionConfig
from src.utils.ml_common.post_training.model_persistence import ModelPersistence
from src.utils.ml_common.post_training.model_evaluation import ModelEvaluator, EvaluationConfig

logger = system_logger.getChild('AnalystEnsembleTraining')

@dataclass
class AnalystEnsembleTrainingConfig:
    """Configuration for Analyst ensemble training."""
    
    # Basic configuration
    model_name: str = "analyst_ensemble"
    timeframe: str = "5m"
    
    # Ensemble configuration
    base_models: List[str] = field(default_factory=lambda: [
        "GRU", "CatBoostRegressor", "LGBMRegressor", "RandomForestRegressor"
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
    
    # Regime configuration
    min_samples_per_regime: int = 1000
    enable_regime_merging: bool = True
    regime_merge_threshold: int = 500
    
    # Data augmentation
    enable_data_augmentation: bool = True
    augmentation_method: str = "smote"
    augmentation_ratio: float = 1.0
    
    # Training configuration
    validation_split: float = 0.2
    test_split: float = 0.1
    enable_cross_validation: bool = True
    cv_folds: int = 5
    
    # Model saving
    save_models: bool = True
    model_save_path: str = "./models/analyst_ensemble"
    save_format: str = "joblib"
    
    # Evaluation configuration
    enable_evaluation: bool = True
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        "mse", "mae", "r2", "mape", "smape"
    ])
    
    # Overfitting prevention
    enable_overfitting_prevention: bool = True
    overfitting_threshold: float = 0.1

class AnalystEnsembleTrainingStep:
    """
    Analyst Ensemble Training Step with per-regime ensemble training, HPO, saving, and metrics.
    
    This step trains ensemble models for each regime, including:
    1. Per-regime ensemble training with HPO
    2. Model saving and persistence
    3. Comprehensive metrics analysis
    4. Regime-specific feature integration
    5. HMM cluster/regime state integration
    """
    
    def __init__(self, config: Optional[AnalystEnsembleTrainingConfig] = None):
        """Initialize Analyst ensemble training step."""
        self.config = config or AnalystEnsembleTrainingConfig()
        self.logger = logger.getChild('AnalystEnsembleTrainingStep')
        
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
        self.regime_ensembles = {}
        self.regime_metadata = {}
        
        self.logger.info("✅ Analyst Ensemble Training Step initialized")
    
    def execute(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        hmm_states: Optional[np.ndarray] = None,
        base_models: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Execute Analyst ensemble training step.
        
        Args:
            X: Input features
            y: Target values (analyst outputs)
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            hmm_states: HMM cluster/regime states
            base_models: Pre-trained base models from analyst_models_training step
            
        Returns:
            Dictionary containing training results and metadata
        """
        self.logger.info("🚀 Starting Analyst ensemble training step")
        start_time = time.time()
        
        # Step 1: Analyze regimes and prepare data
        self.logger.info("🔄 Step 1: Analyzing regimes and preparing data...")
        regime_analysis = self._analyze_regimes(regime_labels)
        regime_data = self._prepare_regime_data(X, y, regime_labels, regime_analysis, hmm_states)
        
        # Step 2: Train ensemble models for each regime
        self.logger.info("🔄 Step 2: Training ensemble models for each regime...")
        regime_results = self._train_regime_ensembles(regime_data, feature_names, base_models)
        
        # Step 3: Save ensemble models
        if self.config.save_models:
            self.logger.info("🔄 Step 3: Saving trained ensemble models...")
            self._save_ensemble_models(regime_results)
        
        # Step 4: Evaluate ensemble performance
        if self.config.enable_evaluation:
            self.logger.info("🔄 Step 4: Evaluating ensemble performance...")
            evaluation_results = self._evaluate_ensembles(regime_results, X, y, regime_labels)
        else:
            evaluation_results = {}
        
        # Create final results
        total_time = time.time() - start_time
        results = {
            'regime_ensembles': regime_results['ensembles'],
            'regime_metadata': regime_results['metadata'],
            'evaluation_results': evaluation_results,
            'training_time': total_time,
            'config': self.config,
            'regime_analysis': regime_analysis
        }
        
        self.training_results = results
        
        self.logger.info(f"✅ Analyst ensemble training completed in {total_time:.2f}s")
        self.logger.info(f"📊 Regimes trained: {len(regime_results['ensembles'])}")
        
        return results
    
    def _analyze_regimes(self, regime_labels: np.ndarray) -> Dict[str, Any]:
        """Analyze regime distribution and characteristics."""
        
        unique_regimes, regime_counts = np.unique(regime_labels, return_counts=True)
        
        regime_analysis = {
            'unique_regimes': unique_regimes,
            'regime_counts': regime_counts,
            'total_samples': len(regime_labels),
            'regime_proportions': regime_counts / len(regime_labels)
        }
        
        # Identify regimes with sufficient data
        sufficient_regimes = unique_regimes[regime_counts >= self.config.min_samples_per_regime]
        insufficient_regimes = unique_regimes[regime_counts < self.config.min_samples_per_regime]
        
        regime_analysis['sufficient_regimes'] = sufficient_regimes
        regime_analysis['insufficient_regimes'] = insufficient_regimes
        
        # Identify regimes to merge
        if self.config.enable_regime_merging:
            merge_candidates = unique_regimes[regime_counts < self.config.regime_merge_threshold]
            regime_analysis['merge_candidates'] = merge_candidates
        else:
            regime_analysis['merge_candidates'] = []
        
        self.logger.info(f"📊 Regime analysis:")
        self.logger.info(f"📊 - Total regimes: {len(unique_regimes)}")
        self.logger.info(f"📊 - Sufficient data: {len(sufficient_regimes)}")
        self.logger.info(f"📊 - Insufficient data: {len(insufficient_regimes)}")
        self.logger.info(f"📊 - Merge candidates: {len(regime_analysis['merge_candidates'])}")
        
        return regime_analysis
    
    def _prepare_regime_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        regime_analysis: Dict[str, Any],
        hmm_states: Optional[np.ndarray]
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """Prepare data for each regime with HMM state integration."""
        
        regime_data = {}
        
        for regime in regime_analysis['unique_regimes']:
            regime_mask = regime_labels == regime
            regime_X = X[regime_mask]
            regime_y = y[regime_mask]
            
            # Add HMM states as features if available
            if hmm_states is not None:
                regime_hmm_states = hmm_states[regime_mask]
                # One-hot encode HMM states
                hmm_features = pd.get_dummies(regime_hmm_states, prefix='hmm_state').values
                regime_X = np.hstack([regime_X, hmm_features])
            
            # Check if regime has sufficient data
            if len(regime_X) >= self.config.min_samples_per_regime:
                # Sufficient data - use as is
                regime_data[regime] = {
                    'X': regime_X,
                    'y': regime_y,
                    'samples': len(regime_X),
                    'augmented': False,
                    'hmm_states': regime_hmm_states if hmm_states is not None else None
                }
            elif self.config.enable_data_augmentation and len(regime_X) > 100:
                # Insufficient data but enough for augmentation
                augmented_X, augmented_y = self._augment_regime_data(regime_X, regime_y)
                regime_data[regime] = {
                    'X': augmented_X,
                    'y': augmented_y,
                    'samples': len(augmented_X),
                    'augmented': True,
                    'hmm_states': regime_hmm_states if hmm_states is not None else None
                }
            else:
                # Too little data - mark for global model fallback
                regime_data[regime] = {
                    'X': regime_X,
                    'y': regime_y,
                    'samples': len(regime_X),
                    'augmented': False,
                    'use_global': True,
                    'hmm_states': regime_hmm_states if hmm_states is not None else None
                }
            
            self.logger.debug(f"📊 Regime {regime}: {regime_data[regime]['samples']} samples, "
                            f"augmented: {regime_data[regime]['augmented']}, "
                            f"use_global: {regime_data[regime].get('use_global', False)}")
        
        return regime_data
    
    def _augment_regime_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Augment data for regimes with insufficient samples."""
        
        if self.config.augmentation_method == "smote":
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=42, sampling_strategy=self.config.augmentation_ratio)
                X_aug, y_aug = smote.fit_resample(X, y)
                return X_aug, y_aug
            except ImportError:
                self.logger.warning("⚠️ SMOTE not available, skipping augmentation")
                return X, y
        elif self.config.augmentation_method == "adasyn":
            try:
                from imblearn.over_sampling import ADASYN
                adasyn = ADASYN(random_state=42, sampling_strategy=self.config.augmentation_ratio)
                X_aug, y_aug = adasyn.fit_resample(X, y)
                return X_aug, y_aug
            except ImportError:
                self.logger.warning("⚠️ ADASYN not available, skipping augmentation")
                return X, y
        else:
            self.logger.warning(f"⚠️ Unknown augmentation method: {self.config.augmentation_method}")
            return X, y
    
    def _train_regime_ensembles(
        self,
        regime_data: Dict[int, Dict[str, np.ndarray]],
        feature_names: Optional[List[str]],
        base_models: Optional[Dict[int, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Train ensemble models for each regime with HPO."""
        
        regime_ensembles = {}
        regime_metadata = {}
        
        for regime, data in regime_data.items():
            if data.get('use_global', False):
                self.logger.info(f"⏭️ Skipping regime {regime} (insufficient data, will use global model)")
                continue
            
            self.logger.info(f"🔄 Training ensemble for regime {regime} ({data['samples']} samples)...")
            
            # Get base models for this regime
            if base_models and regime in base_models:
                regime_base_models = base_models[regime]
            else:
                self.logger.warning(f"⚠️ No base models found for regime {regime}, creating new ones")
                regime_base_models = self._create_base_models_for_regime(regime, data['X'], data['y'])
            
            # Train ensemble for this regime
            if self.config.enable_hpo:
                ensemble_result = self._optimize_ensemble(
                    regime_base_models, data['X'], data['y'], regime, feature_names
                )
            else:
                ensemble_result = self._train_single_ensemble(
                    regime_base_models, data['X'], data['y'], regime, feature_names
                )
            
            regime_ensembles[regime] = ensemble_result
            
            # Store regime metadata
            regime_metadata[regime] = {
                'samples': data['samples'],
                'augmented': data['augmented'],
                'hmm_states': data.get('hmm_states'),
                'base_models': list(regime_base_models.keys()),
                'training_time': time.time()
            }
            
            self.logger.info(f"✅ Regime {regime} ensemble trained")
        
        return {
            'ensembles': regime_ensembles,
            'metadata': regime_metadata
        }
    
    def _create_base_models_for_regime(
        self,
        regime: int,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Create base models for regime if not provided."""
        
        from src.utils.ml_common.models import EnhancedModelFactory, ModelType, ModelConfig
        
        model_factory = EnhancedModelFactory()
        base_models = {}
        
        for model_type in self.config.base_models:
            model_config = ModelConfig(
                model_type=ModelType[model_type.upper()],
                model_name=f"analyst_{model_type.lower()}_regime_{regime}",
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
        regime: int,
        feature_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Optimize ensemble using HPO."""
        
        self.logger.debug(f"🔄 Optimizing ensemble for regime {regime}...")
        
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
            'regime': regime,
            'training_time': training_time,
            'config': ensemble_config
        }
    
    def _train_single_ensemble(
        self,
        base_models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        regime: int,
        feature_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Train single ensemble without HPO."""
        
        self.logger.debug(f"🔄 Training ensemble for regime {regime} (no HPO)...")
        
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
            'regime': regime,
            'training_time': training_time,
            'config': ensemble_config
        }
    
    def _get_model_params(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters for model type."""
        
        default_params = {
            'GRU': {
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.2,
                'recurrent_dropout': 0.1,
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
            'RANDOMFORESTREGRESSOR': {
                'n_estimators': 500,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True
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
    
    def _save_ensemble_models(self, regime_results: Dict[str, Any]) -> None:
        """Save trained ensemble models."""
        
        save_path = Path(self.config.model_save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for regime, ensemble_result in regime_results['ensembles'].items():
            regime_path = save_path / f"regime_{regime}"
            regime_path.mkdir(exist_ok=True)
            
            # Save ensemble manager
            ensemble_file = regime_path / f"ensemble.{self.config.save_format}"
            
            if self.config.save_format == "joblib":
                joblib.dump(ensemble_result['ensemble_manager'], ensemble_file)
            elif self.config.save_format == "pickle":
                import pickle
                with open(ensemble_file, 'wb') as f:
                    pickle.dump(ensemble_result['ensemble_manager'], f)
            
            self.logger.debug(f"💾 Saved ensemble for regime {regime} to {ensemble_file}")
        
        self.logger.info(f"💾 All ensemble models saved to {save_path}")
    
    def _evaluate_ensembles(
        self,
        regime_results: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate ensemble performance."""
        
        evaluation_results = {}
        
        for regime, ensemble_result in regime_results['ensembles'].items():
            regime_mask = regime_labels == regime
            regime_X = X[regime_mask]
            regime_y = y[regime_mask]
            
            ensemble_manager = ensemble_result['ensemble_manager']
            
            # Make predictions
            y_pred = ensemble_manager.predict(regime_X)
            
            # Calculate metrics
            metrics = {}
            for metric in self.config.evaluation_metrics:
                if metric == 'mse':
                    from sklearn.metrics import mean_squared_error
                    metrics[metric] = mean_squared_error(regime_y, y_pred)
                elif metric == 'mae':
                    from sklearn.metrics import mean_absolute_error
                    metrics[metric] = mean_absolute_error(regime_y, y_pred)
                elif metric == 'r2':
                    from sklearn.metrics import r2_score
                    metrics[metric] = r2_score(regime_y, y_pred)
                elif metric == 'mape':
                    metrics[metric] = np.mean(np.abs((regime_y - y_pred) / regime_y)) * 100
                elif metric == 'smape':
                    metrics[metric] = np.mean(2 * np.abs(regime_y - y_pred) / (np.abs(regime_y) + np.abs(y_pred))) * 100
            
            evaluation_results[regime] = metrics
        
        return evaluation_results

# Convenience functions
def create_analyst_ensemble_training_step(config: Optional[AnalystEnsembleTrainingConfig] = None) -> AnalystEnsembleTrainingStep:
    """Create Analyst ensemble training step."""
    return AnalystEnsembleTrainingStep(config)

def execute_analyst_ensemble_training(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[AnalystEnsembleTrainingConfig] = None,
    feature_names: Optional[List[str]] = None,
    hmm_states: Optional[np.ndarray] = None,
    base_models: Optional[Dict[int, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Execute Analyst ensemble training step."""
    step = create_analyst_ensemble_training_step(config)
    return step.execute(X, y, regime_labels, feature_names, hmm_states, base_models)