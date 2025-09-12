"""
Regime-Aware Analyst Training Implementation

This module implements per-regime training for the Analyst model:
- Train separate Analyst models for each market regime
- Use regime-specific features and targets
- Implement fallback strategies for regimes with insufficient data
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime

from src.utils.logger import system_logger
from src.utils.ml_common.models import EnhancedModelFactory, ModelType, ModelConfig
from src.utils.ml_common.ensembles import StackingEnsembleManager, StackingEnsembleConfig

logger = system_logger.getChild('RegimeAwareAnalystTraining')

@dataclass
class RegimeAwareAnalystConfig:
    """Configuration for regime-aware Analyst training."""
    # Basic configuration
    model_name: str = "regime_aware_analyst"
    timeframe: str = "5m"
    
    # Analyst outputs
    analyst_output_names: List[str] = field(default_factory=lambda: [
        "signal_strength", "confidence", "risk_score", "regime_label"
    ])
    
    # Base models for Analyst
    analyst_base_models: Dict[str, str] = field(default_factory=lambda: {
        "gru": "GRU",
        "catboost": "CatBoostRegressor",
        "lightgbm": "LGBMRegressor", 
        "ensemble_rf": "RandomForestRegressor"
    })
    
    # Meta model configuration
    meta_model_type: str = "Ridge"
    meta_model_params: Dict[str, Any] = field(default_factory=lambda: {
        "alpha": 1.0,
        "fit_intercept": True
    })
    
    # Regime configuration
    min_samples_per_regime: int = 1000
    enable_regime_merging: bool = True
    regime_merge_threshold: int = 500  # Merge regimes with fewer samples
    
    # Data augmentation
    enable_data_augmentation: bool = True
    augmentation_method: str = "smote"
    augmentation_ratio: float = 1.0
    
    # Training configuration
    validation_split: float = 0.2
    test_split: float = 0.1
    enable_cross_validation: bool = True
    cv_folds: int = 5
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    performance_threshold: float = 0.6

class RegimeAwareAnalystTrainer:
    """
    Regime-Aware Analyst Trainer that trains separate models for each market regime.
    
    This trainer implements per-regime training:
    1. Identify market regimes
    2. Train separate Analyst models for each regime
    3. Implement fallback strategies for small regimes
    4. Create ensemble of regime-specific models
    """
    
    def __init__(self, config: Optional[RegimeAwareAnalystConfig] = None):
        """Initialize regime-aware Analyst trainer."""
        self.config = config or RegimeAwareAnalystConfig()
        self.logger = logger.getChild('RegimeAwareAnalystTrainer')
        
        # Initialize components
        self.model_factory = EnhancedModelFactory()
        self.regime_models = {}
        self.global_model = None
        self.regime_metadata = {}
        
        self.logger.info("✅ Regime-Aware Analyst Trainer initialized")
    
    def train_analyst(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train Analyst using regime-aware approach.
        
        Args:
            X: Input features
            y: Target values (analyst outputs)
            regime_labels: Regime labels for each sample
            feature_names: Names of input features
            
        Returns:
            Dictionary containing trained models and results
        """
        self.logger.info("🚀 Starting regime-aware Analyst training")
        start_time = time.time()
        
        # Step 1: Analyze regimes
        self.logger.info("🔄 Step 1: Analyzing regimes...")
        regime_analysis = self._analyze_regimes(regime_labels)
        
        # Step 2: Prepare regime data
        self.logger.info("🔄 Step 2: Preparing regime data...")
        regime_data = self._prepare_regime_data(X, y, regime_labels, regime_analysis)
        
        # Step 3: Train regime-specific models
        self.logger.info("🔄 Step 3: Training regime-specific models...")
        regime_results = self._train_regime_models(regime_data, feature_names)
        
        # Step 4: Train global fallback model
        self.logger.info("🔄 Step 4: Training global fallback model...")
        global_results = self._train_global_model(X, y, feature_names)
        
        # Step 5: Evaluate performance
        self.logger.info("🔄 Step 5: Evaluating performance...")
        performance_results = self._evaluate_performance(X, y, regime_labels, regime_results, global_results)
        
        # Create results
        total_time = time.time() - start_time
        results = {
            'regime_models': regime_results['models'],
            'regime_meta_models': regime_results['meta_models'],
            'global_model': global_results['model'],
            'global_meta_model': global_results['meta_model'],
            'regime_analysis': regime_analysis,
            'regime_metadata': self.regime_metadata,
            'performance': performance_results,
            'training_time': total_time,
            'config': self.config
        }
        
        self.logger.info(f"✅ Regime-aware Analyst training completed in {total_time:.2f}s")
        self.logger.info(f"📊 Regimes trained: {len(regime_results['models'])}")
        self.logger.info(f"📊 Overall performance: {performance_results['overall_r2']:.4f} R²")
        
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
        regime_analysis: Dict[str, Any]
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """Prepare data for each regime."""
        
        regime_data = {}
        
        for regime in regime_analysis['unique_regimes']:
            regime_mask = regime_labels == regime
            regime_X = X[regime_mask]
            regime_y = y[regime_mask]
            
            # Check if regime has sufficient data
            if len(regime_X) >= self.config.min_samples_per_regime:
                # Sufficient data - use as is
                regime_data[regime] = {
                    'X': regime_X,
                    'y': regime_y,
                    'samples': len(regime_X),
                    'augmented': False
                }
            elif self.config.enable_data_augmentation and len(regime_X) > 100:
                # Insufficient data but enough for augmentation
                augmented_X, augmented_y = self._augment_regime_data(regime_X, regime_y)
                regime_data[regime] = {
                    'X': augmented_X,
                    'y': augmented_y,
                    'samples': len(augmented_X),
                    'augmented': True
                }
            else:
                # Too little data - mark for global model fallback
                regime_data[regime] = {
                    'X': regime_X,
                    'y': regime_y,
                    'samples': len(regime_X),
                    'augmented': False,
                    'use_global': True
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
    
    def _train_regime_models(
        self,
        regime_data: Dict[int, Dict[str, np.ndarray]],
        feature_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Train models for each regime."""
        
        regime_models = {}
        regime_meta_models = {}
        
        for regime, data in regime_data.items():
            if data.get('use_global', False):
                self.logger.info(f"⏭️ Skipping regime {regime} (insufficient data, will use global model)")
                continue
            
            self.logger.info(f"🔄 Training models for regime {regime} ({data['samples']} samples)...")
            
            # Train base models for this regime
            regime_base_models = self._train_regime_base_models(
                data['X'], data['y'], regime, feature_names
            )
            
            # Train meta model for this regime
            regime_meta_model = self._train_regime_meta_model(
                data['X'], data['y'], regime_base_models, regime
            )
            
            regime_models[regime] = regime_base_models
            regime_meta_models[regime] = regime_meta_model
            
            # Store regime metadata
            self.regime_metadata[regime] = {
                'samples': data['samples'],
                'augmented': data['augmented'],
                'base_models': list(regime_base_models.keys()),
                'training_time': time.time()
            }
            
            self.logger.info(f"✅ Regime {regime} models trained")
        
        return {
            'models': regime_models,
            'meta_models': regime_meta_models
        }
    
    def _train_regime_base_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime: int,
        feature_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Train base models for a specific regime."""
        
        base_models = {}
        
        for output_idx, output_name in enumerate(self.config.analyst_output_names):
            self.logger.debug(f"🔄 Training base models for regime {regime}, output {output_name}...")
            
            # Get target for this output
            y_output = y[:, output_idx] if y.ndim > 1 else y
            
            output_models = {}
            
            for model_name, model_type in self.config.analyst_base_models.items():
                self.logger.debug(f"🔄 Training {model_name} for regime {regime}, output {output_name}...")
                
                # Create model
                model_config = ModelConfig(
                    model_type=ModelType[model_type.upper()],
                    model_name=f"analyst_regime_{regime}_{model_name}_{output_name}",
                    model_params=self._get_model_params(model_type)
                )
                
                model = self.model_factory.create_model(model_config)
                
                # Train model
                model.fit(X, y_output)
                
                output_models[model_name] = model
            
            base_models[output_name] = output_models
        
        return base_models
    
    def _train_regime_meta_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_models: Dict[str, Dict[str, Any]],
        regime: int
    ) -> Dict[str, Any]:
        """Train meta model for a specific regime."""
        
        meta_models = {}
        
        for output_idx, output_name in enumerate(self.config.analyst_output_names):
            if output_name not in base_models:
                continue
            
            self.logger.debug(f"🔄 Training meta model for regime {regime}, output {output_name}...")
            
            # Get target for this output
            y_output = y[:, output_idx] if y.ndim > 1 else y
            
            # Get base predictions
            base_predictions = []
            for model_name, model in base_models[output_name].items():
                pred = model.predict(X)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                base_predictions.append(pred)
            
            base_pred_array = np.hstack(base_predictions)
            
            # Combine original features with base predictions
            meta_features = np.hstack([X, base_pred_array])
            
            # Create meta model
            meta_model_config = ModelConfig(
                model_type=ModelType[self.config.meta_model_type.upper()],
                model_name=f"analyst_regime_{regime}_meta_{output_name}",
                model_params=self.config.meta_model_params
            )
            
            meta_model = self.model_factory.create_model(meta_model_config)
            
            # Train meta model
            meta_model.fit(meta_features, y_output)
            
            meta_models[output_name] = meta_model
        
        return meta_models
    
    def _train_global_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Train global fallback model."""
        
        self.logger.info("🔄 Training global fallback model...")
        
        # Train base models
        global_base_models = self._train_regime_base_models(X, y, -1, feature_names)
        
        # Train meta model
        global_meta_model = self._train_regime_meta_model(X, y, global_base_models, -1)
        
        return {
            'model': global_base_models,
            'meta_model': global_meta_model
        }
    
    def _evaluate_performance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        regime_results: Dict[str, Any],
        global_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate performance across regimes."""
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        performance = {
            'by_regime': {},
            'overall': {}
        }
        
        # Evaluate each regime
        for regime in np.unique(regime_labels):
            regime_mask = regime_labels == regime
            regime_X = X[regime_mask]
            regime_y = y[regime_mask]
            
            if regime in regime_results['models']:
                # Use regime-specific model
                regime_performance = self._evaluate_regime_performance(
                    regime_X, regime_y, regime_results['models'][regime], regime_results['meta_models'][regime]
                )
            else:
                # Use global model
                regime_performance = self._evaluate_regime_performance(
                    regime_X, regime_y, global_results['model'], global_results['meta_model']
                )
            
            performance['by_regime'][regime] = regime_performance
        
        # Calculate overall performance
        overall_r2 = np.mean([p['overall_r2'] for p in performance['by_regime'].values()])
        overall_mse = np.mean([p['overall_mse'] for p in performance['by_regime'].values()])
        overall_mae = np.mean([p['overall_mae'] for p in performance['by_regime'].values()])
        
        performance['overall'] = {
            'r2': overall_r2,
            'mse': overall_mse,
            'mae': overall_mae
        }
        
        return performance
    
    def _evaluate_regime_performance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_models: Dict[str, Dict[str, Any]],
        meta_models: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate performance for a specific regime."""
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        output_performance = {}
        
        for output_idx, output_name in enumerate(self.config.analyst_output_names):
            if output_name not in meta_models:
                continue
            
            # Get target for this output
            y_output = y[:, output_idx] if y.ndim > 1 else y
            
            # Get base predictions
            if output_name in base_models:
                base_predictions = []
                for model_name, model in base_models[output_name].items():
                    pred = model.predict(X)
                    if pred.ndim == 1:
                        pred = pred.reshape(-1, 1)
                    base_predictions.append(pred)
                
                base_pred_array = np.hstack(base_predictions)
                
                # Create meta features
                meta_features = np.hstack([X, base_pred_array])
                
                # Make predictions
                meta_model = meta_models[output_name]
                y_pred = meta_model.predict(meta_features)
                
                # Calculate metrics
                mse = mean_squared_error(y_output, y_pred)
                mae = mean_absolute_error(y_output, y_pred)
                r2 = r2_score(y_output, y_pred)
                
                output_performance[output_name] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                }
        
        # Calculate overall performance for this regime
        if output_performance:
            overall_r2 = np.mean([p['r2'] for p in output_performance.values()])
            overall_mse = np.mean([p['mse'] for p in output_performance.values()])
            overall_mae = np.mean([p['mae'] for p in output_performance.values()])
        else:
            overall_r2 = overall_mse = overall_mae = 0.0
        
        output_performance['overall_r2'] = overall_r2
        output_performance['overall_mse'] = overall_mse
        output_performance['overall_mae'] = overall_mae
        
        return output_performance
    
    def _get_model_params(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters for model type."""
        
        default_params = {
            'GRU': {
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.1
            },
            'CATBOOSTREGRESSOR': {
                'n_estimators': 1000,
                'learning_rate': 0.1,
                'depth': 6,
                'random_seed': 42,
                'verbose': False
            },
            'LGBMREGRESSOR': {
                'n_estimators': 1000,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42,
                'verbosity': -1
            },
            'RANDOMFORESTREGRESSOR': {
                'n_estimators': 500,
                'max_depth': 10,
                'random_state': 42
            },
            'RIDGE': {
                'alpha': 1.0,
                'random_state': 42
            }
        }
        
        return default_params.get(model_type.upper(), {})

# Convenience functions
def create_regime_aware_analyst_trainer(config: Optional[RegimeAwareAnalystConfig] = None) -> RegimeAwareAnalystTrainer:
    """Create a regime-aware Analyst trainer."""
    return RegimeAwareAnalystTrainer(config)

def train_regime_aware_analyst(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray,
    config: Optional[RegimeAwareAnalystConfig] = None
) -> Dict[str, Any]:
    """Train Analyst using regime-aware approach."""
    trainer = create_regime_aware_analyst_trainer(config)
    return trainer.train_analyst(X, y, regime_labels)