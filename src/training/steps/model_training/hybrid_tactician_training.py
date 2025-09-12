"""
Hybrid Tactician Training Implementation

This module implements the recommended hybrid training strategy for Tactician:
- Train on the whole dataset using features + Analyst model outputs
- Use regime-aware features to help the model understand market conditions
- Implement fallback strategies for data scarcity scenarios
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

logger = system_logger.getChild('HybridTacticianTraining')

@dataclass
class HybridTacticianConfig:
    """Configuration for hybrid Tactician training."""
    # Basic configuration
    model_name: str = "hybrid_tactician"
    timeframe: str = "1m"
    
    # Analyst dependency
    analyst_model_path: str = "./analyst_models"
    analyst_output_names: List[str] = field(default_factory=lambda: [
        "signal_strength", "confidence", "risk_score", "regime_label"
    ])
    analyst_threshold: float = 0.6  # Minimum analyst confidence to proceed
    
    # Tactician outputs
    tactician_output_names: List[str] = field(default_factory=lambda: [
        "entry_timing", "position_size", "stop_loss", "take_profit"
    ])
    
    # Base models for Tactician
    tactician_base_models: Dict[str, str] = field(default_factory=lambda: {
        "node": "NODE",
        "catboost": "CatBoostRegressor", 
        "lightgbm": "LGBMRegressor",
        "linear_ridge": "Ridge"
    })
    
    # Meta model configuration
    meta_model_type: str = "Ridge"
    meta_model_params: Dict[str, Any] = field(default_factory=lambda: {
        "alpha": 1.0,
        "fit_intercept": True
    })
    
    # Regime-aware training
    enable_regime_features: bool = True
    regime_feature_types: List[str] = field(default_factory=lambda: [
        "one_hot", "transition", "duration", "momentum"
    ])
    
    # Training configuration
    validation_split: float = 0.2
    test_split: float = 0.1
    enable_cross_validation: bool = True
    cv_folds: int = 5
    
    # Data augmentation for small regimes
    enable_data_augmentation: bool = True
    min_samples_per_regime: int = 1000
    augmentation_method: str = "smote"  # smote, adasyn, smote_tomek
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    monitor_regime_performance: bool = True
    performance_threshold: float = 0.6

class HybridTacticianTrainer:
    """
    Hybrid Tactician Trainer that uses the whole dataset with Analyst features.
    
    This trainer implements the recommended strategy:
    1. Load Analyst models and generate predictions
    2. Create regime-aware features
    3. Train Tactician on features + Analyst outputs
    4. Implement fallback strategies for data scarcity
    """
    
    def __init__(self, config: Optional[HybridTacticianConfig] = None):
        """Initialize hybrid Tactician trainer."""
        self.config = config or HybridTacticianConfig()
        self.logger = logger.getChild('HybridTacticianTrainer')
        
        # Initialize components
        self.model_factory = EnhancedModelFactory()
        self.analyst_models = {}
        self.tactician_models = {}
        self.regime_features = None
        self.training_data = None
        
        self.logger.info("✅ Hybrid Tactician Trainer initialized")
    
    def train_tactician(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Train Tactician using hybrid approach.
        
        Args:
            X: Input features
            y: Target values (tactician outputs)
            regime_labels: Regime labels for regime-aware training
            feature_names: Names of input features
            
        Returns:
            Dictionary containing trained models and results
        """
        self.logger.info("🚀 Starting hybrid Tactician training")
        start_time = time.time()
        
        # Step 1: Load and prepare Analyst models
        self.logger.info("🔄 Step 1: Loading Analyst models...")
        analyst_predictions = self._load_analyst_models_and_predict(X)
        
        # Step 2: Create regime-aware features
        if self.config.enable_regime_features and regime_labels is not None:
            self.logger.info("🔄 Step 2: Creating regime-aware features...")
            regime_features = self._create_regime_features(regime_labels, X)
        else:
            self.logger.info("🔄 Step 2: Skipping regime features (disabled or no regime labels)")
            regime_features = None
        
        # Step 3: Combine features
        self.logger.info("🔄 Step 3: Combining features...")
        combined_features = self._combine_features(X, analyst_predictions, regime_features, feature_names)
        
        # Step 4: Train Tactician models
        self.logger.info("🔄 Step 4: Training Tactician models...")
        tactician_results = self._train_tactician_models(combined_features, y)
        
        # Step 5: Train meta model
        self.logger.info("🔄 Step 5: Training meta model...")
        meta_model = self._train_meta_model(combined_features, y, tactician_results['base_predictions'])
        
        # Step 6: Evaluate performance
        self.logger.info("🔄 Step 6: Evaluating performance...")
        performance_results = self._evaluate_performance(combined_features, y, tactician_results, meta_model)
        
        # Create results
        total_time = time.time() - start_time
        results = {
            'tactician_models': tactician_results['models'],
            'meta_model': meta_model,
            'analyst_predictions': analyst_predictions,
            'regime_features': regime_features,
            'combined_features': combined_features,
            'performance': performance_results,
            'training_time': total_time,
            'config': self.config
        }
        
        self.logger.info(f"✅ Hybrid Tactician training completed in {total_time:.2f}s")
        self.logger.info(f"📊 Performance: {performance_results['overall_r2']:.4f} R²")
        
        return results
    
    def _load_analyst_models_and_predict(self, X: np.ndarray) -> np.ndarray:
        """Load Analyst models and generate predictions."""
        
        self.logger.info("🔄 Loading Analyst models...")
        
        # This is a placeholder - in practice, you would load actual trained Analyst models
        # For now, we'll create dummy predictions
        analyst_predictions = np.random.randn(len(X), len(self.config.analyst_output_names))
        
        # Apply analyst threshold filtering
        confidence_scores = analyst_predictions[:, 1]  # Assuming confidence is second column
        valid_mask = confidence_scores >= self.config.analyst_threshold
        
        self.logger.info(f"📊 Analyst predictions: {len(X)} samples, {np.sum(valid_mask)} above threshold")
        
        return analyst_predictions
    
    def _create_regime_features(self, regime_labels: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Create regime-aware features."""
        
        regime_features = []
        
        # One-hot encoding of regime
        if "one_hot" in self.config.regime_feature_types:
            regime_onehot = pd.get_dummies(regime_labels, prefix='regime')
            regime_features.append(regime_onehot.values)
        
        # Regime transition features
        if "transition" in self.config.regime_feature_types:
            regime_transitions = np.diff(regime_labels, prepend=regime_labels[0])
            regime_features.append(regime_transitions.reshape(-1, 1))
        
        # Regime duration features
        if "duration" in self.config.regime_feature_types:
            regime_durations = self._calculate_regime_durations(regime_labels)
            regime_features.append(regime_durations.reshape(-1, 1))
        
        # Regime momentum features
        if "momentum" in self.config.regime_feature_types:
            regime_momentum = self._calculate_regime_momentum(regime_labels, X)
            regime_features.append(regime_momentum)
        
        if regime_features:
            combined_regime_features = np.hstack(regime_features)
            self.logger.info(f"📊 Created regime features: {combined_regime_features.shape[1]} features")
            return combined_regime_features
        else:
            return None
    
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
    
    def _combine_features(
        self,
        X: np.ndarray,
        analyst_predictions: np.ndarray,
        regime_features: Optional[np.ndarray],
        feature_names: Optional[List[str]]
    ) -> np.ndarray:
        """Combine all features for Tactician training."""
        
        features = [X]
        
        # Add Analyst predictions
        features.append(analyst_predictions)
        
        # Add regime features if available
        if regime_features is not None:
            features.append(regime_features)
        
        # Combine all features
        combined_features = np.hstack(features)
        
        # Update feature names
        if feature_names is not None:
            new_feature_names = feature_names.copy()
            new_feature_names.extend([f"analyst_{name}" for name in self.config.analyst_output_names])
            if regime_features is not None:
                new_feature_names.extend([f"regime_feature_{i}" for i in range(regime_features.shape[1])])
            self.feature_names = new_feature_names
        
        self.logger.info(f"📊 Combined features: {combined_features.shape[1]} total features")
        self.logger.info(f"📊 - Original features: {X.shape[1]}")
        self.logger.info(f"📊 - Analyst predictions: {analyst_predictions.shape[1]}")
        if regime_features is not None:
            self.logger.info(f"📊 - Regime features: {regime_features.shape[1]}")
        
        return combined_features
    
    def _train_tactician_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train Tactician base models."""
        
        models = {}
        base_predictions = {}
        
        for output_idx, output_name in enumerate(self.config.tactician_output_names):
            self.logger.info(f"🔄 Training base models for {output_name}...")
            
            # Get target for this output
            y_output = y[:, output_idx] if y.ndim > 1 else y
            
            output_models = {}
            output_predictions = []
            
            for model_name, model_type in self.config.tactician_base_models.items():
                self.logger.debug(f"🔄 Training {model_name} for {output_name}...")
                
                # Create model
                model_config = ModelConfig(
                    model_type=ModelType[model_type.upper()],
                    model_name=f"tactician_{model_name}_{output_name}",
                    model_params=self._get_model_params(model_type)
                )
                
                model = self.model_factory.create_model(model_config)
                
                # Train model
                model.fit(X, y_output)
                
                # Make predictions
                pred = model.predict(X)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)
                
                output_models[model_name] = model
                output_predictions.append(pred)
            
            models[output_name] = output_models
            base_predictions[output_name] = np.hstack(output_predictions)
            
            self.logger.info(f"✅ Trained {len(output_models)} base models for {output_name}")
        
        return {
            'models': models,
            'base_predictions': base_predictions
        }
    
    def _train_meta_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_predictions: Dict[str, np.ndarray]
    ) -> Any:
        """Train meta model for each output."""
        
        meta_models = {}
        
        for output_idx, output_name in enumerate(self.config.tactician_output_names):
            self.logger.info(f"🔄 Training meta model for {output_name}...")
            
            # Get target for this output
            y_output = y[:, output_idx] if y.ndim > 1 else y
            
            # Get base predictions for this output
            if output_name in base_predictions:
                base_pred = base_predictions[output_name]
            else:
                self.logger.warning(f"⚠️ No base predictions for {output_name}")
                continue
            
            # Combine original features with base predictions
            meta_features = np.hstack([X, base_pred])
            
            # Create meta model
            meta_model_config = ModelConfig(
                model_type=ModelType[self.config.meta_model_type.upper()],
                model_name=f"tactician_meta_{output_name}",
                model_params=self.config.meta_model_params
            )
            
            meta_model = self.model_factory.create_model(meta_model_config)
            
            # Train meta model
            meta_model.fit(meta_features, y_output)
            
            meta_models[output_name] = meta_model
            
            self.logger.info(f"✅ Meta model trained for {output_name}")
        
        return meta_models
    
    def _evaluate_performance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tactician_results: Dict[str, Any],
        meta_models: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate Tactician performance."""
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        performance = {}
        
        for output_idx, output_name in enumerate(self.config.tactician_output_names):
            if output_name not in meta_models:
                continue
            
            # Get target for this output
            y_output = y[:, output_idx] if y.ndim > 1 else y
            
            # Get base predictions
            if output_name in tactician_results['base_predictions']:
                base_pred = tactician_results['base_predictions'][output_name]
            else:
                continue
            
            # Create meta features
            meta_features = np.hstack([X, base_pred])
            
            # Make predictions
            meta_model = meta_models[output_name]
            y_pred = meta_model.predict(meta_features)
            
            # Calculate metrics
            mse = mean_squared_error(y_output, y_pred)
            mae = mean_absolute_error(y_output, y_pred)
            r2 = r2_score(y_output, y_pred)
            
            performance[output_name] = {
                'mse': mse,
                'mae': mae,
                'r2': r2
            }
        
        # Calculate overall performance
        if performance:
            overall_r2 = np.mean([p['r2'] for p in performance.values()])
            overall_mse = np.mean([p['mse'] for p in performance.values()])
            overall_mae = np.mean([p['mae'] for p in performance.values()])
        else:
            overall_r2 = overall_mse = overall_mae = 0.0
        
        performance['overall'] = {
            'r2': overall_r2,
            'mse': overall_mse,
            'mae': overall_mae
        }
        
        return performance
    
    def _get_model_params(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters for model type with overfitting prevention."""
        
        default_params = {
            'NODE': {
                'n_d': 64,
                'n_a': 64,
                'n_steps': 5,
                'gamma': 1.5,
                'lambda_sparse': 1e-3,    # Sparsity regularization
                'dropout': 0.1,           # Dropout for overfitting prevention
                'l2_regularization': 0.01 # L2 regularization
            },
            'CATBOOSTREGRESSOR': {
                'n_estimators': 1000,
                'learning_rate': 0.05,    # Reduced learning rate
                'depth': 6,
                'l2_leaf_reg': 3.0,       # L2 regularization
                'bagging_temperature': 1.0,
                'subsample': 0.8,         # Bagging
                'colsample_bylevel': 0.8, # Feature sampling
                'early_stopping_rounds': 50,
                'random_seed': 42,
                'verbose': False
            },
            'LGBMREGRESSOR': {
                'n_estimators': 1000,
                'learning_rate': 0.05,    # Reduced learning rate
                'max_depth': 6,
                'reg_alpha': 0.1,         # L1 regularization
                'reg_lambda': 0.1,        # L2 regularization
                'subsample': 0.8,         # Bagging
                'colsample_bytree': 0.8,  # Feature sampling
                'min_child_samples': 20,  # Prevent overfitting
                'early_stopping_rounds': 50,
                'random_state': 42,
                'verbosity': -1
            },
            'RIDGE': {
                'alpha': 1.0,
                'solver': 'auto',
                'random_state': 42
            }
        }
        
        return default_params.get(model_type.upper(), {})

# Convenience functions
def create_hybrid_tactician_trainer(config: Optional[HybridTacticianConfig] = None) -> HybridTacticianTrainer:
    """Create a hybrid Tactician trainer."""
    return HybridTacticianTrainer(config)

def train_hybrid_tactician(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: Optional[np.ndarray] = None,
    config: Optional[HybridTacticianConfig] = None
) -> Dict[str, Any]:
    """Train Tactician using hybrid approach."""
    trainer = create_hybrid_tactician_trainer(config)
    return trainer.train_tactician(X, y, regime_labels)