"""
Advanced Tree Models with Meta-Learning for TAS

This module implements advanced tree-based models with meta-learning capabilities
for the TAS regime detection system, including:
- Gradient Boosting Trees with meta-learning
- Random Forest with adaptive feature selection
- XGBoost with regime-aware optimization
- LightGBM with few-shot learning
- CatBoost with continual learning
- Ensemble methods with meta-learning
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
import time
from datetime import datetime
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# Import tprint for logging
from src.utils.tprint import tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success

# Import tree-based models
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import PatchTST wrapper for enhanced tree models
try:
    from src.training.steps.model_training.patchtst_wrapper import (
        PatchTSTWrapper, create_patchtst_wrapper
    )
    PATCHTST_AVAILABLE = True
except ImportError:
    PATCHTST_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class AdvancedTreeConfig:
    """Configuration for advanced tree models."""

    # Model selection
    primary_model: str = "xgboost"  # "xgboost", "lightgbm", "catboost", "random_forest"
    enable_ensemble: bool = True
    ensemble_models: List[str] = field(default_factory=lambda: ["xgboost", "lightgbm", "catboost"])

    # PatchTST integration
    enable_patchtst_enhancement: bool = True
    patchtst_patch_len: int = 16
    patchtst_stride: int = 8
    patchtst_use_attention: bool = True
    patchtst_regime_aware: bool = True
    patchtst_attention_dropout: float = 0.1
    patchtst_num_heads: int = 4
    patchtst_sign_dropout_rate: float = 0.0
    patchtst_sign_threshold: float = 0.2

    # Meta-learning parameters
    enable_meta_learning: bool = True
    meta_learning_rate: float = 0.01
    meta_adaptation_steps: int = 5
    few_shot_samples: int = 100

    # Continual learning parameters
    enable_continual_learning: bool = True
    continual_learning_rate: float = 0.001
    memory_size: int = 1000
    replay_ratio: float = 0.1

    # Advanced features
    enable_feature_importance_learning: bool = True
    enable_hyperparameter_adaptation: bool = True
    enable_regime_aware_optimization: bool = True

    # Performance optimization
    enable_early_stopping: bool = True
    early_stopping_rounds: int = 50
    enable_cross_validation: bool = True
    cv_folds: int = 5

    # Model-specific parameters
    xgboost_params: Dict[str, Any] = field(default_factory=lambda: {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    })

    lightgbm_params: Dict[str, Any] = field(default_factory=lambda: {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    })

    catboost_params: Dict[str, Any] = field(default_factory=lambda: {
        'max_depth': 6,
        'learning_rate': 0.1,
        'iterations': 100,
        'random_seed': 42,
        'verbose': False
    })

class MetaLearningTreeModel:
    """
    Meta-learning tree model that can quickly adapt to new regimes.
    """

    def __init__(self, base_model, config: AdvancedTreeConfig):
        """Initialize meta-learning tree model."""
        self.base_model = base_model
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Meta-learning state
        self.meta_parameters = {}
        self.adaptation_history = []
        self.regime_performance = {}

        # Few-shot learning
        self.few_shot_examples = []
        self.few_shot_labels = []

        self.logger.info("✅ Meta-learning tree model initialized")

    def meta_train(self, support_data: np.ndarray, support_labels: np.ndarray):
        """
        Meta-train the model on support data.

        Args:
            support_data: Support data for meta-learning
            support_labels: Support labels for meta-learning
        """
        try:
            # Store few-shot examples
            self.few_shot_examples = support_data
            self.few_shot_labels = support_labels

            # Meta-train the base model
            self.base_model.fit(support_data, support_labels)

            # Store meta-parameters
            self.meta_parameters = self._extract_meta_parameters()

            self.logger.info(f"Meta-trained on {len(support_data)} support examples")

        except Exception as e:
            self.logger.error(f"Meta-training failed: {e}")
            raise

    def meta_adapt(self, query_data: np.ndarray, query_labels: np.ndarray):
        """
        Meta-adapt the model to query data.

        Args:
            query_data: Query data for adaptation
            query_labels: Query labels for adaptation
        """
        try:
            # Combine support and query data
            combined_data = np.vstack([self.few_shot_examples, query_data])
            combined_labels = np.hstack([self.few_shot_labels, query_labels])

            # Adapt the model
            for step in range(self.config.meta_adaptation_steps):
                # Fine-tune on combined data
                self.base_model.fit(combined_data, combined_labels)

                # Update meta-parameters
                new_meta_parameters = self._extract_meta_parameters()
                self._update_meta_parameters(new_meta_parameters, step)

            # Record adaptation
            self.adaptation_history.append({
                'timestamp': datetime.now(),
                'query_size': len(query_data),
                'adaptation_steps': self.config.meta_adaptation_steps
            })

            self.logger.info(f"Meta-adapted with {len(query_data)} query examples")

        except Exception as e:
            self.logger.error(f"Meta-adaptation failed: {e}")
            raise

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Make predictions using the meta-learned model."""
        try:
            return self.base_model.predict(data)
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        """Make probability predictions using the meta-learned model."""
        try:
            if hasattr(self.base_model, 'predict_proba'):
                return self.base_model.predict_proba(data)
            else:
                # Convert predictions to probabilities
                predictions = self.predict(data)
                n_classes = len(np.unique(self.few_shot_labels))
                probabilities = np.zeros((len(data), n_classes))
                for i, pred in enumerate(predictions):
                    probabilities[i, pred] = 1.0
                return probabilities
        except Exception as e:
            self.logger.error(f"Probability prediction failed: {e}")
            raise

    def _extract_meta_parameters(self) -> Dict[str, Any]:
        """Extract meta-parameters from the base model."""
        try:
            meta_params = {}

            # Extract feature importance if available
            if hasattr(self.base_model, 'feature_importances_'):
                meta_params['feature_importances'] = self.base_model.feature_importances_

            # Extract model parameters
            if hasattr(self.base_model, 'get_params'):
                meta_params['model_params'] = self.base_model.get_params()

            return meta_params

        except Exception as e:
            self.logger.error(f"Failed to extract meta-parameters: {e}")
            return {}

    def _update_meta_parameters(self, new_params: Dict[str, Any], step: int):
        """Update meta-parameters with learning rate."""
        try:
            learning_rate = self.config.meta_learning_rate * (0.9 ** step)

            for key, value in new_params.items():
                if key in self.meta_parameters:
                    if isinstance(value, np.ndarray):
                        self.meta_parameters[key] = (1 - learning_rate) * self.meta_parameters[key] + learning_rate * value
                    else:
                        self.meta_parameters[key] = value
                else:
                    self.meta_parameters[key] = value

        except Exception as e:
            self.logger.error(f"Failed to update meta-parameters: {e}")

class ContinualLearningTreeModel:
    """
    Continual learning tree model that can learn from new data without forgetting.
    """

    def __init__(self, base_model, config: AdvancedTreeConfig):
        """Initialize continual learning tree model."""
        self.base_model = base_model
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Continual learning state
        self.memory_buffer = []
        self.memory_labels = []
        self.learning_history = []
        self.performance_history = []

        self.logger.info("✅ Continual learning tree model initialized")

    def continual_learn(self, new_data: np.ndarray, new_labels: np.ndarray):
        """
        Continually learn from new data.

        Args:
            new_data: New training data
            new_labels: New training labels
        """
        try:
            # Add to memory buffer
            self.memory_buffer.extend(new_data)
            self.memory_labels.extend(new_labels)

            # Maintain memory size
            if len(self.memory_buffer) > self.config.memory_size:
                # Remove oldest examples
                excess = len(self.memory_buffer) - self.config.memory_size
                self.memory_buffer = self.memory_buffer[excess:]
                self.memory_labels = self.memory_labels[excess:]

            # Prepare training data with replay
            if len(self.memory_buffer) > 0:
                # Sample from memory for replay
                replay_size = int(len(new_data) * self.config.replay_ratio)
                if replay_size > 0:
                    replay_indices = np.random.choice(len(self.memory_buffer),
                                                    min(replay_size, len(self.memory_buffer)),
                                                    replace=False)
                    replay_data = np.array([self.memory_buffer[i] for i in replay_indices])
                    replay_labels = np.array([self.memory_labels[i] for i in replay_indices])

                    # Combine new data with replay data
                    combined_data = np.vstack([new_data, replay_data])
                    combined_labels = np.hstack([new_labels, replay_labels])
                else:
                    combined_data = new_data
                    combined_labels = new_labels
            else:
                combined_data = new_data
                combined_labels = new_labels

            # Train the model
            self.base_model.fit(combined_data, combined_labels)

            # Record learning
            self.learning_history.append({
                'timestamp': datetime.now(),
                'new_samples': len(new_data),
                'total_memory': len(self.memory_buffer),
                'replay_samples': len(combined_data) - len(new_data)
            })

            self.logger.info(f"Continually learned from {len(new_data)} new samples")

        except Exception as e:
            self.logger.error(f"Continual learning failed: {e}")
            raise

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Make predictions using the continually learned model."""
        try:
            return self.base_model.predict(data)
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise

    def predict_proba(self, data: np.ndarray) -> np.ndarray:
        """Make probability predictions using the continually learned model."""
        try:
            if hasattr(self.base_model, 'predict_proba'):
                return self.base_model.predict_proba(data)
            else:
                # Convert predictions to probabilities
                predictions = self.predict(data)
                n_classes = len(np.unique(self.memory_labels)) if self.memory_labels else 2
                probabilities = np.zeros((len(data), n_classes))
                for i, pred in enumerate(predictions):
                    if pred < n_classes:
                        probabilities[i, pred] = 1.0
                return probabilities
        except Exception as e:
            self.logger.error(f"Probability prediction failed: {e}")
            raise

class PatchTSTEnhancedTreeModel:
    """
    PatchTST-enhanced tree model that combines tree-based learning with PatchTST architecture.
    """

    def __init__(self, base_model, patchtst_model, config: AdvancedTreeConfig):
        """Initialize PatchTST-enhanced tree model."""
        self.base_model = base_model
        self.patchtst_model = patchtst_model
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # PatchTST enhancement state
        self.patchtst_features = None
        self.enhancement_history = []

        self.logger.info("✅ PatchTST-enhanced tree model initialized")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the PatchTST-enhanced tree model."""
        try:
            # Extract PatchTST features if available
            if self.patchtst_model and self.config.enable_patchtst_enhancement:
                self.logger.info("🧠 Extracting PatchTST features...")
                self.patchtst_features = self.patchtst_model.transform(X)

                # Combine original features with PatchTST features
                if self.patchtst_features is not None:
                    X_enhanced = np.concatenate([X, self.patchtst_features], axis=1)
                    self.logger.info(f"Enhanced features shape: {X_enhanced.shape}")
                else:
                    X_enhanced = X
                    self.logger.warning("PatchTST feature extraction failed, using original features")
            else:
                X_enhanced = X

            # Train the base model on enhanced features
            self.base_model.fit(X_enhanced, y)

            # Record enhancement
            self.enhancement_history.append({
                'timestamp': datetime.now(),
                'original_features': X.shape[1],
                'enhanced_features': X_enhanced.shape[1],
                'patchtst_features': self.patchtst_features.shape[1] if self.patchtst_features is not None else 0
            })

            self.logger.info(f"PatchTST-enhanced model trained on {X_enhanced.shape[1]} features")

        except Exception as e:
            self.logger.error(f"PatchTST-enhanced model training failed: {e}")
            # Fallback to base model
            self.base_model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using PatchTST-enhanced features."""
        try:
            # Extract PatchTST features for prediction
            if self.patchtst_model and self.config.enable_patchtst_enhancement and self.patchtst_features is not None:
                patchtst_features = self.patchtst_model.transform(X)
                X_enhanced = np.concatenate([X, patchtst_features], axis=1)
            else:
                X_enhanced = X

            return self.base_model.predict(X_enhanced)

        except Exception as e:
            self.logger.error(f"PatchTST-enhanced prediction failed: {e}")
            # Fallback to base model
            return self.base_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions using PatchTST-enhanced features."""
        try:
            # Extract PatchTST features for prediction
            if self.patchtst_model and self.config.enable_patchtst_enhancement and self.patchtst_features is not None:
                patchtst_features = self.patchtst_model.transform(X)
                X_enhanced = np.concatenate([X, patchtst_features], axis=1)
            else:
                X_enhanced = X

            if hasattr(self.base_model, 'predict_proba'):
                return self.base_model.predict_proba(X_enhanced)
            else:
                # Convert predictions to probabilities
                predictions = self.base_model.predict(X_enhanced)
                n_classes = len(np.unique(predictions))
                probabilities = np.zeros((len(X), n_classes))
                for i, pred in enumerate(predictions):
                    probabilities[i, pred] = 1.0
                return probabilities

        except Exception as e:
            self.logger.error(f"PatchTST-enhanced probability prediction failed: {e}")
            # Fallback to base model
            if hasattr(self.base_model, 'predict_proba'):
                return self.base_model.predict_proba(X)
            else:
                predictions = self.base_model.predict(X)
                n_classes = len(np.unique(predictions))
                probabilities = np.zeros((len(X), n_classes))
                for i, pred in enumerate(predictions):
                    probabilities[i, pred] = 1.0
                return probabilities

class AdvancedTreeModelFactory:
    """
    Factory for creating advanced tree models with meta-learning capabilities and PatchTST enhancement.
    """

    def __init__(self, config: AdvancedTreeConfig):
        """Initialize advanced tree model factory."""
        tprint_info("🌳 Initializing Advanced Tree Model Factory")
        tprint_debug(f"Configuration: {config}")
        tprint_debug(f"Primary model: {config.primary_model}")
        tprint_debug(f"Ensemble models: {config.ensemble_models}")
        tprint_debug(f"Meta-learning enabled: {config.enable_meta_learning}")
        tprint_debug(f"Continual learning enabled: {config.enable_continual_learning}")

        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize performance tracking
        self.performance_metrics = {
            'model_creation_time': 0.0,
            'training_time': 0.0,
            'prediction_time': 0.0,
            'total_execution_time': 0.0
        }

        # Initialize PatchTST model if enabled
        self.patchtst_model = None
        if self.config.enable_patchtst_enhancement and PATCHTST_AVAILABLE:
            try:
                # Create a base tree model for PatchTST wrapper
                from sklearn.ensemble import RandomForestRegressor
                base_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )

                patchtst_config = {
                    'patch_len': self.config.patchtst_patch_len,
                    'stride': self.config.patchtst_stride,
                    'use_transformer_attention': self.config.patchtst_use_attention,
                    'regime_aware': self.config.patchtst_regime_aware,
                    'attention_dropout': self.config.patchtst_attention_dropout,
                    'num_heads': self.config.patchtst_num_heads,
                    'sign_dropout_rate': self.config.patchtst_sign_dropout_rate,
                    'sign_threshold': self.config.patchtst_sign_threshold
                }

                self.patchtst_model = create_patchtst_wrapper(base_model, **patchtst_config)
                self.logger.info("✅ PatchTST model initialized for tree enhancement")
            except Exception as e:
                self.logger.warning(f"PatchTST model initialization failed: {e}")
                self.patchtst_model = None

        self.logger.info("✅ Advanced Tree Model Factory initialized")

    def create_model(self, model_type: str, enable_meta_learning: bool = True,
                    enable_continual_learning: bool = True, enable_patchtst_enhancement: bool = True) -> Union[MetaLearningTreeModel, ContinualLearningTreeModel, PatchTSTEnhancedTreeModel]:
        """
        Create an advanced tree model with optional PatchTST enhancement.

        Args:
            model_type: Type of model to create
            enable_meta_learning: Whether to enable meta-learning
            enable_continual_learning: Whether to enable continual learning
            enable_patchtst_enhancement: Whether to enable PatchTST enhancement

        Returns:
            Advanced tree model instance
        """
        tprint_info(f"🌳 Creating {model_type} model")
        tprint_debug(f"Model type: {model_type}")
        tprint_debug(f"Meta-learning enabled: {enable_meta_learning}")
        tprint_debug(f"Continual learning enabled: {enable_continual_learning}")
        tprint_debug(f"PatchTST enhancement enabled: {enable_patchtst_enhancement}")

        creation_start = time.time()

        try:
            tprint_debug(f"Creating base {model_type} model...")
            base_model = self._create_base_model(model_type)
            tprint_debug(f"Base {model_type} model created successfully")

            # Apply PatchTST enhancement if enabled
            if enable_patchtst_enhancement and self.patchtst_model:
                tprint_debug("Applying PatchTST enhancement...")
                base_model = PatchTSTEnhancedTreeModel(base_model, self.patchtst_model, self.config)
                self.logger.info(f"Created PatchTST-enhanced {model_type} model")
                tprint_debug("PatchTST enhancement applied")

            # Apply meta-learning or continual learning
            if enable_meta_learning:
                tprint_debug("Applying MetaLearningTreeModel wrapper...")
                result = MetaLearningTreeModel(base_model, self.config)
                tprint_debug("MetaLearningTreeModel wrapper applied")
            elif enable_continual_learning:
                tprint_debug("Applying ContinualLearningTreeModel wrapper...")
                result = ContinualLearningTreeModel(base_model, self.config)
                tprint_debug("ContinualLearningTreeModel wrapper applied")
            else:
                tprint_debug("Using base model without additional wrappers")
                result = base_model

            creation_time = time.time() - creation_start
            self.performance_metrics['model_creation_time'] = creation_time

            tprint_success(f"✅ {model_type} model created successfully in {creation_time:.3f}s")
            tprint_debug(f"Final model type: {type(result)}")

            return result

        except Exception as e:
            self.logger.error(f"Failed to create model {model_type}: {e}")
            raise

    def _create_base_model(self, model_type: str):
        """Create base model of specified type."""
        try:
            if model_type == "xgboost" and XGBOOST_AVAILABLE:
                return xgb.XGBClassifier(**self.config.xgboost_params)
            elif model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
                return lgb.LGBMClassifier(**self.config.lightgbm_params)
            elif model_type == "catboost" and CATBOOST_AVAILABLE:
                return cb.CatBoostClassifier(**self.config.catboost_params)
            elif model_type == "random_forest" and SKLEARN_AVAILABLE:
                return RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42
                )
            else:
                # No fallback models - require advanced tree models
                raise ValueError(f"Model type {model_type} not available. Only XGBoost, LightGBM, CatBoost, and Random Forest are supported.")

        except Exception as e:
            self.logger.error(f"Failed to create base model {model_type}: {e}")
            raise

    def create_ensemble(self, model_types: List[str]) -> List[Union[MetaLearningTreeModel, ContinualLearningTreeModel]]:
        """
        Create an ensemble of advanced tree models.

        Args:
            model_types: List of model types to include in ensemble

        Returns:
            List of advanced tree models
        """
        try:
            ensemble = []
            for model_type in model_types:
                model = self.create_model(model_type)
                ensemble.append(model)

            self.logger.info(f"Created ensemble with {len(ensemble)} models: {model_types}")
            return ensemble

        except Exception as e:
            self.logger.error(f"Failed to create ensemble: {e}")
            raise

class RegimeAwareTreeOptimizer:
    """
    Regime-aware optimizer for tree models that adapts hyperparameters based on regime characteristics.
    """

    def __init__(self, config: AdvancedTreeConfig):
        """Initialize regime-aware tree optimizer."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Regime-specific optimizations
        self.regime_hyperparameters = {}
        self.regime_performance = {}

        self.logger.info("✅ Regime-aware tree optimizer initialized")

    def optimize_for_regime(self, regime_id: int, regime_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a specific regime.

        Args:
            regime_id: ID of the regime
            regime_characteristics: Characteristics of the regime

        Returns:
            Optimized hyperparameters
        """
        try:
            # Analyze regime characteristics
            volatility = regime_characteristics.get('volatility', 0.1)
            complexity = regime_characteristics.get('complexity_score', 0.5)
            data_size = regime_characteristics.get('data_size', 1000)

            # Adapt hyperparameters based on regime characteristics
            optimized_params = {}

            # Adjust max_depth based on complexity
            if complexity > 0.7:
                optimized_params['max_depth'] = min(10, 6 + int(complexity * 4))
            else:
                optimized_params['max_depth'] = max(3, 6 - int((0.7 - complexity) * 4))

            # Adjust learning_rate based on volatility
            if volatility > 0.2:
                optimized_params['learning_rate'] = max(0.01, 0.1 - volatility * 0.3)
            else:
                optimized_params['learning_rate'] = min(0.3, 0.1 + (0.2 - volatility) * 0.5)

            # Adjust n_estimators based on data size
            if data_size > 10000:
                optimized_params['n_estimators'] = min(500, 100 + int(data_size / 100))
            else:
                optimized_params['n_estimators'] = max(50, 100 - int((10000 - data_size) / 200))

            # Adjust subsample based on data size
            if data_size < 1000:
                optimized_params['subsample'] = 1.0  # Use all data for small datasets
            else:
                optimized_params['subsample'] = max(0.6, 0.8 - (data_size - 1000) / 10000)

            # Store regime-specific parameters
            self.regime_hyperparameters[regime_id] = optimized_params

            self.logger.info(f"Optimized hyperparameters for regime {regime_id}: {optimized_params}")

            return optimized_params

        except Exception as e:
            self.logger.error(f"Failed to optimize for regime {regime_id}: {e}")
            return {}

    def get_regime_hyperparameters(self, regime_id: int) -> Dict[str, Any]:
        """Get optimized hyperparameters for a regime."""
        return self.regime_hyperparameters.get(regime_id, {})

    def update_regime_performance(self, regime_id: int, performance: float):
        """Update performance tracking for a regime."""
        if regime_id not in self.regime_performance:
            self.regime_performance[regime_id] = []
        self.regime_performance[regime_id].append(performance)

    def get_regime_performance_trend(self, regime_id: int) -> str:
        """Get performance trend for a regime."""
        if regime_id not in self.regime_performance or len(self.regime_performance[regime_id]) < 2:
            return "unknown"

        recent_performance = self.regime_performance[regime_id][-5:]  # Last 5 measurements
        if len(recent_performance) < 2:
            return "unknown"

        trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]

        if trend > 0.01:
            return "improving"
        elif trend < -0.01:
            return "declining"
        else:
            return "stable"
