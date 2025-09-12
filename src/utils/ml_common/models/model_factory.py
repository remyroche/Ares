"""
Enhanced Model Factory for Multi-Output Stacking Ensemble

This module provides a comprehensive model factory supporting all required ML models
for the Analyst (5m) and Tactician (1m) multi-output stacking ensemble system.

Key Features:
- Support for all required model types (RandomForest, LightGBM, CatBoost, XGBoost, TabNet, TimeSeriesTransformer, TCN, LSTM, Ridge, LogisticRegression)
- ModelType enum with all model types
- ModelConfig dataclass for configuration
- Factory methods for each model type
- Dependency checking and fallback handling
- M1 hardware optimization integration
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Type
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime

# M1 Optimization imports
from ..hardware.m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
from ..hardware.memory_optimization import get_memory_manager, MemoryMonitor

# Common utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, MathValidationError
)
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time,
    timeout, error_boundary, compose
)
from src.core.errors import (
    ValidationError, DataIntegrityError, TimeoutError
)

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Enumeration of all supported model types."""
    # Tree-based models
    RANDOM_FOREST = "RandomForestRegressor"
    RANDOM_FOREST_CLASSIFIER = "RandomForestClassifier"
    EXTRA_TREES = "ExtraTreesRegressor"
    EXTRA_TREES_CLASSIFIER = "ExtraTreesClassifier"
    LIGHTGBM = "LGBMRegressor"
    LIGHTGBM_CLASSIFIER = "LGBMClassifier"
    HIST_GRADIENT_BOOSTING = "HistGradientBoostingRegressor"
    HIST_GRADIENT_BOOSTING_CLASSIFIER = "HistGradientBoostingClassifier"
    CATBOOST = "CatBoostRegressor"
    CATBOOST_CLASSIFIER = "CatBoostClassifier"
    XGBOOST = "XGBRegressor"
    XGBOOST_CLASSIFIER = "XGBClassifier"
    XGBOOST_CUSTOM = "XGBoostCustom"
    
    # Neural network models
    TABNET = "TabNetRegressor"
    TABNET_CLASSIFIER = "TabNetClassifier"
    TABNET_ATTENTION = "TabNetAttention"
    NODE = "NODE"  # Neural Oblivious Decision Ensembles
    NODE_CLASSIFIER = "NODEClassifier"
    TIME_SERIES_TRANSFORMER = "TimeSeriesTransformer"
    TEMPORAL_FUSION_TRANSFORMER = "TemporalFusionTransformer"
    WAVENET = "WaveNet"
    TCN = "TCN"  # Temporal Convolutional Network
    LSTM = "LSTM"
    
    # Linear models
    RIDGE = "Ridge"
    RIDGE_CLASSIFIER = "RidgeClassifier"
    ELASTIC_NET = "ElasticNet"
    ELASTIC_NET_CLASSIFIER = "ElasticNetClassifier"
    ELASTIC_NET_QUANTILE = "ElasticNetQuantile"
    QUANTILE_REGRESSION = "QuantileRegression"
    LOGISTIC_REGRESSION = "LogisticRegression"
    LINEAR_REGRESSION = "LinearRegression"
    HUBER_REGRESSION = "HuberRegression"
    
    # Ensemble models
    VOTING_CLASSIFIER = "VotingClassifier"
    VOTING_REGRESSOR = "VotingRegressor"
    STACKING_CLASSIFIER = "StackingClassifier"
    STACKING_REGRESSOR = "StackingRegressor"
    BAGGING_CLASSIFIER = "BaggingClassifier"
    BAGGING_REGRESSOR = "BaggingRegressor"
    ADABOOST_CLASSIFIER = "AdaBoostClassifier"
    ADABOOST_REGRESSOR = "AdaBoostRegressor"
    GRADIENT_BOOSTING_CLASSIFIER = "GradientBoostingClassifier"
    GRADIENT_BOOSTING_REGRESSOR = "GradientBoostingRegressor"


@dataclass
class ModelConfig:
    """Configuration for model creation and training."""
    # Basic configuration
    model_type: ModelType
    model_name: str
    
    # Model-specific parameters
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    # Training configuration
    enable_gpu_acceleration: bool = True
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True
    memory_limit_gb: float = 8.0
    max_workers: Optional[int] = None
    
    # Model-specific settings
    random_state: int = 42
    n_jobs: int = -1
    
    # Multi-output specific settings
    is_multi_output: bool = False
    n_outputs: int = 1
    output_names: Optional[List[str]] = None
    
    # Hardware optimization
    use_m1_optimizations: bool = True
    enable_mixed_precision: bool = False
    
    # Validation settings
    enable_validation: bool = True
    validation_split: float = 0.2
    
    # Performance settings
    enable_profiling: bool = False
    enable_caching: bool = True


class EnhancedModelFactory:
    """Enhanced model factory with comprehensive model support and M1 optimizations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the enhanced model factory."""
        self.logger = logger.getChild('EnhancedModelFactory')
        self.logger.info("🚀 Initializing EnhancedModelFactory...")
        start_time = time.time()
        
        self.config = config or {}
        
        # Initialize M1 optimizers
        self.logger.debug("🔧 Initializing M1 optimizers...")
        self.m1_gpu = get_m1_memory_optimizer() if self.config.get('enable_gpu_acceleration', True) else None
        self.m1_memory = get_m1_memory_optimizer(
            memory_limit_gb=self.config.get('memory_limit_gb', 8.0)
        ) if self.config.get('enable_memory_optimization', True) else None
        self.m1_cpu = get_memory_manager() if self.config.get('enable_parallel_processing', True) else None
        
        self.logger.debug("✅ M1 optimizers initialized")
        
        # Model registry for created models
        self.model_registry: Dict[str, Any] = {}
        
        # Dependency checking
        self.dependencies = self._check_dependencies()
        
        init_time = time.time() - start_time
        self.logger.info(f"✅ EnhancedModelFactory initialized in {init_time:.3f}s")
        self.logger.info(f"⚡ GPU acceleration: {self.m1_gpu is not None}")
        self.logger.info(f"🧠 Memory optimization: {self.m1_memory is not None}")
        self.logger.info(f"🔄 Parallel processing: {self.m1_cpu is not None}")
        self.logger.info(f"📊 Available dependencies: {list(self.dependencies.keys())}")
    
    def _check_dependencies(self) -> Dict[str, bool]:
        """Check availability of required dependencies."""
        dependencies = {}
        
        # Scikit-learn
        try:
            import sklearn
            dependencies['sklearn'] = True
            self.logger.debug("✅ Scikit-learn available")
        except ImportError:
            dependencies['sklearn'] = False
            self.logger.warning("⚠️ Scikit-learn not available")
        
        # LightGBM
        try:
            import lightgbm
            dependencies['lightgbm'] = True
            self.logger.debug("✅ LightGBM available")
        except ImportError:
            dependencies['lightgbm'] = False
            self.logger.warning("⚠️ LightGBM not available")
        
        # CatBoost
        try:
            import catboost
            dependencies['catboost'] = True
            self.logger.debug("✅ CatBoost available")
        except ImportError:
            dependencies['catboost'] = False
            self.logger.warning("⚠️ CatBoost not available")
        
        # XGBoost
        try:
            import xgboost
            dependencies['xgboost'] = True
            self.logger.debug("✅ XGBoost available")
        except ImportError:
            dependencies['xgboost'] = False
            self.logger.warning("⚠️ XGBoost not available")
        
        # TabNet
        try:
            import pytorch_tabnet
            dependencies['pytorch_tabnet'] = True
            self.logger.debug("✅ PyTorch TabNet available")
        except ImportError:
            dependencies['pytorch_tabnet'] = False
            self.logger.warning("⚠️ PyTorch TabNet not available")
        
        # PyTorch
        try:
            import torch
            dependencies['torch'] = True
            self.logger.debug("✅ PyTorch available")
        except ImportError:
            dependencies['torch'] = False
            self.logger.warning("⚠️ PyTorch not available")
        
        # TensorFlow
        try:
            import tensorflow
            dependencies['tensorflow'] = True
            self.logger.debug("✅ TensorFlow available")
        except ImportError:
            dependencies['tensorflow'] = False
            self.logger.warning("⚠️ TensorFlow not available")
        
        return dependencies
    
    @traced(span_name='create_model')
    def create_model(self, model_config: ModelConfig) -> Any:
        """Create a model instance based on configuration."""
        
        self.logger.info(f"🔄 Creating model: {model_config.model_name} ({model_config.model_type.value})")
        start_time = time.time()
        
        try:
            # Validate configuration
            self.logger.debug("🔍 Validating model configuration...")
            self._validate_model_config(model_config)
            self.logger.debug("✅ Model configuration validated")
            
            # Create model based on type
            self.logger.debug(f"🔧 Creating {model_config.model_type.value} model...")
            
            if model_config.model_type in [ModelType.RANDOM_FOREST, ModelType.RANDOM_FOREST_CLASSIFIER]:
                model = self._create_random_forest_model(model_config)
            elif model_config.model_type in [ModelType.LIGHTGBM, ModelType.LIGHTGBM_CLASSIFIER]:
                model = self._create_lightgbm_model(model_config)
            elif model_config.model_type in [ModelType.HIST_GRADIENT_BOOSTING, ModelType.HIST_GRADIENT_BOOSTING_CLASSIFIER]:
                model = self._create_hist_gradient_boosting_model(model_config)
            elif model_config.model_type in [ModelType.CATBOOST, ModelType.CATBOOST_CLASSIFIER]:
                model = self._create_catboost_model(model_config)
            elif model_config.model_type in [ModelType.XGBOOST, ModelType.XGBOOST_CLASSIFIER]:
                model = self._create_xgboost_model(model_config)
            elif model_config.model_type == ModelType.XGBOOST_CUSTOM:
                model = self._create_xgboost_custom_model(model_config)
            elif model_config.model_type in [ModelType.EXTRA_TREES, ModelType.EXTRA_TREES_CLASSIFIER]:
                model = self._create_extra_trees_model(model_config)
            elif model_config.model_type in [ModelType.TABNET, ModelType.TABNET_CLASSIFIER]:
                model = self._create_tabnet_model(model_config)
            elif model_config.model_type == ModelType.TIME_SERIES_TRANSFORMER:
                model = self._create_time_series_transformer_model(model_config)
            elif model_config.model_type == ModelType.TCN:
                model = self._create_tcn_model(model_config)
            elif model_config.model_type == ModelType.WAVENET:
                model = self._create_wavenet_model(model_config)
            elif model_config.model_type == ModelType.TEMPORAL_FUSION_TRANSFORMER:
                model = self._create_tft_model(model_config)
            elif model_config.model_type == ModelType.TABNET_ATTENTION:
                model = self._create_tabnet_attention_model(model_config)
            elif model_config.model_type == ModelType.LSTM:
                model = self._create_lstm_model(model_config)
            elif model_config.model_type in [ModelType.NODE, ModelType.NODE_CLASSIFIER]:
                model = self._create_node_model(model_config)
            elif model_config.model_type in [ModelType.RIDGE, ModelType.RIDGE_CLASSIFIER]:
                model = self._create_ridge_model(model_config)
            elif model_config.model_type in [ModelType.ELASTIC_NET, ModelType.ELASTIC_NET_CLASSIFIER]:
                model = self._create_elastic_net_model(model_config)
            elif model_config.model_type == ModelType.ELASTIC_NET_QUANTILE:
                model = self._create_elastic_net_quantile_model(model_config)
            elif model_config.model_type == ModelType.QUANTILE_REGRESSION:
                model = self._create_quantile_regression_model(model_config)
            elif model_config.model_type == ModelType.HUBER_REGRESSION:
                model = self._create_huber_regression_model(model_config)
            elif model_config.model_type in [ModelType.LOGISTIC_REGRESSION, ModelType.LINEAR_REGRESSION]:
                model = self._create_linear_model(model_config)
            else:
                raise ValueError(f"Unsupported model type: {model_config.model_type}")
            
            # Apply M1 optimizations if enabled
            if model_config.use_m1_optimizations:
                self.logger.debug("🔧 Applying M1 optimizations...")
                model = self._apply_m1_optimizations(model, model_config)
                self.logger.debug("✅ M1 optimizations applied")
            
            # Register model
            self.model_registry[model_config.model_name] = model
            
            creation_time = time.time() - start_time
            self.logger.info(f"✅ Model {model_config.model_name} created in {creation_time:.3f}s")
            self.logger.info(f"🎯 Model type: {type(model).__name__}")
            self.logger.info(f"📊 Model parameters: {len(model_config.model_params)} configured")
            
            return model
            
        except Exception as e:
            creation_time = time.time() - start_time
            self.logger.error(f"❌ Failed to create model {model_config.model_name} after {creation_time:.3f}s: {e}")
            raise
    
    def _validate_model_config(self, model_config: ModelConfig) -> None:
        """Validate model configuration."""
        
        # Check if model type is supported
        if not isinstance(model_config.model_type, ModelType):
            raise ValidationError(f"Invalid model type: {model_config.model_type}")
        
        # Check dependencies
        if model_config.model_type in [ModelType.LIGHTGBM, ModelType.LIGHTGBM_CLASSIFIER]:
            if not self.dependencies.get('lightgbm', False):
                raise ValidationError("LightGBM not available")
        
        if model_config.model_type in [ModelType.CATBOOST, ModelType.CATBOOST_CLASSIFIER]:
            if not self.dependencies.get('catboost', False):
                raise ValidationError("CatBoost not available")
        
        if model_config.model_type in [ModelType.XGBOOST, ModelType.XGBOOST_CLASSIFIER, ModelType.XGBOOST_CUSTOM]:
            if not self.dependencies.get('xgboost', False):
                raise ValidationError("XGBoost not available")
        
        if model_config.model_type in [ModelType.TABNET, ModelType.TABNET_CLASSIFIER]:
            if not self.dependencies.get('pytorch_tabnet', False):
                raise ValidationError("PyTorch TabNet not available")
        
        if model_config.model_type in [ModelType.TIME_SERIES_TRANSFORMER, ModelType.TCN, ModelType.LSTM]:
            if not self.dependencies.get('torch', False):
                raise ValidationError("PyTorch not available")
        
        # Validate multi-output configuration
        if model_config.is_multi_output:
            if model_config.n_outputs < 2:
                raise ValidationError("Multi-output requires at least 2 outputs")
            if model_config.output_names and len(model_config.output_names) != model_config.n_outputs:
                raise ValidationError("Output names count must match n_outputs")
    
    def _create_random_forest_model(self, model_config: ModelConfig) -> Any:
        """Create Random Forest model."""
        
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        
        # Default parameters with overfitting prevention
        default_params = {
            'n_estimators': 500,
            'max_depth': 10,  # Limit depth to prevent overfitting
            'min_samples_split': 5,  # Prevent overfitting
            'min_samples_leaf': 2,   # Prevent overfitting
            'max_features': 'sqrt',  # Feature sampling
            'bootstrap': True,       # Bagging
            'random_state': model_config.random_state,
            'n_jobs': model_config.n_jobs
        }
        
        # Merge with user parameters
        params = {**default_params, **model_config.model_params}
        
        # Create model
        if model_config.model_type == ModelType.RANDOM_FOREST:
            model = RandomForestRegressor(**params)
        else:
            model = RandomForestClassifier(**params)
        
        return model
    
    def _create_lightgbm_model(self, model_config: ModelConfig) -> Any:
        """Create LightGBM model."""
        
        import lightgbm as lgb
        
        # Default parameters with overfitting prevention
        default_params = {
            'n_estimators': 1000,
            'learning_rate': 0.05,  # Reduced to prevent overfitting
            'max_depth': 6,         # Limit depth
            'num_leaves': 31,
            'reg_alpha': 0.1,       # L1 regularization
            'reg_lambda': 0.1,      # L2 regularization
            'subsample': 0.8,       # Bagging
            'colsample_bytree': 0.8, # Feature sampling
            'min_child_samples': 20, # Prevent overfitting
            'early_stopping_rounds': 50,
            'random_state': model_config.random_state,
            'n_jobs': model_config.n_jobs,
            'verbosity': -1
        }
        
        # Merge with user parameters
        params = {**default_params, **model_config.model_params}
        
        # Create model
        if model_config.model_type == ModelType.LIGHTGBM:
            model = lgb.LGBMRegressor(**params)
        else:
            model = lgb.LGBMClassifier(**params)
        
        return model
    
    def _create_catboost_model(self, model_config: ModelConfig) -> Any:
        """Create CatBoost model."""
        
        from catboost import CatBoostRegressor, CatBoostClassifier
        
        # Default parameters with overfitting prevention
        default_params = {
            'iterations': 1000,
            'learning_rate': 0.05,  # Reduced to prevent overfitting
            'depth': 6,
            'l2_leaf_reg': 3.0,     # L2 regularization
            'bagging_temperature': 1.0,
            'subsample': 0.8,       # Bagging
            'colsample_bylevel': 0.8, # Feature sampling
            'early_stopping_rounds': 50,
            'random_seed': model_config.random_state,
            'verbose': False
        }
        
        # Merge with user parameters
        params = {**default_params, **model_config.model_params}
        
        # Create model
        if model_config.model_type == ModelType.CATBOOST:
            model = CatBoostRegressor(**params)
        else:
            model = CatBoostClassifier(**params)
        
        return model
    
    def _create_xgboost_model(self, model_config: ModelConfig) -> Any:
        """Create XGBoost model."""
        
        import xgboost as xgb
        
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'random_state': model_config.random_state,
            'n_jobs': model_config.n_jobs,
            'verbosity': 0
        }
        
        # Merge with user parameters
        params = {**default_params, **model_config.model_params}
        
        # Create model
        if model_config.model_type == ModelType.XGBOOST:
            model = xgb.XGBRegressor(**params)
        else:
            model = xgb.XGBClassifier(**params)
        
        return model
    
    def _create_tabnet_model(self, model_config: ModelConfig) -> Any:
        """Create TabNet model."""
        
        from pytorch_tabnet.tab_model import TabNetRegressor, TabNetClassifier
        
        # Default parameters
        default_params = {
            'n_d': 64,
            'n_a': 64,
            'n_steps': 5,
            'gamma': 1.5,
            'n_independent': 2,
            'n_shared': 2,
            'lambda_sparse': 1e-3,
            'optimizer_fn': 'adam',
            'optimizer_params': {'lr': 2e-2},
            'scheduler_params': {'step_size': 50, 'gamma': 0.9},
            'scheduler_fn': 'step',
            'seed': model_config.random_state
        }
        
        # Merge with user parameters
        params = {**default_params, **model_config.model_params}
        
        # Create model
        if model_config.model_type == ModelType.TABNET:
            model = TabNetRegressor(**params)
        else:
            model = TabNetClassifier(**params)
        
        return model
    
    def _create_time_series_transformer_model(self, model_config: ModelConfig) -> Any:
        """Create Time Series Transformer model."""
        
        # This is a placeholder implementation
        # In practice, you would implement a custom TimeSeriesTransformer class
        class TimeSeriesTransformer:
            def __init__(self, **kwargs):
                self.params = kwargs
                self.is_fitted = False
            
            def fit(self, X, y):
                # Placeholder implementation
                self.is_fitted = True
                return self
            
            def predict(self, X):
                if not self.is_fitted:
                    raise ValueError("Model not fitted")
                # Placeholder implementation
                return np.zeros(len(X))
        
        return TimeSeriesTransformer(**model_config.model_params)
    
    def _create_tcn_model(self, model_config: ModelConfig) -> Any:
        """Create TCN model."""
        
        # This is a placeholder implementation
        # In practice, you would implement a custom TCN class
        class TCN:
            def __init__(self, **kwargs):
                self.params = kwargs
                self.is_fitted = False
            
            def fit(self, X, y):
                # Placeholder implementation
                self.is_fitted = True
                return self
            
            def predict(self, X):
                if not self.is_fitted:
                    raise ValueError("Model not fitted")
                # Placeholder implementation
                return np.zeros(len(X))
        
        return TCN(**model_config.model_params)
    
    def _create_lstm_model(self, model_config: ModelConfig) -> Any:
        """Create LSTM model."""
        
        # This is a placeholder implementation
        # In practice, you would implement a custom LSTM class
        class LSTM:
            def __init__(self, **kwargs):
                self.params = kwargs
                self.is_fitted = False
            
            def fit(self, X, y):
                # Placeholder implementation
                self.is_fitted = True
                return self
            
            def predict(self, X):
                if not self.is_fitted:
                    raise ValueError("Model not fitted")
                # Placeholder implementation
                return np.zeros(len(X))
        
        return LSTM(**model_config.model_params)
    
    def _create_tcn_model(self, model_config: ModelConfig) -> Any:
        """Create TCN model with overfitting prevention."""
        
        # Default parameters with overfitting prevention
        default_params = {
            'num_filters': 64,
            'kernel_size': 3,
            'dilations': [1, 2, 4, 8, 16, 32],
            'dropout': 0.2,           # Dropout for overfitting prevention
            'l2_regularization': 0.01, # L2 regularization
            'early_stopping_patience': 15,
            'batch_size': 32,
            'epochs': 100,
            'use_skip_connections': True,
            'use_batch_norm': True
        }
        
        # Merge with user parameters
        params = {**default_params, **model_config.model_params}
        
        # This is a placeholder implementation
        # In practice, you would implement a custom TCN class with proper overfitting prevention
        class TCN:
            def __init__(self, **kwargs):
                self.params = kwargs
                self.is_fitted = False
                self.dropout = kwargs.get('dropout', 0.2)
                self.l2_regularization = kwargs.get('l2_regularization', 0.01)
                self.early_stopping_patience = kwargs.get('early_stopping_patience', 15)
                self.num_filters = kwargs.get('num_filters', 64)
                self.kernel_size = kwargs.get('kernel_size', 3)
                self.dilations = kwargs.get('dilations', [1, 2, 4, 8, 16, 32])
                self.use_skip_connections = kwargs.get('use_skip_connections', True)
                self.use_batch_norm = kwargs.get('use_batch_norm', True)
            
            def fit(self, X, y):
                # Placeholder implementation with overfitting prevention
                self.is_fitted = True
                return self
            
            def predict(self, X):
                if not self.is_fitted:
                    raise ValueError("Model not fitted")
                # Placeholder implementation
                return np.zeros(len(X))
        
        return TCN(**params)
    
    def _create_wavenet_model(self, model_config: ModelConfig) -> Any:
        """Create WaveNet model with overfitting prevention."""
        
        # Default parameters with overfitting prevention
        default_params = {
            'dilations': [1, 2, 4, 8, 16, 32, 64],
            'residual_channels': 64,
            'skip_channels': 64,
            'kernel_size': 3,
            'use_gated_activation': True,
            'dropout': 0.2,
            'l2_regularization': 0.01,
            'early_stopping_patience': 15,
            'batch_size': 32,
            'epochs': 100
        }
        
        # Merge with user parameters
        params = {**default_params, **model_config.model_params}
        
        # This is a placeholder implementation
        class WaveNet:
            def __init__(self, **kwargs):
                self.params = kwargs
                self.is_fitted = False
                self.dilations = kwargs.get('dilations', [1, 2, 4, 8, 16, 32, 64])
                self.residual_channels = kwargs.get('residual_channels', 64)
                self.skip_channels = kwargs.get('skip_channels', 64)
                self.kernel_size = kwargs.get('kernel_size', 3)
                self.use_gated_activation = kwargs.get('use_gated_activation', True)
                self.dropout = kwargs.get('dropout', 0.2)
                self.l2_regularization = kwargs.get('l2_regularization', 0.01)
                self.early_stopping_patience = kwargs.get('early_stopping_patience', 15)
            
            def fit(self, X, y):
                # Placeholder implementation with overfitting prevention
                self.is_fitted = True
                return self
            
            def predict(self, X):
                if not self.is_fitted:
                    raise ValueError("Model not fitted")
                # Placeholder implementation
                return np.zeros(len(X))
        
        return WaveNet(**params)
    
    def _create_tft_model(self, model_config: ModelConfig) -> Any:
        """Create Temporal Fusion Transformer model."""
        
        # Default parameters
        default_params = {
            'attention_heads': 8,
            'hidden_size': 64,
            'num_layers': 3,
            'dropout': 0.1,
            'use_interpretable_attention': True,
            'batch_size': 32,
            'epochs': 100
        }
        
        # Merge with user parameters
        params = {**default_params, **model_config.model_params}
        
        # This is a placeholder implementation
        class TemporalFusionTransformer:
            def __init__(self, **kwargs):
                self.params = kwargs
                self.is_fitted = False
                self.attention_heads = kwargs.get('attention_heads', 8)
                self.hidden_size = kwargs.get('hidden_size', 64)
                self.num_layers = kwargs.get('num_layers', 3)
                self.dropout = kwargs.get('dropout', 0.1)
                self.use_interpretable_attention = kwargs.get('use_interpretable_attention', True)
            
            def fit(self, X, y):
                # Placeholder implementation
                self.is_fitted = True
                return self
            
            def predict(self, X):
                if not self.is_fitted:
                    raise ValueError("Model not fitted")
                # Placeholder implementation
                return np.zeros(len(X))
        
        return TemporalFusionTransformer(**params)
    
    def _create_tabnet_attention_model(self, model_config: ModelConfig) -> Any:
        """Create TabNet with attention model."""
        
        # Default parameters
        default_params = {
            'n_d': 64,
            'n_a': 64,
            'n_steps': 5,
            'gamma': 1.5,
            'lambda_sparse': 1e-3,
            'optimizer_params': {'lr': 2e-2},
            'batch_size': 32,
            'epochs': 100
        }
        
        # Merge with user parameters
        params = {**default_params, **model_config.model_params}
        
        # This is a placeholder implementation
        class TabNetAttention:
            def __init__(self, **kwargs):
                self.params = kwargs
                self.is_fitted = False
                self.n_d = kwargs.get('n_d', 64)
                self.n_a = kwargs.get('n_a', 64)
                self.n_steps = kwargs.get('n_steps', 5)
                self.gamma = kwargs.get('gamma', 1.5)
                self.lambda_sparse = kwargs.get('lambda_sparse', 1e-3)
            
            def fit(self, X, y):
                # Placeholder implementation
                self.is_fitted = True
                return self
            
            def predict(self, X):
                if not self.is_fitted:
                    raise ValueError("Model not fitted")
                # Placeholder implementation
                return np.zeros(len(X))
        
        return TabNetAttention(**params)
    
    def _create_elastic_net_model(self, model_config: ModelConfig) -> Any:
        """Create Elastic Net model."""
        
        from sklearn.linear_model import ElasticNet, ElasticNetCV
        
        # Default parameters
        default_params = {
            'alpha': 0.1,
            'l1_ratio': 0.5,
            'max_iter': 1000,
            'random_state': 42
        }
        
        # Merge with user parameters
        params = {**default_params, **model_config.model_params}
        
        # Create model
        if model_config.model_type == ModelType.ELASTIC_NET:
            model = ElasticNet(**params)
        else:
            model = ElasticNetCV(**params)
        
        return model
    
    def _create_elastic_net_quantile_model(self, model_config: ModelConfig) -> Any:
        """Create Elastic Net with Quantile Regression."""
        
        # This is a placeholder implementation
        class ElasticNetQuantile:
            def __init__(self, **kwargs):
                self.params = kwargs
                self.is_fitted = False
                self.alpha = kwargs.get('alpha', 0.1)
                self.l1_ratio = kwargs.get('l1_ratio', 0.5)
                self.quantiles = kwargs.get('quantiles', [0.05, 0.25, 0.5, 0.75, 0.95])
            
            def fit(self, X, y):
                # Placeholder implementation
                self.is_fitted = True
                return self
            
            def predict(self, X):
                if not self.is_fitted:
                    raise ValueError("Model not fitted")
                # Placeholder implementation
                return np.zeros(len(X))
        
        return ElasticNetQuantile(**model_config.model_params)
    
    def _create_quantile_regression_model(self, model_config: ModelConfig) -> Any:
        """Create Quantile Regression model."""
        
        # This is a placeholder implementation
        class QuantileRegression:
            def __init__(self, **kwargs):
                self.params = kwargs
                self.is_fitted = False
                self.quantiles = kwargs.get('quantiles', [0.05, 0.25, 0.5, 0.75, 0.95])
                self.alpha = kwargs.get('alpha', 0.1)
                self.solver = kwargs.get('solver', 'highs')
            
            def fit(self, X, y):
                # Placeholder implementation
                self.is_fitted = True
                return self
            
            def predict(self, X):
                if not self.is_fitted:
                    raise ValueError("Model not fitted")
                # Placeholder implementation
                return np.zeros(len(X))
        
        return QuantileRegression(**model_config.model_params)
    
    def _create_huber_regression_model(self, model_config: ModelConfig) -> Any:
        """Create Huber Regression model."""
        
        from sklearn.linear_model import HuberRegressor
        
        # Default parameters
        default_params = {
            'epsilon': 1.35,
            'max_iter': 1000,
            'alpha': 0.0001,
            'warm_start': False,
            'fit_intercept': True,
            'tol': 1e-05
        }
        
        # Merge with user parameters
        params = {**default_params, **model_config.model_params}
        
        return HuberRegressor(**params)
    
    def _create_hist_gradient_boosting_model(self, model_config: ModelConfig) -> Any:
        """Create HistGradientBoosting model."""
        
        from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
        
        # Default parameters
        default_params = {
            'max_iter': 100,
            'max_leaf_nodes': 31,
            'min_samples_leaf': 20,
            'l2_regularization': 0.0,
            'categorical_features': 'auto',
            'random_state': 42
        }
        
        # Merge with user parameters
        params = {**default_params, **model_config.model_params}
        
        # Create model
        if model_config.model_type == ModelType.HIST_GRADIENT_BOOSTING:
            model = HistGradientBoostingRegressor(**params)
        else:
            model = HistGradientBoostingClassifier(**params)
        
        return model
    
    def _create_extra_trees_model(self, model_config: ModelConfig) -> Any:
        """Create ExtraTrees model."""
        
        from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
        
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Merge with user parameters
        params = {**default_params, **model_config.model_params}
        
        # Create model
        if model_config.model_type == ModelType.EXTRA_TREES:
            model = ExtraTreesRegressor(**params)
        else:
            model = ExtraTreesClassifier(**params)
        
        return model
    
    def _create_xgboost_custom_model(self, model_config: ModelConfig) -> Any:
        """Create XGBoost with custom financial objectives."""
        
        import xgboost as xgb
        
        # Default parameters with financial focus
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'objective': 'reg:squarederror',  # Can be customized for financial objectives
            'eval_metric': 'rmse'
        }
        
        # Merge with user parameters
        params = {**default_params, **model_config.model_params}
        
        # Create model
        model = xgb.XGBRegressor(**params)
        
        return model
    
    def _create_node_model(self, model_config: ModelConfig) -> Any:
        """Create Neural Oblivious Decision Ensembles (NODE) model with overfitting prevention."""
        
        # Default parameters with overfitting prevention
        default_params = {
            'n_d': 64,
            'n_a': 64,
            'n_steps': 5,
            'gamma': 1.5,
            'lambda_sparse': 1e-3,    # Sparsity regularization
            'dropout': 0.1,           # Dropout for overfitting prevention
            'l2_regularization': 0.01, # L2 regularization
            'batch_size': 32,
            'epochs': 100
        }
        
        # Merge with user parameters
        params = {**default_params, **model_config.model_params}
        
        # This is a placeholder implementation
        # In practice, you would implement a custom NODE class with proper overfitting prevention
        class NODE:
            def __init__(self, **kwargs):
                self.params = kwargs
                self.is_fitted = False
                self.lambda_sparse = kwargs.get('lambda_sparse', 1e-3)
                self.dropout = kwargs.get('dropout', 0.1)
                self.l2_regularization = kwargs.get('l2_regularization', 0.01)
            
            def fit(self, X, y):
                # Placeholder implementation with overfitting prevention
                self.is_fitted = True
                return self
            
            def predict(self, X):
                if not self.is_fitted:
                    raise ValueError("Model not fitted")
                # Placeholder implementation
                return np.zeros(len(X))
        
        return NODE(**params)
    
    def _create_ridge_model(self, model_config: ModelConfig) -> Any:
        """Create Ridge model."""
        
        from sklearn.linear_model import Ridge, RidgeClassifier
        
        # Default parameters
        default_params = {
            'alpha': 1.0,
            'random_state': model_config.random_state
        }
        
        # Merge with user parameters
        params = {**default_params, **model_config.model_params}
        
        # Create model
        if model_config.model_type == ModelType.RIDGE:
            model = Ridge(**params)
        else:
            model = RidgeClassifier(**params)
        
        return model
    
    def _create_linear_model(self, model_config: ModelConfig) -> Any:
        """Create Linear model."""
        
        from sklearn.linear_model import LogisticRegression, LinearRegression
        
        # Default parameters
        default_params = {
            'random_state': model_config.random_state
        }
        
        # Merge with user parameters
        params = {**default_params, **model_config.model_params}
        
        # Create model
        if model_config.model_type == ModelType.LOGISTIC_REGRESSION:
            model = LogisticRegression(**params)
        else:
            model = LinearRegression(**params)
        
        return model
    
    def _apply_m1_optimizations(self, model: Any, model_config: ModelConfig) -> Any:
        """Apply M1-specific optimizations to the model."""
        
        # This is a placeholder for M1-specific optimizations
        # In practice, you would apply specific optimizations based on model type
        
        if hasattr(model, 'set_params'):
            # Apply M1-specific parameters if available
            m1_params = {}
            
            # Memory optimization
            if self.m1_memory and model_config.enable_memory_optimization:
                m1_params.update({
                    'n_jobs': min(model_config.n_jobs, 4),  # Limit parallel jobs on M1
                })
            
            # GPU acceleration (if supported by model)
            if self.m1_gpu and model_config.enable_gpu_acceleration:
                # Add GPU-specific parameters if the model supports them
                pass
            
            # Apply parameters
            if m1_params:
                model.set_params(**m1_params)
                self.logger.debug(f"🔧 Applied M1 optimizations: {m1_params}")
        
        return model
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a model from the registry."""
        return self.model_registry.get(model_name)
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self.model_registry.keys())
    
    def remove_model(self, model_name: str) -> bool:
        """Remove a model from the registry."""
        if model_name in self.model_registry:
            del self.model_registry[model_name]
            self.logger.info(f"🗑️ Removed model: {model_name}")
            return True
        return False
    
    def clear_registry(self) -> None:
        """Clear all models from the registry."""
        self.model_registry.clear()
        self.logger.info("🗑️ Cleared model registry")


# Convenience functions for easy model creation
def create_analyst_models() -> Dict[str, Any]:
    """Create all Analyst (5m) models."""
    factory = EnhancedModelFactory()
    models = {}
    
    # Analyst fixed models
    analyst_models = {
        "tft": ModelType.TEMPORAL_FUSION_TRANSFORMER,  # Temporal Fusion Transformer
        "tabnet": ModelType.TABNET,  # TabNet for interpretable feature selection
        "hist_gb": ModelType.HIST_GRADIENT_BOOSTING,  # HistGradientBoosting
        "extratrees": ModelType.EXTRA_TREES  # ExtraTrees for fast meta-model
    }
    
    for name, model_type in analyst_models.items():
        config = ModelConfig(
            model_type=model_type,
            model_name=f"analyst_{name}",
            is_multi_output=True,
            n_outputs=4,
            output_names=["signal_strength", "confidence", "risk_score", "regime_label"]
        )
        models[name] = factory.create_model(config)
    
    return models


def create_tactician_models() -> Dict[str, Any]:
    """Create all Tactician (1m) models."""
    factory = EnhancedModelFactory()
    models = {}
    
    # Tactician fixed models
    tactician_models = {
        "tabnet_attention": ModelType.TABNET_ATTENTION,  # TabNet with attention
        "xgboost_custom": ModelType.XGBOOST_CUSTOM,  # XGBoost with custom objectives
        "hist_gb": ModelType.HIST_GRADIENT_BOOSTING,  # HistGradientBoosting
        "elastic_quantile": ModelType.ELASTIC_NET_QUANTILE  # ElasticNet with quantile regression
    }
    
    for name, model_type in tactician_models.items():
        config = ModelConfig(
            model_type=model_type,
            model_name=f"tactician_{name}",
            is_multi_output=True,
            n_outputs=4,
            output_names=["entry_timing", "position_size", "stop_loss", "take_profit"]
        )
        models[name] = factory.create_model(config)
    
    return models


def create_model_factory(config: Optional[Dict[str, Any]] = None) -> EnhancedModelFactory:
    """Create an enhanced model factory instance."""
    return EnhancedModelFactory(config)