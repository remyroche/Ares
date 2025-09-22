"""
Unified Model Factory

Provides centralized model creation with standardized configurations
across all HMM training components. Thread-safe implementation with
enhanced regularization for small datasets.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import warnings
import threading
from copy import deepcopy
import numpy as np

warnings.filterwarnings('ignore')


class UnifiedModelFactory:
    """Thread-safe unified factory for creating model instances with standardized configuration."""

    # Enhanced regularization thresholds
    SMALL_DATASET_THRESHOLD = 100  # samples per regime
    MEDIUM_DATASET_THRESHOLD = 500  # samples per regime

    # Regularization scaling factors for small datasets
    SMALL_DATASET_REG_MULTIPLIER = 2.0
    MEDIUM_DATASET_REG_MULTIPLIER = 1.5

    @classmethod
    def _calculate_adaptive_regularization(cls, regime_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate adaptive regularization parameters based on dataset size.

        Args:
            regime_labels: Array of regime labels to analyze sample distribution

        Returns:
            Dictionary with adaptive regularization parameters
        """
        base_reg_alpha = 0.1
        base_reg_lambda = 0.1

        if regime_labels is None:
            return {
                'reg_alpha': base_reg_alpha,
                'reg_lambda': base_reg_lambda,
                'dataset_size': 'unknown'
            }

        # Calculate samples per regime
        unique_regimes, regime_counts = np.unique(regime_labels, return_counts=True)
        min_samples_per_regime = np.min(regime_counts)
        avg_samples_per_regime = np.mean(regime_counts)

        # Determine regularization scaling based on dataset size
        if min_samples_per_regime < cls.SMALL_DATASET_THRESHOLD:
            # Small dataset: increase regularization significantly
            reg_multiplier = cls.SMALL_DATASET_REG_MULTIPLIER
            dataset_size = 'small'
        elif min_samples_per_regime < cls.MEDIUM_DATASET_THRESHOLD:
            # Medium dataset: moderate increase in regularization
            reg_multiplier = cls.MEDIUM_DATASET_REG_MULTIPLIER
            dataset_size = 'medium'
        else:
            # Large dataset: use base regularization
            reg_multiplier = 1.0
            dataset_size = 'large'

        adaptive_reg_alpha = base_reg_alpha * reg_multiplier
        adaptive_reg_lambda = base_reg_lambda * reg_multiplier

        return {
            'reg_alpha': adaptive_reg_alpha,
            'reg_lambda': adaptive_reg_lambda,
            'dataset_size': dataset_size,
            'min_samples_per_regime': int(min_samples_per_regime),
            'avg_samples_per_regime': float(avg_samples_per_regime),
            'reg_multiplier': reg_multiplier
        }

    _model_configs = {
        'lightgbm': {
            'class': 'lightgbm.LGBMClassifier',
            'default_params': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42,
                'verbosity': -1,
                'objective': 'multiclass',
                'num_class': 3
            }
        },
        'elastic_net_lr': {
            'class': 'sklearn.linear_model.LogisticRegression',
            'default_params': {
                'C': 1.0,
                'max_iter': 2000,
                'random_state': 42,
                'class_weight': 'balanced',
                'penalty': 'elasticnet',
                'l1_ratio': 0.5,
                'solver': 'saga'
            }
        },
        'elastic_net_cv': {
            'class': 'sklearn.linear_model.ElasticNetCV',
            'default_params': {
                'l1_ratio': [0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
                'cv': 5,
                'random_state': 42,
                'max_iter': 2000,
                'n_jobs': -1
            }
        },
        'xgboost': {
            'class': 'xgboost.XGBClassifier',
            'default_params': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42,
                'verbosity': 0,
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss'
            }
        },
        'random_forest': {
            'class': 'sklearn.ensemble.RandomForestClassifier',
            'default_params': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            }
        },
        'logistic_regression': {
            'class': 'sklearn.linear_model.LogisticRegression',
            'default_params': {
                'C': 1.0,
                'max_iter': 2000,
                'random_state': 42,
                'class_weight': 'balanced'
            }
        }
    }
    
    _lock = threading.RLock()  # Class-level lock for thread safety
    
    @classmethod
    def create_model(cls, model_type: str, **custom_params) -> Any:
        """
        Thread-safe create model instance with standardized configuration.
        
        Args:
            model_type: Type of model to create
            **custom_params: Additional model parameters
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model type is not supported
            ImportError: If required model library is not available
        """
        with cls._lock:
            if model_type not in cls._model_configs:
                available_types = list(cls._model_configs.keys())
                raise ValueError(f"Unknown model type: {model_type}. Available types: {available_types}")
            
            # Deep copy to avoid modifying the original config
            config = deepcopy(cls._model_configs[model_type])
        
        try:
            # Import the class dynamically
            class_path = config['class']
            module_name, class_name = class_path.rsplit('.', 1)
            module = __import__(module_name, fromlist=[class_name])
            model_class = getattr(module, class_name)
            
            # Merge default and custom parameters
            params = {**config['default_params'], **custom_params}
            return model_class(**params)

        except ImportError as e:
            raise ImportError(f"Failed to import {model_type} model: {e}")
        except Exception as e:
            raise ValueError(f"Failed to create {model_type} model: {e}")

    @classmethod
    def create_model_with_adaptive_regularization(
        cls,
        model_type: str,
        regime_labels: Optional[np.ndarray] = None,
        **custom_params
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Create model instance with adaptive regularization based on dataset size.

        Args:
            model_type: Type of model to create
            regime_labels: Array of regime labels to analyze sample distribution
            **custom_params: Additional model parameters

        Returns:
            Tuple of (model_instance, regularization_info_dict)
        """
        # Calculate adaptive regularization parameters
        reg_info = cls._calculate_adaptive_regularization(regime_labels)

        # Add adaptive regularization to custom parameters
        adaptive_params = {
            'reg_alpha': reg_info['reg_alpha'],
            'reg_lambda': reg_info['reg_lambda'],
            **custom_params
        }

        # Create model with adaptive parameters
        model = cls.create_model(model_type, **adaptive_params)

        return model, reg_info
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Thread-safe get list of available model types."""
        with cls._lock:
            return list(cls._model_configs.keys())
    
    @classmethod
    def get_model_config(cls, model_type: str) -> Dict[str, Any]:
        """Thread-safe get configuration for a specific model type."""
        with cls._lock:
            if model_type not in cls._model_configs:
                raise ValueError(f"Unknown model type: {model_type}")
            return deepcopy(cls._model_configs[model_type])
    
    @classmethod
    def add_model_config(cls, model_type: str, class_path: str, default_params: Dict[str, Any]) -> None:
        """
        Thread-safe add a new model configuration.
        
        Args:
            model_type: Name of the model type
            class_path: Full path to the model class
            default_params: Default parameters for the model
        """
        with cls._lock:
            cls._model_configs[model_type] = {
                'class': class_path,
                'default_params': deepcopy(default_params)
            }
    
    @classmethod
    def validate_model_type(cls, model_type: str) -> bool:
        """Thread-safe validate if a model type is supported."""
        with cls._lock:
            return model_type in cls._model_configs
    
    @classmethod
    def remove_model_config(cls, model_type: str) -> bool:
        """
        Thread-safe remove a model configuration.
        
        Args:
            model_type: Name of the model type to remove
            
        Returns:
            True if removed, False if not found
        """
        with cls._lock:
            if model_type in cls._model_configs:
                del cls._model_configs[model_type]
                return True
            return False
    
    @classmethod
    def update_model_config(cls, model_type: str, **updates) -> bool:
        """
        Thread-safe update existing model configuration.
        
        Args:
            model_type: Name of the model type to update
            **updates: Configuration updates
            
        Returns:
            True if updated, False if not found
        """
        with cls._lock:
            if model_type not in cls._model_configs:
                return False
            
            config = cls._model_configs[model_type]
            if 'default_params' in updates:
                config['default_params'].update(updates['default_params'])
            if 'class' in updates:
                config['class'] = updates['class']
            
            return True