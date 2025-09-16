"""
Unified Model Factory

Provides centralized model creation with standardized configurations
across all HMM training components.
"""

from typing import Any, Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')


class UnifiedModelFactory:
    """Unified factory for creating model instances with standardized configuration."""
    
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
                'max_iter': 1000,
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
                'max_iter': 1000,
                'random_state': 42,
                'class_weight': 'balanced'
            }
        }
    }
    
    @classmethod
    def create_model(cls, model_type: str, **custom_params) -> Any:
        """
        Create model instance with standardized configuration.
        
        Args:
            model_type: Type of model to create
            **custom_params: Additional model parameters
            
        Returns:
            Model instance
            
        Raises:
            ValueError: If model type is not supported
            ImportError: If required model library is not available
        """
        if model_type not in cls._model_configs:
            available_types = list(cls._model_configs.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available types: {available_types}")
        
        config = cls._model_configs[model_type]
        
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
    def get_available_models(cls) -> List[str]:
        """Get list of available model types."""
        return list(cls._model_configs.keys())
    
    @classmethod
    def get_model_config(cls, model_type: str) -> Dict[str, Any]:
        """Get configuration for a specific model type."""
        if model_type not in cls._model_configs:
            raise ValueError(f"Unknown model type: {model_type}")
        return cls._model_configs[model_type].copy()
    
    @classmethod
    def add_model_config(cls, model_type: str, class_path: str, default_params: Dict[str, Any]) -> None:
        """
        Add a new model configuration.
        
        Args:
            model_type: Name of the model type
            class_path: Full path to the model class
            default_params: Default parameters for the model
        """
        cls._model_configs[model_type] = {
            'class': class_path,
            'default_params': default_params
        }
    
    @classmethod
    def validate_model_type(cls, model_type: str) -> bool:
        """Validate if a model type is supported."""
        return model_type in cls._model_configs