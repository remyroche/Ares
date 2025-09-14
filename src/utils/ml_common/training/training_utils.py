"""
Training Utilities

Common training logic patterns shared across all training modules.
Uses existing hardware optimization utilities for M1 optimization.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Use existing utilities
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
from src.utils.common_operations import get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer
from src.utils.logger import system_logger

from src.utils.ml_common.models import EnhancedModelFactory, ModelType, ModelConfig
from src.utils.ml_common.optimization import HierarchicalHPO, HierarchicalHPOConfig, HPOPhaseConfig
from src.utils.ml_common.optimization.overfitting_prevention import OverfittingPrevention, OverfittingPreventionConfig
from src.utils.ml_common.evaluation.evaluation_utils import EvaluationUtils

logger = system_logger.getChild('TrainingUtils')


class TrainingUtils:
    """Common training utilities."""
    
    def __init__(self, config: Any):
        """
        Initialize training utilities with hardware optimization.

        Args:
            config: Training configuration object or dict
        """
        # Handle case where config is a dict instead of proper config object
        if isinstance(config, dict):
            # Convert dict to BaseTrainingConfig
            from .config.base_training_config import BaseTrainingConfig
            default_config = BaseTrainingConfig()
            config_dict = {**default_config.__dict__, **config}
            config = BaseTrainingConfig(**config_dict)

        self.config = config
        self.model_factory = EnhancedModelFactory()
        self.overfitting_prevention = OverfittingPrevention(
            OverfittingPreventionConfig() if config.enable_overfitting_prevention else None
        )
        
        # Initialize hardware optimizers
        self.gpu_manager = get_m1_gpu_manager()
        self.memory_optimizer = get_m1_memory_optimizer()
        self.cpu_optimizer = get_m1_cpu_optimizer()
        
        if self.gpu_manager:
            logger.info("🚀 M1 GPU optimization enabled")
        if self.memory_optimizer:
            logger.info("🧠 M1 memory optimization enabled")
        if self.cpu_optimizer:
            logger.info("⚡ M1 CPU optimization enabled")
    
    def create_model(
        self, 
        model_type: str, 
        model_name: str,
        model_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Create a model instance using the model factory.
        
        Args:
            model_type: Type of model to create
            model_name: Name for the model
            model_params: Optional model parameters
            
        Returns:
            Created model instance
        """
        if model_params is None:
            model_params = {}
        
        # Map string model type to ModelType enum
        model_type_enum = self._map_string_to_model_type(model_type)

        model_config = ModelConfig(
            model_type=model_type_enum,
            model_name=model_name,
            model_params=model_params
        )
        
        model = self.model_factory.create_model(model_config)
        
        # Apply overfitting prevention if enabled
        if self.config.enable_overfitting_prevention:
            model = self.overfitting_prevention.apply_regularization(model, model_type)
        
        return model

    def _map_string_to_model_type(self, model_type_str: str) -> 'ModelType':
        """
        Map string model type to ModelType enum.

        Args:
            model_type_str: String representation of model type

        Returns:
            ModelType enum value

        Raises:
            ValueError: If model type string cannot be mapped
        """
        from src.utils.ml_common.models.model_factory import ModelType

        # Create mapping from string to enum value
        string_to_enum_mapping = {}

        # Build the mapping by checking the .value attribute of each enum
        for enum_member in ModelType:
            string_to_enum_mapping[enum_member.value] = enum_member

        # Handle common variations and aliases
        aliases = {
            'XGBClassifier': 'XGBClassifier',
            'XGBRegressor': 'XGBRegressor',
            'LGBMClassifier': 'LGBMClassifier',
            'LGBMRegressor': 'LGBMRegressor',
            'CatBoostClassifier': 'CatBoostClassifier',
            'CatBoostRegressor': 'CatBoostRegressor',
            'RandomForestClassifier': 'RandomForestClassifier',
            'RandomForestRegressor': 'RandomForestRegressor',
            'ExtraTreesClassifier': 'ExtraTreesClassifier',
            'ExtraTreesRegressor': 'ExtraTreesRegressor',
            'HistGradientBoostingClassifier': 'HistGradientBoostingClassifier',
            'HistGradientBoostingRegressor': 'HistGradientBoostingRegressor',
            'RidgeClassifier': 'RidgeClassifier',
            'Ridge': 'Ridge',
            'LogisticRegression': 'LogisticRegression',
            'LinearRegression': 'LinearRegression',
            'TabNetClassifier': 'TabNetClassifier',
            'TabNetRegressor': 'TabNetRegressor',
            'TCN': 'TCN',
            'LSTM': 'LSTM',
            'WaveNet': 'WaveNet',
            'NODE': 'NODE',
            'NODEClassifier': 'NODEClassifier',
            'VotingClassifier': 'VotingClassifier',
            'VotingRegressor': 'VotingRegressor',
            'StackingClassifier': 'StackingClassifier',
            'StackingRegressor': 'StackingRegressor',
            'BaggingClassifier': 'BaggingClassifier',
            'BaggingRegressor': 'BaggingRegressor',
            'AdaBoostClassifier': 'AdaBoostClassifier',
            'AdaBoostRegressor': 'AdaBoostRegressor'
        }

        # Update mapping with aliases
        for alias, target in aliases.items():
            if target in string_to_enum_mapping:
                string_to_enum_mapping[alias] = string_to_enum_mapping[target]

        # Try direct mapping first
        if model_type_str in string_to_enum_mapping:
            return string_to_enum_mapping[model_type_str]

        # Try case-insensitive matching
        model_type_upper = model_type_str.upper()
        for enum_value, enum_member in string_to_enum_mapping.items():
            if enum_value.upper() == model_type_upper:
                return enum_member

        # If no match found, raise error with helpful message
        available_types = list(string_to_enum_mapping.keys())[:10]  # Show first 10
        raise ValueError(f"Unknown model type: '{model_type_str}'. Available types: {available_types}...")

    def optimize_model_with_hpo(
        self, 
        model_type: str, 
        X: np.ndarray, 
        y: np.ndarray,
        search_space: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Optimize model using HPO.
        
        Args:
            model_type: Type of model to optimize
            X: Input features
            y: Target values
            search_space: HPO search space
            model_name: Optional model name
            
        Returns:
            Dictionary containing optimization results
        """
        if model_name is None:
            model_name = f"{model_type.lower()}_optimized"
        
        logger.debug(f"🔄 Optimizing {model_type}...")
        
        # Create base model
        base_model = self.create_model(model_type, model_name)
        
        # Get search space
        if search_space is None:
            search_space = getattr(self.config, 'hpo_search_spaces', {}).get(model_type, {})
        
        # Create HPO configuration
        hpo_config = HierarchicalHPOConfig(
            phase1_config=HPOPhaseConfig(
                phase_name=f"{model_type}_optimization",
                models={model_type: base_model},
                search_spaces={model_type: search_space},
                n_trials=self.config.hpo_n_trials,
                timeout_seconds=self.config.hpo_timeout_seconds,
                cv_folds=self.config.hpo_cv_folds
            ),
            phase2_config=HPOPhaseConfig(
                phase_name="meta_models",
                models={},
                search_spaces={},
                n_trials=0
            )
        )
        
        # Perform HPO
        hpo = HierarchicalHPO(hpo_config)
        hpo_results = hpo.optimize_ensemble(X, y)
        
        # Extract optimized model
        optimized_model = hpo_results['base_models'][model_type]
        
        return {
            'model': optimized_model,
            'hpo_results': hpo_results,
            'model_type': model_type,
            'optimization_time': hpo_results.get('optimization_time', 0)
        }
    
    def train_single_model(
        self, 
        model_type: str, 
        X: np.ndarray, 
        y: np.ndarray,
        model_params: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train single model without HPO.
        
        Args:
            model_type: Type of model to train
            X: Input features
            y: Target values
            model_params: Optional model parameters
            model_name: Optional model name
            
        Returns:
            Dictionary containing training results
        """
        if model_name is None:
            model_name = f"{model_type.lower()}_trained"
        
        logger.debug(f"🔄 Training {model_type} (no HPO)...")
        
        # Create model
        model = self.create_model(model_type, model_name, model_params)
        
        # Train model
        start_time = time.time()
        model.fit(X, y)
        training_time = time.time() - start_time
        
        return {
            'model': model,
            'model_type': model_type,
            'training_time': training_time
        }
    
    def train_models(
        self, 
        model_types: List[str], 
        X: np.ndarray, 
        y: np.ndarray,
        enable_hpo: bool = True,
        search_spaces: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Train multiple models.
        
        Args:
            model_types: List of model types to train
            X: Input features
            y: Target values
            enable_hpo: Whether to use HPO
            search_spaces: HPO search spaces for each model type
            
        Returns:
            Dictionary containing training results
        """
        model_results = {}
        training_metadata = {}
        
        for model_type in model_types:
            logger.info(f"🔄 Training {model_type}...")
            
            # Get search space for this model type
            search_space = None
            if search_spaces and model_type in search_spaces:
                search_space = search_spaces[model_type]
            
            # Train model
            if enable_hpo and search_space:
                model_result = self.optimize_model_with_hpo(
                    model_type, X, y, search_space
                )
            else:
                model_result = self.train_single_model(model_type, X, y)
            
            model_results[model_type] = model_result
            
            # Store training metadata
            training_metadata[model_type] = {
                'model_type': model_type,
                'training_time': model_result.get('training_time', 0),
                'optimization_time': model_result.get('optimization_time', 0),
                'samples': len(X),
                'features': X.shape[1]
            }
            
            logger.info(f"✅ {model_type} trained")
        
        return {
            'models': model_results,
            'metadata': training_metadata
        }
    
    def prepare_training_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        stratify: Optional[np.ndarray] = None,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data with train/validation/test splits.
        
        Args:
            X: Input features
            y: Target values
            test_size: Proportion of data for testing
            validation_size: Proportion of data for validation
            stratify: Array for stratification
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=stratify
        )
        
        # Second split: separate train and validation
        val_size = validation_size / (1 - test_size)  # Adjust for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state,
            stratify=stratify[y_temp] if stratify is not None else None
        )
        
        logger.info(f"📊 Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(
        self, 
        X_train: np.ndarray, 
        X_val: Optional[np.ndarray] = None, 
        X_test: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], StandardScaler]:
        """
        Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_val: Optional validation features
            X_test: Optional test features
            
        Returns:
            Tuple of scaled features and fitted scaler
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        X_val_scaled = None
        if X_val is not None:
            X_val_scaled = scaler.transform(X_val)
        
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
    
    def evaluate_models(
        self, 
        models: Dict[str, Any], 
        X: np.ndarray, 
        y: np.ndarray,
        is_classification: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate multiple models.
        
        Args:
            models: Dictionary of trained models
            X: Input features
            y: True target values
            is_classification: Whether this is a classification task
            
        Returns:
            Dictionary containing evaluation results for each model
        """
        evaluation_results = {}
        
        for model_name, model_result in models.items():
            model = model_result['model']
            
            try:
                metrics = EvaluationUtils.evaluate_model_performance(
                    model, X, y, 
                    metrics=self.config.evaluation_metrics,
                    is_classification=is_classification
                )
                evaluation_results[model_name] = metrics
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to evaluate {model_name}: {e}")
                evaluation_results[model_name] = {'error': str(e)}
        
        return evaluation_results
    
    def get_model_params(self, model_type: str) -> Dict[str, Any]:
        """
        Get default parameters for model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary of default parameters
        """
        default_params = {
            'TCN': {
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.2,
                'recurrent_dropout': 0.1,
                'l2_regularization': 0.01
            },
            'CATBOOST': {
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3.0,
                'subsample': 0.8,
                'colsample_bylevel': 0.8
            },
            'LIGHTGBM': {
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'max_depth': 6,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'RANDOM_FOREST': {
                'n_estimators': 500,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True
            },
            'NODE': {
                'n_d': 64,
                'n_a': 64,
                'n_steps': 5,
                'gamma': 1.5,
                'lambda_sparse': 1e-3,
                'dropout': 0.1,
                'l2_regularization': 0.01
            },
            'RIDGE': {
                'alpha': 1.0,
                'solver': 'auto',
                'random_state': 42
            }
        }
        
        return default_params.get(model_type.upper(), {})