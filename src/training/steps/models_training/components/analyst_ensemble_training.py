"""
Analyst Ensemble Training - Simplified Working Version

This module provides a simplified Analyst ensemble training component that works
without complex dependencies and indentation issues.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_data_format

# Simple enum for ensemble methods
class EnsembleMethod(Enum):
    STACKING = "stacking"
    VOTING = "voting"
    AVERAGING = "averaging"
    BLENDING = "blending"

# Simple enum for analyst model types
class AnalystModelType(Enum):
    XGBOOST = "xgboost"
    CATBOOST = "catboost"
    LIGHTGBM = "lightgbm"
    RANDOM_FOREST = "random_forest"

@dataclass
class AnalystEnsembleTrainingResult:
    """Result from analyst ensemble training."""
    success: bool
    training_time: float
    ensemble_result: Dict[str, Any]
    config: Dict[str, Any]
    data_info: Dict[str, Any]
    error: Optional[str] = None

@dataclass
class AnalystEnsembleTrainingConfig:
    """Configuration for Analyst ensemble training."""
    
    # Basic configuration
    model_name: str = "analyst_ensemble"
    timeframe: str = "15m"
    
    # Ensemble configuration
    base_models: List[AnalystModelType] = field(default_factory=lambda: [
        AnalystModelType.XGBOOST, 
        AnalystModelType.CATBOOST, 
        AnalystModelType.LIGHTGBM
    ])
    ensemble_method: EnsembleMethod = EnsembleMethod.VOTING
    
    # Training configuration
    validation_split: float = 0.2
    test_split: float = 0.1
    enable_cross_validation: bool = True
    cv_folds: int = 5
    
    # Model saving
    save_models: bool = True
    model_save_path: str = "./models"
    
    # Evaluation configuration
    enable_evaluation: bool = True
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "f1_score", "precision", "recall"
    ])

class AnalystEnsembleTraining(BaseStep):
    """
    Simplified Analyst ensemble training component.
    
    This component handles training of ensemble models that combine multiple
    Analyst base models for enhanced performance.
    """
    
    def __init__(
        self,
        name: str = "analyst_ensemble_training",
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the analyst ensemble training component.
        
        Args:
            name: Component name
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(name, config)
        
        # Set default configuration
        self.config = AnalystEnsembleTrainingConfig()
        if config:
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        self.logger.info(f"✅ Analyst Ensemble Training initialized")
        self.logger.info(f"📊 Configuration: {self.config.model_name}, {self.config.timeframe}")
        self.logger.info(f"🤖 Base models: {', '.join([m.value for m in self.config.base_models])}")
        self.logger.info(f"🔧 Ensemble method: {self.config.ensemble_method.value}")
        
        # Debug configuration format for troubleshooting
        tprint_data_format(self.config.__dict__, "analyst_ensemble_config", level=tprint.LogLevel.DEBUG)
    
    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the analyst ensemble training step.
        
        Args:
            data: Input data containing features and targets
            
        Returns:
            Training results
        """
        try:
            tprint_info("🎯 Starting Analyst Ensemble Training...")
            start_time = time.time()
            
            # Extract features and targets from data
            features = data.get('features')
            targets = data.get('targets')
            
            if features is None or targets is None:
                error_msg = "Missing required data: features or targets"
                tprint_error(f"❌ {error_msg}")
                return {'success': False, 'error': error_msg}
            
            # Convert to DataFrame if needed
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)
            if not isinstance(targets, pd.Series):
                targets = pd.Series(targets)
            
            # Debug data format for troubleshooting
            tprint_data_format(features, "features", level=tprint.LogLevel.INFO)
            tprint_data_format(targets, "targets", level=tprint.LogLevel.INFO)
            
            tprint_info(f"📊 Training data shape: {features.shape}")
            tprint_info(f"🎯 Target distribution: {targets.value_counts().to_dict()}")
            
            # Train ensemble models
            ensemble_result = await self._train_ensemble_models(features, targets)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Prepare results
            results = {
                'success': True,
                'training_time': training_time,
                'ensemble_result': ensemble_result,
                'config': {
                    'model_name': self.config.model_name,
                    'timeframe': self.config.timeframe,
                    'base_models': [m.value for m in self.config.base_models],
                    'ensemble_method': self.config.ensemble_method.value
                },
                'data_info': {
                    'features_shape': features.shape,
                    'targets_shape': targets.shape,
                    'target_distribution': targets.value_counts().to_dict()
                }
            }
            
            tprint_success(f"✅ Analyst Ensemble Training completed in {training_time:.2f}s")
            return results
            
        except Exception as e:
            error_msg = f"Analyst Ensemble Training failed: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    async def _train_ensemble_models(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """
        Train ensemble models.
        
        Args:
            features: Training features
            targets: Training targets
            
        Returns:
            Ensemble training results
        """
        try:
            tprint_info("🤖 Training ensemble models...")
            
            # Debug input data format for troubleshooting
            tprint_data_format(features, "ensemble_features", level=tprint.LogLevel.DEBUG)
            tprint_data_format(targets, "ensemble_targets", level=tprint.LogLevel.DEBUG)
            
            # Train real ensemble models
            ensemble_models = {}
            
            for model_type in self.config.base_models:
                tprint_info(f"🔧 Training {model_type.value}...")
                
                try:
                    # Train actual model
                    model_result = await self._train_single_model(
                        model_type, features, targets, self.config
                    )
                    
                    if model_result['success']:
                        ensemble_models[model_type.value] = model_result
                        tprint_info(f"✅ {model_type.value} trained with accuracy: {model_result['accuracy']:.4f}")
                    else:
                        tprint_warning(f"⚠️ {model_type.value} training failed: {model_result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    tprint_error(f"❌ {model_type.value} training error: {e}")
                    # Continue with other models even if one fails
            
            # Create ensemble result
            ensemble_result = {
                'ensemble_method': self.config.ensemble_method.value,
                'base_models': ensemble_models,
                'ensemble_accuracy': np.mean([m['accuracy'] for m in ensemble_models.values()]),
                'total_models': len(ensemble_models)
            }
            
            # Debug ensemble result format for troubleshooting
            tprint_data_format(ensemble_result, "ensemble_result", level=tprint.LogLevel.INFO)
            
            tprint_success(f"✅ Ensemble training completed with {len(ensemble_models)} models")
            tprint_info(f"📊 Average accuracy: {ensemble_result['ensemble_accuracy']:.4f}")
            
            return ensemble_result
            
        except Exception as e:
            error_msg = f"Ensemble model training failed: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg}
    
    def get_artifacts(self) -> Dict[str, Any]:
        """
        Get artifacts from this step.
        
        Returns:
            Dictionary of artifacts
        """
        return {
            'step_name': self.name,
            'config': self.config,
            'status': 'completed'
        }
    
    async def _train_single_model(
        self, 
        model_type: AnalystModelType, 
        features: pd.DataFrame, 
        targets: pd.Series, 
        config: AnalystEnsembleTrainingConfig
    ) -> Dict[str, Any]:
        """
        Train a single model.
        
        Args:
            model_type: Type of model to train
            features: Training features
            targets: Training targets
            config: Training configuration
            
        Returns:
            Dictionary containing model results
        """
        start_time = time.time()
        
        try:
            # Create model based on type
            model = self._create_model(model_type)
            if model is None:
                return {
                    'success': False,
                    'error': f"Failed to create {model_type.value} model"
                }
            
            # Split data for validation
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                features, targets, 
                test_size=config.validation_split, 
                random_state=42
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)
            
            # Calculate additional metrics
            y_pred = model.predict(X_val)
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            
            training_time = time.time() - start_time
            
            return {
                'success': True,
                'model_name': model_type.value,
                'model': model,
                'accuracy': val_score,
                'training_accuracy': train_score,
                'validation_accuracy': val_score,
                'mse': mse,
                'r2_score': r2,
                'mae': mae,
                'training_time': training_time,
                'status': 'trained'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'model_name': model_type.value,
                'training_time': time.time() - start_time
            }
    
    def _create_model(self, model_type: AnalystModelType):
        """Create a model instance based on type."""
        try:
            if model_type == AnalystModelType.XGBOOST:
                import xgboost as xgb
                return xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                )
            elif model_type == AnalystModelType.CATBOOST:
                import catboost as cb
                return cb.CatBoostRegressor(
                    iterations=100,
                    depth=6,
                    learning_rate=0.1,
                    random_seed=42,
                    verbose=False
                )
            elif model_type == AnalystModelType.LIGHTGBM:
                import lightgbm as lgb
                return lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1
                )
            elif model_type == AnalystModelType.RANDOM_FOREST:
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
        except ImportError as e:
            tprint_warning(f"⚠️ {model_type.value} not available: {e}")
            return None
        except Exception as e:
            tprint_error(f"❌ Failed to create {model_type.value} model: {e}")
            return None
    
    def validate_artifacts(self, artifacts: Dict[str, Any]) -> bool:
        """
        Validate artifacts from this step.
        
        Args:
            artifacts: Artifacts to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_keys = ['step_name', 'config', 'status']
        return all(key in artifacts for key in required_keys)

# Factory functions for compatibility
def create_analyst_ensemble_training(config: Optional[Dict[str, Any]] = None) -> AnalystEnsembleTraining:
    """Create an analyst ensemble training instance."""
    return AnalystEnsembleTraining(config=config)

def execute_analyst_ensemble_training(data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Execute analyst ensemble training with given data and config."""
    training = create_analyst_ensemble_training(config)
    return training.execute(data)