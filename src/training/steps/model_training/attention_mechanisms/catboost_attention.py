"""
CatBoost with Attention Mechanisms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass

from .tree_attention import TreeAttentionMechanism, AttentionConfig, AttentionType

try:
    from catboost import CatBoostRegressor, CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from src.utils.logger import system_logger


@dataclass
class CatBoostAttentionConfig(AttentionConfig):
    """Configuration for CatBoost with attention."""
    
    # CatBoost specific parameters
    catboost_params: Dict[str, Any] = None
    
    # Attention integration
    use_attention_for_features: bool = True
    use_attention_for_predictions: bool = True
    attention_weight: float = 0.5  # Weight for attention vs tree predictions
    
    def __post_init__(self):
        super().__post_init__()
        if self.catboost_params is None:
            self.catboost_params = {
                'iterations': 1000,
                'learning_rate': 0.1,
                'depth': 6,
                'l2_leaf_reg': 3,
                'random_seed': 42,
                'verbose': False
            }


class CatBoostAttentionWrapper(TreeAttentionMechanism):
    """CatBoost model with attention mechanisms."""
    
    def __init__(self, config: CatBoostAttentionConfig, task_type: str = 'regression'):
        """Initialize CatBoost with attention."""
        super().__init__(config)
        
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not available. Install with: pip install catboost")
        
        self.task_type = task_type
        self.catboost_model = None
        self.attention_model = None
        self.is_fitted = False
        
        # Initialize CatBoost model
        if task_type == 'regression':
            self.catboost_model = CatBoostRegressor(**config.catboost_params)
        else:
            self.catboost_model = CatBoostClassifier(**config.catboost_params)
    
    def fit(self, X: np.ndarray, y: np.ndarray, tree_predictions: Optional[np.ndarray] = None) -> 'CatBoostAttentionWrapper':
        """Fit CatBoost model with attention mechanism."""
        self.logger.info("Fitting CatBoost with attention mechanism")
        
        # Preprocess features
        X_processed = self._preprocess_features(X)
        
        # Fit CatBoost model
        self.logger.info("Training CatBoost model...")
        self.catboost_model.fit(X_processed, y)
        
        # Get tree predictions
        if tree_predictions is None:
            tree_predictions = self.catboost_model.predict(X_processed)
        
        # Initialize and train attention mechanism
        if self.config.use_attention_for_predictions:
            self.logger.info("Training attention mechanism...")
            self.attention_model = self._create_attention_model(X_processed.shape[1])
            self._train_attention_model(X_processed, y)
        
        self.is_fitted = True
        self.logger.info("CatBoost with attention fitted successfully")
        
        return self
    
    def predict(self, X: np.ndarray, tree_predictions: Optional[np.ndarray] = None) -> np.ndarray:
        """Make predictions using CatBoost and attention."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Preprocess features
        X_processed = self._preprocess_features(X)
        
        # Get tree predictions
        if tree_predictions is None:
            tree_predictions = self.catboost_model.predict(X_processed)
        
        # Apply attention mechanism if enabled
        if self.config.use_attention_for_predictions and self.attention_model is not None:
            attention_predictions = self._get_attention_predictions(X_processed)
            
            # Combine tree and attention predictions
            combined_predictions = (
                self.config.attention_weight * attention_predictions +
                (1 - self.config.attention_weight) * tree_predictions
            )
            
            return combined_predictions
        else:
            return tree_predictions
    
    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """Get attention weights for input features."""
        if not self.is_fitted or self.attention_model is None:
            # Return uniform weights if no attention model
            return np.ones((X.shape[0], X.shape[1])) / X.shape[1]
        
        X_processed = self._preprocess_features(X)
        
        if hasattr(self.attention_model, 'attention'):
            # PyTorch model
            import torch
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_processed)
                _, attention_weights = self.attention_model(X_tensor)
                return attention_weights.numpy()
        else:
            # TensorFlow model - extract attention weights
            # This is a simplified implementation
            return np.ones((X.shape[0], X.shape[1])) / X.shape[1]
    
    def _get_attention_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from attention model."""
        if self.attention_model is None:
            return np.zeros(X.shape[0])
        
        if hasattr(self.attention_model, 'forward'):
            # PyTorch model
            import torch
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                predictions, _ = self.attention_model(X_tensor)
                return predictions.numpy().flatten()
        else:
            # TensorFlow model
            predictions = self.attention_model.predict(X)
            return predictions.flatten()
    
    def get_feature_importance(self, X: np.ndarray) -> np.ndarray:
        """Get combined feature importance from CatBoost and attention."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Get CatBoost feature importance
        catboost_importance = self.catboost_model.get_feature_importance()
        
        if self.attention_model is not None:
            # Get attention-based feature importance
            attention_importance = super().get_feature_importance(X)
            
            # Combine importances
            combined_importance = (
                self.config.attention_weight * attention_importance +
                (1 - self.config.attention_weight) * catboost_importance
            )
            
            return combined_importance
        else:
            return catboost_importance
    
    def get_model_parameters(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'catboost_params': self.catboost_model.get_params(),
            'attention_config': self.config,
            'is_fitted': self.is_fitted
        }
    
    def save_model(self, filepath: str) -> None:
        """Save the complete model."""
        import joblib
        
        model_data = {
            'catboost_model': self.catboost_model,
            'attention_model': self.attention_model,
            'config': self.config,
            'is_fitted': self.is_fitted,
            'scaler': getattr(self, 'scaler', None),
            'selected_features': getattr(self, 'selected_features', None)
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'CatBoostAttentionWrapper':
        """Load the complete model."""
        import joblib
        
        model_data = joblib.load(filepath)
        
        self.catboost_model = model_data['catboost_model']
        self.attention_model = model_data['attention_model']
        self.config = model_data['config']
        self.is_fitted = model_data['is_fitted']
        
        if 'scaler' in model_data:
            self.scaler = model_data['scaler']
        if 'selected_features' in model_data:
            self.selected_features = model_data['selected_features']
        
        self.logger.info(f"Model loaded from {filepath}")
        return self
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation."""
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Create a temporary model for CV
        temp_model = CatBoostAttentionWrapper(self.config, self.task_type)
        
        # Perform cross-validation
        cv_scores = cross_val_score(temp_model, X, y, cv=cv, scoring='neg_mean_squared_error')
        
        return {
            'cv_mean': -cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': -cv_scores
        }
    
    def hyperparameter_tune(self, X: np.ndarray, y: np.ndarray, 
                           param_grid: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform hyperparameter tuning."""
        from sklearn.model_selection import GridSearchCV
        
        if param_grid is None:
            param_grid = {
                'config.attention_dim': [32, 64, 128],
                'config.learning_rate': [0.001, 0.01, 0.1],
                'config.regularization': [0.01, 0.1, 1.0],
                'config.attention_weight': [0.3, 0.5, 0.7]
            }
        
        # Create parameter grid for wrapper
        wrapper_params = {}
        for param, values in param_grid.items():
            wrapper_params[param] = values
        
        # Perform grid search
        grid_search = GridSearchCV(
            CatBoostAttentionWrapper(self.config, self.task_type),
            wrapper_params,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }