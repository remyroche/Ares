#!/usr/bin/env python3
"""
Advanced Surrogate Models for Optimization

This module provides advanced surrogate models including:
- Ensemble methods
- Deep learning models
- Specialized kernels
- Multi-task learning
- Transfer learning capabilities
"""

from src.core.decorators import handles_errors

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
import joblib
import time

# Core ML libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern, RationalQuadratic
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler

# Advanced ML libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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

# Utilities
from src.utils.logger import system_logger

import copy

class BaseSurrogateModel(ABC):
    """Base class for all surrogate models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild(self.__class__.__name__)
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.training_time = 0.0
        self.prediction_time = 0.0
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the surrogate model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict values and uncertainties."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass
    
    def save_model(self, filepath: str) -> None:
        """Save the model to disk."""
        if self.model is not None:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'config': self.config,
                'training_time': self.training_time
            }, filepath)
            self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load the model from disk."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.config = data['config']
        self.training_time = data['training_time']
        self.is_fitted = True
        self.logger.info(f"Model loaded from {filepath}")

class EnsembleSurrogateModel(BaseSurrogateModel):
    """Ensemble of multiple surrogate models for robust predictions."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.models = {}
        self.weights = {}
        self.ensemble_method = config.get('ensemble_method', 'weighted_average')
        
    def add_model(self, name: str, model: BaseSurrogateModel, weight: float = 1.0) -> None:
        """Add a model to the ensemble."""
        self.models[name] = model
        self.weights[name] = weight
        self.logger.info(f"Added model '{name}' to ensemble with weight {weight}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit all models in the ensemble."""
        start_time = time.time()
        
        for name, model in self.models.items():
            self.logger.info(f"Fitting ensemble model: {name}")
            model.fit(X, y)
        
        # Optionally optimize weights based on cross-validation
        if self.config.get('optimize_weights', False):
            self._optimize_weights(X, y)
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        self.logger.info(f"Ensemble training completed in {self.training_time:.2f}s")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using ensemble methods."""
        if not self.is_fitted:
            raise ValueError("Ensemble model must be fitted before prediction")
        
        start_time = time.time()
        
        predictions = {}
        uncertainties = {}
        
        # Get predictions from all models
        for name, model in self.models.items():
            pred, unc = model.predict(X)
            predictions[name] = pred
            uncertainties[name] = unc
        
        # Combine predictions based on ensemble method
        if self.ensemble_method == 'weighted_average':
            final_pred, final_unc = self._weighted_average_ensemble(predictions, uncertainties)
        elif self.ensemble_method == 'stacking':
            final_pred, final_unc = self._stacking_ensemble(predictions, uncertainties)
        elif self.ensemble_method == 'bagging':
            final_pred, final_unc = self._bagging_ensemble(predictions, uncertainties)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        self.prediction_time = time.time() - start_time
        return final_pred, final_unc
    
    def _weighted_average_ensemble(
        self, 
        predictions: Dict[str, np.ndarray], 
        uncertainties: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Weighted average ensemble method."""
        total_weight = sum(self.weights.values())
        
        # Weighted average of predictions
        final_pred = np.zeros_like(list(predictions.values())[0])
        for name, pred in predictions.items():
            weight = self.weights[name] / total_weight
            final_pred += weight * pred
        
        # Weighted average of uncertainties
        final_unc = np.zeros_like(list(uncertainties.values())[0])
        for name, unc in uncertainties.items():
            weight = self.weights[name] / total_weight
            final_unc += weight * unc
        
        return final_pred, final_unc
    
    def _stacking_ensemble(
        self, 
        predictions: Dict[str, np.ndarray], 
        uncertainties: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stacking ensemble method."""
        # For now, use simple weighted average
        # Could be extended with meta-learner
        return self._weighted_average_ensemble(predictions, uncertainties)
    
    def _bagging_ensemble(
        self, 
        predictions: Dict[str, np.ndarray], 
        uncertainties: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bagging ensemble method."""
        # Use mean and std of predictions
        pred_array = np.array(list(predictions.values()))
        unc_array = np.array(list(uncertainties.values()))
        
        final_pred = np.mean(pred_array, axis=0)
        final_unc = np.std(pred_array, axis=0) + np.mean(unc_array, axis=0)
        
        return final_pred, final_unc
    
    def _optimize_weights(self, X: np.ndarray, y: np.ndarray) -> None:
        """Optimize ensemble weights using cross-validation."""
        # Simple optimization: weight inversely proportional to CV error
        cv_scores = {}
        
        for name, model in self.models.items():
            try:
                cv_score = cross_val_score(model.model, X, y, cv=5, scoring='neg_mean_squared_error')
                cv_scores[name] = -np.mean(cv_score)
            except Exception as e:
                self.logger.warning(f"CV failed for model {name}: {e}")
                cv_scores[name] = 1.0
        
        # Set weights inversely proportional to CV error
        total_inv_error = sum(1.0 / max(score, 1e-6) for score in cv_scores.values())
        for name, error in cv_scores.items():
            self.weights[name] = (1.0 / max(error, 1e-6)) / total_inv_error
        
        self.logger.info(f"Optimized ensemble weights: {self.weights}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get ensemble model information."""
        return {
            'ensemble_method': self.ensemble_method,
            'num_models': len(self.models),
            'model_names': list(self.models.keys()),
            'weights': self.weights,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time
        }

class DeepSurrogateModel(BaseSurrogateModel):
    """Deep learning surrogate model using PyTorch."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DeepSurrogateModel")
        
        self.network_config = config.get('network', {})
        self.training_config = config.get('training', {})
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.optimizer = None
        self.criterion = None
        
    def _build_network(self, input_dim: int) -> nn.Module:
        """Build the neural network architecture."""
        layers = []
        hidden_dims = self.network_config.get('hidden_dims', [100, 50, 25])
        dropout_rate = self.network_config.get('dropout_rate', 0.1)
        activation = self.network_config.get('activation', 'relu')
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(self._get_activation(activation))
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer (mean and variance)
        layers.append(nn.Linear(hidden_dims[-1], 2))
        
        return nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            return nn.ReLU()
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the deep surrogate model."""
        start_time = time.time()
        
        # Prepare data
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Build network
        self.model = self._build_network(X.shape[1]).to(self.device)
        
        # Setup training
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.training_config.get('learning_rate', 0.001),
            weight_decay=self.training_config.get('weight_decay', 1e-5)
        )
        
        self.criterion = self._get_loss_function()
        
        # Training loop
        epochs = self.training_config.get('epochs', 1000)
        batch_size = self.training_config.get('batch_size', 32)
        patience = self.training_config.get('patience', 50)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(batch_X)
                loss = self.criterion(output, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 100 == 0:
                self.logger.info(f"Epoch {epoch}, Loss: {epoch_loss:.6f}")
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        self.logger.info(f"Deep model training completed in {self.training_time:.2f}s")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using the deep surrogate model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        start_time = time.time()
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            output = self.model(X_tensor)
            
            # Split output into mean and variance
            mean = output[:, 0].cpu().numpy()
            variance = torch.exp(output[:, 1]).cpu().numpy()  # Ensure positive variance
        
        self.prediction_time = time.time() - start_time
        return mean, np.sqrt(variance)  # Return mean and std
    
    def _get_loss_function(self):
        """Get loss function for training."""
        loss_type = self.training_config.get('loss', 'mse')
        
        if loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'mae':
            return nn.L1Loss()
        elif loss_type == 'huber':
            return nn.HuberLoss()
        else:
            return nn.MSELoss()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get deep model information."""
        return {
            'model_type': 'deep_neural_network',
            'device': str(self.device),
            'network_config': self.network_config,
            'training_config': self.training_config,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time
        }

class AdvancedGaussianProcessModel(BaseSurrogateModel):
    """Advanced Gaussian Process with specialized kernels."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.kernel_config = config.get('kernel', {})
        self.gp_config = config.get('gaussian_process', {})
        
    def _build_kernel(self, input_dim: int) -> Any:
        """Build advanced kernel based on configuration."""
        kernel_type = self.kernel_config.get('type', 'rbf_constant_white')
        
        if kernel_type == 'rbf_constant_white':
            return (
                ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) *
                RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) +
                WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-3))
            )
        elif kernel_type == 'matern':
            nu = self.kernel_config.get('nu', 1.5)
            return Matern(length_scale=1.0, nu=nu, length_scale_bounds=(1e-2, 1e2))
        elif kernel_type == 'rational_quadratic':
            alpha = self.kernel_config.get('alpha', 1.0)
            return RationalQuadratic(length_scale=1.0, alpha=alpha, length_scale_bounds=(1e-2, 1e2))
        elif kernel_type == 'composite':
            return self._build_composite_kernel(input_dim)
        else:
            return RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    
    def _build_composite_kernel(self, input_dim: int) -> Any:
        """Build composite kernel for different input dimensions."""
        # This could be extended with domain-specific kernels
        return (
            ConstantKernel(1.0) * RBF(length_scale=1.0) +
            WhiteKernel(noise_level=1e-5)
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the advanced Gaussian Process model."""
        start_time = time.time()
        
        # Build kernel
        kernel = self._build_kernel(X.shape[1])
        
        # Create GP model
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.gp_config.get('alpha', 1e-6),
            n_restarts_optimizer=self.gp_config.get('n_restarts_optimizer', 10),
            random_state=self.gp_config.get('random_state', 42)
        )
        
        # Fit model
        self.model.fit(X, y)
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        self.logger.info(f"Advanced GP training completed in {self.training_time:.2f}s")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using the advanced Gaussian Process model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        start_time = time.time()
        
        mean, std = self.model.predict(X, return_std=True)
        
        self.prediction_time = time.time() - start_time
        return mean, std
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get advanced GP model information."""
        return {
            'model_type': 'advanced_gaussian_process',
            'kernel_config': self.kernel_config,
            'gp_config': self.gp_config,
            'kernel': str(self.model.kernel_) if self.model else None,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time
        }

class MultiTaskSurrogateModel(BaseSurrogateModel):
    """Multi-task surrogate model for related optimization problems."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.task_config = config.get('multi_task', {})
        self.models = {}
        self.task_relationships = {}
        
    def add_task(self, task_name: str, model: BaseSurrogateModel) -> None:
        """Add a task to the multi-task model."""
        self.models[task_name] = model
        self.logger.info(f"Added task '{task_name}' to multi-task model")
    
    def set_task_relationship(self, task1: str, task2: str, relationship: float) -> None:
        """Set relationship between tasks (0-1, where 1 is highly related)."""
        self.task_relationships[(task1, task2)] = relationship
        self.task_relationships[(task2, task1)] = relationship
        self.logger.info(f"Set relationship between {task1} and {task2}: {relationship}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, task_names: List[str]) -> None:
        """Fit the multi-task surrogate model."""
        start_time = time.time()
        
        # Fit individual models
        for task_name, model in self.models.items():
            if task_name in task_names:
                self.logger.info(f"Fitting multi-task model for: {task_name}")
                model.fit(X, y)
        
        # Learn task relationships if not provided
        if not self.task_relationships:
            self._learn_task_relationships(X, y, task_names)
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        self.logger.info(f"Multi-task training completed in {self.training_time:.2f}s")
    
    def predict(self, X: np.ndarray, task_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Predict for a specific task."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if task_name not in self.models:
            raise ValueError(f"Task '{task_name}' not found in multi-task model")
        
        # Get base prediction
        base_pred, base_unc = self.models[task_name].predict(X)
        
        # Apply task relationships if available
        if self.task_relationships:
            adjusted_pred, adjusted_unc = self._apply_task_relationships(
                X, base_pred, base_unc, task_name
            )
            return adjusted_pred, adjusted_unc
        
        return base_pred, base_unc
    
    def _learn_task_relationships(self, X: np.ndarray, y: np.ndarray, task_names: List[str]) -> None:
        """Learn relationships between tasks based on data."""
        # Simple correlation-based relationship learning
        for i, task1 in enumerate(task_names):
            for j, task2 in enumerate(task_names):
                if i < j:
                    # Calculate correlation between task predictions
                    pred1, _ = self.models[task1].predict(X)
                    pred2, _ = self.models[task2].predict(X)
                    
                    correlation = np.corrcoef(pred1, pred2)[0, 1]
                    relationship = abs(correlation) if not np.isnan(correlation) else 0.0
                    
                    self.set_task_relationship(task1, task2, relationship)
    
    def _apply_task_relationships(
        self, 
        X: np.ndarray, 
        base_pred: np.ndarray, 
        base_unc: np.ndarray, 
        task_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply task relationships to improve predictions."""
        # Simple weighted combination of related task predictions
        weighted_pred = base_pred.copy()
        weighted_unc = base_unc.copy()
        
        for other_task, relationship in self.task_relationships.items():
            if other_task[0] == task_name and other_task[1] != task_name:
                other_task_name = other_task[1]
                if other_task_name in self.models:
                    other_pred, other_unc = self.models[other_task_name].predict(X)
                    
                    # Weighted combination
                    weight = relationship * 0.1  # Small weight for regularization
                    weighted_pred = (1 - weight) * weighted_pred + weight * other_pred
                    weighted_unc = np.sqrt((1 - weight) * weighted_unc**2 + weight * other_unc**2)
        
        return weighted_pred, weighted_unc
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get multi-task model information."""
        return {
            'model_type': 'multi_task_surrogate',
            'num_tasks': len(self.models),
            'task_names': list(self.models.keys()),
            'task_relationships': self.task_relationships,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time
        }

class SurrogateModelFactory:
    """Factory for creating different types of surrogate models."""
    
    @staticmethod
    def create_model(model_type: str, config: Dict[str, Any]) -> BaseSurrogateModel:
        """Create a surrogate model of the specified type."""
        if model_type == "ensemble":
            return EnsembleSurrogateModel(config)
        elif model_type == "deep":
            return DeepSurrogateModel(config)
        elif model_type == "advanced_gp":
            return AdvancedGaussianProcessModel(config)
        elif model_type == "multi_task":
            return MultiTaskSurrogateModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available model types."""
        models = ["ensemble", "advanced_gp"]
        
        if TORCH_AVAILABLE:
            models.append("deep")
        
        models.append("multi_task")
        
        return models