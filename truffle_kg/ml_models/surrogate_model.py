"""
Surrogate Models for Truffle Cultivation Simulation
AI-accelerated design-space search and control optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import joblib
from pathlib import Path

# ML libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Deep learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available, neural network models disabled")

logger = logging.getLogger(__name__)

@dataclass
class SurrogateModelConfig:
    """Configuration for surrogate models"""
    model_type: str = "random_forest"  # random_forest, gradient_boosting, neural_network, gaussian_process
    hidden_layers: List[int] = None
    learning_rate: float = 0.001
    n_estimators: int = 100
    max_depth: int = 10
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.2

class SurrogateModel:
    """Base class for surrogate models"""
    
    def __init__(self, config: SurrogateModelConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.target_names = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Fit the surrogate model"""
        if self.model is None:
            raise ValueError("Model not initialized. Use a specific model class like RandomForestSurrogate.")
        
        logger.info(f"Fitting {self.__class__.__name__} surrogate model...")
        
        # Validate inputs
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length. Got X: {len(X)}, y: {len(y)}")
        
        if len(X) == 0:
            raise ValueError("Cannot fit model with empty data")
        
        # Store feature and target information
        self.feature_names = [f"feature_{i}" for i in range(X.shape[1])] if self.feature_names is None else self.feature_names
        self.target_names = [f"target_{i}" for i in range(y.shape[1])] if len(y.shape) > 1 and self.target_names is None else self.target_names
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        
        # Fit the model
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        self.is_fitted = True
        logger.info(f"{self.__class__.__name__} fitted successfully. R² = {metrics['r2']:.3f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.model is None:
            raise ValueError("Model not initialized. Use a specific model class like RandomForestSurrogate.")
        
        # Validate input shape
        if len(X.shape) != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        
        # Scale features using the fitted scaler
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        logger.debug(f"Made predictions for {len(X)} samples")
        return predictions
    
    def save(self, filepath: str):
        """Save the model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        if self.model is None:
            raise ValueError("Model not initialized. Use a specific model class like RandomForestSurrogate.")
        
        # Prepare model data for saving
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'config': self.config,
            'model_class': self.__class__.__name__
        }
        
        # Save using joblib for most models
        joblib.dump(model_data, filepath)
        logger.info(f"{self.__class__.__name__} model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load the model"""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model data
        model_data = joblib.load(filepath)
        
        # Restore model state
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_fitted = model_data['is_fitted']
        self.feature_names = model_data['feature_names']
        self.target_names = model_data['target_names']
        self.config = model_data['config']
        
        logger.info(f"{self.__class__.__name__} model loaded from {filepath}")
        
        # Validate loaded model
        if self.model is None:
            raise ValueError("Loaded model is None. File may be corrupted.")
        
        if not self.is_fitted:
            logger.warning("Loaded model is not fitted. You may need to retrain it.")

class RandomForestSurrogate(SurrogateModel):
    """Random Forest surrogate model"""
    
    def __init__(self, config: SurrogateModelConfig):
        super().__init__(config)
        self.model = RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=config.random_state,
            n_jobs=-1
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Fit the Random Forest model"""
        logger.info("Fitting Random Forest surrogate model...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        
        # Fit model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        self.is_fitted = True
        logger.info(f"Random Forest fitted. R² = {metrics['r2']:.3f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.feature_importances_
    
    def save(self, filepath: str):
        """Save the model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'config': self.config
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Random Forest model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load the model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_fitted = model_data['is_fitted']
        self.feature_names = model_data['feature_names']
        self.target_names = model_data['target_names']
        self.config = model_data['config']
        logger.info(f"Random Forest model loaded from {filepath}")

class GradientBoostingSurrogate(SurrogateModel):
    """Gradient Boosting surrogate model"""
    
    def __init__(self, config: SurrogateModelConfig):
        super().__init__(config)
        self.model = GradientBoostingRegressor(
            n_estimators=config.n_estimators,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            random_state=config.random_state
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Fit the Gradient Boosting model"""
        logger.info("Fitting Gradient Boosting surrogate model...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        
        # Fit model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        self.is_fitted = True
        logger.info(f"Gradient Boosting fitted. R² = {metrics['r2']:.3f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        return self.model.feature_importances_
    
    def save(self, filepath: str):
        """Save the model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'config': self.config
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Gradient Boosting model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load the model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_fitted = model_data['is_fitted']
        self.feature_names = model_data['feature_names']
        self.target_names = model_data['target_names']
        self.config = model_data['config']
        logger.info(f"Gradient Boosting model loaded from {filepath}")

class NeuralNetworkSurrogate(SurrogateModel):
    """Neural Network surrogate model using PyTorch"""
    
    def __init__(self, config: SurrogateModelConfig):
        super().__init__(config)
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for neural network models")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
    def _create_network(self, input_size: int, output_size: int):
        """Create the neural network architecture"""
        layers = []
        prev_size = input_size
        
        for hidden_size in self.config.hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        return nn.Sequential(*layers)
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32) -> Dict[str, float]:
        """Fit the Neural Network model"""
        logger.info("Fitting Neural Network surrogate model...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        
        # Create network
        input_size = X_train.shape[1]
        output_size = y_train.shape[1] if len(y_train.shape) > 1 else 1
        self.network = self._create_network(input_size, output_size).to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        # Training loop
        self.network.train()
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.network(X_train_tensor)
            if len(y_train_tensor.shape) == 1:
                y_pred = y_pred.squeeze()
            
            loss = self.criterion(y_pred, y_train_tensor)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Evaluate
        self.network.eval()
        with torch.no_grad():
            y_pred = self.network(X_test_tensor)
            if len(y_test_tensor.shape) == 1:
                y_pred = y_pred.squeeze()
            
            y_pred_np = y_pred.cpu().numpy()
            y_test_np = y_test_tensor.cpu().numpy()
            
            metrics = {
                'mse': mean_squared_error(y_test_np, y_pred_np),
                'rmse': np.sqrt(mean_squared_error(y_test_np, y_pred_np)),
                'mae': mean_absolute_error(y_test_np, y_pred_np),
                'r2': r2_score(y_test_np, y_pred_np)
            }
        
        self.is_fitted = True
        logger.info(f"Neural Network fitted. R² = {metrics['r2']:.3f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.network.eval()
        with torch.no_grad():
            y_pred = self.network(X_tensor)
            if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
                y_pred = y_pred.squeeze()
        
        return y_pred.cpu().numpy()
    
    def save(self, filepath: str):
        """Save the model"""
        model_data = {
            'network_state_dict': self.network.state_dict() if self.network else None,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'config': self.config,
            'device': str(self.device)
        }
        torch.save(model_data, filepath)
        logger.info(f"Neural Network model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load the model"""
        model_data = torch.load(filepath, map_location=self.device)
        
        if model_data['network_state_dict']:
            input_size = len(model_data['feature_names']) if model_data['feature_names'] else 10
            output_size = len(model_data['target_names']) if model_data['target_names'] else 1
            self.network = self._create_network(input_size, output_size).to(self.device)
            self.network.load_state_dict(model_data['network_state_dict'])
        
        if model_data['optimizer_state_dict'] and self.network:
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
            self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
        
        self.scaler = model_data['scaler']
        self.is_fitted = model_data['is_fitted']
        self.feature_names = model_data['feature_names']
        self.target_names = model_data['target_names']
        self.config = model_data['config']
        self.device = torch.device(model_data['device'])
        
        logger.info(f"Neural Network model loaded from {filepath}")

class GaussianProcessSurrogate(SurrogateModel):
    """Gaussian Process surrogate model"""
    
    def __init__(self, config: SurrogateModelConfig):
        super().__init__(config)
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            random_state=config.random_state,
            n_restarts_optimizer=10
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Fit the Gaussian Process model"""
        logger.info("Fitting Gaussian Process surrogate model...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        
        # Fit model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred, y_std = self.model.predict(X_test, return_std=True)
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        self.is_fitted = True
        logger.info(f"Gaussian Process fitted. R² = {metrics['r2']:.3f}")
        
        return metrics
    
    def predict(self, X: np.ndarray, return_std: bool = False) -> np.ndarray:
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        if return_std:
            return self.model.predict(X_scaled, return_std=True)
        else:
            return self.model.predict(X_scaled)
    
    def save(self, filepath: str):
        """Save the model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'config': self.config
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Gaussian Process model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load the model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_fitted = model_data['is_fitted']
        self.feature_names = model_data['feature_names']
        self.target_names = model_data['target_names']
        self.config = model_data['config']
        logger.info(f"Gaussian Process model loaded from {filepath}")

class SurrogateModelFactory:
    """Factory for creating surrogate models"""
    
    @staticmethod
    def create_model(config: SurrogateModelConfig) -> SurrogateModel:
        """Create a surrogate model based on configuration"""
        if config.model_type == "random_forest":
            return RandomForestSurrogate(config)
        elif config.model_type == "gradient_boosting":
            return GradientBoostingSurrogate(config)
        elif config.model_type == "neural_network":
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch is required for neural network models")
            return NeuralNetworkSurrogate(config)
        elif config.model_type == "gaussian_process":
            return GaussianProcessSurrogate(config)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

class TruffleSurrogateManager:
    """Manager for truffle cultivation surrogate models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.training_data = None
        
    def create_model(self, name: str, model_type: str, **kwargs) -> SurrogateModel:
        """Create a new surrogate model"""
        config = SurrogateModelConfig(model_type=model_type, **kwargs)
        model = SurrogateModelFactory.create_model(config)
        self.models[name] = model
        logger.info(f"Created {model_type} surrogate model: {name}")
        return model
    
    def train_model(self, name: str, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
        """Train a surrogate model"""
        if name not in self.models:
            raise ValueError(f"Model {name} not found")
        
        model = self.models[name]
        metrics = model.fit(X, y, **kwargs)
        
        # Store training data
        self.training_data = {'X': X, 'y': y}
        
        logger.info(f"Model {name} trained successfully")
        return metrics
    
    def predict(self, name: str, X: np.ndarray, **kwargs) -> np.ndarray:
        """Make predictions with a surrogate model"""
        if name not in self.models:
            raise ValueError(f"Model {name} not found")
        
        return self.models[name].predict(X, **kwargs)
    
    def save_model(self, name: str, filepath: str):
        """Save a surrogate model"""
        if name not in self.models:
            raise ValueError(f"Model {name} not found")
        
        self.models[name].save(filepath)
    
    def load_model(self, name: str, filepath: str):
        """Load a surrogate model"""
        # Determine model type from file
        model_data = joblib.load(filepath) if filepath.endswith('.pkl') else torch.load(filepath)
        model_type = model_data['config'].model_type
        
        # Create and load model
        model = self.create_model(name, model_type)
        model.load(filepath)
    
    def compare_models(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Compare performance of all models"""
        results = []
        
        for name, model in self.models.items():
            try:
                metrics = model.fit(X, y)
                results.append({
                    'Model': name,
                    'Type': model.config.model_type,
                    'R²': metrics['r2'],
                    'RMSE': metrics['rmse'],
                    'MAE': metrics['mae']
                })
            except Exception as e:
                logger.error(f"Error training model {name}: {e}")
                results.append({
                    'Model': name,
                    'Type': model.config.model_type,
                    'R²': np.nan,
                    'RMSE': np.nan,
                    'MAE': np.nan
                })
        
        return pd.DataFrame(results)

def main():
    """Example usage of surrogate models"""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X**2, axis=1) + 0.1 * np.random.randn(n_samples)
    
    # Create surrogate manager
    manager = TruffleSurrogateManager({})
    
    # Create and train different models
    models_to_test = [
        ("rf", "random_forest"),
        ("gb", "gradient_boosting"),
        ("gp", "gaussian_process")
    ]
    
    if TORCH_AVAILABLE:
        models_to_test.append(("nn", "neural_network"))
    
    for name, model_type in models_to_test:
        try:
            if model_type == "neural_network":
                model = manager.create_model(name, model_type, hidden_layers=[64, 32, 16])
            else:
                model = manager.create_model(name, model_type)
            
            metrics = manager.train_model(name, X, y)
            print(f"{name}: R² = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.3f}")
            
        except Exception as e:
            print(f"Error with {name}: {e}")
    
    # Compare models
    comparison = manager.compare_models(X, y)
    print("\nModel Comparison:")
    print(comparison)

if __name__ == "__main__":
    main()