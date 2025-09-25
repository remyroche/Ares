"""
Neural Ordinary Differential Equations (Neural ODEs)

A comprehensive implementation of Neural ODEs for financial time series modeling,
integrating with the existing utility framework for optimal performance on Apple Silicon.

This module provides:
- Neural ODE implementation for continuous-time modeling
- Integration with M1 GPU/Memory/CPU optimizers
- Cross-validation and hyperparameter optimization
- Matrix operations with hardware acceleration
- Comprehensive logging and serialization
- Financial time series specific features
"""

import logging
import time
import gc
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from contextlib import contextmanager
from pathlib import Path
import pickle
import json

# Core dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
    warnings.warn("NumPy not available - NeuralODE functionality will be limited")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# PyTorch for Neural ODE implementation
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchdiffeq import odeint, odeint_adjoint
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = nn = optim = odeint = odeint_adjoint = None
    warnings.warn("PyTorch not available - NeuralODE requires PyTorch for full functionality")

# SciPy for ODE solvers (fallback)
try:
    from scipy.integrate import odeint as scipy_odeint
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    scipy_odeint = minimize = None

# Import utility frameworks
try:
    from src.utils.common_operations import (
        safe_json_dump, safe_json_load, ensure_directory,
        safe_file_exists, get_m1_gpu_manager, get_m1_memory_optimizer,
        get_m1_cpu_optimizer, integrate_with_m1_optimizers,
        memory_checkpoint, gpu_context, optimize_memory,
        safe_divide, safe_log, safe_sqrt, validate_finite,
        safe_matrix_inverse, timed_operation
    )
    from src.utils.math_validation import (
        validate_numeric_array, safe_correlation, safe_mean, safe_std,
        MathValidation, MathValidationError
    )
    from src.utils.serialization_utils import UniversalSerializer
    from src.utils.tprint import (
        tprint, tprint_info, tprint_error, tprint_warning, tprint_success,
        tprint_performance, tprint_progress, tprint_with_level, LogLevel
    )
    UTILITIES_AVAILABLE = True
except ImportError as e:
    UTILITIES_AVAILABLE = False
    warnings.warn(f"Some utilities not available: {e}")

# ML Common utilities
try:
    from src.utils.ml_common.validation.unified_cv import perform_cross_validation
    from src.utils.ml_common.optimization.hpo_utils import HyperparameterOptimizer
    ML_COMMON_AVAILABLE = True
except ImportError:
    ML_COMMON_AVAILABLE = False
    perform_cross_validation = None
    HyperparameterOptimizer = None

# Matrix operations
try:
    from src.utils.matrix_operations.unified_operations import (
        UnifiedMatrixOperations, MatrixOperationConfig
    )
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False
    UnifiedMatrixOperations = None

# Setup logging
logger = logging.getLogger(__name__)

class NeuralODEModel(nn.Module):
    """
    Neural ODE model for continuous-time dynamics modeling.
    
    This class implements a neural network that parameterizes the derivative
    of a dynamical system, enabling continuous-time modeling of time series.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 64],
        output_dim: Optional[int] = None,
        activation: str = 'tanh',
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        device: Optional[str] = None
    ):
        """
        Initialize Neural ODE model.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (defaults to input_dim)
            activation: Activation function ('tanh', 'relu', 'elu')
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            device: Device to use ('cpu', 'cuda', 'mps')
        """
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for NeuralODEModel")
        
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.hidden_dims = hidden_dims
        self.device = device or self._get_optimal_device()
        
        # Build network layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            if use_batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], self.output_dim))
        
        # Activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.Tanh()
        
        # Initialize weights
        self._initialize_weights()
        
        # Move to device
        self.to(self.device)
        
        tprint_info(f"Initialized NeuralODEModel: {input_dim} -> {hidden_dims} -> {self.output_dim}")
    
    def _get_optimal_device(self) -> str:
        """Get optimal device for the current hardware."""
        if not TORCH_AVAILABLE:
            return 'cpu'
        
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.
        
        Args:
            t: Time tensor (batch_size,)
            y: State tensor (batch_size, input_dim)
            
        Returns:
            Derivative tensor (batch_size, output_dim)
        """
        x = y
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Apply activation after linear layers (except the last one)
            if i < len(self.layers) - 1 and not isinstance(layer, (nn.BatchNorm1d, nn.Dropout)):
                x = self.activation(x)
        
        return x


class NeuralODE:
    """
    Neural Ordinary Differential Equations implementation for financial time series modeling.
    
    This class provides a comprehensive implementation of Neural ODEs with:
    - Continuous-time modeling capabilities
    - Integration with hardware optimizers
    - Cross-validation and hyperparameter optimization
    - Financial time series specific features
    - Comprehensive logging and serialization
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 64],
        output_dim: Optional[int] = None,
        activation: str = 'tanh',
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        solver: str = 'dopri5',
        rtol: float = 1e-3,
        atol: float = 1e-4,
        use_adjoint: bool = True,
        device: Optional[str] = None,
        enable_hardware_optimization: bool = True,
        enable_logging: bool = True
    ):
        """
        Initialize NeuralODE.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (defaults to input_dim)
            activation: Activation function
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            solver: ODE solver ('dopri5', 'adams', 'euler', 'rk4')
            rtol: Relative tolerance for ODE solver
            atol: Absolute tolerance for ODE solver
            use_adjoint: Whether to use adjoint method for memory efficiency
            device: Device to use
            enable_hardware_optimization: Whether to enable M1 optimizations
            enable_logging: Whether to enable comprehensive logging
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for NeuralODE")
        
        # Store configuration
        self.config = {
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'output_dim': output_dim or input_dim,
            'activation': activation,
            'dropout': dropout,
            'use_batch_norm': use_batch_norm,
            'solver': solver,
            'rtol': rtol,
            'atol': atol,
            'use_adjoint': use_adjoint,
            'device': device,
            'enable_hardware_optimization': enable_hardware_optimization,
            'enable_logging': enable_logging
        }
        
        # Initialize components
        self.logger = logger.getChild('NeuralODE')
        self.math_validator = MathValidation() if UTILITIES_AVAILABLE else None
        
        # Hardware optimization setup
        if enable_hardware_optimization and UTILITIES_AVAILABLE:
            self._setup_hardware_optimization()
        
        # Initialize neural ODE model
        self.model = NeuralODEModel(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            device=device
        )
        
        # Initialize optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        
        # Training history
        self.training_history = {
            'loss': [],
            'validation_loss': [],
            'epochs': [],
            'best_loss': float('inf'),
            'best_model_state': None
        }
        
        # Performance metrics
        self.performance_metrics = {}
        
        # Serialization
        self.serializer = UniversalSerializer() if UTILITIES_AVAILABLE else None
        
        if enable_logging:
            tprint_success(f"Initialized NeuralODE: {input_dim}D -> {hidden_dims} -> {self.config['output_dim']}D")
    
    def _setup_hardware_optimization(self):
        """Setup hardware optimization for M1 systems."""
        try:
            # Integrate with M1 optimizers
            integration_result = integrate_with_m1_optimizers()
            if integration_result.get('success', False):
                tprint_info("M1 hardware optimization enabled")
                self.hardware_optimized = True
            else:
                tprint_warning("M1 hardware optimization failed, using fallback")
                self.hardware_optimized = False
        except Exception as e:
            tprint_warning(f"Hardware optimization setup failed: {e}")
            self.hardware_optimized = False
    
    def _validate_input(self, data: np.ndarray, name: str = "data") -> np.ndarray:
        """Validate input data."""
        if not NUMPY_AVAILABLE:
            raise ImportError("NumPy is required for data validation")
        
        if self.math_validator:
            return validate_numeric_array(data, name)
        else:
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            if not np.all(np.isfinite(data)):
                raise ValueError(f"{name} contains non-finite values")
            return data
    
    def _to_torch(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert data to PyTorch tensor."""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float().to(self.model.device)
        return data.float().to(self.model.device)
    
    def _to_numpy(self, data: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to NumPy array."""
        return data.detach().cpu().numpy()
    
    @contextmanager
    def _hardware_context(self, operation_name: str):
        """Hardware optimization context manager."""
        if self.config['enable_hardware_optimization'] and UTILITIES_AVAILABLE:
            with memory_checkpoint(f"neural_ode_{operation_name}"):
                with gpu_context(f"neural_ode_{operation_name}"):
                    yield
        else:
            yield
    
    def set_optimizer(
        self,
        optimizer_type: str = 'adam',
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_type: Optional[str] = None,
        scheduler_params: Optional[Dict] = None
    ):
        """
        Set optimizer and scheduler for training.
        
        Args:
            optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop')
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            scheduler_type: Type of scheduler ('cosine', 'step', 'plateau')
            scheduler_params: Additional scheduler parameters
        """
        # Set optimizer
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        elif optimizer_type.lower() == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # Set scheduler
        if scheduler_type:
            scheduler_params = scheduler_params or {}
            if scheduler_type.lower() == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, **scheduler_params
                )
            elif scheduler_type.lower() == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, **scheduler_params
                )
            elif scheduler_type.lower() == 'plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, **scheduler_params
                )
        
        if self.config['enable_logging']:
            tprint_info(f"Set optimizer: {optimizer_type} (lr={learning_rate})")
    
    def solve_ode(
        self,
        initial_condition: torch.Tensor,
        time_points: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Solve the Neural ODE for given initial condition and time points.
        
        Args:
            initial_condition: Initial state (batch_size, input_dim)
            time_points: Time points to evaluate (num_points,)
            **kwargs: Additional solver parameters
            
        Returns:
            Solution tensor (batch_size, num_points, input_dim)
        """
        if self.optimizer is None:
            raise ValueError("Optimizer not set. Call set_optimizer() first.")
        
        # Choose solver function
        if self.config['use_adjoint']:
            solver_func = odeint_adjoint
        else:
            solver_func = odeint
        
        # Solve ODE
        solution = solver_func(
            self.model,
            initial_condition,
            time_points,
            rtol=self.config['rtol'],
            atol=self.config['atol'],
            method=self.config['solver'],
            **kwargs
        )
        
        return solution
    
    def predict(
        self,
        initial_condition: Union[np.ndarray, torch.Tensor],
        time_horizon: float,
        num_steps: int = 100,
        return_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Predict future states using the Neural ODE.
        
        Args:
            initial_condition: Initial state
            time_horizon: Time horizon for prediction
            num_steps: Number of time steps
            return_numpy: Whether to return NumPy array
            
        Returns:
            Predicted trajectory
        """
        # Validate and convert input
        if isinstance(initial_condition, np.ndarray):
            initial_condition = self._validate_input(initial_condition, "initial_condition")
        
        initial_condition = self._to_torch(initial_condition)
        time_points = torch.linspace(0, time_horizon, num_steps, device=self.model.device)
        
        # Solve ODE
        with torch.no_grad():
            trajectory = self.solve_ode(initial_condition, time_points)
        
        if return_numpy:
            return self._to_numpy(trajectory)
        return trajectory
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ):
        """
        Train the Neural ODE model.
        
        Args:
            X: Input time series (samples, time_steps, features)
            y: Target time series (samples, time_steps, features)
            validation_data: Validation data tuple (X_val, y_val)
            epochs: Number of training epochs
            batch_size: Batch size for training
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print training progress
        """
        if self.optimizer is None:
            raise ValueError("Optimizer not set. Call set_optimizer() first.")
        
        # Validate inputs
        X = self._validate_input(X, "X")
        y = self._validate_input(y, "y")
        
        # Convert to PyTorch tensors
        X_torch = self._to_torch(X)
        y_torch = self._to_torch(y)
        
        # Training setup
        num_samples, num_timesteps, num_features = X_torch.shape
        time_points = torch.linspace(0, 1, num_timesteps, device=self.model.device)
        
        best_loss = float('inf')
        patience_counter = 0
        
        if verbose:
            tprint_info(f"Starting training: {epochs} epochs, batch_size={batch_size}")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Training loop
            for i in range(0, num_samples, batch_size):
                batch_X = X_torch[i:i + batch_size]
                batch_y = y_torch[i:i + batch_size]
                
                # Forward pass
                self.optimizer.zero_grad()
                
                # Get initial conditions (first timestep)
                initial_conditions = batch_X[:, 0, :]
                
                # Solve ODE
                predicted = self.solve_ode(initial_conditions, time_points)
                
                # Compute loss
                loss = self.criterion(predicted, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Average loss
            avg_loss = epoch_loss / num_batches
            
            # Validation
            val_loss = None
            if validation_data is not None:
                val_loss = self._evaluate_validation(validation_data, time_points)
            
            # Update training history
            self.training_history['loss'].append(avg_loss)
            self.training_history['epochs'].append(epoch)
            if val_loss is not None:
                self.training_history['validation_loss'].append(val_loss)
            
            # Update best model
            current_loss = val_loss if val_loss is not None else avg_loss
            if current_loss < best_loss:
                best_loss = current_loss
                self.training_history['best_loss'] = best_loss
                self.training_history['best_model_state'] = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(current_loss)
                else:
                    self.scheduler.step()
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    tprint_warning(f"Early stopping at epoch {epoch}")
                break
            
            # Progress logging
            if verbose and (epoch + 1) % 10 == 0:
                if val_loss is not None:
                    tprint_progress(epoch + 1, epochs, f"Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    tprint_progress(epoch + 1, epochs, f"Loss: {avg_loss:.6f}")
        
        # Restore best model
        if self.training_history['best_model_state'] is not None:
            self.model.load_state_dict(self.training_history['best_model_state'])
        
        if verbose:
            tprint_success(f"Training completed. Best loss: {best_loss:.6f}")
    
    def _evaluate_validation(
        self,
        validation_data: Tuple[np.ndarray, np.ndarray],
        time_points: torch.Tensor
    ) -> float:
        """Evaluate validation loss."""
        X_val, y_val = validation_data
        X_val = self._validate_input(X_val, "X_val")
        y_val = self._validate_input(y_val, "y_val")
        
        X_val_torch = self._to_torch(X_val)
        y_val_torch = self._to_torch(y_val)
        
        with torch.no_grad():
            initial_conditions = X_val_torch[:, 0, :]
            predicted = self.solve_ode(initial_conditions, time_points)
            val_loss = self.criterion(predicted, y_val_torch).item()
        
        return val_loss
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metrics: List[str] = ['mse', 'mae', 'mape']
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Input time series
            y: Target time series
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of computed metrics
        """
        X = self._validate_input(X, "X")
        y = self._validate_input(y, "y")
        
        # Make predictions
        predictions = self.predict(
            X[:, 0, :],  # Initial conditions
            time_horizon=1.0,
            num_steps=y.shape[1],
            return_numpy=True
        )
        
        # Flatten for metric computation
        y_flat = y.reshape(-1)
        pred_flat = predictions.reshape(-1)
        
        # Compute metrics
        results = {}
        
        for metric in metrics:
            if metric == 'mse':
                results[metric] = np.mean((y_flat - pred_flat) ** 2)
            elif metric == 'mae':
                results[metric] = np.mean(np.abs(y_flat - pred_flat))
            elif metric == 'mape':
                mask = y_flat != 0
                results[metric] = np.mean(np.abs((y_flat[mask] - pred_flat[mask]) / y_flat[mask])) * 100
            elif metric == 'rmse':
                results[metric] = np.sqrt(np.mean((y_flat - pred_flat) ** 2))
        
        self.performance_metrics.update(results)
        return results
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
        scoring: str = 'neg_mean_squared_error',
        **cv_params
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on the model.
        
        Args:
            X: Input data
            y: Target data
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric
            **cv_params: Additional cross-validation parameters
            
        Returns:
            Cross-validation results
        """
        if not ML_COMMON_AVAILABLE or perform_cross_validation is None:
            raise ImportError("ML Common utilities not available for cross-validation")
        
        def model_fit_score(X_train, y_train, X_test, y_test):
            """Fit model and return score."""
            # Create temporary model
            temp_model = NeuralODE(**self.config)
            temp_model.set_optimizer()
            
            # Fit model
            temp_model.fit(X_train, y_train, epochs=50, verbose=False)
            
            # Evaluate
            results = temp_model.evaluate(X_test, y_test)
            
            # Return appropriate score
            if scoring == 'neg_mean_squared_error':
                return -results['mse']
            elif scoring == 'neg_mean_absolute_error':
                return -results['mae']
            else:
                return results.get('mse', 0.0)
        
        # Perform cross-validation
        cv_results = perform_cross_validation(
            X, y, model_fit_score, cv_folds=cv_folds, **cv_params
        )
        
        return cv_results
    
    def hyperparameter_optimization(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Dict[str, List],
        cv_folds: int = 3,
        n_trials: int = 50,
        optimization_method: str = 'bayesian'
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization.
        
        Args:
            X: Input data
            y: Target data
            param_grid: Parameter grid for optimization
            cv_folds: Number of cross-validation folds
            n_trials: Number of optimization trials
            optimization_method: Optimization method ('grid', 'bayesian', 'random')
            
        Returns:
            Optimization results
        """
        if not ML_COMMON_AVAILABLE or HyperparameterOptimizer is None:
            raise ImportError("ML Common utilities not available for hyperparameter optimization")
        
        def objective_function(params):
            """Objective function for optimization."""
            # Update model configuration
            for key, value in params.items():
                if key in self.config:
                    self.config[key] = value
            
            # Create new model with updated config
            model = NeuralODE(**self.config)
            model.set_optimizer()
            
            # Perform cross-validation
            cv_results = model.cross_validate(X, y, cv_folds=cv_folds)
            
            # Return negative score for minimization
            return -cv_results['mean_score']
        
        # Create optimizer
        optimizer = HyperparameterOptimizer(
            param_grid=param_grid,
            optimization_method=optimization_method,
            n_trials=n_trials
        )
        
        # Perform optimization
        results = optimizer.optimize(objective_function)
        
        # Update model with best parameters
        for key, value in results['best_params'].items():
            if key in self.config:
                self.config[key] = value
        
        return results
    
    def save_model(self, filepath: str, include_data: bool = False) -> bool:
        """
        Save the model to file.
        
        Args:
            filepath: Path to save the model
            include_data: Whether to include training data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_data = {
                'config': self.config,
                'model_state': self.model.state_dict(),
                'training_history': self.training_history,
                'performance_metrics': self.performance_metrics
            }
            
            if include_data and hasattr(self, 'training_data'):
                model_data['training_data'] = self.training_data
            
            # Use serializer if available
            if self.serializer:
                success = self.serializer.save(model_data, filepath)
            else:
                # Fallback to pickle
                with open(filepath, 'wb') as f:
                    pickle.dump(model_data, f)
                success = True
            
            if success and self.config['enable_logging']:
                tprint_success(f"Model saved to {filepath}")
            
            return success
            
        except Exception as e:
            if self.config['enable_logging']:
                tprint_error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load model from file.
        
        Args:
            filepath: Path to the model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not safe_file_exists(filepath) if UTILITIES_AVAILABLE else not Path(filepath).exists():
                raise FileNotFoundError(f"Model file not found: {filepath}")
            
            # Use serializer if available
            if self.serializer:
                model_data = self.serializer.load(filepath)
            else:
                # Fallback to pickle
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
            
            if model_data is None:
                raise ValueError("Failed to load model data")
            
            # Restore configuration
            self.config.update(model_data.get('config', {}))
            
            # Restore model state
            if 'model_state' in model_data:
                self.model.load_state_dict(model_data['model_state'])
            
            # Restore training history
            if 'training_history' in model_data:
                self.training_history.update(model_data['training_history'])
            
            # Restore performance metrics
            if 'performance_metrics' in model_data:
                self.performance_metrics.update(model_data['performance_metrics'])
            
            # Restore training data if present
            if 'training_data' in model_data:
                self.training_data = model_data['training_data']
            
            if self.config['enable_logging']:
                tprint_success(f"Model loaded from {filepath}")
            
            return True
            
        except Exception as e:
            if self.config['enable_logging']:
                tprint_error(f"Failed to load model: {e}")
            return False
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model summary.
        
        Returns:
            Dictionary containing model summary information
        """
        summary = {
            'config': self.config.copy(),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'training_history': self.training_history.copy(),
            'performance_metrics': self.performance_metrics.copy(),
            'hardware_optimization': self.hardware_optimized if hasattr(self, 'hardware_optimized') else False,
            'device': str(self.model.device),
            'utilities_available': UTILITIES_AVAILABLE,
            'torch_available': TORCH_AVAILABLE
        }
        
        return summary
    
    def __repr__(self) -> str:
        """String representation of the NeuralODE."""
        return (f"NeuralODE(input_dim={self.config['input_dim']}, "
                f"hidden_dims={self.config['hidden_dims']}, "
                f"output_dim={self.config['output_dim']}, "
                f"device={self.model.device})")


# Utility functions for Neural ODEs

def create_neural_ode_from_config(config: Dict[str, Any]) -> NeuralODE:
    """
    Create NeuralODE instance from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        NeuralODE instance
    """
    return NeuralODE(**config)


def compare_neural_ode_models(
    models: List[NeuralODE],
    X_test: np.ndarray,
    y_test: np.ndarray,
    metrics: List[str] = ['mse', 'mae', 'rmse']
) -> pd.DataFrame:
    """
    Compare multiple NeuralODE models.
    
    Args:
        models: List of NeuralODE models
        X_test: Test input data
        y_test: Test target data
        metrics: List of metrics to compute
        
    Returns:
        DataFrame with comparison results
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("Pandas is required for model comparison")
    
    results = []
    
    for i, model in enumerate(models):
        model_results = {'model_id': i}
        model_results.update(model.evaluate(X_test, y_test, metrics))
        results.append(model_results)
    
    return pd.DataFrame(results)


def ensemble_neural_odes(
    models: List[NeuralODE],
    X: np.ndarray,
    time_horizon: float = 1.0,
    num_steps: int = 100,
    aggregation_method: str = 'mean'
) -> np.ndarray:
    """
    Create ensemble predictions from multiple NeuralODE models.
    
    Args:
        models: List of NeuralODE models
        X: Input data
        time_horizon: Time horizon for prediction
        num_steps: Number of time steps
        aggregation_method: Aggregation method ('mean', 'median', 'weighted')
        
    Returns:
        Ensemble predictions
    """
    predictions = []
    
    for model in models:
        pred = model.predict(
            X[:, 0, :],  # Initial conditions
            time_horizon=time_horizon,
            num_steps=num_steps,
            return_numpy=True
        )
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    if aggregation_method == 'mean':
        return np.mean(predictions, axis=0)
    elif aggregation_method == 'median':
        return np.median(predictions, axis=0)
    elif aggregation_method == 'weighted':
        # Simple equal weighting - can be extended
        return np.mean(predictions, axis=0)
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation_method}")


# Export main class and utilities
__all__ = [
    'NeuralODE',
    'NeuralODEModel',
    'create_neural_ode_from_config',
    'compare_neural_ode_models',
    'ensemble_neural_odes'
]
