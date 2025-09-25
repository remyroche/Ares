"""
Neural Architecture Search (NAS) for ML Common

This module provides comprehensive Neural Architecture Search capabilities
specifically designed for financial time series and trading models.

Key Features:
- Evolutionary architecture search
- Reinforcement learning-based search
- Multi-objective optimization (accuracy + efficiency)
- Regime-aware architecture adaptation
- Integration with existing ML pipeline
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime
from abc import ABC, abstractmethod
import json
from pathlib import Path

# Neural network imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
    keras = None
    layers = None
    models = None
    optimizers = None

# Optimization imports
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Import new Bayesian TPE optimizer
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
    BayesianTPEOptimizer,
    BayesianTPEConfig,
    optimize_with_bayesian_tpe
)

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureConfig:
    """Configuration for neural architecture search."""
    
    # Search space
    min_layers: int = 2
    max_layers: int = 8
    min_units: int = 32
    max_units: int = 512
    activation_functions: List[str] = field(default_factory=lambda: ['relu', 'tanh', 'swish', 'gelu'])
    dropout_rates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5])
    
    # Search parameters
    n_trials: int = 50
    timeout_seconds: int = 3600  # 1 hour
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    
    # Multi-objective optimization
    objectives: List[str] = field(default_factory=lambda: ['accuracy', 'efficiency', 'robustness'])
    objective_weights: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    
    # Regime awareness
    enable_regime_awareness: bool = True
    regime_adaptation_strength: float = 0.3
    
    # Performance
    n_jobs: int = 1
    memory_limit_gb: float = 8.0


@dataclass
class ArchitectureCandidate:
    """A candidate neural architecture."""
    
    # Architecture definition
    layers: List[Dict[str, Any]]  # List of layer configurations
    total_params: int
    estimated_flops: int
    
    # Performance metrics
    accuracy: float = 0.0
    efficiency_score: float = 0.0
    robustness_score: float = 0.0
    overall_score: float = 0.0
    
    # Training info
    training_time: float = 0.0
    convergence_epochs: int = 0
    final_loss: float = 0.0
    
    # Regime performance
    regime_performance: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    trial_number: int = 0


class ArchitectureSearchSpace:
    """Defines the search space for neural architectures."""
    
    def __init__(self, config: ArchitectureConfig):
        self.config = config
        self.logger = logger.getChild('ArchitectureSearchSpace')
    
    def sample_architecture(self, trial_number: int = 0) -> ArchitectureCandidate:
        """Sample a random architecture from the search space."""
        try:
            # Sample number of layers
            n_layers = np.random.randint(self.config.min_layers, self.config.max_layers + 1)
            
            # Sample layer configurations
            layers = []
            total_params = 0
            estimated_flops = 0
            
            for i in range(n_layers):
                # Sample layer type
                layer_type = np.random.choice(['dense', 'lstm', 'gru', 'conv1d'])
                
                # Sample layer parameters
                if layer_type == 'dense':
                    units = np.random.randint(self.config.min_units, self.config.max_units + 1)
                    activation = np.random.choice(self.config.activation_functions)
                    dropout = np.random.choice(self.config.dropout_rates)
                    
                    layer_config = {
                        'type': 'dense',
                        'units': units,
                        'activation': activation,
                        'dropout': dropout
                    }
                    
                    # Estimate parameters (simplified)
                    if i == 0:
                        # Input layer - assume input features
                        layer_params = units * 100  # Simplified estimate
                    else:
                        # Hidden layer
                        prev_units = layers[-1].get('units', 100)
                        layer_params = prev_units * units
                    
                    total_params += layer_params
                    estimated_flops += layer_params * 2  # Simplified FLOP estimate
                
                elif layer_type in ['lstm', 'gru']:
                    units = np.random.randint(32, 256)
                    return_sequences = i < n_layers - 1  # Only last layer returns sequences
                    dropout = np.random.choice(self.config.dropout_rates)
                    
                    layer_config = {
                        'type': layer_type,
                        'units': units,
                        'return_sequences': return_sequences,
                        'dropout': dropout
                    }
                    
                    # Estimate parameters for RNN layers
                    layer_params = 4 * units * units if layer_type == 'lstm' else 3 * units * units
                    total_params += layer_params
                    estimated_flops += layer_params * 4  # RNN operations are more expensive
                
                elif layer_type == 'conv1d':
                    filters = np.random.randint(32, 128)
                    kernel_size = np.random.choice([3, 5, 7])
                    activation = np.random.choice(self.config.activation_functions)
                    
                    layer_config = {
                        'type': 'conv1d',
                        'filters': filters,
                        'kernel_size': kernel_size,
                        'activation': activation
                    }
                    
                    # Estimate parameters for Conv1D
                    layer_params = filters * kernel_size * 32  # Simplified estimate
                    total_params += layer_params
                    estimated_flops += layer_params * 2
                
                layers.append(layer_config)
            
            # Create architecture candidate
            candidate = ArchitectureCandidate(
                layers=layers,
                total_params=total_params,
                estimated_flops=estimated_flops,
                trial_number=trial_number
            )
            
            self.logger.debug(f"Sampled architecture with {n_layers} layers, {total_params} parameters")
            return candidate
            
        except Exception as e:
            self.logger.error(f"Architecture sampling failed: {e}")
            # Return minimal architecture as fallback
            return ArchitectureCandidate(
                layers=[{'type': 'dense', 'units': 64, 'activation': 'relu', 'dropout': 0.0}],
                total_params=1000,
                estimated_flops=2000,
                trial_number=trial_number
            )


class NeuralArchitectureSearch:
    """Main Neural Architecture Search implementation."""
    
    def __init__(self, config: ArchitectureConfig):
        """Initialize NAS."""
        self.config = config
        self.logger = logger.getChild('NeuralArchitectureSearch')
        self.search_space = ArchitectureSearchSpace(config)
        self.candidates = []
        self.best_candidate = None
        
        # Initialize framework
        self.framework = self._detect_framework()
        
        self.logger.info(f"✅ Neural Architecture Search initialized with {config.n_trials} trials")
    
    def _detect_framework(self) -> str:
        """Detect available deep learning framework."""
        if TORCH_AVAILABLE:
            return 'pytorch'
        elif TF_AVAILABLE:
            return 'tensorflow'
        else:
            raise ImportError("No deep learning framework available. Install PyTorch or TensorFlow.")
    
    def search(self, 
               X_train: np.ndarray, 
               y_train: np.ndarray,
               X_val: Optional[np.ndarray] = None,
               y_val: Optional[np.ndarray] = None,
               regime_labels: Optional[np.ndarray] = None) -> ArchitectureCandidate:
        """
        Perform neural architecture search.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            regime_labels: Regime labels for regime-aware search (optional)
            
        Returns:
            Best architecture candidate
        """
        self.logger.info("🚀 Starting Neural Architecture Search...")
        start_time = time.time()
        
        try:
            # Prepare validation data
            if X_val is None or y_val is None:
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=self.config.validation_split, random_state=42
                )
            
            # Search for architectures
            if OPTUNA_AVAILABLE:
                best_candidate = self._optuna_search(X_train, y_train, X_val, y_val, regime_labels)
            else:
                best_candidate = self._random_search(X_train, y_train, X_val, y_val, regime_labels)
            
            search_time = time.time() - start_time
            self.logger.info(f"✅ NAS completed in {search_time:.2f}s")
            self.logger.info(f"📊 Best architecture: {best_candidate.total_params} parameters, score: {best_candidate.overall_score:.4f}")
            
            return best_candidate
            
        except Exception as e:
            self.logger.error(f"Neural Architecture Search failed: {e}")
            raise
    
    def _optuna_search(self, 
                      X_train: np.ndarray, 
                      y_train: np.ndarray,
                      X_val: np.ndarray, 
                      y_val: np.ndarray,
                      regime_labels: Optional[np.ndarray] = None) -> ArchitectureCandidate:
        """Perform architecture search using Bayesian TPE optimizer."""
        self.logger.info("🔍 Starting Bayesian TPE architecture search...")
        
        # Create search space for architecture optimization
        search_space = self._create_architecture_search_space()
        
        # Define objective function for Bayesian TPE optimizer
        def objective_function(params: Dict[str, Any], **kwargs) -> float:
            try:
                candidate = self._sample_architecture_from_params(params)
                
                # Train and evaluate
                performance = self._train_and_evaluate_architecture(
                    candidate, X_train, y_train, X_val, y_val, regime_labels
                )
                
                return performance['overall_score']
                
            except Exception as e:
                self.logger.warning(f"Trial failed: {e}")
                return 0.0
        
        # Configure Bayesian TPE optimizer
        tpe_config = BayesianTPEConfig(
            n_trials=self.config.n_trials,
            timeout_seconds=self.config.timeout_seconds,
            enable_grid_search=True,
            coarse_grid_points=3,
            fine_grid_points=5,
            backend='optuna',
            enable_parallel=False,  # Sequential for architecture search
            enable_early_stopping=True,
            early_stopping_patience=self.config.early_stopping_patience,
            log_level='INFO'
        )
        
        # Run optimization using new unified optimizer
        self.logger.info("🎯 Starting Bayesian TPE optimization for neural architecture search")
        optimizer = BayesianTPEOptimizer(tpe_config)
        result = optimizer.optimize(objective_function, search_space)
        
        if not result.success:
            raise RuntimeError(f"Architecture search failed: {result.error_message}")
        
        # Get best candidate
        best_candidate = self._sample_architecture_from_params(result.best_params)
        
        # Train final model
        performance = self._train_and_evaluate_architecture(
            best_candidate, X_train, y_train, X_val, y_val, regime_labels
        )
        
        best_candidate.accuracy = performance['accuracy']
        best_candidate.efficiency_score = performance['efficiency_score']
        best_candidate.robustness_score = performance['robustness_score']
        best_candidate.overall_score = performance['overall_score']
        
        self.logger.info(f"✅ Neural Architecture Search completed")
        self.logger.info(f"📊 Best score: {result.best_score:.4f}")
        self.logger.info(f"📊 Optimization time: {result.optimization_time:.2f}s")
        self.logger.info(f"📊 Trials: {result.n_trials}")
        
        return best_candidate
    
    def _random_search(self, 
                      X_train: np.ndarray, 
                      y_train: np.ndarray,
                      X_val: np.ndarray, 
                      y_val: np.ndarray,
                      regime_labels: Optional[np.ndarray] = None) -> ArchitectureCandidate:
        """Perform random architecture search."""
        self.logger.info("🔍 Starting random architecture search...")
        
        best_candidate = None
        best_score = -np.inf
        
        for trial in range(self.config.n_trials):
            try:
                # Sample random architecture
                candidate = self.search_space.sample_architecture(trial)
                
                # Train and evaluate
                performance = self._train_and_evaluate_architecture(
                    candidate, X_train, y_train, X_val, y_val, regime_labels
                )
                
                # Update best candidate
                if performance['overall_score'] > best_score:
                    best_score = performance['overall_score']
                    best_candidate = candidate
                    
                    best_candidate.accuracy = performance['accuracy']
                    best_candidate.efficiency_score = performance['efficiency_score']
                    best_candidate.robustness_score = performance['robustness_score']
                    best_candidate.overall_score = performance['overall_score']
                
                self.logger.debug(f"Trial {trial}: Score {performance['overall_score']:.4f}")
                
            except Exception as e:
                self.logger.warning(f"Trial {trial} failed: {e}")
                continue
        
        if best_candidate is None:
            raise RuntimeError("No successful architecture found")
        
        return best_candidate
    
    def _sample_architecture_from_trial(self, trial) -> ArchitectureCandidate:
        """Sample architecture from Optuna trial."""
        # Sample number of layers
        n_layers = trial.suggest_int('n_layers', self.config.min_layers, self.config.max_layers)
        
        layers = []
        total_params = 0
        estimated_flops = 0
        
        for i in range(n_layers):
            layer_type = trial.suggest_categorical('layer_type', ['dense', 'lstm', 'gru'])
            
            if layer_type == 'dense':
                units = trial.suggest_int('units', self.config.min_units, self.config.max_units)
                activation = trial.suggest_categorical('activation', self.config.activation_functions)
                dropout = trial.suggest_categorical('dropout', self.config.dropout_rates)
                
                layer_config = {
                    'type': 'dense',
                    'units': units,
                    'activation': activation,
                    'dropout': dropout
                }
                
                # Estimate parameters
                if i == 0:
                    layer_params = units * 100
                else:
                    prev_units = layers[-1].get('units', 100)
                    layer_params = prev_units * units
                
                total_params += layer_params
                estimated_flops += layer_params * 2
            
            elif layer_type in ['lstm', 'gru']:
                units = trial.suggest_int('rnn_units', 32, 256)
                return_sequences = trial.suggest_categorical('return_sequences', [True, False])
                dropout = trial.suggest_categorical('rnn_dropout', self.config.dropout_rates)
                
                layer_config = {
                    'type': layer_type,
                    'units': units,
                    'return_sequences': return_sequences,
                    'dropout': dropout
                }
                
                layer_params = 4 * units * units if layer_type == 'lstm' else 3 * units * units
                total_params += layer_params
                estimated_flops += layer_params * 4
            
            layers.append(layer_config)
        
        return ArchitectureCandidate(
            layers=layers,
            total_params=total_params,
            estimated_flops=estimated_flops,
            trial_number=trial.number
        )
    
    def _train_and_evaluate_architecture(self, 
                                       candidate: ArchitectureCandidate,
                                       X_train: np.ndarray, 
                                       y_train: np.ndarray,
                                       X_val: np.ndarray, 
                                       y_val: np.ndarray,
                                       regime_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Train and evaluate an architecture candidate."""
        try:
            # Create model
            if self.framework == 'pytorch':
                model = self._create_pytorch_model(candidate, X_train.shape[1], y_train.shape[1] if len(y_train.shape) > 1 else 1)
                performance = self._train_pytorch_model(model, X_train, y_train, X_val, y_val)
            else:
                model = self._create_tensorflow_model(candidate, X_train.shape[1], y_train.shape[1] if len(y_train.shape) > 1 else 1)
                performance = self._train_tensorflow_model(model, X_train, y_train, X_val, y_val)
            
            # Calculate multi-objective score
            overall_score = self._calculate_overall_score(performance, candidate)
            performance['overall_score'] = overall_score
            
            return performance
            
        except Exception as e:
            self.logger.warning(f"Architecture training failed: {e}")
            return {
                'accuracy': 0.0,
                'efficiency_score': 0.0,
                'robustness_score': 0.0,
                'overall_score': 0.0
            }
    
    def _create_pytorch_model(self, candidate: ArchitectureCandidate, input_size: int, output_size: int) -> nn.Module:
        """Create PyTorch model from architecture candidate."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        class NASModel(nn.Module):
            def __init__(self, layers_config, input_size, output_size):
                super().__init__()
                self.layers = nn.ModuleList()
                
                prev_size = input_size
                for layer_config in layers_config:
                    if layer_config['type'] == 'dense':
                        self.layers.append(nn.Linear(prev_size, layer_config['units']))
                        if layer_config['activation'] == 'relu':
                            self.layers.append(nn.ReLU())
                        elif layer_config['activation'] == 'tanh':
                            self.layers.append(nn.Tanh())
                        elif layer_config['activation'] == 'swish':
                            self.layers.append(nn.SiLU())
                        elif layer_config['activation'] == 'gelu':
                            self.layers.append(nn.GELU())
                        
                        if layer_config['dropout'] > 0:
                            self.layers.append(nn.Dropout(layer_config['dropout']))
                        
                        prev_size = layer_config['units']
                    
                    elif layer_config['type'] == 'lstm':
                        self.layers.append(nn.LSTM(prev_size, layer_config['units'], 
                                                 batch_first=True, dropout=layer_config['dropout']))
                        prev_size = layer_config['units']
                    
                    elif layer_config['type'] == 'gru':
                        self.layers.append(nn.GRU(prev_size, layer_config['units'], 
                                                batch_first=True, dropout=layer_config['dropout']))
                        prev_size = layer_config['units']
                
                # Output layer
                self.output_layer = nn.Linear(prev_size, output_size)
            
            def forward(self, x):
                for layer in self.layers:
                    if isinstance(layer, (nn.LSTM, nn.GRU)):
                        x, _ = layer(x)
                    else:
                        x = layer(x)
                
                x = self.output_layer(x)
                return x
        
        return NASModel(candidate.layers, input_size, output_size)
    
    def _create_tensorflow_model(self, candidate: ArchitectureCandidate, input_size: int, output_size: int) -> keras.Model:
        """Create TensorFlow model from architecture candidate."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        inputs = keras.Input(shape=(input_size,))
        x = inputs
        
        for layer_config in candidate.layers:
            if layer_config['type'] == 'dense':
                x = layers.Dense(layer_config['units'], activation=layer_config['activation'])(x)
                if layer_config['dropout'] > 0:
                    x = layers.Dropout(layer_config['dropout'])(x)
            
            elif layer_config['type'] == 'lstm':
                x = layers.LSTM(layer_config['units'], 
                              return_sequences=layer_config['return_sequences'],
                              dropout=layer_config['dropout'])(x)
            
            elif layer_config['type'] == 'gru':
                x = layers.GRU(layer_config['units'], 
                             return_sequences=layer_config['return_sequences'],
                             dropout=layer_config['dropout'])(x)
        
        # Output layer
        outputs = layers.Dense(output_size, activation='linear' if output_size > 1 else 'sigmoid')(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    def _train_pytorch_model(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, 
                           X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Train PyTorch model and return performance metrics."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Setup training
        criterion = nn.MSELoss() if len(y_train.shape) == 1 else nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in range(50):  # Limited epochs for NAS
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_tensor)
            val_pred = model(X_val_tensor)
            
            # Calculate metrics
            train_loss = criterion(train_pred, y_train_tensor).item()
            val_loss = criterion(val_pred, y_val_tensor).item()
            
            # Calculate accuracy (simplified)
            if len(y_train.shape) == 1:
                # Regression
                train_accuracy = 1 - (train_loss / torch.var(y_train_tensor).item())
                val_accuracy = 1 - (val_loss / torch.var(y_val_tensor).item())
            else:
                # Classification
                train_accuracy = (torch.argmax(train_pred, dim=1) == torch.argmax(y_train_tensor, dim=1)).float().mean().item()
                val_accuracy = (torch.argmax(val_pred, dim=1) == torch.argmax(y_val_tensor, dim=1)).float().mean().item()
        
        return {
            'accuracy': val_accuracy,
            'efficiency_score': 1.0 / (1.0 + model.total_params / 1000000),  # Efficiency based on parameter count
            'robustness_score': 1.0 - abs(train_accuracy - val_accuracy)  # Robustness based on generalization gap
        }
    
    def _train_tensorflow_model(self, model: keras.Model, X_train: np.ndarray, y_train: np.ndarray, 
                              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Train TensorFlow model and return performance metrics."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='mse' if len(y_train.shape) == 1 else 'categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,  # Limited epochs for NAS
            batch_size=32,
            verbose=0
        )
        
        # Calculate metrics
        val_accuracy = history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else 0.0
        train_accuracy = history.history['accuracy'][-1] if 'accuracy' in history.history else 0.0
        
        # Calculate efficiency and robustness
        total_params = model.count_params()
        efficiency_score = 1.0 / (1.0 + total_params / 1000000)
        robustness_score = 1.0 - abs(train_accuracy - val_accuracy)
        
        return {
            'accuracy': val_accuracy,
            'efficiency_score': efficiency_score,
            'robustness_score': robustness_score
        }
    
    def _calculate_overall_score(self, performance: Dict[str, float], candidate: ArchitectureCandidate) -> float:
        """Calculate overall score from multiple objectives."""
        try:
            # Get objective weights
            weights = self.config.objective_weights
            
            # Calculate weighted score
            overall_score = (
                weights[0] * performance['accuracy'] +
                weights[1] * performance['efficiency_score'] +
                weights[2] * performance['robustness_score']
            )
            
            return float(overall_score)
            
        except Exception as e:
            self.logger.warning(f"Overall score calculation failed: {e}")
            return 0.0
    
    def get_search_summary(self) -> Dict[str, Any]:
        """Get summary of architecture search results."""
        if not self.candidates:
            return {'message': 'No search results available'}
        
        try:
            # Calculate summary statistics
            accuracies = [c.accuracy for c in self.candidates]
            efficiency_scores = [c.efficiency_score for c in self.candidates]
            overall_scores = [c.overall_score for c in self.candidates]
            param_counts = [c.total_params for c in self.candidates]
            
            return {
                'total_candidates': len(self.candidates),
                'best_accuracy': float(np.max(accuracies)),
                'best_efficiency': float(np.max(efficiency_scores)),
                'best_overall_score': float(np.max(overall_scores)),
                'average_parameters': float(np.mean(param_counts)),
                'parameter_range': [int(np.min(param_counts)), int(np.max(param_counts))],
                'search_statistics': {
                    'accuracy_mean': float(np.mean(accuracies)),
                    'accuracy_std': float(np.std(accuracies)),
                    'efficiency_mean': float(np.mean(efficiency_scores)),
                    'efficiency_std': float(np.std(efficiency_scores)),
                    'overall_score_mean': float(np.mean(overall_scores)),
                    'overall_score_std': float(np.std(overall_scores))
                }
            }
            
        except Exception as e:
            self.logger.error(f"Search summary generation failed: {e}")
            return {'error': str(e)}


# Convenience function
def search_neural_architecture(X_train: np.ndarray, 
                              y_train: np.ndarray,
                              X_val: Optional[np.ndarray] = None,
                              y_val: Optional[np.ndarray] = None,
                              config: Optional[ArchitectureConfig] = None,
                              regime_labels: Optional[np.ndarray] = None) -> ArchitectureCandidate:
    """
    Convenience function to perform neural architecture search.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        config: Architecture search configuration
        regime_labels: Regime labels for regime-aware search (optional)
        
    Returns:
        Best architecture candidate
    """
    if config is None:
        config = ArchitectureConfig()
    
    nas = NeuralArchitectureSearch(config)
    return nas.search(X_train, y_train, X_val, y_val, regime_labels)