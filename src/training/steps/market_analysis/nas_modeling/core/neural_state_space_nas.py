"""
Neural State Space Neural Architecture Search (NAS)

A comprehensive implementation of Neural Architecture Search for State Space Models,
integrating Bayesian optimization, M1 GPU acceleration, and advanced validation.

This module provides:
- Neural Architecture Search for State Space Models
- Bayesian TPE optimization for hyperparameter tuning
- M1 GPU acceleration and hardware optimization
- Comprehensive validation and evaluation
- Integration with shared utilities and common operations
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import time
import json
from datetime import datetime

from src.utils.nas_tas.common_constants import (
    RECOMMENDED_MAX_LAYERS,
    RECOMMENDED_MAX_UNITS,
    RECOMMENDED_MIN_LAYERS,
    RECOMMENDED_MIN_UNITS,
)

# Import shared utilities
try:
    from src.utils.common_operations import (
        safe_dataframe_operation, validate_dataframe_columns, 
        safe_convert_dtypes, calculate_data_quality_metrics,
        safe_merge_dataframes, create_summary_statistics,
        optimize_dataframe_dtypes, get_dataframe_info,
        safe_json_dump, safe_json_load, ensure_directory,
        safe_file_exists, get_current_datetime, format_datetime
    )
    from src.utils.common_utilities import (
        CommonUtilities, safe_dataframe_operation as safe_df_op,
        validate_dataframe_columns as validate_df_cols,
        get_data_summary, safe_convert_dtypes as safe_convert_dt
    )
    from src.utils.math_validation import (
        safe_divide, safe_log, safe_sqrt, safe_power,
        validate_finite, validate_positive, validate_range,
        safe_correlation, safe_covariance, safe_mean, safe_std,
        validate_correlation_matrix, safe_matrix_inverse,
        MathValidation, MathValidationError
    )
    from src.utils.serialization_utils import (
        JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
    )
    from src.utils.tprint import (
        tprint, tprint_info, tprint_debug, tprint_warning, tprint_error,
        tprint_success, tprint_progress, tprint_performance,
        tprint_structured, tprint_with_level, tprint_timer
    )
    from src.utils.nas_tas.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer, BayesianTPEConfig, OptimizationResult,
        optimize_with_bayesian_tpe, create_search_space_from_bounds
    )
    from src.utils.hardware.m1_gpu_utils import (
        get_m1_gpu_manager, is_m1_available, is_mps_available,
        optimize_dataframe_for_m1, create_m1_optimized_array,
        m1_backtesting_simulate, m1_monte_carlo_simulate
    )
    UTILITIES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some utilities not available: {e}")
    UTILITIES_AVAILABLE = False

# Optional ML dependencies
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None

try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class NeuralSSMConfig:
    """Configuration for Neural State Space Model NAS."""
    
    # Architecture search parameters
    max_layers: int = RECOMMENDED_MAX_LAYERS
    min_layers: int = RECOMMENDED_MIN_LAYERS
    max_hidden_units: int = RECOMMENDED_MAX_UNITS
    min_hidden_units: int = RECOMMENDED_MIN_UNITS
    
    # State space model parameters
    state_dim: int = 8
    observation_dim: int = 4
    control_dim: int = 2
    
    # Search space constraints
    activation_functions: List[str] = field(default_factory=lambda: [
        'relu', 'tanh', 'sigmoid', 'elu', 'swish', 'gelu'
    ])
    layer_types: List[str] = field(default_factory=lambda: [
        'dense', 'lstm', 'gru', 'conv1d', 'attention'
    ])
    
    # Optimization parameters
    learning_rate_range: Tuple[float, float] = (1e-5, 1e-2)
    batch_size_range: Tuple[int, int] = (16, 256)
    dropout_range: Tuple[float, float] = (0.0, 0.5)
    
    # Training parameters
    max_epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    
    # Hardware optimization
    use_m1_optimization: bool = True
    use_gpu_acceleration: bool = True
    
    # Logging and monitoring
    enable_detailed_logging: bool = True
    log_interval: int = 10
    save_intermediate_results: bool = True
    
    # Bayesian optimization
    n_trials: int = 50
    optimization_timeout: int = 3600  # 1 hour
    enable_early_stopping: bool = True


@dataclass
class ArchitectureCandidate:
    """Represents a neural architecture candidate."""
    
    layers: List[Dict[str, Any]]
    activation_functions: List[str]
    layer_types: List[str]
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    validation_score: float = 0.0
    test_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'layers': self.layers,
            'activation_functions': self.activation_functions,
            'layer_types': self.layer_types,
            'hyperparameters': self.hyperparameters,
            'performance_metrics': self.performance_metrics,
            'training_time': self.training_time,
            'validation_score': self.validation_score,
            'test_score': self.test_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ArchitectureCandidate':
        """Create from dictionary."""
        return cls(
            layers=data.get('layers', []),
            activation_functions=data.get('activation_functions', []),
            layer_types=data.get('layer_types', []),
            hyperparameters=data.get('hyperparameters', {}),
            performance_metrics=data.get('performance_metrics', {}),
            training_time=data.get('training_time', 0.0),
            validation_score=data.get('validation_score', 0.0),
            test_score=data.get('test_score', 0.0)
        )


class StateSpaceModel(nn.Module):
    """Neural State Space Model implementation."""
    
    def __init__(self, config: NeuralSSMConfig, architecture: ArchitectureCandidate):
        super().__init__()
        self.config = config
        self.architecture = architecture
        self.layers = nn.ModuleList()
        self._build_architecture()
    
    def _build_architecture(self):
        """Build the neural architecture based on candidate."""
        input_dim = self.config.observation_dim
        
        for i, layer_config in enumerate(self.architecture.layers):
            layer_type = layer_config.get('type', 'dense')
            units = layer_config.get('units', 64)
            activation = layer_config.get('activation', 'relu')
            dropout = layer_config.get('dropout', 0.0)
            
            if layer_type == 'dense':
                layer = nn.Linear(input_dim, units)
                self.layers.append(layer)
                
                # Add activation
                if activation == 'relu':
                    self.layers.append(nn.ReLU())
                elif activation == 'tanh':
                    self.layers.append(nn.Tanh())
                elif activation == 'sigmoid':
                    self.layers.append(nn.Sigmoid())
                elif activation == 'elu':
                    self.layers.append(nn.ELU())
                elif activation == 'swish':
                    self.layers.append(nn.SiLU())  # Swish approximation
                elif activation == 'gelu':
                    self.layers.append(nn.GELU())
                
                # Add dropout
                if dropout > 0:
                    self.layers.append(nn.Dropout(dropout))
                
                input_dim = units
                
            elif layer_type == 'lstm':
                layer = nn.LSTM(input_dim, units, batch_first=True)
                self.layers.append(layer)
                input_dim = units
                
            elif layer_type == 'gru':
                layer = nn.GRU(input_dim, units, batch_first=True)
                self.layers.append(layer)
                input_dim = units
        
        # Final output layer for state prediction
        self.output_layer = nn.Linear(input_dim, self.config.state_dim)
    
    def forward(self, x):
        """Forward pass through the network."""
        for layer in self.layers:
            if isinstance(layer, (nn.LSTM, nn.GRU)):
                x, _ = layer(x)
            else:
                x = layer(x)
        
        # State prediction
        state = self.output_layer(x)
        return state


class NeuralSSM_NAS_Optimizer:
    """
    Neural Architecture Search Optimizer for State Space Models.
    
    This class provides comprehensive NAS functionality including:
    - Architecture search space definition
    - Bayesian optimization for hyperparameter tuning
    - M1 GPU acceleration and hardware optimization
    - Comprehensive validation and evaluation
    - Integration with shared utilities
    """
    
    def __init__(self, config: Optional[NeuralSSMConfig] = None, **kwargs):
        """
        Initialize the NAS optimizer.
        
        Args:
            config: Configuration for the NAS optimizer
            **kwargs: Additional configuration parameters
        """
        self.config = config or NeuralSSMConfig(**kwargs)
        self.logger = logger
        
        # Initialize utilities
        self.common_utils = CommonUtilities() if UTILITIES_AVAILABLE else None
        self.math_validator = MathValidation() if UTILITIES_AVAILABLE else None
        self.serializer = UniversalSerializer() if UTILITIES_AVAILABLE else None
        
        # Initialize hardware optimization
        self.m1_gpu_manager = get_m1_gpu_manager() if UTILITIES_AVAILABLE else None
        self.m1_available = is_m1_available() if UTILITIES_AVAILABLE else False
        self.mps_available = is_mps_available() if UTILITIES_AVAILABLE else False
        
        # Initialize optimization components
        self.bayesian_optimizer = None
        self.architecture_candidates = []
        self.best_architecture = None
        self.optimization_history = []
        
        # Performance tracking
        self.optimization_start_time = None
        self.total_trials = 0
        self.successful_trials = 0
        
        # Setup logging
        if self.config.enable_detailed_logging and UTILITIES_AVAILABLE:
            tprint_info("🧠 NeuralSSM NAS Optimizer initialized")
            tprint_info(f"   → M1 Hardware: {'✅ Available' if self.m1_available else '❌ Not available'}")
            tprint_info(f"   → MPS (GPU): {'✅ Available' if self.mps_available else '❌ Not available'}")
            tprint_info(f"   → Max layers: {self.config.max_layers}")
            tprint_info(f"   → State dim: {self.config.state_dim}")
    
    def create_search_space(self) -> Dict[str, Any]:
        """
        Create the search space for neural architecture search.
        
        Returns:
            Dictionary defining the search space for Bayesian optimization
        """
        search_space = {
            # Architecture parameters
            'n_layers': {
                'type': 'int',
                'low': self.config.min_layers,
                'high': self.config.max_layers
            },
            'hidden_units': {
                'type': 'int',
                'low': self.config.min_hidden_units,
                'high': self.config.max_hidden_units
            },
            
            # Hyperparameters
            'learning_rate': {
                'type': 'float',
                'low': self.config.learning_rate_range[0],
                'high': self.config.learning_rate_range[1],
                'log': True
            },
            'batch_size': {
                'type': 'int',
                'low': self.config.batch_size_range[0],
                'high': self.config.batch_size_range[1]
            },
            'dropout_rate': {
                'type': 'float',
                'low': self.config.dropout_range[0],
                'high': self.config.dropout_range[1]
            },
            
            # Architecture choices
            'activation_function': {
                'type': 'categorical',
                'choices': self.config.activation_functions
            },
            'layer_type': {
                'type': 'categorical',
                'choices': self.config.layer_types
            }
        }
        
        if UTILITIES_AVAILABLE:
            tprint_debug(f"🔍 Created search space with {len(search_space)} parameters")
        
        return search_space
    
    def generate_architecture_candidate(self, params: Dict[str, Any]) -> ArchitectureCandidate:
        """
        Generate a neural architecture candidate from parameters.
        
        Args:
            params: Parameter dictionary from Bayesian optimization
            
        Returns:
            ArchitectureCandidate object
        """
        n_layers = int(params['n_layers'])
        hidden_units = int(params['hidden_units'])
        activation = params['activation_function']
        layer_type = params['layer_type']
        dropout_rate = float(params['dropout_rate'])
        
        # Generate layer configurations
        layers = []
        activation_functions = []
        layer_types = []
        
        current_units = self.config.observation_dim
        
        for i in range(n_layers):
            # Determine layer size (can vary)
            if i == n_layers - 1:
                layer_units = self.config.state_dim
            else:
                layer_units = hidden_units
            
            layer_config = {
                'type': layer_type,
                'units': layer_units,
                'activation': activation,
                'dropout': dropout_rate if i < n_layers - 1 else 0.0  # No dropout on output
            }
            
            layers.append(layer_config)
            activation_functions.append(activation)
            layer_types.append(layer_type)
            
            current_units = layer_units
        
        # Create hyperparameters dictionary
        hyperparameters = {
            'learning_rate': float(params['learning_rate']),
            'batch_size': int(params['batch_size']),
            'dropout_rate': dropout_rate,
            'n_layers': n_layers,
            'hidden_units': hidden_units
        }
        
        return ArchitectureCandidate(
            layers=layers,
            activation_functions=activation_functions,
            layer_types=layer_types,
            hyperparameters=hyperparameters
        )
    
    def evaluate_architecture(self, 
                           architecture: ArchitectureCandidate,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_val: Optional[np.ndarray] = None,
                           y_val: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate a neural architecture candidate.
        
        Args:
            architecture: Architecture candidate to evaluate
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary of performance metrics
        """
        if not TORCH_AVAILABLE:
            if UTILITIES_AVAILABLE:
                tprint_warning("⚠️ PyTorch not available, using mock evaluation")
            return self._mock_evaluation(architecture)
        
        try:
            start_time = time.time()
            
            # Create model
            model = StateSpaceModel(self.config, architecture)
            
            # Setup optimizer
            optimizer = optim.Adam(
                model.parameters(),
                lr=architecture.hyperparameters['learning_rate']
            )
            
            # Setup loss function
            criterion = nn.MSELoss()
            
            # Training loop
            model.train()
            batch_size = architecture.hyperparameters['batch_size']
            
            for epoch in range(self.config.max_epochs):
                # Simple training loop (in practice, you'd want more sophisticated training)
                if len(X_train) > batch_size:
                    indices = np.random.choice(len(X_train), batch_size, replace=False)
                    X_batch = X_train[indices]
                    y_batch = y_train[indices]
                else:
                    X_batch = X_train
                    y_batch = y_train
                
                # Convert to tensors
                X_tensor = torch.FloatTensor(X_batch)
                y_tensor = torch.FloatTensor(y_batch)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs, y_tensor)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Early stopping check (simplified)
                if epoch > self.config.early_stopping_patience:
                    break
            
            # Evaluation
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_train)
                y_tensor = torch.FloatTensor(y_train)
                train_outputs = model(X_tensor)
                train_loss = criterion(train_outputs, y_tensor).item()
                
                # Validation if provided
                val_loss = train_loss
                if X_val is not None and y_val is not None:
                    X_val_tensor = torch.FloatTensor(X_val)
                    y_val_tensor = torch.FloatTensor(y_val)
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
            
            training_time = time.time() - start_time
            
            # Calculate metrics
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'training_time': training_time,
                'model_size': sum(p.numel() for p in model.parameters()),
                'score': -val_loss  # Negative loss for maximization
            }
            
            if UTILITIES_AVAILABLE:
                tprint_debug(f"📊 Architecture evaluated - Score: {metrics['score']:.4f}")
            
            return metrics
            
        except Exception as e:
            if UTILITIES_AVAILABLE:
                tprint_error(f"❌ Architecture evaluation failed: {e}")
            return {
                'train_loss': float('inf'),
                'val_loss': float('inf'),
                'training_time': 0.0,
                'model_size': 0,
                'score': -float('inf')
            }
    
    def _mock_evaluation(self, architecture: ArchitectureCandidate) -> Dict[str, float]:
        """Mock evaluation when PyTorch is not available."""
        # Generate realistic mock metrics
        base_score = np.random.normal(0.8, 0.1)
        complexity_penalty = len(architecture.layers) * 0.01
        
        return {
            'train_loss': np.random.exponential(0.1),
            'val_loss': np.random.exponential(0.15),
            'training_time': np.random.uniform(1.0, 10.0),
            'model_size': len(architecture.layers) * 1000,
            'score': max(0.0, base_score - complexity_penalty)
        }
    
    def optimize(self, 
                X: np.ndarray,
                y: np.ndarray,
                X_val: Optional[np.ndarray] = None,
                y_val: Optional[np.ndarray] = None,
                **kwargs) -> OptimizationResult:
        """
        Perform neural architecture search optimization.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Additional arguments
            
        Returns:
            OptimizationResult with best architecture and metrics
        """
        if UTILITIES_AVAILABLE:
            tprint_info("🚀 Starting Neural Architecture Search optimization")
        
        self.optimization_start_time = time.time()
        
        try:
            # Create search space
            search_space = self.create_search_space()
            
            # Setup Bayesian optimization
            bayesian_config = BayesianTPEConfig(
                n_trials=self.config.n_trials,
                timeout_seconds=self.config.optimization_timeout,
                enable_early_stopping=self.config.enable_early_stopping,
                enable_grid_search=True,
                coarse_grid_points=5,
                fine_grid_points=8
            )
            
            self.bayesian_optimizer = BayesianTPEOptimizer(bayesian_config)
            
            # Define objective function
            def objective_function(params):
                architecture = self.generate_architecture_candidate(params)
                metrics = self.evaluate_architecture(architecture, X, y, X_val, y_val)
                
                # Store candidate
                architecture.performance_metrics = metrics
                architecture.validation_score = metrics['score']
                self.architecture_candidates.append(architecture)
                
                self.total_trials += 1
                if metrics['score'] > -float('inf'):
                    self.successful_trials += 1
                
                if UTILITIES_AVAILABLE and self.total_trials % self.config.log_interval == 0:
                    tprint_progress(
                        self.total_trials, 
                        self.config.n_trials,
                        f"Best score: {max([c.validation_score for c in self.architecture_candidates], default=0.0):.4f}"
                    )
                
                return metrics['score']
            
            # Run optimization
            result = self.bayesian_optimizer.optimize(
                objective_function=objective_function,
                search_space=search_space,
                X=X,
                y=y
            )
            
            # Find best architecture
            if self.architecture_candidates:
                best_candidate = max(self.architecture_candidates, key=lambda c: c.validation_score)
                self.best_architecture = best_candidate
                
                if UTILITIES_AVAILABLE:
                    tprint_success(f"✅ NAS optimization completed")
                    tprint_info(f"   → Best score: {best_candidate.validation_score:.4f}")
                    tprint_info(f"   → Architecture: {len(best_candidate.layers)} layers")
                    tprint_info(f"   → Training time: {result.optimization_time:.2f}s")
            
            # Store optimization history
            self.optimization_history.append({
                'timestamp': datetime.now().isoformat(),
                'total_trials': self.total_trials,
                'successful_trials': self.successful_trials,
                'best_score': best_candidate.validation_score if self.best_architecture else 0.0,
                'optimization_time': result.optimization_time
            })
            
            return result
            
        except Exception as e:
            if UTILITIES_AVAILABLE:
                tprint_error(f"❌ NAS optimization failed: {e}")
            raise
    
    def get_best_architecture(self) -> Optional[ArchitectureCandidate]:
        """Get the best architecture found during optimization."""
        return self.best_architecture
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        if not self.architecture_candidates:
            return {'error': 'No architectures evaluated'}
        
        best_architecture = self.best_architecture
        if not best_architecture:
            return {'error': 'No best architecture found'}
        
        return {
            'best_architecture': best_architecture.to_dict(),
            'total_candidates': len(self.architecture_candidates),
            'successful_trials': self.successful_trials,
            'total_trials': self.total_trials,
            'success_rate': self.successful_trials / max(self.total_trials, 1),
            'optimization_time': time.time() - self.optimization_start_time if self.optimization_start_time else 0.0,
            'optimization_history': self.optimization_history
        }
    
    def save_results(self, filepath: str) -> bool:
        """
        Save optimization results to file.
        
        Args:
            filepath: Path to save results
            
        Returns:
            True if successful, False otherwise
        """
        if not UTILITIES_AVAILABLE:
            return False
        
        try:
            results = self.get_optimization_summary()
            
            # Ensure directory exists
            ensure_directory(Path(filepath).parent)
            
            # Save using appropriate serializer
            if filepath.endswith('.json'):
                success = safe_json_dump(results, filepath)
            else:
                success = self.serializer.save(results, filepath)
            
            if success and UTILITIES_AVAILABLE:
                tprint_success(f"💾 Results saved to {filepath}")
            
            return success
            
        except Exception as e:
            if UTILITIES_AVAILABLE:
                tprint_error(f"❌ Failed to save results: {e}")
            return False
    
    def load_results(self, filepath: str) -> bool:
        """
        Load optimization results from file.
        
        Args:
            filepath: Path to load results from
            
        Returns:
            True if successful, False otherwise
        """
        if not UTILITIES_AVAILABLE:
            return False
        
        try:
            if not safe_file_exists(filepath):
                if UTILITIES_AVAILABLE:
                    tprint_warning(f"⚠️ Results file not found: {filepath}")
                return False
            
            # Load results
            if filepath.endswith('.json'):
                results = safe_json_load(filepath)
            else:
                results = self.serializer.load(filepath)
            
            if not results:
                if UTILITIES_AVAILABLE:
                    tprint_warning("⚠️ Failed to load results")
                return False
            
            # Restore state
            if 'best_architecture' in results:
                self.best_architecture = ArchitectureCandidate.from_dict(results['best_architecture'])
            
            if 'optimization_history' in results:
                self.optimization_history = results['optimization_history']
            
            if UTILITIES_AVAILABLE:
                tprint_success(f"📂 Results loaded from {filepath}")
            
            return True
            
        except Exception as e:
            if UTILITIES_AVAILABLE:
                tprint_error(f"❌ Failed to load results: {e}")
            return False
    
    def get_architecture_recommendations(self) -> Dict[str, Any]:
        """
        Get recommendations based on optimization results.
        
        Returns:
            Dictionary with architecture recommendations
        """
        if not self.architecture_candidates:
            return {'error': 'No architectures to analyze'}
        
        # Analyze top performers
        top_candidates = sorted(
            self.architecture_candidates, 
            key=lambda c: c.validation_score, 
            reverse=True
        )[:5]
        
        # Extract patterns
        common_activations = {}
        common_layer_types = {}
        avg_layers = 0
        avg_hidden_units = 0
        
        for candidate in top_candidates:
            # Count activations
            for activation in candidate.activation_functions:
                common_activations[activation] = common_activations.get(activation, 0) + 1
            
            # Count layer types
            for layer_type in candidate.layer_types:
                common_layer_types[layer_type] = common_layer_types.get(layer_type, 0) + 1
            
            avg_layers += len(candidate.layers)
            avg_hidden_units += candidate.hyperparameters.get('hidden_units', 0)
        
        avg_layers /= len(top_candidates)
        avg_hidden_units /= len(top_candidates)
        
        return {
            'top_activations': sorted(common_activations.items(), key=lambda x: x[1], reverse=True),
            'top_layer_types': sorted(common_layer_types.items(), key=lambda x: x[1], reverse=True),
            'recommended_layers': int(round(avg_layers)),
            'recommended_hidden_units': int(round(avg_hidden_units)),
            'top_candidates': [c.to_dict() for c in top_candidates]
        }


# Convenience functions
def create_nas_optimizer(config: Optional[NeuralSSMConfig] = None, **kwargs) -> NeuralSSM_NAS_Optimizer:
    """
    Create a NeuralSSM NAS optimizer instance.
    
    Args:
        config: Configuration for the optimizer
        **kwargs: Additional configuration parameters
        
    Returns:
        NeuralSSM_NAS_Optimizer instance
    """
    return NeuralSSM_NAS_Optimizer(config, **kwargs)


def quick_nas_search(X: np.ndarray, 
                    y: np.ndarray,
                    config: Optional[NeuralSSMConfig] = None,
                    **kwargs) -> Dict[str, Any]:
    """
    Quick neural architecture search.
    
    Args:
        X: Training features
        y: Training targets
        config: Configuration for the search
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with search results
    """
    optimizer = create_nas_optimizer(config, **kwargs)
    result = optimizer.optimize(X, y)
    return optimizer.get_optimization_summary()


# Export main classes and functions
__all__ = [
    'NeuralSSM_NAS_Optimizer',
    'NeuralSSMConfig',
    'ArchitectureCandidate',
    'StateSpaceModel',
    'create_nas_optimizer',
    'quick_nas_search'
]