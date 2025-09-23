"""
Neural Architecture Search Space

This module defines the search space for neural architectures,
including layer types, activation functions, and architectural patterns.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import itertools

from ..utils.nas_utils import NASUtils

logger = logging.getLogger(__name__)

class LayerType(Enum):
    """Types of neural network layers."""
    DENSE = "dense"
    CONV1D = "conv1d"
    CONV2D = "conv2d"
    LSTM = "lstm"
    GRU = "gru"
    ATTENTION = "attention"
    DROPOUT = "dropout"
    BATCH_NORM = "batch_norm"
    RESIDUAL = "residual"

class ActivationType(Enum):
    """Activation function types."""
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    GELU = "gelu"
    SWISH = "swish"
    NONE = "none"

class ProblemType(Enum):
    """Problem types for architecture optimization."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    HMM = "hmm"
    REGIME_DETECTION = "regime_detection"
    TIME_SERIES = "time_series"

@dataclass
class ArchitectureConfig:
    """Configuration for a neural architecture."""
    name: str
    input_dim: int
    output_dim: int
    hidden_dims: List[int] = field(default_factory=list)
    activation: str = "relu"
    dropout_rate: float = 0.0
    batch_norm: bool = False
    use_residual: bool = False
    problem_type: str = "classification"

    # Layer-specific parameters
    layer_types: List[str] = field(default_factory=list)
    attention_heads: int = 4
    embed_dim: int = 64

    # Architecture patterns
    use_attention: bool = False
    use_lstm: bool = False
    use_convolution: bool = False
    num_layers: int = 3

    # Regularization
    l1_regularization: float = 0.0
    l2_regularization: float = 0.0

    # Metadata
    complexity_score: float = 0.0
    estimated_params: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dims': self.hidden_dims,
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'use_residual': self.use_residual,
            'problem_type': self.problem_type,
            'layer_types': self.layer_types,
            'attention_heads': self.attention_heads,
            'embed_dim': self.embed_dim,
            'use_attention': self.use_attention,
            'use_lstm': self.use_lstm,
            'use_convolution': self.use_convolution,
            'num_layers': self.num_layers,
            'l1_regularization': self.l1_regularization,
            'l2_regularization': self.l2_regularization,
            'complexity_score': self.complexity_score,
            'estimated_params': self.estimated_params
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ArchitectureConfig':
        """Create from dictionary."""
        return cls(**config_dict)

    def calculate_complexity(self) -> float:
        """Calculate architecture complexity score."""
        complexity = 0.0

        # Base complexity from layers
        complexity += len(self.hidden_dims) * 0.1

        # Activation complexity
        activation_complexity = {
            'relu': 0.1,
            'tanh': 0.2,
            'sigmoid': 0.3,
            'leaky_relu': 0.15,
            'elu': 0.25,
            'gelu': 0.3,
            'swish': 0.35
        }
        complexity += activation_complexity.get(self.activation, 0.1)

        # Special components
        if self.use_attention:
            complexity += 0.5
        if self.use_lstm:
            complexity += 0.3
        if self.use_convolution:
            complexity += 0.4
        if self.use_residual:
            complexity += 0.2
        if self.batch_norm:
            complexity += 0.1
        if self.dropout_rate > 0:
            complexity += self.dropout_rate * 0.2

        # Regularization
        complexity += self.l1_regularization * 0.1
        complexity += self.l2_regularization * 0.1

        self.complexity_score = complexity
        return complexity

    def estimate_parameters(self) -> int:
        """Estimate number of parameters in the architecture."""
        params = 0

        # Input to first hidden
        if self.hidden_dims:
            params += self.input_dim * self.hidden_dims[0]

            # Hidden to hidden
            for i in range(1, len(self.hidden_dims)):
                params += self.hidden_dims[i-1] * self.hidden_dims[i]

            # Last hidden to output
            params += self.hidden_dims[-1] * self.output_dim
        else:
            # Direct input to output
            params += self.input_dim * self.output_dim

        # Special layers
        if self.use_attention:
            # Multi-head attention parameters
            params += self.embed_dim * self.embed_dim * 3 * self.attention_heads

        if self.use_lstm:
            # LSTM parameters (4 gates * hidden * (input + hidden + bias))
            lstm_params = 4 * self.hidden_dims[0] * (self.input_dim + self.hidden_dims[0] + 1)
            params += lstm_params

        if self.use_convolution:
            # Simple 1D convolution parameters
            conv_params = self.input_dim * 64 * 3  # 64 filters, kernel size 3
            params += conv_params

        # Batch norm parameters (scale and shift for each feature)
        if self.batch_norm:
            total_features = sum(self.hidden_dims) + self.output_dim
            params += total_features * 2

        self.estimated_params = params
        return params

class SearchSpace:
    """
    Neural Architecture Search Space

    Defines the space of possible architectures that can be searched,
    including constraints and valid combinations.
    """

    def __init__(self):
        """Initialize search space."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Define search dimensions
        self.input_dims = [10, 20, 50, 100, 200, 500, 1000]
        self.output_dims = [2, 3, 5, 10, 20, 50]
        self.hidden_dims_options = [
            [32], [64], [128], [256],
            [32, 16], [64, 32], [128, 64], [256, 128],
            [64, 32, 16], [128, 64, 32], [256, 128, 64],
            [128, 64, 32, 16], [256, 128, 64, 32]
        ]

        self.activation_options = [act.value for act in ActivationType]

        self.dropout_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

        self.layer_type_options = [layer.value for layer in LayerType]

        # Architecture patterns
        self.attention_heads_options = [1, 2, 4, 8, 16]
        self.embed_dims = [32, 64, 128, 256]

        # Problem-specific configurations
        self.problem_configs = self._define_problem_configs()

        # Constraints
        self.max_layers = 6
        self.max_hidden_dim = 1024
        self.min_hidden_dim = 8
        self.max_complexity = 10.0
        self.max_parameters = 10000000  # 10M parameters

        self.logger.info("🔍 Search space initialized")

    def _define_problem_configs(self) -> Dict[str, Dict[str, Any]]:
        """Define configurations for different problem types."""
        return {
            'classification': {
                'output_dims': [2, 3, 5, 10, 20],
                'preferred_activations': ['relu', 'leaky_relu', 'gelu'],
                'use_attention': False,
                'use_lstm': False
            },
            'regression': {
                'output_dims': [1],
                'preferred_activations': ['relu', 'tanh', 'elu'],
                'use_attention': False,
                'use_lstm': False
            },
            'hmm': {
                'output_dims': [3, 5, 10, 15, 20],  # Number of states
                'preferred_activations': ['tanh', 'relu'],
                'use_attention': False,
                'use_lstm': True
            },
            'regime_detection': {
                'output_dims': [5, 8, 10, 12, 15, 20],
                'preferred_activations': ['relu', 'leaky_relu'],
                'use_attention': False,
                'use_lstm': False
            },
            'time_series': {
                'output_dims': [1, 2, 3, 5],
                'preferred_activations': ['relu', 'tanh'],
                'use_attention': True,
                'use_lstm': True
            }
        }

    def generate_random_architecture(self,
                                   input_dim: int,
                                   output_dim: int,
                                   problem_type: str = "classification") -> ArchitectureConfig:
        """
        Generate a random architecture configuration.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            problem_type: Type of problem

        Returns:
            Random architecture configuration
        """
        # Get problem-specific config
        problem_config = self.problem_configs.get(problem_type, self.problem_configs['classification'])

        # Randomly select hidden dimensions
        hidden_dims = self.nas_utils.random_choice(self.hidden_dims_options)

        # Randomly select activation
        if problem_config['preferred_activations']:
            activation = self.nas_utils.random_choice(problem_config['preferred_activations'])
        else:
            activation = self.nas_utils.random_choice(self.activation_options)

        # Random dropout
        dropout_rate = self.nas_utils.random_choice(self.dropout_options)

        # Random layer types
        num_layers = len(hidden_dims) + 1  # +1 for output layer
        layer_types = ['dense'] * num_layers

        # Randomly add special layers
        if self.nas_utils.random_choice([True, False]) and problem_config.get('use_attention', False):
            layer_types[self.nas_utils.random_int(1, len(layer_types)-1)] = 'attention'

        if self.nas_utils.random_choice([True, False]) and problem_config.get('use_lstm', False):
            layer_types[self.nas_utils.random_int(1, len(layer_types)-1)] = 'lstm'

        # Create architecture name
        name = self._generate_architecture_name(hidden_dims, activation, dropout_rate)

        # Create configuration
        config = ArchitectureConfig(
            name=name,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout_rate=dropout_rate,
            batch_norm=self.nas_utils.random_choice([True, False]),
            use_residual=self.nas_utils.random_choice([True, False]),
            problem_type=problem_type,
            layer_types=layer_types,
            attention_heads=self.nas_utils.random_choice(self.attention_heads_options),
            embed_dim=self.nas_utils.random_choice(self.embed_dims),
            use_attention=problem_config.get('use_attention', False),
            use_lstm=problem_config.get('use_lstm', False),
            use_convolution=self.nas_utils.random_choice([True, False]),
            num_layers=len(hidden_dims) + 1
        )

        # Calculate complexity and parameters
        config.calculate_complexity()
        config.estimate_parameters()

        # Apply constraints
        if not self._check_constraints(config):
            # If constraints violated, generate a simpler architecture
            config = self._simplify_architecture(config)

        return config

    def generate_architecture_grid(self,
                                 input_dim: int,
                                 output_dim: int,
                                 problem_type: str = "classification",
                                 max_architectures: int = 100) -> List[ArchitectureConfig]:
        """
        Generate a grid of architectures covering the search space.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            problem_type: Type of problem
            max_architectures: Maximum number of architectures to generate

        Returns:
            List of architecture configurations
        """
        architectures = []

        # Generate combinations
        hidden_dim_combinations = self.hidden_dims_options[:min(len(self.hidden_dims_options), 20)]
        activation_combinations = self.activation_options[:min(len(self.activation_options), 5)]
        dropout_combinations = self.dropout_options[:min(len(self.dropout_options), 4)]

        total_combinations = len(hidden_dim_combinations) * len(activation_combinations) * len(dropout_combinations)

        if total_combinations > max_architectures:
            # Sample a subset
            indices = np.random.choice(total_combinations, max_architectures, replace=False)
            combinations = []
            for idx in indices:
                h_idx = idx // (len(activation_combinations) * len(dropout_combinations))
                a_idx = (idx // len(dropout_combinations)) % len(activation_combinations)
                d_idx = idx % len(dropout_combinations)
                combinations.append((hidden_dim_combinations[h_idx], activation_combinations[a_idx], dropout_combinations[d_idx]))
        else:
            # Use all combinations
            combinations = list(itertools.product(hidden_dim_combinations, activation_combinations, dropout_combinations))

        for hidden_dims, activation, dropout_rate in combinations:
            name = self._generate_architecture_name(hidden_dims, activation, dropout_rate)

            config = ArchitectureConfig(
                name=name,
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
                activation=activation,
                dropout_rate=dropout_rate,
                batch_norm=False,  # Keep it simple for grid
                use_residual=False,
                problem_type=problem_type
            )

            config.calculate_complexity()
            config.estimate_parameters()

            architectures.append(config)

        self.logger.info(f"📐 Generated {len(architectures)} architectures for grid search")
        return architectures

    def _generate_architecture_name(self, hidden_dims: List[int], activation: str, dropout_rate: float) -> str:
        """Generate a descriptive name for the architecture.

        Args:
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            dropout_rate: Dropout rate

        Returns:
            Architecture name
        """
        dims_str = '_'.join(map(str, hidden_dims))
        dropout_str = f"_d{dropout_rate}" if dropout_rate > 0 else ""
        return f"arch_{dims_str}_{activation}{dropout_str}"

    def _check_constraints(self, config: ArchitectureConfig) -> bool:
        """Check if architecture satisfies constraints.

        Args:
            config: Architecture configuration

        Returns:
            True if constraints satisfied
        """
        # Layer constraints
        if len(config.hidden_dims) > self.max_layers:
            return False

        # Dimension constraints
        for dim in config.hidden_dims:
            if dim > self.max_hidden_dim or dim < self.min_hidden_dim:
                return False

        # Complexity constraint
        if config.complexity_score > self.max_complexity:
            return False

        # Parameter constraint
        if config.estimated_params > self.max_parameters:
            return False

        return True

    def _simplify_architecture(self, config: ArchitectureConfig) -> ArchitectureConfig:
        """Simplify architecture to meet constraints.

        Args:
            config: Architecture configuration

        Returns:
            Simplified architecture configuration
        """
        # Reduce hidden dimensions
        if config.hidden_dims:
            config.hidden_dims = [max(dim // 2, self.min_hidden_dim) for dim in config.hidden_dims]

        # Reduce complexity
        if config.use_attention:
            config.use_attention = False

        if config.use_lstm:
            config.use_lstm = False

        if config.use_convolution:
            config.use_convolution = False

        # Recalculate metrics
        config.calculate_complexity()
        config.estimate_parameters()

        return config

    def get_search_space_size(self, problem_type: str = "classification") -> int:
        """Estimate the size of the search space.

        Args:
            problem_type: Type of problem

        Returns:
            Approximate search space size
        """
        hidden_options = len(self.hidden_dims_options)
        activation_options = len(self.activation_options)
        dropout_options = len(self.dropout_options)

        # Base space
        base_size = hidden_options * activation_options * dropout_options

        # Problem-specific multipliers
        problem_config = self.problem_configs.get(problem_type, self.problem_configs['classification'])

        if problem_config.get('use_attention'):
            base_size *= len(self.attention_heads_options)

        if problem_config.get('use_lstm'):
            base_size *= 2  # LSTM on/off

        return base_size

    @property
    def nas_utils(self) -> NASUtils:
        """Get NAS utilities instance."""
        return NASUtils()