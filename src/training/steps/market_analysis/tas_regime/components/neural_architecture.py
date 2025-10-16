"""
Neural Architecture Components for Advanced TAS

This module provides neural network architectures for trading including:
- LSTM-based models for time series
- Attention mechanisms
- Neural ODEs
- Neural State Space models
- Hybrid tree-neural ensembles
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import logging
from datetime import datetime

from ..core.tas_config import TASConfig, TASArchitectureType, TradingObjective

logger = logging.getLogger(__name__)

@dataclass
class NeuralArchitectureConfig:
    """Configuration for neural architecture components."""
    input_dim: int
    output_dim: int
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    activation: str = "relu"
    dropout_rate: float = 0.2
    use_lstm: bool = False
    use_attention: bool = False
    use_neural_ode: bool = False
    use_batch_norm: bool = True
    sequence_length: int = 20
    attention_heads: int = 4
    problem_type: str = "regression"

class LSTMTradingModel(nn.Module):
    """LSTM-based trading model for time series forecasting."""

    def __init__(self, config: NeuralArchitectureConfig):
        """Initialize LSTM trading model.

        Args:
            config: Neural architecture configuration
        """
        super(LSTMTradingModel, self).__init__()

        self.config = config
        self.hidden_dims = config.hidden_dims
        self.use_batch_norm = config.use_batch_norm
        self.dropout_rate = config.dropout_rate
        self.sequence_length = config.sequence_length

        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        input_dim = config.input_dim

        for i, hidden_dim in enumerate(self.hidden_dims):
            lstm_layer = nn.LSTM(
                input_size=input_dim if i == 0 else self.hidden_dims[i-1],
                hidden_size=hidden_dim,
                batch_first=True,
                dropout=self.dropout_rate if i > 0 else 0,
                num_layers=1
            )
            self.lstm_layers.append(lstm_layer)

            if self.use_batch_norm:
                self.lstm_layers.append(nn.BatchNorm1d(hidden_dim))

            input_dim = hidden_dim

        # Output layer
        self.output_layer = nn.Linear(self.hidden_dims[-1], config.output_dim)

        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)

        # Activation function
        self.activation = self._get_activation_function(config.activation)

    def _get_activation_function(self, activation_name: str) -> Callable:
        """Get activation function by name."""
        activation_map = {
            'relu': F.relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'leaky_relu': F.leaky_relu,
            'elu': F.elu,
            'gelu': F.gelu,
            'swish': self._swish
        }
        return activation_map.get(activation_name, F.relu)

    def _swish(self, x: torch.Tensor) -> torch.Tensor:
        """Swish activation function."""
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)

        Returns:
            Output tensor
        """
        # LSTM processing
        for i, layer in enumerate(self.lstm_layers):
            if isinstance(layer, nn.LSTM):
                x, _ = layer(x)
                x = self.activation(x)
                if self.dropout_rate > 0 and i < len(self.lstm_layers) - 1:
                    x = self.dropout(x)
            elif isinstance(layer, nn.BatchNorm1d):
                # Reshape for batch norm: (batch_size * seq_len, hidden_dim)
                batch_size, seq_len, hidden_dim = x.shape
                x_flat = x.view(-1, hidden_dim)
                x_flat = layer(x_flat)
                x = x_flat.view(batch_size, seq_len, hidden_dim)

        # Take the last output of the sequence
        if x.dim() == 3:
            x = x[:, -1, :]  # (batch_size, hidden_dim)

        # Output layer
        output = self.output_layer(x)

        return output

class AttentionTradingModel(nn.Module):
    """Attention-based trading model for capturing long-range dependencies."""

    def __init__(self, config: NeuralArchitectureConfig):
        """Initialize attention trading model.

        Args:
            config: Neural architecture configuration
        """
        super(AttentionTradingModel, self).__init__()

        self.config = config
        self.hidden_dims = config.hidden_dims
        self.attention_heads = config.attention_heads
        self.use_batch_norm = config.use_batch_norm
        self.dropout_rate = config.dropout_rate

        # Input projection
        self.input_projection = nn.Linear(config.input_dim, self.hidden_dims[0])

        # Multi-head attention layers
        self.attention_layers = nn.ModuleList()
        for i in range(len(self.hidden_dims)):
            attention_layer = nn.MultiheadAttention(
                embed_dim=self.hidden_dims[i],
                num_heads=self.attention_heads,
                dropout=self.dropout_rate,
                batch_first=True
            )
            self.attention_layers.append(attention_layer)

            if self.use_batch_norm:
                self.attention_layers.append(nn.BatchNorm1d(self.hidden_dims[i]))

        # Feed-forward layers
        self.feedforward_layers = nn.ModuleList()
        for i in range(len(self.hidden_dims)):
            ff_layer = nn.Linear(self.hidden_dims[i], self.hidden_dims[i])
            self.feedforward_layers.append(ff_layer)

        # Output layer
        self.output_layer = nn.Linear(self.hidden_dims[-1], config.output_dim)

        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(self.hidden_dims[0])
        self.dropout = nn.Dropout(self.dropout_rate)

        # Activation function
        self.activation = self._get_activation_function(config.activation)

    def _get_activation_function(self, activation_name: str) -> Callable:
        """Get activation function by name."""
        activation_map = {
            'relu': F.relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'leaky_relu': F.leaky_relu,
            'elu': F.elu,
            'gelu': F.gelu,
            'swish': self._swish
        }
        return activation_map.get(activation_name, F.relu)

    def _swish(self, x: torch.Tensor) -> torch.Tensor:
        """Swish activation function."""
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through attention model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)

        Returns:
            Output tensor
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x_proj = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)
        x_proj = self.layer_norm(x_proj)

        # Apply attention layers
        attention_out = x_proj
        for i, layer in enumerate(self.attention_layers):
            if isinstance(layer, nn.MultiheadAttention):
                attn_out, _ = layer(attention_out, attention_out, attention_out)
                attention_out = attention_out + attn_out  # Residual connection
                attention_out = self.layer_norm(attention_out)
                attention_out = self.dropout(attention_out)
            elif isinstance(layer, nn.BatchNorm1d):
                # Flatten for batch norm
                attn_flat = attention_out.view(-1, attention_out.size(-1))
                attn_flat = layer(attn_flat)
                attention_out = attn_flat.view(batch_size, seq_len, -1)

        # Feed-forward processing
        for ff_layer in self.feedforward_layers:
            attention_out = ff_layer(attention_out)
            attention_out = self.activation(attention_out)

        # Global average pooling
        pooled_out = torch.mean(attention_out, dim=1)  # (batch_size, hidden_dim)

        # Output layer
        output = self.output_layer(pooled_out)

        return output

class NeuralODETradingModel(nn.Module):
    """Neural ODE-based trading model for continuous-time modeling."""

    def __init__(self, config: NeuralArchitectureConfig):
        """Initialize Neural ODE trading model.

        Args:
            config: Neural architecture configuration
        """
        super(NeuralODETradingModel, self).__init__()

        self.config = config
        self.hidden_dims = config.hidden_dims

        # ODE function (dynamics)
        self.ode_func = ODENet(
            input_dim=config.input_dim,
            hidden_dim=self.hidden_dims[0],
            activation=config.activation
        )

        # Initial state encoder
        self.initial_encoder = nn.Sequential(
            nn.Linear(config.input_dim, self.hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[0])
        )

        # Output decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(self.hidden_dims[1], config.output_dim)
        )

        # Time points for ODE solving
        self.time_points = torch.linspace(0, 1, config.sequence_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Neural ODE model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)

        Returns:
            Output tensor
        """
        batch_size, seq_len, input_dim = x.shape

        # Encode initial state from first time point
        initial_state = self.initial_encoder(x[:, 0, :])  # (batch_size, hidden_dim)

        # Solve ODE to get state trajectory
        state_trajectory = self._solve_ode(initial_state, x)

        # Decode final state to output
        output = self.decoder(state_trajectory[:, -1, :])

        return output

    def _solve_ode(self, initial_state: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Solve ODE using simple Euler integration for trading."""
        batch_size = initial_state.shape[0]

        # Initialize state trajectory
        state = initial_state.unsqueeze(1).repeat(1, self.time_points.shape[0], 1)

        # Simple Euler integration
        dt = self.time_points[1] - self.time_points[0]

        for i in range(1, len(self.time_points)):
            # Current state
            current_state = state[:, i-1, :]

            # ODE dynamics
            state_derivative = self.ode_func(current_state)

            # Euler step
            next_state = current_state + dt * state_derivative

            # Store next state
            state[:, i, :] = next_state

        return state

class ODENet(nn.Module):
    """Neural ODE function representing dynamics."""

    def __init__(self, input_dim: int, hidden_dim: int, activation: str = "tanh"):
        """Initialize ODE network.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            activation: Activation function
        """
        super(ODENet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ODE function.

        Args:
            x: State tensor

        Returns:
            State derivative
        """
        return self.net(x)

class NeuralStateSpaceModel(nn.Module):
    """Neural State Space model for regime detection."""

    def __init__(self, config: NeuralArchitectureConfig, n_regimes: int = 10):
        """Initialize Neural State Space model.

        Args:
            config: Neural architecture configuration
            n_regimes: Number of regimes
        """
        super(NeuralStateSpaceModel, self).__init__()

        self.config = config
        self.n_regimes = n_regimes
        self.hidden_dims = config.hidden_dims

        # State transition network
        self.transition_net = nn.Sequential(
            nn.Linear(config.input_dim + n_regimes, self.hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(self.hidden_dims[1], n_regimes),
            nn.Softmax(dim=-1)
        )

        # Observation network
        self.observation_net = nn.Sequential(
            nn.Linear(n_regimes, self.hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(self.hidden_dims[0], self.hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(self.hidden_dims[1], config.output_dim)
        )

        # Initial state distribution
        self.initial_state = nn.Parameter(torch.randn(n_regimes))
        self.initial_state.data = F.softmax(self.initial_state, dim=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through state space model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)

        Returns:
            Tuple of (observations, state_probabilities)
        """
        batch_size, seq_len, _ = x.shape

        # Initialize state probabilities
        state_probs = []
        observations = []

        # Initial state
        current_state = self.initial_state.unsqueeze(0).repeat(batch_size, 1)

        for t in range(seq_len):
            # State transition
            combined_input = torch.cat([x[:, t, :], current_state], dim=1)
            transition_logits = self.transition_net(combined_input)
            current_state = F.softmax(transition_logits, dim=1)

            # Observation
            observation = self.observation_net(current_state)

            state_probs.append(current_state)
            observations.append(observation)

        # Stack results
        state_probabilities = torch.stack(state_probs, dim=1)  # (batch_size, seq_len, n_regimes)
        observations = torch.stack(observations, dim=1)  # (batch_size, seq_len, output_dim)

        return observations, state_probabilities

class HybridTreeNeuralModel(nn.Module):
    """Hybrid model combining tree-based and neural approaches."""

    def __init__(self, config: NeuralArchitectureConfig, tree_config: Dict[str, Any]):
        """Initialize hybrid tree-neural model.

        Args:
            config: Neural architecture configuration
            tree_config: Tree model configuration
        """
        super(HybridTreeNeuralModel, self).__init__()

        self.config = config
        self.tree_config = tree_config

        # Tree-inspired neural components
        self.tree_layers = nn.ModuleList()
        input_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            # Tree-like splitting: multiple branches
            tree_layer = TreeInspiredLayer(input_dim, hidden_dim)
            self.tree_layers.append(tree_layer)
            input_dim = hidden_dim

        # Final output layer
        self.output_layer = nn.Linear(input_dim, config.output_dim)

        # Attention mechanism for ensemble-like behavior
        if config.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=config.attention_heads,
                batch_first=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through hybrid model.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # Process through tree-inspired layers
        for tree_layer in self.tree_layers:
            x = tree_layer(x)

        # Apply attention if enabled
        if hasattr(self, 'attention'):
            x, _ = self.attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
            x = x.squeeze(1)

        # Output layer
        output = self.output_layer(x)

        return output

class TreeInspiredLayer(nn.Module):
    """Tree-inspired neural layer with splitting behavior."""

    def __init__(self, input_dim: int, output_dim: int):
        """Initialize tree-inspired layer.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
        """
        super(TreeInspiredLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Split into multiple branches (like tree splits)
        self.n_branches = min(4, output_dim // 8 + 1)
        self.branches = nn.ModuleList()

        for _ in range(self.n_branches):
            branch = nn.Sequential(
                nn.Linear(input_dim, output_dim // self.n_branches),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            self.branches.append(branch)

        # Final combination layer
        self.combine = nn.Linear(output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through tree-inspired layer.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # Process through branches
        branch_outputs = []
        for branch in self.branches:
            branch_output = branch(x)
            branch_outputs.append(branch_output)

        # Combine branches
        combined = torch.cat(branch_outputs, dim=-1)
        output = self.combine(combined)

        return output

class TASNeuralModel(nn.Module):
    """Main neural model class for TAS that combines different architectures."""

    def __init__(self, config: TASConfig, architecture_type: TASArchitectureType = TASArchitectureType.HYBRID_TREE_NEURAL):
        """Initialize TAS neural model.

        Args:
            config: TAS configuration
            architecture_type: Type of architecture to use
        """
        super(TASNeuralModel, self).__init__()

        self.config = config
        self.architecture_type = architecture_type
        self.neural_config = NeuralArchitectureConfig(
            input_dim=config.search_space_config.get('neural_input_dim', 20),
            output_dim=config.search_space_config.get('neural_output_dim', 1),
            hidden_dims=config.get_neural_search_space().get('hidden_dims', [[64, 32]]),
            activation=config.get_neural_search_space().get('activation', ['relu'])[0],
            dropout_rate=config.get_neural_search_space().get('dropout_rates', [0.2])[0],
            use_lstm=config.get_neural_search_space().get('use_lstm', [False])[0],
            use_attention=config.get_neural_search_space().get('use_attention', [False])[0],
            use_batch_norm=config.get_neural_search_space().get('use_batch_norm', [True])[0],
            sequence_length=config.regime_detection_window
        )

        # Create the appropriate model based on architecture type
        if architecture_type == TASArchitectureType.NEURAL_ONLY:
            if self.neural_config.use_lstm:
                self.model = LSTMTradingModel(self.neural_config)
            elif self.neural_config.use_attention:
                self.model = AttentionTradingModel(self.neural_config)
            else:
                self.model = HybridTreeNeuralModel(self.neural_config, {})
        elif architecture_type == TASArchitectureType.HYBRID_TREE_NEURAL:
            self.model = HybridTreeNeuralModel(self.neural_config, config.get_tree_search_space())
        else:
            # Default to hybrid model
            self.model = HybridTreeNeuralModel(self.neural_config, config.get_tree_search_space())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return self.model(x)

    def get_model_complexity(self) -> float:
        """Get model complexity score."""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Base complexity from parameters
        complexity = np.log10(total_params + 1)

        # Add complexity for special components
        if hasattr(self.model, 'lstm_layers'):
            complexity += 0.5
        if hasattr(self.model, 'attention'):
            complexity += 0.3

        return complexity

    def get_architecture_summary(self) -> Dict[str, Any]:
        """Get architecture summary for analysis."""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'architecture_type': self.architecture_type.value,
            'total_parameters': total_params,
            'complexity_score': self.get_model_complexity(),
            'hidden_dims': self.neural_config.hidden_dims,
            'activation': self.neural_config.activation,
            'dropout_rate': self.neural_config.dropout_rate,
            'use_lstm': self.neural_config.use_lstm,
            'use_attention': self.neural_config.use_attention,
            'use_batch_norm': self.neural_config.use_batch_norm,
            'sequence_length': self.neural_config.sequence_length
        }
