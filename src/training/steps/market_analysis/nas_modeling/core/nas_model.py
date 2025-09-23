"""
Neural Architecture Search Model

This module provides PyTorch model implementations for NAS architectures,
including dynamic model creation from architecture configurations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field

from ..search.search_space import ArchitectureConfig

logger = logging.getLogger(__name__)

@dataclass
class NASModelConfig:
    """Configuration for NAS model creation."""
    input_dim: int
    output_dim: int
    hidden_dims: List[int] = field(default_factory=list)
    activation: str = "relu"
    dropout_rate: float = 0.0
    batch_norm: bool = False
    use_residual: bool = False
    problem_type: str = "classification"

class NASModel(nn.Module):
    """
    Neural Architecture Search Model

    Creates dynamic PyTorch models based on architecture configurations
    for different problem types (classification, regression, HMM).
    """

    def __init__(self, config: ArchitectureConfig):
        """Initialize NAS model from architecture config.

        Args:
            config: Architecture configuration
        """
        super(NASModel, self).__init__()

        self.config = config
        self.problem_type = config.problem_type

        # Get activation function
        self.activation_fn = self._get_activation_function(config.activation)

        # Build model layers
        self.layers = nn.ModuleList()
        self.build_model()

        # Initialize weights
        self.apply(self._init_weights)

        logger.info(f"🏗️ NAS Model created: {config.name}")
        logger.info(f"📊 Model parameters: {self.get_n_params():,}")

    def build_model(self):
        """Build model architecture based on configuration."""
        input_dim = self.config.input_dim
        hidden_dims = self.config.hidden_dims
        output_dim = self.config.output_dim

        # Input layer
        if hidden_dims:
            self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
            self._add_layer_components(0)

            # Hidden layers
            for i in range(1, len(hidden_dims)):
                self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
                self._add_layer_components(i)

            # Output layer
            self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        else:
            # Direct input to output
            self.layers.append(nn.Linear(input_dim, output_dim))

        # Output activation for specific problem types
        self.output_activation = self._get_output_activation()

    def _add_layer_components(self, layer_idx: int):
        """Add batch norm and dropout to layer."""
        if self.config.batch_norm:
            self.layers.append(nn.BatchNorm1d(self.config.hidden_dims[layer_idx]))

        if self.config.dropout_rate > 0:
            self.layers.append(nn.Dropout(self.config.dropout_rate))

    def _get_activation_function(self, activation_name: str) -> Callable:
        """Get activation function by name."""
        activation_map = {
            'relu': F.relu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'leaky_relu': F.leaky_relu,
            'elu': F.elu,
            'gelu': F.gelu,
            'swish': self._swish,
            'none': lambda x: x
        }
        return activation_map.get(activation_name, F.relu)

    def _swish(self, x: torch.Tensor) -> torch.Tensor:
        """Swish activation function."""
        return x * torch.sigmoid(x)

    def _get_output_activation(self) -> Optional[Callable]:
        """Get output activation based on problem type."""
        if self.problem_type == "classification":
            return F.log_softmax
        elif self.problem_type == "regression":
            return None
        elif self.problem_type == "hmm":
            return F.log_softmax  # For state probabilities
        else:
            return None

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        for i, layer in enumerate(self.layers):
            # Apply linear transformation
            if isinstance(layer, nn.Linear):
                x = layer(x)

                # Apply activation (except for last layer)
                if i < len(self.layers) - 1:  # Not the last layer
                    x = self.activation_fn(x)

        # Apply output activation if needed
        if self.output_activation is not None:
            x = self.output_activation(x, dim=-1)

        return x

    def get_n_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def create_from_config(cls, config: ArchitectureConfig, problem_type: str) -> 'NASModel':
        """Create model from architecture configuration.

        Args:
            config: Architecture configuration
            problem_type: Type of problem

        Returns:
            NASModel instance
        """
        # Create model configuration
        model_config = NASModelConfig(
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
            dropout_rate=config.dropout_rate,
            batch_norm=config.batch_norm,
            use_residual=config.use_residual,
            problem_type=problem_type
        )

        # Create architecture config with model-specific settings
        arch_config = ArchitectureConfig(
            name=config.name,
            input_dim=config.input_dim,
            output_dim=config.output_dim,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
            dropout_rate=config.dropout_rate,
            batch_norm=config.batch_norm,
            use_residual=config.use_residual,
            problem_type=problem_type,
            layer_types=config.layer_types,
            attention_heads=config.attention_heads,
            embed_dim=config.embed_dim
        )

        return cls(arch_config)

class HMM_NAS_Model(NASModel):
    """
    NAS Model specifically designed for HMM state modeling.

    This model learns HMM state representations and transition patterns
    using neural networks.
    """

    def __init__(self, config: ArchitectureConfig, n_states: int = 5):
        """Initialize HMM NAS model.

        Args:
            config: Architecture configuration
            n_states: Number of HMM states
        """
        self.n_states = n_states
        super().__init__(config)

        # HMM-specific components
        self.state_encoder = self._build_state_encoder()
        self.transition_predictor = self._build_transition_predictor()

    def _build_state_encoder(self) -> nn.Module:
        """Build state encoder for HMM states."""
        return nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dims[0] if self.config.hidden_dims else 64),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dims[0] if self.config.hidden_dims else 64, self.n_states),
            nn.LogSoftmax(dim=-1)
        )

    def _build_transition_predictor(self) -> nn.Module:
        """Build transition predictor for HMM state transitions."""
        hidden_dim = self.config.hidden_dims[0] if self.config.hidden_dims else 64

        return nn.Sequential(
            nn.Linear(self.n_states * 2, hidden_dim),  # current state + next state
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_states),  # Predict next state
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for HMM model.

        Args:
            x: Input tensor (market features)

        Returns:
            Tuple of (state_probabilities, transition_probabilities)
        """
        # Encode current state
        state_probs = self.state_encoder(x)

        # Predict transitions (this would be trained with sequences)
        # For simplicity, return state probabilities for both
        batch_size = x.size(0)
        transition_probs = torch.zeros(batch_size, self.n_states, self.n_states, device=x.device)

        # Simple transition: stay in same state with high probability
        for i in range(self.n_states):
            transition_probs[:, i, i] = 0.8  # High probability to stay
            # Distribute remaining probability to other states
            remaining_prob = 0.2 / (self.n_states - 1)
            for j in range(self.n_states):
                if j != i:
                    transition_probs[:, i, j] = remaining_prob

        return state_probs, transition_probs

class Regime_NAS_Model(NASModel):
    """
    NAS Model specifically designed for market regime detection.

    This model learns market regime representations and regime transitions.
    """

    def __init__(self, config: ArchitectureConfig, n_regimes: int = 10):
        """Initialize regime NAS model.

        Args:
            config: Architecture configuration
            n_regimes: Number of market regimes
        """
        self.n_regimes = n_regimes
        super().__init__(config)

        # Regime-specific components
        self.regime_classifier = self._build_regime_classifier()
        self.regime_encoder = self._build_regime_encoder()

    def _build_regime_classifier(self) -> nn.Module:
        """Build regime classifier."""
        hidden_dim = self.config.hidden_dims[0] if self.config.hidden_dims else 64

        return nn.Sequential(
            nn.Linear(self.config.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.n_regimes),
            nn.LogSoftmax(dim=-1)
        )

    def _build_regime_encoder(self) -> nn.Module:
        """Build regime encoder for feature extraction."""
        hidden_dim = self.config.hidden_dims[0] if self.config.hidden_dims else 64

        return nn.Sequential(
            nn.Linear(self.config.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for regime model.

        Args:
            x: Input tensor (market features)

        Returns:
            Tuple of (regime_probabilities, encoded_features)
        """
        # Classify regime
        regime_probs = self.regime_classifier(x)

        # Encode features for downstream tasks
        encoded_features = self.regime_encoder(x)

        return regime_probs, encoded_features

class TimeSeries_NAS_Model(NASModel):
    """
    NAS Model specifically designed for time series analysis.

    Includes temporal components like LSTM, GRU, or attention mechanisms.
    """

    def __init__(self, config: ArchitectureConfig, sequence_length: int = 20):
        """Initialize time series NAS model.

        Args:
            config: Architecture configuration
            sequence_length: Length of input sequences
        """
        self.sequence_length = sequence_length
        super().__init__(config)

        # Time series components
        if hasattr(config, 'use_lstm') and config.use_lstm:
            self.temporal_layer = self._build_lstm_layer()
        elif hasattr(config, 'use_attention') and config.use_attention:
            self.temporal_layer = self._build_attention_layer()
        else:
            self.temporal_layer = None

    def _build_lstm_layer(self) -> nn.Module:
        """Build LSTM layer for temporal modeling."""
        hidden_dim = self.config.hidden_dims[0] if self.config.hidden_dims else 64
        lstm_hidden = hidden_dim // 2

        return nn.LSTM(
            input_size=self.config.input_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=self.config.dropout_rate if hasattr(self.config, 'lstm_dropout') else 0.0
        )

    def _build_attention_layer(self) -> nn.Module:
        """Build attention layer for temporal modeling."""
        hidden_dim = self.config.hidden_dims[0] if self.config.hidden_dims else 64

        return nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=getattr(self.config, 'attention_heads', 4),
            dropout=self.config.dropout_rate,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for time series model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)

        Returns:
            Output tensor
        """
        if self.temporal_layer is not None:
            if isinstance(self.temporal_layer, nn.LSTM):
                # LSTM processing
                lstm_out, _ = self.temporal_layer(x)
                x = lstm_out[:, -1, :]  # Take last output
            elif isinstance(self.temporal_layer, nn.MultiheadAttention):
                # Attention processing
                attn_out, _ = self.temporal_layer(x, x, x)
                x = attn_out.mean(dim=1)  # Average over sequence

        # Pass through regular layers
        return super().forward(x)