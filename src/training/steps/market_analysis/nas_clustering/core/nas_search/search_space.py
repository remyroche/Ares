"""
Search Space Definition for Neural Architecture Search

This module defines the search space for neural architectures optimized for regime detection,
including layer types, activation functions, connection patterns, and constraints.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class LayerType(Enum):
    """Types of layers available in the architecture search space."""
    # Dense layers
    DENSE = "dense"
    
    # Convolutional layers for temporal patterns
    CONV1D = "conv1d"
    CONV2D = "conv2d"
    
    # Recurrent layers for sequential modeling
    LSTM = "lstm"
    GRU = "gru"
    RNN = "rnn"
    
    # Attention mechanisms
    ATTENTION = "attention"
    MULTI_HEAD_ATTENTION = "multi_head_attention"
    SELF_ATTENTION = "self_attention"
    
    # Pooling layers
    MAX_POOLING = "max_pooling"
    AVG_POOLING = "avg_pooling"
    GLOBAL_POOLING = "global_pooling"
    
    # Normalization layers
    BATCH_NORM = "batch_norm"
    LAYER_NORM = "layer_norm"
    
    # Regularization layers
    DROPOUT = "dropout"
    
    # Skip connections
    RESIDUAL = "residual"
    DENSE_CONNECTION = "dense_connection"


class ActivationFunction(Enum):
    """Activation functions available in the search space."""
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SWISH = "swish"
    GELU = "gelu"
    ELU = "elu"
    SELU = "selu"
    MISH = "mish"
    LINEAR = "linear"


class ConnectionType(Enum):
    """Types of connections between layers."""
    SEQUENTIAL = "sequential"  # Standard sequential connection
    RESIDUAL = "residual"      # Skip connection with addition
    CONCATENATION = "concatenation"  # Skip connection with concatenation
    ATTENTION = "attention"    # Attention-based connection
    HIGHWAY = "highway"        # Highway connection


@dataclass
class LayerConfig:
    """Configuration for a neural network layer."""
    layer_type: LayerType
    activation: ActivationFunction
    units: int
    dropout_rate: float = 0.0
    batch_norm: bool = False
    kernel_size: Optional[int] = None  # For conv layers
    stride: Optional[int] = None       # For conv layers
    padding: Optional[str] = None      # For conv layers
    return_sequences: Optional[bool] = None  # For RNN layers
    attention_heads: Optional[int] = None    # For attention layers
    attention_dim: Optional[int] = None      # For attention layers


@dataclass
class ConnectionConfig:
    """Configuration for connections between layers."""
    connection_type: ConnectionType
    from_layer: int  # Layer index
    to_layer: int    # Layer index
    weight: float = 1.0  # Connection strength/weight


@dataclass
class ArchitectureConstraints:
    """Constraints for architecture generation."""
    # Layer constraints
    min_layers: int = 2
    max_layers: int = 10
    min_units: int = 16
    max_units: int = 512
    
    # Layer type constraints
    max_conv_layers: int = 3
    max_rnn_layers: int = 3
    max_attention_layers: int = 2
    
    # Connection constraints
    max_skip_connections: int = 5
    max_residual_connections: int = 3
    
    # Complexity constraints
    max_total_parameters: int = 1000000  # 1M parameters
    max_inference_time_ms: float = 100.0  # 100ms max inference
    
    # Regime-specific constraints
    require_temporal_layers: bool = True  # Must have at least one temporal layer
    require_attention: bool = False       # Optional attention mechanism
    require_skip_connections: bool = False  # Optional skip connections


@dataclass
class SearchSpace:
    """Complete search space definition for neural architecture search."""
    
    # Available layer types
    available_layer_types: List[LayerType] = field(default_factory=lambda: [
        LayerType.DENSE,
        LayerType.CONV1D,
        LayerType.LSTM,
        LayerType.GRU,
        LayerType.ATTENTION,
        LayerType.MULTI_HEAD_ATTENTION,
        LayerType.BATCH_NORM,
        LayerType.DROPOUT,
        LayerType.RESIDUAL
    ])
    
    # Available activation functions
    available_activations: List[ActivationFunction] = field(default_factory=lambda: [
        ActivationFunction.RELU,
        ActivationFunction.LEAKY_RELU,
        ActivationFunction.TANH,
        ActivationFunction.SWISH,
        ActivationFunction.GELU
    ])
    
    # Available connection types
    available_connections: List[ConnectionType] = field(default_factory=lambda: [
        ConnectionType.SEQUENTIAL,
        ConnectionType.RESIDUAL,
        ConnectionType.CONCATENATION,
        ConnectionType.ATTENTION
    ])
    
    # Layer size options
    layer_size_options: List[int] = field(default_factory=lambda: [
        16, 32, 64, 128, 256, 512
    ])
    
    # Dropout rate options
    dropout_options: List[float] = field(default_factory=lambda: [
        0.0, 0.1, 0.2, 0.3, 0.5
    ])
    
    # Kernel size options for conv layers
    kernel_size_options: List[int] = field(default_factory=lambda: [
        3, 5, 7, 9, 11
    ])
    
    # Attention head options
    attention_head_options: List[int] = field(default_factory=lambda: [
        1, 2, 4, 8
    ])
    
    # Constraints
    constraints: ArchitectureConstraints = field(default_factory=ArchitectureConstraints)
    
    def validate_layer_config(self, layer_config: LayerConfig) -> bool:
        """Validate a layer configuration against constraints."""
        try:
            # Check layer count constraints (handled at architecture level)
            
            # Check units constraints
            if not (self.constraints.min_units <= layer_config.units <= self.constraints.max_units):
                return False
            
            # Check dropout rate
            if not (0.0 <= layer_config.dropout_rate <= 1.0):
                return False
            
            # Check layer-specific constraints
            if layer_config.layer_type in [LayerType.CONV1D, LayerType.CONV2D]:
                if layer_config.kernel_size is None:
                    return False
                if layer_config.kernel_size not in self.kernel_size_options:
                    return False
            
            if layer_config.layer_type in [LayerType.ATTENTION, LayerType.MULTI_HEAD_ATTENTION]:
                if layer_config.attention_heads is None:
                    return False
                if layer_config.attention_heads not in self.attention_head_options:
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Layer validation failed: {e}")
            return False
    
    def validate_architecture(self, layers: List[LayerConfig], connections: List[ConnectionConfig]) -> Tuple[bool, List[str]]:
        """Validate a complete architecture against constraints."""
        errors = []
        
        try:
            # Check layer count
            if len(layers) < self.constraints.min_layers:
                errors.append(f"Too few layers: {len(layers)} < {self.constraints.min_layers}")
            if len(layers) > self.constraints.max_layers:
                errors.append(f"Too many layers: {len(layers)} > {self.constraints.max_layers}")
            
            # Check individual layers
            for i, layer in enumerate(layers):
                if not self.validate_layer_config(layer):
                    errors.append(f"Invalid layer {i}: {layer.layer_type}")
            
            # Check layer type constraints
            conv_count = sum(1 for layer in layers if layer.layer_type in [LayerType.CONV1D, LayerType.CONV2D])
            if conv_count > self.constraints.max_conv_layers:
                errors.append(f"Too many conv layers: {conv_count} > {self.constraints.max_conv_layers}")
            
            rnn_count = sum(1 for layer in layers if layer.layer_type in [LayerType.LSTM, LayerType.GRU, LayerType.RNN])
            if rnn_count > self.constraints.max_rnn_layers:
                errors.append(f"Too many RNN layers: {rnn_count} > {self.constraints.max_rnn_layers}")
            
            attention_count = sum(1 for layer in layers if layer.layer_type in [LayerType.ATTENTION, LayerType.MULTI_HEAD_ATTENTION])
            if attention_count > self.constraints.max_attention_layers:
                errors.append(f"Too many attention layers: {attention_count} > {self.constraints.max_attention_layers}")
            
            # Check temporal layer requirement
            if self.constraints.require_temporal_layers:
                temporal_layers = sum(1 for layer in layers if layer.layer_type in [
                    LayerType.LSTM, LayerType.GRU, LayerType.RNN, LayerType.CONV1D
                ])
                if temporal_layers == 0:
                    errors.append("Architecture must contain at least one temporal layer")
            
            # Check connection constraints
            residual_connections = sum(1 for conn in connections if conn.connection_type == ConnectionType.RESIDUAL)
            if residual_connections > self.constraints.max_residual_connections:
                errors.append(f"Too many residual connections: {residual_connections} > {self.constraints.max_residual_connections}")
            
            skip_connections = sum(1 for conn in connections if conn.connection_type in [
                ConnectionType.RESIDUAL, ConnectionType.CONCATENATION
            ])
            if skip_connections > self.constraints.max_skip_connections:
                errors.append(f"Too many skip connections: {skip_connections} > {self.constraints.max_skip_connections}")
            
            # Check attention requirement
            if self.constraints.require_attention:
                if attention_count == 0:
                    errors.append("Architecture must contain at least one attention layer")
            
            # Check skip connection requirement
            if self.constraints.require_skip_connections:
                if skip_connections == 0:
                    errors.append("Architecture must contain at least one skip connection")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            logger.error(f"Architecture validation failed: {e}")
            return False, [f"Validation error: {str(e)}"]
    
    def estimate_parameters(self, layers: List[LayerConfig], connections: List[ConnectionConfig]) -> int:
        """Estimate the number of parameters in an architecture."""
        try:
            total_params = 0
            
            for layer in layers:
                if layer.layer_type == LayerType.DENSE:
                    # Assuming input size from previous layer
                    # This is a rough estimate
                    total_params += layer.units * 64  # Rough estimate
                elif layer.layer_type in [LayerType.LSTM, LayerType.GRU]:
                    # LSTM/GRU parameters: 4 * (input_size + hidden_size) * hidden_size
                    total_params += 4 * layer.units * layer.units
                elif layer.layer_type == LayerType.CONV1D:
                    # Conv1D parameters: kernel_size * input_channels * output_channels
                    if layer.kernel_size:
                        total_params += layer.kernel_size * 32 * layer.units  # Rough estimate
                elif layer.layer_type in [LayerType.ATTENTION, LayerType.MULTI_HEAD_ATTENTION]:
                    # Attention parameters: roughly proportional to units^2
                    total_params += layer.units * layer.units
            
            return total_params
            
        except Exception as e:
            logger.warning(f"Parameter estimation failed: {e}")
            return 0
    
    def get_random_layer_config(self, layer_type: Optional[LayerType] = None) -> LayerConfig:
        """Generate a random layer configuration."""
        try:
            if layer_type is None:
                layer_type = np.random.choice(self.available_layer_types)
            
            # Random activation
            activation = np.random.choice(self.available_activations)
            
            # Random units
            units = np.random.choice(self.layer_size_options)
            
            # Random dropout
            dropout_rate = np.random.choice(self.dropout_options)
            
            # Random batch norm
            batch_norm = np.random.choice([True, False])
            
            # Layer-specific parameters
            kernel_size = None
            attention_heads = None
            attention_dim = None
            
            if layer_type in [LayerType.CONV1D, LayerType.CONV2D]:
                kernel_size = np.random.choice(self.kernel_size_options)
            
            if layer_type in [LayerType.ATTENTION, LayerType.MULTI_HEAD_ATTENTION]:
                attention_heads = np.random.choice(self.attention_head_options)
                attention_dim = units
            
            return LayerConfig(
                layer_type=layer_type,
                activation=activation,
                units=units,
                dropout_rate=dropout_rate,
                batch_norm=batch_norm,
                kernel_size=kernel_size,
                attention_heads=attention_heads,
                attention_dim=attention_dim
            )
            
        except Exception as e:
            logger.warning(f"Random layer generation failed: {e}")
            # Return a simple default layer
            return LayerConfig(
                layer_type=LayerType.DENSE,
                activation=ActivationFunction.RELU,
                units=64,
                dropout_rate=0.2,
                batch_norm=True
            )
    
    def get_random_connection_config(self, from_layer: int, to_layer: int) -> ConnectionConfig:
        """Generate a random connection configuration."""
        try:
            connection_type = np.random.choice(self.available_connections)
            weight = np.random.uniform(0.5, 1.0)
            
            return ConnectionConfig(
                connection_type=connection_type,
                from_layer=from_layer,
                to_layer=to_layer,
                weight=weight
            )
            
        except Exception as e:
            logger.warning(f"Random connection generation failed: {e}")
            return ConnectionConfig(
                connection_type=ConnectionType.SEQUENTIAL,
                from_layer=from_layer,
                to_layer=to_layer,
                weight=1.0
            )


# Predefined search spaces for different regime detection tasks
def get_volatility_regime_search_space() -> SearchSpace:
    """Get search space optimized for volatility regime detection."""
    space = SearchSpace()
    
    # Focus on temporal and attention layers for volatility patterns
    space.available_layer_types = [
        LayerType.DENSE,
        LayerType.LSTM,
        LayerType.GRU,
        LayerType.ATTENTION,
        LayerType.BATCH_NORM,
        LayerType.DROPOUT
    ]
    
    space.constraints.require_temporal_layers = True
    space.constraints.require_attention = True
    space.constraints.max_conv_layers = 1
    space.constraints.max_rnn_layers = 3
    
    return space


def get_trend_regime_search_space() -> SearchSpace:
    """Get search space optimized for trend regime detection."""
    space = SearchSpace()
    
    # Focus on convolutional and dense layers for trend patterns
    space.available_layer_types = [
        LayerType.DENSE,
        LayerType.CONV1D,
        LayerType.LSTM,
        LayerType.BATCH_NORM,
        LayerType.DROPOUT,
        LayerType.RESIDUAL
    ]
    
    space.constraints.require_temporal_layers = True
    space.constraints.require_skip_connections = True
    space.constraints.max_conv_layers = 3
    space.constraints.max_rnn_layers = 2
    
    return space


def get_volume_regime_search_space() -> SearchSpace:
    """Get search space optimized for volume regime detection."""
    space = SearchSpace()
    
    # Focus on dense layers and attention for volume patterns
    space.available_layer_types = [
        LayerType.DENSE,
        LayerType.ATTENTION,
        LayerType.MULTI_HEAD_ATTENTION,
        LayerType.BATCH_NORM,
        LayerType.DROPOUT
    ]
    
    space.constraints.require_attention = True
    space.constraints.max_conv_layers = 0
    space.constraints.max_rnn_layers = 1
    
    return space


def get_hybrid_regime_search_space() -> SearchSpace:
    """Get comprehensive search space for hybrid regime detection."""
    space = SearchSpace()
    
    # Include all layer types for comprehensive search
    space.available_layer_types = [
        LayerType.DENSE,
        LayerType.CONV1D,
        LayerType.LSTM,
        LayerType.GRU,
        LayerType.ATTENTION,
        LayerType.MULTI_HEAD_ATTENTION,
        LayerType.BATCH_NORM,
        LayerType.DROPOUT,
        LayerType.RESIDUAL
    ]
    
    space.constraints.require_temporal_layers = True
    space.constraints.require_skip_connections = False
    space.constraints.max_conv_layers = 3
    space.constraints.max_rnn_layers = 3
    space.constraints.max_attention_layers = 2
    
    return space