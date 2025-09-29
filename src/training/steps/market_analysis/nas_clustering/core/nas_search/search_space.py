"""
Search Space

Implementation for NAS search space definition.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class LayerType(Enum):
    """Types of neural network layers."""
    DENSE = "dense"
    CONV2D = "conv2d"
    LSTM = "lstm"
    GRU = "gru"
    ATTENTION = "attention"


class ActivationType(Enum):
    """Types of activation functions."""
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SWISH = "swish"
    GELU = "gelu"


@dataclass
class SearchSpace:
    """Neural Architecture Search space definition."""
    
    # Layer configuration
    layer_types: List[LayerType] = None
    layer_widths: List[int] = None
    activations: List[ActivationType] = None
    
    # Architecture constraints
    max_layers: int = 10
    min_layers: int = 2
    max_width: int = 512
    min_width: int = 32
    
    # Regularization
    dropout_rates: List[float] = None
    weight_decay_rates: List[float] = None
    
    # Optimization
    learning_rates: List[float] = None
    batch_sizes: List[int] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.layer_types is None:
            self.layer_types = [LayerType.DENSE, LayerType.CONV2D, LayerType.LSTM]
        
        if self.layer_widths is None:
            self.layer_widths = [32, 64, 128, 256, 512]
        
        if self.activations is None:
            self.activations = [ActivationType.RELU, ActivationType.TANH, ActivationType.SIGMOID]
        
        if self.dropout_rates is None:
            self.dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.5]
        
        if self.weight_decay_rates is None:
            self.weight_decay_rates = [0.0, 1e-4, 1e-3, 1e-2]
        
        if self.learning_rates is None:
            self.learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]
        
        if self.batch_sizes is None:
            self.batch_sizes = [16, 32, 64, 128]
    
    def validate(self) -> bool:
        """Validate search space configuration."""
        # Check layer constraints
        if self.max_layers < self.min_layers:
            raise ValueError("max_layers must be >= min_layers")
        
        if self.max_width < self.min_width:
            raise ValueError("max_width must be >= min_width")
        
        # Check that we have at least one option for each parameter
        if not self.layer_types:
            raise ValueError("At least one layer type must be specified")
        
        if not self.layer_widths:
            raise ValueError("At least one layer width must be specified")
        
        if not self.activations:
            raise ValueError("At least one activation must be specified")
        
        return True
    
    def get_layer_types(self) -> List[str]:
        """Get list of layer types as strings."""
        return [layer_type.value for layer_type in self.layer_types]
    
    def get_activations(self) -> List[str]:
        """Get list of activations as strings."""
        return [activation.value for activation in self.activations]
    
    def get_valid_widths(self) -> List[int]:
        """Get list of valid layer widths."""
        return [w for w in self.layer_widths if self.min_width <= w <= self.max_width]
    
    def sample_architecture(self, num_layers: Optional[int] = None) -> Dict:
        """Sample a random architecture from the search space.
        
        Args:
            num_layers: Number of layers (random if None)
            
        Returns:
            Random architecture specification
        """
        if num_layers is None:
            num_layers = np.random.randint(self.min_layers, self.max_layers + 1)
        
        architecture = {
            'layers': [],
            'num_layers': num_layers
        }
        
        for i in range(num_layers):
            layer = {
                'type': np.random.choice(self.get_layer_types()),
                'width': np.random.choice(self.get_valid_widths()),
                'activation': np.random.choice(self.get_activations()),
                'dropout': np.random.choice(self.dropout_rates)
            }
            architecture['layers'].append(layer)
        
        return architecture
    
    def get_architecture_space_size(self) -> int:
        """Calculate the size of the architecture search space."""
        # Calculate combinations
        layer_type_combinations = len(self.layer_types) ** self.max_layers
        width_combinations = len(self.get_valid_widths()) ** self.max_layers
        activation_combinations = len(self.activations) ** self.max_layers
        dropout_combinations = len(self.dropout_rates) ** self.max_layers
        
        # Total combinations for each layer count
        total_combinations = 0
        for num_layers in range(self.min_layers, self.max_layers + 1):
            layer_combinations = (len(self.layer_types) * 
                                len(self.get_valid_widths()) * 
                                len(self.activations) * 
                                len(self.dropout_rates)) ** num_layers
            total_combinations += layer_combinations
        
        return total_combinations
    
    def is_valid_architecture(self, architecture: Dict) -> bool:
        """Check if architecture is valid within search space.
        
        Args:
            architecture: Architecture to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(architecture, dict):
            return False
        
        layers = architecture.get('layers', [])
        if not isinstance(layers, list):
            return False
        
        # Check number of layers
        if not (self.min_layers <= len(layers) <= self.max_layers):
            return False
        
        # Check each layer
        for layer in layers:
            if not isinstance(layer, dict):
                return False
            
            # Check layer type
            if 'type' not in layer or layer['type'] not in self.get_layer_types():
                return False
            
            # Check width
            if 'width' not in layer or not (self.min_width <= layer['width'] <= self.max_width):
                return False
            
            # Check activation
            if 'activation' not in layer or layer['activation'] not in self.get_activations():
                return False
            
            # Check dropout
            if 'dropout' in layer and layer['dropout'] not in self.dropout_rates:
                return False
        
        return True
    
    def get_architecture_complexity(self, architecture: Dict) -> float:
        """Calculate complexity score for architecture.
        
        Args:
            architecture: Architecture to analyze
            
        Returns:
            Complexity score
        """
        if not self.is_valid_architecture(architecture):
            return 0.0
        
        layers = architecture.get('layers', [])
        if not layers:
            return 0.0
        
        # Calculate complexity based on layers and parameters
        num_layers = len(layers)
        total_params = sum(layer.get('width', 0) for layer in layers)
        
        # Complexity score
        complexity = num_layers * 0.1 + total_params / 1000
        
        return complexity
    
    def get_architecture_parameters(self, architecture: Dict) -> int:
        """Calculate total parameter count for architecture.
        
        Args:
            architecture: Architecture to analyze
            
        Returns:
            Total parameter count
        """
        if not self.is_valid_architecture(architecture):
            return 0
        
        layers = architecture.get('layers', [])
        total_params = 0
        
        for i, layer in enumerate(layers):
            width = layer.get('width', 0)
            if i == 0:
                # Input layer
                total_params += width * 10  # Assume 10 input features
            else:
                # Hidden layers
                prev_width = layers[i-1].get('width', 0)
                total_params += prev_width * width + width  # weights + bias
        
        return total_params
    
    def get_search_space_summary(self) -> Dict:
        """Get summary of search space."""
        return {
            'layer_types': self.get_layer_types(),
            'layer_widths': self.get_valid_widths(),
            'activations': self.get_activations(),
            'dropout_rates': self.dropout_rates,
            'weight_decay_rates': self.weight_decay_rates,
            'learning_rates': self.learning_rates,
            'batch_sizes': self.batch_sizes,
            'max_layers': self.max_layers,
            'min_layers': self.min_layers,
            'max_width': self.max_width,
            'min_width': self.min_width,
            'space_size': self.get_architecture_space_size()
        }


def get_default_search_space() -> SearchSpace:
    """Get default search space configuration.
    
    Returns:
        Default search space
    """
    return SearchSpace(
        layer_types=[LayerType.DENSE, LayerType.CONV2D, LayerType.LSTM],
        layer_widths=[32, 64, 128, 256, 512],
        activations=[ActivationType.RELU, ActivationType.TANH, ActivationType.SIGMOID],
        max_layers=8,
        min_layers=2,
        max_width=512,
        min_width=32,
        dropout_rates=[0.0, 0.1, 0.2, 0.3, 0.5],
        weight_decay_rates=[0.0, 1e-4, 1e-3, 1e-2],
        learning_rates=[1e-4, 1e-3, 1e-2, 1e-1],
        batch_sizes=[16, 32, 64, 128]
    )


def create_custom_search_space(layer_types: List[str], 
                              layer_widths: List[int],
                              activations: List[str],
                              max_layers: int = 10,
                              min_layers: int = 2) -> SearchSpace:
    """Create custom search space.
    
    Args:
        layer_types: List of layer types
        layer_widths: List of layer widths
        activations: List of activations
        max_layers: Maximum number of layers
        min_layers: Minimum number of layers
        
    Returns:
        Custom search space
    """
    # Convert strings to enums
    layer_type_enums = [LayerType(lt) for lt in layer_types]
    activation_enums = [ActivationType(act) for act in activations]
    
    return SearchSpace(
        layer_types=layer_type_enums,
        layer_widths=layer_widths,
        activations=activation_enums,
        max_layers=max_layers,
        min_layers=min_layers
    )
