"""
Search Space Definition for NAS

Defines the search space for neural architecture search.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class SearchSpace:
    """
    Search space definition for neural architecture search.
    """
    
    def __init__(self, input_size: int, output_size: int = 3):
        """
        Initialize search space.
        
        Args:
            input_size: Size of input features
            output_size: Size of output (number of regimes)
        """
        self.input_size = input_size
        self.output_size = output_size
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Define search space parameters
        self.layer_types = ['linear', 'conv1d', 'lstm', 'gru']
        self.activations = ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu']
        self.optimizers = ['adam', 'sgd', 'rmsprop']
        self.learning_rates = [0.001, 0.01, 0.1]
        self.batch_sizes = [16, 32, 64, 128]
        
        # Layer size ranges
        self.min_layer_size = 16
        self.max_layer_size = 512
        self.min_layers = 2
        self.max_layers = 8
        
        self.logger.info(f"Search space initialized for input_size={input_size}, output_size={output_size}")
    
    def get_random_architecture(self) -> Dict[str, Any]:
        """Generate a random architecture from the search space."""
        try:
            # Random number of layers
            n_layers = np.random.randint(self.min_layers, self.max_layers + 1)
            
            layers = []
            current_size = self.input_size
            
            for i in range(n_layers):
                if i == n_layers - 1:
                    # Output layer
                    layer = {
                        'type': 'linear',
                        'input_size': current_size,
                        'output_size': self.output_size,
                        'activation': 'none'
                    }
                else:
                    # Hidden layer
                    layer_size = np.random.randint(self.min_layer_size, self.max_layer_size + 1)
                    layer = {
                        'type': np.random.choice(self.layer_types),
                        'input_size': current_size,
                        'output_size': layer_size,
                        'activation': np.random.choice(self.activations),
                        'dropout': np.random.uniform(0.0, 0.5)
                    }
                    
                    if layer['type'] in ['lstm', 'gru']:
                        layer['hidden_size'] = layer_size
                        layer['num_layers'] = np.random.randint(1, 3)
                    
                    current_size = layer_size
                
                layers.append(layer)
            
            # Training parameters
            training_params = {
                'optimizer': np.random.choice(self.optimizers),
                'learning_rate': np.random.choice(self.learning_rates),
                'batch_size': np.random.choice(self.batch_sizes),
                'epochs': np.random.randint(50, 200)
            }
            
            architecture = {
                'layers': layers,
                'training_params': training_params,
                'parameters_count': self._calculate_parameters(layers),
                'fitness_score': 0.0,
                'complexity_score': 0.0,
                'efficiency_score': 0.0
            }
            
            return architecture
            
        except Exception as e:
            self.logger.warning(f"Random architecture generation failed: {e}")
            return self._get_default_architecture()
    
    def _calculate_parameters(self, layers: List[Dict[str, Any]]) -> int:
        """Calculate total number of parameters in architecture."""
        try:
            total_params = 0
            
            for layer in layers:
                if layer['type'] == 'linear':
                    input_size = layer['input_size']
                    output_size = layer['output_size']
                    total_params += input_size * output_size + output_size
                elif layer['type'] == 'conv1d':
                    # Simplified conv1d parameter calculation
                    kernel_size = layer.get('kernel_size', 3)
                    input_channels = layer.get('input_channels', 1)
                    output_channels = layer.get('output_channels', 32)
                    total_params += kernel_size * input_channels * output_channels + output_channels
                elif layer['type'] in ['lstm', 'gru']:
                    # Simplified RNN parameter calculation
                    input_size = layer['input_size']
                    hidden_size = layer['hidden_size']
                    num_layers = layer.get('num_layers', 1)
                    
                    # LSTM: 4 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)
                    # GRU: 3 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)
                    multiplier = 4 if layer['type'] == 'lstm' else 3
                    layer_params = multiplier * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)
                    total_params += layer_params * num_layers
            
            return total_params
            
        except Exception as e:
            self.logger.warning(f"Parameter calculation failed: {e}")
            return 1000
    
    def _get_default_architecture(self) -> Dict[str, Any]:
        """Get default architecture as fallback."""
        return {
            'layers': [
                {
                    'type': 'linear',
                    'input_size': self.input_size,
                    'output_size': 64,
                    'activation': 'relu',
                    'dropout': 0.2
                },
                {
                    'type': 'linear',
                    'input_size': 64,
                    'output_size': self.output_size,
                    'activation': 'none'
                }
            ],
            'training_params': {
                'optimizer': 'adam',
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 100
            },
            'parameters_count': self.input_size * 64 + 64 * self.output_size,
            'fitness_score': 0.0,
            'complexity_score': 0.5,
            'efficiency_score': 0.5
        }
    
    def validate_architecture(self, architecture: Dict[str, Any]) -> bool:
        """Validate architecture against search space constraints."""
        try:
            if 'layers' not in architecture:
                return False
            
            layers = architecture['layers']
            
            # Check layer count
            if len(layers) < self.min_layers or len(layers) > self.max_layers:
                return False
            
            # Check each layer
            for layer in layers:
                if not self._validate_layer(layer):
                    return False
            
            # Check layer connections
            for i in range(1, len(layers)):
                if layers[i]['input_size'] != layers[i-1]['output_size']:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Architecture validation failed: {e}")
            return False
    
    def _validate_layer(self, layer: Dict[str, Any]) -> bool:
        """Validate individual layer."""
        try:
            # Check required fields
            required_fields = ['type', 'input_size', 'output_size']
            for field in required_fields:
                if field not in layer:
                    return False
            
            # Check layer type
            if layer['type'] not in self.layer_types:
                return False
            
            # Check sizes
            if layer['input_size'] < 1 or layer['output_size'] < 1:
                return False
            
            if layer['output_size'] < self.min_layer_size or layer['output_size'] > self.max_layer_size:
                return False
            
            # Check activation
            if 'activation' in layer and layer['activation'] not in self.activations + ['none']:
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Layer validation failed: {e}")
            return False
    
    def get_search_space_summary(self) -> Dict[str, Any]:
        """Get summary of search space parameters."""
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'layer_types': self.layer_types,
            'activations': self.activations,
            'optimizers': self.optimizers,
            'learning_rates': self.learning_rates,
            'batch_sizes': self.batch_sizes,
            'min_layer_size': self.min_layer_size,
            'max_layer_size': self.max_layer_size,
            'min_layers': self.min_layers,
            'max_layers': self.max_layers,
            'total_combinations': self._estimate_total_combinations()
        }
    
    def _estimate_total_combinations(self) -> int:
        """Estimate total number of possible architecture combinations."""
        try:
            # Rough estimate based on layer types, sizes, and activations
            layer_type_combinations = len(self.layer_types) ** self.max_layers
            size_combinations = ((self.max_layer_size - self.min_layer_size + 1) ** self.max_layers)
            activation_combinations = len(self.activations) ** self.max_layers
            
            # This is a very rough estimate
            total = layer_type_combinations * size_combinations * activation_combinations
            return min(total, 10**12)  # Cap at reasonable number
            
        except Exception as e:
            self.logger.warning(f"Combination estimation failed: {e}")
            return 1000000

def get_default_search_space(input_size: int, output_size: int = 3) -> SearchSpace:
    """Get default search space configuration."""
    return SearchSpace(input_size, output_size)
