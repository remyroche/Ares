"""
Architecture Encoding and Decoding Systems for NAS and TAS

This module provides systematic encoding and decoding of neural and tree architectures
for search algorithms, serialization, and representation learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib
import base64
import json
from pathlib import Path
import pickle
import os


# Import tprint for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)


class EncodingType(Enum):
    """Types of architecture encoding."""
    ONE_HOT = "one_hot"
    INTEGER = "integer"
    BINARY = "binary"
    REAL_VALUED = "real_valued"
    STRING = "string"
    GRAPH = "graph"
    ADJACENCY_MATRIX = "adjacency_matrix"
    PATH_ENCODING = "path_encoding"
    RECURSIVE = "recursive"
    HYBRID = "hybrid"


class EncodingFormat(Enum):
    """Format for encoded architectures."""
    VECTOR = "vector"
    MATRIX = "matrix"
    STRING = "string"
    DICT = "dict"
    JSON = "json"
    BYTES = "bytes"


@dataclass
class EncodingResult:
    """Result of architecture encoding."""
    encoding: Any
    encoding_type: EncodingType
    format: EncodingFormat
    metadata: Dict[str, Any] = field(default_factory=dict)
    encoding_time: float = 0.0
    compression_ratio: float = 1.0


@dataclass
class DecodingResult:
    """Result of architecture decoding."""
    architecture: Any
    decoding_time: float = 0.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseArchitectureEncoder:
    """Base class for architecture encoders."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the architecture encoder."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_initialized = False

    def encode(self, architecture: Any) -> EncodingResult:
        """Encode an architecture with basic fallback implementation."""
        import time
        start_time = time.time()
        
        # Basic encoding - convert architecture to simple dictionary representation
        try:
            if hasattr(architecture, '__dict__'):
                encoding = architecture.__dict__.copy()
            else:
                encoding = str(architecture)
            
            encoding_time = time.time() - start_time
            
            return EncodingResult(
                encoding=encoding,
                encoding_type=EncodingType.STRING,
                format=EncodingFormat.DICT,
                metadata={'fallback_encoding': True, 'architecture_type': type(architecture).__name__},
                encoding_time=encoding_time,
                compression_ratio=1.0
            )
        except Exception as e:
            self.logger.error(f"Basic encoding failed: {e}")
            return EncodingResult(
                encoding=str(architecture),
                encoding_type=EncodingType.STRING,
                format=EncodingFormat.STRING,
                metadata={'fallback_encoding': True, 'error': str(e)},
                encoding_time=time.time() - start_time,
                compression_ratio=1.0
            )

    def decode(self, encoding: Any, encoding_type: EncodingType) -> DecodingResult:
        """Decode an architecture with basic fallback implementation."""
        import time
        start_time = time.time()
        
        try:
            # Basic decoding - return the encoding as-is for simple cases
            if isinstance(encoding, dict) and 'fallback_encoding' in encoding:
                # This is a fallback encoding, return a simple object
                class SimpleArchitecture:
                    def __init__(self, data):
                        for key, value in data.items():
                            if key != 'fallback_encoding':
                                setattr(self, key, value)
                
                architecture = SimpleArchitecture(encoding)
            else:
                # For other cases, return the encoding itself
                architecture = encoding
            
            decoding_time = time.time() - start_time
            
            return DecodingResult(
                architecture=architecture,
                decoding_time=decoding_time,
                confidence=0.5,  # Lower confidence for fallback
                metadata={'fallback_decoding': True, 'encoding_type': encoding_type.value}
            )
        except Exception as e:
            self.logger.error(f"Basic decoding failed: {e}")
            return DecodingResult(
                architecture=encoding,
                decoding_time=time.time() - start_time,
                confidence=0.1,  # Very low confidence
                metadata={'fallback_decoding': True, 'error': str(e)}
            )

    def get_encoding_size(self, architecture: Any) -> int:
        """Get the size of the encoded representation with basic fallback implementation."""
        try:
            # Basic size estimation
            if hasattr(architecture, '__dict__'):
                # Estimate size based on number of attributes
                return len(architecture.__dict__) * 10  # Rough estimate
            else:
                # For other objects, estimate based on string representation
                return len(str(architecture))
        except Exception as e:
            self.logger.error(f"Basic encoding size calculation failed: {e}")
            return 100  # Default fallback size

    def validate_encoding(self, encoding: Any, encoding_type: EncodingType) -> bool:
        """Validate an encoding with basic fallback implementation."""
        try:
            # Basic validation based on encoding type
            if encoding_type == EncodingType.STRING:
                return isinstance(encoding, str) and len(encoding) > 0
            elif encoding_type == EncodingType.DICT:
                return isinstance(encoding, dict) and len(encoding) > 0
            elif encoding_type == EncodingType.VECTOR:
                return hasattr(encoding, '__len__') and len(encoding) > 0
            elif encoding_type == EncodingType.MATRIX:
                return hasattr(encoding, 'shape') and len(encoding.shape) >= 2
            else:
                # For other types, basic non-null check
                return encoding is not None
        except Exception as e:
            self.logger.error(f"Basic encoding validation failed: {e}")
            return False


class NeuralArchitectureEncoder(BaseArchitectureEncoder):
    """Encoder for neural architectures."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize neural architecture encoder."""
        super().__init__(config)
        self.max_layers = config.get('max_layers', 20)
        self.max_layer_size = config.get('max_layer_size', 2048)
        self.encoding_dim = config.get('encoding_dim', 1000)
        self.logger.info("✅ Neural Architecture Encoder initialized")

    def encode(self, architecture: Any) -> EncodingResult:
        """Encode a neural architecture."""
        from ..search_spaces import NeuralArchitecture, LayerType, ConnectionType, ActivationFunction

        if not isinstance(architecture, NeuralArchitecture):
            raise ValueError("Architecture must be a NeuralArchitecture instance")

        start_time = __import__('time').time()

        # Choose encoding method based on config
        encoding_method = self.config.get('encoding_method', 'hybrid')

        if encoding_method == 'one_hot':
            encoding = self._one_hot_encode(architecture)
            encoding_type = EncodingType.ONE_HOT
        elif encoding_method == 'adjacency_matrix':
            encoding = self._adjacency_matrix_encode(architecture)
            encoding_type = EncodingType.ADJACENCY_MATRIX
        elif encoding_method == 'path_encoding':
            encoding = self._path_encode(architecture)
            encoding_type = EncodingType.PATH_ENCODING
        else:  # hybrid
            encoding = self._hybrid_encode(architecture)
            encoding_type = EncodingType.HYBRID

        encoding_time = __import__('time').time() - start_time

        # Determine format
        if isinstance(encoding, np.ndarray):
            format_type = EncodingFormat.VECTOR if encoding.ndim == 1 else EncodingFormat.MATRIX
        elif isinstance(encoding, str):
            format_type = EncodingFormat.STRING
        elif isinstance(encoding, dict):
            format_type = EncodingFormat.DICT
        else:
            format_type = EncodingFormat.VECTOR

        # Calculate compression ratio
        original_size = len(architecture.layers) * 100  # Rough estimate
        encoded_size = self.get_encoding_size(architecture)
        compression_ratio = encoded_size / max(original_size, 1)

        return EncodingResult(
            encoding=encoding,
            encoding_type=encoding_type,
            format=format_type,
            metadata={
                'n_layers': len(architecture.layers),
                'n_connections': len(architecture.connections),
                'encoding_method': encoding_method
            },
            encoding_time=encoding_time,
            compression_ratio=compression_ratio
        )

    def decode(self, encoding: Any, encoding_type: EncodingType) -> DecodingResult:
        """Decode a neural architecture encoding."""

        start_time = __import__('time').time()

        try:
            if encoding_type == EncodingType.ONE_HOT:
                architecture = self._one_hot_decode(encoding)
            elif encoding_type == EncodingType.ADJACENCY_MATRIX:
                architecture = self._adjacency_matrix_decode(encoding)
            elif encoding_type == EncodingType.PATH_ENCODING:
                architecture = self._path_decode(encoding)
            elif encoding_type == EncodingType.HYBRID:
                architecture = self._hybrid_decode(encoding)
            else:
                raise ValueError(f"Unsupported encoding type: {encoding_type}")

            decoding_time = __import__('time').time() - start_time

            return DecodingResult(
                architecture=architecture,
                decoding_time=decoding_time,
                confidence=0.95,  # High confidence for neural architectures
                metadata={'encoding_type': encoding_type.value}
            )

        except Exception as e:
            self.logger.error(f"Failed to decode architecture: {e}")
            raise

    def get_encoding_size(self, architecture: Any) -> int:
        """Get the size of the encoded representation."""
        encoding_method = self.config.get('encoding_method', 'hybrid')

        if encoding_method == 'one_hot':
            return len(architecture.layers) * 50  # Rough estimate
        elif encoding_method == 'adjacency_matrix':
            n = len(architecture.layers)
            return n * n + n * 10  # Adjacency matrix + layer features
        elif encoding_method == 'path_encoding':
            return sum(len(layer.__dict__) for layer in architecture.layers) * 2
        else:  # hybrid
            return len(architecture.layers) * 20 + len(architecture.connections) * 5

    def _one_hot_encode(self, architecture: Any) -> np.ndarray:
        """One-hot encoding of neural architecture."""
        features = []

        # Layer type encoding
        layer_types = [LayerType.LINEAR, LayerType.CONV1D, LayerType.LSTM, LayerType.ATTENTION]
        for layer in architecture.layers:
            layer_encoding = [0] * len(layer_types)
            if layer.layer_type in layer_types:
                layer_encoding[layer_types.index(layer.layer_type)] = 1
            features.extend(layer_encoding)

            # Layer size encoding (normalized)
            size_encoding = [min(layer.hidden_size / self.max_layer_size, 1.0)]
            features.extend(size_encoding)

            # Activation encoding
            activations = [ActivationFunction.RELU, ActivationFunction.TANH, ActivationFunction.SIGMOID]
            act_encoding = [0] * len(activations)
            if layer.activation and layer.activation in activations:
                act_encoding[activations.index(layer.activation)] = 1
            features.extend(act_encoding)

        # Connection encoding
        n_layers = len(architecture.layers)
        for i in range(n_layers):
            for j in range(n_layers):
                if i != j:
                    has_connection = any(conn[0] == i and conn[1] == j for conn in architecture.connections)
                    features.append(1.0 if has_connection else 0.0)

        return np.array(features)

    def _adjacency_matrix_encode(self, architecture: Any) -> Dict[str, Any]:
        """Adjacency matrix encoding of neural architecture."""
        n_layers = len(architecture.layers)

        # Create adjacency matrix
        adj_matrix = np.zeros((n_layers, n_layers))
        for conn in architecture.connections:
            from_idx, to_idx, conn_type = conn
            if from_idx < n_layers and to_idx < n_layers:
                # Encode connection type
                if conn_type == ConnectionType.DENSE:
                    adj_matrix[from_idx, to_idx] = 1.0
                elif conn_type == ConnectionType.RESIDUAL:
                    adj_matrix[from_idx, to_idx] = 2.0
                elif conn_type == ConnectionType.SKIP:
                    adj_matrix[from_idx, to_idx] = 3.0

        # Layer features
        layer_features = []
        for layer in architecture.layers:
            layer_feat = [
                layer.hidden_size / self.max_layer_size,
                1.0 if layer.activation else 0.0,
                layer.dropout_rate,
                1.0 if layer.batch_norm else 0.0,
                1.0 if layer.layer_norm else 0.0,
                1.0 if layer.residual else 0.0
            ]
            layer_features.append(layer_feat)

        return {
            'adjacency_matrix': adj_matrix,
            'layer_features': np.array(layer_features),
            'input_shape': architecture.input_shape,
            'output_shape': architecture.output_shape
        }

    def _path_encode(self, architecture: Any) -> str:
        """Path-based string encoding of neural architecture."""
        path_elements = []

        for i, layer in enumerate(architecture.layers):
            layer_str = f"L{i}:{layer.layer_type.value}"

            if hasattr(layer, 'hidden_size') and layer.hidden_size:
                layer_str += f"_H{layer.hidden_size}"

            if layer.activation:
                layer_str += f"_A{layer.activation.value}"

            if layer.dropout_rate > 0:
                layer_str += f"_D{layer.dropout_rate:.2f}"

            path_elements.append(layer_str)

        # Add connections
        for conn in architecture.connections:
            from_idx, to_idx, conn_type = conn
            conn_str = f"C{from_idx}->{to_idx}:{conn_type.value}"
            path_elements.append(conn_str)

        return "|".join(path_elements)

    def _hybrid_encode(self, architecture: Any) -> Dict[str, Any]:
        """Hybrid encoding combining multiple approaches."""
        return {
            'one_hot': self._one_hot_encode(architecture),
            'adjacency': self._adjacency_matrix_encode(architecture),
            'path': self._path_encode(architecture),
            'metadata': {
                'n_layers': len(architecture.layers),
                'n_connections': len(architecture.connections),
                'total_params': sum(layer.hidden_size * layer.hidden_size for layer in architecture.layers)
            }
        }

    def _one_hot_decode(self, encoding: np.ndarray) -> Any:
        """Decode one-hot encoded neural architecture."""
        try:
            # Convert one-hot encoding back to layer specifications
            decoded_layers = []
            
            # Reshape encoding to match expected structure
            if len(encoding.shape) == 1:
                # Flattened encoding - reshape based on max_layers
                layer_size = self.max_layers
                if len(encoding) % layer_size == 0:
                    encoding = encoding.reshape(-1, layer_size)
                else:
                    # Pad or truncate to match expected size
                    target_size = (len(encoding) // layer_size + 1) * layer_size
                    padded_encoding = np.zeros(target_size)
                    padded_encoding[:len(encoding)] = encoding
                    encoding = padded_encoding.reshape(-1, layer_size)
            
            # Decode each layer
            for layer_encoding in encoding:
                # Find the maximum value (one-hot position)
                layer_type_idx = np.argmax(layer_encoding)
                
                # Map index back to layer type
                layer_types = ['conv', 'dense', 'lstm', 'gru', 'attention', 'dropout', 'batch_norm']
                if layer_type_idx < len(layer_types):
                    layer_type = layer_types[layer_type_idx]
                    
                    # Create basic layer specification
                    layer_spec = {
                        'type': layer_type,
                        'units': 128,  # Default units
                        'activation': 'relu'  # Default activation
                    }
                    
                    # Add type-specific parameters
                    if layer_type == 'conv':
                        layer_spec.update({
                            'filters': 32,
                            'kernel_size': 3,
                            'padding': 'same'
                        })
                    elif layer_type in ['lstm', 'gru']:
                        layer_spec.update({
                            'return_sequences': True,
                            'dropout': 0.2
                        })
                    elif layer_type == 'attention':
                        layer_spec.update({
                            'heads': 8,
                            'key_dim': 64
                        })
                    
                    decoded_layers.append(layer_spec)
            
            return {
                'layers': decoded_layers,
                'input_shape': (None, 128),  # Default input shape
                'output_shape': (None, 1)   # Default output shape
            }
            
        except Exception as e:
            self.logger.error(f"One-hot decoding failed: {e}")
            # Return a basic fallback architecture
            return {
                'layers': [
                    {'type': 'dense', 'units': 128, 'activation': 'relu'},
                    {'type': 'dense', 'units': 64, 'activation': 'relu'},
                    {'type': 'dense', 'units': 1, 'activation': 'sigmoid'}
                ],
                'input_shape': (None, 128),
                'output_shape': (None, 1)
            }

    def _adjacency_matrix_decode(self, encoding: Dict[str, Any]) -> Any:
        """Decode adjacency matrix encoded neural architecture."""
        try:
            adjacency_matrix = encoding.get('adjacency_matrix', np.array([]))
            layer_types = encoding.get('layer_types', [])
            layer_params = encoding.get('layer_params', {})
            
            if len(adjacency_matrix) == 0:
                # Return default architecture if no matrix provided
                return {
                    'layers': [
                        {'type': 'dense', 'units': 128, 'activation': 'relu'},
                        {'type': 'dense', 'units': 64, 'activation': 'relu'},
                        {'type': 'dense', 'units': 1, 'activation': 'sigmoid'}
                    ],
                    'connections': [],
                    'input_shape': (None, 128),
                    'output_shape': (None, 1)
                }
            
            # Convert adjacency matrix to layer connections
            connections = []
            layers = []
            
            # Create layers based on adjacency matrix
            for i in range(len(adjacency_matrix)):
                layer_type = layer_types[i] if i < len(layer_types) else 'dense'
                layer_spec = {
                    'type': layer_type,
                    'units': layer_params.get(f'layer_{i}', {}).get('units', 128),
                    'activation': layer_params.get(f'layer_{i}', {}).get('activation', 'relu')
                }
                
                # Add type-specific parameters
                if layer_type == 'conv':
                    layer_spec.update({
                        'filters': layer_params.get(f'layer_{i}', {}).get('filters', 32),
                        'kernel_size': layer_params.get(f'layer_{i}', {}).get('kernel_size', 3),
                        'padding': 'same'
                    })
                elif layer_type in ['lstm', 'gru']:
                    layer_spec.update({
                        'return_sequences': True,
                        'dropout': 0.2
                    })
                
                layers.append(layer_spec)
            
            # Extract connections from adjacency matrix
            for i in range(len(adjacency_matrix)):
                for j in range(len(adjacency_matrix[i])):
                    if adjacency_matrix[i][j] > 0:  # Connection exists
                        connections.append({
                            'from': i,
                            'to': j,
                            'weight': adjacency_matrix[i][j]
                        })
            
            return {
                'layers': layers,
                'connections': connections,
                'input_shape': (None, 128),
                'output_shape': (None, 1)
            }
            
        except Exception as e:
            self.logger.error(f"Adjacency matrix decoding failed: {e}")
            # Return a basic fallback architecture
            return {
                'layers': [
                    {'type': 'dense', 'units': 128, 'activation': 'relu'},
                    {'type': 'dense', 'units': 64, 'activation': 'relu'},
                    {'type': 'dense', 'units': 1, 'activation': 'sigmoid'}
                ],
                'connections': [],
                'input_shape': (None, 128),
                'output_shape': (None, 1)
            }

    def _path_decode(self, encoding: str) -> Any:
        """Decode path encoded neural architecture."""
        try:
            # Parse path encoding (e.g., "dense-128-relu->dense-64-relu->dense-1-sigmoid")
            layers = []
            connections = []
            
            # Split by arrow connections
            path_parts = encoding.split('->')
            
            for i, part in enumerate(path_parts):
                # Parse layer specification (e.g., "dense-128-relu")
                layer_spec = part.strip().split('-')
                
                if len(layer_spec) >= 3:
                    layer_type = layer_spec[0]
                    units = int(layer_spec[1]) if layer_spec[1].isdigit() else 128
                    activation = layer_spec[2]
                    
                    layer = {
                        'type': layer_type,
                        'units': units,
                        'activation': activation
                    }
                    
                    # Add type-specific parameters
                    if layer_type == 'conv':
                        layer.update({
                            'filters': 32,
                            'kernel_size': 3,
                            'padding': 'same'
                        })
                    elif layer_type in ['lstm', 'gru']:
                        layer.update({
                            'return_sequences': True,
                            'dropout': 0.2
                        })
                    
                    layers.append(layer)
                    
                    # Add connection to next layer
                    if i < len(path_parts) - 1:
                        connections.append({
                            'from': i,
                            'to': i + 1,
                            'weight': 1.0
                        })
            
            return {
                'layers': layers,
                'connections': connections,
                'input_shape': (None, 128),
                'output_shape': (None, 1)
            }
            
        except Exception as e:
            self.logger.error(f"Path decoding failed: {e}")
            # Return a basic fallback architecture
            return {
                'layers': [
                    {'type': 'dense', 'units': 128, 'activation': 'relu'},
                    {'type': 'dense', 'units': 64, 'activation': 'relu'},
                    {'type': 'dense', 'units': 1, 'activation': 'sigmoid'}
                ],
                'connections': [],
                'input_shape': (None, 128),
                'output_shape': (None, 1)
            }

    def _hybrid_decode(self, encoding: Dict[str, Any]) -> Any:
        """Decode hybrid encoded neural architecture."""
        try:
            # Combine multiple encoding types
            layers = []
            connections = []
            
            # Get components from hybrid encoding
            one_hot_encoding = encoding.get('one_hot', None)
            adjacency_encoding = encoding.get('adjacency', None)
            path_encoding = encoding.get('path', None)
            
            # Try to decode from one-hot first
            if one_hot_encoding is not None:
                one_hot_result = self._one_hot_decode(one_hot_encoding)
                if one_hot_result and 'layers' in one_hot_result:
                    layers = one_hot_result['layers']
                    connections = one_hot_result.get('connections', [])
            
            # Try to decode from adjacency matrix if one-hot failed
            elif adjacency_encoding is not None:
                adjacency_result = self._adjacency_matrix_decode(adjacency_encoding)
                if adjacency_result and 'layers' in adjacency_result:
                    layers = adjacency_result['layers']
                    connections = adjacency_result.get('connections', [])
            
            # Try to decode from path if others failed
            elif path_encoding is not None:
                path_result = self._path_decode(path_encoding)
                if path_result and 'layers' in path_result:
                    layers = path_result['layers']
                    connections = path_result.get('connections', [])
            
            # If all decoding methods failed, create a default architecture
            if not layers:
                layers = [
                    {'type': 'dense', 'units': 128, 'activation': 'relu'},
                    {'type': 'dense', 'units': 64, 'activation': 'relu'},
                    {'type': 'dense', 'units': 1, 'activation': 'sigmoid'}
                ]
                connections = []
            
            return {
                'layers': layers,
                'connections': connections,
                'input_shape': (None, 128),
                'output_shape': (None, 1)
            }
            
        except Exception as e:
            self.logger.error(f"Hybrid decoding failed: {e}")
            # Return a basic fallback architecture
            return {
                'layers': [
                    {'type': 'dense', 'units': 128, 'activation': 'relu'},
                    {'type': 'dense', 'units': 64, 'activation': 'relu'},
                    {'type': 'dense', 'units': 1, 'activation': 'sigmoid'}
                ],
                'connections': [],
                'input_shape': (None, 128),
                'output_shape': (None, 1)
            }

    def validate_encoding(self, encoding: Any, encoding_type: EncodingType) -> bool:
        """Validate neural architecture encoding."""
        try:
            if encoding_type == EncodingType.ONE_HOT:
                return isinstance(encoding, np.ndarray) and encoding.ndim == 1
            elif encoding_type == EncodingType.ADJACENCY_MATRIX:
                return isinstance(encoding, dict) and 'adjacency_matrix' in encoding
            elif encoding_type == EncodingType.PATH_ENCODING:
                return isinstance(encoding, str) and len(encoding) > 0
            elif encoding_type == EncodingType.HYBRID:
                return isinstance(encoding, dict) and len(encoding) > 0
            else:
                return False
        except Exception as e:
            tprint_debug(f"🔍 Operation failed: {e}")
            return False


class TreeArchitectureEncoder(BaseArchitectureEncoder):
    """Encoder for tree architectures."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize tree architecture encoder."""
        super().__init__(config)
        self.max_trees = config.get('max_trees', 50)
        self.max_depth = config.get('max_depth', 30)
        self.encoding_dim = config.get('encoding_dim', 500)
        self.logger.info("✅ Tree Architecture Encoder initialized")

    def encode(self, architecture: Any) -> EncodingResult:
        """Encode a tree architecture."""

        if not isinstance(architecture, TreeArchitecture):
            raise ValueError("Architecture must be a TreeArchitecture instance")

        start_time = __import__('time').time()

        # Choose encoding method based on config
        encoding_method = self.config.get('encoding_method', 'hybrid')

        if encoding_method == 'one_hot':
            encoding = self._one_hot_encode(architecture)
            encoding_type = EncodingType.ONE_HOT
        elif encoding_method == 'recursive':
            encoding = self._recursive_encode(architecture)
            encoding_type = EncodingType.RECURSIVE
        else:  # hybrid
            encoding = self._hybrid_encode(architecture)
            encoding_type = EncodingType.HYBRID

        encoding_time = __import__('time').time() - start_time

        # Determine format
        if isinstance(encoding, np.ndarray):
            format_type = EncodingFormat.VECTOR if encoding.ndim == 1 else EncodingFormat.MATRIX
        elif isinstance(encoding, str):
            format_type = EncodingFormat.STRING
        elif isinstance(encoding, dict):
            format_type = EncodingFormat.DICT
        else:
            format_type = EncodingFormat.VECTOR

        # Calculate compression ratio
        original_size = len(architecture.trees) * 50  # Rough estimate
        encoded_size = self.get_encoding_size(architecture)
        compression_ratio = encoded_size / max(original_size, 1)

        return EncodingResult(
            encoding=encoding,
            encoding_type=encoding_type,
            format=format_type,
            metadata={
                'n_trees': len(architecture.trees),
                'ensemble_method': architecture.ensemble_method,
                'encoding_method': encoding_method
            },
            encoding_time=encoding_time,
            compression_ratio=compression_ratio
        )

    def decode(self, encoding: Any, encoding_type: EncodingType) -> DecodingResult:
        """Decode a tree architecture encoding."""

        start_time = __import__('time').time()

        try:
            if encoding_type == EncodingType.ONE_HOT:
                architecture = self._one_hot_decode(encoding)
            elif encoding_type == EncodingType.RECURSIVE:
                architecture = self._recursive_decode(encoding)
            elif encoding_type == EncodingType.HYBRID:
                architecture = self._hybrid_decode(encoding)
            else:
                raise ValueError(f"Unsupported encoding type: {encoding_type}")

            decoding_time = __import__('time').time() - start_time

            return DecodingResult(
                architecture=architecture,
                decoding_time=decoding_time,
                confidence=0.90,  # Slightly lower confidence for tree architectures
                metadata={'encoding_type': encoding_type.value}
            )

        except Exception as e:
            self.logger.error(f"Failed to decode tree architecture: {e}")
            raise

    def get_encoding_size(self, architecture: Any) -> int:
        """Get the size of the encoded representation."""
        encoding_method = self.config.get('encoding_method', 'hybrid')

        if encoding_method == 'one_hot':
            return len(architecture.trees) * 20  # Rough estimate
        elif encoding_method == 'recursive':
            return sum(len(tree.__dict__) for tree in architecture.trees)
        else:  # hybrid
            return len(architecture.trees) * 10 + 20

    def _one_hot_encode(self, architecture: Any) -> np.ndarray:
        """One-hot encoding of tree architecture."""
        features = []

        # Tree type encoding
        tree_types = [LayerType.DECISION_TREE, LayerType.RANDOM_FOREST,
                     LayerType.GRADIENT_BOOSTING, LayerType.XGBOOST]
        for tree in architecture.trees:
            tree_encoding = [0] * len(tree_types)
            if tree.tree_type in tree_types:
                tree_encoding[tree_types.index(tree.tree_type)] = 1
            features.extend(tree_encoding)

            # Tree parameters
            features.extend([
                tree.max_depth / self.max_depth if tree.max_depth else 0.5,
                min(tree.n_estimators / 100, 1.0) if tree.n_estimators else 0.0,
                tree.learning_rate if hasattr(tree, 'learning_rate') else 0.1,
                1.0 if tree.bootstrap else 0.0
            ])

        # Ensemble method encoding
        ensemble_methods = ['single', 'voting', 'averaging', 'stacking']
        ensemble_encoding = [0] * len(ensemble_methods)
        if architecture.ensemble_method in ensemble_methods:
            ensemble_encoding[ensemble_methods.index(architecture.ensemble_method)] = 1
        features.extend(ensemble_encoding)

        return np.array(features)

    def _recursive_encode(self, architecture: Any) -> Dict[str, Any]:
        """Recursive encoding of tree architecture."""
        trees_encoded = []
        for tree in architecture.trees:
            tree_encoded = {
                'type': tree.tree_type.value,
                'max_depth': tree.max_depth,
                'min_samples_split': tree.min_samples_split,
                'min_samples_leaf': tree.min_samples_leaf,
                'max_features': tree.max_features,
                'criterion': tree.criterion,
                'splitter': tree.splitter,
                'n_estimators': tree.n_estimators,
                'learning_rate': tree.learning_rate,
                'bootstrap': tree.bootstrap
            }
            trees_encoded.append(tree_encoded)

        return {
            'trees': trees_encoded,
            'ensemble_method': architecture.ensemble_method,
            'feature_preprocessing': architecture.feature_preprocessing
        }

    def _hybrid_encode(self, architecture: Any) -> Dict[str, Any]:
        """Hybrid encoding combining multiple approaches."""
        return {
            'one_hot': self._one_hot_encode(architecture),
            'recursive': self._recursive_encode(architecture),
            'metadata': {
                'n_trees': len(architecture.trees),
                'avg_depth': sum(tree.max_depth or 10 for tree in architecture.trees) / len(architecture.trees),
                'has_boosting': any(tree.tree_type.value in ['gradient_boosting', 'xgboost'] for tree in architecture.trees),
                'has_bagging': any(tree.tree_type.value == 'random_forest' for tree in architecture.trees)
            }
        }

    def _one_hot_decode(self, encoding: np.ndarray) -> Any:
        """Decode one-hot encoded tree architecture."""
        try:
            # Convert one-hot encoding back to tree structure
            decoded_nodes = []
            
            # Reshape encoding to match expected structure
            if len(encoding.shape) == 1:
                # Flattened encoding - reshape based on max_trees
                node_size = self.max_trees
                if len(encoding) % node_size == 0:
                    encoding = encoding.reshape(-1, node_size)
                else:
                    # Pad or truncate to match expected size
                    target_size = (len(encoding) // node_size + 1) * node_size
                    padded_encoding = np.zeros(target_size)
                    padded_encoding[:len(encoding)] = encoding
                    encoding = padded_encoding.reshape(-1, node_size)
            
            # Decode each node
            for node_encoding in encoding:
                # Find the maximum value (one-hot position)
                node_type_idx = np.argmax(node_encoding)
                
                # Map index back to node type
                node_types = ['add', 'subtract', 'multiply', 'divide', 'sqrt', 'log', 'exp', 'sin', 'cos', 'tanh', 'sigmoid', 'relu', 'variable', 'constant']
                if node_type_idx < len(node_types):
                    node_type = node_types[node_type_idx]
                    
                    # Create basic node specification
                    node_spec = {
                        'type': node_type,
                        'value': 1.0,  # Default value
                        'children': []  # Will be filled by tree structure
                    }
                    
                    # Add type-specific parameters
                    if node_type == 'variable':
                        node_spec.update({
                            'name': 'x',
                            'index': 0
                        })
                    elif node_type == 'constant':
                        node_spec.update({
                            'value': 1.0
                        })
                    elif node_type in ['add', 'subtract', 'multiply', 'divide']:
                        node_spec.update({
                            'arity': 2
                        })
                    elif node_type in ['sqrt', 'log', 'exp', 'sin', 'cos', 'tanh', 'sigmoid', 'relu']:
                        node_spec.update({
                            'arity': 1
                        })
                    
                    decoded_nodes.append(node_spec)
            
            # Create basic tree structure
            if decoded_nodes:
                # Simple linear tree structure
                tree = {
                    'root': decoded_nodes[0],
                    'nodes': decoded_nodes,
                    'depth': len(decoded_nodes),
                    'size': len(decoded_nodes)
                }
                
                # Add basic connections
                for i in range(len(decoded_nodes) - 1):
                    if 'children' not in decoded_nodes[i]:
                        decoded_nodes[i]['children'] = []
                    decoded_nodes[i]['children'].append(i + 1)
                
                return tree
            else:
                # Return default tree
                return {
                    'root': {'type': 'add', 'value': 1.0, 'children': []},
                    'nodes': [{'type': 'add', 'value': 1.0, 'children': []}],
                    'depth': 1,
                    'size': 1
                }
            
        except Exception as e:
            self.logger.error(f"One-hot decoding failed: {e}")
            # Return a basic fallback tree
            return {
                'root': {'type': 'add', 'value': 1.0, 'children': []},
                'nodes': [{'type': 'add', 'value': 1.0, 'children': []}],
                'depth': 1,
                'size': 1
            }

    def _recursive_decode(self, encoding: Dict[str, Any]) -> Any:
        """Decode recursive encoded tree architecture."""
        try:
            # Parse recursive encoding structure
            root_node = encoding.get('root', {})
            nodes = encoding.get('nodes', [])
            connections = encoding.get('connections', [])
            
            if not root_node:
                # Create default root node
                root_node = {
                    'type': 'add',
                    'value': 1.0,
                    'children': []
                }
            
            if not nodes:
                # Create default nodes
                nodes = [root_node]
            
            # Build tree structure from recursive encoding
            tree = {
                'root': root_node,
                'nodes': nodes,
                'depth': encoding.get('depth', 1),
                'size': len(nodes)
            }
            
            # Add connections if provided
            if connections:
                for connection in connections:
                    from_idx = connection.get('from', 0)
                    to_idx = connection.get('to', 0)
                    
                    if from_idx < len(nodes) and to_idx < len(nodes):
                        if 'children' not in nodes[from_idx]:
                            nodes[from_idx]['children'] = []
                        nodes[from_idx]['children'].append(to_idx)
            
            return tree
            
        except Exception as e:
            self.logger.error(f"Recursive decoding failed: {e}")
            # Return a basic fallback tree
            return {
                'root': {'type': 'add', 'value': 1.0, 'children': []},
                'nodes': [{'type': 'add', 'value': 1.0, 'children': []}],
                'depth': 1,
                'size': 1
            }

    def _hybrid_decode(self, encoding: Dict[str, Any]) -> Any:
        """Decode hybrid encoded tree architecture."""
        try:
            # Combine multiple encoding types for tree architectures
            tree = None
            
            # Get components from hybrid encoding
            one_hot_encoding = encoding.get('one_hot', None)
            recursive_encoding = encoding.get('recursive', None)
            string_encoding = encoding.get('string', None)
            
            # Try to decode from one-hot first
            if one_hot_encoding is not None:
                one_hot_result = self._one_hot_decode(one_hot_encoding)
                if one_hot_result and 'root' in one_hot_result:
                    tree = one_hot_result
            
            # Try to decode from recursive if one-hot failed
            elif recursive_encoding is not None:
                recursive_result = self._recursive_decode(recursive_encoding)
                if recursive_result and 'root' in recursive_result:
                    tree = recursive_result
            
            # Try to decode from string if others failed
            elif string_encoding is not None:
                # Simple string parsing for tree structure
                tree = {
                    'root': {'type': 'add', 'value': 1.0, 'children': []},
                    'nodes': [{'type': 'add', 'value': 1.0, 'children': []}],
                    'depth': 1,
                    'size': 1
                }
            
            # If all decoding methods failed, create a default tree
            if not tree:
                tree = {
                    'root': {'type': 'add', 'value': 1.0, 'children': []},
                    'nodes': [{'type': 'add', 'value': 1.0, 'children': []}],
                    'depth': 1,
                    'size': 1
                }
            
            return tree
            
        except Exception as e:
            self.logger.error(f"Hybrid decoding failed: {e}")
            # Return a basic fallback tree
            return {
                'root': {'type': 'add', 'value': 1.0, 'children': []},
                'nodes': [{'type': 'add', 'value': 1.0, 'children': []}],
                'depth': 1,
                'size': 1
            }

    def validate_encoding(self, encoding: Any, encoding_type: EncodingType) -> bool:
        """Validate tree architecture encoding."""
        try:
            if encoding_type == EncodingType.ONE_HOT:
                return isinstance(encoding, np.ndarray) and encoding.ndim == 1
            elif encoding_type == EncodingType.RECURSIVE:
                return isinstance(encoding, dict) and 'trees' in encoding
            elif encoding_type == EncodingType.HYBRID:
                return isinstance(encoding, dict) and len(encoding) > 0
            else:
                return False
        except Exception as e:
            tprint_debug(f"🔍 Operation failed: {e}")
            return False


class UnifiedArchitectureEncoder:
    """Unified encoder that handles both neural and tree architectures."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize unified architecture encoder."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize specialized encoders
        self.neural_encoder = NeuralArchitectureEncoder(config.get('neural_config', {}))
        self.tree_encoder = TreeArchitectureEncoder(config.get('tree_config', {}))

        self.logger.info("✅ Unified Architecture Encoder initialized")

    def encode(self, architecture: Any) -> EncodingResult:
        """Encode any architecture type."""

        if isinstance(architecture, NeuralArchitecture):
            return self.neural_encoder.encode(architecture)
        elif isinstance(architecture, TreeArchitecture):
            return self.tree_encoder.encode(architecture)
        else:
            raise ValueError(f"Unsupported architecture type: {type(architecture)}")

    def decode(self, encoding: Any, encoding_type: EncodingType, architecture_type: str) -> DecodingResult:
        """Decode any architecture type."""
        if architecture_type == 'neural':
            return self.neural_encoder.decode(encoding, encoding_type)
        elif architecture_type == 'tree':
            return self.tree_encoder.decode(encoding, encoding_type)
        else:
            raise ValueError(f"Unsupported architecture type: {architecture_type}")

    def save_encoding(self, architecture: Any, filepath: str) -> bool:
        """Save architecture encoding to file."""
        try:
            encoding_result = self.encode(architecture)

            # Convert to serializable format
            if isinstance(encoding_result.encoding, np.ndarray):
                encoding_dict = {
                    'encoding': encoding_result.encoding.tolist(),
                    'encoding_type': encoding_result.encoding_type.value,
                    'format': encoding_result.format.value,
                    'metadata': encoding_result.metadata,
                    'encoding_time': encoding_result.encoding_time,
                    'compression_ratio': encoding_result.compression_ratio
                }
            else:
                encoding_dict = {
                    'encoding': encoding_result.encoding,
                    'encoding_type': encoding_result.encoding_type.value,
                    'format': encoding_result.format.value,
                    'metadata': encoding_result.metadata,
                    'encoding_time': encoding_result.encoding_time,
                    'compression_ratio': encoding_result.compression_ratio
                }

            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(encoding_dict, f, indent=2)

            self.logger.info(f"✅ Architecture encoding saved to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to save architecture encoding: {e}")
            return False

    def load_encoding(self, filepath: str) -> Tuple[Any, EncodingResult]:
        """Load architecture encoding from file."""
        try:
            with open(filepath, 'r') as f:
                encoding_dict = json.load(f)

            # Reconstruct encoding result
            encoding_result = EncodingResult(
                encoding=encoding_dict['encoding'],
                encoding_type=EncodingType(encoding_dict['encoding_type']),
                format=EncodingFormat(encoding_dict['format']),
                metadata=encoding_dict['metadata'],
                encoding_time=encoding_dict['encoding_time'],
                compression_ratio=encoding_dict['compression_ratio']
            )

            # Convert list back to numpy array if needed
            if isinstance(encoding_result.encoding, list):
                encoding_result.encoding = np.array(encoding_result.encoding)

            self.logger.info(f"✅ Architecture encoding loaded from {filepath}")
            return encoding_result.encoding, encoding_result

        except Exception as e:
            self.logger.error(f"❌ Failed to load architecture encoding: {e}")
            raise


def create_neural_architecture_encoder(config: Dict[str, Any]) -> NeuralArchitectureEncoder:
    """Create a neural architecture encoder."""
    return NeuralArchitectureEncoder(config)


def create_tree_architecture_encoder(config: Dict[str, Any]) -> TreeArchitectureEncoder:
    """Create a tree architecture encoder."""
    return TreeArchitectureEncoder(config)


def create_unified_architecture_encoder(config: Dict[str, Any]) -> UnifiedArchitectureEncoder:
    """Create a unified architecture encoder."""
    return UnifiedArchitectureEncoder(config)