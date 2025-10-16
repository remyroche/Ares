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
        """Encode an architecture."""
        try:
            # Extract architecture features
            features = self._extract_architecture_features(architecture)

            # Create encoding based on architecture type
            if hasattr(architecture, 'architecture_type'):
                arch_type = architecture.architecture_type
            else:
                arch_type = 'unknown'

            # Generate encoding
            encoding = self._generate_encoding(features, arch_type)

            # Create encoding result
            result = EncodingResult(
                encoding=encoding,
                encoding_type=EncodingType.ARCHITECTURE,
                size=len(encoding),
                features=features,
                metadata={
                    'architecture_type': arch_type,
                    'encoding_method': self.__class__.__name__,
                    'timestamp': time.time()
                }
            )

            return result

        except Exception as e:
            tprint(f"⚠️ [ENCODER] Error encoding architecture: {e}", color="yellow")
            # Return empty encoding
            return EncodingResult(
                encoding=[],
                encoding_type=EncodingType.ARCHITECTURE,
                size=0,
                features={},
                metadata={'error': str(e)}
            )

    def decode(self, encoding: Any, encoding_type: EncodingType) -> DecodingResult:
        """Decode an architecture."""
        try:
            # Validate encoding
            if not self.validate_encoding(encoding, encoding_type):
                return DecodingResult(
                    architecture=None,
                    success=False,
                    error="Invalid encoding"
                )

            # Decode based on encoding type
            if encoding_type == EncodingType.ARCHITECTURE:
                architecture = self._decode_architecture(encoding)
            elif encoding_type == EncodingType.FEATURES:
                architecture = self._decode_features(encoding)
            else:
                return DecodingResult(
                    architecture=None,
                    success=False,
                    error=f"Unsupported encoding type: {encoding_type}"
                )

            # Create decoding result
            result = DecodingResult(
                architecture=architecture,
                success=True,
                error=None,
                metadata={
                    'encoding_type': encoding_type,
                    'decoding_method': self.__class__.__name__,
                    'timestamp': time.time()
                }
            )

            return result

        except Exception as e:
            tprint(f"⚠️ [ENCODER] Error decoding architecture: {e}", color="yellow")
            return DecodingResult(
                architecture=None,
                success=False,
                error=str(e)
            )

    def get_encoding_size(self, architecture: Any) -> int:
        """Get the size of the encoded representation."""
        try:
            # Extract architecture features
            features = self._extract_architecture_features(architecture)

            # Calculate encoding size based on features
            size = 0

            # Add size for each feature
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    size += 1
                elif isinstance(value, str):
                    size += len(value)
                elif isinstance(value, list):
                    size += len(value)
                elif isinstance(value, dict):
                    size += len(value)

            return max(1, size)  # Minimum size of 1

        except Exception as e:
            tprint(f"⚠️ [ENCODER] Error getting encoding size: {e}", color="yellow")
            return 1  # Default size

    def validate_encoding(self, encoding: Any, encoding_type: EncodingType) -> bool:
        """Validate an encoding."""
        try:
            # Check if encoding is not None
            if encoding is None:
                return False

            # Check if encoding is a list or array
            if not isinstance(encoding, (list, tuple, np.ndarray)):
                return False

            # Check if encoding has minimum size
            if len(encoding) == 0:
                return False

            # Check if encoding contains valid values
            for value in encoding:
                if not isinstance(value, (int, float, str)):
                    return False
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    return False

            return True

        except Exception as e:
            tprint(f"⚠️ [ENCODER] Error validating encoding: {e}", color="yellow")
            return False

    def _extract_architecture_features(self, architecture: Any) -> Dict[str, Any]:
        """Extract features from an architecture."""
        try:
            features = {}

            # Get basic architecture properties
            if hasattr(architecture, 'depth'):
                features['depth'] = architecture.depth
            elif hasattr(architecture, 'layers'):
                features['depth'] = len(architecture.layers)
            else:
                features['depth'] = 1

            if hasattr(architecture, 'width'):
                features['width'] = architecture.width
            elif hasattr(architecture, 'hidden_size'):
                features['width'] = architecture.hidden_size
            else:
                features['width'] = 64

            if hasattr(architecture, 'activation'):
                features['activation'] = str(architecture.activation)
            else:
                features['activation'] = 'relu'

            if hasattr(architecture, 'optimizer'):
                features['optimizer'] = str(architecture.optimizer)
            else:
                features['optimizer'] = 'adam'

            if hasattr(architecture, 'learning_rate'):
                features['learning_rate'] = architecture.learning_rate
            else:
                features['learning_rate'] = 0.001

            if hasattr(architecture, 'batch_size'):
                features['batch_size'] = architecture.batch_size
            else:
                features['batch_size'] = 32

            if hasattr(architecture, 'dropout'):
                features['dropout'] = architecture.dropout
            else:
                features['dropout'] = 0.0

            if hasattr(architecture, 'regularization'):
                features['regularization'] = architecture.regularization
            else:
                features['regularization'] = 0.0

            if hasattr(architecture, 'architecture_type'):
                features['architecture_type'] = str(architecture.architecture_type)
            else:
                features['architecture_type'] = 'unknown'

            # Get parameter count if available
            if hasattr(architecture, 'parameters'):
                features['num_parameters'] = sum(p.numel() for p in architecture.parameters())
            else:
                features['num_parameters'] = 0

            return features

        except Exception as e:
            tprint(f"⚠️ [ENCODER] Error extracting features: {e}", color="yellow")
            return {
                'depth': 1,
                'width': 64,
                'activation': 'relu',
                'optimizer': 'adam',
                'learning_rate': 0.001,
                'batch_size': 32,
                'dropout': 0.0,
                'regularization': 0.0,
                'architecture_type': 'unknown',
                'num_parameters': 0
            }

    def _generate_encoding(self, features: Dict[str, Any], arch_type: str) -> List[float]:
        """Generate encoding from features."""
        try:
            encoding = []

            # Add numerical features
            encoding.append(features.get('depth', 1))
            encoding.append(features.get('width', 64))
            encoding.append(features.get('learning_rate', 0.001))
            encoding.append(features.get('batch_size', 32))
            encoding.append(features.get('dropout', 0.0))
            encoding.append(features.get('regularization', 0.0))
            encoding.append(features.get('num_parameters', 0))

            # Add categorical features as one-hot encoded
            activation = features.get('activation', 'relu')
            if activation == 'relu':
                encoding.extend([1, 0, 0])
            elif activation == 'sigmoid':
                encoding.extend([0, 1, 0])
            else:
                encoding.extend([0, 0, 1])

            optimizer = features.get('optimizer', 'adam')
            if optimizer == 'adam':
                encoding.extend([1, 0, 0])
            elif optimizer == 'sgd':
                encoding.extend([0, 1, 0])
            else:
                encoding.extend([0, 0, 1])

            # Add architecture type encoding
            if arch_type == 'neural':
                encoding.extend([1, 0, 0])
            elif arch_type == 'tree':
                encoding.extend([0, 1, 0])
            else:
                encoding.extend([0, 0, 1])

            return encoding

        except Exception as e:
            tprint(f"⚠️ [ENCODER] Error generating encoding: {e}", color="yellow")
            return [1, 64, 0.001, 32, 0.0, 0.0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]

    def _decode_architecture(self, encoding: List[float]) -> Dict[str, Any]:
        """Decode architecture from encoding."""
        try:
            if len(encoding) < 16:
                # Not enough values in encoding
                return self._create_default_architecture()

            # Decode numerical features
            architecture = {
                'depth': int(encoding[0]),
                'width': int(encoding[1]),
                'learning_rate': float(encoding[2]),
                'batch_size': int(encoding[3]),
                'dropout': float(encoding[4]),
                'regularization': float(encoding[5]),
                'num_parameters': int(encoding[6])
            }

            # Decode categorical features
            activation_encoding = encoding[7:10]
            if activation_encoding[0] > 0.5:
                architecture['activation'] = 'relu'
            elif activation_encoding[1] > 0.5:
                architecture['activation'] = 'sigmoid'
            else:
                architecture['activation'] = 'tanh'

            optimizer_encoding = encoding[10:13]
            if optimizer_encoding[0] > 0.5:
                architecture['optimizer'] = 'adam'
            elif optimizer_encoding[1] > 0.5:
                architecture['optimizer'] = 'sgd'
            else:
                architecture['optimizer'] = 'rmsprop'

            arch_type_encoding = encoding[13:16]
            if arch_type_encoding[0] > 0.5:
                architecture['architecture_type'] = 'neural'
            elif arch_type_encoding[1] > 0.5:
                architecture['architecture_type'] = 'tree'
            else:
                architecture['architecture_type'] = 'hybrid'

            return architecture

        except Exception as e:
            tprint(f"⚠️ [ENCODER] Error decoding architecture: {e}", color="yellow")
            return self._create_default_architecture()

    def _decode_features(self, encoding: List[float]) -> Dict[str, Any]:
        """Decode features from encoding."""
        try:
            # This is a simplified implementation
            # In practice, this would decode specific features
            return self._decode_architecture(encoding)

        except Exception as e:
            tprint(f"⚠️ [ENCODER] Error decoding features: {e}", color="yellow")
            return self._create_default_architecture()

    def _create_default_architecture(self) -> Dict[str, Any]:
        """Create a default architecture."""
        return {
            'depth': 1,
            'width': 64,
            'activation': 'relu',
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'batch_size': 32,
            'dropout': 0.0,
            'regularization': 0.0,
            'architecture_type': 'neural',
            'num_parameters': 0
        }

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
        # Simple one-hot decoding implementation
        if len(encoding) == 0:
            return {'layers': []}

        # Find the index with maximum value (assuming one-hot encoding)
        max_idx = np.argmax(encoding)

        # Create a simple architecture based on the index
        architecture = {
            'layers': [{
                'width': 32 + max_idx * 32,  # Simple width calculation
                'activation': 'relu',
                'dropout': 0.0
            }]
        }

        return architecture

    def _adjacency_matrix_decode(self, encoding: Dict[str, Any]) -> Any:
        """Decode adjacency matrix encoded neural architecture."""
        # Simple adjacency matrix decoding implementation
        if 'adjacency_matrix' not in encoding:
            return {'layers': []}

        matrix = encoding['adjacency_matrix']
        if len(matrix) == 0:
            return {'layers': []}

        # Create layers based on adjacency matrix
        layers = []
        for i in range(len(matrix)):
            layer = {
                'width': 32 + i * 16,  # Simple width calculation
                'activation': 'relu',
                'dropout': 0.0
            }
            layers.append(layer)

        return {'layers': layers}

    def _path_decode(self, encoding: str) -> Any:
        """Decode path encoded neural architecture."""
        # Simple path decoding implementation
        if not encoding:
            return {'layers': []}

        # Parse path encoding (assuming format like "32-64-128")
        try:
            widths = [int(x) for x in encoding.split('-')]
        except ValueError:
            widths = [32, 64]  # Default fallback

        # Create layers based on path
        layers = []
        for width in widths:
            layer = {
                'width': width,
                'activation': 'relu',
                'dropout': 0.0
            }
            layers.append(layer)

        return {'layers': layers}

    def _hybrid_decode(self, encoding: Dict[str, Any]) -> Any:
        """Decode hybrid encoded neural architecture."""
        # Simple hybrid decoding implementation
        if not encoding:
            return {'layers': []}

        # Try to extract information from hybrid encoding
        layers = []

        # Check for layer information
        if 'layers' in encoding:
            layers = encoding['layers']
        elif 'widths' in encoding:
            # Create layers from widths
            widths = encoding['widths']
            for width in widths:
                layer = {
                    'width': width,
                    'activation': 'relu',
                    'dropout': 0.0
                }
                layers.append(layer)
        else:
            # Default fallback
            layers = [{
                'width': 64,
                'activation': 'relu',
                'dropout': 0.0
            }]

        return {'layers': layers}

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
        except:
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
        # Simple one-hot decoding implementation for tree architecture
        if len(encoding) == 0:
            return {'trees': []}

        # Find the index with maximum value (assuming one-hot encoding)
        max_idx = np.argmax(encoding)

        # Create a simple tree architecture based on the index
        tree_architecture = {
            'trees': [{
                'depth': 2 + max_idx % 3,  # Simple depth calculation
                'nodes': 2 ** (2 + max_idx % 3),
                'tree_type': 'decision_tree'
            }]
        }

        return tree_architecture

    def _recursive_decode(self, encoding: Dict[str, Any]) -> Any:
        """Decode recursive encoded tree architecture."""
        # Simple recursive decoding implementation
        if 'trees' not in encoding:
            return {'trees': []}

        trees = encoding['trees']
        if not trees:
            return {'trees': []}

        # Create tree architecture from recursive encoding
        tree_architecture = {
            'trees': []
        }

        for tree_info in trees:
            tree = {
                'depth': tree_info.get('depth', 3),
                'nodes': tree_info.get('nodes', 8),
                'tree_type': tree_info.get('tree_type', 'decision_tree')
            }
            tree_architecture['trees'].append(tree)

        return tree_architecture

    def _hybrid_decode(self, encoding: Dict[str, Any]) -> Any:
        """Decode hybrid encoded tree architecture."""
        # Simple hybrid decoding implementation for tree architecture
        if not encoding:
            return {'trees': []}

        # Try to extract tree information from hybrid encoding
        trees = []

        # Check for tree information
        if 'trees' in encoding:
            trees = encoding['trees']
        elif 'depths' in encoding:
            # Create trees from depths
            depths = encoding['depths']
            for depth in depths:
                tree = {
                    'depth': depth,
                    'nodes': 2 ** depth,
                    'tree_type': 'decision_tree'
                }
                trees.append(tree)
        else:
            # Default fallback
            trees = [{
                'depth': 3,
                'nodes': 8,
                'tree_type': 'decision_tree'
            }]

        return {'trees': trees}

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
        except:
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
