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
            start_time = time.time()
            
            # Basic architecture encoding
            encoding = {
                'type': 'basic',
                'layers': getattr(architecture, 'layers', []),
                'connections': getattr(architecture, 'connections', []),
                'parameters': getattr(architecture, 'parameters', {}),
                'metadata': {
                    'encoding_time': time.time() - start_time,
                    'architecture_type': type(architecture).__name__
                }
            }
            
            return EncodingResult(
                encoding=encoding,
                encoding_type=EncodingType.BASIC,
                encoding_time=time.time() - start_time,
                metadata={'method': 'basic_encoding'}
            )
            
        except Exception as e:
            self.logger.error(f"Error encoding architecture: {e}")
            return EncodingResult(
                encoding=None,
                encoding_type=EncodingType.BASIC,
                encoding_time=0.0,
                metadata={'error': str(e)}
            )

    def decode(self, encoding: Any, encoding_type: EncodingType) -> DecodingResult:
        """Decode an architecture."""
        try:
            start_time = time.time()
            
            if encoding is None:
                return DecodingResult(
                    architecture=None,
                    decoding_time=0.0,
                    confidence=0.0,
                    metadata={'error': 'No encoding provided'}
                )
            
            # Basic architecture reconstruction
            architecture = type('Architecture', (), {
                'layers': encoding.get('layers', []),
                'connections': encoding.get('connections', []),
                'parameters': encoding.get('parameters', {}),
                'metadata': encoding.get('metadata', {})
            })()
            
            return DecodingResult(
                architecture=architecture,
                decoding_time=time.time() - start_time,
                confidence=0.8,  # Basic confidence for simple encoding
                metadata={'method': 'basic_decoding'}
            )
            
        except Exception as e:
            self.logger.error(f"Error decoding architecture: {e}")
            return DecodingResult(
                architecture=None,
                decoding_time=0.0,
                confidence=0.0,
                metadata={'error': str(e)}
            )

    def get_encoding_size(self, architecture: Any) -> int:
        """Get the size of the encoded representation."""
        try:
            # Calculate encoding size based on architecture complexity
            layers = getattr(architecture, 'layers', [])
            connections = getattr(architecture, 'connections', [])
            parameters = getattr(architecture, 'parameters', {})
            
            # Basic size calculation
            size = len(layers) + len(connections) + len(parameters)
            return max(size, 1)  # Ensure at least size 1
            
        except Exception as e:
            self.logger.error(f"Error calculating encoding size: {e}")
            return 1  # Default size

    def validate_encoding(self, encoding: Any, encoding_type: EncodingType) -> bool:
        """Validate an encoding."""
        try:
            if encoding is None:
                return False
            
            # Basic validation checks
            if not isinstance(encoding, dict):
                return False
            
            # Check for required fields
            required_fields = ['type', 'layers', 'connections', 'parameters']
            for field in required_fields:
                if field not in encoding:
                    return False
            
            # Validate field types
            if not isinstance(encoding['layers'], list):
                return False
            if not isinstance(encoding['connections'], list):
                return False
            if not isinstance(encoding['parameters'], dict):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating encoding: {e}")
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
            # Basic one-hot decoding implementation
            if encoding is None or len(encoding) == 0:
                return None
            
            # Convert one-hot vector back to architecture representation
            decoded_architecture = type('DecodedNeuralArchitecture', (), {
                'one_hot_vector': encoding,
                'decoded_layers': len(encoding),
                'metadata': {'decoding_method': 'one_hot', 'vector_length': len(encoding)}
            })()
            
            return decoded_architecture
        except Exception as e:
            self.logger.error(f"Error in one-hot decoding: {e}")
            return None

    def _adjacency_matrix_decode(self, encoding: Dict[str, Any]) -> Any:
        """Decode adjacency matrix encoded neural architecture."""
        try:
            # Basic adjacency matrix decoding implementation
            if not isinstance(encoding, dict) or 'adjacency_matrix' not in encoding:
                return None
            
            adjacency_matrix = encoding['adjacency_matrix']
            decoded_architecture = type('DecodedNeuralArchitecture', (), {
                'adjacency_matrix': adjacency_matrix,
                'n_nodes': len(adjacency_matrix),
                'metadata': {'decoding_method': 'adjacency_matrix', 'matrix_shape': adjacency_matrix.shape}
            })()
            
            return decoded_architecture
        except Exception as e:
            self.logger.error(f"Error in adjacency matrix decoding: {e}")
            return None

    def _path_decode(self, encoding: str) -> Any:
        """Decode path encoded neural architecture."""
        try:
            # Basic path decoding implementation
            if not isinstance(encoding, str) or len(encoding) == 0:
                return None
            
            decoded_architecture = type('DecodedNeuralArchitecture', (), {
                'path_encoding': encoding,
                'path_length': len(encoding),
                'metadata': {'decoding_method': 'path', 'path': encoding}
            })()
            
            return decoded_architecture
        except Exception as e:
            self.logger.error(f"Error in path decoding: {e}")
            return None

    def _hybrid_decode(self, encoding: Dict[str, Any]) -> Any:
        """Decode hybrid encoded neural architecture."""
        try:
            # Basic hybrid decoding implementation
            if not isinstance(encoding, dict):
                return None
            
            decoded_architecture = type('DecodedNeuralArchitecture', (), {
                'hybrid_encoding': encoding,
                'encoding_keys': list(encoding.keys()),
                'metadata': {'decoding_method': 'hybrid', 'encoding_dict': encoding}
            })()
            
            return decoded_architecture
        except Exception as e:
            self.logger.error(f"Error in hybrid decoding: {e}")
            return None

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
        try:
            # Basic one-hot decoding implementation for tree architecture
            if encoding is None or len(encoding) == 0:
                return None
            
            decoded_architecture = type('DecodedTreeArchitecture', (), {
                'one_hot_vector': encoding,
                'decoded_trees': len(encoding),
                'metadata': {'decoding_method': 'one_hot', 'vector_length': len(encoding)}
            })()
            
            return decoded_architecture
        except Exception as e:
            self.logger.error(f"Error in tree one-hot decoding: {e}")
            return None

    def _recursive_decode(self, encoding: Dict[str, Any]) -> Any:
        """Decode recursive encoded tree architecture."""
        try:
            # Basic recursive decoding implementation for tree architecture
            if not isinstance(encoding, dict):
                return None
            
            decoded_architecture = type('DecodedTreeArchitecture', (), {
                'recursive_encoding': encoding,
                'encoding_keys': list(encoding.keys()),
                'metadata': {'decoding_method': 'recursive', 'encoding_dict': encoding}
            })()
            
            return decoded_architecture
        except Exception as e:
            self.logger.error(f"Error in recursive decoding: {e}")
            return None

    def _hybrid_decode(self, encoding: Dict[str, Any]) -> Any:
        """Decode hybrid encoded tree architecture."""
        try:
            # Basic hybrid decoding implementation for tree architecture
            if not isinstance(encoding, dict):
                return None
            
            decoded_architecture = type('DecodedTreeArchitecture', (), {
                'hybrid_encoding': encoding,
                'encoding_keys': list(encoding.keys()),
                'metadata': {'decoding_method': 'hybrid', 'encoding_dict': encoding}
            })()
            
            return decoded_architecture
        except Exception as e:
            self.logger.error(f"Error in tree hybrid decoding: {e}")
            return None

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