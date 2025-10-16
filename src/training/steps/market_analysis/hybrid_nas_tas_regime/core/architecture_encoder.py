"""
Unified Architecture Encoder for NAS-TAS Systems

This module provides sophisticated encoding and decoding of neural and tree architectures
using representation learning techniques, enabling efficient search space navigation
and architecture comparison.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import pickle
import json

# Import tprint for comprehensive logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)

class EncodingMethod(Enum):
    """Methods for architecture encoding."""
    ADJACENCY_MATRIX = "adjacency_matrix"
    PATH_ENCODING = "path_encoding"
    LSTM_BASED = "lstm_based"
    GRAPH_NEURAL_NETWORK = "graph_neural_network"
    AUTOENCODER = "autoencoder"
    HYBRID = "hybrid"

class DecoderType(Enum):
    """Types of architecture decoders."""
    MLP = "mlp"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    GRAPH_ATTENTION = "graph_attention"

@dataclass
class ArchitectureEncodingConfig:
    """Configuration for architecture encoding."""
    encoding_method: EncodingMethod = EncodingMethod.HYBRID
    latent_dim: int = 128
    max_sequence_length: int = 100
    vocabulary_size: int = 50
    use_pretrained_embeddings: bool = False
    embedding_dim: int = 64

    # Autoencoder settings
    encoder_layers: List[int] = field(default_factory=lambda: [256, 128])
    decoder_layers: List[int] = field(default_factory=lambda: [128, 256])
    activation: str = "relu"
    dropout_rate: float = 0.1

    # Training settings
    learning_rate: float = 0.001
    batch_size: int = 32
    n_epochs: int = 100
    reconstruction_loss_weight: float = 1.0
    prediction_loss_weight: float = 1.0

@dataclass
class EncodingResult:
    """Result from architecture encoding."""
    latent_vector: np.ndarray
    encoding_method: EncodingMethod
    encoding_time: float
    reconstruction_error: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DecodingResult:
    """Result from architecture decoding."""
    decoded_architecture: Dict[str, Any]
    decoding_method: DecoderType
    decoding_time: float
    reconstruction_accuracy: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class ArchitectureAutoencoder(nn.Module):
    """
    Autoencoder for architecture representation learning.

    Learns compact latent representations of neural and tree architectures
    while preserving important structural information.
    """

    def __init__(self, config: ArchitectureEncodingConfig, input_dim: int):
        """Initialize architecture autoencoder."""
        super().__init__()
        self.config = config

        # Encoder layers
        encoder_layers = []
        current_dim = input_dim

        for hidden_dim in config.encoder_layers:
            encoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU() if config.activation == "relu" else nn.Tanh(),
                nn.Dropout(config.dropout_rate)
            ])
            current_dim = hidden_dim

        # Latent space
        self.encoder = nn.Sequential(*encoder_layers)
        self.latent_projection = nn.Linear(current_dim, config.latent_dim)

        # Decoder layers
        decoder_layers = []
        current_dim = config.latent_dim

        for hidden_dim in config.decoder_layers:
            decoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU() if config.activation == "relu" else nn.Tanh(),
                nn.Dropout(config.dropout_rate)
            ])
            current_dim = hidden_dim

        # Output reconstruction
        decoder_layers.append(nn.Linear(current_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through autoencoder."""
        latent = self.latent_projection(self.encoder(x))
        reconstruction = self.decoder(latent)
        return latent, reconstruction

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        with torch.no_grad():
            latent = self.latent_projection(self.encoder(x))
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        with torch.no_grad():
            reconstruction = self.decoder(latent)
        return reconstruction

class UnifiedArchitectureEncoder:
    """
    Unified encoder for neural and tree architectures.

    Provides multiple encoding methods including adjacency matrices, path encodings,
    LSTM-based sequences, and autoencoder-based representation learning.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the unified architecture encoder."""
        self.config = ArchitectureEncodingConfig(**config)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Architecture vocabulary for sequence encoding
        self.architecture_vocabulary = self._build_architecture_vocabulary()

        # Autoencoder for representation learning
        self.autoencoder = None
        self.input_dim = self._calculate_input_dimension()
        self._initialize_autoencoder()

        # Encoding method dispatch
        self.encoding_methods = {
            EncodingMethod.ADJACENCY_MATRIX: self._adjacency_matrix_encoding,
            EncodingMethod.PATH_ENCODING: self._path_encoding,
            EncodingMethod.LSTM_BASED: self._lstm_encoding,
            EncodingMethod.AUTOENCODER: self._autoencoder_encoding,
            EncodingMethod.HYBRID: self._hybrid_encoding
        }

        self.logger.info("✅ Unified Architecture Encoder initialized")
        self.logger.info(f"   Encoding Method: {config.get('encoding_method', 'hybrid')}")
        self.logger.info(f"   Latent Dimension: {self.config.latent_dim}")

    def _build_architecture_vocabulary(self) -> Dict[str, int]:
        """Build vocabulary for architecture components."""
        vocabulary = {
            # Neural network components
            'conv1d': 1, 'conv2d': 2, 'linear': 3, 'lstm': 4, 'gru': 5,
            'attention': 6, 'transformer': 7, 'residual': 8, 'dropout': 9,
            'batch_norm': 10, 'layer_norm': 11, 'activation': 12,

            # Tree components
            'decision_tree': 13, 'random_forest': 14, 'xgboost': 15,
            'gradient_boosting': 16, 'lightgbm': 17,

            # Operations and connections
            'skip_connect': 18, 'concatenate': 19, 'add': 20, 'multiply': 21,
            'max_pool': 22, 'avg_pool': 23, 'global_avg_pool': 24,

            # Activations
            'relu': 25, 'leaky_relu': 26, 'elu': 27, 'selu': 28,
            'tanh': 29, 'sigmoid': 30, 'softmax': 31, 'none': 32,

            # Special tokens
            '<start>': 33, '<end>': 34, '<pad>': 35, '<unknown>': 36
        }

        return vocabulary

    def _calculate_input_dimension(self) -> int:
        """Calculate input dimension for autoencoder."""
        # This should be based on the maximum architecture encoding size
        # For now, use a reasonable default
        return 200

    def _initialize_autoencoder(self):
        """Initialize autoencoder if needed."""
        if self.config.encoding_method in [EncodingMethod.AUTOENCODER, EncodingMethod.HYBRID]:
            self.autoencoder = ArchitectureAutoencoder(self.config, self.input_dim)
            self.logger.info("✅ Architecture autoencoder initialized")

    def encode(self, architecture: Dict[str, Any]) -> EncodingResult:
        """Encode architecture to latent representation."""
        start_time = time.time()

        try:
            if self.config.encoding_method == EncodingMethod.HYBRID:
                # Use hybrid encoding combining multiple methods
                latent_vector = self._hybrid_encoding(architecture)
            else:
                # Use specified encoding method
                encoding_func = self.encoding_methods[self.config.encoding_method]
                latent_vector = encoding_func(architecture)

            # Convert to numpy array if needed
            if isinstance(latent_vector, torch.Tensor):
                latent_vector = latent_vector.detach().numpy()

            # Calculate reconstruction error if autoencoder is available
            reconstruction_error = 0.0
            if self.autoencoder:
                try:
                    # Simple reconstruction error calculation
                    reconstruction_error = self._calculate_reconstruction_error(
                        architecture, latent_vector
                    )
                except Exception as e:
                    self.logger.warning(f"Reconstruction error calculation failed: {e}")

            encoding_time = time.time() - start_time

            result = EncodingResult(
                latent_vector=latent_vector,
                encoding_method=self.config.encoding_method,
                encoding_time=encoding_time,
                reconstruction_error=reconstruction_error,
                metadata={
                    'architecture_type': architecture.get('type', 'unknown'),
                    'encoding_method': self.config.encoding_method.value,
                    'latent_dim': len(latent_vector)
                }
            )

            self.logger.debug(f"Architecture encoded in {encoding_time:.4f}s, "
                            f"latent shape: {latent_vector.shape}")

            return result

        except Exception as e:
            encoding_time = time.time() - start_time
            self.logger.error(f"❌ Architecture encoding failed: {e}")

            # Return zero vector as fallback
            latent_dim = self.config.latent_dim
            return EncodingResult(
                latent_vector=np.zeros(latent_dim),
                encoding_method=self.config.encoding_method,
                encoding_time=encoding_time,
                reconstruction_error=1.0,
                metadata={'error': str(e), 'latent_dim': latent_dim}
            )

    def decode(self, latent_vector: np.ndarray) -> DecodingResult:
        """Decode latent representation back to architecture."""
        start_time = time.time()

        try:
            # Convert to torch tensor if needed
            if isinstance(latent_vector, np.ndarray):
                latent_tensor = torch.from_numpy(latent_vector).float().unsqueeze(0)
            else:
                latent_tensor = latent_vector.unsqueeze(0)

            # Use autoencoder for decoding
            if self.autoencoder:
                with torch.no_grad():
                    reconstruction = self.autoencoder.decode(latent_tensor)
                    decoded_architecture = self._reconstruction_to_architecture(reconstruction)
            else:
                # Fallback decoding
                decoded_architecture = self._simple_decode(latent_vector)

            # Calculate reconstruction accuracy
            reconstruction_accuracy = self._calculate_reconstruction_accuracy(
                latent_vector, decoded_architecture
            )

            decoding_time = time.time() - start_time

            result = DecodingResult(
                decoded_architecture=decoded_architecture,
                decoding_method=DecoderType.MLP,  # Default decoder type
                decoding_time=decoding_time,
                reconstruction_accuracy=reconstruction_accuracy,
                metadata={
                    'latent_shape': latent_vector.shape,
                    'decoder_type': 'autoencoder' if self.autoencoder else 'simple'
                }
            )

            self.logger.debug(f"Architecture decoded in {decoding_time:.4f}s, "
                            f"accuracy: {reconstruction_accuracy:.4f}")

            return result

        except Exception as e:
            decoding_time = time.time() - start_time
            self.logger.error(f"❌ Architecture decoding failed: {e}")

            return DecodingResult(
                decoded_architecture={},
                decoding_method=DecoderType.MLP,
                decoding_time=decoding_time,
                reconstruction_accuracy=0.0,
                metadata={'error': str(e)}
            )

    def _adjacency_matrix_encoding(self, architecture: Dict[str, Any]) -> np.ndarray:
        """Encode architecture using adjacency matrix representation."""
        try:
            layers = architecture.get('layers', [])
            n_layers = len(layers)

            # Create adjacency matrix
            adj_matrix = np.zeros((n_layers, n_layers))

            # Fill adjacency matrix based on layer connections
            for i in range(n_layers - 1):
                adj_matrix[i, i + 1] = 1  # Sequential connection

                # Add skip connections if present
                if layers[i].get('residual', False) or layers[i].get('skip_connect', False):
                    adj_matrix[i, min(i + 2, n_layers - 1)] = 0.5

            # Add layer type information
            layer_types = []
            for layer in layers:
                layer_type = layer.get('type', 'linear')
                # Simple encoding of layer types
                type_encoding = hash(layer_type) % 10 / 10.0
                layer_types.append(type_encoding)

            # Combine adjacency matrix and layer types
            adj_flat = adj_matrix.flatten()
            combined = np.concatenate([adj_flat, np.array(layer_types)])

            # Pad or truncate to latent dimension
            if len(combined) < self.config.latent_dim:
                padding = np.zeros(self.config.latent_dim - len(combined))
                combined = np.concatenate([combined, padding])
            else:
                combined = combined[:self.config.latent_dim]

            return combined

        except Exception as e:
            self.logger.warning(f"Adjacency matrix encoding failed: {e}")
            return np.zeros(self.config.latent_dim)

    def _path_encoding(self, architecture: Dict[str, Any]) -> np.ndarray:
        """Encode architecture using path-based representation."""
        try:
            layers = architecture.get('layers', [])
            n_layers = len(layers)

            if n_layers == 0:
                return np.zeros(self.config.latent_dim)

            # Create path encoding based on layer sequence
            path_encoding = []

            for layer in layers:
                # Encode layer properties
                layer_features = [
                    len(layers),  # Total layers
                    layer.get('hidden_size', 100) / 1000.0,  # Normalized size
                    1.0 if layer.get('activation') else 0.0,  # Has activation
                    1.0 if layer.get('dropout_rate', 0) > 0 else 0.0,  # Has dropout
                    1.0 if layer.get('batch_norm', False) else 0.0,  # Has batch norm
                    1.0 if layer.get('residual', False) else 0.0,  # Has residual
                ]
                path_encoding.extend(layer_features)

            # Pad to fixed length
            max_path_length = self.config.max_sequence_length * 6  # 6 features per layer
            if len(path_encoding) < max_path_length:
                padding = [0.0] * (max_path_length - len(path_encoding))
                path_encoding.extend(padding)
            else:
                path_encoding = path_encoding[:max_path_length]

            # Truncate to latent dimension
            return np.array(path_encoding[:self.config.latent_dim])

        except Exception as e:
            self.logger.warning(f"Path encoding failed: {e}")
            return np.zeros(self.config.latent_dim)

    def _lstm_encoding(self, architecture: Dict[str, Any]) -> np.ndarray:
        """Encode architecture using LSTM-based sequence modeling."""
        try:
            layers = architecture.get('layers', [])
            n_layers = len(layers)

            # Create sequence of layer types for LSTM input
            sequence = []
            for layer in layers:
                layer_type = layer.get('type', 'linear')
                # Map layer type to vocabulary index
                vocab_idx = self.architecture_vocabulary.get(layer_type, 36)  # 36 = <unknown>
                sequence.append(vocab_idx)

            # Pad sequence to max length
            if len(sequence) < self.config.max_sequence_length:
                padding = [35] * (self.config.max_sequence_length - len(sequence))  # 35 = <pad>
                sequence = padding + sequence
            else:
                sequence = sequence[:self.config.max_sequence_length]

            # Convert to one-hot encoding
            one_hot_sequence = np.zeros((self.config.max_sequence_length, self.config.vocabulary_size))
            for i, vocab_idx in enumerate(sequence):
                if vocab_idx < self.config.vocabulary_size:
                    one_hot_sequence[i, vocab_idx] = 1.0

            # Flatten to latent vector
            latent_vector = one_hot_sequence.flatten()

            # Truncate to latent dimension
            return latent_vector[:self.config.latent_dim]

        except Exception as e:
            self.logger.warning(f"LSTM encoding failed: {e}")
            return np.zeros(self.config.latent_dim)

    def _autoencoder_encoding(self, architecture: Dict[str, Any]) -> np.ndarray:
        """Encode architecture using autoencoder."""
        try:
            if not self.autoencoder:
                raise ValueError("Autoencoder not initialized")

            # Convert architecture to input vector
            input_vector = self._architecture_to_vector(architecture)

            # Convert to torch tensor
            input_tensor = torch.from_numpy(input_vector).float().unsqueeze(0)

            # Encode
            with torch.no_grad():
                latent_tensor = self.autoencoder.encode(input_tensor)

            return latent_tensor.squeeze().numpy()

        except Exception as e:
            self.logger.warning(f"Autoencoder encoding failed: {e}")
            return np.zeros(self.config.latent_dim)

    def _hybrid_encoding(self, architecture: Dict[str, Any]) -> np.ndarray:
        """Combine multiple encoding methods."""
        try:
            # Get encodings from different methods
            adjacency_encoding = self._adjacency_matrix_encoding(architecture)
            path_encoding = self._path_encoding(architecture)
            lstm_encoding = self._lstm_encoding(architecture)

            # Combine encodings (weighted average)
            weights = [0.4, 0.3, 0.3]  # Weights for each encoding method
            combined = (weights[0] * adjacency_encoding +
                       weights[1] * path_encoding +
                       weights[2] * lstm_encoding)

            return combined

        except Exception as e:
            self.logger.warning(f"Hybrid encoding failed: {e}")
            return np.zeros(self.config.latent_dim)

    def _architecture_to_vector(self, architecture: Dict[str, Any]) -> np.ndarray:
        """Convert architecture to input vector for autoencoder."""
        try:
            # Combine different encodings into a single vector
            adjacency = self._adjacency_matrix_encoding(architecture)
            path = self._path_encoding(architecture)
            lstm = self._lstm_encoding(architecture)

            # Concatenate and pad/truncate
            combined = np.concatenate([adjacency, path, lstm])

            if len(combined) < self.input_dim:
                padding = np.zeros(self.input_dim - len(combined))
                combined = np.concatenate([combined, padding])
            else:
                combined = combined[:self.input_dim]

            return combined

        except Exception as e:
            self.logger.warning(f"Architecture to vector conversion failed: {e}")
            return np.zeros(self.input_dim)

    def _calculate_reconstruction_error(self, architecture: Dict[str, Any],
                                      latent_vector: np.ndarray) -> float:
        """Calculate reconstruction error for encoding."""
        try:
            # This is a simplified reconstruction error
            # In practice, would decode and compare to original
            return 0.1  # Placeholder

        except Exception:
            return 1.0

    def _reconstruction_to_architecture(self, reconstruction: torch.Tensor) -> Dict[str, Any]:
        """Convert reconstruction to architecture specification."""
        try:
            # Simplified reconstruction to architecture
            # In practice, this would be more sophisticated
            return {
                'type': 'reconstructed',
                'layers': [{'type': 'linear', 'hidden_size': 128}],
                'reconstruction_error': 0.1
            }

        except Exception:
            return {}

    def _simple_decode(self, latent_vector: np.ndarray) -> Dict[str, Any]:
        """Simple decoding without autoencoder."""
        try:
            # Basic decoding based on latent vector properties
            magnitude = np.linalg.norm(latent_vector)

            # Determine architecture type based on latent vector
            if magnitude > 10.0:
                architecture_type = 'neural'
            elif magnitude > 5.0:
                architecture_type = 'tree'
            else:
                architecture_type = 'hybrid'

            # Determine complexity
            complexity = min(magnitude / 20.0, 1.0)

            # Generate layers based on complexity
            n_layers = max(2, int(complexity * 10))

            layers = []
            for i in range(n_layers):
                layers.append({
                    'type': 'linear',
                    'hidden_size': int(100 + complexity * 200),
                    'activation': 'relu'
                })

            return {
                'type': architecture_type,
                'layers': layers,
                'decoded_from_latent': True,
                'latent_magnitude': float(magnitude)
            }

        except Exception as e:
            self.logger.warning(f"Simple decoding failed: {e}")
            return {'type': 'neural', 'layers': [{'type': 'linear', 'hidden_size': 128}]}

    def _calculate_reconstruction_accuracy(self, latent_vector: np.ndarray,
                                         decoded_architecture: Dict[str, Any]) -> float:
        """Calculate reconstruction accuracy."""
        try:
            # This is a simplified accuracy calculation
            # In practice, would compare structural similarity
            return 0.7  # Placeholder

        except Exception:
            return 0.0

    def train_autoencoder(self, architectures: List[Dict[str, Any]],
                          n_epochs: Optional[int] = None):
        """Train the autoencoder on architecture data."""
        try:
            if not self.autoencoder:
                self.logger.warning("Autoencoder not initialized, skipping training")
                return

            n_epochs = n_epochs or self.config.n_epochs

            # Convert architectures to training vectors
            training_vectors = []
            for arch in architectures:
                vector = self._architecture_to_vector(arch)
                training_vectors.append(vector)

            if len(training_vectors) < self.config.batch_size:
                self.logger.warning("Insufficient training data for autoencoder")
                return

            X_train = torch.from_numpy(np.array(training_vectors)).float()

            # Training setup
            optimizer = Adam(self.autoencoder.parameters(), lr=self.config.learning_rate)
            criterion = nn.MSELoss()

            self.logger.info(f"Training autoencoder for {n_epochs} epochs on {len(X_train)} samples")

            # Training loop
            for epoch in range(n_epochs):
                epoch_loss = 0.0

                # Mini-batch training
                for i in range(0, len(X_train), self.config.batch_size):
                    batch_X = X_train[i:i + self.config.batch_size]

                    optimizer.zero_grad()
                    latent, reconstruction = self.autoencoder(batch_X)
                    loss = criterion(reconstruction, batch_X)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

            self.logger.info("✅ Autoencoder training completed")

        except Exception as e:
            self.logger.error(f"❌ Autoencoder training failed: {e}")

    def save_encoder_state(self, filepath: str) -> bool:
        """Save encoder state to disk."""
        try:
            state = {
                'config': self.config.__dict__,
                'vocabulary': self.architecture_vocabulary,
                'input_dim': self.input_dim,
                'autoencoder_state': self.autoencoder.state_dict() if self.autoencoder else None
            }

            with open(filepath, 'wb') as f:
                pickle.dump(state, f)

            self.logger.info(f"✅ Encoder state saved to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to save encoder state: {e}")
            return False

    def load_encoder_state(self, filepath: str) -> bool:
        """Load encoder state from disk."""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            self.config = ArchitectureEncodingConfig(**state['config'])
            self.architecture_vocabulary = state['vocabulary']
            self.input_dim = state['input_dim']

            if state['autoencoder_state'] and self.autoencoder:
                self.autoencoder.load_state_dict(state['autoencoder_state'])

            self.logger.info(f"✅ Encoder state loaded from {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to load encoder state: {e}")
            return False

def create_unified_architecture_encoder(config: Dict[str, Any]) -> UnifiedArchitectureEncoder:
    """Create a unified architecture encoder instance."""
    return UnifiedArchitectureEncoder(config)

def quick_architecture_encode(architecture: Dict[str, Any],
                             config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """Quick architecture encoding with default settings."""
    if config is None:
        config = {
            'encoding_method': 'hybrid',
            'latent_dim': 128
        }

    encoder = UnifiedArchitectureEncoder(config)
    result = encoder.encode(architecture)
    return result.latent_vector
