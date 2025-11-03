"""
Standalone GRU Embedding Generator for Tactician

This module provides a scikit-learn compatible transformer for generating
embeddings from sequential data using a simple GRU model.

It replaces the embedding logic previously integrated into LGBMGRUEmbedding.
"""

import numpy as np
import warnings
import logging
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# Suppress warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# GPU/MPS device detection for Apple Silicon
def get_torch_device():
    """Get the best available PyTorch device (MPS for Apple Silicon, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

@dataclass
class GRUGeneratorConfig:
    """Configuration for the GRU embedding generator."""
    lookback_hours: int = 3
    hidden_size: int = 48
    num_layers: int = 1
    dropout: float = 0.05

class SimpleGRU(nn.Module):
    """Simple GRU implementation for embedding generation."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through GRU."""
        output, hidden = self.gru(x)
        last_hidden = hidden[-1]  # Take the last layer's hidden state
        last_hidden = self.dropout(last_hidden)
        return last_hidden

class GRUEmbeddingGenerator(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that generates embeddings
    from sequential data using a SimpleGRU model.
    """

    def __init__(self, config: Optional[GRUGeneratorConfig] = None):
        """Initialize the generator."""
        self.config = config or GRUGeneratorConfig()
        self.scaler = None
        self.gru_model = None
        self.input_size_ = None
        self.lookback_bars_ = self.config.lookback_hours * 12
        self.device = get_torch_device()
        logger.info(f"🚀 GRU using device: {self.device}")

    def _prepare_sequences(self, X: np.ndarray) -> np.ndarray:
        """Prepare sequences for GRU input."""
        try:
            sequences = []
            for i in range(self.lookback_bars_, len(X)):
                sequence = X[i-self.lookback_bars_:i]
                sequences.append(sequence)

            if not sequences:
                padded_X = np.zeros((self.lookback_bars_, X.shape[1]))
                padded_X[-len(X):] = X
                return padded_X.reshape(1, self.lookback_bars_, -1)

            return np.array(sequences)
        except Exception as e:
            logger.warning(f"⚠️ Sequence preparation failed: {e}")
            if len(X) < self.lookback_bars_:
                padded_X = np.zeros((self.lookback_bars_, X.shape[1]))
                padded_X[-len(X):] = X
                return padded_X.reshape(1, self.lookback_bars_, -1)
            return X[-self.lookback_bars_:].reshape(1, self.lookback_bars_, -1)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'GRUEmbeddingGenerator':
        """
        Fit the generator:
        1. Fit the StandardScaler.
        2. Initialize the GRU model.
        """
        # 1. Fit scaler
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        
        # 2. Initialize GRU model
        self.input_size_ = X.shape[1]
        self.gru_model = SimpleGRU(
            input_size=self.input_size_,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        )
        self.gru_model.to(self.device)  # Move model to GPU/MPS if available
        self.gru_model.eval()  # Set to evaluation mode
        
        logger.info(f"✅ GRUEmbeddingGenerator fitted (Scaler + GRU initialized on {self.device})")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data into GRU embeddings.
        """
        if self.scaler is None or self.gru_model is None:
            raise ValueError("Generator must be fitted before transforming data.")
        
        try:
            # 1. Scale features
            X_scaled = self.scaler.transform(X)

            # 2. Prepare sequences
            sequences = self._prepare_sequences(X_scaled)
            
            if sequences.size == 0:
                return np.zeros((X.shape[0], self.config.hidden_size))

            # 3. Convert to tensor and move to device
            X_tensor = torch.FloatTensor(sequences).to(self.device)

            # 4. Generate embeddings
            with torch.no_grad():
                embeddings = self.gru_model(X_tensor)
                embeddings = embeddings.cpu().numpy()  # Move back to CPU for numpy conversion

            # 5. Ensure correct shape (pad missing initial rows)
            if len(embeddings) < X.shape[0]:
                # Pad with the first available embedding
                padding = np.tile(embeddings[0], (X.shape[0] - len(embeddings), 1))
                embeddings = np.vstack([padding, embeddings])
            elif len(embeddings) > X.shape[0]:
                # Truncate to match
                embeddings = embeddings[:X.shape[0]]
            
            return embeddings

        except Exception as e:
            logger.warning(f"⚠️ GRU embedding creation failed: {e}")
            return np.zeros((X.shape[0], self.config.hidden_size))
