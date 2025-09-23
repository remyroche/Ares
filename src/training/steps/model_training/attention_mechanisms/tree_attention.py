"""
Base attention mechanism for tree-based models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import system_logger

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class AttentionType(Enum):
    """Types of attention mechanisms."""
    SELF_ATTENTION = "self_attention"
    CROSS_ATTENTION = "cross_attention"
    TEMPORAL_ATTENTION = "temporal_attention"
    FEATURE_ATTENTION = "feature_attention"
    MULTI_HEAD_ATTENTION = "multi_head_attention"


@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms."""
    
    # Attention type
    attention_type: AttentionType = AttentionType.SELF_ATTENTION
    
    # Model parameters
    attention_dim: int = 64
    num_heads: int = 8
    dropout_rate: float = 0.1
    learning_rate: float = 0.001
    regularization: float = 0.01
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    
    # Feature processing
    normalize_features: bool = True
    feature_selection: bool = True
    max_features: int = 1000
    
    # Model architecture
    hidden_layers: List[int] = None
    activation: str = 'relu'
    output_activation: str = 'linear'
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [128, 64, 32]


class TreeAttentionMechanism(ABC):
    """Base class for attention mechanisms in tree-based models."""
    
    def __init__(self, config: AttentionConfig):
        """Initialize attention mechanism."""
        self.config = config
        self.logger = system_logger.getChild(self.__class__.__name__)
        
        # Initialize attention model
        self.attention_model = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, tree_predictions: np.ndarray) -> 'TreeAttentionMechanism':
        """Fit attention mechanism to data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray, tree_predictions: np.ndarray) -> np.ndarray:
        """Apply attention mechanism to predictions."""
        pass
    
    @abstractmethod
    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """Get attention weights for input features."""
        pass
    
    def _preprocess_features(self, X: np.ndarray) -> np.ndarray:
        """Preprocess features for attention mechanism."""
        if self.config.normalize_features:
            from sklearn.preprocessing import StandardScaler
            if not hasattr(self, 'scaler'):
                self.scaler = StandardScaler()
                X = self.scaler.fit_transform(X)
            else:
                X = self.scaler.transform(X)
        
        if self.config.feature_selection and X.shape[1] > self.config.max_features:
            # Simple feature selection based on variance
            feature_vars = np.var(X, axis=0)
            top_features = np.argsort(feature_vars)[-self.config.max_features:]
            X = X[:, top_features]
            if not hasattr(self, 'selected_features'):
                self.selected_features = top_features
        
        return X
    
    def _create_attention_model(self, input_dim: int) -> Any:
        """Create attention model architecture."""
        if TORCH_AVAILABLE:
            return self._create_torch_attention_model(input_dim)
        elif TENSORFLOW_AVAILABLE:
            return self._create_tensorflow_attention_model(input_dim)
        else:
            raise ImportError("Neither PyTorch nor TensorFlow available for attention models")
    
    def _create_torch_attention_model(self, input_dim: int) -> nn.Module:
        """Create PyTorch attention model."""
        class TorchAttentionModel(nn.Module):
            def __init__(self, input_dim, config):
                super().__init__()
                self.config = config
                
                # Input projection
                self.input_projection = nn.Linear(input_dim, config.attention_dim)
                
                # Multi-head attention
                self.attention = nn.MultiheadAttention(
                    embed_dim=config.attention_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout_rate,
                    batch_first=True
                )
                
                # Hidden layers
                self.hidden_layers = nn.ModuleList()
                prev_dim = config.attention_dim
                for hidden_dim in config.hidden_layers:
                    self.hidden_layers.append(nn.Linear(prev_dim, hidden_dim))
                    self.hidden_layers.append(nn.ReLU())
                    self.hidden_layers.append(nn.Dropout(config.dropout_rate))
                    prev_dim = hidden_dim
                
                # Output layer
                self.output_layer = nn.Linear(prev_dim, 1)
                
            def forward(self, x):
                # Project input
                x = self.input_projection(x)
                
                # Self-attention
                attn_output, attn_weights = self.attention(x, x, x)
                x = attn_output
                
                # Hidden layers
                for layer in self.hidden_layers:
                    x = layer(x)
                
                # Output
                x = self.output_layer(x)
                return x, attn_weights
        
        return TorchAttentionModel(input_dim, self.config)
    
    def _create_tensorflow_attention_model(self, input_dim: int) -> Model:
        """Create TensorFlow attention model."""
        # Input layer
        inputs = tf.keras.Input(shape=(input_dim,))
        
        # Input projection
        x = layers.Dense(self.config.attention_dim, activation='relu')(inputs)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=self.config.num_heads,
            key_dim=self.config.attention_dim // self.config.num_heads,
            dropout=self.config.dropout_rate
        )(x, x)
        
        # Add & Norm
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization()(x)
        
        # Hidden layers
        for hidden_dim in self.config.hidden_layers:
            x = layers.Dense(hidden_dim, activation=self.config.activation)(x)
            x = layers.Dropout(self.config.dropout_rate)(x)
        
        # Output layer
        outputs = layers.Dense(1, activation=self.config.output_activation)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def _train_attention_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train attention model."""
        if self.attention_model is None:
            raise ValueError("Attention model not initialized")
        
        if TORCH_AVAILABLE and isinstance(self.attention_model, nn.Module):
            self._train_torch_model(X, y)
        elif TENSORFLOW_AVAILABLE and isinstance(self.attention_model, Model):
            self._train_tensorflow_model(X, y)
        else:
            raise ValueError("Unsupported attention model type")
    
    def _train_torch_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train PyTorch attention model."""
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1))
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Initialize optimizer and loss
        optimizer = optim.Adam(self.attention_model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        self.attention_model.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                predictions, attention_weights = self.attention_model(batch_X)
                loss = criterion(predictions, batch_y)
                
                # Add regularization
                if self.config.regularization > 0:
                    l2_reg = sum(p.pow(2.0).sum() for p in self.attention_model.parameters())
                    loss += self.config.regularization * l2_reg
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")
    
    def _train_tensorflow_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train TensorFlow attention model."""
        # Compile model
        self.attention_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = []
        if self.config.early_stopping_patience > 0:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        # Train model
        self.attention_model.fit(
            X, y,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
    
    def get_feature_importance(self, X: np.ndarray) -> np.ndarray:
        """Get feature importance from attention weights."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        attention_weights = self.get_attention_weights(X)
        
        # Calculate feature importance as mean attention weight
        feature_importance = np.mean(attention_weights, axis=0)
        
        return feature_importance
    
    def explain_prediction(self, X: np.ndarray, sample_idx: int = 0) -> Dict[str, Any]:
        """Explain prediction for a specific sample."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before explaining predictions")
        
        # Get attention weights for the sample
        attention_weights = self.get_attention_weights(X[sample_idx:sample_idx+1])
        
        # Get prediction
        prediction = self.predict(X[sample_idx:sample_idx+1], np.zeros((1, 1)))
        
        return {
            'prediction': prediction[0],
            'attention_weights': attention_weights[0],
            'feature_importance': self.get_feature_importance(X[sample_idx:sample_idx+1])[0],
            'sample_idx': sample_idx
        }