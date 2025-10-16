"""
Causal Dilated TCN Model for Tactician

This module implements a Causal Dilated Temporal Convolutional Network (TCN)
for the tactician models. TCNs are particularly effective for time series
prediction tasks due to their ability to capture long-range dependencies
with dilated convolutions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class CausalTCNConfig:
    """Configuration for Causal Dilated TCN model."""
    num_filters: int = 64
    kernel_size: int = 3
    dilation_base: int = 2
    num_layers: int = 4
    dropout: float = 0.1
    activation: str = "relu"
    use_skip_connections: bool = True
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    random_state: int = 42


class CausalDilatedConv1d(nn.Module):
    """Causal dilated 1D convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int, dropout: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Calculate padding for causal convolution
        self.padding = (kernel_size - 1) * dilation
        
        # Convolution layer
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )
        
        # Normalization
        self.norm = nn.BatchNorm1d(out_channels)
        
        # Activation
        self.activation = nn.ReLU()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through causal dilated convolution."""
        # Apply convolution
        out = self.conv(x)
        
        # Remove future information (causal padding)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        
        # Apply normalization, activation, and dropout
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        return out


class ResidualBlock(nn.Module):
    """Residual block with causal dilated convolution."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float = 0.0):
        super().__init__()
        
        # First causal dilated convolution
        self.conv1 = CausalDilatedConv1d(
            in_channels, out_channels, kernel_size, dilation, dropout
        )
        
        # Second causal dilated convolution
        self.conv2 = CausalDilatedConv1d(
            out_channels, out_channels, kernel_size, dilation, dropout
        )
        
        # Skip connection (1x1 convolution if dimensions don't match)
        if in_channels != out_channels:
            self.skip_conv = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.skip_conv = None
        
        # Final activation
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block."""
        # First convolution
        out = self.conv1(x)
        
        # Second convolution
        out = self.conv2(out)
        
        # Skip connection
        if self.skip_conv is not None:
            skip = self.skip_conv(x)
        else:
            skip = x
        
        # Add skip connection
        out = out + skip
        
        # Final activation
        out = self.activation(out)
        
        return out


class CausalDilatedTCN(nn.Module):
    """Causal Dilated Temporal Convolutional Network."""
    
    def __init__(self, input_size: int, num_filters: int = 64, kernel_size: int = 3,
                 dilation_base: int = 2, num_layers: int = 4, dropout: float = 0.1,
                 use_skip_connections: bool = True):
        super().__init__()
        
        self.input_size = input_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dilation_base = dilation_base
        self.num_layers = num_layers
        self.use_skip_connections = use_skip_connections
        
        # Input projection
        self.input_projection = nn.Conv1d(input_size, num_filters, 1)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = dilation_base ** i
            self.residual_blocks.append(
                ResidualBlock(
                    in_channels=num_filters,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output projection
        self.output_projection = nn.Linear(num_filters, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through TCN."""
        # x shape: (batch_size, seq_len, input_size)
        # Transpose for Conv1d: (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_projection(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Global average pooling
        x = self.global_pool(x)  # (batch_size, num_filters, 1)
        x = x.squeeze(-1)  # (batch_size, num_filters)
        
        # Output projection
        x = self.output_projection(x)  # (batch_size, 1)
        
        return x


class CausalDilatedTCNModel(BaseEstimator, RegressorMixin):
    """
    Causal Dilated TCN Model for Tactician.
    
    This model uses dilated convolutions to capture long-range dependencies
    in time series data while maintaining causality (no future information leakage).
    """
    
    def __init__(self, config: Optional[CausalTCNConfig] = None):
        """Initialize the Causal Dilated TCN model."""
        self.config = config or CausalTCNConfig()
        
        # Components
        self.tcn_model = None
        self.scaler = None
        
        # State
        self.fitted = False
        self.feature_names = None
        
    def _prepare_sequences(self, X: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for TCN input."""
        try:
            sequences = []
            targets = []
            
            for i in range(sequence_length, len(X)):
                sequence = X[i-sequence_length:i]
                target = X[i]  # Use the next timestep as target
                sequences.append(sequence)
                targets.append(target)
            
            if not sequences:
                # If no sequences can be created, create a single sequence
                if len(X) < sequence_length:
                    # Pad the sequence
                    padded_X = np.zeros((sequence_length, X.shape[1]))
                    padded_X[-len(X):] = X
                    sequences = [padded_X]
                    targets = [X[-1]]  # Use last available target
                else:
                    sequences = [X[-sequence_length:]]
                    targets = [X[-1]]
            
            return np.array(sequences), np.array(targets)
            
        except Exception as e:
            logger.warning(f"⚠️ Sequence preparation failed: {e}")
            # Fallback: create single sequence
            if len(X) < sequence_length:
                padded_X = np.zeros((sequence_length, X.shape[1]))
                padded_X[-len(X):] = X
                return padded_X.reshape(1, sequence_length, -1), X[-1:].reshape(1, -1)
            return X[-sequence_length:].reshape(1, sequence_length, -1), X[-1:].reshape(1, -1)
    
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'CausalDilatedTCNModel':
        """Fit the Causal Dilated TCN model."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            
            # Store feature names if available
            if hasattr(X, 'columns'):
                self.feature_names = list(X.columns)
                X = X.values
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Prepare sequences
            sequence_length = min(50, len(X) // 4)  # Adaptive sequence length
            X_seq, y_seq = self._prepare_sequences(X_scaled, sequence_length)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_seq)
            y_tensor = torch.FloatTensor(y_seq)
            
            # Create TCN model
            self.tcn_model = CausalDilatedTCN(
                input_size=X.shape[1],
                num_filters=self.config.num_filters,
                kernel_size=self.config.kernel_size,
                dilation_base=self.config.dilation_base,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
                use_skip_connections=self.config.use_skip_connections
            )
            
            # Training setup
            optimizer = optim.Adam(self.tcn_model.parameters(), lr=self.config.learning_rate)
            criterion = nn.MSELoss()
            
            # Data loader
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(
                dataset, 
                batch_size=self.config.batch_size, 
                shuffle=True
            )
            
            # Training loop
            self.tcn_model.train()
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.config.epochs):
                epoch_loss = 0.0
                
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    predictions = self.tcn_model(batch_X)
                    loss = criterion(predictions.squeeze(), batch_y.squeeze())
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(dataloader)
                
                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
            
            self.fitted = True
            logger.info(f"✅ Causal Dilated TCN model fitted with {X.shape[1]} features")
            
            return self
            
        except ImportError:
            logger.warning("⚠️ PyTorch not available, using fallback linear model")
            return self._fit_fallback(X, y, sample_weight)
        except Exception as e:
            logger.error(f"❌ Causal Dilated TCN model fitting failed: {e}")
            return self._fit_fallback(X, y, sample_weight)
    
    def _fit_fallback(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'CausalDilatedTCNModel':
        """Fallback to simple linear model."""
        try:
            from sklearn.linear_model import LinearRegression
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Simple linear model as fallback
            self.tcn_model = LinearRegression()
            self.tcn_model.fit(X_scaled, y, sample_weight)
            
            self.fitted = True
            logger.info("✅ Fallback linear model fitted")
            
            return self
            
        except Exception as e:
            logger.error(f"❌ Fallback model fitting failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted model."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Convert to numpy if pandas DataFrame
            if hasattr(X, 'values'):
                X = X.values
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Check if model is PyTorch model
            if hasattr(self.tcn_model, 'forward'):
                import torch
                
                # Prepare sequences
                sequence_length = min(50, len(X_scaled) // 4)
                X_seq, _ = self._prepare_sequences(X_scaled, sequence_length)
                
                # Convert to tensor
                X_tensor = torch.FloatTensor(X_seq)
                
                # Predict
                self.tcn_model.eval()
                with torch.no_grad():
                    predictions = self.tcn_model(X_tensor)
                    predictions = predictions.squeeze().numpy()
                
                # Ensure we have the right number of predictions
                if len(predictions) < X.shape[0]:
                    # Pad with the last prediction
                    padding = np.full(X.shape[0] - len(predictions), predictions[-1])
                    predictions = np.concatenate([predictions, padding])
                elif len(predictions) > X.shape[0]:
                    # Truncate to match
                    predictions = predictions[:X.shape[0]]
                
                return predictions
            else:
                # Fallback model
                return self.tcn_model.predict(X_scaled)
                
        except Exception as e:
            logger.error(f"❌ Causal Dilated TCN model prediction failed: {e}")
            raise
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance (not directly available for TCN)."""
        if not self.fitted:
            return np.array([])
        
        try:
            if hasattr(self.tcn_model, 'coef_'):
                return np.abs(self.tcn_model.coef_)
            else:
                # For TCN, we can't easily extract feature importance
                # Return uniform importance as placeholder
                return np.ones(self.input_size) / self.input_size
        except Exception as e:
            logger.warning(f"⚠️ Could not get feature importance: {e}")
            return np.array([])


# Factory function
def create_causal_dilated_tcn(config: Optional[CausalTCNConfig] = None) -> CausalDilatedTCNModel:
    """Create Causal Dilated TCN model."""
    return CausalDilatedTCNModel(config)