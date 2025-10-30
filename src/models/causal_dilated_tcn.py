"""
Causal Dilated TCN Model

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
import os  # <-- Added for path checking
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from src.utils.logging_config import setup_logging  # <-- Added for logging

# Try to import PyTorch - gracefully handle if not available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    # --- New Imports (conditional on PyTorch) ---
    from src.analyst.autoencoder_feature_generator import TimeSeriesAutoencoder, load_autoencoder_config
    # --- End New Imports ---
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    # Create dummy nn module for class definitions
    class nn:
        class Module:
            pass
        class Conv1d:
            pass
        class Dropout:
            pass
        class ReLU:
            pass
        class Sequential:
            pass
        class MSELoss:
            pass
    # --- Add dummy classes for AE if torch fails ---
    class TimeSeriesAutoencoder(nn.Module):
        pass
    def load_autoencoder_config(path):
        return None
    # --- End dummy classes ---

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
setup_logging()
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

    # --- Autoencoder Integration ---
    autoencoder_model_path: str = "models/autoencoder.pth"
    autoencoder_config_path: str = "models/autoencoder_config.json"
    use_autoencoder: bool = True  # Flag to enable/disable autoencoder

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
    This model uses dilated convolutions to capture long-range dependencies
    in time series data while maintaining causality (no future information leakage).
    """

    def __init__(self, config: Optional[CausalTCNConfig] = None):
        """Initialize the Causal Dilated TCN model."""
        self.config = config or CausalTCNConfig()

        # Components
        self.tcn_model = None
        self.scaler = None
        self.autoencoder = None  # <-- Autoencoder model
        self.latent_dim = None  # <-- AE latent dimension

        # State
        self.fitted = False
        self.feature_names = None

        # --- NEW: Load Frozen Autoencoder ---
        if PYTORCH_AVAILABLE and self.config.use_autoencoder:
            try:
                logger.info("Attempting to load frozen autoencoder...")
                ae_config_path = self.config.autoencoder_config_path
                ae_model_path = self.config.autoencoder_model_path

                if not (os.path.exists(ae_config_path) and os.path.exists(ae_model_path)):
                    logger.warning(f"Autoencoder model/config not found at {ae_config_path}/{ae_model_path}. Disabling AE.")
                    self.config.use_autoencoder = False
                else:
                    # 1. Load AE Config
                    ae_config = load_autoencoder_config(ae_config_path)
                    if 'latent_dim' not in ae_config or 'input_dim' not in ae_config:
                        logger.warning("AE config missing 'latent_dim' or 'input_dim'. Disabling AE.")
                        self.config.use_autoencoder = False
                    else:
                        self.latent_dim = ae_config['latent_dim']
                        
                        # 2. Initialize AE Model
                        # Assumes TimeSeriesAutoencoder init matches the config dict keys
                        self.autoencoder = TimeSeriesAutoencoder(**ae_config)
                        
                        # 3. Load Saved Weights
                        self.autoencoder.load_state_dict(torch.load(ae_model_path))
                        
                        # 4. Freeze Weights and set to eval mode
                        self.autoencoder.eval()
                        for param in self.autoencoder.parameters():
                            param.requires_grad = False
                        
                        logger.info(f"✅ Successfully loaded and froze autoencoder. Latent dim: {self.latent_dim}")
            except Exception as e:
                logger.error(f"❌ Failed to load autoencoder: {e}. Disabling AE.")
                self.config.use_autoencoder = False
                self.autoencoder = None
        # --- End Autoencoder Load ---
    

    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for TCN input.
        
        Args:
            X: Feature array (n_samples, n_features) - can be engineered features
            y: Target array (n_samples,) - actual prediction targets
            sequence_length: Length of temporal sequences to create
            
        Returns:
            X_seq: Sequence array (n_sequences, sequence_length, n_features)
            y_seq: Target array (n_sequences,)
        """
        try:
            # Validate inputs
            if len(X) != len(y):
                raise ValueError(f"X and y must have same length: X={len(X)}, y={len(y)}")
            
            # Check for NaN values and clean them
            if np.any(np.isnan(X)):
                logger.warning(f"⚠️ Found {np.sum(np.isnan(X))} NaN values in X, filling with 0")
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            if np.any(np.isnan(y)):
                logger.warning(f"⚠️ Found {np.sum(np.isnan(y))} NaN values in y, filling with 0")
                y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            
            sequences = []
            targets = []

            # Create sliding window sequences
            for i in range(sequence_length, len(X)):
                sequence = X[i-sequence_length:i]  # Shape: (sequence_length, n_features)
                target = y[i]  # Shape: (,) - scalar target
                sequences.append(sequence)
                targets.append(target)

            if not sequences:
                # If no sequences can be created, create a single sequence with padding
                if len(X) < sequence_length:
                    # Pad the sequence with zeros at the beginning
                    padded_X = np.zeros((sequence_length, X.shape[1]))
                    padded_X[-len(X):] = X
                    sequences = [padded_X]
                    targets = [y[-1]]  # Use last available target
                else:
                    # Use the last available sequence
                    sequences = [X[-sequence_length:]]
                    targets = [y[-1]]

            # Convert to numpy arrays with correct shapes
            sequences_array = np.array(sequences)  # Shape: (n_sequences, sequence_length, n_features)
            targets_array = np.array(targets)  # Shape: (n_sequences,)
            
            logger.info(f"✅ Created {len(sequences_array)} sequences with shape {sequences_array.shape}")
            return sequences_array, targets_array

        except Exception as e:
            logger.warning(f"⚠️ Sequence preparation failed: {e}")
            # Fallback: create single sequence with proper error handling
            if len(X) < sequence_length:
                padded_X = np.zeros((sequence_length, X.shape[1]))
                padded_X[-len(X):] = X
                return padded_X.reshape(1, sequence_length, -1), np.array([y[-1]])
            return X[-sequence_length:].reshape(1, sequence_length, -1), np.array([y[-1]])

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'CausalDilatedTCNModel':
        """Fit the Causal Dilated TCN model.
        
        Args:
            X: Feature array (n_samples, n_features) - accepts engineered features
            y: Target array (n_samples,) - regression targets
            sample_weight: Optional sample weights (not currently used)
            
        Returns:
            self: Fitted model
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset

            # Store feature names if available
            if hasattr(X, 'columns'):
                self.feature_names = list(X.columns)
                X = X.values
            
            # Convert y to numpy if needed
            if hasattr(y, 'values'):
                y = y.values
            
            # Clean NaN values before scaling
            if np.any(np.isnan(X)):
                logger.warning(f"⚠️ Found {np.sum(np.isnan(X))} NaN values in X before scaling, filling with 0")
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            if np.any(np.isnan(y)):
                logger.warning(f"⚠️ Found {np.sum(np.isnan(y))} NaN values in y, filling with 0")
                y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            
            logger.info(f"📊 TCN input: X shape={X.shape}, y shape={y.shape}")

            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Verify no NaN after scaling
            if np.any(np.isnan(X_scaled)):
                logger.error("❌ NaN values found after scaling, this should not happen")
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            # Prepare sequences with targets
            sequence_length = min(50, len(X) // 4)  # Adaptive sequence length
            if sequence_length < 10:
                sequence_length = min(10, len(X) - 1)  # Ensure minimum sequence length
            
            logger.info(f"📊 Creating sequences with length={sequence_length}")
            X_seq, y_seq = self._prepare_sequences(X_scaled, y, sequence_length)

            # Verify sequence shapes
            logger.info(f"✅ Sequence shapes - X_seq: {X_seq.shape}, y_seq: {y_seq.shape}")
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_seq)
            y_tensor = torch.FloatTensor(y_seq)
            
                # --- NEW: Apply Autoencoder ---
                if self.autoencoder is not None:
                    logger.info(f"Compressing sequences with autoencoder...")
                    with torch.no_grad():
                        # AE expects [N, S, F], which X_tensor already is
                        X_tensor = self.autoencoder.encoder(X_tensor)
                    logger.info(f"✅ Compressed sequences shape: {X_tensor.shape}")
                # --- End Apply Autoencoder ---
                
                logger.info(f"✅ Tensor shapes - X_tensor: {X_tensor.shape}, y_tensor: {y_tensor.shape}")

                # Create TCN model with correct input size (number of features or latent_dim)
                # X_seq shape is (n_sequences, sequence_length, n_features)
                # X_tensor shape is (n_sequences, sequence_length, n_features_or_latent_dim)
                actual_n_features = X_tensor.shape[2]
                logger.info(f"📊 Creating TCN with input_size={actual_n_features} features")
                
                self.tcn_model = CausalDilatedTCN(
                    input_size=actual_n_features,
                    num_filters=self.config.num_filters,
                    
    def _fit_fallback(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'CausalDilatedTCNModel':
        """Fallback to simple linear model."""
        try:
            from sklearn.linear_model import Ridge

            # Clean NaN values
            if np.any(np.isnan(X)):
                logger.warning(f"⚠️ Cleaning {np.sum(np.isnan(X))} NaN values in fallback X")
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            if np.any(np.isnan(y)):
                logger.warning(f"⚠️ Cleaning {np.sum(np.isnan(y))} NaN values in fallback y")
                y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Verify no NaN after scaling
            if np.any(np.isnan(X_scaled)):
                logger.error("❌ NaN values found after scaling in fallback")
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            # Use Ridge regression as fallback (more stable than LinearRegression)
            self.tcn_model = Ridge(alpha=1.0)
            self.tcn_model.fit(X_scaled, y, sample_weight)
            
            # Store input size
            self.input_size = X.shape[1]

            self.fitted = True
            logger.info("✅ Fallback Ridge regression model fitted")

            return self

        except Exception as e:
            logger.error(f"❌ Fallback model fitting failed: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted model.
        
        Args:
            X: Feature array (n_samples, n_features) - same features as training
            
        Returns:
            predictions: Predicted values (n_samples,)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        try:
            # Convert to numpy if pandas DataFrame
            if hasattr(X, 'values'):
                X = X.values
            
            # Clean NaN values
            if np.any(np.isnan(X)):
                logger.warning(f"⚠️ Found {np.sum(np.isnan(X))} NaN values in prediction X, filling with 0")
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Verify no NaN after scaling
            if np.any(np.isnan(X_scaled)):
                logger.warning("⚠️ NaN values found after scaling in predict, cleaning")
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            # Check if model is PyTorch model
            if hasattr(self.tcn_model, 'forward'):
                import torch

                # Prepare sequences (for prediction, we use dummy targets)
                sequence_length = min(50, len(X_scaled) // 4)
                if sequence_length < 10:
                    sequence_length = min(10, len(X_scaled) - 1)
                
                # Create dummy targets for sequence preparation
                dummy_targets = np.zeros(len(X_scaled))
                X_seq, _ = self._prepare_sequences(X_scaled, dummy_targets, sequence_length)

                # Convert to tensor
                X_tensor = torch.FloatTensor(X_seq)
                
                if self.autoencoder is not None:
                    logger.info(f"Compressing prediction sequences with autoencoder...")
                    with torch.no_grad():
                        # AE expects [N, S, F]
                        X_tensor = self.autoencoder.encoder(X_tensor)
                    logger.info(f"✅ Compressed prediction sequences shape: {X_tensor.shape}")
                        
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
