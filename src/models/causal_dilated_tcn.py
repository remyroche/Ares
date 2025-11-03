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
    import json
    import pickle
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

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# GPU/MPS device detection for Apple Silicon
def get_torch_device():
    """Get the best available PyTorch device (MPS for Apple Silicon, CUDA, or CPU)."""
    if not PYTORCH_AVAILABLE:
        return None
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


class PyTorchAutoencoder(nn.Module):
    """
    PyTorch-based Autoencoder for feature compression.
    This is a lightweight encoder that compresses high-dimensional features into a latent space.
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder: input_dim -> hidden_dim -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.Tanh()  # Tanh activation for bounded latent space
        )
        
        # Decoder: latent_dim -> hidden_dim -> input_dim (for pre-training only)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full autoencoder forward pass (for pre-training)."""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        return self.encoder(x)
    
    def save_encoder(self, path: str):
        """Save only the encoder part."""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim
        }, path)
        logger.info(f"✅ Encoder saved to {path}")
    
    @staticmethod
    def load_encoder(path: str) -> 'PyTorchAutoencoder':
        """Load a pre-trained encoder."""
        checkpoint = torch.load(path, map_location='cpu')
        model = PyTorchAutoencoder(
            input_dim=checkpoint['input_dim'],
            latent_dim=checkpoint['latent_dim']
        )
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        model.eval()
        # Freeze encoder weights
        for param in model.encoder.parameters():
            param.requires_grad = False
        logger.info(f"✅ Encoder loaded from {path} and frozen")
        return model


@dataclass
class CausalTCNConfig:
    """Configuration for Causal Dilated TCN model."""
    num_filters: int = 32
    kernel_size: int = 3
    dilation_base: int = 2
    num_layers: int = 5
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
    use_autoencoder: bool = False  # Flag to enable/disable autoencoder compression
    autoencoder_path: str = "models/analyst_autoencoder_encoder.pth"  # Path to frozen encoder
    latent_dim: int = 16  # Latent dimension for compressed features
    # If use_autoencoder=True and encoder doesn't exist, train one
    train_autoencoder_if_missing: bool = True
    autoencoder_epochs: int = 50  # Epochs for autoencoder pre-training

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
        self.frozen_encoder = None  # Frozen autoencoder for feature compression
        self.input_size = None  # Will be set during fit

        # State
        self.fitted = False
        self.feature_names = None
        
        # Device detection
        self.device = get_torch_device()
        if self.device:
            logger.info(f"🚀 TCN using device: {self.device}")

        # --- Load Frozen Autoencoder if enabled ---
        if PYTORCH_AVAILABLE and self.config.use_autoencoder:
            try:
                logger.info("🔧 Autoencoder compression enabled for TCN")
                if os.path.exists(self.config.autoencoder_path):
                    logger.info(f"📂 Loading pre-trained frozen encoder from: {self.config.autoencoder_path}")
                    self.frozen_encoder = PyTorchAutoencoder.load_encoder(self.config.autoencoder_path)
                    logger.info(f"✅ Frozen encoder loaded successfully! Latent dim: {self.frozen_encoder.latent_dim}")
                else:
                    if self.config.train_autoencoder_if_missing:
                        logger.warning(f"⚠️ Encoder not found at {self.config.autoencoder_path}")
                        logger.info("💡 Autoencoder will be trained during fit() if use_autoencoder=True")
                        self.frozen_encoder = None
                    else:
                        logger.warning(f"❌ Encoder not found and train_autoencoder_if_missing=False. Disabling autoencoder.")
                        self.config.use_autoencoder = False
            except Exception as e:
                logger.error(f"❌ Failed to load autoencoder: {e}. Disabling AE compression.")
                import traceback
                traceback.print_exc()
                self.config.use_autoencoder = False
                self.frozen_encoder = None
        # --- End Autoencoder Load ---
    
    def _train_autoencoder(self, X_scaled: np.ndarray) -> None:
        """
        Train a new autoencoder on the provided data and freeze it.
        
        Args:
            X_scaled: Scaled feature array (n_samples, n_features)
        """
        if not PYTORCH_AVAILABLE:
            logger.error("❌ PyTorch not available, cannot train autoencoder")
            return
        
        try:
            logger.info("🏋️ Training new autoencoder for feature compression...")
            logger.info(f"📊 Input features: {X_scaled.shape[1]}, Target latent dim: {self.config.latent_dim}")
            
            # Create autoencoder and move to device
            autoencoder = PyTorchAutoencoder(
                input_dim=X_scaled.shape[1],
                latent_dim=self.config.latent_dim
            )
            if self.device:
                autoencoder = autoencoder.to(self.device)
            
            # Prepare training data and move to device
            X_tensor = torch.FloatTensor(X_scaled)
            if self.device:
                X_tensor = X_tensor.to(self.device)
            
            # Split into train/val
            n_train = int(0.8 * len(X_tensor))
            X_train = X_tensor[:n_train]
            X_val = X_tensor[n_train:]
            
            # Create dataloaders
            train_dataset = TensorDataset(X_train, X_train)  # Autoencoder reconstructs input
            val_dataset = TensorDataset(X_val, X_val)
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.Adam(autoencoder.parameters(), lr=self.config.learning_rate)
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.config.autoencoder_epochs):
                # Training
                autoencoder.train()
                train_loss = 0.0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    reconstructed = autoencoder(batch_X)
                    loss = criterion(reconstructed, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                
                # Validation
                autoencoder.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        reconstructed = autoencoder(batch_X)
                        loss = criterion(reconstructed, batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    os.makedirs(os.path.dirname(self.config.autoencoder_path), exist_ok=True)
                    autoencoder.save_encoder(self.config.autoencoder_path)
                else:
                    patience_counter += 1
                
                if epoch % 10 == 0:
                    logger.info(f"📈 Epoch {epoch}/{self.config.autoencoder_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"⏹️ Early stopping at epoch {epoch}")
                    break
            
            # Load the best frozen encoder
            self.frozen_encoder = PyTorchAutoencoder.load_encoder(self.config.autoencoder_path)
            logger.info(f"✅ Autoencoder training completed! Best val loss: {best_val_loss:.6f}")
            
        except Exception as e:
            logger.error(f"❌ Autoencoder training failed: {e}")
            import traceback
            traceback.print_exc()
            self.config.use_autoencoder = False
            self.frozen_encoder = None

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
            
            # Train autoencoder if needed
            if self.config.use_autoencoder and self.frozen_encoder is None:
                if self.config.train_autoencoder_if_missing:
                    logger.info("🏋️ Training autoencoder since none was found...")
                    self._train_autoencoder(X_scaled)
                else:
                    logger.warning("⚠️ Autoencoder disabled - encoder not found and training disabled")
                    self.config.use_autoencoder = False
            
            # Convert to tensors and move to device
            X_tensor = torch.FloatTensor(X_seq)
            y_tensor = torch.FloatTensor(y_seq)
            if self.device:
                X_tensor = X_tensor.to(self.device)
                y_tensor = y_tensor.to(self.device)
            
            # --- Apply Frozen Autoencoder Compression ---
            if self.config.use_autoencoder and self.frozen_encoder is not None:
                logger.info(f"🗜️ Compressing sequences with frozen autoencoder...")
                logger.info(f"   Original shape: {X_tensor.shape}")
                
                # Move frozen encoder to device if needed
                if self.device:
                    self.frozen_encoder = self.frozen_encoder.to(self.device)
                
                # Reshape for encoder: (n_sequences, sequence_length, n_features) -> (n_sequences * sequence_length, n_features)
                batch_size, seq_len, n_features = X_tensor.shape
                X_flat = X_tensor.reshape(-1, n_features)
                
                # Compress with frozen encoder
                self.frozen_encoder.eval()
                with torch.no_grad():
                    X_compressed = self.frozen_encoder.encode(X_flat)
                
                # Reshape back to sequences: (n_sequences * sequence_length, latent_dim) -> (n_sequences, sequence_length, latent_dim)
                X_tensor = X_compressed.reshape(batch_size, seq_len, -1)
                logger.info(f"   ✅ Compressed to: {X_tensor.shape} (latent_dim={X_tensor.shape[2]})")
            # --- End Autoencoder Compression ---
            
            logger.info(f"✅ Final tensor shapes - X_tensor: {X_tensor.shape}, y_tensor: {y_tensor.shape}")

            # Determine input size for TCN (number of features or latent_dim after compression)
            actual_n_features = X_tensor.shape[2]
            self.input_size = actual_n_features
            logger.info(f"📊 Creating TCN with input_size={actual_n_features} features")
            
            # Create TCN model and move to device
            self.tcn_model = CausalDilatedTCN(
                input_size=actual_n_features,
                num_filters=self.config.num_filters,
                kernel_size=self.config.kernel_size,
                dilation_base=self.config.dilation_base,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
                use_skip_connections=self.config.use_skip_connections
            )
            if self.device:
                self.tcn_model = self.tcn_model.to(self.device)
                logger.info(f"✅ TCN model moved to {self.device}")
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.tcn_model.parameters(), lr=self.config.learning_rate)
            
            # Create dataloader
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
            
            # Training loop (ONLY TCN weights are updated, encoder is frozen)
            logger.info(f"🏋️ Training TCN model for {self.config.epochs} epochs...")
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.config.epochs):
                self.tcn_model.train()
                epoch_loss = 0.0
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    predictions = self.tcn_model(batch_X)
                    loss = criterion(predictions.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                epoch_loss /= len(dataloader)
                
                # Early stopping check
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if epoch % 10 == 0:
                    logger.info(f"📈 Epoch {epoch}/{self.config.epochs} - Loss: {epoch_loss:.6f}")
                
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"⏹️ Early stopping at epoch {epoch}")
                    break
            
            logger.info(f"✅ TCN training completed! Best loss: {best_loss:.6f}")
            self.fitted = True
            return self
            
        except Exception as e:
            logger.error(f"❌ TCN model fitting failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to simple model
            logger.warning("⚠️ Falling back to Ridge regression")
            return self._fit_fallback(X, y, sample_weight)
                    
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

                # Convert to tensor and move to device
                X_tensor = torch.FloatTensor(X_seq)
                if self.device:
                    X_tensor = X_tensor.to(self.device)
                
                # --- Apply Frozen Autoencoder Compression (same as in training) ---
                if self.config.use_autoencoder and self.frozen_encoder is not None:
                    logger.info(f"🗜️ Compressing prediction sequences with frozen autoencoder...")
                    
                    # Move frozen encoder to device if needed
                    if self.device:
                        self.frozen_encoder = self.frozen_encoder.to(self.device)
                    
                    # Reshape for encoder
                    batch_size, seq_len, n_features = X_tensor.shape
                    X_flat = X_tensor.reshape(-1, n_features)
                    
                    # Compress with frozen encoder
                    self.frozen_encoder.eval()
                    with torch.no_grad():
                        X_compressed = self.frozen_encoder.encode(X_flat)
                    
                    # Reshape back to sequences
                    X_tensor = X_compressed.reshape(batch_size, seq_len, -1)
                    logger.info(f"   ✅ Compressed prediction sequences shape: {X_tensor.shape}")
                # --- End Autoencoder Compression ---
                        
                # Predict
                self.tcn_model.eval()
                with torch.no_grad():
                    predictions = self.tcn_model(X_tensor)
                    predictions = predictions.squeeze().cpu().numpy()  # Move to CPU before numpy conversion

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
