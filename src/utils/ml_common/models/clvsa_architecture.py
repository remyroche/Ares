"""
CLVSA Architecture: Convolutional + LSTM + Attention + Variational

This module implements the CLVSA (Convolutional-LSTM-Variational-Attention) architecture
for advanced time series prediction with uncertainty quantification.

Key Components:
1. Convolutional Layers: Spatial feature extraction from multi-dimensional data
2. LSTM Layers: Temporal dependencies modeling with memory
3. Attention Mechanism: Focus on relevant time periods dynamically
4. Variational Components: Uncertainty quantification through probabilistic modeling

Architecture Benefits:
- Enhanced spatial feature extraction for market microstructure
- Improved temporal modeling for regime transitions
- Dynamic attention for market condition adaptation
- Uncertainty quantification for risk management
- Multi-scale feature learning capabilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import logging
import time
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal, Independent
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("⚠️ PyTorch not available, CLVSA architecture will use fallback implementations")


class CLVSAConfig:
    """Configuration class for CLVSA architecture."""

    def __init__(self,
                 input_dim: int = 100,
                 output_dim: int = 4,
                 seq_length: int = 200,
                 conv_channels: List[int] = [32, 64, 128],
                 conv_kernel_sizes: List[int] = [3, 5, 7],
                 lstm_hidden_dim: int = 128,
                 lstm_layers: int = 2,
                 attention_heads: int = 8,
                 attention_dim: int = 256,
                 variational_dim: int = 64,
                 dropout: float = 0.2,
                 use_batch_norm: bool = True,
                 regime_aware: bool = True,
                 multi_scale: bool = True,
                 uncertainty_quantification: bool = True):
        """Initialize CLVSA configuration.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (number of targets)
            seq_length: Sequence length for temporal modeling
            conv_channels: Convolutional channel sizes
            conv_kernel_sizes: Convolutional kernel sizes
            lstm_hidden_dim: LSTM hidden dimension
            lstm_layers: Number of LSTM layers
            attention_heads: Number of attention heads
            attention_dim: Attention dimension
            variational_dim: Variational latent dimension
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            regime_aware: Whether to include regime-specific components
            multi_scale: Whether to use multi-scale processing
            uncertainty_quantification: Whether to enable uncertainty quantification
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.conv_channels = conv_channels
        self.conv_kernel_sizes = conv_kernel_sizes
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.attention_heads = attention_heads
        self.attention_dim = attention_dim
        self.variational_dim = variational_dim
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.regime_aware = regime_aware
        self.multi_scale = multi_scale
        self.uncertainty_quantification = uncertainty_quantification


class ConvolutionalFeatureExtractor(nn.Module):
    """Convolutional feature extraction module for spatial patterns."""

    def __init__(self, config: CLVSAConfig):
        super(ConvolutionalFeatureExtractor, self).__init__()

        self.config = config
        self.conv_layers = nn.ModuleList()

        # Multi-scale convolutional layers
        for i, (in_channels, out_channels, kernel_size) in enumerate(
            zip([config.input_dim] + config.conv_channels[:-1],
                config.conv_channels,
                config.conv_kernel_sizes)):
            conv_layer = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=1
            )
            self.conv_layers.append(conv_layer)

            if config.use_batch_norm:
                self.conv_layers.append(nn.BatchNorm1d(out_channels))

            self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.Dropout(config.dropout))

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional feature extractor.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)

        Returns:
            Extracted features of shape (batch_size, conv_channels[-1])
        """
        batch_size = x.size(0)

        # Reshape for convolutional layers: (batch_size, input_dim, seq_length)
        x = x.transpose(1, 2)

        # Apply convolutional layers
        for layer in self.conv_layers:
            x = layer(x)

        # Global pooling
        x = self.global_pool(x)
        x = self.flatten(x)

        return x  # Shape: (batch_size, conv_channels[-1])


class VariationalLSTMEncoder(nn.Module):
    """Variational LSTM encoder for temporal modeling with uncertainty."""

    def __init__(self, config: CLVSAConfig):
        super(VariationalLSTMEncoder, self).__init__()

        self.config = config
        self.lstm = nn.LSTM(
            input_size=config.conv_channels[-1] if config.multi_scale else config.input_dim,
            hidden_size=config.lstm_hidden_dim,
            num_layers=config.lstm_layers,
            dropout=config.dropout,
            batch_first=True,
            bidirectional=True
        )

        # Variational components
        if config.uncertainty_quantification:
            self.mu_head = nn.Linear(config.lstm_hidden_dim * 2, config.variational_dim)
            self.logvar_head = nn.Linear(config.lstm_hidden_dim * 2, config.variational_dim)
            self.latent_to_hidden = nn.Linear(config.variational_dim, config.lstm_hidden_dim * 2)
        else:
            self.hidden_projection = nn.Linear(config.lstm_hidden_dim * 2, config.attention_dim)

    def forward(self, x: torch.Tensor, regime_embedding: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through variational LSTM encoder.

        Args:
            x: Input tensor of shape (batch_size, seq_length, features)
            regime_embedding: Optional regime embedding tensor

        Returns:
            Tuple of (latent_mu, latent_logvar, hidden_states)
        """
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Concatenate forward and backward hidden states
        hidden_states = torch.cat((h_n[-1], h_n[-2]), dim=1)  # (batch_size, lstm_hidden_dim * 2)

        if self.config.uncertainty_quantification:
            # Variational encoding
            latent_mu = self.mu_head(hidden_states)
            latent_logvar = self.logvar_head(hidden_states)

            # Reparameterization trick
            latent_std = torch.exp(0.5 * latent_logvar)
            epsilon = torch.randn_like(latent_std)
            latent_sample = latent_mu + epsilon * latent_std

            # Project latent sample to hidden dimension
            hidden_states = self.latent_to_hidden(latent_sample)
        else:
            hidden_states = self.hidden_projection(hidden_states)

        return latent_mu, latent_logvar, hidden_states


class MultiHeadAttentionModule(nn.Module):
    """Multi-head attention module for temporal focus."""

    def __init__(self, config: CLVSAConfig):
        super(MultiHeadAttentionModule, self).__init__()

        self.config = config
        self.attention = nn.MultiheadAttention(
            embed_dim=config.attention_dim,
            num_heads=config.attention_heads,
            dropout=config.dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(config.attention_dim)
        self.norm2 = nn.LayerNorm(config.attention_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(config.attention_dim, config.attention_dim * 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.attention_dim * 4, config.attention_dim)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_length, attention_dim)
            mask: Optional attention mask

        Returns:
            Attention-enhanced tensor
        """
        # Multi-head attention
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        attn_output = self.norm1(x + attn_output)

        # Feed-forward
        ff_output = self.feed_forward(attn_output)
        output = self.norm2(attn_output + ff_output)

        return output


class CLVSAPredictor(nn.Module):
    """Main CLVSA predictor combining all components."""

    def __init__(self, config: CLVSAConfig):
        super(CLVSAPredictor, self).__init__()

        self.config = config

        # Feature extraction
        self.conv_extractor = ConvolutionalFeatureExtractor(config)

        # Temporal modeling
        self.variational_lstm = VariationalLSTMEncoder(config)

        # Attention mechanism
        self.attention_module = MultiHeadAttentionModule(config)

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(config.attention_dim, config.attention_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.attention_dim // 2, config.output_dim)
        )

        # Regime embedding (if regime-aware)
        if config.regime_aware:
            self.regime_embedding = nn.Embedding(10, config.variational_dim)  # 10 possible regimes

    def forward(self, x: torch.Tensor, regime_id: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through CLVSA predictor.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            regime_id: Optional regime identifier

        Returns:
            Tuple of (predictions, latent_mu, latent_logvar)
        """
        batch_size = x.size(0)

        # Convolutional feature extraction
        conv_features = self.conv_extractor(x)  # (batch_size, conv_channels[-1])

        # Prepare for LSTM: repeat conv features for sequence length
        conv_features_repeated = conv_features.unsqueeze(1).repeat(1, x.size(1), 1)

        # Add regime embedding if available
        if self.config.regime_aware and regime_id is not None:
            regime_emb = self.regime_embedding(torch.tensor([regime_id]).to(x.device))
            regime_emb = regime_emb.unsqueeze(1).repeat(1, x.size(1), 1)
            conv_features_repeated = conv_features_repeated + regime_emb

        # Variational LSTM encoding
        latent_mu, latent_logvar, hidden_states = self.variational_lstm(conv_features_repeated)

        # Multi-head attention
        attention_output = self.attention_module(hidden_states.unsqueeze(1))

        # Output projection
        predictions = self.output_projection(attention_output.squeeze(1))

        return predictions, latent_mu, latent_logvar


class CLVSARegressor(BaseEstimator, RegressorMixin):
    """CLVSA regressor for multi-output time series prediction."""

    def __init__(self,
                 config: Optional[CLVSAConfig] = None,
                 device: str = 'auto',
                 epochs: int = 100,
                 batch_size: int = 64,
                 learning_rate: float = 1e-3,
                 early_stopping_patience: int = 15):
        """Initialize CLVSA regressor.

        Args:
            config: CLVSA configuration
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            early_stopping_patience: Early stopping patience
        """
        self.config = config or CLVSAConfig()
        self.device = self._get_device(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience

        # Initialize model
        self.model = CLVSAPredictor(self.config)
        self.model.to(self.device)

        # Training components
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.kl_loss_weight = 1e-6  # KL divergence weight for VAE

        # Scalers
        self.input_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        # Training history
        self.history = {'train_loss': [], 'val_loss': []}

    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device for training."""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        else:
            return torch.device(device)

    def _prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data for training/inference."""
        # Scale inputs and targets
        X_scaled = self.input_scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_scaled = self.target_scaler.fit_transform(y)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y_scaled).to(self.device)

        return X_tensor, y_tensor

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None, regimes: Optional[np.ndarray] = None) -> 'CLVSARegressor':
        """Fit the CLVSA model.

        Args:
            X: Training features of shape (n_samples, seq_length, n_features)
            y: Target values of shape (n_samples, n_outputs)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            regimes: Regime labels for regime-aware training (optional)

        Returns:
            Self for method chaining
        """
        try:
            # Prepare data
            X_tensor, y_tensor = self._prepare_data(X, y)

            # Create dataset and dataloader
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # Validation data
            if X_val is not None and y_val is not None:
                X_val_tensor, y_val_tensor = self._prepare_data(X_val, y_val)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

            best_val_loss = float('inf')
            patience_counter = 0

            logger.info(f"🚀 Training CLVSA model on {self.device}")
            logger.info(f"   - Training samples: {len(X_tensor)}")
            logger.info(f"   - Sequence length: {X_tensor.size(1)}")
            logger.info(f"   - Input features: {X_tensor.size(2)}")
            logger.info(f"   - Output dimensions: {y_tensor.size(1)}")

            # Training loop
            for epoch in range(self.epochs):
                self.model.train()
                train_loss = 0.0

                for batch_X, batch_y in dataloader:
                    self.optimizer.zero_grad()

                    # Forward pass
                    predictions, latent_mu, latent_logvar = self.model(batch_X)

                    # Compute losses
                    mse_loss = self.mse_loss(predictions, batch_y)

                    # KL divergence for variational components
                    if self.config.uncertainty_quantification:
                        kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mu.pow(2) - latent_logvar.exp())
                        total_loss = mse_loss + self.kl_loss_weight * kl_loss
                    else:
                        total_loss = mse_loss

                    # Backward pass
                    total_loss.backward()
                    self.optimizer.step()

                    train_loss += total_loss.item()

                # Validation
                if X_val is not None and y_val is not None:
                    self.model.eval()
                    val_loss = 0.0

                    with torch.no_grad():
                        for batch_X_val, batch_y_val in val_dataloader:
                            val_predictions, _, _ = self.model(batch_X_val)
                            val_mse = self.mse_loss(val_predictions, batch_y_val)
                            val_loss += val_mse.item()

                    val_loss /= len(val_dataloader)
                    self.history['val_loss'].append(val_loss)

                    # Learning rate scheduling
                    self.scheduler.step(val_loss)

                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        torch.save(self.model.state_dict(), 'clvsa_best_model.pth')
                    else:
                        patience_counter += 1

                    if patience_counter >= self.early_stopping_patience:
                        logger.info(f"⏹️ Early stopping at epoch {epoch + 1}")
                        break

                avg_train_loss = train_loss / len(dataloader)
                self.history['train_loss'].append(avg_train_loss)

                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{self.epochs} - "
                              f"Train Loss: {avg_train_loss:.6f}" +
                              (f" - Val Loss: {val_loss:.6f}" if X_val is not None else ""))

            # Load best model
            if X_val is not None:
                self.model.load_state_dict(torch.load('clvsa_best_model.pth'))

            logger.info("✅ CLVSA model training completed")
            return self

        except Exception as e:
            logger.error(f"❌ CLVSA training failed: {e}")
            raise

    def predict(self, X: np.ndarray, return_uncertainty: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions using the trained CLVSA model.

        Args:
            X: Input features of shape (n_samples, seq_length, n_features)
            return_uncertainty: Whether to return prediction uncertainty

        Returns:
            Predictions (and uncertainty if requested)
        """
        try:
            self.model.eval()

            # Prepare data
            X_scaled = self.input_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)

            with torch.no_grad():
                if return_uncertainty and self.config.uncertainty_quantification:
                    # Monte Carlo dropout for uncertainty estimation
                    predictions_list = []
                    for _ in range(10):  # 10 forward passes
                        predictions, _, _ = self.model(X_tensor)
                        predictions_list.append(predictions.cpu().numpy())

                    predictions = np.mean(predictions_list, axis=0)
                    uncertainty = np.std(predictions_list, axis=0)
                else:
                    predictions, _, _ = self.model(X_tensor)
                    predictions = predictions.cpu().numpy()
                    uncertainty = None

            # Inverse transform predictions
            predictions = self.target_scaler.inverse_transform(predictions)

            if return_uncertainty:
                return predictions, uncertainty
            else:
                return predictions

        except Exception as e:
            logger.error(f"❌ CLVSA prediction failed: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and training statistics."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'model_type': 'CLVSA',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_dim': self.config.input_dim,
            'output_dim': self.config.output_dim,
            'sequence_length': self.config.seq_length,
            'conv_channels': self.config.conv_channels,
            'lstm_hidden_dim': self.config.lstm_hidden_dim,
            'attention_heads': self.config.attention_heads,
            'variational_dim': self.config.variational_dim,
            'regime_aware': self.config.regime_aware,
            'uncertainty_quantification': self.config.uncertainty_quantification,
            'device': str(self.device),
            'training_epochs': len(self.history['train_loss']),
            'final_train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'final_val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None
        }


# Factory functions for creating CLVSA models
def create_clvsa_model(config: Dict[str, Any]) -> CLVSARegressor:
    """Create CLVSA model from configuration."""
    clvsa_config = CLVSAConfig(**config.get('clvsa_params', {}))
    model_config = config.get('model_params', {})

    return CLVSARegressor(
        config=clvsa_config,
        device=config.get('device', 'auto'),
        epochs=model_config.get('epochs', 100),
        batch_size=model_config.get('batch_size', 64),
        learning_rate=model_config.get('learning_rate', 1e-3),
        early_stopping_patience=model_config.get('early_stopping_patience', 15)
    )


# Fallback implementation for when PyTorch is not available
class FallbackCLVSARegressor(BaseEstimator, RegressorMixin):
    """Fallback CLVSA regressor when PyTorch is not available."""

    def __init__(self, **kwargs):
        self.params = kwargs
        self.is_fitted = False

    def fit(self, X, y):
        self.is_fitted = True
        logger.warning("⚠️ Using fallback CLVSA implementation without PyTorch")
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        # Return zero predictions as fallback
        return np.zeros((len(X), 4))


def get_clvsa_model(config: Dict[str, Any]) -> Union[CLVSARegressor, FallbackCLVSARegressor]:
    """Get CLVSA model with fallback support."""
    if TORCH_AVAILABLE:
        return create_clvsa_model(config)
    else:
        logger.warning("⚠️ PyTorch not available, using fallback CLVSA implementation")
        return FallbackCLVSARegressor()