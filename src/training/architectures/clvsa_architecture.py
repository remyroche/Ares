"""
CLVSA Architecture: Convolutional-LSTM-Variational-Spatial-Attention

This module implements the CLVSA architecture for advanced financial time series modeling:
1. Convolutional Layers (Spatial Feature Extraction)
2. LSTM Layers (Temporal Dependencies) 
3. Attention Mechanism (Relevant Time Focus)
4. Variational Components (Uncertainty Quantification)

The architecture is designed for regime-aware trading with uncertainty quantification
and multi-scale temporal pattern recognition.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
import math

# Optimized imports
from src.utils.tprint import tprint
from src.utils.logger import get_logger
from src.core.decorators import handles_errors, traced, validates, log_execution_time
from src.utils.common_operations import (
    validate_dataframe, validate_dataframe_columns, safe_dataframe_operation,
    timed_operation, memory_checkpoint, gpu_context
)

logger = get_logger('CLVSAArchitecture')

@dataclass
class CLVSAConfig:
    """Configuration for CLVSA architecture."""
    
    # Input configuration
    input_features: int = 50
    sequence_length: int = 60  # 5-minute bars for 5 hours of history
    num_regimes: int = 3  # Low, Medium, High volatility regimes
    
    # Convolutional layers
    conv_filters: List[int] = field(default_factory=lambda: [32, 64, 128])
    conv_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    conv_dropout: float = 0.2
    
    # LSTM configuration
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3
    lstm_bidirectional: bool = True
    
    # Attention mechanism
    attention_heads: int = 8
    attention_dim: int = 64
    attention_dropout: float = 0.1
    
    # Variational components
    latent_dim: int = 32
    variational_dropout: float = 0.2
    kl_weight: float = 0.1  # KL divergence weight
    
    # Output configuration
    num_outputs: int = 10  # Multiple prediction targets
    output_activation: str = 'sigmoid'  # For probability outputs
    
    # Training configuration
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    
    def __post_init__(self):
        """Calculate derived properties."""
        # Calculate total LSTM hidden size (bidirectional)
        self.total_lstm_hidden = self.lstm_hidden_size * (2 if self.lstm_bidirectional else 1)
        
        # Calculate attention input dimension
        self.attention_input_dim = self.total_lstm_hidden


class SpatialFeatureExtractor(nn.Module):
    """Convolutional layers for spatial feature extraction."""
    
    def __init__(self, config: CLVSAConfig):
        super().__init__()
        self.config = config
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = 1  # Start with single channel
        
        for i, (filters, kernel_size) in enumerate(zip(config.conv_filters, config.conv_kernel_sizes)):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, filters, kernel_size, padding=kernel_size//2),
                    nn.BatchNorm1d(filters),
                    nn.ReLU(),
                    nn.Dropout(config.conv_dropout)
                )
            )
            in_channels = filters
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spatial feature extractor.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_features)
            
        Returns:
            Spatial features of shape (batch_size, conv_filters[-1])
        """
        # Reshape for convolution: (batch, features, sequence)
        x = x.transpose(1, 2)
        
        # Apply convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)  # Remove sequence dimension
        
        return x


class TemporalDependencyModel(nn.Module):
    """LSTM layers for temporal dependency modeling."""
    
    def __init__(self, config: CLVSAConfig):
        super().__init__()
        self.config = config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_features,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0,
            bidirectional=config.lstm_bidirectional,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(
            config.total_lstm_hidden,
            config.attention_input_dim
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through temporal dependency model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_features)
            
        Returns:
            Temporal features of shape (batch_size, sequence_length, attention_input_dim)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Project to attention input dimension
        temporal_features = self.output_projection(lstm_out)
        
        return temporal_features


class AttentionMechanism(nn.Module):
    """Multi-head attention mechanism for relevant time focus."""
    
    def __init__(self, config: CLVSAConfig):
        super().__init__()
        self.config = config
        
        # Multi-head attention
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=config.attention_input_dim,
            num_heads=config.attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(config.attention_input_dim)
        self.layer_norm2 = nn.LayerNorm(config.attention_input_dim)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(config.attention_input_dim, config.attention_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.attention_dropout),
            nn.Linear(config.attention_dim * 2, config.attention_input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through attention mechanism.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, attention_input_dim)
            
        Returns:
            Tuple of (attended_features, attention_weights)
        """
        # Multi-head attention
        attn_output, attn_weights = self.multihead_attention(x, x, x)
        
        # Residual connection and layer norm
        x = self.layer_norm1(x + attn_output)
        
        # Feed-forward network
        ff_output = self.feed_forward(x)
        
        # Residual connection and layer norm
        output = self.layer_norm2(x + ff_output)
        
        return output, attn_weights


class VariationalEncoder(nn.Module):
    """Variational encoder for uncertainty quantification."""
    
    def __init__(self, config: CLVSAConfig):
        super().__init__()
        self.config = config
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(config.attention_input_dim, config.attention_dim),
            nn.ReLU(),
            nn.Dropout(config.variational_dropout),
            nn.Linear(config.attention_dim, config.latent_dim * 2)  # Mean and log_var
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through variational encoder.
        
        Args:
            x: Input tensor of shape (batch_size, attention_input_dim)
            
        Returns:
            Tuple of (z, mu, log_var) for reparameterization trick
        """
        # Encode to latent space
        encoded = self.encoder(x)
        mu, log_var = torch.chunk(encoded, 2, dim=-1)
        
        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, log_var


class VariationalDecoder(nn.Module):
    """Variational decoder for uncertainty quantification."""
    
    def __init__(self, config: CLVSAConfig):
        super().__init__()
        self.config = config
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.attention_dim),
            nn.ReLU(),
            nn.Dropout(config.variational_dropout),
            nn.Linear(config.attention_dim, config.attention_input_dim)
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through variational decoder.
        
        Args:
            z: Latent tensor of shape (batch_size, latent_dim)
            
        Returns:
            Reconstructed features of shape (batch_size, attention_input_dim)
        """
        return self.decoder(z)


class CLVSAArchitecture(nn.Module):
    """Complete CLVSA architecture implementation."""
    
    def __init__(self, config: CLVSAConfig):
        super().__init__()
        self.config = config
        
        # Initialize components
        self.spatial_extractor = SpatialFeatureExtractor(config)
        self.temporal_model = TemporalDependencyModel(config)
        self.attention_mechanism = AttentionMechanism(config)
        self.variational_encoder = VariationalEncoder(config)
        self.variational_decoder = VariationalDecoder(config)
        
        # Output layers
        self.output_layers = nn.ModuleDict({
            'regime_prediction': nn.Linear(config.attention_input_dim, config.num_regimes),
            'price_prediction': nn.Linear(config.attention_input_dim, config.num_outputs),
            'uncertainty': nn.Linear(config.attention_input_dim, 1)
        })
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through CLVSA architecture.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_features)
            
        Returns:
            Dictionary containing predictions and uncertainty measures
        """
        batch_size = x.size(0)
        
        # 1. Spatial feature extraction
        spatial_features = self.spatial_extractor(x)
        
        # 2. Temporal dependency modeling
        temporal_features = self.temporal_model(x)
        
        # 3. Attention mechanism
        attended_features, attention_weights = self.attention_mechanism(temporal_features)
        
        # 4. Global pooling for final features
        global_features = attended_features.mean(dim=1)  # (batch_size, attention_input_dim)
        
        # 5. Variational encoding for uncertainty
        z, mu, log_var = self.variational_encoder(global_features)
        
        # 6. Variational decoding
        reconstructed_features = self.variational_decoder(z)
        
        # 7. Output predictions
        outputs = {}
        for name, layer in self.output_layers.items():
            outputs[name] = layer(global_features)
        
        # Add uncertainty measures
        outputs['latent_z'] = z
        outputs['mu'] = mu
        outputs['log_var'] = log_var
        outputs['attention_weights'] = attention_weights
        outputs['reconstructed_features'] = reconstructed_features
        
        return outputs
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive loss including variational components.
        
        Args:
            outputs: Model outputs
            targets: Target values
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Prediction losses
        if 'regime' in targets:
            regime_loss = F.cross_entropy(outputs['regime_prediction'], targets['regime'])
            losses['regime_loss'] = regime_loss
        
        if 'price' in targets:
            price_loss = F.mse_loss(outputs['price_prediction'], targets['price'])
            losses['price_loss'] = price_loss
        
        # Variational loss (KL divergence)
        mu = outputs['mu']
        log_var = outputs['log_var']
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        kl_loss = kl_loss.mean()
        losses['kl_loss'] = kl_loss
        
        # Reconstruction loss
        if 'reconstructed_features' in outputs:
            recon_loss = F.mse_loss(outputs['reconstructed_features'], outputs.get('global_features', torch.zeros_like(outputs['reconstructed_features'])))
            losses['reconstruction_loss'] = recon_loss
        
        # Total loss
        total_loss = (
            losses.get('regime_loss', 0) * 0.4 +
            losses.get('price_loss', 0) * 0.4 +
            kl_loss * self.config.kl_weight +
            losses.get('reconstruction_loss', 0) * 0.1
        )
        losses['total_loss'] = total_loss
        
        return losses


class CLVSATrainer:
    """Trainer for CLVSA architecture."""
    
    def __init__(self, model: CLVSAArchitecture, config: CLVSAConfig):
        self.model = model
        self.config = config
        self.logger = get_logger('CLVSATrainer')
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
    @traced(span_name='train_epoch')
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {'total_loss': 0, 'regime_loss': 0, 'price_loss': 0, 'kl_loss': 0}
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            
            # Compute loss
            losses = self.model.compute_loss(outputs, targets)
            
            # Backward pass
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            
            self.optimizer.step()
            
            # Accumulate losses
            for key, value in losses.items():
                epoch_losses[key] += value.item()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(dataloader)
        
        return epoch_losses
    
    @traced(span_name='validate_epoch')
    def validate_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses = {'total_loss': 0, 'regime_loss': 0, 'price_loss': 0, 'kl_loss': 0}
        
        with torch.no_grad():
            for data, targets in dataloader:
                outputs = self.model(data)
                losses = self.model.compute_loss(outputs, targets)
                
                for key, value in losses.items():
                    epoch_losses[key] += value.item()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= len(dataloader)
        
        return epoch_losses
    
    def train(self, train_loader: torch.utils.data.DataLoader, 
              val_loader: torch.utils.data.DataLoader, 
              epochs: int = 100) -> Dict[str, List[float]]:
        """Complete training loop."""
        history = {'train_loss': [], 'val_loss': [], 'lr': []}
        
        for epoch in range(epochs):
            # Training
            train_losses = self.train_epoch(train_loader)
            
            # Validation
            val_losses = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_losses['total_loss'])
            
            # Record history
            history['train_loss'].append(train_losses['total_loss'])
            history['val_loss'].append(val_losses['total_loss'])
            history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Log progress
            if epoch % 10 == 0:
                self.logger.info(f'Epoch {epoch}: Train Loss: {train_losses["total_loss"]:.4f}, '
                               f'Val Loss: {val_losses["total_loss"]:.4f}, '
                               f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
        
        return history


# Factory functions
def create_clvsa_model(config: Optional[CLVSAConfig] = None) -> CLVSAArchitecture:
    """Create CLVSA model with default configuration."""
    if config is None:
        config = CLVSAConfig()
    
    return CLVSAArchitecture(config)


def create_clvsa_trainer(model: CLVSAArchitecture, config: Optional[CLVSAConfig] = None) -> CLVSATrainer:
    """Create CLVSA trainer."""
    if config is None:
        config = CLVSAConfig()
    
    return CLVSATrainer(model, config)


# Test function
if __name__ == '__main__':
    tprint('🧪 Testing CLVSA Architecture')
    
    # Test configuration
    config = CLVSAConfig(
        input_features=50,
        sequence_length=60,
        num_regimes=3
    )
    
    tprint(f'📊 CLVSA Configuration:')
    tprint(f'   → Input features: {config.input_features}')
    tprint(f'   → Sequence length: {config.sequence_length}')
    tprint(f'   → Number of regimes: {config.num_regimes}')
    tprint(f'   → LSTM hidden size: {config.lstm_hidden_size}')
    tprint(f'   → Attention heads: {config.attention_heads}')
    tprint(f'   → Latent dimension: {config.latent_dim}')
    
    # Test model creation
    try:
        model = create_clvsa_model(config)
        trainer = create_clvsa_trainer(model, config)
        
        # Test forward pass
        batch_size = 32
        test_input = torch.randn(batch_size, config.sequence_length, config.input_features)
        
        with torch.no_grad():
            outputs = model(test_input)
        
        tprint('✅ CLVSA model created successfully')
        tprint(f'   → Output keys: {list(outputs.keys())}')
        tprint(f'   → Regime prediction shape: {outputs["regime_prediction"].shape}')
        tprint(f'   → Price prediction shape: {outputs["price_prediction"].shape}')
        tprint(f'   → Uncertainty shape: {outputs["uncertainty"].shape}')
        
        # Test loss computation
        test_targets = {
            'regime': torch.randint(0, config.num_regimes, (batch_size,)),
            'price': torch.randn(batch_size, config.num_outputs)
        }
        
        losses = model.compute_loss(outputs, test_targets)
        tprint(f'   → Loss components: {list(losses.keys())}')
        tprint(f'   → Total loss: {losses["total_loss"].item():.4f}')
        
    except Exception as e:
        tprint(f'❌ Error creating CLVSA model: {e}')
    
    tprint('✅ CLVSA Architecture test completed!')