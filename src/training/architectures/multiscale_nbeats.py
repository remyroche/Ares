"""
MultiScaleNBEATS: Enhanced Neural Basis Expansion Analysis for Time Series

This module implements MultiScaleNBEATS architecture for improved ML model performance
in financial time series forecasting with multi-scale temporal pattern recognition.

Key features:
- Multi-scale temporal decomposition
- Hierarchical pattern recognition
- Regime-aware forecasting
- Uncertainty quantification
- Meta-learning capabilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Optimized imports
from src.utils.tprint import tprint
from src.utils.logger import get_logger
from src.core.decorators import handles_errors, traced, validates, log_execution_time
from src.utils.common_operations import (
    validate_dataframe, validate_dataframe_columns, safe_dataframe_operation,
    timed_operation, memory_checkpoint, gpu_context
)

logger = get_logger('MultiScaleNBEATS')

@dataclass
class MultiScaleNBEATSConfig:
    """Configuration for MultiScaleNBEATS architecture."""
    
    # Input configuration
    input_features: int = 50
    sequence_length: int = 60
    forecast_horizon: int = 12  # 1 hour ahead (12 * 5min)
    
    # Multi-scale configuration
    scales: List[int] = field(default_factory=lambda: [1, 3, 6, 12])  # Different time scales
    scale_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.2, 0.1])
    
    # NBEATS blocks configuration
    num_blocks: int = 3
    block_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    block_dropout: float = 0.1
    
    # Basis functions
    basis_functions: List[str] = field(default_factory=lambda: ['trend', 'seasonality', 'residual'])
    trend_degree: int = 3
    seasonality_periods: List[int] = field(default_factory=lambda: [12, 24, 48])  # 1h, 2h, 4h periods
    
    # Attention mechanism
    use_attention: bool = True
    attention_heads: int = 4
    attention_dim: int = 64
    
    # Regime awareness
    num_regimes: int = 3
    regime_embedding_dim: int = 16
    
    # Training configuration
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    
    def __post_init__(self):
        """Calculate derived properties."""
        # Validate scale weights sum to 1
        if abs(sum(self.scale_weights) - 1.0) > 1e-6:
            self.scale_weights = [w / sum(self.scale_weights) for w in self.scale_weights]


class TrendBasis(nn.Module):
    """Trend basis function for NBEATS."""
    
    def __init__(self, degree: int, input_dim: int, output_dim: int):
        super().__init__()
        self.degree = degree
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Trend coefficients
        self.trend_coeffs = nn.Linear(input_dim, degree + 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute trend basis functions."""
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Get trend coefficients
        coeffs = self.trend_coeffs(x)  # (batch_size, degree + 1)
        
        # Create time indices
        t = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(0) / seq_len
        
        # Compute polynomial basis
        trend = torch.zeros(batch_size, seq_len, device=x.device)
        for i in range(self.degree + 1):
            trend += coeffs[:, i:i+1] * (t ** i)
        
        return trend


class SeasonalityBasis(nn.Module):
    """Seasonality basis function for NBEATS."""
    
    def __init__(self, periods: List[int], input_dim: int, output_dim: int):
        super().__init__()
        self.periods = periods
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Seasonality coefficients
        total_coeffs = sum(2 * p for p in periods)  # 2 coefficients per period (sin, cos)
        self.seasonal_coeffs = nn.Linear(input_dim, total_coeffs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute seasonality basis functions."""
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Get seasonality coefficients
        coeffs = self.seasonal_coeffs(x)  # (batch_size, total_coeffs)
        
        # Create time indices
        t = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(0)
        
        # Compute seasonal basis
        seasonal = torch.zeros(batch_size, seq_len, device=x.device)
        coeff_idx = 0
        
        for period in self.periods:
            # Sine component
            sin_coeff = coeffs[:, coeff_idx:coeff_idx+1]
            seasonal += sin_coeff * torch.sin(2 * math.pi * t / period)
            coeff_idx += 1
            
            # Cosine component
            cos_coeff = coeffs[:, coeff_idx:coeff_idx+1]
            seasonal += cos_coeff * torch.cos(2 * math.pi * t / period)
            coeff_idx += 1
        
        return seasonal


class ResidualBasis(nn.Module):
    """Residual basis function for NBEATS."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Residual network
        self.residual_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute residual basis functions."""
        return self.residual_net(x)


class NBEATSBlock(nn.Module):
    """Single NBEATS block with multi-scale capabilities."""
    
    def __init__(self, config: MultiScaleNBEATSConfig, scale: int):
        super().__init__()
        self.config = config
        self.scale = scale
        
        # Input projection
        self.input_projection = nn.Linear(config.input_features, config.block_layers[0])
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(config.block_layers) - 1):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(config.block_layers[i], config.block_layers[i+1]),
                    nn.ReLU(),
                    nn.Dropout(config.block_dropout)
                )
            )
        
        # Basis functions
        self.basis_functions = nn.ModuleDict({
            'trend': TrendBasis(config.trend_degree, config.block_layers[-1], config.forecast_horizon),
            'seasonality': SeasonalityBasis(config.seasonality_periods, config.block_layers[-1], config.forecast_horizon),
            'residual': ResidualBasis(config.block_layers[-1], config.forecast_horizon)
        })
        
        # Attention mechanism (if enabled)
        if config.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=config.block_layers[-1],
                num_heads=config.attention_heads,
                dropout=config.block_dropout,
                batch_first=True
            )
        else:
            self.attention = None
        
        # Scale-specific output projection
        self.scale_projection = nn.Linear(config.block_layers[-1], config.forecast_horizon)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through NBEATS block."""
        # Input projection
        h = self.input_projection(x)
        
        # Hidden layers
        for layer in self.hidden_layers:
            h = layer(h)
        
        # Attention mechanism
        if self.attention is not None:
            # Reshape for attention
            h_reshaped = h.unsqueeze(1)  # (batch_size, 1, hidden_dim)
            attn_out, _ = self.attention(h_reshaped, h_reshaped, h_reshaped)
            h = attn_out.squeeze(1)
        
        # Basis function decomposition
        basis_outputs = {}
        for basis_name, basis_func in self.basis_functions.items():
            basis_outputs[basis_name] = basis_func(h)
        
        # Scale-specific output
        scale_output = self.scale_projection(h)
        
        # Combine basis functions
        combined_output = sum(basis_outputs.values()) + scale_output
        
        return {
            'forecast': combined_output,
            'basis_components': basis_outputs,
            'scale_output': scale_output,
            'hidden_state': h
        }


class RegimeAwareAttention(nn.Module):
    """Regime-aware attention mechanism."""
    
    def __init__(self, config: MultiScaleNBEATSConfig):
        super().__init__()
        self.config = config
        
        # Regime embedding
        self.regime_embedding = nn.Embedding(config.num_regimes, config.regime_embedding_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=config.regime_embedding_dim,
            num_heads=config.attention_heads,
            dropout=config.block_dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(config.regime_embedding_dim, 1)
        
    def forward(self, regime_ids: torch.Tensor, scale_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Apply regime-aware attention to scale outputs."""
        batch_size = regime_ids.size(0)
        
        # Get regime embeddings
        regime_emb = self.regime_embedding(regime_ids)  # (batch_size, regime_embedding_dim)
        
        # Stack scale outputs
        scale_stack = torch.stack(scale_outputs, dim=1)  # (batch_size, num_scales, forecast_horizon)
        
        # Apply attention
        attended, _ = self.attention(
            regime_emb.unsqueeze(1),  # Query
            scale_stack,  # Key
            scale_stack   # Value
        )
        
        # Output projection
        attention_weights = self.output_projection(attended).squeeze(1)  # (batch_size, num_scales)
        
        # Weighted combination
        weighted_output = torch.sum(scale_stack * attention_weights.unsqueeze(-1), dim=1)
        
        return weighted_output, attention_weights


class MultiScaleNBEATS(nn.Module):
    """Complete MultiScaleNBEATS architecture."""
    
    def __init__(self, config: MultiScaleNBEATSConfig):
        super().__init__()
        self.config = config
        
        # Multi-scale NBEATS blocks
        self.scale_blocks = nn.ModuleDict()
        for scale in config.scales:
            self.scale_blocks[str(scale)] = NBEATSBlock(config, scale)
        
        # Regime-aware attention
        self.regime_attention = RegimeAwareAttention(config)
        
        # Final output layers
        self.output_layers = nn.ModuleDict({
            'forecast': nn.Linear(config.forecast_horizon, config.forecast_horizon),
            'uncertainty': nn.Linear(config.forecast_horizon, 1),
            'regime_prediction': nn.Linear(config.forecast_horizon, config.num_regimes)
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
    
    def forward(self, x: torch.Tensor, regime_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MultiScaleNBEATS.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_features)
            regime_ids: Optional regime IDs of shape (batch_size,)
            
        Returns:
            Dictionary containing forecasts and uncertainty measures
        """
        batch_size = x.size(0)
        
        # Process through each scale
        scale_outputs = []
        scale_forecasts = []
        
        for scale in self.config.scales:
            scale_block = self.scale_blocks[str(scale)]
            scale_result = scale_block(x)
            
            scale_outputs.append(scale_result['forecast'])
            scale_forecasts.append(scale_result)
        
        # Regime-aware attention combination
        if regime_ids is not None:
            final_forecast, attention_weights = self.regime_attention(regime_ids, scale_outputs)
        else:
            # Simple weighted combination
            weights = torch.tensor(self.config.scale_weights, device=x.device)
            final_forecast = torch.sum(torch.stack(scale_outputs) * weights.view(-1, 1, 1), dim=0)
            attention_weights = weights.unsqueeze(0).expand(batch_size, -1)
        
        # Final outputs
        outputs = {
            'forecast': self.output_layers['forecast'](final_forecast),
            'uncertainty': self.output_layers['uncertainty'](final_forecast),
            'regime_prediction': self.output_layers['regime_prediction'](final_forecast),
            'scale_forecasts': scale_forecasts,
            'attention_weights': attention_weights
        }
        
        return outputs
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute comprehensive loss."""
        losses = {}
        
        # Forecast loss
        if 'forecast' in targets:
            forecast_loss = F.mse_loss(outputs['forecast'], targets['forecast'])
            losses['forecast_loss'] = forecast_loss
        
        # Regime prediction loss
        if 'regime' in targets:
            regime_loss = F.cross_entropy(outputs['regime_prediction'], targets['regime'])
            losses['regime_loss'] = regime_loss
        
        # Uncertainty loss (encourage appropriate uncertainty)
        if 'uncertainty' in targets:
            uncertainty_loss = F.mse_loss(outputs['uncertainty'], targets['uncertainty'])
            losses['uncertainty_loss'] = uncertainty_loss
        
        # Scale consistency loss
        scale_forecasts = outputs['scale_forecasts']
        consistency_loss = 0
        for i in range(len(scale_forecasts)):
            for j in range(i+1, len(scale_forecasts)):
                consistency_loss += F.mse_loss(scale_forecasts[i]['forecast'], scale_forecasts[j]['forecast'])
        
        if len(scale_forecasts) > 1:
            consistency_loss /= (len(scale_forecasts) * (len(scale_forecasts) - 1) / 2)
        
        losses['consistency_loss'] = consistency_loss
        
        # Total loss
        total_loss = (
            losses.get('forecast_loss', 0) * 0.5 +
            losses.get('regime_loss', 0) * 0.2 +
            losses.get('uncertainty_loss', 0) * 0.2 +
            consistency_loss * 0.1
        )
        losses['total_loss'] = total_loss
        
        return losses


class MultiScaleNBEATSTrainer:
    """Trainer for MultiScaleNBEATS architecture."""
    
    def __init__(self, model: MultiScaleNBEATS, config: MultiScaleNBEATSConfig):
        self.model = model
        self.config = config
        self.logger = get_logger('MultiScaleNBEATSTrainer')
        
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
        epoch_losses = {'total_loss': 0, 'forecast_loss': 0, 'regime_loss': 0, 'uncertainty_loss': 0, 'consistency_loss': 0}
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data, targets.get('regime_ids'))
            
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
        epoch_losses = {'total_loss': 0, 'forecast_loss': 0, 'regime_loss': 0, 'uncertainty_loss': 0, 'consistency_loss': 0}
        
        with torch.no_grad():
            for data, targets in dataloader:
                outputs = self.model(data, targets.get('regime_ids'))
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
def create_multiscale_nbeats(config: Optional[MultiScaleNBEATSConfig] = None) -> MultiScaleNBEATS:
    """Create MultiScaleNBEATS model with default configuration."""
    if config is None:
        config = MultiScaleNBEATSConfig()
    
    return MultiScaleNBEATS(config)


def create_multiscale_nbeats_trainer(model: MultiScaleNBEATS, config: Optional[MultiScaleNBEATSConfig] = None) -> MultiScaleNBEATSTrainer:
    """Create MultiScaleNBEATS trainer."""
    if config is None:
        config = MultiScaleNBEATSConfig()
    
    return MultiScaleNBEATSTrainer(model, config)


# Test function
if __name__ == '__main__':
    tprint('🧪 Testing MultiScaleNBEATS Architecture')
    
    # Test configuration
    config = MultiScaleNBEATSConfig(
        input_features=50,
        sequence_length=60,
        forecast_horizon=12
    )
    
    tprint(f'📊 MultiScaleNBEATS Configuration:')
    tprint(f'   → Input features: {config.input_features}')
    tprint(f'   → Sequence length: {config.sequence_length}')
    tprint(f'   → Forecast horizon: {config.forecast_horizon}')
    tprint(f'   → Scales: {config.scales}')
    tprint(f'   → Scale weights: {config.scale_weights}')
    tprint(f'   → Number of regimes: {config.num_regimes}')
    
    # Test model creation
    try:
        model = create_multiscale_nbeats(config)
        trainer = create_multiscale_nbeats_trainer(model, config)
        
        # Test forward pass
        batch_size = 32
        test_input = torch.randn(batch_size, config.sequence_length, config.input_features)
        test_regime_ids = torch.randint(0, config.num_regimes, (batch_size,))
        
        with torch.no_grad():
            outputs = model(test_input, test_regime_ids)
        
        tprint('✅ MultiScaleNBEATS model created successfully')
        tprint(f'   → Output keys: {list(outputs.keys())}')
        tprint(f'   → Forecast shape: {outputs["forecast"].shape}')
        tprint(f'   → Uncertainty shape: {outputs["uncertainty"].shape}')
        tprint(f'   → Regime prediction shape: {outputs["regime_prediction"].shape}')
        tprint(f'   → Attention weights shape: {outputs["attention_weights"].shape}')
        
        # Test loss computation
        test_targets = {
            'forecast': torch.randn(batch_size, config.forecast_horizon),
            'regime': test_regime_ids,
            'uncertainty': torch.randn(batch_size, 1),
            'regime_ids': test_regime_ids
        }
        
        losses = model.compute_loss(outputs, test_targets)
        tprint(f'   → Loss components: {list(losses.keys())}')
        tprint(f'   → Total loss: {losses["total_loss"].item():.4f}')
        
    except Exception as e:
        tprint(f'❌ Error creating MultiScaleNBEATS model: {e}')
    
    tprint('✅ MultiScaleNBEATS Architecture test completed!')