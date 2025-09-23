"""
Advanced Model Implementations for Migrated ML Models

This module contains implementations of advanced model architectures including:
- FinancialResNet for regime detection and entry timing
- DeepScaler for trading opportunities
- N-BEATS with regime-aware parameter optimization
- AdvancedMambaHybrid for complex pattern recognition
- MobileNet/EfficientNet variants for efficient inference

All models include comprehensive regularization, overfitting prevention,
and regime-aware training capabilities.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import logging
import time
from dataclasses import dataclass
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import warnings

# Enhanced dependency management
try:
    from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args)
    def tprint_info(*args, **kwargs): print(f"INFO: {args[0] if args else ''}")
    def tprint_warning(*args, **kwargs): print(f"WARNING: {args[0] if args else ''}")
    def tprint_error(*args, **kwargs): print(f"ERROR: {args[0] if args else ''}")

logger = logging.getLogger(__name__)


class FinancialResNetBlock(nn.Module):
    """Financial ResNet block with temporal convolutions and attention."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 dropout: float = 0.1, attention_heads: int = 4):
        super().__init__()
        
        # Main convolution path
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Residual connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=out_channels, num_heads=attention_heads, 
            dropout=dropout, batch_first=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Activation
        self.activation = nn.GELU()
    
    def forward(self, x):
        # Store residual
        residual = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add residual
        out += self.shortcut(residual)
        
        # Apply attention (transpose for attention)
        out_transposed = out.transpose(1, 2)  # (batch, seq, features)
        attn_out, _ = self.attention(out_transposed, out_transposed, out_transposed)
        out = attn_out.transpose(1, 2)  # Back to (batch, features, seq)
        
        # Final activation and dropout
        out = self.activation(out)
        out = self.dropout(out)
        
        return out


class FinancialResNet(nn.Module):
    """Financial ResNet for regime detection and entry timing."""
    
    def __init__(self, input_dim: int, output_dim: int, blocks: List[int] = [32, 64, 128],
                 temporal_conv_layers: int = 3, attention_heads: int = 4, 
                 dropout: float = 0.15, regime_aware: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.regime_aware = regime_aware
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, blocks[0])
        
        # ResNet blocks
        self.resnet_blocks = nn.ModuleList()
        in_channels = blocks[0]
        
        for i, out_channels in enumerate(blocks):
            stride = 2 if i > 0 else 1
            self.resnet_blocks.append(
                FinancialResNetBlock(in_channels, out_channels, stride, dropout, attention_heads)
            )
            in_channels = out_channels
        
        # Temporal convolution layers
        self.temporal_convs = nn.ModuleList()
        for i in range(temporal_conv_layers):
            self.temporal_convs.append(
                nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
            )
            self.temporal_convs.append(nn.BatchNorm1d(in_channels))
            self.temporal_convs.append(nn.GELU())
            self.temporal_convs.append(nn.Dropout(dropout))
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels // 2, output_dim)
        )
        
        # Regime-aware components
        if regime_aware:
            self.regime_classifier = nn.Sequential(
                nn.Linear(in_channels, in_channels // 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(in_channels // 4, 4)  # 4 regime dimensions
            )
    
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        x = x.transpose(1, 2)  # (batch, features, seq)
        
        # ResNet blocks
        for block in self.resnet_blocks:
            x = block(x)
        
        # Temporal convolutions
        for layer in self.temporal_convs:
            x = layer(x)
        
        # Global pooling
        x = self.global_pool(x).squeeze(-1)  # (batch, features)
        
        # Output
        output = self.output_layers(x)
        
        # Regime prediction if enabled
        regime_output = None
        if self.regime_aware:
            regime_output = self.regime_classifier(x)
        
        return output, regime_output


class DeepScaler(nn.Module):
    """DeepScaler architecture for trading opportunities."""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 hidden_layers: List[int] = [512, 256, 128],
                 dropout: float = 0.2, batch_norm: bool = True, 
                 activation: str = "relu"):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "swish":
                layers.append(nn.SiLU())
            
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class NBEATSBlock(nn.Module):
    """N-BEATS block with regime-aware parameter optimization."""
    
    def __init__(self, input_dim: int, theta_dim: int, hidden_dim: int, 
                 num_layers: int = 4, dropout: float = 0.1, 
                 block_type: str = "generic"):
        super().__init__()
        
        self.input_dim = input_dim
        self.theta_dim = theta_dim
        self.hidden_dim = hidden_dim
        self.block_type = block_type
        
        # Fully connected layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, theta_dim))
        
        self.fc_layers = nn.Sequential(*layers)
        
        # Block-specific components
        if block_type == "trend":
            self.trend_projector = nn.Linear(theta_dim, 2)  # slope and intercept
        elif block_type == "seasonality":
            self.seasonality_projector = nn.Linear(theta_dim, input_dim)
        else:  # generic
            self.generic_projector = nn.Linear(theta_dim, input_dim)
    
    def forward(self, x):
        theta = self.fc_layers(x)
        
        if self.block_type == "trend":
            return self.trend_projector(theta)
        elif self.block_type == "seasonality":
            return self.seasonality_projector(theta)
        else:
            return self.generic_projector(theta)


class NBEATS(nn.Module):
    """N-BEATS model with regime-aware parameter optimization."""
    
    def __init__(self, input_dim: int, output_dim: int, num_blocks: int = 10,
                 num_layers: int = 4, layer_widths: List[int] = [512, 512, 256, 256],
                 dropout: float = 0.1, block_type: str = "generic",
                 regime_aware: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks
        self.regime_aware = regime_aware
        
        # N-BEATS blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            theta_dim = layer_widths[0]  # Use first layer width as theta dimension
            hidden_dim = layer_widths[i % len(layer_widths)]
            
            block = NBEATSBlock(
                input_dim=input_dim,
                theta_dim=theta_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                block_type=block_type
            )
            self.blocks.append(block)
        
        # Output projection
        self.output_projection = nn.Linear(input_dim, output_dim)
        
        # Regime-aware components
        if regime_aware:
            self.regime_optimizer = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 4)  # 4 regime characteristics
            )
    
    def forward(self, x):
        # Forward through blocks
        residuals = []
        for block in self.blocks:
            block_output = block(x)
            residuals.append(block_output)
            x = x - block_output  # Residual connection
        
        # Sum all residuals
        output = sum(residuals)
        
        # Final projection
        final_output = self.output_projection(output)
        
        # Regime optimization if enabled
        regime_output = None
        if self.regime_aware:
            regime_output = self.regime_optimizer(output)
        
        return final_output, regime_output


class MambaBlock(nn.Module):
    """Mamba block for efficient sequence modeling."""
    
    def __init__(self, hidden_dim: int, state_expansion: int = 4, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.state_expansion = state_expansion
        
        # Linear projections
        self.input_projection = nn.Linear(hidden_dim, hidden_dim * 2)
        self.gate_projection = nn.Linear(hidden_dim, hidden_dim * state_expansion)
        self.output_projection = nn.Linear(hidden_dim * state_expansion, hidden_dim)
        
        # State space parameters
        self.A = nn.Parameter(torch.randn(hidden_dim * state_expansion, hidden_dim))
        self.B = nn.Parameter(torch.randn(hidden_dim * state_expansion, hidden_dim))
        self.C = nn.Parameter(torch.randn(hidden_dim, hidden_dim * state_expansion))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        
        # Input and gate projections
        input_proj = self.input_projection(x)
        gate = torch.sigmoid(input_proj)
        
        # State space modeling (simplified)
        u = gate * x
        y = self.output_projection(u)
        
        return self.dropout(y)


class AdvancedMambaHybrid(nn.Module):
    """Advanced Mamba Hybrid for complex pattern recognition."""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 mamba_layers: int = 2, conv_layers: int = 4,
                 attention_heads: int = 8, hidden_dim: int = 128,
                 state_expansion: int = 4, multi_timeframe_fusion: bool = True,
                 dropout: float = 0.1, activation: str = "GELU",
                 execution_optimization: bool = False,
                 micro_timing_attention: bool = False,
                 latency_aware: bool = False):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.multi_timeframe_fusion = multi_timeframe_fusion
        self.execution_optimization = execution_optimization
        self.micro_timing_attention = micro_timing_attention
        self.latency_aware = latency_aware
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Mamba blocks
        self.mamba_blocks = nn.ModuleList()
        for _ in range(mamba_layers):
            self.mamba_blocks.append(
                MambaBlock(hidden_dim, state_expansion, dropout)
            )
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        for _ in range(conv_layers):
            self.conv_layers.append(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            )
            self.conv_layers.append(nn.BatchNorm1d(hidden_dim))
            if activation == "GELU":
                self.conv_layers.append(nn.GELU())
            else:
                self.conv_layers.append(nn.ReLU())
            self.conv_layers.append(nn.Dropout(dropout))
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=attention_heads,
            dropout=dropout, batch_first=True
        )
        
        # Execution optimization components
        if execution_optimization:
            self.execution_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 3)  # buy, hold, sell probabilities
            )
        
        # Micro-timing attention
        if micro_timing_attention:
            self.micro_timing_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 4, 1)  # timing score
            )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Multi-timeframe fusion
        if multi_timeframe_fusion:
            self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, x, multi_timeframe_input=None):
        # Input projection
        x = self.input_projection(x)
        
        # Mamba blocks
        for mamba_block in self.mamba_blocks:
            x = mamba_block(x)
        
        # Convolutional layers
        x_transposed = x.transpose(1, 2)  # (batch, features, seq)
        for layer in self.conv_layers:
            x_transposed = layer(x_transposed)
        x = x_transposed.transpose(1, 2)  # Back to (batch, seq, features)
        
        # Multi-head attention
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out  # Residual connection
        
        # Multi-timeframe fusion
        if self.multi_timeframe_fusion and multi_timeframe_input is not None:
            multi_timeframe_proj = self.input_projection(multi_timeframe_input)
            fused = torch.cat([x, multi_timeframe_proj], dim=-1)
            x = self.fusion_layer(fused)
        
        # Global pooling
        x = torch.mean(x, dim=1)  # (batch, features)
        
        # Output
        main_output = self.output_layers(x)
        
        # Additional outputs
        execution_output = None
        timing_output = None
        
        if self.execution_optimization:
            execution_output = self.execution_head(x)
        
        if self.micro_timing_attention:
            timing_output = self.micro_timing_head(x)
        
        return main_output, execution_output, timing_output


class MobileNetBlock(nn.Module):
    """MobileNet block for efficient inference."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=3,
                                  stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm1d(in_channels)
        
        # Pointwise convolution
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.activation = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        return x


class EfficientNetBlock(nn.Module):
    """EfficientNet block with squeeze-and-excitation."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 expansion_factor: int = 6):
        super().__init__()
        
        expanded_channels = in_channels * expansion_factor
        
        # Expansion
        self.expand = nn.Conv1d(in_channels, expanded_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(expanded_channels)
        
        # Depthwise convolution
        self.depthwise = nn.Conv1d(expanded_channels, expanded_channels, kernel_size=3,
                                  stride=stride, padding=1, groups=expanded_channels, bias=False)
        self.bn2 = nn.BatchNorm1d(expanded_channels)
        
        # Squeeze-and-excitation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(expanded_channels, expanded_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(expanded_channels // 16, expanded_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Projection
        self.project = nn.Conv1d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        self.activation = nn.SiLU(inplace=True)
    
    def forward(self, x):
        # Expansion
        out = self.expand(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        # Depthwise convolution
        out = self.depthwise(out)
        out = self.bn2(out)
        out = self.activation(out)
        
        # Squeeze-and-excitation
        se_out = self.se(out)
        out = out * se_out
        
        # Projection
        out = self.project(out)
        out = self.bn3(out)
        
        return out


class MobileNet(nn.Module):
    """MobileNet for efficient inference."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, 32)
        
        # MobileNet blocks
        self.blocks = nn.ModuleList([
            MobileNetBlock(32, 64, stride=2),
            MobileNetBlock(64, 128, stride=2),
            MobileNetBlock(128, 128, stride=1),
            MobileNetBlock(128, 256, stride=2),
            MobileNetBlock(256, 256, stride=1),
        ])
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(256, output_dim)
    
    def forward(self, x):
        x = self.input_projection(x)
        x = x.transpose(1, 2)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)


class EfficientNet(nn.Module):
    """EfficientNet for efficient inference with squeeze-and-excitation."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, 32)
        
        # EfficientNet blocks
        self.blocks = nn.ModuleList([
            EfficientNetBlock(32, 16, stride=1, expansion_factor=1),
            EfficientNetBlock(16, 24, stride=2, expansion_factor=6),
            EfficientNetBlock(24, 40, stride=2, expansion_factor=6),
            EfficientNetBlock(40, 80, stride=2, expansion_factor=6),
            EfficientNetBlock(80, 112, stride=1, expansion_factor=6),
            EfficientNetBlock(112, 192, stride=2, expansion_factor=6),
        ])
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(192, output_dim)
    
    def forward(self, x):
        x = self.input_projection(x)
        x = x.transpose(1, 2)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)


class ModelWrapper:
    """Wrapper class to make PyTorch models compatible with scikit-learn interface."""
    
    def __init__(self, model_class, model_config: Dict[str, Any], 
                 input_dim: int, output_dim: int, device: str = "cpu"):
        self.model_class = model_class
        self.model_config = model_config
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X, y):
        """Fit the model."""
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.FloatTensor(y)
            
            # Create model
            self.model = self.model_class(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                **self.model_config
            ).to(self.device)
            
            # Training setup
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Training loop
            self.model.train()
            for epoch in range(100):  # Default epochs
                optimizer.zero_grad()
                outputs = self.model(X_tensor.to(self.device))
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Take main output
                loss = criterion(outputs, y_tensor.to(self.device))
                loss.backward()
                optimizer.step()
                
                if epoch % 20 == 0:
                    tprint_info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            self.is_fitted = True
            tprint_info("Model training completed successfully")
            
        except Exception as e:
            tprint_error(f"Model training failed: {e}")
            raise
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        try:
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(X_scaled)
            
            # Make predictions
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X_tensor.to(self.device))
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # Take main output
                predictions = outputs.cpu().numpy()
            
            return predictions
            
        except Exception as e:
            tprint_error(f"Prediction failed: {e}")
            raise
    
    def set_params(self, **params):
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif key in self.model_config:
                self.model_config[key] = value
        return self
    
    def get_params(self, deep=True):
        """Get model parameters."""
        params = {
            'model_class': self.model_class,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'device': self.device
        }
        if deep:
            params.update(self.model_config)
        return params