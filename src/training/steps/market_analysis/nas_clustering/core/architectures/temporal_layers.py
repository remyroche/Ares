"""
Temporal Layers for Regime Detection

This module provides specialized temporal layers optimized for financial time series
regime detection, including temporal convolutions, LSTM/GRU variants, and attention mechanisms.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

# Neural network imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

logger = logging.getLogger(__name__)


class TemporalConvolutionLayer(nn.Module):
    """Temporal convolution layer optimized for regime detection."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, 
                 padding: int = 1, dilation: int = 1,
                 dropout: float = 0.0, activation: str = 'relu'):
        """Initialize temporal convolution layer."""
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        if TORCH_AVAILABLE:
            # Convolution layer
            self.conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation
            )
            
            # Batch normalization
            self.batch_norm = nn.BatchNorm1d(out_channels)
            
            # Dropout
            self.dropout = nn.Dropout(dropout) if dropout > 0 else None
            
            # Activation
            if activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'leaky_relu':
                self.activation = nn.LeakyReLU()
            elif activation == 'tanh':
                self.activation = nn.Tanh()
            elif activation == 'gelu':
                self.activation = nn.GELU()
            else:
                self.activation = nn.ReLU()  # Default
        else:
            self.conv = None
            self.batch_norm = None
            self.dropout = None
            self.activation = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through temporal convolution layer."""
        try:
            if not TORCH_AVAILABLE or self.conv is None:
                return x  # Return input unchanged if PyTorch not available
            
            # Convolution
            x = self.conv(x)
            
            # Batch normalization
            x = self.batch_norm(x)
            
            # Activation
            x = self.activation(x)
            
            # Dropout
            if self.dropout is not None:
                x = self.dropout(x)
            
            return x
            
        except Exception as e:
            logger.warning(f"Temporal convolution forward pass failed: {e}")
            return x


class RegimeLSTMLayer(nn.Module):
    """LSTM layer specialized for regime detection."""
    
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, dropout: float = 0.0,
                 bidirectional: bool = False, batch_first: bool = True):
        """Initialize regime LSTM layer."""
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        
        if TORCH_AVAILABLE:
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=batch_first
            )
            
            self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        else:
            self.lstm = None
            self.dropout_layer = None
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through regime LSTM layer."""
        try:
            if not TORCH_AVAILABLE or self.lstm is None:
                # Return simplified output if PyTorch not available
                batch_size = x.size(0) if x.dim() > 1 else 1
                seq_len = x.size(1) if x.dim() > 1 else x.size(0)
                output_size = self.hidden_size * (2 if self.bidirectional else 1)
                
                output = torch.zeros(batch_size, seq_len, output_size)
                hidden = (torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size),
                         torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size))
                
                return output, hidden
            
            # LSTM forward pass
            lstm_out, hidden = self.lstm(x, hidden)
            
            # Apply dropout
            if self.dropout_layer is not None:
                lstm_out = self.dropout_layer(lstm_out)
            
            return lstm_out, hidden
            
        except Exception as e:
            logger.warning(f"Regime LSTM forward pass failed: {e}")
            # Return input as output in case of error
            return x, None


class RegimeGRULayer(nn.Module):
    """GRU layer specialized for regime detection."""
    
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int = 1, dropout: float = 0.0,
                 bidirectional: bool = False, batch_first: bool = True):
        """Initialize regime GRU layer."""
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        
        if TORCH_AVAILABLE:
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=batch_first
            )
            
            self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        else:
            self.gru = None
            self.dropout_layer = None
    
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through regime GRU layer."""
        try:
            if not TORCH_AVAILABLE or self.gru is None:
                # Return simplified output if PyTorch not available
                batch_size = x.size(0) if x.dim() > 1 else 1
                seq_len = x.size(1) if x.dim() > 1 else x.size(0)
                output_size = self.hidden_size * (2 if self.bidirectional else 1)
                
                output = torch.zeros(batch_size, seq_len, output_size)
                hidden = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size)
                
                return output, hidden
            
            # GRU forward pass
            gru_out, hidden = self.gru(x, hidden)
            
            # Apply dropout
            if self.dropout_layer is not None:
                gru_out = self.dropout_layer(gru_out)
            
            return gru_out, hidden
            
        except Exception as e:
            logger.warning(f"Regime GRU forward pass failed: {e}")
            # Return input as output in case of error
            return x, None


class MultiScaleTemporalLayer(nn.Module):
    """Multi-scale temporal layer for capturing different regime timescales."""
    
    def __init__(self, input_size: int, hidden_size: int,
                 scales: List[int] = [1, 2, 4], dropout: float = 0.0):
        """Initialize multi-scale temporal layer."""
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.scales = scales
        self.dropout = dropout
        
        if TORCH_AVAILABLE:
            # Create temporal layers for different scales
            self.temporal_layers = nn.ModuleList()
            for scale in scales:
                layer = TemporalConvolutionLayer(
                    in_channels=input_size,
                    out_channels=hidden_size // len(scales),
                    kernel_size=3,
                    dilation=scale,
                    dropout=dropout
                )
                self.temporal_layers.append(layer)
            
            # Fusion layer
            total_out_channels = (hidden_size // len(scales)) * len(scales)
            self.fusion_layer = nn.Conv1d(total_out_channels, hidden_size, kernel_size=1)
            
            self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        else:
            self.temporal_layers = None
            self.fusion_layer = None
            self.dropout_layer = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through multi-scale temporal layer."""
        try:
            if not TORCH_AVAILABLE or self.temporal_layers is None:
                return x  # Return input unchanged if PyTorch not available
            
            # Apply temporal layers at different scales
            scale_outputs = []
            for temporal_layer in self.temporal_layers:
                scale_out = temporal_layer(x)
                scale_outputs.append(scale_out)
            
            # Concatenate scale outputs
            multi_scale_out = torch.cat(scale_outputs, dim=1)
            
            # Fusion layer
            fused_out = self.fusion_layer(multi_scale_out)
            
            # Apply dropout
            if self.dropout_layer is not None:
                fused_out = self.dropout_layer(fused_out)
            
            return fused_out
            
        except Exception as e:
            logger.warning(f"Multi-scale temporal layer forward pass failed: {e}")
            return x


class TemporalAttentionLayer(nn.Module):
    """Temporal attention layer for regime detection."""
    
    def __init__(self, input_size: int, attention_size: int,
                 dropout: float = 0.0, temperature: float = 1.0):
        """Initialize temporal attention layer."""
        super().__init__()
        
        self.input_size = input_size
        self.attention_size = attention_size
        self.dropout = dropout
        self.temperature = temperature
        
        if TORCH_AVAILABLE:
            # Attention layers
            self.query_layer = nn.Linear(input_size, attention_size)
            self.key_layer = nn.Linear(input_size, attention_size)
            self.value_layer = nn.Linear(input_size, input_size)
            
            # Dropout
            self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
            
            # Output projection
            self.output_projection = nn.Linear(input_size, input_size)
        else:
            self.query_layer = None
            self.key_layer = None
            self.value_layer = None
            self.dropout_layer = None
            self.output_projection = None
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through temporal attention layer."""
        try:
            if not TORCH_AVAILABLE or self.query_layer is None:
                return x  # Return input unchanged if PyTorch not available
            
            batch_size, seq_len, input_size = x.size()
            
            # Compute queries, keys, and values
            queries = self.query_layer(x)  # (batch, seq_len, attention_size)
            keys = self.key_layer(x)       # (batch, seq_len, attention_size)
            values = self.value_layer(x)   # (batch, seq_len, input_size)
            
            # Compute attention scores
            attention_scores = torch.bmm(queries, keys.transpose(1, 2))  # (batch, seq_len, seq_len)
            attention_scores = attention_scores / (self.attention_size ** 0.5)  # Scale by sqrt(d_k)
            attention_scores = attention_scores / self.temperature  # Apply temperature
            
            # Apply mask if provided
            if mask is not None:
                attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
            # Apply softmax
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            # Apply dropout to attention weights
            if self.dropout_layer is not None:
                attention_weights = self.dropout_layer(attention_weights)
            
            # Apply attention to values
            attended_values = torch.bmm(attention_weights, values)  # (batch, seq_len, input_size)
            
            # Output projection
            output = self.output_projection(attended_values)
            
            # Residual connection
            output = output + x
            
            return output
            
        except Exception as e:
            logger.warning(f"Temporal attention layer forward pass failed: {e}")
            return x


class RegimeTransitionLayer(nn.Module):
    """Layer for modeling regime transitions."""
    
    def __init__(self, input_size: int, hidden_size: int,
                 num_regimes: int, dropout: float = 0.0):
        """Initialize regime transition layer."""
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_regimes = num_regimes
        self.dropout = dropout
        
        if TORCH_AVAILABLE:
            # Transition modeling layers
            self.transition_lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=0.0
            )
            
            # Regime prediction
            self.regime_classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_regimes)
            )
            
            # Transition probability matrix
            self.transition_matrix = nn.Parameter(
                torch.randn(num_regimes, num_regimes)
            )
            
            self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        else:
            self.transition_lstm = None
            self.regime_classifier = None
            self.transition_matrix = None
            self.dropout_layer = None
    
    def forward(self, x: torch.Tensor, previous_regime: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through regime transition layer."""
        try:
            if not TORCH_AVAILABLE or self.transition_lstm is None:
                # Return simplified output if PyTorch not available
                batch_size = x.size(0)
                seq_len = x.size(1)
                
                regime_logits = torch.randn(batch_size, seq_len, self.num_regimes)
                transition_probs = torch.softmax(self.transition_matrix, dim=-1) if self.transition_matrix is not None else torch.ones(self.num_regimes, self.num_regimes) / self.num_regimes
                
                return {
                    'regime_logits': regime_logits,
                    'regime_probabilities': torch.softmax(regime_logits, dim=-1),
                    'transition_probabilities': transition_probs
                }
            
            # LSTM forward pass
            lstm_out, _ = self.transition_lstm(x)
            
            # Apply dropout
            if self.dropout_layer is not None:
                lstm_out = self.dropout_layer(lstm_out)
            
            # Regime classification
            regime_logits = self.regime_classifier(lstm_out)
            regime_probabilities = F.softmax(regime_logits, dim=-1)
            
            # Compute transition probabilities
            if previous_regime is not None:
                # Use previous regime to compute transition probabilities
                transition_probs = F.softmax(self.transition_matrix, dim=-1)
                # Select transition probabilities based on previous regime
                batch_size = previous_regime.size(0)
                transition_probs_selected = transition_probs[previous_regime.long()]
            else:
                # Use default transition probabilities
                transition_probs_selected = F.softmax(self.transition_matrix, dim=-1)
            
            return {
                'regime_logits': regime_logits,
                'regime_probabilities': regime_probabilities,
                'transition_probabilities': transition_probs_selected,
                'lstm_output': lstm_out
            }
            
        except Exception as e:
            logger.warning(f"Regime transition layer forward pass failed: {e}")
            # Return simplified output in case of error
            batch_size = x.size(0)
            seq_len = x.size(1)
            regime_logits = torch.zeros(batch_size, seq_len, self.num_regimes)
            regime_probabilities = torch.ones(batch_size, seq_len, self.num_regimes) / self.num_regimes
            
            return {
                'regime_logits': regime_logits,
                'regime_probabilities': regime_probabilities,
                'transition_probabilities': torch.ones(self.num_regimes, self.num_regimes) / self.num_regimes
            }


class AdaptiveTemporalLayer(nn.Module):
    """Adaptive temporal layer that adjusts based on regime characteristics."""
    
    def __init__(self, input_size: int, hidden_size: int,
                 num_regimes: int, dropout: float = 0.0):
        """Initialize adaptive temporal layer."""
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_regimes = num_regimes
        self.dropout = dropout
        
        if TORCH_AVAILABLE:
            # Regime-specific temporal layers
            self.regime_layers = nn.ModuleList()
            for _ in range(num_regimes):
                regime_layer = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size)
                )
                self.regime_layers.append(regime_layer)
            
            # Regime selector
            self.regime_selector = nn.Sequential(
                nn.Linear(input_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, num_regimes)
            )
            
            # Output fusion
            self.output_fusion = nn.Linear(hidden_size, hidden_size)
            
            self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        else:
            self.regime_layers = None
            self.regime_selector = None
            self.output_fusion = None
            self.dropout_layer = None
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through adaptive temporal layer."""
        try:
            if not TORCH_AVAILABLE or self.regime_layers is None:
                # Return simplified output if PyTorch not available
                batch_size = x.size(0)
                seq_len = x.size(1)
                
                regime_weights = torch.ones(batch_size, seq_len, self.num_regimes) / self.num_regimes
                output = x
                
                return {
                    'output': output,
                    'regime_weights': regime_weights
                }
            
            # Compute regime weights
            regime_logits = self.regime_selector(x)
            regime_weights = F.softmax(regime_logits, dim=-1)
            
            # Apply regime-specific processing
            regime_outputs = []
            for i, regime_layer in enumerate(self.regime_layers):
                regime_out = regime_layer(x)
                regime_outputs.append(regime_out)
            
            # Weight regime outputs
            regime_outputs_tensor = torch.stack(regime_outputs, dim=-1)  # (batch, seq, hidden, num_regimes)
            regime_weights_expanded = regime_weights.unsqueeze(-2)  # (batch, seq, 1, num_regimes)
            weighted_output = torch.sum(regime_outputs_tensor * regime_weights_expanded, dim=-1)  # (batch, seq, hidden)
            
            # Output fusion
            fused_output = self.output_fusion(weighted_output)
            
            # Apply dropout
            if self.dropout_layer is not None:
                fused_output = self.dropout_layer(fused_output)
            
            return {
                'output': fused_output,
                'regime_weights': regime_weights,
                'regime_outputs': regime_outputs
            }
            
        except Exception as e:
            logger.warning(f"Adaptive temporal layer forward pass failed: {e}")
            # Return input as output in case of error
            batch_size = x.size(0)
            seq_len = x.size(1)
            regime_weights = torch.ones(batch_size, seq_len, self.num_regimes) / self.num_regimes
            
            return {
                'output': x,
                'regime_weights': regime_weights
            }