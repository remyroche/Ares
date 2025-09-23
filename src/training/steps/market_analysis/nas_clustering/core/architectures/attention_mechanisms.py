"""
Attention Mechanisms for Regime Detection

This module provides specialized attention mechanisms optimized for financial time series
regime detection, including self-attention, cross-attention, and multi-head attention.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import math

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


class RegimeAttention(nn.Module):
    """Base attention mechanism for regime detection."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8,
                 dropout: float = 0.0, bias: bool = True):
        """Initialize regime attention."""
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        
        if TORCH_AVAILABLE:
            # Linear layers for Q, K, V
            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            
            # Output projection
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            
            # Dropout
            self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
            
            # Scale factor
            self.scale_factor = 1.0 / math.sqrt(self.head_dim)
        else:
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
            self.out_proj = None
            self.dropout_layer = None
            self.scale_factor = 1.0
    
    def forward(self, query: torch.Tensor, key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None, 
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through regime attention."""
        try:
            if not TORCH_AVAILABLE or self.q_proj is None:
                # Return simplified output if PyTorch not available
                batch_size, seq_len = query.size(0), query.size(1)
                attention_weights = torch.ones(batch_size, self.num_heads, seq_len, seq_len) / seq_len
                output = query
                return output, attention_weights
            
            # Use query as key and value if not provided (self-attention)
            if key is None:
                key = query
            if value is None:
                value = query
            
            batch_size, seq_len = query.size(0), query.size(1)
            
            # Project to Q, K, V
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
            
            # Reshape for multi-head attention
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, seq, head_dim)
            k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, key_seq, head_dim)
            v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, value_seq, head_dim)
            
            # Compute attention scores
            attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale_factor  # (batch, heads, seq, key_seq)
            
            # Apply attention mask
            if attn_mask is not None:
                attention_scores = attention_scores.masked_fill(attn_mask == 0, -1e9)
            
            # Apply key padding mask
            if key_padding_mask is not None:
                attention_scores = attention_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), -1e9)
            
            # Apply softmax
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            # Apply dropout
            if self.dropout_layer is not None:
                attention_weights = self.dropout_layer(attention_weights)
            
            # Apply attention to values
            attended_values = torch.matmul(attention_weights, v)  # (batch, heads, seq, head_dim)
            
            # Reshape back
            attended_values = attended_values.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
            
            # Output projection
            output = self.out_proj(attended_values)
            
            # Average attention weights across heads for visualization
            avg_attention_weights = attention_weights.mean(dim=1)
            
            return output, avg_attention_weights
            
        except Exception as e:
            logger.warning(f"Regime attention forward pass failed: {e}")
            return query, torch.ones(query.size(0), query.size(1), query.size(1)) / query.size(1)


class MultiHeadRegimeAttention(nn.Module):
    """Multi-head attention mechanism for regime detection."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8,
                 dropout: float = 0.0, bias: bool = True):
        """Initialize multi-head regime attention."""
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        if TORCH_AVAILABLE:
            # Multi-head attention layer
            self.attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
                batch_first=True
            )
            
            # Layer normalization
            self.layer_norm = nn.LayerNorm(embed_dim)
            
            # Dropout for residual connection
            self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        else:
            self.attention = None
            self.layer_norm = None
            self.dropout_layer = None
    
    def forward(self, query: torch.Tensor, key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through multi-head regime attention."""
        try:
            if not TORCH_AVAILABLE or self.attention is None:
                # Return simplified output if PyTorch not available
                attention_weights = torch.ones(query.size(0), query.size(1), query.size(1)) / query.size(1)
                return query, attention_weights
            
            # Use query as key and value if not provided (self-attention)
            if key is None:
                key = query
            if value is None:
                value = query
            
            # Store input for residual connection
            residual = query
            
            # Multi-head attention
            attn_output, attention_weights = self.attention(
                query=query,
                key=key,
                value=value,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                average_attn_weights=False
            )
            
            # Apply dropout
            if self.dropout_layer is not None:
                attn_output = self.dropout_layer(attn_output)
            
            # Residual connection
            output = attn_output + residual
            
            # Layer normalization
            output = self.layer_norm(output)
            
            # Average attention weights across heads
            if attention_weights.dim() == 4:  # (batch, heads, seq, seq)
                avg_attention_weights = attention_weights.mean(dim=1)
            else:  # (batch, seq, seq)
                avg_attention_weights = attention_weights
            
            return output, avg_attention_weights
            
        except Exception as e:
            logger.warning(f"Multi-head regime attention forward pass failed: {e}")
            return query, torch.ones(query.size(0), query.size(1), query.size(1)) / query.size(1)


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for time series regime detection."""
    
    def __init__(self, input_dim: int, attention_dim: int,
                 num_heads: int = 8, dropout: float = 0.0):
        """Initialize temporal attention."""
        super().__init__()
        
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        if TORCH_AVAILABLE:
            # Positional encoding for temporal information
            self.pos_encoding = nn.Parameter(torch.randn(1, 1000, attention_dim))
            
            # Multi-head attention
            self.attention = MultiHeadRegimeAttention(
                embed_dim=attention_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            
            # Input projection
            self.input_proj = nn.Linear(input_dim, attention_dim)
            
            # Output projection
            self.output_proj = nn.Linear(attention_dim, input_dim)
            
            # Temporal position embedding
            self.temporal_embedding = nn.Embedding(1000, attention_dim)
        else:
            self.pos_encoding = None
            self.attention = None
            self.input_proj = None
            self.output_proj = None
            self.temporal_embedding = None
    
    def forward(self, x: torch.Tensor, 
                temporal_positions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through temporal attention."""
        try:
            if not TORCH_AVAILABLE or self.attention is None:
                # Return simplified output if PyTorch not available
                attention_weights = torch.ones(x.size(0), x.size(1), x.size(1)) / x.size(1)
                return x, attention_weights
            
            batch_size, seq_len, input_dim = x.size()
            
            # Project input
            projected_input = self.input_proj(x)
            
            # Add positional encoding
            if temporal_positions is not None:
                # Use provided temporal positions
                pos_emb = self.temporal_embedding(temporal_positions)
                projected_input = projected_input + pos_emb
            else:
                # Use learned positional encoding
                pos_emb = self.pos_encoding[:, :seq_len, :]
                projected_input = projected_input + pos_emb
            
            # Apply temporal attention
            attended_output, attention_weights = self.attention(projected_input)
            
            # Project output back to input dimension
            output = self.output_proj(attended_output)
            
            return output, attention_weights
            
        except Exception as e:
            logger.warning(f"Temporal attention forward pass failed: {e}")
            return x, torch.ones(x.size(0), x.size(1), x.size(1)) / x.size(1)


class CrossRegimeAttention(nn.Module):
    """Cross-attention mechanism between different regime representations."""
    
    def __init__(self, query_dim: int, key_dim: int, value_dim: int,
                 attention_dim: int, num_heads: int = 8, dropout: float = 0.0):
        """Initialize cross-regime attention."""
        super().__init__()
        
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        if TORCH_AVAILABLE:
            # Projections for different dimensions
            self.query_proj = nn.Linear(query_dim, attention_dim)
            self.key_proj = nn.Linear(key_dim, attention_dim)
            self.value_proj = nn.Linear(value_dim, attention_dim)
            
            # Multi-head attention
            self.attention = nn.MultiheadAttention(
                embed_dim=attention_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            
            # Output projection
            self.output_proj = nn.Linear(attention_dim, query_dim)
            
            # Layer normalization
            self.layer_norm = nn.LayerNorm(query_dim)
            
            self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        else:
            self.query_proj = None
            self.key_proj = None
            self.value_proj = None
            self.attention = None
            self.output_proj = None
            self.layer_norm = None
            self.dropout_layer = None
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through cross-regime attention."""
        try:
            if not TORCH_AVAILABLE or self.attention is None:
                # Return simplified output if PyTorch not available
                attention_weights = torch.ones(query.size(0), query.size(1), key.size(1)) / key.size(1)
                return query, attention_weights
            
            # Store query for residual connection
            residual = query
            
            # Project to attention dimension
            q = self.query_proj(query)
            k = self.key_proj(key)
            v = self.value_proj(value)
            
            # Apply cross-attention
            attn_output, attention_weights = self.attention(
                query=q,
                key=k,
                value=v,
                average_attn_weights=False
            )
            
            # Project back to query dimension
            output = self.output_proj(attn_output)
            
            # Apply dropout
            if self.dropout_layer is not None:
                output = self.dropout_layer(output)
            
            # Residual connection
            output = output + residual
            
            # Layer normalization
            output = self.layer_norm(output)
            
            # Average attention weights across heads
            if attention_weights.dim() == 4:  # (batch, heads, seq, seq)
                avg_attention_weights = attention_weights.mean(dim=1)
            else:  # (batch, seq, seq)
                avg_attention_weights = attention_weights
            
            return output, avg_attention_weights
            
        except Exception as e:
            logger.warning(f"Cross-regime attention forward pass failed: {e}")
            return query, torch.ones(query.size(0), query.size(1), key.size(1)) / key.size(1)


class SelfRegimeAttention(nn.Module):
    """Self-attention mechanism specifically designed for regime detection."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8,
                 dropout: float = 0.0, use_positional_encoding: bool = True):
        """Initialize self-regime attention."""
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_positional_encoding = use_positional_encoding
        
        if TORCH_AVAILABLE:
            # Multi-head self-attention
            self.self_attention = MultiHeadRegimeAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            
            # Positional encoding
            if use_positional_encoding:
                self.pos_encoding = nn.Parameter(torch.randn(1, 1000, embed_dim))
            else:
                self.pos_encoding = None
            
            # Feed-forward network
            self.feed_forward = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim)
            )
            
            # Layer normalization
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            
            # Dropout for residual connections
            self.dropout1 = nn.Dropout(dropout) if dropout > 0 else None
            self.dropout2 = nn.Dropout(dropout) if dropout > 0 else None
        else:
            self.self_attention = None
            self.pos_encoding = None
            self.feed_forward = None
            self.norm1 = None
            self.norm2 = None
            self.dropout1 = None
            self.dropout2 = None
    
    def forward(self, x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through self-regime attention."""
        try:
            if not TORCH_AVAILABLE or self.self_attention is None:
                # Return simplified output if PyTorch not available
                attention_weights = torch.ones(x.size(0), x.size(1), x.size(1)) / x.size(1)
                return x, attention_weights
            
            # Add positional encoding
            if self.pos_encoding is not None:
                seq_len = x.size(1)
                pos_emb = self.pos_encoding[:, :seq_len, :]
                x = x + pos_emb
            
            # Self-attention with residual connection
            residual1 = x
            attn_output, attention_weights = self.self_attention(x, attn_mask=attention_mask)
            if self.dropout1 is not None:
                attn_output = self.dropout1(attn_output)
            x = self.norm1(attn_output + residual1)
            
            # Feed-forward with residual connection
            residual2 = x
            ff_output = self.feed_forward(x)
            if self.dropout2 is not None:
                ff_output = self.dropout2(ff_output)
            x = self.norm2(ff_output + residual2)
            
            return x, attention_weights
            
        except Exception as e:
            logger.warning(f"Self-regime attention forward pass failed: {e}")
            return x, torch.ones(x.size(0), x.size(1), x.size(1)) / x.size(1)


class RegimeTransitionAttention(nn.Module):
    """Attention mechanism for modeling regime transitions."""
    
    def __init__(self, embed_dim: int, num_regimes: int,
                 num_heads: int = 8, dropout: float = 0.0):
        """Initialize regime transition attention."""
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_regimes = num_regimes
        self.num_heads = num_heads
        self.dropout = dropout
        
        if TORCH_AVAILABLE:
            # Regime embedding
            self.regime_embedding = nn.Embedding(num_regimes, embed_dim)
            
            # Transition attention
            self.transition_attention = MultiHeadRegimeAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            
            # Transition probability prediction
            self.transition_predictor = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, num_regimes)
            )
            
            # Layer normalization
            self.layer_norm = nn.LayerNorm(embed_dim)
            
            self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
        else:
            self.regime_embedding = None
            self.transition_attention = None
            self.transition_predictor = None
            self.layer_norm = None
            self.dropout_layer = None
    
    def forward(self, x: torch.Tensor, current_regime: torch.Tensor,
                previous_regimes: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through regime transition attention."""
        try:
            if not TORCH_AVAILABLE or self.transition_attention is None:
                # Return simplified output if PyTorch not available
                batch_size, seq_len = x.size(0), x.size(1)
                attention_weights = torch.ones(batch_size, seq_len, seq_len) / seq_len
                transition_logits = torch.randn(batch_size, seq_len, self.num_regimes)
                
                return {
                    'output': x,
                    'attention_weights': attention_weights,
                    'transition_logits': transition_logits,
                    'transition_probabilities': torch.softmax(transition_logits, dim=-1)
                }
            
            # Get regime embeddings
            current_regime_emb = self.regime_embedding(current_regime)  # (batch, seq, embed_dim)
            
            # Apply transition attention
            attended_output, attention_weights = self.transition_attention(
                query=x,
                key=current_regime_emb,
                value=current_regime_emb
            )
            
            # Apply dropout
            if self.dropout_layer is not None:
                attended_output = self.dropout_layer(attended_output)
            
            # Layer normalization
            attended_output = self.layer_norm(attended_output)
            
            # Predict transition probabilities
            transition_logits = self.transition_predictor(attended_output)
            transition_probabilities = F.softmax(transition_logits, dim=-1)
            
            return {
                'output': attended_output,
                'attention_weights': attention_weights,
                'transition_logits': transition_logits,
                'transition_probabilities': transition_probabilities
            }
            
        except Exception as e:
            logger.warning(f"Regime transition attention forward pass failed: {e}")
            # Return simplified output in case of error
            batch_size, seq_len = x.size(0), x.size(1)
            attention_weights = torch.ones(batch_size, seq_len, seq_len) / seq_len
            transition_logits = torch.zeros(batch_size, seq_len, self.num_regimes)
            transition_probabilities = torch.ones(batch_size, seq_len, self.num_regimes) / self.num_regimes
            
            return {
                'output': x,
                'attention_weights': attention_weights,
                'transition_logits': transition_logits,
                'transition_probabilities': transition_probabilities
            }