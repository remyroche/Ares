"""
Advanced Neural Architectures for Regime Detection

This module implements cutting-edge neural architectures specifically designed for regime detection:
- Transformer-based regime detection with self-attention mechanisms
- Graph Neural Networks for market relationship modeling
- Temporal Convolutional Networks for time series patterns
- Hybrid architectures combining multiple approaches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)


class ArchitectureType(Enum):
    """Available advanced architecture types."""
    TRANSFORMER_REGIME = "transformer_regime"
    GRAPH_NEURAL_NETWORK = "graph_neural_network"
    TEMPORAL_CONVOLUTIONAL = "temporal_convolutional"
    HYBRID_TRANSFORMER_GNN = "hybrid_transformer_gnn"
    MULTI_SCALE_TRANSFORMER = "multi_scale_transformer"
    ADAPTIVE_TRANSFORMER = "adaptive_transformer"


@dataclass
class AdvancedArchitectureConfig:
    """Configuration for advanced neural architectures."""
    architecture_type: ArchitectureType = ArchitectureType.TRANSFORMER_REGIME
    input_dim: int = 64
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 6
    num_regimes: int = 8
    dropout: float = 0.1
    use_positional_encoding: bool = True
    use_regime_attention: bool = True
    use_temporal_attention: bool = True
    max_sequence_length: int = 1000
    enable_adaptive_attention: bool = True
    enable_multi_scale_fusion: bool = True


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer architectures."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class RegimeAttention(nn.Module):
    """Specialized attention mechanism for regime detection."""
    
    def __init__(self, d_model: int, num_heads: int, num_regimes: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_regimes = num_regimes
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.regime_embedding = nn.Embedding(num_regimes, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, regime_labels=None):
        batch_size, seq_len, d_model = x.size()
        
        # Create regime-aware queries, keys, values
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        # Add regime information if available
        if regime_labels is not None:
            regime_emb = self.regime_embedding(regime_labels)
            q = q + regime_emb
            k = k + regime_emb
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection
        output = self.out_linear(context)
        output = self.layer_norm(output + x)
        
        return output, attention_weights


class TransformerRegimeDetector(nn.Module):
    """Transformer-based regime detection with specialized attention mechanisms."""
    
    def __init__(self, config: AdvancedArchitectureConfig):
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        self.pos_encoding = PositionalEncoding(config.hidden_dim, config.max_sequence_length)
        
        # Regime-aware attention layers
        self.regime_attention_layers = nn.ModuleList([
            RegimeAttention(config.hidden_dim, config.num_heads, config.num_regimes, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # Temporal attention for time series patterns
        self.temporal_attention = nn.MultiheadAttention(
            config.hidden_dim, config.num_heads, dropout=config.dropout, batch_first=True
        )
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim * 4, config.hidden_dim)
            )
            for _ in range(config.num_layers)
        ])
        
        # Regime classification head
        self.regime_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_regimes)
        )
        
        # Regime transition prediction
        self.transition_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_regimes)
        )
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim) for _ in range(config.num_layers)
        ])
        
    def forward(self, x, regime_labels=None):
        batch_size, seq_len, input_dim = x.size()
        
        # Input projection and positional encoding
        x = self.input_projection(x)
        if self.config.use_positional_encoding:
            x = x.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)
            x = self.pos_encoding(x)
            x = x.transpose(0, 1)  # (batch_size, seq_len, hidden_dim)
        
        # Regime-aware transformer layers
        attention_weights = []
        for i, (regime_attn, ffn, layer_norm) in enumerate(
            zip(self.regime_attention_layers, self.ffn_layers, self.layer_norms)
        ):
            # Regime-aware attention
            if self.config.use_regime_attention:
                x_attn, attn_weights = regime_attn(x, regime_labels)
                attention_weights.append(attn_weights)
            else:
                x_attn, _ = self.temporal_attention(x, x, x)
                attention_weights.append(None)
            
            # Feed-forward network
            x_ffn = ffn(x_attn)
            
            # Residual connection and layer norm
            x = layer_norm(x_attn + x_ffn)
        
        # Regime classification
        regime_logits = self.regime_classifier(x)
        
        # Regime transition prediction (using consecutive hidden states)
        if seq_len > 1:
            transitions = []
            for i in range(seq_len - 1):
                transition_input = torch.cat([x[:, i], x[:, i + 1]], dim=-1)
                transition_logits = self.transition_predictor(transition_input)
                transitions.append(transition_logits)
            transition_logits = torch.stack(transitions, dim=1)
        else:
            transition_logits = None
        
        return {
            'regime_logits': regime_logits,
            'transition_logits': transition_logits,
            'hidden_states': x,
            'attention_weights': attention_weights
        }


class GraphNeuralNetworkRegimeDetector(nn.Module):
    """Graph Neural Network for market relationship modeling and regime detection."""
    
    def __init__(self, config: AdvancedArchitectureConfig):
        super().__init__()
        self.config = config
        
        # Node embedding
        self.node_embedding = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(config.hidden_dim, config.hidden_dim // config.num_heads, config.num_heads)
            for _ in range(config.num_layers)
        ])
        
        # Temporal graph convolution
        self.temporal_gcn = TemporalGraphConvolution(config.hidden_dim, config.hidden_dim)
        
        # Regime classification
        self.regime_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_regimes)
        )
        
    def forward(self, node_features, adjacency_matrix, timestamps=None):
        batch_size, num_nodes, seq_len, input_dim = node_features.size()
        
        # Node embedding
        node_emb = self.node_embedding(node_features.view(-1, input_dim))
        node_emb = node_emb.view(batch_size, num_nodes, seq_len, -1)
        
        # Process each time step
        regime_logits_list = []
        for t in range(seq_len):
            # Current time step node features
            current_nodes = node_emb[:, :, t, :]  # (batch_size, num_nodes, hidden_dim)
            
            # Graph attention layers
            x = current_nodes
            for gat_layer in self.gat_layers:
                x = gat_layer(x, adjacency_matrix)
            
            # Temporal graph convolution if timestamps available
            if timestamps is not None and t > 0:
                x = self.temporal_gcn(x, node_emb[:, :, t-1, :], timestamps[:, t])
            
            # Regime classification for this time step
            regime_logits = self.regime_classifier(x.mean(dim=1))  # Global pooling
            regime_logits_list.append(regime_logits)
        
        regime_logits = torch.stack(regime_logits_list, dim=1)  # (batch_size, seq_len, num_regimes)
        
        return {
            'regime_logits': regime_logits,
            'node_embeddings': node_emb
        }


class GraphAttentionLayer(nn.Module):
    """Graph Attention Network layer."""
    
    def __init__(self, in_features: int, out_features: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        
        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Parameter(torch.randn(num_heads, 2 * self.head_dim))
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, h, adj):
        batch_size, num_nodes, _ = h.size()
        
        # Linear transformation
        h_transformed = self.W(h)  # (batch_size, num_nodes, out_features)
        
        # Reshape for multi-head attention
        h_reshaped = h_transformed.view(batch_size, num_nodes, self.num_heads, self.head_dim)
        
        # Compute attention coefficients
        attention_scores = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj[i, j] > 0:  # Only consider connected nodes
                    # Concatenate node features
                    concat_features = torch.cat([
                        h_reshaped[:, i, :, :], 
                        h_reshaped[:, j, :, :]
                    ], dim=-1)  # (batch_size, num_heads, 2 * head_dim)
                    
                    # Compute attention score
                    score = torch.sum(self.a * concat_features, dim=-1)  # (batch_size, num_heads)
                    score = self.activation(score)
                    attention_scores.append(score)
        
        # Apply attention
        output = torch.zeros_like(h_transformed)
        attention_idx = 0
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj[i, j] > 0:
                    attention_weight = F.softmax(attention_scores[attention_idx], dim=0)
                    attention_weight = self.dropout(attention_weight)
                    
                    # Apply attention weight
                    weighted_features = h_reshaped[:, j, :, :] * attention_weight.unsqueeze(-1)
                    output[:, i, :] += weighted_features.view(batch_size, self.out_features)
                    attention_idx += 1
        
        return output


class TemporalGraphConvolution(nn.Module):
    """Temporal graph convolution for time series graph data."""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.temporal_conv = nn.Conv1d(in_features, out_features, kernel_size=3, padding=1)
        self.graph_conv = nn.Linear(in_features, out_features)
        
    def forward(self, current_nodes, previous_nodes, time_delta):
        # Temporal convolution
        temporal_features = self.temporal_conv(
            current_nodes.transpose(1, 2)  # (batch_size, features, num_nodes)
        ).transpose(1, 2)  # (batch_size, num_nodes, features)
        
        # Graph convolution with temporal weighting
        time_weight = torch.exp(-time_delta.unsqueeze(-1))  # Temporal decay
        graph_features = self.graph_conv(previous_nodes) * time_weight
        
        # Combine temporal and graph features
        output = temporal_features + graph_features
        
        return output


class TemporalConvolutionalRegimeDetector(nn.Module):
    """Temporal Convolutional Network for time series regime detection."""
    
    def __init__(self, config: AdvancedArchitectureConfig):
        super().__init__()
        self.config = config
        
        # Multi-scale temporal convolutions
        self.temporal_convs = nn.ModuleList([
            nn.Conv1d(config.input_dim, config.hidden_dim, kernel_size=k, padding=k//2)
            for k in [3, 5, 7, 9]  # Multiple temporal scales
        ])
        
        # Dilated convolutions for long-range dependencies
        self.dilated_convs = nn.ModuleList([
            nn.Conv1d(config.hidden_dim, config.hidden_dim, kernel_size=3, 
                     padding=2**i, dilation=2**i)
            for i in range(4)  # Different dilation rates
        ])
        
        # Attention mechanism for temporal patterns
        self.temporal_attention = nn.MultiheadAttention(
            config.hidden_dim, config.num_heads, dropout=config.dropout, batch_first=True
        )
        
        # Regime classification
        self.regime_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_regimes)
        )
        
    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        
        # Transpose for conv1d: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Multi-scale temporal convolutions
        conv_outputs = []
        for conv in self.temporal_convs:
            conv_out = F.relu(conv(x))
            conv_outputs.append(conv_out)
        
        # Concatenate multi-scale features
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, hidden_dim * 4, seq_len)
        
        # Reduce dimensions
        x = F.adaptive_avg_pool1d(x, seq_len)  # Ensure same length
        
        # Dilated convolutions
        for dilated_conv in self.dilated_convs:
            residual = x
            x = F.relu(dilated_conv(x))
            x = x + residual  # Residual connection
        
        # Transpose back: (batch_size, seq_len, hidden_dim)
        x = x.transpose(1, 2)
        
        # Temporal attention
        x_attn, _ = self.temporal_attention(x, x, x)
        x = x + x_attn  # Residual connection
        
        # Regime classification
        regime_logits = self.regime_classifier(x)
        
        return {
            'regime_logits': regime_logits,
            'temporal_features': x
        }


class HybridTransformerGNN(nn.Module):
    """Hybrid architecture combining Transformer and Graph Neural Network approaches."""
    
    def __init__(self, config: AdvancedArchitectureConfig):
        super().__init__()
        self.config = config
        
        # Transformer component
        self.transformer = TransformerRegimeDetector(config)
        
        # GNN component
        self.gnn = GraphNeuralNetworkRegimeDetector(config)
        
        # Temporal CNN component
        self.temporal_cnn = TemporalConvolutionalRegimeDetector(config)
        
        # Fusion mechanism
        self.fusion_attention = nn.MultiheadAttention(
            config.hidden_dim * 3, config.num_heads, dropout=config.dropout, batch_first=True
        )
        
        # Final classification
        self.final_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_regimes)
        )
        
    def forward(self, x, adjacency_matrix=None, regime_labels=None):
        # Get outputs from each component
        transformer_out = self.transformer(x, regime_labels)
        temporal_out = self.temporal_cnn(x)
        
        # GNN output (if adjacency matrix provided)
        if adjacency_matrix is not None:
            # Create node features from sequence data
            node_features = x.unsqueeze(2)  # Add node dimension
            gnn_out = self.gnn(node_features, adjacency_matrix)
        else:
            # Use transformer hidden states as GNN input
            gnn_out = {'regime_logits': transformer_out['regime_logits']}
        
        # Fuse representations
        transformer_features = transformer_out['hidden_states']
        temporal_features = temporal_out['temporal_features']
        
        # Create GNN features (use transformer features if no GNN output)
        if adjacency_matrix is not None:
            gnn_features = gnn_out['node_embeddings'].mean(dim=1)  # Global pooling
        else:
            gnn_features = transformer_features
        
        # Concatenate features
        fused_features = torch.cat([transformer_features, temporal_features, gnn_features], dim=-1)
        
        # Fusion attention
        fused_attn, _ = self.fusion_attention(fused_features, fused_features, fused_features)
        
        # Final classification
        final_logits = self.final_classifier(fused_attn)
        
        return {
            'regime_logits': final_logits,
            'transformer_logits': transformer_out['regime_logits'],
            'temporal_logits': temporal_out['regime_logits'],
            'gnn_logits': gnn_out['regime_logits'] if adjacency_matrix is not None else transformer_out['regime_logits'],
            'fused_features': fused_attn,
            'attention_weights': transformer_out['attention_weights']
        }


def create_advanced_architecture(config: AdvancedArchitectureConfig) -> nn.Module:
    """Factory function to create advanced neural architectures."""
    
    if config.architecture_type == ArchitectureType.TRANSFORMER_REGIME:
        return TransformerRegimeDetector(config)
    elif config.architecture_type == ArchitectureType.GRAPH_NEURAL_NETWORK:
        return GraphNeuralNetworkRegimeDetector(config)
    elif config.architecture_type == ArchitectureType.TEMPORAL_CONVOLUTIONAL:
        return TemporalConvolutionalRegimeDetector(config)
    elif config.architecture_type == ArchitectureType.HYBRID_TRANSFORMER_GNN:
        return HybridTransformerGNN(config)
    else:
        raise ValueError(f"Unsupported architecture type: {config.architecture_type}")


class AdvancedArchitectureManager:
    """Manager for advanced neural architectures."""
    
    def __init__(self, config: AdvancedArchitectureConfig):
        self.config = config
        self.architecture = create_advanced_architecture(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def forward(self, x, **kwargs):
        """Forward pass through the architecture."""
        return self.architecture(x, **kwargs)
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Get information about the architecture."""
        total_params = sum(p.numel() for p in self.architecture.parameters())
        trainable_params = sum(p.numel() for p in self.architecture.parameters() if p.requires_grad)
        
        return {
            'architecture_type': self.config.architecture_type.value,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_dim': self.config.input_dim,
            'hidden_dim': self.config.hidden_dim,
            'num_heads': self.config.num_heads,
            'num_layers': self.config.num_layers,
            'num_regimes': self.config.num_regimes
        }
    
    def save_architecture(self, filepath: str):
        """Save the architecture to file."""
        torch.save({
            'config': self.config,
            'state_dict': self.architecture.state_dict(),
            'architecture_info': self.get_architecture_info()
        }, filepath)
        self.logger.info(f"Advanced architecture saved to {filepath}")
    
    def load_architecture(self, filepath: str):
        """Load the architecture from file."""
        checkpoint = torch.load(filepath)
        self.config = checkpoint['config']
        self.architecture = create_advanced_architecture(self.config)
        self.architecture.load_state_dict(checkpoint['state_dict'])
        self.logger.info(f"Advanced architecture loaded from {filepath}")