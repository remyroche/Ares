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

# Import tprint for comprehensive debugging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)

class ArchitectureType(Enum):
    """Available advanced architecture types."""
    TRANSFORMER_REGIME = "transformer_regime"
    GRAPH_NEURAL_NETWORK = "graph_neural_network"
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
        tprint("🧠 [TRANSFORMER] Initializing Transformer Regime Detector", color="blue", bold=True)
        tprint(f"📊 [TRANSFORMER] Config: input_dim={config.input_dim}, hidden_dim={config.hidden_dim}, num_heads={config.num_heads}", color="cyan")

        self.config = config
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        self.pos_encoding = PositionalEncoding(config.hidden_dim, config.max_sequence_length)

        tprint("✅ [TRANSFORMER] Input projection and positional encoding initialized", color="green")

        # Regime-aware attention layers
        tprint(f"🔧 [TRANSFORMER] Creating {config.num_layers} regime attention layers", color="yellow")
        self.regime_attention_layers = nn.ModuleList([
            RegimeAttention(config.hidden_dim, config.num_heads, config.num_regimes, config.dropout)
            for _ in range(config.num_layers)
        ])

        # Temporal attention for time series patterns
        tprint("🔧 [TRANSFORMER] Creating temporal attention layer", color="yellow")
        self.temporal_attention = nn.MultiheadAttention(
            config.hidden_dim, config.num_heads, dropout=config.dropout, batch_first=True
        )

        # Feed-forward networks
        tprint(f"🔧 [TRANSFORMER] Creating {config.num_layers} feed-forward networks", color="yellow")
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
        tprint(f"🔧 [TRANSFORMER] Creating regime classifier for {config.num_regimes} regimes", color="yellow")
        self.regime_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_regimes)
        )

        # Regime transition prediction
        tprint("🔧 [TRANSFORMER] Creating regime transition predictor", color="yellow")
        self.transition_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_regimes)
        )

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim) for _ in range(config.num_layers)
        ])

        tprint_success("✅ [TRANSFORMER] Transformer Regime Detector fully initialized")

    def forward(self, x, regime_labels=None):
        tprint("🚀 [TRANSFORMER] Starting forward pass", color="blue")
        batch_size, seq_len, input_dim = x.size()
        tprint(f"📊 [TRANSFORMER] Input shape: batch_size={batch_size}, seq_len={seq_len}, input_dim={input_dim}", color="cyan")

        # Input projection and positional encoding
        tprint("🔧 [TRANSFORMER] Applying input projection", color="yellow")
        x = self.input_projection(x)
        tprint(f"📊 [TRANSFORMER] After projection: {x.shape}", color="cyan")

        if self.config.use_positional_encoding:
            tprint("🔧 [TRANSFORMER] Applying positional encoding", color="yellow")
            x = x.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)
            x = self.pos_encoding(x)
            x = x.transpose(0, 1)  # (batch_size, seq_len, hidden_dim)
            tprint(f"📊 [TRANSFORMER] After positional encoding: {x.shape}", color="cyan")
        else:
            tprint("⚠️ [TRANSFORMER] Positional encoding disabled", color="yellow")

        # Regime-aware transformer layers
        tprint(f"🔧 [TRANSFORMER] Processing {len(self.regime_attention_layers)} transformer layers", color="yellow")
        attention_weights = []
        for i, (regime_attn, ffn, layer_norm) in enumerate(
            zip(self.regime_attention_layers, self.ffn_layers, self.layer_norms)
        ):
            tprint(f"🔧 [TRANSFORMER] Processing layer {i+1}/{len(self.regime_attention_layers)}", color="yellow")

            # Regime-aware attention
            if self.config.use_regime_attention:
                tprint(f"🔧 [TRANSFORMER] Using regime-aware attention for layer {i+1}", color="yellow")
                x_attn, attn_weights = regime_attn(x, regime_labels)
                attention_weights.append(attn_weights)
                tprint(f"📊 [TRANSFORMER] Layer {i+1} attention weights shape: {attn_weights.shape if attn_weights is not None else 'None'}", color="cyan")
            else:
                tprint(f"🔧 [TRANSFORMER] Using temporal attention for layer {i+1}", color="yellow")
                x_attn, _ = self.temporal_attention(x, x, x)
                attention_weights.append(None)

            # Feed-forward network
            tprint(f"🔧 [TRANSFORMER] Applying feed-forward network for layer {i+1}", color="yellow")
            x_ffn = ffn(x_attn)

            # Residual connection and layer norm
            tprint(f"🔧 [TRANSFORMER] Applying residual connection and layer norm for layer {i+1}", color="yellow")
            x = layer_norm(x_attn + x_ffn)
            tprint(f"📊 [TRANSFORMER] Layer {i+1} output shape: {x.shape}", color="cyan")

        # Regime classification
        tprint("🔧 [TRANSFORMER] Applying regime classification", color="yellow")
        regime_logits = self.regime_classifier(x)
        tprint(f"📊 [TRANSFORMER] Regime logits shape: {regime_logits.shape}", color="cyan")

        # Regime transition prediction (using consecutive hidden states)
        if seq_len > 1:
            tprint(f"🔧 [TRANSFORMER] Computing transition probabilities for {seq_len-1} transitions", color="yellow")
            transitions = []
            for i in range(seq_len - 1):
                transition_input = torch.cat([x[:, i], x[:, i + 1]], dim=-1)
                transition_logits = self.transition_predictor(transition_input)
                transitions.append(transition_logits)
            transition_logits = torch.stack(transitions, dim=1)
            tprint(f"📊 [TRANSFORMER] Transition logits shape: {transition_logits.shape}", color="cyan")
        else:
            tprint("⚠️ [TRANSFORMER] Sequence length too short for transition prediction", color="yellow")
            transition_logits = None

        result = {
            'regime_logits': regime_logits,
            'transition_logits': transition_logits,
            'hidden_states': x,
            'attention_weights': attention_weights
        }

        tprint_success("✅ [TRANSFORMER] Forward pass completed successfully")
        tprint(f"📊 [TRANSFORMER] Final output shapes: regime_logits={regime_logits.shape}, hidden_states={x.shape}", color="cyan")

        return result

class GraphNeuralNetworkRegimeDetector(nn.Module):
    """Graph Neural Network for market relationship modeling and regime detection."""

    def __init__(self, config: AdvancedArchitectureConfig):
        super().__init__()
        tprint("🌐 [GNN] Initializing Graph Neural Network Regime Detector", color="blue", bold=True)
        tprint(f"📊 [GNN] Config: input_dim={config.input_dim}, hidden_dim={config.hidden_dim}, num_heads={config.num_heads}", color="cyan")

        self.config = config

        # Node embedding
        tprint("🔧 [GNN] Creating node embedding layer", color="yellow")
        self.node_embedding = nn.Linear(config.input_dim, config.hidden_dim)

        # Graph attention layers
        tprint(f"🔧 [GNN] Creating {config.num_layers} graph attention layers", color="yellow")
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(config.hidden_dim, config.hidden_dim // config.num_heads, config.num_heads)
            for _ in range(config.num_layers)
        ])

        # Temporal graph convolution
        tprint("🔧 [GNN] Creating temporal graph convolution layer", color="yellow")
        self.temporal_gcn = TemporalGraphConvolution(config.hidden_dim, config.hidden_dim)

        # Regime classification
        tprint(f"🔧 [GNN] Creating regime classifier for {config.num_regimes} regimes", color="yellow")
        self.regime_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_regimes)
        )

        tprint_success("✅ [GNN] Graph Neural Network Regime Detector fully initialized")

    def forward(self, node_features, adjacency_matrix, timestamps=None):
        tprint("🚀 [GNN] Starting forward pass", color="blue")
        batch_size, num_nodes, seq_len, input_dim = node_features.size()
        tprint(f"📊 [GNN] Input shapes: node_features={node_features.shape}, adjacency_matrix={adjacency_matrix.shape}", color="cyan")

        # Node embedding
        tprint("🔧 [GNN] Applying node embedding", color="yellow")
        node_emb = self.node_embedding(node_features.view(-1, input_dim))
        node_emb = node_emb.view(batch_size, num_nodes, seq_len, -1)
        tprint(f"📊 [GNN] Node embeddings shape: {node_emb.shape}", color="cyan")

        # Process each time step
        tprint(f"🔧 [GNN] Processing {seq_len} time steps", color="yellow")
        regime_logits_list = []
        for t in range(seq_len):
            tprint(f"🔧 [GNN] Processing time step {t+1}/{seq_len}", color="yellow")

            # Current time step node features
            current_nodes = node_emb[:, :, t, :]  # (batch_size, num_nodes, hidden_dim)
            tprint(f"📊 [GNN] Time step {t+1} nodes shape: {current_nodes.shape}", color="cyan")

            # Graph attention layers
            x = current_nodes
            for i, gat_layer in enumerate(self.gat_layers):
                tprint(f"🔧 [GNN] Applying GAT layer {i+1}/{len(self.gat_layers)}", color="yellow")
                x = gat_layer(x, adjacency_matrix)
                tprint(f"📊 [GNN] GAT layer {i+1} output shape: {x.shape}", color="cyan")

            # Temporal graph convolution if timestamps available
            if timestamps is not None and t > 0:
                tprint(f"🔧 [GNN] Applying temporal graph convolution at time {t+1}", color="yellow")
                x = self.temporal_gcn(x, node_emb[:, :, t-1, :], timestamps[:, t])
                tprint(f"📊 [GNN] Temporal GCN output shape: {x.shape}", color="cyan")
            else:
                tprint(f"⚠️ [GNN] No temporal graph convolution at time {t+1} (no timestamps or first timestep)", color="yellow")

            # Regime classification for this time step
            tprint(f"🔧 [GNN] Computing regime classification for time {t+1}", color="yellow")
            regime_logits = self.regime_classifier(x.mean(dim=1))  # Global pooling
            regime_logits_list.append(regime_logits)
            tprint(f"📊 [GNN] Time {t+1} regime logits shape: {regime_logits.shape}", color="cyan")

        regime_logits = torch.stack(regime_logits_list, dim=1)  # (batch_size, seq_len, num_regimes)
        tprint(f"📊 [GNN] Final regime logits shape: {regime_logits.shape}", color="cyan")

        result = {
            'regime_logits': regime_logits,
            'node_embeddings': node_emb
        }

        tprint_success("✅ [GNN] Forward pass completed successfully")
        tprint(f"📊 [GNN] Final output shapes: regime_logits={regime_logits.shape}, node_embeddings={node_emb.shape}", color="cyan")

        return result

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

# Temporal Convolutional Networks removed as not necessary

class HybridTransformerGNN(nn.Module):
    """Hybrid architecture combining Transformer and Graph Neural Network approaches."""

    def __init__(self, config: AdvancedArchitectureConfig):
        super().__init__()
        tprint("🔀 [HYBRID] Initializing Hybrid Transformer-GNN Architecture", color="blue", bold=True)
        tprint(f"📊 [HYBRID] Config: input_dim={config.input_dim}, hidden_dim={config.hidden_dim}, num_heads={config.num_heads}", color="cyan")

        self.config = config

        # Transformer component
        tprint("🔧 [HYBRID] Creating Transformer component", color="yellow")
        self.transformer = TransformerRegimeDetector(config)

        # GNN component
        tprint("🔧 [HYBRID] Creating GNN component", color="yellow")
        self.gnn = GraphNeuralNetworkRegimeDetector(config)

        # Fusion mechanism
        tprint("🔧 [HYBRID] Creating fusion attention mechanism", color="yellow")
        self.fusion_attention = nn.MultiheadAttention(
            config.hidden_dim * 2, config.num_heads, dropout=config.dropout, batch_first=True
        )

        # Final classification
        tprint(f"🔧 [HYBRID] Creating final classifier for {config.num_regimes} regimes", color="yellow")
        self.final_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_regimes)
        )

        tprint_success("✅ [HYBRID] Hybrid Transformer-GNN Architecture fully initialized")

    def forward(self, x, adjacency_matrix=None, regime_labels=None):
        tprint("🚀 [HYBRID] Starting hybrid forward pass", color="blue")
        tprint(f"📊 [HYBRID] Input shape: {x.shape}, has_adjacency: {adjacency_matrix is not None}", color="cyan")

        # Get outputs from each component
        tprint("🔧 [HYBRID] Getting Transformer output", color="yellow")
        transformer_out = self.transformer(x, regime_labels)
        tprint(f"📊 [HYBRID] Transformer output shapes: regime_logits={transformer_out['regime_logits'].shape}, hidden_states={transformer_out['hidden_states'].shape}", color="cyan")

        # GNN output (if adjacency matrix provided)
        if adjacency_matrix is not None:
            tprint("🔧 [HYBRID] Processing GNN component with adjacency matrix", color="yellow")
            # Create node features from sequence data
            node_features = x.unsqueeze(2)  # Add node dimension
            tprint(f"📊 [HYBRID] Node features shape: {node_features.shape}", color="cyan")
            gnn_out = self.gnn(node_features, adjacency_matrix)
            tprint(f"📊 [HYBRID] GNN output shapes: regime_logits={gnn_out['regime_logits'].shape}, node_embeddings={gnn_out['node_embeddings'].shape}", color="cyan")
        else:
            tprint("⚠️ [HYBRID] No adjacency matrix provided, using transformer features for GNN", color="yellow")
            # Use transformer hidden states as GNN input
            gnn_out = {'regime_logits': transformer_out['regime_logits']}

        # Fuse representations
        tprint("🔧 [HYBRID] Fusing representations from components", color="yellow")
        transformer_features = transformer_out['hidden_states']

        # Create GNN features (use transformer features if no GNN output)
        if adjacency_matrix is not None:
            gnn_features = gnn_out['node_embeddings'].mean(dim=1)  # Global pooling
            tprint(f"📊 [HYBRID] GNN features after global pooling: {gnn_features.shape}", color="cyan")
        else:
            gnn_features = transformer_features
            tprint(f"📊 [HYBRID] Using transformer features as GNN features: {gnn_features.shape}", color="cyan")

        # Concatenate features
        tprint("🔧 [HYBRID] Concatenating features from all components", color="yellow")
        fused_features = torch.cat([transformer_features, gnn_features], dim=-1)
        tprint(f"📊 [HYBRID] Fused features shape: {fused_features.shape}", color="cyan")

        # Fusion attention
        tprint("🔧 [HYBRID] Applying fusion attention mechanism", color="yellow")
        fused_attn, _ = self.fusion_attention(fused_features, fused_features, fused_features)
        tprint(f"📊 [HYBRID] Fusion attention output shape: {fused_attn.shape}", color="cyan")

        # Final classification
        tprint("🔧 [HYBRID] Applying final classification", color="yellow")
        final_logits = self.final_classifier(fused_attn)
        tprint(f"📊 [HYBRID] Final logits shape: {final_logits.shape}", color="cyan")

        result = {
            'regime_logits': final_logits,
            'transformer_logits': transformer_out['regime_logits'],
            'gnn_logits': gnn_out['regime_logits'] if adjacency_matrix is not None else transformer_out['regime_logits'],
            'fused_features': fused_attn,
            'attention_weights': transformer_out['attention_weights']
        }

        tprint_success("✅ [HYBRID] Hybrid forward pass completed successfully")
        tprint(f"📊 [HYBRID] Final output shapes: regime_logits={final_logits.shape}, fused_features={fused_attn.shape}", color="cyan")

        return result

def create_advanced_architecture(config: AdvancedArchitectureConfig) -> nn.Module:
    """Factory function to create advanced neural architectures."""
    tprint("🏭 [FACTORY] Creating advanced neural architecture", color="blue", bold=True)
    tprint(f"📊 [FACTORY] Architecture type: {config.architecture_type.value}", color="cyan")

    if config.architecture_type == ArchitectureType.TRANSFORMER_REGIME:
        tprint("🔧 [FACTORY] Creating Transformer Regime Detector", color="yellow")
        return TransformerRegimeDetector(config)
    elif config.architecture_type == ArchitectureType.GRAPH_NEURAL_NETWORK:
        tprint("🔧 [FACTORY] Creating Graph Neural Network Regime Detector", color="yellow")
        return GraphNeuralNetworkRegimeDetector(config)
    elif config.architecture_type == ArchitectureType.HYBRID_TRANSFORMER_GNN:
        tprint("🔧 [FACTORY] Creating Hybrid Transformer-GNN", color="yellow")
        return HybridTransformerGNN(config)
    else:
        tprint_error(f"❌ [FACTORY] Unsupported architecture type: {config.architecture_type}")
        raise ValueError(f"Unsupported architecture type: {config.architecture_type}")

class AdvancedArchitectureManager:
    """Manager for advanced neural architectures."""

    def __init__(self, config: AdvancedArchitectureConfig):
        tprint("🎛️ [MANAGER] Initializing Advanced Architecture Manager", color="blue", bold=True)
        tprint(f"📊 [MANAGER] Config: {config.architecture_type.value}", color="cyan")

        self.config = config
        self.architecture = create_advanced_architecture(config)
        self.logger = logging.getLogger(self.__class__.__name__)

        tprint_success("✅ [MANAGER] Advanced Architecture Manager initialized")

    def forward(self, x, **kwargs):
        """Forward pass through the architecture."""
        tprint("🚀 [MANAGER] Starting forward pass through architecture", color="blue")
        tprint(f"📊 [MANAGER] Input shape: {x.shape}", color="cyan")

        result = self.architecture(x, **kwargs)

        tprint_success("✅ [MANAGER] Forward pass completed")
        tprint(f"📊 [MANAGER] Output keys: {list(result.keys())}", color="cyan")

        return result

    def get_architecture_info(self) -> Dict[str, Any]:
        """Get information about the architecture."""
        tprint("📊 [MANAGER] Computing architecture information", color="yellow")

        total_params = sum(p.numel() for p in self.architecture.parameters())
        trainable_params = sum(p.numel() for p in self.architecture.parameters() if p.requires_grad)

        info = {
            'architecture_type': self.config.architecture_type.value,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_dim': self.config.input_dim,
            'hidden_dim': self.config.hidden_dim,
            'num_heads': self.config.num_heads,
            'num_layers': self.config.num_layers,
            'num_regimes': self.config.num_regimes
        }

        tprint(f"📊 [MANAGER] Architecture info: {total_params:,} total params, {trainable_params:,} trainable", color="cyan")

        return info

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
