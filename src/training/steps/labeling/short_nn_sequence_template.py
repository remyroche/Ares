"""
Short Neural Network Sequence Template

Provides lightweight neural network modules for temporal pattern learning:
- ConvEncoder: 1D convolutions for local patterns
- LSTMEncoder: Recurrent layer for sequential state
- FeatureAttention: Dynamic feature weighting per sample (not used in stacked encoder)
- StackedSequenceEncoder: Combines Conv + LSTM (Attention disabled by default - too slow)

All modules output fixed-size embeddings that can be used as LGBM features.

USAGE STRATEGY FOR HPO:
  Precompute embeddings ONCE before HPO starts, then reuse for all trials.
  See `precompute_nn_embeddings_for_hpo()` function.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List, Union
import warnings

try:
    from src.utils.tprint import tprint_info, tprint_warning, tprint_success
except Exception:  # pragma: no cover
    def tprint_info(*args, **kwargs):
        print(*args)

    def tprint_warning(*args, **kwargs):
        print(*args)

    def tprint_success(*args, **kwargs):
        print(*args)

# Try importing PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


# ============================================================================
# Configuration Defaults
# ============================================================================
DEFAULT_SEQ_LEN = 24
DEFAULT_CHANNELS = 4  # close_pct, high_pct, low_pct, vol_z
DEFAULT_EMBED_DIM = 8
DEFAULT_HIDDEN_DIM = 16


# ============================================================================
# Sequence Preparation (shared utility)
# ============================================================================
def prepare_ohlcv_sequences(
    market_data: pd.DataFrame,
    seq_len: int = DEFAULT_SEQ_LEN,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
) -> Tuple[np.ndarray, pd.Index]:
    """
    Prepare normalized OHLCV sequences for neural network input.
    
    Returns:
        (sequences: (N, seq_len, 4), valid_indices)
    """
    if market_data is None or len(market_data) < seq_len:
        return np.zeros((0, seq_len, DEFAULT_CHANNELS), dtype=np.float32), pd.Index([])
    
    close = market_data[close_col].values.astype(float) if close_col in market_data.columns else np.zeros(len(market_data))
    high = market_data[high_col].values.astype(float) if high_col in market_data.columns else close
    low = market_data[low_col].values.astype(float) if low_col in market_data.columns else close
    volume = market_data[volume_col].values.astype(float) if volume_col in market_data.columns else None
    
    sequences = []
    valid_indices = []
    
    for i in range(seq_len, len(market_data)):
        # Extract window
        c = close[i-seq_len:i]
        h = high[i-seq_len:i]
        l = low[i-seq_len:i]
        v = volume[i-seq_len:i] if volume is not None else np.zeros(seq_len)
        
        # Normalize: price as % from first bar, volume as z-score
        base = c[0] if c[0] > 0 else 1.0
        c_pct = np.clip((c / base - 1.0) * 100.0, -20, 20)
        h_pct = np.clip((h / base - 1.0) * 100.0, -20, 20)
        l_pct = np.clip((l / base - 1.0) * 100.0, -20, 20)
        
        v_mean, v_std = np.nanmean(v), np.nanstd(v)
        v_z = np.clip((v - v_mean) / (v_std + 1e-8), -5, 5) if v_std > 1e-8 else np.zeros_like(v)
        v_z = np.nan_to_num(v_z, nan=0.0)
        
        seq = np.stack([c_pct, h_pct, l_pct, v_z], axis=-1).astype(np.float32)
        sequences.append(seq)
        valid_indices.append(market_data.index[i])
    
    if not sequences:
        return np.zeros((0, seq_len, DEFAULT_CHANNELS), dtype=np.float32), pd.Index([])
    
    return np.stack(sequences, axis=0), pd.Index(valid_indices)


# ============================================================================
# Neural Network Modules (PyTorch)
# ============================================================================
if TORCH_AVAILABLE:

    class ConvEncoder(nn.Module):
        """1D Convolutional encoder for local temporal patterns."""
        
        def __init__(
            self,
            in_channels: int = DEFAULT_CHANNELS,
            hidden_dim: int = DEFAULT_HIDDEN_DIM,
            embed_dim: int = DEFAULT_EMBED_DIM,
            kernel_size: int = 3,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=1)
            self.conv2 = nn.Conv1d(hidden_dim, embed_dim, kernel_size, padding=1)
            self.dropout = nn.Dropout(dropout)
            self.embed_dim = embed_dim
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: (batch, seq_len, channels) -> (batch, embed_dim)"""
            x = x.transpose(1, 2)  # (batch, channels, seq_len)
            x = F.relu(self.conv1(x))
            x = self.dropout(x)
            x = F.relu(self.conv2(x))
            return x.mean(dim=2)  # Global avg pool


    class LSTMEncoder(nn.Module):
        """LSTM encoder for sequential state and memory."""
        
        def __init__(
            self,
            in_channels: int = DEFAULT_CHANNELS,
            hidden_dim: int = DEFAULT_HIDDEN_DIM,
            embed_dim: int = DEFAULT_EMBED_DIM,
            num_layers: int = 1,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=in_channels,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(hidden_dim, embed_dim)
            self.embed_dim = embed_dim
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: (batch, seq_len, channels) -> (batch, embed_dim)"""
            _, (h_n, _) = self.lstm(x)  # h_n: (layers, batch, hidden)
            h_last = h_n[-1]  # Take last layer
            return self.fc(h_last)


    class FeatureAttention(nn.Module):
        """
        Learns per-sample feature importance weights.
        
        Input: feature vector (batch, num_features)
        Output: attention-weighted features (batch, num_features)
        """
        
        def __init__(self, num_features: int, temperature: float = 1.0):
            super().__init__()
            self.attention = nn.Linear(num_features, num_features)
            self.temperature = temperature
        
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Returns: (weighted_features, attention_weights)
            """
            # Compute attention weights
            attn_logits = self.attention(x) / self.temperature
            attn_weights = F.softmax(attn_logits, dim=-1)
            
            # Apply attention (element-wise, like a learned mask)
            weighted = x * attn_weights * x.shape[-1]  # Scale to preserve magnitude
            
            return weighted, attn_weights



    # NOTE: TemporalSelfAttention is commented out for now - it's slower and
    # doesn't add much value over Conv+LSTM for typical trading use cases.
    # Uncomment if you need long-range temporal dependencies.
    #
    # class TemporalSelfAttention(nn.Module):
    #     """
    #     Self-attention over temporal sequence.
    #     Captures long-range dependencies between time steps.
    #     """
    #     
    #     def __init__(
    #         self,
    #         in_channels: int = DEFAULT_CHANNELS,
    #         embed_dim: int = DEFAULT_EMBED_DIM,
    #         num_heads: int = 2,
    #         dropout: float = 0.1,
    #     ):
    #         super().__init__()
    #         self.input_proj = nn.Linear(in_channels, embed_dim)
    #         self.attention = nn.MultiheadAttention(
    #             embed_dim=embed_dim,
    #             num_heads=num_heads,
    #             dropout=dropout,
    #             batch_first=True,
    #         )
    #         self.output_proj = nn.Linear(embed_dim, embed_dim)
    #         self.embed_dim = embed_dim
    #     
    #     def forward(self, x: torch.Tensor) -> torch.Tensor:
    #         """x: (batch, seq_len, channels) -> (batch, embed_dim)"""
    #         x = self.input_proj(x)
    #         attn_out, _ = self.attention(x, x, x)
    #         pooled = attn_out.mean(dim=1)
    #         return self.output_proj(pooled)



    class StackedSequenceEncoder(nn.Module):
        """
        Combines Conv + LSTM + optional Attention for comprehensive
        temporal pattern capture.
        
        Output: concatenated embeddings from all sub-encoders.
        """
        
        def __init__(
            self,
            seq_len: int = DEFAULT_SEQ_LEN,
            in_channels: int = DEFAULT_CHANNELS,
            embed_dim_per_encoder: int = 4,  # Each encoder outputs this many dims
            use_conv: bool = True,
            use_lstm: bool = True,
            use_attention: bool = False,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.encoders = nn.ModuleDict()
            self.embed_dim = 0
            
            if use_conv:
                self.encoders["conv"] = ConvEncoder(
                    in_channels=in_channels,
                    embed_dim=embed_dim_per_encoder,
                    dropout=dropout,
                )
                self.embed_dim += embed_dim_per_encoder
            
            if use_lstm:
                self.encoders["lstm"] = LSTMEncoder(
                    in_channels=in_channels,
                    embed_dim=embed_dim_per_encoder,
                    dropout=dropout,
                )
                self.embed_dim += embed_dim_per_encoder
            
            if use_attention:
                self.encoders["attn"] = TemporalSelfAttention(
                    in_channels=in_channels,
                    embed_dim=embed_dim_per_encoder,
                    dropout=dropout,
                )
                self.embed_dim += embed_dim_per_encoder
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: (batch, seq_len, channels) -> (batch, total_embed_dim)"""
            outputs = []
            for name, encoder in self.encoders.items():
                outputs.append(encoder(x))
            
            if not outputs:
                return torch.zeros(x.size(0), 1, device=x.device)
            
            return torch.cat(outputs, dim=-1)
        
        @torch.no_grad()
        def generate_embeddings(
            self,
            sequences: np.ndarray,
            batch_size: int = 256,
            device: str = "cpu",
        ) -> np.ndarray:
            """Generate embeddings for batch of sequences."""
            self.eval()
            self.to(device)
            
            all_embeddings = []
            for i in range(0, len(sequences), batch_size):
                batch = torch.from_numpy(sequences[i:i+batch_size]).float().to(device)
                emb = self(batch)
                all_embeddings.append(emb.cpu().numpy())
            
            if not all_embeddings:
                return np.zeros((0, self.embed_dim), dtype=np.float32)
            
            return np.concatenate(all_embeddings, axis=0)


    def train_sequence_encoder(
        encoder: nn.Module,
        sequences: np.ndarray,
        targets: np.ndarray,
        epochs: int = 15,
        batch_size: int = 128,
        lr: float = 1e-3,
        device: str = "cpu",
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train sequence encoder on direction prediction task.
        
        Args:
            encoder: Any encoder with .embed_dim attribute
            sequences: (N, seq_len, channels)
            targets: (N,) binary targets (1 if next bar up, 0 otherwise)
        
        Returns:
            Training history dict
        """
        encoder.to(device)
        encoder.train()
        
        # Classification head
        classifier = nn.Linear(encoder.embed_dim, 1).to(device)
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(classifier.parameters()),
            lr=lr,
        )
        criterion = nn.BCEWithLogitsLoss()
        
        dataset = TensorDataset(
            torch.from_numpy(sequences).float(),
            torch.from_numpy(targets).float().unsqueeze(1),
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        history = {"loss": [], "accuracy": []}
        
        for epoch in range(epochs):
            epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0
            
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                emb = encoder(batch_x)
                logits = classifier(emb)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * len(batch_x)
                preds = (torch.sigmoid(logits) > 0.5).float()
                epoch_correct += (preds == batch_y).sum().item()
                epoch_total += len(batch_x)
            
            avg_loss = epoch_loss / epoch_total
            accuracy = epoch_correct / epoch_total
            history["loss"].append(avg_loss)
            history["accuracy"].append(accuracy)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, acc={accuracy:.4f}")
        
        encoder.eval()
        return history


else:
    # Fallback stubs when PyTorch not available
    class ConvEncoder:
        def __init__(self, *args, **kwargs):
            warnings.warn("PyTorch not available")
            self.embed_dim = kwargs.get("embed_dim", DEFAULT_EMBED_DIM)
        def generate_embeddings(self, seq, **kw):
            return np.zeros((len(seq), self.embed_dim), dtype=np.float32)
    
    class LSTMEncoder(ConvEncoder):
        pass
    
    class FeatureAttention:
        def __init__(self, *args, **kwargs):
            warnings.warn("PyTorch not available")
    
    class TemporalSelfAttention(ConvEncoder):
        pass
    
    class StackedSequenceEncoder(ConvEncoder):
        pass
    
    def train_sequence_encoder(*args, **kwargs):
        return {"loss": [], "accuracy": []}


# ============================================================================
# High-Level API
# ============================================================================
def generate_nn_sequence_embeddings(
    market_data: pd.DataFrame,
    encoder_type: str = "stacked",  # "conv", "lstm", "attention", "stacked"
    seq_len: int = DEFAULT_SEQ_LEN,
    embed_dim: int = DEFAULT_EMBED_DIM,
    use_conv: bool = True,
    use_lstm: bool = True,
    use_attention: bool = False,
    device: str = "cpu",
    pretrained_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate neural network embedding features from OHLCV sequences.
    
    Args:
        market_data: DataFrame with OHLCV columns
        encoder_type: Which encoder to use
        seq_len: Lookback window in bars
        embed_dim: Output embedding dimension
        use_conv: Include Conv encoder (for stacked)
        use_lstm: Include LSTM encoder (for stacked)
        use_attention: Include Attention encoder (for stacked)
        device: PyTorch device
        pretrained_path: Path to pretrained weights
    
    Returns:
        DataFrame with nn_embed_* columns
    """
    try:
        tprint_info(
            "[nn_sequence] generate_embeddings start "
            f"rows={int(len(market_data)) if market_data is not None else 0} "
            f"encoder_type={encoder_type} seq_len={int(seq_len)} embed_dim={int(embed_dim)} "
            f"device={device} torch_available={bool(TORCH_AVAILABLE)}"
        )
    except Exception:
        pass

    if market_data is None or not isinstance(market_data, pd.DataFrame) or market_data.empty:
        cols = [f"nn_embed_{i}" for i in range(int(embed_dim))]
        return pd.DataFrame(0.0, index=pd.Index([]), columns=cols)

    if not TORCH_AVAILABLE:
        cols = [f"nn_embed_{i}" for i in range(int(embed_dim))]
        tprint_warning("[nn_sequence] PyTorch not available; returning zero-filled nn_embed_* features")
        return pd.DataFrame(0.0, index=market_data.index, columns=cols)
    
    # Prepare sequences
    sequences, valid_indices = prepare_ohlcv_sequences(market_data, seq_len=seq_len)

    try:
        tprint_info(
            "[nn_sequence] prepared_sequences "
            f"n_sequences={int(len(sequences))} valid_indices={int(len(valid_indices))} "
            f"seq_shape={tuple(sequences.shape) if hasattr(sequences, 'shape') else None}"
        )
    except Exception:
        pass
    
    if len(sequences) == 0:
        cols = [f"nn_embed_{i}" for i in range(int(embed_dim))]
        try:
            tprint_warning("[nn_sequence] no_sequences_prepared; returning zero-filled nn_embed_* features")
        except Exception:
            pass
        return pd.DataFrame(0.0, index=market_data.index, columns=cols)
    
    # Create encoder
    if encoder_type == "conv":
        encoder = ConvEncoder(embed_dim=embed_dim)
    elif encoder_type == "lstm":
        encoder = LSTMEncoder(embed_dim=embed_dim)
    elif encoder_type == "attention":
        encoder = TemporalSelfAttention(embed_dim=embed_dim)
    else:  # stacked
        n_encoders = int(use_conv) + int(use_lstm) + int(use_attention)
        embed_per = max(1, embed_dim // max(1, n_encoders))
        encoder = StackedSequenceEncoder(
            seq_len=seq_len,
            embed_dim_per_encoder=embed_per,
            use_conv=use_conv,
            use_lstm=use_lstm,
            use_attention=use_attention,
        )
    
    # Load pretrained if available
    if pretrained_path and os.path.exists(pretrained_path):
        try:
            encoder.load_state_dict(torch.load(pretrained_path, map_location=device))
            tprint_success(f"[nn_sequence] loaded_pretrained_weights path={pretrained_path}")
        except Exception as e:
            tprint_warning(f"[nn_sequence] failed_to_load_pretrained_weights path={pretrained_path} error={e}")
    
    # Generate embeddings
    embeddings = encoder.generate_embeddings(sequences, device=device)

    try:
        emb_nan_rate = float(np.isnan(embeddings).mean()) if isinstance(embeddings, np.ndarray) and embeddings.size else 0.0
        tprint_info(
            "[nn_sequence] generated_embeddings "
            f"shape={tuple(embeddings.shape) if hasattr(embeddings, 'shape') else None} "
            f"nan_rate={emb_nan_rate:.6f}"
        )
    except Exception:
        pass
    
    # Create output DataFrame
    actual_dim = embeddings.shape[1] if len(embeddings) > 0 else embed_dim
    cols = [f"nn_embed_{i}" for i in range(actual_dim)]
    embed_df = pd.DataFrame(embeddings, index=valid_indices, columns=cols)

    # Reindex to match market_data
    out = embed_df.reindex(market_data.index)
    try:
        out = out.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    except Exception:
        out = out.fillna(0.0)

    try:
        tprint_success(f"[nn_sequence] generate_embeddings done out_shape={tuple(out.shape)}")
    except Exception:
        pass
    return out


# ============================================================================
# HPO Precomputation Strategy
# ============================================================================
def precompute_nn_embeddings_for_hpo(
    market_data: pd.DataFrame,
    cache_path: str,
    seq_len: int = DEFAULT_SEQ_LEN,
    embed_dim: int = DEFAULT_EMBED_DIM,
    use_conv: bool = True,
    use_lstm: bool = True,
    device: str = "cpu",
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Precompute NN embeddings ONCE before HPO, cache to disk.
    
    Call this before starting HPO loop. All trials will then use the
    cached embeddings via `load_cached_nn_embeddings()`.
    
    Args:
        market_data: Full OHLCV DataFrame
        cache_path: Path to save/load embeddings parquet file
        seq_len: Sequence length
        embed_dim: Embedding dimension
        use_conv: Include Conv encoder
        use_lstm: Include LSTM encoder
        device: PyTorch device
        force_recompute: If True, recompute even if cache exists
    
    Returns:
        DataFrame with nn_embed_* columns (also saved to cache_path)
    """
    try:
        tprint_info(
            "[nn_sequence] precompute_for_hpo start "
            f"rows={int(len(market_data)) if market_data is not None else 0} "
            f"cache_path={cache_path} seq_len={int(seq_len)} embed_dim={int(embed_dim)} "
            f"force_recompute={bool(force_recompute)}"
        )
    except Exception:
        pass

    # Check cache
    if not force_recompute and os.path.exists(cache_path):
        try:
            cached = pd.read_parquet(cache_path)
            # Verify alignment
            if len(cached) == len(market_data):
                tprint_success(f"[nn_sequence] loaded_cache path={cache_path} shape={tuple(cached.shape)}")
                out = cached.set_index(market_data.index)
                try:
                    out = out.apply(pd.to_numeric, errors="coerce").fillna(0.0)
                except Exception:
                    out = out.fillna(0.0)
                return out
        except Exception as e:
            tprint_warning(f"[nn_sequence] failed_to_load_cache path={cache_path} error={e}")

    tprint_info("[nn_sequence] computing_embeddings (runs once before HPO)")
    
    # Generate embeddings
    embed_df = generate_nn_sequence_embeddings(
        market_data=market_data,
        encoder_type="stacked",
        seq_len=seq_len,
        embed_dim=embed_dim,
        use_conv=use_conv,
        use_lstm=use_lstm,
        use_attention=False,  # Disabled for speed
        device=device,
    )
    
    # Save to cache
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        embed_df.to_parquet(cache_path)
        tprint_success(f"[nn_sequence] saved_cache path={cache_path} shape={tuple(embed_df.shape)}")
    except Exception as e:
        tprint_warning(f"[nn_sequence] failed_to_save_cache path={cache_path} error={e}")
    
    return embed_df


def load_cached_nn_embeddings(
    cache_path: str,
    target_index: pd.Index,
) -> Optional[pd.DataFrame]:
    """
    Load precomputed NN embeddings from cache.
    
    Use this inside HPO objective function to get embeddings without recomputing.
    
    Args:
        cache_path: Path to cached embeddings parquet
        target_index: Index to align embeddings to
    
    Returns:
        DataFrame with nn_embed_* columns, or None if cache not found
    """
    if not os.path.exists(cache_path):
        try:
            tprint_warning(f"[nn_sequence] cache_not_found path={cache_path}")
        except Exception:
            pass
        return None
    
    try:
        cached = pd.read_parquet(cache_path)

        # Check for index overlap
        overlap = cached.index.intersection(target_index)
        if overlap.empty and len(target_index) > 0 and len(cached) > 0:
            try:
                tprint_warning(
                    f"[nn_sequence] No index overlap between cache (range={cached.index.min()}..{cached.index.max()}) "
                    f"and target (range={target_index.min()}..{target_index.max()}). "
                    "Reindexing will produce all zeros/NaNs."
                )
            except Exception:
                pass

        # Align to target index
        # SAFETY: Removed unsafe index overwriting (cached.index = target_index[:len(cached)]).
        # We strictly rely on the cached index matching the target index via reindex().
        # This prevents accidental mapping of future/past data to current timestamps if lengths differ.
        out = cached.reindex(target_index)
        try:
            out = out.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        except Exception:
            out = out.fillna(0.0)
        try:
            tprint_success(f"[nn_sequence] loaded_cache_for_index path={cache_path} out_shape={tuple(out.shape)}")
        except Exception:
            pass
        return out
    except Exception as e:
        tprint_warning(f"[nn_sequence] failed_to_load_cache_for_index path={cache_path} error={e}")
        return None
