"""
Temporal Convolutional Encoder for OHLCV Sequences

Generates 8-dimensional temporal embeddings from recent price/volume data 
to capture patterns that LGBM's tabular features may miss.

Architecture:
- Input: (batch, seq_len=24, channels=4) where channels = [close_pct, high_pct, low_pct, vol_z]
- 2 Conv1D layers → GlobalAveragePool → 8-dim embedding
- Output: 8 features per sample (temporal_embed_0 ... temporal_embed_7)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List, Union
import warnings

# Try importing PyTorch, fallback to numpy-only if not available
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


# Default parameters
DEFAULT_SEQ_LEN = 24  # 24 bars = 6 hours at 15m
DEFAULT_EMBED_DIM = 8
DEFAULT_HIDDEN_DIM = 16
DEFAULT_KERNEL_SIZE = 3
DEFAULT_CHANNELS = 4  # close_pct, high_pct, low_pct, vol_z


def _normalize_ohlcv_sequence(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: Optional[np.ndarray],
    seq_len: int = DEFAULT_SEQ_LEN,
) -> np.ndarray:
    """
    Normalize a single OHLCV sequence for neural network input.
    
    Args:
        close: Close prices for sequence (seq_len,)
        high: High prices for sequence (seq_len,)
        low: Low prices for sequence (seq_len,)
        volume: Volume for sequence (seq_len,) or None
        seq_len: Expected sequence length
    
    Returns:
        Normalized sequence (seq_len, 4) with channels [close_pct, high_pct, low_pct, vol_z]
    """
    if len(close) < 2:
        return np.zeros((seq_len, DEFAULT_CHANNELS), dtype=np.float32)
    
    # Normalize price as percentage change from first bar
    base_price = close[0] if close[0] > 0 else 1.0
    close_pct = (close / base_price - 1.0) * 100.0  # In percentage
    high_pct = (high / base_price - 1.0) * 100.0
    low_pct = (low / base_price - 1.0) * 100.0
    
    # Volume as z-score
    if volume is not None and len(volume) > 0:
        vol_mean = np.nanmean(volume)
        vol_std = np.nanstd(volume)
        if vol_std > 1e-8:
            vol_z = (volume - vol_mean) / vol_std
        else:
            vol_z = np.zeros_like(volume)
        vol_z = np.nan_to_num(vol_z, nan=0.0)
    else:
        vol_z = np.zeros(len(close), dtype=np.float32)
    
    # Clip extreme values
    close_pct = np.clip(close_pct, -20.0, 20.0)
    high_pct = np.clip(high_pct, -20.0, 20.0)
    low_pct = np.clip(low_pct, -20.0, 20.0)
    vol_z = np.clip(vol_z, -5.0, 5.0)
    
    # Stack channels: (seq_len, 4)
    seq = np.stack([close_pct, high_pct, low_pct, vol_z], axis=-1).astype(np.float32)
    
    # Pad or truncate to seq_len
    if len(seq) < seq_len:
        pad = np.zeros((seq_len - len(seq), DEFAULT_CHANNELS), dtype=np.float32)
        seq = np.concatenate([pad, seq], axis=0)  # Pad at beginning
    elif len(seq) > seq_len:
        seq = seq[-seq_len:]  # Take last seq_len bars
    
    return seq


def prepare_sequences(
    market_data: pd.DataFrame,
    seq_len: int = DEFAULT_SEQ_LEN,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    volume_col: str = "volume",
    stride: int = 1,
) -> Tuple[np.ndarray, pd.Index]:
    """
    Prepare normalized sequences for all valid indices in market data.
    
    Args:
        market_data: DataFrame with OHLCV columns
        seq_len: Sequence length (lookback window)
        close_col: Name of close price column
        high_col: Name of high price column
        low_col: Name of low price column  
        volume_col: Name of volume column
        stride: Step between consecutive sequences (1 = every bar)
    
    Returns:
        Tuple of (sequences array (N, seq_len, 4), valid indices)
    """
    if market_data is None or len(market_data) < seq_len:
        return np.zeros((0, seq_len, DEFAULT_CHANNELS), dtype=np.float32), pd.Index([])
    
    close = market_data[close_col].values.astype(float) if close_col in market_data.columns else np.zeros(len(market_data))
    high = market_data[high_col].values.astype(float) if high_col in market_data.columns else close
    low = market_data[low_col].values.astype(float) if low_col in market_data.columns else close
    volume = market_data[volume_col].values.astype(float) if volume_col in market_data.columns else None
    
    sequences = []
    valid_indices = []
    
    for i in range(seq_len, len(market_data), stride):
        seq = _normalize_ohlcv_sequence(
            close=close[i-seq_len:i],
            high=high[i-seq_len:i],
            low=low[i-seq_len:i],
            volume=volume[i-seq_len:i] if volume is not None else None,
            seq_len=seq_len,
        )
        sequences.append(seq)
        valid_indices.append(market_data.index[i])
    
    if not sequences:
        return np.zeros((0, seq_len, DEFAULT_CHANNELS), dtype=np.float32), pd.Index([])
    
    return np.stack(sequences, axis=0), pd.Index(valid_indices)


if TORCH_AVAILABLE:
    class TemporalConvEncoder(nn.Module):
        """
        1D Convolutional encoder for temporal OHLCV patterns.
        
        Architecture:
            Input (seq_len, 4) → Conv1D(16, k=3) → Conv1D(8, k=3) → GlobalAvgPool → Dense(8)
        """
        
        def __init__(
            self,
            seq_len: int = DEFAULT_SEQ_LEN,
            in_channels: int = DEFAULT_CHANNELS,
            hidden_dim: int = DEFAULT_HIDDEN_DIM,
            embed_dim: int = DEFAULT_EMBED_DIM,
            kernel_size: int = DEFAULT_KERNEL_SIZE,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.seq_len = seq_len
            self.in_channels = in_channels
            self.embed_dim = embed_dim
            
            # Conv layers (input: batch, channels, seq_len)
            self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=kernel_size, padding=1)
            self.conv2 = nn.Conv1d(hidden_dim, embed_dim, kernel_size=kernel_size, padding=1)
            self.dropout = nn.Dropout(dropout)
            
            # Output projection
            self.fc = nn.Linear(embed_dim, embed_dim)
            
            # Initialize weights
            self._init_weights()
            
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                x: Input tensor (batch, seq_len, channels)
            
            Returns:
                Embeddings (batch, embed_dim)
            """
            # Transpose to (batch, channels, seq_len) for Conv1d
            x = x.transpose(1, 2)
            
            # Conv layers with ReLU
            x = F.relu(self.conv1(x))
            x = self.dropout(x)
            x = F.relu(self.conv2(x))
            
            # Global average pooling over sequence
            x = x.mean(dim=2)  # (batch, embed_dim)
            
            # Final projection
            x = self.fc(x)
            
            return x
        
        @torch.no_grad()
        def generate_embeddings(
            self,
            sequences: np.ndarray,
            batch_size: int = 256,
            device: str = "cpu",
        ) -> np.ndarray:
            """
            Generate embeddings for a batch of sequences.
            
            Args:
                sequences: Input sequences (N, seq_len, channels)
                batch_size: Batch size for inference
                device: Device to run on ("cpu" or "cuda")
            
            Returns:
                Embeddings (N, embed_dim)
            """
            self.eval()
            self.to(device)
            
            all_embeddings = []
            
            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i+batch_size]
                batch_tensor = torch.from_numpy(batch).float().to(device)
                embeddings = self(batch_tensor)
                all_embeddings.append(embeddings.cpu().numpy())
            
            if not all_embeddings:
                return np.zeros((0, self.embed_dim), dtype=np.float32)
            
            return np.concatenate(all_embeddings, axis=0)
    
    
    def train_encoder_self_supervised(
        encoder: TemporalConvEncoder,
        sequences: np.ndarray,
        targets: np.ndarray,
        epochs: int = 10,
        batch_size: int = 128,
        lr: float = 1e-3,
        device: str = "cpu",
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train encoder using self-supervised task (predict next-bar direction).
        
        Args:
            encoder: TemporalConvEncoder model
            sequences: Training sequences (N, seq_len, channels)
            targets: Binary targets (N,) - 1 if next bar is up, 0 otherwise
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            device: Device to train on
            verbose: Print training progress
        
        Returns:
            Dict with training history
        """
        encoder.to(device)
        encoder.train()
        
        # Add classification head for training
        classifier = nn.Linear(encoder.embed_dim, 1).to(device)
        
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(classifier.parameters()),
            lr=lr,
        )
        criterion = nn.BCEWithLogitsLoss()
        
        # Create dataloader
        dataset = TensorDataset(
            torch.from_numpy(sequences).float(),
            torch.from_numpy(targets).float().unsqueeze(1),
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        history = {"loss": [], "accuracy": []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                
                # Forward
                embeddings = encoder(batch_x)
                logits = classifier(embeddings)
                loss = criterion(logits, batch_y)
                
                # Backward
                loss.backward()
                optimizer.step()
                
                # Stats
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
    # Fallback when PyTorch is not available
    class TemporalConvEncoder:
        """Dummy encoder when PyTorch is not available."""
        
        def __init__(self, *args, **kwargs):
            warnings.warn("PyTorch not available. TemporalConvEncoder will output zeros.")
            self.embed_dim = kwargs.get("embed_dim", DEFAULT_EMBED_DIM)
        
        def generate_embeddings(self, sequences: np.ndarray, **kwargs) -> np.ndarray:
            return np.zeros((len(sequences), self.embed_dim), dtype=np.float32)
    
    def train_encoder_self_supervised(*args, **kwargs):
        warnings.warn("PyTorch not available. Cannot train encoder.")
        return {"loss": [], "accuracy": []}


def generate_temporal_embeddings(
    market_data: pd.DataFrame,
    encoder: Optional[TemporalConvEncoder] = None,
    seq_len: int = DEFAULT_SEQ_LEN,
    embed_dim: int = DEFAULT_EMBED_DIM,
    device: str = "cpu",
    pretrained_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate temporal embedding features for all bars in market data.
    
    Args:
        market_data: DataFrame with OHLCV data
        encoder: Optional pretrained encoder (creates new if None)
        seq_len: Sequence length for lookback
        embed_dim: Embedding dimension
        device: Device for inference
        pretrained_path: Path to pretrained weights (optional)
    
    Returns:
        DataFrame with columns temporal_embed_0 ... temporal_embed_{embed_dim-1}
        aligned to market_data index, NaN for first seq_len-1 rows
    """
    if market_data is None or len(market_data) == 0:
        return pd.DataFrame()
    
    # Initialize encoder
    if encoder is None:
        encoder = TemporalConvEncoder(seq_len=seq_len, embed_dim=embed_dim)
        
        # Load pretrained weights if available
        if pretrained_path is not None and os.path.exists(pretrained_path):
            if TORCH_AVAILABLE:
                try:
                    encoder.load_state_dict(torch.load(pretrained_path, map_location=device))
                except Exception as e:
                    warnings.warn(f"Failed to load pretrained weights: {e}")
    
    # Prepare sequences
    sequences, valid_indices = prepare_sequences(market_data, seq_len=seq_len)
    
    if len(sequences) == 0:
        # Return empty DataFrame with correct columns
        cols = [f"temporal_embed_{i}" for i in range(embed_dim)]
        return pd.DataFrame(index=market_data.index, columns=cols)
    
    # Generate embeddings
    embeddings = encoder.generate_embeddings(sequences, device=device)
    
    # Create output DataFrame
    cols = [f"temporal_embed_{i}" for i in range(embed_dim)]
    embed_df = pd.DataFrame(embeddings, index=valid_indices, columns=cols)
    
    # Reindex to match market_data (NaN for rows without sequences)
    embed_df = embed_df.reindex(market_data.index)
    
    return embed_df


def self_supervised_pretrain(
    market_data: pd.DataFrame,
    seq_len: int = DEFAULT_SEQ_LEN,
    embed_dim: int = DEFAULT_EMBED_DIM,
    epochs: int = 20,
    save_path: Optional[str] = None,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[TemporalConvEncoder, Dict[str, Any]]:
    """
    Pretrain encoder using self-supervised direction prediction.
    
    Args:
        market_data: DataFrame with OHLCV data
        seq_len: Sequence length
        embed_dim: Embedding dimension
        epochs: Training epochs
        save_path: Path to save trained weights
        device: Training device
        verbose: Print training progress
    
    Returns:
        Tuple of (trained encoder, training history)
    """
    if not TORCH_AVAILABLE:
        warnings.warn("PyTorch not available. Cannot pretrain encoder.")
        return TemporalConvEncoder(embed_dim=embed_dim), {}
    
    # Prepare sequences
    sequences, valid_indices = prepare_sequences(market_data, seq_len=seq_len)
    
    if len(sequences) < 100:
        warnings.warn(f"Not enough sequences for training ({len(sequences)} < 100)")
        return TemporalConvEncoder(seq_len=seq_len, embed_dim=embed_dim), {}
    
    # Create direction targets (1 if next bar close > current close)
    close_col = "close" if "close" in market_data.columns else market_data.columns[0]
    close = market_data[close_col].values
    
    targets = []
    for idx in valid_indices:
        pos = market_data.index.get_loc(idx)
        if pos + 1 < len(close):
            targets.append(1.0 if close[pos + 1] > close[pos] else 0.0)
        else:
            targets.append(0.5)  # Neutral for edge case
    
    targets = np.array(targets, dtype=np.float32)
    
    # Create and train encoder
    encoder = TemporalConvEncoder(seq_len=seq_len, embed_dim=embed_dim)
    history = train_encoder_self_supervised(
        encoder=encoder,
        sequences=sequences,
        targets=targets,
        epochs=epochs,
        device=device,
        verbose=verbose,
    )
    
    # Save weights
    if save_path is not None and TORCH_AVAILABLE:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(encoder.state_dict(), save_path)
            if verbose:
                print(f"Saved encoder weights to {save_path}")
        except Exception as e:
            warnings.warn(f"Failed to save weights: {e}")
    
    return encoder, history
