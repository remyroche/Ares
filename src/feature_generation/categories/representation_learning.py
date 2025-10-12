"""
Representation Learning Feature Generator

This module provides self-supervised representation learning features
using PatchTST and TFT encoders for generating latent vectors that
summarize recent market dynamics for use in tree-based models.

Features implemented:
- PatchTST self-supervised learning
- TFT encoder representations
- Autoencoder-based embeddings
- Contrastive learning representations
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
import logging

from ..core.feature_generator import (
    FeatureGenerator,
    FeatureConfig,
    FeatureCategory,
    VectorizedFeatureGenerator
)

# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

logger = logging.getLogger(__name__)


class PatchTSTRepresentationGenerator(VectorizedFeatureGenerator):
    """Generator for PatchTST-based self-supervised representation learning."""

    def __init__(self, patch_length: int = 16, num_patches: int = 8, embedding_dim: int = 64):
        config = FeatureConfig(
            name=f"patchtst_repr_{patch_length}_{num_patches}_{embedding_dim}",
            category=FeatureCategory.AUTOENCODER,
            description=f"PatchTST representation learning with patch_length={patch_length}, num_patches={num_patches}, embedding_dim={embedding_dim}",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=patch_length * num_patches * 2,
            min_lookback=patch_length * num_patches,
            max_lookback=patch_length * num_patches * 4,
            parameters={
                "patch_length": patch_length,
                "num_patches": num_patches,
                "embedding_dim": embedding_dim,
                "masking_ratio": 0.5
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.patch_length = patch_length
        self.num_patches = num_patches
        self.embedding_dim = embedding_dim
        self.masking_ratio = 0.5

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate PatchTST representation features."""
        try:
            # Extract price sequence
            price_sequence = data["close"].values

            # Create patches
            patches = self._create_patches(price_sequence)

            # Apply masking for self-supervised learning
            masked_patches, mask = self._apply_masking(patches)

            # Learn representations (simplified - would use actual PatchTST model)
            representations = self._learn_patch_representations(masked_patches)

            # Return concatenated representation as single feature
            # In practice, this would be the learned embedding
            return pd.Series(representations.mean(axis=1), index=data.index[-len(representations):])

        except Exception as e:
            logger.warning(f"Error in PatchTST representation generation: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

    def _create_patches(self, sequence: np.ndarray) -> np.ndarray:
        """Create patches from price sequence."""
        seq_len = len(sequence)
        patch_size = self.patch_length * self.num_patches

        if seq_len < patch_size:
            # Pad sequence if too short
            padded = np.pad(sequence, (0, patch_size - seq_len), mode='edge')
        else:
            padded = sequence[-patch_size:]

        # Reshape into patches
        patches = padded.reshape(self.num_patches, self.patch_length)
        return patches

    def _apply_masking(self, patches: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply masking for self-supervised learning."""
        masked_patches = patches.copy()
        mask = np.zeros_like(patches, dtype=bool)

        # Random masking
        total_elements = patches.size
        num_masked = int(total_elements * self.masking_ratio)

        masked_indices = np.random.choice(total_elements, num_masked, replace=False)
        mask.flat[masked_indices] = True

        # Apply masking (set to zero or mean)
        masked_patches.flat[masked_indices] = 0.0

        return masked_patches, mask

    def _learn_patch_representations(self, patches: np.ndarray) -> np.ndarray:
        """Learn patch representations (simplified implementation)."""
        # In practice, this would use a trained PatchTST model
        # For now, we'll use simple statistical representations

        # Calculate statistical features for each patch
        patch_means = patches.mean(axis=1)
        patch_stds = patches.std(axis=1)
        patch_trends = np.polyfit(np.arange(self.patch_length), patches.T, 1)[0]  # Linear trend

        # Combine into representation vectors
        representations = np.column_stack([patch_means, patch_stds, patch_trends])

        return representations


    

class TFTEncoderRepresentationGenerator(FeatureGenerator):
    """Generator for TFT (Temporal Fusion Transformer) encoder representations."""

    def __init__(self, seq_length: int = 60, hidden_size: int = 64, num_heads: int = 4):
        config = FeatureConfig(
            name=f"tft_encoder_repr_{seq_length}_{hidden_size}_{num_heads}",
            category=FeatureCategory.AUTOENCODER,
            description=f"TFT encoder representation learning with seq_length={seq_length}, hidden_size={hidden_size}",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=seq_length * 2,
            min_lookback=seq_length,
            max_lookback=seq_length * 4,
            parameters={
                "seq_length": seq_length,
                "hidden_size": hidden_size,
                "num_heads": num_heads
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_heads = num_heads

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate TFT encoder representation features."""
        try:
            # Extract multi-variate time series
            features = self._extract_features(data)

            # Apply self-attention mechanism (simplified)
            attention_output = self._apply_self_attention(features)

            # Generate temporal representations
            temporal_repr = self._temporal_fusion(attention_output)

            # Return representation as feature
            return pd.Series(temporal_repr.mean(axis=1), index=data.index[-len(temporal_repr):])

        except Exception as e:
            logger.warning(f"Error in TFT encoder representation generation: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for TFT input."""
        features = []

        # Price features
        # Calculate price features using VectorBT
        if VECTORBT_AVAILABLE and len(data) > 1000:
            try:
                close_mean_20 = rolling_mean(data["close"], window=20)
                close_std_20 = rolling_std(data["close"], window=20)
                close_mean_10 = rolling_mean(data["close"], window=10)
                close_mean_30 = rolling_mean(data["close"], window=30)
            except Exception as e:
                logger.warning(f"VectorBT price features failed: {e}, using pandas fallback")
                close_mean_20 = data["close"].rolling(window=20).mean()
                close_std_20 = data["close"].rolling(window=20).std()
                close_mean_10 = data["close"].rolling(window=10).mean()
                close_mean_30 = data["close"].rolling(window=30).mean()
        else:
            close_mean_20 = data["close"].rolling(window=20).mean()
            close_std_20 = data["close"].rolling(window=20).std()
            close_mean_10 = data["close"].rolling(window=10).mean()
            close_mean_30 = data["close"].rolling(window=30).mean()
        
        price_features = [
            data["close"].pct_change(),
            (data["close"] - close_mean_20) / close_std_20,
            close_mean_10 / close_mean_30 - 1,
        ]

        # Volatility features
        returns = data["close"].pct_change()
        volatility_features = [
            self._vectorbt_rolling_operation(returns, "std", 20),
            self._vectorbt_rolling_operation(returns, "std", 5) / self._vectorbt_rolling_operation(returns, "std", 20),
        ]

        # Volume features (if available)
        if "volume" in data.columns:
            # Calculate volume features using VectorBT
            if VECTORBT_AVAILABLE and len(data) > 1000:
                try:
                    volume_mean_20 = rolling_mean(data["volume"], window=20)
                except Exception as e:
                    logger.warning(f"VectorBT volume features failed: {e}, using pandas fallback")
                    volume_mean_20 = data["volume"].rolling(window=20).mean()
            else:
                volume_mean_20 = data["volume"].rolling(window=20).mean()
            
            volume_features = [
                data["volume"] / volume_mean_20,
                data["volume"].pct_change(),
            ]
        else:
            volume_features = [np.zeros(len(data)), np.zeros(len(data))]

        # Combine all features
        all_features = price_features + volatility_features + volume_features
        feature_matrix = np.column_stack([f.fillna(0).values for f in all_features])

        # Truncate to sequence length
        if len(feature_matrix) > self.seq_length:
            feature_matrix = feature_matrix[-self.seq_length:]

        return feature_matrix

    def _apply_self_attention(self, features: np.ndarray) -> np.ndarray:
        """Apply self-attention mechanism (simplified)."""
        # Simple attention mechanism - in practice would use multi-head attention

        # Calculate attention scores (simplified)
        query = features
        key = features
        value = features

        # Attention weights (using cosine similarity as proxy)
        attention_scores = np.dot(query, key.T) / (np.linalg.norm(query, axis=1, keepdims=True) * np.linalg.norm(key, axis=1, keepdims=True).T + 1e-8)

        # Apply softmax
        attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=1, keepdims=True)

        # Weighted sum
        attention_output = np.dot(attention_weights, value)

        return attention_output

    def _temporal_fusion(self, attention_output: np.ndarray) -> np.ndarray:
        """Apply temporal fusion to generate final representations."""
        # Simple temporal fusion - in practice would use TFT's temporal fusion decoder

        # Use rolling statistics as temporal representation
        window_sizes = [5, 10, 20]
        temporal_features = []

        for window in window_sizes:
            if len(attention_output) >= window:
                # Rolling mean using VectorBT
                attention_series = pd.Series(attention_output.mean(axis=1))
                if VECTORBT_AVAILABLE and len(attention_series) > 1000:
                    try:
                        rolling_mean = rolling_mean(attention_series, window=window).fillna(0)
                    except Exception:
                        rolling_mean = attention_series.rolling(window=window).mean().fillna(0)
                else:
                    rolling_mean = attention_series.rolling(window=window).mean().fillna(0)
                temporal_features.append(rolling_mean.values)
                
                # Rolling std using VectorBT
                if VECTORBT_AVAILABLE and len(attention_series) > 1000:
                    try:
                        rolling_std = rolling_std(attention_series, window=window).fillna(0)
                    except Exception:
                        rolling_std = attention_series.rolling(window=window).std().fillna(0)
                else:
                    rolling_std = attention_series.rolling(window=window).std().fillna(0)
                temporal_features.append(rolling_std.values)

        if temporal_features:
            return np.column_stack(temporal_features)
        else:
            return attention_output


    

class AutoencoderRepresentationGenerator(FeatureGenerator):
    """Generator for autoencoder-based representation learning."""

    def __init__(self, encoding_dim: int = 32, sequence_length: int = 60):
        config = FeatureConfig(
            name=f"autoencoder_repr_{encoding_dim}_{sequence_length}",
            category=FeatureCategory.AUTOENCODER,
            description=f"Autoencoder representation learning with encoding_dim={encoding_dim}, sequence_length={sequence_length}",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=sequence_length * 2,
            min_lookback=sequence_length,
            max_lookback=sequence_length * 4,
            parameters={
                "encoding_dim": encoding_dim,
                "sequence_length": sequence_length
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.encoding_dim = encoding_dim
        self.sequence_length = sequence_length

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate autoencoder representation features."""
        try:
            # Create input features
            input_features = self._create_input_features(data)

            # Apply autoencoder (simplified - would use trained model)
            encoded_repr = self._encode_features(input_features)

            # Return encoded representation
            return pd.Series(encoded_repr.mean(axis=1), index=data.index[-len(encoded_repr):])

        except Exception as e:
            logger.warning(f"Error in autoencoder representation generation: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

    def _create_input_features(self, data: pd.DataFrame) -> np.ndarray:
        """Create input features for autoencoder."""
        features = []

        # Technical indicators as features
        # Calculate indicators using VectorBT
        if VECTORBT_AVAILABLE and len(data) > 1000:
            try:
                close_mean_10 = rolling_mean(data["close"], window=10)
                close_std_20 = rolling_std(data["close"], window=20)
            except Exception as e:
                logger.warning(f"VectorBT indicators failed: {e}, using pandas fallback")
                close_mean_10 = data["close"].rolling(window=10).mean()
                close_std_20 = data["close"].rolling(window=20).std()
        else:
            close_mean_10 = data["close"].rolling(window=10).mean()
            close_std_20 = data["close"].rolling(window=20).std()
        
        indicators = [
            data["close"].pct_change(),
            close_mean_10,
            close_std_20,
            data["close"] / close_mean_10 - 1,
        ]

        if "volume" in data.columns:
            # Calculate volume features using VectorBT
            if VECTORBT_AVAILABLE and len(data) > 1000:
                try:
                    volume_mean_20 = rolling_mean(data["volume"], window=20)
                except Exception as e:
                    logger.warning(f"VectorBT volume indicators failed: {e}, using pandas fallback")
                    volume_mean_20 = data["volume"].rolling(window=20).mean()
            else:
                volume_mean_20 = data["volume"].rolling(window=20).mean()
            
            indicators.extend([
                data["volume"] / volume_mean_20,
                data["volume"].pct_change(),
            ])

        # Stack features
        feature_matrix = np.column_stack([ind.fillna(0).values for ind in indicators])

        # Ensure we have enough data
        if feature_matrix.shape[0] > self.sequence_length:
            feature_matrix = feature_matrix[-self.sequence_length:]

        return feature_matrix

    def _encode_features(self, features: np.ndarray) -> np.ndarray:
        """Encode features using autoencoder (simplified implementation)."""
        # Simple dimensionality reduction using PCA as autoencoder proxy
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.embedding_dim)
            encoded = pca.fit_transform(features)
            return encoded
        except ImportError:
            # Fallback to simple linear transformation
            return features[:, :self.embedding_dim] if features.shape[1] >= self.embedding_dim else features

warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

            # Reshape for time series encoding
            seq_len, feature_dim = features.shape

            # Flatten for PCA
            flattened = features.flatten().reshape(1, -1)

            # Apply PCA for dimensionality reduction
            if feature_dim >= self.encoding_dim:
                pca = PCA(n_components=self.encoding_dim)
                encoded = pca.fit_transform(flattened)
            else:
                # Pad if needed
                encoded = np.pad(flattened, ((0, 0), (0, self.encoding_dim - feature_dim)), mode='constant')

            # Reshape back to sequence format
            encoded_reshaped = encoded.reshape(seq_len, -1)

            return encoded_reshaped

        except ImportError:
            # Fallback to simple averaging
            return features.mean(axis=1, keepdims=True)


    

class ContrastiveLearningGenerator(FeatureGenerator):
    """Generator for contrastive learning representations."""

    def __init__(self, embedding_dim: int = 64, temperature: float = 0.1):
        config = FeatureConfig(
            name=f"contrastive_repr_{embedding_dim}_{temperature}",
            category=FeatureCategory.AUTOENCODER,
            description=f"Contrastive learning representation with embedding_dim={embedding_dim}, temperature={temperature}",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=100,
            min_lookback=50,
            max_lookback=200,
            parameters={
                "embedding_dim": embedding_dim,
                "temperature": temperature
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.embedding_dim = embedding_dim
        self.temperature = temperature

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate contrastive learning representation features."""
        try:
            # Create positive and negative samples
            positive_samples, negative_samples = self._create_contrastive_samples(data)

            # Learn representations through contrastive loss (simplified)
            representations = self._contrastive_learning(positive_samples, negative_samples)

            return pd.Series(representations, index=data.index[-len(representations):])

        except Exception as e:
            logger.warning(f"Error in contrastive learning generation: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

    def _create_contrastive_samples(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create positive and negative samples for contrastive learning."""
        # Extract price sequence
        price_sequence = data["close"].values

        # Create overlapping windows as positive pairs
        window_size = 20
        stride = 5

        positive_samples = []
        for i in range(0, len(price_sequence) - window_size * 2, stride):
            # Positive pair: consecutive windows
            window1 = price_sequence[i:i+window_size]
            window2 = price_sequence[i+stride:i+stride+window_size]

            if len(window1) == window_size and len(window2) == window_size:
                positive_samples.append((window1, window2))

        # Negative samples: random windows
        negative_samples = []
        num_negative = len(positive_samples)

        for _ in range(num_negative):
            # Randomly sample two different windows
            idx1, idx2 = np.random.choice(len(price_sequence) - window_size, 2, replace=False)
            neg1 = price_sequence[idx1:idx1+window_size]
            neg2 = price_sequence[idx2:idx2+window_size]
            negative_samples.append((neg1, neg2))

        return np.array(positive_samples), np.array(negative_samples)

    def _contrastive_learning(self, positive_samples: np.ndarray, negative_samples: np.ndarray) -> np.ndarray:
        """Apply contrastive learning (simplified)."""
        # Simple contrastive learning using correlation as similarity

        representations = []

        for pos_pair, neg_pair in zip(positive_samples, negative_samples):
            # Calculate similarities
            pos_sim = np.corrcoef(pos_pair[0], pos_pair[1])[0, 1]
            neg_sim = np.corrcoef(neg_pair[0], neg_pair[1])[0, 1]

            # Contrastive representation (push positive pairs together, negative apart)
            contrastive_repr = pos_sim - neg_sim
            representations.append(contrastive_repr)

        return np.array(representations)
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
