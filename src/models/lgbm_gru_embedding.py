"""
LGBM + Small GRU Embedding Model for Tactician

This module implements the LGBM + small GRU as embedding model for the tactician
with the following specifications:
- Lookback: 2-4h (configurable)
- Hidden size: 32-64 (configurable)
- Layers: 1
- Dropout: ≤0.1
- Export: last-hidden → PCA to 8-12 dims (fit on train only)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class LGBMGRUConfig:
    """Configuration for LGBM + GRU embedding model."""
    # GRU parameters
    lookback_hours: int = 3  # 2-4h range
    hidden_size: int = 48  # 32-64 range
    num_layers: int = 1
    dropout: float = 0.05  # ≤0.1
    pca_dims: int = 10  # 8-12 range
    fit_pca_on_train_only: bool = True
    
    # LGBM parameters (updated hyperparameters)
    max_depth: int = 3  # 3-4 range
    num_leaves: int = 12  # 8-16 range
    min_child_samples: int = 800  # 600-1000 range
    lambda_l2: float = 30.0  # 10-50 range
    feature_fraction: float = 0.7  # 0.6-0.8 range
    learning_rate: float = 0.05
    n_estimators: int = 500
    
    # Training parameters
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = -1


class SimpleGRU(nn.Module):
    """Simple GRU implementation for embedding generation."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through GRU."""
        # x shape: (batch_size, seq_len, input_size)
        output, hidden = self.gru(x)
        
        # Use the last hidden state
        # hidden shape: (num_layers, batch_size, hidden_size)
        last_hidden = hidden[-1]  # Take the last layer's hidden state
        
        # Apply dropout
        last_hidden = self.dropout(last_hidden)
        
        return last_hidden


class LGBMGRUEmbedding(BaseEstimator, RegressorMixin):
    """
    LGBM + Small GRU Embedding Model for Tactician.
    
    This model combines:
    1. A small GRU to generate embeddings from sequential data
    2. PCA dimensionality reduction on GRU embeddings (fitted on train only)
    3. LightGBM trained on original features + GRU embeddings
    """
    
    def __init__(self, config: Optional[LGBMGRUConfig] = None):
        """Initialize the LGBM + GRU embedding model."""
        self.config = config or LGBMGRUConfig()
        
        # Components
        self.gru_model = None
        self.pca = None
        self.scaler = None
        self.lgbm_model = None
        
        # State
        self.fitted = False
        self.feature_names = None
        
    def _prepare_sequences(self, X: np.ndarray, lookback_bars: int) -> np.ndarray:
        """Prepare sequences for GRU input."""
        try:
            sequences = []
            for i in range(lookback_bars, len(X)):
                sequence = X[i-lookback_bars:i]
                sequences.append(sequence)
            
            if not sequences:
                # If no sequences can be created, pad the data
                padded_X = np.zeros((lookback_bars, X.shape[1]))
                padded_X[-len(X):] = X
                return padded_X.reshape(1, lookback_bars, -1)
            
            return np.array(sequences)
        except Exception as e:
            logger.warning(f"⚠️ Sequence preparation failed: {e}")
            # Fallback: create single sequence with padding
            if len(X) < lookback_bars:
                padded_X = np.zeros((lookback_bars, X.shape[1]))
                padded_X[-len(X):] = X
                return padded_X.reshape(1, lookback_bars, -1)
            return X[-lookback_bars:].reshape(1, lookback_bars, -1)
    
    def _create_gru_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Create GRU embeddings from sequential data."""
        try:
            import torch
            import torch.nn as nn
            
            # Calculate lookback in bars (assuming 5m bars for tactician)
            # 1 hour = 12 bars (5m each), so 3 hours = 36 bars
            lookback_bars = self.config.lookback_hours * 12
            
            # Prepare sequences
            sequences = self._prepare_sequences(X, lookback_bars)
            
            if sequences.size == 0:
                return np.zeros((X.shape[0], self.config.hidden_size))
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(sequences)
            
            # Create GRU model if not exists
            if self.gru_model is None:
                self.gru_model = SimpleGRU(
                    input_size=X.shape[1],
                    hidden_size=self.config.hidden_size,
                    num_layers=self.config.num_layers,
                    dropout=self.config.dropout
                )
            
            # Generate embeddings
            self.gru_model.eval()
            with torch.no_grad():
                embeddings = self.gru_model(X_tensor)
                embeddings = embeddings.numpy()
            
            # Ensure we have the right number of embeddings
            if len(embeddings) < X.shape[0]:
                # Pad with the last embedding
                padding = np.tile(embeddings[-1:], (X.shape[0] - len(embeddings), 1))
                embeddings = np.vstack([embeddings, padding])
            elif len(embeddings) > X.shape[0]:
                # Truncate to match
                embeddings = embeddings[:X.shape[0]]
            
            return embeddings
            
        except ImportError:
            logger.warning("⚠️ PyTorch not available, using fallback linear embedding")
            return self._create_fallback_embeddings(X)
        except Exception as e:
            logger.warning(f"⚠️ GRU embedding creation failed: {e}")
            return self._create_fallback_embeddings(X)
    
    def _create_fallback_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Fallback embedding creation using simple linear transformation."""
        try:
            # Simple linear transformation as fallback
            np.random.seed(self.config.random_state)
            W = np.random.randn(X.shape[1], self.config.hidden_size) * 0.1
            embeddings = X @ W
            return embeddings
        except Exception as e:
            logger.warning(f"⚠️ Fallback embedding creation failed: {e}")
            return np.zeros((X.shape[0], self.config.hidden_size))
    
    def _apply_pca_reduction(self, embeddings: np.ndarray, fit: bool = False) -> np.ndarray:
        """Apply PCA reduction to GRU embeddings."""
        try:
            if fit:
                # Fit PCA on training data only
                self.pca = PCA(n_components=min(self.config.pca_dims, embeddings.shape[1]))
                reduced_embeddings = self.pca.fit_transform(embeddings)
            else:
                # Transform using fitted PCA
                if self.pca is None:
                    logger.warning("⚠️ PCA not fitted, using original embeddings")
                    return embeddings
                reduced_embeddings = self.pca.transform(embeddings)
            
            return reduced_embeddings
            
        except Exception as e:
            logger.warning(f"⚠️ PCA reduction failed: {e}")
            return embeddings
    
    def _combine_features(self, X: np.ndarray, gru_embeddings: np.ndarray) -> np.ndarray:
        """Combine original features with GRU embeddings."""
        try:
            # Scale features if scaler is available
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X
            
            # Combine original features with GRU embeddings
            combined_features = np.hstack([X_scaled, gru_embeddings])
            
            return combined_features
            
        except Exception as e:
            logger.warning(f"⚠️ Feature combination failed: {e}")
            return X
    
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'LGBMGRUEmbedding':
        """Fit the LGBM + GRU embedding model."""
        try:
            # Store feature names if available
            if hasattr(X, 'columns'):
                self.feature_names = list(X.columns)
                X = X.values
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Create GRU embeddings
            gru_embeddings = self._create_gru_embeddings(X_scaled)
            
            # Apply PCA reduction (fit on train only)
            if self.config.fit_pca_on_train_only:
                gru_embeddings_reduced = self._apply_pca_reduction(gru_embeddings, fit=True)
            else:
                gru_embeddings_reduced = gru_embeddings
            
            # Combine features
            X_combined = self._combine_features(X_scaled, gru_embeddings_reduced)
            
            # Train LightGBM
            import lightgbm as lgb
            
            self.lgbm_model = lgb.LGBMRegressor(
                max_depth=self.config.max_depth,
                num_leaves=self.config.num_leaves,
                min_child_samples=self.config.min_child_samples,
                reg_lambda=self.config.lambda_l2,
                feature_fraction=self.config.feature_fraction,
                learning_rate=self.config.learning_rate,
                n_estimators=self.config.n_estimators,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                verbose=self.config.verbose
            )
            
            if sample_weight is not None:
                self.lgbm_model.fit(X_combined, y, sample_weight=sample_weight)
            else:
                self.lgbm_model.fit(X_combined, y)
            
            self.fitted = True
            logger.info(f"✅ LGBM + GRU embedding model fitted with {X.shape[1]} original features + {gru_embeddings_reduced.shape[1]} GRU embeddings")
            
            return self
            
        except Exception as e:
            logger.error(f"❌ LGBM + GRU embedding model fitting failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted model."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Convert to numpy if pandas DataFrame
            if hasattr(X, 'values'):
                X = X.values
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Create GRU embeddings
            gru_embeddings = self._create_gru_embeddings(X_scaled)
            
            # Apply PCA reduction
            gru_embeddings_reduced = self._apply_pca_reduction(gru_embeddings, fit=False)
            
            # Combine features
            X_combined = self._combine_features(X_scaled, gru_embeddings_reduced)
            
            # Make predictions
            predictions = self.lgbm_model.predict(X_combined)
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ LGBM + GRU embedding model prediction failed: {e}")
            raise
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance from the LightGBM model."""
        if not self.fitted or self.lgbm_model is None:
            return np.array([])
        
        try:
            return self.lgbm_model.feature_importances_
        except Exception as e:
            logger.warning(f"⚠️ Could not get feature importance: {e}")
            return np.array([])
    
    def get_gru_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Get GRU embeddings for analysis."""
        if not self.fitted:
            raise ValueError("Model must be fitted before getting embeddings")
        
        try:
            # Convert to numpy if pandas DataFrame
            if hasattr(X, 'values'):
                X = X.values
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Create GRU embeddings
            gru_embeddings = self._create_gru_embeddings(X_scaled)
            
            # Apply PCA reduction
            gru_embeddings_reduced = self._apply_pca_reduction(gru_embeddings, fit=False)
            
            return gru_embeddings_reduced
            
        except Exception as e:
            logger.error(f"❌ GRU embedding extraction failed: {e}")
            return np.zeros((X.shape[0], self.config.pca_dims))


# Factory function
def create_lgbm_gru_embedding(config: Optional[LGBMGRUConfig] = None) -> LGBMGRUEmbedding:
    """Create LGBM + GRU embedding model."""
    return LGBMGRUEmbedding(config)