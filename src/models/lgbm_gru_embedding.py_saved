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
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin

from src.training.utils.embedding_postprocessing import filter_embedding_features

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
    cv_folds: int = 5
    corr_threshold: float = 0.8
    ic_threshold: float = 0.05
    min_embeddings: int = 6
    max_embeddings: int = 10
    oof_output_dir: Optional[str] = None

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

class LGBMGRUEmbedding(BaseEstimator, ClassifierMixin):
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
        self.oof_embeddings: Optional[np.ndarray] = None
        self.oof_embeddings_per_fold: List[Dict[str, Any]] = []
        self.oof_selected_indices: Optional[List[int]] = None
        self.embedding_filter_metadata: Dict[str, Any] = {}
        self.embedding_feature_names: Optional[List[str]] = None

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

    def _create_gru_embeddings(self, X: np.ndarray, gru_model: Optional[nn.Module] = None) -> np.ndarray:
        """Create GRU embeddings from sequential data."""
        try:
            # Calculate lookback in bars (assuming 5m bars for tactician)
            # 1 hour = 12 bars (5m each), so 3 hours = 36 bars
            lookback_bars = self.config.lookback_hours * 12

            # Prepare sequences
            sequences = self._prepare_sequences(X, lookback_bars)

            if sequences.size == 0:
                return np.zeros((X.shape[0], self.config.hidden_size))

            # Convert to tensors
            X_tensor = torch.FloatTensor(sequences)

            # Create or reuse GRU model
            if gru_model is None:
                if self.gru_model is None:
                    self.gru_model = SimpleGRU(
                        input_size=X.shape[1],
                        hidden_size=self.config.hidden_size,
                        num_layers=self.config.num_layers,
                        dropout=self.config.dropout
                    )
                gru_model = self.gru_model

            gru_model.eval()
            with torch.no_grad():
                embeddings = gru_model(X_tensor)
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

    def _generate_oof_embeddings(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Generate out-of-fold embeddings using time-series cross-validation."""

        n_samples = X.shape[0]
        if n_samples < 3 or self.config.cv_folds < 2:
            logger.warning("⚠️ Not enough samples for OOF embedding generation")
            return None

        n_splits = min(self.config.cv_folds, n_samples - 1)
        if n_splits < 2:
            logger.warning("⚠️ Unable to configure TimeSeriesSplit for OOF embeddings")
            return None

        try:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            embed_dim = min(self.config.pca_dims, self.config.hidden_size)
            oof_embeddings = np.full((n_samples, embed_dim), np.nan, dtype=float)
            assigned_mask = np.zeros(n_samples, dtype=bool)
            fold_records: List[Dict[str, Any]] = []
            y_array = y.ravel() if y is not None else None

            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X[train_idx])
                X_val_scaled = scaler.transform(X[val_idx])

                fold_gru = SimpleGRU(
                    input_size=X.shape[1],
                    hidden_size=self.config.hidden_size,
                    num_layers=self.config.num_layers,
                    dropout=self.config.dropout
                )

                train_embeddings = self._create_gru_embeddings(X_train_scaled, gru_model=fold_gru)
                pca = PCA(n_components=min(self.config.pca_dims, train_embeddings.shape[1]))
                pca.fit(train_embeddings)

                val_embeddings = self._create_gru_embeddings(X_val_scaled, gru_model=fold_gru)
                val_reduced = pca.transform(val_embeddings)

                # Align shapes
                if val_reduced.shape[1] < embed_dim:
                    padding = np.zeros((val_reduced.shape[0], embed_dim - val_reduced.shape[1]))
                    val_reduced = np.hstack([val_reduced, padding])
                elif val_reduced.shape[1] > embed_dim:
                    val_reduced = val_reduced[:, :embed_dim]

                oof_embeddings[val_idx] = val_reduced
                assigned_mask[val_idx] = True

                fold_info = {
                    'fold': fold_idx,
                    'train_indices': train_idx.tolist(),
                    'val_indices': val_idx.tolist(),
                    'embedding_shape': val_reduced.shape
                }
                fold_records.append(fold_info)

                if self.config.oof_output_dir:
                    output_dir = Path(self.config.oof_output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    np.save(output_dir / f'gru_oof_fold_{fold_idx}.npy', val_reduced)

            # Fill any unassigned rows with the earliest available embedding
            valid_rows = np.where(assigned_mask)[0]
            if valid_rows.size == 0:
                logger.warning("⚠️ OOF embedding generation produced no validation folds")
                return None

            first_valid_embedding = oof_embeddings[valid_rows[0]]
            for idx in range(n_samples):
                if not assigned_mask[idx]:
                    oof_embeddings[idx] = first_valid_embedding

            embedding_names = [f'gru_oof_{i}' for i in range(oof_embeddings.shape[1])]
            parent_names = self.feature_names if self.feature_names else None

            filtered_embeddings, filter_metadata = filter_embedding_features(
                parent_features=X,
                embedding_features=oof_embeddings,
                target=y_array,
                parent_feature_names=parent_names,
                embedding_names=embedding_names,
                corr_threshold=self.config.corr_threshold,
                ic_threshold=self.config.ic_threshold,
                min_embeddings=self.config.min_embeddings,
                max_embeddings=self.config.max_embeddings
            )

            self.oof_embeddings = filtered_embeddings
            self.oof_embeddings_per_fold = fold_records
            self.oof_selected_indices = filter_metadata.get('selected_indices')
            self.embedding_filter_metadata = filter_metadata
            self.embedding_feature_names = filter_metadata.get('retained_embedding_names')

            if not filter_metadata.get('within_budget', False):
                logger.warning(
                    "⚠️ Filtered GRU embeddings are outside the configured embedding budget"
                )

            return filtered_embeddings

        except Exception as e:
            logger.warning(f"⚠️ Failed to generate OOF embeddings: {e}")
            return None

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

            # Generate out-of-fold embeddings prior to downstream training
            self._generate_oof_embeddings(X, y)

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

            # Apply embedding post-processing / filtering
            if self.oof_selected_indices is not None:
                gru_embeddings_filtered = gru_embeddings_reduced[:, self.oof_selected_indices]
            else:
                embedding_names = [f'gru_pca_{i}' for i in range(gru_embeddings_reduced.shape[1])]
                parent_names = self.feature_names if self.feature_names else None
                gru_embeddings_filtered, filter_metadata = filter_embedding_features(
                    parent_features=X,
                    embedding_features=gru_embeddings_reduced,
                    target=y,
                    parent_feature_names=parent_names,
                    embedding_names=embedding_names,
                    corr_threshold=self.config.corr_threshold,
                    ic_threshold=self.config.ic_threshold,
                    min_embeddings=self.config.min_embeddings,
                    max_embeddings=self.config.max_embeddings
                )
                self.embedding_filter_metadata = filter_metadata
                self.oof_selected_indices = filter_metadata.get('selected_indices')
                self.embedding_feature_names = filter_metadata.get('retained_embedding_names')

            if self.embedding_feature_names is None and self.oof_selected_indices is not None:
                self.embedding_feature_names = [
                    f'gru_pca_{idx}' for idx in self.oof_selected_indices
                ]

            if gru_embeddings_filtered.size == 0:
                logger.warning("⚠️ No GRU embeddings retained after filtering; reverting to leading components")
                fallback_count = min(
                    self.config.max_embeddings,
                    max(self.config.min_embeddings, gru_embeddings_reduced.shape[1])
                )
                gru_embeddings_filtered = gru_embeddings_reduced[:, :fallback_count]
                self.oof_selected_indices = list(range(fallback_count))
                self.embedding_feature_names = [
                    f'gru_fallback_{i}' for i in range(fallback_count)
                ]
                if self.oof_embeddings is not None and self.oof_embeddings.shape[1] >= fallback_count:
                    self.oof_embeddings = self.oof_embeddings[:, :fallback_count]
                else:
                    self.oof_embeddings = np.zeros((X.shape[0], fallback_count))
                self.embedding_filter_metadata['retained_count'] = fallback_count
                self.embedding_filter_metadata['within_budget'] = (
                    self.config.min_embeddings <= fallback_count <= self.config.max_embeddings
                )

            # Combine features
            X_combined = self._combine_features(X, gru_embeddings_filtered)

            # Train LightGBM
            import lightgbm as lgb

            self.lgbm_model = lgb.LGBMClassifier(
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
            logger.info(
                "✅ LGBM + GRU embedding model fitted with %d original features + %d GRU embeddings",
                X.shape[1],
                X_combined.shape[1] - X.shape[1]
            )

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

            if self.oof_selected_indices is not None:
                gru_embeddings_filtered = gru_embeddings_reduced[:, self.oof_selected_indices]
            else:
                gru_embeddings_filtered = gru_embeddings_reduced

            # Combine features
            X_combined = self._combine_features(X, gru_embeddings_filtered)

            # Make predictions (return positive class probability)
            proba = self.lgbm_model.predict_proba(X_combined)
            if proba.ndim == 2 and proba.shape[1] > 1:
                predictions = proba[:, 1]
            else:
                predictions = proba.ravel()

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
            if self.oof_selected_indices is not None:
                return gru_embeddings_reduced[:, self.oof_selected_indices]

            return gru_embeddings_reduced

        except Exception as e:
            logger.error(f"❌ GRU embedding extraction failed: {e}")
            return np.zeros((X.shape[0], self.config.pca_dims))

# Factory function
def create_lgbm_gru_embedding(config: Optional[LGBMGRUConfig] = None) -> LGBMGRUEmbedding:
    """Create LGBM + GRU embedding model."""
    return LGBMGRUEmbedding(config)
