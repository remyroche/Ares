"""
TabR (Tabular Regression) Model for Financial Time Series

This module provides a TabR regressor implementation for financial time series tasks.
TabR is an efficient tabular regression model that uses context retrieval
and attention mechanisms for time series forecasting.

Features:
- Scikit-learn compatible API (fit/predict)
- Context-based learning with k-nearest neighbors
- Efficient tabular processing with Mambular library
- Configurable architecture parameters
- Early stopping and validation support
- Comprehensive error handling
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Any, Dict
import logging

try:
    from mambular.models import TabRRegressor as MambularTabRRegressor
    MAMBULAR_AVAILABLE = True
except ImportError:
    MAMBULAR_AVAILABLE = False
    MambularTabRRegressor = None

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success

logger = system_logger.getChild('TabRRegressor')

class TabRRegressor(BaseEstimator, RegressorMixin):
    """
    TabR (Tabular Regression) model with sklearn compatibility.

    This implementation uses Mambular library's TabR model for efficient
    tabular regression with context-based learning. It's designed as a
    replacement for DepthwiseSeparableCNN in financial time series tasks.

    Architecture:
    - Context retrieval with k-nearest neighbors
    - Linear encoder (TabR-S style for simplicity)
    - Single predictor block
    - Embedding-based feature representation

    Parameters:
    -----------
    k_neighbors : int, default=96
        Number of nearest neighbors for context retrieval

    use_embeddings : bool, default=False
        Whether to use embeddings (TabR-S style when False)

    n_encoder_layers : int, default=0
        Number of encoder layers (0 for linear/TabR-S style)

    n_predictor_layers : int, default=1
        Number of predictor layers

    d_embedding : int, default=64
        Embedding dimension for feature representation

    learning_rate : float, default=1e-4
        Learning rate for optimization

    weight_decay : float, default=1e-6
        Weight decay (L2 regularization)

    batch_size : int, default=256
        Batch size for training

    max_epochs : int, default=200
        Maximum number of training epochs

    early_stopping_patience : int, default=15
        Patience for early stopping

    lr_scheduler_patience : int, default=10
        Patience for learning rate scheduler

    dropout : float, default=0.0
        Dropout rate for regularization

    random_state : int, default=42
        Random seed for reproducibility

    verbose : int, default=0
        Verbosity level (0=silent, 1=progress, 2=detailed)

    Attributes:
    -----------
    model_ : fitted TabR model
        The underlying Mambular TabR model

    scaler_ : StandardScaler
        Feature scaler fitted during training

    is_fitted_ : bool
        Whether model has been fitted

    n_features_in_ : int
        Number of features seen during fitting
    """

    def __init__(
        self,
        k_neighbors: int = 96,
        use_embeddings: bool = False,
        n_encoder_layers: int = 0,
        n_predictor_layers: int = 1,
        d_embedding: int = 64,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        batch_size: int = 256,
        max_epochs: int = 200,
        early_stopping_patience: int = 15,
        lr_scheduler_patience: int = 10,
        dropout: float = 0.0,
        random_state: int = 42,
        verbose: int = 0
    ):
        """Initialize TabR regressor."""
        if not MAMBULAR_AVAILABLE:
            raise ImportError(
                "Mambular library is required for TabRRegressor. "
                "Install with: pip install mambular"
            )

        self.k_neighbors = k_neighbors
        self.use_embeddings = use_embeddings
        self.n_encoder_layers = n_encoder_layers
        self.n_predictor_layers = n_predictor_layers
        self.d_embedding = d_embedding
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.lr_scheduler_patience = lr_scheduler_patience
        self.dropout = dropout
        self.random_state = random_state
        self.verbose = verbose

        # Model components (initialized during fit)
        self.model_ = None
        self.scaler_ = None
        self.is_fitted_ = False
        self.n_features_in_ = None

    def _build_model(self) -> MambularTabRRegressor:
        """Build TabR model with specified parameters."""
        try:
            model = MambularTabRRegressor(
                k=self.k_neighbors,
                lr=self.learning_rate,
                lr_patience=self.lr_scheduler_patience,
                weight_decay=self.weight_decay,
                n_layers_encoder=self.n_encoder_layers,
                n_layers_predictor=self.n_predictor_layers,
                d_model=self.d_embedding,
                batch_size=self.batch_size,
                max_epochs=self.max_epochs,
                patience=self.early_stopping_patience,
                use_embeddings=self.use_embeddings,
                dropout=self.dropout,
                seed=self.random_state,
                verbose=self.verbose
            )
            return model
        except Exception as e:
            logger.error(f"Failed to build TabR model: {e}")
            raise

    def fit(self, X_data, y_data, **fit_params):
        """
        Fit TabR model to training data.

        Args:
            X_data: Training features (n_samples, n_features)
            y_data: Target values (n_samples,)
            **fit_params: Additional parameters for model.fit()

        Returns:
            self: Fitted estimator
        """
        # Validate input
        X_data, y_data = check_X_y(X_data, y_data)
        self.n_features_in_ = X_data.shape[1]

        # Set random seed for reproducibility
        np.random.seed(self.random_state)

        # Scale features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_data)

        if self.verbose > 0:
            tprint_info(f"🚀 Training TabR model...")
            tprint_info(f"   Features: {X_data.shape[1]}")
            tprint_info(f"   Samples: {X_data.shape[0]}")
            tprint_info(f"   K-neighbors: {self.k_neighbors}")
            tprint_info(f"   Embedding dim: {self.d_embedding}")
            tprint_info(f"   Max epochs: {self.max_epochs}")

        # Build and train model
        self.model_ = self._build_model()

        # Fit model
        try:
            self.model_.fit(X_scaled, y_data, **fit_params)
            self.is_fitted_ = True

            if self.verbose > 0:
                tprint_success(f"✅ TabR model trained successfully")
            return self

        except Exception as e:
            logger.error(f"TabR model fitting failed: {e}")
            raise

    def predict(self, X_data):
        """
        Make predictions using the fitted TabR model.

        Args:
            X_data: Features (n_samples, n_features)

        Returns:
            Predictions (n_samples,)
        """
        # Validate fitted
        if not self.is_fitted_ or self.model_ is None:
            raise ValueError("Model must be fitted before prediction")

        # Validate input
        X_data = check_array(X_data)

        if X_data.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_data.shape[1]} features, but model expects {self.n_features_in_}"
            )

        # Scale features
        X_scaled = self.scaler_.transform(X_data)

        # Make predictions
        try:
            predictions = self.model_.predict(X_scaled)
            return predictions
        except Exception as e:
            logger.error(f"TabR prediction failed: {e}")
            raise

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for sklearn compatibility."""
        return {
            'k_neighbors': self.k_neighbors,
            'use_embeddings': self.use_embeddings,
            'n_encoder_layers': self.n_encoder_layers,
            'n_predictor_layers': self.n_predictor_layers,
            'd_embedding': self.d_embedding,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'max_epochs': self.max_epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'lr_scheduler_patience': self.lr_scheduler_patience,
            'dropout': self.dropout,
            'random_state': self.random_state,
            'verbose': self.verbose
        }

    def set_params(self, **params) -> 'TabRRegressor':
        """Set parameters for sklearn compatibility."""
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                logger.warning(f"Unknown parameter: {param}")
        return self

def create_tabr_regressor(
    k_neighbors: int = 96,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-6,
    n_layers_encoder: int = 0,
    n_layers_predictor: int = 1,
    d_model: int = 64,
    batch_size: int = 256,
    max_epochs: int = 200,
    patience: int = 15,
    lr_patience: int = 10,
    **kwargs
) -> TabRRegressor:
    """
    Factory function to create TabR regressor with sensible defaults.

    Args:
        k_neighbors: Number of nearest neighbors for context (default: 96)
        learning_rate: Learning rate (default: 1e-4)
        weight_decay: Weight decay (default: 1e-6)
        n_layers_encoder: Number of encoder layers (default: 0)
        n_layers_predictor: Number of predictor layers (default: 1)
        d_model: Model dimension (default: 64)
        batch_size: Batch size (default: 256)
        max_epochs: Maximum epochs (default: 200)
        patience: Early stopping patience (default: 15)
        lr_patience: LR scheduler patience (default: 10)
        **kwargs: Additional parameters

    Returns:
        TabRRegressor instance
    """
    return TabRRegressor(
        k_neighbors=k_neighbors,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        n_encoder_layers=n_layers_encoder,
        n_predictor_layers=n_layers_predictor,
        d_embedding=d_model,
        batch_size=batch_size,
        max_epochs=max_epochs,
        early_stopping_patience=patience,
        lr_scheduler_patience=lr_patience,
        **kwargs
    )

# Backward compatibility aliases
DepthwiseSeparableCNNRegressor = TabRRegressor
TCNRegressor = TabRRegressor

def create_depthwise_cnn_regressor(**kwargs):
    """Factory function for backward compatibility - creates TabR regressor."""
    tprint_warning("⚠️ DepthwiseSeparableCNN is deprecated, using TabR instead")
    return create_tabr_regressor(**kwargs)

def create_tcn_regressor(**kwargs):
    """Factory function for backward compatibility - creates TabR regressor."""
    tprint_warning("⚠️ TCNRegressor is deprecated, using TabR instead")
    return create_tabr_regressor(**kwargs)

# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate synthetic tabular data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    # Create synthetic features with temporal patterns
    X_data = np.random.randn(n_samples, n_features)
    # Add temporal autocorrelation
    for i in range(1, n_samples):
        X_data[i] = 0.7 * X_data[i-1] + 0.3 * X_data[i]

    # Create target with feature dependence
    y_data = np.sum(X_data[:, :5], axis=1) + 0.5 * np.random.randn(n_samples)

    # Split data
    split = int(0.8 * n_samples)
    X_train, X_test = X_data[:split], X_data[split:]
    y_train, y_test = y_data[:split], y_data[split:]

    print("🚀 Testing TabR Regressor")
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")

    # Create and train model
    tabr = TabRRegressor(
        k_neighbors=96,
        lr=1e-4,
        lr_patience=10,
        weight_decay=1e-6,
        n_layers_encoder=0,       # TabR-S config
        n_layers_predictor=1,
        d_model=64,
        batch_size=256,
        max_epochs=200,
        patience=15,
        verbose=1
    )

    print("\n📊 Training TabR model...")
    tabr.fit(X_train, y_train)

    # Make predictions
    y_pred = tabr.predict(X_test)

    # Calculate metrics
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

    print(f"\n✅ Test Results:")
    print(f"   MSE: {mse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   R²: {r2:.4f}")

    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('TabR Predictions vs Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('tabr_predictions.png')
    print(f"\n📊 Prediction plot saved to tabr_predictions.png")
