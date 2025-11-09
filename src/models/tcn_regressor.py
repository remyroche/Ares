"""
Temporal Convolutional Network (TCN) Regressor

A reusable scikit-learn compatible TCN model for time series regression tasks.
This module provides a TCN implementation that can be used across multiple training pipelines.

Features:
- Scikit-learn compatible API (fit/predict)
- Automatic feature scaling
- Configurable architecture (filters, kernel size, dropout)
- GPU acceleration support (TensorFlow backend)
- Built-in early stopping and validation
- Comprehensive error handling
"""

import numpy as np
from typing import Optional, Tuple, Any
import logging

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, Dense, Dropout, GlobalMaxPooling1D, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    import tensorflow as tf
    from tensorflow.keras.layers import Conv1D, Dense, Dropout, GlobalMaxPooling1D, BatchNormalization, SeparableConv1D
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    Sequential = None
    tf = None

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success

logger = system_logger.getChild('DepthwiseSeparableCNNRegressor')

class DepthwiseSeparableCNNRegressor(BaseEstimator, RegressorMixin):
    """
    Depthwise Separable 1D CNN Regressor with sklearn compatibility.

    This implementation uses Depthwise Separable 1D convolutions for efficient
    temporal pattern recognition, offering a lightweight alternative to a standard TCN.
    It is suitable for time series regression tasks in trading strategies.

    Architecture:
    - SeparableConv1D layer (configurable filters)
    - Dropout
    - SeparableConv1D layer (2x filters)
    - Dropout
    - GlobalMaxPooling1D
    - Dense layer (50 units)
    - Dropout
    - Output layer (1 unit, linear)

    Parameters:
    -----------
    filters : int, default=64
        Number of filters in first Conv1D layer

    kernel_size : int, default=3
        Size of convolution kernel

    dropout : float, default=0.2
        Dropout rate for regularization

    epochs : int, default=50
        Number of training epochs

    batch_size : int, default=32
        Batch size for training

    learning_rate : float, default=0.001
        Learning rate for Adam optimizer

    validation_split : float, default=0.2
        Fraction of training data to use for validation

    early_stopping_patience : int, default=10
        Number of epochs with no improvement before stopping

    reduce_lr_patience : int, default=5
        Number of epochs with no improvement before reducing learning rate

    verbose : int, default=0
        Verbosity level (0=silent, 1=progress, 2=detailed)

    random_state : int, default=42
        Random seed for reproducibility

    use_batch_norm : bool, default=False
        Whether to use batch normalization
    """

    def __init__(
        self,
        filters: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        reduce_lr_patience: int = 5,
        verbose: int = 0,
        random_state: int = 42,
        use_batch_norm: bool = False
    ):
        """Initialize CNN Regressor."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for DepthwiseSeparableCNNRegressor. "
                "Install with: pip install tensorflow>=2.8.0"
            )

        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.verbose = verbose
        self.random_state = random_state
        self.use_batch_norm = use_batch_norm

        # Model components (initialized during fit)
        self.model_ = None
        self.scaler_ = None
        self.history_ = None
        self.n_features_in_ = None
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build Depthwise Separable CNN architecture.
    
        Args:
            input_shape: Shape of input (timesteps, features)
    
        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential()
    
        # First convolutional block
        model.add(SeparableConv1D(
            self.filters,
            self.kernel_size,
            activation='relu',
            input_shape=input_shape,
            padding='same',
            depthwise_initializer='glorot_uniform',
            pointwise_initializer='glorot_uniform'
        ))
        if self.use_batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(self.dropout))
    
        # Second convolutional block (double filters for hierarchical features)
        model.add(SeparableConv1D(
            self.filters * 2,
            self.kernel_size,
            activation='relu',
            padding='same',
            depthwise_initializer='glorot_uniform',
            pointwise_initializer='glorot_uniform'
        ))
        if self.use_batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(self.dropout))
    
        # Global pooling and dense layers
        model.add(GlobalMaxPooling1D())
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(1, activation='linear'))
    
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
    
        return model

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params) -> 'DepthwiseSeparableCNNRegressor':
        """
        Fit TCN model.

        Args:
            X: Training features (n_samples, n_features)
            y: Target values (n_samples,)
            **fit_params: Additional parameters for model.fit()

        Returns:
            self: Fitted estimator
        """
        # Validate input
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        # Set random seed for reproducibility
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

        # Reshape X for Conv1D (samples, timesteps, features)
        if len(X.shape) == 2:
            X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        else:
            X_reshaped = X

        # Scale features
        self.scaler_ = StandardScaler()
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled_flat = self.scaler_.fit_transform(X_flat)
        X_scaled = X_scaled_flat.reshape(X_reshaped.shape)

        # Build model
        self.model_ = self._build_model(
            input_shape=(X_reshaped.shape[1], X_reshaped.shape[2])
        )

        # Setup callbacks
        callbacks = []

        # Early stopping
        if self.early_stopping_patience > 0:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                verbose=self.verbose
            )
            callbacks.append(early_stopping)

        # Learning rate reduction
        if self.reduce_lr_patience > 0:
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.reduce_lr_patience,
                min_lr=1e-7,
                verbose=self.verbose
            )
            callbacks.append(reduce_lr)

        # Train model
        self.history_ = self.model_.fit(
            X_scaled,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=callbacks if callbacks else None,
            verbose=self.verbose,
            **fit_params
        )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predictions (n_samples,)
        """
        # Validate fitted
        if self.model_ is None or self.scaler_ is None:
            raise ValueError("Model must be fitted before prediction")

        # Validate input
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model expects {self.n_features_in_}"
            )

        # Reshape X for Conv1D
        if len(X.shape) == 2:
            X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        else:
            X_reshaped = X

        # Scale features
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled_flat = self.scaler_.transform(X_flat)
        X_scaled = X_scaled_flat.reshape(X_reshaped.shape)

        # Predict
        predictions = self.model_.predict(X_scaled, verbose=0)
        return predictions.flatten()

    def get_training_history(self) -> Optional[Any]:
        """
        Get training history.

        Returns:
            Training history object or None if not fitted
        """
        return self.history_

    def get_params(self, deep: bool = True) -> dict:
        """Get parameters for sklearn compatibility."""
        return {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'validation_split': self.validation_split,
            'early_stopping_patience': self.early_stopping_patience,
            'reduce_lr_patience': self.reduce_lr_patience,
            'verbose': self.verbose,
            'random_state': self.random_state,
            'use_batch_norm': self.use_batch_norm
        }

    def set_params(self, **params) -> 'DepthwiseSeparableCNNRegressor':
        """Set parameters for sklearn compatibility."""
        for param, value in params.items():
            setattr(self, param, value)
        return self

def create_depthwise_cnn_regressor(
    filters: int = 64,
    kernel_size: int = 3,
    dropout: float = 0.2,
    epochs: int = 50,
    batch_size: int = 32,
    **kwargs
) -> DepthwiseSeparableCNNRegressor:
    """
    Factory function to create DepthwiseSeparableCNNRegressor with sensible defaults.
    ... (args) ...
    Returns:
        DepthwiseSeparableCNNRegressor instance
    """
    return DepthwiseSeparableCNNRegressor(
        filters=filters,
        kernel_size=kernel_size,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        **kwargs
    )
    
# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate synthetic time series data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20

    # Create synthetic features with temporal patterns
    X = np.random.randn(n_samples, n_features)
    # Add temporal autocorrelation
    for i in range(1, n_samples):
        X[i] = 0.7 * X[i-1] + 0.3 * X[i]

    # Create target with temporal dependency
    y = np.sum(X[:, :5], axis=1) + 0.5 * np.random.randn(n_samples)

    # Split data
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("🚀 Testing TCN Regressor")
    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")

    # Create and train model
    tcn = DepthwiseSeparableCNNRegressor(
        filters=32,
        kernel_size=3,
        dropout=0.2,
        epochs=30,
        batch_size=32,
        early_stopping_patience=5,
        verbose=1
    )

    print("\n📊 Training TCN model...")
    tcn.fit(X_train, y_train)

    # Make predictions
    y_pred = tcn.predict(X_test)

    # Calculate metrics
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)

    print(f"\n✅ Test Results:")
    print(f"   MSE: {mse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   R²: {r2:.4f}")

    # Plot training history
    if tcn.history_ is not None:
        history = tcn.history_.history

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training History - Loss')

        plt.subplot(1, 2, 2)
        plt.plot(history['mae'], label='Training MAE')
        plt.plot(history['val_mae'], label='Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.title('Training History - MAE')

        plt.tight_layout()
        plt.savefig('tcn_training_history.png')
        print("\n📊 Training history saved to tcn_training_history.png")


# Backward compatibility alias
TCNRegressor = DepthwiseSeparableCNNRegressor


def create_tcn_regressor(**kwargs):
    """Factory function to create a TCN regressor with custom parameters."""
    return DepthwiseSeparableCNNRegressor(**kwargs)
