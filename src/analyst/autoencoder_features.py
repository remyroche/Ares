# feature_engineering/autoencoder_features.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import logging

logger = logging.getLogger(__name__)

class ImprovedAutoencoderPreprocessor:
    """Enhanced preprocessor to prevent extreme values in autoencoder features"""
    
    def __init__(self, use_robust_scaling=True, outlier_threshold=3.0):
        self.use_robust_scaling = use_robust_scaling
        self.scaler = RobustScaler() if use_robust_scaling else StandardScaler()
        self.outlier_threshold = outlier_threshold
        self.is_fitted = False
        
    def fit_transform(self, X):
        """Fit preprocessor and transform data"""
        X_clean = self._remove_extreme_outliers(X)
        X_scaled = self.scaler.fit_transform(X_clean)
        X_clipped = np.clip(X_scaled, -self.outlier_threshold, self.outlier_threshold)
        self.is_fitted = True
        return X_clipped
    
    def transform(self, X):
        """Transform new data using fitted preprocessor"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        X_clean = self._remove_extreme_outliers(X)
        X_scaled = self.scaler.transform(X_clean)
        X_clipped = np.clip(X_scaled, -self.outlier_threshold, self.outlier_threshold)
        return X_clipped
    
    def _remove_extreme_outliers(self, X):
        """Remove extreme outliers using IQR method"""
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        
        # Use 3*IQR as threshold (more conservative than your current system)
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        X_clean = np.where(X < lower_bound, lower_bound, X)
        X_clean = np.where(X_clean > upper_bound, upper_bound, X_clean)
        
        return X_clean

class RobustAutoencoder:
    """Improved autoencoder architecture to prevent extreme feature values"""
    
    def __init__(self, input_dim, encoding_dim=32, dropout_rate=0.3):
        self.input_dim = input_dim
        self.encoding_dim = min(encoding_dim, input_dim // 2)  # Ensure reasonable encoding size
        self.dropout_rate = dropout_rate
        self.autoencoder = None
        self.encoder = None
        self.preprocessor = ImprovedAutoencoderPreprocessor()
        
    def build_model(self):
        """Build robust autoencoder with regularization"""
        # Input layer
        input_layer = layers.Input(shape=(self.input_dim,))
        
        # Encoder with progressive dimension reduction
        x = layers.Dense(
            min(256, self.input_dim * 2), 
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
            kernel_initializer='he_normal'
        )(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Dense(
            min(128, self.input_dim), 
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
            kernel_initializer='he_normal'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Bottleneck layer with constraint to prevent extreme values
        bottleneck = layers.Dense(
            self.encoding_dim, 
            activation='tanh',  # Bounded activation [-1, 1]
            kernel_regularizer=regularizers.l2(0.001),
            kernel_constraint=tf.keras.constraints.max_norm(2.0),  # Constrain weights
            name='bottleneck'
        )(x)
        
        # Decoder (mirror of encoder)
        x = layers.Dense(
            min(128, self.input_dim), 
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
            kernel_initializer='he_normal'
        )(bottleneck)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        x = layers.Dense(
            min(256, self.input_dim * 2), 
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
            kernel_initializer='he_normal'
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer
        output_layer = layers.Dense(
            self.input_dim, 
            activation='linear',
            kernel_initializer='glorot_normal'
        )(x)
        
        # Create models
        self.autoencoder = Model(input_layer, output_layer)
        self.encoder = Model(input_layer, bottleneck)
        
        # Compile with gradient clipping
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            clipnorm=1.0  # Gradient clipping
        )
        
        self.autoencoder.compile(
            optimizer=optimizer,
            loss='huber',  # More robust than MSE
            metrics=['mae']
        )
        
        return self.autoencoder
    
    def fit(self, X_train, validation_split=0.2, epochs=100, batch_size=32, verbose=1):
        """Train autoencoder with proper preprocessing"""
        # Preprocess training data
        X_processed = self.preprocessor.fit_transform(X_train)
        
        # Validation split
        if validation_split > 0:
            split_idx = int((1 - validation_split) * len(X_processed))
            X_train_split = X_processed[:split_idx]
            X_val_split = X_processed[split_idx:]
            validation_data = (X_val_split, X_val_split)
        else:
            X_train_split = X_processed
            validation_data = None
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=15,
                restore_best_weights=True,
                min_delta=1e-6
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train
        history = self.autoencoder.fit(
            X_train_split, X_train_split,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def get_encoded_features(self, X):
        """Get encoded features with additional safety checks"""
        # Preprocess input
        X_processed = self.preprocessor.transform(X)
        
        # Get encoded features
        encoded_features = self.encoder.predict(X_processed, verbose=0)
        
        # Additional safety: clip to reasonable bounds
        # Since we use tanh activation, values should be in [-1, 1], but add buffer
        encoded_features = np.clip(encoded_features, -2.0, 2.0)
        
        # Check for any remaining extreme values
        extreme_mask = np.abs(encoded_features) > 1.5
        if np.any(extreme_mask):
            n_extreme = np.sum(extreme_mask)
            logger.warning(f"Found {n_extreme} values > 1.5 in encoded features, clipping...")
            encoded_features = np.clip(encoded_features, -1.5, 1.5)
        
        return encoded_features
