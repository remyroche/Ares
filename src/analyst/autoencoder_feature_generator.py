# feature_engineering/autoencoder_feature_generator.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import logging

# Set up a basic logger for standalone execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedAutoencoderPreprocessor:
    """Enhanced preprocessor to prevent extreme values in autoencoder features"""
    
    def __init__(self, use_robust_scaling=True, outlier_threshold=3.0):
        """
        Initializes the preprocessor.

        Args:
            use_robust_scaling (bool): If True, use RobustScaler which is less sensitive to outliers.
                                       If False, use StandardScaler.
            outlier_threshold (float): The number of standard deviations to clip scaled data at.
        """
        self.use_robust_scaling = use_robust_scaling
        self.scaler = RobustScaler() if use_robust_scaling else StandardScaler()
        self.outlier_threshold = outlier_threshold
        self.is_fitted = False
        logger.info(f"Preprocessor initialized with robust_scaling={use_robust_scaling}, outlier_threshold={outlier_threshold}")
        
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit preprocessor and transform data"""
        logger.info(f"Fitting and transforming data with shape {X.shape}")
        X_clean = self._remove_extreme_outliers(X.values)
        X_scaled = self.scaler.fit_transform(X_clean)
        X_clipped = np.clip(X_scaled, -self.outlier_threshold, self.outlier_threshold)
        self.is_fitted = True
        logger.info(f"Fit-transform complete. Output shape: {X_clipped.shape}")
        return X_clipped
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using the fitted preprocessor"""
        if not self.is_fitted:
            logger.error("Preprocessor must be fitted before transform can be called.")
            raise ValueError("Preprocessor must be fitted before transform")
        
        logger.info(f"Transforming data with shape {X.shape}")
        X_clean = self._remove_extreme_outliers(X.values)
        X_scaled = self.scaler.transform(X_clean)
        X_clipped = np.clip(X_scaled, -self.outlier_threshold, self.outlier_threshold)
        logger.info(f"Transform complete. Output shape: {X_clipped.shape}")
        return X_clipped
    
    def _remove_extreme_outliers(self, X: np.ndarray) -> np.ndarray:
        """Remove extreme outliers using the IQR method"""
        logger.debug(f"Removing extreme outliers from data with shape {X.shape}")
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        
        # Use 3*IQR as the threshold for identifying extreme outliers
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # Log how many values are being clipped
        lower_clipped_count = np.sum(X < lower_bound)
        upper_clipped_count = np.sum(X > upper_bound)
        if lower_clipped_count > 0 or upper_clipped_count > 0:
            logger.debug(f"Clipping {lower_clipped_count} lower-bound outliers and {upper_clipped_count} upper-bound outliers.")

        # Clip the data to the bounds
        X_clean = np.where(X < lower_bound, lower_bound, X)
        X_clean = np.where(X_clean > upper_bound, upper_bound, X_clean)
        
        return X_clean

class RobustAutoencoder:
    """Improved autoencoder architecture to prevent extreme feature values"""
    
    def __init__(self, input_dim: int, encoding_dim: int = 32, dropout_rate: float = 0.3):
        """
        Initializes the autoencoder.

        Args:
            input_dim (int): The number of input features.
            encoding_dim (int): The dimensionality of the encoded representation (bottleneck).
            dropout_rate (float): The dropout rate to use for regularization.
        """
        self.input_dim = input_dim
        self.encoding_dim = min(encoding_dim, input_dim // 2)  # Ensure reasonable encoding size
        self.dropout_rate = dropout_rate
        self.autoencoder = None
        self.encoder = None
        self.preprocessor = ImprovedAutoencoderPreprocessor()
        logger.info(f"RobustAutoencoder initialized with input_dim={input_dim}, encoding_dim={self.encoding_dim}, dropout_rate={dropout_rate}")

    def build_model(self):
        """Build a robust autoencoder with regularization and constraints."""
        logger.info("Building autoencoder model...")
        # Input layer
        input_layer = layers.Input(shape=(self.input_dim,))
        
        # Encoder with progressive dimension reduction and regularization
        x = layers.Dense(min(256, self.input_dim * 2), activation='relu', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(min(128, self.input_dim), activation='relu', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Bottleneck layer with constraints to prevent extreme values
        bottleneck = layers.Dense(self.encoding_dim, activation='tanh', kernel_regularizer=regularizers.l2(0.001), kernel_constraint=tf.keras.constraints.max_norm(2.0), name='bottleneck')(x)
        
        # Decoder (mirror of encoder)
        x = layers.Dense(min(128, self.input_dim), activation='relu', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(bottleneck)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(min(256, self.input_dim * 2), activation='relu', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer
        output_layer = layers.Dense(self.input_dim, activation='linear', kernel_initializer='glorot_normal')(x)
        
        # Create models
        self.autoencoder = Model(input_layer, output_layer)
        self.encoder = Model(input_layer, bottleneck)
        
        # Compile with gradient clipping and a robust loss function
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        
        self.autoencoder.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
        logger.info("Autoencoder model built and compiled successfully.")
        self.autoencoder.summary(print_fn=logger.info)
        
        return self.autoencoder
    
    def fit(self, X_train: pd.DataFrame, validation_split=0.2, epochs=100, batch_size=32, verbose=1):
        """Train the autoencoder with proper preprocessing and callbacks."""
        logger.info(f"Starting autoencoder training for {epochs} epochs with batch size {batch_size}.")
        # Preprocess training data
        X_processed = self.preprocessor.fit_transform(X_train)
        
        # Create validation split from the processed data
        if validation_split > 0:
            split_idx = int((1 - validation_split) * len(X_processed))
            X_train_split, X_val_split = X_processed[:split_idx], X_processed[split_idx:]
            validation_data = (X_val_split, X_val_split)
            logger.info(f"Training data shape: {X_train_split.shape}, Validation data shape: {X_val_split.shape}")
        else:
            X_train_split = X_processed
            validation_data = None
            logger.info(f"Training data shape: {X_train_split.shape}, No validation split.")
        
        # Callbacks for robust training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss' if validation_data else 'loss', patience=15, restore_best_weights=True, min_delta=1e-6),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss' if validation_data else 'loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1)
        ]
        
        # Train the model
        history = self.autoencoder.fit(
            X_train_split, X_train_split,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        logger.info("Autoencoder training finished.")
        return history
    
    def get_encoded_features(self, X: pd.DataFrame) -> np.ndarray:
        """Get encoded features with additional safety checks."""
        logger.info(f"Generating encoded features for input data with shape {X.shape}")
        # Preprocess input data using the already fitted preprocessor
        X_processed = self.preprocessor.transform(X)
        
        # Get encoded features from the encoder part of the model
        encoded_features = self.encoder.predict(X_processed, verbose=0)
        logger.info(f"Encoded features generated with shape {encoded_features.shape}")
        
        # Additional safety: clip to reasonable bounds.
        encoded_features = np.clip(encoded_features, -2.0, 2.0)
        
        # Check for any remaining extreme values
        extreme_mask = np.abs(encoded_features) > 1.5
        if np.any(extreme_mask):
            n_extreme = np.sum(extreme_mask)
            logger.warning(f"Found {n_extreme} values > 1.5 in encoded features, clipping to [-1.5, 1.5]...")
            encoded_features = np.clip(encoded_features, -1.5, 1.5)
        
        return encoded_features
