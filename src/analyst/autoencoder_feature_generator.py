# feature_engineering/autoencoder_feature_generator.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import logging
from typing import List, Dict, Any

# Set up a basic logger for standalone execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedAutoencoderPreprocessor:
    """Enhanced preprocessor to prevent extreme values in autoencoder features"""
    
    def __init__(self, use_robust_scaling=True, outlier_threshold=3.0):
        self.use_robust_scaling = use_robust_scaling
        self.scaler = RobustScaler() if use_robust_scaling else StandardScaler()
        self.outlier_threshold = outlier_threshold
        self.is_fitted = False
        logger.info(f"Preprocessor initialized with robust_scaling={use_robust_scaling}, outlier_threshold={outlier_threshold}")
        
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        logger.info(f"Fitting and transforming data with shape {X.shape}")
        X_clean = self._remove_extreme_outliers(X.values)
        X_scaled = self.scaler.fit_transform(X_clean)
        X_clipped = np.clip(X_scaled, -self.outlier_threshold, self.outlier_threshold)
        self.is_fitted = True
        logger.info(f"Fit-transform complete. Output shape: {X_clipped.shape}")
        return X_clipped
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
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
        logger.debug(f"Removing extreme outliers from data with shape {X.shape}")
        Q1, Q3 = np.percentile(X, [25, 75], axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        lower_clipped_count = np.sum(X < lower_bound)
        upper_clipped_count = np.sum(X > upper_bound)
        if lower_clipped_count > 0 or upper_clipped_count > 0:
            logger.debug(f"Clipping {lower_clipped_count} lower-bound outliers and {upper_clipped_count} upper-bound outliers.")
        return np.clip(X, lower_bound, upper_bound)

class RobustAutoencoder:
    """Improved autoencoder architecture to prevent extreme feature values"""
    
    def __init__(self, input_dim: int, encoding_dim: int = 32, dropout_rate: float = 0.3):
        self.input_dim = input_dim
        self.encoding_dim = min(encoding_dim, input_dim // 2)
        self.dropout_rate = dropout_rate
        self.autoencoder = None
        self.encoder = None
        self.preprocessor = ImprovedAutoencoderPreprocessor()
        logger.info(f"RobustAutoencoder initialized with input_dim={input_dim}, encoding_dim={self.encoding_dim}, dropout_rate={dropout_rate}")

    def build_model(self):
        logger.info("Building autoencoder model...")
        input_layer = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(min(256, self.input_dim * 2), activation='relu', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(min(128, self.input_dim), activation='relu', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        bottleneck = layers.Dense(self.encoding_dim, activation='tanh', kernel_regularizer=regularizers.l2(0.001), kernel_constraint=tf.keras.constraints.max_norm(2.0), name='bottleneck')(x)
        x = layers.Dense(min(128, self.input_dim), activation='relu', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(bottleneck)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(min(256, self.input_dim * 2), activation='relu', kernel_regularizer=regularizers.l2(0.001), kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout_rate)(x)
        output_layer = layers.Dense(self.input_dim, activation='linear', kernel_initializer='glorot_normal')(x)
        self.autoencoder = Model(input_layer, output_layer)
        self.encoder = Model(input_layer, bottleneck)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
        self.autoencoder.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
        logger.info("Autoencoder model built and compiled successfully.")
        self.autoencoder.summary(print_fn=logger.info)
        return self.autoencoder
    
    def fit(self, X_train: pd.DataFrame, validation_split=0.2, epochs=100, batch_size=32, verbose=1):
        logger.info(f"Starting autoencoder training for {epochs} epochs with batch size {batch_size}.")
        X_processed = self.preprocessor.fit_transform(X_train)
        if validation_split > 0:
            split_idx = int((1 - validation_split) * len(X_processed))
            X_train_split, X_val_split = X_processed[:split_idx], X_processed[split_idx:]
            validation_data = (X_val_split, X_val_split)
            logger.info(f"Training data shape: {X_train_split.shape}, Validation data shape: {X_val_split.shape}")
        else:
            X_train_split, validation_data = X_processed, None
            logger.info(f"Training data shape: {X_train_split.shape}, No validation split.")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss' if validation_data else 'loss', patience=15, restore_best_weights=True, min_delta=1e-6),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss' if validation_data else 'loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1)
        ]
        history = self.autoencoder.fit(X_train_split, X_train_split, epochs=epochs, batch_size=batch_size, validation_data=validation_data, callbacks=callbacks, verbose=verbose)
        logger.info("Autoencoder training finished.")
        return history
    
    def get_encoded_features(self, X: pd.DataFrame) -> np.ndarray:
        logger.info(f"Generating encoded features for input data with shape {X.shape}")
        X_processed = self.preprocessor.transform(X)
        encoded_features = self.encoder.predict(X_processed, verbose=0)
        logger.info(f"Encoded features generated with shape {encoded_features.shape}")
        encoded_features = np.clip(encoded_features, -2.0, 2.0)
        extreme_mask = np.abs(encoded_features) > 1.5
        if np.any(extreme_mask):
            n_extreme = np.sum(extreme_mask)
            logger.warning(f"Found {n_extreme} values > 1.5 in encoded features, clipping to [-1.5, 1.5]...")
            encoded_features = np.clip(encoded_features, -1.5, 1.5)
        return encoded_features

class AutoencoderFeaturePipeline:
    """
    A full pipeline to select, preprocess, train, and generate autoencoder features.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('autoencoder', {})
        self.feature_list = self.config.get('feature_list', [
            'RSI', 'MACD_hist', 'MOM', 'ADX', 'Williams_R', 'CCI', 'MFI', 'OBV',
            'volume_change', 'VROC', 'order_flow_imbalance', 'ATR_normalized',
            'realized_volatility_30d', 'BB_width', 'Funding_Extreme'
        ])
        self.encoding_dim = self.config.get('encoding_dim', 16)
        self.dropout_rate = self.config.get('dropout_rate', 0.3)
        self.epochs = self.config.get('epochs', 50)
        self.batch_size = self.config.get('batch_size', 64)
        self.model = None
        logger.info("AutoencoderFeaturePipeline initialized.")

    def generate(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        The main method to run the entire autoencoder feature generation process.
        
        Args:
            features_df: The DataFrame containing all previously generated features.
            
        Returns:
            A DataFrame with the new encoded features.
        """
        logger.info("Starting autoencoder feature generation pipeline...")
        
        # 1. Select and clean the features for the autoencoder
        selected_features = self._select_and_clean_features(features_df)
        
        if selected_features.empty:
            logger.warning("No data available for autoencoder training after cleaning. Skipping.")
            return pd.DataFrame(index=features_df.index)
            
        # 2. Initialize and build the autoencoder model
        self.model = RobustAutoencoder(
            input_dim=selected_features.shape[1],
            encoding_dim=self.encoding_dim,
            dropout_rate=self.dropout_rate
        )
        self.model.build_model()
        
        # 3. Train the model
        self.model.fit(
            selected_features,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1
        )
        
        # 4. Generate encoded features for the entire dataset
        encoded_features_array = self.model.get_encoded_features(selected_features)
        
        # 5. Format into a DataFrame
        ae_feature_names = [f'autoencoder_{i}' for i in range(self.encoding_dim)]
        encoded_df = pd.DataFrame(encoded_features_array, index=selected_features.index, columns=ae_feature_names)
        
        logger.info("Autoencoder feature generation pipeline finished successfully.")
        return encoded_df

    def _select_and_clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Selects features, handles NaNs, and returns a clean DataFrame."""
        # Filter for features that actually exist in the dataframe
        available_features = [f for f in self.feature_list if f in df.columns]
        logger.info(f"Selected {len(available_features)} available features for autoencoder.")
        
        # Create a subset and handle missing values robustly
        subset_df = df[available_features].copy()
        initial_nan_count = subset_df.isnull().sum().sum()
        
        if initial_nan_count > 0:
            logger.warning(f"Found {initial_nan_count} NaN values in selected features. Applying ffill and bfill.")
            subset_df.fillna(method='ffill', inplace=True)
            subset_df.fillna(method='bfill', inplace=True)
            # Any remaining NaNs (if entire columns are NaN) are filled with 0
            subset_df.fillna(0, inplace=True)

        return subset_df
