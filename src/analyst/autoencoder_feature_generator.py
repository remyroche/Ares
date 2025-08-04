# src/analyst/autoencoder_feature_generator.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import logging
from typing import List, Dict, Any, Optional
import os
from pathlib import Path

# Set up comprehensive logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors


class ImprovedAutoencoderPreprocessor:
    """Enhanced preprocessor to prevent extreme values in autoencoder features"""
    
    def __init__(self, use_robust_scaling=True, outlier_threshold=3.0):
        self.use_robust_scaling = use_robust_scaling
        self.scaler = RobustScaler() if use_robust_scaling else StandardScaler()
        self.outlier_threshold = outlier_threshold
        self.is_fitted = False
        self.logger = system_logger.getChild("AutoencoderPreprocessor")
        self.logger.info(f"Preprocessor initialized with robust_scaling={use_robust_scaling}, outlier_threshold={outlier_threshold}")
        
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        self.logger.info(f"Fitting and transforming data with shape {X.shape}")
        try:
            X_clean = self._remove_extreme_outliers(X.values)
            X_scaled = self.scaler.fit_transform(X_clean)
            X_clipped = np.clip(X_scaled, -self.outlier_threshold, self.outlier_threshold)
            self.is_fitted = True
            self.logger.info(f"Fit-transform complete. Output shape: {X_clipped.shape}")
            return X_clipped
        except Exception as e:
            self.logger.error(f"Error in fit_transform: {e}")
            return X.values
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            self.logger.error("Preprocessor must be fitted before transform can be called.")
            raise ValueError("Preprocessor must be fitted before transform")
        self.logger.info(f"Transforming data with shape {X.shape}")
        try:
            X_clean = self._remove_extreme_outliers(X.values)
            X_scaled = self.scaler.transform(X_clean)
            X_clipped = np.clip(X_scaled, -self.outlier_threshold, self.outlier_threshold)
            self.logger.info(f"Transform complete. Output shape: {X_clipped.shape}")
            return X_clipped
        except Exception as e:
            self.logger.error(f"Error in transform: {e}")
            return X.values
    
    def _remove_extreme_outliers(self, X: np.ndarray) -> np.ndarray:
        self.logger.debug(f"Removing extreme outliers from data with shape {X.shape}")
        try:
            Q1, Q3 = np.percentile(X, [25, 75], axis=0)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            lower_clipped_count = np.sum(X < lower_bound)
            upper_clipped_count = np.sum(X > upper_bound)
            if lower_clipped_count > 0 or upper_clipped_count > 0:
                self.logger.debug(f"Clipping {lower_clipped_count} lower-bound outliers and {upper_clipped_count} upper-bound outliers.")
            return np.clip(X, lower_bound, upper_bound)
        except Exception as e:
            self.logger.error(f"Error removing outliers: {e}")
            return X


class RobustAutoencoder:
    """Improved autoencoder architecture to prevent extreme feature values"""
    
    def __init__(self, input_dim: int, encoding_dim: int = 32, dropout_rate: float = 0.3):
        self.input_dim = input_dim
        self.encoding_dim = min(encoding_dim, input_dim // 2)
        self.dropout_rate = dropout_rate
        self.autoencoder = None
        self.encoder = None
        self.preprocessor = ImprovedAutoencoderPreprocessor()
        self.logger = system_logger.getChild("RobustAutoencoder")
        self.logger.info(f"RobustAutoencoder initialized with input_dim={input_dim}, encoding_dim={self.encoding_dim}, dropout_rate={dropout_rate}")

    def build_model(self):
        self.logger.info("Building autoencoder model...")
        try:
            input_layer = layers.Input(shape=(self.input_dim,))
            
            # Encoder layers
            x = layers.Dense(min(256, self.input_dim * 2), activation='relu', 
                           kernel_regularizer=regularizers.l2(0.001), 
                           kernel_initializer='he_normal')(input_layer)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)
            
            x = layers.Dense(min(128, self.input_dim), activation='relu', 
                           kernel_regularizer=regularizers.l2(0.001), 
                           kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)
            
            # Bottleneck layer
            bottleneck = layers.Dense(self.encoding_dim, activation='tanh', 
                                   kernel_regularizer=regularizers.l2(0.001), 
                                   kernel_constraint=tf.keras.constraints.max_norm(2.0), 
                                   name='bottleneck')(x)
            
            # Decoder layers
            x = layers.Dense(min(128, self.input_dim), activation='relu', 
                           kernel_regularizer=regularizers.l2(0.001), 
                           kernel_initializer='he_normal')(bottleneck)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)
            
            x = layers.Dense(min(256, self.input_dim * 2), activation='relu', 
                           kernel_regularizer=regularizers.l2(0.001), 
                           kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)
            
            # Output layer
            output_layer = layers.Dense(self.input_dim, activation='linear', 
                                      kernel_initializer='glorot_normal')(x)
            
            self.autoencoder = Model(input_layer, output_layer)
            self.encoder = Model(input_layer, bottleneck)
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
            self.autoencoder.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
            
            self.logger.info("Autoencoder model built and compiled successfully.")
            self.autoencoder.summary(print_fn=self.logger.info)
            return self.autoencoder
        except Exception as e:
            self.logger.error(f"Error building autoencoder model: {e}")
            raise
    
    def fit(self, X_train: pd.DataFrame, validation_split=0.2, epochs=100, batch_size=32, verbose=1):
        self.logger.info(f"Starting autoencoder training for {epochs} epochs with batch size {batch_size}.")
        try:
            X_processed = self.preprocessor.fit_transform(X_train)
            
            if validation_split > 0:
                split_idx = int((1 - validation_split) * len(X_processed))
                X_train_split, X_val_split = X_processed[:split_idx], X_processed[split_idx:]
                validation_data = (X_val_split, X_val_split)
            else:
                validation_data = None
            
            # Early stopping callback
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
            
            # Reduce learning rate callback
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
            
            # Training callbacks
            callbacks = [early_stopping, reduce_lr]
            
            history = self.autoencoder.fit(
                X_train_split, X_train_split,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
            
            self.logger.info("Autoencoder training completed successfully.")
            return history
        except Exception as e:
            self.logger.error(f"Error during autoencoder training: {e}")
            raise

    def get_encoded_features(self, X: pd.DataFrame) -> np.ndarray:
        """Get encoded features from the bottleneck layer."""
        self.logger.info(f"Getting encoded features for data with shape {X.shape}")
        try:
            X_processed = self.preprocessor.transform(X)
            encoded_features = self.encoder.predict(X_processed, verbose=0)
            self.logger.info(f"Encoded features shape: {encoded_features.shape}")
            return encoded_features
        except Exception as e:
            self.logger.error(f"Error getting encoded features: {e}")
            return np.zeros((X.shape[0], self.encoding_dim))


class AutoencoderFeaturePipeline:
    """Complete pipeline for generating autoencoder-based features"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("AutoencoderFeaturePipeline")
        self.autoencoder = None
        self.feature_columns = []
        self.logger.info("AutoencoderFeaturePipeline initialized")

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError),
        default_return=pd.DataFrame(),
        context="autoencoder feature generation",
    )
    def generate(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate autoencoder-based features.
        
        Args:
            features_df: Input features DataFrame
            
        Returns:
            DataFrame with original features plus autoencoder features
        """
        self.logger.info("Starting autoencoder feature generation...")
        
        if features_df.empty:
            self.logger.warning("Empty features DataFrame provided")
            return features_df
        
        try:
            # Select and clean features for autoencoder
            clean_features = self._select_and_clean_features(features_df)
            
            if clean_features.empty:
                self.logger.warning("No suitable features found for autoencoder")
                return features_df
            
            self.logger.info(f"Selected {clean_features.shape[1]} features for autoencoder")
            
            # Initialize autoencoder if not already done
            if self.autoencoder is None:
                self.logger.info("Initializing autoencoder...")
                self.autoencoder = RobustAutoencoder(
                    input_dim=clean_features.shape[1],
                    encoding_dim=min(32, clean_features.shape[1] // 2),
                    dropout_rate=0.3
                )
                self.autoencoder.build_model()
            
            # Train autoencoder if we have enough data
            if len(clean_features) > 100:
                self.logger.info("Training autoencoder...")
                self.autoencoder.fit(clean_features, epochs=50, batch_size=32, verbose=0)
            
            # Generate encoded features
            self.logger.info("Generating encoded features...")
            encoded_features = self.autoencoder.get_encoded_features(clean_features)
            
            # Add encoded features to original DataFrame
            result_df = features_df.copy()
            for i in range(encoded_features.shape[1]):
                col_name = f'autoencoder_feature_{i+1}'
                result_df[col_name] = encoded_features[:, i]
            
            # Add reconstruction error as a feature
            self.logger.info("Calculating reconstruction error...")
            reconstructed = self.autoencoder.autoencoder.predict(
                self.autoencoder.preprocessor.transform(clean_features), 
                verbose=0
            )
            reconstruction_error = np.mean((clean_features.values - reconstructed) ** 2, axis=1)
            result_df['autoencoder_reconstruction_error'] = reconstruction_error
            
            self.logger.info(f"Autoencoder feature generation completed. Added {encoded_features.shape[1] + 1} new features.")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error in autoencoder feature generation: {e}")
            return features_df

    def _select_and_clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and clean features for autoencoder training."""
        self.logger.info("Selecting and cleaning features for autoencoder...")
        
        try:
            # Select numeric features only
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove columns with too many missing values
            missing_threshold = 0.5
            missing_counts = df[numeric_cols].isnull().sum()
            valid_cols = missing_counts[missing_counts / len(df) < missing_threshold].index.tolist()
            
            # Remove columns with zero variance
            variance = df[valid_cols].var()
            variance_cols = variance[variance > 0].index.tolist()
            
            # Remove columns with infinite values
            finite_cols = []
            for col in variance_cols:
                if np.isfinite(df[col]).all():
                    finite_cols.append(col)
            
            # Fill remaining missing values
            selected_df = df[finite_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            self.logger.info(f"Selected {len(finite_cols)} features for autoencoder")
            return selected_df
            
        except Exception as e:
            self.logger.error(f"Error selecting and cleaning features: {e}")
            return pd.DataFrame()

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the autoencoder pipeline."""
        try:
            return {
                "autoencoder_initialized": self.autoencoder is not None,
                "feature_columns_count": len(self.feature_columns),
                "encoding_dim": self.autoencoder.encoding_dim if self.autoencoder else None,
                "input_dim": self.autoencoder.input_dim if self.autoencoder else None,
                "dropout_rate": self.autoencoder.dropout_rate if self.autoencoder else None
            }
        except Exception as e:
            self.logger.error(f"Error getting pipeline info: {e}")
            return {}


class AutoencoderFeatureGenerator:
    """Main class for autoencoder feature generation with comprehensive logging"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("AutoencoderFeatureGenerator")
        self.pipeline = AutoencoderFeaturePipeline(config)
        self.logger.info("AutoencoderFeatureGenerator initialized successfully")

    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError),
        default_return=pd.DataFrame(),
        context="autoencoder feature generation",
    )
    def generate_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate autoencoder-based features with comprehensive logging.
        
        Args:
            features_df: Input features DataFrame
            
        Returns:
            DataFrame with original features plus autoencoder features
        """
        self.logger.info("🚀 Starting autoencoder feature generation...")
        
        if features_df.empty:
            self.logger.warning("⚠️ Empty features DataFrame provided")
            return features_df
        
        try:
            # Log input statistics
            self.logger.info(f"📊 Input features shape: {features_df.shape}")
            self.logger.info(f"📊 Input features columns: {list(features_df.columns)}")
            
            # Generate autoencoder features
            result_df = self.pipeline.generate(features_df)
            
            # Log output statistics
            self.logger.info(f"✅ Autoencoder feature generation completed")
            self.logger.info(f"📊 Output features shape: {result_df.shape}")
            self.logger.info(f"📊 Added autoencoder features: {[col for col in result_df.columns if 'autoencoder' in col]}")
            
            # Get pipeline info
            pipeline_info = self.pipeline.get_pipeline_info()
            self.logger.info(f"🔧 Pipeline info: {pipeline_info}")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"❌ Error in autoencoder feature generation: {e}")
            return features_df

    def get_generator_info(self) -> Dict[str, Any]:
        """Get information about the autoencoder feature generator."""
        try:
            pipeline_info = self.pipeline.get_pipeline_info()
            return {
                "generator_type": "AutoencoderFeatureGenerator",
                "pipeline_info": pipeline_info,
                "config": self.config
            }
        except Exception as e:
            self.logger.error(f"Error getting generator info: {e}")
            return {}
