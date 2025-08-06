# src/analyst/autoencoder_feature_generator.py

import logging
import os
import pickle
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import optuna
import shap
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from optuna.integration import TFKerasPruningCallback

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
import sys

sys.path.insert(0, str(project_root))

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


class AutoencoderConfig:
    """Configuration manager for autoencoder feature generator."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "src/analyst/autoencoder_config.yaml"
        self.config = self._load_config()
        self.logger = system_logger.getChild("AutoencoderConfig")
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            self.logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails."""
        return {
            "preprocessing": {
                "scaler_type": "robust",
                "outlier_threshold": 3.0,
                "missing_value_strategy": "forward_fill",
                "feature_selection_threshold": 0.99
            },
            "sequence": {
                "timesteps": 10,
                "overlap": 0.5,
                "min_sequence_length": 5
            },
            "autoencoder": {
                "encoding_dim": 32,
                "dropout_rate": 0.3,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100,
                "validation_split": 0.2,
                "early_stopping_patience": 10,
                "reduce_lr_patience": 5,
                "min_lr": 1e-6
            },
            "cnn": {
                "filters": [16, 32, 64],
                "kernel_size": [3, 5, 7],
                "pool_size": 2,
                "activation": "relu",
                "kernel_regularizer": 0.001,
                "max_norm_constraint": 2.0
            },
            "training": {
                "n_trials": 50,
                "timeout_seconds": 1800,
                "n_jobs": -1,
                "pruning_enabled": True,
                "pruning_metric": "val_loss",
                "pruning_patience": 5
            },
            "feature_filtering": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
                "importance_threshold": 0.99
            },
            "triple_barrier": {
                "profit_take": 0.001,
                "stop_loss": 0.0005,
                "time_barrier": 300
            },
            "output": {
                "save_autoencoder": True,
                "save_config": True,
                "save_preprocessor": True,
                "output_dir": "models/autoencoder_features",
                "config_filename": "autoencoder_config.yaml"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def save_config(self, output_path: str) -> None:
        """Save current configuration to file."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)
            self.logger.info(f"Configuration saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")


class TripleBarrierLabeler:
    """Triple Barrier Method implementation for labeling."""
    
    def __init__(self, config: AutoencoderConfig):
        self.config = config
        self.logger = system_logger.getChild("TripleBarrierLabeler")
        
    def apply_triple_barrier(self, data: pd.DataFrame) -> np.ndarray:
        """Apply triple barrier method to generate labels."""
        try:
            labels = []
            profit_take = self.config.get("triple_barrier.profit_take", 0.001)
            stop_loss = self.config.get("triple_barrier.stop_loss", 0.0005)
            time_barrier = self.config.get("triple_barrier.time_barrier", 300)
            
            for i in range(len(data)):
                if i < 10:  # Skip first few points
                    labels.append(0)
                    continue
                
                current_price = data.iloc[i]["close"]
                entry_price = data.iloc[i - 1]["close"]
                
                # Calculate barriers
                profit_barrier = entry_price * (1 + profit_take)
                stop_barrier = entry_price * (1 - stop_loss)
                time_barrier_idx = i + time_barrier
                
                # Check if barriers are hit
                label = 0  # Neutral
                
                # Look forward to see if barriers are hit
                for j in range(i + 1, min(len(data), int(time_barrier_idx))):
                    high_price = data.iloc[j]["high"]
                    low_price = data.iloc[j]["low"]
                    
                    if high_price >= profit_barrier:
                        label = 1  # Profit take hit
                        break
                    if low_price <= stop_barrier:
                        label = -1  # Stop loss hit
                        break
                
                labels.append(label)
            
            return np.array(labels)
            
        except Exception as e:
            self.logger.error(f"Error applying triple barrier: {e}")
            return np.array([])


class FeatureFilter:
    """Random Forest + SHAP feature filtering."""
    
    def __init__(self, config: AutoencoderConfig):
        self.config = config
        self.logger = system_logger.getChild("FeatureFilter")
        self.rf_model = None
        self.feature_importance = None
        
    def filter_features(self, features_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        """Filter features using Random Forest + SHAP importance."""
        try:
            self.logger.info("Starting feature filtering with Random Forest + SHAP...")
            
            # Prepare data
            X = features_df.select_dtypes(include=[np.number]).fillna(0)
            y = labels
            
            if len(np.unique(y)) < 2:
                self.logger.warning("Insufficient unique labels for classification")
                return features_df
            
            # Train Random Forest
            self.rf_model = RandomForestClassifier(
                n_estimators=self.config.get("feature_filtering.n_estimators", 100),
                max_depth=self.config.get("feature_filtering.max_depth", 10),
                min_samples_split=self.config.get("feature_filtering.min_samples_split", 5),
                min_samples_leaf=self.config.get("feature_filtering.min_samples_leaf", 2),
                random_state=self.config.get("feature_filtering.random_state", 42),
                n_jobs=-1
            )
            
            self.rf_model.fit(X, y)
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(self.rf_model)
            shap_values = explainer.shap_values(X)
            
            # Get feature importance (mean absolute SHAP values)
            if len(shap_values.shape) == 3:  # Multi-class
                feature_importance = np.mean(np.abs(shap_values), axis=(0, 1))
            else:  # Binary
                feature_importance = np.mean(np.abs(shap_values), axis=0)
            
            # Calculate cumulative importance
            sorted_indices = np.argsort(feature_importance)[::-1]
            cumulative_importance = np.cumsum(feature_importance[sorted_indices])
            total_importance = cumulative_importance[-1]
            
            # Find features that contribute to threshold
            threshold = self.config.get("feature_filtering.importance_threshold", 0.99)
            importance_threshold = threshold * total_importance
            
            selected_indices = sorted_indices[cumulative_importance <= importance_threshold]
            
            # Get selected feature names
            selected_features = X.columns[selected_indices].tolist()
            
            self.logger.info(f"Selected {len(selected_features)} features out of {len(X.columns)}")
            self.logger.info(f"Selected features: {selected_features}")
            
            # Return filtered DataFrame
            filtered_df = features_df[selected_features].copy()
            
            return filtered_df
            
        except Exception as e:
            self.logger.error(f"Error in feature filtering: {e}")
            return features_df


class ImprovedAutoencoderPreprocessor:
    """Enhanced preprocessor with separate fit/transform and RobustScaler."""
    
    def __init__(self, config: AutoencoderConfig):
        self.config = config
        self.scaler_type = config.get("preprocessing.scaler_type", "robust")
        self.outlier_threshold = config.get("preprocessing.outlier_threshold", 3.0)
        self.missing_value_strategy = config.get("preprocessing.missing_value_strategy", "forward_fill")
        
        # Initialize scaler based on config
        if self.scaler_type == "robust":
            self.scaler = RobustScaler()
        elif self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = RobustScaler()
            
        self.is_fitted = False
        self.logger = system_logger.getChild("AutoencoderPreprocessor")
        self.logger.info(f"Preprocessor initialized with scaler_type={self.scaler_type}")
    
    def fit(self, X: pd.DataFrame) -> None:
        """Fit the preprocessor on training data only."""
        self.logger.info(f"Fitting preprocessor on data with shape {X.shape}")
        try:
            X_clean = self._handle_missing_values(X)
            X_clean = self._remove_extreme_outliers(X_clean)
            self.scaler.fit(X_clean.values)
            self.is_fitted = True
            self.logger.info("Preprocessor fitted successfully")
        except Exception as e:
            self.logger.error(f"Error in fit: {e}")
            raise
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        self.logger.info(f"Transforming data with shape {X.shape}")
        try:
            X_clean = self._handle_missing_values(X)
            X_clean = self._remove_extreme_outliers(X_clean)
            X_scaled = self.scaler.transform(X_clean.values)
            X_clipped = np.clip(X_scaled, -self.outlier_threshold, self.outlier_threshold)
            self.logger.info(f"Transform complete. Output shape: {X_clipped.shape}")
            return X_clipped
        except Exception as e:
            self.logger.error(f"Error in transform: {e}")
            return X.values
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on strategy."""
        if self.missing_value_strategy == "forward_fill":
            return X.fillna(method="ffill").fillna(method="bfill").fillna(0)
        elif self.missing_value_strategy == "backward_fill":
            return X.fillna(method="bfill").fillna(method="ffill").fillna(0)
        elif self.missing_value_strategy == "interpolate":
            return X.interpolate().fillna(0)
        else:  # "zero"
            return X.fillna(0)
    
    def _remove_extreme_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove extreme outliers using IQR method."""
        self.logger.debug(f"Removing extreme outliers from data with shape {X.shape}")
        try:
            X_numeric = X.select_dtypes(include=[np.number])
            Q1 = X_numeric.quantile(0.25)
            Q3 = X_numeric.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Clip outliers
            X_clipped = X_numeric.clip(lower=lower_bound, upper=upper_bound)
            
            # Log clipping statistics
            lower_clipped_count = (X_numeric < lower_bound).sum().sum()
            upper_clipped_count = (X_numeric > upper_bound).sum().sum()
            
            if lower_clipped_count > 0 or upper_clipped_count > 0:
                self.logger.debug(f"Clipped {lower_clipped_count} lower-bound and {upper_clipped_count} upper-bound outliers")
            
            return X_clipped
        except Exception as e:
            self.logger.error(f"Error removing outliers: {e}")
            return X


def create_sequences(X: np.ndarray, timesteps: int, overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Convert 2D array to 3D sequences for CNN/LSTM models."""
    sequences = []
    targets = []
    
    step_size = max(1, int(timesteps * (1 - overlap)))
    
    for i in range(0, len(X) - timesteps, step_size):
        sequence = X[i:i + timesteps]
        target = X[i + timesteps - 1]  # Target is the last timestep
        sequences.append(sequence)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)


class SequenceAwareAutoencoder:
    """1D-CNN based sequence-aware autoencoder."""
    
    def __init__(self, config: AutoencoderConfig):
        self.config = config
        self.logger = system_logger.getChild("SequenceAwareAutoencoder")
        self.autoencoder = None
        self.encoder = None
        self.preprocessor = ImprovedAutoencoderPreprocessor(config)
        
    def build_model(self, input_shape: Tuple[int, int], trial: Optional[optuna.Trial] = None) -> Model:
        """Build 1D-CNN autoencoder model."""
        self.logger.info("Building 1D-CNN autoencoder model...")
        
        try:
            timesteps, features = input_shape
            
            # Get hyperparameters (from trial if provided, otherwise from config)
            if trial:
                filters = trial.suggest_categorical("filters", self.config.get("cnn.filters", [16, 32, 64]))
                kernel_size = trial.suggest_int("kernel_size", 3, 7)
                dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
                learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
                encoding_dim = trial.suggest_int("encoding_dim", 8, 64)
            else:
                filters = self.config.get("cnn.filters", [32])[0]
                kernel_size = self.config.get("cnn.kernel_size", [5])[0]
                dropout_rate = self.config.get("autoencoder.dropout_rate", 0.3)
                learning_rate = self.config.get("autoencoder.learning_rate", 0.001)
                encoding_dim = self.config.get("autoencoder.encoding_dim", 32)
            
            # Input layer
            input_layer = layers.Input(shape=(timesteps, features))
            
            # Encoder - 1D CNN layers
            x = layers.Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                activation=self.config.get("cnn.activation", "relu"),
                padding="same",
                kernel_regularizer=regularizers.l2(self.config.get("cnn.kernel_regularizer", 0.001))
            )(input_layer)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)
            
            x = layers.Conv1D(
                filters=filters // 2,
                kernel_size=kernel_size,
                activation=self.config.get("cnn.activation", "relu"),
                padding="same",
                kernel_regularizer=regularizers.l2(self.config.get("cnn.kernel_regularizer", 0.001))
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)
            
            # Global average pooling to get fixed-size representation
            x = layers.GlobalAveragePooling1D()(x)
            
            # Bottleneck layer
            bottleneck = layers.Dense(
                encoding_dim,
                activation="tanh",
                kernel_regularizer=regularizers.l2(self.config.get("cnn.kernel_regularizer", 0.001)),
                kernel_constraint=tf.keras.constraints.max_norm(self.config.get("cnn.max_norm_constraint", 2.0)),
                name="bottleneck"
            )(x)
            
            # Decoder - Dense layers
            x = layers.Dense(
                timesteps * features // 2,
                activation="relu",
                kernel_regularizer=regularizers.l2(self.config.get("cnn.kernel_regularizer", 0.001))
            )(bottleneck)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)
            
            x = layers.Dense(
                timesteps * features,
                activation="relu",
                kernel_regularizer=regularizers.l2(self.config.get("cnn.kernel_regularizer", 0.001))
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(dropout_rate)(x)
            
            # Reshape to original sequence shape
            output_layer = layers.Reshape((timesteps, features))(x)
            
            # Create models
            self.autoencoder = Model(input_layer, output_layer)
            self.encoder = Model(input_layer, bottleneck)
            
            # Compile
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
            self.autoencoder.compile(optimizer=optimizer, loss="huber", metrics=["mae"])
            
            self.logger.info("1D-CNN autoencoder model built and compiled successfully")
            return self.autoencoder
            
        except Exception as e:
            self.logger.error(f"Error building autoencoder model: {e}")
            raise
    
    def fit(self, X_train: np.ndarray, X_val: Optional[np.ndarray] = None, trial: Optional[optuna.Trial] = None) -> Any:
        """Train the autoencoder."""
        self.logger.info("Training sequence-aware autoencoder...")
        
        try:
            # Prepare callbacks
            callbacks = []
            
            # Early stopping
            early_stopping = EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=self.config.get("autoencoder.early_stopping_patience", 10),
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
            
            # Learning rate reduction
            reduce_lr = ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=self.config.get("autoencoder.reduce_lr_patience", 5),
                min_lr=self.config.get("autoencoder.min_lr", 1e-6),
                verbose=1
            )
            callbacks.append(reduce_lr)
            
            # Optuna pruning callback
            if trial and self.config.get("training.pruning_enabled", True):
                pruning_callback = TFKerasPruningCallback(
                    trial, 
                    self.config.get("training.pruning_metric", "val_loss")
                )
                callbacks.append(pruning_callback)
            
            # Training parameters
            batch_size = self.config.get("autoencoder.batch_size", 32)
            epochs = self.config.get("autoencoder.epochs", 100)
            validation_split = self.config.get("autoencoder.validation_split", 0.2)
            
            # Train
            history = self.autoencoder.fit(
                X_train, X_train,
                validation_data=(X_val, X_val) if X_val is not None else None,
                validation_split=validation_split if X_val is None else 0,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            self.logger.info("Autoencoder training completed successfully")
            return history
            
        except Exception as e:
            self.logger.error(f"Error during autoencoder training: {e}")
            raise
    
    def get_encoded_features(self, X: np.ndarray) -> np.ndarray:
        """Get encoded features from the bottleneck layer."""
        self.logger.info(f"Getting encoded features for data with shape {X.shape}")
        try:
            encoded_features = self.encoder.predict(X, verbose=0)
            self.logger.info(f"Encoded features shape: {encoded_features.shape}")
            return encoded_features
        except Exception as e:
            self.logger.error(f"Error getting encoded features: {e}")
            return np.zeros((X.shape[0], self.config.get("autoencoder.encoding_dim", 32)))


class OptunaOptimizer:
    """Optuna-based hyperparameter optimization for autoencoder."""
    
    def __init__(self, config: AutoencoderConfig):
        self.config = config
        self.logger = system_logger.getChild("OptunaOptimizer")
        self.best_params = None
        self.best_score = float('inf')
        
    def objective(self, trial: optuna.Trial, X_train: np.ndarray, X_val: np.ndarray) -> float:
        """Objective function for Optuna optimization."""
        try:
            # Build model with trial parameters
            autoencoder = SequenceAwareAutoencoder(self.config)
            model = autoencoder.build_model(X_train.shape[1:], trial)
            
            # Train model
            history = autoencoder.fit(X_train, X_val, trial)
            
            # Get validation loss
            val_loss = min(history.history['val_loss'])
            
            # Update best score
            if val_loss < self.best_score:
                self.best_score = val_loss
                self.best_params = trial.params
            
            return val_loss
            
        except Exception as e:
            self.logger.error(f"Error in objective function: {e}")
            return float('inf')
    
    def optimize(self, X_train: np.ndarray, X_val: np.ndarray) -> Dict[str, Any]:
        """Run Optuna optimization."""
        self.logger.info("Starting Optuna hyperparameter optimization...")
        
        try:
            study = optuna.create_study(
                direction="minimize",
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=5,
                    n_warmup_steps=10,
                    interval_steps=1
                ) if self.config.get("training.pruning_enabled", True) else None
            )
            
            study.optimize(
                lambda trial: self.objective(trial, X_train, X_val),
                n_trials=self.config.get("training.n_trials", 50),
                timeout=self.config.get("training.timeout_seconds", 1800),
                n_jobs=self.config.get("training.n_jobs", -1)
            )
            
            self.logger.info(f"Optimization completed. Best score: {study.best_value}")
            self.logger.info(f"Best parameters: {study.best_params}")
            
            return study.best_params
            
        except Exception as e:
            self.logger.error(f"Error in optimization: {e}")
            return {}


class AutoencoderFeatureGenerator:
    """Main class for autoencoder feature generation with all improvements."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = AutoencoderConfig(config_path)
        self.logger = system_logger.getChild("AutoencoderFeatureGenerator")
        self.triple_barrier = TripleBarrierLabeler(self.config)
        self.feature_filter = FeatureFilter(self.config)
        self.autoencoder = None
        self.optimizer = OptunaOptimizer(self.config)
        
    @handle_errors(
        exceptions=(ValueError, AttributeError, KeyError),
        default_return=pd.DataFrame(),
        context="autoencoder feature generation",
    )
    def generate_features(self, features_df: pd.DataFrame, regime_name: str) -> pd.DataFrame:
        """
        Generate autoencoder-based features for a specific market regime.
        
        Args:
            features_df: Input features DataFrame
            regime_name: Name of the market regime
            
        Returns:
            DataFrame with original features plus autoencoder features
        """
        self.logger.info(f"🚀 Starting autoencoder feature generation for regime: {regime_name}")
        
        if features_df.empty:
            self.logger.warning("⚠️ Empty features DataFrame provided")
            return features_df
        
        try:
            # Step 1: Apply triple barrier labeling
            self.logger.info("Step 1: Applying triple barrier labeling...")
            labels = self.triple_barrier.apply_triple_barrier(features_df)
            
            # Step 2: Filter features using Random Forest + SHAP
            self.logger.info("Step 2: Filtering features with Random Forest + SHAP...")
            filtered_features = self.feature_filter.filter_features(features_df, labels)
            
            if filtered_features.empty:
                self.logger.warning("No features selected after filtering")
                return features_df
            
            # Step 3: Prepare sequences for sequence-aware model
            self.logger.info("Step 3: Preparing sequences...")
            timesteps = self.config.get("sequence.timesteps", 10)
            overlap = self.config.get("sequence.overlap", 0.5)
            
            # Preprocess features
            preprocessor = ImprovedAutoencoderPreprocessor(self.config)
            X_processed = preprocessor.fit_transform(filtered_features)
            
            # Create sequences
            X_sequences, X_targets = create_sequences(X_processed, timesteps, overlap)
            
            if len(X_sequences) < 10:
                self.logger.warning("Insufficient sequences for training")
                return features_df
            
            # Step 4: Optimize hyperparameters with Optuna
            self.logger.info("Step 4: Optimizing hyperparameters with Optuna...")
            split_idx = int(0.8 * len(X_sequences))
            X_train_seq = X_sequences[:split_idx]
            X_val_seq = X_sequences[split_idx:]
            
            best_params = self.optimizer.optimize(X_train_seq, X_val_seq)
            
            # Step 5: Train final autoencoder with best parameters
            self.logger.info("Step 5: Training final autoencoder...")
            self.autoencoder = SequenceAwareAutoencoder(self.config)
            
            # Update config with best parameters
            for key, value in best_params.items():
                if key == "filters":
                    self.config.config["cnn"]["filters"] = [value]
                elif key == "kernel_size":
                    self.config.config["cnn"]["kernel_size"] = [value]
                elif key == "dropout_rate":
                    self.config.config["autoencoder"]["dropout_rate"] = value
                elif key == "learning_rate":
                    self.config.config["autoencoder"]["learning_rate"] = value
                elif key == "encoding_dim":
                    self.config.config["autoencoder"]["encoding_dim"] = value
            
            # Build and train model
            model = self.autoencoder.build_model(X_sequences.shape[1:])
            history = self.autoencoder.fit(X_sequences, X_val_seq)
            
            # Step 6: Generate encoded features
            self.logger.info("Step 6: Generating encoded features...")
            encoded_features = self.autoencoder.get_encoded_features(X_sequences)
            
            # Step 7: Add features to original DataFrame
            result_df = features_df.copy()
            
            # Add encoded features
            for i in range(encoded_features.shape[1]):
                col_name = f"autoencoder_feature_{i+1}"
                # Align with original data length
                if len(encoded_features) < len(result_df):
                    # Pad with zeros if needed
                    padded_features = np.zeros(len(result_df))
                    padded_features[:len(encoded_features)] = encoded_features[:, i]
                    result_df[col_name] = padded_features
                else:
                    result_df[col_name] = encoded_features[:len(result_df), i]
            
            # Add reconstruction error
            reconstructed = self.autoencoder.autoencoder.predict(X_sequences, verbose=0)
            reconstruction_error = np.mean((X_targets - reconstructed) ** 2, axis=1)
            
            if len(reconstruction_error) < len(result_df):
                padded_error = np.zeros(len(result_df))
                padded_error[:len(reconstruction_error)] = reconstruction_error
                result_df["autoencoder_reconstruction_error"] = padded_error
            else:
                result_df["autoencoder_reconstruction_error"] = reconstruction_error[:len(result_df)]
            
            # Step 8: Save models and configuration
            self.logger.info("Step 8: Saving models and configuration...")
            self._save_models_and_config(regime_name)
            
            self.logger.info("✅ Autoencoder feature generation completed successfully")
            self.logger.info(f"📊 Added {encoded_features.shape[1] + 1} new features")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"❌ Error in autoencoder feature generation: {e}")
            return features_df
    
    def _save_models_and_config(self, regime_name: str) -> None:
        """Save trained models and configuration."""
        try:
            output_dir = self.config.get("output.output_dir", "models/autoencoder_features")
            regime_dir = os.path.join(output_dir, regime_name)
            os.makedirs(regime_dir, exist_ok=True)
            
            # Save autoencoder
            if self.config.get("output.save_autoencoder", True) and self.autoencoder:
                autoencoder_path = os.path.join(regime_dir, "autoencoder.h5")
                self.autoencoder.autoencoder.save(autoencoder_path)
                self.logger.info(f"Autoencoder saved to {autoencoder_path}")
                
                # Save encoder separately
                encoder_path = os.path.join(regime_dir, "encoder.h5")
                self.autoencoder.encoder.save(encoder_path)
                self.logger.info(f"Encoder saved to {encoder_path}")
            
            # Save preprocessor
            if self.config.get("output.save_preprocessor", True):
                preprocessor_path = os.path.join(regime_dir, "preprocessor.pkl")
                with open(preprocessor_path, 'wb') as f:
                    pickle.dump(self.autoencoder.preprocessor, f)
                self.logger.info(f"Preprocessor saved to {preprocessor_path}")
            
            # Save configuration
            if self.config.get("output.save_config", True):
                config_path = os.path.join(regime_dir, self.config.get("output.config_filename", "config.yaml"))
                self.config.save_config(config_path)
                self.logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving models and config: {e}")
    
    def get_generator_info(self) -> Dict[str, Any]:
        """Get information about the autoencoder feature generator."""
        try:
            return {
                "generator_type": "AutoencoderFeatureGenerator",
                "config": self.config.config,
                "autoencoder_initialized": self.autoencoder is not None,
                "best_optimization_score": self.optimizer.best_score,
                "best_optimization_params": self.optimizer.best_params,
            }
        except Exception as e:
            self.logger.error(f"Error getting generator info: {e}")
            return {}


# Legacy compatibility function
def generate_autoencoder_features(features_df: pd.DataFrame, config_path: Optional[str] = None) -> pd.DataFrame:
    """Legacy function for backward compatibility."""
    generator = AutoencoderFeatureGenerator(config_path)
    return generator.generate_features(features_df, "default_regime")
