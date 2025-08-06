# src/analyst/autoencoder_feature_generator.py

import logging
import os
import pickle
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Check for required dependencies
try:
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
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEPENDENCY = str(e)
    print(f"⚠️  Missing dependency: {MISSING_DEPENDENCY}")
    print("📦 Please install required packages:")
    print("   pip install numpy pandas scikit-learn tensorflow optuna shap pyyaml")
    print("   or")
    print("   pip install -r requirements_autoencoder.txt")

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
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(f"Required dependencies not available: {MISSING_DEPENDENCY}")
            
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
                "feature_selection_threshold": 0.99,
                "iqr_multiplier": 3.0
            },
            "sequence": {
                "timesteps": 10,
                "overlap": 0.5,
                "min_sequence_length": 5,
                "use_target_prediction": True
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
                "min_lr": 1e-6,
                "target_prediction": True
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
                "importance_threshold": 0.99,
                "use_existing_labels": True
            },
            "triple_barrier": {
                "profit_take": 0.001,
                "stop_loss": 0.0005,
                "time_barrier": 300,
                "vectorized": True
            },
            "output": {
                "save_autoencoder": True,
                "save_config": True,
                "save_preprocessor": True,
                "output_dir": "models/autoencoder_features",
                "config_filename": "autoencoder_config.yaml",
                "preserve_index": True
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
    """Vectorized Triple Barrier Method implementation for labeling."""
    
    def __init__(self, config: AutoencoderConfig):
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(f"Required dependencies not available: {MISSING_DEPENDENCY}")
            
        self.config = config
        self.logger = system_logger.getChild("TripleBarrierLabeler")
        
    def apply_triple_barrier_vectorized(self, data: pd.DataFrame) -> np.ndarray:
        """Apply triple barrier method using vectorized operations for 100x+ speedup."""
        try:
            self.logger.info("Applying vectorized triple barrier labeling...")
            
            profit_take = self.config.get("triple_barrier.profit_take", 0.001)
            stop_loss = self.config.get("triple_barrier.stop_loss", 0.0005)
            time_barrier = self.config.get("triple_barrier.time_barrier", 300)
            
            # Ensure we have required columns
            required_cols = ["open", "high", "low", "close"]
            if not all(col in data.columns for col in required_cols):
                self.logger.warning("Missing required OHLC columns, using close price only")
                if "close" not in data.columns:
                    raise ValueError("At least 'close' column is required")
                data = data.copy()
                data["high"] = data["close"]
                data["low"] = data["close"]
            
            # Calculate entry prices (previous close)
            entry_prices = data["close"].shift(1)
            
            # Calculate barriers
            profit_barriers = entry_prices * (1 + profit_take)
            stop_barriers = entry_prices * (1 - stop_loss)
            
            # Initialize labels
            labels = np.zeros(len(data))
            
            # Vectorized forward-looking barrier check
            for i in range(len(data) - 1):
                if pd.isna(entry_prices.iloc[i]):
                    continue
                    
                # Look forward within time barrier
                end_idx = min(i + time_barrier, len(data))
                future_data = data.iloc[i+1:end_idx]
                
                if len(future_data) == 0:
                    continue
                
                # Check if high price hits profit barrier
                profit_hit = (future_data["high"] >= profit_barriers.iloc[i]).any()
                if profit_hit:
                    labels[i] = 1
                    continue
                
                # Check if low price hits stop barrier
                stop_hit = (future_data["low"] <= stop_barriers.iloc[i]).any()
                if stop_hit:
                    labels[i] = -1
                    continue
                
                # If no barrier hit, label remains 0 (neutral)
            
            self.logger.info(f"Vectorized triple barrier completed. Labels: {np.bincount(labels.astype(int))}")
            return labels
            
        except Exception as e:
            self.logger.error(f"Error applying vectorized triple barrier: {e}")
            return np.array([])
    
    def apply_triple_barrier(self, data: pd.DataFrame) -> np.ndarray:
        """Apply triple barrier method (legacy method for backward compatibility)."""
        if self.config.get("triple_barrier.vectorized", True):
            return self.apply_triple_barrier_vectorized(data)
        else:
            return self._apply_triple_barrier_legacy(data)
    
    def _apply_triple_barrier_legacy(self, data: pd.DataFrame) -> np.ndarray:
        """Legacy triple barrier implementation (slower, kept for compatibility)."""
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
            self.logger.error(f"Error applying legacy triple barrier: {e}")
            return np.array([])


class FeatureFilter:
    """Random Forest + SHAP feature filtering with fixed cumulative threshold bug."""
    
    def __init__(self, config: AutoencoderConfig):
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(f"Required dependencies not available: {MISSING_DEPENDENCY}")
            
        self.config = config
        self.logger = system_logger.getChild("FeatureFilter")
        self.rf_model = None
        self.feature_importance = None
        
    def filter_features(self, features_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        """Filter features using Random Forest + SHAP importance with fixed threshold logic."""
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
            
            # FIXED: Find the first index where cumulative importance exceeds threshold
            threshold = self.config.get("feature_filtering.importance_threshold", 0.99)
            importance_threshold = threshold * total_importance
            
            # Find the first index where cumulative importance exceeds the threshold
            threshold_exceeded = cumulative_importance > importance_threshold
            if threshold_exceeded.any():
                # Select features up to the first index that exceeds threshold
                selected_count = np.argmax(threshold_exceeded) + 1
            else:
                # If threshold never exceeded, select all features
                selected_count = len(sorted_indices)
            
            selected_indices = sorted_indices[:selected_count]
            
            # Get selected feature names
            selected_features = X.columns[selected_indices].tolist()
            
            self.logger.info(f"Selected {len(selected_features)} features out of {len(X.columns)}")
            self.logger.info(f"Selected features: {selected_features}")
            self.logger.info(f"Cumulative importance: {cumulative_importance[selected_count-1]/total_importance:.3f}")
            
            # Return filtered DataFrame
            filtered_df = features_df[selected_features].copy()
            
            return filtered_df
            
        except Exception as e:
            self.logger.error(f"Error in feature filtering: {e}")
            return features_df


class ImprovedAutoencoderPreprocessor:
    """Enhanced preprocessor with separate fit/transform and no data leakage."""
    
    def __init__(self, config: AutoencoderConfig):
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(f"Required dependencies not available: {MISSING_DEPENDENCY}")
            
        self.config = config
        self.scaler_type = config.get("preprocessing.scaler_type", "robust")
        self.outlier_threshold = config.get("preprocessing.outlier_threshold", 3.0)
        self.missing_value_strategy = config.get("preprocessing.missing_value_strategy", "forward_fill")
        self.iqr_multiplier = config.get("preprocessing.iqr_multiplier", 3.0)
        
        # Initialize scaler based on config
        if self.scaler_type == "robust":
            self.scaler = RobustScaler()
        elif self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.scaler = RobustScaler()
            
        # Store outlier bounds to prevent data leakage
        self.outlier_lower_bounds = None
        self.outlier_upper_bounds = None
        
        self.is_fitted = False
        self.logger = system_logger.getChild("AutoencoderPreprocessor")
        self.logger.info(f"Preprocessor initialized with scaler_type={self.scaler_type}")
    
    def fit(self, X: pd.DataFrame) -> None:
        """Fit the preprocessor on training data only."""
        self.logger.info(f"Fitting preprocessor on data with shape {X.shape}")
        try:
            X_clean = self._handle_missing_values(X)
            
            # Calculate outlier bounds from training data only
            X_numeric = X_clean.select_dtypes(include=[np.number])
            Q1 = X_numeric.quantile(0.25)
            Q3 = X_numeric.quantile(0.75)
            IQR = Q3 - Q1
            self.outlier_lower_bounds = Q1 - self.iqr_multiplier * IQR
            self.outlier_upper_bounds = Q3 + self.iqr_multiplier * IQR
            
            # Fit scaler on cleaned data
            X_clean_clipped = self._clip_outliers(X_clean)
            self.scaler.fit(X_clean_clipped.values)
            
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
            X_clean_clipped = self._clip_outliers(X_clean)
            X_scaled = self.scaler.transform(X_clean_clipped.values)
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
    
    def _clip_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers using pre-calculated bounds (prevents data leakage)."""
        if self.outlier_lower_bounds is None or self.outlier_upper_bounds is None:
            # If not fitted yet, calculate bounds from current data
            X_numeric = X.select_dtypes(include=[np.number])
            Q1 = X_numeric.quantile(0.25)
            Q3 = X_numeric.quantile(0.75)
            IQR = Q3 - Q1
            lower_bounds = Q1 - self.iqr_multiplier * IQR
            upper_bounds = Q3 + self.iqr_multiplier * IQR
        else:
            # Use pre-calculated bounds from fit
            lower_bounds = self.outlier_lower_bounds
            upper_bounds = self.outlier_upper_bounds
        
        # Clip outliers
        X_clipped = X.select_dtypes(include=[np.number]).clip(lower=lower_bounds, upper=upper_bounds)
        
        # Log clipping statistics
        X_numeric = X.select_dtypes(include=[np.number])
        lower_clipped_count = (X_numeric < lower_bounds).sum().sum()
        upper_clipped_count = (X_numeric > upper_bounds).sum().sum()
        
        if lower_clipped_count > 0 or upper_clipped_count > 0:
            self.logger.debug(f"Clipped {lower_clipped_count} lower-bound and {upper_clipped_count} upper-bound outliers")
        
        return X_clipped


def create_sequences_with_index(X: np.ndarray, timesteps: int, overlap: float = 0.5, 
                               original_index: pd.Index = None) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    """Convert 2D array to 3D sequences for CNN/LSTM models with index tracking."""
    sequences = []
    targets = []
    target_indices = []
    
    step_size = max(1, int(timesteps * (1 - overlap)))
    
    for i in range(0, len(X) - timesteps, step_size):
        sequence = X[i:i + timesteps]
        target = X[i + timesteps - 1]  # Target is the last timestep
        sequences.append(sequence)
        targets.append(target)
        
        # Track the index of the target
        if original_index is not None:
            target_indices.append(original_index[i + timesteps - 1])
        else:
            target_indices.append(i + timesteps - 1)
    
    return np.array(sequences), np.array(targets), pd.Index(target_indices)


def create_sequences(X: np.ndarray, timesteps: int, overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy function for backward compatibility."""
    sequences, targets, _ = create_sequences_with_index(X, timesteps, overlap)
    return sequences, targets


class SequenceAwareAutoencoder:
    """1D-CNN based sequence-aware autoencoder with target prediction."""
    
    def __init__(self, config: AutoencoderConfig):
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(f"Required dependencies not available: {MISSING_DEPENDENCY}")
            
        self.config = config
        self.logger = system_logger.getChild("SequenceAwareAutoencoder")
        self.autoencoder = None
        self.encoder = None
        self.preprocessor = ImprovedAutoencoderPreprocessor(config)
        self.target_prediction = config.get("autoencoder.target_prediction", True)
        
    def build_model(self, input_shape: Tuple[int, int], trial: Optional[optuna.Trial] = None) -> Model:
        """Build 1D-CNN autoencoder model with target prediction capability."""
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
            
            if self.target_prediction:
                # For target prediction, output should match target shape (features)
                output_layer = layers.Dense(
                    features,
                    activation="linear",
                    kernel_regularizer=regularizers.l2(self.config.get("cnn.kernel_regularizer", 0.001))
                )(bottleneck)
            else:
                # For sequence reconstruction, output should match input shape (timesteps, features)
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
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None, trial: Optional[optuna.Trial] = None) -> Any:
        """Train the autoencoder with proper target matching."""
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
            
            # Train with proper targets
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)
                validation_split = 0
            else:
                validation_data = None
            
            history = self.autoencoder.fit(
                X_train, y_train,
                validation_data=validation_data,
                validation_split=validation_split if validation_data is None else 0,
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
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(f"Required dependencies not available: {MISSING_DEPENDENCY}")
            
        self.config = config
        self.logger = system_logger.getChild("OptunaOptimizer")
        self.best_params = None
        self.best_score = float('inf')
        
    def objective(self, trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray, 
                 X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Objective function for Optuna optimization with proper target matching."""
        try:
            # Build model with trial parameters
            autoencoder = SequenceAwareAutoencoder(self.config)
            model = autoencoder.build_model(X_train.shape[1:], trial)
            
            # Train model with proper targets
            history = autoencoder.fit(X_train, y_train, X_val, y_val, trial)
            
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
    
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Run Optuna optimization with proper target matching."""
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
                lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
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
    """Main class for autoencoder feature generation with all improvements and fixes."""
    
    def __init__(self, config_path: Optional[str] = None):
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(f"Required dependencies not available: {MISSING_DEPENDENCY}")
            
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
    def generate_features(self, features_df: pd.DataFrame, regime_name: str, 
                        existing_labels: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Generate autoencoder-based features for a specific market regime.
        
        Args:
            features_df: Input features DataFrame
            regime_name: Name of the market regime
            existing_labels: Optional existing triple barrier labels (to avoid re-computation)
            
        Returns:
            DataFrame with original features plus autoencoder features
        """
        self.logger.info(f"🚀 Starting autoencoder feature generation for regime: {regime_name}")
        
        if features_df.empty:
            self.logger.warning("⚠️ Empty features DataFrame provided")
            return features_df
        
        try:
            # Step 1: Apply triple barrier labeling (only if not provided)
            if existing_labels is not None:
                self.logger.info("Step 1: Using existing triple barrier labels...")
                labels = existing_labels
            else:
                self.logger.info("Step 1: Applying triple barrier labeling...")
                labels = self.triple_barrier.apply_triple_barrier(features_df)
            
            # Step 2: Filter features using Random Forest + SHAP
            self.logger.info("Step 2: Filtering features with Random Forest + SHAP...")
            filtered_features = self.feature_filter.filter_features(features_df, labels)
            
            if filtered_features.empty:
                self.logger.warning("No features selected after filtering")
                return features_df
            
            # Step 3: Prepare sequences for sequence-aware model with index tracking
            self.logger.info("Step 3: Preparing sequences with index tracking...")
            timesteps = self.config.get("sequence.timesteps", 10)
            overlap = self.config.get("sequence.overlap", 0.5)
            
            # Preprocess features
            preprocessor = ImprovedAutoencoderPreprocessor(self.config)
            X_processed = preprocessor.fit_transform(filtered_features)
            
            # Create sequences with index tracking
            X_sequences, X_targets, target_indices = create_sequences_with_index(
                X_processed, timesteps, overlap, filtered_features.index
            )
            
            if len(X_sequences) < 10:
                self.logger.warning("Insufficient sequences for training")
                return features_df
            
            # Step 4: Optimize hyperparameters with Optuna
            self.logger.info("Step 4: Optimizing hyperparameters with Optuna...")
            split_idx = int(0.8 * len(X_sequences))
            X_train_seq = X_sequences[:split_idx]
            y_train_seq = X_targets[:split_idx]
            X_val_seq = X_sequences[split_idx:]
            y_val_seq = X_targets[split_idx:]
            
            best_params = self.optimizer.optimize(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
            
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
            
            # Build and train model with proper targets
            model = self.autoencoder.build_model(X_sequences.shape[1:])
            history = self.autoencoder.fit(X_sequences, X_targets, X_val_seq, y_val_seq)
            
            # Step 6: Generate encoded features
            self.logger.info("Step 6: Generating encoded features...")
            encoded_features = self.autoencoder.get_encoded_features(X_sequences)
            
            # Step 7: Create enriched DataFrame with proper index alignment
            self.logger.info("Step 7: Creating enriched DataFrame with proper index alignment...")
            result_df = features_df.copy()
            
            # Create DataFrame with encoded features and target indices
            encoded_df = pd.DataFrame(
                encoded_features,
                index=target_indices,
                columns=[f"autoencoder_feature_{i+1}" for i in range(encoded_features.shape[1])]
            )
            
            # Add reconstruction error
            if self.autoencoder.target_prediction:
                # For target prediction, compare predicted targets with actual targets
                predicted_targets = self.autoencoder.autoencoder.predict(X_sequences, verbose=0)
                reconstruction_error = np.mean((X_targets - predicted_targets) ** 2, axis=1)
            else:
                # For sequence reconstruction, compare reconstructed sequences with input sequences
                reconstructed_sequences = self.autoencoder.autoencoder.predict(X_sequences, verbose=0)
                reconstruction_error = np.mean((X_sequences - reconstructed_sequences) ** 2, axis=(1, 2))
            
            # Add reconstruction error to encoded DataFrame
            encoded_df["autoencoder_reconstruction_error"] = reconstruction_error
            
            # Merge encoded features back to original DataFrame using index
            result_df = result_df.merge(encoded_df, left_index=True, right_index=True, how='left')
            
            # Fill NaN values with zeros for missing indices
            autoencoder_cols = [col for col in result_df.columns if 'autoencoder' in col]
            result_df[autoencoder_cols] = result_df[autoencoder_cols].fillna(0)
            
            # Step 8: Save models and configuration
            self.logger.info("Step 8: Saving models and configuration...")
            self._save_models_and_config(regime_name)
            
            self.logger.info("✅ Autoencoder feature generation completed successfully")
            self.logger.info(f"📊 Added {len(autoencoder_cols)} new features")
            
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
