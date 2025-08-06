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

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
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
            self.logger.error(f"Error loading config: {e}, using default config.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails."""
        # This default config is a fallback and should be customized in the YAML file.
        return {
            "preprocessing": {
                "scaler_type": "robust", "outlier_threshold": 3.0,
                "missing_value_strategy": "forward_fill", "iqr_multiplier": 3.0
            },
            "sequence": {"timesteps": 10, "overlap": 0.5},
            "autoencoder": {
                "epochs": 100, "early_stopping_patience": 10, "reduce_lr_patience": 5, "min_lr": 1e-6
            },
            "training": {"n_trials": 50, "n_jobs": 1, "pruning_enabled": True},
            "feature_filtering": {"n_estimators": 100, "max_depth": 10, "importance_threshold": 0.99},
            "output": {"output_dir": "models/autoencoder_features"}
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


class FeatureFilter:
    """Random Forest + SHAP feature filtering."""
    
    def __init__(self, config: AutoencoderConfig):
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(f"Required dependencies not available: {MISSING_DEPENDENCY}")
        self.config = config
        self.logger = system_logger.getChild("FeatureFilter")
        
    def filter_features(self, features_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        """Filter features using Random Forest + SHAP importance."""
        try:
            self.logger.info("Starting feature filtering with Random Forest + SHAP...")
            
            X = features_df.select_dtypes(include=[np.number]).fillna(0)
            y = labels
            
            if len(np.unique(y)) < 2:
                self.logger.warning("Insufficient unique labels for classification, skipping filtering.")
                return features_df
            
            rf_model = RandomForestClassifier(
                n_estimators=self.config.get("feature_filtering.n_estimators", 100),
                max_depth=self.config.get("feature_filtering.max_depth", 10),
                random_state=self.config.get("feature_filtering.random_state", 42),
                n_jobs=-1
            )
            rf_model.fit(X, y)
            
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(X)
            
            # Use mean absolute SHAP values across all classes for importance
            if isinstance(shap_values, list): # Multi-class case
                feature_importance = np.mean([np.abs(sv) for sv in shap_values], axis=(0, 1))
            else: # Binary case
                feature_importance = np.mean(np.abs(shap_values), axis=0)

            sorted_indices = np.argsort(feature_importance)[::-1]
            sorted_importance = feature_importance[sorted_indices]
            cumulative_importance = np.cumsum(sorted_importance)
            total_importance = cumulative_importance[-1]
            
            threshold = self.config.get("feature_filtering.importance_threshold", 0.99)
            importance_cutoff = threshold * total_importance
            
            # CORRECTED LOGIC: Find the first index where cumulative importance exceeds the threshold
            cutoff_index = np.where(cumulative_importance >= importance_cutoff)[0][0] + 1
            selected_indices = sorted_indices[:cutoff_index]
            
            selected_features = X.columns[selected_indices].tolist()
            
            self.logger.info(f"Selected {len(selected_features)} features out of {len(X.columns)} to capture {threshold*100}% of importance.")
            
            return features_df[selected_features].copy()
            
        except Exception as e:
            self.logger.error(f"Error in feature filtering: {e}")
            return features_df


class ImprovedAutoencoderPreprocessor:
    """Enhanced preprocessor with separate fit/transform and no data leakage."""
    
    def __init__(self, config: AutoencoderConfig):
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(f"Required dependencies not available: {MISSING_DEPENDENCY}")
        self.config = config
        scaler_type = config.get("preprocessing.scaler_type", "robust")
        
        if scaler_type == "robust": self.scaler = RobustScaler()
        elif scaler_type == "standard": self.scaler = StandardScaler()
        else: self.scaler = MinMaxScaler()
            
        self.outlier_lower_bounds_ = None
        self.outlier_upper_bounds_ = None
        self.is_fitted = False
        self.logger = system_logger.getChild("AutoencoderPreprocessor")

    def fit(self, X: pd.DataFrame) -> 'ImprovedAutoencoderPreprocessor':
        """Fit the preprocessor on training data only."""
        self.logger.info(f"Fitting preprocessor on data with shape {X.shape}")
        X_clean = self._handle_missing_values(X)
        
        X_numeric = X_clean.select_dtypes(include=[np.number])
        Q1 = X_numeric.quantile(0.25)
        Q3 = X_numeric.quantile(0.75)
        IQR = Q3 - Q1
        iqr_mult = self.config.get("preprocessing.iqr_multiplier", 3.0)
        self.outlier_lower_bounds_ = Q1 - iqr_mult * IQR
        self.outlier_upper_bounds_ = Q3 + iqr_mult * IQR
        
        X_clipped = X_numeric.clip(lower=self.outlier_lower_bounds_, upper=self.outlier_upper_bounds_)
        self.scaler.fit(X_clipped.values)
        
        self.is_fitted = True
        self.logger.info("Preprocessor fitted successfully")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform can be called.")
        
        X_clean = self._handle_missing_values(X)
        X_clipped = self._clip_outliers(X_clean)
        X_scaled = self.scaler.transform(X_clipped.values)
        
        final_threshold = self.config.get("preprocessing.outlier_threshold", 3.0)
        return np.clip(X_scaled, -final_threshold, final_threshold)

    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on strategy."""
        strategy = self.config.get("preprocessing.missing_value_strategy", "forward_fill")
        if strategy == "forward_fill":
            return X.fillna(method="ffill").fillna(method="bfill").fillna(0)
        else: # Default to zero fill
            return X.fillna(0)

    def _clip_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers using pre-calculated bounds to prevent data leakage."""
        X_numeric = X.select_dtypes(include=[np.number])
        return X_numeric.clip(lower=self.outlier_lower_bounds_, upper=self.outlier_upper_bounds_)


def create_sequences_with_index(X: np.ndarray, timesteps: int, original_index: pd.Index) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
    """Convert 2D array to 3D sequences, tracking the index of the target."""
    sequences, targets, target_indices = [], [], []
    
    for i in range(len(X) - timesteps + 1):
        sequence = X[i : i + timesteps]
        target = X[i + timesteps - 1]
        sequences.append(sequence)
        targets.append(target)
        target_indices.append(original_index[i + timesteps - 1])
    
    return np.array(sequences), np.array(targets), pd.Index(target_indices)


class SequenceAwareAutoencoder:
    """1D-CNN based autoencoder that learns to reconstruct the last timestep of a sequence."""
    
    def __init__(self, config: AutoencoderConfig):
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(f"Required dependencies not available: {MISSING_DEPENDENCY}")
        self.config = config
        self.logger = system_logger.getChild("SequenceAwareAutoencoder")
        self.autoencoder = None
        self.encoder = None
        
    def build_model(self, input_shape: Tuple[int, int], trial: Optional[optuna.Trial] = None) -> Model:
        """Build 1D-CNN autoencoder model."""
        timesteps, features = input_shape
        
        if trial:
            filters = trial.suggest_categorical("filters", [16, 32, 64])
            kernel_size = trial.suggest_int("kernel_size", 3, 7)
            dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            encoding_dim = trial.suggest_int("encoding_dim", 8, 64)
        else: # Fallback to config for final training
            encoding_dim = self.config.get("autoencoder.encoding_dim", 32)
            # Use a dictionary for best_params from Optuna
            best_params = self.config.get("best_params", {})
            filters = best_params.get("filters", 32)
            kernel_size = best_params.get("kernel_size", 5)
            dropout_rate = best_params.get("dropout_rate", 0.3)
            learning_rate = best_params.get("learning_rate", 0.001)

        input_layer = layers.Input(shape=(timesteps, features))
        
        # Encoder
        x = layers.Conv1D(filters=filters, kernel_size=kernel_size, activation="relu", padding="same")(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Conv1D(filters=filters // 2, kernel_size=kernel_size, activation="relu", padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        
        bottleneck = layers.Dense(encoding_dim, activation="tanh", name="bottleneck")(x)
        
        # Decoder - Reconstructs the feature vector of the last timestep
        output_layer = layers.Dense(features, activation="linear")(bottleneck)
        
        self.autoencoder = Model(inputs=input_layer, outputs=output_layer)
        self.encoder = Model(inputs=input_layer, outputs=bottleneck)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.autoencoder.compile(optimizer=optimizer, loss="huber", metrics=["mae"])
        return self.autoencoder
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, trial: Optional[optuna.Trial] = None) -> Any:
        """Train the autoencoder."""
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=self.config.get("autoencoder.early_stopping_patience", 10), restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", patience=self.config.get("autoencoder.reduce_lr_patience", 5), min_lr=self.config.get("autoencoder.min_lr", 1e-6))
        ]
        if trial and self.config.get("training.pruning_enabled", True):
            callbacks.append(TFKerasPruningCallback(trial, "val_loss"))
        
        history = self.autoencoder.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.get("autoencoder.epochs", 100),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]) if trial else self.config.get("best_params", {}).get("batch_size", 32),
            callbacks=callbacks,
            verbose=0
        )
        return history


class AutoencoderFeatureGenerator:
    """Main class for the complete autoencoder feature generation workflow."""
    
    def __init__(self, config_path: Optional[str] = None):
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(f"Required dependencies not available: {MISSING_DEPENDENCY}")
        self.config = AutoencoderConfig(config_path)
        self.logger = system_logger.getChild("AutoencoderFeatureGenerator")
        
    def generate_features(self, features_df: pd.DataFrame, regime_name: str, labels: np.ndarray) -> pd.DataFrame:
        """Generate autoencoder-based features for a specific market regime."""
        self.logger.info(f"🚀 Starting autoencoder feature generation for regime: {regime_name}")
        
        # Step 1: Filter features using Random Forest + SHAP
        feature_filter = FeatureFilter(self.config)
        filtered_features = feature_filter.filter_features(features_df, labels)
        
        # Step 2: Preprocess and create sequences
        preprocessor = ImprovedAutoencoderPreprocessor(self.config)
        preprocessor.fit(filtered_features)
        X_processed = preprocessor.transform(filtered_features)
        
        timesteps = self.config.get("sequence.timesteps", 10)
        X_sequences, y_targets, target_indices = create_sequences_with_index(X_processed, timesteps, filtered_features.index)
        
        # Step 3: Optimize hyperparameters with Optuna
        split_idx = int(0.8 * len(X_sequences))
        X_train, y_train = X_sequences[:split_idx], y_targets[:split_idx]
        X_val, y_val = X_sequences[split_idx:], y_targets[split_idx:]
        
        best_params = self._run_optuna_optimization(X_train, y_train, X_val, y_val)
        self.config.config['best_params'] = best_params # Store best params for final training

        # Step 4: Train final model and generate features
        final_autoencoder = SequenceAwareAutoencoder(self.config)
        final_autoencoder.build_model(X_sequences.shape[1:])
        final_autoencoder.fit(X_train, y_train, X_val, y_val)
        
        encoded_features = final_autoencoder.encoder.predict(X_sequences)
        reconstructed = final_autoencoder.autoencoder.predict(X_sequences)
        recon_error = np.mean((y_targets - reconstructed) ** 2, axis=1)

        # Step 5: Create enriched DataFrame
        encoded_df = pd.DataFrame(
            encoded_features, index=target_indices,
            columns=[f"autoencoder_{i+1}" for i in range(encoded_features.shape[1])]
        )
        encoded_df["autoencoder_recon_error"] = recon_error
        
        result_df = features_df.merge(encoded_df, left_index=True, right_index=True, how='left')
        autoencoder_cols = [col for col in result_df.columns if 'autoencoder' in col]
        result_df[autoencoder_cols] = result_df[autoencoder_cols].fillna(0)
        
        self.logger.info(f"✅ Autoencoder feature generation completed. Added {len(autoencoder_cols)} new features.")
        return result_df

    def _run_optuna_optimization(self, X_train, y_train, X_val, y_val):
        """Helper to encapsulate the Optuna study logic."""
        def objective(trial):
            autoencoder = SequenceAwareAutoencoder(self.config)
            autoencoder.build_model(X_train.shape[1:], trial)
            history = autoencoder.fit(X_train, y_train, X_val, y_val, trial)
            return min(history.history['val_loss'])

        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
        study.optimize(
            objective,
            n_trials=self.config.get("training.n_trials", 50),
            n_jobs=self.config.get("training.n_jobs", 1) # Default to 1 for GPU
        )
        self.logger.info(f"Optimization completed. Best score: {study.best_value}")
        self.logger.info(f"Best parameters: {study.best_params}")
        return study.best_params
