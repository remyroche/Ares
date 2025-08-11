# src/analyst/autoencoder_feature_generator.py

import logging
import os
import signal
import time
from pathlib import Path
from typing import Any

import yaml

# Check for required dependencies
try:
    import numpy as np
    import optuna
    import pandas as pd
    import shap
    import tensorflow as tf
    from optuna.integration import TFKerasPruningCallback
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
    from tensorflow.keras import Model, layers, regularizers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEPENDENCY = str(e)
    print(missing(" Missing dependency: {MISSING_DEPENDENCY}"))
    print("📦 Please install required packages:")
    print("   pip install numpy pandas scikit-learn tensorflow optuna shap pyyaml")

# Set up comprehensive logging
from src.utils.logger import setup_logging
from src.utils.warning_symbols import (
    error,
)

setup_logging()
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent.parent
import sys

sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger


class AutoencoderConfig:
    """Configuration manager for autoencoder feature generator."""

    def __init__(self, config_path: str | None = None):
        if not DEPENDENCIES_AVAILABLE:
            msg = f"Required dependencies not available: {MISSING_DEPENDENCY}"
            raise ImportError(
                msg,
            )

        # Initialize logger first
        self.logger = system_logger.getChild("AutoencoderConfig")

        self.config_path = config_path or "src/analyst/autoencoder_config.yaml"
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path) as file:
                config = yaml.safe_load(file)
            self.logger.info(
                f"📋 Configuration loaded successfully from {self.config_path}"
            )
            self.logger.info(f"📊 Configuration sections: {list(config.keys())}")
            return config
        except Exception:
            self.logger.exception(
                "⚠️ Error loading config file, using default configuration"
            )
            return self._get_default_config()

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration if file loading fails."""
        # This default config is a fallback and should be customized in the YAML file.
        default_config = {
            "preprocessing": {
                "scaler_type": "robust",
                "outlier_threshold": 3.0,
                "missing_value_strategy": "forward_fill",
                "iqr_multiplier": 3.0,
                "use_price_returns": True,  # NEW: Use price differences instead of absolute prices
                "price_return_method": "pct_change",  # NEW: Method for calculating returns
                "primary_price_feature": "close",
                "primary_volume_feature": "volume",
                "enable_feature_selection": True,
            },
            "sequence": {"timesteps": 10, "overlap": 0.5},
            "autoencoder": {
                "epochs": 100,
                "early_stopping_patience": 10,
                "reduce_lr_patience": 5,
                "min_lr": 1e-6,
            },
            "training": {"n_trials": 50, "n_jobs": 1, "pruning_enabled": True},
            "feature_filtering": {
                "n_estimators": 100,
                "max_depth": 10,
                "importance_threshold": 0.99,
                "shap_imbalance_threshold": 100.0,  # Threshold for extreme imbalance in SHAP computation
                "enable_shap_imbalance_handling": True,  # Enable extreme imbalance handling for SHAP
            },
            "feature_analysis": {
                "enable_analysis": True,  # Enable feature importance analysis
                "high_correlation_threshold": 0.7,  # Threshold for high correlation features
                "low_correlation_threshold": 0.1,  # Threshold for low correlation features
                "stability_window": 100,  # Window size for stability analysis
                "stability_threshold": 0.7,  # Threshold for stable features
                "regime_analysis_enabled": True,  # Enable regime-specific analysis
                "comparison_with_original": True,  # Compare with original features
            },
            "output": {"output_dir": "models/autoencoder_features"},
        }
        return default_config

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split(".")
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
            with open(output_path, "w") as file:
                yaml.dump(self.config, file, default_flow_style=False)
            self.logger.info(f"📋 Configuration saved successfully to {output_path}")
        except Exception:
            self.logger.exception("⚠️ Error saving config file")


class PriceReturnConverter:
    """Convert price features to returns (price differences) for better autoencoder training."""

    def __init__(self, config: AutoencoderConfig):
        self.config = config
        self.logger = system_logger.getChild("PriceReturnConverter")
        self.use_price_returns = config.get("preprocessing.use_price_returns", True)
        self.price_return_method = config.get(
            "preprocessing.price_return_method", "pct_change"
        )
        # New configuration for feature selection
        self.primary_price_feature = config.get(
            "preprocessing.primary_price_feature", "close"
        )
        self.primary_volume_feature = config.get(
            "preprocessing.primary_volume_feature", "volume"
        )
        self.enable_feature_selection = config.get(
            "preprocessing.enable_feature_selection", True
        )
        # Columns that are not true engineered features and should be excluded
        self.non_feature_columns = {
            "timestamp", "time", "year", "month", "day", "day_of_week", "day_of_month", "quarter",
            "exchange", "symbol", "timeframe", "split"
        }

    def convert_price_features_to_returns(
        self, features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert price features to returns (price differences) to improve autoencoder training.
        Optimized to select only one representative price feature and one volume feature
        to avoid redundancy.
        """
        if not self.use_price_returns:
            self.logger.info(
                "📊 Price return conversion disabled, using original features"
            )
            return features_df

        self.logger.info(
            "🔄 Converting price features to returns for autoencoder training..."
        )

        # Create a copy to avoid modifying the original
        converted_df = features_df.copy()

        # Drop non-feature calendar/metadata columns up-front
        drop_cols = [c for c in converted_df.columns if c in self.non_feature_columns]
        if drop_cols:
            self.logger.info(f"🗑️ Dropping non-feature columns before conversion: {drop_cols}")
            converted_df = converted_df.drop(columns=drop_cols)

        if self.enable_feature_selection:
            # OPTIMIZED APPROACH: Select only one price feature and one volume feature
            self.logger.info("🎯 Using optimized feature selection to avoid redundancy")

            # Find available price and volume features
            available_price_features = []
            available_volume_features = []

            for col in converted_df.columns:
                col_lower = col.lower()

                # Skip regime and categorical and already-return-like engineered features
                if any(
                    exclude_pattern in col_lower
                    for exclude_pattern in [
                        "regime",
                        "categorical",
                        "class",
                        "label",
                        "category",
                        "type",
                    ]
                ):
                    continue

                # Skip already engineered return-like features
                if any(suffix in col_lower for suffix in ["_returns", "_diff", "_log_returns", "_ratio"]):
                    continue

                # Skip array-valued columns
                try:
                    sample_value = converted_df[col].iloc[0]
                    if isinstance(sample_value, (np.ndarray, list)):
                        self.logger.warning(f"Skipping array-valued column {col}: contains {type(sample_value)}")
                        continue
                except Exception:
                    pass

                # Skip low-cardinality categorical-like columns
                try:
                    unique_count = converted_df[col].nunique()
                    if unique_count <= 5:
                        continue
                except (TypeError, ValueError):
                    continue

                # Categorize features
                if col_lower in self._RAW_PRICE_COLS:
                    available_price_features.append(col)
                elif col_lower in self._RAW_VOLUME_COLS:
                    available_volume_features.append(col)

            # Log only true raw candidates (not engineered proxies)
            self.logger.info(
                f"📊 Found {len(available_price_features)} price features: {available_price_features}"
            )
            self.logger.info(
                f"📊 Found {len(available_volume_features)} volume features: {available_volume_features}"
            )

            # Select primary features
            selected_price_feature = None
            selected_volume_feature = None

            if self.primary_price_feature in available_price_features:
                selected_price_feature = self.primary_price_feature
            elif available_price_features:
                selected_price_feature = available_price_features[0]
                if selected_price_feature not in {"open","high","low","close","avg_price","min_price","max_price"}:
                    self.logger.info(
                        f"🎯 Selected engineered price proxy '{selected_price_feature}' (preferred '{self.primary_price_feature}' not available); will not convert to returns"
                    )
                else:
                    self.logger.info(
                        f"🎯 Selected '{selected_price_feature}' as primary price feature (preferred '{self.primary_price_feature}' not available)"
                    )

            if self.primary_volume_feature in available_volume_features:
                selected_volume_feature = self.primary_volume_feature
            elif available_volume_features:
                selected_volume_feature = available_volume_features[0]
                if selected_volume_feature not in {"volume","trade_volume"}:
                    self.logger.info(
                        f"🎯 Selected engineered volume proxy '{selected_volume_feature}' (preferred '{self.primary_volume_feature}' not available); will not convert to returns"
                    )
                else:
                    self.logger.info(
                        f"🎯 Selected '{selected_volume_feature}' as primary volume feature (preferred '{self.primary_volume_feature}' not available)"
                    )

            # Remove redundant raw price and volume columns, but keep engineered features
            features_to_remove = []
            for col in converted_df.columns:
                col_lower = col.lower()

                # Skip non-feature or engineered return-like
                if col in self.non_feature_columns:
                    features_to_remove.append(col)
                    continue
                if any(suffix in col_lower for suffix in ["_returns", "_diff", "_log_returns", "_ratio"]):
                    continue

                # Remove redundant price features (keep only selected one)
                if any(
                    price_pattern == col_lower
                    for price_pattern in [
                        "open",
                        "high",
                        "low",
                        "close",
                        "avg_price",
                        "min_price",
                        "max_price",
                    ]
                ):
                    if selected_price_feature and col != selected_price_feature:
                        features_to_remove.append(col)

                # Remove redundant volume features (keep only selected one)
                elif col_lower in self._RAW_VOLUME_COLS:
                    if selected_volume_feature and col != selected_volume_feature:
                        features_to_remove.append(col)

            if features_to_remove:
                # Only remove exact raw columns, do not remove engineered proxies
                raw_only = [c for c in features_to_remove if c in (self._RAW_PRICE_COLS | self._RAW_VOLUME_COLS)]
                if raw_only:
                    self.logger.info(
                        f"🗑️ Removing {len(raw_only)} redundant raw features: {raw_only}"
                    )
                    converted_df = converted_df.drop(columns=raw_only)

            # Convert selected features to returns
            features_to_convert = []
            # Only convert if the selected features are truly raw OHLCV, not engineered proxies
            if selected_price_feature in self._RAW_PRICE_COLS:
                features_to_convert.append(selected_price_feature)
            if selected_volume_feature in self._RAW_VOLUME_COLS:
                features_to_convert.append(selected_volume_feature)

            self.logger.info(
                f"📊 Converting {len(features_to_convert)} selected features to returns: {features_to_convert}"
            )

        else:
            # LEGACY APPROACH: Convert all price-related features (for backward compatibility)
            self.logger.info(
                "📊 Using legacy approach - converting all price-related features"
            )

            # Define price-related feature patterns to convert
            price_patterns = [
                # OHLCV patterns
                "open",
                "high",
                "low",
                "close",
                "volume",
                # Price-related technical indicators
                "price",
                "Price",
                "PRICE",
                "sma_",
                "ema_",
                "SMA_",
                "EMA_",
                "bb_",
                "BB_",
                "bollinger",
                "atr",
                "ATR",
                "average_true_range",
                "vwap",
                "VWAP",
                # Price ratios and levels
                "price_",
                "Price_",
                "PRICE_",
                "level_",
                "Level_",
                "LEVEL_",
                "support_",
                "resistance_",
                "Support_",
                "Resistance_",
                # Moving averages (price-based)
                "ma_",
                "MA_",
                "moving_average",
                # Price momentum and change
                "momentum",
                "Momentum",
                "MOMENTUM",
                "change",
                "Change",
                "CHANGE",
                # Volume-weighted price features
                "vwap",
                "VWAP",
                "volume_weighted",
                # Price-based oscillators
                "cci",
                "CCI",
                "commodity_channel",
                "williams_r",
                "Williams_R",
                "WILLIAMS_R",
                # Price-based patterns
                "pattern_",
                "Pattern_",
                "PATTERN_",
                "candlestick_",
                "Candlestick_",
                "CANDLESTICK_",
            ]

            # Find columns that match price patterns
            features_to_convert = []
            for col in converted_df.columns:
                # Skip regime and categorical features
                if any(
                    exclude_pattern in col.lower()
                    for exclude_pattern in [
                        "regime",
                        "categorical",
                        "class",
                        "label",
                        "category",
                        "type",
                    ]
                ):
                    continue

                # Skip features with very limited unique values
                unique_count = converted_df[col].nunique()
                if unique_count <= 5:
                    continue

                # Check if column matches price patterns
                if any(pattern in col.lower() for pattern in price_patterns):
                    # Skip columns that are already returns or differences
                    if any(
                        skip_pattern in col.lower()
                        for skip_pattern in ["return", "diff", "change", "pct", "ratio"]
                    ):
                        continue
                    features_to_convert.append(col)

        # Convert selected features to returns
        converted_count = 0
        for col in features_to_convert:
            try:
                if col in converted_df.columns:
                    # CRITICAL: Double-check for known problematic features
                    if col.lower() in [
                        "volume_regime",
                        "volatility_regime",
                        "trend_regime",
                    ]:
                        self.logger.warning(
                            f"⚠️ Skipping known regime feature '{col}' to prevent infinite values"
                        )
                        continue

                    original_values = converted_df[col].copy()

                    # Handle different return calculation methods
                    if self.price_return_method == "pct_change":
                        # Percentage change (most common). Guard log-derived columns to avoid extreme spikes.
                        if col.endswith("_log"):
                            # For log series x=log(price), use diff(x) which equals log returns
                            returns = original_values.diff().fillna(0)
                            new_col_name = f"{col}_diff"
                        else:
                            returns = original_values.pct_change().fillna(0)
                            new_col_name = f"{col}_returns"
                    elif self.price_return_method == "diff":
                        # Simple difference
                        returns = original_values.diff().fillna(0)
                        new_col_name = f"{col}_diff"
                    elif self.price_return_method == "log_returns":
                        # Log returns (for financial data)
                        returns = np.log(
                            original_values / original_values.shift(1)
                        ).fillna(0)
                        new_col_name = f"{col}_log_returns"
                    else:
                        # Default to percentage change
                        returns = original_values.pct_change().fillna(0)
                        new_col_name = f"{col}_returns"

                    # CRITICAL: Handle infinite values that can crash scikit-learn models
                    inf_count_before = np.isinf(returns).sum()
                    if inf_count_before > 0:
                        self.logger.warning(
                            f"⚠️ Found {inf_count_before} infinite values in '{col}' returns - replacing with NaN"
                        )

                    returns = returns.replace([np.inf, -np.inf], np.nan)
                    returns = returns.fillna(0)

                    # Additional safety: clip extreme values to prevent numerical issues
                    max_abs_value = 1000  # Reasonable limit for percentage changes
                    extreme_count_before = (np.abs(returns) > max_abs_value).sum()
                    if extreme_count_before > 0:
                        self.logger.warning(
                            f"⚠️ Found {extreme_count_before} extreme values (>±{max_abs_value}) in '{col}' returns - clipping"
                        )

                    returns = np.clip(returns, -max_abs_value, max_abs_value)

                    # Replace the original column with returns
                    converted_df[col] = returns
                    converted_count += 1

                    # Log conversion details for first few columns
                    if converted_count <= 5:
                        self.logger.info(
                            f"   📊 Converted '{col}' to returns (method: {self.price_return_method})"
                        )
                        self.logger.info(
                            f"      Original range: [{original_values.min():.6f}, {original_values.max():.6f}]"
                        )
                        self.logger.info(
                            f"      Returns range: [{returns.min():.6f}, {returns.max():.6f}]"
                        )

            except Exception as e:
                self.logger.warning(
                    f"⚠️ Failed to convert price feature '{col}' to returns: {e}"
                )
                continue

        self.logger.info(
            f"✅ Successfully converted {converted_count} price features to returns"
        )
        self.logger.info(f"📊 Final feature count: {converted_df.shape[1]} columns")

        # Final validation: ensure no infinite or extreme values remain
        final_inf_count = (
            np.isinf(converted_df.select_dtypes(include=[np.number])).sum().sum()
        )
        if final_inf_count > 0:
            self.logger.error(
                f"🚨 CRITICAL: {final_inf_count} infinite values still present after conversion!"
            )
            # Emergency cleanup
            converted_df = converted_df.replace([np.inf, -np.inf], 0)
        else:
            self.logger.info("✅ Final validation passed: no infinite values detected")

        return converted_df


class FeatureFilter:
    """Random Forest + SHAP feature filtering."""

    def __init__(self, config: AutoencoderConfig):
        if not DEPENDENCIES_AVAILABLE:
            msg = f"Required dependencies not available: {MISSING_DEPENDENCY}"
            raise ImportError(
                msg,
            )
        self.config = config
        self.logger = system_logger.getChild("FeatureFilter")
        # Treat calendar/metadata columns as non-features
        self.non_feature_columns = {
            "timestamp", "time", "year", "month", "day", "day_of_week", "day_of_month", "quarter",
            "exchange", "symbol", "timeframe", "split"
        }
        # Define all raw/non-feature columns to be excluded
        # Canonical raw/non-engineered column names
        self._RAW_PRICE_COLS = {"open","high","low","close","avg_price","min_price","max_price"}
        self._RAW_VOLUME_COLS = {"volume","trade_volume"}
        self._RAW_META_COLS = {"trade_count","funding_rate","volume_ratio"}
        self._PROJECT_EXCLUDE = {"volume_price_impact"}  # engineered proxy to exclude from AE filtering
        self.raw_columns = self._RAW_PRICE_COLS | self._RAW_VOLUME_COLS | self._RAW_META_COLS | self._PROJECT_EXCLUDE

    def _exclude_raw_and_meta(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in df.columns if c in self.non_feature_columns or c in self.raw_columns]
        if cols_to_drop:
            self.logger.warning(f"🚨 Excluding non-feature/raw columns: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
            self.logger.info(f"📊 Features shape after exclusion: {df.shape}")
        return df

    def filter_features(
        self,
        features_df: pd.DataFrame,
        labels: np.ndarray,
    ) -> pd.DataFrame:
        """Filter features using Random Forest + SHAP importance."""
        try:
            self.logger.info(
                "🔍 Starting feature filtering with Random Forest + SHAP..."
            )
            self.logger.info(f"📊 Input data shape: {features_df.shape}")
            self.logger.info(f"🎯 Number of unique labels: {len(np.unique(labels))}")
            self.logger.info(
                f"📈 Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}"
            )

            # CRITICAL: Filter out all raw/non-feature columns first
            features_df = self._exclude_raw_and_meta(features_df)
            if features_df.empty:
                self.logger.error("🚨 CRITICAL: No engineered features remaining after exclusion")
                return pd.DataFrame()

            X = features_df.select_dtypes(include=[np.number]).fillna(0)
            # Safety: ensure raw numeric columns are excluded
            raw_in_numeric = [c for c in X.columns if c in self.raw_columns]
            if raw_in_numeric:
                self.logger.warning(f"🚨 Removing raw numeric columns from candidate features: {raw_in_numeric}")
                X = X.drop(columns=raw_in_numeric)
                features_df = features_df.drop(columns=raw_in_numeric)
            y = labels

            # Check if we have any numeric features
            if X.empty or X.shape[1] == 0:
                self.logger.warning("⚠️ No numeric features available for filtering")
                self.logger.warning("⚠️ Returning original features without filtering")
                return features_df

            self.logger.info(f"🔢 Numeric features selected: {len(X.columns)}")
            self.logger.info(f"📏 Feature names: {list(X.columns)}")

            if len(np.unique(y)) < 2:
                self.logger.warning(
                    "⚠️ Insufficient unique labels for classification, skipping filtering.",
                )
                return features_df

            # Random Forest training
            self.logger.info(
                "🌲 Training Random Forest model for feature importance..."
            )
            n_estimators = self.config.get("feature_filtering.n_estimators", 100)
            max_depth = self.config.get("feature_filtering.max_depth", 10)
            random_state = self.config.get("feature_filtering.random_state", 42)

            self.logger.info(
                f"🌲 RF Parameters: n_estimators={n_estimators}, max_depth={max_depth}, random_state={random_state}"
            )

            rf_model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
            )

            import time

            start_time = time.time()
            rf_model.fit(X, y)
            training_time = time.time() - start_time

            self.logger.info(
                f"✅ Random Forest training completed in {training_time:.2f} seconds"
            )
            self.logger.info(f"🎯 RF training score: {rf_model.score(X, y):.4f}")

            # ENHANCED SHAP analysis with multiple efficiency optimizations
            self.logger.info("🔍 Computing SHAP values with enhanced efficiency...")
            start_time = time.time()

            # Try new import path first, then fallback to old path
            try:
                from shap.explainers import TreeExplainer

                self.logger.info("📦 Using SHAP TreeExplainer from shap.explainers")
            except ImportError:
                from shap import TreeExplainer

                self.logger.info("📦 Using SHAP TreeExplainer from shap")

            # EFFICIENCY OPTIMIZATION 1: Percentage-based adaptive sampling
            # Get sampling configuration from config
            sample_percentage = self.config.get(
                "feature_filtering.sample_percentage", 10.0
            )  # Default 10%
            min_sample_size = self.config.get(
                "feature_filtering.min_sample_size", 1000
            )  # Minimum 1000 rows
            max_sample_size = self.config.get(
                "feature_filtering.max_sample_size", 1000000
            )  # Maximum 10000 rows

            # Calculate sample size based on percentage
            sample_size = int(len(X) * sample_percentage / 100)

            # Apply min/max constraints
            sample_size = max(min_sample_size, min(sample_size, max_sample_size))

            self.logger.info(f"📊 Dataset size: {len(X)} rows")
            self.logger.info(f"📊 Sample percentage: {sample_percentage}%")
            self.logger.info(f"📊 Calculated sample size: {sample_size} rows")

            # EFFICIENCY OPTIMIZATION 2: Enhanced stratified sampling with fallback
            if sample_size < len(X):
                self.logger.info(
                    "🔄 Applying stratified sampling to maintain class balance..."
                )

                # Check if we have enough samples per class for stratification
                unique_labels, label_counts = np.unique(y, return_counts=True)
                min_class_count = label_counts.min()
                max_class_count = label_counts.max()
                imbalance_ratio = (
                    max_class_count / min_class_count
                    if min_class_count > 0
                    else float("inf")
                )

                self.logger.info(f"📊 Label distribution analysis:")
                self.logger.info(f"   Unique labels: {unique_labels}")
                self.logger.info(f"   Label counts: {label_counts}")
                self.logger.info(f"   Min class count: {min_class_count}")
                self.logger.info(f"   Imbalance ratio: {imbalance_ratio:.1f}")

                # CRITICAL FIX: Handle extreme imbalance for SHAP computation
                shap_imbalance_threshold = self.config.get(
                    "feature_filtering.shap_imbalance_threshold", 100.0
                )
                enable_shap_imbalance_handling = self.config.get(
                    "feature_filtering.enable_shap_imbalance_handling", True
                )

                if (
                    enable_shap_imbalance_handling
                    and imbalance_ratio > shap_imbalance_threshold
                ):
                    self.logger.warning(
                        f"🚨 CRITICAL FIX: Extreme label imbalance detected (ratio={imbalance_ratio:.1f} > {shap_imbalance_threshold})"
                    )
                    self.logger.info("🔄 Using random sampling for SHAP computation...")
                    sample_indices = np.random.choice(
                        len(X), sample_size, replace=False
                    )
                    X_sample = X.iloc[sample_indices]
                    y_sample = y[sample_indices]
                    self.logger.info(f"📊 Random sample size: {len(X_sample)} rows")

                elif (
                    min_class_count >= 10
                ):  # Need at least 10 samples per class for stratification
                    try:
                        from sklearn.model_selection import train_test_split

                        # Calculate stratified sample size per class
                        class_sample_sizes = {}
                        for label, count in zip(unique_labels, label_counts):
                            class_sample_size = int(count * sample_percentage / 100)
                            class_sample_size = max(
                                5, min(class_sample_size, count)
                            )  # At least 5, at most all
                            class_sample_sizes[label] = class_sample_size

                        # Perform stratified sampling
                        X_sample, _, y_sample, _ = train_test_split(
                            X, y, train_size=sample_size, stratify=y, random_state=42
                        )

                        # Verify stratification worked
                        original_dist = dict(zip(unique_labels, label_counts))
                        sample_dist = dict(
                            zip(*np.unique(y_sample, return_counts=True))
                        )

                        self.logger.info(f"✅ Stratified sampling successful!")
                        self.logger.info(
                            f"📊 Original class distribution: {original_dist}"
                        )
                        self.logger.info(f"📊 Sample class distribution: {sample_dist}")
                        self.logger.info(
                            f"📊 Sample size: {len(X_sample)} rows ({len(X_sample)/len(X)*100:.1f}%)"
                        )

                    except Exception as e:
                        self.logger.warning(f"⚠️ Stratified sampling failed: {e}")
                        self.logger.info("🔄 Falling back to random sampling...")
                        sample_indices = np.random.choice(
                            len(X), sample_size, replace=False
                        )
                        X_sample = X.iloc[sample_indices]
                        y_sample = y[sample_indices]
                        self.logger.info(f"📊 Random sample size: {len(X_sample)} rows")
                else:
                    self.logger.warning(
                        f"⚠️ Insufficient samples per class for stratification (min: {min_class_count})"
                    )
                    self.logger.info("🔄 Using random sampling...")
                    sample_indices = np.random.choice(
                        len(X), sample_size, replace=False
                    )
                    X_sample = X.iloc[sample_indices]
                    y_sample = y[sample_indices]
                    self.logger.info(f"📊 Random sample size: {len(X_sample)} rows")
            else:
                X_sample = X
                y_sample = y
                self.logger.info("📊 Using full dataset (no sampling needed)")

            # EFFICIENCY OPTIMIZATION 3: Feature reduction before SHAP
            enable_prefiltering = self.config.get(
                "feature_filtering.enable_feature_prefiltering", True
            )
            max_features_for_shap = self.config.get(
                "feature_filtering.max_features_for_shap", 50
            )

            if enable_prefiltering and len(X_sample.columns) > max_features_for_shap:
                self.logger.info(
                    f"📊 High feature count ({len(X_sample.columns)}), applying pre-filtering"
                )
                self.logger.info(f"📊 Target feature count: {max_features_for_shap}")

                # Use Random Forest feature importance for initial filtering
                pre_filter_rf = RandomForestClassifier(
                    n_estimators=50,  # Fewer trees for speed
                    max_depth=8,  # Shallow trees for speed
                    random_state=42,
                    n_jobs=-1,
                )
                pre_filter_rf.fit(X_sample, y_sample)

                # Keep top features based on importance
                feature_importance = pre_filter_rf.feature_importances_
                top_feature_indices = np.argsort(feature_importance)[
                    -max_features_for_shap:
                ]  # Keep top N features
                X_sample = X_sample.iloc[:, top_feature_indices]

                self.logger.info(
                    f"📊 Pre-filtered to top {len(X_sample.columns)} features"
                )
                self.logger.info(f"📊 Selected features: {list(X_sample.columns)}")
            else:
                self.logger.info(
                    f"📊 No pre-filtering needed (features: {len(X_sample.columns)}, max: {max_features_for_shap})"
                )

            # EFFICIENCY OPTIMIZATION 4: Optimized Random Forest for SHAP
            # Use configurable parameters for SHAP model optimization
            shap_n_estimators = self.config.get(
                "feature_filtering.shap_n_estimators", 50
            )
            shap_max_depth = self.config.get("feature_filtering.shap_max_depth", 8)
            shap_min_samples_split = self.config.get(
                "feature_filtering.shap_min_samples_split", 10
            )
            shap_min_samples_leaf = self.config.get(
                "feature_filtering.shap_min_samples_leaf", 5
            )

            shap_rf_model = RandomForestClassifier(
                n_estimators=shap_n_estimators,
                max_depth=shap_max_depth,
                min_samples_split=shap_min_samples_split,
                min_samples_leaf=shap_min_samples_leaf,
                random_state=42,
                n_jobs=-1,
            )

            self.logger.info(
                f"🌲 SHAP RF Parameters: n_estimators={shap_n_estimators}, max_depth={shap_max_depth}"
            )

            self.logger.info("🌲 Training optimized Random Forest for SHAP...")
            shap_rf_model.fit(X_sample, y_sample)
            self.logger.info(
                f"✅ Optimized RF training completed (score: {shap_rf_model.score(X_sample, y_sample):.4f})"
            )

            # EFFICIENCY OPTIMIZATION 5: Create explainer with optimized settings
            explainer = TreeExplainer(
                shap_rf_model,
                feature_names=X_sample.columns.tolist(),
                model_output="raw",  # Use raw output for compatibility
            )
            self.logger.info("🔧 Optimized SHAP explainer created successfully")

            # Add timeout protection for SHAP computation
            import signal
            import threading
            import platform

            # EFFICIENCY OPTIMIZATION 6: Flexible timeout based on dataset size
            # Calculate timeout based on sample size: 1 minute per 5000 samples
            base_timeout_per_5000 = self.config.get(
                "feature_filtering.timeout_per_5000_samples", 60
            )  # 1 minute per 5000
            calculated_timeout = int(len(X_sample) / 5000 * base_timeout_per_5000)

            # More flexible bounds: Min 30s, Max 15 minutes (increased from 5min)
            timeout_seconds = max(30, min(900, calculated_timeout))

            self.logger.info(f"⏱️ Flexible timeout calculation:")
            self.logger.info(f"   📊 Sample size: {len(X_sample)} rows")
            self.logger.info(
                f"   📊 Base rate: {base_timeout_per_5000}s per 5000 samples"
            )
            self.logger.info(f"   📊 Calculated timeout: {calculated_timeout}s")
            self.logger.info(
                f"   ⏱️ Final timeout: {timeout_seconds}s (bounded: 30s-900s)"
            )

            if platform.system() != "Windows":
                # Unix-like systems can use signal.SIGALRM
                def timeout_handler(signum, frame):
                    raise TimeoutError("SHAP computation timed out")

                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)
            else:
                # Windows doesn't support SIGALRM, we'll use a simpler approach
                self.logger.info(
                    "⚠️ Windows detected - using simplified timeout protection"
                )

            try:
                # EFFICIENCY OPTIMIZATION 7: Use background values for faster computation
                # Calculate background values from a small subset
                background_size = min(100, len(X_sample) // 10)
                background_indices = np.random.choice(
                    len(X_sample), background_size, replace=False
                )
                background_values = X_sample.iloc[background_indices]

                self.logger.info(
                    f"📊 Computing SHAP values with background set of {len(background_values)} samples..."
                )

                # Compute SHAP values (simplified for compatibility)
                shap_values = explainer.shap_values(X_sample)

                self.logger.info(
                    "✅ SHAP values computed successfully with optimizations"
                )

                # Cancel the alarm on Unix systems
                if platform.system() != "Windows":
                    signal.alarm(0)
                shap_time = time.time() - start_time
                self.logger.info(f"✅ SHAP values computed in {shap_time:.2f} seconds")

            except TimeoutError:
                # Cancel the alarm on Unix systems
                if platform.system() != "Windows":
                    signal.alarm(0)
                self.logger.warning(
                    "⏰ SHAP computation timed out, falling back to Random Forest feature importance"
                )
                # Fallback to Random Forest feature importance
                feature_importance = rf_model.feature_importances_
                shap_time = time.time() - start_time
                self.logger.info(
                    f"✅ Fallback feature importance computed in {shap_time:.2f} seconds"
                )

                # Skip SHAP-specific processing and go directly to feature selection
                sorted_indices = np.argsort(feature_importance)[::-1]
                sorted_importance = feature_importance[sorted_indices]
                cumulative_importance = np.cumsum(sorted_importance)
                total_importance = cumulative_importance[-1]

                self.logger.info(f"📊 Total importance: {total_importance:.6f}")
                self.logger.info(f"🏆 Top 5 most important features:")
                for i in range(min(5, len(sorted_indices))):
                    feature_name = X.columns[sorted_indices[i]]
                    importance = sorted_importance[i]
                    cumulative = cumulative_importance[i]
                    self.logger.info(
                        f"   {i+1}. {feature_name}: {importance:.6f} (cumulative: {cumulative:.6f})"
                    )

                threshold = self.config.get(
                    "feature_filtering.importance_threshold", 0.99
                )
                importance_cutoff = threshold * total_importance

                self.logger.info(
                    f"🎯 Importance threshold: {threshold} ({threshold*100}%)"
                )
                self.logger.info(f"📊 Importance cutoff: {importance_cutoff:.6f}")

                # Find the first index where cumulative importance exceeds the threshold
                cutoff_index = (
                    np.where(cumulative_importance >= importance_cutoff)[0][0] + 1
                )
                selected_indices = sorted_indices[:cutoff_index]

                self.logger.info(
                    f"📊 Features needed to reach threshold: {cutoff_index}"
                )
                self.logger.info(
                    f"📊 Cumulative importance at cutoff: {cumulative_importance[cutoff_index-1]:.6f}"
                )

                # Enforce minimum number of features
                min_features = self.config.get(
                    "feature_filtering.min_features_for_ae", 15
                )
                self.logger.info(f"🔒 Minimum features required: {min_features}")

                if len(selected_indices) < min_features:
                    self.logger.info(
                        f"⚠️ Selected features ({len(selected_indices)}) below minimum ({min_features}), expanding selection"
                    )
                    selected_indices = sorted_indices[
                        : max(min_features, len(sorted_indices))
                    ]
                    self.logger.info(
                        f"📊 Expanded selection to {len(selected_indices)} features"
                    )

                selected_features = X.columns.to_numpy()[selected_indices].tolist()

                self.logger.info(
                    f"✅ Selected {len(selected_features)} features out of {len(X.columns)} using fallback method.",
                )
                self.logger.info(f"📊 Selected features: {selected_features}")

                # Ensure we keep at least min_features after any refinement
                shap_refine_min = self.config.get(
                    "feature_filtering.min_features_for_shap", 20
                )
                self.logger.info(f"🔒 SHAP refinement minimum: {shap_refine_min}")

                final_features = selected_features
                if len(selected_features) > shap_refine_min:
                    k = max(min_features, len(selected_features))
                    final_features = selected_features[:k]
                    self.logger.info(
                        f"📊 Refined selection to {len(final_features)} features"
                    )

                self.logger.info(
                    f"🎉 Feature filtering completed successfully with fallback method!"
                )
                self.logger.info(f"📊 Final feature count: {len(final_features)}")
                self.logger.info(f"📊 Final features: {final_features}")

                return features_df[final_features].copy()

            # Compute mean absolute SHAP importance per feature
            self.logger.info("📊 Computing feature importance from SHAP values...")
            start_time = time.time()

            # Handle multiple SHAP return formats across versions
            if hasattr(shap_values, "values"):
                self.logger.info("📦 SHAP values format: shap_values.values")
                shap_arr = np.asarray(shap_values.values)
            elif isinstance(shap_values, list):
                self.logger.info(
                    f"📦 SHAP values format: list of {len(shap_values)} arrays"
                )
                # List of arrays per class -> (n_classes, n_samples, n_features)
                shap_arr = np.stack([np.asarray(sv) for sv in shap_values], axis=0)
            else:
                self.logger.info("📦 SHAP values format: numpy array")
                shap_arr = np.asarray(shap_values)

            self.logger.info(f"📐 SHAP array shape: {shap_arr.shape}")

            # Ensure we always end up with shape (..., n_samples, n_features)
            if shap_arr.ndim == 2:
                # (n_samples, n_features) -> add class axis
                self.logger.info("🔄 Adding class dimension to SHAP array")
                shap_arr = shap_arr[None, ...]
            elif shap_arr.ndim == 1:
                # Degenerate case, treat as single feature vector over samples
                self.logger.info("🔄 Reshaping SHAP array for single feature")
                shap_arr = shap_arr[None, :, None]

            # Now take mean abs over classes and samples → per-feature importance
            self.logger.info(
                "📊 Computing mean absolute SHAP importance per feature..."
            )
            feature_importance = np.nanmean(np.abs(shap_arr), axis=(0, 1))

            # Guard against any NaNs/inf
            feature_importance = np.nan_to_num(
                feature_importance, nan=0.0, posinf=0.0, neginf=0.0
            )

            importance_time = time.time() - start_time
            self.logger.info(
                f"✅ Feature importance computed in {importance_time:.2f} seconds"
            )

            # Sort features by importance
            self.logger.info("📈 Sorting features by importance...")
            sorted_indices = np.argsort(feature_importance)[::-1]
            sorted_importance = feature_importance[sorted_indices]
            cumulative_importance = np.cumsum(sorted_importance)
            total_importance = cumulative_importance[-1]

            self.logger.info(f"📊 Total importance: {total_importance:.6f}")
            self.logger.info(f"🏆 Top features by importance:")
            for i in range(min(5, len(sorted_indices))):
                feature_name = X.columns[sorted_indices[i]]
                importance = sorted_importance[i]
                cumulative = cumulative_importance[i]
                self.logger.info(
                    f"   {i+1}. {feature_name}: {importance:.6f} (cumulative: {cumulative:.6f})"
                )

            # Suspicious dominance detection (relative, warn-only)
            try:
                total_imp = float(total_importance) if total_importance else 0.0
                if total_imp > 0 and len(sorted_indices) > 0:
                    top1 = float(sorted_importance[0])
                    top2 = float(sorted_importance[0] + (sorted_importance[1] if len(sorted_importance) > 1 else 0.0))
                    top1_ratio = top1 / total_imp
                    top2_ratio = top2 / total_imp
                    thresh1 = float(self.config.get("feature_filtering.suspicious_top1_ratio", 0.60))
                    thresh2 = float(self.config.get("feature_filtering.suspicious_top2_ratio", 0.85))
                    if top1_ratio >= thresh1 or top2_ratio >= thresh2:
                        suspicious_names = [X.columns[sorted_indices[0]]]
                        if len(sorted_indices) > 1:
                            suspicious_names.append(X.columns[sorted_indices[1]])
                        patterns = ["price_impact", "volume_price_impact", "market_depth", "order_flow_imbalance", "liquidity_score", "bid_ask_spread"]
                        matched = [n for n in suspicious_names if any(p in n.lower() for p in patterns)]
                        if matched:
                            self.logger.warning(
                                f"⚠️ Suspicious dominance: top features {matched} account for top1={top1_ratio:.1%}, top2={top2_ratio:.1%} of total importance"
                            )
                        else:
                            self.logger.warning(
                                f"⚠️ Suspicious dominance: top features {suspicious_names} account for top1={top1_ratio:.1%}, top2={top2_ratio:.1%} of total importance"
                            )
                    # Also warn on very low effective number of features (importance concentration)
                    denom = float(np.sum(np.square(sorted_importance))) if np.isfinite(np.sum(np.square(sorted_importance))) else 0.0
                    if total_imp > 0 and denom > 0:
                        eff = (total_imp ** 2) / denom
                        if eff < float(self.config.get("feature_filtering.suspicious_effective_features", 5.0)):
                            self.logger.warning(f"⚠️ Importance highly concentrated: effective_features≈{eff:.1f}")
            except Exception:
                pass

            # Get feature selection parameters from config
            threshold = self.config.get(
                "feature_filtering.importance_threshold", 0.90
            )
            min_features = self.config.get(
                "feature_filtering.min_features_to_keep", 15
            )
            max_features = self.config.get(
                "feature_filtering.max_features_to_keep", 80
            )
            min_importance_per_feature = self.config.get(
                "feature_filtering.min_importance_per_feature", 0.001
            )

            importance_cutoff = threshold * total_importance

            self.logger.info(f"🎯 Feature selection parameters:")
            self.logger.info(
                f"   📊 Importance threshold: {threshold:.3f} ({threshold*100:.1f}%)"
            )
            self.logger.info(f"   📊 Min features to keep: {min_features}")
            self.logger.info(f"   📊 Max features to keep: {max_features}")
            self.logger.info(
                f"   📊 Min importance per feature: {min_importance_per_feature:.6f}"
            )
            self.logger.info(f"   📊 Importance cutoff: {importance_cutoff:.6f}")

            # ENHANCED LOGIC: Use multiple criteria for feature selection
            # 1. Features that meet the cumulative importance threshold
            threshold_cutoff = np.where(cumulative_importance >= importance_cutoff)[0]
            threshold_cutoff = (
                threshold_cutoff[0] + 1
                if len(threshold_cutoff) > 0
                else len(sorted_indices)
            )

            # 2. Features that meet minimum individual importance
            min_importance_cutoff = np.where(
                sorted_importance >= min_importance_per_feature
            )[0]
            min_importance_cutoff = (
                len(min_importance_cutoff) if len(min_importance_cutoff) > 0 else 0
            )

            # 3. Use the larger of the two cutoffs to be less aggressive
            cutoff_index = max(threshold_cutoff, min_importance_cutoff, min_features)
            selected_indices = sorted_indices[:cutoff_index]

            self.logger.info(f"📊 Enhanced selection analysis:")
            self.logger.info(f"   📊 Threshold cutoff: {threshold_cutoff} features")
            self.logger.info(
                f"   📊 Min importance cutoff: {min_importance_cutoff} features"
            )
            self.logger.info(f"   📊 Min features requirement: {min_features} features")
            self.logger.info(f"   📊 Final cutoff: {cutoff_index} features")
            self.logger.info(f"📊 Initial selection results:")
            self.logger.info(f"   📊 Features selected: {cutoff_index}")
            # Ensure we don't exceed array bounds
            actual_cutoff = min(cutoff_index, len(cumulative_importance))
            self.logger.info(
                f"   📊 Cumulative importance at cutoff: {cumulative_importance[actual_cutoff-1]:.6f}"
            )
            self.logger.info(
                f"   📊 Actual importance captured: {cumulative_importance[actual_cutoff-1]/total_importance*100:.1f}%"
            )

            # Apply minimum feature constraint
            if len(selected_indices) < min_features:
                self.logger.warning(
                    f"⚠️ Selected features ({len(selected_indices)}) below minimum ({min_features})"
                )
                self.logger.info(
                    f"🔄 Expanding selection to meet minimum requirement..."
                )
                # Ensure we don't exceed the available features
                actual_min_features = min(min_features, len(sorted_indices))
                selected_indices = sorted_indices[:actual_min_features]
                actual_importance = (
                    cumulative_importance[actual_min_features - 1]
                    if len(sorted_importance) >= actual_min_features
                    else cumulative_importance[-1]
                )
                self.logger.info(
                    f"📊 Expanded to {len(selected_indices)} features (importance: {actual_importance/total_importance*100:.1f}%)"
                )

                # If we still don't have enough features, we need to map back to original features
                if len(selected_indices) < min_features and hasattr(
                    self, "_prefiltered_features"
                ):
                    self.logger.warning(
                        f"⚠️ Still below minimum after expansion. This may indicate insufficient important features in the dataset."
                    )

            # Apply maximum feature constraint
            if len(selected_indices) > max_features:
                self.logger.warning(
                    f"⚠️ Selected features ({len(selected_indices)}) above maximum ({max_features})"
                )
                self.logger.info(
                    f"🔄 Truncating selection to meet maximum requirement..."
                )
                selected_indices = sorted_indices[:max_features]
                actual_importance = (
                    cumulative_importance[max_features - 1]
                    if len(sorted_importance) >= max_features
                    else cumulative_importance[-1]
                )
                self.logger.info(
                    f"📊 Truncated to {len(selected_indices)} features (importance: {actual_importance/total_importance*100:.1f}%)"
                )

            selected_features = X.columns.to_numpy()[selected_indices].tolist()

            self.logger.info(f"✅ Final feature selection:")
            self.logger.info(
                f"   📊 Features selected: {len(selected_features)} out of {len(X.columns)}"
            )
            self.logger.info(
                f"   📊 Importance captured: {cumulative_importance[len(selected_indices)-1]/total_importance*100:.1f}%"
            )
            self.logger.info(f"   📊 Selected features: {selected_features}")

            self.logger.info(f"🎉 Feature filtering completed successfully!")
            self.logger.info(f"📊 Final feature count: {len(selected_features)}")
            self.logger.info(f"📊 Final features: {selected_features}")

            return features_df[selected_features].copy()

        except Exception:
            self.logger.exception("Error in feature filtering")
            return features_df


class ImprovedAutoencoderPreprocessor:
    """Enhanced preprocessor with separate fit/transform and no data leakage."""

    def __init__(self, config: AutoencoderConfig):
        if not DEPENDENCIES_AVAILABLE:
            msg = f"Required dependencies not available: {MISSING_DEPENDENCY}"
            raise ImportError(
                msg,
            )
        self.config = config
        scaler_type = config.get("preprocessing.scaler_type", "robust")

        if scaler_type == "robust":
            self.scaler = RobustScaler()
        elif scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()

        self.outlier_lower_bounds_ = None
        self.outlier_upper_bounds_ = None
        self.is_fitted = False
        self.logger = system_logger.getChild("AutoencoderPreprocessor")

    def fit(self, X: pd.DataFrame) -> "ImprovedAutoencoderPreprocessor":
        """Fit the preprocessor on training data only."""
        self.logger.info(f"🔧 Fitting preprocessor on data with shape {X.shape}")

        # Handle missing values
        self.logger.info("📊 Handling missing values...")
        X_clean = self._handle_missing_values(X)
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            self.logger.info(
                f"📊 Missing values handled: {missing_count} values filled"
            )

        # Calculate outlier bounds using IQR method
        X_numeric = X_clean.select_dtypes(include=[np.number])
        self.logger.info(
            f"📊 Numeric features for outlier detection: {X_numeric.shape[1]} columns"
        )

        Q1 = X_numeric.quantile(0.25)
        Q3 = X_numeric.quantile(0.75)
        IQR = Q3 - Q1
        iqr_mult = self.config.get("preprocessing.iqr_multiplier", 3.0)

        self.outlier_lower_bounds_ = Q1 - iqr_mult * IQR
        self.outlier_upper_bounds_ = Q3 + iqr_mult * IQR

        self.logger.info(f"📊 Outlier detection: IQR method with multiplier {iqr_mult}")
        self.logger.info(
            f"📊 Outlier bounds calculated for {len(self.outlier_lower_bounds_)} features"
        )

        # Align bounds to columns to avoid pandas alignment errors
        lower_bounds = self.outlier_lower_bounds_.reindex(X_numeric.columns)
        upper_bounds = self.outlier_upper_bounds_.reindex(X_numeric.columns)
        X_clipped = X_numeric.clip(lower=lower_bounds, upper=upper_bounds, axis=1)

        # Count outliers clipped
        outliers_clipped = (
            ((X_numeric < lower_bounds) | (X_numeric > upper_bounds)).sum().sum()
        )
        if outliers_clipped > 0:
            self.logger.info(f"📊 Outliers clipped: {outliers_clipped} values")

        # Fit scaler
        scaler_type = self.config.get("preprocessing.scaler_type", "robust")
        self.logger.info(f"📊 Fitting {scaler_type} scaler...")
        self.scaler.fit(X_clipped.values)

        self.is_fitted = True
        self.logger.info("✅ Preprocessor fitted successfully")
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessor."""
        if not self.is_fitted:
            msg = "Preprocessor must be fitted before transform can be called."
            raise ValueError(
                msg,
            )

        self.logger.info(f"🔧 Transforming data with shape {X.shape}")

        # Handle missing values
        X_clean = self._handle_missing_values(X)

        # Clip outliers
        X_clipped = self._clip_outliers(X_clean)

        # Scale data
        self.logger.info("📊 Scaling data...")
        X_scaled = self.scaler.transform(X_clipped.values)

        # Final clipping for extreme values
        final_threshold = self.config.get("preprocessing.outlier_threshold", 3.0)
        X_final = np.clip(X_scaled, -final_threshold, final_threshold)

        # Count extreme values clipped
        extreme_values_clipped = (
            (X_scaled < -final_threshold) | (X_scaled > final_threshold)
        ).sum()
        if extreme_values_clipped > 0:
            self.logger.info(
                f"📊 Extreme values clipped: {extreme_values_clipped} values (threshold: ±{final_threshold})"
            )

        try:
            self.logger.info(f"✅ Transform completed successfully")
            self.logger.info(f"📊 Input shape: {X.shape}")
            self.logger.info(f"📊 Output shape: {X_final.shape}")
            self.logger.info(f"📊 Final clipping threshold: ±{final_threshold}")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not log transform details: {str(e)}")

        return X_final

    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on strategy."""
        strategy = self.config.get(
            "preprocessing.missing_value_strategy",
            "forward_fill",
        )
        if strategy == "forward_fill":
            return X.fillna(method="ffill").fillna(method="bfill").fillna(0)
        # Default to zero fill
        return X.fillna(0)

    def _clip_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers using pre-calculated bounds to prevent data leakage."""
        X_numeric = X.select_dtypes(include=[np.number])
        lower_bounds = self.outlier_lower_bounds_.reindex(X_numeric.columns)
        upper_bounds = self.outlier_upper_bounds_.reindex(X_numeric.columns)
        return X_numeric.clip(lower=lower_bounds, upper=upper_bounds, axis=1)


def create_sequences_with_index(
    X: np.ndarray,
    timesteps: int,
    original_index: pd.Index,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Convert 2D array to 3D sequences, tracking the index of the target."""
    sequences, targets, target_indices = [], [], []

    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Calculate sequence information
    total_samples = len(X)
    num_sequences = total_samples - timesteps + 1
    overlap_percentage = ((timesteps - 1) / timesteps) * 100

    # Log sequence creation details
    logger = logging.getLogger(__name__)
    logger.info(f"📊 Creating sequences from {total_samples} samples")
    logger.info(
        f"📊 Sequence configuration: {timesteps} timesteps, {num_sequences} sequences"
    )
    logger.info(f"📊 Overlap: {overlap_percentage:.1f}% between consecutive sequences")
    logger.info(
        f"📊 Input shape: {X.shape} -> Output shapes: ({num_sequences}, {timesteps}, {X.shape[1]})"
    )

    for i in range(num_sequences):
        sequence = X[i : i + timesteps]
        target = X[i + timesteps - 1]  # Target is the last timestep
        sequences.append(sequence)
        targets.append(target)
        target_indices.append(original_index[i + timesteps - 1])

    sequences_array = np.array(sequences)
    targets_array = np.array(targets)
    target_indices_array = pd.Index(target_indices)

    logger.info(f"✅ Sequence creation completed")
    logger.info(f"📊 Sequences shape: {sequences_array.shape}")
    logger.info(f"📊 Targets shape: {targets_array.shape}")
    logger.info(f"📊 Target indices: {len(target_indices_array)} samples")

    return sequences_array, targets_array, target_indices_array


class SequenceAwareAutoencoder:
    """1D-CNN based autoencoder that learns to reconstruct the last timestep of a sequence."""

    def __init__(self, config: AutoencoderConfig):
        if not DEPENDENCIES_AVAILABLE:
            msg = f"Required dependencies not available: {MISSING_DEPENDENCY}"
            raise ImportError(
                msg,
            )
        self.config = config
        self.logger = system_logger.getChild("SequenceAwareAutoencoder")
        self.autoencoder = None
        self.encoder = None

    def build_model(
        self,
        input_shape: tuple[int, int],
        trial: optuna.Trial | None = None,
    ) -> Model:
        """Build 1D-CNN autoencoder model."""
        timesteps, features = input_shape

        if trial:
            filters = trial.suggest_categorical("filters", [16, 32, 64])
            kernel_size = trial.suggest_int("kernel_size", 3, 7)
            dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            encoding_dim = trial.suggest_int("encoding_dim", 8, 64)
        else:  # Fallback to config for final training
            encoding_dim = self.config.get("autoencoder.encoding_dim", 32)
            # Use a dictionary for best_params from Optuna
            best_params = self.config.get("best_params", {})
            filters = best_params.get("filters", 32)
            kernel_size = best_params.get("kernel_size", 5)
            dropout_rate = best_params.get("dropout_rate", 0.3)
            learning_rate = best_params.get("learning_rate", 0.001)

        # Log the model configuration
        self.logger.info(f"🔧 Building autoencoder model architecture...")
        self.logger.info(
            f"📊 Input shape: (timesteps={timesteps}, features={features})"
        )
        self.logger.info(f"📊 Model hyperparameters:")
        self.logger.info(f"   📊 Filters: {filters}")
        self.logger.info(f"   📊 Kernel size: {kernel_size}")
        self.logger.info(f"   📊 Dropout rate: {dropout_rate}")
        self.logger.info(f"   📊 Encoding dimension: {encoding_dim}")
        self.logger.info(f"   📊 Learning rate: {learning_rate}")

        input_layer = layers.Input(shape=(timesteps, features))

        # Encoder
        x = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
        )(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Conv1D(
            filters=filters // 2,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)

        bottleneck = layers.Dense(encoding_dim, activation="tanh", name="bottleneck")(x)

        # Decoder - Reconstructs the feature vector of the last timestep
        output_layer = layers.Dense(features, activation="linear")(bottleneck)

        self.autoencoder = Model(inputs=input_layer, outputs=output_layer)
        self.encoder = Model(inputs=input_layer, outputs=bottleneck)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.autoencoder.compile(optimizer=optimizer, loss="huber", metrics=["mae"])

        try:
            total_params = int(
                np.sum([np.prod(v.shape) for v in self.autoencoder.trainable_weights])
            )
            self.logger.info(f"✅ Model compiled successfully!")
            self.logger.info(f"📊 Optimizer: Adam(learning_rate={learning_rate})")
            self.logger.info(f"📊 Loss function: Huber")
            self.logger.info(f"📊 Metrics: MAE")
            self.logger.info(f"📊 Total trainable parameters: {total_params:,}")

            # Model complexity assessment
            if total_params < 10000:
                complexity = "Lightweight"
            elif total_params < 100000:
                complexity = "Moderate"
            elif total_params < 1000000:
                complexity = "Complex"
            else:
                complexity = "Very Complex"

            self.logger.info(f"📊 Model complexity: {complexity}")

        except Exception as e:
            self.logger.warning(f"⚠️ Could not calculate model parameters: {str(e)}")

        return self.autoencoder

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        trial: optuna.Trial | None = None,
    ) -> Any:
        """Train the autoencoder with enhanced logging."""

        # Configure callbacks
        early_stopping_patience = self.config.get(
            "autoencoder.early_stopping_patience", 10
        )
        reduce_lr_patience = self.config.get("autoencoder.reduce_lr_patience", 5)
        min_lr = self.config.get("autoencoder.min_lr", 1e-6)

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                patience=reduce_lr_patience,
                min_lr=min_lr,
            ),
        ]

        if trial and self.config.get("training.pruning_enabled", True):
            callbacks.append(TFKerasPruningCallback(trial, "val_loss"))
            self.logger.info("📊 Optuna pruning callback enabled")

        # Determine batch size
        if trial:
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
            self.logger.info(f"📊 Trial batch size: {batch_size}")
        else:
            batch_size = self.config.get("best_params", {}).get("batch_size", 32)
            self.logger.info(f"📊 Final training batch size: {batch_size}")

        epochs = self.config.get("autoencoder.epochs", 100)

        self.logger.info(f"🚀 Starting autoencoder training...")
        self.logger.info(
            f"📊 Training data: {X_train.shape[0]} sequences, {X_train.shape[1]} timesteps, {X_train.shape[2]} features"
        )
        self.logger.info(
            f"📊 Validation data: {X_val.shape[0]} sequences, {X_val.shape[1]} timesteps, {X_val.shape[2]} features"
        )
        self.logger.info(
            f"📊 Training configuration: epochs={epochs}, batch_size={batch_size}"
        )
        self.logger.info(
            f"📊 Callbacks: EarlyStopping(patience={early_stopping_patience}), ReduceLROnPlateau(patience={reduce_lr_patience})"
        )

        # Track training time
        start_time = time.time()

        history = self.autoencoder.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        training_time = time.time() - start_time

        # Enhanced training summary
        try:
            val_losses = history.history.get("val_loss", [])
            train_losses = history.history.get("loss", [])
            val_mae = history.history.get("val_mae", [])
            train_mae = history.history.get("mae", [])

            if val_losses:
                best_epoch = int(np.argmin(val_losses))
                best_val_loss = val_losses[best_epoch]
                final_train_loss = train_losses[-1] if train_losses else 0
                final_val_loss = val_losses[-1]

                self.logger.info(f"✅ Autoencoder training completed successfully!")
                self.logger.info(f"📊 Training time: {training_time:.2f} seconds")
                self.logger.info(f"📊 Epochs trained: {len(val_losses)}")
                self.logger.info(f"📊 Best epoch: {best_epoch + 1}")
                self.logger.info(f"📊 Best validation loss: {best_val_loss:.6f}")
                self.logger.info(f"📊 Final training loss: {final_train_loss:.6f}")
                self.logger.info(f"📊 Final validation loss: {final_val_loss:.6f}")

                if val_mae:
                    best_val_mae = val_mae[best_epoch]
                    final_val_mae = val_mae[-1]
                    self.logger.info(f"📊 Best validation MAE: {best_val_mae:.6f}")
                    self.logger.info(f"📊 Final validation MAE: {final_val_mae:.6f}")

                # Performance assessment
                if best_val_loss < 0.1:
                    performance = "Excellent"
                elif best_val_loss < 0.3:
                    performance = "Good"
                elif best_val_loss < 0.5:
                    performance = "Acceptable"
                else:
                    performance = "Needs improvement"

                self.logger.info(f"📊 Model performance: {performance}")

        except Exception as e:
            self.logger.warning(
                f"⚠️ Could not extract detailed training metrics: {str(e)}"
            )

        return history


class AutoencoderFeatureAnalyzer:
    """Comprehensive feature importance analysis for autoencoder-generated features."""

    def __init__(self, config: AutoencoderConfig):
        if not DEPENDENCIES_AVAILABLE:
            msg = f"Required dependencies not available: {MISSING_DEPENDENCY}"
            raise ImportError(msg)

        self.config = config
        self.logger = system_logger.getChild("AutoencoderFeatureAnalyzer")

        # Analysis results storage
        self.importance_scores = {}
        self.correlation_analysis = {}
        self.stability_metrics = {}
        self.regime_analysis = {}

    def analyze_feature_importance(
        self,
        encoded_features: pd.DataFrame,
        labels: np.ndarray,
        original_features: pd.DataFrame | None = None,
        regime_labels: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Comprehensive analysis of autoencoder feature importance.

        Args:
            encoded_features: DataFrame with autoencoder features
            labels: Target labels for prediction
            original_features: Original features for comparison (optional)
            regime_labels: Market regime labels for regime-specific analysis (optional)

        Returns:
            Dictionary containing all analysis results
        """
        try:
            self.logger.info(
                "🔍 Starting comprehensive autoencoder feature importance analysis..."
            )
            self.logger.info(f"📊 Encoded features shape: {encoded_features.shape}")
            self.logger.info(f"🎯 Labels shape: {labels.shape}")
            self.logger.info(f"📈 Unique labels: {len(np.unique(labels))}")

            # Store analysis results
            analysis_results = {
                "feature_importance": {},
                "correlation_analysis": {},
                "stability_metrics": {},
                "regime_analysis": {},
                "summary_statistics": {},
                "recommendations": [],
            }

            # 1. Statistical Correlation Analysis
            self.logger.info("📊 Performing statistical correlation analysis...")
            correlation_results = self._analyze_correlations(encoded_features, labels)
            analysis_results["correlation_analysis"] = correlation_results

            # 2. Machine Learning Feature Importance
            self.logger.info("🤖 Computing ML-based feature importance...")
            ml_importance = self._compute_ml_importance(encoded_features, labels)
            analysis_results["feature_importance"] = ml_importance

            # 3. Feature Stability Analysis
            self.logger.info("📈 Analyzing feature stability...")
            stability_results = self._analyze_feature_stability(encoded_features)
            analysis_results["stability_metrics"] = stability_results

            # 4. Regime-Specific Analysis (if regime labels provided)
            if regime_labels is not None:
                self.logger.info("🔄 Performing regime-specific analysis...")
                regime_results = self._analyze_regime_specific_importance(
                    encoded_features, labels, regime_labels
                )
                analysis_results["regime_analysis"] = regime_results

            # 5. Comparison with Original Features (if provided)
            if original_features is not None:
                self.logger.info("🔄 Comparing with original features...")
                comparison_results = self._compare_with_original_features(
                    encoded_features, original_features, labels
                )
                analysis_results["original_comparison"] = comparison_results

            # 6. Generate Summary and Recommendations
            self.logger.info("📋 Generating summary and recommendations...")
            summary, recommendations = self._generate_summary_and_recommendations(
                analysis_results
            )
            analysis_results["summary_statistics"] = summary
            analysis_results["recommendations"] = recommendations

            # Store results for later access
            self.importance_scores = ml_importance
            self.correlation_analysis = correlation_results
            self.stability_metrics = stability_results
            if regime_labels is not None:
                self.regime_analysis = analysis_results["regime_analysis"]

            self.logger.info(
                "✅ Autoencoder feature importance analysis completed successfully!"
            )
            return analysis_results

        except Exception as e:
            self.logger.exception(f"❌ Error in feature importance analysis: {e}")
            return {"error": str(e)}

    def _analyze_correlations(
        self, encoded_features: pd.DataFrame, labels: np.ndarray
    ) -> dict[str, Any]:
        """Analyze statistical correlations between features and labels."""
        try:
            # Create DataFrame with features and labels
            analysis_df = encoded_features.copy()
            analysis_df["target"] = labels

            # Calculate correlations
            correlations = analysis_df.corr()["target"].drop("target")

            # Sort by absolute correlation
            abs_correlations = correlations.abs().sort_values(ascending=False)

            # Calculate mutual information (if scikit-learn available)
            try:
                from sklearn.feature_selection import (
                    mutual_info_classif,
                    mutual_info_regression,
                )

                # Determine if classification or regression
                unique_labels = len(np.unique(labels))
                if unique_labels <= 10:  # Classification
                    mi_scores = mutual_info_classif(
                        encoded_features, labels, random_state=42
                    )
                else:  # Regression
                    mi_scores = mutual_info_regression(
                        encoded_features, labels, random_state=42
                    )

                mi_df = pd.DataFrame(
                    {"feature": encoded_features.columns, "mutual_info": mi_scores}
                ).sort_values("mutual_info", ascending=False)

                self.logger.info(
                    f"📊 Mutual information computed for {len(encoded_features.columns)} features"
                )

            except ImportError:
                self.logger.warning(
                    "⚠️ scikit-learn not available, skipping mutual information"
                )
                mi_df = None

            # Identify highly correlated features
            high_corr_threshold = self.config.get(
                "feature_analysis.high_correlation_threshold", 0.7
            )
            high_correlations = correlations[correlations.abs() > high_corr_threshold]

            # Identify low correlation features
            low_corr_threshold = self.config.get(
                "feature_analysis.low_correlation_threshold", 0.1
            )
            low_correlations = correlations[correlations.abs() < low_corr_threshold]

            results = {
                "pearson_correlations": correlations.to_dict(),
                "abs_correlations": abs_correlations.to_dict(),
                "mutual_information": mi_df.to_dict("records")
                if mi_df is not None
                else None,
                "high_correlations": high_correlations.to_dict(),
                "low_correlations": low_correlations.to_dict(),
                "correlation_summary": {
                    "mean_correlation": correlations.mean(),
                    "std_correlation": correlations.std(),
                    "max_correlation": correlations.max(),
                    "min_correlation": correlations.min(),
                    "high_corr_count": len(high_correlations),
                    "low_corr_count": len(low_correlations),
                },
            }

            self.logger.info(f"📊 Correlation analysis complete:")
            self.logger.info(
                f"   📈 Mean correlation: {results['correlation_summary']['mean_correlation']:.4f}"
            )
            self.logger.info(
                f"   📈 Max correlation: {results['correlation_summary']['max_correlation']:.4f}"
            )
            self.logger.info(
                f"   📈 High correlation features: {results['correlation_summary']['high_corr_count']}"
            )
            self.logger.info(
                f"   📈 Low correlation features: {results['correlation_summary']['low_corr_count']}"
            )

            return results

        except Exception as e:
            self.logger.exception(f"❌ Error in correlation analysis: {e}")
            return {"error": str(e)}

    def _compute_ml_importance(
        self, encoded_features: pd.DataFrame, labels: np.ndarray
    ) -> dict[str, Any]:
        """Compute machine learning-based feature importance."""
        try:
            # Prepare data
            X = encoded_features.select_dtypes(include=[np.number]).fillna(0)
            # Safety: ensure raw numeric columns are excluded
            raw_in_numeric = [c for c in X.columns if c in self.raw_columns]
            if raw_in_numeric:
                self.logger.warning(f"🚨 Removing raw numeric columns from candidate features: {raw_in_numeric}")
                X = X.drop(columns=raw_in_numeric)
                features_df = features_df.drop(columns=raw_in_numeric)
            y = labels

            if len(np.unique(y)) < 2:
                self.logger.warning(
                    "⚠️ Insufficient unique labels for ML importance analysis"
                )
                return {"error": "Insufficient unique labels"}

            # 1. Random Forest Importance
            self.logger.info("🌲 Computing Random Forest feature importance...")
            rf_model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
            rf_model.fit(X, y)
            rf_importance = pd.DataFrame(
                {"feature": X.columns, "importance": rf_model.feature_importances_}
            ).sort_values("importance", ascending=False)

            # 2. Gradient Boosting Importance (if available)
            try:
                from sklearn.ensemble import GradientBoostingClassifier

                gb_model = GradientBoostingClassifier(
                    n_estimators=100, max_depth=6, random_state=42
                )
                gb_model.fit(X, y)
                gb_importance = pd.DataFrame(
                    {"feature": X.columns, "importance": gb_model.feature_importances_}
                ).sort_values("importance", ascending=False)

                self.logger.info("🌳 Gradient Boosting importance computed")

            except ImportError:
                self.logger.warning("⚠️ Gradient Boosting not available")
                gb_importance = None

            # 3. Permutation Importance (more robust)
            try:
                from sklearn.inspection import permutation_importance
                from sklearn.model_selection import train_test_split

                # Split data for permutation importance
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.3,
                    random_state=42,
                    stratify=y if len(np.unique(y)) <= 10 else None,
                )

                # Use a simple model for permutation importance
                from sklearn.linear_model import LogisticRegression

                perm_model = LogisticRegression(random_state=42, max_iter=1000)
                perm_model.fit(X_train, y_train)

                # Compute permutation importance
                perm_importance = permutation_importance(
                    perm_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
                )

                perm_df = pd.DataFrame(
                    {
                        "feature": X.columns,
                        "importance": perm_importance.importances_mean,
                        "std": perm_importance.importances_std,
                    }
                ).sort_values("importance", ascending=False)

                self.logger.info("🔄 Permutation importance computed")

            except ImportError:
                self.logger.warning("⚠️ Permutation importance not available")
                perm_df = None

            # 4. Aggregate importance scores
            importance_methods = {
                "random_forest": rf_importance,
                "gradient_boosting": gb_importance,
                "permutation": perm_df,
            }

            # Compute ensemble importance (average across methods)
            available_methods = {
                k: v for k, v in importance_methods.items() if v is not None
            }

            if len(available_methods) > 1:
                # Normalize importance scores to [0, 1] range
                normalized_importance = {}
                for method_name, method_df in available_methods.items():
                    if method_df is not None:
                        normalized_importance[method_name] = (
                            method_df["importance"] / method_df["importance"].max()
                        )

                # Compute ensemble importance
                ensemble_scores = pd.DataFrame(normalized_importance).mean(axis=1)
                ensemble_df = pd.DataFrame(
                    {"feature": X.columns, "ensemble_importance": ensemble_scores}
                ).sort_values("ensemble_importance", ascending=False)

                self.logger.info(
                    "🎯 Ensemble importance computed from multiple methods"
                )
            else:
                ensemble_df = rf_importance.copy()
                ensemble_df.columns = ["feature", "ensemble_importance"]

            results = {
                "random_forest": rf_importance.to_dict("records"),
                "gradient_boosting": gb_importance.to_dict("records")
                if gb_importance is not None
                else None,
                "permutation": perm_df.to_dict("records")
                if perm_df is not None
                else None,
                "ensemble": ensemble_df.to_dict("records"),
                "importance_summary": {
                    "top_features": ensemble_df.head(10)["feature"].tolist(),
                    "bottom_features": ensemble_df.tail(10)["feature"].tolist(),
                    "mean_importance": ensemble_df["ensemble_importance"].mean(),
                    "std_importance": ensemble_df["ensemble_importance"].std(),
                },
            }

            self.logger.info(f"🤖 ML importance analysis complete:")
            self.logger.info(
                f"   🏆 Top 5 features: {results['importance_summary']['top_features'][:5]}"
            )
            self.logger.info(
                f"   📊 Mean importance: {results['importance_summary']['mean_importance']:.4f}"
            )

            return results

        except Exception as e:
            self.logger.exception(f"❌ Error in ML importance analysis: {e}")
            return {"error": str(e)}

    def _analyze_feature_stability(
        self, encoded_features: pd.DataFrame
    ) -> dict[str, Any]:
        """Analyze feature stability over time."""
        try:
            # Calculate rolling statistics to assess stability
            window_size = self.config.get("feature_analysis.stability_window", 100)

            stability_metrics = {}

            for column in encoded_features.columns:
                if column in ["autoencoder_recon_error"]:
                    continue  # Skip reconstruction error for stability analysis

                feature_data = encoded_features[column].dropna()

                if len(feature_data) < window_size * 2:
                    continue

                # Rolling statistics
                rolling_mean = feature_data.rolling(
                    window=window_size, min_periods=window_size // 2
                ).mean()
                rolling_std = feature_data.rolling(
                    window=window_size, min_periods=window_size // 2
                ).std()

                # Stability metrics
                mean_stability = 1 - (rolling_std / (rolling_mean.abs() + 1e-8)).mean()
                trend_stability = 1 - abs(rolling_mean.diff().mean()) / (
                    feature_data.std() + 1e-8
                )

                # Coefficient of variation
                cv = feature_data.std() / (feature_data.mean() + 1e-8)

                stability_metrics[column] = {
                    "mean_stability": mean_stability,
                    "trend_stability": trend_stability,
                    "coefficient_of_variation": cv,
                    "overall_stability": (mean_stability + trend_stability) / 2,
                }

            # Create stability DataFrame
            stability_df = pd.DataFrame.from_dict(stability_metrics, orient="index")
            stability_df = stability_df.sort_values(
                "overall_stability", ascending=False
            )

            # Identify stable and unstable features
            stability_threshold = self.config.get(
                "feature_analysis.stability_threshold", 0.7
            )
            stable_features = stability_df[
                stability_df["overall_stability"] > stability_threshold
            ].index.tolist()
            unstable_features = stability_df[
                stability_df["overall_stability"] < (1 - stability_threshold)
            ].index.tolist()

            results = {
                "stability_metrics": stability_df.to_dict("index"),
                "stable_features": stable_features,
                "unstable_features": unstable_features,
                "stability_summary": {
                    "mean_stability": stability_df["overall_stability"].mean(),
                    "stable_count": len(stable_features),
                    "unstable_count": len(unstable_features),
                    "stability_threshold": stability_threshold,
                },
            }

            self.logger.info(f"📈 Stability analysis complete:")
            self.logger.info(
                f"   📊 Mean stability: {results['stability_summary']['mean_stability']:.4f}"
            )
            self.logger.info(
                f"   📊 Stable features: {results['stability_summary']['stable_count']}"
            )
            self.logger.info(
                f"   📊 Unstable features: {results['stability_summary']['unstable_count']}"
            )

            return results

        except Exception as e:
            self.logger.exception(f"❌ Error in stability analysis: {e}")
            return {"error": str(e)}

    def _analyze_regime_specific_importance(
        self,
        encoded_features: pd.DataFrame,
        labels: np.ndarray,
        regime_labels: np.ndarray,
    ) -> dict[str, Any]:
        """Analyze feature importance across different market regimes."""
        try:
            unique_regimes = np.unique(regime_labels)
            self.logger.info(
                f"🔄 Analyzing feature importance across {len(unique_regimes)} regimes: {unique_regimes}"
            )

            regime_importance = {}

            for regime in unique_regimes:
                regime_mask = regime_labels == regime
                regime_features = encoded_features[regime_mask]
                regime_labels_subset = labels[regime_mask]

                if len(regime_features) < 50:  # Skip regimes with too few samples
                    self.logger.warning(
                        f"⚠️ Regime {regime} has insufficient samples ({len(regime_features)})"
                    )
                    continue

                # Compute importance for this regime
                regime_importance[regime] = self._compute_ml_importance(
                    regime_features, regime_labels_subset
                )

                self.logger.info(
                    f"🔄 Regime {regime}: {len(regime_features)} samples analyzed"
                )

            # Compare importance across regimes
            if len(regime_importance) > 1:
                # Find common features across all regimes
                all_features = set(encoded_features.columns)
                common_features = all_features.copy()

                for regime, importance_data in regime_importance.items():
                    if "ensemble" in importance_data:
                        regime_features = set(
                            [item["feature"] for item in importance_data["ensemble"]]
                        )
                        common_features &= regime_features

                # Compute importance consistency across regimes
                consistency_scores = {}
                for feature in common_features:
                    importances = []
                    for regime, importance_data in regime_importance.items():
                        if "ensemble" in importance_data:
                            feature_importance = next(
                                (
                                    item["ensemble_importance"]
                                    for item in importance_data["ensemble"]
                                    if item["feature"] == feature
                                ),
                                0,
                            )
                            importances.append(feature_importance)

                    if importances:
                        consistency_scores[feature] = {
                            "mean_importance": np.mean(importances),
                            "std_importance": np.std(importances),
                            "consistency": 1
                            - (np.std(importances) / (np.mean(importances) + 1e-8)),
                        }

                consistency_df = pd.DataFrame.from_dict(
                    consistency_scores, orient="index"
                )
                consistency_df = consistency_df.sort_values(
                    "consistency", ascending=False
                )

                results = {
                    "regime_importance": regime_importance,
                    "consistency_analysis": consistency_df.to_dict("index"),
                    "consistent_features": consistency_df[
                        consistency_df["consistency"] > 0.8
                    ].index.tolist(),
                    "inconsistent_features": consistency_df[
                        consistency_df["consistency"] < 0.3
                    ].index.tolist(),
                }
            else:
                results = {
                    "regime_importance": regime_importance,
                    "consistency_analysis": {},
                    "consistent_features": [],
                    "inconsistent_features": [],
                }

            self.logger.info(f"🔄 Regime analysis complete:")
            self.logger.info(f"   📊 Regimes analyzed: {len(regime_importance)}")
            self.logger.info(
                f"   📊 Consistent features: {len(results.get('consistent_features', []))}"
            )

            return results

        except Exception as e:
            self.logger.exception(f"❌ Error in regime analysis: {e}")
            return {"error": str(e)}

    def _compare_with_original_features(
        self,
        encoded_features: pd.DataFrame,
        original_features: pd.DataFrame,
        labels: np.ndarray,
    ) -> dict[str, Any]:
        """Compare autoencoder features with original features."""
        try:
            # Compute importance for both feature sets
            encoded_importance = self._compute_ml_importance(encoded_features, labels)
            original_importance = self._compute_ml_importance(original_features, labels)

            # Compare performance
            comparison_results = {
                "encoded_importance": encoded_importance,
                "original_importance": original_importance,
                "comparison_metrics": {},
            }

            # Extract top features for comparison
            if "ensemble" in encoded_importance and "ensemble" in original_importance:
                encoded_top = [
                    item["feature"] for item in encoded_importance["ensemble"][:10]
                ]
                original_top = [
                    item["feature"] for item in original_importance["ensemble"][:10]
                ]

                # Calculate overlap
                overlap = set(encoded_top) & set(original_top)
                overlap_ratio = len(overlap) / len(encoded_top)

                comparison_results["comparison_metrics"] = {
                    "top_feature_overlap": len(overlap),
                    "overlap_ratio": overlap_ratio,
                    "encoded_top_features": encoded_top,
                    "original_top_features": original_top,
                }

            self.logger.info(f"🔄 Feature comparison complete:")
            if "comparison_metrics" in comparison_results:
                self.logger.info(
                    f"   📊 Top feature overlap: {comparison_results['comparison_metrics']['top_feature_overlap']}"
                )
                self.logger.info(
                    f"   📊 Overlap ratio: {comparison_results['comparison_metrics']['overlap_ratio']:.3f}"
                )

            return comparison_results

        except Exception as e:
            self.logger.exception(f"❌ Error in feature comparison: {e}")
            return {"error": str(e)}

    def _generate_summary_and_recommendations(
        self, analysis_results: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        """Generate summary statistics and actionable recommendations."""
        try:
            summary = {}
            recommendations = []

            # Extract key metrics
            if (
                "feature_importance" in analysis_results
                and "importance_summary" in analysis_results["feature_importance"]
            ):
                importance_summary = analysis_results["feature_importance"][
                    "importance_summary"
                ]
                summary["top_features"] = importance_summary.get("top_features", [])
                summary["mean_importance"] = importance_summary.get(
                    "mean_importance", 0
                )

            if (
                "correlation_analysis" in analysis_results
                and "correlation_summary" in analysis_results["correlation_analysis"]
            ):
                corr_summary = analysis_results["correlation_analysis"][
                    "correlation_summary"
                ]
                summary["mean_correlation"] = corr_summary.get("mean_correlation", 0)
                summary["high_corr_count"] = corr_summary.get("high_corr_count", 0)

            if (
                "stability_metrics" in analysis_results
                and "stability_summary" in analysis_results["stability_metrics"]
            ):
                stability_summary = analysis_results["stability_metrics"][
                    "stability_summary"
                ]
                summary["mean_stability"] = stability_summary.get("mean_stability", 0)
                summary["stable_count"] = stability_summary.get("stable_count", 0)

            # Generate recommendations
            if summary.get("mean_importance", 0) < 0.3:
                recommendations.append(
                    "⚠️ Low feature importance detected. Consider retraining autoencoder with different parameters."
                )

            if summary.get("mean_correlation", 0) < 0.1:
                recommendations.append(
                    "⚠️ Low correlation with targets. Autoencoder features may not be capturing relevant patterns."
                )

            if summary.get("mean_stability", 0) < 0.5:
                recommendations.append(
                    "⚠️ Low feature stability. Consider using more stable features or retraining with different data."
                )

            if summary.get("high_corr_count", 0) > 5:
                recommendations.append(
                    "💡 High correlation features detected. Consider feature selection to reduce redundancy."
                )

            if summary.get("stable_count", 0) > 10:
                recommendations.append(
                    "✅ Good feature stability detected. These features should perform well in production."
                )

            # Add positive recommendations
            if summary.get("mean_importance", 0) > 0.7:
                recommendations.append(
                    "🎉 High feature importance detected. Autoencoder is generating valuable features."
                )

            if summary.get("mean_correlation", 0) > 0.3:
                recommendations.append(
                    "🎉 Good correlation with targets. Autoencoder features are capturing relevant patterns."
                )

            return summary, recommendations

        except Exception as e:
            self.logger.exception(f"❌ Error generating summary: {e}")
            return {}, [f"Error generating summary: {e}"]

    def get_feature_ranking(self, method: str = "ensemble") -> pd.DataFrame:
        """Get feature ranking based on specified method."""
        if method not in self.importance_scores:
            self.logger.warning(
                f"⚠️ Method '{method}' not available. Available methods: {list(self.importance_scores.keys())}"
            )
            return pd.DataFrame()

        if "ensemble" in self.importance_scores:
            return pd.DataFrame(self.importance_scores["ensemble"])
        else:
            return pd.DataFrame(self.importance_scores[method])

    def get_stable_features(self, threshold: float = 0.7) -> list[str]:
        """Get list of stable features above threshold."""
        if "stability_metrics" not in self.stability_metrics:
            return []

        stable_features = []
        for feature, metrics in self.stability_metrics["stability_metrics"].items():
            if metrics.get("overall_stability", 0) > threshold:
                stable_features.append(feature)

        return stable_features

    def get_high_correlation_features(self, threshold: float = 0.5) -> list[str]:
        """Get list of features with high correlation to target."""
        if "correlation_analysis" not in self.correlation_analysis:
            return []

        high_corr_features = []
        correlations = self.correlation_analysis["correlation_analysis"].get(
            "pearson_correlations", {}
        )

        for feature, corr in correlations.items():
            if abs(corr) > threshold:
                high_corr_features.append(feature)

        return high_corr_features


class AutoencoderFeatureGenerator:
    """Main class for the complete autoencoder feature generation workflow."""

    def __init__(self, config: str | dict | None = None):
        if not DEPENDENCIES_AVAILABLE:
            msg = f"Required dependencies not available: {MISSING_DEPENDENCY}"
            raise ImportError(
                msg,
            )

        # Handle both string (config path) and dict (config object) inputs
        if isinstance(config, dict):
            # Create a config object with proper initialization
            temp_config = AutoencoderConfig()
            temp_config.config = config
            # Ensure logger is properly set
            temp_config.logger = system_logger.getChild("AutoencoderConfig")
            self.config = temp_config
        else:
            # Handle string path or None
            self.config = AutoencoderConfig(config)

        self.logger = system_logger.getChild("AutoencoderFeatureGenerator")

    def generate_features(
        self,
        features_df: pd.DataFrame,
        regime_name: str,
        labels: np.ndarray,
        regime_labels: np.ndarray | None = None,
        enable_analysis: bool | None = None,
    ) -> pd.DataFrame:
        """
        Generate autoencoder features from input features.
        
        CRITICAL: This method should only receive engineered features, not raw OHLCV data.
        Raw price data like 'volume', 'close', 'open', 'high', 'low' should be excluded.
        """
        # CRITICAL: Filter out raw OHLCV data that should not be used as features
        raw_ohlcv_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'time', 'trade_volume', 'trade_count', 'avg_price', 'min_price', 'max_price']
        raw_ohlcv_columns = [col for col in raw_ohlcv_columns if col in features_df.columns]
        
        # Drop calendar/metadata columns that are not predictive features
        calendar_cols = [c for c in features_df.columns if c in {
            'year','month','day','day_of_week','day_of_month','quarter','exchange','symbol','timeframe','split'
        }]
        drop_cols = list(dict.fromkeys([*raw_ohlcv_columns, *calendar_cols]))
        
        if drop_cols:
            self.logger.warning(f"🚨 Removing non-feature columns to prevent leakage/noise: {drop_cols}")
            features_df = features_df.drop(columns=drop_cols)
            self.logger.info(f"✅ Removed {len(drop_cols)} non-feature columns from features")
            self.logger.info(f"📊 Features shape after removal: {features_df.shape}")
            
            if features_df.empty:
                self.logger.error("🚨 CRITICAL: No engineered features remaining after removing non-feature columns")
                self.logger.error("🚨 This indicates a serious data pipeline issue")
                return pd.DataFrame()
        """Generate autoencoder-based features for a specific market regime."""
        try:
            self.logger.info(
                f"🚀 Starting autoencoder feature generation for regime: {regime_name}",
            )

            # CRITICAL DATA LEAKAGE PREVENTION: Check for label columns in features
            potential_label_columns = [
                "label",
                "target",
                "y",
                "class",
                "Label",
                "Target",
                "Y",
                "Class",
                "labels",
                "targets",
                "classes",
                "Labels",
                "Targets",
                "Classes",
                "signal",
                "prediction",
                "direction",
                "Signal",
                "Prediction",
                "Direction",
                "buy_sell",
                "position",
                "trade_signal",
                "Buy_Sell",
                "Position",
                "Trade_Signal",
                "future_return",
                "next_return",
                "price_change",
                "Future_Return",
                "Next_Return",
                "Price_Change",
                "binary_target",
                "binary_label",
                "Binary_Target",
                "Binary_Label",
                "multi_target",
                "multi_label",
                "Multi_Target",
                "Multi_Label",
                "label_encoded",
                "target_encoded",
                "Label_Encoded",
                "Target_Encoded",
                "meta_label",
                "meta_target",
                "Meta_Label",
                "Meta_Target",
                "triple_barrier_label",
                "barrier_label",
                "Triple_Barrier_Label",
                "Barrier_Label",
            ]

            actual_label_columns = [
                col for col in features_df.columns if col in potential_label_columns
            ]

            if actual_label_columns:
                self.logger.error(f"🚨 CRITICAL DATA LEAKAGE DETECTED in autoencoder!")
                self.logger.error(
                    f"🚨 Found label columns in autoencoder input: {actual_label_columns}"
                )
                self.logger.error(
                    "🚨 This will cause severe data leakage! Removing these columns from autoencoder analysis."
                )

                # Remove the label columns
                features_df = features_df.drop(columns=actual_label_columns)
                self.logger.info(
                    f"📊 Autoencoder features after leakage prevention: {features_df.shape[1]} columns"
                )

            # Check if we have enough data
            if len(features_df) < 10:
                self.logger.warning(
                    "⚠️ Insufficient data for autoencoder feature generation, returning original features",
                )
                return features_df

            # NEW STEP: Convert price features to returns for better autoencoder training
            self.logger.info(
                "🔄 NEW STEP: Converting price features to returns for autoencoder training..."
            )
            price_converter = PriceReturnConverter(self.config)
            features_df = price_converter.convert_price_features_to_returns(features_df)
            self.logger.info(
                f"✅ Price return conversion completed. Features shape: {features_df.shape}"
            )

            # Step 1: Filter features using Random Forest + SHAP
            self.logger.info("🔄 Step 1/5: Feature filtering with Random Forest + SHAP")
            self.logger.info(f"📊 Starting with {features_df.shape[1]} input features")

            feature_filter = FeatureFilter(self.config)
            filtered_features = feature_filter.filter_features(features_df, labels)

            # Check if features_df has any columns to avoid division by zero
            if features_df.shape[1] == 0:
                self.logger.warning("⚠️ No features available for filtering - returning original features")
                return features_df
            
            feature_reduction = features_df.shape[1] - filtered_features.shape[1]
            reduction_percentage = (feature_reduction / features_df.shape[1]) * 100
            self.logger.info(f"✅ Feature filtering completed successfully!")
            self.logger.info(
                f"📊 Results: {filtered_features.shape[1]} features selected from {features_df.shape[1]} input features"
            )
            self.logger.info(
                f"📉 Feature reduction: {feature_reduction} features removed ({reduction_percentage:.1f}% reduction)"
            )

            # AE input health: enforce minimum feature count and per-feature std threshold
            self.logger.info(
                "🔍 Validating feature quality for autoencoder training..."
            )

            min_features_for_ae = int(
                self.config.get("feature_filtering.min_features_for_ae", 15)
            )
            numeric_features = filtered_features.select_dtypes(include=[np.number])
            actual_numeric_features = numeric_features.shape[1]

            self.logger.info(
                f"📊 Numeric features available: {actual_numeric_features}"
            )
            self.logger.info(f"📊 Minimum features required: {min_features_for_ae}")

            if actual_numeric_features < min_features_for_ae:
                self.logger.warning(f"⚠️ Insufficient features for autoencoder training")
                self.logger.warning(
                    f"📊 Have: {actual_numeric_features} numeric features, Need: {min_features_for_ae}+ features"
                )
                self.logger.info(
                    "🔄 Returning original features without autoencoder enhancement"
                )
                return features_df

            # Check feature variance/standard deviation
            std_threshold = float(self.config.get("autoencoder.min_feature_std", 1e-6))
            per_feature_std = numeric_features.std(axis=0, skipna=True)
            low_std_cols = per_feature_std.index[
                per_feature_std <= std_threshold
            ].tolist()

            if len(low_std_cols) > 0:
                preview = ", ".join(low_std_cols[:10]) + (
                    "..." if len(low_std_cols) > 10 else ""
                )
                self.logger.warning(f"⚠️ Low variance features detected")
                self.logger.warning(
                    f"📊 {len(low_std_cols)} features have std <= {std_threshold:g}"
                )
                self.logger.warning(f"📊 Examples: {preview}")
                self.logger.info(
                    "🔄 Returning original features without autoencoder enhancement"
                )
                return features_df

            self.logger.info(
                "✅ Feature quality validation passed - proceeding with autoencoder training"
            )

            # Step 2: Preprocess and create sequences
            self.logger.info("🔄 Step 2/5: Data preprocessing and sequence creation")

            # Data preprocessing
            self.logger.info("🔧 Initializing data preprocessor...")
            preprocessor = ImprovedAutoencoderPreprocessor(self.config)

            self.logger.info("🔧 Fitting preprocessor on filtered features...")
            preprocessor.fit(filtered_features)

            self.logger.info("🔧 Transforming features for autoencoder input...")
            X_processed = preprocessor.transform(filtered_features)
            self.logger.info(f"✅ Preprocessing completed successfully")
            self.logger.info(f"📊 Processed data shape: {X_processed.shape}")

            # Sequence creation
            timesteps = self.config.get("sequence.timesteps", 10)
            self.logger.info(f"📊 Creating sequences with {timesteps} timesteps...")

            X_sequences, y_targets, target_indices = create_sequences_with_index(
                X_processed,
                timesteps,
                filtered_features.index,
            )

            self.logger.info(f"✅ Sequence creation completed successfully")
            self.logger.info(
                f"📊 Sequence shapes: X_sequences={X_sequences.shape}, y_targets={y_targets.shape}"
            )
            self.logger.info(
                f"📊 Sequence configuration: timesteps={timesteps}, overlap=50%"
            )
            self.logger.info(
                f"📊 Target indices: {len(target_indices)} samples with preserved timestamps"
            )

            # Check if we have enough sequences
            min_sequences = 5
            if len(X_sequences) < min_sequences:
                self.logger.warning(
                    f"⚠️ Insufficient sequences for autoencoder training"
                )
                self.logger.warning(
                    f"📊 Have: {len(X_sequences)} sequences, Need: {min_sequences}+ sequences"
                )
                self.logger.info(
                    "🔄 Returning original features without autoencoder enhancement"
                )
                return features_df

            # Step 3: Optimize hyperparameters with Optuna
            self.logger.info("🔄 Step 3/5: Hyperparameter optimization with Optuna")

            # Data splitting for training/validation
            split_ratio = 0.8
            split_idx = int(split_ratio * len(X_sequences))
            X_train, y_train = X_sequences[:split_idx], y_targets[:split_idx]
            X_val, y_val = X_sequences[split_idx:], y_targets[split_idx:]

            self.logger.info(
                f"📊 Data split configuration: {split_ratio*100:.0f}% train, {(1-split_ratio)*100:.0f}% validation"
            )
            self.logger.info(
                f"📊 Training set: {X_train.shape[0]} sequences ({X_train.shape[0]/len(X_sequences)*100:.1f}%)"
            )
            self.logger.info(
                f"📊 Validation set: {X_val.shape[0]} sequences ({X_val.shape[0]/len(X_sequences)*100:.1f}%)"
            )

            # Optuna optimization
            n_trials = self.config.get("training.n_trials", 50)
            n_jobs = self.config.get("training.n_jobs", 1)

            self.logger.info(f"🔍 Starting Optuna hyperparameter optimization")
            self.logger.info(
                f"📊 Optimization parameters: n_trials={n_trials}, n_jobs={n_jobs}"
            )
            self.logger.info(
                f"📊 Search space: filters=[16,32,64], kernel_size=[3-7], dropout=[0.1-0.5], lr=[1e-4-1e-2], encoding_dim=[8-64]"
            )

            best_params = self._run_optuna_optimization(X_train, y_train, X_val, y_val)
            self.config.config["best_params"] = (
                best_params  # Store best params for final training
            )

            self.logger.info(f"✅ Hyperparameter optimization completed successfully")
            self.logger.info(f"🏆 Best hyperparameters selected:")
            for param, value in best_params.items():
                self.logger.info(f"   📊 {param}: {value}")

            # Step 4: Train final model and generate features
            self.logger.info(
                "🔄 Step 4/5: Final autoencoder training and feature generation"
            )

            # Build and train final model
            self.logger.info(
                "🔧 Building final autoencoder model with optimized hyperparameters..."
            )
            final_autoencoder = SequenceAwareAutoencoder(self.config)
            final_autoencoder.build_model(X_sequences.shape[1:])

            self.logger.info("🔧 Training final autoencoder model...")
            training_history = final_autoencoder.fit(X_train, y_train, X_val, y_val)

            # Extract training metrics
            if hasattr(training_history, "history"):
                final_train_loss = training_history.history.get("loss", [0])[-1]
                final_val_loss = training_history.history.get("val_loss", [0])[-1]
                self.logger.info(f"✅ Final model training completed")
                self.logger.info(f"📊 Final training loss: {final_train_loss:.6f}")
                self.logger.info(f"📊 Final validation loss: {final_val_loss:.6f}")
                self.logger.info(
                    f"📊 Model performance: {'Good' if final_val_loss < 0.1 else 'Acceptable' if final_val_loss < 0.5 else 'Needs improvement'}"
                )

            # Generate encoded features and reconstructions
            self.logger.info("🔧 Generating encoded features and reconstructions...")
            self.logger.info("📊 Using encoder to extract latent representations...")
            encoded_features = final_autoencoder.encoder.predict(X_sequences, verbose=0)

            self.logger.info("📊 Using full autoencoder to generate reconstructions...")
            reconstructed = final_autoencoder.autoencoder.predict(
                X_sequences, verbose=0
            )

            self.logger.info(f"✅ Feature generation completed successfully")
            self.logger.info(f"📊 Encoded features shape: {encoded_features.shape}")
            self.logger.info(f"📊 Reconstructed features shape: {reconstructed.shape}")

            # Calculate reconstruction error
            self.logger.info("📊 Calculating reconstruction error...")
            recon_error = np.mean((y_targets - reconstructed) ** 2, axis=1)
            mean_recon_error = np.mean(recon_error)
            std_recon_error = np.std(recon_error)

            self.logger.info(f"📊 Reconstruction error statistics:")
            self.logger.info(f"   📊 Mean reconstruction error: {mean_recon_error:.6f}")
            self.logger.info(f"   📊 Std reconstruction error: {std_recon_error:.6f}")
            self.logger.info(
                f"   📊 Min reconstruction error: {np.min(recon_error):.6f}"
            )
            self.logger.info(
                f"   📊 Max reconstruction error: {np.max(recon_error):.6f}"
            )

            # Step 5: Create enriched DataFrame
            self.logger.info("🔄 Step 5/5: Creating enriched feature DataFrame")

            # Create encoded features DataFrame
            self.logger.info("📊 Creating encoded features DataFrame...")
            encoded_df = pd.DataFrame(
                encoded_features,
                index=target_indices,
                columns=[
                    f"autoencoder_{i+1}" for i in range(encoded_features.shape[1])
                ],
            )
            encoded_df["autoencoder_recon_error"] = recon_error

            self.logger.info(f"✅ Encoded features DataFrame created successfully")
            self.logger.info(f"📊 Encoded DataFrame shape: {encoded_df.shape}")
            self.logger.info(
                f"📊 Encoded features: {encoded_features.shape[1]} latent dimensions + 1 reconstruction error"
            )

            # Merge with original features
            self.logger.info("📊 Merging encoded features with original features...")
            result_df = features_df.merge(
                encoded_df,
                left_index=True,
                right_index=True,
                how="left",
            )

            # Identify autoencoder columns and handle missing values
            autoencoder_cols = [
                col for col in result_df.columns if "autoencoder" in col
            ]
            result_df[autoencoder_cols] = result_df[autoencoder_cols].fillna(0)

            self.logger.info(f"✅ Feature merging completed successfully")
            self.logger.info(f"📊 Original features: {features_df.shape[1]} columns")
            self.logger.info(
                f"📊 Autoencoder features added: {len(autoencoder_cols)} columns"
            )
            self.logger.info(f"📊 Final result shape: {result_df.shape}")
            self.logger.info(
                f"📊 Feature enhancement: {len(autoencoder_cols)} new features added ({len(autoencoder_cols)/features_df.shape[1]*100:.1f}% increase)"
            )

            # Step 6: Feature Importance Analysis (if enabled)
            enable_analysis = (
                enable_analysis
                if enable_analysis is not None
                else self.config.get("feature_analysis.enable_analysis", True)
            )

            if enable_analysis:
                self.logger.info("🔍 Starting feature importance analysis...")
                try:
                    # Initialize feature analyzer
                    feature_analyzer = AutoencoderFeatureAnalyzer(self.config)

                    # Extract autoencoder features for analysis
                    autoencoder_features = result_df[autoencoder_cols].copy()

                    # Perform comprehensive analysis
                    analysis_results = feature_analyzer.analyze_feature_importance(
                        encoded_features=autoencoder_features,
                        labels=labels,
                        original_features=features_df
                        if self.config.get(
                            "feature_analysis.comparison_with_original", True
                        )
                        else None,
                        regime_labels=regime_labels
                        if self.config.get(
                            "feature_analysis.regime_analysis_enabled", True
                        )
                        else None,
                    )

                    # Log analysis results
                    if "error" not in analysis_results:
                        self.logger.info(
                            "📊 Feature importance analysis completed successfully!"
                        )

                        # Log key findings
                        if "summary_statistics" in analysis_results:
                            summary = analysis_results["summary_statistics"]
                            self.logger.info(f"📈 Analysis Summary:")
                            self.logger.info(
                                f"   🏆 Top features: {summary.get('top_features', [])[:5]}"
                            )
                            self.logger.info(
                                f"   📊 Mean importance: {summary.get('mean_importance', 0):.4f}"
                            )
                            self.logger.info(
                                f"   📊 Mean correlation: {summary.get('mean_correlation', 0):.4f}"
                            )
                            self.logger.info(
                                f"   📊 Mean stability: {summary.get('mean_stability', 0):.4f}"
                            )

                        # Log recommendations
                        if "recommendations" in analysis_results:
                            recommendations = analysis_results["recommendations"]
                            if recommendations:
                                self.logger.info("💡 Recommendations:")
                                for rec in recommendations[
                                    :5
                                ]:  # Show top 5 recommendations
                                    self.logger.info(f"   {rec}")

                        # Store analysis results for later access
                        self.last_analysis_results = analysis_results

                    else:
                        self.logger.warning(
                            f"⚠️ Feature analysis failed: {analysis_results['error']}"
                        )

                except Exception as e:
                    self.logger.exception(
                        f"❌ Error in feature importance analysis: {e}"
                    )
                    self.logger.info("🔄 Continuing without feature analysis...")

            # Final summary
            self.logger.info(
                "🎉 Autoencoder feature generation pipeline completed successfully!"
            )
            self.logger.info(f"📊 Summary for regime '{regime_name}':")
            self.logger.info(f"   📊 Input features: {features_df.shape[1]} columns")
            self.logger.info(f"   📊 Output features: {result_df.shape[1]} columns")
            self.logger.info(
                f"   📊 New autoencoder features: {len(autoencoder_cols)} columns"
            )
            self.logger.info(f"   📊 Data samples: {result_df.shape[0]} rows")
            self.logger.info(
                f"   📊 Autoencoder performance: {'Good' if mean_recon_error < 0.1 else 'Acceptable' if mean_recon_error < 0.5 else 'Needs improvement'}"
            )

            return result_df

        except Exception as e:
            self.logger.exception("❌ Error in autoencoder feature generation pipeline")
            self.logger.error(f"📊 Error details: {str(e)}")
            self.logger.info(
                "🔄 Returning original features without autoencoder enhancement"
            )
            return features_df

    def _run_optuna_optimization(self, X_train, y_train, X_val, y_val):
        """Helper to encapsulate the Optuna study logic."""

        def objective(trial):
            try:
                autoencoder = SequenceAwareAutoencoder(self.config)
                autoencoder.build_model(X_train.shape[1:], trial)
                history = autoencoder.fit(X_train, y_train, X_val, y_val, trial)
                return min(history.history["val_loss"])
            except Exception as e:
                self.logger.warning(f"⚠️ Trial failed: {str(e)}")
                return float("inf")  # Return high loss for failed trials

        # Create Optuna study with enhanced logging
        self.logger.info("🔧 Creating Optuna study for hyperparameter optimization...")
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(),
        )

        n_trials = self.config.get("training.n_trials", 50)
        n_jobs = self.config.get("training.n_jobs", 1)

        self.logger.info(f"🚀 Starting Optuna optimization with {n_trials} trials...")
        self.logger.info(
            f"📊 Parallel jobs: {n_jobs} (1 recommended for GPU compatibility)"
        )

        # Track optimization progress
        start_time = time.time()

        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,  # Default to 1 for GPU
        )

        optimization_time = time.time() - start_time

        self.logger.info(f"✅ Optuna optimization completed successfully!")
        self.logger.info(f"📊 Optimization time: {optimization_time:.2f} seconds")
        self.logger.info(f"📊 Trials completed: {len(study.trials)}")
        self.logger.info(
            f"📊 Successful trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}"
        )
        self.logger.info(
            f"📊 Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}"
        )
        self.logger.info(f"🏆 Best validation loss: {study.best_value:.6f}")
        self.logger.info(f"🏆 Best trial number: {study.best_trial.number}")

        return study.best_params

    def get_last_analysis_results(self) -> dict[str, Any] | None:
        """Get the results from the last feature importance analysis."""
        return getattr(self, "last_analysis_results", None)

    def get_feature_ranking(self, method: str = "ensemble") -> pd.DataFrame:
        """Get feature ranking from the last analysis."""
        analysis_results = self.get_last_analysis_results()
        if analysis_results and "feature_importance" in analysis_results:
            feature_importance = analysis_results["feature_importance"]
            if method in feature_importance and feature_importance[method] is not None:
                return pd.DataFrame(feature_importance[method])
        return pd.DataFrame()

    def get_stable_features(self, threshold: float = 0.7) -> list[str]:
        """Get list of stable features from the last analysis."""
        analysis_results = self.get_last_analysis_results()
        if analysis_results and "stability_metrics" in analysis_results:
            stability_metrics = analysis_results["stability_metrics"]
            if "stable_features" in stability_metrics:
                return stability_metrics["stable_features"]
        return []

    def get_high_correlation_features(self, threshold: float = 0.5) -> list[str]:
        """Get list of features with high correlation to target from the last analysis."""
        analysis_results = self.get_last_analysis_results()
        if analysis_results and "correlation_analysis" in analysis_results:
            correlation_analysis = analysis_results["correlation_analysis"]
            if "high_correlations" in correlation_analysis:
                return list(correlation_analysis["high_correlations"].keys())
        return []

    def get_recommendations(self) -> list[str]:
        """Get recommendations from the last analysis."""
        analysis_results = self.get_last_analysis_results()
        if analysis_results and "recommendations" in analysis_results:
            return analysis_results["recommendations"]
        return []
