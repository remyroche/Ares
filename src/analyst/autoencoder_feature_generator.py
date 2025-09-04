import copy
import logging
import os
import time
from pathlib import Path
from typing import Any
import yaml
from src.utils.logger import system_logger
import time
import os.path
from typing import Dict, List, Optional, Union, Any, Tuple
try:
    import numpy as np
    import optuna
    import pandas as pd
    import tensorflow as tf
    from optuna.integration import TFKerasPruningCallback
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
    from tensorflow.keras import Model, layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEPENDENCY = str(e)
    print(f' Missing dependency: {MISSING_DEPENDENCY}')
    print('📦 Please install required packages:')
    print('   pip install numpy pandas scikit-learn tensorflow optuna shap pyyaml')
from src.utils.logger import setup_logging
setup_logging()
logger = logging.getLogger(__name__)
project_root = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, str(project_root))
from src.config import CONFIG
from src.core.decorators import handles_errors
from src.utils.warning_symbols import error, warning
from src.core.decorators import validates as comprehensive_data_validation, validates as validate_data_quality, traced as with_tracing_span

class AutoencoderConfig:
    """Configuration manager for autoencoder feature generator."""

    def __init__(self, config_path: str | None=None) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        if not DEPENDENCIES_AVAILABLE:
            msg = f'Required dependencies not available: {MISSING_DEPENDENCY}'
            raise ImportError(msg)
        self.logger = system_logger.getChild('AutoencoderConfig')
        self.config_path = config_path or 'src/analyst/autoencoder_config.yaml'
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path) as file:
                config = yaml.safe_load(file)
            self.logger.info(f'📋 Configuration loaded successfully from {self.config_path}')
            self.logger.info(f'📊 Configuration sections: {list(config.keys())}')
            return config
        except Exception:
            self.logger.exception('⚠️ Error loading config file, using default configuration')
            return self._get_default_config()

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration if file loading fails."""
        default_config = {'preprocessing': {'scaler_type': 'robust', 'outlier_threshold': 3.0, 'missing_value_strategy': 'forward_fill', 'iqr_multiplier': 3.0, 'use_price_returns': True, 'price_return_method': 'pct_change', 'primary_price_feature': 'close', 'primary_volume_feature': 'volume', 'enable_feature_selection': True}, 'sequence': {'timesteps': 10, 'overlap': 0.5}, 'autoencoder': {'epochs': 100, 'early_stopping_patience': 10, 'reduce_lr_patience': 5, 'min_lr': 1e-06}, 'training': {'n_trials': 50, 'n_jobs': 1, 'pruning_enabled': True}, 'feature_filtering': {'n_estimators': 100, 'max_depth': 10, 'importance_threshold': 0.99, 'shap_imbalance_threshold': 100.0, 'enable_shap_imbalance_handling': True}, 'feature_analysis': {'enable_analysis': True, 'high_correlation_threshold': 0.7, 'low_correlation_threshold': 0.1, 'stability_window': 100, 'stability_threshold': 0.7, 'regime_analysis_enabled': True, 'comparison_with_original': True}, 'output': {'output_dir': 'models/autoencoder_features'}}
        return default_config

    def get(self, key: str, default: Any=None) -> Any:
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
            self.logger.info(f'📋 Configuration saved successfully to {output_path}')
        except Exception:
            self.logger.exception('⚠️ Error saving config file')

class PriceReturnConverter:
    """Convert price features to returns (price differences) for better autoencoder training."""

    def __init__(self, config: AutoencoderConfig) -> None:
        self.config = config
        self.logger = system_logger.getChild('PriceReturnConverter')
        self.use_price_returns = config.get('preprocessing.use_price_returns', True)
        self.price_return_method = config.get('preprocessing.price_return_method', 'pct_change')
        self.primary_price_feature = config.get('preprocessing.primary_price_feature', 'close')
        self.primary_volume_feature = config.get('preprocessing.primary_volume_feature', 'volume')
        self.enable_feature_selection = config.get('preprocessing.enable_feature_selection', True)

    def convert_price_features_to_returns(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert price features to returns (price differences) to improve autoencoder training.
        Optimized to select only one representative price feature and one volume feature
        to avoid redundancy.

        Args:
            features_df: DataFrame containing features, potentially including price data

        Returns:
            DataFrame with optimized price features converted to returns
        """
        if not self.use_price_returns:
            self.logger.info('📊 Price return conversion disabled, using original features')
            return features_df
        self.logger.info('🔄 Converting price features to returns for autoencoder training...')
        converted_df = features_df.copy()
        if self.enable_feature_selection:
            self.logger.info('🎯 Using optimized feature selection to avoid redundancy')
            available_price_features = []
            available_volume_features = []
            for col in converted_df.columns:
                col_lower = col.lower()
                if any((exclude_pattern in col_lower for exclude_pattern in ['regime', 'categorical', 'class', 'label', 'category', 'type'])):
                    continue
                try:
                    sample_value = converted_df[col].iloc[0]
                    if isinstance(sample_value, (np.ndarray, list)):
                        self.logger.warning(f'Skipping array-valued column {col}: contains {type(sample_value)}')
                        continue
                    unique_count = converted_df[col].nunique()
                    if unique_count <= 5:
                        continue
                except (TypeError, ValueError) as e:
                    self.logger.warning(f'Error checking uniqueness for column {col}: {e}')
                    continue
                if any((price_pattern in col_lower for price_pattern in ['open', 'high', 'low', 'close', 'price', 'avg_price', 'min_price', 'max_price'])):
                    if col not in available_price_features:
                        available_price_features.append(col)
                elif any((volume_pattern in col_lower for volume_pattern in ['volume', 'trade_volume', 'vol'])):
                    if col not in available_volume_features:
                        available_volume_features.append(col)
            self.logger.info(f'📊 Found {len(available_price_features)} price features: {available_price_features}')
            self.logger.info(f'📊 Found {len(available_volume_features)} volume features: {available_volume_features}')
            selected_price_feature = None
            selected_volume_feature = None
            if self.primary_price_feature in available_price_features:
                selected_price_feature = self.primary_price_feature
            elif available_price_features:
                selected_price_feature = available_price_features[0]
                self.logger.info(f"🎯 Selected '{selected_price_feature}' as primary price feature (preferred '{self.primary_price_feature}' not available)")
            else:
                self.logger.warning('⚠️ No price features found for conversion')
            if self.primary_volume_feature in available_volume_features:
                selected_volume_feature = self.primary_volume_feature
            elif available_volume_features:
                selected_volume_feature = available_volume_features[0]
                self.logger.info(f"🎯 Selected '{selected_volume_feature}' as primary volume feature (preferred '{self.primary_volume_feature}' not available)")
            else:
                self.logger.warning('⚠️ No volume features found for conversion')
            features_to_remove = []
            for col in converted_df.columns:
                col_lower = col.lower()
                if any((exclude_pattern in col_lower for exclude_pattern in ['regime', 'categorical', 'class', 'label', 'category', 'type'])):
                    continue
                try:
                    unique_count = converted_df[col].nunique()
                    if unique_count <= 5:
                        continue
                except (TypeError, ValueError) as e:
                    self.logger.warning(f'Error checking uniqueness for column {col}: {e}')
                    continue
                if any((price_pattern in col_lower for price_pattern in ['open', 'high', 'low', 'close', 'price', 'avg_price', 'min_price', 'max_price'])):
                    if col != selected_price_feature:
                        features_to_remove.append(col)
                elif any((volume_pattern in col_lower for volume_pattern in ['volume', 'trade_volume', 'vol'])):
                    if col != selected_volume_feature:
                        features_to_remove.append(col)
            if features_to_remove:
                self.logger.info(f'🗑️ Removing {len(features_to_remove)} redundant features: {features_to_remove}')
                converted_df = converted_df.drop(columns=features_to_remove)
            features_to_convert = []
            if selected_price_feature:
                features_to_convert.append(selected_price_feature)
            if selected_volume_feature:
                features_to_convert.append(selected_volume_feature)
            self.logger.info(f'📊 Converting {len(features_to_convert)} selected features to returns: {features_to_convert}')
        else:
            self.logger.info('📊 Using legacy approach - converting all price-related features')
            price_patterns = ['open', 'high', 'low', 'close', 'volume', 'price', 'Price', 'PRICE', 'sma_', 'ema_', 'SMA_', 'EMA_', 'bb_', 'BB_', 'bollinger', 'atr', 'ATR', 'average_true_range', 'vwap', 'VWAP', 'price_', 'Price_', 'PRICE_', 'level_', 'Level_', 'LEVEL_', 'support_', 'resistance_', 'Support_', 'Resistance_', 'ma_', 'MA_', 'moving_average', 'momentum', 'Momentum', 'MOMENTUM', 'change', 'Change', 'CHANGE', 'vwap', 'VWAP', 'volume_weighted', 'cci', 'CCI', 'commodity_channel', 'williams_r', 'Williams_R', 'WILLIAMS_R', 'pattern_', 'Pattern_', 'PATTERN_', 'candlestick_', 'Candlestick_', 'CANDLESTICK_']
            features_to_convert = []
            for col in converted_df.columns:
                if any((exclude_pattern in col.lower() for exclude_pattern in ['regime', 'categorical', 'class', 'label', 'category', 'type'])):
                    continue
                unique_count = converted_df[col].nunique()
                if unique_count <= 5:
                    continue
                if any((pattern in col.lower() for pattern in price_patterns)):
                    if any((skip_pattern in col.lower() for skip_pattern in ['return', 'diff', 'change', 'pct', 'ratio'])):
                        continue
                    features_to_convert.append(col)
        converted_count = 0
        for col in features_to_convert:
            try:
                if col in converted_df.columns:
                    if col.lower() in ['volume_regime', 'volatility_regime', 'trend_regime']:
                        self.logger.warning(f"⚠️ Skipping known regime feature '{col}' to prevent infinite values")
                        continue
                    original_values = converted_df[col].copy()
                    if self.price_return_method == 'pct_change':
                        returns = original_values.pct_change().fillna(0)
                    elif self.price_return_method == 'diff':
                        returns = original_values.diff().fillna(0)
                    elif self.price_return_method == 'log_returns':
                        returns = np.log(original_values / original_values.shift(1)).fillna(0)
                    else:
                        returns = original_values.pct_change().fillna(0)
                    inf_count_before = np.isinf(returns).sum()
                    if inf_count_before > 0:
                        self.logger.warning(f"⚠️ Found {inf_count_before} infinite values in '{col}' returns - replacing with NaN")
                    returns = returns.replace([np.inf, -np.inf], np.nan)
                    returns = returns.fillna(0)
                    max_abs_value = 1000
                    extreme_count_before = (np.abs(returns) > max_abs_value).sum()
                    if extreme_count_before > 0:
                        self.logger.warning(f"⚠️ Found {extreme_count_before} extreme values (>±{max_abs_value}) in '{col}' returns - clipping")
                    returns = np.clip(returns, -max_abs_value, max_abs_value)
                    converted_df[col] = returns
                    converted_count += 1
                    if converted_count <= 5:
                        self.logger.info(f"   📊 Converted '{col}' to returns (method: {self.price_return_method})")
                        self.logger.info(f'      Original range: [{original_values.min():.6f}, {original_values.max():.6f}]')
                        self.logger.info(f'      Returns range: [{returns.min():.6f}, {returns.max():.6f}]')
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to convert price feature '{col}' to returns: {e}")
                continue
        self.logger.info(f'✅ Successfully converted {converted_count} price features to returns')
        self.logger.info(f'📊 Final feature count: {converted_df.shape[1]} columns')
        final_inf_count = np.isinf(converted_df.select_dtypes(include=[np.number])).sum().sum()
        if final_inf_count > 0:
            self.logger.error(f'🚨 CRITICAL: {final_inf_count} infinite values still present after conversion!')
            converted_df = converted_df.replace([np.inf, -np.inf], 0)
        else:
            self.logger.info('✅ Final validation passed: no infinite values detected')
        return converted_df

class FeatureFilter:
    """Random Forest + SHAP feature filtering."""

    def __init__(self, config: AutoencoderConfig) -> None:
        if not DEPENDENCIES_AVAILABLE:
            msg = f'Required dependencies not available: {MISSING_DEPENDENCY}'
            raise ImportError(msg)
        self.config = config
        self.logger = system_logger.getChild('FeatureFilter')

    def filter_features(self, features_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        """Filter features using Random Forest + SHAP importance."""
        try:
            self.logger.info('🔍 Starting feature filtering with Random Forest + SHAP...')
            self.logger.info(f'📊 Input data shape: {features_df.shape}')
            self.logger.info(f'🎯 Number of unique labels: {len(np.unique(labels))}')
            self.logger.info(f'📈 Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}')
            raw_ohlcv_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'time']
            raw_ohlcv_columns = [col for col in raw_ohlcv_columns if col in features_df.columns]
            if raw_ohlcv_columns:
                self.logger.warning(f'🚨 CRITICAL: Found raw OHLCV columns in features: {raw_ohlcv_columns}')
                self.logger.warning('🚨 These should be excluded from feature filtering')
                self.logger.warning('🚨 Raw price data should be processed into engineered features first')
                features_df = features_df.drop(columns=raw_ohlcv_columns)
                self.logger.info(f'✅ Removed {len(raw_ohlcv_columns)} raw OHLCV columns from features')
                self.logger.info(f'📊 Features shape after removal: {features_df.shape}')
                if features_df.empty:
                    self.logger.error('🚨 CRITICAL: No engineered features remaining after removing raw OHLCV data')
                    self.logger.error('🚨 This indicates a serious data pipeline issue')
                    return pd.DataFrame()
            X = features_df.select_dtypes(include=[np.number]).fillna(0)
            y = labels
            if X.empty or X.shape[1] == 0:
                self.logger.warning('⚠️ No numeric features available for filtering')
                self.logger.warning('⚠️ Returning original features without filtering')
                return features_df
            self.logger.info(f'🔢 Numeric features selected: {len(X.columns)}')
            self.logger.info(f'📏 Feature names: {list(X.columns)}')
            if len(np.unique(y)) < 2:
                self.logger.warning('⚠️ Insufficient unique labels for classification, skipping filtering.')
                return features_df
            self.logger.info('🌲 Training Random Forest model for feature importance...')
            n_estimators = self.config.get('feature_filtering.n_estimators', 100)
            max_depth = self.config.get('feature_filtering.max_depth', 10)
            random_state = self.config.get('feature_filtering.random_state', 42)
            self.logger.info(f'🌲 RF Parameters: n_estimators={n_estimators}, max_depth={max_depth}, random_state={random_state}')
            rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
            start_time = time.time()
            rf_model.fit(X, y)
            training_time = time.time() - start_time
            self.logger.info(f'✅ Random Forest training completed in {training_time:.2f} seconds')
            self.logger.info(f'🎯 RF training score: {rf_model.score(X, y):.4f}')
            self.logger.info('🔍 Computing SHAP values with enhanced efficiency...')
            start_time = time.time()
            try:
                from shap.explainers import TreeExplainer
                self.logger.info('📦 Using SHAP TreeExplainer from shap.explainers')
            except ImportError:
                from shap import TreeExplainer
                self.logger.info('📦 Using SHAP TreeExplainer from shap')
            sample_percentage = self.config.get('feature_filtering.sample_percentage', 10.0)
            min_sample_size = self.config.get('feature_filtering.min_sample_size', 1000)
            max_sample_size = self.config.get('feature_filtering.max_sample_size', 1000000)
            sample_size = int(len(X) * sample_percentage / 100)
            sample_size = max(min_sample_size, min(sample_size, max_sample_size))
            self.logger.info(f'📊 Dataset size: {len(X)} rows')
            self.logger.info(f'📊 Sample percentage: {sample_percentage}%')
            self.logger.info(f'📊 Calculated sample size: {sample_size} rows')
            if sample_size < len(X):
                self.logger.info('🔄 Applying stratified sampling to maintain class balance...')
                unique_labels, label_counts = np.unique(y, return_counts=True)
                min_class_count = label_counts.min()
                max_class_count = label_counts.max()
                imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')
                self.logger.info('📊 Label distribution analysis:')
                self.logger.info(f'   Unique labels: {unique_labels}')
                self.logger.info(f'   Label counts: {label_counts}')
                self.logger.info(f'   Min class count: {min_class_count}')
                self.logger.info(f'   Imbalance ratio: {imbalance_ratio:.1f}')
                shap_imbalance_threshold = self.config.get('feature_filtering.shap_imbalance_threshold', 100.0)
                enable_shap_imbalance_handling = self.config.get('feature_filtering.enable_shap_imbalance_handling', True)
                if enable_shap_imbalance_handling and imbalance_ratio > shap_imbalance_threshold:
                    self.logger.warning(f'🚨 CRITICAL FIX: Extreme label imbalance detected (ratio={imbalance_ratio:.1f} > {shap_imbalance_threshold})')
                    self.logger.info('🔄 Using random sampling for SHAP computation...')
                    sample_indices = np.random.choice(len(X), sample_size, replace=False)
                    X_sample = X.iloc[sample_indices]
                    y_sample = y[sample_indices]
                    self.logger.info(f'📊 Random sample size: {len(X_sample)} rows')
                elif min_class_count >= 10:
                    try:
                        from sklearn.model_selection import train_test_split
                        class_sample_sizes = {}
                        for label, count in zip(unique_labels, label_counts):
                            class_sample_size = int(count * sample_percentage / 100)
                            class_sample_size = max(5, min(class_sample_size, count))
                            class_sample_sizes[label] = class_sample_size
                        X_sample, _, y_sample, _ = train_test_split(X, y, train_size=sample_size, stratify=y, random_state=42)
                        original_dist = dict(zip(unique_labels, label_counts))
                        sample_dist = dict(zip(*np.unique(y_sample, return_counts=True)))
                        self.logger.info('✅ Stratified sampling successful!')
                        self.logger.info(f'📊 Original class distribution: {original_dist}')
                        self.logger.info(f'📊 Sample class distribution: {sample_dist}')
                        self.logger.info(f'📊 Sample size: {len(X_sample)} rows ({len(X_sample) / len(X) * 100:.1f}%)')
                    except Exception as e:
                        self.logger.warning(f'⚠️ Stratified sampling failed: {e}')
                        self.logger.info('🔄 Falling back to random sampling...')
                        sample_indices = np.random.choice(len(X), sample_size, replace=False)
                        X_sample = X.iloc[sample_indices]
                        y_sample = y[sample_indices]
                        self.logger.info(f'📊 Random sample size: {len(X_sample)} rows')
                else:
                    self.logger.warning(f'⚠️ Insufficient samples per class for stratification (min: {min_class_count})')
                    self.logger.info('🔄 Using random sampling...')
                    sample_indices = np.random.choice(len(X), sample_size, replace=False)
                    X_sample = X.iloc[sample_indices]
                    y_sample = y[sample_indices]
                    self.logger.info(f'📊 Random sample size: {len(X_sample)} rows')
            else:
                X_sample = X
                y_sample = y
                self.logger.info('📊 Using full dataset (no sampling needed)')
            enable_prefiltering = self.config.get('feature_filtering.enable_feature_prefiltering', True)
            max_features_for_shap = self.config.get('feature_filtering.max_features_for_shap', 50)
            if enable_prefiltering and len(X_sample.columns) > max_features_for_shap:
                self.logger.info(f'📊 High feature count ({len(X_sample.columns)}), applying pre-filtering')
                self.logger.info(f'📊 Target feature count: {max_features_for_shap}')
                pre_filter_rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
                pre_filter_rf.fit(X_sample, y_sample)
                feature_importance = pre_filter_rf.feature_importances_
                top_feature_indices = np.argsort(feature_importance)[-max_features_for_shap:]
                X_sample = X_sample.iloc[:, top_feature_indices]
                self.logger.info(f'📊 Pre-filtered to top {len(X_sample.columns)} features')
                self.logger.info(f'📊 Selected features: {list(X_sample.columns)}')
            else:
                self.logger.info(f'📊 No pre-filtering needed (features: {len(X_sample.columns)}, max: {max_features_for_shap})')
            shap_n_estimators = self.config.get('feature_filtering.shap_n_estimators', 50)
            shap_max_depth = self.config.get('feature_filtering.shap_max_depth', 8)
            shap_min_samples_split = self.config.get('feature_filtering.shap_min_samples_split', 10)
            shap_min_samples_leaf = self.config.get('feature_filtering.shap_min_samples_leaf', 5)
            shap_rf_model = RandomForestClassifier(n_estimators=shap_n_estimators, max_depth=shap_max_depth, min_samples_split=shap_min_samples_split, min_samples_leaf=shap_min_samples_leaf, random_state=42, n_jobs=-1)
            self.logger.info(f'🌲 SHAP RF Parameters: n_estimators={shap_n_estimators}, max_depth={shap_max_depth}')
            self.logger.info('🌲 Training optimized Random Forest for SHAP...')
            shap_rf_model.fit(X_sample, y_sample)
            self.logger.info(f'✅ Optimized RF training completed (score: {shap_rf_model.score(X_sample, y_sample):.4f})')
            explainer = TreeExplainer(shap_rf_model, feature_names=X_sample.columns.tolist(), model_output='raw')
            self.logger.info('🔧 Optimized SHAP explainer created successfully')
            import signal
            import platform
            base_timeout_per_5000 = self.config.get('feature_filtering.timeout_per_5000_samples', 60)
            calculated_timeout = int(len(X_sample) / 5000 * base_timeout_per_5000)
            timeout_seconds = max(30, min(900, calculated_timeout))
            self.logger.info('⏱️ Flexible timeout calculation:')
            self.logger.info(f'   📊 Sample size: {len(X_sample)} rows')
            self.logger.info(f'   📊 Base rate: {base_timeout_per_5000}s per 5000 samples')
            self.logger.info(f'   📊 Calculated timeout: {calculated_timeout}s')
            self.logger.info(f'   ⏱️ Final timeout: {timeout_seconds}s (bounded: 30s-900s)')
            if platform.system() != 'Windows':

                def timeout_handler(signum: Any, frame: Any) -> None:
                    raise TimeoutError('SHAP computation timed out')
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)
            else:
                self.logger.info('⚠️ Windows detected - using simplified timeout protection')
            try:
                background_size = min(100, len(X_sample) // 10)
                background_indices = np.random.choice(len(X_sample), background_size, replace=False)
                background_values = X_sample.iloc[background_indices]
                self.logger.info(f'📊 Computing SHAP values with background set of {len(background_values)} samples...')
                shap_values = explainer.shap_values(X_sample)
                self.logger.info('✅ SHAP values computed successfully with optimizations')
                if platform.system() != 'Windows':
                    signal.alarm(0)
                shap_time = time.time() - start_time
                self.logger.info(f'✅ SHAP values computed in {shap_time:.2f} seconds')
            except TimeoutError:
                if platform.system() != 'Windows':
                    signal.alarm(0)
                self.logger.warning('⏰ SHAP computation timed out, falling back to Random Forest feature importance')
                feature_importance = rf_model.feature_importances_
                shap_time = time.time() - start_time
                self.logger.info(f'✅ Fallback feature importance computed in {shap_time:.2f} seconds')
                sorted_indices = np.argsort(feature_importance)[::-1]
                sorted_importance = feature_importance[sorted_indices]
                cumulative_importance = np.cumsum(sorted_importance)
                total_importance = cumulative_importance[-1]
                self.logger.info(f'📊 Total importance: {total_importance:.6f}')
                self.logger.info('🏆 Top 5 most important features:')
                for i in range(min(5, len(sorted_indices))):
                    feature_name = X.columns[sorted_indices[i]]
                    importance = sorted_importance[i]
                    cumulative = cumulative_importance[i]
                    self.logger.info(f'   {i + 1}. {feature_name}: {importance:.6f} (cumulative: {cumulative:.6f})')
                threshold = self.config.get('feature_filtering.importance_threshold', 0.99)
                importance_cutoff = threshold * total_importance
                self.logger.info(f'🎯 Importance threshold: {threshold} ({threshold * 100}%)')
                self.logger.info(f'📊 Importance cutoff: {importance_cutoff:.6f}')
                cutoff_index = np.where(cumulative_importance >= importance_cutoff)[0][0] + 1
                selected_indices = sorted_indices[:cutoff_index]
                self.logger.info(f'📊 Features needed to reach threshold: {cutoff_index}')
                self.logger.info(f'📊 Cumulative importance at cutoff: {cumulative_importance[cutoff_index - 1]:.6f}')
                min_features = self.config.get('feature_filtering.min_features_for_ae', 15)
                self.logger.info(f'🔒 Minimum features required: {min_features}')
                if len(selected_indices) < min_features:
                    self.logger.info(f'⚠️ Selected features ({len(selected_indices)}) below minimum ({min_features}), expanding selection')
                    selected_indices = sorted_indices[:max(min_features, len(sorted_indices))]
                    self.logger.info(f'📊 Expanded selection to {len(selected_indices)} features')
                selected_features = X.columns.to_numpy()[selected_indices].tolist()
                self.logger.info(f'✅ Selected {len(selected_features)} features out of {len(X.columns)} using fallback method.')
                self.logger.info(f'📊 Selected features: {selected_features}')
                shap_refine_min = self.config.get('feature_filtering.min_features_for_shap', 20)
                self.logger.info(f'🔒 SHAP refinement minimum: {shap_refine_min}')
                final_features = selected_features
                if len(selected_features) > shap_refine_min:
                    k = max(min_features, len(selected_features))
                    final_features = selected_features[:k]
                    self.logger.info(f'📊 Refined selection to {len(final_features)} features')
                self.logger.info('🎉 Feature filtering completed successfully with fallback method!')
                self.logger.info(f'📊 Final feature count: {len(final_features)}')
                self.logger.info(f'📊 Final features: {final_features}')
                return features_df[final_features].copy()
            self.logger.info('📊 Computing feature importance from SHAP values...')
            start_time = time.time()
            if hasattr(shap_values, 'values'):
                self.logger.info('📦 SHAP values format: shap_values.values')
                shap_arr = np.asarray(shap_values.values)
            elif isinstance(shap_values, list):
                self.logger.info(f'📦 SHAP values format: list of {len(shap_values)} arrays')
                shap_arr = np.stack([np.asarray(sv) for sv in shap_values], axis=0)
            else:
                self.logger.info('📦 SHAP values format: numpy array')
                shap_arr = np.asarray(shap_values)
            self.logger.info(f'📐 SHAP array shape: {shap_arr.shape}')
            if shap_arr.ndim == 2:
                self.logger.info('🔄 Adding class dimension to SHAP array')
                shap_arr = shap_arr[None, ...]
            elif shap_arr.ndim == 1:
                self.logger.info('🔄 Reshaping SHAP array for single feature')
                shap_arr = shap_arr[None, :, None]
            self.logger.info('📊 Computing mean absolute SHAP importance per feature...')
            feature_importance = np.nanmean(np.abs(shap_arr), axis=(0, 1))
            feature_importance = np.nan_to_num(feature_importance, nan=0.0, posinf=0.0, neginf=0.0)
            importance_time = time.time() - start_time
            self.logger.info(f'✅ Feature importance computed in {importance_time:.2f} seconds')
            self.logger.info('📈 Sorting features by importance...')
            sorted_indices = np.argsort(feature_importance)[::-1]
            sorted_importance = feature_importance[sorted_indices]
            cumulative_importance = np.cumsum(sorted_importance)
            total_importance = cumulative_importance[-1]
            self.logger.info(f'📊 Total importance: {total_importance:.6f}')
            self.logger.info('🏆 Top 5 most important features:')
            for i in range(min(5, len(sorted_indices))):
                feature_name = X.columns[sorted_indices[i]]
                importance = sorted_importance[i]
                cumulative = cumulative_importance[i]
                self.logger.info(f'   {i + 1}. {feature_name}: {importance:.6f} (cumulative: {cumulative:.6f})')
            threshold = self.config.get('feature_filtering.importance_threshold', 0.95)
            min_features = self.config.get('feature_filtering.min_features_to_keep', 5)
            max_features = self.config.get('feature_filtering.max_features_to_keep', 50)
            min_importance_per_feature = self.config.get('feature_filtering.min_importance_per_feature', 0.001)
            importance_cutoff = threshold * total_importance
            self.logger.info('🎯 Feature selection parameters:')
            self.logger.info(f'   📊 Importance threshold: {threshold:.3f} ({threshold * 100:.1f}%)')
            self.logger.info(f'   📊 Min features to keep: {min_features}')
            self.logger.info(f'   📊 Max features to keep: {max_features}')
            self.logger.info(f'   📊 Min importance per feature: {min_importance_per_feature:.6f}')
            self.logger.info(f'   📊 Importance cutoff: {importance_cutoff:.6f}')
            threshold_cutoff = np.where(cumulative_importance >= importance_cutoff)[0]
            threshold_cutoff = threshold_cutoff[0] + 1 if len(threshold_cutoff) > 0 else len(sorted_indices)
            min_importance_cutoff = np.where(sorted_importance >= min_importance_per_feature)[0]
            min_importance_cutoff = len(min_importance_cutoff) if len(min_importance_cutoff) > 0 else 0
            cutoff_index = max(threshold_cutoff, min_importance_cutoff, min_features)
            selected_indices = sorted_indices[:cutoff_index]
            self.logger.info('📊 Enhanced selection analysis:')
            self.logger.info(f'   📊 Threshold cutoff: {threshold_cutoff} features')
            self.logger.info(f'   📊 Min importance cutoff: {min_importance_cutoff} features')
            self.logger.info(f'   📊 Min features requirement: {min_features} features')
            self.logger.info(f'   📊 Final cutoff: {cutoff_index} features')
            self.logger.info('📊 Initial selection results:')
            self.logger.info(f'   📊 Features selected: {cutoff_index}')
            actual_cutoff = min(cutoff_index, len(cumulative_importance))
            self.logger.info(f'   📊 Cumulative importance at cutoff: {cumulative_importance[actual_cutoff - 1]:.6f}')
            self.logger.info(f'   📊 Actual importance captured: {cumulative_importance[actual_cutoff - 1] / total_importance * 100:.1f}%')
            if len(selected_indices) < min_features:
                self.logger.warning(f'⚠️ Selected features ({len(selected_indices)}) below minimum ({min_features})')
                self.logger.info('🔄 Expanding selection to meet minimum requirement...')
                actual_min_features = min(min_features, len(sorted_indices))
                selected_indices = sorted_indices[:actual_min_features]
                actual_importance = cumulative_importance[actual_min_features - 1] if len(sorted_importance) >= actual_min_features else cumulative_importance[-1]
                self.logger.info(f'📊 Expanded to {len(selected_indices)} features (importance: {actual_importance / total_importance * 100:.1f}%)')
                if len(selected_indices) < min_features and hasattr(self, '_prefiltered_features'):
                    self.logger.warning('⚠️ Still below minimum after expansion. This may indicate insufficient important features in the dataset.')
            if len(selected_indices) > max_features:
                self.logger.warning(f'⚠️ Selected features ({len(selected_indices)}) above maximum ({max_features})')
                self.logger.info('🔄 Truncating selection to meet maximum requirement...')
                selected_indices = sorted_indices[:max_features]
                actual_importance = cumulative_importance[max_features - 1] if len(sorted_importance) >= max_features else cumulative_importance[-1]
                self.logger.info(f'📊 Truncated to {len(selected_indices)} features (importance: {actual_importance / total_importance * 100:.1f}%)')
            selected_features = X.columns.to_numpy()[selected_indices].tolist()
            self.logger.info('✅ Final feature selection:')
            self.logger.info(f'   📊 Features selected: {len(selected_features)} out of {len(X.columns)}')
            self.logger.info(f'   📊 Importance captured: {cumulative_importance[len(selected_indices) - 1] / total_importance * 100:.1f}%')
            self.logger.info(f'   📊 Selected features: {selected_features}')
            self.logger.info('🎉 Feature filtering completed successfully!')
            self.logger.info(f'📊 Final feature count: {len(selected_features)}')
            self.logger.info(f'📊 Final features: {selected_features}')
            return features_df[selected_features].copy()
        except Exception:
            self.logger.exception('Error in feature filtering')
            return features_df

class ImprovedAutoencoderPreprocessor:
    """Enhanced preprocessor with separate fit/transform and no data leakage."""

    def __init__(self, config: AutoencoderConfig) -> None:
        if not DEPENDENCIES_AVAILABLE:
            msg = f'Required dependencies not available: {MISSING_DEPENDENCY}'
            raise ImportError(msg)
        self.config = config
        scaler_type = config.get('preprocessing.scaler_type', 'robust')
        if scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        self.outlier_lower_bounds_ = None
        self.outlier_upper_bounds_ = None
        self.is_fitted = False
        self.logger = system_logger.getChild('AutoencoderPreprocessor')

    def fit(self, X: pd.DataFrame) -> 'ImprovedAutoencoderPreprocessor':
        """Fit the preprocessor on training data only."""
        self.logger.info(f'🔧 Fitting preprocessor on data with shape {X.shape}')
        self.logger.info('📊 Handling missing values...')
        X_clean = self._handle_missing_values(X)
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            self.logger.info(f'📊 Missing values handled: {missing_count} values filled')
        X_numeric = X_clean.select_dtypes(include=[np.number])
        self.logger.info(f'📊 Numeric features for outlier detection: {X_numeric.shape[1]} columns')
        Q1 = X_numeric.quantile(0.25)
        Q3 = X_numeric.quantile(0.75)
        IQR = Q3 - Q1
        iqr_mult = self.config.get('preprocessing.iqr_multiplier', 3.0)
        self.outlier_lower_bounds_ = Q1 - iqr_mult * IQR
        self.outlier_upper_bounds_ = Q3 + iqr_mult * IQR
        self.logger.info(f'📊 Outlier detection: IQR method with multiplier {iqr_mult}')
        self.logger.info(f'📊 Outlier bounds calculated for {len(self.outlier_lower_bounds_)} features')
        lower_bounds = self.outlier_lower_bounds_.reindex(X_numeric.columns)
        upper_bounds = self.outlier_upper_bounds_.reindex(X_numeric.columns)
        X_clipped = X_numeric.clip(lower=lower_bounds, upper=upper_bounds, axis=1)
        outliers_clipped = ((X_numeric < lower_bounds) | (X_numeric > upper_bounds)).sum().sum()
        if outliers_clipped > 0:
            self.logger.info(f'📊 Outliers clipped: {outliers_clipped} values')
        scaler_type = self.config.get('preprocessing.scaler_type', 'robust')
        self.logger.info(f'📊 Fitting {scaler_type} scaler...')
        self.scaler.fit(X_clipped.values)
        self.is_fitted = True
        self.logger.info('✅ Preprocessor fitted successfully')
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessor."""
        if not self.is_fitted:
            msg = 'Preprocessor must be fitted before transform can be called.'
            raise ValueError(msg)
        self.logger.info(f'🔧 Transforming data with shape {X.shape}')
        X_clean = self._handle_missing_values(X)
        X_clipped = self._clip_outliers(X_clean)
        self.logger.info('📊 Scaling data...')
        X_scaled = self.scaler.transform(X_clipped.values)
        final_threshold = self.config.get('preprocessing.outlier_threshold', 3.0)
        X_final = np.clip(X_scaled, -final_threshold, final_threshold)
        extreme_values_clipped = ((X_scaled < -final_threshold) | (X_scaled > final_threshold)).sum()
        if extreme_values_clipped > 0:
            self.logger.info(f'📊 Extreme values clipped: {extreme_values_clipped} values (threshold: ±{final_threshold})')
        try:
            self.logger.info('✅ Transform completed successfully')
            self.logger.info(f'📊 Input shape: {X.shape}')
            self.logger.info(f'📊 Output shape: {X_final.shape}')
            self.logger.info(f'📊 Final clipping threshold: ±{final_threshold}')
        except Exception as e:
            self.logger.warning(f'⚠️ Could not log transform details: {str(e)}')
        return X_final

    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on strategy."""
        strategy = self.config.get('preprocessing.missing_value_strategy', 'forward_fill')
        if strategy == 'forward_fill':
            return X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        return X.fillna(0)

    def _clip_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers using pre-calculated bounds to prevent data leakage."""
        X_numeric = X.select_dtypes(include=[np.number])
        lower_bounds = self.outlier_lower_bounds_.reindex(X_numeric.columns)
        upper_bounds = self.outlier_upper_bounds_.reindex(X_numeric.columns)
        return X_numeric.clip(lower=lower_bounds, upper=upper_bounds, axis=1)

def create_sequences_with_index(X: np.ndarray, timesteps: int, original_index: pd.Index) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Convert 2D array to 3D sequences, tracking the index of the target."""
    sequences, targets, target_indices = ([], [], [])
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    total_samples = len(X)
    num_sequences = total_samples - timesteps + 1
    overlap_percentage = (timesteps - 1) / timesteps * 100
    logger = logging.getLogger(__name__)
    logger.info(f'📊 Creating sequences from {total_samples} samples')
    logger.info(f'📊 Sequence configuration: {timesteps} timesteps, {num_sequences} sequences')
    logger.info(f'📊 Overlap: {overlap_percentage:.1f}% between consecutive sequences')
    logger.info(f'📊 Input shape: {X.shape} -> Output shapes: ({num_sequences}, {timesteps}, {X.shape[1]})')
    for i in range(num_sequences):
        sequence = X[i:i + timesteps]
        target = X[i + timesteps - 1]
        sequences.append(sequence)
        targets.append(target)
        target_indices.append(original_index[i + timesteps - 1])
    sequences_array = np.array(sequences)
    targets_array = np.array(targets)
    target_indices_array = pd.Index(target_indices)
    logger.info('✅ Sequence creation completed')
    logger.info(f'📊 Sequences shape: {sequences_array.shape}')
    logger.info(f'📊 Targets shape: {targets_array.shape}')
    logger.info(f'📊 Target indices: {len(target_indices_array)} samples')
    return (sequences_array, targets_array, target_indices_array)

class SequenceAwareAutoencoder:
    """1D-CNN based autoencoder that learns to reconstruct the last timestep of a sequence."""

    def __init__(self, config: AutoencoderConfig) -> None:
        if not DEPENDENCIES_AVAILABLE:
            msg = f'Required dependencies not available: {MISSING_DEPENDENCY}'
            raise ImportError(msg)
        self.config = config
        self.logger = system_logger.getChild('SequenceAwareAutoencoder')
        self.autoencoder = None
        self.encoder = None

    def build_model(self, input_shape: tuple[int, int], trial: optuna.Trial | None=None) -> Model:
        """Build 1D-CNN autoencoder model."""
        timesteps, features = input_shape
        if trial:
            filters = trial.suggest_categorical('filters', [16, 32, 64])
            kernel_size = trial.suggest_int('kernel_size', 3, 7)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
            encoding_dim = trial.suggest_int('encoding_dim', 8, 64)
        else:
            encoding_dim = self.config.get('autoencoder.encoding_dim', 32)
            best_params = self.config.get('best_params', {})
            filters = best_params.get('filters', 32)
            kernel_size = best_params.get('kernel_size', 5)
            dropout_rate = best_params.get('dropout_rate', 0.3)
            learning_rate = best_params.get('learning_rate', 0.001)
        self.logger.info('🔧 Building autoencoder model architecture...')
        self.logger.info(f'📊 Input shape: (timesteps={timesteps}, features={features})')
        self.logger.info('📊 Model hyperparameters:')
        self.logger.info(f'   📊 Filters: {filters}')
        self.logger.info(f'   📊 Kernel size: {kernel_size}')
        self.logger.info(f'   📊 Dropout rate: {dropout_rate}')
        self.logger.info(f'   📊 Encoding dimension: {encoding_dim}')
        self.logger.info(f'   📊 Learning rate: {learning_rate}')
        input_layer = layers.Input(shape=(timesteps, features))
        x = layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Conv1D(filters=filters // 2, kernel_size=kernel_size, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        bottleneck = layers.Dense(encoding_dim, activation='tanh', name='bottleneck')(x)
        output_layer = layers.Dense(features, activation='linear')(bottleneck)
        self.autoencoder = Model(inputs=input_layer, outputs=output_layer)
        self.encoder = Model(inputs=input_layer, outputs=bottleneck)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.autoencoder.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
        try:
            total_params = int(np.sum([np.prod(v.shape) for v in self.autoencoder.trainable_weights]))
            self.logger.info('✅ Model compiled successfully!')
            self.logger.info(f'📊 Optimizer: Adam(learning_rate={learning_rate})')
            self.logger.info('📊 Loss function: Huber')
            self.logger.info('📊 Metrics: MAE')
            self.logger.info(f'📊 Total trainable parameters: {total_params:,}')
            if total_params < 10000:
                complexity = 'Lightweight'
            elif total_params < 100000:
                complexity = 'Moderate'
            elif total_params < 1000000:
                complexity = 'Complex'
            else:
                complexity = 'Very Complex'
            self.logger.info(f'📊 Model complexity: {complexity}')
        except Exception as e:
            self.logger.warning(f'⚠️ Could not calculate model parameters: {str(e)}')
        return self.autoencoder

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, trial: optuna.Trial | None=None) -> Any:
        """Train the autoencoder with enhanced logging."""
        early_stopping_patience = self.config.get('autoencoder.early_stopping_patience', 10)
        reduce_lr_patience = self.config.get('autoencoder.reduce_lr_patience', 5)
        min_lr = self.config.get('autoencoder.min_lr', 1e-06)
        callbacks = [EarlyStopping(monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True), ReduceLROnPlateau(monitor='val_loss', patience=reduce_lr_patience, min_lr=min_lr)]
        if trial and self.config.get('training.pruning_enabled', True):
            callbacks.append(TFKerasPruningCallback(trial, 'val_loss'))
            self.logger.info('📊 Optuna pruning callback enabled')
        if trial:
            batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
            self.logger.info(f'📊 Trial batch size: {batch_size}')
        else:
            batch_size = self.config.get('best_paramsf', {}).get('batch_size', 32)
            self.logger.info(f'📊 Final training batch size: {batch_size}')
        epochs = self.config.get('autoencoder.epochs', 100)
        self.logger.info('🚀 Starting autoencoder training...')
        self.logger.info(f'📊 Training data: {X_train.shape[0]} sequences, {X_train.shape[1]} timesteps, {X_train.shape[2]} features')
        self.logger.info(f'📊 Validation data: {X_val.shape[0]} sequences, {X_val.shape[1]} timesteps, {X_val.shape[2]} features')
        self.logger.info(f'📊 Training configuration: epochs={epochs}, batch_size={batch_size}')
        self.logger.info(f'📊 Callbacks: EarlyStopping(patience={early_stopping_patience}), ReduceLROnPlateau(patience={reduce_lr_patience})')
        start_time = time.time()
        history = self.autoencoder.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
        training_time = time.time() - start_time
        try:
            val_losses = history.history.get('val_loss', [])
            train_losses = history.history.get('loss', [])
            val_mae = history.history.get('val_mae', [])
            history.history.get('mae', [])
            if val_losses:
                best_epoch = int(np.argmin(val_losses))
                best_val_loss = val_losses[best_epoch]
                final_train_loss = train_losses[-1] if train_losses else 0
                final_val_loss = val_losses[-1]
                self.logger.info('✅ Autoencoder training completed successfully!')
                self.logger.info(f'📊 Training time: {training_time:.2f} seconds')
                self.logger.info(f'📊 Epochs trained: {len(val_losses)}')
                self.logger.info(f'📊 Best epoch: {best_epoch + 1}')
                self.logger.info(f'📊 Best validation loss: {best_val_loss:.6f}')
                self.logger.info(f'📊 Final training loss: {final_train_loss:.6f}')
                self.logger.info(f'📊 Final validation loss: {final_val_loss:.6f}')
                if val_mae:
                    best_val_mae = val_mae[best_epoch]
                    final_val_mae = val_mae[-1]
                    self.logger.info(f'📊 Best validation MAE: {best_val_mae:.6f}')
                    self.logger.info(f'📊 Final validation MAE: {final_val_mae:.6f}')
                if best_val_loss < 0.1:
                    performance = 'Excellent'
                elif best_val_loss < 0.3:
                    performance = 'Good'
                elif best_val_loss < 0.5:
                    performance = 'Acceptable'
                else:
                    performance = 'Needs improvement'
                self.logger.info(f'📊 Model performance: {performance}')
        except Exception as e:
            self.logger.warning(f'⚠️ Could not extract detailed training metrics: {str(e)}')
        return history

class AutoencoderFeatureAnalyzer:
    """Comprehensive feature importance analysis for autoencoder-generated features."""

    def __init__(self, config: AutoencoderConfig) -> None:
        if not DEPENDENCIES_AVAILABLE:
            msg = f'Required dependencies not available: {MISSING_DEPENDENCY}'
            raise ImportError(msg)
        self.config = config
        self.logger = system_logger.getChild('AutoencoderFeatureAnalyzer')
        self.importance_scores = {}
        self.correlation_analysis = {}
        self.stability_metrics = {}
        self.regime_analysis = {}

    def analyze_feature_importance(self, encoded_features: pd.DataFrame, labels: np.ndarray, original_features: pd.DataFrame | None=None, regime_labels: np.ndarray | None=None) -> dict[str, Any]:
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
            self.logger.info('🔍 Starting comprehensive autoencoder feature importance analysis...')
            self.logger.info(f'📊 Encoded features shape: {encoded_features.shape}')
            self.logger.info(f'🎯 Labels shape: {labels.shape}')
            self.logger.info(f'📈 Unique labels: {len(np.unique(labels))}')
            analysis_results = {'feature_importance': {}, 'correlation_analysis': {}, 'stability_metrics': {}, 'regime_analysis': {}, 'summary_statistics': {}, 'recommendations': []}
            self.logger.info('📊 Performing statistical correlation analysis...')
            correlation_results = self._analyze_correlations(encoded_features, labels)
            analysis_results['correlation_analysis'] = correlation_results
            self.logger.info('🤖 Computing ML-based feature importance...')
            ml_importance = self._compute_ml_importance(encoded_features, labels)
            analysis_results['feature_importance'] = ml_importance
            self.logger.info('📈 Analyzing feature stability...')
            stability_results = self._analyze_feature_stability(encoded_features)
            analysis_results['stability_metrics'] = stability_results
            if regime_labels is not None:
                self.logger.info('🔄 Performing regime-specific analysis...')
                regime_results = self._analyze_regime_specific_importance(encoded_features, labels, regime_labels)
                analysis_results['regime_analysis'] = regime_results
            if original_features is not None:
                self.logger.info('🔄 Comparing with original features...')
                comparison_results = self._compare_with_original_features(encoded_features, original_features, labels)
                analysis_results['original_comparison'] = comparison_results
            self.logger.info('📋 Generating summary and recommendations...')
            summary, recommendations = self._generate_summary_and_recommendations(analysis_results)
            analysis_results['summary_statistics'] = summary
            analysis_results['recommendations'] = recommendations
            self.importance_scores = ml_importance
            self.correlation_analysis = correlation_results
            self.stability_metrics = stability_results
            if regime_labels is not None:
                self.regime_analysis = analysis_results['regime_analysis']
            self.logger.info('✅ Autoencoder feature importance analysis completed successfully!')
            return analysis_results
        except Exception as e:
            self.logger.exception(f'❌ Error in feature importance analysis: {e}')
            return {'error': str(e)}

    def _analyze_correlations(self, encoded_features: pd.DataFrame, labels: np.ndarray) -> dict[str, Any]:
        """Analyze statistical correlations between features and labels."""
        try:
            analysis_df = encoded_features.copy()
            analysis_df['target'] = labels
            correlations = analysis_df.corr()['target'].drop('target')
            abs_correlations = correlations.abs().sort_values(ascending=False)
            try:
                from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
                unique_labels = len(np.unique(labels))
                if unique_labels <= 10:
                    mi_scores = mutual_info_classif(encoded_features, labels, random_state=42)
                else:
                    mi_scores = mutual_info_regression(encoded_features, labels, random_state=42)
                mi_df = pd.DataFrame({'feature': encoded_features.columns, 'mutual_info': mi_scores}).sort_values('mutual_info', ascending=False)
                self.logger.info(f'📊 Mutual information computed for {len(encoded_features.columns)} features')
            except ImportError:
                self.logger.warning('⚠️ scikit-learn not available, skipping mutual information')
                mi_df = None
            high_corr_threshold = self.config.get('feature_analysis.high_correlation_threshold', 0.7)
            high_correlations = correlations[correlations.abs() > high_corr_threshold]
            low_corr_threshold = self.config.get('feature_analysis.low_correlation_threshold', 0.1)
            low_correlations = correlations[correlations.abs() < low_corr_threshold]
            results = {'pearson_correlations': correlations.to_dict(), 'abs_correlations': abs_correlations.to_dict(), 'mutual_information': mi_df.to_dict('records') if mi_df is not None else None, 'high_correlations': high_correlations.to_dict(), 'low_correlations': low_correlations.to_dict(), 'correlation_summary': {'mean_correlation': correlations.mean(), 'std_correlation': correlations.std(), 'max_correlation': correlations.max(), 'min_correlation': correlations.min(), 'high_corr_count': len(high_correlations), 'low_corr_count': len(low_correlations)}}
            self.logger.info('📊 Correlation analysis complete:')
            self.logger.info(f"   📈 Mean correlation: {results['correlation_summary']['mean_correlation']:.4f}")
            self.logger.info(f"   📈 Max correlation: {results['correlation_summary']['max_correlation']:.4f}")
            self.logger.info(f"   📈 High correlation features: {results['correlation_summary']['high_corr_count']}")
            self.logger.info(f"   📈 Low correlation features: {results['correlation_summary']['low_corr_count']}")
            return results
        except Exception as e:
            self.logger.exception(f'❌ Error in correlation analysis: {e}')
            return {'error': str(e)}

    def _compute_ml_importance(self, encoded_features: pd.DataFrame, labels: np.ndarray) -> dict[str, Any]:
        """Compute machine learning-based feature importance."""
        try:
            X = encoded_features.select_dtypes(include=[np.number]).fillna(0)
            y = labels
            if len(np.unique(y)) < 2:
                self.logger.warning('⚠️ Insufficient unique labels for ML importance analysis')
                return {'error': 'Insufficient unique labels'}
            self.logger.info('🌲 Computing Random Forest feature importance...')
            rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rf_model.fit(X, y)
            rf_importance = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_}).sort_values('importance', ascending=False)
            try:
                from sklearn.ensemble import GradientBoostingClassifier
                gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
                gb_model.fit(X, y)
                gb_importance = pd.DataFrame({'feature': X.columns, 'importance': gb_model.feature_importances_}).sort_values('importance', ascending=False)
                self.logger.info('🌳 Gradient Boosting importance computed')
            except Exception as e:
                pass
            except ImportError:
                self.logger.warning('⚠️ Gradient Boosting not available')
                gb_importance = None
            try:
                from sklearn.inspection import permutation_importance
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if len(np.unique(y)) <= 10 else None)
                from sklearn.linear_model import LogisticRegression
                perm_model = LogisticRegression(random_state=42, max_iter=1000)
                perm_model.fit(X_train, y_train)
                perm_importance = permutation_importance(perm_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
                perm_df = pd.DataFrame({'feature': X.columns, 'importance': perm_importance.importances_mean, 'std': perm_importance.importances_std}).sort_values('importance', ascending=False)
                self.logger.info('🔄 Permutation importance computed')
            except ImportError:
                self.logger.warning('⚠️ Permutation importance not available')
                perm_df = None
            importance_methods = {'random_forest': rf_importance, 'gradient_boosting': gb_importance, 'permutation': perm_df}
            available_methods = {k: v for k, v in importance_methods.items() if v is not None}
            if len(available_methods) > 1:
                normalized_importance = {}
                for method_name, method_df in available_methods.items():
                    if method_df is not None:
                        normalized_importance[method_name] = method_df['importance'] / method_df['importance'].max()
                ensemble_scores = pd.DataFrame(normalized_importance).mean(axis=1)
                ensemble_df = pd.DataFrame({'feature': X.columns, 'ensemble_importance': ensemble_scores}).sort_values('ensemble_importance', ascending=False)
                self.logger.info('🎯 Ensemble importance computed from multiple methods')
            else:
                ensemble_df = rf_importance.copy()
                ensemble_df.columns = ['feature', 'ensemble_importance']
            results = {'random_forest': rf_importance.to_dict('records'), 'gradient_boosting': gb_importance.to_dict('records') if gb_importance is not None else None, 'permutation': perm_df.to_dict('records') if perm_df is not None else None, 'ensemble': ensemble_df.to_dict('records'), 'importance_summary': {'top_features': ensemble_df.head(10)['feature'].tolist(), 'bottom_features': ensemble_df.tail(10)['feature'].tolist(), 'mean_importance': ensemble_df['ensemble_importance'].mean(), 'std_importance': ensemble_df['ensemble_importance'].std()}}
            self.logger.info('🤖 ML importance analysis complete:')
            self.logger.info(f"   🏆 Top 5 features: {results['importance_summary']['top_features'][:5]}")
            self.logger.info(f"   📊 Mean importance: {results['importance_summary']['mean_importance']:.4f}")
            return results
        except Exception as e:
            self.logger.exception(f'❌ Error in ML importance analysis: {e}')
            return {'error': str(e)}

    def _analyze_feature_stability(self, encoded_features: pd.DataFrame) -> dict[str, Any]:
        """Analyze feature stability over time."""
        try:
            window_size = self.config.get('feature_analysis.stability_window', 100)
            stability_metrics = {}
            for column in encoded_features.columns:
                if column in ['autoencoder_recon_error']:
                    continue
                feature_data = encoded_features[column].dropna()
                if len(feature_data) < window_size * 2:
                    continue
                rolling_mean = feature_data.rolling(window=window_size, min_periods=window_size // 2).mean()
                rolling_std = feature_data.rolling(window=window_size, min_periods=window_size // 2).std()
                mean_stability = 1 - (rolling_std / (rolling_mean.abs() + 1e-08)).mean()
                trend_stability = 1 - abs(rolling_mean.diff().mean()) / (feature_data.std() + 1e-08)
                cv = feature_data.std() / (feature_data.mean() + 1e-08)
                stability_metrics[column] = {'mean_stability': mean_stability, 'trend_stability': trend_stability, 'coefficient_of_variation': cv, 'overall_stability': (mean_stability + trend_stability) / 2}
            stability_df = pd.DataFrame.from_dict(stability_metrics, orient='index')
            stability_df = stability_df.sort_values('overall_stability', ascending=False)
            stability_threshold = self.config.get('feature_analysis.stability_threshold', 0.7)
            stable_features = stability_df[stability_df['overall_stability'] > stability_threshold].index.tolist()
            unstable_features = stability_df[stability_df['overall_stability'] < 1 - stability_threshold].index.tolist()
            results = {'stability_metrics': stability_df.to_dict('index'), 'stable_features': stable_features, 'unstable_features': unstable_features, 'stability_summary': {'mean_stability': stability_df['overall_stability'].mean(), 'stable_count': len(stable_features), 'unstable_count': len(unstable_features), 'stability_threshold': stability_threshold}}
            self.logger.info('📈 Stability analysis complete:')
            self.logger.info(f"   📊 Mean stability: {results['stability_summary']['mean_stability']:.4f}")
            self.logger.info(f"   📊 Stable features: {results['stability_summary']['stable_count']}")
            self.logger.info(f"   📊 Unstable features: {results['stability_summary']['unstable_count']}")
            return results
        except Exception as e:
            self.logger.exception(f'❌ Error in stability analysis: {e}')
            return {'error': str(e)}

    def _analyze_regime_specific_importance(self, encoded_features: pd.DataFrame, labels: np.ndarray, regime_labels: np.ndarray) -> dict[str, Any]:
        """Analyze feature importance across different market regimes."""
        try:
            unique_regimes = np.unique(regime_labels)
            self.logger.info(f'🔄 Analyzing feature importance across {len(unique_regimes)} regimes: {unique_regimes}')
            regime_importance = {}
            for regime in unique_regimes:
                regime_mask = regime_labels == regime
                regime_features = encoded_features[regime_mask]
                regime_labels_subset = labels[regime_mask]
                if len(regime_features) < 50:
                    self.logger.warning(f'⚠️ Regime {regime} has insufficient samples ({len(regime_features)})')
                    continue
                regime_importance[regime] = self._compute_ml_importance(regime_features, regime_labels_subset)
                self.logger.info(f'🔄 Regime {regime}: {len(regime_features)} samples analyzed')
            if len(regime_importance) > 1:
                all_features = set(encoded_features.columns)
                common_features = all_features.copy()
                for regime, importance_data in regime_importance.items():
                    if 'ensemble' in importance_data:
                        regime_features = set([item['feature'] for item in importance_data['ensemble']])
                        common_features &= regime_features
                consistency_scores = {}
                for feature in common_features:
                    importances = []
                    for regime, importance_data in regime_importance.items():
                        if 'ensemble' in importance_data:
                            feature_importance = next((item['ensemble_importance'] for item in importance_data['ensemble'] if item['feature'] == feature), 0)
                            importances.append(feature_importance)
                    if importances:
                        consistency_scores[feature] = {'mean_importance': np.mean(importances), 'std_importance': np.std(importances), 'consistency': 1 - np.std(importances) / (np.mean(importances) + 1e-08)}
                consistency_df = pd.DataFrame.from_dict(consistency_scores, orient='index')
                consistency_df = consistency_df.sort_values('consistency', ascending=False)
                results = {'regime_importance': regime_importance, 'consistency_analysis': consistency_df.to_dict('index'), 'consistent_features': consistency_df[consistency_df['consistency'] > 0.8].index.tolist(), 'inconsistent_features': consistency_df[consistency_df['consistency'] < 0.3].index.tolist()}
            else:
                results = {'regime_importance': regime_importance, 'consistency_analysis': {}, 'consistent_features': [], 'inconsistent_features': []}
            self.logger.info('🔄 Regime analysis complete:')
            self.logger.info(f'   📊 Regimes analyzed: {len(regime_importance)}')
            self.logger.info(f"   📊 Consistent features: {len(results.get('consistent_features', []))}")
            return results
        except Exception as e:
            self.logger.exception(f'❌ Error in regime analysis: {e}')
            return {'error': str(e)}

    def _compare_with_original_features(self, encoded_features: pd.DataFrame, original_features: pd.DataFrame, labels: np.ndarray) -> dict[str, Any]:
        """Compare autoencoder features with original features."""
        try:
            encoded_importance = self._compute_ml_importance(encoded_features, labels)
            original_importance = self._compute_ml_importance(original_features, labels)
            comparison_results = {'encoded_importance': encoded_importance, 'original_importance': original_importance, 'comparison_metrics': {}}
            if 'ensemble' in encoded_importance and 'ensemble' in original_importance:
                encoded_top = [item['feature'] for item in encoded_importance['ensemble'][:10]]
                original_top = [item['feature'] for item in original_importance['ensemble'][:10]]
                overlap = set(encoded_top) & set(original_top)
                overlap_ratio = len(overlap) / len(encoded_top)
                comparison_results['comparison_metrics'] = {'top_feature_overlap': len(overlap), 'overlap_ratio': overlap_ratio, 'encoded_top_features': encoded_top, 'original_top_features': original_top}
            self.logger.info('🔄 Feature comparison complete:')
            if 'comparison_metrics' in comparison_results:
                self.logger.info(f"   📊 Top feature overlap: {comparison_results['comparison_metrics']['top_feature_overlap']}")
                self.logger.info(f"   📊 Overlap ratio: {comparison_results['comparison_metrics']['overlap_ratio']:.3f}")
            return comparison_results
        except Exception as e:
            self.logger.exception(f'❌ Error in feature comparison: {e}')
            return {'error': str(e)}

    def _generate_summary_and_recommendations(self, analysis_results: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Generate summary statistics and actionable recommendations."""
        try:
            summary = {}
            recommendations = []
            if 'feature_importance' in analysis_results and 'importance_summary' in analysis_results['feature_importance']:
                importance_summary = analysis_results['feature_importance']['importance_summary']
                summary['top_features'] = importance_summary.get('top_features', [])
                summary['mean_importance'] = importance_summary.get('mean_importance', 0)
            if 'correlation_analysis' in analysis_results and 'correlation_summary' in analysis_results['correlation_analysis']:
                corr_summary = analysis_results['correlation_analysis']['correlation_summary']
                summary['mean_correlation'] = corr_summary.get('mean_correlation', 0)
                summary['high_corr_count'] = corr_summary.get('high_corr_count', 0)
            if 'stability_metrics' in analysis_results and 'stability_summary' in analysis_results['stability_metrics']:
                stability_summary = analysis_results['stability_metrics']['stability_summary']
                summary['mean_stability'] = stability_summary.get('mean_stability', 0)
                summary['stable_count'] = stability_summary.get('stable_count', 0)
            if summary.get('mean_importance', 0) < 0.3:
                recommendations.append('⚠️ Low feature importance detected. Consider retraining autoencoder with different parameters.')
            if summary.get('mean_correlation', 0) < 0.1:
                recommendations.append('⚠️ Low correlation with targets. Autoencoder features may not be capturing relevant patterns.')
            if summary.get('mean_stability', 0) < 0.5:
                recommendations.append('⚠️ Low feature stability. Consider using more stable features or retraining with different data.')
            if summary.get('high_corr_count', 0) > 5:
                recommendations.append('💡 High correlation features detected. Consider feature selection to reduce redundancy.')
            if summary.get('stable_count', 0) > 10:
                recommendations.append('✅ Good feature stability detected. These features should perform well in production.')
            if summary.get('mean_importance', 0) > 0.7:
                recommendations.append('🎉 High feature importance detected. Autoencoder is generating valuable features.')
            if summary.get('mean_correlation', 0) > 0.3:
                recommendations.append('🎉 Good correlation with targets. Autoencoder features are capturing relevant patterns.')
            return (summary, recommendations)
        except Exception as e:
            self.logger.exception(f'❌ Error generating summary: {e}')
            return ({}, [f'Error generating summary: {e}'])

    def get_feature_ranking(self, method: str='ensemble') -> pd.DataFrame:
        """Get feature ranking based on specified method."""
        if method not in self.importance_scores:
            self.logger.warning(f"⚠️ Method '{method}' not available. Available methods: {list(self.importance_scores.keys())}")
            return pd.DataFrame()
        if 'ensemble' in self.importance_scores:
            return pd.DataFrame(self.importance_scores['ensemble'])
        else:
            return pd.DataFrame(self.importance_scores[method])

    def get_stable_features(self, threshold: float=0.7) -> list[str]:
        """Get list of stable features above threshold."""
        if 'stability_metrics' not in self.stability_metrics:
            return []
        stable_features = []
        for feature, metrics in self.stability_metrics['stability_metrics'].items():
            if metrics.get('overall_stability', 0) > threshold:
                stable_features.append(feature)
        return stable_features

    def get_high_correlation_features(self, threshold: float=0.5) -> list[str]:
        """Get list of features with high correlation to target."""
        if 'correlation_analysis' not in self.correlation_analysis:
            return []
        high_corr_features = []
        correlations = self.correlation_analysis['correlation_analysis'].get('pearson_correlations', {})
        for feature, corr in correlations.items():
            if abs(corr) > threshold:
                high_corr_features.append(feature)
        return high_corr_features

class AutoencoderFeatureGenerator:
    """Main class for the complete autoencoder feature generation workflow."""

    def __init__(self, config: str | dict | None=None) -> None:
        if not DEPENDENCIES_AVAILABLE:
            msg = f'Required dependencies not available: {MISSING_DEPENDENCY}'
            raise ImportError(msg)
        if isinstance(config, dict):
            temp_config = AutoencoderConfig()
            temp_config.config = config
            temp_config.logger = system_logger.getChild('AutoencoderConfig')
            self.config = temp_config
        else:
            self.config = AutoencoderConfig(config)
        self.logger = system_logger.getChild('AutoencoderFeatureGenerator')

    @comprehensive_data_validation
    @traced('autoencoder_feature_generation')
    def generate_features(self, features_df: pd.DataFrame, regime_name: str, labels: np.ndarray, regime_labels: np.ndarray | None=None, enable_analysis: bool | None=None) -> pd.DataFrame:
        """
        Generate autoencoder features from input features.

        CRITICAL: This method should only receive engineered features, not raw OHLCV data.
        Raw price data like 'volume', 'close', 'open', 'high', 'low' should be excluded.
        """
        raw_ohlcv_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'time']
        raw_ohlcv_columns = [col for col in raw_ohlcv_columns if col in features_df.columns]
        if raw_ohlcv_columns:
            self.logger.warning(f'🚨 CRITICAL: Found raw OHLCV columns in features: {raw_ohlcv_columns}')
            self.logger.warning('🚨 These should be excluded from autoencoder feature generation')
            self.logger.warning('🚨 Raw price data should be processed into engineered features first')
            features_df = features_df.drop(columns=raw_ohlcv_columns)
            self.logger.info(f'✅ Removed {len(raw_ohlcv_columns)} raw OHLCV columns from features')
            self.logger.info(f'📊 Features shape after removal: {features_df.shape}')
            if features_df.empty:
                self.logger.error('🚨 CRITICAL: No engineered features remaining after removing raw OHLCV data')
                self.logger.error('🚨 This indicates a serious data pipeline issue')
                return pd.DataFrame()
        'Generate autoencoder-based features for a specific market regime.'
        try:
            self.logger.info(f'🚀 Starting autoencoder feature generation for regime: {regime_name}')
            potential_label_columns = ['label', 'target', 'y', 'class', 'Label', 'Target', 'Y', 'Class', 'labels', 'targets', 'classes', 'Labels', 'Targets', 'Classes', 'signal', 'prediction', 'direction', 'Signal', 'Prediction', 'Direction', 'buy_sell', 'position', 'trade_signal', 'Buy_Sell', 'Position', 'Trade_Signal', 'future_return', 'next_return', 'price_change', 'Future_Return', 'Next_Return', 'Price_Change', 'binary_target', 'binary_label', 'Binary_Target', 'Binary_Label', 'multi_target', 'multi_label', 'Multi_Target', 'Multi_Label', 'label_encoded', 'target_encoded', 'Label_Encoded', 'Target_Encoded', 'meta_label', 'meta_target', 'Meta_Label', 'Meta_Target', 'triple_barrier_label', 'barrier_label', 'Triple_Barrier_Label', 'Barrier_Label']
            actual_label_columns = [col for col in features_df.columns if col in potential_label_columns]
            if actual_label_columns:
                self.logger.error('🚨 CRITICAL DATA LEAKAGE DETECTED in autoencoder!')
                self.logger.error(f'🚨 Found label columns in autoencoder input: {actual_label_columns}')
                self.logger.error('🚨 This will cause severe data leakage! Removing these columns from autoencoder analysis.')
                features_df = features_df.drop(columns=actual_label_columns)
                self.logger.info(f'📊 Autoencoder features after leakage prevention: {features_df.shape[1]} columns')
            if len(features_df) < 10:
                self.logger.warning('⚠️ Insufficient data for autoencoder feature generation, returning original features')
                return features_df
            self.logger.info('🔄 NEW STEP: Converting price features to returns for autoencoder training...')
            price_converter = PriceReturnConverter(self.config)
            features_df = price_converter.convert_price_features_to_returns(features_df)
            self.logger.info(f'✅ Price return conversion completed. Features shape: {features_df.shape}')
            self.logger.info('🔄 Step 1/5: Feature filtering with Random Forest + SHAP')
            self.logger.info(f'📊 Starting with {features_df.shape[1]} input features')
            feature_filter = FeatureFilter(self.config)
            filtered_features = feature_filter.filter_features(features_df, labels)
            if features_df.shape[1] == 0:
                self.logger.warning('⚠️ No features available for filtering - returning original features')
                return features_df
            feature_reduction = features_df.shape[1] - filtered_features.shape[1]
            reduction_percentage = feature_reduction / features_df.shape[1] * 100
            self.logger.info('✅ Feature filtering completed successfully!')
            self.logger.info(f'📊 Results: {filtered_features.shape[1]} features selected from {features_df.shape[1]} input features')
            self.logger.info(f'📉 Feature reduction: {feature_reduction} features removed ({reduction_percentage:.1f}% reduction)')
            self.logger.info('🔍 Validating feature quality for autoencoder training...')
            min_features_for_ae = int(self.config.get('feature_filtering.min_features_for_ae', 15))
            numeric_features = filtered_features.select_dtypes(include=[np.number])
            actual_numeric_features = numeric_features.shape[1]
            self.logger.info(f'📊 Numeric features available: {actual_numeric_features}')
            self.logger.info(f'📊 Minimum features required: {min_features_for_ae}')
            if actual_numeric_features < min_features_for_ae:
                self.logger.warning('⚠️ Insufficient features for autoencoder training')
                self.logger.warning(f'📊 Have: {actual_numeric_features} numeric features, Need: {min_features_for_ae}+ features')
                self.logger.info('🔄 Returning original features without autoencoder enhancement')
                return features_df
            std_threshold = float(self.config.get('autoencoder.min_feature_std', 1e-06))
            per_feature_std = numeric_features.std(axis=0, skipna=True)
            low_std_cols = per_feature_std.index[per_feature_std <= std_threshold].tolist()
            if len(low_std_cols) > 0:
                preview = ', '.join(low_std_cols[:10]) + ('...' if len(low_std_cols) > 10 else '')
                self.logger.warning('⚠️ Low variance features detected')
                self.logger.warning(f'📊 {len(low_std_cols)} features have std <= {std_threshold:g}')
                self.logger.warning(f'📊 Examples: {preview}')
                self.logger.info('🔄 Returning original features without autoencoder enhancement')
                return features_df
            self.logger.info('✅ Feature quality validation passed - proceeding with autoencoder training')
            self.logger.info('🔄 Step 2/5: Data preprocessing and sequence creation')
            self.logger.info('🔧 Initializing data preprocessor...')
            preprocessor = ImprovedAutoencoderPreprocessor(self.config)
            self.logger.info('🔧 Fitting preprocessor on filtered features...')
            preprocessor.fit(filtered_features)
            self.logger.info('🔧 Transforming features for autoencoder input...')
            X_processed = preprocessor.transform(filtered_features)
            self.logger.info('✅ Preprocessing completed successfully')
            self.logger.info(f'📊 Processed data shape: {X_processed.shape}')
            timesteps = self.config.get('sequence.timesteps', 10)
            self.logger.info(f'📊 Creating sequences with {timesteps} timesteps...')
            X_sequences, y_targets, target_indices = create_sequences_with_index(X_processed, timesteps, filtered_features.index)
            self.logger.info('✅ Sequence creation completed successfully')
            self.logger.info(f'📊 Sequence shapes: X_sequences={X_sequences.shape}, y_targets={y_targets.shape}')
            self.logger.info(f'📊 Sequence configuration: timesteps={timesteps}, overlap=50%')
            self.logger.info(f'📊 Target indices: {len(target_indices)} samples with preserved timestamps')
            min_sequences = 5
            if len(X_sequences) < min_sequences:
                self.logger.warning('⚠️ Insufficient sequences for autoencoder training')
                self.logger.warning(f'📊 Have: {len(X_sequences)} sequences, Need: {min_sequences}+ sequences')
                self.logger.info('🔄 Returning original features without autoencoder enhancement')
                return features_df
            self.logger.info('🔄 Step 3/5: Hyperparameter optimization with Optuna')
            split_ratio = 0.8
            split_idx = int(split_ratio * len(X_sequences))
            X_train, y_train = (X_sequences[:split_idx], y_targets[:split_idx])
            X_val, y_val = (X_sequences[split_idx:], y_targets[split_idx:])
            self.logger.info(f'📊 Data split configuration: {split_ratio * 100:.0f}% train, {(1 - split_ratio) * 100:.0f}% validation')
            self.logger.info(f'📊 Training set: {X_train.shape[0]} sequences ({X_train.shape[0] / len(X_sequences) * 100:.1f}%)')
            self.logger.info(f'📊 Validation set: {X_val.shape[0]} sequences ({X_val.shape[0] / len(X_sequences) * 100:.1f}%)')
            n_trials = self.config.get('training.n_trials', 50)
            n_jobs = self.config.get('training.n_jobs', 1)
            self.logger.info('🔍 Starting Optuna hyperparameter optimization')
            self.logger.info(f'📊 Optimization parameters: n_trials={n_trials}, n_jobs={n_jobs}')
            self.logger.info('📊 Search space: filters=[16,32,64], kernel_size=[3-7], dropout=[0.1-0.5], lr=[1e-4-1e-2], encoding_dim=[8-64]')
            best_params = self._run_optuna_optimization(X_train, y_train, X_val, y_val)
            self.config.config['best_params'] = best_params
            self.logger.info('✅ Hyperparameter optimization completed successfully')
            self.logger.info('🏆 Best hyperparameters selected:')
            for param, value in best_params.items():
                self.logger.info(f'   📊 {param}: {value}')
            self.logger.info('🔄 Step 4/5: Final autoencoder training and feature generation')
            self.logger.info('🔧 Building final autoencoder model with optimized hyperparameters...')
            final_autoencoder = SequenceAwareAutoencoder(self.config)
            final_autoencoder.build_model(X_sequences.shape[1:])
            self.logger.info('🔧 Training final autoencoder model...')
            training_history = final_autoencoder.fit(X_train, y_train, X_val, y_val)
            if hasattr(training_history, 'history'):
                final_train_loss = training_history.history.get('loss', [0])[-1]
                final_val_loss = training_history.history.get('val_loss', [0])[-1]
                self.logger.info('✅ Final model training completed')
                self.logger.info(f'📊 Final training loss: {final_train_loss:.6f}')
                self.logger.info(f'📊 Final validation loss: {final_val_loss:.6f}')
                self.logger.info(f"📊 Model performance: {('Good' if final_val_loss < 0.1 else 'Acceptable' if final_val_loss < 0.5 else 'Needs improvement')}")
            self.logger.info('🔧 Generating encoded features and reconstructions...')
            self.logger.info('📊 Using encoder to extract latent representations...')
            encoded_features = final_autoencoder.encoder.predict(X_sequences, verbose=0)
            self.logger.info('📊 Using full autoencoder to generate reconstructions...')
            reconstructed = final_autoencoder.autoencoder.predict(X_sequences, verbose=0)
            self.logger.info('✅ Feature generation completed successfully')
            self.logger.info(f'📊 Encoded features shape: {encoded_features.shape}')
            self.logger.info(f'📊 Reconstructed features shape: {reconstructed.shape}')
            self.logger.info('📊 Calculating reconstruction error...')
            recon_error = np.mean((y_targets - reconstructed) ** 2, axis=1)
            mean_recon_error = np.mean(recon_error)
            std_recon_error = np.std(recon_error)
            self.logger.info('📊 Reconstruction error statistics:')
            self.logger.info(f'   📊 Mean reconstruction error: {mean_recon_error:.6f}')
            self.logger.info(f'   📊 Std reconstruction error: {std_recon_error:.6f}')
            self.logger.info(f'   📊 Min reconstruction error: {np.min(recon_error):.6f}')
            self.logger.info(f'   📊 Max reconstruction error: {np.max(recon_error):.6f}')
            self.logger.info('🔄 Step 5/5: Creating enriched feature DataFrame')
            self.logger.info('📊 Creating encoded features DataFrame...')
            encoded_df = pd.DataFrame(encoded_features, index=target_indices, columns=[f'autoencoder_{i + 1}' for i in range(encoded_features.shape[1])])
            encoded_df['autoencoder_recon_error'] = recon_error
            self.logger.info('✅ Encoded features DataFrame created successfully')
            self.logger.info(f'📊 Encoded DataFrame shape: {encoded_df.shape}')
            self.logger.info(f'📊 Encoded features: {encoded_features.shape[1]} latent dimensions + 1 reconstruction error')
            self.logger.info('📊 Merging encoded features with original features...')
            result_df = features_df.merge(encoded_df, left_index=True, right_index=True, how='left')
            autoencoder_cols = [col for col in result_df.columns if 'autoencoder' in col]
            result_df[autoencoder_cols] = result_df[autoencoder_cols].fillna(0)
            self.logger.info('✅ Feature merging completed successfully')
            self.logger.info(f'📊 Original features: {features_df.shape[1]} columns')
            self.logger.info(f'📊 Autoencoder features added: {len(autoencoder_cols)} columns')
            self.logger.info(f'📊 Final result shape: {result_df.shape}')
            self.logger.info(f'📊 Feature enhancement: {len(autoencoder_cols)} new features added ({len(autoencoder_cols) / features_df.shape[1] * 100:.1f}% increase)')
            enable_analysis = enable_analysis if enable_analysis is not None else self.config.get('feature_analysis.enable_analysis', True)
            if enable_analysis:
                self.logger.info('🔍 Starting feature importance analysis...')
                try:
                    feature_analyzer = AutoencoderFeatureAnalyzer(self.config)
                    autoencoder_features = result_df[autoencoder_cols].copy()
                    analysis_results = feature_analyzer.analyze_feature_importance(encoded_features=autoencoder_features, labels=labels, original_features=features_df if self.config.get('feature_analysis.comparison_with_original', True) else None, regime_labels=regime_labels if self.config.get('feature_analysis.regime_analysis_enabled', True) else None)
                    if 'error' not in analysis_results:
                        self.logger.info('📊 Feature importance analysis completed successfully!')
                        if 'summary_statistics' in analysis_results:
                            summary = analysis_results['summary_statistics']
                            self.logger.info('📈 Analysis Summary:')
                            self.logger.info(f"   🏆 Top features: {summary.get('top_features', [])[:5]}")
                            self.logger.info(f"   📊 Mean importance: {summary.get('mean_importance', 0):.4f}")
                            self.logger.info(f"   📊 Mean correlation: {summary.get('mean_correlation', 0):.4f}")
                            self.logger.info(f"   📊 Mean stability: {summary.get('mean_stability', 0):.4f}")
                        if 'recommendations' in analysis_results:
                            recommendations = analysis_results['recommendations']
                            if recommendations:
                                self.logger.info('💡 Recommendations:')
                                for rec in recommendations[:5]:
                                    self.logger.info(f'   {rec}')
                        self.last_analysis_results = analysis_results
                    else:
                        self.logger.warning(f"⚠️ Feature analysis failed: {analysis_results['error']}")
                except Exception as e:
                    self.logger.exception(f'❌ Error in feature importance analysis: {e}')
                    self.logger.info('🔄 Continuing without feature analysis...')
            self.logger.info('🎉 Autoencoder feature generation pipeline completed successfully!')
            self.logger.info(f"📊 Summary for regime '{regime_name}':")
            self.logger.info(f'   📊 Input features: {features_df.shape[1]} columns')
            self.logger.info(f'   📊 Output features: {result_df.shape[1]} columns')
            self.logger.info(f'   📊 New autoencoder features: {len(autoencoder_cols)} columns')
            self.logger.info(f'   📊 Data samples: {result_df.shape[0]} rows')
            self.logger.info(f"   📊 Autoencoder performance: {('Good' if mean_recon_error < 0.1 else 'Acceptable' if mean_recon_error < 0.5 else 'Needs improvement')}")
            return result_df
        except Exception as e:
            self.logger.exception('❌ Error in autoencoder feature generation pipeline')
            self.logger.error(f'📊 Error details: {str(e)}')
            self.logger.info('🔄 Returning original features without autoencoder enhancement')
            return features_df

    def _run_optuna_optimization(self, X_train: Any, y_train: Any, X_val: Any, y_val: Any) -> None:
        """Helper to encapsulate the Optuna study logic."""

        def objective(trial: Any) -> None:
            try:
                autoencoder = SequenceAwareAutoencoder(self.config)
                autoencoder.build_model(X_train.shape[1:], trial)
                history = autoencoder.fit(X_train, y_train, X_val, y_val, trial)
                return min(history.history['val_loss'])
            except Exception as e:
                self.logger.warning(f'⚠️ Trial failed: {str(e)}')
                return float('inf')
        self.logger.info('🔧 Creating Optuna study for hyperparameter optimization...')
        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        n_trials = self.config.get('training.n_trials', 50)
        n_jobs = self.config.get('training.n_jobs', 1)
        self.logger.info(f'🚀 Starting Optuna optimization with {n_trials} trials...')
        self.logger.info(f'📊 Parallel jobs: {n_jobs} (1 recommended for GPU compatibility)')
        start_time = time.time()
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
        optimization_time = time.time() - start_time
        self.logger.info('✅ Optuna optimization completed successfully!')
        self.logger.info(f'📊 Optimization time: {optimization_time:.2f} seconds')
        self.logger.info(f'📊 Trials completed: {len(study.trials)}')
        self.logger.info(f'📊 Successful trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}')
        self.logger.info(f'📊 Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}')
        self.logger.info(f'🏆 Best validation loss: {study.best_value:.6f}')
        self.logger.info(f'🏆 Best trial number: {study.best_trial.number}')
        return study.best_params

    def get_last_analysis_results(self) -> dict[str, Any] | None:
        """Get the results from the last feature importance analysis."""
        return getattr(self, 'last_analysis_results', None)

    def get_feature_ranking(self, method: str='ensemble') -> pd.DataFrame:
        """Get feature ranking from the last analysis."""
        analysis_results = self.get_last_analysis_results()
        if analysis_results and 'feature_importance' in analysis_results:
            feature_importance = analysis_results['feature_importance']
            if method in feature_importance and feature_importance[method] is not None:
                return pd.DataFrame(feature_importance[method])
        return pd.DataFrame()

    def get_stable_features(self, threshold: float=0.7) -> list[str]:
        """Get list of stable features from the last analysis."""
        analysis_results = self.get_last_analysis_results()
        if analysis_results and 'stability_metrics' in analysis_results:
            stability_metrics = analysis_results['stability_metrics']
            if 'stable_features' in stability_metrics:
                return stability_metrics['stable_features']
        return []

    def get_high_correlation_features(self, threshold: float=0.5) -> list[str]:
        """Get list of features with high correlation to target from the last analysis."""
        analysis_results = self.get_last_analysis_results()
        if analysis_results and 'correlation_analysis' in analysis_results:
            correlation_analysis = analysis_results['correlation_analysis']
            if 'high_correlations' in correlation_analysis:
                return list(correlation_analysis['high_correlations'].keys())
        return []

    def get_recommendations(self) -> list[str]:
        """Get recommendations from the last analysis."""
        analysis_results = self.get_last_analysis_results()
        if analysis_results and 'recommendations' in analysis_results:
            return analysis_results['recommendations']
        return []