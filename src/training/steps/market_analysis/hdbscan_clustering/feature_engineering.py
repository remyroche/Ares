"""
Enhanced Feature Engineering Pipeline

This module provides intelligent feature selection, preprocessing, and engineering
for HDBSCAN clustering with proper validation and correlation handling.
Enhanced with VectorBT optimizations and comprehensive tprint logging.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import time
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
from scipy.signal import find_peaks
import warnings

# Import VectorBT optimizations
try:
    from src.utils.vectorbt_compat import (
        vbt, rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
        rolling_sum, rolling_apply, VECTORBT_AVAILABLE
    )
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None

# Import tprint system
from src.utils.tprint import (
    tprint, tprint_data_preview, tprint_data_format, 
    tprint_performance
)

# Import hardware optimizations
from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager
from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer

# Import feature selection methods
try:
    from mrmr import mrmr_regression
    MRMR_AVAILABLE = True
except ImportError:
    MRMR_AVAILABLE = False
    mrmr_regression = None

logger = logging.getLogger(__name__)


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering pipeline."""
    correlation_threshold: float = 0.95
    enable_feature_selection: bool = True
    feature_selection_method: str = "mrmr"  # 'mrmr', 'lasso', 'mutual_info', 'f_test'
    max_features: int = 50
    enable_regime_features: bool = True
    enable_entropy_features: bool = True
    enable_spectral_features: bool = True
    enable_temporal_features: bool = True
    enable_technical_indicators: bool = True
    enable_volatility_features: bool = True
    enable_momentum_features: bool = True
    preprocessing_method: str = "robust"  # 'standard', 'robust', 'minmax'
    enable_outlier_detection: bool = True
    outlier_threshold: float = 3.0  # Z-score threshold
    enable_feature_interactions: bool = False
    max_interaction_features: int = 10


class FeatureValidator:
    """Validator for feature quality and selection."""
    
    def __init__(self, correlation_threshold: float = 0.95):
        self.correlation_threshold = correlation_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_features(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate feature quality and identify issues.
        
        Args:
            features_df: DataFrame containing features
            
        Returns:
            Validation results with recommendations
        """
        validation_results = {
            'passed': True,
            'issues': [],
            'recommendations': [],
            'feature_stats': {}
        }
        
        # Check for missing values
        missing_ratios = features_df.isnull().sum() / len(features_df)
        high_missing = missing_ratios[missing_ratios > 0.1]
        
        if len(high_missing) > 0:
            validation_results['issues'].append(f"High missing values in columns: {high_missing.index.tolist()}")
            validation_results['recommendations'].append("Consider imputation or removal of high-missing columns")
        
        # Check for constant features
        constant_features = features_df.nunique() == 1
        if constant_features.any():
            validation_results['issues'].append(f"Constant features found: {constant_features[constant_features].index.tolist()}")
            validation_results['recommendations'].append("Remove constant features")
        
        # Check for highly correlated features
        correlation_matrix = features_df.corr().abs()
        high_corr_pairs = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if corr_value > self.correlation_threshold:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        corr_value
                    ))
        
        if high_corr_pairs:
            validation_results['issues'].append(f"High correlation pairs found: {len(high_corr_pairs)}")
            validation_results['recommendations'].append(f"Consider removing one feature from each high-correlation pair")
        
        # Check for infinite values
        inf_features = features_df.isin([np.inf, -np.inf]).any()
        if inf_features.any():
            validation_results['issues'].append(f"Infinite values found in: {inf_features[inf_features].index.tolist()}")
            validation_results['recommendations'].append("Handle infinite values before clustering")
        
        # Calculate feature statistics
        validation_results['feature_stats'] = {
            'n_features': len(features_df.columns),
            'n_samples': len(features_df),
            'missing_ratio': missing_ratios.mean(),
            'constant_features': constant_features.sum(),
            'high_corr_pairs': len(high_corr_pairs),
            'inf_features': inf_features.sum()
        }
        
        # Overall validation
        validation_results['passed'] = len(validation_results['issues']) == 0
        
        return validation_results
    
    def remove_problematic_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Remove problematic features based on validation results."""
        cleaned_df = features_df.copy()
        
        # Remove constant features
        constant_features = cleaned_df.nunique() == 1
        if constant_features.any():
            cleaned_df = cleaned_df.drop(columns=constant_features[constant_features].index)
            self.logger.info(f"Removed {constant_features.sum()} constant features")
        
        # Remove features with high missing values
        missing_ratios = cleaned_df.isnull().sum() / len(cleaned_df)
        high_missing = missing_ratios[missing_ratios > 0.1]
        if high_missing.any():
            cleaned_df = cleaned_df.drop(columns=high_missing.index)
            self.logger.info(f"Removed {len(high_missing)} features with high missing values")
        
        # Remove features with infinite values
        inf_features = cleaned_df.isin([np.inf, -np.inf]).any()
        if inf_features.any():
            cleaned_df = cleaned_df.drop(columns=inf_features[inf_features].index)
            self.logger.info(f"Removed {inf_features.sum()} features with infinite values")
        
        return cleaned_df


class FeatureSelector:
    """Intelligent feature selection using multiple methods."""
    
    def __init__(self, method: str = "mrmr", max_features: int = 50, enable_feature_selection: bool = True):
        self.method = method
        self.max_features = max_features
        self.enable_feature_selection = enable_feature_selection
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def select_features(self, features_df: pd.DataFrame, target: Optional[np.ndarray] = None) -> List[str]:
        """
        Select features using the specified method.
        
        Args:
            features_df: DataFrame containing features
            target: Optional target variable for supervised selection
            
        Returns:
            List of selected feature names
        """
        if not self.enable_feature_selection:
            return features_df.columns.tolist()
        
        if len(features_df.columns) <= self.max_features:
            return features_df.columns.tolist()
        
        try:
            if self.method == "mrmr" and MRMR_AVAILABLE:
                return self._select_features_mrmr(features_df, target)
            elif self.method == "lasso":
                return self._select_features_lasso(features_df, target)
            elif self.method == "mutual_info":
                return self._select_features_mutual_info(features_df, target)
            elif self.method == "f_test":
                return self._select_features_f_test(features_df, target)
            else:
                self.logger.warning(f"Unknown feature selection method: {self.method}, using mutual_info")
                return self._select_features_mutual_info(features_df, target)
                
        except Exception as e:
            self.logger.error(f"Feature selection failed: {e}")
            # Fallback to correlation-based selection
            return self._select_features_correlation(features_df)
    
    def _select_features_mrmr(self, features_df: pd.DataFrame, target: Optional[np.ndarray] = None) -> List[str]:
        """Select features using mRMR (minimum Redundancy Maximum Relevance)."""
        if target is None:
            # Use first column as target for unsupervised selection
            target = features_df.iloc[:, 0].values
        
        try:
            selected_features = mrmr_regression(
                features_df, target, K=self.max_features
            )
            return selected_features
        except Exception as e:
            self.logger.warning(f"mRMR selection failed: {e}, falling back to mutual_info")
            return self._select_features_mutual_info(features_df, target)
    
    def _select_features_lasso(self, features_df: pd.DataFrame, target: Optional[np.ndarray] = None) -> List[str]:
        """Select features using Lasso regularization."""
        if target is None:
            # Use first column as target for unsupervised selection
            target = features_df.iloc[:, 0].values
        
        # Handle missing values
        features_clean = features_df.fillna(features_df.median())
        
        # Fit Lasso with cross-validation
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(features_clean, target)
        
        # Get selected features (non-zero coefficients)
        selected_mask = lasso.coef_ != 0
        selected_features = features_df.columns[selected_mask].tolist()
        
        # If too many features selected, take top max_features
        if len(selected_features) > self.max_features:
            feature_importance = np.abs(lasso.coef_)
            top_indices = np.argsort(feature_importance)[-self.max_features:]
            selected_features = features_df.columns[top_indices].tolist()
        
        return selected_features
    
    def _select_features_mutual_info(self, features_df: pd.DataFrame, target: Optional[np.ndarray] = None) -> List[str]:
        """Select features using mutual information."""
        if target is None:
            # Use first column as target for unsupervised selection
            target = features_df.iloc[:, 0].values
        
        # Handle missing values
        features_clean = features_df.fillna(features_df.median())
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(features_clean, target, random_state=42)
        
        # Select top features
        top_indices = np.argsort(mi_scores)[-self.max_features:]
        selected_features = features_df.columns[top_indices].tolist()
        
        return selected_features
    
    def _select_features_f_test(self, features_df: pd.DataFrame, target: Optional[np.ndarray] = None) -> List[str]:
        """Select features using F-test."""
        if target is None:
            # Use first column as target for unsupervised selection
            target = features_df.iloc[:, 0].values
        
        # Handle missing values
        features_clean = features_df.fillna(features_df.median())
        
        # Use F-test for feature selection
        selector = SelectKBest(score_func=f_regression, k=self.max_features)
        selector.fit(features_clean, target)
        
        selected_features = features_df.columns[selector.get_support()].tolist()
        return selected_features
    
    def _select_features_correlation(self, features_df: pd.DataFrame) -> List[str]:
        """Fallback correlation-based feature selection."""
        # Calculate correlation matrix
        corr_matrix = features_df.corr().abs()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        # Remove one feature from each high-correlation pair
        features_to_remove = set()
        for feat1, feat2 in high_corr_pairs:
            if feat1 not in features_to_remove:
                features_to_remove.add(feat2)
        
        selected_features = [col for col in features_df.columns if col not in features_to_remove]
        
        # If still too many features, take first max_features
        if len(selected_features) > self.max_features:
            selected_features = selected_features[:self.max_features]
        
        return selected_features


class AdvancedFeatureGenerator:
    """Generator for advanced features including regime-specific features with VectorBT optimizations."""
    
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.hardware_manager = UnifiedHardwareManager()
        self.memory_optimizer = M1MemoryOptimizer()
        self.vectorbt_available = VECTORBT_AVAILABLE
        
        if self.vectorbt_available:
            tprint("✅ VectorBT optimizations enabled for feature generation")
        else:
            tprint("⚠️ VectorBT not available - using pandas fallback methods")
    
    def generate_features(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive feature set for clustering with VectorBT optimizations.
        
        Args:
            data_df: Input data DataFrame
            
        Returns:
            DataFrame with generated features
        """
        tprint("🚀 Starting VectorBT-optimized feature generation")
        tprint_data_preview(data_df, "Input data")
        
        start_time = time.time()
        features_df = data_df.copy()
        
        # Generate different types of features
        if self.config.enable_technical_indicators:
            tprint("📊 Generating technical indicators with VectorBT")
            features_df = self._add_technical_indicators(features_df)
        
        if self.config.enable_volatility_features:
            tprint("📈 Generating volatility features with VectorBT")
            features_df = self._add_volatility_features(features_df)
        
        if self.config.enable_momentum_features:
            tprint("⚡ Generating momentum features with VectorBT")
            features_df = self._add_momentum_features(features_df)
        
        if self.config.enable_entropy_features:
            tprint("🔍 Generating entropy features")
            features_df = self._add_entropy_features(features_df)
        
        if self.config.enable_spectral_features:
            tprint("🌊 Generating spectral features")
            features_df = self._add_spectral_features(features_df)
        
        if self.config.enable_temporal_features:
            tprint("⏰ Generating temporal features")
            features_df = self._add_temporal_features(features_df)
        
        if self.config.enable_regime_features:
            tprint("🎯 Generating regime-specific features")
            features_df = self._add_regime_features(features_df)
        
        if self.config.enable_feature_interactions:
            tprint("🔗 Generating feature interactions")
            features_df = self._add_feature_interactions(features_df)
        
        processing_time = time.time() - start_time
        tprint_performance(f"Feature generation completed in {processing_time:.2f}s")
        tprint_data_preview(features_df, "Generated features")
        
        return features_df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators with VectorBT optimizations."""
        if 'close' not in df.columns:
            tprint("⚠️ Close price not available for technical indicators")
            return df
        
        close = df['close']
        
        if self.vectorbt_available:
            try:
                # Moving averages with VectorBT
                df['sma_5'] = rolling_mean(close, window=5)
                df['sma_20'] = rolling_mean(close, window=20)
                df['sma_50'] = rolling_mean(close, window=50)
                
                # Exponential moving averages
                df['ema_5'] = close.ewm(span=5).mean()
                df['ema_20'] = close.ewm(span=20).mean()
                
                # Bollinger Bands with VectorBT
                sma_20 = rolling_mean(close, window=20)
                std_20 = rolling_std(close, window=20)
                df['bb_upper'] = sma_20 + (std_20 * 2)
                df['bb_lower'] = sma_20 - (std_20 * 2)
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma_20
                df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
                
                # RSI with VectorBT
                delta = close.diff()
                gain = rolling_mean(delta.where(delta > 0, 0), window=14)
                loss = rolling_mean((-delta.where(delta < 0, 0)), window=14)
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                
                # MACD with VectorBT
                ema_12 = close.ewm(span=12).mean()
                ema_26 = close.ewm(span=26).mean()
                df['macd'] = ema_12 - ema_26
                df['macd_signal'] = df['macd'].ewm(span=9).mean()
                df['macd_histogram'] = df['macd'] - df['macd_signal']
                
                tprint("✅ Technical indicators generated with VectorBT")
                
            except Exception as e:
                tprint(f"⚠️ VectorBT technical indicators failed: {e}, using pandas fallback")
                df = self._add_pandas_technical_indicators(df)
        else:
            df = self._add_pandas_technical_indicators(df)
        
        return df
    
    def _add_pandas_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback pandas implementation for technical indicators."""
        close = df['close']
        
        # Moving averages
        df['sma_5'] = close.rolling(5).mean()
        df['sma_20'] = close.rolling(20).mean()
        df['sma_50'] = close.rolling(50).mean()
        
        # Exponential moving averages
        df['ema_5'] = close.ewm(span=5).mean()
        df['ema_20'] = close.ewm(span=20).mean()
        
        # Bollinger Bands
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma_20
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features with VectorBT optimizations."""
        if 'close' not in df.columns:
            tprint("⚠️ Close price not available for volatility features")
            return df
        
        close = df['close']
        returns = close.pct_change()
        
        if self.vectorbt_available:
            try:
                # Rolling volatility with VectorBT
                df['volatility_5'] = rolling_std(returns, window=5)
                df['volatility_20'] = rolling_std(returns, window=20)
                df['volatility_50'] = rolling_std(returns, window=50)
                
                # GARCH-like features
                df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
                
                # Volatility trend using VectorBT rolling apply
                def volatility_trend_func(x):
                    if len(x) < 2:
                        return np.nan
                    return np.polyfit(range(len(x)), x, 1)[0]
                
                df['volatility_trend'] = rolling_apply(
                    df['volatility_20'], window=10, func=volatility_trend_func
                )
                
                # High-low volatility
                if 'high' in df.columns and 'low' in df.columns:
                    df['hl_volatility'] = (df['high'] - df['low']) / close
                    df['hl_volatility_ma'] = rolling_mean(df['hl_volatility'], window=20)
                
                tprint("✅ Volatility features generated with VectorBT")
                
            except Exception as e:
                tprint(f"⚠️ VectorBT volatility features failed: {e}, using pandas fallback")
                df = self._add_pandas_volatility_features(df)
        else:
            df = self._add_pandas_volatility_features(df)
        
        return df
    
    def _add_pandas_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback pandas implementation for volatility features."""
        close = df['close']
        returns = close.pct_change()
        
        # Rolling volatility
        df['volatility_5'] = returns.rolling(5).std()
        df['volatility_20'] = returns.rolling(20).std()
        df['volatility_50'] = returns.rolling(50).std()
        
        # GARCH-like features
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
        df['volatility_trend'] = df['volatility_20'].rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else np.nan
        )
        
        # High-low volatility
        if 'high' in df.columns and 'low' in df.columns:
            df['hl_volatility'] = (df['high'] - df['low']) / close
            df['hl_volatility_ma'] = df['hl_volatility'].rolling(20).mean()
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features with VectorBT optimizations."""
        if 'close' not in df.columns:
            tprint("⚠️ Close price not available for momentum features")
            return df
        
        close = df['close']
        
        if self.vectorbt_available:
            try:
                # Price momentum with VectorBT
                df['momentum_5'] = close / rolling_mean(close.shift(5), window=1) - 1
                df['momentum_20'] = close / rolling_mean(close.shift(20), window=1) - 1
                df['momentum_50'] = close / rolling_mean(close.shift(50), window=1) - 1
                
                # Rate of change
                df['roc_5'] = close.pct_change(5)
                df['roc_20'] = close.pct_change(20)
                
                # Momentum indicators
                df['momentum_ratio'] = df['momentum_5'] / df['momentum_20']
                
                # Momentum trend using VectorBT rolling apply
                def momentum_trend_func(x):
                    if len(x) < 2:
                        return np.nan
                    return np.polyfit(range(len(x)), x, 1)[0]
                
                df['momentum_trend'] = rolling_apply(
                    df['momentum_20'], window=10, func=momentum_trend_func
                )
                
                tprint("✅ Momentum features generated with VectorBT")
                
            except Exception as e:
                tprint(f"⚠️ VectorBT momentum features failed: {e}, using pandas fallback")
                df = self._add_pandas_momentum_features(df)
        else:
            df = self._add_pandas_momentum_features(df)
        
        return df
    
    def _add_pandas_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback pandas implementation for momentum features."""
        close = df['close']
        
        # Price momentum
        df['momentum_5'] = close / close.shift(5) - 1
        df['momentum_20'] = close / close.shift(20) - 1
        df['momentum_50'] = close / close.shift(50) - 1
        
        # Rate of change
        df['roc_5'] = close.pct_change(5)
        df['roc_20'] = close.pct_change(20)
        
        # Momentum indicators
        df['momentum_ratio'] = df['momentum_5'] / df['momentum_20']
        df['momentum_trend'] = df['momentum_20'].rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else np.nan
        )
        
        return df
    
    def _add_entropy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add entropy-based features with tprint logging."""
        if 'close' not in df.columns:
            tprint("⚠️ Close price not available for entropy features")
            return df
        
        close = df['close']
        returns = close.pct_change().dropna()
        
        tprint("🔍 Calculating entropy features...")
        
        # Rolling entropy of returns
        window = 20
        entropy_values = []
        
        for i in range(len(returns)):
            if i < window - 1:
                entropy_values.append(np.nan)
            else:
                window_returns = returns.iloc[i-window+1:i+1]
                # Handle NaN values
                window_returns_clean = window_returns.dropna()
                if len(window_returns_clean) < 2:
                    entropy_values.append(np.nan)
                    continue
                
                # Check if all values are the same
                if window_returns_clean.nunique() == 1:
                    entropy_values.append(0.0)
                    continue
                
                # Discretize returns into bins
                bins = np.histogram_bin_edges(window_returns_clean, bins=10)
                hist, _ = np.histogram(window_returns_clean, bins=bins)
                # Calculate entropy
                probabilities = hist / hist.sum()
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
                entropy_values.append(entropy)
        
        df['entropy_20'] = pd.Series(entropy_values, index=returns.index)
        
        # Entropy of volatility
        volatility = returns.rolling(10).std()
        vol_entropy_values = []
        
        for i in range(len(volatility)):
            if i < window - 1:
                vol_entropy_values.append(np.nan)
            else:
                window_vol = volatility.iloc[i-window+1:i+1]
                # Handle NaN values
                window_vol_clean = window_vol.dropna()
                if len(window_vol_clean) < 2:
                    vol_entropy_values.append(np.nan)
                    continue
                
                # Check if all values are the same
                if window_vol_clean.nunique() == 1:
                    vol_entropy_values.append(0.0)
                    continue
                
                bins = np.histogram_bin_edges(window_vol_clean, bins=10)
                hist, _ = np.histogram(window_vol_clean, bins=bins)
                probabilities = hist / hist.sum()
                entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
                vol_entropy_values.append(entropy)
        
        df['vol_entropy_20'] = pd.Series(vol_entropy_values, index=volatility.index)
        
        tprint("✅ Entropy features calculated")
        return df
    
    def _add_spectral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add spectral features."""
        if 'close' not in df.columns:
            return df
        
        close = df['close']
        returns = close.pct_change().dropna()
        
        # Rolling spectral features
        window = 50
        spectral_features = []
        
        for i in range(len(returns)):
            if i < window - 1:
                spectral_features.append({'dominant_freq': np.nan, 'spectral_centroid': np.nan})
            else:
                window_returns = returns.iloc[i-window+1:i+1]
                # Calculate FFT
                fft = np.fft.fft(window_returns)
                freqs = np.fft.fftfreq(len(window_returns))
                
                # Find dominant frequency
                power_spectrum = np.abs(fft) ** 2
                dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1
                dominant_freq = freqs[dominant_freq_idx]
                
                # Calculate spectral centroid
                spectral_centroid = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
                
                spectral_features.append({
                    'dominant_freq': dominant_freq,
                    'spectral_centroid': spectral_centroid
                })
        
        df['dominant_freq'] = pd.Series([f['dominant_freq'] for f in spectral_features], index=returns.index)
        df['spectral_centroid'] = pd.Series([f['spectral_centroid'] for f in spectral_features], index=returns.index)
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features."""
        if 'close' not in df.columns:
            return df
        
        close = df['close']
        
        # Time-based features
        df['hour'] = df.index.hour if hasattr(df.index, 'hour') else 0
        df['day_of_week'] = df.index.dayofweek if hasattr(df.index, 'dayofweek') else 0
        df['month'] = df.index.month if hasattr(df.index, 'month') else 0
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Rolling time-based features
        df['close_ma_ratio'] = close / close.rolling(20).mean()
        df['close_std_ratio'] = close / close.rolling(20).std()
        
        return df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add regime-specific features."""
        if 'close' not in df.columns:
            return df
        
        close = df['close']
        returns = close.pct_change()
        
        # Regime detection features
        df['volatility_regime'] = returns.rolling(20).std() > returns.rolling(50).std()
        df['trend_regime'] = close.rolling(20).mean() > close.rolling(50).mean()
        
        # Regime persistence
        df['volatility_regime_persistence'] = df['volatility_regime'].rolling(10).sum()
        df['trend_regime_persistence'] = df['trend_regime'].rolling(10).sum()
        
        # Regime transition features
        df['volatility_regime_changes'] = df['volatility_regime'].diff().abs()
        df['trend_regime_changes'] = df['trend_regime'].diff().abs()
        
        return df
    
    def _add_feature_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add feature interactions."""
        if len(df.columns) < 2:
            return df
        
        # Select numeric columns for interactions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return df
        
        # Create interaction features
        interaction_count = 0
        for i, col1 in enumerate(numeric_cols):
            if interaction_count >= self.config.max_interaction_features:
                break
            
            for col2 in numeric_cols[i+1:]:
                if interaction_count >= self.config.max_interaction_features:
                    break
                
                # Create interaction feature
                interaction_name = f"{col1}_x_{col2}"
                df[interaction_name] = df[col1] * df[col2]
                interaction_count += 1
        
        return df


class FeaturePreprocessor:
    """Preprocessor for feature normalization and outlier handling."""
    
    def __init__(self, method: str = "robust", outlier_threshold: float = 3.0):
        self.method = method
        self.outlier_threshold = outlier_threshold
        self.scaler = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def fit_transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessor and transform features."""
        # Handle missing values
        features_clean = features_df.fillna(features_df.median())
        
        # Handle outliers
        if self.outlier_threshold > 0:
            features_clean = self._handle_outliers(features_clean)
        
        # Initialize scaler
        if self.method == "standard":
            self.scaler = StandardScaler()
        elif self.method == "robust":
            self.scaler = RobustScaler()
        elif self.method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            self.logger.warning(f"Unknown preprocessing method: {self.method}, using robust")
            self.scaler = RobustScaler()
        
        # Fit and transform
        features_scaled = self.scaler.fit_transform(features_clean)
        features_scaled_df = pd.DataFrame(
            features_scaled,
            columns=features_clean.columns,
            index=features_clean.index
        )
        
        return features_scaled_df
    
    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted preprocessor."""
        if self.scaler is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        # Handle missing values
        features_clean = features_df.fillna(features_df.median())
        
        # Transform
        features_scaled = self.scaler.transform(features_clean)
        features_scaled_df = pd.DataFrame(
            features_scaled,
            columns=features_clean.columns,
            index=features_clean.index
        )
        
        return features_scaled_df
    
    def _handle_outliers(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using Z-score method."""
        features_clean = features_df.copy()
        
        for column in features_clean.columns:
            if features_clean[column].dtype in [np.number]:
                z_scores = np.abs(stats.zscore(features_clean[column].dropna()))
                outlier_mask = z_scores > self.outlier_threshold
                
                if outlier_mask.any():
                    # Cap outliers at threshold
                    median = features_clean[column].median()
                    std = features_clean[column].std()
                    upper_bound = median + self.outlier_threshold * std
                    lower_bound = median - self.outlier_threshold * std
                    
                    features_clean[column] = features_clean[column].clip(
                        lower=lower_bound, upper=upper_bound
                    )
        
        return features_clean


class EnhancedFeatureEngineeringPipeline:
    """Enhanced feature engineering pipeline with intelligent selection and validation."""
    
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config
        self.validator = FeatureValidator(config.correlation_threshold)
        self.selector = FeatureSelector(
            config.feature_selection_method, 
            config.max_features, 
            config.enable_feature_selection
        )
        self.generator = AdvancedFeatureGenerator(config)
        self.preprocessor = FeaturePreprocessor(config.preprocessing_method, config.outlier_threshold)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.hardware_manager = UnifiedHardwareManager()
        self.memory_optimizer = M1MemoryOptimizer()
        
        tprint("🚀 Enhanced feature engineering pipeline initialized")
        tprint_data_format("Pipeline configuration", config.__dict__)
    
    def process_features(self, data_df: pd.DataFrame, 
                        target: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process features through the complete pipeline with comprehensive logging.
        
        Args:
            data_df: Input data DataFrame
            target: Optional target variable for supervised feature selection
            
        Returns:
            Tuple of (processed_features, processing_info)
        """
        tprint("🚀 Starting enhanced feature processing pipeline")
        tprint_data_preview(data_df, "Input data")
        
        start_time = time.time()
        processing_info = {
            'original_features': len(data_df.columns),
            'processing_steps': [],
            'validation_results': {},
            'selection_results': {},
            'preprocessing_method': self.config.preprocessing_method,
            'processing_time': 0.0,
            'vectorbt_optimizations': self.generator.vectorbt_available
        }
        
        # Step 1: Generate features
        if any([
            self.config.enable_technical_indicators,
            self.config.enable_volatility_features,
            self.config.enable_momentum_features,
            self.config.enable_entropy_features,
            self.config.enable_spectral_features,
            self.config.enable_temporal_features,
            self.config.enable_regime_features,
            self.config.enable_feature_interactions
        ]):
            tprint("📊 Step 1: VectorBT-optimized feature generation")
            data_df = self.generator.generate_features(data_df)
            processing_info['processing_steps'].append('feature_generation')
            processing_info['generated_features'] = len(data_df.columns)
            tprint_data_preview(data_df, "After feature generation")
        
        # Step 2: Validate features
        tprint("🔍 Step 2: Feature validation")
        validation_results = self.validator.validate_features(data_df)
        processing_info['validation_results'] = validation_results
        
        if not validation_results['passed']:
            tprint(f"⚠️ Feature validation failed: {validation_results['issues']}")
            # Clean problematic features
            data_df = self.validator.remove_problematic_features(data_df)
            processing_info['processing_steps'].append('feature_cleaning')
            tprint_data_preview(data_df, "After feature cleaning")
        
        # Step 3: Select features
        if self.config.enable_feature_selection:
            tprint("🎯 Step 3: Intelligent feature selection")
            selected_features = self.selector.select_features(data_df, target)
            data_df = data_df[selected_features]
            processing_info['selection_results'] = {
                'method': self.config.feature_selection_method,
                'selected_features': len(selected_features),
                'feature_names': selected_features
            }
            processing_info['processing_steps'].append('feature_selection')
            tprint_data_preview(data_df, "After feature selection")
        
        # Step 4: Preprocess features
        tprint("🔧 Step 4: Feature preprocessing")
        data_df = self.preprocessor.fit_transform(data_df)
        processing_info['processing_steps'].append('preprocessing')
        tprint_data_preview(data_df, "After preprocessing")
        
        # Final validation
        tprint("✅ Step 5: Final validation")
        final_validation = self.validator.validate_features(data_df)
        processing_info['final_validation'] = final_validation
        
        processing_info['final_features'] = len(data_df.columns)
        processing_info['processing_time'] = time.time() - start_time
        
        tprint_performance(f"Feature processing completed in {processing_info['processing_time']:.2f}s")
        tprint(f"📊 Final feature count: {processing_info['final_features']}")
        
        return data_df, processing_info


def create_feature_engineering_pipeline(config: FeatureEngineeringConfig) -> EnhancedFeatureEngineeringPipeline:
    """Factory function to create a feature engineering pipeline."""
    return EnhancedFeatureEngineeringPipeline(config)