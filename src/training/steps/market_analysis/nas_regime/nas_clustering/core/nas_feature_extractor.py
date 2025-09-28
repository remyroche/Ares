"""
NAS Feature Extractor

Extracts and enhances features for neural architecture search regime detection.
Now uses shared balanced feature extraction to prevent clustering imbalance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import time
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

# Import shared balanced feature extractor
try:
    from src.training.steps.market_analysis.shared_utils.balanced_feature_extractor import (
        BalancedFeatureExtractor, BalancedFeatureConfig, create_nas_config
    )
    BALANCED_FEATURES_AVAILABLE = True
except ImportError:
    BALANCED_FEATURES_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class FeatureExtractionResult:
    """Result of feature extraction."""
    features: np.ndarray
    feature_names: List[str]
    extraction_metrics: Dict[str, float]
    execution_time: float

class NASFeatureExtractor:
    """
    NAS Feature Extractor for regime detection.
    
    Extracts and enhances features for neural architecture search.
    """
    
    def __init__(self, enable_hardware_optimization: bool = True, enable_matrix_optimization: bool = True):
        """
        Initialize the NAS Feature Extractor.
        
        Args:
            enable_hardware_optimization: Whether to enable hardware optimization
            enable_matrix_optimization: Whether to enable matrix optimization
        """
        self.enable_hardware_optimization = enable_hardware_optimization
        self.enable_matrix_optimization = enable_matrix_optimization
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize feature extraction components
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.pca = None
        
        self.logger.info("NAS Feature Extractor initialized")
    
    def extract_features(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract balanced features using shared utility to prevent clustering imbalance.
        
        Args:
            data: Input data for feature extraction
            labels: Optional labels for supervised feature selection
            
        Returns:
            Enhanced features array with balanced distributions
        """
        start_time = time.time()
        self.logger.info(f"Starting balanced feature extraction for data shape: {data.shape}")
        
        try:
            if BALANCED_FEATURES_AVAILABLE:
                # Use shared balanced feature extractor
                config = create_nas_config()
                extractor = BalancedFeatureExtractor(config)
                result = extractor.extract_balanced_features(data, labels)
                
                if result.success:
                    execution_time = time.time() - start_time
                    self.logger.info(f"Balanced feature extraction completed in {execution_time:.2f}s")
                    self.logger.info(f"Final shape: {result.features.shape}")
                    self.logger.info(f"Balance metrics: {result.balance_metrics}")
                    return result.features
                else:
                    self.logger.warning(f"Balanced feature extraction failed: {result.error_message}")
                    # Fallback to original method
                    return self._extract_features_fallback(data, labels)
            else:
                self.logger.warning("Balanced features not available, using fallback method")
                return self._extract_features_fallback(data, labels)
                
        except Exception as e:
            self.logger.error(f"Balanced feature extraction failed: {e}")
            return self._extract_features_fallback(data, labels)
    
    def _extract_features_fallback(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> np.ndarray:
        """Fallback feature extraction method."""
        try:
            # Basic feature extraction pipeline
            features = self._extract_basic_features(data)
            
            # Statistical features
            features = self._extract_statistical_features(data, features)
            
            # Technical features (if applicable)
            features = self._extract_technical_features(data, features)
            
            # Feature scaling
            features = self._scale_features(features)
            
            # Feature selection (if labels provided)
            if labels is not None and len(labels) == len(features):
                features = self._select_features(features, labels)
            
            # Dimensionality reduction (if needed)
            if features.shape[1] > 100:  # Reduce if too many features
                features = self._reduce_dimensions(features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Fallback feature extraction failed: {e}")
            return data  # Return original data as fallback
    
    def _extract_basic_features(self, data: np.ndarray) -> np.ndarray:
        """Extract balanced basic features from the data."""
        try:
            features = []
            
            # Original features (normalized to prevent scale issues)
            normalized_data = self._normalize_data(data)
            features.append(normalized_data)
            
            # Quantile-based features (more balanced than rolling averages)
            quantile_features = self._extract_quantile_features(data)
            if quantile_features is not None:
                features.append(quantile_features)
            
            # Balanced price differences (normalized)
            if len(data) > 1:
                normalized_diff = self._extract_balanced_differences(data)
                features.append(normalized_diff)
            
            # Balanced price ratios (log-based to reduce extreme values)
            if len(data) > 1:
                log_ratios = self._extract_balanced_ratios(data)
                features.append(log_ratios)
            
            # Combine all features
            combined_features = np.concatenate(features, axis=1)
            
            self.logger.debug(f"Balanced basic features extracted. Shape: {combined_features.shape}")
            return combined_features
            
        except Exception as e:
            self.logger.warning(f"Balanced basic feature extraction failed: {e}")
            return data
    
    def _extract_statistical_features(self, data: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Extract statistical features."""
        try:
            statistical_features = []
            
            # Rolling statistics
            for window in [5, 10, 20]:
                if len(data) > window:
                    # Rolling mean
                    rolling_mean = self._rolling_statistic(data, window, np.mean)
                    statistical_features.append(rolling_mean)
                    
                    # Rolling std
                    rolling_std = self._rolling_statistic(data, window, np.std)
                    statistical_features.append(rolling_std)
                    
                    # Rolling min/max
                    rolling_min = self._rolling_statistic(data, window, np.min)
                    rolling_max = self._rolling_statistic(data, window, np.max)
                    statistical_features.append(rolling_min)
                    statistical_features.append(rolling_max)
            
            # Volatility measures
            if len(data) > 1:
                volatility = np.std(np.diff(data, axis=0), axis=0, keepdims=True)
                volatility_tiled = np.tile(volatility, (len(data), 1))
                statistical_features.append(volatility_tiled)
            
            # Skewness and kurtosis (if enough data)
            if len(data) > 10:
                try:
                    from scipy import stats
                    skewness = stats.skew(data, axis=0, keepdims=True)
                    kurtosis = stats.kurtosis(data, axis=0, keepdims=True)
                    skewness_tiled = np.tile(skewness, (len(data), 1))
                    kurtosis_tiled = np.tile(kurtosis, (len(data), 1))
                    statistical_features.append(skewness_tiled)
                    statistical_features.append(kurtosis_tiled)
                except ImportError:
                    self.logger.debug("SciPy not available for skewness/kurtosis")
            
            if statistical_features:
                combined_features = np.concatenate([features] + statistical_features, axis=1)
                self.logger.debug(f"Statistical features added. Shape: {combined_features.shape}")
                return combined_features
            else:
                return features
                
        except Exception as e:
            self.logger.warning(f"Statistical feature extraction failed: {e}")
            return features
    
    def _extract_technical_features(self, data: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Extract technical analysis features."""
        try:
            technical_features = []
            
            # RSI-like momentum
            for window in [7, 14]:
                if len(data) > window:
                    momentum = self._calculate_momentum(data, window)
                    technical_features.append(momentum)
            
            # Bollinger Bands-like features
            for window in [10, 20]:
                if len(data) > window:
                    bb_features = self._calculate_bollinger_bands(data, window)
                    technical_features.extend(bb_features)
            
            # MACD-like features
            if len(data) > 26:
                macd_features = self._calculate_macd(data)
                technical_features.extend(macd_features)
            
            if technical_features:
                combined_features = np.concatenate([features] + technical_features, axis=1)
                self.logger.debug(f"Technical features added. Shape: {combined_features.shape}")
                return combined_features
            else:
                return features
                
        except Exception as e:
            self.logger.warning(f"Technical feature extraction failed: {e}")
            return features
    
    def _scale_features(self, features: np.ndarray) -> np.ndarray:
        """Scale features using robust scaling."""
        try:
            # Use RobustScaler to handle outliers
            scaler = RobustScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Handle any remaining NaN or inf values
            scaled_features = np.nan_to_num(scaled_features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            self.logger.debug(f"Features scaled. Shape: {scaled_features.shape}")
            return scaled_features
            
        except Exception as e:
            self.logger.warning(f"Feature scaling failed: {e}")
            return features
    
    def _select_features(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Select most relevant features."""
        try:
            n_features = min(50, features.shape[1] // 2)  # Select top 50 or half of features
            if n_features < 10:
                n_features = min(10, features.shape[1])
            
            selector = SelectKBest(score_func=f_classif, k=n_features)
            selected_features = selector.fit_transform(features, labels)
            
            self.logger.debug(f"Feature selection completed. Selected {n_features} features")
            return selected_features
            
        except Exception as e:
            self.logger.warning(f"Feature selection failed: {e}")
            return features
    
    def _reduce_dimensions(self, features: np.ndarray) -> np.ndarray:
        """Reduce dimensionality using PCA."""
        try:
            n_components = min(50, features.shape[1], features.shape[0] - 1)
            
            pca = PCA(n_components=n_components)
            reduced_features = pca.fit_transform(features)
            
            explained_variance = np.sum(pca.explained_variance_ratio_)
            self.logger.debug(f"PCA completed. Reduced to {n_components} components, explained variance: {explained_variance:.3f}")
            
            return reduced_features
            
        except Exception as e:
            self.logger.warning(f"Dimensionality reduction failed: {e}")
            return features
    
    def _moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate moving average."""
        try:
            if len(data) < window:
                return np.tile(np.mean(data, axis=0, keepdims=True), (len(data), 1))
            
            ma = np.zeros_like(data)
            for i in range(len(data)):
                start_idx = max(0, i - window + 1)
                end_idx = i + 1
                ma[i] = np.mean(data[start_idx:end_idx], axis=0)
            
            return ma
        except Exception:
            return data
    
    def _rolling_statistic(self, data: np.ndarray, window: int, func) -> np.ndarray:
        """Calculate rolling statistic."""
        try:
            result = np.zeros_like(data)
            for i in range(len(data)):
                start_idx = max(0, i - window + 1)
                end_idx = i + 1
                result[i] = func(data[start_idx:end_idx], axis=0)
            return result
        except Exception:
            return data
    
    def _calculate_momentum(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate momentum indicator."""
        try:
            momentum = np.zeros_like(data)
            for i in range(len(data)):
                if i >= window:
                    momentum[i] = data[i] - data[i - window]
                else:
                    momentum[i] = 0.0
            return momentum
        except Exception:
            return np.zeros_like(data)
    
    def _calculate_bollinger_bands(self, data: np.ndarray, window: int) -> List[np.ndarray]:
        """Calculate Bollinger Bands features."""
        try:
            ma = self._moving_average(data, window)
            std = self._rolling_statistic(data, window, np.std)
            
            upper_band = ma + 2 * std
            lower_band = ma - 2 * std
            
            return [ma, upper_band, lower_band]
        except Exception:
            return []
    
    def _calculate_macd(self, data: np.ndarray) -> List[np.ndarray]:
        """Calculate MACD features."""
        try:
            if len(data) < 26:
                return []
            
            ema_12 = self._calculate_ema(data, 12)
            ema_26 = self._calculate_ema(data, 26)
            
            macd_line = ema_12 - ema_26
            signal_line = self._calculate_ema(macd_line, 9)
            histogram = macd_line - signal_line
            
            return [macd_line, signal_line, histogram]
        except Exception:
            return []
    
    def _calculate_ema(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate exponential moving average."""
        try:
            alpha = 2.0 / (window + 1)
            ema = np.zeros_like(data)
            ema[0] = data[0]
            
            for i in range(1, len(data)):
                ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
            
            return ema
        except Exception:
            return data
