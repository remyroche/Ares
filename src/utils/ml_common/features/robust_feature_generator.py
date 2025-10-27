"""
Robust Feature Generator with Fast Fail.

This module provides robust feature generation with fast fail behavior
instead of fallback mechanisms.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple, List, Dict, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class FeatureGenerationError(Exception):
    """Exception raised when feature generation fails."""
    pass


class BaseFeatureGenerator(ABC):
    """Base class for feature generators."""
    
    @abstractmethod
    def generate(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Generate features from data.
        
        Args:
            data: Input data
            
        Returns:
            Feature matrix and feature names
            
        Raises:
            FeatureGenerationError: If generation fails
        """
        pass
    
    @abstractmethod
    def get_min_features(self) -> int:
        """Get minimum number of features required."""
        pass


class TechnicalIndicatorGenerator(BaseFeatureGenerator):
    """Generate basic technical indicators."""
    
    def __init__(self, min_features: int = 20):
        self.min_features = min_features
    
    def generate(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Generate technical indicators."""
        try:
            features = []
            feature_names = []
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise FeatureGenerationError(f"Missing required columns: {missing_cols}")
            
            # Price-based features
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            volume = data['volume'].values
            
            # Returns
            returns = np.diff(close) / close[:-1]
            features.append(returns)
            feature_names.append('returns')
            
            # Log returns
            log_returns = np.diff(np.log(close))
            features.append(log_returns)
            feature_names.append('log_returns')
            
            # Price ratios
            hl_ratio = high / low
            oc_ratio = data['open'] / close
            features.extend([hl_ratio, oc_ratio])
            feature_names.extend(['hl_ratio', 'oc_ratio'])
            
            # Moving averages
            for window in [5, 10, 20]:
                if len(close) > window:
                    ma = pd.Series(close).rolling(window=window).mean().values
                    features.append(ma)
                    feature_names.append(f'ma_{window}')
                    
                    # Price relative to MA
                    price_ma_ratio = close / ma
                    features.append(price_ma_ratio)
                    feature_names.append(f'price_ma_ratio_{window}')
            
            # Volatility features
            for window in [5, 10, 20]:
                if len(returns) > window:
                    volatility = pd.Series(returns).rolling(window=window).std().values
                    features.append(volatility)
                    feature_names.append(f'volatility_{window}')
            
            # Volume features
            volume_ma = pd.Series(volume).rolling(window=10).mean().values
            volume_ratio = volume / volume_ma
            features.append(volume_ratio)
            feature_names.append('volume_ratio')
            
            # RSI
            if len(close) > 14:
                rsi = self._calculate_rsi(close)
                features.append(rsi)
                feature_names.append('rsi')
            
            # Combine all features
            feature_matrix = np.column_stack(features)
            
            # Remove NaN values
            valid_mask = ~np.isnan(feature_matrix).any(axis=1)
            feature_matrix = feature_matrix[valid_mask]
            
            if len(feature_matrix) < 10:
                raise FeatureGenerationError(f"Insufficient valid samples: {len(feature_matrix)}")
            
            if feature_matrix.shape[1] < self.min_features:
                raise FeatureGenerationError(f"Insufficient features: {feature_matrix.shape[1]} < {self.min_features}")
            
            logger.info(f"Generated {feature_matrix.shape[1]} technical indicators")
            return feature_matrix, feature_names
            
        except Exception as e:
            raise FeatureGenerationError(f"Technical indicator generation failed: {e}")
    
    def _calculate_rsi(self, prices: np.ndarray, window: int = 14) -> np.ndarray:
        """Calculate RSI indicator."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).rolling(window=window).mean().values
        avg_losses = pd.Series(losses).rolling(window=window).mean().values
        
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_min_features(self) -> int:
        return self.min_features


class RegimeFeatureGenerator(BaseFeatureGenerator):
    """Generate regime-specific features."""
    
    def __init__(self, min_features: int = 15):
        self.min_features = min_features
    
    def generate(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Generate regime-specific features."""
        try:
            features = []
            feature_names = []
            
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            volume = data['volume'].values
            
            # Regime transition features
            price_changes = np.diff(close)
            features.append(price_changes)
            feature_names.append('price_changes')
            
            # Volatility clustering
            returns = np.diff(close) / close[:-1]
            abs_returns = np.abs(returns)
            features.append(abs_returns)
            feature_names.append('abs_returns')
            
            # Trend strength
            for window in [5, 10, 20]:
                if len(close) > window:
                    trend = pd.Series(close).rolling(window=window).apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0]
                    ).values
                    features.append(trend)
                    feature_names.append(f'trend_strength_{window}')
            
            # Regime persistence features
            for window in [3, 5, 10]:
                if len(close) > window:
                    # Rolling correlation with lagged prices
                    corr = pd.Series(close).rolling(window=window).corr(
                        pd.Series(close).shift(1)
                    ).values
                    features.append(corr)
                    feature_names.append(f'price_autocorr_{window}')
            
            # Volume-price relationship
            volume_price_corr = pd.Series(volume).rolling(window=10).corr(
                pd.Series(close)
            ).values
            features.append(volume_price_corr)
            feature_names.append('volume_price_corr')
            
            # Combine all features
            feature_matrix = np.column_stack(features)
            
            # Remove NaN values
            valid_mask = ~np.isnan(feature_matrix).any(axis=1)
            feature_matrix = feature_matrix[valid_mask]
            
            if len(feature_matrix) < 10:
                raise FeatureGenerationError(f"Insufficient valid samples: {len(feature_matrix)}")
            
            if feature_matrix.shape[1] < self.min_features:
                raise FeatureGenerationError(f"Insufficient features: {feature_matrix.shape[1]} < {self.min_features}")
            
            logger.info(f"Generated {feature_matrix.shape[1]} regime features")
            return feature_matrix, feature_names
            
        except Exception as e:
            raise FeatureGenerationError(f"Regime feature generation failed: {e}")
    
    def get_min_features(self) -> int:
        return self.min_features


class RobustFeatureGenerator:
    """
    Robust feature generator with fast fail behavior.
    """
    
    def __init__(self, min_total_features: int = 50, min_samples: int = 100):
        """
        Initialize robust feature generator.
        
        Args:
            min_total_features: Minimum total features required
            min_samples: Minimum samples required
        """
        self.min_total_features = min_total_features
        self.min_samples = min_samples
        
        # Initialize generators
        self.generators = [
            TechnicalIndicatorGenerator(min_features=20),
            RegimeFeatureGenerator(min_features=15)
        ]
    
    def generate_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Generate features with fast fail behavior.
        
        Args:
            data: Input data
            
        Returns:
            Feature matrix and feature names
            
        Raises:
            FeatureGenerationError: If feature generation fails
        """
        logger.info("Starting robust feature generation")
        
        # Validate input data
        self._validate_input_data(data)
        
        all_features = []
        all_feature_names = []
        
        # Try each generator
        for generator in self.generators:
            try:
                features, names = generator.generate(data)
                all_features.append(features)
                all_feature_names.extend(names)
                logger.info(f"Successfully generated {features.shape[1]} features from {generator.__class__.__name__}")
            except FeatureGenerationError as e:
                logger.error(f"Feature generator {generator.__class__.__name__} failed: {e}")
                raise  # Fast fail - don't try other generators
        
        # Combine all features
        if not all_features:
            raise FeatureGenerationError("No features generated by any generator")
        
        # Ensure all feature matrices have the same number of rows
        min_rows = min(f.shape[0] for f in all_features)
        if min_rows < self.min_samples:
            raise FeatureGenerationError(f"Insufficient samples after feature generation: {min_rows} < {self.min_samples}")
        
        # Truncate all features to the same length
        truncated_features = [f[:min_rows] for f in all_features]
        feature_matrix = np.column_stack(truncated_features)
        
        # Validate final result
        if feature_matrix.shape[1] < self.min_total_features:
            raise FeatureGenerationError(
                f"Insufficient total features: {feature_matrix.shape[1]} < {self.min_total_features}"
            )
        
        logger.info(f"Successfully generated {feature_matrix.shape[1]} features from {len(all_features)} generators")
        return feature_matrix, all_feature_names
    
    def _validate_input_data(self, data: pd.DataFrame):
        """Validate input data."""
        if data is None:
            raise FeatureGenerationError("Input data is None")
        
        if len(data) < self.min_samples:
            raise FeatureGenerationError(f"Insufficient data: {len(data)} < {self.min_samples}")
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise FeatureGenerationError(f"Missing required columns: {missing_cols}")
        
        # Check for sufficient non-null data
        for col in required_cols:
            null_count = data[col].isnull().sum()
            if null_count > len(data) * 0.1:  # More than 10% nulls
                raise FeatureGenerationError(f"Too many null values in column {col}: {null_count}")


def generate_features_fast_fail(data: pd.DataFrame, 
                               min_total_features: int = 50,
                               min_samples: int = 100) -> Tuple[np.ndarray, List[str]]:
    """
    Generate features with fast fail behavior.
    
    Args:
        data: Input data
        min_total_features: Minimum total features required
        min_samples: Minimum samples required
        
    Returns:
        Feature matrix and feature names
        
    Raises:
        FeatureGenerationError: If feature generation fails
    """
    generator = RobustFeatureGenerator(min_total_features, min_samples)
    return generator.generate_features(data)