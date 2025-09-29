"""
Shared feature preparation utilities for NAS-TAS regime detection.

This module provides common feature engineering functionality that eliminates
redundancy between NAS and TAS components.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import time
import psutil
from src.utils.tprint import tprint, tprint_debug, tprint_success, tprint_warning, tprint_error


@dataclass
class FeatureConfig:
    """Configuration for feature preparation."""
    # Feature categories to include
    feature_categories: List[str] = None
    
    # Lookback periods for various features
    returns_lookback: int = 1
    volatility_lookback: int = 20
    ma_short_lookback: int = 10
    ma_long_lookback: int = 30
    volume_lookback: int = 10
    momentum_lookback: int = 14
    
    # Feature processing options
    use_standardized_features: bool = True
    drop_highly_correlated: bool = True
    correlation_threshold: float = 0.95
    
    # Data validation
    handle_missing_values: bool = True
    min_observations: int = 100
    
    def __post_init__(self):
        if self.feature_categories is None:
            self.feature_categories = ['momentum', 'volatility', 'volume', 'trend', 'price_action']


def prepare_market_features(
    market_data: pd.DataFrame,
    feature_config: Optional[FeatureConfig] = None,
    verbose: bool = False
) -> Optional[np.ndarray]:
    """
    Prepare market features for regime detection and clustering.
    
    This function consolidates the feature preparation logic used by both
    NAS and TAS components, eliminating code duplication.
    
    Args:
        market_data: Market data DataFrame with OHLCV columns
        feature_config: Configuration for feature preparation
        verbose: Whether to enable verbose logging
        
    Returns:
        Feature array or None if preparation fails
        
    Raises:
        ValueError: If market data is invalid or insufficient
    """
    if feature_config is None:
        feature_config = FeatureConfig()
    
    feature_prep_start = time.time()
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    if verbose:
        tprint("🔧 [SHARED_FEATURES] ===== FEATURE PREPARATION =====", color="blue", bold=True)
        tprint_debug(f"📊 [SHARED_FEATURES] Market data shape: {market_data.shape}")
        tprint_debug(f"📊 [SHARED_FEATURES] Available columns: {list(market_data.columns)}")
    
    try:
        # Validate input data
        if market_data is None or market_data.empty:
            if verbose:
                tprint_error("❌ [SHARED_FEATURES] Market data is None or empty")
            raise ValueError("Market data is None or empty")
        
        if len(market_data) < feature_config.min_observations:
            if verbose:
                tprint_error(f"❌ [SHARED_FEATURES] Insufficient data: {len(market_data)} < {feature_config.min_observations}")
            raise ValueError(f"Insufficient data: {len(market_data)} < {feature_config.min_observations}")
        
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in market_data.columns]
        if missing_columns:
            if verbose:
                tprint_error(f"❌ [SHARED_FEATURES] Missing required columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if verbose:
            tprint_debug(f"✅ [SHARED_FEATURES] Data validation passed")
        
        # Initialize features dictionary
        features_dict = {}
        
        # Calculate returns
        if 'momentum' in feature_config.feature_categories:
            if verbose:
                tprint_debug("📈 [SHARED_FEATURES] Calculating returns")
            features_dict['returns'] = market_data['close'].pct_change(feature_config.returns_lookback)
        
        # Calculate volatility
        if 'volatility' in feature_config.feature_categories:
            if verbose:
                tprint_debug("📊 [SHARED_FEATURES] Calculating volatility")
            returns = market_data['close'].pct_change()
            features_dict['volatility'] = returns.rolling(window=feature_config.volatility_lookback).std()
        
        # Calculate moving average ratios
        if 'trend' in feature_config.feature_categories:
            if verbose:
                tprint_debug("📈 [SHARED_FEATURES] Calculating moving average ratios")
            ma_short = market_data['close'].rolling(window=feature_config.ma_short_lookback).mean()
            ma_long = market_data['close'].rolling(window=feature_config.ma_long_lookback).mean()
            features_dict['ma_ratio'] = ma_short / ma_long
            features_dict['ma_spread'] = (ma_short - ma_long) / ma_long
        
        # Calculate volume ratios
        if 'volume' in feature_config.feature_categories:
            if verbose:
                tprint_debug("📊 [SHARED_FEATURES] Calculating volume ratios")
            volume_ma = market_data['volume'].rolling(window=feature_config.volume_lookback).mean()
            features_dict['volume_ratio'] = market_data['volume'] / volume_ma
            features_dict['volume_change'] = market_data['volume'].pct_change()
        
        # Calculate high-low spread
        if 'price_action' in feature_config.feature_categories:
            if verbose:
                tprint_debug("📊 [SHARED_FEATURES] Calculating price action features")
            features_dict['hl_spread'] = (market_data['high'] - market_data['low']) / market_data['close']
            features_dict['oc_spread'] = (market_data['close'] - market_data['open']) / market_data['open']
            features_dict['body_size'] = abs(market_data['close'] - market_data['open']) / (market_data['high'] - market_data['low'])
        
        # Calculate momentum indicators
        if 'momentum' in feature_config.feature_categories:
            if verbose:
                tprint_debug("🚀 [SHARED_FEATURES] Calculating momentum indicators")
            # RSI-like momentum
            returns = market_data['close'].pct_change()
            positive_returns = returns.where(returns > 0, 0).rolling(window=feature_config.momentum_lookback).mean()
            negative_returns = (-returns.where(returns < 0, 0)).rolling(window=feature_config.momentum_lookback).mean()
            features_dict['momentum'] = positive_returns / (positive_returns + negative_returns)
            
            # Price momentum
            features_dict['price_momentum'] = (market_data['close'] - market_data['close'].shift(feature_config.momentum_lookback)) / market_data['close'].shift(feature_config.momentum_lookback)
        
        # Handle missing values
        if feature_config.handle_missing_values:
            if verbose:
                tprint_debug("🔧 [SHARED_FEATURES] Handling missing values")
            for feature_name, feature_data in features_dict.items():
                features_dict[feature_name] = feature_data.fillna(method='ffill').fillna(0)
        
        # Convert to DataFrame for easier manipulation
        features_df = pd.DataFrame(features_dict)
        
        if verbose:
            tprint_debug(f"📊 [SHARED_FEATURES] Features DataFrame shape: {features_df.shape}")
            tprint_debug(f"📊 [SHARED_FEATURES] Feature columns: {list(features_df.columns)}")
        
        # Remove highly correlated features
        if feature_config.drop_highly_correlated:
            if verbose:
                tprint_debug("🔧 [SHARED_FEATURES] Removing highly correlated features")
            features_df = _remove_correlated_features(features_df, feature_config.correlation_threshold)
            if verbose:
                tprint_debug(f"📊 [SHARED_FEATURES] Features after correlation removal: {features_df.shape}")
        
        # Standardize features if requested
        if feature_config.use_standardized_features:
            if verbose:
                tprint_debug("🔧 [SHARED_FEATURES] Standardizing features")
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            features_array = scaler.fit_transform(features_df.fillna(0))
        else:
            features_array = features_df.fillna(0).values
        
        # Final validation
        if np.isnan(features_array).any() or np.isinf(features_array).any():
            if verbose:
                tprint_warning("⚠️ [SHARED_FEATURES] Features contain NaN or Inf values, cleaning...")
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        feature_prep_time = time.time() - feature_prep_start
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        if verbose:
            tprint_success(f"✅ [SHARED_FEATURES] Features prepared: {features_array.shape} in {feature_prep_time:.3f}s")
            tprint_debug(f"📊 [SHARED_FEATURES] Feature array memory usage: {features_array.nbytes / 1024 / 1024:.1f} MB")
            tprint_debug(f"📊 [SHARED_FEATURES] Memory used: {memory_used:.1f} MB")
            tprint_debug(f"📊 [SHARED_FEATURES] Feature statistics:")
            tprint_debug(f"   - Mean: {np.mean(features_array):.6f}")
            tprint_debug(f"   - Std: {np.std(features_array):.6f}")
            tprint_debug(f"   - Min: {np.min(features_array):.6f}")
            tprint_debug(f"   - Max: {np.max(features_array):.6f}")
        
        return features_array
        
    except Exception as e:
        feature_prep_time = time.time() - feature_prep_start
        if verbose:
            tprint_error(f"❌ [SHARED_FEATURES] Feature preparation failed: {e}")
        raise ValueError(f"Feature preparation failed: {e}")


def _remove_correlated_features(features_df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """
    Remove highly correlated features to reduce redundancy.
    
    Args:
        features_df: DataFrame with features
        threshold: Correlation threshold for removal
        
    Returns:
        DataFrame with correlated features removed
    """
    try:
        # Calculate correlation matrix
        corr_matrix = features_df.corr().abs()
        
        # Find pairs of highly correlated features
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        
        # Drop correlated features
        if to_drop:
            features_df = features_df.drop(columns=to_drop)
        
        return features_df
        
    except Exception as e:
        # If correlation removal fails, return original DataFrame
        return features_df


def get_feature_names(feature_config: FeatureConfig) -> List[str]:
    """
    Get the expected feature names for a given configuration.
    
    Args:
        feature_config: Feature configuration
        
    Returns:
        List of expected feature names
    """
    feature_names = []
    
    if 'momentum' in feature_config.feature_categories:
        feature_names.extend(['returns', 'momentum', 'price_momentum'])
    
    if 'volatility' in feature_config.feature_categories:
        feature_names.append('volatility')
    
    if 'trend' in feature_config.feature_categories:
        feature_names.extend(['ma_ratio', 'ma_spread'])
    
    if 'volume' in feature_config.feature_categories:
        feature_names.extend(['volume_ratio', 'volume_change'])
    
    if 'price_action' in feature_config.feature_categories:
        feature_names.extend(['hl_spread', 'oc_spread', 'body_size'])
    
    return feature_names


def validate_features(features: np.ndarray, expected_shape: Optional[tuple] = None) -> bool:
    """
    Validate feature array quality.
    
    Args:
        features: Feature array to validate
        expected_shape: Expected shape (n_samples, n_features)
        
    Returns:
        True if features are valid, False otherwise
    """
    try:
        # Check if features is None or empty
        if features is None or features.size == 0:
            return False
        
        # Check shape if provided
        if expected_shape is not None:
            if features.shape != expected_shape:
                return False
        
        # Check for NaN or Inf values
        if np.isnan(features).any() or np.isinf(features).any():
            return False
        
        # Check for constant features (all same value)
        if np.all(features == features[0]):
            return False
        
        return True
        
    except Exception:
        return False