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
    Prepare comprehensive market features for regime detection and clustering.
    
    This function uses the feature generation system to create comprehensive
    features from all specified categories, providing much richer feature sets
    than the basic implementation.
    
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
        tprint("🔧 [SHARED_FEATURES] ===== COMPREHENSIVE FEATURE PREPARATION =====", color="blue", bold=True)
        tprint_debug(f"📊 [SHARED_FEATURES] Market data shape: {market_data.shape}")
        tprint_debug(f"📊 [SHARED_FEATURES] Available columns: {list(market_data.columns)}")
        tprint_debug(f"📊 [SHARED_FEATURES] Feature categories: {feature_config.feature_categories}")
    
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
        
        # Use balanced feature extractor for comprehensive features
        if verbose:
            tprint_debug("🚀 [SHARED_FEATURES] Using balanced feature extractor for comprehensive features")
        
        try:
            # Use regime-focused feature generation instead of balanced extractor
            from src.feature_generation.categories.regime_feature_integration import (
                generate_regime_features, RegimeFeatureConfig
            )
            
            # Create regime-focused configuration
            regime_config = RegimeFeatureConfig(
                include_volatility_regime='regime_volatility' in feature_config.feature_categories,
                include_volume_regime='regime_volume' in feature_config.feature_categories,
                include_structural_trend='regime_structural_trend' in feature_config.feature_categories,
                include_statistical_regime='regime_statistical' in feature_config.feature_categories,
                min_regime_persistence=0.7,
                max_feature_noise_ratio=0.3,
                min_temporal_stability=0.6,
                optimize_for_15m=True,
                trade_duration_minutes=(5, 30)
            )
            
            if verbose:
                tprint_debug(f"📊 [SHARED_FEATURES] Extracting regime-focused features for categories: {feature_config.feature_categories}")
            
            # Generate regime-focused features
            features_dict, summary = generate_regime_features(
                data=market_data,
                config=regime_config
            )
            
            if not features_dict or len(features_dict) == 0:
                if verbose:
                    tprint_warning("⚠️ [SHARED_FEATURES] No regime features generated, falling back to basic features")
                # Fall back to basic feature generation
                features_df = _generate_basic_features(market_data, feature_config, verbose)
            else:
                # Convert regime features to DataFrame
                if verbose:
                    tprint_debug(f"✅ [SHARED_FEATURES] Generated {len(features_dict)} regime features")
                    tprint_debug(f"📊 [SHARED_FEATURES] Feature summary: {summary}")
                
                # Convert features dict to DataFrame
                features_array = np.column_stack(list(features_dict.values()))
                features_df = pd.DataFrame(
                    features_array, 
                    columns=list(features_dict.keys()),
                    index=market_data.index[:len(features_array)]
                )
            
            if verbose:
                tprint_debug(f"📊 [SHARED_FEATURES] Generated {len(features_df.columns)} features")
                tprint_debug(f"📊 [SHARED_FEATURES] Feature columns: {list(features_df.columns)}")
            
        except Exception as e:
            if verbose:
                tprint_warning(f"⚠️ [SHARED_FEATURES] Balanced feature extractor failed: {e}, falling back to basic features")
            # Fall back to basic feature generation
            features_df = _generate_basic_features(market_data, feature_config, verbose)
        
        # Convert to numpy array and handle missing values
        if verbose:
            tprint_debug("🔧 [SHARED_FEATURES] Handling missing values")
        
        # Fill missing values with forward fill, then backward fill
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')
        
        # Convert to numpy array
        features_array = features_df.values
        
        # Remove rows with any remaining NaN values
        valid_rows = ~np.isnan(features_array).any(axis=1)
        features_array = features_array[valid_rows]
        
        if len(features_array) == 0:
            if verbose:
                tprint_error("❌ [SHARED_FEATURES] No valid features after cleaning")
            raise ValueError("No valid features after cleaning")
        
        # Remove highly correlated features if requested
        if feature_config.drop_highly_correlated:
            if verbose:
                tprint_debug("🔧 [SHARED_FEATURES] Removing highly correlated features")
            
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(features_array.T)
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix)):
                for j in range(i+1, len(corr_matrix)):
                    if abs(corr_matrix[i, j]) > 0.95:  # High correlation threshold
                        high_corr_pairs.append((i, j))
            
            # Remove one feature from each highly correlated pair
            features_to_remove = set()
            for i, j in high_corr_pairs:
                if i not in features_to_remove:
                    features_to_remove.add(j)
            
            # Remove highly correlated features
            if features_to_remove:
                keep_indices = [i for i in range(features_array.shape[1]) if i not in features_to_remove]
                features_array = features_array[:, keep_indices]
                
                if verbose:
                    tprint_debug(f"📊 [SHARED_FEATURES] Removed {len(features_to_remove)} highly correlated features")
        
        # Standardize features if requested
        if feature_config.use_standardized_features:
            if verbose:
                tprint_debug("🔧 [SHARED_FEATURES] Standardizing features")
            
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            features_array = scaler.fit_transform(features_array)
        
        # Final validation
        if features_array.shape[0] < feature_config.min_observations:
            if verbose:
                tprint_error(f"❌ [SHARED_FEATURES] Insufficient valid observations: {features_array.shape[0]} < {feature_config.min_observations}")
            raise ValueError(f"Insufficient valid observations: {features_array.shape[0]} < {feature_config.min_observations}")
        
        # Performance metrics
        feature_prep_time = time.time() - feature_prep_start
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        if verbose:
            tprint_success(f"✅ [SHARED_FEATURES] Features prepared: {features_array.shape} in {feature_prep_time:.3f}s")
            tprint_debug(f"📊 [SHARED_FEATURES] Feature array memory usage: {features_array.nbytes / 1024 / 1024:.1f} MB")
            tprint_debug(f"📊 [SHARED_FEATURES] Memory used: {memory_used:.1f} MB")
            
            # Feature statistics
            if features_array.size > 0:
                tprint_debug(f"📊 [SHARED_FEATURES] Feature statistics:")
                tprint_debug(f"   - Mean: {np.mean(features_array):.6f}")
                tprint_debug(f"   - Std: {np.std(features_array):.6f}")
                tprint_debug(f"   - Min: {np.min(features_array):.6f}")
                tprint_debug(f"   - Max: {np.max(features_array):.6f}")
        
        return features_array
        
    except Exception as e:
        if verbose:
            tprint_error(f"❌ [SHARED_FEATURES] Feature preparation failed: {e}")
        raise


def _generate_basic_features(market_data: pd.DataFrame, feature_config: FeatureConfig, verbose: bool = False) -> pd.DataFrame:
    """
    Generate basic features as fallback when comprehensive feature generation fails.
    
    Args:
        market_data: Market data DataFrame
        feature_config: Feature configuration
        verbose: Whether to enable verbose logging
        
    Returns:
        DataFrame with basic features
    """
    if verbose:
        tprint_debug("🔧 [SHARED_FEATURES] Generating basic features as fallback")
    
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
    
    # Calculate price action features
    if 'price' in feature_config.feature_categories:
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
        features_dict['price_momentum'] = market_data['close'].pct_change(feature_config.momentum_lookback)
    
    # Calculate entropy features
    if 'entropy' in feature_config.feature_categories:
        if verbose:
            tprint_debug("🔍 [SHARED_FEATURES] Calculating entropy features")
        # Simple entropy measure based on price changes
        returns = market_data['close'].pct_change()
        returns_abs = abs(returns)
        features_dict['entropy'] = returns_abs.rolling(window=feature_config.entropy_lookback).std()
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_dict, index=market_data.index)
    
    if verbose:
        tprint_debug(f"📊 [SHARED_FEATURES] Basic features DataFrame shape: {features_df.shape}")
        tprint_debug(f"📊 [SHARED_FEATURES] Feature columns: {list(features_df.columns)}")
    
    return features_df


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