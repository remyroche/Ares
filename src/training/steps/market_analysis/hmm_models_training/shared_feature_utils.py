"""
Shared Enhanced Feature Utilities for HMM Training

This module provides consistent enhanced feature creation for both hmm_models_training 
and hmm_ensemble_training to ensure they use the same feature engineering pipeline.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from src.utils.tprint import tprint


class HMMEnhancedFeatureCreator:
    """
    Shared utility class for creating enhanced features for HMM training.
    
    This ensures both hmm_models_training and hmm_ensemble_training use the exact
    same feature engineering pipeline, eliminating inconsistencies.
    """
    
    @staticmethod
    def create_enhanced_features(X: np.ndarray, regime_labels: np.ndarray) -> np.ndarray:
        """
        Create enhanced features for HMM models training with no information leakage.
        
        Args:
            X: Original features (close_return, volume_return, price_range_pct)
            regime_labels: Regime labels
            
        Returns:
            Enhanced feature matrix with diverse, non-leaking features
        """
        try:
            enhanced_features = [X]
            n_samples = X.shape[0]
            
            # 1. MOMENTUM FEATURES (no lookahead bias)
            if X.shape[1] >= 1:  # close_return available
                close_return = X[:, 0]
                
                # RSI-like momentum (14-period)
                rsi_feature = HMMEnhancedFeatureCreator._calculate_rsi_from_returns(close_return, period=14)
                enhanced_features.append(rsi_feature.reshape(-1, 1))
                
                # MACD-like momentum
                macd_feature = HMMEnhancedFeatureCreator._calculate_macd_from_returns(close_return, fast=12, slow=26)
                enhanced_features.append(macd_feature.reshape(-1, 1))
            
            # 2. VOLATILITY FEATURES (rolling windows)
            if X.shape[1] >= 3:  # price_range_pct available
                price_range = X[:, 2]
                
                # Rolling volatility (10-period)
                vol_10 = pd.Series(price_range).rolling(10, min_periods=1).std().fillna(0).values
                enhanced_features.append(vol_10.reshape(-1, 1))
                
                # Volatility regime indicator
                vol_regime = HMMEnhancedFeatureCreator._calculate_volatility_regime(price_range, window=20)
                enhanced_features.append(vol_regime.reshape(-1, 1))
            
            # 3. VOLUME FEATURES (if available)
            if X.shape[1] >= 2:  # volume_return available
                volume_return = X[:, 1]
                
                # Volume momentum
                vol_momentum = pd.Series(volume_return).rolling(5, min_periods=1).mean().fillna(0).values
                enhanced_features.append(vol_momentum.reshape(-1, 1))
                
                # Volume trend strength
                vol_trend = HMMEnhancedFeatureCreator._calculate_volume_trend(volume_return, window=10)
                enhanced_features.append(vol_trend.reshape(-1, 1))
            
            # 4. LAGGED REGIME FEATURES (historical only)
            lagged_regime_features = HMMEnhancedFeatureCreator._create_lagged_regime_features(regime_labels)
            enhanced_features.append(lagged_regime_features)
            
            # Combine all features
            X_enhanced = np.hstack(enhanced_features)
            
            return X_enhanced
            
        except Exception as e:
            tprint(f"⚠️ Enhanced feature creation failed: {e}")
            # Fallback to original features
            return X
    
    @staticmethod
    def get_enhanced_feature_names(original_feature_count: int) -> List[str]:
        """
        Get names for enhanced features.
        
        Args:
            original_feature_count: Number of original features
            
        Returns:
            List of feature names
        """
        feature_names = []
        
        # Original features
        if original_feature_count >= 1:
            feature_names.append("close_return")
        if original_feature_count >= 2:
            feature_names.append("volume_return")
        if original_feature_count >= 3:
            feature_names.append("price_range_pct")
        
        # Enhanced features
        feature_names.extend([
            "rsi_momentum",
            "macd_momentum",
            "volatility_rolling_10",
            "volatility_regime",
            "volume_momentum",
            "volume_trend_strength",
            "regime_stability",
            "time_in_regime",
            "regime_transition_freq"
        ])
        
        return feature_names
    
    @staticmethod
    def create_enhanced_features_with_names(X: np.ndarray, regime_labels: np.ndarray, 
                                          original_feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Create enhanced features and return both the feature matrix and feature names.
        
        Args:
            X: Original features
            regime_labels: Regime labels
            original_feature_names: Names of original features
            
        Returns:
            Tuple of (enhanced_features, feature_names)
        """
        # Create enhanced features
        X_enhanced = HMMEnhancedFeatureCreator.create_enhanced_features(X, regime_labels)
        
        # Get feature names
        if original_feature_names is None:
            enhanced_feature_names = HMMEnhancedFeatureCreator.get_enhanced_feature_names(X.shape[1])
        else:
            # Use provided original feature names
            enhanced_feature_names = original_feature_names.copy()
            enhanced_feature_names.extend([
                "rsi_momentum",
                "macd_momentum", 
                "volatility_rolling_10",
                "volatility_regime",
                "volume_momentum",
                "volume_trend_strength",
                "regime_stability",
                "time_in_regime",
                "regime_transition_freq"
            ])
        
        return X_enhanced, enhanced_feature_names
    
    @staticmethod
    def _calculate_rsi_from_returns(returns: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI from returns without lookahead bias."""
        returns_series = pd.Series(returns)
        gains = returns_series.where(returns_series > 0, 0)
        losses = -returns_series.where(returns_series < 0, 0)
        
        avg_gains = gains.rolling(period, min_periods=1).mean()
        avg_losses = losses.rolling(period, min_periods=1).mean()
        
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50).values  # Neutral RSI for NaN values
    
    @staticmethod
    def _calculate_macd_from_returns(returns: np.ndarray, fast: int = 12, slow: int = 26) -> np.ndarray:
        """Calculate MACD from returns without lookahead bias."""
        returns_series = pd.Series(returns)
        
        ema_fast = returns_series.ewm(span=fast, min_periods=1).mean()
        ema_slow = returns_series.ewm(span=slow, min_periods=1).mean()
        
        macd = ema_fast - ema_slow
        return macd.fillna(0).values
    
    @staticmethod
    def _calculate_volatility_regime(price_range: np.ndarray, window: int = 20) -> np.ndarray:
        """Calculate volatility regime indicator."""
        price_range_series = pd.Series(price_range)
        rolling_vol = price_range_series.rolling(window, min_periods=1).std()
        vol_percentile = rolling_vol.rolling(window*2, min_periods=1).rank(pct=True)
        return vol_percentile.fillna(0.5).values
    
    @staticmethod
    def _calculate_volume_trend(volume_return: np.ndarray, window: int = 10) -> np.ndarray:
        """Calculate volume trend strength."""
        volume_series = pd.Series(volume_return)
        volume_sma = volume_series.rolling(window, min_periods=1).mean()
        trend_strength = (volume_series - volume_sma) / (volume_sma + 1e-10)
        return trend_strength.fillna(0).values
    
    @staticmethod
    def _create_lagged_regime_features(regime_labels: np.ndarray) -> np.ndarray:
        """Create lagged regime features without information leakage."""
        n_samples = len(regime_labels)
        lagged_features = np.zeros((n_samples, 3))
        
        for i in range(n_samples):
            if i >= 5:  # Need at least 5 historical points
                recent_regimes = regime_labels[max(0, i-5):i]
                
                # Regime stability (how often regime changed recently)
                lagged_features[i, 0] = len(np.unique(recent_regimes)) / len(recent_regimes)
                
                # Time in current regime
                current_regime = regime_labels[i-1] if i > 0 else regime_labels[0]
                time_in_regime = 0
                for j in range(i-1, -1, -1):
                    if regime_labels[j] == current_regime:
                        time_in_regime += 1
                    else:
                        break
                lagged_features[i, 1] = min(time_in_regime / 20.0, 1.0)  # Normalize
                
                # Regime transition frequency
                if len(recent_regimes) > 1:
                    transitions = np.sum(np.diff(recent_regimes) != 0)
                    lagged_features[i, 2] = transitions / (len(recent_regimes) - 1)
        
        return lagged_features


# Convenience functions for easy importing
def create_enhanced_features(X: np.ndarray, regime_labels: np.ndarray) -> np.ndarray:
    """Create enhanced features for HMM training."""
    return HMMEnhancedFeatureCreator.create_enhanced_features(X, regime_labels)


def get_enhanced_feature_names(original_feature_count: int) -> List[str]:
    """Get names for enhanced features."""
    return HMMEnhancedFeatureCreator.get_enhanced_feature_names(original_feature_count)


def create_enhanced_features_with_names(X: np.ndarray, regime_labels: np.ndarray, 
                                      original_feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """Create enhanced features with names."""
    return HMMEnhancedFeatureCreator.create_enhanced_features_with_names(X, regime_labels, original_feature_names)
