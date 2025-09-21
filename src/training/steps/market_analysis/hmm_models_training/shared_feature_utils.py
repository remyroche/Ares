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

                # Ensure consistent length for volatility
                if len(vol_10) != n_samples:
                    if len(vol_10) < n_samples:
                        padding = np.full(n_samples - len(vol_10), 0.0)
                        vol_10 = np.concatenate([padding, vol_10])
                    else:
                        vol_10 = vol_10[:n_samples]

                enhanced_features.append(vol_10.reshape(-1, 1))

                # Volatility regime indicator
                vol_regime = HMMEnhancedFeatureCreator._calculate_volatility_regime(price_range, window=20)
                enhanced_features.append(vol_regime.reshape(-1, 1))

            # 3. VOLUME FEATURES (if available)
            if X.shape[1] >= 2:  # volume_return available
                volume_return = X[:, 1]

                # Volume momentum
                vol_momentum = pd.Series(volume_return).rolling(5, min_periods=1).mean().fillna(0).values

                # Ensure consistent length for volume momentum
                if len(vol_momentum) != n_samples:
                    if len(vol_momentum) < n_samples:
                        padding = np.full(n_samples - len(vol_momentum), 0.0)
                        vol_momentum = np.concatenate([padding, vol_momentum])
                    else:
                        vol_momentum = vol_momentum[:n_samples]

                enhanced_features.append(vol_momentum.reshape(-1, 1))

                # Volume trend strength
                vol_trend = HMMEnhancedFeatureCreator._calculate_volume_trend(volume_return, window=10)
                enhanced_features.append(vol_trend.reshape(-1, 1))

            # 4. LAGGED REGIME FEATURES (historical only)
            lagged_regime_features = HMMEnhancedFeatureCreator._create_lagged_regime_features(regime_labels)
            enhanced_features.append(lagged_regime_features)

            # Validate all features have the same length before concatenation
            feature_lengths = [feat.shape[0] for feat in enhanced_features]
            expected_length = n_samples

            if not all(length == expected_length for length in feature_lengths):
                # Log the mismatch details
                mismatch_info = [(i, length) for i, length in enumerate(feature_lengths) if length != expected_length]
                tprint(f"⚠️ Feature length mismatch detected: {mismatch_info}")
                tprint(f"⚠️ Expected length: {expected_length}, Actual lengths: {feature_lengths}")

                # Try to fix by truncating or padding
                fixed_features = []
                for i, feat in enumerate(enhanced_features):
                    if feat.shape[0] < expected_length:
                        # Pad with zeros or appropriate neutral values
                        if feat.ndim == 1:
                            padding = np.zeros(expected_length - feat.shape[0])
                            fixed_feat = np.concatenate([padding, feat])
                        else:
                            padding = np.zeros((expected_length - feat.shape[0], feat.shape[1]))
                            fixed_feat = np.vstack([padding, feat])
                        fixed_features.append(fixed_feat)
                        tprint(f"📊 Fixed feature {i}: padded from {feat.shape[0]} to {expected_length}")
                    elif feat.shape[0] > expected_length:
                        # Truncate
                        fixed_feat = feat[:expected_length]
                        fixed_features.append(fixed_feat)
                        tprint(f"📊 Fixed feature {i}: truncated from {feat.shape[0]} to {expected_length}")
                    else:
                        fixed_features.append(feat)

                enhanced_features = fixed_features

            # Combine all features
            X_enhanced = np.hstack(enhanced_features)

            return X_enhanced
            
        except Exception as e:
            tprint(f"⚠️ Enhanced feature creation failed: {e}")
            # Fallback to original features
            return X
    
    @staticmethod
    def get_enhanced_feature_names(original_feature_count: int, original_feature_names: Optional[List[str]] = None) -> List[str]:
        """
        Get names for enhanced features.
        
        Args:
            original_feature_count: Number of original features
            original_feature_names: Optional names of original features
            
        Returns:
            List of feature names
        """
        feature_names = []
        
        # Use provided original feature names if available, otherwise use generic names
        if original_feature_names is not None and len(original_feature_names) >= original_feature_count:
            feature_names.extend(original_feature_names[:original_feature_count])
        else:
            # Generate generic feature names based on count
            for i in range(original_feature_count):
                feature_names.append(f"feature_{i}")
        
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
        
        # Get feature names using the updated method
        enhanced_feature_names = HMMEnhancedFeatureCreator.get_enhanced_feature_names(
            X.shape[1], original_feature_names
        )
        
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

        # Ensure consistent length and fill NaN values
        rsi_values = rsi.fillna(50).values  # Neutral RSI for NaN values

        # If length doesn't match input, pad or truncate to match
        if len(rsi_values) != len(returns):
            if len(rsi_values) < len(returns):
                # Pad with neutral value
                padding = np.full(len(returns) - len(rsi_values), 50.0)
                rsi_values = np.concatenate([padding, rsi_values])
            else:
                # Truncate to match input length
                rsi_values = rsi_values[:len(returns)]

        return rsi_values
    
    @staticmethod
    def _calculate_macd_from_returns(returns: np.ndarray, fast: int = 12, slow: int = 26) -> np.ndarray:
        """Calculate MACD from returns without lookahead bias."""
        returns_series = pd.Series(returns)

        ema_fast = returns_series.ewm(span=fast, min_periods=1).mean()
        ema_slow = returns_series.ewm(span=slow, min_periods=1).mean()

        macd = ema_fast - ema_slow

        # Ensure consistent length and fill NaN values
        macd_values = macd.fillna(0).values  # Zero for NaN values

        # If length doesn't match input, pad or truncate to match
        if len(macd_values) != len(returns):
            if len(macd_values) < len(returns):
                # Pad with neutral value
                padding = np.full(len(returns) - len(macd_values), 0.0)
                macd_values = np.concatenate([padding, macd_values])
            else:
                # Truncate to match input length
                macd_values = macd_values[:len(returns)]

        return macd_values
    
    @staticmethod
    def _calculate_volatility_regime(price_range: np.ndarray, window: int = 20) -> np.ndarray:
        """Calculate volatility regime indicator without lookahead bias."""
        price_range_series = pd.Series(price_range)
        rolling_vol = price_range_series.rolling(window, min_periods=1).std()

        # Use expanding window for percentiles to avoid lookahead bias
        # This calculates the percentile rank based only on historical data
        vol_percentile = rolling_vol.expanding(min_periods=1).rank(pct=True)

        # Ensure consistent length and fill NaN values
        vol_regime_values = vol_percentile.fillna(0.5).values  # Neutral value for NaN

        # If length doesn't match input, pad or truncate to match
        if len(vol_regime_values) != len(price_range):
            if len(vol_regime_values) < len(price_range):
                # Pad with neutral value
                padding = np.full(len(price_range) - len(vol_regime_values), 0.5)
                vol_regime_values = np.concatenate([padding, vol_regime_values])
            else:
                # Truncate to match input length
                vol_regime_values = vol_regime_values[:len(price_range)]

        return vol_regime_values
    
    @staticmethod
    def _calculate_volume_trend(volume_return: np.ndarray, window: int = 10) -> np.ndarray:
        """Calculate volume trend strength."""
        volume_series = pd.Series(volume_return)
        volume_sma = volume_series.rolling(window, min_periods=1).mean()
        trend_strength = (volume_series - volume_sma) / (volume_sma + 1e-10)

        # Ensure consistent length and fill NaN values
        trend_values = trend_strength.fillna(0).values  # Zero for NaN values

        # If length doesn't match input, pad or truncate to match
        if len(trend_values) != len(volume_return):
            if len(trend_values) < len(volume_return):
                # Pad with neutral value
                padding = np.full(len(volume_return) - len(trend_values), 0.0)
                trend_values = np.concatenate([padding, trend_values])
            else:
                # Truncate to match input length
                trend_values = trend_values[:len(volume_return)]

        return trend_values
    
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


def get_enhanced_feature_names(original_feature_count: int, original_feature_names: Optional[List[str]] = None) -> List[str]:
    """Get names for enhanced features."""
    return HMMEnhancedFeatureCreator.get_enhanced_feature_names(original_feature_count, original_feature_names)


def create_enhanced_features_with_names(X: np.ndarray, regime_labels: np.ndarray, 
                                      original_feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """Create enhanced features with names."""
    return HMMEnhancedFeatureCreator.create_enhanced_features_with_names(X, regime_labels, original_feature_names)
