"""
Enhanced Regime Feature Engineering for HDBSCAN Clustering

This module provides regime-specific feature engineering that addresses the current
issues with feature quality and regime discrimination.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler, RobustScaler

logger = logging.getLogger(__name__)

@dataclass
class RegimeFeatureEngineeringConfig:
    """Configuration for enhanced regime feature engineering."""
    # Regime detection parameters
    regime_window: int = 20
    transition_threshold: float = 0.1
    persistence_threshold: float = 0.7
    
    # Feature engineering parameters
    enable_regime_transition_features: bool = True
    enable_regime_persistence_features: bool = True
    enable_economic_regime_features: bool = True
    enable_volatility_regime_features: bool = True
    enable_volume_regime_features: bool = True
    
    # Quality thresholds
    min_regime_duration: int = 5
    max_regime_duration: int = 100
    regime_stability_threshold: float = 0.5

class EnhancedRegimeFeatureEngineering:
    """
    Enhanced regime feature engineering for better HDBSCAN clustering.
    
    Addresses current issues:
    1. Poor regime transition detection (currently 0.0)
    2. Low economic distinctiveness (currently 0.000)
    3. Negative silhouette scores
    4. Lack of regime-specific features
    """
    
    def __init__(self, config: Optional[RegimeFeatureEngineeringConfig] = None):
        """Initialize enhanced regime feature engineering."""
        self.config = config or RegimeFeatureEngineeringConfig()
        self.performance_stats = {
            'features_generated': 0,
            'processing_time': 0.0,
            'regime_transitions_detected': 0,
            'regime_persistence_score': 0.0
        }
        
        logger.info("✅ EnhancedRegimeFeatureEngineering initialized")
    
    def generate_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate enhanced regime-specific features.
        
        Args:
            data: Input market data with OHLCV columns
            
        Returns:
            DataFrame with enhanced regime features
        """
        start_time = time.time()
        logger.info(f"🚀 Generating enhanced regime features for {data.shape[0]} samples")
        
        # Create features DataFrame
        features_df = data.copy()
        
        # Generate regime-specific features
        if self.config.enable_regime_transition_features:
            features_df = self._add_regime_transition_features(features_df)
        
        if self.config.enable_regime_persistence_features:
            features_df = self._add_regime_persistence_features(features_df)
        
        if self.config.enable_economic_regime_features:
            features_df = self._add_economic_regime_features(features_df)
        
        if self.config.enable_volatility_regime_features:
            features_df = self._add_volatility_regime_features(features_df)
        
        if self.config.enable_volume_regime_features:
            features_df = self._add_volume_regime_features(features_df)
        
        # Update performance stats
        processing_time = time.time() - start_time
        self.performance_stats['processing_time'] = processing_time
        self.performance_stats['features_generated'] = features_df.shape[1] - data.shape[1]
        
        logger.info(f"✅ Enhanced regime features generated: {self.performance_stats['features_generated']} features in {processing_time:.2f}s")
        
        return features_df
    
    def _add_regime_transition_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add regime transition detection features."""
        features = {}
        
        if 'close' in data.columns:
            close_prices = data['close'].values
            
            # 1. Price regime transitions
            features['price_regime_transition'] = self._detect_price_regime_transitions(close_prices)
            
            # 2. Volatility regime transitions
            returns = np.diff(np.log(close_prices))
            features['volatility_regime_transition'] = self._detect_volatility_regime_transitions(returns)
            
            # 3. Trend regime transitions
            features['trend_regime_transition'] = self._detect_trend_regime_transitions(close_prices)
            
            # 4. Regime change frequency
            features['regime_change_frequency'] = self._calculate_regime_change_frequency(close_prices)
            
            # 5. Regime transition strength
            features['regime_transition_strength'] = self._calculate_regime_transition_strength(close_prices)
        
        # Add features to DataFrame
        for feature_name, feature_values in features.items():
            if len(feature_values) == len(data):
                data[feature_name] = feature_values
            else:
                # Pad with zeros if length doesn't match
                padded_values = np.zeros(len(data))
                padded_values[:len(feature_values)] = feature_values
                data[feature_name] = padded_values
        
        return data
    
    def _add_regime_persistence_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add regime persistence features."""
        features = {}
        
        if 'close' in data.columns:
            close_prices = data['close'].values
            
            # 1. Regime persistence score
            features['regime_persistence_score'] = self._calculate_regime_persistence(close_prices)
            
            # 2. Regime stability index
            features['regime_stability_index'] = self._calculate_regime_stability(close_prices)
            
            # 3. Regime duration features
            features['avg_regime_duration'] = self._calculate_avg_regime_duration(close_prices)
            features['max_regime_duration'] = self._calculate_max_regime_duration(close_prices)
            
            # 4. Regime consistency
            features['regime_consistency'] = self._calculate_regime_consistency(close_prices)
        
        # Add features to DataFrame
        for feature_name, feature_values in features.items():
            if len(feature_values) == len(data):
                data[feature_name] = feature_values
            else:
                padded_values = np.zeros(len(data))
                padded_values[:len(feature_values)] = feature_values
                data[feature_name] = padded_values
        
        return data
    
    def _add_economic_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add economic regime features."""
        features = {}
        
        if 'close' in data.columns:
            close_prices = data['close'].values
            returns = np.diff(np.log(close_prices))
            
            # 1. Economic regime classification
            features['economic_regime'] = self._classify_economic_regime(returns)
            
            # 2. Economic regime strength
            features['economic_regime_strength'] = self._calculate_economic_regime_strength(returns)
            
            # 3. Economic regime transitions
            features['economic_regime_transitions'] = self._detect_economic_regime_transitions(returns)
            
            # 4. Economic regime persistence
            features['economic_regime_persistence'] = self._calculate_economic_regime_persistence(returns)
        
        # Add features to DataFrame
        for feature_name, feature_values in features.items():
            if len(feature_values) == len(data):
                data[feature_name] = feature_values
            else:
                padded_values = np.zeros(len(data))
                padded_values[:len(feature_values)] = feature_values
                data[feature_name] = padded_values
        
        return data
    
    def _add_volatility_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility regime features."""
        features = {}
        
        if 'close' in data.columns:
            close_prices = data['close'].values
            returns = np.diff(np.log(close_prices))
            
            # 1. Volatility regime classification
            features['volatility_regime'] = self._classify_volatility_regime(returns)
            
            # 2. Volatility clustering strength
            features['volatility_clustering_strength'] = self._calculate_volatility_clustering_strength(returns)
            
            # 3. Volatility regime transitions
            features['volatility_regime_transitions'] = self._detect_volatility_regime_transitions(returns)
            
            # 4. Volatility regime persistence
            features['volatility_regime_persistence'] = self._calculate_volatility_regime_persistence(returns)
        
        # Add features to DataFrame
        for feature_name, feature_values in features.items():
            if len(feature_values) == len(data):
                data[feature_name] = feature_values
            else:
                padded_values = np.zeros(len(data))
                padded_values[:len(feature_values)] = feature_values
                data[feature_name] = padded_values
        
        return data
    
    def _add_volume_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume regime features."""
        features = {}
        
        if 'volume' in data.columns and 'close' in data.columns:
            volume = data['volume'].values
            close_prices = data['close'].values
            
            # 1. Volume regime classification
            features['volume_regime'] = self._classify_volume_regime(volume)
            
            # 2. Volume-price relationship regime
            features['volume_price_regime'] = self._classify_volume_price_regime(volume, close_prices)
            
            # 3. Volume regime transitions
            features['volume_regime_transitions'] = self._detect_volume_regime_transitions(volume)
            
            # 4. Volume regime persistence
            features['volume_regime_persistence'] = self._calculate_volume_regime_persistence(volume)
        
        # Add features to DataFrame
        for feature_name, feature_values in features.items():
            if len(feature_values) == len(data):
                data[feature_name] = feature_values
            else:
                padded_values = np.zeros(len(data))
                padded_values[:len(feature_values)] = feature_values
                data[feature_name] = padded_values
        
        return data
    
    # Regime transition detection methods
    def _detect_price_regime_transitions(self, prices: np.ndarray) -> np.ndarray:
        """Detect price regime transitions using rolling statistics."""
        window = self.config.regime_window
        transitions = np.zeros(len(prices))
        
        for i in range(window, len(prices)):
            # Calculate rolling statistics
            recent_window = prices[i-window:i]
            current_window = prices[i-window//2:i]
            
            # Compare recent vs current statistics
            recent_mean = np.mean(recent_window)
            current_mean = np.mean(current_window)
            recent_std = np.std(recent_window)
            current_std = np.std(current_window)
            
            # Detect significant changes
            mean_change = abs(current_mean - recent_mean) / (recent_std + 1e-8)
            std_change = abs(current_std - recent_std) / (recent_std + 1e-8)
            
            if mean_change > self.config.transition_threshold or std_change > self.config.transition_threshold:
                transitions[i] = 1
        
        return transitions
    
    def _detect_volatility_regime_transitions(self, returns: np.ndarray) -> np.ndarray:
        """Detect volatility regime transitions."""
        window = self.config.regime_window
        transitions = np.zeros(len(returns) + 1)  # +1 to match price length
        
        for i in range(window, len(returns)):
            # Calculate rolling volatility
            recent_vol = np.std(returns[i-window:i])
            current_vol = np.std(returns[i-window//2:i])
            
            # Detect volatility regime change
            vol_change = abs(current_vol - recent_vol) / (recent_vol + 1e-8)
            
            if vol_change > self.config.transition_threshold:
                transitions[i+1] = 1  # +1 to align with price index
        
        return transitions
    
    def _detect_trend_regime_transitions(self, prices: np.ndarray) -> np.ndarray:
        """Detect trend regime transitions."""
        window = self.config.regime_window
        transitions = np.zeros(len(prices))
        
        for i in range(window, len(prices)):
            # Calculate trend strength
            recent_trend = np.polyfit(range(window), prices[i-window:i], 1)[0]
            current_trend = np.polyfit(range(window//2), prices[i-window//2:i], 1)[0]
            
            # Detect trend change
            trend_change = abs(current_trend - recent_trend) / (abs(recent_trend) + 1e-8)
            
            if trend_change > self.config.transition_threshold:
                transitions[i] = 1
        
        return transitions
    
    def _calculate_regime_change_frequency(self, prices: np.ndarray) -> np.ndarray:
        """Calculate regime change frequency."""
        window = self.config.regime_window
        frequency = np.zeros(len(prices))
        
        for i in range(window, len(prices)):
            # Count transitions in recent window
            recent_transitions = self._detect_price_regime_transitions(prices[i-window:i])
            frequency[i] = np.sum(recent_transitions) / window
        
        return frequency
    
    def _calculate_regime_transition_strength(self, prices: np.ndarray) -> np.ndarray:
        """Calculate regime transition strength."""
        window = self.config.regime_window
        strength = np.zeros(len(prices))
        
        for i in range(window, len(prices)):
            # Calculate transition strength based on magnitude of change
            recent_window = prices[i-window:i]
            current_window = prices[i-window//2:i]
            
            recent_mean = np.mean(recent_window)
            current_mean = np.mean(current_window)
            recent_std = np.std(recent_window)
            
            strength[i] = abs(current_mean - recent_mean) / (recent_std + 1e-8)
        
        return strength
    
    # Regime persistence methods
    def _calculate_regime_persistence(self, prices: np.ndarray) -> np.ndarray:
        """Calculate regime persistence score."""
        window = self.config.regime_window
        persistence = np.zeros(len(prices))
        
        for i in range(window, len(prices)):
            # Calculate how long current regime has persisted
            recent_window = prices[i-window:i]
            current_regime = self._classify_price_regime(recent_window)
            
            # Count consecutive periods with same regime
            consecutive_count = 0
            for j in range(i-1, max(0, i-window), -1):
                if j >= window:
                    prev_window = prices[j-window:j]
                    prev_regime = self._classify_price_regime(prev_window)
                    if prev_regime == current_regime:
                        consecutive_count += 1
                    else:
                        break
            
            persistence[i] = consecutive_count / window
        
        return persistence
    
    def _calculate_regime_stability(self, prices: np.ndarray) -> np.ndarray:
        """Calculate regime stability index."""
        window = self.config.regime_window
        stability = np.zeros(len(prices))
        
        for i in range(window, len(prices)):
            # Calculate stability based on variance of recent regime classifications
            recent_window = prices[i-window:i]
            regime_classifications = []
            
            for j in range(window//2, window):
                sub_window = recent_window[j-window//2:j]
                regime_classifications.append(self._classify_price_regime(sub_window))
            
            # Stability is inverse of variance
            if len(regime_classifications) > 1:
                stability[i] = 1.0 / (np.var(regime_classifications) + 1e-8)
            else:
                stability[i] = 1.0
        
        return stability
    
    def _calculate_avg_regime_duration(self, prices: np.ndarray) -> np.ndarray:
        """Calculate average regime duration."""
        window = self.config.regime_window
        avg_duration = np.zeros(len(prices))
        
        for i in range(window, len(prices)):
            # Calculate average duration of recent regimes
            recent_window = prices[i-window:i]
            regime_durations = self._extract_regime_durations(recent_window)
            
            if regime_durations:
                avg_duration[i] = np.mean(regime_durations)
            else:
                avg_duration[i] = window
        
        return avg_duration
    
    def _calculate_max_regime_duration(self, prices: np.ndarray) -> np.ndarray:
        """Calculate maximum regime duration."""
        window = self.config.regime_window
        max_duration = np.zeros(len(prices))
        
        for i in range(window, len(prices)):
            # Calculate maximum duration of recent regimes
            recent_window = prices[i-window:i]
            regime_durations = self._extract_regime_durations(recent_window)
            
            if regime_durations:
                max_duration[i] = np.max(regime_durations)
            else:
                max_duration[i] = window
        
        return max_duration
    
    def _calculate_regime_consistency(self, prices: np.ndarray) -> np.ndarray:
        """Calculate regime consistency."""
        window = self.config.regime_window
        consistency = np.zeros(len(prices))
        
        for i in range(window, len(prices)):
            # Calculate consistency based on regime classification stability
            recent_window = prices[i-window:i]
            regime_classifications = []
            
            for j in range(window//4, window, window//4):
                sub_window = recent_window[j-window//4:j]
                regime_classifications.append(self._classify_price_regime(sub_window))
            
            # Consistency is based on how similar the classifications are
            if len(regime_classifications) > 1:
                unique_regimes = len(set(regime_classifications))
                consistency[i] = 1.0 / unique_regimes
            else:
                consistency[i] = 1.0
        
        return consistency
    
    # Economic regime methods
    def _classify_economic_regime(self, returns: np.ndarray) -> np.ndarray:
        """Classify economic regime based on returns."""
        window = self.config.regime_window
        regimes = np.zeros(len(returns) + 1)  # +1 to match price length
        
        for i in range(window, len(returns)):
            recent_returns = returns[i-window:i]
            
            # Classify based on mean and volatility
            mean_return = np.mean(recent_returns)
            volatility = np.std(recent_returns)
            
            if mean_return > 0 and volatility < np.percentile(returns, 33):
                regimes[i+1] = 0  # Bull low vol
            elif mean_return > 0 and volatility > np.percentile(returns, 67):
                regimes[i+1] = 1  # Bull high vol
            elif mean_return < 0 and volatility < np.percentile(returns, 33):
                regimes[i+1] = 2  # Bear low vol
            elif mean_return < 0 and volatility > np.percentile(returns, 67):
                regimes[i+1] = 3  # Bear high vol
            else:
                regimes[i+1] = 4  # Neutral
        
        return regimes
    
    def _calculate_economic_regime_strength(self, returns: np.ndarray) -> np.ndarray:
        """Calculate economic regime strength."""
        window = self.config.regime_window
        strength = np.zeros(len(returns) + 1)
        
        for i in range(window, len(returns)):
            recent_returns = returns[i-window:i]
            
            # Strength based on magnitude of mean return and volatility
            mean_return = np.mean(recent_returns)
            volatility = np.std(recent_returns)
            
            strength[i+1] = abs(mean_return) * volatility
        
        return strength
    
    def _detect_economic_regime_transitions(self, returns: np.ndarray) -> np.ndarray:
        """Detect economic regime transitions."""
        window = self.config.regime_window
        transitions = np.zeros(len(returns) + 1)
        
        for i in range(window, len(returns)):
            recent_regime = self._classify_economic_regime(returns[i-window:i])
            current_regime = self._classify_economic_regime(returns[i-window//2:i])
            
            if recent_regime[-1] != current_regime[-1]:
                transitions[i+1] = 1
        
        return transitions
    
    def _calculate_economic_regime_persistence(self, returns: np.ndarray) -> np.ndarray:
        """Calculate economic regime persistence."""
        window = self.config.regime_window
        persistence = np.zeros(len(returns) + 1)
        
        for i in range(window, len(returns)):
            recent_returns = returns[i-window:i]
            regimes = self._classify_economic_regime(recent_returns)
            
            # Count consecutive periods with same regime
            consecutive_count = 0
            current_regime = regimes[-1]
            
            for j in range(len(regimes)-2, -1, -1):
                if regimes[j] == current_regime:
                    consecutive_count += 1
                else:
                    break
            
            persistence[i+1] = consecutive_count / window
        
        return persistence
    
    # Volatility regime methods
    def _classify_volatility_regime(self, returns: np.ndarray) -> np.ndarray:
        """Classify volatility regime."""
        window = self.config.regime_window
        regimes = np.zeros(len(returns) + 1)
        
        for i in range(window, len(returns)):
            recent_returns = returns[i-window:i]
            volatility = np.std(recent_returns)
            
            # Classify based on volatility percentiles
            if volatility < np.percentile(returns, 33):
                regimes[i+1] = 0  # Low volatility
            elif volatility > np.percentile(returns, 67):
                regimes[i+1] = 2  # High volatility
            else:
                regimes[i+1] = 1  # Medium volatility
        
        return regimes
    
    def _calculate_volatility_clustering_strength(self, returns: np.ndarray) -> np.ndarray:
        """Calculate volatility clustering strength."""
        window = self.config.regime_window
        strength = np.zeros(len(returns) + 1)
        
        for i in range(window, len(returns)):
            recent_returns = returns[i-window:i]
            
            # Calculate autocorrelation of squared returns (volatility clustering)
            squared_returns = recent_returns ** 2
            if len(squared_returns) > 1:
                autocorr = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
                strength[i+1] = abs(autocorr) if not np.isnan(autocorr) else 0
        
        return strength
    
    def _detect_volatility_regime_transitions(self, returns: np.ndarray) -> np.ndarray:
        """Detect volatility regime transitions."""
        window = self.config.regime_window
        transitions = np.zeros(len(returns) + 1)
        
        for i in range(window, len(returns)):
            recent_vol = np.std(returns[i-window:i])
            current_vol = np.std(returns[i-window//2:i])
            
            vol_change = abs(current_vol - recent_vol) / (recent_vol + 1e-8)
            
            if vol_change > self.config.transition_threshold:
                transitions[i+1] = 1
        
        return transitions
    
    def _calculate_volatility_regime_persistence(self, returns: np.ndarray) -> np.ndarray:
        """Calculate volatility regime persistence."""
        window = self.config.regime_window
        persistence = np.zeros(len(returns) + 1)
        
        for i in range(window, len(returns)):
            recent_returns = returns[i-window:i]
            regimes = self._classify_volatility_regime(recent_returns)
            
            consecutive_count = 0
            current_regime = regimes[-1]
            
            for j in range(len(regimes)-2, -1, -1):
                if regimes[j] == current_regime:
                    consecutive_count += 1
                else:
                    break
            
            persistence[i+1] = consecutive_count / window
        
        return persistence
    
    # Volume regime methods
    def _classify_volume_regime(self, volume: np.ndarray) -> np.ndarray:
        """Classify volume regime."""
        window = self.config.regime_window
        regimes = np.zeros(len(volume))
        
        for i in range(window, len(volume)):
            recent_volume = volume[i-window:i]
            avg_volume = np.mean(recent_volume)
            
            # Classify based on volume percentiles
            if avg_volume < np.percentile(volume, 33):
                regimes[i] = 0  # Low volume
            elif avg_volume > np.percentile(volume, 67):
                regimes[i] = 2  # High volume
            else:
                regimes[i] = 1  # Medium volume
        
        return regimes
    
    def _classify_volume_price_regime(self, volume: np.ndarray, prices: np.ndarray) -> np.ndarray:
        """Classify volume-price relationship regime."""
        window = self.config.regime_window
        regimes = np.zeros(len(volume))
        
        for i in range(window, len(volume)):
            recent_volume = volume[i-window:i]
            recent_prices = prices[i-window:i]
            
            # Calculate volume-price correlation
            if len(recent_volume) > 1 and len(recent_prices) > 1:
                corr = np.corrcoef(recent_volume, recent_prices)[0, 1]
                
                if not np.isnan(corr):
                    if corr > 0.3:
                        regimes[i] = 0  # Positive correlation
                    elif corr < -0.3:
                        regimes[i] = 2  # Negative correlation
                    else:
                        regimes[i] = 1  # Weak correlation
                else:
                    regimes[i] = 1
        
        return regimes
    
    def _detect_volume_regime_transitions(self, volume: np.ndarray) -> np.ndarray:
        """Detect volume regime transitions."""
        window = self.config.regime_window
        transitions = np.zeros(len(volume))
        
        for i in range(window, len(volume)):
            recent_volume = volume[i-window:i]
            current_volume = volume[i-window//2:i]
            
            recent_avg = np.mean(recent_volume)
            current_avg = np.mean(current_volume)
            
            vol_change = abs(current_avg - recent_avg) / (recent_avg + 1e-8)
            
            if vol_change > self.config.transition_threshold:
                transitions[i] = 1
        
        return transitions
    
    def _calculate_volume_regime_persistence(self, volume: np.ndarray) -> np.ndarray:
        """Calculate volume regime persistence."""
        window = self.config.regime_window
        persistence = np.zeros(len(volume))
        
        for i in range(window, len(volume)):
            recent_volume = volume[i-window:i]
            regimes = self._classify_volume_regime(recent_volume)
            
            consecutive_count = 0
            current_regime = regimes[-1]
            
            for j in range(len(regimes)-2, -1, -1):
                if regimes[j] == current_regime:
                    consecutive_count += 1
                else:
                    break
            
            persistence[i] = consecutive_count / window
        
        return persistence
    
    # Helper methods
    def _classify_price_regime(self, prices: np.ndarray) -> int:
        """Classify price regime for a given window."""
        if len(prices) < 2:
            return 0
        
        # Simple regime classification based on trend and volatility
        trend = np.polyfit(range(len(prices)), prices, 1)[0]
        volatility = np.std(prices)
        
        if trend > 0 and volatility < np.percentile(prices, 50):
            return 0  # Bull low vol
        elif trend > 0 and volatility >= np.percentile(prices, 50):
            return 1  # Bull high vol
        elif trend <= 0 and volatility < np.percentile(prices, 50):
            return 2  # Bear low vol
        else:
            return 3  # Bear high vol
    
    def _extract_regime_durations(self, prices: np.ndarray) -> List[int]:
        """Extract regime durations from price series."""
        if len(prices) < 2:
            return []
        
        regimes = []
        for i in range(len(prices)//2, len(prices)):
            sub_window = prices[i-len(prices)//2:i]
            regimes.append(self._classify_price_regime(sub_window))
        
        durations = []
        current_duration = 1
        current_regime = regimes[0]
        
        for i in range(1, len(regimes)):
            if regimes[i] == current_regime:
                current_duration += 1
            else:
                durations.append(current_duration)
                current_duration = 1
                current_regime = regimes[i]
        
        durations.append(current_duration)
        return durations
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()

# Convenience function
def create_enhanced_regime_feature_engineering(
    regime_window: int = 20,
    transition_threshold: float = 0.1,
    enable_regime_transition_features: bool = True,
    enable_regime_persistence_features: bool = True,
    enable_economic_regime_features: bool = True,
    enable_volatility_regime_features: bool = True,
    enable_volume_regime_features: bool = True
) -> EnhancedRegimeFeatureEngineering:
    """
    Create enhanced regime feature engineering instance.
    
    Args:
        regime_window: Window size for regime analysis
        transition_threshold: Threshold for regime transition detection
        enable_regime_transition_features: Enable regime transition features
        enable_regime_persistence_features: Enable regime persistence features
        enable_economic_regime_features: Enable economic regime features
        enable_volatility_regime_features: Enable volatility regime features
        enable_volume_regime_features: Enable volume regime features
        
    Returns:
        EnhancedRegimeFeatureEngineering instance
    """
    config = RegimeFeatureEngineeringConfig(
        regime_window=regime_window,
        transition_threshold=transition_threshold,
        enable_regime_transition_features=enable_regime_transition_features,
        enable_regime_persistence_features=enable_regime_persistence_features,
        enable_economic_regime_features=enable_economic_regime_features,
        enable_volatility_regime_features=enable_volatility_regime_features,
        enable_volume_regime_features=enable_volume_regime_features
    )
    
    return EnhancedRegimeFeatureEngineering(config)