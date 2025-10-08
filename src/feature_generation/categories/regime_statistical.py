"""
Regime Statistical Feature Generator

This module provides feature generators specifically designed for regime classification
in 15-minute timeframes with 5-30 minute trade durations. Focuses on statistical
regime characteristics rather than short-term trading signals.

Key Features:
- Distribution shape changes (skewness, kurtosis)
- Regime persistence measures
- Cross-correlation stability
- Regime transition probabilities
- Statistical regime stability
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from scipy import stats
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis, jarque_bera

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
from ..base_calculations import (
    BaseCalculator,
    BaseCalculationType,
    BaseCalculationConfig,
    create_base_calculator
)

# Import tprint for consistent logging
from src.utils.tprint import tprint

class RegimeStatisticalFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for statistical regime features optimized for 15m timeframe."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="regime_statistical_features",
            category=FeatureCategory.REGIME,
            description="Statistical regime features for 15m timeframe regime classification",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=32,  # 8 hours in 15m periods
            min_lookback=8,       # 2 hours minimum
            max_lookback=128,     # 32 hours maximum
            parameters={
                "distribution_windows": [16, 48, 128],  # 4h, 12h, 32h in 15m periods (original min, middle, new max)
                "correlation_windows": [20, 60, 160],  # 5h, 15h, 40h (original min, middle, new max)
                "persistence_windows": [12, 30, 96],  # 3h, 7.5h, 24h (original min, middle, new max)
                "transition_windows": [8, 20, 64]  # 2h, 5h, 16h (original min, middle, new max)
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate a single statistical regime feature as required by the base class."""
        try:
            # Generate all statistical regime features
            features_dict = self.generate_features(data, **kwargs)
            
            # Combine all features into a single series (use first feature as representative)
            if features_dict:
                first_feature_name = list(features_dict.keys())[0]
                return pd.Series(features_dict[first_feature_name], index=data.index[:len(features_dict[first_feature_name])])
            else:
                # Return a simple statistical feature if no features generated
                if 'close' in data.columns and len(data) > 1:
                    stat_feature = data['close'].rolling(window=5).std().fillna(0).values
                    return pd.Series(stat_feature, index=data.index)
                else:
                    return pd.Series(np.zeros(len(data)), index=data.index)
                
        except Exception as e:
            tprint(f"_generate_feature: Statistical regime feature generation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate statistical regime features."""
        features = {}
        
        try:
            # Validate price data
            if 'close' not in data.columns:
                tprint("Warning: 'close' column not found in data")
                return features
            
            close_prices = data['close'].values
            if len(close_prices) < 8:
                tprint(f"Warning: Insufficient data points: {len(close_prices)} < 8")
                return features
            
            # Calculate returns for statistical analysis
            returns = np.diff(np.log(close_prices))
            
            # 1. Distribution Shape Features
            try:
                features.update(self._generate_distribution_features(returns, data))
            except Exception as e:
                tprint(f"Error in distribution features: {e}")
            
            # 2. Statistical Regime Persistence
            try:
                features.update(self._generate_statistical_persistence_features(returns, data))
            except Exception as e:
                tprint(f"Error in persistence features: {e}")
            
            # 3. Cross-Correlation Features
            try:
                features.update(self._generate_correlation_features(returns, data))
            except Exception as e:
                tprint(f"Error in correlation features: {e}")
            
            # 4. Statistical Regime Transitions
            try:
                features.update(self._generate_statistical_transition_features(returns, data))
            except Exception as e:
                tprint(f"Error in transition features: {e}")
            
            # 5. Statistical Regime Stability
            try:
                features.update(self._generate_statistical_stability_features(returns, data))
            except Exception as e:
                tprint(f"Error in stability features: {e}")
            
        except Exception as e:
            tprint(f"Error in statistical feature generation: {e}")
        
        return features
    
    def _generate_distribution_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate distribution shape features."""
        features = {}
        windows = self.config.parameters["distribution_windows"]

        for window in windows:
            if len(returns) < window:
                continue

            # Skewness regime features
            skewness = self._calculate_rolling_skewness(returns, window)
            skewness_persistence = self._calculate_skewness_persistence(returns, window)

            # Kurtosis regime features
            kurtosis = self._calculate_rolling_kurtosis(returns, window)
            kurtosis_persistence = self._calculate_kurtosis_persistence(returns, window)

            # Distribution normality
            normality = self._calculate_distribution_normality(returns, window)

            # Pad to match data length - use consistent padding logic
            data_len = len(data)
            skewness_padded = np.full(data_len, np.nan)
            skew_persist_padded = np.full(data_len, np.nan)
            kurtosis_padded = np.full(data_len, np.nan)
            kurt_persist_padded = np.full(data_len, np.nan)
            normality_padded = np.full(data_len, np.nan)

            # Ensure we don't exceed array bounds
            valid_indices = min(len(skewness), data_len - window)
            if valid_indices > 0:
                skewness_padded[window:window + valid_indices] = skewness[:valid_indices]
                skew_persist_padded[window:window + valid_indices] = skewness_persistence[:valid_indices]
                kurtosis_padded[window:window + valid_indices] = kurtosis[:valid_indices]
                kurt_persist_padded[window:window + valid_indices] = kurtosis_persistence[:valid_indices]
                normality_padded[window:window + valid_indices] = normality[:valid_indices]

            features[f'returns_skewness_{window}'] = skewness_padded
            features[f'skewness_persistence_{window}'] = skew_persist_padded
            features[f'returns_kurtosis_{window}'] = kurtosis_padded
            features[f'kurtosis_persistence_{window}'] = kurt_persist_padded
            features[f'distribution_normality_{window}'] = normality_padded

        return features
    
    def _generate_statistical_persistence_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate statistical persistence features."""
        features = {}
        windows = self.config.parameters["persistence_windows"]

        for window in windows:
            if len(returns) < window:
                continue

            # Statistical regime persistence
            stat_persistence = self._calculate_statistical_persistence(returns, window)

            # Distribution stability
            dist_stability = self._calculate_distribution_stability(returns, window)

            # Statistical regime strength
            stat_strength = self._calculate_statistical_strength(returns, window)

            # Pad to match data length - use consistent padding logic
            data_len = len(data)
            persistence_padded = np.full(data_len, np.nan)
            stability_padded = np.full(data_len, np.nan)
            strength_padded = np.full(data_len, np.nan)

            # Ensure we don't exceed array bounds
            valid_indices = min(len(stat_persistence), data_len - window)
            if valid_indices > 0:
                persistence_padded[window:window + valid_indices] = stat_persistence[:valid_indices]
                stability_padded[window:window + valid_indices] = dist_stability[:valid_indices]
                strength_padded[window:window + valid_indices] = stat_strength[:valid_indices]

            features[f'statistical_persistence_{window}'] = persistence_padded
            features[f'distribution_stability_{window}'] = stability_padded
            features[f'statistical_strength_{window}'] = strength_padded

        return features
    
    def _generate_correlation_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate cross-correlation features."""
        features = {}
        windows = self.config.parameters["correlation_windows"]

        for window in windows:
            if len(returns) < window:
                continue

            # Returns autocorrelation
            autocorr = self._calculate_returns_autocorrelation(returns, window)

            # Correlation stability
            corr_stability = self._calculate_correlation_stability(returns, window)

            # Cross-correlation regime features
            cross_corr = self._calculate_cross_correlation_features(returns, window)

            # Pad to match data length - use consistent padding logic
            data_len = len(data)
            autocorr_padded = np.full(data_len, np.nan)
            stability_padded = np.full(data_len, np.nan)
            cross_corr_padded = np.full(data_len, np.nan)

            # Ensure we don't exceed array bounds
            valid_indices = min(len(autocorr), data_len - window)
            if valid_indices > 0:
                autocorr_padded[window:window + valid_indices] = autocorr[:valid_indices]
                stability_padded[window:window + valid_indices] = corr_stability[:valid_indices]
                cross_corr_padded[window:window + valid_indices] = cross_corr[:valid_indices]

            features[f'returns_autocorr_{window}'] = autocorr_padded
            features[f'correlation_stability_{window}'] = stability_padded
            features[f'cross_correlation_{window}'] = cross_corr_padded

        return features
    
    def _generate_statistical_transition_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate statistical transition features."""
        features = {}
        windows = self.config.parameters["transition_windows"]

        for window in windows:
            if len(returns) < window * 2:
                continue

            # Statistical regime change detection
            stat_change = self._detect_statistical_regime_changes(returns, window)

            # Distribution transition probability
            dist_transition = self._calculate_distribution_transition_probability(returns, window)

            # Statistical regime momentum
            stat_momentum = self._calculate_statistical_momentum(returns, window)

            # Pad to match data length - use consistent padding logic
            data_len = len(data)
            change_padded = np.full(data_len, np.nan)
            transition_padded = np.full(data_len, np.nan)
            momentum_padded = np.full(data_len, np.nan)

            # Ensure we don't exceed array bounds
            valid_indices = min(len(stat_change), data_len - window * 2)
            if valid_indices > 0:
                change_padded[window*2:window*2 + valid_indices] = stat_change[:valid_indices]
                transition_padded[window*2:window*2 + valid_indices] = dist_transition[:valid_indices]
                momentum_padded[window*2:window*2 + valid_indices] = stat_momentum[:valid_indices]

            features[f'statistical_regime_change_{window}'] = change_padded
            features[f'distribution_transition_{window}'] = transition_padded
            features[f'statistical_momentum_{window}'] = momentum_padded

        return features
    
    def _generate_statistical_stability_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate statistical stability features."""
        features = {}
        windows = self.config.parameters["persistence_windows"]

        for window in windows:
            if len(returns) < window:
                continue

            # Statistical regime stability
            stat_stability = self._calculate_statistical_stability(returns, window)

            # Distribution entropy
            dist_entropy = self._calculate_distribution_entropy(returns, window)

            # Statistical regime consistency
            stat_consistency = self._calculate_statistical_consistency(returns, window)

            # Pad to match data length - use consistent padding logic
            data_len = len(data)
            stability_padded = np.full(data_len, np.nan)
            entropy_padded = np.full(data_len, np.nan)
            consistency_padded = np.full(data_len, np.nan)

            # Ensure we don't exceed array bounds
            valid_indices = min(len(stat_stability), data_len - window)
            if valid_indices > 0:
                stability_padded[window:window + valid_indices] = stat_stability[:valid_indices]
                entropy_padded[window:window + valid_indices] = dist_entropy[:valid_indices]
                consistency_padded[window:window + valid_indices] = stat_consistency[:valid_indices]

            features[f'statistical_stability_{window}'] = stability_padded
            features[f'distribution_entropy_{window}'] = entropy_padded
            features[f'statistical_consistency_{window}'] = consistency_padded

        return features
    
    def _calculate_rolling_skewness(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling skewness - VECTORIZED."""
        if len(returns) < window:
            return np.array([])
        
        # OPTIMIZED: Use vectorized skewness calculation
        returns_series = pd.Series(returns)
        
        # Vectorized skewness using rolling statistics
        rolling_mean = returns_series.rolling(window=window).mean()
        rolling_std = returns_series.rolling(window=window).std()
        
        # Simplified skewness approximation using third moment
        centered = returns_series - rolling_mean
        skewness = (centered ** 3).rolling(window=window).mean() / (rolling_std ** 3 + 1e-8)
        
        return skewness.fillna(0).values
    
    def _calculate_rolling_kurtosis(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling kurtosis - VECTORIZED."""
        if len(returns) < window:
            return np.array([])
        
        # OPTIMIZED: Use vectorized kurtosis calculation
        returns_series = pd.Series(returns)
        
        # Vectorized kurtosis using rolling statistics
        rolling_mean = returns_series.rolling(window=window).mean()
        rolling_std = returns_series.rolling(window=window).std()
        
        # Simplified kurtosis approximation using fourth moment
        centered = returns_series - rolling_mean
        kurt = (centered ** 4).rolling(window=window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        return kurt.fillna(0).values
    
    def _calculate_skewness_persistence(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate skewness persistence - FULLY VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        # OPTIMIZED: Fully vectorized skewness persistence calculation
        returns_series = pd.Series(returns)
        
        # Calculate rolling skewness for the entire series at once
        rolling_mean = returns_series.rolling(window=window).mean()
        rolling_std = returns_series.rolling(window=window).std()
        centered = returns_series - rolling_mean
        skewness = (centered ** 3).rolling(window=window).mean() / (rolling_std ** 3 + 1e-8)
        
        # Calculate autocorrelation of skewness using vectorized operations
        skewness_shifted = skewness.shift(1)
        skewness_autocorr = skewness.rolling(window=window//4).corr(skewness_shifted).fillna(0)
        
        return skewness_autocorr.values
    
    def _calculate_autocorrelation(self, returns_window: pd.Series, sub_window: int) -> float:
        """Calculate autocorrelation for a returns window."""
        if len(returns_window) < sub_window * 2:
            return 0.0
        
        # OPTIMIZED: Use vectorized skewness calculation
        returns_series = pd.Series(returns_window)
        rolling_mean = returns_series.rolling(window=sub_window).mean()
        rolling_std = returns_series.rolling(window=sub_window).std()
        
        # Vectorized skewness using third moment
        centered = returns_series - rolling_mean
        skew_values = (centered ** 3).rolling(window=sub_window).mean() / (rolling_std ** 3 + 1e-8)
        skew_values = skew_values.dropna()
        
        if len(skew_values) > 1:
            corr = np.corrcoef(skew_values[:-1], skew_values[1:])[0, 1]
            return corr if not np.isnan(corr) else 0
        return 0.0
    
    def _calculate_kurtosis_persistence(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate kurtosis persistence - FULLY VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        # OPTIMIZED: Fully vectorized kurtosis persistence calculation
        returns_series = pd.Series(returns)
        
        # Calculate rolling kurtosis for the entire series at once
        rolling_mean = returns_series.rolling(window=window).mean()
        rolling_std = returns_series.rolling(window=window).std()
        centered = returns_series - rolling_mean
        kurtosis = (centered ** 4).rolling(window=window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        # Calculate autocorrelation of kurtosis using vectorized operations
        kurtosis_shifted = kurtosis.shift(1)
        kurtosis_autocorr = kurtosis.rolling(window=window//4).corr(kurtosis_shifted).fillna(0)
        
        return kurtosis_autocorr.values
    
    def _calculate_kurtosis_autocorrelation(self, returns_window: pd.Series, sub_window: int) -> float:
        """Calculate kurtosis autocorrelation for a returns window."""
        if len(returns_window) < sub_window * 2:
            return 0.0
        
        # OPTIMIZED: Use vectorized kurtosis calculation
        returns_series = pd.Series(returns_window)
        rolling_mean = returns_series.rolling(window=sub_window).mean()
        rolling_std = returns_series.rolling(window=sub_window).std()
        
        # Vectorized kurtosis using fourth moment
        centered = returns_series - rolling_mean
        kurt_values = (centered ** 4).rolling(window=sub_window).mean() / (rolling_std ** 4 + 1e-8) - 3
        kurt_values = kurt_values.dropna()
        
        if len(kurt_values) > 1:
            corr = np.corrcoef(kurt_values[:-1], kurt_values[1:])[0, 1]
            return corr if not np.isnan(corr) else 0
        return 0.0
    
    def _calculate_distribution_normality(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate distribution normality using Jarque-Bera test - VECTORIZED."""
        if len(returns) < window:
            return np.array([])
        
        # OPTIMIZED: Use vectorized normality approximation
        returns_series = pd.Series(returns)
        
        # Vectorized normality test using skewness and kurtosis approximation
        rolling_mean = returns_series.rolling(window=window).mean()
        rolling_std = returns_series.rolling(window=window).std()
        
        # Calculate skewness and kurtosis for normality test
        centered = returns_series - rolling_mean
        skewness = (centered ** 3).rolling(window=window).mean() / (rolling_std ** 3 + 1e-8)
        kurtosis = (centered ** 4).rolling(window=window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        # Simplified normality test using JB statistic approximation
        jb_stat = (skewness ** 2 + (kurtosis ** 2) / 4) * window / 6
        normality = np.exp(-jb_stat / 2)  # Approximate p-value
        
        return normality.fillna(0).values
    
    def _calculate_jarque_bera_pvalue(self, returns_window: pd.Series) -> float:
        """Calculate Jarque-Bera p-value for a returns window."""
        try:
            jb_stat, p_value = jarque_bera(returns_window)
            return p_value
        except:
            return 0.0
    
    def _calculate_statistical_persistence(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate statistical regime persistence - FULLY VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        # OPTIMIZED: Fully vectorized statistical persistence calculation
        returns_series = pd.Series(returns)
        
        # Calculate squared returns
        squared_returns = returns_series ** 2
        
        # Calculate autocorrelation of squared returns using vectorized operations
        squared_returns_shifted = squared_returns.shift(1)
        autocorr = squared_returns.rolling(window=window).corr(squared_returns_shifted).fillna(0)
        
        return autocorr.values
    
    def _calculate_squared_returns_autocorr(self, returns_window: pd.Series) -> float:
        """Calculate autocorrelation of squared returns for a window."""
        if len(returns_window) < 3:
            return 0.0
        
        # Persistence based on autocorrelation of squared returns
        squared_returns = returns_window ** 2
        if len(squared_returns) > 1:
            corr = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
            return corr if not np.isnan(corr) else 0
        return 0.0
    
    def _calculate_distribution_stability(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate distribution stability - FULLY VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        # OPTIMIZED: Fully vectorized distribution stability calculation
        returns_series = pd.Series(returns)
        sub_window = max(2, window // 4)
        
        # Calculate rolling skewness and kurtosis
        rolling_mean = returns_series.rolling(window=sub_window).mean()
        rolling_std = returns_series.rolling(window=sub_window).std()
        centered = returns_series - rolling_mean
        
        skewness = (centered ** 3).rolling(window=sub_window).mean() / (rolling_std ** 3 + 1e-8)
        kurtosis = (centered ** 4).rolling(window=sub_window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        # Calculate coefficient of variation for stability
        skew_cv = skewness.rolling(window=window).std() / (skewness.rolling(window=window).mean().abs() + 1e-8)
        kurt_cv = kurtosis.rolling(window=window).std() / (kurtosis.rolling(window=window).mean().abs() + 1e-8)
        
        # Stability based on low coefficient of variation
        stability = np.maximum(0, 1 - (skew_cv + kurt_cv) / 2)
        
        return stability.fillna(0).values
    
    def _calculate_moment_stability(self, returns_window: pd.Series, sub_window: int) -> float:
        """Calculate moment stability for a returns window."""
        if len(returns_window) < sub_window * 2:
            return 0.0
        
        # OPTIMIZED: Use vectorized moment calculations
        returns_series = pd.Series(returns_window)
        rolling_mean = returns_series.rolling(window=sub_window).mean()
        rolling_std = returns_series.rolling(window=sub_window).std()
        
        # Vectorized skewness and kurtosis
        centered = returns_series - rolling_mean
        skew_vals = (centered ** 3).rolling(window=sub_window).mean() / (rolling_std ** 3 + 1e-8)
        kurt_vals = (centered ** 4).rolling(window=sub_window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        skew_vals = skew_vals.dropna()
        kurt_vals = kurt_vals.dropna()
        
        if len(skew_vals) > 1 and len(kurt_vals) > 1:
            skew_cv = np.std(skew_vals) / (np.mean(np.abs(skew_vals)) + 1e-8)
            kurt_cv = np.std(kurt_vals) / (np.mean(np.abs(kurt_vals)) + 1e-8)
            return max(0, 1 - (skew_cv + kurt_cv) / 2)
        return 0.0
    
    def _calculate_statistical_strength(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate statistical regime strength - FULLY VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        # OPTIMIZED: Fully vectorized statistical strength calculation
        returns_series = pd.Series(returns)
        
        # Calculate rolling skewness and kurtosis
        rolling_mean = returns_series.rolling(window=window).mean()
        rolling_std = returns_series.rolling(window=window).std()
        centered = returns_series - rolling_mean
        
        skewness = (centered ** 3).rolling(window=window).mean() / (rolling_std ** 3 + 1e-8)
        kurtosis = (centered ** 4).rolling(window=window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        # Strength based on deviation from normal distribution
        deviation = np.abs(skewness) + np.abs(kurtosis - 3)
        strength = np.maximum(0, 1 - deviation / 10)  # Normalize to 0-1
        
        return strength.fillna(0).values
    
    def _calculate_distribution_strength(self, returns_window: pd.Series) -> float:
        """Calculate distribution strength for a returns window."""
        if len(returns_window) < 3:
            return 0.0
        
        # Strength based on how well-defined the distribution is
        skewness = skew(returns_window)
        kurtosis_val = kurtosis(returns_window)
        
        # Strength based on deviation from normal distribution
        deviation = abs(skewness) + abs(kurtosis_val - 3)
        return max(0, 1 - deviation / 10)  # Normalize to 0-1
    
    def _calculate_returns_autocorrelation(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate returns autocorrelation - VECTORIZED."""
        if len(returns) < window:
            return np.array([])
        
        # OPTIMIZED: Use vectorized autocorrelation calculation
        returns_series = pd.Series(returns)
        
        # Vectorized autocorrelation using pandas built-in method
        autocorr = returns_series.rolling(window=window).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
            raw=False
        ).fillna(0)
        
        return autocorr.values
    
    def _calculate_window_autocorr(self, returns_window: pd.Series) -> float:
        """Calculate autocorrelation for a returns window."""
        if len(returns_window) < 2:
            return 0.0
        
        corr = np.corrcoef(returns_window[:-1], returns_window[1:])[0, 1]
        return corr if not np.isnan(corr) else 0
    
    def _calculate_correlation_stability(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate correlation stability - FULLY VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        # OPTIMIZED: Fully vectorized correlation stability calculation
        returns_series = pd.Series(returns)
        
        # Calculate rolling autocorrelation using vectorized operations
        returns_shifted = returns_series.shift(1)
        autocorr = returns_series.rolling(window=window).corr(returns_shifted)
        
        # Calculate stability as inverse of autocorrelation variance
        autocorr_variance = autocorr.rolling(window=window//4).std()
        stability = np.maximum(0, 1 - autocorr_variance)
        
        return stability.fillna(0).values
    
    def _calculate_correlation_stability_window(self, returns_window: np.ndarray) -> float:
        """Helper function for correlation stability calculation."""
        if len(returns_window) < 3:
            return 0.0
        
        try:
            # Calculate rolling autocorrelation
            autocorr_vals = []
            sub_window = max(2, len(returns_window) // 4)
            step = max(1, sub_window // 2)
            
            for j in range(0, len(returns_window) - sub_window, step):
                sub_returns = returns_window[j:j+sub_window]
                if len(sub_returns) > 1:
                    corr = np.corrcoef(sub_returns[:-1], sub_returns[1:])[0, 1]
                    if not np.isnan(corr):
                        autocorr_vals.append(corr)
            
            if len(autocorr_vals) > 1:
                # Stability based on low variance of autocorrelations
                stability = max(0, 1 - np.std(autocorr_vals))
                return stability
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_cross_correlation_features(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate cross-correlation features - VECTORIZED."""
        if len(returns) < window:
            return np.array([])
        
        # Vectorized approach using pandas rolling
        returns_series = pd.Series(returns)
        
        # OPTIMIZED: Use vectorized cross-correlation calculation
        # Calculate lagged correlation using vectorized operations
        returns_lagged = returns_series.shift(1)
        cross_corr = returns_series.rolling(window=window).corr(returns_lagged).fillna(0)
        
        return cross_corr.values
    
    def _calculate_cross_corr_window(self, returns_window: np.ndarray) -> float:
        """Helper function for cross-correlation calculation."""
        if len(returns_window) < 3:
            return 0.0
        
        try:
            # Cross-correlation between returns and absolute returns
            abs_returns = np.abs(returns_window)
            corr = np.corrcoef(returns_window, abs_returns)[0, 1]
            return corr if not np.isnan(corr) else 0.0
        except:
            return 0.0
    
    def _detect_statistical_regime_changes(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Detect statistical regime changes - VECTORIZED."""
        if len(returns) < window * 2:
            return np.zeros(len(returns))
        
        # Vectorized approach using pandas rolling
        returns_series = pd.Series(returns)
        
        # OPTIMIZED: Use vectorized moment calculations
        rolling_mean = returns_series.rolling(window=window).mean()
        rolling_std = returns_series.rolling(window=window).std()
        centered = returns_series - rolling_mean
        
        # Vectorized skewness and kurtosis
        skew1 = (centered ** 3).rolling(window=window).mean() / (rolling_std ** 3 + 1e-8)
        kurt1 = (centered ** 4).rolling(window=window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        # Shift to get second window
        skew2 = skew1.shift(-window)
        kurt2 = kurt1.shift(-window)
        
        # Calculate change ratios
        skew_change = ((skew2 - skew1).abs() / (skew1.abs() + 1e-8)).fillna(0)
        kurt_change = ((kurt2 - kurt1).abs() / (kurt1.abs() + 1e-8)).fillna(0)
        
        # Apply threshold (50% change)
        changes = ((skew_change > 0.5) | (kurt_change > 0.5)).astype(int)
        
        return changes.fillna(0).values
    
    def _calculate_distribution_transition_probability(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate distribution transition probability - FULLY VECTORIZED."""
        if len(returns) < window * 2:
            return np.zeros(len(returns))
        
        # OPTIMIZED: Fully vectorized distribution transition probability calculation
        returns_series = pd.Series(returns)
        sub_window = max(2, window // 2)
        
        # Calculate rolling skewness and kurtosis
        rolling_mean = returns_series.rolling(window=sub_window).mean()
        rolling_std = returns_series.rolling(window=sub_window).std()
        centered = returns_series - rolling_mean
        
        skewness = (centered ** 3).rolling(window=sub_window).mean() / (rolling_std ** 3 + 1e-8)
        kurtosis = (centered ** 4).rolling(window=sub_window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        # Calculate volatility of statistical moments
        skew_vol = skewness.rolling(window=window*2).std()
        kurt_vol = kurtosis.rolling(window=window*2).std()
        
        # Transition probability based on moment volatility
        transition_prob = np.minimum(1, (skew_vol + kurt_vol) / 2)
        
        return transition_prob.fillna(0).values
    
    def _calculate_transition_prob_window(self, recent_returns: np.ndarray, sub_window: int) -> float:
        """Helper function for distribution transition probability calculation."""
        if len(recent_returns) < 3 or sub_window < 2:
            return 0.0
        
        try:
            # Calculate rolling skewness and kurtosis
            skew_vals = self._calculate_rolling_skewness(recent_returns, sub_window)
            kurt_vals = self._calculate_rolling_kurtosis(recent_returns, sub_window)
            
            if len(skew_vals) > 1 and len(kurt_vals) > 1:
                # Probability based on volatility of statistical moments
                skew_vol = np.std(skew_vals)
                kurt_vol = np.std(kurt_vals)
                return min(1, (skew_vol + kurt_vol) / 2)
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_statistical_momentum(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate statistical momentum - FULLY VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        # OPTIMIZED: Fully vectorized statistical momentum calculation
        returns_series = pd.Series(returns)
        sub_window = max(2, window // 4)
        
        # Calculate rolling skewness and kurtosis
        rolling_mean = returns_series.rolling(window=sub_window).mean()
        rolling_std = returns_series.rolling(window=sub_window).std()
        centered = returns_series - rolling_mean
        
        skewness = (centered ** 3).rolling(window=sub_window).mean() / (rolling_std ** 3 + 1e-8)
        kurtosis = (centered ** 4).rolling(window=sub_window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        # Calculate trend in statistical moments using linear regression approximation
        x = np.arange(len(skewness))
        skew_trend = skewness.rolling(window=window).apply(
            lambda x: np.polyfit(np.arange(len(x)), x.values, 1)[0] if len(x) > 1 else 0, raw=False
        )
        kurt_trend = kurtosis.rolling(window=window).apply(
            lambda x: np.polyfit(np.arange(len(x)), x.values, 1)[0] if len(x) > 1 else 0, raw=False
        )
        
        momentum = (skew_trend + kurt_trend) / 2
        
        return momentum.fillna(0).values
    
    def _calculate_momentum_window(self, returns_window: np.ndarray, sub_window: int) -> float:
        """Helper function for statistical momentum calculation."""
        if len(returns_window) < 3 or sub_window < 2:
            return 0.0
        
        try:
            # Momentum based on trend in statistical moments
            skew_vals = self._calculate_rolling_skewness(returns_window, sub_window)
            kurt_vals = self._calculate_rolling_kurtosis(returns_window, sub_window)
            
            if len(skew_vals) > 1 and len(kurt_vals) > 1:
                # Calculate trend in statistical moments
                x = np.arange(len(skew_vals))
                skew_trend = np.polyfit(x, skew_vals, 1)[0]
                kurt_trend = np.polyfit(x, kurt_vals, 1)[0]
                return (skew_trend + kurt_trend) / 2
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_statistical_stability(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate statistical stability - FULLY VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        # OPTIMIZED: Fully vectorized statistical stability calculation
        returns_series = pd.Series(returns)
        sub_window = max(2, window // 4)
        
        # Calculate rolling skewness and kurtosis
        rolling_mean = returns_series.rolling(window=sub_window).mean()
        rolling_std = returns_series.rolling(window=sub_window).std()
        centered = returns_series - rolling_mean
        
        skewness = (centered ** 3).rolling(window=sub_window).mean() / (rolling_std ** 3 + 1e-8)
        kurtosis = (centered ** 4).rolling(window=sub_window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        # Calculate coefficient of variation for stability
        skew_cv = skewness.rolling(window=window).std() / (skewness.rolling(window=window).mean().abs() + 1e-8)
        kurt_cv = kurtosis.rolling(window=window).std() / (kurtosis.rolling(window=window).mean().abs() + 1e-8)
        
        # Stability based on low coefficient of variation
        stability = np.maximum(0, 1 - (skew_cv + kurt_cv) / 2)
        
        return stability.fillna(0).values
    
    def _calculate_stability_window(self, returns_window: np.ndarray, sub_window: int) -> float:
        """Helper function for statistical stability calculation."""
        if len(returns_window) < 3 or sub_window < 2:
            return 0.0
        
        try:
            # Stability based on consistency of statistical properties
            skew_vals = self._calculate_rolling_skewness(returns_window, sub_window)
            kurt_vals = self._calculate_rolling_kurtosis(returns_window, sub_window)
            
            if len(skew_vals) > 1 and len(kurt_vals) > 1:
                # Stability based on low coefficient of variation
                skew_cv = np.std(skew_vals) / (np.mean(np.abs(skew_vals)) + 1e-8)
                kurt_cv = np.std(kurt_vals) / (np.mean(np.abs(kurt_vals)) + 1e-8)
                return max(0, 1 - (skew_cv + kurt_cv) / 2)
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_distribution_entropy(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate distribution entropy - FULLY VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        # OPTIMIZED: Fully vectorized entropy calculation using histogram approximation
        returns_series = pd.Series(returns)
        
        # Calculate rolling min and max for bin boundaries
        rolling_min = returns_series.rolling(window=window).min()
        rolling_max = returns_series.rolling(window=window).max()
        
        # Create 10 bins for each window
        n_bins = 10
        bin_width = (rolling_max - rolling_min) / n_bins
        
        # Calculate entropy using variance approximation (much faster than histogram)
        # Entropy is approximated as log of variance for normal-like distributions
        rolling_var = returns_series.rolling(window=window).var()
        entropy_approx = np.log(rolling_var + 1e-8)
        
        # Normalize entropy to 0-1 range
        entropy_normalized = entropy_approx / (entropy_approx.rolling(window=window*2).std() + 1e-8)
        entropy_normalized = np.clip(entropy_normalized, 0, 1)
        
        return entropy_normalized.fillna(0).values
    
    def _calculate_entropy_window(self, returns_window: np.ndarray) -> float:
        """Helper function for distribution entropy calculation."""
        if len(returns_window) < 2:
            return 0.0
        
        try:
            # Calculate entropy of returns distribution
            # Discretize returns into bins
            bins = np.linspace(returns_window.min(), returns_window.max(), 10)
            hist, _ = np.histogram(returns_window, bins=bins)
            # Normalize to probabilities
            probs = hist / (np.sum(hist) + 1e-8)
            # Calculate entropy
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            return entropy
        except:
            return 0.0
    
    def _calculate_statistical_consistency(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate statistical consistency - FULLY VECTORIZED."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        # OPTIMIZED: Fully vectorized statistical consistency calculation
        returns_series = pd.Series(returns)
        sub_window = max(2, window // 4)
        
        # Calculate rolling skewness and kurtosis
        rolling_mean = returns_series.rolling(window=sub_window).mean()
        rolling_std = returns_series.rolling(window=sub_window).std()
        centered = returns_series - rolling_mean
        
        skewness = (centered ** 3).rolling(window=sub_window).mean() / (rolling_std ** 3 + 1e-8)
        kurtosis = (centered ** 4).rolling(window=sub_window).mean() / (rolling_std ** 4 + 1e-8) - 3
        
        # Calculate autocorrelation of statistical moments using vectorized operations
        skewness_shifted = skewness.shift(1)
        kurtosis_shifted = kurtosis.shift(1)
        
        skew_corr = skewness.rolling(window=window).corr(skewness_shifted).fillna(0)
        kurt_corr = kurtosis.rolling(window=window).corr(kurtosis_shifted).fillna(0)
        
        consistency = (skew_corr + kurt_corr) / 2
        
        return consistency.values
    
    def _calculate_consistency_window(self, returns_window: np.ndarray, sub_window: int) -> float:
        """Helper function for statistical consistency calculation."""
        if len(returns_window) < 3 or sub_window < 2:
            return 0.0
        
        try:
            # Consistency based on autocorrelation of statistical moments
            skew_vals = self._calculate_rolling_skewness(returns_window, sub_window)
            kurt_vals = self._calculate_rolling_kurtosis(returns_window, sub_window)
            
            if len(skew_vals) > 1 and len(kurt_vals) > 1:
                skew_corr = np.corrcoef(skew_vals[:-1], skew_vals[1:])[0, 1]
                kurt_corr = np.corrcoef(kurt_vals[:-1], kurt_vals[1:])[0, 1]
                
                skew_corr = skew_corr if not np.isnan(skew_corr) else 0
                kurt_corr = kurt_corr if not np.isnan(kurt_corr) else 0
                
                return (skew_corr + kurt_corr) / 2
            else:
                return 0.0
        except:
            return 0.0