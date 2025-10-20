"""
Regime Feature Extractor

This module provides regime-specific feature extraction capabilities for
HDBSCAN-based regime discovery, focusing on features that are particularly
useful for identifying and characterizing market regimes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.signal import find_peaks
import warnings

# Import enhanced hardware optimization tools
from src.utils.hardware import (
    smart_cache, auto_optimize, memory_efficient, performance_tracked,
    optimize_dataframe_default, optimize_numpy_array_default
)

logger = logging.getLogger(__name__)

@dataclass
class RegimeFeatureConfig:
    """Configuration for regime feature extraction."""
    # Basic features
    include_returns: bool = True
    include_volatility: bool = True
    include_volume: bool = True
    
    # Advanced features
    include_entropy: bool = True
    include_spectral: bool = True
    include_pid: bool = True
    include_hybrid: bool = True
    
    # Regime-specific features
    include_regime_persistence: bool = True
    include_regime_transitions: bool = True
    include_regime_volatility: bool = True
    include_regime_trend: bool = True
    
    # Technical parameters
    lookback_window: int = 20
    volatility_window: int = 10
    entropy_window: int = 15
    spectral_window: int = 30
    pid_window: int = 25
    
    # Feature engineering
    include_interactions: bool = True
    include_polynomial: bool = False
    include_ratios: bool = True
    
    # Normalization
    normalize_features: bool = True
    robust_scaling: bool = False

class RegimeFeatureExtractor:
    """
    Extracts regime-specific features for HDBSCAN clustering.
    
    This class focuses on features that are particularly useful for
    identifying market regimes, including regime persistence, transitions,
    and regime-specific statistical measures.
    """
    
    def __init__(self, config: Optional[RegimeFeatureConfig] = None):
        """
        Initialize regime feature extractor.
        
        Args:
            config: Configuration for feature extraction
        """
        self.config = config or RegimeFeatureConfig()
        self.feature_names = []
        self.feature_stats = {}
        
    @smart_cache(ttl=1800)  # Cache features for 30 minutes
    @auto_optimize(optimize_inputs=True, optimize_outputs=True)
    @memory_efficient(memory_threshold_mb=100.0, auto_cleanup=True)
    @performance_tracked(log_performance=True, track_memory=True)
    def extract_features(self, 
                        market_data: pd.DataFrame,
                        existing_features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Extract regime-specific features from market data.
        
        Args:
            market_data: Market data with OHLCV columns
            existing_features: Existing features to augment (optional)
            
        Returns:
            DataFrame with regime-specific features
        """
        try:
            logger.info("🔍 Extracting regime-specific features...")
            
            features_list = []
            feature_names = []
            
            # Basic regime features
            if self.config.include_returns:
                returns_features, returns_names = self._extract_returns_features(market_data)
                features_list.append(returns_features)
                feature_names.extend(returns_names)
            
            if self.config.include_volatility:
                vol_features, vol_names = self._extract_volatility_features(market_data)
                features_list.append(vol_features)
                feature_names.extend(vol_names)
            
            if self.config.include_volume:
                volume_features, volume_names = self._extract_volume_features(market_data)
                features_list.append(volume_features)
                feature_names.extend(volume_names)
            
            # Advanced regime features
            if self.config.include_entropy:
                entropy_features, entropy_names = self._extract_entropy_features(market_data)
                features_list.append(entropy_features)
                feature_names.extend(entropy_names)
            
            if self.config.include_spectral:
                spectral_features, spectral_names = self._extract_spectral_features(market_data)
                features_list.append(spectral_features)
                feature_names.extend(spectral_names)
            
            if self.config.include_pid:
                pid_features, pid_names = self._extract_pid_features(market_data)
                features_list.append(pid_features)
                feature_names.extend(pid_names)
            
            if self.config.include_hybrid:
                hybrid_features, hybrid_names = self._extract_hybrid_features(market_data)
                features_list.append(hybrid_features)
                feature_names.extend(hybrid_names)
            
            # Regime-specific features
            if self.config.include_regime_persistence:
                persistence_features, persistence_names = self._extract_regime_persistence_features(market_data)
                features_list.append(persistence_features)
                feature_names.extend(persistence_names)
            
            if self.config.include_regime_transitions:
                transition_features, transition_names = self._extract_regime_transition_features(market_data)
                features_list.append(transition_features)
                feature_names.extend(transition_names)
            
            if self.config.include_regime_volatility:
                regime_vol_features, regime_vol_names = self._extract_regime_volatility_features(market_data)
                features_list.append(regime_vol_features)
                feature_names.extend(regime_vol_names)
            
            if self.config.include_regime_trend:
                trend_features, trend_names = self._extract_regime_trend_features(market_data)
                features_list.append(trend_features)
                feature_names.extend(trend_names)
            
            # Combine all features
            if features_list:
                regime_features = np.column_stack(features_list)
            else:
                regime_features = np.zeros((len(market_data), 1))
                feature_names = ['regime_placeholder']
            
            # Create DataFrame with hardware optimization
            features_df = pd.DataFrame(regime_features, columns=feature_names, index=market_data.index)
            features_df = optimize_dataframe_default(features_df)
            
            # Add interaction features
            if self.config.include_interactions:
                features_df = self._add_interaction_features(features_df)
                features_df = optimize_dataframe_default(features_df)
            
            # Add polynomial features
            if self.config.include_polynomial:
                features_df = self._add_polynomial_features(features_df)
                features_df = optimize_dataframe_default(features_df)
            
            # Add ratio features
            if self.config.include_ratios:
                features_df = self._add_ratio_features(features_df)
            
            # Combine with existing features
            if existing_features is not None:
                features_df = pd.concat([existing_features, features_df], axis=1)
            
            # Normalize features
            if self.config.normalize_features:
                features_df = self._normalize_features(features_df)
            
            # Store feature names and stats
            self.feature_names = list(features_df.columns)
            self.feature_stats = self._calculate_feature_stats(features_df)
            
            logger.info(f"✅ Extracted {len(self.feature_names)} regime features")
            
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Regime feature extraction failed: {e}")
            # Return empty features as fallback
            return pd.DataFrame(index=market_data.index)
    
    def _extract_returns_features(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract returns-based features."""
        try:
            # Get price data
            prices = self._get_price_data(market_data)
            returns = np.diff(prices) / prices[:-1]
            returns = np.concatenate([[0], returns])  # Add 0 for first period
            
            features = []
            names = []
            
            # Basic return statistics
            features.append(returns)
            names.append('returns')
            
            # Rolling return statistics
            window = self.config.lookback_window
            if len(returns) >= window:
                rolling_mean = pd.Series(returns).rolling(window=window).mean().values
                rolling_std = pd.Series(returns).rolling(window=window).std().values
                rolling_skew = pd.Series(returns).rolling(window=window).skew().values
                rolling_kurt = pd.Series(returns).rolling(window=window).kurt().values
                
                features.extend([rolling_mean, rolling_std, rolling_skew, rolling_kurt])
                names.extend(['returns_rolling_mean', 'returns_rolling_std', 
                            'returns_rolling_skew', 'returns_rolling_kurt'])
            
            # Return momentum
            if len(returns) >= 5:
                momentum_5 = pd.Series(returns).rolling(window=5).sum().values
                momentum_10 = pd.Series(returns).rolling(window=10).sum().values
                
                features.extend([momentum_5, momentum_10])
                names.extend(['returns_momentum_5', 'returns_momentum_10'])
            
            # Return acceleration
            if len(returns) >= 3:
                acceleration = np.diff(returns, n=2)
                acceleration = np.concatenate([[0, 0], acceleration])  # Pad with zeros
                
                features.append(acceleration)
                names.append('returns_acceleration')
            
            return np.column_stack(features), names
            
        except Exception as e:
            logger.error(f"❌ Returns feature extraction failed: {e}")
            return np.zeros((len(market_data), 1)), ['returns_error']
    
    def _extract_volatility_features(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract volatility-based features."""
        try:
            # Get price data
            prices = self._get_price_data(market_data)
            returns = np.diff(prices) / prices[:-1]
            returns = np.concatenate([[0], returns])
            
            features = []
            names = []
            
            # Rolling volatility
            window = self.config.volatility_window
            if len(returns) >= window:
                rolling_vol = pd.Series(returns).rolling(window=window).std().values
                
                features.append(rolling_vol)
                names.append('volatility_rolling')
                
                # Volatility of volatility
                vol_of_vol = pd.Series(rolling_vol).rolling(window=window).std().values
                features.append(vol_of_vol)
                names.append('volatility_of_volatility')
                
                # Volatility clustering
                vol_autocorr = pd.Series(rolling_vol).rolling(window=window).apply(
                    lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
                ).values
                features.append(vol_autocorr)
                names.append('volatility_clustering')
            
            # GARCH-like features
            if len(returns) >= 10:
                # Exponentially weighted volatility
                ew_vol = pd.Series(returns).ewm(span=window).std().values
                features.append(ew_vol)
                names.append('volatility_ewm')
                
                # Volatility regime indicator
                vol_regime = (rolling_vol > np.percentile(rolling_vol[~np.isnan(rolling_vol)], 75)).astype(float)
                features.append(vol_regime)
                names.append('volatility_regime_high')
            
            return np.column_stack(features), names
            
        except Exception as e:
            logger.error(f"❌ Volatility feature extraction failed: {e}")
            return np.zeros((len(market_data), 1)), ['volatility_error']
    
    def _extract_volume_features(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract volume-based features."""
        try:
            # Get volume data
            volume = self._get_volume_data(market_data)
            if volume is None:
                return np.zeros((len(market_data), 1)), ['volume_error']
            
            features = []
            names = []
            
            # Basic volume features
            features.append(volume)
            names.append('volume')
            
            # Volume statistics
            window = self.config.lookback_window
            if len(volume) >= window:
                rolling_mean = pd.Series(volume).rolling(window=window).mean().values
                rolling_std = pd.Series(volume).rolling(window=window).std().values
                
                features.extend([rolling_mean, rolling_std])
                names.extend(['volume_rolling_mean', 'volume_rolling_std'])
                
                # Volume ratio
                volume_ratio = volume / (rolling_mean + 1e-10)
                features.append(volume_ratio)
                names.append('volume_ratio')
                
                # Volume momentum
                volume_momentum = pd.Series(volume).rolling(window=5).sum().values
                features.append(volume_momentum)
                names.append('volume_momentum')
            
            return np.column_stack(features), names
            
        except Exception as e:
            logger.error(f"❌ Volume feature extraction failed: {e}")
            return np.zeros((len(market_data), 1)), ['volume_error']
    
    def _extract_entropy_features(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract entropy-based features."""
        try:
            # Get price data
            prices = self._get_price_data(market_data)
            returns = np.diff(prices) / prices[:-1]
            returns = np.concatenate([[0], returns])
            
            features = []
            names = []
            
            window = self.config.entropy_window
            if len(returns) >= window:
                # Shannon entropy of returns
                shannon_entropy = self._calculate_shannon_entropy(returns, window)
                features.append(shannon_entropy)
                names.append('shannon_entropy')
                
                # Approximate entropy
                approx_entropy = self._calculate_approximate_entropy(returns, window)
                features.append(approx_entropy)
                names.append('approximate_entropy')
                
                # Sample entropy
                sample_entropy = self._calculate_sample_entropy(returns, window)
                features.append(sample_entropy)
                names.append('sample_entropy')
            
            return np.column_stack(features), names
            
        except Exception as e:
            logger.error(f"❌ Entropy feature extraction failed: {e}")
            return np.zeros((len(market_data), 1)), ['entropy_error']
    
    def _extract_spectral_features(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract spectral features."""
        try:
            # Get price data
            prices = self._get_price_data(market_data)
            returns = np.diff(prices) / prices[:-1]
            returns = np.concatenate([[0], returns])
            
            features = []
            names = []
            
            window = self.config.spectral_window
            if len(returns) >= window:
                # Power spectral density
                psd_features = self._calculate_psd_features(returns, window)
                features.extend(psd_features)
                names.extend(['psd_peak_freq', 'psd_peak_power', 'psd_total_power'])
                
                # Spectral centroid
                spectral_centroid = self._calculate_spectral_centroid(returns, window)
                features.append(spectral_centroid)
                names.append('spectral_centroid')
                
                # Spectral rolloff
                spectral_rolloff = self._calculate_spectral_rolloff(returns, window)
                features.append(spectral_rolloff)
                names.append('spectral_rolloff')
            
            return np.column_stack(features), names
            
        except Exception as e:
            logger.error(f"❌ Spectral feature extraction failed: {e}")
            return np.zeros((len(market_data), 1)), ['spectral_error']
    
    def _extract_pid_features(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract PID controller features."""
        try:
            # Get price data
            prices = self._get_price_data(market_data)
            returns = np.diff(prices) / prices[:-1]
            returns = np.concatenate([[0], returns])
            
            features = []
            names = []
            
            window = self.config.pid_window
            if len(returns) >= window:
                # PID controller features
                pid_features = self._calculate_pid_features(returns, window)
                features.extend(pid_features)
                names.extend(['pid_proportional', 'pid_integral', 'pid_derivative'])
                
                # PID stability
                pid_stability = self._calculate_pid_stability(returns, window)
                features.append(pid_stability)
                names.append('pid_stability')
            
            return np.column_stack(features), names
            
        except Exception as e:
            logger.error(f"❌ PID feature extraction failed: {e}")
            return np.zeros((len(market_data), 1)), ['pid_error']
    
    def _extract_hybrid_features(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract hybrid features combining multiple approaches."""
        try:
            # Get price data
            prices = self._get_price_data(market_data)
            returns = np.diff(prices) / prices[:-1]
            returns = np.concatenate([[0], returns])
            
            features = []
            names = []
            
            window = self.config.lookback_window
            if len(returns) >= window:
                # Regime change indicator
                regime_change = self._calculate_regime_change_indicator(returns, window)
                features.append(regime_change)
                names.append('regime_change_indicator')
                
                # Market stress indicator
                stress_indicator = self._calculate_market_stress(returns, window)
                features.append(stress_indicator)
                names.append('market_stress')
                
                # Trend strength
                trend_strength = self._calculate_trend_strength(returns, window)
                features.append(trend_strength)
                names.append('trend_strength')
            
            return np.column_stack(features), names
            
        except Exception as e:
            logger.error(f"❌ Hybrid feature extraction failed: {e}")
            return np.zeros((len(market_data), 1)), ['hybrid_error']
    
    def _extract_regime_persistence_features(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract regime persistence features."""
        try:
            # Get price data
            prices = self._get_price_data(market_data)
            returns = np.diff(prices) / prices[:-1]
            returns = np.concatenate([[0], returns])
            
            features = []
            names = []
            
            window = self.config.lookback_window
            if len(returns) >= window:
                # Regime persistence measure
                persistence = self._calculate_regime_persistence(returns, window)
                features.append(persistence)
                names.append('regime_persistence')
                
                # Regime stability
                stability = self._calculate_regime_stability(returns, window)
                features.append(stability)
                names.append('regime_stability')
                
                # Regime duration
                duration = self._calculate_regime_duration(returns, window)
                features.append(duration)
                names.append('regime_duration')
            
            return np.column_stack(features), names
            
        except Exception as e:
            logger.error(f"❌ Regime persistence feature extraction failed: {e}")
            return np.zeros((len(market_data), 1)), ['persistence_error']
    
    def _extract_regime_transition_features(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract regime transition features."""
        try:
            # Get price data
            prices = self._get_price_data(market_data)
            returns = np.diff(prices) / prices[:-1]
            returns = np.concatenate([[0], returns])
            
            features = []
            names = []
            
            window = self.config.lookback_window
            if len(returns) >= window:
                # Transition probability
                transition_prob = self._calculate_transition_probability(returns, window)
                features.append(transition_prob)
                names.append('transition_probability')
                
                # Transition intensity
                transition_intensity = self._calculate_transition_intensity(returns, window)
                features.append(transition_intensity)
                names.append('transition_intensity')
                
                # Transition smoothness
                transition_smoothness = self._calculate_transition_smoothness(returns, window)
                features.append(transition_smoothness)
                names.append('transition_smoothness')
            
            return np.column_stack(features), names
            
        except Exception as e:
            logger.error(f"❌ Regime transition feature extraction failed: {e}")
            return np.zeros((len(market_data), 1)), ['transition_error']
    
    def _extract_regime_volatility_features(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract regime-specific volatility features."""
        try:
            # Get price data
            prices = self._get_price_data(market_data)
            returns = np.diff(prices) / prices[:-1]
            returns = np.concatenate([[0], returns])
            
            features = []
            names = []
            
            window = self.config.volatility_window
            if len(returns) >= window:
                # Regime volatility clustering
                vol_clustering = self._calculate_volatility_clustering(returns, window)
                features.append(vol_clustering)
                names.append('regime_vol_clustering')
                
                # Regime volatility persistence
                vol_persistence = self._calculate_volatility_persistence(returns, window)
                features.append(vol_persistence)
                names.append('regime_vol_persistence')
                
                # Regime volatility regime
                vol_regime = self._calculate_volatility_regime(returns, window)
                features.append(vol_regime)
                names.append('regime_vol_regime')
            
            return np.column_stack(features), names
            
        except Exception as e:
            logger.error(f"❌ Regime volatility feature extraction failed: {e}")
            return np.zeros((len(market_data), 1)), ['regime_vol_error']
    
    def _extract_regime_trend_features(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract regime-specific trend features."""
        try:
            # Get price data
            prices = self._get_price_data(market_data)
            returns = np.diff(prices) / prices[:-1]
            returns = np.concatenate([[0], returns])
            
            features = []
            names = []
            
            window = self.config.lookback_window
            if len(returns) >= window:
                # Trend strength
                trend_strength = self._calculate_trend_strength(returns, window)
                features.append(trend_strength)
                names.append('regime_trend_strength')
                
                # Trend persistence
                trend_persistence = self._calculate_trend_persistence(returns, window)
                features.append(trend_persistence)
                names.append('regime_trend_persistence')
                
                # Trend regime
                trend_regime = self._calculate_trend_regime(returns, window)
                features.append(trend_regime)
                names.append('regime_trend_regime')
            
            return np.column_stack(features), names
            
        except Exception as e:
            logger.error(f"❌ Regime trend feature extraction failed: {e}")
            return np.zeros((len(market_data), 1)), ['regime_trend_error']
    
    def _get_price_data(self, market_data: pd.DataFrame) -> np.ndarray:
        """Get price data from market data."""
        try:
            if 'close' in market_data.columns:
                return market_data['close'].values
            elif 'Close' in market_data.columns:
                return market_data['Close'].values
            else:
                # Try to find price column
                price_cols = [col for col in market_data.columns if 'close' in col.lower()]
                if price_cols:
                    return market_data[price_cols[0]].values
                else:
                    raise ValueError("No price column found")
        except Exception as e:
            logger.error(f"❌ Price data extraction failed: {e}")
            return np.zeros(len(market_data))
    
    def _get_volume_data(self, market_data: pd.DataFrame) -> Optional[np.ndarray]:
        """Get volume data from market data."""
        try:
            if 'volume' in market_data.columns:
                return market_data['volume'].values
            elif 'Volume' in market_data.columns:
                return market_data['Volume'].values
            else:
                # Try to find volume column
                volume_cols = [col for col in market_data.columns if 'volume' in col.lower()]
                if volume_cols:
                    return market_data[volume_cols[0]].values
                else:
                    return None
        except Exception as e:
            logger.error(f"❌ Volume data extraction failed: {e}")
            return None
    
    def _calculate_shannon_entropy(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate Shannon entropy."""
        try:
            entropy = np.zeros(len(data))
            for i in range(window, len(data)):
                window_data = data[i-window:i]
                # Discretize data into bins
                hist, _ = np.histogram(window_data, bins=10)
                hist = hist / hist.sum()
                hist = hist[hist > 0]  # Remove zero probabilities
                entropy[i] = -np.sum(hist * np.log2(hist))
            return entropy
        except Exception as e:
            logger.debug(f"Shannon entropy calculation failed: {e}")
            return np.zeros(len(data))
    
    def _calculate_approximate_entropy(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate approximate entropy."""
        try:
            entropy = np.zeros(len(data))
            m = 2  # Embedding dimension
            r = 0.2 * np.std(data)  # Tolerance
            
            for i in range(window, len(data)):
                window_data = data[i-window:i]
                if len(window_data) < m + 1:
                    continue
                
                # Calculate phi functions
                phi_m = self._calculate_phi(data, m, r)
                phi_m1 = self._calculate_phi(data, m + 1, r)
                
                if phi_m > 0 and phi_m1 > 0:
                    entropy[i] = phi_m - phi_m1
                else:
                    entropy[i] = 0
            
            return entropy
        except Exception as e:
            logger.debug(f"Approximate entropy calculation failed: {e}")
            return np.zeros(len(data))
    
    def _calculate_sample_entropy(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate sample entropy."""
        try:
            entropy = np.zeros(len(data))
            m = 2  # Embedding dimension
            r = 0.2 * np.std(data)  # Tolerance
            
            for i in range(window, len(data)):
                window_data = data[i-window:i]
                if len(window_data) < m + 1:
                    continue
                
                # Calculate sample entropy
                entropy[i] = self._calculate_sample_entropy_single(window_data, m, r)
            
            return entropy
        except Exception as e:
            logger.debug(f"Sample entropy calculation failed: {e}")
            return np.zeros(len(data))
    
    def _calculate_phi(self, data: np.ndarray, m: int, r: float) -> float:
        """Calculate phi function for approximate entropy."""
        try:
            N = len(data)
            patterns = np.array([data[i:i+m] for i in range(N-m+1)])
            
            # Calculate distances
            distances = np.abs(patterns[:, np.newaxis] - patterns[np.newaxis, :])
            max_distances = np.max(distances, axis=2)
            
            # Count matches
            matches = np.sum(max_distances <= r, axis=1)
            matches = matches[matches > 0]
            
            if len(matches) == 0:
                return 0
            
            phi = np.mean(np.log(matches / (N - m + 1)))
            return phi
        except Exception as e:
            logger.debug(f"Phi calculation failed: {e}")
            return 0
    
    def _calculate_sample_entropy_single(self, data: np.ndarray, m: int, r: float) -> float:
        """Calculate sample entropy for a single window."""
        try:
            N = len(data)
            if N < m + 1:
                return 0
            
            # Create patterns
            patterns_m = np.array([data[i:i+m] for i in range(N-m+1)])
            patterns_m1 = np.array([data[i:i+m+1] for i in range(N-m)])
            
            # Calculate distances for m patterns
            distances_m = np.abs(patterns_m[:, np.newaxis] - patterns_m[np.newaxis, :])
            max_distances_m = np.max(distances_m, axis=2)
            
            # Calculate distances for m+1 patterns
            distances_m1 = np.abs(patterns_m1[:, np.newaxis] - patterns_m1[np.newaxis, :])
            max_distances_m1 = np.max(distances_m1, axis=2)
            
            # Count matches
            matches_m = np.sum(max_distances_m <= r, axis=1) - 1  # Exclude self
            matches_m1 = np.sum(max_distances_m1 <= r, axis=1) - 1  # Exclude self
            
            # Calculate probabilities
            phi_m = np.mean(np.log(matches_m / (N - m)))
            phi_m1 = np.mean(np.log(matches_m1 / (N - m)))
            
            if phi_m > 0 and phi_m1 > 0:
                return phi_m - phi_m1
            else:
                return 0
        except Exception as e:
            logger.debug(f"Sample entropy calculation failed: {e}")
            return 0
    
    def _calculate_psd_features(self, data: np.ndarray, window: int) -> List[np.ndarray]:
        """Calculate power spectral density features."""
        try:
            from scipy.signal import welch
            
            peak_freq = np.zeros(len(data))
            peak_power = np.zeros(len(data))
            total_power = np.zeros(len(data))
            
            for i in range(window, len(data)):
                window_data = data[i-window:i]
                if len(window_data) < 10:
                    continue
                
                # Calculate PSD
                freqs, psd = welch(window_data, nperseg=min(len(window_data), 16))
                
                # Find peak frequency
                peak_idx = np.argmax(psd)
                peak_freq[i] = freqs[peak_idx]
                peak_power[i] = psd[peak_idx]
                total_power[i] = np.sum(psd)
            
            return [peak_freq, peak_power, total_power]
        except Exception as e:
            logger.debug(f"PSD calculation failed: {e}")
            return [np.zeros(len(data)), np.zeros(len(data)), np.zeros(len(data))]
    
    def _calculate_spectral_centroid(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate spectral centroid."""
        try:
            from scipy.signal import welch
            
            centroid = np.zeros(len(data))
            
            for i in range(window, len(data)):
                window_data = data[i-window:i]
                if len(window_data) < 10:
                    continue
                
                # Calculate PSD
                freqs, psd = welch(window_data, nperseg=min(len(window_data), 16))
                
                # Calculate centroid
                if np.sum(psd) > 0:
                    centroid[i] = np.sum(freqs * psd) / np.sum(psd)
                else:
                    centroid[i] = 0
            
            return centroid
        except Exception as e:
            logger.debug(f"Spectral centroid calculation failed: {e}")
            return np.zeros(len(data))
    
    def _calculate_spectral_rolloff(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate spectral rolloff."""
        try:
            from scipy.signal import welch
            
            rolloff = np.zeros(len(data))
            
            for i in range(window, len(data)):
                window_data = data[i-window:i]
                if len(window_data) < 10:
                    continue
                
                # Calculate PSD
                freqs, psd = welch(window_data, nperseg=min(len(window_data), 16))
                
                # Calculate rolloff (85% of total power)
                cumsum_psd = np.cumsum(psd)
                total_power = cumsum_psd[-1]
                rolloff_threshold = 0.85 * total_power
                
                rolloff_idx = np.where(cumsum_psd >= rolloff_threshold)[0]
                if len(rolloff_idx) > 0:
                    rolloff[i] = freqs[rolloff_idx[0]]
                else:
                    rolloff[i] = freqs[-1]
            
            return rolloff
        except Exception as e:
            logger.debug(f"Spectral rolloff calculation failed: {e}")
            return np.zeros(len(data))
    
    def _calculate_pid_features(self, data: np.ndarray, window: int) -> List[np.ndarray]:
        """Calculate PID controller features."""
        try:
            proportional = np.zeros(len(data))
            integral = np.zeros(len(data))
            derivative = np.zeros(len(data))
            
            for i in range(window, len(data)):
                window_data = data[i-window:i]
                if len(window_data) < 3:
                    continue
                
                # PID components
                error = window_data[-1]  # Current error
                integral[i] = np.sum(window_data)  # Integral of error
                derivative[i] = window_data[-1] - window_data[-2]  # Derivative of error
                proportional[i] = error  # Proportional term
            
            return [proportional, integral, derivative]
        except Exception as e:
            logger.debug(f"PID calculation failed: {e}")
            return [np.zeros(len(data)), np.zeros(len(data)), np.zeros(len(data))]
    
    def _calculate_pid_stability(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate PID stability measure."""
        try:
            stability = np.zeros(len(data))
            
            for i in range(window, len(data)):
                window_data = data[i-window:i]
                if len(window_data) < 5:
                    continue
                
                # Calculate stability as inverse of variance
                stability[i] = 1.0 / (np.var(window_data) + 1e-10)
            
            return stability
        except Exception as e:
            logger.debug(f"PID stability calculation failed: {e}")
            return np.zeros(len(data))
    
    def _calculate_regime_change_indicator(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate regime change indicator."""
        try:
            indicator = np.zeros(len(data))
            
            for i in range(window, len(data)):
                window_data = data[i-window:i]
                if len(window_data) < 5:
                    continue
                
                # Calculate regime change as change in statistical properties
                first_half = window_data[:len(window_data)//2]
                second_half = window_data[len(window_data)//2:]
                
                if len(first_half) > 0 and len(second_half) > 0:
                    mean_change = abs(np.mean(second_half) - np.mean(first_half))
                    std_change = abs(np.std(second_half) - np.std(first_half))
                    indicator[i] = mean_change + std_change
            
            return indicator
        except Exception as e:
            logger.debug(f"Regime change indicator calculation failed: {e}")
            return np.zeros(len(data))
    
    def _calculate_market_stress(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate market stress indicator."""
        try:
            stress = np.zeros(len(data))
            
            for i in range(window, len(data)):
                window_data = data[i-window:i]
                if len(window_data) < 5:
                    continue
                
                # Calculate stress as combination of volatility and extreme values
                volatility = np.std(window_data)
                extreme_values = np.sum(np.abs(window_data) > 2 * np.std(window_data))
                stress[i] = volatility * (1 + extreme_values / len(window_data))
            
            return stress
        except Exception as e:
            logger.debug(f"Market stress calculation failed: {e}")
            return np.zeros(len(data))
    
    def _calculate_trend_strength(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend strength."""
        try:
            strength = np.zeros(len(data))
            
            for i in range(window, len(data)):
                window_data = data[i-window:i]
                if len(window_data) < 3:
                    continue
                
                # Calculate trend strength as correlation with time
                time_indices = np.arange(len(window_data))
                correlation = np.corrcoef(time_indices, window_data)[0, 1]
                strength[i] = abs(correlation) if not np.isnan(correlation) else 0
            
            return strength
        except Exception as e:
            logger.debug(f"Trend strength calculation failed: {e}")
            return np.zeros(len(data))
    
    def _calculate_regime_persistence(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate regime persistence."""
        try:
            persistence = np.zeros(len(data))
            
            for i in range(window, len(data)):
                window_data = data[i-window:i]
                if len(window_data) < 5:
                    continue
                
                # Calculate persistence as autocorrelation
                if len(window_data) > 1:
                    autocorr = np.corrcoef(window_data[:-1], window_data[1:])[0, 1]
                    persistence[i] = autocorr if not np.isnan(autocorr) else 0
                else:
                    persistence[i] = 0
            
            return persistence
        except Exception as e:
            logger.debug(f"Regime persistence calculation failed: {e}")
            return np.zeros(len(data))
    
    def _calculate_regime_stability(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate regime stability."""
        try:
            stability = np.zeros(len(data))
            
            for i in range(window, len(data)):
                window_data = data[i-window:i]
                if len(window_data) < 5:
                    continue
                
                # Calculate stability as inverse of variance
                stability[i] = 1.0 / (np.var(window_data) + 1e-10)
            
            return stability
        except Exception as e:
            logger.debug(f"Regime stability calculation failed: {e}")
            return np.zeros(len(data))
    
    def _calculate_regime_duration(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate regime duration."""
        try:
            duration = np.zeros(len(data))
            
            for i in range(window, len(data)):
                window_data = data[i-window:i]
                if len(window_data) < 5:
                    continue
                
                # Calculate duration as length of current regime
                # This is a simplified calculation
                duration[i] = len(window_data)
            
            return duration
        except Exception as e:
            logger.debug(f"Regime duration calculation failed: {e}")
            return np.zeros(len(data))
    
    def _calculate_transition_probability(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate transition probability."""
        try:
            prob = np.zeros(len(data))
            
            for i in range(window, len(data)):
                window_data = data[i-window:i]
                if len(window_data) < 5:
                    continue
                
                # Calculate transition probability as change frequency
                changes = np.sum(np.diff(window_data) != 0)
                prob[i] = changes / (len(window_data) - 1)
            
            return prob
        except Exception as e:
            logger.debug(f"Transition probability calculation failed: {e}")
            return np.zeros(len(data))
    
    def _calculate_transition_intensity(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate transition intensity."""
        try:
            intensity = np.zeros(len(data))
            
            for i in range(window, len(data)):
                window_data = data[i-window:i]
                if len(window_data) < 5:
                    continue
                
                # Calculate intensity as magnitude of changes
                changes = np.abs(np.diff(window_data))
                intensity[i] = np.mean(changes) if len(changes) > 0 else 0
            
            return intensity
        except Exception as e:
            logger.debug(f"Transition intensity calculation failed: {e}")
            return np.zeros(len(data))
    
    def _calculate_transition_smoothness(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate transition smoothness."""
        try:
            smoothness = np.zeros(len(data))
            
            for i in range(window, len(data)):
                window_data = data[i-window:i]
                if len(window_data) < 5:
                    continue
                
                # Calculate smoothness as inverse of second derivative
                second_deriv = np.diff(window_data, n=2)
                smoothness[i] = 1.0 / (np.var(second_deriv) + 1e-10)
            
            return smoothness
        except Exception as e:
            logger.debug(f"Transition smoothness calculation failed: {e}")
            return np.zeros(len(data))
    
    def _calculate_volatility_clustering(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate volatility clustering."""
        try:
            clustering = np.zeros(len(data))
            
            for i in range(window, len(data)):
                window_data = data[i-window:i]
                if len(window_data) < 5:
                    continue
                
                # Calculate volatility clustering as autocorrelation of squared returns
                squared_returns = window_data ** 2
                if len(squared_returns) > 1:
                    autocorr = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
                    clustering[i] = autocorr if not np.isnan(autocorr) else 0
                else:
                    clustering[i] = 0
            
            return clustering
        except Exception as e:
            logger.debug(f"Volatility clustering calculation failed: {e}")
            return np.zeros(len(data))
    
    def _calculate_volatility_persistence(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate volatility persistence."""
        try:
            persistence = np.zeros(len(data))
            
            for i in range(window, len(data)):
                window_data = data[i-window:i]
                if len(window_data) < 5:
                    continue
                
                # Calculate persistence as autocorrelation of volatility
                volatility = np.abs(window_data)
                if len(volatility) > 1:
                    autocorr = np.corrcoef(volatility[:-1], volatility[1:])[0, 1]
                    persistence[i] = autocorr if not np.isnan(autocorr) else 0
                else:
                    persistence[i] = 0
            
            return persistence
        except Exception as e:
            logger.debug(f"Volatility persistence calculation failed: {e}")
            return np.zeros(len(data))
    
    def _calculate_volatility_regime(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate volatility regime."""
        try:
            regime = np.zeros(len(data))
            
            for i in range(window, len(data)):
                window_data = data[i-window:i]
                if len(window_data) < 5:
                    continue
                
                # Calculate volatility regime as high/low indicator
                volatility = np.std(window_data)
                threshold = np.percentile(np.abs(window_data), 75)
                regime[i] = 1 if volatility > threshold else 0
            
            return regime
        except Exception as e:
            logger.debug(f"Volatility regime calculation failed: {e}")
            return np.zeros(len(data))
    
    def _calculate_trend_persistence(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend persistence."""
        try:
            persistence = np.zeros(len(data))
            
            for i in range(window, len(data)):
                window_data = data[i-window:i]
                if len(window_data) < 5:
                    continue
                
                # Calculate trend persistence as consistency of direction
                signs = np.sign(window_data)
                consistency = np.sum(signs == signs[0]) / len(signs)
                persistence[i] = consistency
            
            return persistence
        except Exception as e:
            logger.debug(f"Trend persistence calculation failed: {e}")
            return np.zeros(len(data))
    
    def _calculate_trend_regime(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend regime."""
        try:
            regime = np.zeros(len(data))
            
            for i in range(window, len(data)):
                window_data = data[i-window:i]
                if len(window_data) < 5:
                    continue
                
                # Calculate trend regime as bullish/bearish indicator
                mean_return = np.mean(window_data)
                regime[i] = 1 if mean_return > 0 else -1
            
            return regime
        except Exception as e:
            logger.debug(f"Trend regime calculation failed: {e}")
            return np.zeros(len(data))
    
    def _add_interaction_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features."""
        try:
            interaction_features = []
            interaction_names = []
            
            # Get numeric columns
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            
            # Add pairwise interactions for top features
            if len(numeric_cols) > 1:
                # Select top 5 features by variance
                variances = features_df[numeric_cols].var()
                top_features = variances.nlargest(5).index
                
                for i, col1 in enumerate(top_features):
                    for col2 in top_features[i+1:]:
                        interaction = features_df[col1] * features_df[col2]
                        interaction_features.append(interaction)
                        interaction_names.append(f"{col1}_x_{col2}")
            
            # Add interaction features to DataFrame
            if interaction_features:
                interaction_df = pd.DataFrame(
                    np.column_stack(interaction_features),
                    columns=interaction_names,
                    index=features_df.index
                )
                features_df = pd.concat([features_df, interaction_df], axis=1)
            
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Interaction features failed: {e}")
            return features_df
    
    def _add_polynomial_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add polynomial features."""
        try:
            polynomial_features = []
            polynomial_names = []
            
            # Get numeric columns
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            
            # Add quadratic terms for top features
            if len(numeric_cols) > 0:
                # Select top 3 features by variance
                variances = features_df[numeric_cols].var()
                top_features = variances.nlargest(3).index
                
                for col in top_features:
                    quadratic = features_df[col] ** 2
                    polynomial_features.append(quadratic)
                    polynomial_names.append(f"{col}_squared")
            
            # Add polynomial features to DataFrame
            if polynomial_features:
                polynomial_df = pd.DataFrame(
                    np.column_stack(polynomial_features),
                    columns=polynomial_names,
                    index=features_df.index
                )
                features_df = pd.concat([features_df, polynomial_df], axis=1)
            
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Polynomial features failed: {e}")
            return features_df
    
    def _add_ratio_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add ratio features."""
        try:
            ratio_features = []
            ratio_names = []
            
            # Get numeric columns
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            
            # Add ratios for top features
            if len(numeric_cols) > 1:
                # Select top 4 features by variance
                variances = features_df[numeric_cols].var()
                top_features = variances.nlargest(4).index
                
                for i, col1 in enumerate(top_features):
                    for col2 in top_features[i+1:]:
                        ratio = features_df[col1] / (features_df[col2] + 1e-10)
                        ratio_features.append(ratio)
                        ratio_names.append(f"{col1}_div_{col2}")
            
            # Add ratio features to DataFrame
            if ratio_features:
                ratio_df = pd.DataFrame(
                    np.column_stack(ratio_features),
                    columns=ratio_names,
                    index=features_df.index
                )
                features_df = pd.concat([features_df, ratio_df], axis=1)
            
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Ratio features failed: {e}")
            return features_df
    
    def _normalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features."""
        try:
            if self.config.robust_scaling:
                # Use robust scaling
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                features_df[numeric_cols] = scaler.fit_transform(features_df[numeric_cols])
            else:
                # Use standard scaling
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                features_df[numeric_cols] = scaler.fit_transform(features_df[numeric_cols])
            
            return features_df
            
        except Exception as e:
            logger.error(f"❌ Feature normalization failed: {e}")
            return features_df
    
    def _calculate_feature_stats(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate feature statistics."""
        try:
            stats = {}
            
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                stats[col] = {
                    'mean': features_df[col].mean(),
                    'std': features_df[col].std(),
                    'min': features_df[col].min(),
                    'max': features_df[col].max(),
                    'skewness': features_df[col].skew(),
                    'kurtosis': features_df[col].kurtosis()
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Feature stats calculation failed: {e}")
            return {}
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names.copy()
    
    def get_feature_stats(self) -> Dict[str, Any]:
        """Get feature statistics."""
        return self.feature_stats.copy()