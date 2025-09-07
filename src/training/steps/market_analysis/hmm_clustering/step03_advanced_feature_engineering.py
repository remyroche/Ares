#!/usr/bin/env python3
"""Advanced Feature Engineering for ML Transition Detection.

from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

This module provides sophisticated feature engineering alternatives using
domain-specific knowledge, advanced signal processing, and automated feature
generation techniques.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Import centralized systems
from .step03_imports import get_import_manager, safe_import, check_feature_availability
from .step03_config import Step03Config
from .step03_memory_manager import get_memory_manager, memory_aware_processing
from .step03_technical_indicators import get_technical_indicators


@dataclass
class FeatureEngineeringConfig:
    """Configuration for advanced feature engineering."""
    
    # Signal Processing Features
    enable_fourier_features: bool = True
    enable_wavelet_features: bool = True
    enable_spectral_features: bool = True
    
    # Domain-Specific Features
    enable_market_microstructure: bool = True
    enable_regime_transition_features: bool = True
    enable_volatility_regime_features: bool = True
    
    # Advanced Statistical Features
    enable_higher_moments: bool = True
    enable_entropy_features: bool = True
    enable_fractal_features: bool = True
    
    # Automated Feature Generation
    enable_polynomial_features: bool = True
    enable_interaction_features: bool = True
    enable_ratio_features: bool = True
    
    # Feature Selection
    max_features: int = 200
    feature_selection_method: str = 'mutual_info'  # 'mutual_info', 'f_score', 'variance'
    correlation_threshold: float = 0.95
    
    # Signal Processing Parameters
    fourier_frequencies: int = 20
    wavelet_family: str = 'db4'
    wavelet_levels: int = 4
    
    # Market Microstructure Parameters
    tick_size: float = 0.01
    min_trade_size: float = 100.0
    
    # Regime Transition Parameters
    transition_window: int = 10
    prediction_horizon: int = 5


class AdvancedFeatureEngineer:
    """Advanced feature engineering for regime transition detection."""
    @log_important_calls
    
    def __init__(self, config: Step03Config):
        self.config = config
        self.feature_config = FeatureEngineeringConfig()
        self.logger = logging.getLogger('AdvancedFeatureEngineer')
        self.memory_manager = get_memory_manager(config.memory.__dict__)
        self.technical_indicators = get_technical_indicators()
        
        # Feature storage
        self.feature_cache = {}
        self.feature_importance = {}
    @log_all_calls
        
    def _calculate_fourier_features(self, data: pd.Series, n_frequencies: int = 20) -> pd.DataFrame:
        """Calculate Fourier transform features."""
        features = pd.DataFrame(index = data.index)
        
        try:
            # Remove NaN values
            clean_data = data.dropna()
            if len(clean_data) < 50:  # Need minimum data for FFT
                return features
            
            # Apply FFT
            fft_values = fft(clean_data.values)
            frequencies = fftfreq(len(clean_data))
            
            # Get positive frequencies only
            positive_freq_mask = frequencies > 0
            positive_frequencies = frequencies[positive_freq_mask]
            positive_fft = fft_values[positive_freq_mask]
            
            # Calculate power spectral density
            psd = np.abs(positive_fft) ** 2
            
            # Select top frequencies
            top_freq_indices = np.argsort(psd)[-n_frequencies:]
            top_frequencies = positive_frequencies[top_freq_indices]
            top_powers = psd[top_freq_indices]
            
            # Create features
            for i, (freq, power) in enumerate(zip(top_frequencies, top_powers)):
                features[f'fourier_freq_{i}'] = freq
                features[f'fourier_power_{i}'] = power
            
            # Statistical features of frequency domain
            features['fourier_peak_frequency'] = top_frequencies[np.argmax(top_powers)]
            features['fourier_total_power'] = np.sum(psd)
            features['fourier_spectral_centroid'] = np.sum(positive_frequencies * psd) / np.sum(psd)
            features['fourier_spectral_bandwidth'] = np.sqrt(
                np.sum(((positive_frequencies - features['fourier_spectral_centroid'].iloc[0]) ** 2) * psd) / np.sum(psd)
            )
            
        except Exception as e:
            self.logger.warning(f"Fourier feature calculation failed: {e}")
        
        return features
    @log_all_calls
    
    def _calculate_wavelet_features(self, data: pd.Series) -> pd.DataFrame:
        """Calculate wavelet transform features."""
        features = pd.DataFrame(index = data.index)
        
        try:
            # Import PyWavelets if available
            pywt = safe_import('pywt')
            if not pywt:
                self.logger.warning("PyWavelets not available, skipping wavelet features")
                return features
            
            # Remove NaN values
            clean_data = data.dropna()
            if len(clean_data) < 32:  # Need minimum data for wavelets
                return features
            
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(clean_data.values, self.feature_config.wavelet_family, 
                                level = self.feature_config.wavelet_levels)
            
            # Extract features from each level
            for level, coeff in enumerate(coeffs):
                if len(coeff) > 0:
                    features[f'wavelet_energy_level_{level}'] = np.sum(coeff ** 2)
                    features[f'wavelet_std_level_{level}'] = np.std(coeff)
                    features[f'wavelet_mean_level_{level}'] = np.mean(coeff)
                    features[f'wavelet_skewness_level_{level}'] = stats.skew(coeff)
                    features[f'wavelet_kurtosis_level_{level}'] = stats.kurtosis(coeff)
            
            # Cross-level features
            if len(coeffs) > 1:
                features['wavelet_energy_ratio_high_low'] = (
                    features['wavelet_energy_level_0'] / features[f'wavelet_energy_level_{len(coeffs)-1}']
                )
            
        except Exception as e:
            self.logger.warning(f"Wavelet feature calculation failed: {e}")
        
        return features
    @log_all_calls
    
    def _calculate_spectral_features(self, data: pd.Series) -> pd.DataFrame:
        """Calculate spectral analysis features."""
        features = pd.DataFrame(index = data.index)
        
        try:
            # Remove NaN values
            clean_data = data.dropna()
            if len(clean_data) < 50:
                return features
            
            # Calculate power spectral density using Welch's method
            frequencies, psd = signal.welch(clean_data.values, nperseg = min(256, len(clean_data)//4))
            
            # Spectral features
            features['spectral_peak_frequency'] = frequencies[np.argmax(psd)]
            features['spectral_peak_power'] = np.max(psd)
            features['spectral_total_power'] = np.sum(psd)
            features['spectral_mean_frequency'] = np.sum(frequencies * psd) / np.sum(psd)
            features['spectral_bandwidth'] = np.sqrt(
                np.sum(((frequencies - features['spectral_mean_frequency'].iloc[0]) ** 2) * psd) / np.sum(psd)
            )
            
            # Spectral rolloff (frequency below which 85% of power lies)
            cumulative_power = np.cumsum(psd)
            total_power = cumulative_power[-1]
            rolloff_threshold = 0.85 * total_power
            rolloff_index = np.where(cumulative_power >= rolloff_threshold)[0]
            if len(rolloff_index) > 0:
                features['spectral_rolloff'] = frequencies[rolloff_index[0]]
            else:
                features['spectral_rolloff'] = frequencies[-1]
            
            # Spectral flatness (Wiener entropy)
            geometric_mean = np.exp(np.mean(np.log(psd + 1e-10)))
            arithmetic_mean = np.mean(psd)
            features['spectral_flatness'] = geometric_mean / arithmetic_mean
            
        except Exception as e:
            self.logger.warning(f"Spectral feature calculation failed: {e}")
        
        return features
    @log_all_calls
    
    def _calculate_market_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market microstructure features."""
        features = pd.DataFrame(index = data.index)
        
        try:
            # Price impact features
            if 'volume' in data.columns and 'close' in data.columns:
                # Volume-weighted average price (VWAP)
                features['vwap'] = (data['close'] * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()
                
                # Price impact (price change per unit volume)
                price_change = data['close'].diff()
                volume_change = data['volume'].diff()
                features['price_impact'] = price_change / (volume_change + 1e-10)
                
                # Volume-price trend
                features['vpt'] = (price_change * data['volume']).cumsum()
                
                # On-balance volume
                features['obv'] = (data['volume'] * np.sign(price_change)).cumsum()
            
            # Bid-ask spread proxy (using high-low range)
            if 'high' in data.columns and 'low' in data.columns:
                features['spread_proxy'] = (data['high'] - data['low']) / data['close']
                features['spread_volatility'] = features['spread_proxy'].rolling(20).std()
            
            # Order flow imbalance (using volume and price direction)
            if 'volume' in data.columns and 'close' in data.columns:
                price_direction = np.sign(data['close'].diff())
                features['order_flow_imbalance'] = (data['volume'] * price_direction).rolling(10).sum()
            
            # Market depth proxy (using volume distribution)
            if 'volume' in data.columns:
                features['volume_imbalance'] = (data['volume'] - data['volume'].rolling(20).mean()) / data['volume'].rolling(20).std()
                features['volume_clustering'] = data['volume'].rolling(10).apply(lambda x: len(np.unique(np.round(x, 2))))
            
        except Exception as e:
            self.logger.warning(f"Market microstructure feature calculation failed: {e}")
        
        return features
    @log_all_calls
    
    def _calculate_regime_transition_features(self, data: pd.DataFrame, regimes: np.ndarray) -> pd.DataFrame:
        """Calculate regime transition-specific features."""
        features = pd.DataFrame(index = data.index)
        
        try:
            # Regime persistence
            regime_persistence = np.zeros(len(regimes))
            current_regime = regimes[0]
            current_count = 0
            
            for i, regime in enumerate(regimes):
                if regime == current_regime:
                    current_count += 1
                else:
                    current_count = 1
                    current_regime = regime
                regime_persistence[i] = current_count
            
            features['regime_persistence'] = regime_persistence
            
            # Regime stability (inverse of recent regime changes)
            window = self.feature_config.transition_window
            regime_stability = np.zeros(len(regimes))
            
            for i in range(len(regimes)):
                start_idx = max(0, i - window + 1)
                recent_regimes = regimes[start_idx:i + 1]
                stability = 1 / (1 + np.std(recent_regimes))
                regime_stability[i] = stability
            
            features['regime_stability'] = regime_stability
            
            # Transition probability
            unique_regimes = np.unique(regimes)
            n_regimes = len(unique_regimes)
            
            if n_regimes > 1:
                # Calculate transition matrix
                transition_matrix = np.zeros((n_regimes, n_regimes))
                regime_map = {regime: i for i, regime in enumerate(unique_regimes)}
                
                for i in range(len(regimes) - 1):
                    current_idx = regime_map[regimes[i]]
                    next_idx = regime_map[regimes[i + 1]]
                    transition_matrix[current_idx, next_idx] += 1
                
                # Normalize to probabilities
                row_sums = transition_matrix.sum(axis = 1, keepdims = True)
                transition_matrix = np.divide(transition_matrix, row_sums, where = row_sums > 0)
                
                # Calculate transition probability for each point
                transition_prob = np.zeros(len(regimes))
                for i in range(len(regimes)):
                    current_idx = regime_map[regimes[i]]
                    other_probs = transition_matrix[current_idx, :]
                    other_probs[current_idx] = 0  # Exclude staying in same regime
                    transition_prob[i] = np.sum(other_probs)
                
                features['transition_probability'] = transition_prob
            
            # Regime duration features
            regime_duration = np.zeros(len(regimes))
            current_regime = regimes[0]
            current_duration = 0
            
            for i, regime in enumerate(regimes):
                if regime == current_regime:
                    current_duration += 1
                else:
                    current_duration = 1
                    current_regime = regime
                regime_duration[i] = current_duration
            
            features['regime_duration'] = regime_duration
            features['regime_duration_normalized'] = regime_duration / regime_duration.rolling(100).max()
            
        except Exception as e:
            self.logger.warning(f"Regime transition feature calculation failed: {e}")
        
        return features
    @log_all_calls
    
    def _calculate_volatility_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility regime features."""
        features = pd.DataFrame(index = data.index)
        
        try:
            if 'close' in data.columns:
                returns = data['close'].pct_change()
                
                # Multiple volatility measures
                for window in [5, 10, 20, 50]:
                    features[f'volatility_{window}'] = returns.rolling(window).std()
                    features[f'volatility_ratio_{window}'] = (
                        features[f'volatility_{window}'] / features[f'volatility_{window}'].rolling(100).mean()
                    )
                
                # Volatility clustering
                vol_20 = features['volatility_20']
                features['volatility_clustering'] = vol_20.rolling(50).apply(
                    lambda x: x.autocorr(lag = 1) if len(x) > 1 else 0
                )
                
                # Volatility regime classification
                vol_100 = returns.rolling(100).std()
                low_threshold = vol_100.rolling(100).quantile(0.33)
                high_threshold = vol_100.rolling(100).quantile(0.67)
                
                vol_regime = pd.Series(1, index = data.index)
                vol_regime[vol_100 > high_threshold] = 3
                vol_regime[(vol_100 > low_threshold) & (vol_100 <= high_threshold)] = 2
                features['volatility_regime'] = vol_regime.fillna(1)
                
                # Volatility of volatility
                features['vol_of_vol'] = vol_20.rolling(20).std()
                
                # GARCH-like features
                features['volatility_momentum'] = vol_20.pct_change()
                features['volatility_acceleration'] = features['volatility_momentum'].diff()
                
        except Exception as e:
            self.logger.warning(f"Volatility regime feature calculation failed: {e}")
        
        return features
    @log_all_calls
    
    def _calculate_higher_moment_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate higher moment features."""
        features = pd.DataFrame(index = data.index)
        
        try:
            if 'close' in data.columns:
                returns = data['close'].pct_change()
                
                for window in [10, 20, 50]:
                    # Skewness
                    features[f'skewness_{window}'] = returns.rolling(window).skew()
                    
                    # Kurtosis
                    features[f'kurtosis_{window}'] = returns.rolling(window).kurt()
                    
                    # Jarque-Bera test statistic
                    def jarque_bera_stat(x):
                        if len(x) < 4:
                            return 0
                        try:
                            return stats.jarque_bera(x)[0]
                        except:
                            return 0
                    
                    features[f'jarque_bera_{window}'] = returns.rolling(window).apply(jarque_bera_stat)
                    
                    # Tail risk measures
                    features[f'var_95_{window}'] = returns.rolling(window).quantile(0.05)
                    features[f'var_99_{window}'] = returns.rolling(window).quantile(0.01)
                    features[f'expected_shortfall_{window}'] = (
                        returns.rolling(window).apply(lambda x: x[x <= x.quantile(0.05)].mean())
                    )
                
        except Exception as e:
            self.logger.warning(f"Higher moment feature calculation failed: {e}")
        
        return features
    @log_all_calls
    
    def _calculate_entropy_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate entropy-based features."""
        features = pd.DataFrame(index = data.index)
        
        try:
            if 'close' in data.columns:
                returns = data['close'].pct_change()
                
                for window in [20, 50]:
                    # Shannon entropy of returns
                    def shannon_entropy(x):
                        if len(x) < 2:
                            return 0
                        try:
                            # Discretize returns into bins
                            bins = np.linspace(x.min(), x.max(), 10)
                            hist, _ = np.histogram(x, bins = bins)
                            hist = hist[hist > 0]  # Remove zero counts
                            prob = hist / hist.sum()
                            return -np.sum(prob * np.log2(prob))
                        except:
                            return 0
                    
                    features[f'shannon_entropy_{window}'] = returns.rolling(window).apply(shannon_entropy)
                    
                    # Approximate entropy
                    def approximate_entropy(x, m = 2, r = 0.2):
                        if len(x) < 10:
                            return 0
                        try:
                            N = len(x)
                            r = r * np.std(x)
    @log_all_calls
                            
                            def _maxdist(xi, xj, N):
                                return max([abs(ua - va) for ua, va in zip(xi, xj)])
    @log_all_calls
                            
    @log_all_calls
                            def _approximate_entropy(U, m, r):
                                def _phi(m):
                                    C = np.zeros(N - m + 1)
                                    for i in range(N - m + 1):
                                        template_i = U[i:i + m]
                                        for j in range(N - m + 1):
                                            template_j = U[j:j + m]
                                            if _maxdist(template_i, template_j, m) <= r:
                                                C[i] += 1.0
                                    
                                    phi = np.mean(np.log(C / float(N - m + 1.0)))
                                    return phi
                                
                                return _phi(m) - _phi(m + 1)
                            
                            return _approximate_entropy(x.values, m, r)
                        except:
                            return 0
                    
                    features[f'approximate_entropy_{window}'] = returns.rolling(window).apply(approximate_entropy)
                
        except Exception as e:
            self.logger.warning(f"Entropy feature calculation failed: {e}")
        
        return features
    @log_all_calls
    
    def _calculate_fractal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate fractal dimension features."""
        features = pd.DataFrame(index = data.index)
        
        try:
            if 'close' in data.columns:
                prices = data['close']
                
                for window in [50, 100]:
                    # Hurst exponent
                    def hurst_exponent(x):
                        if len(x) < 10:
                            return 0.5
                        try:
                            lags = range(2, min(20, len(x)//4))
                            tau = [np.sqrt(np.std(np.subtract(x[lag:], x[:-lag]))) for lag in lags]
                            poly = np.polyfit(np.log(lags), np.log(tau), 1)
                            return poly[0] * 2.0
                        except:
                            return 0.5
                    
                    features[f'hurst_exponent_{window}'] = prices.rolling(window).apply(hurst_exponent)
                    
                    # Detrended Fluctuation Analysis (DFA)
                    def dfa(x):
                        if len(x) < 20:
                            return 0
                        try:
                            # Detrend the series
                            y = np.cumsum(x - np.mean(x))
                            
                            # Calculate DFA for different scales
                            scales = np.logspace(0.5, 2, 10).astype(int)
                            scales = scales[scales < len(y)//4]
                            
                            fluctuations = []
                            for scale in scales:
                                # Divide into segments
                                segments = len(y) // scale
                                if segments < 2:
                                    continue
                                
                                # Detrend each segment
                                detrended = []
                                for i in range(segments):
                                    segment = y[i*scale:(i+1)*scale]
                                    x_seg = np.arange(len(segment))
                                    poly = np.polyfit(x_seg, segment, 1)
                                    trend = np.polyval(poly, x_seg)
                                    detrended.extend(segment - trend)
                                
                                # Calculate fluctuation
                                fluctuation = np.sqrt(np.mean(np.array(detrended)**2))
                                fluctuations.append(fluctuation)
                            
                            if len(fluctuations) > 1:
                                poly = np.polyfit(np.log(scales[:len(fluctuations)]), np.log(fluctuations), 1)
                                return poly[0]
                            else:
                                return 0
                        except:
                            return 0
                    
                    features[f'dfa_{window}'] = prices.rolling(window).apply(dfa)
                
        except Exception as e:
            self.logger.warning(f"Fractal feature calculation failed: {e}")
        
        return features
    @log_all_calls
    
    def _generate_polynomial_features(self, features: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """Generate polynomial features."""
        try:
            from sklearn.preprocessing import PolynomialFeatures
            
            # Select numeric columns only
            numeric_features = features.select_dtypes(include=[np.number])
            
            if len(numeric_features.columns) == 0:
                return pd.DataFrame(index = features.index)
            
            # Limit to top features to avoid explosion
            top_features = numeric_features.iloc[:, :min(10, len(numeric_features.columns))]
            
            # Generate polynomial features
            poly = PolynomialFeatures(degree = degree, include_bias = False, interaction_only = True)
            poly_features = poly.fit_transform(top_features.fillna(0))
            
            # Create feature names
            feature_names = [f"poly_{i}" for i in range(poly_features.shape[1])]
            
            return pd.DataFrame(poly_features, index = features.index, columns = feature_names)
            
        except Exception as e:
            self.logger.warning(f"Polynomial feature generation failed: {e}")
            return pd.DataFrame(index = features.index)
    @log_all_calls
    
    def _generate_interaction_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features between key variables."""
        interaction_features = pd.DataFrame(index = features.index)
        
        try:
            # Key feature combinations for regime transitions
            key_features = ['rsi', 'macd', 'volatility_20', 'volume_ratio_10']
            available_features = [f for f in key_features if f in features.columns]
            
            if len(available_features) >= 2:
                for i, feat1 in enumerate(available_features):
                    for feat2 in available_features[i+1:]:
                        if feat1 in features.columns and feat2 in features.columns:
                            interaction_features[f'{feat1}_x_{feat2}'] = (
                                features[feat1] * features[feat2]
                            )
                            interaction_features[f'{feat1}_div_{feat2}'] = (
                                features[feat1] / (features[feat2] + 1e-10)
                            )
            
        except Exception as e:
            self.logger.warning(f"Interaction feature generation failed: {e}")
        
        return interaction_features
    @log_all_calls
    
    def _select_features(self, features: pd.DataFrame, target: np.ndarray) -> pd.DataFrame:
        """Select most relevant features."""
        try:
            # Remove highly correlated features
            numeric_features = features.select_dtypes(include=[np.number])
            corr_matrix = numeric_features.corr().abs()
            
            # Find pairs of highly correlated features
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.feature_config.correlation_threshold)]
            
            features_cleaned = numeric_features.drop(columns = to_drop)
            
            # Feature selection
            if self.feature_config.feature_selection_method == 'mutual_info':
                selector = SelectKBest(score_func = mutual_info_classif, k = min(self.feature_config.max_features, len(features_cleaned.columns)))
            elif self.feature_config.feature_selection_method == 'f_score':
                selector = SelectKBest(score_func = f_classif, k = min(self.feature_config.max_features, len(features_cleaned.columns)))
            else:
                # Variance-based selection
                from sklearn.feature_selection import VarianceThreshold
                selector = VarianceThreshold(threshold = 0.01)
            
            # Fit selector
            features_selected = selector.fit_transform(features_cleaned.fillna(0), target)
            
            # Get selected feature names
            if hasattr(selector, 'get_support'):
                selected_indices = selector.get_support(indices = True)
                selected_columns = features_cleaned.columns[selected_indices]
            else:
                selected_columns = features_cleaned.columns
            
            return pd.DataFrame(features_selected, index = features.index, columns = selected_columns)
            
        except Exception as e:
            self.logger.warning(f"Feature selection failed: {e}")
            return features.select_dtypes(include=[np.number]).iloc[:, :self.feature_config.max_features]
    
    def create_advanced_features(self, data: pd.DataFrame, regimes: np.ndarray) -> pd.DataFrame:
        """Create comprehensive advanced features for regime transition detection."""
        self.logger.info("🚀 Creating advanced features for regime transition detection...")
        
        all_features = pd.DataFrame(index = data.index)
        
        with memory_aware_processing("feature_engineering", self.config.memory.__dict__):
            # Basic technical indicators
            self.logger.info("📊 Creating basic technical indicators...")
            basic_features = self.technical_indicators.calculate_all_indicators(data)
            all_features = pd.concat([all_features, basic_features], axis = 1)
            
            # Signal processing features
            if self.feature_config.enable_fourier_features and 'close' in data.columns:
                self.logger.info("🌊 Creating Fourier features...")
                fourier_features = self._calculate_fourier_features(data['close'])
                all_features = pd.concat([all_features, fourier_features], axis = 1)
            
            if self.feature_config.enable_wavelet_features and 'close' in data.columns:
                self.logger.info("🌊 Creating wavelet features...")
                wavelet_features = self._calculate_wavelet_features(data['close'])
                all_features = pd.concat([all_features, wavelet_features], axis = 1)
            
            if self.feature_config.enable_spectral_features and 'close' in data.columns:
                self.logger.info("📡 Creating spectral features...")
                spectral_features = self._calculate_spectral_features(data['close'])
                all_features = pd.concat([all_features, spectral_features], axis = 1)
            
            # Domain-specific features
            if self.feature_config.enable_market_microstructure:
                self.logger.info("🏪 Creating market microstructure features...")
                microstructure_features = self._calculate_market_microstructure_features(data)
                all_features = pd.concat([all_features, microstructure_features], axis = 1)
            
            if self.feature_config.enable_regime_transition_features:
                self.logger.info("🔄 Creating regime transition features...")
                transition_features = self._calculate_regime_transition_features(data, regimes)
                all_features = pd.concat([all_features, transition_features], axis = 1)
            
            if self.feature_config.enable_volatility_regime_features:
                self.logger.info("📈 Creating volatility regime features...")
                volatility_features = self._calculate_volatility_regime_features(data)
                all_features = pd.concat([all_features, volatility_features], axis = 1)
            
            # Advanced statistical features
            if self.feature_config.enable_higher_moments:
                self.logger.info("📊 Creating higher moment features...")
                moment_features = self._calculate_higher_moment_features(data)
                all_features = pd.concat([all_features, moment_features], axis = 1)
            
            if self.feature_config.enable_entropy_features:
                self.logger.info("🔀 Creating entropy features...")
                entropy_features = self._calculate_entropy_features(data)
                all_features = pd.concat([all_features, entropy_features], axis = 1)
            
            if self.feature_config.enable_fractal_features:
                self.logger.info("🌀 Creating fractal features...")
                fractal_features = self._calculate_fractal_features(data)
                all_features = pd.concat([all_features, fractal_features], axis = 1)
            
            # Automated feature generation
            if self.feature_config.enable_interaction_features:
                self.logger.info("🔗 Creating interaction features...")
                interaction_features = self._generate_interaction_features(all_features)
                all_features = pd.concat([all_features, interaction_features], axis = 1)
            
            if self.feature_config.enable_polynomial_features:
                self.logger.info("📐 Creating polynomial features...")
                poly_features = self._generate_polynomial_features(all_features)
                all_features = pd.concat([all_features, poly_features], axis = 1)
        
        # Clean and select features
        self.logger.info("🧹 Cleaning and selecting features...")
        all_features = all_features.fillna(method='forward').fillna(0)
        
        # Create target variable for feature selection
        target = self._create_transition_target(regimes)
        
        # Select most relevant features
        selected_features = self._select_features(all_features, target)
        
        self.logger.info(f"✅ Advanced feature engineering completed")
        self.logger.info(f"   Created {len(all_features.columns)} total features")
        self.logger.info(f"   Selected {len(selected_features.columns)} features")
        
        return selected_features
    @log_all_calls
    
    def _create_transition_target(self, regimes: np.ndarray) -> np.ndarray:
        """Create target variable for regime transitions."""
        target = np.zeros(len(regimes))
        prediction_horizon = self.feature_config.prediction_horizon
        
        for i in range(len(regimes) - prediction_horizon):
            current_regime = regimes[i]
            future_regimes = regimes[i + 1:i + prediction_horizon + 1]
            if np.any(future_regimes != current_regime):
                target[i] = 1
        
        return target

"""
Advanced Feature Engineering for ML Transition Detection.

This module provides sophisticated feature engineering alternatives using
domain-specific knowledge, advanced signal processing, and automated feature
generation techniques.
"""