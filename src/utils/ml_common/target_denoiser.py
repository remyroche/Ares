"""
Target Denoising for Specialist Orthogonalization

Fast and effective target denoising methods optimized for binary labels
in financial machine learning applications.

Methods:
- Kalman Filter: Linear-time smoothing
- Hampel Filter: Outlier removal
- Savitzky-Golay: Trend smoothing
- Volume-Weighted: Domain-aware conviction-based denoising
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from scipy import signal
from scipy.ndimage import median_filter
import logging
from functools import lru_cache
import hashlib

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_success

logger = system_logger.getChild('TargetDenoiser')


@dataclass
class DenoisingConfig:
    """Configuration for target denoising"""
    method: str = 'kalman'
    kalman_process_noise: float = 1e-4
    kalman_measurement_noise: float = 0.01
    hampel_window: int = 5
    hampel_threshold: float = 3.0
    savgol_window: int = 7
    savgol_polyorder: int = 2
    volume_window: int = 20
    volume_threshold: float = 0.3
    confidence_threshold: float = 0.7
    enable_caching: bool = True
    parallel_workers: int = 2


@dataclass
class DenoisingResult:
    """Result of target denoising"""
    denoised_target: pd.Series
    confidence_scores: pd.Series
    noise_analysis: Dict[str, Any]
    denoising_stats: Dict[str, float]
    method_used: str
    processing_time: float


class TargetDenoiser:
    """Fast and effective target denoising for orthogonalization"""
    
    def __init__(self, config: Optional[DenoisingConfig] = None):
        self.config = config or DenoisingConfig()
        self._cache = {} if self.config.enable_caching else None
        
        # Initialize denoising methods
        self.denoising_methods = {
            'kalman': self._kalman_denoise,
            'hampel': self._hampel_denoise,
            'savgol': self._savgol_denoise,
            'volume': self._volume_weighted_denoise,
            'ensemble': self._ensemble_denoise
        }
        
        tprint_info(f"🔇 Target denoiser initialized with method: {self.config.method}")
    
    def denoise_target(self, target_series: pd.Series, 
                      features: Optional[pd.DataFrame] = None,
                      volume_series: Optional[pd.Series] = None) -> DenoisingResult:
        """Denoise target using configured method"""
        
        import time
        start_time = time.time()
        
        # Validate input
        if not isinstance(target_series, pd.Series):
            raise ValueError("target_series must be a pandas Series")
        
        if len(target_series) < 10:
            tprint_warning("Target series too short for denoising")
            return DenoisingResult(
                denoised_target=target_series.copy(),
                confidence_scores=pd.Series(1.0, index=target_series.index),
                noise_analysis={'error': 'Series too short'},
                denoising_stats={'no_processing': True},
                method_used='none',
                processing_time=0.0
            )
        
        # Check cache first
        if self._cache is not None:
            cache_key = self._get_cache_key(target_series, self.config.method)
            if cache_key in self._cache:
                cached_result = self._cache[cache_key]
                tprint_info(f"📂 Using cached denoised target")
                return cached_result
        
        # Analyze noise characteristics
        noise_analysis = self._analyze_target_noise(target_series, features, volume_series)
        
        # Apply denoising
        if self.config.method in self.denoising_methods:
            denoised_target = self.denoising_methods[self.config.method](
                target_series, noise_analysis, volume_series
            )
        else:
            raise ValueError(f"Unknown denoising method: {self.config.method}")
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(
            target_series, denoised_target, noise_analysis
        )
        
        # Apply confidence-based filtering
        final_target = self._apply_confidence_filter(
            denoised_target, confidence_scores
        )
        
        # Calculate denoising statistics
        denoising_stats = self._calculate_denoising_stats(target_series, final_target)
        
        processing_time = time.time() - start_time
        
        result = DenoisingResult(
            denoised_target=final_target,
            confidence_scores=confidence_scores,
            noise_analysis=noise_analysis,
            denoising_stats=denoising_stats,
            method_used=self.config.method,
            processing_time=processing_time
        )
        
        # Cache result
        if self._cache is not None:
            self._cache[cache_key] = result
        
        tprint_info(f"🔇 Denoised target in {processing_time:.3f}s using {self.config.method}")
        tprint_info(f"   Noise reduction: {denoising_stats.get('noise_reduction', 0):.1%}")
        tprint_info(f"   Confidence: {denoising_stats.get('mean_confidence', 0):.3f}")
        
        return result
    
    def _kalman_denoise(self, target_series: pd.Series, 
                       noise_analysis: Dict[str, Any],
                       volume_series: Optional[pd.Series] = None) -> pd.Series:
        """Kalman filter denoising - O(n) time complexity"""
        
        # Convert to continuous for filtering
        continuous_target = target_series.astype(float).values
        
        # Kalman parameters
        Q = self.config.kalman_process_noise
        R = self.config.kalman_measurement_noise
        
        # Initialize Kalman filter
        n = len(continuous_target)
        x_hat = np.zeros(n)
        P = np.ones(n)
        
        # Initial state
        x_hat[0] = continuous_target[0]
        P[0] = 1.0
        
        # Kalman filter loop
        for i in range(1, n):
            # Prediction
            x_hat_minus = x_hat[i-1]
            P_minus = P[i-1] + Q
            
            # Update
            K = P_minus / (P_minus + R)
            x_hat[i] = x_hat_minus + K * (continuous_target[i] - x_hat_minus)
            P[i] = (1 - K) * P_minus
        
        # Convert back to binary with threshold
        smoothed_prob = 1 / (1 + np.exp(-x_hat))  # Sigmoid for probability
        denoised = (smoothed_prob > 0.5).astype(int)
        
        return pd.Series(denoised, index=target_series.index)
    
    def _hampel_denoise(self, target_series: pd.Series,
                       noise_analysis: Dict[str, Any],
                       volume_series: Optional[pd.Series] = None) -> pd.Series:
        """Hampel filter denoising - O(n×w) time complexity"""
        
        window = self.config.hampel_window
        threshold = self.config.hampel_threshold
        
        # Ensure odd window
        if window % 2 == 0:
            window += 1
        
        half_window = window // 2
        target_values = target_series.values
        filtered_values = target_values.copy()
        
        # Apply Hampel filter
        for i in range(half_window, len(target_values) - half_window):
            window_data = target_values[i - half_window:i + half_window + 1]
            median = np.median(window_data)
            
            # Median Absolute Deviation
            mad = np.median(np.abs(window_data - median))
            
            if mad > 0:
                # Check if point is outlier
                if abs(target_values[i] - median) > threshold * mad:
                    filtered_values[i] = median
        
        return pd.Series(filtered_values, index=target_series.index)
    
    def _savgol_denoise(self, target_series: pd.Series,
                       noise_analysis: Dict[str, Any],
                       volume_series: Optional[pd.Series] = None) -> pd.Series:
        """Savitzky-Golay filter denoising - O(n×w) time complexity"""
        
        window = self.config.savgol_window
        polyorder = self.config.savgol_polyorder
        
        # Ensure valid parameters
        if window % 2 == 0:
            window += 1
        
        if polyorder >= window:
            polyorder = window - 1
        
        try:
            # Convert to continuous for filtering
            continuous_target = target_series.astype(float).values
            
            # Apply Savitzky-Golay filter
            smoothed = signal.savgol_filter(continuous_target, window, polyorder)
            
            # Convert back to binary
            denoised = (smoothed > 0.5).astype(int)
            
            return pd.Series(denoised, index=target_series.index)
            
        except Exception as e:
            tprint_warning(f"Savitzky-Golay filter failed: {e}, returning original")
            return target_series.copy()
    
    def _volume_weighted_denoise(self, target_series: pd.Series,
                                noise_analysis: Dict[str, Any],
                                volume_series: Optional[pd.Series] = None) -> pd.Series:
        """Volume-weighted denoising - low volume = low conviction"""
        
        if volume_series is None:
            tprint_warning("Volume series not provided, using Hampel filter instead")
            return self._hampel_denoise(target_series, noise_analysis, volume_series)
        
        # Align series
        aligned_target, aligned_volume = target_series.align(volume_series, join='inner')
        
        if len(aligned_target) < 10:
            return target_series.copy()
        
        # Calculate volume percentiles
        window = self.config.volume_window
        volume_threshold = self.config.volume_threshold
        
        # Rolling volume percentile
        volume_percentile = aligned_volume.rolling(window, min_periods=1).rank(pct=True)
        
        # Low conviction mask (low volume periods)
        low_conviction_mask = volume_percentile < volume_threshold
        
        # Apply Hampel filter to low conviction periods
        denoised_values = aligned_target.values.copy()
        
        if low_conviction_mask.any():
            # Apply Hampel filter only to low conviction periods
            hampel_window = min(self.config.hampel_window, len(denoised_values) // 4)
            if hampel_window % 2 == 0:
                hampel_window += 1
            
            half_window = hampel_window // 2
            
            for i in range(len(denoised_values)):
                if low_conviction_mask.iloc[i] and half_window <= i < len(denoised_values) - half_window:
                    window_data = denoised_values[i - half_window:i + half_window + 1]
                    median = np.median(window_data)
                    mad = np.median(np.abs(window_data - median))
                    
                    if mad > 0:
                        threshold = self.config.hampel_threshold
                        if abs(denoised_values[i] - median) > threshold * mad:
                            denoised_values[i] = median
        
        denoised = pd.Series(denoised_values, index=aligned_target.index)
        
        # Reindex to match original
        return denoised.reindex(target_series.index, fill_value=target_series)
    
    def _ensemble_denoise(self, target_series: pd.Series,
                         noise_analysis: Dict[str, Any],
                         volume_series: Optional[pd.Series] = None) -> pd.Series:
        """Ensemble denoising - combines multiple methods"""
        
        methods = ['kalman', 'hampel', 'savgol']
        if volume_series is not None:
            methods.append('volume')
        
        denoised_results = {}
        for method in methods:
            try:
                denoised_results[method] = self.denoising_methods[method](
                    target_series, noise_analysis, volume_series
                )
            except Exception as e:
                tprint_warning(f"Method {method} failed in ensemble: {e}")
                continue
        
        if not denoised_results:
            return target_series.copy()
        
        # Weighted ensemble based on noise characteristics
        weights = self._calculate_ensemble_weights(noise_analysis)
        
        # Combine results
        ensemble_values = np.zeros(len(target_series))
        total_weight = 0
        
        for method, result in denoised_results.items():
            weight = weights.get(method, 1.0)
            ensemble_values += weight * result.values
            total_weight += weight
        
        if total_weight > 0:
            ensemble_values /= total_weight
        
        # Convert to binary
        ensemble_binary = (ensemble_values > 0.5).astype(int)
        
        return pd.Series(ensemble_binary, index=target_series.index)
    
    def _analyze_target_noise(self, target_series: pd.Series,
                             features: Optional[pd.DataFrame] = None,
                             volume_series: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Analyze noise characteristics in target"""
        
        try:
            # Basic statistics
            values = target_series.values
            n_samples = len(values)
            
            # Label transitions (measure of temporal consistency)
            transitions = np.sum(np.abs(np.diff(values)))
            transition_rate = transitions / (n_samples - 1) if n_samples > 1 else 0
            
            # Run analysis (measure of randomness)
            runs = self._count_runs(values)
            expected_runs = (2 * np.sum(values == 1) * np.sum(values == 0) / n_samples) + 1
            runs_statistic = (runs - expected_runs) / np.sqrt(expected_runs) if expected_runs > 0 else 0
            
            # Noise level estimate
            noise_level = min(transition_rate, abs(runs_statistic) / n_samples)
            
            # Feature-target correlation if features available
            feature_correlation = 0.0
            if features is not None and not features.empty:
                correlations = []
                for col in features.select_dtypes(include=[np.number]).columns:
                    try:
                        corr = np.corrcoef(features[col].values, values)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
                    except:
                        continue
                feature_correlation = np.mean(correlations) if correlations else 0.0
            
            # Volume analysis if available
            volume_volatility = 0.0
            if volume_series is not None:
                aligned_volume = volume_series.reindex(target_series.index, fill_value=0)
                volume_volatility = aligned_volume.rolling(20).std().mean()
            
            return {
                'n_samples': n_samples,
                'transition_rate': transition_rate,
                'runs_statistic': runs_statistic,
                'noise_level': noise_level,
                'feature_correlation': feature_correlation,
                'volume_volatility': volume_volatility,
                'recommended_method': self._recommend_method(noise_level, feature_correlation)
            }
            
        except Exception as e:
            logger.warning(f"Noise analysis failed: {e}")
            return {'error': str(e), 'noise_level': 0.5}
    
    def _count_runs(self, values: np.ndarray) -> int:
        """Count runs in binary sequence"""
        runs = 1
        for i in range(1, len(values)):
            if values[i] != values[i-1]:
                runs += 1
        return runs
    
    def _recommend_method(self, noise_level: float, feature_correlation: float) -> str:
        """Recommend denoising method based on characteristics"""
        
        if noise_level < 0.1:
            return 'none'  # Already clean
        elif noise_level < 0.3:
            return 'kalman'  # Light smoothing
        elif feature_correlation < 0.1:
            return 'hampel'  # Likely outliers
        else:
            return 'ensemble'  # Complex noise pattern
    
    def _calculate_ensemble_weights(self, noise_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate weights for ensemble methods"""
        
        noise_level = noise_analysis.get('noise_level', 0.5)
        feature_correlation = noise_analysis.get('feature_correlation', 0.0)
        
        # Base weights
        weights = {
            'kalman': 0.3,
            'hampel': 0.3,
            'savgol': 0.2,
            'volume': 0.2
        }
        
        # Adjust based on noise characteristics
        if noise_level > 0.4:
            weights['hampel'] += 0.1  # More outlier removal
            weights['kalman'] -= 0.1
        
        if feature_correlation > 0.2:
            weights['savgol'] += 0.1  # Better for trends
            weights['volume'] -= 0.1
        
        return weights
    
    def _calculate_confidence_scores(self, original_target: pd.Series,
                                   denoised_target: pd.Series,
                                   noise_analysis: Dict[str, Any]) -> pd.Series:
        """Calculate confidence scores for denoised targets"""
        
        # Base confidence from noise analysis
        base_confidence = 1.0 - noise_analysis.get('noise_level', 0.5)
        
        # Agreement confidence
        agreement = (original_target == denoised_target).astype(float)
        
        # Combine confidences
        confidence = base_confidence * 0.7 + agreement * 0.3
        
        return pd.Series(confidence, index=original_target.index)
    
    def _apply_confidence_filter(self, denoised_target: pd.Series,
                               confidence_scores: pd.Series) -> pd.Series:
        """Apply confidence-based filtering"""
        
        threshold = self.config.confidence_threshold
        
        # Keep original values where confidence is low
        filtered_target = denoised_target.copy()
        low_confidence_mask = confidence_scores < threshold
        
        if low_confidence_mask.any():
            # For low confidence points, use more conservative approach
            filtered_target[low_confidence_mask] = denoised_target[low_confidence_mask]
        
        return filtered_target
    
    def _calculate_denoising_stats(self, original_target: pd.Series,
                                 denoised_target: pd.Series) -> Dict[str, float]:
        """Calculate denoising statistics"""
        
        # Noise reduction
        original_transitions = np.sum(np.abs(np.diff(original_target.values)))
        denoised_transitions = np.sum(np.abs(np.diff(denoised_target.values)))
        
        noise_reduction = 0.0
        if original_transitions > 0:
            noise_reduction = (original_transitions - denoised_transitions) / original_transitions
        
        # Agreement rate
        agreement_rate = (original_target == denoised_target).mean()
        
        # Mean confidence
        confidence_scores = self._calculate_confidence_scores(
            original_target, denoised_target, {'noise_level': 0.5}
        )
        mean_confidence = confidence_scores.mean()
        
        return {
            'noise_reduction': noise_reduction,
            'agreement_rate': agreement_rate,
            'mean_confidence': mean_confidence,
            'original_transitions': original_transitions,
            'denoised_transitions': denoised_transitions
        }
    
    def _get_cache_key(self, target_series: pd.Series, method: str) -> str:
        """Generate cache key for target series"""
        # Use hash of series data and method
        data_hash = hashlib.md5(str(target_series.values.tobytes()).encode()).hexdigest()
        return f"{method}_{data_hash}"
    
    def get_available_methods(self) -> List[str]:
        """Get list of available denoising methods"""
        return list(self.denoising_methods.keys())
    
    def clear_cache(self):
        """Clear denoising cache"""
        if self._cache is not None:
            self._cache.clear()
            tprint_info("🗑️ Cleared denoising cache")


def create_target_denoiser(method: str = 'kalman', **kwargs) -> TargetDenoiser:
    """Factory function to create target denoiser"""
    config = DenoisingConfig(method=method, **kwargs)
    return TargetDenoiser(config)


# Convenience functions for specific methods
def kalman_denoise(target_series: pd.Series, 
                   process_noise: float = 1e-4,
                   measurement_noise: float = 0.01) -> pd.Series:
    """Quick Kalman denoising"""
    denoiser = create_target_denoiser('kalman', 
                                     kalman_process_noise=process_noise,
                                     kalman_measurement_noise=measurement_noise)
    result = denoiser.denoise_target(target_series)
    return result.denoised_target


def hampel_denoise(target_series: pd.Series,
                   window: int = 5,
                   threshold: float = 3.0) -> pd.Series:
    """Quick Hampel denoising"""
    denoiser = create_target_denoiser('hampel',
                                     hampel_window=window,
                                     hampel_threshold=threshold)
    result = denoiser.denoise_target(target_series)
    return result.denoised_target


def savgol_denoise(target_series: pd.Series,
                   window: int = 7,
                   polyorder: int = 2) -> pd.Series:
    """Quick Savitzky-Golay denoising"""
    denoiser = create_target_denoiser('savgol',
                                     savgol_window=window,
                                     savgol_polyorder=polyorder)
    result = denoiser.denoise_target(target_series)
    return result.denoised_target


def volume_weighted_denoise(target_series: pd.Series,
                           volume_series: pd.Series,
                           window: int = 20,
                           threshold: float = 0.3) -> pd.Series:
    """Quick volume-weighted denoising"""
    denoiser = create_target_denoiser('volume',
                                     volume_window=window,
                                     volume_threshold=threshold)
    result = denoiser.denoise_target(target_series, volume_series=volume_series)
    return result.denoised_target
