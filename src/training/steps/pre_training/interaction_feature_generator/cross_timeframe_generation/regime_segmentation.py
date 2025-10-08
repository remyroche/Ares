"""
Regime Segmentation and Change-Point Detection

Implements regime segmentation before scoring with:
- Change-point detection (PELT/CUSUM) on volatility proxy
- Vol regime classification (low/high) via EW quantiles
- BOCPD (Bayesian Online Change-Point Detection) for real-time adaptation
- Regime-aware feature scoring and optimization
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from .config import RegimeConfig

# Import tprint for enhanced runtime diagnostics
try:
    from src.utils.tprint import (
        tprint,
        tprint_debug,
        tprint_info,
        tprint_warning,
        tprint_error,
        tprint_success,
    )
    TPRINT_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback for limited environments
    TPRINT_AVAILABLE = False

    def tprint(*args, **kwargs):
        print(*args, **kwargs)

    def tprint_debug(*args, **kwargs):
        print("DEBUG:", *args, **kwargs)

    def tprint_info(*args, **kwargs):
        print("INFO:", *args, **kwargs)

    def tprint_warning(*args, **kwargs):
        print("WARNING:", *args, **kwargs)

    def tprint_error(*args, **kwargs):
        print("ERROR:", *args, **kwargs)

    def tprint_success(*args, **kwargs):
        print("SUCCESS:", *args, **kwargs)


# Try to import ruptures for change-point detection
try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    logging.warning("ruptures not available, using simplified change-point detection")


@dataclass
class RegimeSegment:
    """Represents a regime segment."""
    start_idx: int
    end_idx: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    regime_type: str  # 'low_vol', 'high_vol', 'transition'
    volatility_level: float
    mean_return: float
    volatility_proxy: float
    metadata: Dict[str, Any]


@dataclass
class ChangePoint:
    """Represents a detected change point."""
    idx: int
    timestamp: pd.Timestamp
    confidence: float
    method: str
    metadata: Dict[str, Any]


class ChangePointDetector:
    """Change-point detection using PELT or CUSUM."""
    
    def __init__(self, method: str = 'PELT', penalty: float = 10.0):
        self.method = method
        self.penalty = penalty
        self.logger = logging.getLogger(__name__)
    
    def detect_change_points(self,
                           data: pd.Series,
                           min_segment_length: int = 50) -> List[ChangePoint]:
        """
        Detect change points in the data.
        
        Args:
            data: Time series data
            min_segment_length: Minimum length of segments
            
        Returns:
            List of detected change points
        """
        tprint_debug(
            "Running change-point detection",
            {
                'method': self.method,
                'data_points': len(data),
                'min_segment_length': min_segment_length,
            },
        )
        if len(data) < min_segment_length * 2:
            tprint_warning(
                "Insufficient data for change-point detection",
                {
                    'data_points': len(data),
                    'required': min_segment_length * 2,
                },
            )
            return []

        if self.method == 'PELT' and RUPTURES_AVAILABLE:
            result = self._detect_pelt(data, min_segment_length)
        elif self.method == 'CUSUM':
            result = self._detect_cusum(data, min_segment_length)
        else:
            # Fallback to simple variance-based detection
            result = self._detect_variance_based(data, min_segment_length)

        tprint_info(
            "Change-point detection complete",
            {
                'method': self.method,
                'change_points_detected': len(result),
            },
        )
        return result

    def _detect_pelt(self, data: pd.Series, min_segment_length: int) -> List[ChangePoint]:
        """Detect change points using PELT algorithm."""
        try:
            # Convert to numpy array
            values = data.dropna().values
            
            if len(values) < min_segment_length * 2:
                return []
            
            # Use PELT with normal cost function
            model = rpt.Pelt(model="rbf").fit(values.reshape(-1, 1))
            change_points = model.predict(pen=self.penalty)
            
            # Convert to ChangePoint objects
            change_points = [cp for cp in change_points if cp < len(values)]
            change_points = [cp for cp in change_points if cp >= min_segment_length]
            change_points = [cp for cp in change_points if len(values) - cp >= min_segment_length]
            
            result = []
            for cp in change_points:
                timestamp = data.dropna().index[cp]
                result.append(ChangePoint(
                    idx=cp,
                    timestamp=timestamp,
                    confidence=0.8,  # PELT doesn't provide confidence
                    method='PELT',
                    metadata={'penalty': self.penalty}
                ))
            
            return result

        except Exception as e:
            self.logger.warning(f"PELT detection failed: {e}, falling back to variance-based")
            tprint_error(
                "PELT detection failed, falling back to variance-based method",
                {'error': str(e)},
            )
            return self._detect_variance_based(data, min_segment_length)

    def _detect_cusum(self, data: pd.Series, min_segment_length: int) -> List[ChangePoint]:
        """Detect change points using CUSUM algorithm."""
        values = data.dropna().values

        if len(values) < min_segment_length * 2:
            tprint_warning(
                "CUSUM detection skipped due to insufficient data",
                {
                    'data_points': len(values),
                    'required': min_segment_length * 2,
                },
            )
            return []

        # Calculate CUSUM statistics
        mean_val = np.mean(values)
        cusum = np.cumsum(values - mean_val)
        
        # Find peaks in CUSUM (potential change points)
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(np.abs(cusum), distance=min_segment_length)
        
        result = []
        for peak in peaks:
            if peak >= min_segment_length and len(values) - peak >= min_segment_length:
                timestamp = data.dropna().index[peak]
                confidence = min(0.9, np.abs(cusum[peak]) / np.std(cusum))
                
                result.append(ChangePoint(
                    idx=peak,
                    timestamp=timestamp,
                    confidence=confidence,
                    method='CUSUM',
                    metadata={'cusum_value': cusum[peak]}
                ))

        tprint_debug(
            "CUSUM detection results",
            {
                'peaks_found': len(peaks),
                'change_points_kept': len(result),
            },
        )
        return result

    def _detect_variance_based(self, data: pd.Series, min_segment_length: int) -> List[ChangePoint]:
        """Fallback variance-based change point detection."""
        values = data.dropna().values

        if len(values) < min_segment_length * 2:
            tprint_warning(
                "Variance-based detection skipped due to insufficient data",
                {
                    'data_points': len(values),
                    'required': min_segment_length * 2,
                },
            )
            return []

        # Calculate rolling variance
        window_size = min_segment_length
        rolling_var = pd.Series(values).rolling(window_size).var()
        
        # Find significant changes in variance
        var_changes = rolling_var.diff().abs()
        threshold = var_changes.quantile(0.9)  # Top 10% of changes
        
        change_points = []
        for i in range(window_size, len(values) - window_size):
            if (var_changes.iloc[i] > threshold and 
                i >= min_segment_length and 
                len(values) - i >= min_segment_length):
                
                timestamp = data.dropna().index[i]
                confidence = min(0.7, var_changes.iloc[i] / threshold)
                
                change_points.append(ChangePoint(
                    idx=i,
                    timestamp=timestamp,
                    confidence=confidence,
                    method='variance_based',
                    metadata={'variance_change': var_changes.iloc[i]}
                ))

        tprint_debug(
            "Variance-based detection results",
            {
                'threshold': threshold,
                'change_points_detected': len(change_points),
            },
        )
        return change_points


class BOCPD:
    """Bayesian Online Change-Point Detection for real-time adaptation."""
    
    def __init__(self, hazard: float = 1/200, alpha: float = 1.0, beta: float = 1.0):
        self.hazard = hazard
        self.alpha = alpha
        self.beta = beta
        self.logger = logging.getLogger(__name__)
        
        # State variables
        self.run_length = 0
        self.alpha_t = alpha
        self.beta_t = beta
        self.mu_t = 0.0
        self.kappa_t = 0.0
        self.nu_t = 0.0
        self.phi_t = 0.0
        
    def update(self, observation: float) -> Dict[str, Any]:
        """
        Update BOCPD with new observation.
        
        Args:
            observation: New data point
            
        Returns:
            Dictionary with change point probability and other statistics
        """
        tprint_debug(
            "Updating BOCPD",
            {
                'observation': observation,
                'current_run_length': self.run_length,
            },
        )
        # Calculate change point probability
        cp_prob = self.hazard / (self.hazard + self.run_length)
        
        # Update sufficient statistics
        self.run_length += 1
        self.kappa_t += 1
        self.nu_t += 1
        self.phi_t += observation
        
        # Update parameters
        self.mu_t = self.phi_t / self.kappa_t
        self.alpha_t = self.alpha + self.nu_t / 2
        self.beta_t = self.beta + (self.phi_t**2) / (2 * self.kappa_t)
        
        # Check for change point
        change_detected = cp_prob > 0.5  # Threshold for change detection

        if change_detected:
            # Reset state
            self.run_length = 0
            self.alpha_t = self.alpha
            self.beta_t = self.beta
            self.mu_t = observation
            self.kappa_t = 1
            self.nu_t = 1
            self.phi_t = observation
            tprint_info(
                "BOCPD change detected; state reset",
                {'observation': observation, 'cp_probability': cp_prob},
            )

        return {
            'change_point_probability': cp_prob,
            'change_detected': change_detected,
            'run_length': self.run_length,
            'mean': self.mu_t,
            'variance': self.beta_t / (self.alpha_t - 1) if self.alpha_t > 1 else 1.0
        }
    
    def get_regime_posterior(self) -> Dict[str, float]:
        """Get current regime posterior probabilities."""
        return {
            'low_vol': 1 - self.hazard,
            'high_vol': self.hazard,
            'transition': 0.0  # Simplified
        }


class RegimeClassifier:
    """Classify regimes based on volatility levels."""
    
    def __init__(self, vol_quantile: float = 0.6):
        self.vol_quantile = vol_quantile
        self.logger = logging.getLogger(__name__)
    
    def classify_regimes(self, 
                        segments: List[RegimeSegment],
                        volatility_proxy: pd.Series) -> List[RegimeSegment]:
        """
        Classify segments into volatility regimes.
        
        Args:
            segments: List of regime segments
            volatility_proxy: Volatility proxy series
            
        Returns:
            Updated segments with regime classifications
        """
        if not segments:
            tprint_warning("No segments provided for regime classification")
            return segments
        
        # Calculate volatility levels for each segment
        segment_vols = []
        for segment in segments:
            start_idx = segment.start_idx
            end_idx = segment.end_idx
            
            if start_idx < len(volatility_proxy) and end_idx <= len(volatility_proxy):
                segment_vol = volatility_proxy.iloc[start_idx:end_idx].mean()
                segment_vols.append(segment_vol)
            else:
                segment_vols.append(0.0)
        
        # Determine threshold (Q60 of segment volatilities)
        if segment_vols:
            threshold = np.percentile(segment_vols, self.vol_quantile * 100)
        else:
            threshold = 0.0
        
        # Classify segments
        updated_segments = []
        for i, segment in enumerate(segments):
            vol_level = segment_vols[i] if i < len(segment_vols) else 0.0
            
            if vol_level >= threshold:
                regime_type = 'high_vol'
            else:
                regime_type = 'low_vol'
            
            updated_segment = RegimeSegment(
                start_idx=segment.start_idx,
                end_idx=segment.end_idx,
                start_time=segment.start_time,
                end_time=segment.end_time,
                regime_type=regime_type,
                volatility_level=vol_level,
                mean_return=segment.mean_return,
                volatility_proxy=vol_level,
                metadata=segment.metadata
            )
            updated_segments.append(updated_segment)

        tprint_info(
            "Regime classification complete",
            {
                'segments_processed': len(segments),
                'threshold': threshold,
            },
        )
        return updated_segments


class RegimeSegmentation:
    """Main regime segmentation system."""

    def __init__(self, config: RegimeConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.change_point_detector = ChangePointDetector(
            method=config.change_point_method,
            penalty=10.0  # BIC penalty
        )
        self.regime_classifier = RegimeClassifier(
            vol_quantile=config.regime_vol_quantile
        )
        self.bocpd = BOCPD(hazard=config.bocpd_hazard)
    
    def segment_regimes(self, 
                       sessionized_data: Dict[str, Any],
                       targets: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Perform regime segmentation on the data.
        
        Args:
            sessionized_data: Sessionized and aligned data
            targets: Target variables (optional)
            
        Returns:
            Dictionary containing regime segmentation results
        """
        self.logger.info("Starting regime segmentation")
        tprint_info("Starting regime segmentation", {'sessions': len(sessionized_data)})

        aligned_data = sessionized_data['aligned_data']
        
        # Create volatility proxy (r1 variance or VIX proxy)
        volatility_proxy = self._create_volatility_proxy(aligned_data)
        
        # Detect change points
        change_points = self.change_point_detector.detect_change_points(volatility_proxy)

        # Create segments from change points
        segments = self._create_segments_from_change_points(
            change_points, volatility_proxy, aligned_data
        )
        
        # Classify regimes
        classified_segments = self.regime_classifier.classify_regimes(
            segments, volatility_proxy
        )
        
        # Calculate segment statistics
        segment_stats = self._calculate_segment_statistics(
            classified_segments, aligned_data, targets
        )
        
        # Setup BOCPD for real-time monitoring
        bocpd_state = self._initialize_bocpd(volatility_proxy)
        
        results = {
            'segments': classified_segments,
            'change_points': change_points,
            'volatility_proxy': volatility_proxy,
            'segment_statistics': segment_stats,
            'bocpd_state': bocpd_state,
            'regime_transitions': self._identify_regime_transitions(classified_segments)
        }

        self.logger.info(f"Regime segmentation completed: {len(classified_segments)} segments")
        tprint_success(
            "Regime segmentation completed",
            {
                'segments': len(classified_segments),
                'change_points': len(change_points),
            },
        )
        return results
    
    def _create_volatility_proxy(self, data: pd.DataFrame) -> pd.Series:
        """Create volatility proxy from OHLCV data."""
        # Use r1 variance as volatility proxy
        r1 = np.log(data['close'] / data['close'].shift(1))
        r1_var = r1.rolling(20).var()
        
        # Fill NaN values with forward fill
        r1_var = r1_var.ffill().fillna(0)
        
        return r1_var
    
    def _create_segments_from_change_points(self, 
                                          change_points: List[ChangePoint],
                                          volatility_proxy: pd.Series,
                                          data: pd.DataFrame) -> List[RegimeSegment]:
        """Create regime segments from change points."""
        segments = []
        
        if not change_points:
            # Single segment covering all data
            segment = RegimeSegment(
                start_idx=0,
                end_idx=len(data),
                start_time=data.index[0],
                end_time=data.index[-1],
                regime_type='unknown',
                volatility_level=volatility_proxy.mean(),
                mean_return=0.0,
                volatility_proxy=volatility_proxy.mean(),
                metadata={}
            )
            segments.append(segment)
            return segments
        
        # Create segments between change points
        start_idx = 0
        for cp in change_points:
            if cp.idx > start_idx:
                segment = RegimeSegment(
                    start_idx=start_idx,
                    end_idx=cp.idx,
                    start_time=data.index[start_idx],
                    end_time=data.index[cp.idx - 1],
                    regime_type='unknown',
                    volatility_level=volatility_proxy.iloc[start_idx:cp.idx].mean(),
                    mean_return=0.0,  # Will be calculated later
                    volatility_proxy=volatility_proxy.iloc[start_idx:cp.idx].mean(),
                    metadata={'change_point': cp}
                )
                segments.append(segment)
                start_idx = cp.idx
        
        # Add final segment
        if start_idx < len(data):
            segment = RegimeSegment(
                start_idx=start_idx,
                end_idx=len(data),
                start_time=data.index[start_idx],
                end_time=data.index[-1],
                regime_type='unknown',
                volatility_level=volatility_proxy.iloc[start_idx:].mean(),
                mean_return=0.0,
                volatility_proxy=volatility_proxy.iloc[start_idx:].mean(),
                metadata={}
            )
            segments.append(segment)
        
        return segments
    
    def _calculate_segment_statistics(self, 
                                    segments: List[RegimeSegment],
                                    data: pd.DataFrame,
                                    targets: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Calculate statistics for each segment."""
        stats = {}
        
        for i, segment in enumerate(segments):
            start_idx = segment.start_idx
            end_idx = segment.end_idx
            
            if start_idx >= len(data) or end_idx > len(data):
                continue
            
            segment_data = data.iloc[start_idx:end_idx]
            
            # Calculate basic statistics
            segment_stats = {
                'length': end_idx - start_idx,
                'duration_hours': (segment.end_time - segment.start_time).total_seconds() / 3600,
                'mean_volatility': segment.volatility_level,
                'volatility_std': segment_data['close'].pct_change().std(),
                'mean_return': segment_data['close'].pct_change().mean(),
                'return_std': segment_data['close'].pct_change().std(),
                'volume_mean': segment_data['volume'].mean() if 'volume' in segment_data.columns else 0,
                'volume_std': segment_data['volume'].std() if 'volume' in segment_data.columns else 0
            }
            
            # Add target statistics if available
            if targets is not None and start_idx < len(targets) and end_idx <= len(targets):
                segment_targets = targets.iloc[start_idx:end_idx]
                segment_stats.update({
                    'target_mean': segment_targets.mean(),
                    'target_std': segment_targets.std(),
                    'target_ic': segment_data['close'].pct_change().corr(segment_targets) if len(segment_targets) > 1 else 0
                })
            
            stats[f'segment_{i}'] = segment_stats
        
        return stats
    
    def _initialize_bocpd(self, volatility_proxy: pd.Series) -> Dict[str, Any]:
        """Initialize BOCPD with historical data."""
        # Feed historical data to BOCPD to initialize state
        tprint_debug(
            "Initializing BOCPD",
            {
                'initial_points': min(len(volatility_proxy.dropna()), 100),
                'hazard': self.bocpd.hazard,
            },
        )
        for value in volatility_proxy.dropna().values[:100]:  # Use first 100 points
            self.bocpd.update(value)

        return {
            'hazard': self.bocpd.hazard,
            'current_state': self.bocpd.get_regime_posterior(),
            'run_length': self.bocpd.run_length
        }
    
    def _identify_regime_transitions(self, segments: List[RegimeSegment]) -> List[Dict[str, Any]]:
        """Identify regime transitions between segments."""
        transitions = []
        
        for i in range(1, len(segments)):
            prev_segment = segments[i-1]
            curr_segment = segments[i]
            
            if prev_segment.regime_type != curr_segment.regime_type:
                transition = {
                    'transition_time': curr_segment.start_time,
                    'from_regime': prev_segment.regime_type,
                    'to_regime': curr_segment.regime_type,
                    'volatility_change': curr_segment.volatility_level - prev_segment.volatility_level,
                    'confidence': 0.8  # Simplified
                }
                transitions.append(transition)
        
        return transitions
    
    def update_regime_monitoring(self, new_observation: float) -> Dict[str, Any]:
        """Update regime monitoring with new observation."""
        tprint_debug("Updating regime monitoring", {'observation': new_observation})
        bocpd_result = self.bocpd.update(new_observation)

        tprint_info(
            "Regime monitoring updated",
            {
                'change_detected': bocpd_result['change_detected'],
                'cp_probability': bocpd_result['change_point_probability'],
            },
        )
        return {
            'bocpd_result': bocpd_result,
            'current_regime_posterior': self.bocpd.get_regime_posterior(),
            'regime_change_detected': bocpd_result['change_detected']
        }