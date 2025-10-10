"""
Data-Driven Period Selection for Cross-Timeframe Features

This module implements intelligent period selection based on data characteristics
rather than using hardcoded periods. It analyzes the data to determine optimal
periods for cross-timeframe feature generation.

Key Features:
- Analyzes data frequency and length
- Detects natural market cycles
- Optimizes periods for feature diversity
- Considers computational constraints
- Adapts to different timeframes (5m, 15m, 60m)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.signal import find_peaks
import warnings

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

logger = logging.getLogger(__name__)


@dataclass
class PeriodAnalysisResult:
    """Result of period analysis."""
    optimal_periods: List[int]
    period_categories: Dict[str, List[int]]
    analysis_metadata: Dict[str, Any]
    confidence_score: float


class DataDrivenPeriodSelector:
    """
    Selects optimal periods for cross-timeframe features based on data characteristics.
    """
    
    def __init__(self, 
                 min_period: int = 2,
                 max_period: int = 200,
                 max_periods: int = 8,
                 min_data_points: int = 100):
        """
        Initialize the period selector.
        
        Args:
            min_period: Minimum period to consider
            max_period: Maximum period to consider
            max_periods: Maximum number of periods to return
            min_data_points: Minimum data points required for analysis
        """
        self.min_period = min_period
        self.max_period = max_period
        self.max_periods = max_periods
        self.min_data_points = min_data_points
        
        tprint_info(f"🔧 Data-driven period selector initialized")
        tprint_info(f"📊 Period range: {min_period} - {max_period}")
        tprint_info(f"📊 Max periods: {max_periods}")
    
    def analyze_data_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data characteristics to inform period selection."""
        characteristics = {}
        
        # Basic data info
        characteristics['data_length'] = len(data)
        characteristics['data_frequency'] = self._detect_frequency(data)
        characteristics['timeframe_minutes'] = self._get_timeframe_minutes(data)
        
        # Volatility analysis
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            characteristics['volatility'] = returns.std()
            characteristics['volatility_clusters'] = self._detect_volatility_clusters(returns)
        
        # Volume analysis
        if 'volume' in data.columns:
            characteristics['volume_patterns'] = self._analyze_volume_patterns(data['volume'])
        
        # Price trend analysis
        if 'close' in data.columns:
            characteristics['trend_cycles'] = self._detect_trend_cycles(data['close'])
            characteristics['seasonality'] = self._detect_seasonality(data['close'])
        
        # Market regime analysis
        characteristics['regime_changes'] = self._detect_regime_changes(data)
        
        return characteristics
    
    def select_optimal_periods(self, data: pd.DataFrame, 
                             target_timeframe: Optional[str] = None) -> PeriodAnalysisResult:
        """
        Select optimal periods for cross-timeframe features.
        
        Args:
            data: Input data
            target_timeframe: Target timeframe (5m, 15m, 60m, etc.)
            
        Returns:
            PeriodAnalysisResult with optimal periods
        """
        tprint_info("🔍 Analyzing data characteristics for period selection...")
        
        # Analyze data characteristics
        characteristics = self.analyze_data_characteristics(data)
        
        # Check if we have enough data
        if characteristics['data_length'] < self.min_data_points:
            tprint_warning(f"⚠️ Insufficient data ({characteristics['data_length']} < {self.min_data_points})")
            return self._get_fallback_periods(characteristics)
        
        # Get base periods from timeframe
        base_periods = self._get_base_periods_from_timeframe(
            characteristics.get('timeframe_minutes', 15),
            target_timeframe
        )
        
        # Analyze market cycles
        cycle_periods = self._detect_market_cycles(data, characteristics)
        
        # Analyze volatility patterns
        volatility_periods = self._analyze_volatility_periods(data, characteristics)
        
        # Analyze volume patterns
        volume_periods = self._analyze_volume_periods(data, characteristics)
        
        # Combine and optimize periods
        all_candidate_periods = list(set(
            base_periods + cycle_periods + volatility_periods + volume_periods
        ))
        
        # Filter and rank periods
        filtered_periods = self._filter_periods(all_candidate_periods, characteristics)
        ranked_periods = self._rank_periods(filtered_periods, data, characteristics)
        
        # Select final periods
        optimal_periods = ranked_periods[:self.max_periods]
        
        # Categorize periods
        period_categories = self._categorize_periods(optimal_periods, characteristics)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(optimal_periods, characteristics)
        
        tprint_success(f"✅ Selected {len(optimal_periods)} optimal periods: {optimal_periods}")
        tprint_info(f"📊 Confidence score: {confidence_score:.2f}")
        
        return PeriodAnalysisResult(
            optimal_periods=optimal_periods,
            period_categories=period_categories,
            analysis_metadata=characteristics,
            confidence_score=confidence_score
        )
    
    def _detect_frequency(self, data: pd.DataFrame) -> str:
        """Detect the frequency of the data."""
        if not isinstance(data.index, pd.DatetimeIndex):
            return 'unknown'
        
        if len(data) < 2:
            return 'unknown'
        
        # Calculate time differences
        time_diffs = data.index.to_series().diff().dropna()
        median_diff = time_diffs.median()
        
        # Convert to minutes
        if median_diff < pd.Timedelta(minutes=1):
            return 'sub-minute'
        elif median_diff < pd.Timedelta(minutes=5):
            return '1m'
        elif median_diff < pd.Timedelta(minutes=10):
            return '5m'
        elif median_diff < pd.Timedelta(minutes=20):
            return '15m'
        elif median_diff < pd.Timedelta(minutes=90):
            return '60m'
        elif median_diff < pd.Timedelta(hours=2):
            return '4h'
        elif median_diff < pd.Timedelta(hours=6):
            return '1d'
        else:
            return 'weekly'
    
    def _get_timeframe_minutes(self, data: pd.DataFrame) -> int:
        """Get timeframe in minutes."""
        if not isinstance(data.index, pd.DatetimeIndex):
            return 15  # Default
        
        if len(data) < 2:
            return 15
        
        time_diffs = data.index.to_series().diff().dropna()
        median_diff = time_diffs.median()
        
        return int(median_diff.total_seconds() / 60)
    
    def _get_base_periods_from_timeframe(self, timeframe_minutes: int, 
                                       target_timeframe: Optional[str] = None) -> List[int]:
        """Get base periods based on timeframe."""
        if target_timeframe:
            # Parse target timeframe
            if target_timeframe.endswith('m'):
                target_minutes = int(target_timeframe[:-1])
            elif target_timeframe.endswith('h'):
                target_minutes = int(target_timeframe[:-1]) * 60
            elif target_timeframe.endswith('d'):
                target_minutes = int(target_timeframe[:-1]) * 24 * 60
            else:
                target_minutes = 15  # Default
        else:
            target_minutes = timeframe_minutes
        
        # Calculate periods based on target timeframe
        # Use multiples that make sense for the target timeframe
        base_periods = []
        
        # Short-term periods (2-10x current timeframe)
        for multiplier in [2, 3, 5, 10]:
            period = multiplier * (target_minutes // timeframe_minutes)
            if self.min_period <= period <= self.max_period:
                base_periods.append(period)
        
        # Medium-term periods (20-50x current timeframe)
        for multiplier in [20, 30, 50]:
            period = multiplier * (target_minutes // timeframe_minutes)
            if self.min_period <= period <= self.max_period:
                base_periods.append(period)
        
        # Long-term periods (100x+ current timeframe)
        for multiplier in [100, 200]:
            period = multiplier * (target_minutes // timeframe_minutes)
            if self.min_period <= period <= self.max_period:
                base_periods.append(period)
        
        return base_periods
    
    def _detect_market_cycles(self, data: pd.DataFrame, 
                            characteristics: Dict[str, Any]) -> List[int]:
        """Detect market cycles using spectral analysis."""
        if 'close' not in data.columns or len(data) < 50:
            return []
        
        try:
            prices = data['close'].values
            returns = np.diff(np.log(prices))
            
            # Use FFT to detect cycles
            fft = np.fft.fft(returns)
            freqs = np.fft.fftfreq(len(returns))
            
            # Find significant frequencies
            power_spectrum = np.abs(fft) ** 2
            significant_freqs = freqs[power_spectrum > np.percentile(power_spectrum, 90)]
            
            # Convert frequencies to periods
            cycle_periods = []
            for freq in significant_freqs:
                if freq > 0:  # Only positive frequencies
                    period = int(1 / freq)
                    if self.min_period <= period <= self.max_period:
                        cycle_periods.append(period)
            
            return cycle_periods[:5]  # Limit to top 5 cycles
            
        except Exception as e:
            tprint_debug(f"⚠️ Cycle detection failed: {e}")
            return []
    
    def _detect_volatility_clusters(self, returns: pd.Series) -> List[int]:
        """Detect volatility clustering periods."""
        try:
            # Calculate rolling volatility
            vol_windows = [5, 10, 20, 50, 100]
            vol_clusters = []
            
            for window in vol_windows:
                if len(returns) > window * 2:
                    rolling_vol = returns.rolling(window).std()
                    
                    # Find volatility clusters (high vol periods)
                    vol_threshold = rolling_vol.quantile(0.8)
                    clusters = rolling_vol > vol_threshold
                    
                    # Calculate average cluster length
                    cluster_lengths = []
                    in_cluster = False
                    current_length = 0
                    
                    for is_high_vol in clusters:
                        if is_high_vol:
                            if not in_cluster:
                                in_cluster = True
                                current_length = 1
                            else:
                                current_length += 1
                        else:
                            if in_cluster:
                                cluster_lengths.append(current_length)
                                in_cluster = False
                                current_length = 0
                    
                    if cluster_lengths:
                        avg_cluster_length = np.mean(cluster_lengths)
                        if self.min_period <= avg_cluster_length <= self.max_period:
                            vol_clusters.append(int(avg_cluster_length))
            
            return vol_clusters[:3]  # Limit to top 3
            
        except Exception as e:
            tprint_debug(f"⚠️ Volatility clustering failed: {e}")
            return []
    
    def _analyze_volume_patterns(self, volume: pd.Series) -> Dict[str, Any]:
        """Analyze volume patterns to inform period selection."""
        try:
            # Calculate volume moving averages
            vol_ma_5 = volume.rolling(5).mean()
            vol_ma_20 = volume.rolling(20).mean()
            
            # Find volume spikes
            vol_spikes = volume > vol_ma_20 * 2
            spike_periods = self._find_pattern_periods(vol_spikes)
            
            return {
                'spike_periods': spike_periods,
                'volume_trend': self._detect_trend_cycles(volume)
            }
            
        except Exception as e:
            tprint_debug(f"⚠️ Volume analysis failed: {e}")
            return {}
    
    def _detect_trend_cycles(self, series: pd.Series) -> List[int]:
        """Detect trend cycles using peak detection."""
        try:
            if len(series) < 20:
                return []
            
            # Smooth the series
            smoothed = series.rolling(5).mean()
            
            # Find peaks and troughs
            peaks, _ = find_peaks(smoothed, distance=5)
            troughs, _ = find_peaks(-smoothed, distance=5)
            
            # Calculate cycle lengths
            cycle_lengths = []
            all_extrema = sorted(list(peaks) + list(troughs))
            
            for i in range(1, len(all_extrema)):
                cycle_length = all_extrema[i] - all_extrema[i-1]
                if self.min_period <= cycle_length <= self.max_period:
                    cycle_lengths.append(cycle_length)
            
            # Return most common cycle lengths
            if cycle_lengths:
                from collections import Counter
                most_common = Counter(cycle_lengths).most_common(3)
                return [length for length, count in most_common]
            
            return []
            
        except Exception as e:
            tprint_debug(f"⚠️ Trend cycle detection failed: {e}")
            return []
    
    def _detect_seasonality(self, series: pd.Series) -> List[int]:
        """Detect seasonal patterns."""
        try:
            if len(series) < 100:
                return []
            
            # Look for daily, weekly patterns
            daily_period = 24 * 60 // self._get_timeframe_minutes(pd.DataFrame(index=series.index))
            weekly_period = daily_period * 7
            
            seasonal_periods = []
            for period in [daily_period, weekly_period]:
                if self.min_period <= period <= self.max_period:
                    seasonal_periods.append(period)
            
            return seasonal_periods
            
        except Exception as e:
            tprint_debug(f"⚠️ Seasonality detection failed: {e}")
            return []
    
    def _detect_regime_changes(self, data: pd.DataFrame) -> List[int]:
        """Detect market regime changes."""
        try:
            if 'close' not in data.columns or len(data) < 50:
                return []
            
            # Use volatility and trend to detect regimes
            returns = data['close'].pct_change().dropna()
            volatility = returns.rolling(20).std()
            
            # Find regime changes (high vol periods)
            vol_threshold = volatility.quantile(0.7)
            regime_changes = volatility > vol_threshold
            
            # Calculate regime lengths
            regime_lengths = self._find_pattern_periods(regime_changes)
            
            return regime_lengths[:3]  # Limit to top 3
            
        except Exception as e:
            tprint_debug(f"⚠️ Regime detection failed: {e}")
            return []
    
    def _find_pattern_periods(self, pattern: pd.Series) -> List[int]:
        """Find periods in a boolean pattern."""
        try:
            # Find consecutive True values
            pattern_lengths = []
            in_pattern = False
            current_length = 0
            
            for is_true in pattern:
                if is_true:
                    if not in_pattern:
                        in_pattern = True
                        current_length = 1
                    else:
                        current_length += 1
                else:
                    if in_pattern:
                        pattern_lengths.append(current_length)
                        in_pattern = False
                        current_length = 0
            
            # Return average pattern length if it's reasonable
            if pattern_lengths:
                avg_length = np.mean(pattern_lengths)
                if self.min_period <= avg_length <= self.max_period:
                    return [int(avg_length)]
            
            return []
            
        except Exception as e:
            tprint_debug(f"⚠️ Pattern analysis failed: {e}")
            return []
    
    def _analyze_volatility_periods(self, data: pd.DataFrame, 
                                  characteristics: Dict[str, Any]) -> List[int]:
        """Analyze volatility patterns for period selection."""
        if 'close' not in data.columns:
            return []
        
        try:
            returns = data['close'].pct_change().dropna()
            volatility = returns.rolling(20).std()
            
            # Find volatility clusters
            vol_clusters = characteristics.get('volatility_clusters', [])
            
            # Find volatility mean reversion periods
            vol_ma = volatility.rolling(50).mean()
            vol_ratio = volatility / vol_ma
            
            # Find periods where volatility reverts to mean
            mean_reversion = (vol_ratio < 0.8) | (vol_ratio > 1.2)
            reversion_periods = self._find_pattern_periods(mean_reversion)
            
            return vol_clusters + reversion_periods
            
        except Exception as e:
            tprint_debug(f"⚠️ Volatility period analysis failed: {e}")
            return []
    
    def _analyze_volume_periods(self, data: pd.DataFrame, 
                              characteristics: Dict[str, Any]) -> List[int]:
        """Analyze volume patterns for period selection."""
        if 'volume' not in data.columns:
            return []
        
        try:
            volume_patterns = characteristics.get('volume_patterns', {})
            return volume_patterns.get('spike_periods', [])
            
        except Exception as e:
            tprint_debug(f"⚠️ Volume period analysis failed: {e}")
            return []
    
    def _filter_periods(self, periods: List[int], 
                       characteristics: Dict[str, Any]) -> List[int]:
        """Filter periods based on data characteristics."""
        filtered = []
        
        for period in periods:
            # Check if period is within bounds
            if not (self.min_period <= period <= self.max_period):
                continue
            
            # Check if period is reasonable for data length
            data_length = characteristics.get('data_length', 0)
            if period > data_length // 4:  # Don't use periods longer than 1/4 of data
                continue
            
            # Check if period makes sense for timeframe
            timeframe_minutes = characteristics.get('timeframe_minutes', 15)
            if period < 2:  # At least 2 periods
                continue
            
            filtered.append(period)
        
        return sorted(list(set(filtered)))
    
    def _rank_periods(self, periods: List[int], data: pd.DataFrame, 
                     characteristics: Dict[str, Any]) -> List[int]:
        """Rank periods by their potential usefulness."""
        if not periods:
            return []
        
        try:
            scores = []
            
            for period in periods:
                score = 0
                
                # Diversity score (prefer periods that are different from others)
                other_periods = [p for p in periods if p != period]
                if other_periods:
                    min_diff = min(abs(period - p) for p in other_periods)
                    score += min_diff / max(period, 1)
                
                # Data coverage score (prefer periods that use more data)
                data_length = characteristics.get('data_length', 0)
                coverage = min(period, data_length) / data_length
                score += coverage
                
                # Stability score (prefer periods that are stable across different windows)
                if 'close' in data.columns and len(data) > period * 2:
                    try:
                        returns = data['close'].pct_change().dropna()
                        rolling_vol = returns.rolling(period).std()
                        vol_stability = 1 / (rolling_vol.std() + 1e-8)
                        score += vol_stability
                    except:
                        pass
                
                scores.append((score, period))
            
            # Sort by score (descending)
            scores.sort(reverse=True)
            return [period for score, period in scores]
            
        except Exception as e:
            tprint_debug(f"⚠️ Period ranking failed: {e}")
            return periods
    
    def _categorize_periods(self, periods: List[int], 
                          characteristics: Dict[str, Any]) -> Dict[str, List[int]]:
        """Categorize periods by their characteristics."""
        categories = {
            'short_term': [],
            'medium_term': [],
            'long_term': [],
            'volatility_driven': [],
            'trend_driven': [],
            'volume_driven': []
        }
        
        data_length = characteristics.get('data_length', 0)
        
        for period in periods:
            # Time-based categorization
            if period <= data_length // 20:
                categories['short_term'].append(period)
            elif period <= data_length // 10:
                categories['medium_term'].append(period)
            else:
                categories['long_term'].append(period)
            
            # Pattern-based categorization (simplified)
            if period in characteristics.get('volatility_clusters', []):
                categories['volatility_driven'].append(period)
            
            if period in characteristics.get('trend_cycles', []):
                categories['trend_driven'].append(period)
        
        return categories
    
    def _calculate_confidence_score(self, periods: List[int], 
                                  characteristics: Dict[str, Any]) -> float:
        """Calculate confidence score for the selected periods."""
        if not periods:
            return 0.0
        
        try:
            score = 0.0
            
            # Data sufficiency score
            data_length = characteristics.get('data_length', 0)
            if data_length > 1000:
                score += 0.3
            elif data_length > 500:
                score += 0.2
            elif data_length > 100:
                score += 0.1
            
            # Period diversity score
            if len(periods) >= 3:
                score += 0.2
            elif len(periods) >= 2:
                score += 0.1
            
            # Analysis completeness score
            analysis_components = ['volatility_clusters', 'trend_cycles', 'volume_patterns']
            completed_analyses = sum(1 for comp in analysis_components if comp in characteristics)
            score += (completed_analyses / len(analysis_components)) * 0.3
            
            # Period reasonableness score
            reasonable_periods = sum(1 for p in periods if 2 <= p <= data_length // 4)
            score += (reasonable_periods / len(periods)) * 0.2
            
            return min(score, 1.0)
            
        except Exception as e:
            tprint_debug(f"⚠️ Confidence calculation failed: {e}")
            return 0.5
    
    def _get_fallback_periods(self, characteristics: Dict[str, Any]) -> PeriodAnalysisResult:
        """Get fallback periods when analysis fails."""
        data_length = characteristics.get('data_length', 100)
        timeframe_minutes = characteristics.get('timeframe_minutes', 15)
        
        # Simple fallback based on data length
        if data_length < 50:
            periods = [2, 5, 10]
        elif data_length < 200:
            periods = [5, 10, 20]
        else:
            periods = [10, 20, 50]
        
        # Adjust for timeframe
        periods = [p * (timeframe_minutes // 15) for p in periods]
        periods = [p for p in periods if self.min_period <= p <= self.max_period]
        
        return PeriodAnalysisResult(
            optimal_periods=periods,
            period_categories={'fallback': periods},
            analysis_metadata=characteristics,
            confidence_score=0.3
        )


# Convenience function
def get_data_driven_periods(data: pd.DataFrame, 
                          target_timeframe: Optional[str] = None,
                          max_periods: int = 8) -> List[int]:
    """
    Get data-driven periods for cross-timeframe features.
    
    Args:
        data: Input data
        target_timeframe: Target timeframe (5m, 15m, 60m, etc.)
        max_periods: Maximum number of periods to return
        
    Returns:
        List of optimal periods
    """
    selector = DataDrivenPeriodSelector(max_periods=max_periods)
    result = selector.select_optimal_periods(data, target_timeframe)
    return result.optimal_periods