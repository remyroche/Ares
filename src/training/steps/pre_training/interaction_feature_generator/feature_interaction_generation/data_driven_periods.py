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
- VectorBT-optimized rolling operations
- Memory-efficient batch processing
- Parallel period analysis
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy import stats
from scipy.signal import find_peaks
import warnings
import time
from contextlib import contextmanager

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

# Import VectorBT optimization utilities
from src.feature_generation.utils.vectorbt_rolling_optimizer import (
    VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
)
from src.feature_generation.utils.unified_vectorization_manager import (
    UnifiedVectorizationManager, VectorizationConfig, get_unified_vectorization_manager
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
    
    Enhanced with VectorBT optimizations for improved performance and memory efficiency.
    """
    
    def __init__(self, 
                 min_period: int = 2,
                 max_period: int = 200,
                 max_periods: int = 8,
                 min_data_points: int = 100,
                 enable_vectorbt: bool = True,
                 enable_parallel: bool = True,
                 memory_efficient: bool = True,
                 chunk_size: int = 1000):
        """
        Initialize the period selector with VectorBT optimizations.
        
        Args:
            min_period: Minimum period to consider
            max_period: Maximum period to consider
            max_periods: Maximum number of periods to return
            min_data_points: Minimum data points required for analysis
            enable_vectorbt: Enable VectorBT optimizations
            enable_parallel: Enable parallel processing
            memory_efficient: Enable memory optimization
            chunk_size: Size of data chunks for processing
        """
        self.min_period = min_period
        self.max_period = max_period
        self.max_periods = max_periods
        self.min_data_points = min_data_points
        self.enable_vectorbt = enable_vectorbt
        self.enable_parallel = enable_parallel
        self.memory_efficient = memory_efficient
        self.chunk_size = chunk_size
        
        # Initialize VectorBT optimizers
        self.rolling_optimizer = None
        self.vectorization_manager = None
        self._initialize_vectorbt_components()
        
        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'batch_operations': 0,
            'memory_optimizations': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Cache for computed results
        self._result_cache = {}
        self._cache_enabled = True
        self._max_cache_size = 100
        
        tprint_info(f"🔧 Data-driven period selector initialized with VectorBT optimizations")
        tprint_info(f"📊 Period range: {min_period} - {max_period}")
        tprint_info(f"📊 Max periods: {max_periods}")
        tprint_info(f"🚀 VectorBT enabled: {enable_vectorbt}")
        tprint_info(f"⚡ Parallel processing: {enable_parallel}")
        tprint_info(f"💾 Memory efficient: {memory_efficient}")
    
    def _initialize_vectorbt_components(self):
        """Initialize VectorBT optimization components."""
        try:
            if self.enable_vectorbt:
                # Initialize VectorBT rolling optimizer
                self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=False,  # Can be enabled if needed
                    enable_parallel=self.enable_parallel,
                    memory_efficient=self.memory_efficient,
                    chunk_size=self.chunk_size
                )
                
                # Initialize unified vectorization manager
                config = VectorizationConfig(
                    enable_vectorbt=True,
                    enable_gpu=False,
                    enable_parallel=self.enable_parallel,
                    memory_efficient=self.memory_efficient,
                    chunk_size=self.chunk_size,
                    enable_monitoring=True,
                    batch_size=10000,
                    enable_batch_processing=True
                )
                self.vectorization_manager = get_unified_vectorization_manager(config)
                
                tprint_success("✅ VectorBT components initialized successfully")
            else:
                tprint_info("ℹ️ VectorBT optimizations disabled")
                
        except Exception as e:
            tprint_warning(f"⚠️ VectorBT initialization failed: {e}")
            self.rolling_optimizer = None
            self.vectorization_manager = None
    
    def analyze_data_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data characteristics to inform period selection using VectorBT optimizations."""
        start_time = time.time()
        self.performance_stats['total_operations'] += 1
        
        # Check cache first
        if self._cache_enabled:
            cache_key = self._generate_cache_key('analyze_characteristics', data)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                self.performance_stats['cache_hits'] += 1
                return cached_result
            self.performance_stats['cache_misses'] += 1
        
        characteristics = {}
        
        # Basic data info
        characteristics['data_length'] = len(data)
        characteristics['data_frequency'] = self._detect_frequency(data)
        characteristics['timeframe_minutes'] = self._get_timeframe_minutes(data)
        
        # Optimize data for processing if memory efficient mode is enabled
        if self.memory_efficient and self.vectorization_manager:
            data = self.vectorization_manager.optimize_dataframe(data)
            self.performance_stats['memory_optimizations'] += 1
        
        # Batch process multiple analyses using VectorBT
        if self.vectorization_manager and len(data) > 1000:
            characteristics.update(self._batch_analyze_characteristics(data))
        else:
            # Fallback to individual analysis
            characteristics.update(self._individual_analyze_characteristics(data))
        
        # Cache result
        if self._cache_enabled:
            self._put_in_cache(cache_key, characteristics)
        
        self.performance_stats['total_time'] += time.time() - start_time
        return characteristics
    
    def _batch_analyze_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Batch analyze characteristics using VectorBT optimizations."""
        characteristics = {}
        
        try:
            # Prepare feature configurations for batch processing
            feature_configs = []
            
            if 'close' in data.columns:
                # Volatility analysis
                feature_configs.extend([
                    {'name': 'volatility_5', 'type': 'rolling', 'params': {'operation': 'std', 'window': 5, 'column': 'close'}},
                    {'name': 'volatility_20', 'type': 'rolling', 'params': {'operation': 'std', 'window': 20, 'column': 'close'}},
                    {'name': 'volatility_50', 'type': 'rolling', 'params': {'operation': 'std', 'window': 50, 'column': 'close'}},
                ])
                
                # Trend analysis
                feature_configs.extend([
                    {'name': 'sma_5', 'type': 'rolling', 'params': {'operation': 'mean', 'window': 5, 'column': 'close'}},
                    {'name': 'sma_20', 'type': 'rolling', 'params': {'operation': 'mean', 'window': 20, 'column': 'close'}},
                    {'name': 'sma_50', 'type': 'rolling', 'params': {'operation': 'mean', 'window': 50, 'column': 'close'}},
                ])
            
            if 'volume' in data.columns:
                # Volume analysis
                feature_configs.extend([
                    {'name': 'volume_sma_5', 'type': 'rolling', 'params': {'operation': 'mean', 'window': 5, 'column': 'volume'}},
                    {'name': 'volume_sma_20', 'type': 'rolling', 'params': {'operation': 'mean', 'window': 20, 'column': 'volume'}},
                ])
            
            # Process all features in batch
            if feature_configs:
                features = self.vectorization_manager.batch_process_features(data, feature_configs)
                self.performance_stats['batch_operations'] += 1
                
                # Extract characteristics from batch results
                if 'close' in data.columns:
                    returns = data['close'].pct_change().dropna()
                    characteristics['volatility'] = returns.std()
                    characteristics['volatility_clusters'] = self._detect_volatility_clusters_vectorbt(features)
                    characteristics['trend_cycles'] = self._detect_trend_cycles_vectorbt(features)
                
                if 'volume' in data.columns:
                    characteristics['volume_patterns'] = self._analyze_volume_patterns_vectorbt(features)
            
            # Market regime analysis
            characteristics['regime_changes'] = self._detect_regime_changes_vectorbt(data, features if 'close' in data.columns else None)
            
        except Exception as e:
            tprint_warning(f"⚠️ Batch analysis failed: {e}, falling back to individual analysis")
            return self._individual_analyze_characteristics(data)
        
        return characteristics
    
    def _individual_analyze_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Individual analysis fallback when batch processing is not available."""
        characteristics = {}
        
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
    
    def _detect_volatility_clusters_vectorbt(self, features: pd.DataFrame) -> List[int]:
        """Detect volatility clustering periods using VectorBT-optimized features."""
        try:
            vol_clusters = []
            
            # Use pre-computed volatility features
            for col in features.columns:
                if col.startswith('volatility_'):
                    vol_series = features[col].dropna()
                    if len(vol_series) > 10:
                        # Find volatility clusters using pre-computed rolling volatility
                        vol_threshold = vol_series.quantile(0.8)
                        clusters = vol_series > vol_threshold
                        
                        # Calculate average cluster length
                        cluster_lengths = self._find_pattern_periods(clusters)
                        if cluster_lengths:
                            avg_length = np.mean(cluster_lengths)
                            if self.min_period <= avg_length <= self.max_period:
                                vol_clusters.append(int(avg_length))
            
            return vol_clusters[:3]  # Limit to top 3
            
        except Exception as e:
            tprint_debug(f"⚠️ VectorBT volatility clustering failed: {e}")
            return []
    
    def _detect_trend_cycles_vectorbt(self, features: pd.DataFrame) -> List[int]:
        """Detect trend cycles using VectorBT-optimized features."""
        try:
            cycle_lengths = []
            
            # Use pre-computed SMA features to detect cycles
            sma_cols = [col for col in features.columns if col.startswith('sma_')]
            
            for col in sma_cols:
                sma_series = features[col].dropna()
                if len(sma_series) > 20:
                    # Find peaks and troughs in SMA
                    peaks, _ = find_peaks(sma_series, distance=5)
                    troughs, _ = find_peaks(-sma_series, distance=5)
                    
                    # Calculate cycle lengths
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
            tprint_debug(f"⚠️ VectorBT trend cycle detection failed: {e}")
            return []
    
    def _analyze_volume_patterns_vectorbt(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns using VectorBT-optimized features."""
        try:
            volume_patterns = {}
            
            # Use pre-computed volume features
            volume_cols = [col for col in features.columns if col.startswith('volume_')]
            
            if volume_cols:
                # Find volume spikes using pre-computed moving averages
                volume_sma_5 = features.get('volume_sma_5', pd.Series())
                volume_sma_20 = features.get('volume_sma_20', pd.Series())
                
                if not volume_sma_5.empty and not volume_sma_20.empty:
                    # Find volume spikes
                    vol_spikes = volume_sma_5 > volume_sma_20 * 1.5
                    spike_periods = self._find_pattern_periods(vol_spikes)
                    volume_patterns['spike_periods'] = spike_periods
                
                # Volume trend analysis
                volume_patterns['volume_trend'] = self._detect_trend_cycles_vectorbt(features)
            
            return volume_patterns
            
        except Exception as e:
            tprint_debug(f"⚠️ VectorBT volume analysis failed: {e}")
            return {}
    
    def _detect_regime_changes_vectorbt(self, data: pd.DataFrame, features: Optional[pd.DataFrame] = None) -> List[int]:
        """Detect market regime changes using VectorBT-optimized features."""
        try:
            if 'close' not in data.columns or len(data) < 50:
                return []
            
            # Use pre-computed volatility features if available
            if features is not None and 'volatility_20' in features.columns:
                volatility = features['volatility_20'].dropna()
            else:
                # Fallback to manual calculation
                returns = data['close'].pct_change().dropna()
                if self.rolling_optimizer:
                    volatility = self.rolling_optimizer.rolling_std(returns, window=20)
                else:
                    volatility = returns.rolling(20).std()
            
            # Find regime changes (high vol periods)
            vol_threshold = volatility.quantile(0.7)
            regime_changes = volatility > vol_threshold
            
            # Calculate regime lengths
            regime_lengths = self._find_pattern_periods(regime_changes)
            
            return regime_lengths[:3]  # Limit to top 3
            
        except Exception as e:
            tprint_debug(f"⚠️ VectorBT regime detection failed: {e}")
            return []
    
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
        """Detect volatility clustering periods using VectorBT optimizations."""
        try:
            # Calculate rolling volatility using VectorBT optimizer
            vol_windows = [5, 10, 20, 50, 100]
            vol_clusters = []
            
            for window in vol_windows:
                if len(returns) > window * 2:
                    # Use VectorBT rolling optimizer if available
                    if self.rolling_optimizer:
                        rolling_vol = self.rolling_optimizer.rolling_std(returns, window=window)
                        self.performance_stats['vectorbt_operations'] += 1
                    else:
                        rolling_vol = returns.rolling(window).std()
                        self.performance_stats['pandas_fallbacks'] += 1
                    
                    # Find volatility clusters (high vol periods)
                    vol_threshold = rolling_vol.quantile(0.8)
                    clusters = rolling_vol > vol_threshold
                    
                    # Calculate average cluster length
                    cluster_lengths = self._find_pattern_periods(clusters)
                    
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
    
    def _generate_cache_key(self, operation: str, data: pd.DataFrame) -> str:
        """Generate cache key for operation."""
        import hashlib
        
        # Create hash of data characteristics and operation
        data_hash = hashlib.md5(str(data.shape).encode()).hexdigest()[:8]
        return f"{operation}_{data_hash}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get result from cache."""
        if not self._cache_enabled:
            return None
        
        try:
            if cache_key in self._result_cache:
                return self._result_cache[cache_key]
        except Exception as e:
            tprint_debug(f"Cache retrieval failed: {e}")
        
        return None
    
    def _put_in_cache(self, cache_key: str, result: Any):
        """Put result in cache."""
        if not self._cache_enabled:
            return
        
        try:
            # Limit cache size
            if len(self._result_cache) >= self._max_cache_size:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self._result_cache))
                del self._result_cache[oldest_key]
            
            self._result_cache[cache_key] = result
            
        except Exception as e:
            tprint_debug(f"Cache storage failed: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add VectorBT rolling optimizer stats if available
        if self.rolling_optimizer:
            rolling_stats = self.rolling_optimizer.get_performance_stats()
            stats.update(rolling_stats)
        
        # Add unified vectorization manager stats if available
        if self.vectorization_manager:
            vectorization_stats = self.vectorization_manager.get_performance_stats()
            stats.update(vectorization_stats)
        
        # Calculate efficiency metrics
        if stats['total_operations'] > 0:
            stats['average_operation_time'] = stats['total_time'] / stats['total_operations']
            stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_operations']
            stats['batch_usage_rate'] = stats['batch_operations'] / stats['total_operations']
            
            # Cache statistics
            total_cache_ops = stats['cache_hits'] + stats['cache_misses']
            if total_cache_ops > 0:
                stats['cache_hit_rate'] = (stats['cache_hits'] / total_cache_ops) * 100
            else:
                stats['cache_hit_rate'] = 0
        else:
            stats['average_operation_time'] = 0
            stats['vectorbt_usage_rate'] = 0
            stats['batch_usage_rate'] = 0
            stats['cache_hit_rate'] = 0
        
        return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'batch_operations': 0,
            'memory_optimizations': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Reset component stats
        if self.rolling_optimizer:
            self.rolling_optimizer.reset_stats()
        
        if self.vectorization_manager:
            self.vectorization_manager.reset_stats()
        
        # Clear cache
        self._result_cache.clear()
    
    @contextmanager
    def performance_monitoring(self, operation_name: str):
        """Context manager for performance monitoring."""
        start_time = time.time()
        start_memory = 0  # Could add memory monitoring here
        
        try:
            yield
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            
            tprint_performance(f"Operation {operation_name}: {execution_time:.3f}s")
    
    def optimize_for_large_datasets(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data for large dataset processing."""
        if not self.memory_efficient or len(data) < self.chunk_size:
            return data
        
        try:
            if self.vectorization_manager:
                return self.vectorization_manager.optimize_dataframe(data)
            else:
                # Basic optimization
                optimized_data = data.copy()
                
                # Optimize data types
                for column in optimized_data.columns:
                    if optimized_data[column].dtype == 'float64':
                        if (optimized_data[column].min() >= np.finfo(np.float32).min and 
                            optimized_data[column].max() <= np.finfo(np.float32).max):
                            optimized_data[column] = optimized_data[column].astype(np.float32)
                
                return optimized_data
                
        except Exception as e:
            tprint_warning(f"⚠️ Data optimization failed: {e}")
            return data
    
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


# Convenience functions
def get_data_driven_periods(data: pd.DataFrame, 
                          target_timeframe: Optional[str] = None,
                          max_periods: int = 8,
                          enable_vectorbt: bool = True,
                          enable_parallel: bool = True,
                          memory_efficient: bool = True) -> List[int]:
    """
    Get data-driven periods for cross-timeframe features with VectorBT optimizations.
    
    Args:
        data: Input data
        target_timeframe: Target timeframe (5m, 15m, 60m, etc.)
        max_periods: Maximum number of periods to return
        enable_vectorbt: Enable VectorBT optimizations
        enable_parallel: Enable parallel processing
        memory_efficient: Enable memory optimization
        
    Returns:
        List of optimal periods
    """
    selector = DataDrivenPeriodSelector(
        max_periods=max_periods,
        enable_vectorbt=enable_vectorbt,
        enable_parallel=enable_parallel,
        memory_efficient=memory_efficient
    )
    result = selector.select_optimal_periods(data, target_timeframe)
    return result.optimal_periods


def get_data_driven_periods_with_stats(data: pd.DataFrame, 
                                     target_timeframe: Optional[str] = None,
                                     max_periods: int = 8,
                                     enable_vectorbt: bool = True,
                                     enable_parallel: bool = True,
                                     memory_efficient: bool = True) -> Tuple[List[int], Dict[str, Any]]:
    """
    Get data-driven periods with performance statistics.
    
    Args:
        data: Input data
        target_timeframe: Target timeframe (5m, 15m, 60m, etc.)
        max_periods: Maximum number of periods to return
        enable_vectorbt: Enable VectorBT optimizations
        enable_parallel: Enable parallel processing
        memory_efficient: Enable memory optimization
        
    Returns:
        Tuple of (optimal periods, performance statistics)
    """
    selector = DataDrivenPeriodSelector(
        max_periods=max_periods,
        enable_vectorbt=enable_vectorbt,
        enable_parallel=enable_parallel,
        memory_efficient=memory_efficient
    )
    result = selector.select_optimal_periods(data, target_timeframe)
    stats = selector.get_performance_stats()
    return result.optimal_periods, stats


def benchmark_period_selector(data: pd.DataFrame, 
                            target_timeframe: Optional[str] = None,
                            max_periods: int = 8,
                            trials: int = 3) -> Dict[str, Any]:
    """
    Benchmark period selector performance across different configurations.
    
    Args:
        data: Input data
        target_timeframe: Target timeframe (5m, 15m, 60m, etc.)
        max_periods: Maximum number of periods to return
        trials: Number of trials to run for each configuration
        
    Returns:
        Benchmarking results
    """
    configurations = [
        {'enable_vectorbt': False, 'enable_parallel': False, 'memory_efficient': False, 'name': 'baseline'},
        {'enable_vectorbt': True, 'enable_parallel': False, 'memory_efficient': False, 'name': 'vectorbt_only'},
        {'enable_vectorbt': True, 'enable_parallel': True, 'memory_efficient': False, 'name': 'vectorbt_parallel'},
        {'enable_vectorbt': True, 'enable_parallel': True, 'memory_efficient': True, 'name': 'vectorbt_optimized'},
    ]
    
    results = {}
    
    for config in configurations:
        config_name = config.pop('name')
        times = []
        
        for trial in range(trials):
            try:
                selector = DataDrivenPeriodSelector(max_periods=max_periods, **config)
                start_time = time.time()
                result = selector.select_optimal_periods(data, target_timeframe)
                execution_time = time.time() - start_time
                times.append(execution_time)
            except Exception as e:
                tprint_warning(f"⚠️ Configuration {config_name} trial {trial} failed: {e}")
                continue
        
        if times:
            results[config_name] = {
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'trials_completed': len(times)
            }
    
    return results

