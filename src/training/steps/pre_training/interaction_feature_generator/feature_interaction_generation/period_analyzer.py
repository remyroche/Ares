"""
Period Analyzer for Data-Driven Period Selection

This module provides data analysis capabilities for determining optimal periods
based on data characteristics, market cycles, and patterns.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
from scipy.signal import find_peaks

from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_performance
)

from .period_analysis_utils import (
    PeriodAnalysisUtils, ValidationError, AnalysisError,
    performance_monitoring, safe_validate_and_execute
)

# Import VectorBT optimization utilities
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    )
    from src.feature_generation.utils.unified_vectorization_manager import (
        UnifiedVectorizationManager, VectorizationConfig, get_unified_vectorization_manager
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    VectorBTRollingOptimizer = None
    get_vectorbt_rolling_optimizer = None
    UnifiedVectorizationManager = None
    VectorizationConfig = None
    get_unified_vectorization_manager = None

logger = logging.getLogger(__name__)


class PeriodAnalyzer:
    """
    Analyzes data characteristics to determine optimal periods for cross-timeframe features.
    
    This class handles all data analysis operations including volatility clustering,
    trend cycle detection, volume pattern analysis, and market regime detection.
    """
    
    def __init__(self, 
                 enable_vectorbt: bool = True,
                 enable_parallel: bool = True,
                 memory_efficient: bool = True,
                 chunk_size: int = 1000):
        """
        Initialize the period analyzer.
        
        Args:
            enable_vectorbt: Enable VectorBT optimizations
            enable_parallel: Enable parallel processing
            memory_efficient: Enable memory optimization
            chunk_size: Size of data chunks for processing
        """
        self.enable_vectorbt = enable_vectorbt
        self.enable_parallel = enable_parallel
        self.memory_efficient = memory_efficient
        self.chunk_size = chunk_size
        
        # Initialize VectorBT components
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
            'total_time': 0.0
        }
        
        tprint_info(f"🔧 Period analyzer initialized")
        tprint_info(f"🚀 VectorBT enabled: {enable_vectorbt}")
        tprint_info(f"⚡ Parallel processing: {enable_parallel}")
        tprint_info(f"💾 Memory efficient: {memory_efficient}")
    
    def _initialize_vectorbt_components(self):
        """Initialize VectorBT optimization components."""
        if not self.enable_vectorbt or not VECTORBT_AVAILABLE:
            tprint_info("ℹ️ VectorBT optimizations disabled or unavailable")
            return
        
        try:
            # Initialize VectorBT rolling optimizer
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=False,
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
            
        except Exception as e:
            tprint_warning(f"⚠️ VectorBT initialization failed: {e}")
            self.rolling_optimizer = None
            self.vectorization_manager = None
    
    def analyze_data_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze data characteristics to inform period selection.
        
        Args:
            data: Input data for analysis
            
        Returns:
            Dictionary containing data characteristics
            
        Raises:
            ValidationError: If input data is invalid
            AnalysisError: If analysis fails
        """
        def _validate_inputs():
            PeriodAnalysisUtils.validate_dataframe(data, min_length=100, operation_name="data_analysis")
        
        def _analyze_characteristics():
            characteristics = {}
            
            # Basic data info
            characteristics['data_length'] = len(data)
            characteristics['data_frequency'] = PeriodAnalysisUtils.detect_frequency(data)
            characteristics['timeframe_minutes'] = PeriodAnalysisUtils.get_timeframe_minutes(data)
            
            # Optimize data for processing if memory efficient mode is enabled
            if self.memory_efficient and self.vectorization_manager:
                data = self.vectorization_manager.optimize_dataframe(data)
                self.performance_stats['memory_optimizations'] += 1
            
            # Choose analysis method based on data size and available components
            if self.vectorization_manager and len(data) > 1000:
                characteristics.update(self._batch_analyze_characteristics(data))
            else:
                characteristics.update(self._individual_analyze_characteristics(data))
            
            return characteristics
        
        return safe_validate_and_execute(
            _validate_inputs, _analyze_characteristics, "data_characteristics_analysis"
        )
    
    def _batch_analyze_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Batch analyze characteristics using VectorBT optimizations."""
        if not self.vectorization_manager:
            raise AnalysisError("Vectorization manager not initialized for batch processing")
        
        characteristics = {}
        
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
        
        if not feature_configs:
            raise AnalysisError("No valid columns found for analysis")
        
        # Process all features in batch
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
        
        return characteristics
    
    def _individual_analyze_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Individual analysis fallback when batch processing is not available."""
        if not self.rolling_optimizer:
            raise AnalysisError("Rolling optimizer not initialized for individual analysis")
        
        characteristics = {}
        
        # Volatility analysis
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            characteristics['volatility'] = returns.std()
            
            # Use VectorBT-optimized volatility clustering
            vol_windows = [5, 10, 20, 50, 100]
            vol_clusters = []
            
            for window in vol_windows:
                if len(returns) > window * 2:
                    rolling_vol = self.rolling_optimizer.rolling_std(returns, window=window)
                    self.performance_stats['vectorbt_operations'] += 1
                    
                    vol_threshold = rolling_vol.quantile(0.8)
                    clusters = rolling_vol > vol_threshold
                    cluster_lengths = PeriodAnalysisUtils.find_pattern_periods(clusters)
                    
                    if cluster_lengths:
                        avg_cluster_length = np.mean(cluster_lengths)
                        vol_clusters.append(int(avg_cluster_length))
            
            characteristics['volatility_clusters'] = vol_clusters[:3]
        
        # Volume analysis
        if 'volume' in data.columns:
            vol_ma_5 = self.rolling_optimizer.rolling_mean(data['volume'], window=5)
            vol_ma_20 = self.rolling_optimizer.rolling_mean(data['volume'], window=20)
            self.performance_stats['vectorbt_operations'] += 2
            
            vol_spikes = data['volume'] > vol_ma_20 * 2
            spike_periods = PeriodAnalysisUtils.find_pattern_periods(vol_spikes)
            
            characteristics['volume_patterns'] = {
                'spike_periods': spike_periods,
                'volume_trend': []
            }
        
        # Price trend analysis
        if 'close' in data.columns:
            sma_5 = self.rolling_optimizer.rolling_mean(data['close'], window=5)
            sma_20 = self.rolling_optimizer.rolling_mean(data['close'], window=20)
            self.performance_stats['vectorbt_operations'] += 2
            
            # Find peaks and troughs in SMA
            peaks, _ = find_peaks(sma_20, distance=5)
            troughs, _ = find_peaks(-sma_20, distance=5)
            
            cycle_lengths = []
            all_extrema = sorted(list(peaks) + list(troughs))
            
            for i in range(1, len(all_extrema)):
                cycle_length = all_extrema[i] - all_extrema[i-1]
                cycle_lengths.append(cycle_length)
            
            if cycle_lengths:
                from collections import Counter
                most_common = Counter(cycle_lengths).most_common(3)
                characteristics['trend_cycles'] = [length for length, count in most_common]
            else:
                characteristics['trend_cycles'] = []
            
            characteristics['seasonality'] = self._detect_seasonality(data['close'])
        
        # Market regime analysis
        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            volatility = self.rolling_optimizer.rolling_std(returns, window=20)
            self.performance_stats['vectorbt_operations'] += 1
            
            vol_threshold = volatility.quantile(0.7)
            regime_changes = volatility > vol_threshold
            regime_lengths = PeriodAnalysisUtils.find_pattern_periods(regime_changes)
            characteristics['regime_changes'] = regime_lengths[:3]
        
        return characteristics
    
    def _detect_volatility_clusters_vectorbt(self, features: pd.DataFrame) -> List[int]:
        """Detect volatility clustering periods using VectorBT-optimized features."""
        if features.empty:
            return []
        
        vol_clusters = []
        volatility_cols = [col for col in features.columns if col.startswith('volatility_')]
        
        if not volatility_cols:
            return []
        
        # Use pre-computed volatility features
        for col in volatility_cols:
            vol_series = features[col].dropna()
            if len(vol_series) < 10:
                continue
            
            vol_threshold = vol_series.quantile(0.8)
            if pd.isna(vol_threshold):
                continue
            
            clusters = vol_series > vol_threshold
            cluster_lengths = PeriodAnalysisUtils.find_pattern_periods(clusters)
            
            if cluster_lengths:
                avg_length = np.mean(cluster_lengths)
                vol_clusters.append(int(avg_length))
        
        return vol_clusters[:3]
    
    def _detect_trend_cycles_vectorbt(self, features: pd.DataFrame) -> List[int]:
        """Detect trend cycles using VectorBT-optimized features."""
        if features.empty:
            return []
        
        cycle_lengths = []
        sma_cols = [col for col in features.columns if col.startswith('sma_')]
        
        if not sma_cols:
            return []
        
        for col in sma_cols:
            sma_series = features[col].dropna()
            if len(sma_series) < 20:
                continue
            
            try:
                peaks, _ = find_peaks(sma_series, distance=5)
                troughs, _ = find_peaks(-sma_series, distance=5)
                
                all_extrema = sorted(list(peaks) + list(troughs))
                for i in range(1, len(all_extrema)):
                    cycle_length = all_extrema[i] - all_extrema[i-1]
                    cycle_lengths.append(cycle_length)
            except Exception:
                continue
        
        if cycle_lengths:
            from collections import Counter
            most_common = Counter(cycle_lengths).most_common(3)
            return [length for length, count in most_common]
        
        return []
    
    def _analyze_volume_patterns_vectorbt(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns using VectorBT-optimized features."""
        volume_patterns = {'spike_periods': [], 'volume_trend': []}
        
        volume_sma_5 = features.get('volume_sma_5', pd.Series())
        volume_sma_20 = features.get('volume_sma_20', pd.Series())
        
        if not volume_sma_5.empty and not volume_sma_20.empty:
            vol_spikes = volume_sma_5 > volume_sma_20 * 1.5
            spike_periods = PeriodAnalysisUtils.find_pattern_periods(vol_spikes)
            volume_patterns['spike_periods'] = spike_periods
        
        volume_patterns['volume_trend'] = self._detect_trend_cycles_vectorbt(features)
        
        return volume_patterns
    
    def _detect_regime_changes_vectorbt(self, data: pd.DataFrame, features: Optional[pd.DataFrame] = None) -> List[int]:
        """Detect market regime changes using VectorBT-optimized features."""
        if 'close' not in data.columns:
            return []
        
        if len(data) < 50:
            return []
        
        try:
            # Use pre-computed volatility features if available
            if features is not None and 'volatility_20' in features.columns:
                volatility = features['volatility_20'].dropna()
            else:
                if not self.rolling_optimizer:
                    return []
                
                returns = data['close'].pct_change().dropna()
                if len(returns) == 0:
                    return []
                
                volatility = self.rolling_optimizer.rolling_std(returns, window=20)
            
            if volatility.empty or volatility.isna().all():
                return []
            
            vol_threshold = volatility.quantile(0.7)
            if pd.isna(vol_threshold):
                return []
            
            regime_changes = volatility > vol_threshold
            regime_lengths = PeriodAnalysisUtils.find_pattern_periods(regime_changes)
            
            return regime_lengths[:3]
        except Exception:
            return []
    
    def _detect_seasonality(self, series: pd.Series) -> List[int]:
        """Detect seasonal patterns."""
        if len(series) < 100:
            return []
        
        try:
            # Get timeframe from series index
            timeframe_minutes = PeriodAnalysisUtils.get_timeframe_minutes(pd.DataFrame(index=series.index))
            
            # Look for daily, weekly patterns
            daily_period = 24 * 60 // timeframe_minutes
            weekly_period = daily_period * 7
            
            seasonal_periods = []
            for period in [daily_period, weekly_period]:
                if 2 <= period <= 200:  # Reasonable range
                    seasonal_periods.append(period)
            
            return seasonal_periods
        except Exception:
            return []
    
    def detect_market_cycles(self, data: pd.DataFrame) -> List[int]:
        """
        Detect market cycles using spectral analysis.
        
        Args:
            data: Input data with 'close' column
            
        Returns:
            List of detected cycle periods
        """
        def _validate_inputs():
            PeriodAnalysisUtils.validate_dataframe(data, min_length=50, required_columns=['close'], operation_name="cycle_detection")
        
        def _detect_cycles():
            prices = data['close'].values
            returns = np.diff(np.log(prices))
            
            if len(returns) == 0:
                return []
            
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
                    if 2 <= period <= 200:  # Reasonable range
                        cycle_periods.append(period)
            
            return cycle_periods[:5]  # Limit to top 5 cycles
        
        return safe_validate_and_execute(
            _validate_inputs, _detect_cycles, "market_cycle_detection"
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add VectorBT component stats if available
        if self.rolling_optimizer:
            try:
                rolling_stats = self.rolling_optimizer.get_performance_stats()
                if rolling_stats:
                    stats.update(rolling_stats)
            except Exception:
                pass
        
        if self.vectorization_manager:
            try:
                vectorization_stats = self.vectorization_manager.get_performance_stats()
                if vectorization_stats:
                    stats.update(vectorization_stats)
            except Exception:
                pass
        
        # Calculate efficiency metrics
        if stats['total_operations'] > 0:
            stats['average_operation_time'] = stats['total_time'] / stats['total_operations']
            stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_operations']
            stats['batch_usage_rate'] = stats['batch_operations'] / stats['total_operations']
        else:
            stats['average_operation_time'] = 0
            stats['vectorbt_usage_rate'] = 0
            stats['batch_usage_rate'] = 0
        
        return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'batch_operations': 0,
            'memory_optimizations': 0,
            'total_time': 0.0
        }
        
        # Reset component stats
        if self.rolling_optimizer:
            try:
                self.rolling_optimizer.reset_stats()
            except Exception:
                pass
        
        if self.vectorization_manager:
            try:
                self.vectorization_manager.reset_stats()
            except Exception:
                pass