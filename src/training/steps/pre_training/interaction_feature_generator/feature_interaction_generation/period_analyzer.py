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
        """Initialize VectorBT optimization components with comprehensive error handling."""
        tprint_debug("🔍 Initializing VectorBT components...")
        
        if not self.enable_vectorbt:
            tprint_info("ℹ️ VectorBT optimizations disabled by configuration")
            self.rolling_optimizer = None
            self.vectorization_manager = None
            return
        
        if not VECTORBT_AVAILABLE:
            tprint_warning("⚠️ VectorBT not available - falling back to pandas operations")
            tprint_warning("📊 Install VectorBT with: pip install vectorbt")
            self.rolling_optimizer = None
            self.vectorization_manager = None
            return
        
        try:
            tprint_debug("🔧 Initializing VectorBT rolling optimizer...")
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=False,
                enable_parallel=self.enable_parallel,
                memory_efficient=self.memory_efficient,
                chunk_size=self.chunk_size
            )
            
            if self.rolling_optimizer is None:
                tprint_error("❌ VectorBT rolling optimizer initialization returned None")
                raise AnalysisError("VectorBT rolling optimizer initialization failed - returned None")
            
            tprint_debug("🔧 Initializing unified vectorization manager...")
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
            
            if self.vectorization_manager is None:
                tprint_error("❌ VectorBT vectorization manager initialization returned None")
                raise AnalysisError("VectorBT vectorization manager initialization failed - returned None")
            
            tprint_success("✅ VectorBT components initialized successfully")
            tprint_debug(f"📊 Rolling optimizer: {type(self.rolling_optimizer).__name__}")
            tprint_debug(f"📊 Vectorization manager: {type(self.vectorization_manager).__name__}")
            
        except ImportError as e:
            tprint_error(f"❌ VectorBT import failed: {e}")
            tprint_error("📊 VectorBT dependencies may be missing")
            raise AnalysisError(f"VectorBT import failed: {e}") from e
        except Exception as e:
            tprint_error(f"❌ VectorBT initialization failed: {e}")
            tprint_error(f"📊 Error type: {type(e).__name__}")
            tprint_error("📊 Falling back to pandas operations")
            self.rolling_optimizer = None
            self.vectorization_manager = None
            # Don't raise here - allow fallback to pandas
    
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
        tprint_debug("🔍 Detecting volatility clusters using VectorBT features...")
        
        if features.empty:
            tprint_warning("⚠️ Empty features DataFrame provided for volatility clustering")
            return []
        
        vol_clusters = []
        volatility_cols = [col for col in features.columns if col.startswith('volatility_')]
        
        if not volatility_cols:
            tprint_warning("⚠️ No volatility columns found in features for clustering analysis")
            return []
        
        tprint_debug(f"📊 Found {len(volatility_cols)} volatility columns: {volatility_cols}")
        
        # Use pre-computed volatility features
        for col in volatility_cols:
            tprint_debug(f"🔍 Processing volatility column: {col}")
            vol_series = features[col].dropna()
            
            if len(vol_series) < 10:
                tprint_warning(f"⚠️ Skipping {col}: insufficient data ({len(vol_series)} points)")
                continue
            
            vol_threshold = vol_series.quantile(0.8)
            if pd.isna(vol_threshold):
                tprint_warning(f"⚠️ Skipping {col}: invalid threshold (NaN)")
                continue
            
            clusters = vol_series > vol_threshold
            cluster_lengths = PeriodAnalysisUtils.find_pattern_periods(clusters)
            
            if cluster_lengths:
                avg_length = np.mean(cluster_lengths)
                vol_clusters.append(int(avg_length))
                tprint_debug(f"✅ Added volatility cluster: {int(avg_length)} from {col}")
            else:
                tprint_debug(f"ℹ️ No clusters found in {col}")
        
        result = vol_clusters[:3]
        tprint_debug(f"✅ Volatility clustering complete: {len(result)} clusters found")
        return result
    
    def _detect_trend_cycles_vectorbt(self, features: pd.DataFrame) -> List[int]:
        """Detect trend cycles using VectorBT-optimized features."""
        tprint_debug("🔍 Detecting trend cycles using VectorBT features...")
        
        if features.empty:
            tprint_warning("⚠️ Empty features DataFrame provided for trend cycle detection")
            return []
        
        cycle_lengths = []
        sma_cols = [col for col in features.columns if col.startswith('sma_')]
        
        if not sma_cols:
            tprint_warning("⚠️ No SMA columns found in features for trend cycle detection")
            return []
        
        tprint_debug(f"📊 Found {len(sma_cols)} SMA columns: {sma_cols}")
        
        for col in sma_cols:
            tprint_debug(f"🔍 Processing SMA column: {col}")
            sma_series = features[col].dropna()
            
            if len(sma_series) < 20:
                tprint_warning(f"⚠️ Skipping {col}: insufficient data ({len(sma_series)} points)")
                continue
            
            try:
                peaks, _ = find_peaks(sma_series, distance=5)
                troughs, _ = find_peaks(-sma_series, distance=5)
                
                tprint_debug(f"📊 Found {len(peaks)} peaks and {len(troughs)} troughs in {col}")
                
                all_extrema = sorted(list(peaks) + list(troughs))
                for i in range(1, len(all_extrema)):
                    cycle_length = all_extrema[i] - all_extrema[i-1]
                    cycle_lengths.append(cycle_length)
                    tprint_debug(f"✅ Added cycle length: {cycle_length} from {col}")
                    
            except Exception as e:
                tprint_warning(f"⚠️ Peak detection failed for {col}: {e}")
                continue
        
        if cycle_lengths:
            from collections import Counter
            most_common = Counter(cycle_lengths).most_common(3)
            result = [length for length, count in most_common]
            tprint_debug(f"✅ Trend cycle detection complete: {len(result)} cycles found")
            return result
        
        tprint_warning("⚠️ No trend cycles detected")
        return []
    
    def _analyze_volume_patterns_vectorbt(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns using VectorBT-optimized features."""
        tprint_debug("🔍 Analyzing volume patterns using VectorBT features...")
        
        if features.empty:
            tprint_warning("⚠️ Empty features DataFrame provided for volume pattern analysis")
            return {'spike_periods': [], 'volume_trend': []}
        
        volume_patterns = {'spike_periods': [], 'volume_trend': []}
        
        volume_sma_5 = features.get('volume_sma_5', pd.Series())
        volume_sma_20 = features.get('volume_sma_20', pd.Series())
        
        tprint_debug(f"📊 Volume SMA columns available: 5-period={not volume_sma_5.empty}, 20-period={not volume_sma_20.empty}")
        
        if not volume_sma_5.empty and not volume_sma_20.empty:
            tprint_debug("🔍 Analyzing volume spikes...")
            vol_spikes = volume_sma_5 > volume_sma_20 * 1.5
            spike_periods = PeriodAnalysisUtils.find_pattern_periods(vol_spikes)
            volume_patterns['spike_periods'] = spike_periods
            tprint_debug(f"✅ Found {len(spike_periods)} volume spike periods")
        else:
            tprint_warning("⚠️ Volume SMA columns not available for spike analysis")
        
        tprint_debug("🔍 Analyzing volume trends...")
        volume_patterns['volume_trend'] = self._detect_trend_cycles_vectorbt(features)
        
        tprint_debug(f"✅ Volume pattern analysis complete: {volume_patterns}")
        return volume_patterns
    
    def _detect_regime_changes_vectorbt(self, data: pd.DataFrame, features: Optional[pd.DataFrame] = None) -> List[int]:
        """Detect market regime changes using VectorBT-optimized features."""
        tprint_debug("🔍 Detecting market regime changes using VectorBT...")
        
        if 'close' not in data.columns:
            tprint_warning("⚠️ Missing 'close' column for regime detection")
            return []
        
        if len(data) < 50:
            tprint_warning(f"⚠️ Insufficient data for regime detection: {len(data)} < 50 required")
            return []
        
        try:
            # Use pre-computed volatility features if available
            if features is not None and 'volatility_20' in features.columns:
                tprint_debug("📊 Using pre-computed volatility features for regime detection")
                volatility = features['volatility_20'].dropna()
            else:
                tprint_debug("📊 Calculating volatility manually for regime detection")
                if not self.rolling_optimizer:
                    tprint_warning("⚠️ Rolling optimizer not available for regime detection")
                    return []
                
                returns = data['close'].pct_change().dropna()
                if len(returns) == 0:
                    tprint_warning("⚠️ No valid returns calculated for regime detection")
                    return []
                
                volatility = self.rolling_optimizer.rolling_std(returns, window=20)
            
            if volatility.empty or volatility.isna().all():
                tprint_warning("⚠️ Invalid volatility data for regime detection")
                return []
            
            vol_threshold = volatility.quantile(0.7)
            if pd.isna(vol_threshold):
                tprint_warning("⚠️ Invalid volatility threshold for regime detection")
                return []
            
            tprint_debug(f"📊 Volatility threshold: {vol_threshold:.6f}")
            regime_changes = volatility > vol_threshold
            regime_lengths = PeriodAnalysisUtils.find_pattern_periods(regime_changes)
            
            result = regime_lengths[:3]
            tprint_debug(f"✅ Regime detection complete: {len(result)} regime periods found")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Regime detection failed: {e}")
            tprint_error(f"📊 Error type: {type(e).__name__}")
            return []
    
    def _detect_seasonality(self, series: pd.Series) -> List[int]:
        """Detect seasonal patterns."""
        tprint_debug("🔍 Detecting seasonal patterns...")
        
        if len(series) < 100:
            tprint_warning(f"⚠️ Insufficient data for seasonality detection: {len(series)} < 100 required")
            return []
        
        try:
            # Get timeframe from series index
            timeframe_minutes = PeriodAnalysisUtils.get_timeframe_minutes(pd.DataFrame(index=series.index))
            tprint_debug(f"📊 Detected timeframe: {timeframe_minutes} minutes")
            
            # Look for daily, weekly patterns
            daily_period = 24 * 60 // timeframe_minutes
            weekly_period = daily_period * 7
            
            tprint_debug(f"📊 Calculated periods - daily: {daily_period}, weekly: {weekly_period}")
            
            seasonal_periods = []
            for period in [daily_period, weekly_period]:
                if 2 <= period <= 200:  # Reasonable range
                    seasonal_periods.append(period)
                    tprint_debug(f"✅ Added seasonal period: {period}")
                else:
                    tprint_debug(f"⚠️ Skipped period {period}: outside reasonable range [2, 200]")
            
            tprint_debug(f"✅ Seasonality detection complete: {len(seasonal_periods)} periods found")
            return seasonal_periods
            
        except Exception as e:
            tprint_error(f"❌ Seasonality detection failed: {e}")
            tprint_error(f"📊 Error type: {type(e).__name__}")
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
                    tprint_debug("✅ Added VectorBT rolling optimizer stats")
                else:
                    tprint_warning("⚠️ VectorBT rolling optimizer returned empty stats")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to get VectorBT rolling optimizer stats: {e}")
        
        if self.vectorization_manager:
            try:
                vectorization_stats = self.vectorization_manager.get_performance_stats()
                if vectorization_stats:
                    stats.update(vectorization_stats)
                    tprint_debug("✅ Added VectorBT vectorization manager stats")
                else:
                    tprint_warning("⚠️ VectorBT vectorization manager returned empty stats")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to get VectorBT vectorization manager stats: {e}")
        
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
                tprint_debug("✅ VectorBT rolling optimizer stats reset")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to reset VectorBT rolling optimizer stats: {e}")
        
        if self.vectorization_manager:
            try:
                self.vectorization_manager.reset_stats()
                tprint_debug("✅ VectorBT vectorization manager stats reset")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to reset VectorBT vectorization manager stats: {e}")