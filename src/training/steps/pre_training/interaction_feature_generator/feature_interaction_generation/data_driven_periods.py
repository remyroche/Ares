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
from scipy.signal import find_peaks
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
        """Initialize VectorBT optimization components with fast fail."""
        tprint_debug("🔍 Initializing VectorBT components...")
        
        if not self.enable_vectorbt:
            tprint_info("ℹ️ VectorBT optimizations disabled")
            self.rolling_optimizer = None
            self.vectorization_manager = None
            return
        
        try:
            # Initialize VectorBT rolling optimizer
            tprint_debug("📊 Initializing VectorBT rolling optimizer...")
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=False,  # Can be enabled if needed
                enable_parallel=self.enable_parallel,
                memory_efficient=self.memory_efficient,
                chunk_size=self.chunk_size
            )
            
            if self.rolling_optimizer is None:
                tprint_error("❌ VectorBT rolling optimizer initialization returned None")
                raise RuntimeError("VectorBT rolling optimizer initialization failed")
            
            # Initialize unified vectorization manager
            tprint_debug("📊 Initializing unified vectorization manager...")
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
                tprint_error("❌ Unified vectorization manager initialization returned None")
                raise RuntimeError("Unified vectorization manager initialization failed")
            
            tprint_success("✅ VectorBT components initialized successfully")
            
        except Exception as e:
            tprint_error(f"❌ VectorBT initialization failed: {e}")
            # Fast fail - don't continue with degraded functionality
            raise RuntimeError(f"VectorBT initialization failed: {e}")
    
    def analyze_data_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data characteristics to inform period selection using VectorBT optimizations."""
        tprint_info("🔍 Starting data characteristics analysis...")
        start_time = time.time()
        self.performance_stats['total_operations'] += 1
        
        # Fast fail for invalid inputs
        if not isinstance(data, pd.DataFrame):
            tprint_error("❌ Invalid input: expected pandas DataFrame")
            raise ValueError("Expected pandas DataFrame, got {type(data).__name__}")
        
        if len(data) == 0:
            tprint_error("❌ Empty DataFrame provided")
            raise ValueError("DataFrame cannot be empty")
        
        # Check cache first
        if self._cache_enabled:
            tprint_debug("🔍 Checking cache for existing results...")
            cache_key = self._generate_cache_key('analyze_characteristics', data)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                self.performance_stats['cache_hits'] += 1
                tprint_success("✅ Cache hit - returning cached results")
                return cached_result
            self.performance_stats['cache_misses'] += 1
            tprint_debug("❌ Cache miss - proceeding with analysis")
        
        characteristics = {}
        
        try:
            # Basic data info
            tprint_debug("📊 Extracting basic data information...")
            characteristics['data_length'] = len(data)
            characteristics['data_frequency'] = self._detect_frequency(data)
            characteristics['timeframe_minutes'] = self._get_timeframe_minutes(data)
            
            tprint_debug(f"📊 Data info: length={characteristics['data_length']}, freq={characteristics['data_frequency']}, timeframe={characteristics['timeframe_minutes']}min")
            
            # Optimize data for processing if memory efficient mode is enabled
            if self.memory_efficient and self.vectorization_manager:
                tprint_debug("📊 Optimizing data for memory efficiency...")
                data = self.vectorization_manager.optimize_dataframe(data)
                self.performance_stats['memory_optimizations'] += 1
                tprint_debug(f"✅ Data optimized: {data.shape}")
            
            # Batch process multiple analyses using VectorBT
            if self.vectorization_manager and len(data) > 1000:
                tprint_debug("📊 Using batch processing (data size > 1000)...")
                characteristics.update(self._batch_analyze_characteristics(data))
            else:
                tprint_debug("📊 Using individual analysis (data size <= 1000 or no vectorization manager)...")
                characteristics.update(self._individual_analyze_characteristics(data))
            
            # Cache result
            if self._cache_enabled:
                tprint_debug("📊 Caching analysis results...")
                self._put_in_cache(cache_key, characteristics)
            
            self.performance_stats['total_time'] += time.time() - start_time
            tprint_success(f"✅ Data characteristics analysis completed in {time.time() - start_time:.3f}s")
            return characteristics
            
        except Exception as e:
            tprint_error(f"❌ Data characteristics analysis failed: {e}")
            raise RuntimeError(f"Data characteristics analysis failed: {e}")
    
    def _batch_analyze_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Batch analyze characteristics using VectorBT optimizations with fast fail."""
        tprint_debug("🔍 Starting batch analysis of characteristics...")
        
        # Fast fail for invalid inputs
        if not isinstance(data, pd.DataFrame):
            tprint_error("❌ Invalid input: expected pandas DataFrame")
            raise ValueError("Expected pandas DataFrame, got {type(data).__name__}")
        
        if len(data) == 0:
            tprint_error("❌ Empty DataFrame provided")
            raise ValueError("DataFrame cannot be empty")
        
        if self.vectorization_manager is None:
            tprint_error("❌ Vectorization manager not initialized")
            raise RuntimeError("Vectorization manager must be initialized for batch processing")
        
        characteristics = {}
        
        try:
            # Prepare feature configurations for batch processing
            feature_configs = []
            
            if 'close' in data.columns:
                tprint_debug("📊 Adding close price analysis features...")
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
                tprint_debug("📊 Adding volume analysis features...")
                # Volume analysis
                feature_configs.extend([
                    {'name': 'volume_sma_5', 'type': 'rolling', 'params': {'operation': 'mean', 'window': 5, 'column': 'volume'}},
                    {'name': 'volume_sma_20', 'type': 'rolling', 'params': {'operation': 'mean', 'window': 20, 'column': 'volume'}},
                ])
            
            if not feature_configs:
                tprint_error("❌ No valid columns found for analysis")
                raise ValueError("DataFrame must contain 'close' or 'volume' columns")
            
            # Process all features in batch
            tprint_debug(f"📊 Processing {len(feature_configs)} features in batch...")
            features = self.vectorization_manager.batch_process_features(data, feature_configs)
            
            if features is None or features.empty:
                tprint_error("❌ Batch processing returned empty results")
                raise RuntimeError("Batch processing failed to produce results")
            
            self.performance_stats['batch_operations'] += 1
            tprint_debug(f"✅ Batch processing completed: {features.shape}")
            
            # Extract characteristics from batch results
            if 'close' in data.columns:
                tprint_debug("📊 Extracting volatility and trend characteristics...")
                returns = data['close'].pct_change().dropna()
                if len(returns) == 0:
                    tprint_error("❌ No valid returns calculated")
                    raise ValueError("Failed to calculate returns from close prices")
                
                characteristics['volatility'] = returns.std()
                characteristics['volatility_clusters'] = self._detect_volatility_clusters_vectorbt(features)
                characteristics['trend_cycles'] = self._detect_trend_cycles_vectorbt(features)
            
            if 'volume' in data.columns:
                tprint_debug("📊 Extracting volume characteristics...")
                characteristics['volume_patterns'] = self._analyze_volume_patterns_vectorbt(features)
            
            # Market regime analysis
            tprint_debug("📊 Analyzing market regime changes...")
            characteristics['regime_changes'] = self._detect_regime_changes_vectorbt(data, features if 'close' in data.columns else None)
            
            tprint_success("✅ Batch analysis completed successfully")
            
        except Exception as e:
            tprint_error(f"❌ Batch analysis failed: {e}")
            # Fast fail instead of fallback
            raise RuntimeError(f"Batch analysis failed: {e}")
        
        return characteristics
    
    def _individual_analyze_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Individual analysis fallback when batch processing is not available."""
        tprint_debug("🔍 Starting individual analysis of characteristics...")
        
        # Fast fail for invalid inputs
        if not isinstance(data, pd.DataFrame):
            tprint_error("❌ Invalid input: expected pandas DataFrame")
            raise ValueError("Expected pandas DataFrame, got {type(data).__name__}")
        
        if len(data) == 0:
            tprint_error("❌ Empty DataFrame provided")
            raise ValueError("DataFrame cannot be empty")
        
        if self.rolling_optimizer is None:
            tprint_error("❌ Rolling optimizer not initialized")
            raise RuntimeError("Rolling optimizer must be initialized for individual analysis")
        
        characteristics = {}
        
        try:
            # Volatility analysis
            if 'close' in data.columns:
                tprint_debug("📊 Analyzing volatility characteristics...")
                returns = data['close'].pct_change().dropna()
                if len(returns) == 0:
                    tprint_error("❌ No valid returns calculated")
                    raise ValueError("Failed to calculate returns from close prices")
                
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
                        cluster_lengths = self._find_pattern_periods(clusters)
                        
                        if cluster_lengths:
                            avg_cluster_length = np.mean(cluster_lengths)
                            if self.min_period <= avg_cluster_length <= self.max_period:
                                vol_clusters.append(int(avg_cluster_length))
                                tprint_debug(f"✅ Added volatility cluster: {int(avg_cluster_length)} (window: {window})")
                
                characteristics['volatility_clusters'] = vol_clusters[:3]
                tprint_debug(f"✅ Volatility analysis complete: {len(vol_clusters)} clusters")
            
            # Volume analysis
            if 'volume' in data.columns:
                tprint_debug("📊 Analyzing volume characteristics...")
                # Use VectorBT-optimized volume analysis
                vol_ma_5 = self.rolling_optimizer.rolling_mean(data['volume'], window=5)
                vol_ma_20 = self.rolling_optimizer.rolling_mean(data['volume'], window=20)
                self.performance_stats['vectorbt_operations'] += 2
                
                vol_spikes = data['volume'] > vol_ma_20 * 2
                spike_periods = self._find_pattern_periods(vol_spikes)
                
                characteristics['volume_patterns'] = {
                    'spike_periods': spike_periods,
                    'volume_trend': []  # Simplified for individual analysis
                }
                tprint_debug(f"✅ Volume analysis complete: {len(spike_periods)} spike periods")
            
            # Price trend analysis
            if 'close' in data.columns:
                tprint_debug("📊 Analyzing trend characteristics...")
                # Use VectorBT-optimized trend cycle detection
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
                    if self.min_period <= cycle_length <= self.max_period:
                        cycle_lengths.append(cycle_length)
                
                if cycle_lengths:
                    from collections import Counter
                    most_common = Counter(cycle_lengths).most_common(3)
                    characteristics['trend_cycles'] = [length for length, count in most_common]
                    tprint_debug(f"✅ Found {len(characteristics['trend_cycles'])} trend cycles")
                else:
                    characteristics['trend_cycles'] = []
                    tprint_debug("⚠️ No valid trend cycles found")
                
                characteristics['seasonality'] = self._detect_seasonality(data['close'])
            
            # Market regime analysis
            if 'close' in data.columns:
                tprint_debug("📊 Analyzing market regime characteristics...")
                returns = data['close'].pct_change().dropna()
                volatility = self.rolling_optimizer.rolling_std(returns, window=20)
                self.performance_stats['vectorbt_operations'] += 1
                
                vol_threshold = volatility.quantile(0.7)
                regime_changes = volatility > vol_threshold
                regime_lengths = self._find_pattern_periods(regime_changes)
                characteristics['regime_changes'] = regime_lengths[:3]
                tprint_debug(f"✅ Found {len(regime_lengths)} regime change periods")
            
            tprint_success("✅ Individual analysis completed successfully")
            
        except Exception as e:
            tprint_error(f"❌ Individual analysis failed: {e}")
            raise RuntimeError(f"Individual analysis failed: {e}")
        
        return characteristics
    
    def _detect_volatility_clusters_vectorbt(self, features: pd.DataFrame) -> List[int]:
        """Detect volatility clustering periods using VectorBT-optimized features."""
        tprint_debug("🔍 Detecting volatility clusters using VectorBT features...")
        
        # Fast fail for invalid inputs
        if not isinstance(features, pd.DataFrame):
            tprint_error("❌ Invalid input: expected pandas DataFrame")
            raise ValueError("Expected pandas DataFrame, got {type(features).__name__}")
        
        if features.empty:
            tprint_error("❌ Empty features DataFrame")
            raise ValueError("Features DataFrame cannot be empty")
        
        try:
            vol_clusters = []
            volatility_cols = [col for col in features.columns if col.startswith('volatility_')]
            
            if not volatility_cols:
                tprint_warning("⚠️ No volatility columns found in features")
                return []
            
            tprint_debug(f"📊 Processing {len(volatility_cols)} volatility columns...")
            
            # Use pre-computed volatility features
            for col in volatility_cols:
                vol_series = features[col].dropna()
                if len(vol_series) < 10:
                    tprint_debug(f"⚠️ Skipping {col}: insufficient data ({len(vol_series)} points)")
                    continue
                
                # Find volatility clusters using pre-computed rolling volatility
                vol_threshold = vol_series.quantile(0.8)
                if pd.isna(vol_threshold):
                    tprint_debug(f"⚠️ Skipping {col}: invalid threshold")
                    continue
                
                clusters = vol_series > vol_threshold
                
                # Calculate average cluster length
                cluster_lengths = self._find_pattern_periods(clusters)
                if cluster_lengths:
                    avg_length = np.mean(cluster_lengths)
                    if self.min_period <= avg_length <= self.max_period:
                        vol_clusters.append(int(avg_length))
                        tprint_debug(f"✅ Added volatility cluster: {int(avg_length)} from {col}")
            
            result = vol_clusters[:3]  # Limit to top 3
            tprint_debug(f"✅ Detected {len(result)} volatility clusters: {result}")
            return result
            
        except Exception as e:
            tprint_error(f"❌ VectorBT volatility clustering failed: {e}")
            raise RuntimeError(f"Volatility clustering failed: {e}")
    
    def _detect_trend_cycles_vectorbt(self, features: pd.DataFrame) -> List[int]:
        """Detect trend cycles using VectorBT-optimized features."""
        tprint_debug("🔍 Detecting trend cycles using VectorBT features...")
        
        # Fast fail for invalid inputs
        if not isinstance(features, pd.DataFrame):
            tprint_error("❌ Invalid input: expected pandas DataFrame")
            raise ValueError("Expected pandas DataFrame, got {type(features).__name__}")
        
        if features.empty:
            tprint_error("❌ Empty features DataFrame")
            raise ValueError("Features DataFrame cannot be empty")
        
        try:
            cycle_lengths = []
            
            # Use pre-computed SMA features to detect cycles
            sma_cols = [col for col in features.columns if col.startswith('sma_')]
            
            if not sma_cols:
                tprint_warning("⚠️ No SMA columns found in features")
                return []
            
            tprint_debug(f"📊 Processing {len(sma_cols)} SMA columns...")
            
            for col in sma_cols:
                sma_series = features[col].dropna()
                if len(sma_series) < 20:
                    tprint_debug(f"⚠️ Skipping {col}: insufficient data ({len(sma_series)} points)")
                    continue
                
                # Find peaks and troughs in SMA
                try:
                    peaks, _ = find_peaks(sma_series, distance=5)
                    troughs, _ = find_peaks(-sma_series, distance=5)
                    
                    # Calculate cycle lengths
                    all_extrema = sorted(list(peaks) + list(troughs))
                    for i in range(1, len(all_extrema)):
                        cycle_length = all_extrema[i] - all_extrema[i-1]
                        if self.min_period <= cycle_length <= self.max_period:
                            cycle_lengths.append(cycle_length)
                            tprint_debug(f"✅ Added cycle length: {cycle_length} from {col}")
                except Exception as e:
                    tprint_debug(f"⚠️ Peak detection failed for {col}: {e}")
                    continue
            
            # Return most common cycle lengths
            if cycle_lengths:
                from collections import Counter
                most_common = Counter(cycle_lengths).most_common(3)
                result = [length for length, count in most_common]
                tprint_debug(f"✅ Detected {len(result)} trend cycles: {result}")
                return result
            
            tprint_debug("⚠️ No valid trend cycles detected")
            return []
            
        except Exception as e:
            tprint_error(f"❌ VectorBT trend cycle detection failed: {e}")
            raise RuntimeError(f"Trend cycle detection failed: {e}")
    
    def _analyze_volume_patterns_vectorbt(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns using VectorBT-optimized features."""
        tprint_debug("🔍 Analyzing volume patterns using VectorBT features...")
        
        # Fast fail for invalid inputs
        if not isinstance(features, pd.DataFrame):
            tprint_error("❌ Invalid input: expected pandas DataFrame")
            raise ValueError("Expected pandas DataFrame, got {type(features).__name__}")
        
        if features.empty:
            tprint_error("❌ Empty features DataFrame")
            raise ValueError("Features DataFrame cannot be empty")
        
        try:
            volume_patterns = {}
            
            # Use pre-computed volume features
            volume_cols = [col for col in features.columns if col.startswith('volume_')]
            
            if not volume_cols:
                tprint_warning("⚠️ No volume columns found in features")
                return {'spike_periods': [], 'volume_trend': []}
            
            tprint_debug(f"📊 Processing {len(volume_cols)} volume columns...")
            
            # Find volume spikes using pre-computed moving averages
            volume_sma_5 = features.get('volume_sma_5', pd.Series())
            volume_sma_20 = features.get('volume_sma_20', pd.Series())
            
            if not volume_sma_5.empty and not volume_sma_20.empty:
                tprint_debug("📊 Analyzing volume spikes...")
                # Find volume spikes
                vol_spikes = volume_sma_5 > volume_sma_20 * 1.5
                spike_periods = self._find_pattern_periods(vol_spikes)
                volume_patterns['spike_periods'] = spike_periods
                tprint_debug(f"✅ Found {len(spike_periods)} volume spike periods")
            else:
                tprint_debug("⚠️ Volume SMA columns not available")
                volume_patterns['spike_periods'] = []
            
            # Volume trend analysis
            tprint_debug("📊 Analyzing volume trends...")
            volume_patterns['volume_trend'] = self._detect_trend_cycles_vectorbt(features)
            
            tprint_debug(f"✅ Volume pattern analysis complete: {volume_patterns}")
            return volume_patterns
            
        except Exception as e:
            tprint_error(f"❌ VectorBT volume analysis failed: {e}")
            raise RuntimeError(f"Volume pattern analysis failed: {e}")
    
    def _detect_regime_changes_vectorbt(self, data: pd.DataFrame, features: Optional[pd.DataFrame] = None) -> List[int]:
        """Detect market regime changes using VectorBT-optimized features."""
        tprint_debug("🔍 Detecting market regime changes using VectorBT...")
        
        # Fast fail for invalid inputs
        if not isinstance(data, pd.DataFrame):
            tprint_error("❌ Invalid input: expected pandas DataFrame")
            raise ValueError("Expected pandas DataFrame, got {type(data).__name__}")
        
        if 'close' not in data.columns:
            tprint_error("❌ Missing 'close' column for regime detection")
            raise ValueError("DataFrame must contain 'close' column for regime detection")
        
        if len(data) < 50:
            tprint_error("❌ Insufficient data for regime detection (need >= 50 points)")
            raise ValueError("Need at least 50 data points for regime detection")
        
        try:
            # Use pre-computed volatility features if available
            if features is not None and 'volatility_20' in features.columns:
                tprint_debug("📊 Using pre-computed volatility features...")
                volatility = features['volatility_20'].dropna()
            else:
                tprint_debug("📊 Calculating volatility manually...")
                # Fallback to manual calculation
                returns = data['close'].pct_change().dropna()
                if len(returns) == 0:
                    tprint_error("❌ No valid returns calculated")
                    raise ValueError("Failed to calculate returns from close prices")
                
                if self.rolling_optimizer:
                    volatility = self.rolling_optimizer.rolling_std(returns, window=20)
                else:
                    tprint_error("❌ Rolling optimizer not available")
                    raise RuntimeError("Rolling optimizer required for regime detection")
            
            if volatility.empty or volatility.isna().all():
                tprint_error("❌ Invalid volatility data")
                raise ValueError("Volatility calculation produced invalid results")
            
            # Find regime changes (high vol periods)
            vol_threshold = volatility.quantile(0.7)
            if pd.isna(vol_threshold):
                tprint_error("❌ Invalid volatility threshold")
                raise ValueError("Failed to calculate volatility threshold")
            
            regime_changes = volatility > vol_threshold
            
            # Calculate regime lengths
            regime_lengths = self._find_pattern_periods(regime_changes)
            
            result = regime_lengths[:3]  # Limit to top 3
            tprint_debug(f"✅ Detected {len(result)} regime change periods: {result}")
            return result
            
        except Exception as e:
            tprint_error(f"❌ VectorBT regime detection failed: {e}")
            raise RuntimeError(f"Regime detection failed: {e}")
    
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
        
        # Check if we have enough data - fast fail
        if characteristics['data_length'] < self.min_data_points:
            tprint_error(f"❌ Insufficient data ({characteristics['data_length']} < {self.min_data_points})")
            raise ValueError(f"Insufficient data: {characteristics['data_length']} < {self.min_data_points} required")
        
        # Get base periods from timeframe
        base_periods = self._get_base_periods_from_timeframe(
            characteristics.get('timeframe_minutes', 15),
            target_timeframe
        )
        
        # Analyze market cycles
        cycle_periods = self._detect_market_cycles(data, characteristics)
        
        # Extract periods from characteristics (already computed by VectorBT-optimized methods)
        volatility_periods = characteristics.get('volatility_clusters', [])
        volume_patterns = characteristics.get('volume_patterns', {})
        volume_periods = volume_patterns.get('spike_periods', [])
        
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
        """Detect the frequency of the data with fast fail."""
        tprint_debug("🔍 Detecting data frequency...")
        
        # Fast fail for invalid inputs
        if not isinstance(data, pd.DataFrame):
            tprint_error("❌ Invalid input: expected pandas DataFrame")
            raise ValueError("Expected pandas DataFrame, got {type(data).__name__}")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            tprint_error("❌ Non-datetime index, cannot determine frequency")
            raise ValueError("DataFrame must have DatetimeIndex to determine frequency")
        
        if len(data) < 2:
            tprint_error("❌ Insufficient data for frequency detection")
            raise ValueError("Need at least 2 data points to determine frequency")
        
        try:
            # Calculate time differences
            time_diffs = data.index.to_series().diff().dropna()
            median_diff = time_diffs.median()
            
            if pd.isna(median_diff):
                tprint_error("❌ Invalid time differences")
                raise ValueError("Failed to calculate time differences")
            
            tprint_debug(f"📊 Median time difference: {median_diff}")
            
            # Convert to minutes
            if median_diff < pd.Timedelta(minutes=1):
                frequency = 'sub-minute'
            elif median_diff < pd.Timedelta(minutes=5):
                frequency = '1m'
            elif median_diff < pd.Timedelta(minutes=10):
                frequency = '5m'
            elif median_diff < pd.Timedelta(minutes=20):
                frequency = '15m'
            elif median_diff < pd.Timedelta(minutes=90):
                frequency = '60m'
            elif median_diff < pd.Timedelta(hours=2):
                frequency = '4h'
            elif median_diff < pd.Timedelta(hours=6):
                frequency = '1d'
            else:
                frequency = 'weekly'
            
            tprint_debug(f"✅ Detected frequency: {frequency}")
            return frequency
            
        except Exception as e:
            tprint_error(f"❌ Frequency detection failed: {e}")
            raise RuntimeError(f"Frequency detection failed: {e}")
    
    def _get_timeframe_minutes(self, data: pd.DataFrame) -> int:
        """Get timeframe in minutes with fast fail."""
        tprint_debug("🔍 Getting timeframe from DataFrame...")
        
        # Fast fail for invalid inputs
        if not isinstance(data, pd.DataFrame):
            tprint_error("❌ Invalid input: expected pandas DataFrame")
            raise ValueError("Expected pandas DataFrame, got {type(data).__name__}")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            tprint_error("❌ Non-datetime index, cannot determine timeframe")
            raise ValueError("DataFrame must have DatetimeIndex to determine timeframe")
        
        if len(data) < 2:
            tprint_error("❌ Insufficient data points for timeframe detection")
            raise ValueError("Need at least 2 data points to determine timeframe")
        
        try:
            time_diffs = data.index.to_series().diff().dropna()
            median_diff = time_diffs.median()
            timeframe_minutes = int(median_diff.total_seconds() / 60)
            
            if timeframe_minutes <= 0:
                tprint_error("❌ Invalid timeframe detected: {timeframe_minutes} minutes")
                raise ValueError(f"Invalid timeframe: {timeframe_minutes} minutes")
            
            tprint_debug(f"✅ Detected timeframe: {timeframe_minutes} minutes")
            return timeframe_minutes
            
        except Exception as e:
            tprint_error(f"❌ Timeframe detection failed: {e}")
            raise ValueError(f"Failed to detect timeframe: {e}")
    
    def _get_base_periods_from_timeframe(self, timeframe_minutes: int, 
                                       target_timeframe: Optional[str] = None) -> List[int]:
        """Get base periods based on timeframe."""
        tprint_debug(f"🔍 Getting base periods for timeframe: {timeframe_minutes}min, target: {target_timeframe}")
        
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
                tprint_debug("⚠️ Unknown target timeframe format, using default 15 minutes")
        else:
            target_minutes = timeframe_minutes
        
        tprint_debug(f"📊 Target minutes: {target_minutes}, Current timeframe: {timeframe_minutes}")
        
        # Calculate periods based on target timeframe
        # Use multiples that make sense for the target timeframe
        base_periods = []
        
        # Short-term periods (2-10x current timeframe)
        for multiplier in [2, 3, 5, 10]:
            period = multiplier * (target_minutes // timeframe_minutes)
            if self.min_period <= period <= self.max_period:
                base_periods.append(period)
                tprint_debug(f"✅ Added short-term period: {period} (multiplier: {multiplier})")
        
        # Medium-term periods (20-50x current timeframe)
        for multiplier in [20, 30, 50]:
            period = multiplier * (target_minutes // timeframe_minutes)
            if self.min_period <= period <= self.max_period:
                base_periods.append(period)
                tprint_debug(f"✅ Added medium-term period: {period} (multiplier: {multiplier})")
        
        # Long-term periods (100x+ current timeframe)
        for multiplier in [100, 200]:
            period = multiplier * (target_minutes // timeframe_minutes)
            if self.min_period <= period <= self.max_period:
                base_periods.append(period)
                tprint_debug(f"✅ Added long-term period: {period} (multiplier: {multiplier})")
        
        tprint_debug(f"✅ Generated {len(base_periods)} base periods: {base_periods}")
        return base_periods
    
    def _detect_market_cycles(self, data: pd.DataFrame, 
                            characteristics: Dict[str, Any]) -> List[int]:
        """Detect market cycles using spectral analysis with fast fail."""
        tprint_debug("🔍 Detecting market cycles using spectral analysis...")
        
        # Fast fail for invalid inputs
        if not isinstance(data, pd.DataFrame):
            tprint_error("❌ Invalid input: expected pandas DataFrame")
            raise ValueError("Expected pandas DataFrame, got {type(data).__name__}")
        
        if 'close' not in data.columns:
            tprint_error("❌ Missing 'close' column for cycle detection")
            raise ValueError("DataFrame must contain 'close' column for cycle detection")
        
        if len(data) < 50:
            tprint_error("❌ Insufficient data for cycle detection (need >= 50 points)")
            raise ValueError("Need at least 50 data points for cycle detection")
        
        try:
            prices = data['close'].values
            if len(prices) == 0:
                tprint_error("❌ Empty price data")
                raise ValueError("Price data cannot be empty")
            
            returns = np.diff(np.log(prices))
            if len(returns) == 0:
                tprint_error("❌ No valid returns calculated")
                raise ValueError("Failed to calculate returns from prices")
            
            tprint_debug(f"📊 Analyzing {len(returns)} return observations")
            
            # Use FFT to detect cycles
            fft = np.fft.fft(returns)
            freqs = np.fft.fftfreq(len(returns))
            
            # Find significant frequencies
            power_spectrum = np.abs(fft) ** 2
            significant_freqs = freqs[power_spectrum > np.percentile(power_spectrum, 90)]
            
            tprint_debug(f"📊 Found {len(significant_freqs)} significant frequencies")
            
            # Convert frequencies to periods
            cycle_periods = []
            for freq in significant_freqs:
                if freq > 0:  # Only positive frequencies
                    period = int(1 / freq)
                    if self.min_period <= period <= self.max_period:
                        cycle_periods.append(period)
                        tprint_debug(f"✅ Added cycle period: {period} (frequency: {freq:.4f})")
            
            # Limit to top 5 cycles
            cycle_periods = cycle_periods[:5]
            tprint_debug(f"✅ Detected {len(cycle_periods)} market cycles: {cycle_periods}")
            return cycle_periods
            
        except Exception as e:
            tprint_error(f"❌ Cycle detection failed: {e}")
            raise RuntimeError(f"Cycle detection failed: {e}")
    
    
    
    
    def _detect_seasonality(self, series: pd.Series) -> List[int]:
        """Detect seasonal patterns with fast fail."""
        tprint_debug("🔍 Detecting seasonal patterns...")
        
        # Fast fail for invalid inputs
        if not isinstance(series, pd.Series):
            tprint_error("❌ Invalid input: expected pandas Series")
            raise ValueError("Expected pandas Series, got {type(series).__name__}")
        
        if len(series) < 100:
            tprint_error("❌ Insufficient data for seasonality detection (need >= 100 points)")
            raise ValueError("Need at least 100 data points for seasonality detection")
        
        try:
            # Get timeframe from series index directly
            timeframe_minutes = self._get_timeframe_minutes(pd.DataFrame(index=series.index))
            
            # Look for daily, weekly patterns
            daily_period = 24 * 60 // timeframe_minutes
            weekly_period = daily_period * 7
            
            seasonal_periods = []
            for period in [daily_period, weekly_period]:
                if self.min_period <= period <= self.max_period:
                    seasonal_periods.append(period)
                    tprint_debug(f"✅ Added seasonal period: {period}")
            
            tprint_debug(f"✅ Found {len(seasonal_periods)} seasonal periods: {seasonal_periods}")
            return seasonal_periods
            
        except Exception as e:
            tprint_error(f"❌ Seasonality detection failed: {e}")
            raise RuntimeError(f"Seasonality detection failed: {e}")
    
    
    def _find_pattern_periods(self, pattern: pd.Series) -> List[int]:
        """Find periods in a boolean pattern with fast fail."""
        tprint_debug("🔍 Finding pattern periods...")
        
        # Fast fail for invalid inputs
        if not isinstance(pattern, pd.Series):
            tprint_error("❌ Invalid input: expected pandas Series")
            raise ValueError("Expected pandas Series, got {type(pattern).__name__}")
        
        if len(pattern) == 0:
            tprint_error("❌ Empty pattern series")
            raise ValueError("Pattern series cannot be empty")
        
        if not pattern.dtype == 'bool':
            tprint_error("❌ Pattern series must be boolean")
            raise ValueError("Pattern series must be boolean type")
        
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
            
            # Handle case where pattern ends with True values
            if in_pattern:
                pattern_lengths.append(current_length)
            
            tprint_debug(f"📊 Found pattern lengths: {pattern_lengths}")
            
            # Return average pattern length if it's reasonable
            if pattern_lengths:
                avg_length = np.mean(pattern_lengths)
                if self.min_period <= avg_length <= self.max_period:
                    tprint_debug(f"✅ Valid average pattern length: {int(avg_length)}")
                    return [int(avg_length)]
                else:
                    tprint_debug(f"⚠️ Average pattern length {int(avg_length)} outside valid range [{self.min_period}, {self.max_period}]")
            
            tprint_debug("⚠️ No valid pattern periods found")
            return []
            
        except Exception as e:
            tprint_error(f"❌ Pattern analysis failed: {e}")
            raise RuntimeError(f"Pattern analysis failed: {e}")
    
    
    
    def _filter_periods(self, periods: List[int], 
                       characteristics: Dict[str, Any]) -> List[int]:
        """Filter periods based on data characteristics with fast fail."""
        tprint_debug(f"🔍 Filtering {len(periods)} periods: {periods}")
        
        # Fast fail for invalid inputs
        if not isinstance(periods, list):
            tprint_error("❌ Invalid input: expected list of periods")
            raise ValueError("Expected list of periods, got {type(periods).__name__}")
        
        if not isinstance(characteristics, dict):
            tprint_error("❌ Invalid input: expected characteristics dictionary")
            raise ValueError("Expected characteristics dictionary, got {type(characteristics).__name__}")
        
        if not periods:
            tprint_warning("⚠️ Empty periods list provided")
            return []
        
        try:
            filtered = []
            data_length = characteristics.get('data_length', 0)
            timeframe_minutes = characteristics.get('timeframe_minutes', 15)
            
            if data_length <= 0:
                tprint_error("❌ Invalid data length in characteristics")
                raise ValueError("Data length must be positive")
            
            tprint_debug(f"📊 Data length: {data_length}, Timeframe: {timeframe_minutes}min")
            tprint_debug(f"📊 Period bounds: [{self.min_period}, {self.max_period}]")
            
            for period in periods:
                if not isinstance(period, int):
                    tprint_debug(f"⚠️ Skipping non-integer period: {period}")
                    continue
                
                # Check if period is within bounds
                if not (self.min_period <= period <= self.max_period):
                    tprint_debug(f"❌ Period {period} outside bounds [{self.min_period}, {self.max_period}]")
                    continue
                
                # Check if period is reasonable for data length
                if period > data_length // 4:  # Don't use periods longer than 1/4 of data
                    tprint_debug(f"❌ Period {period} too long for data length {data_length} (max: {data_length // 4})")
                    continue
                
                # Check if period makes sense for timeframe
                if period < 2:  # At least 2 periods
                    tprint_debug(f"❌ Period {period} too short (minimum: 2)")
                    continue
                
                filtered.append(period)
                tprint_debug(f"✅ Period {period} passed all filters")
            
            result = sorted(list(set(filtered)))
            tprint_debug(f"✅ Filtered to {len(result)} periods: {result}")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Period filtering failed: {e}")
            raise RuntimeError(f"Period filtering failed: {e}")
    
    def _rank_periods(self, periods: List[int], data: pd.DataFrame, 
                     characteristics: Dict[str, Any]) -> List[int]:
        """Rank periods by their potential usefulness with fast fail."""
        tprint_debug(f"🔍 Ranking {len(periods)} periods: {periods}")
        
        # Fast fail for invalid inputs
        if not isinstance(periods, list):
            tprint_error("❌ Invalid input: expected list of periods")
            raise ValueError("Expected list of periods, got {type(periods).__name__}")
        
        if not isinstance(data, pd.DataFrame):
            tprint_error("❌ Invalid input: expected pandas DataFrame")
            raise ValueError("Expected pandas DataFrame, got {type(data).__name__}")
        
        if not isinstance(characteristics, dict):
            tprint_error("❌ Invalid input: expected characteristics dictionary")
            raise ValueError("Expected characteristics dictionary, got {type(characteristics).__name__}")
        
        if not periods:
            tprint_warning("⚠️ No periods to rank")
            return []
        
        try:
            scores = []
            data_length = characteristics.get('data_length', 0)
            
            if data_length <= 0:
                tprint_error("❌ Invalid data length in characteristics")
                raise ValueError("Data length must be positive")
            
            for period in periods:
                if not isinstance(period, int):
                    tprint_debug(f"⚠️ Skipping non-integer period: {period}")
                    continue
                
                score = 0
                score_components = {}
                
                # Diversity score (prefer periods that are different from others)
                other_periods = [p for p in periods if p != period and isinstance(p, int)]
                if other_periods:
                    min_diff = min(abs(period - p) for p in other_periods)
                    diversity_score = min_diff / max(period, 1)
                    score += diversity_score
                    score_components['diversity'] = diversity_score
                
                # Data coverage score (prefer periods that use more data)
                coverage = min(period, data_length) / data_length
                score += coverage
                score_components['coverage'] = coverage
                
                # Stability score (prefer periods that are stable across different windows)
                if 'close' in data.columns and len(data) > period * 2:
                    try:
                        returns = data['close'].pct_change().dropna()
                        if len(returns) > period:
                            rolling_vol = returns.rolling(period).std()
                            vol_stability = 1 / (rolling_vol.std() + 1e-8)
                            score += vol_stability
                            score_components['stability'] = vol_stability
                        else:
                            score_components['stability'] = 0
                    except Exception as e:
                        tprint_debug(f"⚠️ Stability calculation failed for period {period}: {e}")
                        score_components['stability'] = 0
                
                scores.append((score, period))
                tprint_debug(f"📊 Period {period}: score={score:.3f}, components={score_components}")
            
            if not scores:
                tprint_warning("⚠️ No valid periods to rank")
                return []
            
            # Sort by score (descending)
            scores.sort(reverse=True)
            ranked_periods = [period for score, period in scores]
            
            tprint_debug(f"✅ Ranked periods: {ranked_periods}")
            return ranked_periods
            
        except Exception as e:
            tprint_error(f"❌ Period ranking failed: {e}")
            raise RuntimeError(f"Period ranking failed: {e}")
    
    def _categorize_periods(self, periods: List[int], 
                          characteristics: Dict[str, Any]) -> Dict[str, List[int]]:
        """Categorize periods by their characteristics with fast fail."""
        tprint_debug(f"🔍 Categorizing {len(periods)} periods: {periods}")
        
        # Fast fail for invalid inputs
        if not isinstance(periods, list):
            tprint_error("❌ Invalid input: expected list of periods")
            raise ValueError("Expected list of periods, got {type(periods).__name__}")
        
        if not isinstance(characteristics, dict):
            tprint_error("❌ Invalid input: expected characteristics dictionary")
            raise ValueError("Expected characteristics dictionary, got {type(characteristics).__name__}")
        
        try:
            categories = {
                'short_term': [],
                'medium_term': [],
                'long_term': [],
                'volatility_driven': [],
                'trend_driven': [],
                'volume_driven': []
            }
            
            data_length = characteristics.get('data_length', 0)
            volatility_clusters = characteristics.get('volatility_clusters', [])
            trend_cycles = characteristics.get('trend_cycles', [])
            
            if data_length <= 0:
                tprint_error("❌ Invalid data length in characteristics")
                raise ValueError("Data length must be positive")
            
            tprint_debug(f"📊 Data length: {data_length}")
            tprint_debug(f"📊 Volatility clusters: {volatility_clusters}")
            tprint_debug(f"📊 Trend cycles: {trend_cycles}")
            
            for period in periods:
                if not isinstance(period, int):
                    tprint_debug(f"⚠️ Skipping non-integer period: {period}")
                    continue
                
                # Time-based categorization
                if period <= data_length // 20:
                    categories['short_term'].append(period)
                    tprint_debug(f"✅ Period {period} categorized as short_term")
                elif period <= data_length // 10:
                    categories['medium_term'].append(period)
                    tprint_debug(f"✅ Period {period} categorized as medium_term")
                else:
                    categories['long_term'].append(period)
                    tprint_debug(f"✅ Period {period} categorized as long_term")
                
                # Pattern-based categorization (simplified)
                if period in volatility_clusters:
                    categories['volatility_driven'].append(period)
                    tprint_debug(f"✅ Period {period} categorized as volatility_driven")
                
                if period in trend_cycles:
                    categories['trend_driven'].append(period)
                    tprint_debug(f"✅ Period {period} categorized as trend_driven")
            
            tprint_debug(f"✅ Period categorization complete: {categories}")
            return categories
            
        except Exception as e:
            tprint_error(f"❌ Period categorization failed: {e}")
            raise RuntimeError(f"Period categorization failed: {e}")
    
    def _calculate_confidence_score(self, periods: List[int], 
                                  characteristics: Dict[str, Any]) -> float:
        """Calculate confidence score for the selected periods with fast fail."""
        tprint_debug(f"🔍 Calculating confidence score for {len(periods)} periods")
        
        # Fast fail for invalid inputs
        if not isinstance(periods, list):
            tprint_error("❌ Invalid input: expected list of periods")
            raise ValueError("Expected list of periods, got {type(periods).__name__}")
        
        if not isinstance(characteristics, dict):
            tprint_error("❌ Invalid input: expected characteristics dictionary")
            raise ValueError("Expected characteristics dictionary, got {type(characteristics).__name__}")
        
        if not periods:
            tprint_warning("⚠️ No periods provided, returning 0.0")
            return 0.0
        
        try:
            score = 0.0
            score_components = {}
            
            # Data sufficiency score
            data_length = characteristics.get('data_length', 0)
            if data_length <= 0:
                tprint_error("❌ Invalid data length in characteristics")
                raise ValueError("Data length must be positive")
            
            if data_length > 1000:
                data_score = 0.3
            elif data_length > 500:
                data_score = 0.2
            elif data_length > 100:
                data_score = 0.1
            else:
                data_score = 0.0
            
            score += data_score
            score_components['data_sufficiency'] = data_score
            tprint_debug(f"📊 Data sufficiency score: {data_score} (data_length: {data_length})")
            
            # Period diversity score
            if len(periods) >= 3:
                diversity_score = 0.2
            elif len(periods) >= 2:
                diversity_score = 0.1
            else:
                diversity_score = 0.0
            
            score += diversity_score
            score_components['diversity'] = diversity_score
            tprint_debug(f"📊 Diversity score: {diversity_score} (periods: {len(periods)})")
            
            # Analysis completeness score
            analysis_components = ['volatility_clusters', 'trend_cycles', 'volume_patterns']
            completed_analyses = sum(1 for comp in analysis_components if comp in characteristics)
            completeness_score = (completed_analyses / len(analysis_components)) * 0.3
            score += completeness_score
            score_components['completeness'] = completeness_score
            tprint_debug(f"📊 Completeness score: {completeness_score} (completed: {completed_analyses}/{len(analysis_components)})")
            
            # Period reasonableness score
            reasonable_periods = sum(1 for p in periods if isinstance(p, int) and 2 <= p <= data_length // 4)
            reasonableness_score = (reasonable_periods / len(periods)) * 0.2
            score += reasonableness_score
            score_components['reasonableness'] = reasonableness_score
            tprint_debug(f"📊 Reasonableness score: {reasonableness_score} (reasonable: {reasonable_periods}/{len(periods)})")
            
            final_score = min(score, 1.0)
            tprint_debug(f"✅ Final confidence score: {final_score:.3f} (components: {score_components})")
            return final_score
            
        except Exception as e:
            tprint_error(f"❌ Confidence calculation failed: {e}")
            raise RuntimeError(f"Confidence calculation failed: {e}")
    
    def _generate_cache_key(self, operation: str, data: pd.DataFrame) -> str:
        """Generate cache key for operation with fast fail."""
        tprint_debug(f"🔍 Generating cache key for operation: {operation}")
        
        # Fast fail for invalid inputs
        if not isinstance(operation, str) or not operation:
            tprint_error("❌ Invalid operation name for cache key")
            raise ValueError("Operation must be a non-empty string")
        
        if not isinstance(data, pd.DataFrame):
            tprint_error("❌ Invalid data type for cache key")
            raise ValueError("Data must be a pandas DataFrame")
        
        try:
            import hashlib
            
            # Create hash of data characteristics and operation
            data_hash = hashlib.md5(str(data.shape).encode()).hexdigest()[:8]
            cache_key = f"{operation}_{data_hash}"
            tprint_debug(f"✅ Generated cache key: {cache_key}")
            return cache_key
            
        except Exception as e:
            tprint_error(f"❌ Cache key generation failed: {e}")
            raise RuntimeError(f"Cache key generation failed: {e}")
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get result from cache with fast fail."""
        tprint_debug(f"🔍 Checking cache for key: {cache_key}")
        
        # Fast fail for invalid inputs
        if not isinstance(cache_key, str) or not cache_key:
            tprint_error("❌ Invalid cache key")
            raise ValueError("Cache key must be a non-empty string")
        
        if not self._cache_enabled:
            tprint_debug("⚠️ Cache disabled")
            return None
        
        try:
            if cache_key in self._result_cache:
                tprint_debug(f"✅ Cache hit for key: {cache_key}")
                return self._result_cache[cache_key]
            else:
                tprint_debug(f"❌ Cache miss for key: {cache_key}")
                return None
        except Exception as e:
            tprint_error(f"❌ Cache retrieval failed: {e}")
            raise RuntimeError(f"Cache retrieval failed: {e}")
    
    def _put_in_cache(self, cache_key: str, result: Any):
        """Put result in cache with fast fail."""
        tprint_debug(f"🔍 Storing result in cache with key: {cache_key}")
        
        # Fast fail for invalid inputs
        if not isinstance(cache_key, str) or not cache_key:
            tprint_error("❌ Invalid cache key for storage")
            raise ValueError("Cache key must be a non-empty string")
        
        if result is None:
            tprint_error("❌ Cannot cache None result")
            raise ValueError("Cannot cache None result")
        
        if not self._cache_enabled:
            tprint_debug("⚠️ Cache disabled, not storing")
            return
        
        try:
            # Limit cache size
            if len(self._result_cache) >= self._max_cache_size:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self._result_cache))
                del self._result_cache[oldest_key]
                tprint_debug(f"🗑️ Removed oldest cache entry: {oldest_key}")
            
            self._result_cache[cache_key] = result
            tprint_debug(f"✅ Stored result in cache (size: {len(self._result_cache)}/{self._max_cache_size})")
            
        except Exception as e:
            tprint_error(f"❌ Cache storage failed: {e}")
            raise RuntimeError(f"Cache storage failed: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics with fast fail."""
        tprint_debug("🔍 Gathering performance statistics...")
        
        try:
            stats = self.performance_stats.copy()
            
            # Add VectorBT rolling optimizer stats if available
            if self.rolling_optimizer:
                tprint_debug("📊 Adding VectorBT rolling optimizer stats")
                try:
                    rolling_stats = self.rolling_optimizer.get_performance_stats()
                    if rolling_stats:
                        stats.update(rolling_stats)
                        tprint_debug("✅ VectorBT rolling optimizer stats added")
                    else:
                        tprint_debug("⚠️ VectorBT rolling optimizer returned empty stats")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to get VectorBT rolling optimizer stats: {e}")
            else:
                tprint_debug("⚠️ VectorBT rolling optimizer not available")
            
            # Add unified vectorization manager stats if available
            if self.vectorization_manager:
                tprint_debug("📊 Adding unified vectorization manager stats")
                try:
                    vectorization_stats = self.vectorization_manager.get_performance_stats()
                    if vectorization_stats:
                        stats.update(vectorization_stats)
                        tprint_debug("✅ Unified vectorization manager stats added")
                    else:
                        tprint_debug("⚠️ Unified vectorization manager returned empty stats")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to get unified vectorization manager stats: {e}")
            else:
                tprint_debug("⚠️ Unified vectorization manager not available")
            
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
                
                tprint_debug(f"📊 Performance metrics calculated: {len(stats)} total metrics")
            else:
                stats['average_operation_time'] = 0
                stats['vectorbt_usage_rate'] = 0
                stats['batch_usage_rate'] = 0
                stats['cache_hit_rate'] = 0
                tprint_debug("⚠️ No operations recorded, using default values")
            
            tprint_debug(f"✅ Performance stats ready: {len(stats)} metrics")
            return stats
            
        except Exception as e:
            tprint_error(f"❌ Performance stats gathering failed: {e}")
            raise RuntimeError(f"Performance stats gathering failed: {e}")
    
    def reset_performance_stats(self):
        """Reset performance statistics with fast fail."""
        tprint_info("🔄 Resetting performance statistics...")
        
        try:
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
                tprint_debug("🔄 Resetting VectorBT rolling optimizer stats")
                try:
                    self.rolling_optimizer.reset_stats()
                    tprint_debug("✅ VectorBT rolling optimizer stats reset")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to reset VectorBT rolling optimizer stats: {e}")
            else:
                tprint_debug("⚠️ VectorBT rolling optimizer not available for reset")
            
            if self.vectorization_manager:
                tprint_debug("🔄 Resetting unified vectorization manager stats")
                try:
                    self.vectorization_manager.reset_stats()
                    tprint_debug("✅ Unified vectorization manager stats reset")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to reset unified vectorization manager stats: {e}")
            else:
                tprint_debug("⚠️ Unified vectorization manager not available for reset")
            
            # Clear cache
            cache_size = len(self._result_cache)
            self._result_cache.clear()
            tprint_debug(f"🗑️ Cleared cache ({cache_size} entries removed)")
            
            tprint_success("✅ Performance statistics reset complete")
            
        except Exception as e:
            tprint_error(f"❌ Performance stats reset failed: {e}")
            raise RuntimeError(f"Performance stats reset failed: {e}")
    
    @contextmanager
    def performance_monitoring(self, operation_name: str):
        """Context manager for performance monitoring."""
        tprint_debug(f"🔍 Starting performance monitoring for: {operation_name}")
        start_time = time.time()
        start_memory = 0  # Could add memory monitoring here
        
        try:
            yield
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            
            tprint_performance(f"Operation {operation_name}: {execution_time:.3f}s")
            tprint_debug(f"✅ Performance monitoring complete for: {operation_name}")
    
    def optimize_for_large_datasets(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data for large dataset processing with fast fail."""
        tprint_debug(f"🔍 Optimizing data for large datasets (size: {len(data)}, memory_efficient: {self.memory_efficient})")
        
        # Fast fail for invalid inputs
        if not isinstance(data, pd.DataFrame):
            tprint_error("❌ Invalid input: expected pandas DataFrame")
            raise ValueError("Expected pandas DataFrame, got {type(data).__name__}")
        
        if len(data) == 0:
            tprint_error("❌ Empty DataFrame provided")
            raise ValueError("DataFrame cannot be empty")
        
        if not self.memory_efficient or len(data) < self.chunk_size:
            tprint_debug("⚠️ Skipping optimization (memory_efficient=False or data too small)")
            return data
        
        try:
            if self.vectorization_manager:
                tprint_debug("📊 Using VectorBT vectorization manager for optimization")
                optimized_data = self.vectorization_manager.optimize_dataframe(data)
                if optimized_data is None or optimized_data.empty:
                    tprint_error("❌ VectorBT optimization returned invalid data")
                    raise RuntimeError("VectorBT optimization failed to produce valid data")
                tprint_debug(f"✅ VectorBT optimization complete (shape: {optimized_data.shape})")
                return optimized_data
            else:
                tprint_debug("📊 Using basic optimization (VectorBT manager not available)")
                # Basic optimization
                optimized_data = data.copy()
                
                # Optimize data types
                optimized_columns = 0
                for column in optimized_data.columns:
                    if optimized_data[column].dtype == 'float64':
                        if (optimized_data[column].min() >= np.finfo(np.float32).min and 
                            optimized_data[column].max() <= np.finfo(np.float32).max):
                            optimized_data[column] = optimized_data[column].astype(np.float32)
                            optimized_columns += 1
                
                tprint_debug(f"✅ Basic optimization complete (optimized {optimized_columns} columns)")
                return optimized_data
                
        except Exception as e:
            tprint_error(f"❌ Data optimization failed: {e}")
            raise RuntimeError(f"Data optimization failed: {e}")
    
    def _get_fallback_periods(self, characteristics: Dict[str, Any]) -> PeriodAnalysisResult:
        """Get fallback periods when analysis fails - this should rarely be called with fast fail."""
        tprint_error("❌ CRITICAL: Fallback periods requested - this indicates a serious issue")
        
        # Fast fail - this method should not be used with proper error handling
        raise RuntimeError("Fallback periods should not be used with fast fail error handling. Fix the underlying issue instead.")


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
        
    Raises:
        ValueError: If input data is invalid
        RuntimeError: If analysis fails
    """
    tprint_info(f"🚀 Getting data-driven periods (data_shape: {data.shape}, target: {target_timeframe})")
    tprint_debug(f"📊 Configuration: max_periods={max_periods}, vectorbt={enable_vectorbt}, parallel={enable_parallel}, memory_efficient={memory_efficient}")
    
    # Fast fail for invalid inputs
    if not isinstance(data, pd.DataFrame):
        tprint_error("❌ Invalid input: expected pandas DataFrame")
        raise ValueError("Expected pandas DataFrame, got {type(data).__name__}")
    
    if len(data) == 0:
        tprint_error("❌ Empty DataFrame provided")
        raise ValueError("DataFrame cannot be empty")
    
    if not isinstance(max_periods, int) or max_periods <= 0:
        tprint_error("❌ Invalid max_periods: must be positive integer")
        raise ValueError("max_periods must be a positive integer")
    
    try:
        selector = DataDrivenPeriodSelector(
            max_periods=max_periods,
            enable_vectorbt=enable_vectorbt,
            enable_parallel=enable_parallel,
            memory_efficient=memory_efficient
        )
        result = selector.select_optimal_periods(data, target_timeframe)
        
        tprint_success(f"✅ Data-driven periods retrieved: {result.optimal_periods}")
        return result.optimal_periods
        
    except Exception as e:
        tprint_error(f"❌ Failed to get data-driven periods: {e}")
        raise RuntimeError(f"Failed to get data-driven periods: {e}")


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
        
    Raises:
        ValueError: If input data is invalid
        RuntimeError: If analysis fails
    """
    tprint_info(f"🚀 Getting data-driven periods with stats (data_shape: {data.shape}, target: {target_timeframe})")
    tprint_debug(f"📊 Configuration: max_periods={max_periods}, vectorbt={enable_vectorbt}, parallel={enable_parallel}, memory_efficient={memory_efficient}")
    
    # Fast fail for invalid inputs
    if not isinstance(data, pd.DataFrame):
        tprint_error("❌ Invalid input: expected pandas DataFrame")
        raise ValueError("Expected pandas DataFrame, got {type(data).__name__}")
    
    if len(data) == 0:
        tprint_error("❌ Empty DataFrame provided")
        raise ValueError("DataFrame cannot be empty")
    
    if not isinstance(max_periods, int) or max_periods <= 0:
        tprint_error("❌ Invalid max_periods: must be positive integer")
        raise ValueError("max_periods must be a positive integer")
    
    try:
        selector = DataDrivenPeriodSelector(
            max_periods=max_periods,
            enable_vectorbt=enable_vectorbt,
            enable_parallel=enable_parallel,
            memory_efficient=memory_efficient
        )
        result = selector.select_optimal_periods(data, target_timeframe)
        stats = selector.get_performance_stats()
        
        tprint_success(f"✅ Data-driven periods with stats retrieved: {result.optimal_periods}")
        tprint_debug(f"📊 Performance stats: {len(stats)} metrics collected")
        return result.optimal_periods, stats
        
    except Exception as e:
        tprint_error(f"❌ Failed to get data-driven periods with stats: {e}")
        raise RuntimeError(f"Failed to get data-driven periods with stats: {e}")


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
        
    Raises:
        ValueError: If input data is invalid
        RuntimeError: If benchmarking fails
    """
    tprint_info(f"🚀 Starting period selector benchmark (data_shape: {data.shape}, trials: {trials})")
    tprint_debug(f"📊 Target timeframe: {target_timeframe}, max_periods: {max_periods}")
    
    # Fast fail for invalid inputs
    if not isinstance(data, pd.DataFrame):
        tprint_error("❌ Invalid input: expected pandas DataFrame")
        raise ValueError("Expected pandas DataFrame, got {type(data).__name__}")
    
    if len(data) == 0:
        tprint_error("❌ Empty DataFrame provided")
        raise ValueError("DataFrame cannot be empty")
    
    if not isinstance(trials, int) or trials <= 0:
        tprint_error("❌ Invalid trials: must be positive integer")
        raise ValueError("trials must be a positive integer")
    
    if not isinstance(max_periods, int) or max_periods <= 0:
        tprint_error("❌ Invalid max_periods: must be positive integer")
        raise ValueError("max_periods must be a positive integer")
    
    try:
        configurations = [
            {'enable_vectorbt': False, 'enable_parallel': False, 'memory_efficient': False, 'name': 'baseline'},
            {'enable_vectorbt': True, 'enable_parallel': False, 'memory_efficient': False, 'name': 'vectorbt_only'},
            {'enable_vectorbt': True, 'enable_parallel': True, 'memory_efficient': False, 'name': 'vectorbt_parallel'},
            {'enable_vectorbt': True, 'enable_parallel': True, 'memory_efficient': True, 'name': 'vectorbt_optimized'},
        ]
        
        results = {}
        
        for config in configurations:
            config_name = config.pop('name')
            tprint_info(f"🔄 Benchmarking configuration: {config_name}")
            tprint_debug(f"📊 Config: {config}")
            
            times = []
            
            for trial in range(trials):
                try:
                    tprint_debug(f"🔄 Trial {trial + 1}/{trials} for {config_name}")
                    selector = DataDrivenPeriodSelector(max_periods=max_periods, **config)
                    start_time = time.time()
                    result = selector.select_optimal_periods(data, target_timeframe)
                    execution_time = time.time() - start_time
                    times.append(execution_time)
                    tprint_debug(f"✅ Trial {trial + 1} completed in {execution_time:.3f}s")
                except Exception as e:
                    tprint_warning(f"⚠️ Configuration {config_name} trial {trial + 1} failed: {e}")
                    continue
            
            if times:
                results[config_name] = {
                    'avg_time': np.mean(times),
                    'std_time': np.std(times),
                    'min_time': np.min(times),
                    'max_time': np.max(times),
                    'trials_completed': len(times)
                }
                tprint_success(f"✅ {config_name}: {results[config_name]['avg_time']:.3f}s ± {results[config_name]['std_time']:.3f}s ({len(times)}/{trials} trials)")
            else:
                tprint_error(f"❌ {config_name}: All trials failed")
        
        tprint_success(f"✅ Benchmark complete: {len(results)} configurations tested")
        return results
        
    except Exception as e:
        tprint_error(f"❌ Benchmarking failed: {e}")
        raise RuntimeError(f"Benchmarking failed: {e}")

