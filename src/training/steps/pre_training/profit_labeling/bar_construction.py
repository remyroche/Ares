"""
Event-Based Bar Construction with Microstructure Filtering

This module implements event-based bar construction that suppresses microstructure noise
by reshaping the tape to reduce microstructure effects and create cleaner bars for labeling.

Key Features:
- Event-based bars (dollar bars, volume bars) instead of time bars
- Outlier-robust OHLC computation using median prices
- Microstructure filtering to remove ultra-tight ranges
- Volume and duration filtering
- Return capping to de-spike labels
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta

# Import matrix operations for vectorized computations
try:
    from src.utils.matrix_operations import UnifiedMatrixOperations
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False

# Import hardware optimization utilities
try:
    from src.utils.hardware.m1_gpu_utils import M1GPUManager
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False

# Import existing utilities
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.common_operations import (
    safe_divide, safe_log, safe_sqrt, safe_mean, safe_std,
    validate_finite, validate_positive, validate_range
)
from src.utils.math_validation import MathValidation

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


class BarType(Enum):
    """Enumeration of bar types."""
    TIME = "time"
    VOLUME = "volume"
    DOLLAR = "dollar"
    TICK = "tick"


@dataclass
class BarConstructionConfig:
    """Configuration for bar construction."""
    
    # Bar type and size
    bar_type: BarType = BarType.DOLLAR
    bar_size: float = 1000000.0  # Dollar value for dollar bars, volume for volume bars
    min_bar_duration_seconds: int = 30  # Minimum bar duration
    max_bar_duration_seconds: int = 3600  # Maximum bar duration
    
    # Microstructure filtering
    enable_microstructure_filter: bool = True
    min_spread_ratio: float = 0.0001  # Minimum (high-low)/mid ratio
    min_volume_percentile: float = 10.0  # Minimum volume percentile
    max_return_percentile: float = 99.9  # Cap returns at this percentile
    
    # OHLC computation
    use_median_prices: bool = True
    use_vwap: bool = True
    median_window: int = 5  # Window for median price calculation
    
    # Outlier handling
    enable_outlier_capping: bool = True
    outlier_threshold: float = 3.0  # Standard deviations for outlier detection
    
    # Quality checks
    min_bars_required: int = 100
    max_missing_data_ratio: float = 0.1  # Maximum ratio of missing data

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self._validate_config()

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.bar_size <= 0:
            raise ValueError("bar_size must be positive")

        if self.min_bar_duration_seconds <= 0:
            raise ValueError("min_bar_duration_seconds must be positive")

        if self.max_bar_duration_seconds <= self.min_bar_duration_seconds:
            raise ValueError("max_bar_duration_seconds must be greater than min_bar_duration_seconds")

        if not (0 < self.min_spread_ratio < 1):
            raise ValueError("min_spread_ratio must be between 0 and 1")

        if not (0 < self.min_volume_percentile < 100):
            raise ValueError("min_volume_percentile must be between 0 and 100")

        if not (0 < self.max_return_percentile < 100):
            raise ValueError("max_return_percentile must be between 0 and 100")

        if self.median_window <= 0:
            raise ValueError("median_window must be positive")

        if self.outlier_threshold <= 0:
            raise ValueError("outlier_threshold must be positive")

        if not (0 <= self.max_missing_data_ratio <= 1):
            raise ValueError("max_missing_data_ratio must be between 0 and 1")


@dataclass
class BarConstructionResult:
    """Result container for bar construction."""
    
    # Core results
    cleaned_bars: pd.DataFrame
    original_bars: pd.DataFrame
    
    # Statistics
    n_original_bars: int = 0
    n_cleaned_bars: int = 0
    bars_removed: int = 0
    removal_reasons: Dict[str, int] = field(default_factory=dict)
    
    # Quality metrics
    data_quality_score: float = 0.0
    microstructure_noise_ratio: float = 0.0
    volume_consistency: float = 0.0
    
    # Metadata
    config_used: BarConstructionConfig = None
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class EventBasedBarConstructor:
    """
    Event-Based Bar Constructor with Microstructure Filtering
    
    This class constructs event-based bars that suppress microstructure noise by:
    1. Using dollar/volume bars instead of time bars
    2. Computing outlier-robust OHLC using median prices
    3. Filtering out ultra-tight ranges dominated by microstructure
    4. Capping extreme returns to de-spike labels
    """
    
    def __init__(self, config: Optional[BarConstructionConfig] = None):
        """Initialize event-based bar constructor."""
        self.config = config or BarConstructionConfig()
        self.logger = logging.getLogger('EventBasedBarConstructor')

        # Initialize matrix operations for vectorized computations
        if MATRIX_OPS_AVAILABLE:
            self.matrix_ops = UnifiedMatrixOperations()
            tprint_info("   → Matrix operations: Available")
        else:
            self.matrix_ops = None
            tprint_warning("   → Matrix operations: Not available, using fallback")

        # Initialize hardware optimization utilities
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            self.gpu_manager = M1GPUManager()
            self.cpu_optimizer = M1CPUOptimizer()
            self.memory_optimizer = M1MemoryOptimizer()

            # Check if hardware optimization is beneficial
            gpu_info = self.gpu_manager.get_gpu_info()
            if gpu_info['is_m1'] and gpu_info['mps_available']:
                tprint_info("   → Hardware optimization: M1 GPU available")
            else:
                tprint_info("   → Hardware optimization: CPU optimization available")
        else:
            self.gpu_manager = None
            self.cpu_optimizer = None
            self.memory_optimizer = None
            tprint_warning("   → Hardware optimization: Not available")

        tprint_info("🔧 Event-Based Bar Constructor initialized")
        tprint_info(f"   → Bar type: {self.config.bar_type.value}")
        tprint_info(f"   → Bar size: {self.config.bar_size}")
        tprint_info(f"   → Microstructure filtering: {self.config.enable_microstructure_filter}")
    
    def construct_bars(self, market_data: pd.DataFrame) -> BarConstructionResult:
        """
        Construct event-based bars with microstructure filtering.
        
        Args:
            market_data: OHLCV market data with datetime index
            
        Returns:
            BarConstructionResult with cleaned bars and statistics
        """
        start_time = datetime.now()
        tprint_info("📊 Constructing event-based bars")
        
        # Initialize result container
        result = BarConstructionResult(
            cleaned_bars=pd.DataFrame(),
            original_bars=pd.DataFrame(),
            config_used=self.config
        )
        
        try:
            # Step 1: Validate input data
            if not self._validate_input_data(market_data):
                return result
            
            # Step 2: Construct event-based bars
            tprint_info("📈 Step 1: Creating event-based bars")
            event_bars = self._create_event_bars(market_data)
            result.original_bars = event_bars
            result.n_original_bars = len(event_bars)
            
            if event_bars.empty:
                tprint_warning("⚠️ No event bars created")
                return result
            
            # Step 3: Apply microstructure filtering
            if self.config.enable_microstructure_filter:
                tprint_info("🔇 Step 2: Applying microstructure filtering")
                filtered_bars = self._apply_microstructure_filtering(event_bars)
                result.cleaned_bars = filtered_bars
                result.bars_removed = len(event_bars) - len(filtered_bars)
            else:
                result.cleaned_bars = event_bars
                result.bars_removed = 0
            
            result.n_cleaned_bars = len(result.cleaned_bars)
            
            # Step 4: Calculate quality metrics
            tprint_info("📊 Step 3: Calculating quality metrics")
            quality_metrics = self._calculate_quality_metrics(result.cleaned_bars)
            result.data_quality_score = quality_metrics['data_quality_score']
            result.microstructure_noise_ratio = quality_metrics['microstructure_noise_ratio']
            result.volume_consistency = quality_metrics['volume_consistency']
            
            # Step 5: Validate minimum requirements
            if len(result.cleaned_bars) < self.config.min_bars_required:
                tprint_warning(f"⚠️ Insufficient bars: {len(result.cleaned_bars)} < {self.config.min_bars_required}")
                result.cleaned_bars = pd.DataFrame()
            
        except Exception as e:
            tprint_error(f"❌ Bar construction failed: {e}")
            return result
        
        # Calculate processing time
        result.processing_time = (datetime.now() - start_time).total_seconds()
        
        tprint_success("✅ Event-based bar construction completed")
        tprint_info(f"   → Original bars: {result.n_original_bars}")
        tprint_info(f"   → Cleaned bars: {result.n_cleaned_bars}")
        tprint_info(f"   → Bars removed: {result.bars_removed}")
        tprint_info(f"   → Quality score: {result.data_quality_score:.3f}")
        
        return result
    
    def _validate_input_data(self, market_data: pd.DataFrame) -> bool:
        """Validate input market data."""
        try:
            # Check if DataFrame is empty
            if market_data.empty:
                tprint_warning("⚠️ Input data is empty")
                return False
            
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = set(required_columns) - set(market_data.columns)
            if missing_columns:
                tprint_warning(f"⚠️ Missing required columns: {missing_columns}")
                return False
            
            # Check for datetime index
            if not isinstance(market_data.index, pd.DatetimeIndex):
                tprint_warning("⚠️ Index must be DatetimeIndex")
                return False
            
            # Check for non-finite values
            if market_data[required_columns].isnull().any().any():
                tprint_warning("⚠️ Data contains null values")
                return False
            
            if not np.isfinite(market_data[required_columns].values).all():
                tprint_warning("⚠️ Data contains non-finite values")
                return False
            
            return True
            
        except Exception as e:
            tprint_error(f"❌ Data validation failed: {e}")
            return False
    
    def _create_event_bars(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create event-based bars from market data."""
        try:
            if self.config.bar_type == BarType.DOLLAR:
                return self._create_dollar_bars(market_data)
            elif self.config.bar_type == BarType.VOLUME:
                return self._create_volume_bars(market_data)
            elif self.config.bar_type == BarType.TICK:
                return self._create_tick_bars(market_data)
            else:  # TIME
                return self._create_time_bars(market_data)
                
        except Exception as e:
            tprint_error(f"❌ Event bar creation failed: {e}")
            return pd.DataFrame()
    
    def _create_dollar_bars(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create dollar bars using median volume for more robust sizing."""
        try:
            # Calculate dollar volume (works for both USD and USDT)
            market_data = market_data.copy()
            market_data['dollar_volume'] = market_data['close'] * market_data['volume']

            # Use median volume over a window for more stable bar sizing
            window_size = min(20, len(market_data) // 10)  # Adaptive window size
            if window_size > 1:
                market_data['median_volume'] = market_data['volume'].rolling(
                    window=window_size, min_periods=1
                ).median()
                # Use median volume for bar size calculation when available
                market_data['effective_volume'] = market_data['median_volume'].fillna(market_data['volume'])
            else:
                market_data['effective_volume'] = market_data['volume']

            # Calculate cumulative dollar volume using effective volume
            market_data['cum_dollar_volume'] = (market_data['close'] * market_data['effective_volume']).cumsum()

            # Vectorized bar boundary detection
            cum_dollar_vol = market_data['cum_dollar_volume'].values
            target_dollar_volume = self.config.bar_size

            # Find where cumulative volume exceeds target (vectorized)
            bar_start_volumes = cum_dollar_vol[0]  # Start from first bar

            # Calculate volume differences from start of each potential bar
            volume_diffs = cum_dollar_vol - bar_start_volumes

            # Find where volume differences exceed target
            exceeds_target = volume_diffs >= target_dollar_volume

            # Find boundary indices where target is first exceeded
            boundary_mask = np.diff(exceeds_target.astype(int), prepend=0) > 0
            bar_boundaries = market_data.index[boundary_mask].tolist()

            # Ensure we don't lose the last bar if it doesn't reach target
            if len(bar_boundaries) == 0 or bar_boundaries[-1] != market_data.index[-1]:
                if volume_diffs[-1] > 0:
                    bar_boundaries.append(market_data.index[-1])

            # Vectorized bar creation using matrix operations where possible
            bars = []

            if self.matrix_ops and MATRIX_OPS_AVAILABLE:
                # Use matrix operations for efficient bar creation
                boundary_indices = [market_data.index.get_loc(boundary) for boundary in bar_boundaries]

                for i, boundary_idx in enumerate(boundary_indices):
                    start_idx = 0 if i == 0 else boundary_indices[i-1] + 1
                    end_idx = boundary_idx + 1

                    if start_idx < end_idx:
                        bar_data = market_data.iloc[start_idx:end_idx]
                        bar = self._create_single_bar_vectorized(bar_data)
                        if bar is not None:
                            bars.append(bar)
            else:
                # Fallback to original loop
                prev_boundary = 0
                for boundary in bar_boundaries:
                    bar_data = market_data.iloc[prev_boundary:boundary+1]

                    if len(bar_data) > 0:
                        bar = self._create_single_bar(bar_data)
                        if bar is not None:
                            bars.append(bar)

                    prev_boundary = boundary + 1

            if bars:
                return pd.DataFrame(bars).set_index('timestamp')
            else:
                return pd.DataFrame()
                
        except Exception as e:
            tprint_error(f"❌ Dollar bar creation failed: {e}")
            return pd.DataFrame()
    
    def _create_volume_bars(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create volume bars."""
        try:
            # Calculate cumulative volume
            market_data = market_data.copy()
            market_data['cum_volume'] = market_data['volume'].cumsum()
            
            # Vectorized bar boundary detection for volume bars
            cum_volume = market_data['cum_volume'].values
            target_volume = self.config.bar_size

            # Find where cumulative volume exceeds target (vectorized)
            bar_start_volumes = cum_volume[0]  # Start from first bar

            # Calculate volume differences from start of each potential bar
            volume_diffs = cum_volume - bar_start_volumes

            # Find where volume differences exceed target
            exceeds_target = volume_diffs >= target_volume

            # Find boundary indices where target is first exceeded
            boundary_mask = np.diff(exceeds_target.astype(int), prepend=0) > 0
            bar_boundaries = market_data.index[boundary_mask].tolist()

            # Ensure we don't lose the last bar if it doesn't reach target
            if len(bar_boundaries) == 0 or bar_boundaries[-1] != market_data.index[-1]:
                if volume_diffs[-1] > 0:
                    bar_boundaries.append(market_data.index[-1])

            # Vectorized bar creation using matrix operations where possible
            bars = []

            if self.matrix_ops and MATRIX_OPS_AVAILABLE:
                # Use matrix operations for efficient bar creation
                boundary_indices = [market_data.index.get_loc(boundary) for boundary in bar_boundaries]

                for i, boundary_idx in enumerate(boundary_indices):
                    start_idx = 0 if i == 0 else boundary_indices[i-1] + 1
                    end_idx = boundary_idx + 1

                    if start_idx < end_idx:
                        bar_data = market_data.iloc[start_idx:end_idx]
                        bar = self._create_single_bar_vectorized(bar_data)
                        if bar is not None:
                            bars.append(bar)
            else:
                # Fallback to original loop
                prev_boundary = 0
                for boundary in bar_boundaries:
                    bar_data = market_data.iloc[prev_boundary:boundary+1]

                    if len(bar_data) > 0:
                        bar = self._create_single_bar(bar_data)
                        if bar is not None:
                            bars.append(bar)

                    prev_boundary = boundary + 1
            
            if bars:
                return pd.DataFrame(bars).set_index('timestamp')
            else:
                return pd.DataFrame()
                
        except Exception as e:
            tprint_error(f"❌ Volume bar creation failed: {e}")
            return pd.DataFrame()
    
    def _create_tick_bars(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create tick bars (every N ticks)."""
        try:
            # For tick bars, we'll use every N rows
            tick_interval = int(self.config.bar_size)
            
            bars = []
            for i in range(0, len(market_data), tick_interval):
                end_idx = min(i + tick_interval, len(market_data))
                bar_data = market_data.iloc[i:end_idx]
                
                if len(bar_data) > 0:
                    bar = self._create_single_bar(bar_data)
                    if bar is not None:
                        bars.append(bar)
            
            if bars:
                return pd.DataFrame(bars).set_index('timestamp')
            else:
                return pd.DataFrame()
                
        except Exception as e:
            tprint_error(f"❌ Tick bar creation failed: {e}")
            return pd.DataFrame()
    
    def _create_time_bars(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create time bars (resample by time)."""
        try:
            # Resample by time interval
            time_interval = f"{int(self.config.bar_size)}T"  # Minutes
            
            resampled = market_data.resample(time_interval).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            # Convert to our bar format
            bars = []
            for timestamp, row in resampled.iterrows():
                bar = self._create_single_bar_from_ohlcv(timestamp, row)
                if bar is not None:
                    bars.append(bar)
            
            if bars:
                return pd.DataFrame(bars).set_index('timestamp')
            else:
                return pd.DataFrame()
                
        except Exception as e:
            tprint_error(f"❌ Time bar creation failed: {e}")
            return pd.DataFrame()
    
    def _create_single_bar(self, bar_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Create a single bar from bar data."""
        try:
            if len(bar_data) == 0:
                return None
            
            # Calculate timestamp (use last timestamp in the bar)
            timestamp = bar_data.index[-1]
            
            # Calculate OHLC
            if self.config.use_median_prices:
                # Use median prices for robustness
                open_price = bar_data['open'].iloc[0]
                high_price = bar_data['high'].max()
                low_price = bar_data['low'].min()
                close_price = bar_data['close'].iloc[-1]
                
                # Calculate median mid-price for additional robustness
                if self.config.use_vwap:
                    vwap = (bar_data['close'] * bar_data['volume']).sum() / bar_data['volume'].sum()
                    # Use VWAP as a reference for median calculation
                    mid_prices = (bar_data['high'] + bar_data['low']) / 2
                    median_mid = mid_prices.median()
                    
                    # Adjust OHLC if median is significantly different from VWAP
                    if abs(median_mid - vwap) / vwap > 0.01:  # 1% threshold
                        adjustment = median_mid - vwap
                        open_price += adjustment
                        high_price += adjustment
                        low_price += adjustment
                        close_price += adjustment
            else:
                # Standard OHLC
                open_price = bar_data['open'].iloc[0]
                high_price = bar_data['high'].max()
                low_price = bar_data['low'].min()
                close_price = bar_data['close'].iloc[-1]
            
            # Calculate volume
            volume = bar_data['volume'].sum()
            
            # Calculate duration
            duration = (bar_data.index[-1] - bar_data.index[0]).total_seconds()
            
            # Validate bar quality
            if not self._validate_bar_quality(open_price, high_price, low_price, close_price, volume, duration):
                return None
            
            return {
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'duration': duration,
                'ticks': len(bar_data)
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Error creating single bar: {e}")
            return None

    def _create_single_bar_vectorized(self, bar_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Create a single bar from bar data using vectorized operations."""
        try:
            if len(bar_data) == 0:
                return None

            # Vectorized OHLC calculation
            timestamp = bar_data.index[-1]

            # Use matrix operations for efficient calculations if available
            if self.matrix_ops and MATRIX_OPS_AVAILABLE:
                # Convert to numpy arrays for vectorized operations
                open_prices = bar_data['open'].values
                high_prices = bar_data['high'].values
                low_prices = bar_data['low'].values
                close_prices = bar_data['close'].values
                volumes = bar_data['volume'].values

                # Vectorized OHLC calculation
                open_price = open_prices[0]
                high_price = self.matrix_ops.matrix_max(high_prices) if hasattr(self.matrix_ops, 'matrix_max') else high_prices.max()
                low_price = self.matrix_ops.matrix_min(low_prices) if hasattr(self.matrix_ops, 'matrix_min') else low_prices.min()
                close_price = close_prices[-1]

                # Vectorized VWAP calculation
                if self.config.use_vwap:
                    total_volume = volumes.sum()
                    if total_volume > 0:
                        vwap = (close_prices * volumes).sum() / total_volume
                        mid_prices = (high_prices + low_prices) / 2
                        median_mid = np.median(mid_prices)

                        # Vectorized adjustment
                        if abs(median_mid - vwap) / vwap > 0.01:  # 1% threshold
                            adjustment = median_mid - vwap
                            open_price += adjustment
                            high_price += adjustment
                            low_price += adjustment
                            close_price += adjustment
            else:
                # Fallback to original logic
                open_price = bar_data['open'].iloc[0]
                high_price = bar_data['high'].max()
                low_price = bar_data['low'].min()
                close_price = bar_data['close'].iloc[-1]

                if self.config.use_median_prices and self.config.use_vwap:
                    vwap = (bar_data['close'] * bar_data['volume']).sum() / bar_data['volume'].sum()
                    mid_prices = (bar_data['high'] + bar_data['low']) / 2
                    median_mid = mid_prices.median()

                    if abs(median_mid - vwap) / vwap > 0.01:
                        adjustment = median_mid - vwap
                        open_price += adjustment
                        high_price += adjustment
                        low_price += adjustment
                        close_price += adjustment

            # Vectorized volume and duration calculation
            volume = bar_data['volume'].sum()
            duration = (bar_data.index[-1] - bar_data.index[0]).total_seconds()

            # Validate bar quality
            if not self._validate_bar_quality(open_price, high_price, low_price, close_price, volume, duration):
                return None

            return {
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'duration': duration,
                'ticks': len(bar_data)
            }

        except Exception as e:
            tprint_warning(f"⚠️ Error creating vectorized single bar: {e}")
            return None

    def _create_single_bar_from_ohlcv(self, timestamp: pd.Timestamp, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Create a single bar from OHLCV row."""
        try:
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            volume = row['volume']
            
            # Validate bar quality
            if not self._validate_bar_quality(open_price, high_price, low_price, close_price, volume, 0):
                return None
            
            return {
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'duration': 0,  # Will be calculated later if needed
                'ticks': 1
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Error creating single bar from OHLCV: {e}")
            return None
    
    def _validate_bar_quality(self, open_price: float, high_price: float, low_price: float, 
                            close_price: float, volume: float, duration: float) -> bool:
        """Validate bar quality."""
        try:
            # Check for non-finite values
            if not all(np.isfinite([open_price, high_price, low_price, close_price, volume])):
                return False
            
            # Check for positive values
            if not all([open_price > 0, high_price > 0, low_price > 0, close_price > 0, volume > 0]):
                return False
            
            # Check OHLC relationships
            if not (low_price <= min(open_price, close_price) and 
                   high_price >= max(open_price, close_price)):
                return False
            
            # Check duration if specified
            if duration > 0 and (duration < self.config.min_bar_duration_seconds or 
                               duration > self.config.max_bar_duration_seconds):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _apply_microstructure_filtering(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Apply microstructure filtering to remove noise-dominated bars."""
        try:
            if bars.empty:
                return bars
            
            filtered_bars = bars.copy()
            removal_reasons = {}
            
            # Calculate mid-price and spread ratio
            filtered_bars['mid_price'] = (filtered_bars['high'] + filtered_bars['low']) / 2
            filtered_bars['spread_ratio'] = (filtered_bars['high'] - filtered_bars['low']) / filtered_bars['mid_price']
            
            # Filter 1: Remove ultra-tight ranges (microstructure noise)
            tight_range_mask = filtered_bars['spread_ratio'] < self.config.min_spread_ratio
            if tight_range_mask.any():
                removal_reasons['tight_range'] = tight_range_mask.sum()
                filtered_bars = filtered_bars[~tight_range_mask]
            
            # Filter 2: Remove bars with abnormally low volume
            if len(filtered_bars) > 0:
                volume_threshold = filtered_bars['volume'].quantile(self.config.min_volume_percentile / 100)
                low_volume_mask = filtered_bars['volume'] < volume_threshold
                if low_volume_mask.any():
                    removal_reasons['low_volume'] = low_volume_mask.sum()
                    filtered_bars = filtered_bars[~low_volume_mask]
            
            # Filter 3: Cap extreme returns
            if self.config.enable_outlier_capping and len(filtered_bars) > 0:
                filtered_bars = self._cap_extreme_returns(filtered_bars)
            
            # Filter 4: Remove bars with very short duration (if duration is available)
            if 'duration' in filtered_bars.columns:
                short_duration_mask = (filtered_bars['duration'] < self.config.min_bar_duration_seconds) & (filtered_bars['duration'] > 0)
                if short_duration_mask.any():
                    removal_reasons['short_duration'] = short_duration_mask.sum()
                    filtered_bars = filtered_bars[~short_duration_mask]
            
            # Store removal reasons
            self.removal_reasons = removal_reasons
            
            return filtered_bars
            
        except Exception as e:
            tprint_error(f"❌ Microstructure filtering failed: {e}")
            return bars
    
    def _cap_extreme_returns(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Cap extreme returns to de-spike labels."""
        try:
            if bars.empty:
                return bars
            
            # Calculate returns
            bars = bars.copy()
            bars['returns'] = bars['close'].pct_change()
            
            # Calculate return thresholds
            return_threshold = bars['returns'].quantile(self.config.max_return_percentile / 100)
            
            # Cap extreme returns
            extreme_returns_mask = bars['returns'].abs() > return_threshold
            if extreme_returns_mask.any():
                bars.loc[extreme_returns_mask, 'returns'] = np.sign(bars.loc[extreme_returns_mask, 'returns']) * return_threshold
                
                # Recalculate close prices from capped returns
                bars['close'] = bars['open'] * (1 + bars['returns']).cumprod()
                
                # Recalculate high/low to maintain OHLC relationships
                bars['high'] = np.maximum(bars['high'], bars[['open', 'close']].max(axis=1))
                bars['low'] = np.minimum(bars['low'], bars[['open', 'close']].min(axis=1))
            
            return bars
            
        except Exception as e:
            tprint_warning(f"⚠️ Error capping extreme returns: {e}")
            return bars
    
    def _calculate_quality_metrics(self, bars: pd.DataFrame) -> Dict[str, float]:
        """Calculate quality metrics for the bars."""
        try:
            if bars.empty:
                return {
                    'data_quality_score': 0.0,
                    'microstructure_noise_ratio': 1.0,
                    'volume_consistency': 0.0
                }
            
            # Data quality score (based on completeness and consistency)
            data_quality_score = 1.0
            
            # Check for missing data
            missing_ratio = bars.isnull().sum().sum() / (len(bars) * len(bars.columns))
            data_quality_score -= missing_ratio
            
            # Check for extreme values
            if 'spread_ratio' in bars.columns:
                extreme_spread_ratio = (bars['spread_ratio'] > 0.1).sum() / len(bars)
                data_quality_score -= extreme_spread_ratio * 0.5
            
            # Microstructure noise ratio
            if 'spread_ratio' in bars.columns:
                microstructure_noise_ratio = (bars['spread_ratio'] < 0.001).sum() / len(bars)
            else:
                microstructure_noise_ratio = 0.0
            
            # Volume consistency (coefficient of variation)
            if 'volume' in bars.columns and bars['volume'].std() > 0:
                volume_consistency = 1.0 - (bars['volume'].std() / bars['volume'].mean())
            else:
                volume_consistency = 0.0
            
            return {
                'data_quality_score': max(0.0, min(1.0, data_quality_score)),
                'microstructure_noise_ratio': microstructure_noise_ratio,
                'volume_consistency': max(0.0, min(1.0, volume_consistency))
            }
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating quality metrics: {e}")
            return {
                'data_quality_score': 0.0,
                'microstructure_noise_ratio': 1.0,
                'volume_consistency': 0.0
            }


# Convenience functions
def create_event_based_bar_constructor(config: Optional[BarConstructionConfig] = None) -> EventBasedBarConstructor:
    """Create event-based bar constructor with specified configuration."""
    return EventBasedBarConstructor(config)


def construct_event_bars(market_data: pd.DataFrame,
                        config: Optional[BarConstructionConfig] = None) -> BarConstructionResult:
    """Construct event-based bars with default configuration."""
    constructor = EventBasedBarConstructor(config)
    return constructor.construct_bars(market_data)

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
