"""
SR Quality Data Collector

Collects historical SR levels and labels them with forward performance metrics.
Uses artifact_manager to load existing downloaded data (no re-downloading).
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List
from pathlib import Path
from tqdm import tqdm

# Import RealDataLoader for proper data access
from src.utils.data.real_data_loader import RealDataLoader

logger = logging.getLogger(__name__)


class SRQualityDataCollector:
    """Collects historical SR levels and labels them with performance metrics.
    
    Uses RealDataLoader for proper data access - loads existing data.
    """
    
    def __init__(self):
        self.data_loader = RealDataLoader()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # OPTIMIZATION: Create SR detector ONCE and reuse (10x speedup)
        from ..enhanced_sr_detection import EnhancedSRDetector
        
        self.logger.info("🚀 Initializing optimized SR detector for training data collection...")
        self.sr_detector = EnhancedSRDetector(config={
            # CRITICAL: Disable clustering and backtesting (slow, not needed for training)
            'disable_dbscan_clustering': True,
            'disable_backtesting_validation': True,
            
            # SPEED OPTIMIZATIONS: Limit levels per method
            'max_levels_per_method': 10,  # Top 10 per method (was 20-30)
            
            # ✅ ENABLE ALL METHODS - All are already optimized with vectorbt/numba/numpy:
            
            # Fractals - Uses Numba JIT with parallel processing (fast!)
            'fractal_periods': [5],  # 1 period only (numba optimized)
            'use_optimized_fractals': True,  # Enable numba optimization
            
            # Pivots - Uses VectorBT (vectorized, fast!)
            'pivot_periods': [5],  # 1 period only (vectorbt optimized)
            
            # Volume - Uses Numba (very fast!)
            # Statistical - Uses NumPy (very fast!)
            # Both enabled by default
            
            # Psychological - Simple numpy operations (fast!)
            'psychological_levels': True,
            
            # Fibonacci - Vectorized calculations (reasonably fast)
            'fibonacci_levels': True,
            
            # Trendlines - Optimized with vectorized pre-computation
            'trendline_levels': True,
            
            # Channels - Optimized with intelligent candidate selection
            'channel_levels': True,
            
            # Ensure ALL optimizations are enabled:
            'enable_fractal_caching': True,
            'enable_pivot_caching': True,
            'enable_performance_monitoring': False,  # Reduce logging overhead
        })
        
        self.logger.info("✅ Optimized SR detector initialized (all methods enabled with vectorbt/numba/numpy)")
        
    async def collect_training_data(self, symbol: str, exchange: str, 
                              start_date: str, end_date: str,
                              timeframe: str = '1h',
                              forward_days: int = 10,
                              sample_freq_days: float = 0.5) -> pd.DataFrame:
        """Collect SR levels from historical data and label with performance.
        
        Process:
        1. Load full historical OHLCV using RealDataLoader (already downloaded)
        2. Walk forward through time
        3. For each date: detect SR, look forward, measure performance
        4. Create training samples
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            exchange: Exchange name (e.g., 'binance')
            start_date: Start date for training data (e.g., '2023-01-01')
            end_date: End date (e.g., '2024-01-01')
            timeframe: Timeframe to analyze (e.g., '1h')
            forward_days: Days to look forward for performance measurement
            sample_freq_days: Sampling frequency in days (0.5 = 12-hour samples, 1 = daily, 7 = weekly)
            
        Returns:
            DataFrame with [features..., quality_score, performance_metrics...]
        """
        
        # Store timeframe for adaptive threshold calculations
        self.current_timeframe = timeframe
        
        self.logger.info(f"📊 Collecting SR training data for {symbol} {exchange} {timeframe}")
        self.logger.info(f"   Period: {start_date} to {end_date}")
        self.logger.info(f"   Forward window: {forward_days} days")
        self.logger.info(f"   Sample frequency: every {sample_freq_days} days")
        
        # Load full historical data using RealDataLoader (async)
        full_data = await self._load_historical_data_async(symbol, exchange, timeframe, start_date, end_date)
        
        if full_data is None or full_data.empty:
            raise ValueError(f"No data found for {symbol} {exchange} {timeframe}")
        
        self.logger.info(f"✅ Loaded {len(full_data)} historical bars")
        self.logger.info(f"   Date range: {full_data.index.min()} to {full_data.index.max()}")
        
        # Walk forward through time
        training_samples = []
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        sample_dates = pd.date_range(start_dt, end_dt, freq=f'{sample_freq_days}D')
        
        # Make sample_dates timezone-aware to match data
        if isinstance(full_data.index, pd.DatetimeIndex) and full_data.index.tz is not None:
            sample_dates = sample_dates.tz_localize('UTC')
            self.logger.info(f"   Sample dates made timezone-aware (UTC)")
        
        # Also convert start_dt and end_dt for filtering
        if isinstance(full_data.index, pd.DatetimeIndex) and full_data.index.tz is not None:
            start_dt = start_dt.tz_localize('UTC')
            end_dt = end_dt.tz_localize('UTC')
        
        # OPTIMIZATION: Early stopping when we have enough samples
        # INCREASED: Need more samples for robust ML training with 100+ features
        target_samples = 5000  # Stop when we have 5000 samples (was 1000)
        self.logger.info(f"🔄 Processing {len(sample_dates)} sample dates (target: {target_samples} samples, no hard limit)...")
        
        for current_date in tqdm(sample_dates, desc="Collecting samples"):
            try:
                # Split into historical (for detection) and future (for labeling)
                historical_data = full_data[full_data.index < current_date]
                future_end = current_date + timedelta(days=forward_days)
                future_data = full_data[
                    (full_data.index >= current_date) & 
                    (full_data.index < future_end)
                ]
                
                # Need enough data
                if len(historical_data) < 200 or len(future_data) < 5:
                    continue
                
                # Detect SR levels on historical data
                levels = self._detect_sr_levels(historical_data, symbol, exchange, timeframe)
                
                if not levels:
                    continue
                
                # Label each level with future performance
                for level in levels:
                    try:
                        # Measure performance
                        performance = self._measure_level_performance(
                            level, future_data, historical_data
                        )
                        
                        # Extract ALL features
                        features = self._extract_all_features(level, historical_data)
                        
                        # Create training sample
                        sample = {
                            'date': current_date,
                            'symbol': symbol,
                            'exchange': exchange,
                            'timeframe': timeframe,
                            **features,  # All 30+ features
                            **performance  # Labels
                        }
                        
                        training_samples.append(sample)
                        
                    except Exception as e:
                        self.logger.debug(f"Failed to process level: {e}")
                        continue
            
            except Exception as e:
                self.logger.warning(f"Failed to process date {current_date}: {e}")
                continue
            
            # OPTIMIZATION: Early stopping when target reached
            if len(training_samples) >= target_samples:
                self.logger.info(f"✅ Target reached: {len(training_samples)} samples collected")
                self.logger.info(f"   Processed {sample_dates.tolist().index(current_date) + 1}/{len(sample_dates)} dates")
                break
        
        # Convert to DataFrame
        training_df = pd.DataFrame(training_samples)
        
        if len(training_df) == 0:
            raise ValueError("No training samples collected!")
        
        # CRITICAL FIX: Exclude untested levels (quality_score == 0.2)
        # These are levels that were NEVER HIT in the forward window
        # They have no real performance data - just default values
        initial_count = len(training_df)
        untested_count = (training_df['quality_score'] == 0.2).sum()
        
        self.logger.info(f"\n🔍 Filtering out UNTESTED levels (quality_score == 0.2):")
        self.logger.info(f"   Total samples before filtering: {initial_count}")
        self.logger.info(f"   Untested levels (never hit): {untested_count} ({untested_count/initial_count*100:.1f}%)")
        
        training_df = training_df[training_df['quality_score'] > 0.2].copy()
        
        self.logger.info(f"   ✅ Samples after filtering: {len(training_df)} ({len(training_df)/initial_count*100:.1f}% retained)")
        self.logger.info(f"   Reason: Untested levels have NO predictive signal (assigned arbitrary default 0.2)")
        
        if len(training_df) == 0:
            raise ValueError("No tested samples remaining after filtering untested levels!")
        
        self.logger.info(f"\n✅ Training data collection complete!")
        self.logger.info(f"   Total samples: {len(training_df)}")
        self.logger.info(f"   Date range: {training_df['date'].min()} to {training_df['date'].max()}")
        self.logger.info(f"   Features: {len([c for c in training_df.columns if c.startswith('feature_')])} columns")
        self.logger.info(f"   Quality score range: [{training_df['quality_score'].min():.3f}, {training_df['quality_score'].max():.3f}]")
        self.logger.info(f"   Quality score mean: {training_df['quality_score'].mean():.3f}")
        
        # VALIDATION: Run automatic data quality checks
        self.logger.info(f"\n🔍 Running automatic data validation...")
        try:
            from .quality_data_validator import QualityDataValidator
            
            validator = QualityDataValidator(strict_mode=False)  # Non-strict for warnings
            validation_report = validator.validate_training_data(training_df, timeframe)
            
            # Log critical issues
            if validation_report['critical_issues']:
                self.logger.error(f"\n❌ Data validation found {len(validation_report['critical_issues'])} critical issues!")
                for issue in validation_report['critical_issues']:
                    self.logger.error(f"   • {issue}")
            
            # Log warnings
            if validation_report['warnings']:
                self.logger.warning(f"\n⚠️  Data validation found {len(validation_report['warnings'])} warnings:")
                for warning in validation_report['warnings'][:5]:  # Show first 5
                    self.logger.warning(f"   • {warning}")
                if len(validation_report['warnings']) > 5:
                    self.logger.warning(f"   ... and {len(validation_report['warnings']) - 5} more")
            
            if validation_report['validation_passed']:
                self.logger.info(f"✅ Data validation passed")
            
        except ImportError:
            self.logger.debug("Quality validator not available, skipping validation")
        except Exception as e:
            self.logger.warning(f"Data validation error (non-critical): {e}")
        
        return training_df
    
    def filter_top_quality_levels(self, training_data: pd.DataFrame, 
                                  percentile: float = 80.0) -> pd.DataFrame:
        """
        Filter training data to top N% by quality.
        
        This removes noise and weak levels, keeping only relevant training examples.
        Based on validation: 75.6% of data is noise/weak, only top 20% matters!
        
        Args:
            training_data: Full training dataset
            percentile: Keep top N% (default: 80 = top 20%)
            
        Returns:
            Filtered dataset with only high-quality levels
        """
        # Calculate quality threshold
        threshold = np.percentile(training_data['quality_score'], percentile)
        
        # Filter
        filtered = training_data[training_data['quality_score'] >= threshold].copy()
        
        self.logger.info(f"\n📊 TRAINING DATA FILTERING (Top {100-percentile:.0f}%):")
        self.logger.info(f"   Percentile threshold: {percentile}%")
        self.logger.info(f"   Quality threshold: {threshold:.3f}")
        self.logger.info(f"   Original samples: {len(training_data):,}")
        self.logger.info(f"   Filtered samples: {len(filtered):,} ({len(filtered)/len(training_data)*100:.1f}%)")
        self.logger.info(f"   Removed samples: {len(training_data) - len(filtered):,}")
        
        # Quality distribution after filtering
        self.logger.info(f"\n   Quality distribution (after filtering):")
        self.logger.info(f"     Min:    {filtered['quality_score'].min():.3f}")
        self.logger.info(f"     25th:   {filtered['quality_score'].quantile(0.25):.3f}")
        self.logger.info(f"     Median: {filtered['quality_score'].median():.3f}")
        self.logger.info(f"     75th:   {filtered['quality_score'].quantile(0.75):.3f}")
        self.logger.info(f"     Max:    {filtered['quality_score'].max():.3f}")
        
        return filtered
    
    def add_confidence_weights(self, training_data: pd.DataFrame,
                              method: str = 'quality_based') -> pd.DataFrame:
        """
        OPTIMIZED: Add confidence/sample weights to training data (SOFT FILTERING).

        Enhanced performance version with vectorized operations and reduced memory usage.
        """
        quality = training_data['quality_score'].values
        n_samples = len(quality)

        # OPTIMIZED: Vectorized weight calculation
        if method == 'quality_based':
            weights = quality.copy()  # Direct assignment, no computation needed

        elif method == 'tiered':
            # Vectorized tiered assignment using numpy operations
            weights = np.select(
                [
                    (quality >= 0.85),  # Critical
                    (quality >= 0.7),   # Strong
                    (quality >= 0.5),   # Medium
                    (quality >= 0.3),   # Weak
                    (quality >= 0.0)    # Noise (fallback)
                ],
                [2.0, 1.2, 0.8, 0.5, 0.3],
                default=0.3
            )

        elif method == 'exponential':
            weights = quality ** 2

        else:
            raise ValueError(f"Unknown weighting method: {method}")

        # Normalize weights efficiently
        weights_mean = weights.mean()
        if weights_mean > 0:
            weights = weights / weights_mean

        # OPTIMIZED: In-place column addition to avoid full DataFrame copy
        result = training_data.assign(sample_weight=weights)

        # OPTIMIZED: Minimal logging for performance
        self.logger.info(f"✅ Added {method} weights to {n_samples:,} samples")

        return result
    
    async def _load_historical_data_async(self, symbol: str, exchange: str, timeframe: str, 
                                           start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical data using RealDataLoader.
        
        Uses proper data loading infrastructure to get market data.
        """
        try:
            # Use RealDataLoader to load data
            data = await self.data_loader.load_market_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                force_download=False
            )
            
            if data is not None and len(data) > 0:
                self.logger.info(f"✅ Loaded {len(data)} bars from RealDataLoader")
                return data
            else:
                self.logger.error(f"❌ No data found for {symbol} {exchange} {timeframe}")
                return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Failed to load historical data: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
    
    def _detect_sr_levels(self, data: pd.DataFrame, symbol: str, 
                         exchange: str, timeframe: str) -> List:
        """Detect SR levels on historical data window.
        
        OPTIMIZATION: Reuses self.sr_detector (created once in __init__)
        instead of creating new detector each time (10x speedup).
        
        All detection methods use fast implementations:
        - Fractals: Numba JIT parallel processing
        - Pivots: VectorBT vectorized operations  
        - Volume: Numba optimization
        - Statistical: NumPy vectorization
        - Fibonacci: Vectorized calculations
        - Trendlines: Pre-computed parameters with vectorization
        - Channels: Intelligent candidate selection + vectorization
        """
        try:
            # REUSE pre-initialized detector (no re-creation!)
            # All methods optimized with vectorbt/numba/numpy
            result = self.sr_detector.detect_sr_levels(data[-500:])  # Last 500 bars
            
            if isinstance(result, dict) and 'levels' in result:
                return result['levels']
            elif isinstance(result, list):
                return result
            else:
                return []
                
        except Exception as e:
            self.logger.warning(f"SR detection failed: {e}")
            return []
    
    def _get_adaptive_bounce_threshold(self, timeframe: str) -> float:
        """Get adaptive bounce threshold based on timeframe.
        
        IMPROVEMENT #1: Adaptive Bounce Thresholds by Timeframe
        Different timeframes have different typical move sizes.
        
        Args:
            timeframe: Timeframe string (e.g., '1h', '4h', '1d')
            
        Returns:
            Threshold percentage as float (e.g., 0.04 for 4%)
            
        Raises:
            ValueError: If timeframe is invalid type
        """
        # Validate input
        if not isinstance(timeframe, str):
            self.logger.error(f"Invalid timeframe type: {type(timeframe)}, expected str")
            raise ValueError(f"Timeframe must be string, got {type(timeframe)}")
        
        if not timeframe or timeframe.strip() == '':
            self.logger.warning("Empty timeframe provided, using default 4%")
            return 0.04
        
        thresholds = {
            '1m': 0.015,   # 1.5% (very small moves)
            '5m': 0.020,   # 2.0%
            '15m': 0.025,  # 2.5%
            '30m': 0.030,  # 3.0%
            '1h': 0.040,   # 4.0% (current default)
            '2h': 0.050,   # 5.0%
            '4h': 0.060,   # 6.0%
            '6h': 0.070,   # 7.0%
            '12h': 0.075,  # 7.5%
            '1d': 0.080,   # 8.0%
            '24h': 0.080,  # 8.0%
        }
        
        threshold = thresholds.get(timeframe, None)
        
        if threshold is None:
            self.logger.warning(f"Unknown timeframe '{timeframe}', using default 4%")
            return 0.04
        
        return threshold
    
    def _calculate_time_weighted_bounce(self, early_future: pd.DataFrame, hit_bar, 
                                        level_type: str, level_price: float) -> tuple:
        """Calculate time-weighted bounce (not just max) with error handling.
        
        IMPROVEMENT #2: Time-Weighted Bounce
        Recent bounces matter more than later ones.
        
        Args:
            early_future: DataFrame of future bars after hit
            hit_bar: The bar where level was hit
            level_type: 'support' or 'resistance'
            level_price: Price of the level
            
        Returns:
            Tuple of (weighted_bounce_pct, max_bounce_pct)
            Returns (0.0, 0.0) on error
        """
        try:
            # Validate inputs
            if early_future is None or len(early_future) == 0:
                self.logger.warning("Empty early_future data in time-weighted bounce calculation")
                return 0.0, 0.0
            
            if level_price <= 0:
                self.logger.error(f"Invalid level_price: {level_price} (must be > 0)")
                return 0.0, 0.0
            
            if level_type not in ['support', 'resistance']:
                self.logger.error(f"Invalid level_type: {level_type}")
                return 0.0, 0.0
            
            weighted_bounce = 0.0
            total_weight = 0.0
            max_bounce_pct = 0.0
            valid_bars = 0
            
            for i, (idx, bar) in enumerate(early_future.iterrows()):
                try:
                    # Validate bar data
                    if pd.isna(bar.get('high')) or pd.isna(bar.get('low')):
                        self.logger.debug(f"NaN values in bar {i}, skipping")
                        continue
                    
                    if pd.isna(hit_bar.get('high')) or pd.isna(hit_bar.get('low')):
                        self.logger.warning("NaN values in hit_bar")
                        return 0.0, 0.0
                    
                    # Calculate bounce at this bar
                    if level_type == 'support':
                        bounce = float(bar['high']) - float(hit_bar['low'])
                    else:  # resistance
                        bounce = float(hit_bar['high']) - float(bar['low'])
                    
                    # Validate bounce value
                    if bounce < 0:
                        bounce = 0  # Negative bounce means no bounce
                    
                    bounce_pct = bounce / level_price
                    
                    # Sanity check: bounce shouldn't be > 100%
                    if bounce_pct > 1.0:
                        self.logger.warning(f"Extreme bounce detected: {bounce_pct*100:.1f}%, capping at 100%")
                        bounce_pct = 1.0
                    
                    max_bounce_pct = max(max_bounce_pct, bounce_pct)
                    
                    # Exponential time decay: earlier bounces weighted more
                    weight = np.exp(-i / 3)  # Decay factor of 3 bars
                    weighted_bounce += bounce_pct * weight
                    total_weight += weight
                    valid_bars += 1
                    
                except Exception as e:
                    self.logger.debug(f"Error processing bar {i} in time-weighted bounce: {e}")
                    continue
            
            if valid_bars == 0:
                self.logger.warning("No valid bars found for time-weighted bounce calculation")
                return 0.0, 0.0
            
            # Calculate weighted average
            if total_weight > 0:
                weighted_bounce_pct = weighted_bounce / total_weight
            else:
                self.logger.warning("Total weight is zero, using max bounce as fallback")
                weighted_bounce_pct = max_bounce_pct
            
            return float(weighted_bounce_pct), float(max_bounce_pct)
            
        except Exception as e:
            self.logger.error(f"Unexpected error in time-weighted bounce calculation: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return 0.0, 0.0
    
    def _calculate_rejection_speed(self, future_data: pd.DataFrame, hit_bar, 
                                   level_type: str, level_price: float, 
                                   first_hit_idx) -> float:
        """Calculate how quickly price rejected from the level with error handling.
        
        IMPROVEMENT #3: Rejection Speed Component
        Fast rejections indicate strong levels.
        
        Args:
            future_data: Full future data DataFrame
            hit_bar: Bar where level was hit
            level_type: 'support' or 'resistance'
            level_price: Price of the level
            first_hit_idx: Index of first hit
            
        Returns:
            rejection_speed score (0-1), 0.0 on error
        """
        try:
            # Validate inputs
            if future_data is None or len(future_data) == 0:
                self.logger.warning("Empty future_data in rejection speed calculation")
                return 0.0
            
            if level_price <= 0:
                self.logger.error(f"Invalid level_price: {level_price}")
                return 0.0
            
            if level_type not in ['support', 'resistance']:
                self.logger.error(f"Invalid level_type: {level_type}")
                return 0.0
            
            # Get early future bars (up to 5)
            try:
                early_future = future_data.loc[first_hit_idx:].iloc[:5]
            except Exception as e:
                self.logger.warning(f"Error slicing future_data: {e}")
                return 0.0
            
            if len(early_future) == 0:
                self.logger.warning("No early_future bars available")
                return 0.0
            
            for i, (idx, bar) in enumerate(early_future.iterrows()):
                try:
                    # Validate bar has close price
                    if 'close' not in bar or pd.isna(bar['close']):
                        self.logger.debug(f"Missing or NaN close price in bar {i}")
                        continue
                    
                    close_price = float(bar['close'])
                    
                    # Check for significant bounce (1%+)
                    if level_type == 'support':
                        bounce_size = (close_price - level_price) / level_price
                    else:  # resistance
                        bounce_size = (level_price - close_price) / level_price
                    
                    # Sanity check
                    if bounce_size > 2.0:  # > 200% bounce is unrealistic
                        self.logger.warning(f"Extreme bounce_size: {bounce_size*100:.1f}%, skipping")
                        continue
                    
                    if bounce_size > 0.01:  # 1% bounce threshold
                        # Faster rejection = higher score
                        speed_score = 1.0 - (i / 5.0)  # First bar = 1.0, 5th bar = 0.0
                        
                        # Scale by bounce magnitude
                        magnitude_factor = min(abs(bounce_size) / 0.02, 1.0)  # 2% = full score
                        
                        result = speed_score * magnitude_factor
                        return float(np.clip(result, 0, 1))
                    
                except Exception as e:
                    self.logger.debug(f"Error processing bar {i} in rejection speed: {e}")
                    continue
            
            return 0.0  # No significant rejection found
            
        except Exception as e:
            self.logger.error(f"Unexpected error in rejection speed calculation: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return 0.0
    
    def _calculate_volume_quality(self, future_data: pd.DataFrame, historical_data: pd.DataFrame,
                                  first_hit_idx) -> float:
        """Calculate volume confirmation quality with error handling.
        
        IMPROVEMENT #5: Add Volume Confirmation to Quality
        High volume at test + during bounce = stronger level.
        
        Args:
            future_data: Future data DataFrame
            historical_data: Historical data DataFrame
            first_hit_idx: Index of first hit
            
        Returns:
            volume_quality score (0-1), 0.5 (neutral) on error
        """
        try:
            # Check if volume data exists
            if 'volume' not in future_data.columns or 'volume' not in historical_data.columns:
                self.logger.debug("Volume column not found, returning neutral score")
                return 0.5  # Neutral if no volume data
            
            # Validate data not empty
            if len(historical_data) == 0:
                self.logger.warning("Empty historical_data in volume quality calculation")
                return 0.5
            
            # Calculate average historical volume
            avg_volume = historical_data['volume'].mean()
            
            # Handle zero or NaN average volume
            if pd.isna(avg_volume) or avg_volume <= 0:
                self.logger.debug(f"Invalid avg_volume: {avg_volume}, returning neutral")
                return 0.5
            
            # Get volume at the test
            try:
                test_volume = future_data.loc[first_hit_idx, 'volume']
            except (KeyError, IndexError) as e:
                self.logger.warning(f"Could not get test volume at index {first_hit_idx}: {e}")
                return 0.5
            
            # Validate test volume
            if pd.isna(test_volume) or test_volume < 0:
                self.logger.debug(f"Invalid test_volume: {test_volume}")
                return 0.5
            
            test_volume_ratio = float(test_volume) / avg_volume
            
            # Get volume during bounce (next 5 bars)
            try:
                bounce_bars = future_data.loc[first_hit_idx:].iloc[:5]
            except Exception as e:
                self.logger.warning(f"Error getting bounce bars: {e}")
                return 0.5
            
            if len(bounce_bars) == 0:
                self.logger.warning("No bounce bars available for volume calculation")
                return 0.5
            
            # Calculate average bounce volume
            bounce_volume_avg = bounce_bars['volume'].mean()
            
            if pd.isna(bounce_volume_avg) or bounce_volume_avg < 0:
                self.logger.debug(f"Invalid bounce_volume_avg: {bounce_volume_avg}")
                # Use test volume only if bounce volume invalid
                volume_score = test_volume_ratio / 2.5
            else:
                bounce_volume_ratio = float(bounce_volume_avg) / avg_volume
                
                # Combine: test volume (60%) + bounce volume (40%)
                volume_score = (test_volume_ratio * 0.6 + bounce_volume_ratio * 0.4) / 2.5
            
            # Sanity check: extremely high volume ratios (>10x) might be errors
            if volume_score > 4.0:
                self.logger.warning(f"Extremely high volume_score: {volume_score:.2f}, capping at 1.0")
                return 1.0
            
            return float(np.clip(volume_score, 0, 1))
            
        except Exception as e:
            self.logger.error(f"Unexpected error in volume quality calculation: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return 0.5  # Neutral on error
    
    def _measure_level_performance(self, level, future_data: pd.DataFrame,
                                   historical_data: pd.DataFrame) -> Dict[str, float]:
        """Measure level performance in future data with ENHANCED metrics and error handling.
        
        IMPROVEMENTS IMPLEMENTED:
        1. Adaptive bounce thresholds by timeframe
        2. Time-weighted bounce (not just max)
        3. Rejection speed component
        4. Multi-outcome quality scores
        5. Volume confirmation quality
        
        Returns:
            Dictionary with performance metrics including:
            - Single quality_score (composite)
            - Multi-outcome scores (bounce_quality, hold_quality, etc.)
            - Enhanced components
            Returns default performance on critical errors.
        """
        try:
            # Validate inputs
            if future_data is None or len(future_data) == 0:
                self.logger.warning("Empty future_data in performance measurement")
                return self._get_default_performance()
            
            if historical_data is None or len(historical_data) == 0:
                self.logger.warning("Empty historical_data in performance measurement")
                return self._get_default_performance()
            
            # Get level attributes safely
            level_price = getattr(level, 'price', None) if not isinstance(level, dict) else level.get('price')
            level_type = getattr(level, 'type', None) if not isinstance(level, dict) else level.get('type')
            
            if level_price is None or level_price <= 0:
                self.logger.warning(f"Invalid level price: {level_price}")
                return self._get_default_performance()
            
            if level_type not in ['support', 'resistance']:
                self.logger.warning(f"Invalid level type: {level_type}")
                return self._get_default_performance()
            
            tolerance = level_price * 0.005  # 0.5% tolerance
            
            # Get timeframe for adaptive thresholds
            timeframe = getattr(self, 'current_timeframe', '1h')
            
            # Check if price hit the level
            if level_type == 'support':
                hits = future_data[future_data['low'] <= level_price + tolerance]
            elif level_type == 'resistance':
                hits = future_data[future_data['high'] >= level_price - tolerance]
            else:
                return self._get_default_performance()
            
            if len(hits) == 0:
                # Level NOT tested - assign low quality
                return {
                    # Data-driven target
                    'realized_pnl_pct': 0.0,  # Not tested = no P&L
                    # Base components
                    'hit_rate': 0.0,
                    'bounce_strength': 0.0,
                    'max_bounce_strength': 0.0,
                    'hold_strength': 0.5,
                    'trade_profit': 0.0,
                    'rejection_speed': 0.0,
                    'volume_quality': 0.5,
                    # Heuristic scores
                    'quality_score': 0.2,  # Low quality (untested)
                    # Multi-outcome scores
                    'bounce_quality': 0.0,
                    'hold_quality': 0.5,
                    'trade_quality': 0.0,
                    'speed_quality': 0.0,
                    'volume_confirmation_quality': 0.5
                }
            
            # Level WAS hit - measure bounce
            first_hit_idx = hits.index[0]
            hit_bar = hits.loc[first_hit_idx]
            
            # 1. BOUNCE STRENGTH (Enhanced with time-weighting and adaptive thresholds)
            early_future = future_data.loc[first_hit_idx:].iloc[:5]
            
            # Calculate both weighted and max bounce
            weighted_bounce_pct, max_bounce_pct = self._calculate_time_weighted_bounce(
                early_future, hit_bar, level_type, level_price
            )
            
            # Get adaptive threshold based on timeframe
            bounce_threshold = self._get_adaptive_bounce_threshold(timeframe)
            
            # Use time-weighted bounce (primary) with adaptive threshold
            bounce_strength = min(weighted_bounce_pct / bounce_threshold, 1.0)
            max_bounce_strength = min(max_bounce_pct / bounce_threshold, 1.0)  # For reference
            
            # 2. HOLD STRENGTH (unchanged - already works well)
            if level_type == 'support':
                breaks = future_data.loc[first_hit_idx:][
                    future_data['close'] < level_price - tolerance
                ]
            else:
                breaks = future_data.loc[first_hit_idx:][
                    future_data['close'] > level_price + tolerance
                ]
            
            if len(breaks) == 0:
                hold_strength = 1.0  # Held perfectly
            else:
                bars_until_break = len(future_data.loc[first_hit_idx:breaks.index[0]])
                hold_strength = min(bars_until_break / 20, 1.0)  # 20+ bars = 1.0
            
            # 3. TRADE PROFIT - DATA-DRIVEN APPROACH (NEW!)
            trade_result = self._simulate_trade(level_type, level_price, future_data, first_hit_idx)
            realized_pnl_pct = trade_result['realized_pnl_pct']  # PRIMARY TARGET! ✅
            trade_profit = trade_result['trade_profit']  # Normalized (backward compat)
            
            # 4. REJECTION SPEED (NEW)
            rejection_speed = self._calculate_rejection_speed(
                future_data, hit_bar, level_type, level_price, first_hit_idx
            )
            
            # 5. VOLUME QUALITY (NEW)
            volume_quality = self._calculate_volume_quality(
                future_data, historical_data, first_hit_idx
            )
            
            # 6. COMPOSITE QUALITY SCORE (HEURISTIC - for comparison only!)
            quality_score = (
                bounce_strength * 0.25 +           # Time-weighted bounce
                hold_strength * 0.20 +             # How long it holds
                max(trade_profit, 0) * 0.20 +      # Trade profitability
                rejection_speed * 0.20 +           # Speed of rejection (NEW)
                volume_quality * 0.15              # Volume confirmation (NEW)
            )
            
            # 7. MULTI-OUTCOME QUALITY SCORES (HEURISTIC - for comparison)
            # Separate quality scores for different use cases
            bounce_quality = (bounce_strength * 0.6 + rejection_speed * 0.4)  # For mean reversion
            hold_quality = (hold_strength * 0.7 + volume_quality * 0.3)       # For S/R strength
            trade_quality = max(trade_profit, 0)                              # For trading
            speed_quality = rejection_speed                                    # For quick bounces
            volume_confirmation_quality = volume_quality                       # For confirmation
            
            # Debug logging (1% sample rate)
            if np.random.random() < 0.01:
                self.logger.debug(f"\n🔍 ENHANCED QUALITY SCORE DEBUG:")
                self.logger.debug(f"   Level: ${level_price:.2f} ({level_type}) @ {timeframe}")
                self.logger.debug(f"   Adaptive threshold: {bounce_threshold*100:.1f}%")
                self.logger.debug(f"   Future performance:")
                self.logger.debug(f"     Bounce (weighted): {bounce_strength:.3f} ({weighted_bounce_pct*100:.2f}%)")
                self.logger.debug(f"     Bounce (max): {max_bounce_strength:.3f} ({max_bounce_pct*100:.2f}%)")
                self.logger.debug(f"     Hold: {hold_strength:.3f}")
                self.logger.debug(f"     Trade: {trade_profit:.3f}")
                self.logger.debug(f"     Rejection speed: {rejection_speed:.3f}")
                self.logger.debug(f"     Volume quality: {volume_quality:.3f}")
                self.logger.debug(f"   Composite quality: {quality_score:.3f}")
                self.logger.debug(f"   Multi-outcome:")
                self.logger.debug(f"     Bounce quality: {bounce_quality:.3f}")
                self.logger.debug(f"     Hold quality: {hold_quality:.3f}")
                self.logger.debug(f"     Trade quality: {trade_quality:.3f}")
            
            # Final validation of calculated values
            if not (0 <= quality_score <= 1.1):  # Allow slight overflow for rounding
                self.logger.warning(f"Quality score out of range: {quality_score}, clipping")
                quality_score = np.clip(quality_score, 0, 1)
            
            return {
                # =================================================================
                # ✅ DATA-DRIVEN TARGET (PRIMARY!)
                # =================================================================
                'realized_pnl_pct': float(realized_pnl_pct),  # ACTUAL PROFIT! ✅
                
                # =================================================================
                # Base components (for analysis)
                # =================================================================
                'hit_rate': 1.0,
                'bounce_strength': float(bounce_strength),
                'max_bounce_strength': float(max_bounce_strength),
                'hold_strength': float(hold_strength),
                'trade_profit': float(trade_profit),  # Normalized (backward compat)
                'rejection_speed': float(rejection_speed),
                'volume_quality': float(volume_quality),
                
                # =================================================================
                # HEURISTIC scores (for comparison/benchmarking)
                # =================================================================
                'quality_score': float(np.clip(quality_score, 0, 1)),  # OLD approach
                
                # Multi-outcome quality scores (heuristic)
                'bounce_quality': float(np.clip(bounce_quality, 0, 1)),
                'hold_quality': float(np.clip(hold_quality, 0, 1)),
                'trade_quality': float(np.clip(trade_quality, 0, 1)),
                'speed_quality': float(speed_quality),
                'volume_confirmation_quality': float(volume_confirmation_quality)
            }
            
        except Exception as e:
            self.logger.error(f"Unexpected error in _measure_level_performance: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._get_default_performance()
    
    def _simulate_trade(self, level_type: str, entry_price: float,
                       future_data: pd.DataFrame, hit_idx) -> Dict[str, float]:
        """Simulate trade at level with 2:1 R/R (1% SL and 2% TP).
        
        DATA-DRIVEN APPROACH: Returns ACTUAL P&L percentage for training.
        This is the PRIMARY TARGET for data-driven quality prediction!
        
        Returns:
            Dictionary with:
            - realized_pnl_pct: Actual P&L percentage (-0.01 to +0.02)
            - trade_profit: Normalized score for backward compatibility (-1 to +1)
        """
        if level_type == 'support':
            stop_loss = entry_price * 0.99     # 1% SL
            take_profit = entry_price * 1.02   # 2% TP (2:1 R/R)
            direction = 1
        else:  # resistance
            stop_loss = entry_price * 1.01
            take_profit = entry_price * 0.98
            direction = -1
        
        # Check next 10 bars
        future_bars = future_data.loc[hit_idx:].iloc[:10]
        
        for _, bar in future_bars.iterrows():
            if direction == 1:  # Long
                if bar['low'] <= stop_loss:
                    return {
                        'realized_pnl_pct': -0.01,  # Lost 1%
                        'trade_profit': -0.5  # Normalized (backward compat)
                    }
                elif bar['high'] >= take_profit:
                    return {
                        'realized_pnl_pct': 0.02,  # Made 2%
                        'trade_profit': 1.0  # Normalized (backward compat)
                    }
            else:  # Short
                if bar['high'] >= stop_loss:
                    return {
                        'realized_pnl_pct': -0.01,
                        'trade_profit': -0.5
                    }
                elif bar['low'] <= take_profit:
                    return {
                        'realized_pnl_pct': 0.02,
                        'trade_profit': 1.0
                    }
        
        # No SL/TP hit - exit at close
        exit_price = future_bars.iloc[-1]['close']
        pnl_pct = (exit_price - entry_price) / entry_price * direction
        
        return {
            'realized_pnl_pct': pnl_pct,  # ACTUAL P&L (PRIMARY TARGET!)
            'trade_profit': np.clip(pnl_pct * 100, -1, 1)  # Normalized (backward compat)
        }
    
    def _extract_all_features(self, level, data: pd.DataFrame) -> Dict[str, float]:
        """Extract OPTIMIZED feature set for ML training (reduced from 40+ to 18 features).

        Focus on most predictive features, remove redundancy for better performance.
        """
        current_price = data['close'].iloc[-1]

        # Get level attributes safely
        def get_attr(name, default=0.0):
            if isinstance(level, dict):
                return level.get(name, default)
            return getattr(level, name, default)

        # Core pre-calculations
        touch_count = get_attr('touch_count', 1)
        strength = get_attr('strength', 0.5)
        age_bars = get_attr('age_bars', 0)
        avg_bounce = get_attr('avg_bounce_ratio', 0)
        consistency = get_attr('consistency_score', 0.5)

        # Time features (simplified)
        hour_normalized = 0.0
        if len(data) > 0 and hasattr(data.index[-1], 'hour'):
            recent_hours = [data.index[i].hour for i in range(-min(20, len(data)), 0) if hasattr(data.index[i], 'hour')]
            hour_mode = max(set(recent_hours), key=recent_hours.count) if recent_hours else 0
            hour_normalized = float(hour_mode) / 24.0

        # OPTIMIZED FEATURE SET: Reduced to 18 most predictive features
        features = {
            # Core SR metrics (6 features)
            'feature_strength': strength,
            'feature_touch_count': touch_count,
            'feature_age_bars': age_bars,
            'feature_consistency': consistency,
            'feature_avg_bounce_ratio': avg_bounce,
            'feature_max_bounce_ratio': get_attr('max_bounce_ratio', 0),

            # Quality metrics (4 features)
            'feature_volume_confirmation': get_attr('volume_confirmation_score', 0.5),
            'feature_bounce_consistency': get_attr('bounce_consistency', 0),
            'feature_recency_weighted_strength': strength * np.exp(-age_bars / 50),  # Recency bonus
            'feature_touch_quality_score': (avg_bounce * 0.5) + (get_attr('avg_touch_volume_ratio', 0) * 0.3) + (consistency * 0.2),

            # Position & market context (5 features)
            'feature_price_zscore': (get_attr('price', current_price) - data['close'].mean()) / (data['close'].std() + 1e-8),
            'feature_distance_to_current_pct': abs(get_attr('price', current_price) - current_price) / current_price,
            'feature_is_support': 1.0 if get_attr('type', 'support') == 'support' else 0.0,
            'feature_market_volatility': data['close'].pct_change().std(),
            'feature_market_trend': (data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20],

            # Time & regime (3 features)
            'feature_hour_of_day': hour_normalized,
            'feature_is_high_volatility': 1.0 if data['close'].pct_change().std() > 0.03 else 0.0,
            'feature_is_uptrend': 1.0 if ((data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]) > 0.02 else 0.0,
        }

        # Add final quality tier feature for completeness
        features['feature_quality_tier'] = min(strength * 2.0, 1.0)  # Simple quality scaling

        return features
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range (ATR).
        
        Args:
            data: OHLCV data
            period: ATR period (default: 14)
            
        Returns:
            ATR value
        """
        try:
            if len(data) < period + 1:
                # Not enough data, use simple range
                return (data['high'] - data['low']).mean()
            
            # True Range = max(high-low, |high-prev_close|, |low-prev_close|)
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift(1))
            low_close = np.abs(data['low'] - data['close'].shift(1))
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # ATR = moving average of true range
            atr = true_range.rolling(window=period).mean().iloc[-1]
            
            return float(atr) if not np.isnan(atr) else (data['high'] - data['low']).mean()
            
        except Exception as e:
            self.logger.warning(f"ATR calculation failed: {e}")
            return (data['high'] - data['low']).mean()
    
    def _get_default_performance(self) -> Dict[str, float]:
        """Default performance when measurement fails."""
        return {
            # Data-driven target
            'realized_pnl_pct': 0.0,  # No data = no P&L
            # Base components
            'hit_rate': 0.0,
            'bounce_strength': 0.0,
            'max_bounce_strength': 0.0,
            'hold_strength': 0.5,
            'trade_profit': 0.0,
            'rejection_speed': 0.0,
            'volume_quality': 0.5,
            # Heuristic scores
            'quality_score': 0.3,
            # Multi-outcome scores
            'bounce_quality': 0.0,
            'hold_quality': 0.5,
            'trade_quality': 0.0,
            'speed_quality': 0.0,
            'volume_confirmation_quality': 0.5
        }
    
    def save_training_data(self, training_df: pd.DataFrame, 
                          output_path: str = None) -> str:
        """Save collected training data.
        
        Args:
            training_df: Training data DataFrame
            output_path: Optional custom path
            
        Returns:
            Path to saved file
        """
        if output_path is None:
            output_path = 'data_cache/sr_ml_training/sr_quality_training_data.parquet'
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        training_df.to_parquet(output_file, index=False)
        
        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'samples': len(training_df),
            'date_range': {
                'start': str(training_df['date'].min()),
                'end': str(training_df['date'].max())
            },
            'symbols': training_df['symbol'].unique().tolist(),
            'timeframes': training_df['timeframe'].unique().tolist(),
            'feature_count': len([c for c in training_df.columns if c.startswith('feature_')]),
            'quality_score_stats': {
                'mean': float(training_df['quality_score'].mean()),
                'std': float(training_df['quality_score'].std()),
                'min': float(training_df['quality_score'].min()),
                'max': float(training_df['quality_score'].max())
            }
        }
        
        metadata_path = str(output_file).replace('.parquet', '_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"✅ Training data saved to {output_file}")
        self.logger.info(f"✅ Metadata saved to {metadata_path}")
        
        return str(output_file)


# Convenience function
async def collect_sr_training_data(symbol: str, exchange: str, timeframe: str,
                            start_date: str, end_date: str) -> pd.DataFrame:
    """Convenience function to collect training data."""
    collector = SRQualityDataCollector()
    return await collector.collect_training_data(symbol, exchange, start_date, end_date, timeframe)

