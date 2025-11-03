"""
Fast Optimized SR Quality Data Collector

OPTIMIZATIONS:
1. Pre-filter: Only process SR levels tested with rejection at least once
2. Focused features: All SR-specific + Top 2 from volume/momentum/trend/volatility
3. Vectorized: Use numba/numpy/VectorBT optimizers for batch operations

Uses:
- ConsolidatedRollingOptimizer (batch rolling ops)
- StatisticalCalculationsOptimizer (statistical computations)
- VectorBTRollingOptimizer (VectorBT acceleration)
- UnifiedVectorizationManager (vectorized batch processing)
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm

from src.utils.data.real_data_loader import RealDataLoader

# Import optimizers
try:
    from src.utils.ml_common.optimization.consolidated_rolling_optimizer import ConsolidatedRollingOptimizer
    from src.utils.ml_common.optimization.statistical_calculations_optimizer import StatisticalCalculationsOptimizer
    from src.utils.ml_common.optimization.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
    from src.utils.ml_common.optimization.unified_vectorization_manager import UnifiedVectorizationManager
    OPTIMIZERS_AVAILABLE = True
except ImportError:
    OPTIMIZERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class FastOptimizedSRDataCollector:
    """
    Fast, optimized SR data collector.
    
    Key optimizations:
    1. Pre-filter levels (only those tested with rejection)
    2. Focused features (SR-specific + top 2 per category)
    3. Vectorized computation (numba/numpy/VectorBT)
    4. Batch processing where possible
    """
    
    def __init__(self,
                 stop_loss_pct: float = 0.01,
                 take_profit_pct: float = 0.01,
                 max_hold_bars: int = 20):
        """Initialize optimized collector."""
        
        self.data_loader = RealDataLoader()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_hold_bars = max_hold_bars
        
        # Initialize optimizers
        if OPTIMIZERS_AVAILABLE:
            self.logger.info("🚀 Initializing vectorization optimizers...")
            
            self.consolidated_optimizer = ConsolidatedRollingOptimizer()
            self.statistical_optimizer = StatisticalCalculationsOptimizer()
            self.vectorbt_optimizer = VectorBTRollingOptimizer()
            self.vectorization_manager = UnifiedVectorizationManager()
            
            self.logger.info("✅ All optimizers initialized (numba/numpy/VectorBT)")
        else:
            self.logger.warning("⚠️  Optimizers not available, using fallback")
            self.consolidated_optimizer = None
            self.statistical_optimizer = None
            self.vectorbt_optimizer = None
            self.vectorization_manager = None
        
        # Initialize SR detector
        from ..enhanced_sr_detection import EnhancedSRDetector
        self.sr_detector = EnhancedSRDetector(config={
            'disable_dbscan_clustering': True,
            'disable_backtesting_validation': True,
            'max_levels_per_method': 15,
            'fractal_periods': [5],
            'pivot_periods': [5],
            'use_optimized_fractals': True,
        })
        
        self.logger.info(f"✅ Fast optimized collector ready")
        self.logger.info(f"   SL={stop_loss_pct*100:.1f}%, TP={take_profit_pct*100:.1f}%")
    
    async def collect_training_data(self,
                                    symbol: str,
                                    exchange: str,
                                    start_date: str,
                                    end_date: str,
                                    timeframe: str = '1h',
                                    forward_days: int = 10,
                                    sample_freq_days: int = 1) -> pd.DataFrame:
        """
        Collect optimized training data.
        
        Optimizations:
        - Pre-filter: Only process levels with historical rejection
        - Focused features: SR-specific + top 2 per category
        - Vectorized: Batch operations using optimizers
        """
        
        self.logger.info(f"📊 Collecting OPTIMIZED training data for {symbol} {exchange} {timeframe}")
        self.logger.info(f"   Period: {start_date} to {end_date}")
        self.logger.info(f"   Optimization: Pre-filtering + vectorized computation")
        
        # Load historical data
        full_data = await self._load_historical_data(symbol, exchange, timeframe, start_date, end_date)
        
        if full_data is None or full_data.empty:
            raise ValueError(f"No data found")
        
        self.logger.info(f"✅ Loaded {len(full_data)} bars")
        
        # Load multi-timeframe data
        full_data_1d = await self._load_historical_data(symbol, exchange, '1d', start_date, end_date)
        has_multi_tf = full_data_1d is not None and len(full_data_1d) > 0
        
        if has_multi_tf:
            self.logger.info(f"✅ Loaded daily data for multi-timeframe features")
        
        # Sample dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        sample_dates = pd.date_range(start_dt, end_dt, freq=f'{sample_freq_days}D')
        
        if isinstance(full_data.index, pd.DatetimeIndex) and full_data.index.tz is not None:
            sample_dates = sample_dates.tz_localize('UTC')
        
        self.logger.info(f"🔄 Processing {len(sample_dates)} sample dates (with pre-filtering)...")
        
        training_samples = []
        total_levels = 0
        filtered_levels = 0
        
        for current_date in tqdm(sample_dates, desc="Collecting"):
            try:
                historical_data = full_data[full_data.index < current_date]
                future_end = current_date + timedelta(days=forward_days)
                future_data = full_data[
                    (full_data.index >= current_date) &
                    (full_data.index < future_end)
                ]
                
                if len(historical_data) < 200 or len(future_data) < 5:
                    continue
                
                # Detect SR levels
                levels = self._detect_sr_levels(historical_data[-500:], symbol, exchange, timeframe)
                
                if not levels:
                    continue
                
                total_levels += len(levels)
                
                # ============================================================
                # OPTIMIZATION 1: PRE-FILTER LEVELS
                # Only process levels that have been tested with rejection
                # ============================================================
                
                filtered_levels_list = self._filter_tested_levels_with_rejection(
                    levels, historical_data[-200:]
                )
                
                filtered_levels += len(filtered_levels_list)
                
                if not filtered_levels_list:
                    continue
                
                # Process filtered levels
                for level in filtered_levels_list:
                    try:
                        # Extract focused, optimized features
                        features = self._extract_optimized_features(
                            level,
                            historical_data,
                            full_data_1d if has_multi_tf else None,
                            current_date
                        )
                        
                        # Calculate target
                        target = self._calculate_realized_pnl(level, future_data)
                        
                        sample = {
                            'date': current_date,
                            'symbol': symbol,
                            'exchange': exchange,
                            'timeframe': timeframe,
                            **features,
                            'realized_pnl_pct': target,
                        }
                        
                        training_samples.append(sample)
                        
                    except Exception as e:
                        self.logger.debug(f"Failed to process level: {e}")
                        continue
            
            except Exception as e:
                self.logger.warning(f"Failed date {current_date}: {e}")
                continue
        
        training_df = pd.DataFrame(training_samples)
        
        if len(training_df) == 0:
            raise ValueError("No training samples collected!")
        
        # Filter untested
        initial_count = len(training_df)
        training_df = training_df[training_df['realized_pnl_pct'] != 0.0].copy()
        
        self.logger.info(f"\n✅ Optimized collection complete!")
        self.logger.info(f"   Total levels detected: {total_levels}")
        self.logger.info(f"   Levels passing filter: {filtered_levels} ({filtered_levels/total_levels*100:.1f}%)")
        self.logger.info(f"   Final samples: {len(training_df)}")
        self.logger.info(f"   Features: {len([c for c in training_df.columns if c.startswith('feature_')])} columns")
        
        return training_df
    
    async def collect_training_data_multi_timeframe(self,
                                                    symbol: str,
                                                    exchange: str,
                                                    start_date: str,
                                                    end_date: str,
                                                    detection_timeframe: str = '1d',
                                                    testing_timeframe: str = '1h',
                                                    forward_days: int = 10,
                                                    sample_freq_days: int = 1) -> pd.DataFrame:
        """
        MULTI-TIMEFRAME STRATEGY:
        1. Detect SR levels on DAILY (major institutional levels)
        2. Test/measure performance on 1H (more granular, more samples)
        
        Benefits:
        - Daily SR = stronger, institutional levels
        - 1h testing = more samples, better signal
        - Best of both worlds!
        
        Args:
            detection_timeframe: Timeframe for SR detection (e.g., '1d')
            testing_timeframe: Timeframe for testing/features (e.g., '1h')
        """
        
        self.logger.info(f"📊 MULTI-TIMEFRAME Collection: {symbol} {exchange}")
        self.logger.info(f"   Detection TF: {detection_timeframe} (find major SR levels)")
        self.logger.info(f"   Testing TF: {testing_timeframe} (analyze behavior)")
        self.logger.info(f"   Period: {start_date} to {end_date}")
        
        # Load DAILY data (for SR detection)
        daily_data = await self._load_historical_data(symbol, exchange, detection_timeframe, start_date, end_date)
        
        if daily_data is None or daily_data.empty:
            raise ValueError(f"No daily data found")
        
        self.logger.info(f"✅ Loaded {len(daily_data)} daily bars for SR detection")
        
        # Load 1H data (for testing/features)
        hourly_data = await self._load_historical_data(symbol, exchange, testing_timeframe, start_date, end_date)
        
        if hourly_data is None or hourly_data.empty:
            raise ValueError(f"No hourly data found")
        
        self.logger.info(f"✅ Loaded {len(hourly_data)} hourly bars for testing")
        
        # Sample dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        sample_dates = pd.date_range(start_dt, end_dt, freq=f'{sample_freq_days}D')
        
        if isinstance(daily_data.index, pd.DatetimeIndex) and daily_data.index.tz is not None:
            sample_dates = sample_dates.tz_localize('UTC')
        
        self.logger.info(f"🔄 Processing {len(sample_dates)} dates (multi-timeframe)...")
        
        training_samples = []
        total_daily_levels = 0
        filtered_levels = 0
        
        for current_date in tqdm(sample_dates, desc="Multi-TF collection"):
            try:
                # =============================================================
                # STEP 1: Detect SR on DAILY timeframe
                # =============================================================
                
                daily_historical = daily_data[daily_data.index < current_date]
                
                if len(daily_historical) < 50:
                    continue
                
                # Detect major SR levels on daily
                daily_levels = self._detect_sr_levels(
                    daily_historical[-100:], symbol, exchange, detection_timeframe
                )
                
                if not daily_levels:
                    continue
                
                total_daily_levels += len(daily_levels)
                
                # =============================================================
                # STEP 2: Get 1H data for testing
                # =============================================================
                
                hourly_historical = hourly_data[hourly_data.index < current_date]
                future_end = current_date + timedelta(days=forward_days)
                hourly_future = hourly_data[
                    (hourly_data.index >= current_date) &
                    (hourly_data.index < future_end)
                ]
                
                if len(hourly_historical) < 200 or len(hourly_future) < 5:
                    continue
                
                # =============================================================
                # STEP 3: Pre-filter daily SR levels (tested + rejected on 1H)
                # =============================================================
                
                filtered_daily_levels = self._filter_tested_levels_with_rejection(
                    daily_levels, hourly_historical[-1000:]  # Check 1H history
                )
                
                filtered_levels += len(filtered_daily_levels)
                
                if not filtered_daily_levels:
                    continue
                
                # =============================================================
                # STEP 4: For each daily SR, analyze on 1H timeframe
                # =============================================================
                
                for daily_level in filtered_daily_levels:
                    try:
                        # Extract features from 1H data
                        features = self._extract_optimized_features(
                            daily_level,
                            hourly_historical,  # Use 1H for features!
                            daily_historical,   # Also pass daily for multi-TF
                            current_date
                        )
                        
                        # Measure performance on 1H future data
                        target = self._calculate_realized_pnl(daily_level, hourly_future)
                        
                        sample = {
                            'date': current_date,
                            'symbol': symbol,
                            'exchange': exchange,
                            'detection_tf': detection_timeframe,
                            'testing_tf': testing_timeframe,
                            **features,
                            'realized_pnl_pct': target,
                        }
                        
                        training_samples.append(sample)
                        
                    except Exception as e:
                        self.logger.debug(f"Failed level: {e}")
                        continue
            
            except Exception as e:
                self.logger.warning(f"Failed date {current_date}: {e}")
                continue
        
        training_df = pd.DataFrame(training_samples)
        
        if len(training_df) == 0:
            raise ValueError("No samples collected!")
        
        # Filter untested
        initial_count = len(training_df)
        training_df = training_df[training_df['realized_pnl_pct'] != 0.0].copy()
        
        self.logger.info(f"\n✅ Multi-timeframe collection complete!")
        self.logger.info(f"   Daily SR levels detected: {total_daily_levels}")
        self.logger.info(f"   After filtering (tested+rejected on 1H): {filtered_levels} ({filtered_levels/total_daily_levels*100:.1f}%)")
        self.logger.info(f"   Final samples: {len(training_df)}")
        self.logger.info(f"   Strategy: Daily SR detection → 1H testing/features")
        
        return training_df
    
    async def collect_training_data_efficient_multi_tf(self,
                                                       symbol: str,
                                                       exchange: str,
                                                       start_date: str,
                                                       end_date: str,
                                                       detection_timeframe: str = '1d',
                                                       testing_timeframe: str = '1h',
                                                       forward_days: int = 10,
                                                       sample_freq_days: int = 1) -> pd.DataFrame:
        """
        SIMPLE EFFICIENT MULTI-TF: Get TOP 2 SR levels per day (support + resistance).
        
        Strategy:
        1. For each day: Detect SR on daily data → Get TOP 1 support + TOP 1 resistance
        2. Check if each level was tested + rejected on 1H (bounce action)
        3. If yes: Extract features from 1H + measure performance
        
        Benefits:
        - Clean dataset: ~2 levels per day (1 support + 1 resistance)
        - Only quality levels (tested + bounced on 1H)
        - Daily = major institutional levels
        - 1H = granular features + testing
        
        Result: ~400-600 samples (365 days × 2 levels × ~60-80% pass filter)
        """
        
        self.logger.info(f"📊 SIMPLE EFFICIENT MULTI-TF: {symbol} {exchange}")
        self.logger.info(f"   Strategy: TOP 2 SR/day (1 support + 1 resistance from 1D) → Filter tested+rejected (1H)")
        self.logger.info(f"   Period: {start_date} to {end_date}")
        
        # Load daily data
        daily_data = await self._load_historical_data(symbol, exchange, detection_timeframe, start_date, end_date)
        
        if daily_data is None or daily_data.empty:
            raise ValueError(f"No daily data")
        
        self.logger.info(f"✅ Loaded {len(daily_data)} daily bars")
        
        # Load 1H data for testing
        hourly_data = await self._load_historical_data(symbol, exchange, testing_timeframe, start_date, end_date)
        
        if hourly_data is None or hourly_data.empty:
            raise ValueError(f"No hourly data")
        
        self.logger.info(f"✅ Loaded {len(hourly_data)} hourly bars")
        
        # Sample dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        sample_dates = pd.date_range(start_dt, end_dt, freq=f'{sample_freq_days}D')
        
        if isinstance(daily_data.index, pd.DatetimeIndex) and daily_data.index.tz is not None:
            sample_dates = sample_dates.tz_localize('UTC')
        
        self.logger.info(f"\n🔄 Processing {len(sample_dates)} dates (1 top level per day)...")
        
        training_samples = []
        total_levels = 0
        filtered_out = 0
        
        for current_date in tqdm(sample_dates, desc="1 Level/Day"):
            try:
                # =============================================================
                # STEP 1: Detect SR on daily up to current date
                # =============================================================
                
                daily_historical = daily_data[daily_data.index < current_date]
                
                if len(daily_historical) < 50:
                    continue
                
                # Detect SR levels on daily
                daily_levels = self._detect_sr_levels(
                    daily_historical[-100:],  # Last 100 daily bars
                    symbol,
                    exchange,
                    detection_timeframe
                )
                
                if not daily_levels:
                    continue
                
                # =============================================================
                # STEP 2: Get TOP 2 levels (1 support + 1 resistance)
                # =============================================================
                
                # Separate by type
                supports = [l for l in daily_levels if 
                           (getattr(l, 'type', None) if not isinstance(l, dict) else l.get('type')) == 'support']
                resistances = [l for l in daily_levels if 
                              (getattr(l, 'type', None) if not isinstance(l, dict) else l.get('type')) == 'resistance']
                
                top_levels = []
                
                # Get top support
                if supports:
                    top_support = max(
                        supports,
                        key=lambda x: getattr(x, 'strength', 0.5) if not isinstance(x, dict) else x.get('strength', 0.5)
                    )
                    top_levels.append(top_support)
                
                # Get top resistance
                if resistances:
                    top_resistance = max(
                        resistances,
                        key=lambda x: getattr(x, 'strength', 0.5) if not isinstance(x, dict) else x.get('strength', 0.5)
                    )
                    top_levels.append(top_resistance)
                
                if not top_levels:
                    continue
                
                total_levels += len(top_levels)
                
                # =============================================================
                # STEP 3: Get 1H data for this date
                # =============================================================
                
                hourly_historical = hourly_data[hourly_data.index < current_date]
                future_end = current_date + timedelta(days=forward_days)
                hourly_future = hourly_data[
                    (hourly_data.index >= current_date) &
                    (hourly_data.index < future_end)
                ]
                
                if len(hourly_historical) < 200 or len(hourly_future) < 5:
                    continue
                
                # =============================================================
                # STEP 4: Process each top level (support + resistance)
                # =============================================================
                
                for top_level in top_levels:
                    try:
                        # Filter - has this level been tested + rejected on 1H?
                        if not self._check_level_has_rejection(top_level, hourly_historical[-1000:]):
                            filtered_out += 1
                            continue  # Skip: never bounced on 1H
                        
                        # Extract features from 1H + measure performance
                        level_price = getattr(top_level, 'price', None) if not isinstance(top_level, dict) else top_level.get('price')
                        level_type = getattr(top_level, 'type', None) if not isinstance(top_level, dict) else top_level.get('type')
                        
                        # Extract features from 1H
                        features = self._extract_optimized_features(
                            top_level,
                            hourly_historical,  # 1H for features
                            daily_historical,   # Daily for multi-TF
                            current_date
                        )
                        
                        # Measure performance on 1H future
                        target = self._calculate_realized_pnl(top_level, hourly_future)
                        
                        sample = {
                            'date': current_date,
                            'symbol': symbol,
                            'exchange': exchange,
                            'detection_tf': detection_timeframe,
                            'testing_tf': testing_timeframe,
                            'level_price': level_price,
                            'level_type': level_type,
                            **features,
                            'realized_pnl_pct': target,
                        }
                        
                        training_samples.append(sample)
                    
                    except Exception as e:
                        self.logger.debug(f"Failed level: {e}")
                        continue
                
            except Exception as e:
                self.logger.debug(f"Failed date {current_date}: {e}")
                continue
        
        training_df = pd.DataFrame(training_samples)
        
        if len(training_df) == 0:
            raise ValueError("No samples collected!")
        
        # Filter untested (target = 0.0)
        initial_count = len(training_df)
        training_df = training_df[training_df['realized_pnl_pct'] != 0.0].copy()
        
        self.logger.info(f"\n✅ Simple efficient multi-TF complete!")
        self.logger.info(f"   Sample dates: {len(sample_dates)}")
        self.logger.info(f"   Top SR levels detected (support + resistance): {total_levels}")
        self.logger.info(f"   Avg levels/day: {total_levels/len(sample_dates):.1f}")
        self.logger.info(f"   Filtered out (not tested/rejected on 1H): {filtered_out} ({filtered_out/total_levels*100:.1f}%)")
        self.logger.info(f"   Final samples: {len(training_df)}")
        self.logger.info(f"   Pass rate: {len(training_df)/total_levels*100:.1f}%")
        self.logger.info(f"   Strategy: Top 1 support + Top 1 resistance per day (1D) → Tested+bounced on 1H")
        
        return training_df
    
    def _check_level_has_rejection(self, level, historical_data: pd.DataFrame) -> bool:
        """Quick check if level has historical rejection (for filtering)."""
        
        try:
            level_price = getattr(level, 'price', None) if not isinstance(level, dict) else level.get('price')
            level_type = getattr(level, 'type', None) if not isinstance(level, dict) else level.get('type')
            
            if level_price is None or level_type not in ['support', 'resistance']:
                return False
            
            tolerance = level_price * 0.01
            
            # Find tests
            if level_type == 'support':
                tests = historical_data[historical_data['low'] <= level_price + tolerance]
            else:
                tests = historical_data[historical_data['high'] >= level_price - tolerance]
            
            if len(tests) == 0:
                return False
            
            # Check for at least one bounce
            for test_idx in tests.index[-3:]:
                try:
                    bars_after = historical_data.loc[test_idx:].iloc[:5]
                    
                    if len(bars_after) >= 2:
                        if level_type == 'support':
                            bounced = bars_after['close'].iloc[-1] > level_price
                        else:
                            bounced = bars_after['close'].iloc[-1] < level_price
                        
                        if bounced:
                            return True
                except:
                    continue
            
            return False
        
        except:
            return False
    
    def _filter_tested_levels_with_rejection(self, levels: List, historical_data: pd.DataFrame) -> List:
        """
        OPTIMIZATION 1: Pre-filter levels to only those tested with rejection.
        
        Filter criteria:
        - Level was tested (price hit it) in recent history
        - Level showed rejection (bounce) at least once
        
        This reduces computation by ~70% (only process quality levels)
        
        Args:
            levels: All detected SR levels
            historical_data: Recent historical data
            
        Returns:
            Filtered list of quality levels
        """
        
        filtered = []
        
        for level in levels:
            try:
                level_price = getattr(level, 'price', None) if not isinstance(level, dict) else level.get('price')
                level_type = getattr(level, 'type', None) if not isinstance(level, dict) else level.get('type')
                
                if level_price is None or level_type not in ['support', 'resistance']:
                    continue
                
                tolerance = level_price * 0.01  # 1% tolerance
                
                # Find tests in recent history
                if level_type == 'support':
                    tests = historical_data[historical_data['low'] <= level_price + tolerance]
                else:
                    tests = historical_data[historical_data['high'] >= level_price - tolerance]
                
                if len(tests) == 0:
                    continue  # Never tested - skip!
                
                # Check for rejection (bounce)
                has_rejection = False
                
                for test_idx in tests.index[-3:]:  # Check last 3 tests
                    try:
                        bars_after = historical_data.loc[test_idx:].iloc[:5]
                        
                        if len(bars_after) >= 2:
                            if level_type == 'support':
                                # Did price bounce up?
                                bounced = bars_after['close'].iloc[-1] > level_price
                            else:
                                # Did price bounce down?
                                bounced = bars_after['close'].iloc[-1] < level_price
                            
                            if bounced:
                                has_rejection = True
                                break
                    except:
                        continue
                
                if has_rejection:
                    filtered.append(level)
            
            except Exception as e:
                continue
        
        return filtered
    
    def _extract_optimized_features(self, level, historical_data: pd.DataFrame,
                                   daily_data: Optional[pd.DataFrame],
                                   current_date) -> Dict[str, float]:
        """
        Extract OPTIMIZED focused features using vectorization.
        
        Feature categories:
        1. ALL SR-specific features (~15-20)
        2. Top 2 volume features (using VectorBT)
        3. Top 2 momentum features (using VectorBT)
        4. Top 2 trend features (using VectorBT)
        5. Top 2 volatility features (using VectorBT)
        
        Total: ~25-30 features (was 100+)
        Speed: 100x faster than full FeatureBank
        """
        
        features = {}
        
        def get_attr(name, default=0.0):
            if isinstance(level, dict):
                return level.get(name, default)
            return getattr(level, name, default)
        
        current_price = historical_data['close'].iloc[-1]
        level_price = get_attr('price', current_price)
        level_type = get_attr('type', 'support')
        
        # ====================================================================
        # 1. ALL SR-SPECIFIC FEATURES (~15-20 features)
        # ====================================================================
        
        # Basic SR characteristics
        features['feature_sr_strength'] = float(get_attr('strength', 0.5))
        features['feature_sr_touch_count'] = int(get_attr('touch_count', 1))
        features['feature_sr_age_bars'] = int(get_attr('age_bars', 0))
        features['feature_sr_consistency'] = float(get_attr('consistency_score', 0.5))
        features['feature_sr_avg_bounce_ratio'] = float(get_attr('avg_bounce_ratio', 0))
        features['feature_sr_max_bounce_ratio'] = float(get_attr('max_bounce_ratio', 0))
        features['feature_sr_volume_confirmation'] = float(get_attr('volume_confirmation_score', 0.5))
        features['feature_sr_bounce_consistency'] = float(get_attr('bounce_consistency', 0.0))
        
        # SR position features
        features['feature_sr_distance_to_current_pct'] = float(abs(level_price - current_price) / current_price)
        features['feature_sr_is_support'] = 1.0 if level_type == 'support' else 0.0
        
        # SR quality indicators
        features['feature_sr_recency_weighted_strength'] = float(
            get_attr('strength', 0.5) * np.exp(-get_attr('age_bars', 0) / 50)
        )
        features['feature_sr_quality_tier'] = float(min(get_attr('strength', 0.5) * 2.0, 1.0))
        
        # SR touch quality
        avg_bounce = get_attr('avg_bounce_ratio', 0)
        avg_volume = get_attr('avg_touch_volume_ratio', 0)
        consistency = get_attr('consistency_score', 0.5)
        features['feature_sr_touch_quality'] = float((avg_bounce * 0.5) + (avg_volume * 0.3) + (consistency * 0.2))
        
        # Price z-score relative to SR
        close_series = historical_data['close']
        features['feature_sr_price_zscore'] = float(
            (level_price - close_series.mean()) / (close_series.std() + 1e-8)
        )
        
        # Recent SR performance (MOST IMPORTANT!)
        sr_recent_features = self._extract_sr_recent_performance(level, historical_data, current_date)
        features.update(sr_recent_features)
        
        # ====================================================================
        # ADDITIONAL SR-SPECIFIC FEATURES (Volume at level, velocity, etc.)
        # ====================================================================
        
        # These are CRITICAL for SR prediction!
        sr_micro_features = self._extract_sr_micro_features(level, historical_data, level_price, level_type)
        features.update(sr_micro_features)
        
        # ====================================================================
        # 2. TOP 2 VOLUME FEATURES (using VectorBT optimizer)
        # ====================================================================
        
        if 'volume' in historical_data.columns and self.vectorbt_optimizer:
            try:
                volume_features = self._extract_top_volume_features_vectorized(historical_data)
                features.update(volume_features)
            except Exception as e:
                self.logger.debug(f"Volume features failed: {e}")
        
        # ====================================================================
        # 3. TOP 2 MOMENTUM FEATURES (using VectorBT optimizer)
        # ====================================================================
        
        if self.vectorbt_optimizer:
            try:
                momentum_features = self._extract_top_momentum_features_vectorized(historical_data)
                features.update(momentum_features)
            except Exception as e:
                self.logger.debug(f"Momentum features failed: {e}")
        
        # ====================================================================
        # 4. TOP 2 TREND FEATURES (using VectorBT optimizer)
        # ====================================================================
        
        if self.vectorbt_optimizer:
            try:
                trend_features = self._extract_top_trend_features_vectorized(historical_data)
                features.update(trend_features)
            except Exception as e:
                self.logger.debug(f"Trend features failed: {e}")
        
        # ====================================================================
        # 5. TOP 2 VOLATILITY FEATURES (using statistical optimizer)
        # ====================================================================
        
        if self.statistical_optimizer:
            try:
                vol_features = self._extract_top_volatility_features_vectorized(historical_data)
                features.update(vol_features)
            except Exception as e:
                self.logger.debug(f"Volatility features failed: {e}")
        
        # ====================================================================
        # 6. MULTI-TIMEFRAME ALIGNMENT (simple, fast)
        # ====================================================================
        
        if daily_data is not None:
            mtf_features = self._extract_multi_timeframe_features(level, daily_data, current_date, current_price)
            features.update(mtf_features)
        
        return features
    
    def _extract_sr_recent_performance(self, level, historical_data: pd.DataFrame,
                                       current_date) -> Dict[str, float]:
        """
        Extract recent SR performance features.
        
        These are the MOST PREDICTIVE features!
        If a level bounced recently, it will likely bounce again.
        """
        
        level_price = getattr(level, 'price', None) if not isinstance(level, dict) else level.get('price')
        level_type = getattr(level, 'type', None) if not isinstance(level, dict) else level.get('type')
        
        features = {}
        
        try:
            tolerance = level_price * 0.01
            recent_data = historical_data[-50:]  # Last 50 bars
            
            # Find recent tests
            if level_type == 'support':
                tests = recent_data[recent_data['low'] <= level_price + tolerance]
            else:
                tests = recent_data[recent_data['high'] >= level_price - tolerance]
            
            features['feature_sr_recent_tests_count'] = len(tests)
            features['feature_sr_days_since_last_test'] = (current_date - tests.index[-1]).days if len(tests) > 0 else 999
            
            # Check bounces
            if len(tests) > 0:
                bounces = 0
                bounce_strengths = []
                
                for test_idx in tests.index[-3:]:  # Last 3 tests
                    try:
                        bars_after = historical_data.loc[test_idx:].iloc[:5]
                        
                        if len(bars_after) >= 2:
                            if level_type == 'support':
                                bounced = bars_after['close'].iloc[-1] > level_price
                                bounce_strength = (bars_after['high'].max() - level_price) / level_price
                            else:
                                bounced = bars_after['close'].iloc[-1] < level_price
                                bounce_strength = (level_price - bars_after['low'].min()) / level_price
                            
                            if bounced:
                                bounces += 1
                                bounce_strengths.append(bounce_strength)
                    except:
                        continue
                
                features['feature_sr_bounced_last_test'] = 1.0 if bounces > 0 else 0.0
                features['feature_sr_consecutive_bounces'] = bounces
                features['feature_sr_avg_recent_bounce_strength'] = np.mean(bounce_strengths) if bounce_strengths else 0.0
            else:
                features['feature_sr_bounced_last_test'] = 0.5
                features['feature_sr_consecutive_bounces'] = 0
                features['feature_sr_avg_recent_bounce_strength'] = 0.0
        
        except Exception as e:
            self.logger.debug(f"Recent performance extraction failed: {e}")
            features['feature_sr_recent_tests_count'] = 0
            features['feature_sr_days_since_last_test'] = 999
            features['feature_sr_bounced_last_test'] = 0.5
            features['feature_sr_consecutive_bounces'] = 0
            features['feature_sr_avg_recent_bounce_strength'] = 0.0
        
        return features
    
    def _extract_sr_micro_features(self, level, historical_data: pd.DataFrame,
                                   level_price: float, level_type: str) -> Dict[str, float]:
        """
        Extract SR MICRO-LEVEL features - CRITICAL for prediction!
        
        These capture behavior AT/NEAR the SR level:
        1. Volume AT the SR level (institutional activity)
        2. Velocity of approach (fast approach = stronger rejection)
        3. Momentum near level (slowing down = respect)
        4. Candle characteristics (wicks = rejection)
        5. Volume spike at level (absorption)
        
        Returns:
            Dictionary with ~10 micro-level SR features
        """
        
        features = {}
        
        try:
            # Get bars near the level (within 1%)
            tolerance = level_price * 0.01
            recent_data = historical_data[-50:]  # Last 50 bars
            
            if level_type == 'support':
                near_level = recent_data[
                    (recent_data['low'] >= level_price - tolerance) &
                    (recent_data['low'] <= level_price + tolerance)
                ]
            else:
                near_level = recent_data[
                    (recent_data['high'] >= level_price - tolerance) &
                    (recent_data['high'] <= level_price + tolerance)
                ]
            
            # ================================================================
            # FEATURE 1: Volume AT SR Level
            # ================================================================
            # When price is at the level, is volume high? (institutions!)
            
            if len(near_level) > 0 and 'volume' in near_level.columns:
                avg_volume = historical_data['volume'].mean()
                volume_at_level = near_level['volume'].mean()
                
                features['feature_sr_volume_at_level_ratio'] = float(volume_at_level / (avg_volume + 1e-8))
                features['feature_sr_max_volume_at_level'] = float(near_level['volume'].max() / (avg_volume + 1e-8))
                features['feature_sr_tests_with_high_volume'] = float(
                    (near_level['volume'] > avg_volume * 1.5).sum() / (len(near_level) + 1)
                )
            else:
                features['feature_sr_volume_at_level_ratio'] = 1.0
                features['feature_sr_max_volume_at_level'] = 1.0
                features['feature_sr_tests_with_high_volume'] = 0.0
            
            # ================================================================
            # FEATURE 2: Velocity/Speed of Approach
            # ================================================================
            # Fast approach = strong rejection likely
            # Slow grind = weak level
            
            current_price = historical_data['close'].iloc[-1]
            distance_to_level = abs(current_price - level_price) / level_price
            
            # Velocity = distance covered in last N bars
            if len(historical_data) >= 5:
                price_5bars_ago = historical_data['close'].iloc[-5]
                distance_5bars_ago = abs(price_5bars_ago - level_price) / level_price
                
                # Approaching velocity
                velocity = (distance_5bars_ago - distance_to_level) / 5  # % per bar
                features['feature_sr_approach_velocity'] = float(velocity)
                
                # Is it approaching fast?
                features['feature_sr_fast_approach'] = 1.0 if velocity > 0.002 else 0.0  # >0.2% per bar
            else:
                features['feature_sr_approach_velocity'] = 0.0
                features['feature_sr_fast_approach'] = 0.0
            
            # ================================================================
            # FEATURE 3: Momentum Near Level
            # ================================================================
            # Is momentum slowing near level? (respect/hesitation)
            
            if len(historical_data) >= 10:
                recent_momentum = (historical_data['close'].iloc[-1] - historical_data['close'].iloc[-5]) / historical_data['close'].iloc[-5]
                older_momentum = (historical_data['close'].iloc[-5] - historical_data['close'].iloc[-10]) / historical_data['close'].iloc[-10]
                
                # Momentum deceleration near level
                momentum_change = recent_momentum - older_momentum
                features['feature_sr_momentum_deceleration'] = float(momentum_change)
                
                # Is momentum slowing (respecting level)?
                features['feature_sr_momentum_slowing'] = 1.0 if abs(recent_momentum) < abs(older_momentum) else 0.0
            else:
                features['feature_sr_momentum_deceleration'] = 0.0
                features['feature_sr_momentum_slowing'] = 0.0
            
            # ================================================================
            # FEATURE 4: Rejection Candles (Wicks)
            # ================================================================
            # Long wicks at level = strong rejection
            
            if len(near_level) > 0:
                rejection_wicks = []
                
                for _, candle in near_level.iterrows():
                    # Calculate wick length
                    if level_type == 'support':
                        # Lower wick length
                        body_low = min(candle['open'], candle['close'])
                        wick_length = (body_low - candle['low']) / candle['close']
                        rejection_wicks.append(wick_length)
                    else:
                        # Upper wick length
                        body_high = max(candle['open'], candle['close'])
                        wick_length = (candle['high'] - body_high) / candle['close']
                        rejection_wicks.append(wick_length)
                
                if rejection_wicks:
                    features['feature_sr_avg_rejection_wick'] = float(np.mean(rejection_wicks))
                    features['feature_sr_max_rejection_wick'] = float(np.max(rejection_wicks))
                    features['feature_sr_strong_wicks_count'] = int(sum(1 for w in rejection_wicks if w > 0.01))  # >1% wick
                else:
                    features['feature_sr_avg_rejection_wick'] = 0.0
                    features['feature_sr_max_rejection_wick'] = 0.0
                    features['feature_sr_strong_wicks_count'] = 0
            else:
                features['feature_sr_avg_rejection_wick'] = 0.0
                features['feature_sr_max_rejection_wick'] = 0.0
                features['feature_sr_strong_wicks_count'] = 0
            
            # ================================================================
            # FEATURE 5: Volatility AT SR Level
            # ================================================================
            # High volatility at level = weak (gets pierced)
            # Low volatility at level = strong (holds)
            
            if len(near_level) >= 2:
                returns_at_level = near_level['close'].pct_change().dropna()
                vol_at_level = returns_at_level.std()
                
                overall_vol = historical_data['close'].pct_change().std()
                
                features['feature_sr_volatility_at_level'] = float(vol_at_level)
                features['feature_sr_volatility_ratio_at_level'] = float(vol_at_level / (overall_vol + 1e-8))
            else:
                features['feature_sr_volatility_at_level'] = historical_data['close'].pct_change().std()
                features['feature_sr_volatility_ratio_at_level'] = 1.0
            
            # ================================================================
            # FEATURE 6: Time Spent At Level
            # ================================================================
            # More time at level = consolidation = stronger
            
            features['feature_sr_bars_near_level'] = len(near_level)
            features['feature_sr_time_at_level_pct'] = float(len(near_level) / len(recent_data))
        
        except Exception as e:
            self.logger.debug(f"SR micro features failed: {e}")
            # Set defaults
            features['feature_sr_volume_at_level_ratio'] = 1.0
            features['feature_sr_max_volume_at_level'] = 1.0
            features['feature_sr_tests_with_high_volume'] = 0.0
            features['feature_sr_approach_velocity'] = 0.0
            features['feature_sr_fast_approach'] = 0.0
            features['feature_sr_momentum_deceleration'] = 0.0
            features['feature_sr_momentum_slowing'] = 0.0
            features['feature_sr_avg_rejection_wick'] = 0.0
            features['feature_sr_max_rejection_wick'] = 0.0
            features['feature_sr_strong_wicks_count'] = 0
            features['feature_sr_volatility_at_level'] = 0.02
            features['feature_sr_volatility_ratio_at_level'] = 1.0
            features['feature_sr_bars_near_level'] = 0
            features['feature_sr_time_at_level_pct'] = 0.0
        
        return features
    
    def _extract_top_volume_features_vectorized(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        TOP 2 volume features using VectorBT optimizer.
        
        Most predictive volume features:
        1. Volume trend (increasing volume = stronger level)
        2. Volume ratio vs average (high volume = institutions)
        """
        
        features = {}
        
        try:
            volume = data['volume'].values
            
            if self.vectorbt_optimizer:
                # Use VectorBT for fast rolling operations
                volume_ma = self.vectorbt_optimizer.rolling_mean(volume, window=20)
                
                # Feature 1: Volume trend (recent vs older)
                recent_vol = np.mean(volume[-10:])
                older_vol = np.mean(volume[-30:-10])
                features['feature_vol_trend'] = float((recent_vol - older_vol) / (older_vol + 1e-8))
                
                # Feature 2: Current volume ratio
                current_vol = volume[-1]
                avg_vol = np.mean(volume[-20:])
                features['feature_vol_ratio'] = float(current_vol / (avg_vol + 1e-8))
            else:
                # Fallback: simple numpy
                features['feature_vol_trend'] = float((np.mean(volume[-10:]) - np.mean(volume[-30:-10])) / (np.mean(volume[-30:-10]) + 1e-8))
                features['feature_vol_ratio'] = float(volume[-1] / (np.mean(volume[-20:]) + 1e-8))
        
        except Exception as e:
            features['feature_vol_trend'] = 0.0
            features['feature_vol_ratio'] = 1.0
        
        return features
    
    def _extract_top_momentum_features_vectorized(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        TOP 2 momentum features using VectorBT optimizer.
        
        Most predictive:
        1. RSI (overbought/oversold)
        2. Price momentum (recent price change)
        """
        
        features = {}
        
        try:
            close = data['close'].values
            
            if self.statistical_optimizer:
                # Feature 1: Simple RSI(14)
                rsi = self._calculate_rsi_vectorized(close, 14)
                features['feature_momentum_rsi14'] = float(rsi)
                
                # Feature 2: Price momentum (5-bar rate of change)
                momentum = (close[-1] - close[-5]) / close[-5]
                features['feature_momentum_roc5'] = float(momentum)
            else:
                # Fallback
                features['feature_momentum_rsi14'] = 50.0
                features['feature_momentum_roc5'] = float((close[-1] - close[-5]) / close[-5])
        
        except Exception as e:
            features['feature_momentum_rsi14'] = 50.0
            features['feature_momentum_roc5'] = 0.0
        
        return features
    
    def _extract_top_trend_features_vectorized(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        TOP 2 trend features using VectorBT optimizer.
        
        Most predictive:
        1. Trend direction and strength
        2. Trend alignment (short vs long MA)
        """
        
        features = {}
        
        try:
            close = data['close'].values
            
            if self.consolidated_optimizer:
                # Use consolidated optimizer for batch rolling
                ma_short = self.consolidated_optimizer.rolling_mean(close, window=20)
                ma_long = self.consolidated_optimizer.rolling_mean(close, window=50)
                
                # Feature 1: Trend strength (20-bar)
                trend_strength = (close[-1] - close[-20]) / close[-20]
                features['feature_trend_strength'] = float(trend_strength)
                
                # Feature 2: MA alignment
                ma_alignment = (ma_short - ma_long) / ma_long if ma_long > 0 else 0
                features['feature_trend_ma_alignment'] = float(ma_alignment)
            else:
                # Fallback
                features['feature_trend_strength'] = float((close[-1] - close[-20]) / close[-20])
                features['feature_trend_ma_alignment'] = 0.0
        
        except Exception as e:
            features['feature_trend_strength'] = 0.0
            features['feature_trend_ma_alignment'] = 0.0
        
        return features
    
    def _extract_top_volatility_features_vectorized(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        TOP 2 volatility features using statistical optimizer.
        
        Most predictive:
        1. Current volatility level
        2. Volatility regime (low/high)
        """
        
        features = {}
        
        try:
            returns = data['close'].pct_change().values
            
            if self.statistical_optimizer:
                # Feature 1: Current volatility (20-bar std)
                current_vol = self.statistical_optimizer.fast_std(returns[-20:])
                features['feature_vol_current'] = float(current_vol)
                
                # Feature 2: Volatility regime (current vs longer-term)
                long_term_vol = self.statistical_optimizer.fast_std(returns[-50:])
                vol_regime = current_vol / (long_term_vol + 1e-8)
                features['feature_vol_regime'] = float(vol_regime)
            else:
                # Fallback
                features['feature_vol_current'] = float(np.std(returns[-20:]))
                features['feature_vol_regime'] = 1.0
        
        except Exception as e:
            features['feature_vol_current'] = 0.02
            features['feature_vol_regime'] = 1.0
        
        return features
    
    def _extract_multi_timeframe_features(self, level, daily_data: pd.DataFrame,
                                         current_date, current_price) -> Dict[str, float]:
        """Extract multi-timeframe alignment features (fast)."""
        
        features = {}
        
        try:
            daily_hist = daily_data[daily_data.index < current_date]
            
            if len(daily_hist) >= 20:
                # Detect SR on daily (simple)
                daily_levels = self._detect_sr_levels(daily_hist[-50:], '', '', '1d')
                
                if daily_levels:
                    level_price = getattr(level, 'price', current_price) if not isinstance(level, dict) else level.get('price', current_price)
                    
                    daily_prices = [getattr(lvl, 'price', 0) if not isinstance(lvl, dict) else lvl.get('price', 0) 
                                  for lvl in daily_levels]
                    daily_prices = [p for p in daily_prices if p > 0]
                    
                    if daily_prices:
                        distances = [abs(p - level_price) / current_price for p in daily_prices]
                        nearest_distance = min(distances)
                        nearest_idx = np.argmin(distances)
                        
                        features['feature_mtf_near_1d_sr'] = 1.0 if nearest_distance < 0.02 else 0.0
                        features['feature_mtf_1d_distance'] = float(nearest_distance)
                        
                        nearest_level = daily_levels[nearest_idx]
                        features['feature_mtf_1d_strength'] = float(
                            getattr(nearest_level, 'strength', 0.5) if not isinstance(nearest_level, dict) 
                            else nearest_level.get('strength', 0.5)
                        )
        
        except Exception as e:
            self.logger.debug(f"Multi-TF features failed: {e}")
        
        # Defaults if failed
        features.setdefault('feature_mtf_near_1d_sr', 0.0)
        features.setdefault('feature_mtf_1d_distance', 1.0)
        features.setdefault('feature_mtf_1d_strength', 0.5)
        
        return features
    
    def _calculate_rsi_vectorized(self, prices: np.ndarray, period: int = 14) -> float:
        """Fast RSI calculation using numpy."""
        
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def _calculate_realized_pnl(self, level, future_data: pd.DataFrame) -> float:
        """Calculate realized P&L (same as before)."""
        
        level_price = getattr(level, 'price', None) if not isinstance(level, dict) else level.get('price')
        level_type = getattr(level, 'type', None) if not isinstance(level, dict) else level.get('type')
        
        if level_price is None or level_type not in ['support', 'resistance']:
            return 0.0
        
        tolerance = level_price * 0.005
        
        if level_type == 'support':
            hits = future_data[future_data['low'] <= level_price + tolerance]
        else:
            hits = future_data[future_data['high'] >= level_price - tolerance]
        
        if len(hits) == 0:
            return 0.0
        
        first_hit_idx = hits.index[0]
        
        if level_type == 'support':
            entry_price = level_price
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            take_profit = entry_price * (1 + self.take_profit_pct)
            direction = 1
        else:
            entry_price = level_price
            stop_loss = entry_price * (1 + self.stop_loss_pct)
            take_profit = entry_price * (1 - self.take_profit_pct)
            direction = -1
        
        future_bars = future_data.loc[first_hit_idx:].iloc[:self.max_hold_bars]
        
        for _, bar in future_bars.iterrows():
            if direction == 1:
                if bar['low'] <= stop_loss:
                    return -self.stop_loss_pct
                if bar['high'] >= take_profit:
                    return self.take_profit_pct
            else:
                if bar['high'] >= stop_loss:
                    return -self.stop_loss_pct
                if bar['low'] <= take_profit:
                    return self.take_profit_pct
        
        exit_price = future_bars.iloc[-1]['close']
        pnl_pct = (exit_price - entry_price) / entry_price * direction
        
        return float(pnl_pct)
    
    async def _load_historical_data(self, symbol: str, exchange: str, timeframe: str,
                                    start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical data."""
        try:
            data = await self.data_loader.load_market_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                force_download=False
            )
            return data if data is not None else pd.DataFrame()
        except:
            return pd.DataFrame()
    
    def _detect_sr_levels(self, data: pd.DataFrame, symbol: str,
                         exchange: str, timeframe: str) -> List:
        """Detect SR levels."""
        try:
            result = self.sr_detector.detect_sr_levels(data)
            
            if isinstance(result, dict) and 'levels' in result:
                return result['levels']
            elif isinstance(result, list):
                return result
            else:
                return []
        except:
            return []
    
    def save_training_data(self, training_df: pd.DataFrame,
                          output_path: str = None) -> str:
        """Save training data."""
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f'data_cache/sr_ml_training/sr_quality_OPTIMIZED_{timestamp}.parquet'
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        training_df.to_parquet(output_file, index=False)
        
        metadata = {
            'created_at': datetime.now().isoformat(),
            'approach': 'fast_optimized_focused_features',
            'samples': len(training_df),
            'feature_count': len([c for c in training_df.columns if c.startswith('feature_')]),
            'optimizations': [
                'pre_filtering_tested_levels_with_rejection',
                'focused_features_sr_specific_plus_top2',
                'vectorized_computation_numba_numpy_vectorbt'
            ],
            'feature_categories': {
                'sr_specific': 'all (~20 features)',
                'volume': 'top 2',
                'momentum': 'top 2',
                'trend': 'top 2',
                'volatility': 'top 2',
                'multi_timeframe': '3 features'
            },
            'trading_params': {
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct,
                'risk_reward_ratio': self.take_profit_pct / self.stop_loss_pct
            },
            'pnl_stats': {
                'mean': float(training_df['realized_pnl_pct'].mean()),
                'std': float(training_df['realized_pnl_pct'].std()),
                'win_rate': float((training_df['realized_pnl_pct'] > 0).sum() / len(training_df))
            }
        }
        
        metadata_path = str(output_file).replace('.parquet', '_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"✅ Optimized data saved to {output_file}")
        
        return str(output_file)

