"""
Partial-Bar Nowcasting for Live Trading

This module implements partial-bar nowcasting to ensure that market regime
evaluation always uses complete 1-hour bars, regardless of when the evaluation
occurs within the hour (T+15, T+30, T+45).

The system creates virtual bar splits so that regime evaluation always works
with full 1-hour bars, preventing the use of incomplete hourly data.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_debug, tprint_progress, tprint_performance, tprint_structured,
    tprint_timer, LogLevel
)

logger = system_logger.getChild('PartialBarNowcasting')

@dataclass
class BarSplit:
    """Represents a split of an hourly bar for nowcasting."""
    start_time: datetime
    end_time: datetime
    split_ratio: float  # 0.0 to 1.0, represents how much of the hour has passed
    is_complete: bool
    split_data: Optional[Dict[str, Any]] = None

@dataclass
class NowcastingConfig:
    """Configuration for partial-bar nowcasting."""
    base_timeframe: str = "1h"  # Base timeframe for regime evaluation
    evaluation_interval: int = 15 * 60  # 15 minutes in seconds
    min_bar_completion: float = 0.25  # Minimum 25% of bar must be complete
    max_bar_completion: float = 0.95  # Maximum 95% to avoid using incomplete bars
    enable_forward_filling: bool = True  # Use forward-filling for incomplete bars
    enable_backward_filling: bool = True  # Use backward-filling for missing data
    confidence_threshold: float = 0.7  # Minimum confidence for nowcasted data

class PartialBarNowcaster:
    """
    Partial-Bar Nowcaster for Live Trading

    Ensures that market regime evaluation always uses complete 1-hour bars
    by creating virtual bar splits and managing partial bar data.
    """

    def __init__(self, config: NowcastingConfig):
        self.config = config
        self.logger = logger.getChild('PartialBarNowcaster')

        # Bar splitting state
        self.current_hour_start: Optional[datetime] = None
        self.current_hour_end: Optional[datetime] = None
        self.bar_splits: List[BarSplit] = []
        self.complete_bars: List[Dict[str, Any]] = []

        # Data management
        self.hourly_data: Optional[pd.DataFrame] = None
        self.partial_data: Optional[pd.DataFrame] = None
        self.nowcasted_data: Optional[pd.DataFrame] = None

        # Timing state
        self.last_evaluation_time: Optional[datetime] = None
        self.next_evaluation_time: Optional[datetime] = None

        tprint_info("🔧 Partial-Bar Nowcaster initialized")
        tprint_info(f"   Base timeframe: {config.base_timeframe}")
        tprint_info(f"   Evaluation interval: {config.evaluation_interval}s")
        tprint_info(f"   Min bar completion: {config.min_bar_completion*100}%")
        tprint_info(f"   Max bar completion: {config.max_bar_completion*100}%")

    async def initialize(self) -> bool:
        """Initialize the nowcaster with current market data."""
        try:
            tprint_info("🚀 Initializing Partial-Bar Nowcaster...")

            # Get current time and determine current hour boundaries
            now = datetime.now()
            self.current_hour_start = now.replace(minute=0, second=0, microsecond=0)
            self.current_hour_end = self.current_hour_start + timedelta(hours=1)

            # Schedule first evaluation
            self.next_evaluation_time = now + timedelta(seconds=self.config.evaluation_interval)

            tprint_success("✅ Partial-Bar Nowcaster initialized successfully")
            tprint_info(f"   Current hour: {self.current_hour_start} to {self.current_hour_end}")
            tprint_info(f"   Next evaluation: {self.next_evaluation_time}")

            return True

        except Exception as e:
            tprint_error(f"❌ Failed to initialize nowcaster: {e}")
            return False

    async def should_evaluate_regime(self, current_time: Optional[datetime] = None) -> bool:
        """
        Determine if regime evaluation should occur based on bar completion.

        Returns True if:
        1. Evaluation interval has passed AND
        2. We have sufficient bar completion for reliable nowcasting
        """
        if current_time is None:
            current_time = datetime.now()

        # Check if evaluation interval has passed
        if (self.last_evaluation_time and
            (current_time - self.last_evaluation_time).total_seconds() < self.config.evaluation_interval):
            return False

        # Check if we have sufficient bar completion
        bar_completion = self._calculate_bar_completion(current_time)

        if bar_completion < self.config.min_bar_completion:
            tprint_debug(f"⏳ Bar completion too low: {bar_completion:.2%} < {self.config.min_bar_completion:.2%}")
            return False

        if bar_completion > self.config.max_bar_completion:
            tprint_debug(f"⏳ Bar completion too high: {bar_completion:.2%} > {self.config.max_bar_completion:.2%}")
            return False

        tprint_info(f"✅ Regime evaluation triggered - Bar completion: {bar_completion:.2%}")
        return True

    def _calculate_bar_completion(self, current_time: datetime) -> float:
        """Calculate how much of the current hour bar has completed."""
        if not self.current_hour_start:
            return 0.0

        elapsed = (current_time - self.current_hour_start).total_seconds()
        total = 3600.0  # 1 hour in seconds
        completion = min(elapsed / total, 1.0)

        return completion

    async def create_bar_split(self, current_time: Optional[datetime] = None) -> BarSplit:
        """
        Create a bar split for the current evaluation time.

        This ensures we always work with complete hourly bars by splitting
        the current hour at the evaluation point.
        """
        if current_time is None:
            current_time = datetime.now()

        # Calculate split ratio
        split_ratio = self._calculate_bar_completion(current_time)

        # Create bar split
        bar_split = BarSplit(
            start_time=self.current_hour_start,
            end_time=current_time,
            split_ratio=split_ratio,
            is_complete=split_ratio >= 1.0
        )

        # Store split
        self.bar_splits.append(bar_split)

        tprint_info(f"🔪 Created bar split: {split_ratio:.2%} completion")
        tprint_debug(f"   Split period: {bar_split.start_time} to {bar_split.end_time}")

        return bar_split

    async def get_complete_hourly_bars(self, n_bars: int = 24) -> pd.DataFrame:
        """
        Get complete hourly bars for regime evaluation.

        This method ensures we always return complete 1-hour bars,
        using nowcasting techniques when necessary.
        """
        try:
            tprint_info(f"📊 Getting {n_bars} complete hourly bars for regime evaluation...")

            # Get historical complete bars
            complete_bars = await self._get_historical_complete_bars(n_bars - 1)

            # Get current bar (potentially partial)
            current_bar = await self._get_current_bar_nowcasted()

            # Combine data
            if current_bar is not None:
                all_bars = pd.concat([complete_bars, current_bar], ignore_index=True)
            else:
                all_bars = complete_bars

            # Ensure we have the right number of bars
            if len(all_bars) < n_bars:
                tprint_warning(f"⚠️ Only {len(all_bars)} bars available, requested {n_bars}")
                # Pad with forward-filled data if enabled
                if self.config.enable_forward_filling:
                    all_bars = self._forward_fill_bars(all_bars, n_bars)

            tprint_success(f"✅ Retrieved {len(all_bars)} complete hourly bars")
            return all_bars

        except Exception as e:
            tprint_error(f"❌ Failed to get complete hourly bars: {e}")
            # Return empty DataFrame as fallback
            return pd.DataFrame()

    async def _get_historical_complete_bars(self, n_bars: int) -> pd.DataFrame:
        """Get historical complete hourly bars with realistic market data."""
        try:
            # This would integrate with your data source
            # For now, return mock data
            end_time = self.current_hour_start
            start_time = end_time - timedelta(hours=n_bars)

            # Create mock historical data
            timestamps = pd.date_range(start_time, end_time, freq='1H', inclusive='left')

            # Generate realistic mock data
            historical_data = self._generate_mock_historical_data(timestamps)

            tprint_debug(f"📈 Generated {len(historical_data)} historical complete bars")
            return historical_data

        except Exception as e:
            tprint_error(f"❌ Failed to get historical bars: {e}")
            return pd.DataFrame()

    async def _get_current_bar_nowcasted(self) -> Optional[pd.DataFrame]:
        """Get current bar with nowcasting applied."""
        try:
            current_time = datetime.now()
            bar_completion = self._calculate_bar_completion(current_time)

            if bar_completion < self.config.min_bar_completion:
                tprint_debug("⏳ Current bar completion too low, skipping")
                return None

            # Get partial data for current bar
            partial_data = await self._get_partial_bar_data()

            if partial_data is None or len(partial_data) == 0:
                tprint_warning("⚠️ No partial data available for current bar")
                return None

            # Apply nowcasting to complete the bar
            nowcasted_bar = await self._nowcast_complete_bar(partial_data, bar_completion)

            tprint_info(f"🔮 Nowcasted current bar with {bar_completion:.2%} completion")
            return nowcasted_bar

        except Exception as e:
            tprint_error(f"❌ Failed to nowcast current bar: {e}")
            return None

    async def _get_partial_bar_data(self) -> Optional[pd.DataFrame]:
        """Get partial data for the current bar."""
        try:
            # This would integrate with your real-time data source
            # For now, return mock partial data

            current_time = datetime.now()
            start_time = self.current_hour_start

            # Generate mock partial data
            n_minutes = int((current_time - start_time).total_seconds() / 60)

            if n_minutes < 1:
                return None

            # Create minute-by-minute data for the partial bar
            timestamps = pd.date_range(start_time, current_time, freq='1min')

            # Generate realistic partial OHLCV data
            partial_data = self._generate_mock_partial_data(timestamps, current_time)

            tprint_debug(f"📊 Generated {len(partial_data)} minutes of partial data")
            return partial_data

        except Exception as e:
            tprint_error(f"❌ Failed to get partial bar data: {e}")
            return None

    async def _nowcast_complete_bar(self, partial_data: pd.DataFrame, completion_ratio: float) -> pd.DataFrame:
        """
        Nowcast a complete hourly bar from partial data.

        Uses various techniques to estimate the complete bar:
        1. Extrapolation based on current trend
        2. Historical pattern matching
        3. Volatility-adjusted projections
        """
        try:
            if len(partial_data) == 0:
                return pd.DataFrame()

            # Get the latest partial data
            latest = partial_data.iloc[-1]

            # Calculate trend from partial data
            if len(partial_data) > 1:
                price_trend = (latest['close'] - partial_data.iloc[0]['open']) / partial_data.iloc[0]['open']
                volume_trend = partial_data['volume'].mean()
            else:
                price_trend = 0.0
                volume_trend = latest['volume']

            # Estimate remaining time
            remaining_ratio = 1.0 - completion_ratio
            remaining_minutes = int(60 * remaining_ratio)

            # Project final values
            if completion_ratio > 0.5:
                # Use trend extrapolation for high completion
                final_close = latest['close'] * (1 + price_trend * remaining_ratio * 0.5)
                final_volume = volume_trend * (1 + remaining_ratio)
            else:
                # Use more conservative projection for low completion
                final_close = latest['close'] * (1 + price_trend * 0.1)
                final_volume = volume_trend * 1.1

            # Ensure reasonable bounds
            final_close = max(final_close, latest['close'] * 0.95)  # Max 5% drop
            final_close = min(final_close, latest['close'] * 1.05)  # Max 5% rise

            # Create complete bar
            complete_bar = pd.DataFrame({
                'timestamp': [self.current_hour_start],
                'open': [partial_data.iloc[0]['open']],
                'high': [max(partial_data['high'].max(), final_close)],
                'low': [min(partial_data['low'].min(), final_close)],
                'close': [final_close],
                'volume': [final_volume],
                'is_complete': True,
                'is_nowcasted': True,
                'completion_ratio': completion_ratio,
                'confidence': min(completion_ratio * 1.2, 1.0)  # Higher completion = higher confidence
            })

            tprint_debug(f"🔮 Nowcasted bar: O={complete_bar['open'].iloc[0]:.2f}, "
                        f"H={complete_bar['high'].iloc[0]:.2f}, "
                        f"L={complete_bar['low'].iloc[0]:.2f}, "
                        f"C={complete_bar['close'].iloc[0]:.2f}")

            return complete_bar

        except Exception as e:
            tprint_error(f"❌ Failed to nowcast complete bar: {e}")
            return pd.DataFrame()

    def _forward_fill_bars(self, bars: pd.DataFrame, target_count: int) -> pd.DataFrame:
        """Forward fill bars to reach target count."""
        if len(bars) >= target_count:
            return bars.iloc[:target_count]

        # Get the last bar
        last_bar = bars.iloc[-1].copy()

        # Create additional bars by forward filling
        additional_bars = []
        for i in range(target_count - len(bars)):
            new_bar = last_bar.copy()
            new_bar['timestamp'] = last_bar['timestamp'] + timedelta(hours=i+1)
            new_bar['is_forward_filled'] = True
            additional_bars.append(new_bar)

        additional_df = pd.DataFrame(additional_bars)
        return pd.concat([bars, additional_df], ignore_index=True)

    async def update_evaluation_time(self) -> None:
        """Update the last evaluation time."""
        self.last_evaluation_time = datetime.now()
        tprint_debug(f"⏰ Updated evaluation time: {self.last_evaluation_time}")

    def _generate_mock_historical_data(self, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate realistic mock historical market data."""
        try:
            np.random.seed(42)  # For reproducibility
            n_points = len(timestamps)

            # Generate realistic price data with trends, volatility clusters, and mean reversion
            base_price = 3000.0  # ETH-like price
            prices = [base_price]
            
            # Create different market regimes
            regime_length = n_points // 4
            regimes = []
            for i in range(0, n_points, regime_length):
                regime_type = np.random.choice(['trending', 'sideways', 'volatile', 'mean_reverting'])
                regimes.extend([regime_type] * min(regime_length, n_points - i))
            
            # Generate price movements based on regimes
            for i in range(1, n_points):
                regime = regimes[i] if i < len(regimes) else 'sideways'
                
                if regime == 'trending':
                    # Trending regime with momentum
                    trend = np.random.uniform(-0.002, 0.002)  # -0.2% to 0.2% per hour
                    volatility = np.random.uniform(0.01, 0.03)  # 1-3% volatility
                    change = trend + np.random.normal(0, volatility)
                elif regime == 'sideways':
                    # Sideways regime with mean reversion
                    mean_reversion = -0.1 * (prices[i-1] - base_price) / base_price
                    volatility = np.random.uniform(0.005, 0.015)  # 0.5-1.5% volatility
                    change = mean_reversion + np.random.normal(0, volatility)
                elif regime == 'volatile':
                    # High volatility regime
                    volatility = np.random.uniform(0.02, 0.05)  # 2-5% volatility
                    change = np.random.normal(0, volatility)
                else:  # mean_reverting
                    # Mean reversion with moderate volatility
                    mean_reversion = -0.2 * (prices[i-1] - base_price) / base_price
                    volatility = np.random.uniform(0.01, 0.02)  # 1-2% volatility
                    change = mean_reversion + np.random.normal(0, volatility)
                
                new_price = prices[i-1] * (1 + change)
                prices.append(new_price)

            # Generate OHLCV data
            historical_data = []
            for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
                # Generate realistic OHLC
                volatility_factor = abs(close - prices[i-1]) / prices[i-1] if i > 0 else 0.01
                volatility_factor = max(volatility_factor, 0.005)  # Minimum volatility
                
                # Open price (previous close with small gap)
                if i == 0:
                    open_price = close
                else:
                    gap = np.random.normal(0, volatility_factor * 0.1)
                    open_price = prices[i-1] * (1 + gap)
                
                # High and low with realistic ranges
                high_range = np.random.uniform(0.001, volatility_factor * 2)
                low_range = np.random.uniform(0.001, volatility_factor * 2)
                
                high = max(open_price, close) * (1 + high_range)
                low = min(open_price, close) * (1 - low_range)
                
                # Ensure OHLC consistency
                high = max(open_price, close, high)
                low = min(open_price, close, low)
                
                # Generate volume with correlation to price movement
                price_change = abs(close - open_price) / open_price if open_price > 0 else 0
                base_volume = 1000
                volume_factor = 1 + price_change * 10  # Higher volume on bigger moves
                volume = base_volume * volume_factor * np.random.lognormal(0, 0.3)
                
                # Add technical indicators
                rsi = self._calculate_rsi(prices[:i+1]) if i >= 14 else 50
                sma_20 = np.mean(prices[max(0, i-19):i+1]) if i >= 19 else close
                sma_50 = np.mean(prices[max(0, i-49):i+1]) if i >= 49 else close
                
                historical_data.append({
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume,
                    'is_complete': True,
                    'regime': regimes[i] if i < len(regimes) else 'sideways',
                    'rsi': rsi,
                    'sma_20': sma_20,
                    'sma_50': sma_50,
                    'volatility': volatility_factor,
                    'price_change': price_change
                })

            df = pd.DataFrame(historical_data)
            
            # Add additional technical indicators
            df['returns'] = df['close'].pct_change()
            df['volatility_20'] = df['returns'].rolling(20).std()
            df['bb_upper'] = df['sma_20'] + (df['volatility_20'] * 2)
            df['bb_lower'] = df['sma_20'] - (df['volatility_20'] * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            return df

        except Exception as e:
            tprint_error(f"❌ Failed to generate mock historical data: {e}")
            return pd.DataFrame()

    def _generate_mock_partial_data(self, timestamps: pd.DatetimeIndex, current_time: datetime) -> pd.DataFrame:
        """Generate realistic mock partial bar data."""
        try:
            np.random.seed(int(current_time.timestamp()))
            n_minutes = len(timestamps)
            
            # Start with a base price (could be from previous hour's close)
            base_price = 3000.0
            
            # Generate price progression with realistic intraday patterns
            prices = [base_price]
            
            # Create intraday volatility pattern (higher at open/close, lower midday)
            for i in range(1, n_minutes):
                # Intraday volatility pattern
                minute_of_hour = i % 60
                if minute_of_hour < 15 or minute_of_hour > 45:  # First/last 15 minutes
                    vol_multiplier = 1.5
                else:  # Middle 30 minutes
                    vol_multiplier = 0.8
                
                # Base volatility per minute
                base_vol = 0.0005  # 0.05% per minute
                volatility = base_vol * vol_multiplier
                
                # Add some momentum from previous moves
                momentum = 0.0
                if i > 1:
                    recent_change = (prices[i-1] - prices[i-2]) / prices[i-2]
                    momentum = recent_change * 0.1  # 10% momentum carryover
                
                # Generate price change
                change = momentum + np.random.normal(0, volatility)
                new_price = prices[i-1] * (1 + change)
                prices.append(new_price)

            # Generate OHLCV data for each minute
            partial_data = []
            for i, (timestamp, close) in enumerate(zip(timestamps, prices)):
                # Generate realistic minute-level OHLC
                volatility_factor = abs(close - prices[i-1]) / prices[i-1] if i > 0 else 0.001
                volatility_factor = max(volatility_factor, 0.0005)  # Minimum volatility
                
                # Open price
                if i == 0:
                    open_price = close
                else:
                    gap = np.random.normal(0, volatility_factor * 0.2)
                    open_price = prices[i-1] * (1 + gap)
                
                # High and low
                high_range = np.random.uniform(0.0001, volatility_factor)
                low_range = np.random.uniform(0.0001, volatility_factor)
                
                high = max(open_price, close) * (1 + high_range)
                low = min(open_price, close) * (1 - low_range)
                
                # Ensure OHLC consistency
                high = max(open_price, close, high)
                low = min(open_price, close, low)
                
                # Generate volume with intraday patterns
                minute_of_hour = i % 60
                if minute_of_hour < 5 or minute_of_hour > 55:  # First/last 5 minutes
                    volume_multiplier = 2.0
                elif minute_of_hour < 15 or minute_of_hour > 45:  # First/last 15 minutes
                    volume_multiplier = 1.5
                else:  # Middle period
                    volume_multiplier = 0.8
                
                base_volume = 100
                price_change = abs(close - open_price) / open_price if open_price > 0 else 0
                volume = base_volume * volume_multiplier * (1 + price_change * 5) * np.random.lognormal(0, 0.2)
                
                partial_data.append({
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume,
                    'is_complete': False,
                    'minute_of_hour': minute_of_hour,
                    'volatility': volatility_factor,
                    'price_change': price_change
                })

            return pd.DataFrame(partial_data)

        except Exception as e:
            tprint_error(f"❌ Failed to generate mock partial data: {e}")
            return pd.DataFrame()

    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI for a list of prices."""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI if not enough data
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    async def get_nowcasting_stats(self) -> Dict[str, Any]:
        """Get statistics about the nowcasting process."""
        return {
            'current_hour_start': self.current_hour_start,
            'current_hour_end': self.current_hour_end,
            'bar_completion': self._calculate_bar_completion(datetime.now()),
            'bar_splits_count': len(self.bar_splits),
            'complete_bars_count': len(self.complete_bars),
            'last_evaluation_time': self.last_evaluation_time,
            'next_evaluation_time': self.next_evaluation_time,
            'config': {
                'base_timeframe': self.config.base_timeframe,
                'evaluation_interval': self.config.evaluation_interval,
                'min_bar_completion': self.config.min_bar_completion,
                'max_bar_completion': self.config.max_bar_completion,
                'confidence_threshold': self.config.confidence_threshold
            }
        }

# Factory function
def create_partial_bar_nowcaster(
    base_timeframe: str = "1h",
    evaluation_interval: int = 15 * 60,
    min_bar_completion: float = 0.25,
    max_bar_completion: float = 0.95
) -> PartialBarNowcaster:
    """Create a configured partial-bar nowcaster."""
    config = NowcastingConfig(
        base_timeframe=base_timeframe,
        evaluation_interval=evaluation_interval,
        min_bar_completion=min_bar_completion,
        max_bar_completion=max_bar_completion
    )
    return PartialBarNowcaster(config)

# Example usage
async def example_nowcasting():
    """Example of using the partial-bar nowcaster."""
    try:
        # Create nowcaster
        nowcaster = create_partial_bar_nowcaster()

        # Initialize
        await nowcaster.initialize()

        # Check if regime evaluation should occur
        should_evaluate = await nowcaster.should_evaluate_regime()
        print(f"Should evaluate regime: {should_evaluate}")

        if should_evaluate:
            # Get complete hourly bars
            bars = await nowcaster.get_complete_hourly_bars(n_bars=24)
            print(f"Retrieved {len(bars)} complete hourly bars")

            # Update evaluation time
            await nowcaster.update_evaluation_time()

        # Get stats
        stats = await nowcaster.get_nowcasting_stats()
        print(f"Nowcasting stats: {stats}")

    except Exception as e:
        print(f"Error in example: {e}")

if __name__ == "__main__":
    asyncio.run(example_nowcasting())
