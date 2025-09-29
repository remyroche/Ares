"""
Tests for Partial-Bar Nowcasting System

This module tests the partial-bar nowcasting functionality to ensure
that market regime evaluation always uses complete 1-hour bars.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.trading.execution.partial_bar_nowcasting import (
    PartialBarNowcaster, create_partial_bar_nowcaster, NowcastingConfig
)
from src.trading.execution.live_trading_scheduler import LiveTradingScheduler

class TestPartialBarNowcasting:
    """Test cases for partial-bar nowcasting functionality."""
    
    @pytest.fixture
    def nowcaster(self):
        """Create a nowcaster for testing."""
        config = NowcastingConfig(
            base_timeframe="1h",
            evaluation_interval=15 * 60,
            min_bar_completion=0.25,
            max_bar_completion=0.95,
            enable_forward_filling=True,
            enable_backward_filling=True,
            confidence_threshold=0.7
        )
        return PartialBarNowcaster(config)
    
    @pytest.fixture
    def scheduler(self):
        """Create a live trading scheduler for testing."""
        return LiveTradingScheduler(symbol="ETH", exchange="binance")
    
    @pytest.mark.asyncio
    async def test_nowcaster_initialization(self, nowcaster):
        """Test nowcaster initialization."""
        success = await nowcaster.initialize()
        assert success, "Nowcaster should initialize successfully"
        assert nowcaster.current_hour_start is not None, "Current hour start should be set"
        assert nowcaster.current_hour_end is not None, "Current hour end should be set"
        assert nowcaster.next_evaluation_time is not None, "Next evaluation time should be set"
    
    @pytest.mark.asyncio
    async def test_bar_completion_calculation(self, nowcaster):
        """Test bar completion calculation at different times."""
        await nowcaster.initialize()
        
        # Test different completion levels
        test_cases = [
            (15, 0.25),   # T+15 minutes = 25% completion
            (30, 0.50),   # T+30 minutes = 50% completion
            (45, 0.75),   # T+45 minutes = 75% completion
            (60, 1.00),   # T+60 minutes = 100% completion
        ]
        
        for minutes, expected_completion in test_cases:
            test_time = nowcaster.current_hour_start + timedelta(minutes=minutes)
            completion = nowcaster._calculate_bar_completion(test_time)
            assert abs(completion - expected_completion) < 0.01, \
                f"Completion at T+{minutes} should be ~{expected_completion:.2%}, got {completion:.2%}"
    
    @pytest.mark.asyncio
    async def test_evaluation_timing(self, nowcaster):
        """Test evaluation timing logic."""
        await nowcaster.initialize()
        
        # Test scenarios where evaluation should occur
        should_evaluate_times = [
            nowcaster.current_hour_start + timedelta(minutes=15),  # T+15
            nowcaster.current_hour_start + timedelta(minutes=30),  # T+30
            nowcaster.current_hour_start + timedelta(minutes=45),  # T+45
        ]
        
        for test_time in should_evaluate_times:
            should_evaluate = await nowcaster.should_evaluate_regime(test_time)
            assert should_evaluate, f"Should evaluate at {test_time.strftime('%H:%M')}"
        
        # Test scenarios where evaluation should NOT occur
        should_not_evaluate_times = [
            nowcaster.current_hour_start + timedelta(minutes=5),   # T+5 - too early
            nowcaster.current_hour_start + timedelta(minutes=58),  # T+58 - too late
        ]
        
        for test_time in should_not_evaluate_times:
            should_evaluate = await nowcaster.should_evaluate_regime(test_time)
            assert not should_evaluate, f"Should NOT evaluate at {test_time.strftime('%H:%M')}"
    
    @pytest.mark.asyncio
    async def test_bar_split_creation(self, nowcaster):
        """Test bar split creation."""
        await nowcaster.initialize()
        
        test_time = nowcaster.current_hour_start + timedelta(minutes=30)
        bar_split = await nowcaster.create_bar_split(test_time)
        
        assert bar_split is not None, "Bar split should be created"
        assert bar_split.start_time == nowcaster.current_hour_start, "Start time should match hour start"
        assert bar_split.end_time == test_time, "End time should match test time"
        assert 0.0 <= bar_split.split_ratio <= 1.0, "Split ratio should be between 0 and 1"
        assert len(nowcaster.bar_splits) == 1, "Bar split should be stored"
    
    @pytest.mark.asyncio
    async def test_complete_bar_retrieval(self, nowcaster):
        """Test complete bar retrieval."""
        await nowcaster.initialize()
        
        # Get complete bars
        complete_bars = await nowcaster.get_complete_hourly_bars(n_bars=24)
        
        assert isinstance(complete_bars, pd.DataFrame), "Should return DataFrame"
        assert len(complete_bars) > 0, "Should return some bars"
        
        # Check required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            assert col in complete_bars.columns, f"Should have {col} column"
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(complete_bars['timestamp']), "Timestamp should be datetime"
        assert pd.api.types.is_numeric_dtype(complete_bars['open']), "Open should be numeric"
        assert pd.api.types.is_numeric_dtype(complete_bars['high']), "High should be numeric"
        assert pd.api.types.is_numeric_dtype(complete_bars['low']), "Low should be numeric"
        assert pd.api.types.is_numeric_dtype(complete_bars['close']), "Close should be numeric"
        assert pd.api.types.is_numeric_dtype(complete_bars['volume']), "Volume should be numeric"
    
    @pytest.mark.asyncio
    async def test_partial_data_processing(self, nowcaster):
        """Test partial data processing and nowcasting."""
        await nowcaster.initialize()
        
        # Create mock partial data
        partial_data = self._create_mock_partial_data(0.5)  # 50% completion
        
        # Test nowcasting
        complete_bar = await nowcaster._nowcast_complete_bar(partial_data, 0.5)
        
        assert isinstance(complete_bar, pd.DataFrame), "Should return DataFrame"
        assert len(complete_bar) == 1, "Should return single bar"
        
        bar = complete_bar.iloc[0]
        assert 'open' in bar, "Should have open price"
        assert 'high' in bar, "Should have high price"
        assert 'low' in bar, "Should have low price"
        assert 'close' in bar, "Should have close price"
        assert 'volume' in bar, "Should have volume"
        assert 'is_nowcasted' in bar, "Should indicate if nowcasted"
        assert 'confidence' in bar, "Should have confidence score"
        
        # Validate OHLC relationships
        assert bar['high'] >= bar['open'], "High should be >= open"
        assert bar['high'] >= bar['close'], "High should be >= close"
        assert bar['low'] <= bar['open'], "Low should be <= open"
        assert bar['low'] <= bar['close'], "Low should be <= close"
        assert bar['volume'] > 0, "Volume should be positive"
        assert 0.0 <= bar['confidence'] <= 1.0, "Confidence should be between 0 and 1"
    
    @pytest.mark.asyncio
    async def test_scheduler_integration(self, scheduler):
        """Test integration with live trading scheduler."""
        # Test scheduler initialization
        assert scheduler.nowcaster is not None, "Scheduler should have nowcaster"
        assert scheduler.model_configs[ModelType.HMM].custom_params.get('use_nowcasting', False), \
            "HMM should have nowcasting enabled"
        
        # Test nowcaster initialization through scheduler
        success = await scheduler.nowcaster.initialize()
        assert success, "Scheduler nowcaster should initialize"
    
    @pytest.mark.asyncio
    async def test_forward_filling(self, nowcaster):
        """Test forward filling for insufficient data."""
        await nowcaster.initialize()
        
        # Create insufficient data
        insufficient_bars = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(hours=2)],
            'open': [50000.0],
            'high': [51000.0],
            'low': [49000.0],
            'close': [50500.0],
            'volume': [1000.0],
            'is_complete': [True]
        })
        
        # Test forward filling
        filled_bars = nowcaster._forward_fill_bars(insufficient_bars, 5)
        
        assert len(filled_bars) == 5, "Should have 5 bars after forward filling"
        assert all(filled_bars['is_forward_filled'].iloc[1:]), "Additional bars should be forward filled"
        assert not filled_bars['is_forward_filled'].iloc[0], "Original bar should not be forward filled"
    
    @pytest.mark.asyncio
    async def test_statistics_retrieval(self, nowcaster):
        """Test statistics retrieval."""
        await nowcaster.initialize()
        
        stats = await nowcaster.get_nowcasting_stats()
        
        assert isinstance(stats, dict), "Should return dictionary"
        assert 'current_hour_start' in stats, "Should have current hour start"
        assert 'current_hour_end' in stats, "Should have current hour end"
        assert 'bar_completion' in stats, "Should have bar completion"
        assert 'bar_splits_count' in stats, "Should have bar splits count"
        assert 'config' in stats, "Should have config information"
        
        # Validate config information
        config = stats['config']
        assert config['base_timeframe'] == "1h", "Should have correct base timeframe"
        assert config['evaluation_interval'] == 15 * 60, "Should have correct evaluation interval"
    
    def _create_mock_partial_data(self, completion_ratio: float) -> pd.DataFrame:
        """Create mock partial data for testing."""
        n_minutes = max(1, int(60 * completion_ratio))
        base_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        timestamps = pd.date_range(base_time, periods=n_minutes, freq='1min')
        
        # Generate realistic price data
        np.random.seed(42)
        base_price = 50000.0
        price_changes = np.random.normal(0, 0.001, n_minutes)
        prices = [base_price]
        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change))
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'volume': np.random.lognormal(8, 0.2, n_minutes),
            'is_complete': False
        })

class TestTimingScenarios:
    """Test different timing scenarios for regime evaluation."""
    
    @pytest.mark.asyncio
    async def test_t15_scenario(self):
        """Test T+15 scenario (25% completion)."""
        nowcaster = create_partial_bar_nowcaster()
        await nowcaster.initialize()
        
        test_time = nowcaster.current_hour_start + timedelta(minutes=15)
        should_evaluate = await nowcaster.should_evaluate_regime(test_time)
        
        assert should_evaluate, "Should evaluate at T+15"
        
        completion = nowcaster._calculate_bar_completion(test_time)
        assert 0.24 <= completion <= 0.26, f"Completion should be ~25%, got {completion:.2%}"
    
    @pytest.mark.asyncio
    async def test_t30_scenario(self):
        """Test T+30 scenario (50% completion)."""
        nowcaster = create_partial_bar_nowcaster()
        await nowcaster.initialize()
        
        test_time = nowcaster.current_hour_start + timedelta(minutes=30)
        should_evaluate = await nowcaster.should_evaluate_regime(test_time)
        
        assert should_evaluate, "Should evaluate at T+30"
        
        completion = nowcaster._calculate_bar_completion(test_time)
        assert 0.49 <= completion <= 0.51, f"Completion should be ~50%, got {completion:.2%}"
    
    @pytest.mark.asyncio
    async def test_t45_scenario(self):
        """Test T+45 scenario (75% completion)."""
        nowcaster = create_partial_bar_nowcaster()
        await nowcaster.initialize()
        
        test_time = nowcaster.current_hour_start + timedelta(minutes=45)
        should_evaluate = await nowcaster.should_evaluate_regime(test_time)
        
        assert should_evaluate, "Should evaluate at T+45"
        
        completion = nowcaster._calculate_bar_completion(test_time)
        assert 0.74 <= completion <= 0.76, f"Completion should be ~75%, got {completion:.2%}"
    
    @pytest.mark.asyncio
    async def test_early_scenario(self):
        """Test early scenario (T+5, should not evaluate)."""
        nowcaster = create_partial_bar_nowcaster()
        await nowcaster.initialize()
        
        test_time = nowcaster.current_hour_start + timedelta(minutes=5)
        should_evaluate = await nowcaster.should_evaluate_regime(test_time)
        
        assert not should_evaluate, "Should NOT evaluate at T+5"
    
    @pytest.mark.asyncio
    async def test_late_scenario(self):
        """Test late scenario (T+58, should not evaluate)."""
        nowcaster = create_partial_bar_nowcaster()
        await nowcaster.initialize()
        
        test_time = nowcaster.current_hour_start + timedelta(minutes=58)
        should_evaluate = await nowcaster.should_evaluate_regime(test_time)
        
        assert not should_evaluate, "Should NOT evaluate at T+58"

# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])