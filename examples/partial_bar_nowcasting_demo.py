"""
Partial-Bar Nowcasting Demo

This example demonstrates the partial-bar nowcasting system for live trading.
It shows how the system ensures market regime evaluation always uses complete
1-hour bars, regardless of when the evaluation occurs within the hour.

Key Features Demonstrated:
1. Bar completion detection
2. Partial-bar nowcasting
3. Complete bar reconstruction
4. Timing-based evaluation control
5. Integration with live trading scheduler
"""

import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import the nowcasting system
from src.trading.execution.partial_bar_nowcasting import (
    PartialBarNowcaster, create_partial_bar_nowcaster, NowcastingConfig
)
from src.trading.execution.live_trading_scheduler import (
    LiveTradingScheduler, ModelType
)
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_debug, tprint_progress, tprint_performance, tprint_structured,
    tprint_timer, LogLevel
)

async def demo_bar_completion_detection():
    """Demonstrate bar completion detection at different times."""
    tprint_info("🔍 Demo: Bar Completion Detection")
    tprint_info("=" * 50)
    
    # Create nowcaster
    nowcaster = create_partial_bar_nowcaster()
    await nowcaster.initialize()
    
    # Simulate different times within an hour
    current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
    test_times = [
        current_hour + timedelta(minutes=15),  # T+15
        current_hour + timedelta(minutes=30),  # T+30
        current_hour + timedelta(minutes=45),  # T+45
        current_hour + timedelta(minutes=59),  # T+59
    ]
    
    for test_time in test_times:
        completion = nowcaster._calculate_bar_completion(test_time)
        should_evaluate = await nowcaster.should_evaluate_regime(test_time)
        
        tprint_info(f"⏰ Time: {test_time.strftime('%H:%M')} (T+{test_time.minute})")
        tprint_info(f"   Bar completion: {completion:.2%}")
        tprint_info(f"   Should evaluate: {'✅ Yes' if should_evaluate else '❌ No'}")
        tprint_info("")

async def demo_partial_bar_nowcasting():
    """Demonstrate partial-bar nowcasting with different completion levels."""
    tprint_info("🔮 Demo: Partial-Bar Nowcasting")
    tprint_info("=" * 50)
    
    # Create nowcaster with relaxed settings for demo
    config = NowcastingConfig(
        base_timeframe="1h",
        evaluation_interval=15 * 60,
        min_bar_completion=0.1,  # 10% minimum for demo
        max_bar_completion=0.9,  # 90% maximum for demo
        enable_forward_filling=True,
        enable_backward_filling=True,
        confidence_threshold=0.5
    )
    
    nowcaster = PartialBarNowcaster(config)
    await nowcaster.initialize()
    
    # Test different completion scenarios
    scenarios = [
        ("T+15 (25% complete)", 0.25),
        ("T+30 (50% complete)", 0.50),
        ("T+45 (75% complete)", 0.75),
    ]
    
    for scenario_name, completion_ratio in scenarios:
        tprint_info(f"📊 Scenario: {scenario_name}")
        
        # Simulate partial data
        partial_data = create_mock_partial_data(completion_ratio)
        
        # Nowcast complete bar
        complete_bar = await nowcaster._nowcast_complete_bar(partial_data, completion_ratio)
        
        if len(complete_bar) > 0:
            bar = complete_bar.iloc[0]
            tprint_info(f"   Original OHLC: O={partial_data.iloc[0]['open']:.2f}, "
                       f"H={partial_data['high'].max():.2f}, "
                       f"L={partial_data['low'].min():.2f}, "
                       f"C={partial_data.iloc[-1]['close']:.2f}")
            tprint_info(f"   Nowcasted OHLC: O={bar['open']:.2f}, "
                       f"H={bar['high']:.2f}, "
                       f"L={bar['low']:.2f}, "
                       f"C={bar['close']:.2f}")
            tprint_info(f"   Confidence: {bar.get('confidence', 0):.2%}")
            tprint_info(f"   Is nowcasted: {bar.get('is_nowcasted', False)}")
        else:
            tprint_warning("   ⚠️ No complete bar generated")
        
        tprint_info("")

def create_mock_partial_data(completion_ratio: float) -> pd.DataFrame:
    """Create mock partial data for testing."""
    # Create minute-by-minute data for the partial period
    n_minutes = int(60 * completion_ratio)
    if n_minutes < 1:
        n_minutes = 1
    
    base_price = 50000.0
    timestamps = pd.date_range(
        start=datetime.now().replace(minute=0, second=0, microsecond=0),
        periods=n_minutes,
        freq='1min'
    )
    
    # Generate realistic price progression
    np.random.seed(42)
    price_changes = np.random.normal(0, 0.001, n_minutes)  # 0.1% volatility per minute
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

async def demo_complete_bar_retrieval():
    """Demonstrate complete bar retrieval for regime evaluation."""
    tprint_info("📈 Demo: Complete Bar Retrieval")
    tprint_info("=" * 50)
    
    # Create nowcaster
    nowcaster = create_partial_bar_nowcaster()
    await nowcaster.initialize()
    
    # Get complete bars for regime evaluation
    complete_bars = await nowcaster.get_complete_hourly_bars(n_bars=24)
    
    tprint_info(f"📊 Retrieved {len(complete_bars)} complete hourly bars")
    
    if len(complete_bars) > 0:
        # Show statistics
        nowcasted_count = len(complete_bars[complete_bars.get('is_nowcasted', False)])
        complete_count = len(complete_bars[complete_bars.get('is_complete', True)])
        
        tprint_info(f"   Complete bars: {complete_count}")
        tprint_info(f"   Nowcasted bars: {nowcasted_count}")
        tprint_info(f"   Time range: {complete_bars['timestamp'].min()} to {complete_bars['timestamp'].max()}")
        
        # Show sample data
        tprint_info("   Sample data:")
        sample = complete_bars.head(3)
        for _, row in sample.iterrows():
            tprint_info(f"     {row['timestamp']}: O={row['open']:.2f}, "
                       f"H={row['high']:.2f}, L={row['low']:.2f}, "
                       f"C={row['close']:.2f}, V={row['volume']:.0f}")
    else:
        tprint_warning("   ⚠️ No complete bars available")

async def demo_live_trading_integration():
    """Demonstrate integration with live trading scheduler."""
    tprint_info("🚀 Demo: Live Trading Integration")
    tprint_info("=" * 50)
    
    # Create live trading scheduler
    scheduler = LiveTradingScheduler(symbol="ETH", exchange="binance")
    
    # Start scheduler
    success = await scheduler.start_scheduler()
    if not success:
        tprint_error("❌ Failed to start scheduler")
        return
    
    tprint_success("✅ Live trading scheduler started")
    
    # Run for a short period to demonstrate
    tprint_info("⏳ Running scheduler for 30 seconds...")
    await asyncio.sleep(30)
    
    # Get statistics
    scheduler_stats = scheduler.get_scheduler_stats()
    nowcasting_stats = await scheduler.get_nowcasting_stats()
    
    tprint_info("📊 Scheduler Statistics:")
    tprint_structured(scheduler_stats, LogLevel.INFO)
    
    tprint_info("🔮 Nowcasting Statistics:")
    tprint_structured(nowcasting_stats, LogLevel.INFO)
    
    # Stop scheduler
    await scheduler.stop_scheduler()
    tprint_success("✅ Scheduler stopped")

async def demo_timing_scenarios():
    """Demonstrate different timing scenarios for regime evaluation."""
    tprint_info("⏰ Demo: Timing Scenarios")
    tprint_info("=" * 50)
    
    # Create nowcaster
    nowcaster = create_partial_bar_nowcaster()
    await nowcaster.initialize()
    
    # Simulate different market conditions
    scenarios = [
        {
            "name": "Market Open (T+15)",
            "time": datetime.now().replace(minute=15, second=0, microsecond=0),
            "expected": "Should evaluate - sufficient completion"
        },
        {
            "name": "Mid-Hour (T+30)",
            "time": datetime.now().replace(minute=30, second=0, microsecond=0),
            "expected": "Should evaluate - good completion"
        },
        {
            "name": "Late Hour (T+45)",
            "time": datetime.now().replace(minute=45, second=0, microsecond=0),
            "expected": "Should evaluate - high completion"
        },
        {
            "name": "Very Early (T+5)",
            "time": datetime.now().replace(minute=5, second=0, microsecond=0),
            "expected": "Should NOT evaluate - insufficient completion"
        },
        {
            "name": "Very Late (T+58)",
            "time": datetime.now().replace(minute=58, second=0, microsecond=0),
            "expected": "Should NOT evaluate - too close to completion"
        }
    ]
    
    for scenario in scenarios:
        tprint_info(f"📅 Scenario: {scenario['name']}")
        tprint_info(f"   Time: {scenario['time'].strftime('%H:%M')}")
        tprint_info(f"   Expected: {scenario['expected']}")
        
        # Check if evaluation should occur
        should_evaluate = await nowcaster.should_evaluate_regime(scenario['time'])
        completion = nowcaster._calculate_bar_completion(scenario['time'])
        
        tprint_info(f"   Bar completion: {completion:.2%}")
        tprint_info(f"   Should evaluate: {'✅ Yes' if should_evaluate else '❌ No'}")
        
        if should_evaluate:
            # Simulate getting complete bars
            complete_bars = await nowcaster.get_complete_hourly_bars(n_bars=24)
            tprint_info(f"   Complete bars available: {len(complete_bars)}")
        
        tprint_info("")

async def main():
    """Run all demonstrations."""
    tprint_info("🎯 Partial-Bar Nowcasting Demo")
    tprint_info("=" * 60)
    tprint_info("This demo shows how partial-bar nowcasting ensures")
    tprint_info("market regime evaluation always uses complete 1-hour bars.")
    tprint_info("=" * 60)
    tprint_info("")
    
    try:
        # Run demonstrations
        await demo_bar_completion_detection()
        await demo_partial_bar_nowcasting()
        await demo_complete_bar_retrieval()
        await demo_timing_scenarios()
        await demo_live_trading_integration()
        
        tprint_success("🎉 All demonstrations completed successfully!")
        
    except Exception as e:
        tprint_error(f"❌ Demo failed: {e}")
        raise

if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())