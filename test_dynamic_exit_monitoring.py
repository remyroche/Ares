#!/usr/bin/env python3
"""
Test script for Dynamic Exit Monitoring

This script demonstrates how continuous monitoring of evolving market conditions
provides superior exit timing compared to static triple barrier methods.
"""

import asyncio
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.tactician.dynamic_exit_monitor import DynamicExitMonitor, MarketCondition, ExitSignal


def create_test_config():
    """Create test configuration for dynamic exit monitoring."""
    config = {
        "dynamic_exit_monitor": {
            "lookback_window": 50,
            "update_frequency": 1,  # bars
            "confidence_threshold": 0.7,
            "min_position_age": 5,  # minutes
            "volatility_threshold": 0.02,
            "momentum_threshold": 0.5,
            "trend_strength_threshold": 0.6
        }
    }
    return config


def create_test_market_data():
    """Create realistic test market data with evolving conditions."""
    np.random.seed(42)
    
    # Create 200 minutes of test data
    start_time = datetime(2024, 1, 1, 9, 0, 0)
    timestamps = [start_time + timedelta(minutes=i) for i in range(200)]
    
    # Generate realistic price movements with evolving conditions
    base_price = 100.0
    prices = [base_price]
    
    # Simulate different market phases
    for i in range(1, 200):
        if i < 50:
            # Phase 1: Trending up
            change = np.random.normal(0.001, 0.002)  # Positive trend
        elif i < 100:
            # Phase 2: High volatility
            change = np.random.normal(0, 0.005)  # High volatility
        elif i < 150:
            # Phase 3: Trending down
            change = np.random.normal(-0.001, 0.002)  # Negative trend
        else:
            # Phase 4: Sideways consolidation
            change = np.random.normal(0, 0.001)  # Low volatility
            
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create OHLC data
    data = []
    for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
        # Create realistic OHLC from price
        high = price * (1 + abs(np.random.normal(0, 0.001)))
        low = price * (1 - abs(np.random.normal(0, 0.001)))
        open_price = prices[i-1] if i > 0 else price
        close_price = price
        
        # Vary volume based on market conditions
        if i < 50:  # Trending phase
            volume = np.random.randint(5000, 15000)
        elif i < 100:  # Volatile phase
            volume = np.random.randint(8000, 20000)
        else:  # Consolidation phase
            volume = np.random.randint(3000, 8000)
        
        data.append({
            "timestamp": timestamp,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close_price,
            "volume": volume
        })
    
    df = pd.DataFrame(data)
    df.set_index("timestamp", inplace=True)
    return df


def create_test_position_data():
    """Create test position data."""
    return {
        "side": "long",
        "entry_price": 100.0,
        "entry_time": datetime(2024, 1, 1, 9, 0, 0),
        "age_minutes": 30,
        "unrealized_pnl_pct": 0.015,  # 1.5% profit
        "position_size": 0.1,
        "leverage": 20
    }


async def test_dynamic_exit_monitoring():
    """Test dynamic exit monitoring with evolving market conditions."""
    print("🚀 Testing Dynamic Exit Monitoring")
    print("=" * 60)
    
    # Create configuration and data
    config = create_test_config()
    market_data = create_test_market_data()
    position_data = create_test_position_data()
    
    # Initialize dynamic exit monitor
    monitor = DynamicExitMonitor(config)
    await monitor.initialize()
    
    print(f"📊 Market data: {len(market_data)} periods")
    print(f"🎯 Position: {position_data['side']} {position_data['position_size']} @ {position_data['entry_price']}")
    print(f"💰 Unrealized PnL: {position_data['unrealized_pnl_pct']:.3f}")
    print()
    
    # Simulate continuous monitoring
    print("🔄 Simulating continuous market monitoring...")
    print()
    
    exit_decisions = []
    
    for i in range(50, len(market_data), 5):  # Monitor every 5 bars
        # Get current market data window
        current_data = market_data.iloc[max(0, i-50):i+1]
        
        # Market context
        market_context = {
            "time_of_day": current_data.index[-1].hour,
            "day_of_week": current_data.index[-1].weekday(),
            "overall_trend": "bullish" if i < 100 else "bearish"
        }
        
        # Update position data
        current_price = current_data['close'].iloc[-1]
        position_data['age_minutes'] = i
        position_data['unrealized_pnl_pct'] = (current_price - position_data['entry_price']) / position_data['entry_price']
        
        # Get exit decision
        decision = await monitor.monitor_and_decide(current_data, position_data, market_context)
        
        # Store decision
        exit_decisions.append({
            "timestamp": current_data.index[-1],
            "price": current_price,
            "decision": decision,
            "market_condition": monitor.current_market_state.market_condition.value if monitor.current_market_state else None
        })
        
        # Print decision if significant
        if decision.should_exit or decision.confidence > 0.8:
            print(f"⏰ {current_data.index[-1].strftime('%H:%M')} - Price: {current_price:.2f}")
            print(f"   📊 Market: {monitor.current_market_state.market_condition.value if monitor.current_market_state else 'Unknown'}")
            print(f"   🎯 Decision: {decision.exit_signal.value}")
            print(f"   📈 Confidence: {decision.confidence:.2f}")
            print(f"   ⚡ Urgency: {decision.urgency:.2f}")
            print(f"   💬 Reason: {decision.reason}")
            print(f"   💰 PnL: {position_data['unrealized_pnl_pct']:.3f}")
            print()
    
    # Analyze results
    await analyze_monitoring_results(exit_decisions, monitor)
    
    print("=" * 60)


async def analyze_monitoring_results(exit_decisions, monitor):
    """Analyze the results of dynamic exit monitoring."""
    print("📊 Dynamic Exit Monitoring Analysis")
    print("-" * 40)
    
    # Count decisions by type
    decision_counts = {}
    for decision_data in exit_decisions:
        signal = decision_data['decision'].exit_signal.value
        decision_counts[signal] = decision_counts.get(signal, 0) + 1
    
    print("🎯 Exit Signal Distribution:")
    for signal, count in decision_counts.items():
        percentage = (count / len(exit_decisions)) * 100
        print(f"   {signal}: {count} ({percentage:.1f}%)")
    
    # Analyze market condition changes
    condition_changes = []
    for i in range(1, len(exit_decisions)):
        prev_condition = exit_decisions[i-1]['market_condition']
        curr_condition = exit_decisions[i]['market_condition']
        if prev_condition != curr_condition:
            condition_changes.append({
                "from": prev_condition,
                "to": curr_condition,
                "timestamp": exit_decisions[i]['timestamp']
            })
    
    print(f"\n🔄 Market Condition Changes: {len(condition_changes)}")
    for change in condition_changes:
        print(f"   {change['timestamp'].strftime('%H:%M')}: {change['from']} → {change['to']}")
    
    # Analyze high-confidence decisions
    high_confidence_decisions = [
        d for d in exit_decisions 
        if d['decision'].confidence > 0.8
    ]
    
    print(f"\n🎯 High-Confidence Decisions: {len(high_confidence_decisions)}")
    for decision_data in high_confidence_decisions:
        print(f"   {decision_data['timestamp'].strftime('%H:%M')}: {decision_data['decision'].exit_signal.value} "
              f"(conf: {decision_data['decision'].confidence:.2f})")
    
    # Get monitoring summary
    summary = await monitor.get_monitoring_summary()
    print(f"\n📈 Monitoring Summary:")
    print(f"   Market history: {summary['market_history_size']} states")
    print(f"   Exit signals: {summary['exit_signals_count']} total")
    print(f"   Current condition: {summary['current_market_state']['condition']}")
    print(f"   Current confidence: {summary['current_market_state']['confidence']:.2f}")


def explain_dynamic_monitoring_advantages():
    """Explain the advantages of dynamic monitoring over static methods."""
    print("\n📚 Dynamic Exit Monitoring Advantages")
    print("=" * 60)
    
    print("""
🎯 **Why Continuous Monitoring is Superior:**

1. **Real-Time Adaptation**
   - Responds to market changes as they happen
   - No fixed barriers that become irrelevant
   - Adapts to volatility, momentum, and trend changes

2. **Evolving Market Conditions**
   - Detects trend reversals before they fully develop
   - Identifies momentum loss early
   - Recognizes volatility spikes immediately
   - Spots breakouts and consolidations

3. **Context-Aware Decisions**
   - Considers position age and profit/loss
   - Evaluates trend alignment
   - Weighs urgency vs. confidence
   - Adapts to different market regimes

4. **Multiple Exit Signals**
   - Trend reversal detection
   - Momentum loss/gain monitoring
   - Volatility spike alerts
   - Breakout identification
   - Support/resistance proximity

5. **Adaptive Thresholds**
   - Thresholds adjust based on performance
   - Learns from successful/failed exits
   - Adapts to changing market conditions
   - Optimizes for current regime

🔄 **Continuous Monitoring Process:**

1. **Market State Update** (every bar)
   - Calculate real-time volatility, momentum, trend strength
   - Detect support/resistance levels
   - Determine current market condition
   - Calculate confidence in analysis

2. **Evolution Analysis** (continuous)
   - Track how conditions are changing
   - Identify trend strengthening/weakening
   - Monitor volatility evolution
   - Analyze momentum acceleration/deceleration

3. **Exit Signal Detection** (real-time)
   - Check for trend reversals
   - Detect momentum loss
   - Identify volatility spikes
   - Spot breakouts
   - Monitor support/resistance proximity

4. **Adaptive Decision Making** (context-aware)
   - Consider signal strength and urgency
   - Evaluate position context
   - Weigh profit/loss situation
   - Check trend alignment
   - Make final exit decision

5. **Performance Learning** (continuous)
   - Track exit success rates
   - Update adaptive thresholds
   - Learn from market behavior
   - Optimize decision parameters

🚫 **Problems with Static Triple Barrier:**

- ❌ Fixed barriers don't adapt to market changes
- ❌ Time-based exits can cut profitable trends short
- ❌ No consideration of evolving conditions
- ❌ Ignores momentum and volatility changes
- ❌ Can't detect early reversal signals
- ❌ No learning from market behavior
- ❌ One-size-fits-all approach

✅ **Benefits of Dynamic Monitoring:**

- ✅ Adapts to real-time market changes
- ✅ Detects early warning signals
- ✅ Considers multiple market factors
- ✅ Learns and improves over time
- ✅ Context-aware decision making
- ✅ Regime-specific behavior
- ✅ Continuous optimization
""")


if __name__ == "__main__":
    print("🧪 Dynamic Exit Monitoring Test")
    print("=" * 60)
    
    # Explain advantages
    explain_dynamic_monitoring_advantages()
    
    # Run test
    asyncio.run(test_dynamic_exit_monitoring())