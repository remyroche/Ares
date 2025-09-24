#!/usr/bin/env python3
"""
Exit Strategy Optimization Runner

This script demonstrates how to run exit strategy parameter optimization
and integrate the results with the position monitor.

Usage:
    python run_exit_strategy_optimization.py
"""

import asyncio
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.training.steps.backtesting.exit_strategy_optimization import (
    ExitStrategyOptimizer, 
    optimize_exit_strategy_parameters,
    create_exit_strategy_config_from_optimization
)
from src.tactician.position_monitor import PositionMonitor

async def run_exit_strategy_optimization():
    """Run comprehensive exit strategy optimization."""
    
    print("🚀 Exit Strategy Parameter Optimization")
    print("=" * 60)
    
    # 1. Create sample market data for optimization
    print("📊 Creating sample market data...")
    market_data = create_sample_market_data()
    print(f"   ✅ Created {len(market_data)} data points")
    
    # 2. Create sample calibration results
    print("📈 Creating sample calibration results...")
    calibration_results = create_sample_calibration_results()
    print(f"   ✅ Created calibration results with {len(calibration_results)} keys")
    
    # 3. Load optimization configuration
    print("⚙️ Loading optimization configuration...")
    config = load_optimization_config()
    print(f"   ✅ Loaded configuration with {config['n_trials']} trials")
    
    # 4. Run optimization
    print("🔬 Running exit strategy optimization...")
    print(f"   📊 Trials: {config['n_trials']}")
    print(f"   ⏱️ Timeout: {config['timeout']}s")
    
    try:
        result = await optimize_exit_strategy_parameters(
            market_data, calibration_results, config
        )
        
        print("✅ Optimization completed successfully!")
        print(f"📊 Best Sharpe Ratio: {result.optimization_metrics['sharpe_ratio']:.3f}")
        print(f"📈 Best Profit Factor: {result.optimization_metrics['profit_factor']:.3f}")
        print(f"📉 Max Drawdown: {result.optimization_metrics['max_drawdown']:.3f}")
        print(f"🎯 Win Rate: {result.optimization_metrics['win_rate']:.1%}")
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return
    
    # 5. Create position monitor configuration
    print("\n🔧 Creating position monitor configuration...")
    position_config = create_exit_strategy_config_from_optimization(result)
    
    # 6. Test position monitor with optimized parameters
    print("🧪 Testing position monitor with optimized parameters...")
    await test_position_monitor_with_optimized_params(position_config)
    
    # 7. Display optimization results
    print("\n📋 Optimization Results Summary:")
    print("-" * 40)
    display_optimization_results(result)
    
    print("\n✅ Exit strategy optimization completed successfully!")

def create_sample_market_data() -> pd.DataFrame:
    """Create sample market data for optimization."""
    
    # Generate realistic market data
    np.random.seed(42)
    n_points = 2000
    
    # Generate price series with trend and volatility
    returns = np.random.normal(0.0001, 0.02, n_points)  # 0.01% mean return, 2% volatility
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLC data
    data = []
    for i, price in enumerate(prices):
        volatility = np.random.uniform(0.01, 0.03)
        high = price * (1 + volatility * np.random.uniform(0, 1))
        low = price * (1 - volatility * np.random.uniform(0, 1))
        open_price = prices[i-1] if i > 0 else price
        close = price
        
        data.append({
            'timestamp': datetime.now() - timedelta(hours=n_points-i),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(1000, 10000)
        })
    
    return pd.DataFrame(data)

def create_sample_calibration_results() -> Dict[str, Any]:
    """Create sample calibration results."""
    
    return {
        'model_confidence': 0.75,
        'regime_probabilities': np.random.rand(2000, 3),
        'signal_strength': np.random.rand(2000),
        'analyst_confidence': np.random.uniform(0.6, 0.9, 2000),
        'tactician_confidence': np.random.uniform(0.5, 0.8, 2000),
        'regime_classification': np.random.choice(['trending', 'ranging', 'volatile'], 2000),
        'volatility_regime': np.random.choice(['low', 'medium', 'high'], 2000)
    }

def load_optimization_config() -> Dict[str, Any]:
    """Load optimization configuration."""
    
    return {
        'n_trials': 20,  # Reduced for demo
        'timeout': 300,
        'study_name': 'exit_strategy_optimization_demo',
        'output_path': 'results/exit_strategy_optimization_demo.json'
    }

async def test_position_monitor_with_optimized_params(config: Dict[str, Any]):
    """Test position monitor with optimized parameters."""
    
    try:
        # Initialize position monitor with optimized config
        monitor = PositionMonitor(config)
        await monitor.initialize()
        
        # Get optimization status
        status = monitor.get_optimization_status()
        
        print(f"   ✅ Position monitor initialized")
        print(f"   📊 Optimization loaded: {status['optimization_loaded']}")
        print(f"   🎯 Confidence thresholds: {status['confidence_thresholds']}")
        print(f"   💰 Profit target: {status['pnl_thresholds']['profit_target']:.1%}")
        print(f"   🛡️ Stop loss: {status['pnl_thresholds']['stop_loss']:.1%}")
        print(f"   ⏰ Max hold time: {status['max_position_age']/3600:.1f} hours")
        
        # Test with sample position
        sample_position = {
            "position_id": "test_optimized",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "quantity": 1.0,
            "entry_price": 50000.0,
            "current_price": 52000.0,  # 4% profit
            "entry_time": datetime.now() - timedelta(minutes=30),
            "unrealized_pnl": 2000.0
        }
        
        # Test decision making with optimized parameters
        action, reason = monitor._determine_position_action(sample_position, 0.8)
        print(f"   🧪 Test decision: {action.value} - {reason}")
        
        await monitor.cleanup()
        
    except Exception as e:
        print(f"   ❌ Position monitor test failed: {e}")

def display_optimization_results(result):
    """Display optimization results in a readable format."""
    
    print(f"📊 Optimization Metrics:")
    for metric, value in result.optimization_metrics.items():
        if isinstance(value, float):
            print(f"   {metric}: {value:.3f}")
        else:
            print(f"   {metric}: {value}")
    
    print(f"\n🎯 Best Parameters:")
    params = result.best_parameters
    
    print(f"   Confidence Thresholds:")
    for level, threshold in params.confidence_thresholds.items():
        print(f"     {level}: {threshold:.2f}")
    
    print(f"   Profit Taking:")
    print(f"     Base target: {params.profit_taking['base_profit_target']:.1%}")
    print(f"     Min confidence: {params.profit_taking['min_confidence_for_profit']:.2f}")
    print(f"     Scaling levels: {params.profit_taking['scaling_levels']}")
    
    print(f"   Stop Loss:")
    print(f"     Base stop: {params.stop_loss['base_stop_loss']:.1%}")
    print(f"     ATR multiplier: {params.stop_loss['atr_multiplier']:.1f}")
    
    print(f"   Time Based:")
    print(f"     Max hold: {params.time_based['max_hold_time']/3600:.1f} hours")
    print(f"     Min hold: {params.time_based['min_hold_time']/60:.1f} minutes")
    
    print(f"   Trailing Stop:")
    print(f"     ATR multiplier: {params.trailing_stop['atr_multiplier']:.1f}")
    print(f"     Min distance: {params.trailing_stop['min_distance']:.1%}")
    print(f"     Activation confidence: {params.trailing_stop['confidence_activation']:.2f}")

async def main():
    """Main function."""
    try:
        await run_exit_strategy_optimization()
    except KeyboardInterrupt:
        print("\n⚠️ Optimization interrupted by user")
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())