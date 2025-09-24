#!/usr/bin/env python3
"""
Test Optimized Exit Strategy Integration

This script tests the integration between exit strategy optimization
and the position monitor with optimized parameters.
"""

import asyncio
import sys
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.tactician.position_monitor import PositionMonitor, PositionAction

async def test_optimized_exit_strategy():
    """Test the optimized exit strategy integration."""
    
    print("🧪 Testing Optimized Exit Strategy Integration")
    print("=" * 60)
    
    # 1. Create mock optimization results
    print("📊 Creating mock optimization results...")
    optimization_results = create_mock_optimization_results()
    
    # Save mock results to file
    results_path = "results/exit_strategy_optimization.json"
    Path("results").mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(optimization_results, f, indent=2)
    
    print(f"   ✅ Mock optimization results saved to: {results_path}")
    
    # 2. Initialize position monitor with optimization support
    print("\n🔧 Initializing position monitor with optimization support...")
    config = {
        "position_monitor": {
            "monitoring_interval": 10,
            # These will be overridden by optimized parameters
            "confidence_thresholds": {
                "very_low": 0.2,
                "low": 0.4,
                "medium": 0.6,
                "high": 0.8
            }
        }
    }
    
    monitor = PositionMonitor(config)
    await monitor.initialize()
    
    # 3. Check optimization status
    print("\n📋 Checking optimization status...")
    status = monitor.get_optimization_status()
    
    print(f"   📊 Optimization loaded: {status['optimization_loaded']}")
    if status['optimization_loaded']:
        print(f"   🎯 Confidence thresholds: {status['confidence_thresholds']}")
        print(f"   💰 Profit target: {status['pnl_thresholds']['profit_target']:.1%}")
        print(f"   🛡️ Stop loss: {status['pnl_thresholds']['stop_loss']:.1%}")
        print(f"   ⏰ Max hold time: {status['max_position_age']/3600:.1f} hours")
    else:
        print("   ⚠️ No optimization loaded, using default parameters")
    
    # 4. Test position scenarios with optimized parameters
    print("\n🧪 Testing position scenarios with optimized parameters...")
    
    test_scenarios = [
        {
            "name": "High Confidence + High Profit (Optimized)",
            "position_data": {
                "position_id": "test_1",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "quantity": 1.0,
                "entry_price": 50000.0,
                "current_price": 52000.0,  # 4% profit
                "entry_time": datetime.now() - timedelta(minutes=30),
                "unrealized_pnl": 2000.0
            },
            "confidence": 0.9
        },
        {
            "name": "Medium Confidence + Medium Profit (Optimized)",
            "position_data": {
                "position_id": "test_2",
                "symbol": "ETHUSDT",
                "side": "LONG",
                "quantity": 10.0,
                "entry_price": 3000.0,
                "current_price": 3090.0,  # 3% profit
                "entry_time": datetime.now() - timedelta(minutes=45),
                "unrealized_pnl": 900.0
            },
            "confidence": 0.65
        },
        {
            "name": "Low Confidence + Any Profit (Optimized)",
            "position_data": {
                "position_id": "test_3",
                "symbol": "ADAUSDT",
                "side": "LONG",
                "quantity": 1000.0,
                "entry_price": 0.5,
                "current_price": 0.52,  # 4% profit
                "entry_time": datetime.now() - timedelta(minutes=20),
                "unrealized_pnl": 20.0
            },
            "confidence": 0.3
        },
        {
            "name": "Stop Loss Scenario (Optimized)",
            "position_data": {
                "position_id": "test_4",
                "symbol": "LINKUSDT",
                "side": "LONG",
                "quantity": 50.0,
                "entry_price": 15.0,
                "current_price": 14.25,  # -5% loss
                "entry_time": datetime.now() - timedelta(minutes=10),
                "unrealized_pnl": -37.5
            },
            "confidence": 0.7
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🔍 Scenario {i}: {scenario['name']}")
        print(f"   Confidence: {scenario['confidence']:.2f}")
        print(f"   PnL: {scenario['position_data']['unrealized_pnl']:.2f}")
        
        # Determine position action with optimized parameters
        action, reason = monitor._determine_position_action(
            scenario['position_data'], 
            scenario['confidence']
        )
        
        print(f"   Action: {action.value}")
        print(f"   Reason: {reason}")
        
        # Analyze the decision
        if action == PositionAction.TAKE_PROFIT:
            print("   ✅ Profit taking triggered (optimized)")
        elif action == PositionAction.PARTIAL_PROFIT:
            print("   ✅ Partial profit taking triggered (optimized)")
        elif action == PositionAction.STOP_LOSS:
            print("   ⚠️ Stop loss triggered (optimized)")
        elif action == PositionAction.FULL_CLOSE:
            print("   🚨 Full close triggered (optimized)")
        elif action == PositionAction.SCALE_DOWN:
            print("   📉 Scale down triggered (optimized)")
        elif action == PositionAction.STAY:
            print("   📍 Position maintained (optimized)")
    
    # 5. Test parameter refresh
    print("\n🔄 Testing parameter refresh...")
    
    # Create new optimization results
    new_optimization_results = create_mock_optimization_results_v2()
    
    # Refresh parameters
    monitor.refresh_optimized_parameters(new_optimization_results)
    
    # Check updated status
    updated_status = monitor.get_optimization_status()
    print(f"   ✅ Parameters refreshed")
    print(f"   🎯 New confidence thresholds: {updated_status['confidence_thresholds']}")
    print(f"   💰 New profit target: {updated_status['pnl_thresholds']['profit_target']:.1%}")
    
    # 6. Cleanup
    await monitor.cleanup()
    
    print("\n✅ Optimized exit strategy integration test completed!")
    
    # 7. Display key benefits
    print("\n🎯 Key Benefits of Optimized Exit Strategy:")
    print("1. ✅ Parameters optimized through backtesting")
    print("2. ✅ Confidence thresholds tuned for maximum performance")
    print("3. ✅ Profit-taking levels optimized for risk-reward balance")
    print("4. ✅ Stop-loss parameters calibrated for market conditions")
    print("5. ✅ Time-based exits optimized for trade duration")
    print("6. ✅ Regime-aware parameters for different market conditions")
    print("7. ✅ Dynamic parameter refresh without system restart")

def create_mock_optimization_results():
    """Create mock optimization results for testing."""
    
    return {
        "best_parameters": {
            "confidence_thresholds": {
                "very_low": 0.15,  # More aggressive
                "low": 0.35,       # More aggressive
                "medium": 0.55,    # More aggressive
                "high": 0.75       # More conservative
            },
            "profit_taking": {
                "base_profit_target": 0.035,  # 3.5% (optimized)
                "min_confidence_for_profit": 0.65,  # Higher threshold
                "confidence_profit_multiplier": 0.6,  # More scaling
                "scaling_levels": [0.3, 0.6, 0.8]  # Optimized levels
            },
            "stop_loss": {
                "base_stop_loss": -0.045,  # -4.5% (tighter)
                "atr_multiplier": 1.8,     # Higher ATR multiplier
                "volatility_adjustment_factor": 1.2
            },
            "time_based": {
                "max_hold_time": 9000,     # 2.5 hours (shorter)
                "min_hold_time": 600,     # 10 minutes
                "confidence_time_scaling_factor": 1.3
            },
            "trailing_stop": {
                "atr_multiplier": 1.8,     # Higher multiplier
                "min_distance": 0.012,    # 1.2% minimum
                "confidence_activation": 0.75  # Higher activation
            },
            "regime_aware": {
                "transition_penalty": 0.12,  # Higher penalty
                "regime_specific_scaling": 1.1  # More scaling
            }
        },
        "optimization_metrics": {
            "sharpe_ratio": 2.3,
            "profit_factor": 2.1,
            "max_drawdown": 0.11,
            "win_rate": 0.68,
            "total_return": 0.42,
            "num_trades": 134
        },
        "backtest_results": {
            "sharpe_ratio": 2.3,
            "profit_factor": 2.1,
            "max_drawdown": 0.11,
            "win_rate": 0.68,
            "total_return": 0.42,
            "num_trades": 134
        },
        "statistical_significance": {
            "sharpe_ratio_p_value": 0.018,
            "profit_factor_p_value": 0.012,
            "win_rate_p_value": 0.025,
            "overall_significance": 0.016
        },
        "regime_performance": {
            "trending": {
                "sharpe_ratio": 2.8,
                "profit_factor": 2.9,
                "win_rate": 0.75,
                "avg_return": 0.042
            },
            "ranging": {
                "sharpe_ratio": 2.1,
                "profit_factor": 2.0,
                "win_rate": 0.68,
                "avg_return": 0.031
            },
            "volatile": {
                "sharpe_ratio": 1.9,
                "profit_factor": 1.8,
                "win_rate": 0.62,
                "avg_return": 0.025
            }
        },
        "optimization_timestamp": datetime.now().isoformat()
    }

def create_mock_optimization_results_v2():
    """Create updated mock optimization results for refresh testing."""
    
    return {
        "best_parameters": {
            "confidence_thresholds": {
                "very_low": 0.18,  # Slightly different
                "low": 0.38,       # Slightly different
                "medium": 0.58,    # Slightly different
                "high": 0.78       # Slightly different
            },
            "profit_taking": {
                "base_profit_target": 0.032,  # 3.2% (slightly different)
                "min_confidence_for_profit": 0.68,  # Slightly higher
                "confidence_profit_multiplier": 0.65,  # Slightly more scaling
                "scaling_levels": [0.28, 0.58, 0.82]  # Slightly different
            },
            "stop_loss": {
                "base_stop_loss": -0.042,  # -4.2% (slightly different)
                "atr_multiplier": 1.9,     # Slightly higher
                "volatility_adjustment_factor": 1.25
            },
            "time_based": {
                "max_hold_time": 8500,     # 2.36 hours (slightly different)
                "min_hold_time": 650,     # 10.8 minutes
                "confidence_time_scaling_factor": 1.35
            },
            "trailing_stop": {
                "atr_multiplier": 1.9,     # Slightly higher
                "min_distance": 0.013,    # 1.3% minimum
                "confidence_activation": 0.78  # Slightly higher
            },
            "regime_aware": {
                "transition_penalty": 0.13,  # Slightly higher
                "regime_specific_scaling": 1.15  # Slightly more scaling
            }
        },
        "optimization_metrics": {
            "sharpe_ratio": 2.4,
            "profit_factor": 2.2,
            "max_drawdown": 0.10,
            "win_rate": 0.70,
            "total_return": 0.45,
            "num_trades": 142
        }
    }

async def main():
    """Main function."""
    try:
        await test_optimized_exit_strategy()
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())