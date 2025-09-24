#!/usr/bin/env python3
"""
Test script for enhanced exit strategy with confidence-based exits and profit-taking.

This script demonstrates the improved exit strategy functionality.
"""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.tactician.position_monitor import PositionMonitor, PositionAction
from src.utils.logger import system_logger

async def test_enhanced_exit_strategy():
    """Test the enhanced exit strategy with various scenarios."""
    
    logger = system_logger.getChild("ExitStrategyTest")
    logger.info("🧪 Testing Enhanced Exit Strategy")
    
    # Load enhanced configuration
    config = {
        "position_monitor": {
            "monitoring_interval": 5,
            "confidence_thresholds": {
                "very_low": 0.2,
                "low": 0.4, 
                "medium": 0.6,
                "high": 0.8
            },
            "pnl_thresholds": {
                "stop_loss": -0.05,
                "profit_target": 0.04,
                "scaling_levels": [0.25, 0.5, 0.75]
            },
            "profit_taking": {
                "confidence_scaling": True,
                "min_confidence_for_profit": 0.6,
                "confidence_profit_multiplier": 0.5,
                "tiered_profit_taking": True,
                "trailing_stop_enabled": True,
                "trailing_stop_atr_multiplier": 1.5
            }
        }
    }
    
    # Initialize position monitor
    monitor = PositionMonitor(config)
    await monitor.initialize()
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "High Confidence + High Profit",
            "position_data": {
                "position_id": "test_1",
                "symbol": "BTCUSDT",
                "side": "LONG",
                "quantity": 1.0,
                "entry_price": 50000.0,
                "current_price": 52000.0,  # 4% profit
                "entry_time": datetime.now() - timedelta(minutes=30),
                "unrealized_pnl": 2000.0  # 4% profit
            },
            "confidence": 0.9
        },
        {
            "name": "Medium Confidence + Medium Profit", 
            "position_data": {
                "position_id": "test_2",
                "symbol": "ETHUSDT",
                "side": "LONG", 
                "quantity": 10.0,
                "entry_price": 3000.0,
                "current_price": 3090.0,  # 3% profit
                "entry_time": datetime.now() - timedelta(minutes=45),
                "unrealized_pnl": 900.0  # 3% profit
            },
            "confidence": 0.65
        },
        {
            "name": "Low Confidence + Any Profit",
            "position_data": {
                "position_id": "test_3", 
                "symbol": "ADAUSDT",
                "side": "LONG",
                "quantity": 1000.0,
                "entry_price": 0.5,
                "current_price": 0.52,  # 4% profit
                "entry_time": datetime.now() - timedelta(minutes=20),
                "unrealized_pnl": 20.0  # 4% profit
            },
            "confidence": 0.3
        },
        {
            "name": "Very Low Confidence + Any PnL",
            "position_data": {
                "position_id": "test_4",
                "symbol": "DOTUSDT", 
                "side": "LONG",
                "quantity": 100.0,
                "entry_price": 7.0,
                "current_price": 7.14,  # 2% profit
                "entry_time": datetime.now() - timedelta(minutes=15),
                "unrealized_pnl": 14.0  # 2% profit
            },
            "confidence": 0.15
        },
        {
            "name": "Stop Loss Scenario",
            "position_data": {
                "position_id": "test_5",
                "symbol": "LINKUSDT",
                "side": "LONG",
                "quantity": 50.0,
                "entry_price": 15.0,
                "current_price": 14.25,  # -5% loss
                "entry_time": datetime.now() - timedelta(minutes=10),
                "unrealized_pnl": -37.5  # -5% loss
            },
            "confidence": 0.7
        },
        {
            "name": "Time-based Exit",
            "position_data": {
                "position_id": "test_6",
                "symbol": "UNIUSDT",
                "side": "LONG", 
                "quantity": 20.0,
                "entry_price": 6.0,
                "current_price": 6.12,  # 2% profit
                "entry_time": datetime.now() - timedelta(hours=4),  # 4 hours old
                "unrealized_pnl": 2.4  # 2% profit
            },
            "confidence": 0.8
        }
    ]
    
    logger.info("📊 Running Exit Strategy Test Scenarios")
    logger.info("=" * 60)
    
    for i, scenario in enumerate(test_scenarios, 1):
        logger.info(f"\n🔍 Scenario {i}: {scenario['name']}")
        logger.info(f"   Confidence: {scenario['confidence']:.2f}")
        logger.info(f"   PnL: {scenario['position_data']['unrealized_pnl']:.2f}")
        
        # Determine position action
        action, reason = monitor._determine_position_action(
            scenario['position_data'], 
            scenario['confidence']
        )
        
        logger.info(f"   Action: {action.value}")
        logger.info(f"   Reason: {reason}")
        
        # Analyze the decision
        if action == PositionAction.TAKE_PROFIT:
            logger.info("   ✅ Profit taking triggered")
        elif action == PositionAction.PARTIAL_PROFIT:
            logger.info("   ✅ Partial profit taking triggered")
        elif action == PositionAction.STOP_LOSS:
            logger.info("   ⚠️  Stop loss triggered")
        elif action == PositionAction.FULL_CLOSE:
            logger.info("   🚨 Full close triggered")
        elif action == PositionAction.SCALE_DOWN:
            logger.info("   📉 Scale down triggered")
        elif action == PositionAction.STAY:
            logger.info("   📍 Position maintained")
        else:
            logger.info(f"   ❓ Unknown action: {action.value}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Enhanced Exit Strategy Test Completed")
    
    # Test confidence-based profit scaling
    logger.info("\n🎯 Testing Confidence-Based Profit Scaling")
    logger.info("-" * 40)
    
    base_profit_target = 0.04  # 4%
    confidence_multiplier = 0.5
    
    for confidence in [0.6, 0.7, 0.8, 0.9]:
        confidence_factor = 1.0 - (confidence - 0.5) * confidence_multiplier
        scaled_target = base_profit_target * confidence_factor
        logger.info(f"Confidence {confidence:.1f}: Target {scaled_target:.1%} (Factor: {confidence_factor:.2f})")
    
    await monitor.cleanup()

if __name__ == "__main__":
    asyncio.run(test_enhanced_exit_strategy())