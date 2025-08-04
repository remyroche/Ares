#!/usr/bin/env python3
"""
Test script for Position Monitor functionality.

This script demonstrates how the position monitor re-assesses confidence scores
and position decisions every 10 seconds when positions are open.
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.tactician.position_monitor import setup_position_monitor, PositionMonitor
from src.utils.logger import setup_logging


async def test_position_monitor():
    """Test the position monitor functionality."""
    print("🧪 Testing Position Monitor with 10-second confidence re-assessment")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    
    # Create test configuration
    config = {
        "position_monitoring_interval": 10,  # 10 seconds
        "max_assessment_history": 100,
        "high_risk_threshold": 0.8,
        "medium_risk_threshold": 0.6,
        "low_risk_threshold": 0.3,
        "position_division": {
            "entry_confidence_threshold": 0.7,
            "additional_position_threshold": 0.8,
            "max_positions": 3,
            "take_profit_confidence_decrease": 0.1,
            "take_profit_short_term_decrease": 0.08,
            "stop_loss_confidence_threshold": 0.3,
            "stop_loss_short_term_threshold": 0.24,
            "stop_loss_price_threshold": -0.05,
            "full_close_confidence_threshold": 0.2,
            "full_close_short_term_threshold": 0.16,
            "max_position_hold_hours": 12.0,
        }
    }
    
    # Setup position monitor
    position_monitor = await setup_position_monitor(config)
    if not position_monitor:
        print("❌ Failed to setup position monitor")
        return
    
    print("✅ Position Monitor initialized successfully")
    
    # Add test positions
    test_positions = {
        "pos_001": {
            "symbol": "ETHUSDT",
            "direction": "LONG",
            "entry_price": 1850.0,
            "current_price": 1860.0,
            "position_size": 0.1,
            "leverage": 1.0,
            "entry_confidence": 0.75,
            "entry_timestamp": datetime.now().isoformat(),
            "time_in_position_hours": 2.5,
            "market_volatility": 0.15,
            "trend_strength": 0.6,
            "base_confidence": 0.7,
        },
        "pos_002": {
            "symbol": "BTCUSDT",
            "direction": "SHORT",
            "entry_price": 42000.0,
            "current_price": 41800.0,
            "position_size": 0.05,
            "leverage": 2.0,
            "entry_confidence": 0.65,
            "entry_timestamp": datetime.now().isoformat(),
            "time_in_position_hours": 1.0,
            "market_volatility": 0.25,
            "trend_strength": 0.4,
            "base_confidence": 0.6,
        },
        "pos_003": {
            "symbol": "ADAUSDT",
            "direction": "LONG",
            "entry_price": 0.45,
            "current_price": 0.44,
            "position_size": 0.2,
            "leverage": 3.0,
            "entry_confidence": 0.55,
            "entry_timestamp": datetime.now().isoformat(),
            "time_in_position_hours": 4.0,
            "market_volatility": 0.35,
            "trend_strength": 0.2,
            "base_confidence": 0.4,
        }
    }
    
    # Add positions to monitor
    for position_id, position_data in test_positions.items():
        position_monitor.add_position(position_id, position_data)
        print(f"➕ Added position {position_id}: {position_data['symbol']} {position_data['direction']}")
    
    print(f"\n🔍 Monitoring {len(test_positions)} positions every 10 seconds...")
    print("Press Ctrl+C to stop monitoring")
    print("-" * 60)
    
    try:
        # Start monitoring for 60 seconds (6 assessments)
        monitoring_duration = 60
        start_time = datetime.now()
        
        while (datetime.now() - start_time).total_seconds() < monitoring_duration:
            await asyncio.sleep(10)  # Wait for next assessment
            
            # Get latest assessments
            assessments = position_monitor.get_assessment_history(limit=3)
            
            print(f"\n📊 Assessment at {datetime.now().strftime('%H:%M:%S')}:")
            for assessment in assessments:
                print(f"  Position {assessment.position_id}:")
                print(f"    Confidence: {assessment.current_confidence:.3f} ({assessment.confidence_change:+.3f})")
                print(f"    Action: {assessment.recommended_action.value}")
                print(f"    Reason: {assessment.action_reason}")
                print(f"    Risk: {assessment.risk_level}, Market: {assessment.market_conditions}")
                print()
        
        print("✅ Monitoring test completed successfully")
        
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped by user")
    
    finally:
        # Stop monitoring
        await position_monitor.stop_monitoring()
        print("🛑 Position Monitor stopped")
        
        # Show final statistics
        final_assessments = position_monitor.get_assessment_history()
        print(f"\n📈 Final Statistics:")
        print(f"  Total assessments: {len(final_assessments)}")
        print(f"  Active positions: {len(position_monitor.get_active_positions())}")
        
        # Show action breakdown
        action_counts = {}
        for assessment in final_assessments:
            action = assessment.recommended_action.value
            action_counts[action] = action_counts.get(action, 0) + 1
        
        print(f"  Action breakdown:")
        for action, count in action_counts.items():
            print(f"    {action}: {count}")


async def test_confidence_reassessment():
    """Test confidence score re-assessment logic."""
    print("\n🧪 Testing Confidence Score Re-assessment Logic")
    print("=" * 60)
    
    # Create a simple position monitor for testing
    config = {"position_monitoring_interval": 5}  # 5 seconds for faster testing
    monitor = PositionMonitor(config)
    await monitor.initialize()
    
    # Test position with different scenarios
    test_scenarios = [
        {
            "name": "High Confidence - Should Stay",
            "position_data": {
                "base_confidence": 0.8,
                "time_in_position_hours": 1.0,
                "market_volatility": 0.1,
                "entry_confidence": 0.8,
                "current_price": 100.0,
                "entry_price": 100.0,
                "position_size": 0.1,
                "leverage": 1.0,
            }
        },
        {
            "name": "Low Confidence - Should Exit",
            "position_data": {
                "base_confidence": 0.2,
                "time_in_position_hours": 3.0,
                "market_volatility": 0.3,
                "entry_confidence": 0.7,
                "current_price": 95.0,
                "entry_price": 100.0,
                "position_size": 0.2,
                "leverage": 2.0,
            }
        },
        {
            "name": "High Risk - Should Hedge",
            "position_data": {
                "base_confidence": 0.6,
                "time_in_position_hours": 2.0,
                "market_volatility": 0.4,
                "entry_confidence": 0.7,
                "current_price": 98.0,
                "entry_price": 100.0,
                "position_size": 0.3,
                "leverage": 3.0,
            }
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📋 Scenario {i}: {scenario['name']}")
        
        # Add position to monitor
        position_id = f"test_pos_{i}"
        monitor.add_position(position_id, scenario['position_data'])
        
        # Wait for assessment
        await asyncio.sleep(6)  # Wait for one assessment cycle
        
        # Get assessment
        assessment = monitor.get_position_status(position_id)
        if assessment and assessment.get('latest_assessment'):
            latest = assessment['latest_assessment']
            print(f"  Confidence: {latest.current_confidence:.3f}")
            print(f"  Action: {latest.recommended_action.value}")
            print(f"  Reason: {latest.action_reason}")
        else:
            print("  No assessment available")
    
    await monitor.stop_monitoring()
    print("\n✅ Confidence re-assessment test completed")


async def main():
    """Main test function."""
    print("🚀 Position Monitor Test Suite")
    print("=" * 60)
    
    # Test 1: Basic position monitoring
    await test_position_monitor()
    
    # Test 2: Confidence re-assessment logic
    await test_confidence_reassessment()
    
    print("\n🎉 All tests completed successfully!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 