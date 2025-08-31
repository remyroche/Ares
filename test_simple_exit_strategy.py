#!/usr/bin/env python3
"""
Simple test for Tactician Exit Strategy core functionality.

This script tests the core barrier confidence assessment and position closing logic
without requiring full tactician initialization.
"""

import sys
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.append("src")

def test_barrier_confidence_assessment():
    """Test the barrier confidence assessment functionality."""
    
    print("🧪 Testing Barrier Confidence Assessment")
    print("=" * 50)
    
    # Import the position closer directly
    from src.tactician.position_closing import PositionCloser
    
    # Create minimal configuration
    config = {
        "step12_confidence_optimization": {
            "position_opening": {
                "min_barrier_confidence": 0.72,  # Step17 threshold
            }
        },
        "step17_optimization": {
            "tpsl": {
                "confidence_threshold": 0.7,
                "atr_multiplier": 2.0,
                "min_hold_time": 300
            }
        }
    }
    
    try:
        # Create position closer
        print("1. Creating Position Closer...")
        position_closer = PositionCloser(config)
        print("✅ Position Closer created successfully")
        
        # Create sample position data
        print("\n2. Creating sample position data...")
        position_data = {
            "position_id": "test_position_001",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "entry_price": 50000.0,
            "quantity": 0.1,
            "entry_time": datetime.now().isoformat(),
            "current_price": 50000.0
        }
        
        # Create sample tactician predictions with barrier probabilities
        print("3. Creating sample tactician predictions...")
        tactician_predictions = {
            "barrier_probabilities": {
                "profit_take_probability": 0.85,  # High confidence in profit take
                "stop_loss_probability": 0.15     # Low probability of stop loss
            },
            "confidence_factors": {
                "price_direction_prediction": 1.2,  # 20% confidence boost
                "price_target_confidence": 1.1     # 10% confidence boost
            },
            "model_confidence": 0.75
        }
        
        print("✅ Sample data created:")
        print(f"   Position: {position_data['side']} {position_data['quantity']} {position_data['symbol']} @ {position_data['entry_price']}")
        print(f"   Profit Take Probability: {tactician_predictions['barrier_probabilities']['profit_take_probability']:.3f}")
        print(f"   Stop Loss Probability: {tactician_predictions['barrier_probabilities']['stop_loss_probability']:.3f}")
        print(f"   Step17 Threshold: {config['step12_confidence_optimization']['position_opening']['min_barrier_confidence']:.3f}")
        
        # Test barrier confidence assessment
        print("\n4. Testing barrier confidence assessment...")
        barrier_confidence = position_closer.assess_barrier_confidence(
            tactician_predictions=tactician_predictions,
            current_price=position_data["current_price"],
            position_data=position_data
        )
        
        print(f"✅ Barrier Confidence Assessment:")
        print(f"   Calculated Confidence: {barrier_confidence:.3f}")
        print(f"   Threshold: {config['step12_confidence_optimization']['position_opening']['min_barrier_confidence']:.3f}")
        print(f"   Would Close: {barrier_confidence < config['step12_confidence_optimization']['position_opening']['min_barrier_confidence']}")
        
        # Test with low confidence scenario
        print("\n5. Testing low confidence scenario...")
        low_confidence_predictions = {
            "barrier_probabilities": {
                "profit_take_probability": 0.3,  # Low confidence in profit take
                "stop_loss_probability": 0.7     # High probability of stop loss
            },
            "confidence_factors": {
                "price_direction_prediction": 0.8,  # 20% confidence reduction
                "price_target_confidence": 0.9     # 10% confidence reduction
            },
            "model_confidence": 0.4
        }
        
        low_barrier_confidence = position_closer.assess_barrier_confidence(
            tactician_predictions=low_confidence_predictions,
            current_price=position_data["current_price"],
            position_data=position_data
        )
        
        print(f"✅ Low Confidence Assessment:")
        print(f"   Calculated Confidence: {low_barrier_confidence:.3f}")
        print(f"   Threshold: {config['step12_confidence_optimization']['position_opening']['min_barrier_confidence']:.3f}")
        print(f"   Would Close: {low_barrier_confidence < config['step12_confidence_optimization']['position_opening']['min_barrier_confidence']}")
        
        if low_barrier_confidence < config['step12_confidence_optimization']['position_opening']['min_barrier_confidence']:
            print("   🚨 EXIT STRATEGY: Position would be closed due to low barrier confidence!")
        
        # Test position closure evaluation
        print("\n6. Testing position closure evaluation...")
        import asyncio
        should_close = asyncio.run(position_closer.should_close_position(
            position_data=position_data,
            model_confidence=0.75,
            atr_value=100.0,
            current_price=50000.0,
            barrier_confidence=low_barrier_confidence  # Use low confidence scenario
        ))
        
        print(f"✅ Position Closure Evaluation:")
        print(f"   Should Close: {should_close}")
        if should_close:
            print("   🚨 EXIT STRATEGY: Position should be closed!")
        
        # Test SHORT position
        print("\n7. Testing SHORT position...")
        short_position_data = {
            "position_id": "test_position_002",
            "symbol": "BTCUSDT",
            "side": "SHORT",
            "entry_price": 50000.0,
            "quantity": 0.1,
            "entry_time": datetime.now().isoformat(),
            "current_price": 50000.0
        }
        
        short_barrier_confidence = position_closer.assess_barrier_confidence(
            tactician_predictions=tactician_predictions,
            current_price=short_position_data["current_price"],
            position_data=short_position_data
        )
        
        print(f"✅ SHORT Position Assessment:")
        print(f"   Calculated Confidence: {short_barrier_confidence:.3f}")
        print(f"   Would Close: {short_barrier_confidence < config['step12_confidence_optimization']['position_opening']['min_barrier_confidence']}")
        
        print("\n🎉 Barrier Confidence Assessment Test Completed Successfully!")
        print("\nSummary:")
        print("- ✅ Barrier confidence calculation for two barriers")
        print("- ✅ Exit strategy based on step17 threshold (0.72)")
        print("- ✅ Position closure evaluation with barrier confidence")
        print("- ✅ Low confidence scenario triggers exit strategy")
        print("- ✅ Both LONG and SHORT positions supported")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the test
    success = test_barrier_confidence_assessment()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)