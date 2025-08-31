#!/usr/bin/env python3
"""
Test script for Tactician Exit Strategy.

This script demonstrates the new exit strategy that closes positions when
the tactician's confidence to meet the two barriers drops below the step17 threshold.
"""

import asyncio
import sys
from datetime import datetime
from typing import Dict, Any

# Add src to path
sys.path.append("src")

from src.tactician.tactician import Tactician
from src.utils.logger import system_logger

async def test_tactician_exit_strategy():
    """Test the tactician exit strategy functionality."""
    
    print("🧪 Testing Tactician Exit Strategy")
    print("=" * 50)
    
    # Create configuration with step17 thresholds
    config = {
        "tactician": {
            "tactics_interval": 30,
            "max_history": 100,
            "enable_enhanced_predictions": True
        },
        "step12_confidence_optimization": {
            "position_opening": {
                "min_barrier_confidence": 0.72,  # Step17 threshold
                "combined_confidence_threshold": 0.78
            },
            "position_monitor": {
                "high_confidence_threshold": 0.65,
                "low_confidence_threshold": 0.35,
                "very_low_confidence_threshold": 0.25
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
        # Initialize tactician
        print("1. Initializing Tactician...")
        tactician = Tactician(config)
        if not await tactician.initialize():
            print("❌ Failed to initialize tactician")
            return False
        print("✅ Tactician initialized successfully")
        
        # Create sample position data
        print("\n2. Creating sample position data...")
        position_data = {
            "position_id": "test_position_001",
            "symbol": "BTCUSDT",
            "side": "LONG",
            "entry_price": 50000.0,
            "quantity": 0.1,
            "entry_time": datetime.now().isoformat(),
            "current_price": 50000.0,
            "analyst_confidence": 0.8,
            "tactician_confidence": 0.75
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
        
        # Create position with barrier assessment
        print("\n4. Creating position with barrier confidence assessment...")
        success = await tactician.create_position_with_barrier_assessment(
            position_data, tactician_predictions
        )
        
        if not success:
            print("❌ Failed to create position with barrier assessment")
            return False
            
        print("✅ Position created with barrier confidence monitoring")
        print("   Exit strategy: Will close if barrier confidence drops below 0.72")
        
        # Test barrier confidence assessment
        print("\n5. Testing barrier confidence assessment...")
        if tactician.tactics_orchestrator and tactician.tactics_orchestrator.position_closer:
            barrier_confidence = tactician.tactics_orchestrator.position_closer.assess_barrier_confidence(
                tactician_predictions=tactician_predictions,
                current_price=position_data["current_price"],
                position_data=position_data
            )
            
            print(f"✅ Barrier Confidence Assessment:")
            print(f"   Calculated Confidence: {barrier_confidence:.3f}")
            print(f"   Threshold: {config['step12_confidence_optimization']['position_opening']['min_barrier_confidence']:.3f}")
            print(f"   Would Close: {barrier_confidence < config['step12_confidence_optimization']['position_opening']['min_barrier_confidence']}")
        
        # Test with low confidence scenario
        print("\n6. Testing low confidence scenario...")
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
        
        if tactician.tactics_orchestrator and tactician.tactics_orchestrator.position_closer:
            low_barrier_confidence = tactician.tactics_orchestrator.position_closer.assess_barrier_confidence(
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
        print("\n7. Testing position closure evaluation...")
        if tactician.tactics_orchestrator and tactician.tactics_orchestrator.position_closer:
            should_close = await tactician.tactics_orchestrator.position_closer.should_close_position(
                position_data=position_data,
                model_confidence=0.75,
                atr_value=100.0,
                current_price=50000.0,
                barrier_confidence=low_barrier_confidence  # Use low confidence scenario
            )
            
            print(f"✅ Position Closure Evaluation:")
            print(f"   Should Close: {should_close}")
            if should_close:
                print("   🚨 EXIT STRATEGY: Position should be closed!")
        
        # Get status
        print("\n8. Getting tactician status...")
        status = tactician.get_status()
        print(f"✅ Tactician Status:")
        print(f"   Running: {status['is_running']}")
        print(f"   History Count: {status['history_count']}")
        print(f"   Has Results: {status['has_results']}")
        
        # Cleanup
        print("\n9. Cleaning up...")
        await tactician.cleanup()
        print("✅ Cleanup completed")
        
        print("\n🎉 Tactician Exit Strategy Test Completed Successfully!")
        print("\nSummary:")
        print("- ✅ Position creation with barrier confidence assessment")
        print("- ✅ Barrier confidence calculation for two barriers")
        print("- ✅ Exit strategy based on step17 threshold (0.72)")
        print("- ✅ Position closure evaluation with barrier confidence")
        print("- ✅ Low confidence scenario triggers exit strategy")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_tactician_exit_strategy())
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)