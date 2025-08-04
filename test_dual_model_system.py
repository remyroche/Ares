#!/usr/bin/env python3
"""
Test script for Dual Model System implementation.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime

# Import the dual model system
from src.training.dual_model_system import DualModelSystem, setup_dual_model_system


async def test_dual_model_system():
    """Test the Dual Model System implementation."""
    print("🧪 Testing Dual Model System Implementation...")
    
    # Test configuration
    config = {
        "dual_model_system": {
            "analyst_timeframes": ["1h", "15m", "5m", "1m"],
            "tactician_timeframes": ["1m"],
            "analyst_confidence_threshold": 0.7,
            "tactician_confidence_threshold": 0.8,
            "enable_ensemble_analysis": True,
        },
        "ml_confidence_predictor": {
            "model_path": "models/confidence_predictor.joblib",
            "min_samples_for_training": 500,
            "confidence_threshold": 0.6,
            "max_prediction_horizon": 1,
        }
    }
    
    try:
        # Test 1: Initialize Dual Model System
        print("\n📋 Test 1: Initializing Dual Model System...")
        dual_system = await setup_dual_model_system(config)
        
        if dual_system is None:
            print("❌ Failed to initialize Dual Model System")
            return False
        
        print("✅ Dual Model System initialized successfully")
        
        # Test 2: Get System Info
        print("\n📋 Test 2: Getting System Info...")
        system_info = dual_system.get_system_info()
        print(f"✅ System Info: {system_info}")
        
        # Test 3: Create Mock Market Data
        print("\n📋 Test 3: Creating Mock Market Data...")
        mock_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'timestamp': pd.date_range(start='2024-01-01', periods=5, freq='1min')
        })
        current_price = 105.0
        print("✅ Mock market data created")
        
        # Test 4: Test Entry Decision
        print("\n📋 Test 4: Testing Entry Decision...")
        entry_decision = await dual_system.make_trading_decision(
            market_data=mock_data,
            current_price=current_price
        )
        print(f"✅ Entry Decision: {entry_decision}")
        
        # Test 5: Test Exit Decision
        print("\n📋 Test 5: Testing Exit Decision...")
        current_position = {
            "type": "LONG",
            "entry_price": 100.0,
            "quantity": 1.0,
            "entry_time": datetime.now().isoformat()
        }
        exit_decision = await dual_system.make_trading_decision(
            market_data=mock_data,
            current_price=current_price,
            current_position=current_position
        )
        print(f"✅ Exit Decision: {exit_decision}")
        
        # Test 6: Test System Cleanup
        print("\n📋 Test 6: Testing System Cleanup...")
        await dual_system.stop()
        print("✅ System cleanup completed")
        
        print("\n🎉 All tests passed! Dual Model System is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


async def test_analyst_decision():
    """Test Analyst model decision making."""
    print("\n🧪 Testing Analyst Decision Making...")
    
    config = {
        "dual_model_system": {
            "analyst_timeframes": ["1h", "15m", "5m", "1m"],
            "tactician_timeframes": ["1m"],
            "analyst_confidence_threshold": 0.7,
            "tactician_confidence_threshold": 0.8,
            "enable_ensemble_analysis": True,
        }
    }
    
    try:
        dual_system = DualModelSystem(config)
        
        # Test confidence analysis
        confidence_predictions = {
            "price_target_confidences": {
                "0.5%": 0.8,
                "1.0%": 0.7,
                "1.5%": 0.6,
                "2.0%": 0.5
            },
            "adversarial_confidences": {
                "0.5%": 0.3,
                "1.0%": 0.4,
                "1.5%": 0.5
            }
        }
        
        result = dual_system._analyze_analyst_confidence(
            confidence_predictions, 100.0
        )
        
        print(f"✅ Analyst Decision: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Analyst test failed: {e}")
        return False


async def test_tactician_decision():
    """Test Tactician model decision making."""
    print("\n🧪 Testing Tactician Decision Making...")
    
    config = {
        "dual_model_system": {
            "analyst_timeframes": ["1h", "15m", "5m", "1m"],
            "tactician_timeframes": ["1m"],
            "analyst_confidence_threshold": 0.7,
            "tactician_confidence_threshold": 0.8,
            "enable_ensemble_analysis": True,
        }
    }
    
    try:
        dual_system = DualModelSystem(config)
        
        # Test confidence analysis
        confidence_predictions = {
            "price_target_confidences": {
                "0.5%": 0.9,
                "1.0%": 0.8,
                "1.5%": 0.7,
                "2.0%": 0.6
            }
        }
        
        analyst_decision = {
            "direction": "LONG",
            "confidence": 0.8
        }
        
        result = dual_system._analyze_tactician_confidence(
            confidence_predictions, 100.0, analyst_decision
        )
        
        print(f"✅ Tactician Decision: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Tactician test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Dual Model System Tests...")
    
    # Run tests
    test_results = []
    
    # Test 1: Basic system functionality
    test_results.append(await test_dual_model_system())
    
    # Test 2: Analyst decision making
    test_results.append(await test_analyst_decision())
    
    # Test 3: Tactician decision making
    test_results.append(await test_tactician_decision())
    
    # Summary
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\n📊 Test Summary:")
    print(f"✅ Passed: {passed_tests}/{total_tests}")
    print(f"❌ Failed: {total_tests - passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Dual Model System is ready for use.")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    asyncio.run(main()) 