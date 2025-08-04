#!/usr/bin/env python3
"""
Test script to validate SR breakout predictor migration to enhanced predictive ensembles.

This script validates that:
1. Enhanced predictive ensembles include SR context features
2. Tactician uses enhanced predictive ensembles instead of SR breakout predictor
3. All functionality is preserved
4. No dependencies on the old SR breakout predictor remain
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime

import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import CONFIG
from src.analyst.predictive_ensembles.regime_ensembles.base_ensemble import BaseEnsemble
from src.analyst.sr_analyzer import SRLevelAnalyzer


async def test_enhanced_predictive_ensembles():
    """Test that enhanced predictive ensembles include SR context features."""
    print("🧪 Testing Enhanced Predictive Ensembles with SR Context Features")
    
    try:
        # Initialize base ensemble
        ensemble = BaseEnsemble(CONFIG, "test_ensemble")
        
        # Check that SR features are included in sequence_features
        sr_features = [
            "distance_to_sr",
            "sr_strength", 
            "sr_type",
            "price_position",
            "momentum_5",
            "momentum_10",
            "volume_ratio",
            "volatility"
        ]
        
        missing_features = []
        for feature in sr_features:
            if feature not in ensemble.sequence_features:
                missing_features.append(feature)
        
        if missing_features:
            print(f"❌ Missing SR features in sequence_features: {missing_features}")
            return False
        else:
            print("✅ All SR context features present in sequence_features")
        
        # Test SR context feature calculation
        print("\n🔍 Testing SR context feature calculation...")
        
        # Create test data
        dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1h')
        test_data = pd.DataFrame({
            'open': np.random.uniform(2000, 2100, len(dates)),
            'high': np.random.uniform(2100, 2200, len(dates)),
            'low': np.random.uniform(1900, 2000, len(dates)),
            'close': np.random.uniform(2000, 2100, len(dates)),
            'volume': np.random.uniform(1000, 10000, len(dates))
        }, index=dates)
        
        # Calculate SR features
        sr_features_df = ensemble._calculate_sr_context_features(test_data)
        
        if sr_features_df is not None and not sr_features_df.empty:
            print(f"✅ SR context features calculated successfully")
            print(f"   Features shape: {sr_features_df.shape}")
            print(f"   Features: {list(sr_features_df.columns)}")
        else:
            print("❌ Failed to calculate SR context features")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing enhanced predictive ensembles: {e}")
        return False


async def test_tactician_integration():
    """Test that tactician uses enhanced predictive ensembles."""
    print("\n🧪 Testing Tactician Integration")
    
    try:
        # Import tactician
        from src.tactician.tactician import Tactician
        
        # Initialize tactician
        tactician = Tactician(CONFIG)
        await tactician.initialize()
        
        # Check that SR breakout predictor is deprecated
        if hasattr(tactician, 'sr_breakout_predictor'):
            if tactician.sr_breakout_predictor is None:
                print("✅ SR breakout predictor properly deprecated in tactician")
            else:
                print("⚠️ SR breakout predictor still initialized in tactician")
        
        # Check that enhanced prediction method exists
        if hasattr(tactician, '_get_sr_breakout_prediction_enhanced'):
            print("✅ Enhanced SR breakout prediction method available")
        else:
            print("❌ Enhanced SR breakout prediction method missing")
            return False
        
        # Check that fallback method exists
        if hasattr(tactician, '_get_sr_breakout_prediction_fallback'):
            print("✅ Fallback SR breakout prediction method available")
        else:
            print("❌ Fallback SR breakout prediction method missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing tactician integration: {e}")
        return False


async def test_dependency_removal():
    """Test that SR breakout predictor dependencies are removed."""
    print("\n🧪 Testing Dependency Removal")
    
    try:
        # Check that SR breakout predictor file exists (for reference)
        sr_breakout_file = "src/analyst/sr_breakout_predictor.py"
        if os.path.exists(sr_breakout_file):
            print("ℹ️ SR breakout predictor file still exists (for reference)")
        else:
            print("✅ SR breakout predictor file removed")
        
        # Check imports in tactician
        tactician_file = "src/tactician/tactician.py"
        with open(tactician_file, 'r') as f:
            tactician_content = f.read()
            
        if "from src.analyst.sr_breakout_predictor import" in tactician_content:
            print("❌ SR breakout predictor import still present in tactician")
            return False
        else:
            print("✅ SR breakout predictor import removed from tactician")
        
        # Check imports in training manager
        training_manager_file = "src/training/enhanced_training_manager.py"
        with open(training_manager_file, 'r') as f:
            training_manager_content = f.read()
            
        if "from src.analyst.sr_breakout_predictor import" in training_manager_content:
            print("❌ SR breakout predictor import still present in training manager")
            return False
        else:
            print("✅ SR breakout predictor import removed from training manager")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing dependency removal: {e}")
        return False


async def test_functionality_preservation():
    """Test that all functionality is preserved."""
    print("\n🧪 Testing Functionality Preservation")
    
    try:
        # Test that SR analyzer still works
        print("🔍 Testing SR Analyzer functionality...")
        sr_analyzer = SRLevelAnalyzer(CONFIG.get("analyst", {}).get("sr_analyzer", {}))
        await sr_analyzer.initialize()
        
        # Test SR zone detection
        test_price = 2000.0
        sr_context = sr_analyzer.detect_sr_zone_proximity(test_price)
        
        if sr_context is not None:
            print("✅ SR analyzer functionality preserved")
        else:
            print("❌ SR analyzer functionality broken")
            return False
        
        # Test that predictive ensembles can be trained
        print("🔍 Testing Predictive Ensembles training...")
        
        # Create test data for training
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1h')
        training_data = pd.DataFrame({
            'open': np.random.uniform(2000, 2100, len(dates)),
            'high': np.random.uniform(2100, 2200, len(dates)),
            'low': np.random.uniform(1900, 2000, len(dates)),
            'close': np.random.uniform(2000, 2100, len(dates)),
            'volume': np.random.uniform(1000, 10000, len(dates)),
            'rsi': np.random.uniform(0, 100, len(dates)),
            'macd': np.random.uniform(-10, 10, len(dates)),
            'atr': np.random.uniform(10, 50, len(dates)),
            'adx': np.random.uniform(0, 100, len(dates))
        }, index=dates)
        
        # Create target data
        targets = pd.Series(np.random.choice(['BULL', 'BEAR', 'SIDEWAYS'], len(dates)), index=dates)
        
        # Test ensemble training
        ensemble = BaseEnsemble(CONFIG, "test_ensemble")
        ensemble.train_ensemble(training_data, targets)
        
        if ensemble.trained:
            print("✅ Predictive ensemble training functionality preserved")
        else:
            print("❌ Predictive ensemble training functionality broken")
            return False
        
        # Test prediction
        test_features = training_data.tail(1)
        prediction = ensemble.get_prediction(test_features)
        
        if prediction and 'prediction' in prediction:
            print("✅ Predictive ensemble prediction functionality preserved")
        else:
            print("❌ Predictive ensemble prediction functionality broken")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing functionality preservation: {e}")
        return False


async def main():
    """Run all validation tests."""
    print("🚀 SR Breakout Predictor Migration Validation")
    print("=" * 50)
    
    tests = [
        ("Enhanced Predictive Ensembles", test_enhanced_predictive_ensembles),
        ("Tactician Integration", test_tactician_integration),
        ("Dependency Removal", test_dependency_removal),
        ("Functionality Preservation", test_functionality_preservation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            result = await test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Migration successful!")
        return True
    else:
        print("⚠️ Some tests failed. Please review the migration.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate SR breakout predictor migration")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    success = asyncio.run(main())
    sys.exit(0 if success else 1) 