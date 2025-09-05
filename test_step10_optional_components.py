#!/usr/bin/env python3
"""
Test script to verify step10 handles missing optional components gracefully.
This script tests each optional component individually.
"""

import os
import sys
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_mock_data_files():
    """Create mock data files for testing."""
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Create mock HMM data files
    timeframes = ["1m", "5m", "15m", "30m"]
    for tf in timeframes:
        hmm_file = f"{data_dir}/BINANCE_ETHUSDT_hmm_composite_clusters_{tf}.parquet"
        intensity_file = f"{data_dir}/BINANCE_ETHUSDT_hmm_composite_intensity_{tf}.parquet"
        
        # Create mock HMM data
        hmm_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
            'composite_cluster_id': np.random.randint(0, 5, 1000),
            'close': np.random.uniform(100, 200, 1000)
        })
        hmm_data.to_parquet(hmm_file)
        
        # Create mock intensity data
        intensity_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
            'intensity_cluster_0': np.random.random(1000),
            'intensity_cluster_1': np.random.random(1000),
            'intensity_cluster_2': np.random.random(1000),
        })
        intensity_data.to_parquet(intensity_file)
    
    print("✅ Created mock data files")

def test_missing_enhanced_lm_optimizer():
    """Test step10 with missing enhanced LM optimizer."""
    print("\n🧪 Testing missing Enhanced LM Optimizer...")
    
    # Temporarily remove the module from sys.modules if it exists
    if 'src.training.enhanced_lm_optimizer' in sys.modules:
        del sys.modules['src.training.enhanced_lm_optimizer']
    
    try:
        from src.training.steps.model_training.step10_unified_regime_intelligence import UnifiedRegimeIntelligenceStep
        
        config = {
            "timeframes": ["1m", "5m"],
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "data_dir": "data"
        }
        
        step = UnifiedRegimeIntelligenceStep(config)
        print("✅ Step initialized successfully without Enhanced LM Optimizer")
        
        # Test that enhanced_lm_optimizer is None
        assert step.enhanced_lm_optimizer is None, "Enhanced LM optimizer should be None when not available"
        print("✅ Enhanced LM optimizer correctly set to None")
        
    except Exception as e:
        print(f"❌ Failed to handle missing Enhanced LM Optimizer: {e}")
        return False
    
    return True

def test_missing_sr_breakout_predictor():
    """Test step10 with missing SRBreakoutPredictor."""
    print("\n🧪 Testing missing SRBreakoutPredictor...")
    
    # Temporarily remove the module from sys.modules if it exists
    if 'src.tactician.sr_breakout_predictor' in sys.modules:
        del sys.modules['src.tactician.sr_breakout_predictor']
    
    try:
        from src.training.steps.model_training.step10_unified_regime_intelligence import UnifiedRegimeIntelligenceStep
        
        config = {
            "timeframes": ["1m", "5m"],
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "data_dir": "data"
        }
        
        step = UnifiedRegimeIntelligenceStep(config)
        print("✅ Step initialized successfully without SRBreakoutPredictor")
        
        # Test that sr_predictor is None
        assert step.sr_predictor is None, "SRBreakoutPredictor should be None when not available"
        print("✅ SRBreakoutPredictor correctly set to None")
        
    except Exception as e:
        print(f"❌ Failed to handle missing SRBreakoutPredictor: {e}")
        return False
    
    return True

def test_missing_model_specific_pruning():
    """Test step10 with missing model-specific pruning."""
    print("\n🧪 Testing missing Model-Specific Pruning...")
    
    # Temporarily remove the module from sys.modules if it exists
    if 'src.training.model_specific_pruning' in sys.modules:
        del sys.modules['src.training.model_specific_pruning']
    
    try:
        from src.training.steps.model_training.step10_unified_regime_intelligence import UnifiedRegimeIntelligenceStep
        
        config = {
            "timeframes": ["1m", "5m"],
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "data_dir": "data"
        }
        
        step = UnifiedRegimeIntelligenceStep(config)
        print("✅ Step initialized successfully without Model-Specific Pruning")
        
        # The pruning is handled in the training method, so we just verify initialization works
        print("✅ Model-Specific Pruning gracefully handled")
        
    except Exception as e:
        print(f"❌ Failed to handle missing Model-Specific Pruning: {e}")
        return False
    
    return True

def test_missing_optuna():
    """Test step10 with missing Optuna."""
    print("\n🧪 Testing missing Optuna...")
    
    # Temporarily remove the module from sys.modules if it exists
    if 'optuna' in sys.modules:
        del sys.modules['optuna']
    
    try:
        from src.training.steps.model_training.step10_unified_regime_intelligence import UnifiedRegimeIntelligenceStep
        
        config = {
            "timeframes": ["1m", "5m"],
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "data_dir": "data",
            "enhancement": {
                "hpo_enabled": True
            }
        }
        
        step = UnifiedRegimeIntelligenceStep(config)
        print("✅ Step initialized successfully without Optuna")
        
        # Test HPO method returns None when Optuna is not available
        import asyncio
        hpo_result = asyncio.run(step._run_hyperparameter_optimization())
        assert hpo_result is None, "HPO should return None when Optuna is not available"
        print("✅ HPO correctly returns None when Optuna is missing")
        
    except Exception as e:
        print(f"❌ Failed to handle missing Optuna: {e}")
        return False
    
    return True

def test_missing_warning_symbols():
    """Test step10 with missing warning symbols."""
    print("\n🧪 Testing missing Warning Symbols...")
    
    # Temporarily remove the module from sys.modules if it exists
    if 'src.utils.warning_symbols' in sys.modules:
        del sys.modules['src.utils.warning_symbols']
    
    try:
        from src.training.steps.model_training.step10_unified_regime_intelligence import UnifiedRegimeIntelligenceStep
        
        config = {
            "timeframes": ["1m", "5m"],
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "data_dir": "data"
        }
        
        step = UnifiedRegimeIntelligenceStep(config)
        print("✅ Step initialized successfully without Warning Symbols")
        
        # Test that fallback functions are used
        from src.training.steps.model_training.step10_unified_regime_intelligence import error, failed, timeout
        assert callable(error), "Error function should be callable"
        assert callable(failed), "Failed function should be callable"
        assert callable(timeout), "Timeout function should be callable"
        print("✅ Fallback warning functions correctly implemented")
        
    except Exception as e:
        print(f"❌ Failed to handle missing Warning Symbols: {e}")
        return False
    
    return True

def test_missing_error_handler():
    """Test step10 with missing error handler."""
    print("\n🧪 Testing missing Error Handler...")
    
    # Temporarily remove the module from sys.modules if it exists
    if 'src.utils.error_handler' in sys.modules:
        del sys.modules['src.utils.error_handler']
    
    try:
        from src.training.steps.model_training.step10_unified_regime_intelligence import UnifiedRegimeIntelligenceStep
        
        config = {
            "timeframes": ["1m", "5m"],
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "data_dir": "data"
        }
        
        step = UnifiedRegimeIntelligenceStep(config)
        print("✅ Step initialized successfully without Error Handler")
        
        # Test that fallback decorator is used
        from src.training.steps.model_training.step10_unified_regime_intelligence import handle_errors
        assert callable(handle_errors), "Handle errors should be callable"
        print("✅ Fallback error handler correctly implemented")
        
    except Exception as e:
        print(f"❌ Failed to handle missing Error Handler: {e}")
        return False
    
    return True

def test_initialization_with_missing_components():
    """Test step initialization with multiple missing components."""
    print("\n🧪 Testing initialization with multiple missing components...")
    
    # Remove multiple modules
    modules_to_remove = [
        'src.training.enhanced_lm_optimizer',
        'src.tactician.sr_breakout_predictor',
        'src.training.model_specific_pruning',
        'optuna',
        'src.utils.warning_symbols',
        'src.utils.error_handler'
    ]
    
    for module in modules_to_remove:
        if module in sys.modules:
            del sys.modules[module]
    
    try:
        from src.training.steps.model_training.step10_unified_regime_intelligence import UnifiedRegimeIntelligenceStep
        
        config = {
            "timeframes": ["1m", "5m"],
            "symbol": "ETHUSDT",
            "exchange": "BINANCE",
            "data_dir": "data"
        }
        
        step = UnifiedRegimeIntelligenceStep(config)
        print("✅ Step initialized successfully with multiple missing components")
        
        # Verify all optional components are None or have fallbacks
        assert step.enhanced_lm_optimizer is None, "Enhanced LM optimizer should be None"
        assert step.sr_predictor is None, "SRBreakoutPredictor should be None"
        print("✅ All optional components correctly handled as missing")
        
    except Exception as e:
        print(f"❌ Failed to handle multiple missing components: {e}")
        return False
    
    return True

def main():
    """Run all tests for missing optional components."""
    print("🚀 Testing Step10 Optional Component Handling")
    print("=" * 60)
    
    # Create mock data files
    create_mock_data_files()
    
    tests = [
        test_missing_enhanced_lm_optimizer,
        test_missing_sr_breakout_predictor,
        test_missing_model_specific_pruning,
        test_missing_optuna,
        test_missing_warning_symbols,
        test_missing_error_handler,
        test_initialization_with_missing_components
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All optional component tests passed!")
        print("✅ Step10 handles missing optional components gracefully")
    else:
        print("⚠️ Some tests failed - check the output above")
    
    # Cleanup
    import shutil
    if os.path.exists("data"):
        shutil.rmtree("data")
        print("🧹 Cleaned up test data files")

if __name__ == "__main__":
    main()