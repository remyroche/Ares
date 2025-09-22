"""
Test script to verify HMM clustering consolidation works correctly.
"""

import sys
import logging
from pathlib import Path

# Add the workspace root to the Python path
workspace_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(workspace_root))

# Also add the src directory
src_path = workspace_root / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all imports work correctly."""
    print("🧪 Testing HMM clustering consolidation imports...")
    
    try:
        # Test core imports
        print("  📦 Testing core imports...")
        from training.steps.market_analysis.hmm_clustering import (
            MatrixOptimizedClusterer,
            EnhancedMatrixOptimizedClusterer,
            OptimalRegimeClusteringOrchestrator
        )
        print("    ✅ Core imports successful")
        
        # Test metrics imports
        print("  📊 Testing metrics imports...")
        from training.steps.market_analysis.hmm_clustering import (
            BasicClusteringMetrics,
            DetailedClusteringMetrics,
            MetricsEvolutionReporter
        )
        print("    ✅ Metrics imports successful")
        
        # Test integration imports
        print("  🔗 Testing integration imports...")
        from training.steps.market_analysis.hmm_clustering import (
            EnhancedClusteringIntegration,
            FastFailManager
        )
        print("    ✅ Integration imports successful")
        
        # Test component import
        print("  🧩 Testing component import...")
        from training.steps.market_analysis.hmm_clustering import OptimalRegimeClusteringComponent
        print("    ✅ Component import successful")
        
        # Test config import
        print("  ⚙️ Testing config import...")
        from training.steps.market_analysis.hmm_clustering import HMMClusteringConfig
        print("    ✅ Config import successful")
        
        print("🎉 All imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_config_creation():
    """Test configuration creation."""
    print("\n🧪 Testing configuration creation...")
    
    try:
        from training.steps.market_analysis.hmm_clustering import HMMClusteringConfig
        
        # Test default config creation
        config = HMMClusteringConfig.create_default()
        print(f"  ✅ Default config created: {config.mode}")
        
        # Test config update
        config.update_config({'mode': 'standard'})
        print(f"  ✅ Config updated: {config.mode}")
        
        print("🎉 Configuration test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_component_factory_integration():
    """Test integration with component factory."""
    print("\n🧪 Testing component factory integration...")
    
    try:
        from training.steps.market_analysis.components.component_factory import ComponentFactory
        
        # Test that the component is registered
        factory = ComponentFactory()
        component_class = factory._components.get('hmm_clustering')
        
        if component_class is not None:
            print(f"  ✅ Component registered: {component_class.__name__}")
            
            # Test component instantiation
            from training.steps.market_analysis.components.base_component import ComponentConfig
            config = ComponentConfig()
            config.symbol = 'BTCUSDT'
            config.exchange = 'binance'
            config.timeframe = '15m'
            config.data_dir = 'historical_data'
            
            component = component_class(config)
            print(f"  ✅ Component instantiated: {component.get_component_name()}")
            
            print("🎉 Component factory integration successful!")
            return True
        else:
            print("❌ Component not found in factory")
            return False
            
    except Exception as e:
        print(f"❌ Component factory integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting HMM clustering consolidation tests...\n")
    
    # Set up logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
    
    tests = [
        test_imports,
        test_config_creation,
        test_component_factory_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! HMM clustering consolidation is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)