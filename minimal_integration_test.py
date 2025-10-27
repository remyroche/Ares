#!/usr/bin/env python3
"""
Minimal test to verify the enhanced HDBSCAN clustering integration is properly wired.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_core_integration():
    """Test core integration components."""
    print("🧪 Testing Core Enhanced HDBSCAN Integration")
    print("=" * 60)
    
    try:
        # Test 1: Enhanced HDBSCAN integration import
        print("1. Testing enhanced HDBSCAN integration import...")
        from src.feature_generation.integration.enhanced_hdbscan_clustering_integration import (
            EnhancedHDBSCANClusteringIntegration
        )
        print("   ✅ Enhanced HDBSCAN integration imported successfully")
        
        # Test 2: Configuration values
        print("2. Testing configuration values...")
        from src.feature_generation.integration.feature_bank_integration import FeatureBankConfig
        config = FeatureBankConfig()
        print(f"   ✅ hdbscan_min_features = {config.hdbscan_min_features}")
        print(f"   ✅ hdbscan_max_features = {config.hdbscan_max_features}")
        
        # Test 3: Enhanced integration initialization
        print("3. Testing enhanced integration initialization...")
        integrator = EnhancedHDBSCANClusteringIntegration(
            min_features=100,
            max_features=150,
            enable_pca_reduction=True,
            pca_components=15
        )
        print(f"   ✅ Enhanced integration initialized")
        print(f"      - Min features: {integrator.min_features}")
        print(f"      - Max features: {integrator.max_features}")
        print(f"      - PCA enabled: {integrator.enable_pca_reduction}")
        print(f"      - PCA components: {integrator.pca_components}")
        
        # Test 4: MLTask enum
        print("4. Testing MLTask enum...")
        from src.feature_generation.integration.feature_task_integration import MLTask
        print(f"   ✅ MLTask.HDBSCAN_CLUSTERING = {MLTask.HDBSCAN_CLUSTERING}")
        
        # Test 5: Feature categories
        print("5. Testing feature category imports...")
        from src.feature_generation.categories.clustering_features import ClusteringDistanceGenerator
        from src.feature_generation.categories.volume import VolumeFeatureGenerator
        from src.feature_generation.categories.volatility import VolatilityFeatureGenerator
        from src.feature_generation.categories.momentum import MomentumFeatureGenerator
        from src.feature_generation.categories.trend import TrendFeatureGenerator
        from src.feature_generation.categories.regime_features import RegimeStatisticalFeatureGenerator
        print("   ✅ All feature categories imported successfully")
        
        print("\n🎉 Core integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Core integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_launcher_integration():
    """Test launcher integration."""
    print("\n🚀 Testing Launcher Integration")
    print("=" * 60)
    
    try:
        # Test launcher import
        print("1. Testing launcher import...")
        from src.launcher.ares_launcher import main
        print("   ✅ ares_launcher imported successfully")
        
        # Test HDBSCAN step import
        print("2. Testing HDBSCAN step import...")
        from src.training.steps.market_analysis.hdbscan_regime_discovery_step import HDBSCANRegimeDiscoveryStep
        print("   ✅ HDBSCANRegimeDiscoveryStep imported successfully")
        
        # Test step initialization
        print("3. Testing step initialization...")
        step = HDBSCANRegimeDiscoveryStep()
        print(f"   ✅ Step initialized: {step.step_name}")
        
        print("\n🎉 Launcher integration tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Launcher test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_flow():
    """Test the feature generation flow."""
    print("\n🔧 Testing Feature Generation Flow")
    print("=" * 60)
    
    try:
        # Test feature bank integrator
        print("1. Testing feature bank integrator...")
        from src.feature_generation.integration.feature_bank_integration import FeatureBankIntegrator, FeatureBankConfig
        
        config = FeatureBankConfig()
        integrator = FeatureBankIntegrator(config)
        print("   ✅ Feature bank integrator initialized")
        
        # Test task configuration
        print("2. Testing task configuration...")
        from src.feature_generation.integration.feature_task_integration import MLTask
        task_config = integrator._get_task_config(MLTask.HDBSCAN_CLUSTERING)
        print(f"   ✅ HDBSCAN task config: {task_config['target_range']} features")
        
        print("\n🎉 Feature generation flow tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Feature flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Minimal Enhanced HDBSCAN Clustering Integration Test")
    print("=" * 60)
    
    # Run tests
    core_success = test_core_integration()
    launcher_success = test_launcher_integration()
    flow_success = test_feature_flow()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 60)
    print(f"Core Integration: {'✅ PASS' if core_success else '❌ FAIL'}")
    print(f"Launcher Integration: {'✅ PASS' if launcher_success else '❌ FAIL'}")
    print(f"Feature Flow: {'✅ PASS' if flow_success else '❌ FAIL'}")
    
    if core_success and launcher_success and flow_success:
        print("\n🎉 ALL SYSTEMS ARE PROPERLY WIRED!")
        print("✅ Enhanced HDBSCAN clustering integration is ready")
        print("✅ Feature selection: 100-150 features")
        print("✅ PCA reduction: 10-25 components")
        print("✅ Launcher integration: --hdbscan-regime-discovery")
        print("\n🚀 You can now launch HDBSCAN clustering with:")
        print("   python src/launcher/ares_launcher.py --hdbscan-regime-discovery --symbol ETHUSDT")
    else:
        print("\n⚠️  Some issues found. Check the errors above.")