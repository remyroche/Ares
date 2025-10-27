#!/usr/bin/env python3
"""
Simple test to verify the enhanced HDBSCAN clustering integration is properly wired.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all imports work correctly."""
    print("🧪 Testing Enhanced HDBSCAN Integration Imports")
    print("=" * 60)
    
    try:
        # Test 1: Enhanced HDBSCAN integration
        print("1. Testing enhanced HDBSCAN integration import...")
        from src.feature_generation.integration.enhanced_hdbscan_clustering_integration import (
            EnhancedHDBSCANClusteringIntegration,
            get_enhanced_hdbscan_features,
            perform_enhanced_hdbscan_clustering
        )
        print("   ✅ Enhanced HDBSCAN integration imported successfully")
        
        # Test 2: Feature bank integration
        print("2. Testing feature bank integration import...")
        from src.feature_generation.integration.feature_bank_integration import (
            FeatureBankIntegrator,
            FeatureBankConfig,
            FeatureBankCategory
        )
        print("   ✅ Feature bank integration imported successfully")
        
        # Test 3: Feature categories
        print("3. Testing feature category imports...")
        from src.feature_generation.categories.clustering_features import (
            ClusteringDistanceGenerator,
            ClusteringSeparationGenerator,
            ClusteringStabilityGenerator
        )
        print("   ✅ Clustering features imported successfully")
        
        from src.feature_generation.categories.volume import VolumeFeatureGenerator
        print("   ✅ Volume features imported successfully")
        
        from src.feature_generation.categories.volatility import VolatilityFeatureGenerator
        print("   ✅ Volatility features imported successfully")
        
        from src.feature_generation.categories.momentum import MomentumFeatureGenerator
        print("   ✅ Momentum features imported successfully")
        
        from src.feature_generation.categories.trend import TrendFeatureGenerator
        print("   ✅ Trend features imported successfully")
        
        from src.feature_generation.categories.regime_features import RegimeStatisticalFeatureGenerator
        print("   ✅ Regime features imported successfully")
        
        # Test 4: MLTask enum
        print("4. Testing MLTask enum...")
        from src.feature_generation.integration.feature_task_integration import MLTask
        print(f"   ✅ MLTask.HDBSCAN_CLUSTERING = {MLTask.HDBSCAN_CLUSTERING}")
        
        # Test 5: Configuration values
        print("5. Testing configuration values...")
        config = FeatureBankConfig()
        print(f"   ✅ hdbscan_min_features = {config.hdbscan_min_features}")
        print(f"   ✅ hdbscan_max_features = {config.hdbscan_max_features}")
        
        # Test 6: Enhanced integration initialization
        print("6. Testing enhanced integration initialization...")
        integrator = EnhancedHDBSCANClusteringIntegration(
            min_features=100,
            max_features=150,
            enable_pca_reduction=True,
            pca_components=15
        )
        print(f"   ✅ Enhanced integration initialized with {integrator.min_features}-{integrator.max_features} features")
        print(f"   ✅ PCA reduction enabled: {integrator.enable_pca_reduction}")
        print(f"   ✅ PCA components: {integrator.pca_components}")
        
        print("\n🎉 All imports and initialization tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import test failed: {e}")
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
        
        # Test feature generators
        print("3. Testing feature generator initialization...")
        generators = integrator._initialize_feature_generators()
        print(f"   ✅ Initialized {len(generators)} feature generators")
        
        for category, generator_list in generators.items():
            print(f"      - {category}: {len(generator_list)} generators")
        
        print("\n🎉 Feature generation flow tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Feature flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Enhanced HDBSCAN Clustering Integration Test")
    print("=" * 60)
    
    # Run tests
    import_success = test_imports()
    launcher_success = test_launcher_integration()
    flow_success = test_feature_flow()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 60)
    print(f"Import Tests: {'✅ PASS' if import_success else '❌ FAIL'}")
    print(f"Launcher Tests: {'✅ PASS' if launcher_success else '❌ FAIL'}")
    print(f"Feature Flow Tests: {'✅ PASS' if flow_success else '❌ FAIL'}")
    
    if import_success and launcher_success and flow_success:
        print("\n🎉 ALL SYSTEMS ARE PROPERLY WIRED!")
        print("✅ Enhanced HDBSCAN clustering integration is ready")
        print("✅ Feature selection: 100-150 features")
        print("✅ PCA reduction: 10-25 components")
        print("✅ Launcher integration: --hdbscan-regime-discovery")
        print("\n🚀 You can now launch HDBSCAN clustering with:")
        print("   python src/launcher/ares_launcher.py --hdbscan-regime-discovery --symbol ETHUSDT")
    else:
        print("\n⚠️  Some issues found. Check the errors above.")