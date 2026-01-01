#!/usr/bin/env python3
"""
Test script for optimized orthogonalization implementation
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

def test_optimized_orthogonalization():
    """Test the complete optimized orthogonalization pipeline"""
    
    try:
        from src.utils.ml_common.specialist_orthogonalizer import OptimizedSpecialistOrthogonalizer
        print("✅ Import successful")
        
        # Test initialization with all optimizations
        orthogonalizer = OptimizedSpecialistOrthogonalizer(
            enable_cache=True,
            enable_feature_optimization=True,
            enable_orthogonal_hpo=False,  # Skip HPO for quick test
            enable_conservative_pruning=False
        )
        print("✅ Initialization with optimizations successful")
        
        # Create sample data
        sample_data = pd.DataFrame({
            # XGB Macro features
            'macro_trend_1': np.random.randn(100),
            'xgb_macro_signal': np.random.randn(100),
            'regime_macro_prob': np.random.random(100),
            
            # Risk features
            'risk_score': np.random.random(100),
            'risk_regime_0_prob': np.random.random(100),
            'risk_pred_1': np.random.random(100),
            
            # Liquidity features
            'liquidity_regime_1_prob': np.random.random(100),
            'liquidity_score': np.random.random(100),
            
            # Momentum features
            'momentum_persistence_5': np.random.random(100),
            'momentum_signal': np.random.random(100),
            
            # Volume features
            'vol_force_breakout': np.random.random(100),
            'vol_force_magnitude': np.random.random(100),
        })
        
        target_series = pd.Series(np.random.randint(0, 2, 100), index=sample_data.index)
        sample_weights = pd.Series(np.random.random(100), index=sample_data.index)
        
        print("✅ Sample data created")
        
        # Test feature optimization
        if orthogonalizer.enable_feature_optimization and orthogonalizer.feature_optimizer is not None:
            print("🔧 Testing feature optimization...")
            optimized_categories, feature_analysis = orthogonalizer.feature_optimizer.optimize_feature_pipeline(
                orthogonalizer.specialist_categories, sample_data
            )
            print(f"  Feature analysis: {feature_analysis.total_features} total, {feature_analysis.unique_features} unique")
        else:
            print("⚠️ Feature optimization not available")
        
        # Test optimized orthogonalization (without HPO for speed)
        print("🚀 Testing optimized orthogonalization...")
        results = orthogonalizer.run_optimized_orthogonalization(
            specialist_df=sample_data,
            target_series=target_series,
            sample_weights=sample_weights,
            run_hpo=False,
            run_pruning=False,
            optimize_features=orthogonalizer.feature_optimizer is not None
        )
        
        print("✅ Optimized orthogonalization successful")
        print(f"  Optimization time: {results['optimization_time']:.2f}s")
        print(f"  Pruned specialists: {len(results['pruned_specialists'])}")
        
        # Test performance summary
        perf_summary = results.get('performance_summary', {})
        if perf_summary and 'n_specialists' in perf_summary:
            print(f"  Performance: {perf_summary['n_specialists']} specialists, "
                  f"mean AUC: {perf_summary.get('mean_auc', 0.5):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hpo_components():
    """Test HPO components individually"""
    
    try:
        from src.utils.ml_common.optimization.orthogonalization_aware_hpo import OrthogonalizationAwareHPO, HPOConfig
        from src.utils.ml_common.optimization.conservative_ensemble_pruner import ConservativeEnsemblePruner, PruningConfig
        
        print("✅ HPO components import successful")
        
        # Test HPO config
        hpo_config = HPOConfig(n_trials=5, timeout=60)  # Quick test
        hpo = OrthogonalizationAwareHPO(hpo_config)
        print("✅ Orthogonalization-aware HPO initialized")
        
        # Test pruner config
        pruning_config = PruningConfig(min_ensemble_size=3, max_ensemble_size=5)
        pruner = ConservativeEnsemblePruner(pruning_config)
        print("✅ Conservative pruner initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ HPO component test failed: {e}")
        return False

def test_cache_component():
    """Test specialist cache component"""
    
    try:
        from src.utils.ml_common.specialist_cache import SpecialistModelCache, CacheMetadata
        from datetime import datetime
        
        print("✅ Cache component import successful")
        
        # Test cache initialization
        cache = SpecialistModelCache(max_memory_gb=0.1, max_disk_gb=1.0)
        print("✅ Specialist cache initialized")
        
        # Test metadata
        metadata = CacheMetadata(
            specialist_name="test_specialist",
            model_type="LGBM",
            timestamp=datetime.now().isoformat(),
            data_hash="test_hash",
            config_hash="config_hash",
            performance_metrics={"auc": 0.75},
            training_time=10.5,
            model_size_mb=2.3
        )
        print("✅ Cache metadata created")
        
        return True
        
    except Exception as e:
        print(f"❌ Cache component test failed: {e}")
        return False

def main():
    """Run all tests"""
    
    print("🧪 Testing Optimized Orthogonalization Implementation")
    print("=" * 60)
    
    tests = [
        ("Basic Optimized Orthogonalization", test_optimized_orthogonalization),
        ("HPO Components", test_hpo_components),
        ("Cache Component", test_cache_component),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Optimized orthogonalization implementation is ready!")
        return 0
    else:
        print(f"❌ {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
